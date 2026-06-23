"""
Management Quality PIT Backtest
================================
Survivorship-bias-free, point-in-time management quality factor backtest.
Data: Tiingo (prices) + SEC XBRL (fundamentals) + SEC Form 4 (insider transactions).

Sections:
  1. Config & imports
  2. Tiingo helpers
  3. SEC helpers
  4. Form 4 insider signals
  5. Universe builder
  6. Signal computation
  7. Return calculation
  8. Portfolio construction
  9. Backtest engine
  10. Metrics & output
"""

# =============================================================================
# SECTION 1: Config & imports
# =============================================================================

import concurrent.futures
import io
import json
import os
import re
import socket
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

socket.setdefaulttimeout(25)  # hard OS-level kill for hung TCP connections

import numpy as np
import pandas as pd
import requests
from xml.etree import ElementTree as ET

TIINGO_KEY = os.environ.get("TIINGO_API_KEY")
SEC_UA = "Hugh Archer hugharcher5@gmail.com"

CACHE = Path(__file__).parent / "cache"
RESULTS = Path(__file__).parent / "results"

# Ensure dirs exist
for _d in [
    CACHE / "tiingo",
    CACHE / "sec_facts",
    CACHE / "sec_subs",
    CACHE / "form4",
    RESULTS,
]:
    _d.mkdir(parents=True, exist_ok=True)

REBALANCE_DATES = pd.date_range("2015-01-01", "2025-10-01", freq="QS")
IN_SAMPLE_END = pd.Timestamp("2022-01-01")
OUT_SAMPLE_END = pd.Timestamp("2025-10-01")

MCAP_MIN, MCAP_MAX = 100e6, 2e9
DOLLAR_VOL_MIN = 1e6  # $1M/day average
SIC_EXCLUDE = set(range(6000, 6800))  # financials + real estate

US_EXCHANGES = {"NYSE", "NASDAQ", "AMEX"}

MAX_FETCH_PER_RUN = 3000  # circuit breaker handles rate limits; keep cap above universe size

_TIINGO_LAST_CALL: float = 0.0
_SEC_LAST_CALL: float = 0.0


def _tiingo_get(url: str, params: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    """Rate-limited GET for Tiingo (5s between calls → ~720/hr, safe for free tier)."""
    global _TIINGO_LAST_CALL
    wait = 5.0 - (time.time() - _TIINGO_LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, params=params, timeout=(10, 20))
    _TIINGO_LAST_CALL = time.time()
    return resp


def _sec_get(url: str, timeout: int = 30) -> requests.Response:
    """Rate-limited GET for SEC (5 req/sec → 0.2s delay)."""
    global _SEC_LAST_CALL
    wait = 0.2 - (time.time() - _SEC_LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=(10, 30))
    _SEC_LAST_CALL = time.time()
    return resp


# =============================================================================
# SECTION 2: Tiingo helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    """
    Download supported_tickers.zip once (cached to cache/tickers.csv).
    Return DataFrame filtered to US exchanges (NYSE, NASDAQ, AMEX) and assetType==Stock.
    Columns: ticker, exchange, assetType, priceCurrency, startDate, endDate
    """
    cache_path = CACHE / "tickers.csv"

    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df)} tickers from cache.")
        return df

    print("[Tiingo] Downloading supported_tickers.zip...")
    url = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as f:
            df_all = pd.read_csv(f)

    df_all.columns = [c.strip() for c in df_all.columns]

    # Filter US exchanges, stocks only
    df = df_all[
        df_all["exchange"].isin(US_EXCHANGES) &
        (df_all["assetType"] == "Stock")
    ].copy()

    # Parse dates — handle NaT gracefully
    for col in ["startDate", "endDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["startDate"])
    df = df.reset_index(drop=True)

    df.to_csv(cache_path, index=False)
    print(f"[Tiingo] {len(df)} US-exchange stock tickers saved to cache.")
    return df


# =============================================================================
# SEC TICKER→CIK MAP + PRE-FILTER (run before any Tiingo pulls)
# =============================================================================

_SEC_TICKER_CIK_CACHE = CACHE / "sec_ticker_cik.json"
_SEC_PREFILTER_CACHE = CACHE / "sec_prefilter.csv"

# ETF/fund keyword exclusion list
_FUND_KEYWORDS = [
    "etf", "fund", "trust", "index", "portfolio", "series",
    "income fund", "growth fund", "select fund", "spdr", "ishares",
    "invesco", "vanguard etf", "powershares",
]


def load_sec_ticker_cik_map() -> dict:
    """
    Download SEC's company_tickers.json (one HTTP call, ~2MB).
    Returns dict of uppercase ticker → CIK (int).
    Caches to cache/sec_ticker_cik.json.
    Also pre-populates the existing cik_map.json cache so lookup_cik() hits cache.
    """
    if _SEC_TICKER_CIK_CACHE.exists():
        with open(_SEC_TICKER_CIK_CACHE) as f:
            raw = json.load(f)
        # raw is {"AAPL": 320193, ...}
        print(f"[SEC] Loaded {len(raw)} ticker→CIK pairs from cache.")
        return raw

    print("[SEC] Downloading company_tickers.json...")
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = _sec_get(url)
    resp.raise_for_status()
    data = resp.json()

    # data = {0: {"cik_str": "0000320193", "ticker": "AAPL", "title": "Apple Inc."}, ...}
    ticker_cik: dict = {}
    for entry in data.values():
        t = entry.get("ticker", "").upper().strip()
        cik_raw = entry.get("cik_str", "")
        if t and cik_raw:
            try:
                ticker_cik[t] = int(cik_raw)
            except ValueError:
                pass

    with open(_SEC_TICKER_CIK_CACHE, "w") as f:
        json.dump(ticker_cik, f)
    print(f"[SEC] {len(ticker_cik)} ticker→CIK pairs saved.")

    # Pre-populate the existing cik_map.json so lookup_cik() hits cache
    existing = _load_cik_map()
    merged = {**ticker_cik, **existing}  # existing entries (EFTS-resolved) take priority
    _save_cik_map(merged)

    return ticker_cik


def _has_filings_in_window(submissions: dict) -> bool:
    """
    Return True if this company filed a 10-K or 10-Q between 2013-01-01 and 2026-01-01.
    Checks both filings.recent and filings.files date ranges (no extra API calls).
    """
    target_forms = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
    window_start = "2013-01-01"
    window_end = "2026-01-01"

    try:
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filedDate", recent.get("filingDate", []))
        for i, frm in enumerate(forms):
            if frm in target_forms:
                d = dates[i] if i < len(dates) else ""
                if window_start <= d <= window_end:
                    return True

        # Check older filing batches via their date range metadata (no fetch needed)
        for ffile in submissions.get("filings", {}).get("files", []):
            filed_from = ffile.get("filedFrom", "")
            filed_to = ffile.get("filedTo", "")
            # Overlaps our window?
            if filed_to >= window_start and filed_from <= window_end:
                return True  # company was filing somewhere in range; accept
    except Exception:
        pass
    return False


def _is_fund_or_etf(name: str, entity_type: str) -> bool:
    """Exclude ETFs, mutual funds, investment trusts by name/entity type heuristics."""
    name_lower = name.lower() if name else ""
    etype_lower = entity_type.lower() if entity_type else ""
    if any(kw in name_lower for kw in _FUND_KEYWORDS):
        return True
    # SEC entity types for funds
    if "investment company" in etype_lower or "registered investment" in etype_lower:
        return True
    return False


def build_sec_prefilter(ticker_cik_map: dict) -> pd.DataFrame:
    """
    For each CIK in ticker_cik_map, fetch submissions (cached), and keep only:
      - Non-financial SIC (exclude 6000–6799)
      - Not an ETF/fund
      - Has 10-K/10-Q filings in the 2013–2026 window

    CRITICAL: No market-cap filter here. That is applied point-in-time later.

    Returns DataFrame with columns [ticker, cik, sic, name, passed].
    Caches to cache/sec_prefilter.csv so subsequent runs are instant.
    """
    if _SEC_PREFILTER_CACHE.exists():
        df = pd.read_csv(_SEC_PREFILTER_CACHE)
        survivors = df[df["passed"]]["ticker"].tolist()
        print(f"[PreFilter] Loaded from cache: {len(df)} checked, {len(survivors)} survivors.")
        return df

    print(f"[PreFilter] Screening {len(ticker_cik_map)} tickers via SEC submissions...")
    rows = []
    total = len(ticker_cik_map)

    for i, (ticker, cik) in enumerate(ticker_cik_map.items()):
        if (i + 1) % 500 == 0:
            passed_so_far = sum(1 for r in rows if r["passed"])
            print(f"  [PreFilter] {i+1}/{total}  passed={passed_so_far}  ticker={ticker}")

        subs = fetch_submissions(cik)
        if subs is None:
            rows.append({"ticker": ticker, "cik": cik, "sic": None, "name": None, "passed": False, "reason": "no_submissions"})
            continue

        name = subs.get("name", "")
        entity_type = subs.get("entityType", "")

        # SIC filter
        sic_raw = subs.get("sic")
        sic = int(sic_raw) if sic_raw else None
        if sic is not None and sic in SIC_EXCLUDE:
            rows.append({"ticker": ticker, "cik": cik, "sic": sic, "name": name, "passed": False, "reason": "financial_sic"})
            continue

        # Fund/ETF filter
        if _is_fund_or_etf(name, entity_type):
            rows.append({"ticker": ticker, "cik": cik, "sic": sic, "name": name, "passed": False, "reason": "fund_etf"})
            continue

        # Fundamentals-in-window filter
        if not _has_filings_in_window(subs):
            rows.append({"ticker": ticker, "cik": cik, "sic": sic, "name": name, "passed": False, "reason": "no_filings_in_window"})
            continue

        rows.append({"ticker": ticker, "cik": cik, "sic": sic, "name": name, "passed": True, "reason": "ok"})

    df = pd.DataFrame(rows)
    df.to_csv(_SEC_PREFILTER_CACHE, index=False)

    survivors = df[df["passed"]]
    print(f"\n[PreFilter] Results:")
    print(f"  Total screened:   {len(df)}")
    for reason, grp in df.groupby("reason"):
        print(f"  {reason:<30} {len(grp):>6}")
    print(f"  SURVIVORS:        {len(survivors)}")

    return df


def prefetch_tiingo_survivors(survivor_tickers: list) -> None:
    """
    Pull Tiingo EOD price history for each surviving ticker.
    Skips any already cached. Prints estimated time and progress.
    """
    already_cached = {
        p.stem for p in (CACHE / "tiingo").glob("*.csv")
    }
    sentinel_cached = {
        p.stem for p in (CACHE / "tiingo").glob("*.EMPTY")
    }

    to_fetch = [
        t for t in survivor_tickers
        if t not in already_cached and t not in sentinel_cached
    ]
    cached_count = len(survivor_tickers) - len(to_fetch)

    deferred = 0
    if len(to_fetch) > MAX_FETCH_PER_RUN:
        deferred = len(to_fetch) - MAX_FETCH_PER_RUN
        to_fetch = to_fetch[:MAX_FETCH_PER_RUN]

    est_seconds = len(to_fetch) * 5.1  # 5s throttle + small overhead
    est_minutes = est_seconds / 60

    print(f"\n[Tiingo Prefetch]")
    print(f"  Survivors:        {len(survivor_tickers)}")
    print(f"  Already cached:   {cached_count}")
    print(f"  To fetch:         {len(to_fetch)}")
    if deferred:
        print(f"  Deferred:         {deferred} (capped at {MAX_FETCH_PER_RUN}/run)")
    print(f"  Estimated time:   {est_minutes:.0f} min ({est_seconds/3600:.1f} hrs)")

    if not to_fetch:
        print("  [Tiingo Prefetch] All already cached — skipping.")
        return

    failed_log = CACHE / "failed_tickers.txt"
    consecutive_fails = 0
    MAX_CONSECUTIVE_FAILS = 5

    for i, ticker in enumerate(to_fetch):
        if (i + 1) % 50 == 0:
            remaining_est = (len(to_fetch) - i - 1) * 5.1
            print(
                f"  [Tiingo] {i+1}/{len(to_fetch)} | "
                f"~{remaining_est/60:.0f} min remaining | "
                f"last={ticker}"
            )

        # One executor per ticker so a timed-out thread doesn't block the next submit.
        # shutdown(wait=False) abandons hung threads rather than waiting for them.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            fetch_tiingo_prices,
            ticker,
            pd.Timestamp("2000-01-01"),
            pd.Timestamp.now(),
        )
        got_data = False
        try:
            future.result(timeout=45)
            cache_file = CACHE / "tiingo" / f"{ticker}.csv"
            sentinel = CACHE / "tiingo" / f"{ticker}.EMPTY"
            got_data = cache_file.exists() or sentinel.exists()
        except concurrent.futures.TimeoutError:
            print(f"  [Tiingo] TIMEOUT on {ticker} — skipping.")
            with open(failed_log, "a") as fh:
                fh.write(f"{ticker}\tTIMEOUT\n")
        except Exception as e:
            print(f"  [Tiingo] ERROR on {ticker}: {type(e).__name__}: {e}")
            with open(failed_log, "a") as fh:
                fh.write(f"{ticker}\t{type(e).__name__}\n")
        finally:
            executor.shutdown(wait=False)

        if got_data:
            consecutive_fails = 0
        else:
            consecutive_fails += 1
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                print(f"\n  [Tiingo Prefetch] STOPPING: {MAX_CONSECUTIVE_FAILS} consecutive "
                      f"failures — API likely rate-limited or IP blocked.")
                print(f"  Fetched {i + 1}/{len(to_fetch)} this session. "
                      f"Re-run later to continue.")
                return


def fetch_tiingo_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch Tiingo daily prices for ticker in [start, end].
    Checks cache/tiingo/{ticker}.csv first.
    Uses `close` field (NOT adjClose).
    Returns DataFrame with columns [date, open, high, low, close, volume].
    Empty DataFrame if unavailable.
    """
    cache_file = CACHE / "tiingo" / f"{ticker}.csv"
    sentinel = CACHE / "tiingo" / f"{ticker}.EMPTY"

    # If we already know it's a dead end, return empty
    if sentinel.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    now = pd.Timestamp.now()
    needs_fetch = True

    if cache_file.exists():
        mtime = pd.Timestamp(datetime.fromtimestamp(cache_file.stat().st_mtime))
        age_days = (now - mtime).days
        # Use cache if fresh (≤30 days) or if the file has data
        cached_df = pd.read_csv(cache_file, parse_dates=["date"])
        if len(cached_df) == 0:
            needs_fetch = age_days > 30
        else:
            # Check if cache covers the requested range
            max_cached = cached_df["date"].max()
            # If the end of the request is within the cached range, no need to re-fetch
            if end <= max_cached or age_days <= 30:
                needs_fetch = False

        if not needs_fetch:
            mask = (cached_df["date"] >= start) & (cached_df["date"] <= end)
            return cached_df[mask].copy().reset_index(drop=True)

    # Fetch from API — always pull full history to maximise cache utility
    if not TIINGO_KEY:
        raise EnvironmentError("TIINGO_API_KEY not set.")

    fetch_start = "2000-01-01"
    fetch_end = now.strftime("%Y-%m-%d")
    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": fetch_start,
        "endDate": fetch_end,
        "token": TIINGO_KEY,
        "columns": "date,open,high,low,close,volume,splitFactor,divCash",
    }

    _EMPTY = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

    # Single retry on 429/network error; anything else is treated as transient.
    # The outer ThreadPoolExecutor in prefetch_tiingo_survivors enforces a hard
    # per-ticker wall-clock limit, so we keep retries short here.
    for attempt in range(2):
        try:
            resp = _tiingo_get(url, params=params)
        except Exception as e:
            print(f"[Tiingo] Network error for {ticker}: {type(e).__name__}")
            return _EMPTY  # caller's ThreadPoolExecutor will log and move on

        if resp.status_code == 429:
            if attempt == 0:
                print(f"[Tiingo] 429 on {ticker} — sleeping 60s then retrying once...")
                time.sleep(60)
                continue
            # Still 429 after one retry — give up, caller logs to failed_tickers.txt
            print(f"[Tiingo] 429 on {ticker} persists — skipping.")
            return _EMPTY

        if resp.status_code in (404, 400):
            sentinel.touch()
            return _EMPTY

        if resp.status_code != 200:
            print(f"[Tiingo] HTTP {resp.status_code} for {ticker}")
            return _EMPTY

        try:
            data = resp.json()
        except Exception:
            return _EMPTY

        if not data:
            sentinel.touch()
            return _EMPTY

        break  # success
    else:
        return _EMPTY

    df = pd.DataFrame(data)

    # Normalise columns — Tiingo returns camelCase
    col_map = {}
    for c in df.columns:
        col_map[c] = c.lower()
    df = df.rename(columns=col_map)

    # Date normalisation
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()

    needed = [c for c in ["date", "open", "high", "low", "close", "volume", "splitfactor", "divcash"] if c in df.columns]
    df = df[needed].drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Persist full history to cache
    df.to_csv(cache_file, index=False)

    mask = (df["date"] >= start) & (df["date"] <= end)
    return df[mask].copy().reset_index(drop=True)


def get_last_close_on_or_before(ticker: str, as_of: pd.Timestamp) -> Optional[float]:
    """Return last available close price on or before as_of date."""
    df = fetch_tiingo_prices(ticker, pd.Timestamp("2000-01-01"), as_of)
    if df.empty:
        return None
    return float(df.iloc[-1]["close"])


def get_avg_dollar_volume(ticker: str, as_of: pd.Timestamp, lookback_days: int = 60) -> float:
    """Average daily dollar volume over the last lookback_days trading days before as_of."""
    start = as_of - pd.Timedelta(days=lookback_days * 2)  # extra buffer for non-trading days
    df = fetch_tiingo_prices(ticker, start, as_of)
    if df.empty:
        return 0.0
    df = df[df["date"] <= as_of].tail(lookback_days)
    if df.empty:
        return 0.0
    dvol = (df["close"] * df["volume"]).mean()
    return float(dvol)


# =============================================================================
# SECTION 3: SEC helpers
# =============================================================================

_CIK_MAP_PATH = CACHE / "cik_map.json"


def _load_cik_map() -> dict:
    if _CIK_MAP_PATH.exists():
        with open(_CIK_MAP_PATH) as f:
            return json.load(f)
    return {}


def _save_cik_map(cik_map: dict) -> None:
    with open(_CIK_MAP_PATH, "w") as f:
        json.dump(cik_map, f)


def lookup_cik(ticker: str) -> Optional[int]:
    """
    Look up SEC CIK for ticker.
    Strategy 1: EFTS full-text search (fast, sometimes 500).
    Strategy 2: EDGAR company search (slower, more reliable fallback).
    Cached in cache/cik_map.json.
    Returns int or None.
    """
    cik_map = _load_cik_map()

    if ticker in cik_map:
        return cik_map[ticker]

    cik = None

    # Strategy 1: EFTS search (URL-encode the ticker)
    from urllib.parse import quote
    ticker_enc = quote(f'"{ticker}"')
    url = (
        f"https://efts.sec.gov/LATEST/search-index?q={ticker_enc}"
        f"&forms=10-K&dateRange=custom&startdt=2000-01-01&enddt=2026-01-01"
    )
    try:
        resp = _sec_get(url)
        if resp.status_code == 200:
            body = resp.text
            matches = re.findall(r"CIK (\d+)", body)
            if matches:
                cik = int(matches[0])
    except Exception as e:
        pass  # fall through to strategy 2

    # Strategy 2: EDGAR company search (reliable fallback)
    if cik is None:
        try:
            url2 = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&output=atom"
            resp2 = _sec_get(url2)
            if resp2.status_code == 200:
                # Extract CIK from atom XML: <company-info><cik>0000945436</cik>
                m = re.search(r"<cik>(\d+)</cik>", resp2.text)
                if m:
                    cik = int(m.group(1))
        except Exception:
            pass

    cik_map[ticker] = cik
    _save_cik_map(cik_map)
    return cik


def _cik_str(cik: int) -> str:
    return str(cik).zfill(10)


def fetch_company_facts(cik: int) -> Optional[dict]:
    """
    Fetch SEC XBRL company facts.
    Cached at cache/sec_facts/{cik}.json.
    Returns parsed dict or None.
    """
    cache_file = CACHE / "sec_facts" / f"{cik}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{_cik_str(cik)}.json"
    try:
        resp = _sec_get(url)
    except Exception as e:
        print(f"[SEC] company_facts network error for CIK {cik}: {e}")
        return None

    if resp.status_code == 404:
        return None

    if resp.status_code != 200:
        print(f"[SEC] company_facts HTTP {resp.status_code} for CIK {cik}")
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data


def fetch_submissions(cik: int) -> Optional[dict]:
    """
    Fetch SEC submissions (includes all filings, form 4 data).
    Cached at cache/sec_subs/{cik}.json.
    Returns parsed dict or None.
    """
    cache_file = CACHE / "sec_subs" / f"{cik}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"https://data.sec.gov/submissions/CIK{_cik_str(cik)}.json"
    try:
        resp = _sec_get(url)
    except Exception as e:
        print(f"[SEC] submissions network error for CIK {cik}: {e}")
        return None

    if resp.status_code == 404:
        return None

    if resp.status_code != 200:
        print(f"[SEC] submissions HTTP {resp.status_code} for CIK {cik}")
        return None

    try:
        data = resp.json()
    except Exception:
        return None

    with open(cache_file, "w") as f:
        json.dump(data, f)

    return data


def get_gaap_value_pit(
    facts: dict,
    concept: str,
    as_of_date: pd.Timestamp,
    namespace: str = "us-gaap",
) -> tuple:
    """
    Return the most recent value for a GAAP concept where filed_date ≤ as_of_date.
    Handles both 'instant' (balance sheet) and 'duration' (income) forms.
    Returns (value, period_end_date, filed_date) or (None, None, None).

    For duration concepts, prefer the most recent annual (12M) period.
    For instant concepts, take the most recent filing.
    """
    try:
        units_dict = facts["facts"][namespace][concept]["units"]
    except (KeyError, TypeError):
        return (None, None, None)

    # Collect all data points (across unit types, usually USD or shares)
    records = []
    for unit_label, entries in units_dict.items():
        for entry in entries:
            filed = entry.get("filed")
            end = entry.get("end")
            val = entry.get("val")
            form = entry.get("form", "")
            if filed is None or end is None or val is None:
                continue
            try:
                filed_ts = pd.Timestamp(filed)
                end_ts = pd.Timestamp(end)
            except Exception:
                continue
            # PIT filter: only use filings available on or before as_of_date
            if filed_ts > as_of_date:
                continue
            records.append({
                "val": val,
                "end": end_ts,
                "filed": filed_ts,
                "form": form,
                "unit": unit_label,
            })

    if not records:
        return (None, None, None)

    df = pd.DataFrame(records)

    # For duration concepts, try to find annual periods (start/end span ~12M)
    # The XBRL data has 'start' field for duration entries
    has_start = any("start" in e for entries in units_dict.values() for e in entries)

    if has_start:
        # Re-collect with start field
        records2 = []
        for unit_label, entries in units_dict.items():
            for entry in entries:
                filed = entry.get("filed")
                end = entry.get("end")
                start = entry.get("start")
                val = entry.get("val")
                if filed is None or end is None or val is None:
                    continue
                try:
                    filed_ts = pd.Timestamp(filed)
                    end_ts = pd.Timestamp(end)
                    start_ts = pd.Timestamp(start) if start else None
                except Exception:
                    continue
                if filed_ts > as_of_date:
                    continue
                period_days = (end_ts - start_ts).days if start_ts else 0
                records2.append({
                    "val": val,
                    "end": end_ts,
                    "filed": filed_ts,
                    "form": entry.get("form", ""),
                    "unit": unit_label,
                    "period_days": period_days,
                })

        if records2:
            df = pd.DataFrame(records2)
            # Prefer annual (10-K, ~365 days). Allow 300-400 day range.
            annual = df[(df["period_days"] >= 300) & (df["period_days"] <= 400)]
            if not annual.empty:
                df = annual

    # Sort by filed desc, then end desc — take most recent
    df = df.sort_values(["filed", "end"], ascending=False)
    best = df.iloc[0]
    return (float(best["val"]), best["end"], best["filed"])


def get_gaap_value_pit_2yr_ago(
    facts: dict,
    concept: str,
    as_of_date: pd.Timestamp,
    namespace: str = "us-gaap",
) -> tuple:
    """
    Same as get_gaap_value_pit but returns the value whose period_end is approximately
    2 fiscal years (24 months) before the most-recent period_end.
    Returns (value, period_end_date, filed_date) or (None, None, None).
    """
    # First get the most recent value to find the anchor period_end
    val_now, end_now, filed_now = get_gaap_value_pit(facts, concept, as_of_date, namespace)
    if end_now is None:
        return (None, None, None)

    # Target: period end ~2 years before end_now
    target_end = end_now - pd.DateOffset(years=2)
    window_start = target_end - pd.Timedelta(days=120)
    window_end = target_end + pd.Timedelta(days=120)

    try:
        units_dict = facts["facts"][namespace][concept]["units"]
    except (KeyError, TypeError):
        return (None, None, None)

    records = []
    for unit_label, entries in units_dict.items():
        for entry in entries:
            filed = entry.get("filed")
            end = entry.get("end")
            start = entry.get("start")
            val = entry.get("val")
            if filed is None or end is None or val is None:
                continue
            try:
                filed_ts = pd.Timestamp(filed)
                end_ts = pd.Timestamp(end)
                start_ts = pd.Timestamp(start) if start else None
            except Exception:
                continue
            if filed_ts > as_of_date:
                continue
            if not (window_start <= end_ts <= window_end):
                continue
            period_days = (end_ts - start_ts).days if start_ts else 0
            records.append({
                "val": val,
                "end": end_ts,
                "filed": filed_ts,
                "period_days": period_days,
            })

    if not records:
        return (None, None, None)

    df = pd.DataFrame(records)
    # Prefer annual
    annual = df[(df["period_days"] >= 300) & (df["period_days"] <= 400)]
    if not annual.empty:
        df = annual

    df = df.sort_values(["filed", "end"], ascending=False)
    best = df.iloc[0]
    return (float(best["val"]), best["end"], best["filed"])


def get_sic_code(cik: int, submissions: dict) -> Optional[int]:
    """Extract SIC code from submissions JSON."""
    try:
        sic = submissions.get("sic")
        if sic is not None:
            return int(sic)
    except (ValueError, TypeError):
        pass
    return None


# =============================================================================
# SECTION 4: Form 4 insider signals
# =============================================================================

def fetch_form4_filings(cik: int, submissions: dict) -> list:
    """
    Extract Form 4 filings from submissions['filings']['recent'].
    Returns list of (accn, filedDate, primaryDoc) tuples.
    """
    try:
        recent = submissions["filings"]["recent"]
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        filed_dates = recent.get("filedDate", recent.get("filingDate", []))
        primary_docs = recent.get("primaryDocument", [])
    except (KeyError, TypeError):
        return []

    results = []
    for i, form in enumerate(forms):
        if form == "4":
            try:
                accn = accns[i]
                filed = filed_dates[i]
                doc = primary_docs[i] if i < len(primary_docs) else ""
                results.append((accn, filed, doc))
            except IndexError:
                continue

    return results


def parse_form4_xml(cik: int, accn: str, doc: str) -> list:
    """
    Fetch and cache Form 4 XML. Parse nonDerivativeTransaction elements.
    Returns list of dicts with keys: date, shares, price, action (A or D).
    """
    accn_nodash = accn.replace("-", "")
    cache_file = CACHE / "form4" / f"{accn_nodash}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    if not doc:
        return []

    # The primaryDocument field sometimes includes an XSLT viewer prefix like
    # "xslF345X03/doc4.xml". The raw XML lives at the root of the filing, not
    # inside the viewer subdirectory. Strip any such prefix.
    doc_clean = doc.split("/")[-1] if "/" in doc else doc
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{doc_clean}"
    try:
        resp = _sec_get(url)
    except Exception as e:
        print(f"[SEC] Form4 XML network error {accn}: {e}")
        return []

    if resp.status_code != 200:
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return []

    transactions = []
    # Handle both namespaced and plain XML
    ns_map = {"": ""}  # fallback

    def find_text(element, tag):
        """Find text of child element, handling optional namespace."""
        el = element.find(tag)
        if el is None:
            # Try with namespace prefix stripping
            for child in element:
                if child.tag.endswith(tag) or child.tag == tag:
                    return child.text
        return el.text if el is not None else None

    for nd_tx in root.iter():
        if not nd_tx.tag.endswith("nonDerivativeTransaction"):
            continue

        # Find the transactionAmounts sub-element
        tx_amounts = None
        for child in nd_tx:
            if child.tag.endswith("transactionAmounts"):
                tx_amounts = child
                break

        if tx_amounts is None:
            continue

        shares_el = None
        price_el = None
        action_el = None
        date_el = None

        for child in nd_tx:
            if child.tag.endswith("transactionDate"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        date_el = sub
                        break

        for child in tx_amounts:
            tag = child.tag
            if tag.endswith("transactionShares"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        shares_el = sub
                        break
            elif tag.endswith("transactionPricePerShare"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        price_el = sub
                        break
            elif tag.endswith("transactionAcquiredDisposedCode"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        action_el = sub
                        break

        try:
            date_str = date_el.text if date_el is not None else None
            shares = float(shares_el.text) if shares_el is not None else None
            price = float(price_el.text) if price_el is not None else 0.0
            action = action_el.text.strip() if action_el is not None else None

            if date_str and shares is not None and action in ("A", "D"):
                transactions.append({
                    "date": date_str,
                    "shares": shares,
                    "price": price,
                    "action": action,
                })
        except (ValueError, AttributeError):
            continue

    with open(cache_file, "w") as f:
        json.dump(transactions, f)

    return transactions


def compute_m1_insider_buying(
    cik: int,
    submissions: dict,
    as_of_date: pd.Timestamp,
    shares_outstanding: float,
) -> float:
    """
    Aggregate Form 4 transactions in the 6 months before as_of_date.
    Net shares = A shares - D shares.
    Normalized by shares_outstanding.
    Returns float (positive = net buying, negative = net selling).
    Returns 0.0 if data unavailable.
    """
    if shares_outstanding is None or shares_outstanding <= 0:
        return 0.0

    window_start = as_of_date - pd.DateOffset(months=6)
    filings = fetch_form4_filings(cik, submissions)

    net_shares = 0.0
    parsed_any = False

    for accn, filed_date_str, doc in filings:
        try:
            filed_ts = pd.Timestamp(filed_date_str)
        except Exception:
            continue

        if not (window_start <= filed_ts <= as_of_date):
            continue

        txns = parse_form4_xml(cik, accn, doc)
        for txn in txns:
            try:
                txn_date = pd.Timestamp(txn["date"])
            except Exception:
                continue
            if not (window_start <= txn_date <= as_of_date):
                continue
            parsed_any = True
            if txn["action"] == "A":
                net_shares += txn["shares"]
            elif txn["action"] == "D":
                net_shares -= txn["shares"]

    if not parsed_any:
        return 0.0

    return net_shares / shares_outstanding


# =============================================================================
# SECTION 5: Universe builder
# =============================================================================

def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    prefilter_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the investable universe as of rebalance_date.
    prefilter_df: output of build_sec_prefilter() — only survivors are considered.
    Returns DataFrame with columns:
      [ticker, cik, mcap, price, shares, sic, avg_dvol]
    """
    print(f"\n[Universe] Building universe for {rebalance_date.date()}")

    # Pre-filter survivors: build a ticker→(cik, sic) map from prefilter results
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    survivors_set = set(survivors["ticker"].tolist())
    cik_sic_map = {row["ticker"]: (int(row["cik"]), row["sic"]) for _, row in survivors.iterrows()}

    # Step 1: filter Tiingo tickers active on rebalance_date AND in prefilter survivors
    active = tiingo_tickers[
        (tiingo_tickers["startDate"] <= rebalance_date) &
        (
            tiingo_tickers["endDate"].isna() |
            (tiingo_tickers["endDate"] >= rebalance_date)
        ) &
        tiingo_tickers["ticker"].isin(survivors_set)
    ].copy()
    print(f"  [Funnel] Active on date (Tiingo ∩ SEC pre-filter): {len(active)}")

    rows = []
    n_total = len(active)

    for idx, (_, row) in enumerate(active.iterrows()):
        ticker = row["ticker"]

        if (idx + 1) % 200 == 0:
            print(f"  [Funnel] Processed {idx+1}/{n_total} tickers...")

        # CIK and SIC already known from prefilter — no API call needed
        cik, sic = cik_sic_map.get(ticker, (None, None))
        if cik is None:
            continue

        # Step 3: fetch company facts (XBRL) — usually already cached
        facts = fetch_company_facts(cik)
        if facts is None:
            continue

        # Step 4: fetch submissions (for Form 4) — usually already cached
        subs = fetch_submissions(cik)
        if subs is None:
            continue

        # SIC already pre-filtered; no repeat check needed here.

        # Step 6: PIT shares outstanding
        shares_val, shares_end, shares_filed = get_gaap_value_pit(
            facts, "CommonStockSharesOutstanding", rebalance_date
        )
        if shares_val is None or shares_val <= 0:
            # Try EntityCommonStockSharesOutstanding as fallback
            shares_val, shares_end, shares_filed = get_gaap_value_pit(
                facts, "EntityCommonStockSharesOutstanding", rebalance_date
            )
        if shares_val is None or shares_val <= 0:
            continue

        # Step 7: PIT price
        price = get_last_close_on_or_before(ticker, rebalance_date)
        if price is None or price <= 0:
            continue

        # Step 8: market cap
        mcap = shares_val * price
        if not (MCAP_MIN <= mcap <= MCAP_MAX):
            continue

        # Step 9: average dollar volume
        avg_dvol = get_avg_dollar_volume(ticker, rebalance_date, lookback_days=60)
        if avg_dvol < DOLLAR_VOL_MIN:
            continue

        rows.append({
            "ticker": ticker,
            "cik": cik,
            "mcap": mcap,
            "price": price,
            "shares": shares_val,
            "sic": sic,
            "avg_dvol": avg_dvol,
        })

    df = pd.DataFrame(rows)
    print(f"  [Funnel] Final universe: {len(df)} names")
    return df


# =============================================================================
# SECTION 6: Signal computation
# =============================================================================

def _rank_normalize(series: pd.Series) -> pd.Series:
    """Rank-normalize a series to [0, 1]. NaN stays NaN."""
    filled = series.fillna(series.median())
    ranks = filled.rank(method="average")
    return (ranks - 1) / max(len(ranks) - 1, 1)


def compute_signals(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Compute M1, M2, M3 signals for each company in universe_df.
    Adds columns: M1, M2, M3, M1_rank, M2_rank, M3_rank, m_score.
    """
    df = universe_df.copy()

    m1_vals = []
    m2_vals = []
    m3_vals = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        cik = int(row["cik"])
        shares_now = row["shares"]

        # --- M1: net insider buying (normalized by shares outstanding) ---
        try:
            subs = fetch_submissions(cik)
            m1 = compute_m1_insider_buying(cik, subs, rebalance_date, shares_now) if subs else 0.0
        except Exception as e:
            print(f"  [M1] Error for {ticker}: {e}")
            m1 = 0.0
        m1_vals.append(m1)

        # --- M2: no dilution signal ---
        try:
            facts = fetch_company_facts(cik)
            shares_2yr_val, _, _ = get_gaap_value_pit_2yr_ago(
                facts, "CommonStockSharesOutstanding", rebalance_date
            )
            if shares_2yr_val is None or shares_2yr_val <= 0:
                shares_2yr_val, _, _ = get_gaap_value_pit_2yr_ago(
                    facts, "EntityCommonStockSharesOutstanding", rebalance_date
                )
            if shares_2yr_val is not None and shares_2yr_val > 0 and shares_now > 0:
                m2 = (shares_2yr_val - shares_now) / shares_2yr_val
            else:
                m2 = 0.0
        except Exception as e:
            print(f"  [M2] Error for {ticker}: {e}")
            m2 = 0.0
        m2_vals.append(m2)

        # --- M3: ROIC improving ---
        try:
            facts = fetch_company_facts(cik)

            def _get_roic(facts, as_of):
                oi_val, _, _ = get_gaap_value_pit(facts, "OperatingIncomeLoss", as_of)
                tax_val, _, _ = get_gaap_value_pit(facts, "IncomeTaxExpenseBenefit", as_of)
                pretax_val, _, _ = get_gaap_value_pit(facts, "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest", as_of)
                assets_val, _, _ = get_gaap_value_pit(facts, "Assets", as_of)
                curr_liab_val, _, _ = get_gaap_value_pit(facts, "LiabilitiesCurrent", as_of)

                if any(v is None for v in [oi_val, assets_val, curr_liab_val]):
                    return None

                # Effective tax rate
                if pretax_val is not None and pretax_val != 0 and tax_val is not None:
                    etr = tax_val / pretax_val
                    etr = max(0.0, min(0.5, etr))
                else:
                    etr = 0.25  # default

                nopat = oi_val * (1 - etr)
                ic = assets_val - curr_liab_val

                if ic <= 0:
                    return None

                return nopat / ic

            roic_t1 = _get_roic(facts, rebalance_date)

            # t0 = 1 year prior
            one_yr_ago = rebalance_date - pd.DateOffset(years=1)
            roic_t0 = _get_roic(facts, one_yr_ago)

            if roic_t1 is not None and roic_t0 is not None:
                m3 = roic_t1 - roic_t0
            else:
                m3 = 0.0

        except Exception as e:
            print(f"  [M3] Error for {ticker}: {e}")
            m3 = 0.0
        m3_vals.append(m3)

    df["M1"] = m1_vals
    df["M2"] = m2_vals
    df["M3"] = m3_vals

    # Rank-normalize each signal to [0, 1]
    df["M1_rank"] = _rank_normalize(df["M1"])
    df["M2_rank"] = _rank_normalize(df["M2"])
    df["M3_rank"] = _rank_normalize(df["M3"])

    # m_score = equal-weight average of ranks
    df["m_score"] = (df["M1_rank"] + df["M2_rank"] + df["M3_rank"]) / 3.0

    # Also compute earnings yield for benchmark
    ey_vals = []
    for _, row in df.iterrows():
        cik = int(row["cik"])
        mcap = row["mcap"]
        try:
            facts = fetch_company_facts(cik)
            oi_val, _, _ = get_gaap_value_pit(facts, "OperatingIncomeLoss", rebalance_date)
            if oi_val is not None and mcap > 0:
                ey_vals.append(oi_val / mcap)
            else:
                ey_vals.append(None)
        except Exception:
            ey_vals.append(None)

    df["earnings_yield"] = ey_vals
    df["earnings_yield"] = df["earnings_yield"].fillna(df["earnings_yield"].median())

    print(f"  [Signals] M1 range: [{df['M1'].min():.4f}, {df['M1'].max():.4f}]")
    print(f"  [Signals] M2 range: [{df['M2'].min():.4f}, {df['M2'].max():.4f}]")
    print(f"  [Signals] M3 range: [{df['M3'].min():.4f}, {df['M3'].max():.4f}]")
    print(f"  [Signals] m_score range: [{df['m_score'].min():.4f}, {df['m_score'].max():.4f}]")

    return df


# =============================================================================
# SECTION 7: Return calculation
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute holding period returns for each ticker in universe_df.
    Handles delistings using last available close as terminal value.
    Adds columns: entry_price, exit_price, raw_return, delisted, delisting_date.
    """
    df = universe_df.copy()

    # Build a map of ticker → endDate for delistings
    delisted_map = {}
    for _, row in tiingo_tickers.iterrows():
        end = row["endDate"]
        if pd.notna(end) and end < hold_end:
            delisted_map[row["ticker"]] = end

    entry_prices = []
    exit_prices = []
    raw_returns = []
    delisted_flags = []
    delisting_dates = []

    for _, row in df.iterrows():
        ticker = row["ticker"]

        # Entry price: close on hold_start or first available after
        prices_full = fetch_tiingo_prices(ticker, hold_start, hold_end + pd.Timedelta(days=5))

        if prices_full.empty:
            entry_prices.append(np.nan)
            exit_prices.append(np.nan)
            raw_returns.append(np.nan)
            delisted_flags.append(False)
            delisting_dates.append(pd.NaT)
            continue

        # Entry: first available close on or after hold_start
        entry_candidates = prices_full[prices_full["date"] >= hold_start]
        if entry_candidates.empty:
            entry_prices.append(np.nan)
            exit_prices.append(np.nan)
            raw_returns.append(np.nan)
            delisted_flags.append(False)
            delisting_dates.append(pd.NaT)
            continue

        entry_price = float(entry_candidates.iloc[0]["close"])

        # Check if delisted during hold period
        is_delisted = ticker in delisted_map
        delist_date = delisted_map.get(ticker, pd.NaT)

        if is_delisted and pd.notna(delist_date):
            # Exit at last available close before/on delist_date
            exit_candidates = prices_full[prices_full["date"] <= delist_date]
            if exit_candidates.empty:
                exit_candidates = prices_full  # fallback: use whatever we have

            exit_price = float(exit_candidates.iloc[-1]["close"])
            delisted_flag = True
        else:
            # Exit at close on or before hold_end
            exit_candidates = prices_full[prices_full["date"] <= hold_end]
            if exit_candidates.empty:
                exit_price = entry_price  # no movement data
            else:
                exit_price = float(exit_candidates.iloc[-1]["close"])
            delisted_flag = False
            delist_date = pd.NaT

        raw_ret = exit_price / entry_price - 1

        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        raw_returns.append(raw_ret)
        delisted_flags.append(delisted_flag)
        delisting_dates.append(delist_date)

    df["entry_price"] = entry_prices
    df["exit_price"] = exit_prices
    df["raw_return"] = raw_returns
    df["delisted"] = delisted_flags
    df["delisting_date"] = delisting_dates

    n_delisted = df["delisted"].sum()
    n_nan = df["raw_return"].isna().sum()
    print(f"  [Returns] {len(df)} names, {n_delisted} delisted, {n_nan} NaN returns")

    return df


# =============================================================================
# SECTION 8: Portfolio construction
# =============================================================================

def build_portfolio(
    signals_df: pd.DataFrame,
    period: str,
) -> dict:
    """
    Build long/short portfolios for management strategy and benchmark.
    period: 'in_sample' or 'out_of_sample'

    Returns dict with:
      mgmt_longs, mgmt_shorts, bench_longs, bench_shorts,
      mgmt_long_weights, mgmt_short_weights, bench_long_weights, bench_short_weights
    """
    df = signals_df.dropna(subset=["raw_return"]).copy()

    if len(df) == 0:
        return {
            "mgmt_longs": [], "mgmt_shorts": [],
            "bench_longs": [], "bench_shorts": [],
            "mgmt_long_weights": {}, "mgmt_short_weights": {},
            "bench_long_weights": {}, "bench_short_weights": {},
            "n_universe": 0,
        }

    n = len(df)
    top_n = max(int(np.ceil(n * 0.20)), 30)
    bottom_n = max(int(np.ceil(n * 0.20)), 30)

    # Clamp to universe size
    top_n = min(top_n, n)
    bottom_n = min(bottom_n, n)

    # Management strategy: sort by m_score
    df_sorted_mgmt = df.sort_values("m_score", ascending=False)
    mgmt_longs = df_sorted_mgmt.iloc[:top_n]["ticker"].tolist()
    mgmt_shorts = df_sorted_mgmt.iloc[-bottom_n:]["ticker"].tolist()

    # Benchmark strategy: sort by earnings_yield
    df_sorted_bench = df.sort_values("earnings_yield", ascending=False)
    bench_longs = df_sorted_bench.iloc[:top_n]["ticker"].tolist()
    bench_shorts = df_sorted_bench.iloc[-bottom_n:]["ticker"].tolist()

    # Equal weights
    mgmt_long_weights = {t: 1.0 / len(mgmt_longs) for t in mgmt_longs}
    mgmt_short_weights = {t: 1.0 / len(mgmt_shorts) for t in mgmt_shorts}
    bench_long_weights = {t: 1.0 / len(bench_longs) for t in bench_longs}
    bench_short_weights = {t: 1.0 / len(bench_shorts) for t in bench_shorts}

    return {
        "mgmt_longs": mgmt_longs,
        "mgmt_shorts": mgmt_shorts,
        "bench_longs": bench_longs,
        "bench_shorts": bench_shorts,
        "mgmt_long_weights": mgmt_long_weights,
        "mgmt_short_weights": mgmt_short_weights,
        "bench_long_weights": bench_long_weights,
        "bench_short_weights": bench_short_weights,
        "n_universe": n,
    }


def _compute_portfolio_return(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool = False,
) -> float:
    """
    Compute weighted portfolio return.
    Transaction costs:
      - Long:  75bps one-way = 150bps round-trip → quarterly cost = 1.50%
      - Short: 75bps one-way + 37.5bps borrow per quarter → quarterly cost = 1.875%
    (Simplified: apply cost to full turnover each period since we rebalance quarterly.)
    """
    TC_LONG = 0.0150   # 150 bps round-trip
    TC_SHORT = 0.01875  # 187.5 bps round-trip + borrow

    ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()

    if not tickers:
        return 0.0

    total = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w = weights.get(t, 0.0)
        total += w * r

    if is_short:
        total = -total - TC_SHORT
    else:
        total = total - TC_LONG

    return total


# =============================================================================
# SECTION 9: Backtest engine
# =============================================================================

def run_backtest(tiingo_tickers: pd.DataFrame, prefilter_df: pd.DataFrame) -> dict:
    """
    Run the full backtest across all rebalance dates.
    Returns dict with period_records and metadata.
    """
    period_records = []
    universe_audit = []
    n_rebalance = len(REBALANCE_DATES)

    for i, rebalance_date in enumerate(REBALANCE_DATES):
        # Determine hold end (next rebalance date)
        if i + 1 < n_rebalance:
            hold_end = REBALANCE_DATES[i + 1]
        else:
            hold_end = OUT_SAMPLE_END

        # Don't backtest beyond OUT_SAMPLE_END
        if rebalance_date >= OUT_SAMPLE_END:
            break

        period = "in_sample" if rebalance_date < IN_SAMPLE_END else "out_of_sample"

        print(f"\n{'='*60}")
        print(f"Rebalance: {rebalance_date.date()}  Hold→{hold_end.date()}  [{period}]  ({i+1}/{n_rebalance})")
        print(f"{'='*60}")

        # Step 1: Universe
        try:
            universe_df = build_universe(rebalance_date, tiingo_tickers, prefilter_df)
        except Exception as e:
            print(f"  [ERROR] build_universe failed: {e}")
            continue

        if universe_df.empty:
            print(f"  [SKIP] Empty universe at {rebalance_date.date()}")
            continue

        # Step 2: Signals
        try:
            signals_df = compute_signals(universe_df, rebalance_date)
        except Exception as e:
            print(f"  [ERROR] compute_signals failed: {e}")
            continue

        # Step 3: Returns
        try:
            returns_df = compute_returns(signals_df, rebalance_date, hold_end, tiingo_tickers)
        except Exception as e:
            print(f"  [ERROR] compute_returns failed: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        n_total = len(returns_df)
        pct_delisted = n_delisted / n_total if n_total > 0 else 0.0

        universe_audit.append({
            "rebalance_date": rebalance_date,
            "hold_end": hold_end,
            "n_universe": n_total,
            "n_delisted": n_delisted,
            "pct_delisted": pct_delisted,
            "period": period,
        })

        print(f"  [Audit] Universe: {n_total}, Delisted during hold: {n_delisted} ({pct_delisted:.1%})")

        # Step 4: Portfolio
        try:
            portfolio = build_portfolio(returns_df, period)
        except Exception as e:
            print(f"  [ERROR] build_portfolio failed: {e}")
            continue

        # Step 5: Period returns
        mgmt_long_ret = _compute_portfolio_return(
            returns_df, portfolio["mgmt_longs"], portfolio["mgmt_long_weights"], is_short=False
        )
        mgmt_short_ret = _compute_portfolio_return(
            returns_df, portfolio["mgmt_shorts"], portfolio["mgmt_short_weights"], is_short=True
        )
        mgmt_return = mgmt_long_ret + mgmt_short_ret

        bench_long_ret = _compute_portfolio_return(
            returns_df, portfolio["bench_longs"], portfolio["bench_long_weights"], is_short=False
        )
        bench_short_ret = _compute_portfolio_return(
            returns_df, portfolio["bench_shorts"], portfolio["bench_short_weights"], is_short=True
        )
        bench_return = bench_long_ret + bench_short_ret

        # Per-signal IC (correlation with next-period return)
        valid = returns_df.dropna(subset=["raw_return"])
        ic_m1 = valid["M1"].corr(valid["raw_return"]) if "M1" in valid.columns else np.nan
        ic_m2 = valid["M2"].corr(valid["raw_return"]) if "M2" in valid.columns else np.nan
        ic_m3 = valid["M3"].corr(valid["raw_return"]) if "M3" in valid.columns else np.nan

        record = {
            "rebalance_date": rebalance_date,
            "hold_end": hold_end,
            "period": period,
            "n_universe": n_total,
            "n_delisted": n_delisted,
            "mgmt_long_ret": mgmt_long_ret,
            "mgmt_short_ret": mgmt_short_ret,
            "mgmt_return": mgmt_return,
            "bench_long_ret": bench_long_ret,
            "bench_short_ret": bench_short_ret,
            "bench_return": bench_return,
            "ic_m1": ic_m1,
            "ic_m2": ic_m2,
            "ic_m3": ic_m3,
            "n_mgmt_longs": len(portfolio["mgmt_longs"]),
            "n_mgmt_shorts": len(portfolio["mgmt_shorts"]),
        }
        period_records.append(record)

        print(f"  [Result] Mgmt L/S: {mgmt_return:.3%} | Bench L/S: {bench_return:.3%}")
        print(f"  [IC] M1: {ic_m1:.3f}  M2: {ic_m2:.3f}  M3: {ic_m3:.3f}")

    return {
        "period_records": period_records,
        "universe_audit": universe_audit,
    }


# =============================================================================
# SECTION 10: Metrics & output
# =============================================================================

def compute_metrics(returns_series: pd.Series) -> dict:
    """
    Compute performance metrics for a series of quarterly returns.
    """
    s = returns_series.dropna()
    n = len(s)

    if n == 0:
        return {
            "ann_return": np.nan,
            "ann_sharpe": np.nan,
            "t_stat": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "n": 0,
        }

    mean_q = s.mean()
    std_q = s.std(ddof=1) if n > 1 else 0.0

    ann_return = (1 + mean_q) ** 4 - 1
    ann_sharpe = (mean_q / std_q) * np.sqrt(4) if std_q > 0 else np.nan
    t_stat = mean_q / (std_q / np.sqrt(n)) if std_q > 0 else np.nan
    win_rate = (s > 0).mean()

    # Max drawdown via cumulative returns
    cum = (1 + s).cumprod()
    roll_max = cum.cummax()
    drawdowns = (cum - roll_max) / roll_max
    max_dd = drawdowns.min()

    return {
        "ann_return": ann_return,
        "ann_sharpe": ann_sharpe,
        "t_stat": t_stat,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n": n,
    }


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    """
    Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio threshold.
    Approximate: SR* ≈ ((1 - gamma) * Z^{-1}(1 - 1/n_trials) + gamma * Z^{-1}(1 - 1/(n_trials*e)))
    Using the simplified formula:
      SR_threshold ≈ (1 - 1/n_trials) × sqrt((1/n_obs) × (1 + sqrt(log(n_trials))))
    The key correction: multiply observed Sharpe by sqrt(n_obs) to see if it clears this bar.
    """
    from scipy import stats
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    # Expected max of n_trials iid N(0,1)
    expected_max = stats.norm.ppf(1 - 1.0 / n_trials)
    dsr_threshold = expected_max / np.sqrt(n_obs)
    return dsr_threshold


def print_results(backtest_results: dict) -> None:
    """
    Print comprehensive backtest results to console.
    """
    records = backtest_results["period_records"]
    audit = backtest_results["universe_audit"]

    if not records:
        print("\n[ERROR] No backtest periods completed.")
        return

    df = pd.DataFrame(records)
    audit_df = pd.DataFrame(audit)

    # =========================================================
    print("\n" + "=" * 70)
    print("PRE-CHECK SUMMARY")
    print("=" * 70)
    print(f"  Total rebalance periods computed:    {len(df)}")
    print(f"  In-sample periods (2015–2022):       {(df['period']=='in_sample').sum()}")
    print(f"  Out-of-sample periods (2022–2025):   {(df['period']=='out_of_sample').sum()}")
    if not audit_df.empty:
        avg_univ = audit_df["n_universe"].mean()
        avg_del = audit_df["n_delisted"].mean()
        print(f"  Avg universe size per period:        {avg_univ:.0f}")
        print(f"  Avg delistings per period:           {avg_del:.1f}")

    # =========================================================
    print("\n" + "=" * 70)
    print("SURVIVORSHIP AUDIT — Per Period")
    print("=" * 70)
    if not audit_df.empty:
        print(f"  {'Date':<14} {'N_Universe':>11} {'N_Delisted':>11} {'%_Delisted':>11}  {'Period'}")
        print(f"  {'-'*14} {'-'*11} {'-'*11} {'-'*11}  {'-'*15}")
        for _, r in audit_df.iterrows():
            print(
                f"  {str(r['rebalance_date'].date()):<14} "
                f"{r['n_universe']:>11.0f} "
                f"{r['n_delisted']:>11.0f} "
                f"{r['pct_delisted']:>10.1%}  "
                f"{r['period']}"
            )

    # =========================================================
    print("\n" + "=" * 70)
    print("IN-SAMPLE PERFORMANCE (2015-01-01 to 2022-01-01)")
    print("=" * 70)

    is_df = df[df["period"] == "in_sample"]
    os_df = df[df["period"] == "out_of_sample"]

    def _print_strategy_metrics(label: str, series: pd.Series, n_trials: int = 1) -> None:
        m = compute_metrics(series)
        print(f"\n  {label}")
        print(f"    Annualized Return:  {m['ann_return']:>9.2%}")
        print(f"    Annualized Sharpe:  {m['ann_sharpe']:>9.3f}")
        print(f"    T-Statistic:        {m['t_stat']:>9.3f}")
        print(f"    Max Drawdown:       {m['max_drawdown']:>9.2%}")
        print(f"    Win Rate:           {m['win_rate']:>9.1%}")
        print(f"    N Quarters:         {m['n']:>9d}")

        if n_trials > 1 and m["n"] > 1:
            try:
                dsr_thr = deflated_sharpe_threshold(n_trials, m["n"])
                obs_sr_ann = m["ann_sharpe"]
                obs_sr_q = obs_sr_ann / np.sqrt(4)  # quarterly Sharpe
                print(f"    Deflated SR threshold (N_trials={n_trials}): {dsr_thr:.3f} (annualized: {dsr_thr*2:.3f})")
                clears = "YES" if obs_sr_q > dsr_thr else "NO"
                print(f"    Clears DSR threshold? {clears}")
            except Exception:
                pass

    _print_strategy_metrics(
        "MANAGEMENT STRATEGY (M1+M2+M3 L/S)",
        is_df["mgmt_return"] if not is_df.empty else pd.Series(dtype=float),
        n_trials=3,  # 3 signals
    )
    _print_strategy_metrics(
        "BENCHMARK STRATEGY (Earnings Yield L/S)",
        is_df["bench_return"] if not is_df.empty else pd.Series(dtype=float),
    )

    # =========================================================
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE PERFORMANCE (2022-01-01 to 2025-10-01)")
    print("=" * 70)

    _print_strategy_metrics(
        "MANAGEMENT STRATEGY (M1+M2+M3 L/S)",
        os_df["mgmt_return"] if not os_df.empty else pd.Series(dtype=float),
        n_trials=1,  # locked thresholds, pure OOS
    )
    _print_strategy_metrics(
        "BENCHMARK STRATEGY (Earnings Yield L/S)",
        os_df["bench_return"] if not os_df.empty else pd.Series(dtype=float),
    )

    # =========================================================
    print("\n" + "=" * 70)
    print("PER-SIGNAL BREAKDOWN — Information Coefficient (IC)")
    print("  IC = cross-sectional correlation of signal with next-period return")
    print("=" * 70)

    for sig_label, col in [("M1 (Net Insider Buying)", "ic_m1"), ("M2 (No Dilution)", "ic_m2"), ("M3 (ROIC Improving)", "ic_m3")]:
        if col not in df.columns:
            continue
        ic_is = is_df[col].dropna()
        ic_os = os_df[col].dropna()
        mean_ic_is = ic_is.mean() if len(ic_is) > 0 else np.nan
        mean_ic_os = ic_os.mean() if len(ic_os) > 0 else np.nan
        t_ic_is = (ic_is.mean() / (ic_is.std() / np.sqrt(len(ic_is)))) if len(ic_is) > 1 else np.nan
        print(f"\n  {sig_label}")
        print(f"    Mean IC (in-sample):     {mean_ic_is:>8.4f}  (t={t_ic_is:.2f}, N={len(ic_is)})")
        print(f"    Mean IC (out-of-sample): {mean_ic_os:>8.4f}  (N={len(ic_os)})")

    # =========================================================
    print("\n" + "=" * 70)
    print("DEFLATED SHARPE — Multiple Testing Note")
    print("=" * 70)
    print("  N signal variations tested: 3 (M1, M2, M3)")
    print("  Note: The composite m_score was not optimized post-hoc;")
    print("  it is an equal-weight combination decided a priori.")
    print("  Deflated SR thresholds applied above assume N_trials=3 for in-sample.")
    print("  Out-of-sample Sharpe is NOT deflated (zero look-ahead).")

    # =========================================================
    print("\n" + "=" * 70)
    print("QUARTERLY RETURNS LOG")
    print("=" * 70)
    print(f"  {'Date':<14} {'Mgmt L/S':>10} {'Bench L/S':>10} {'IC_M1':>8} {'IC_M2':>8} {'IC_M3':>8}  {'Period'}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*15}")
    for _, r in df.iterrows():
        print(
            f"  {str(r['rebalance_date'].date()):<14} "
            f"{r['mgmt_return']:>9.2%}  "
            f"{r['bench_return']:>9.2%}  "
            f"{r.get('ic_m1', np.nan):>7.3f}  "
            f"{r.get('ic_m2', np.nan):>7.3f}  "
            f"{r.get('ic_m3', np.nan):>7.3f}  "
            f"{r['period']}"
        )


def save_results(backtest_results: dict) -> None:
    """Save results CSVs to results/ directory."""
    records = backtest_results["period_records"]
    audit = backtest_results["universe_audit"]

    if audit:
        audit_df = pd.DataFrame(audit)
        path = RESULTS / "universe_audit.csv"
        audit_df.to_csv(path, index=False)
        print(f"\n[Output] Saved {path}")

    if records:
        df = pd.DataFrame(records)
        is_df = df[df["period"] == "in_sample"]
        os_df = df[df["period"] == "out_of_sample"]

        is_path = RESULTS / "in_sample_perf.csv"
        os_path = RESULTS / "out_sample_perf.csv"
        is_df.to_csv(is_path, index=False)
        os_df.to_csv(os_path, index=False)
        print(f"[Output] Saved {is_path}")
        print(f"[Output] Saved {os_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run one period (2015-Q1) with a small ticker sample to validate end-to-end.")
    parser.add_argument("--prefilter-only", action="store_true",
                        help="Run SEC pre-filter only — stop before the Tiingo pull.")
    args = parser.parse_args()

    print("=" * 70)
    print("MANAGEMENT QUALITY PIT BACKTEST")
    if args.test:
        print("  *** TEST MODE: single period, small sample ***")
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not TIINGO_KEY:
        raise EnvironmentError(
            "TIINGO_API_KEY environment variable not set. "
            "Export it before running: export TIINGO_API_KEY=your_key"
        )

    # ── Step 1: Tiingo ticker list (exchange/date metadata only, no prices yet) ──
    print("\n[Step 1] Loading Tiingo ticker list (metadata only)...")
    tiingo_tickers = load_tiingo_tickers()
    print(f"  Total US-exchange stock tickers: {len(tiingo_tickers)}")

    # ── Step 2: SEC ticker→CIK map (one file, instant) ──────────────────────────
    print("\n[Step 2] Loading SEC ticker→CIK map...")
    sec_ticker_cik = load_sec_ticker_cik_map()
    print(f"  SEC CIK map entries: {len(sec_ticker_cik)}")

    # Restrict to tickers that appear in BOTH Tiingo and SEC maps
    tiingo_set = set(tiingo_tickers["ticker"].tolist())
    overlap_map = {t: c for t, c in sec_ticker_cik.items() if t in tiingo_set}
    print(f"  Overlap (Tiingo ∩ SEC): {len(overlap_map)} tickers")

    # ── Step 3: SEC pre-filter (non-financial, non-fund, has 2013–2026 filings) ─
    print("\n[Step 3] Running SEC pre-filter (cheap — no Tiingo calls)...")
    prefilter_df = build_sec_prefilter(overlap_map)
    survivors = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    n_survivors = len(survivor_tickers)

    # ── Fail-fast gate ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FAIL-FAST GATE")
    print(f"{'='*70}")
    print(f"  15,620 Tiingo tickers")
    print(f"  → {len(overlap_map):>6,}  after SEC CIK overlap filter")
    print(f"  → {n_survivors:>6,}  after non-financial / non-fund / has-filings filter")
    est_tiingo_minutes = min(n_survivors, MAX_FETCH_PER_RUN) * 5.1 / 60
    print(f"  Estimated Tiingo fetch time (new pulls only): ~{est_tiingo_minutes:.0f} min")
    print()

    if n_survivors > 10_000:
        raise RuntimeError(
            f"FAIL-FAST: {n_survivors} survivors is suspiciously high (>10,000). "
            "The SEC pre-filter may not be working correctly. Inspect sec_prefilter.csv."
        )
    if n_survivors < 500:
        raise RuntimeError(
            f"FAIL-FAST: Only {n_survivors} survivors (<500). "
            "The SEC pre-filter may be too aggressive. Inspect sec_prefilter.csv."
        )

    print(f"  [GATE PASSED] {n_survivors} survivors — proceeding to Tiingo pull.")

    if args.prefilter_only:
        print("\n[--prefilter-only] Stopping after pre-filter as requested.")
        return

    # ── Step 4: Tiingo price prefetch (survivors only) ───────────────────────────
    print("\n[Step 4] Fetching Tiingo prices for SEC-filtered survivors...")

    if args.test:
        # In test mode: use a tiny curated sample including known delisted names
        global REBALANCE_DATES
        REBALANCE_DATES = pd.DatetimeIndex([pd.Timestamp("2015-01-01")])
        test_seed = ["SUNE", "OUTR", "KATE", "BONT", "CPLA",
                     "AAPL", "MSFT", "WMT", "CVS", "HOG",
                     "CORE", "FLIR", "SBCF", "HAFC", "PFSI",
                     "RGEN", "CALD", "REGI", "IPXL", "AMED"]
        # Add up to 40 more from the filtered list
        extra = [t for t in survivor_tickers if t not in test_seed][:40]
        fetch_these = list(dict.fromkeys(test_seed + extra))[:60]
        # Restrict Tiingo tickers to the sample
        tiingo_tickers = tiingo_tickers[tiingo_tickers["ticker"].isin(fetch_these)].copy()
        # Restrict prefilter to the sample (inject test seed as "passed" even if not in SEC map)
        test_rows = [{"ticker": t, "cik": sec_ticker_cik.get(t), "sic": None, "name": t, "passed": True, "reason": "test"}
                     for t in test_seed if t not in survivor_tickers and sec_ticker_cik.get(t)]
        prefilter_df = pd.concat([prefilter_df, pd.DataFrame(test_rows)], ignore_index=True)
        survivor_tickers = fetch_these
        print(f"  [TEST] Restricted to {len(survivor_tickers)} tickers.")

    prefetch_tiingo_survivors(survivor_tickers)

    # ── Step 5: Backtest ─────────────────────────────────────────────────────────
    print("\n[Step 5] Running backtest (all data cached)...")
    backtest_results = run_backtest(tiingo_tickers, prefilter_df)

    print("\n[Step 6] Printing results...")
    print_results(backtest_results)

    print("\n[Step 7] Saving results...")
    save_results(backtest_results)

    print("\n[Done]")


if __name__ == "__main__":
    main()
