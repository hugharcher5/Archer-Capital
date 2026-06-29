"""
S-IVOL: Low Idiosyncratic Volatility  (corrected run v2)
=========================================================
Signal   : IVOL = sqrt(RSS / (n−2)) of daily r_i = α + β·r_mkt + ε
           over a rolling 60-trading-day window ending at t (dates ≤ t only).
           Market proxy = cap-weighted daily return of PIT universe,
           weights fixed at rebalance-date market cap (no look-ahead).
           Signal layer: adjClose daily returns (split+dividend adjusted) —
           prevents mid-window splits from injecting fake ±50% daily return
           into the volatility estimate.  Falls back to raw close (logged).
Portfolio: Long bottom 20% (lowest IVOL) / short top 20% (highest IVOL).
           Min 20 names per leg.  BETA-NEUTRAL (not dollar-neutral).
           Equal-weighted within each leg; leg dollar allocations scaled so
           ex-ante net portfolio beta ≈ 0:
             long_$ × mean_β_long = short_$ × mean_β_short
           Beta reused from the same 60-day OLS regression as IVOL — no
           extra computation.  Gross exposure maintained at 2× (same as
           50/50 baseline).  This is an intentional, logged deviation from
           the dollar-neutral default, specific to IVOL because its legs
           have structurally asymmetric beta (low-IVOL ≈ low-beta;
           high-IVOL ≈ high-beta).
Rebalance: Quarterly.  Hold: 1 quarter.
In-sample: 2015-01-01 → 2021-01-01 (24 periods).  No walk-forward.
Trial    : #7 (after H1, E1–E4, S9).  CORRECTED RUN — replaces prior v1
           entry in registry (same hypothesis, NOT a new independent trial).
           DSR note: IVOL and the upcoming MAX strategy test the same
           hypothesis (extreme-return aversion) and are treated as one
           family — N_TRIALS=7 unchanged.

Corrections vs v1 (prior run):
  1. Beta-neutral leg sizing replaces 50/50 dollar-neutral.
     v1 produced β = −0.655 (t = −5.1), contaminating −76.9% return with
     a directional short-market bet rather than the IVOL signal.
  2. AdjClose coverage repaired: all 1,143 .EMPTY sentinels retried with
     a 30s connect timeout and 3 attempts before accepting "genuinely empty".
     Prior run had ~438 ConnectTimeout false negatives → fallback inflated.

No-survivorship guarantee
  - Delisted tickers retained with returns to their last trading day.
  - Universe rebuilt PIT from SEC filings each period.

No-look-ahead guarantee
  - IVOL window uses adjClose for dates strictly ≤ t.
  - Market proxy also built from data ≤ t.
  - Market cap = SEC shares-as-filed (filing_date ≤ t) × last_close ≤ t.
  - Entry price = first close on or after t; returns measured t → t+1Q.
"""

# =============================================================================
# SECTION 1: Imports & config
# =============================================================================

import io
import json
import os
import pickle
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats


def _load_dotenv() -> None:
    """Load .env from repo root if TIINGO_API_KEY not already in environment."""
    if os.environ.get("TIINGO_API_KEY"):
        return
    env_path = Path(__file__).parent.parent.parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")
SEC_UA     = "Hugh Archer hugharcher5@gmail.com"

# Shared cache with mgmt_pit (prices, sec_facts, prefilter all already built)
MGMT_PIT = Path(__file__).parent.parent / "mgmt_pit"
CACHE    = MGMT_PIT / "cache"
RESULTS  = Path(__file__).parent / "results"
REGISTRY = Path(__file__).parent.parent / "trial_registry.csv"

for _d in [RESULTS, CACHE / "tiingo", CACHE / "tiingo_adj", CACHE / "sec_facts"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S9) ──────────────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
SIC_EXCLUDE    = set(range(6000, 6800))
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── IVOL signal parameters ───────────────────────────────────────────────────
IVOL_WINDOW_DAYS  = 60    # trading days for the regression window
IVOL_MIN_DAYS     = 40    # minimum non-NaN observations to compute valid IVOL
IVOL_LOOKBACK_CAL = 95    # calendar days to look back (covers ≥60 trading days)

# ── Backtest dates ───────────────────────────────────────────────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = _all_dates[_all_dates < IS_END]   # 24 periods

# ── Cost model (identical to S9 / mgmt_pit) ─────────────────────────────────
TURNOVER_RATE     = 0.50
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.02

# ── DSR ──────────────────────────────────────────────────────────────────────
# IVOL + upcoming MAX are the same family (extreme-return aversion);
# treated as one independent trial.  Total distinct strategies = 7.
N_TRIALS = 7   # H1 + E1 + E2 + E3 + E4 + S9 + (IVOL/MAX family)

_TIINGO_LAST: float = 0.0
_SEC_LAST:    float = 0.0


# =============================================================================
# SECTION 2: Rate-limited HTTP
# =============================================================================

def _tiingo_get(url: str, params: dict = None) -> requests.Response:
    global _TIINGO_LAST
    wait = 1.5 - (time.time() - _TIINGO_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, params=params, timeout=(10, 20))
    _TIINGO_LAST = time.time()
    return r


def _sec_get(url: str) -> requests.Response:
    global _SEC_LAST
    wait = 0.2 - (time.time() - _SEC_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=(10, 30))
    _SEC_LAST = time.time()
    return r


# =============================================================================
# SECTION 3: Tiingo helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    """Load Tiingo supported_tickers (cached). US exchange stocks only."""
    path = CACHE / "tickers.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
        return df

    print("[Tiingo] Downloading supported_tickers.zip...")
    url  = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as f:
            df_all = pd.read_csv(f)
    df_all.columns = [c.strip() for c in df_all.columns]
    df = df_all[
        df_all["exchange"].isin(US_EXCHANGES) & (df_all["assetType"] == "Stock")
    ].copy()
    for col in ["startDate", "endDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["startDate"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[Tiingo] {len(df):,} tickers saved.")
    return df


# =============================================================================
# SECTION 4: SEC helpers
# =============================================================================

def load_sec_ticker_cik_map() -> dict:
    """Load SEC ticker→CIK map (cached)."""
    path = CACHE / "sec_ticker_cik.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        print(f"[SEC] Loaded {len(raw):,} ticker→CIK pairs from cache.")
        return raw

    print("[SEC] Downloading company_tickers.json...")
    resp = _sec_get("https://www.sec.gov/files/company_tickers.json")
    resp.raise_for_status()
    data = resp.json()
    out: dict = {}
    for entry in data.values():
        t       = entry.get("ticker", "").upper().strip()
        cik_raw = entry.get("cik_str", "")
        if t and cik_raw:
            try:
                out[t] = int(cik_raw)
            except ValueError:
                pass
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def load_prefilter() -> pd.DataFrame:
    """Load mgmt_pit's SEC pre-filter cache (already built)."""
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"sec_prefilter.csv not found at {path}. Run mgmt_pit/run.py first."
        )
    df        = pd.read_csv(path)
    survivors = df[df["passed"]]
    print(f"[PreFilter] {len(df):,} screened, {len(survivors):,} survivors (mgmt_pit cache).")
    return df


# =============================================================================
# SECTION 5: In-memory raw price cache  —  RETURN LAYER
# =============================================================================

_PRICE_CACHE:    dict[str, pd.DataFrame] = {}
_PRICE_SENTINEL: set[str]               = set()
_PRICE_EMPTY = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

_PRICES_PKL = CACHE / "prices_all.pkl"


def _load_single(ticker: str) -> pd.DataFrame:
    if ticker in _PRICE_CACHE:
        return _PRICE_CACHE[ticker]
    if ticker in _PRICE_SENTINEL:
        return _PRICE_EMPTY
    sentinel = CACHE / "tiingo" / f"{ticker}.EMPTY"
    if sentinel.exists():
        _PRICE_SENTINEL.add(ticker)
        return _PRICE_EMPTY
    csv = CACHE / "tiingo" / f"{ticker}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["date"])
        _PRICE_CACHE[ticker] = df
        return df
    return _PRICE_EMPTY


def preload_all_prices(tickers: list[str]) -> None:
    """Bulk-load all raw close CSVs into memory once."""
    t0 = time.time()
    if _PRICES_PKL.exists():
        with open(_PRICES_PKL, "rb") as f:
            data = pickle.load(f)
        _PRICE_CACHE.update(data.get("prices", {}))
        _PRICE_SENTINEL.update(data.get("sentinels", set()))
        for t in tickers:
            if t not in _PRICE_CACHE and t not in _PRICE_SENTINEL:
                _load_single(t)
        print(f"[Preload] Raw prices: {len(_PRICE_CACHE):,} tickers in {time.time()-t0:.1f}s")
        return

    for t in tickers:
        _load_single(t)
    elapsed = time.time() - t0
    print(f"[Preload] Raw prices: {len(_PRICE_CACHE):,} CSVs in {elapsed:.1f}s — saving pickle...")
    t1 = time.time()
    with open(_PRICES_PKL, "wb") as f:
        pickle.dump({"prices": dict(_PRICE_CACHE), "sentinels": set(_PRICE_SENTINEL)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Preload] Pickle saved in {time.time()-t1:.1f}s")


# =============================================================================
# SECTION 5b: Adjusted-price cache  —  SIGNAL LAYER
# =============================================================================
#
# Tiingo adjClose = backward-adjusted for BOTH splits AND dividends.
# Current price is unchanged; all prior prices are scaled by each corporate
# action factor.  Using adjClose for daily return computation prevents a 2:1
# split from injecting a −50% outlier into the 60-day variance estimate.

_ADJ_CACHE:    dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str]               = set()
_ADJ_EMPTY = pd.DataFrame(columns=["date", "adjClose"])

_ADJ_DIR = CACHE / "tiingo_adj"
_ADJ_PKL = CACHE / "adjclose_all.pkl"
_ADJ_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_adj_from_tiingo(ticker: str) -> pd.DataFrame:
    """Fetch adjClose series from Tiingo daily prices endpoint."""
    if not TIINGO_KEY:
        return _ADJ_EMPTY
    url    = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {"startDate": "2012-01-01", "token": TIINGO_KEY}
    try:
        resp = _tiingo_get(url, params)
        if resp.status_code == 404:
            return _ADJ_EMPTY
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return _ADJ_EMPTY
        rows = []
        for r in data:
            adj = r.get("adjClose")
            if adj is None:
                continue
            rows.append({"date": pd.Timestamp(str(r["date"])[:10]), "adjClose": float(adj)})
        if not rows:
            return _ADJ_EMPTY
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"  [AdjFetch] {ticker}: {type(e).__name__}: {e}")
        return _ADJ_EMPTY


def _load_adj(ticker: str) -> pd.DataFrame:
    """Load adjClose from disk cache; fetch from Tiingo on miss."""
    if ticker in _ADJ_CACHE:
        return _ADJ_CACHE[ticker]
    if ticker in _ADJ_SENTINEL:
        return _ADJ_EMPTY

    sentinel = _ADJ_DIR / f"{ticker}.EMPTY"
    if sentinel.exists():
        _ADJ_SENTINEL.add(ticker)
        return _ADJ_EMPTY

    csv = _ADJ_DIR / f"{ticker}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["date"])
        _ADJ_CACHE[ticker] = df
        return df

    df = _fetch_adj_from_tiingo(ticker)
    if df.empty:
        sentinel.touch()
        _ADJ_SENTINEL.add(ticker)
        return _ADJ_EMPTY
    df.to_csv(csv, index=False)
    _ADJ_CACHE[ticker] = df
    return df


def preload_all_adj_prices(tickers: list[str]) -> None:
    """Bulk-load adjClose.  First run: Tiingo API (~1.5s/ticker).  Subsequent: pickle."""
    t0 = time.time()

    if _ADJ_PKL.exists():
        with open(_ADJ_PKL, "rb") as f:
            data = pickle.load(f)
        _ADJ_CACHE.update(data.get("adj", {}))
        _ADJ_SENTINEL.update(data.get("sentinels", set()))
        print(f"[AdjPreload] {len(_ADJ_CACHE):,} tickers from pickle in {time.time()-t0:.1f}s")

    missing = [t for t in tickers if t not in _ADJ_CACHE and t not in _ADJ_SENTINEL]
    need_api: list[str] = []
    for t in missing:
        if (_ADJ_DIR / f"{t}.EMPTY").exists():
            _ADJ_SENTINEL.add(t)
        elif (_ADJ_DIR / f"{t}.csv").exists():
            df = pd.read_csv(_ADJ_DIR / f"{t}.csv", parse_dates=["date"])
            _ADJ_CACHE[t] = df
        else:
            need_api.append(t)

    if need_api:
        if not TIINGO_KEY:
            print(f"[AdjPreload] WARNING: TIINGO_API_KEY not set — {len(need_api)} tickers will "
                  f"fall back to raw close for signal computation. Adj layer disabled.")
        else:
            est_min = len(need_api) * 1.5 / 60
            print(f"[AdjPreload] {len(need_api):,} tickers need API fetch (est {est_min:.0f} min)...")
            for i, t in enumerate(need_api):
                if (i + 1) % 100 == 0:
                    rem = (len(need_api) - i - 1) * 1.5 / 60
                    print(f"  [AdjPreload] {i+1}/{len(need_api)} | ~{rem:.0f} min remaining | last={t}")
                _load_adj(t)
            print(f"[AdjPreload] API fetch done in {time.time()-t0:.1f}s")
            with open(_ADJ_PKL, "wb") as f:
                pickle.dump({"adj": dict(_ADJ_CACHE), "sentinels": set(_ADJ_SENTINEL)}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[AdjPreload] Pickle saved.")

    loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    print(f"[AdjPreload] Total: {loaded:,} tickers with adjClose ({time.time()-t0:.1f}s)")


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return adjClose rows for [start, end]. No I/O — in-memory only."""
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 5c: Adj-sentinel repair  (retry ConnectTimeout false negatives)
# =============================================================================

def repair_adj_sentinels(tickers: list[str]) -> dict:
    """
    Retry adjClose fetch for all .EMPTY-sentinelled tickers.
    Prior runs wrote .EMPTY for ConnectTimeouts (not genuine missing data).
    Uses 30s connect timeout and up to 3 attempts before accepting failure.
    Removes .EMPTY and writes CSV for any ticker that now returns data.
    Rebuilds adjclose_all.pkl when done.
    Returns {"retried": N, "recovered": N, "still_empty": N}.
    """
    if not TIINGO_KEY:
        print("[AdjRepair] No API key — skipping repair.")
        return {"retried": 0, "recovered": 0, "still_empty": 0}

    to_retry = sorted(
        t for t in tickers
        if (_ADJ_DIR / f"{t}.EMPTY").exists()
        and not (_ADJ_DIR / f"{t}.csv").exists()
    )
    if not to_retry:
        print("[AdjRepair] No .EMPTY sentinels found — nothing to retry.")
        return {"retried": 0, "recovered": 0, "still_empty": 0}

    print(f"[AdjRepair] {len(to_retry):,} .EMPTY sentinels to retry "
          f"(30s timeout, up to 3 attempts) — est {len(to_retry)*1.5/60:.0f} min...")

    recovered   = 0
    still_empty = 0
    t0 = time.time()
    global _TIINGO_LAST

    for i, ticker in enumerate(to_retry):
        if (i + 1) % 100 == 0:
            rem = (len(to_retry) - i - 1) * 1.5 / 60
            pct = 100 * recovered / max(i + 1, 1)
            print(f"  [AdjRepair] {i+1}/{len(to_retry)} | ~{rem:.0f} min left | "
                  f"recovered {recovered} ({pct:.0f}%) | last={ticker}")

        df = _ADJ_EMPTY
        url    = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
        params = {"startDate": "2012-01-01", "token": TIINGO_KEY}

        for attempt in range(3):
            wait = 1.5 - (time.time() - _TIINGO_LAST)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = requests.get(url, params=params, timeout=(30, 60))
                _TIINGO_LAST = time.time()
                if resp.status_code == 404:
                    break                       # genuine not found
                resp.raise_for_status()
                data = resp.json()
                if not data:
                    break
                rows = []
                for r in data:
                    adj = r.get("adjClose")
                    if adj is None:
                        continue
                    rows.append({"date": pd.Timestamp(str(r["date"])[:10]),
                                 "adjClose": float(adj)})
                if rows:
                    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
                break
            except Exception:
                _TIINGO_LAST = time.time()
                if attempt < 2:
                    time.sleep(2 ** attempt)    # 1s, 2s back-off before next attempt

        if not df.empty:
            csv_path = _ADJ_DIR / f"{ticker}.csv"
            df.to_csv(csv_path, index=False)
            (_ADJ_DIR / f"{ticker}.EMPTY").unlink(missing_ok=True)
            _ADJ_CACHE[ticker] = df
            _ADJ_SENTINEL.discard(ticker)
            recovered += 1
        else:
            still_empty += 1

    elapsed = time.time() - t0
    print(f"[AdjRepair] Done in {elapsed:.1f}s ({elapsed/60:.1f} min): "
          f"{recovered:,} recovered, {still_empty:,} genuinely empty.")

    t1 = time.time()
    with open(_ADJ_PKL, "wb") as f:
        pickle.dump({"adj": dict(_ADJ_CACHE), "sentinels": set(_ADJ_SENTINEL)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[AdjRepair] Pickle rebuilt in {time.time()-t1:.1f}s")

    return {"retried": len(to_retry), "recovered": recovered, "still_empty": still_empty}


# =============================================================================
# SECTION 6: SEC facts (shares outstanding)
# =============================================================================

_FACTS_CACHE: dict[int, Optional[dict]] = {}
_FACTS_TRIMMED = CACHE / "sec_facts_trimmed.json"
_SHARES_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]


def preload_sec_facts(ciks: list[int]) -> None:
    """Load shares-outstanding facts from trimmed consolidated cache (mgmt_pit)."""
    t0 = time.time()
    if _FACTS_TRIMMED.exists():
        with open(_FACTS_TRIMMED) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _FACTS_CACHE[int(k)] = v
        for cik in ciks:
            if cik not in _FACTS_CACHE:
                p = CACHE / "sec_facts" / f"{cik}.json"
                if p.exists():
                    try:
                        with open(p) as f2:
                            full = json.load(f2)
                        trimmed: dict = {"facts": {"us-gaap": {}}}
                        ug = full.get("facts", {}).get("us-gaap", {})
                        for c in _SHARES_CONCEPTS:
                            if c in ug:
                                trimmed["facts"]["us-gaap"][c] = ug[c]
                        _FACTS_CACHE[cik] = trimmed
                    except Exception:
                        _FACTS_CACHE[cik] = None
                else:
                    _FACTS_CACHE[cik] = None
        loaded = sum(1 for v in _FACTS_CACHE.values() if v is not None)
        print(f"[Preload] SEC facts: {loaded:,} loaded in {time.time()-t0:.1f}s")
        return

    loaded = 0
    for cik in ciks:
        p = CACHE / "sec_facts" / f"{cik}.json"
        if p.exists():
            try:
                with open(p) as f:
                    full = json.load(f)
                trimmed = {"facts": {"us-gaap": {}}}
                ug = full.get("facts", {}).get("us-gaap", {})
                for c in _SHARES_CONCEPTS:
                    if c in ug:
                        trimmed["facts"]["us-gaap"][c] = ug[c]
                _FACTS_CACHE[cik] = trimmed
                loaded += 1
            except Exception:
                _FACTS_CACHE[cik] = None
        else:
            _FACTS_CACHE[cik] = None
    print(f"[Preload] SEC facts: {loaded:,} loaded (individual files) in {time.time()-t0:.1f}s")


def _get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    """Most recent CommonStockSharesOutstanding where filing_date ≤ as_of."""
    facts    = _FACTS_CACHE.get(cik)
    if facts is None:
        return None
    as_of_str = as_of.strftime("%Y-%m-%d")

    for concept in _SHARES_CONCEPTS:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue

        best: Optional[tuple] = None
        for entries in units_dict.values():
            for entry in entries:
                filed = entry.get("filed")
                end   = entry.get("end")
                val   = entry.get("val")
                if filed is None or end is None or val is None:
                    continue
                if filed > as_of_str:
                    continue
                key = (filed, end)
                if best is None or key > best[:2]:
                    best = (filed, end, float(val))
        if best is not None and best[2] > 0:
            return best[2]

    return None


# =============================================================================
# SECTION 7: Universe builder
# =============================================================================

def _last_close(ticker: str, as_of: pd.Timestamp) -> Optional[float]:
    """Last raw close on or before as_of."""
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return None
    sub = df[df["date"] <= as_of]
    if sub.empty:
        return None
    return float(sub.iloc[-1]["close"])


def _avg_dollar_vol(ticker: str, as_of: pd.Timestamp, lookback: int = 60) -> float:
    """Average daily dollar volume over last `lookback` trading days before as_of."""
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return 0.0
    sub = df[df["date"] <= as_of].tail(lookback)
    if sub.empty:
        return 0.0
    return float((sub["close"] * sub["volume"]).mean())


def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
) -> pd.DataFrame:
    """
    PIT investable universe at rebalance_date.
    Returns DataFrame[ticker, cik, mcap, price, shares, avg_dvol].
    Filters: active on t, SEC survivors, mcap $100M–$2B, ADTV ≥ $1M, price ≥ $1.
    """
    survivors_set = set(cik_sic_map.keys())
    active = tiingo_tickers[
        (tiingo_tickers["startDate"] <= rebalance_date)
        & (
            tiingo_tickers["endDate"].isna()
            | (tiingo_tickers["endDate"] >= rebalance_date)
        )
        & tiingo_tickers["ticker"].isin(survivors_set)
    ]

    rows = []
    for _, row in active.iterrows():
        ticker    = row["ticker"]
        cik, sic  = cik_sic_map.get(ticker, (None, None))
        if cik is None:
            continue

        price = _last_close(ticker, rebalance_date)
        if price is None or price < PRICE_MIN:
            continue

        shares = _get_shares_pit(cik, rebalance_date)
        if shares is None or shares <= 0:
            continue
        mcap = shares * price
        if not (MCAP_MIN <= mcap <= MCAP_MAX):
            continue

        dvol = _avg_dollar_vol(ticker, rebalance_date)
        if dvol < DOLLAR_VOL_MIN:
            continue

        rows.append({"ticker": ticker, "cik": cik, "mcap": mcap,
                     "price": price, "shares": shares, "avg_dvol": dvol})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    print(f"  [Universe] {rebalance_date.date()} → {len(df):,} names "
          f"(active={len(active):,}, post-filter={len(df):,})")
    return df


# =============================================================================
# SECTION 8: IVOL signal  (signal layer — adjClose)
# =============================================================================

def compute_ivol_signals(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple:
    """
    Compute IVOL for all universe members in one vectorized pass.

    Algorithm:
      1. Collect adjClose (or raw close fallback) over [t-95 cal days, t].
      2. Build price matrix → daily return matrix.
      3. Cap-weighted market return (weights = rebalance-date mcap, renormalized
         each day by available tickers — no look-ahead).
      4. Per-ticker OLS: r_i = α + β·r_mkt + ε
         IVOL = sqrt(RSS / (n−2))  [unbiased residual std, df=2 for intercept+slope]
      5. Portfolio signal = −IVOL  (higher signal → lower IVOL → long leg).

    SIGNAL LAYER: adjClose (split+dividend adjusted).
    Falls back to raw close when adjClose unavailable — counted in n_fallback.

    Returns:
      ivol_df     : DataFrame[ticker, ivol_signal]
      n_fallback  : tickers that used raw close (should be ≈0 after adj preload)
      n_total     : tickers that entered computation (had enough data)
    """
    lookback_start = rebalance_date - pd.Timedelta(days=IVOL_LOOKBACK_CAL)

    # ── Collect price series ─────────────────────────────────────────────────
    price_series: dict[str, pd.Series] = {}
    raw_fallback:  set[str]            = set()

    for _, row in universe_df.iterrows():
        ticker = row["ticker"]

        adj = _adj_slice(ticker, lookback_start, rebalance_date)
        if len(adj) >= IVOL_MIN_DAYS + 1:
            price_series[ticker] = adj.set_index("date")["adjClose"]
            continue

        # Fallback: raw close
        raw_df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
        if raw_df.empty:
            continue
        window = raw_df[
            (raw_df["date"] >= lookback_start) & (raw_df["date"] <= rebalance_date)
        ]
        if len(window) >= IVOL_MIN_DAYS + 1:
            price_series[ticker] = window.set_index("date")["close"]
            raw_fallback.add(ticker)

    n_total    = len(price_series)
    n_fallback = len(raw_fallback)

    if n_total < 10:
        return pd.DataFrame(columns=["ticker", "ivol_signal"]), n_fallback, n_total

    # ── Daily return matrix ──────────────────────────────────────────────────
    price_mat = pd.DataFrame(price_series).sort_index()
    ret_mat   = price_mat.pct_change().iloc[1:]          # drop first all-NaN row

    # Use only the last IVOL_WINDOW_DAYS rows (≈60 trading days ending at t)
    ret_mat = ret_mat.tail(IVOL_WINDOW_DAYS)

    # Keep tickers with enough valid observations
    valid_counts = ret_mat.notna().sum()
    ret_mat = ret_mat.loc[:, valid_counts >= IVOL_MIN_DAYS]

    if ret_mat.shape[1] < 5:
        return pd.DataFrame(columns=["ticker", "ivol_signal"]), n_fallback, n_total

    # ── Cap-weighted market return ───────────────────────────────────────────
    # Weights fixed at rebalance-date market cap (no intra-period rebalancing).
    # Renormalized daily by available tickers so missing data doesn't bias mkt ret.
    mcap_s  = universe_df.set_index("ticker")["mcap"]
    w       = mcap_s.reindex(ret_mat.columns).fillna(0.0)
    total_w = w.sum()
    w_norm  = w / total_w if total_w > 0 else pd.Series(
        1.0 / len(ret_mat.columns), index=ret_mat.columns
    )

    avail       = ret_mat.notna()
    weighted_r  = ret_mat.multiply(w_norm, axis=1)
    denom       = avail.multiply(w_norm, axis=1).sum(axis=1)
    mkt_ret     = weighted_r.sum(axis=1) / denom.replace(0, np.nan)

    # ── OLS IVOL per ticker ──────────────────────────────────────────────────
    records: list[dict] = []
    for ticker in ret_mat.columns:
        r_i  = ret_mat[ticker]
        mask = r_i.notna() & mkt_ret.notna()
        n    = int(mask.sum())
        if n < IVOL_MIN_DAYS or n <= 2:
            continue

        y = r_i[mask].to_numpy(dtype=float)
        x = mkt_ret[mask].to_numpy(dtype=float)

        X = np.column_stack([np.ones(n), x])
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        if rank < 2:
            continue

        eps  = y - X @ coeffs
        ivol = float(np.sqrt(np.dot(eps, eps) / (n - 2)))   # sqrt(RSS / df)

        if not np.isfinite(ivol) or ivol <= 0:
            continue

        # Negate so that high signal = low IVOL = long; consistent with build_portfolio
        # coeffs[1] = ex-ante market beta; reused in build_portfolio for beta-neutral sizing
        records.append({"ticker": ticker, "ivol_signal": -ivol, "beta": float(coeffs[1])})

    ivol_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["ticker", "ivol_signal", "beta"])
    )
    return ivol_df, n_fallback, n_total


# =============================================================================
# SECTION 9: Return calculation  (return layer — raw close)
# =============================================================================

def _get_price_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return _PRICE_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
    delisted_map: dict,
) -> pd.DataFrame:
    """
    Survivorship-bias-free holding-period returns using RAW close.
    Delistings: exit at last available close ≤ delist_date.
    Entry: first close on or after hold_start.
    Exit : close on or before hold_end (or earlier delist_date).
    """
    df = universe_df.copy()
    entry_prices, exit_prices, raw_returns, delisted_flags = [], [], [], []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        prices = _get_price_slice(ticker, hold_start, hold_end + pd.Timedelta(days=5))

        if prices.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan);  delisted_flags.append(False)
            continue

        entry_cands = prices[prices["date"] >= hold_start]
        if entry_cands.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan);  delisted_flags.append(False)
            continue

        entry_price = float(entry_cands.iloc[0]["close"])
        delist_date = delisted_map.get(ticker, pd.NaT)
        is_delisted = pd.notna(delist_date) and (delist_date < hold_end)

        if is_delisted:
            exit_cands  = prices[prices["date"] <= delist_date]
            exit_price  = float(
                (exit_cands if not exit_cands.empty else prices).iloc[-1]["close"]
            )
            delisted_flags.append(True)
        else:
            exit_cands  = prices[prices["date"] <= hold_end]
            exit_price  = float(exit_cands.iloc[-1]["close"]) if not exit_cands.empty else entry_price
            delisted_flags.append(False)

        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        raw_returns.append(exit_price / entry_price - 1)

    df["entry_price"] = entry_prices
    df["exit_price"]  = exit_prices
    df["raw_return"]  = raw_returns
    df["delisted"]    = delisted_flags
    return df


# =============================================================================
# SECTION 10: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    """One-way half-spread: 100bps at $100M → 20bps at $2B."""
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    """Annual borrow: 2% at $100M → 0.5% at $2B."""
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame) -> tuple:
    """
    Long top 20% of ivol_signal (= lowest IVOL), short bottom 20% (= highest IVOL).
    Min 20 per leg.  Equal-weight within each leg.
    Beta-neutral leg sizing:
      long_$ × mean_β_long = short_$ × mean_β_short  (gross = 2×, same as 50/50 baseline)
    Returns (longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short).
    """
    df = signals_df.dropna(subset=["ivol_signal", "raw_return"]).copy()
    n  = len(df)
    if n < 40:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan

    leg = max(int(np.ceil(n * 0.20)), 20)
    leg = min(leg, n // 2)

    ranked = df.sort_values("ivol_signal", ascending=False)   # highest signal first = lowest IVOL
    longs  = ranked.iloc[:leg]["ticker"].tolist()             # lowest IVOL
    shorts = ranked.iloc[-leg:]["ticker"].tolist()            # highest IVOL

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    # ── Beta-neutral leg sizing ───────────────────────────────────────────────
    # Reuse ex-ante beta from the 60-day OLS regression already computed in
    # compute_ivol_signals.  Scale so: long_$ × β_L = short_$ × β_S.
    # With gross = 2: long_$ = 2β_S/(β_L+β_S), short_$ = 2β_L/(β_L+β_S).
    if "beta" in df.columns:
        beta_map    = df.set_index("ticker")["beta"].to_dict()
        bl_vals     = [beta_map.get(t, np.nan) for t in longs]
        bs_vals     = [beta_map.get(t, np.nan) for t in shorts]
        mean_beta_long  = float(np.nanmean(bl_vals))
        mean_beta_short = float(np.nanmean(bs_vals))
        denom = mean_beta_long + mean_beta_short
        if np.isfinite(denom) and denom > 0.05:
            long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
            short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
        else:
            print(f"  [Portfolio] β_L+β_S ≤ 0.05 — falling back to 50/50.")
            long_scale = short_scale = 1.0
    else:
        mean_beta_long = mean_beta_short = np.nan
        long_scale = short_scale = 1.0

    return longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short


def _leg_return(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool,
) -> float:
    """Weighted gross return for one leg, net of bid-ask + borrow costs."""
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()

    gross = 0.0
    cost  = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, (MCAP_MIN + MCAP_MAX) / 2)
        gross += w * r
        cost  += TURNOVER_RATE * 2 * _spread(mc) * w
        if is_short:
            cost += (_borrow(mc) / 4) * w   # quarterly borrow

    return (-gross - cost) if is_short else (gross - cost)


# =============================================================================
# SECTION 10b: Split-contamination validation  (adj-layer sanity check)
# =============================================================================

_MOMENTUM_MIN_DAYS = 180   # minimum trading days for the 12-month validation window


def validate_split_fix() -> None:
    """
    Scan raw-close cache for 2:1 splits inside a 12-month formation window.
    Shows momentum signal under raw close vs Tiingo adjClose to confirm the
    adj layer is populated with real API data (not manual reconstruction).
    This is a one-shot sanity check — remove once satisfied.
    """
    print("\n=== SPLIT CONTAMINATION VALIDATION (adj-layer check) ===")
    print("adjClose: Tiingo adjusts BACKWARD for SPLITS + DIVIDENDS (total-return proxy).")
    print("Scanning raw close cache for 2:1 splits (overnight ratio ≈ 0.50) in 2013–2020...\n")

    found: list[dict] = []
    for csv_path in sorted((CACHE / "tiingo").glob("*.csv")):
        if len(found) >= 3:
            break
        ticker = csv_path.stem
        try:
            raw = pd.read_csv(csv_path, parse_dates=["date"])
            if len(raw) < 100:
                continue
            raw = raw.sort_values("date").reset_index(drop=True)
            sub = raw[(raw["date"] >= "2013-01-01") & (raw["date"] <= "2021-01-01")]
            if len(sub) < 20:
                continue
            close  = sub["close"].values
            dates  = sub["date"].values
            ratios = close[1:] / close[:-1]
            for i, r in enumerate(ratios):
                if 0.47 <= r <= 0.53 and close[i] >= 5.0:
                    found.append({
                        "ticker":       ticker,
                        "split_date":   pd.Timestamp(dates[i + 1]),
                        "price_before": float(close[i]),
                        "price_after":  float(close[i + 1]),
                    })
                    break
        except Exception:
            continue

    if not found:
        print("  No 2:1 split candidates found in raw cache. Validation skipped.")
        print("=== END VALIDATION ===\n")
        return

    print(f"  Found {len(found)} candidate 2:1 split(s).\n")

    for cand in found[:2]:
        ticker     = cand["ticker"]
        split_date = cand["split_date"]

        matching = [
            d for d in REBALANCE_DATES
            if (d - pd.DateOffset(months=12)) <= split_date < (d - pd.DateOffset(months=1))
        ]
        if not matching:
            print(f"  {ticker}: split {split_date.date()} outside all formation windows — skip\n")
            continue

        t         = matching[0]
        mom_start = t - pd.DateOffset(months=12)
        mom_end   = t - pd.DateOffset(months=1)

        print(f"  Ticker        : {ticker}")
        print(f"  Split detected: {split_date.date()}  "
              f"(${cand['price_before']:.2f} → ${cand['price_after']:.2f}, "
              f"ratio {cand['price_after']/cand['price_before']:.3f} ≈ 2:1)")
        print(f"  Formation wnd : {mom_start.date()} → {mom_end.date()}")

        # ── Raw-close signal (BEFORE fix) ────────────────────────────────────
        raw_csv = CACHE / "tiingo" / f"{ticker}.csv"
        raw_sig = None
        if raw_csv.exists():
            raw_df = pd.read_csv(raw_csv, parse_dates=["date"])
            w = raw_df[(raw_df["date"] >= mom_start) & (raw_df["date"] <= mom_end)]
            if len(w) >= _MOMENTUM_MIN_DAYS:
                sp, ep  = float(w.iloc[0]["close"]), float(w.iloc[-1]["close"])
                raw_sig = ep / sp - 1
                print(f"  Raw close signal  (BEFORE fix): {raw_sig:>+9.2%}"
                      f"  (start=${sp:.2f}, end=${ep:.2f}  — split distorts window)")

        # ── AdjClose signal (AFTER fix) ──────────────────────────────────────
        adj_df  = _load_adj(ticker)   # loads from disk cache (already fetched)
        adj_sig = None
        if not adj_df.empty:
            wa = adj_df[(adj_df["date"] >= mom_start) & (adj_df["date"] <= mom_end)]
            if len(wa) >= _MOMENTUM_MIN_DAYS:
                sp_a, ep_a = float(wa.iloc[0]["adjClose"]), float(wa.iloc[-1]["adjClose"])
                adj_sig    = ep_a / sp_a - 1
                print(f"  AdjClose signal   (AFTER fix):  {adj_sig:>+9.2%}"
                      f"  (start=${sp_a:.4f}, end=${ep_a:.4f}  — real Tiingo adjClose)")
        else:
            print(f"  AdjClose not available for {ticker} — skipping adj comparison.")

        if raw_sig is not None and adj_sig is not None:
            delta = adj_sig - raw_sig
            print(f"  Signal delta                  : {delta:>+9.2%}")
            if raw_sig < -0.10 and adj_sig > 0.05:
                print(f"  ★ Winner misclassified as LOSER by raw close → would have been SHORTED")
            elif raw_sig > 0.10 and adj_sig < -0.05:
                print(f"  ★ Loser misclassified as WINNER by raw close → would have been LONGED")
            else:
                sign_str = "preserved" if (raw_sig * adj_sig > 0) else "FLIPPED"
                print(f"  Signal corrupted by {abs(delta):.1%}; sign {sign_str}")
        print()

    print("=== END VALIDATION ===\n")


# =============================================================================
# SECTION 11: Backtest engine
# =============================================================================

def run_backtest(
    tiingo_tickers: pd.DataFrame,
    prefilter_df: pd.DataFrame,
) -> dict:
    """
    Run in-sample backtest (2015-Q1 → 2020-Q4, 24 periods).
    All data loaded into memory before entering the rebalance loop.
    Returns dict with period_records, universe_audit, adj_coverage, elapsed_s.
    """
    t_total_start = time.time()

    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {
        r["ticker"]: (int(r["cik"]), r["sic"])
        for _, r in survivors.iterrows()
    }

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        end = r["endDate"]
        if pd.notna(end):
            delisted_map[r["ticker"]] = end

    period_records: list = []
    universe_audit: list = []
    adj_coverage:   list = []

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        t_next     = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        hold_start = t
        hold_end   = t_next

        print(f"\n{'='*60}")
        print(f"Rebalance {t.date()} → hold end {hold_end.date()}  ({i+1}/{n_periods})")
        print(f"{'='*60}")

        # ── 1. Universe ──────────────────────────────────────────────────────
        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERROR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] Empty universe.")
            continue

        # ── 2. IVOL signals (vectorized over all universe members) ───────────
        ivol_df, n_fallback, n_computed = compute_ivol_signals(univ, t)
        n_adj_ok = n_computed - n_fallback

        pct_fallback = 100 * n_fallback / max(n_computed, 1)
        print(f"  [Signal] {len(ivol_df)}/{len(univ)} have valid IVOL  "
              f"| adj={n_adj_ok}  raw_fallback={n_fallback} ({pct_fallback:.1f}%)")

        if pct_fallback > 25:
            print(f"  [WARNING] >25% of tickers used raw close for IVOL signal. "
                  f"Adj layer may not be fully populated. Results may be split-contaminated.")

        adj_coverage.append({
            "rebalance_date": t,
            "n_universe":     len(univ),
            "n_computed":     n_computed,
            "n_valid_ivol":   len(ivol_df),
            "n_adj":          n_adj_ok,
            "n_fallback":     n_fallback,
            "pct_fallback":   pct_fallback,
        })

        # ── 3. Merge signals into universe; compute hold-period returns ───────
        univ = univ.merge(ivol_df, on="ticker", how="left")

        try:
            returns_df = compute_returns(univ, hold_start, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERROR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        n_valid    = returns_df["raw_return"].notna().sum()

        universe_audit.append({
            "rebalance_date":  t,
            "hold_end":        hold_end,
            "n_universe":      len(returns_df),
            "n_with_signal":   int(returns_df["ivol_signal"].notna().sum()),
            "n_delisted":      n_delisted,
            "n_valid_returns": n_valid,
        })

        # ── 4. Portfolio ──────────────────────────────────────────────────────
        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short) = build_portfolio(returns_df)
        if not longs:
            print(f"  [SKIP] Insufficient names (need 40, have {n_valid} with valid signal+return).")
            continue

        # ── 5. Leg returns (net of costs), beta-neutral scaled ────────────────
        # _leg_return gives per-$1-of-leg-exposure net return.
        # Multiply by leg scales (sum = 2 = gross exposure) for beta-neutral book.
        long_ret  = _leg_return(returns_df, longs,  lw, is_short=False)
        short_ret = _leg_return(returns_df, shorts, sw, is_short=True)
        ls_ret    = long_scale * long_ret + short_scale * short_ret

        # ── 6. Market proxy: equal-weight full universe ───────────────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if len(valid_all) > 0 else np.nan

        # ── 7. IC: Spearman(-ivol_signal, realized_return) ───────────────────
        # Using negated signal so IC > 0 means low-IVOL outperforms high-IVOL.
        valid_sig = returns_df.dropna(subset=["ivol_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(-valid_sig["ivol_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        record = {
            "rebalance_date":   t,
            "hold_end":         hold_end,
            "n_universe":       len(returns_df),
            "n_with_signal":    int(returns_df["ivol_signal"].notna().sum()),
            "n_delisted":       n_delisted,
            "long_ret":         long_ret,
            "short_ret":        short_ret,
            "ls_ret":           ls_ret,
            "long_scale":       long_scale,
            "short_scale":      short_scale,
            "mean_beta_long":   mean_beta_long,
            "mean_beta_short":  mean_beta_short,
            "mkt_ret":          mkt_ret,
            "ic":               ic,
            "n_long":           len(longs),
            "n_short":          len(shorts),
        }
        period_records.append(record)

        print(f"  [Result] L/S={ls_ret:.3%}  "
              f"Long={long_ret:.3%}×{long_scale:.2f}  Short={short_ret:.3%}×{short_scale:.2f}  "
              f"β_L={mean_beta_long:.2f}  β_S={mean_beta_short:.2f}  "
              f"Mkt={mkt_ret:.3%}  IC={ic:.3f}  n={len(longs)}/{len(shorts)}")

    elapsed = time.time() - t_total_start
    return {
        "period_records": period_records,
        "universe_audit": universe_audit,
        "adj_coverage":   adj_coverage,
        "elapsed_s":      elapsed,
    }


# =============================================================================
# SECTION 12: Metrics
# =============================================================================

def compute_metrics(returns: pd.Series) -> dict:
    s = returns.dropna()
    n = len(s)
    if n == 0:
        return {k: np.nan for k in
                ["total_ret", "ann_ret", "sharpe", "calmar", "max_dd",
                 "t_stat", "win_rate", "n", "drawdowns"]}

    mean_q   = s.mean()
    std_q    = s.std(ddof=1) if n > 1 else 0.0

    cum_ret  = (1 + s).prod() - 1
    ann_ret  = (1 + mean_q) ** 4 - 1
    sharpe   = (mean_q / std_q) * np.sqrt(4) if std_q > 0 else np.nan
    t_stat   = mean_q / (std_q / np.sqrt(n))  if std_q > 0 else np.nan
    win_rate = (s > 0).mean()

    cum       = (1 + s).cumprod()
    roll_max  = cum.cummax()
    drawdowns = (cum - roll_max) / roll_max
    max_dd    = float(drawdowns.min())
    calmar    = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "total_ret": cum_ret,
        "ann_ret":   ann_ret,
        "sharpe":    sharpe,
        "calmar":    calmar,
        "max_dd":    max_dd,
        "t_stat":    t_stat,
        "win_rate":  win_rate,
        "n":         n,
        "drawdowns": drawdowns,
    }


def compute_market_beta(ls_returns: pd.Series, mkt_returns: pd.Series) -> dict:
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < 4:
        return {"beta": np.nan, "alpha_q": np.nan, "r_squared": np.nan, "beta_t": np.nan}
    slope, intercept, r_value, _, se = stats.linregress(combined["mkt"], combined["ls"])
    return {
        "beta":      slope,
        "alpha_q":   intercept,
        "r_squared": r_value ** 2,
        "beta_t":    slope / se if se > 0 else np.nan,
    }


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    """Bailey & Lopez de Prado (2014) DSR threshold (quarterly Sharpe units)."""
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 13: Output & registry
# =============================================================================

def print_results(results: dict) -> None:
    records  = results["period_records"]
    audit    = results["universe_audit"]
    coverage = results["adj_coverage"]
    elapsed  = results["elapsed_s"]

    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df   = pd.DataFrame(records)
    aud  = pd.DataFrame(audit)
    cov  = pd.DataFrame(coverage)

    ls_series  = df["ls_ret"]
    mkt_series = df["mkt_ret"]
    ic_series  = df["ic"].dropna()
    m          = compute_metrics(ls_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    obs_sr_q   = m["sharpe"] / np.sqrt(4) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (
        (obs_sr_q > dsr_thr)
        if (not np.isnan(obs_sr_q) and not np.isnan(dsr_thr))
        else False
    )

    mean_ic  = float(ic_series.mean())    if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    sep = "=" * 72

    # ── Adj-layer coverage ───────────────────────────────────────────────────
    total_fallback   = int(cov["n_fallback"].sum())
    total_computed   = int(cov["n_computed"].sum())
    overall_pct_fall = 100 * total_fallback / max(total_computed, 1)

    print(f"\n{sep}")
    print("SIGNAL LAYER — ADJ CLOSE COVERAGE CHECK")
    print(sep)
    print(f"  Tickers using adjClose (split-adjusted): "
          f"{total_computed - total_fallback:,} / {total_computed:,}")
    print(f"  Tickers using raw close (fallback)      : "
          f"{total_fallback:,} / {total_computed:,}  ({overall_pct_fall:.1f}%)")
    if overall_pct_fall > 10:
        print(f"\n  *** WARNING: {overall_pct_fall:.1f}% of ticker-period IVOL values used raw close.")
        print(f"  *** Stock splits may have inflated IVOL scores, biasing the short leg.")
        print(f"  *** Re-run after populating adjclose_all.pkl with Tiingo API.")
    else:
        print(f"\n  Adj-layer coverage is adequate (<10% fallback). Signal integrity confirmed.")

    print(f"\n  Per-period adj coverage:")
    print(f"    {'Date':<12} {'Universe':>9} {'Computed':>9} {'AdjOK':>7} {'Fallback':>9} {'Pct%':>7}")
    print(f"    {'-'*12} {'-'*9} {'-'*9} {'-'*7} {'-'*9} {'-'*7}")
    for _, r in cov.iterrows():
        print(f"    {str(r['rebalance_date'].date()):<12} "
              f"{int(r['n_universe']):>9} {int(r['n_computed']):>9} "
              f"{int(r['n_adj']):>7} {int(r['n_fallback']):>9} "
              f"{r['pct_fallback']:>6.1f}%")

    # ── Survivorship audit ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SURVIVORSHIP AUDIT — In-Sample 2015-Q1 to 2020-Q4")
    print(sep)
    total_name_periods = int(aud["n_universe"].sum())
    total_delistings   = int(aud["n_delisted"].sum())
    first_n            = int(aud.iloc[0]["n_universe"])
    print(f"  Total name-periods across all 24 rebalances : {total_name_periods:,}")
    print(f"  Total delisting events retained             : {total_delistings}")
    print(f"  Names at 2015-Q1 rebalance                 : {first_n}")
    print(f"  Delistings retained (returns to last close) : {total_delistings}  (NOT dropped)")
    print(f"\n  No-look-ahead confirmations:")
    print(f"    IVOL window uses adjClose for dates ≤ t only")
    print(f"    Market proxy built from universe with adjClose ≤ t")
    print(f"    Market cap: SEC filing_date ≤ t, price last close ≤ t")
    print(f"    Entry price: first close on or after t; returns measured t→t+1Q")
    print(f"\n  Per-period breakdown:")
    print(f"    {'Date':<12} {'Universe':>9} {'w/Signal':>9} {'Delisted':>9} {'ValidRet':>9}")
    print(f"    {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for _, r in aud.iterrows():
        print(f"    {str(r['rebalance_date'].date()):<12} "
              f"{int(r['n_universe']):>9} {int(r['n_with_signal']):>9} "
              f"{int(r['n_delisted']):>9} {int(r['n_valid_returns']):>9}")

    # ── Portfolio construction summary ───────────────────────────────────────
    mean_long_scale    = float(df["long_scale"].mean())   if "long_scale"    in df else np.nan
    mean_short_scale   = float(df["short_scale"].mean())  if "short_scale"   in df else np.nan
    mean_bl            = float(df["mean_beta_long"].mean())  if "mean_beta_long"  in df else np.nan
    mean_bs            = float(df["mean_beta_short"].mean()) if "mean_beta_short" in df else np.nan

    # ── Performance summary ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("S-IVOL: LOW IDIOSYNCRATIC VOLATILITY — IN-SAMPLE RESULTS  (v2 beta-neutral)")
    print("In-sample period: 2015-01-01 → 2021-01-01  |  Signal: -IVOL (long lowest, short highest)")
    print("Construction: BETA-NEUTRAL  (long_$ × β_L = short_$ × β_S, gross = 2×)")
    print(sep)
    print(f"\n  {'Metric':<42} {'Value':>15}")
    print(f"  {'-'*42} {'-'*15}")
    print(f"  {'Total return (cumulative)':<42} {m['total_ret']:>14.2%}")
    print(f"  {'Annualized return':<42} {m['ann_ret']:>14.2%}")
    print(f"  {'Sharpe ratio (annualized)':<42} {m['sharpe']:>14.3f}")
    print(f"  {'CALMAR ratio':<42} {m['calmar']:>14.3f}")
    print(f"  {'Max drawdown':<42} {m['max_dd']:>14.2%}")
    print(f"  {'t-stat (mean quarterly return)':<42} {m['t_stat']:>14.3f}")
    print(f"  {'Market beta (vs EW universe)':<42} {beta_stats['beta']:>14.3f}")
    print(f"  {'  beta t-stat':<42} {beta_stats['beta_t']:>14.3f}")
    print(f"  {'  R-squared':<42} {beta_stats['r_squared']:>14.3f}")
    print(f"  {'Mean gross long exposure ($)':<42} {mean_long_scale:>14.3f}")
    print(f"  {'Mean gross short exposure ($)':<42} {mean_short_scale:>14.3f}")
    print(f"  {'Mean ex-ante β long leg':<42} {mean_bl:>14.3f}")
    print(f"  {'Mean ex-ante β short leg':<42} {mean_bs:>14.3f}")
    print(f"  {'Mean IC (Spearman, low-IVOL vs ret)':<42} {mean_ic:>14.4f}")
    print(f"  {'IC t-stat':<42} {ic_tstat:>14.3f}")
    print(f"  {'Win rate':<42} {m['win_rate']:>14.1%}")
    print(f"  {'Periods (N)':<42} {m['n']:>14d}")
    print(f"  {'AdjClose fallback rate':<42} {overall_pct_fall:>13.1f}%")
    print(f"\n  --- DSR Check (N_trials={N_TRIALS}, IVOL/MAX counted as one family) ---")
    print(f"  {'Observed quarterly Sharpe':<42} {obs_sr_q:>14.4f}")
    print(f"  {'DSR threshold (quarterly Sharpe)':<42} {dsr_thr:>14.4f}")
    print(f"  {'Clears DSR':<42} {'YES' if clears_dsr else 'NO':>14}")
    print(f"\n  Raw quarterly returns (compute α vs T-bill Rf externally):")
    print(f"  Mean quarterly L/S return : {ls_series.mean():.4%}")
    print(f"  Std quarterly L/S return  : {ls_series.std(ddof=1):.4%}")

    # ── Quarterly returns log ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("QUARTERLY RETURNS LOG  (IC > 0 means low-IVOL outperformed high-IVOL)")
    print("LScale/SScale = gross dollar allocation per $1 of capital (sum ≈ 2)")
    print(sep)
    print(f"  {'Date':<12} {'L/S Ret':>8} {'Long':>8} {'LScale':>7} {'Short':>8} {'SScale':>7} "
          f"{'Mkt':>8} {'IC':>6}  {'nL':>4} {'nS':>4} {'Dlist':>5}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*7} "
          f"{'-'*8} {'-'*6}  {'-'*4} {'-'*4} {'-'*5}")
    for _, r in df.iterrows():
        ls  = r.get("long_scale",  1.0)
        ss  = r.get("short_scale", 1.0)
        print(f"  {str(r['rebalance_date'].date()):<12} "
              f"{r['ls_ret']:>7.2%} {r['long_ret']:>7.2%} {ls:>6.3f}× "
              f"{r['short_ret']:>7.2%} {ss:>6.3f}× "
              f"{r['mkt_ret']:>7.2%} {r['ic']:>6.3f}  "
              f"{int(r['n_long']):>4} {int(r['n_short']):>4} {int(r['n_delisted']):>5}")

    # ── Drawdown series ──────────────────────────────────────────────────────
    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (quarterly):")
    for i, (date, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        print(f"    {str(date.date()):<12}  {val:>7.2%}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def save_results(results: dict) -> None:
    records  = results["period_records"]
    audit    = results["universe_audit"]
    coverage = results["adj_coverage"]
    if not records:
        return

    pd.DataFrame(records).to_csv(RESULTS / "ivol_period_returns.csv", index=False)
    pd.DataFrame(audit).to_csv(RESULTS / "ivol_universe_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "ivol_adj_coverage.csv", index=False)
    print(f"\n[Saved] Results → {RESULTS}/")


def log_registry(results: dict, repair_stats: dict | None = None) -> None:
    records = results["period_records"]
    if not records:
        return

    df         = pd.DataFrame(records)
    ls_series  = df["ls_ret"]
    mkt_series = df["mkt_ret"]
    ic_series  = df["ic"].dropna()
    m          = compute_metrics(ls_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    obs_sr_q   = m["sharpe"] / np.sqrt(4) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (
        bool(obs_sr_q > dsr_thr)
        if not np.isnan(obs_sr_q) and not np.isnan(dsr_thr)
        else False
    )
    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    mean_long_scale  = float(df["long_scale"].mean())  if "long_scale"  in df else np.nan
    mean_short_scale = float(df["short_scale"].mean()) if "short_scale" in df else np.nan
    mean_bl = float(df["mean_beta_long"].mean())  if "mean_beta_long"  in df else np.nan
    mean_bs = float(df["mean_beta_short"].mean()) if "mean_beta_short" in df else np.nan

    repair_note = ""
    if repair_stats:
        repair_note = (
            f" AdjClose repair: {repair_stats.get('recovered',0)} tickers recovered, "
            f"{repair_stats.get('still_empty',0)} genuinely empty after retry."
        )

    row = {
        "timestamp":         datetime.now().isoformat(),
        "study":             "ivol_S-IVOL",
        "hypothesis":        (
            "Low-IVOL small/mid-cap stocks earn positive abnormal returns relative to "
            "high-IVOL lottery stocks. Lottery-like stocks overpriced by undiversified investors; "
            "effect strongest in low-analyst-coverage names. "
            "IVOL + MAX strategy treated as one family for DSR."
        ),
        "data_source":       "Tiingo (adjClose for IVOL signal, raw close for returns) + SEC XBRL (PIT shares)",
        "status":            "completed_in_sample",
        "trial_number":      7,
        "n_trials_dsr":      N_TRIALS,
        "ann_return":        m["ann_ret"],
        "sharpe":            m["sharpe"],
        "calmar":            m["calmar"],
        "max_drawdown":      m["max_dd"],
        "t_stat":            m["t_stat"],
        "market_beta":       beta_stats["beta"],
        "market_beta_tstat": beta_stats["beta_t"],
        "mean_long_scale":   mean_long_scale,
        "mean_short_scale":  mean_short_scale,
        "mean_beta_long":    mean_bl,
        "mean_beta_short":   mean_bs,
        "mean_ic":           mean_ic,
        "ic_t_stat":         ic_tstat,
        "win_rate":          m["win_rate"],
        "n_periods":         m["n"],
        "obs_sr_q":          obs_sr_q,
        "dsr_threshold":     dsr_thr,
        "clears_dsr":        clears_dsr,
        "notes":             (
            "CORRECTED RUN v2 — replaces prior trial #7 entry (same hypothesis, not new trial). "
            "Fix 1: Beta-neutral leg sizing (long_$ × β_L = short_$ × β_S, gross=2×); "
            "beta reused from 60-day OLS regression, no extra computation. "
            "Intentional deviation from 50/50 dollar-neutral: IVOL legs have structurally "
            "asymmetric beta (low-IVOL≈low-β, high-IVOL≈high-β); 50/50 was a net short market bet. "
            "Fix 2: .EMPTY sentinels retried with 30s timeout (prior ConnectTimeout false negatives)."
            + repair_note
        ),
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    # Replace prior ivol_S-IVOL entry (corrected run, not a new trial)
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "ivol_S-IVOL"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] Trial #7 (ivol_S-IVOL) replaced in registry → {REGISTRY}")


# =============================================================================
# SECTION 14: Main
# =============================================================================

def main() -> None:
    print("=" * 72)
    print("S-IVOL: Low Idiosyncratic Volatility  (v2 — beta-neutral, adj repaired)")
    print("In-sample: 2015-01-01 → 2021-01-01  |  24 quarterly periods")
    print("Signal: IVOL via single-factor market-model, 60-trading-day window")
    print("Signal layer: adjClose (split+div adjusted) — no split contamination")
    print("Construction: BETA-NEUTRAL  (long_$ × β_L = short_$ × β_S)")
    print("=" * 72)

    if not TIINGO_KEY:
        print("\n[WARNING] TIINGO_API_KEY not set. "
              "AdjClose layer disabled — all signals will use raw close.")

    # ── Step 1: Tiingo tickers ──────────────────────────────────────────────
    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    # ── Step 2: SEC prefilter ──────────────────────────────────────────────
    print("\n[Step 2] Loading SEC pre-filter (mgmt_pit cache)...")
    prefilter_df = load_prefilter()

    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    # ── Step 3: Preload price caches (raw + adj) ───────────────────────────
    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle — signal layer...")
    preload_all_adj_prices(survivor_tickers)

    # ── Step 3r: Repair .EMPTY sentinels (ConnectTimeout false negatives) ──
    print(f"\n[Step 3r] Repairing .EMPTY sentinels (30s timeout, 3 attempts each)...")
    repair_stats = repair_adj_sentinels(survivor_tickers)

    # ── Coverage gate ───────────────────────────────────────────────────────
    adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    pct_adj    = 100 * adj_loaded / max(len(survivor_tickers), 1)
    pct_miss   = 100 - pct_adj
    print(f"\n[Step 3r] Post-repair adjClose coverage: "
          f"{adj_loaded:,}/{len(survivor_tickers):,} ({pct_adj:.1f}%)  "
          f"| estimated fallback rate: {pct_miss:.1f}%")
    if pct_miss > 10:
        print(f"\n{'!'*72}")
        print(f"  COVERAGE GATE: {pct_miss:.1f}% estimated fallback (>{10}% threshold).")
        print(f"  AdjClose coverage not sufficient. Backtest would not be trustworthy.")
        print(f"  Repair recovered {repair_stats['recovered']:,} of "
              f"{repair_stats['retried']:,} sentinels; {repair_stats['still_empty']:,} genuine empties.")
        print(f"  STOPPING before backtest. Investigate adjClose coverage and re-run.")
        print(f"{'!'*72}")
        return
    else:
        print(f"  Coverage adequate (<10% est. fallback). Proceeding to backtest.")

    # ── Step 3c: Split-fix validation (adj layer sanity check) ────────────
    print("\n[Step 3c] Running split-contamination validation...")
    validate_split_fix()

    # ── Step 4: SEC facts ──────────────────────────────────────────────────
    print(f"\n[Step 4] Preloading SEC facts ({len(survivor_ciks):,} CIKs)...")
    preload_sec_facts(survivor_ciks)

    # ── Step 5: Run backtest ───────────────────────────────────────────────
    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    # ── Step 6: Print + save ───────────────────────────────────────────────
    print_results(results)
    save_results(results)
    log_registry(results, repair_stats=repair_stats)


if __name__ == "__main__":
    main()
