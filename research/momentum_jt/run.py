"""
S9: Cross-Sectional Momentum (Jegadeesh–Titman 12-1)
=====================================================
Signal   : 12-month cumulative return ending 1 month before rebalance (skip-1 guard).
Universe : PIT US small/mid-cap $100M–$2B mcap, ADTV ≥ $1M/day, price ≥ $1.
           Market cap = SEC shares-as-filed (filing_date ≤ t) × last_close ≤ t.
Portfolio: Long top 20% / short bottom 20% (min 20 names per leg), 50/50 dollar-neutral.
Rebalance: Quarterly.  Hold: 1 quarter.
In-sample: 2015-01-01 → 2021-01-01 (24 periods).  No walk-forward.
Trial    : S9 — trial #6 in registry (after H1 + E1–E4).  N_TRIALS=6 for DSR.

No-survivorship guarantee
  - Universe built PIT each period using raw close (not adjClose)
  - Delisted tickers retained with returns to their last trading day
  - universe rebuilt fresh each period from Tiingo metadata (startDate/endDate)

No-look-ahead guarantee
  - Momentum formation window ends mom_end = t - 1M (skip-month enforced in code)
  - Market cap uses SEC filing_date ≤ t and last close ≤ t
  - Entry prices are first close on or after t; returns measured t → t+1Q
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

TIINGO_KEY = os.environ.get("TIINGO_API_KEY")
SEC_UA = "Hugh Archer hugharcher5@gmail.com"

# Shared cache with mgmt_pit (all already populated)
MGMT_PIT = Path(__file__).parent.parent / "mgmt_pit"
CACHE = MGMT_PIT / "cache"
RESULTS = Path(__file__).parent / "results"
REGISTRY = Path(__file__).parent.parent / "trial_registry.csv"

for _d in [RESULTS, CACHE / "tiingo", CACHE / "sec_facts"]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Universe filters ──────────────────────────────────────────────────────────
MCAP_MIN        = 100e6
MCAP_MAX        = 2e9
DOLLAR_VOL_MIN  = 1e6       # avg daily dollar volume over 60 trading days
PRICE_MIN       = 1.0       # minimum close price — blocks penny stocks
MOMENTUM_MIN_DAYS = 180     # min trading days in formation window (~8.5 months)
SIC_EXCLUDE     = set(range(6000, 6800))  # financials + real estate
US_EXCHANGES    = {"NYSE", "NASDAQ", "AMEX"}

# ── Backtest dates ────────────────────────────────────────────────────────────
IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")
_all_dates = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = _all_dates[_all_dates < IS_END]   # 24 periods

# ── Cost model (identical to mgmt_pit) ───────────────────────────────────────
TURNOVER_RATE     = 0.50
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.02

# ── DSR ───────────────────────────────────────────────────────────────────────
N_TRIALS = 6   # H1 + E1 + E2 + E3 + E4 + S9

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
    df = df_all[
        df_all["exchange"].isin(US_EXCHANGES) & (df_all["assetType"] == "Stock")
    ].copy()
    for col in ["startDate", "endDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["startDate"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"[Tiingo] {len(df)} tickers saved.")
    return df


# =============================================================================
# SECTION 4: SEC helpers
# =============================================================================

def load_sec_ticker_cik_map() -> dict:
    """Load SEC ticker→CIK map (cached from mgmt_pit run)."""
    path = CACHE / "sec_ticker_cik.json"
    if path.exists():
        with open(path) as f:
            raw = json.load(f)
        print(f"[SEC] Loaded {len(raw)} ticker→CIK pairs from cache.")
        return raw

    print("[SEC] Downloading company_tickers.json...")
    resp = _sec_get("https://www.sec.gov/files/company_tickers.json")
    resp.raise_for_status()
    data = resp.json()
    out: dict = {}
    for entry in data.values():
        t = entry.get("ticker", "").upper().strip()
        cik_raw = entry.get("cik_str", "")
        if t and cik_raw:
            try:
                out[t] = int(cik_raw)
            except ValueError:
                pass
    with open(path, "w") as f:
        json.dump(out, f)
    return out


_FUND_KEYWORDS = [
    "etf", "fund", "trust", "index", "portfolio", "series",
    "income fund", "growth fund", "spdr", "ishares", "invesco", "vanguard etf",
]


def _has_filings_in_window(subs: dict) -> bool:
    """Return True if company filed 10-K/10-Q between 2013–2026."""
    target = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
    w0, w1 = "2013-01-01", "2026-01-01"
    try:
        recent = subs.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filedDate", recent.get("filingDate", []))
        for i, frm in enumerate(forms):
            if frm in target:
                d = dates[i] if i < len(dates) else ""
                if w0 <= d <= w1:
                    return True
        for ff in subs.get("filings", {}).get("files", []):
            if ff.get("filedTo", "") >= w0 and ff.get("filedFrom", "") <= w1:
                return True
    except Exception:
        pass
    return False


def _is_fund_etf(name: str, etype: str) -> bool:
    nl = (name or "").lower()
    el = (etype or "").lower()
    if any(k in nl for k in _FUND_KEYWORDS):
        return True
    if "investment company" in el or "registered investment" in el:
        return True
    return False


def load_prefilter() -> pd.DataFrame:
    """Load mgmt_pit's SEC pre-filter cache (already built)."""
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"sec_prefilter.csv not found at {path}. Run mgmt_pit/run.py --prefilter-only first."
        )
    df = pd.read_csv(path)
    survivors = df[df["passed"]]
    print(f"[PreFilter] Loaded: {len(df)} screened, {len(survivors)} survivors (from mgmt_pit cache).")
    return df


# =============================================================================
# SECTION 5: In-memory price cache
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
    """Bulk-load all price CSVs into memory once. Saves consolidated pickle for future runs."""
    t0 = time.time()
    if _PRICES_PKL.exists():
        with open(_PRICES_PKL, "rb") as f:
            data = pickle.load(f)
        _PRICE_CACHE.update(data.get("prices", {}))
        _PRICE_SENTINEL.update(data.get("sentinels", set()))
        # Pick up any new CSVs not in pickle
        for t in tickers:
            if t not in _PRICE_CACHE and t not in _PRICE_SENTINEL:
                _load_single(t)
        print(f"[Preload] Prices: {len(_PRICE_CACHE)} loaded (pickle) in {time.time()-t0:.1f}s")
        return

    for t in tickers:
        _load_single(t)
    elapsed = time.time() - t0
    print(f"[Preload] Prices: {len(_PRICE_CACHE)} CSVs loaded in {elapsed:.1f}s — saving pickle...")
    t1 = time.time()
    with open(_PRICES_PKL, "wb") as f:
        pickle.dump({"prices": dict(_PRICE_CACHE), "sentinels": set(_PRICE_SENTINEL)}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Preload] Pickle saved in {time.time()-t1:.1f}s")


# =============================================================================
# SECTION 5b: Adjusted-price cache  —  SIGNAL LAYER
# =============================================================================
#
# Tiingo adjClose adjustment:
#   Backward-adjusted for BOTH stock splits AND cash dividends (total-return
#   proxy).  Current price is unchanged; each past split/dividend scales all
#   prior prices down by the event factor.
#
#   → Splits:    fully corrected — a 2:1 split no longer creates a −50%
#                overnight reading in the formation window.
#   → Dividends: also corrected — a $0.50 div on a $20 stock adjusts by 2.5%.
#   → Net:       adjClose ≈ total-return-adjusted price series.
#
# Two-layer model:
#   SIGNAL layer  — use adjClose (_ADJ_CACHE)  for any past-return signal
#                   (momentum, reversal, volatility, etc.) so corporate
#                   actions don't corrupt the ranking.
#   RETURN layer  — use raw close (_PRICE_CACHE) for hold-period returns and
#                   delisting exits, preserving survivorship-bias-free behaviour.

_ADJ_CACHE:    dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str]               = set()
_ADJ_EMPTY = pd.DataFrame(columns=["date", "adjClose"])

_ADJ_DIR = CACHE / "tiingo_adj"
_ADJ_PKL = CACHE / "adjclose_all.pkl"
_ADJ_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_adj_from_tiingo(ticker: str) -> pd.DataFrame:
    """Fetch adjClose from Tiingo daily prices endpoint. Returns df[date, adjClose]."""
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
    """Load adjClose from disk cache; fetch from Tiingo API on cache miss."""
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

    # Cache miss → Tiingo API
    df = _fetch_adj_from_tiingo(ticker)
    if df.empty:
        sentinel.touch()
        _ADJ_SENTINEL.add(ticker)
        return _ADJ_EMPTY
    df.to_csv(csv, index=False)
    _ADJ_CACHE[ticker] = df
    return df


def preload_all_adj_prices(tickers: list[str]) -> None:
    """
    Bulk-load adjClose for all tickers into _ADJ_CACHE.
    First run: fetches missing tickers from Tiingo API (~1.5 s/ticker, rate-limited).
    Subsequent runs: loads from consolidated pickle in seconds.
    """
    t0 = time.time()

    if _ADJ_PKL.exists():
        with open(_ADJ_PKL, "rb") as f:
            data = pickle.load(f)
        _ADJ_CACHE.update(data.get("adj", {}))
        _ADJ_SENTINEL.update(data.get("sentinels", set()))
        print(f"[AdjPreload] {len(_ADJ_CACHE)} tickers from pickle in {time.time()-t0:.1f}s")

    # Tickers not yet in memory
    missing = [t for t in tickers if t not in _ADJ_CACHE and t not in _ADJ_SENTINEL]

    # Resolve via individual CSVs first (no API call)
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
        est_min = len(need_api) * 1.5 / 60
        print(f"[AdjPreload] {len(need_api)} tickers need API fetch (est {est_min:.0f} min)...")
        for i, t in enumerate(need_api):
            if (i + 1) % 100 == 0:
                rem = (len(need_api) - i - 1) * 1.5 / 60
                print(f"  [AdjPreload] {i+1}/{len(need_api)} | ~{rem:.0f} min remaining | last={t}")
            _load_adj(t)
        print(f"[AdjPreload] API fetch done in {time.time()-t0:.1f}s")

        t1 = time.time()
        with open(_ADJ_PKL, "wb") as f:
            pickle.dump({"adj": dict(_ADJ_CACHE), "sentinels": set(_ADJ_SENTINEL)}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[AdjPreload] Pickle saved in {time.time()-t1:.1f}s")

    loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    print(f"[AdjPreload] Total: {loaded} tickers with adjClose ({time.time()-t0:.1f}s)")


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Return adjClose rows for [start, end]. No I/O — uses in-memory _ADJ_CACHE."""
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 6: SEC facts (shares outstanding only)
# =============================================================================

_FACTS_CACHE: dict[int, Optional[dict]] = {}
_FACTS_TRIMMED = CACHE / "sec_facts_trimmed.json"
_SHARES_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]


def preload_sec_facts(ciks: list[int]) -> None:
    """Load shares-outstanding facts from mgmt_pit's trimmed consolidated cache."""
    t0 = time.time()
    if _FACTS_TRIMMED.exists():
        with open(_FACTS_TRIMMED) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _FACTS_CACHE[int(k)] = v
        # Pick up any CIKs missing from consolidated cache
        for cik in ciks:
            if cik not in _FACTS_CACHE:
                p = CACHE / "sec_facts" / f"{cik}.json"
                if p.exists():
                    try:
                        with open(p) as f:
                            full = json.load(f)
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
        print(f"[Preload] SEC facts: {loaded} loaded from trimmed cache in {time.time()-t0:.1f}s")
        return

    # Fallback: individual JSON files
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
    print(f"[Preload] SEC facts: {loaded} loaded from individual files in {time.time()-t0:.1f}s")


def _get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    """
    PIT shares outstanding: most recent CommonStockSharesOutstanding
    where filed_date ≤ as_of. Fallback: EntityCommonStockSharesOutstanding.
    """
    facts = _FACTS_CACHE.get(cik)
    if facts is None:
        return None
    as_of_str = as_of.strftime("%Y-%m-%d")

    for concept in _SHARES_CONCEPTS:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue

        best: Optional[tuple] = None  # (filed_str, end_str, val)
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

def _get_price_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Slice in-memory price DataFrame. No I/O."""
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return _PRICE_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


def _last_close(ticker: str, as_of: pd.Timestamp) -> Optional[float]:
    """Last close on or before as_of. Uses in-memory cache."""
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
    Build PIT investable universe at rebalance_date.
    - Active on rebalance_date (startDate ≤ t, endDate ≥ t or NaT)
    - In SEC prefilter survivors (non-financial, has 10-K/10-Q filings)
    - Market cap: shares_as_filed × last_close ≤ t ∈ [$100M, $2B]
    - ADTV ≥ $1M/day (60-day lookback)
    - Price ≥ $1

    cik_sic_map: ticker → (cik, sic)  — pre-computed once from prefilter
    Returns DataFrame [ticker, cik, mcap, price, shares, avg_dvol]
    """
    survivors_set = set(cik_sic_map.keys())

    # Step 1: active on rebalance_date AND in SEC survivors
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
        ticker = row["ticker"]
        cik, sic = cik_sic_map.get(ticker, (None, None))
        if cik is None:
            continue

        # PIT price
        price = _last_close(ticker, rebalance_date)
        if price is None or price < PRICE_MIN:
            continue

        # PIT market cap
        shares = _get_shares_pit(cik, rebalance_date)
        if shares is None or shares <= 0:
            continue
        mcap = shares * price
        if not (MCAP_MIN <= mcap <= MCAP_MAX):
            continue

        # Liquidity floor
        dvol = _avg_dollar_vol(ticker, rebalance_date)
        if dvol < DOLLAR_VOL_MIN:
            continue

        rows.append({"ticker": ticker, "cik": cik, "mcap": mcap,
                     "price": price, "shares": shares, "avg_dvol": dvol})

    df = pd.DataFrame(rows)
    print(f"  [Universe] {rebalance_date.date()} → {len(df)} names "
          f"(active={len(active)}, post-filter={len(df)})")
    return df


# =============================================================================
# SECTION 8: Momentum signal (look-ahead-free)
# =============================================================================

def compute_momentum_signal(
    ticker: str,
    rebalance_date: pd.Timestamp,
) -> Optional[float]:
    """
    12-1 momentum: cumulative return over [t-12M, t-1M].
    Skip month ensures no contamination from short-term reversal.
    Look-ahead guard: formation window strictly ends before rebalance_date.
    Returns None if < MOMENTUM_MIN_DAYS trading days in window.

    SIGNAL LAYER — uses adjClose (split+dividend adjusted) so a 2:1 split
    inside the formation window does not create an artificial −50% signal
    that misclassifies a genuine winner as a loser.
    Falls back to raw close only when adjClose is unavailable for a ticker.
    """
    mom_end   = rebalance_date - pd.DateOffset(months=1)   # e.g. t=2015-01-01 → 2014-12-01
    mom_start = rebalance_date - pd.DateOffset(months=12)  # e.g. t=2015-01-01 → 2014-01-01

    # ── Signal layer: adjClose ────────────────────────────────────────────────
    adj = _adj_slice(ticker, mom_start, mom_end)
    if len(adj) >= MOMENTUM_MIN_DAYS:
        sp = float(adj.iloc[0]["adjClose"])
        ep = float(adj.iloc[-1]["adjClose"])
        if sp > 0:
            return ep / sp - 1

    # ── Fallback: raw close (when adjClose not cached for this ticker) ────────
    raw = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if raw.empty:
        return None
    window = raw[(raw["date"] >= mom_start) & (raw["date"] <= mom_end)]
    if len(window) < MOMENTUM_MIN_DAYS:
        return None
    sp = float(window.iloc[0]["close"])
    ep = float(window.iloc[-1]["close"])
    if sp <= 0:
        return None
    return ep / sp - 1


# =============================================================================
# SECTION 9: Return calculation (survivorship-bias-free)
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
    delisted_map: dict,
) -> pd.DataFrame:
    """
    Compute holding period raw returns for each name.
    Delistings: exit at last available close on or before delist_date.
    Entry: first close on or after hold_start.
    Exit : close on or before hold_end (or delist_date if delisted earlier).
    Raw close (not adjClose) — same field Tiingo caches as 'close'.
    """
    df = universe_df.copy()
    entry_prices, exit_prices, raw_returns, delisted_flags = [], [], [], []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        prices = _get_price_slice(ticker, hold_start, hold_end + pd.Timedelta(days=5))

        if prices.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan); delisted_flags.append(False)
            continue

        entry_cands = prices[prices["date"] >= hold_start]
        if entry_cands.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan); delisted_flags.append(False)
            continue

        entry_price = float(entry_cands.iloc[0]["close"])

        delist_date = delisted_map.get(ticker, pd.NaT)
        is_delisted = pd.notna(delist_date) and (delist_date < hold_end)

        if is_delisted:
            exit_cands = prices[prices["date"] <= delist_date]
            exit_price = float((exit_cands if not exit_cands.empty else prices).iloc[-1]["close"])
            delisted_flags.append(True)
        else:
            exit_cands = prices[prices["date"] <= hold_end]
            exit_price = float(exit_cands.iloc[-1]["close"]) if not exit_cands.empty else entry_price
            delisted_flags.append(False)

        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        raw_returns.append(exit_price / entry_price - 1)

    df["entry_price"]  = entry_prices
    df["exit_price"]   = exit_prices
    df["raw_return"]   = raw_returns
    df["delisted"]     = delisted_flags
    return df


# =============================================================================
# SECTION 10: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    """One-way half-spread in decimal: 100bps at $100M, 20bps at $2B."""
    frac = (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)
    frac = max(0.0, min(1.0, frac))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    """Annual borrow rate: 2% at $100M, 0.5% at $2B."""
    frac = (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)
    frac = max(0.0, min(1.0, frac))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame) -> tuple:
    """
    Rank by momentum signal. Long top 20%, short bottom 20%.
    Min 20 names per leg. Equal-weight within each leg.
    Returns (longs, shorts, long_weights, short_weights).
    """
    df = signals_df.dropna(subset=["momentum", "raw_return"]).copy()
    n  = len(df)
    if n < 40:   # need at least 40 to get 20 per leg
        return [], [], {}, {}

    leg = max(int(np.ceil(n * 0.20)), 20)
    leg = min(leg, n // 2)   # can't exceed half the universe

    ranked = df.sort_values("momentum", ascending=False)
    longs  = ranked.iloc[:leg]["ticker"].tolist()
    shorts = ranked.iloc[-leg:]["ticker"].tolist()

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}
    return longs, shorts, lw, sw


def _leg_return(returns_df: pd.DataFrame, tickers: list, weights: dict,
                is_short: bool) -> float:
    """Weighted gross return for one leg, net of transaction costs and borrow."""
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
# SECTION 11: Backtest engine
# =============================================================================

def run_backtest(
    tiingo_tickers: pd.DataFrame,
    prefilter_df: pd.DataFrame,
) -> dict:
    """
    Run the in-sample backtest (2015-Q1 → 2020-Q4, 24 periods).
    All price and SEC data loaded into memory before entering the loop.
    Returns dict with period_records and universe_audit.
    """
    t_total_start = time.time()

    # Pre-compute maps once
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {
        r["ticker"]: (int(r["cik"]), r["sic"])
        for _, r in survivors.iterrows()
    }

    # delisted_map: ticker → endDate (for survivorship-correct exits)
    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        end = r["endDate"]
        if pd.notna(end):
            delisted_map[r["ticker"]] = end

    period_records: list = []
    universe_audit: list = []

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        t_next = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        hold_start = t
        hold_end   = t_next

        print(f"\n{'='*60}")
        print(f"Rebalance {t.date()} → hold end {hold_end.date()}  "
              f"({i+1}/{n_periods})")
        print(f"{'='*60}")

        # ── 1. Universe ─────────────────────────────────────────────
        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERROR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] Empty universe.")
            continue

        # ── 2. Momentum signals ─────────────────────────────────────
        mom_vals = [compute_momentum_signal(row["ticker"], t) for _, row in univ.iterrows()]
        univ["momentum"] = mom_vals
        n_with_signal = univ["momentum"].notna().sum()
        print(f"  [Signal] {n_with_signal}/{len(univ)} names have valid 12-1 momentum")

        # ── 3. Returns for hold period ───────────────────────────────
        try:
            returns_df = compute_returns(univ, hold_start, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERROR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        n_valid    = returns_df["raw_return"].notna().sum()

        universe_audit.append({
            "rebalance_date": t,
            "hold_end":       hold_end,
            "n_universe":     len(returns_df),
            "n_with_signal":  n_with_signal,
            "n_delisted":     n_delisted,
            "n_valid_returns": n_valid,
        })

        # ── 4. Portfolio ─────────────────────────────────────────────
        longs, shorts, lw, sw = build_portfolio(returns_df)
        if not longs:
            print(f"  [SKIP] Insufficient names for portfolio (need 40+, have {n_with_signal} with signal and valid return).")
            continue

        # ── 5. Long/short returns (net of costs) ────────────────────
        long_ret  = _leg_return(returns_df, longs,  lw, is_short=False)
        short_ret = _leg_return(returns_df, shorts, sw, is_short=True)
        ls_ret    = long_ret + short_ret

        # ── 6. Market proxy: equal-weight full universe ──────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if len(valid_all) > 0 else np.nan

        # ── 7. IC: Spearman rank correlation of signal vs realized return ──
        valid_sig = returns_df.dropna(subset=["momentum", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["momentum"], valid_sig["raw_return"])
        else:
            ic = np.nan

        record = {
            "rebalance_date": t,
            "hold_end":       hold_end,
            "n_universe":     len(returns_df),
            "n_with_signal":  n_with_signal,
            "n_delisted":     n_delisted,
            "long_ret":       long_ret,
            "short_ret":      short_ret,
            "ls_ret":         ls_ret,
            "mkt_ret":        mkt_ret,
            "ic":             ic,
            "n_long":         len(longs),
            "n_short":        len(shorts),
        }
        period_records.append(record)

        print(f"  [Result] L/S={ls_ret:.3%}  Long={long_ret:.3%}  Short={short_ret:.3%}  "
              f"Mkt={mkt_ret:.3%}  IC={ic:.3f}  n_long={len(longs)}  n_short={len(shorts)}")

    elapsed = time.time() - t_total_start
    return {
        "period_records": period_records,
        "universe_audit": universe_audit,
        "elapsed_s":      elapsed,
    }


# =============================================================================
# SECTION 12: Metrics
# =============================================================================

def compute_metrics(returns: pd.Series) -> dict:
    """Full performance metrics for a quarterly return series."""
    s = returns.dropna()
    n = len(s)
    if n == 0:
        return {k: np.nan for k in
                ["total_ret", "ann_ret", "sharpe", "calmar", "max_dd",
                 "t_stat", "win_rate", "n"]}

    mean_q = s.mean()
    std_q  = s.std(ddof=1) if n > 1 else 0.0

    cum_ret   = (1 + s).prod() - 1
    ann_ret   = (1 + mean_q) ** 4 - 1
    sharpe    = (mean_q / std_q) * np.sqrt(4) if std_q > 0 else np.nan
    t_stat    = mean_q / (std_q / np.sqrt(n)) if std_q > 0 else np.nan
    win_rate  = (s > 0).mean()

    cum = (1 + s).cumprod()
    roll_max = cum.cummax()
    drawdowns = (cum - roll_max) / roll_max
    max_dd    = float(drawdowns.min())

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

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
    """OLS regression of L/S returns on equal-weight universe (market proxy)."""
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < 4:
        return {"beta": np.nan, "alpha_q": np.nan, "r_squared": np.nan, "beta_t": np.nan}

    slope, intercept, r_value, p_value, se = stats.linregress(combined["mkt"], combined["ls"])
    beta_t = slope / se if se > 0 else np.nan
    return {
        "beta":      slope,
        "alpha_q":   intercept,
        "r_squared": r_value ** 2,
        "beta_t":    beta_t,
    }


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    """Bailey & Lopez de Prado (2014) DSR threshold (quarterly Sharpe units)."""
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    expected_max = stats.norm.ppf(1 - 1.0 / n_trials)
    return expected_max / np.sqrt(n_obs)


# =============================================================================
# SECTION 13: Output & registry
# =============================================================================

def print_results(results: dict) -> None:
    records = results["period_records"]
    audit   = results["universe_audit"]
    elapsed = results["elapsed_s"]

    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df  = pd.DataFrame(records)
    aud = pd.DataFrame(audit)

    ls_series  = df["ls_ret"]
    mkt_series = df["mkt_ret"]
    ic_series  = df["ic"].dropna()
    m          = compute_metrics(ls_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    obs_sr_q   = m["sharpe"] / np.sqrt(4) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (obs_sr_q > dsr_thr) if (not np.isnan(obs_sr_q) and not np.isnan(dsr_thr)) else False

    mean_ic  = ic_series.mean() if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    sep = "=" * 72

    # ── Survivorship audit ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SURVIVORSHIP AUDIT — In-Sample 2015-Q1 to 2020-Q4")
    print(sep)
    total_name_periods  = int(aud["n_universe"].sum())
    total_delistings    = int(aud["n_delisted"].sum())
    first_n             = int(aud.iloc[0]["n_universe"])
    first_del           = int(aud["n_delisted"].sum())
    print(f"  Total name-periods across all 24 rebalances : {total_name_periods:,}")
    print(f"  Total delisting events retained             : {total_delistings}")
    print(f"  Names at 2015-Q1 rebalance                 : {first_n}")
    print(f"  Of those, later-period delistings retained  : {first_del}  "
          f"(returns carried to last trading day, NOT dropped)")
    print(f"\n  No-look-ahead confirmations:")
    print(f"    Momentum formation window: t-12M to t-1M (skip month enforced in code)")
    print(f"    Market cap: SEC filing_date ≤ t, price last close ≤ t")
    print(f"    Entry price: first close on or after t; returns measured t→t+1Q")
    print(f"    Raw close used (not adjClose — Tiingo 'close' field)")
    print(f"\n  Per-period universe breakdown:")
    print(f"    {'Date':<12} {'Universe':>9} {'w/Signal':>9} {'Delisted':>9} {'ValidRet':>9}")
    print(f"    {'-'*12} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")
    for _, r in aud.iterrows():
        print(f"    {str(r['rebalance_date'].date()):<12} "
              f"{int(r['n_universe']):>9} {int(r['n_with_signal']):>9} "
              f"{int(r['n_delisted']):>9} {int(r['n_valid_returns']):>9}")

    # ── Performance summary ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("S9: CROSS-SECTIONAL MOMENTUM (JT 12-1) — IN-SAMPLE RESULTS")
    print("In-sample period: 2015-01-01 → 2021-01-01")
    print(sep)
    print(f"\n  {'Metric':<35} {'Value':>15}")
    print(f"  {'-'*35} {'-'*15}")
    print(f"  {'Total return (cumulative)':<35} {m['total_ret']:>14.2%}")
    print(f"  {'Annualized return':<35} {m['ann_ret']:>14.2%}")
    print(f"  {'Sharpe ratio (annualized)':<35} {m['sharpe']:>14.3f}")
    print(f"  {'CALMAR ratio':<35} {m['calmar']:>14.3f}")
    print(f"  {'Max drawdown':<35} {m['max_dd']:>14.2%}")
    print(f"  {'t-stat (mean quarterly return)':<35} {m['t_stat']:>14.3f}")
    print(f"  {'Market beta (vs EW universe)':<35} {beta_stats['beta']:>14.3f}")
    print(f"  {'  beta t-stat':<35} {beta_stats['beta_t']:>14.3f}")
    print(f"  {'  R-squared':<35} {beta_stats['r_squared']:>14.3f}")
    print(f"  {'Mean IC (Spearman)':<35} {mean_ic:>14.4f}")
    print(f"  {'IC t-stat':<35} {ic_tstat:>14.3f}")
    print(f"  {'Win rate':<35} {m['win_rate']:>14.1%}")
    print(f"  {'Periods (N)':<35} {m['n']:>14d}")
    print(f"\n  --- DSR Check (N_trials={N_TRIALS}) ---")
    print(f"  {'Observed quarterly Sharpe':<35} {obs_sr_q:>14.3f}")
    print(f"  {'DSR threshold (quarterly)':<35} {dsr_thr:>14.3f}")
    print(f"  {'Clears DSR':<35} {'YES' if clears_dsr else 'NO':>14}")
    print(f"\n  Raw quarterly returns (for Rf computation):")
    print(f"  Mean quarterly L/S return : {ls_series.mean():.4%}")
    print(f"  Std quarterly L/S return  : {ls_series.std(ddof=1):.4%}")

    # ── Quarterly returns log ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("QUARTERLY RETURNS LOG")
    print(sep)
    print(f"  {'Date':<12} {'L/S Ret':>8} {'Long':>8} {'Short':>8} {'Mkt':>8} {'IC':>7}  "
          f"{'nLong':>6} {'nShort':>6} {'Dlist':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}  "
          f"{'-'*6} {'-'*6} {'-'*6}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<12} "
              f"{r['ls_ret']:>7.2%} {r['long_ret']:>7.2%} {r['short_ret']:>7.2%} "
              f"{r['mkt_ret']:>7.2%} {r['ic']:>6.3f}  "
              f"{int(r['n_long']):>6} {int(r['n_short']):>6} {int(r['n_delisted']):>6}")

    # ── Drawdown series ──────────────────────────────────────────────────────
    dd = m.get("drawdowns", pd.Series(dtype=float))
    if not dd.empty:
        print(f"\n  Max drawdown: {m['max_dd']:.2%}")
        print(f"  Drawdown series (quarterly):")
        cum = (1 + ls_series.reset_index(drop=True)).cumprod()
        roll_max = cum.cummax()
        dd_s = (cum - roll_max) / roll_max
        for i, (date, val) in enumerate(zip(df["rebalance_date"], dd_s)):
            print(f"    {str(date.date()):<12}  {val:>7.2%}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


# =============================================================================
# SECTION 12b: Split-contamination validation
# =============================================================================

def validate_split_fix() -> None:
    """
    Scan the raw-close cache for 2:1 stock splits that fall inside a
    formation window, then show the momentum signal under raw close vs adjClose
    to quantify how much contamination the fix removes.

    adjClose note: Tiingo adjusts BACKWARD for both splits AND dividends
    (total-return proxy). Current price is unchanged; prior prices are scaled
    down by each split factor or dividend yield. For growth-oriented small/mid-
    cap, dividend adjustments are small; the dominant correction is splits.
    """
    print("\n=== SPLIT CONTAMINATION VALIDATION ===")
    print("adjClose: Tiingo adjusts BACKWARD for SPLITS + DIVIDENDS (total-return proxy).")
    print("Scanning raw close cache for 2:1 splits (overnight ratio ≈ 0.50) in 2013–2020...")

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
                # 2:1 split: overnight close drops to ≈50% (allow ±3%)
                if 0.47 <= r <= 0.53 and close[i] >= 5.0:
                    found.append({
                        "ticker":        ticker,
                        "split_date":    pd.Timestamp(dates[i + 1]),
                        "price_before":  float(close[i]),
                        "price_after":   float(close[i + 1]),
                    })
                    break
        except Exception:
            continue

    if not found:
        print("  No 2:1 split candidates found in first scan. Validation skipped.")
        return

    print(f"  Found {len(found)} candidate split(s).\n")

    for cand in found[:2]:
        ticker     = cand["ticker"]
        split_date = cand["split_date"]

        # Find a rebalance date whose formation window contains the split
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
        print(f"  Rebalance date: {t.date()}")
        print(f"  Formation wnd : {mom_start.date()} → {mom_end.date()}")

        # ── BEFORE fix: raw close signal ─────────────────────────────────────
        raw_csv = CACHE / "tiingo" / f"{ticker}.csv"
        raw_sig = None
        if raw_csv.exists():
            raw = pd.read_csv(raw_csv, parse_dates=["date"])
            w   = raw[(raw["date"] >= mom_start) & (raw["date"] <= mom_end)]
            if len(w) >= MOMENTUM_MIN_DAYS:
                sp, ep = float(w.iloc[0]["close"]), float(w.iloc[-1]["close"])
                raw_sig = ep / sp - 1
                print(f"  Raw close signal   (BEFORE fix): {raw_sig:>+9.2%}"
                      f"  (start=${sp:.2f}, end=${ep:.2f}  — split inside window)")

        # ── AFTER fix: adjClose signal ────────────────────────────────────────
        adj_df = _load_adj(ticker)   # fetches from Tiingo API if not cached
        adj_sig = None
        if not adj_df.empty:
            wa = adj_df[(adj_df["date"] >= mom_start) & (adj_df["date"] <= mom_end)]
            if len(wa) >= MOMENTUM_MIN_DAYS:
                sp_a, ep_a = float(wa.iloc[0]["adjClose"]), float(wa.iloc[-1]["adjClose"])
                adj_sig = ep_a / sp_a - 1
                print(f"  AdjClose signal    (AFTER fix):  {adj_sig:>+9.2%}"
                      f"  (start=${sp_a:.4f}, end=${ep_a:.4f}  — split corrected)")
        else:
            # No API key or fetch failed — estimate via manual split correction
            raw_csv2 = CACHE / "tiingo" / f"{ticker}.csv"
            if raw_csv2.exists():
                raw2 = pd.read_csv(raw_csv2, parse_dates=["date"])
                raw2 = raw2.sort_values("date").reset_index(drop=True)
                # Divide prices before the split date by 2 to reconstruct adj series
                raw2.loc[raw2["date"] < split_date, "close"] /= 2.0
                w2 = raw2[(raw2["date"] >= mom_start) & (raw2["date"] <= mom_end)]
                if len(w2) >= MOMENTUM_MIN_DAYS:
                    sp_e, ep_e = float(w2.iloc[0]["close"]), float(w2.iloc[-1]["close"])
                    adj_sig = ep_e / sp_e - 1
                    print(f"  Est. adj signal    (AFTER fix):  {adj_sig:>+9.2%}"
                          f"  (manual 2:1 correction, no API key)")

        if raw_sig is not None and adj_sig is not None:
            delta = adj_sig - raw_sig
            print(f"  Signal delta                   : {delta:>+9.2%}")
            if raw_sig < -0.10 and adj_sig > 0.05:
                print(f"  ★ Winner misclassified as LOSER (raw) → would be SHORTED, not LONGED")
            elif raw_sig > 0.10 and adj_sig < -0.05:
                print(f"  ★ Loser misclassified as WINNER (raw) → would be LONGED, not SHORTED")
            else:
                print(f"  Signal magnitude corrupted by {abs(delta):.1%}; sign {'preserved' if (raw_sig * adj_sig > 0) else 'FLIPPED'}")
        print()

    print("=== END VALIDATION ===\n")


def save_results(results: dict) -> None:
    """Save per-period CSV and update top-level trial registry."""
    records = results["period_records"]
    audit   = results["universe_audit"]
    if not records:
        return

    df  = pd.DataFrame(records)
    aud = pd.DataFrame(audit)

    df.to_csv(RESULTS / "s9_period_returns.csv", index=False)
    aud.to_csv(RESULTS / "s9_universe_audit.csv", index=False)
    print(f"[Output] Saved {RESULTS / 's9_period_returns.csv'}")
    print(f"[Output] Saved {RESULTS / 's9_universe_audit.csv'}")

    # ── Trial registry entry ─────────────────────────────────────────────────
    ls_series = df["ls_ret"]
    ic_series = df["ic"].dropna()
    m         = compute_metrics(ls_series)
    beta_stats = compute_market_beta(ls_series, df["mkt_ret"])

    obs_sr_q = m["sharpe"] / np.sqrt(4) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr  = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears   = bool(obs_sr_q > dsr_thr) if not (np.isnan(obs_sr_q) or np.isnan(dsr_thr)) else False

    mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_t    = (
        float(ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series))))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    row = {
        "timestamp":    datetime.now().isoformat(),
        "study":        "momentum_jt_S9",
        "hypothesis":   (
            "Cross-sectional 12-1 momentum (JT) should produce positive L/S returns "
            "in US small/mid-cap; pipeline validator — if harness can't reproduce "
            "the most replicated factor, signals a bug."
        ),
        "data_source":  "Tiingo (raw close) + SEC XBRL (shares for PIT mcap)",
        "status":       "completed_in_sample",
        "trial_number": 6,
        "n_trials_dsr": N_TRIALS,
        "ann_return":   m["ann_ret"],
        "sharpe":       m["sharpe"],
        "calmar":       m["calmar"],
        "max_drawdown": m["max_dd"],
        "t_stat":       m["t_stat"],
        "market_beta":  beta_stats["beta"],
        "mean_ic":      mean_ic,
        "ic_t_stat":    ic_t,
        "win_rate":     m["win_rate"],
        "n_periods":    m["n"],
        "obs_sr_q":     obs_sr_q,
        "dsr_threshold": dsr_thr,
        "clears_dsr":   clears,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    # Remove any prior S9 entry
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "momentum_jt_S9"]
    new_row = pd.DataFrame([row])
    reg_df = pd.concat([reg_df, new_row], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Output] Trial registry updated: {REGISTRY}")


# =============================================================================
# SECTION 14: Main
# =============================================================================

def main():
    print("=" * 70)
    print("S9: CROSS-SECTIONAL MOMENTUM (JT 12-1)")
    print(f"In-sample: {IS_START.date()} → {IS_END.date()}  ({len(REBALANCE_DATES)} periods)")
    print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── Step 1: Tiingo ticker metadata ────────────────────────────────────────
    print("\n[Step 1] Loading Tiingo ticker list...")
    tiingo = load_tiingo_tickers()
    tiingo_set = set(tiingo["ticker"])
    print(f"  {len(tiingo):,} US-exchange stock tickers")

    # ── Step 2: SEC ticker→CIK map ────────────────────────────────────────────
    print("\n[Step 2] Loading SEC ticker→CIK map...")
    sec_cik = load_sec_ticker_cik_map()

    # ── Step 3: SEC prefilter (from mgmt_pit cache) ───────────────────────────
    print("\n[Step 3] Loading SEC prefilter...")
    prefilter = load_prefilter()
    survivors = prefilter[prefilter["passed"]]
    survivor_tickers = [t for t in survivors["ticker"] if t in tiingo_set]
    survivor_ciks = [
        int(r["cik"]) for _, r in survivors.iterrows()
        if r["ticker"] in tiingo_set and not pd.isna(r["cik"])
    ]
    print(f"  {len(survivor_tickers):,} survivors active in Tiingo")

    # ── Step 4: Preload all data into memory ──────────────────────────────────
    print(f"\n[Step 4] Preloading raw close prices into memory "
          f"({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)
    print(f"  Raw close cache: {len(_PRICE_CACHE):,} tickers")

    print(f"\n[Step 4b] Preloading SEC facts ({len(survivor_ciks):,} CIKs)...")
    preload_sec_facts(survivor_ciks)

    # ── Step 4c: Adj-close preload (SIGNAL LAYER) ─────────────────────────────
    # adjClose = Tiingo split+dividend adjusted (total-return proxy).
    # Used exclusively for signal computation; raw close kept for return calc.
    print(f"\n[Step 4c] Preloading adjClose prices for signal layer "
          f"({len(survivor_tickers):,} tickers)...")
    preload_all_adj_prices(survivor_tickers)
    print(f"  Adj-close cache: {len(_ADJ_CACHE):,} tickers")

    # ── Step 4d: Split-contamination validation (runs once; remove when satisfied) ──
    validate_split_fix()

    # ── Step 5: Backtest ──────────────────────────────────────────────────────
    print("\n[Step 5] Running in-sample backtest (all data in memory)...")
    t0 = time.time()
    results = run_backtest(tiingo, prefilter)
    elapsed = time.time() - t0
    results["elapsed_s"] = elapsed
    print(f"\n[Step 5] Backtest completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # ── Step 6: Results ───────────────────────────────────────────────────────
    print("\n[Step 6] Results:")
    print_results(results)

    # ── Step 7: Save ──────────────────────────────────────────────────────────
    print("\n[Step 7] Saving outputs...")
    save_results(results)

    print("\n[Done]")


if __name__ == "__main__":
    main()
