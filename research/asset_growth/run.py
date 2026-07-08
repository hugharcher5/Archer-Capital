"""
S5: Asset Growth Anomaly  (independent trial #12)
==================================================
Signal   : YoY Total-Asset Growth = (Assets_t - Assets_t-4q) / Assets_t-4q
           Assets sourced directly from the SEC XBRL "Assets" instant tag
           (balance-sheet concept — no TTM accumulation needed, unlike GP/A).
           Both the current and prior-year values are gated on FILING DATE
           ≤ rebalance date — never a value that would not yet be public.

Look-ahead discipline:
  - Assets_t   : most recent Assets entry with filed ≤ as_of.
  - Assets_t-4q: Assets entry whose PERIOD END is ~365 calendar days before
    Assets_t's period end, ALSO gated on filed ≤ as_of (a restated prior-year
    figure filed after t is not usable even though its period end is old).
  - Direct match: entry with lag ∈ [320, 410] days (365 ± 45), closest to 365.
  - Fallback match: no direct entry found, but one exists with lag ∈
    [270, 320) ∪ (410, 460] days (365 ± up to 95) — data-coverage gap in the
    Assets tag (e.g. an irregular fiscal-period filing). Widen the window
    once; if still nothing, the name is dropped (no signal) for that period.

Portfolio: Long BOTTOM quintile (lowest asset growth), short TOP quintile
           (highest asset growth). Sign convention: firms that aggressively
           grew the balance sheet subsequently UNDERPERFORM (long = low growth).
           Beta-neutral: 60-day OLS vs EW universe return (adjClose layer).
           Equal-weight within each leg.
Name-count guard: if the natural quintile leg (ceil(n*0.20)) would hold
           fewer than 20 names (Assets-tag coverage gaps thinning the
           eligible pool), widen to TERCILES for that period only. This is
           a construction rule, not a new trial — how often it fires is
           logged and reported.
Rebalance: Quarterly, pre-registered (no sweep). Hold: 1 quarter.
           24 periods over 2015-2021.
In-sample: 2015-01-01 -> 2021-01-01. Walk-forward 2021-2025 stays reserved.
Trial    : #12 — independent family (does not overlap S3/quality [gp, trial
           #8], S1-S4/IVOL-MAX [trial #7], or S6-S9/momentum [trial #6]).
           DSR threshold computed at N_TRIALS=12.

Falsification diagnostics (embedded in primary design, not bolted on after):
  - Sign-convention audit: assert mean growth of the long leg (low-growth)
    is below the mean growth of the short leg (high-growth) every period
    before results are interpreted. A flipped sign here would silently
    invert the whole result (same risk class as the S9 split-adjustment bug).
  - Concentration check: report the single quarter and the single SIC
    major-group sector contributing the largest share of cumulative L/S
    return, flagged if either exceeds a disproportionate share.

Coverage note: Assets is a near-universal balance-sheet tag (unlike GP,
which needs a Revenue/COGS split many service firms don't report). The
primary failure mode here is COVERAGE GAPS — a company that only files
annually, or skips a quarter's XBRL tagging — not concept absence. Direct
vs fallback prior-year match rates are reported each period and in
aggregate; fallback use > 25% of computed signals is flagged (same
threshold precedent as the IVOL adjClose-repair data-quality check).

No-survivorship guarantee:
  - Delisted tickers retained with returns to last trading day.
  - Universe rebuilt PIT from SEC filings each quarter.

Namespacing: all outputs (results, coverage/cache) live under
research/asset_growth/ — kept fully separate from the concurrently running
S2/PEAD trial's caches and result paths.
"""

import json
import os
import pickle
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats


# =============================================================================
# SECTION 1: Paths, config, constants
# =============================================================================

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
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

MGMT_PIT = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE    = MGMT_PIT / "cache"                              # shared, read-only
OWN_CACHE = Path(__file__).resolve().parent / "cache"      # S5-private cache
RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (same as all prior experiments) ─────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── Beta window (for beta-neutral sizing) ─────────────────────────────────────
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Asset-growth XBRL concepts ────────────────────────────────────────────────
ASSETS_CONCEPT   = "Assets"
SHARES_CONCEPTS  = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]
AG_NEEDED = {ASSETS_CONCEPT} | set(SHARES_CONCEPTS)

# ── YoY lag windows (calendar days between current and prior period-end) ─────
DIRECT_LAG_MIN   = 320
DIRECT_LAG_MAX   = 410
FALLBACK_LAG_MIN = 270
FALLBACK_LAG_MAX = 460

# Plausibility floor on any Assets value used in the signal. Filters shell-
# company / pre-reverse-merger placeholder XBRL entries (observed: a real
# operating company's CIK carrying a prior "$1,000 total assets" entry from
# its shell-registrant period, which — if not filtered — lands inside the
# lag window and produces a nonsensical multiple-orders-of-magnitude growth
# ratio). $1M is far below any plausible balance sheet for a $100M-2B mcap
# universe member; it only screens out data artifacts, not small real firms.
ASSETS_FLOOR = 1e6

# ── Backtest dates ────────────────────────────────────────────────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 24 quarterly periods

# ── Cost model ───────────────────────────────────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR: this trial is independent, not part of any existing family ─────────
N_TRIALS = 12   # trial #12 — H1+E1-E4+S9+(IVOL/MAX family)+S3-GP+... +S5-AssetGrowth


# =============================================================================
# SECTION 2: Rate-limited HTTP
# =============================================================================

_TIINGO_LAST: float = 0.0
_SEC_LAST:    float = 0.0


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
# SECTION 3: Tiingo metadata helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
        return df
    if not TIINGO_KEY:
        raise RuntimeError("No TIINGO_API_KEY and no cached tickers.csv.")
    import io, zipfile
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
    return df


# =============================================================================
# SECTION 4: SEC prefilter
# =============================================================================

def load_prefilter() -> pd.DataFrame:
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run mgmt_pit/run.py first.")
    df = pd.read_csv(path)
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors.")
    return df


# =============================================================================
# SECTION 5: Raw price cache (return layer)
# =============================================================================

_PRICES_PKL   = CACHE / "prices_all.pkl"
_PRICE_CACHE: dict[str, pd.DataFrame] = {}
_PRICE_EMPTY  = pd.DataFrame(columns=["date", "close", "volume"])


def preload_all_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if not _PRICES_PKL.exists():
        raise FileNotFoundError(f"Missing {_PRICES_PKL}. Run research/ivol/run.py first.")
    with open(_PRICES_PKL, "rb") as f:
        data = pickle.load(f)
    _PRICE_CACHE.update(data.get("prices", {}))
    print(f"[Preload] Raw prices: {len(_PRICE_CACHE):,} tickers in {time.time()-t0:.1f}s")


def _last_close(ticker: str, as_of: pd.Timestamp) -> Optional[float]:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return None
    sub = df[df["date"] <= as_of]
    return float(sub.iloc[-1]["close"]) if not sub.empty else None


def _avg_dollar_vol(ticker: str, as_of: pd.Timestamp, lookback: int = 60) -> float:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return 0.0
    sub = df[df["date"] <= as_of].tail(lookback)
    return float((sub["close"] * sub["volume"]).mean()) if not sub.empty else 0.0


def _get_price_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return _PRICE_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 6: AdjClose cache (beta layer)
# =============================================================================

_ADJ_PKL      = CACHE / "adjclose_all.pkl"
_ADJ_CACHE:   dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str] = set()
_ADJ_EMPTY    = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if not _ADJ_PKL.exists():
        print("[AdjPreload] No pickle — beta will use raw close fallback.")
        return
    with open(_ADJ_PKL, "rb") as f:
        data = pickle.load(f)
    _ADJ_CACHE.update(data.get("adj", {}))
    _ADJ_SENTINEL.update(data.get("sentinels", set()))
    adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    print(f"[AdjPreload] {adj_loaded:,} tickers in {time.time()-t0:.1f}s")


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 7: SEC Asset-Growth facts cache (S5-private, trimmed)
# =============================================================================
#
# We only need the "Assets" instant tag + shares-outstanding per CIK. Rather
# than share GP's trimmed cache (research/mgmt_pit/cache/gp_facts_trimmed.json)
# or PEAD's, we build our own trimmed copy under research/asset_growth/cache/
# so this trial's cache lifecycle is fully independent of any concurrently
# running trial. The raw per-CIK source (mgmt_pit/cache/sec_facts/*.json) is
# shared, read-only infrastructure — safe to read concurrently.

_AG_FACTS_CACHE: dict[int, Optional[dict]] = {}
_AG_FACTS_PKL   = OWN_CACHE / "asset_growth_facts_trimmed.json"


def preload_ag_facts(ciks: list[int]) -> None:
    t0 = time.time()

    if _AG_FACTS_PKL.exists():
        with open(_AG_FACTS_PKL) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _AG_FACTS_CACHE[int(k)] = v
        missing = [c for c in ciks if c not in _AG_FACTS_CACHE]
        if missing:
            _load_ag_facts_from_files(missing, save=False)
        loaded = sum(1 for v in _AG_FACTS_CACHE.values() if v is not None)
        print(f"[AGFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")
        return

    print(f"[AGFacts] First run — loading {len(ciks):,} full fact files "
          f"(this takes ~1 min)...")
    _load_ag_facts_from_files(ciks, save=True)
    loaded = sum(1 for v in _AG_FACTS_CACHE.values() if v is not None)
    print(f"[AGFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


def _load_ag_facts_from_files(ciks: list[int], save: bool = True) -> None:
    for cik in ciks:
        p = CACHE / "sec_facts" / f"{cik}.json"
        if not p.exists():
            _AG_FACTS_CACHE[cik] = None
            continue
        try:
            with open(p) as f:
                full = json.load(f)
            trimmed: dict = {"facts": {"us-gaap": {}}}
            ug = full.get("facts", {}).get("us-gaap", {})
            for concept in AG_NEEDED:
                if concept in ug:
                    trimmed["facts"]["us-gaap"][concept] = ug[concept]
            _AG_FACTS_CACHE[cik] = trimmed
        except Exception:
            _AG_FACTS_CACHE[cik] = None
    if save:
        print("[AGFacts] Saving asset-growth facts trimmed cache...")
        serializable = {str(k): v for k, v in _AG_FACTS_CACHE.items()}
        with open(_AG_FACTS_PKL, "w") as f:
            json.dump(serializable, f)


def _get_shares_pit(facts: Optional[dict], as_of_str: str) -> Optional[float]:
    """Most recent shares outstanding where filed ≤ as_of."""
    if facts is None:
        return None
    for concept in SHARES_CONCEPTS:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        best: Optional[tuple] = None
        for entries in units_dict.values():
            for e in entries:
                filed = e.get("filed")
                end   = e.get("end")
                val   = e.get("val")
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
# SECTION 8: Universe builder (identical to S-IVOL / S-MAX / S3-GP)
# =============================================================================

def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
) -> pd.DataFrame:
    """
    PIT investable universe at rebalance_date.
    Shares-outstanding sourced from _AG_FACTS_CACHE (already loaded).
    """
    as_of_str     = rebalance_date.strftime("%Y-%m-%d")
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
        ticker   = row["ticker"]
        cik, sic = cik_sic_map.get(ticker, (None, None))
        if cik is None:
            continue

        price = _last_close(ticker, rebalance_date)
        if price is None or price < PRICE_MIN:
            continue

        facts  = _AG_FACTS_CACHE.get(cik)
        shares = _get_shares_pit(facts, as_of_str)
        if shares is None or shares <= 0:
            continue
        mcap = shares * price
        if not (MCAP_MIN <= mcap <= MCAP_MAX):
            continue

        dvol = _avg_dollar_vol(ticker, rebalance_date)
        if dvol < DOLLAR_VOL_MIN:
            continue

        rows.append({"ticker": ticker, "cik": cik, "sic": sic, "mcap": mcap,
                     "price": price, "shares": shares, "avg_dvol": dvol})

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "cik", "sic", "mcap", "price", "shares", "avg_dvol"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    print(f"  [Universe] {rebalance_date.date()} → {len(df):,} names", end="")
    return df


# =============================================================================
# SECTION 9: Asset-growth signal computation (purely fundamental)
# =============================================================================

def _all_assets_entries(facts: Optional[dict], as_of_str: str) -> list:
    """
    All (filed, end, val) Assets entries filed <= as_of, any unit key,
    below ASSETS_FLOOR excluded (shell/placeholder filter), deduplicated
    by end-date keeping the most-recently-filed revision.
    """
    if facts is None:
        return []
    try:
        units_dict = facts["facts"]["us-gaap"][ASSETS_CONCEPT]["units"]
    except (KeyError, TypeError):
        return []
    by_end: dict = {}
    for entries in units_dict.values():
        for e in entries:
            filed = e.get("filed")
            end   = e.get("end")
            val   = e.get("val")
            if filed is None or end is None or val is None:
                continue
            if filed > as_of_str:
                continue
            val = float(val)
            if val < ASSETS_FLOOR:
                continue
            if end not in by_end or filed > by_end[end][0]:
                by_end[end] = (filed, end, val)
    return list(by_end.values())


def _get_asset_growth_pit(facts: Optional[dict], as_of_str: str) -> tuple:
    """
    YoY asset growth, point-in-time.

    Current : entry with the latest period-end date (filed <= as_of);
              ties broken by latest filed date.
    Prior   : entry ~365 calendar days before the current period-end,
              also filed <= as_of. Direct window [320,410]d; fallback
              window [270,320) U (410,460]d (data-coverage gap in the
              Assets tag). Closest-to-365 wins within whichever window
              has a match.

    Returns (growth, method) where method in {"direct", "fallback", None}.
    growth is None if no valid current or prior value exists.
    """
    entries = _all_assets_entries(facts, as_of_str)
    if not entries:
        return None, None

    filed_c, end_c, val_c = max(entries, key=lambda x: (x[1], x[0]))
    if val_c <= 0:
        return None, None

    end_c_date = date.fromisoformat(end_c)
    candidates = []
    for filed, end, val in entries:
        if end >= end_c or val <= 0:
            continue
        lag = (end_c_date - date.fromisoformat(end)).days
        candidates.append((lag, filed, end, val))

    direct = [c for c in candidates if DIRECT_LAG_MIN <= c[0] <= DIRECT_LAG_MAX]
    if direct:
        lag, filed, end, val_p = min(direct, key=lambda c: abs(c[0] - 365))
        return (val_c - val_p) / val_p, "direct"

    fallback = [
        c for c in candidates
        if (FALLBACK_LAG_MIN <= c[0] < DIRECT_LAG_MIN)
        or (DIRECT_LAG_MAX < c[0] <= FALLBACK_LAG_MAX)
    ]
    if fallback:
        lag, filed, end, val_p = min(fallback, key=lambda c: abs(c[0] - 365))
        return (val_c - val_p) / val_p, "fallback"

    return None, None


def compute_asset_growth_signals(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple:
    """
    Compute YoY asset growth for each universe member.

    Coverage returns:
      n_signal   : names with valid growth signal
      n_direct   : names where prior-year Assets matched the direct window
      n_fallback : names where prior-year Assets matched only the wider,
                   data-coverage-gap fallback window
      n_no_data  : names with no usable current/prior Assets pair at all
    """
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    records   = []
    n_direct = n_fallback = n_no_data = 0

    for _, row in universe_df.iterrows():
        cik   = int(row["cik"])
        facts = _AG_FACTS_CACHE.get(cik)

        growth, method = _get_asset_growth_pit(facts, as_of_str)
        if growth is None or not np.isfinite(growth):
            n_no_data += 1
            continue

        if method == "direct":
            n_direct += 1
        else:
            n_fallback += 1

        records.append({
            "ticker":        row["ticker"],
            "growth_signal": growth,
            "growth_method": method,
        })

    signals_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["ticker", "growth_signal", "growth_method"])
    )
    coverage = {
        "n_universe":  len(universe_df),
        "n_signal":    len(signals_df),
        "n_direct":    n_direct,
        "n_fallback":  n_fallback,
        "n_no_data":   n_no_data,
        "pct_dropped": 100 * (len(universe_df) - len(signals_df)) / max(len(universe_df), 1),
    }
    return signals_df, coverage


# =============================================================================
# SECTION 10: Beta computation (60-day EW adjClose market model)
# =============================================================================

def compute_betas(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> dict:
    """
    Compute per-ticker 60-day OLS beta vs EW universe return.
    Returns dict: ticker → beta (np.nan if insufficient data).
    """
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)

    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj    = _adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            raw_df = _PRICE_CACHE.get(ticker, _ADJ_EMPTY)
            if raw_df.empty:
                continue
            raw_w = raw_df[(raw_df["date"] >= beta_start)
                           & (raw_df["date"] <= rebalance_date)]
            if raw_w.empty:
                continue
            s = raw_w.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()
        else:
            s   = adj.set_index("date")["adjClose"]
            s   = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()

        if len(ret) >= BETA_MIN_DAYS:
            adj_series[ticker] = ret

    if not adj_series:
        return {}

    ret_mat = pd.DataFrame(adj_series).sort_index()
    ew_mkt  = ret_mat.clip(-0.5, 0.5).mean(axis=1)

    betas: dict[str, float] = {}
    for ticker, ret_s in adj_series.items():
        y_s = ret_s
        x_s = ew_mkt.reindex(y_s.index).dropna()
        y_s = y_s.reindex(x_s.index).dropna()
        x_s = x_s.reindex(y_s.index)
        n   = len(y_s)
        if n < BETA_MIN_DAYS:
            betas[ticker] = np.nan
            continue
        X = np.column_stack([np.ones(n), x_s.values])
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(X, y_s.values, rcond=None)
            betas[ticker] = float(coeffs[1]) if rank >= 2 else np.nan
        except Exception:
            betas[ticker] = np.nan

    return betas


# =============================================================================
# SECTION 11: Return calculation (raw close, survivorship-bias-free)
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
    delisted_map: dict,
) -> pd.DataFrame:
    """
    Entry: first close on or after hold_start.
    Exit : last close on or before hold_end (or delist_date if earlier).
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
                (exit_cands if not exit_cands.empty else prices).iloc[-1]["close"])
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
# SECTION 12: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame, betas: dict) -> tuple:
    """
    Long BOTTOM 20% asset growth (lowest — long leg), short TOP 20%
    (highest — short leg). Min structural leg size 20 via quintiles;
    if fewer than 20 names would land in a quintile leg, widen to
    TERCILES for this period only (name-count guard, logged not a new trial).
    """
    df = signals_df.dropna(subset=["growth_signal", "raw_return"]).copy()
    n  = len(df)
    if n < 6:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, False

    leg_quintile = int(np.ceil(n * 0.20))
    guard_fired  = leg_quintile < 20
    leg          = int(np.ceil(n / 3.0)) if guard_fired else leg_quintile
    leg          = min(leg, n // 2)
    if leg < 1:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, guard_fired

    ranked = df.sort_values("growth_signal", ascending=True)   # low → high
    longs  = ranked.iloc[:leg]["ticker"].tolist()    # LOW growth  = long
    shorts = ranked.iloc[-leg:]["ticker"].tolist()   # HIGH growth = short

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    mean_beta_long  = float(np.nanmean([betas.get(t, np.nan) for t in longs]))
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short, guard_fired


def _leg_return_detail(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool,
    turnover: float,
) -> tuple:
    """Quarterly cost model: borrow /4, bid-ask x actual turnover."""
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (MCAP_MIN + MCAP_MAX) / 2

    gross = ba_cost = borrow_cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross   += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / 4.0) * w

    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    else:
        return ( gross - total_cost),   gross,  ba_cost, borrow_cost


# =============================================================================
# SECTION 13: Backtest engine
# =============================================================================

def run_backtest(
    tiingo_tickers: pd.DataFrame,
    prefilter_df: pd.DataFrame,
) -> dict:
    t_total = time.time()

    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list   = []
    coverage_records: list = []
    universe_audit: list   = []
    sector_contrib: dict   = {}   # sic major group -> cumulative $ contribution
    sign_violations: int   = 0
    guard_fire_count: int  = 0

    prev_long_set:  set = set()
    prev_short_set: set = set()

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        hold_end = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END

        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} → {hold_end.date()}")

        # ── 1. Universe ───────────────────────────────────────────────────
        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        # ── 2. Asset-growth signal (fundamental — no price layer) ──────────
        signals_df, cov = compute_asset_growth_signals(univ, t)
        n_dropped = cov["n_universe"] - cov["n_signal"]
        pct_drop  = cov["pct_dropped"]
        fallback_frac = 100 * cov["n_fallback"] / max(cov["n_signal"], 1)
        print(f"  AG: {cov['n_signal']}/{cov['n_universe']} with signal "
              f"({pct_drop:.0f}% dropped) | direct={cov['n_direct']} "
              f"fallback={cov['n_fallback']} ({fallback_frac:.0f}% of signal) "
              f"noData={cov['n_no_data']}")
        if fallback_frac > 25:
            print(f"  *** WARNING: fallback prior-year match used for "
                  f"{fallback_frac:.0f}% of signals (>25% threshold). ***")

        coverage_records.append({"rebalance_date": t, **cov})

        # ── 3. Merge signals; compute hold-period returns ───────────────────
        univ_sig = univ.merge(signals_df[["ticker", "growth_signal"]], on="ticker", how="left")
        try:
            returns_df = compute_returns(univ_sig, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())

        universe_audit.append({
            "rebalance_date":  t,
            "hold_end":        hold_end,
            "n_universe":      len(returns_df),
            "n_with_signal":   int(returns_df["growth_signal"].notna().sum()),
            "n_delisted":      n_delisted,
            "n_valid_returns": int(returns_df["raw_return"].notna().sum()),
        })

        # ── 4. Beta computation (60-day EW adjClose) ────────────────────────
        betas = compute_betas(univ, t)

        # ── 5. Portfolio (beta-neutral, name-count guard) ───────────────────
        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short, guard_fired) = build_portfolio(returns_df, betas)
        if not longs:
            print(f"  [SKIP] n<6 after signal filter")
            continue
        if guard_fired:
            guard_fire_count += 1
            print(f"  [GUARD] quintile leg <20 names → widened to terciles "
                  f"(leg={len(longs)})")

        # ── 5b. Sign-convention audit ────────────────────────────────────────
        mean_growth_long  = float(signals_df.set_index("ticker")["growth_signal"]
                                   .reindex(longs).mean())
        mean_growth_short = float(signals_df.set_index("ticker")["growth_signal"]
                                   .reindex(shorts).mean())
        sign_ok = mean_growth_long < mean_growth_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg growth ({mean_growth_long:.3f}) "
                  f">= short-leg growth ({mean_growth_short:.3f}) ***")

        # ── 6. Turnover ───────────────────────────────────────────────────────
        if prev_long_set:
            long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1)
            short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0

        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        # ── 7. Leg returns (quarterly cost model) ─────────────────────────────
        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_detail(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw
        ls_cost  = ls_ba + ls_bw

        # ── 7b. Sector contribution (concentration check) ────────────────────
        sic_map  = returns_df.set_index("ticker")["sic"].to_dict()
        ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
        for tk in longs:
            r = ret_map.get(tk, np.nan)
            if np.isnan(r):
                continue
            sic = sic_map.get(tk, np.nan)
            grp = int(sic // 100) if pd.notna(sic) else -1
            sector_contrib[grp] = sector_contrib.get(grp, 0.0) + long_scale * lw.get(tk, 0.0) * r
        for tk in shorts:
            r = ret_map.get(tk, np.nan)
            if np.isnan(r):
                continue
            sic = sic_map.get(tk, np.nan)
            grp = int(sic // 100) if pd.notna(sic) else -1
            sector_contrib[grp] = sector_contrib.get(grp, 0.0) - short_scale * sw.get(tk, 0.0) * r

        # ── 8. EW market return ───────────────────────────────────────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        # ── 9. IC: Spearman(growth_signal, realized_return) ───────────────────
        # IC < 0 expected: high growth -> low subsequent return (hypothesis).
        # We report IC on the signal AS DEFINED (low growth ranks low), so a
        # NEGATIVE IC confirms the hypothesis here (opposite convention to GP).
        valid_sig = returns_df.dropna(subset=["growth_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["growth_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} "
              f"to={avg_turnover:.0%} β_L={mean_beta_long:.2f} β_S={mean_beta_short:.2f} "
              f"IC={ic:.3f} Dlist={n_delisted} gL={mean_growth_long:.3f} gS={mean_growth_short:.3f}")

        period_records.append({
            "rebalance_date":   t,
            "hold_end":         hold_end,
            "n_universe":       len(returns_df),
            "n_with_signal":    int(returns_df["growth_signal"].notna().sum()),
            "n_delisted":       n_delisted,
            "pct_dropped":      pct_drop,
            "long_ret":         long_net,
            "short_ret":        short_net,
            "ls_ret":           ls_net,
            "ls_gross":         ls_gross,
            "ls_cost":          ls_cost,
            "ls_ba_cost":       ls_ba,
            "ls_borrow_cost":   ls_bw,
            "long_scale":       long_scale,
            "short_scale":      short_scale,
            "mean_beta_long":   mean_beta_long,
            "mean_beta_short":  mean_beta_short,
            "long_turnover":    long_turnover,
            "short_turnover":   short_turnover,
            "avg_turnover":     avg_turnover,
            "mkt_ret":          mkt_ret,
            "ic":               ic,
            "n_long":           len(longs),
            "n_short":          len(shorts),
            "guard_fired":      guard_fired,
            "mean_growth_long":  mean_growth_long,
            "mean_growth_short": mean_growth_short,
            "sign_ok":          sign_ok,
        })

    elapsed = time.time() - t_total
    return {
        "period_records":   period_records,
        "universe_audit":   universe_audit,
        "coverage_records": coverage_records,
        "sector_contrib":   sector_contrib,
        "sign_violations":  sign_violations,
        "guard_fire_count": guard_fire_count,
        "elapsed_s":        elapsed,
    }


# =============================================================================
# SECTION 14: Metrics
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
    """Quarterly series → annualized (4 periods/year)."""
    ls = ls_series.dropna().reset_index(drop=True)
    n  = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "max_dd": np.nan, "t_stat": np.nan,
                "win_rate": np.nan, "n": n}

    cum     = (1 + ls).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (4.0 / n) - 1)
    mu      = float(ls.mean())
    sigma   = float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(4)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s    = (cum - roll_mx) / roll_mx
    max_dd  = float(dd_s.min())
    calmar  = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat  = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt  = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "t_stat": t_stat,
            "win_rate": win_rt, "n": n}


def compute_market_beta(ls_returns: pd.Series, mkt_returns: pd.Series) -> dict:
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < 5:
        return {"beta": np.nan, "beta_t": np.nan}
    x = combined["mkt"].values
    y = combined["ls"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"beta": np.nan, "beta_t": np.nan}
    beta   = float(coeffs[1])
    y_hat  = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n      = len(y)
    se     = (np.sqrt(ss_res / max(n - 2, 1))
              / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    """Per-period SR threshold (Bailey & Lopez de Prado 2014)."""
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 15: Output & registry
# =============================================================================

def print_results(results: dict) -> None:
    records  = results["period_records"]
    coverage = results["coverage_records"]
    elapsed  = results["elapsed_s"]

    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df   = pd.DataFrame(records)
    cov  = pd.DataFrame(coverage)

    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m          = compute_metrics(ls_series)
    m_gross    = compute_metrics(gross_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
    dsr_thr       = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr    = (bool(per_period_sr > dsr_thr)
                     if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False)

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    mean_monthly_ba_bps = float(df["ls_ba_cost"].mean())   * 4 * 10_000
    mean_monthly_bw_bps = float(df["ls_borrow_cost"].mean()) * 4 * 10_000
    annual_cost_drag    = (mean_monthly_ba_bps + mean_monthly_bw_bps)
    mean_turnover       = float(df["avg_turnover"].mean())
    mean_long_scale     = float(df["long_scale"].mean())
    mean_short_scale    = float(df["short_scale"].mean())
    mean_bl             = float(df["mean_beta_long"].mean())
    mean_bs             = float(df["mean_beta_short"].mean())
    mean_drop           = float(df["pct_dropped"].mean())

    guard_fire_count = int(results["guard_fire_count"])
    sign_violations  = int(results["sign_violations"])
    n_periods_run    = len(df)

    # Coverage summary across all periods
    total_univ   = int(cov["n_universe"].sum())
    total_sig    = int(cov["n_signal"].sum())
    total_direct = int(cov["n_direct"].sum())
    total_fb     = int(cov["n_fallback"].sum())
    overall_drop = 100 * (total_univ - total_sig) / max(total_univ, 1)
    overall_fb_frac = 100 * total_fb / max(total_sig, 1)

    sep = "=" * 78

    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT")
    print(sep)
    print(f"  Periods where long-leg growth >= short-leg growth (violation): "
          f"{sign_violations}/{n_periods_run}")
    if sign_violations == 0:
        print("  PASS — long leg is lowest-growth every period, as intended.")
    else:
        print("  *** FAIL — sign convention broken in at least one period. "
              "Results below may be inverted. ***")

    print(f"\n{sep}")
    print("ASSETS-TAG COVERAGE AUDIT")
    print(sep)
    print(f"  Total stock-period slots (universe × quarters):  {total_univ:,}")
    print(f"  With valid growth signal:                        {total_sig:,} ({100*total_sig/max(total_univ,1):.1f}%)")
    print(f"    — direct prior-year match (320-410d lag):       {total_direct:,}")
    print(f"    — fallback match (data-coverage gap, wider window): {total_fb:,} "
          f"({overall_fb_frac:.1f}% of signals)")
    print(f"  Dropped (no signal):                              {total_univ-total_sig:,} ({overall_drop:.1f}%)")
    if overall_fb_frac > 25:
        print(f"\n  *** WARNING: fallback used for {overall_fb_frac:.0f}% of signals "
              f"(>25% threshold — IVOL data-quality precedent). ***")
    else:
        print(f"\n  Fallback usage acceptable ({overall_fb_frac:.1f}% < 25% threshold).")

    print(f"\n  Name-count guard (quintile leg <20 -> widen to terciles) fired: "
          f"{guard_fire_count}/{n_periods_run} periods "
          f"({100*guard_fire_count/max(n_periods_run,1):.0f}%)")

    print(f"\n{sep}")
    print("S5: ASSET GROWTH ANOMALY — IN-SAMPLE RESULTS  (independent trial #12)")
    print("Signal: YoY Asset Growth = (Assets_t - Assets_t-4q)/Assets_t-4q (filed <= t)")
    print("Long BOTTOM 20% growth | Short TOP 20% growth | Beta-neutral | Quarterly")
    print(sep)

    print(f"\n  *** INFORMATION COEFFICIENT (cost-independent signal quality) ***")
    print(f"  Mean IC (Spearman, growth vs ret)      : {mean_ic:>12.4f}")
    print(f"  IC t-stat                              : {ic_tstat:>12.3f}")
    print(f"  IC < 0 = high asset growth underperformed (hypothesis confirmed).")
    print(f"  |IC| t > 2 = signal has statistically significant predictive power.")

    print(f"\n  {'Metric':<46} {'Value':>14}")
    print(f"  {'-'*46} {'-'*14}")
    print(f"  {'Total cumulative return (net)':<46} {m['total_ret']:>13.2%}")
    print(f"  {'Annualized return (net)':<46} {m['ann_ret']:>13.2%}")
    print(f"  {'Annualized return (GROSS)':<46} {m_gross['ann_ret']:>13.2%}")
    print(f"  {'Cost drag (annualized bps)':<46} {annual_cost_drag:>13.0f}")
    print(f"    {'  bid-ask (bps/yr)':<44} {mean_monthly_ba_bps:>13.0f}")
    print(f"    {'  borrow  (bps/yr)':<44} {mean_monthly_bw_bps:>13.0f}")
    print(f"  {'Mean actual turnover per quarter':<46} {mean_turnover:>13.0%}")
    print(f"  {'Sharpe ratio (annualized, net)':<46} {m['sharpe']:>13.3f}")
    print(f"  {'CALMAR ratio':<46} {m['calmar']:>13.3f}")
    print(f"  {'Max drawdown':<46} {m['max_dd']:>13.2%}")
    print(f"  {'t-stat (mean quarterly net return)':<46} {m['t_stat']:>13.3f}")
    print(f"  {'Market beta (vs EW universe)':<46} {beta_stats['beta']:>13.3f}")
    print(f"    {'beta t-stat (should be ~0)':<44} {beta_stats['beta_t']:>13.3f}")
    print(f"  {'Mean gross long exposure ($)':<46} {mean_long_scale:>13.3f}")
    print(f"  {'Mean gross short exposure ($)':<46} {mean_short_scale:>13.3f}")
    print(f"  {'Mean ex-ante beta long leg':<46} {mean_bl:>13.3f}")
    print(f"  {'Mean ex-ante beta short leg':<46} {mean_bs:>13.3f}")
    print(f"  {'Win rate':<46} {m['win_rate']:>13.1%}")
    print(f"  {'Periods (N)':<46} {m['n']:>13d}")
    print(f"  {'Mean universe names dropped (%)':<46} {mean_drop:>12.1f}%")

    print(f"\n  --- DSR Check (N_trials={N_TRIALS}, quarterly, per-period SR units) ---")
    print(f"  {'Per-period Sharpe':<46} {per_period_sr:>13.4f}")
    print(f"  {'DSR threshold (per-quarter SR)':<46} {dsr_thr:>13.4f}")
    print(f"  {'Annualized Sharpe threshold':<46} {dsr_thr*2:>13.4f}")
    print(f"  {'Clears DSR':<46} {'YES' if clears_dsr else 'NO':>13}")

    print(f"\n  Raw quarterly returns (compute alpha vs T-bill Rf externally):")
    print(f"  Mean quarterly L/S (net) : {ls_series.mean():.4%}")
    print(f"  Std  quarterly L/S (net) : {ls_series.std(ddof=1):.4%}")

    # ── Concentration check: single-quarter contribution ──────────────────────
    cum_final   = float((1 + ls_series.reset_index(drop=True)).cumprod().iloc[-1] - 1)
    abs_period_sum = float(ls_series.abs().sum())
    print(f"\n{sep}")
    print("CONCENTRATION CHECK (single-episode artifact risk)")
    print(sep)
    period_share = (ls_series.abs() / abs_period_sum) if abs_period_sum > 0 else ls_series * 0
    top_idx   = period_share.idxmax()
    top_share = float(period_share.loc[top_idx])
    top_date  = df.loc[top_idx, "rebalance_date"]
    print(f"  Largest single-quarter share of total |return| mass: "
          f"{top_date.date()} = {top_share:.1%}")
    if top_share > 0.30:
        print(f"  *** WARNING: single quarter drives >{30:.0f}% of return mass — "
              f"possible single-episode artifact. ***")
    else:
        print(f"  No single quarter dominates (<30% threshold).")

    # ── Concentration check: sector contribution ───────────────────────────────
    sector_contrib = results.get("sector_contrib", {})
    if sector_contrib:
        total_abs = sum(abs(v) for v in sector_contrib.values())
        if total_abs > 0:
            top_sic, top_val = max(sector_contrib.items(), key=lambda kv: abs(kv[1]))
            top_sec_share = abs(top_val) / total_abs
            print(f"  Largest single SIC major-group (2-digit) contribution: "
                  f"SIC {top_sic:02d} = {top_sec_share:.1%} of total |contribution|")
            if top_sec_share > 0.40:
                print(f"  *** WARNING: single sector drives >{40:.0f}% of return — "
                      f"possible sector concentration artifact. ***")
            else:
                print(f"  No single sector dominates (<40% threshold).")

    # Quarterly returns log
    print(f"\n{sep}")
    print("QUARTERLY RETURNS LOG")
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} {'TOver':>6} "
          f"{'LScl':>5} {'SScl':>5} {'Mkt':>7} {'IC':>6} {'nL':>3} {'Drop%':>5} {'Guard':>5}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*5} "
          f"{'-'*7} {'-'*6} {'-'*3} {'-'*5} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
              f"{r['avg_turnover']:>5.0%} "
              f"{r['long_scale']:>4.2f}x {r['short_scale']:>4.2f}x "
              f"{r['mkt_ret']:>6.2%} {r['ic']:>6.3f} "
              f"{int(r['n_long']):>3} {r['pct_dropped']:>4.0f}% "
              f"{'Y' if r['guard_fired'] else '':>5}")

    # Drawdown series
    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (quarterly):")
    for i, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " <- MAX DD" if abs(val - m["max_dd"]) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{marker}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def save_results(results: dict) -> None:
    records  = results["period_records"]
    audit    = results["universe_audit"]
    coverage = results["coverage_records"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "asset_growth_period_returns.csv", index=False)
    pd.DataFrame(audit).to_csv(RESULTS / "asset_growth_universe_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "asset_growth_coverage_audit.csv", index=False)
    sector_df = pd.DataFrame(
        [{"sic_major_group": k, "contribution": v}
         for k, v in results.get("sector_contrib", {}).items()]
    ).sort_values("contribution", ascending=False)
    sector_df.to_csv(RESULTS / "asset_growth_sector_contribution.csv", index=False)
    print(f"\n[Saved] Results → {RESULTS}/")


def log_registry(results: dict) -> None:
    records = results["period_records"]
    if not records:
        return

    df           = pd.DataFrame(records)
    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()
    m            = compute_metrics(ls_series)
    m_gross      = compute_metrics(gross_series)
    beta_stats   = compute_market_beta(ls_series, mkt_series)

    per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
    dsr_thr       = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr    = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) else False

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    annual_cost_drag = (float(df["ls_ba_cost"].mean()) + float(df["ls_borrow_cost"].mean())) * 4 * 10_000
    cov = pd.DataFrame(results["coverage_records"])
    total_univ = int(cov["n_universe"].sum())
    total_sig  = int(cov["n_signal"].sum())
    total_fb   = int(cov["n_fallback"].sum())
    overall_fb_frac = 100 * total_fb / max(total_sig, 1)

    sign_violations  = int(results["sign_violations"])
    guard_fire_count = int(results["guard_fire_count"])
    n_periods_run    = len(df)

    verdict_bits = []
    verdict_bits.append("CLEARS DSR" if clears_dsr else "FAILS DSR")
    if sign_violations > 0:
        verdict_bits.append(f"SIGN AUDIT FAILED ({sign_violations} periods)")
    if overall_fb_frac > 25:
        verdict_bits.append(f"FALLBACK USAGE {overall_fb_frac:.0f}% (>25% threshold)")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "assetgrowth_S5",
        "hypothesis":           (
            "YoY total-asset growth: firms that aggressively grow the balance "
            "sheet subsequently underperform. Long bottom quintile (lowest "
            "asset growth), short top quintile (highest asset growth). "
            "Independent family — no overlap with S3/quality, S1-S4/IVOL-MAX, "
            "or S6-S9/momentum."
        ),
        "data_source":          "Tiingo (raw close for returns) + SEC XBRL (Assets tag, PIT)",
        "status":               "completed_in_sample",
        "trial_number":         12,
        "trial_note":           "Independent trial #12. Not a family member of any prior trial.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "quarterly",
        "rebalance_note":       "Pre-registered single frequency, no sweep (S3 lesson).",
        "portfolio_note":       (
            f"Beta-neutral (EW market proxy, 60-day adjClose). Name-count guard "
            f"(quintile leg<20 -> terciles) fired {guard_fire_count}/{n_periods_run} periods."
        ),
        "signal_note":          (
            "AssetGrowth = (Assets_t - Assets_t-4q)/Assets_t-4q, filed <= t; "
            f"fallback prior-year window used for {overall_fb_frac:.1f}% of signals."
        ),
        "ann_return_net":       m["ann_ret"],
        "ann_return_gross":     m_gross["ann_ret"],
        "annual_cost_drag_bps": annual_cost_drag,
        "mean_turnover":        float(df["avg_turnover"].mean()),
        "sharpe":               m["sharpe"],
        "calmar":               m["calmar"],
        "max_drawdown":         m["max_dd"],
        "t_stat":               m["t_stat"],
        "market_beta":          beta_stats["beta"],
        "market_beta_tstat":    beta_stats["beta_t"],
        "mean_long_scale":      float(df["long_scale"].mean()),
        "mean_short_scale":     float(df["short_scale"].mean()),
        "mean_beta_long":       float(df["mean_beta_long"].mean()),
        "mean_beta_short":      float(df["mean_beta_short"].mean()),
        "mean_ic":              mean_ic,
        "ic_t_stat":            ic_tstat,
        "win_rate":             m["win_rate"],
        "n_periods":            m["n"],
        "per_period_sr":        per_period_sr,
        "dsr_threshold":        dsr_thr,
        "clears_dsr":           clears_dsr,
        "mean_pct_dropped":     float(df["pct_dropped"].mean()),
        "adj_fallback_pct":     overall_fb_frac,
        "notes":                verdict,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "assetgrowth_S5"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] assetgrowth_S5 logged (trial #12) → {REGISTRY}")
    print(f"[Verdict] {verdict}")


# =============================================================================
# SECTION 16: Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S5: ASSET GROWTH ANOMALY  (independent trial #12)")
    print("Signal: YoY Asset Growth = (Assets_t - Assets_t-4q)/Assets_t-4q (SEC XBRL, PIT)")
    print("Long BOTTOM 20% growth | Short TOP 20% growth | Beta-neutral")
    print("Quarterly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 24 periods")
    print(f"DSR: N_TRIALS={N_TRIALS} | independent family, not folded into any prior trial")
    print("=" * 78)

    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = load_prefilter()
    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle (beta layer)...")
    preload_all_adj_prices(survivor_tickers)

    print(f"\n[Step 4] Loading Asset-Growth XBRL facts ({len(survivor_ciks):,} CIKs)...")
    preload_ag_facts(survivor_ciks)

    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} quarterly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    print_results(results)
    save_results(results)
    log_registry(results)


if __name__ == "__main__":
    main()
