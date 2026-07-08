"""
S7: IVOL-Conditional Value (double-sort)  — independent trial #13
====================================================================
This is a conditional double-sort, NOT a four-way sort against the full
universe. Both cutoffs are pre-registered here, before any result is
computed, per the falsification risk this signal was flagged for.

STEP 1 — IVOL sort (reused verbatim from S1/S-IVOL):
  IVOL = sqrt(RSS / (n-2)) of daily r_i = alpha + beta * r_mkt + eps over a
  rolling 60-trading-day window ending at t. Market proxy = cap-weighted
  daily return of the PIT universe (weights fixed at rebalance-date mcap).
  Signal layer: adjClose (falls back to raw close, logged). This is the
  IDENTICAL function used by S1 — same window, same market proxy, same
  fallback behavior — reused, not reimplemented, so the IVOL leg of this
  trial cannot silently diverge from S1's definition.
  High-IVOL subset = TOP TERCILE (top 33%) of IVOL each period. FIXED.

STEP 2 — Value sort, WITHIN the high-IVOL subset only:
  Value metric = E/P = TTM NetIncomeLoss / market cap (higher = cheaper =
  more earnings per dollar of price). TTM = sum of the 4 most-recently-
  FILED quarterly periods (filed <= t), annual-period fallback when <4
  quarters available — identical TTM convention to S3-GP's GP/A construction
  and S5's Assets lookup (same filing-date PIT gating). FIXED before any
  result is computed: earnings yield, not book-to-market.
  Within the high-IVOL subset: long TOP TERCILE of value (cheapest), short
  BOTTOM TERCILE of value (most expensive). FIXED.

Net portfolio: long = high-IVOL + cheap.  short = high-IVOL + expensive.
Both legs drawn EXCLUSIVELY from the high-IVOL subset.

Name-count guard (VALUE CUT ONLY): if the natural value-tercile leg within
the high-IVOL subset would hold fewer than 15 names, widen the VALUE cut
to HALVES (median split) within the SAME high-IVOL subset for that period
only. The IVOL cut (top tercile) is NEVER touched by this guard — logged,
and flagged as a universe-size concern (not a reason to loosen the
pre-registered cutoffs) if it fires on >20% of periods.

UNIVERSE & REBALANCE: identical to S1/S3 — PIT $100M-2B mcap, ADTV >=
$1M/day, price >= $1, survivorship-bias-free. Quarterly rebalance, 1-quarter
hold, pre-registered to match S1/S3 cadence for comparability.

Beta-neutral leg sizing: reuses the SAME 60-day OLS beta computed inside
the IVOL regression (no extra computation) — identical convention to S1.

Trial #13: new, independent DSR family. Borrows S1's IVOL computation but
tests a distinct interaction hypothesis (value effect concentrated in
high-IVOL names) — not folded into the S1/S4 IVOL-MAX family, not Quality,
not Momentum. DSR recomputed fresh at N=13.

Secondary, informational-only comparison: a standalone value-only sort
(same E/P metric, full universe, no IVOL conditioning, top/bottom quintile,
beta-neutral using the same reused beta) is computed alongside for context.
This does NOT change the primary construction and is not itself a trial.

No-survivorship guarantee: delisted tickers retained with returns to last
trading day; universe rebuilt PIT from SEC filings each quarter.

Namespacing: research/ivol_value/ — fully separate cache/results from any
concurrently running trial.
"""

import json
import os
import pickle
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

MGMT_PIT  = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE     = MGMT_PIT / "cache"                                # shared, read-only
OWN_CACHE = Path(__file__).resolve().parent / "cache"          # S7-private
RESULTS   = Path(__file__).resolve().parent / "results"
REGISTRY  = Path(__file__).resolve().parents[1] / "trial_registry.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S1/S3/S5) ─────────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── IVOL signal parameters (identical to S1) ─────────────────────────────────
IVOL_WINDOW_DAYS  = 60
IVOL_MIN_DAYS     = 40
IVOL_LOOKBACK_CAL = 95

# ── Double-sort cutoffs — PRE-REGISTERED, fixed before any result exists ────
IVOL_TERCILE_FRAC   = 1.0 / 3.0   # high-IVOL subset = top tercile of IVOL
VALUE_TERCILE_FRAC  = 1.0 / 3.0   # long/short legs = top/bottom tercile of value
VALUE_GUARD_MIN_LEG = 15          # if value-tercile leg < 15 -> widen to halves
STANDALONE_VALUE_LEG_FRAC = 0.20  # secondary comparison: standard quintile convention
STANDALONE_MIN_LEG        = 20

# ── Value metric: TTM flow concept + PIT dedup thresholds (same as S3-GP) ───
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
TTM_QUARTERS  = 4
QTR_MIN_DAYS  = 75
QTR_MAX_DAYS  = 115
ANN_MIN_DAYS  = 300
ANN_MAX_DAYS  = 400
VALUE_NEEDED  = set(NET_INCOME_CONCEPTS) | {
    "CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"
}

# ── Backtest dates ────────────────────────────────────────────────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 24 periods

# ── Cost model (identical to S1/S3/S5) ───────────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR: new, independent family ─────────────────────────────────────────────
TRIAL_NUMBER = 13
N_TRIALS     = 13


# =============================================================================
# SECTION 2: Tiingo / SEC helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
        return df
    raise FileNotFoundError(f"Not found: {path}. Run mgmt_pit/run.py first.")


def load_prefilter() -> pd.DataFrame:
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run mgmt_pit/run.py first.")
    df = pd.read_csv(path)
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors.")
    return df


# =============================================================================
# SECTION 3: Raw price cache (RETURN layer)
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
# SECTION 4: AdjClose cache (SIGNAL layer — IVOL only)
# =============================================================================

_ADJ_PKL      = CACHE / "adjclose_all.pkl"
_ADJ_CACHE:   dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str] = set()
_ADJ_EMPTY    = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if not _ADJ_PKL.exists():
        raise FileNotFoundError(f"Missing {_ADJ_PKL}. Run research/ivol/run.py first.")
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
# SECTION 5: SEC facts — shares (reuse S9/S1's trimmed cache) + value (own cache)
# =============================================================================

_SHARES_CACHE: dict[int, Optional[dict]] = {}
_SHARES_TRIMMED = CACHE / "sec_facts_trimmed.json"
_SHARES_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]


def preload_shares_facts(ciks: list[int]) -> None:
    t0 = time.time()
    if _SHARES_TRIMMED.exists():
        with open(_SHARES_TRIMMED) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _SHARES_CACHE[int(k)] = v
        loaded = sum(1 for v in _SHARES_CACHE.values() if v is not None)
        print(f"[SharesFacts] {loaded:,}/{len(ciks):,} loaded from trimmed cache in {time.time()-t0:.1f}s")
        return
    raise FileNotFoundError(f"Missing {_SHARES_TRIMMED}. Run research/momentum_jt/run.py first.")


def _get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    facts = _SHARES_CACHE.get(cik)
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
            for e in entries:
                filed = e.get("filed"); end = e.get("end"); val = e.get("val")
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


# ── Value-specific trimmed cache (NetIncomeLoss/ProfitLoss) — S7-private ─────
_VALUE_FACTS_CACHE: dict[int, Optional[dict]] = {}
_VALUE_FACTS_PKL   = OWN_CACHE / "value_facts_trimmed.json"


def preload_value_facts(ciks: list[int]) -> None:
    t0 = time.time()
    if _VALUE_FACTS_PKL.exists():
        with open(_VALUE_FACTS_PKL) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _VALUE_FACTS_CACHE[int(k)] = v
        missing = [c for c in ciks if c not in _VALUE_FACTS_CACHE]
        if missing:
            _load_value_facts_from_files(missing, save=False)
        loaded = sum(1 for v in _VALUE_FACTS_CACHE.values() if v is not None)
        print(f"[ValueFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")
        return
    print(f"[ValueFacts] First run — loading {len(ciks):,} full fact files...")
    _load_value_facts_from_files(ciks, save=True)
    loaded = sum(1 for v in _VALUE_FACTS_CACHE.values() if v is not None)
    print(f"[ValueFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


def _load_value_facts_from_files(ciks: list[int], save: bool = True) -> None:
    for cik in ciks:
        p = CACHE / "sec_facts" / f"{cik}.json"
        if not p.exists():
            _VALUE_FACTS_CACHE[cik] = None
            continue
        try:
            with open(p) as f:
                full = json.load(f)
            trimmed: dict = {"facts": {"us-gaap": {}}}
            ug = full.get("facts", {}).get("us-gaap", {})
            for concept in VALUE_NEEDED:
                if concept in ug:
                    trimmed["facts"]["us-gaap"][concept] = ug[concept]
            _VALUE_FACTS_CACHE[cik] = trimmed
        except Exception:
            _VALUE_FACTS_CACHE[cik] = None
    if save:
        print("[ValueFacts] Saving value facts trimmed cache...")
        serializable = {str(k): v for k, v in _VALUE_FACTS_CACHE.items()}
        with open(_VALUE_FACTS_PKL, "w") as f:
            json.dump(serializable, f)


def _period_days(start_str: str, end_str: str) -> int:
    try:
        s = date.fromisoformat(start_str)
        e = date.fromisoformat(end_str)
        return (e - s).days
    except Exception:
        return 0


def _get_ttm_flow(facts: Optional[dict], concept_list: list[str], as_of_str: str) -> tuple:
    """
    TTM value for a flow concept (here: NetIncomeLoss). Identical convention
    to S3-GP's _get_ttm_flow: sum of 4 most-recently-FILED, non-overlapping
    quarterly periods (75-115 days); falls back to the most recent annual
    period (300-400 days) when <4 quarters are available.
    Returns (value, method) where method in {"quarterly", "annual", None}.
    """
    if facts is None:
        return None, None
    for concept in concept_list:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue

        quarterly: list = []
        annual:    list = []
        for entries in units_dict.values():
            for e in entries:
                filed = e.get("filed"); end = e.get("end")
                val   = e.get("val");   start = e.get("start")
                if filed is None or end is None or val is None or start is None:
                    continue
                if filed > as_of_str:
                    continue
                days = _period_days(start, end)
                if QTR_MIN_DAYS <= days <= QTR_MAX_DAYS:
                    quarterly.append((filed, end, start, float(val)))
                elif ANN_MIN_DAYS <= days <= ANN_MAX_DAYS:
                    annual.append((filed, end, start, float(val)))

        if quarterly:
            by_end: dict = {}
            for filed, end, start, val in quarterly:
                if end not in by_end or filed > by_end[end][0]:
                    by_end[end] = (filed, end, start, val)
            sorted_q = sorted(by_end.values(), key=lambda x: x[1], reverse=True)
            selected: list = []
            for q in sorted_q:
                if len(selected) >= TTM_QUARTERS:
                    break
                q_start, q_end = q[2], q[1]
                overlap = any(
                    not (q_end <= s[2] or q_start >= s[1]) for s in selected
                )
                if not overlap:
                    selected.append(q)
            if len(selected) == TTM_QUARTERS:
                return sum(q[3] for q in selected), "quarterly"

        if annual:
            by_end_a: dict = {}
            for filed, end, start, val in annual:
                if end not in by_end_a or filed > by_end_a[end][0]:
                    by_end_a[end] = (filed, end, start, val)
            best_a = max(by_end_a.values(), key=lambda x: (x[0], x[1]))
            return best_a[3], "annual"
    return None, None


# =============================================================================
# SECTION 6: Universe builder (identical to S1/S3/S5)
# =============================================================================

def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
) -> pd.DataFrame:
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
        shares = _get_shares_pit(cik, rebalance_date)
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
# SECTION 7: IVOL signal — REUSED VERBATIM FROM S1/S-IVOL
# =============================================================================

def compute_ivol_signals(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> tuple:
    """
    Identical algorithm to research/ivol/run.py's compute_ivol_signals:
      1. adjClose (raw close fallback) over [t-95cal, t].
      2. Cap-weighted market return (weights = rebalance-date mcap).
      3. Per-ticker OLS r_i = alpha + beta*r_mkt + eps over last 60 trading days.
      4. IVOL = sqrt(RSS/(n-2)).
    Returns ivol_df[ticker, ivol, ivol_signal(=-ivol), beta], n_fallback, n_total.
    ivol_signal kept for fidelity with S1's sign convention; "ivol" (positive
    magnitude) is what THIS trial's tercile cut actually sorts on, to avoid
    any sign-confusion in the double-sort logic.
    """
    lookback_start = rebalance_date - pd.Timedelta(days=IVOL_LOOKBACK_CAL)

    price_series: dict[str, pd.Series] = {}
    raw_fallback: set[str] = set()

    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj = _adj_slice(ticker, lookback_start, rebalance_date)
        if len(adj) >= IVOL_MIN_DAYS + 1:
            price_series[ticker] = adj.set_index("date")["adjClose"]
            continue
        raw_df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
        if raw_df.empty:
            continue
        window = raw_df[(raw_df["date"] >= lookback_start) & (raw_df["date"] <= rebalance_date)]
        if len(window) >= IVOL_MIN_DAYS + 1:
            price_series[ticker] = window.set_index("date")["close"]
            raw_fallback.add(ticker)

    n_total    = len(price_series)
    n_fallback = len(raw_fallback)
    if n_total < 10:
        return pd.DataFrame(columns=["ticker", "ivol", "ivol_signal", "beta"]), n_fallback, n_total

    price_mat = pd.DataFrame(price_series).sort_index()
    ret_mat   = price_mat.pct_change().iloc[1:]
    ret_mat   = ret_mat.tail(IVOL_WINDOW_DAYS)

    valid_counts = ret_mat.notna().sum()
    ret_mat = ret_mat.loc[:, valid_counts >= IVOL_MIN_DAYS]
    if ret_mat.shape[1] < 5:
        return pd.DataFrame(columns=["ticker", "ivol", "ivol_signal", "beta"]), n_fallback, n_total

    mcap_s  = universe_df.set_index("ticker")["mcap"]
    w       = mcap_s.reindex(ret_mat.columns).fillna(0.0)
    total_w = w.sum()
    w_norm  = w / total_w if total_w > 0 else pd.Series(1.0 / len(ret_mat.columns), index=ret_mat.columns)

    avail      = ret_mat.notna()
    weighted_r = ret_mat.multiply(w_norm, axis=1)
    denom      = avail.multiply(w_norm, axis=1).sum(axis=1)
    mkt_ret    = weighted_r.sum(axis=1) / denom.replace(0, np.nan)

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
        ivol = float(np.sqrt(np.dot(eps, eps) / (n - 2)))
        if not np.isfinite(ivol) or ivol <= 0:
            continue
        records.append({"ticker": ticker, "ivol": ivol, "ivol_signal": -ivol, "beta": float(coeffs[1])})

    ivol_df = (pd.DataFrame(records) if records
               else pd.DataFrame(columns=["ticker", "ivol", "ivol_signal", "beta"]))
    return ivol_df, n_fallback, n_total


# =============================================================================
# SECTION 8: Value signal (E/P = TTM NetIncomeLoss / mcap)
# =============================================================================

def compute_value_signals(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> tuple:
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    records = []
    n_direct = n_annual_fb = n_no_data = 0

    for _, row in universe_df.iterrows():
        cik = int(row["cik"])
        facts = _VALUE_FACTS_CACHE.get(cik)
        ni_ttm, method = _get_ttm_flow(facts, NET_INCOME_CONCEPTS, as_of_str)
        if ni_ttm is None:
            n_no_data += 1
            continue
        if method == "annual":
            n_annual_fb += 1
        else:
            n_direct += 1
        ep = ni_ttm / row["mcap"]
        if not np.isfinite(ep):
            n_no_data += 1
            continue
        records.append({"ticker": row["ticker"], "value_signal": ep, "value_method": method})

    value_df = (pd.DataFrame(records) if records
                else pd.DataFrame(columns=["ticker", "value_signal", "value_method"]))
    coverage = {
        "n_universe": len(universe_df), "n_signal": len(value_df),
        "n_direct": n_direct, "n_annual_fb": n_annual_fb, "n_no_data": n_no_data,
        "pct_dropped": 100 * (len(universe_df) - len(value_df)) / max(len(universe_df), 1),
    }
    return value_df, coverage


# =============================================================================
# SECTION 9: Return calculation (identical pattern)
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame, hold_start: pd.Timestamp, hold_end: pd.Timestamp,
    delisted_map: dict,
) -> pd.DataFrame:
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
    df["entry_price"] = entry_prices
    df["exit_price"]  = exit_prices
    df["raw_return"]  = raw_returns
    df["delisted"]    = delisted_flags
    return df


# =============================================================================
# SECTION 10: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_double_sort_portfolio(merged: pd.DataFrame) -> dict:
    """
    Stage 1: high-IVOL subset = top tercile of `ivol` (highest actual vol).
    Stage 2: within that subset, long = top tercile of value_signal (cheap),
             short = bottom tercile (expensive). Guard widens Stage-2 cut to
             halves (never Stage 1) if a leg would hold <15 names.

    Returns a dict with longs/shorts/weights/scales plus subset sizes at
    every stage, for the "report subset sizes at each stage" requirement.
    """
    df = merged.dropna(subset=["ivol", "value_signal", "raw_return", "beta"]).copy()
    n_universe = len(df)
    empty = {
        "longs": [], "shorts": [], "lw": {}, "sw": {},
        "long_scale": 1.0, "short_scale": 1.0,
        "mean_beta_long": np.nan, "mean_beta_short": np.nan,
        "n_universe": n_universe, "n_high_ivol": 0, "guard_fired": False,
    }
    if n_universe < 12:
        return empty

    # ── Stage 1: high-IVOL tercile (top third by actual ivol) ────────────────
    n_high_ivol = max(int(np.ceil(n_universe * IVOL_TERCILE_FRAC)), 1)
    high_ivol_df = df.sort_values("ivol", ascending=False).iloc[:n_high_ivol].copy()

    if len(high_ivol_df) < 6:
        empty["n_high_ivol"] = len(high_ivol_df)
        return empty

    # ── Stage 2: value tercile WITHIN the high-IVOL subset ───────────────────
    n_sub = len(high_ivol_df)
    leg_tercile = int(np.ceil(n_sub * VALUE_TERCILE_FRAC))
    guard_fired = leg_tercile < VALUE_GUARD_MIN_LEG
    leg = int(np.ceil(n_sub / 2.0)) if guard_fired else leg_tercile
    leg = min(leg, n_sub // 2)
    if leg < 1:
        empty["n_high_ivol"] = n_sub
        return empty

    ranked = high_ivol_df.sort_values("value_signal", ascending=False)  # high E/P = cheap first
    longs  = ranked.iloc[:leg]["ticker"].tolist()     # cheap  = long
    shorts = ranked.iloc[-leg:]["ticker"].tolist()    # expensive = short

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    beta_map = high_ivol_df.set_index("ticker")["beta"].to_dict()
    mean_beta_long  = float(np.nanmean([beta_map.get(t, np.nan) for t in longs]))
    mean_beta_short = float(np.nanmean([beta_map.get(t, np.nan) for t in shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return {
        "longs": longs, "shorts": shorts, "lw": lw, "sw": sw,
        "long_scale": long_scale, "short_scale": short_scale,
        "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
        "n_universe": n_universe, "n_high_ivol": n_sub, "guard_fired": guard_fired,
    }


def build_standalone_value_portfolio(merged: pd.DataFrame) -> dict:
    """
    SECONDARY / INFORMATIONAL ONLY: plain value sort, full universe, no IVOL
    conditioning. Top/bottom quintile, min 20/leg (standard single-sort
    convention used elsewhere in this registry, e.g. S3-GP). Beta-neutral
    using the same reused IVOL-regression beta.
    """
    df = merged.dropna(subset=["value_signal", "raw_return", "beta"]).copy()
    n = len(df)
    empty = {"longs": [], "shorts": [], "lw": {}, "sw": {}, "long_scale": 1.0, "short_scale": 1.0}
    if n < 40:
        return empty
    leg = max(int(np.ceil(n * STANDALONE_VALUE_LEG_FRAC)), STANDALONE_MIN_LEG)
    leg = min(leg, n // 2)
    ranked = df.sort_values("value_signal", ascending=False)
    longs  = ranked.iloc[:leg]["ticker"].tolist()
    shorts = ranked.iloc[-leg:]["ticker"].tolist()
    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}
    beta_map = df.set_index("ticker")["beta"].to_dict()
    mbl = float(np.nanmean([beta_map.get(t, np.nan) for t in longs]))
    mbs = float(np.nanmean([beta_map.get(t, np.nan) for t in shorts]))
    denom = mbl + mbs
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mbs / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mbl / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0
    return {"longs": longs, "shorts": shorts, "lw": lw, "sw": sw,
            "long_scale": long_scale, "short_scale": short_scale}


def _leg_return_detail(returns_df: pd.DataFrame, tickers: list, weights: dict,
                        is_short: bool, turnover: float) -> tuple:
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
    return (gross - total_cost), gross, ba_cost, borrow_cost


def _leg_return_simple(returns_df: pd.DataFrame, tickers: list, weights: dict, is_short: bool) -> float:
    """Simplified leg return (fixed 50% turnover assumption) for the secondary standalone comparison."""
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (MCAP_MIN + MCAP_MAX) / 2
    gross = cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross += w * r
        cost  += 0.5 * 2.0 * _spread(mc) * w
        if is_short:
            cost += (_borrow(mc) / 4.0) * w
    return (-gross - cost) if is_short else (gross - cost)


# =============================================================================
# SECTION 11: Backtest engine
# =============================================================================

def run_backtest(tiingo_tickers: pd.DataFrame, prefilter_df: pd.DataFrame) -> dict:
    t_total = time.time()

    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list   = []
    coverage_records: list = []
    subset_records: list   = []
    sector_contrib: dict   = {}
    sign_violations: int   = 0
    guard_fire_count: int  = 0

    prev_long_set:  set = set()
    prev_short_set: set = set()

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        hold_end = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} → {hold_end.date()}")

        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        ivol_df, n_fallback, n_computed = compute_ivol_signals(univ, t)
        value_df, val_cov = compute_value_signals(univ, t)

        merged = univ.merge(ivol_df, on="ticker", how="left").merge(
            value_df, on="ticker", how="left")

        try:
            returns_df = compute_returns(merged, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        coverage_records.append({
            "rebalance_date": t, "n_universe": len(univ),
            "n_ivol": int(returns_df["ivol"].notna().sum()),
            "n_value": val_cov["n_signal"], "n_value_direct": val_cov["n_direct"],
            "n_value_annual_fb": val_cov["n_annual_fb"], "n_value_no_data": val_cov["n_no_data"],
            "pct_value_dropped": val_cov["pct_dropped"],
        })

        # ── Primary double-sort ──────────────────────────────────────────────
        port = build_double_sort_portfolio(returns_df)
        longs, shorts = port["longs"], port["shorts"]

        subset_records.append({
            "rebalance_date": t, "n_universe": port["n_universe"],
            "n_high_ivol": port["n_high_ivol"],
            "n_long": len(longs), "n_short": len(shorts),
            "guard_fired": port["guard_fired"],
        })

        if not longs:
            print(f"  [SKIP] insufficient names for double-sort (universe={len(returns_df)}, "
                  f"high_ivol={port['n_high_ivol']})")
            continue
        if port["guard_fired"]:
            guard_fire_count += 1
            print(f"  [GUARD] value tercile <15 within high-IVOL subset → widened to halves "
                  f"(leg={len(longs)})")

        lw, sw = port["lw"], port["sw"]
        long_scale, short_scale = port["long_scale"], port["short_scale"]
        mean_beta_long, mean_beta_short = port["mean_beta_long"], port["mean_beta_short"]

        # ── Sign-convention audit ────────────────────────────────────────────
        val_map = returns_df.set_index("ticker")["value_signal"]
        mean_val_long  = float(val_map.reindex(longs).mean())
        mean_val_short = float(val_map.reindex(shorts).mean())
        sign_ok = mean_val_long > mean_val_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg value ({mean_val_long:.4f}) "
                  f"<= short-leg value ({mean_val_short:.4f}) ***")

        if prev_long_set:
            long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1)
            short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0
        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_detail(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw
        ls_cost  = ls_ba + ls_bw

        # ── Secondary: standalone value-only comparison (informational) ─────
        sv_port = build_standalone_value_portfolio(returns_df)
        if sv_port["longs"]:
            sv_long  = _leg_return_simple(returns_df, sv_port["longs"],  sv_port["lw"], is_short=False)
            sv_short = _leg_return_simple(returns_df, sv_port["shorts"], sv_port["sw"], is_short=True)
            sv_ls    = sv_port["long_scale"] * sv_long + sv_port["short_scale"] * sv_short
        else:
            sv_ls = np.nan

        # ── Sector contribution (concentration check) ────────────────────────
        sic_map = returns_df.set_index("ticker")["sic"].to_dict()
        ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
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

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        # ── IC: within-high-IVOL value rank vs forward return ────────────────
        high_ivol_tickers = set(returns_df.sort_values("ivol", ascending=False)
                                 .iloc[:port["n_high_ivol"]]["ticker"])
        sub = returns_df[returns_df["ticker"].isin(high_ivol_tickers)]
        valid_sig = sub.dropna(subset=["value_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["value_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  U={port['n_universe']} HighIVOL={port['n_high_ivol']} nL={len(longs)} nS={len(shorts)} "
              f"L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} SV={sv_ls:+.2%} "
              f"to={avg_turnover:.0%} IC={ic:.3f} valL={mean_val_long:.4f} valS={mean_val_short:.4f}")

        period_records.append({
            "rebalance_date": t, "hold_end": hold_end,
            "n_universe": port["n_universe"], "n_high_ivol": port["n_high_ivol"],
            "n_delisted": n_delisted,
            "long_ret": long_net, "short_ret": short_net,
            "ls_ret": ls_net, "ls_gross": ls_gross, "ls_cost": ls_cost,
            "ls_ba_cost": ls_ba, "ls_borrow_cost": ls_bw,
            "ls_ret_standalone_value": sv_ls,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
            "avg_turnover": avg_turnover, "mkt_ret": mkt_ret, "ic": ic,
            "n_long": len(longs), "n_short": len(shorts), "guard_fired": port["guard_fired"],
            "mean_val_long": mean_val_long, "mean_val_short": mean_val_short, "sign_ok": sign_ok,
        })

    elapsed = time.time() - t_total
    return {
        "period_records": period_records, "coverage_records": coverage_records,
        "subset_records": subset_records, "sector_contrib": sector_contrib,
        "sign_violations": sign_violations, "guard_fire_count": guard_fire_count,
        "elapsed_s": elapsed,
    }


# =============================================================================
# SECTION 12: Metrics
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
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
    se     = (np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 13: Output & registry
# =============================================================================

def print_results(results: dict) -> None:
    records  = results["period_records"]
    coverage = results["coverage_records"]
    subsets  = results["subset_records"]
    elapsed  = results["elapsed_s"]
    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df   = pd.DataFrame(records)
    cov  = pd.DataFrame(coverage)
    sub  = pd.DataFrame(subsets)

    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    sv_series    = df["ls_ret_standalone_value"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m       = compute_metrics(ls_series)
    m_gross = compute_metrics(gross_series)
    m_sv    = compute_metrics(sv_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (bool(per_period_sr > dsr_thr)
                  if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False)

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)

    mean_ba_bps = float(df["ls_ba_cost"].mean())     * 4 * 10_000
    mean_bw_bps = float(df["ls_borrow_cost"].mean()) * 4 * 10_000
    annual_cost_drag = mean_ba_bps + mean_bw_bps
    mean_turnover    = float(df["avg_turnover"].mean())

    guard_fire_count = int(results["guard_fire_count"])
    sign_violations  = int(results["sign_violations"])
    n_periods_run    = len(df)

    total_univ  = int(cov["n_universe"].sum())
    total_val   = int(cov["n_value"].sum())
    total_direct = int(cov["n_value_direct"].sum())
    total_annfb  = int(cov["n_value_annual_fb"].sum())
    overall_drop = 100 * (total_univ - total_val) / max(total_univ, 1)

    sep = "=" * 78

    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT")
    print(sep)
    print(f"  Periods where long-leg value <= short-leg value (violation): "
          f"{sign_violations}/{n_periods_run}")
    print("  PASS — long leg (cheap) always has higher E/P than short leg (expensive)."
          if sign_violations == 0 else
          "  *** FAIL — sign convention broken. Results may be inverted. ***")

    print(f"\n{sep}")
    print("VALUE-METRIC COVERAGE (TTM NetIncomeLoss / mcap)")
    print(sep)
    print(f"  Total stock-period slots: {total_univ:,}")
    print(f"  With valid value signal:  {total_val:,} ({100*total_val/max(total_univ,1):.1f}%)")
    print(f"    — quarterly TTM (direct): {total_direct:,}")
    print(f"    — annual fallback:        {total_annfb:,}")
    print(f"  Dropped (no signal): {total_univ-total_val:,} ({overall_drop:.1f}%)")

    print(f"\n{sep}")
    print("SUBSET SIZES AT EACH SORT STAGE")
    print(sep)
    print(f"  {'Date':<12} {'Universe':>9} {'HighIVOL':>9} {'Long':>6} {'Short':>6} {'Guard':>6}")
    print(f"  {'-'*12} {'-'*9} {'-'*9} {'-'*6} {'-'*6} {'-'*6}")
    for _, r in sub.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<12} {int(r['n_universe']):>9} "
              f"{int(r['n_high_ivol']):>9} {int(r['n_long']):>6} {int(r['n_short']):>6} "
              f"{'Y' if r['guard_fired'] else '':>6}")
    print(f"\n  Name-count guard (value tercile<15 within high-IVOL -> halves) fired: "
          f"{guard_fire_count}/{n_periods_run} periods ({100*guard_fire_count/max(n_periods_run,1):.0f}%)")
    if guard_fire_count / max(n_periods_run, 1) > 0.20:
        print("  *** FLAG: guard fires on >20% of periods — universe-size concern for this "
              "construction (tercile-of-a-tercile is thin), NOT a reason to loosen the "
              "pre-registered cutoffs. ***")

    print(f"\n{sep}")
    print("S7: IVOL-CONDITIONAL VALUE — IN-SAMPLE RESULTS (independent trial #13)")
    print("High-IVOL top tercile -> long/short value tercile within it -> Beta-neutral -> Quarterly")
    print(sep)

    print(f"\n  *** INFORMATION COEFFICIENT (within high-IVOL subset only) ***")
    print(f"  Mean IC (Spearman, value vs ret)       : {mean_ic:>12.4f}")
    print(f"  IC t-stat                              : {ic_tstat:>12.3f}")

    print(f"\n  {'Metric':<46} {'Value':>14}")
    print(f"  {'-'*46} {'-'*14}")
    print(f"  {'Total cumulative return (net)':<46} {m['total_ret']:>13.2%}")
    print(f"  {'Annualized return (net)':<46} {m['ann_ret']:>13.2%}")
    print(f"  {'Annualized return (GROSS)':<46} {m_gross['ann_ret']:>13.2%}")
    print(f"  {'Cost drag (annualized bps)':<46} {annual_cost_drag:>13.0f}")
    print(f"    {'  bid-ask (bps/yr)':<44} {mean_ba_bps:>13.0f}")
    print(f"    {'  borrow  (bps/yr)':<44} {mean_bw_bps:>13.0f}")
    print(f"  {'Mean actual turnover per quarter':<46} {mean_turnover:>13.0%}")
    print(f"  {'Sharpe ratio (annualized, net)':<46} {m['sharpe']:>13.3f}")
    print(f"  {'CALMAR ratio':<46} {m['calmar']:>13.3f}")
    print(f"  {'Max drawdown':<46} {m['max_dd']:>13.2%}")
    print(f"  {'t-stat (mean quarterly net return)':<46} {m['t_stat']:>13.3f}")
    print(f"  {'Market beta (vs EW universe)':<46} {beta_stats['beta']:>13.3f}")
    print(f"    {'beta t-stat (should be ~0)':<44} {beta_stats['beta_t']:>13.3f}")
    print(f"  {'Win rate':<46} {m['win_rate']:>13.1%}")
    print(f"  {'Periods (N)':<46} {m['n']:>13d}")

    print(f"\n  --- DSR Check (independent family, N_trials={N_TRIALS}, quarterly SR units) ---")
    print(f"  {'Per-period Sharpe':<46} {per_period_sr:>13.4f}")
    print(f"  {'DSR threshold (per-quarter SR)':<46} {dsr_thr:>13.4f}")
    print(f"  {'Clears DSR':<46} {'YES' if clears_dsr else 'NO':>13}")

    print(f"\n{sep}")
    print("SECONDARY COMPARISON: standalone value sort (no IVOL conditioning, informational only)")
    print(sep)
    print(f"  {'Metric':<40} {'IVOL-Cond Value':>16} {'Standalone Value':>18}")
    print(f"  {'-'*40} {'-'*16} {'-'*18}")
    print(f"  {'Annualized return (net)':<40} {m['ann_ret']:>15.2%} {m_sv['ann_ret']:>17.2%}")
    print(f"  {'Sharpe (net)':<40} {m['sharpe']:>15.3f} {m_sv['sharpe']:>17.3f}")
    print(f"  {'Max drawdown':<40} {m['max_dd']:>15.2%} {m_sv['max_dd']:>17.2%}")
    print(f"  {'CALMAR':<40} {m['calmar']:>15.3f} {m_sv['calmar']:>17.3f}")
    delta_sharpe = m['sharpe'] - m_sv['sharpe'] if not (np.isnan(m['sharpe']) or np.isnan(m_sv['sharpe'])) else np.nan
    print(f"\n  IVOL-conditioning Sharpe delta vs standalone value: {delta_sharpe:+.3f}")
    print("  (Positive = IVOL-conditioning added value over plain value; informational only —"
          " this comparison does not alter the primary construction.)")

    abs_period_sum = float(ls_series.abs().sum())
    period_share = (ls_series.abs() / abs_period_sum) if abs_period_sum > 0 else ls_series * 0
    top_idx   = period_share.idxmax()
    top_share = float(period_share.loc[top_idx])
    top_date  = df.loc[top_idx, "rebalance_date"]
    print(f"\n{sep}")
    print("CONCENTRATION CHECK (single-episode artifact risk — small eligible pool here)")
    print(sep)
    print(f"  Largest single-quarter share of total |return| mass: "
          f"{top_date.date()} = {top_share:.1%}")
    print("  *** WARNING: single quarter dominates (>30%). ***" if top_share > 0.30
          else "  No single quarter dominates (<30% threshold).")

    sector_contrib = results.get("sector_contrib", {})
    if sector_contrib:
        total_abs = sum(abs(v) for v in sector_contrib.values())
        if total_abs > 0:
            top_sic, top_val = max(sector_contrib.items(), key=lambda kv: abs(kv[1]))
            top_sec_share = abs(top_val) / total_abs
            print(f"  Largest single SIC major-group contribution: "
                  f"SIC {top_sic:02d} = {top_sec_share:.1%} of total |contribution|")
            print("  *** WARNING: single sector dominates (>40%). ***" if top_sec_share > 0.40
                  else "  No single sector dominates (<40% threshold).")

    print(f"\n{sep}")
    print("QUARTERLY RETURNS LOG")
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'StdVal':>7} {'Cost':>6} {'TOver':>6} "
          f"{'IC':>6} {'nL':>3} {'nHiIV':>6} {'Guard':>5}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*6} {'-'*3} {'-'*6} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_ret_standalone_value']:>6.2%} "
              f"{r['ls_cost']:>5.2%} {r['avg_turnover']:>5.0%} "
              f"{r['ic']:>6.3f} {int(r['n_long']):>3} {int(r['n_high_ivol']):>6} "
              f"{'Y' if r['guard_fired'] else '':>5}")

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
    coverage = results["coverage_records"]
    subsets  = results["subset_records"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "s7_period_returns.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "s7_coverage_audit.csv", index=False)
    pd.DataFrame(subsets).to_csv(RESULTS / "s7_subset_sizes.csv", index=False)
    sector_df = pd.DataFrame(
        [{"sic_major_group": k, "contribution": v}
         for k, v in results.get("sector_contrib", {}).items()]
    ).sort_values("contribution", ascending=False)
    sector_df.to_csv(RESULTS / "s7_sector_contribution.csv", index=False)
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
    ic_tstat = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)

    annual_cost_drag = (float(df["ls_ba_cost"].mean()) + float(df["ls_borrow_cost"].mean())) * 4 * 10_000
    guard_fire_count = int(results["guard_fire_count"])
    sign_violations  = int(results["sign_violations"])
    n_periods_run    = len(df)

    verdict_bits = ["CLEARS DSR" if clears_dsr else "FAILS DSR"]
    if sign_violations > 0:
        verdict_bits.append(f"SIGN AUDIT FAILED ({sign_violations} periods)")
    if guard_fire_count / max(n_periods_run, 1) > 0.20:
        verdict_bits.append(f"GUARD FIRED {guard_fire_count}/{n_periods_run} (>20% — universe-size concern)")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "ivolvalue_S7",
        "hypothesis":           (
            "IVOL-conditional value: within the top-tercile high-IVOL subset, "
            "cheap (high E/P) names outperform expensive (low E/P) names. "
            "Double-sort: IVOL tercile fixed pre-registration, value tercile "
            "within it. Independent trial #13 — distinct interaction hypothesis "
            "from S1/S4 IVOL-MAX family, not Quality, not Momentum."
        ),
        "data_source":          "Tiingo (adjClose for IVOL, raw close for returns) + SEC XBRL (NetIncomeLoss TTM, PIT shares)",
        "status":               "completed_in_sample",
        "trial_number":         TRIAL_NUMBER,
        "trial_note":           "Independent trial #13. Borrows S1 IVOL computation but is a distinct interaction hypothesis.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "quarterly",
        "rebalance_note":       "Matches S1/S3 cadence for comparability.",
        "portfolio_note":       (
            f"Double-sort: high-IVOL top tercile -> value tercile within it (halves if "
            f"leg<15). Beta-neutral (reused IVOL-regression beta). Guard fired "
            f"{guard_fire_count}/{n_periods_run} periods."
        ),
        "signal_note":          (
            "Value = TTM NetIncomeLoss / mcap (E/P), filed <= t, quarterly TTM with "
            "annual fallback (same convention as S3-GP/S5)."
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
        "mean_ic":              mean_ic,
        "ic_t_stat":            ic_tstat,
        "win_rate":             m["win_rate"],
        "n_periods":            m["n"],
        "per_period_sr":        per_period_sr,
        "dsr_threshold":        dsr_thr,
        "clears_dsr":           clears_dsr,
        "notes":                verdict,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "ivolvalue_S7"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] ivolvalue_S7 logged (trial #{TRIAL_NUMBER}) → {REGISTRY}")
    print(f"[Verdict] {verdict}")


# =============================================================================
# SECTION 14: Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S7: IVOL-CONDITIONAL VALUE (double-sort)  — independent trial #13")
    print("Stage 1: high-IVOL top tercile (S1's exact IVOL computation)")
    print("Stage 2: value tercile within it — long cheap, short expensive")
    print("Quarterly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 24 periods")
    print(f"DSR: N_TRIALS={N_TRIALS} (new, independent family)")
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

    print(f"\n[Step 3b] Loading adjClose pickle (IVOL signal layer)...")
    preload_all_adj_prices(survivor_tickers)

    print(f"\n[Step 4] Loading SEC shares facts ({len(survivor_ciks):,} CIKs)...")
    preload_shares_facts(survivor_ciks)

    print(f"\n[Step 5] Loading Value (NetIncomeLoss) XBRL facts ({len(survivor_ciks):,} CIKs)...")
    preload_value_facts(survivor_ciks)

    print(f"\n[Step 6] Running backtest ({len(REBALANCE_DATES)} quarterly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    print_results(results)
    save_results(results)
    log_registry(results)


if __name__ == "__main__":
    main()
