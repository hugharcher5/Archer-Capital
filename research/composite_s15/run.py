"""
S15: Low-Turnover Composite of Validated-IC Signals  (Trial #19, N_TRIALS=19)
==============================================================================
NOT a re-tune of any prior trial. Combines three ALREADY-INDEPENDENTLY-TESTED
signals, each with real, correctly-signed IC despite failing as a standalone
strategy for three different reasons:
  - GP/A quarterly       (S3-Q,  trial #8):  IC +0.040 (t=2.28) — killed by cost drag
  - Residual reversal    (S8,    trial #14): IC +0.036 (t=3.40) — gross return itself negative
  - IVOL-conditioned value (S7,  trial #13): IC +0.036 (t=1.36) — weaker alone, but showed
                                              real diversification value vs plain value

Hypothesis: three distinct economic mechanisms (profitability level, short-term
mean-reversion, value-mispricing conditional on arbitrage difficulty) combined
into one portfolio could produce a genuinely tradeable Sharpe/CALMAR through
diversification, even though none works alone.

CONSTRUCTION (PRE-REGISTERED — no tuning after seeing results)
----------------------------------------------------------------------------
Universe: standard $100M-$2B PIT, survivorship-bias-free, RESTRICTED each
  quarter to the intersection of names with a valid value for ALL THREE
  component signals that quarter. A name missing any one signal is excluded
  from that quarter's universe entirely (not partially scored).

Composite score: cross-sectional rank-percentile (pandas .rank(pct=True)) of
  each of the three signals, computed independently each quarter on the
  intersected universe, then EQUAL-WEIGHTED average of the three percentiles.
  No optimization / no backtest-selected weights.

  Component 1 — GP/A:        raw_signal = TTM GrossProfit / Assets (higher=long).
                              Identical construction to gp/run.py (trial #8).
  Component 2 — ResidRev:    raw_signal = -(1-month residual return) from a
                              36-month (min 24mo) single-factor market-model
                              OLS regression, EW-survivor market factor.
                              Identical construction to residual_reversal/run.py
                              (trial #14), EXCEPT evaluated at QUARTERLY dates
                              instead of monthly (see deviation note below).
  Component 3 — IVOL-Value:  S7's double-sort (hard IVOL-tercile gate, then
                              value-tercile within it) has no natural
                              continuous scalar — it was never designed to
                              score the full universe. We define a NEW
                              continuous generalization that preserves the
                              interaction/conditional spirit S7 tested:
                                  ivol_value_raw = rank_pct(IVOL) * rank_pct(E/P)
                              i.e. the product of cross-sectional percentile
                              ranks of IVOL (60-day OLS residual vol, cap-
                              weighted market) and E/P (TTM NetIncomeLoss /
                              mcap). Both ranks lie in (0,1], so the product
                              rewards names that are BOTH high-IVOL AND cheap,
                              exactly the S7 hypothesis, but continuously and
                              over the FULL universe (no hard gate). This is
                              genuinely new methodology, not a re-tune of S7,
                              and is flagged as such throughout.
                              ivol_value_raw is then itself rank-percentiled
                              (consistent with the pre-registered rule that
                              all three components enter the composite as
                              rank-percentiles).

Pre-registered deviation flagged explicitly: S8's residual-reversal signal is
  natively monthly; reading it at quarterly dates is a valid mechanical reuse
  of compute_residual_reversal() (nothing in the computation is monthly-
  specific) but redefines what's captured — "1-month residual as of the last
  month of the quarter" rather than a full monthly-refreshed series — and
  loses S8's turnover-discipline carry mechanism (no carry needed here since
  the composite fully reforms every quarter, matching GP/A's cadence).

Portfolio: long top 20% composite score / short bottom 20%. Name-count guard
  (same convention as S5/S10/S11/asset_growth): leg = ceil(n*0.20); if that
  leg would hold <20 names, widen to TERCILES (ceil(n/3)) for that period
  only, logged, not a new pre-registered cutoff change.
Beta-neutral leg sizing: standardized on the 60-day-OLS-vs-EW-universe-
  adjClose beta (compute_betas(), the same construction used by GP/S8 and
  matching S7's IVOL-regression beta in form) — used identically for the
  composite AND for all three "standalone-alone" comparison portfolios below,
  so beta-neutral construction cannot itself explain any Sharpe difference.
Rebalance: quarterly, matching GP/A's cadence. 24 periods, 2015-01-01 to
  2021-01-01. adjClose for signal/beta computation, raw close for hold-period
  P&L (standard convention, identical to S3/S7/S8).
Cost model: identical linear-interp-by-mcap spread+borrow model as
  S3/S5/S7/S10 (quarterly divisor, no carry — full quarterly reform).

DIRECT COMPARISON (the actual point of this trial)
  Three "component-alone" portfolios (GP/A-alone, ResidRev-alone, IVOL-Value-
  alone) are recomputed on the EXACT SAME per-quarter intersected universe,
  same dates, same beta source, same cost model as the composite — NOT simply
  re-quoted from the original trial #8/#13/#14 registry rows (which used each
  signal's own broader universe, and for S8, monthly cadence). This isolates
  the diversification effect from any universe-composition or cadence
  difference. Original registry numbers are also printed alongside for
  reference. All four portfolios' Sharpe/CALMAR/beta reported side-by-side.

FALSIFICATION DIAGNOSTICS (pre-registered)
  1. Sign-convention audit: IC of composite_score and each raw component vs
     forward return, on the shared universe — confirm same sign as each
     component's original registry-logged IC.
  2. Concentration check: per period, share of total |dollar-weighted return
     contribution| (both legs pooled) coming from the single largest and top-3
     names in the composite portfolio.
  3. Pairwise correlation of the three component-alone quarterly L/S net
     return series, in-sample.

Registry key: compositeIC_S15. Logged regardless of outcome.
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

MGMT_PIT     = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE        = MGMT_PIT / "cache"                                            # shared, read-only
IVOL_VALUE_CACHE = Path(__file__).resolve().parents[1] / "ivol_value" / "cache"  # read-only reuse
OWN_CACHE    = Path(__file__).resolve().parent / "cache"                     # S15-private
RESULTS      = Path(__file__).resolve().parent / "results"
REGISTRY     = Path(__file__).resolve().parents[1] / "trial_registry.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S3/S5/S7/S8/S10) ──────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0

# ── GP/A parameters (identical to S3-GP) ─────────────────────────────────────
TTM_QUARTERS  = 4
QTR_MIN_DAYS  = 75
QTR_MAX_DAYS  = 115
ANN_MIN_DAYS  = 300
ANN_MAX_DAYS  = 400
GROSS_PROFIT_CONCEPTS = ["GrossProfit", "GrossProfitLoss"]
REVENUE_CONCEPTS = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
]
COGS_CONCEPTS = ["CostOfGoodsSold", "CostOfRevenue", "CostOfGoodsSoldAndServicesCost"]
SHARES_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]
GP_NEEDED = set(GROSS_PROFIT_CONCEPTS + REVENUE_CONCEPTS + COGS_CONCEPTS + ["Assets"] + SHARES_CONCEPTS)

# ── Value (E/P) parameters (identical to S7) ─────────────────────────────────
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
VALUE_NEEDED = set(NET_INCOME_CONCEPTS) | set(SHARES_CONCEPTS)

# ── IVOL parameters (identical to S1/S7) ──────────────────────────────────────
IVOL_WINDOW_DAYS  = 60
IVOL_MIN_DAYS     = 40
IVOL_LOOKBACK_CAL = 95

# ── Residual-reversal / market-factor parameters (identical to S8) ───────────
TRADING_DAYS_PER_MONTH = 21
REG_WINDOW_MONTHS = 36
REG_MIN_MONTHS    = 24
REG_MIN_DAYS      = REG_MIN_MONTHS * TRADING_DAYS_PER_MONTH
RESID_1M_MIN_DAYS      = 10
RESID_12M_MONTHLY_MIN  = 6

# ── Beta window for leg-sizing (identical to S3/S7/S8) ───────────────────────
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Backtest dates: quarterly, matching GP/A's cadence, 24 periods ───────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])

# ── Cost model (identical to S3/S5/S7/S8/S10) ────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR: new independent trial, next slot after the registry's current max (18) ─
TRIAL_NUMBER = 19
N_TRIALS     = 19

PORTFOLIOS = ["composite", "gp_alone", "residrev_alone", "ivolvalue_alone"]
SCORE_COL  = {
    "composite":       "composite_score",
    "gp_alone":        "gp_signal",
    "residrev_alone":  "resid_signal",
    "ivolvalue_alone": "ivol_value_raw",
}


# =============================================================================
# SECTION 2: Tiingo / SEC metadata helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run mgmt_pit/run.py first.")
    df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
    print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
    return df


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
# SECTION 4: AdjClose cache (SIGNAL layer)
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


def _adj_returns(ticker: str) -> pd.Series:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["adjClose"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


# =============================================================================
# SECTION 5: SEC facts caches (shares / GP / value) — all read-only reuse
# =============================================================================

_SHARES_CACHE: dict[int, Optional[dict]] = {}
_SHARES_TRIMMED = CACHE / "sec_facts_trimmed.json"


def preload_shares_facts(ciks: list[int]) -> None:
    t0 = time.time()
    if not _SHARES_TRIMMED.exists():
        raise FileNotFoundError(f"Missing {_SHARES_TRIMMED}. Run research/momentum_jt/run.py first.")
    with open(_SHARES_TRIMMED) as f:
        raw = json.load(f)
    for k, v in raw.items():
        _SHARES_CACHE[int(k)] = v
    loaded = sum(1 for v in _SHARES_CACHE.values() if v is not None)
    print(f"[SharesFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


def _get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    facts = _SHARES_CACHE.get(cik)
    if facts is None:
        return None
    as_of_str = as_of.strftime("%Y-%m-%d")
    for concept in SHARES_CONCEPTS:
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


_GP_FACTS_CACHE: dict[int, Optional[dict]] = {}
_GP_FACTS_PKL   = CACHE / "gp_facts_trimmed.json"


def preload_gp_facts(ciks: list[int]) -> None:
    t0 = time.time()
    if not _GP_FACTS_PKL.exists():
        raise FileNotFoundError(f"Missing {_GP_FACTS_PKL}. Run research/gp/run.py first.")
    with open(_GP_FACTS_PKL) as f:
        raw = json.load(f)
    for k, v in raw.items():
        _GP_FACTS_CACHE[int(k)] = v
    missing = [c for c in ciks if c not in _GP_FACTS_CACHE]
    if missing:
        _load_facts_from_files(missing, _GP_FACTS_CACHE, GP_NEEDED, save_path=None)
    loaded = sum(1 for v in _GP_FACTS_CACHE.values() if v is not None)
    print(f"[GPFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


_VALUE_FACTS_CACHE: dict[int, Optional[dict]] = {}
_VALUE_FACTS_PKL   = IVOL_VALUE_CACHE / "value_facts_trimmed.json"


def preload_value_facts(ciks: list[int]) -> None:
    t0 = time.time()
    if not _VALUE_FACTS_PKL.exists():
        raise FileNotFoundError(f"Missing {_VALUE_FACTS_PKL}. Run research/ivol_value/run.py first.")
    with open(_VALUE_FACTS_PKL) as f:
        raw = json.load(f)
    for k, v in raw.items():
        _VALUE_FACTS_CACHE[int(k)] = v
    missing = [c for c in ciks if c not in _VALUE_FACTS_CACHE]
    if missing:
        _load_facts_from_files(missing, _VALUE_FACTS_CACHE, VALUE_NEEDED, save_path=None)
    loaded = sum(1 for v in _VALUE_FACTS_CACHE.values() if v is not None)
    print(f"[ValueFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


def _load_facts_from_files(ciks: list[int], target_cache: dict, needed: set, save_path) -> None:
    for cik in ciks:
        p = CACHE / "sec_facts" / f"{cik}.json"
        if not p.exists():
            target_cache[cik] = None
            continue
        try:
            with open(p) as f:
                full = json.load(f)
            trimmed: dict = {"facts": {"us-gaap": {}}}
            ug = full.get("facts", {}).get("us-gaap", {})
            for concept in needed:
                if concept in ug:
                    trimmed["facts"]["us-gaap"][concept] = ug[concept]
            target_cache[cik] = trimmed
        except Exception:
            target_cache[cik] = None


# =============================================================================
# SECTION 6: Market factor for residual reversal (EW across survivors, S8-identical)
# =============================================================================

def build_market_factor(tickers: list[str]) -> pd.Series:
    t0 = time.time()
    series_map: dict[str, pd.Series] = {}
    for t in tickers:
        r = _adj_returns(t)
        if len(r) >= REG_MIN_DAYS:
            series_map[t] = r
    if not series_map:
        raise RuntimeError("No tickers with sufficient adjClose history for market factor.")
    mat = pd.concat(series_map, axis=1)
    mkt = mat.clip(-0.5, 0.5).mean(axis=1).dropna()
    print(f"[MktFactor] EW factor from {len(series_map):,} tickers, "
          f"{len(mkt):,} days ({mkt.index.min().date()} -> {mkt.index.max().date()}) "
          f"in {time.time()-t0:.1f}s")
    return mkt


# =============================================================================
# SECTION 7: TTM flow helper (shared by GP/A and E/P)
# =============================================================================

def _period_days(start_str: str, end_str: str) -> int:
    try:
        s = date.fromisoformat(start_str)
        e = date.fromisoformat(end_str)
        return (e - s).days
    except Exception:
        return 0


def _get_ttm_flow(facts: Optional[dict], concept_list: list[str], as_of_str: str) -> tuple:
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
                overlap = any(not (q_end <= s[2] or q_start >= s[1]) for s in selected)
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


def _get_assets_pit(facts: Optional[dict], as_of_str: str) -> Optional[float]:
    if facts is None:
        return None
    try:
        units_dict = facts["facts"]["us-gaap"]["Assets"]["units"]
    except (KeyError, TypeError):
        return None
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


# =============================================================================
# SECTION 8: Universe builder (identical to S3/S5/S7/S8/S10)
# =============================================================================

def build_universe(rebalance_date: pd.Timestamp, tiingo_tickers: pd.DataFrame, cik_sic_map: dict) -> pd.DataFrame:
    survivors_set = set(cik_sic_map.keys())
    active = tiingo_tickers[
        (tiingo_tickers["startDate"] <= rebalance_date)
        & (tiingo_tickers["endDate"].isna() | (tiingo_tickers["endDate"] >= rebalance_date))
        & tiingo_tickers["ticker"].isin(survivors_set)
    ]
    rows = []
    for _, row in active.iterrows():
        ticker = row["ticker"]
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
    print(f"  [Universe] {rebalance_date.date()} -> {len(df):,} names", end="")
    return df


# =============================================================================
# SECTION 9: Component signal computations
# =============================================================================

def compute_gp_signal(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    records = []
    for _, row in universe_df.iterrows():
        cik = int(row["cik"])
        facts = _GP_FACTS_CACHE.get(cik)
        gp_ttm, gp_method = _get_ttm_flow(facts, GROSS_PROFIT_CONCEPTS, as_of_str)
        if gp_ttm is None:
            rev, rev_method = _get_ttm_flow(facts, REVENUE_CONCEPTS, as_of_str)
            cgs, cgs_method = _get_ttm_flow(facts, COGS_CONCEPTS, as_of_str)
            if rev is not None and cgs is not None:
                gp_ttm = rev - cgs
            else:
                continue
        assets = _get_assets_pit(facts, as_of_str)
        if assets is None or assets <= 0:
            continue
        gp_a = gp_ttm / assets
        if not np.isfinite(gp_a):
            continue
        records.append({"ticker": row["ticker"], "gp_signal": gp_a})
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "gp_signal"])


def compute_value_signal(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    records = []
    for _, row in universe_df.iterrows():
        cik = int(row["cik"])
        facts = _VALUE_FACTS_CACHE.get(cik)
        ni_ttm, method = _get_ttm_flow(facts, NET_INCOME_CONCEPTS, as_of_str)
        if ni_ttm is None:
            continue
        ep = ni_ttm / row["mcap"]
        if not np.isfinite(ep):
            continue
        records.append({"ticker": row["ticker"], "value_signal": ep})
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "value_signal"])


def compute_ivol_signal(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    """Identical algorithm to S1/S7's compute_ivol_signals. Returns ticker, ivol (magnitude)."""
    lookback_start = rebalance_date - pd.Timedelta(days=IVOL_LOOKBACK_CAL)
    price_series: dict[str, pd.Series] = {}
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

    if len(price_series) < 10:
        return pd.DataFrame(columns=["ticker", "ivol"])

    price_mat = pd.DataFrame(price_series).sort_index()
    ret_mat   = price_mat.pct_change().iloc[1:].tail(IVOL_WINDOW_DAYS)
    valid_counts = ret_mat.notna().sum()
    ret_mat = ret_mat.loc[:, valid_counts >= IVOL_MIN_DAYS]
    if ret_mat.shape[1] < 5:
        return pd.DataFrame(columns=["ticker", "ivol"])

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
        records.append({"ticker": ticker, "ivol": ivol})
    return pd.DataFrame(records) if records else pd.DataFrame(columns=["ticker", "ivol"])


def compute_residual_reversal(ticker: str, rebalance_date: pd.Timestamp, mkt_factor: pd.Series) -> Optional[float]:
    """Identical to S8's compute_residual_reversal, signal only (no vol-filter gate applied
    here — the composite trades the continuous rank, not a decile+filter subset)."""
    reg_start = rebalance_date - pd.DateOffset(months=REG_WINDOW_MONTHS)
    ret = _adj_returns(ticker)
    if ret.empty:
        return None
    window_ret = ret[(ret.index >= reg_start) & (ret.index <= rebalance_date)]
    if len(window_ret) < REG_MIN_DAYS:
        return None
    x = mkt_factor.reindex(window_ret.index).dropna()
    y = window_ret.reindex(x.index)
    if len(y) < REG_MIN_DAYS:
        return None
    X = np.column_stack([np.ones(len(x)), x.values])
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(X, y.values, rcond=None)
    except Exception:
        return None
    if rank < 2:
        return None
    y_hat = X @ coeffs
    resid = pd.Series(y.values - y_hat, index=y.index)
    rev_start = rebalance_date - pd.DateOffset(months=1)
    resid_1m_window = resid[(resid.index >= rev_start) & (resid.index <= rebalance_date)]
    if len(resid_1m_window) < RESID_1M_MIN_DAYS:
        return None
    resid_1m = float(resid_1m_window.sum())
    return -resid_1m   # reversal: prior loser -> positive signal -> long


# =============================================================================
# SECTION 10: Beta computation for leg-sizing (standardized 60-day EW OLS)
# =============================================================================

def compute_betas(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict:
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)
    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj    = _adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            raw_df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
            if raw_df.empty:
                continue
            raw_w = raw_df[(raw_df["date"] >= beta_start) & (raw_df["date"] <= rebalance_date)]
            if raw_w.empty:
                continue
            s = raw_w.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()
        else:
            s = adj.set_index("date")["adjClose"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
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
# SECTION 11: Return calculation (identical pattern)
# =============================================================================

def compute_returns(universe_df: pd.DataFrame, hold_start: pd.Timestamp, hold_end: pd.Timestamp,
                     delisted_map: dict) -> pd.DataFrame:
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
# SECTION 12: Composite score construction
# =============================================================================

def build_composite_universe(gp_df, value_df, ivol_df, resid_df, base_df) -> pd.DataFrame:
    """
    Merge all 4 signal sources onto the base universe; keep only names with
    ALL THREE component signals present (GP/A, ResidRev, and BOTH ivol+value
    since ivol_value_raw needs both). Compute rank-percentiles + composite.
    """
    merged = base_df.merge(gp_df, on="ticker", how="left") \
                     .merge(value_df, on="ticker", how="left") \
                     .merge(ivol_df, on="ticker", how="left") \
                     .merge(resid_df, on="ticker", how="left")

    complete = merged.dropna(subset=["gp_signal", "value_signal", "ivol", "resid_signal"]).copy()
    if complete.empty:
        return complete

    complete["rank_ivol"]  = complete["ivol"].rank(pct=True)
    complete["rank_value"] = complete["value_signal"].rank(pct=True)
    complete["ivol_value_raw"] = complete["rank_ivol"] * complete["rank_value"]

    complete["pct_gp"]        = complete["gp_signal"].rank(pct=True)
    complete["pct_resid"]     = complete["resid_signal"].rank(pct=True)
    complete["pct_ivolvalue"] = complete["ivol_value_raw"].rank(pct=True)
    complete["composite_score"] = (complete["pct_gp"] + complete["pct_resid"] + complete["pct_ivolvalue"]) / 3.0

    return complete


# =============================================================================
# SECTION 13: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(df: pd.DataFrame, score_col: str, betas: dict) -> dict:
    """Quintile long-top/short-bottom on score_col (all 4 scores coded higher=long).
    Name-count guard: quintile leg<20 -> widen to terciles (S5/S10/S11 convention)."""
    sub = df.dropna(subset=[score_col, "raw_return"]).copy()
    n = len(sub)
    empty = {"longs": [], "shorts": [], "lw": {}, "sw": {}, "long_scale": 1.0, "short_scale": 1.0,
             "mean_beta_long": np.nan, "mean_beta_short": np.nan, "guard_fired": False, "n": n}
    if n < 6:
        return empty

    leg_quintile = int(np.ceil(n * 0.20))
    guard_fired  = leg_quintile < 20
    leg = int(np.ceil(n / 3.0)) if guard_fired else leg_quintile
    leg = min(leg, n // 2)
    if leg < 1:
        empty["guard_fired"] = guard_fired
        return empty

    ranked = sub.sort_values(score_col, ascending=False)
    longs  = ranked.iloc[:leg]["ticker"].tolist()
    shorts = ranked.iloc[-leg:]["ticker"].tolist()
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

    return {"longs": longs, "shorts": shorts, "lw": lw, "sw": sw,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
            "guard_fired": guard_fired, "n": n}


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


def _contribution_map(returns_df: pd.DataFrame, tickers: list, weights: dict, is_short: bool) -> dict:
    """Per-name |dollar-weighted return contribution| for concentration checks."""
    ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
    out = {}
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w = weights.get(t, 0.0)
        contrib = -w * r if is_short else w * r
        out[t] = contrib
    return out


# =============================================================================
# SECTION 14: Backtest engine
# =============================================================================

def run_backtest(tiingo_tickers: pd.DataFrame, prefilter_df: pd.DataFrame) -> dict:
    t_total = time.time()

    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    survivor_tickers = list(cik_sic_map.keys())

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    print("\n[MktFactor] Building EW residual-reversal market factor (one-time)...")
    mkt_factor = build_market_factor(survivor_tickers)

    period_records: list = []
    universe_audit: list = []
    concentration_records: list = []
    prev_sets: dict = {p: (set(), set()) for p in PORTFOLIOS}

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        hold_end = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] empty base universe")
            continue

        gp_df    = compute_gp_signal(univ, t)
        value_df = compute_value_signal(univ, t)
        ivol_df  = compute_ivol_signal(univ, t)

        resid_records = []
        for ticker in univ["ticker"]:
            sig = compute_residual_reversal(ticker, t, mkt_factor)
            if sig is not None:
                resid_records.append({"ticker": ticker, "resid_signal": sig})
        resid_df = pd.DataFrame(resid_records) if resid_records else pd.DataFrame(columns=["ticker", "resid_signal"])

        complete = build_composite_universe(gp_df, value_df, ivol_df, resid_df, univ)
        print(f"  | GP={len(gp_df)} Value={len(value_df)} IVOL={len(ivol_df)} "
              f"ResidRev={len(resid_df)} -> intersection={len(complete)}")

        if complete.empty:
            print("  [SKIP] empty intersected (composite) universe")
            continue

        try:
            returns_df = compute_returns(complete, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        betas = compute_betas(complete, t)

        universe_audit.append({
            "rebalance_date": t, "hold_end": hold_end,
            "n_base_universe": len(univ), "n_gp": len(gp_df), "n_value": len(value_df),
            "n_ivol": len(ivol_df), "n_residrev": len(resid_df),
            "n_intersection": len(complete), "n_delisted": n_delisted,
            "n_valid_returns": int(returns_df["raw_return"].notna().sum()),
        })

        mkt_ret = float(returns_df["raw_return"].dropna().mean()) if returns_df["raw_return"].notna().any() else np.nan

        row: dict = {"rebalance_date": t, "hold_end": hold_end, "n_universe": len(complete), "mkt_ret": mkt_ret}

        for pf in PORTFOLIOS:
            score_col = SCORE_COL[pf]
            port = build_portfolio(returns_df, score_col, betas)
            longs, shorts = port["longs"], port["shorts"]

            prev_long_set, prev_short_set = prev_sets[pf]
            if longs and shorts:
                if prev_long_set or prev_short_set:
                    long_to  = len(set(longs)  - prev_long_set)  / max(len(longs), 1)
                    short_to = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
                else:
                    long_to = short_to = 1.0
                prev_sets[pf] = (set(longs), set(shorts))
            else:
                long_to = short_to = np.nan
                prev_sets[pf] = (prev_long_set, prev_short_set)

            if not longs or not shorts:
                for suf in ["long_ret", "short_ret", "ls_ret", "ls_gross", "ls_cost", "ls_ba_cost",
                            "ls_borrow_cost", "long_scale", "short_scale", "mean_beta_long",
                            "mean_beta_short", "avg_turnover", "n_long", "guard_fired", "ic"]:
                    row[f"{pf}__{suf}"] = np.nan
                continue

            long_net, long_gross, long_ba, long_bw = _leg_return_detail(
                returns_df, longs, port["lw"], is_short=False, turnover=long_to)
            short_net, short_gross, short_ba, short_bw = _leg_return_detail(
                returns_df, shorts, port["sw"], is_short=True, turnover=short_to)

            ls_net   = port["long_scale"] * long_net   + port["short_scale"] * short_net
            ls_gross = port["long_scale"] * long_gross + port["short_scale"] * short_gross
            ls_ba    = port["long_scale"] * long_ba    + port["short_scale"] * short_ba
            ls_bw    = port["long_scale"] * long_bw    + port["short_scale"] * short_bw

            valid_sig = returns_df.dropna(subset=[score_col, "raw_return"])
            if len(valid_sig) >= 10:
                ic, _ = stats.spearmanr(valid_sig[score_col], valid_sig["raw_return"])
            else:
                ic = np.nan

            row[f"{pf}__long_ret"]        = long_net
            row[f"{pf}__short_ret"]       = short_net
            row[f"{pf}__ls_ret"]          = ls_net
            row[f"{pf}__ls_gross"]        = ls_gross
            row[f"{pf}__ls_cost"]         = ls_ba + ls_bw
            row[f"{pf}__ls_ba_cost"]      = ls_ba
            row[f"{pf}__ls_borrow_cost"]  = ls_bw
            row[f"{pf}__long_scale"]      = port["long_scale"]
            row[f"{pf}__short_scale"]     = port["short_scale"]
            row[f"{pf}__mean_beta_long"]  = port["mean_beta_long"]
            row[f"{pf}__mean_beta_short"] = port["mean_beta_short"]
            row[f"{pf}__avg_turnover"]    = (long_to + short_to) / 2.0
            row[f"{pf}__n_long"]          = len(longs)
            row[f"{pf}__guard_fired"]     = port["guard_fired"]
            row[f"{pf}__ic"]              = ic

            if pf == "composite":
                long_contrib  = _contribution_map(returns_df, longs, port["lw"], is_short=False)
                short_contrib = _contribution_map(returns_df, shorts, port["sw"], is_short=True)
                all_contrib = {**long_contrib, **short_contrib}
                abs_vals = {k: abs(v) for k, v in all_contrib.items()}
                total_abs = sum(abs_vals.values())
                if total_abs > 0:
                    sorted_abs = sorted(abs_vals.values(), reverse=True)
                    top1_share = sorted_abs[0] / total_abs
                    top3_share = sum(sorted_abs[:3]) / total_abs
                else:
                    top1_share = top3_share = np.nan
                concentration_records.append({
                    "rebalance_date": t, "top1_share": top1_share, "top3_share": top3_share,
                    "n_names": len(all_contrib),
                })

        print(f"  composite L/S={row.get('composite__ls_ret', np.nan):+.2%} "
              f"| gp={row.get('gp_alone__ls_ret', np.nan):+.2%} "
              f"| resid={row.get('residrev_alone__ls_ret', np.nan):+.2%} "
              f"| ivolval={row.get('ivolvalue_alone__ls_ret', np.nan):+.2%}")

        period_records.append(row)

    elapsed = time.time() - t_total
    return {"period_records": period_records, "universe_audit": universe_audit,
            "concentration_records": concentration_records, "elapsed_s": elapsed}


# =============================================================================
# SECTION 15: Metrics
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan, "calmar": np.nan,
                "max_dd": np.nan, "t_stat": np.nan, "win_rate": np.nan, "n": n}
    cum     = (1 + ls).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (4.0 / n) - 1)
    mu, sigma = float(ls.mean()), float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(4)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s    = (cum - roll_mx) / roll_mx
    max_dd  = float(dd_s.min())
    calmar  = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat  = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt  = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe, "calmar": calmar,
            "max_dd": max_dd, "t_stat": t_stat, "win_rate": win_rt, "n": n}


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
    beta = float(coeffs[1])
    y_hat = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n = len(y)
    se = (np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 16: Output & registry
# =============================================================================

# Original registry numbers for reference (trial #8 / #13 / #14, each on its
# own broader universe / cadence — NOT the same as this trial's "-alone"
# recomputation on the shared intersected universe).
ORIGINAL_REGISTRY = {
    "gp_S3-GP":         {"sharpe": 0.1338, "calmar": 0.0332, "mean_ic": 0.0404, "ic_t": 2.283},
    "ivolvalue_S7":     {"sharpe": -0.5444, "calmar": -0.2095, "mean_ic": 0.0358, "ic_t": 1.356},
    "residreversal_S8": {"sharpe": -1.1075, "calmar": -0.2154, "mean_ic": 0.0360, "ic_t": 3.397},
}


def _portfolio_metrics(df: pd.DataFrame, pf: str) -> dict:
    ls_series    = df[f"{pf}__ls_ret"]
    gross_series = df[f"{pf}__ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df[f"{pf}__ic"].dropna()

    m       = compute_metrics(ls_series)
    m_gross = compute_metrics(gross_series)
    beta    = compute_market_beta(ls_series, mkt_series)
    mean_ic = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_t    = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
               if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)
    annual_cost_drag = (float(df[f"{pf}__ls_ba_cost"].mean(skipna=True))
                         + float(df[f"{pf}__ls_borrow_cost"].mean(skipna=True))) * 4 * 10_000
    mean_turnover = float(df[f"{pf}__avg_turnover"].mean(skipna=True))
    guard_fire = int(df[f"{pf}__guard_fired"].fillna(False).astype(bool).sum())

    per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
    dsr_thr = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False

    return {"metrics": m, "metrics_gross": m_gross, "beta": beta, "mean_ic": mean_ic, "ic_t": ic_t,
            "annual_cost_drag_bps": annual_cost_drag, "mean_turnover": mean_turnover,
            "guard_fire_count": guard_fire, "per_period_sr": per_period_sr,
            "dsr_threshold": dsr_thr, "clears_dsr": clears_dsr, "n_periods": m["n"]}


def year_by_year_returns(df: pd.DataFrame, pf: str) -> pd.DataFrame:
    tmp = df[["rebalance_date", f"{pf}__ls_ret", f"{pf}__ls_gross"]].copy()
    tmp["year"] = tmp["rebalance_date"].dt.year
    rows = []
    for yr, grp in tmp.groupby("year"):
        net = grp[f"{pf}__ls_ret"].dropna()
        gross = grp[f"{pf}__ls_gross"].dropna()
        ann_net   = float((1 + net).prod() - 1) if len(net) else np.nan
        ann_gross = float((1 + gross).prod() - 1) if len(gross) else np.nan
        rows.append({"year": yr, "n_quarters": len(net), "ann_return_net": ann_net, "ann_return_gross": ann_gross})
    return pd.DataFrame(rows)


def print_and_save_results(results: dict) -> dict:
    records = results["period_records"]
    if not records:
        print("\n[ERROR] No periods completed.")
        return {}
    df = pd.DataFrame(records)

    stats_by_pf = {pf: _portfolio_metrics(df, pf) for pf in PORTFOLIOS}

    sep = "=" * 90
    print(f"\n{sep}")
    print("S15: LOW-TURNOVER COMPOSITE OF VALIDATED-IC SIGNALS  (trial #19, N_TRIALS=19)")
    print(sep)

    print(f"\n  {'Metric':<32} {'COMPOSITE':>14} {'GP/A-alone':>14} {'ResidRev-alone':>16} {'IVOLValue-alone':>17}")
    print(f"  {'-'*32} {'-'*14} {'-'*14} {'-'*16} {'-'*17}")
    def _row(label, key, fmt):
        vals = []
        for pf in PORTFOLIOS:
            s = stats_by_pf[pf]
            if key == "sharpe": v = s["metrics"]["sharpe"]
            elif key == "calmar": v = s["metrics"]["calmar"]
            elif key == "ann_net": v = s["metrics"]["ann_ret"]
            elif key == "ann_gross": v = s["metrics_gross"]["ann_ret"]
            elif key == "beta": v = s["beta"]["beta"]
            elif key == "beta_t": v = s["beta"]["beta_t"]
            elif key == "max_dd": v = s["metrics"]["max_dd"]
            elif key == "mean_ic": v = s["mean_ic"]
            elif key == "ic_t": v = s["ic_t"]
            elif key == "win_rate": v = s["metrics"]["win_rate"]
            elif key == "turnover": v = s["mean_turnover"]
            elif key == "cost_drag": v = s["annual_cost_drag_bps"]
            elif key == "dsr": v = "YES" if s["clears_dsr"] else "NO"
            vals.append(v)
        if fmt == "s":
            print(f"  {label:<32} " + " ".join(f"{v:>14}" if i==0 else f"{v:>16}" if i==2 else (f"{v:>17}" if i==3 else f"{v:>14}") for i, v in enumerate(vals)))
        else:
            print(f"  {label:<32} " + " ".join(format(v, fmt).rjust(14 if i==0 else (16 if i==2 else (17 if i==3 else 14))) if not (isinstance(v,float) and np.isnan(v)) else "n/a".rjust(14 if i==0 else (16 if i==2 else (17 if i==3 else 14))) for i, v in enumerate(vals)))
    _row("Sharpe (net, ann.)", "sharpe", ".3f")
    _row("CALMAR", "calmar", ".3f")
    _row("Ann. return (net)", "ann_net", ".2%")
    _row("Ann. return (gross)", "ann_gross", ".2%")
    _row("Market beta", "beta", ".3f")
    _row("Market beta t-stat", "beta_t", ".3f")
    _row("Max drawdown", "max_dd", ".2%")
    _row("Mean IC (Spearman)", "mean_ic", ".4f")
    _row("IC t-stat", "ic_t", ".3f")
    _row("Win rate", "win_rate", ".1%")
    _row("Mean turnover/qtr", "turnover", ".0%")
    _row("Cost drag (bps/yr)", "cost_drag", ".0f")
    _row("Clears DSR (N=19)", "dsr", "s")

    print(f"\n  N periods: {stats_by_pf['composite']['n_periods']}  "
          f"| Guard fired (composite): {stats_by_pf['composite']['guard_fire_count']}/{len(df)}")

    print(f"\n{sep}")
    print("ORIGINAL REGISTRY NUMBERS FOR REFERENCE (different universe/cadence per signal)")
    print(sep)
    for k, v in ORIGINAL_REGISTRY.items():
        print(f"  {k:<20} sharpe={v['sharpe']:+.4f}  calmar={v['calmar']:+.4f}  "
              f"mean_ic={v['mean_ic']:+.4f}  ic_t={v['ic_t']:+.3f}")

    print(f"\n{sep}")
    print("YEAR-BY-YEAR RETURNS")
    print(sep)
    yby = {}
    for pf in PORTFOLIOS:
        yby[pf] = year_by_year_returns(df, pf)
    years = sorted(df["rebalance_date"].dt.year.unique())
    print(f"  {'Year':<6} {'Composite(net)':>15} {'GP/A(net)':>12} {'ResidRev(net)':>14} {'IVOLValue(net)':>15}")
    for yr in years:
        vals = []
        for pf in PORTFOLIOS:
            row = yby[pf][yby[pf]["year"] == yr]
            vals.append(float(row["ann_return_net"].iloc[0]) if not row.empty else np.nan)
        print(f"  {yr:<6} " + " ".join(f"{v:>14.2%}" if not np.isnan(v) else f"{'n/a':>14}" for v in vals[:1])
              + " " + " ".join(f"{v:>11.2%}" if not np.isnan(v) else f"{'n/a':>11}" for v in vals[1:2])
              + " " + " ".join(f"{v:>13.2%}" if not np.isnan(v) else f"{'n/a':>13}" for v in vals[2:3])
              + " " + " ".join(f"{v:>14.2%}" if not np.isnan(v) else f"{'n/a':>14}" for v in vals[3:4]))

    print(f"\n{sep}")
    print("COMPONENT PAIRWISE CORRELATION (quarterly net L/S returns, in-sample, shared universe)")
    print(sep)
    corr_df = df[["gp_alone__ls_ret", "residrev_alone__ls_ret", "ivolvalue_alone__ls_ret"]].rename(
        columns={"gp_alone__ls_ret": "GP/A", "residrev_alone__ls_ret": "ResidRev", "ivolvalue_alone__ls_ret": "IVOLValue"})
    corr_mat = corr_df.corr()
    print(corr_mat.round(3).to_string())

    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT (mean IC on shared composite universe, each component)")
    print(sep)
    for pf in ["gp_alone", "residrev_alone", "ivolvalue_alone"]:
        s = stats_by_pf[pf]
        print(f"  {pf:<20} mean_ic={s['mean_ic']:+.4f}  ic_t={s['ic_t']:+.3f}  "
              f"(original registry sign: positive for all three -- expect positive here too)")

    conc = pd.DataFrame(results["concentration_records"])
    print(f"\n{sep}")
    print("CONCENTRATION CHECK (composite portfolio, per-period |return contribution| share)")
    print(sep)
    if not conc.empty:
        print(f"  Mean top-1 name share: {conc['top1_share'].mean():.1%}  |  Max: {conc['top1_share'].max():.1%}")
        print(f"  Mean top-3 name share: {conc['top3_share'].mean():.1%}  |  Max: {conc['top3_share'].max():.1%}")
        n_flag_top3 = int((conc["top3_share"] > 0.30).sum())
        print(f"  Periods with top-3 share > 30%: {n_flag_top3}/{len(conc)}")

    print(f"\n  Runtime: {results['elapsed_s']:.1f}s ({results['elapsed_s']/60:.1f} min)")

    # ── Save CSVs ──────────────────────────────────────────────────────────
    df.to_csv(RESULTS / "s15_period_returns.csv", index=False)
    pd.DataFrame(results["universe_audit"]).to_csv(RESULTS / "s15_universe_audit.csv", index=False)
    if not conc.empty:
        conc.to_csv(RESULTS / "s15_concentration.csv", index=False)
    corr_mat.to_csv(RESULTS / "s15_component_correlation.csv")
    for pf in PORTFOLIOS:
        yby[pf].to_csv(RESULTS / f"s15_year_by_year_{pf}.csv", index=False)
    print(f"\n[Saved] Results -> {RESULTS}/")

    return {"df": df, "stats_by_pf": stats_by_pf, "corr_mat": corr_mat, "conc": conc, "yby": yby}


def log_registry(summary: dict) -> None:
    if not summary:
        return
    df = summary["df"]
    stats_by_pf = summary["stats_by_pf"]
    comp = stats_by_pf["composite"]

    corr_mat = summary["corr_mat"]
    corr_gp_resid = float(corr_mat.loc["GP/A", "ResidRev"])
    corr_gp_ivv   = float(corr_mat.loc["GP/A", "IVOLValue"])
    corr_resid_ivv = float(corr_mat.loc["ResidRev", "IVOLValue"])

    best_component_sharpe = max(
        stats_by_pf["gp_alone"]["metrics"]["sharpe"],
        stats_by_pf["residrev_alone"]["metrics"]["sharpe"],
        stats_by_pf["ivolvalue_alone"]["metrics"]["sharpe"],
    )
    diversification_helped = (comp["metrics"]["sharpe"] > best_component_sharpe) \
        if not np.isnan(comp["metrics"]["sharpe"]) and not np.isnan(best_component_sharpe) else False
    promotion_met = (comp["metrics"]["calmar"] >= 1.0) or (comp["metrics"]["sharpe"] >= 0.8)

    notes = (
        f"Composite Sharpe={comp['metrics']['sharpe']:.4f} CALMAR={comp['metrics']['calmar']:.4f} "
        f"vs best single component alone Sharpe={best_component_sharpe:.4f} "
        f"(GP/A={stats_by_pf['gp_alone']['metrics']['sharpe']:.4f}, "
        f"ResidRev={stats_by_pf['residrev_alone']['metrics']['sharpe']:.4f}, "
        f"IVOLValue={stats_by_pf['ivolvalue_alone']['metrics']['sharpe']:.4f}). "
        f"Diversification {'HELPED' if diversification_helped else 'DID NOT HELP'} "
        f"(composite beat best single alone: {diversification_helped}). "
        f"Promotion criteria (CALMAR>=1 or Sharpe>=0.8) {'MET' if promotion_met else 'NOT MET'}. "
        f"Component pairwise corr: GP/ResidRev={corr_gp_resid:+.3f}, GP/IVOLValue={corr_gp_ivv:+.3f}, "
        f"ResidRev/IVOLValue={corr_resid_ivv:+.3f}. "
        f"Composite beta={comp['beta']['beta']:.4f} (t={comp['beta']['beta_t']:.3f}) vs component betas "
        f"GP={stats_by_pf['gp_alone']['beta']['beta']:.4f}, ResidRev={stats_by_pf['residrev_alone']['beta']['beta']:.4f}, "
        f"IVOLValue={stats_by_pf['ivolvalue_alone']['beta']['beta']:.4f}. "
        f"{'FAILS DSR' if not comp['clears_dsr'] else 'CLEARS DSR'}."
    )

    row = {
        "timestamp": datetime.now().isoformat(),
        "study": "compositeIC_S15",
        "hypothesis": (
            "Equal-weighted rank-percentile composite of three signals with real, correctly-signed "
            "but individually-failing IC (GP/A quality S3-Q, residual short-term reversal S8, "
            "IVOL-conditioned value S7) could produce a genuinely tradeable Sharpe/CALMAR through "
            "diversification, even though each fails standalone for a different reason."
        ),
        "data_source": "Tiingo (raw close for returns, adjClose for signals/beta) + SEC XBRL (PIT gated) -- fully reused caches, no new data sourcing",
        "status": "completed_in_sample",
        "trial_number": TRIAL_NUMBER,
        "trial_note": "New independent trial #19. Genuine new construction (composite), not a re-tune of #8/#13/#14.",
        "n_trials_dsr": N_TRIALS,
        "rebalance_freq": "quarterly",
        "rebalance_note": "Matches GP/A's (anchor signal) cadence, pre-registered, no frequency sweep.",
        "portfolio_note": (
            "Beta-neutral (60-day OLS vs EW universe adjClose, standardized across composite AND all "
            "3 component-alone comparisons). Quintile long-top-20%/short-bottom-20%; name-count guard "
            "(leg<20 -> terciles) same convention as S5/S10/S11."
        ),
        "signal_note": (
            "composite_score = mean(rank_pct(GP/A), rank_pct(ResidRev signal), rank_pct(ivol_value_raw)) "
            "where ivol_value_raw = rank_pct(IVOL)*rank_pct(E/P) -- a NEW continuous generalization of "
            "S7's hard double-sort (no natural continuous scalar existed in S7 as originally constructed). "
            "ResidRev evaluated at quarterly dates (S8 is natively monthly) -- intentional cadence change "
            "to match composite's quarterly reform; no carry mechanism retained."
        ),
        "ann_return_net": comp["metrics"]["ann_ret"],
        "ann_return_gross": comp["metrics_gross"]["ann_ret"],
        "annual_cost_drag_bps": comp["annual_cost_drag_bps"],
        "mean_turnover": comp["mean_turnover"],
        "sharpe": comp["metrics"]["sharpe"],
        "calmar": comp["metrics"]["calmar"],
        "max_drawdown": comp["metrics"]["max_dd"],
        "t_stat": comp["metrics"]["t_stat"],
        "market_beta": comp["beta"]["beta"],
        "market_beta_tstat": comp["beta"]["beta_t"],
        "mean_ic": comp["mean_ic"],
        "ic_t_stat": comp["ic_t"],
        "win_rate": comp["metrics"]["win_rate"],
        "n_periods": comp["n_periods"],
        "per_period_sr": comp["per_period_sr"],
        "dsr_threshold": comp["dsr_threshold"],
        "clears_dsr": comp["clears_dsr"],
        "guard_fire_days": comp["guard_fire_count"],
        "notes": notes,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "compositeIC_S15"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] compositeIC_S15 logged (trial #{TRIAL_NUMBER}) -> {REGISTRY}")


# =============================================================================
# SECTION 17: Main
# =============================================================================

def main() -> None:
    print("=" * 90)
    print("S15: LOW-TURNOVER COMPOSITE OF VALIDATED-IC SIGNALS  (trial #19, N_TRIALS=19)")
    print("Components: GP/A (S3-Q) + Residual Reversal (S8) + IVOL-Value (S7)")
    print("Quarterly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 24 periods")
    print("=" * 90)

    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = load_prefilter()
    survivors = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle...")
    preload_all_adj_prices(survivor_tickers)

    print(f"\n[Step 4a] Loading shares-outstanding facts ({len(survivor_ciks):,} CIKs)...")
    preload_shares_facts(survivor_ciks)

    print(f"\n[Step 4b] Loading GP/A XBRL facts...")
    preload_gp_facts(survivor_ciks)

    print(f"\n[Step 4c] Loading Value (E/P) XBRL facts...")
    preload_value_facts(survivor_ciks)

    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} quarterly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    print("\n[Step 6] Output...")
    summary = print_and_save_results(results)
    log_registry(summary)


if __name__ == "__main__":
    main()
