"""
Industry-Neutral GP/A  (independent trial -- see PROVISIONAL_N below)
=======================================================================
Reformulation of S3-Q (Gross Profitability, trial #8; semiannual variant
S3-SA is the program's only DSR pass to date). S3-Q ranked GP/A against the
FULL cross-sectional universe each quarter. Hypothesis: part of that
portfolio's return/beta may reflect an accidental sector bet (systematically
long one sector's high-GP/A names, short another sector's low-GP/A names)
rather than pure within-sector quality selection. This trial ranks GP/A
WITHIN each SIC major-group (2-digit) first, then builds long/short from
those industry-neutral percentile ranks -- same idea as an industry-neutral
quality factor (Asness/Frazzini/Pedersen-style). Actual hypothesis under
test: doing this should lower realized beta and improve Sharpe/CALMAR
relative to S3-Q's original full-universe-ranked construction.

UNIVERSE NOTE (deviation from initial request, confirmed with user): actual
point-in-time Russell 2000 constituent membership was requested as the
tradable universe, sourced from research/russell_reconstitution/. That
pipeline only has ANNUAL ADDITION/DELETION EVENT LISTS (which ~150-250 names
joined/left each June), never a full constituent snapshot -- confirmed by
inspecting parse_manifest.csv / parsed_all_rows.csv (list_kind is only ever
"additions" or "deletions", never a full list). There is no base year to roll
forward from, so genuine PIT Russell 2000 membership cannot be reconstructed
from what's already built. User chose (via explicit question): use the
existing $100M-2B liquidity-screened universe, identical to S3-Q's own
universe. This is therefore NOT a "Russell 2000 universe upgrade" -- it is a
same-universe, industry-neutral-ranking reformulation of S3-Q. Reported
honestly as a scope change, not claimed as PIT Russell membership.

Signal: identical GP/A definition, universe, cost model, and cadence as
S3-Q (research/gp/run.py) -- reused verbatim (copied, not imported, matching
this codebase's per-trial-independence convention). New: within-SIC-major-
group (2-digit, sic // 100) rank-percentile of GP/A each quarter. Groups
with < MIN_SECTOR_GROUP_SIZE members (or missing SIC) are pooled into a
single "OTHER" bucket for that quarter's ranking only (pre-registered, not
tuned after seeing results).

Portfolio (BOTH constructions computed from the same per-period universe/
signal/beta/returns data, for an exact apples-to-apples comparison):
  - "original"          : S3-Q's own construction -- rank GP/A across the
                           FULL universe, long top 20%, short bottom 20%.
  - "industry_neutral"   : rank the within-sector percentile across the
                           FULL universe (i.e. compare percentiles, not raw
                           GP/A), long top 20%, short bottom 20%.
Both: beta-neutral leg sizing (60-day OLS vs EW universe return, adjClose),
leg = max(ceil(n*0.20), 20), quarterly rebalance, 24 periods, 2015-2021 —
identical cadence to S3-Q for direct comparability.

Falsification diagnostics:
  - Sign-convention audit (long-leg signal > short-leg signal, both
    constructions, every period).
  - Concentration check: single quarter, single SIC major-group.
  - Sector composition / HHI comparison: does industry-neutral actually
    achieve better sector balance than the original construction?
  - Explicit 2020 sub-period check (S1/S2/S7/E2 precedent).

Reused, not rebuilt: build_universe / compute_gp_signals / compute_betas /
compute_returns / cost model are copied verbatim (with sic added to universe
rows) from research/gp/run.py -- same $100M-2B universe, same GP/A tag
hierarchy, same TTM/annual-fallback logic, same look-ahead discipline (filed
<= t throughout). All underlying data (Tiingo prices, adjClose, SEC facts)
is already fully cached from S3-Q's own run -- no new network fetches
expected.
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
# SECTION 1: Paths, config, constants (identical to S3-Q / research/gp/run.py)
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
SEC_UA = "Hugh Archer hugharcher5@gmail.com"

MGMT_PIT = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
GP_DIR   = Path(__file__).resolve().parents[2] / "research" / "gp"
CACHE    = MGMT_PIT / "cache"
RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
PHASE3_MD = Path(__file__).resolve().parents[1] / "PHASE3_TRIAL_REGISTRY.md"

RESULTS.mkdir(parents=True, exist_ok=True)

MCAP_MIN, MCAP_MAX = 100e6, 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0

TTM_QUARTERS = 4
QTR_MIN_DAYS, QTR_MAX_DAYS = 75, 115
ANN_MIN_DAYS, ANN_MAX_DAYS = 300, 400

BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

GROSS_PROFIT_CONCEPTS = ["GrossProfit", "GrossProfitLoss"]
REVENUE_CONCEPTS = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
]
COGS_CONCEPTS = ["CostOfGoodsSold", "CostOfRevenue", "CostOfGoodsSoldAndServicesCost"]
SHARES_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]

IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")
_all_dates = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 24 quarterly periods

SPREAD_MIN_BPS, SPREAD_MAX_BPS = 20, 100
BORROW_MIN_ANNUAL, BORROW_MAX_ANNUAL = 0.005, 0.020

WINDOW_2020_START = pd.Timestamp("2020-01-01")
WINDOW_2020_END   = pd.Timestamp("2021-01-01")

# ── New for this trial: industry-neutral rank construction ──────────────────
MIN_SECTOR_GROUP_SIZE = 5   # groups smaller than this pooled into "OTHER" (grp=-2)
OTHER_GROUP = -2
UNKNOWN_SIC_GROUP = -1

# Provisional; re-verified against research/trial_registry.csv immediately
# before logging (concurrency lesson from insiderclusters_S13 -- see
# feedback_dsr_trial_numbering memory).
PROVISIONAL_N = 21


# =============================================================================
# SECTION 2: Rate-limited HTTP (kept for interface parity; no new fetches expected)
# =============================================================================

_TIINGO_LAST: float = 0.0
_SEC_LAST:    float = 0.0


def _sec_get(url: str) -> requests.Response:
    global _SEC_LAST
    wait = 0.2 - (time.time() - _SEC_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=(10, 30))
    _SEC_LAST = time.time()
    return r


# =============================================================================
# SECTION 3: Shared read-only caches (reused from mgmt_pit / gp, verbatim)
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    df = pd.read_csv(CACHE / "tickers.csv", parse_dates=["startDate", "endDate"])
    print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
    return df


def load_prefilter() -> pd.DataFrame:
    df = pd.read_csv(CACHE / "sec_prefilter.csv")
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors.")
    return df


_PRICE_CACHE: dict[str, pd.DataFrame] = {}
_PRICE_EMPTY = pd.DataFrame(columns=["date", "close", "volume"])


def preload_all_prices() -> None:
    t0 = time.time()
    with open(CACHE / "prices_all.pkl", "rb") as f:
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


_ADJ_CACHE: dict[str, pd.DataFrame] = {}
_ADJ_EMPTY = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices() -> None:
    t0 = time.time()
    adj_pkl = CACHE / "adjclose_all.pkl"
    if not adj_pkl.exists():
        print("[AdjPreload] No pickle -- beta will use raw close fallback.")
        return
    with open(adj_pkl, "rb") as f:
        data = pickle.load(f)
    _ADJ_CACHE.update(data.get("adj", {}))
    print(f"[AdjPreload] {sum(1 for v in _ADJ_CACHE.values() if not v.empty):,} tickers "
          f"in {time.time()-t0:.1f}s")


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 4: GP/A facts cache (reuse gp/'s already-trimmed cache, read-only)
# =============================================================================

_GP_FACTS_CACHE: dict[int, Optional[dict]] = {}
_GP_FACTS_CONSOLIDATED = CACHE / "gp_facts_trimmed.json"


def preload_gp_facts(ciks: list[int]) -> None:
    t0 = time.time()
    with open(_GP_FACTS_CONSOLIDATED) as f:
        raw = json.load(f)
    for k, v in raw.items():
        _GP_FACTS_CACHE[int(k)] = v
    loaded = sum(1 for v in _GP_FACTS_CACHE.values() if v is not None)
    print(f"[Preload] GP facts: {loaded:,}/{len(ciks):,} CIKs in {time.time()-t0:.1f}s")


def _get_shares_pit(facts: Optional[dict], as_of_str: str) -> Optional[float]:
    if facts is None:
        return None
    for concept in SHARES_CONCEPTS:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        best = None
        for entries in units_dict.values():
            for e in entries:
                filed, val = e.get("filed"), e.get("val")
                if filed is None or val is None or filed > as_of_str:
                    continue
                if best is None or filed > best[0]:
                    best = (filed, float(val))
        if best is not None and best[1] > 0:
            return best[1]
    return None


# =============================================================================
# SECTION 5: Universe builder (S3-Q's construction, verbatim + sic added)
# =============================================================================

def build_universe(rebalance_date: pd.Timestamp, tiingo_tickers: pd.DataFrame,
                    cik_sic_map: dict) -> pd.DataFrame:
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
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
        facts = _GP_FACTS_CACHE.get(cik)
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
    print(f"  [Universe] {rebalance_date.date()} -> {len(df):,} names")
    return df


# =============================================================================
# SECTION 6: GP/A signal computation (S3-Q's construction, verbatim)
# =============================================================================

def _period_days(start_str: str, end_str: str) -> int:
    try:
        return (date.fromisoformat(end_str) - date.fromisoformat(start_str)).days
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
        quarterly, annual = [], []
        for entries in units_dict.values():
            for e in entries:
                filed, end, val, start = e.get("filed"), e.get("end"), e.get("val"), e.get("start")
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
    best = None
    for entries in units_dict.values():
        for e in entries:
            filed, end, val = e.get("filed"), e.get("end"), e.get("val")
            if filed is None or end is None or val is None or filed > as_of_str:
                continue
            key = (filed, end)
            if best is None or key > best[:2]:
                best = (filed, end, float(val))
    return best[2] if (best is not None and best[2] > 0) else None


def compute_gp_signals(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> tuple:
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    records = []
    n_gp_direct = n_gp_calc = n_annual_fb = n_no_cogs = n_no_data = n_no_assets = 0
    for _, row in universe_df.iterrows():
        cik = int(row["cik"])
        facts = _GP_FACTS_CACHE.get(cik)
        gp_ttm, gp_method = _get_ttm_flow(facts, GROSS_PROFIT_CONCEPTS, as_of_str)
        if gp_ttm is not None:
            n_gp_direct += 1
        else:
            rev, rev_method = _get_ttm_flow(facts, REVENUE_CONCEPTS, as_of_str)
            cgs, cgs_method = _get_ttm_flow(facts, COGS_CONCEPTS, as_of_str)
            if rev is not None and cgs is not None:
                gp_ttm, gp_method = rev - cgs, f"rev_minus_cogs({rev_method})"
                n_gp_calc += 1
            elif rev is not None and cgs is None:
                n_no_cogs += 1
                continue
            else:
                n_no_data += 1
                continue
        if gp_method and "annual" in gp_method:
            n_annual_fb += 1
        assets = _get_assets_pit(facts, as_of_str)
        if assets is None or assets <= 0:
            n_no_assets += 1
            continue
        gp_a = gp_ttm / assets
        if not np.isfinite(gp_a):
            n_no_assets += 1
            continue
        records.append({"ticker": row["ticker"], "sic": row["sic"], "gp_signal": gp_a})
    signals_df = (pd.DataFrame(records) if records
                  else pd.DataFrame(columns=["ticker", "sic", "gp_signal"]))
    coverage = {
        "n_universe": len(universe_df), "n_signal": len(signals_df),
        "n_gp_direct": n_gp_direct, "n_gp_calc": n_gp_calc, "n_annual_fb": n_annual_fb,
        "n_no_cogs": n_no_cogs, "n_no_data": n_no_data, "n_no_assets": n_no_assets,
        "pct_dropped": 100 * (len(universe_df) - len(signals_df)) / max(len(universe_df), 1),
    }
    return signals_df, coverage


# =============================================================================
# SECTION 7: Industry-neutral rank transform (NEW for this trial)
# =============================================================================

def add_industry_neutral_rank(signals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Within-SIC-major-group (2-digit) rank-percentile of gp_signal. Groups
    with < MIN_SECTOR_GROUP_SIZE members (or missing SIC) are pooled into a
    single OTHER_GROUP bucket for that quarter's ranking only (pre-registered).
    Adds columns: sector_group, in_rank (0-1, within-group percentile).
    """
    df = signals_df.copy()
    sic_num = pd.to_numeric(df["sic"], errors="coerce")
    df["sector_group"] = np.where(sic_num.notna(), (sic_num // 100).astype("Int64"), UNKNOWN_SIC_GROUP)

    counts = df["sector_group"].value_counts()
    small_groups = set(counts[counts < MIN_SECTOR_GROUP_SIZE].index) | {UNKNOWN_SIC_GROUP}
    df["rank_group"] = df["sector_group"].apply(lambda g: OTHER_GROUP if g in small_groups else g)

    df["in_rank"] = df.groupby("rank_group")["gp_signal"].rank(pct=True, method="average")
    return df


# =============================================================================
# SECTION 8: Beta computation (identical to S3-Q, verbatim)
# =============================================================================

def compute_betas(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict:
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)
    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj = _adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            raw_df = _PRICE_CACHE.get(ticker, _ADJ_EMPTY)
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
    ew_mkt = ret_mat.clip(-0.5, 0.5).mean(axis=1)
    betas: dict[str, float] = {}
    for ticker, ret_s in adj_series.items():
        y_s = ret_s
        x_s = ew_mkt.reindex(y_s.index).dropna()
        y_s = y_s.reindex(x_s.index).dropna()
        x_s = x_s.reindex(y_s.index)
        n = len(y_s)
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
# SECTION 9: Returns (identical to S3-Q, verbatim)
# =============================================================================

def compute_returns(universe_df: pd.DataFrame, hold_start: pd.Timestamp,
                     hold_end: pd.Timestamp, delisted_map: dict) -> pd.DataFrame:
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
    df["entry_price"] = entry_prices
    df["exit_price"] = exit_prices
    df["raw_return"] = raw_returns
    df["delisted"] = delisted_flags
    return df


# =============================================================================
# SECTION 10: Cost model + portfolio construction (both variants)
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def _leg_return_detail(returns_df, tickers, weights, is_short, turnover):
    ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (MCAP_MIN + MCAP_MAX) / 2
    gross = ba_cost = borrow_cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / 4.0) * w
    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    return (gross - total_cost), gross, ba_cost, borrow_cost


def build_portfolio(signals_df: pd.DataFrame, betas: dict, rank_col: str) -> dict:
    """Same leg-sizing rule as S3-Q: leg = max(ceil(n*0.20), 20), min(leg, n//2).
    rank_col selects which ranking to sort on ("gp_signal" = original,
    "in_rank" = industry-neutral)."""
    df = signals_df.dropna(subset=[rank_col, "raw_return"]).copy()
    n = len(df)
    if n < 40:
        return {"longs": [], "shorts": [], "lw": {}, "sw": {}, "long_scale": 1.0,
                "short_scale": 1.0, "mean_beta_long": np.nan, "mean_beta_short": np.nan}

    leg = max(int(np.ceil(n * 0.20)), 20)
    leg = min(leg, n // 2)

    ranked = df.sort_values(rank_col, ascending=False)
    longs = ranked.iloc[:leg]["ticker"].tolist()
    shorts = ranked.iloc[-leg:]["ticker"].tolist()

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    mean_beta_long = float(np.nanmean([betas.get(t, np.nan) for t in longs]))
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return {"longs": longs, "shorts": shorts, "lw": lw, "sw": sw,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short}


def sector_composition(signals_df: pd.DataFrame, tickers: list) -> dict:
    """HHI (sum of squared sector weights) and top-sector share for a leg,
    using equal weight per name (matches the equal-weight leg construction)."""
    if not tickers:
        return {"hhi": np.nan, "top_share": np.nan, "n_sectors": 0}
    sub = signals_df[signals_df["ticker"].isin(tickers)]
    grp_counts = sub["sector_group"].value_counts()
    weights = grp_counts / grp_counts.sum()
    hhi = float((weights ** 2).sum())
    top_share = float(weights.max())
    return {"hhi": hhi, "top_share": top_share, "n_sectors": int(grp_counts.shape[0])}


# =============================================================================
# SECTION 11: Metrics
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "max_dd": np.nan, "t_stat": np.nan,
                "win_rate": np.nan, "n": n}
    cum = (1 + ls).cumprod()
    total = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (4.0 / n) - 1)
    mu = float(ls.mean())
    sigma = float(ls.std(ddof=1))
    sharpe = float((mu / sigma) * np.sqrt(4)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s = (cum - roll_mx) / roll_mx
    max_dd = float(dd_s.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt = float((ls > 0).mean())
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
