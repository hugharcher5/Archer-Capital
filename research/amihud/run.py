"""
S10: Amihud Illiquidity Premium  (independent trial #15 — COST-MODEL STRESS TEST)
====================================================================================
This trial is explicitly NOT run to pass. Per the original ranking notes:
"include only as a cost-stress test: if it survives your bid-ask assumptions
it's real, but expect the cost model to eat most of it." The informative
value here is the GROSS-vs-NET gap, not the headline Sharpe. Do not read a
failure as a bug to fix — read it as evidence about whether the harness's
cost tiers are conservative enough for the least-liquid corner of this
universe.

Signal   : Amihud ILLIQ = mean over the trailing ~1 month (21 trading days)
           of |daily_return| / daily_dollar_volume, at each rebalance date.
           Daily return: adjClose_t / adjClose_(t-1) - 1 (split+dividend
           adjusted, so illiquidity isn't contaminated by corporate-action
           jumps). Daily dollar volume: RAW close x RAW volume — NOT
           adjusted for splits, since dollar volume should reflect actual
           historical trading activity in nominal dollars at the time it
           happened (adjusting volume for splits would rewrite history).
Portfolio: Long TOP quintile ILLIQ (illiquid — earns the liquidity premium
           by being hard to trade), short BOTTOM quintile ILLIQ (most
           liquid). Beta-neutral leg sizing (60-day OLS vs EW universe,
           same convention as every other trial in this program).
Rebalance: Quarterly. Hold: 1 quarter. Matches S1/S3/S5 cadence.
In-sample: 2015-01-01 -> 2021-01-01 (24 periods). No walk-forward.
Trial    : #15 — independent family, final program signal.

CONSTRUCTION — deliberately fights the cost model:
  - NO liquidity floor beyond the standard $100M-2B mcap / $1M ADTV / $1
    price universe screen. The whole point is that the long leg contains
    the worst-spread names the universe allows — filtering them out would
    defeat the test.
  - Name-count guard (widen to terciles if leg <20) exists per program
    convention but is expected to fire rarely — the bottom of a $100M-2B
    universe should have ample illiquid names.
  - The long leg's bid-ask spread is reported on its OWN distribution (not
    just the universe average), because it is adversely selected for width.
    LIMITATION (stated explicitly, not silently absorbed): the harness's
    cost tiers are a function of MARKET CAP ONLY (20bps at $2B -> 100bps at
    $100M), not of realized ILLIQ or ADV directly. Two stocks at the same
    market cap can have very different Amihud ILLIQ. Since this trial
    selects directly on ILLIQ, the mcap-tiered spread assumption is likely
    OPTIMISTIC for the long leg specifically — true spreads for the most
    illiquid names at a given cap tier are probably wider than the tier
    assumes. We do not have finer-grained (ILLIQ-conditional) spread data
    to correct this, so the net-return numbers below should be read as a
    LOWER BOUND on true cost drag, not a precise estimate.

COST MODEL: standard harness bid-ask (cap-tiered) + borrow on shorts. The
short leg (most liquid) should show cheap, plentiful borrow by construction
— reported explicitly as a contrast to the long leg's spread profile.

DSR: independent family, N_TRIALS=15, recomputed fresh (not shared with any
prior family).

No-survivorship guarantee: delisted tickers retained with returns to last
trading day; universe rebuilt PIT from SEC filings each quarter.

Namespacing: research/amihud/ — fully separate cache/results from any
concurrently running trial. Final trial in the ranked S1-S10 program
(alongside S8) — once complete, the full ranked signal list has been tested.
"""

import json
import os
import pickle
import time
from datetime import datetime
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
RESULTS   = Path(__file__).resolve().parent / "results"
REGISTRY  = Path(__file__).resolve().parents[1] / "trial_registry.csv"

RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S1/S3/S5/S9) ──────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── Amihud ILLIQ signal parameters ───────────────────────────────────────────
ILLIQ_WINDOW_DAYS = 21     # ~1 trading month
ILLIQ_MIN_DAYS    = 15     # minimum valid daily observations
ILLIQ_LOOKBACK_CAL = 45    # calendar days to cover >=21 trading days
ILLIQ_DISPLAY_SCALE = 1e6  # cosmetic only — for print formatting, not ranking

# ── Beta window (leg-sizing, same convention as gp/S5/S6/S7) ────────────────
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Portfolio construction ───────────────────────────────────────────────────
LEG_FRAC       = 0.20   # quintile
LEG_MIN_NAMES  = 20     # guard threshold -> widen to terciles if leg would be <20

# ── Backtest dates ────────────────────────────────────────────────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 24 periods

# ── Cost model (identical to every prior trial in this program) ─────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR: new, independent family ─────────────────────────────────────────────
TRIAL_NUMBER = 15
N_TRIALS     = 15


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
# SECTION 3: Raw price cache (RETURN + VOLUME layer — un-adjusted, by design)
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
# SECTION 4: AdjClose cache (used ONLY for the ILLIQ return numerator + beta)
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
# SECTION 5: SEC facts (shares outstanding — reuse S9's trimmed cache)
# =============================================================================

_FACTS_CACHE: dict[int, Optional[dict]] = {}
_FACTS_TRIMMED = CACHE / "sec_facts_trimmed.json"
_SHARES_CONCEPTS = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]


def preload_sec_facts(ciks: list[int]) -> None:
    t0 = time.time()
    if _FACTS_TRIMMED.exists():
        with open(_FACTS_TRIMMED) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _FACTS_CACHE[int(k)] = v
        loaded = sum(1 for v in _FACTS_CACHE.values() if v is not None)
        print(f"[Preload] SEC facts: {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")
        return
    raise FileNotFoundError(f"Missing {_FACTS_TRIMMED}. Run research/momentum_jt/run.py first.")


def _get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    facts = _FACTS_CACHE.get(cik)
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


# =============================================================================
# SECTION 6: Universe builder (identical to S1/S3/S5/S9)
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
        # NOTE: no additional liquidity floor beyond this standard screen —
        # deliberately, per this trial's construction (see module docstring).
        rows.append({"ticker": ticker, "cik": cik, "sic": sic, "mcap": mcap,
                     "price": price, "shares": shares, "avg_dvol": dvol})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "cik", "sic", "mcap", "price", "shares", "avg_dvol"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    print(f"  [Universe] {rebalance_date.date()} → {len(df):,} names", end="")
    return df


# =============================================================================
# SECTION 7: Amihud ILLIQ signal
# =============================================================================

def compute_illiq_signal(ticker: str, rebalance_date: pd.Timestamp) -> Optional[float]:
    """
    ILLIQ = mean_t( |adjClose return_t| / (raw_close_t * raw_volume_t) )
    over the trailing ILLIQ_WINDOW_DAYS (~1 month) trading days ending at t.

    Numerator: adjClose daily return (split+dividend adjusted — a stock
    split must not register as a fake return-driven illiquidity spike).
    Denominator: RAW close x RAW volume — dollar volume in NOMINAL dollars
    as actually traded at the time; NOT split-adjusted, per this trial's
    explicit instruction (adjusting historical volume for a later split
    would misstate what dollar volume actually changed hands each day).

    Returns None if fewer than ILLIQ_MIN_DAYS valid daily observations.
    """
    lookback_start = rebalance_date - pd.Timedelta(days=ILLIQ_LOOKBACK_CAL)

    adj = _adj_slice(ticker, lookback_start, rebalance_date)
    raw = _get_price_slice(ticker, lookback_start, rebalance_date)
    if adj.empty or raw.empty:
        return None

    adj_s = adj.set_index("date")["adjClose"].sort_index()
    adj_s = adj_s[~adj_s.index.duplicated(keep="last")]
    adj_ret = adj_s.pct_change().dropna().tail(ILLIQ_WINDOW_DAYS)
    if len(adj_ret) < ILLIQ_MIN_DAYS:
        return None

    raw_s = raw.set_index("date")
    raw_s = raw_s[~raw_s.index.duplicated(keep="last")]
    dollar_vol = (raw_s["close"] * raw_s["volume"]).reindex(adj_ret.index)

    valid = dollar_vol.notna() & (dollar_vol > 0)
    if int(valid.sum()) < ILLIQ_MIN_DAYS:
        return None

    daily_illiq = adj_ret[valid].abs() / dollar_vol[valid]
    daily_illiq = daily_illiq.replace([np.inf, -np.inf], np.nan).dropna()
    if len(daily_illiq) < ILLIQ_MIN_DAYS:
        return None

    return float(daily_illiq.mean())


# =============================================================================
# SECTION 8: Return calculation (identical pattern — raw close, RETURN layer)
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
# SECTION 9: Beta computation (60-day EW adjClose market model — leg sizing)
# =============================================================================

def compute_betas(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict:
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)
    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj    = _adj_slice(ticker, beta_start, rebalance_date)
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
# SECTION 10: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    """One-way half-spread: 100bps at $100M -> 20bps at $2B (mcap-tiered ONLY)."""
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame, betas: dict) -> tuple:
    """
    Long TOP quintile ILLIQ (illiquid), short BOTTOM quintile (liquid).
    Min 20/leg via quintile; widen to terciles if quintile leg <20 (expected
    to fire rarely per this trial's construction).
    """
    df = signals_df.dropna(subset=["illiq", "raw_return"]).copy()
    n  = len(df)
    if n < 6:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, False

    leg_quintile = int(np.ceil(n * LEG_FRAC))
    guard_fired  = leg_quintile < LEG_MIN_NAMES
    leg          = int(np.ceil(n / 3.0)) if guard_fired else leg_quintile
    leg          = min(leg, n // 2)
    if leg < 1:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, guard_fired

    ranked = df.sort_values("illiq", ascending=False)   # high ILLIQ first
    longs  = ranked.iloc[:leg]["ticker"].tolist()    # illiquid = long
    shorts = ranked.iloc[-leg:]["ticker"].tolist()   # liquid   = short

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

    period_records: list  = []
    long_spread_records: list = []   # per-name spread within the long leg, every period
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

        illiq_vals = [compute_illiq_signal(row["ticker"], t) for _, row in univ.iterrows()]
        univ = univ.copy()
        univ["illiq"] = illiq_vals
        n_with_signal = int(univ["illiq"].notna().sum())
        print(f"  ILLIQ: {n_with_signal}/{len(univ)} with signal")

        try:
            returns_df = compute_returns(univ, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        betas = compute_betas(univ, t)

        (longs, shorts, lw, sw, long_scale, short_scale,
         mean_beta_long, mean_beta_short, guard_fired) = build_portfolio(returns_df, betas)
        if not longs:
            print(f"  [SKIP] n<6 after signal filter")
            continue
        if guard_fired:
            guard_fire_count += 1
            print(f"  [GUARD] quintile leg <20 names → widened to terciles (leg={len(longs)})")

        # ── Sign-convention audit ────────────────────────────────────────────
        illiq_map = returns_df.set_index("ticker")["illiq"]
        mean_illiq_long  = float(illiq_map.reindex(longs).mean())
        mean_illiq_short = float(illiq_map.reindex(shorts).mean())
        sign_ok = mean_illiq_long > mean_illiq_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg ILLIQ ({mean_illiq_long:.2e}) "
                  f"<= short-leg ILLIQ ({mean_illiq_short:.2e}) ***")

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

        # ── Long-leg AND short-leg spread/borrow/ADV distribution (bps) ──────
        mcap_map = returns_df.set_index("ticker")["mcap"]
        dvol_map = returns_df.set_index("ticker")["avg_dvol"]
        for tk in longs:
            long_spread_records.append({
                "rebalance_date": t, "leg": "long",
                "spread_bps": _spread(mcap_map.get(tk, np.nan)) * 10_000,
                "borrow_bps": _borrow(mcap_map.get(tk, np.nan)) * 10_000,
                "avg_dvol": dvol_map.get(tk, np.nan),
                "illiq": illiq_map.get(tk, np.nan),
            })
        for tk in shorts:
            long_spread_records.append({
                "rebalance_date": t, "leg": "short",
                "spread_bps": _spread(mcap_map.get(tk, np.nan)) * 10_000,
                "borrow_bps": _borrow(mcap_map.get(tk, np.nan)) * 10_000,
                "avg_dvol": dvol_map.get(tk, np.nan),
                "illiq": illiq_map.get(tk, np.nan),
            })

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

        valid_sig = returns_df.dropna(subset=["illiq", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["illiq"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} "
              f"to={avg_turnover:.0%} β_L={mean_beta_long:.2f} β_S={mean_beta_short:.2f} "
              f"IC={ic:.3f} Dlist={n_delisted} illiqL={mean_illiq_long:.2e} illiqS={mean_illiq_short:.2e}")

        period_records.append({
            "rebalance_date": t, "hold_end": hold_end,
            "n_universe": len(returns_df), "n_with_signal": n_with_signal,
            "n_delisted": n_delisted,
            "long_ret": long_net, "short_ret": short_net,
            "ls_ret": ls_net, "ls_gross": ls_gross, "ls_cost": ls_cost,
            "ls_ba_cost": ls_ba, "ls_borrow_cost": ls_bw,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
            "avg_turnover": avg_turnover, "mkt_ret": mkt_ret, "ic": ic,
            "n_long": len(longs), "n_short": len(shorts), "guard_fired": guard_fired,
            "mean_illiq_long": mean_illiq_long, "mean_illiq_short": mean_illiq_short,
            "sign_ok": sign_ok,
        })

    elapsed = time.time() - t_total
    return {
        "period_records": period_records, "long_spread_records": long_spread_records,
        "sector_contrib": sector_contrib,
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
    records = results["period_records"]
    spreads = results["long_spread_records"]
    elapsed = results["elapsed_s"]
    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df  = pd.DataFrame(records)
    sdf = pd.DataFrame(spreads)

    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m          = compute_metrics(ls_series)
    m_gross    = compute_metrics(gross_series)
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

    sep = "=" * 78

    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT")
    print(sep)
    print(f"  Periods where long-leg ILLIQ <= short-leg ILLIQ (violation): "
          f"{sign_violations}/{n_periods_run}")
    print("  PASS — long leg is always more illiquid than short leg, as intended."
          if sign_violations == 0 else
          "  *** FAIL — sign convention broken. Results may be inverted. ***")

    print(f"\n{sep}")
    print("NAME-COUNT GUARD")
    print(sep)
    print(f"  Quintile-leg<20 -> tercile-widen fired: {guard_fire_count}/{n_periods_run} periods "
          f"({100*guard_fire_count/max(n_periods_run,1):.0f}%) — expected to be rare.")

    print(f"\n{sep}")
    print("LONG-LEG vs SHORT-LEG LIQUIDITY PROFILE  (the core of this cost-stress test)")
    print(sep)
    if not sdf.empty:
        for leg_name in ["long", "short"]:
            sub = sdf[sdf["leg"] == leg_name]
            print(f"\n  {leg_name.upper()} LEG (n={len(sub):,} name-periods):")
            print(f"    Mean cap-tiered spread (bps, one-way half-spread): {sub['spread_bps'].mean():>8.1f}")
            print(f"    Median cap-tiered spread (bps):                   {sub['spread_bps'].median():>8.1f}")
            print(f"    P90 cap-tiered spread (bps):                      {sub['spread_bps'].quantile(0.9):>8.1f}")
            print(f"    Mean annual borrow rate (bps):                    {sub['borrow_bps'].mean():>8.1f}")
            print(f"    Mean 60-day ADV ($):                              {sub['avg_dvol'].mean():>14,.0f}")
            print(f"    Median 60-day ADV ($):                            {sub['avg_dvol'].median():>14,.0f}")
        print(f"\n  LIMITATION: the spread figures above are the harness's MARKET-CAP-TIERED\n"
              f"  assumption (20bps@$2B -> 100bps@$100M), NOT a direct function of realized\n"
              f"  ILLIQ or ADV. Since the long leg is selected explicitly on ILLIQ (not cap),\n"
              f"  two long-leg names can share a cap tier while differing by an order of\n"
              f"  magnitude in true ADV — the cap-tiered spread is almost certainly OPTIMISTIC\n"
              f"  for the long leg specifically. No finer-grained (ILLIQ-conditional) spread\n"
              f"  data is available in this pipeline to correct for this. Net-return figures\n"
              f"  below should be read as a LOWER BOUND on true cost drag for this trial.")

    print(f"\n{sep}")
    print("S10: AMIHUD ILLIQUIDITY PREMIUM — IN-SAMPLE RESULTS (independent trial #15)")
    print("COST-MODEL STRESS TEST — not expected to pass; the gross-to-net gap IS the finding.")
    print("Long top quintile ILLIQ (illiquid) | Short bottom quintile (liquid) | Quarterly")
    print(sep)

    print(f"\n  *** INFORMATION COEFFICIENT ***")
    print(f"  Mean IC (Spearman, ILLIQ vs ret)       : {mean_ic:>12.4f}")
    print(f"  IC t-stat                              : {ic_tstat:>12.3f}")
    print(f"  IC > 0 = illiquid names outperformed (liquidity premium confirmed in rank terms).")

    print(f"\n{sep}")
    print("GROSS-TO-NET COST EROSION  (the primary finding of this trial)")
    print(sep)
    gross_ann = m_gross["ann_ret"]
    net_ann   = m["ann_ret"]
    print(f"  {'Annualized return, GROSS (before any cost)':<48} {gross_ann:>13.2%}")
    print(f"  {'Annualized cost drag (bid-ask + borrow)':<48} {annual_cost_drag/10000:>13.2%}")
    print(f"    {'  bid-ask (bps/yr)':<46} {mean_ba_bps:>13.0f}")
    print(f"    {'  borrow  (bps/yr)':<46} {mean_bw_bps:>13.0f}")
    print(f"  {'Annualized return, NET (after cost)':<48} {net_ann:>13.2%}")
    if gross_ann > 0:
        pct_eaten = 100 * (annual_cost_drag / 10000) / gross_ann
        print(f"  {'Cost drag as % of GROSS return':<48} {pct_eaten:>12.0f}%")
        if pct_eaten >= 100:
            print(f"  -> Costs ate MORE than 100% of the gross premium — net is negative "
                  f"despite a positive gross signal.")
        else:
            print(f"  -> Costs ate {pct_eaten:.0f}% of the gross premium; {100-pct_eaten:.0f}% survived to net.")
    else:
        print(f"  {'Cost drag as % of GROSS return':<48} {'N/A (gross <= 0)':>13}")
        print(f"  -> Gross return itself is non-positive — there is no gross premium for "
              f"costs to erode; the signal is weak/absent at the gross level, independent "
              f"of the cost model.")

    print(f"\n  {'Metric':<46} {'Value':>14}")
    print(f"  {'-'*46} {'-'*14}")
    print(f"  {'Total cumulative return (net)':<46} {m['total_ret']:>13.2%}")
    print(f"  {'Annualized return (net)':<46} {m['ann_ret']:>13.2%}")
    print(f"  {'Annualized return (GROSS)':<46} {m_gross['ann_ret']:>13.2%}")
    print(f"  {'Mean actual turnover per quarter':<46} {mean_turnover:>13.0%}")
    print(f"  {'Sharpe ratio (annualized, net)':<46} {m['sharpe']:>13.3f}")
    print(f"  {'Sharpe ratio (annualized, GROSS)':<46} {m_gross['sharpe']:>13.3f}")
    print(f"  {'CALMAR ratio (net)':<46} {m['calmar']:>13.3f}")
    print(f"  {'Max drawdown (net)':<46} {m['max_dd']:>13.2%}")
    print(f"  {'t-stat (mean quarterly net return)':<46} {m['t_stat']:>13.3f}")
    print(f"  {'Market beta (vs EW universe)':<46} {beta_stats['beta']:>13.3f}")
    print(f"    {'beta t-stat (should be ~0)':<44} {beta_stats['beta_t']:>13.3f}")
    print(f"  {'Win rate (net)':<46} {m['win_rate']:>13.1%}")
    print(f"  {'Periods (N)':<46} {m['n']:>13d}")

    print(f"\n  --- DSR Check (independent family, N_trials={N_TRIALS}, quarterly SR units) ---")
    print(f"  {'Per-period Sharpe (net)':<46} {per_period_sr:>13.4f}")
    print(f"  {'DSR threshold (per-quarter SR)':<46} {dsr_thr:>13.4f}")
    print(f"  {'Clears DSR':<46} {'YES' if clears_dsr else 'NO':>13}")

    abs_period_sum = float(ls_series.abs().sum())
    period_share = (ls_series.abs() / abs_period_sum) if abs_period_sum > 0 else ls_series * 0
    top_idx   = period_share.idxmax()
    top_share = float(period_share.loc[top_idx])
    top_date  = df.loc[top_idx, "rebalance_date"]
    print(f"\n{sep}")
    print("CONCENTRATION CHECK (single-episode artifact risk)")
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
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} {'TOver':>6} "
          f"{'Mkt':>7} {'IC':>6} {'nL':>3} {'Guard':>5}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*6} {'-'*3} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
              f"{r['avg_turnover']:>5.0%} {r['mkt_ret']:>6.2%} {r['ic']:>6.3f} "
              f"{int(r['n_long']):>3} {'Y' if r['guard_fired'] else '':>5}")

    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (quarterly, net):")
    for i, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " <- MAX DD" if abs(val - m["max_dd"]) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{marker}")

    # ── Verdict: weak signal vs cost-eaten-a-real-premium ────────────────────
    print(f"\n{sep}")
    print("VERDICT LOGIC — weak signal vs. cost model eating a genuine premium")
    print(sep)
    ic_significant = (not np.isnan(ic_tstat)) and abs(ic_tstat) >= 2.0
    ic_inverted    = ic_significant and mean_ic < 0
    gross_meaningfully_positive = gross_ann > 0.02   # >2%/yr gross, a low bar
    if ic_inverted:
        diagnosis = ("HYPOTHESIS REJECTED, NOT A COST-MODEL FINDING — IC is significant AND "
                      "NEGATIVE: illiquid names had LOWER forward returns than liquid names in "
                      "this sample, the opposite of the hypothesized illiquidity premium. This "
                      "is a clean sign-reversal of the underlying effect in this small/mid-cap "
                      "universe/window, not an ambiguous or cost-eaten result — the cost model "
                      "is irrelevant here since the gross-level relationship already runs the "
                      "wrong way.")
    elif ic_significant and gross_meaningfully_positive:
        diagnosis = ("GENUINE GROSS PREMIUM, ERODED BY COSTS — IC is significant and gross "
                      "return is meaningfully positive; the net failure (if any) is a cost-model "
                      "finding, not a signal-quality finding.")
    elif ic_significant and not gross_meaningfully_positive:
        diagnosis = ("SIGNIFICANT IC BUT NO GROSS PREMIUM — rank predictability exists but "
                      "doesn't translate into a positive extreme-quintile spread gross of costs; "
                      "inconclusive on the cost-model question specifically.")
    else:
        diagnosis = ("WEAK/INSIGNIFICANT SIGNAL — IC is not statistically significant "
                      "(|t|<2) and/or gross return is not meaningfully positive; a net failure "
                      "here reflects a weak signal, not a cost-model artifact.")
    print(f"  {diagnosis}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def save_results(results: dict) -> None:
    records = results["period_records"]
    spreads = results["long_spread_records"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "s10_period_returns.csv", index=False)
    pd.DataFrame(spreads).to_csv(RESULTS / "s10_leg_spread_distribution.csv", index=False)
    sector_df = pd.DataFrame(
        [{"sic_major_group": k, "contribution": v}
         for k, v in results.get("sector_contrib", {}).items()]
    ).sort_values("contribution", ascending=False)
    sector_df.to_csv(RESULTS / "s10_sector_contribution.csv", index=False)
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
    sign_violations  = int(results["sign_violations"])
    guard_fire_count = int(results["guard_fire_count"])
    n_periods_run    = len(df)

    gross_ann = m_gross["ann_ret"]
    ic_significant = (not np.isnan(ic_tstat)) and abs(ic_tstat) >= 2.0
    ic_inverted    = ic_significant and mean_ic < 0
    gross_meaningfully_positive = gross_ann > 0.02
    if ic_inverted:
        diagnosis_tag = "HYPOTHESIS REJECTED — IC significant AND NEGATIVE (illiquidity premium inverted, not a cost-model finding)"
    elif ic_significant and gross_meaningfully_positive:
        diagnosis_tag = "COSTS ERODED A GENUINE GROSS PREMIUM"
    elif ic_significant and not gross_meaningfully_positive:
        diagnosis_tag = "SIGNIFICANT IC, NO GROSS PREMIUM (inconclusive on cost model)"
    else:
        diagnosis_tag = "WEAK SIGNAL (not a cost-model artifact)"

    verdict_bits = ["CLEARS DSR" if clears_dsr else "FAILS DSR", diagnosis_tag]
    if sign_violations > 0:
        verdict_bits.append(f"SIGN AUDIT FAILED ({sign_violations} periods)")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "illiquidity_S10",
        "hypothesis":           (
            "Amihud illiquidity premium: long high-ILLIQ (top quintile), short "
            "low-ILLIQ (bottom quintile). Explicit cost-model stress test — "
            "run to reveal the gross-to-net cost erosion gap for the least-"
            "liquid corner of the universe, not expected to pass net of costs. "
            "Independent trial #15, final program signal."
        ),
        "data_source":          "Tiingo (adjClose for return, raw close x raw volume for $vol) + SEC XBRL (PIT shares)",
        "status":               "completed_in_sample",
        "trial_number":         TRIAL_NUMBER,
        "trial_note":           "Independent trial #15. Cost-model stress test — final program signal.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "quarterly",
        "rebalance_note":       "Matches S1/S3/S5 cadence for comparability.",
        "portfolio_note":       (
            f"Beta-neutral (EW market proxy, 60-day adjClose). No liquidity floor beyond "
            f"standard universe screen (by design). Guard fired {guard_fire_count}/{n_periods_run} periods."
        ),
        "signal_note":          (
            "ILLIQ = mean(|adjClose return| / (raw close x raw volume)) over trailing "
            "21 trading days; long=high ILLIQ, short=low ILLIQ."
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
        reg_df = reg_df[reg_df["study"] != "illiquidity_S10"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] illiquidity_S10 logged (trial #{TRIAL_NUMBER}) → {REGISTRY}")
    print(f"[Verdict] {verdict}")


# =============================================================================
# SECTION 14: Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S10: AMIHUD ILLIQUIDITY PREMIUM  (independent trial #15 — COST-MODEL STRESS TEST)")
    print("Not expected to pass — the gross-to-net cost erosion gap IS the finding.")
    print("Long top quintile ILLIQ (illiquid) | Short bottom quintile (liquid)")
    print("Quarterly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 24 periods")
    print(f"DSR: N_TRIALS={N_TRIALS} (new, independent family, final program signal)")
    print("=" * 78)

    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = load_prefilter()
    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close+volume prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle (ILLIQ return numerator + beta)...")
    preload_all_adj_prices(survivor_tickers)

    print(f"\n[Step 4] Loading SEC shares facts ({len(survivor_ciks):,} CIKs)...")
    preload_sec_facts(survivor_ciks)

    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} quarterly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    print_results(results)
    save_results(results)
    log_registry(results)


if __name__ == "__main__":
    main()
