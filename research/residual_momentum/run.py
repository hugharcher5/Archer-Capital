"""
S6: Residual Momentum  (same DSR family as S9 — trial #6, N_TRIALS unchanged)
==============================================================================
Signal   : 12-1 residual momentum. For each stock, run a rolling single-factor
           market-model regression (daily adjClose returns vs an equal-weight
           market factor built from the full SEC-prefilter survivor universe)
           over a ~36-month window (min 24 months of history required).
           Residual_t = actual_return_t − (alpha + beta * market_return_t).
           Ranking signal = SUM of daily residuals over [t-12M, t-1M]
           (skip the most recent month, same convention as S9).

Factor model note: FF3 (SMB/HML) factor data is NOT available anywhere in
this pipeline (checked — no cached Fama-French series). A single-factor
market model is used instead, as explicitly permitted by the trial spec.
Market factor = EW daily adjClose return across all ~3,300 SEC-prefilter
survivors (broader than the current $100M-2B investable universe, built
once over the full 2012-2021 history) — a more stable factor series than
re-anchoring to the shifting quarterly investable set.

Frequency: DAILY returns (not weekly) — held fixed for the whole trial.

adjClose used throughout — for the factor regression AND the residual
momentum ranking window — applying the S9 split-adjustment fix from the
start (S9 originally had a raw-close split-contamination bug; here we use
adjClose end to end so it never occurs).

UNIVERSE & REBALANCE: identical to S9 — PIT $100M-2B mcap, ADTV >= $1M/day,
price >= $1, survivorship-bias-free, SEC-filing-gated PIT shares outstanding.
Rebalance: quarterly. Hold: 1 quarter (3 months) — pre-registered to match
S9's ACTUAL cadence exactly (S9 rebalances quarterly with a 3-month hold,
not monthly — matching this is what makes the crash-window, period-by-period
comparison to S9 meaningful; user confirmed this choice explicitly given the
ambiguity in "recommend 1 month" vs S9's true quarterly design).

CONSTRUCTION
  - Long top quintile of cumulative residual return, short bottom quintile.
  - Name-count guard: quintile leg < 20 names -> widen to terciles for that
    period only (logged, not a new trial). Same convention as S5.
  - Beta-neutral leg sizing (60-day OLS vs EW universe adjClose, same as
    gp/ivol/max) is PRIMARY. Because residual momentum is beta-stripped by
    construction, it should already carry near-zero market beta — so an
    EQUAL-DOLLAR (50/50) comparison variant is also computed each period
    using the identical long/short membership, purely as a diagnostic: if
    beta-neutral and equal-dollar sizing diverge meaningfully, that itself
    is a finding (it would mean the ex-post realized beta of the legs isn't
    actually near zero, contradicting the "beta-stripped by construction"
    assumption).
  - Minimum-history filter (>=24mo) thins the eligible universe early in
    the sample (2015 has less trailing history available than 2018) —
    reported explicitly per period.

DSR: this trial shares the momentum family with S9 (trial #6). It does NOT
consume a new trial slot — N_TRIALS stays 6. The S9 per-quarter DSR
threshold (N_TRIALS=6, N_obs=24) is reused verbatim and cross-checked
against the value already logged in the registry for momentum_jt_S9.

No-survivorship guarantee: delisted tickers retained with returns to last
trading day; universe rebuilt PIT from SEC filings each quarter.

Namespacing: research/residual_momentum/ — fully separate cache/results
from any concurrently running trial.
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

MGMT_PIT   = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE      = MGMT_PIT / "cache"                                    # shared, read-only
OWN_CACHE  = Path(__file__).resolve().parent / "cache"             # S6-private
RESULTS    = Path(__file__).resolve().parent / "results"
REGISTRY   = Path(__file__).resolve().parents[1] / "trial_registry.csv"
S9_RESULTS = Path(__file__).resolve().parents[1] / "momentum_jt" / "results" / "s9_period_returns.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S9) ────────────────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── Factor-regression window ─────────────────────────────────────────────────
TRADING_DAYS_PER_MONTH = 21
REG_WINDOW_MONTHS = 36
REG_MIN_MONTHS    = 24
REG_MIN_DAYS      = REG_MIN_MONTHS * TRADING_DAYS_PER_MONTH   # 504

# ── Residual-momentum ranking window (t-12M to t-1M, skip most recent month) ─
RESID_MOM_MIN_DAYS = 180     # same threshold convention as S9's raw momentum

# ── Beta window for LEG-SIZING (separate from the factor-regression window) ──
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Backtest dates (identical to S9: 24 quarterly periods) ───────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])

# ── Cost model (identical to S9/gp) ───────────────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR: same momentum family as S9 — trial #6, N_TRIALS unchanged ──────────
TRIAL_NUMBER = 6
N_TRIALS     = 6   # H1 + E1 + E2 + E3 + E4 + S9/S6 (shared family slot)

# ── Crash windows (from S9's actual results, for the primary comparison) ────
CRASH_2016_START = pd.Timestamp("2016-01-01")
CRASH_2016_END   = pd.Timestamp("2017-04-01")   # 6 consecutive S9 losing quarters
CRASH_Q4_2020    = pd.Timestamp("2020-10-01")   # S9's -15.6% quarter


# =============================================================================
# SECTION 2: Tiingo / SEC helpers (identical pattern to S9/gp)
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
# SECTION 4: AdjClose cache (SIGNAL layer — factor regression + momentum)
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


def _adj_returns(ticker: str) -> pd.Series:
    """Full-history daily adjClose returns for a ticker, deduped/sorted."""
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return pd.Series(dtype=float)
    s = df.set_index("date")["adjClose"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.pct_change().dropna()


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 5: Market factor (single-factor model — EW across all survivors)
# =============================================================================

_MKT_FACTOR: pd.Series = pd.Series(dtype=float)


def build_market_factor(tickers: list[str]) -> pd.Series:
    """
    EW daily adjClose-return market factor, built ONCE over the full
    2012-2021 history from every SEC-prefilter survivor with adjClose data
    (~3,300 tickers — broader than any single quarter's $100M-2B investable
    set, so the factor doesn't re-anchor to a shifting small subset).
    Clipped at +/-50% per name per day to limit single-name blowups from
    corporate actions from dominating the EW average.
    """
    t0 = time.time()
    series_map: dict[str, pd.Series] = {}
    for t in tickers:
        r = _adj_returns(t)
        if len(r) >= REG_MIN_DAYS:
            series_map[t] = r
    if not series_map:
        raise RuntimeError("No tickers with sufficient adjClose history to build market factor.")
    mat = pd.concat(series_map, axis=1)
    mkt = mat.clip(-0.5, 0.5).mean(axis=1).dropna()
    print(f"[MktFactor] Built EW market factor from {len(series_map):,} tickers, "
          f"{len(mkt):,} trading days ({mkt.index.min().date()} -> {mkt.index.max().date()}) "
          f"in {time.time()-t0:.1f}s")
    return mkt


# =============================================================================
# SECTION 6: SEC facts (shares outstanding only — reuse S9's trimmed cache)
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
        print(f"[Preload] SEC facts: {loaded:,} loaded from trimmed cache in {time.time()-t0:.1f}s")
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
# SECTION 7: Universe builder (identical to S9)
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
# SECTION 8: Residual momentum signal
# =============================================================================

def compute_residual_momentum(
    ticker: str,
    rebalance_date: pd.Timestamp,
    mkt_factor: pd.Series,
) -> tuple:
    """
    1. Rolling regression window: [t-36M, t] of daily adjClose returns vs
       mkt_factor. Requires >= REG_MIN_DAYS (24mo) observations.
    2. Residual_d = actual_d - (alpha + beta * mkt_d) for every day in the
       regression window.
    3. Ranking signal = SUM of residuals over [t-12M, t-1M] (skip most
       recent month, same convention as S9). Requires >= RESID_MOM_MIN_DAYS.

    Returns (signal, beta_used) or (None, None) if insufficient history.
    """
    reg_start = rebalance_date - pd.DateOffset(months=REG_WINDOW_MONTHS)
    ret = _adj_returns(ticker)
    if ret.empty:
        return None, None
    window_ret = ret[(ret.index >= reg_start) & (ret.index <= rebalance_date)]
    if len(window_ret) < REG_MIN_DAYS:
        return None, None

    x = mkt_factor.reindex(window_ret.index).dropna()
    y = window_ret.reindex(x.index)
    if len(y) < REG_MIN_DAYS:
        return None, None

    X = np.column_stack([np.ones(len(x)), x.values])
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(X, y.values, rcond=None)
    except Exception:
        return None, None
    if rank < 2:
        return None, None
    alpha, beta = float(coeffs[0]), float(coeffs[1])

    y_hat = X @ coeffs
    resid = pd.Series(y.values - y_hat, index=y.index)

    mom_end   = rebalance_date - pd.DateOffset(months=1)
    mom_start = rebalance_date - pd.DateOffset(months=12)
    resid_window = resid[(resid.index >= mom_start) & (resid.index <= mom_end)]
    if len(resid_window) < RESID_MOM_MIN_DAYS:
        return None, None

    return float(resid_window.sum()), beta


# =============================================================================
# SECTION 9: Return calculation (identical to S9 — raw close, RETURN layer)
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
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
# SECTION 10: Beta computation for LEG-SIZING (60-day, same as gp/ivol/max)
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
# SECTION 11: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame, betas: dict) -> tuple:
    """
    Long TOP quintile residual momentum (winners), short BOTTOM quintile
    (losers) — same sign convention as S9. Name-count guard: quintile leg
    < 20 -> widen to terciles for this period only.
    """
    df = signals_df.dropna(subset=["resid_mom", "raw_return"]).copy()
    n  = len(df)
    if n < 6:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, False

    leg_quintile = int(np.ceil(n * 0.20))
    guard_fired  = leg_quintile < 20
    leg          = int(np.ceil(n / 3.0)) if guard_fired else leg_quintile
    leg          = min(leg, n // 2)
    if leg < 1:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, guard_fired

    ranked = df.sort_values("resid_mom", ascending=False)   # high -> low
    longs  = ranked.iloc[:leg]["ticker"].tolist()    # winners = long
    shorts = ranked.iloc[-leg:]["ticker"].tolist()   # losers  = short

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
    returns_df: pd.DataFrame, tickers: list, weights: dict,
    is_short: bool, turnover: float,
) -> tuple:
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
# SECTION 12: Backtest engine
# =============================================================================

def run_backtest(tiingo_tickers: pd.DataFrame, prefilter_df: pd.DataFrame) -> dict:
    t_total = time.time()

    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    all_survivor_tickers = survivors["ticker"].tolist()

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    print("\n[MktFactor] Building single-factor market model factor...")
    mkt_factor = build_market_factor(all_survivor_tickers)

    period_records: list   = []
    coverage_records: list = []
    universe_audit: list   = []
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

        # ── Residual momentum signal (min-history filter applies here) ──────
        signals, resid_betas = [], []
        for _, row in univ.iterrows():
            sig, beta_reg = compute_residual_momentum(row["ticker"], t, mkt_factor)
            signals.append(sig)
            resid_betas.append(beta_reg)
        univ = univ.copy()
        univ["resid_mom"]  = signals
        univ["reg_beta"]   = resid_betas
        n_with_signal = int(univ["resid_mom"].notna().sum())
        n_excluded    = len(univ) - n_with_signal
        pct_excluded  = 100 * n_excluded / max(len(univ), 1)
        print(f"  ResidMom: {n_with_signal}/{len(univ)} with signal "
              f"({pct_excluded:.0f}% excluded — insufficient rolling history)")

        coverage_records.append({
            "rebalance_date": t, "n_universe": len(univ),
            "n_with_signal": n_with_signal, "n_excluded": n_excluded,
            "pct_excluded": pct_excluded,
        })

        try:
            returns_df = compute_returns(univ, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        universe_audit.append({
            "rebalance_date": t, "hold_end": hold_end, "n_universe": len(returns_df),
            "n_with_signal": n_with_signal, "n_delisted": n_delisted,
            "n_valid_returns": int(returns_df["raw_return"].notna().sum()),
        })

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
        mean_rm_long  = float(returns_df.set_index("ticker")["resid_mom"].reindex(longs).mean())
        mean_rm_short = float(returns_df.set_index("ticker")["resid_mom"].reindex(shorts).mean())
        sign_ok = mean_rm_long > mean_rm_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg resid-mom ({mean_rm_long:.4f}) "
                  f"<= short-leg resid-mom ({mean_rm_short:.4f}) ***")

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

        # ── Equal-dollar (50/50) comparison, same leg membership ────────────
        ls_net_eqdollar = 1.0 * long_net + 1.0 * short_net

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

        valid_sig = returns_df.dropna(subset=["resid_mom", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["resid_mom"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} "
              f"eqDollar={ls_net_eqdollar:+.2%} to={avg_turnover:.0%} "
              f"β_L={mean_beta_long:.2f} β_S={mean_beta_short:.2f} IC={ic:.3f} "
              f"Dlist={n_delisted} rmL={mean_rm_long:.4f} rmS={mean_rm_short:.4f}")

        period_records.append({
            "rebalance_date": t, "hold_end": hold_end,
            "n_universe": len(returns_df), "n_with_signal": n_with_signal,
            "n_delisted": n_delisted, "pct_excluded": pct_excluded,
            "long_ret": long_net, "short_ret": short_net,
            "ls_ret": ls_net, "ls_gross": ls_gross, "ls_cost": ls_cost,
            "ls_ba_cost": ls_ba, "ls_borrow_cost": ls_bw,
            "ls_ret_eqdollar": ls_net_eqdollar,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
            "long_turnover": long_turnover, "short_turnover": short_turnover,
            "avg_turnover": avg_turnover, "mkt_ret": mkt_ret, "ic": ic,
            "n_long": len(longs), "n_short": len(shorts), "guard_fired": guard_fired,
            "mean_rm_long": mean_rm_long, "mean_rm_short": mean_rm_short, "sign_ok": sign_ok,
        })

    elapsed = time.time() - t_total
    return {
        "period_records": period_records, "universe_audit": universe_audit,
        "coverage_records": coverage_records, "sector_contrib": sector_contrib,
        "sign_violations": sign_violations, "guard_fire_count": guard_fire_count,
        "elapsed_s": elapsed,
    }


# =============================================================================
# SECTION 13: Metrics (identical formulas to S9/gp)
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
# SECTION 14: S6 vs S9 crash-window comparison
# =============================================================================

def load_s9_returns() -> Optional[pd.DataFrame]:
    if not S9_RESULTS.exists():
        print(f"[WARN] S9 results not found at {S9_RESULTS} — crash comparison skipped.")
        return None
    df = pd.read_csv(S9_RESULTS, parse_dates=["rebalance_date", "hold_end"])
    return df


def print_crash_comparison(df6: pd.DataFrame) -> None:
    df9 = load_s9_returns()
    sep = "=" * 78
    print(f"\n{sep}")
    print("S6 vs S9 — DIRECT COMPARISON (residual vs raw 12-1 momentum)")
    print(sep)
    if df9 is None:
        print("  S9 results unavailable — skipping.")
        return

    merged = df6[["rebalance_date", "ls_ret", "ic"]].merge(
        df9[["rebalance_date", "ls_ret", "ic"]],
        on="rebalance_date", suffixes=("_s6", "_s9"), how="inner")

    m6 = compute_metrics(df6["ls_ret"])
    m9 = compute_metrics(df9["ls_ret"])
    ic6 = df6["ic"].dropna().mean()
    ic9 = df9["ic"].dropna().mean()

    print(f"\n  {'Metric':<32} {'S6 (residual)':>16} {'S9 (raw)':>16}")
    print(f"  {'-'*32} {'-'*16} {'-'*16}")
    print(f"  {'Mean IC':<32} {ic6:>16.4f} {ic9:>16.4f}")
    print(f"  {'Sharpe (annualized, net)':<32} {m6['sharpe']:>16.3f} {m9['sharpe']:>16.3f}")
    print(f"  {'CALMAR':<32} {m6['calmar']:>16.3f} {m9['calmar']:>16.3f}")
    print(f"  {'Max drawdown':<32} {m6['max_dd']:>16.2%} {m9['max_dd']:>16.2%}")
    print(f"  {'Annualized return':<32} {m6['ann_ret']:>16.2%} {m9['ann_ret']:>16.2%}")
    print(f"  {'Win rate':<32} {m6['win_rate']:>16.1%} {m9['win_rate']:>16.1%}")

    print(f"\n  --- Crash window 1: 2016 post-Brexit reversal "
          f"({CRASH_2016_START.date()} -> {CRASH_2016_END.date()}, S9's 6 losing quarters) ---")
    w1 = merged[(merged["rebalance_date"] >= CRASH_2016_START)
                & (merged["rebalance_date"] <= CRASH_2016_END)]
    print(f"  {'Date':<12} {'S6 L/S':>9} {'S9 L/S':>9}")
    for _, r in w1.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<12} {r['ls_ret_s6']:>8.2%} {r['ls_ret_s9']:>8.2%}")
    s6_losses = int((w1["ls_ret_s6"] < 0).sum())
    s9_losses = int((w1["ls_ret_s9"] < 0).sum())
    s6_cum = float((1 + w1["ls_ret_s6"]).prod() - 1)
    s9_cum = float((1 + w1["ls_ret_s9"]).prod() - 1)
    print(f"  Losing quarters: S6={s6_losses}/{len(w1)}  S9={s9_losses}/{len(w1)}")
    print(f"  Cumulative return over window: S6={s6_cum:.2%}  S9={s9_cum:.2%}")

    print(f"\n  --- Crash window 2: Q4-2020 vaccine rotation "
          f"({CRASH_Q4_2020.date()}, S9's -15.6% quarter) ---")
    w2 = merged[merged["rebalance_date"] == CRASH_Q4_2020]
    for _, r in w2.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<12} S6={r['ls_ret_s6']:>8.2%}  S9={r['ls_ret_s9']:>8.2%}")

    if not w2.empty:
        s6_q4 = float(w2.iloc[0]["ls_ret_s6"])
        s9_q4 = float(w2.iloc[0]["ls_ret_s9"])
        # ratio of loss magnitudes (both negative here) — how much of S9's
        # drawdown shows up in S6, independent of sign edge cases.
        ratio = (s6_q4 / s9_q4) if s9_q4 < 0 else np.nan
        if s6_q4 >= -0.03 or (not np.isnan(ratio) and ratio <= 0.15):
            verdict = "NOT REPLICATED — residual momentum largely avoided this crash"
        elif not np.isnan(ratio) and ratio <= 0.70:
            verdict = f"MUTED — residual momentum lost {ratio:.0%} of S9's drawdown magnitude"
        else:
            verdict = f"REPLICATED — residual momentum crashed comparably ({ratio:.0%} of S9's magnitude)" if not np.isnan(ratio) else "REPLICATED — residual momentum crashed comparably"
        print(f"  Verdict: {verdict}")


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

    df  = pd.DataFrame(records)
    cov = pd.DataFrame(coverage)

    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    eqdollar     = df["ls_ret_eqdollar"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m          = compute_metrics(ls_series)
    m_gross    = compute_metrics(gross_series)
    m_eqdollar = compute_metrics(eqdollar)
    beta_stats = compute_market_beta(ls_series, mkt_series)
    beta_eq    = compute_market_beta(eqdollar, mkt_series)

    obs_sr_q = m["sharpe"] / np.sqrt(4) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr  = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (bool(obs_sr_q > dsr_thr)
                  if not np.isnan(obs_sr_q) and not np.isnan(dsr_thr) else False)

    # Cross-check against S9's actual logged threshold
    s9_thr = None
    if REGISTRY.exists():
        reg = pd.read_csv(REGISTRY)
        s9_row = reg[reg["study"] == "momentum_jt_S9"]
        if not s9_row.empty:
            s9_thr = float(s9_row.iloc[0]["dsr_threshold"])

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
    print(f"  Periods where long-leg resid-mom <= short-leg resid-mom (violation): "
          f"{sign_violations}/{n_periods_run}")
    print("  PASS — long leg is highest resid-mom every period, as intended."
          if sign_violations == 0 else
          "  *** FAIL — sign convention broken. Results may be inverted. ***")

    print(f"\n{sep}")
    print("MINIMUM-HISTORY COVERAGE (early-period thinness)")
    print(sep)
    first_2015 = cov[cov["rebalance_date"].dt.year == 2015]
    first_2018 = cov[cov["rebalance_date"].dt.year == 2018]
    if not first_2015.empty:
        print(f"  2015 mean exclusion (insufficient 24mo+ history): "
              f"{first_2015['pct_excluded'].mean():.1f}%")
    if not first_2018.empty:
        print(f"  2018 mean exclusion (insufficient 24mo+ history): "
              f"{first_2018['pct_excluded'].mean():.1f}%")
    print(f"  Full-sample mean exclusion: {cov['pct_excluded'].mean():.1f}%")
    print(f"\n  Name-count guard (quintile leg<20 -> terciles) fired: "
          f"{guard_fire_count}/{n_periods_run} periods ({100*guard_fire_count/max(n_periods_run,1):.0f}%)")

    print(f"\n{sep}")
    print("S6: RESIDUAL MOMENTUM — IN-SAMPLE RESULTS (same DSR family as S9, trial #6)")
    print("Signal: SUM(daily residuals, t-12M..t-1M) from 36mo market-model regression")
    print("Long top quintile (winners) | Short bottom quintile (losers) | Quarterly")
    print(sep)

    print(f"\n  *** MARKET BETA — key test for residual momentum ***")
    print(f"  Market beta (beta-neutral sizing, vs EW universe): {beta_stats['beta']:>10.3f} "
          f"(t={beta_stats['beta_t']:.2f})")
    print(f"  Market beta (equal-dollar sizing,   vs EW universe): {beta_eq['beta']:>8.3f} "
          f"(t={beta_eq['beta_t']:.2f})")
    print(f"  Expectation: both should be near zero since the SIGNAL is already beta-stripped.")

    print(f"\n  *** INFORMATION COEFFICIENT ***")
    print(f"  Mean IC (Spearman, resid-mom vs ret)  : {mean_ic:>12.4f}")
    print(f"  IC t-stat                              : {ic_tstat:>12.3f}")

    print(f"\n  {'Metric':<46} {'Beta-Neutral':>14} {'Equal-Dollar':>14}")
    print(f"  {'-'*46} {'-'*14} {'-'*14}")
    print(f"  {'Total cumulative return (net)':<46} {m['total_ret']:>13.2%} {m_eqdollar['total_ret']:>13.2%}")
    print(f"  {'Annualized return (net)':<46} {m['ann_ret']:>13.2%} {m_eqdollar['ann_ret']:>13.2%}")
    print(f"  {'Annualized return (GROSS, beta-neutral only)':<46} {m_gross['ann_ret']:>13.2%} {'':>14}")
    print(f"  {'Cost drag (annualized bps, beta-neutral only)':<46} {annual_cost_drag:>13.0f} {'':>14}")
    print(f"    {'  bid-ask (bps/yr)':<44} {mean_ba_bps:>13.0f} {'':>14}")
    print(f"    {'  borrow  (bps/yr)':<44} {mean_bw_bps:>13.0f} {'':>14}")
    print(f"  {'Mean actual turnover per quarter':<46} {mean_turnover:>13.0%} {'':>14}")
    print(f"  {'Sharpe ratio (annualized, net)':<46} {m['sharpe']:>13.3f} {m_eqdollar['sharpe']:>13.3f}")
    print(f"  {'CALMAR ratio':<46} {m['calmar']:>13.3f} {m_eqdollar['calmar']:>13.3f}")
    print(f"  {'Max drawdown':<46} {m['max_dd']:>13.2%} {m_eqdollar['max_dd']:>13.2%}")
    print(f"  {'t-stat (mean quarterly net return)':<46} {m['t_stat']:>13.3f} {m_eqdollar['t_stat']:>13.3f}")
    print(f"  {'Win rate':<46} {m['win_rate']:>13.1%} {m_eqdollar['win_rate']:>13.1%}")
    print(f"  {'Periods (N)':<46} {m['n']:>13d} {'':>14}")
    beta_delta = abs(beta_stats['beta'] - beta_eq['beta']) if (not np.isnan(beta_stats['beta']) and not np.isnan(beta_eq['beta'])) else np.nan
    sharpe_delta = abs(m['sharpe'] - m_eqdollar['sharpe']) if (not np.isnan(m['sharpe']) and not np.isnan(m_eqdollar['sharpe'])) else np.nan
    print(f"\n  Beta-neutral vs equal-dollar beta delta : {beta_delta:.3f}")
    print(f"  Beta-neutral vs equal-dollar Sharpe delta: {sharpe_delta:.3f}")
    if not np.isnan(beta_delta) and beta_delta > 0.15:
        print(f"  *** NOTE: meaningful beta divergence between sizing schemes — "
              f"ex-post leg betas are NOT as flat as 'beta-stripped by construction' would imply. ***")
    else:
        print(f"  Sizing schemes broadly agree — consistent with signal being beta-stripped by construction.")

    print(f"\n  --- DSR Check (SHARED momentum family with S9 — N_TRIALS={N_TRIALS}, no new slot) ---")
    print(f"  {'Observed quarterly Sharpe (S6)':<46} {obs_sr_q:>13.4f}")
    print(f"  {'DSR threshold (reused from S9, N=6, n_obs=24)':<46} {dsr_thr:>13.4f}")
    if s9_thr is not None:
        print(f"  {'S9 logged threshold (registry cross-check)':<46} {s9_thr:>13.4f}")
        match = "MATCH" if abs(dsr_thr - s9_thr) < 1e-6 else "MISMATCH — investigate"
        print(f"  {'Cross-check':<46} {match:>13}")
    print(f"  {'Clears DSR':<46} {'YES' if clears_dsr else 'NO':>13}")

    # ── Concentration check ──────────────────────────────────────────────────
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
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'EqDlr':>7} {'Cost':>6} {'TOver':>6} "
          f"{'Mkt':>7} {'IC':>6} {'nL':>3} {'Excl%':>5} {'Guard':>5}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*7} {'-'*6} {'-'*3} {'-'*5} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_ret_eqdollar']:>6.2%} "
              f"{r['ls_cost']:>5.2%} {r['avg_turnover']:>5.0%} "
              f"{r['mkt_ret']:>6.2%} {r['ic']:>6.3f} {int(r['n_long']):>3} "
              f"{r['pct_excluded']:>4.0f}% {'Y' if r['guard_fired'] else '':>5}")

    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (quarterly):")
    for i, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " <- MAX DD" if abs(val - m["max_dd"]) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{marker}")

    print_crash_comparison(df)

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def save_results(results: dict) -> None:
    records  = results["period_records"]
    audit    = results["universe_audit"]
    coverage = results["coverage_records"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "s6_period_returns.csv", index=False)
    pd.DataFrame(audit).to_csv(RESULTS / "s6_universe_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "s6_coverage_audit.csv", index=False)
    sector_df = pd.DataFrame(
        [{"sic_major_group": k, "contribution": v}
         for k, v in results.get("sector_contrib", {}).items()]
    ).sort_values("contribution", ascending=False)
    sector_df.to_csv(RESULTS / "s6_sector_contribution.csv", index=False)
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

    obs_sr_q   = m["sharpe"] / np.sqrt(4) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = bool(obs_sr_q > dsr_thr) if not np.isnan(obs_sr_q) else False

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)

    annual_cost_drag = (float(df["ls_ba_cost"].mean()) + float(df["ls_borrow_cost"].mean())) * 4 * 10_000
    sign_violations  = int(results["sign_violations"])
    guard_fire_count = int(results["guard_fire_count"])
    n_periods_run    = len(df)

    verdict_bits = ["CLEARS DSR (shared momentum family, reused S9 threshold)"
                    if clears_dsr else "FAILS DSR (shared momentum family, reused S9 threshold)"]
    if sign_violations > 0:
        verdict_bits.append(f"SIGN AUDIT FAILED ({sign_violations} periods)")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "residmom_S6",
        "hypothesis":           (
            "Residual (beta-stripped) 12-1 momentum: long top quintile / short "
            "bottom quintile of cumulative daily residual return from a 36-month "
            "single-factor market-model regression (min 24mo history). Same "
            "momentum DSR family as S9 (trial #6) — direct test of raw vs "
            "residual momentum, does not consume a new trial slot."
        ),
        "data_source":          "Tiingo (adjClose for signal+regression, raw close for returns) + SEC XBRL (PIT shares)",
        "status":               "completed_in_sample",
        "trial_number":         TRIAL_NUMBER,
        "trial_note":           "Same DSR family as S9 (trial #6). N_TRIALS unchanged.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "quarterly",
        "rebalance_note":       "3-month hold, matching S9's actual cadence exactly for period-aligned crash-window comparison.",
        "portfolio_note":       (
            f"Beta-neutral (EW market proxy, 60-day adjClose) primary; equal-dollar "
            f"comparison computed alongside. Name-count guard fired "
            f"{guard_fire_count}/{n_periods_run} periods."
        ),
        "signal_note":          (
            "ResidMom = SUM(daily residual, t-12M..t-1M) from 36mo (min 24mo) "
            "single-factor market-model regression, adjClose throughout."
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
        "obs_sr_q":             obs_sr_q,
        "dsr_threshold":        dsr_thr,
        "clears_dsr":           clears_dsr,
        "notes":                verdict,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "residmom_S6"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] residmom_S6 logged (trial #{TRIAL_NUMBER}, shared family, N_TRIALS={N_TRIALS} unchanged) → {REGISTRY}")
    print(f"[Verdict] {verdict}")


# =============================================================================
# SECTION 16: Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S6: RESIDUAL MOMENTUM  (same DSR family as S9 — trial #6, no new slot)")
    print("Signal: SUM(daily residuals, t-12M..t-1M), 36mo market-model regression")
    print("Long top quintile (winners) | Short bottom quintile (losers)")
    print("Quarterly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 24 periods")
    print(f"DSR: N_TRIALS={N_TRIALS} (shared momentum family — reusing S9's threshold)")
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

    print(f"\n[Step 3b] Loading adjClose pickle (signal + regression layer)...")
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
