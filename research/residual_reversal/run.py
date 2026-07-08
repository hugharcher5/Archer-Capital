"""
S8: Residual Short-Term Reversal  (Trial #14, N_TRIALS=14)
===========================================================
Signal   : Negative of 1-month residual return from rolling 36-month OLS
           single-factor market-model regression (adjClose throughout, same
           factor construction as S6). Reversal: prior-month residual losers
           go long, winners go short.

Pre-registered double filter (both criteria required to trade a name):
  1. Top/bottom DECILE_FRAC (10%) of 1-month residual return
  2. |1-month residual return| >= VOL_FILTER_SIGMA (1.5) x trailing-12M
     monthly-residual std (std of 12 non-overlapping monthly residual sums
     over [t-13M, t-1M], excluding the current month).
Names NOT crossing both thresholds hold their existing book positions
(no transaction cost, no weight change beyond proportional re-normalization).

Rebalance: monthly (1-month hold). In-sample: 2015-01-01 -> 2021-01-01 (72).

Name-count guard: either leg (full book = active + carry) < NAME_COUNT_MIN
  -> skip rebalance, hold all existing positions, log fire rate.
  Do NOT loosen pre-registered threshold.

Beta-neutral sizing (60-day OLS vs EW universe adjClose, same as S6/gp/ivol).
Beta clipped [0.2, 3.0]; fallback 1.0 if denom <= 0.05.

Secondary informational: naive full-book (all decile names, no vol filter,
full monthly churn) computed in parallel; reported but does not affect DSR.

Falsification diagnostics (pre-registered):
  1. Sign-convention audit: long leg must have negative mean resid_1m
  2. Concentration check: no single month >30% of total |return| mass
  3. Liquidity-shock isolation: is return concentrated in lowest-ADTV tercile?

DSR: N_TRIALS=14 (one new slot). n_obs=72 monthly periods.
     threshold = norm.ppf(1-1/14) / sqrt(72) per-month SR.

Registry key: residreversal_S8
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

MGMT_PIT   = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE      = MGMT_PIT / "cache"                         # shared, read-only
OWN_CACHE  = Path(__file__).resolve().parent / "cache"  # S8-private
RESULTS    = Path(__file__).resolve().parent / "results"
REGISTRY   = Path(__file__).resolve().parents[1] / "trial_registry.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S6/S9) ─────────────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0

# ── Factor-regression window ─────────────────────────────────────────────────
TRADING_DAYS_PER_MONTH = 21
REG_WINDOW_MONTHS = 36
REG_MIN_MONTHS    = 24
REG_MIN_DAYS      = REG_MIN_MONTHS * TRADING_DAYS_PER_MONTH   # 504

# ── Reversal signal windows ───────────────────────────────────────────────────
RESID_1M_MIN_DAYS      = 10    # minimum trading days in the 1M reversal window
RESID_12M_MONTHLY_MIN  = 6     # minimum monthly sums needed for vol estimate

# ── Pre-registered filter parameters ─────────────────────────────────────────
DECILE_FRAC       = 0.10   # top/bottom 10%
VOL_FILTER_SIGMA  = 1.5    # |resid_1m| >= 1.5 * trailing 12M monthly resid std
NAME_COUNT_MIN    = 20     # minimum names per leg (full book); skip if below

# ── Beta window for LEG-SIZING ────────────────────────────────────────────────
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Cost model (identical to S6/S9) ──────────────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── Backtest dates — monthly, 72 periods ─────────────────────────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="MS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 72 months

# ── DSR: new slot, N_TRIALS=14 ────────────────────────────────────────────────
TRIAL_NUMBER = 14
N_TRIALS     = 14


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


def _adj_returns(ticker: str) -> pd.Series:
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
# SECTION 5: Market factor (identical to S6 — EW across all survivors)
# =============================================================================

_MKT_FACTOR: pd.Series = pd.Series(dtype=float)


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
# SECTION 6: SEC facts (PIT shares outstanding — reuse S9/S6 trimmed cache)
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
        print(f"[Preload] SEC facts: {loaded:,} from trimmed cache in {time.time()-t0:.1f}s")
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
# SECTION 7: Universe builder (identical to S6/S9)
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
    print(f"  [Universe] {rebalance_date.date()} -> {len(df):,} names", end="")
    return df


# =============================================================================
# SECTION 8: Residual reversal signal
# =============================================================================

def compute_residual_reversal(
    ticker: str,
    rebalance_date: pd.Timestamp,
    mkt_factor: pd.Series,
) -> tuple:
    """
    Returns (signal, resid_1m, resid_12m_std, beta_reg) or (None, None, None, None).

    signal    = -resid_1m  (reversal: negative recent residual -> positive rank -> long)
    resid_1m  = sum of daily residuals over [t-1M, t]
    resid_12m_std = std of 12 non-overlapping monthly residual sums over [t-13M, t-1M]
                    (12 months prior to the current reversal window, no look-ahead)

    Vol filter (applied in caller): |resid_1m| >= VOL_FILTER_SIGMA * resid_12m_std
    """
    reg_start = rebalance_date - pd.DateOffset(months=REG_WINDOW_MONTHS)
    ret = _adj_returns(ticker)
    if ret.empty:
        return None, None, None, None

    window_ret = ret[(ret.index >= reg_start) & (ret.index <= rebalance_date)]
    if len(window_ret) < REG_MIN_DAYS:
        return None, None, None, None

    x = mkt_factor.reindex(window_ret.index).dropna()
    y = window_ret.reindex(x.index)
    if len(y) < REG_MIN_DAYS:
        return None, None, None, None

    X = np.column_stack([np.ones(len(x)), x.values])
    try:
        coeffs, _, rank, _ = np.linalg.lstsq(X, y.values, rcond=None)
    except Exception:
        return None, None, None, None
    if rank < 2:
        return None, None, None, None

    alpha, beta = float(coeffs[0]), float(coeffs[1])
    y_hat = X @ coeffs
    resid = pd.Series(y.values - y_hat, index=y.index)

    # ── 1-month reversal window: [t-1M, t] ───────────────────────────────────
    rev_start = rebalance_date - pd.DateOffset(months=1)
    resid_1m_window = resid[(resid.index >= rev_start) & (resid.index <= rebalance_date)]
    if len(resid_1m_window) < RESID_1M_MIN_DAYS:
        return None, None, None, None
    resid_1m = float(resid_1m_window.sum())

    # ── Trailing 12M monthly std: 12 monthly sums over [t-13M, t-1M] ─────────
    # Month k covers [t-(k+1)M, t-kM) for k=1..12
    monthly_sums: list[float] = []
    for k in range(1, 13):
        m_end   = rebalance_date - pd.DateOffset(months=k)
        m_start = rebalance_date - pd.DateOffset(months=k + 1)
        m_win   = resid[(resid.index >= m_start) & (resid.index < m_end)]
        if len(m_win) >= RESID_1M_MIN_DAYS:
            monthly_sums.append(float(m_win.sum()))

    if len(monthly_sums) < RESID_12M_MONTHLY_MIN:
        return None, None, None, None

    resid_12m_std = float(np.std(monthly_sums, ddof=1)) if len(monthly_sums) > 1 else np.nan
    if np.isnan(resid_12m_std) or resid_12m_std <= 0:
        return None, None, None, None

    signal = -resid_1m   # reversal: loser (negative resid) -> positive signal -> long
    return signal, resid_1m, resid_12m_std, beta


# =============================================================================
# SECTION 9: Return calculation (identical to S6)
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
            exit_cands = prices[prices["date"] <= delist_date]
            exit_price = float(
                (exit_cands if not exit_cands.empty else prices).iloc[-1]["close"])
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
# SECTION 10: Beta computation for leg sizing (60-day, same as S6/gp/ivol)
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
            s   = raw_w.set_index("date")["close"]
            s   = s[~s.index.duplicated(keep="last")].sort_index()
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
# SECTION 11: Portfolio construction with turnover discipline
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio_disciplined(
    signals_df: pd.DataFrame,
    betas: dict,
    prev_longs: set,
    prev_shorts: set,
) -> tuple:
    """
    Applies the double filter (decile + vol threshold) to determine the active
    set this period. Carry = previous book members that don't pass the filter.
    Full book = active + carry.

    Returns:
        full_longs (list), full_shorts (list),
        active_longs (set), active_shorts (set),
        lw (dict), sw (dict),
        long_scale, short_scale,
        mean_beta_long, mean_beta_short,
        guard_fired (bool), skip_period (bool)
    """
    df = signals_df.dropna(subset=["signal", "resid_1m", "resid_12m_std", "raw_return"]).copy()
    if df.empty:
        return [], [], set(), set(), {}, {}, 1.0, 1.0, np.nan, np.nan, False, True

    n = len(df)
    decile_k = max(1, int(np.ceil(n * DECILE_FRAC)))

    # ── Active set: top/bottom decile AND vol filter ──────────────────────────
    ranked = df.sort_values("signal", ascending=False)
    top_decile    = set(ranked.iloc[:decile_k]["ticker"])
    bottom_decile = set(ranked.iloc[-decile_k:]["ticker"])

    vol_filter_map = {
        row["ticker"]: (abs(row["resid_1m"]) >= VOL_FILTER_SIGMA * row["resid_12m_std"])
        for _, row in df.iterrows()
    }

    active_longs  = {t for t in top_decile    if vol_filter_map.get(t, False)}
    active_shorts = {t for t in bottom_decile if vol_filter_map.get(t, False)}

    # ── Carry: existing book members not in active set ────────────────────────
    universe_tickers = set(df["ticker"])
    # Only carry names that are still in the current universe
    carry_longs  = (prev_longs  - active_longs  - active_shorts) & universe_tickers
    carry_shorts = (prev_shorts - active_shorts - active_longs)  & universe_tickers

    full_longs  = list(active_longs  | carry_longs)
    full_shorts = list(active_shorts | carry_shorts)

    # ── Name-count guard ──────────────────────────────────────────────────────
    if len(full_longs) < NAME_COUNT_MIN or len(full_shorts) < NAME_COUNT_MIN:
        return [], [], active_longs, active_shorts, {}, {}, 1.0, 1.0, np.nan, np.nan, False, True

    # ── Equal weights across full book ────────────────────────────────────────
    lw = {t: 1.0 / len(full_longs)  for t in full_longs}
    sw = {t: 1.0 / len(full_shorts) for t in full_shorts}

    # ── Beta-neutral scales ───────────────────────────────────────────────────
    mean_beta_long  = float(np.nanmean([betas.get(t, np.nan) for t in full_longs]))
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in full_shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return (full_longs, full_shorts, active_longs, active_shorts,
            lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short,
            False, False)


def build_portfolio_naive(
    signals_df: pd.DataFrame,
    betas: dict,
) -> tuple:
    """
    Naive full-book: all decile names, no vol filter, full monthly churn.
    Informational only — not subject to DSR.
    """
    df = signals_df.dropna(subset=["signal", "raw_return"]).copy()
    if df.empty:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan

    n = len(df)
    decile_k = max(1, int(np.ceil(n * DECILE_FRAC)))
    ranked   = df.sort_values("signal", ascending=False)
    longs    = ranked.iloc[:decile_k]["ticker"].tolist()
    shorts   = ranked.iloc[-decile_k:]["ticker"].tolist()

    if not longs or not shorts:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan

    lw = {t: 1.0 / len(longs)  for t in longs}
    sw = {t: 1.0 / len(shorts) for t in shorts}

    mean_beta_long  = float(np.nanmean([betas.get(t, np.nan) for t in longs]))
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short


def _leg_return_detail(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    active_set: set,
    is_short: bool,
) -> tuple:
    """
    Gross/net return for a leg.
    Transaction costs only on active_set members (carry = zero cost).
    """
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
        gross += w * r
        if t in active_set:
            ba_cost += 2.0 * _spread(mc) * w   # full two-way spread on traded names
        if is_short:
            borrow_cost += (_borrow(mc) / 12.0) * w   # monthly fraction

    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    else:
        return ( gross - total_cost),   gross,  ba_cost, borrow_cost


def _leg_return_detail_naive(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool,
) -> tuple:
    """Naive leg: full turnover cost on all names."""
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
        ba_cost += 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / 12.0) * w

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

    survivors        = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map      = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    all_survivor_tks = survivors["ticker"].tolist()

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    print("\n[MktFactor] Building single-factor market model factor...")
    mkt_factor = build_market_factor(all_survivor_tks)

    period_records: list  = []
    universe_audit: list  = []
    sign_violations: int  = 0
    guard_fire_count: int = 0
    skip_count: int       = 0

    # Turnover-discipline book state
    prev_longs:  set = set()
    prev_shorts: set = set()

    # Liquidity-tercile tracking across all periods
    liq_tercile_contrib: dict = {0: 0.0, 1: 0.0, 2: 0.0}  # 0=most illiquid

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
            print("  [SKIP] empty universe")
            continue

        # ── Reversal signals ─────────────────────────────────────────────────
        signals, resid_1ms, resid_12m_stds, reg_betas = [], [], [], []
        for _, row in univ.iterrows():
            sig, r1m, r12std, beta_reg = compute_residual_reversal(row["ticker"], t, mkt_factor)
            signals.append(sig)
            resid_1ms.append(r1m)
            resid_12m_stds.append(r12std)
            reg_betas.append(beta_reg)

        univ = univ.copy()
        univ["signal"]        = signals
        univ["resid_1m"]      = resid_1ms
        univ["resid_12m_std"] = resid_12m_stds
        univ["reg_beta"]      = reg_betas

        n_with_signal = int(univ["signal"].notna().sum())
        n_total       = len(univ)

        # Filter-passing count (diagnostic)
        eligible = univ.dropna(subset=["signal", "resid_1m", "resid_12m_std"])
        n = len(eligible)
        decile_k = max(1, int(np.ceil(n * DECILE_FRAC)))
        ranked_elig = eligible.sort_values("signal", ascending=False)
        top_d    = set(ranked_elig.iloc[:decile_k]["ticker"])
        bot_d    = set(ranked_elig.iloc[-decile_k:]["ticker"])
        n_pass_filter = sum(
            1 for _, row in eligible.iterrows()
            if (row["ticker"] in top_d or row["ticker"] in bot_d)
            and abs(row["resid_1m"]) >= VOL_FILTER_SIGMA * row["resid_12m_std"]
        )
        print(f"  Signal: {n_with_signal}/{n_total} have signal | "
              f"{n_pass_filter} pass double-filter")

        try:
            returns_df = compute_returns(univ, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        universe_audit.append({
            "rebalance_date": t, "hold_end": hold_end,
            "n_universe": n_total, "n_with_signal": n_with_signal,
            "n_pass_filter": n_pass_filter, "n_delisted": n_delisted,
            "n_valid_returns": int(returns_df["raw_return"].notna().sum()),
        })

        betas = compute_betas(univ, t)

        # ── Disciplined portfolio ─────────────────────────────────────────────
        (full_longs, full_shorts, active_longs, active_shorts,
         lw, sw, long_scale, short_scale,
         mean_beta_long, mean_beta_short,
         guard_fired, skip_period) = build_portfolio_disciplined(
             returns_df, betas, prev_longs, prev_shorts)

        if skip_period:
            if guard_fired or (len(full_longs) < NAME_COUNT_MIN):
                guard_fire_count += 1
                print(f"  [GUARD] <{NAME_COUNT_MIN} names in leg — skipping rebalance, holding existing")
            else:
                skip_count += 1
                print(f"  [SKIP] no eligible names after filter")
            # Hold existing — no period record (contributes to guard count not returns)
            continue

        # ── Turnover calculation (only active names incur cost) ───────────────
        entered_longs  = active_longs  - prev_longs
        exited_longs   = prev_longs    - set(full_longs)
        entered_shorts = active_shorts - prev_shorts
        exited_shorts  = prev_shorts   - set(full_shorts)
        total_book     = max(len(full_longs) + len(full_shorts), 1)
        actual_turnover = (len(entered_longs) + len(exited_longs) +
                           len(entered_shorts) + len(exited_shorts)) / total_book

        prev_longs  = set(full_longs)
        prev_shorts = set(full_shorts)

        # ── Sign-convention audit: long leg should have negative mean resid_1m ─
        rm_map = returns_df.set_index("ticker")["resid_1m"].to_dict()
        mean_r1m_long  = float(np.nanmean([rm_map.get(t2, np.nan) for t2 in full_longs]))
        mean_r1m_short = float(np.nanmean([rm_map.get(t2, np.nan) for t2 in full_shorts]))
        # Reversal: long = prior losers (negative resid_1m), short = prior winners (positive)
        sign_ok = mean_r1m_long < mean_r1m_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg mean resid_1m ({mean_r1m_long:.4f}) "
                  f">= short-leg ({mean_r1m_short:.4f}) — reversal sign broken ***")

        # ── Disciplined L/S returns ───────────────────────────────────────────
        long_net, long_gross, long_ba, long_bw = _leg_return_detail(
            returns_df, full_longs,  lw, active_longs,  is_short=False)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, full_shorts, sw, active_shorts, is_short=True)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw
        ls_cost  = ls_ba + ls_bw

        # ── Naive full-book comparison ────────────────────────────────────────
        (n_longs_naive, n_shorts_naive, nlw, nsw,
         nls, nss, _, _) = build_portfolio_naive(returns_df, betas)
        if n_longs_naive:
            nl_net, nl_gross, _, _ = _leg_return_detail_naive(
                returns_df, n_longs_naive,  nlw, is_short=False)
            ns_net, ns_gross, _, _ = _leg_return_detail_naive(
                returns_df, n_shorts_naive, nsw, is_short=True)
            ls_naive_net   = nls * nl_net   + nss * ns_net
            ls_naive_gross = nls * nl_gross + nss * ns_gross
        else:
            ls_naive_net = ls_naive_gross = np.nan

        # ── IC (full eligible set) ────────────────────────────────────────────
        valid_sig = returns_df.dropna(subset=["signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        # ── Liquidity-tercile attribution ─────────────────────────────────────
        dvol_map = returns_df.set_index("ticker")["avg_dvol"].to_dict()
        ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
        all_book = list(full_longs) + list(full_shorts)
        dvols    = sorted([dvol_map.get(tk, 0) for tk in all_book])
        if len(dvols) >= 3:
            t1 = dvols[len(dvols) // 3]
            t2 = dvols[2 * len(dvols) // 3]
            for tk in full_longs:
                r = ret_map.get(tk, np.nan)
                if np.isnan(r):
                    continue
                dv = dvol_map.get(tk, 0)
                tier = 0 if dv <= t1 else (1 if dv <= t2 else 2)
                liq_tercile_contrib[tier] += long_scale * lw.get(tk, 0) * r
            for tk in full_shorts:
                r = ret_map.get(tk, np.nan)
                if np.isnan(r):
                    continue
                dv = dvol_map.get(tk, 0)
                tier = 0 if dv <= t1 else (1 if dv <= t2 else 2)
                liq_tercile_contrib[tier] -= short_scale * sw.get(tk, 0) * r

        mkt_ret = float(returns_df.dropna(subset=["raw_return"])["raw_return"].mean())

        print(f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} "
              f"naive={ls_naive_net:+.2%} to={actual_turnover:.0%} "
              f"act_L={len(active_longs)} act_S={len(active_shorts)} "
              f"carry_L={len(full_longs)-len(active_longs)} "
              f"carry_S={len(full_shorts)-len(active_shorts)} "
              f"IC={ic:.3f} beta_L={mean_beta_long:.2f} beta_S={mean_beta_short:.2f}")

        period_records.append({
            "rebalance_date":   t,
            "hold_end":         hold_end,
            "n_universe":       n_total,
            "n_with_signal":    n_with_signal,
            "n_pass_filter":    n_pass_filter,
            "n_delisted":       n_delisted,
            # Disciplined strategy
            "ls_ret":           ls_net,
            "ls_gross":         ls_gross,
            "ls_cost":          ls_cost,
            "ls_ba_cost":       ls_ba,
            "ls_borrow_cost":   ls_bw,
            "long_ret":         long_net,
            "short_ret":        short_net,
            # Naive comparison
            "ls_naive_net":     ls_naive_net,
            "ls_naive_gross":   ls_naive_gross,
            # Sizing
            "long_scale":       long_scale,
            "short_scale":      short_scale,
            "mean_beta_long":   mean_beta_long,
            "mean_beta_short":  mean_beta_short,
            # Turnover & book composition
            "actual_turnover":  actual_turnover,
            "n_active_longs":   len(active_longs),
            "n_active_shorts":  len(active_shorts),
            "n_carry_longs":    len(full_longs) - len(active_longs),
            "n_carry_shorts":   len(full_shorts) - len(active_shorts),
            "n_long":           len(full_longs),
            "n_short":          len(full_shorts),
            # Diagnostics
            "mkt_ret":          mkt_ret,
            "ic":               ic,
            "mean_r1m_long":    mean_r1m_long,
            "mean_r1m_short":   mean_r1m_short,
            "sign_ok":          sign_ok,
        })

    elapsed = time.time() - t_total
    return {
        "period_records":       period_records,
        "universe_audit":       universe_audit,
        "sign_violations":      sign_violations,
        "guard_fire_count":     guard_fire_count,
        "skip_count":           skip_count,
        "liq_tercile_contrib":  liq_tercile_contrib,
        "elapsed_s":            elapsed,
    }


# =============================================================================
# SECTION 13: Metrics (monthly annualization)
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
    ann_ret = float((1 + total) ** (12.0 / n) - 1)
    mu      = float(ls.mean())
    sigma   = float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(12)) if sigma > 0 else np.nan
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
    beta  = float(coeffs[1])
    y_hat = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n     = len(y)
    se    = (np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 14: Output
# =============================================================================

def print_results(results: dict) -> None:
    records  = results["period_records"]
    elapsed  = results["elapsed_s"]
    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df = pd.DataFrame(records)

    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    naive_series = df["ls_naive_net"].dropna()
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m          = compute_metrics(ls_series)
    m_gross    = compute_metrics(gross_series)
    m_naive    = compute_metrics(naive_series) if len(naive_series) > 1 else {}
    beta_stats = compute_market_beta(ls_series, mkt_series)

    # Per-month SR for DSR
    obs_sr_m   = (m["sharpe"] / np.sqrt(12)) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (bool(obs_sr_m > dsr_thr)
                  if not np.isnan(obs_sr_m) and not np.isnan(dsr_thr) else False)

    mean_ic  = float(ic_series.mean())  if len(ic_series) > 0 else np.nan
    ic_tstat = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)

    annual_ba_bps  = float(df["ls_ba_cost"].mean())    * 12 * 10_000
    annual_bw_bps  = float(df["ls_borrow_cost"].mean()) * 12 * 10_000
    annual_cost    = annual_ba_bps + annual_bw_bps
    mean_turnover  = float(df["actual_turnover"].mean())

    sign_violations  = int(results["sign_violations"])
    guard_fire_count = int(results["guard_fire_count"])
    skip_count       = int(results["skip_count"])
    n_periods_run    = len(df)

    sep = "=" * 78

    # ── Sign-convention audit ─────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT (reversal: long = prior residual losers)")
    print(sep)
    print(f"  Periods where long mean resid_1m >= short mean resid_1m (violation): "
          f"{sign_violations}/{n_periods_run}")
    if sign_violations == 0:
        print("  PASS — long leg has lower mean resid_1m than short every period.")
    else:
        print(f"  *** FAIL — sign convention broken in {sign_violations} periods. ***")

    # ── Concentration check ───────────────────────────────────────────────────
    abs_mass   = float(ls_series.abs().sum())
    period_shr = (ls_series.abs() / abs_mass) if abs_mass > 0 else ls_series * 0
    top_idx    = period_shr.idxmax()
    top_share  = float(period_shr.loc[top_idx])
    top_date   = df.loc[top_idx, "rebalance_date"]

    print(f"\n{sep}")
    print("CONCENTRATION CHECK (single-episode artifact risk)")
    print(sep)
    print(f"  Largest single-month share of total |return| mass: "
          f"{top_date.date()} = {top_share:.1%}")
    print("  *** WARNING: single month dominates (>30%). ***" if top_share > 0.30
          else "  No single month dominates (<30% threshold).")

    # ── Liquidity-shock isolation ─────────────────────────────────────────────
    liq = results.get("liq_tercile_contrib", {})
    print(f"\n{sep}")
    print("LIQUIDITY-SHOCK ISOLATION (is return concentrated in most illiquid tercile?)")
    print(sep)
    total_liq_abs = sum(abs(v) for v in liq.values())
    if total_liq_abs > 0:
        liq_labels = {0: "most illiquid (low ADTV)", 1: "mid-liquidity", 2: "most liquid (high ADTV)"}
        for tier in sorted(liq):
            shr = abs(liq[tier]) / total_liq_abs
            print(f"  Tier {tier} ({liq_labels[tier]}): contrib = {liq[tier]:+.4f} "
                  f"({shr:.1%} of |total|)")
        most_illiq_shr = abs(liq.get(0, 0)) / total_liq_abs
        print("  *** NOTE: >50% of return mass from most-illiquid tercile — "
              "consistent with illiquidity premium (foreshadows S10). ***"
              if most_illiq_shr > 0.50 else
              "  Return not concentrated in most-illiquid tercile.")

    # ── Guard / skip counts ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("FILTER FIRE RATES")
    print(sep)
    total_periods = n_periods_run + guard_fire_count + skip_count
    print(f"  Periods completed:                 {n_periods_run}/{total_periods}")
    print(f"  Guard fired (book <{NAME_COUNT_MIN}/leg, held):  "
          f"{guard_fire_count}/{total_periods} "
          f"({100*guard_fire_count/max(total_periods,1):.0f}%)")
    print(f"  Skipped (no eligible names):       {skip_count}/{total_periods}")
    print(f"  Mean active names per leg:         "
          f"L={df['n_active_longs'].mean():.1f}  S={df['n_active_shorts'].mean():.1f}")
    print(f"  Mean carry names per leg:          "
          f"L={df['n_carry_longs'].mean():.1f}  S={df['n_carry_shorts'].mean():.1f}")

    # ── Main results ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("S8: RESIDUAL SHORT-TERM REVERSAL — IN-SAMPLE RESULTS")
    print("Signal: -(1M residual return) | Filter: decile + 1.5sigma | Monthly")
    print(f"N_TRIALS={N_TRIALS} | n_obs={m['n']} monthly periods")
    print(sep)

    print(f"\n  *** INFORMATION COEFFICIENT ***")
    print(f"  Mean IC (Spearman, signal vs ret)  : {mean_ic:>12.4f}")
    print(f"  IC t-stat                          : {ic_tstat:>12.3f}")

    print(f"\n  *** MARKET BETA ***")
    print(f"  Market beta (beta-neutral sizing)  : {beta_stats['beta']:>12.3f} "
          f"(t={beta_stats['beta_t']:.2f})")

    n_str = str(m["n"]) if not np.isnan(m.get("n", np.nan)) else "n/a"
    print(f"\n  {'Metric':<46} {'Disciplined':>14} {'Naive (info)':>14}")
    print(f"  {'-'*46} {'-'*14} {'-'*14}")
    print(f"  {'Total cumulative return (net)':<46} "
          f"{m['total_ret']:>13.2%} "
          f"{m_naive.get('total_ret', np.nan):>13.2%}")
    print(f"  {'Annualized return (net)':<46} "
          f"{m['ann_ret']:>13.2%} "
          f"{m_naive.get('ann_ret', np.nan):>13.2%}")
    print(f"  {'Annualized return (GROSS, disciplined)':<46} "
          f"{m_gross['ann_ret']:>13.2%} {'':>14}")
    print(f"  {'Cost drag (annualized bps)':<46} "
          f"{annual_cost:>13.0f} {'':>14}")
    print(f"    {'  bid-ask (bps/yr)':<44} {annual_ba_bps:>13.0f}")
    print(f"    {'  borrow  (bps/yr)':<44} {annual_bw_bps:>13.0f}")
    print(f"  {'Mean actual turnover per month':<46} "
          f"{mean_turnover:>13.0%} {'~100%':>14}")
    print(f"  {'Sharpe ratio (annualized, net)':<46} "
          f"{m['sharpe']:>13.3f} "
          f"{m_naive.get('sharpe', np.nan):>13.3f}")
    print(f"  {'CALMAR ratio':<46} "
          f"{m['calmar']:>13.3f} "
          f"{m_naive.get('calmar', np.nan):>13.3f}")
    print(f"  {'Max drawdown':<46} "
          f"{m['max_dd']:>13.2%} "
          f"{m_naive.get('max_dd', np.nan):>13.2%}")
    print(f"  {'t-stat (mean monthly net return)':<46} "
          f"{m['t_stat']:>13.3f} "
          f"{m_naive.get('t_stat', np.nan):>13.3f}")
    print(f"  {'Win rate':<46} "
          f"{m['win_rate']:>13.1%} "
          f"{m_naive.get('win_rate', np.nan):>13.1%}")
    print(f"  {'Periods (N)':<46} {n_str:>13}")

    print(f"\n  --- DSR Check (N_TRIALS={N_TRIALS}, n_obs={m['n']} monthly periods) ---")
    print(f"  {'Observed per-month Sharpe (ann/sqrt12)':<46} {obs_sr_m:>13.4f}")
    print(f"  {'DSR threshold = norm.ppf(1-1/14)/sqrt(72)':<46} {dsr_thr:>13.4f}")
    print(f"  {'Clears DSR':<46} {'YES' if clears_dsr else 'NO':>13}")

    # ── Monthly returns log ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("MONTHLY RETURNS LOG")
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Naive':>7} {'Cost':>6} "
          f"{'TOver':>5} {'ActL':>4} {'ActS':>4} {'CryL':>4} "
          f"{'IC':>6} {'Mkt':>7}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*7} {'-'*6} "
          f"{'-'*5} {'-'*4} {'-'*4} {'-'*4} "
          f"{'-'*6} {'-'*7}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} "
              f"{r['ls_naive_net']:>6.2%} "
              f"{r['ls_cost']:>5.2%} {r['actual_turnover']:>4.0%} "
              f"{int(r['n_active_longs']):>4} {int(r['n_active_shorts']):>4} "
              f"{int(r['n_carry_longs']):>4} "
              f"{r['ic']:>6.3f} {r['mkt_ret']:>6.2%}")

    # Drawdown series
    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Max drawdown series (monthly):")
    for idx, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " <- MAX DD" if abs(val - m["max_dd"]) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{marker}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


# =============================================================================
# SECTION 15: Save results and registry
# =============================================================================

def save_results(results: dict) -> None:
    records = results["period_records"]
    audit   = results["universe_audit"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "s8_period_returns.csv",  index=False)
    pd.DataFrame(audit).to_csv(  RESULTS / "s8_universe_audit.csv",  index=False)

    liq = results.get("liq_tercile_contrib", {})
    liq_df = pd.DataFrame([
        {"adtv_tercile": k, "description": ["most_illiquid","mid","most_liquid"][k], "ls_contribution": v}
        for k, v in sorted(liq.items())
    ])
    liq_df.to_csv(RESULTS / "s8_liquidity_isolation.csv", index=False)
    print(f"\n[Saved] Results -> {RESULTS}/")


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

    obs_sr_m   = m["sharpe"] / np.sqrt(12) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = bool(obs_sr_m > dsr_thr) if not np.isnan(obs_sr_m) else False

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)

    annual_cost    = (float(df["ls_ba_cost"].mean()) + float(df["ls_borrow_cost"].mean())) * 12 * 10_000
    sign_violations  = int(results["sign_violations"])
    guard_fire_count = int(results["guard_fire_count"])
    n_periods_run    = len(df)

    verdict_bits = ["CLEARS DSR" if clears_dsr else "FAILS DSR",
                    f"N_TRIALS={N_TRIALS}, n_obs={m['n']} monthly periods"]
    if sign_violations > 0:
        verdict_bits.append(f"SIGN AUDIT FAILED ({sign_violations} periods)")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "residreversal_S8",
        "hypothesis":           (
            "Residual short-term reversal: long bottom decile / short top decile "
            "of 1-month residual return (rolling 36M market-model). Pre-registered "
            "double filter: decile + |resid_1m| >= 1.5x trailing 12M monthly-residual "
            "std. Turnover discipline: non-threshold names hold existing positions. "
            "Trial #14 in S1-S10 program."
        ),
        "data_source":          "Tiingo (adjClose signal+regression, raw close returns) + SEC XBRL PIT",
        "status":               "completed_in_sample",
        "trial_number":         TRIAL_NUMBER,
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "monthly",
        "signal_note":          (
            "Signal = -(1M residual return) from 36M (min 24M) single-factor OLS. "
            f"Double filter: decile + |resid_1m| >= {VOL_FILTER_SIGMA}x trailing 12M std."
        ),
        "portfolio_note":       (
            f"Beta-neutral sizing. Name-count guard <{NAME_COUNT_MIN}/leg -> skip. "
            f"Guard fired {guard_fire_count}/{n_periods_run + guard_fire_count} periods. "
            f"Mean actual turnover: {float(df['actual_turnover'].mean()):.0%}/month."
        ),
        "ann_return_net":       m["ann_ret"],
        "ann_return_gross":     m_gross["ann_ret"],
        "annual_cost_drag_bps": annual_cost,
        "mean_turnover":        float(df["actual_turnover"].mean()),
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
        "obs_sr_monthly":       obs_sr_m,
        "dsr_threshold":        dsr_thr,
        "clears_dsr":           clears_dsr,
        "notes":                verdict,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "residreversal_S8"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] residreversal_S8 logged (trial #{TRIAL_NUMBER}, "
          f"N_TRIALS={N_TRIALS}) -> {REGISTRY}")
    print(f"[Verdict] {verdict}")


# =============================================================================
# SECTION 16: Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S8: RESIDUAL SHORT-TERM REVERSAL  (Trial #14, N_TRIALS=14)")
    print("Signal: -(1M residual return) from 36M market-model regression")
    print("Filter: top/bottom decile AND |resid_1m| >= 1.5x trailing 12M monthly std")
    print("Turnover discipline: non-threshold names hold existing positions")
    print("Monthly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 72 periods")
    print(f"DSR: N_TRIALS={N_TRIALS}")
    print("=" * 78)

    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df      = load_prefilter()
    survivors         = prefilter_df[prefilter_df["passed"]]
    survivor_tickers  = survivors["ticker"].tolist()
    survivor_ciks     = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle (signal + regression layer)...")
    preload_all_adj_prices(survivor_tickers)

    print(f"\n[Step 4] Loading SEC shares facts ({len(survivor_ciks):,} CIKs)...")
    preload_sec_facts(survivor_ciks)

    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} monthly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    print_results(results)
    save_results(results)
    log_registry(results)


if __name__ == "__main__":
    main()
