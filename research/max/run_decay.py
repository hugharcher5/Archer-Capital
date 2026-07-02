"""
S-MAX Holding-Period Decay Test  (IVOL/MAX family — trial #7)
=============================================================
Sweeps the identical 21-day MAX signal across three non-overlapping holding
periods to measure signal decay and cost drag reduction as hold lengthens.

  1M  → 72 non-overlapping 1-month periods  (Jan 2015 – Dec 2020)
  2M  → 36 non-overlapping 2-month periods  (Jan 2015, Mar 2015, …)
  3M  → 24 non-overlapping 3-month periods  (Jan 2015, Apr 2015, …)

Locked: MAX formation window stays 21 trading days in ALL three variants.
Only the holding period (and therefore the rebalance frequency) changes.

Motivation: The 1M S-MAX run produced IC = +0.034 (t = 2.46) and gross
return +4.76%/yr but net return −16.06%/yr due to ~2,200bps/yr bid-ask
cost from 66% monthly turnover. This tests whether reducing turnover by
lengthening the hold allows net return to turn positive while the signal
still has predictive power.

Key questions:
  1. How fast does gross alpha / IC decay as hold lengthens?
  2. How fast does cost drag fall as turnover drops?
  3. Does the net-return curve go positive at 2M or 3M?

DSR: same IVOL/MAX family, trial #7, N_TRIALS=7 unchanged. Three hold-
period variants are NOT three independent trials. Log one registry entry
per variant, all flagged as hold-period decay study.
"""

import json
import os
import pickle
import sys
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

MGMT_PIT = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE    = MGMT_PIT / "cache"
RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (identical to S-MAX) ─────────────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── Signal parameters (LOCKED — same formation, only hold changes) ────────────
MAX_WINDOW_DAYS   = 21    # 21 trading-day formation window — FIXED across all sweeps
MAX_MIN_DAYS      = 5
MAX_LOOKBACK_CAL  = 35

BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Cost model (same as S-MAX) ────────────────────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── In-sample window ──────────────────────────────────────────────────────────
IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")

# ── Hold periods to sweep ──────────────────────────────────────────────────────
HOLD_MONTHS = [1, 2, 3]

# ── DSR: IVOL/MAX family, trial #7 ────────────────────────────────────────────
N_TRIALS = 7


# =============================================================================
# SECTION 2: Rate-limited HTTP
# =============================================================================

_TIINGO_LAST: float = 0.0

def _tiingo_get(url: str, params: dict = None) -> requests.Response:
    global _TIINGO_LAST
    wait = 1.5 - (time.time() - _TIINGO_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, params=params, timeout=(10, 20))
    _TIINGO_LAST = time.time()
    return r


# =============================================================================
# SECTION 3: Tiingo metadata
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
        return df
    if not TIINGO_KEY:
        raise RuntimeError("No TIINGO_API_KEY and no cached tickers.csv.")
    url = "https://api.tiingo.com/api/ticker_metadata/tickers"
    r   = _tiingo_get(url, {"token": TIINGO_KEY})
    r.raise_for_status()
    df  = pd.read_csv(pd.io.common.BytesIO(r.content), parse_dates=["startDate", "endDate"])
    df  = df[df["exchange"].isin(US_EXCHANGES)].copy()
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
# SECTION 5: Raw price cache
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


def _get_price_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return _PRICE_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


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


# =============================================================================
# SECTION 5b: AdjClose cache (signal layer)
# =============================================================================

_ADJ_PKL      = CACHE / "adjclose_all.pkl"
_ADJ_CACHE:   dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str] = set()
_ADJ_EMPTY    = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if not _ADJ_PKL.exists():
        print("[AdjPreload] No pickle. Run research/ivol/run.py first.")
        sys.exit(1)
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
# SECTION 6: SEC facts (PIT shares outstanding)
# =============================================================================

_FACTS_CACHE: dict[int, Optional[dict]] = {}
_FACTS_TRIMMED   = CACHE / "sec_facts_trimmed.json"
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
        print(f"[Preload] SEC facts: {loaded:,} in {time.time()-t0:.1f}s")
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
    print(f"[Preload] SEC facts: {loaded:,} in {time.time()-t0:.1f}s")


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
# SECTION 7: Universe builder (identical to S-MAX)
# =============================================================================

def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
    verbose: bool = True,
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
        rows.append({"ticker": ticker, "cik": cik, "mcap": mcap,
                     "price": price, "shares": shares, "avg_dvol": dvol})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "cik", "mcap", "price", "shares", "avg_dvol"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    if verbose:
        print(f"  [U] {rebalance_date.date()}→{len(df):,}", end="")
    return df


# =============================================================================
# SECTION 8: MAX signal + 60-day EW beta (identical to S-MAX)
# =============================================================================

def compute_max_signals(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple:
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)
    max_start  = rebalance_date - pd.Timedelta(days=MAX_LOOKBACK_CAL)

    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj    = _adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            continue
        s = adj.set_index("date")["adjClose"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        ret = s.pct_change().dropna()
        if len(ret) >= BETA_MIN_DAYS:
            adj_series[ticker] = ret

    if adj_series:
        ret_mat = pd.DataFrame(adj_series).sort_index()
        ew_mkt  = ret_mat.clip(-0.5, 0.5).mean(axis=1)
    else:
        ew_mkt = pd.Series(dtype=float)

    n_fallback = 0
    n_total    = 0
    records    = []

    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        n_total += 1

        using_raw = False
        if ticker in adj_series:
            ret_full = adj_series[ticker]
            ret_win  = ret_full[ret_full.index >= max_start]
        else:
            raw = _get_price_slice(ticker, max_start, rebalance_date)
            if raw.empty:
                continue
            s = raw.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret_win  = s.pct_change().dropna()
            using_raw = True

        if len(ret_win) < MAX_MIN_DAYS:
            continue

        rets_21 = ret_win.iloc[-MAX_WINDOW_DAYS:] if len(ret_win) >= MAX_WINDOW_DAYS else ret_win
        max_ret = float(rets_21.max())
        if not np.isfinite(max_ret):
            continue
        if using_raw:
            n_fallback += 1

        beta = np.nan
        if ticker in adj_series and len(ew_mkt) >= BETA_MIN_DAYS:
            y_s = adj_series[ticker]
            x_s = ew_mkt.reindex(y_s.index).dropna()
            y_s = y_s.reindex(x_s.index).dropna()
            x_s = x_s.reindex(y_s.index)
            n   = len(y_s)
            if n >= BETA_MIN_DAYS:
                X = np.column_stack([np.ones(n), x_s.values])
                try:
                    coeffs, _, rank, _ = np.linalg.lstsq(X, y_s.values, rcond=None)
                    if rank >= 2:
                        beta = float(coeffs[1])
                except Exception:
                    pass

        records.append({"ticker": ticker, "max_signal": -max_ret, "beta": beta})

    signals_df = (
        pd.DataFrame(records)
        if records
        else pd.DataFrame(columns=["ticker", "max_signal", "beta"])
    )
    return signals_df, n_fallback, n_total


# =============================================================================
# SECTION 9: Hold-period returns (raw close layer)
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
# SECTION 10: Portfolio construction + parameterized cost model
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame) -> tuple:
    """Identical to S-MAX: long lowest MAX, short highest MAX, beta-neutral."""
    df = signals_df.dropna(subset=["max_signal", "raw_return"]).copy()
    n  = len(df)
    if n < 40:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan

    leg = max(int(np.ceil(n * 0.20)), 20)
    leg = min(leg, n // 2)

    ranked = df.sort_values("max_signal", ascending=False)
    longs  = ranked.iloc[:leg]["ticker"].tolist()
    shorts = ranked.iloc[-leg:]["ticker"].tolist()

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    if "beta" in df.columns:
        beta_map        = df.set_index("ticker")["beta"].to_dict()
        mean_beta_long  = float(np.nanmean([beta_map.get(t, np.nan) for t in longs]))
        mean_beta_short = float(np.nanmean([beta_map.get(t, np.nan) for t in shorts]))
        denom = mean_beta_long + mean_beta_short
        if np.isfinite(denom) and denom > 0.05:
            long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
            short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
        else:
            long_scale = short_scale = 1.0
    else:
        mean_beta_long = mean_beta_short = np.nan
        long_scale = short_scale = 1.0

    return longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short


def _leg_return_detail(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool,
    turnover: float,
    periods_per_year: float,
) -> tuple:
    """
    Parameterized cost model.
    Bid-ask: round-trip × actual turnover (same formula as S-MAX, per period).
    Borrow  : annual_rate / periods_per_year  (1M→/12, 2M→/6, 3M→/4).
    Returns (net_return, gross_return, ba_cost, borrow_cost).
    """
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (MCAP_MIN + MCAP_MAX) / 2

    gross       = 0.0
    ba_cost     = 0.0
    borrow_cost = 0.0

    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross   += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / periods_per_year) * w

    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    else:
        return ( gross - total_cost),   gross,  ba_cost, borrow_cost


# =============================================================================
# SECTION 11: Metrics (parameterized for periods_per_year)
# =============================================================================

def compute_metrics(ls_series: pd.Series, ppy: float) -> dict:
    """
    ppy = periods per year: 12 for 1M, 6 for 2M, 4 for 3M.
    Sharpe and annualized return both scale with ppy.
    """
    ls = ls_series.dropna().reset_index(drop=True)
    n  = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "max_dd": np.nan, "t_stat": np.nan,
                "win_rate": np.nan, "n": n}

    cum     = (1 + ls).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (ppy / n) - 1)
    mu      = float(ls.mean())
    sigma   = float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(ppy)) if sigma > 0 else np.nan
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


# =============================================================================
# SECTION 12: One sweep for a given hold period
# =============================================================================

def run_one_sweep(
    hold_months: int,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
    delisted_map: dict,
) -> dict:
    """
    Run the complete S-MAX backtest for a single hold_months value.
    Formation window stays 21 trading days.
    Rebalance dates spaced hold_months apart (non-overlapping).
    Borrow cost scaled by hold_months: 1M→/12, 2M→/6, 3M→/4.
    """
    t0 = time.time()
    ppy = 12.0 / hold_months     # periods per year: 12, 6, or 4

    # Non-overlapping rebalance dates
    all_dates       = pd.date_range(IS_START, IS_END, freq=f"{hold_months}MS")
    rebalance_dates = list(all_dates[all_dates < IS_END])
    n_periods       = len(rebalance_dates)

    print(f"\n{'='*70}")
    print(f"SWEEP: hold={hold_months}M  |  {n_periods} periods  |  ppy={ppy:.0f}")
    print(f"{'='*70}")

    period_records: list = []
    adj_coverage:   list = []

    prev_long_set:  set = set()
    prev_short_set: set = set()

    for i, t in enumerate(rebalance_dates):
        hold_end = t + pd.DateOffset(months=hold_months)
        if hold_end > IS_END:
            hold_end = IS_END

        print(f"\n[{i+1:02d}/{n_periods}] {t.date()}→{hold_end.date()}", end="  ")

        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] {e}")
            continue
        if univ.empty:
            print("  [SKIP]")
            continue

        signals_df, n_fallback, n_computed = compute_max_signals(univ, t)
        n_adj_ok   = n_computed - n_fallback
        pct_fb     = 100 * n_fallback / max(n_computed, 1)

        adj_coverage.append({
            "hold_months":    hold_months,
            "rebalance_date": t,
            "n_computed":     n_computed,
            "n_fallback":     n_fallback,
            "pct_fallback":   pct_fb,
        })

        univ = univ.merge(signals_df, on="ticker", how="left")
        try:
            returns_df = compute_returns(univ, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())

        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short) = build_portfolio(returns_df)
        if not longs:
            print("  [SKIP n<40]")
            continue

        # ── Turnover ──────────────────────────────────────────────────────────
        if prev_long_set:
            long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1)
            short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0

        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        # ── Leg returns with parameterized borrow ─────────────────────────────
        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_detail(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover,
            periods_per_year=ppy)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover,
            periods_per_year=ppy)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw
        ls_cost  = ls_ba + ls_bw

        # ── EW market return (same hold period) ───────────────────────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if len(valid_all) > 0 else np.nan

        # ── IC: max_signal vs realized return (over full hold period) ─────────
        valid_sig = returns_df.dropna(subset=["max_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["max_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  sig={len(signals_df)}/{len(univ)}"
              f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} c={ls_cost:.2%}"
              f"  to={avg_turnover:.0%} IC={ic:.3f}"
              f"  β_L={mean_beta_long:.2f} β_S={mean_beta_short:.2f}")

        period_records.append({
            "hold_months":      hold_months,
            "rebalance_date":   t,
            "hold_end":         hold_end,
            "n_universe":       len(returns_df),
            "n_delisted":       n_delisted,
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
        })

    return {
        "hold_months":    hold_months,
        "ppy":            ppy,
        "period_records": period_records,
        "adj_coverage":   adj_coverage,
        "elapsed_s":      time.time() - t0,
    }


# =============================================================================
# SECTION 13: Comparison table + per-sweep detail
# =============================================================================

def _sweep_summary(sweep: dict) -> dict:
    """Compute all summary statistics for one sweep."""
    records  = sweep["period_records"]
    coverage = sweep["adj_coverage"]
    ppy      = sweep["ppy"]
    hold_months = sweep["hold_months"]

    if not records:
        return {"hold_months": hold_months, "n": 0}

    df  = pd.DataFrame(records)
    cov = pd.DataFrame(coverage)

    ls_series   = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series  = df["mkt_ret"]
    ic_series   = df["ic"].dropna()

    m          = compute_metrics(ls_series,   ppy)
    m_gross    = compute_metrics(gross_series, ppy)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    mean_turnover        = float(df["avg_turnover"].mean())
    mean_monthly_cost    = float(df["ls_cost"].mean())
    annual_cost_drag_bps = mean_monthly_cost * ppy * 10_000
    annual_ba_bps        = float(df["ls_ba_cost"].mean())   * ppy * 10_000
    annual_bw_bps        = float(df["ls_borrow_cost"].mean()) * ppy * 10_000

    total_fallback  = int(cov["n_fallback"].sum()) if not cov.empty else 0
    total_computed  = int(cov["n_computed"].sum())  if not cov.empty else 1
    pct_fallback    = 100 * total_fallback / max(total_computed, 1)

    # DSR in per-period Sharpe units
    per_period_sr = float(df["ls_ret"].mean() / df["ls_ret"].std(ddof=1)) if df["ls_ret"].std(ddof=1) > 0 else np.nan
    dsr_thr = stats.norm.ppf(1 - 1.0 / N_TRIALS) / np.sqrt(m["n"]) if m["n"] > 1 else np.nan
    clears_dsr = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False

    return {
        "hold_months":       hold_months,
        "n":                 m["n"],
        "mean_turnover":     mean_turnover,
        "ann_gross":         m_gross["ann_ret"],
        "cost_drag_bps":     annual_cost_drag_bps,
        "cost_ba_bps":       annual_ba_bps,
        "cost_bw_bps":       annual_bw_bps,
        "ann_net":           m["ann_ret"],
        "sharpe":            m["sharpe"],
        "calmar":            m["calmar"],
        "max_dd":            m["max_dd"],
        "t_stat":            m["t_stat"],
        "beta":              beta_stats["beta"],
        "beta_t":            beta_stats["beta_t"],
        "mean_ic":           mean_ic,
        "ic_t":              ic_tstat,
        "win_rate":          m["win_rate"],
        "pct_fallback":      pct_fallback,
        "per_period_sr":     per_period_sr,
        "dsr_thr":           dsr_thr,
        "clears_dsr":        clears_dsr,
        "elapsed_s":         sweep["elapsed_s"],
    }


def print_comparison_table(summaries: list[dict]) -> None:
    sep = "=" * 80
    print(f"\n\n{sep}")
    print("S-MAX HOLDING-PERIOD DECAY STUDY  |  IVOL/MAX family, trial #7")
    print("Signal: MAX = 21-day trailing peak adjClose return. Formation window FIXED.")
    print("Only the holding period changes. Non-overlapping portfolios.")
    print("In-sample: 2015-01-01 → 2021-01-01")
    print(sep)

    W = 14
    H = 10

    def row(label: str, vals: list, fmt: str = "", pct: bool = False) -> str:
        cells = []
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                cells.append(f"{'n/a':>{W}}")
            elif pct:
                cells.append(f"{v:>{W}.1%}")
            elif fmt:
                cells.append(f"{v:{W}{fmt}}")
            else:
                cells.append(f"{str(v):>{W}}")
        return f"  {label:<{H}}" + "".join(cells)

    def hdr(labels: list) -> str:
        return f"  {'':>{H}}" + "".join(f"{l:>{W}}" for l in labels)

    holds  = [s["hold_months"] for s in summaries]
    hnames = [f"{h}M" for h in holds]

    def g(key: str) -> list:
        return [s.get(key, np.nan) for s in summaries]

    print(hdr(hnames))
    print(f"  {'-'*H}" + f"{'-'*W}" * len(hnames))

    # ── Periods ────────────────────────────────────────────────────────────────
    print(f"  {'N periods':<{H}}" + "".join(f"{s['n']:>{W}d}" for s in summaries))

    # ── Turnover ──────────────────────────────────────────────────────────────
    print(row("Turnover", g("mean_turnover"), pct=True))

    print(f"  {'-'*H}" + f"{'-'*W}" * len(hnames))

    # ── Return decomposition ───────────────────────────────────────────────────
    print(row("GrossAnn", g("ann_gross"), pct=True))
    print(row("CostDrag", [s.get("cost_drag_bps", np.nan) for s in summaries], ".0f"))
    print(row("  BA bps",  [s.get("cost_ba_bps",   np.nan) for s in summaries], ".0f"))
    print(row("  BW bps",  [s.get("cost_bw_bps",   np.nan) for s in summaries], ".0f"))
    print(row("NetAnn",   g("ann_net"),   pct=True))

    print(f"  {'-'*H}" + f"{'-'*W}" * len(hnames))

    # ── Risk/return metrics ────────────────────────────────────────────────────
    print(row("Sharpe",   g("sharpe"),   ".3f"))
    print(row("CALMAR",   g("calmar"),   ".3f"))
    print(row("MaxDD",    g("max_dd"),   pct=True))
    print(row("t-stat",   g("t_stat"),   ".3f"))

    print(f"  {'-'*H}" + f"{'-'*W}" * len(hnames))

    # ── Beta ───────────────────────────────────────────────────────────────────
    print(row("Beta",     g("beta"),     ".3f"))
    beta_t_fmt = []
    for s in summaries:
        bt = s.get("beta_t", np.nan)
        if np.isnan(bt):
            beta_t_fmt.append(f"{'n/a':>{W}}")
        else:
            beta_t_fmt.append(f"{'(t='+f'{bt:.2f}'+')':>{W}}")
    print(f"  {'  beta-t':<{H}}" + "".join(beta_t_fmt))

    print(f"  {'-'*H}" + f"{'-'*W}" * len(hnames))

    # ── IC (signal quality, cost-independent) ─────────────────────────────────
    print(row("Mean IC",  g("mean_ic"),  ".4f"))
    print(row("IC t-stat", g("ic_t"),   ".3f"))
    print(row("WinRate",  g("win_rate"), pct=True))

    print(f"  {'-'*H}" + f"{'-'*W}" * len(hnames))

    # ── DSR ────────────────────────────────────────────────────────────────────
    print(row("SR/period", g("per_period_sr"), ".4f"))
    print(row("DSR thr",   g("dsr_thr"),       ".4f"))
    dsr_row = [f"{'YES' if s.get('clears_dsr') else 'NO':>{W}}" for s in summaries]
    print(f"  {'ClearsDSR':<{H}}" + "".join(dsr_row))

    print(f"\n  Adj fallback : " + "  ".join(
        f"{h}M={s.get('pct_fallback', 0):.1f}%" for h, s in zip(holds, summaries)))
    print(f"  Runtime (s)  : " + "  ".join(
        f"{h}M={s.get('elapsed_s', 0):.0f}s" for h, s in zip(holds, summaries)))

    # ── Interpretation ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("INTERPRETATION")
    print(sep)

    for s in summaries:
        h  = s["hold_months"]
        ic = s.get("mean_ic",  np.nan)
        it = s.get("ic_t",     np.nan)
        gn = s.get("ann_gross", np.nan)
        nn = s.get("ann_net",   np.nan)
        cd = s.get("cost_drag_bps", np.nan)
        ic_sig = "significant (|t|>2)" if not np.isnan(it) and abs(it) > 2 else "NOT significant"
        net_pos = nn > 0 if not np.isnan(nn) else False
        print(f"\n  {h}M hold: "
              f"Gross={gn:+.2%}/yr, Cost={cd:.0f}bps/yr, Net={nn:+.2%}/yr | "
              f"IC={ic:.4f} ({ic_sig})")

    # Summary assessment
    print()
    nets_pos  = [s for s in summaries if not np.isnan(s.get("ann_net", np.nan)) and s["ann_net"] > 0]
    ic_sig_at = [s["hold_months"] for s in summaries
                 if not np.isnan(s.get("ic_t", np.nan)) and abs(s["ic_t"]) > 2]

    if nets_pos and ic_sig_at:
        best_h = nets_pos[-1]["hold_months"]   # longest positive-net hold
        if best_h in ic_sig_at:
            print(f"  *** Net return positive at {best_h}M with IC still significant → "
                  f"WALK-FORWARD CANDIDATE ***")
        else:
            print(f"  Net positive at {best_h}M but IC insignificant at that horizon. "
                  f"Signal decays faster than costs fall — family CLOSES.")
    elif ic_sig_at and not nets_pos:
        print(f"  IC significant through {max(ic_sig_at)}M but net never positive. "
              f"Real signal; costs remain prohibitive across all tested horizons. "
              f"Would need larger-cap universe or passive execution to be viable.")
    else:
        print(f"  IC insignificant and net negative at all tested horizons. Family CLOSES.")


def print_per_sweep_log(sweeps: list[dict]) -> None:
    """Per-period returns log for each sweep."""
    for sweep in sweeps:
        hm = sweep["hold_months"]
        records = sweep["period_records"]
        if not records:
            continue
        df = pd.DataFrame(records)
        print(f"\n{'='*72}")
        print(f"PERIOD LOG  {hm}M hold  ({len(df)} periods)")
        print(f"{'='*72}")
        print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} "
              f"{'TOver':>6} {'IC':>6} {'nL':>4} {'Mkt':>7}")
        print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*6} {'-'*4} {'-'*7}")
        for _, r in df.iterrows():
            print(f"  {str(r['rebalance_date'].date()):<11} "
                  f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
                  f"{r['avg_turnover']:>5.0%} {r['ic']:>6.3f} "
                  f"{int(r['n_long']):>4} {r['mkt_ret']:>6.2%}")


# =============================================================================
# SECTION 14: Save results + registry
# =============================================================================

def save_all_results(sweeps: list[dict]) -> None:
    all_records  = []
    all_coverage = []
    for sweep in sweeps:
        all_records.extend(sweep["period_records"])
        all_coverage.extend(sweep["adj_coverage"])
    if all_records:
        pd.DataFrame(all_records).to_csv(
            RESULTS / "max_decay_period_returns.csv", index=False)
    if all_coverage:
        pd.DataFrame(all_coverage).to_csv(
            RESULTS / "max_decay_adj_coverage.csv", index=False)
    print(f"\n[Saved] Decay results → {RESULTS}/")


def log_decay_registry(summaries: list[dict]) -> None:
    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()

    rows = []
    for s in summaries:
        h = s["hold_months"]
        study_name = f"max_S-MAX_decay_{h}M"
        row = {
            "timestamp":            datetime.now().isoformat(),
            "study":                study_name,
            "hypothesis":           (
                f"S-MAX holding-period decay study ({h}M hold). "
                "Signal: 21-day MAX, formation window fixed. "
                "Tests whether longer hold period allows cost drag to fall "
                "enough to make net return positive."
            ),
            "data_source":          "Tiingo (adjClose for signal, raw close for returns) + SEC XBRL",
            "status":               "completed_in_sample",
            "trial_number":         7,
            "trial_note":           (
                f"Hold-period decay study within IVOL/MAX family (trial #7). "
                f"{h}M hold variant. NOT a new independent trial. N_TRIALS=7 unchanged."
            ),
            "n_trials_dsr":         N_TRIALS,
            "rebalance_freq":       f"{h}-monthly",
            "portfolio_note":       "Beta-neutral (EW market proxy)",
            "n_periods":            s.get("n"),
            "mean_turnover":        s.get("mean_turnover"),
            "ann_return_gross":     s.get("ann_gross"),
            "annual_cost_drag_bps": s.get("cost_drag_bps"),
            "annual_ba_bps":        s.get("cost_ba_bps"),
            "annual_bw_bps":        s.get("cost_bw_bps"),
            "ann_return_net":       s.get("ann_net"),
            "sharpe":               s.get("sharpe"),
            "calmar":               s.get("calmar"),
            "max_drawdown":         s.get("max_dd"),
            "t_stat":               s.get("t_stat"),
            "market_beta":          s.get("beta"),
            "market_beta_tstat":    s.get("beta_t"),
            "mean_ic":              s.get("mean_ic"),
            "ic_t_stat":            s.get("ic_t"),
            "win_rate":             s.get("win_rate"),
            "per_period_sr":        s.get("per_period_sr"),
            "dsr_threshold":        s.get("dsr_thr"),
            "clears_dsr":           s.get("clears_dsr"),
            "adj_fallback_pct":     s.get("pct_fallback"),
        }
        rows.append(row)

    if not reg_df.empty and "study" in reg_df.columns:
        decay_studies = [f"max_S-MAX_decay_{h}M" for h in HOLD_MONTHS]
        reg_df = reg_df[~reg_df["study"].isin(decay_studies)].copy()

    reg_df = pd.concat([reg_df, pd.DataFrame(rows)], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] Decay study ({', '.join(f'{h}M' for h in HOLD_MONTHS)}) logged → {REGISTRY}")


# =============================================================================
# SECTION 15: Main
# =============================================================================

def main() -> None:
    t_wall = time.time()
    print("=" * 70)
    print("S-MAX HOLDING-PERIOD DECAY  (IVOL/MAX family — trial #7)")
    print("Signal: -MAX(21-day adjClose return). Formation window fixed.")
    print(f"Sweeps: {', '.join(f'{h}M' for h in HOLD_MONTHS)}  |  In-sample: 2015–2021")
    print("=" * 70)

    # ── Load caches ONCE ──────────────────────────────────────────────────────
    print("\n[Setup] Loading Tiingo metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("[Setup] Loading SEC prefilter...")
    prefilter_df = load_prefilter()
    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"])
                   for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    print(f"[Setup] Preloading raw close ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"[Setup] Loading adjClose pickle...")
    preload_all_adj_prices(survivor_tickers)
    adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    pct_adj    = 100 * adj_loaded / max(len(survivor_tickers), 1)
    print(f"  AdjClose: {adj_loaded:,}/{len(survivor_tickers):,} ({pct_adj:.1f}%)")
    if pct_adj < 90:
        print("  *** <90% adj coverage. Run research/ivol/run.py first. ***")
        sys.exit(1)

    print(f"[Setup] Preloading SEC facts ({len(survivor_ciks):,} CIKs)...")
    preload_sec_facts(survivor_ciks)

    # ── Run all three sweeps ──────────────────────────────────────────────────
    sweeps: list[dict] = []
    for hm in HOLD_MONTHS:
        result = run_one_sweep(hm, tiingo_tickers, cik_sic_map, delisted_map)
        sweeps.append(result)

    # ── Summarise and print ───────────────────────────────────────────────────
    summaries = [_sweep_summary(s) for s in sweeps]

    print_per_sweep_log(sweeps)
    print_comparison_table(summaries)

    save_all_results(sweeps)
    log_decay_registry(summaries)

    print(f"\n[Total runtime: {(time.time()-t_wall)/60:.1f} min]")


if __name__ == "__main__":
    main()
