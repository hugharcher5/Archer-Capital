"""
S-MAX: Maximum Daily Return (Lottery / MAX Effect)   (trial #7 — IVOL/MAX family)
===================================================================================
Signal   : MAX = single highest daily return over trailing 21 trading days ≤ t.
           Computed from adjClose (split+div adjusted) to prevent splits from
           generating fake extreme returns that falsely flag stocks as lottery names.
           Signal = −MAX  (long lowest MAX, short highest MAX).
Portfolio: Long bottom 20% (lowest MAX), short top 20% (highest MAX).
           Min 20 names per leg.  BETA-NEUTRAL (not dollar-neutral).
           Equal-weighted within each leg; leg dollar allocations scaled so
           ex-ante net portfolio beta ≈ 0:
             long_$ × mean_β_long = short_$ × mean_β_short
           Beta estimated from a separate 60-day OLS regression vs EW universe
           return (MAX signal does not itself require a regression).
Rebalance: MONTHLY. Hold: 1 month. (~72 periods over 2015–2021).
           Intentional deviation from the quarterly default: the MAX / lottery
           characteristic decays within weeks, so a quarterly hold tests a stale
           signal. Monthly rebalancing is signal-appropriate. High turnover is
           expected and modelled explicitly — logged in the registry.
In-sample: 2015-01-01 → 2021-01-01. No walk-forward.
Trial    : IVOL/MAX family — same trial #7 as S-IVOL. This is the MAX sibling
           of the behavioral extreme-return aversion family. NOT a new independent
           trial. N_TRIALS = 7 unchanged.

Key construction differences vs S-IVOL:
  1. Signal: MAX (21-trading-day peak return) instead of IVOL (60-day residual vol).
  2. Rebalance: Monthly (not quarterly). ~72 periods instead of 24.
  3. Beta: Computed from a separate 60-day OLS vs EW universe return (not cap-
     weighted). MAX signal does not itself produce a beta estimate.
  4. Cost model: monthly borrow (/12 not /4); bid-ask scaled by actual per-period
     turnover (not assumed 50%), enabling accurate drag decomposition.
  5. Cost drag reported separately in bps/year: gross return + cost drag → net
     return.  IC (rank correlation, cost-independent) reported prominently to
     distinguish signal quality from cost headwinds.

No-look-ahead guarantee
  - MAX window and beta window use adjClose for dates strictly ≤ t.
  - Market cap = SEC shares-as-filed (filing_date ≤ t) × last_close ≤ t.
  - Entry price: first close on or after t; returns measured t → t+1 month.
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

# ── Universe filters (same as S-IVOL) ─────────────────────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── Signal parameters ─────────────────────────────────────────────────────────
MAX_WINDOW_DAYS   = 21    # trading-day window for MAX signal
MAX_MIN_DAYS      = 5     # minimum observations required
MAX_LOOKBACK_CAL  = 35    # calendar days to cover MAX_WINDOW_DAYS trading days

BETA_WINDOW_DAYS  = 60    # trading-day window for beta regression
BETA_MIN_DAYS     = 20    # minimum obs for valid beta estimate
BETA_LOOKBACK_CAL = 95    # calendar days to cover BETA_WINDOW_DAYS trading days

# ── Rebalance schedule: MONTHLY ───────────────────────────────────────────────
IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="MS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 72 monthly periods

# ── Cost model (monthly variant of S-IVOL) ────────────────────────────────────
SPREAD_MIN_BPS    = 20     # one-way half-spread at $2B mcap
SPREAD_MAX_BPS    = 100    # one-way half-spread at $100M mcap
BORROW_MIN_ANNUAL = 0.005  # 0.5%/yr at $2B
BORROW_MAX_ANNUAL = 0.020  # 2.0%/yr at $100M

# ── DSR: IVOL/MAX = one family, not a new trial ──────────────────────────────
N_TRIALS = 7   # H1 + E1 + E2 + E3 + E4 + S9 + (IVOL/MAX family)


# =============================================================================
# SECTION 2: Rate-limited HTTP (Tiingo, SEC)
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
# SECTION 3: Tiingo metadata helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    """Load Tiingo supported_tickers (cached). US exchange stocks only."""
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
    print(f"[Tiingo] Fetched {len(df):,} US tickers → {path}")
    return df


# =============================================================================
# SECTION 4: SEC prefilter
# =============================================================================

def load_prefilter() -> pd.DataFrame:
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(f"SEC prefilter not found: {path}. Run mgmt_pit/run.py first.")
    df = pd.read_csv(path)
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors (mgmt_pit cache).")
    return df


# =============================================================================
# SECTION 5: Raw price cache
# =============================================================================

_PRICES_PKL   = CACHE / "prices_all.pkl"
_PRICE_CACHE: dict[str, pd.DataFrame] = {}
_PRICE_EMPTY  = pd.DataFrame(columns=["date", "close", "volume"])


def preload_all_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if _PRICES_PKL.exists():
        with open(_PRICES_PKL, "rb") as f:
            data = pickle.load(f)
        _PRICE_CACHE.update(data.get("prices", {}))
        print(f"[Preload] Raw prices: {len(_PRICE_CACHE):,} tickers in {time.time()-t0:.1f}s")
        return
    raise FileNotFoundError(
        f"Raw price pickle not found: {_PRICES_PKL}. Run research/ivol/run.py first."
    )


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
    if sub.empty:
        return None
    return float(sub.iloc[-1]["close"])


def _avg_dollar_vol(ticker: str, as_of: pd.Timestamp, lookback: int = 60) -> float:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return 0.0
    sub = df[df["date"] <= as_of].tail(lookback)
    if sub.empty:
        return 0.0
    return float((sub["close"] * sub["volume"]).mean())


# =============================================================================
# SECTION 5b: Adj-price cache (signal layer — load from pickle)
# =============================================================================

_ADJ_PKL      = CACHE / "adjclose_all.pkl"
_ADJ_DIR      = CACHE / "tiingo_adj"
_ADJ_CACHE:   dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str] = set()
_ADJ_EMPTY    = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices(tickers: list[str]) -> None:
    """Load adjClose from pickle. Expects pickle fully populated (run ivol/run.py first)."""
    t0 = time.time()
    if _ADJ_PKL.exists():
        with open(_ADJ_PKL, "rb") as f:
            data = pickle.load(f)
        _ADJ_CACHE.update(data.get("adj", {}))
        _ADJ_SENTINEL.update(data.get("sentinels", set()))
        adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
        print(f"[AdjPreload] {adj_loaded:,} tickers in {time.time()-t0:.1f}s")
    else:
        print("[AdjPreload] No pickle found — run research/ivol/run.py first to populate cache.")
        sys.exit(1)


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 6: SEC facts (shares outstanding, PIT)
# =============================================================================

_FACTS_CACHE: dict[int, Optional[dict]] = {}
_FACTS_TRIMMED  = CACHE / "sec_facts_trimmed.json"
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
    print(f"[Preload] SEC facts: {loaded:,} loaded in {time.time()-t0:.1f}s")


def _get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    """Most recent CommonStockSharesOutstanding where filing_date ≤ as_of."""
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
# SECTION 7: Universe builder  (identical to S-IVOL)
# =============================================================================

def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
) -> pd.DataFrame:
    """
    PIT investable universe at rebalance_date.
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
    print(f"  [Universe] {rebalance_date.date()} → {len(df):,} names")
    return df


# =============================================================================
# SECTION 8: MAX signal + beta computation  (signal layer — adjClose)
# =============================================================================

def compute_max_signals(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple:
    """
    Compute MAX signal and 60-day market-model beta for each universe member.

    MAX: single highest daily adjClose return over the trailing 21 trading days ≤ t.
    Beta: OLS slope of (daily adj return) ~ (EW universe daily return) over 60 days.
         Market proxy = equal-weighted daily return of all universe members (EW,
         not cap-weighted — as specified for MAX to distinguish from S-IVOL).

    Signal = −MAX  (higher signal → lower MAX → long leg).
    Falls back to raw close for MAX when adjClose unavailable.

    Returns (signals_df[ticker, max_signal, beta], n_fallback, n_total).
    """
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)
    max_start  = rebalance_date - pd.Timedelta(days=MAX_LOOKBACK_CAL)

    # ── Pass 1: collect 60-day adj return series for EW market proxy ──────────
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

    # EW market return: mean across all tickers on each date
    if adj_series:
        ret_mat = pd.DataFrame(adj_series).sort_index()
        # Clip extreme returns (> ±50%) to suppress data errors in market proxy
        ew_mkt  = ret_mat.clip(-0.5, 0.5).mean(axis=1)
    else:
        ew_mkt = pd.Series(dtype=float)

    # ── Pass 2: per-ticker MAX signal and beta ────────────────────────────────
    n_fallback = 0
    n_total    = 0
    records    = []

    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        n_total += 1

        # --- MAX signal ---
        using_raw = False
        if ticker in adj_series:
            ret_full = adj_series[ticker]
            # Restrict to the MAX window
            ret_win  = ret_full[ret_full.index >= max_start]
        else:
            # Fallback: raw close for MAX window
            raw = _get_price_slice(ticker, max_start, rebalance_date)
            if raw.empty:
                continue
            s = raw.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret_win  = s.pct_change().dropna()
            using_raw = True

        if len(ret_win) < MAX_MIN_DAYS:
            continue

        # Last MAX_WINDOW_DAYS trading-day returns
        rets_21  = ret_win.iloc[-MAX_WINDOW_DAYS:] if len(ret_win) >= MAX_WINDOW_DAYS else ret_win
        max_ret  = float(rets_21.max())

        if not np.isfinite(max_ret):
            continue

        if using_raw:
            n_fallback += 1

        # --- Beta (60-day OLS vs EW universe, equal-weighted) ---
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
# SECTION 9: Return calculation  (return layer — raw close)
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
    delisted_map: dict,
) -> pd.DataFrame:
    """
    Survivorship-bias-free holding-period returns using raw close.
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
            exit_cands = prices[prices["date"] <= delist_date]
            exit_price = float(
                (exit_cands if not exit_cands.empty else prices).iloc[-1]["close"]
            )
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
    """One-way half-spread: 100bps at $100M → 20bps at $2B."""
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    """Annual borrow rate: 2%/yr at $100M → 0.5%/yr at $2B."""
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame) -> tuple:
    """
    Long bottom 20% of max_signal (= lowest MAX), short top 20% (= highest MAX).
    Min 20 per leg.  Equal-weight within each leg.
    Beta-neutral sizing: long_$ × β_L = short_$ × β_S, gross = 2×.
    Returns (longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short).
    """
    df = signals_df.dropna(subset=["max_signal", "raw_return"]).copy()
    n  = len(df)
    if n < 40:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan

    leg = max(int(np.ceil(n * 0.20)), 20)
    leg = min(leg, n // 2)

    ranked = df.sort_values("max_signal", ascending=False)   # highest signal = lowest MAX = long
    longs  = ranked.iloc[:leg]["ticker"].tolist()
    shorts = ranked.iloc[-leg:]["ticker"].tolist()

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    # ── Beta-neutral leg scaling ──────────────────────────────────────────────
    if "beta" in df.columns:
        beta_map = df.set_index("ticker")["beta"].to_dict()
        mean_beta_long  = float(np.nanmean([beta_map.get(t, np.nan) for t in longs]))
        mean_beta_short = float(np.nanmean([beta_map.get(t, np.nan) for t in shorts]))
        denom = mean_beta_long + mean_beta_short
        if np.isfinite(denom) and denom > 0.05:
            long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
            short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
        else:
            print("  [Portfolio] β_L+β_S ≤ 0.05 — falling back to 50/50.")
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
) -> tuple:
    """
    Monthly cost model: bid-ask scaled by actual turnover, borrow /12.
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
        ba_cost += turnover * 2.0 * _spread(mc) * w    # round-trip × actual turnover
        if is_short:
            borrow_cost += (_borrow(mc) / 12.0) * w    # monthly borrow

    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    else:
        return (gross  - total_cost),  gross,   ba_cost, borrow_cost


# =============================================================================
# SECTION 11: Backtest engine
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
        end = r["endDate"]
        if pd.notna(end):
            delisted_map[r["ticker"]] = end

    period_records: list = []
    universe_audit: list = []
    adj_coverage:   list = []

    prev_long_set:  set = set()
    prev_short_set: set = set()

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        t_next     = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        hold_start = t
        hold_end   = t_next

        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} → {hold_end.date()}", end="  ")

        # ── 1. Universe ───────────────────────────────────────────────────────
        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERROR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] Empty universe.")
            continue

        # ── 2. MAX signal + beta ──────────────────────────────────────────────
        signals_df, n_fallback, n_computed = compute_max_signals(univ, t)
        n_adj_ok   = n_computed - n_fallback
        pct_fallbk = 100 * n_fallback / max(n_computed, 1)
        print(f"  sig={len(signals_df)}/{len(univ)} adj={n_adj_ok} fb={n_fallback}({pct_fallbk:.0f}%)",
              end="")

        adj_coverage.append({
            "rebalance_date": t,
            "n_universe":     len(univ),
            "n_computed":     n_computed,
            "n_valid_signal": len(signals_df),
            "n_adj":          n_adj_ok,
            "n_fallback":     n_fallback,
            "pct_fallback":   pct_fallbk,
        })

        # ── 3. Merge signals; compute hold-period returns ─────────────────────
        univ = univ.merge(signals_df, on="ticker", how="left")
        try:
            returns_df = compute_returns(univ, hold_start, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERROR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())

        universe_audit.append({
            "rebalance_date":  t,
            "hold_end":        hold_end,
            "n_universe":      len(returns_df),
            "n_with_signal":   int(returns_df["max_signal"].notna().sum()),
            "n_delisted":      n_delisted,
            "n_valid_returns": int(returns_df["raw_return"].notna().sum()),
        })

        # ── 4. Portfolio (beta-neutral) ───────────────────────────────────────
        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short) = build_portfolio(returns_df)
        if not longs:
            print(f"  [SKIP] n<40")
            continue

        # ── 5. Turnover (vs previous period's positions) ──────────────────────
        if prev_long_set:
            long_entries  = set(longs)  - prev_long_set
            short_entries = set(shorts) - prev_short_set
            long_turnover  = len(long_entries)  / max(len(longs),  1)
            short_turnover = len(short_entries) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0   # first period: full entry

        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        # ── 6. Leg returns (monthly cost model, actual turnover) ──────────────
        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_detail(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw
        ls_cost  = ls_ba + ls_bw

        # ── 7. Market proxy (EW full universe) ───────────────────────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if len(valid_all) > 0 else np.nan

        # ── 8. IC: Spearman(max_signal, realized_return) ─────────────────────
        # IC > 0: lower MAX stocks outperformed higher MAX (hypothesis confirmed).
        valid_sig = returns_df.dropna(subset=["max_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["max_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  L/S={ls_net:+.2%}  gross={ls_gross:+.2%}  cost={ls_cost:.3%}"
              f"  to={avg_turnover:.0%}  β_L={mean_beta_long:.2f}  β_S={mean_beta_short:.2f}"
              f"  IC={ic:.3f}")

        period_records.append({
            "rebalance_date":   t,
            "hold_end":         t_next,
            "n_universe":       len(returns_df),
            "n_with_signal":    int(returns_df["max_signal"].notna().sum()),
            "n_delisted":       n_delisted,
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
        })

    elapsed = time.time() - t_total
    return {
        "period_records": period_records,
        "universe_audit": universe_audit,
        "adj_coverage":   adj_coverage,
        "elapsed_s":      elapsed,
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
    # Monthly → annual: 12 periods/year
    ann_ret = float((1 + total) ** (12 / n) - 1)
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
        return {"beta": np.nan, "beta_t": np.nan, "r_squared": np.nan}
    x = combined["mkt"].values
    y = combined["ls"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"beta": np.nan, "beta_t": np.nan, "r_squared": np.nan}
    beta    = float(coeffs[1])
    y_hat   = X @ coeffs
    ss_res  = float(np.dot(y - y_hat, y - y_hat))
    ss_tot  = float(np.dot(y - y.mean(), y - y.mean()))
    r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    n       = len(y)
    se      = np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1))
    beta_t  = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t, "r_squared": r2}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
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
    cov  = pd.DataFrame(coverage)
    aud  = pd.DataFrame(audit)

    ls_series   = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series  = df["mkt_ret"]
    ic_series   = df["ic"].dropna()

    m          = compute_metrics(ls_series)
    m_gross    = compute_metrics(gross_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    obs_sr_m   = m["sharpe"] / np.sqrt(12) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = (
        bool(obs_sr_m > dsr_thr)
        if not np.isnan(obs_sr_m) and not np.isnan(dsr_thr)
        else False
    )

    mean_ic  = float(ic_series.mean())    if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    # Cost drag
    mean_monthly_cost    = float(df["ls_cost"].mean())
    annual_cost_drag_bps = mean_monthly_cost * 12 * 10_000
    mean_monthly_ba_bps  = float(df["ls_ba_cost"].mean()) * 12 * 10_000
    mean_monthly_bw_bps  = float(df["ls_borrow_cost"].mean()) * 12 * 10_000
    mean_turnover        = float(df["avg_turnover"].mean())
    mean_long_scale      = float(df["long_scale"].mean())
    mean_short_scale     = float(df["short_scale"].mean())
    mean_bl              = float(df["mean_beta_long"].mean())
    mean_bs              = float(df["mean_beta_short"].mean())

    # AdjClose coverage
    total_fallback   = int(cov["n_fallback"].sum())
    total_computed   = int(cov["n_computed"].sum())
    overall_pct_fall = 100 * total_fallback / max(total_computed, 1)

    sep = "=" * 76

    # ── AdjClose coverage ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("SIGNAL LAYER — ADJ CLOSE COVERAGE CHECK")
    print(sep)
    print(f"  Tickers using adjClose  : {total_computed - total_fallback:,} / {total_computed:,}")
    print(f"  Tickers using raw close : {total_fallback:,} / {total_computed:,}  ({overall_pct_fall:.1f}%)")
    if overall_pct_fall > 10:
        print(f"\n  *** WARNING: {overall_pct_fall:.1f}% fallback. Signal may be split-contaminated. ***")
    else:
        print(f"\n  Adj-layer coverage adequate (<10% fallback). Signal integrity confirmed.")

    # ── Performance summary ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("S-MAX: MAXIMUM DAILY RETURN — IN-SAMPLE RESULTS  (IVOL/MAX family, trial #7)")
    print("In-sample: 2015-01-01 → 2021-01-01  |  Signal: -MAX (long lowest, short highest)")
    print("Rebalance: MONTHLY  |  Construction: BETA-NEUTRAL  |  Market proxy: EW universe")
    print(sep)

    # ── IC prominently (cost-independent signal quality) ─────────────────────
    print(f"\n  *** INFORMATION COEFFICIENT (cost-independent signal quality) ***")
    print(f"  Mean IC (Spearman, low-MAX vs ret)  : {mean_ic:>12.4f}")
    print(f"  IC t-stat                            : {ic_tstat:>12.3f}")
    print(f"  Interpretation: IC > 0 = low-MAX stocks outperformed (hypothesis confirmed).")
    print(f"                  IC t-stat > 2 = signal has genuine predictive power.")

    print(f"\n  {'Metric':<44} {'Value':>14}")
    print(f"  {'-'*44} {'-'*14}")
    print(f"  {'Total return (cumulative, net)':<44} {m['total_ret']:>13.2%}")
    print(f"  {'Annualized return (net)':<44} {m['ann_ret']:>13.2%}")
    print(f"  {'Annualized return (GROSS, before costs)':<44} {m_gross['ann_ret']:>13.2%}")
    print(f"  {'Total cost drag (annualized bps)':<44} {annual_cost_drag_bps:>13.0f}")
    print(f"    {'  of which: bid-ask (bps/yr)':<42} {mean_monthly_ba_bps:>13.0f}")
    print(f"    {'  of which: borrow  (bps/yr)':<42} {mean_monthly_bw_bps:>13.0f}")
    print(f"  {'Mean actual turnover per period':<44} {mean_turnover:>13.0%}")
    print(f"  {'Sharpe ratio (annualized, net)':<44} {m['sharpe']:>13.3f}")
    print(f"  {'CALMAR ratio':<44} {m['calmar']:>13.3f}")
    print(f"  {'Max drawdown':<44} {m['max_dd']:>13.2%}")
    print(f"  {'t-stat (mean monthly net return)':<44} {m['t_stat']:>13.3f}")
    print(f"  {'Market beta (vs EW universe)':<44} {beta_stats['beta']:>13.3f}")
    print(f"    {'beta t-stat':<42} {beta_stats['beta_t']:>13.3f}")
    print(f"    {'R-squared':<42} {beta_stats['r_squared']:>13.3f}")
    print(f"  {'Mean gross long exposure ($)':<44} {mean_long_scale:>13.3f}")
    print(f"  {'Mean gross short exposure ($)':<44} {mean_short_scale:>13.3f}")
    print(f"  {'Mean ex-ante β long leg':<44} {mean_bl:>13.3f}")
    print(f"  {'Mean ex-ante β short leg':<44} {mean_bs:>13.3f}")
    print(f"  {'Win rate':<44} {m['win_rate']:>13.1%}")
    print(f"  {'Periods (N)':<44} {m['n']:>13d}")
    print(f"  {'AdjClose fallback rate':<44} {overall_pct_fall:>12.1f}%")

    print(f"\n  --- DSR Check (N_trials={N_TRIALS}, IVOL/MAX = one family, monthly Sharpe units) ---")
    print(f"  {'Observed monthly Sharpe':<44} {obs_sr_m:>13.4f}")
    print(f"  {'DSR threshold (monthly Sharpe)':<44} {dsr_thr:>13.4f}")
    print(f"  {'Clears DSR':<44} {'YES' if clears_dsr else 'NO':>13}")

    print(f"\n  Raw monthly returns (compute α vs T-bill Rf externally):")
    print(f"  Mean monthly L/S (net) : {ls_series.mean():.4%}")
    print(f"  Std monthly  L/S (net) : {ls_series.std(ddof=1):.4%}")

    # ── Monthly returns log ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("MONTHLY RETURNS LOG  (IC > 0 = low-MAX outperformed = hypothesis confirmed)")
    print("Gross = before costs.  TOver = actual turnover this period.")
    print(sep)
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} {'TOver':>6} {'LScl':>5} {'SScl':>5} "
          f"{'Mkt':>7} {'IC':>6}  {'nL':>4} {'Dlist':>5}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*5} "
          f"{'-'*7} {'-'*6}  {'-'*4} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
              f"{r['avg_turnover']:>5.0%} "
              f"{r['long_scale']:>4.2f}× {r['short_scale']:>4.2f}× "
              f"{r['mkt_ret']:>6.2%} {r['ic']:>6.3f}  "
              f"{int(r['n_long']):>4} {int(r['n_delisted']):>5}")

    # ── Drawdown series ──────────────────────────────────────────────────────
    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (monthly):")
    for i, (date, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " ◄ MAX DD" if val == m["max_dd"] else ""
        print(f"    {str(date.date()):<11}  {val:>7.2%}{marker}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def save_results(results: dict) -> None:
    records  = results["period_records"]
    audit    = results["universe_audit"]
    coverage = results["adj_coverage"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "max_period_returns.csv", index=False)
    pd.DataFrame(audit).to_csv(RESULTS / "max_universe_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "max_adj_coverage.csv", index=False)
    print(f"\n[Saved] Results → {RESULTS}/")


def log_registry(results: dict) -> None:
    records = results["period_records"]
    if not records:
        return

    df          = pd.DataFrame(records)
    ls_series   = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series  = df["mkt_ret"]
    ic_series   = df["ic"].dropna()
    m           = compute_metrics(ls_series)
    m_gross     = compute_metrics(gross_series)
    beta_stats  = compute_market_beta(ls_series, mkt_series)

    obs_sr_m   = m["sharpe"] / np.sqrt(12) if not np.isnan(m["sharpe"]) else np.nan
    dsr_thr    = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr = bool(obs_sr_m > dsr_thr) if not np.isnan(obs_sr_m) and not np.isnan(dsr_thr) else False

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    mean_monthly_cost    = float(df["ls_cost"].mean())
    annual_cost_drag_bps = mean_monthly_cost * 12 * 10_000
    mean_turnover        = float(df["avg_turnover"].mean())
    cov = pd.DataFrame(results["adj_coverage"])
    overall_fallback_pct = (100 * cov["n_fallback"].sum() / max(cov["n_computed"].sum(), 1)
                            if not cov.empty else np.nan)

    row = {
        "timestamp":             datetime.now().isoformat(),
        "study":                 "max_S-MAX",
        "hypothesis":            (
            "Small/mid-cap stocks with extreme recent single-day gains (lottery tickets) "
            "are overpriced by attention-driven retail investors and subsequently underperform. "
            "Long lowest-MAX (dullest), short highest-MAX (lottery). "
            "IVOL/MAX family: same behavioral family as S-IVOL (extreme-return aversion)."
        ),
        "data_source":           "Tiingo (adjClose for MAX signal, raw close for returns) + SEC XBRL",
        "status":                "completed_in_sample",
        "trial_number":          7,
        "trial_note":            (
            "MAX sibling of IVOL/MAX behavioral family. Same trial #7 as S-IVOL. "
            "NOT a new independent trial. N_TRIALS=7 unchanged."
        ),
        "n_trials_dsr":          N_TRIALS,
        "rebalance_freq":        "monthly",
        "rebalance_note":        (
            "Monthly (intentional deviation from quarterly default): MAX/lottery "
            "effect decays within weeks; quarterly hold tests a stale signal."
        ),
        "portfolio_note":        "Beta-neutral (EW market proxy for beta, not cap-weighted)",
        "ann_return_net":        m["ann_ret"],
        "ann_return_gross":      m_gross["ann_ret"],
        "annual_cost_drag_bps":  annual_cost_drag_bps,
        "mean_turnover":         mean_turnover,
        "sharpe":                m["sharpe"],
        "calmar":                m["calmar"],
        "max_drawdown":          m["max_dd"],
        "t_stat":                m["t_stat"],
        "market_beta":           beta_stats["beta"],
        "market_beta_tstat":     beta_stats["beta_t"],
        "mean_long_scale":       float(df["long_scale"].mean()),
        "mean_short_scale":      float(df["short_scale"].mean()),
        "mean_beta_long":        float(df["mean_beta_long"].mean()),
        "mean_beta_short":       float(df["mean_beta_short"].mean()),
        "mean_ic":               mean_ic,
        "ic_t_stat":             ic_tstat,
        "win_rate":              m["win_rate"],
        "n_periods":             m["n"],
        "obs_sr_monthly":        obs_sr_m,
        "dsr_threshold":         dsr_thr,
        "clears_dsr":            clears_dsr,
        "adj_fallback_pct":      overall_fallback_pct,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    # Remove any prior S-MAX entry (idempotent re-run)
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "max_S-MAX"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] max_S-MAX logged (IVOL/MAX family, trial #7) → {REGISTRY}")


# =============================================================================
# SECTION 14: Main
# =============================================================================

def main() -> None:
    print("=" * 76)
    print("S-MAX: Maximum Daily Return  (IVOL/MAX family — trial #7)")
    print("In-sample: 2015-01-01 → 2021-01-01  |  72 monthly periods")
    print("Signal: MAX = peak adjClose daily return over trailing 21 trading days")
    print("Rebalance: MONTHLY  |  Beta-neutral  |  Market proxy: EW universe")
    print("=" * 76)

    if not TIINGO_KEY:
        print("\n[WARNING] TIINGO_API_KEY not set. AdjClose layer disabled.")

    # ── Step 1: Tiingo metadata ────────────────────────────────────────────────
    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    # ── Step 2: SEC prefilter ──────────────────────────────────────────────────
    print("\n[Step 2] Loading SEC pre-filter (mgmt_pit cache)...")
    prefilter_df = load_prefilter()

    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    # ── Step 3: Price caches ───────────────────────────────────────────────────
    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle (signal layer)...")
    preload_all_adj_prices(survivor_tickers)
    adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    pct_adj    = 100 * adj_loaded / max(len(survivor_tickers), 1)
    print(f"  AdjClose: {adj_loaded:,}/{len(survivor_tickers):,} ({pct_adj:.1f}%)")
    if pct_adj < 90:
        print(f"  *** <90% adj coverage. Run research/ivol/run.py first to repair. ***")
        sys.exit(1)

    # ── Step 4: SEC facts ──────────────────────────────────────────────────────
    print(f"\n[Step 4] Preloading SEC facts ({len(survivor_ciks):,} CIKs)...")
    preload_sec_facts(survivor_ciks)

    # ── Step 5: Backtest ───────────────────────────────────────────────────────
    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} monthly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    # ── Step 6: Output ─────────────────────────────────────────────────────────
    print_results(results)
    save_results(results)
    log_registry(results)


if __name__ == "__main__":
    main()
