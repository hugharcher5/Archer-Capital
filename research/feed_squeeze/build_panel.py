"""
Assemble the single monthly panel used by all hypothesis tests.

Publication lags are applied HERE, in one place, with explicit comments.
After this module runs every value in the returned DataFrame is point-in-time:
a row dated month-end T contains only information that was publicly available
by the close of month T.

Column naming convention
------------------------
  oni_pub          ONI anomaly (°C), pub-lagged 1 month
  fishmeal_pub     Fishmeal price (USD/mt), pub-lagged 1 month
  d_fishmeal       Monthly log-change of fishmeal_pub
  salmon_pub       SSB salmon price (NOK/kg), pub-lagged 0 months
  salmon_ret       Monthly log-return of salmon_pub
  basket_ret       Equal-weighted log-return across ≥2 available farmer tickers
  <TICKER>_ret     Individual farmer log-returns (e.g. MOWI_ret, SALM_ret)
  BMK_ret          Benchmark Holdings (reference, excluded from basket)
"""

import numpy as np
import pandas as pd

from .config import (
    PUB_LAG, FARMER_TICKERS, REF_TICKER, START_DATE, TEST_B_MIN_TICKERS,
)
from .fetch_data import fetch_oni, fetch_fishmeal, fetch_ssb_salmon, fetch_equities


def build_panel(verbose: bool = True) -> "tuple[pd.DataFrame, dict]":
    """
    Fetch raw data, apply publication lags, and join into a monthly panel.

    Returns
    -------
    panel    : pd.DataFrame  — monthly rows aligned on month-end dates
    coverage : dict          — per-ticker equity coverage metadata
    """
    if verbose:
        print("Fetching ONI …")
    oni_raw = fetch_oni()

    if verbose:
        print("Fetching fishmeal (World Bank Pink Sheet) …")
    fm_raw = fetch_fishmeal()

    if verbose:
        print("Fetching SSB salmon price …")
    salmon_raw = fetch_ssb_salmon()

    if verbose:
        print("Fetching equity prices (yfinance) …")
    eq_raw, coverage = fetch_equities()

    # ── Apply publication lags ────────────────────────────────────────────────
    # shift(k) moves the series forward k periods so that at row T the visible
    # value is what was last published as of T (i.e. the value from T-k).

    # ONI: published ~1 month after the centre month.
    oni = oni_raw.shift(PUB_LAG["oni"])
    oni.name = "oni_pub"

    # Fishmeal: World Bank releases monthly data with ~1-month delay.
    if fm_raw is not None:
        fm = fm_raw.shift(PUB_LAG["fishmeal"])
        fm.name = "fishmeal_pub"
    else:
        fm = pd.Series(dtype=float, name="fishmeal_pub")

    # Salmon price: SSB weekly releases within 1-2 weeks; treat as 0 for monthly.
    if salmon_raw is not None:
        sp = salmon_raw.shift(PUB_LAG["salmon_price"])   # shift(0) = no-op
        sp.name = "salmon_pub"
    else:
        sp = pd.Series(dtype=float, name="salmon_pub")

    # ── Log changes / returns ─────────────────────────────────────────────────
    d_fm = np.log(fm / fm.shift(1))
    d_fm.name = "d_fishmeal"

    d_sp = np.log(sp / sp.shift(1))
    d_sp.name = "salmon_ret"

    # Individual farmer equity returns
    farmer_cols = [t for t in FARMER_TICKERS if t in eq_raw.columns]
    eq_prices = eq_raw[farmer_cols]
    eq_ret     = np.log(eq_prices / eq_prices.shift(1))
    eq_ret.columns = [_short(t) + "_ret" for t in farmer_cols]

    # Basket: equal-weighted average across months with ≥ TEST_B_MIN_TICKERS tickers
    n_valid  = eq_ret.notna().sum(axis=1)
    basket   = eq_ret.mean(axis=1)
    basket[n_valid < TEST_B_MIN_TICKERS] = np.nan   # drop months too thin
    basket.name = "basket_ret"

    # Reference ticker (not in basket)
    if REF_TICKER in eq_raw.columns:
        ref_ret = np.log(eq_raw[REF_TICKER] / eq_raw[REF_TICKER].shift(1))
        ref_ret.name = _short(REF_TICKER) + "_ret"
    else:
        ref_ret = pd.Series(dtype=float, name=_short(REF_TICKER) + "_ret")

    # ── Join on month-end index ───────────────────────────────────────────────
    parts = [oni, fm, d_fm, sp, d_sp, basket, ref_ret, eq_ret]
    panel = pd.concat(parts, axis=1).sort_index()
    panel = panel[panel.index >= pd.Timestamp(START_DATE) + pd.offsets.MonthEnd(0)]

    if verbose:
        _print_summary(panel, coverage)

    return panel, coverage


def _short(ticker: str) -> str:
    """'MOWI.OL' → 'MOWI'"""
    return ticker.split(".")[0]


def _print_summary(panel: pd.DataFrame, coverage: dict) -> None:
    print(f"\nPanel: {panel.shape[0]} months × {panel.shape[1]} columns "
          f"({panel.index[0].date()} → {panel.index[-1].date()})\n")

    print("=== EQUITY COVERAGE ===")
    for ticker, info in coverage.items():
        if info is None:
            print(f"  {ticker:<12s}  NO DATA")
        else:
            first, last, n = info
            print(f"  {ticker:<12s}  {first} → {last}  ({n} weekly obs)")

    print("\nNon-null months per series:")
    print(panel.count().to_string())
