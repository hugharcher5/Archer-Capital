"""Compute WACC via CAPM (cost of equity) + Damodaran synthetic rating (cost of debt)."""

from __future__ import annotations
import os
import numpy as np
import yfinance as yf
from dataclasses import dataclass
from dotenv import load_dotenv
from .data import RawData
from .drivers import Drivers

load_dotenv()

_WACC_SIGMA_FALLBACK: float = 0.015

# ── Damodaran implied ERP — easily editable constant ─────────────────────────
ERP: float = 0.045   # 4.5% (Damodaran current implied ERP)

# ── Damodaran synthetic-rating table (large/stable company, Jan 2024) ─────────
# Rows: (coverage_min_inclusive, coverage_max_exclusive, rating_label, spread)
_SYNTHETIC_RATING: list[tuple[float, float, str, float]] = [
    (12.50, float('inf'), 'Aaa/AAA', 0.0063),
    ( 9.50,       12.50, 'Aa2/AA',  0.0078),
    ( 7.50,        9.50, 'A1/A+',   0.0098),
    ( 6.00,        7.50, 'A2/A',    0.0113),
    ( 4.50,        6.00, 'A3/A-',   0.0128),
    ( 4.00,        4.50, 'Baa2/BBB',0.0163),
    ( 3.50,        4.00, 'Ba1/BB+', 0.0238),
    ( 3.00,        3.50, 'Ba2/BB',  0.0313),
    ( 2.50,        3.00, 'B1/B+',   0.0388),
    ( 2.00,        2.50, 'B2/B',    0.0463),
    ( 1.50,        2.00, 'B3/B-',   0.0563),
    ( 1.25,        1.50, 'Caa/CCC', 0.0713),
    ( 0.80,        1.25, 'Ca2/CC',  0.0863),
    ( 0.50,        0.80, 'C2/C',    0.1113),
    (float('-inf'), 0.50, 'D2/D',   0.1413),
]


@dataclass
class WACCResult:
    rf: float
    rf_source: str
    beta_raw: float
    beta_adj: float
    erp: float
    cost_of_equity: float
    coverage_ratio: float
    implied_rating: str
    credit_spread: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    equity_value: float     # market cap in local currency
    debt_value: float       # total debt in local currency
    equity_weight: float
    debt_weight: float
    wacc: float
    std_wacc:            float   # historical σ; equals _WACC_SIGMA_FALLBACK when < 3 years
    wacc_sigma_fallback: bool    # True when insufficient history for reconstruction


def _get_rf() -> tuple[float, str]:
    """Try FRED DGS10 first, fall back to ^TNX from yfinance."""
    api_key = os.getenv('FRED_API_KEY', '').strip()
    if api_key and api_key != 'your_api_key_here':
        try:
            from fredapi import Fred
            series = Fred(api_key=api_key).get_series('DGS10').dropna()
            return float(series.iloc[-1]) / 100, 'FRED DGS10'
        except Exception as e:
            print(f"  [WACC] FRED fetch failed ({e}); falling back to ^TNX.")
    try:
        rate = float(yf.Ticker('^TNX').fast_info.last_price) / 100
        return rate, '^TNX (yfinance)'
    except Exception:
        return 0.043, 'hardcoded fallback (4.30%)'


def _synthetic_rating(coverage: float) -> tuple[str, float]:
    for lo, hi, rating, spread in _SYNTHETIC_RATING:
        if lo <= coverage < hi:
            return rating, spread
    return 'D2/D', 0.1413


def _get_rf_annual_series():
    """Annual-average FRED DGS10 yield (decimal), index = int year.
    Returns empty pandas Series when FRED is unavailable."""
    import pandas as pd
    api_key = os.getenv('FRED_API_KEY', '').strip()
    if not api_key or api_key == 'your_api_key_here':
        return pd.Series(dtype=float)
    try:
        from fredapi import Fred
        daily = Fred(api_key=api_key).get_series('DGS10').dropna()
        return daily.groupby(daily.index.year).mean() / 100.0
    except Exception as e:
        print(f"  [WACC] FRED historical fetch failed ({e}); WACC-sigma fallback used.")
        import pandas as pd
        return pd.Series(dtype=float)


def _reconstruct_wacc_history(raw, drivers, beta_adj, eq_w, dt_w, rf_annual):
    """Reconstruct WACC for each year with available EBIT + interest expense.
    Beta (Blume-adjusted), ERP, and D/E weights are held fixed at current values;
    only the risk-free rate and coverage-derived credit spread vary year-to-year."""
    common_idx = raw.ebit.index.intersection(raw.interest_expense.index)
    reconstructed = []
    for ts in common_idx:
        ebit_val = float(raw.ebit[ts])
        int_val  = float(raw.interest_expense[ts])
        if ebit_val <= 0:
            continue   # skip loss years; negative coverage distorts reconstruction
        yr = ts.year
        if yr in rf_annual.index:
            rf_yr = float(rf_annual[yr])
        elif not rf_annual.empty:
            nearest = min(rf_annual.index, key=lambda y: abs(y - yr))
            rf_yr = float(rf_annual[nearest])
        else:
            continue
        cov_yr = ebit_val / int_val if int_val > 0 else float('inf')
        _, spread_yr = _synthetic_rating(cov_yr)
        coe_yr  = rf_yr + beta_adj * ERP
        cod_yr  = (rf_yr + spread_yr) * (1.0 - drivers.tax_rate)
        reconstructed.append(eq_w * coe_yr + dt_w * cod_yr)
    return reconstructed


def compute_wacc(raw: RawData, drivers: Drivers) -> WACCResult:
    rf, rf_source = _get_rf()

    # ── Cost of equity ────────────────────────────────────────────────────────
    beta_adj       = 0.67 * raw.beta + 0.33
    cost_of_equity = rf + beta_adj * ERP

    # ── Cost of debt via synthetic rating ─────────────────────────────────────
    ebit_last   = float(raw.ebit.iloc[-1])          if not raw.ebit.empty             else 0.0
    int_exp_last = float(raw.interest_expense.iloc[-1]) if not raw.interest_expense.empty else 0.0

    if int_exp_last > 0:
        coverage = ebit_last / int_exp_last
    else:
        coverage = float('inf')   # no interest expense → debt-free → AAA

    implied_rating, credit_spread = _synthetic_rating(coverage)
    cost_of_debt_pretax   = rf + credit_spread
    cost_of_debt_aftertax = cost_of_debt_pretax * (1 - drivers.tax_rate)

    # ── Market-value weights ──────────────────────────────────────────────────
    E = raw.market_cap_local
    D = raw.total_debt
    V = E + D
    eq_w = E / V if V > 0 else 1.0
    dt_w = D / V if V > 0 else 0.0

    wacc = eq_w * cost_of_equity + dt_w * cost_of_debt_aftertax

    # ── Historical WACC sigma ─────────────────────────────────────────────────
    rf_annual   = _get_rf_annual_series()
    hist_waccs  = _reconstruct_wacc_history(raw, drivers, beta_adj, eq_w, dt_w, rf_annual)
    if len(hist_waccs) >= 3:
        sigma_w    = float(np.std(hist_waccs))
        w_fallback = False
        print(f"  WACC historical σ  : {sigma_w:.3%}  "
              f"({len(hist_waccs)} yr reconstructed; Blume β & D/E held fixed)")
    else:
        sigma_w    = _WACC_SIGMA_FALLBACK
        w_fallback = True
        print(f"  WACC historical σ  : {_WACC_SIGMA_FALLBACK:.3%}  "
              f"[FALLBACK — only {len(hist_waccs)} reconstructable year(s)]")

    result = WACCResult(
        rf=rf,
        rf_source=rf_source,
        beta_raw=raw.beta,
        beta_adj=beta_adj,
        erp=ERP,
        cost_of_equity=cost_of_equity,
        coverage_ratio=coverage,
        implied_rating=implied_rating,
        credit_spread=credit_spread,
        cost_of_debt_pretax=cost_of_debt_pretax,
        cost_of_debt_aftertax=cost_of_debt_aftertax,
        equity_value=E,
        debt_value=D,
        equity_weight=eq_w,
        debt_weight=dt_w,
        wacc=wacc,
        std_wacc=sigma_w,
        wacc_sigma_fallback=w_fallback,
    )

    # ── Print ─────────────────────────────────────────────────────────────────
    ccy = raw.currency
    print(f"\n{'='*60}")
    print(f"  WACC COMPONENTS — {raw.ticker}")
    print(f"{'='*60}")
    print(f"\n  Risk-free rate     : {rf:.3%}  ({rf_source})")
    print(f"  Beta (raw)         : {raw.beta:.3f}")
    print(f"  Beta (Blume adj.)  : {beta_adj:.3f}  [= 0.67 × {raw.beta:.3f} + 0.33]")
    print(f"  Equity risk prem.  : {ERP:.2%}  (Damodaran implied ERP constant)")
    print(f"  Cost of equity     : {cost_of_equity:.3%}  [= {rf:.3%} + {beta_adj:.3f} × {ERP:.2%}]")
    print()
    cov_str = f"{coverage:.2f}×" if coverage != float('inf') else "∞ (no debt)"
    print(f"  Interest coverage  : {cov_str}  [EBIT / |Interest expense|, last year]")
    print(f"  Implied rating     : {implied_rating}")
    print(f"  Credit spread      : {credit_spread:.3%}")
    print(f"  Cost of debt (pre) : {cost_of_debt_pretax:.3%}  [= Rf + spread]")
    print(f"  Cost of debt (post): {cost_of_debt_aftertax:.3%}  [= pre × (1 − {drivers.tax_rate:.2%})]")
    print()
    print(f"  Equity (market cap): {ccy} {E/1e9:.2f}B   weight = {eq_w:.2%}")
    print(f"  Debt (total debt)  : {ccy} {D/1e9:.2f}B   weight = {dt_w:.2%}")
    print(f"  V = E + D          : {ccy} {V/1e9:.2f}B")
    print()
    print(f"  WACC               : {wacc:.3%}")
    print(f"  [= {eq_w:.2%} × {cost_of_equity:.3%}  +  {dt_w:.2%} × {cost_of_debt_aftertax:.3%}]")

    return result
