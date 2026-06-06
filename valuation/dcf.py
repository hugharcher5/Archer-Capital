"""
Pure DCF engine.

value(assumptions)         → float  — fast, no I/O; suitable for Monte Carlo.
detailed_value(assumptions) → DCFResult  — same maths plus full forecast table.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ── Module-level constants (referenced externally by run_dcf) ─────────────────
TERMINAL_G: float            = 0.025   # 2.5% default terminal growth rate
HIGH_GROWTH_THRESHOLD: float = 0.15    # use 10-year horizon if starting growth > this


@dataclass
class Assumptions:
    ticker: str
    currency: str               # reporting currency label

    # ── Core DCF inputs (the only things value() needs) ───────────────────────
    start_revenue: float        # most recent annual revenue in reporting currency
    revenue_growth: float       # starting growth rate (decimal, e.g. 0.12)
    ebit_margin: float
    tax_rate: float
    da_pct: float               # D&A / Revenue
    capex_pct: float            # CapEx / Revenue
    nwc_pct: float              # operating NWC / Revenue
    terminal_g: float
    wacc: float
    net_debt: float             # total debt − cash, in reporting currency
    diluted_shares: float       # actual share count
    fx_rate: float              # local → USD  (1.0 if already USD)
    forecast_years: int         # 5 (normal) or 10 (high-growth)

    # ── WACC decomposition — carried for display; not used in value() ─────────
    rf: float
    erp: float
    beta_adj: float
    cost_of_equity: float
    cost_of_debt_pretax: float
    cost_of_debt_aftertax: float
    equity_weight: float
    debt_weight: float
    implied_rating: str

    # ── Market data — for cross-checks ────────────────────────────────────────
    current_price_usd: float
    current_ebitda: float       # most recent actual EBITDA in local currency
    market_ev_local: float      # market cap + net debt, in local currency


@dataclass
class DCFResult:
    value_per_share_usd: float
    value_per_share_local: float
    equity_value_local: float
    ev_local: float
    pv_explicit: float          # PV of explicit forecast FCFFs
    pv_tv: float                # PV of terminal value
    terminal_value_local: float
    implied_ev_ebitda: float
    forecast: pd.DataFrame      # year-by-year table


def _compute(a: Assumptions) -> DCFResult:
    if a.wacc <= a.terminal_g:
        raise ValueError(
            f"WACC ({a.wacc:.2%}) must exceed terminal_g ({a.terminal_g:.2%}); "
            "terminal value would be undefined."
        )

    # Growth schedule: linearly fades from starting rate to terminal_g
    growth_rates = np.linspace(a.revenue_growth, a.terminal_g, a.forecast_years)

    # CapEx% fades to D&A% by the terminal year so that in steady state
    # net reinvestment (CapEx − D&A) → 0, consistent with low terminal growth.
    # If historical CapEx% is already ≤ D&A%, hold it constant (no upward fade).
    capex_terminal = a.da_pct
    if a.capex_pct > capex_terminal:
        capex_pcts = np.linspace(a.capex_pct, capex_terminal, a.forecast_years)
    else:
        capex_pcts = np.full(a.forecast_years, a.capex_pct)

    rows   = []
    prev_rev = a.start_revenue
    prev_nwc = a.start_revenue * a.nwc_pct   # NWC at t=0

    for t, (g, cp) in enumerate(zip(growth_rates, capex_pcts), start=1):
        rev   = prev_rev * (1 + g)
        ebit  = rev * a.ebit_margin
        nopat = ebit * (1 - a.tax_rate)      # EBIT×(1−tax); SBC already in EBIT (Route 1)
        da    = rev * a.da_pct
        capex = rev * cp
        nwc   = rev * a.nwc_pct
        dnwc  = nwc - prev_nwc               # +ve = cash consumed; –ve = cash released
        fcff  = nopat + da - capex - dnwc    # FCFF = NOPAT + D&A − CapEx − ΔNWC
        disc  = 1 / (1 + a.wacc) ** t
        pv    = fcff * disc

        rows.append({
            'Year':      t,
            'Growth':    g,
            'Revenue':   rev,
            'EBIT':      ebit,
            'NOPAT':     nopat,
            'D&A':       da,
            'CapEx%':    cp,
            'CapEx':     capex,
            'ΔNWC':      dnwc,
            'FCFF':      fcff,
            'Disc.':     disc,
            'PV(FCFF)':  pv,
        })

        prev_rev = rev
        prev_nwc = nwc

    forecast   = pd.DataFrame(rows).set_index('Year')
    pv_explicit = float(forecast['PV(FCFF)'].sum())

    # Terminal value: Gordon growth on last year's FCFF
    terminal_fcff = float(forecast['FCFF'].iloc[-1])
    tv    = terminal_fcff * (1 + a.terminal_g) / (a.wacc - a.terminal_g)
    pv_tv = tv / (1 + a.wacc) ** a.forecast_years

    ev             = pv_explicit + pv_tv
    equity_value   = ev - a.net_debt
    vps_local      = equity_value / a.diluted_shares
    vps_usd        = vps_local * a.fx_rate

    implied_ev_ebitda = (ev / a.current_ebitda
                         if a.current_ebitda and a.current_ebitda > 0
                         else float('nan'))

    return DCFResult(
        value_per_share_usd=vps_usd,
        value_per_share_local=vps_local,
        equity_value_local=equity_value,
        ev_local=ev,
        pv_explicit=pv_explicit,
        pv_tv=pv_tv,
        terminal_value_local=tv,
        implied_ev_ebitda=implied_ev_ebitda,
        forecast=forecast,
    )


def value(a: Assumptions) -> float:
    """Pure, fast path — returns intrinsic value per share in USD.
    No I/O. Safe to call in a Monte Carlo loop."""
    return _compute(a).value_per_share_usd


def detailed_value(a: Assumptions) -> DCFResult:
    """Full computation including forecast DataFrame and cross-checks."""
    return _compute(a)
