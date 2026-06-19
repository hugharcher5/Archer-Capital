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
MATURE_MARGIN_DEFAULT: float = 0.20    # floor for target EBIT margin at maturity

# Size-dependent revenue growth ceiling: g_ceiling(R) = max(terminal_g, G_CEILING_A / R**G_CEILING_B)
# R = projected revenue in USD billions (local currency × current spot FX rate / 1e9).
# Prevents economically impossible compound growth at scale: a company already generating
# $1T of revenue cannot sustain 50%/yr growth; the ceiling enforces that achievable rates
# fall as absolute size increases, applied year-by-year as revenue compounds within each path.
# Calibrated to two empirical anchors:
#   R ≈ $250 B  →  ceiling ≈ 25 %   (large but still high-growth company)
#   R ≈ $1 T    →  ceiling ≈  6 %   (mega-cap steady-state territory)
# Two-point solve: A / 250^b = 0.25,  A / 1000^b = 0.06
#   → b = ln(25/6) / ln(4) ≈ 1.029
#   → A = 0.25 × 250^1.029 ≈ 74.0
G_CEILING_A: float = 74.0    # ceiling numerator  (R in USD billions)
G_CEILING_B: float = 1.029   # ceiling exponent   (≈1 → ceiling roughly halves as revenue doubles)


@dataclass
class Assumptions:
    ticker: str
    currency: str               # reporting currency label

    # ── Core DCF inputs (the only things value() needs) ───────────────────────
    start_revenue: float        # most recent annual revenue in reporting currency
    revenue_growth: float       # starting growth rate (decimal, e.g. 0.12)
    ebit_margin: float          # starting EBIT margin (yr 1 of forecast)
    target_margin: float        # mature EBIT margin (fades linearly to this by yr N)
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

    # ── Per-year FX path (Monte Carlo only) ───────────────────────────────────
    # When set, _compute() translates each forecast year's discounted FCF at
    # fx_path[t-1] and the terminal value at fx_path[-1].  fx_rate is then used
    # only for net_debt (current balance-sheet item → terminal-year rate here).
    # None → single-rate: vps_usd = vps_local × fx_rate  (base-case / display).
    fx_path: object = None   # np.ndarray shape (forecast_years,) | None


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
    ceiling_hits: int = 0       # number of forecast years where size ceiling was binding


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

    # EBIT margin fades from starting margin to target (mature) margin.
    margin_path = np.linspace(a.ebit_margin, a.target_margin, a.forecast_years)

    rows           = []
    n_ceiling_hits = 0
    prev_rev = a.start_revenue
    prev_nwc = a.start_revenue * a.nwc_pct   # NWC at t=0

    for t, (g, cp, m) in enumerate(zip(growth_rates, capex_pcts, margin_path), start=1):
        # Size-dependent growth ceiling: cap g before applying to revenue.
        # R_usd_b = last year's revenue in USD billions; ceiling = max(terminal_g, A/R^b).
        # Guards: skip if fx_rate is zero (should never happen) or revenue is zero/negative.
        if a.fx_rate > 0 and prev_rev > 0:
            r_usd_b = prev_rev * a.fx_rate / 1e9
            g_ceil  = max(a.terminal_g, G_CEILING_A / (r_usd_b ** G_CEILING_B))
            if g > g_ceil:
                n_ceiling_hits += 1
                g = g_ceil

        rev   = prev_rev * (1 + g)
        ebit  = rev * m
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
            'Margin%':   m,
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

    ev           = pv_explicit + pv_tv
    equity_value = ev - a.net_debt
    vps_local    = equity_value / a.diluted_shares

    if a.fx_path is not None:
        # Per-year FX: each year's discounted FCF translated at its own path rate.
        # Terminal value translated at the final-year rate (path endpoint).
        # Net debt translated at the terminal-year rate (matches old single-shock
        # semantics — both EV and net debt at the same year-N FX rate).
        fx_arr  = np.asarray(a.fx_path[:a.forecast_years], dtype=float)
        fx_N    = float(fx_arr[-1])
        pv_expl_usd = float(
            (forecast['FCFF'].values * forecast['Disc.'].values * fx_arr).sum()
        )
        vps_usd = (pv_expl_usd + pv_tv * fx_N - a.net_debt * fx_N) / a.diluted_shares
    else:
        vps_usd = vps_local * a.fx_rate

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
        ceiling_hits=n_ceiling_hits,
    )


def value(a: Assumptions) -> float:
    """Pure, fast path — returns intrinsic value per share in USD.
    No I/O. Safe to call in a Monte Carlo loop."""
    return _compute(a).value_per_share_usd


def value_and_ceiling_hits(a: Assumptions) -> tuple[float, int]:
    """Returns (vps_usd, ceiling_hits) — used by Monte Carlo _collect path to
    track how many forecast years the size ceiling was binding in each draw."""
    r = _compute(a)
    return r.value_per_share_usd, r.ceiling_hits


def detailed_value(a: Assumptions) -> DCFResult:
    """Full computation including forecast DataFrame and cross-checks."""
    return _compute(a)
