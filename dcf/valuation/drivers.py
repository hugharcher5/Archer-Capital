"""Compute base-case DCF drivers from historical financials."""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .data import RawData

# ── Plausibility bands ────────────────────────────────────────────────────────
_BANDS = {
    'revenue_growth':  (-0.50,  0.80),
    'ebit_margin':     (-0.30,  0.60),
    'tax_rate':        ( 0.00,  0.60),
    'da_pct':          ( 0.00,  0.30),
    'capex_pct':       ( 0.00,  0.50),
    'nwc_pct':         (-0.30,  0.50),
    'sbc_pct':         ( 0.00,  0.30),
}


@dataclass
class Drivers:
    # Base-case values (mean of available history)
    revenue_growth: float
    ebit_margin: float
    tax_rate: float
    da_pct: float
    capex_pct: float
    nwc_pct: float
    sbc_pct: float

    # Best (peak) historical EBIT margin — anchor for target_margin
    best_ebit_margin: float

    # Standard deviations — for Monte Carlo later
    std_revenue_growth: float
    std_ebit_margin: float
    std_target_margin: float
    std_fcf_pct: float         # std of historical (EBIT×(1−t) + D&A − CapEx) / Revenue

    years_used: int

    # Historical year-by-year DataFrame for display
    hist_df: pd.DataFrame = None


def compute_drivers(raw: RawData) -> Drivers:
    # ── Align all income-stmt + cashflow series ───────────────────────────────
    series = {
        'revenue': raw.revenue,
        'ebit':    raw.ebit,
        'pretax':  raw.pretax_income,
        'tax':     raw.tax_provision,
        'da':      raw.da,
        'capex':   raw.capex,
        'sbc':     raw.sbc if not raw.sbc.empty
                   else pd.Series(0.0, index=raw.revenue.index),
    }
    df = pd.concat(series, axis=1).dropna(subset=['revenue', 'ebit']).sort_index()

    if len(df) < 2:
        raise ValueError(
            f"Need ≥2 years of financial history for {raw.ticker}; "
            f"got {len(df)}. Check yfinance data availability."
        )

    # ── Year-over-year drivers ────────────────────────────────────────────────
    df['revenue_growth'] = df['revenue'].pct_change()
    df['ebit_margin']    = df['ebit']  / df['revenue']
    df['tax_rate']       = (df['tax']  / df['pretax']).clip(0.0, 0.99)
    df['da_pct']         = df['da']    / df['revenue']
    df['capex_pct']      = df['capex'] / df['revenue']
    df['sbc_pct']        = df['sbc']   / df['revenue']

    # NWC from most-recent balance sheet (single observation, applied uniformly)
    nwc_latest = (raw.current_assets - raw.cash) - (raw.current_liabilities - raw.current_debt)
    rev_latest  = float(df['revenue'].iloc[-1])
    nwc_pct = nwc_latest / rev_latest if rev_latest != 0 else 0.0

    # Drop first row (NaN growth from pct_change)
    df = df.dropna(subset=['revenue_growth'])
    years_used = len(df)

    rev_growth   = float(df['revenue_growth'].mean())
    ebit_margin  = float(df['ebit_margin'].mean())
    best_margin  = float(df['ebit_margin'].max())
    tax_rate     = float(df['tax_rate'].mean())
    da_pct       = float(df['da_pct'].mean())
    capex_pct    = float(df['capex_pct'].mean())
    sbc_pct      = float(df['sbc_pct'].mean())
    std_growth   = float(df['revenue_growth'].std())
    std_margin   = float(df['ebit_margin'].std())

    df['fcf_pct'] = (df['ebit'] * (1.0 - df['tax_rate']) + df['da'] - df['capex']) / df['revenue']
    _fcf_clean    = df['fcf_pct'].dropna()
    std_fcf_pct   = float(_fcf_clean.std()) if len(_fcf_clean) >= 2 else 0.0

    # ── Print historical table ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  HISTORICAL DRIVERS — {raw.ticker}  ({years_used} years used)")
    print(f"{'='*60}")

    col_labels = [str(d.year) for d in df.index]
    header = f"  {'Metric':<18}" + "".join(f"  {y:>8}" for y in col_labels)
    print(f"\n{header}")
    print("  " + "─" * (len(header) - 2))

    def _row(label, col, fmt):
        vals = "".join(f"  {fmt(v):>8}" for v in df[col])
        print(f"  {label:<18}{vals}")

    _row("Revenue (B)",    'revenue',      lambda v: f"{v/1e9:.1f}B")
    _row("EBIT (B)",       'ebit',         lambda v: f"{v/1e9:.1f}B")
    _row("Revenue growth", 'revenue_growth',lambda v: f"{v:.1%}")
    _row("EBIT margin",    'ebit_margin',   lambda v: f"{v:.1%}")
    _row("Tax rate",       'tax_rate',      lambda v: f"{v:.1%}")
    _row("D&A %",          'da_pct',        lambda v: f"{v:.1%}")
    _row("CapEx %",        'capex_pct',     lambda v: f"{v:.1%}")
    _row("SBC %",          'sbc_pct',       lambda v: f"{v:.1%}")

    na_cells = "".join(f"  {'n/a':>8}" for _ in range(years_used - 1))
    print(f"  {'NWC %':<18}{na_cells}  {nwc_pct:>7.1%}   ← most recent BS only")

    print(f"\n  {'─'*42}")
    print(f"  Base-case drivers (mean of {years_used} years):")
    print(f"  {'─'*42}")
    print(f"  Revenue growth  : {rev_growth:>7.2%}   σ = {std_growth:.2%}")
    print(f"  EBIT margin     : {ebit_margin:>7.2%}   σ = {std_margin:.2%}   (best historical: {best_margin:.2%})")
    print(f"  Target margin σ : {std_margin:>7.2%}   (= EBIT margin σ; drives target-margin PERT)")
    print(f"  FCF margin σ    : {std_fcf_pct:>7.2%}   (EBIT×(1−t)+D&A−CapEx)/Revenue")
    print(f"  Tax rate        : {tax_rate:>7.2%}")
    print(f"  D&A / Revenue   : {da_pct:>7.2%}")
    print(f"  CapEx / Revenue : {capex_pct:>7.2%}")
    print(f"  NWC / Revenue   : {nwc_pct:>7.2%}   (from most recent balance sheet)")
    print(f"  SBC / Revenue   : {sbc_pct:>7.2%}   [Route 1: already in EBIT — not added back]")

    drivers = Drivers(
        revenue_growth=rev_growth,
        ebit_margin=ebit_margin,
        best_ebit_margin=best_margin,
        tax_rate=max(0.0, min(tax_rate, 0.6)),
        da_pct=da_pct,
        capex_pct=capex_pct,
        nwc_pct=nwc_pct,
        sbc_pct=sbc_pct,
        std_revenue_growth=std_growth,
        std_ebit_margin=std_margin,
        std_target_margin=std_margin,
        std_fcf_pct=std_fcf_pct,
        years_used=years_used,
        hist_df=df,
    )

    # ── Plausibility checks ───────────────────────────────────────────────────
    flags = []
    for name, val in {
        'revenue_growth': rev_growth,
        'ebit_margin':    ebit_margin,
        'tax_rate':       tax_rate,
        'da_pct':         da_pct,
        'capex_pct':      capex_pct,
        'nwc_pct':        nwc_pct,
        'sbc_pct':        sbc_pct,
    }.items():
        lo, hi = _BANDS[name]
        if not (lo <= val <= hi):
            flags.append(f"  ⚠  {name} = {val:.2%}  outside expected [{lo:.0%}, {hi:.0%}]")

    print()
    if flags:
        print("  PLAUSIBILITY FLAGS:")
        for f in flags:
            print(f)
    else:
        print("  Plausibility: all drivers within expected ranges ✓")

    return drivers
