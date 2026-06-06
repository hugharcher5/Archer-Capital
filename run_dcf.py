"""
Top-level DCF runner.

Usage:
    venv/bin/python run_dcf.py NVDA
"""

from __future__ import annotations
import sys
import math
from valuation.data    import fetch_raw
from valuation.drivers import compute_drivers
from valuation.wacc    import compute_wacc
from valuation.dcf     import (
    Assumptions, detailed_value,
    TERMINAL_G, HIGH_GROWTH_THRESHOLD,
)


# ── Assembly ──────────────────────────────────────────────────────────────────

def _assemble(raw, drivers, wacc_r) -> Assumptions:
    start_rev      = float(raw.revenue.iloc[-1])
    net_debt       = raw.total_debt - raw.cash
    forecast_years = 10 if drivers.revenue_growth > HIGH_GROWTH_THRESHOLD else 5

    ebit_last  = float(raw.ebit.iloc[-1]) if not raw.ebit.empty else 0.0
    da_last    = float(raw.da.iloc[-1])   if not raw.da.empty   else 0.0
    cur_ebitda = ebit_last + da_last

    return Assumptions(
        ticker=raw.ticker,
        currency=raw.currency,
        start_revenue=start_rev,
        revenue_growth=drivers.revenue_growth,
        ebit_margin=drivers.ebit_margin,
        tax_rate=drivers.tax_rate,
        da_pct=drivers.da_pct,
        capex_pct=drivers.capex_pct,
        nwc_pct=drivers.nwc_pct,
        terminal_g=TERMINAL_G,
        wacc=wacc_r.wacc,
        net_debt=net_debt,
        diluted_shares=raw.diluted_shares,
        fx_rate=raw.fx_rate,
        forecast_years=forecast_years,
        rf=wacc_r.rf,
        erp=wacc_r.erp,
        beta_adj=wacc_r.beta_adj,
        cost_of_equity=wacc_r.cost_of_equity,
        cost_of_debt_pretax=wacc_r.cost_of_debt_pretax,
        cost_of_debt_aftertax=wacc_r.cost_of_debt_aftertax,
        equity_weight=wacc_r.equity_weight,
        debt_weight=wacc_r.debt_weight,
        implied_rating=wacc_r.implied_rating,
        current_price_usd=raw.current_price_usd,
        current_ebitda=cur_ebitda,
    )


# ── Print helpers ─────────────────────────────────────────────────────────────

def _sec(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_assumptions(a: Assumptions):
    _sec(f"BASE-CASE ASSUMPTIONS — {a.ticker}")
    ccy = a.currency
    print(f"\n  Reporting currency  : {ccy}"
          + (f"  (FX: 1 {ccy} = {a.fx_rate:.6f} USD)" if a.currency != 'USD' else ''))
    print(f"  Forecast horizon    : {a.forecast_years} years"
          f"  [{'≥15% growth → 10yr' if a.forecast_years == 10 else '<15% growth → 5yr'}]")
    print()
    print(f"  Starting revenue    : {ccy} {a.start_revenue/1e9:.3f}B   (most recent fiscal year)")
    print(f"  Revenue growth yr 1 : {a.revenue_growth:.2%}   → fades linearly to {a.terminal_g:.2%} by yr {a.forecast_years}")
    print(f"  EBIT margin         : {a.ebit_margin:.2%}")
    print(f"  Tax rate            : {a.tax_rate:.2%}")
    print(f"  D&A / Revenue       : {a.da_pct:.2%}")
    print(f"  CapEx / Revenue     : {a.capex_pct:.2%}")
    print(f"  NWC / Revenue       : {a.nwc_pct:.2%}")
    print(f"  Terminal growth (g) : {a.terminal_g:.2%}")
    print(f"  WACC                : {a.wacc:.3%}")
    print()
    print(f"  Net debt            : {ccy} {a.net_debt/1e9:.3f}B   (total debt − cash)")
    print(f"  Diluted shares      : {a.diluted_shares/1e9:.3f}B")
    print()
    print(f"  SBC treatment       : Route 1 — SBC is expensed in EBIT; not added back to FCFF.")
    print(f"                        Current diluted shares used; no future SBC dilution modelled.")


def _print_forecast(a: Assumptions, r) -> None:
    _sec(f"YEAR-BY-YEAR FCFF FORECAST  ({a.currency}B)")
    df  = r.forecast
    ccy = a.currency

    hdr = (f"\n  {'Yr':>3}  {'Growth':>7}  {'Revenue':>9}  {'EBIT':>8}  "
           f"{'NOPAT':>8}  {'D&A':>7}  {'CapEx':>7}  {'ΔNWC':>7}  "
           f"{'FCFF':>8}  {'Disc.':>7}  {'PV(FCFF)':>9}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 3))

    for yr, row in df.iterrows():
        print(
            f"  {yr:>3}  "
            f"{row['Growth']:>7.2%}  "
            f"{row['Revenue']/1e9:>9.3f}  "
            f"{row['EBIT']/1e9:>8.3f}  "
            f"{row['NOPAT']/1e9:>8.3f}  "
            f"{row['D&A']/1e9:>7.3f}  "
            f"{row['CapEx']/1e9:>7.3f}  "
            f"{row['ΔNWC']/1e9:>7.3f}  "
            f"{row['FCFF']/1e9:>8.3f}  "
            f"{row['Disc.']:>7.4f}  "
            f"{row['PV(FCFF)']/1e9:>9.3f}"
        )

    print(f"\n  PV of explicit FCFFs : {ccy} {r.pv_explicit/1e9:.3f}B")
    print()
    print(f"  Terminal FCFF (yr {a.forecast_years})  : {ccy} {r.forecast['FCFF'].iloc[-1]/1e9:.3f}B")
    print(f"  Terminal growth (g)  : {a.terminal_g:.2%}")
    print(f"  WACC                 : {a.wacc:.3%}")
    print(f"  Terminal value (TV)  : {ccy} {r.terminal_value_local/1e9:.3f}B"
          f"  [= FCFF_N × (1+g) / (WACC−g)]")
    print(f"  PV of TV             : {ccy} {r.pv_tv/1e9:.3f}B"
          f"  ({r.pv_tv / r.ev_local:.1%} of EV)")

    if any(row['FCFF'] < 0 for _, row in df.iterrows()):
        print(f"\n  ⚠  One or more forecast years have negative FCFF — review assumptions.")


def _print_crosscheck(a: Assumptions, r) -> None:
    _sec("EV / EBITDA CROSS-CHECK")
    ccy = a.currency
    print(f"\n  Current EBITDA (last fiscal yr) : {ccy} {a.current_ebitda/1e9:.3f}B")
    print(f"  Implied EV (from DCF)           : {ccy} {r.ev_local/1e9:.3f}B")
    if not math.isnan(r.implied_ev_ebitda):
        mult = r.implied_ev_ebitda
        print(f"  Implied EV / EBITDA             : {mult:.1f}×")
        if mult < 5:
            print(f"  ⚠  < 5× — assumptions may be too conservative (high WACC or low margins).")
        elif mult > 30:
            print(f"  ⚠  > 30× — assumptions may be too aggressive (high growth or low WACC).")
        else:
            print(f"  ✓  Within 5–30× — reasonable range.")
    else:
        print(f"  Implied EV / EBITDA             : n/a (EBITDA ≤ 0)")


def _print_bridge(a: Assumptions, r) -> None:
    _sec("EV → EQUITY BRIDGE")
    ccy = a.currency
    print(f"\n  Enterprise value (EV)     : {ccy} {r.ev_local/1e9:>10.3f}B")
    nd_sign = "−" if a.net_debt >= 0 else "+"
    print(f"  {nd_sign} Net debt                : {ccy} {abs(a.net_debt)/1e9:>10.3f}B"
          + ("  [debt-free / net cash]" if a.net_debt < 0 else ""))
    print(f"  {'─'*44}")
    print(f"  = Equity value            : {ccy} {r.equity_value_local/1e9:>10.3f}B")
    print(f"  ÷ Diluted shares          :     {a.diluted_shares/1e9:>10.3f}B")
    print(f"  {'─'*44}")
    if a.currency != 'USD':
        print(f"  = Value / share ({ccy:3s})    : {ccy} {r.value_per_share_local:>10.2f}")
        print(f"  × FX rate ({ccy}→USD)     :     {a.fx_rate:>10.6f}")
        print(f"  {'─'*44}")
    print(f"  = Value / share (USD)     : USD {r.value_per_share_usd:>10.2f}")


def _print_verdict(a: Assumptions, r) -> None:
    _sec(f"VALUATION VERDICT — {a.ticker}")
    intrinsic = r.value_per_share_usd
    market    = a.current_price_usd
    pct       = (intrinsic - market) / market * 100 if market > 0 else float('nan')
    direction = "UNDERVALUED" if intrinsic > market else "OVERVALUED"

    print(f"\n  Intrinsic value / share : USD {intrinsic:>8.2f}")
    print(f"  Current market price    : USD {market:>8.2f}")
    print(f"  {'─'*38}")
    if not math.isnan(pct):
        print(f"  Base case: {direction} by {abs(pct):.1f}%")
    print()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_dcf(ticker: str) -> None:
    print(f"\n{'#'*60}")
    print(f"  DCF VALUATION — {ticker.upper()}")
    print(f"{'#'*60}")

    raw      = fetch_raw(ticker)
    drivers  = compute_drivers(raw)
    wacc_r   = compute_wacc(raw, drivers)
    assum    = _assemble(raw, drivers, wacc_r)

    _print_assumptions(assum)

    result = detailed_value(assum)

    _print_forecast(assum, result)
    _print_crosscheck(assum, result)
    _print_bridge(assum, result)
    _print_verdict(assum, result)


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    run_dcf(ticker)
