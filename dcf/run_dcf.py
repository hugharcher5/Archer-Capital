"""
Top-level DCF runner.

Usage:
    venv/bin/python run_dcf.py NVDA
"""

from __future__ import annotations
import sys
import math
from valuation.montecarlo import run_valuation
from valuation.result     import ValuationResult
from valuation.dcf        import TERMINAL_G, HIGH_GROWTH_THRESHOLD, MATURE_MARGIN_DEFAULT


# ── Print helpers ─────────────────────────────────────────────────────────────

def _sec(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _print_assumptions(r: ValuationResult) -> None:
    a = r.assumptions
    _sec(f"BASE-CASE ASSUMPTIONS — {a.ticker}")
    ccy = a.currency
    print(f"\n  Reporting currency  : {ccy}"
          + (f"  (FX: 1 {ccy} = {a.fx_rate:.6f} USD)" if a.currency != 'USD' else ''))
    print(f"  Forecast horizon    : {a.forecast_years} years"
          f"  [{'≥15% growth → 10yr' if a.forecast_years == 10 else '<15% growth → 5yr'}]")
    print()
    print(f"  Starting revenue    : {ccy} {a.start_revenue/1e9:.3f}B   (most recent fiscal year)")
    print(f"  Revenue growth yr 1 : {a.revenue_growth:.2%}   → fades linearly to {a.terminal_g:.2%} by yr {a.forecast_years}")
    print(f"  EBIT margin (start) : {a.ebit_margin:.2%}   → fades linearly to {a.target_margin:.2%} (target) by yr {a.forecast_years}")
    print(f"  Target EBIT margin  : {a.target_margin:.2%}   [= max(best historical, {MATURE_MARGIN_DEFAULT:.0%} default)]")
    print(f"  Tax rate            : {a.tax_rate:.2%}")
    capex_end = a.da_pct if a.capex_pct > a.da_pct else a.capex_pct
    fade_note = f"→ fades to {capex_end:.2%} (= D&A%) by yr {a.forecast_years}" if a.capex_pct > a.da_pct else "no fade needed (≤ D&A%)"
    print(f"  D&A / Revenue       : {a.da_pct:.2%}   (held constant)")
    print(f"  CapEx / Revenue     : {a.capex_pct:.2%}   {fade_note}")
    print(f"  NWC / Revenue       : {a.nwc_pct:.2%}")
    print(f"  Terminal growth (g) : {a.terminal_g:.2%}")
    print(f"  WACC                : {a.wacc:.3%}")
    print()
    print(f"  Net debt            : {ccy} {a.net_debt/1e9:.3f}B   (total debt − cash)")
    print(f"  Diluted shares      : {a.diluted_shares/1e9:.3f}B")
    print()
    print(f"  SBC treatment       : Route 1 — SBC is expensed in EBIT; not added back to FCFF.")
    print(f"                        Current diluted shares used; no future SBC dilution modelled.")


def _print_forecast(r: ValuationResult) -> None:
    a   = r.assumptions
    res = r.dcf
    _sec(f"YEAR-BY-YEAR FCFF FORECAST  ({a.currency}B)")
    df  = res.forecast
    ccy = a.currency

    capex_start  = df['CapEx%'].iloc[0]
    capex_end    = df['CapEx%'].iloc[-1]
    margin_start = df['Margin%'].iloc[0]
    margin_end   = df['Margin%'].iloc[-1]
    da_pct       = a.da_pct

    if abs(capex_start - capex_end) > 0.001:
        print(f"\n  CapEx% fades {capex_start:.2%} → {capex_end:.2%} (= D&A% {da_pct:.2%}) — maintenance reinvestment in terminal year")
    else:
        print(f"\n  CapEx% held constant at {capex_start:.2%} (already ≤ D&A% {da_pct:.2%})")
    if abs(margin_start - margin_end) > 0.001:
        print(f"  EBIT margin fades {margin_start:.2%} → {margin_end:.2%} by yr {a.forecast_years}")
    else:
        print(f"  EBIT margin held constant at {margin_start:.2%}")

    hdr = (f"\n  {'Yr':>3}  {'Growth':>7}  {'Marg%':>6}  {'Revenue':>9}  {'EBIT':>8}  "
           f"{'NOPAT':>8}  {'D&A':>7}  {'CapEx%':>7}  {'CapEx':>8}  {'ΔNWC':>7}  "
           f"{'FCFF':>8}  {'Disc.':>7}  {'PV(FCFF)':>9}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 3))

    for yr, row in df.iterrows():
        print(
            f"  {yr:>3}  "
            f"{row['Growth']:>7.2%}  "
            f"{row['Margin%']:>6.2%}  "
            f"{row['Revenue']/1e9:>9.3f}  "
            f"{row['EBIT']/1e9:>8.3f}  "
            f"{row['NOPAT']/1e9:>8.3f}  "
            f"{row['D&A']/1e9:>7.3f}  "
            f"{row['CapEx%']:>7.2%}  "
            f"{row['CapEx']/1e9:>8.3f}  "
            f"{row['ΔNWC']/1e9:>7.3f}  "
            f"{row['FCFF']/1e9:>8.3f}  "
            f"{row['Disc.']:>7.4f}  "
            f"{row['PV(FCFF)']/1e9:>9.3f}"
        )

    print(f"\n  PV of explicit FCFFs : {ccy} {res.pv_explicit/1e9:.3f}B")
    print()
    print(f"  Terminal FCFF (yr {a.forecast_years})  : {ccy} {res.forecast['FCFF'].iloc[-1]/1e9:.3f}B")
    print(f"  Terminal growth (g)  : {a.terminal_g:.2%}")
    print(f"  WACC                 : {a.wacc:.3%}")
    print(f"  Terminal value (TV)  : {ccy} {res.terminal_value_local/1e9:.3f}B"
          f"  [= FCFF_N × (1+g) / (WACC−g)]")
    print(f"  PV of TV             : {ccy} {res.pv_tv/1e9:.3f}B"
          f"  ({res.pv_tv / res.ev_local:.1%} of EV)")

    if any(row['FCFF'] < 0 for _, row in df.iterrows()):
        print(f"\n  ⚠  One or more forecast years have negative FCFF — review assumptions.")


def _print_crosscheck(r: ValuationResult) -> None:
    a   = r.assumptions
    res = r.dcf
    _sec("EV / EBITDA CROSS-CHECK")
    ccy    = a.currency
    ebitda = a.current_ebitda

    mkt_multiple = (a.market_ev_local / ebitda
                    if ebitda and ebitda > 0 else float('nan'))
    dcf_multiple = res.implied_ev_ebitda

    print(f"\n  Current EBITDA (last fiscal yr)  : {ccy} {ebitda/1e9:.3f}B")
    print()
    print(f"  Market EV (mktcap + net debt)    : {ccy} {a.market_ev_local/1e9:.3f}B")
    if not math.isnan(mkt_multiple):
        print(f"  Market EV / EBITDA               : {mkt_multiple:.1f}×")
    else:
        print(f"  Market EV / EBITDA               : n/a")
    print()
    print(f"  DCF implied EV                   : {ccy} {res.ev_local/1e9:.3f}B")
    if not math.isnan(dcf_multiple):
        print(f"  DCF implied EV / EBITDA          : {dcf_multiple:.1f}×")
    else:
        print(f"  DCF implied EV / EBITDA          : n/a (EBITDA ≤ 0)")

    if not math.isnan(mkt_multiple) and not math.isnan(dcf_multiple) and mkt_multiple > 0:
        diff_x   = dcf_multiple - mkt_multiple
        diff_pct = diff_x / mkt_multiple * 100
        sign     = "premium" if diff_x > 0 else "discount"
        print(f"\n  DCF vs market: {abs(diff_x):.1f}× {sign}  ({abs(diff_pct):.0f}% {'above' if diff_x > 0 else 'below'} market multiple)")
        if abs(diff_pct) > 50:
            direction = "optimistic" if diff_x > 0 else "conservative"
            print(f"  ⚠  Gap > 50% — DCF assumptions appear significantly {direction} vs market pricing.")
        elif abs(diff_pct) > 25:
            print(f"  ~  Gap 25–50% — moderate divergence from market; worth reviewing key assumptions.")
        else:
            print(f"  ✓  Gap < 25% — DCF and market multiples broadly consistent.")


def _print_bridge(r: ValuationResult) -> None:
    a   = r.assumptions
    res = r.dcf
    _sec("EV → EQUITY BRIDGE")
    ccy = a.currency
    print(f"\n  Enterprise value (EV)     : {ccy} {res.ev_local/1e9:>10.3f}B")
    nd_sign = "−" if a.net_debt >= 0 else "+"
    print(f"  {nd_sign} Net debt                : {ccy} {abs(a.net_debt)/1e9:>10.3f}B"
          + ("  [debt-free / net cash]" if a.net_debt < 0 else ""))
    print(f"  {'─'*44}")
    print(f"  = Equity value            : {ccy} {res.equity_value_local/1e9:>10.3f}B")
    print(f"  ÷ Diluted shares          :     {a.diluted_shares/1e9:>10.3f}B")
    print(f"  {'─'*44}")
    if a.currency != 'USD':
        print(f"  = Value / share ({ccy:3s})    : {ccy} {res.value_per_share_local:>10.2f}")
        print(f"  × FX rate ({ccy}→USD)     :     {a.fx_rate:>10.6f}")
        print(f"  {'─'*44}")
    print(f"  = Value / share (USD)     : USD {res.value_per_share_usd:>10.2f}")


def _print_verdict(r: ValuationResult) -> None:
    a        = r.assumptions
    res      = r.dcf
    _sec(f"VALUATION VERDICT — {a.ticker}")
    intrinsic = res.value_per_share_usd
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

    r = run_valuation(ticker, n_sims=0)

    _print_assumptions(r)
    _print_forecast(r)
    _print_crosscheck(r)
    _print_bridge(r)
    _print_verdict(r)


if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
    run_dcf(ticker)
