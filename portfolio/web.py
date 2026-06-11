import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
import streamlit as st

from portfolio.data          import fetch_portfolio
from valuation.montecarlo    import run_valuation
from valuation.result        import ValuationResult
from valuation.dcf           import MATURE_MARGIN_DEFAULT
from valuation.sources       import fetch_yahoo, fetch_fmp, fetch_edgar
from valuation.reconcile     import reconcile

st.set_page_config(page_title="Archer Capital", layout="wide")
st.title("Archer Capital")

tab_portfolio, tab_valuation = st.tabs(["Portfolio", "Valuation"])


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO TAB  (unchanged behaviour)
# ══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    @st.cache_data(ttl=60)
    def get_portfolio():
        return fetch_portfolio()

    rows, totals = get_portfolio()

    df = pd.DataFrame(rows)

    display = pd.DataFrame({
        "Ticker":     df["ticker"],
        "Last":       df["last"].map("${:,.2f}".format),
        "Day %":      df["daily_pct"].map("{:+.2f}%".format),
        "Mkt Value":  df["market_value"].map("${:,.0f}".format),
        "Cost Basis": df["cost_basis"].map("${:,.0f}".format),
        "Unreal P&L": df["unreal_pnl"].map("${:+,.0f}".format),
        "Day P&L":    df["day_pnl"].map("${:+,.0f}".format),
        "Weight":     df["weight"].map("{:.1f}%".format),
    })

    totals_row = pd.DataFrame([{
        "Ticker":     "TOTAL",
        "Last":       "",
        "Day %":      "",
        "Mkt Value":  f"${totals['market_value']:,.0f}",
        "Cost Basis": f"${totals['cost_basis']:,.0f}",
        "Unreal P&L": f"${totals['unreal_pnl']:+,.0f}",
        "Day P&L":    f"${totals['day_pnl']:+,.0f}",
        "Weight":     "100.0%",
    }])

    display = pd.concat([display, totals_row], ignore_index=True)

    def _color_pnl(val):
        if isinstance(val, str):
            if val.startswith("$+") or (val.startswith("+") and not val.startswith("+0")):
                return "color: #2ecc71"
            if val.startswith("$-") or val.startswith("-"):
                return "color: #e74c3c"
        return ""

    styled = display.style.map(_color_pnl, subset=["Unreal P&L", "Day P&L", "Day %"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Market Value",   f"${totals['market_value']:,.0f}")
    col2.metric("Unrealized P&L", f"${totals['unreal_pnl']:+,.0f}")
    col3.metric("Daily P&L",      f"${totals['day_pnl']:+,.0f}")

    st.subheader("Portfolio Weights")
    fig = px.pie(
        df, values="weight", names="ticker",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=350)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  VALUATION TAB
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_valuation(ticker: str) -> ValuationResult:
    sources = []
    for fetcher in (fetch_yahoo, fetch_fmp, fetch_edgar):
        try:
            sources.append(fetcher(ticker))
        except Exception:
            pass
    if len(sources) >= 2:
        recon = reconcile(sources)
        return run_valuation(ticker, n_sims=10_000,
                             sigma_cross=recon.sigma_cross,
                             reconcile_result=recon)
    return run_valuation(ticker, n_sims=10_000)


def _fmt_pct(v: float, decimals: int = 1) -> str:
    return f"{v:+.{decimals}f}%" if not math.isnan(v) else "n/a"


# ── KDE density chart ────────────────────────────────────────────────────────

def _render_histogram(r: ValuationResult) -> None:
    sims  = r.sims
    price = r.current_price_usd

    # Evaluation grid: P1–P99 (same clipping as the old histogram display).
    # KDE is fitted on the full array so the density is correct everywhere.
    p1, p99 = np.percentile(sims, [1, 99])
    x_grid  = np.linspace(p1, p99, 300)
    kde     = scipy.stats.gaussian_kde(sims)
    y_kde   = kde(x_grid)

    fig = go.Figure()

    # ── Full distribution — steelblue filled KDE ──────────────────────────────
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_kde,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(70, 130, 180, 0.25)",
        line=dict(color="steelblue", width=2),
        hovertemplate="$%{x:,.2f}<br>Density: %{y:.5f}<extra></extra>",
        name="KDE",
    ))

    # ── Undervalued region (x > market price) — green overlay ────────────────
    # Shows the P(undervalued) area visually without obscuring the KDE line.
    price_clipped = max(p1, min(p99, price))
    mask   = x_grid >= price_clipped
    if mask.any():
        x_over = np.concatenate([[price_clipped], x_grid[mask]])
        y_over = np.concatenate([kde([price_clipped]), y_kde[mask]])
        fig.add_trace(go.Scatter(
            x=x_over, y=y_over,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.22)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

    # ── Reference lines (unchanged) ───────────────────────────────────────────
    fig.add_vline(x=price,  line_color="crimson",    line_width=2,
                  annotation_text=f"Market  ${price:.2f}",
                  annotation_position="top right")
    fig.add_vline(x=r.p50,  line_color="seagreen",   line_width=2,   line_dash="dash",
                  annotation_text=f"P50  ${r.p50:.2f}",
                  annotation_position="top left")
    fig.add_vline(x=r.p10,  line_color="darkorange", line_width=1.5, line_dash="dot",
                  annotation_text=f"P10  ${r.p10:.2f}",
                  annotation_position="bottom right")
    fig.add_vline(x=r.p90,  line_color="darkorange", line_width=1.5, line_dash="dot",
                  annotation_text=f"P90  ${r.p90:.2f}",
                  annotation_position="top right")

    fig.update_layout(
        title=f"{r.ticker} — Monte Carlo DCF  ({r.n_valid:,} simulations, {r.copula_label} copula)",
        xaxis_title="Intrinsic value per share (USD)",
        yaxis_title="Probability density",
        showlegend=False,
        height=420,
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Expanders ─────────────────────────────────────────────────────────────────

def _recon_source_disagree(recon_fields: dict, field_key: str) -> tuple[str, str]:
    """Return (source_name, disagree_str) for a reconcile field, or ('—', '—') if absent."""
    if not recon_fields or field_key not in recon_fields:
        return "—", "—"
    info  = recon_fields[field_key]
    src   = info["source"]
    dpct  = info["disagree_pct"]
    ftype = info["field_type"]
    if dpct is None:
        return src, "n/a"
    if ftype == "DEFINITIONAL":
        return src, f"{dpct:.1f}% (conv)"
    return src, (f"⚠ {dpct:.1f}%" if dpct > 2.0 else f"{dpct:.1f}%")


def _render_expander_drivers(r: ValuationResult) -> None:
    d  = r.drivers
    rf = r.recon_fields
    with st.expander("Historical Drivers", expanded=False):
        st.markdown(
            "The model averages the last few years of reported financials to anchor "
            "the base-case assumptions.  Standard deviations set the Monte Carlo spread width."
        )
        if d.hist_df is not None:
            disp_cols = [c for c in
                         ["revenue_growth", "ebit_margin", "tax_rate", "da_pct", "capex_pct", "sbc_pct"]
                         if c in d.hist_df.columns]
            hist_display = d.hist_df[disp_cols].copy()
            hist_display.index = [str(i.year) for i in hist_display.index]
            hist_display.columns = ["Rev Growth", "EBIT Margin", "Tax Rate",
                                     "D&A %", "CapEx %", "SBC %"][:len(disp_cols)]
            st.dataframe(
                hist_display.style.format("{:.1%}"),
                use_container_width=True,
            )

        # Map driver label → reconcile field key for source/disagree annotation
        _DRIVER_FIELD = {
            "Revenue growth":  "revenue",
            "EBIT margin":     "ebit",
            "Tax rate":        "tax_rate",
            "D&A / Revenue":   "dep_amort",
            "CapEx / Revenue": "capex",
        }
        base_rows = [
            ("Revenue growth",   f"{d.revenue_growth:.2%}",   f"σ = {d.std_revenue_growth:.2%}"),
            ("EBIT margin",      f"{d.ebit_margin:.2%}",      f"σ = {d.std_ebit_margin:.2%}  (best: {d.best_ebit_margin:.2%})"),
            ("Tax rate",         f"{d.tax_rate:.2%}",         ""),
            ("D&A / Revenue",    f"{d.da_pct:.2%}",           ""),
            ("CapEx / Revenue",  f"{d.capex_pct:.2%}",        ""),
            ("NWC / Revenue",    f"{d.nwc_pct:.2%}",          "most recent balance sheet"),
            ("SBC / Revenue",    f"{d.sbc_pct:.2%}",          "Route 1: already in EBIT"),
        ]
        if rf:
            table_rows = []
            for driver, mean, note in base_rows:
                src, dis = _recon_source_disagree(rf, _DRIVER_FIELD.get(driver, ""))
                table_rows.append((driver, mean, note, src, dis))
            st.dataframe(
                pd.DataFrame(table_rows, columns=["Driver", "Mean", "Note", "Source", "Disagree"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.dataframe(
                pd.DataFrame(base_rows, columns=["Driver", "Mean", "Note"]),
                use_container_width=True, hide_index=True,
            )


def _render_expander_wacc(r: ValuationResult) -> None:
    w   = r.wacc_result
    ccy = r.currency
    rf  = r.recon_fields
    with st.expander("WACC", expanded=False):
        st.markdown(
            "WACC is the discount rate applied to each year's free cash flow.  "
            "Cost of equity comes from CAPM; cost of debt from a Damodaran synthetic rating "
            "based on the interest-coverage ratio."
        )
        st.latex(r"\text{WACC} = w_E \cdot K_E + w_D \cdot K_D (1-t)")

        sh_src, _ = _recon_source_disagree(rf, "diluted_shares")
        td_src, td_dis = _recon_source_disagree(rf, "total_debt")
        eq_note   = f"{ccy} {w.equity_value/1e9:.1f}B market cap"
        debt_note = f"{ccy} {w.debt_value/1e9:.1f}B total debt"
        if rf:
            eq_note   += f"  |  shares: {sh_src}"
            debt_note += f"  |  {td_src}, {td_dis}"

        rows = [
            ("Risk-free rate",         f"{w.rf:.3%}",                   w.rf_source),
            ("Beta (raw / Blume adj)", f"{w.beta_raw:.3f} / {w.beta_adj:.3f}", "0.67×raw + 0.33"),
            ("ERP",                    f"{w.erp:.2%}",                  "Damodaran implied"),
            ("Cost of equity (Kₑ)",    f"{w.cost_of_equity:.3%}",      "Rf + β×ERP"),
            ("Interest coverage",
             f"{w.coverage_ratio:.2f}×" if w.coverage_ratio != float('inf') else "∞",
             "EBIT / |interest expense|"),
            ("Implied rating",         w.implied_rating,               "Damodaran synthetic table"),
            ("Credit spread",          f"{w.credit_spread:.3%}",       ""),
            ("Cost of debt pre-tax",   f"{w.cost_of_debt_pretax:.3%}", "Rf + spread"),
            ("Cost of debt post-tax",  f"{w.cost_of_debt_aftertax:.3%}", "pre × (1−tax)"),
            ("Equity weight",          f"{w.equity_weight:.2%}",       eq_note),
            ("Debt weight",            f"{w.debt_weight:.2%}",         debt_note),
            ("WACC",                   f"{w.wacc:.3%}",                "weighted blend"),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Component", "Value", "Note"]),
            use_container_width=True, hide_index=True,
        )


def _render_expander_assumptions(r: ValuationResult) -> None:
    a   = r.assumptions
    ccy = a.currency
    with st.expander("Base-case Assumptions", expanded=False):
        st.markdown(
            "These are the inputs fed into every year of the forecast.  "
            "Revenue growth and EBIT margin fade linearly toward their terminal values; "
            "CapEx fades to D&A% so that steady-state net reinvestment approaches zero."
        )
        capex_end = a.da_pct if a.capex_pct > a.da_pct else a.capex_pct
        rows = [
            ("Ticker / currency",    f"{a.ticker} / {ccy}",       ""),
            ("Starting revenue",     f"{ccy} {a.start_revenue/1e9:.3f}B", "most recent fiscal year"),
            ("Revenue growth yr 1",  f"{a.revenue_growth:.2%}",   f"→ {a.terminal_g:.2%} by yr {a.forecast_years}"),
            ("EBIT margin (start)",  f"{a.ebit_margin:.2%}",      f"→ {a.target_margin:.2%} target by yr {a.forecast_years}"),
            ("Target EBIT margin",   f"{a.target_margin:.2%}",    f"max(best historical, {MATURE_MARGIN_DEFAULT:.0%} floor)"),
            ("Tax rate",             f"{a.tax_rate:.2%}",         ""),
            ("D&A / Revenue",        f"{a.da_pct:.2%}",           "held constant"),
            ("CapEx / Revenue",      f"{a.capex_pct:.2%}",        f"→ {capex_end:.2%} (= D&A%) by yr {a.forecast_years}"),
            ("NWC / Revenue",        f"{a.nwc_pct:.2%}",          ""),
            ("Terminal growth (g)",  f"{a.terminal_g:.2%}",       "Gordon Growth Model"),
            ("WACC",                 f"{a.wacc:.3%}",             ""),
            ("Forecast horizon",     f"{a.forecast_years} years", "10yr if growth >15%, else 5yr"),
            ("Net debt",             f"{ccy} {a.net_debt/1e9:.3f}B", "total debt − cash"),
            ("Diluted shares",       f"{a.diluted_shares/1e9:.3f}B", ""),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Assumption", "Value", "Note"]),
            use_container_width=True, hide_index=True,
        )


def _render_expander_forecast(r: ValuationResult) -> None:
    a  = r.assumptions
    df = r.dcf.forecast.copy()
    with st.expander("Year-by-Year FCFF Forecast", expanded=False):
        st.markdown(
            "FCFF (Free Cash Flow to the Firm) = NOPAT + D&A − CapEx − ΔNWC.  "
            "Each year's FCFF is discounted back to today using WACC.  "
            "The margin column shows the fade from the starting EBIT margin toward the "
            "target mature margin."
        )
        st.latex(r"\text{FCFF}_t = \text{NOPAT}_t + \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t")

        ccy = a.currency
        display_df = pd.DataFrame({
            "Yr":      df.index,
            "Growth":  df["Growth"].map("{:.1%}".format),
            "Marg%":   df["Margin%"].map("{:.1%}".format),
            "Revenue": (df["Revenue"] / 1e9).map("{:.3f}".format),
            "EBIT":    (df["EBIT"]    / 1e9).map("{:.3f}".format),
            "NOPAT":   (df["NOPAT"]   / 1e9).map("{:.3f}".format),
            "D&A":     (df["D&A"]     / 1e9).map("{:.3f}".format),
            "CapEx%":  df["CapEx%"].map("{:.1%}".format),
            "CapEx":   (df["CapEx"]   / 1e9).map("{:.3f}".format),
            "ΔNWC":    (df["ΔNWC"]    / 1e9).map("{:.3f}".format),
            "FCFF":    (df["FCFF"]    / 1e9).map("{:.3f}".format),
            "Disc.":   df["Disc."].map("{:.4f}".format),
            "PV(FCFF)":(df["PV(FCFF)"] / 1e9).map("{:.3f}".format),
        })
        st.caption(f"Monetary values in {ccy} billions.")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        res = r.dcf
        st.markdown(f"**PV of explicit FCFFs:** {ccy} {res.pv_explicit/1e9:.3f}B")


def _render_expander_terminal(r: ValuationResult) -> None:
    a   = r.assumptions
    res = r.dcf
    ccy = a.currency
    with st.expander("Terminal Value & EV/EBITDA Cross-check", expanded=False):
        st.markdown(
            "The terminal value (TV) captures all cash flows beyond the explicit forecast "
            "using the Gordon Growth Model.  It is then discounted back to today and added "
            "to the PV of explicit FCFFs to get the total enterprise value.  "
            "The EV/EBITDA cross-check compares the DCF-implied multiple to the market's "
            "current multiple as a sanity check."
        )
        st.latex(r"TV = \frac{FCFF_N \times (1+g)}{WACC - g}")

        last_fcff = float(res.forecast["FCFF"].iloc[-1])
        tv_pct    = res.pv_tv / res.ev_local if res.ev_local else float('nan')

        tv_rows = [
            ("Terminal FCFF (last forecast yr)", f"{ccy} {last_fcff/1e9:.3f}B", ""),
            ("Terminal growth (g)",              f"{a.terminal_g:.2%}",         ""),
            ("WACC",                             f"{a.wacc:.3%}",               ""),
            ("Terminal value (TV)",              f"{ccy} {res.terminal_value_local/1e9:.3f}B",
             "FCFF_N × (1+g) / (WACC−g)"),
            ("PV of TV",                         f"{ccy} {res.pv_tv/1e9:.3f}B",
             f"{tv_pct:.1%} of total EV"),
            ("PV of explicit FCFFs",             f"{ccy} {res.pv_explicit/1e9:.3f}B", ""),
            ("Enterprise value (EV)",            f"{ccy} {res.ev_local/1e9:.3f}B",    "PV(TV) + PV(FCFFs)"),
        ]
        st.dataframe(
            pd.DataFrame(tv_rows, columns=["Item", "Value", "Note"]),
            use_container_width=True, hide_index=True,
        )

        st.markdown("**EV / EBITDA cross-check**")
        ebitda = a.current_ebitda
        mkt_ev = a.market_ev_local
        mkt_mult = mkt_ev / ebitda if ebitda and ebitda > 0 else float('nan')
        dcf_mult = res.implied_ev_ebitda

        xc_rows = [
            ("Current EBITDA",          f"{ccy} {ebitda/1e9:.3f}B" if ebitda else "n/a",  "last fiscal year"),
            ("Market EV",               f"{ccy} {mkt_ev/1e9:.3f}B", "market cap + net debt"),
            ("Market EV/EBITDA",        f"{mkt_mult:.1f}×" if not math.isnan(mkt_mult) else "n/a", ""),
            ("DCF implied EV",          f"{ccy} {res.ev_local/1e9:.3f}B", ""),
            ("DCF implied EV/EBITDA",   f"{dcf_mult:.1f}×" if not math.isnan(dcf_mult) else "n/a (EBITDA ≤ 0)", ""),
        ]
        st.dataframe(
            pd.DataFrame(xc_rows, columns=["Item", "Value", "Note"]),
            use_container_width=True, hide_index=True,
        )


def _render_expander_bridge(r: ValuationResult) -> None:
    a   = r.assumptions
    res = r.dcf
    ccy = a.currency
    rf  = r.recon_fields
    with st.expander("EV → Equity Bridge", expanded=False):
        st.markdown(
            "Enterprise value belongs to all capital providers.  "
            "Subtracting net debt (total debt minus cash) isolates the equity value, "
            "which divided by diluted shares gives intrinsic value per share."
        )
        st.latex(r"\text{Equity value} = EV - \text{Net debt} \quad\Rightarrow\quad "
                 r"V/\text{share} = \frac{\text{Equity value}}{\text{Diluted shares}}")

        # Annotate net-debt and shares rows with source/disagree when available.
        # Net debt = total_debt (DEFINITIONAL) − cash (DATA); show both.
        td_src, td_dis = _recon_source_disagree(rf, "total_debt")
        ca_src, ca_dis = _recon_source_disagree(rf, "cash")
        sh_src, sh_dis = _recon_source_disagree(rf, "diluted_shares")
        nd_note = ""
        sh_note = ""
        if rf:
            nd_note = f"debt: {td_src}, {td_dis}  |  cash: {ca_src}, {ca_dis}"
            sh_note = f"{sh_src}, {sh_dis}"

        nd_label = "Net debt (debt − cash)" if a.net_debt >= 0 else "Net cash (cash − debt)"
        bridge_rows = [
            ("Enterprise value (EV)",   f"{ccy} {res.ev_local/1e9:.3f}B",         ""),
            (f"{'−' if a.net_debt >= 0 else '+'} {nd_label}",
             f"{ccy} {abs(a.net_debt)/1e9:.3f}B", nd_note),
            ("= Equity value",          f"{ccy} {res.equity_value_local/1e9:.3f}B", ""),
            ("÷ Diluted shares",        f"{a.diluted_shares/1e9:.3f}B",            sh_note),
        ]
        if a.currency != "USD":
            bridge_rows += [
                (f"= Value/share ({ccy})", f"{ccy} {res.value_per_share_local:.2f}", ""),
                (f"× FX rate ({ccy}→USD)", f"{a.fx_rate:.6f}",                      ""),
            ]
        bridge_rows.append(("= Value/share (USD)", f"USD {res.value_per_share_usd:.2f}", "intrinsic value"))
        st.dataframe(
            pd.DataFrame(bridge_rows, columns=["Step", "Value", "Note"]),
            use_container_width=True, hide_index=True,
        )


# ── MC Input Distributions expander ──────────────────────────────────────────

_MC_VAR_META: dict = {
    'revenue_growth': {'label': 'Revenue Growth',  'is_pct': True},
    'ebit_margin':    {'label': 'EBIT Margin',     'is_pct': True},
    'target_margin':  {'label': 'Target Margin',   'is_pct': True},
    'terminal_g':     {'label': 'Terminal Growth', 'is_pct': True},
    'wacc':           {'label': 'WACC',            'is_pct': True},
    'diluted_shares': {'label': 'Diluted Shares',  'is_pct': False},
    'net_debt':       {'label': 'Net Debt',        'is_pct': False},
}

_MC_RATIONALE: dict = {
    'historical_mean_growth': (
        "**Spread source:** standard deviation of the company's annual revenue growth rate "
        "over the available history. Combined with cross-source disagreement in quadrature: "
        "σ_eff = √(σ_hist² + σ_cross²). "
        "PERT bounds = mode ± 3σ_eff, clamped to [−30 %, +150 %] for plausibility.  "
        "A σ_cross of zero means only one data source is available or all sources agree."
    ),
    'historical_mean_ebit_margin': (
        "**Spread source:** standard deviation of annual EBIT margins over the available history. "
        "Combined with cross-source uncertainty in quadrature. "
        "Bounds clamped to [−20 %, +75 %]. "
        "A negative margin is allowed — the model can handle operating-loss years."
    ),
    'best_historical_ebit_margin': (
        "**Mode:** max(best historical EBIT margin, 20 % floor) — the company should converge "
        "toward at least 20 % mature margin unless its best reported year was higher. "
        "**Spread source:** reuses the same σ as the current EBIT margin (both are derived from "
        "the same reported income statement data). "
        "The PERT half-width is clamped to [3 pp, 20 pp] so that short histories don't produce "
        "a degenerate zero-width PERT, and small companies with erratic margins don't get "
        "absurdly wide ranges."
    ),
    'avg_growth_anchored_to_gdp': (
        "**Mode derivation:** avg_g = (starting revenue growth + 2.5 %) ÷ 2 — the arithmetic "
        "mean of the fade path from current growth to nominal-GDP terminal rate, so "
        "faster-growing companies get a modestly higher terminal assumption.  "
        "Capped at 2.5 % so the terminal rate never exceeds long-run nominal GDP.  "
        "**σ_hist:** macro-derived from US annual nominal GDP growth (FRED GDPA); identical for "
        "every company by design — terminal growth uncertainty is a macro variable, not a "
        "company-specific forecast.  "
        "**Asymmetric bounds:** floor = max(−2 %, 0.3 × avg_g) so economic decline is permitted "
        "but bounded; cap = min(4 %, WACC − 1 %, avg_g + 1 pp). "
        "The WACC − 1 % constraint prevents the Gordon Growth denominator from approaching zero, "
        "which would cause the terminal value to diverge."
    ),
    'capm_synthetic_rating_blume_beta': (
        "**σ_hist:** reconstructed year-by-year by re-running the WACC formula with each year's "
        "FRED DGS10 risk-free rate and the Damodaran synthetic rating implied by that year's "
        "EBIT/interest-coverage ratio.  Beta (Blume-adjusted: 0.67 × raw + 0.33) and capital "
        "structure weights are held fixed at current values across the reconstruction.  "
        "**Fallback (1.5 pp):** applied when fewer than 3 profitable years are available for "
        "reconstruction — for example, pre-profitability growth companies whose coverage ratio "
        "is negative or undefined.  "
        "**No σ_cross:** WACC inputs (beta, Rf) are market-derived and identical across all "
        "data sources, so cross-source disagreement does not add uncertainty here."
    ),
}


def _make_pert_dist_chart(
    samps: np.ndarray,
    lo: float,
    mode: float,
    hi: float,
    label: str,
    is_pct: bool,
) -> go.Figure:
    """Overlay theoretical PERT pdf and KDE of actual draws."""
    n_draws = len(samps)
    p005, p995 = np.percentile(samps, [0.5, 99.5])
    x_lo = min(p005, lo)
    x_hi = max(p995, hi)
    margin = max((x_hi - x_lo) * 0.08, 1e-6)
    x_grid = np.linspace(x_lo - margin, x_hi + margin, 400)

    # KDE of actual draws
    kde    = scipy.stats.gaussian_kde(samps)
    y_kde  = kde(x_grid)

    # Theoretical PERT pdf (scaled Beta on [lo, hi])
    if hi - lo > 1e-12:
        r_range = hi - lo
        alpha   = 1.0 + 4.0 * (mode - lo) / r_range
        beta_p  = 1.0 + 4.0 * (hi - mode) / r_range
        x_norm  = np.clip((x_grid - lo) / r_range, 0.0, 1.0)
        y_pert  = scipy.stats.beta.pdf(x_norm, alpha, beta_p) / r_range
    else:
        y_pert = np.zeros(len(x_grid))

    med = float(np.median(samps))

    tick_fmt = '.1%' if is_pct else '.3f'
    hover_fmt = '.2%' if is_pct else '.3f'

    fig = go.Figure()

    # PERT theoretical (dashed)
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_pert,
        mode='lines',
        line=dict(color='#4a9eff', width=2, dash='dash'),
        name='Theoretical PERT',
        hovertemplate=f'%{{x:{hover_fmt}}}<br>PERT density: %{{y:.5f}}<extra></extra>',
    ))

    # KDE actual draws (filled)
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_kde,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(70,130,180,0.18)',
        line=dict(color='steelblue', width=2),
        name='Realized draws (KDE)',
        hovertemplate=f'%{{x:{hover_fmt}}}<br>KDE density: %{{y:.5f}}<extra></extra>',
    ))

    # Mode marker
    fig.add_vline(x=mode, line_color='gold', line_width=1.5, line_dash='dot',
                  annotation_text='mode', annotation_position='top left',
                  annotation_font_size=11)

    # Realized median marker
    fig.add_vline(x=med, line_color='seagreen', line_width=1.5, line_dash='dash',
                  annotation_text='median', annotation_position='top right',
                  annotation_font_size=11)

    fig.update_layout(
        title=dict(text=f'Theoretical PERT vs {n_draws:,} realized draws', font_size=13),
        xaxis_title=label,
        yaxis_title='Density',
        xaxis_tickformat=tick_fmt,
        height=300,
        margin=dict(t=40, b=36, l=46, r=16),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)', font_size=11),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _render_mc_var_card(
    var: str,
    dp: dict,
    samps_dict: dict,
    r: ValuationResult,
) -> None:
    """Render one per-variable card (table + rationale + distribution popover)."""
    meta    = _MC_VAR_META.get(var, {'label': var, 'is_pct': True})
    label   = meta['label']
    is_pct  = meta['is_pct']
    samps   = samps_dict.get(var, np.array([]))
    info    = dp.get(var)         # None for promoted balance-sheet vars

    n_samps = len(samps)

    def _fp(v: float) -> str:
        """Format value as pct or absolute (billions)."""
        if not math.isfinite(v):
            return '∞' if v > 0 else '−∞'
        return f'{v:.2%}' if is_pct else f'{v/1e9:.3f}B'

    left_col, right_col = st.columns([3, 2])

    with left_col:
        # ── Operating variable (has distribution_params entry) ────────────────
        if info is not None:
            lo  = info['pert_min']
            md  = info['pert_mode']
            hi  = info['pert_max']
            sh  = info['sigma_hist']
            sc  = info['sigma_cross']
            se  = info['sigma_eff']
            cf  = info['clamp_floor']
            cc  = info['clamp_cap']
            cb  = info['clamp_binding']
            rk  = info.get('rationale_key', '')

            draws_median = float(np.median(samps)) if n_samps > 0 else float('nan')
            draws_mean   = float(np.mean(samps))   if n_samps > 0 else float('nan')

            # Human-readable clamp note
            raw_lo_str = _fp(md - 3.0 * se)
            raw_hi_str = _fp(md + 3.0 * se)
            cf_str = _fp(cf) if math.isfinite(cf) else '−∞'
            cc_str = _fp(cc) if math.isfinite(cc) else '∞'
            if cb == 'none':
                clamp_note = 'none — ±3σ_eff lies within plausibility limits'
            elif cb == 'floor':
                clamp_note = f'floor — ±3σ_eff floor is {raw_lo_str} but constrained to {cf_str}'
            elif cb == 'cap':
                clamp_note = f'cap — ±3σ_eff cap is {raw_hi_str} but constrained to {cc_str}'
            else:  # both
                clamp_note = (f'both — ±3σ_eff gives [{raw_lo_str}, {raw_hi_str}] '
                              f'but constrained to [{cf_str}, {cc_str}]')

            # Terminal g: flag whether WACC-1% or absolute 4% is the binding cap
            if var == 'terminal_g':
                wacc_val = r.assumptions.wacc
                wacc_cap = wacc_val - 0.01
                if math.isfinite(cc) and abs(cc - wacc_cap) < 1e-6:
                    clamp_note += f' (WACC−1% = {wacc_cap:.2%} binding)'
                elif math.isfinite(cc):
                    clamp_note += f' (4% GDP ceiling binding; WACC−1% = {wacc_cap:.2%})'

            # WACC fallback annotation
            sigma_hist_str = _fp(sh)
            if var == 'wacc' and r.wacc_result.wacc_sigma_fallback:
                sigma_hist_str += '  ⚠ fallback (< 3 profitable yrs for reconstruction)'

            rows = [
                ('PERT min',     _fp(lo)),
                ('PERT mode',    _fp(md)),
                ('PERT max',     _fp(hi)),
                ('Draws median', _fp(draws_median)),
                ('Draws mean',   _fp(draws_mean)),
                ('σ_hist',       sigma_hist_str),
                ('σ_cross',      _fp(sc)),
                ('σ_eff',        _fp(se)),
                ('Clamp floor',  cf_str),
                ('Clamp cap',    cc_str),
                ('Binding',      clamp_note),
            ]
            st.dataframe(
                pd.DataFrame(rows, columns=['Parameter', 'Value']),
                use_container_width=True, hide_index=True,
            )

            rationale = _MC_RATIONALE.get(rk, '')
            if var == 'wacc' and r.wacc_result.wacc_sigma_fallback:
                rationale += (
                    '\n\n**⚠ Fallback active for this ticker:** fewer than 3 profitable years '
                    'were available for WACC reconstruction (negative or undefined coverage ratio). '
                    'σ_WACC is set to 1.5 pp — a conservative estimate consistent with observed '
                    'WACC variation for investment-grade companies.'
                )
            if rationale:
                st.markdown(rationale)

        # ── Promoted balance-sheet variable ───────────────────────────────────
        else:
            rs_info   = r.recon_sigma.get(var, {})
            sc_abs    = rs_info.get('sigma_cross', 0.0)
            base_v    = (r.assumptions.diluted_shares if var == 'diluted_shares'
                         else r.assumptions.net_debt)
            prom_lo   = max(0.0, base_v - sc_abs) if var == 'diluted_shares' else base_v - sc_abs
            prom_hi   = base_v + sc_abs

            # Disagreement that tripped promotion
            rf_info    = r.recon_fields.get(var, {}) if r.recon_fields else {}
            disagree   = rf_info.get('disagree_pct')
            disagree_s = f'{disagree:.1f}%' if disagree is not None else 'n/a'

            draws_median = float(np.median(samps)) if n_samps > 0 else float('nan')
            draws_mean   = float(np.mean(samps))   if n_samps > 0 else float('nan')

            rows = [
                ('Base value (mode)',     _fp(base_v)),
                ('PERT min',             _fp(prom_lo)),
                ('PERT max',             _fp(prom_hi)),
                ('σ_cross (half-width)', _fp(sc_abs)),
                ('Draws median',         _fp(draws_median)),
                ('Draws mean',           _fp(draws_mean)),
                ('Disagreement',         disagree_s),
                ('Promotion threshold',  '> 2% relative'),
            ]
            st.dataframe(
                pd.DataFrame(rows, columns=['Parameter', 'Value']),
                use_container_width=True, hide_index=True,
            )
            var_name = 'diluted shares' if var == 'diluted_shares' else 'net debt'
            st.markdown(
                f'**Promoted variable** — cross-source disagreement of {disagree_s} exceeded the '
                f'2 % promotion threshold, so {var_name} is sampled in each simulation as a '
                f'narrow PERT centred on the preferred-source value (half-width = σ_cross). '
                f'Sampling is independent of the operating-driver copula (balance-sheet '
                f'uncertainty is orthogonal to the forecast uncertainty).'
            )
            # Use reconstructed bounds for chart
            info_chart = {'pert_min': prom_lo, 'pert_mode': base_v, 'pert_max': prom_hi}
            lo, md, hi = prom_lo, base_v, prom_hi

    with right_col:
        # Resolve lo/md/hi for operating vars (already set for promoted above)
        if info is not None:
            lo, md, hi = info['pert_min'], info['pert_mode'], info['pert_max']

        if n_samps >= 10 and hi > lo + 1e-12:
            fig = _make_pert_dist_chart(samps, lo, md, hi, label, is_pct)
            with st.popover('📊 view distribution'):
                st.plotly_chart(fig, use_container_width=True)
        elif n_samps >= 10:
            st.caption('Distribution degenerate (zero-width PERT).')
        else:
            st.caption('Not enough draws to plot.')


def _render_expander_mc_inputs(r: ValuationResult) -> None:
    t = r.transparency
    if not t:
        return

    dp         = t.get('distribution_params', {})
    samps_dict = t.get('samples', {})
    if not dp:
        return

    with st.expander('Monte Carlo Input Distributions', expanded=False):
        ci = t.get('copula_info', {})
        st.markdown(
            'One card per sampled variable.  '
            'All numbers trace directly to `result.transparency` — nothing is hardcoded.  '
            'The distribution chart overlays the theoretical PERT pdf (dashed) against a KDE '
            'of the actual draws; agreement is the primary check that the sampler is working correctly.'
        )

        # Copula summary banner
        copula_type = ci.get('type', 'gaussian')
        copula_df   = ci.get('df')
        triggers    = ci.get('triggers', [])
        fired       = [tr for tr in triggers if tr.get('fired')]
        if copula_type == 'student-t':
            fired_names = ', '.join(tr['name'] for tr in fired)
            st.info(
                f'**Student-t copula (df={copula_df})** — fatter tails than Gaussian.  '
                f'Triggered by: {fired_names}.  '
                f'{ci.get("note", "")}'
            )
        else:
            st.success(
                '**Gaussian copula** — no fat-tail triggers fired.  '
                f'{ci.get("note", "")}'
            )

        # Trigger detail table
        if triggers:
            trig_rows = []
            for tr in triggers:
                act = tr['actual_value']
                thr = tr['threshold']
                fmt_a = f'{act:.2%}'
                fmt_t = f'{thr:.0%}'
                status = '✅ fired' if tr['fired'] else '—'
                trig_rows.append((tr['name'], fmt_a, fmt_t, status))
            st.dataframe(
                pd.DataFrame(trig_rows, columns=['Trigger', 'Actual', 'Threshold', 'Status']),
                use_container_width=False, hide_index=True,
            )
            st.caption(ci.get('note', ''))

        st.divider()

        # Correlation matrices (target vs realised)
        with st.expander('Correlation matrices (target vs realised)', expanded=False):
            corr_target   = t.get('correlation_target')
            corr_realized = t.get('correlation_realized')
            var_names = ['rev_growth', 'ebit_margin', 'terminal_g', 'wacc', 'tgt_margin']
            if corr_target is not None and corr_realized is not None:
                c1, c2 = st.columns(2)
                with c1:
                    st.caption('Target (CORR matrix)')
                    st.dataframe(
                        pd.DataFrame(corr_target, index=var_names, columns=var_names).style.format('{:.2f}'),
                        use_container_width=True,
                    )
                with c2:
                    st.caption('Realised (from draws)')
                    st.dataframe(
                        pd.DataFrame(corr_realized, index=var_names, columns=var_names).style.format('{:.2f}'),
                        use_container_width=True,
                    )

        st.divider()

        # Per-variable cards
        var_order = ['revenue_growth', 'ebit_margin', 'target_margin', 'terminal_g', 'wacc']
        for pv in ('diluted_shares', 'net_debt'):
            if pv in samps_dict:
                var_order.append(pv)

        for idx, var in enumerate(var_order):
            if var not in dp and var not in samps_dict:
                continue
            if idx > 0:
                st.divider()
            label = _MC_VAR_META.get(var, {}).get('label', var)
            st.subheader(label)
            _render_mc_var_card(var, dp, samps_dict, r)

        # Validation summary
        st.divider()
        val = t.get('validation', {})
        if val:
            ok = val.get('zero_width_test_passed', False)
            st.markdown(
                f'**Zero-width consistency check:** '
                f'{"✅ pass" if ok else "❌ fail"}  |  '
                f'det = \\${val.get("deterministic_base", 0):.4f}  |  '
                f'zero-σ MC P50 = \\${val.get("mc_p50_at_zero_width", 0):.4f}  |  '
                f'|diff| = \\${val.get("abs_diff", 0):.4f}'
            )


# ── Reconciliation expander ───────────────────────────────────────────────────

def _render_expander_reconciliation(r: ValuationResult) -> None:
    rf = r.recon_fields
    rs = r.recon_sigma
    if not rf and not rs:
        return
    with st.expander("Data Reconciliation & Distribution Widening", expanded=False):
        st.markdown(
            "When multiple data sources (Yahoo Finance, FMP, SEC EDGAR) are available, "
            "their disagreement on **DATA** fields widens the Monte Carlo input distributions. "
            "**DEFINITIONAL** fields (D&A, total debt, tax rate) differ by reporting convention "
            "and are resolved by normalisation — they do not add uncertainty."
        )
        st.latex(
            r"\sigma_{\text{eff}} = \sqrt{\,\sigma_{\text{hist}}^2 + \sigma_{\text{cross}}^2\,}"
        )

        if rf:
            st.markdown("**Per-field: source used and cross-source disagreement**")
            _FIELD_LABELS = {
                "revenue":        "Revenue",
                "ebit":           "EBIT",
                "dep_amort":      "D&A",
                "capex":          "CapEx",
                "diluted_shares": "Diluted Shares",
                "total_debt":     "Total Debt",
                "cash":           "Cash",
                "tax_rate":       "Tax Rate",
            }
            field_rows = []
            for f, info in rf.items():
                label  = _FIELD_LABELS.get(f, f)
                src    = info["source"]
                dpct   = info["disagree_pct"]
                ftype  = info["field_type"]
                if ftype == "DEFINITIONAL":
                    type_str = "DEFINITIONAL (conv)"
                    dis_str  = f"{dpct:.1f}% — resolved by convention" if dpct is not None else "n/a"
                elif dpct is None:
                    type_str = "DATA"
                    dis_str  = "n/a (single source)"
                else:
                    type_str = "DATA"
                    dis_str  = f"⚠ {dpct:.1f}%" if dpct > 2.0 else f"{dpct:.1f}%"
                field_rows.append((label, src, dis_str, type_str))
            st.dataframe(
                pd.DataFrame(field_rows, columns=["Field", "Source Used", "Disagreement", "Type"]),
                use_container_width=True, hide_index=True,
            )

        if rs:
            st.markdown("**MC distribution widening: σ decomposition per sampled variable**")
            _VAR_LABELS = {
                "revenue_growth": "Revenue Growth",
                "ebit_margin":    "EBIT Margin",
                "diluted_shares": "Diluted Shares (abs)",
                "net_debt":       "Net Debt (abs)",
            }
            sigma_rows = []
            for var, info in rs.items():
                label = _VAR_LABELS.get(var, var)
                sh, sc, se = info["sigma_hist"], info["sigma_cross"], info["sigma_eff"]
                if var in ("diluted_shares", "net_debt"):
                    sigma_rows.append((label, f"{sh/1e9:.3f}B", f"{sc/1e9:.3f}B", f"{se/1e9:.3f}B",
                                       "promoted to sampled" if sc > 0 else "fixed"))
                else:
                    sigma_rows.append((label, f"{sh:.2%}", f"{sc:.2%}", f"{se:.2%}",
                                       "widened" if sc > 0 else "hist only"))
            st.dataframe(
                pd.DataFrame(sigma_rows,
                             columns=["Variable", "σ_hist", "σ_cross", "σ_eff", "Effect"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "σ_hist = historical std of the annual figure.  "
                "σ_cross = std of the derived rate / level across sources.  "
                "σ_eff = √(σ_hist² + σ_cross²) — combined uncertainty fed into the PERT bounds."
            )


# ── Main render ───────────────────────────────────────────────────────────────

def _render_valuation(r: ValuationResult) -> None:
    price = r.current_price_usd
    det   = r.dcf.value_per_share_usd
    pct_vs_market = (r.p50 - price) / price * 100 if price > 0 else float('nan')
    direction = "undervalued" if pct_vs_market > 0 else "overvalued"

    # ── Headline metrics ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price",      f"${price:.2f}")
    c2.metric("Base Case (det.)",   f"${det:.2f}",
              delta=f"{(det-price)/price*100:+.1f}%")
    c3.metric("Median (P50)",       f"${r.p50:.2f}",
              delta=f"{pct_vs_market:+.1f}%")
    c4.metric("P10 / P90",          f"${r.p10:.0f} – ${r.p90:.0f}")
    c5.metric("P(Undervalued)",     f"{r.pct_undervalued:.1f}%")

    # ── Distribution histogram ────────────────────────────────────────────────
    _render_histogram(r)

    # ── Percentile table ──────────────────────────────────────────────────────
    pct_df = pd.DataFrame([
        {"Percentile": "P10",  "Value (USD)": f"${r.p10:.2f}"},
        {"Percentile": "P25",  "Value (USD)": f"${r.p25:.2f}"},
        {"Percentile": "P50 (median)", "Value (USD)": f"${r.p50:.2f}"},
        {"Percentile": "P75",  "Value (USD)": f"${r.p75:.2f}"},
        {"Percentile": "P90",  "Value (USD)": f"${r.p90:.2f}"},
        {"Percentile": "Mean", "Value (USD)": f"${r.mean_val:.2f}"},
        {"Percentile": "Stdev","Value (USD)": f"${r.std_val:.2f}"},
        {"Percentile": "Current price", "Value (USD)": f"${price:.2f}"},
    ])
    st.dataframe(pct_df, use_container_width=False, hide_index=True)

    # ── Plain-English interpretation ──────────────────────────────────────────
    over_under = f"{abs(pct_vs_market):.1f}% {direction}"
    st.info(
        f"**{r.ticker} — {r.n_valid:,} simulated scenarios ({r.copula_label} copula)**\n\n"
        f"The median intrinsic value across all simulations is **\\${r.p50:.2f}**, "
        f"which is **{over_under}** relative to the current market price of \\${price:.2f}. "
        f"The deterministic (base-case) estimate is \\${det:.2f}.\n\n"
        f"There is a 10% chance intrinsic value is below **\\${r.p10:.2f}** and a 10% chance "
        f"it is above **\\${r.p90:.2f}** — that's the P10–P90 range covering the middle 80% "
        f"of simulated outcomes.\n\n"
        f"**P(Undervalued) = {r.pct_undervalued:.1f}%**: this is the share of simulated "
        f"scenarios in which the model's intrinsic value exceeds today's market price of "
        f"\\${price:.2f}.  It is not a probability of future returns — it reflects how "
        f"often the DCF assumptions (revenue growth, margins, WACC) combine to produce "
        f"a value above the current price."
    )

    st.divider()

    # ── Expanders ─────────────────────────────────────────────────────────────
    _render_expander_drivers(r)
    _render_expander_wacc(r)
    _render_expander_assumptions(r)
    _render_expander_forecast(r)
    _render_expander_terminal(r)
    _render_expander_bridge(r)
    _render_expander_mc_inputs(r)
    _render_expander_reconciliation(r)


# ── Valuation tab layout ──────────────────────────────────────────────────────

with tab_valuation:
    ticker_input = st.text_input("Ticker symbol", value="MSFT", key="val_ticker")
    run_btn      = st.button("Run valuation", key="val_run")

    if run_btn:
        ticker = ticker_input.upper().strip()
        if ticker:
            try:
                with st.spinner(f"Running Monte Carlo DCF for {ticker} — fetching data and running 10,000 simulations…"):
                    result = _cached_valuation(ticker)
                st.session_state["val_result"] = result
            except Exception as exc:
                st.error(f"**Valuation failed for {ticker}**: {exc}")
                # Clear any stale result so the old ticker's output isn't shown
                st.session_state.pop("val_result", None)
        else:
            st.warning("Enter a ticker symbol first.")

    if "val_result" in st.session_state:
        _render_valuation(st.session_state["val_result"])
