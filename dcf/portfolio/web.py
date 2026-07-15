import sys
from pathlib import Path
_DCF_ROOT  = Path(__file__).parent.parent          # .../Archer-Capital/dcf
_REPO_ROOT = _DCF_ROOT.parent                      # .../Archer-Capital
sys.path.insert(0, str(_DCF_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import copy
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
from scipy.optimize import brentq
import streamlit as st

from valuation.montecarlo    import run_valuation
from valuation.result        import ValuationResult
from valuation.dcf           import MATURE_MARGIN_DEFAULT, value as _dcf_value
from valuation.sources       import fetch_yahoo, fetch_fmp, fetch_edgar
from valuation.reconcile     import reconcile

st.set_page_config(page_title="Archer Capital", layout="wide")
st.title("Archer Capital")

tab_strategies, tab_valuation, tab_macro = st.tabs(["Strategies", "Valuation", "Macro Trading"])


# ══════════════════════════════════════════════════════════════════════════════
#  VALUATION TAB
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False, ttl=3600)
def _cached_valuation(ticker: str) -> tuple[ValuationResult, dict[str, str]]:
    sources = []
    # source_status records "ok" or the failure reason per source, so a dropped
    # source never looks identical to a successful full reconciliation in the UI.
    source_status: dict[str, str] = {}
    for name, fetcher in (("Yahoo", fetch_yahoo), ("FMP", fetch_fmp), ("EDGAR", fetch_edgar)):
        try:
            sources.append(fetcher(ticker))
            source_status[name] = "ok"
        except Exception as e:
            source_status[name] = str(e)
    if len(sources) >= 2:
        recon = reconcile(sources)
        result = run_valuation(ticker, n_sims=10_000,
                             sigma_cross=recon.sigma_cross,
                             reconcile_result=recon)
    else:
        result = run_valuation(ticker, n_sims=10_000)
    return result, source_status


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
        title=f"{r.ticker}: Monte Carlo DCF ({r.n_valid:,} simulations, {r.copula_label} copula)",
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
            "The model starts by pulling several years of the company's actual reported "
            "financials and computing the key ratios for each year. I average across years "
            "rather than just using the most recent year to make the base-case inputs more "
            "stable and less sensitive to one-off events. The standard deviations of each "
            "annual figure feed directly into the Monte Carlo: a company that has been "
            "consistently predictable gets narrower distributions, and a company with erratic "
            "history gets wider ones."
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
                hist_display.map(lambda x: f"{x:.1%}" if pd.notna(x) else ""),
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
            "WACC is the rate used to discount all future cash flows back to today. "
            "The intuition is that a dollar of free cash flow in year 5 is worth less than "
            "a dollar today because of the time value of money and the risk that it may not "
            "materialise. Cost of equity comes from CAPM with a Blume-adjusted beta and "
            "reflects the return equity investors require for bearing the company's risk. "
            "Cost of debt is derived from the Damodaran synthetic rating methodology using "
            "the EBIT to interest-coverage ratio, and is reduced by the tax shield on "
            "interest expense."
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
            "These are the inputs that drive every year of the explicit forecast. "
            "Revenue growth and EBIT margin are not held constant: they start at the "
            "historically averaged base-case values and fade linearly toward their terminal "
            "values over the forecast horizon. I chose linear fading because growth and "
            "margins always evolve gradually rather than jumping overnight. CapEx fades to "
            "match D&A by the final year so that the model's steady-state reinvestment "
            "assumption is consistent with the Gordon Growth terminal value, which assumes "
            "the company is no longer growing faster than the economy."
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
            "FCFF (Free Cash Flow to the Firm) is the cash the business generates for all "
            "capital providers after operating expenses, taxes, and reinvestment needs are "
            "accounted for. It equals NOPAT (net operating profit after tax) plus D&A "
            "(added back because it is non-cash) minus CapEx (real cash spent on assets) "
            "minus the change in net working capital (cash consumed or released by the "
            "business growing or contracting). Each year's FCFF is discounted back to today "
            "at WACC to get its present value. The margin column shows the year-by-year "
            "fade from the starting EBIT margin toward the target mature margin."
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
            "The explicit forecast covers a finite number of years. Everything beyond that "
            "window is captured by the terminal value, which assumes the business has reached "
            "a steady state and grows at a constant rate forever. The Gordon Growth Model "
            "formula converts that perpetual cash flow stream into a single lump sum that is "
            "discounted back to today and added to the PV of explicit FCFFs to get the total "
            "enterprise value. The EV/EBITDA cross-check compares the multiple implied by "
            "the DCF against the market's current multiple. If the DCF-implied multiple is "
            "wildly different from where the market trades, it is usually a signal that one "
            "of the assumptions is out of range and worth revisiting."
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
    with st.expander("EV to Equity Bridge", expanded=False):
        st.markdown(
            "Enterprise value represents the total value of the business to all capital "
            "providers combined, both debt holders and equity shareholders. To isolate what "
            "belongs to equity shareholders specifically, net debt (total financial borrowings "
            "minus cash on hand) is subtracted. Dividing by the diluted share count then "
            "converts the total equity value into a per-share figure that can be directly "
            "compared to the current market price."
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


# ── Methodology expander (rendered first in the expander list) ───────────────

def _render_expander_methodology(r: ValuationResult) -> None:
    with st.expander('Methodology: Full Model Specification', expanded=False):
        st.markdown(r"""
### Overview

I built this tool because I wanted a proper way to evaluate the stocks in my own portfolio
and understand what was actually being priced into them beneath all the numbers and
fundamentals. Why was a stock trading at 30 times earnings when its balance sheet said
something completely different? What assumptions would an investor need to believe for the
current price to make sense? Those were the questions I wanted to answer, and I could not
find an existing tool that answered them the way I wanted.

The model is a Discounted Cash Flow (DCF) analysis with a Monte Carlo simulation layer
built on top. The DCF produces a single intrinsic value estimate based on a specific set
of assumptions. The Monte Carlo gives you a full probability distribution of estimates by
running that same DCF tens of thousands of times, varying the inputs slightly on each run.
I added the Monte Carlo because the future is genuinely unpredictable, and presenting a
single number as the answer to what a company is worth always felt dishonest to me.
A distribution of outcomes is more truthful.

The full valuation equation and every modelling assumption are my own independent research.
Every design decision, from how I treat operating leases to how I set the terminal growth
rate, was made by me based on first principles and independent reading of valuation theory.
The code was written using Claude Code across the full stack, covering data ingestion,
financial computation, the Monte Carlo simulation engine, and this web interface. The model
design, the reasoning behind every specification, and all the financial judgement calls are
entirely mine.

---

### 1. Data Sourcing and Reconciliation

Financial data is fetched from up to three independent sources: Yahoo Finance, Financial
Modelling Prep (FMP), and SEC EDGAR. I pull from multiple sources rather than relying on
just one because no single provider is perfectly reliable across all companies. Yahoo Finance
is fast and generally accurate for US reporters. FMP has well-structured financial statement
data. EDGAR is the original source filing, which makes it authoritative but harder to parse
consistently. By comparing all three, I can identify where they disagree and treat that
disagreement as useful information in its own right.

**How specific fields are defined:**

For total debt, I include only bonds payable and finance lease obligations. Operating leases
are deliberately excluded. Under modern accounting standards (IFRS 16 and ASC 842), the cost
of operating leases is already recognised as an operating expense within EBIT on the income
statement. If I also included the operating lease liability in the balance sheet net debt
figure, I would be counting the same obligation twice: once through its impact on operating
margins and again as a deduction from enterprise value. Finance leases represent genuine
borrowed capital and belong in net debt. Operating leases are committed rental contracts
whose cost is already captured inside the margins.

For depreciation and amortisation, I use Yahoo Finance's figure, which is pulled directly
as a single reported line item from the cash flow statement. I tested building D&A from
SEC EDGAR's structured data, but EDGAR constructs the figure by summing separate XML tags
for depreciation, amortisation of intangibles, and lease amortisation independently. This
approach frequently misses components, particularly for companies with complex capital
structures or international operations, and produces a figure that does not match what the
company actually reported. The Yahoo cash flow statement number is the line the company
itself presented to shareholders, which is exactly what I want.

For the tax rate, I calculate it as income tax expense divided by income before tax, both
taken directly from the income statement. I chose this over using the statutory corporate
rate because the effective rate already reflects everything the company actually does in
practice: tax credits, deferred tax balances, R&D incentives, and international structuring.
What matters to a DCF is not the rate the law prescribes but the rate the company actually
pays on its earnings.

**Why disagreements make this model more honest:**

When two data sources report different numbers for the same field, I do not simply pick one
and discard the other. I calculate the relative gap between them. If that gap exceeds 2%, I
treat it as evidence of genuine data uncertainty rather than a rounding difference, and I
widen the Monte Carlo input distribution for that variable accordingly using:
""")
        st.latex(
            r"\sigma_{\text{eff}} = \sqrt{\,\sigma_{\text{hist}}^2 + \sigma_{\text{cross}}^2\,}"
        )
        st.markdown(r"""
This is one of the features I am most deliberately proud of in this model. Most DCF tools
either pick a data source and trust it completely, or ask the user to enter numbers manually.
Mine acknowledges that the data itself carries uncertainty and folds that uncertainty directly
into the output range. When sources disagree substantially, the model's probability
distribution widens to reflect that. A model that is honest about the quality of its own
inputs produces more meaningful output than one that presents false precision.

For fields where cross-source disagreement exceeds 2%, the affected variable is also promoted
to a directly sampled input in the Monte Carlo, meaning its value is independently drawn in
each simulation rather than held fixed. Balance-sheet data uncertainty feeds directly into
the distribution of intrinsic value estimates.

Source priority when values conflict: SEC EDGAR takes precedence, then FMP, then Yahoo Finance.

---

### 2. Financial History

I use three to five fiscal years of annual income statement and cash flow data. For each year
I calculate the key financial ratios: revenue growth, EBIT margin, D&A as a percentage of
revenue, CapEx as a percentage of revenue, stock-based compensation as a percentage of
revenue, and the effective tax rate. These annual figures are averaged to form the base-case
inputs for the DCF, and their standard deviations become the historical uncertainty component
fed into the Monte Carlo distributions.

Stock-based compensation is already expensed in reported EBIT and is not added back anywhere
in the model. This is an intentional choice. It means my operating margins are lower than the
cash-adjusted or non-GAAP margins you commonly see in sell-side analyst reports, but they are
internally consistent across every year of the forecast and through to the terminal value.
I chose not to treat SBC as a non-cash add-back because doing so while also treating diluted
share count as fixed systematically understates the true cost of equity compensation.

**Forecast horizon:** I use a 10-year explicit forecast when trailing revenue growth exceeds
15%, and a 5-year forecast otherwise. High-growth companies need a longer explicit window
before their growth rate can credibly fade down to a terminal rate near nominal GDP. Forcing
a company growing at 35% into a 5-year window and immediately applying a 2.5% terminal rate
would shortchange years 6 through 10 where substantial value is still being created. The
longer window gives those companies the runway their growth profile justifies.

NWC is taken from the most recent balance sheet rather than averaged across years because
year-over-year net working capital changes derived from cash flow statements are calculated
inconsistently across data providers. Using the most recent balance sheet ratio is
conservative and avoids compounding data noise.

---

### 3. DCF Mechanics

The model forecasts free cash flow to the firm (FCFF) for each year of the explicit horizon,
then adds a terminal value to capture all cash flows beyond that point. FCFF is the cash the
business generates for all capital providers (both debt and equity holders) after operating
expenses, taxes, and reinvestment needs are accounted for. It is the cleanest measure of a
business's ability to generate value for its owners without being distorted by capital
structure choices.

**Revenue forecast:** starts at the base-case growth rate and fades linearly to the terminal
growth rate over the forecast horizon. I chose a linear fade rather than holding growth
constant because revenue growth always decelerates as a business matures and scales. Linear
fading is a simple and defensible approximation of that natural deceleration and avoids the
artificial cliff-edge that comes from holding growth constant and then abruptly switching to
a terminal rate.

**Size-dependent growth ceiling:** within each forecast year, the growth rate is capped by
a scale-dependent ceiling before it is applied to revenue. The ceiling is computed as
g_ceiling(R) = max(terminal_g, 74.0 / R^1.029) where R is projected revenue in USD
billions at the start of that year. I calibrated this to two empirical anchors: a company
already generating around $250 billion of annual revenue can realistically sustain roughly
25% growth, while a company generating $1 trillion cannot credibly sustain more than about
6%. The exponent of 1.029 means the ceiling roughly halves as revenue doubles, which is
consistent with how growth constraints work in practice at scale.

This ceiling matters for high-growth companies whose Monte Carlo draws can include very high
sampled growth rates. Without it, a company like NVDA, which had historical revenue growth
above 100% in recent years, would compound those draws across a 10-year forecast horizon
into revenue figures that exceed the entire global economy. The ceiling does not adjust the
terminal growth rate, does not cap the market capitalisation, and does not constrain the
base-case DCF. It only clips the year-by-year applied growth rate inside each simulation
path when that path reaches a revenue scale where the sampled rate becomes economically
implausible.

**EBIT margin:** fades linearly from the starting historically-averaged margin to the target
mature margin. I set the target as the higher of the company's best historical EBIT margin
and a 20% floor. The reasoning is that a business should be capable of reaching at least
20% operating margins at full maturity unless its own history demonstrates it cannot sustain
margins anywhere near that level, in which case the best year it has achieved is the right
anchor.

**CapEx:** fades linearly from the historical average to the D&A percentage by the final
forecast year, so that steady-state net reinvestment (CapEx minus D&A) approaches zero.
This is consistent with the Gordon Growth terminal value assumption that in perpetuity the
company reinvests only enough capital to replace depreciating assets. Without this fade, the
terminal value would implicitly assume the company keeps investing heavily forever, which
significantly overstates value.

**Free cash flow to the firm each year:**
""")
        st.latex(
            r"\text{FCFF}_t = \underbrace{\text{EBIT}_t \cdot (1-t)}_{\text{NOPAT}}"
            r"+ \text{D\&A}_t - \text{CapEx}_t - \Delta\text{NWC}_t"
        )
        st.markdown(r"""
NOPAT is the net operating profit after tax, representing what the business earns from
operations after paying tax but before any financing costs. D&A is added back because it is
a non-cash charge that reduced EBIT but consumed no actual cash. CapEx is subtracted because
it is real cash spent on assets. The change in NWC is subtracted when positive (cash consumed
by growth) and added back when negative (cash released as the business becomes more efficient).

**Terminal value (Gordon Growth applied to the final forecast year's FCFF):**
""")
        st.latex(
            r"TV = \frac{\text{FCFF}_N \cdot (1+g_\infty)}{\text{WACC} - g_\infty}"
        )
        st.markdown(r"""
The terminal value represents the present value of all cash flows from year N+1 to infinity,
assuming they grow at a constant rate forever. The denominator (WACC minus g) is why the
terminal growth rate must always remain below WACC. If g were equal to or above WACC, the
denominator would approach zero and the terminal value would become infinitely large, which
is economically nonsensical.

**Terminal growth rate:** rather than applying a single fixed rate to every company, I anchor
the terminal rate to each company's own growth trajectory. The formula is avg_g = (starting
revenue growth + 2.5%) divided by 2, taking the arithmetic midpoint between where growth is
today and where nominal GDP sits. The rate is then capped at 2.5% so it can never exceed
long-run nominal GDP growth regardless of how fast the company is currently growing. The
lower bound is the maximum of negative 2% and 30% of avg_g, allowing for modest contraction
scenarios. The upper cap is the minimum of 4%, WACC minus 1%, and avg_g plus 1 percentage
point. The WACC minus 1% constraint is the most important one: it keeps the Gordon Growth
denominator well away from zero and prevents the terminal value from becoming unreasonably
large for high-growth companies.

**Enterprise value to equity value:**
""")
        st.latex(
            r"V_{\text{equity}} = EV - D_{\text{net}}, \quad"
            r"V/\text{share} = \frac{V_{\text{equity}}}{\text{diluted shares}}"
        )
        st.markdown(r"""
The DCF produces an enterprise value, which belongs to all capital providers. Subtracting
net debt (total financial debt minus cash) isolates the equity value attributable to
shareholders. Dividing by diluted share count gives intrinsic value per share in local
reporting currency. For non-USD reporters, this is converted to USD using the spot exchange
rate in the base case, or a simulated per-year FX path in Monte Carlo mode.

---

### 4. WACC

The weighted average cost of capital is the discount rate applied to every year's FCFF. It
represents the blended required return that both equity holders and debt holders need to be
compensated for their investment. I chose to compute WACC from first principles rather than
using a consensus analyst estimate because I wanted the discount rate to be fully internally
consistent with the data already being pulled for the DCF inputs.

**Cost of equity via CAPM with Blume-adjusted beta:**
""")
        st.latex(
            r"K_E = R_f + \beta_{\text{adj}} \cdot \text{ERP}, \quad"
            r"\beta_{\text{adj}} = 0.67 \cdot \beta_{\text{raw}} + 0.33"
        )
        st.markdown(r"""
The Blume adjustment mean-reverts beta toward 1. The empirical justification is that extreme
betas tend to moderate over time as companies mature. A beta of 1.5 today is unlikely to
remain at 1.5 for the next 5 to 10 years across a full DCF horizon. Multiplying the raw beta
by 0.67 and adding 0.33 shifts it partway toward the market average of 1, which I believe is
more appropriate for a long-duration valuation than using the historical point estimate
directly. The risk-free rate is taken from the FRED DGS10 series (the 10-year US Treasury
yield) and the equity risk premium is 4.5%, which is Damodaran's implied ERP.

**Cost of debt** is derived using the Damodaran synthetic rating methodology. The EBIT to
interest-coverage ratio maps to an implied credit rating, which then maps to a default
spread. Adding that spread to the risk-free rate gives the pre-tax cost of debt. The
after-tax cost applies the interest tax shield: because interest is tax-deductible, the
effective cost of debt is reduced proportionally by the tax rate. I chose this approach
because it derives cost of debt purely from the income statement without requiring actively
traded debt securities, which makes the model work equally well for small and large companies.

**WACC historical uncertainty:** I reconstruct the WACC for each available historical year
by re-running the formula with that year's DGS10 rate and coverage-implied spread, holding
beta and capital structure weights fixed. The standard deviation of these annual WACC
estimates becomes the historical uncertainty input for the Monte Carlo. When fewer than
three profitable years are available (which occurs for pre-profitability growth companies
where coverage ratios are negative or undefined), I apply a fallback sigma of 1.5 percentage
points, which is consistent with observed WACC variation for investment-grade companies.

---

### 5. Monte Carlo Uncertainty Quantification

Five variables are sampled jointly in each simulation: revenue growth, EBIT margin, target
margin, terminal growth rate, and WACC. Running 10,000 simulations produces a distribution
of 10,000 independent intrinsic value estimates from which I read off percentiles to
characterise the full range of outcomes.

**Why PERT distributions:**

Each variable is drawn from a PERT distribution, also called a scaled Beta distribution.
A PERT is defined by three numbers: a minimum (worst case), a most likely value (the mode),
and a maximum (best case). I chose PERT over a Normal distribution for two reasons. A Normal
distribution technically extends to negative and positive infinity, meaning it could in theory
draw a revenue growth rate of negative 300% or a margin of 500%. PERT stays within its
defined range. Additionally, PERT is flexible: shifting the mode toward one end creates an
asymmetric distribution that can reflect cases where the downside risk is larger than the
upside potential, or vice versa.

The mode is always the base-case value from the historical average. The minimum and maximum
are set at mode minus 3 times sigma_eff and mode plus 3 times sigma_eff respectively. The
combined uncertainty measure is:
""")
        st.latex(
            r"\sigma_{\text{eff}} = \sqrt{\,\sigma_{\text{hist}}^2 + \sigma_{\text{cross}}^2\,}"
        )
        st.markdown(r"""
The PERT shape parameters are derived from the three bounds as follows:
""")
        st.latex(
            r"\alpha = 1 + \frac{4(\text{mode} - \text{min})}{\text{max} - \text{min}}, \quad"
            r"\beta  = 1 + \frac{4(\text{max}  - \text{mode})}{\text{max} - \text{min}}"
        )
        st.markdown(r"""
**How the model preserves relationships between variables (the copula):**

If I sampled each variable independently, I would occasionally draw unrealistic combinations
such as very high revenue growth paired with very low WACC, which would produce an
unreasonably optimistic scenario. Real financial variables have natural relationships. Faster-
growing companies tend to carry higher risk profiles and therefore higher discount rates.
Compressed margins tend to occur at the same time as weaker cash conversion. A copula allows
me to preserve these relationships across all five variables simultaneously while still
drawing each one from its own PERT marginal distribution.

I use a Gaussian copula by default, which captures linear correlations between variables
through a hand-specified 5x5 target correlation matrix. For instance, revenue growth and
WACC carry a positive correlation of 0.20 because faster-growing companies are generally
riskier.

When any of four fundamental uncertainty triggers fire, I switch to a Student-t copula with
5 degrees of freedom. A Student-t copula has heavier tails than Gaussian, which means it
generates more extreme simultaneous adverse scenarios: growth is low AND margins are
compressed AND WACC is elevated, all at the same time. This reflects the well-documented
phenomenon that in stress environments, asset correlations that are normally modest tend to
spike, making bad outcomes more coincident than normal times would suggest.

The four triggers are:

| Trigger | Threshold | Why I chose it |
|---|---|---|
| Revenue growth volatility | above 20% | Structurally erratic revenue makes extreme joint scenarios more likely |
| EBIT margin volatility | above 5 percentage points | High operating leverage amplifies the severity of joint downturns |
| FCF margin volatility | above 8% | Unpredictable cash conversion justifies heavier tails in the joint distribution |
| Maximum cross-source disagreement | above 15 percentage points | Severe data uncertainty warrants a more conservative joint distribution |

Stock-price volatility is deliberately excluded from these triggers. Stock price moves
reflect market sentiment and investor positioning, not the fundamental uncertainty of the
business inputs themselves. Using price volatility to widen a fundamental DCF model would
conflate two completely different things and does not produce a more honest output.

---

### 6. FX Translation

For companies that report in USD, there is no FX simulation. The exchange rate is fixed at
1.0 throughout all simulations.

For foreign-currency reporters, I simulate the exchange rate path over the forecast horizon
using a driftless Geometric Brownian Motion, drawing one simulated rate per forecast year.
Each year's discounted FCF is translated at that year's simulated rate rather than a single
fixed rate applied to the whole forecast. This means FX uncertainty compounds over time:
year-1 cash flows carry less currency risk than year-5 cash flows, which carry less than the
terminal value. The terminal value is translated at the final-year simulated rate.

The process is deliberately driftless. I am not predicting that any currency will appreciate
or depreciate. The negative one-half sigma-squared term in the exponent is the Ito drift
correction and its job is to keep the expected value of the exchange rate equal to today's
spot rate at every point in the simulation. Uncertainty compounds around that anchor, but
the centre of the distribution stays pinned to the current exchange rate.

---

### 7. Model Biases and Known Limitations

I think it is important to be transparent about where this model is deliberately conservative
and where those choices have consequences.

CapEx fading to D&A means that in the terminal year, the model assumes net reinvestment
(CapEx minus D&A) is essentially zero. This is a reasonable assumption for asset-light
businesses like software companies whose growth is not constrained by physical capital. For
asset-heavy businesses such as energy infrastructure or manufacturing, this assumption is
wrong: they need to keep spending more than they depreciate just to sustain operations. The
model will treat asset-light companies fairly and may overvalue asset-heavy ones by assuming
an unrealistic reinvestment holiday in perpetuity.

SBC being expensed in EBIT depresses margins relative to cash-based metrics. High-SBC
companies, particularly cloud software and early-stage technology businesses, will appear
less profitable under this model than they look on an adjusted earnings basis. I accept this
trade-off deliberately. If stock-based compensation were truly immaterial, companies would
not go to such lengths to exclude it from their headline numbers.

Terminal growth is capped at 2.5%, which means the model never awards a company more than
nominal GDP growth in perpetuity. For companies with genuine secular tailwinds who will
plausibly outgrow the broader economy for longer than the explicit forecast horizon, this cap
undervalues them. I accept that constraint because the alternative, allowing company-specific
terminal rates above GDP, opens the door to circular reasoning where the terminal rate is
simply tuned to justify a target price.

The consequence of these three choices taken together is that mature, large-cap technology
companies with high multiples will frequently appear overvalued under this model. That is not
always because the market is wrong. It is because the market is pricing in a longer
high-growth runway and a cash-earnings premium that this model does not grant. The designed
purpose of this tool is to find genuine undervaluation in under-covered or misunderstood
companies where a rigorous FCFF framework surfaces something the market has missed, not to
argue that Microsoft or Apple are 30% overvalued on a technical basis.

**This model does not work for pre-profit or cash-burning companies.** A DCF is built
entirely around the idea that a business generates positive free cash flow that can be
discounted back to today. For a company that is burning cash and has no near-term path to
profitability, the model either produces a negative intrinsic value (which is meaningless,
because equity has a floor of zero due to limited liability) or it exploits the linear
margin fade to project artificial profitability by the final forecast year while producing
negative FCFFs in most of the preceding years. In either case, the output is not a
valuation. It is an artefact of applying the wrong tool to the wrong problem.

The value of an early-stage company depends on things a single-path DCF does not model:
the probability of reaching profitability, the amount of dilutive capital required to get
there, the option value of its technology, and the binary outcomes that separate the large
fraction of such companies that fail from the small fraction that succeed. A scenario
analysis or a real-options framework is the appropriate tool for those cases.

When the model detects that the base-case DCF produces a negative intrinsic value or that
the terminal-year FCFF is non-positive, it suppresses the headline valuation entirely and
replaces it with an explanation. I would rather tell you the model is not applicable than
present a negative number as if it means something.
""")


# ── Correlation Structure & Copula expander ───────────────────────────────────

_CORR_VAR_NAMES = ['rev_growth', 'ebit_margin', 'terminal_g', 'wacc', 'tgt_margin']


def _make_corr_heatmap(matrix: np.ndarray, title: str) -> go.Figure:
    """5×5 annotated diverging heatmap for a correlation matrix."""
    n = len(_CORR_VAR_NAMES)
    text = [[f'{matrix[i, j]:.2f}' for j in range(n)] for i in range(n)]
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=_CORR_VAR_NAMES,
        y=_CORR_VAR_NAMES,
        text=text,
        texttemplate='%{text}',
        textfont=dict(size=11),
        colorscale='RdBu',       # Red=-1, White=0, Blue=+1
        zmin=-1, zmax=1,
        showscale=True,
        colorbar=dict(thickness=10, len=0.85, tickformat='.1f'),
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13),
        yaxis=dict(autorange='reversed'),
        height=290,
        margin=dict(t=38, b=8, l=82, r=16),
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def _render_expander_corr_copula(r: ValuationResult) -> None:
    t = r.transparency
    if not t:
        return
    ci            = t.get('copula_info', {})
    corr_target   = t.get('correlation_target')
    corr_realized = t.get('correlation_realized')
    samps         = t.get('samples', {})
    if not ci:
        return

    with st.expander('Correlation Structure & Copula', expanded=False):

        # ── Copula type banner ────────────────────────────────────────────────
        copula_type = ci.get('type', 'gaussian')
        copula_df   = ci.get('df')
        triggers    = ci.get('triggers', [])
        fired       = [tr for tr in triggers if tr['fired']]

        if copula_type == 'student-t':
            fired_names = ', '.join(tr['name'] for tr in fired)
            st.info(
                f'**Student-t copula (df={copula_df})** used for this run. '
                f'This produces a fat-tailed joint distribution, meaning the simulation generates '
                f'more extreme simultaneous adverse scenarios than a standard Gaussian copula would. '
                f'Triggered by: **{fired_names}**. '
                'The Student-t copula is selected when fundamental uncertainty indicators suggest '
                'the inputs are volatile enough to justify heavier joint tails.'
            )
        else:
            st.success(
                '**Gaussian copula** used for this run. '
                'None of the four fundamental unpredictability triggers fired, so the joint '
                'distribution of sampled inputs follows a standard multivariate normal structure '
                'on the probability scale.'
            )

        # ── Trigger table (always shown, both copula types) ───────────────────
        st.markdown('**Trigger evaluation:** the model switches to Student-t if any one of these fires:')
        trig_rows = []
        for tr in triggers:
            trig_rows.append({
                'Trigger':   tr['name'],
                'Actual':    f'{tr["actual_value"]:.2%}',
                'Threshold': f'{tr["threshold"]:.0%}',
                'Fired':     '✅' if tr['fired'] else 'No',
            })
        st.dataframe(pd.DataFrame(trig_rows), use_container_width=False, hide_index=True)
        st.caption(
            'These triggers measure fundamental unpredictability in the business inputs: '
            'revenue growth volatility, EBIT margin volatility, FCF volatility, and '
            'cross-source data disagreement. '
            'Stock-price volatility is deliberately excluded because it reflects market '
            'sentiment, not the uncertainty of the underlying valuation inputs. '
            'Using price volatility to widen a fundamental DCF would conflate two entirely '
            'different concepts.'
        )

        st.divider()

        # ── Side-by-side correlation heatmaps ─────────────────────────────────
        if corr_target is not None and corr_realized is not None:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    _make_corr_heatmap(corr_target, 'Target correlation matrix'),
                    use_container_width=True,
                )
            with c2:
                st.plotly_chart(
                    _make_corr_heatmap(corr_realized, 'Realized correlation (10,000 draws)'),
                    use_container_width=True,
                )
            st.caption(
                'Realized values should be close to target values. '
                'This confirms the Gaussian/t copula, Cholesky decomposition, and PERT '
                'inverse-CDF pipeline is working correctly end-to-end.'
            )

            # Scatter popover: rev growth vs WACC
            rg = samps.get('revenue_growth', np.array([]))
            wa = samps.get('wacc', np.array([]))
            if len(rg) >= 10 and len(wa) == len(rg):
                r_realized = float(np.corrcoef(rg, wa)[0, 1])
                r_target   = float(corr_target[0, 3])   # CORR row 0=rev_g, col 3=wacc
                idx = np.random.default_rng(0).choice(len(rg), min(2000, len(rg)), replace=False)
                fig_sc = go.Figure(go.Scatter(
                    x=rg[idx], y=wa[idx],
                    mode='markers',
                    marker=dict(size=3, color='steelblue', opacity=0.35),
                    hovertemplate='rev growth: %{x:.1%}<br>WACC: %{y:.1%}<extra></extra>',
                ))
                fig_sc.update_layout(
                    title=dict(
                        text=(f'Revenue Growth vs WACC  '
                              f'(realized r = {r_realized:.3f},  target = {r_target:.2f})'),
                        font_size=13,
                    ),
                    xaxis_title='Revenue Growth',
                    yaxis_title='WACC',
                    xaxis_tickformat='.0%',
                    yaxis_tickformat='.1%',
                    height=380,
                    margin=dict(t=48, b=48, l=60, r=16),
                    plot_bgcolor='rgba(0,0,0,0)',
                    showlegend=False,
                )
                with st.popover('📊 Rev Growth vs WACC scatter'):
                    st.plotly_chart(fig_sc, use_container_width=True)

        st.divider()

        # ── Sampling pipeline ─────────────────────────────────────────────────
        st.markdown('**Sampling pipeline: how the correlated draws are generated**')
        st.markdown(
            r"""
1. Draw **Z**, a matrix of shape (n simulations x 5 variables) of independent standard-normal variates, where every entry is completely random and uncorrelated.
2. Cholesky-decompose the target correlation matrix: find **L** such that **L** multiplied by its own transpose equals **Σ**. This produces a lower-triangular matrix that encodes the desired correlations.
3. Correlate the draws: **Z_C** = **Z** multiplied by **L** transposed. Each row of **Z_C** is now drawn from a multivariate normal distribution with correlation structure **Σ**.
4. *(Student-t only)* draw chi-squared scalars **w** from a chi-squared distribution with ν degrees of freedom, then divide: **T** = **Z_C** divided by the square root of (**w** divided by ν). This produces correlated t-variates with ν degrees of freedom that preserve the same correlation structure but with heavier joint tails.
5. Apply the marginal CDF column-by-column: **U** = Φ(**Z_C**) for Gaussian, or F_{t,ν}(**T**) for Student-t. This converts each column into correlated uniform percentiles on the interval (0, 1).
6. Map each column through the PERT inverse-CDF: x_i = PERT⁻¹(U_i ; min_i, mode_i, max_i). This produces draws from each variable's own PERT distribution while preserving the correlation structure introduced in step 3.

**PERT parameterisation** (scaled Beta):
"""
        )
        st.latex(
            r"\alpha = 1 + \frac{4(\text{mode} - \text{min})}{\text{max} - \text{min}}, \quad"
            r"\beta  = 1 + \frac{4(\text{max}  - \text{mode})}{\text{max} - \text{min}}"
        )
        st.markdown(
            'When the theoretical PERT curve and the KDE of realized draws agree closely '
            '(visible in the Monte Carlo Input Distributions expander), it confirms '
            'the sampler is correctly centred and that steps 5 and 6 are working as intended.'
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
        "**Spread source:** the standard deviation of the company's annual revenue growth rate "
        "over the available historical years. This is combined with any cross-source data "
        "disagreement in quadrature to produce sigma_eff = the square root of (sigma_hist "
        "squared plus sigma_cross squared). The PERT bounds are set at the mode plus or minus "
        "3 times sigma_eff, then clamped to a plausibility range of negative 30% to positive "
        "150%. A sigma_cross of zero simply means only one data source was available or all "
        "sources agreed exactly on this field."
    ),
    'historical_mean_ebit_margin': (
        "**Spread source:** the standard deviation of annual EBIT margins over the available "
        "history, combined with cross-source uncertainty using the same quadrature formula. "
        "Bounds are clamped to the range from negative 20% to positive 75%. Negative margins "
        "are permitted because the model needs to handle companies that have been loss-making "
        "in some historical years without breaking the simulation."
    ),
    'best_historical_ebit_margin': (
        "**Mode:** the higher of the company's best historical EBIT margin and a 20% floor. "
        "The reasoning is that a business at full maturity should reach at least 20% operating "
        "margins unless its own track record suggests the ceiling is lower, in which case the "
        "best year it has achieved is the right anchor. "
        "**Spread source:** the same sigma as the current EBIT margin, because both figures are "
        "derived from the same reported income statement data and the uncertainty is symmetric. "
        "The PERT half-width is clamped to the range of 3 to 20 percentage points so that "
        "companies with short histories do not produce a degenerate zero-width distribution, "
        "and companies with very erratic margins do not produce implausibly wide ones."
    ),
    'avg_growth_anchored_to_gdp': (
        "**Mode derivation:** avg_g = (starting revenue growth + 2.5%) divided by 2. This is "
        "the arithmetic midpoint between where growth is today and where nominal GDP sits, so "
        "faster-growing companies get a modestly higher terminal assumption while remaining "
        "grounded in long-run macro reality. The rate is capped at 2.5% so it can never exceed "
        "nominal GDP growth in perpetuity. "
        "**Sigma_hist:** derived from the historical standard deviation of US annual nominal GDP "
        "growth (FRED GDPA series). This is the same value for every company by design. Terminal "
        "growth uncertainty is a macro variable, not a company-specific forecast. "
        "**Asymmetric bounds:** the floor is the maximum of negative 2% and 30% of avg_g, "
        "permitting modest economic contraction scenarios. The cap is the minimum of 4%, WACC "
        "minus 1%, and avg_g plus 1 percentage point. The WACC minus 1% constraint is critical "
        "because it keeps the Gordon Growth Model denominator away from zero, which would cause "
        "the terminal value to become unreasonably large or infinite."
    ),
    'capm_synthetic_rating_blume_beta': (
        "**Sigma_hist:** reconstructed year-by-year by re-running the WACC formula with each "
        "historical year's FRED DGS10 risk-free rate and the Damodaran synthetic rating implied "
        "by that year's EBIT to interest-coverage ratio. Blume-adjusted beta (0.67 times raw "
        "plus 0.33) and capital structure weights are held fixed at their current values across "
        "all historical years. "
        "**Fallback (1.5 percentage points):** applied when fewer than 3 profitable years are "
        "available for the reconstruction, for example pre-profitability growth companies where "
        "coverage ratios are negative or undefined. "
        "**No sigma_cross:** WACC inputs including beta and the risk-free rate are market-derived "
        "and identical across all data sources, so cross-source disagreement does not apply here."
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
                clamp_note = 'none (±3σ_eff lies within plausibility limits)'
            elif cb == 'floor':
                clamp_note = f'floor active (±3σ_eff floor would be {raw_lo_str} but constrained to {cf_str})'
            elif cb == 'cap':
                clamp_note = f'cap active (±3σ_eff cap would be {raw_hi_str} but constrained to {cc_str})'
            else:  # both
                clamp_note = (f'both active (±3σ_eff gives [{raw_lo_str}, {raw_hi_str}] '
                              f'but constrained to [{cf_str}, {cc_str}])')

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
                    'σ_WACC is set to 1.5 pp, a conservative estimate consistent with observed '
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
                f'**Promoted variable:** cross-source disagreement of {disagree_s} exceeded the '
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
        st.markdown(
            'One card per sampled variable. Every number shown here is sourced directly from '
            'the transparency payload built during the simulation run, nothing is hardcoded or '
            'approximated. The distribution chart in each card overlays the theoretical PERT '
            'curve (dashed line) against a KDE of the actual simulation draws. When the two '
            'lines agree closely, it confirms the sampler is correctly centred on the '
            'base-case mode and drawing from the intended distribution.'
        )

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

        # ── Size-dependent growth ceiling ─────────────────────────────────────
        gc = t.get('g_ceiling', {})
        if gc:
            st.divider()
            st.subheader('Size-Dependent Revenue Growth Ceiling')
            pct_b = gc.get('pct_paths_bound', 0.0)
            avg_y = gc.get('avg_years_bound', 0.0)
            A_val = gc.get('A', 74.0)
            b_val = gc.get('b', 1.029)
            st.markdown(
                f'The DCF loop applies a scale-dependent ceiling to revenue growth inside each '
                f'simulation path, tightening year by year as projected revenue grows. '
                f'The ceiling at any point in the forecast is computed as '
                f'**g_ceiling(R) = max(terminal_g, {A_val:.1f} / R^{b_val:.3f})** where R is '
                f'projected revenue in USD billions at the start of that forecast year. '
                f'I calibrated this to two empirical anchors: a company already at ~$250B of '
                f'revenue can realistically sustain around 25% annual growth, while a ~$1T '
                f'revenue company cannot sustain more than roughly 6% (consistent with '
                f'historical evidence from the largest public companies). The exponent of '
                f'{b_val:.3f} means the ceiling roughly halves as revenue doubles, reflecting '
                f'the well-documented difficulty of growing faster in absolute dollar terms as '
                f'a business scales.'
            )
            st.markdown(
                'This ceiling operates at the input level inside the DCF loop, not as an '
                'output-side market cap constraint. It clips the sampled growth rate before '
                'it is applied to revenue for that year, so the compounding effect accumulates '
                'correctly. Without this, high historical growth rates for companies like NVDA '
                'can compound into economically impossible revenue figures over a 10-year '
                'horizon, producing valuations in the tens of trillions of dollars.'
            )
            col1, col2 = st.columns(2)
            with col1:
                st.metric('Paths where ceiling bound', f'{pct_b:.1f}%')
            with col2:
                st.metric('Avg forecast years bound (per path)', f'{avg_y:.2f}')
            if pct_b > 5.0:
                st.warning(
                    f'The size ceiling was binding in **{pct_b:.1f}%** of simulated paths '
                    f'(average {avg_y:.1f} forecast years per path). This suppresses the right '
                    f'tail of the distribution. The P90 and mean values shown above are lower '
                    f'than they would be without the ceiling. For large-cap high-growth companies '
                    f'this is the intended behaviour.'
                )
            elif pct_b > 1.0:
                st.info(
                    f'The size ceiling was binding in {pct_b:.1f}% of paths '
                    f'(average {avg_y:.2f} forecast years per path). Modest right-tail compression.'
                )
            else:
                st.caption(
                    f'Size ceiling rarely active ({pct_b:.1f}% of paths). '
                    'Revenue is not projected to reach scales where the ceiling becomes constraining.'
                )


# ── Currency Simulation (GBM) expander ───────────────────────────────────────

_GBM_STATIC_EXPLANATION = r"""
**Geometric Brownian Motion with annual steps (driftless):**

Each forecast year, the simulated exchange rate is multiplied by an independently drawn
random lognormal factor. The cumulative simulated path to year t is:
"""

def _render_expander_fx(r: ValuationResult) -> None:
    t_pay = r.transparency
    if not t_pay:
        return
    fx    = t_pay.get('fx_info', {})
    ccy   = r.currency
    spot  = r.assumptions.fx_rate

    with st.expander('Currency Simulation (GBM)', expanded=False):
        if not fx.get('is_foreign', False):
            st.info(
                f'**{r.ticker} reports in USD** so no FX simulation is run. '
                'Every forecast-year FCF and the terminal value are already in USD '
                'and the exchange rate stays fixed at 1.0 across all simulations. '
                'The section below describes how the FX module works for '
                'foreign-currency reporters.'
            )
            st.divider()

        # ── GBM equation (always shown) ───────────────────────────────────────
        st.markdown(_GBM_STATIC_EXPLANATION)
        st.latex(
            r"r_t = r_0 \cdot \exp\!\left(\,"
            r"\sum_{s=1}^{t}"
            r"\left(-\tfrac{1}{2}\sigma^2 + \sigma Z_s\right)"
            r"\,\right), \quad Z_s \overset{\text{iid}}{\sim} \mathcal{N}(0,1)"
        )
        st.markdown(
            r'The $-\tfrac{1}{2}\sigma^2$ term is the Ito drift correction. '
            r'It keeps the expected value of the exchange rate equal to today\'s spot rate '
            r'at every point in time, meaning this process does not predict appreciation or '
            r'depreciation in any direction. It is a random walk centred on the current spot '
            r'rate. Because each year is simulated independently, FX uncertainty '
            r'**compounds with time**: year-1 cash flows carry less currency risk than '
            r'the terminal value.'
        )

        if not fx.get('is_foreign', False):
            return

        sigma       = fx.get('sigma', 0.0)
        per_year    = fx.get('per_year', [])
        paths       = fx.get('sample_paths', np.array([]))
        T           = len(per_year)
        n_paths     = paths.shape[0] if hasattr(paths, 'shape') and paths.ndim == 2 else 0

        if T == 0 or n_paths == 0:
            st.warning('FX simulation data not available in the payload.')
            return

        st.markdown(f'**Currency:** {ccy}/USD  |  **Spot rate (today):** {spot:.5f}  |  **Annual σ:** {sigma:.0%}')

        # ── Spaghetti plot ────────────────────────────────────────────────────
        years = list(range(1, T + 1))

        # Combine all sample paths into one trace (NaN-separated for speed)
        x_spaghetti, y_spaghetti = [], []
        for path_row in paths:
            x_spaghetti += [0] + years + [None]
            y_spaghetti += [spot] + list(path_row) + [None]

        x_band  = [0] + years
        y_p10   = [spot] + [p['p10'] for p in per_year]
        y_p50   = [spot] + [p['p50'] for p in per_year]
        y_p90   = [spot] + [p['p90'] for p in per_year]

        fig = go.Figure()

        # Sample paths (single combined trace, grey, low opacity)
        fig.add_trace(go.Scatter(
            x=x_spaghetti, y=y_spaghetti,
            mode='lines',
            line=dict(color='rgba(180,180,180,0.25)', width=0.8),
            showlegend=False,
            hoverinfo='skip',
        ))

        # P90 / P10 fill band
        fig.add_trace(go.Scatter(
            x=x_band + x_band[::-1],
            y=y_p90 + y_p10[::-1],
            fill='toself',
            fillcolor='rgba(70,130,180,0.12)',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))

        # P10, P50, P90 lines
        for yvals, color, dash, name in [
            (y_p10, 'darkorange', 'dot',   'P10'),
            (y_p50, 'steelblue',  'solid', 'P50 (median)'),
            (y_p90, 'darkorange', 'dot',   'P90'),
        ]:
            fig.add_trace(go.Scatter(
                x=x_band, y=yvals,
                mode='lines',
                line=dict(color=color, width=2, dash=dash),
                name=name,
                hovertemplate=f'{name}: %{{y:.5f}}<extra></extra>',
            ))

        # Spot rate marker at year 0
        fig.add_trace(go.Scatter(
            x=[0], y=[spot],
            mode='markers',
            marker=dict(color='seagreen', size=8, symbol='circle'),
            name=f'Spot ({spot:.5f})',
            hovertemplate=f'Spot today: {spot:.5f}<extra></extra>',
        ))

        fig.update_layout(
            title=dict(
                text=f'{ccy}/USD Exchange Rate: {n_paths} GBM sample paths (σ = {sigma:.0%}/yr)',
                font_size=13,
            ),
            xaxis=dict(title='Forecast year', dtick=1, tick0=0),
            yaxis=dict(title=f'{ccy} per 1 USD (inverted: higher = weaker {ccy})'
                             if spot < 1 else f'USD per 1 {ccy}'),
            height=400,
            margin=dict(t=44, b=44, l=60, r=16),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Per-year table ────────────────────────────────────────────────────
        table_rows = [{'Year': 0, 'P10 rate': f'{spot:.5f}', 'Median rate': f'{spot:.5f}',
                       'P90 rate': f'{spot:.5f}', 'Note': 'spot (fixed)'}]
        for p in per_year:
            table_rows.append({
                'Year':        p['year'],
                'P10 rate':    f"{p['p10']:.5f}",
                'Median rate': f"{p['p50']:.5f}",
                'P90 rate':    f"{p['p90']:.5f}",
                'Note':        '',
            })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=False, hide_index=True)
        st.caption(
            f'All rates are {ccy}/USD. P10/P90 widen each year as uncertainty compounds '
            f'over the forecast horizon. The median path stays near the spot rate by '
            f'construction because the process is driftless.'
        )


# ── Validation & Reproducibility expander ─────────────────────────────────────

def _render_expander_validation(r: ValuationResult) -> None:
    t_pay = r.transparency
    if not t_pay:
        return
    val  = t_pay.get('validation', {})
    if not val:
        return

    with st.expander('Validation & Reproducibility', expanded=False):
        ok       = val.get('zero_width_test_passed', False)
        det      = val.get('deterministic_base', float('nan'))
        mc_p50   = val.get('mc_p50_at_zero_width', float('nan'))
        abs_diff = val.get('abs_diff', float('nan'))
        n_sims   = t_pay.get('n_sims', 0)
        seed     = t_pay.get('random_seed', 'n/a')

        badge = '✅ PASS' if ok else '❌ FAIL'
        color = 'green' if ok else 'red'
        st.markdown(f'**Zero-width collapse test:** :{color}[{badge}]')

        rows = [
            ('Zero-width MC P50',       f'${mc_p50:.4f}',   'median of n=50 draws at σ=0'),
            ('Deterministic base case', f'${det:.4f}',       'value() called directly'),
            ('|Difference|',            f'${abs_diff:.4f}',  'must be < $0.01 to pass'),
            ('Simulations (main run)',  f'{n_sims:,}',         ''),
            ('Random seed',             str(seed),             'fixed for reproducibility'),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=['Check', 'Value', 'Note']),
            use_container_width=False, hide_index=True,
        )

        st.markdown('**Invariants verified by this test:**')
        st.markdown(
            '- **Zero-width collapse:** when all PERT half-widths are set to zero, every draw '
            'must be the base-case mode and the MC P50 must match `value()` to within \\$0.01. '
            'This confirms the simulation is centred on the correct value and the PERT '
            'inverse-CDF is accurate at the mode.\n'
            '- **USD ticker FX path:** when `currency == "USD"`, fx_path is never set and '
            '`vps_usd = vps_local x 1.0`, which is byte-identical to the non-FX code path.\n'
            '- **Realized correlation matching target:** the corrcoef of the 5 sampled input '
            'arrays should be within approximately 0.02 of the target correlation matrix entries '
            'at 10,000 draws (visible in the Correlation Structure and Copula expander).'
        )


# ── Reconciliation expander ───────────────────────────────────────────────────

def _render_expander_reconciliation(r: ValuationResult) -> None:
    rf = r.recon_fields
    rs = r.recon_sigma
    if not rf and not rs:
        return
    with st.expander("Data Reconciliation & Distribution Widening", expanded=False):
        st.markdown(
            "When multiple data sources are available, their field-by-field disagreement on "
            "DATA fields widens the Monte Carlo input distributions. DEFINITIONAL fields such "
            "as D&A, total debt, and tax rate differ between sources because of accounting "
            "convention differences rather than genuine data errors. These are resolved by "
            "normalisation and do not contribute additional uncertainty to the simulation."
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
                    dis_str  = f"{dpct:.1f}% (resolved by convention)" if dpct is not None else "n/a"
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
                "σ_hist = historical std of the annual figure. "
                "σ_cross = std of the derived rate or level across sources. "
                "σ_eff = the square root of (σ_hist squared + σ_cross squared), "
                "which is the combined uncertainty fed into the PERT bounds."
            )


# ── Gap analysis helpers ──────────────────────────────────────────────────────

def _mktcap_tier(r: ValuationResult) -> str:
    mktcap = r.current_price_usd * r.assumptions.diluted_shares
    if mktcap > 500e9:
        return 'mega'
    if mktcap < 2e9:
        return 'small'
    return 'mid'


def _rev_growth_flag(g: float, tier: str) -> str:
    hi = {'mega': 0.15, 'small': 0.35}.get(tier, 0.25)
    lo = {'mega': 0.10, 'small': 0.20}.get(tier, 0.15)
    if g > hi:
        return 'implausible at this scale'
    if g > lo:
        return 'aggressive'
    return 'plausible'


def _margin_flag(m: float) -> str:
    if m > 0.60:
        return 'implausible'
    if m > 0.50:
        return 'aggressive'
    return 'plausible'


def _wacc_flag(w: float) -> str:
    if w < 0.05:
        return 'implausible'
    if w < 0.07:
        return 'aggressive'
    return 'plausible'


def _solve_sensitivity(base, target: float, param: str, lo: float, hi: float):
    """Bisect for the value of `param` that makes DCF value == target. Returns None on failure."""
    def f(x):
        a = copy.copy(base)
        setattr(a, param, x)
        try:
            return _dcf_value(a) - target
        except Exception:
            return float('nan')

    try:
        f_lo, f_hi = f(lo), f(hi)
        if not (math.isfinite(f_lo) and math.isfinite(f_hi)):
            return None
        if f_lo * f_hi >= 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-4, maxiter=100))
    except Exception:
        return None


def _compute_gap_analysis(r: ValuationResult) -> dict:
    price      = r.current_price_usd
    p50, p10, p90 = r.p50, r.p10, r.p90
    base       = r.assumptions

    if not math.isfinite(p50) or p50 == 0:
        return {}

    gap_pct    = (price - p50) / abs(p50)
    is_over    = gap_pct > 0   # price above model median

    if abs(gap_pct) < 0.10:
        direction = 'fair'
    elif is_over:
        direction = 'over'
    else:
        direction = 'under'

    pvr = 'above' if price > p90 else ('below' if price < p10 else 'inside')

    solves = {}
    if pvr != 'inside':
        tg = base.terminal_g
        g_solved  = _solve_sensitivity(base, price, 'revenue_growth', -0.50, 2.00)
        if g_solved is not None:
            solves['revenue_growth'] = g_solved
        tm_solved = _solve_sensitivity(base, price, 'target_margin',  -0.50, 0.95)
        if tm_solved is not None:
            solves['target_margin']  = tm_solved
        w_solved  = _solve_sensitivity(base, price, 'wacc', tg + 0.002, 0.60)
        if w_solved is not None:
            solves['wacc'] = w_solved

    return {
        'gap_pct':   gap_pct,
        'direction': direction,
        'pvr':       pvr,
        'is_over':   is_over,
        'p10': p10, 'p50': p50, 'p90': p90, 'price': price,
        'solves':    solves,
        'tier':      _mktcap_tier(r),
        'base_g':    base.revenue_growth,
        'base_tm':   base.target_margin,
        'base_wacc': base.wacc,
    }


def _render_gap_paragraph(r: ValuationResult) -> None:
    ga = _compute_gap_analysis(r)
    if not ga:
        return

    price = ga['price']
    p50, p10, p90 = ga['p50'], ga['p10'], ga['p90']
    gap_abs = abs(ga['gap_pct'])
    pvr     = ga['pvr']
    is_over = ga['is_over']
    solves  = ga['solves']
    tier    = ga['tier']

    # Sentence 1: gap size and direction
    if ga['direction'] == 'fair':
        adj = 'above' if ga['gap_pct'] < 0 else 'below'
        s1 = (f"The DCF median of \\${p50:.0f} sits {gap_abs:.0%} {adj} the current price "
              f"of \\${price:.2f} — approximately fairly valued by this model.")
    elif is_over:
        s1 = (f"The DCF median of \\${p50:.0f} sits {gap_abs:.0%} below the current price "
              f"of \\${price:.2f} — the market is pricing the stock above the model's central estimate.")
    else:
        s1 = (f"The DCF median of \\${p50:.0f} sits {gap_abs:.0%} above the current price "
              f"of \\${price:.2f} — the market is pricing the stock below the model's central estimate.")

    # Sentence 2: range overlap
    if pvr == 'inside':
        s2 = (f"The market price falls within the model's P10–P90 range "
              f"(\\${p10:.0f}–\\${p90:.0f}) — the gap is consistent with normal parameter "
              f"uncertainty and does not require unusual assumptions to explain.")
    elif pvr == 'above':
        s2 = (f"The market price sits above the P90 outcome (\\${p90:.0f}). "
              f"Base-case assumptions and their uncertainty cannot explain the premium — "
              f"the market is pricing in something the model does not capture.")
    else:
        s2 = (f"The market price sits below the P10 outcome (\\${p10:.0f}). "
              f"Even pessimistic assumptions produce a higher value — "
              f"the market is pricing in risk the model does not capture.")

    # Sentence 3: sensitivity lines (only outside P10-P90)
    s3 = ''
    if solves and pvr != 'inside':
        lines = []

        if 'revenue_growth' in solves:
            g = solves['revenue_growth']
            if is_over:
                flag  = _rev_growth_flag(g, tier)
                lines.append(f"revenue growth of {g:.1%} (base: {ga['base_g']:.1%}, {flag})")
            else:
                lines.append(f"revenue growth collapsing to {g:.1%} (base: {ga['base_g']:.1%})")

        if 'target_margin' in solves:
            tm = solves['target_margin']
            if is_over:
                flag  = _margin_flag(tm)
                lines.append(f"terminal EBIT margins of {tm:.0%} (base: {ga['base_tm']:.0%}, {flag})")
            else:
                lines.append(f"terminal margins of {tm:.0%} (base: {ga['base_tm']:.0%})")

        if 'wacc' in solves:
            w = solves['wacc']
            if is_over:
                flag  = _wacc_flag(w)
                lines.append(f"a WACC of {w:.1%} (base: {ga['base_wacc']:.1%}, {flag})")
            else:
                lines.append(f"a WACC of {w:.1%} (base: {ga['base_wacc']:.1%})")

        if lines:
            verb = 'Closing the gap requires' if is_over else 'Closing the gap downward requires'
            if len(lines) == 1:
                solve_str = lines[0]
            elif len(lines) == 2:
                solve_str = f"{lines[0]} or {lines[1]}"
            else:
                solve_str = f"{', '.join(lines[:-1])}, or {lines[-1]}"
            s3 = f"{verb} {solve_str}."

    # Sentence 4: conclusion (only outside P10-P90)
    s4 = ''
    if pvr == 'above':
        s4 = "The market is pricing in significant outperformance beyond what historical drivers support."
    elif pvr == 'below':
        s4 = "The market is pricing in significant deterioration or risk not captured in historical financials."

    st.info(' '.join(s for s in [s1, s2, s3, s4] if s))


# ── DCF not-applicable banner ─────────────────────────────────────────────────

def _render_dcf_not_applicable(r: ValuationResult, dcf_app: dict) -> None:
    price = r.current_price_usd

    st.error(
        "**DCF not applicable** — this company does not generate positive free cash flow "
        "within the forecast horizon. A discounted-cash-flow model cannot meaningfully "
        "value pre-profit or cash-burning businesses: their value depends on financing, "
        "dilution, and binary future outcomes that a single-path DCF does not capture. "
        "A scenario or real-options approach is more appropriate."
    )

    triggers = dcf_app.get('triggers', [])
    if triggers:
        _trigger_labels = {
            'terminal_fcff_non_positive':    'Terminal-year FCFF is zero or negative — the Gordon Growth terminal value is undefined.',
            'current_ebit_negative':         'Current EBIT is negative — the business is operating at a loss today.',
            'base_case_intrinsic_negative':  'Base-case intrinsic value per share is negative.',
        }
        trigger_lines = []
        for t in triggers:
            if t.startswith('forecast_fcff_negative_in_'):
                trigger_lines.append(f'FCFF is negative in one or more forecast years ({t.replace("_", " ").replace("forecast fcff negative in ", "")}).')
            else:
                trigger_lines.append(_trigger_labels.get(t, t.replace('_', ' ')))
        st.markdown('**Triggers that fired:**\n' + '\n'.join(f'- {line}' for line in trigger_lines))

    c1, c2 = st.columns(2)
    c1.metric('Current Price', f'${price:.2f}')
    c2.metric('Base Case DCF', f'${dcf_app.get("base_case_intrinsic", 0):.2f}',
              help='Shown for diagnostic purposes only. This number is not a meaningful valuation.')

    with st.expander('Diagnostic distribution — NOT a valuation', expanded=False):
        st.caption(
            'The distribution below shows the Monte Carlo output for this ticker. '
            'Because terminal FCFF is non-positive, the majority of simulated paths '
            'produce negative or near-zero intrinsic values. These numbers are not '
            'a valuation. They are shown only to illustrate why the DCF framework '
            'breaks down for this company.'
        )
        _render_histogram(r)

    st.divider()
    _render_expander_methodology(r)
    _render_expander_drivers(r)
    _render_expander_wacc(r)
    _render_expander_assumptions(r)
    _render_expander_forecast(r)
    _render_expander_terminal(r)
    _render_expander_bridge(r)
    _render_expander_reconciliation(r)


# ── Main render ───────────────────────────────────────────────────────────────

def _render_valuation(r: ValuationResult) -> None:
    # ── DCF applicability gate ────────────────────────────────────────────────
    dcf_app = r.transparency.get('dcf_applicability', {}) if r.transparency else {}
    if not dcf_app.get('dcf_applicable', True):
        _render_dcf_not_applicable(r, dcf_app)
        return

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
    c4.metric("P10 / P90",          f"${r.p10:.0f} - ${r.p90:.0f}")
    c5.metric("P(Undervalued)",     f"{r.pct_undervalued:.1f}%")

    # ── Gap interpretation paragraph ──────────────────────────────────────────
    _render_gap_paragraph(r)

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
        f"**{r.ticker}: {r.n_valid:,} simulated scenarios ({r.copula_label} copula)**\n\n"
        f"The median intrinsic value across all simulations is **\\${r.p50:.2f}**, "
        f"which is **{over_under}** relative to the current market price of \\${price:.2f}. "
        f"The deterministic (base-case) estimate is \\${det:.2f}.\n\n"
        f"There is a 10% chance intrinsic value is below **\\${r.p10:.2f}** and a 10% chance "
        f"it is above **\\${r.p90:.2f}**. That is the P10 to P90 range covering the middle 80% "
        f"of simulated outcomes.\n\n"
        f"**P(Undervalued) = {r.pct_undervalued:.1f}%**: this is the share of simulated "
        f"scenarios in which the model's intrinsic value exceeds today's market price of "
        f"\\${price:.2f}. It is not a probability of future returns. It reflects how "
        f"often the DCF assumptions (revenue growth, margins, WACC) combine to produce "
        f"a value above the current price."
    )

    # ── Size ceiling callout (shown when ceiling is actively suppressing paths) ──
    gc = r.transparency.get('g_ceiling', {}) if r.transparency else {}
    if gc and gc.get('pct_paths_bound', 0.0) > 1.0:
        pct_b = gc['pct_paths_bound']
        avg_y = gc['avg_years_bound']
        st.caption(
            f"Size ceiling active in **{pct_b:.1f}%** of simulated paths "
            f"(avg {avg_y:.1f} forecast years per path). "
            "Revenue growth is capped by a scale-dependent ceiling that tightens as projected "
            "revenue grows within each path. This suppresses the right tail of the distribution. "
            "See Monte Carlo Input Distributions for details."
        )

    st.divider()

    # ── Expanders ─────────────────────────────────────────────────────────────
    _render_expander_methodology(r)
    _render_expander_drivers(r)
    _render_expander_wacc(r)
    _render_expander_assumptions(r)
    _render_expander_forecast(r)
    _render_expander_terminal(r)
    _render_expander_bridge(r)
    _render_expander_corr_copula(r)
    _render_expander_mc_inputs(r)
    _render_expander_fx(r)
    _render_expander_validation(r)
    _render_expander_reconciliation(r)


# ── Valuation tab layout ──────────────────────────────────────────────────────

with tab_valuation:
    ticker_input = st.text_input("Ticker symbol", value="MSFT", key="val_ticker")
    run_btn      = st.button("Run valuation", key="val_run")

    if run_btn:
        ticker = ticker_input.upper().strip()
        if ticker:
            try:
                with st.spinner(f"Running Monte Carlo DCF for {ticker}: fetching data and running 10,000 simulations…"):
                    result, source_status = _cached_valuation(ticker)
                st.session_state["val_result"] = result
                st.session_state["val_source_status"] = source_status
            except Exception as exc:
                st.error(f"**Valuation failed for {ticker}**: {exc}")
                # Clear any stale result so the old ticker's output isn't shown
                st.session_state.pop("val_result", None)
                st.session_state.pop("val_source_status", None)
        else:
            st.warning("Enter a ticker symbol first.")

    if "val_result" in st.session_state:
        status = st.session_state.get("val_source_status", {})
        failed = {name: reason for name, reason in status.items() if reason != "ok"}
        if failed:
            n_ok = len(status) - len(failed)
            detail = "  ".join(f"**{name}**: {reason}" for name, reason in failed.items())
            st.warning(
                f"⚠ Data source degraded — this valuation reconciles **{n_ok} of "
                f"{len(status)}** intended sources.  {detail}"
            )
        _render_valuation(st.session_state["val_result"])


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGIES TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_strategies:
    from strategies.strategy_page import render_strategy_page
    from strategies.private_competitor_distress.config import STRATEGY_CONFIG
    from strategies.north_atlantic_salmon.page import render_salmon_page
    from strategies.dcf_tests.page import render_dcf_tests_page
    from strategies.phase3_registry.page import render_phase3_registry_page
    from strategies.phase3_registry.summary_page import render_phase3_summary_page

    _SECTIONS = [
        "DCF Tests — Research",
        "North Atlantic Salmon — Research",
        "Phase 3 — Results Summary",
        "Phase 3 — Signal Research Registry",
        STRATEGY_CONFIG["name"],
    ]

    selected = st.selectbox(
        "Select section",
        options=_SECTIONS,
        key="strategy_select",
    )

    if selected == "DCF Tests — Research":
        render_dcf_tests_page()
    elif selected == "North Atlantic Salmon — Research":
        render_salmon_page()
    elif selected == "Phase 3 — Results Summary":
        render_phase3_summary_page()
    elif selected == "Phase 3 — Signal Research Registry":
        render_phase3_registry_page()
    else:
        _STRATEGY_REGISTRY = {STRATEGY_CONFIG["name"]: STRATEGY_CONFIG}
        render_strategy_page(_STRATEGY_REGISTRY[selected])


# ══════════════════════════════════════════════════════════════════════════════
#  MACRO TRADING TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_macro:
    from strategies.macro_trading.page import render_macro_trading_page

    render_macro_trading_page()
