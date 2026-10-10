"""
Macro Trading tab.

New, separate research track (macro-regime and sector-conditional strategies),
distinct from the Phase 3 Signal Research Registry.

Steps 1-2 of the methodology below are backed by a real, working Macro
Regime Classifier (infrastructure, not a DSR trial -- see
research/macro_regime/README.md). Steps 3-8 remain placeholder text only.
"""
from __future__ import annotations

import streamlit as st

from strategies.macro_trading.regime_data import load_regime_section_data, load_external_signals
from strategies.macro_trading.regime_chart import build_regime_timeline_figure

_REGIME_BADGE_COLOR = {
    "Goldilocks": "#2ECC71",
    "Reflation": "#F1C40F",
    "Stagflation": "#E74C3C",
    "Deflation/Risk-off": "#3498DB",
}


def _render_live_read(live: dict) -> None:
    regime = live["regime"]
    color = _REGIME_BADGE_COLOR.get(regime, "#95A5A6")

    st.markdown(
        f"<span style='background:{color};color:black;padding:4px 14px;"
        f"border-radius:14px;font-size:1.05rem;font-weight:700;'>{regime}</span>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"As of {live['as_of_month'].strftime('%B %Y')}  |  "
        f"Growth: {live['growth_direction']}  ·  Inflation: {live['inflation_direction']}  |  "
        f"GDP YoY (smoothed): {live['gdp_yoy_smoothed']*100:.2f}%  ·  "
        f"Core CPI YoY (smoothed): {live['core_cpi_yoy_smoothed']*100:.2f}%"
    )

    col1, col2 = st.columns(2)
    with col1:
        if live["precarious_daily"]:
            st.warning(f"⚠ Precarious flag: **ON** (as of {live['precarious_daily_asof'].strftime('%Y-%m-%d')})")
        else:
            st.success(f"Precarious flag: off (as of {live['precarious_daily_asof'].strftime('%Y-%m-%d')})")
        conds = live["precarious_conditions"]
        st.caption(
            f"Credit compressed: {conds['credit_compressed']}  ·  "
            f"VIX low: {conds['vix_low']}  ·  "
            f"Curve flat/inverted: {conds['curve_flat_or_inverted']}  ·  "
            f"Stock-bond corr rising: {conds['stockbond_corr_rising']}"
        )
    with col2:
        st.info(f"Policy overlay: **{live['policy_label']}**")
        st.caption(
            f"Fed funds (effective): {live['dff_level']:.2f}%  ·  "
            f"6mo change: {live['dff_change_6m']:+.2f}pp  "
            f"(as of {live['policy_daily_asof'].strftime('%Y-%m-%d')})"
        )

    st.caption(
        f"GDP reading used is first-release, public as of {live['gdp_realtime_start']} "
        f"(next GDP release will update this)  ·  "
        f"CPI reading used is first-release, public as of {live['cpi_realtime_start']}. "
        f"Informational only -- not a backtest."
    )


def _render_external_signals(signals: dict) -> None:
    st.markdown("##### External nowcast/consensus signals `[external, informational]`")
    st.caption(
        "Not part of Part 1's coincident regime label — separate models/surveys, "
        "shown for context only."
    )

    clev = signals["cleveland_nowcast"]
    spf = signals["spf_consensus"]
    growth_status = signals.get("growth_prediction_status")

    if growth_status:
        st.error(f"**{growth_status['label']}**")
        st.caption(growth_status["summary"])
        st.caption(growth_status["note"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Cleveland Fed Inflation Nowcast** (Core CPI)")
        st.caption(
            f"Target month {clev['target_month'].strftime('%B %Y')}: "
            f"{clev['terminal_nowcast_mom_pct']:.2f}% MoM, direction vs. prior "
            f"month **{clev['direction_vs_prior_month']}** "
            f"(as of {clev['terminal_nowcast_date'].strftime('%Y-%m-%d')})"
        )
        st.caption(f"Walk-forward: {clev['walk_forward_result']}")
    with col2:
        st.markdown("**SPF Consensus** (Philadelphia Fed)  `growth reading not validated`")
        survey_q = spf["survey_quarter"]
        st.caption(
            f"Survey quarter {survey_q.year} Q{survey_q.quarter}: "
            f"median forecast for the FOLLOWING quarter — "
            f"Core CPI {spf['median_core_cpi_forecast_next_q_pct']:.2f}% (ann.), "
            f"Real GDP {spf['median_rgdp_growth_forecast_next_q_pct']:.2f}% (ann., "
            f"context only — see growth-prediction status above)"
        )
        st.caption(spf["note"])
        st.caption(f"Walk-forward: {spf['walk_forward_result']}")


def _render_regime_classifier_section() -> None:
    timeline, live, error = load_regime_section_data()

    if error:
        st.warning(f"⚠ Macro Regime Classifier degraded — {error}")

    if live is not None:
        _render_live_read(live)
    elif timeline is None:
        st.info("Macro Regime Classifier has not been run yet.")
        return

    if timeline is not None and not timeline.empty:
        fig = build_regime_timeline_figure(timeline)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Coincident regime classifier only (GDP/core-CPI acceleration-deceleration, "
            "point-in-time gated). Precarious fragility flag and policy overlay are "
            "independent overlays, not blended into the regime label. "
            "Methodology, thresholds, and data-sourcing notes: "
            "research/macro_regime/README.md."
        )

    st.divider()
    ext_signals, ext_error = load_external_signals()
    if ext_error:
        st.warning(f"⚠ External signals degraded — {ext_error}")
    elif ext_signals is not None:
        _render_external_signals(ext_signals)


def render_macro_trading_page() -> None:
    st.header("Macro Trading")

    st.markdown("### Methodology")
    st.markdown(
        "1. Define market regime classifications (as many as make sense)\n\n"
        "2. Detect which regime we're in/entering (coincident labeling first, "
        "predictive detection second — separate difficulty levels)"
    )
    st.caption(
        "⚠ GDP growth-direction prediction: CLOSED search, unsolved with free data "
        "(5 candidates tested — GDPNow, SPF-RGDP consensus/TOP-1/TOP-5-avg, NY Fed "
        "Staff Nowcast — none cleared Bonferroni(5)=0.01). Do not re-attempt a 6th "
        "growth nowcast/consensus candidate without reading "
        "research/macro_regime/README.md's 'Final verdict on GDP growth prediction' "
        "section first. Inflation prediction (Cleveland Fed Nowcast) IS validated — "
        "see the External signals section below."
    )

    _render_regime_classifier_section()

    st.markdown(
        """
3. For each regime, form a qualitative hypothesis on which industries should outperform
4. Test each industry under each regime using a small-cap basket, same point-in-time/DSR/cost discipline as Phase 3
5. Build the sector-regime sensitivity matrix from confirmed results
6. Backtest actual strategies using the matrix (watch for regime-switching turnover/cost drag)
7. Build a live regime read (current regime, for reference)
8. Paper trade anything that clears backtesting
"""
    )
