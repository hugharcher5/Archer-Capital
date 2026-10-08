"""
S22 on its own page: does the DCF engine pick better stocks than a simple
P/E, P/S or P/B ratio? Self-contained (only streamlit and pandas), and the
Phase 3 registry page imports its card from here, so the numbers live in one place.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


# ── S22 group card (long-only valuation-signal comparison, 4 sub-trials) ──────

_S22_VARIANTS = [
    {
        "id": "DCF", "label": "DCF-implied margin of safety",
        "ic": "+0.0014", "ic_t": "0.072",
        "gross": "+12.54%/yr", "cost_bps": "180", "net": "+10.59%/yr",
        "sharpe": "0.452", "calmar": "0.233", "max_dd": "−45.5%", "beta": "1.010 (t=24.5)",
        "dsr_obs": "0.226", "dsr_threshold": "0.368",
        "coverage": "63.4%", "alpha_ew": "−2.74%/yr", "alpha_tbill": "+15.53%/yr",
    },
    {
        "id": "P/E", "label": "Price / TTM net income",
        "ic": "+0.0132", "ic_t": "0.566",
        "gross": "+9.85%/yr", "cost_bps": "186", "net": "+7.88%/yr",
        "sharpe": "0.388", "calmar": "0.127", "max_dd": "−61.9%", "beta": "1.221 (t=20.7)",
        "dsr_obs": "0.194", "dsr_threshold": "0.368",
        "coverage": "58.9%", "alpha_ew": "−5.77%/yr", "alpha_tbill": "+16.32%/yr",
    },
    {
        "id": "P/S", "label": "Price / TTM revenue",
        "ic": "+0.0210", "ic_t": "0.651",
        "gross": "+14.49%/yr", "cost_bps": "144", "net": "+12.91%/yr",
        "sharpe": "0.493", "calmar": "0.225", "max_dd": "−57.3%", "beta": "1.279 (t=19.3)",
        "dsr_obs": "0.246", "dsr_threshold": "0.368",
        "coverage": "90.8%", "alpha_ew": "−1.49%/yr", "alpha_tbill": "+22.57%/yr",
    },
    {
        "id": "P/B", "label": "Price / book equity",
        "ic": "+0.0040", "ic_t": "0.156",
        "gross": "+9.64%/yr", "cost_bps": "179", "net": "+7.72%/yr",
        "sharpe": "0.389", "calmar": "0.132", "max_dd": "−58.5%", "beta": "1.246 (t=26.8)",
        "dsr_obs": "0.194", "dsr_threshold": "0.368",
        "coverage": "94.5%", "alpha_ew": "−6.03%/yr", "alpha_tbill": "+16.54%/yr",
    },
]


def _render_s22_group(expanded: bool = False) -> None:
    with st.expander(
        "S22 — Long-Only Quarterly Valuation Portfolios: DCF vs P/E vs P/S vs P/B",
        expanded=expanded,
    ):
        st.markdown(
            "Top-20%-most-undervalued, equal-weight, **long-only** portfolios (genuinely "
            "new construction vs. every other trial in this registry, which are beta-"
            "neutral long/short) — four valuation signals tested head-to-head on the same "
            "underlying question. All four share DSR trial #28 (N_TRIALS=28, quarterly "
            "threshold 0.368). Beta ≈ 1.0–1.28 (all t > 19) confirms the long-only "
            "construction carries genuine market beta by design, not a defect. Window "
            "2015-01-01→2021-01-01 (24 quarters)."
        )

        comp_rows = []
        for v in _S22_VARIANTS:
            comp_rows.append({
                "Signal":       v["id"],
                "IC (t)":       f"{v['ic']} ({v['ic_t']})",
                "Gross":        v["gross"],
                "Cost (bps/yr)": v["cost_bps"],
                "Net":          v["net"],
                "Sharpe":       v["sharpe"],
                "CALMAR":       v["calmar"],
                "Max DD":       v["max_dd"],
                "Beta (t)":     v["beta"],
                "DSR obs":      v["dsr_obs"],
                "DSR thresh":   v["dsr_threshold"],
                "Coverage":     v["coverage"],
            })
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown(
            "**All four FAIL DSR at N=28.** Alpha vs. the equal-weight universe is "
            "NEGATIVE for every signal (DCF −2.74%, P/E −5.77%, P/S −1.49%, P/B −6.03%/yr) "
            "despite strongly positive alpha vs. the 3-month T-bill (+15.5% to +22.6%/yr) — "
            "absolute returns are substantially explained by high-beta exposure to a rising "
            "2015–2020 small-cap market, not genuine alpha over that market."
        )

        st.markdown(
            "**Original (unequal-universe) ranking:** P/S > DCF > P/E ≈ P/B, with P/S "
            "best on Sharpe/IC/alpha. **This ranking did NOT survive scrutiny.** Signal "
            "coverage was very unequal each quarter (mean universe ~422 names): "
            "P/B 94.5%, P/S 90.8%, DCF 63.4%, P/E 58.9% — DCF and P/E were effectively "
            "ranking from a smaller, more-established/profitable subset (P/E structurally "
            "excludes loss-making names; DCF needs ~12 XBRL concepts plus an "
            "applicability gate vs. P/B/P/S's single-concept dependency)."
        )
        st.markdown(
            "**CORRECTION (2026-07-15) — intersection-universe re-run:** restricting all "
            "four signals to the common subset where all four have valid values each "
            "quarter (mean 179/quarter, down from ~422) made the original ranking "
            "**evaporate**: Sharpe clusters 0.41–0.43 for all four (vs. the original "
            "0.39–0.49 spread), alpha vs. market clusters −3.7% to −4.2% (all four now "
            "negative and close together), and the IC ranking **inverts** — P/E is now "
            "best (0.017), P/B flips to negative (−0.0025), P/S (the original 'winner') "
            "drops to third. **Corrected conclusion: none of the four signals is "
            "distinguishable from the others on a fair, equal-universe basis** — DCF's "
            "substantial added complexity bought no measurable edge, but also no "
            "underperformance, once the universe is held fixed. Re-run was cross-checked "
            "byte-for-byte against the original computation first (max diff 0.0 across all "
            "96 signal-periods) before trusting the corrected numbers. This is a "
            "*stronger* negative finding than the original ('simple beats complex'), not a "
            "weaker one — treat any future multi-signal head-to-head in this registry with "
            "the same coverage-parity check before ranking."
        )
        st.markdown(
            "Two infrastructure fixes made along the way, both reusable and unrelated to "
            "the ranking correction: (1) a new PIT adapter "
            "(`research/valuationcompare_S22/pit_facts.py`, `dcf_adapter.py`) feeds the "
            "existing DCF engine's already-pure computation core "
            "(`compute_drivers`/`compute_wacc`/`_build_base`/`dcf.value`) via one additive "
            "`as_of` parameter on `wacc.py`'s `_get_rf()` — no parallel WACC calculator "
            "needed. (2) `compute_wacc()` was calling a live FRED fetch on every single "
            "invocation, uncached — harmless for the live tool's one-valuation-at-a-time "
            "use but made backtesting ~65x slower than necessary; fixed with module-level "
            "memoization (3.85s/name → 0.059s/name), which also speeds up the live tool."
        )


# ── Main render ───────────────────────────────────────────────────────────────


def render_valuation_compare_page() -> None:
    st.markdown("### DCF vs. simple valuation ratios (S22)")
    st.markdown(
        "Does the DCF engine behind the valuation tool actually find better stocks than a "
        "free, spreadsheet-simple ratio? This test builds long-only portfolios of the 20% "
        "most undervalued US small caps each quarter, ranked four ways (DCF margin of "
        "safety, P/E, P/S and P/B), and compares them on the same universe, window and "
        "trading-cost model."
    )
    _render_s22_group(expanded=True)
