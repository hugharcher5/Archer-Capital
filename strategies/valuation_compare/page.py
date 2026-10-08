"""
S22 on its own page: does the DCF engine pick better stocks than a simple
P/E, P/S or P/B ratio? Reuses the S22 card from the Phase 3 registry page so
the numbers live in one place.
"""

from __future__ import annotations

import streamlit as st

from strategies.phase3_registry.page import _render_s22_group


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
