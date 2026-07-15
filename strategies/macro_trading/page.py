"""
Macro Trading tab.

New, separate research track (macro-regime and sector-conditional strategies),
distinct from the Phase 3 Signal Research Registry. Placeholder only for now.
"""
from __future__ import annotations
import streamlit as st


def render_macro_trading_page() -> None:
    st.header("Macro Trading")

    st.markdown("### Methodology")
    st.markdown(
        """
1. Define market regime classifications (as many as make sense)
2. Detect which regime we're in/entering (coincident labeling first, predictive detection second — separate difficulty levels)
3. For each regime, form a qualitative hypothesis on which industries should outperform
4. Test each industry under each regime using a small-cap basket, same point-in-time/DSR/cost discipline as Phase 3
5. Build the sector-regime sensitivity matrix from confirmed results
6. Backtest actual strategies using the matrix (watch for regime-switching turnover/cost drag)
7. Build a live regime read (current regime, for reference)
8. Paper trade anything that clears backtesting
"""
    )
