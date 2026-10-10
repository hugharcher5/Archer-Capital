"""
Thin bridge between the Macro Trading Streamlit tab and
research/macro_regime/ (the classifier infrastructure lives there, not
under strategies/, since it's shared infrastructure for future sector-regime
trials too -- see research/macro_regime/README.md).
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_MACRO_REGIME_DIR = Path(__file__).resolve().parents[2] / "research" / "macro_regime"
_TIMELINE_PATH = _MACRO_REGIME_DIR / "results" / "regime_timeline.csv"

if str(_MACRO_REGIME_DIR) not in sys.path:
    sys.path.insert(0, str(_MACRO_REGIME_DIR))


@st.cache_data(show_spinner=False, ttl=3600)
def load_regime_section_data():
    """
    Returns (timeline_df, live_read_dict, error_str). error_str is None on
    success; on any failure (e.g. no network for FRED/Tiingo), the timeline
    still loads from its last-saved CSV if present, and error_str carries a
    message for a degraded-source banner -- matches this project's existing
    convention (dcf/portfolio/web.py's data-source-degraded banner) rather
    than crashing the tab.
    """
    timeline = None
    live = None
    error = None

    try:
        timeline = pd.read_csv(_TIMELINE_PATH, parse_dates=["month"]).sort_values("month")
    except Exception as e:
        error = f"Could not load regime_timeline.csv: {e}"

    try:
        from live_read import get_live_read
        live = get_live_read()
    except Exception as e:
        msg = f"Could not compute a fresh live regime read: {e}"
        error = f"{error}  |  {msg}" if error else msg

    return timeline, live, error


@st.cache_data(show_spinner=False, ttl=3600)
def load_external_signals():
    """
    Returns (external_signals_dict, error_str). Separate cache entry from
    Part 1's live read -- these are the external nowcast/consensus signals
    (see README's "External nowcast/consensus signals" section, 2026-07-18
    results) that cleared their own pre-registered walk-forward test.
    Failure here should never take down Part 1's display.
    """
    try:
        from live_read import get_external_signals
        return get_external_signals(), None
    except Exception as e:
        return None, f"Could not load external signals: {e}"
