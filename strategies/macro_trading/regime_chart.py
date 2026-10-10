"""Plotly regime-timeline chart for the Macro Trading tab. Style matches
strategies/strategy_page.py's convention (plotly_dark template)."""

import pandas as pd
import plotly.graph_objects as go

_DARK = "plotly_dark"

REGIME_COLORS = {
    "Goldilocks": "rgba(46, 204, 113, 0.28)",
    "Reflation": "rgba(241, 196, 15, 0.28)",
    "Stagflation": "rgba(231, 76, 60, 0.28)",
    "Deflation/Risk-off": "rgba(52, 152, 219, 0.28)",
}


def _regime_streaks(timeline: pd.DataFrame) -> list[dict]:
    t = timeline.sort_values("month").reset_index(drop=True)
    streak_id = (t["regime"] != t["regime"].shift(1)).cumsum()
    streaks = []
    for _, grp in t.groupby(streak_id):
        streaks.append({
            "regime": grp["regime"].iloc[0],
            "start": grp["month"].iloc[0],
            "end": grp["month"].iloc[-1] + pd.offsets.MonthEnd(0),
        })
    return streaks


def build_regime_timeline_figure(timeline: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    for streak in _regime_streaks(timeline):
        fig.add_vrect(
            x0=streak["start"], x1=streak["end"],
            fillcolor=REGIME_COLORS.get(streak["regime"], "rgba(150,150,150,0.2)"),
            line_width=0, layer="below",
        )

    fig.add_trace(go.Scatter(
        x=timeline["month"], y=timeline["gdp_yoy_smoothed"] * 100,
        name="Real GDP YoY (2q smoothed)", mode="lines",
        line=dict(color="#5DADE2", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=timeline["month"], y=timeline["core_cpi_yoy_smoothed"] * 100,
        name="Core CPI YoY (6mo smoothed)", mode="lines",
        line=dict(color="#F5B041", width=1.8),
    ))

    precarious_months = timeline[timeline["precarious"] == True]  # noqa: E712
    if not precarious_months.empty:
        y0 = min(timeline["gdp_yoy_smoothed"].min(), timeline["core_cpi_yoy_smoothed"].min()) * 100
        fig.add_trace(go.Scatter(
            x=precarious_months["month"],
            y=[y0] * len(precarious_months),
            name="Precarious flag",
            mode="markers",
            marker=dict(color="#E74C3C", symbol="triangle-up", size=9),
        ))

    # Legend swatches for the regime background colors (vrects don't auto-legend).
    for regime, color in REGIME_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[timeline["month"].iloc[0]], y=[None],
            mode="markers", marker=dict(size=10, color=color.replace("0.28", "0.9")),
            name=regime, showlegend=True,
        ))

    fig.update_layout(
        template=_DARK,
        title="Macro Regime Timeline (coincident, monthly)",
        yaxis_title="YoY %",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40),
        height=420,
    )
    return fig
