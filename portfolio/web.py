import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
from portfolio.data import fetch_portfolio

st.set_page_config(page_title="Archer Capital", layout="wide")
st.title("Archer Capital — Portfolio Dashboard")

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()


@st.cache_data(ttl=60)
def get_data():
    return fetch_portfolio()


rows, totals = get_data()

# --- Positions table ---
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

# --- Totals summary ---
col1, col2, col3 = st.columns(3)
col1.metric("Market Value",  f"${totals['market_value']:,.0f}")
col2.metric("Unrealized P&L", f"${totals['unreal_pnl']:+,.0f}")
col3.metric("Daily P&L",     f"${totals['day_pnl']:+,.0f}")

# --- Pie chart ---
st.subheader("Portfolio Weights")
fig = px.pie(
    df, values="weight", names="ticker",
    hole=0.4,
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_traces(textposition="inside", textinfo="percent+label")
fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=350)
st.plotly_chart(fig, use_container_width=True)
