"""
Reusable strategy validation page.

Takes a config dict (from any strategy's config.py) and renders the full
7-layer validation pipeline. Adding a new strategy means creating a new
config.py — no layout code is duplicated.

Usage:
    from strategies.private_competitor_distress.config import STRATEGY_CONFIG
    from strategies.strategy_page import render_strategy_page
    render_strategy_page(STRATEGY_CONFIG)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────

_DARK = "plotly_dark"
_PLACEHOLDER_SEED = 42
_STATUS_COLOURS = {
    "not_started": "🔘",
    "in_progress":  "🟡",
    "passed":       "🟢",
    "failed":       "🔴",
}
_STATUS_LABELS = {
    "not_started": "Not Started",
    "in_progress":  "In Progress",
    "passed":       "Passed",
    "failed":       "Failed",
}


# ── Placeholder chart helpers ─────────────────────────────────────────────────

def _equity_curve(seed: int = _PLACEHOLDER_SEED, n: int = 252,
                  drift: float = 0.0003) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    curve = np.cumprod(1 + rets)
    return np.arange(n), curve


def _drawdown_series(curve: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(curve)
    return (curve - peak) / peak * 100


def _fig_equity(x, curve, title: str = "Equity Curve (PLACEHOLDER)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=curve, mode="lines", name="Strategy",
        line=dict(color="#00d4aa", width=1.5),
    ))
    fig.update_layout(
        template=_DARK, title=title,
        xaxis_title="Trading Days", yaxis_title="Cumulative Return",
        height=280, margin=dict(t=40, b=30, l=50, r=20),
        showlegend=False,
    )
    return fig


def _fig_drawdown(x, dd, title: str = "Drawdown (PLACEHOLDER)") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=dd, fill="tozeroy", mode="lines", name="Drawdown",
        line=dict(color="#ff4b4b", width=1),
        fillcolor="rgba(255,75,75,0.25)",
    ))
    fig.update_layout(
        template=_DARK, title=title,
        xaxis_title="Trading Days", yaxis_title="Drawdown (%)",
        height=200, margin=dict(t=40, b=30, l=50, r=20),
        showlegend=False,
    )
    return fig


def _fig_mcpt_histogram(real_sharpe: float = 0.82, seed: int = _PLACEHOLDER_SEED,
                         n: int = 1000, title: str = "MCPT — Permuted Sharpes (PLACEHOLDER)"
                         ) -> go.Figure:
    rng = np.random.default_rng(seed)
    perm_sharpes = rng.normal(0.0, 0.25, n)
    p_value = float(np.mean(perm_sharpes >= real_sharpe))

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=perm_sharpes, nbinsx=50, name="Permuted",
        marker_color="rgba(99,110,250,0.7)",
    ))
    fig.add_vline(
        x=real_sharpe, line_dash="dash", line_color="#00d4aa",
        annotation_text=f"Real Sharpe {real_sharpe:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        template=_DARK, title=title,
        xaxis_title="Sharpe Ratio", yaxis_title="Count",
        height=280, margin=dict(t=40, b=30, l=50, r=20),
        showlegend=False,
    )
    return fig, p_value


def _fig_walk_forward(n_windows: int = 8, seed: int = _PLACEHOLDER_SEED
                      ) -> tuple[go.Figure, float]:
    rng = np.random.default_rng(seed)
    windows = np.arange(1, n_windows + 1)
    is_sharpes  = rng.uniform(0.6, 1.4, n_windows)
    oos_sharpes = is_sharpes * rng.uniform(0.4, 0.9, n_windows)
    efficiency  = float(np.mean(oos_sharpes) / np.mean(is_sharpes))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=windows, y=is_sharpes,  name="In-Sample",
                         marker_color="rgba(99,110,250,0.8)"))
    fig.add_trace(go.Bar(x=windows, y=oos_sharpes, name="Out-of-Sample",
                         marker_color="rgba(0,212,170,0.8)"))
    fig.update_layout(
        template=_DARK,
        title="Walk-Forward Sharpe per Window (PLACEHOLDER)",
        xaxis_title="Window", yaxis_title="Sharpe Ratio",
        barmode="group", height=280,
        margin=dict(t=40, b=30, l=50, r=20),
    )
    return fig, efficiency


def _fig_crisis(period: str, seed: int, drawdown_pct: float) -> go.Figure:
    rng = np.random.default_rng(seed)
    n = 120
    shock_start = 30
    shock_len   = 40
    base = rng.normal(0.0002, 0.008, n)
    base[shock_start : shock_start + shock_len] = rng.normal(
        -abs(drawdown_pct) / (shock_len * 100), 0.018, shock_len
    )
    curve = np.cumprod(1 + base)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(n), y=curve, mode="lines",
        line=dict(color="#00d4aa", width=1.5),
    ))
    fig.update_layout(
        template=_DARK, title=period,
        height=220, margin=dict(t=40, b=20, l=40, r=10),
        showlegend=False,
        xaxis=dict(showticklabels=False),
    )
    return fig


# ── Live data section renderers ───────────────────────────────────────────────

def _render_bar_ticker(df: pd.DataFrame, description: str) -> None:
    st.caption(description)
    df_sorted = df.sort_values("avg_distress_score", ascending=True)
    colours = [
        "#ff4b4b" if s >= 0.35 else "#ffa500" if s >= 0.15 else "#00d4aa"
        for s in df_sorted["avg_distress_score"]
    ]
    fig = go.Figure(go.Bar(
        x=df_sorted["avg_distress_score"],
        y=df_sorted["ticker"],
        orientation="h",
        marker_color=colours,
        customdata=np.stack([
            df_sorted["n_distressed"].values,
            df_sorted["n_competitors"].values,
            df_sorted["max_distress_score"].values,
            df_sorted["most_distressed_co"].values,
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg score: %{x:.3f}<br>"
            "Distressed: %{customdata[0]}/%{customdata[1]}<br>"
            "Max score: %{customdata[2]:.3f}<br>"
            "Top distressed: %{customdata[3]}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0.35, line_dash="dash", line_color="rgba(255,75,75,0.6)",
                  annotation_text="Signal threshold (0.35)")
    fig.update_layout(
        template=_DARK,
        title="Avg Competitor Distress Score by Listed Ticker",
        xaxis_title="Average Distress Score",
        height=max(300, len(df_sorted) * 26 + 80),
        margin=dict(t=50, b=40, l=80, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Signal callout
    signals = df[df["signal"] == "LONG"]
    if not signals.empty:
        tickers = ", ".join(signals["ticker"].tolist())
        st.error(f"LONG signal active: **{tickers}**  (avg distress ≥ 0.35)")
    else:
        st.info("No LONG signals currently active (all tickers below 0.35 threshold).")


def _render_table_companies(df: pd.DataFrame, description: str) -> None:
    st.caption(description)

    # Filter controls
    c1, c2 = st.columns([2, 1])
    min_score = c1.slider("Minimum distress score", 0.0, 1.0, 0.0, 0.01,
                          key="co_score_filter")
    tickers_available = sorted(
        {t.strip() for ts in df["tickers"].str.split(",") for t in ts}
    )
    ticker_filter = c2.multiselect("Filter by ticker", tickers_available,
                                   key="co_ticker_filter")

    display = df[df["score"] >= min_score].copy()
    if ticker_filter:
        mask = display["tickers"].apply(
            lambda ts: any(t.strip() in ticker_filter for t in ts.split(","))
        )
        display = display[mask]

    display = display.sort_values("score", ascending=False)

    cols_show = ["name", "ch_number", "tickers", "score", "days_late",
                 "company_status", "period_end", "flags"]
    st.dataframe(
        display[cols_show],
        use_container_width=True,
        hide_index=True,
        column_config={
            "score": st.column_config.ProgressColumn(
                "score",
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            ),
        },
    )
    st.caption(f"{len(display)} companies shown.")


def _render_data_sections(config: dict) -> None:
    """Render live data sections (tabular/chart views of pipeline outputs)."""
    sections = config.get("data_sections", [])
    if not sections:
        return

    results_dir: Path = config.get("results_dir", Path())
    any_data = any(
        (results_dir / s["file"]).exists() for s in sections
    )
    if not any_data:
        return

    st.divider()
    st.subheader("Live Signal Data")

    tabs = [s["title"] for s in sections if (results_dir / s["file"]).exists()]
    if not tabs:
        return

    tab_widgets = st.tabs(tabs)
    tab_idx = 0
    for section in sections:
        path = results_dir / section["file"]
        if not path.exists():
            continue
        with tab_widgets[tab_idx]:
            df = pd.read_csv(path)
            chart_type = section.get("chart_type", "table")
            desc = section.get("description", "")
            if chart_type == "bar_ticker":
                _render_bar_ticker(df, desc)
            elif chart_type == "table_companies":
                _render_table_companies(df, desc)
            else:
                st.caption(desc)
                st.dataframe(df, use_container_width=True, hide_index=True)
        tab_idx += 1


# ── Status badge row ──────────────────────────────────────────────────────────

def _render_status_badges(layers: list[dict]) -> None:
    cols = st.columns(len(layers))
    for col, layer in zip(cols, layers):
        status = layer["status"]
        icon   = _STATUS_COLOURS[status]
        label  = _STATUS_LABELS[status]
        col.markdown(
            f"<div style='text-align:center; padding:6px; border-radius:6px;"
            f"background:rgba(255,255,255,0.04)'>"
            f"<div style='font-size:1.3em'>{icon}</div>"
            f"<div style='font-size:0.7em; color:#aaa; margin-top:2px'>Layer {layer['number']}</div>"
            f"<div style='font-size:0.75em; font-weight:600'>{layer['name']}</div>"
            f"<div style='font-size:0.7em; color:#aaa'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Section wrapper ───────────────────────────────────────────────────────────

def _layer_header(layer: dict) -> bool:
    """Renders layer title + description. Returns True if results exist."""
    st.divider()
    status = layer["status"]
    icon   = _STATUS_COLOURS[status]
    label  = _STATUS_LABELS[status]
    st.subheader(f"Layer {layer['number']}: {layer['name']}  {icon} {label}")
    st.caption(layer["description"])

    results_dir: Path = layer.get("results_dir")
    has_results = (
        results_dir is not None
        and results_dir.exists()
        and any(results_dir.iterdir())
    )
    return has_results


def _placeholder_gate(layer: dict) -> bool:
    """
    If no results exist: show warning + checkbox to toggle placeholder charts.
    Returns True if placeholder charts should be rendered.
    """
    st.warning(f"Layer {layer['number']} has not been run yet. No results in results/.")
    return st.checkbox(
        "Show example layout with placeholder data",
        key=f"placeholder_{layer['id']}",
        value=False,
    )


# ── Layer renderers ───────────────────────────────────────────────────────────

def _render_layer1(layer: dict) -> None:
    # Real results live in the top-level results dir (parent of the layer subdir)
    top_results  = layer.get("results_dir", Path()).parent
    metrics_path = top_results / "layer1_metrics.json"
    curve_path   = top_results / "layer1_equity_curve.csv"
    events_path  = top_results / "event_study_results.csv"
    has_real = metrics_path.exists() and curve_path.exists()

    # Pass the right dir so the header badge doesn't say "not run"
    _layer_header({**layer, "results_dir": top_results if has_real else layer.get("results_dir")})

    if not has_real:
        if not _placeholder_gate(layer):
            return
        x, curve = _equity_curve()
        dd = _drawdown_series(curve)
        cagr   = float(curve[-1] ** (252 / len(curve)) - 1)
        mdd    = float(dd.min())
        rets   = np.diff(curve) / curve[:-1]
        sharpe = float(np.mean(rets) / np.std(rets) * np.sqrt(252))
        wins   = float(np.mean(rets > 0) * 100)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Sharpe Ratio", f"{sharpe:.2f}", help="PLACEHOLDER")
        m2.metric("Max Drawdown", f"{mdd:.1f}%",   help="PLACEHOLDER")
        m3.metric("Win Rate",     f"{wins:.1f}%",  help="PLACEHOLDER")
        m4.metric("Total Trades", "247",           help="PLACEHOLDER")
        m5.metric("CAGR",         f"{cagr:.1%}",  help="PLACEHOLDER")
        st.plotly_chart(_fig_equity(x, curve), use_container_width=True)
        st.plotly_chart(_fig_drawdown(x, dd),  use_container_width=True)
        return

    # ── Real results ──────────────────────────────────────────────────────────
    metrics  = json.loads(metrics_path.read_text())
    curve_df = pd.read_csv(curve_path, parse_dates=["date"])

    sharpe = metrics["sharpe"]
    mdd    = metrics["max_drawdown"]
    wr     = metrics["win_rate"]
    trades = metrics["total_trades"]
    cagr   = metrics["cagr"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m2.metric("Max Drawdown", f"{mdd:.1f}%")
    m3.metric("Win Rate",     f"{wr:.1f}%")
    m4.metric("Total Trades", str(trades))
    m5.metric("CAGR",         f"{cagr:.2f}%")

    # Equity curve
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(
        x=curve_df["date"], y=curve_df["equity"],
        mode="lines", name="Strategy",
        line=dict(color="#00d4aa", width=1.5),
    ))
    fig_eq.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig_eq.update_layout(
        template=_DARK,
        title="In-Sample Backtest — Cumulative Abnormal Return vs Peer Basket",
        xaxis_title="Date", yaxis_title="Cumulative Return (abnormal)",
        height=280, margin=dict(t=40, b=30, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown
    eq_arr = curve_df["equity"].values
    peak   = np.maximum.accumulate(eq_arr)
    dd_arr = (eq_arr - peak) / peak * 100
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=curve_df["date"], y=dd_arr,
        fill="tozeroy", mode="lines", name="Drawdown",
        line=dict(color="#ff4b4b", width=1),
        fillcolor="rgba(255,75,75,0.25)",
    ))
    fig_dd.update_layout(
        template=_DARK, title="Drawdown",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=200, margin=dict(t=40, b=30, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    st.caption(metrics.get("annotation", ""))

    # Event study table
    if events_path.exists():
        st.markdown("**Event Study — Cumulative Abnormal Returns**")
        ev_df = pd.read_csv(events_path, parse_dates=["event_date", "entry_date"])
        ev_disp = ev_df[["source", "event_date", "ticker", "trigger", "note", "n_days",
                          "car_21", "car_63", "car_126"]].copy()
        for col in ("car_21", "car_63", "car_126"):
            ev_disp[col] = ev_disp[col].map(
                lambda x: f"{x:.1%}" if pd.notna(x) else ""  # noqa: B023
            )
        ev_disp["event_date"] = ev_disp["event_date"].dt.date
        st.dataframe(ev_disp, use_container_width=True, hide_index=True)


def _render_layer2(layer: dict, real_sharpe: float = 0.82) -> None:
    has = _layer_header(layer)
    if not has:
        if not _placeholder_gate(layer):
            return

    fig, p_value = _fig_mcpt_histogram(
        real_sharpe=real_sharpe,
        title="In-Sample MCPT — Permuted Sharpes (PLACEHOLDER)",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("p-value", f"{p_value:.3f}", help="PLACEHOLDER")
        threshold = 0.05
        if p_value < threshold:
            st.success(f"p < {threshold} — would pass")
        else:
            st.error(f"p ≥ {threshold} — would fail")


def _render_layer3(layer: dict) -> None:
    has = _layer_header(layer)
    if not has:
        if not _placeholder_gate(layer):
            return

    fig, efficiency = _fig_walk_forward()
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("Efficiency Ratio",
                  f"{efficiency:.2f}",
                  help="OOS Sharpe / IS Sharpe. PLACEHOLDER")
        if efficiency >= 0.5:
            st.success("≥ 0.5 — would pass")
        else:
            st.error("< 0.5 — would fail")


def _render_layer4(layer: dict) -> None:
    has = _layer_header(layer)
    if not has:
        if not _placeholder_gate(layer):
            return

    fig, p_value = _fig_mcpt_histogram(
        real_sharpe=0.61, seed=99,
        title="Walk-Forward MCPT — Permuted OOS Sharpes (PLACEHOLDER)",
    )
    c1, c2 = st.columns([3, 1])
    with c1:
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.metric("p-value (OOS)", f"{p_value:.3f}", help="PLACEHOLDER")


def _render_layer5(layer: dict) -> None:
    has = _layer_header(layer)
    if not has:
        if not _placeholder_gate(layer):
            return

    raw_sharpe      = 0.82
    n_trials        = 12
    deflated_sharpe = raw_sharpe * (1 - 0.14 * np.log(n_trials))
    psr             = float(scipy_norm_cdf(deflated_sharpe))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Trials Logged",      str(n_trials),            help="PLACEHOLDER")
    m2.metric("Raw Sharpe",         f"{raw_sharpe:.2f}",      help="PLACEHOLDER")
    m3.metric("Deflated Sharpe",    f"{deflated_sharpe:.2f}", help="PLACEHOLDER")
    m4.metric("PSR (probability)",  f"{psr:.1%}",             help="PLACEHOLDER")

    st.caption(
        "The Deflated Sharpe Ratio (Lopez de Prado, 2018) applies a penalty "
        f"for the {n_trials} strategy variants evaluated during development. "
        "A PSR > 95% is required to pass this layer."
    )


def scipy_norm_cdf(x: float) -> float:
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _render_layer6(layer: dict) -> None:
    has = _layer_header(layer)
    if not has:
        if not _placeholder_gate(layer):
            return

    crises = [
        ("2008 GFC",          11,  -38.2),
        ("2020 COVID",        22,  -18.5),
        ("2022 Rate Shock",   33,  -22.1),
    ]
    cols = st.columns(3)
    for col, (period, seed, dd_pct) in zip(cols, crises):
        with col:
            st.plotly_chart(
                _fig_crisis(period, seed, dd_pct),
                use_container_width=True,
            )
            st.metric("Max Drawdown", f"{dd_pct:.1f}%", help="PLACEHOLDER")


def _render_layer7(layer: dict) -> None:
    has = _layer_header(layer)
    if not has:
        if not _placeholder_gate(layer):
            return

    st.info(
        "Paper trading begins after Layers 1–6 pass. "
        "Signals will appear here once the live feed is connected."
    )
    empty_df = pd.DataFrame(columns=[
        "Date", "Ticker", "Action", "Price", "Size", "P&L", "Signal Strength",
    ])
    st.dataframe(empty_df, use_container_width=True, hide_index=True)


# ── Layer dispatch ────────────────────────────────────────────────────────────

_LAYER_RENDERERS = {
    "in_sample_backtest": _render_layer1,
    "in_sample_mcpt":     _render_layer2,
    "walk_forward":       _render_layer3,
    "walk_forward_mcpt":  _render_layer4,
    "deflated_sharpe":    _render_layer5,
    "crisis_replay":      _render_layer6,
    "paper_trading":      _render_layer7,
}


# ── Main entry point ──────────────────────────────────────────────────────────

def render_strategy_page(config: dict) -> None:
    """Render the full validation pipeline for a strategy from its config dict."""

    results_dir: Path = config.get("results_dir", Path())

    # ── 1. Header ─────────────────────────────────────────────────────────────
    st.title(config["name"])
    st.markdown(config["description"])

    # ── 2. Status badges ──────────────────────────────────────────────────────
    st.markdown("**Validation pipeline status**")
    _render_status_badges(config["validation_layers"])

    # ── 3. Signal code expander ───────────────────────────────────────────────
    signal_file: Path = config.get("signal_file")
    if signal_file and signal_file.exists():
        with st.expander("View Signal Code", expanded=False):
            st.code(signal_file.read_text(), language="python")

    # ── 4. Live data sections (pipeline outputs) ──────────────────────────────
    _render_data_sections(config)

    # ── 5. Validation layers ──────────────────────────────────────────────────
    for layer in config["validation_layers"]:
        layer_with_dir = {**layer, "results_dir": results_dir / layer["id"]}
        renderer = _LAYER_RENDERERS.get(layer["id"])
        if renderer:
            renderer(layer_with_dir)
