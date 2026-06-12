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
    has = _layer_header(layer)
    if not has:
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
    m1.metric("Sharpe Ratio",   f"{sharpe:.2f}",  help="PLACEHOLDER")
    m2.metric("Max Drawdown",   f"{mdd:.1f}%",    help="PLACEHOLDER")
    m3.metric("Win Rate",       f"{wins:.1f}%",   help="PLACEHOLDER")
    m4.metric("Total Trades",   "247",            help="PLACEHOLDER")
    m5.metric("CAGR",           f"{cagr:.1%}",    help="PLACEHOLDER")

    st.plotly_chart(_fig_equity(x, curve), use_container_width=True)
    st.plotly_chart(_fig_drawdown(x, dd),  use_container_width=True)


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

    # ── 4. Validation layers ──────────────────────────────────────────────────
    for layer in config["validation_layers"]:
        layer_with_dir = {**layer, "results_dir": results_dir / layer["id"]}
        renderer = _LAYER_RENDERERS.get(layer["id"])
        if renderer:
            renderer(layer_with_dir)
