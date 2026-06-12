"""
Monte Carlo consistency and sigma_cross wiring tests.

(a) Zero-variance run reproduces deterministic value() to the cent.
(b) Injecting a 10% synthetic sigma_cross on revenue_growth widens the
    P10–P90 band; P50 barely moves.
(c) A DEFINITIONAL field disagreement produces no sigma_cross entry and
    therefore cannot widen any MC distribution.

All tests are pure unit tests — no network calls, no file I/O.
"""

from __future__ import annotations
import math

import numpy as np
import pandas as pd
import pytest

from valuation.dcf import Assumptions, value, TERMINAL_G, MATURE_MARGIN_DEFAULT
from valuation.montecarlo import _run_sims, _ensure_psd, CORR, SPREAD_SIGMA
from valuation.reconcile import reconcile, FIELD_TYPE
from valuation.sources import SourceData


# ── Shared fixture ────────────────────────────────────────────────────────────

def _make_assumptions(**overrides) -> Assumptions:
    """Minimal synthetic Assumptions for a hypothetical USD company."""
    defaults = dict(
        ticker="TEST", currency="USD",
        start_revenue=100e9,
        revenue_growth=0.10,
        ebit_margin=0.25,
        target_margin=max(0.25, MATURE_MARGIN_DEFAULT),
        tax_rate=0.21,
        da_pct=0.05,
        capex_pct=0.06,
        nwc_pct=0.05,
        terminal_g=TERMINAL_G,
        wacc=0.09,
        net_debt=10e9,
        diluted_shares=1e9,
        fx_rate=1.0,
        forecast_years=5,
        rf=0.04, erp=0.045, beta_adj=1.0,
        cost_of_equity=0.085,
        cost_of_debt_pretax=0.04,
        cost_of_debt_aftertax=0.032,
        equity_weight=0.85,
        debt_weight=0.15,
        implied_rating="A",
        current_price_usd=90.0,
        current_ebitda=30e9,
        market_ev_local=400e9,
    )
    defaults.update(overrides)
    return Assumptions(**defaults)


def _corr():
    return _ensure_psd(CORR.copy())


# ── (a) Consistency check ─────────────────────────────────────────────────────

def test_zero_variance_reproduces_deterministic():
    """
    _run_sims with spread_sigma=0 must return values that all equal
    the deterministic value() to within $0.01 per share.
    Passing non-zero sigma_cross-derived sg/sm must not break this because
    spread_sigma=0 collapses all PERT bounds to point masses.
    """
    base = _make_assumptions()
    det  = value(base)

    # Run with a large sg/sm to verify spread_sigma=0 still pins to det
    sims = _run_sims(
        base,
        sg=0.30,   # large, but irrelevant when spread_sigma=0
        sm=0.15,
        corr=_corr(),
        n=50,
        spread_sigma=0.0,
        rng=np.random.default_rng(0),
        copula="gaussian",
    )

    assert len(sims) == 50, "Expected 50 draws, no skips"
    max_err = float(max(abs(v - det) for v in sims))
    assert max_err < 0.01, (
        f"Zero-variance run diverged: det={det:.4f}, max_err={max_err:.4f}"
    )


def test_sigma_cross_zero_leaves_consistency_unchanged():
    """
    When sigma_cross has no entries, σ_eff = σ_hist. The consistency check
    (spread_sigma=0) must still pass — σ_eff is irrelevant at spread_sigma=0.
    """
    base = _make_assumptions()
    det  = value(base)
    sg   = 0.05
    sm   = 0.04

    # σ_eff with no cross-source disagreement
    sg_eff = math.sqrt(sg**2 + 0.0**2)
    sm_eff = math.sqrt(sm**2 + 0.0**2)

    sims = _run_sims(base, sg_eff, sm_eff, _corr(), 50, 0.0,
                     np.random.default_rng(0), "gaussian")
    max_err = float(max(abs(v - det) for v in sims))
    assert max_err < 0.01


# ── (b) Synthetic revenue disagreement widens the band ────────────────────────

def test_revenue_disagreement_widens_p10_p90():
    """
    Injecting sigma_cross=10% on revenue_growth (combined in quadrature with
    historical σ) must widen the P10–P90 band by at least 5%.
    P50 must stay within 10% of the baseline P50 (symmetric PERT → median stable).
    """
    base   = _make_assumptions()
    sg_hist = 0.05
    sm_hist = 0.04
    n       = 5_000
    seed    = 42

    # Baseline: no cross-source input
    sims_base = _run_sims(
        base, sg_hist, sm_hist, _corr(), n, SPREAD_SIGMA,
        np.random.default_rng(seed), "gaussian",
    )

    # With 10% synthetic revenue_growth sigma_cross
    sigma_cross_rg = 0.10
    sg_eff = math.sqrt(sg_hist**2 + sigma_cross_rg**2)
    sims_wide = _run_sims(
        base, sg_eff, sm_hist, _corr(), n, SPREAD_SIGMA,
        np.random.default_rng(seed), "gaussian",
    )

    band_base = np.percentile(sims_base, 90) - np.percentile(sims_base, 10)
    band_wide = np.percentile(sims_wide, 90) - np.percentile(sims_wide, 10)
    p50_base  = float(np.median(sims_base))
    p50_wide  = float(np.median(sims_wide))

    assert band_wide > band_base * 1.05, (
        f"P10–P90 should widen by >5%: baseline={band_base:.2f}, "
        f"widened={band_wide:.2f} ({band_wide/band_base:.1%})"
    )
    assert abs(p50_wide - p50_base) / p50_base < 0.10, (
        f"P50 moved too much: {p50_base:.2f} → {p50_wide:.2f} "
        f"({abs(p50_wide-p50_base)/p50_base:.1%})"
    )


# ── (c) DEFINITIONAL field disagreement does not produce sigma_cross ──────────

def test_definitional_disagreement_does_not_enter_sigma_cross():
    """
    Two sources that disagree ONLY on a DEFINITIONAL field (dep_amort here,
    100% gap) must produce zero sigma_cross entries for MC driver variables.

    Concretely: reconcile() must not put 'dep_amort' into sigma_cross, and
    the revenue_growth / ebit_margin sigma_cross must stay at 0 because both
    sources agree on all DATA fields.
    """
    ts = pd.Timestamp("2024-06-30")
    rev   = pd.Series({ts: 100e9})
    ebit  = pd.Series({ts: 25e9})
    capex = pd.Series({ts: 5e9})

    # Same data everywhere except dep_amort (DEFINITIONAL) differs 100%
    da_low  = pd.Series({ts: 3e9})
    da_high = pd.Series({ts: 6e9})

    s_yahoo = SourceData("Yahoo", "USD", rev, ebit, da_low,  capex, 1e9, 10e9, 5e9, 0.21)
    s_edgar = SourceData("EDGAR", "USD", rev, ebit, da_high, capex, 1e9, 10e9, 5e9, 0.21)

    result = reconcile([s_yahoo, s_edgar])

    # dep_amort is DEFINITIONAL — must not appear in sigma_cross
    assert "dep_amort" not in result.sigma_cross, (
        "DEFINITIONAL field dep_amort must not produce a sigma_cross entry"
    )
    assert FIELD_TYPE["dep_amort"] == "DEFINITIONAL"
    assert FIELD_TYPE["total_debt"] == "DEFINITIONAL"
    assert FIELD_TYPE["tax_rate"]   == "DEFINITIONAL"

    # Revenue agrees exactly → revenue_growth sigma_cross must be ~0
    rg_cross = result.sigma_cross.get("revenue_growth", 0.0)
    assert rg_cross < 1e-9, (
        f"revenue_growth sigma_cross should be 0 when sources agree: {rg_cross}"
    )

    # EBIT agrees exactly → ebit_margin sigma_cross must be ~0
    em_cross = result.sigma_cross.get("ebit_margin", 0.0)
    assert em_cross < 1e-9, (
        f"ebit_margin sigma_cross should be 0 when sources agree: {em_cross}"
    )


def test_data_disagreement_enters_sigma_cross():
    """
    Two sources that disagree on EBIT (DATA field) must produce a non-zero
    ebit_margin sigma_cross.
    """
    ts    = pd.Timestamp("2024-06-30")
    rev   = pd.Series({ts: 100e9})
    da    = pd.Series({ts: 5e9})
    capex = pd.Series({ts: 5e9})

    ebit_lo = pd.Series({ts: 20e9})   # margin = 20%
    ebit_hi = pd.Series({ts: 30e9})   # margin = 30%

    s1 = SourceData("Yahoo", "USD", rev, ebit_lo, da, capex, 1e9, 10e9, 5e9, 0.21)
    s2 = SourceData("EDGAR", "USD", rev, ebit_hi, da, capex, 1e9, 10e9, 5e9, 0.21)

    result = reconcile([s1, s2])

    em_cross = result.sigma_cross.get("ebit_margin", 0.0)
    # std([0.20, 0.30]) = 0.05
    assert abs(em_cross - 0.05) < 1e-9, (
        f"ebit_margin sigma_cross expected 0.05, got {em_cross}"
    )
