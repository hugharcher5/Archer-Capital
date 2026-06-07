"""Structured result object shared by terminal and Streamlit consumers."""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .drivers   import Drivers
from .wacc      import WACCResult
from .dcf       import Assumptions, DCFResult


@dataclass
class ValuationResult:
    # ── Identity ──────────────────────────────────────────────────────────────
    ticker:            str
    currency:          str
    current_price_usd: float

    # ── Computation artifacts (for expanders / terminal print) ────────────────
    drivers:     Drivers
    wacc_result: WACCResult
    assumptions: Assumptions
    dcf:         DCFResult

    # ── Monte Carlo outputs (sims is empty array when n_sims=0) ───────────────
    sims:            np.ndarray   # full un-winsorized array, shape (n_valid,)
    n_valid:         int
    copula_label:    str
    consistency_pass: bool
    cc_p50:          float        # zero-variance run median (for terminal display)

    # ── Pre-computed stats (NaN when no sims) ─────────────────────────────────
    p10:             float
    p25:             float
    p50:             float
    p75:             float
    p90:             float
    mean_val:        float
    std_val:         float
    pct_undervalued: float        # percentage of draws > current price
