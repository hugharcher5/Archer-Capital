"""
Reconcile data from multiple SourceData objects into a single preferred estimate.

For each field, disagreement = (max − min) / |median| across sources
that returned a finite value.  NaN when fewer than two sources have data.
"""

from __future__ import annotations
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sources import SourceData


# Fields that are annual series (we compare on the last available value)
SERIES_FIELDS: list[str] = ["revenue", "ebit", "dep_amort", "capex"]
# Fields that are scalar point estimates
SCALAR_FIELDS: list[str] = ["diluted_shares", "total_debt", "cash", "tax_rate"]
ALL_FIELDS: list[str]    = SERIES_FIELDS + SCALAR_FIELDS


@dataclass
class ReconcileResult:
    preferred:    SourceData           # point-estimate source (Yahoo by default)
    disagreement: dict[str, float]     # field → relative spread  (NaN = can't compute)
    sources:      list[SourceData]     # all sources for reference


def _relative_spread(values: list[float]) -> float:
    """(max − min) / |median|.  Returns NaN when < 2 finite values or median ≈ 0."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    med = abs(float(np.median(finite)))
    if med < 1e-9:
        return float("nan")
    return (max(finite) - min(finite)) / med


def _last_val(s: pd.Series) -> float:
    """Most recent (last index) value, or NaN if empty."""
    return float(s.iloc[-1]) if not s.empty else float("nan")


def reconcile(sources: list[SourceData]) -> ReconcileResult:
    """
    Compare all sources field by field.

    Series fields: compare each source's last (most recent) annual value.
    Scalar fields: compare the scalar directly.

    Returns the first source as `preferred` (caller can override).
    """
    if not sources:
        raise ValueError("reconcile() requires at least one source")

    disagreement: dict[str, float] = {}

    for f in SERIES_FIELDS:
        vals = [_last_val(getattr(s, f)) for s in sources]
        disagreement[f] = _relative_spread(vals)

    for f in SCALAR_FIELDS:
        vals = [getattr(s, f) for s in sources]
        disagreement[f] = _relative_spread(vals)

    return ReconcileResult(
        preferred=sources[0],
        disagreement=disagreement,
        sources=sources,
    )
