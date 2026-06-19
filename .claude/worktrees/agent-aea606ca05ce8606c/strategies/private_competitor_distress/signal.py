"""
Private Competitor Distress — signal entry point.

Thin wrapper that delegates to data_pipelines.distress_pipeline.
Call run() to fetch fresh CH data and recompute scores.

Tunable parameters
------------------
DISTRESS_THRESHOLD   Minimum composite score to flag a competitor as distressed.
                     Scores are in [0, 1].  Default 0.35.
HOLD_PERIOD          Max holding period in calendar days.  Default 90.
MAX_POSITION_PCT     Maximum portfolio weight per ticker.  Default 0.05 (5%).
MIN_MARKET_CAP_GBP   Micro-cap filter.  Default £50M.
MAX_SPREAD_PCT       Bid-ask spread filter.  Default 1.5%.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make data_pipelines importable when this module is run directly
_REPO = Path(__file__).parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from data_pipelines.distress_pipeline import run_pipeline  # noqa: E402

DISTRESS_THRESHOLD  = 0.35
HOLD_PERIOD         = 90
MAX_POSITION_PCT    = 0.05
MIN_MARKET_CAP_GBP  = 50_000_000
MAX_SPREAD_PCT      = 0.015


def run(dry_run: bool = False) -> dict:
    """
    Execute the full pipeline and return results dict.

    Returns
    -------
    {
        "company_scores": list[dict],
        "ticker_signals": list[dict],
    }
    CSVs are also written to strategies/private_competitor_distress/results/.
    """
    return run_pipeline(dry_run=dry_run)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(dry_run=args.dry_run)
