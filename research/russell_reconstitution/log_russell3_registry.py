"""
Log Russell-3 to the main program trial_registry.csv.

Family member of "Russell Reconstitution" (Russell-1, trial #17) -- reuses
the SAME trial_number (17), not a suffixed variant. This follows the
program's existing convention for genuinely distinct sub-hypotheses within
one behavioral family sharing a DSR slot (e.g. S1 IVOL / S4 MAX both use
trial_number=7; S9/S6 momentum both use trial_number=6) -- as opposed to the
"8b"/"8b-WF"-style suffixes used for frequency-SWEEP variants of the SAME
trial (S3-Q/S3-SA/S3-A), which is a different kind of variant.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from run_russell3 import main as run_russell3_main

REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

TRIAL_NUMBER = 17
N_TRIALS = 17  # same family slot as Russell-1, does not increment


def log_registry(result: dict) -> None:
    pooled, additions, deletions = result["pooled"], result["additions"], result["deletions"]

    notes = (
        f"POOLED N={pooled['n']} CAAR={pooled['caar']:+.4%} t={pooled['t_stat']:+.3f} "
        f"win={pooled['win_rate']:.1%} | "
        f"ADDITIONS-ONLY N={additions['n']} CAAR={additions['caar']:+.4%} t={additions['t_stat']:+.3f} "
        f"(wrong-signed vs hypothesis: expected positive) | "
        f"DELETIONS-ONLY N={deletions['n']} CAAR={deletions['caar']:+.4%} t={deletions['t_stat']:+.3f} "
        f"(wrong-signed vs hypothesis: expected negative, marginal p=0.076) | "
        f"No subset materially different from Russell-1's null. "
        f"RECOMMENDATION: close the Russell Reconstitution family."
    )

    row = {
        "timestamp": datetime.now().isoformat(),
        "study": "russell3_boundary_crossing",
        "hypothesis": (
            "Boundary-crossing subset: companies that cross the Russell index-inclusion "
            "boundary (Microcap<->2000 or 2000<->1000) multiple times across 2016-2023 isolate "
            "the pure membership/passive-flow effect more cleanly than the full additions list, "
            "since each company serves as its own control across its own repeated crossings. "
            "Family member of 'Russell Reconstitution' (Russell-1, trial #17) -- shares its DSR "
            "slot; a distinct sub-hypothesis, not a re-tuned version of Russell-1."
        ),
        "data_source": (
            "Reused Russell-1's existing sourced dataset (Wayback-archived FTSE Russell PDFs, "
            "2016-2023) + Tiingo adjClose/close + IWM benchmark -- no new data acquisition. "
            "Added a deletions-side universe (build_russell3.py) parallel to Russell-1's "
            "additions-only universe; r3000 deletions unavailable for 2018/2019 (gap, flagged)."
        ),
        "status": "completed_in_sample",
        "trial_number": TRIAL_NUMBER,
        "trial_note": (
            "Family member of Russell Reconstitution (trial #17), same DSR slot as Russell-1. "
            "Event-study design, no Sharpe/DSR gate -- CAAR t-stat per subset is the "
            "significance test. N reported as crossing EVENTS, not unique tickers (748 unique "
            "repeat-crosser tickers, 1,751 total crossing events, 1,493 priced)."
        ),
        "n_trials_dsr": N_TRIALS,
        "ann_return": None, "sharpe": None, "calmar": None, "max_drawdown": None,
        "t_stat": pooled["t_stat"],
        "market_beta": None, "mean_ic": None, "ic_t_stat": None,
        "win_rate": pooled["win_rate"],
        "n_periods": pooled["n"],
        "obs_sr_q": None, "dsr_threshold": None, "clears_dsr": False,
        "market_beta_tstat": None, "mean_long_scale": None, "mean_short_scale": None,
        "mean_beta_long": None, "mean_beta_short": None,
        "notes": notes,
        "rebalance_freq": "event-level (one entry+exit per crossing, not one per year)",
        "rebalance_note": (
            "Unlike Russell-1 (one basket per year), each repeat-crosser contributes one event "
            "per year it crosses (as addition or deletion); entry/exit dates and IWM benchmark "
            "window per event are Russell-1's cycle_calendar dates for that event's year, unchanged."
        ),
        "portfolio_note": (
            "Long-only window return (px[exit]/px[entry]-1) minus IWM, same sign convention for "
            "additions and deletions (not flipped) -- reported separately (pooled/additions/"
            "deletions) precisely because the two are expected to have opposite signs."
        ),
        "ann_return_net": None, "ann_return_gross": pooled["caar"],
        "annual_cost_drag_bps": None, "mean_turnover": None,
        "obs_sr_monthly": None, "adj_fallback_pct": None,
        "annual_ba_bps": None, "annual_bw_bps": None, "per_period_sr": None,
        "signal_note": (
            "748 repeat-crosser tickers found (appear in 2+ distinct years of the combined "
            "additions+deletions universe) -- NOT a thin sample (contrary to the pre-registered "
            "expectation this might be under 10); ~39% of the full universe are repeat-crossers, "
            "consistent with well-documented Russell 2000/Microcap boundary churn. Concentration "
            "check clean (top-2 tickers <5% of |abnormal return| mass in every subset)."
        ),
        "mean_pct_dropped": None, "hold_days": None, "n_events": pooled["n"], "guard_fire_days": None,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "russell3_boundary_crossing"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"\n[Registry] russell3_boundary_crossing logged (trial #{TRIAL_NUMBER}, "
          f"same 'Russell Reconstitution' family slot as Russell-1) -> {REGISTRY}")


if __name__ == "__main__":
    result = run_russell3_main()
    log_registry(result)
