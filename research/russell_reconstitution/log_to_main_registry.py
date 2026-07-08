"""
Phase 5b: Log Russell-1 to the main program trial_registry.csv.

run_backtest.py's docstring commits to "trial-registry logging" as part of
Phase 4-5 but the script itself only writes local CSVs (russell1_cycle_results.csv,
russell1_ticker_returns.csv) and returns a summary dict without persisting it
to the main registry. This script completes that step, matching every other
trial script in the program (log regardless of outcome).

New independent family: "Russell Reconstitution". Russell-1 is the first
trial in it -> trial #17 (next after accruals_S11, #16). Event-study design
(same class as H1 livestock), so DSR/Sharpe fields are n/a by convention,
matching how H1 is recorded.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from run_backtest import main as run_backtest_main

REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

TRIAL_NUMBER = 17
N_TRIALS = 17  # new independent family; first trial in it


def log_registry(result: dict) -> None:
    verdict = " ".join(result["verdict_lines"])
    row = {
        "timestamp":     datetime.now().isoformat(),
        "study":         "russell1_anticipatory_drift",
        "hypothesis":    (
            "Pre-effective-date anticipatory drift: stocks confirmed for addition to a "
            "Russell US Index (Microcap->2000 or 2000->1000 migration, or new-to-index via "
            "IPO) drift upward between the preliminary-list announcement date and the "
            "reconstitution effective date, as traders anticipate forced mechanical buying "
            "from index-benchmarked funds. Pure-flow hypothesis, no fundamental mechanism. "
            "First trial in a new independent family: 'Russell Reconstitution'."
        ),
        "data_source":   (
            "FTSE Russell / Russell Investments primary-source addition/deletion PDFs, "
            "archived via Wayback Machine CDX (2016-2023; 2015 excluded, only press-release "
            "'highlights' docs recoverable that year, not ticker tables) + Tiingo adjClose/close "
            "+ IWM benchmark."
        ),
        "status":        "completed_in_sample",
        "trial_number":  TRIAL_NUMBER,
        "trial_note":    (
            "New independent family (Russell Reconstitution), trial #17. Event-study design "
            "(same class as H1 livestock) -- no Sharpe/DSR gate; CAAR t-stat is the significance "
            "test. 2024-2025 reserved untouched holdout, NOT unlocked per this trial's own "
            "recommendation (baseline null)."
        ),
        "n_trials_dsr":  N_TRIALS,
        "ann_return":    None,
        "sharpe":        None,
        "calmar":        None,
        "max_drawdown":  None,
        "t_stat":        result["t_stat"],
        "market_beta":   None,
        "mean_ic":       None,
        "ic_t_stat":     None,
        "win_rate":      result["win_rate"],
        "n_periods":     result["n"],
        "obs_sr_q":      None,
        "dsr_threshold": None,
        "clears_dsr":    False,
        "market_beta_tstat": None,
        "mean_long_scale":   None,
        "mean_short_scale":  None,
        "mean_beta_long":    None,
        "mean_beta_short":   None,
        "notes":         verdict,
        "rebalance_freq":  "annual (1 cycle/year, buy-and-hold window)",
        "rebalance_note":  "One entry + one exit per name per year; 8 cycles, 2016-2023.",
        "portfolio_note":  "Equal-weight, long-only basket of confirmed Russell 3000 additions per cycle.",
        "ann_return_net":     result["caar_net"],
        "ann_return_gross":   result["caar_gross"],
        "annual_cost_drag_bps": 50.0,  # flat 50bps round-trip, once per name per cycle (not annualized turnover)
        "mean_turnover":      None,
        "obs_sr_monthly":     None,
        "adj_fallback_pct":   None,
        "annual_ba_bps":      None,
        "annual_bw_bps":      None,
        "per_period_sr":      None,
        "signal_note": (
            "CAAR (event-study abnormal return) = mean(basket_return - IWM_return) across 8 "
            "yearly cycles. Entry = first trading day after preliminary list posted; "
            "exit = historical effective date (last Friday of June, except 2023 schedule shift)."
        ),
        "mean_pct_dropped":  1.0 - result.get("mean_coverage", float("nan")) if "mean_coverage" in result else None,
        "hold_days":     None,
        "n_events":      None,
        "guard_fire_days": None,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "russell1_anticipatory_drift"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"\n[Registry] russell1_anticipatory_drift logged (trial #{TRIAL_NUMBER}, "
          f"new independent family 'Russell Reconstitution') -> {REGISTRY}")


if __name__ == "__main__":
    result = run_backtest_main()
    log_registry(result)
