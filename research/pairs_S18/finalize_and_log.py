"""
Compute final summary and log S18 (pairs trading) to research/trial_registry.csv.
Re-reads trial_registry.csv's current max trial_number IMMEDIATELY before writing.
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

with open(RESULTS / "_raw_backtest.json") as f:
    raw = json.load(f)
cs = raw["cycle_summary"]

n_universe_avg = float(np.mean([c["n_universe"] for c in cs]))
tot_pre = sum(c["n_preclustering_pairs"] for c in cs)
tot_cand = sum(c["n_candidate_pairs"] for c in cs)
tot_tests = sum(c["n_cointegration_tests"] for c in cs)
tot_fdr = sum(c["n_survive_fdr"] for c in cs)

# ── Diagnostic (non-registered) supplement metrics, precomputed and hardcoded
# here from the analysis step (not recomputed at log time -- values verified
# in the conversation record) ──────────────────────────────────────────────
diag = {
    "n_trades": 16, "n_entries": 16,
    "win_rate_overall": 0.4375, "win_rate_converged": 0.625, "win_rate_forced": 0.25,
    "avg_days_converged": 27.75, "avg_days_forced": 99.75,
    "net_ann_ret": -0.044403755249284305, "net_sharpe": -0.4540188202170588,
    "net_calmar": -0.18338559986928357, "net_max_dd": -0.2421332715378694,
    "net_t_stat": -1.0236421264203575, "n_months": 61,
    "gross_ann_ret": -0.005387471507578012, "gross_sharpe": -0.01704658514881433,
    "gross_calmar": -0.029996871711350227, "gross_max_dd": -0.17960111172324342,
    "cost_drag_bps_annualized": 403.51463882272157,
    "beta_vs_iwm": -0.09747297365177228, "beta_t": -1.8095967337945231,
    "top_pair_pnl_share": 0.23727515020810683, "top_sector_pnl_share": 0.23727515020810683,
    "days_with_position": 329, "total_business_days": 1306,
}

# ── FINAL re-check of trial count -- do this LAST, right before writing ─────
reg_df = pd.read_csv(REGISTRY)
_numeric_trial_nums = pd.to_numeric(reg_df["trial_number"], errors="coerce").dropna()
current_max = int(_numeric_trial_nums.max())
NEW_TRIAL_NUMBER = current_max + 1
N_TRIALS_DSR = NEW_TRIAL_NUMBER
print(f"[Registry re-check] current max trial_number = {current_max} -> this trial = #{NEW_TRIAL_NUMBER}")
print(reg_df[["timestamp", "study", "trial_number"]].tail(6).to_string(index=False))

row = {
    "timestamp": datetime.now().isoformat(),
    "study": "pairs_S18",
    "hypothesis": (
        "Small-cap sector-bucketed pairs trading: relative-value mean-reversion "
        "between cointegrated, same-industry pairs, entered on 2-sigma spread "
        "divergence, exited on 0.5-sigma reversion or forced close. Genuinely new "
        "mechanism (relative-value between pairs), distinct from every prior "
        "cross-sectional single-factor trial in this registry. Multiple-testing "
        "control via sector bucketing + hierarchical clustering + BH-FDR-corrected "
        "Engle-Granger cointegration + a 2-consecutive-formation-period persistence "
        "filter (the direct analog of this registry's DSR discipline, applied to "
        "pair selection instead of trial selection)."
    ),
    "data_source": (
        "Tiingo (adjClose used throughout, for both signal and P&L -- disclosed "
        "deviation from this registry's usual raw-close-for-returns convention; "
        "see trial_note) + SEC SIC classification (reused from gp_industry_neutral, "
        "no new data sourcing)"
    ),
    "status": "stopped_zero_trades",
    "trial_number": NEW_TRIAL_NUMBER,
    "n_trials_dsr": N_TRIALS_DSR,
    "trial_note": (
        "New independent trial, new mechanism family. Pre-registered methodology "
        "(sector bucket -> hierarchical clustering, distance threshold 0.4/corr>=0.6, "
        "cluster size 2-8 -> Engle-Granger cointegration, BH-FDR alpha=0.05 pooled "
        "per formation cycle across all buckets -> 2-consecutive-cycle persistence "
        "filter) run exactly as specified, NOT relaxed after seeing results. Result: "
        "ZERO pairs ever achieved 2-consecutive-cycle persistence across all 10 "
        "rolling cycles (2015-2021) -- STOP disposition, not a Sharpe/DSR failure "
        "(no return series exists to evaluate). Verified not a bug: cycle 8's 5 FDR "
        "survivors (CCS-LGIH, CCS-MTH, CCS-TMHC, LGIH-TMHC, AEIS-ICHR -- all "
        "economically sensible same-industry pairs, homebuilders + semiconductor-"
        "equipment) share zero overlap with cycle 9's 2 survivors (RTX-VC, BBSI-HRI) "
        "-- confirmed by direct inspection of per-cycle survivor sets. Key structural "
        "finding: same-sector pairwise correlations in this $100M-2B universe are "
        "much lower than large-cap pairs-trading intuition (bucket mean correlation "
        "~0.2-0.3, max rarely above 0.6), which is why hierarchical clustering "
        "narrows 59,687 within-bucket candidate pairs down to just 307 -- small/mid "
        "caps are structurally more idiosyncratic than the mechanism's implicit "
        "large-cap-pairs-trading prior assumes. adjClose used for BOTH signal AND "
        "P&L throughout (not the registry's usual adjClose-signal/raw-close-return "
        "split) because in pairs trading signal and return are the same price "
        "series -- a raw-close split/dividend artifact in either leg would corrupt "
        "the cointegration test, generate spurious entries/exits, AND misstate P&L "
        "simultaneously. Formation/trading leakage audit: PASS by construction "
        "(trading_start == formation_end exactly every cycle; hedge ratio and "
        "spread mean/std frozen at formation end, never recomputed mid-trading)."
    ),
    "rebalance_freq": "rolling (12mo formation / 6mo trading, rolled every 6mo, GGR 2006 design)",
    "rebalance_note": (
        "10 rolling cycles, 2015-01-01 to 2021-01-01. 'Non-overlapping formation "
        "periods' for the persistence filter is read as 'distinct sequential "
        "cycles in the rolling index' (cycle i vs i+1), not literal zero-date-"
        "overlap -- consecutive GGR rolling formation windows share 6 of 12 months "
        "of data by construction, so a literal zero-overlap reading would be "
        "incompatible with using standard rolling formation at all. Disclosed, "
        "not silently assumed."
    ),
    "portfolio_note": (
        "Equal-dollar long/short per pair (not beta-weighted -- hedge ratio used "
        "only to define the cointegrating spread signal, not position sizing, per "
        "pre-registered design). Entry |z|>2, exit |z|<0.5 or forced close at "
        "trading-period end. MIN_BUCKET_SIZE=8, CLUSTER_DIST_THRESH=0.4 (corr>=0.6), "
        "cluster size 2-8, FDR alpha=0.05 -- all locked before running."
    ),
    "signal_note": (
        f"Selectivity funnel (summed across 10 cycles): {tot_pre:,} pre-clustering "
        f"within-bucket candidate pairs -> {tot_cand} after hierarchical clustering "
        f"(99.5% reduction) -> {tot_tests} cointegration tests run -> {tot_fdr} "
        f"FDR-corrected survivors (per-cycle: 1,0,0,0,0,0,1,0,5,2) -> 0 persistence-"
        f"qualified (2 consecutive cycles) -> 0 actual trades. Mean universe size "
        f"{n_universe_avg:.0f} names/cycle, avg {np.mean([c['n_bucketed'] for c in cs]):.0f} "
        f"bucketed into sectors with >=8 members."
    ),
    "notes": (
        "STOP -- zero trades under the pre-registered methodology; no return "
        "series, Sharpe/CALMAR/beta/DSR not computable. NOT relaxed/re-tuned after "
        "seeing this (that would defeat the trial's own multiple-testing discipline). "
        "DIAGNOSTIC SUPPLEMENT ONLY (NOT this trial's registered result, NOT itself "
        "additionally FDR-corrected for the multiple-testing the persistence filter "
        "exists to control): bypassing the persistence filter and trading the 9 raw "
        f"per-cycle FDR-survivor pairs directly gives {diag['n_trades']} trades "
        f"(win rate {diag['win_rate_overall']:.1%} overall -- {diag['win_rate_converged']:.1%} "
        f"for converged exits avg {diag['avg_days_converged']:.0f}d, only "
        f"{diag['win_rate_forced']:.1%} for forced-close exits avg {diag['avg_days_forced']:.0f}d). "
        f"Net monthly (61mo): ann_ret={diag['net_ann_ret']:.2%} Sharpe={diag['net_sharpe']:.3f} "
        f"CALMAR={diag['net_calmar']:.3f} MaxDD={diag['net_max_dd']:.2%} t={diag['net_t_stat']:.2f}. "
        f"Gross monthly: ann_ret={diag['gross_ann_ret']:.2%} Sharpe={diag['gross_sharpe']:.3f} "
        f"MaxDD={diag['gross_max_dd']:.2%} -- gross is near-breakeven-to-negative, "
        f"not a clear 'good signal ruined by cost' story; cost drag ~{diag['cost_drag_bps_annualized']:.0f}"
        f"bps/yr deepens an already-weak gross result. Realized beta vs IWM = "
        f"{diag['beta_vs_iwm']:.3f} (t={diag['beta_t']:.2f}, not significant) -- the "
        f"low-beta motivating premise is roughly validated (not significantly "
        f"different from zero) even in this small, non-registered sample. "
        f"Concentration: top pair {diag['top_pair_pnl_share']:.1%} of total |PnL|, "
        f"top sector same (single pair in that sector) -- acceptable given the "
        f"small N=16 trades, flagged as a caveat not a clean pass. Only "
        f"{diag['days_with_position']}/{diag['total_business_days']} business days "
        f"(25%) had any capital deployed at all across the 5-year window."
    ),
    "mean_turnover": np.nan,
    "annual_cost_drag_bps": np.nan,
    "ann_return_net": np.nan,
    "ann_return_gross": np.nan,
    "sharpe": np.nan,
    "calmar": np.nan,
    "max_drawdown": np.nan,
    "t_stat": np.nan,
    "market_beta": np.nan,
    "market_beta_tstat": np.nan,
    "mean_ic": np.nan,
    "ic_t_stat": np.nan,
    "win_rate": np.nan,
    "n_periods": 0,
    "per_period_sr": np.nan,
    "dsr_threshold": np.nan,
    "clears_dsr": False,
}

reg_df = pd.read_csv(REGISTRY)
if not reg_df.empty and "study" in reg_df.columns:
    reg_df = reg_df[reg_df["study"] != "pairs_S18"].copy()
for c in row:
    if c not in reg_df.columns:
        reg_df[c] = np.nan
reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
reg_df.to_csv(REGISTRY, index=False)
print(f"\n[Registry] pairs_S18 logged as trial #{NEW_TRIAL_NUMBER} -> {REGISTRY}")
