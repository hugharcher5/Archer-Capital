"""
Compute final metrics/diagnostics for the Volatility-Managed GP/A trial and log
to research/trial_registry.csv + research/PHASE3_TRIAL_REGISTRY.md.

Re-reads trial_registry.csv's current max trial_number IMMEDIATELY before
writing (not earlier in the session) -- see feedback_dsr_trial_numbering
memory / the S13 (#20) collision incident this guards against.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_GP_DIR = Path(__file__).resolve().parents[1] / "gp"
sys.path.insert(0, str(_GP_DIR))
import run as gp  # noqa: E402

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
PHASE3_MD = Path(__file__).resolve().parents[1] / "PHASE3_TRIAL_REGISTRY.md"

df = pd.read_csv(RESULTS / "gp_volmanaged_period_returns.csv", parse_dates=["rebalance_date", "hold_end"])
n = len(df)

m_base = gp.compute_metrics(df["ls_ret_base"])
m_vm   = gp.compute_metrics(df["ls_ret_vm"])
m_base_gross = gp.compute_metrics(df["ls_gross_base"])
m_vm_gross   = gp.compute_metrics(df["ls_gross_vm"])
beta_base = gp.compute_market_beta(df["ls_ret_base"], df["mkt_ret"])
beta_vm   = gp.compute_market_beta(df["ls_ret_vm"], df["mkt_ret"])

ic_series = df["ic"].dropna()
mean_ic = float(ic_series.mean())
ic_t = float(mean_ic / (ic_series.std(ddof=1) / np.sqrt(len(ic_series))))

cost_base_bps = float(df["ls_cost_base"].mean() * 4 * 10_000)
cost_vm_bps   = float(df["ls_cost_vm"].mean()   * 4 * 10_000)
mean_turnover = float(df["avg_turnover"].mean())

per_period_sr_base = float(df["ls_ret_base"].mean() / df["ls_ret_base"].std(ddof=1))
per_period_sr_vm   = float(df["ls_ret_vm"].mean()   / df["ls_ret_vm"].std(ddof=1))

# ── FINAL re-check of trial count -- do this LAST, right before writing ──────
reg_df = pd.read_csv(REGISTRY)
_numeric_trial_nums = pd.to_numeric(reg_df["trial_number"], errors="coerce").dropna()
current_max = int(_numeric_trial_nums.max())
NEW_TRIAL_NUMBER = current_max + 1
N_TRIALS_DSR = NEW_TRIAL_NUMBER
print(f"[Registry re-check] current max trial_number = {current_max} "
      f"-> this trial = #{NEW_TRIAL_NUMBER}")
print(reg_df[["timestamp", "study", "trial_number"]].tail(6).to_string(index=False))

dsr_thr = float(stats.norm.ppf(1 - 1.0 / N_TRIALS_DSR) / np.sqrt(n))
clears_dsr_base = bool(per_period_sr_base > dsr_thr)
clears_dsr_vm   = bool(per_period_sr_vm > dsr_thr)

row = {
    "timestamp":            datetime.now().isoformat(),
    "study":                "gpvolmanaged_S17",
    "hypothesis": (
        "Volatility-managed GP/A (Moreira & Muir 2017 overlay): scaling S3-Q's "
        "beta-neutral GP/A long-short gross exposure inversely to trailing "
        "6-month realized vol, targeting 10% annualized, improves Sharpe/CALMAR "
        "vs the unscaled signal without altering stock selection. Construction-"
        "level overlay, not a re-tune of GP/A. Universe upgraded with a Russell "
        "R3000 confirmed-exit overlay (2016-2023, from existing Wayback-archived "
        "data) -- NOT literal point-in-time Russell 2000 membership (no full "
        "snapshot exists in the archive to reconstruct that; see trial notes)."
    ),
    "data_source": (
        "Tiingo (adjClose for vol-targeting + beta, raw close for hold-period "
        "returns) + SEC XBRL (GP/A signal, PIT) + Wayback-archived FTSE Russell "
        "R3000 additions/deletions (research/russell_reconstitution/results/, "
        "reused unmodified, confirmed-exit overlay only)"
    ),
    "status":               "completed_in_sample",
    "trial_number":         NEW_TRIAL_NUMBER,
    "trial_note": (
        "New independent trial. Construction-level overlay family distinct from "
        "S3-Q (shares base signal, not DSR slot -- genuinely new hypothesis "
        "mechanism per pre-registration). Window 2016-01-01->2024-01-01 (32 "
        "quarterly periods), the closest achievable ~8yr window given Russell "
        "archive coverage (2015 unwired per russell_reconstitution/README.md). "
        "Two parallel series run on IDENTICAL window/universe: 'base' (beta-"
        "neutral only, no vol overlay) vs 'vol-managed' (base + overlay) -- "
        "isolates the overlay's effect. Original S3-Q trial #8 (2015-2021, no "
        "Russell overlay) reported separately as historical reference only, "
        "not a clean A/B (different window/universe)."
    ),
    "n_trials_dsr":         N_TRIALS_DSR,
    "rebalance_freq":       "quarterly",
    "rebalance_note": (
        "Vol-scale recomputed every rebalance from trailing realized vol of the "
        "UNSCALED basket's own daily (adjClose) returns, strictly using data "
        "before t (no look-ahead). First period (2016-01-01) has no trailing "
        "history -> vol_scale=1.0 by construction (flagged, not scaled)."
    ),
    "portfolio_note": (
        "Beta-neutral (EW market proxy, 60-day adjClose) IDENTICAL to S3-Q, "
        "PLUS a multiplicative vol_scale applied equally to both legs "
        "(preserves beta-neutral ratio, moves only gross level). Pre-registered: "
        "TARGET_VOL=10% ann., VOL_WINDOW=126 trading days (~6mo), VOL_MIN=60 "
        "days before scaling activates, SCALE_CLIP=[0.2x,3.0x]. Locked before "
        "running, not tuned after seeing results."
    ),
    "ann_return_net":       m_vm["ann_ret"],
    "ann_return_gross":     m_vm_gross["ann_ret"],
    "annual_cost_drag_bps": cost_vm_bps,
    "mean_turnover":        mean_turnover,
    "sharpe":               m_vm["sharpe"],
    "calmar":               m_vm["calmar"],
    "max_drawdown":         m_vm["max_dd"],
    "t_stat":               m_vm["t_stat"],
    "market_beta":          beta_vm["beta"],
    "market_beta_tstat":    beta_vm["beta_t"],
    "mean_long_scale":      float(df["final_long_scale"].mean()),
    "mean_short_scale":     float(df["final_short_scale"].mean()),
    "mean_beta_long":       float(df["mean_beta_long"].mean()),
    "mean_beta_short":      float(df["mean_beta_short"].mean()),
    "mean_ic":              mean_ic,
    "ic_t_stat":            ic_t,
    "win_rate":             m_vm["win_rate"],
    "n_periods":            n,
    "per_period_sr":        per_period_sr_vm,
    "dsr_threshold":        dsr_thr,
    "clears_dsr":           clears_dsr_vm,
    "mean_pct_dropped":     float(df["pct_dropped"].mean()),
    "signal_note": (
        f"BASE (no vol overlay, same window/universe) for direct comparison: "
        f"ann_ret_net={m_base['ann_ret']:.4f} sharpe={m_base['sharpe']:.4f} "
        f"calmar={m_base['calmar']:.4f} max_dd={m_base['max_dd']:.4f} "
        f"beta={beta_base['beta']:.4f} per_period_sr={per_period_sr_base:.4f} "
        f"clears_dsr={clears_dsr_base}. "
        f"Original S3-Q (trial #8, 2015-2021, no Russell overlay, historical "
        f"reference only): ann_ret_net=0.0094 sharpe=0.1338 calmar=0.0332 "
        f"max_dd=-0.2831 beta=-0.1030 mean_ic=0.0404 (from trial_registry.csv)."
    ),
    "notes": (
        "KNOWN LIMITATIONS (disclosed): (1) 'Russell overlay' = confirmed-exit "
        "only (excludes tickers whose most recent R3000 event is a deletion); "
        "does NOT confirm true inclusion for names never appearing in an event, "
        "and does NOT distinguish R1000 vs R2000 (archive only tags r3000/"
        "rmicro) -- NOT literal point-in-time R2000 membership. (2) R3000 "
        "deletions data has a GAP for 2018-2019 (no PDF recoverable in the "
        "original Russell-1 acquisition) -- exits in those years are missed "
        "until any later recorded event. (3) Cost model (standard, per "
        "instructions) charges turnover-based costs on names that rotate "
        "in/out, not on incremental notional traded when vol_scale itself "
        "changes gross exposure on unchanged names -- likely understates real "
        "cost drag for the vol-managed variant specifically. (4) Per-year "
        "realized beta requested at n=4 quarters/year -- t-stats reported but "
        "essentially uninformative at 2 dof; see full per-year table in report. "
        "(5) 2020 finding: vol overlay reduced max-drawdown TROUGH depth in the "
        "COVID quarter (-17.1%->-14.3%) but underperformed BASE on full-year "
        "2020 cumulative return (-3.78% vs +1.23%) because backward-looking "
        "trailing vol lagged into the Q2/Q3 2020 recovery, throttling upside "
        "capture more than it protected the initial crash quarter."
    ),
}

reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
if not reg_df.empty and "study" in reg_df.columns:
    reg_df = reg_df[reg_df["study"] != "gpvolmanaged_S17"].copy()
# align columns: keep any existing columns, add new ones as needed
for c in row:
    if c not in reg_df.columns:
        reg_df[c] = np.nan
reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
reg_df.to_csv(REGISTRY, index=False)
print(f"\n[Registry] gpvolmanaged_S17 logged as trial #{NEW_TRIAL_NUMBER} -> {REGISTRY}")

# Save a small summary json for the report step
import json
summary = {
    "trial_number": NEW_TRIAL_NUMBER,
    "n_trials_dsr": N_TRIALS_DSR,
    "dsr_threshold": dsr_thr,
    "m_base": m_base, "m_vm": m_vm,
    "m_base_gross": m_base_gross, "m_vm_gross": m_vm_gross,
    "beta_base": beta_base, "beta_vm": beta_vm,
    "mean_ic": mean_ic, "ic_t": ic_t,
    "cost_base_bps": cost_base_bps, "cost_vm_bps": cost_vm_bps,
    "mean_turnover": mean_turnover,
    "per_period_sr_base": per_period_sr_base, "per_period_sr_vm": per_period_sr_vm,
    "clears_dsr_base": clears_dsr_base, "clears_dsr_vm": clears_dsr_vm,
}
with open(RESULTS / "final_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
print(f"[Saved] final summary -> {RESULTS}/final_summary.json")
