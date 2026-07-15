"""
Compute final summary and log S19 (52-week-high anchoring) to
research/trial_registry.csv. Re-reads trial_registry.csv's current max
trial_number IMMEDIATELY before writing.
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as S

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

df = pd.read_csv(RESULTS / "s19_period_returns.csv", parse_dates=["rebalance_date", "hold_end"])
fm = pd.read_csv(RESULTS / "s19_famamacbeth.csv", parse_dates=["rebalance_date"])
corr = pd.read_csv(RESULTS / "s19_correlation.csv", parse_dates=["rebalance_date"])
df["year"] = df["rebalance_date"].dt.year

m = S.compute_metrics(df["ls_ret"])
m_gross = S.compute_metrics(df["ls_gross"])
beta_stats = S.compute_market_beta(df["ls_ret"], df["mkt_ret"])

ic_series = df["ic"].dropna()
mean_ic = float(ic_series.mean())
ic_t = float(mean_ic / (ic_series.std(ddof=1) / np.sqrt(len(ic_series))))

cost_bps = float(df["ls_cost"].mean()) * 12 * 10000
turnover = float(df["avg_turnover"].mean())
per_period_sr = float(df["ls_ret"].mean() / df["ls_ret"].std(ddof=1))

ex2020 = df[df["year"] < 2020]
m_ex2020 = S.compute_metrics(ex2020["ls_ret"])
ex2020_ic = float(ex2020["ic"].mean())

b_near = fm["b_nearness"].dropna()
b_mom = fm["b_momentum"].dropna()
t_near = float(b_near.mean() / (b_near.std(ddof=1) / np.sqrt(len(b_near))))
t_mom = float(b_mom.mean() / (b_mom.std(ddof=1) / np.sqrt(len(b_mom))))

# ── FINAL re-check of trial count -- do this LAST, right before writing ─────
reg_df = pd.read_csv(REGISTRY)
_numeric_trial_nums = pd.to_numeric(reg_df["trial_number"], errors="coerce").dropna()
current_max = int(_numeric_trial_nums.max())
NEW_TRIAL_NUMBER = current_max + 1
N_TRIALS_DSR = NEW_TRIAL_NUMBER
print(f"[Registry re-check] current max trial_number = {current_max} -> this trial = #{NEW_TRIAL_NUMBER}")
print(reg_df[["timestamp", "study", "trial_number"]].tail(6).to_string(index=False))

dsr_thr = S.deflated_sharpe_threshold(N_TRIALS_DSR, m["n"])
clears_dsr = bool(per_period_sr > dsr_thr)

row = {
    "timestamp": datetime.now().isoformat(),
    "study": "fiftytwoweek_S19",
    "hypothesis": (
        "George & Hwang (2004) 52-week-high anchoring: investors anchor on a "
        "stock's 52-week high as a psychological ceiling; stocks trading NEAR "
        "their 52-week high see good news underreacted to, producing positive "
        "drift. Long top quintile nearness (adjClose / trailing 252-day high), "
        "short bottom quintile. Genuinely distinct mechanism from S1/S4 (IVOL/"
        "MAX) -- anchoring to a slow-moving reference price, not extreme "
        "recent-return behavior."
    ),
    "data_source": (
        "Tiingo (adjClose for signal, raw close for returns) + SEC "
        "shares-outstanding facts only (already cached from S9/S1/S4, no new "
        "fetches, no GP/A-style financial-statement dependency)"
    ),
    "status": "completed_in_sample",
    "trial_number": NEW_TRIAL_NUMBER,
    "n_trials_dsr": N_TRIALS_DSR,
    "trial_note": (
        "New independent trial, new mechanism family. Monthly rebalance "
        "pre-registered (matches original literature convention), no frequency "
        "sweep. Beta-neutral leg sizing (60-day OLS vs EW universe adjClose), "
        "actual-turnover cost model. Independence-from-momentum test built "
        "into primary design (not an afterthought): cross-sectional Spearman "
        "correlation vs S9's 12-1 momentum signal at every rebalance, plus a "
        "Fama-MacBeth two-step regression of forward return on both signals' "
        "cross-sectional ranks jointly."
    ),
    "rebalance_freq": "monthly",
    "rebalance_note": "72 monthly periods, 2015-01-01 to 2021-01-01. Locked before running, no sweep.",
    "portfolio_note": (
        "Beta-neutral (EW market proxy, 60-day adjClose). Leg = max(ceil(n*0.20),20). "
        f"Min leg size observed: {int(df['n_long'].min())} names -> max single-name "
        "weight ~1.5%, well-diversified by construction. n_long==n_short every "
        "period (guard clean)."
    ),
    "ann_return_net": m["ann_ret"],
    "ann_return_gross": m_gross["ann_ret"],
    "annual_cost_drag_bps": cost_bps,
    "mean_turnover": turnover,
    "sharpe": m["sharpe"],
    "calmar": m["calmar"],
    "max_drawdown": m["max_dd"],
    "t_stat": m["t_stat"],
    "market_beta": beta_stats["beta"],
    "market_beta_tstat": beta_stats["beta_t"],
    "mean_ic": mean_ic,
    "ic_t_stat": ic_t,
    "win_rate": m["win_rate"],
    "n_periods": m["n"],
    "per_period_sr": per_period_sr,
    "dsr_threshold": dsr_thr,
    "clears_dsr": clears_dsr,
    "signal_note": (
        f"Nearness = adjClose_t / trailing-252-trading-day-high adjClose, MIN_52W_DAYS=200. "
        f"Ex-2020 (2015-2019, 60mo) reference: ann_ret={m_ex2020['ann_ret']:.4f} "
        f"sharpe={m_ex2020['sharpe']:.4f} calmar={m_ex2020['calmar']:.4f} "
        f"max_dd={m_ex2020['max_dd']:.4f} mean_ic={ex2020_ic:.4f} -- already a losing "
        f"strategy before 2020, not purely a COVID-crash story."
    ),
    "notes": (
        f"FAILS DSR (per_period_sr={per_period_sr:.4f} < threshold={dsr_thr:.4f}, N={N_TRIALS_DSR}). "
        f"MOMENTUM INDEPENDENCE CHECK (the core secondary test): mean cross-sectional "
        f"Spearman correlation between nearness and 12-1 momentum = {corr['corr'].mean():.3f} "
        f"(range {corr['corr'].min():.3f} to {corr['corr'].max():.3f}, "
        f"{(corr['corr']>0.5).mean():.0%} of periods >0.5) -- CONTRADICTS the original "
        f"literature's 'largely independent' claim in this small/mid-cap universe; the "
        f"two signals are meaningfully correlated. Fama-MacBeth (return ~ nearness_rank + "
        f"momentum_rank, both included jointly): mean b_nearness={b_near.mean():.4f} "
        f"(t={t_near:.3f}) -- WRONG-SIGNED and only marginally significant once momentum "
        f"is controlled for; mean b_momentum={b_mom.mean():.4f} (t={t_mom:.3f}) -- correctly "
        f"signed, also marginal. CONCLUSION: nearness-to-52-week-high does NOT look like a "
        f"genuinely distinct, exploitable mechanism in this universe -- univariate IC is "
        f"already ~null (t=0.24), the signal is highly correlated with momentum (contradicting "
        f"the paper's independence premise), and controlling for momentum flips its sign rather "
        f"than confirming independent predictive power. It reads as a weaker, noisier restatement "
        f"of momentum here, not a new mechanism. 2020 SUB-PERIOD: catastrophic (-59.4% for the "
        f"year alone; 4 of 12 months had |return|>15%), with the worst months (April +26.8% mkt, "
        f"November +22.3% mkt) occurring DURING market recovery rallies -- consistent with a "
        f"momentum-crash-style regime sensitivity (long near-high/recent-winners, short far-"
        f"below/recent-losers gets destroyed when the most beaten-down names rally hardest). "
        f"Global max drawdown ({m['max_dd']:.1%}) spans peak 2016-01 to trough 2020-11 -- a "
        f"4+ year continuous underwater period, not a single-year event. High cost drag "
        f"(~{cost_bps:.0f} bps/yr) from monthly rebalance turnover (~{turnover:.0%}/month) "
        f"further compounds an already-negative gross result ({m_gross['ann_ret']:.2%}/yr) -- "
        f"not a 'good signal ruined by costs' story either."
    ),
}

reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
if not reg_df.empty and "study" in reg_df.columns:
    reg_df = reg_df[reg_df["study"] != "fiftytwoweek_S19"].copy()
for c in row:
    if c not in reg_df.columns:
        reg_df[c] = np.nan
reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
reg_df.to_csv(REGISTRY, index=False)
print(f"\n[Registry] fiftytwoweek_S19 logged as trial #{NEW_TRIAL_NUMBER} -> {REGISTRY}")
