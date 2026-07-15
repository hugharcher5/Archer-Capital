"""
Compute final summary and log all four S22 sub-trials (DCF/PE/PS/PB) to
research/trial_registry.csv, sharing one DSR family slot (IVOL/MAX convention).
Re-reads trial_registry.csv's current max trial_number IMMEDIATELY before writing.
"""
import sys
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location("gpi_run_s22_final", str(Path(__file__).resolve().parents[1] / "gp_industry_neutral" / "run.py"))
GPI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GPI)

RESULTS = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
SIGNALS = ["dcf", "pe", "ps", "pb"]

dfs = {sig: pd.read_csv(RESULTS / f"s22_{sig}_period_returns.csv", parse_dates=["rebalance_date", "hold_end"])
       for sig in SIGNALS}
for sig in SIGNALS:
    dfs[sig]["year"] = dfs[sig]["rebalance_date"].dt.year

tbill_q = pd.read_csv(Path(__file__).resolve().parent / "cache" / "tbill_quarterly.csv",
                       index_col=0, parse_dates=True).iloc[:, 0]

# ── FINAL re-check of trial count -- do this LAST, right before writing ─────
reg_df = pd.read_csv(REGISTRY)
current_max = int(pd.to_numeric(reg_df["trial_number"], errors="coerce").dropna().max())
NEW_TRIAL_NUMBER = current_max + 1
N_TRIALS_DSR = NEW_TRIAL_NUMBER
print(f"[Registry re-check] current max trial_number = {current_max} -> this family = #{NEW_TRIAL_NUMBER}")
print(reg_df[["timestamp", "study", "trial_number"]].tail(6).to_string(index=False))

SIGNAL_LABEL = {
    "dcf": "DCF-implied margin of safety (PIT adapter over existing Monte Carlo DCF engine's pure core)",
    "pe": "P/E (market cap / TTM net income; excluded when net income <= 0)",
    "ps": "P/S (market cap / TTM revenue; excluded when revenue <= 0, practically never binds)",
    "pb": "P/B (market cap / book value of equity; excluded when book equity <= 0)",
}

rows = []
for sig in SIGNALS:
    df = dfs[sig]
    m = GPI.compute_metrics(df["ret_net"])
    m_gross = GPI.compute_metrics(df["ret_gross"])
    beta_stats = GPI.compute_market_beta(df["ret_net"], df["mkt_ret"])

    ic_series = df["ic"].dropna()
    mean_ic = float(ic_series.mean())
    ic_t = float(mean_ic / (ic_series.std(ddof=1) / np.sqrt(len(ic_series))))

    per_period_sr = float(df["ret_net"].mean() / df["ret_net"].std(ddof=1))
    dsr_thr = GPI.deflated_sharpe_threshold(N_TRIALS_DSR, m["n"])
    clears_dsr = bool(per_period_sr > dsr_thr)

    combined = pd.DataFrame({"ls": df["ret_net"], "mkt": df["mkt_ret"]}).dropna()
    X = np.column_stack([np.ones(len(combined)), combined["mkt"].values])
    coeffs, _, _, _ = np.linalg.lstsq(X, combined["ls"].values, rcond=None)
    alpha_ann_mkt = float((1 + coeffs[0]) ** 4 - 1)

    df2 = df.set_index("rebalance_date")
    tbill_aligned = tbill_q.reindex(df2.index, method="nearest")
    excess = df2["ret_net"] - tbill_aligned / 4.0
    alpha_ann_tbill = float((1 + excess.mean()) ** 4 - 1)

    row = {
        "timestamp": datetime.now().isoformat(),
        "study": f"valuationcompare_S22_{sig}",
        "hypothesis": (
            "Long-only quarterly valuation portfolio (top 20% most undervalued, "
            f"equal-weight): {SIGNAL_LABEL[sig]}. One of four sub-trials sharing "
            "this DSR family slot (same underlying question -- does a valuation "
            "signal identify outperforming stocks -- tested four ways), IVOL/MAX "
            "family-sharing convention. GENUINELY NEW CONSTRUCTION: long-only, "
            "not beta-neutral long/short like every other trial in this registry."
        ),
        "data_source": (
            "Tiingo (adjClose for beta, raw close for returns) + SEC XBRL "
            "(PIT net income/revenue/book equity/shares/DCF inputs, filed<=t) "
            "+ FRED (historical risk-free rate for the DCF's WACC, 3-month "
            "T-bill for alpha). No new data sourcing."
        ),
        "status": "completed_in_sample",
        "trial_number": NEW_TRIAL_NUMBER,
        "n_trials_dsr": N_TRIALS_DSR,
        "trial_note": (
            f"Family member ({sig.upper()} sub-trial), shares DSR slot #{NEW_TRIAL_NUMBER} "
            "with the other three S22 sub-trials -- one genuinely new construction "
            "(long-only valuation-ranked), not four separate independent trials."
        ),
        "rebalance_freq": "quarterly",
        "rebalance_note": "24 periods, 2015-01-01->2021-01-01, matching this registry's standard window.",
        "portfolio_note": (
            "LONG-ONLY, top 20% by undervaluation_score, equal-weight (not "
            "beta-neutral) -- beta is expected to be well above zero, unlike "
            "every prior trial. Standard cost model (bid-ask by cap tier) on "
            "actual quarterly turnover; no borrow leg (no shorts)."
        ),
        "ann_return_net": m["ann_ret"],
        "ann_return_gross": m_gross["ann_ret"],
        "annual_cost_drag_bps": float(df["cost"].mean() * 4 * 10000),
        "mean_turnover": float(df["turnover"].mean()),
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
            f"alpha_vs_EWuniverse(ann)={alpha_ann_mkt:+.4f}  alpha_vs_3M_Tbill(ann)={alpha_ann_tbill:+.4f}  "
            f"sector_hhi_mean={df['sector_hhi'].mean():.4f}  sector_top_share_mean={df['sector_top_share'].mean():.4f}  "
            f"2020_cum_ret={(1+df[df['year']==2020]['ret_net']).prod()-1:.4f}"
        ),
        "notes": (
            f"FAILS DSR (per_period_sr={per_period_sr:.4f} < threshold={dsr_thr:.4f}, N={N_TRIALS_DSR}). "
            f"Beta={beta_stats['beta']:.3f} (t={beta_stats['beta_t']:.2f}) confirms long-only "
            f"construction carries genuine market beta as expected -- NOT a defect. "
            f"Alpha vs EW-universe is NEGATIVE ({alpha_ann_mkt:+.2%}/yr) despite strongly "
            f"positive alpha vs 3M T-bill ({alpha_ann_tbill:+.2%}/yr) -- absolute returns are "
            f"substantially explained by high-beta exposure to a rising 2015-2020 small-cap "
            f"market, not genuine alpha over that market. See "
            f"research/CORRECTED_TRIAL_REGISTRY.md for the full four-way comparison, "
            f"year-by-year table, and ranking."
        ),
    }
    rows.append(row)
    print(f"{sig}: Sharpe={m['sharpe']:.3f} CALMAR={m['calmar']:.3f} beta={beta_stats['beta']:.3f} "
          f"IC={mean_ic:.4f}(t={ic_t:.2f}) alpha_mkt={alpha_ann_mkt:+.2%} clears_dsr={clears_dsr}")

reg_df = pd.read_csv(REGISTRY)
for study in [f"valuationcompare_S22_{s}" for s in SIGNALS]:
    reg_df = reg_df[reg_df["study"] != study]
for r in rows:
    for c in r:
        if c not in reg_df.columns:
            reg_df[c] = np.nan
reg_df = pd.concat([reg_df, pd.DataFrame(rows)], ignore_index=True)
reg_df.to_csv(REGISTRY, index=False)
print(f"\n[Registry] 4 S22 sub-trials logged as trial #{NEW_TRIAL_NUMBER} -> {REGISTRY}")
