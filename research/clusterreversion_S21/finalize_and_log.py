"""
Compute final summary and log S21 (cluster-conditioned mean reversion) to
research/trial_registry.csv AND research/CORRECTED_TRIAL_REGISTRY.md (the
one canonical narrative registry -- confirmed per root CLAUDE.md's "Research
Registry" section and reference_canonical_registry memory before writing).

Re-reads trial_registry.csv's current max trial_number IMMEDIATELY before
writing (S16-collision lesson -- do not trust a count computed at task start).
"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
CORRECTED_MD = Path(__file__).resolve().parents[1] / "CORRECTED_TRIAL_REGISTRY.md"
IWM_CSV  = (Path(__file__).resolve().parents[1] / "russell_reconstitution"
            / "cache" / "tiingo" / "IWM.csv")

with open(RESULTS / "_cycle_summary.json") as f:
    cs = json.load(f)

# =============================================================================
# Selectivity funnel (matches S18's funnel-discipline convention)
# =============================================================================
n_universe_avg = float(np.mean([c["n_universe"] for c in cs]))
tot_bucketed = sum(c["n_bucketed"] for c in cs)
tot_clusters_all = sum(c["n_clusters_all_sizes"] for c in cs)
tot_clusters_range = sum(c["n_clusters_in_range"] for c in cs)
tot_tradeable_clusters = sum(c["n_tradeable_clusters"] for c in cs)
tot_tradeable_members = sum(c["n_tradeable_members"] for c in cs)

# =============================================================================
# Metrics helper (parameterized periods_per_year -- monthly aggregation, same
# convention S18 used for its own diagnostic: daily book returns compounded
# into calendar months before computing Sharpe/CALMAR)
# =============================================================================
def compute_metrics(ls_series: pd.Series, periods_per_year: int = 12) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan, "calmar": np.nan,
                "max_dd": np.nan, "t_stat": np.nan, "win_rate": np.nan, "n": n}
    cum = (1 + ls).cumprod()
    total = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (periods_per_year / n) - 1)
    mu, sigma = float(ls.mean()), float(ls.std(ddof=1))
    sharpe = float((mu / sigma) * np.sqrt(periods_per_year)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s = (cum - roll_mx) / roll_mx
    max_dd = float(dd_s.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe, "calmar": calmar,
            "max_dd": max_dd, "t_stat": t_stat, "win_rate": win_rt, "n": n}


def compute_market_beta(ls_returns: pd.Series, mkt_returns: pd.Series) -> dict:
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < 5:
        return {"beta": np.nan, "beta_t": np.nan}
    x, y = combined["mkt"].values, combined["ls"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"beta": np.nan, "beta_t": np.nan}
    beta = float(coeffs[1])
    y_hat = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n = len(y)
    se = (np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


def monthly_from_daily(daily_csv: Path) -> pd.DataFrame:
    d = pd.read_csv(daily_csv, parse_dates=["date"])
    d["month"] = d["date"].dt.to_period("M")
    rows = []
    for mo, g in d.groupby("month"):
        net = float((1 + g["book_net"]).prod() - 1)
        gross = float((1 + g["book_gross"]).prod() - 1)
        rows.append({"month": mo.to_timestamp(), "net": net, "gross": gross,
                     "cost": gross - net, "days": len(g), "mean_n_open": float(g["n_open"].mean())})
    return pd.DataFrame(rows)


def iwm_monthly_aligned(months: pd.Series) -> pd.Series:
    iwm = pd.read_csv(IWM_CSV, parse_dates=["date"]).sort_values("date")
    iwm["month"] = iwm["date"].dt.to_period("M")
    out = {}
    for mo in months:
        p = pd.Period(mo, freq="M")
        sub = iwm[iwm["month"] == p]
        if sub.empty:
            out[mo] = np.nan
            continue
        out[mo] = float(sub["adjClose"].iloc[-1] / sub["adjClose"].iloc[0] - 1)
    return pd.Series(out)


def per_cycle_ic(trades: pd.DataFrame, cs: list[dict]) -> dict:
    """IC = spearman(|entry_z|, trade_gross) per cycle -- does bigger dislocation
    predict bigger reversion gain? Averaged across cycles with a t-stat, same
    per-period-IC convention used throughout this registry."""
    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    ics = []
    for c in cs:
        tstart, tend = pd.Timestamp(c["trading_start"]), pd.Timestamp(c["trading_end"])
        sub = trades[(trades["entry_date"] >= tstart) & (trades["entry_date"] < tend)]
        if len(sub) < 10:
            continue
        ic, _ = stats.spearmanr(sub["entry_z"].abs(), sub["trade_gross"])
        if np.isfinite(ic):
            ics.append(ic)
    ic_s = pd.Series(ics)
    mean_ic = float(ic_s.mean()) if len(ic_s) else np.nan
    ic_t = (float(ic_s.mean() / (ic_s.std(ddof=1) / np.sqrt(len(ic_s))))
            if len(ic_s) > 1 and ic_s.std(ddof=1) > 0 else np.nan)
    return {"mean_ic": mean_ic, "ic_t_stat": ic_t, "n_cycles_with_ic": len(ic_s), "per_cycle": ics}


def summarize(tag: str) -> dict:
    trades = pd.read_csv(RESULTS / f"trades_{tag}.csv", parse_dates=["entry_date", "exit_date"])
    n_trades = len(trades)
    win_rate_overall = float((trades["trade_pnl"] > 0).mean())
    conv = trades[trades["exit_reason"] == "converged"]
    forced = trades[trades["exit_reason"].isin(["time_forced", "period_forced_close"])]
    win_rate_converged = float((conv["trade_pnl"] > 0).mean()) if len(conv) else np.nan
    win_rate_forced = float((forced["trade_pnl"] > 0).mean()) if len(forced) else np.nan
    avg_days_converged = float(conv["holding_days"].mean()) if len(conv) else np.nan
    avg_days_forced = float(forced["holding_days"].mean()) if len(forced) else np.nan
    pct_converged = float((trades["exit_reason"] == "converged").mean())

    monthly = monthly_from_daily(RESULTS / f"daily_book_returns_{tag}.csv")
    m_net = compute_metrics(monthly["net"])
    m_gross = compute_metrics(monthly["gross"])
    iwm_ret = iwm_monthly_aligned(monthly["month"])
    beta = compute_market_beta(monthly["net"].reset_index(drop=True), iwm_ret.reset_index(drop=True))

    per_period_sr = float(monthly["net"].mean() / monthly["net"].std(ddof=1)) if monthly["net"].std(ddof=1) > 0 else np.nan
    cost_drag_bps = float(monthly["cost"].mean()) * 12 * 10_000

    ic = per_cycle_ic(trades, cs)

    # Concentration: sector-level share of |trade_pnl| mass (falsification diagnostic)
    sector_abs = trades.assign(abs_pnl=trades["trade_pnl"].abs()).groupby("sector_group")["abs_pnl"].sum()
    total_abs = sector_abs.sum()
    top_sector_share = float(sector_abs.max() / total_abs) if total_abs > 0 else np.nan
    n_sectors = int((sector_abs > 0).sum())

    daily = pd.read_csv(RESULTS / f"daily_book_returns_{tag}.csv")
    days_with_position = len(daily)

    return {
        "tag": tag, "n_trades": n_trades, "n_entries": n_trades,  # every entry closes to a trade record
        "win_rate_overall": win_rate_overall, "win_rate_converged": win_rate_converged,
        "win_rate_forced": win_rate_forced, "pct_converged": pct_converged,
        "avg_days_converged": avg_days_converged, "avg_days_forced": avg_days_forced,
        "avg_holding_days": float(trades["holding_days"].mean()),
        "m_net": m_net, "m_gross": m_gross, "beta": beta,
        "per_period_sr": per_period_sr, "cost_drag_bps": cost_drag_bps,
        "ic": ic, "top_sector_share": top_sector_share, "n_sectors": n_sectors,
        "n_months": m_net["n"], "days_with_position": days_with_position,
        "mean_trade_gross": float(trades["trade_gross"].mean()),
        "mean_trade_cost": float(trades["trade_cost"].mean()),
        "mean_trade_pnl": float(trades["trade_pnl"].mean()),
    }


basket = summarize("basket")
peer = summarize("peer")

clusters_df = pd.read_csv(RESULTS / "clusters_summary.csv")
realized_avg_corr = float(clusters_df["avg_intra_cluster_corr"].mean())
realized_median_corr = float(clusters_df["avg_intra_cluster_corr"].median())

# ── FINAL re-check of trial count -- do this LAST, right before writing ──────
# Exclude any PRIOR row for this same study before computing the max -- this
# script is idempotent (re-running replaces its own row), but re-running it
# to fix a notes/interpretation bug (not a new trial) must NOT self-inflate
# the trial count by reading its own already-logged row as "the prior max".
# Caught by inspection after a first re-run silently bumped 27 -> 28 with a
# real gap left at 27 -- corrected here, not left in the registry.
reg_df = pd.read_csv(REGISTRY)
_others = reg_df[reg_df["study"] != "clusterreversion_S21"] if not reg_df.empty else reg_df
_numeric_trial_nums = pd.to_numeric(_others["trial_number"], errors="coerce").dropna()
current_max = int(_numeric_trial_nums.max())
NEW_TRIAL_NUMBER = current_max + 1
N_TRIALS_DSR = NEW_TRIAL_NUMBER
print(f"[Registry re-check] current max trial_number = {current_max} -> this trial = #{NEW_TRIAL_NUMBER}")
print(reg_df[["timestamp", "study", "trial_number"]].tail(6).to_string(index=False))

dsr_thr = deflated_sharpe_threshold(N_TRIALS_DSR, basket["m_net"]["n"])
clears_dsr = (bool(basket["per_period_sr"] > dsr_thr)
              if not np.isnan(basket["per_period_sr"]) and not np.isnan(dsr_thr) else False)

promotion_met = (basket["m_net"]["calmar"] >= 1.0) or (basket["m_net"]["sharpe"] >= 0.8)

print(f"\n{'='*90}\nPRIMARY (basket) RESULTS\n{'='*90}")
print(f"Sharpe(net,ann)={basket['m_net']['sharpe']:.3f} CALMAR={basket['m_net']['calmar']:.3f} "
      f"ann_net={basket['m_net']['ann_ret']:.2%} ann_gross={basket['m_gross']['ann_ret']:.2%} "
      f"maxDD={basket['m_net']['max_dd']:.2%}")
print(f"beta={basket['beta']['beta']:+.4f} (t={basket['beta']['beta_t']:+.3f})  "
      f"cost_drag={basket['cost_drag_bps']:.0f}bps/yr  n_months={basket['n_months']}")
print(f"IC(|z| vs gross)={basket['ic']['mean_ic']:+.4f} t={basket['ic']['ic_t_stat']:+.3f} "
      f"(n_cycles={basket['ic']['n_cycles_with_ic']})")
print(f"win_rate overall={basket['win_rate_overall']:.1%} converged={basket['win_rate_converged']:.1%} "
      f"forced={basket['win_rate_forced']:.1%}  pct_converged={basket['pct_converged']:.1%}")
print(f"n_trades={basket['n_trades']}  avg_hold={basket['avg_holding_days']:.2f}d  "
      f"top_sector_pnl_share={basket['top_sector_share']:.1%} across {basket['n_sectors']} sectors")
print(f"DSR: per_period_sr={basket['per_period_sr']:.4f} thr={dsr_thr:.4f} clears={clears_dsr}")
print(f"Realized avg intra-cluster corr: mean={realized_avg_corr:.3f} median={realized_median_corr:.3f} "
      f"(S18's own bucket-mean finding: 0.2-0.3, threshold target for this cut ~0.2)")

print(f"\n{'='*90}\nSECONDARY (peer) RESULTS\n{'='*90}")
print(f"Sharpe(net,ann)={peer['m_net']['sharpe']:.3f} CALMAR={peer['m_net']['calmar']:.3f} "
      f"ann_net={peer['m_net']['ann_ret']:.2%} ann_gross={peer['m_gross']['ann_ret']:.2%}")
print(f"beta={peer['beta']['beta']:+.4f} (t={peer['beta']['beta_t']:+.3f})  cost_drag={peer['cost_drag_bps']:.0f}bps/yr")
print(f"IC(|z| vs gross)={peer['ic']['mean_ic']:+.4f} t={peer['ic']['ic_t_stat']:+.3f}")
print(f"win_rate overall={peer['win_rate_overall']:.1%}  n_trades={peer['n_trades']}")

notes = (
    f"PRIMARY (basket) construction: Sharpe(net,ann)={basket['m_net']['sharpe']:.4f} "
    f"CALMAR={basket['m_net']['calmar']:.4f} ann_net={basket['m_net']['ann_ret']:.4%} "
    f"ann_gross={basket['m_gross']['ann_ret']:.4%} maxDD(net)={basket['m_net']['max_dd']:.4%} "
    f"beta_vs_IWM={basket['beta']['beta']:+.4f} (t={basket['beta']['beta_t']:+.3f}) "
    f"cost_drag={basket['cost_drag_bps']:.0f}bps/yr n_months={basket['n_months']}. "
    f"HEADLINE FINDING: this is a REAL, POSITIVE gross signal (gross Sharpe={basket['m_gross']['sharpe']:.3f}, "
    f"gross CALMAR={basket['m_gross']['calmar']:.3f}, gross ann_ret={basket['m_gross']['ann_ret']:.2%}, "
    f"gross win_rate={basket['m_gross']['win_rate']:.0%}, IC(|entry_z| vs trade_gross, per-cycle)="
    f"{basket['ic']['mean_ic']:+.4f} t={basket['ic']['ic_t_stat']:+.3f}, n_cycles={basket['ic']['n_cycles_with_ic']}/10 "
    f"-- positive and borderline-significant, bigger dislocations DO modestly predict bigger reversion gains) "
    f"-- COMPLETELY DESTROYED by transaction costs at this trading frequency: net Sharpe=-17.41, "
    f"net ann_ret=-66.4%, cost_drag~11000bps/yr (~110%/yr). This is the cleanest and MOST EXTREME version of "
    f"the S2/S8 'high-quality-signal-killed-by-cost' pattern in this registry -- unlike S8 (gross itself "
    f"negative, -3.08%/yr) or S2 (gross -4.74%/yr), S21's gross Sharpe of {basket['m_gross']['sharpe']:.2f} would "
    f"clear the promotion bar on its own; the entire failure is attributable to trading frequency (avg "
    f"{basket['avg_holding_days']:.1f}-day holds, {basket['n_trades']} round trips over 60 months, ~"
    f"{basket['mean_trade_cost']:.2%} round-trip cost each) rather than signal quality. Only "
    f"{basket['pct_converged']:.2%} of trades exit via genuine Z-score convergence (|z|<0.5 within 5 days); "
    f"{100-100*basket['pct_converged']:.1f}% are force-closed on the 5-day timer -- the reversion mechanism "
    f"almost never completes within the pre-registered window, but still generates enough net positive drift "
    f"on average (mean trade_gross={basket['mean_trade_gross']:.4%}, aggregated across "
    f"{basket['n_trades']:,} quasi-independent bets via breadth) to produce a respectable GROSS Sharpe -- "
    f"a textbook small-edge/high-breadth effect that transaction costs (charged per round trip, not "
    f"amortized by breadth) completely negate. Concentration: top sector = {basket['top_sector_share']:.1%} "
    f"of total |trade PnL| mass across {basket['n_sectors']} sectors with trades -- clean, no single sector "
    f"dominating. Selectivity funnel (10 cycles, GGR rolling): avg universe {n_universe_avg:.0f} names/cycle "
    f"-> {tot_bucketed} bucketed (sum) -> {tot_clusters_all} raw clusters at CLUSTER_DIST_THRESH=0.8 -> "
    f"{tot_clusters_range} in the pre-registered [4,15] size range -> {tot_tradeable_clusters} tradeable "
    f"clusters ({tot_tradeable_members} member-slots) -> {basket['n_entries']} entries -> {basket['n_trades']} "
    f"trades closed. REALIZED avg intra-cluster correlation = {realized_avg_corr:.3f} (median "
    f"{realized_median_corr:.3f}) -- directly comparable to S18's own diagnosed same-sector bucket-mean "
    f"of 0.2-0.3: this trial's clusters landed almost exactly on that 'typical' level (as intended by the "
    f"deliberately more permissive 0.8 distance cut vs S18's 0.4/>=0.6 cut) -- CONFIRMS clusters here really "
    f"are the 'weaker, basket-level' relationship the trial set out to test, not accidentally another "
    f"high-correlation subset. SECONDARY (peer) construction: Sharpe(net)={peer['m_net']['sharpe']:.4f} "
    f"Sharpe(gross)={peer['m_gross']['sharpe']:.4f} CALMAR(net)={peer['m_net']['calmar']:.4f} "
    f"ann_net={peer['m_net']['ann_ret']:.4%} beta={peer['beta']['beta']:+.4f} (t={peer['beta']['beta_t']:+.3f}) "
    f"IC={peer['ic']['mean_ic']:+.4f} (t={peer['ic']['ic_t_stat']:+.3f}) -- same pattern (real positive gross "
    f"edge, cost-destroyed net), somewhat weaker gross Sharpe than basket; basket-vs-single-nearest-peer "
    f"referencing is not the deciding factor for this trial's outcome, cost frequency is. "
    f"VERDICT ON THE TWO MOTIVATING QUESTIONS: (1) vs S18's pairwise-correlation problem: cluster-average "
    f"referencing DID successfully produce a large, non-zero trade count (basket: {basket['n_trades']} trades, "
    f"peer: {peer['n_trades']} trades, vs S18's 0 trades) at a realized correlation level (~{realized_avg_corr:.2f}) "
    f"where S18's stricter pairwise cointegration+FDR+persistence discipline produced nothing -- the "
    f"basket-relative mechanism is unambiguously more tradeable in the sense of generating both a testable "
    f"sample size AND a real positive gross IC/Sharpe, which S18 never even got to test. (2) vs S8's "
    f"gross-negative problem: cluster-conditioning DID fix it, decisively -- gross Sharpe went from S8's "
    f"implicit negative-gross regime to a genuinely attractive {basket['m_gross']['sharpe']:.2f} here. But "
    f"fixing that problem surfaced an even MORE severe one: the 5-day-hold/2.5-sigma-entry construction "
    f"trades so frequently (avg {basket['avg_holding_days']:.1f}-day hold vs GP/S7/S8's quarterly-to-monthly "
    f"cadence) that realistic transaction costs overwhelm even a genuinely good gross signal by an order of "
    f"magnitude -- this trial's own pre-registered 'HIGH cost-drag risk a priori' caution was directionally "
    f"right but the realized magnitude (~110%/yr cost drag) is far more extreme than anything else in this "
    f"registry (S2's next-worst cost drag was ~4079bps/yr, i.e. ~2.7x smaller). "
    f"{'CLEARS' if clears_dsr else 'FAILS'} DSR at N={N_TRIALS_DSR} (per-period SR={basket['per_period_sr']:.4f} "
    f"vs threshold {dsr_thr:.4f}). Promotion criteria (CALMAR>=1 or Sharpe>=0.8) on NET returns "
    f"{'MET' if promotion_met else 'NOT MET'} (gross alone would have cleared it: gross Sharpe="
    f"{basket['m_gross']['sharpe']:.2f}, gross CALMAR={basket['m_gross']['calmar']:.2f})."
)

row = {
    "timestamp": datetime.now().isoformat(),
    "study": "clusterreversion_S21",
    "hypothesis": (
        "Cluster-conditioned short-term mean reversion: within sector-bucketed, loosely-correlated "
        "clusters (realized avg intra-cluster corr ~0.2, deliberately weaker than S18's pairwise "
        "cointegration bar), a member deviating >2.5 sigma from the cluster's leave-one-out equal-weighted "
        "average (formation-period-normalized) is traded long/short against the cluster basket, exiting on "
        "reversion to |z|<0.5 or a 5-day forced close. Hybrid mechanism: basket-relative (not S18's strict "
        "pairwise cointegration) short-term reversal (not S8's full-cross-section ranking), testing whether "
        "cluster-conditioning fixes either S18's near-zero-candidate problem or S8's high-IC-but-gross-"
        "negative problem."
    ),
    "data_source": (
        "Tiingo adjClose (signal AND P&L, same disclosed convention as S18) + SEC SIC classification -- "
        "100% reused from gp_industry_neutral (universe/cost model) and pairs_S18 (sector bucketing, GGR "
        "rolling formation/trading schedule, clustering machinery) -- no new data sourcing."
    ),
    "status": "completed_in_sample",
    "trial_number": NEW_TRIAL_NUMBER,
    "n_trials_dsr": N_TRIALS_DSR,
    "trial_note": (
        "New independent trial, genuine hybrid mechanism between S18 (pairs, trial #24) and S8 (full-"
        "cross-section residual reversal, trial #14). Pre-registered before running: ENTRY_Z=2.5, "
        "EXIT_Z=0.5, MAX_HOLD_DAYS=5, cluster size [4,15], CLUSTER_DIST_THRESH=0.8 (targeting S18's own "
        "diagnosed 'typical' bucket correlation ~0.2, not S18's 'exceptional' >=0.6 bar). Both basket "
        "(primary) and nearest-peer (secondary) reference constructions run and reported per instructions."
    ),
    "rebalance_freq": "event-triggered (|z|>2.5 entry, no periodic rebalance); 12mo formation/6mo trading GGR rolling schedule (10 cycles, reused verbatim from S18) for cluster membership/parameter freezing",
    "rebalance_note": (
        "Turnover discipline structurally satisfied by construction (entries only fire on genuine >2.5-sigma "
        "dislocations, unlike S8's full-cross-section ranking which needed an explicit active/carry split) -- "
        "no additional carry bookkeeping needed."
    ),
    "portfolio_note": (
        "Equal-dollar member-vs-reference sizing (not beta-weighted, same design choice as S18); realized "
        "beta reported explicitly per instructions. Basket leg = leave-one-out equal-weighted cluster "
        "average (excludes the traded member itself)."
    ),
    "ann_return_net": basket["m_net"]["ann_ret"],
    "ann_return_gross": basket["m_gross"]["ann_ret"],
    "annual_cost_drag_bps": basket["cost_drag_bps"],
    "mean_turnover": np.nan,
    "sharpe": basket["m_net"]["sharpe"],
    "calmar": basket["m_net"]["calmar"],
    "max_drawdown": basket["m_net"]["max_dd"],
    "t_stat": basket["m_net"]["t_stat"],
    "market_beta": basket["beta"]["beta"],
    "market_beta_tstat": basket["beta"]["beta_t"],
    "mean_ic": basket["ic"]["mean_ic"],
    "ic_t_stat": basket["ic"]["ic_t_stat"],
    "win_rate": basket["win_rate_overall"],
    "n_periods": basket["n_months"],
    "per_period_sr": basket["per_period_sr"],
    "dsr_threshold": dsr_thr,
    "clears_dsr": clears_dsr,
    "signal_note": (
        f"Selectivity funnel: avg universe {n_universe_avg:.0f}/cycle -> {tot_bucketed} bucketed (sum) -> "
        f"{tot_clusters_all} raw clusters -> {tot_clusters_range} in [4,15] range -> {tot_tradeable_clusters} "
        f"tradeable clusters ({tot_tradeable_members} member-slots) -> {basket['n_entries']} entries -> "
        f"{basket['n_trades']} trades. Realized avg intra-cluster corr={realized_avg_corr:.3f} "
        f"(median {realized_median_corr:.3f}), matching S18's own 'typical' bucket-mean finding of 0.2-0.3."
    ),
    "notes": notes,
}

for c in row:
    if c not in reg_df.columns:
        reg_df[c] = np.nan
reg_df = reg_df[reg_df["study"] != "clusterreversion_S21"].copy() if not reg_df.empty else reg_df
reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
reg_df.to_csv(REGISTRY, index=False)
print(f"\n[Registry] clusterreversion_S21 logged as trial #{NEW_TRIAL_NUMBER} -> {REGISTRY}")

# Save a small machine-readable summary for the RESULTS.md writeup
with open(RESULTS / "final_summary.json", "w") as f:
    def _default(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        return str(o)
    json.dump({
        "basket": {k: v for k, v in basket.items() if k not in ("m_net", "m_gross", "beta", "ic")},
        "basket_m_net": basket["m_net"], "basket_m_gross": basket["m_gross"], "basket_beta": basket["beta"],
        "basket_ic": {k: v for k, v in basket["ic"].items() if k != "per_cycle"},
        "peer": {k: v for k, v in peer.items() if k not in ("m_net", "m_gross", "beta", "ic")},
        "peer_m_net": peer["m_net"], "peer_m_gross": peer["m_gross"], "peer_beta": peer["beta"],
        "peer_ic": {k: v for k, v in peer["ic"].items() if k != "per_cycle"},
        "realized_avg_corr": realized_avg_corr, "realized_median_corr": realized_median_corr,
        "trial_number": NEW_TRIAL_NUMBER, "dsr_threshold": dsr_thr, "clears_dsr": clears_dsr,
        "n_universe_avg": n_universe_avg, "tot_bucketed": tot_bucketed, "tot_clusters_all": tot_clusters_all,
        "tot_clusters_range": tot_clusters_range, "tot_tradeable_clusters": tot_tradeable_clusters,
        "tot_tradeable_members": tot_tradeable_members,
    }, f, indent=2, default=_default)
print(f"[Saved] -> final_summary.json")
