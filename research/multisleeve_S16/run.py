"""
S16: Multi-Sleeve Blend (GP/A + S8 + S7-original)  — new independent trial
==============================================================================
Directly addresses the two causes S15 (compositeIC_S15, trial #19) diagnosed
for its own failure:
  1. GP/A and IVOL-Value component RETURNS were correlated +0.563 in-sample —
     far from the near-zero correlation S15's diversification thesis assumed.
  2. S7 (IVOL-conditioned value) has no natural continuous score — it is a
     hard-gated two-stage double-sort (top-tercile IVOL, then value-tercile
     WITHIN that subset). Forcing it into a continuous rank score for S15's
     merged composite FLIPPED ITS SIGN (IC +0.036 -> -0.015), destroying the
     component's edge before it ever entered the blend.

FIX, this trial:
  1. THREE SEPARATE, independently beta-neutral SLEEVES instead of one merged
     score. No score-merging, no rank-percentile averaging.
  2. S7 kept in its ORIGINAL hard-gated double-sort form — NOT converted to a
     continuous score. GP/A and S8 also kept in their exact original forms.
  3. Correlation is measured FIRST, as a diagnostic, reported prominently —
     not used to pick weights.
  4. Blend weights are PRE-REGISTERED: equal 1/3 capital to each sleeve,
     rebalanced quarterly to maintain equal weight. Not tuned after seeing
     correlation or performance.

DATA SOURCE FOR EACH SLEEVE: none re-run. Each sleeve's ALREADY-COMPLETED,
ALREADY-VALIDATED backtest output is reused verbatim:
  - GP/A       : research/gp/results/gp_period_returns.csv              (trial #8,  quarterly, 24 periods)
  - S7         : research/ivol_value/results/s7_period_returns.csv      (trial #13, quarterly, 24 periods)
  - S8         : research/residual_reversal/results/s8_period_returns.csv (trial #14, monthly, 70 periods,
                 turnover-discipline carry mechanism intact — untouched)
This is the most faithful way to guarantee "exactly as original construction"
for all three sleeves: reusing their own validated output rather than
re-deriving it removes any risk of subtly reimplementing a sleeve differently
from how it was originally tested and registered.

UNIVERSE NOTE (pre-registration deviation, confirmed with user before running):
The instructions called for actual point-in-time Russell 2000 constituent
membership as the tradable universe, sourced from research/russell_reconstitution/.
That directory only contains ANNUAL Russell 3000 (R1000 union R2000, not
R2000-specific) ADDITIONS/DELETIONS EVENT lists (2016-2023, 2015 excluded,
2018/2019 deletions missing) -- never a full point-in-time constituent
roster, and no quarterly granularity. True PIT R2000 membership is not
reconstructable from what's been sourced without new data acquisition, which
the instructions said not to do. Confirmed with user: fall back to the
standard $100M-$2B PIT survivorship-bias-free universe used by every other
trial in this registry (S3/S5/S7/S8/S10/S15/etc). The Russell PIT-membership
upgrade remains an OPEN, UNMET item for a future trial with new data
acquisition -- not silently dropped.

BENCHMARK: IWM (iShares Russell 2000 ETF) adjClose, already fetched and
cached by the Russell reconstitution trial at
research/russell_reconstitution/cache/tiingo/IWM.csv (2015-2026 coverage,
no new fetch needed) -- used both as the year-by-year "Russell 2000 return"
comparison and as the common beta-regression benchmark across all four
portfolios (blend + 3 sleeves), since each sleeve's own originally-logged
beta was computed against a slightly different internal EW small-cap proxy
(not apples-to-apples with each other). Both the internal-proxy betas
(quoted from the registry) and the common-IWM betas (computed fresh here,
on a shared quarterly calendar) are reported.

BLENDING (pre-registered, not tuned):
  Common quarterly reporting calendar = GP/A's own 24-quarter grid
  (2015-01-01 .. 2020-10-01, matching S3-Q/S7's native cadence exactly).
  S8's monthly returns are compounded into each of these 24 quarters
  (product of (1+monthly_net)-1 for the months falling in that quarter).
  blended_net_q  = (gp_net_q + s7_net_q + s8_net_q) / 3
  blended_gross_q = (gp_gross_q + s7_gross_q + s8_gross_q) / 3
  Q1-2015 is a KNOWN partial quarter for the S8 leg only: S8's own
  REBALANCE_DATES start 2015-03-01 (needs history to build its 36-month
  regression + market factor), so Q1-2015 only has 1 of 3 months of S8 data
  compounded in (Jan/Feb contribute 0%, i.e. that sleeve is simply not yet
  active). Flagged explicitly; affects only 1 of 24 quarters.

Registry key: multisleeve_S16.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
REGISTRY = ROOT / "research" / "trial_registry.csv"
CORRECTED_MD = ROOT / "research" / "CORRECTED_TRIAL_REGISTRY.md"

GP_CSV  = ROOT / "research" / "gp" / "results" / "gp_period_returns.csv"
S7_CSV  = ROOT / "research" / "ivol_value" / "results" / "s7_period_returns.csv"
S8_CSV  = ROOT / "research" / "residual_reversal" / "results" / "s8_period_returns.csv"
IWM_CSV = ROOT / "research" / "russell_reconstitution" / "cache" / "tiingo" / "IWM.csv"

IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")


# =============================================================================
# SECTION 1: Load each sleeve's already-validated period returns (no re-run)
# =============================================================================

def load_sleeves() -> dict:
    gp = pd.read_csv(GP_CSV, parse_dates=["rebalance_date", "hold_end"])
    s7 = pd.read_csv(S7_CSV, parse_dates=["rebalance_date", "hold_end"])
    s8 = pd.read_csv(S8_CSV, parse_dates=["rebalance_date", "hold_end"])
    print(f"[Load] GP/A: {len(gp)} quarterly periods ({gp['rebalance_date'].min().date()} -> {gp['rebalance_date'].max().date()})")
    print(f"[Load] S7  : {len(s7)} quarterly periods ({s7['rebalance_date'].min().date()} -> {s7['rebalance_date'].max().date()})")
    print(f"[Load] S8  : {len(s8)} monthly periods   ({s8['rebalance_date'].min().date()} -> {s8['rebalance_date'].max().date()})")
    return {"gp": gp, "s7": s7, "s8": s8}


# =============================================================================
# SECTION 2: Compound S8's monthly returns onto GP/A's quarterly grid
# =============================================================================

def compound_s8_to_quarterly(s8: pd.DataFrame, quarter_dates: list) -> pd.DataFrame:
    rows = []
    for i, qs in enumerate(quarter_dates):
        qe = quarter_dates[i + 1] if i + 1 < len(quarter_dates) else IS_END
        sub = s8[(s8["rebalance_date"] >= qs) & (s8["rebalance_date"] < qe)]
        n_months = len(sub)
        net = float((1 + sub["ls_ret"]).prod() - 1) if n_months else 0.0
        gross = float((1 + sub["ls_gross"]).prod() - 1) if n_months else 0.0
        rows.append({
            "rebalance_date": qs, "s8_net_q": net, "s8_gross_q": gross,
            "s8_n_months": n_months,
            "s8_mean_turnover": float(sub["actual_turnover"].mean()) if n_months else np.nan,
            "s8_mean_ic": float(sub["ic"].mean()) if n_months else np.nan,
        })
    df = pd.DataFrame(rows)
    n_partial = int((df["s8_n_months"] < 3).sum())
    if n_partial:
        print(f"  [Flag] {n_partial} quarter(s) with <3 months of S8 data compounded in "
              f"(expected: only Q1-2015, S8's own ramp-up):")
        print(df.loc[df["s8_n_months"] < 3, ["rebalance_date", "s8_n_months"]].to_string(index=False))
    return df


# =============================================================================
# SECTION 3: IWM benchmark, quarterly returns on the same grid
# =============================================================================

def iwm_quarterly_returns(quarter_dates: list) -> pd.DataFrame:
    iwm = pd.read_csv(IWM_CSV, parse_dates=["date"]).sort_values("date")
    rows = []
    for i, qs in enumerate(quarter_dates):
        qe = quarter_dates[i + 1] if i + 1 < len(quarter_dates) else IS_END
        entry_cands = iwm[iwm["date"] >= qs]
        exit_cands  = iwm[iwm["date"] <= qe]
        if entry_cands.empty or exit_cands.empty:
            rows.append({"rebalance_date": qs, "iwm_ret": np.nan})
            continue
        entry_px = float(entry_cands.iloc[0]["adjClose"])
        exit_px  = float(exit_cands.iloc[-1]["adjClose"])
        rows.append({"rebalance_date": qs, "iwm_ret": exit_px / entry_px - 1})
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 4: Metrics (identical formulas to every other trial in this registry)
# =============================================================================

def compute_metrics(ls_series: pd.Series, periods_per_year: int = 4) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan, "calmar": np.nan,
                "max_dd": np.nan, "t_stat": np.nan, "win_rate": np.nan, "n": n}
    cum     = (1 + ls).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (periods_per_year / n) - 1)
    mu, sigma = float(ls.mean()), float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(periods_per_year)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s    = (cum - roll_mx) / roll_mx
    max_dd  = float(dd_s.min())
    calmar  = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat  = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt  = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe, "calmar": calmar,
            "max_dd": max_dd, "t_stat": t_stat, "win_rate": win_rt, "n": n}


def compute_market_beta(ls_returns: pd.Series, mkt_returns: pd.Series, min_n: int = 5) -> dict:
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < min_n:
        return {"beta": np.nan, "beta_t": np.nan, "n": len(combined)}
    x = combined["mkt"].values
    y = combined["ls"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"beta": np.nan, "beta_t": np.nan, "n": len(combined)}
    beta = float(coeffs[1])
    y_hat = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n = len(y)
    if n <= 2 or np.std(x, ddof=1) == 0:
        return {"beta": beta, "beta_t": np.nan, "n": n}
    se = (np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t, "n": n}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 5: Main analysis
# =============================================================================

def main():
    print("=" * 90)
    print("S16: MULTI-SLEEVE BLEND (GP/A + S8 + S7-original)")
    print("Three independently beta-neutral sleeves, pre-registered equal 1/3 blend")
    print("=" * 90)

    sleeves = load_sleeves()
    gp, s7, s8 = sleeves["gp"], sleeves["s7"], sleeves["s8"]

    quarter_dates = gp["rebalance_date"].tolist()
    assert quarter_dates == s7["rebalance_date"].tolist(), "GP/A and S7 quarterly grids must match exactly"

    print("\n[Step 1] Compounding S8's monthly returns onto the quarterly grid...")
    s8q = compound_s8_to_quarterly(s8, quarter_dates)

    print("\n[Step 2] Loading IWM (Russell 2000 ETF) benchmark, same quarterly grid...")
    iwm_q = iwm_quarterly_returns(quarter_dates)

    blend = gp[["rebalance_date", "ls_ret", "ls_gross"]].rename(
        columns={"ls_ret": "gp_net", "ls_gross": "gp_gross"})
    blend = blend.merge(
        s7[["rebalance_date", "ls_ret", "ls_gross"]].rename(columns={"ls_ret": "s7_net", "ls_gross": "s7_gross"}),
        on="rebalance_date", how="left")
    blend = blend.merge(s8q, on="rebalance_date", how="left")
    blend = blend.merge(iwm_q, on="rebalance_date", how="left")

    blend["blended_net"]   = (blend["gp_net"]   + blend["s7_net"]   + blend["s8_net_q"])   / 3.0
    blend["blended_gross"] = (blend["gp_gross"] + blend["s7_gross"] + blend["s8_gross_q"]) / 3.0
    blend["year"] = blend["rebalance_date"].dt.year

    # ── Correlation matrix (PROMINENT, reported first — the actual diagnostic) ──
    corr_df = blend[["gp_net", "s7_net", "s8_net_q"]].rename(
        columns={"gp_net": "GP/A", "s7_net": "S7 (original)", "s8_net_q": "S8"})
    corr_mat = corr_df.corr()

    sep = "=" * 90
    print(f"\n{sep}")
    print("PAIRWISE CORRELATION OF SLEEVE RETURNS (in-sample, quarterly-aligned) — DIAGNOSTIC, READ FIRST")
    print(sep)
    print(corr_mat.round(3).to_string())
    print("\nFor comparison, S15's merged-composite version of this same trio showed GP/A vs IVOL-Value")
    print("correlation = +0.563 (a major cause of that trial's diversification failure). This trial keeps")
    print("S7 in its ORIGINAL hard-gated double-sort form (not S15's continuous score) — see below whether")
    print("that construction difference alone changes the correlation outcome.")

    # ── Blended portfolio metrics ─────────────────────────────────────────────
    m_net   = compute_metrics(blend["blended_net"])
    m_gross = compute_metrics(blend["blended_gross"])
    beta_common = compute_market_beta(blend["blended_net"], blend["iwm_ret"])

    per_period_sr = float(blend["blended_net"].mean() / blend["blended_net"].std(ddof=1))
    win_rate = float((blend["blended_net"] > 0).mean())
    cost_drag_bps = float((blend["blended_gross"] - blend["blended_net"]).mean()) * 4 * 10_000

    # ── Per-sleeve betas vs the SAME common IWM benchmark (apples-to-apples) ──
    beta_gp_common = compute_market_beta(blend["gp_net"], blend["iwm_ret"])
    beta_s7_common = compute_market_beta(blend["s7_net"], blend["iwm_ret"])
    beta_s8_common = compute_market_beta(blend["s8_net_q"], blend["iwm_ret"])

    # ── Per-year beta (low power at n=4/year, reported as instructed, flagged) ─
    per_year_beta = []
    for yr, g in blend.groupby("year"):
        b = compute_market_beta(g["blended_net"], g["iwm_ret"], min_n=3)
        per_year_beta.append({"year": yr, "beta": b["beta"], "beta_t": b["beta_t"], "n": b["n"]})
    per_year_beta_df = pd.DataFrame(per_year_beta)

    print(f"\n{sep}")
    print("BLENDED PORTFOLIO — REALIZED BETA (vs IWM, common quarterly calendar)")
    print(sep)
    print(f"  Overall: beta={beta_common['beta']:+.4f}  t-stat={beta_common['beta_t']:+.3f}  (n={beta_common['n']})")
    print(f"\n  Per-year (n=4 quarters/year — LOW POWER, reported as requested but treat cautiously):")
    print(per_year_beta_df.round(4).to_string(index=False))

    print(f"\n{sep}")
    print("REALIZED BETA COMPARISON — blend vs each sleeve (SAME common IWM benchmark, apples-to-apples)")
    print(sep)
    print(f"  {'Portfolio':<20} {'Beta (vs IWM)':>15} {'t-stat':>10}")
    print(f"  {'Blend':<20} {beta_common['beta']:>15.4f} {beta_common['beta_t']:>10.3f}")
    print(f"  {'GP/A-alone':<20} {beta_gp_common['beta']:>15.4f} {beta_gp_common['beta_t']:>10.3f}")
    print(f"  {'S7-alone':<20} {beta_s7_common['beta']:>15.4f} {beta_s7_common['beta_t']:>10.3f}")
    print(f"  {'S8-alone':<20} {beta_s8_common['beta']:>15.4f} {beta_s8_common['beta_t']:>10.3f}")
    print("  (Each sleeve's ORIGINAL registry-logged beta — vs its own internal EW small-cap proxy,")
    print("   not IWM — is quoted separately below for reference; not directly comparable to this table.)")

    # ── Original registry numbers, quoted verbatim (not recomputed) ──────────
    reg = pd.read_csv(REGISTRY)
    def _reg_row(key):
        r = reg[reg["study"] == key]
        return r.iloc[-1] if not r.empty else None
    reg_gp = _reg_row("gp_S3-GP")
    reg_s7 = _reg_row("ivolvalue_S7")
    reg_s8 = _reg_row("residreversal_S8")

    print(f"\n{sep}")
    print("DIRECT COMPARISON — blend vs each sleeve's OWN ORIGINAL construction (native cadence, quoted from registry)")
    print(sep)
    header = f"  {'Metric':<28} {'BLEND':>12} {'GP/A(#8)':>12} {'S7(#13)':>12} {'S8(#14)':>12}"
    print(header)
    print(f"  {'-'*28} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Sharpe (net, ann.)':<28} {m_net['sharpe']:>12.3f} {reg_gp['sharpe']:>12.3f} {reg_s7['sharpe']:>12.3f} {reg_s8['sharpe']:>12.3f}")
    print(f"  {'CALMAR':<28} {m_net['calmar']:>12.3f} {reg_gp['calmar']:>12.3f} {reg_s7['calmar']:>12.3f} {reg_s8['calmar']:>12.3f}")
    print(f"  {'Ann. return (net)':<28} {m_net['ann_ret']:>12.2%} {reg_gp['ann_return_net']:>12.2%} {reg_s7['ann_return_net']:>12.2%} {reg_s8['ann_return_net']:>12.2%}")
    print(f"  {'Ann. return (gross)':<28} {m_gross['ann_ret']:>12.2%} {reg_gp['ann_return_gross']:>12.2%} {reg_s7['ann_return_gross']:>12.2%} {reg_s8['ann_return_gross']:>12.2%}")
    print(f"  {'Market beta (own proxy)':<28} {'n/a':>12} {reg_gp['market_beta']:>12.4f} {reg_s7['market_beta']:>12.4f} {reg_s8['market_beta']:>12.4f}")
    print(f"  {'Market beta (vs IWM)':<28} {beta_common['beta']:>12.4f} {beta_gp_common['beta']:>12.4f} {beta_s7_common['beta']:>12.4f} {beta_s8_common['beta']:>12.4f}")
    print(f"  {'Max drawdown':<28} {m_net['max_dd']:>12.2%} {reg_gp['max_drawdown']:>12.2%} {reg_s7['max_drawdown']:>12.2%} {reg_s8['max_drawdown']:>12.2%}")
    print(f"  {'Mean IC':<28} {'n/a':>12} {reg_gp['mean_ic']:>12.4f} {reg_s7['mean_ic']:>12.4f} {reg_s8['mean_ic']:>12.4f}")
    print(f"  {'IC t-stat':<28} {'n/a':>12} {reg_gp['ic_t_stat']:>12.3f} {reg_s7['ic_t_stat']:>12.3f} {reg_s8['ic_t_stat']:>12.3f}")
    print(f"  {'Win rate':<28} {win_rate:>12.1%} {reg_gp['win_rate']:>12.1%} {reg_s7['win_rate']:>12.1%} {reg_s8['win_rate']:>12.1%}")
    print(f"  {'Mean turnover (own cadence)':<28} {'n/a':>12} {reg_gp['mean_turnover']:>12.1%} {reg_s7['mean_turnover']:>12.1%} {reg_s8['mean_turnover']:>12.1%}")
    print(f"  {'Cost drag (bps/yr)':<28} {cost_drag_bps:>12.0f} {reg_gp['annual_cost_drag_bps']:>12.0f} {reg_s7['annual_cost_drag_bps']:>12.0f} {reg_s8['annual_cost_drag_bps']:>12.0f}")
    print(f"  {'N periods (own cadence)':<28} {'24 (qtr)':>12} {'24 (qtr)':>12} {'24 (qtr)':>12} {'70 (mo)':>12}")

    # ── Year-by-year: blend vs IWM (Russell 2000) ─────────────────────────────
    yby_rows = []
    for yr, g in blend.groupby("year"):
        blend_yr = float((1 + g["blended_net"]).prod() - 1)
        iwm_yr   = float((1 + g["iwm_ret"]).prod() - 1) if g["iwm_ret"].notna().all() else np.nan
        gp_yr    = float((1 + g["gp_net"]).prod() - 1)
        s7_yr    = float((1 + g["s7_net"]).prod() - 1)
        s8_yr    = float((1 + g["s8_net_q"]).prod() - 1)
        yby_rows.append({"year": yr, "blend_net": blend_yr, "iwm_ret": iwm_yr,
                          "gp_net": gp_yr, "s7_net": s7_yr, "s8_net": s8_yr})
    yby = pd.DataFrame(yby_rows)

    print(f"\n{sep}")
    print("YEAR-BY-YEAR: BLEND vs RUSSELL 2000 (IWM) — plus each sleeve for context")
    print(sep)
    print(f"  {'Year':<6} {'Blend(net)':>12} {'IWM(R2000)':>12} {'GP/A':>10} {'S7':>10} {'S8':>10}")
    for _, r in yby.iterrows():
        print(f"  {int(r['year']):<6} {r['blend_net']:>11.2%} {r['iwm_ret']:>11.2%} "
              f"{r['gp_net']:>9.2%} {r['s7_net']:>9.2%} {r['s8_net']:>9.2%}")

    # ── 2020 sub-period explicit check ────────────────────────────────────────
    r2020 = yby[yby["year"] == 2020].iloc[0]
    q2020 = blend[blend["year"] == 2020][["rebalance_date", "blended_net", "gp_net", "s7_net", "s8_net_q", "iwm_ret"]]
    print(f"\n{sep}")
    print("2020 SUB-PERIOD CHECK (COVID crash + recovery)")
    print(sep)
    print(f"  Full-year 2020: Blend={r2020['blend_net']:+.2%}  IWM={r2020['iwm_ret']:+.2%}  "
          f"GP/A={r2020['gp_net']:+.2%}  S7={r2020['s7_net']:+.2%}  S8={r2020['s8_net']:+.2%}")
    print(f"\n  Quarter-by-quarter 2020:")
    print(q2020.round(4).to_string(index=False))

    # ── Sign-convention audit (quoted, not recomputed — nothing changed) ──────
    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT (quoted from each sleeve's own already-validated registry entry)")
    print(sep)
    print(f"  GP/A (#8):  mean_ic={reg_gp['mean_ic']:+.4f}  ic_t={reg_gp['ic_t_stat']:+.3f}  (positive, as originally logged)")
    print(f"  S7   (#13): mean_ic={reg_s7['mean_ic']:+.4f}  ic_t={reg_s7['ic_t_stat']:+.3f}  (positive, as originally logged —")
    print(f"              kept in ORIGINAL double-sort form this trial, unlike S15's continuous conversion which flipped it to -0.0151)")
    print(f"  S8   (#14): mean_ic={reg_s8['mean_ic']:+.4f}  ic_t={reg_s8['ic_t_stat']:+.3f}  (positive, as originally logged)")
    print("  All three positive and unchanged from their original trials — no re-derivation risk since")
    print("  each sleeve's signal/construction code was not touched for this trial.")

    # ── Sleeve-level concentration check on the blend ─────────────────────────
    contrib = blend[["gp_net", "s7_net", "s8_net_q"]].abs() / 3.0
    total = contrib.sum(axis=1)
    top1_share = contrib.max(axis=1) / total
    dominant_sleeve = contrib.idxmax(axis=1)
    conc_df = pd.DataFrame({
        "rebalance_date": blend["rebalance_date"], "top1_share": top1_share, "dominant_sleeve": dominant_sleeve,
    })
    n_over_50 = int((top1_share > 0.50).sum())
    n_over_60 = int((top1_share > 0.60).sum())
    dom_counts = dominant_sleeve.value_counts()

    print(f"\n{sep}")
    print("SLEEVE-LEVEL CONCENTRATION CHECK (blended portfolio — is the blend secretly one sleeve?)")
    print(sep)
    print(f"  Mean top-1-sleeve share of |return| mass: {top1_share.mean():.1%}  |  Max: {top1_share.max():.1%}")
    print(f"  Quarters with one sleeve > 50% of |return| mass: {n_over_50}/24")
    print(f"  Quarters with one sleeve > 60% of |return| mass: {n_over_60}/24")
    print(f"  Dominant-sleeve count across 24 quarters:\n{dom_counts.to_string()}")
    print("  (Equal 1/3 capital weight is fixed by construction — this measures how much of the blend's")
    print("   quarter-to-quarter |return| variation traces to one sleeve vs. genuine three-way blending;")
    print("   it is NOT a within-sleeve, per-stock concentration check — that would require re-running")
    print("   each sleeve with new per-name instrumentation, which conflicts with reusing their exact")
    print("   original, already-validated construction unchanged.)")

    # ── DSR — re-check trial count IMMEDIATELY before logging (numbering lesson) ──
    reg_fresh = pd.read_csv(REGISTRY)
    current_max_n = int(reg_fresh["n_trials_dsr"].max())
    n_trials = current_max_n + 1
    dsr_thr = deflated_sharpe_threshold(n_trials, m_net["n"])
    clears_dsr = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False

    print(f"\n{sep}")
    print(f"DSR CHECK — re-read registry immediately before logging: current max n_trials_dsr={current_max_n} -> this trial = #{n_trials}")
    print(sep)
    print(f"  Per-period SR: {per_period_sr:.4f}  |  DSR threshold (N={n_trials}, n_obs={m_net['n']}): {dsr_thr:.4f}  |  Clears DSR: {'YES' if clears_dsr else 'NO'}")

    promotion_met = (m_net["calmar"] >= 1.0) or (m_net["sharpe"] >= 0.8)
    print(f"\n  Promotion criteria (CALMAR>=1 or Sharpe>=0.8): {'MET' if promotion_met else 'NOT MET'}")

    # ── Fix verdict (does this construction fix S15's two diagnosed problems?) ─
    corr_gp_s7 = float(corr_mat.loc["GP/A", "S7 (original)"])
    corr_gp_s8 = float(corr_mat.loc["GP/A", "S8"])
    corr_s7_s8 = float(corr_mat.loc["S7 (original)", "S8"])
    fix1_correlation = abs(corr_gp_s7) < 0.3   # "meaningfully lower than S15's +0.563" bar
    fix2_s7_sign = float(reg_s7["mean_ic"]) > 0  # S7 kept original form -> should trivially hold

    print(f"\n{sep}")
    print("VERDICT: DID THIS FIX S15's TWO DIAGNOSED PROBLEMS?")
    print(sep)
    print(f"  Problem 1 (GP/A vs IVOL-Value correlation +0.563 in S15) -> this trial's GP/A vs S7 correlation = {corr_gp_s7:+.3f}")
    print(f"    {'FIXED' if fix1_correlation else 'PERSISTS'}: correlation is {'meaningfully lower' if fix1_correlation else 'still material'} than S15's +0.563.")
    print(f"  Problem 2 (S7 continuous-score conversion flipped IC to -0.015 in S15) -> this trial keeps S7's")
    print(f"    original double-sort, IC = {reg_s7['mean_ic']:+.4f} (unchanged from its own original trial #13).")
    print(f"    {'FIXED' if fix2_s7_sign else 'PERSISTS'}: S7's edge is preserved in its native construction (trivially true by design — not re-derived).")
    print(f"\n  ONE-LINE VERDICT: Blend Sharpe={m_net['sharpe']:.3f}, CALMAR={m_net['calmar']:.3f} — "
          f"{'MEETS' if promotion_met else 'MISSES'} the Sharpe>=0.8/CALMAR~1 promotion bar; "
          f"{'CLEARS' if clears_dsr else 'FAILS'} DSR at N={n_trials}.")

    # ── Save outputs ───────────────────────────────────────────────────────────
    blend.to_csv(RESULTS / "s16_blended_period_returns.csv", index=False)
    corr_mat.to_csv(RESULTS / "s16_sleeve_correlation.csv")
    yby.to_csv(RESULTS / "s16_year_by_year.csv", index=False)
    conc_df.to_csv(RESULTS / "s16_concentration.csv", index=False)
    per_year_beta_df.to_csv(RESULTS / "s16_per_year_beta.csv", index=False)
    print(f"\n[Saved] Results -> {RESULTS}/")

    summary = {
        "m_net": m_net, "m_gross": m_gross, "beta_common": beta_common,
        "beta_gp_common": beta_gp_common, "beta_s7_common": beta_s7_common, "beta_s8_common": beta_s8_common,
        "per_period_sr": per_period_sr, "win_rate": win_rate, "cost_drag_bps": cost_drag_bps,
        "n_trials": n_trials, "dsr_thr": dsr_thr, "clears_dsr": clears_dsr, "promotion_met": promotion_met,
        "corr_mat": corr_mat, "corr_gp_s7": corr_gp_s7, "corr_gp_s8": corr_gp_s8, "corr_s7_s8": corr_s7_s8,
        "yby": yby, "conc_df": conc_df, "reg_gp": reg_gp, "reg_s7": reg_s7, "reg_s8": reg_s8,
        "fix1_correlation": fix1_correlation, "fix2_s7_sign": fix2_s7_sign,
        "n_partial_s8_quarters": int((s8q["s8_n_months"] < 3).sum()),
        "current_max_n_before": current_max_n,
    }
    return summary


def log_registry(s: dict) -> None:
    reg_fresh = pd.read_csv(REGISTRY)
    current_max_n = int(reg_fresh["n_trials_dsr"].max())
    if current_max_n != s["current_max_n_before"]:
        print(f"[WARN] Registry count changed since analysis ({s['current_max_n_before']} -> {current_max_n}) "
              f"— another concurrent trial logged in between. Recomputing DSR at the true current N.")
    n_trials = current_max_n + 1
    dsr_thr = deflated_sharpe_threshold(n_trials, s["m_net"]["n"])
    clears_dsr = bool(s["per_period_sr"] > dsr_thr) if not np.isnan(s["per_period_sr"]) and not np.isnan(dsr_thr) else False

    notes = (
        f"Blend Sharpe={s['m_net']['sharpe']:.4f} CALMAR={s['m_net']['calmar']:.4f} vs sleeves "
        f"GP/A(#8)={s['reg_gp']['sharpe']:.4f}, S7(#13)={s['reg_s7']['sharpe']:.4f}, S8(#14)={s['reg_s8']['sharpe']:.4f}. "
        f"Sleeve pairwise correlation: GP/S7={s['corr_gp_s7']:+.3f}, GP/S8={s['corr_gp_s8']:+.3f}, S7/S8={s['corr_s7_s8']:+.3f} "
        f"(S15's merged-composite version of GP/IVOLValue was +0.563 -- {'improved' if s['fix1_correlation'] else 'did not improve'} here). "
        f"S7 kept in ORIGINAL double-sort form (IC={s['reg_s7']['mean_ic']:+.4f}, unchanged from trial #13) -- "
        f"S15's continuous-score conversion of S7 flipped this to -0.0151; that failure mode does not apply here by construction. "
        f"Blend beta vs IWM={s['beta_common']['beta']:+.4f} (t={s['beta_common']['beta_t']:+.3f}) vs sleeve betas (same IWM benchmark) "
        f"GP/A={s['beta_gp_common']['beta']:+.4f}, S7={s['beta_s7_common']['beta']:+.4f}, S8={s['beta_s8_common']['beta']:+.4f}. "
        f"Promotion criteria (CALMAR>=1 or Sharpe>=0.8) {'MET' if s['promotion_met'] else 'NOT MET'}. "
        f"{'CLEARS' if clears_dsr else 'FAILS'} DSR at N={n_trials}. "
        f"UNIVERSE NOTE: PIT Russell 2000 constituent membership was requested but not reconstructable from "
        f"research/russell_reconstitution/ (annual R3000 union additions/deletions EVENTS only, no full roster, "
        f"no quarterly granularity, 2018/2019 deletions gap) -- confirmed with user, fell back to standard "
        f"$100M-$2B PIT universe (same as every sleeve's own original construction). IWM (Russell 2000 ETF) used "
        f"as the year-by-year / beta benchmark instead, reusing the Russell reconstitution trial's already-fetched cache. "
        f"S8's Q1-2015 blend quarter is partial ({s['n_partial_s8_quarters']} of 24 quarters affected): S8's own native "
        f"backtest starts 2015-03-01 (needs 24mo+ history), so Jan/Feb 2015 contribute 0% from that sleeve."
    )

    row = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "study": "multisleeve_S16",
        "hypothesis": (
            "Three independently beta-neutral sleeves (GP/A quality, S8 residual reversal, S7 IVOL-conditioned "
            "value kept in its ORIGINAL hard-gated double-sort form), blended at pre-registered equal 1/3 capital "
            "weight, should achieve genuine diversification where S15's merged rank-percentile composite of the "
            "same three signals failed -- directly testing S15's own diagnosed causes of failure (component "
            "correlation, and S7's sign flip under continuous-score conversion)."
        ),
        "data_source": (
            "Fully reused, no new data: research/gp/results/gp_period_returns.csv (trial #8), "
            "research/ivol_value/results/s7_period_returns.csv (trial #13), "
            "research/residual_reversal/results/s8_period_returns.csv (trial #14), "
            "research/russell_reconstitution/cache/tiingo/IWM.csv (benchmark)"
        ),
        "status": "completed_in_sample",
        "trial_number": n_trials,
        "trial_note": (
            "New independent trial. Directly follows up compositeIC_S15 (trial #19) by fixing its two "
            "diagnosed failure causes via separate-sleeve construction instead of merged scoring."
        ),
        "n_trials_dsr": n_trials,
        "rebalance_freq": "quarterly (blend-level reporting calendar; each sleeve keeps its own native cadence internally -- GP/A/S7 quarterly, S8 monthly with turnover-discipline carry)",
        "rebalance_note": "Pre-registered equal 1/3 capital weight per sleeve, rebalanced quarterly to maintain equal weight. Not tuned after seeing correlation or performance.",
        "portfolio_note": (
            "Each sleeve independently beta-neutral using its own original construction (GP/A: 60-day OLS vs EW "
            "universe; S7: reused IVOL-regression beta; S8: 60-day OLS vs EW universe). No merged score, no "
            "rank-percentile averaging -- three separate books blended at the return level."
        ),
        "ann_return_net": s["m_net"]["ann_ret"],
        "ann_return_gross": s["m_gross"]["ann_ret"],
        "annual_cost_drag_bps": s["cost_drag_bps"],
        "mean_turnover": np.nan,
        "sharpe": s["m_net"]["sharpe"],
        "calmar": s["m_net"]["calmar"],
        "max_drawdown": s["m_net"]["max_dd"],
        "t_stat": s["m_net"]["t_stat"],
        "market_beta": s["beta_common"]["beta"],
        "market_beta_tstat": s["beta_common"]["beta_t"],
        "mean_ic": np.nan,
        "ic_t_stat": np.nan,
        "win_rate": s["win_rate"],
        "n_periods": s["m_net"]["n"],
        "per_period_sr": s["per_period_sr"],
        "dsr_threshold": dsr_thr,
        "clears_dsr": clears_dsr,
        "signal_note": (
            "Blend = equal-weight (1/3 each) of GP/A (#8), S7-original-double-sort (#13), S8 (#14) quarterly "
            "returns; S8's monthly returns compounded onto the quarterly grid."
        ),
        "notes": notes,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "multisleeve_S16"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] multisleeve_S16 logged (trial #{n_trials}, N_TRIALS_DSR={n_trials}) -> {REGISTRY}")


if __name__ == "__main__":
    summary = main()
    log_registry(summary)
