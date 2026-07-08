"""
S3-SA Walk-Forward Validation (2021-01-01 -> 2025-01-01, RESERVED window)
===========================================================================
Purpose: out-of-sample test of gp_S3b-semiannual (trial #8b) — the ONLY
Phase-3 trial that cleared its in-sample DSR gate. This is the first time
the reserved 2021-2025 walk-forward window has been touched for ANY trial
in this program.

CRITICAL — NO RE-OPTIMIZATION. This script imports the exact signal,
universe, portfolio-construction, and cost-model primitives from run.py
(the original S3-GP implementation) UNCHANGED. Only the rebalance-date
range and the borrow-cost periods-per-year divisor (2, for semiannual —
identical to what run_decay.py used for the in-sample S3-SA cut) differ.
Nothing about GP/A computation, quintile cutoffs, beta-neutral sizing, or
the bid-ask/borrow cost tiers is touched or re-tuned based on how this
window performs.

Reserved window: 2021-01-01 -> 2025-01-01 (NOT extended to the full cache,
which runs through mid-2026 — the pre-registered walk-forward window is
exactly 4 years / 8 semiannual periods, matching what was reserved when
S3-SA was declared closed to further in-sample tuning).

This run does NOT consume a new DSR trial slot — it is the walk-forward
test of existing trial #8b, not a new hypothesis. N_TRIALS stays 8.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import (
    load_tiingo_tickers, load_prefilter,
    preload_all_prices, preload_all_adj_prices, preload_gp_facts,
    build_universe, compute_gp_signals, compute_betas,
    compute_returns, build_portfolio,
    _spread, _borrow,
    compute_market_beta, deflated_sharpe_threshold,
    N_TRIALS, CACHE, RESULTS, REGISTRY,
    MCAP_MIN, MCAP_MAX,
)

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

# =============================================================================
# Reserved walk-forward window — semiannual only (the frequency that cleared
# the in-sample gate). NOT a sweep — S3-SA locked semiannual; re-testing
# quarterly/annual here would itself be a form of re-optimization.
# =============================================================================

WF_START = pd.Timestamp("2021-01-01")
WF_END   = pd.Timestamp("2025-01-01")
PPY      = 2   # semiannual

_all_wf = pd.date_range(WF_START, WF_END, freq="6MS")
WF_REBALANCE_DATES = [t for t in _all_wf if t < WF_END]

# In-sample S3-SA reference values (trial #8b, already logged in registry —
# pulled here for the side-by-side comparison table, NOT recomputed).
IS_LABEL       = "In-Sample (2015-2021)"
IS_N_PERIODS   = 12
IS_MEAN_IC     = 0.0540866218958777
IS_IC_TSTAT    = 1.6970277603763435
IS_ANN_RET_NET = 0.0685094865946969
IS_ANN_RET_GROSS = 0.1016462707436853
IS_COST_DRAG_BPS = 318.1656834005664
IS_SHARPE      = 0.5694996798222055
IS_CALMAR      = 0.3825328011454172
IS_MAX_DD      = -0.179094410700361
IS_TSTAT       = 1.3949836242427962
IS_MARKET_BETA = -0.0780053327960194
IS_BETA_TSTAT  = -0.4479010624107541
IS_MEAN_TURNOVER = 0.3295210297734618
IS_WIN_RATE    = 0.6666666666666666
IS_PER_PERIOD_SR = 0.4026970854858491
IS_DSR_THRESHOLD = 0.3320772622111037   # per-period SR units, N_TRIALS=8, n_obs=12
IS_CLEARS_DSR  = True

# In-sample frequency-sweep IC values (from the original S3-GP frequency
# robustness check) — for the IC-decay check.
IS_IC_QUARTERLY   = 0.0404399415571849   # gp_S3-GP, trial #8
IS_IC_SEMIANNUAL  = IS_MEAN_IC            # 0.0541 — same value, same run
IS_IC_ANNUAL      = 0.0669024616247168   # from gp_S3-GP registry note "0.040 -> 0.054 -> 0.067"


# =============================================================================
# Frequency-parameterized leg return (identical to run_decay.py's _leg_return_ppy)
# =============================================================================

def _leg_return_ppy(returns_df, tickers, weights, is_short, turnover, periods_per_year):
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid      = (MCAP_MIN + MCAP_MAX) / 2
    gross = ba_cost = borrow_cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross   += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / periods_per_year) * w
    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    return (gross - total_cost), gross, ba_cost, borrow_cost


def compute_metrics_ppy(ls_series: pd.Series, ppy: int) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n  = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "max_dd": np.nan, "t_stat": np.nan,
                "win_rate": np.nan, "n": n}
    cum     = (1 + ls).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (ppy / n) - 1)
    mu      = float(ls.mean())
    sigma   = float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(ppy)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s    = (cum - roll_mx) / roll_mx
    max_dd  = float(dd_s.min())
    calmar  = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat  = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt  = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "t_stat": t_stat,
            "win_rate": win_rt, "n": n}


# =============================================================================
# Walk-forward backtest engine (semiannual only, reserved window)
# =============================================================================

def run_walkforward(tiingo_tickers: pd.DataFrame, prefilter_df: pd.DataFrame) -> dict:
    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list = []
    sector_contrib: dict = {}
    sign_violations = 0
    prev_long_set:  set = set()
    prev_short_set: set = set()

    n_periods = len(WF_REBALANCE_DATES)
    for i, t in enumerate(WF_REBALANCE_DATES):
        hold_end = WF_REBALANCE_DATES[i + 1] if i + 1 < n_periods else WF_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} → {hold_end.date()}")

        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        signals_df, cov = compute_gp_signals(univ, t)
        pct_drop = cov["pct_dropped"]

        univ_sig = univ.merge(signals_df[["ticker", "gp_signal"]], on="ticker", how="left")
        try:
            returns_df = compute_returns(univ_sig, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())
        betas = compute_betas(univ, t)

        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short) = build_portfolio(returns_df, betas)
        if not longs:
            print(f"  [SKIP] insufficient names for portfolio")
            continue

        # ── Sign-convention audit: long (high GP/A) must exceed short (low) ──
        gp_map = returns_df.set_index("ticker")["gp_signal"]
        mean_gp_long  = float(gp_map.reindex(longs).mean())
        mean_gp_short = float(gp_map.reindex(shorts).mean())
        sign_ok = mean_gp_long > mean_gp_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg GP/A ({mean_gp_long:.4f}) "
                  f"<= short-leg GP/A ({mean_gp_short:.4f}) ***")

        long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1) if prev_long_set  else 1.0
        short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1) if prev_short_set else 1.0
        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_ppy(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover, periods_per_year=PPY)
        short_net, short_gross, short_ba, short_bw = _leg_return_ppy(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover, periods_per_year=PPY)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        valid_sig = returns_df.dropna(subset=["gp_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["gp_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        # ── Sector contribution (concentration check) ────────────────────────
        # build_universe() in run.py does not carry a "sic" column through —
        # look it up from cik_sic_map (ticker -> (cik, sic)) instead.
        sic_map = {tk: cik_sic_map.get(tk, (None, None))[1] for tk in returns_df["ticker"]}
        ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
        for tk in longs:
            r = ret_map.get(tk, np.nan)
            if np.isnan(r):
                continue
            sic = sic_map.get(tk, np.nan)
            grp = int(sic // 100) if pd.notna(sic) else -1
            sector_contrib[grp] = sector_contrib.get(grp, 0.0) + long_scale * lw.get(tk, 0.0) * r
        for tk in shorts:
            r = ret_map.get(tk, np.nan)
            if np.isnan(r):
                continue
            sic = sic_map.get(tk, np.nan)
            grp = int(sic // 100) if pd.notna(sic) else -1
            sector_contrib[grp] = sector_contrib.get(grp, 0.0) - short_scale * sw.get(tk, 0.0) * r

        print(f"  U={len(returns_df)} nL={len(longs)} nS={len(shorts)} "
              f"L/S={ls_net:+.2%} gr={ls_gross:+.2%} to={avg_turnover:.0%} "
              f"IC={ic:.3f} Dlist={n_delisted} gpL={mean_gp_long:.4f} gpS={mean_gp_short:.4f}")

        period_records.append({
            "rebalance_date": t, "hold_end": hold_end, "year": t.year,
            "n_universe": len(returns_df), "n_with_signal": int(returns_df["gp_signal"].notna().sum()),
            "n_delisted": n_delisted, "pct_dropped": pct_drop,
            "long_ret": long_net, "short_ret": short_net,
            "ls_ret": ls_net, "ls_gross": ls_gross,
            "ls_cost": ls_ba + ls_bw, "ls_ba_cost": ls_ba, "ls_borrow_cost": ls_bw,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
            "avg_turnover": avg_turnover, "mkt_ret": mkt_ret, "ic": ic,
            "n_long": len(longs), "n_short": len(shorts),
            "mean_gp_long": mean_gp_long, "mean_gp_short": mean_gp_short, "sign_ok": sign_ok,
        })

    return {
        "period_records": period_records, "sector_contrib": sector_contrib,
        "sign_violations": sign_violations,
    }


# =============================================================================
# Output
# =============================================================================

def print_results(results: dict) -> dict:
    records = results["period_records"]
    if not records:
        print("\n[ERROR] No periods completed.")
        return {}

    df = pd.DataFrame(records)
    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m       = compute_metrics_ppy(ls_series, PPY)
    m_gross = compute_metrics_ppy(gross_series, PPY)
    beta_s  = compute_market_beta(ls_series, mkt_series)

    sigma = float(ls_series.std(ddof=1))
    per_period_sr = float(ls_series.mean() / sigma) if sigma > 0 else np.nan

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)

    ba_bps  = float(df["ls_ba_cost"].mean())     * PPY * 10_000
    bw_bps  = float(df["ls_borrow_cost"].mean()) * PPY * 10_000
    cost_bps = ba_bps + bw_bps
    mean_turnover = float(df["avg_turnover"].mean())

    sign_violations = results["sign_violations"]
    n_periods = len(df)

    sep = "=" * 78
    print(f"\n{sep}")
    print("S3-SA WALK-FORWARD — FULL RESULTS TABLE (2021-01-01 -> 2025-01-01, semiannual)")
    print(sep)

    print(f"\n{'─'*78}")
    print("SIGN-CONVENTION AUDIT")
    print(f"{'─'*78}")
    print(f"  Periods where long-leg GP/A <= short-leg GP/A (violation): {sign_violations}/{n_periods}")
    print("  PASS — long leg (high GP/A) always exceeds short leg." if sign_violations == 0
          else "  *** FAIL — sign convention broken; results may be inverted. ***")

    print(f"\n{'─'*78}")
    print("PERIOD-LEVEL DETAIL")
    print(f"{'─'*78}")
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} {'TOver':>6} "
          f"{'LScl':>5} {'SScl':>5} {'Mkt':>8} {'IC':>6} {'nL':>3}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*5} {'-'*8} {'-'*6} {'-'*3}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
              f"{r['avg_turnover']:>5.0%} {r['long_scale']:>4.2f}× {r['short_scale']:>4.2f}× "
              f"{r['mkt_ret']:>7.2%} {r['ic']:>6.3f} {int(r['n_long']):>3}")

    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (semiannual):")
    for i, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " ◄ MAX DD" if abs(val - m["max_dd"]) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{marker}")

    print(f"\n{'─'*78}")
    print("SUMMARY METRICS")
    print(f"{'─'*78}")
    print(f"  {'Metric':<40} {'Value':>14}")
    print(f"  {'-'*40} {'-'*14}")
    print(f"  {'Periods (N)':<40} {n_periods:>14d}")
    print(f"  {'Mean IC (Spearman)':<40} {mean_ic:>14.4f}")
    print(f"  {'IC t-stat':<40} {ic_tstat:>14.3f}")
    print(f"  {'Annualized return (GROSS)':<40} {m_gross['ann_ret']:>13.2%}")
    print(f"  {'Cost drag (bps/yr)':<40} {cost_bps:>13.0f}")
    print(f"    {'bid-ask (bps/yr)':<38} {ba_bps:>13.0f}")
    print(f"    {'borrow  (bps/yr)':<38} {bw_bps:>13.0f}")
    print(f"  {'Annualized return (net)':<40} {m['ann_ret']:>13.2%}")
    print(f"  {'Sharpe (annualized, net)':<40} {m['sharpe']:>13.3f}")
    print(f"  {'CALMAR':<40} {m['calmar']:>13.3f}")
    print(f"  {'Max drawdown':<40} {m['max_dd']:>13.2%}")
    print(f"  {'t-stat (period returns)':<40} {m['t_stat']:>13.3f}")
    print(f"  {'Market beta (vs EW universe)':<40} {beta_s['beta']:>13.3f}")
    print(f"    {'beta t-stat':<38} {beta_s['beta_t']:>13.3f}")
    print(f"  {'Mean turnover / period':<40} {mean_turnover:>13.0%}")
    print(f"  {'Win rate':<40} {m['win_rate']:>13.1%}")

    print(f"\n{'─'*78}")
    print("DSR REFERENCE (informational — this is a walk-forward test, not a fresh in-sample trial)")
    print(f"{'─'*78}")
    print(f"  Walk-forward per-period SR:                    {per_period_sr:>10.4f}")
    print(f"  In-sample DSR threshold (N=8, n_obs=12):        {IS_DSR_THRESHOLD:>10.4f}")
    print(f"  Walk-forward SR vs in-sample threshold:  {'ABOVE' if per_period_sr > IS_DSR_THRESHOLD else 'BELOW':>10}")
    wf_dsr_thr_own = deflated_sharpe_threshold(N_TRIALS, n_periods)
    print(f"  (informational) recomputed threshold at N=8, n_obs={n_periods}: {wf_dsr_thr_own:.4f}")
    print(f"  Note: DSR is a multiple-testing correction for IN-SAMPLE selection. Applying it")
    print(f"  to an out-of-sample window is not a formal test — reported only as a reference")
    print(f"  point against the bar the strategy had to clear in-sample.")

    return {
        "n_periods": n_periods, "mean_ic": mean_ic, "ic_tstat": ic_tstat,
        "ann_ret_gross": m_gross["ann_ret"], "ann_ret_net": m["ann_ret"],
        "cost_drag_bps": cost_bps, "ba_bps": ba_bps, "bw_bps": bw_bps,
        "sharpe": m["sharpe"], "calmar": m["calmar"], "max_dd": m["max_dd"],
        "t_stat": m["t_stat"], "market_beta": beta_s["beta"], "beta_tstat": beta_s["beta_t"],
        "mean_turnover": mean_turnover, "win_rate": m["win_rate"],
        "per_period_sr": per_period_sr, "df": df,
    }


def print_comparison_table(wf: dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print("IN-SAMPLE (2015-2021) vs WALK-FORWARD (2021-2025) — SIDE BY SIDE")
    print(sep)

    def pct(v): return f"{v:.2%}" if v is not None and not np.isnan(v) else "—"
    def f3(v):  return f"{v:.3f}" if v is not None and not np.isnan(v) else "—"
    def bps(v): return f"{v:.0f}" if v is not None and not np.isnan(v) else "—"

    w0, w1 = 34, 16
    print(f"  {'Metric':<{w0}} {'In-Sample':>{w1}} {'Walk-Forward':>{w1}} {'Delta':>{w1}}")
    print(f"  {'-'*w0} {'-'*w1} {'-'*w1} {'-'*w1}")

    def row(label, is_v, wf_v, fmt):
        d = wf_v - is_v if (is_v is not None and wf_v is not None
                             and not np.isnan(is_v) and not np.isnan(wf_v)) else np.nan
        d_str = fmt(d) if not (isinstance(d, float) and np.isnan(d)) else "—"
        print(f"  {label:<{w0}} {fmt(is_v):>{w1}} {fmt(wf_v):>{w1}} {d_str:>{w1}}")

    row("Periods (N)", IS_N_PERIODS, wf["n_periods"], lambda v: str(int(v)) if v is not None and not (isinstance(v,float) and np.isnan(v)) else "—")
    row("Mean IC (Spearman)", IS_MEAN_IC, wf["mean_ic"], f3)
    row("IC t-stat", IS_IC_TSTAT, wf["ic_tstat"], f3)
    row("Ann. return (GROSS)", IS_ANN_RET_GROSS, wf["ann_ret_gross"], pct)
    row("Cost drag (bps/yr)", IS_COST_DRAG_BPS, wf["cost_drag_bps"], bps)
    row("Ann. return (net)", IS_ANN_RET_NET, wf["ann_ret_net"], pct)
    row("Sharpe (annualized, net)", IS_SHARPE, wf["sharpe"], f3)
    row("CALMAR", IS_CALMAR, wf["calmar"], f3)
    row("Max drawdown", IS_MAX_DD, wf["max_dd"], pct)
    row("t-stat (period returns)", IS_TSTAT, wf["t_stat"], f3)
    row("Market beta", IS_MARKET_BETA, wf["market_beta"], f3)
    row("Market beta t-stat", IS_BETA_TSTAT, wf["beta_tstat"], f3)
    row("Mean turnover / period", IS_MEAN_TURNOVER, wf["mean_turnover"], pct)
    row("Win rate", IS_WIN_RATE, wf["win_rate"], pct)
    row("Per-period SR", IS_PER_PERIOD_SR, wf["per_period_sr"], f3)


def print_ic_decay_check(wf: dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print("IC DECAY CHECK — was the frequency-selection finding signal or noise?")
    print(sep)
    print(f"  In-sample frequency sweep (original S3-GP robustness check):")
    print(f"    Quarterly  IC: {IS_IC_QUARTERLY:+.4f}")
    print(f"    Semiannual IC: {IS_IC_SEMIANNUAL:+.4f}   <- frequency that was selected as S3-SA")
    print(f"    Annual     IC: {IS_IC_ANNUAL:+.4f}")
    print(f"    (Rising IC with holding period was part of what made semiannual look best.)")
    print(f"\n  Walk-forward semiannual IC: {wf['mean_ic']:+.4f} (t={wf['ic_tstat']:.3f})")

    delta = wf["mean_ic"] - IS_IC_SEMIANNUAL
    print(f"  Delta vs in-sample semiannual IC: {delta:+.4f}")
    if wf["mean_ic"] < 0:
        verdict = ("IC HAS FLIPPED SIGN out-of-sample. The in-sample rising-IC-with-horizon "
                    "pattern does not replicate — strong evidence the frequency selection was "
                    "fitting in-sample noise, not a real horizon effect.")
    elif wf["mean_ic"] < IS_IC_SEMIANNUAL * 0.5:
        verdict = ("IC is positive but MEANINGFULLY LOWER than in-sample (less than half). "
                    "Consistent with regression to the mean expected from a best-of-3 frequency "
                    "selection — the in-sample level was likely inflated by selection.")
    elif wf["mean_ic"] < IS_IC_SEMIANNUAL:
        verdict = ("IC is lower than in-sample but in the same ballpark — some decay, "
                    "consistent with mild selection inflation but not a wholesale collapse.")
    else:
        verdict = ("IC held up at or above the in-sample semiannual level — notable given the "
                    "base-rate expectation of decay for a best-of-3 selected frequency.")
    print(f"\n  {verdict}")


def print_regime_and_concentration(wf: dict, results: dict) -> None:
    df = wf["df"]
    sep = "=" * 78
    print(f"\n{sep}")
    print("REGIME CHECK — does a single year/event dominate the walk-forward return?")
    print(sep)

    yearly = df.groupby("year")["ls_ret"].apply(lambda s: float((1 + s).prod() - 1))
    print(f"  Cumulative L/S return by year:")
    for yr, ret in yearly.items():
        print(f"    {yr}: {ret:+.2%}")

    ls_series = df["ls_ret"]
    abs_sum = float(ls_series.abs().sum())
    period_share = (ls_series.abs() / abs_sum) if abs_sum > 0 else ls_series * 0
    top_idx  = period_share.idxmax()
    top_share = float(period_share.loc[top_idx])
    top_date = df.loc[top_idx, "rebalance_date"]
    print(f"\n  Largest single half-year share of total |return| mass: "
          f"{top_date.date()} = {top_share:.1%}")
    if top_share > 0.35:
        print(f"  *** FLAG: single half-year dominates (>35%) — walk-forward result may be a "
              f"single-episode artifact, similar to how 2020 dominated several in-sample trials "
              f"(S1, S2, S7). ***")
    else:
        print(f"  No single half-year dominates (<35% threshold) — return is spread across "
              f"the window, not a single-episode artifact like the 2020 pattern seen elsewhere.")

    sector_contrib = results.get("sector_contrib", {})
    if sector_contrib:
        total_abs = sum(abs(v) for v in sector_contrib.values())
        if total_abs > 0:
            top_sic, top_val = max(sector_contrib.items(), key=lambda kv: abs(kv[1]))
            top_sec_share = abs(top_val) / total_abs
            print(f"\n  Largest single SIC major-group contribution: "
                  f"SIC {top_sic:02d} = {top_sec_share:.1%} of total |contribution|")
            if top_sec_share > 0.40:
                print(f"  *** FLAG: single sector dominates (>40%). ***")
            else:
                print(f"  No single sector dominates (<40% threshold).")

    # 2022 rate-hiking cycle flag specifically
    if 2022 in yearly.index:
        print(f"\n  2022 (rate-hiking cycle) return: {yearly[2022]:+.2%} "
              f"— flagged explicitly per the regime-check requirement.")


def print_verdict(wf: dict) -> None:
    sep = "=" * 78
    print(f"\n{sep}")
    print("HONEST VERDICT")
    print(sep)

    ic_holds     = wf["mean_ic"] > 0 and wf["mean_ic"] >= IS_IC_SEMIANNUAL * 0.5
    ic_flipped   = wf["mean_ic"] < 0
    sharpe_holds = wf["sharpe"] > 0 and wf["sharpe"] >= IS_SHARPE * 0.5
    sharpe_negative = wf["sharpe"] < 0

    if ic_flipped or sharpe_negative:
        verdict = (
            "The in-sample S3-SA edge does NOT appear to survive out-of-sample. "
            f"Walk-forward IC is {wf['mean_ic']:+.4f} (t={wf['ic_tstat']:.2f}) versus "
            f"{IS_IC_SEMIANNUAL:+.4f} in-sample, and walk-forward net Sharpe is "
            f"{wf['sharpe']:.3f} versus {IS_SHARPE:.3f} in-sample. Given that S3-SA's "
            "in-sample pass was already a best-of-3 frequency selection (quarterly and "
            "annual were both weaker or failed), this is close to the base-rate expectation "
            "for a selection artifact, not a surprising result. The most defensible reading "
            "is that the in-sample DSR clear was primarily a product of choosing the best of "
            "three correlated frequency cuts on the same underlying signal and universe, and "
            "that GP/A does not have a robust, reusable edge in this small/mid-cap universe "
            "at semiannual rebalance. This trial should be treated as CLOSED — not retuned "
            "or re-tested at other frequencies using the walk-forward data, which would just "
            "repeat the original selection problem on a new sample."
        )
    elif ic_holds and sharpe_holds:
        verdict = (
            f"The in-sample S3-SA edge holds up reasonably well out-of-sample. Walk-forward "
            f"IC is {wf['mean_ic']:+.4f} (t={wf['ic_tstat']:.2f}) versus {IS_IC_SEMIANNUAL:+.4f} "
            f"in-sample, and walk-forward net Sharpe is {wf['sharpe']:.3f} versus "
            f"{IS_SHARPE:.3f} in-sample — same sign, same rough magnitude. That said, this is "
            "a single 4-year walk-forward window with only 8 semiannual observations — nowhere "
            "near enough to independently confirm a strategy on its own, and the in-sample pass "
            "was a best-of-3 selection, which raises the bar for what 'holding up' should mean "
            "here. This is encouraging evidence, not confirmation; the honest framing is that "
            "S3-SA has NOT been falsified by this test, but n is far too small to promote it "
            "past 'still plausible, worth another walk-forward window before deploying capital.'"
        )
    else:
        verdict = (
            f"The walk-forward result is mixed: IC is {wf['mean_ic']:+.4f} (t={wf['ic_tstat']:.2f}) "
            f"versus {IS_IC_SEMIANNUAL:+.4f} in-sample, and net Sharpe is {wf['sharpe']:.3f} versus "
            f"{IS_SHARPE:.3f} in-sample — meaningfully weaker on at least one dimension without "
            "an outright sign flip. This sits between clean survival and clean failure. Given that "
            "S3-SA's in-sample pass was a best-of-3 frequency selection (a higher bar than a clean "
            "independent trial), the base-rate expectation was exactly this kind of partial decay — "
            "some edge, meaningfully weaker than in-sample. The most defensible reading is that part "
            "of the in-sample result was a selection artifact, and what remains is not strong enough "
            "on its own to treat GP/A-semiannual as a validated strategy; n=8 periods is also too "
            "small to be more decisive than this."
        )
    print(f"\n  {verdict}")


def save_results(wf: dict) -> None:
    df = wf["df"]
    df.to_csv(RESULTS / "s3sa_walkforward_period_returns.csv", index=False)
    print(f"\n[Saved] {RESULTS / 's3sa_walkforward_period_returns.csv'}")


def log_registry(wf: dict) -> None:
    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "gp_S3-SA_walkforward",
        "hypothesis":           (
            "Out-of-sample walk-forward test of gp_S3b-semiannual (trial #8b) — the only "
            "Phase-3 trial that cleared its in-sample DSR gate. Reserved window "
            "2021-01-01 -> 2025-01-01. No re-optimization: identical signal, universe, "
            "portfolio construction, and cost model as the in-sample run."
        ),
        "data_source":          "Tiingo + SEC XBRL (PIT gated)",
        "status":               "completed_walkforward",
        "trial_number":         "8b-WF",
        "trial_note":           "Walk-forward validation of trial #8b. Not a new trial — N_TRIALS unchanged at 8.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "semiannual",
        "rebalance_note":       "Reserved out-of-sample window, pre-registered at S3-SA close. No frequency re-tuning performed.",
        "portfolio_note":       "Beta-neutral (EW market proxy, 60-day adjClose) — identical to in-sample S3-SA.",
        "signal_note":          "GP/A = TTM GrossProfit / Assets, filed <= t (identical construction to in-sample S3-SA).",
        "ann_return_net":       wf["ann_ret_net"],
        "ann_return_gross":     wf["ann_ret_gross"],
        "annual_cost_drag_bps": wf["cost_drag_bps"],
        "mean_turnover":        wf["mean_turnover"],
        "sharpe":               wf["sharpe"],
        "calmar":               wf["calmar"],
        "max_drawdown":         wf["max_dd"],
        "t_stat":               wf["t_stat"],
        "market_beta":          wf["market_beta"],
        "market_beta_tstat":    wf["beta_tstat"],
        "mean_ic":              wf["mean_ic"],
        "ic_t_stat":            wf["ic_tstat"],
        "win_rate":             wf["win_rate"],
        "n_periods":            wf["n_periods"],
        "per_period_sr":        wf["per_period_sr"],
        "dsr_threshold":        IS_DSR_THRESHOLD,
        "clears_dsr":           bool(wf["per_period_sr"] > IS_DSR_THRESHOLD),
        "notes":                (
            f"WALK-FORWARD (not in-sample). Compared against in-sample threshold "
            f"{IS_DSR_THRESHOLD:.4f} (N=8, n_obs=12) for reference only — DSR is an "
            f"in-sample multiple-testing correction, not a formal out-of-sample test."
        ),
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "gp_S3-SA_walkforward"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] gp_S3-SA_walkforward logged (walk-forward of trial #8b, N_TRIALS unchanged) → {REGISTRY}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S3-SA WALK-FORWARD VALIDATION (RESERVED WINDOW)")
    print("Semiannual GP/A — identical construction to in-sample trial #8b")
    print(f"Window: {WF_START.date()} -> {WF_END.date()}  ({len(WF_REBALANCE_DATES)} semiannual periods)")
    print("NO RE-OPTIMIZATION — this is a pure validation run, not a new search.")
    print("=" * 78)

    print("\n[Load] Tiingo metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("[Load] SEC pre-filter...")
    prefilter_df = load_prefilter()
    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    print(f"[Load] Raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print("[Load] AdjClose pickle...")
    preload_all_adj_prices(survivor_tickers)

    print(f"[Load] GP XBRL facts ({len(survivor_ciks):,} CIKs)...")
    preload_gp_facts(survivor_ciks)

    print(f"\n[Run] Walk-forward backtest ({len(WF_REBALANCE_DATES)} semiannual periods)...")
    results = run_walkforward(tiingo_tickers, prefilter_df)

    wf = print_results(results)
    if not wf:
        return
    print_comparison_table(wf)
    print_ic_decay_check(wf)
    print_regime_and_concentration(wf, results)
    print_verdict(wf)

    save_results(wf)
    log_registry(wf)


if __name__ == "__main__":
    main()
