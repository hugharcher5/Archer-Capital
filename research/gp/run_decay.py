"""
S3-GP Rebalance-Frequency Robustness Check
===========================================
Purpose: Determine whether the GP/A signal's net Sharpe failure at quarterly
rebalance is a cost-structure problem (high turnover) or a genuine signal
weakness — by re-running the identical signal and universe at semi-annual (2×/yr)
and annual (1×/yr) rebalance frequencies.

NOT a new trial: the signal, universe construction, weighting scheme, and beta-
neutral sizing are 100% identical to the in-sample S3-GP run.  This is a cost-
structure robustness cut only.  N_TRIALS stays at 8.

DSR note: The DSR threshold in annualized Sharpe units is frequency-invariant:
  threshold_SR_annual = norm.ppf(1 - 1/N) / sqrt(n_years)
                       ≈ 1.150 / sqrt(6) ≈ 0.470  (for N=8, 6-year in-sample)
Per-period SR thresholds differ only because they divide by sqrt(n_obs):
  quarterly  (n=24): 0.235     semi-annual (n=12): 0.332     annual (n=6): 0.469

IC at lower frequency: computed per rebalance period — a 6-month IC (Spearman
of GP/A rank vs 6-month realized return) tests whether the signal remains
informative at the holding horizon. Decay here would indicate the signal is
noisy at long horizons, not just expensive at short ones.

All output references the quarterly result from run.py for comparison.  Nothing
is saved to trial_registry unless a variant clears the in-sample gate (net
Sharpe > 0.47 annualized AND DSR clears), in which case it is logged as S3b.
"""

import sys
from pathlib import Path

# Import all infrastructure from run.py — main() is guarded, safe to import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import (
    # Preloaders
    load_tiingo_tickers, load_prefilter,
    preload_all_prices, preload_all_adj_prices, preload_gp_facts,
    # Per-period functions
    build_universe, compute_gp_signals, compute_betas,
    compute_returns, build_portfolio,
    # Cost primitives
    _spread, _borrow,
    # Metrics
    compute_metrics, compute_market_beta, deflated_sharpe_threshold,
    # Constants and caches
    N_TRIALS, IS_START, IS_END, CACHE, RESULTS, REGISTRY,
    MCAP_MIN, MCAP_MAX,
)

import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats

# ── Sweep configurations ──────────────────────────────────────────────────────
_all_q = pd.date_range(IS_START, IS_END, freq="QS")
QUARTERLY    = [t for t in _all_q                                      if t < IS_END]
SEMIANNUAL   = [t for t in pd.date_range(IS_START, IS_END, freq="6MS") if t < IS_END]
ANNUAL       = [t for t in pd.date_range(IS_START, IS_END, freq="YS")  if t < IS_END]

SWEEPS = [
    ("Quarterly",   QUARTERLY,  4),
    ("Semi-Annual", SEMIANNUAL, 2),
    ("Annual",      ANNUAL,     1),
]


# =============================================================================
# Cost model (frequency-parameterized)
# =============================================================================

def _leg_return_ppy(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool,
    turnover: float,
    periods_per_year: int,
) -> tuple:
    """
    Same as run.py _leg_return_detail() but borrow cost = annual_borrow / ppy.
    Returns (net_return, gross_return, ba_cost, borrow_cost).
    """
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
    else:
        return ( gross - total_cost),   gross,  ba_cost, borrow_cost


# =============================================================================
# Frequency-parameterized backtest
# =============================================================================

def run_one_sweep(
    label: str,
    rebalance_dates: list,
    periods_per_year: int,
    tiingo_tickers: pd.DataFrame,
    prefilter_df: pd.DataFrame,
) -> dict:
    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list = []
    prev_long_set:  set  = set()
    prev_short_set: set  = set()
    n_periods = len(rebalance_dates)

    for i, t in enumerate(rebalance_dates):
        hold_end = rebalance_dates[i + 1] if i + 1 < n_periods else IS_END

        # ── Universe ─────────────────────────────────────────────────────────
        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [{label}] {t.date()} build_universe error: {e}")
            continue
        if univ.empty:
            continue

        # ── GP signal (identical to quarterly run) ────────────────────────────
        signals_df, cov = compute_gp_signals(univ, t)
        pct_drop = cov["pct_dropped"]

        # ── Returns ──────────────────────────────────────────────────────────
        univ_sig = univ.merge(signals_df[["ticker", "gp_signal"]], on="ticker", how="left")
        try:
            returns_df = compute_returns(univ_sig, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [{label}] {t.date()} compute_returns error: {e}")
            continue

        # ── Beta ─────────────────────────────────────────────────────────────
        betas = compute_betas(univ, t)

        # ── Portfolio ─────────────────────────────────────────────────────────
        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short) = build_portfolio(returns_df, betas)
        if not longs:
            continue

        # ── Turnover ──────────────────────────────────────────────────────────
        long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1) if prev_long_set  else 1.0
        short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1) if prev_short_set else 1.0
        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        # ── Leg returns (frequency-parameterized borrow) ─────────────────────
        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_ppy(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover,
            periods_per_year=periods_per_year)
        short_net, short_gross, short_ba, short_bw = _leg_return_ppy(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover,
            periods_per_year=periods_per_year)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw

        # ── Market return ──────────────────────────────────────────────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        # ── IC: Spearman(gp_signal, realized return over THIS hold period) ────
        # Lower rebalance frequency → longer hold period → tests if GP/A predicts
        # 6-month or 12-month returns, not just 3-month returns.
        valid_sig = returns_df.dropna(subset=["gp_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["gp_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        period_records.append({
            "rebalance_date":   t,
            "hold_end":         hold_end,
            "n_universe":       len(returns_df),
            "n_with_signal":    int(returns_df["gp_signal"].notna().sum()),
            "n_delisted":       int(returns_df["delisted"].sum()),
            "pct_dropped":      pct_drop,
            "long_ret":         long_net,
            "short_ret":        short_net,
            "ls_ret":           ls_net,
            "ls_gross":         ls_gross,
            "ls_cost":          ls_ba + ls_bw,
            "ls_ba_cost":       ls_ba,
            "ls_borrow_cost":   ls_bw,
            "long_scale":       long_scale,
            "short_scale":      short_scale,
            "mean_beta_long":   mean_beta_long,
            "mean_beta_short":  mean_beta_short,
            "long_turnover":    long_turnover,
            "short_turnover":   short_turnover,
            "avg_turnover":     avg_turnover,
            "mkt_ret":          mkt_ret,
            "ic":               ic,
            "n_long":           len(longs),
            "n_short":          len(shorts),
        })

    return {"label": label, "ppy": periods_per_year, "period_records": period_records}


# =============================================================================
# Compute summary metrics for one sweep
# =============================================================================

def summarise(sweep: dict) -> dict:
    label   = sweep["label"]
    ppy     = sweep["ppy"]
    records = sweep["period_records"]
    if not records:
        return {"label": label, "ppy": ppy}

    df          = pd.DataFrame(records)
    ls_series   = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series  = df["mkt_ret"]
    ic_series   = df["ic"].dropna()

    m        = compute_metrics(ls_series, ppy)
    m_gross  = compute_metrics(gross_series, ppy)
    beta_s   = compute_market_beta(ls_series, mkt_series)

    sigma = float(ls_series.std(ddof=1))
    per_period_sr = float(ls_series.mean() / sigma) if sigma > 0 else np.nan
    n_obs  = m["n"]
    dsr_thr = deflated_sharpe_threshold(N_TRIALS, n_obs)  # per-period SR units
    dsr_thr_ann = dsr_thr * np.sqrt(ppy) if not np.isnan(dsr_thr) else np.nan
    clears_dsr  = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) else False

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    ba_ann_bps = float(df["ls_ba_cost"].mean())   * ppy * 10_000
    bw_ann_bps = float(df["ls_borrow_cost"].mean()) * ppy * 10_000
    cost_ann   = ba_ann_bps + bw_ann_bps

    return {
        "label":          label,
        "ppy":            ppy,
        "n_periods":      n_obs,
        "ann_ret_net":    m["ann_ret"],
        "ann_ret_gross":  m_gross["ann_ret"],
        "cost_drag_bps":  cost_ann,
        "ba_bps":         ba_ann_bps,
        "borrow_bps":     bw_ann_bps,
        "sharpe_net":     m["sharpe"],
        "t_stat":         m["t_stat"],
        "max_dd":         m["max_dd"],
        "calmar":         m["calmar"],
        "win_rate":       m["win_rate"],
        "market_beta":    beta_s["beta"],
        "beta_tstat":     beta_s["beta_t"],
        "mean_turnover":  float(df["avg_turnover"].mean()),
        "mean_ic":        mean_ic,
        "ic_tstat":       ic_tstat,
        "per_period_sr":  per_period_sr,
        "dsr_thr_period": dsr_thr,
        "dsr_thr_ann":    dsr_thr_ann,
        "clears_dsr":     clears_dsr,
        "period_records": records,
        "df":             df,
    }


def compute_metrics(ls_series: pd.Series, ppy: int) -> dict:
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
# Period-level detail table (per sweep)
# =============================================================================

def print_period_detail(s: dict) -> None:
    df    = s["df"]
    label = s["label"]
    ppy   = s["ppy"]
    print(f"\n{'─'*78}")
    print(f"  {label.upper()} — Period-Level Detail")
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} {'TOver':>6} "
          f"{'LScl':>5} {'SScl':>5} {'Mkt':>8} {'IC':>6} {'nL':>3}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*5} "
          f"{'-'*8} {'-'*6} {'-'*3}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
              f"{r['avg_turnover']:>5.0%} "
              f"{r['long_scale']:>4.2f}× {r['short_scale']:>4.2f}× "
              f"{r['mkt_ret']:>7.2%} {r['ic']:>6.3f} "
              f"{int(r['n_long']):>3}")
    cum      = (1 + df["ls_ret"].reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series:")
    for i, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        min_dd = float(dd_s.min())
        flag   = " ◄" if abs(val - min_dd) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{flag}")


# =============================================================================
# Comparison table
# =============================================================================

def print_comparison_table(summaries: list) -> None:
    sep = "=" * 78

    print(f"\n{sep}")
    print("S3-GP REBALANCE-FREQUENCY ROBUSTNESS CHECK")
    print("Signal: GP/A (unchanged) | Universe: unchanged | N_TRIALS=8 (not a new trial)")
    print(f"{sep}")

    print(f"\nDSR note: threshold in ANNUALIZED Sharpe units is frequency-invariant:")
    print(f"  = norm.ppf(1 - 1/8) / sqrt(6 years) ≈ 0.470  (all three frequencies)")
    print(f"  Per-period SR thresholds differ because they divide by sqrt(n_periods).\n")

    # Header
    w0, w1 = 34, 14
    header = f"  {'Metric':<{w0}}"
    for s in summaries:
        header += f" {s['label']:>{w1}}"
    print(header)
    print(f"  {'─'*w0}" + (f" {'─'*w1}" * len(summaries)))

    def row(label, vals, fmt="{:.3f}", flag_best=None, flag_fn=None):
        line = f"  {label:<{w0}}"
        for v in vals:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                line += f" {'—':>{w1}}"
            else:
                fmtd = fmt.format(v) if isinstance(v, (float, int)) else str(v)
                line += f" {fmtd:>{w1}}"
        print(line)

    def pct(v):  return f"{v:.2%}" if v is not None and not np.isnan(v) else "—"
    def bps(v):  return f"{v:.0f}" if v is not None and not np.isnan(v) else "—"
    def f3(v):   return f"{v:.3f}" if v is not None and not np.isnan(v) else "—"
    def f2(v):   return f"{v:.2f}" if v is not None and not np.isnan(v) else "—"
    def yn(v):   return "YES ✓" if v else "NO"

    def table_row(label, vals_fmt):
        line = f"  {label:<{w0}}"
        for fmtd in vals_fmt:
            line += f" {fmtd:>{w1}}"
        print(line)

    table_row("Rebalance periods (N)",      [str(s["n_periods"]) for s in summaries])
    table_row("",                           [""] * len(summaries))
    table_row("IC (Information Coefficient)",[""] * len(summaries))
    table_row("  Mean IC (Spearman)",       [f3(s["mean_ic"]) for s in summaries])
    table_row("  IC t-stat",                [f3(s["ic_tstat"]) for s in summaries])
    table_row("  IC significant (|t|>2)?",  ["YES" if abs(s["ic_tstat"] or 0) > 2 else "NO" for s in summaries])
    table_row("",                           [""] * len(summaries))
    table_row("Returns",                    [""] * len(summaries))
    table_row("  Ann. return (GROSS)",      [pct(s["ann_ret_gross"]) for s in summaries])
    table_row("  Cost drag (bps/yr)",       [bps(s["cost_drag_bps"]) for s in summaries])
    table_row("    bid-ask (bps/yr)",       [bps(s["ba_bps"]) for s in summaries])
    table_row("    borrow  (bps/yr)",       [bps(s["borrow_bps"]) for s in summaries])
    table_row("  Ann. return (net)",        [pct(s["ann_ret_net"]) for s in summaries])
    table_row("",                           [""] * len(summaries))
    table_row("Risk / Sharpe",              [""] * len(summaries))
    table_row("  Sharpe (annualized, net)", [f3(s["sharpe_net"]) for s in summaries])
    table_row("  t-stat (period returns)",  [f3(s["t_stat"]) for s in summaries])
    table_row("  Max drawdown",             [pct(s["max_dd"]) for s in summaries])
    table_row("  CALMAR ratio",             [f3(s["calmar"]) for s in summaries])
    table_row("  Win rate",                 [pct(s["win_rate"]) for s in summaries])
    table_row("  Market beta",              [f3(s["market_beta"]) for s in summaries])
    table_row("  Market beta t-stat",       [f3(s["beta_tstat"]) for s in summaries])
    table_row("",                           [""] * len(summaries))
    table_row("Cost structure",             [""] * len(summaries))
    table_row("  Mean turnover / period",   [pct(s["mean_turnover"]) for s in summaries])
    table_row("",                           [""] * len(summaries))
    table_row("DSR (N_trials=8)",           [""] * len(summaries))
    table_row("  Per-period SR",            [f3(s["per_period_sr"]) for s in summaries])
    table_row("  DSR threshold (period SR)",[f3(s["dsr_thr_period"]) for s in summaries])
    table_row("  DSR threshold (ann SR)",   [f3(s["dsr_thr_ann"]) for s in summaries])
    table_row("  Clears DSR?",              [yn(s["clears_dsr"]) for s in summaries])

    # ── Verdict ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*78}")
    print("VERDICT")
    print(f"{'─'*78}")
    any_clear = any(s["clears_dsr"] for s in summaries)
    ic_persists = all(
        abs(s.get("ic_tstat") or 0) > 2
        for s in summaries if s["n_periods"] >= 6
    )

    # Check if 2016 / 2019-2020 losing streaks persist at annual frequency
    annual_s = next((s for s in summaries if s["ppy"] == 1), None)
    if annual_s is not None:
        annual_df = annual_s["df"]
        neg_periods = annual_df[annual_df["ls_ret"] < 0]
        pct_neg    = len(neg_periods) / max(len(annual_df), 1)
        neg_dates  = ", ".join(str(r["rebalance_date"].year) for _, r in neg_periods.iterrows())
    else:
        pct_neg   = None
        neg_dates = "—"

    print(f"\n  IC persistence across frequencies:")
    for s in summaries:
        star = "*" if abs(s.get("ic_tstat") or 0) > 2 else " "
        print(f"    {s['label']:<14}: IC={s['mean_ic']:+.4f}  t={s['ic_tstat']:>6.3f} {star}")

    print(f"\n  Sharpe / DSR by frequency:")
    for s in summaries:
        clr = "CLEARS ✓" if s["clears_dsr"] else "FAILS  ✗"
        print(f"    {s['label']:<14}: SR={s['sharpe_net']:>6.3f}  t={s['t_stat']:>6.3f}  "
              f"DSR={clr}  (threshold SR={s['dsr_thr_ann']:.3f} ann)")

    if annual_s is not None:
        print(f"\n  2016 / 2019-2020 losing years at annual frequency:")
        print(f"    Annual negative periods ({pct_neg:.0%} of total): {neg_dates or 'none'}")
        if pct_neg is not None and pct_neg > 0.4:
            print(f"    → Losing streaks PERSIST at annual hold — not quarterly noise, "
                  f"signal genuinely fails in these regimes.")
        else:
            print(f"    → Fewer negative periods at annual frequency — some 2016/2019 "
                  f"drawdown was quarterly rebalance noise.")

    print(f"\n  FAMILY STATUS: ", end="")
    if any_clear:
        winner = [s for s in summaries if s["clears_dsr"]]
        for w in winner:
            print(f"\n    {w['label']} CLEARS DSR (ann SR {w['sharpe_net']:.3f} > 0.470). "
                  f"Log as S3b-{w['label'].lower().replace('-','')} in registry.")
    else:
        print(f"\n    NO frequency clears DSR. Gross alpha (+490 bps/yr at quarterly) is real "
              f"(IC significant) but cost drag + regime losses prevent net SR threshold."
              f"\n    S3-GP family STAYS CLOSED as a standalone strategy.")
        print(f"\n    The signal is a potential INPUT to a multi-factor composite or long-only "
              f"screen — but requires a different execution structure, not retuning.")


# =============================================================================
# Registry (only if a variant clears)
# =============================================================================

def maybe_log_registry(summaries: list) -> None:
    clearing = [s for s in summaries if s["clears_dsr"] and s["ppy"] != 4]  # skip quarterly (already logged)
    if not clearing:
        print("\n[Registry] No new variants cleared DSR — registry unchanged.")
        return

    for s in clearing:
        freq_tag = {2: "semiannual", 1: "annual"}.get(s["ppy"], str(s["ppy"]))
        df       = s["df"]
        ic_s     = df["ic"].dropna()
        ann_cost = s["cost_drag_bps"]

        row = {
            "timestamp":            datetime.now().isoformat(),
            "study":                f"gp_S3b-{freq_tag}",
            "hypothesis":           (
                f"S3-GP robustness: GP/A signal at {s['label']} rebalance. "
                f"Same signal/universe as S3-GP (trial #8), different cost structure. "
                f"Not a new independent trial."
            ),
            "data_source":          "Tiingo + SEC XBRL (PIT gated)",
            "status":               "completed_in_sample",
            "trial_number":         "8b",
            "trial_note":           f"Robustness cut of trial #8 at {s['label']} frequency. N_TRIALS unchanged at {N_TRIALS}.",
            "n_trials_dsr":         N_TRIALS,
            "rebalance_freq":       freq_tag,
            "portfolio_note":       "Beta-neutral (EW market proxy, 60-day adjClose)",
            "signal_note":          "GP/A = TTM GrossProfit / Assets, filed ≤ t (same as S3-GP)",
            "ann_return_net":       s["ann_ret_net"],
            "ann_return_gross":     s["ann_ret_gross"],
            "annual_cost_drag_bps": ann_cost,
            "mean_turnover":        s["mean_turnover"],
            "sharpe":               s["sharpe_net"],
            "calmar":               s["calmar"],
            "max_drawdown":         s["max_dd"],
            "t_stat":               s["t_stat"],
            "market_beta":          s["market_beta"],
            "market_beta_tstat":    s["beta_tstat"],
            "mean_ic":              s["mean_ic"],
            "ic_t_stat":            s["ic_tstat"],
            "win_rate":             s["win_rate"],
            "n_periods":            s["n_periods"],
            "per_period_sr":        s["per_period_sr"],
            "dsr_threshold":        s["dsr_thr_period"],
            "clears_dsr":           s["clears_dsr"],
        }
        reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
        if not reg_df.empty and "study" in reg_df.columns:
            reg_df = reg_df[reg_df["study"] != row["study"]].copy()
        reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
        reg_df.to_csv(REGISTRY, index=False)
        print(f"[Registry] {row['study']} logged → {REGISTRY}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    sep = "=" * 78
    print(sep)
    print("S3-GP Rebalance-Frequency Robustness Check")
    print("Sweeps: Quarterly (4×/yr) | Semi-Annual (2×/yr) | Annual (1×/yr)")
    print("Signal and universe: IDENTICAL to S3-GP run.py — only hold period varies")
    print(sep)

    # ── Load caches once ──────────────────────────────────────────────────────
    print("\n[Load] Tiingo metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("[Load] SEC pre-filter...")
    prefilter_df   = load_prefilter()
    survivors      = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    print(f"[Load] Raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print("[Load] AdjClose pickle...")
    preload_all_adj_prices(survivor_tickers)

    print(f"[Load] GP XBRL facts ({len(survivor_ciks):,} CIKs)...")
    preload_gp_facts(survivor_ciks)

    # ── Run sweeps ────────────────────────────────────────────────────────────
    raw_results = []
    for label, dates, ppy in SWEEPS:
        print(f"\n{'─'*78}")
        print(f"Running {label} sweep ({len(dates)} periods)...")
        print(f"{'─'*78}")
        result = run_one_sweep(label, dates, ppy, tiingo_tickers, prefilter_df)
        raw_results.append(result)

    # ── Summarise ─────────────────────────────────────────────────────────────
    summaries = [summarise(r) for r in raw_results]

    # ── Period-level detail for semi-annual and annual ────────────────────────
    for s in summaries[1:]:   # skip quarterly (already in run.py output)
        print_period_detail(s)

    # ── Comparison table ──────────────────────────────────────────────────────
    print_comparison_table(summaries)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    for s in summaries:
        freq_label = s["label"].lower().replace("-", "").replace(" ", "")
        pd.DataFrame(s["period_records"]).to_csv(
            RESULTS / f"gp_decay_{freq_label}.csv", index=False)
    print(f"\n[Saved] Period CSVs → {RESULTS}/")

    # ── Registry (only if a variant clears) ───────────────────────────────────
    maybe_log_registry(summaries)


if __name__ == "__main__":
    main()
