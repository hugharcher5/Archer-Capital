"""
Industry-Neutral GP/A -- reporting, diagnostics, registry logging.
"""
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

import run as R
from backtest import CONSTRUCTIONS, RANK_COL

# S3-Q's ORIGINAL, already-logged registry numbers (trial #8), pulled fresh
# at report time for the direct-comparison table (not hand-copied).
S3Q_STUDY_NAME = "gp_S3-GP"


def load_s3q_registry_row() -> dict:
    if not R.REGISTRY.exists():
        return {}
    df = pd.read_csv(R.REGISTRY)
    match = df[df["study"] == S3Q_STUDY_NAME]
    if match.empty:
        return {}
    return match.iloc[-1].to_dict()


def compound(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    return float((1 + s).prod() - 1)


def year_by_year_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["rebalance_date"].dt.year
    rows = []
    for yr, g in df.groupby("year"):
        row = {"year": yr, "n_quarters": len(g), "market_ret": compound(g["mkt_ret"])}
        for c in CONSTRUCTIONS:
            row[f"{c}_ret"] = compound(g[f"{c}_ls_ret"])
            beta_stats = R.compute_market_beta(g[f"{c}_ls_ret"], g["mkt_ret"])
            row[f"{c}_beta"] = beta_stats["beta"]
            row[f"{c}_beta_t"] = beta_stats["beta_t"]
        rows.append(row)
    return pd.DataFrame(rows)


def window_2020_check(df: pd.DataFrame) -> dict:
    in_2020 = df[(df["rebalance_date"] >= R.WINDOW_2020_START) & (df["rebalance_date"] < R.WINDOW_2020_END)]
    out_2020 = df[~df.index.isin(in_2020.index)]
    result = {}
    for c in CONSTRUCTIONS:
        col = f"{c}_ls_ret"
        abs_sum = float(df[col].abs().sum())
        share_2020 = float(in_2020[col].abs().sum() / abs_sum) if abs_sum > 0 and not in_2020.empty else np.nan
        result[c] = {
            "dates": [str(d.date()) for d in in_2020["rebalance_date"]],
            "sum_2020": float(in_2020[col].sum()) if not in_2020.empty else np.nan,
            "share_2020": share_2020,
            "mean_2020": float(in_2020[col].mean()) if not in_2020.empty else np.nan,
            "mean_ex_2020": float(out_2020[col].mean()) if not out_2020.empty else np.nan,
        }
    return result


def build_summary(results: dict, n_trials: int) -> dict:
    df = pd.DataFrame(results["period_records"])
    summary = {"n_trials": n_trials, "df": df}

    for c in CONSTRUCTIONS:
        col = f"{c}_ls_ret"
        gross_col = f"{c}_ls_gross"
        ls_series = df[col]
        m = R.compute_metrics(ls_series)
        m_gross = R.compute_metrics(df[gross_col])
        beta_stats = R.compute_market_beta(ls_series, df["mkt_ret"])
        ic_series = df[f"{c}_ic"].dropna()
        mean_ic = float(ic_series.mean()) if len(ic_series) else np.nan
        ic_t = (ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
                if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan)
        per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
        dsr_thr = R.deflated_sharpe_threshold(n_trials, m["n"])
        clears_dsr = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False
        annual_cost_drag = (float(df[f"{c}_ls_ba_cost"].mean()) + float(df[f"{c}_ls_borrow_cost"].mean())) * 4 * 10_000
        mean_turnover = float(df[f"{c}_avg_turnover"].mean())
        mean_long_hhi = float(df[f"{c}_long_hhi"].mean())
        mean_short_hhi = float(df[f"{c}_short_hhi"].mean())
        mean_long_top_share = float(df[f"{c}_long_top_share"].mean())
        mean_short_top_share = float(df[f"{c}_short_top_share"].mean())
        mean_long_n_sectors = float(df[f"{c}_long_n_sectors"].mean())
        mean_short_n_sectors = float(df[f"{c}_short_n_sectors"].mean())

        summary[c] = {
            "m": m, "m_gross": m_gross, "beta_stats": beta_stats,
            "mean_ic": mean_ic, "ic_t": ic_t,
            "per_period_sr": per_period_sr, "dsr_thr": dsr_thr, "clears_dsr": clears_dsr,
            "annual_cost_drag": annual_cost_drag, "mean_turnover": mean_turnover,
            "mean_long_hhi": mean_long_hhi, "mean_short_hhi": mean_short_hhi,
            "mean_long_top_share": mean_long_top_share, "mean_short_top_share": mean_short_top_share,
            "mean_long_n_sectors": mean_long_n_sectors, "mean_short_n_sectors": mean_short_n_sectors,
            "sign_violations": results["sign_violations"][c],
        }
    return summary


def print_report(results: dict, summary: dict) -> None:
    df = summary["df"]
    sep = "=" * 82
    print(f"\n{sep}")
    print("INDUSTRY-NEUTRAL GP/A -- RESULTS")
    print(sep)
    print(f"N_TRIALS_DSR = {summary['n_trials']}")
    print("UNIVERSE NOTE: $100M-2B liquidity-screened universe (same as S3-Q) -- "
          "NOT Russell 2000 PIT membership. See run.py docstring for why.")

    print(f"\n{sep}")
    print("SECTOR COMPOSITION -- DOES INDUSTRY-NEUTRAL ACTUALLY IMPROVE BALANCE?")
    print(sep)
    print(f"  {'Construction':<16} {'Leg':<6} {'MeanHHI':>9} {'MeanTopShare':>13} {'MeanNSectors':>13}")
    for c in CONSTRUCTIONS:
        label = "S3-Q original" if c == "orig" else "Industry-neutral"
        s = summary[c]
        print(f"  {label:<16} {'long':<6} {s['mean_long_hhi']:>9.3f} {s['mean_long_top_share']:>12.1%} "
              f"{s['mean_long_n_sectors']:>13.1f}")
        print(f"  {'':<16} {'short':<6} {s['mean_short_hhi']:>9.3f} {s['mean_short_top_share']:>12.1%} "
              f"{s['mean_short_n_sectors']:>13.1f}")
    print("  Lower HHI / top-share, higher n_sectors = better diversified across sectors.")

    print(f"\n{sep}")
    print("DIRECT COMPARISON: S3-Q ORIGINAL (recomputed) vs INDUSTRY-NEUTRAL")
    print(sep)
    hdr = f"  {'Metric':<32} {'Orig (recomputed)':>20} {'Industry-Neutral':>20}"
    print(hdr)
    rows = [
        ("Mean IC (Spearman)", "mean_ic", ".4f"),
        ("IC t-stat", "ic_t", ".3f"),
        ("Ann. return (net)", "m.ann_ret", ".2%"),
        ("Ann. return (gross)", "m_gross.ann_ret", ".2%"),
        ("Cost drag (bps/yr)", "annual_cost_drag", ".0f"),
        ("Mean turnover/quarter", "mean_turnover", ".0%"),
        ("Sharpe (net, ann.)", "m.sharpe", ".3f"),
        ("CALMAR", "m.calmar", ".3f"),
        ("Max drawdown", "m.max_dd", ".2%"),
        ("Market beta (full-sample)", "beta_stats.beta", ".3f"),
        ("Beta t-stat", "beta_stats.beta_t", ".3f"),
        ("Win rate", "m.win_rate", ".1%"),
        ("N periods", "m.n", "d"),
        ("Per-period SR", "per_period_sr", ".4f"),
        ("DSR threshold", "dsr_thr", ".4f"),
    ]
    for label, path, fmt in rows:
        vals = []
        for c in CONSTRUCTIONS:
            obj = summary[c]
            v = obj
            for part in path.split("."):
                v = v[part] if isinstance(v, dict) else v
            vals.append(v)
        try:
            v0 = format(vals[0], fmt)
            v1 = format(vals[1], fmt)
        except (ValueError, TypeError):
            v0, v1 = str(vals[0]), str(vals[1])
        print(f"  {label:<32} {v0:>20} {v1:>20}")
    print(f"  {'Clears DSR':<32} {str(summary['orig']['clears_dsr']):>20} {str(summary['in']['clears_dsr']):>20}")

    s3q_row = load_s3q_registry_row()
    if s3q_row:
        print(f"\n  Cross-check vs S3-Q's ALREADY-LOGGED registry numbers (trial #8, "
              f"study={S3Q_STUDY_NAME}):")
        print(f"    Logged Sharpe={s3q_row.get('sharpe'):.3f}  IC={s3q_row.get('mean_ic'):.4f}  "
              f"beta={s3q_row.get('market_beta'):.3f}  (vs this trial's recomputed 'orig' column above)")

    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT")
    print(sep)
    n_periods = summary[CONSTRUCTIONS[0]]["m"]["n"]
    for c in CONSTRUCTIONS:
        label = "S3-Q original" if c == "orig" else "Industry-neutral"
        print(f"  {label}: {summary[c]['sign_violations']} violation(s) "
              f"(long-leg mean sort-key <= short-leg)")

    print(f"\n{sep}")
    print("CONCENTRATION CHECK")
    print(sep)
    for c in CONSTRUCTIONS:
        label = "S3-Q original" if c == "orig" else "Industry-neutral"
        col = f"{c}_ls_ret"
        abs_sum = float(df[col].abs().sum())
        if abs_sum > 0:
            share = df[col].abs() / abs_sum
            top_idx = share.idxmax()
            print(f"  [{label}] Largest single-quarter share of |return| mass: "
                  f"{df.loc[top_idx, 'rebalance_date'].date()} = {float(share.loc[top_idx]):.1%}")
        sc = results["sector_contrib"][c]
        total_abs = sum(abs(v) for v in sc.values())
        if total_abs > 0:
            top_sic, top_val = max(sc.items(), key=lambda kv: abs(kv[1]))
            print(f"  [{label}] Largest single SIC major-group contribution: "
                  f"SIC {top_sic} = {abs(top_val)/total_abs:.1%} of total |contribution|")

    print(f"\n{sep}")
    print("2020 SUB-PERIOD CHECK (S1/S2/S7/E2 precedent)")
    print(sep)
    w2020 = window_2020_check(df)
    for c in CONSTRUCTIONS:
        label = "S3-Q original" if c == "orig" else "Industry-neutral"
        w = w2020[c]
        print(f"  [{label}] 2020 dates: {w['dates']}")
        print(f"    Sum 2020 L/S: {w['sum_2020']:.2%}  Share of |return| mass: {w['share_2020']:.1%}  "
              f"Mean 2020: {w['mean_2020']:.2%}  Mean ex-2020: {w['mean_ex_2020']:.2%}")
        if not np.isnan(w["share_2020"]) and w["share_2020"] > 0.40:
            print(f"    *** WARNING: 2020 drives >40% of |return| mass. ***")

    print(f"\n{sep}")
    print("YEAR-BY-YEAR: STRATEGY vs MARKET (EW universe return)")
    print(sep)
    yby = year_by_year_table(df)
    print(f"  {'Year':<6} {'nQ':>3} {'Market':>8} {'Orig ret':>9} {'Orig beta':>10} {'Orig t':>7} "
          f"{'InNeu ret':>10} {'InNeu beta':>11} {'InNeu t':>8}")
    for _, r in yby.iterrows():
        print(f"  {int(r['year']):<6} {int(r['n_quarters']):>3} {r['market_ret']:>7.2%} "
              f"{r['orig_ret']:>8.2%} {r['orig_beta']:>10.3f} {r['orig_beta_t']:>7.2f} "
              f"{r['in_ret']:>9.2%} {r['in_beta']:>11.3f} {r['in_beta_t']:>8.2f}")

    print(f"\nRuntime: {results['elapsed_s']:.1f}s ({results['elapsed_s']/60:.1f} min)")


def save_csvs(results: dict, yby: pd.DataFrame) -> None:
    pd.DataFrame(results["period_records"]).to_csv(
        R.RESULTS / "gpindneutral_period_returns.csv", index=False)
    pd.DataFrame(results["coverage_records"]).to_csv(
        R.RESULTS / "gpindneutral_coverage.csv", index=False)
    yby.to_csv(R.RESULTS / "gpindneutral_year_by_year.csv", index=False)
    print(f"\n[Saved] CSVs -> {R.RESULTS}/")


def _current_max_trial_number() -> int:
    """Re-read the registry fresh -- lesson from insiderclusters_S13's
    trial-number collision (see feedback_dsr_trial_numbering memory)."""
    if not R.REGISTRY.exists():
        return 0
    df = pd.read_csv(R.REGISTRY)
    nums = []
    for v in df["trial_number"].dropna():
        try:
            nums.append(int(str(v).split("b")[0].split("-")[0]))
        except ValueError:
            continue
    return max(nums) if nums else 0


def log_registry(results: dict, summary: dict) -> int:
    """Returns the true trial_number actually used (re-verified at write time)."""
    current_max = _current_max_trial_number()
    true_n = current_max + 1
    if true_n != summary["n_trials"]:
        print(f"\n*** N_TRIALS CORRECTION: provisional was {summary['n_trials']}, "
              f"registry now shows max trial_number={current_max} -> using true N={true_n}. ***")
        # Recompute DSR at the corrected N for both constructions.
        for c in CONSTRUCTIONS:
            s = summary[c]
            s["dsr_thr"] = R.deflated_sharpe_threshold(true_n, s["m"]["n"])
            s["clears_dsr"] = (bool(s["per_period_sr"] > s["dsr_thr"])
                               if not np.isnan(s["per_period_sr"]) and not np.isnan(s["dsr_thr"]) else False)
        summary["n_trials"] = true_n

    s3q_row = load_s3q_registry_row()
    orig, ind = summary["orig"], summary["in"]

    verdict_bits = ["CLEARS DSR (industry-neutral)" if ind["clears_dsr"] else "FAILS DSR (industry-neutral)"]
    beta_improved = (abs(ind["beta_stats"]["beta"]) < abs(orig["beta_stats"]["beta"])
                     if not np.isnan(ind["beta_stats"]["beta"]) and not np.isnan(orig["beta_stats"]["beta"]) else None)
    sharpe_improved = (ind["m"]["sharpe"] > orig["m"]["sharpe"]
                       if not np.isnan(ind["m"]["sharpe"]) and not np.isnan(orig["m"]["sharpe"]) else None)
    if beta_improved is not None:
        verdict_bits.append(f"|beta| {'LOWER' if beta_improved else 'NOT LOWER'} than original "
                             f"({ind['beta_stats']['beta']:.3f} vs {orig['beta_stats']['beta']:.3f})")
    if sharpe_improved is not None:
        verdict_bits.append(f"Sharpe {'IMPROVED' if sharpe_improved else 'NOT IMPROVED'} vs original "
                             f"({ind['m']['sharpe']:.3f} vs {orig['m']['sharpe']:.3f})")
    hhi_improved = ind["mean_long_hhi"] < orig["mean_long_hhi"] and ind["mean_short_hhi"] < orig["mean_short_hhi"]
    verdict_bits.append(f"Sector HHI {'IMPROVED' if hhi_improved else 'NOT IMPROVED'} "
                        f"(long {ind['mean_long_hhi']:.3f} vs {orig['mean_long_hhi']:.3f}, "
                        f"short {ind['mean_short_hhi']:.3f} vs {orig['mean_short_hhi']:.3f})")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp": datetime.now().isoformat(),
        "study": "gpindneutral_S16",
        "hypothesis": (
            "Industry-neutral reformulation of S3-Q (GP/A, trial #8): rank GP/A "
            "WITHIN each SIC major-group first (pooling groups <5 members into "
            "an OTHER bucket), then build long/short from those industry-"
            "neutral percentiles, vs S3-Q's original full-universe rank. "
            "Hypothesis: removing accidental sector bets lowers realized beta "
            "and improves Sharpe/CALMAR. UNIVERSE NOTE: uses the same $100M-2B "
            "universe as S3-Q, NOT Russell 2000 PIT membership -- the existing "
            "research/russell_reconstitution/ pipeline only has annual "
            "addition/deletion event lists, never a full constituent snapshot, "
            "so genuine PIT Russell 2000 membership could not be reconstructed "
            "without new data sourcing; user chose to keep S3-Q's universe "
            "when asked directly."
        ),
        "data_source": (
            "100% reused from S3-Q/mgmt_pit's existing cache (Tiingo prices, "
            "adjClose, SEC XBRL GP/A facts, SEC prefilter) -- no new fetches. "
            "New: within-SIC-major-group (sic // 100) rank-percentile transform."
        ),
        "status": "completed_in_sample",
        "trial_number": true_n,
        "trial_note": (
            f"Independent trial #{true_n}. Genuine reformulation of S3-Q "
            "(trial #8), not a re-tune -- new industry-neutral ranking "
            "mechanism, consumes its own DSR slot. N_TRIALS re-verified "
            "against the registry immediately before logging (see "
            "feedback_dsr_trial_numbering memory)."
        ),
        "n_trials_dsr": true_n,
        "rebalance_freq": "quarterly",
        "rebalance_note": "Matches S3-Q's cadence exactly for direct comparability.",
        "portfolio_note": (
            f"Beta-neutral (EW market proxy, 60-day adjClose). Leg = "
            f"max(ceil(n*0.20),20), identical to S3-Q. Industry-neutral leg "
            f"HHI: long={ind['mean_long_hhi']:.3f} short={ind['mean_short_hhi']:.3f} "
            f"vs original long={orig['mean_long_hhi']:.3f} short={orig['mean_short_hhi']:.3f}."
        ),
        "ann_return_net": ind["m"]["ann_ret"],
        "ann_return_gross": ind["m_gross"]["ann_ret"],
        "annual_cost_drag_bps": ind["annual_cost_drag"],
        "mean_turnover": ind["mean_turnover"],
        "sharpe": ind["m"]["sharpe"],
        "calmar": ind["m"]["calmar"],
        "max_drawdown": ind["m"]["max_dd"],
        "t_stat": ind["m"]["t_stat"],
        "market_beta": ind["beta_stats"]["beta"],
        "market_beta_tstat": ind["beta_stats"]["beta_t"],
        "mean_ic": ind["mean_ic"],
        "ic_t_stat": ind["ic_t"],
        "win_rate": ind["m"]["win_rate"],
        "n_periods": ind["m"]["n"],
        "per_period_sr": ind["per_period_sr"],
        "dsr_threshold": ind["dsr_thr"],
        "clears_dsr": ind["clears_dsr"],
        "signal_note": (
            f"Direct comparison (recomputed 'orig' vs this trial's 'in', same "
            f"universe/window/cost-model): IC {orig['mean_ic']:.4f}->{ind['mean_ic']:.4f}, "
            f"Sharpe {orig['m']['sharpe']:.3f}->{ind['m']['sharpe']:.3f}, "
            f"beta {orig['beta_stats']['beta']:.3f}->{ind['beta_stats']['beta']:.3f}. "
            f"S3-Q's own already-logged registry numbers (trial #8): "
            f"Sharpe={s3q_row.get('sharpe', 'n/a')}, IC={s3q_row.get('mean_ic', 'n/a')}, "
            f"beta={s3q_row.get('market_beta', 'n/a')}."
        ),
        "notes": verdict,
    }

    reg_df = pd.read_csv(R.REGISTRY) if R.REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "gpindneutral_S16"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(R.REGISTRY, index=False)
    print(f"\n[Registry] gpindneutral_S16 logged (trial #{true_n}) -> {R.REGISTRY}")
    print(f"[Verdict] {verdict}")
    return true_n


def main():
    print("=" * 82)
    print("INDUSTRY-NEUTRAL GP/A (provisional trial number -- re-verified at write time)")
    print("=" * 82)

    tiingo = R.load_tiingo_tickers()
    prefilter = R.load_prefilter()
    survivor_tickers = prefilter[prefilter["passed"]]["ticker"].tolist()

    R.preload_all_prices()
    R.preload_all_adj_prices()
    R.preload_gp_facts(prefilter[prefilter["passed"]]["cik"].dropna().astype(int).tolist())

    from backtest import run_backtest
    results = run_backtest(tiingo, prefilter)

    summary = build_summary(results, R.PROVISIONAL_N)
    print_report(results, summary)
    yby = year_by_year_table(summary["df"])
    save_csvs(results, yby)
    true_n = log_registry(results, summary)
    return true_n


if __name__ == "__main__":
    main()
