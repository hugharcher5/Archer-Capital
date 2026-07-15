"""
S22 follow-up: re-run all four signals restricted to the INTERSECTION universe
(only names where DCF, P/E, P/S, and P/B all have a valid signal value that
quarter) -- the common subset all four signals could actually have ranked
from. Tests whether the original ranking (P/S > DCF > P/E ~= P/B) is an
artifact of DCF/P/E drawing from a smaller, more-established/profitable
subset than P/S/P/B, or whether it holds on a fairer, equal-universe basis.

Reuses run.py's exact per-name signal computation (imported, not
reimplemented) so the intersection portfolios are built from IDENTICAL
underlying signal values to the original full-universe run -- only the
ELIGIBLE SET each quarter differs.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
_spec = importlib.util.spec_from_file_location("s22_run_orig", str(Path(__file__).resolve().parent / "run.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

RESULTS = R.RESULTS
SIGNALS = R.SIGNALS


def run_backtest_with_intersection() -> dict:
    print("[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = R.GPI.load_tiingo_tickers()
    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = R.GPI.load_prefilter()
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()

    print("\n[Step 3a] Preloading raw close prices...")
    R.GPI.preload_all_prices()
    print("\n[Step 3b] Preloading adjClose prices...")
    R.GPI.preload_all_adj_prices()
    print(f"\n[Step 4] Preloading GP-style shares facts ({len(survivor_ciks):,} CIKs)...")
    R.GPI.preload_gp_facts(survivor_ciks)
    print("\n[Step 5] Preloading S22 valuation facts...")
    R.PF.preload_facts(survivor_ciks)

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    full_records = {sig: [] for sig in SIGNALS}
    inter_records = {sig: [] for sig in SIGNALS}
    prev_full = {sig: set() for sig in SIGNALS}
    prev_inter = {sig: set() for sig in SIGNALS}
    all_signal_rows = []

    n_periods = len(R.GPI.REBALANCE_DATES)
    for i, t in enumerate(R.GPI.REBALANCE_DATES):
        hold_end = R.GPI.REBALANCE_DATES[i + 1] if i + 1 < n_periods else R.GPI.IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        univ = R.GPI.build_universe(t, tiingo_tickers, cik_sic_map)
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        ratios = R.compute_ratio_signals(univ, t)
        betas = R.GPI.compute_betas(univ, t)
        dcf_scores = []
        for _, r in univ.iterrows():
            beta = betas.get(r["ticker"], 1.0)
            mos = R.DA.pit_dcf_margin_of_safety(r["ticker"], int(r["cik"]), t, r["price"], r["shares"], beta)
            dcf_scores.append(mos)
        ratios["score_dcf"] = dcf_scores

        try:
            returns_df = R.GPI.compute_returns(
                univ.merge(ratios.drop(columns=["mcap", "sic"]), on=["ticker", "cik"]),
                t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        snap = returns_df[["ticker", "cik", "sic", "score_dcf", "score_pe", "score_ps", "score_pb", "raw_return"]].copy()
        snap["rebalance_date"] = t
        all_signal_rows.append(snap)

        # ── Intersection subset: all four signals valid this quarter ────────
        inter_df = returns_df.dropna(subset=["score_dcf", "score_pe", "score_ps", "score_pb"]).copy()
        n_inter = len(inter_df)
        print(f"  [Universe] n={len(returns_df)}  intersection(all 4 valid)={n_inter}")

        for sig in SIGNALS:
            score_col = f"score_{sig}"

            # FULL-universe portfolio (cross-check against original run.py results)
            port_full = R.build_long_only_portfolio(returns_df, score_col)
            if port_full["longs"]:
                longs, w = port_full["longs"], port_full["w"]
                turnover = (len(set(longs) - prev_full[sig]) / max(len(longs), 1)
                            if prev_full[sig] else 1.0)
                prev_full[sig] = set(longs)
                net, gross, ba_cost, _ = R.GPI._leg_return_detail(returns_df, longs, w, is_short=False, turnover=turnover)
                valid_sig = returns_df.dropna(subset=[score_col, "raw_return"])
                ic = (stats.spearmanr(valid_sig[score_col], valid_sig["raw_return"])[0]
                      if len(valid_sig) >= 10 else np.nan)
                full_records[sig].append({"rebalance_date": t, "ret_net": net, "ret_gross": gross,
                                           "cost": ba_cost, "turnover": turnover, "mkt_ret": mkt_ret,
                                           "ic": ic, "n_long": len(longs), "n_universe": len(returns_df)})

            # INTERSECTION-universe portfolio (all four signals valid)
            port_inter = R.build_long_only_portfolio(inter_df, score_col)
            if port_inter["longs"]:
                longs, w = port_inter["longs"], port_inter["w"]
                turnover = (len(set(longs) - prev_inter[sig]) / max(len(longs), 1)
                            if prev_inter[sig] else 1.0)
                prev_inter[sig] = set(longs)
                net, gross, ba_cost, _ = R.GPI._leg_return_detail(inter_df, longs, w, is_short=False, turnover=turnover)
                valid_sig = inter_df.dropna(subset=[score_col, "raw_return"])
                ic = (stats.spearmanr(valid_sig[score_col], valid_sig["raw_return"])[0]
                      if len(valid_sig) >= 10 else np.nan)
                inter_records[sig].append({"rebalance_date": t, "ret_net": net, "ret_gross": gross,
                                            "cost": ba_cost, "turnover": turnover, "mkt_ret": mkt_ret,
                                            "ic": ic, "n_long": len(longs), "n_universe": n_inter})
                print(f"  [{sig:3s}] full ret={full_records[sig][-1]['ret_net']:+.2%} n={full_records[sig][-1]['n_long']}  "
                      f"| inter ret={net:+.2%} n={len(longs)}")

    return {"full": full_records, "inter": inter_records, "signal_snapshots": pd.concat(all_signal_rows, ignore_index=True)}


if __name__ == "__main__":
    results = run_backtest_with_intersection()
    for sig in SIGNALS:
        pd.DataFrame(results["full"][sig]).to_csv(RESULTS / f"s22_{sig}_full_crosscheck.csv", index=False)
        pd.DataFrame(results["inter"][sig]).to_csv(RESULTS / f"s22_{sig}_intersection.csv", index=False)
    results["signal_snapshots"].to_csv(RESULTS / "s22_all_signal_snapshots.csv", index=False)
    print(f"\n[Saved] -> {RESULTS}/s22_{{signal}}_intersection.csv (x4), s22_{{signal}}_full_crosscheck.csv (x4), s22_all_signal_snapshots.csv")
