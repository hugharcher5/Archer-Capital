"""
Industry-Neutral GP/A -- backtest engine.
See run.py module docstring for full hypothesis/design/pre-registration and
the universe-scope deviation note (confirmed with user).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

import run as R

CONSTRUCTIONS = ["orig", "in"]   # orig = S3-Q original, in = industry-neutral
RANK_COL = {"orig": "gp_signal", "in": "in_rank"}


def run_backtest(tiingo_tickers: pd.DataFrame, prefilter_df: pd.DataFrame) -> dict:
    t_total = time.time()

    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records = []
    coverage_records = []
    sector_contrib = {c: {} for c in CONSTRUCTIONS}     # for concentration check
    sign_violations = {c: 0 for c in CONSTRUCTIONS}

    prev_long = {c: set() for c in CONSTRUCTIONS}
    prev_short = {c: set() for c in CONSTRUCTIONS}

    n_periods = len(R.REBALANCE_DATES)
    for i, t in enumerate(R.REBALANCE_DATES):
        hold_end = R.REBALANCE_DATES[i + 1] if i + 1 < n_periods else R.IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        univ = R.build_universe(t, tiingo_tickers, cik_sic_map)
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        signals_df, cov = R.compute_gp_signals(univ, t)
        signals_df = R.add_industry_neutral_rank(signals_df)
        pct_drop = cov["pct_dropped"]
        print(f"  GP: {cov['n_signal']}/{cov['n_universe']} with signal ({pct_drop:.0f}% dropped)")
        coverage_records.append({"rebalance_date": t, **cov})

        univ_sig = univ.merge(
            signals_df[["ticker", "gp_signal", "sector_group", "in_rank"]],
            on="ticker", how="left")
        returns_df = R.compute_returns(univ_sig, t, hold_end, delisted_map)
        n_delisted = int(returns_df["delisted"].sum())

        betas = R.compute_betas(univ, t)

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        record = {"rebalance_date": t, "hold_end": hold_end, "n_universe": len(returns_df),
                  "n_delisted": n_delisted, "mkt_ret": mkt_ret, "pct_dropped": pct_drop}

        for c in CONSTRUCTIONS:
            rank_col = RANK_COL[c]
            port = R.build_portfolio(returns_df, betas, rank_col)
            longs, shorts = port["longs"], port["shorts"]

            if not longs:
                for k in ["ls_ret", "ls_gross", "ls_cost", "ls_ba_cost", "ls_borrow_cost",
                          "long_scale", "short_scale", "mean_beta_long", "mean_beta_short",
                          "avg_turnover", "ic", "n_long", "n_short",
                          "long_hhi", "short_hhi", "long_top_share", "short_top_share",
                          "long_n_sectors", "short_n_sectors", "mean_sig_long", "mean_sig_short"]:
                    record[f"{c}_{k}"] = np.nan
                continue

            if prev_long[c]:
                long_to = len(set(longs) - prev_long[c]) / max(len(longs), 1)
                short_to = len(set(shorts) - prev_short[c]) / max(len(shorts), 1)
            else:
                long_to = short_to = 1.0
            avg_to = (long_to + short_to) / 2.0
            prev_long[c], prev_short[c] = set(longs), set(shorts)

            long_net, long_gross, long_ba, long_bw = R._leg_return_detail(
                returns_df, longs, port["lw"], is_short=False, turnover=long_to)
            short_net, short_gross, short_ba, short_bw = R._leg_return_detail(
                returns_df, shorts, port["sw"], is_short=True, turnover=short_to)

            ls_net = port["long_scale"] * long_net + port["short_scale"] * short_net
            ls_gross = port["long_scale"] * long_gross + port["short_scale"] * short_gross
            ls_ba = port["long_scale"] * long_ba + port["short_scale"] * short_ba
            ls_bw = port["long_scale"] * long_bw + port["short_scale"] * short_bw

            valid_sig = returns_df.dropna(subset=[rank_col, "raw_return"])
            if len(valid_sig) >= 10:
                ic, _ = stats.spearmanr(valid_sig[rank_col], valid_sig["raw_return"])
            else:
                ic = np.nan

            mean_sig_long = float(returns_df.set_index("ticker")[rank_col].reindex(longs).mean())
            mean_sig_short = float(returns_df.set_index("ticker")[rank_col].reindex(shorts).mean())
            if not (mean_sig_long > mean_sig_short):
                sign_violations[c] += 1

            long_sec = R.sector_composition(returns_df, longs)
            short_sec = R.sector_composition(returns_df, shorts)

            ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
            sic_map = returns_df.set_index("ticker")["sector_group"].to_dict()
            for tk in longs:
                r_ = ret_map.get(tk, np.nan)
                if np.isnan(r_):
                    continue
                grp = sic_map.get(tk, R.UNKNOWN_SIC_GROUP)
                sector_contrib[c][grp] = sector_contrib[c].get(grp, 0.0) + port["long_scale"] * port["lw"].get(tk, 0.0) * r_
            for tk in shorts:
                r_ = ret_map.get(tk, np.nan)
                if np.isnan(r_):
                    continue
                grp = sic_map.get(tk, R.UNKNOWN_SIC_GROUP)
                sector_contrib[c][grp] = sector_contrib[c].get(grp, 0.0) - port["short_scale"] * port["sw"].get(tk, 0.0) * r_

            record.update({
                f"{c}_ls_ret": ls_net, f"{c}_ls_gross": ls_gross,
                f"{c}_ls_cost": ls_ba + ls_bw, f"{c}_ls_ba_cost": ls_ba, f"{c}_ls_borrow_cost": ls_bw,
                f"{c}_long_scale": port["long_scale"], f"{c}_short_scale": port["short_scale"],
                f"{c}_mean_beta_long": port["mean_beta_long"], f"{c}_mean_beta_short": port["mean_beta_short"],
                f"{c}_avg_turnover": avg_to, f"{c}_ic": ic,
                f"{c}_n_long": len(longs), f"{c}_n_short": len(shorts),
                f"{c}_long_hhi": long_sec["hhi"], f"{c}_short_hhi": short_sec["hhi"],
                f"{c}_long_top_share": long_sec["top_share"], f"{c}_short_top_share": short_sec["top_share"],
                f"{c}_long_n_sectors": long_sec["n_sectors"], f"{c}_short_n_sectors": short_sec["n_sectors"],
                f"{c}_mean_sig_long": mean_sig_long, f"{c}_mean_sig_short": mean_sig_short,
            })

            print(f"  [{c:4s}] L/S={ls_net:+.2%} IC={ic:.3f} betaL={port['mean_beta_long']:.2f} "
                  f"betaS={port['mean_beta_short']:.2f} HHI(L/S)={long_sec['hhi']:.2f}/{short_sec['hhi']:.2f} "
                  f"nSect(L/S)={long_sec['n_sectors']}/{short_sec['n_sectors']}")

        period_records.append(record)

    elapsed = time.time() - t_total
    return {
        "period_records": period_records,
        "coverage_records": coverage_records,
        "sector_contrib": sector_contrib,
        "sign_violations": sign_violations,
        "n_periods_total": n_periods,
        "elapsed_s": elapsed,
    }
