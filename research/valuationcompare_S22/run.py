"""
S22: Long-Only Quarterly Valuation Portfolios -- DCF vs. P/E vs. P/S vs. P/B
==============================================================================
Four sub-trials sharing one DSR family slot (same underlying question -- does
a valuation signal identify outperforming stocks -- tested four ways), same
family-sharing convention as the IVOL/MAX trials.

GENUINELY NEW CONSTRUCTION for this registry: LONG-ONLY, not beta-neutral
long/short. Beta is expected to be well above zero -- reported prominently,
not treated as a defect.

Universe/window/cost-model/beta/sector infra reused verbatim (file-path
import) from gp_industry_neutral/run.py -- same $100M-$2B PIT universe,
same 2015-01-01->2021-01-01 quarterly window (24 periods) already used by
nearly every trial in this registry, same cost model (_spread, _leg_return_
detail with is_short=False -- no borrow leg, this is long-only).

Signal convention (uniform across all four, for direct comparability):
  undervaluation_score, HIGHER = more undervalued = long candidate.
    DCF:  margin_of_safety = (DCF value - price) / price        [already higher=better]
    P/E:  -PE   (only when TTM net income > 0; else excluded)
    P/S:  -PS   (only when TTM revenue > 0; else excluded -- practically never binds)
    P/B:  -PB   (only when book equity > 0; else excluded)
Portfolio: top 20% by undervaluation_score, equal-weight, long-only. Leg size
floor matches this registry's standard convention (max(ceil(n*0.20), 20)).
"""
import contextlib
import importlib.util
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pit_facts as PF          # noqa: E402
import dcf_adapter as DA        # noqa: E402

_GPI_PATH = Path(__file__).resolve().parents[1] / "gp_industry_neutral" / "run.py"
_spec = importlib.util.spec_from_file_location("gpi_run_s22", _GPI_PATH)
GPI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(GPI)

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
IWM_CSV = (Path(__file__).resolve().parents[1] / "russell_reconstitution"
           / "cache" / "tiingo" / "IWM.csv")

SIGNALS = ["dcf", "pe", "ps", "pb"]
MIN_LEG = 20


def sector_group_of(sic) -> int:
    try:
        s = float(sic)
        return int(s // 100)
    except (TypeError, ValueError):
        return -1


def compute_ratio_signals(univ: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    as_of_str = as_of.strftime("%Y-%m-%d")
    rows = []
    for _, r in univ.iterrows():
        cik = int(r["cik"])
        facts = PF.get_facts(cik)
        mcap = r["mcap"]

        ni_series = PF.annual_series(facts, PF.NET_INCOME_CONCEPTS, as_of_str)
        rev_series = PF.annual_series(facts, PF.REVENUE_CONCEPTS, as_of_str)
        book_eq = PF.scalar_pit(facts, PF.STOCKHOLDERS_EQUITY_CONCEPTS, as_of_str)

        ni = float(ni_series.iloc[-1]) if not ni_series.empty else np.nan
        rev = float(rev_series.iloc[-1]) if not rev_series.empty else np.nan

        pe = mcap / ni if (not np.isnan(ni) and ni > 0) else np.nan
        ps = mcap / rev if (not np.isnan(rev) and rev > 0) else np.nan
        pb = mcap / book_eq if (not np.isnan(book_eq) and book_eq > 0) else np.nan

        rows.append({
            "ticker": r["ticker"], "cik": cik, "sic": r.get("sic"),
            "mcap": mcap,
            "score_pe": (-pe) if not np.isnan(pe) else np.nan,
            "score_ps": (-ps) if not np.isnan(ps) else np.nan,
            "score_pb": (-pb) if not np.isnan(pb) else np.nan,
            "pe_raw": pe, "ps_raw": ps, "pb_raw": pb,
        })
    return pd.DataFrame(rows)


def build_long_only_portfolio(df: pd.DataFrame, score_col: str) -> dict:
    d = df.dropna(subset=[score_col, "raw_return"]).copy()
    n = len(d)
    if n < 40:
        return {"longs": [], "w": {}}
    leg = max(int(np.ceil(n * 0.20)), MIN_LEG)
    leg = min(leg, n // 2)
    ranked = d.sort_values(score_col, ascending=False)
    longs = ranked.iloc[:leg]["ticker"].tolist()
    w = {t: 1.0 / leg for t in longs}
    return {"longs": longs, "w": w}


def run_backtest() -> dict:
    print("[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = GPI.load_tiingo_tickers()
    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = GPI.load_prefilter()
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices...")
    GPI.preload_all_prices()
    print(f"\n[Step 3b] Preloading adjClose prices...")
    GPI.preload_all_adj_prices()
    print(f"\n[Step 4] Preloading GP-style shares facts (for PIT mcap, {len(survivor_ciks):,} CIKs)...")
    GPI.preload_gp_facts(survivor_ciks)
    print(f"\n[Step 5] Preloading S22 valuation facts (net income/revenue/equity + DCF inputs)...")
    PF.preload_facts(survivor_ciks)

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records = {sig: [] for sig in SIGNALS}
    prev_sets = {sig: set() for sig in SIGNALS}
    sector_records = {sig: [] for sig in SIGNALS}

    n_periods = len(GPI.REBALANCE_DATES)
    for i, t in enumerate(GPI.REBALANCE_DATES):
        hold_end = GPI.REBALANCE_DATES[i + 1] if i + 1 < n_periods else GPI.IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        univ = GPI.build_universe(t, tiingo_tickers, cik_sic_map)
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        ratios = compute_ratio_signals(univ, t)
        n_pe = ratios["score_pe"].notna().sum()
        n_ps = ratios["score_ps"].notna().sum()
        n_pb = ratios["score_pb"].notna().sum()

        dcf_scores = []
        betas = GPI.compute_betas(univ, t)
        for _, r in univ.iterrows():
            beta = betas.get(r["ticker"], 1.0)
            mos = DA.pit_dcf_margin_of_safety(r["ticker"], int(r["cik"]), t, r["price"], r["shares"], beta)
            dcf_scores.append(mos)
        ratios["score_dcf"] = dcf_scores
        n_dcf = ratios["score_dcf"].notna().sum()
        print(f"  [Signal] dcf={n_dcf} pe={n_pe} ps={n_ps} pb={n_pb} (of {len(univ)} universe)")

        try:
            returns_df = GPI.compute_returns(univ.merge(ratios.drop(columns=["mcap", "sic"]), on=["ticker", "cik"]),
                                              t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        for sig in SIGNALS:
            score_col = "score_dcf" if sig == "dcf" else f"score_{sig}"
            port = build_long_only_portfolio(returns_df, score_col)
            longs, w = port["longs"], port["w"]
            if not longs:
                print(f"  [{sig}] [SKIP] n<40 after signal filter")
                continue

            turnover = (len(set(longs) - prev_sets[sig]) / max(len(longs), 1)
                        if prev_sets[sig] else 1.0)
            prev_sets[sig] = set(longs)

            net, gross, ba_cost, _ = GPI._leg_return_detail(returns_df, longs, w, is_short=False, turnover=turnover)

            valid_sig = returns_df.dropna(subset=[score_col, "raw_return"])
            ic = (stats.spearmanr(valid_sig[score_col], valid_sig["raw_return"])[0]
                  if len(valid_sig) >= 10 else np.nan)

            sub = returns_df[returns_df["ticker"].isin(longs)]
            grp_counts = sub["sic"].map(sector_group_of).value_counts()
            weights = grp_counts / grp_counts.sum() if grp_counts.sum() > 0 else grp_counts
            hhi = float((weights ** 2).sum()) if len(weights) else np.nan
            top_share = float(weights.max()) if len(weights) else np.nan

            period_records[sig].append({
                "rebalance_date": t, "hold_end": hold_end,
                "n_universe": len(returns_df), "n_with_signal": int(returns_df[score_col].notna().sum()),
                "ret_net": net, "ret_gross": gross, "cost": ba_cost,
                "turnover": turnover, "mkt_ret": mkt_ret, "ic": ic,
                "n_long": len(longs), "sector_hhi": hhi, "sector_top_share": top_share,
            })
            sector_records[sig].append({"rebalance_date": t, "top_sic_group": weights.idxmax() if len(weights) else None,
                                         "top_share": top_share})
            print(f"  [{sig:3s}] ret={net:+.2%} gross={gross:+.2%} cost={ba_cost:.3%} "
                  f"to={turnover:.0%} IC={ic:.3f} n={len(longs)} sectorHHI={hhi:.3f}")

    return {"period_records": period_records, "sector_records": sector_records}


if __name__ == "__main__":
    results = run_backtest()
    for sig in SIGNALS:
        pd.DataFrame(results["period_records"][sig]).to_csv(RESULTS / f"s22_{sig}_period_returns.csv", index=False)
        pd.DataFrame(results["sector_records"][sig]).to_csv(RESULTS / f"s22_{sig}_sector.csv", index=False)
    print(f"\n[Saved] -> {RESULTS}/s22_{{signal}}_period_returns.csv (x4), s22_{{signal}}_sector.csv (x4)")
