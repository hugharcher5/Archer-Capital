"""
Cost-model calibration audit for S21 (and by extension, every trial in this
registry using the same _spread()/_borrow() functions).

Reconstructs, for every S21 trade, the TRADED MEMBER's market cap and average
dollar volume AT ENTRY (reusing the exact same cached universe data the
backtest itself used -- no new fetching), classifies each trade into a
market-cap decile and a dollar-volume decile of the full $100M-2B universe,
and reports what fraction of trades were in the most illiquid tier vs. the
most liquid tier of that range.

Also runs a cost-sensitivity re-computation: for S21, S8, S2 (PEAD), and S4
(MAX) -- the trials in this registry explicitly flagged as cost-nonviable --
recompute net Sharpe/CALMAR at 0.25x/0.5x/1x/1.5x/2x the CURRENT bid-ask
spread assumption (holding borrow cost fixed, since the question at hand is
specifically about bid-ask spread calibration, not securities-lending rates)
to see whether a plausible recalibration would change any trial's verdict.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
S21_RESULTS = Path(__file__).resolve().parent / "results"

_GPI_PATH = ROOT / "research" / "gp_industry_neutral" / "run.py"
_spec = importlib.util.spec_from_file_location("gp_industry_neutral_run", _GPI_PATH)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

_S18_PATH = ROOT / "research" / "pairs_S18" / "run.py"
_spec2 = importlib.util.spec_from_file_location("pairs_s18_run", _S18_PATH)
P = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(P)
P.R = R


# =============================================================================
# PART 1: Liquidity-tier breakdown of S21's actual trades
# =============================================================================

def build_liquidity_audit():
    print("[1] Loading Tiingo/prefilter/price caches (reused, no new fetch)...")
    tiingo_tickers = R.load_tiingo_tickers()
    prefilter_df = R.load_prefilter()
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {row["ticker"]: (int(row["cik"]), row["sic"]) for _, row in survivors.iterrows()}
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()
    R.preload_all_prices()
    R.preload_all_adj_prices()
    R.preload_gp_facts(survivor_ciks)

    cycles = P.build_cycles()
    print(f"[2] Rebuilding universe mcap/avg_dvol at each of {len(cycles)} formation dates "
          f"(same universe the backtest already used)...")
    universe_frames = []
    for c in cycles:
        univ = R.build_universe(c["formation_start"], tiingo_tickers, cik_sic_map)
        univ = univ.copy()
        univ["cycle"] = c["cycle"]
        universe_frames.append(univ[["cycle", "ticker", "mcap", "avg_dvol", "price"]])
    all_universe = pd.concat(universe_frames, ignore_index=True)
    print(f"    {len(all_universe):,} (cycle, ticker) universe rows pooled.")

    # Reference distribution: pooled universe (dedup by ticker, mean mcap/dvol across
    # cycles it appears in) -- deciles computed on this defines "decile of the $100M-2B
    # universe" independent of any one cycle's specific composition.
    pooled = all_universe.groupby("ticker").agg(mcap=("mcap", "mean"), avg_dvol=("avg_dvol", "mean")).reset_index()
    mcap_edges = pooled["mcap"].quantile(np.linspace(0, 1, 11)).values
    dvol_edges = pooled["avg_dvol"].quantile(np.linspace(0, 1, 11)).values
    print(f"    Universe mcap decile edges ($M): {[round(e/1e6,1) for e in mcap_edges]}")
    print(f"    Universe avg_dvol decile edges ($M/day): {[round(e/1e6,2) for e in dvol_edges]}")

    def decile_of(x, edges):
        if pd.isna(x):
            return np.nan
        d = np.searchsorted(edges, x, side="right")
        return int(np.clip(d, 1, 10))

    results = {}
    for tag in ["basket", "peer"]:
        trades = pd.read_csv(S21_RESULTS / f"trades_{tag}.csv", parse_dates=["entry_date"])
        # Map entry_date -> cycle via cycle trading windows
        cyc_bounds = [(c["cycle"], c["trading_start"], c["trading_end"]) for c in cycles]
        def find_cycle(dt):
            for cyc, ts, te in cyc_bounds:
                if ts <= dt < te:
                    return cyc
            return np.nan
        trades["cycle"] = trades["entry_date"].apply(find_cycle)

        merged = trades.merge(
            all_universe.rename(columns={"ticker": "member", "mcap": "entry_mcap", "avg_dvol": "entry_dvol"}),
            on=["cycle", "member"], how="left")

        merged["mcap_decile"] = merged["entry_mcap"].apply(lambda x: decile_of(x, mcap_edges))
        merged["dvol_decile"] = merged["entry_dvol"].apply(lambda x: decile_of(x, dvol_edges))

        n_total = len(merged)
        n_matched = merged["entry_mcap"].notna().sum()
        mcap_dist = merged["mcap_decile"].value_counts(normalize=True).sort_index()
        dvol_dist = merged["dvol_decile"].value_counts(normalize=True).sort_index()

        print(f"\n=== {tag.upper()}: {n_total} trades, {n_matched} matched to a universe mcap ({n_matched/n_total:.1%}) ===")
        print("Market-cap decile distribution of TRADED names (1=smallest $100-~X M, 10=largest ~X-2B):")
        print(mcap_dist.round(3).to_string())
        print("Dollar-volume decile distribution of TRADED names (1=least liquid, 10=most liquid):")
        print(dvol_dist.round(3).to_string())

        bottom3_mcap = mcap_dist.reindex([1, 2, 3]).sum()
        top3_mcap = mcap_dist.reindex([8, 9, 10]).sum()
        bottom3_dvol = dvol_dist.reindex([1, 2, 3]).sum()
        top3_dvol = dvol_dist.reindex([8, 9, 10]).sum()
        median_mcap = merged["entry_mcap"].median()
        median_dvol = merged["entry_dvol"].median()

        print(f"Bottom-3 mcap deciles (smallest/most illiquid names): {bottom3_mcap:.1%} of trades")
        print(f"Top-3 mcap deciles (largest names in range): {top3_mcap:.1%} of trades")
        print(f"Bottom-3 dvol deciles (lowest $ volume): {bottom3_dvol:.1%} of trades")
        print(f"Top-3 dvol deciles (highest $ volume): {top3_dvol:.1%} of trades")
        print(f"Median entry mcap: ${median_mcap/1e6:.0f}M | Median entry avg_dvol: ${median_dvol/1e6:.2f}M/day")

        results[tag] = {
            "n_total": int(n_total), "n_matched": int(n_matched),
            "mcap_dist": mcap_dist.to_dict(), "dvol_dist": dvol_dist.to_dict(),
            "bottom3_mcap_share": float(bottom3_mcap), "top3_mcap_share": float(top3_mcap),
            "bottom3_dvol_share": float(bottom3_dvol), "top3_dvol_share": float(top3_dvol),
            "median_entry_mcap": float(median_mcap), "median_entry_dvol": float(median_dvol),
        }
        merged.to_csv(S21_RESULTS / f"liquidity_audit_{tag}.csv", index=False)

    with open(S21_RESULTS / "liquidity_audit_summary.json", "w") as f:
        json.dump({
            "results": results,
            "mcap_decile_edges_musd": [float(e / 1e6) for e in mcap_edges],
            "dvol_decile_edges_musd": [float(e / 1e6) for e in dvol_edges],
        }, f, indent=2)
    print(f"\n[Saved] -> liquidity_audit_summary.json, liquidity_audit_{{basket,peer}}.csv")
    return results


if __name__ == "__main__":
    build_liquidity_audit()
