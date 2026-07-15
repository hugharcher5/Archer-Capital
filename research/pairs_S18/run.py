"""
S18: Small-Cap Pairs Trading (Sector-Bucketed, FDR-Corrected, Persistence-Filtered)
=====================================================================================
New independent trial. Genuinely new mechanism (relative-value mean-reversion
between correlated pairs), distinct from every prior cross-sectional
single-factor trial in this registry.

METHODOLOGY (pre-registered, locked before running):

1. SECTOR-BUCKET FIRST. Reuses the SIC major-group (sic // 100) bucketing
   infrastructure built for gp_industry_neutral (imported, not rebuilt).
   Unlike that trial's "pool small groups into OTHER" convention, small
   buckets here are EXCLUDED from pair search entirely (MIN_BUCKET_SIZE=8) --
   a grab-bag OTHER bucket of unrelated micro-sectors has no economic-linkage
   story for co-movement, which is the entire justification for bucketing in
   this trial. Pairs are only ever searched for WITHIN a qualifying bucket,
   never across the full universe.

2. FORMATION (12mo) / TRADING (6mo) SPLIT, rolled forward every 6 months
   (standard Gatev-Goetzmann-Rouwenhorst 2006 design): formation windows
   overlap each other by 6 months by construction (this is how GGR rolling
   works), but TRADING periods are back-to-back and non-overlapping, covering
   the full window with no gaps. Pairs are selected using ONLY formation-
   period data; the trading period that follows uses the formation period's
   own frozen hedge ratio and spread mean/std -- no reselection, no parameter
   refresh mid-trading-period. See leakage audit in run output.

3. WITHIN EACH BUCKET: hierarchical clustering (scipy, average linkage on
   1-correlation distance, formation-period daily log returns), cut at a
   pre-registered distance threshold of 0.4 (avg intra-cluster correlation
   >= 0.6). Candidate pairs are ALL pairwise combinations within a cluster of
   size 2-8 (clusters >8 are skipped -- not "tight" by this trial's
   definition, and capped to bound the total test count). This tests
   structure in the whole correlation matrix rather than cherry-picking the
   single lowest pairwise distance out of hundreds of raw combinations.

4. COINTEGRATION + FDR CORRECTION. Engle-Granger test (statsmodels.tsa.
   stattools.coint) on formation-period adjClose price LEVELS for every
   candidate pair. ALL p-values from ALL buckets in that formation period are
   pooled and Benjamini-Hochberg FDR-corrected together (alpha=0.05,
   pre-registered) -- this is the direct analog of this registry's DSR
   discipline, applied to pair selection instead of trial selection.

5. PERSISTENCE FILTER (hard eligibility gate, not just a diagnostic). A pair
   must survive FDR correction in the CURRENT rolling formation cycle AND the
   IMMEDIATELY PRECEDING one (2 consecutive cycles in the rolling sequence)
   before it is eligible to trade. Note on "non-overlapping": consecutive
   rolling formation windows share 6 of 12 months of underlying data by
   construction (that's how GGR rolling works) -- "non-overlapping" is read
   here as "distinct, sequential entries in the rolling cycle index" (cycle i
   vs cycle i+1, not the same cycle re-tested and not skipping around), since
   a literal zero-date-overlap reading would be incompatible with using
   standard rolling formation windows at all. This interpretation is
   disclosed here rather than silently assumed.

TRADING SIGNAL (persistence-qualified pairs only): spread_t = log(P_A,t) -
beta*log(P_B,t), beta = the formation-period OLS hedge ratio (A regressed on
B). z_t = (spread_t - formation_mean) / formation_std. Entry at |z_t| > 2:
long the underperformer / short the outperformer, EQUAL DOLLAR amounts (not
beta-weighted -- the hedge ratio is used only to define the statistically
motivated spread signal, not position sizing, per the pre-registered design).
Exit at |z_t| < 0.5, or forced close at the end of the trading period.

ADJCLOSE USED THROUGHOUT (disclosed deviation from this registry's usual
"adjClose for signal, raw close for returns" split): in pairs trading, signal
and return are the SAME price series (there is no separate rank axis), so a
raw-close split/dividend artifact in either leg would corrupt both the
cointegration test AND generate spurious entries/exits AND misstate P&L
simultaneously. adjClose is used as a total-return proxy for realized P&L
throughout, consistent with standard pairs-trading literature practice.

Cost model: standard bid-ask by cap tier (reused from gp/gp_industry_neutral,
_spread()/_borrow()) -- charged at both entry and exit (one-way each,
round-trip = 2x), borrow accrued daily on whichever leg is short while a
position is open. No new data sourcing: reuses Tiingo price caches and SEC
SIC classification 100% -- see gp_industry_neutral/run.py for the underlying
fetchers.
"""

import sys
import json
import importlib.util
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from statsmodels.tsa.stattools import coint
from statsmodels.stats.multitest import multipletests

# NOTE: both this file and gp_industry_neutral's are literally named run.py --
# a plain `sys.path.insert + import run` would collide with this module's own
# name whenever anything imports pairs_S18.run under the bare name "run" (e.g.
# interactive testing). Load it by explicit file path under a unique name instead.
_GPI_RUN_PATH = Path(__file__).resolve().parents[1] / "gp_industry_neutral" / "run.py"
_spec = importlib.util.spec_from_file_location("gp_industry_neutral_run", _GPI_RUN_PATH)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
IWM_CSV  = (Path(__file__).resolve().parents[1] / "russell_reconstitution"
            / "cache" / "tiingo" / "IWM.csv")
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Pre-registered parameters (locked before running) ────────────────────────
MIN_BUCKET_SIZE     = 8      # buckets smaller than this excluded from pair search entirely
MIN_FORMATION_DAYS  = 200    # minimum overlapping adjClose trading days required in formation
CLUSTER_DIST_THRESH = 0.4    # fcluster distance threshold (avg intra-cluster corr >= 0.6)
CLUSTER_MIN_SIZE    = 2
CLUSTER_MAX_SIZE    = 8      # clusters larger than this skipped (not "tight"; bounds test count)
FDR_ALPHA           = 0.05
ENTRY_Z             = 2.0
EXIT_Z              = 0.5

IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")


# =============================================================================
# Rolling cycle schedule (GGR: 12mo formation, 6mo trading, rolled every 6mo)
# =============================================================================

def build_cycles() -> list[dict]:
    cycles = []
    i = 0
    while True:
        cycle_start = IS_START + pd.DateOffset(months=6 * i)
        formation_end = cycle_start + pd.DateOffset(months=12)
        trading_end = formation_end + pd.DateOffset(months=6)
        if trading_end > IS_END:
            break
        cycles.append({
            "cycle": i,
            "formation_start": cycle_start,
            "formation_end": formation_end,
            "trading_start": formation_end,
            "trading_end": trading_end,
        })
        i += 1
    return cycles


# =============================================================================
# Sector bucketing (reuses R's sic // 100 convention; NO pooling into OTHER --
# small buckets are excluded from pair search, not merged into a grab-bag)
# =============================================================================

def bucket_universe(univ: pd.DataFrame) -> dict[int, list[str]]:
    df = univ.copy()
    sic_num = pd.to_numeric(df["sic"], errors="coerce")
    df["sector_group"] = np.where(sic_num.notna(), (sic_num // 100).astype("Int64"), -1)
    buckets: dict[int, list[str]] = {}
    for grp, sub in df.groupby("sector_group"):
        if grp == -1:
            continue
        tickers = sub["ticker"].tolist()
        if len(tickers) >= MIN_BUCKET_SIZE:
            buckets[int(grp)] = tickers
    return buckets


# =============================================================================
# Formation-period log-return / log-price panels
# =============================================================================

def formation_log_prices(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames = {}
    for t in tickers:
        adj = R._adj_slice(t, start, end)
        if adj.empty:
            continue
        s = adj.set_index("date")["adjClose"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s[s > 0]
        if len(s) >= MIN_FORMATION_DAYS:
            frames[t] = np.log(s)
    if not frames:
        return pd.DataFrame()
    panel = pd.DataFrame(frames).sort_index()
    return panel


# =============================================================================
# Clustering within a bucket -> candidate pairs
# =============================================================================

def cluster_candidate_pairs(log_price_panel: pd.DataFrame) -> tuple[list[tuple], int]:
    """Returns (candidate_pairs, n_preclustering_pairs) for one bucket's formation panel."""
    tickers = log_price_panel.columns.tolist()
    n = len(tickers)
    n_preclustering = n * (n - 1) // 2
    if n < 2:
        return [], n_preclustering

    log_ret = log_price_panel.diff().dropna(how="all")
    corr = log_ret.corr(min_periods=int(MIN_FORMATION_DAYS * 0.8))
    corr = corr.fillna(0.0)
    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)  # enforce exact symmetry

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=CLUSTER_DIST_THRESH, criterion="distance")

    clusters: dict[int, list[str]] = {}
    for tkr, cid in zip(tickers, cluster_ids):
        clusters.setdefault(cid, []).append(tkr)

    candidate_pairs = []
    for members in clusters.values():
        if CLUSTER_MIN_SIZE <= len(members) <= CLUSTER_MAX_SIZE:
            candidate_pairs.extend(combinations(sorted(members), 2))
    return candidate_pairs, n_preclustering


# =============================================================================
# Cointegration test + hedge ratio for one candidate pair
# =============================================================================

def test_pair(log_price_panel: pd.DataFrame, a: str, b: str) -> dict | None:
    sub = log_price_panel[[a, b]].dropna()
    if len(sub) < MIN_FORMATION_DAYS:
        return None
    logA, logB = sub[a].values, sub[b].values
    try:
        score, pvalue, _ = coint(logA, logB, trend="c")
    except Exception:
        return None
    beta, alpha = np.polyfit(logB, logA, 1)
    spread = logA - beta * logB
    spread_mean = float(np.mean(spread))
    spread_std = float(np.std(spread, ddof=1))
    if spread_std <= 0 or not np.isfinite(spread_std):
        return None
    return {
        "a": a, "b": b, "pvalue": float(pvalue), "beta": float(beta),
        "spread_mean": spread_mean, "spread_std": spread_std,
    }


# =============================================================================
# Main formation-cycle processor
# =============================================================================

def run_formation_cycle(cycle: dict, tiingo_tickers, cik_sic_map) -> dict:
    fstart, fend = cycle["formation_start"], cycle["formation_end"]
    univ = R.build_universe(fstart, tiingo_tickers, cik_sic_map)
    buckets = bucket_universe(univ)

    n_universe = len(univ)
    n_bucketed = sum(len(v) for v in buckets.values())
    n_excluded_small_bucket = n_universe - n_bucketed  # (incl. unknown-SIC and sub-threshold buckets)

    all_candidates: list[tuple[int, str, str]] = []   # (sector_group, a, b)
    n_preclustering_total = 0
    panels: dict[int, pd.DataFrame] = {}

    for grp, tickers in buckets.items():
        panel = formation_log_prices(tickers, fstart, fend)
        panels[grp] = panel
        if panel.shape[1] < 2:
            continue
        pairs, n_pre = cluster_candidate_pairs(panel)
        n_preclustering_total += n_pre
        for a, b in pairs:
            all_candidates.append((grp, a, b))

    # ── Cointegration test on every candidate, pooled FDR correction ─────────
    test_results = []
    for grp, a, b in all_candidates:
        res = test_pair(panels[grp], a, b)
        if res is not None:
            res["sector_group"] = grp
            test_results.append(res)

    n_tests = len(test_results)
    if n_tests > 0:
        pvals = [r["pvalue"] for r in test_results]
        reject, pvals_adj, _, _ = multipletests(pvals, alpha=FDR_ALPHA, method="fdr_bh")
        for r, rej, padj in zip(test_results, reject, pvals_adj):
            r["fdr_reject"] = bool(rej)
            r["pvalue_adj"] = float(padj)
        n_survive_fdr = int(sum(reject))
        adj_thresh = max([r["pvalue"] for r in test_results if r["fdr_reject"]], default=np.nan)
    else:
        n_survive_fdr = 0
        adj_thresh = np.nan

    surviving = {(r["a"], r["b"]): r for r in test_results if r.get("fdr_reject")}

    return {
        "cycle": cycle["cycle"],
        "formation_start": fstart, "formation_end": fend,
        "trading_start": cycle["trading_start"], "trading_end": cycle["trading_end"],
        "n_universe": n_universe,
        "n_buckets": len(buckets),
        "n_bucketed": n_bucketed,
        "n_excluded_small_bucket": n_excluded_small_bucket,
        "n_preclustering_pairs": n_preclustering_total,
        "n_candidate_pairs": len(all_candidates),
        "n_cointegration_tests": n_tests,
        "n_survive_fdr": n_survive_fdr,
        "fdr_adj_pvalue_threshold": adj_thresh,
        "surviving_pairs": surviving,   # dict {(a,b): result}
        "universe_mcap": univ.set_index("ticker")["mcap"].to_dict(),
    }


# =============================================================================
# Persistence filter: pair must survive FDR in 2 consecutive rolling cycles
# =============================================================================

def apply_persistence_filter(cycle_results: list[dict]) -> dict[int, dict]:
    """Returns {cycle_index: {(a,b): formation_result}} of persistence-qualified,
    tradeable pairs for that cycle's trading period (uses THIS cycle's own
    formation stats, gated on ALSO having survived FDR in the prior cycle)."""
    qualified: dict[int, dict] = {}
    for i in range(1, len(cycle_results)):
        prev_survivors = set(cycle_results[i - 1]["surviving_pairs"].keys())
        cur_survivors = cycle_results[i]["surviving_pairs"]
        qualified[i] = {
            pair: res for pair, res in cur_survivors.items() if pair in prev_survivors
        }
    return qualified


# =============================================================================
# Trading simulation for one cycle's persistence-qualified pairs
# =============================================================================

def simulate_trading_period(cycle: dict, qualified_pairs: dict, universe_mcap: dict) -> dict:
    """Returns per-pair trade records and a daily book-return contribution table.
    Each daily_contrib entry is a dict {net, gross, cost} so cost drag can be
    reported cleanly (gross - net = cost), matching this registry's convention
    elsewhere (ls_gross / ls_cost / ls_net)."""
    tstart, tend = cycle["trading_start"], cycle["trading_end"]
    trades = []
    daily_contrib: dict[pd.Timestamp, list] = {}
    n_entries = 0

    for (a, b), res in qualified_pairs.items():
        beta = res["beta"]
        spread_mean, spread_std = res["spread_mean"], res["spread_std"]
        mcap_a = universe_mcap.get(a, (R.MCAP_MIN + R.MCAP_MAX) / 2)
        mcap_b = universe_mcap.get(b, (R.MCAP_MIN + R.MCAP_MAX) / 2)

        adj_a = R._adj_slice(a, tstart, tend)
        adj_b = R._adj_slice(b, tstart, tend)
        if adj_a.empty or adj_b.empty:
            continue
        sa = adj_a.set_index("date")["adjClose"]
        sa = sa[~sa.index.duplicated(keep="last")].sort_index()
        sb = adj_b.set_index("date")["adjClose"]
        sb = sb[~sb.index.duplicated(keep="last")].sort_index()
        df = pd.DataFrame({"pa": sa, "pb": sb}).dropna()
        df = df[(df["pa"] > 0) & (df["pb"] > 0)]
        if len(df) < 5:
            continue

        df["log_spread"] = np.log(df["pa"]) - beta * np.log(df["pb"])
        df["z"] = (df["log_spread"] - spread_mean) / spread_std
        df["ra"] = df["pa"].pct_change()
        df["rb"] = df["pb"].pct_change()

        in_position = False
        direction = None   # "long_a_short_b" or "long_b_short_a"
        entry_date = None
        trade_pnl = trade_gross = trade_cost = 0.0
        dates = df.index.tolist()

        for j, dt in enumerate(dates):
            z = df.loc[dt, "z"]
            if not in_position:
                if z > ENTRY_Z or z < -ENTRY_Z:
                    direction = "long_b_short_a" if z > ENTRY_Z else "long_a_short_b"
                    in_position, entry_date = True, dt
                    trade_pnl = trade_gross = trade_cost = 0.0
                    entry_cost = R._spread(mcap_a) + R._spread(mcap_b)
                    daily_contrib.setdefault(dt, []).append(
                        {"net": -entry_cost, "gross": 0.0, "cost": entry_cost})
                    trade_pnl -= entry_cost
                    trade_cost += entry_cost
                    n_entries += 1
                continue

            ra, rb = df.loc[dt, "ra"], df.loc[dt, "rb"]
            if pd.isna(ra) or pd.isna(rb):
                ra, rb = 0.0, 0.0
            if direction == "long_b_short_a":
                pair_ret = 0.5 * rb - 0.5 * ra
                borrow_mcap = mcap_a
            else:
                pair_ret = 0.5 * ra - 0.5 * rb
                borrow_mcap = mcap_b
            daily_borrow = R._borrow(borrow_mcap) / 365.0 * 0.5

            is_last_day = (j == len(dates) - 1)
            converged = abs(z) < EXIT_Z
            if converged or is_last_day:
                exit_cost = R._spread(mcap_a) + R._spread(mcap_b)
                day_cost = daily_borrow + exit_cost
                net_ret = pair_ret - day_cost
                daily_contrib.setdefault(dt, []).append(
                    {"net": net_ret, "gross": pair_ret, "cost": day_cost})
                trade_pnl += net_ret
                trade_gross += pair_ret
                trade_cost += day_cost
                trades.append({
                    "a": a, "b": b, "sector_group": res.get("sector_group"),
                    "entry_date": entry_date, "exit_date": dt,
                    "holding_days": (dt - entry_date).days,
                    "exit_reason": "converged" if converged else "forced_close",
                    "direction": direction,
                    "trade_pnl": trade_pnl, "trade_gross": trade_gross, "trade_cost": trade_cost,
                    "profitable": trade_pnl > 0,
                })
                in_position, direction, entry_date = False, None, None
            else:
                net_ret = pair_ret - daily_borrow
                daily_contrib.setdefault(dt, []).append(
                    {"net": net_ret, "gross": pair_ret, "cost": daily_borrow})
                trade_pnl += net_ret
                trade_gross += pair_ret
                trade_cost += daily_borrow
                trade_pnl += net_ret

    return {"trades": trades, "daily_contrib": daily_contrib, "n_entries": n_entries}


# =============================================================================
# Full backtest driver
# =============================================================================

def run_backtest() -> dict:
    print("[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = R.load_tiingo_tickers()
    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = R.load_prefilter()
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {row["ticker"]: (int(row["cik"]), row["sic"]) for _, row in survivors.iterrows()}
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    R.preload_all_prices()
    print(f"\n[Step 3b] Loading adjClose pickle...")
    R.preload_all_adj_prices()
    print(f"\n[Step 4] Loading GP XBRL facts (for shares/mcap only, {len(survivor_ciks):,} CIKs)...")
    R.preload_gp_facts(survivor_ciks)

    cycles = build_cycles()
    print(f"\n[Step 5] {len(cycles)} rolling cycles "
          f"({cycles[0]['formation_start'].date()} -> {cycles[-1]['trading_end'].date()})")

    cycle_results = []
    for c in cycles:
        print(f"\n[Cycle {c['cycle']}] formation {c['formation_start'].date()}->{c['formation_end'].date()} "
              f"| trading {c['trading_start'].date()}->{c['trading_end'].date()}")
        res = run_formation_cycle(c, tiingo_tickers, cik_sic_map)
        print(f"  universe={res['n_universe']} bucketed={res['n_bucketed']} "
              f"(excl_small_bucket={res['n_excluded_small_bucket']}) buckets={res['n_buckets']} "
              f"preclustering_pairs={res['n_preclustering_pairs']} candidates={res['n_candidate_pairs']} "
              f"coint_tests={res['n_cointegration_tests']} fdr_survive={res['n_survive_fdr']}")
        cycle_results.append(res)

    print("\n[Step 6] Applying persistence filter (2 consecutive cycles)...")
    qualified_by_cycle = apply_persistence_filter(cycle_results)
    for i, q in qualified_by_cycle.items():
        print(f"  cycle {i}: {len(q)} persistence-qualified pairs "
              f"(of {cycle_results[i]['n_survive_fdr']} FDR-survivors that cycle)")

    print("\n[Step 7] Trading simulation...")
    all_trades = []
    all_daily_contrib: dict[pd.Timestamp, list] = {}
    for i, qualified in qualified_by_cycle.items():
        if not qualified:
            continue
        c = cycles[i]
        sim = simulate_trading_period(c, qualified, cycle_results[i]["universe_mcap"])
        all_trades.extend(sim["trades"])
        for dt, vals in sim["daily_contrib"].items():
            all_daily_contrib.setdefault(dt, []).extend(vals)
        print(f"  cycle {i}: {len(sim['trades'])} trades closed")

    return {
        "cycles": cycles,
        "cycle_results": cycle_results,
        "qualified_by_cycle": qualified_by_cycle,
        "trades": all_trades,
        "daily_contrib": all_daily_contrib,
    }


if __name__ == "__main__":
    results = run_backtest()
    with open(RESULTS / "_raw_backtest.json", "w") as f:
        def _ser(o):
            if isinstance(o, (pd.Timestamp, )):
                return o.isoformat()
            if isinstance(o, float) and np.isnan(o):
                return None
            if isinstance(o, dict):
                return {str(k): v for k, v in o.items()}
            return str(o)
        json.dump({
            "trades": results["trades"],
            "n_cycles": len(results["cycles"]),
            "cycle_summary": [
                {k: v for k, v in r.items() if k not in ("surviving_pairs", "universe_mcap")}
                for r in results["cycle_results"]
            ],
            # Persisted for the persistence-filter audit: exactly which pairs survived FDR
            # each cycle, so cross-cycle matching (or lack thereof) can be verified, not assumed.
            "surviving_pairs_by_cycle": [
                {"cycle": r["cycle"], "pairs": [list(p) for p in r["surviving_pairs"].keys()]}
                for r in results["cycle_results"]
            ],
        }, f, indent=2, default=_ser)
    print(f"\n[Saved] -> {RESULTS}/_raw_backtest.json")

    # Save daily contrib table (book-level daily return series) for the analysis step
    rows = []
    for dt, vals in results["daily_contrib"].items():
        rows.append({
            "date": dt,
            "book_net": float(np.mean([v["net"] for v in vals])),
            "book_gross": float(np.mean([v["gross"] for v in vals])),
            "book_cost": float(np.mean([v["cost"] for v in vals])),
            "n_open": len(vals),
        })
    daily_df = (pd.DataFrame(rows).sort_values("date") if rows
                else pd.DataFrame(columns=["date", "book_net", "book_gross", "book_cost", "n_open"]))
    daily_df.to_csv(RESULTS / "daily_book_returns.csv", index=False)
    print(f"[Saved] -> {RESULTS}/daily_book_returns.csv ({len(daily_df)} days)")

    trades_df = (pd.DataFrame(results["trades"]) if results["trades"] else
                 pd.DataFrame(columns=["a", "b", "sector_group", "entry_date", "exit_date",
                                       "holding_days", "exit_reason", "direction",
                                       "trade_pnl", "trade_gross", "trade_cost", "profitable"]))
    trades_df.to_csv(RESULTS / "trades.csv", index=False)
    print(f"[Saved] -> {RESULTS}/trades.csv ({len(trades_df)} trades)")
