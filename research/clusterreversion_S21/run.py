"""
S21: Cluster-Conditioned Short-Term Mean Reversion (Z-Score Spread)
====================================================================
New independent trial. A genuine hybrid: basket/cluster-relative mean
reversion, not strict pairwise cointegration (S18) and not full-cross-section
reversal (S8) — reuses infrastructure and lessons from both, but is neither.

WHY THIS DIFFERS FROM S18 (pairs_S18, trial #24, STOPPED zero trades):
S18 found same-sector pairwise correlations in this $100M-2B universe are
structurally low (bucket mean correlation ~0.2-0.3, rarely above 0.6) --
strict 1-to-1 cointegration essentially doesn't survive FDR + persistence
discipline here. This trial tests whether a WEAKER, basket-level relationship
(deviation from a cluster's own average, not one specific partner) fares
better: a stock can meaningfully deviate from a group average even if no
single pairwise correlation in that group is individually strong. Concretely,
this trial's clustering cut is deliberately set to target ~0.2 average
intra-cluster correlation (CLUSTER_DIST_THRESH=0.8, distance=1-corr) --
S18's OWN empirically-found "typical" bucket correlation level, not S18's
"exceptional" 0.6 bar. If this trial also collapses to near-zero tradeable
clusters, that's strong independent confirmation of S18's structural finding,
not just a restatement of it.

WHY THIS DIFFERS FROM S8 (residual_reversal, trial #14, FAILS -- real IC
t=3.40 but gross portfolio return itself negative, the "high-IC-but-gross-
negative" pattern shared with S2/PEAD): S8 used FULL-CROSS-SECTION residual
reversal (rank every name in the universe against a market-model residual,
trade the extremes). This trial conditions the same short-term-reversal
mechanism on CLUSTER membership instead -- comparing a name only to its own
sector-correlated peer group, not the whole market -- testing whether that
narrower reference group changes the gross-return problem (the same logic
that made S7's IVOL-conditioning of value a distinct, worthwhile test even
though neither S7 nor plain value passed standalone).

METHODOLOGY (pre-registered, locked before running):

1. UNIVERSE + SECTOR BUCKETING: standard $100M-2B PIT survivorship-bias-free
   universe (R.build_universe, reused from gp_industry_neutral). Sector
   bucketing via sic // 100 (R's convention), MIN_BUCKET_SIZE=8 -- reused
   VERBATIM from pairs_S18's bucket_universe(), not rebuilt.

2. FORMATION (12mo) / TRADING (6mo) rolling cycles, rolled every 6mo -- reused
   VERBATIM from pairs_S18's build_cycles() (same GGR 2006 schedule, same
   10 cycles 2015-2021). Cluster membership is locked at formation and NEVER
   reselected mid-trading-period (same look-ahead discipline as S18):
   trading_start == formation_end exactly, spread mean/std frozen at
   formation end.

3. CLUSTERING WITHIN EACH BUCKET: hierarchical clustering (scipy, average
   linkage on 1-correlation distance, formation-period daily log returns) --
   SAME clustering machinery as S18 (reused), but a DELIBERATELY MORE
   PERMISSIVE cut: CLUSTER_DIST_THRESH=0.8 (avg intra-cluster corr >= ~0.2,
   vs S18's 0.4/>=0.6). This is the core methodological difference from S18
   and the actual thing this trial tests. Cluster size filter PRE-REGISTERED
   at 4-15 members (too small collapses back into pairs-trading's already-
   diagnosed problem; too large and cluster-average deviation becomes noise).
   Clusters outside [4,15] are dropped that cycle, reported in the funnel.

4. SIGNAL: within each qualifying cluster, for each member i, define a
   cumulative log-return index anchored at the FORMATION start:
       r_i,t = ln(P_i,t) - ln(P_i,formation_start)      (adjClose throughout)
   Basket reference is LEAVE-ONE-OUT (excludes member i itself, to avoid the
   position partially cancelling against its own reference basket):
       basket_avg_{-i},t = mean_{j in cluster, j != i} r_j,t
       spread_i,t = r_i,t - basket_avg_{-i},t
   Z-scored using the FORMATION-period distribution of spread_i (mean/std
   computed ONLY over t in [formation_start, formation_end], frozen, reused
   unchanged through the trading period -- no look-ahead):
       z_i,t = (spread_i,t - spread_mean_i) / spread_std_i

5. ENTRY: |z_i,t| > 2.5 (PRE-REGISTERED, locked -- "per the idea's own
   specification"). z > 2.5: member i has OVER-performed its cluster ->
   SHORT member i, LONG the leave-one-out basket. z < -2.5: member i has
   UNDER-performed -> LONG member i, SHORT the basket. Equal-dollar sizing
   (50/50 member vs basket), NOT beta-weighted -- same design choice as S18
   ("the reference construction defines the signal, not position sizing");
   realized beta reported explicitly regardless, per instructions.

   SECONDARY comparison construction (reported alongside, not the primary
   registered result): same entry/exit/Z logic but against the single
   NEAREST PEER within the cluster (lowest formation-period 1-corr distance)
   instead of the leave-one-out basket -- directly answers whether basket-
   level referencing is more tradeable than single-partner referencing
   within the SAME weak-correlation cluster universe.

6. EXIT: |z_i,t| < 0.5 (tighter band, matching S18's EXIT_Z) OR
   MAX_HOLD_DAYS=5 trading days elapsed (PRE-REGISTERED -- upper end of the
   idea's stated 2-5 day window, giving genuine reversion the most room to
   occur before a forced close), whichever comes first. Forced close also at
   trading-period end if still open. Fraction of trades exiting via
   convergence vs forced-close reported explicitly (same as S18).

   TURNOVER DISCIPLINE: unlike S8 (which needed an explicit active/carry
   split because its full-cross-section decile ranking would otherwise force
   full-book reformation every period), this trial's entries are ALREADY
   event-triggered (only |z|>2.5 opens a position) -- there is no periodic
   full-book re-ranking to discipline. This structurally satisfies the "only
   trade genuine dislocations" principle by construction; no additional
   carry/active bookkeeping is needed or added.

7. COST MODEL: standard cap-tier bid-ask (R._spread) charged at entry+exit
   (round-trip) on BOTH legs -- member's own spread, and for the basket leg,
   the mean spread across the (up to 14) other basket members (equal-weight
   approximation, since the basket is itself equal-weighted); for the peer
   construction, the single peer's own spread. Borrow (R._borrow) accrued
   daily on whichever side is short -- member's mcap if short-member, mean
   basket-member mcap if short-basket. Given the 2-5 day hold, this trial is
   treated as HIGH cost-drag risk a priori (same caution as S2/S4/S8) --
   turnover/cost drag reported prominently, not as an afterthought.

adjClose used throughout for BOTH signal AND P&L (same disclosed deviation as
S18, same reasoning: signal and return are the same price series here, a
raw-close split/dividend artifact would corrupt the Z-score AND the P&L
simultaneously).

Registry key: clusterreversion_S21.
"""

import json
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# ── Reuse gp_industry_neutral's infra (universe/prices/cost model/metrics) and
# pairs_S18's cycle schedule + bucketing + formation log-price panels, loaded
# by explicit file path (both are named run.py -- a bare `import run` would
# collide) under unique module names, exactly as pairs_S18 itself does. ──────
_GPI_PATH = Path(__file__).resolve().parents[1] / "gp_industry_neutral" / "run.py"
_spec = importlib.util.spec_from_file_location("gp_industry_neutral_run", _GPI_PATH)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

_S18_PATH = Path(__file__).resolve().parents[1] / "pairs_S18" / "run.py"
_spec2 = importlib.util.spec_from_file_location("pairs_s18_run", _S18_PATH)
P = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(P)

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Pre-registered parameters (locked before running) ────────────────────────
MIN_BUCKET_SIZE     = P.MIN_BUCKET_SIZE       # 8, reused verbatim from S18
MIN_FORMATION_DAYS  = P.MIN_FORMATION_DAYS    # 200, reused verbatim from S18
CLUSTER_DIST_THRESH = 0.8    # avg intra-cluster corr >= ~0.2 (S18's "typical", not "exceptional" 0.6 bar)
CLUSTER_MIN_SIZE    = 4
CLUSTER_MAX_SIZE    = 15
ENTRY_Z             = 2.5
EXIT_Z              = 0.5
MAX_HOLD_DAYS        = 5

IS_START = P.IS_START
IS_END   = P.IS_END

# P (pairs_S18) loaded its OWN separate copy of gp_industry_neutral internally
# (same importlib-by-path pattern, executed at P's import time) -- that copy's
# price/adjClose/facts caches are empty until preloaded. P.formation_log_prices
# calls that internal R._adj_slice(). Point P's internal R at THIS module's
# already-preloaded R instance instead of leaving P with its own empty one --
# otherwise every formation panel silently comes back with 0 columns (found by
# inspecting a first run: n_clusters_all_sizes=0 in every cycle despite
# 350-450-name universes -- not a clustering-threshold problem, a stale-cache
# problem from the dual-import).
P.R = R


# =============================================================================
# Cluster formation within a bucket (relaxed cut vs S18; size-filtered 4-15)
# =============================================================================

def cluster_bucket(log_price_panel: pd.DataFrame) -> tuple[list[list[str]], dict, list[float]]:
    """Returns (clusters_of_4_to_15, funnel_counts, realized_avg_intra_cluster_corr_per_cluster)
    for one bucket's formation panel."""
    tickers = log_price_panel.columns.tolist()
    n = len(tickers)
    funnel = {"n_bucket_members": n, "n_clusters_all_sizes": 0, "n_clusters_in_range": 0}
    if n < CLUSTER_MIN_SIZE:
        return [], funnel, []

    log_ret = log_price_panel.diff().dropna(how="all")
    corr = log_ret.corr(min_periods=int(MIN_FORMATION_DAYS * 0.8))
    corr = corr.fillna(0.0)
    dist = 1.0 - corr.values
    np.fill_diagonal(dist, 0.0)
    dist = np.clip((dist + dist.T) / 2.0, 0.0, 2.0)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=CLUSTER_DIST_THRESH, criterion="distance")

    raw_clusters: dict[int, list[str]] = {}
    for tkr, cid in zip(tickers, cluster_ids):
        raw_clusters.setdefault(cid, []).append(tkr)

    funnel["n_clusters_all_sizes"] = len(raw_clusters)
    qualifying = [sorted(members) for members in raw_clusters.values()
                  if CLUSTER_MIN_SIZE <= len(members) <= CLUSTER_MAX_SIZE]
    funnel["n_clusters_in_range"] = len(qualifying)

    # Realized average pairwise correlation within each qualifying cluster (for the
    # "direct comparison note" vs S18's own diagnosed bucket-mean of 0.2-0.3) --
    # the empirically achieved value, not just the CLUSTER_DIST_THRESH cut target.
    avg_corrs = []
    for members in qualifying:
        sub = corr.loc[members, members].values
        iu = np.triu_indices_from(sub, k=1)
        if len(iu[0]):
            avg_corrs.append(float(np.mean(sub[iu])))
    return qualifying, funnel, avg_corrs


def nearest_peer_map(log_price_panel: pd.DataFrame, cluster: list[str]) -> dict[str, str]:
    """For each member, its nearest neighbor (lowest 1-corr distance) within the SAME cluster."""
    sub = log_price_panel[cluster]
    log_ret = sub.diff().dropna(how="all")
    corr = log_ret.corr(min_periods=int(MIN_FORMATION_DAYS * 0.8)).fillna(0.0)
    peers = {}
    for t in cluster:
        others = corr[t].drop(index=t)
        if others.empty:
            continue
        peers[t] = others.idxmax()   # highest correlation = nearest peer
    return peers


# =============================================================================
# Formation-cycle processor: universe -> buckets -> clusters -> signal params
# =============================================================================

def run_formation_cycle(cycle: dict, tiingo_tickers, cik_sic_map) -> dict:
    fstart, fend = cycle["formation_start"], cycle["formation_end"]
    univ = R.build_universe(fstart, tiingo_tickers, cik_sic_map)
    buckets = P.bucket_universe(univ)

    n_universe = len(univ)
    n_bucketed = sum(len(v) for v in buckets.values())

    all_clusters: list[dict] = []   # {sector_group, members, panel}
    n_clusters_all_sizes = n_clusters_in_range = 0

    for grp, tickers in buckets.items():
        panel = P.formation_log_prices(tickers, fstart, fend)
        if panel.shape[1] < CLUSTER_MIN_SIZE:
            continue
        clusters, funnel, avg_corrs = cluster_bucket(panel)
        n_clusters_all_sizes += funnel["n_clusters_all_sizes"]
        n_clusters_in_range += funnel["n_clusters_in_range"]
        for members, avg_corr in zip(clusters, avg_corrs):
            sub_panel = panel[members]
            peers = nearest_peer_map(panel, members)
            # Per-member spread_mean/spread_std over the FORMATION window (frozen).
            # Anchor EACH member at its OWN first valid observation, not a shared
            # panel row-0 -- names with staggered start dates within the formation
            # window (e.g. a late IPO) would otherwise get an all-NaN r_i,t series
            # if row-0 happened to predate their own first print.
            first_valid = {m: sub_panel[m].dropna().iloc[0] if sub_panel[m].notna().any() else np.nan
                           for m in members}
            r = sub_panel.subtract(pd.Series(first_valid), axis=1)
            member_stats = {}
            valid_members = []
            for m in members:
                others = [x for x in members if x != m]
                basket_avg = r[others].mean(axis=1)
                spread = (r[m] - basket_avg).dropna()
                if len(spread) < MIN_FORMATION_DAYS:
                    continue
                sm, ss = float(spread.mean()), float(spread.std(ddof=1))
                if ss <= 0 or not np.isfinite(ss):
                    continue
                # Peer-spread stats (secondary construction)
                peer = peers.get(m)
                peer_sm = peer_ss = np.nan
                if peer is not None:
                    pspread = (r[m] - r[peer]).dropna()
                    if len(pspread) >= MIN_FORMATION_DAYS:
                        peer_sm, peer_ss = float(pspread.mean()), float(pspread.std(ddof=1))
                member_stats[m] = {
                    "spread_mean": sm, "spread_std": ss,
                    "peer": peer, "peer_spread_mean": peer_sm, "peer_spread_std": peer_ss,
                }
                valid_members.append(m)
            if len(valid_members) < CLUSTER_MIN_SIZE:
                continue
            all_clusters.append({
                "sector_group": grp, "members": valid_members,
                "formation_anchor": {m: float(first_valid[m]) for m in valid_members},
                "member_stats": member_stats,
                "avg_intra_cluster_corr": avg_corr,
            })

    return {
        "cycle": cycle["cycle"],
        "formation_start": fstart, "formation_end": fend,
        "trading_start": cycle["trading_start"], "trading_end": cycle["trading_end"],
        "n_universe": n_universe,
        "n_buckets": len(buckets),
        "n_bucketed": n_bucketed,
        "n_clusters_all_sizes": n_clusters_all_sizes,
        "n_clusters_in_range": n_clusters_in_range,
        "n_tradeable_clusters": len(all_clusters),
        "n_tradeable_members": sum(len(c["members"]) for c in all_clusters),
        "clusters": all_clusters,
        "universe_mcap": univ.set_index("ticker")["mcap"].to_dict(),
    }


# =============================================================================
# Trading simulation for one cycle's clusters (basket construction, PRIMARY)
# =============================================================================

def simulate_trading_period(cycle: dict, clusters: list[dict], universe_mcap: dict,
                             construction: str = "basket") -> dict:
    """construction: 'basket' (primary, leave-one-out cluster average) or
    'peer' (secondary, nearest single cluster member)."""
    tstart, tend = cycle["trading_start"], cycle["trading_end"]
    trades = []
    daily_contrib: dict[pd.Timestamp, list] = {}
    n_entries = 0
    mid_mcap = (R.MCAP_MIN + R.MCAP_MAX) / 2

    for cl in clusters:
        members = cl["members"]
        anchors = cl["formation_anchor"]

        # Load TRADING-window adjClose for every member; the formation-period
        # anchor (each member's own first valid formation log-price) is already
        # frozen in `anchors` from run_formation_cycle -- no need to re-fetch
        # formation-window prices here, just re-index the live price to it.
        px: dict[str, pd.Series] = {}
        for m in members:
            if m not in anchors:
                continue
            adj = R._adj_slice(m, tstart, tend)
            if adj.empty:
                continue
            s = adj.set_index("date")["adjClose"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            s = s[s > 0]
            if s.empty:
                continue
            px[m] = np.log(s) - anchors[m]   # r_m,t anchored at member's own formation first-print
        if len(px) < CLUSTER_MIN_SIZE:
            continue

        r_trading = pd.DataFrame(px).sort_index()
        if r_trading.empty:
            continue
        ret_panel = {}
        for m in px:
            adj = R._adj_slice(m, tstart, tend)
            if adj.empty:
                continue
            s = adj.set_index("date")["adjClose"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret_panel[m] = s.pct_change()
        ret_df = pd.DataFrame(ret_panel).reindex(r_trading.index)

        for m in members:
            stats = cl["member_stats"].get(m)
            if stats is None or m not in r_trading.columns:
                continue

            if construction == "basket":
                others = [x for x in members if x != m and x in r_trading.columns]
                if len(others) < CLUSTER_MIN_SIZE - 1:
                    continue
                basket_avg = r_trading[others].mean(axis=1)
                spread = r_trading[m] - basket_avg
                sm, ss = stats["spread_mean"], stats["spread_std"]
                ref_mcaps = [universe_mcap.get(x, mid_mcap) for x in others]
                ref_spread_cost = float(np.mean([R._spread(mc) for mc in ref_mcaps]))
                ref_borrow = lambda: float(np.mean([R._borrow(mc) for mc in ref_mcaps]))
                ref_ret = lambda dt: float(ret_df.loc[dt, others].mean()) if dt in ret_df.index else np.nan
            else:  # peer
                peer = stats.get("peer")
                if peer is None or peer not in r_trading.columns:
                    continue
                spread = r_trading[m] - r_trading[peer]
                sm, ss = stats["peer_spread_mean"], stats["peer_spread_std"]
                if np.isnan(sm) or np.isnan(ss) or ss <= 0:
                    continue
                peer_mcap = universe_mcap.get(peer, mid_mcap)
                ref_spread_cost = R._spread(peer_mcap)
                ref_borrow = lambda: R._borrow(peer_mcap)
                ref_ret = lambda dt: float(ret_df.loc[dt, peer]) if (dt in ret_df.index and peer in ret_df.columns) else np.nan

            if ss <= 0 or not np.isfinite(ss):
                continue
            z = (spread - sm) / ss
            mcap_m = universe_mcap.get(m, mid_mcap)

            in_position = False
            direction = None
            entry_date = None
            entry_z = None
            hold_days = 0
            trade_pnl = trade_gross = trade_cost = 0.0
            dates = r_trading.index.tolist()

            for j, dt in enumerate(dates):
                zt = z.get(dt, np.nan)
                if pd.isna(zt):
                    continue
                if not in_position:
                    if zt > ENTRY_Z or zt < -ENTRY_Z:
                        direction = "long_ref_short_m" if zt > ENTRY_Z else "long_m_short_ref"
                        in_position, entry_date, entry_z, hold_days = True, dt, float(zt), 0
                        trade_pnl = trade_gross = trade_cost = 0.0
                        entry_cost = R._spread(mcap_m) + ref_spread_cost
                        daily_contrib.setdefault(dt, []).append(
                            {"net": -entry_cost, "gross": 0.0, "cost": entry_cost})
                        trade_pnl -= entry_cost
                        trade_cost += entry_cost
                        n_entries += 1
                    continue

                hold_days += 1
                rm = ret_df.loc[dt, m] if (dt in ret_df.index and m in ret_df.columns) else np.nan
                rref = ref_ret(dt)
                rm = 0.0 if pd.isna(rm) else rm
                rref = 0.0 if pd.isna(rref) else rref

                if direction == "long_ref_short_m":
                    pair_ret = 0.5 * rref - 0.5 * rm
                    daily_borrow = R._borrow(mcap_m) / 365.0 * 0.5   # short leg = member
                else:
                    pair_ret = 0.5 * rm - 0.5 * rref
                    daily_borrow = ref_borrow() / 365.0 * 0.5        # short leg = reference (basket/peer)

                is_last_day = (j == len(dates) - 1)
                converged = abs(zt) < EXIT_Z
                forced_time = hold_days >= MAX_HOLD_DAYS
                if converged or forced_time or is_last_day:
                    exit_cost = R._spread(mcap_m) + ref_spread_cost
                    day_cost = daily_borrow + exit_cost
                    net_ret = pair_ret - day_cost
                    daily_contrib.setdefault(dt, []).append(
                        {"net": net_ret, "gross": pair_ret, "cost": day_cost})
                    trade_pnl += net_ret
                    trade_gross += pair_ret
                    trade_cost += day_cost
                    exit_reason = "converged" if converged else ("time_forced" if forced_time else "period_forced_close")
                    trades.append({
                        "member": m, "reference": cl["members"] if construction == "basket" else stats.get("peer"),
                        "construction": construction, "sector_group": cl["sector_group"],
                        "entry_date": entry_date, "exit_date": dt, "holding_days": hold_days,
                        "exit_reason": exit_reason, "direction": direction, "entry_z": entry_z,
                        "trade_pnl": trade_pnl, "trade_gross": trade_gross, "trade_cost": trade_cost,
                        "profitable": trade_pnl > 0,
                    })
                    in_position, direction, entry_date, entry_z, hold_days = False, None, None, None, 0
                else:
                    net_ret = pair_ret - daily_borrow
                    daily_contrib.setdefault(dt, []).append(
                        {"net": net_ret, "gross": pair_ret, "cost": daily_borrow})
                    trade_pnl += net_ret
                    trade_gross += pair_ret
                    trade_cost += daily_borrow

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
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices...")
    R.preload_all_prices()
    print(f"\n[Step 3b] Loading adjClose pickle...")
    R.preload_all_adj_prices()
    print(f"\n[Step 4] Loading GP XBRL facts (shares/mcap only, {len(survivor_ciks):,} CIKs)...")
    R.preload_gp_facts(survivor_ciks)

    cycles = P.build_cycles()
    print(f"\n[Step 5] {len(cycles)} rolling cycles "
          f"({cycles[0]['formation_start'].date()} -> {cycles[-1]['trading_end'].date()})")

    cycle_results = []
    for c in cycles:
        print(f"\n[Cycle {c['cycle']}] formation {c['formation_start'].date()}->{c['formation_end'].date()} "
              f"| trading {c['trading_start'].date()}->{c['trading_end'].date()}")
        res = run_formation_cycle(c, tiingo_tickers, cik_sic_map)
        print(f"  universe={res['n_universe']} bucketed={res['n_bucketed']} buckets={res['n_buckets']} "
              f"clusters_all_sizes={res['n_clusters_all_sizes']} clusters_in_range[4,15]={res['n_clusters_in_range']} "
              f"tradeable_clusters={res['n_tradeable_clusters']} tradeable_members={res['n_tradeable_members']}")
        cycle_results.append(res)

    print("\n[Step 6] Trading simulation (PRIMARY: basket construction)...")
    all_trades_basket, all_daily_basket = [], {}
    for i, res in enumerate(cycle_results):
        if not res["clusters"]:
            continue
        sim = simulate_trading_period(cycles[i], res["clusters"], res["universe_mcap"], construction="basket")
        all_trades_basket.extend(sim["trades"])
        for dt, vals in sim["daily_contrib"].items():
            all_daily_basket.setdefault(dt, []).extend(vals)
        print(f"  cycle {i}: {sim['n_entries']} entries, {len(sim['trades'])} trades closed (basket)")

    print("\n[Step 7] Trading simulation (SECONDARY: nearest-peer construction)...")
    all_trades_peer, all_daily_peer = [], {}
    for i, res in enumerate(cycle_results):
        if not res["clusters"]:
            continue
        sim = simulate_trading_period(cycles[i], res["clusters"], res["universe_mcap"], construction="peer")
        all_trades_peer.extend(sim["trades"])
        for dt, vals in sim["daily_contrib"].items():
            all_daily_peer.setdefault(dt, []).extend(vals)
        print(f"  cycle {i}: {sim['n_entries']} entries, {len(sim['trades'])} trades closed (peer)")

    return {
        "cycles": cycles, "cycle_results": cycle_results,
        "trades_basket": all_trades_basket, "daily_basket": all_daily_basket,
        "trades_peer": all_trades_peer, "daily_peer": all_daily_peer,
    }


def _save(results: dict) -> None:
    def _ser(o):
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, float) and np.isnan(o):
            return None
        if isinstance(o, dict):
            return {str(k): v for k, v in o.items()}
        return str(o)

    with open(RESULTS / "_cycle_summary.json", "w") as f:
        json.dump([
            {k: v for k, v in r.items() if k not in ("clusters", "universe_mcap")}
            for r in results["cycle_results"]
        ], f, indent=2, default=_ser)

    # Realized (not just target-cut) average intra-cluster correlation per qualifying
    # cluster -- the direct comparison point vs S18's own diagnosed bucket-mean of 0.2-0.3.
    cluster_rows = []
    for res in results["cycle_results"]:
        for cl in res["clusters"]:
            cluster_rows.append({
                "cycle": res["cycle"], "sector_group": cl["sector_group"],
                "n_members": len(cl["members"]), "avg_intra_cluster_corr": cl["avg_intra_cluster_corr"],
            })
    pd.DataFrame(cluster_rows).to_csv(RESULTS / "clusters_summary.csv", index=False)
    print(f"[Saved] -> clusters_summary.csv ({len(cluster_rows)} clusters)")

    for tag, trades in [("basket", results["trades_basket"]), ("peer", results["trades_peer"])]:
        df = (pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["member", "reference", "construction", "sector_group", "entry_date", "exit_date",
                     "holding_days", "exit_reason", "direction", "entry_z", "trade_pnl", "trade_gross", "trade_cost", "profitable"]))
        df.to_csv(RESULTS / f"trades_{tag}.csv", index=False)
        print(f"[Saved] -> trades_{tag}.csv ({len(df)} trades)")

    for tag, daily in [("basket", results["daily_basket"]), ("peer", results["daily_peer"])]:
        rows = []
        for dt, vals in daily.items():
            rows.append({
                "date": dt,
                "book_net": float(np.mean([v["net"] for v in vals])),
                "book_gross": float(np.mean([v["gross"] for v in vals])),
                "book_cost": float(np.mean([v["cost"] for v in vals])),
                "n_open": len(vals),
            })
        df = (pd.DataFrame(rows).sort_values("date") if rows else
              pd.DataFrame(columns=["date", "book_net", "book_gross", "book_cost", "n_open"]))
        df.to_csv(RESULTS / f"daily_book_returns_{tag}.csv", index=False)
        print(f"[Saved] -> daily_book_returns_{tag}.csv ({len(df)} days)")


if __name__ == "__main__":
    results = run_backtest()
    _save(results)
