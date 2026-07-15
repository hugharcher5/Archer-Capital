# S21: Cluster-Conditioned Short-Term Mean Reversion (Z-Score Spread)

**Trial #27, N_TRIALS=27 (DSR).** Registry key: `clusterreversion_S21`. Status: `completed_in_sample`, **FAILS DSR**.

A genuine hybrid construction — basket/cluster-relative mean reversion, not strict pairwise cointegration (S18) and not full-cross-section reversal (S8) — reusing infrastructure and lessons from both.

## Verdict (one line)

**This is the cleanest and most extreme "real signal, destroyed by cost" result in the registry.** Gross Sharpe = 0.95 (basket construction) — genuinely attractive, would clear the promotion bar on its own — but net Sharpe = **-17.41**, net annualized return **-66.4%**, from **~11,000 bps/yr (110%/yr) cost drag**. Cluster-conditioning fixed both of the problems it was built to test: it produced a large, tradeable sample (4,683 trades vs S18's zero) and a genuinely positive gross edge (vs S8's negative gross). But it introduced a new, more severe problem — trading frequency (avg 4.9-day hold, thousands of round trips) that makes even a good signal completely non-viable once realistic transaction costs apply. **FAILS DSR at N=27.**

## Universe note

Standard $100M-$2B PIT survivorship-bias-free universe, sector-bucketed (SIC major group, `sic // 100`), reused verbatim from `gp_industry_neutral`. No new universe construction.

## Direct comparison table

| Metric | **Basket (primary)** | Peer (secondary) |
|---|---:|---:|
| Sharpe (net, ann.) | **-17.411** | -9.395 |
| Sharpe (gross, ann.) | **0.954** | 0.623 |
| CALMAR (net) | -0.668 | -0.626 |
| CALMAR (gross) | 0.653 | 0.413 |
| Ann. return (net) | -66.44% | -62.11% |
| Ann. return (gross) | +5.78% | +5.85% |
| Max drawdown (net) | -99.53% | -99.17% |
| Max drawdown (gross) | -8.84% | -14.15% |
| Market beta (vs IWM) | +0.0900 (t=+2.82) | +0.0742 (t=+1.34) |
| Cost drag (bps/yr) | **10,999** | 9,888 |
| Mean IC (\|entry z\| vs trade_gross, per-cycle) | +0.0207 (t=+1.57) | +0.0393 (t=+1.61) |
| Win rate (overall) | 17.0% | 23.1% |
| Win rate (converged exits) | 100.0% | 100.0% |
| Win rate (forced exits) | 16.7% | 22.8% |
| N trades | 4,683 | 4,858 |
| Avg holding period | 4.88 days | 4.89 days |
| N months (net series) | 60 | 60 |
| DSR per-period SR | -5.026 | (not primary) |
| DSR threshold (N=27) | 0.2306 | — |
| Clears DSR | **NO** | (not primary) |

Both constructions tell the same story; basket is the pre-registered primary result.

## The headline finding, explained

Every individual trade's gross return looks tiny (mean +0.19% per trade for basket) — but with 4,683 quasi-independent bets over 60 months, that small average edge compounds via breadth into a genuinely respectable **gross Sharpe of 0.95** (win rate 55% at the monthly-aggregated gross level, gross CALMAR 0.65). This is a textbook small-edge/high-breadth effect (Grinold's "fundamental law of active management" in miniature). The IC confirms it's real, not noise: bigger entry dislocations modestly but consistently predict bigger reversion gains (IC +0.021 to +0.039, t≈1.6 both constructions).

The problem is entirely on the cost side. Each round trip costs ~2.67% (basket) / ~2.43% (peer) — four spread charges (entry: member + basket-average; exit: member + basket-average again). With an average 4.9-day hold and ~22 concurrently open positions, capital effectively rotates through a fresh ~2.5-3% cost event roughly every week. That per-dollar cost rate, applied at this frequency, compounds to **~110%/year** — completely swamping the 5.8%/year gross edge. This isn't a modeling artifact of the book-normalization convention (verified: a fixed-slot capital allocation would show the same order-of-magnitude cost drag, since cost is a per-turnover-event rate, not diluted by book size). It's a direct, unavoidable consequence of trading this frequently at this cost tier.

Only 0.4% of trades exit via genuine Z-score convergence (|z|<0.5 within 5 days) — the rest are forced closed on the 5-day timer, having drifted (favorably or not) rather than cleanly reverted. This is consistent with the realized correlation structure: at ~0.33 average intra-cluster correlation, the "cluster" isn't tight enough for most dislocations to snap back within 5 trading days.

## Realized correlation vs S18's finding — the actual diagnostic this trial exists to run

| | S18 (pairwise, trial #24) | S21 (cluster, this trial) |
|---|---:|---:|
| Cut target | avg corr ≥ 0.6 (distance ≤0.4) | avg corr ≥ ~0.2 (distance ≤0.8) |
| **Realized** avg correlation | ~0.2-0.3 (bucket mean, diagnosed as too low to clear the cut) | **0.328** (mean), 0.306 (median), range 0.228-0.599 |
| Result | 307 candidates → 9 FDR survivors → **0 persistence-qualified trades** | 109 tradeable clusters → **4,683-4,858 trades** |

S21's clusters landed almost exactly on the "typical" correlation level S18 itself diagnosed as structurally common in this universe (not accidentally another high-correlation subset — confirms the deliberately more permissive 0.8 cut worked as intended). At that correlation level, basket-relative referencing **is** tradeable (produces both volume and a real, positive gross edge) where S18's stricter pairwise-cointegration-plus-persistence discipline produced literally nothing. This directly validates the trial's founding hypothesis about basket-vs-pairwise relationships — the failure mode that emerged (cost, not lack of signal) is a completely different, and in some ways more informative, problem than S18's.

## Selectivity funnel

| Stage | Count |
|---|---:|
| Avg universe size / cycle | 413.8 |
| Bucketed into sectors ≥8 members (sum, 10 cycles) | 3,085 |
| Raw clusters formed (all sizes, sum) | 637 |
| Clusters in pre-registered [4,15] size range | 109 |
| Tradeable clusters | 109 (100% of in-range) |
| Tradeable member-slots | 848 |
| Entries (basket) | 4,683 |
| Trades closed (basket) | 4,683 |

Every cluster in the size-qualified range became tradeable (no further filtering stage) — unlike S18, where FDR correction and persistence filtering eliminated 99%+ of candidates before any trade could occur.

## Falsification diagnostics

**Formation/trading leakage audit: PASS by construction.** Cluster membership and each member's formation-period spread mean/std are frozen at `formation_end`; each member is anchored at its own first valid formation-period log-price (not a shared panel row, which would break for staggered listing dates); trading-period Z-scores reuse these frozen formation statistics with no recomputation or reselection mid-trading-period — identical discipline to S18.

**Concentration check: PASS.** Top sector = 16.8% (basket) / 14.6% (peer) of total |trade PnL| mass across 19 sectors with trades — no single cluster or sector dominates.

**Sign-convention audit:** IC is positive for both constructions (basket +0.021, peer +0.039), consistent with the mean-reversion hypothesis (bigger dislocation → bigger predicted reversion gain). No sign flip.

## Beta

Realized beta vs IWM is small but statistically significant for the basket construction (+0.090, t=+2.82) — a mild, unintended positive market tilt, likely because the entry rule fires disproportionately during broad sector-wide moves (a whole bucket drifting together still produces individual-name dispersion vs the equal-weighted basket average). Peer construction's beta (+0.074, t=1.34) is not significant. Neither is large in absolute terms, but the basket result is a genuine (if minor) deviation from pure market-neutrality worth flagging for any future revisit of this construction.

## Why this differs from S18 and S8 (per pre-registration)

- **vs S18 (pairwise cointegration):** cluster-average referencing is unambiguously more tradeable — it generated both volume (4,683 vs 0 trades) and a real positive gross IC/Sharpe that S18's stricter discipline never got the chance to test, at a correlation level S18 itself diagnosed as the "typical" ceiling for this universe.
- **vs S8 (full-cross-section residual reversal, real IC but gross-negative):** cluster-conditioning fixed the gross-negative problem decisively — gross Sharpe here (0.95) is far better than S8's own gross return (-3.08%/yr). But it surfaced a starker version of the SAME underlying risk S8 also faced (cost drag): S21's cost drag (~11,000 bps/yr) is roughly **2.7x** S2's own next-worst cost drag in this registry (~4,079 bps/yr) and dwarfs S8's (~761 bps/yr). Conditioning fixed signal quality; it did nothing about — and arguably worsened — trading frequency.

## Construction notes (pre-registered, locked before running)

- ENTRY_Z = 2.5, EXIT_Z = 0.5, MAX_HOLD_DAYS = 5 (upper end of the idea's 2-5 day window, to give reversion maximal room before forced close).
- Cluster size range [4, 15], CLUSTER_DIST_THRESH = 0.8 (distance = 1-corr) — deliberately targeting S18's own diagnosed "typical" correlation level (~0.2), not S18's "exceptional" 0.6 bar.
- Formation (12mo) / trading (6mo) GGR rolling schedule, 10 cycles 2015-2021 — reused verbatim from `pairs_S18`.
- Basket reference = leave-one-out equal-weighted average of the OTHER cluster members (excludes the traded member itself, to avoid the position partially cancelling against its own reference).
- Equal-dollar member-vs-reference sizing (not beta-weighted), same design choice as S18; realized beta reported regardless.
- adjClose used throughout for both signal and P&L (same disclosed deviation as S18: signal and return are the same price series here).
- Turnover discipline satisfied structurally: entries only fire on genuine >2.5σ dislocations (no periodic full-book re-ranking to discipline, unlike S8).

## Infrastructure note (bug found and fixed during this trial)

Both `pairs_S18` and `gp_industry_neutral` are named `run.py`; loading both via `importlib` by explicit file path (to avoid a bare-import collision) creates **two separate module instances** of `gp_industry_neutral`, each with its own independent price/fact caches. The first backtest run produced **zero clusters in every cycle** despite 350-450-name universes, because `pairs_S18.formation_log_prices()` was calling its own internal, never-preloaded copy of `gp_industry_neutral`'s `_adj_slice()` — not the caller's preloaded one. Fixed by reassigning `P.R = R` (pointing the imported module's internal reference at the already-preloaded instance) immediately after import. Flagged here since any future trial reusing this dual-import pattern (importing two modules that each independently import a shared third module) will hit the same silent-empty-cache failure mode unless it explicitly checks for and reconciles duplicate module instances.

## Registry housekeeping note

`finalize_and_log.py`'s "re-check trial count immediately before writing" logic initially read its own already-logged row as part of "the current max," so re-running the script a second time (to correct an interpretive error in the notes text, not a new backtest) self-inflated the trial number from 27 to 28 and left a permanent gap at 27. Caught and fixed before this was left in the registry: the script now excludes its own study's prior row before computing the max, and the registry is correctly at trial #27 (verified: exactly one row for `clusterreversion_S21`, N_TRIALS_DSR=27, no gap, no duplicate). Flagged as a reusable lesson for any future trial whose finalize/logging step might be re-run more than once.

## Data / infrastructure

No new data sourcing. Reused: `gp_industry_neutral/run.py` (universe, price/fact caches, cost model), `pairs_S18/run.py` (sector bucketing, GGR rolling cycle schedule, formation log-price panels, clustering machinery), `research/russell_reconstitution/cache/tiingo/IWM.csv` (benchmark, already cached).

Output files: `research/clusterreversion_S21/results/{trades_basket.csv, trades_peer.csv, daily_book_returns_basket.csv, daily_book_returns_peer.csv, clusters_summary.csv, _cycle_summary.json, final_summary.json}`.
