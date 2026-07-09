# S13: Opportunistic Insider Cluster Buying — Results (trial #20)

**Study:** `insiderclusters_S13` · **Registry:** `research/trial_registry.csv` · **Status:** completed (in-sample, coverage-limited)

## Hypothesis

E1 (mgmt_pit trial #1, old 4-trial-count era) tested naive net insider buying
(Form 4, all trade types pooled) and found essentially no signal
(IC ≈ +0.0002, t = 0.02). Cohen, Malloy & Pomorski (2012, "Decoding Inside
Information") argue this pooling destroys signal: ROUTINE trades (scheduled/
recurring — proxied here by same-calendar-month purchases in ≥2 of the prior
3 years, or an explicit 10b5-1 flag) carry little information, while
OPPORTUNISTIC trades (discretionary, clustered) carry real information. This
trial isolates a specific, pre-registered refinement: **cluster buying** — 3+
different insiders making opportunistic purchases at the same company within
a rolling 30-day window — as the long signal, against a "no insider buying at
all" short leg (a clean tilt-vs-no-signal design, not short-on-selling).

## Trial accounting

- **N_TRIALS_DSR = 20.** At task start, N was 17 (Russell Reconstitution
  family). This trial's data pipeline (Form-4 owner-identity re-fetch across
  ~10.6k filings + backtest) ran for ~2.5 hours; two other trials
  (`analystrevision_S14` → #18, `compositeIC_S15` → #19) were logged to the
  shared registry by concurrent sessions during that window. This was only
  discovered via a `trial_number` collision at registry-write time and was
  corrected post-hoc: this trial is **#20**, DSR threshold recomputed at the
  true N (verdict unchanged either way — still fails). **Process lesson:**
  read `n_trials_dsr` fresh from the registry immediately before writing
  results, not just once at task start, for any trial with a long runtime.

## Data reuse (per instructions — no new data infrastructure built)

Universe construction, Tiingo prices, SEC XBRL shares, and SEC
submissions/Form-4 filing lists were reused 100% from `mgmt_pit`'s existing
(E1) cache, read-only. The existing parsed Form-4 cache had transaction
date/shares/price/action but not reporting-owner identity or the 10b5-1
flag — both needed to tell "3 different insiders" apart. Of ~40,104
previously-cached filings, only 10,649 contained an open-market purchase
(action=A, price>0); only those needed their raw XML re-fetched once to
extract owner identity. 10,646/10,649 (100.0%) resolved successfully (3
network-timeout errors). Confirmed empirically that the `aff10b5One` flag
does not exist in the Form-4 XML schema before 2023 (checked a live 2017
Apple Form 4 vs. a 2026 one) — so the 10b5-1 branch of the routine
classification is a structural no-op for this entire in-sample window, as
anticipated.

## ⚠️ Data-coverage finding (discovered by this trial, not caused by it)

The reused `mgmt_pit` Form-4 cache (`form4_all.json`, E1's own fetch output)
has meaningful transaction density only **through 2017-12**. Density craters
to a single stray entry after that. As a result:

- **11/24 quarters (2018-04-01 onward) are total data blackouts** — zero
  Form-4 transactions found for *any* company in the universe, confirmed by
  spot-checking that real Form-4 filings exist in SEC submissions for that
  window but were simply never fetched into E1's cache.
- This is **not evidence that cluster events became rare** — it is a
  pre-existing gap in the reused pipeline (likely an incomplete/interrupted
  historical E1 run). The effective supported window for this signal is
  **13/24 quarters: 2015-01-01 through 2018-01-01**.
- **Within those 13 data-covered quarters, the name-count guard (long leg
  <20 names) never fired** — every single covered quarter had ≥38
  cluster-flagged names. Cluster buying was *not* rare when data actually
  existed; the apparent "54% of quarters usable" statistic is a data-cutoff
  artifact, not a rarity finding.
- **Open item for any future work reusing this cache:** the Form-4 pipeline
  needs a real re-fetch for 2018-Q2 through 2020-Q4 before this signal (or
  any insider-buying signal) can be tested over the full 2015–2021 window.

## Signal counts (within the 13 data-covered quarters)

| Metric | Value |
|---|---|
| Total cluster-buying events (company-quarters) | 822 |
| Total opportunistic purchases classified | 12,328 |
| Total routine purchases classified | 438 |
| Owner-identity resolved | 10,646/10,649 (100.0%) |

## Information Coefficient (Spearman, cost-independent)

| Signal | Mean IC | t-stat | n |
|---|---|---|---|
| **Cluster-flag (this trial)** | **+0.0138** | **0.850** | 13 |
| Naive M1 net-buying re-run (E1, same universe/window) | +0.0043 | 0.308 | 14 |
| E1 original (2015–2021, full window, old universe build) | +0.0002 | 0.02 | 24 |

The cluster refinement's IC is directionally positive and ~3x the magnitude
of the naive re-run's IC on the identical universe/window — some evidence
the routine/opportunistic + clustering refinement isolates *more* signal
than naive pooling — but at t=0.85 (n=13) it is **far from statistically
significant**. The naive re-run's IC (+0.0043, t=0.31) is also indistinguishable
from E1's original null (+0.0002, t=0.02), confirming this isn't an artifact
of a different universe/window — both naive versions agree there's ~no signal.

## P&L metrics (13 usable, data-covered quarters)

| Metric | Value |
|---|---|
| Annualized return (net) | −8.26% |
| Annualized return (gross) | −0.74% |
| Cost drag (annualized) | 780 bps |
| Mean turnover / quarter | 59% |
| Sharpe (net, annualized) | −0.857 |
| CALMAR | −0.306 |
| Max drawdown | −27.03% |
| Market beta | 0.056 (t=0.264 — beta-neutral construction working) |
| Win rate | 38.5% |
| N periods (usable) | 13 |

**DSR check (N_TRIALS=20):** per-period SR = −0.4285, threshold = 0.4562 →
**FAILS DSR.**

## Falsification diagnostics

- **Sign-convention audit:** cluster-leg mean return exceeded the
  no-buying-leg mean return in only **6/13** quarters (i.e. the "wrong" sign
  — cluster leg underperformed — in 7/13, a majority). This cuts directly
  against the hypothesis's directional prediction and is consistent with the
  negative net P&L above.
- **Concentration check:** no single quarter (max 16.1% of |return| mass,
  2016-07-01) or SIC major-group (max 11.2%, SIC 34) dominates — the negative
  result is not a single-episode or single-sector artifact.
- **Event-count caveat:** 822 cluster events is not a small sample in
  absolute terms, but it is confined to only 13 quarters (2015–2018) rather
  than the full 24-quarter window, per the data-coverage finding above.

## Comparison to E1

| | E1 (original) | S13 naive re-run (same universe/window) | S13 cluster refinement |
|---|---|---|---|
| IC | +0.0002 | +0.0043 | **+0.0138** |
| IC t-stat | 0.02 | 0.308 | 0.850 |
| Net Sharpe | −0.075 | n/a (IC comparison only) | −0.857 |
| DSR | FAIL | — | FAIL |

## Verdict

**The routine/opportunistic + clustering refinement did not rescue the
insider-buying signal.** The cluster-flag IC is directionally larger than
the naive re-run's IC (+0.0138 vs +0.0043, both computed on the identical
universe/window) — weak evidence the refinement isolates *somewhat* more
signal than naive pooling, consistent with the literature's mechanism — but
it remains statistically indistinguishable from zero (t=0.85, n=13), the
P&L loses money net of costs (Sharpe −0.86), fails DSR at the true N=20, and
the sign-convention audit shows the cluster leg underperforming the
no-activity leg in a majority of quarters. Genuinely null-to-negative result,
though clouded by a real data-coverage limitation (only 13 of the intended 24
quarters have usable Form-4 data) that should be fixed before this family is
revisited. Logged as `insiderclusters_S13` regardless of outcome, per
instructions.
