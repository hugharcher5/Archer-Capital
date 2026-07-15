> **SUPERSEDED (2026-07-12) — see `research/CORRECTED_TRIAL_REGISTRY.md` for the
> current canonical registry.** This file's only trial (gpvolmanaged_S17, #23) has
> been merged there (Section 4, "S17 — Volatility-Managed GP/A"), including the
> full exposure-scaling-factor-by-period table, year-by-year table, and per-year
> beta table below, which are preserved here as the source detail. Do not add new
> trials to this file — log them to `CORRECTED_TRIAL_REGISTRY.md` instead. Before
> creating any new registry file, check whether one already exists.

# PHASE 3 TRIAL REGISTRY (SUPERSEDED)

This file is the narrative companion to `research/trial_registry.csv` for
trials run in the open-ended Phase 3 research phase (per the root `CLAUDE.md`).
It did not exist before this trial; it starts here rather than attempting to
backfill every historical trial (those are already covered in
`research/CORRECTED_TRIAL_REGISTRY.md` and are authoritative in
`research/trial_registry.csv`). Going forward, new Phase 3 trials should add a
dated section below.

**Note on trial numbering**: `trial_registry.csv` is written to concurrently by
multiple sessions. Always re-read its current max `trial_number` immediately
before logging a new trial, not just at task start — see the S13/#20 collision
incident (`feedback_dsr_trial_numbering` memory).

---

## 2026-07-10 — Volatility-Managed GP/A (trial #23, `gpvolmanaged_S17`)

### Hypothesis

Moreira & Muir (2017), "Volatility-Managed Portfolios": scaling a factor
portfolio's gross exposure inversely to its own trailing realized volatility —
de-risking in high-vol periods, re-risking in calm periods, targeting a
constant annualized volatility — improves Sharpe ratio for many equity factors
by avoiding the worst of volatility-clustering drawdowns, without altering the
underlying stock-selection signal. Tested here as a pure construction-level
overlay on top of S3-Q (Gross Profitability, trial #8) — same signal, same
quintile long-short, same beta-neutral leg sizing. This is **not** a re-tune of
GP/A; it is a genuinely new mechanism (position sizing over time, not stock
selection) and was pre-registered as an independent trial.

### Universe upgrade — what was actually achievable, and what wasn't

The task asked for "ACTUAL point-in-time Russell 2000 constituent membership,"
reusing the existing Wayback-archived FTSE Russell data in
`research/russell_reconstitution/`. Before building anything, this was checked
and found **not fully achievable**:

- The archive contains R3000-level **additions/deletions events per annual
  cycle** (2016–2023, 8 cycles — 2015 is explicitly unwired per that
  directory's own README, "a deliberate methodology decision, not something to
  graft on silently"), not an absolute full-membership snapshot at any date.
- It does not separately tag Russell 1000 vs Russell 2000 membership (only
  `r3000` and `rmicro`).
- No seed/full snapshot exists anywhere in the repo to cumulative-track true
  membership from. Both prior Russell trials (Russell-1, Russell-3) only ever
  used this data as *events*, never as a full constituent list.

Presented to the user as a blocking decision (not silently resolved). Approved
approach: **confirmed-exit overlay** — keep S3-Q's existing SEC/Tiingo +
$100M–$2B market-cap universe as the base, and additionally exclude any ticker
from the date it's confirmed deleted from the R3000. This adds real
point-in-time discipline where the archive supports it (confirmed exits) but
does **not** confirm true R2000 inclusion for the rest of the universe — it is
explicitly *not* "actual Russell 2000 membership," and is reported as such.

**Further disclosed gap**: R3000 deletions data itself has a hole for 2018 and
2019 (no deletions PDF was recoverable in the original Russell-1 acquisition).
Any ticker whose only true R3000 exit happened in those two years is **missed**
by this overlay and remains in the universe until any later recorded event.

**Window**: 2016-01-01 → 2024-01-01 (32 quarterly periods) — the closest
achievable ~8-year window given the archive's actual 2016–2023 coverage, not
the requested 2015–2023.

### Construction

- Base signal/portfolio: unchanged from S3-Q (`research/gp/run.py`), reused by
  import, not reimplemented.
- Vol-targeting overlay, **pre-registered before running**:
  - `TARGET_VOL = 10%` annualized
  - Trailing realized vol computed from the **unscaled** (beta-neutral-only)
    basket's own daily adjClose returns, up to 126 trading days (~6 months),
    minimum 60 days of history before scaling activates (first quarter,
    2016-01-01, has no prior history → `vol_scale = 1.0`, flagged).
  - `vol_scale = clip(TARGET_VOL / trailing_vol, 0.2, 3.0)` — clip bounds reused
    from this codebase's existing beta-scale convention, not a new invention.
  - Applied multiplicatively to **both** legs' scale factors (preserves the
    beta-neutral ratio; moves only the overall gross level).
- Two parallel series run on the **identical** window/universe to isolate the
  overlay's effect cleanly:
  - **BASE** — beta-neutral GP/A, Russell overlay, no vol-targeting (control)
  - **VOL-MANAGED (VM)** — BASE + vol-targeting overlay (treatment)
- Original S3-Q (trial #8, 2015–2021, no Russell overlay) reported separately
  as historical reference only — different window and universe, not a clean
  A/B against BASE/VM.
- **Disclosed cost-model simplification**: the standard cost model (per
  instructions, reused unmodified) charges transaction costs on the fraction of
  *names* that rotate in/out of a leg, not on the incremental notional traded
  when `vol_scale` itself changes gross exposure on unchanged names. This
  likely understates real-world cost drag for the vol-managed variant
  specifically (Moreira & Muir's own paper flags vol-timing as adding turnover)
  — not fixed here, flagged as an open item.

### Exposure-scaling factor by period

| Rebalance | Trail. vol | vol_scale | BASE ret | VM ret |
|---|---|---|---|---|
| 2016-01-01 | n/a | 1.00x | +1.98% | +1.98% |
| 2016-04-01 | 18.2% | 0.55x | −9.90% | −5.43% |
| 2016-07-01 | 16.5% | 0.61x | −6.94% | −4.22% |
| 2016-10-01 | 12.6% | 0.79x | −1.61% | −1.28% |
| 2017-01-01 | 11.3% | 0.88x | −3.04% | −2.68% |
| 2017-04-01 | 11.8% | 0.85x | +5.31% | +4.50% |
| 2017-07-01 | 11.1% | 0.90x | −3.15% | −2.85% |
| 2017-10-01 | 10.6% | 0.94x | +5.77% | +5.43% |
| 2018-01-01 | 10.7% | 0.93x | +9.29% | +8.65% |
| 2018-04-01 | 11.6% | 0.86x | +9.13% | +7.86% |
| 2018-07-01 | 12.4% | 0.81x | +12.26% | +9.92% |
| 2018-10-01 | 14.3% | 0.70x | +7.80% | +5.47% |
| 2019-01-01 | 16.7% | 0.60x | −1.25% | −0.75% |
| 2019-04-01 | 15.0% | 0.67x | −3.73% | −2.49% |
| 2019-07-01 | 11.6% | 0.86x | +2.01% | +1.73% |
| 2019-10-01 | 11.3% | 0.89x | −4.15% | −3.68% |
| **2020-01-01** | **11.2%** | **0.89x** | **−10.80%** | **−9.66%** |
| **2020-04-01** | **16.6%** | **0.60x** | **+7.03%** | **+4.24%** |
| **2020-07-01** | **19.1%** | **0.52x** | **+16.27%** | **+8.50%** |
| **2020-10-01** | **15.1%** | **0.66x** | **−8.81%** | **−5.82%** |
| 2021-01-01 | 12.1% | 0.83x | +16.06% | +13.27% |
| 2021-04-01 | 17.5% | 0.57x | +7.53% | +4.31% |
| 2021-07-01 | 18.8% | 0.53x | −0.43% | −0.23% |
| 2021-10-01 | 16.3% | 0.61x | +8.51% | +5.22% |
| 2022-01-01 | 17.3% | 0.58x | −11.30% | −6.51% |
| 2022-04-01 | 19.8% | 0.50x | +8.96% | +4.52% |
| 2022-07-01 | 21.0% | 0.48x | −6.59% | −3.14% |
| 2022-10-01 | 18.5% | 0.54x | +8.29% | +4.49% |
| 2023-01-01 | 15.5% | 0.65x | +2.38% | +1.54% |
| 2023-04-01 | 14.6% | 0.69x | −3.93% | −2.70% |
| 2023-07-01 | 15.3% | 0.65x | +4.23% | +2.77% |
| 2023-10-01 | 14.6% | 0.69x | −2.00% | −1.37% |

**vol_scale never once exceeds 1.0x across all 32 periods.** This quintile GP/A
long-short portfolio's baseline realized volatility is structurally well above
the 10% target throughout the whole window (min trailing vol observed: 10.6%),
so the overlay is a pure de-risking mechanism in this backtest, never a
re-risking one — the "increase exposure in calm periods" half of the mechanism
never actually triggers.

### Year-by-year: strategy vs. universe (EW) return

| Year | BASE | VM | EW-universe mkt | Mean vol_scale |
|---|---|---|---|---|
| 2016 | −15.87% | −8.81% | +26.25% | 0.74x |
| 2017 | +4.60% | +4.16% | +18.72% | 0.89x |
| 2018 | +44.34% | +35.87% | −10.36% | 0.83x |
| 2019 | −7.04% | −5.17% | +22.16% | 0.75x |
| **2020** | **+1.23%** | **−3.78%** | **+37.29%** | **0.67x** |
| 2021 | +34.84% | +24.02% | +17.95% | 0.64x |
| 2022 | −2.24% | −1.11% | −24.88% | 0.52x |
| 2023 | +0.47% | +0.14% | +13.10% | 0.67x |

### Realized beta

**Overall (32 periods, at scaled position size)**: VM beta = +0.056 (t = 0.92),
BASE beta = +0.058 (t = 0.67) — both statistically indistinguishable from zero,
confirming the beta-neutral construction holds under vol-scaling as designed
(scaling both legs proportionally preserves the neutrality ratio).

**Per-year (n=4 quarters/year — 2 degrees of freedom, t-stats reported but
essentially uninformative at this sample size, shown for completeness only)**:

| Year | Beta (VM) | t-stat |
|---|---|---|
| 2016 | −0.552 | −1.56 |
| 2017 | −3.264 | −1.21 |
| 2018 | +0.083 | +1.28 |
| 2019 | −0.117 | −0.99 |
| 2020 | +0.107 | +0.78 |
| 2021 | +0.327 | +3.70 |
| 2022 | −0.019 | −0.05 |
| 2023 | −0.200 | −1.44 |

### Full results table

| Metric | BASE (no overlay) | VOL-MANAGED | S3-Q original (ref. only, different window) |
|---|---|---|---|
| Window | 2016–2023 (32q) | 2016–2023 (32q) | 2015–2021 (24q) |
| Universe | Russell-overlay + $100M–2B | Russell-overlay + $100M–2B | $100M–2B, no Russell overlay |
| Ann. return (net) | +5.91% | +4.72% | +0.94% |
| Ann. return (gross) | +9.88% | +7.57% | +4.93% |
| Cost drag (bps/yr) | 375 | 273 | n/a (see original log) |
| Mean turnover/qtr | 21% | 21% | — |
| Sharpe (net, ann.) | 0.451 | 0.481 | 0.134 |
| CALMAR | 0.295 | 0.329 | 0.033 |
| Max drawdown | −20.0% | −14.3% | −28.3% |
| t-stat (mean qtly ret) | 1.28 | 1.36 | — |
| Market beta (t-stat) | +0.058 (0.67) | +0.056 (0.92) | −0.103 (−1.11) |
| Mean IC (Spearman) | 0.046 (same signal, unaffected by scaling) | | 0.040 |
| IC t-stat | 2.70 | | 2.28 |
| Win rate | 53.1% | 53.1% | 50.0% |
| N periods | 32 | 32 | 24 |
| Per-period SR | 0.226 | 0.241 | 0.067 |
| DSR threshold (N=23, this trial's slot) | 0.303 | 0.303 | n/a (N=8 at its own time) |
| **Clears DSR** | **NO** | **NO** | NO |

IC is computed on the unscaled signal (rank-return relationship is unaffected
by position sizing), so it is identical for BASE and VM by construction —
shown once.

### Falsification diagnostics

**Sign-convention audit**: mean IC = +0.046 (t = 2.70), positive — high GP/A
outperformed low GP/A, matching S3-Q's original direction. No sign flip
introduced by the universe or overlay changes. Long leg count == short leg
count in 100% of periods (53–78 names/leg across the window), confirming the
quintile construction stayed symmetric under the Russell overlay's added
exclusions.

**Concentration check**: equal-weight within each leg; minimum leg size
observed was 53 names → maximum single-name weight ≈ 1.9%, well within normal
bounds. No period required a fallback (`n<40` skip never triggered).

**2020 sub-period check** (the specific diagnostic this overlay was designed
for):

- The Q1 2020 crash quarter's trailing vol input was only 11.2% (not yet
  elevated — the vol spike hadn't shown up in the trailing window yet), so
  vol_scale was 0.89x, barely reduced. The overlay provided almost no
  protection during the crash quarter itself.
- The real de-risking kicked in **after** the crash, in Q2 and Q3 2020 (0.60x,
  0.52x) — the recovery-rally quarters, where BASE returned +7.03% and +16.27%
  but VM only captured +4.24% and +8.50%.
- Net effect: full-year 2020 cumulative return is *worse* under vol-management
  (−3.78% vs BASE's +1.23%) — the backward-looking trailing-vol estimator
  lagged the actual vol spike, missing the crash-quarter protection and instead
  throttling the recovery.
- However, looking at max-drawdown TROUGH depth specifically (not full-year
  return): the Q1 2020 trough was −17.1% for BASE vs −14.3% for VM — a real,
  if modest, reduction. The 2016–2017 drawdown episode (BASE's actual global
  max DD at −20.0%) also improved substantially under VM (−13.0%) — a larger
  proportional improvement than in the COVID episode.
- **Conclusion**: the overlay's overall Sharpe/CALMAR improvement is not really
  a "2020 crash-avoidance" story — 2020 alone got slightly worse on a
  full-year-return basis. The net improvement comes from milder drawdown
  reduction spread across multiple higher-vol stretches (most clearly the
  2016–2017 episode), traded off against reduced upside capture during
  recoveries.

### One-line verdict

**FAILS the Sharpe ≥ 0.8 / CALMAR ~1 target and FAILS DSR at N=23** (VM Sharpe
0.48, CALMAR 0.33 — a real but modest improvement over BASE's 0.45/0.30, driven
more by broad-based drawdown smoothing than by the COVID-specific crash
protection the hypothesis was framed around; the technique works roughly as
advertised directionally but nowhere near the magnitude needed to clear this
program's bar.

### Known limitations (carried into `trial_registry.csv`'s `notes` field)

1. "Russell overlay" = confirmed-exit only; does not confirm true inclusion,
   does not distinguish R1000/R2000. Not literal point-in-time R2000
   membership — see Universe Upgrade section above.
2. R3000 deletions data gap for 2018–2019 (pre-existing, inherited from
   Russell-1/Russell-3's original acquisition).
3. Cost model doesn't charge extra for vol-scale-driven resizing of unchanged
   positions — likely understates real cost drag for VM specifically.
4. Per-year realized beta has only 2 degrees of freedom; t-stats shown but not
   meaningful at this N.

### Files

- `research/gp_volmanaged/run.py` — backtest engine (imports `research/gp/run.py`
  for all unchanged base-signal logic; adds the Russell overlay and vol-scaling
  overlay only)
- `research/gp_volmanaged/finalize_and_log.py` — metrics, DSR check, registry
  logging (re-checks `trial_registry.csv`'s current max trial number
  immediately before writing)
- `research/gp_volmanaged/results/gp_volmanaged_period_returns.csv` — full
  32-period record (BASE and VM series side by side, vol_scale, trailing_vol,
  universe counts, betas, turnover, IC, everything needed to reproduce every
  table above)
- `research/gp_volmanaged/results/final_summary.json` — metrics snapshot

**Streamlit UI not updated** (per instructions) — two other related trials
(industry-neutral GP/A, multi-sleeve blend) are running in parallel; the
"Phase 3 — Signal Research Registry" tab update is deferred until all three are
complete, to avoid colliding partial edits.
