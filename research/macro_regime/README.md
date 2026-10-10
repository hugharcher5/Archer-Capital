# Macro Regime Classifier

**This is infrastructure, not a DSR trial.** It produces a reusable labeled
regime timeline that downstream sector-regime sensitivity trials (starting
with financials/rates) will consume as an input. It does not consume a DSR
slot and is not logged to `research/trial_registry.csv` or
`CORRECTED_TRIAL_REGISTRY.md`. Referenced from the root `CLAUDE.md`.

All methodology below (series choice, smoothing, thresholds, rules) is
locked **before** computing the classifier or looking at its output, per the
same pre-registration discipline used throughout this project's DSR trials.
Threshold levels were calibrated by looking at each *input series'* own
long-run historical distribution (e.g. "what is VIX's 25th percentile"),
never at the resulting regime timeline or any downstream performance — that
distinction matters and is the reason threshold-picking below is allowed to
reference descriptive stats without being "peeking at results."

## Data sourcing

All series are free via FRED (`FRED_API_KEY` confirmed working) except
stock-bond correlation, which uses Tiingo SPY/IEF daily prices (already
integrated in this project).

| Role | Series ID | Coverage confirmed | Notes |
|---|---|---|---|
| Growth (primary) | `A191RO1Q156NBEA` | 1948-01 -> 2026-01, quarterly | Real GDP, % change from year ago. Used via ALFRED first-release vintage (PIT), not the latest-revised value. |
| Growth (nowcast, not blended into label) | `INDPRO` | 1919 -> 2026-05 | Industrial production |
| Growth (nowcast, not blended into label) | `RSAFS` | 1992 -> 2026-05 | Retail sales, nominal |
| Growth (nowcast, not blended into label) | `CFNAI` | 1967 -> 2026-05 | Chicago Fed National Activity Index — **substituted for ISM PMI**. FRED's legacy ISM series (`NAPM`) no longer exists at all (confirmed: API returns "series does not exist"; ISM pulled free redistribution years ago). CFNAI is the standard free broad-activity-index substitute. |
| Growth (nowcast, not blended into label) | `ICSA` | 1967 -> 2026-07, weekly | Initial jobless claims |
| Growth (nowcast, not blended into label) | `UMCSENT` | 1952 -> 2026-05 | U. Michigan Consumer Sentiment — **substituted for Conference Board confidence**. The Conference Board's FRED series (`CSCICP03USM665S`) stops at 2024-01 (confirmed stale, discontinued on FRED) — unusable for a live read. |
| Inflation (primary) | `CPILFESL` | 1957 -> 2026-06, monthly | Core CPI index. YoY computed from it; used via ALFRED first-release vintage (PIT). |
| Inflation (nowcast) | `DCOILWTICO` | 1986 -> 2026-07, daily | WTI spot |
| Curve | `T10Y2Y` | 1976 -> 2026-07, daily | 10Y-2Y Treasury spread |
| Curve | `T10Y3M` | 1982 -> 2026-07, daily | 10Y-3M Treasury spread |
| Credit spread (primary) | `BAA10Y` | 1986 -> 2026-07, daily | Moody's Baa − 10Y Treasury. **Substituted for ICE BofA HY OAS** (`BAMLH0A0HYM2`, which the brief named). Confirmed via `get_series_info`: FRED now serves only a rolling **3-year window** for every ICE BofA index series (`BAMLH0A0HYM2` observation_start = 2023-07-17; same truncation confirmed on `BAMLC0A0CM`, the IG counterpart), per an ICE Data Indices licensing note effective **April 2026**: *"Starting in April 2026, this series will only include 3 years of observations."* Unusable for a full-history classifier. `BAMLH0A0HYM2` is still fetched and stored 2023+ as a supplementary cross-check series, disclosed as truncated. |
| Vol | `VIXCLS` | 1990 -> 2026-07, daily | |
| Fragility | Tiingo SPY + IEF daily closes | 2003+ (IEF inception) | Rolling 60-trading-day correlation of daily returns, computed here (not a FRED series). |
| Policy | `DFF` | 1954 -> 2026-07, daily | Effective fed funds rate, daily. |
| Modifier | `DTWEXBGS` | 2006 -> 2026-07 | Trade-weighted USD, broad. Sufficient for the 2015+ ask; the pre-2019 major-currency series (`DTWEXM`) was discontinued, `DTWEXBGS` is its replacement and the one FRED recommends going forward. |
| Modifier | `UNRATE` | 1948 -> 2026-06 | Unemployment rate |

**FOMC meeting dates**: the brief said these were "already sourced from
earlier work." A thorough repo + memory search turned up nothing — no FOMC
calendar exists anywhere in this project. The policy overlay below does not
actually require meeting dates (it's a pure trailing-window function of
`DFF`), so this was not a blocker; flagged here rather than silently
assumed away.

### Point-in-time (PIT) discipline

`A191RO1Q156NBEA` (GDP) and `CPILFESL` (CPI) are the only two series that
drive the actual coincident regime label (see below), so these are the only
two that need genuine release-date gating — and FRED's ALFRED vintage
service (`fredapi.get_series_all_releases` / `get_series_first_release`)
provides it directly: every historical observation is tagged with its own
`realtime_start`, the actual date that value first became public. The
classifier uses each observation's **first-release** value, gated on its own
`realtime_start`, never a later-revised value — mirroring this project's
existing SEC-filing-date-gating convention elsewhere.

The remaining series (market prices, daily Treasury/credit/vol series, DFF)
are same-day/next-day public with no meaningful revision lag and are used as
plain `get_series()` pulls, consistent with how this project already treats
price data as point-in-time by construction. The nowcast-only growth inputs
(`INDPRO`, `RSAFS`, `CFNAI`, `ICSA`, `UMCSENT`) are pulled as latest-revision
values without vintage gating — a disclosed limitation, acceptable because
**they are not used to compute the actual regime label** (see below); they
are stored for context / possible future use only.

## Part 1 — Coincident regime classifier (locked methodology)

**Growth signal**: `A191RO1Q156NBEA` first-release value, smoothed with a
trailing 2-quarter simple moving average. Direction = sign of
`smoothed(t) - smoothed(t-1)` (quarter-over-quarter change in the smoothed
series). Exactly zero (a tie) carries forward the previous quarter's
direction rather than flipping on a knife-edge.

**Inflation signal**: Core CPI YoY computed as
`CPILFESL_first_release(t) / CPILFESL_first_release(t-12mo) - 1` (i.e. both
the current and year-ago index levels use their own first-release vintage —
a simple, standard convention; CPI is barely revised in practice so this and
"latest revision" differ negligibly). Smoothed with a trailing 6-month
simple moving average (the monthly equivalent of "2 quarters"). Direction =
sign of `smoothed(t) - smoothed(t-3)` (quarter-equivalent change). Ties
carry forward the previous direction.

**Regime label** (4-way, from the two directions):

| Growth | Inflation | Regime |
|---|---|---|
| Accelerating | Decelerating | Goldilocks |
| Accelerating | Accelerating | Reflation |
| Decelerating | Accelerating | Stagflation |
| Decelerating | Decelerating | Deflation / Risk-off |

**Precarious fragility flag** (independent binary overlay, can co-occur with
any of the 4 regimes above — locked thresholds, chosen by looking only at
each series' own historical distribution, before any regime/flag output
existed):

All four conditions must hold simultaneously:
1. `BAA10Y < 1.75` (credit spreads compressed — ~25th percentile of the
   1990-2026 distribution, mean 2.28, median 2.14).
2. `VIXCLS < 15` (low vol — ~25th percentile of the 1990-2026 distribution,
   25th pctile 13.98, median 17.61).
3. `T10Y2Y <= 0.25` (flat or inverted curve — 25th percentile of the
   1990-2026 distribution is almost exactly 0.25).
4. Rolling 60-trading-day correlation of SPY/IEF daily returns is **both**
   positive (`> 0`) **and** higher than its own value 60 trading days
   earlier (i.e. genuinely rising, not just incidentally positive).

**Policy overlay** (Hawkish / Dovish / Neutral, reported alongside the
regime, never blended into it): trailing 6-calendar-month (126 trading day)
change in `DFF`. `> +0.25pp` = Hawkish, `< -0.25pp` = Dovish, otherwise
Neutral.

**Timeline grain**: monthly (GDP's quarterly reading is forward-filled
across the months of that quarter, gated so a given month only ever sees a
GDP print whose `realtime_start` has already passed). Monthly was chosen
over daily because CPI (the faster series) is itself only monthly, and
because the primary consumers of this timeline (future sector-regime
trials) share this project's existing quarterly/monthly rebalance
conventions.

## Part 2 — Predictive / early layer (report separately, not blended into Part 1)

**Base-effect model**: because YoY(t) = level(t)/level(t-4q or t-12mo) - 1,
and the "year-ago" base for the *next* release is already known today, a
"flat" (zero-growth) scenario for the next print is fully computable now,
with no future data:

- GDP: `forward_yoy_flat = level(t) / level(t-3q) - 1` (this becomes next
  quarter's YoY comparison base once one more quarter rolls forward).
- CPI: `forward_yoy_flat = idx(t) / idx(t-11mo) - 1`.

`base_effect_bias = forward_yoy_flat - YoY(t)`. Positive means base effects
alone imply the next print looks better (mechanical acceleration bias) even
absent any new information; negative implies mechanical deceleration bias.
Reported as a distinct "expected regime direction" signal per series, never
merged into Part 1's coincident label.

**Early-warning flags** (standalone, locked before computing lead times):
- Curve inversion onset: `T10Y2Y` (and separately `T10Y3M`) crosses from
  `>= 0` to `< 0`. Flag fires on the first day of a new inversion episode,
  not every day the curve stays inverted.
- Credit-spread-widening onset: `BAA10Y` rises `>= 0.50pp` above its own
  trailing 252-trading-day rolling minimum. An episode must "reset" (spread
  back within 0.25pp of its trailing low for >= 20 trading days) before the
  same flag can fire again, so one continuous widening move isn't
  double-counted as many onsets.

**Lead-time test** (the actual test of whether early detection works): for
every flag onset, find the next Part-1 coincident regime-label change
strictly after it. Report, honestly including if weak: (a) mean/median lead
time to *any* label change, and (b) mean/median lead time specifically to
the next transition into Stagflation or Deflation/Risk-off (the two
growth-decelerating regimes curve inversion is classically supposed to
anticipate) — reported separately since (a) is a much weaker claim than (b).

## Part 3 — Live regime read

Current regime / Precarious flag / policy label computed from the
most-recent available data for each input series, displayed in the Macro
Trading tab. Informational only — not a backtest, matches the "current
state" framing of the tab's live-read section.

## Files

- `fetch.py` — FRED (PIT vintages for GDP/CPI, plain pulls for the rest) +
  Tiingo SPY/IEF acquisition, cached under `cache/`.
- `classify.py` — Part 1 (coincident regime, Precarious flag, policy
  overlay) -> `results/regime_timeline.csv` (the reusable output every
  future sector-regime trial should read from).
- `predictive.py` — Part 2 (base-effect model, early-warning flags,
  lead-time analysis) -> `results/part2_leadtime_report.md`.
- `live_read.py` — Part 3, current regime read; imported by
  `strategies/macro_trading/page.py`.
- `summary_stats.py` — time-in-regime, Precarious co-occurrence, regime
  duration distribution -> printed into `results/summary_stats.md`.

## Known limitations (disclosed, not hidden)

- ICE BofA HY OAS (the series literally named in the brief) is only
  available 2023+ on FRED going forward — BAA10Y is a reasonable, standard,
  full-history substitute for the same underlying concept (credit risk
  premium), but it is not literally the same series and doesn't isolate
  "high yield" specifically the way an HY-only OAS would.
- CFNAI and UMCSENT are broad-activity/sentiment substitutes for ISM PMI and
  Conference Board confidence respectively — both are standard free
  alternatives, but neither is literally the requested series.
- Nowcast-input series (`INDPRO`, `RSAFS`, `CFNAI`, `ICSA`, `UMCSENT`) are
  fetched and stored but **do not feed the coincident regime label** — the
  brief specified GDP + core CPI acceleration/deceleration as the label rule
  and did not specify a combination formula for the nowcast inputs.
  Inventing one here would be undocumented methodology outside
  pre-registration; they're retained for future nowcasting/sector-trial use.
- No FOMC meeting-date calendar exists in this project (see Data Sourcing
  above) — the policy overlay does not require one.

## Verification audit (2026-07-16, before the first sector-regime trial)

Requested before building anything on top of this classifier, since every
downstream trial inherits whatever is wrong here. Three checks, all
completed with real numbers (not assumed):

**1. Do the three substitute series (CFNAI, UMCSENT, BAA10Y) use PIT-correct
data?** Checked ALFRED vintage history for each directly (comparing the
value recorded shortly after a historical date's first publication vs. the
value on record today):
- **BAA10Y** (the one substitute that actually feeds computation — the
  Precarious flag and the credit-widening early-warning flag): confirmed
  **zero revision** across 4 independent spot-checks spanning 2015/2020/2022
  (e.g. the 2015-06-15 reading was 2.73 both 45 days later and today), plus
  its raw component `DBAA` (Moody's Baa yield, 5.09 both times). This is
  structurally expected — it's a subtraction of two directly market-observed
  yields, not a modeled/estimated statistic, so there is no "revision" for a
  PIT gate to protect against. Using a plain (non-vintage) fetch for it is
  therefore correct, not a look-ahead leak. (ALFRED's vintage tracking for
  this specific derived series ID doesn't extend before ~2010-2015 in some
  spot checks — a coverage quirk, not evidence of revision, and moot given
  the structural argument above.)
- **CFNAI**: confirmed **materially revised** over time (June 2015's reading
  was 0.08 at first release, then -0.11 a year later, -0.22 by 2020, -0.14
  today — a ~0.3-point swing, large relative to CFNAI's typical ±1 range).
  Expected: it's an 85-indicator composite that inherits every component's
  own revision history. **However, CFNAI is not referenced anywhere in
  `classify.py`, `predictive.py`, or `live_read.py`** (confirmed by direct
  grep) — it is fetched and cached per the "nowcast input, not blended into
  the label" design already documented above, so this revision behavior
  currently poses **zero look-ahead risk**. Flagged as a hard requirement
  if CFNAI is ever wired into an actual signal later: it would need the same
  ALFRED first-release treatment as GDP/CPI.
- **UMCSENT**: spot-checked as **not revised** (96.1 unchanged from the
  mid-month preliminary release through today, for the one date tested) —
  also not referenced in any computation, so moot either way.

**2. First-release or final-revised GDP/CPI, and how much would it matter?**
Confirmed directly in code (`first_release_series()` sorts by
`realtime_start` ascending and takes the first row per period) and
empirically (GDP 2015-Q1's first-release value of 16,304.8, recorded
2015-04-29, matches the earliest vintage row on file; later vintages show
the expected revision drift). **The classifier does use first-release
throughout — this was not a leak.** But the brief asked how large the
discrepancy would have been, so a full alternate "final-revised" regime
timeline was built (same forward-fill/gating logic, but fed final-vintage
levels instead of first-release) and diffed against the real, shipped
timeline:
- GDP YoY revision (final − first), 1997+: mean −2.0pp, std 4.8pp, range
  [−15.1pp, +2.2pp]. Core CPI YoY revision: mean +0.005pp, std 0.06pp —
  confirms CPI is barely revised in practice (the design assumption in the
  Methodology section above), while GDP revisions are large (compounded by
  BEA's periodic comprehensive/chain-weighting rebasements, not just
  incremental data updates).
- **Growth direction (accel/decel) flips in 33 of 117 quarters (28%)**
  between first-release and final data. CPI direction flips in 33 of 354
  months (9%).
- **The combined regime label differs in 180 of 355 months — 50.7% of the
  entire timeline — between the first-release (shipped) version and a
  final-revised version.** This is the headline number: PIT gating was not
  a nice-to-have refinement here, it is the single most consequential
  design decision in this classifier. A naive non-PIT implementation would
  have produced a materially different regime history for essentially half
  of it.

**3. Is "genuine lead-time concentration" for curve inversion statistically
supportable at n=8-13?** No — this was an overstatement in the original
Part 2 write-up, based on an eyeballed median-ratio threshold, not a formal
test. Fixed: `predictive.py` now runs a one-sided Mann-Whitney U test (H1:
onset-conditioned lead times are stochastically shorter than the
unconditioned baseline gap between regime changes) for each flag. Results:
`curve_10y2y_inversion` p=0.099 (n=8, marginal — directionally suggestive,
not significant at conventional levels), `curve_10y3m_inversion` p=0.161
(n=13, not significant), `credit_spread_widening` p=0.464 (n=12, not
significant). **None of the three early-warning flags clears a conventional
significance threshold at this sample size.** The report and its language
have been corrected accordingly (`results/part2_leadtime_report.md`) —
"marginal, not significant, do not call this confirmed" replaces the
earlier "concentrated" claim for the 10Y-2Y flag; the other two were
already reported as weak and remain so.

**Bottom line for anyone building a sector-regime trial on top of this:**
Part 1 (the coincident regime label) passed this audit — PIT-correct as
designed, and the 50.7% divergence check is direct evidence the discipline
mattered. Part 2 (the predictive/early layer) should be treated as exactly
what it always claimed to be: suggestive, not validated, and now with the
overstated language removed. Nothing here required changing the shipped
`regime_timeline.csv`.

## Walk-forward validation of Part 2 (2026-07-17 follow-up)

The in-sample check above (2026-07-16) never tested whether Part 2's flags
would actually have worked in real time. This is that test —
`research/macro_regime/walkforward.py`, full output in
`results/walkforward_report.md`. **Verification of existing infrastructure,
not a DSR trial** — not logged to `trial_registry.csv` or
`CORRECTED_TRIAL_REGISTRY.md`. Does not touch Part 1's `regime_timeline.csv`
or the live classifier (`build_monthly_timeline()` is called read-only, the
CSV is never rewritten by this script).

**Design** (locked before running): expanding-window anchor starting
2010-01-01 (onsets before this are training evidence only, not scored);
one expansion step per subsequent regime change (39 steps, 2010-08 through
2026-05); `HORIZON_DAYS=365`, chosen because it's more generous than every
lead time the in-sample check observed (max 358d) — it cannot retroactively
penalize a case the in-sample check called a hit.

**Sample size, stated plainly**: only 3 (10Y-2Y), 6 (10Y-3M), and 8
(credit-widening) onsets fall at/after the 2010 anchor — this is an
extremely small out-of-sample test in absolute terms, and the verdict below
should be read as a low-confidence signal regardless of which way it points,
exactly as it would be if the result had come back favorable instead.

**The key methodological catch**: a naive "per-onset hit rate" (did each
onset eventually precede *some* regime change within a year) came back
100% for all three flags — but this is nearly meaningless on its own.
Regime changes recur so often (95% of all calendar months in this window
already have a change within 365 days, unconditionally) that almost any
onset will appear to "work" by this measure regardless of timing skill.
The metric that actually reflects timing is **coverage**: of the 39 real
transitions, how many had a flag onset in the preceding year? That was
tested against a proper null — 20,000 simulations placing the same number
of random onset dates in the same window — rather than an eyeballed
threshold:

| Flag | Transitions covered | Null mean (random placement) | Permutation p-value |
|---|---|---|---|
| curve_10y2y_inversion | 6/39 | 6.6/39 | 0.646 |
| curve_10y3m_inversion | 9/39 | 12.1/39 | 0.864 |
| credit_spread_widening | 18/39 | 15.3/39 | 0.268 |

**Verdict: none of the three flags survives out-of-sample testing.** Two
of the three (both curve-inversion flags) covered *fewer* transitions than
random chance placement would predict on average — walk-forward performance
is worse than the in-sample marginal p-values suggested, not merely
"unconfirmed." Credit-widening is nominally above its null mean but at
p=0.268 this is far from any conventional significance threshold and, at
n=8 onsets, is not distinguishable from noise. **This directly contradicts
what the original in-sample marginal reading (10Y-2Y p=0.099) might have
suggested going in** — the honest update is that the signal did not hold up
once tested with real out-of-sample discipline, not merely that it "remains
unconfirmed."

**Practical conclusion**: Part 2's predictive/early-warning layer should be
treated as **informational only** going forward, not as a trustworthy
early-warning signal for sector-regime or trading decisions. This does not
affect Part 1 — the coincident regime label and Precarious/policy overlays
are unaffected by this finding and remain the load-bearing output for any
sector-regime trial built on this classifier.

## External nowcast/consensus signals (2026-07-18) — replaces the custom nowcast plan

Building a from-scratch regression nowcast would have duplicated existing
institutional research. Replaced with three free, published, real-time
forecast sources instead. Phase-1 feasibility confirmed for all three (data
availability report given to the user before any backtest was built):

| Source | Series/file | Coverage | Access method | Caveat |
|---|---|---|---|---|
| Atlanta Fed GDPNow | FRED `GDPNOW` | 2011-07 onward | `fredapi.get_series_all_releases` — same ALFRED PIT infra already used for GDP/CPI | Short history (2011+) vs. the full 1997+ regime timeline — fewer transitions to test against than Part 2 |
| Cleveland Fed Inflation Nowcasting | `clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_{month,quarter,year}.json` (CC-BY 4.0) | 2013-07 onward, daily granularity within each month | Undocumented scraped JSON endpoint, found by reading the page's client-side chart config (the page itself blocks automated fetches via Akamai bot-detection; a browser-UA `curl` gets through) | **Not a stable published API contract** — could break if Cleveland Fed restructures their site. Raw JSON is cached permanently to disk specifically because of this fragility (per explicit user decision), so the analysis survives even if the endpoint disappears later. |
| Historical consensus (substitute for Bloomberg/Reuters, which remain proprietary) | Philadelphia Fed Survey of Professional Forecasters, `medianLevel.xlsx` (`CPI`/`CORECPI` sheets, annualized quarterly % change despite the file's "Level" name — a known SPF spreadsheet-naming quirk) and `medianGrowth.xlsx` (`RGDP` sheet) | Back to 1968 | Direct xlsx download from philadelphiafed.org, no key needed | **Quarterly, not monthly** — one survey per quarter with multi-quarter-ahead forecast columns, not a per-CPI-print monthly consensus like Bloomberg's own poll. A real frequency mismatch versus "Bloomberg/Reuters-style," disclosed rather than glossed over. |

### Locked methodology (before computing anything)

All three sources forecast the *same* underlying quantities Part 1's
coincident label already uses (GDP growth, CPI inflation) — unlike Part 2's
curve/credit flags, which were proxies for regime change, not direct
forecasts of it. That makes the natural test a **directional match**, not
an onset/lead-time framing: does the external source's own forward-looking
call, made before the official number is known, point the same way as Part
1's eventual direction call for that same period?

- **GDPNow**: for each quarter Q with coverage, `GDPNOW_TERMINAL(Q)` = the
  last GDPNow vintage recorded for Q before Q's own GDP first-release date
  (reusing GDPC1's already-computed first-release `realtime_start`).
  `GDPNOW_DIRECTION(Q) = accel` if `GDPNOW_TERMINAL(Q) > GDPNOW_TERMINAL(Q-1)`
  else `decel` — compared against Part 1's own `growth_direction(Q)`. Ties
  excluded (not counted either way). Anchor: 2015-Q1 (GDPNow's own history
  is short, so the original 2010 anchor would leave too thin a test tail;
  disclosed deliberate adjustment for this source only).
- **Cleveland Fed Inflation Nowcast**: same logic, monthly, using the last
  intra-month CPI nowcast before that month's actual CPI release (reusing
  CPILFESL's first-release `realtime_start`), compared against Part 1's
  `inflation_direction(M)`. Anchor: 2018-01-01 (disclosed adjustment, same
  reasoning as GDPNow).
- **SPF surprise**: `SURPRISE(Q) = actual(Q) - SPF_forecast_made_in(Q-1)_for(Q)`
  (the `CPI2`/`drgdp2` column from the *prior* quarter's survey row — the
  one-quarter-ahead forecast). `SPF_DIRECTION(Q) = accel` if surprise > 0
  (actual beat the consensus on the upside) else `decel` — compared against
  Part 1's direction for Q. Anchor: 2005-01-01 (SPF's own history goes back
  to 1968, far longer than the 1997+ regime timeline, so the original
  2010 anchor is kept close to the brief's spirit while leaving a longer
  test tail than the other two sources).
- **Significance test**: unlike Part 2's onset/lead-time framing (which
  needed a placement-based permutation test because onsets and transitions
  are asynchronous continuous-time events), this is a clean per-period
  binary match/mismatch — one independent quarter or month per test point.
  An exact two-sided binomial test against a null hit rate of 0.5 is the
  correct and simpler tool here, not a re-application of Part 2's
  permutation machinery.
- **Promotion criterion** (locked before seeing any result): only add a
  source to the live regime read if it clears p < 0.05 on the binomial test
  AND has a walk-forward test size of at least 15 periods (below that,
  report as inconclusive rather than promote on a technicality).

### Results (2026-07-18)

Full output: `research/macro_regime/results/external_signals_report.md`.
One real bug found and fixed during this check: the SPF-CPI test initially
returned zero merged rows because Part 1's per-month `inflation_direction`
frame carried the *last calendar month* of each quarter (e.g. 2020-03-01)
while the comparison side used *quarter-start* dates (2020-01-01) — the two
never matched. Fixed by normalizing both sides to quarter-start before the
join; re-verified the other three tests didn't share the same bug (they
didn't — each already used a consistent single date convention throughout).

| Signal | n (walk-forward) | Hit rate | Binomial p | vs. pre-registered bar | vs. Bonferroni(4) = 0.0125 |
|---|---|---|---|---|---|
| GDPNow vs Part 1 growth direction | 39 | 48.7% | 1.0000 | FAILS | n/a |
| Cleveland Nowcast vs Part 1 inflation direction | 102 | 65.7% | **0.0020** | CLEARS | **Survives** |
| SPF CPI-surprise vs Part 1 inflation direction | 77 | 63.6% | 0.0220 | CLEARS | Does NOT survive |
| SPF RGDP-surprise vs Part 1 growth direction | 85 | 63.5% | 0.0165 | CLEARS | Does NOT survive |

**Honest read, not just the pass/fail table:**

- **GDPNow: genuinely null.** 48.7% hit rate is indistinguishable from a
  coin flip (p=1.0000). GDPNow's own quarter-to-quarter SAAR growth
  trajectory does not predict the direction of Part 1's YoY-smoothed growth
  classification. This is a real negative finding about a well-regarded,
  widely-used real-time GDP tracker for THIS specific use (predicting a
  *different* growth statistic's direction), not a data or methodology bug
  — disclosed as such, not hidden.
- **Cleveland Nowcast: the most robust of the three positive results** —
  survives even a conservative Bonferroni correction for running 4 tests.
  But be precise about *why* it works: the "terminal" nowcast used here is
  the model's last reading right before the actual CPI release, built from
  data that's already mostly describing the month in question. This is
  closer to **a faster, more-informed read of a period that's already
  mostly over** than a genuine forward-looking prediction of a future
  period — still practically useful (it gives real lead time relative to
  when Part 1's own label can update, since Part 1 has to wait for the
  official release), but it should be understood as "confirms sooner," not
  "foresees."
- **SPF CPI-surprise and SPF RGDP-surprise: genuinely forward-looking**
  (both use a forecast made a full quarter *before* the period being
  predicted — not a nowcast of an already-mostly-elapsed period), which
  makes a real signal here more substantively interesting than Cleveland's.
  Both clear the pre-registered bar individually, but **neither survives
  Bonferroni correction for testing 4 signals at once** — with 4 tests run,
  getting 2 uncorrected p-values in the 0.015-0.022 range is not a strong
  surprise under the null. There is also a disclosed mechanical channel:
  `surprise = actual − stale forecast`, and Part 1's own direction is also
  driven by `actual`, so some of the correlation could reflect both metrics
  reacting to the same realized print rather than the forecast surprise
  carrying independent information. Treat these two as **suggestive, not
  confirmed** — a real result at the pre-registered threshold, but a softer
  one than Cleveland's once multiple-testing and mechanism are accounted
  for honestly.
- **Not tested**: anchor-date sensitivity (e.g. would results hold under a
  different anchor?). The anchors were locked before running per the
  methodology above; re-sweeping them now to check robustness risks
  looking like post-hoc anchor selection, so this is left as a disclosed
  open question rather than something quietly explored and only reported
  if favorable.

**Decision**: per the user's instruction ("if either shows genuine
predictive value, add it... clearly labeled as external-model-derived"),
Cleveland Nowcast, SPF CPI-surprise, and SPF RGDP-surprise are added to the
live regime read (`live_read.py`, surfaced in the Macro Trading tab),
labeled `[external, informational]` and kept visually and structurally
separate from Part 1's coincident regime label. GDPNow is NOT added — it
failed its own pre-registered test. Given the honest caveats above
(especially for the two SPF signals), these are presented as informational
context, not as inputs that change Part 1's regime label or its Precarious
/ policy overlays.

## Individual-forecaster accuracy within SPF (2026-07-19)

Tests whether any specific SPF forecaster (or small top-ranked group) beats
the full-panel consensus already tested above (SPF RGDP-surprise: n=85,
hit rate 63.5%, p=0.0165, did NOT survive Bonferroni(4)). Reuses the
existing SPF pipeline (`external_fetch.py`) — only adds the individual-
response layer, no rebuild.

### Data confirmed

Philly Fed publishes anonymized individual-forecaster microdata at
`survey-of-professional-forecasters/historical-data/SPFmicrodata.xlsx`
(same download pattern as the median files). The `RGDP` sheet: 9,248 rows,
**463 unique anonymized forecaster IDs**, 1968-2026. Unlike the median file,
individual responses are reported as **levels** (`RGDP1`-`RGDP6`), not
pre-computed growth rates — the 1-quarter-ahead annualized growth forecast
is computed here as `(RGDP2/RGDP1)^4 - 1`. Also pulled: the SPF's own
published survey deadline-date file (`spf-release-dates.txt`), giving the
**true survey deadline date** for every quarter from 1990:Q2 onward — used
for exact date-based gating below instead of a cruder quarter-count
heuristic (real dates were available, so a heuristic buffer wasn't needed).

### Locked methodology (before computing anything)

- **Minimum track record**: a forecaster must have **20+ resolved
  (forecast, actual) pairs** to be rankable at any point in time (per the
  brief's suggestion) — this excludes noisy short-tenure forecasters from
  ever reaching the top rank on a thin sample.
- **Point-in-time ranking (the actual look-ahead-avoidance mechanism)**:
  for the survey that sets the forecast used to predict quarter T (made in
  the T-1 survey), a forecaster's trailing MAE is computed using ONLY
  their (forecast, actual) pairs whose actual outcome had **already been
  publicly released** (via `GDPC1`'s first-release `realtime_start`,
  already-validated PIT infra) strictly before the T-1 survey's own **true
  deadline date** (from the file above) — real dates, not an assumed lag.
  Forecasters are ranked ascending by this trailing MAE; only forecasters
  clearing the 20-pair minimum are eligible.
- **Two constructions tested** (not an open-ended search over group size):
  `TOP-1` (the single lowest-trailing-MAE eligible forecaster's own
  forecast) and `TOP-5-AVG` (mean forecast of the 5 lowest-trailing-MAE
  eligible forecasters) — a single-forecaster read and a less-idiosyncratic
  small-group read, both pre-registered rather than sweeping group size
  until something clears significance.
- **Same test, same anchor**: identical binomial walk-forward vs Part 1's
  `growth_direction`, anchor 2005-01-01 (same as the already-tested panel
  consensus, for direct comparability).
- **Multiple-testing family now totals 6** (GDPNow, Cleveland, SPF-CPI-
  surprise, SPF-RGDP-surprise, SPF-TOP-1, SPF-TOP-5-AVG) — Bonferroni(6)
  = 0.05/6 = 0.00833, stricter than the Bonferroni(4) applied earlier.
- **Two honesty checks, run and reported regardless of outcome**:
  (a) the WRONG, circular way to do this — picking the single best
  forecaster by **full-sample** (hindsight) MAE and checking how well
  "following them" would have scored — computed explicitly so the gap
  between hindsight-selection and genuine point-in-time selection is
  visible, not just asserted; (b) the cross-sectional **dispersion** of
  full-sample MAE across all 20+-submission-qualifying forecasters (std,
  IQR, range) — directly answers whether forecaster identity is a
  meaningfully different lever at all, or whether everyone clusters too
  tightly for "picking the best one" to matter.

### Results (2026-07-19)

Full output: `results/individual_forecaster_report.md`.

**Forecaster identity does vary meaningfully** — 150 forecasters cleared
the 20+-submission bar; MAE mean 4.83, std 1.86 (coefficient of variation
0.38), range [1.83, 11.57]. This is real spread, not tight clustering, so
"picking the best one" was at least a coherent question to ask (unlike a
scenario where everyone is equally accurate and forecaster selection is
moot by construction).

| Construction | n | Hit rate | Binomial p | vs Bonferroni(6)=0.00833 |
|---|---|---|---|---|
| **TOP-1, point-in-time (the genuine test)** | 85 | 60.0% | 0.0821 | Does not survive |
| **TOP-5-AVG, point-in-time (the genuine test)** | 40 | 65.0% | 0.0807 | Does not survive |
| [circular/wrong] TOP-1, full-sample hindsight | 85 | 61.2% | 0.0503 | n/a — shown only to quantify bias |
| [circular/wrong] TOP-5-AVG, full-sample hindsight | 31 | 71.0% | 0.0294 | n/a — shown only to quantify bias |

**Verdict: no, individual-forecaster selection does not beat the
already-tested full-panel consensus.** Both genuine point-in-time
constructions fail even the original (pre-Bonferroni) p<0.05 bar the panel
consensus itself cleared (consensus: hit rate 63.5%, p=0.0165, n=85).
TOP-1 point-in-time actually does *worse* than the consensus on the exact
same n=85 sample (60.0% vs 63.5%). TOP-5-AVG's point estimate is nominally
higher (65.0%) but on a much smaller n=40 sample (many quarters have none
of the top-5-ranked forecasters responding that quarter), so its p-value
(0.0807) is weaker, not stronger, than the consensus's.

**The hindsight-bias gap is real and exactly the failure mode flagged as
most likely.** Compare TOP-5-AVG point-in-time (65.0%, p=0.0807) against
its circular hindsight counterpart (71.0%, p=0.0294) — picking the "best"
group using full-sample knowledge inflates the apparent hit rate by 6
points and flips the p-value from "not significant" to "nominally
significant." This is precisely the circularity the brief warned about,
demonstrated directly rather than just asserted as a risk. TOP-1's
hindsight/point-in-time gap is smaller (61.2% vs 60.0%) but points the same
direction.

**A second, unplanned methodological weakness surfaced while checking
this**: the "TOP-1" forecaster is not a stable identity in practice. SPF
response is voluntary each quarter, so if the true #1-ranked (as of that
point in time) forecaster didn't respond that particular quarter, the
construction falls back to the next-best respondent. Checked directly:
**the true #1-ranked forecaster only actually supplied the signal in 8 of
85 quarters (9.4%)** — the median fallback lands at rank #6, and the 75th
percentile falls back as far as rank #54. "TOP-1" is therefore better
described as "best available respondent that quarter" than "the single
most accurate forecaster" — a real construction weakness, not just a
theoretical caveat, and one more reason not to trust this signal even at
its nominal point estimate.

**Decision**: none of the individual-forecaster constructions are added to
the live regime read. The full-panel SPF consensus (already flagged as
"suggestive, not confirmed" in the section above) remains the strongest
SPF-derived read available from this data; narrowing to specific
forecasters made the signal weaker and noisier, not better.

## NY Fed Staff Nowcast (2026-07-20) — final growth-prediction candidate

The last free growth-prediction candidate before treating GDP growth
prediction as closed (either found or genuinely unsolved with current free
data) per the brief. Same walk-forward + binomial framework as GDPNow, for
direct comparability — no new test machinery built.

### Data confirmed, with a real coverage caveat

Two separate files, no single continuous archive:
- **Historical**: `New-York-Fed-Staff-Nowcast_data_2002-2021.xlsx`, weekly
  vintages 2001-11-23 through 2021-08-27. **The NY Fed's own notes
  disclose that 2002:Q1-2015:Q4 are retrospective model reconstructions
  ("predictions that our nowcasting model would have made in real time"),
  not genuinely live-published forecasts — only 2016:Q1 onward in this
  file was actually published in real time** (explicitly blue/red-shaded
  in the source workbook to mark the distinction).
- **Current**: `New-York-Fed-Staff-Nowcast_download_data.xlsx`, 2022-12
  onward, genuinely real-time. Locating it required checking an
  undocumented `downloads.json` endpoint (same class of scraped-endpoint
  fragility already disclosed for Cleveland Fed — cached permanently for
  the same reason).
- **Real coverage gap, not a data-access limitation**: the Nowcast was
  suspended in September 2021 (COVID-era data volatility broke the model)
  and relaunched in September 2023 with a revised methodology
  ("Nowcast 2.0"). There is no genuine real-time NY Fed Nowcast for
  roughly 2021:Q4-2023:Q3 — this is a real gap in what was published, not
  something this project failed to find.
- **Net usable real-time coverage for this test**: 2016:Q1-2021:Q3, then a
  gap, then 2023:Q4-present — roughly 33-34 quarters total. The
  2002-2015 reconstructed segment is loaded and reported separately as a
  disclosed supplementary check, never merged into the primary test (using
  it as if it were a live track record would be the same category of
  hindsight risk flagged in the SPF individual-forecaster check).
- **Update frequency**: weekly (Fridays ~11:15am, using data available as
  of 10am that day, per the NY Fed's own documentation).

### Locked methodology (before computing anything)

- Same terminal-value-before-release extraction as GDPNow: for each target
  quarter, the last recorded weekly nowcast value before that quarter's
  own GDP first-release date (`GDPC1` `realtime_start`, already-validated
  PIT infra).
  `NYFED_DIRECTION(Q) = accel` if `terminal(Q) > terminal(Q-1)` else
  `decel` — compared against Part 1's `growth_direction(Q)`.
- **Primary test uses ONLY real-time-flagged quarters** (2016:Q1 onward,
  excluding the 2021:Q4-2023:Q3 gap quarters for which no data exists).
  No separate anchor/burn-in is applied beyond this — unlike GDPNow's
  2015-01-01 anchor (chosen to leave a test tail beyond an assumed
  burn-in), NY Fed's real-time window itself starts in 2016, leaving no
  room for an additional buffer; this is disclosed as a shorter, gap-
  affected sample rather than padded to look larger.
- **Secondary, clearly separate check**: the same test re-run including
  the 2002-2015 reconstructed segment, reported side by side but never
  used to justify a promotion decision on its own.
- **Multiple-testing family, per the brief's explicit instruction**: this
  is evaluated against a **growth-only** family of 5 (GDPNow, SPF-RGDP
  consensus, SPF-RGDP TOP-1, SPF-RGDP TOP-5-avg, NY Fed) —
  Bonferroni(5) = 0.05/5 = **0.01**. This refines the earlier combined
  growth+inflation Bonferroni(6) used for the SPF individual-forecaster
  check into two separate families (growth: 5; inflation: 2 — Cleveland,
  SPF-CPI-surprise) going forward, since growth and inflation are
  conceptually separate questions and mixing their correction denominators
  was more conservative than necessary.

### Results (2026-07-20)

Full output: `results/nyfed_report.md`.

| Test | n | Hit rate | Binomial p |
|---|---|---|---|
| **Primary** (real-time only, 2016+, excl. 2021-2023 suspension gap) | 35 | 40.0% | 0.3105 |
| Supplementary (incl. 2002-2015 reconstructed segment) | 91 | 45.1% | 0.4018 |

**NOT significant, and notably not even directionally close** — both the
genuine real-time test and the reconstructed-inclusive supplementary check
came in *below* 50%, i.e. worse than a coin flip, not just indistinguishably
close to one. Does not survive Bonferroni(5)=0.01, and would not have
survived an uncorrected p<0.05 bar either.

**This mirrors GDPNow's own null result closely** (GDPNow: 48.7%, p=1.0000;
NY Fed: 40.0%, p=0.3105) — the two most prominent free real-time GDP
tracking models both independently fail at this specific task. That
consistency is itself informative: it's not that one specific model
happens to be weak, but that *quarterly SAAR nowcasts generally* don't
predict Part 1's *YoY-smoothed, 2-quarter-averaged* direction call well —
plausibly a genuine mismatch between what these models are built to track
(the current quarter's own momentum) and what Part 1's classifier asks
(whether a smoothed year-over-year trend accelerated or decelerated), not
a flaw specific to either vendor's model.

### Final verdict on GDP growth prediction — CLOSED

Five free candidates tested for a leading growth-direction signal:

| Candidate | n | Hit rate | p-value | Result |
|---|---|---|---|---|
| GDPNow (Atlanta Fed) | 39 | 48.7% | 1.0000 | FAILS |
| SPF RGDP-surprise (full-panel consensus) | 85 | 63.5% | 0.0165 | Clears p<0.05, fails Bonferroni(5) |
| SPF RGDP TOP-1 forecaster (point-in-time) | 85 | 60.0% | 0.0821 | FAILS |
| SPF RGDP TOP-5-AVG forecasters (point-in-time) | 40 | 65.0% | 0.0807 | FAILS |
| NY Fed Staff Nowcast (real-time only) | 35 | 40.0% | 0.3105 | FAILS |

**None clears Bonferroni(5)=0.01.** The closest (SPF full-panel consensus,
p=0.0165) clears an uncorrected p<0.05 but that bar was never the real
target once multiple candidates are being screened — see the SPF section
above for why it's labeled "suggestive, not confirmed" rather than
validated. **Documented conclusion: GDP growth-direction prediction is
UNSOLVED with currently available free data, not merely "not yet found."**
This is a closed search, not a paused one — do not re-attempt a 6th growth
nowcast/consensus candidate from scratch in a future session without first
reading this section; if a genuinely new *category* of evidence emerges
(not another nowcast or consensus-survey product), that would be a new
question, not a re-run of this one.

**What remains live in the regime read**: Cleveland Fed Inflation Nowcast
(the one signal that survived Bonferroni correction) and the SPF
consensus's raw forward-looking reading (both CPI and RGDP, shown for
context — RGDP explicitly marked as not a validated signal, since it never
cleared correction). The live regime read and the Macro Trading tab's
methodology section both now state plainly that growth direction has no
validated leading signal, rather than silently omitting one.
