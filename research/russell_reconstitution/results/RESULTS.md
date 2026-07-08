# Russell-1: Pre-Effective-Date Anticipatory Drift — Results

Trial #17 in the program registry; first trial in a new independent DSR
family, **"Russell Reconstitution"**. Logged as `russell1_anticipatory_drift`
in `research/trial_registry.csv`. Event-study design (same class as H1
livestock) — no Sharpe/DSR gate; the CAAR t-stat across cycles is the
significance test.

## Per-cycle results

| Year | Entry | Exit | Confirmed adds | Priced (cov%) | Basket gross | IWM bench | Abnormal (gross) | Abnormal (net, 50bps) |
|-----:|-------|------|---------------:|---------------:|-------------:|----------:|------------------:|------------------------:|
| 2016 | 2016-06-13 | 2016-06-24 | 183 | 159 (86.9%) | -0.30% | -1.97% | **+1.67%** | +1.17% |
| 2017 | 2017-06-01 | 2017-06-23 | 195 | 165 (84.6%) | +0.33% | +1.41% | -1.08% | -1.58% |
| 2018 | 2018-06-11 | 2018-06-22 | 182 | 152 (83.5%) | +3.24% | +0.63% | **+2.61%** | +2.11% |
| 2019 | 2019-06-10 | 2019-06-28 | 181 | 150 (82.9%) | +5.14% | +2.83% | **+2.31%** | +1.81% |
| 2020 | 2020-06-15 | 2020-06-26 | 203 | 180 (88.7%) | +0.69% | -3.06% | **+3.75%** | +3.25% |
| 2021 | 2021-06-07 | 2021-06-25 | 254 | 220 (86.6%) | +0.58% | +0.95% | -0.37% | -0.87% |
| 2022 | 2022-06-06 | 2022-06-24 | 292 | 269 (92.1%) | -15.43% | -6.55% | **-8.87%** | -9.37% |
| 2023 | 2023-05-22 | 2023-06-23 | 281 | 263 (93.6%) | +6.64% | +1.57% | **+5.07%** | +4.57% |

## CAAR summary (8 cycles, 2016-2023 in-sample; 2024-2025 untouched holdout)

- **CAAR (gross)**: +0.637% | **CAAR (net of 50bps round-trip)**: +0.137%
- Cross-cycle std dev: 4.33% | **t-stat: +0.416** | two-sided p ≈ 0.69
- **Win rate: 62.5%** (5/8 cycles positive abnormal return)
- Mean coverage (priced/confirmed): 87.4% | fallback-to-raw-close tickers: 0
- Mean fraction of addition names inside Archer's core $100M-$2B universe: **38.9%** (secondary breakdown, as requested — most Russell additions are outside Archer's core screen, skewing toward either sub-$100M microcap-origin names or multi-billion-dollar R1000 promotions)

## Cost structure vs. prior high-turnover trials

One entry + one exit per name per year (buy-and-hold the window) — flat 50bps
round-trip assumption (see caveat below), charged once per name per cycle.
Compare annualized cost drag elsewhere in the program: S2 PEAD 4,079 bps/yr,
S4 MAX 2,202 bps/yr, S8 residual reversal 761 bps/yr. Russell-1's structural
turnover is a small fraction of these by design — confirmed: net CAAR (+0.14%)
is barely below gross (+0.64%), a 50bps drag vs. hundreds/thousands of bps
elsewhere.

## Falsification diagnostics

1. **Sign-convention audit**: PASS by construction — basket built exclusively
   from `list_kind='additions'` rows; no deletions contamination possible.
2. **Concentration check**: largest single-cycle contribution to total
   |abnormal return| is 34.5% (2022, the negative outlier) — under the 40%
   flag threshold, so no single year dominates the (already-null) result.
3. **Decay-over-time check**: corr(year, abnormal_ret) = -0.15 — flat, no
   clear crowding-out pattern detectable in 8 points.

## Verdict

**No statistically significant pre-effective-date drift** (t = +0.416,
p ≈ 0.69). Sign is positive but weak and not distinguishable from zero;
2022 alone (-8.87% abnormal) offsets most of the other years' modest gains.
**Recommendation: do NOT unlock the 2024-2025 holdout** — the in-sample
baseline shows nothing to validate out-of-sample.

## Data & methodology caveats (read before reusing this result)

- **2015 excluded from in-sample**: only press-release "highlights" PDFs
  (not ticker-level tables) were recoverable for 2015 via the URL patterns
  this pipeline checked. *(Addendum, found in a later session reusing this
  pipeline: a separate URL path — the live HTML tool page rather than a PDF —
  does have real 2015 ticker-level tables, archived at Wayback snapshot
  `20150627172039`. It is FINAL-only, no preliminary snapshot recoverable, so
  it doesn't fit this trial's entry/exit design without a proxy decision. Not
  wired into the backtest; raw snapshot kept at
  `cache/2015_addendum/additions-deletions_20150627172039.html` as an open
  item if a 9th cycle is ever added — see README.md addendum.)*
- **2018 "final" list is a proxy**: no PDF explicitly labeled "final" survives
  in the archive for 2018; the last-captured "preliminary"-titled snapshot
  (crawled 2 days after the official final-close date) is used as the
  confirmed universe.
- **2020 preliminary date is a proxy**: the COVID-disrupted 2020 schedule's
  exact first-preliminary-post date wasn't directly confirmed by press
  search; the query-period-end date is used as a conservative estimate.
- **Benchmark is cap-weighted IWM**, not a constructed equal-weight small-cap
  index — a pragmatic simplification (real, tradable, directly the fund
  tracking the index under study) rather than building a full equal-weight
  universe return series for this standalone trial.
- **Cost model is a flat 25bps/leg assumption**, not the cap-tier bid-ask
  formula used elsewhere in the program — Russell-eligible names are
  index-liquidity-screened (structurally more liquid than the general
  $100M-$2B microcap tail), and precise point-in-time market cap per ticker
  wasn't cheaply available without building fresh SEC XBRL shares-outstanding
  infrastructure for ~1,500 tickers largely outside the program's existing
  cached universe.
- **Coverage is 87.4% mean**, not 100% — some confirmed additions (mostly
  thinly-traded micro-caps and a handful of tickers that were later
  acquired/delisted/renamed) have no Tiingo price history.
