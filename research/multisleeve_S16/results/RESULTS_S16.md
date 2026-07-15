# S16: Multi-Sleeve Blend (GP/A + S8 + S7-original)

**Trial #21, N_TRIALS=21 (DSR).** Registry key: `multisleeve_S16`. Status: `completed_in_sample`, **FAILS DSR**.

Direct follow-up to compositeIC_S15 (trial #19), built specifically to fix the two causes S15 diagnosed for its own failure. Uses three separately beta-neutral sleeves (no merged score) blended at pre-registered equal 1/3 capital weight.

## Verdict (one line)

**Both of S15's diagnosed problems were fixed — and the blend still failed, for a different, more fundamental reason.** GP/A-vs-S7 correlation dropped from S15's +0.563 to +0.280, and S7's IC stayed correctly signed (+0.036, unchanged) by keeping its original double-sort form. But the blend's Sharpe (-0.740) is *worse* than S15's own composite (-0.070) and worse than two of the three individual sleeves. Fixing the construction problems exposed the real one: S7's original double-sort, even correctly gated and signed, has a standalone Sharpe of -0.544 and max drawdown of -70% — a big enough loser on its own that giving it a full, undiluted 1/3 of the book (instead of S15's smoothed continuous score) makes things worse, not better. **FAILS DSR at N=21** (per-period SR -0.370 vs threshold 0.341). Misses the promotion bar (CALMAR≥1 or Sharpe≥0.8) by a wide margin.

## Universe note (confirmed with user before running)

The instructions called for actual point-in-time Russell 2000 constituent membership, sourced from `research/russell_reconstitution/`. That directory only contains **annual** Russell 3000 (R1000∪R2000, not R2000-specific) additions/deletions **event** lists (2016-2023, 2015 excluded, 2018/2019 deletions missing) — never a full point-in-time roster, and no quarterly granularity. Not reconstructable without new data acquisition, which the brief said not to do. **Confirmed with user: fell back to the standard $100M-$2B PIT survivorship-bias-free universe** used by every sleeve's own original construction and every other trial in this registry. IWM (iShares Russell 2000 ETF, already fetched by the Russell reconstitution trial — no new fetch) is used as the benchmark instead, both for beta regression and the year-by-year comparison. **The Russell PIT-membership upgrade remains an open, unmet item.**

## Pairwise correlation of sleeve returns — the diagnostic this trial exists to check

| | GP/A | S7 (original) | S8 |
|---|---:|---:|---:|
| **GP/A** | 1.000 | **0.280** | -0.288 |
| **S7 (original)** | 0.280 | 1.000 | **-0.515** |
| **S8** | -0.288 | -0.515 | 1.000 |

For comparison, S15's merged-composite version of this same trio showed GP/A-vs-IVOL-Value correlation of **+0.563**. Keeping S7 in its original hard-gated double-sort form (instead of S15's continuous rank-product score) **did reduce that correlation to +0.280** — a real, meaningful improvement, though still not negligible. GP/A-vs-S8 (-0.288) and S7-vs-S8 (-0.515) are both negative — genuinely diversifying pairs, if either had positive expected return to diversify.

## Headline comparison table

| Metric | **Blend** | GP/A (#8) | S7 (#13) | S8 (#14) |
|---|---:|---:|---:|---:|
| Sharpe (net, ann.) | **-0.740** | 0.134 | -0.544 | -1.107 |
| CALMAR | **-0.182** | 0.033 | -0.210 | -0.215 |
| Ann. return (net) | -7.12% | 0.94% | -14.70% | -10.22% |
| Ann. return (gross) | -0.64% | 4.93% | -6.52% | -3.08% |
| Market beta (own internal proxy) | n/a | -0.1030 | -0.0855 | -0.0114 |
| Market beta (vs IWM, common calendar) | -0.0579 | -0.1136 | -0.0219 | -0.0382 |
| Max drawdown | -39.06% | -28.31% | -70.15% | -47.43% |
| Mean IC | n/a | +0.0404 | +0.0358 | +0.0360 |
| IC t-stat | n/a | +2.283 | +1.356 | +3.397 |
| Win rate | 33.3% | 50.0% | 50.0% | 30.0% |
| Cost drag (bps/yr) | 668 | 390 | 882 | 761 |
| N periods (own cadence) | 24 (qtr) | 24 (qtr) | 24 (qtr) | 70 (mo) |

GP/A/S7 columns quoted verbatim from trial_registry.csv (unchanged since original trials — no re-derivation risk). S8 quoted at its own native monthly cadence (Sharpe -1.107, matching registry exactly).

The blend is worse than GP/A alone (best of the three) and worse than S15's own merged composite (-0.070). **This is the actual finding of this trial: fixing the diversification mechanics doesn't help when 2 of 3 legs have negative standalone expected return — there's nothing for the third leg to diversify into.**

## Realized beta — does blending three beta-neutral sleeves stay beta-neutral?

| Portfolio | Beta (vs IWM) | t-stat |
|---|---:|---:|
| **Blend** | -0.0579 | -0.808 |
| GP/A-alone | -0.1136 | -0.970 |
| S7-alone | -0.0219 | -0.120 |
| S8-alone | -0.0382 | -0.607 |

All four are statistically indistinguishable from zero (|t| < 1) on this common IWM benchmark — **yes, blending stayed beta-neutral here**, in contrast to S15 where the merged composite's ResidRev leg showed a significant positive beta (t=2.223) that survived into the composite. The separate-sleeve construction with each leg's own original beta-neutral sizing produced a genuinely flat blend.

Per-year beta (n=4 quarters/year, **low statistical power, reported as requested but not to be over-read**):

| Year | Beta | t-stat |
|---|---:|---:|
| 2015 | +0.238 | +1.53 |
| 2016 | -0.238 | -0.37 |
| 2017 | -0.808 | -0.31 |
| 2018 | -0.198 | -2.01 |
| 2019 | -0.298 | -1.94 |
| 2020 | -0.005 | -0.02 |

No consistent directional drift year to year; 2017's beta point estimate is large but wildly insignificant (n=4, t=-0.31).

## Year-by-year: blend vs Russell 2000 (IWM)

| Year | Blend (net) | IWM (R2000) | GP/A | S7 | S8 |
|---|---:|---:|---:|---:|---:|
| 2015 | -7.80% | -3.91% | +4.78% | -8.44% | -19.11% |
| 2016 | -4.12% | +24.88% | -13.86% | +15.17% | -13.76% |
| 2017 | -5.67% | +12.89% | +7.77% | -17.66% | -8.24% |
| 2018 | +2.43% | -10.36% | +47.19% | -18.29% | -14.42% |
| 2019 | -12.31% | +24.69% | -14.69% | -20.44% | -2.09% |
| 2020 | -14.31% | +19.95% | -13.39% | -31.73% | -0.58% |

The blend was negative in 5 of 6 years — 2018 is the lone positive year, driven almost entirely by GP/A's standout +47.19% that year (a small-n quarter effect already flagged in GP/A's own record). IWM (the actual market) had a strong period overall (positive in 4 of 6 years) while the blend, being market-neutral by design, didn't participate in that — expected for a beta-neutral L/S book, not itself a red flag.

## 2020 sub-period check (COVID crash + recovery)

Full-year 2020: **Blend -14.31%** | IWM +19.95% | GP/A -13.39% | S7 -31.73% | S8 -0.58%

| Quarter | Blend (net) | GP/A | S7 | S8 | IWM |
|---|---:|---:|---:|---:|---:|
| 2020-Q1 | -7.85% | -10.62% | -17.18% | +4.26% | -35.42% |
| 2020-Q2 | -11.60% | -5.98% | -30.35% | +1.52% | +33.40% |
| 2020-Q3 | +8.70% | +16.29% | +10.52% | -0.72% | +7.73% |
| 2020-Q4 | -3.22% | -11.38% | +7.10% | -5.39% | +29.24% |

S7 is the standout drag in 2020 (-31.73% for the year, -30.35% in Q2 alone during the COVID crash) — its double-sort's high-IVOL long/short legs appear to have been badly exposed to the volatility spike itself, separate from any value-mispricing signal. S8 was the only genuinely defensive sleeve in 2020 (-0.58% for the year, small moves each quarter) — consistent with its short holding period limiting crash exposure. Notably, S8's 2020 performance here (compounded from its own monthly returns) is much better than the -8.17%/-8.17%-magnitude single-quarter draws S15's own composite saw from residual-reversal in 2020 — S15 captured an unusually strong +30.25% single quarter from ResidRev in 2020-Q2 that doesn't appear here; that's because S15 evaluated ResidRev on a quarterly-redefined signal (last-month-of-quarter residual only, on S15's own composite-intersected universe), not S8's native monthly signal compounded across all three months on S8's own universe — different constructions produce different numbers, as expected and previously flagged.

## Falsification diagnostics

### Sign-convention audit

All three sleeves' IC signs are unchanged from their own original, already-validated trials (no signal code was touched):

| Sleeve | Mean IC | IC t-stat |
|---|---:|---:|
| GP/A (#8) | +0.0404 | +2.283 |
| S7 (#13, original double-sort) | +0.0358 | +1.356 |
| S8 (#14) | +0.0360 | +3.397 |

S7's IC stays correctly signed here specifically *because* it was never converted to a continuous score — this is the mechanical fix for S15's Problem 2, confirmed trivially by construction rather than re-derived.

### Sleeve-level concentration check (is the blend secretly one sleeve?)

Since the blend's "positions" are three sleeves at fixed 1/3 weight (not individual stocks directly), concentration is measured as each sleeve's share of the blend's quarterly |return| mass:

- Mean top-1-sleeve share: **60.1%** | Max: 84.7%
- Quarters where one sleeve is >50% of |return| mass: **17/24**
- Quarters where one sleeve is >60% of |return| mass: 11/24
- Dominant sleeve by quarter count: S7 (14), GP/A (7), S8 (3)

This is a real finding: **the blend's quarter-to-quarter movement is usually dominated by a single sleeve, most often S7** — the equal capital weighting doesn't translate into equal *contribution*, because S7's return swings are simply larger in magnitude than GP/A's or S8's (its max drawdown alone is -70% vs GP/A's -28% and S8's -47%). This is a genuine three-way blend in the sense that all three sleeves take turns dominating (not one sleeve every quarter), but it is not a smooth, evenly-diversified blend either. Note: this checks sleeve-level concentration, not within-sleeve per-stock concentration — the latter would require re-instrumenting each sleeve's original backtest, which was intentionally not done to avoid any risk of diverging from each sleeve's exact original, already-validated construction.

## Why this trial still failed, despite fixing both diagnosed problems

1. **Correlation was reduced but not eliminated** (+0.563 → +0.280 for the GP/A-S7 pair), and the other two pairs are usefully negative (-0.288, -0.515). Diversification mechanics genuinely improved.
2. **S7's sign was preserved** by not converting it to a continuous score — its IC stayed positive and matches its original trial exactly.
3. **But neither fix addresses the real constraint**: S7 and S8 both have negative standalone net Sharpe (-0.544 and -1.107) even in their correctly-constructed original forms. Diversification can smooth out volatility across legs with genuinely different return DRIVERS, but it cannot manufacture positive expected return from legs that don't have it. Only GP/A (Sharpe +0.134) has a positive edge among the three, and one positive-Sharpe leg diluted to 1/3 weight against two negative-Sharpe legs at 1/3 weight each nets out negative, exactly as observed.

**The lesson for any future signal-combination trial in this registry: check each candidate sleeve's standalone net Sharpe BEFORE combining, not just pairwise correlation.** A near-zero-correlation blend of three losers is still a loser; the diversification benefit only pays off when combined with at least a couple of legs that have genuine standalone positive expected return.

## Data / infrastructure

No new data sourcing, no re-running of any sleeve's backtest — pure aggregation over already-validated CSVs: `research/gp/results/gp_period_returns.csv`, `research/ivol_value/results/s7_period_returns.csv`, `research/residual_reversal/results/s8_period_returns.csv`, and `research/russell_reconstitution/cache/tiingo/IWM.csv`. Runtime: seconds.

Output files: `research/multisleeve_S16/results/{s16_blended_period_returns.csv, s16_sleeve_correlation.csv, s16_year_by_year.csv, s16_concentration.csv, s16_per_year_beta.csv}`.

## Note on registry filename

The brief asked to log to `research/PHASE3_TRIAL_REGISTRY.md`. No file by that exact name exists; the file that actually plays that role (and that the Streamlit "Phase 3 — Signal Research Registry" tab's code comments cite as its narrative companion) is `research/CORRECTED_TRIAL_REGISTRY.md`. Logged there instead — see the entry appended under "Trial #21" in that file. That document already flags itself as stale beyond DSR #13 and defers to `trial_registry.csv` as the authoritative source for anything not detailed in its narrative sections; this trial's full detail lives in this file and in the registry row, with only a summary entry added there.
