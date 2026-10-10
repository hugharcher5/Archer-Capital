# Part 2 Walk-Forward Validation (2026-07-17)
Out-of-sample follow-up to the original in-sample lead-time test (see 'Verification audit' in README.md). This is verification of existing infrastructure, not a DSR trial -- not logged to the registry. Does not change Part 1's regime_timeline.csv or the live classifier.

Design: expanding-window anchor starting 2010-01-01, one expansion step per subsequent regime change, HORIZON_DAYS=365 (locked before running -- more generous than every in-sample lead time observed, so it cannot retroactively penalize a prior 'hit').

**Read this before the tables below**: sample sizes here are extremely small (3-8 onsets being walk-forward tested per flag, since the anchor already consumes the pre-2010 onsets as training evidence). This is a low-confidence read regardless of outcome -- not proof a flag works, and not proof it doesn't.

## curve_10y2y_inversion

### Per-transition scorecard (39 expansion steps)

step_change_date anticipated_by_onset  lead_time_days verdict
      2010-08-01                 None             NaN  MISSED
      2011-05-01                 None             NaN  MISSED
      2012-08-01                 None             NaN  MISSED
      2012-09-01                 None             NaN  MISSED
      2013-02-01                 None             NaN  MISSED
      2013-08-01                 None             NaN  MISSED
      2014-02-01                 None             NaN  MISSED
      2014-04-01                 None             NaN  MISSED
      2014-07-01                 None             NaN  MISSED
      2014-08-01                 None             NaN  MISSED
      2015-01-01                 None             NaN  MISSED
      2015-02-01                 None             NaN  MISSED
      2015-05-01                 None             NaN  MISSED
      2015-06-01                 None             NaN  MISSED
      2015-07-01                 None             NaN  MISSED
      2015-09-01                 None             NaN  MISSED
      2016-08-01                 None             NaN  MISSED
      2016-10-01                 None             NaN  MISSED
      2016-12-01                 None             NaN  MISSED
      2017-01-01                 None             NaN  MISSED
      2017-02-01                 None             NaN  MISSED
      2017-05-01                 None             NaN  MISSED
      2018-02-01                 None             NaN  MISSED
      2019-03-01                 None             NaN  MISSED
      2019-10-01           2019-08-27            35.0     HIT
      2020-05-01           2019-08-27           248.0     HIT
      2021-01-01                 None             NaN  MISSED
      2021-02-01                 None             NaN  MISSED
      2021-04-01                 None             NaN  MISSED
      2021-06-01                 None             NaN  MISSED
      2022-02-01                 None             NaN  MISSED
      2022-10-01           2022-07-06            87.0     HIT
      2022-12-01           2022-07-06           148.0     HIT
      2023-02-01           2022-07-06           210.0     HIT
      2023-05-01           2022-07-06           299.0     HIT
      2023-08-01                 None             NaN  MISSED
      2024-11-01                 None             NaN  MISSED
      2025-11-01                 None             NaN  MISSED
      2026-05-01                 None             NaN  MISSED

### Per-onset scorecard (3 onsets tested, i.e. onsets >= anchor cutoff)

onset_date next_change  lead_time_days status
2019-08-27  2019-10-01              35    HIT
2022-04-01  2022-10-01             183    HIT
2022-07-06  2022-10-01              87    HIT

**Summary**: 6/39 transitions correctly anticipated, 33/39 missed (no prior flag). 3/3 onsets led to a timely change (hit), 0/3 confirmed false positives, 0/3 still pending (too recent to resolve).

**Null/base-rate comparison (per-onset framing)**: 95% of ALL calendar months in the walk-forward window (n=187) already had a regime change within 365 days, with NO conditioning on any flag -- this is why the per-onset '100% hit rate' above is NOT evidence of skill: it's close to guaranteed by base rate given how frequently regime changes occur.

**Permutation test (the real test — per-transition COVERAGE framing)**: placing 3 random dates in the same window 20000 times, the null distribution covers a mean of 6.6 / 39 transitions (median 6) purely by chance placement. The real flag covered 6 / 39 -- empirical p-value = 0.646 (fraction of random placements that did at least as well).

## curve_10y3m_inversion

### Per-transition scorecard (39 expansion steps)

step_change_date anticipated_by_onset  lead_time_days verdict
      2010-08-01                 None             NaN  MISSED
      2011-05-01                 None             NaN  MISSED
      2012-08-01                 None             NaN  MISSED
      2012-09-01                 None             NaN  MISSED
      2013-02-01                 None             NaN  MISSED
      2013-08-01                 None             NaN  MISSED
      2014-02-01                 None             NaN  MISSED
      2014-04-01                 None             NaN  MISSED
      2014-07-01                 None             NaN  MISSED
      2014-08-01                 None             NaN  MISSED
      2015-01-01                 None             NaN  MISSED
      2015-02-01                 None             NaN  MISSED
      2015-05-01                 None             NaN  MISSED
      2015-06-01                 None             NaN  MISSED
      2015-07-01                 None             NaN  MISSED
      2015-09-01                 None             NaN  MISSED
      2016-08-01                 None             NaN  MISSED
      2016-10-01                 None             NaN  MISSED
      2016-12-01                 None             NaN  MISSED
      2017-01-01                 None             NaN  MISSED
      2017-02-01                 None             NaN  MISSED
      2017-05-01                 None             NaN  MISSED
      2018-02-01                 None             NaN  MISSED
      2019-03-01                 None             NaN  MISSED
      2019-10-01           2019-05-13           141.0     HIT
      2020-05-01           2020-01-31            91.0     HIT
      2021-01-01           2020-01-31           336.0     HIT
      2021-02-01                 None             NaN  MISSED
      2021-04-01                 None             NaN  MISSED
      2021-06-01                 None             NaN  MISSED
      2022-02-01                 None             NaN  MISSED
      2022-10-01                 None             NaN  MISSED
      2022-12-01           2022-10-18            44.0     HIT
      2023-02-01           2022-10-18           106.0     HIT
      2023-05-01           2022-10-18           195.0     HIT
      2023-08-01           2022-10-18           287.0     HIT
      2024-11-01                 None             NaN  MISSED
      2025-11-01           2025-10-16            16.0     HIT
      2026-05-01           2025-10-16           197.0     HIT

### Per-onset scorecard (6 onsets tested, i.e. onsets >= anchor cutoff)

onset_date next_change  lead_time_days status
2019-03-22  2019-10-01             193    HIT
2019-05-13  2019-10-01             141    HIT
2020-01-31  2020-05-01              91    HIT
2022-10-18  2022-12-01              44    HIT
2025-02-26  2025-11-01             248    HIT
2025-10-16  2025-11-01              16    HIT

**Summary**: 9/39 transitions correctly anticipated, 30/39 missed (no prior flag). 6/6 onsets led to a timely change (hit), 0/6 confirmed false positives, 0/6 still pending (too recent to resolve).

**Null/base-rate comparison (per-onset framing)**: 95% of ALL calendar months in the walk-forward window (n=187) already had a regime change within 365 days, with NO conditioning on any flag -- this is why the per-onset '100% hit rate' above is NOT evidence of skill: it's close to guaranteed by base rate given how frequently regime changes occur.

**Permutation test (the real test — per-transition COVERAGE framing)**: placing 6 random dates in the same window 20000 times, the null distribution covers a mean of 12.1 / 39 transitions (median 12) purely by chance placement. The real flag covered 9 / 39 -- empirical p-value = 0.864 (fraction of random placements that did at least as well).

## credit_spread_widening

### Per-transition scorecard (39 expansion steps)

step_change_date anticipated_by_onset  lead_time_days verdict
      2010-08-01           2010-05-26            67.0     HIT
      2011-05-01           2010-05-26           340.0     HIT
      2012-08-01           2011-08-09           358.0     HIT
      2012-09-01                 None             NaN  MISSED
      2013-02-01                 None             NaN  MISSED
      2013-08-01                 None             NaN  MISSED
      2014-02-01                 None             NaN  MISSED
      2014-04-01                 None             NaN  MISSED
      2014-07-01                 None             NaN  MISSED
      2014-08-01                 None             NaN  MISSED
      2015-01-01                 None             NaN  MISSED
      2015-02-01           2015-01-15            17.0     HIT
      2015-05-01           2015-01-15           106.0     HIT
      2015-06-01           2015-01-15           137.0     HIT
      2015-07-01           2015-01-15           167.0     HIT
      2015-09-01           2015-01-15           229.0     HIT
      2016-08-01                 None             NaN  MISSED
      2016-10-01                 None             NaN  MISSED
      2016-12-01                 None             NaN  MISSED
      2017-01-01                 None             NaN  MISSED
      2017-02-01                 None             NaN  MISSED
      2017-05-01                 None             NaN  MISSED
      2018-02-01                 None             NaN  MISSED
      2019-03-01           2018-11-14           107.0     HIT
      2019-10-01           2018-11-14           321.0     HIT
      2020-05-01           2020-03-03            59.0     HIT
      2021-01-01           2020-03-03           304.0     HIT
      2021-02-01           2020-03-03           335.0     HIT
      2021-04-01                 None             NaN  MISSED
      2021-06-01                 None             NaN  MISSED
      2022-02-01                 None             NaN  MISSED
      2022-10-01           2022-02-24           219.0     HIT
      2022-12-01           2022-02-24           280.0     HIT
      2023-02-01           2022-02-24           342.0     HIT
      2023-05-01                 None             NaN  MISSED
      2023-08-01           2023-05-15            78.0     HIT
      2024-11-01                 None             NaN  MISSED
      2025-11-01           2025-04-03           212.0     HIT
      2026-05-01                 None             NaN  MISSED

### Per-onset scorecard (8 onsets tested, i.e. onsets >= anchor cutoff)

onset_date next_change  lead_time_days status
2010-05-26  2010-08-01              67    HIT
2011-08-09  2012-08-01             358    HIT
2015-01-15  2015-02-01              17    HIT
2018-11-14  2019-03-01             107    HIT
2020-03-03  2020-05-01              59    HIT
2022-02-24  2022-10-01             219    HIT
2023-05-15  2023-08-01              78    HIT
2025-04-03  2025-11-01             212    HIT

**Summary**: 18/39 transitions correctly anticipated, 21/39 missed (no prior flag). 8/8 onsets led to a timely change (hit), 0/8 confirmed false positives, 0/8 still pending (too recent to resolve).

**Null/base-rate comparison (per-onset framing)**: 95% of ALL calendar months in the walk-forward window (n=187) already had a regime change within 365 days, with NO conditioning on any flag -- this is why the per-onset '100% hit rate' above is NOT evidence of skill: it's close to guaranteed by base rate given how frequently regime changes occur.

**Permutation test (the real test — per-transition COVERAGE framing)**: placing 8 random dates in the same window 20000 times, the null distribution covers a mean of 15.3 / 39 transitions (median 15) purely by chance placement. The real flag covered 18 / 39 -- empirical p-value = 0.268 (fraction of random placements that did at least as well).

## Overall summary table

                  flag transitions_hit onsets_confirmed_fp  onsets_pending  null_hit_rate_pct (per-onset)  perm_test_p_value (per-transition)
 curve_10y2y_inversion            6/39                 0/3               0                           95.0                               0.646
 curve_10y3m_inversion            9/39                 0/6               0                           95.0                               0.864
credit_spread_widening           18/39                 0/8               0                           95.0                               0.268

## Verdict

See README.md's Verification audit section (2026-07-17 entry) for the full honest verdict on whether any flag survives this out-of-sample test.
