# Part 2 -- Predictive/Early Layer: Lead-Time Report
Reported separately from Part 1's coincident label per README. This is the harder ask; a weak result here is reported as-is, not hidden.

Total coincident regime-label changes in history: 73
Regime timeline coverage: 1997-01-01 -> 2026-07-01

**Sample-size caveat**: each flag has only 8-13 onsets within the timeline's coverage window. A one-sided Mann-Whitney U test (H1: lead times are stochastically shorter than the baseline gap between regime changes) is reported below for each flag, but at this sample size the test has very limited power -- a non-significant p-value means the effect isn't ESTABLISHED at this sample size, not that it's disproven. Treat all of Part 2 as suggestive, not conclusive.

**Base-rate baseline (read this before the flags below):** regime labels change on average every 148 days (median 92d) with NO conditioning on any flag at all. Because changes happen this often, a flag finding '100% of onsets preceded a subsequent change' is a weak claim by itself -- almost any onset will precede *some* change within a few months just by base rate. The only honest test is whether a flag's lead time is meaningfully **shorter** than this baseline (concentration), not merely nonzero -- which is exactly what the Mann-Whitney U test below checks formally, rather than an eyeballed threshold.

## curve_10y2y_inversion
- Onsets within timeline coverage: 8 (6 earlier onsets excluded -- predate the regime timeline, no valid lead-time baseline)
- Lead time to ANY subsequent regime-label change: 8/8 onsets had one; mean 88d, median 62d
- Lead time to a change specifically INTO Stagflation/Deflation-Risk-off: 8/8 onsets had one; mean 148d, median 116d
- Mann-Whitney U vs. base-rate baseline (lead times shorter than baseline gaps?): **marginal, p=0.099 (n=8) -- directionally suggestive but NOT significant at conventional levels; do not call this 'concentrated' or 'confirmed'**

## curve_10y3m_inversion
- Onsets within timeline coverage: 13 (4 earlier onsets excluded -- predate the regime timeline, no valid lead-time baseline)
- Lead time to ANY subsequent regime-label change: 13/13 onsets had one; mean 110d, median 104d
- Lead time to a change specifically INTO Stagflation/Deflation-Risk-off: 13/13 onsets had one; mean 140d, median 117d
- Mann-Whitney U vs. base-rate baseline (lead times shorter than baseline gaps?): **NOT significant, p=0.161 (n=13)**

## credit_spread_widening
- Onsets within timeline coverage: 12 (5 earlier onsets excluded -- predate the regime timeline, no valid lead-time baseline)
- Lead time to ANY subsequent regime-label change: 12/12 onsets had one; mean 145d, median 92d
- Lead time to a change specifically INTO Stagflation/Deflation-Risk-off: 12/12 onsets had one; mean 249d, median 228d
- Mann-Whitney U vs. base-rate baseline (lead times shorter than baseline gaps?): **NOT significant, p=0.464 (n=12)**

## Base-effect model -- most recent readings

GDP (last 4 quarters):

    period  gdp_yoy  gdp_forward_yoy_flat  gdp_base_effect_bias gdp_implied_direction
2025-04-01 0.033446              0.012787             -0.020659            decel-bias
2025-07-01 0.027311              0.020996             -0.006316            decel-bias
2025-10-01 0.024688              0.024898              0.000210            accel-bias
2026-01-01 0.027563              0.020656             -0.006907            decel-bias

Core CPI (last 6 months):

    period  cpi_yoy  cpi_forward_yoy_flat  cpi_base_effect_bias cpi_implied_direction
2026-01-01 0.024801              0.022484             -0.002317            decel-bias
2026-02-01 0.024693              0.024114             -0.000579            decel-bias
2026-03-01 0.026119              0.023696             -0.002424            decel-bias
2026-04-01 0.027550              0.026217             -0.001333            decel-bias
2026-05-01 0.028352              0.026010             -0.002342            decel-bias
2026-06-01 0.025839              0.022543             -0.003296            decel-bias
