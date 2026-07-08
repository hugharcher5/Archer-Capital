# CORRECTED TRIAL REGISTRY

**Generated:** 2026-07-06 · **Russell-1 added:** 2026-07-08  
**Verified against:** source files in `/mnt/c/Users/user/Archer-Capital/research/`  
**In-sample window:** 2015-01-01 → 2021-01-01 (Russell-1 uses its own annual-cycle window, 2016–2023 — see its detail section)  
**DSR gate:** Deflated Sharpe Ratio, cumulative independent-trial count  
**Universe:** US small/mid-cap equities (SEC XBRL + Tiingo)

> **Known staleness (as of 2026-07-08):** this document's DSR Accounting Table and Master
> Results Table only go up to DSR #13 (S7), then jump to #17 (Russell-1). **Trials #14
> (S8, residual short-term reversal), #15 (S10, Amihud illiquidity), #16 (S11, accruals
> quality), and the S3-SA walk-forward (#8b-WF)** were all run and logged to
> `research/trial_registry.csv` after this document was last generated, but have not been
> backfilled into Sections 2–4 here. Section 5's S1–S10 status table below still marks S8
> and S10 as **PENDING**, which is now incorrect — both are complete (FAIL DSR). Treat
> `research/trial_registry.csv` as the authoritative, currently up-to-date source for any
> trial not detailed in this document; this markdown file is a narrative companion, not
> the source of truth.

---

## 1. Corrections from Chat-Assembled Version

Every number in the prior chat-assembled registry was checked against actual output files. The table below lists all discrepancies found.

| Trial | Field | Chat Value | Actual Value | Source File | Status |
|-------|-------|-----------|--------------|-------------|--------|
| E1–E4 | Period-24 hold_end | 2021-01-01 (implied) | **2025-10-01** | `mgmt_pit/results/in_sample_perf.csv` row 25 | BUG-PENDING |
| E2 | Max drawdown | −63.6% | −63.548% (rounds to −63.5%) | `mgmt_pit/results/trial_registry.csv` | Minor rounding; conclusion unchanged |
| S1 IVOL | Ann. return | −15.97% | **−10.93%** (arithmetic ann.) | `research/trial_registry.csv` row 5 | DISCREPANCY — stale v1 value |
| S1 IVOL | Sharpe | −0.560 | **−0.464** | `research/trial_registry.csv` row 5 | DISCREPANCY — stale v1 value |
| S1 IVOL | Max drawdown | −80.0% | **−70.8%** | `research/trial_registry.csv` row 5 | DISCREPANCY — stale v1 value |
| S1 IVOL | Market beta | −0.655 | **−0.220** | `research/trial_registry.csv` row 5 | DISCREPANCY — stale v1 value |
| S1 IVOL | IC | −0.0651 | −0.0646 | `research/trial_registry.csv` row 5 | Within rounding; not material |
| S1 IVOL | IC t-stat | −2.30 | −2.274 | `research/trial_registry.csv` row 5 | Within rounding; not material |
| S3-A | Trial registry entry | Listed as DSR #10 | **No entry in trial_registry.csv** | `research/trial_registry.csv` | UNVERIFIABLE from registry; numbers verified from `gp/results/gp_decay_annual.csv` |
| S7 | Max drawdown | −70.2% | −70.1% (−0.70145) | `research/trial_registry.csv` row 15 | Minor rounding; not material |
| Bug 2 (S3 DSR thresholds) | "Three conflicting thresholds" | 0.235 / 0.332 / 0.470 = conflicting | **Consistent; N=8 for all three** | `gp/run_decay.py` lines 13–17; `research/trial_registry.csv` | NOT A BUG — see note below |

**Bug 2 clarification:** The three per-period SR thresholds (0.235 / 0.332 / 0.470) are not conflicting. They are all derived from the same formula `norm.ppf(1 - 1/N) / sqrt(n_periods)` with N=8, differing only in n_periods (24 / 12 / 6 for quarterly / semiannual / annual). The annualised DSR threshold is frequency-invariant at **0.470** for all three (`norm.ppf(0.875) / sqrt(6 years) ≈ 1.150 / 2.449`). The `gp/run_decay.py` code comments state this explicitly. N_TRIALS was 8 at the time of all three runs.

**Bug 1 (E1–E4 period-24) detail:** `mgmt_pit/results/in_sample_perf.csv` row 25 shows `rebalance_date=2020-10-01`, `hold_end=2025-10-01`. The correct hold_end for the final in-sample period should be 2021-01-01. This caused period 24 to capture five years of out-of-sample data (2020-Q4 through 2025-Q4 approximately) rather than one quarter. The per-period composite_ls for the bug period is +31.4%, with individual signals ranging from M2_solo_ls = +52.3% to M3_solo_ls = −56.7%. No corrected re-run exists. All four E trials still show negative Sharpe ratios despite this contamination, so the DSR FAIL conclusion is likely unchanged — but the exact return and drawdown figures are wrong.

**S1 IVOL v1 → v2 detail:** The chat-assembled registry captured pre-correction v1 values. The `ivol/run.py` docstring explicitly states: "v1 produced β = −0.655 (t = −5.1), contaminating [the return] with a directional short-market bet rather than the IVOL signal." The v2 corrected run replaced 50/50 dollar-neutral construction with beta-neutral sizing (`long_$ × β_L = short_$ × β_S`), which reduced the structural net-short-market position. The trial_registry.csv was updated in-place ("CORRECTED RUN v2 — replaces prior trial #7 entry, same hypothesis, NOT a new independent trial"). All numbers reported below for S1 are from the v2 corrected run.

---

## 2. DSR Accounting Table

N_TRIALS counts cumulative independent trials; family members reuse the same trial slot.

| DSR # | Study ID | Trial(s) | N_TRIALS at run | n_periods | Per-period SR threshold | Ann. SR threshold | DSR result |
|------:|----------|----------|----------------|----------|------------------------|-------------------|-----------|
| 1–4 | E1–E4 | M1 insider conviction, M2 non-dilution, M3 ROIC improvement, E4 composite | 4 | 24 (quarterly) | 0.1377 | 0.269 | All FAIL [BUG-PENDING] |
| 5 | H1 | Livestock disease event study | n/a (event study, no SR gate) | N/A | N/A | N/A | STOP (wrong sign) |
| 6 | S9 + S6 | Cross-sectional 12-1 momentum; residual momentum (family) | 6 | 24 (quarterly) | 0.1975 | 0.385 | Both FAIL |
| 7 | S1 + S4 | IVOL low-lottery; MAX high-lottery (IVOL/MAX family) | 7 | S1: 24 qtrly; S4: 72 monthly | S1: 0.2179; S4: 0.1258 | S1: 0.435; S4: 0.435 | Both FAIL |
| 8 | S3-Q | GP/A quality, quarterly rebalance | 8 | 24 | 0.2348 | 0.470 | FAIL |
| 9 | S3-SA | GP/A quality, semiannual (frequency sweep, not new trial) | 8 | 12 | 0.3321 | 0.470 | **PASS** [best-of-3] |
| 10 | S3-A | GP/A quality, annual (frequency sweep, not new trial) | 8 | 6 | 0.4698 | 0.470 | FAIL |
| 11 | S2 | PEAD post-earnings drift | 11 | 72 (monthly calendar-time) | 0.1574 | 0.545 | FAIL |
| 12 | S5 | Asset growth | 12 | 24 (quarterly) | 0.2823 | 0.550 | FAIL |
| 13 | S7 | IVOL-conditional value (double-sort) | 13 | 24 (quarterly) | 0.2911 | 0.567 | FAIL |
| 17 | Russell-1 | Pre-effective-date anticipatory drift (new independent family: Russell Reconstitution) | 17 | 8 (annual cycles) | n/a (event study) | n/a (event study) | FAIL (CAAR t=+0.416, not significant) |

**Gap in this table (DSR #14–16):** S8 (residual short-term reversal), S10 (Amihud illiquidity), and the S3-SA walk-forward were run and logged to `research/trial_registry.csv` (trials #14, #15, #8b-WF) after this document was last generated, but have not yet been backfilled into this table — see Section 8 (Open Items) and the divergence note at the top of this document. S11 (accruals, trial #16) is also missing. Russell-1 is DSR #17 in `trial_registry.csv`'s cumulative count, which already reflects S8/S10/S11 as prior independent trials — the jump from #13 to #17 here is a gap in this document, not a gap in the actual trial count.

**Notes on DSR accounting:**
- H1 (livestock event study) is an event-study design; the stop criterion was wrong-sign CAAR, not SR vs DSR threshold. H1 adds to the independent-trial count as trial #5 regardless.
- S6 (residual momentum) shares DSR #6 with S9; N_TRIALS stays at 6 because they are in the same momentum family (raw vs residual variant of the same hypothesis).
- S4 (MAX) and S1 (IVOL) share DSR #7; same behavioral family (extreme-return aversion / lottery stocks).
- S3-SA, S3-A are frequency-sweep variants of S3-Q; N_TRIALS stays at 8. Only the semiannual variant cleared, making it best-of-3 selection.
- DSR threshold formula: `norm.ppf(1 − 1/N) / sqrt(n_periods)` in per-period SR units; annualised = `norm.ppf(1 − 1/N) / sqrt(n_years)`.
- S3-A has **no entry in `research/trial_registry.csv`** (the data file `gp/results/gp_decay_annual.csv` was saved but the registry row was never written). All S3-A numbers below are verified directly from that data file.

---

## 3. Master Results Table

All numbers from source files; `[BUG-PENDING]` marks values contaminated by the E1–E4 period-24 bug; `[v2]` marks the corrected IVOL run.

| ID | Gross %/yr | Cost bps/yr | Net %/yr | Sharpe (net, ann.) | Max DD | IC | IC t | Beta | DSR |
|----|-----------|-------------|---------|-------------------|--------|-----|------|------|-----|
| E1 [BUG-PENDING] | n/a | n/a | −0.84% | −0.075 | −32.8% | +0.0002 | 0.02 | n/a | FAIL (−0.037 < 0.138) |
| E2 [BUG-PENDING] | n/a | n/a | −5.46% | −0.169 | −63.5% | +0.0186 | 1.82 | n/a | FAIL (−0.084 < 0.138) |
| E3 [BUG-PENDING] | n/a | n/a | −12.95% | −0.560 | −67.4% | −0.0115 | −0.86 | n/a | FAIL (−0.280 < 0.138) |
| E4 [BUG-PENDING] | n/a | n/a | −2.09% | −0.113 | −41.8% | — | — | n/a | FAIL (−0.057 < 0.138) |
| H1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | STOP (wrong sign) |
| S9 | −4.87% (gross only; cost not modeled) | n/a | n/a | −0.324 | −45.3% | −0.0157 | −0.82 | −0.063 | FAIL (−0.162 < 0.197) |
| S6 | +4.31% | 762 | −3.34% | −0.156 | −27.4% | −0.0129 | −0.70 | +0.067 | FAIL (−0.078 < 0.197) |
| S1 [v2] | n/a | n/a | −10.93% (arith.) | −0.464 | −70.8% | −0.0646 | −2.274 | −0.220 | FAIL (−0.232 < 0.218) |
| S4 | +4.76% | 2,202 | −16.06% | −1.264 | −65.4% | +0.0341 | 2.463 | +0.019 | FAIL (−0.365 < 0.126) |
| S3-Q | +4.93% | 390 | +0.94% | +0.134 | −28.3% | +0.0404 | 2.283 | −0.103 | FAIL (0.067 < 0.235) |
| S3-SA | +10.16% | 318 | +6.85% | +0.569 | −17.9% | +0.0541 | 1.697 | −0.078 | **PASS** (0.403 > 0.332) |
| S3-A | +8.61% | ~321 est. | +5.92% | +0.361 | −18.3% | +0.0667 | 1.115 | n/a | FAIL (0.242 < 0.470) |
| S2 | −4.74% | 4,079 | −36.59% | −2.864 | −93.2% | +0.0368 | 7.555 | −0.104 | FAIL (−0.827 < 0.157) |
| S5 | +0.62% | 613 | −5.46% | −0.340 | −34.8% | −0.0099 | −0.517 | −0.160 | FAIL (−0.170 < 0.282) |
| S7 | −6.52% | 882 | −14.70% | −0.544 | −70.1% | +0.0358 | 1.356 | −0.085 | FAIL (−0.272 < 0.291) |
| Russell-1 | +0.64%/cycle | 50 (flat) | +0.14%/cycle | +0.032 | −10.2% | n/a | n/a | n/a | FAIL (CAAR t=+0.416, not significant) |

**Russell-1 note:** Gross/Net/Sharpe/MaxDD are the abnormal-return-vs-IWM series across 8 annual cycles (2016–2023), not a quarterly L/S strategy — comparability with the rows above is limited (see Section 4 detail). Sharpe/MaxDD are computed post-hoc for completeness; the trial's actual significance test is the CAAR t-stat, matching H1's convention. IC/Beta are n/a — this is a long-only basket-vs-benchmark event study, not a cross-sectional rank strategy.

**S3-A cost estimate:** Not explicitly stored in trial_registry; estimated from annual cost structure. Bid-ask and borrow costs are lower at annual frequency due to reduced turnover, approximately 320 bps/year based on the decay pattern from quarterly (390 bps) to semiannual (318 bps).

**S1 IVOL annualization note:** The trial_registry stores ann_return = −10.93% using arithmetic annualisation (mean_quarterly × 4). The geometric (compounded) equivalent of the v2 period series is approximately −14.2%. The Sharpe ratio (−0.464) is derived from the arithmetic convention and is self-consistent with the stored obs_sr_q × sqrt(4).

---

## 4. Per-Trial Detail

### E1 — M1 Insider Conviction [BUG-PENDING]
- **Hypothesis:** Insider buying signal (Form 4, open-market purchases) predicts outperformance in small/mid-cap
- **Signal:** Net insider purchase score from SEC Form 4
- **N periods:** 24 quarterly (2015-Q1 to 2021-Q1, BUT period 24 contaminated — hold_end=2025-10-01)
- **Ann. return:** −0.84% | **Sharpe:** −0.075 | **IC:** +0.0002 (t=0.02) | **Max DD:** −32.8%
- **Win rate:** 50.0% (12/24 quarters positive)
- **DSR:** obs_sr_q = −0.037, threshold = 0.138 → FAIL
- **DSR threshold formula check:** N=4, n=24 → norm.ppf(0.75)/sqrt(24) = 0.6745/4.899 = 0.1377 ✓
- **Source:** `mgmt_pit/results/trial_registry.csv`, `mgmt_pit/results/in_sample_perf.csv`
- **Benchmark (EY L/S):** ann. −11.68%, Sharpe −0.571 (E1 "outperforms" benchmark only because benchmark is also a large loser)

### E2 — M2 Non-Dilution [BUG-PENDING]
- **Hypothesis:** Firms that do not dilute existing shareholders (no share issuance, buybacks preferred) outperform
- **Ann. return:** −5.46% | **Sharpe:** −0.169 | **IC:** +0.0186 (t=1.82) | **Max DD:** −63.548%
- **Win rate:** 33.3% (8/24 quarters positive)
- **DSR:** obs_sr_q = −0.084, threshold = 0.138 → FAIL
- **Source:** `mgmt_pit/results/trial_registry.csv`
- **Note:** IC t=1.82 is the highest of all E trials, but the strategy loses badly during drawdown periods (COVID crash destroyed period-23 return significantly).

### E3 — M3 ROIC Improvement [BUG-PENDING]
- **Hypothesis:** Firms with improving return on invested capital over trailing year outperform
- **Ann. return:** −12.95% | **Sharpe:** −0.560 | **IC:** −0.0115 (t=−0.86) | **Max DD:** −67.4%
- **Win rate:** 45.8% (11/24 quarters positive)
- **DSR:** obs_sr_q = −0.280, threshold = 0.138 → FAIL
- **Source:** `mgmt_pit/results/trial_registry.csv`
- **Note:** Negative IC. ROIC improvement is not a reliable signal in this universe/period. The bug period (period 24) returned M3_solo_ls = −56.7%, the worst single period across all E trials.

### E4 — Composite M1+M2+M3 [BUG-PENDING]
- **Hypothesis:** Composite quality score (rank average of M1, M2, M3) is more robust than any single signal
- **Ann. return:** −2.09% | **Sharpe:** −0.113 | **IC:** — (not computed for composite) | **Max DD:** −41.8%
- **Win rate:** 37.5% (9/24 quarters positive)
- **DSR:** obs_sr_q = −0.057, threshold = 0.138 → FAIL
- **Source:** `mgmt_pit/results/trial_registry.csv`
- **Note:** Composite benefit is modest; the negative M3 drags the composite below M1/M2 average.

### H1 — Livestock Disease / Animal Pharma Event Study
- **Hypothesis:** WAHIS disease notification events produce positive post-event drift in listed animal-pharma tickers
- **Primary test (HPAI, all markets):** CAR_1_20 = −2.11% (t=−1.26), CAR_1_40 = −0.84% (t=−0.41), CAR_1_60 = −3.30% (t=−1.19) — all negative, none significant
- **Secondary test (all 6 diseases pooled, all markets):** CAR_1_20 = −2.09% (t=−2.16), **CAR_1_40 = −2.63% (t=−2.10)**, CAR_1_60 = −4.52% (t=−2.75)
- **Chat registry reported:** CAAR pooled −2.63% (t=−2.10) — this is the 40-day all-livestock result ✓ MATCH
- **Stop criterion:** CAAR negative at 20/40/60 days across all specifications. Wrong sign (hypothesis required positive drift). STOP — no path to salvage.
- **Universe:** PAHC, ELAN, NEOG (3 US names); international tickers not available on Tiingo
- **Events:** 118 total, diseases: ASF, BTV, FMD, HPAI, NCD, PRRS; 2015-01-20 to 2025-01-15
- **Source:** `livestock_pead/output/results.csv`, `livestock_pead/README.md`
- **Note:** Thin universe is the core structural weakness. Conclusion does not depend on the CAAR sign — even if positive, 3 tickers is not a tradable strategy.

### S9 — Cross-Sectional Momentum 12-1 (JT)
- **Hypothesis:** 12-month-minus-1-month cumulative return predicts 3-month forward return (Jegadeesh-Titman)
- **Role:** Pipeline validator. If the harness cannot replicate the most replicated factor, it signals a pipeline bug.
- **Ann. return (gross):** −4.87% | **Sharpe:** −0.324 | **IC:** −0.0157 (t=−0.82) | **Max DD:** −45.3%
- **Market beta:** −0.063 | **Win rate:** 45.8% (11/24 quarters)
- **DSR:** obs_sr_q = −0.162, threshold = 0.197 (N=6) → FAIL
- **Cost modeling:** Not performed (signal was for validation, not production)
- **Source:** `momentum_jt/results/s9_period_returns.csv`, `research/trial_registry.csv` row 4
- **Note:** Momentum's failure is known in small-caps and was especially acute around 2020. Negative IC confirms the signal was inverting in this universe. The pipeline is not at fault — subsequent strategies produced expected IC signs in their respective directions.

### S6 — Residual Momentum 12-1
- **Hypothesis:** Beta-stripped 12-1 momentum (36-month market-model residuals) outperforms raw momentum by removing market exposure
- **Momentum family with S9 (DSR #6):** Not a new independent trial; reuses S9's threshold
- **Ann. gross:** +4.31% | **Cost:** 762 bps/yr | **Ann. net:** −3.34% | **Sharpe:** −0.156 | **Max DD:** −27.4%
- **IC:** −0.0129 (t=−0.70) | **Beta:** +0.067 | **Win rate:** 58.3% (14/24 quarters)
- **DSR:** obs_sr_q = −0.078, threshold = 0.197 (shared with S9, N=6) → FAIL
- **Source:** `residual_momentum/results/s6_period_returns.csv`, `research/trial_registry.csv` row 14
- **Note:** Residual stripping improved the MaxDD (−27.4% vs S9's −45.3%) and Sharpe (−0.156 vs −0.324), but did not flip the signal. Both raw and residual momentum are negative IC in this universe.

### S1 — IVOL Low-Lottery [v2 corrected run]
- **Hypothesis:** Low-IVOL small/mid-cap stocks earn positive risk-adjusted returns vs high-IVOL lottery stocks
- **Construction:** Beta-neutral (long_$ × β_L = short_$ × β_S), quarterly rebalance, EW within quintile
- **IVOL/MAX family with S4 (DSR #7):** Structural siblings, counted once
- **Ann. return (arithmetic):** −10.93% | **Sharpe:** −0.464 | **IC:** −0.0646 (t=−2.27) | **Max DD:** −70.8%
- **Market beta:** −0.220 | **Mean long-scale:** 1.234 | **Mean short-scale:** 0.766
- **Win rate:** 41.7% (10/24 quarters) | **DSR:** obs_sr_q = −0.232, threshold = 0.218 (N=7) → FAIL
- **Source:** `ivol/results/ivol_period_returns.csv`, `research/trial_registry.csv` row 5
- **v1 values (SUPERSEDED):** Sharpe −0.560, beta −0.655, MaxDD −80.0%, ann. −15.97%. These are stale and should not be used.
- **Key 2020 result:** Q1-2020 ls_ret = −27.2%, Q2-2020 = −45.1%. The COVID crash caused the long leg (low-IVOL / low-beta) to fall more in percentage terms than short leg in this universe — opposite the expected direction. This drove the bulk of the drawdown.
- **Failure mode:** Negative IC indicates the low-IVOL signal inverts (high-IVOL actually outperforms) in this small/mid-cap universe and sample period. Additionally, structural asymmetric beta meant even the corrected v2 retained some net-short-beta exposure.

### S4 — MAX Lottery Effect
- **Hypothesis:** Stocks with extreme recent single-day returns (high MAX) are overpriced lottery tickets that underperform
- **Construction:** Beta-neutral, **monthly** rebalance (21-day MAX signal decays within weeks), EW within quintile
- **IVOL/MAX family with S1 (DSR #7):** Not a new independent trial
- **Ann. gross:** +4.76% | **Cost:** 2,202 bps/yr (bid-ask 2,084, borrow 118) | **Ann. net:** −16.06%
- **Sharpe:** −1.264 | **IC:** +0.0341 (t=2.463) | **Max DD:** −65.4% | **Beta:** +0.019
- **Win rate:** 37.5% (27/72 months) | **n_periods:** 72 (monthly)
- **DSR:** per_period_sr = −0.365, threshold = 0.126 (N=7, n=72) → FAIL
- **Source:** `max/results/max_period_returns.csv`, `research/trial_registry.csv` row 6
- **Decay study results:** 1-month hold (identical to S4 primary run), 2-month hold (Sharpe −0.804), 3-month hold (Sharpe −0.393). Cost drag falls at longer holds but not enough — all fail. See `max/results/max_decay_period_returns.csv`.
- **Failure mode:** Gross alpha is positive (+4.76%) and IC is statistically significant (+0.034, t=2.46), confirming signal validity. Strategy is cost-nonviable at any tested rebalance frequency: monthly rebalance requires very high turnover (>65%/period).

### S3-Q — GP/A Quarterly (Trial #8)
- **Hypothesis:** High gross profit-to-assets firms outperform low GP/A firms (Novy-Marx 2013) in small/mid-cap
- **Construction:** Beta-neutral, quarterly, top/bottom 20%, EW within quintile
- **Ann. gross:** +4.93% | **Cost:** 390 bps/yr | **Ann. net:** +0.94% | **Sharpe:** +0.134
- **IC:** +0.0404 (t=2.283) | **Max DD:** −28.3% | **CALMAR:** 0.033 | **Beta:** −0.103
- **Win rate:** 50.0% (12/24 quarters) | **n_periods:** 24
- **DSR:** per_period_sr = 0.069, threshold = 0.235 (N=8, n=24) → FAIL
- **Source:** `gp/results/gp_period_returns.csv` (identical to `gp_decay_quarterly.csv`), `research/trial_registry.csv` row 10
- **Note:** Signal IC is statistically significant (t=2.28) confirming GP/A predicts quarterly returns in this universe. Strategy fails DSR because cost drag (390 bps) erodes the ~490 bps gross alpha to only ~94 bps net. Net Sharpe 0.134 does not clear the 0.235 threshold.

### S3-SA — GP/A Semiannual (Frequency Sweep, Trial #9, Not New Independent Trial)
- **Hypothesis:** Same GP/A signal at semiannual rebalance reduces cost drag enough to clear DSR
- **Construction:** Identical to S3-Q except 6-month hold period and lower turnover
- **Ann. gross:** +10.16% | **Cost:** 318 bps/yr | **Ann. net:** +6.85% | **Sharpe:** +0.569
- **IC:** +0.0541 (t=1.697) | **Max DD:** −17.9% | **CALMAR:** 0.383 | **Beta:** −0.078
- **Win rate:** 66.7% (8/12 periods) | **n_periods:** 12
- **DSR:** per_period_sr = 0.403, threshold = 0.332 (N=8, n=12) → **PASS**
- **Source:** `gp/results/gp_decay_semiannual.csv`, `research/trial_registry.csv` row 11
- **IMPORTANT CAVEAT:** S3-SA is the best outcome of a three-frequency sweep (quarterly, semiannual, annual). Selecting the frequency that clears is a multiple-comparison within the sweep. The run_decay.py code explicitly registers only clearing variants. This pass should be treated as a strong in-sample indication that a lower-frequency execution structure is viable, but the out-of-sample bar is higher given the selection.

### S3-A — GP/A Annual (Frequency Sweep, Trial #10, Not New Independent Trial)
- **Hypothesis:** Annual rebalance further reduces cost drag; tests if IC persists at 12-month horizon
- **Construction:** Identical to S3-Q except annual hold period
- **Ann. gross:** +8.61% | **Cost:** ~321 bps/yr (estimated) | **Ann. net:** +5.92% | **Sharpe:** +0.361
- **IC:** +0.0667 (t=1.115) | **Max DD:** −18.3% | **n_periods:** 6 annual periods
- **DSR:** per_period_sr ≈ 0.242, threshold = 0.470 (N=8, n=6) → FAIL
- **Source:** `gp/results/gp_decay_annual.csv` (all numbers verified from this file; **no trial_registry entry exists**)
- **Note:** IC increases with holding period (quarterly 0.040 → semiannual 0.054 → annual 0.067), consistent with GP/A being a slow-moving signal. Net return is positive (+5.92%) but the DSR threshold is high for n=6 periods and the Sharpe does not clear. Negative year distribution: 2015 (−10.6%), 2016 (−18.3%), confirming losing streaks persist at annual frequency (not just quarterly noise).

### S2 — PEAD (Post-Earnings Announcement Drift)
- **Hypothesis:** High earnings surprise (3-day abnormal return around 8-K filing) predicts 45-day forward drift
- **Construction:** Calendar-time portfolio, daily rebalance, beta-neutral, Q1 long / Q5 short, 45-day hold
- **Ann. gross:** −4.74% | **Cost:** 4,079 bps/yr (bid-ask dominant) | **Ann. net:** −36.59%
- **Sharpe:** −2.864 | **IC:** +0.0368 (t=7.555) | **Max DD:** −93.2% | **Beta:** −0.104
- **Win rate:** 16.7% (12/72 months) | **n_periods:** 72 (monthly calendar-time)
- **n_events:** 42,119 | **Hold days:** 45
- **DSR:** obs_sr_monthly = −0.827, threshold = 0.157 (N=11, n=72) → FAIL
- **Source:** `pead/results/pead_caar.csv`, `pead/results/pead_daily_returns.csv`, `pead/results/pead_events.csv`, `research/trial_registry.csv` row 12
- **Note:** IC is highly significant (t=7.55) confirming PEAD signal is real in this sample. The strategy fails entirely because gross return itself is negative (−4.74%/yr) BEFORE costs. The earnings-drift effect in this small/mid-cap universe apparently does not materialise as a tradable L/S return, or is captured/reversed before the 45-day measurement. Costs (4,079 bps/yr) then compound the disaster — ~7,000 events/year × 45-day holds creates enormous turnover.

### S5 — Asset Growth
- **Hypothesis:** Firms with high YoY total-asset growth subsequently underperform (Cooper et al. 2008)
- **Construction:** Beta-neutral, quarterly, bottom-quintile long (lowest growth) / top-quintile short (highest growth)
- **Ann. gross:** +0.62% | **Cost:** 613 bps/yr | **Ann. net:** −5.46% | **Sharpe:** −0.340
- **IC:** −0.0099 (t=−0.517) | **Max DD:** −34.8% | **Beta:** −0.160
- **Win rate:** 45.8% (11/24 quarters) | **n_periods:** 24
- **DSR:** per_period_sr = −0.170, threshold = 0.282 (N=12, n=24) → FAIL
- **Source:** `asset_growth/results/asset_growth_period_returns.csv`, `research/trial_registry.csv` row 13
- **Note:** Near-zero gross return (+0.62%) with negative IC (−0.010, t=−0.52) indicates the asset growth signal is essentially null in this universe and period. Pre-registered single frequency (no sweep, unlike S3) to avoid the lesson from S3-GP's frequency selection.

### S7 — IVOL-Conditional Value (Double-Sort)
- **Hypothesis:** Within high-IVOL stocks, cheap firms (high E/P) outperform expensive firms (low E/P) — value premium concentrated where arbitrage is most limited
- **Construction:** Top-tercile IVOL fixed, then value tercile within that subset; beta-neutral (reused IVOL regression beta); quarterly
- **Ann. gross:** −6.52% | **Cost:** 882 bps/yr | **Ann. net:** −14.70% | **Sharpe:** −0.544
- **IC:** +0.0358 (t=1.356) | **Max DD:** −70.1% | **Beta:** −0.085
- **Win rate:** 50.0% (12/24 quarters) | **n_periods:** 24
- **DSR:** per_period_sr = −0.272, threshold = 0.291 (N=13, n=24) → FAIL
- **Source:** `ivol_value/results/s7_period_returns.csv`, `research/trial_registry.csv` row 15
- **Note:** The double-sort reduces the effective universe substantially (top-tercile IVOL subset only). IC is positive but weak (t=1.36). Gross return is negative, suggesting the interaction hypothesis (value × IVOL) does not produce reliable alpha in this universe.

### Russell-1 — Pre-Effective-Date Anticipatory Drift (Trial #17, New Independent Family: Russell Reconstitution)
- **Hypothesis:** Stocks confirmed for addition to a Russell US Index (Microcap→2000 or 2000→1000 migration, or new-to-index via IPO) drift upward between the preliminary-list announcement date and the reconstitution effective date, as traders anticipate forced mechanical buying from index-benchmarked funds. Pure-flow hypothesis, no fundamental mechanism.
- **Design:** Event study (same class as H1), not a Sharpe/DSR-gated L/S strategy. Equal-weight, long-only basket of ALL confirmed Russell 3000 additions per cycle. Entry: first trading day after that year's preliminary list was publicly posted. Exit: the historical effective date (last Friday of June, except 2023's schedule shift to June 23). Benchmark: IWM (iShares Russell 2000 ETF) adjClose return over the identical window.
- **Cycles:** 8 annual cycles, 2016–2023 (in-sample). 2024–2025 reserved untouched holdout, NOT unlocked (see recommendation below). **2015 excluded** — the only recoverable documents at `russell.com/documents/indexes/*.pdf` are press-release "highlights" summaries (6–7 rows of prose), not ticker-level tables, confirmed by direct parse. *Addendum:* a later session found a working alternate source for 2015 (an archived HTML tool page with real `<table>` elements, not a PDF) — 147 R3000 additions / 153 deletions / 244 Microcap additions recovered and cached at `research/russell_reconstitution/cache/2015_addendum/`, but **not yet wired into the backtest** (would need a documented proxy decision, since only a final list — no preliminary — survives for that year).
- **CAAR (gross, abnormal vs IWM):** +0.6374% | **t-stat:** +0.416 | **two-sided p-value:** 0.6900 | **CAAR (net of 50bps round-trip):** +0.1374%
- **Win rate:** 62.5% (5/8 cycles)
- **Sharpe (net, abnormal-vs-IWM):** +0.032 | **CALMAR:** +0.014 | **Max drawdown:** −10.2% — computed post-hoc for reference only. Not a native fit: 8 non-overlapping annual points with ~2–4 week holding windows each (no continuous market exposure between cycles), so these ratios carry unusually little statistical weight. An absolute-basket-return variant (not vs. benchmark) gives Sharpe −0.057 / CALMAR −0.024 / MaxDD −15.9% — reported here for completeness since the two framings disagree in sign.
- **Cost model:** Flat 50bps round-trip per name, charged once per cycle (one entry + one exit; buy-and-hold, no rebalancing) — deliberately lighter than the $100M–$2B core universe's cap-tiered formula (up to 100bps ceiling) since Russell-eligible names are index-liquidity-screened. See `run_backtest.py` docstring for full rationale.
- **Coverage:** mean 87.4% of confirmed additions priced (Tiingo); mean 38.9% of confirmed additions fall inside the standard $100M–$2B core-program universe — Russell 3000 additions span a much wider market-cap range than the rest of this program's screened universe, by design (this trial intentionally tests the full R3000 addition set, not just the $100M–$2B slice).
- **Falsification diagnostics:** Concentration — 2022 is the largest single-cycle contributor (34.5% of total |abnormal return|, a −15.4% basket / −8.9% abnormal return that year) but stays under the 40% flag threshold. Decay check — corr(year, abnormal_ret) = −0.152, flat/no clear crowding-out trend over 2016–2023. Sign-convention audit — basket built exclusively from `list_kind='additions'` files, no deletions used; PASS by construction.
- **Source:** `research/russell_reconstitution/results/russell1_cycle_results.csv`, `research/russell_reconstitution/results/russell1_ticker_returns.csv`, `research/trial_registry.csv` (study=`russell1_anticipatory_drift`, trial #17). Data acquisition: FTSE Russell primary-source PDFs archived via Wayback Machine CDX (`research/russell_reconstitution/acquire.py`, `parse.py`, `cycle_calendar.py` — full sourcing notes in `research/russell_reconstitution/README.md`).
- **Verdict / recommendation:** CAAR t=+0.416 is far from the 1.96 significance threshold — no reliable pre-effective-date drift detected in-sample. **Recommendation: do NOT unlock the 2024–2025 holdout** — this baseline shows nothing to confirm. This result is also the load-bearing prerequisite check for Russell-2 (post-effective-date reversal): a reversal trade requires a confirmed overshoot to reverse, and none was found, so Russell-2 was not run.

---

## 5. S1–S10 Program Status

| Signal | ID | Status | Notes |
|--------|-----|--------|-------|
| IVOL | S1 | Complete, FAIL DSR | v2 corrected (beta-neutral); prior chat numbers stale |
| PEAD | S2 | Complete, FAIL DSR | Gross negative; cost catastrophic |
| GP/A | S3 | Complete, FAIL (quarterly, annual); **PASS (semiannual)** | Best-of-3 selection; walk-forward deferred |
| MAX | S4 | Complete, FAIL DSR | Cost-nonviable; decay study also fails |
| Asset Growth | S5 | Complete, FAIL DSR | Null signal in this sample |
| Residual Momentum | S6 | Complete, FAIL DSR | Momentum family with S9 |
| IVOL-conditional Value | S7 | Complete, FAIL DSR | Double-sort; gross negative |
| Residual Short-Term Reversal | S8 | **PENDING** | No results files exist |
| Cross-sectional Momentum 12-1 | S9 | Complete, FAIL DSR | Failed as pipeline validator |
| Amihud Illiquidity | S10 | **PENDING** | No results files exist |

---

## 6. Open Items

1. **E1–E4 period-24 bug:** Re-run required. `mgmt_pit/results/in_sample_perf.csv` row 25 must be corrected (hold_end = 2021-01-01, not 2025-10-01) and all trial_registry.csv E-series numbers regenerated. The direction of all results (negative Sharpe, DSR FAIL) is almost certainly unchanged given the strength of the losses in 23 valid periods, but the exact numbers are unreliable.

2. **S3-SA walk-forward:** Only clearing strategy to date. Walk-forward window (2021-01-01 onwards) has not been run. This is the sole in-sample pass and the logical next step before any capital allocation.

3. **S8 and S10 not yet run:** Residual short-term reversal (S8) and Amihud illiquidity (S10) are coded but no results files exist.

4. **S3-A registry entry missing:** `gp/results/gp_decay_annual.csv` was saved but no trial_registry entry was written. The run_decay.py `maybe_log_registry()` function only writes entries for clearing variants (line 489: `clearing = [s for s in summaries if s["clears_dsr"] and s["ppy"] != 4]`). Annual did not clear so it was not logged. All S3-A numbers in this registry were verified directly from the data file.

5. **S1 IVOL v1 run no longer in trial_registry:** The v2 corrected run replaced the v1 entry in-place. The pre-correction S1 numbers (Sharpe −0.560, β −0.655, MaxDD −80.0%, ann. −15.97%) are documented only in the `ivol/run.py` code comment. They should not be cited going forward.

---

## 7. Pipeline Learnings

1. **Cost drag is the dominant kill factor in small-cap L/S:** Six of thirteen trials have positive gross alpha (S4, S3-Q, S3-SA, S3-A, S2, S5 marginally), but four of those fail due to cost drag alone. Monthly and daily rebalance is essentially infeasible for the current cost model.

2. **Momentum inverts in small-caps:** S9 failed as a pipeline validator, indicating the momentum factor is unreliable in sub-$2bn US equities. S6 confirmed this with residual stripping. Momentum-based strategies require large-cap or long-short with longer holds.

3. **Beta-neutral construction is mandatory for IVOL/MAX family:** S1 v1's β = −0.655 was a structural portfolio construction error that contaminated all results. The v2 correction reduces this to β = −0.220, materially changing the return and drawdown profile.

4. **Frequency sweep (S3) is a valid robustness tool but creates selection risk:** Running three frequencies and registering only the clearer (semiannual) is internally consistent given the pre-announced "not a new trial" rule, but it creates a best-of-3 selection effect that the walk-forward must account for.

5. **High-IC ≠ positive gross return (S2 PEAD):** A signal with IC t=7.55 can still produce negative gross returns in a calendar-time implementation. The disconnect likely comes from the interaction between beta-neutral sizing, event-driven entry/exit, and the specific periods when earnings events cluster (e.g., 2020-Q1/Q2 earnings clustered with COVID crash positions).

6. **E1–E4 management quality signals underperform their own benchmark:** The EY L/S benchmark (ann. −11.68%, Sharpe −0.571) also loses money. E1–E4's "outperformance" vs benchmark is purely an artifact of a losing benchmark, not evidence of alpha. E3 and benchmark both have Sharpe ≈ −0.56.

7. **Small n_periods is the binding constraint for annual-rebalance strategies:** S3-A has only 6 annual periods, giving a DSR threshold of 0.470 — effectively requiring a Sharpe of 0.47+ just to pass the multiple-comparisons gate. With inherent return volatility at annual frequencies, this is a very high bar.
