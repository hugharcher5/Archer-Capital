# CORRECTED TRIAL REGISTRY

**This is the single canonical narrative registry for this research program.**
Before creating any new registry-style file (`*_TRIAL_REGISTRY.md`,
`RESULTS_SUMMARY.md`, or similar), check this file and `research/trial_registry.csv`
first — do not create a fourth. See the consolidation note below and the root
`CLAUDE.md` for the canonical-file rule.

**Generated:** 2026-07-06 · **Last consolidated:** 2026-07-12 (backfilled through trial #26) · **Updated:** 2026-07-14 (added trial #27, S21; trial #28, S22)
**Verified against:** source files in `/mnt/c/Users/user/Archer-Capital/research/`, cross-checked against `research/trial_registry.csv`
**In-sample window:** 2015-01-01 → 2021-01-01 for the original 15-trial program (individual trials from #14 onward use their own windows — noted per trial; several extend to 2023 or use annual-cycle designs)
**DSR gate:** Deflated Sharpe Ratio, cumulative independent-trial count
**Universe:** US small/mid-cap equities (SEC XBRL + Tiingo), except where a trial's detail section notes a different universe (e.g. Russell-family event studies, gpvolmanaged's Russell confirmed-exit overlay)

> **2026-07-12 consolidation note:** This document, `research/PHASE3_TRIAL_REGISTRY.md`
> (created 2026-07-10, contained only trial #23), and `research/RESULTS_SUMMARY.md`
> (an earlier, now-superseded draft covering trials through #15) were found fragmented
> across three files. This file is designated canonical (most complete, most actively
> maintained, contains the corrected 15-trial program plus prior open-ended-phase
> addenda). All content from the other two has been merged in below; both are now
> marked `SUPERSEDED` at the top and should not be edited further. `research/trial_registry.csv`
> remains the authoritative machine-readable source for every number — this document is
> the narrative companion and is now current through **trial #27** (all trials, including
> the ones previously missing full write-ups: S8, S10, S11, the S3-SA walk-forward, S14,
> S15, S13, both S16 trials, S17, S18, S19, S20, S21, and S22). If a trial-number gap appears here in
> the future, treat `trial_registry.csv` as ground truth and backfill this file — don't start a new one.

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
| 14 | S8 | Residual short-term reversal | 14 | 70 (monthly) | 0.1751 | 0.607 | FAIL |
| 15 | S10 | Amihud illiquidity (cost-model stress test) | 15 | 24 (quarterly) | 0.3064 | 0.613 | FAIL (wrong-signed IC) |
| 16 | S11 | Total accruals (Sloan) | 16 | 24 (quarterly) | 0.3132 | 0.626 | FAIL |
| 8b-WF | S3-SA walk-forward | Out-of-sample validation of trial #8b (not a new trial, N_TRIALS unchanged at 8) | 8 | 8 (semiannual) | 0.3321 | 0.470 | FAIL (reference only — DSR is an in-sample construct) |
| 17 | Russell-1 | Pre-effective-date anticipatory drift (new independent family: Russell Reconstitution) | 17 | 8 (annual cycles) | n/a (event study) | n/a (event study) | FAIL (CAAR t=+0.416, not significant) |
| 17 | Russell-3 | Boundary-crossing subset (family member of Russell Reconstitution, shares DSR #17) | 17 | n/a (event study) | n/a | n/a | see detail — shares slot with Russell-1 |
| 18 | S14 | Analyst estimate revision — STOPPED at Phase 1 (no usable data source) | 18 | n/a (no backtest run) | n/a | n/a | STOP (data infeasibility, not a performance result) |
| 19 | S15 | Composite of GP/A + S8 + S7 (rank-percentile blend) | 19 | 24 (quarterly) | 0.2967* | 0.661 | FAIL |
| 20 | S13 | Opportunistic insider cluster buying (E1 reformulation) | 20 | 13 (quarterly, coverage-limited) | 0.4562 | 0.912 | FAIL |
| 21 | S16-multisleeve | Three-sleeve blend (GP/A + S8 + S7-original), separate-book construction | 21 | 24 (quarterly) | 0.3406 | 0.681 | FAIL |
| 22 | S16-gpindneutral | Industry-neutral GP/A reformulation (SIC-major-group ranking) | 22 | 24 (quarterly) | 0.3451 | 0.690 | FAIL |
| 23 | S17-gpvolmanaged | Volatility-managed GP/A (Moreira & Muir overlay) | 23 | 32 (quarterly) | 0.3026 | 0.605 | FAIL |
| 24 | S18-pairs | Small-cap sector-bucketed pairs trading (FDR + persistence filtered) | 24 | 0 (zero trades) | n/a | n/a | STOP (0 pairs survived the persistence filter — no return series) |
| 25 | S19 | 52-week-high anchoring (George & Hwang 2004) | 25 | 72 (monthly) | 0.2063 | 0.715 | FAIL |
| 26 | S20 | Institutional 13F "smart money" accumulation | 26 | n/a (Phase 1 stop) | n/a | n/a | STOP (CUSIP crosswalk vendor quota exhausted, not a data-existence gap) |
| 27 | S21 | Cluster-conditioned short-term mean reversion (basket Z-score) | 27 | 60 (monthly, basket book) | 0.2306 | 0.799 | FAIL (per-period SR −5.026 — real gross Sharpe 0.95, destroyed by ~110%/yr cost drag) |
| 28 | S22 | Long-only valuation portfolios: DCF/P-E/P-S/P-B (4 sub-trials, shared slot) | 28 | 24 (quarterly) | 0.3680 | 0.736 | All FAIL, both original AND fairness-corrected intersection-universe re-run — all four statistically indistinguishable once compared fairly (see Section 4) |

*S15's own per-period-SR threshold is `norm.ppf(1-1/19)/sqrt(24)` ≈ 0.297; the trial's actual per-period SR (−0.035) fails by a wide margin regardless of the exact threshold value quoted in different source notes.

**All trials through #26 are now reflected in this table** (previously #14–16, #18–22 were logged only to `research/trial_registry.csv` — see the 2026-07-12 consolidation note at the top of this document). S16 appears twice under two distinct DSR slots (#21 multi-sleeve blend, #22 industry-neutral GP/A) because they are two genuinely independent reformulations that happened to share a hypothesis-family label in their `study` field, not a shared trial family — each was verified against the registry as its own new trial at logging time.

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
| Russell-3 | +0.31%/event (pooled) | n/a | n/a | n/a | n/a | n/a | n/a | n/a | FAIL (pooled CAAR t=+0.615, not significant) |
| S8 | −3.08% | 761 | −10.22% | −1.107 | −47.4% | +0.0360 | 3.397 | −0.011 | FAIL (−0.320 < 0.175, monthly n=70) |
| S10 | −7.23% | 617 | −12.97% | −1.129 | −51.7% | −0.0530 | −4.604 | −0.151 | FAIL (−0.565 < 0.306) [wrong-signed IC] |
| S11 | −1.11% | 432 | −5.35% | −0.336 | −40.5% | +0.0235 | 1.682 | +0.046 | FAIL (−0.168 < 0.313) |
| S3-SA-WF | +8.51% | 344 | +5.04% | +0.449 | −13.5% | +0.0588 | 2.385 | +0.403 | FAIL (0.317 < 0.332) [walk-forward, reference only] |
| S14 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | STOP (no usable data — not a performance result) |
| S15 | +6.30% | 801 | −1.85% | −0.070 | −32.6% | +0.0286 | 1.644 | +0.093 | FAIL (−0.035 < 0.297) |
| S13 | −0.74% | 780 | −8.26% | −0.857 | −27.0% | +0.0138 | 0.850 | +0.056 | FAIL (−0.428 < 0.456, n=13 coverage-limited) |
| S16-multi | −0.64% | 668 | −7.12% | −0.740 | −39.1% | n/a (blend of 3 sleeves) | n/a | −0.058 | FAIL (−0.370 < 0.341) |
| S16-indneu | +7.49% | 471 | +2.63% | +0.270 | −20.0% | +0.0336 | 2.265 | −0.057 | FAIL (0.135 < 0.345) |
| S17-volmgd | +7.57% | 273 | +4.72% | +0.481 | −14.3% | +0.0458 | 2.699 | +0.056 | FAIL (0.241 < 0.303) |
| S18-pairs | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | STOP (0 trades — see diagnostic supplement, not registered) |
| S19 | −8.37% | 1,305 | −19.77% | −0.908 | −75.7% | +0.0044 | 0.238 | −0.020 | FAIL (−0.262 < 0.206) |
| S20 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | STOP (Phase 1, CUSIP crosswalk vendor quota — no backtest run) |
| S21-basket | +5.78% | 10,999 | −66.44% | −17.411 | −99.5% | +0.0207 | 1.566 | +0.090 | FAIL (−5.026 < 0.231) — gross Sharpe 0.954 alone would clear the promotion bar |
| S21-peer | +5.85% | 9,888 | −62.11% | −9.395 | −99.2% | +0.0393 | 1.605 | +0.074 | FAIL (secondary construction, same pattern) |
| S22-DCF | +12.54% | 180 | +10.59% | +0.452 | −45.5% | +0.0014 | 0.07 | +1.010 | FAIL (0.226 < 0.368) |
| S22-PE | +9.85% | 186 | +7.88% | +0.388 | −61.9% | +0.0132 | 0.57 | +1.221 | FAIL (0.194 < 0.368) |
| S22-PS | +14.49% | 144 | +12.91% | +0.493 | −57.3% | +0.0210 | 0.65 | +1.279 | FAIL (0.246 < 0.368) |
| S22-PB | +9.64% | 179 | +7.72% | +0.389 | −58.5% | +0.0040 | 0.16 | +1.246 | FAIL (0.194 < 0.368) |

**Russell-1 note:** Gross/Net/Sharpe/MaxDD are the abnormal-return-vs-IWM series across 8 annual cycles (2016–2023), not a quarterly L/S strategy — comparability with the rows above is limited (see Section 4 detail). Sharpe/MaxDD are computed post-hoc for completeness; the trial's actual significance test is the CAAR t-stat, matching H1's convention. IC/Beta are n/a — this is a long-only basket-vs-benchmark event study, not a cross-sectional rank strategy.

**Russell-3 note:** Event-level design (1,493 priced crossing events from 748 repeat-crosser tickers), not a per-cycle basket — Gross/Net/Sharpe/MaxDD/IC/Beta columns don't apply cleanly and are shown as n/a; the CAAR t-stat is the actual significance test, reported pooled (additions+deletions together). See Section 4 detail for the additions-only and deletions-only subset breakdowns, both of which were also wrong-signed relative to the hypothesis.

**S16-multi IC note:** `n/a` in the IC column because S16-multisleeve is a blend of three separately-scored sleeves (GP/A, S8, S7), not a single rank signal — see Section 4 for each sleeve's own IC.

**S8/S10/S11/S13/S15/S16/S17 note:** these trials (#14–16, #19–23) were logged to `research/trial_registry.csv` between 2026-07-06 and 2026-07-10 but only backfilled into this table during the 2026-07-12 consolidation — see the note at the top of this document.

**S18-pairs note:** all columns `n/a` because the pre-registered methodology produced zero trades (no return series to compute Gross/Net/Sharpe/CALMAR/IC/Beta from) — see Section 4 for the full selectivity funnel and the non-registered diagnostic supplement (which bypasses the persistence filter to give illustrative, non-registered numbers only).

**S20 note:** all columns `n/a` because this trial stopped at the Phase 1 feasibility check — no backtest was run. Unlike S14/S18 (data genuinely unavailable/exhausted), S20's underlying 13F data is fully accessible; the CUSIP-to-ticker crosswalk hit a vendor quota wall. See Section 4 for the full diagnosis and the partial coverage-check side observation.

**S22 beta note:** the Beta column shows +1.0 to +1.28 for all four S22 sub-trials — unlike every other row in this table (all beta-neutral by construction, beta≈0 expected), S22 is explicitly **long-only**, so a beta near 1 (matching the broad small-cap market) is the correct, expected result, not an error or a failed neutralization. See Section 4 for the full alpha-vs-market / alpha-vs-T-bill decomposition this makes necessary.

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

### Russell-3 — Boundary-Crossing Subset (Trial #17, Family Member of Russell Reconstitution, Shares DSR Slot with Russell-1)
- **Hypothesis:** Companies that cross the Russell index-inclusion boundary (Microcap↔2000 or 2000↔1000) multiple times across 2016–2023 isolate the pure membership/passive-flow effect more cleanly than Russell-1's full additions list, since each company serves as its own control across its own repeated crossings. A distinct sub-hypothesis, not a re-tuned version of Russell-1 — shares its DSR slot as a family member.
- **Data reuse:** Russell-1's existing sourced dataset (Wayback-archived FTSE Russell PDFs, 2016–2023) plus a new deletions-side universe (`build_russell3.py`), built parallel to Russell-1's additions-only universe. R3000 deletions data is unavailable for 2018/2019 (same pre-existing gap noted elsewhere in this document). No new data acquisition.
- **Sample:** 748 unique repeat-crosser tickers found (appear in 2+ distinct years of the combined additions+deletions universe) — **not** a thin sample, ~39% of the full universe, consistent with well-documented Russell 2000/Microcap boundary churn. 1,751 total crossing events, 1,493 priced.
- **Design:** Event-level (one entry+exit per crossing, not one per year); N reported as crossing events, not unique tickers. Long-only window return minus IWM, same sign convention for additions and deletions (not flipped) — reported separately (pooled/additions-only/deletions-only) since the two subsets are expected to have opposite signs under the hypothesis.
- **Results:**
  - **Pooled (N=1,493):** CAAR = +0.3064%, t = +0.615, win rate 46.0% — not significant.
  - **Additions-only (N=822):** CAAR = −0.6000%, t = −0.958 — **wrong-signed** vs. hypothesis (expected positive).
  - **Deletions-only (N=671):** CAAR = +1.4167%, t = +1.775 (marginal, p=0.076) — **wrong-signed** vs. hypothesis (expected negative).
- **Falsification diagnostics:** Concentration check clean (top-2 tickers <5% of |abnormal return| mass in every subset).
- **Source:** `research/trial_registry.csv` (study=`russell3_boundary_crossing`, trial #17, shares slot with Russell-1).
- **Verdict / recommendation:** No subset is materially different from Russell-1's null result — both wrong-signed subsets undercut the hypothesis rather than merely failing to confirm it. **Recommendation: close the Russell Reconstitution family** — two independent tests (full additions basket, and this self-controlled boundary-crossing refinement) both failed to find the pre-effective-date drift the family's founding hypothesis predicted.

### S8 — Residual Short-Term Reversal (Trial #14)
- **Hypothesis:** Negative 1-month residual return from a rolling 36-month single-factor OLS market-model (same factor construction as S6) predicts a reversal: long bottom decile / short top decile of residual return.
- **Construction:** Pre-registered double filter — decile membership AND |1-month residual| ≥ 1.5× trailing-12-month monthly-residual standard deviation. Turnover discipline: names not crossing both thresholds hold their existing book position with no transaction cost charged (reduces unnecessary churn vs. a naive full-decile-rebuild-every-month design).
- **Ann. gross:** −3.08% | **Cost:** 761 bps/yr (bid-ask 614 + borrow 147) | **Ann. net:** −10.22% | **Sharpe:** −1.107 | **CALMAR:** −0.215
- **IC:** +0.0360 (t=3.397) | **Max DD:** −47.43% | **Beta:** −0.011 (t=−0.276, beta-neutral construction working) | **Mean actual turnover:** 21%/month
- **Win rate:** 30.0% | **n_periods:** 70 (monthly)
- **DSR:** obs_sr_monthly = −0.320, threshold = 0.175 (N=14, n=70) → FAIL
- **Source:** `research/RESULTS_SUMMARY.md` (now superseded, content merged here), `research/trial_registry.csv` row (study=`residreversal_S8`, trial #14)
- **Falsification diagnostics (all clean):** (1) sign-convention audit passed, 0/70 violations — the long leg had lower mean 1-month residual return than the short leg every period; (2) no single month dominated total |return| mass (max 5.2%, well below a 30% flag threshold); (3) liquidity-shock isolation showed no concentration in the least-liquid ADTV tercile (77% of return mass from the mid-liquidity tier).
- **Note:** The IC (t=3.40) is real and statistically significant — the third-highest IC t-stat in the program (below S2's t=7.55, above S4's t=2.46). Yet gross return is negative (−3.08%/yr) before costs, and 761 bps/yr of cost drag at only 21% actual monthly turnover deepens the loss further. This is the **same failure mode as S2 (PEAD)**: a statistically real cross-sectional IC does not guarantee a gross-positive L/S portfolio return — signal content and portfolio implementability are genuinely separable quantities. A naive full-book comparison (all-decile, no vol filter, full monthly churn) produced −20.84%/yr net — the turnover-discipline filter added real value over unfiltered reversal, but could not convert a gross-negative underlying signal into a viable strategy.

### S10 — Amihud Illiquidity (Trial #15, Cost-Model Stress Test)
- **Hypothesis:** Amihud illiquidity premium — long high-ILLIQ (top quintile), short low-ILLIQ (bottom quintile). Pre-registered explicitly as a cost-model stress test for the least-liquid corner of the universe, not expected to pass net of costs — the final signal in the original 15-trial program.
- **Signal:** ILLIQ = mean(|adjClose return| / (raw close × raw volume)) over trailing 21 trading days.
- **Ann. gross:** −7.23% | **Cost:** 617 bps/yr | **Ann. net:** −12.97% | **Sharpe:** −1.129 | **CALMAR:** −0.251
- **IC:** −0.0530 (t=−4.604) | **Max DD:** −51.73% | **Beta:** −0.151 (t=−2.359) | **Mean turnover:** 48%/quarter
- **Win rate:** 29.2% | **n_periods:** 24 (quarterly)
- **DSR:** per_period_sr = −0.565, threshold = 0.306 (N=15, n=24) → FAIL
- **Source:** `research/trial_registry.csv` (study=`illiquidity_S10`, trial #15)
- **Note:** This is **not merely a cost-model finding** — the IC is both statistically significant AND negative (t=−4.6, the second-most-significant IC in the program after S2's PEAD), meaning the illiquidity premium is genuinely **inverted** in this universe: less-liquid names underperformed more-liquid ones, the opposite of the classical Amihud hypothesis. The pre-registered expectation was "gross-positive, cost kills it" (matching S4/S3-Q's cost-nonviable pattern); instead the signal itself is wrong-signed. Combined with a significantly negative market beta (t=−2.36), the long (illiquid) leg may be systematically capturing a different risk factor (small/distressed-name beta) than intended.

### S11 — Total Accruals (Trial #16)
- **Hypothesis:** Sloan (1996) accruals anomaly: (TTM NetIncome − TTM OperatingCashFlow) / Assets. High-accruals firms (earnings driven by accounting adjustments) subsequently underperform; low-accruals firms (cash-backed earnings) subsequently outperform. Long bottom quintile (lowest accruals), short top quintile (highest accruals). Independent family — shares GP/A's slow-moving fundamental-only structural profile but tests earnings QUALITY, not LEVEL (GP/A, S3) or balance-sheet GROWTH (S5).
- **Construction:** NetIncomeLoss/CFO matched by SEC accession number per period (same-filing gating, to avoid mixing data across different-vintage filings for the two flow concepts). Pre-registered single frequency, no sweep (explicit S3 lesson).
- **Ann. gross:** −1.11% | **Cost:** 432 bps/yr | **Ann. net:** −5.35% | **Sharpe:** −0.336 | **CALMAR:** −0.132
- **IC:** +0.0235 (t=1.682) | **Max DD:** −40.50% | **Beta:** +0.046 (t=0.546) | **Mean turnover:** 25%/quarter
- **Win rate:** 33.3% | **n_periods:** 24 (quarterly)
- **DSR:** per_period_sr = −0.168, threshold = 0.313 (N=16, n=24) → FAIL
- **Source:** `research/trial_registry.csv` (study=`accruals_S11`, trial #16)
- **Note:** IC is correctly signed (+0.0235) but sub-significant (t=1.68). Annual-TTM fallback (rather than the preferred quarterly-summed TTM) was needed for 30.0% of signals — above the program's informal 25% quality flag threshold, meaning a meaningful minority of accruals values are up to ~1 year stale. Gross return is near-zero and slightly negative before that cost drag is even applied, consistent with a weak-to-null signal in this universe/period rather than a cost-viability story.

### S3-SA Walk-Forward — Out-of-Sample Validation (Not a New Trial, N_TRIALS Unchanged at 8)
- **Purpose:** S3-SA (semiannual GP/A, trial #8b) was the only in-sample pass in the entire 15-trial program. This is its mandatory out-of-sample check on the reserved 2021-01-01 → 2025-01-01 window — identical signal, universe, portfolio construction, and cost model as the in-sample run, no re-optimization.
- **Ann. gross:** +8.51% | **Cost:** 344 bps/yr | **Ann. net:** +5.04% | **Sharpe:** +0.449 | **CALMAR:** +0.372
- **IC:** +0.0588 (t=2.385) | **Max DD:** −13.5% | **Beta:** +0.403 (t=2.730 — notably higher and now statistically significant, vs. the in-sample −0.078) | **Mean turnover:** 36%/period
- **Win rate:** 50.0% | **n_periods:** 8 (semiannual)
- **DSR (reference only):** per_period_sr = 0.317, in-sample threshold = 0.332 (N=8, n=12) → falls just short. **Important:** DSR is formally an in-sample multiple-testing correction; comparing an out-of-sample Sharpe against the in-sample threshold is done here for reference only, not as a formal pass/fail test.
- **Source:** `research/trial_registry.csv` (study=`gp_S3-SA_walkforward`, trial `8b-WF`)
- **Note:** IC held up well out-of-sample (+0.059 vs. in-sample +0.054) and net Sharpe (+0.449) is broadly consistent with the in-sample result (+0.569), which is reassuring given this is the program's only candidate for further consideration. However, the walk-forward beta (+0.403, t=2.73) is a material and statistically significant shift from the in-sample beta (−0.078, not significant) — the strategy picked up real net-long-market exposure during 2021–2025 that beta-neutral construction did not fully remove, which should be investigated (not just accepted) before treating this as a validated result.

### S14 — Analyst Estimate Revision (Trial #18, STOPPED — No Usable Data Source)
- **Hypothesis:** In thinly-covered small/mid-cap names, analyst EPS estimate revisions are underreacted to, given how much new information a single revision carries relative to consensus at 1–3 covering analysts. A distinct mechanism from S2/PEAD (tests analyst estimate CHANGES, not realized-earnings price-reaction surprise).
- **Status:** **STOPPED at the Phase 1 data-feasibility check, before any backtest was attempted** — logged as a documented dead end, per program convention (same disposition class as a wrong-signed or null result, not a gap in the record).
- **Findings:** Only `FMP_API_KEY` is analyst-estimate-adjacent among this project's configured vendors — no IBES/Refinitiv/FactSet/Zacks/Estimize, and Tiingo (the project's primary vendor) has no analyst estimates at any tier. Live-testing FMP's current subscription found: (1) small/mid-cap symbol coverage is walled off entirely by plan tier (HTTP 402 "not available under your current subscription" for every small/mid-cap ticker tested — SMTC, PLAB, CENT, UFPT, VIAV) — not merely thin, actually blocked; (2) even for accessible large-caps, FMP's `analyst-estimates` endpoint returns only a live current-snapshot forward estimate with no historical vintage/publish-date field — not usable for a point-in-time backtest even where symbol access exists.
- **Side finding (unrelated to this trial):** `dcf/valuation/sources.py`'s `fetch_fmp()` was found to be calling FMP's deprecated `/api/v3/` endpoints (403 "Legacy Endpoint," retired 2025-08-31) — flagged and separately fixed (migrated to `/stable/`) in a follow-up session; see `dcf/valuation/sources.py` and its own commit history for that fix, not part of this trial.
- **Source:** `research/trial_registry.csv` (study=`analystrevision_S14`, trial #18, status=`stopped_no_data`)
- **Verdict:** Kill before backtest. Upgrading the FMP plan tier was not attempted — a paid-subscription decision explicitly left to the user rather than assumed. If revisited, confirm **both** small/mid-cap symbol coverage **and** historical point-in-time estimate vintages exist under any candidate new plan/vendor before resuming — confirming only one is not sufficient.

### S15 — Low-Turnover Composite of Validated-IC Signals (Trial #19)
- **Hypothesis:** An equal-weighted rank-percentile composite of three previously-tested signals with real, correctly-signed standalone IC (GP/A quality — S3-Q #8; residual short-term reversal — S8 #14; IVOL-conditioned value — S7 #13), each of which failed as a standalone strategy for a *different* reason, could achieve genuine diversification benefit even though none clears alone.
- **Construction:** All four portfolios (composite + three "-alone" recomputations) built on the **identical** per-quarter intersected universe (names with valid GP/A, ResidRev, and IVOL-Value signals that quarter) and the same beta-neutral sizing — isolating the diversification effect from any universe/cadence difference. IVOL-Value's original hard double-sort has no natural continuous scalar, so a new `rank_pct(IVOL) × rank_pct(E/P)` full-universe score was built specifically for this composite.
- **Ann. gross:** +6.30% | **Cost:** 801 bps/yr | **Ann. net:** −1.85% | **Sharpe:** −0.070 | **CALMAR:** −0.057
- **IC:** +0.0286 (t=1.644) | **Max DD:** −32.6% | **Beta:** +0.093 (t=1.122) | **Mean turnover:** 59%/quarter
- **Win rate:** 41.7% | **n_periods:** 24 (quarterly)
- **DSR:** per_period_sr = −0.035, threshold ≈ 0.297 (N=19, n=24) → FAIL
- **Component comparison (recomputed on the identical shared universe, not each signal's original broader universe):** GP/A-alone Sharpe +0.007, ResidRev-alone −0.222, IVOLValue-alone −1.119 — composite (−0.070) is *worse* than the best individual component alone.
- **Diagnosed root causes (both confirmed by falsification checks):** (1) the new continuous IVOL-Value score **flipped sign** vs. S7's original double-sort (IC +0.036 → −0.015 here) — S7's edge lived specifically in the conditional/interaction structure (value only within the highest-IVOL tercile), and a full-universe multiplicative score dilutes rather than preserves that structure; (2) GP/A and IVOL-Value were **meaningfully correlated** (+0.563 in-sample), far from the near-zero the diversification thesis assumed — the effective number of independent bets was closer to two than three.
- **Falsification diagnostics:** Concentration clean (top-3 names never exceeded 30% of |return| mass in any period, max 24.1%). Sign-convention audit confirmed the IVOL-Value sign flip as the diagnostic finding (see above). Composite beta (+0.093, t=1.12) not significant, but this masked a significantly positive ResidRev-alone beta (+0.220, t=2.223) being offset by GP/A's and IVOL-Value's negative betas — three beta-neutral legs combined does not guarantee the combination stays beta-neutral by construction.
- **Source:** `research/composite_s15/results/RESULTS_S15.md` (full detail, retained as the source document), `research/trial_registry.csv` (study=`compositeIC_S15`, trial #19)
- **Verdict:** Diversification did not help. Composite Sharpe (−0.070) is worse than GP/A alone (+0.007, still itself far below the DSR bar). FAILS DSR, misses promotion bar (CALMAR ≥ 1 or Sharpe ≥ 0.8) by a wide margin. Direct precursor to S16-multisleeve (trial #21), which tests whether fixing both diagnosed causes (keep S7 in original form, use separate un-merged sleeves) is enough — see below.

### S13 — Opportunistic Insider Cluster Buying (Trial #20)
- **Hypothesis:** E1 (mgmt_pit's original insider-conviction trial, naive net insider buying, IC≈+0.0002/t=0.02 — essentially null) pooled all Form-4 purchase activity together. Cohen, Malloy & Pomorski (2012) argue this destroys signal: routine (scheduled/recurring) trades carry little information while opportunistic (discretionary, clustered) trades carry real information. This trial isolates **cluster buying** — 3+ different insiders making opportunistic purchases at the same company within a rolling 30-day window — as the long signal, against a "zero insider buying activity" short leg (a clean tilt-vs-no-signal design, not short-on-selling).
- **Data reuse:** 100% of universe/price/shares/Form-4 filing-list infrastructure reused read-only from `mgmt_pit`'s existing (E1) cache. New: raw Form-4 XML re-fetched for the ~10,649 filings containing an open-market purchase, to extract reporting-owner identity + 10b5-1 flag (100.0% resolved, 3 network-timeout errors). Confirmed empirically the 10b5-1 flag does not exist in Form-4 XML before 2023, so that branch of the routine/opportunistic classifier is a structural no-op for this entire in-sample window.
- **⚠️ Data-coverage finding (discovered by this trial, not caused by it):** the reused `mgmt_pit` Form-4 cache has meaningful transaction density only through 2017-12 — 11/24 quarters (2018-04-01 onward) are total blackouts, a pre-existing gap in the reused pipeline (not evidence cluster events became rare — spot-checked that real Form-4 filings exist in SEC submissions for that window but were never fetched into the E1-era cache). Effective supported window: 13/24 quarters (2015-01-01 through 2018-01-01). Within those 13 covered quarters, the long-leg name-count guard never fired (≥38 cluster-flagged names every covered quarter) — cluster events were not rare when data existed.
- **Ann. gross:** −0.74% | **Cost:** 780 bps/yr | **Ann. net:** −8.26% | **Sharpe:** −0.857 | **CALMAR:** −0.306
- **IC:** +0.0138 (t=0.850, n=13) | **Max DD:** −27.0% | **Beta:** +0.056 (t=0.264, beta-neutral construction working) | **Mean turnover:** 59%/quarter
- **Win rate:** 38.5% | **n_periods:** 13 (quarterly, coverage-limited, not 24)
- **DSR:** per_period_sr = −0.428, threshold = 0.456 (N=20, n=13) → FAIL
- **Trial-numbering note:** originally run when N=17 (Russell family latest); the ~2.5hr Form-4 re-fetch/backtest runtime overlapped with two concurrent sessions logging `analystrevision_S14` (#18) and `compositeIC_S15` (#19), discovered via a `trial_number` collision at write-time and corrected post-hoc to the true N=20 (verdict unchanged). This incident is the origin of the "re-check the registry immediately before writing, not just at task start" rule now in `feedback_dsr_trial_numbering` memory.
- **Falsification diagnostics:** Sign-convention audit — cluster leg underperformed the no-buying leg in 7/13 quarters (a majority — the "wrong" sign), consistent with the negative net P&L. Concentration check clean (max 16.1% of |return| mass in any quarter, max 11.2% in any SIC major-group).
- **Comparison to E1:** cluster-flag IC (+0.0138, t=0.850) is directionally ~3× the naive same-universe/window re-run's IC (+0.0043, t=0.308) — weak evidence the routine/opportunistic + clustering refinement isolates somewhat more signal than naive pooling, consistent with the literature's mechanism — but remains statistically indistinguishable from zero at this n.
- **Source:** `research/insider_clusters/results/RESULTS_INSIDERCLUSTERS_S13.md` (full detail, retained as the source document), `research/trial_registry.csv` (study=`insiderclusters_S13`, trial #20)
- **Verdict:** The refinement did not rescue the insider-buying signal. Genuinely null-to-negative result, though clouded by a real, pre-existing data-coverage limitation (only 13 of the intended 24 quarters usable) that should be fixed (a real Form-4 re-fetch for 2018-Q2 through 2020-Q4) before this family is revisited.

### S16-A — Multi-Sleeve Blend (Trial #21)
- **Hypothesis:** Three independently beta-neutral sleeves (GP/A quality, S8 residual reversal, S7 IVOL-conditioned value kept in its **original** hard-gated double-sort form — not S15's continuous-score conversion), blended at pre-registered equal 1/3 capital weight with no merged score and no weight tuning, should achieve the genuine diversification that S15's merged rank-percentile composite (trial #19) failed to deliver. Direct follow-up to S15, testing whether fixing both of S15's diagnosed failure causes is enough.
- **Construction:** All three sleeves reused verbatim from their own already-validated backtests (`gp_period_returns.csv`, `s7_period_returns.csv`, `s8_period_returns.csv`) — no re-run, no re-derivation risk. S8's native monthly returns compounded onto GP/A's quarterly grid for blend-level reporting.
- **Universe deviation (confirmed with user before running):** the brief asked for actual point-in-time Russell 2000 constituent membership sourced from `research/russell_reconstitution/`. That directory only has annual R3000-union additions/deletions **event** lists (no full roster, no quarterly granularity, 2018/2019 deletions gap) — not reconstructable without new data acquisition. Fell back to the standard $100M–$2B PIT universe (same as every sleeve's own original construction); used IWM (already-cached Russell 2000 ETF) as the benchmark instead.
- **Ann. gross:** −0.64% | **Cost:** 668 bps/yr | **Ann. net:** −7.12% | **Sharpe:** −0.740 | **CALMAR:** −0.182
- **Max DD:** −39.1% | **Beta (vs IWM):** −0.058 (t=−0.808) | **Mean turnover:** — (blend-level, not directly comparable across sleeves' own native cadences)
- **Win rate:** 33.3% | **n_periods:** 24 (quarterly)
- **DSR:** per_period_sr = −0.370, threshold = 0.341 (N=21, n=24) → FAIL
- **Result: both of S15's diagnosed problems were fixed, and the blend still failed — for a different, more fundamental reason.** GP/A-vs-S7 correlation dropped to +0.280 (from S15's +0.563); S7's IC stayed correctly signed at +0.036 (kept in original form, not re-derived, unlike S15's flipped continuous score). But Blend Sharpe (−0.740) is worse than S15's own composite (−0.070) and worse than GP/A alone (+0.134, the only positive-Sharpe sleeve of the three).
- **Root cause once construction issues were ruled out:** S7's original double-sort has a standalone net Sharpe of −0.544 and max drawdown of −70% even correctly gated and signed — giving it a full, undiluted 1/3 of the book (rather than S15's smoothed continuous score) makes the blend worse, not better. Diversification smooths volatility across genuinely different return drivers; it cannot manufacture positive expected return from two of three legs (S7, S8) that don't have it standalone. Sleeve-level concentration check: one sleeve (most often S7, 14/24 quarters) dominates >50% of the blend's quarter-to-quarter |return| mass in 17/24 quarters — equal capital weight did not translate into equal contribution, since S7's swings are structurally larger than GP/A's or S8's.
- **Source:** `research/multisleeve_S16/results/RESULTS_S16.md` (full detail, retained as the source document), `research/trial_registry.csv` (study=`multisleeve_S16`, trial #21)
- **Lesson for future signal-combination trials:** check each candidate sleeve's own standalone net Sharpe before combining, not just pairwise correlation — a near-zero-correlation blend of three losers is still a loser.

### S16-B — Industry-Neutral GP/A (Trial #22)
- **Hypothesis:** Industry-neutral reformulation of S3-Q (GP/A, trial #8): rank GP/A **within** each SIC major-group first (pooling groups with <5 members into an OTHER bucket), then build long/short from those industry-neutral percentiles — vs. S3-Q's original full-universe rank. Hypothesis: removing accidental sector bets lowers realized beta and improves Sharpe/CALMAR.
- **Universe note:** uses the same $100M–$2B universe as S3-Q, **not** Russell 2000 PIT membership — the `research/russell_reconstitution/` pipeline only has annual addition/deletion event lists, never a full constituent snapshot, so genuine PIT Russell 2000 membership could not be reconstructed without new data sourcing; the user chose to keep S3-Q's universe when asked directly (same category of decision as S16-A's and gpvolmanaged's).
- **Data reuse:** 100% reused from S3-Q/mgmt_pit's existing cache (Tiingo prices, adjClose, SEC XBRL GP/A facts, SEC prefilter) — no new fetches. New: within-SIC-major-group (`sic // 100`) rank-percentile transform only.
- **Ann. gross:** +7.49% | **Cost:** 471 bps/yr | **Ann. net:** +2.63% | **Sharpe:** +0.270 | **CALMAR:** +0.131
- **IC:** +0.0336 (t=2.265) | **Max DD:** −20.0% | **Beta:** −0.057 (t=−0.736) | **Mean turnover:** 29%/quarter
- **Win rate:** 54.2% | **n_periods:** 24 (quarterly)
- **DSR:** per_period_sr = 0.135, threshold = 0.345 (N=22, n=24) → FAIL
- **Direct comparison (recomputed original-ranking 'orig' vs. this trial's industry-neutral 'in', identical universe/window/cost-model):** IC 0.0373→0.0336, Sharpe 0.433→0.270, beta 0.054→−0.057. (S3-Q's own already-logged registry numbers, trial #8, for a third reference point: Sharpe=0.134, IC=0.040, beta=−0.103.)
- **Result: hypothesis not confirmed on either axis.** |Beta| was **not** lower than the original full-universe ranking (−0.057 vs. 0.054 in magnitude terms, essentially unchanged), and Sharpe was **not** improved (0.270 vs. 0.433 on the same recomputed-original comparison) — industry-neutral ranking made this specific run *worse* on both metrics it was designed to help. Sector concentration (Herfindahl index) did improve as expected: long-leg HHI 0.081→0.063, short-leg 0.126→0.073 — the reformulation achieved its narrow mechanical goal (less sector concentration) without the expected downstream benefit to beta or risk-adjusted return.
- **Source:** `research/trial_registry.csv` (study=`gpindneutral_S16`, trial #22)
- **Trial-numbering note:** N_TRIALS was re-verified against the registry immediately before logging (per the `feedback_dsr_trial_numbering` lesson from S13's collision).
- **Verdict:** FAILS DSR. Removing sector concentration is not sufficient on its own to improve GP/A's risk-adjusted return in this universe/period — the original full-universe ranking's beta and Sharpe were not, in fact, being driven by uncontrolled sector bets to the degree the hypothesis assumed.

### S17 — Volatility-Managed GP/A (Trial #23)
- **Hypothesis:** Moreira & Muir (2017), "Volatility-Managed Portfolios": scaling S3-Q's beta-neutral GP/A long-short gross exposure inversely to trailing 6-month realized volatility, targeting 10% annualized, improves Sharpe/CALMAR vs. the unscaled signal without altering stock selection at all. Construction-level overlay, not a re-tune of GP/A.
- **Universe upgrade — what was achievable, and what wasn't:** the brief asked for actual point-in-time Russell 2000 constituent membership, reusing the Wayback-archived FTSE Russell data. Checked before building and found **not fully achievable**: the archive has only R3000-level additions/deletions events per annual cycle (2016–2023, no full snapshot at any date, no R1000/R2000 split, no seed to cumulative-track true membership from — the same limitation S16-A and S16-B independently hit). Presented to the user as a blocking decision; approved approach was a **confirmed-exit overlay** — keep S3-Q's $100M–$2B universe as the base, additionally exclude any ticker from the date it's confirmed deleted from the R3000. Explicitly **not** "actual Russell 2000 membership" — reported as such. R3000 deletions data also has the same pre-existing 2018/2019 gap noted elsewhere in this document. Window: 2016-01-01→2024-01-01 (32 quarterly periods), the closest achievable ~8-year window given the archive's actual coverage, not the requested 2015–2023.
- **Construction:** Base signal/portfolio unchanged from S3-Q (reused by import). Vol-targeting overlay pre-registered before running: `TARGET_VOL=10%` annualized, trailing realized vol from the unscaled basket's own daily adjClose returns (up to 126 trading days / ~6 months, minimum 60 days before activating), `vol_scale = clip(TARGET_VOL / trailing_vol, 0.2, 3.0)`, applied multiplicatively to both legs (preserves the beta-neutral ratio). Two parallel series run on the identical window/universe to isolate the overlay's effect: BASE (beta-neutral only) vs. VOL-MANAGED (VM, base + overlay).
- **Ann. gross:** +7.57% | **Cost:** 273 bps/yr | **Ann. net:** +4.72% | **Sharpe:** +0.481 | **CALMAR:** +0.329
- **IC:** +0.0458 (t=2.699, same signal as BASE — unaffected by position sizing) | **Max DD:** −14.3% | **Beta:** +0.056 (t=0.919) | **Mean turnover:** 21%/quarter
- **Win rate:** 53.1% | **n_periods:** 32 (quarterly)
- **DSR:** per_period_sr = 0.241, threshold = 0.303 (N=23, n=32) → FAIL
- **BASE comparison (identical window/universe, no vol overlay):** Sharpe 0.451, CALMAR 0.295, Max DD −20.0%, beta +0.058 (t=0.669) — VM is a real but modest improvement over BASE on Sharpe/CALMAR/drawdown.
- **vol_scale never once exceeded 1.0x across all 32 periods** — this factor's realized volatility sits structurally above the 10% target throughout the window (min trailing vol observed: 10.6%), so in this backtest the overlay is a pure de-risking mechanism, never a re-risking one.
- **2020 finding (the key falsification diagnostic for this trial):** the Q1 2020 crash quarter's trailing-vol input was only 11.2% (the spike hadn't shown up in the trailing window yet), so vol_scale barely moved (0.89x) — almost no crash-quarter protection. The real de-risking arrived **after** the crash, in the Q2/Q3 2020 recovery quarters (0.60x, 0.52x), throttling the rebound. Net effect: full-year 2020 cumulative return is *worse* under vol-management (−3.78% vs. BASE's +1.23%) — the backward-looking estimator lagged the actual vol spike. Max-drawdown trough depth did improve in the COVID quarter specifically (−17.1%→−14.3%), and improved more in the unrelated 2016–2017 drawdown episode (−20.0%→−13.0%) — the overall Sharpe/CALMAR gain is better attributed to broad-based drawdown smoothing than to a "2020 crash-avoidance" story.
- **Known limitations (disclosed):** confirmed-exit overlay ≠ true R2000 membership; 2018–2019 R3000 deletions gap; standard cost model doesn't charge extra for vol-scale-driven resizing of unchanged positions (likely understates real cost drag for VM specifically); per-year realized beta has only 2 degrees of freedom (t-stats reported but not meaningful at this N).
- **Source:** `research/gp_volmanaged/results/gp_volmanaged_period_returns.csv` (full 32-period record, both BASE and VM series), `research/trial_registry.csv` (study=`gpvolmanaged_S17`, trial #23). Full exposure-scaling-factor-by-period table, year-by-year vs. market table, and per-year beta table are preserved in `research/PHASE3_TRIAL_REGISTRY.md` (superseded pointer — see that file's header) pending a future consolidation pass to inline them here if needed.
- **Verdict:** FAILS the Sharpe≥0.8/CALMAR~1 promotion target and FAILS DSR at N=23. The overlay's direction is right (Sharpe/CALMAR both improved modestly vs. the unscaled control) but the magnitude is far short of the program's bar, and the improvement is not really a COVID-specific effect despite that being the overlay's original motivating case.

### S18 — Small-Cap Pairs Trading, Sector-Bucketed / FDR-Corrected / Persistence-Filtered (Trial #24, STOPPED — Zero Trades)
- **Hypothesis:** Genuinely new mechanism, distinct from every prior cross-sectional single-factor trial in this registry — relative-value mean-reversion between cointegrated, same-industry pairs (Gatev-Goetzmann-Rouwenhorst 2006 design), not a rank-based long/short. Multiple-testing control applied to *pair selection* (analogous to this registry's DSR discipline for *trial* selection): sector-bucket first, hierarchical clustering within each bucket, Engle-Granger cointegration with Benjamini-Hochberg FDR correction pooled across all buckets per formation period, then a hard 2-consecutive-formation-period persistence filter before a pair is eligible to trade at all.
- **Construction:** 10 rolling cycles (12-month formation / 6-month trading, rolled every 6 months, standard GGR design), 2015-01-01→2021-01-01. Sector buckets reused the SIC major-group (`sic // 100`) infrastructure from gp_industry_neutral, but — unlike that trial's "pool small groups into OTHER" convention — buckets below `MIN_BUCKET_SIZE=8` are **excluded from pair search entirely**, since a grab-bag OTHER bucket of unrelated micro-sectors has no economic-linkage story for co-movement (the entire justification for bucketing here). Within each qualifying bucket, hierarchical clustering (scipy, average linkage on 1-correlation distance, formation-period daily log returns) cut at a pre-registered distance threshold of 0.4 (avg intra-cluster correlation ≥ 0.6); candidate pairs are all combinations within clusters of size 2–8. Engle-Granger cointegration on formation-period adjClose price levels for every candidate; all p-values from all buckets in that formation period pooled and BH-FDR corrected together (α=0.05). A pair must survive FDR correction in the current cycle **and** the immediately preceding one before it's eligible to trade (disclosed reading of "non-overlapping": distinct sequential cycles in the rolling index, not literal zero-date-overlap — GGR's standard rolling formation windows share 6 of 12 months of data by construction, so a literal reading would be incompatible with using standard rolling formation at all).
- **Signal (on persistence-qualified pairs only, none existed — see result):** spread = log(P_A) − β·log(P_B), β = formation-period OLS hedge ratio. Entry at |z|>2 (equal-dollar long underperformer / short outperformer — hedge ratio used only to define the signal, not position sizing), exit at |z|<0.5 or forced close at trading-period end.
- **Disclosed deviation:** adjClose used throughout for both signal and P&L (not this registry's usual adjClose-signal/raw-close-return split) — in pairs trading signal and return are the same price series, so a split/dividend artifact in either leg would corrupt the cointegration test, generate spurious entries/exits, and misstate P&L simultaneously, all at once.
- **Selectivity funnel (summed across all 10 cycles):** 59,687 pre-clustering within-bucket candidate pairs → **307** after hierarchical clustering (99.5% reduction) → 307 cointegration tests run → **9** FDR-corrected survivors (per-cycle: 1, 0, 0, 0, 0, 0, 1, 0, 5, 2) → **0** persistence-qualified pairs (2 consecutive cycles) → **0** actual trades.
- **Result: STOP.** Zero pairs ever achieved 2-consecutive-cycle persistence across the full window. No return series exists; Sharpe/CALMAR/beta/DSR are not computable — this is a disposition like H1's or S14's STOP, not a numeric performance failure. Pre-registered thresholds were **not** relaxed after seeing this.
- **Verified not a bug:** cycle 8's 5 FDR survivors (CCS-LGIH, CCS-MTH, CCS-TMHC, LGIH-TMHC, AEIS-ICHR — all economically sensible: homebuilders and semiconductor-equipment suppliers) share **zero overlap** with cycle 9's 2 survivors (RTX-VC, BBSI-HRI), confirmed by direct inspection of the per-cycle survivor sets. The specific pair identities that clear FDR shift between consecutive 6-month-apart formation windows even within a genuinely correlated sub-sector, so the 2-consecutive-pass gate filtered all of them out.
- **Key structural finding:** same-sector pairwise correlations in this $100M–$2B universe are structurally much lower than large-cap pairs-trading intuition assumes — mean bucket correlation ~0.2–0.3, max rarely above 0.6 even in cycle 0's sample check (12 buckets, only 3 had *any* pair above the 0.6 clustering threshold). This is *why* clustering alone narrows 59,687 candidates to 307 before any statistical test is even run — small/mid-caps are more idiosyncratic than the mechanism's implicit large-cap prior (e.g. Coke/Pepsi-style pairs) assumes.
- **DIAGNOSTIC SUPPLEMENT (explicitly NOT this trial's registered result, and not itself additionally FDR-corrected for the multiple-testing the persistence filter exists to control)** — bypassing the persistence filter and trading the 9 raw per-cycle FDR-survivor pairs directly: 16 trades, win rate 43.8% overall (62.5% for converged exits, avg 27.75 days; only 25.0% for forced-close exits, avg 99.75 days — consistent with forced closes being relationships that broke down rather than reverted). Net monthly (61mo, continuous daily series 2016–2021, zero-filled on no-position days): ann. return −4.44%, Sharpe −0.454, CALMAR −0.183, max DD −24.2%, t=−1.02. Gross monthly: ann. return −0.54%, Sharpe −0.017, max DD −18.0% — gross is near-breakeven-to-negative, **not** a clean "good signal ruined by cost" story like S4/S3-Q; cost drag (~404 bps/yr annualized) deepens an already-weak gross result rather than flipping a genuinely positive one. Realized beta vs. IWM = −0.097 (t=−1.81, not significant) — the trial's core low-beta motivating premise is roughly validated even in this small, non-registered sample. Concentration: top single pair = 23.7% of total |P&L| (same figure for top sector, since it's one pair) — acceptable but flagged given N=16 is small. Only 329 of 1,306 business days (25%) had any capital deployed across the full 5-year window.
- **Falsification diagnostics:** Formation/trading leakage audit — PASS by construction (`trading_start == formation_end` exactly every cycle; hedge ratio and spread mean/std frozen at formation end, never recomputed mid-trading-period). Concentration check — see diagnostic supplement above (N/A for the main zero-trade result). Selectivity funnel — reported in full above; this is itself the trial's central diagnostic finding.
- **Source:** `research/pairs_S18/run.py` (full pipeline), `research/pairs_S18/finalize_and_log.py` (registry logging), `research/pairs_S18/results/{_raw_backtest.json, trades.csv, daily_book_returns.csv, diagnostic_nopersistence_daily.csv, diagnostic_nopersistence_trades.csv}`, `research/trial_registry.csv` (study=`pairs_S18`, trial #24, status=`stopped_zero_trades`).
- **Verdict:** STOP — the pre-registered multiple-testing discipline (sector-bucket → clustering → FDR → persistence) is, in this small/mid-cap universe and window, strict enough to eliminate every candidate before any trade occurs. This is itself the honest result of applying DSR-equivalent rigor to pair selection, not a pipeline failure. The non-registered diagnostic (bypassing persistence) suggests that even without the final gate, the mechanism would likely have been a weak-to-negative, cost-sensitive strategy in this universe — consistent with, not contradicted by, the main zero-trade outcome.

### S19 — 52-Week High Anchoring (Trial #25)
- **Hypothesis:** George & Hwang (2004): investors anchor on a stock's 52-week high as a psychological ceiling. Stocks trading near their 52-week high see good news underreacted to (reluctance to bid "through" the anchor), producing positive drift as the bias corrects. Genuinely distinct mechanism from S1/S4 (IVOL/MAX) — anchoring to a slow-moving reference price, not extreme recent-return behavior. Requires only Tiingo price data + the already-cached shares-outstanding facts used for PIT market cap (same minimal dependency as S9/S1/S4) — no new data sourcing, avoiding the feasibility risk that stopped S14 and constrained S18.
- **Signal:** nearness = adjClose<sub>t</sub> / trailing-252-trading-day-high adjClose, computed using only data available as of t (no look-ahead), `MIN_52W_DAYS=200`. Long top quintile (nearest to 52w high), short bottom quintile. Standard $100M–$2B PIT universe, reused verbatim from momentum_jt (S9) by file-path import — universe builder, PIT market cap, price/adjClose caches, and the 12-1 momentum signal function itself all reused unmodified, not reimplemented.
- **Construction:** Monthly rebalance, pre-registered before running per the original literature's convention — no frequency sweep on this pass (a deliberate frequency-robustness follow-up, if warranted, would be separately logged per the S3 lesson, not folded in here). Beta-neutral leg sizing (60-day OLS vs. EW-universe adjClose). Actual-turnover cost model (not an assumed fixed rate) — standard bid-ask by cap tier + borrow on the short leg, monthly-scaled.
- **Ann. gross:** −8.37% | **Cost:** 1,305 bps/yr | **Ann. net:** −19.77% | **Sharpe:** −0.908 | **CALMAR:** −0.261
- **IC:** +0.0044 (t=0.238 — essentially null) | **Max DD:** −75.7% | **Beta:** −0.020 (t=−0.216, beta-neutral construction working) | **Mean turnover:** 35%/month
- **Win rate:** 47.2% | **n_periods:** 72 (monthly)
- **DSR:** per_period_sr = −0.262, threshold = 0.206 (N=25, n=72) → FAIL
- **Momentum independence check (the trial's core secondary test):** mean cross-sectional Spearman correlation between nearness and 12-1 momentum = **0.611** (range 0.41–0.80, 82% of periods >0.5) — this **contradicts** the original literature's "largely independent" claim in this small/mid-cap universe; the two signals move together substantially, plausibly because a stock in a strong upward 12-month run is mechanically likely to also be near its own 52-week high.
- **Fama-MacBeth check (forward return regressed on both signals' cross-sectional ranks jointly, every period):** mean b<sub>nearness</sub> = **−0.0198** (t=−1.644 — **wrong-signed** and only marginal) vs. mean b<sub>momentum</sub> = +0.0134 (t=1.882 — correctly signed, also marginal). Once momentum is controlled for, nearness's already-weak univariate IC does not just wash out — it **flips sign**, while momentum retains its (weak) expected sign.
- **Ex-2020 reference (2015–2019, 60mo):** ann. return −8.07%, Sharpe −0.456, CALMAR −0.199, max DD −40.6%, mean IC +0.017 — **already a losing strategy before 2020**, not purely a COVID-crash story.
- **2020 sub-period check:** catastrophic — full-year 2020 return −59.4% (vs. the EW-universe market's +32.9%), 4 of 12 months had |return|>15% (March −23.4%, April −17.7%, August −16.0%, November −15.6%). Critically, the worst months outside the initial crash (April, November) occurred **during market recovery rallies** (mkt_ret +26.8% and +22.3% respectively) — consistent with a momentum-crash-style regime sensitivity: long near-high/recent-winner names and short far-below/recent-loser names gets destroyed specifically when the most beaten-down names rally hardest (April 2020 snapback, November 2020 vaccine/value rotation). The global max drawdown window spans peak 2016-01 to trough 2020-11 — a 4+ year continuous underwater period, not a single-year event.
- **Falsification diagnostics:** Sign-convention/guard audit — clean (`n_long == n_short` every period; min leg 68 names, well above the 20-name floor). Concentration — equal-weight legs with a minimum of 68 names per leg bound any single name to ≤1.5% weight by construction; no concentration risk. 2020 sub-period — see above.
- **Source:** `research/fiftytwoweek_S19/run.py`, `research/fiftytwoweek_S19/finalize_and_log.py`, `research/fiftytwoweek_S19/results/{s19_period_returns.csv, s19_famamacbeth.csv, s19_correlation.csv}`, `research/trial_registry.csv` (study=`fiftytwoweek_S19`, trial #25).
- **Verdict:** FAILS the Sharpe≥0.8/CALMAR~1 target decisively (both solidly negative). **This does not look like a genuinely distinct effect from momentum in this universe** — univariate IC is already statistically indistinguishable from zero (t=0.24), the signal is substantially correlated with momentum (directly contradicting the paper's independence premise), and controlling for momentum flips nearness's sign rather than confirming independent predictive power. At best it reads as a weaker, noisier restatement of momentum here, not a new exploitable mechanism. High monthly-rebalance cost drag (~1,305 bps/yr) compounds an already-negative gross result — not a "good signal ruined by costs" story either.

### S20 — Institutional "Smart Money" Accumulation, 13F-Based (Trial #26, STOPPED at Phase 1 — CUSIP Crosswalk Blocked)
- **Hypothesis:** A new or meaningfully increased institutional (Form 13F) position in a thinly-covered small/mid-cap name may signal information the market hasn't priced in yet. Distinct mechanism from S13 (company insiders) — tests external professional capital allocation, not internal executives.
- **Phase 1 finding: the underlying 13F data is fully feasible** — SEC's bulk structured 13F data sets (`SUBMISSION.tsv` + `INFOTABLE.tsv`, quarterly ZIPs, free, back to 2013) were downloaded and inspected directly. `SUBMISSION.tsv` has a genuine `FILING_DATE` field distinct from `PERIODOFREPORT`, confirmed by direct query (e.g. as of 2015-01-01, only the `2014q4` bulk file's filings with `FILING_DATE <= 2015-01-01` are available — 4,050 filings — while the `2015q1` file correctly contributes zero, since those are all filed after) — the ~45-day reporting lag can be enforced correctly with no look-ahead risk. This part of the mechanism is **not** a repeat of S14's problem.
- **Where it actually broke:** 13F holdings are keyed by **CUSIP**, not ticker, and this project has no ticker→CUSIP crosswalk. FMP's `/profile` endpoint (already configured) returns CUSIP and — unlike the analyst-estimates endpoints that blocked S14 — works for small/mid-cap tickers, not just large-caps. But partway through building the crosswalk for just the two Phase-1 sample-check dates (~560 ticker lookups needed), FMP returned `429 "Limit Reach"` and started blocking **every** subsequent call, including previously-working large-caps (MSFT) — confirmed to be a hard account-wide quota wall, not a symbol-tier restriction. OpenFIGI (genuinely free, no key required) was checked as a substitute and does not expose CUSIP in its mapping output at all (Bloomberg lacks CUSIP redistribution rights) — not usable. No other CUSIP source is configured in this project.
- **User decision:** presented three options (stop and log as dead end; wait for an unconfirmed quota reset and resume; proceed with the 233 already-resolved CUSIPs as a partial, order-biased universe). User chose **stop now, log as documented dead end** — explicitly declining the partial-universe option to avoid forcing a low-quality, selection-biased pipeline.
- **Side observation (partial, order-biased sample — the 233 tickers whose CUSIPs happened to resolve before the quota wall hit, NOT a clean random sample, reported for context only):** of 142 resolved 2015-universe names, 116 (81.7%) were held by ≥1 13F filer as of 2015-01-01 (filing-date gated); of 180 resolved 2020-universe names, 159 (88.3%) were held by ≥1 13F filer as of 2020-01-01. This suggests institutional 13F coverage in this universe is likely **high** if the crosswalk problem were solved — a promising signal for a future attempt, not a reason to doubt the underlying hypothesis's testability, only its *feasibility right now*.
- **Source:** `research/trial_registry.csv` (study=`institutional13f_S20`, trial #26, status=`stopped_no_data`). Downloaded SEC bulk files and CUSIP crosswalk work were done ad hoc in this session (not saved as a standalone script, since the trial stopped at Phase 1) — the SEC bulk 13F endpoint pattern (`https://www.sec.gov/files/structureddata/data/form-13f-data-sets/{year}q{n}_form13f.zip`, containing `SUBMISSION.tsv`/`INFOTABLE.tsv`/`COVERPAGE.tsv`) and the FMP `/stable/profile?symbol=X` CUSIP field are recorded here for whoever picks this up next.
- **Verdict:** **Kill before backtest — but distinct in kind from S14 and S18's stops.** Unlike S14 (data did not exist / was blocked at the source for the target universe), the 13F data itself here is fully free and unblocked; the CUSIP crosswalk specifically is rate-limited on the only available vendor. This is a solvable-with-more-time-or-budget blocker, not a fundamental data-existence gap. To resume: either (a) wait for the FMP quota to reset (unconfirmed period, likely daily for this tier) and spread the ~3,300-ticker crosswalk across multiple sessions, or (b) source CUSIPs from a different vendor/tier — a real budget/time decision left to the user, not assumed or fixed here.

### S21 — Cluster-Conditioned Short-Term Mean Reversion, Z-Score Spread (Trial #27)
- **Hypothesis:** A genuine hybrid mechanism between S18 (strict pairwise cointegration, STOPPED zero trades) and S8 (full-cross-section residual reversal, real IC but gross-negative). Tests whether a WEAKER, basket-level relationship (deviation from a sector-correlated cluster's own average, not one specific cointegrated partner) is more tradeable than S18's pairwise approach, and whether conditioning short-term reversal on cluster membership (instead of the full cross-section) fixes S8's gross-negative problem.
- **Construction:** Standard $100M–$2B PIT universe, sector-bucketed (`sic // 100`, reused from gp_industry_neutral). 10 rolling GGR cycles (12mo formation/6mo trading, reused verbatim from pairs_S18). Within each qualifying bucket, hierarchical clustering (same machinery as S18) cut at a DELIBERATELY more permissive distance threshold (0.8, vs S18's 0.4) — targeting S18's own diagnosed "typical" bucket correlation (~0.2), not S18's "exceptional" ≥0.6 bar. Clusters filtered to a pre-registered size range of 4–15 members. Signal: for each member, a formation-anchored cumulative log-return spread vs the cluster's LEAVE-ONE-OUT equal-weighted average (excludes the member itself), Z-scored using FROZEN formation-period mean/std (no look-ahead). Entry at |z|>2.5 (pre-registered), exit at |z|<0.5 or a 5-day forced close (pre-registered — upper end of the idea's 2-5 day window), whichever first. Equal-dollar member-vs-basket sizing (not beta-weighted, same choice as S18); realized beta reported regardless. Two constructions run: PRIMARY = leave-one-out cluster basket; SECONDARY = single nearest peer within the cluster (by formation correlation).
- **Selectivity funnel:** avg universe 413.8 names/cycle → 3,085 bucketed (sum) → 637 raw clusters → **109** in the [4,15] size range (100% became tradeable) → 848 tradeable member-slots → **4,683 entries/trades** (basket) / 4,858 (peer) — a working, populated funnel, unlike S18's collapse to zero.
- **Realized correlation structure (the direct comparison to S18):** mean intra-cluster correlation = **0.328** (median 0.306, range 0.228–0.599) — landed almost exactly on S18's own diagnosed "typical" bucket-mean level (0.2–0.3), confirming the deliberately permissive cut worked as intended and these clusters genuinely represent the weaker, basket-level relationship the trial set out to test.
- **Result — PRIMARY (basket):** Sharpe (net, ann.) = **−17.411**, Sharpe (gross, ann.) = **+0.954**, CALMAR (net) = −0.668, CALMAR (gross) = 0.653, ann. return net = **−66.44%**, ann. return gross = **+5.78%**, max DD (net) = −99.53%, max DD (gross) = −8.84%, beta vs IWM = +0.090 (t=+2.82), **cost drag = 10,999 bps/yr (~110%/yr)**, mean IC (|entry z| vs trade_gross, per-cycle) = +0.0207 (t=+1.57, n=10 cycles), win rate 17.0% overall (100% for the 0.4% of trades that genuinely converge, 16.7% for the 99.6% force-closed on the 5-day timer). n=4,683 trades, avg hold 4.88 days, n=60 months. **FAILS DSR** at N=27 (per-period SR=−5.026 vs threshold 0.2306).
- **Result — SECONDARY (peer):** Sharpe (net) = −9.395, Sharpe (gross) = +0.623, CALMAR (net) = −0.626, ann. net = −62.11%, ann. gross = +5.85%, beta = +0.074 (t=1.34), cost drag = 9,888 bps/yr, IC = +0.0393 (t=+1.61). Same pattern as basket, somewhat weaker gross Sharpe; basket-vs-peer referencing is not the deciding factor for the outcome.
- **The headline finding:** this is the cleanest and most extreme "real signal, destroyed by cost" result in the registry. Each individual trade's gross return is tiny (mean +0.19%), but with 4,683 quasi-independent bets it compounds via breadth into a genuinely respectable gross Sharpe (0.95) — a textbook small-edge/high-breadth effect, and the IC confirms it's real (not noise): bigger dislocations modestly predict bigger reversion gains. The entire failure is cost, not signal quality: at ~4.9-day average holds and ~2.5–2.7% round-trip cost, capital effectively rotates through a fresh cost event roughly weekly, compounding to ~110%/yr — an order of magnitude worse than any other cost-driven failure in this registry (S2's next-worst cost drag was ~4,079 bps/yr, ~2.7x smaller).
- **Falsification diagnostics:** Formation/trading leakage audit — PASS by construction (cluster membership and formation spread stats frozen at `formation_end`, each member anchored at its own first valid formation observation, not a shared panel row — matters for staggered listing dates). Concentration check — PASS (top sector = 16.8%/14.6% of total |trade PnL| mass across 19 sectors, basket/peer). Sign-convention audit — PASS (IC positive both constructions, consistent with the reversion hypothesis).
- **Infrastructure bug found and fixed:** `pairs_S18` and `gp_industry_neutral` are both named `run.py`; loading both via `importlib` by file path (to avoid a bare-import collision) creates TWO separate `gp_industry_neutral` module instances with independent, initially-empty price caches. The first backtest attempt produced zero clusters in every cycle (despite 350–450-name universes) because `pairs_S18.formation_log_prices()` was silently calling its own never-preloaded internal copy. Fixed by reassigning `P.R = R` after import. Flagged for any future trial reusing this dual-import pattern.
- **Registry housekeeping note:** the finalization script's "re-check trial count immediately before writing" logic initially read its own already-logged row as part of "the current max" when re-run a second time (to correct a wrong interpretive claim in the notes text, not a new backtest), self-inflating the trial number from 27 to 28 and leaving a gap at 27. Caught and corrected before being left in the registry — the script now excludes its own study's prior row before computing the max.
- **Source:** `research/clusterreversion_S21/run.py` (backtest), `research/clusterreversion_S21/finalize_and_log.py` (metrics + registry logging), `research/clusterreversion_S21/results/{trades_basket.csv, trades_peer.csv, daily_book_returns_basket.csv, daily_book_returns_peer.csv, clusters_summary.csv, final_summary.json, RESULTS_S21.md}`, `research/trial_registry.csv` (study=`clusterreversion_S21`, trial #27).
- **Verdict on the two motivating questions:** (1) vs S18's pairwise-correlation problem — cluster-average referencing is unambiguously more tradeable, generating both volume and a real positive gross IC/Sharpe at a correlation level S18 itself diagnosed as this universe's ceiling. (2) vs S8's gross-negative problem — cluster-conditioning fixed it decisively (gross Sharpe 0.95 vs S8's negative gross return), but surfaced a more severe problem: trading frequency this high makes even a genuinely good gross signal completely non-viable once realistic costs apply. **FAILS the Sharpe≥0.8/CALMAR~1 target on net returns** (gross alone would have cleared it) and **FAILS DSR at N=27**.

### S22 — Long-Only Quarterly Valuation Portfolios: DCF vs. P/E vs. P/S vs. P/B (Trial #28)
- **Hypothesis:** Four sub-trials sharing one DSR family slot (IVOL/MAX family-sharing convention) — same underlying question (does a valuation signal identify outperforming stocks) tested four ways: DCF-implied margin of safety, P/E, P/S, P/B. **Genuinely new construction for this registry: LONG-ONLY**, not beta-neutral long/short like every other trial — beta is expected to be well above zero, reported prominently rather than treated as a defect.
- **Architecture finding, resolved before building:** the existing "Phase 2 Monte Carlo DCF engine" was found to have a fully pure, reusable *computation* core (`dcf.value()`, `compute_drivers()`, `compute_wacc()`, `_build_base()` — no I/O) but a *data-fetching* layer (yfinance + live EDGAR) with **zero point-in-time capability** — neither has an `as_of` parameter, both only return "today's" financials. Resolved by building a new PIT `RawData` constructor (reusing the same SEC XBRL tag hierarchies already established in `gp/run.py`/`accruals/run.py`) that feeds the *existing, unmodified* computation chain — not by rewriting the DCF's math. One small additive patch was needed: `wacc.py`'s `_get_rf()` gained an optional `as_of` parameter (defaults to `None`, preserving the live tool's exact original behavior) to source that year's FRED historical annual rate instead of today's, for the one line in `compute_wacc()` that wasn't already a pure function of its inputs.
- **Performance bug found and fixed in existing code:** initial DCF-adapter testing measured ~3.85s/name (infeasible at ~9,600 name-quarters). Root cause: `compute_wacc()` calls `_get_rf_annual_series()` — a live FRED API fetch of the full historical DGS10 series — on **every single invocation**, uncached. Harmless for the live tool (one valuation at a time) but a severe bottleneck in a backtest loop. Fixed with simple module-level memoization (fetch once per process); reduced to 0.059s/name (65x), making the full backtest feasible in the ~15-minute range. This benefits the live tool too (removes redundant FRED calls on every valuation), not just this backtest.
- **DCF signal type (confirmed with user before building):** deterministic margin-of-safety = `(dcf.value(assumptions) - price) / price`, **not** the full Monte Carlo probability-of-undervaluation — one clean calc per ticker-quarter, no 10,000-path simulation loop, per instructions' explicit allowance of "margin of safety percentage" as the ranking metric. The DCF's own existing applicability gate (`terminal_fcff > 0 and value_per_share > 0`, reused verbatim from `montecarlo.py`'s `run_valuation()`) is applied to exclude degenerate cases.
- **DCF signal characteristic (disclosed, not patched):** a meaningful right tail of extreme margin-of-safety values (some >500%) was traced to the DCF engine's own existing `MATURE_MARGIN_DEFAULT=0.20` assumption — low-current-margin small caps get modeled fading up to a 20% "mature" EBIT margin over the forecast horizon, which is a genuine, pre-existing design choice in the general-purpose DCF tool, not a bug introduced here. Not patched (that would mean redesigning the already-built tool rather than reusing it) — flagged as a real characteristic of applying this tool to a small-cap universe with many currently-thin-margin names. Since portfolio construction uses rank order only (equal-weight top 20%), extreme magnitude doesn't distort position sizing, only which names qualify.
- **Construction (shared across all four):** Standard $100M–$2B PIT universe (reused verbatim from `gp_industry_neutral`), 2015-01-01→2021-01-01 quarterly (24 periods, this registry's standard window — cached data supports extending further but this matches convention). Uniform signal convention for direct comparability: `undervaluation_score` where higher = more undervalued = long candidate (DCF: margin of safety as-is; P/E, P/S, P/B: negated ratio, i.e. `-PE`/`-PS`/`-PB`). Top 20% by score, equal-weight, long-only (leg = `max(ceil(n*0.20), 20)`). Negative/missing denominators excluded per signal, not force-ranked (P/E excludes non-positive net income; P/B excludes non-positive book equity; P/S exclusion practically never binds). Standard cost model (bid-ask by cap tier) on actual quarterly turnover, no borrow leg (long-only).
- **CORRECTION (2026-07-15) — signal coverage was materially understated in the original write-up.** An earlier version of this section claimed P/E's exclusion cost "~4–9% of universe/period beyond missing-data cases" — that figure was wrong. The actual mean coverage (fraction of the ~422-name average universe with a valid signal each quarter), measured directly from the saved period records, is:

| Signal | Mean coverage | Mean N with valid signal | Mean N in long leg |
|---|---:|---:|---:|
| P/B | 94.5% | 399 | 80 |
| P/S | 90.8% | 384 | 77 |
| DCF | 63.4% | 267 | 54 |
| P/E | 58.9% | 248 | 50 |

  P/E's real gap is ~35–40 percentage points of the universe, not single digits — driven almost entirely by the loss-making-company exclusion (a large fraction of $100M–$2B small-caps are unprofitable in any given quarter). DCF's ~63% coverage (never quantified in the original write-up at all) reflects its much larger data footprint — ~12 XBRL concepts plus the applicability gate, versus P/B's and P/S's single-concept dependency.
- **This is a real limitation on the head-to-head comparison's fairness, not a footnote.** The four signals were **not** ranking from equivalent eligible universes each quarter — DCF and P/E were effectively selecting from a smaller, more-established/profitable subset (mean ~257 names), while P/S and P/B saw nearly the full universe (mean ~392 names). A signal that looks better partly because it's drawn from an easier, pre-filtered-by-profitability subset is not directly comparable to one drawn from the full universe. See the intersection-universe re-analysis below, which restricts all four signals to the common subset every quarter to test whether the original ranking survives this correction.
- **Full four-way comparison table:**

| Metric | DCF | P/E | P/S | P/B |
|---|---:|---:|---:|---:|
| Ann. return (net) | +10.59% | +7.88% | **+12.91%** | +7.72% |
| Ann. return (gross) | +12.54% | +9.85% | **+14.49%** | +9.64% |
| Cost drag (bps/yr) | 180 | 186 | **144** | 179 |
| Mean turnover/qtr | 31% | 34% | **25%** | 30% |
| Sharpe (net, ann.) | 0.452 | 0.388 | **0.493** | 0.389 |
| CALMAR | **0.233** | 0.127 | 0.225 | 0.132 |
| Max drawdown | **−45.5%** | −61.9% | −57.3% | −58.5% |
| Market beta (t-stat) | 1.010 (24.5) | 1.221 (20.7) | 1.279 (19.3) | 1.246 (26.8) |
| Alpha vs. EW-universe (ann.) | −2.74% | −5.77% | **−1.49%** | −6.03% |
| Alpha vs. 3M T-bill (ann.) | +15.53% | +16.32% | **+22.57%** | +16.54% |
| Mean IC (Spearman) | 0.0014 | 0.0132 | **0.0210** | 0.0040 |
| IC t-stat | 0.07 | 0.57 | **0.65** | 0.16 |
| Win rate | 62.5% | 58.3% | **66.7%** | 62.5% |
| N periods | 24 | 24 | 24 | 24 |
| Per-period SR | 0.226 | 0.194 | **0.246** | 0.194 |
| DSR threshold (N=28, n=24) | 0.368 | 0.368 | 0.368 | 0.368 |
| **Clears DSR** | NO | NO | NO | NO |

- **Beta note (the trial's own explicit premise, confirmed):** all four betas are 1.0–1.28 with t-stats of 19–27 — decisively, statistically real market exposure, unlike every beta-neutral trial elsewhere in this registry. This is expected and correct for a long-only construction, not a construction flaw.
- **Year-by-year net return vs. EW-universe benchmark (all four use the identical universe/period, so the benchmark column is shared):**

| Year | DCF | P/E | P/S | P/B | EW-universe (mkt) | IWM |
|---|---:|---:|---:|---:|---:|---:|
| 2015 | −7.3% | −12.9% | −19.2% | −19.8% | −7.4% | −3.9% |
| 2016 | +34.9% | +40.1% | +63.3% | +53.4% | +27.6% | +21.6% |
| 2017 | +9.5% | +15.0% | +20.4% | +17.0% | +18.5% | +14.6% |
| 2018 | −14.1% | −20.4% | −21.7% | −22.3% | −9.4% | −11.1% |
| 2019 | +9.3% | +8.6% | +19.7% | +15.3% | +22.1% | +25.4% |
| 2020 | +42.2% | +30.1% | +38.9% | +21.0% | +37.7% | +20.0% |

- **Alpha decomposition — the central, most informative finding:** alpha vs. the EW-universe is **negative for all four signals** (−1.5% to −6.0%/yr) despite strongly positive alpha vs. 3-month T-bills (+15.5% to +22.6%/yr). This means the large absolute returns these portfolios earned are substantially explained by their high-beta exposure to a broadly rising 2015–2020 small-cap market (EW-universe was positive in 4 of 6 years, strongly so in 2016/2019/2020) — not by genuine stock-picking alpha over that market. A passive beta-matched exposure to the same universe would likely have done as well or better than three of the four signals.
- **Falsification diagnostics:** Sector concentration — clean for all four (mean sector HHI 0.05–0.06, max single-sector share 10–21%); none is a disguised sector bet, including P/B (well-documented in the literature to load on financials elsewhere, not observed as a problem here). Sign-convention audit — mean IC positive for all four as hypothesized, but only 50–54% of individual quarters show the correct sign — an honest, thin, marginal edge consistent with the uniformly weak t-stats (all <1). 2020 sub-period — all four show the standard crash-then-recovery pattern (Q1 2020: −35% to −51%); DCF had both the mildest crash-quarter and the best full-year 2020 result (+42.2%, beating the market's +37.7%), while P/S's crash-quarter IC (−0.396) was the most severely wrong-signed of the four before recovering strongly the rest of the year.
- **Direct ranking on the ORIGINAL (unequal-universe) comparison — superseded below, kept for the record:** by Sharpe and by every alpha/IC measure, P/S looked best or tied-best on every single metric, with DCF competitive on Sharpe/CALMAR but an essentially null IC (0.0014, t=0.07). This produced an initial ranking of P/S > DCF > P/E ≈ P/B. **This ranking did not survive the fairness correction below and should not be cited on its own.**
- **CORRECTION (2026-07-15) — intersection-universe re-analysis.** Prompted by the coverage-gap finding above (DCF/P/E draw from a smaller, more-established/profitable subset than P/S/P/B each quarter), all four signals were re-run restricted to the **intersection universe** — only names where all four signals have a valid value that quarter (mean 179 names/quarter, vs. ~422 in the original unrestricted universe; same 20%-equal-weight construction, leg size shrinks to a mean of 36 names for all four identically). The full-universe portion of this re-run was cross-checked byte-for-byte against the original results (max return difference across all 96 signal-periods: 0.0) before trusting the intersection numbers.

| Metric | DCF (orig → inter) | P/E (orig → inter) | P/S (orig → inter) | P/B (orig → inter) |
|---|---:|---:|---:|---:|
| Ann. return (net) | 10.59% → 9.60% | 7.88% → 9.22% | 12.91% → 9.61% | 7.72% → 9.61% |
| Sharpe (net, ann.) | 0.452 → **0.422** | 0.388 → **0.413** | 0.493 → **0.425** | 0.389 → **0.428** |
| CALMAR | 0.233 → 0.196 | 0.127 → 0.167 | 0.225 → 0.166 | 0.132 → 0.179 |
| Max drawdown | −45.5% → −48.9% | −61.9% → −55.4% | −57.3% → −57.9% | −58.5% → −53.7% |
| Market beta | 1.010 → 1.029 | 1.221 → 1.107 | 1.279 → 1.214 | 1.246 → 1.154 |
| Alpha vs. EW-universe | −2.74% → −3.69% | −5.77% → −4.17% | −1.49% → −4.08% | −6.03% → −3.87% |
| Alpha vs. 3M T-bill | +15.53% → +14.81% | +16.32% → +15.86% | +22.57% → +18.16% | +16.54% → +17.18% |
| Mean IC | 0.0014 → **0.0092** | 0.0132 → **0.0167** | 0.0210 → **0.0087** | 0.0040 → **−0.0025** |
| IC t-stat | 0.07 → 0.52 | 0.57 → 0.61 | 0.65 → 0.28 | 0.16 → −0.11 |
| Per-period SR | 0.226 → 0.211 | 0.194 → 0.206 | 0.246 → 0.212 | 0.194 → 0.214 |

  **The original ranking evaporates.** On the fair, identical-universe comparison, Sharpe for all four clusters into a narrow 0.41–0.43 band (P/S's original lead of +0.04–0.10 over the others shrinks to essentially nothing), CALMAR clusters 0.17–0.20, and alpha vs. the market clusters −3.7% to −4.2% (all four, not just three). The IC ranking **inverts**: P/E now has the best IC (0.0167) and P/B's flips to **negative** (−0.0025) — P/S, the original "winner," drops to third by IC and its t-stat roughly halves (0.65 → 0.28). None of the four clears DSR on the intersection universe either (per-period SR 0.21–0.21 vs. threshold 0.368).
- **Is the original conclusion robust? No — the specific ranking is not, but the higher-level verdict is, and comes back stronger.** The original "P/S beats DCF, simple beats complex" finding was substantially an artifact of DCF and P/E being evaluated on a smaller, easier (more-established/profitable) subset of the universe than P/S and P/B. Once all four are forced to compete on the identical opportunity set, **no signal is distinguishable from any other** — same Sharpe tier, same alpha tier, same DSR failure, and the "best IC" signal changes depending on which universe you evaluate on. That is itself a more damning finding for the DCF specifically: its added complexity (a new PIT data pipeline, WACC reconstruction, applicability gating) bought it exactly nothing once the universe is held fixed — it does not outperform, and does not underperform, P/E/P/S/P/B on a fair basis. All four are equally weak, interchangeable signals in this universe/window.
- **Source:** `research/valuationcompare_S22/{run.py, run_intersection.py, pit_facts.py, dcf_adapter.py, finalize_and_log.py}`, `research/valuationcompare_S22/results/s22_{dcf,pe,ps,pb}_{period_returns,sector,intersection,full_crosscheck}.csv`, `s22_all_signal_snapshots.csv`, `s22_intersection_metrics.json`, `dcf/valuation/wacc.py` (additive `as_of` patch + FRED memoization fix, both backward-compatible), `research/trial_registry.csv` (study=`valuationcompare_S22_{dcf,pe,ps,pb}`, trial #28, shared DSR slot — the intersection re-analysis is a robustness check on the same registered trial, not a new trial number).
- **Verdict:** All four **FAIL** the Sharpe≥0.8/CALMAR~1 target and **FAIL DSR at N=28**, on both the original and the intersection universe. The originally-reported ranking (P/S > DCF > P/E ≈ P/B) does **not** survive a fair, equal-universe comparison — on the intersection universe all four are statistically indistinguishable (Sharpe 0.41–0.43, alpha vs. market −3.7% to −4.2%, IC t-stats all under 1 and their relative order inverts depending on universe). The honest conclusion is not "simple beats complex" but **"none of the four is distinguishable from the others, and the DCF's substantial added complexity bought no measurable edge over the cheapest ratio once compared fairly."**

---

## 5. S1–S10 Program Status

| Signal | ID | Status | Notes |
|--------|-----|--------|-------|
| IVOL | S1 | Complete, FAIL DSR | v2 corrected (beta-neutral); prior chat numbers stale |
| PEAD | S2 | Complete, FAIL DSR | Gross negative; cost catastrophic |
| GP/A | S3 | Complete, FAIL (quarterly, annual); **PASS (semiannual)** | Best-of-3 selection; walk-forward now run — FAILS (reference only, see Section 4) |
| MAX | S4 | Complete, FAIL DSR | Cost-nonviable; decay study also fails |
| Asset Growth | S5 | Complete, FAIL DSR | Null signal in this sample |
| Residual Momentum | S6 | Complete, FAIL DSR | Momentum family with S9 |
| IVOL-conditional Value | S7 | Complete, FAIL DSR | Double-sort; gross negative |
| Residual Short-Term Reversal | S8 | Complete, FAIL DSR | Trial #14 — real IC (t=3.40) but gross-negative |
| Cross-sectional Momentum 12-1 | S9 | Complete, FAIL DSR | Failed as pipeline validator |
| Amihud Illiquidity | S10 | Complete, FAIL DSR | Trial #15 — IC significant AND wrong-signed |

**Program has since extended well past S1–S10** — trials #16–23 (S11 accruals, the S3-SA walk-forward, S14 analyst revision [stopped], S15 composite, S13 insider clusters, S16-A multi-sleeve, S16-B industry-neutral GP/A, S17 volatility-managed GP/A) are all complete and detailed in Section 4. This table is kept for historical continuity with the original 15-trial program naming (S1–S10); see Section 2's DSR Accounting Table for the full current list.

---

## 6. Open Items

1. **E1–E4 period-24 bug:** Re-run required. `mgmt_pit/results/in_sample_perf.csv` row 25 must be corrected (hold_end = 2021-01-01, not 2025-10-01) and all trial_registry.csv E-series numbers regenerated. The direction of all results (negative Sharpe, DSR FAIL) is almost certainly unchanged given the strength of the losses in 23 valid periods, but the exact numbers are unreliable.

2. **S3-SA walk-forward — now complete, FAILS (reference only):** the out-of-sample check on the 2021-01-01→2025-01-01 window has been run (Sharpe +0.449, IC held up at +0.059, but a material and statistically significant positive beta of +0.403 (t=2.73) emerged that was not present in-sample — worth investigating before this signal is considered further. See Section 4 for full detail.

3. **Insider-buying Form-4 cache has a real coverage gap:** discovered by S13 (trial #20), not caused by it — the reused `mgmt_pit` Form-4 cache has zero transaction density from 2018-Q2 onward (11/24 quarters are total blackouts). Any future insider-buying-based trial needs a real re-fetch for that window first.

4. **S3-A registry entry missing:** `gp/results/gp_decay_annual.csv` was saved but no trial_registry entry was written. The run_decay.py `maybe_log_registry()` function only writes entries for clearing variants (line 489: `clearing = [s for s in summaries if s["clears_dsr"] and s["ppy"] != 4]`). Annual did not clear so it was not logged. All S3-A numbers in this registry were verified directly from the data file.

5. **S1 IVOL v1 run no longer in trial_registry:** The v2 corrected run replaced the v1 entry in-place. The pre-correction S1 numbers (Sharpe −0.560, β −0.655, MaxDD −80.0%, ann. −15.97%) are documented only in the `ivol/run.py` code comment. They should not be cited going forward.

6. **Cost model calibration — checked, no change needed (2026-07-14):** S21's extreme cost drag (~110%/yr against a real gross Sharpe of 0.95) prompted a full audit of the standard `_spread()`/`_borrow()` cost model used identically across every trial since S1. Findings: (a) S21's own trades are NOT concentrated in illiquid names (roughly uniform across market-cap deciles, median entry ~$800M); (b) Tiingo carries no historical bid/ask data (EOD-only endpoint), so the model's 20–100bps one-way tiers were never back-tested against this project's own data and remain an external calibration; (c) triangulated against the SEC's 2013 small-cap market-quality study (nearly identical $100M–$2B bins) and other sources, the tiers read as realistic to mildly conservative, not miscalibrated; (d) a sensitivity sweep (0×–2× the current spread assumption) shows no cost-flagged FAIL (S2, S4, S8, S21) crosses the Sharpe≥0.8 bar at any non-degenerate cost level — these verdicts are robust to cost-model uncertainty in either direction. Full detail: `research/COST_MODEL_CALIBRATION_AUDIT.md`.

---

## 7. Pipeline Learnings

1. **Cost drag is the dominant kill factor in small-cap L/S:** Six of thirteen trials have positive gross alpha (S4, S3-Q, S3-SA, S3-A, S2, S5 marginally), but four of those fail due to cost drag alone. Monthly and daily rebalance is essentially infeasible for the current cost model.

2. **Momentum inverts in small-caps:** S9 failed as a pipeline validator, indicating the momentum factor is unreliable in sub-$2bn US equities. S6 confirmed this with residual stripping. Momentum-based strategies require large-cap or long-short with longer holds.

3. **Beta-neutral construction is mandatory for IVOL/MAX family:** S1 v1's β = −0.655 was a structural portfolio construction error that contaminated all results. The v2 correction reduces this to β = −0.220, materially changing the return and drawdown profile.

4. **Frequency sweep (S3) is a valid robustness tool but creates selection risk:** Running three frequencies and registering only the clearer (semiannual) is internally consistent given the pre-announced "not a new trial" rule, but it creates a best-of-3 selection effect that the walk-forward must account for.

5. **High-IC ≠ positive gross return (S2 PEAD):** A signal with IC t=7.55 can still produce negative gross returns in a calendar-time implementation. The disconnect likely comes from the interaction between beta-neutral sizing, event-driven entry/exit, and the specific periods when earnings events cluster (e.g., 2020-Q1/Q2 earnings clustered with COVID crash positions).

6. **E1–E4 management quality signals underperform their own benchmark:** The EY L/S benchmark (ann. −11.68%, Sharpe −0.571) also loses money. E1–E4's "outperformance" vs benchmark is purely an artifact of a losing benchmark, not evidence of alpha. E3 and benchmark both have Sharpe ≈ −0.56.

7. **Small n_periods is the binding constraint for annual-rebalance strategies:** S3-A has only 6 annual periods, giving a DSR threshold of 0.470 — effectively requiring a Sharpe of 0.47+ just to pass the multiple-comparisons gate. With inherent return volatility at annual frequencies, this is a very high bar.

---

**[Former "Section 8 Addendum" for S16 Multi-Sleeve Blend has been merged into Section 4 above (see "S16-A — Multi-Sleeve Blend, Trial #21") during the 2026-07-12 consolidation — removed from here to avoid duplication.]**
