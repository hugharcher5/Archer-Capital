# RESULTS SUMMARY — Archer Capital Quantitative Research Program

**Generated:** 2026-07-06  
**Universe:** US small/mid-cap equities (SEC XBRL + Tiingo)  
**In-sample window:** 2015-01-01 → 2021-01-01  
**DSR gate:** Deflated Sharpe Ratio, cumulative independent-trial count N=14  
**Status as of this writing:** 0 of 14 tested strategies have cleared the in-sample DSR gate (semiannual GP/A is the sole technical pass; see note under S3-SA). S10 (Amihud illiquidity) remains pending.

---

## E1 — Management Quality: Insider Conviction (M1)

**Result:** IC=+0.0002 (t=0.02) | Gross=n/a | Net=−0.84%/yr | Cost drag=n/a | Sharpe=−0.075 | DSR: FAIL (obs −0.037 < threshold 0.138) [BUG-PENDING]

**Notes:** The insider-conviction signal (net open-market purchase score from SEC Form 4) is essentially uncorrelated with subsequent quarterly returns (IC t=0.02). The strategy loses money slowly and monotonically across the sample. The in-sample performance data is contaminated by a period-24 data bug (hold_end set to 2025-10-01 instead of 2021-01-01); no corrected re-run exists. Even accounting for potential bug contamination, the near-zero IC and slightly negative return make a DSR pass implausible. The failure mode is genuine null signal: insider purchases in small/mid-cap may lag price moves or be too sparse to aggregate into a reliable systematic factor.

---

## E2 — Management Quality: Non-Dilution (M2)

**Result:** IC=+0.0186 (t=1.82) | Net=−5.46%/yr | Sharpe=−0.169 | Max DD=−63.5% | DSR: FAIL (obs −0.084 < threshold 0.138) [BUG-PENDING]

**Notes:** The non-dilution signal (firms avoiding share issuance, preferring buybacks) has the highest IC of the E-family (t=1.82, marginally sub-significant), suggesting some predictive content. The strategy nonetheless produces −5.46% annual return and a near-63% drawdown. The period-24 bug inflated period-24 M2_solo_ls to +52.3% — removing that period would make results even worse. The failure mode is crash-prone: non-diluting firms may be concentrated in sectors that sold off sharply during the COVID downturn (financials and industrials with buyback programs), negating the positive IC.

---

## E3 — Management Quality: ROIC Improvement (M3)

**Result:** IC=−0.0115 (t=−0.86) | Net=−12.95%/yr | Sharpe=−0.560 | Max DD=−67.4% | DSR: FAIL (obs −0.280 < threshold 0.138) [BUG-PENDING]

**Notes:** The ROIC-improvement signal inverts: firms improving ROIC subsequently underperform in this sample. Negative IC (−0.0115) is consistent across subperiods. The annualised return of −12.95% makes M3 the worst individual E-trial. The period-24 bug contributed M3_solo_ls = −56.7% in that (contaminated) period, which was the single worst observation across all trials. The failure mode is wrong-sign signal: ROIC improvement may reflect capital-intensive expansion in small-caps rather than capital efficiency, with the market already pricing the improvement.

---

## E4 — Management Quality: Composite M1+M2+M3

**Result:** IC=— | Net=−2.09%/yr | Sharpe=−0.113 | Max DD=−41.8% | DSR: FAIL (obs −0.057 < threshold 0.138) [BUG-PENDING]

**Notes:** Compositing the three signals does not rescue the family: M3's negative IC dilutes M1+M2's weak positive ICs. The composite return (−2.09%) is better than any individual signal except M1 (−0.84%), but the benefit is modest. The E-family finding is that management quality signals derived from insider filings, dilution, and ROIC in small/mid-cap do not produce reliable alpha in the 2015–2021 period. All four trials fail with negative Sharpe ratios despite the period-24 contamination, which would have required an improbably large positive effect to reverse the conclusion.

---

## H1 — Livestock Disease / Animal-Pharma Event Study

**Result:** IC=n/a | CAAR(40d, pooled)=−2.63% (t=−2.10) | Gross=n/a | Net=n/a | Cost drag=n/a | Sharpe=n/a | DSR: STOP (wrong sign)

**Notes:** The hypothesis — that WAHIS-notified livestock disease outbreaks produce positive post-announcement drift in animal-pharma stocks — failed on sign. Every tested window (20/40/60 days, HPAI-only, all-disease pooled, major/minor market splits) produced negative CAARs. The most statistically significant result was the 60-day all-minor-market pooled CAAR (−7.37%, t=−3.95), but in the wrong direction. The structural weakness is the universe: only three investable US names (PAHC, ELAN, NEOG), each with mixed species exposure, making the event-matching noisy. The failure mode is a combination of wrong-sign and thin universe — the hypothesis is untestable at scale with US-listed securities.

---

## S9 — Cross-Sectional Momentum 12-1 (Jegadeesh-Titman)

**Result:** IC=−0.0157 (t=−0.82) | Gross=−4.87%/yr (cost not modeled) | Net=n/a | Sharpe=−0.324 | Max DD=−45.3% | Beta=−0.063 | DSR: FAIL (obs −0.162 < threshold 0.197)

**Notes:** Run primarily as a pipeline validator — if the harness cannot replicate the most-replicated factor in finance, it signals a pipeline bug. The strategy not only failed to pass DSR; it produced negative IC, confirming that raw 12-1 momentum inverts in US small/mid-cap over this period. The failure is not a harness bug but a genuine empirical finding: momentum is well-documented to be weak or absent in small-caps, and the 2020 COVID reversal (momentum crash) fell entirely within the in-sample window. The failure mode is regime-sensitive: momentum strategies are acutely sensitive to sharp reversals, which occurred in 2018-Q4 and 2020.

---

## S6 — Residual Momentum 12-1

**Result:** IC=−0.0129 (t=−0.70) | Gross=+4.31%/yr | Cost drag=762 bps/yr | Net=−3.34%/yr | Sharpe=−0.156 | Max DD=−27.4% | Beta=+0.067 | DSR: FAIL (obs −0.078 < threshold 0.197, shared with S9)

**Notes:** Beta-stripping the momentum signal (36-month market model) reduced the drawdown materially (−27.4% vs S9's −45.3%) and improved Sharpe (−0.156 vs −0.324), confirming that part of S9's loss was net-long-beta exposure. However, gross alpha turned positive (+4.31%) and was immediately eaten by cost drag (762 bps). The signal IC remains negative (−0.013), meaning the directional momentum hypothesis still inverts. Residual momentum is a cleaner construction but does not rescue the underlying signal failure in this universe. The 762 bps/yr cost drag is decisive even if the IC sign eventually flips in a different period.

---

## S1 — IVOL Low-Lottery (v2, beta-neutral)

**Result:** IC=−0.0646 (t=−2.27) | Net=−10.93%/yr (arithmetic) | Sharpe=−0.464 | Max DD=−70.8% | Beta=−0.220 | DSR: FAIL (obs −0.232 < threshold 0.218)

**Notes:** [Note: the chat-assembled registry reported stale v1 numbers — Sharpe −0.560, beta −0.655, MaxDD −80.0%, ann. −15.97% — from a dollar-neutral construction that was structurally net-short-market due to the asymmetric beta of IVOL quintiles. The v2 corrected run uses beta-neutral sizing; all numbers above are from v2.] The IC is negative and significant (t=−2.27), confirming that low-IVOL stocks underperformed high-IVOL stocks in this sample — opposite the hypothesis. The 2020 COVID crash was catastrophic (Q1: −27.2%, Q2: −45.1% ls_ret), as high-IVOL / high-beta short positions rallied during the recovery. The failure mode is genuine null plus wrong-sign: the IVOL premium has weakened post-publication (Hou and Loh 2016 concerns) and may have reversed in small-caps where retail participation is disproportionate.

---

## S4 — MAX Lottery Effect

**Result:** IC=+0.034 (t=2.46) | Gross=+4.76%/yr | Cost drag=2,202 bps/yr (B-A 2,084 + borrow 118) | Net=−16.06%/yr | Sharpe=−1.264 | Max DD=−65.4% | Beta=+0.019 | DSR: FAIL (obs −0.365 < threshold 0.126)

**Notes:** The strongest IC of any tested strategy and the only trial with a statistically significant IC in the expected direction (t=2.46 for MAX). Yet the Sharpe is −1.26. The failure is entirely cost-driven: monthly rebalance requires >65% portfolio turnover per month, generating 2,202 bps/yr of friction against only 476 bps gross alpha. The holding-period decay study (1M, 2M, 3M) confirms that longer holds reduce costs but also reduce signal efficacy — net Sharpe improves but never approaches the DSR gate. The MAX signal is real (the lottery-stock mispricing hypothesis has empirical support) but is cost-nonviable in small/mid-cap without a structural execution advantage.

---

## S3-Q — GP/A Quality Factor, Quarterly Rebalance

**Result:** IC=+0.040 (t=2.28) | Gross=+4.93%/yr | Cost drag=390 bps/yr | Net=+0.94%/yr | Sharpe=+0.134 | Max DD=−28.3% | CALMAR=0.033 | Beta=−0.103 | DSR: FAIL (obs 0.069 < threshold 0.235)

**Notes:** GP/A (gross profit-to-assets) produces a positive IC with t=2.28, confirming the Novy-Marx (2013) signal is present in this small/mid-cap universe. Gross alpha (+4.93%) is real but cost drag (390 bps/yr) reduces net return to +0.94%, giving a net Sharpe of only 0.134. The DSR threshold of 0.235 requires more than twice the actual per-period SR to compensate for the N=8 multiple comparisons penalty. The signal appears to carry information; the problem is execution cost at quarterly frequency.

---

## S3-SA — GP/A Quality Factor, Semiannual Rebalance (frequency sweep)

**Result:** IC=+0.054 (t=1.70) | Gross=+10.16%/yr | Cost drag=318 bps/yr | Net=+6.85%/yr | Sharpe=+0.569 | Max DD=−17.9% | CALMAR=0.383 | Beta=−0.078 | DSR: PASS (obs 0.403 > threshold 0.332)

**Notes:** Reducing rebalance frequency to semi-annual cuts turnover significantly, lowers cost drag from 390 to 318 bps, and — more importantly — allows the gross return to compound more effectively (quarterly gross 4.93%/yr vs semiannual 10.16%/yr). The result clears the in-sample DSR gate. However, S3-SA is the best of three outcomes in the frequency sweep (quarterly fails, annual fails); selecting the frequency that passes is a form of in-sample optimisation even within the "not a new trial" framework. The IC slightly weakens at semiannual frequency (t=1.70 vs quarterly t=2.28), suggesting the signal is noisier at 6-month horizons. This is the only trial to clear DSR in the entire program. Walk-forward validation is the mandatory next step before any practical application.

---

## S3-A — GP/A Quality Factor, Annual Rebalance (frequency sweep)

**Result:** IC=+0.067 (t=1.12) | Gross=+8.61%/yr | Cost drag=~321 bps/yr (est.) | Net=+5.92%/yr | Sharpe=+0.361 | Max DD=−18.3% | DSR: FAIL (obs ~0.242 < threshold 0.470)

**Notes:** Annual rebalance produces the highest IC of the sweep (0.067 vs 0.040/0.054 at quarterly/semiannual), consistent with GP/A being a slow-moving signal that improves as the holding horizon extends. However, the DSR threshold for n=6 annual observations is a demanding 0.470 annualised Sharpe — the highest gate in the program — and the actual Sharpe of 0.361 does not clear it. The losing years (2015: −10.6%, 2016: −18.3% net) reflect regime drawdowns that are genuine, not quarterly rebalance noise. The combination of too few observations (6) and a non-trivial two-year losing streak makes in-sample certification infeasible at annual frequency even if the long-run signal is sound. No trial_registry entry was written (the `run_decay.py` only logs clearing variants); all S3-A numbers are verified from `gp/results/gp_decay_annual.csv`.

---

## S2 — PEAD (Post-Earnings Announcement Drift)

**Result:** IC=+0.037 (t=7.55) | Gross=−4.74%/yr | Cost drag=4,079 bps/yr | Net=−36.59%/yr | Sharpe=−2.864 | Max DD=−93.2% | Beta=−0.104 | DSR: FAIL (obs −0.827 < threshold 0.157)

**Notes:** The PEAD IC is the most statistically significant of any signal in the program (t=7.55 across 42,119 events), confirming that short-window earnings surprise (3-day abnormal return around 8-K filing) predicts 45-day forward returns. Yet the strategy is disastrous. Gross alpha is negative (−4.74%/yr) before costs, and costs of 4,079 bps/yr make the strategy deeply loss-making (net −36.6%/yr). The win rate of 16.7% across 72 calendar months indicates the portfolio is losing in the vast majority of months. Two failure modes combine: (1) the calendar-time portfolio aggregates events during the 2020 COVID crash, where many positions are open simultaneously and the beta-neutral construction does not protect adequately; (2) the daily-event structure creates massive turnover even in stable periods. The PEAD IC disconnect (signal valid, return negative) is a key pipeline learning about the gap between signal content and portfolio implementability.

---

## S5 — Asset Growth

**Result:** IC=−0.0099 (t=−0.52) | Gross=+0.62%/yr | Cost drag=613 bps/yr | Net=−5.46%/yr | Sharpe=−0.340 | Max DD=−34.8% | Beta=−0.160 | DSR: FAIL (obs −0.170 < threshold 0.282)

**Notes:** Asset growth (YoY change in total assets) produces essentially zero IC (−0.010, t=−0.52), indicating the signal is genuinely null in this sample. Gross alpha is near-zero (+0.62%), confirming no predictive content. Cost drag (613 bps/yr) turns the result negative. The asset-growth anomaly is documented primarily in large-cap and may have been arbitraged away or may require SIC-specific controls to isolate the balance-sheet-expansion effect from organic growth. The failure mode is genuinely null — there is no evidence this signal carries information in US small/mid-cap 2015–2021.

---

## S7 — IVOL-Conditional Value (Double-Sort)

**Result:** IC=+0.036 (t=1.36) | Gross=−6.52%/yr | Cost drag=882 bps/yr | Net=−14.70%/yr | Sharpe=−0.544 | Max DD=−70.1% | Beta=−0.085 | DSR: FAIL (obs −0.272 < threshold 0.291)

**Notes:** The interaction hypothesis — that value (E/P) is concentrated within the high-IVOL subset — produces a positive but weak IC (t=1.36). The double-sort substantially shrinks the effective universe (top-tercile IVOL only, further sub-divided into value terciles), which reduces diversification and increases per-stock concentration. Gross return is negative (−6.52%), making the strategy fail even before costs. The failure mode is a combination of weak signal (t=1.36 is sub-threshold) and regime-sensitive neutralisation: the COVID crash in Q1/Q2 2020 again dominates, with large positive drawdowns in both periods. This is a distinct hypothesis from S1/S4 (it tests value within high-IVOL, not the IVOL factor itself), but shares the same structural 2020 fragility.

---

## S8 — Residual Short-Term Reversal

**Result:** IC=+0.036 (t=3.40) | Gross=−3.08%/yr | Cost drag=761 bps/yr (B-A 614 + borrow 147) | Net=−10.22%/yr | Sharpe=−1.107 | CALMAR=−0.215 | Max DD=−47.43% | Beta=−0.011 | Actual turnover=21%/month | DSR: FAIL (obs −0.320 < threshold 0.175, N_TRIALS=14, n=70 monthly periods)

**Notes:** Signal = negative 1-month residual return from a rolling 36-month OLS market-model (adjClose, same factor as S6). Pre-registered double filter: top/bottom decile AND |1-month residual| ≥ 1.5× trailing 12M monthly-residual std (standard deviation of 12 non-overlapping prior monthly residual sums). Turnover discipline: names not crossing both thresholds hold existing book positions with no transaction cost charged. The IC of +0.036 (t=3.40) is real and statistically significant — the third-highest IC t-stat in the program, above S4 (t=2.46) and below S2 (t=7.55). Yet gross return is −3.08%/yr before costs. Cost drag of 761 bps/yr at only 21% actual monthly turnover deepens the loss to −10.22%/yr net, Sharpe −1.107. The failure mode is the same as S2 (PEAD): strong IC signal paired with a gross-negative portfolio — signal content and portfolio implementability are genuinely separable.

Three pre-registered falsification diagnostics were all clean: (1) sign-convention audit passed with 0/70 violations (the long leg had lower mean 1M residual return than the short leg every period); (2) no single month dominated the total |return| mass (maximum 5.2%, well below the 30% threshold); (3) liquidity-shock isolation showed no concentration in the most-illiquid ADTV tercile (77% of return mass came from the mid-liquidity tier). The near-zero beta (−0.011, t=−0.28) confirms the residual construction is doing its job of removing market exposure. The naive full-book comparison (all-decile, no vol filter, full monthly churn) produced −20.84%/yr net — the turnover discipline added material value compared to unfiltered reversal, but could not convert a gross-negative underlying signal into a viable strategy.

---

## S10 — Amihud Illiquidity (PENDING)

**Result:** No results files exist. Code exists in research directory structure but no backtest has been run.

---

## Synthesis

### Failure taxonomy

**Wrong-sign signal (signal inverts in this universe/period):** E3 (ROIC improvement), S9 (momentum), S6 (residual momentum), S1 (IVOL). In each case the IC is either near-zero negative or statistically significant in the wrong direction. For momentum and IVOL, this inversion has a plausible economic explanation in small-caps.

**Genuinely null / near-null signal:** E1 (insider conviction, IC ≈ 0), E4 (composite of weak/null signals), S5 (asset growth, IC ≈ −0.010 not distinguishable from zero). No alpha; the hypothesis is not supported in this sample.

**Cost-nonviable (positive gross, positive IC, cost kills):** S4 (MAX, IC=+0.034 t=2.46, gross +4.76% but 2,202 bps cost), S3-Q (IC=+0.040 t=2.28, gross +4.93% but 390 bps kills net Sharpe). The signal is real but the execution structure makes it untradable. S6 also belongs partly here (gross +4.31%, 762 bps cost).

**Crash-prone / regime-sensitive neutralisation:** E2 (non-dilution), S1 v2 (IVOL), S7 (IVOL-conditional value). These strategies had large drawdowns concentrated in 2020 that overwhelmed the period-mean performance. Beta-neutral construction was insufficient to protect against the COVID reversal in small-caps.

**Positive-IC-but-gross-negative (S2 and S8):** A distinct and unusual failure mode that has now appeared twice. S2 (PEAD): IC t=7.55, gross −4.74%/yr — the calendar-time event portfolio aggregates many positions simultaneously during stress periods, and the short-leg recovery dynamics likely dominate. S8 (residual reversal): IC t=3.40, gross −3.08%/yr — the reversal signal is real but prior-month losers apparently continue to underperform on the gross portfolio level despite the IC. In both cases, the signal content and portfolio implementability are genuinely separable: what the IC measures (cross-sectional rank-correlation) and what the L/S portfolio earns (a scale-weighted bet on that correlation) are not the same quantity when beta-neutral scales and the distribution of returns interact unfavourably.

**Only in-sample pass:** S3-SA (semiannual GP/A, Sharpe +0.569, clears DSR 0.332). This is the best of three outcomes in the GP/A frequency sweep. The signal has persistent positive IC across all three frequencies (0.040 / 0.054 / 0.067 quarterly/semi/annual), consistent with a genuine but slow-moving quality signal. Semiannual execution reduces cost drag enough to make net Sharpe viable. Walk-forward testing has not been performed.

### Blunt statement

As of this writing, **zero of 14 tested strategies have cleared the in-sample DSR gate in an unambiguous sense.** The S3-SA semiannual GP/A result is a technical pass (per-period SR 0.403 vs threshold 0.332), but it is the single best frequency selected after observing all three outcomes — the quarterly and annual variants both fail. No strategy has been run on the walk-forward window. S8 (residual reversal) is now complete: FAIL, gross-negative despite IC t=3.40, matching S2's failure mode. S10 (Amihud illiquidity) remains the only pending trial.
