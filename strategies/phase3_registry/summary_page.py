"""
Phase 3 — Results Summary tab.

One entry per experiment, chronological order, condensed format.
Numbers sourced from CORRECTED_TRIAL_REGISTRY.md / underlying CSVs.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st


_EXPERIMENTS = [
    {
        "id": "E1–E4",
        "title": "Management Quality Factor (M1 Insider / M2 Non-Dilution / M3 ROIC / E4 Composite)",
        "hypothesis": "Management quality signals (insider buying, non-dilution, ROIC improvement) predict outperformance in small/mid-cap where analyst coverage is thin.",
        "result_line": "IC ≈ 0 (M1), +0.019 (M2, t=1.82), −0.012 (M3) | Net: −0.84% / −5.46% / −12.95% / −2.09%/yr | Sharpe: −0.075 / −0.169 / −0.560 / −0.113 | DSR: FAIL (all four)",
        "verdict": "FAIL",
        "bug": True,
        "notes": "No signal has positive IC above t=2. The benchmark (earnings-yield L/S) itself lost −11.68%/yr; 'outperformance vs. benchmark' is an artifact of a losing benchmark. M2 (non-dilution, t=1.82) is the only borderline thread, undermined by a −63.5% max drawdown. Period-24 rebalance-date bug confirmed in source files — exact figures provisional, FAIL verdict unchanged. Failure modes: genuinely null (M1, E4), wrong-sign (M3), crash-prone (M2).",
    },
    {
        "id": "H1",
        "title": "Livestock Disease / Animal Pharma Event Study",
        "hypothesis": "WAHIS-notified livestock disease outbreaks produce positive post-event drift in listed animal-pharma names (PEAD analog). Pre-registered 5-hypothesis tree; H1 was the gate.",
        "result_line": "CAAR(40d, all-disease): −2.63% (t=−2.10) | CAAR(60d): −4.52% (t=−2.75) | DSR: STOP (wrong sign)",
        "verdict": "STOP",
        "bug": False,
        "notes": "Every tested window and specification produced negative CAAR — significant in the wrong direction. Placebo clean (effect is real, not a methodology artifact). Structural problem: only 3 US investable names (PAHC, ELAN, NEOG) — not a tradeable strategy even if the sign had been correct. H2–H5 correctly never run. Failure modes: wrong-sign + thin universe.",
    },
    {
        "id": "S9",
        "title": "Momentum 12-1, Raw (Jegadeesh-Titman)",
        "hypothesis": "12-month-minus-1-month cumulative return predicts 3-month forward return in small/mid-cap (pipeline validator).",
        "result_line": "IC: −0.016 (t=−0.82) | Gross: −4.87%/yr | Sharpe: −0.324 | Max DD: −45.3% | Beta: −0.063 | DSR: FAIL (−0.162 < 0.197)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Run as pipeline validator. Failure is not a harness bug — subsequent signals produced expected IC signs. Raw 12-1 momentum inverts in US small/mid-cap: 2020 COVID reversal (−15.6% in Q4-2020) and 2016–17 value reversal both fall inside the in-sample window. Failure mode: regime-sensitive.",
    },
    {
        "id": "S6",
        "title": "Momentum 12-1, Residual (Beta-Stripped)",
        "hypothesis": "Beta-stripped 12-1 momentum (36-month market-model residuals) removes market contamination and survives momentum crashes.",
        "result_line": "IC: −0.013 (t=−0.70) | Gross: +4.31%/yr | Cost: 762bps/yr | Net: −3.34%/yr | Sharpe: −0.156 | Max DD: −27.4% | DSR: FAIL (−0.078 < 0.197)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Beta-stripping halved the drawdown (−27.4% vs S9's −45.3%) and improved Sharpe (−0.156 vs −0.324) — confirms S9 had real net-beta contamination. Gross alpha turned positive (+4.31%) but cost drag (762bps) erased it. IC still negative: stripping doesn't fix the underlying signal inversion. Key generalizable finding: beta-stripping mitigates momentum-crash exposure by ~50% but doesn't eliminate it. Family with S9 — same DSR slot.",
    },
    {
        "id": "S1",
        "title": "Low IVOL  [v2, beta-neutral, corrected]",
        "hypothesis": "Low-idiosyncratic-volatility small/mid-caps earn positive risk-adjusted returns versus high-IVOL lottery names.",
        "result_line": "IC: −0.065 (t=−2.27) | Net: −10.93%/yr | Sharpe: −0.464 | Max DD: −70.8% | Beta: −0.220 | DSR: FAIL (−0.232 < 0.218)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "v2 uses beta-neutral sizing; v1's dollar-neutral construction (β=−0.655) was a structural error — v1 numbers are superseded. IC is negative and significant (t=−2.27): low-IVOL genuinely underperforms high-IVOL in this sample, opposite the hypothesis. COVID crash (Q1: −27.2%, Q2: −45.1%) dominated as high-IVOL short positions rallied hardest in recovery. Failure mode: wrong-sign. Family with S4 — same DSR slot.",
    },
    {
        "id": "S4",
        "title": "MAX / Lottery Effect",
        "hypothesis": "Stocks with extreme recent single-day returns are overpriced lottery tickets that subsequently underperform.",
        "result_line": "IC: +0.034 (t=2.46) | Gross: +4.76%/yr | Cost: 2,202bps/yr | Net: −16.06%/yr | Sharpe: −1.264 | Max DD: −65.4% | DSR: FAIL (−0.365 < 0.126)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Strongest, most statistically significant IC in the expected direction of any trial (t=2.46). Failure is entirely cost-driven: monthly rebalance requires >65% turnover per month, generating 2,202bps/yr friction against 476bps gross alpha. Decay study (1M/2M/3M) confirms longer holds reduce costs but also reduce signal efficacy — net Sharpe never clears DSR at any horizon. Signal is real; strategy is cost-nonviable in small/mid-cap. Family with S1 — same DSR slot.",
    },
    {
        "id": "S3-Q / SA / A",
        "title": "Gross Profitability GP/A — Frequency Sweep (quarterly / semiannual / annual)",
        "hypothesis": "High gross profit-to-assets firms outperform low GP/A firms (Novy-Marx 2013) in small/mid-cap. Three frequencies tested as a pre-announced robustness sweep — all share DSR trial #8.",
        "result_line": "Q: IC +0.040 (t=2.28), gross +4.93%, net +0.94%, Sharpe +0.134 — FAIL | SA: IC +0.054 (t=1.70), gross +10.16%, net +6.85%, Sharpe +0.569 — PASS (0.403 > 0.332) | A: IC +0.067 (t=1.12), gross +8.61%, net +5.92%, Sharpe +0.361 — FAIL",
        "verdict": "PASS (SA only — best-of-3 caveat)",
        "bug": False,
        "notes": "IC rises with hold length (0.040→0.054→0.067), confirming GP/A is a slow-moving fundamental signal. Quarterly fails on cost drag alone (390bps erases ~490bps gross). Semiannual (S3-SA) is the only DSR clear in the 13-trial program — but it is the best of three outcomes in the frequency sweep. Selecting the frequency that passes is a form of in-sample best-of-3 optimisation. Annual fails because n=6 periods pushes the DSR threshold to 0.470, the highest gate in the program. Walk-forward (2021–2025) not yet run — the mandatory next step.",
    },
    {
        "id": "S2",
        "title": "PEAD — Post-Earnings Announcement Drift",
        "hypothesis": "High 3-day abnormal return around SEC 8-K earnings filing predicts positive 45-day forward drift. Calendar-time daily portfolio, beta-neutral, Q1 long / Q5 short.",
        "result_line": "IC: +0.037 (t=7.55, N=42,119) | Gross: −4.74%/yr | Cost: 4,079bps/yr | Net: −36.59%/yr | Sharpe: −2.864 | Max DD: −93.2% | DSR: FAIL (−0.827 < 0.157)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Highest IC t-stat in the program (t=7.55) — signal is real. Yet gross return is negative before costs. Two failures combine: (1) daily-event structure generates ~5.6 new events/day/leg creating 4,079bps/yr turnover cost; (2) regime-sensitive neutralization — pre-COVID betas broke during March 2020, long leg (high-surprise/growth) crashed harder than the short. Post-hoc sanity checks confirmed gross negativity is not a construction bug (equal-dollar gross also −4.30%/yr). Clearest example in the program of the gap between signal content and portfolio implementability.",
    },
    {
        "id": "S5",
        "title": "Asset Growth Anomaly",
        "hypothesis": "Firms with high YoY total-asset growth subsequently underperform (Cooper et al. 2008).",
        "result_line": "IC: −0.010 (t=−0.52) | Gross: +0.62%/yr | Cost: 613bps/yr | Net: −5.46%/yr | Sharpe: −0.340 | Max DD: −34.8% | DSR: FAIL (−0.170 < 0.282)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "IC is essentially zero (t=−0.52) — no evidence the signal carries information in this sample. Gross alpha near-zero; cost drag turns it negative. Neither crash-prone nor cost-nonviable: this is a genuinely null effect. Asset-growth anomaly is documented primarily in large-cap and may require SIC-specific controls to isolate balance-sheet expansion from organic growth. Failure mode: genuinely null.",
    },
    {
        "id": "S7",
        "title": "IVOL-Conditional Value (Double-Sort)",
        "hypothesis": "Value premium concentrates within high-IVOL stocks where arbitrage costs are highest. Double-sort: top-tercile IVOL, then earnings-yield tercile within.",
        "result_line": "IC: +0.036 (t=1.36) | Gross: −6.52%/yr | Cost: 882bps/yr | Net: −14.70%/yr | Sharpe: −0.544 | Max DD: −70.1% | DSR: FAIL (−0.272 < 0.291)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "IC positive but sub-threshold (t=1.36 < 2.0). Secondary comparison: plain value-only Sharpe −0.809 vs double-sort −0.544 — IVOL-conditioning improved Sharpe by +0.264, confirming the interaction is directionally present. Starting from high-IVOL names inherits ~56%/quarter turnover (IVOL/MAX family structure), not Quality-style low-turnover. Failure mode: weak-signal-cost-decisive.",
    },
    {
        "id": "S8",
        "title": "Residual Short-Term Reversal",
        "hypothesis": "Prior-month residual losers outperform winners the following month. Pre-registered double filter: top/bottom decile AND |1M residual| ≥ 1.5× trailing 12M monthly-residual std. Turnover discipline: non-threshold names hold existing positions.",
        "result_line": "IC: +0.036 (t=3.40) | Gross: −3.08%/yr | Cost: 761bps/yr | Net: −10.22%/yr | Sharpe: −1.107 | Max DD: −47.43% | Beta: −0.011 | Turnover: 21%/month | DSR: FAIL (−0.320 < 0.175)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "IC = +0.036 (t=3.40) — signal is real and statistically significant, third-highest IC t-stat in the program. But gross return is −3.08%/yr before any costs. Cost drag (761bps/yr even at 21% actual turnover from the discipline) brings net to −10.22%/yr. Failure mode matches S2 (PEAD): strong IC signal, gross-negative portfolio. Liquidity isolation: return not concentrated in most-illiquid tercile, ruling out an Amihud-style friction premium as the mechanism. Beta near zero (−0.011). Concentration check passed (max single-month 5.2%). Sign audit passed (0/70 violations).",
    },
    {
        "id": "S10",
        "title": "Amihud Illiquidity",
        "hypothesis": "Explicit cost-model stress test, final ranked S1–S10 program signal: long high-ILLIQ (top quintile), short low-ILLIQ (bottom quintile) — pre-registered as not expected to pass net of costs.",
        "result_line": "IC: −0.053 (t=−4.60) | Gross: −7.23%/yr | Cost: 617bps/yr | Net: −12.97%/yr | Sharpe: −1.129 | Max DD: −51.7% | Beta: −0.151 | DSR: FAIL (−0.565 < 0.306)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Strongest IC t-stat magnitude of any trial in the program (t=−4.60), but negative: the illiquidity premium is genuinely inverted in this sample, not merely a cost-model artifact — least-liquid small/mid-caps underperformed the most-liquid names outright. Failure mode: wrong-sign, and significantly so.",
    },
    {
        "id": "Russell-1",
        "title": "Pre-Effective-Date Anticipatory Drift (Russell Reconstitution)",
        "hypothesis": "First trial in a new independent family. Stocks confirmed for addition to a Russell US index drift upward between the preliminary-list announcement and the effective date, as funds anticipate forced index buying. Equal-weight long-only basket of confirmed R3000 additions vs IWM.",
        "result_line": "CAAR (abnormal vs IWM, gross): +0.64% | t=+0.416 | win rate 62.5% (5/8 cycles, 2016–2023) — event study, no SR gate.",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Not significant — no reliable pre-effective-date drift detected. Sharpe/CALMAR (+0.032/+0.014) shown for completeness only; this is an 8-point event study, not a continuous strategy. 2022 (−8.9% abnormal) is the largest single-cycle contributor but stays under the 40% concentration flag. 2015 excluded (no ticker table recoverable at the time). 2024–2025 holdout reserved, untouched.",
    },
    {
        "id": "Russell-3",
        "title": "Boundary-Crossing Subset (Russell Reconstitution family)",
        "hypothesis": "Family member sharing Russell-1's DSR slot: companies that cross the Russell index-inclusion boundary multiple times across 2016–2023 isolate the pure membership/passive-flow effect more cleanly than Russell-1's additions-only design, since each company is its own control across repeated crossings.",
        "result_line": "POOLED N=1493 CAAR=+0.31% t=+0.615 | ADDITIONS-ONLY N=822 CAAR=−0.60% t=−0.958 (wrong-signed) | DELETIONS-ONLY N=671 CAAR=+1.42% t=+1.775 (wrong-signed, marginal)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "748 repeat-crosser tickers found (~39% of the full universe) — not a thin sample, contrary to the pre-registered expectation it might be under 10. No subset (pooled/additions/deletions) is materially different from Russell-1's own null. RECOMMENDATION (stated in the trial itself): close the Russell Reconstitution family — two independent constructions both failed to find a tradeable effect.",
    },
    {
        "id": "S11",
        "title": "Accruals Quality (Sloan 1996)",
        "hypothesis": "Total accruals (earnings driven by accounting adjustments rather than cash generation) predict underperformance in high-accruals firms. Selected as the highest-probability candidate to replicate GP/A's structural profile (slow-moving, XBRL-only, no price-timing).",
        "result_line": "IC: +0.024 (t=1.68, wrong-signed vs hypothesis) | Gross: −1.11%/yr | Turnover: 25%/quarter | Sharpe: −0.336 | Max DD: −40.5% | DSR: FAIL (−0.168 < 0.313, N=16)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "IC not statistically significant and opposite-signed to the hypothesis. Turnover matched GP/A's low profile as predicted, confirming this is a cost-independent test — no real signal exists to be killed by execution costs. Failure mode: genuinely null. Methodology note: an accession-number-matching bug initially inflated the annual-TTM fallback rate to 85–93%; fixed by matching on the intersection of accession numbers between NetIncomeLoss and CFO, correcting fallback to a genuine 30%.",
    },
    {
        "id": "S14",
        "title": "Analyst EPS Revisions (thinly-covered small/mid-cap)",
        "hypothesis": "In thinly-covered small/mid-cap names (1–3 analysts), EPS estimate revisions are underreacted to given how much information a single revision carries relative to consensus.",
        "result_line": "PHASE 1 FEASIBILITY CHECK ONLY — STOPPED BEFORE BACKTEST.",
        "verdict": "STOP",
        "bug": False,
        "notes": "No usable point-in-time analyst-estimate data source exists in this project. Tiingo has no analyst-estimate tier; FMP (the only analyst-estimate-adjacent vendor configured) returns HTTP 402 for every small/mid-cap ticker tested — the paywall excludes exactly the universe the hypothesis targets. Even for accessible large-caps, FMP's endpoint is a live forward-looking snapshot with no historical vintage field, unusable for a PIT backtest. Side finding: FMP's v3 statement endpoints (used elsewhere in this project) now return 403 — a pre-existing production break, flagged separately. VERDICT: kill before backtest; upgrading FMP's tier was not attempted (real cost, left as an open item).",
    },
    {
        "id": "S15",
        "title": "Composite IC (rank-percentile blend: GP/A + Residual Reversal + IVOL-Value)",
        "hypothesis": "Equal-weighted rank-percentile composite of three signals with real, correctly-signed but individually-failing IC could produce a genuinely tradeable Sharpe through diversification, even though each fails standalone for a different reason.",
        "result_line": "IC: +0.029 (t=1.64) | Gross: +6.30%/yr | Cost: 801bps/yr | Net: −1.85%/yr | Sharpe: −0.070 | Max DD: −32.6% | DSR: FAIL (−0.035 < 0.331)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Composite Sharpe (−0.070) did not beat the best single component alone (GP/A +0.0068) — diversification did not help. Component pairwise correlation GP/IVOLValue was high (+0.563), and IVOL-Value's continuous-score conversion (needed to blend into one rank) flipped its own IC sign relative to S7's original hard double-sort — both diagnosed causes were fixed in the S16-A follow-up, which still failed to rescue the family. Promotion criteria not met.",
    },
    {
        "id": "S13",
        "title": "Opportunistic Insider Cluster Buying",
        "hypothesis": "Cohen/Malloy/Pomorski (2012) routine-vs-opportunistic distinction applied to E1's naive net-insider-buying signal: cluster = ≥3 different insiders buying at the same company within 30 days. Long = confirmed cluster; short = zero insider activity.",
        "result_line": "IC: +0.014 (t=0.85) | Gross: −0.74%/yr | Cost: 780bps/yr | Net: −8.26%/yr | Sharpe: −0.857 | Max DD: −27.0% | DSR: FAIL (−0.428 < 0.456)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Reformulated signal modestly improved over E1's naive rerun (IC 0.0043→0.014) but remains statistically insignificant. Major pre-existing data-coverage gap discovered (not caused by this trial): the reused mgmt_pit Form-4 cache is a total blackout for 11 of 24 quarters — effective window only 13/24. Cluster events were not rare when data existed (822 events); sign audit found 7 violations. Concurrency note: a trial_number collision with two concurrent-session trials (S14, S15) was discovered and corrected post-hoc.",
    },
    {
        "id": "S16-A",
        "title": "Multi-Sleeve Blend (GP/A + S7-original + S8, equal 1/3 weight)",
        "hypothesis": "Three independently beta-neutral sleeves, blended at return level (not merged rank score like S15), should achieve genuine diversification by directly fixing S15's two diagnosed failure causes (component correlation, S7's sign flip under continuous scoring).",
        "result_line": "Gross: −0.64%/yr | Cost: 668bps/yr | Net: −7.12%/yr | Sharpe: −0.740 | Max DD: −39.1% | DSR: FAIL (−0.370 < 0.341)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Both diagnosed fixes confirmed working (S7/S8 correlation improved to −0.515; S7 kept its correct-signed original IC of +0.0358) — yet the blend Sharpe (−0.740) is worse than S15's already-failing composite (−0.070). Separate-sleeve construction did not rescue the family either. Universe note: genuine PIT Russell 2000 membership was requested but not reconstructable from existing archives; fell back to the standard universe, confirmed with user.",
    },
    {
        "id": "S16-B",
        "title": "Industry-Neutral GP/A (within-SIC ranking)",
        "hypothesis": "Ranking GP/A within each SIC major-group first (instead of S3-Q's full-universe rank) removes accidental sector bets, which should lower realized beta and improve Sharpe/CALMAR.",
        "result_line": "IC: +0.034 (t=2.27) | Gross: +7.49%/yr | Cost: 471bps/yr | Net: +2.63%/yr | Sharpe: 0.270 | Max DD: −20.0% | DSR: FAIL (0.135 < 0.345)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Neither motivating premise held: |beta| was not lower (−0.057 vs S3-Q's own 0.054) and Sharpe was not improved (0.270 vs S3-Q's 0.433). Sector concentration did improve mechanically (long-leg HHI 0.063 vs 0.081) but didn't translate into better risk-adjusted returns. IC held up (0.034 vs original 0.040, both significant).",
    },
    {
        "id": "S17",
        "title": "Volatility-Managed GP/A (Moreira & Muir overlay)",
        "hypothesis": "Scaling S3-Q's beta-neutral GP/A gross exposure inversely to trailing 6-month realized vol (targeting 10% annualized) improves Sharpe/CALMAR without altering stock selection.",
        "result_line": "IC: +0.046 (t=2.70) | Gross: +7.57%/yr | Cost: 273bps/yr | Net: +4.72%/yr | Sharpe: 0.481 | CALMAR: 0.329 | Max DD: −14.3% | DSR: FAIL (0.241 < 0.303, N=32)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Best risk-adjusted result of any GP/A variant tested (Sharpe 0.481, Max DD only −14.3%) but still short of the DSR bar. A parallel 'base' (no overlay) run on the identical window confirmed the overlay improves both Sharpe and CALMAR directionally (base Sharpe 0.451) — works, just not enough. 2020: the overlay reduced max-drawdown trough but underperformed base on full-year cumulative return, since backward-looking trailing vol lagged into the Q2/Q3 2020 recovery.",
    },
    {
        "id": "S18",
        "title": "Small-Cap Sector-Bucketed Pairs Trading",
        "hypothesis": "Relative-value mean-reversion between cointegrated, same-industry pairs: sector bucketing → hierarchical clustering → BH-FDR-corrected Engle-Granger cointegration → 2-consecutive-cycle persistence filter (the pair-selection analog of this registry's own DSR discipline).",
        "result_line": "STOP — zero trades under the registered methodology.",
        "verdict": "STOP",
        "bug": False,
        "notes": "Zero pairs ever achieved 2-consecutive-cycle persistence across all 10 rolling cycles (2015–2021) — not relaxed after seeing this, which would defeat the trial's own multiple-testing control. Verified not a bug: same-sector correlations in this universe are structurally much lower than large-cap pairs-trading intuition assumes (~0.2–0.3, rarely above 0.6). A diagnostic-only (non-registered) bypass of the persistence filter gave 16 trades, net Sharpe −0.454, gross near-breakeven — not a 'good signal ruined by costs' story either. Directly motivated S21's basket-relative redesign.",
    },
    {
        "id": "S19",
        "title": "52-Week-High Anchoring",
        "hypothesis": "George & Hwang (2004): investors anchor on a stock's 52-week high as a psychological ceiling; nearness to it should predict positive drift as good news is underreacted to.",
        "result_line": "IC: +0.004 (t=0.24) | Gross: −8.37%/yr | Cost: 1305bps/yr | Net: −19.77%/yr | Sharpe: −0.908 | Max DD: −75.7% | DSR: FAIL (−0.262 < 0.206)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "IC essentially null. Momentum independence check (core secondary test): nearness correlates 0.611 with S9's 12-1 momentum — contradicts the literature's 'largely independent' claim in this universe. A joint Fama-MacBeth regression flips nearness's sign once momentum is controlled for, while momentum stays correctly signed — nearness reads as a weaker, noisier restatement of momentum here, not an independent mechanism. Already losing pre-2020 too, not purely a COVID story.",
    },
    {
        "id": "S20",
        "title": "Institutional 13F 'Smart Money' Accumulation",
        "hypothesis": "A new or meaningfully increased 13F institutional position in a thinly-covered small/mid-cap name may signal information the market hasn't priced in — distinct mechanism from S13 (external professional capital vs. internal insiders).",
        "result_line": "STOP — CUSIP-to-ticker crosswalk blocked before backtest.",
        "verdict": "STOP",
        "bug": False,
        "notes": "SEC EDGAR bulk 13F data itself is free and not blocked; the CUSIP crosswalk is: FMP's /profile endpoint hit a hard account-wide quota wall, and OpenFIGI (free alternative) doesn't expose CUSIP at all. Stopped per explicit user confirmation rather than force a partial, order-biased universe. Side observation (partial sample, not validated): 82–88% of resolved names were held by ≥1 13F filer, suggesting real coverage would likely be high if solved. Unlike S14, this is a solvable-with-more-time-or-budget blocker. A free rapidfuzz-based crosswalk alternative was later confirmed feasible (78–84% coverage) but Phase 2 has not been authorized/run.",
    },
    {
        "id": "S21",
        "title": "Cluster-Conditioned Short-Term Reversion",
        "hypothesis": "Hybrid of S18 (pairs) and S8 (full-cross-section reversal): within sector-bucketed, loosely-correlated clusters, a member deviating >2.5σ from its cluster's leave-one-out average is traded long/short against the basket, exiting on reversion or a 5-day forced close.",
        "result_line": "IC: +0.021 (t=1.57) | Gross: +5.78%/yr | Cost: 10,999bps/yr | Net: −66.44%/yr | Sharpe: −17.411 (net) / 0.954 (gross) | Max DD: −99.5% | DSR: FAIL (−5.026 < 0.231)",
        "verdict": "FAIL",
        "bug": False,
        "notes": "HEADLINE FINDING: a real, positive gross signal (gross Sharpe 0.954, IC positive across all 10/10 cycles) completely destroyed by transaction costs — the most extreme version of the registry's 'signal-killed-by-cost' pattern (next-worst trial's cost drag was ~2.7x smaller), attributable entirely to trading frequency (avg 4.9-day holds; 99.6% of trades force-close on the timer rather than genuinely reverting). Successfully solved both motivating questions from S18/S8 (produced 4,683 tradeable events where S18 found zero; gross flipped from S8's negative regime to attractive) — but surfaced an even more severe cost problem. FAILS DSR by the widest margin in the registry.",
    },
    {
        "id": "S22 (DCF/P-E/P-S/P-B)",
        "title": "Long-Only Quarterly Valuation Portfolio Comparison",
        "hypothesis": "Top-20%-most-undervalued, equal-weight, long-only portfolios (genuinely new construction vs. every other beta-neutral trial in this registry) — four valuation signals (DCF margin of safety, P/E, P/S, P/B) tested head-to-head on the same underlying question.",
        "result_line": "All FAIL DSR at N=28 (per-period SR 0.194–0.246 vs threshold 0.368). Beta 1.01–1.28 (all t>19, expected for long-only). Sharpe: DCF 0.452, P/E 0.388, P/S 0.493, P/B 0.389.",
        "verdict": "FAIL",
        "bug": False,
        "notes": "Alpha vs. the equal-weight universe is negative for all four (−1.5% to −6.0%/yr) despite strongly positive alpha vs. the 3M T-bill (+15.5% to +22.6%/yr) — absolute returns are mostly high-beta exposure to a rising market, not genuine alpha. CORRECTED FINDING (2026-07-15): the original ranking (P/S > DCF > P/E ≈ P/B) was an artifact of unequal signal coverage each quarter (P/B 94.5%, P/S 90.8%, DCF 63.4%, P/E 58.9%). Re-run on the common intersection universe (mean 179/quarter) made the ranking evaporate — Sharpe clusters 0.41–0.43 for all four and the IC ranking inverts. Corrected conclusion: none of the four signals is distinguishable from the others on a fair, equal-universe basis. A ~65x DCF-backtest performance bug and a genuine PIT adapter for the existing DCF engine were also built along the way (see registry).",
    },
]

_VERDICT_COLOR = {
    "PASS (SA only — best-of-3 caveat)": "success",
    "STOP": "warning",
    "PENDING": "info",
    "FAIL": "error",
}


def _verdict_badge(verdict: str) -> None:
    color = _VERDICT_COLOR.get(verdict, "error")
    getattr(st, color)(f"**{verdict}**")


def render_phase3_summary_page() -> None:
    st.header("Phase 3 — Results Summary")

    st.info(
        "**Trials #1 through #28 logged (cumulative N_TRIALS=28)  ·  "
        "1 technical pass (S3-SA, best-of-3 caveat)  ·  4 STOP  ·  everything else FAIL**\n\n"
        "Zero trials have cleared the in-sample DSR gate in an unambiguous sense. "
        "4 trials stopped before a full backtest (H1: wrong-sign event study; S14, S20: "
        "Phase-1 data-feasibility blocks; S18: zero trades survived the persistence "
        "filter) — none pending."
    )

    # ── Master table ──────────────────────────────────────────────────────────
    rows = []
    for e in _EXPERIMENTS:
        rows.append({
            "ID":      e["id"],
            "Signal":  e["title"].split("(")[0].strip().split("[")[0].strip(),
            "Verdict": e["verdict"],
        })
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── Per-experiment cards ──────────────────────────────────────────────────
    for e in _EXPERIMENTS:
        with st.expander(f"**{e['id']}** — {e['title']}", expanded=False):
            _verdict_badge(e["verdict"])

            if e.get("bug"):
                st.caption("⚠ Period-24 rebalance-date bug — figures provisional, verdict unchanged.")

            st.markdown(f"**Hypothesis:** {e['hypothesis']}")
            st.markdown(f"**Result:** {e['result_line']}")
            st.markdown(f"**Notes:** {e['notes']}")

    st.divider()

    # ── Synthesis ─────────────────────────────────────────────────────────────
    with st.expander("Overall synthesis", expanded=False):
        st.markdown("""
**Failure taxonomy across all trials (#1–#28):**

| Mode | Trials |
|---|---|
| Wrong-sign signal (IC inverts) | E3, S9, S6, S1, S10 |
| Genuinely null / near-null (IC ≈ 0) | E1, E4, S5, S11, S13, S15, S19 |
| Cost-nonviable (real signal, cost kills) | S4, S3-Q, S6 (partly), S21 |
| Crash-prone / regime-sensitive neutralization | E2, S1, S7, S2 |
| Weak-signal-cost-decisive | S7 |
| High-IC-but-gross-negative (distinct) | S2, S8 |
| Reformulation did not beat the original | S16-A, S16-B, S17 |
| Unequal-universe artifact (corrected) | S22 (DCF/P-E/P-S/P-B) |
| Feasibility stop — no PIT data source | S14, S20 |
| Feasibility stop — construction yielded nothing | S18 |
| Event study — null/wrong-signed CAAR | Russell-1, Russell-3 |

**Blunt conclusion:** As of this writing, zero trials (out of 28 logged) have cleared
the in-sample DSR gate in an unambiguous sense. S3-SA (semiannual GP/A, Sharpe +0.569)
is the sole technical pass, but it is the best of three frequencies in a sweep —
quarterly and annual both fail. Walk-forward (2021–2025) is complete for S3-SA: IC held
up out-of-sample but a beta-drift caveat is unresolved (see the Signal Research Registry
tab). The most notable recent finding is S21 (cluster-conditioned reversion): a real,
statistically positive gross Sharpe (0.95) destroyed almost entirely by transaction
costs at ~110%/yr — the most extreme signal-vs-cost gap in the registry. S22 (long-only
valuation comparison) is a cautionary methodology example: an apparent ranking among
four valuation signals evaporated once compared on a fair, equal-coverage universe.
Three trials (S14, S18, S20) stopped before a full backtest for feasibility reasons,
not DSR failures; both members of the Russell Reconstitution family came back null and
the family is recommended closed. Nothing remains pending.
        """)
