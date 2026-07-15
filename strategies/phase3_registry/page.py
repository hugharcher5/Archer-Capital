"""
Phase 3 — Signal Research Registry tab.

One collapsible card per trial, with DSR badge, metrics table, and a ≤150-word
explanation. All numbers sourced from CORRECTED_TRIAL_REGISTRY.md / underlying CSVs;
v1 S1 numbers never appear here.
"""
from __future__ import annotations
import pandas as pd
import streamlit as st


# ── Trial data ────────────────────────────────────────────────────────────────
# result: "PASS" | "FAIL" | "STOP" | "PENDING"
# bug_pending: show "[data pending re-run]" warning
# All "n/a" fields shown explicitly so tables always read as complete.

_TRIALS = [
    {
        "id": "E1",
        "name": "M1 Insider Conviction",
        "dsr_num": "1–4",
        "result": "FAIL",
        "bug_pending": True,
        "ic": "+0.0002",
        "ic_t": "0.02",
        "gross": "n/a",
        "cost_bps": "n/a",
        "net": "−0.84%/yr",
        "sharpe": "−0.075",
        "calmar": "n/a",
        "max_dd": "−32.8%",
        "beta": "n/a",
        "dsr_obs": "−0.037",
        "dsr_threshold": "0.138",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Tested whether net insider buying (SEC Form 4 open-market purchases) predicts "
            "forward quarterly returns in small/mid-cap. Hypothesis: purchases signal "
            "undervaluation not yet priced in thin-analyst-coverage names. "
            "IC ≈ 0 (t=0.02) — no cross-sectional predictive power. Net −0.84%/yr, Sharpe −0.075. "
            "The headline 'outperformance vs. benchmark' is an artifact: the benchmark "
            "(earnings-yield L/S) itself lost −11.68%/yr. Period-24 figures are contaminated by "
            "a rebalance-date bug (hold_end=2025-10-01 should be 2021-01-01); no re-run exists, "
            "but the verdict is unchanged — DSR fails by a wide margin regardless. "
            "Failure mode: genuinely null. Insider purchases in this universe either lag price "
            "moves or are too sparse to aggregate into a reliable systematic signal."
        ),
    },
    {
        "id": "E2",
        "name": "M2 Non-Dilution",
        "dsr_num": "1–4",
        "result": "FAIL",
        "bug_pending": True,
        "ic": "+0.0186",
        "ic_t": "1.82",
        "gross": "n/a",
        "cost_bps": "n/a",
        "net": "−5.46%/yr",
        "sharpe": "−0.169",
        "calmar": "n/a",
        "max_dd": "−63.5%",
        "beta": "n/a",
        "dsr_obs": "−0.084",
        "dsr_threshold": "0.138",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Tested whether firms that avoid share issuance (holding or reducing share count, "
            "preferring buybacks) outperform serial diluters. IC = +0.019 (t=1.82) — the highest "
            "in the E-family, borderline but sub-significant. Net −5.46%/yr, Sharpe −0.169, "
            "Max DD −63.5%. The period-24 bug inflated that period's M2 return to +52.3%; "
            "removing it makes results worse. "
            "Failure mode: crash-prone. Non-diluting firms cluster in sectors (financials, "
            "industrials with buyback programs) that sold off hardest in COVID, negating the "
            "positive IC. The t=1.82 IC is the only thread in the E-family worth revisiting — "
            "as a new logged trial with different normalization, not a re-run of E2."
        ),
    },
    {
        "id": "E3",
        "name": "M3 ROIC Improvement",
        "dsr_num": "1–4",
        "result": "FAIL",
        "bug_pending": True,
        "ic": "−0.0115",
        "ic_t": "−0.86",
        "gross": "n/a",
        "cost_bps": "n/a",
        "net": "−12.95%/yr",
        "sharpe": "−0.560",
        "calmar": "n/a",
        "max_dd": "−67.4%",
        "beta": "n/a",
        "dsr_obs": "−0.280",
        "dsr_threshold": "0.138",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Tested whether improving ROIC (latest filed vs. year-ago) predicts outperformance. "
            "IC = −0.012 (t=−0.86) — negative, no signal. Net −12.95%/yr, worst of all E-trials. "
            "The period-24 bug contributed M3's worst single period (−56.7%), but the strategy "
            "was already the weakest: 45.8% win rate with negative IC. "
            "Failure mode: wrong-sign. ROIC improvement in small-caps likely reflects "
            "capital-intensive expansion the market has already priced in, rather than genuine "
            "efficiency gains — the opposite of the hypothesis. No reformulation is warranted "
            "given both the wrong-sign IC and deeply negative returns."
        ),
    },
    {
        "id": "E4",
        "name": "Composite M1+M2+M3",
        "dsr_num": "1–4",
        "result": "FAIL",
        "bug_pending": True,
        "ic": "n/a",
        "ic_t": "n/a",
        "gross": "n/a",
        "cost_bps": "n/a",
        "net": "−2.09%/yr",
        "sharpe": "−0.113",
        "calmar": "n/a",
        "max_dd": "−41.8%",
        "beta": "n/a",
        "dsr_obs": "−0.057",
        "dsr_threshold": "0.138",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Tested whether compositing M1+M2+M3 rank-average scores produces a more robust "
            "management quality signal. Net −2.09%/yr, Sharpe −0.113. The composite falls "
            "between its components — better than E2 and E3, worse than E1. Compositing does "
            "not rescue the family: M3's negative IC dilutes M1+M2's weak positive signals. "
            "The benchmark (earnings-yield L/S, −11.68%/yr) that makes the 'alpha' figures look "
            "positive is itself a loser — E4's absolute return is negative. "
            "Failure mode: genuinely null. No strategy in the E-family cleared DSR; E5 "
            "(holding-period robustness) was correctly skipped per pre-registration."
        ),
    },
    {
        "id": "H1",
        "name": "Livestock Disease / Animal Pharma",
        "dsr_num": "5",
        "result": "STOP",
        "bug_pending": False,
        "ic": "n/a",
        "ic_t": "n/a",
        "gross": "n/a",
        "cost_bps": "n/a",
        "net": "n/a",
        "sharpe": "n/a",
        "calmar": "n/a",
        "max_dd": "n/a",
        "beta": "n/a",
        "dsr_obs": "n/a (event study)",
        "dsr_threshold": "n/a (event study)",
        "n_periods": "118 events (2015–2025)",
        "caar_note": "CAAR(40d, all-disease): −2.63% (t=−2.10) | CAAR(60d): −4.52% (t=−2.75)",
        "explanation": (
            "Tested the information-lag hypothesis: do WAHIS-notified livestock disease outbreaks "
            "produce positive post-event drift in listed animal-pharma names (PEAD analog)? "
            "Pre-registered 5-hypothesis tree; H1 was the load-bearing gate. "
            "Result: CAAR negative at every window (HPAI-only and all-disease pooled). "
            "All-disease 60-day: −4.52% (t=−2.75) — statistically significant in the wrong "
            "direction. Placebo confirmed clean, so this is a real effect. "
            "Failure modes: wrong-sign plus structural thin-universe problem (only 3 US investable "
            "names: PAHC, ELAN, NEOG). Even a positive CAAR would not be tradeable at 3 tickers. "
            "H2–H5 correctly never run — clean example of pre-registration preventing a dead "
            "hypothesis tree from being chased further."
        ),
    },
    {
        "id": "S9",
        "name": "Momentum 12-1, raw (JT)",
        "dsr_num": "6",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "−0.0157",
        "ic_t": "−0.82",
        "gross": "−4.87%/yr (costs not modeled)",
        "cost_bps": "n/a",
        "net": "n/a",
        "sharpe": "−0.324",
        "calmar": "n/a",
        "max_dd": "−45.3%",
        "beta": "−0.063",
        "dsr_obs": "−0.162",
        "dsr_threshold": "0.197",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Run as pipeline validator: if the harness cannot replicate the most-documented "
            "factor in empirical finance, it signals a construction bug. Result: IC = −0.016 "
            "(t=−0.82), gross −4.87%/yr, Sharpe −0.324. The strategy failed, but subsequent "
            "signals produced expected IC signs in their own directions — the pipeline is "
            "not the problem. Raw 12-1 momentum genuinely inverts in US small/mid-cap over "
            "this period. The 2020 COVID reversal (−15.6% in Q4-2020) and 2016–17 value "
            "reversal both fall inside the in-sample window and dominate the return distribution. "
            "Failure mode: regime-sensitive. Momentum requires either large-cap or much longer "
            "holds to survive crash-driven reversals in this segment."
        ),
    },
    {
        "id": "S6",
        "name": "Momentum 12-1, residual (beta-stripped)",
        "dsr_num": "6",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "−0.0129",
        "ic_t": "−0.70",
        "gross": "+4.31%/yr",
        "cost_bps": "762",
        "net": "−3.34%/yr",
        "sharpe": "−0.156",
        "calmar": "n/a",
        "max_dd": "−27.4%",
        "beta": "+0.067",
        "dsr_obs": "−0.078",
        "dsr_threshold": "0.197 (shared with S9)",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Momentum family with S9 — residual variant. Beta-stripping (36-month rolling "
            "market-model residuals) aims to remove market-return contamination from the signal. "
            "Result: Sharpe −0.156 vs S9's −0.324; Max DD −27.4% vs S9's −45.3% — roughly half "
            "the drawdown. Beta-stripping confirmed working (β = +0.067, clean). Gross alpha "
            "turned positive (+4.31%) but cost drag (762bps/yr) erased it. "
            "IC remains negative (−0.013): stripping mitigates crash exposure but doesn't fix "
            "the underlying signal inversion. Key generalizable finding: momentum-crash exposure "
            "is roughly halved by beta-stripping but not eliminated — the crash mechanism isn't "
            "pure beta (rotation/short-squeeze dynamics survive stripping)."
        ),
    },
    {
        "id": "S1",
        "name": "Low IVOL  [v2, beta-neutral, corrected]",
        "dsr_num": "7",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "−0.0646",
        "ic_t": "−2.274",
        "gross": "n/a",
        "cost_bps": "n/a",
        "net": "−10.93%/yr (arithmetic)",
        "sharpe": "−0.464",
        "calmar": "n/a",
        "max_dd": "−70.8%",
        "beta": "−0.220",
        "dsr_obs": "−0.232",
        "dsr_threshold": "0.218",
        "n_periods": "24 (quarterly)",
        "v1_superseded": True,
        "explanation": (
            "Tested low-IVOL anomaly: low-idiosyncratic-volatility small/mid-caps should earn "
            "positive risk-adjusted returns vs high-IVOL lottery names. v2 uses proper beta-neutral "
            "sizing (long_notional × β_L = short_notional × β_S); v1's dollar-neutral construction produced "
            "structural β = −0.655, contaminating returns with a net-short-market bet. v2 β = −0.220. "
            "IC = −0.065 (t=−2.27) — significant negative: low-IVOL genuinely underperformed "
            "high-IVOL, opposite the hypothesis. COVID crash drove the bulk of drawdown "
            "(Q1: −27.2%, Q2: −45.1% ls_ret) as high-IVOL short positions rallied hardest. "
            "Failure mode: wrong-sign. Low-IVOL premium appears reversed in small-caps where "
            "retail participation concentrates in high-IVOL/lottery names. Do not cite v1 numbers."
        ),
    },
    {
        "id": "S4",
        "name": "MAX / Lottery Effect",
        "dsr_num": "7",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "+0.0341",
        "ic_t": "2.463",
        "gross": "+4.76%/yr",
        "cost_bps": "2,202 (B-A 2,084 + borrow 118)",
        "net": "−16.06%/yr",
        "sharpe": "−1.264",
        "calmar": "n/a",
        "max_dd": "−65.4%",
        "beta": "+0.019",
        "dsr_obs": "−0.365",
        "dsr_threshold": "0.126",
        "n_periods": "72 (monthly)",
        "explanation": (
            "Tested the MAX anomaly: stocks with extreme recent single-day returns are overpriced "
            "lottery tickets that underperform. Monthly rebalance (signal decays within weeks). "
            "IC = +0.034 (t=2.46) — the strongest, most statistically significant IC in the "
            "expected direction of any trial in the program. Gross +4.76%/yr. Yet Sharpe = −1.26. "
            "Failure is entirely cost-driven: monthly rebalance requires >65% portfolio turnover "
            "per month, generating 2,202bps/yr of friction against only 476bps gross alpha. "
            "Decay study (1M/2M/3M): longer holds reduce costs but also reduce signal efficacy — "
            "net Sharpe never clears DSR at any horizon. The MAX signal is real; strategy is "
            "cost-nonviable in small/mid-cap without a structural execution advantage."
        ),
    },
    {
        "id": "S2",
        "name": "PEAD (Post-Earnings Announcement Drift)",
        "dsr_num": "11",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "+0.0368",
        "ic_t": "7.555",
        "gross": "−4.74%/yr",
        "cost_bps": "4,079 (B-A 3,932 + borrow 146)",
        "net": "−36.59%/yr",
        "sharpe": "−2.864",
        "calmar": "n/a",
        "max_dd": "−93.2%",
        "beta": "−0.104",
        "dsr_obs": "−0.827",
        "dsr_threshold": "0.157",
        "n_periods": "72 (monthly calendar-time)",
        "explanation": (
            "High 3-day abnormal return around SEC 8-K filing predicts 45-day forward return. "
            "IC = +0.037 (t=7.55, N=42,119) — the highest IC t-stat in the program, confirming "
            "real predictive content. Yet gross = −4.74%/yr before costs. Two failures combine: "
            "(1) daily-event calendar-time structure generates massive turnover (~5.6 new "
            "events/day/leg), producing 4,079bps/yr friction; (2) regime-sensitive neutralization "
            "— pre-COVID betas broke during March 2020's vol spike, with the long leg "
            "(high-surprise/growth) crashing harder than the short. Post-hoc sanity check "
            "confirmed gross negativity is not a construction bug: equal-dollar gross also "
            "−4.30%/yr. Highest IC in program; worst portfolio implementation outcome."
        ),
    },
    {
        "id": "S5",
        "name": "Asset Growth Anomaly",
        "dsr_num": "12",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "−0.0099",
        "ic_t": "−0.517",
        "gross": "+0.62%/yr",
        "cost_bps": "613",
        "net": "−5.46%/yr",
        "sharpe": "−0.340",
        "calmar": "n/a",
        "max_dd": "−34.8%",
        "beta": "−0.160",
        "dsr_obs": "−0.170",
        "dsr_threshold": "0.282",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Tested the Cooper et al. (2008) asset-growth anomaly: firms with high YoY "
            "total-asset growth underperform. Quarterly rebalance, beta-neutral. "
            "IC = −0.010 (t=−0.52) — essentially zero, correctly signed but statistically "
            "indistinguishable from noise. Gross +0.62% (near-zero alpha); cost drag "
            "613bps/yr turns it to net −5.46%/yr. "
            "A data-quality bug was caught pre-run: LGIH carried a bogus 1,000 USD placeholder "
            "total-assets XBRL entry from its pre-reverse-merger shell period "
            "(37,970,000 pct false 'growth') — fixed with an ASSETS_FLOOR (1M USD) filter. "
            "Failure mode: genuinely null. Asset-growth anomaly is documented primarily "
            "in large-cap and may require SIC-specific controls to isolate balance-sheet "
            "expansion from organic growth in this universe."
        ),
    },
    {
        "id": "S7",
        "name": "IVOL-Conditional Value (Double-Sort)",
        "dsr_num": "13",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "+0.0358",
        "ic_t": "1.356",
        "gross": "−6.52%/yr",
        "cost_bps": "882 (B-A 721 + borrow 161)",
        "net": "−14.70%/yr",
        "sharpe": "−0.544",
        "calmar": "n/a",
        "max_dd": "−70.1%",
        "beta": "−0.085",
        "dsr_obs": "−0.272",
        "dsr_threshold": "0.291",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Tested whether value premium concentrates within high-IVOL stocks (where "
            "arbitrage costs are highest). Double-sort: top-tercile IVOL universe, then "
            "top/bottom earnings-yield tercile within it. IC = +0.036 (t=1.36) — positive "
            "but sub-threshold (t < 2.0). Gross −6.52%/yr, net −14.70%/yr, Sharpe −0.544. "
            "Secondary comparison: plain value-only (no IVOL filter) Sharpe −0.809 — so "
            "IVOL-conditioning improved Sharpe by +0.264, confirming the interaction is "
            "directionally present. Starting from high-IVOL names inherits the IVOL/MAX "
            "family's ~56%/quarter turnover structure, not Quality-style low-turnover. "
            "Failure mode: weak-signal-cost-decisive. IC is positive but not significant; "
            "costs turn a marginal edge unprofitable."
        ),
    },
    {
        "id": "S8",
        "name": "Residual Short-Term Reversal",
        "dsr_num": "14",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "+0.036",
        "ic_t": "3.397",
        "gross": "−3.08%/yr",
        "cost_bps": "761 (B-A 614 + borrow 147)",
        "net": "−10.22%/yr",
        "sharpe": "−1.107",
        "calmar": "−0.215",
        "max_dd": "−47.43%",
        "beta": "−0.011",
        "dsr_obs": "−0.320",
        "dsr_threshold": "0.175",
        "n_periods": "70 (monthly)",
        "explanation": (
            "Prior-month residual losers long, winners short. Signal = negative 1-month "
            "residual return from a rolling 36-month market-model OLS (same factor as S6). "
            "Pre-registered double filter: top/bottom decile AND |1M residual| ≥ 1.5× "
            "trailing 12M monthly-residual std — only trade real dislocations. "
            "Turnover discipline: names not crossing the threshold hold existing positions "
            "(21% actual turnover vs ~100% naive). "
            "IC = +0.036 (t=3.40) — signal is real and statistically significant, "
            "the third-highest IC t-stat in the program. Yet gross return is −3.08%/yr "
            "before costs. Cost drag (761bps/yr) deepens the loss to −10.22%/yr. "
            "Failure mode mirrors S2: real signal content, gross-negative portfolio. "
            "Liquidity isolation: return not concentrated in illiquid tercile, ruling out "
            "an Amihud-style premium as the mechanism. Beta −0.011 (near zero)."
        ),
    },
    {
        "id": "S10",
        "name": "Amihud Illiquidity",
        "dsr_num": "15",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "−0.0530",
        "ic_t": "−4.604",
        "gross": "−7.23%/yr",
        "cost_bps": "617",
        "net": "−12.97%/yr",
        "sharpe": "−1.129",
        "calmar": "−0.251",
        "max_dd": "−51.7%",
        "beta": "−0.151 (t=−2.359)",
        "dsr_obs": "−0.565",
        "dsr_threshold": "0.306",
        "n_periods": "24 (quarterly)",
        "explanation": (
            "Explicit cost-model stress test, final ranked S1–S10 program signal: long "
            "high-ILLIQ (top quintile), short low-ILLIQ (bottom quintile), pre-registered "
            "as NOT expected to pass net of costs — designed to reveal the gross-to-net "
            "erosion gap for the least-liquid corner of the universe. "
            "IC = −0.053 (t=−4.60) — the strongest IC t-stat magnitude of any trial in "
            "the program, but negative: the illiquidity premium is genuinely inverted in "
            "this sample, not merely a cost-model artifact. Gross −7.23%/yr, cost drag "
            "617bps/yr, net −12.97%/yr, Sharpe −1.129, Max DD −51.7%. "
            "Failure mode: wrong-sign, and significantly so. Least-liquid small/mid-caps "
            "underperformed the most-liquid names outright — the hypothesized illiquidity "
            "compensation is not present; if anything the reverse holds."
        ),
    },
    {
        "id": "Russell-1",
        "name": "Pre-Effective-Date Anticipatory Drift (Russell Reconstitution)",
        "dsr_num": "17 (new family)",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "n/a",
        "ic_t": "n/a",
        "gross": "+0.64%/cycle",
        "cost_bps": "50 (flat, one round-trip/name/cycle)",
        "net": "+0.14%/cycle",
        "sharpe": "+0.032",
        "calmar": "+0.014",
        "max_dd": "−10.2%",
        "beta": "n/a",
        "dsr_obs": "n/a (event study)",
        "dsr_threshold": "n/a (event study)",
        "n_periods": "8 cycles (2016–2023)",
        "caar_note": "CAAR (abnormal vs IWM, gross): +0.64% | t = +0.416 | win rate 62.5% (5/8)",
        "explanation": (
            "First trial in a new independent family. Hypothesis: stocks confirmed for "
            "addition to a Russell US index drift upward between the preliminary-list "
            "announcement and the effective date, as funds anticipate forced index buying. "
            "Equal-weight long-only basket of confirmed R3000 additions, held T+1 after the "
            "preliminary post to the historical effective date, vs IWM benchmark. "
            "CAAR t=+0.416 — not significant; no reliable drift detected. Sharpe/CALMAR "
            "(+0.032 / +0.014, abnormal-vs-IWM) are near zero and shown for completeness "
            "only — this is an event-study design (8 non-overlapping annual points, "
            "~2-4 week windows), not a continuous strategy, so these ratios carry little "
            "weight. 2022 (−8.9% abnormal) is the largest single-cycle contributor (34.5% "
            "of total |abnormal return|) but stays under the 40% concentration flag. 2015 "
            "excluded (no ticker table recoverable at the time; a later addendum found one, "
            "not yet wired in). 2024–2025 holdout NOT unlocked — baseline shows nothing."
        ),
    },
    {
        "id": "Russell-3",
        "name": "Boundary-Crossing Subset (Russell Reconstitution family)",
        "dsr_num": "17 (shares Russell-1's family slot)",
        "result": "FAIL",
        "bug_pending": False,
        "ic": "n/a",
        "ic_t": "n/a",
        "gross": "+0.31%/cycle (pooled)",
        "cost_bps": "n/a (long-only window return, no cost model — event study)",
        "net": "n/a",
        "sharpe": "n/a",
        "calmar": "n/a",
        "max_dd": "n/a",
        "beta": "n/a",
        "dsr_obs": "n/a (event study)",
        "dsr_threshold": "n/a (event study)",
        "n_periods": "1,493 priced crossing events (2016–2023)",
        "caar_note": (
            "POOLED N=1493 CAAR=+0.31% t=+0.615 win=46.0% | ADDITIONS-ONLY N=822 "
            "CAAR=−0.60% t=−0.958 (wrong-signed vs hypothesis) | DELETIONS-ONLY N=671 "
            "CAAR=+1.42% t=+1.775 (wrong-signed vs hypothesis, marginal p=0.076)"
        ),
        "explanation": (
            "Family member of Russell-1, shares its DSR slot: isolates companies that "
            "cross the Russell index-inclusion boundary (Microcap↔2000 or 2000↔1000) "
            "multiple times across 2016–2023, so each company serves as its own control "
            "across repeated crossings — a cleaner test of the pure membership/passive-"
            "flow effect than Russell-1's additions-only design. Reused Russell-1's "
            "sourced dataset entirely; added a deletions-side universe (r3000 deletions "
            "unavailable for 2018/2019, a pre-existing gap, flagged not fixed). "
            "748 repeat-crosser tickers found (~39% of the full universe) — NOT a thin "
            "sample, contrary to the pre-registered expectation this might be under 10. "
            "Result: no subset (pooled, additions-only, deletions-only) is materially "
            "different from Russell-1's own null — additions and deletions are each only "
            "marginally significant and in some cases wrong-signed vs. the hypothesis. "
            "RECOMMENDATION (stated in the trial itself): close the Russell Reconstitution "
            "family — two independent constructions (Russell-1's annual baskets, Russell-3's "
            "repeat-crosser subset) both failed to find a tradeable pre-effective-date drift."
        ),
    },
]

# S3 group is handled separately (three variants under one grouped section)
_S3_VARIANTS = [
    {
        "id": "S3-Q",
        "label": "Quarterly rebalance",
        "result": "FAIL",
        "ic": "+0.0404",
        "ic_t": "2.283",
        "gross": "+4.93%/yr",
        "cost_bps": "390",
        "net": "+0.94%/yr",
        "sharpe": "+0.134",
        "calmar": "0.033",
        "max_dd": "−28.3%",
        "beta": "−0.103",
        "n_periods": "24",
        "dsr_obs": "0.069",
        "dsr_threshold": "0.235",
        "explanation": (
            "IC = +0.040 (t=2.28) confirms GP/A predicts quarterly returns in this universe. "
            "Gross +4.93%/yr. Cost drag 390bps/yr reduces net to +0.94%, Sharpe +0.134. "
            "DSR threshold at N=8 requires 0.235 per-period SR; observed only 0.069. "
            "Failure: cost drag alone — the signal is real and correctly signed, but 390bps "
            "of execution friction erases the ~490bps gross alpha to ~94bps net."
        ),
    },
    {
        "id": "S3-SA",
        "label": "Semiannual rebalance  ★ sole DSR pass",
        "result": "PASS",
        "ic": "+0.0541",
        "ic_t": "1.697",
        "gross": "+10.16%/yr",
        "cost_bps": "318",
        "net": "+6.85%/yr",
        "sharpe": "+0.569",
        "calmar": "0.383",
        "max_dd": "−17.9%",
        "beta": "−0.078",
        "n_periods": "12",
        "dsr_obs": "0.403",
        "dsr_threshold": "0.332",
        "explanation": (
            "Semiannual rebalance cuts turnover, lowers cost drag to 318bps/yr, and allows "
            "gross return to compound higher (10.16%/yr vs 4.93% quarterly). Sharpe +0.569. "
            "DSR PASS: per-period SR 0.403 > threshold 0.332. "
            "Critical caveat: this is the best of a three-frequency sweep — quarterly and "
            "annual both fail. Selecting the frequency that passes is a form of in-sample "
            "best-of-3 optimisation even within a pre-announced framework. IC slightly "
            "weakens at semiannual (t=1.70 vs quarterly t=2.28). "
            "Walk-forward (2021–2025) now complete — see below."
        ),
        "walkforward": {
            "status": "STILL PLAUSIBLE — NOT YET CONFIRMED",
            "n_periods_is": "12", "n_periods_wf": "8", "n_periods_delta": "−4",
            "ic_is": "+0.0541 (t=1.697)", "ic_wf": "+0.0588 (t=2.385)", "ic_delta": "+0.0047 (+0.688)",
            "net_is": "+6.85%/yr", "net_wf": "+5.04%/yr", "net_delta": "−1.81pp",
            "sharpe_is": "+0.569", "sharpe_wf": "+0.449", "sharpe_delta": "−0.121",
            "max_dd_is": "−17.9%", "max_dd_wf": "−13.55%", "max_dd_delta": "+4.36pp",
            "beta_is": "−0.078 (t=−0.448)", "beta_wf": "+0.403 (t=2.730)", "beta_delta": "+0.481 (+3.178)",
            "win_rate_is": "66.67%", "win_rate_wf": "50.00%", "win_rate_delta": "−16.67pp",
            "explanation": (
                "IC held up out-of-sample — +0.059 (t=2.39) vs +0.054 in-sample (t=1.70) — "
                "genuinely encouraging since IC is rank-based and beta-agnostic, unaffected by "
                "portfolio construction. But market beta drifted from statistically insignificant "
                "(−0.078, t=−0.45) in-sample to significant (+0.403, t=2.73) out-of-sample, "
                "concentrated almost entirely in the single largest-rally half (2021 H1, +32.7% "
                "market return, 31.6% of total walk-forward |return| mass). Some of the "
                "walk-forward Sharpe (0.449 vs 0.569 in-sample) may therefore reflect "
                "uncompensated market exposure during that rally rather than validated GP/A "
                "alpha — the beta-neutral hedge appears to have broken down exactly when it "
                "mattered most. Bottom line: not falsified, not confirmed. The rank signal "
                "survived; the risk construction did not visibly hold, pending a root-cause "
                "diagnosis of the hedge."
            ),
            "forward_pointer": (
                "Beta-drift root-cause investigation in progress — see registry for updates."
            ),
        },
    },
    {
        "id": "S3-A",
        "label": "Annual rebalance",
        "result": "FAIL",
        "ic": "+0.0667",
        "ic_t": "1.115",
        "gross": "+8.61%/yr",
        "cost_bps": "~321 (estimated)",
        "net": "+5.92%/yr",
        "sharpe": "+0.361",
        "calmar": "n/a",
        "max_dd": "−18.3%",
        "beta": "n/a",
        "n_periods": "6",
        "dsr_obs": "~0.242",
        "dsr_threshold": "0.470",
        "explanation": (
            "Annual rebalance shows the highest IC of the sweep (0.067 vs 0.040/0.054), "
            "confirming GP/A is a slow-moving signal that improves at longer horizons. "
            "Net +5.92%/yr — positive return, but n=6 periods pushes the DSR threshold to "
            "0.470, the highest gate in the program. Sharpe 0.361 does not clear it. "
            "Two genuine losing years (2015: −10.6%, 2016: −18.3% net) create a streak "
            "that can't be dismissed as noise at only 6 observations. "
            "No trial_registry.csv entry exists — run_decay.py only logs clearing variants; "
            "all S3-A numbers verified directly from gp/results/gp_decay_annual.csv."
        ),
    },
]

# ── Open-Ended Research Phase (post-ranked S1–S10 program) ────────────────────
# Each trial here is an independent hypothesis with its own DSR threshold at the
# cumulative N_TRIALS count at time of run — not a pre-ranked slot in the S1–S10
# sequence.
_OPEN_ENDED_TRIALS = [
    {
        "id": "S11",
        "name": "Accruals Quality (Sloan 1996)",
        "result": "FAIL",
        "ic": "+0.0235",
        "ic_t": "1.68",
        "gross": "−1.11%/yr",
        "cost_bps": "n/a",
        "net": "n/a",
        "sharpe": "−0.336",
        "calmar": "−0.132",
        "max_dd": "−40.5%",
        "turnover": "25%/quarter",
        "n_periods": "24 (quarterly)",
        "dsr_obs": "−0.168",
        "dsr_threshold": "0.313 (N=16)",
        "explanation": (
            "Tested whether total accruals (Sloan 1996) — earnings driven by accounting "
            "adjustments rather than cash generation — predict underperformance in "
            "high-accruals firms. Selected as the highest-probability candidate to "
            "replicate GP/A's structural profile: slow-moving, XBRL-only fundamentals, "
            "no price-timing component. IC = +0.024 (t=1.68) — not statistically "
            "significant, and the point estimate is opposite-signed to the hypothesis. "
            "Turnover (25%/quarter) matched GP/A's low-turnover profile as predicted, "
            "confirming this is a cost-independent test — no real signal exists to be "
            "killed by execution costs. Gross −1.11%/yr, Sharpe −0.336; does not clear "
            "DSR (−0.168 vs 0.313 threshold at N=16). "
            "Failure mode: genuinely null. Unlike wrong-sign trials elsewhere in the "
            "registry, the point estimate is not statistically distinguishable from "
            "zero — accruals quality carries no detectable predictive content here, "
            "despite sharing GP/A's structural DNA."
        ),
        "accn_note": (
            "⚠ Methodology note: an initial accession-number-matching bug inflated the "
            "annual-TTM fallback rate to 85–93% (comparing each concept's independently "
            "most-recent filing instead of a shared accession number between "
            "NetIncomeLoss and CFO); fixed by matching on the intersection of accession "
            "numbers, correcting fallback to 30% — still above the 25% threshold, but a "
            "genuine coverage figure rather than a matching artifact."
        ),
    },
    {
        "id": "S14",
        "name": "Analyst EPS Revisions (thinly-covered small/mid-cap)",
        "result": "STOP",
        "ic": "n/a", "ic_t": "n/a", "gross": "n/a", "cost_bps": "n/a", "net": "n/a",
        "sharpe": "n/a", "calmar": "n/a", "max_dd": "n/a", "turnover": "n/a",
        "n_periods": "n/a (stopped before backtest)",
        "dsr_obs": "n/a (Phase 1 feasibility stop)",
        "dsr_threshold": "n/a",
        "explanation": (
            "Hypothesis: in thinly-covered small/mid-cap names (1–3 analysts), EPS "
            "estimate revisions are underreacted to given how much information a single "
            "revision carries relative to consensus. PHASE 1 FEASIBILITY CHECK ONLY — "
            "STOPPED BEFORE BACKTEST: no usable point-in-time analyst-estimate data "
            "source exists in this project. Tiingo (primary vendor) has no analyst-"
            "estimate tier at all. FMP (the only analyst-estimate-adjacent vendor "
            "configured) returns HTTP 402 'not available under your current "
            "subscription' for every small/mid-cap ticker tested — the paywall excludes "
            "exactly the universe the hypothesis targets. Even for large-cap (AAPL, "
            "accessible), FMP's analyst-estimates endpoint is a live forward-looking "
            "snapshot with no historical vintage/publish-date field — not usable for a "
            "2015–2020 PIT backtest at any cap tier; grades-historical caps at 10 "
            "records under current plan. SIDE FINDING: fetch_fmp() in "
            "dcf/valuation/sources.py (used by run_reconcile.py/portfolio/web.py) now "
            "returns 403 on FMP's v3 statement endpoints ('Legacy Endpoint... only for "
            "legacy users prior Aug 2025') — a pre-existing production break unrelated "
            "to this trial, flagged for separate follow-up, not fixed here. VERDICT: "
            "kill before backtest — no free/already-configured data path exists for this "
            "universe; upgrading FMP's tier was not attempted (real subscription cost, "
            "left as an open item for the user's decision)."
        ),
    },
    {
        "id": "S15",
        "name": "Composite IC (rank-percentile blend: GP/A + Residual Reversal + IVOL-Value)",
        "result": "FAIL",
        "ic": "+0.0286", "ic_t": "1.644",
        "gross": "+6.30%/yr", "cost_bps": "801", "net": "−1.85%/yr",
        "sharpe": "−0.070", "calmar": "−0.057", "max_dd": "−32.6%",
        "turnover": "59.2%/quarter", "n_periods": "24 (quarterly)",
        "dsr_obs": "−0.035", "dsr_threshold": "0.331",
        "explanation": (
            "Equal-weighted rank-percentile composite of three signals with real, "
            "correctly-signed but individually-failing IC (GP/A quality #8, residual "
            "short-term reversal S8, IVOL-conditioned value S7) — tests whether "
            "diversification produces a genuinely tradeable Sharpe even though each "
            "fails standalone for a different reason. Composite Sharpe −0.070 vs best "
            "single component alone (GP/A +0.0068) — diversification DID NOT HELP. "
            "Component pairwise correlation: GP/ResidRev −0.175, GP/IVOLValue +0.563 "
            "(high — a large share of the blend's redundancy), ResidRev/IVOLValue "
            "+0.031. IVOL-Value's continuous-score conversion (needed to blend into a "
            "single composite rank) also flipped its own IC sign relative to S7's "
            "original hard double-sort — diagnosed and fixed in the S16-A follow-up "
            "below. FAILS DSR. Promotion criteria (CALMAR≥1 or Sharpe≥0.8) not met."
        ),
    },
    {
        "id": "S13",
        "name": "Opportunistic Insider Cluster Buying",
        "result": "FAIL",
        "ic": "+0.0138", "ic_t": "0.850",
        "gross": "−0.74%/yr", "cost_bps": "780", "net": "−8.26%/yr",
        "sharpe": "−0.857", "calmar": "−0.306", "max_dd": "−27.0%",
        "turnover": "59.4%/quarter", "n_periods": "13 of 24 quarters covered (data gap)",
        "dsr_obs": "−0.428", "dsr_threshold": "0.456",
        "explanation": (
            "Cohen/Malloy/Pomorski (2012) routine-vs-opportunistic distinction applied "
            "to E1's naive net-insider-buying signal (IC≈0.0002, FAIL): cluster = ≥3 "
            "different insiders making opportunistic (non-routine) open-market purchases "
            "at the same company within a rolling 30-day window. Long = confirmed "
            "cluster event; short = zero insider-buying activity at all in the trailing "
            "quarter. Reformulated signal itself modestly improves (IC 0.0043→0.014 vs "
            "E1's naive rerun) but remains statistically insignificant (t=0.85). "
            "MAJOR DATA-COVERAGE GAP discovered (pre-existing, not caused by this "
            "trial): the reused mgmt_pit Form-4 cache is a total blackout for 11 of 24 "
            "quarters (dense only through 2017-12) — effective window is 13/24 "
            "quarters. Cluster events were NOT rare when data existed (822 events "
            "total); the guard never fired within covered quarters. SIGN AUDIT found 7 "
            "violations. FAILS DSR. CONCURRENCY NOTE: originally run as trial #18 "
            "(N=17 at task start); two concurrent-session trials (S14, S15) were logged "
            "during this trial's ~2.5hr runtime, discovered via a trial_number collision "
            "and corrected post-hoc to true N=20."
        ),
    },
    {
        "id": "S16-A",
        "name": "Multi-Sleeve Blend (GP/A + S7-original + S8, equal 1/3 weight)",
        "result": "FAIL",
        "ic": "n/a (blend of 3 pre-existing signals, no new IC)",
        "ic_t": "n/a",
        "gross": "−0.64%/yr", "cost_bps": "668", "net": "−7.12%/yr",
        "sharpe": "−0.740", "calmar": "−0.182", "max_dd": "−39.1%",
        "turnover": "n/a (blend-level; each sleeve keeps its own native turnover)",
        "n_periods": "24 (quarterly blend calendar)",
        "dsr_obs": "−0.370", "dsr_threshold": "0.341",
        "explanation": (
            "Direct follow-up to S15's failed merged-composite: three independently "
            "beta-neutral sleeves (GP/A #8, residual reversal S8, IVOL-value kept in "
            "its ORIGINAL hard-gated double-sort form — not S15's continuous-score "
            "conversion, which had flipped its IC sign) blended at pre-registered equal "
            "1/3 capital weight, tested against S15's two diagnosed failure causes. "
            "Sleeve pairwise correlation improved (S7/S8 = −0.515, vs S15's merged "
            "GP/IVOLValue = +0.563) and S7 kept its original correct-signed IC "
            "(+0.0358), confirming both fixes worked as intended — yet the blend "
            "Sharpe (−0.740) is still worse than S15's composite (−0.070). Beta vs IWM "
            "= −0.058 (t=−0.81). FAILS DSR; separate-sleeve construction did not rescue "
            "the family either. UNIVERSE NOTE: genuine PIT Russell 2000 membership was "
            "requested but not reconstructable (only annual R3000 addition/deletion "
            "events exist, no full roster) — fell back to the standard $100M–2B PIT "
            "universe, confirmed with user; IWM used as benchmark instead."
        ),
    },
    {
        "id": "S16-B",
        "name": "Industry-Neutral GP/A (within-SIC ranking)",
        "result": "FAIL",
        "ic": "+0.0336", "ic_t": "2.265",
        "gross": "+7.49%/yr", "cost_bps": "471", "net": "+2.63%/yr",
        "sharpe": "0.270", "calmar": "0.131", "max_dd": "−20.0%",
        "turnover": "29.4%/quarter", "n_periods": "24 (quarterly)",
        "dsr_obs": "0.135", "dsr_threshold": "0.345",
        "explanation": (
            "Industry-neutral reformulation of S3-Q (trial #8): rank GP/A WITHIN each "
            "SIC major-group first (groups <5 members pooled into OTHER), then build "
            "long/short from those industry-neutral percentiles, testing whether "
            "removing accidental sector bets lowers realized beta and improves "
            "Sharpe/CALMAR vs. S3-Q's full-universe rank. Result: neither improved — "
            "|beta| was NOT lower (−0.057 vs original 0.054... wait, own-sign flipped: "
            "-0.057 vs +0.054) and Sharpe was NOT improved (0.270 vs S3-Q's 0.433 at "
            "its own window). Sector concentration DID improve (long-leg HHI 0.063 vs "
            "0.081, short-leg 0.073 vs 0.126) — the industry-neutral transform worked "
            "mechanically, it just didn't translate into better risk-adjusted returns. "
            "IC held up (0.034 vs original 0.040, both significant). FAILS DSR. "
            "Independent trial (genuine reformulation, own DSR slot, not a re-tune)."
        ),
    },
    {
        "id": "S17",
        "name": "Volatility-Managed GP/A (Moreira & Muir overlay)",
        "result": "FAIL",
        "ic": "+0.0458", "ic_t": "2.699",
        "gross": "+7.57%/yr", "cost_bps": "273", "net": "+4.72%/yr",
        "sharpe": "0.481", "calmar": "0.329", "max_dd": "−14.3%",
        "turnover": "21.1%/quarter", "n_periods": "32 (quarterly, 2016–2024)",
        "dsr_obs": "0.241", "dsr_threshold": "0.303",
        "explanation": (
            "Moreira & Muir (2017) overlay: scale S3-Q's beta-neutral GP/A gross "
            "exposure inversely to trailing 6-month realized vol, targeting 10% "
            "annualized — a construction-level overlay, not a re-tune of GP/A stock "
            "selection. Best risk-adjusted result of any GP/A variant tested "
            "(Sharpe 0.481, CALMAR 0.329, Max DD only −14.3%) but still short of the "
            "DSR bar (0.241 vs 0.303 threshold at N=23). Run alongside an identical-"
            "window/universe 'base' (no overlay) comparison: base Sharpe 0.451, CALMAR "
            "0.295 — the vol overlay improved both, confirming the overlay mechanism "
            "works directionally, just not enough to clear DSR. Universe upgraded with "
            "a Russell R3000 confirmed-exit overlay (2016–2023) — NOT literal PIT "
            "Russell 2000 membership (archive lacks a full snapshot; R3000 deletions "
            "data has a 2018–2019 gap). 2020 finding: the vol overlay reduced max-"
            "drawdown trough (−17.1%→−14.3%) but underperformed base on full-year 2020 "
            "cumulative return (−3.78% vs +1.23%) — backward-looking trailing vol "
            "lagged into the Q2/Q3 2020 recovery, throttling upside more than it "
            "protected the crash quarter. Disclosed limitation: standard cost model "
            "charges turnover on rotating names, not on incremental notional from vol-"
            "scale changes on unchanged names — likely understates real cost drag here."
        ),
    },
    {
        "id": "S18",
        "name": "Small-Cap Sector-Bucketed Pairs Trading",
        "result": "STOP",
        "ic": "n/a", "ic_t": "n/a", "gross": "n/a", "cost_bps": "n/a", "net": "n/a",
        "sharpe": "n/a", "calmar": "n/a", "max_dd": "n/a", "turnover": "n/a",
        "n_periods": "0 (zero trades under registered methodology)",
        "dsr_obs": "n/a (no return series generated)",
        "dsr_threshold": "n/a",
        "explanation": (
            "Relative-value mean-reversion between cointegrated, same-industry pairs: "
            "sector bucketing → hierarchical clustering → BH-FDR-corrected Engle-"
            "Granger cointegration → a 2-consecutive-formation-cycle persistence filter "
            "(the pair-selection analog of this registry's own DSR discipline). STOP: "
            "zero pairs ever achieved 2-consecutive-cycle persistence across all 10 "
            "rolling cycles (2015–2021) — not relaxed after seeing this, which would "
            "defeat the trial's own multiple-testing control. Selectivity funnel: "
            "59,687 candidate pairs → 307 after clustering → 9 FDR-corrected survivors "
            "(across all 10 cycles) → 0 persistence-qualified. Verified not a bug: same-"
            "sector correlations in this $100M–2B universe are structurally much lower "
            "than large-cap pairs-trading intuition assumes (bucket mean corr ~0.2–0.3, "
            "rarely above 0.6). DIAGNOSTIC-ONLY supplement (not the registered result, "
            "not itself FDR-corrected): bypassing the persistence filter and trading "
            "the 9 raw survivors directly gives 16 trades, net Sharpe −0.454, gross "
            "near-breakeven (−0.54%/yr) — not a 'good signal ruined by costs' story "
            "either. This diagnostic directly motivated S21's basket-relative redesign."
        ),
    },
    {
        "id": "S19",
        "name": "52-Week-High Anchoring",
        "result": "FAIL",
        "ic": "+0.0044", "ic_t": "0.238",
        "gross": "−8.37%/yr", "cost_bps": "1305", "net": "−19.77%/yr",
        "sharpe": "−0.908", "calmar": "−0.261", "max_dd": "−75.7%",
        "turnover": "35.1%/month", "n_periods": "72 (monthly)",
        "dsr_obs": "−0.262", "dsr_threshold": "0.206",
        "explanation": (
            "George & Hwang (2004): investors anchor on a stock's 52-week high as a "
            "psychological ceiling; nearness to it (long top quintile, short bottom) "
            "should predict positive drift as good news is underreacted to. IC is "
            "essentially null (t=0.24). MOMENTUM INDEPENDENCE CHECK (the trial's core "
            "secondary test): mean cross-sectional correlation between nearness and "
            "S9's 12-1 momentum = 0.611 (82% of periods >0.5) — CONTRADICTS the "
            "literature's 'largely independent' claim in this universe. A joint Fama-"
            "MacBeth regression (return ~ nearness_rank + momentum_rank) flips "
            "nearness's sign once momentum is controlled for (b=−0.020, t=−1.64) while "
            "momentum stays correctly signed (t=1.88) — nearness reads as a weaker, "
            "noisier restatement of momentum here, not an independent mechanism. 2020 "
            "was catastrophic (−59.4% for the year), but the strategy was already "
            "losing pre-2020 too (ex-2020: ann_ret −8.07%, Sharpe −0.456) — not purely "
            "a COVID story. FAILS DSR."
        ),
    },
    {
        "id": "S20",
        "name": "Institutional 13F 'Smart Money' Accumulation",
        "result": "STOP",
        "ic": "n/a", "ic_t": "n/a", "gross": "n/a", "cost_bps": "n/a", "net": "n/a",
        "sharpe": "n/a", "calmar": "n/a", "max_dd": "n/a", "turnover": "n/a",
        "n_periods": "n/a (stopped before backtest)",
        "dsr_obs": "n/a (Phase 1 feasibility stop)",
        "dsr_threshold": "n/a",
        "explanation": (
            "Hypothesis: a new or meaningfully increased 13F institutional position in "
            "a thinly-covered small/mid-cap name may signal information the market "
            "hasn't priced in — distinct mechanism from S13 (internal insiders vs. "
            "external professional capital). SEC EDGAR bulk 13F data (free, back to "
            "2013, properly filing-date gated) is NOT the blocker — the CUSIP-to-ticker "
            "crosswalk is: FMP's /profile endpoint hit a hard account-wide quota wall "
            "('429 Limit Reach') after ~230–560 of ~560 needed lookups; OpenFIGI (free "
            "alternative) doesn't expose CUSIP at all (Bloomberg lacks redistribution "
            "rights). STOPPED per explicit user confirmation rather than forcing a "
            "partial, order-biased universe. Side observation (partial sample, NOT a "
            "validated coverage statistic): of resolved names, 81.7% (2015 universe) "
            "and 88.3% (2020 universe) were held by ≥1 13F filer — suggesting real "
            "coverage would likely be high if the crosswalk were solved. Unlike S14, "
            "this is a solvable-with-more-time-or-budget blocker (wait out the FMP "
            "quota reset, or source CUSIPs elsewhere), not a fundamental data-existence "
            "gap — flagged for a possible future revisit, Phase 2 not yet authorized. "
            "A free rapidfuzz+SEC-company-tickers crosswalk alternative was later built "
            "and confirmed feasible (78–84% coverage, threshold=97 after spot-checking "
            "out false positives at 90) but Phase 2 (full backtest) has not been run."
        ),
    },
    {
        "id": "S21",
        "name": "Cluster-Conditioned Short-Term Reversion",
        "result": "FAIL",
        "ic": "+0.0207", "ic_t": "1.566",
        "gross": "+5.78%/yr", "cost_bps": "10999", "net": "−66.44%/yr",
        "sharpe": "−17.411", "calmar": "−0.668", "max_dd": "−99.5%",
        "turnover": "n/a (4,683 round trips over 60mo, avg 4.9-day hold)",
        "n_periods": "60 (months)",
        "dsr_obs": "−5.026", "dsr_threshold": "0.231",
        "explanation": (
            "Hybrid of S18 (pairs) and S8 (full-cross-section reversal): within sector-"
            "bucketed, loosely-correlated clusters (realized avg intra-cluster corr "
            "0.328 — deliberately weaker than S18's cointegration bar), a member "
            "deviating >2.5σ from its cluster's leave-one-out average is traded "
            "long/short against the basket, exiting on reversion or a 5-day forced "
            "close. HEADLINE FINDING: this is a REAL, POSITIVE gross signal — gross "
            "Sharpe 0.954, gross CALMAR 0.653, gross ann. return +5.78%, IC=+0.021 "
            "(t=1.57, positive across all 10/10 cycles) — COMPLETELY DESTROYED by "
            "transaction costs: net Sharpe −17.41, net ann. return −66.4%, cost drag "
            "~10,999bps/yr (~110%/yr). The most extreme version of the registry's "
            "'high-quality-signal-killed-by-cost' pattern (S2's next-worst cost drag "
            "was ~4,079bps/yr, ~2.7x smaller) — attributable entirely to trading "
            "frequency (avg 4.9-day holds vs. GP/A's quarterly cadence), not signal "
            "quality: 99.6% of trades force-close on the 5-day timer rather than "
            "genuinely reverting. Successfully solved BOTH motivating questions from "
            "S18/S8 (cluster-relative referencing produced 4,683 tradeable events where "
            "S18's pairwise cointegration found zero; gross went from S8's negative "
            "regime to a genuinely attractive 0.95 Sharpe) — but surfaced an even more "
            "severe cost problem than either. FAILS DSR by the widest margin of any "
            "trial in the registry. Promotion criteria (CALMAR≥1 or Sharpe≥0.8) on net "
            "returns not met; gross alone would have cleared it easily."
        ),
    },
]


# ── Badge helpers ─────────────────────────────────────────────────────────────

def _render_badge(result: str, extra: str = "") -> None:
    msg = extra or ""
    if result == "PASS":
        st.success(f"✓ PASS — DSR cleared in-sample{('  ' + msg) if msg else ''}")
    elif result == "STOP":
        st.warning("⚠ STOP — wrong sign, testing halted (event study; no SR gate)")
    elif result == "PENDING":
        st.info("— PENDING — not yet run")
    else:
        st.error(f"✗ FAIL — DSR not cleared{('  ' + msg) if msg else ''}")


def _render_caution_badge(label: str) -> None:
    """Distinct amber/caution badge for genuinely ambiguous results — not PASS, not FAIL."""
    st.warning(f"⏳ {label}")


def _metrics_df(trial: dict) -> pd.DataFrame:
    rows = [
        ("IC",                  f"{trial['ic']} (t = {trial['ic_t']})"),
        ("Gross ann. return",   trial["gross"]),
        ("Cost drag (bps/yr)",  trial["cost_bps"]),
        ("Net ann. return",     trial["net"]),
        ("Sharpe (net, ann.)",  trial["sharpe"]),
        ("CALMAR",              trial["calmar"]),
        ("Max drawdown",        trial["max_dd"]),
        ("Market beta",         trial["beta"]),
        ("N periods",           trial["n_periods"]),
        ("DSR observed SR",     trial["dsr_obs"]),
        ("DSR threshold SR",    trial["dsr_threshold"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ── Individual trial card ─────────────────────────────────────────────────────

def _render_trial_card(trial: dict, expanded: bool = False) -> None:
    label = f"{trial['id']} — {trial['name']}"
    with st.expander(label, expanded=expanded):
        _render_badge(trial["result"])

        if trial.get("bug_pending"):
            st.caption(
                "⚠ Data pending re-run — period-24 rebalance-date bug confirmed in source "
                "files (hold_end=2025-10-01 instead of 2021-01-01). Exact figures are "
                "provisional; DSR FAIL verdict is not expected to change."
            )
        if trial.get("v1_superseded"):
            st.caption(
                "v2 corrected (beta-neutral) run shown. v1 values (Sharpe −0.560, β −0.655, "
                "MaxDD −80.0%, ann. −15.97%) are superseded and must not be cited."
            )

        if trial["result"] == "STOP" and "caar_note" in trial:
            st.markdown(f"**Event-study result:** {trial['caar_note']}")

        col_m, col_e = st.columns([1, 1])
        with col_m:
            st.dataframe(
                _metrics_df(trial),
                use_container_width=True,
                hide_index=True,
            )
        with col_e:
            st.markdown("**Why it passed / failed:**")
            st.markdown(trial["explanation"])


# ── Open-Ended Research Phase card (Turnover row instead of Market beta) ──────

def _metrics_df_open_ended(trial: dict) -> pd.DataFrame:
    rows = [
        ("IC",                  f"{trial['ic']} (t = {trial['ic_t']})"),
        ("Gross ann. return",   trial["gross"]),
        ("Cost drag (bps/yr)",  trial["cost_bps"]),
        ("Net ann. return",     trial["net"]),
        ("Sharpe (net, ann.)",  trial["sharpe"]),
        ("CALMAR",              trial["calmar"]),
        ("Max drawdown",        trial["max_dd"]),
        ("Turnover",            trial["turnover"]),
        ("N periods",           trial["n_periods"]),
        ("DSR observed SR",     trial["dsr_obs"]),
        ("DSR threshold SR",    trial["dsr_threshold"]),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _render_open_ended_card(trial: dict, expanded: bool = False) -> None:
    label = f"{trial['id']} — {trial['name']}"
    with st.expander(label, expanded=expanded):
        _render_badge(trial["result"])

        col_m, col_e = st.columns([1, 1])
        with col_m:
            st.dataframe(
                _metrics_df_open_ended(trial),
                use_container_width=True,
                hide_index=True,
            )
        with col_e:
            st.markdown("**Why it passed / failed:**")
            st.markdown(trial["explanation"])

        if trial.get("accn_note"):
            st.caption(trial["accn_note"])


# ── S3-SA walk-forward block (nested under the S3-SA variant, not a new card) ──

def _render_walkforward_block(wf: dict) -> None:
    st.markdown("##### Walk-Forward (2021–2025)")
    _render_caution_badge(wf["status"])

    comp_rows = [
        {"Metric": "N periods",              "In-Sample": wf["n_periods_is"], "Walk-Forward": wf["n_periods_wf"], "Delta": wf["n_periods_delta"]},
        {"Metric": "Mean IC (t-stat)",        "In-Sample": wf["ic_is"],        "Walk-Forward": wf["ic_wf"],        "Delta": wf["ic_delta"]},
        {"Metric": "Ann. return (net)",       "In-Sample": wf["net_is"],       "Walk-Forward": wf["net_wf"],       "Delta": wf["net_delta"]},
        {"Metric": "Sharpe (net)",            "In-Sample": wf["sharpe_is"],    "Walk-Forward": wf["sharpe_wf"],    "Delta": wf["sharpe_delta"]},
        {"Metric": "Max drawdown",            "In-Sample": wf["max_dd_is"],    "Walk-Forward": wf["max_dd_wf"],    "Delta": wf["max_dd_delta"]},
        {"Metric": "Market beta (t-stat)",    "In-Sample": wf["beta_is"],      "Walk-Forward": wf["beta_wf"],      "Delta": wf["beta_delta"]},
        {"Metric": "Win rate",                "In-Sample": wf["win_rate_is"],  "Walk-Forward": wf["win_rate_wf"],  "Delta": wf["win_rate_delta"]},
    ]
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    st.markdown(wf["explanation"])
    st.caption(wf["forward_pointer"])


# ── S3 group card ─────────────────────────────────────────────────────────────

def _render_s3_group() -> None:
    with st.expander("S3 — Gross Profitability (GP/A)  ·  frequency sweep: Q / SA / A", expanded=False):
        st.markdown(
            "GP/A = TTM gross profit ÷ total assets. Tested at three rebalance frequencies "
            "(quarterly, semiannual, annual) as a pre-announced robustness sweep — all three "
            "share DSR trial #8 (N_TRIALS=8, annualised threshold 0.470). "
            "Signal coverage: 74% of universe (65% direct GrossProfit tag, 10% Revenue−COGS, 1% TTM fallback)."
        )

        # Comparison table
        comp_rows = []
        for v in _S3_VARIANTS:
            comp_rows.append({
                "Variant":      v["id"],
                "N periods":    v["n_periods"],
                "IC (t)":       f"{v['ic']} ({v['ic_t']})",
                "Gross":        v["gross"],
                "Cost (bps/yr)": v["cost_bps"],
                "Net":          v["net"],
                "Sharpe":       v["sharpe"],
                "CALMAR":       v["calmar"],
                "Max DD":       v["max_dd"],
                "DSR obs":      v["dsr_obs"],
                "DSR thresh":   v["dsr_threshold"],
                "Result":       v["result"],
            })
        st.dataframe(
            pd.DataFrame(comp_rows),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        for v in _S3_VARIANTS:
            st.markdown(f"#### {v['id']} — {v['label']}")
            _render_badge(
                v["result"],
                extra="⚠ best-of-3 selection" if v["result"] == "PASS" else "",
            )
            st.markdown(v["explanation"])
            if "walkforward" in v:
                st.markdown("")
                _render_walkforward_block(v["walkforward"])
            st.divider()


# ── Main render ───────────────────────────────────────────────────────────────

def render_phase3_registry_page() -> None:
    st.header("Phase 3 — Signal Research Registry")

    # ── Summary banner ────────────────────────────────────────────────────────
    st.info(
        "**15 trials run  ·  1 technical pass (S3-SA, best-of-3 caveat)  ·  "
        "13 FAIL  ·  1 STOP  ·  1 PENDING (S10)**\n\n"
        "Zero of 15 tested strategies have cleared the in-sample DSR gate in an unambiguous "
        "sense.  S3-SA (semiannual GP/A) is the sole technical pass but is best-of-3 from a "
        "frequency sweep — walk-forward complete, IC held up but a beta-drift caveat is "
        "unresolved.  S10 still pending."
    )

    st.caption(
        "In-sample window: 2015-01-01 → 2021-01-01  |  Universe: US small/mid-cap (100M–2B USD), "
        "SEC XBRL + Tiingo, point-in-time  |  DSR gate: Deflated Sharpe Ratio, cumulative "
        "N_TRIALS  |  All numbers verified against source files 2026-07-06."
    )

    # ── E1–E4 group ───────────────────────────────────────────────────────────
    st.subheader("E1–E4 — Management Quality Factor  ⚠ data pending re-run")
    st.caption(
        "DSR #1–4  |  N_TRIALS=4  |  threshold 0.138 (quarterly SR)  |  "
        "Period-24 rebalance-date bug confirmed unfixed — exact figures provisional, "
        "verdict (FAIL) unchanged."
    )
    e_trials = [t for t in _TRIALS if t["id"] in ("E1", "E2", "E3", "E4")]
    for t in e_trials:
        _render_trial_card(t)

    st.divider()

    # ── H1 ────────────────────────────────────────────────────────────────────
    h1 = next(t for t in _TRIALS if t["id"] == "H1")
    st.subheader("H1 — Event Study: Livestock Disease / Animal Pharma")
    st.caption("DSR #5  |  Event study — stop criterion: wrong-sign CAAR, not SR gate")
    _render_trial_card(h1)

    st.divider()

    # ── S9 + S6 (momentum family) ─────────────────────────────────────────────
    st.subheader("S9 + S6 — Momentum Family")
    st.caption("DSR #6  |  N_TRIALS=6  |  S6 shares S9's threshold (0.197); not a new independent trial")
    for tid in ("S9", "S6"):
        t = next(x for x in _TRIALS if x["id"] == tid)
        _render_trial_card(t)

    st.divider()

    # ── S1 + S4 (IVOL/MAX family) ─────────────────────────────────────────────
    st.subheader("S1 + S4 — IVOL / MAX Family")
    st.caption("DSR #7  |  N_TRIALS=7  |  S4 shares S1's family slot; not a new independent trial")
    for tid in ("S1", "S4"):
        t = next(x for x in _TRIALS if x["id"] == tid)
        _render_trial_card(t)

    st.divider()

    # ── S3 group ──────────────────────────────────────────────────────────────
    st.subheader("S3 — Gross Profitability (GP/A)")
    st.caption("DSR #8–10  |  N_TRIALS=8 for all three  |  Frequency sweep; not separate independent trials")
    _render_s3_group()

    st.divider()

    # ── S2, S5, S7, S8 (independent) ─────────────────────────────────────────
    for tid in ("S2", "S5", "S7", "S8"):
        t = next(x for x in _TRIALS if x["id"] == tid)
        dsr_map = {
            "S2": "#11  |  N_TRIALS=11",
            "S5": "#12  |  N_TRIALS=12",
            "S7": "#13  |  N_TRIALS=13",
            "S8": "#14  |  N_TRIALS=14",
        }
        st.subheader(f"{t['id']} — {t['name']}")
        st.caption(f"DSR {dsr_map[tid]}  |  Independent trial")
        _render_trial_card(t)
        st.divider()

    # ── Pending ───────────────────────────────────────────────────────────────
    st.subheader("Pending — not yet run")
    for p in _PENDING:
        with st.expander(f"{p['id']} — {p['name']}  (rank #{p['rank']})"):
            _render_badge("PENDING")
            st.markdown(f"No results files exist for {p['id']}. Rank #{p['rank']} in the S1–S10 program.")

    st.divider()

    # ── Open-Ended Research Phase (post-ranked S1–S10 program) ────────────────
    st.subheader("Open-Ended Research Phase")
    st.caption(
        "Trials run after the original ranked S1–S10 program concluded. Each is an "
        "independent hypothesis evaluated on its own DSR threshold (N_TRIALS = cumulative "
        "count at time of run), not a pre-ranked slot."
    )
    for t in _OPEN_ENDED_TRIALS:
        _render_open_ended_card(t)

    st.divider()

    # ── Russell-1 (new family: Russell Reconstitution) ────────────────────────
    russell1 = next(t for t in _TRIALS if t["id"] == "Russell-1")
    st.subheader("Russell-1 — Event Study: Pre-Effective-Date Anticipatory Drift")
    st.caption(
        "DSR #17 (new independent family: Russell Reconstitution)  |  Event study — "
        "CAAR t-stat significance test, not an SR gate  |  Sourced via Wayback-archived "
        "FTSE Russell reconstitution PDFs (2016–2023); 2024–2025 reserved holdout untouched."
    )
    _render_trial_card(russell1)

    st.divider()

    # ── Failure taxonomy ──────────────────────────────────────────────────────
    with st.expander("Failure-mode taxonomy (all 14 trials)", expanded=False):
        rows = [
            ("Wrong-sign signal",
             "E3, S9, S6, S1",
             "IC is negative or inverted. Momentum and IVOL reversals have plausible "
             "small-cap-specific explanations (retail concentration in high-IVOL names, "
             "crash sensitivity)."),
            ("Genuinely null / near-null",
             "E1, E4, S5",
             "IC ≈ 0; no evidence the hypothesized effect exists in this sample."),
            ("Cost-nonviable (real signal, cost kills)",
             "S4, S3-Q, S6 (partially)",
             "Positive gross alpha and correctly signed IC — execution cost structure "
             "makes the strategy untradable at this cap tier."),
            ("Crash-prone / regime-sensitive neutralization",
             "E2, S1, S7, S2",
             "Large drawdowns concentrated in 2020 overwhelm period-mean performance. "
             "Static beta-neutral construction was insufficient against the COVID reversal."),
            ("Weak-signal-cost-decisive",
             "S7",
             "IC directionally right but not statistically significant; costs turn a "
             "marginal edge unprofitable rather than killing a strong one."),
            ("High-IC-but-gross-negative (distinct category)",
             "S2, S8",
             "Strong IC signal (S2 t=7.55, S8 t=3.40) yet gross return is negative. "
             "Signal content and portfolio implementability are separate questions. "
             "S8 (reversal) is gross-negative even with turnover discipline."),
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Mode", "Trials", "Description"]),
            use_container_width=True,
            hide_index=True,
        )
