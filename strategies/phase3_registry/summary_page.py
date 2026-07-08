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
        "hypothesis": "Pending.",
        "result_line": "Not yet run.",
        "verdict": "PENDING",
        "bug": False,
        "notes": "No results files exist. Rank #10 in the S1–S10 program.",
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
        "**14 trials run  ·  1 technical pass (S3-SA, best-of-3 caveat)  ·  "
        "12 FAIL  ·  1 STOP  ·  1 PENDING (S10)**\n\n"
        "Zero of 14 tested strategies have cleared the in-sample DSR gate in an unambiguous sense."
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
**Failure taxonomy across all 13 trials:**

| Mode | Trials |
|---|---|
| Wrong-sign signal (IC inverts) | E3, S9, S6, S1 |
| Genuinely null / near-null (IC ≈ 0) | E1, E4, S5 |
| Cost-nonviable (real signal, cost kills) | S4, S3-Q, S6 (partly) |
| Crash-prone / regime-sensitive neutralization | E2, S1, S7, S2 |
| Weak-signal-cost-decisive | S7 |
| High-IC-but-gross-negative (distinct) | S2, S8 |

**Blunt conclusion:** As of this writing, zero of 14 tested strategies have cleared the
in-sample DSR gate in an unambiguous sense. S3-SA (semiannual GP/A, Sharpe +0.569)
is the sole technical pass, but it is the best of three frequencies in a sweep —
quarterly and annual both fail. Walk-forward (2021–2025) has not been run.
S8 (residual reversal, FAIL — gross-negative despite IC t=3.40) is complete.
S10 (Amihud illiquidity) remains pending.
        """)
