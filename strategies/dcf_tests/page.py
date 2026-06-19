"""
US Small-Cap DCF Strategy Research Hub
Renders all three hypothesis write-ups in the Strategies tab.
"""

from __future__ import annotations
import streamlit as st


# ── status badge helpers ───────────────────────────────────────────────────────

_BADGE = {
    "no_signal":    ("🔴", "No Signal",         "#c0392b"),
    "no_edge":      ("🔴", "No Incremental Edge","#c0392b"),
    "biased":       ("🟠", "Biased Signal",      "#d35400"),
    "in_progress":  ("🔵", "In Progress",        "#2980b9"),
}


def _status_pill(key: str) -> None:
    icon, label, colour = _BADGE[key]
    st.markdown(
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">'
        f'{icon} {label}</span>',
        unsafe_allow_html=True,
    )


def _divider() -> None:
    st.markdown("---")


def _h2(text: str) -> None:
    st.markdown(f"### {text}")


# ── main render function ───────────────────────────────────────────────────────

def render_dcf_tests_page() -> None:

    st.markdown("## US Small-Cap DCF Strategy")
    st.caption(
        "Three quantitative hypothesis tests on US small-cap stocks using SEC EDGAR "
        "point-in-time data. Each hypothesis was run once on the full available history "
        "with no parameter tuning. Period: June 2015 to June 2024 (10 annual rebalance dates)."
    )

    _divider()

    # ── Background ─────────────────────────────────────────────────────────────
    with st.expander("Background and Motivation", expanded=True):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
**The central question**

A Monte Carlo DCF that runs 2,000 simulated cash flow paths per ticker is expensive to
build and expensive to run. The first question any serious researcher must answer before
defending that complexity is whether it actually outperforms a free multiple that any
analyst can compute in a spreadsheet. Frankel and Lee (1998) showed that an
intrinsic-value-to-price ratio predicts future abnormal returns over three years and the
effect survives controls for size and book-to-market. Xu (2007) counters directly: the
V/P ratio has no incremental power beyond its components. That debate is the gate this
research tries to force open.

**Why small-cap stocks?**

Analyst coverage thins dramatically below a 500 million dollar market cap. Sell-side
models are sparse, many names are held primarily by retail investors, and SEC filings are
processed slowly by the market. This creates conditions where a quantitative strategy that
reads those filings systematically might find an edge that does not exist in large-cap stocks.

**What the DCF engine does**

The engine extracts point-in-time fundamentals from SEC EDGAR XBRL filings: revenue, EBIT,
D&A, CapEx, and NWC from 10-K submissions available at each rebalance date. It computes a
WACC using a synthetic credit rating from the interest coverage ratio, then runs 2,000 Monte
Carlo paths with PERT-distributed inputs and Gaussian copula correlations across five
parameters: revenue growth, EBIT margin, target margin, terminal growth, and WACC.
""")

        with col_b:
            st.markdown("""
**How point-in-time data works**

Every EDGAR value used in a signal was filtered to filings with a filed date on or before
the rebalance date. The most recent 10-K available as of June 30 in each year was used.
Revenue for fiscal year 2019 filed in February 2020 would not appear in the June 2019
rebalance but would appear in June 2020. This mirrors what an investor actually knew at
each date and eliminates look-ahead bias in the fundamental data.

**Universe**

686 unique US small-cap tickers, non-financial, with sufficient EDGAR filing history.
Financials (banks, insurers, REITs) were excluded because the Piotroski score and DCF
model do not apply to their balance sheet structure. Approximately 86% of the universe
(590 tickers) passed the DCF applicability gate on average across the ten dates, requiring
at least two years of consistent revenue and positive terminal free cash flow.

**The survivorship bias problem**

The universe was drawn from a 2026 company list. Every company that was delisted, went
bankrupt, or was acquired between 2015 and 2026 is absent from the sample. All absolute
return figures in this research are upper bounds. Return differentials between arms of the
same comparison are partially robust because the bias is shared by both sides.
""")

    _divider()

    # ── Scoreboard ─────────────────────────────────────────────────────────────
    _h2("Research Scoreboard")
    st.caption("Three hypothesis tests, each run once with no parameter tuning after seeing results.")

    cols = st.columns(3)
    summaries = [
        ("H1\nDCF vs Free Multiple",   "no_edge"),
        ("H2\nQuality Filter",          "biased"),
        ("H3\nMC Dispersion",           "no_signal"),
    ]
    for col, (label, status) in zip(cols, summaries):
        with col:
            icon, lbl, colour = _BADGE[status]
            st.markdown(
                f'<div style="border:1px solid {colour};border-radius:8px;'
                f'padding:14px 10px;text-align:center;">'
                f'<div style="font-size:0.8rem;font-weight:600;white-space:pre-line;">{label}</div>'
                f'<div style="margin-top:8px;background:{colour};color:white;border-radius:8px;'
                f'padding:3px 0;font-size:0.72rem;">{icon} {lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Universe", "686 tickers")
    m2.metric("Rebalance dates", "10 annual")
    m3.metric("Period", "2015 to 2024")
    m4.metric("Top FM t-stat", "3.48 (Mgmt)")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 1 — DCF vs FREE MULTIPLE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H1: Does a calibrated DCF engine outperform a free valuation multiple?")
    _status_pill("no_edge")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

Modelling a company's future free cash flows explicitly should identify undervalued names
that a backward-looking multiple misses. A price-to-book ratio looks at assets as they
were reported. An EV/EBITDA ratio normalises for operating earnings without modelling
growth, reinvestment, or the cost of capital. A DCF takes all of those factors into
account. The hypothesis is that this precision translates into a portfolio edge that
survives after the multiple is controlled for.

**The chain of logic**
```
EDGAR 10-K filings (point-in-time, filed date on or before rebalance)
        ↓
Revenue, EBIT, D&A, CapEx, NWC extracted per filing period
        ↓
WACC from synthetic credit rating (interest coverage ratio)
        ↓
Deterministic DCF: 5-year explicit period, terminal value, net debt
        ↓
Signal: DCF intrinsic value / market price  (higher = cheaper)
        ↓
Top quintile sorted annually, equal-weight, held one year
        ↓
Fama-MacBeth: does DCF V/P survive after B/P and EBITDA/EV are included?
```

**Results: horse race**

| Signal | Ann. Return | Sharpe | vs Benchmark |
|---|---|---|---|
| DCF V/P cheap quintile | 18.9% | 0.43 | +8.0% |
| EBITDA/EV cheap quintile | 20.4% | 0.46 | +9.5% |
| B/P cheap quintile | 17.1% | 0.39 | +6.2% |
| Benchmark (equal-weight) | 10.9% | 0.32 | |

DCF underperformed the EBITDA/EV screen by 1.5% per year at the raw quintile level.

**Results: Fama-MacBeth gate (the decision point)**

| Factor | t-stat | Verdict |
|---|---|---|
| DCF V/P | 0.99 | not significant after controls |
| B/P | 0.44 | not significant |
| EBITDA/EV | 0.61 | not significant |
| SIZE | 6.69 | significant at 0.001 level |
| MOM | 0.31 | not significant |

After controlling for B/P and EBITDA/EV simultaneously, DCF V/P turns insignificant at
t=0.99. Xu (2007) wins the gate test.

**Results: double-sort within B/P terciles**

DCF V/P separates returns within each B/P tercile by 9 to 10 percentage points from
cheapest to most expensive, but this spread is driven by SIZE. Stocks cheap on DCF within
a B/P bucket are systematically smaller than those expensive on DCF within the same bucket.
""")

    with col2:
        st.error("""
**Gate verdict: Xu (2007) wins**

DCF V/P adds nothing that B/P and EBITDA/EV do not already capture in the joint
regression. The free multiple does the same job.
""")

        st.info("""
**Why SIZE dominates (t=6.69)**

Within a small-cap universe, the size premium is the primary driver of cross-sectional
returns. Both DCF and multiples identify cheap stocks that skew toward the smaller end of
the universe. The return spread you see at the quintile level is mostly compensation for
owning smaller companies, not for the valuation signal itself.
""")

        st.info("""
**When does the DCF add value?**

The gap between DCF and EBITDA/EV narrows to zero at the raw quintile level, but opens to
5.8 percentage points when quality filtering is applied (see H2). A DCF requires stable
forward-looking inputs, which are most reliable for companies that also pass a quality
screen. The engine adds most value as a precision tool within a quality-filtered universe,
not as a standalone screener.
""")

        st.markdown("""
**Verdict:** The DCF engine fails the incremental-power gate. It cannot be justified by its
marginal contribution over a free ratio. This is the single most important result of the
three hypotheses: the complexity is not earning its keep on its own.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 2 — QUALITY FILTER
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H2: Does a quality filter separate winners from value traps, and does management quality add independent power?")
    _status_pill("biased")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

Piotroski (2000) showed that applying a fundamental-health score to cheap stocks improves
the value strategy by at least 7.5% per year. About 57% of cheap stocks underperform the
market individually: the premium comes from separating genuine recovery situations from
value traps. The effect is largest in small-caps where analyst coverage is thin. The second
layer: management quality signals embedded in Form 4 transactions and annual filings are
not systematically processed for small-cap names. Insiders buying their own stock, management
teams not diluting shareholders, and improving capital allocation efficiency are signals that
take effort to extract and are therefore underpriced.

**The chain of logic**
```
Value basket: top quintile by DCF V/P or EBITDA/EV
        ↓
Layer 1: Piotroski F-Score from two consecutive 10-K filings (PIT)
  9 binary tests: profitability, leverage, liquidity, operating efficiency
  Threshold: F-Score at least 7
        ↓
Layer 2: Management quality score (0 to 3)
  M1: net insider buying in 180 days (Form 4, EDGAR)
  M2: diluted share count not rising year-over-year
  M3: ROIC improving year-over-year
  Threshold: management score at least 2
        ↓
Does each layer add return and Sharpe?
Does management add independently in a six-variable Fama-MacBeth?
```

**Results: layer contribution**

| Basket | DCF Ann. Return | DCF Sharpe | EBITDA/EV Ann. Return | EBITDA/EV Sharpe |
|---|---|---|---|---|
| A: value cheap only | 18.8% | 0.42 | 19.9% | 0.45 |
| B: plus F-Score at least 7 | 22.5% | 0.56 | 19.3% | 0.47 |
| C: plus F-Score and Mgmt at least 2 | 24.1% | 0.58 | 18.3% | 0.45 |

F-Score adds 3.7% per year in the DCF arm. Management adds a further 1.6%. In the
EBITDA/EV arm, F-Score provides no benefit. Quality screening amplifies the DCF signal and
has no effect on the multiple, the most instructive asymmetry in the study.

**Results: Fama-MacBeth gate (six factors)**

| Factor | t-stat | Verdict |
|---|---|---|
| DCF V/P | 1.27 | not significant |
| EBITDA/EV | 0.50 | not significant |
| SIZE | 6.99 | significant at 0.001 |
| MOM | 0.18 | not significant |
| F-Score | 0.91 | not significant after controls |
| Management score | 3.48 | significant at 0.01 |

**Results: management component breakdown**

| Component | Ann. Return | Sharpe | vs Basket B |
|---|---|---|---|
| M2: no dilution | 28.7% | 0.72 | +6.1% |
| M3: ROIC improving | 18.9% | 0.45 | negative 3.7% |
| Basket B (F-Score baseline) | 22.5% | 0.56 | |

The no-dilution screen is the driver. A company that grows without issuing new equity is
demonstrating capital discipline in the most direct way possible.
""")

    with col2:
        st.warning("""
**Survivorship bias hits this result hardest**

Companies that survived to 2026 are disproportionately those that did not dilute
shareholders and did not destroy capital. Excluding the failures inflates exactly the
signal being tested. The t-stat of 3.48 should be read as directional, not precise.
""")

        st.success("""
**Why management quality is still the standout result**

The management signal was the only non-size factor that passed conventional significance
thresholds across six variables. The next highest valuation factor had a t-statistic below
1.3. Even with the survivorship caveat, the margin is striking.
""")

        st.info("""
**Why F-Score does not add cross-sectional power**

F-Score adds return in the basket horse race (+3.7%) but has no independent Fama-MacBeth
coefficient after controlling for SIZE and valuation. This is consistent with F-Score
serving as a quality-of-earnings proxy that overlaps substantially with the value signals
already in the regression. Management quality does not overlap in the same way: it captures
alignment and capital discipline, not historical profitability.
""")

        st.info("""
**M1 insider buying was unavailable**

Form 4 filing coverage from the EDGAR historical archive produced only 7.8% coverage
across the universe. M1 was excluded from the effective management score. The reported
results reflect M2 plus M3 only (range 0 to 2). M1 at scale would require a direct
scrape of the EDGAR full-text search index.
""")

        st.markdown("""
**Verdict:** Management quality, specifically the no-dilution screen, was the standout
factor across all three hypotheses. The result is directionally clear and robust within
the biased universe. A debiased retest is the next step.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 3 — MONTE CARLO DISPERSION
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H3: Does the spread of the Monte Carlo distribution predict returns?")
    _status_pill("no_signal")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

A single DCF output is a point estimate that says nothing about how confident the model is.
Running 2,000 Monte Carlo paths produces a full distribution of intrinsic values. The
coefficient of variation (CV = standard deviation divided by mean) measures how much the
estimated value changes as inputs are perturbed. Low CV means the business is forecastable.
High CV means small changes in assumptions produce large changes in value. The hypothesis:
within cheap stocks, tight CV (confident undervaluation) should outperform wide CV
(uncertain undervaluation). If you are highly confident a stock is mispriced, that is a
better bet than one that is cheap under one set of assumptions and fairly valued under another.

**The chain of logic**
```
2,000-path Monte Carlo DCF per (ticker, rebalance date)
        ↓
PERT distributions: revenue growth, EBIT margin, target margin,
terminal growth, and WACC with Gaussian copula correlations
        ↓
CV = std(simulated intrinsic values) / mean(simulated intrinsic values)
        ↓
Within DCF-cheap basket (top quintile DCF V/P), sort by CV into quintiles
  Q1 = tight (low CV, high confidence)   Q5 = wide (high CV, high uncertainty)
        ↓
Does Q1 outperform Q5? Does CV survive Fama-MacBeth after vol and leverage controls?
```

**Results: horse race within DCF-cheap basket**

| Quintile | Ann. Return | Sharpe | vs Benchmark |
|---|---|---|---|
| Q1: tight (low CV) | 29.0% | 0.39 | +14.2% |
| Q2 | 18.8% | 0.43 | +4.0% |
| Q3 | 15.0% | 0.34 | +0.2% |
| Q4 | 21.8% | 0.39 | +7.0% |
| Q5: wide (high CV) | 42.5% | 0.68 | +27.7% |
| Benchmark (equal-weight DCF universe) | 14.8% | 0.37 | |

Q1 minus Q5: negative 13.6%. Wide uncertainty outperformed tight uncertainty by
13.6% per year. The direction is the opposite of the hypothesis.

**Results: Fama-MacBeth with full controls**

| Factor | Mean coefficient | t-stat | |
|---|---|---|---|
| DCF V/P | 0.067 | 2.22 | significant |
| CV (dispersion) | 0.012 | 0.58 | not significant |
| Realized volatility | 0.030 | 0.48 | not significant |
| Leverage | 0.010 | 0.46 | not significant |
| SIZE | 0.216 | 4.91 | significant at 0.001 |
| MOM | 0.014 | 0.71 | not significant |

**Results: control delta**

Without vol and leverage controls: CV coef = 0.015, t = 0.97.
With vol and leverage controls: CV coef = 0.009, t = 0.47.
38% of the already small CV effect is absorbed by known risk proxies.

**The test that falsifies the hypothesis**

If CV were carrying information specific to the Monte Carlo simulation, it should only
predict returns inside a DCF-selected basket. Running the same CV sort inside an
EBITDA/EV-cheap basket (where no DCF was used at all), Q5 again outperforms Q1 by 15.0%
per year, and the Fama-MacBeth CV coefficient is significant at t=2.11. CV is not a
simulation-specific signal.
""")

    with col2:
        st.error("""
**Gate verdict: risk premium, not Monte Carlo edge**

CV is significant even in a non-DCF basket (t=2.11 in EBITDA/EV universe). If it were
a simulation-specific edge, it should be null outside the DCF universe. It is not.
""")

        st.info("""
**What is actually driving the result**

Wide CV means high business uncertainty. High business uncertainty earned a risk premium
over 2015 to 2024, a predominantly bull market with significant small-cap dispersion.
High-uncertainty businesses earned higher returns as compensation for that uncertainty.
This is a standard risk premium: you could measure the same exposure with realized
volatility or leverage, without running 2,000 simulations.
""")

        st.info("""
**What this means for the Monte Carlo engine**

The engine is not worthless. The DCF V/P signal does work in H1 (positive quintile
spread), and in H2 the DCF arm responds to quality filtering in a way the EBITDA/EV arm
does not. But the distribution itself, the CV across 2,000 paths, adds no alpha beyond
what simpler measures already capture. The complexity of the simulation is not rewarded
in the CV dimension.
""")

        st.info("""
**Could distribution shape help?**

CV only captures the first two moments of the distribution. Left-tail probability, the
probability that simulated intrinsic value falls below current price, or skewness of the
distribution might carry non-linear information that CV does not. These are testable
extensions on the same infrastructure and data.
""")

        st.markdown("""
**Verdict:** The Monte Carlo distribution adds no predictive information beyond what
simpler risk measures capture. The CV of simulated intrinsic values is measuring
fundamental uncertainty, which also drives realized volatility. The complexity is not
earning its keep in this dimension.
""")

    _divider()

    # ── Conclusion ─────────────────────────────────────────────────────────────
    _h2("Conclusion and Next Steps")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The survivorship bias problem**

After completing all three hypotheses, I identified that the backtest universe was drawn
from the current small-cap company population. Every company that went bankrupt, was
acquired at a distressed valuation, or was delisted for non-performance over the prior
decade was excluded from the analysis. This is a serious methodological flaw. It inflates
returns across the board and inflates the management signal most severely, since poor
management is a leading indicator of exactly the kinds of corporate failure that delisting
and bankruptcy represent.

**Why management is still worth investigating**

Even with the survivorship issue acknowledged fully, the management signal dominated the
six-variable Fama-MacBeth regression by a margin that is striking. The next highest
valuation factor had a t-statistic below 1.3. Management quality was the only factor
other than SIZE that passed conventional significance thresholds, and it was the only
signal where I was not competing against a known, widely-arbitraged risk premium.

The structural case is clear. Small-cap companies receive less analyst coverage. Signals
embedded in Form 4 transactions and annual filings are processed more slowly by the market.
Quantifying management quality requires reading proxy statements, tracking Form 4 histories
over multiple years, and evaluating capital allocation decisions across business cycles.
That friction reduces the number of investors systematically acting on the signal, which is
precisely the condition that creates a durable edge.

A disciplined, market-neutral strategy that goes long small-cap companies with strong
management scores and shorts those with weak scores could produce genuine alpha with
limited directional market exposure, if validated on a dataset that includes the failures
as well as the survivors.

**Building the correct universe**

Three free data sources together cover the full historical population of US public
companies, including those subsequently delisted:

- **SEC XBRL company facts API** (`data.sec.gov/api/xbrl/companyfacts/{CIK}.json`) —
  point-in-time fundamentals for all registered companies including deregistered entities,
  accessed via the EDGAR API rather than bulk ZIPs
- **SEC Submissions API** (`data.sec.gov/submissions/CIK{CIK}.json`) — Form 4 insider
  transaction filings, fetched per company
- **Tiingo daily prices API** — raw close prices (not adjusted close, because adjusted
  close is contaminated by split and dividend factors cached against the wrong entity when
  a ticker is reused after a delisting), covering delisted stocks going back to the 1990s

Point-in-time market cap is computed as Tiingo raw close multiplied by SEC shares
outstanding at each rebalance date, not from any current-snapshot source. This removes the
survivorship bias that yfinance current market cap would reintroduce.

Together these allow construction of a 2015 small-cap universe that includes every company
that was publicly traded in that year, regardless of what happened to it afterwards, with
point-in-time fundamentals, insider transaction history, and full price series.
""")

    with col2:
        st.metric("Best FM t-stat", "3.48", "Management quality (M2+M3)")
        st.metric("No-dilution screen return", "28.7%", "Sharpe 0.72")
        st.metric("DCF vs EBITDA/EV after quality", "+3.2%", "DCF edges ahead post-filter")

        st.warning("""
**Key caveat on all numbers**

All absolute return figures are survivorship-biased upper bounds. The management t-stat
of 3.48 is directionally clear but its precise magnitude is unreliable until tested on a
debiased dataset that includes delisted companies.
""")

        st.success("""
**What holds up after bias correction**

The direction of every result should be preserved on a debiased dataset:
- DCF adds value within quality-filtered baskets
- Management quality (especially no-dilution) is the strongest signal
- MC dispersion is a risk premium, not a simulation edge

The magnitude will almost certainly compress. How much is the question the next
stage answers.
""")

        st.info("""
**Next step: debiased universe**

Retesting the management quality hypothesis on a 2015 universe that includes delisted
companies, using the SEC XBRL and Submissions APIs combined with Tiingo historical prices,
is the primary next step. This work is currently in progress.
""")

    _divider()

    # ── Reproducibility ────────────────────────────────────────────────────────
    with st.expander("Reproducibility and Data Sources"):
        st.markdown("""
**Pipeline code**

```
research/dcf_vs_multiple/run.py        — H1: DCF vs free multiples
research/quality_filter/run.py         — H2: Piotroski F-Score and management quality
research/monte_carlo_dispersion/run.py — H3: Monte Carlo CV signal
```

All three pipelines cache results to their respective `_cache/` directories.
The EDGAR facts cache is shared from `research/dcf_vs_multiple/_cache/edgar_facts/`.
Subsequent runs load from disk and complete in under five minutes.

**Data sources**

| Source | Data | Used in |
|---|---|---|
| SEC EDGAR XBRL API | Point-in-time 10-K filings with filed date filtering | H1, H2, H3 |
| SEC EDGAR Submissions API | Form 4 insider transaction filings (M1) | H2 |
| yfinance | Adjusted daily prices, 10-year Treasury yield, beta | H1, H2, H3 |
| SEC XBRL company facts API | Point-in-time fundamentals for all registered companies including deregistered entities (next stage) | Upcoming |
| SEC Submissions API (per-company) | Form 4 insider transactions for debiased universe (next stage) | Upcoming |
| Tiingo daily prices API | Raw close prices for delisted stocks; raw rather than adjusted to avoid ticker-reuse contamination (next stage) | Upcoming |
""")
