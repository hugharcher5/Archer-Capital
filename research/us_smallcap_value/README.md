# US Small-Cap Value: A Quantitative Alpha Search

**Author:** Hugh Archer &nbsp;|&nbsp; **Period:** 2025–2026 &nbsp;|&nbsp; **Status:** Ongoing — management quality signal proceeding to bias-corrected universe

---

## Executive Summary

US small-cap stocks are inefficiently priced relative to their fundamentals. Analyst coverage thins dramatically below a $500 million market cap, sell-side models are sparse, and many names are held primarily by retail investors who do not read SEC filings systematically. This creates conditions for a quantitative strategy that reads those filings carefully and computes what the stock is worth.

Three independent hypotheses were constructed and tested, each with a distinct signal chain and data source. The universe was the set of companies in the S&P small-cap index as of mid-2026 with sufficient EDGAR filing history. All signals were computed point-in-time using only data that would have been available at each annual rebalance date.

**The most significant finding — and the most significant limitation — emerged together.** The management quality factor produced the strongest cross-sectional signal of anything tested, with a Fama-MacBeth t-statistic of 3.48 on a universe that turned out to be materially survivorship-biased. The universe was drawn from a 2026 company list, which means every company that was delisted, went bankrupt, or was acquired between 2015 and 2026 was excluded from the backtest entirely. This inflates all results, but it inflates the management signal in a particularly interesting way: the companies that survive a decade tend to be exactly the ones with good management. Stripping that bias out is the next stage of this work.

| Hypothesis | Status | Key finding |
|---|---|---|
| H1: DCF vs. Free Multiples | Inconclusive | DCF and EBITDA/EV perform similarly at the raw valuation level; DCF edges ahead by 3.2% per year once a quality filter is applied |
| H2: Monte Carlo Dispersion | No Signal | High uncertainty (wide CV) outperforms low uncertainty within cheap stocks; CV is a risk premium proxy, not information from the simulation |
| H3: Management Quality | Strongest Signal (Biased) | Fama-MacBeth t-stat of 3.48, highest of any factor tested; result is qualified by survivorship bias and is being retested on a clean universe |

---

## Scoreboard

All three hypotheses run once on full data (June 30 rebalance dates, 2015 through 2024). No parameters tuned after seeing results. Absolute return figures are upper bounds due to survivorship bias; return differentials are partially robust to this (the bias is common to both arms of each comparison).

| Hypothesis | Status | Key statistic | Why it matters |
|---|---|---|---|
| H1: DCF vs. Free Multiples | Inconclusive | DCF+Quality: 24.1% ann. return, Sharpe 0.58. EBITDA/EV+Quality: 18.3%, Sharpe 0.45. Raw DCF: 18.8%, EBITDA/EV: 19.9% | The DCF engine earns its complexity when combined with quality screening but does not outperform a simple ratio on its own |
| H2: Monte Carlo Dispersion | No Signal | Q1 (tight CV): 29.0%, Sharpe 0.39. Q5 (wide CV): 42.5%, Sharpe 0.68. Fama-MacBeth CV t-stat: 0.58 | High uncertainty predicts higher returns, consistent with risk premium. CV is significant in non-DCF baskets too, confirming it is not a simulation-specific edge |
| H3: Management Quality | Biased but Strong | Mgmt score: t=3.48***, coef=+0.031. F-Score: t=0.91 (non-significant). Dilution screen: 28.7% ann. return, Sharpe 0.72 | Management quality was the standout factor in a six-variable Fama-MacBeth. No dilution (M2) was the most powerful individual component |

---

## Hypothesis Results

### H1: Does a calibrated DCF engine outperform free valuation multiples in cross-sectional US small-cap stock selection?

**Inconclusive**

**Mechanism.** A carefully constructed DCF model, applied point-in-time with only data available from SEC filings at each rebalance date, should produce intrinsic value estimates that differ meaningfully from simple ratios like EBITDA/EV. A firm's value is the present value of its future cash flows, and modelling those cash flows explicitly should identify cheap stocks that a backward-looking multiple misses. The hypothesis was that a top-quintile DCF V/P basket would outperform a top-quintile EBITDA/EV basket by a margin that survives a Fama-MacBeth regression.

**Chain of logic:**
```
EDGAR 10-K filings (PIT, filed date <= rebalance)
  ↓
Point-in-time fundamental extraction (revenue, EBIT, D&A, CapEx, NWC)
  ↓
Calibrated WACC (synthetic credit rating from interest coverage)
  ↓
Deterministic DCF: intrinsic value per share in USD
  ↓
Signal: DCF V/P = intrinsic value / market price
  ↓
Top quintile sorted annually, equal-weight, held one year
```

**Data.** SEC EDGAR XBRL company facts API: all 10-K and 10-K/A filings for 686 US small-cap companies, 2015 through 2024 (ten rebalance dates). Point-in-time filtering: only filings with a filed date on or before each June 30 rebalance were used. Market prices and betas from yfinance. Risk-free rate from the 10-year Treasury yield (^TNX).

**What was tested.** Three signals in a horse race: DCF V/P, EBITDA/EV, and B/P. Each ranked annually across the universe; top quintile held equal-weight for one year. Fama-MacBeth cross-sectional regression of forward return on all three signals plus SIZE and MOM, run on ten annual cross-sections.

**Results:**

| Basket | Ann. Return | Sharpe | MaxDD | vs Benchmark |
|---|---|---|---|---|
| DCF V/P cheap | 18.8% | 0.42 | -21.9% | +7.9% |
| EBITDA/EV cheap | 19.9% | 0.45 | -19.0% | +9.1% |
| Benchmark (EW universe) | 10.9% | 0.32 | -25.1% | |

At the raw quintile level, EBITDA/EV marginally outperforms DCF V/P in both return and Sharpe. The DCF model closes this gap and opens a lead when quality signals are layered on top: the DCF arm of the combined value-plus-quality basket (see H3) reaches 24.1% versus 18.3% for the EBITDA/EV arm, a differential of 5.8 percentage points.

In the Fama-MacBeth regression, neither DCF V/P nor EBITDA/EV is individually significant after controlling for SIZE and MOM (DCF V/P t=1.27, EBITDA/EV t=0.50). SIZE dominates the cross-sectional regression (t=6.99***). This suggests that within a small-cap universe, the size premium is the primary driver of cross-sectional returns, and valuation signals provide incremental but modest lift.

**Verdict.** The DCF engine does not beat a free multiple in a straight head-to-head on the pre-quality basket. The gap emerges when quality filtering selects for companies whose business model is stable enough to be modelled reliably. A DCF requires forward-looking inputs; those inputs are most reliable for companies with consistent revenue and margin histories, which also tend to be companies that pass a quality screen. The DCF adds most value as a precision tool applied within a quality-filtered universe.

**What would change this verdict:** A larger universe with more cross-sectional variation in DCF applicability, and a properly debiased dataset including delisted companies. The current result may understate DCF accuracy on the surviving firms if the selection effect is correlated with forecastability.

---

### H2: Does the spread of the Monte Carlo intrinsic value distribution predict returns within a value-selected basket?

**No Signal**

**Mechanism.** A single DCF output is a point estimate; it says nothing about how confident the model is. If the PERT-based Monte Carlo simulation is run with 2,000 paths for each company, the distribution of simulated intrinsic values tells you something different from the mean: the coefficient of variation (CV = standard deviation divided by mean) measures how much the value changes as inputs are perturbed. Low CV means the model is confident. High CV means the business is harder to forecast. The hypothesis was that tight uncertainty (low CV) within cheap stocks should produce better risk-adjusted returns than wide uncertainty, because the model is doing useful work when it is confident and adding noise when it is not.

**Chain of logic:**
```
2,000-path Monte Carlo DCF per (ticker, rebalance date)
  ↓
PERT distributions for revenue growth, EBIT margin, target margin,
terminal growth, and WACC with Gaussian copula correlations
  ↓
CV = std(simulated intrinsic values) / mean(simulated intrinsic values)
  ↓
Sort DCF-cheap basket by CV (Q1 = tight, Q5 = wide)
  ↓
Does Q1 outperform Q5? Does CV survive Fama-MacBeth controls?
```

**Data.** The same EDGAR facts cache used in H1, with the full Monte Carlo simulation layer from the production DCF engine. A vectorised batch implementation was written to run 2,000 paths per ticker-date pair in approximately 5 milliseconds per call, making the full 3,579-row computation feasible in under two minutes. CV was cached after the first run.

**Coverage:** 3,514 of 3,579 DCF-applicable rows produced valid MC distributions (98%). Realized 252-day volatility and EDGAR leverage (total debt divided by total assets) were added as control variables for the Fama-MacBeth gate test.

**Results (Test 1: horse race within DCF-cheap):**

| Quintile | Ann. Return | Sharpe | MaxDD | vs Benchmark |
|---|---|---|---|---|
| Q1 (tight, low CV) | 29.0% | 0.39 | -23.9% | +14.2% |
| Q2 | 18.8% | 0.43 | -26.1% | +4.0% |
| Q3 | 15.0% | 0.34 | -33.3% | +0.2% |
| Q4 | 21.8% | 0.39 | -26.7% | +7.0% |
| Q5 (wide, high CV) | 42.5% | 0.68 | -21.1% | +27.7% |
| Benchmark (EW DCF-applicable) | 14.8% | 0.37 | -25.1% | |

**Headline: Q1-minus-Q5 differential = -13.6 percentage points.** Wide CV outperformed tight CV by 13.6% per year. The direction is the opposite of the hypothesis.

**Results (Test 2: Fama-MacBeth with full controls):**

| Factor | Mean coef | t-stat | |
|---|---|---|---|
| DCF V/P | -0.067 | -2.22 | ** |
| CV (dispersion) | +0.012 | +0.58 | |
| Realized vol | +0.030 | +0.48 | |
| Leverage | +0.010 | +0.46 | |
| SIZE | -0.216 | -4.91 | *** |
| MOM | -0.014 | -0.71 | |

CV coefficient: positive (+0.012) but not significant (t=0.58). Adding volatility and leverage controls reduces the CV coefficient from +0.015 to +0.009 (38% absorbed by risk proxies). The residual is economically small and statistically indistinguishable from zero.

**Robustness check (Test 4):** When the same CV sort is applied inside an EBITDA/EV-cheap basket rather than a DCF-cheap basket, Q5 again outperforms Q1 by 15.0% per year, and CV carries a Fama-MacBeth t-statistic of 2.11 inside the EBITDA/EV universe. CV is significant in a non-DCF-selected basket.

**Verdict.** The Monte Carlo distribution carries no information beyond the risk already captured by realized volatility and leverage. Higher CV means higher business uncertainty, and uncertain businesses earned higher returns over this period as compensation for that uncertainty. This is a risk premium, not a simulation-specific edge. The fact that CV predicts returns equally well inside an EBITDA/EV basket confirms that the signal does not depend on having run the DCF at all. The complexity of 2,000-path simulation is not rewarded in the cross-section.

**What would change this verdict:** An analysis of the full distribution shape (skewness, left-tail probability) rather than just CV might find non-linear information that CV does not capture. The tail structure of the simulation could matter even if the first two moments do not.

---

### H3: Does quantifiable management quality predict returns in US small-cap stocks, and does it survive alongside valuation and risk controls?

**Strongest Signal in Sample — Biased Universe**

**Mechanism.** Sell-side analyst coverage of small-cap stocks is thin. A company with a $200 million market cap may have zero formal analyst coverage. In this environment, the quality and alignment of management is priced slowly and incompletely by the market. Insiders who buy their own stock in the open market are making a direct financial commitment with personal capital. Management teams that grow the business without diluting shareholders are demonstrating that they are not extracting value through equity compensation. Companies where return on invested capital is improving are compounding efficiently. These signals are all contained in public SEC filings and are not systematically processed by the market at this scale.

**Chain of logic:**
```
Form 4 insider transaction filings (SEC EDGAR XBRL, PIT)
  ↓
M1: net insider buying in 180 days before rebalance date
EDGAR 10-K balance sheet and income statement (PIT)
  ↓
M2: diluted share count not increasing year-over-year (no dilution)
EDGAR return on invested capital from operating data (PIT)
  ↓
M3: ROIC improving year-over-year
  ↓
Management score = M1 + M2 + M3 (integer 0 to 3)
  ↓
Fama-MacBeth regression vs forward return: does Mgmt add after DCF V/P,
EBITDA/EV, SIZE, MOM, and F-Score are already included?
```

**Data.** SEC EDGAR submissions API for Form 4 insider transactions. EDGAR XBRL company facts for point-in-time balance sheet and income statement items (same cache as H1 and H2). Piotroski F-Score (nine binary tests from two consecutive 10-K filings) was computed alongside management score to separate the quality contribution of each. Insider transaction data required parsing XML-formatted Form 4 filings from the EDGAR full-text search index.

**What was tested.** An eight-basket horse race: two value arms (DCF V/P and EBITDA/EV) each filtered progressively through F-Score and management score thresholds. Fama-MacBeth cross-sectional regression with six factors (DCF V/P, EBITDA/EV, SIZE, MOM, F-Score, Mgmt score). Individual management component breakdown to identify which of M1, M2, M3 carried the signal.

**Results (Test 1: horse race, DCF arm):**

| Basket | Ann. Return | Sharpe | MaxDD | vs Benchmark |
|---|---|---|---|---|
| A: DCF cheap only | 18.8% | 0.42 | -21.9% | +7.9% |
| B: DCF cheap + F-Score at least 7 | 22.5% | 0.56 | -19.1% | +11.6% |
| C: DCF cheap + F-Score at least 7 + Mgmt at least 2 | 24.1% | 0.58 | -19.1% | +13.2% |
| Benchmark (EW universe) | 10.9% | 0.32 | | |

Each quality layer adds return and Sharpe. The EBITDA/EV arm does not show the same pattern: EBITDA/EV cheap (19.9%) deteriorates slightly with quality filtering (18.3% at basket C). This is the key distinction. Quality screening amplifies the DCF signal and hurts the multiple, suggesting that the DCF identifies a different and more precise set of companies than a backward-looking ratio does.

**Results (Test 3: Fama-MacBeth, six factors):**

| Factor | Mean coef | t-stat | |
|---|---|---|---|
| DCF V/P | -0.025 | -1.27 | |
| EBITDA/EV | -0.006 | -0.50 | |
| SIZE | -0.156 | -6.99 | *** |
| MOM | +0.009 | +0.18 | |
| F-Score | -0.024 | -0.91 | |
| Mgmt score | +0.031 | +3.48 | *** |

Management score was the highest t-statistic of any non-size factor, at 3.48 with significance at the 0.01 level. F-Score was non-positive and non-significant in the same regression, confirming that the two quality signals capture different information. F-Score likely overlaps substantially with value signals already in the regression; management quality does not.

**Component breakdown (Test 4a, DCF arm filtered to F-Score at least 7):**

| Component | Ann. Return | Sharpe | vs Basket B |
|---|---|---|---|
| M2 (no dilution) | 28.7% | 0.72 | +6.1% |
| M3 (ROIC improving) | 18.9% | 0.45 | -3.7% |
| Basket B (F-Score at least 7 only) | 22.5% | 0.56 | baseline |

The no-dilution screen (M2) was the most powerful individual management signal, improving both return and Sharpe substantially above the quality baseline. Insider buying (M1) produced no usable data due to sparse Form 4 filing coverage at this scale in the historical EDGAR archive.

**Verdict.** Management quality, specifically the absence of shareholder dilution, was the strongest factor in the cross-sectional regression. This result is directionally clear. The problem is that the universe was constructed from a 2026 company list, and any company that was acquired, went bankrupt, or was delisted between 2015 and 2026 does not appear in the sample. Surviving companies systematically tend to be companies with better management, which inflates the management signal far more than it inflates a pure price-to-value ratio. The t-statistic of 3.48 is interesting rather than conclusive.

**What would change this verdict:** A universe that includes all US small-cap companies that existed in 2015, including those subsequently delisted, will produce a cleaner test. That work is ongoing and is described in the Conclusion.

---

## How I Avoided Fooling Myself

**One pass, no tuning.** Each hypothesis was designed before seeing results. Signal definitions, regression specifications, and basket construction rules were fixed before running any code. The results above are the first results produced by each pipeline.

**Point-in-time data.** Every EDGAR value used in a signal was filtered to filings with a filed date on or before the rebalance date. The most recent 10-K available as of June 30 in each year was used. Revenue for fiscal year 2019 filed in February 2020 would not appear in the June 2019 rebalance but would appear in June 2020. This mirrors what an investor actually knew at each date and eliminates look-ahead bias in the fundamental data.

**Publication lag as the edge test.** Any signal that requires data that was not yet public at the rebalance date is invalid. EDGAR filings include both the fiscal year end date and the filed date; the filter runs on the filed date, not the period end date.

**Deflated Sharpe.** Multiple tests were run on the same dataset. Twelve basket definitions were tested across H1 and H3 (four baskets times two value arms). The Fama-MacBeth results for six factors are reported from a single pre-specified regression. The trial count is tracked; no individual finding should be read in isolation.

**Relative metrics where possible.** The management score uses year-over-year changes in ROIC and share count rather than absolute levels. This controls for industry and business model differences.

---

## Honest Limitations

**Survivorship bias.** This is the central limitation of the entire study. The universe was drawn from a 2026 company list. All companies delisted between 2015 and 2026 are absent. This biases absolute return estimates upward and biases the management signal most severely, since poor management is a leading indicator of delisting. Every return figure in this document should be read as an upper bound. The differentials (e.g. basket C minus basket A) are partially robust because the bias applies to both arms, but the magnitude is still unreliable.

**Insider buying data gaps.** M1 (net insider buying) produced no usable signal due to sparse Form 4 coverage in the EDGAR submission archive for the historical period. This may be a data-engineering issue rather than an absence of the signal. A direct scrape of the EDGAR full-text search index for historical Form 4 filings would resolve this.

**Small sample.** T = 10 annual cross-sections is a small sample for Fama-MacBeth inference. The t-statistic of 3.48 for management quality is computed across ten annual coefficient estimates. With autocorrelation in the coefficient series and only ten observations, the effective degrees of freedom may be closer to six or seven. The direction of the result is the primary takeaway, not the precise magnitude of significance.

**Universe concentration.** The DCF filter requires at least two years of consistent 10-K revenue and EBIT history and a positive terminal-year free cash flow. Roughly 82% of the universe (590 of 686 tickers) passes this gate on average across the ten dates. Companies that fail the gate are not missing at random: they tend to be earlier-stage, more volatile, or more financially complex. The results describe a specific subset of small-cap companies.

**No transaction costs.** All returns are gross of transaction costs, bid-ask spreads, and borrowing costs. Small-cap stocks, particularly those at the lower end of the universe, have meaningful spreads and limited liquidity. A round-trip cost of 50 to 100 basis points per name per year is realistic and would reduce reported Sharpe ratios.

---

## Conclusion

The main finding of this research is not what was originally expected. The DCF engine is valuable as a precision instrument within a quality-filtered universe, but it does not produce a standalone alpha that a simple EBITDA/EV ratio cannot approximate. The Monte Carlo distribution adds no predictive information beyond what realized volatility already captures. What did stand out, clearly, was management quality.

**The survivorship bias problem.** After completing all three hypotheses, I identified that the backtest universe was drawn from the current S&P small-cap index, which only includes companies that survived to 2026. Every company that went bankrupt, was acquired at a distressed valuation, or was delisted for non-performance over the prior decade was excluded from the analysis. This is a serious methodological flaw. It inflates returns and inflates the management signal in particular. Failing companies tend to be run by poor management before they fail; including those companies would hurt the management score and reduce the reported t-statistic. The true t-statistic is almost certainly lower than 3.48 on a debiased dataset.

**Why management is still worth investigating.** Even with full acknowledgement of the survivorship issue, the management signal dominated the Fama-MacBeth regression by a margin that is striking. The next-highest valuation factor had a t-statistic below 1.3. The management signal was the only factor other than size that passed conventional significance thresholds. In a universe where analysts are not systematically reading 10-K filings and Form 4 transactions for 200 million dollar market cap companies, the signal from shareholder-aligned management may be genuinely underpriced.

The structural case is clear. Small-cap companies receive less analyst coverage, which means signals embedded in SEC filings are processed more slowly by the market. Management quality is particularly underexplored because it does not reduce to a single line item in the income statement. Quantifying it requires reading proxy statements, Form 4 histories, and capital allocation decisions over multiple years, which is friction that reduces the number of investors acting on the signal. A disciplined, market-neutral strategy that goes long small-cap companies with strong management scores and shorts those with weak scores could produce genuine alpha with limited directional exposure to the market.

**Building the correct universe.** The path to a clean backtest runs through three free data sources that together cover the full historical population of US public companies, including those subsequently delisted. SEC DERA (the Division of Economic and Risk Analysis) provides fundamental data and insider transaction records for all registered companies from 2010 onwards, including deregistered entities, through the EDGAR bulk data download. Tiingo's free API tier provides adjusted historical prices for delisted stocks going back more than 30 years. yfinance provides current market cap and liquidity metrics for constructing the universe filter. Together these sources allow construction of a 2015 small-cap universe that includes every company trading in that year, regardless of what happened to it afterwards, with point-in-time fundamentals, insider data, and historical prices.

This work is in progress. The management quality hypothesis will be retested on that debiased universe as the primary next step.

---

## Data Sources

| Source | Data | Used in |
|---|---|---|
| SEC EDGAR XBRL API | Point-in-time 10-K filings: revenue, EBIT, D&A, CapEx, NWC, balance sheet items. Filed date filtering to eliminate look-ahead bias | H1, H2, H3 |
| SEC EDGAR Submissions API | Form 4 insider transaction XML filings for net buying computation (M1) | H3 |
| yfinance | Adjusted daily close prices for US small-cap universe (2013 to 2025), 10-year Treasury yield, SPY for beta calculation | H1, H2, H3 |
| SEC DERA bulk data | Full historical population of registered US companies including deregistered and delisted entities (next stage) | Upcoming |
| Tiingo free API | Adjusted historical prices for delisted stocks going back to 1990s (next stage) | Upcoming |

---

## Reproducibility

All analysis code lives in this repository. The pipeline is Python 3.12 with standard scientific libraries.

```
research/dcf_vs_multiple/run.py       — H1 DCF vs. free multiples horse race and Fama-MacBeth
research/monte_carlo_dispersion/run.py — H2 Monte Carlo CV signal with controls
research/quality_filter/run.py        — H3 Piotroski F-Score and management quality layering
```

All three pipelines cache results to their respective `_cache/` directories. The EDGAR facts cache is shared across all three pipelines from `research/dcf_vs_multiple/_cache/edgar_facts/`. Running any pipeline for the first time fetches and caches the required data; subsequent runs load from disk and complete in under five minutes.
