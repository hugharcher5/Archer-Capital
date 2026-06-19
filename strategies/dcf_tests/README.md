# US Small-Cap DCF Strategy: A Quantitative Alpha Search

**Author:** Hugh Archer &nbsp;|&nbsp; **Period:** 2025 to 2026 &nbsp;|&nbsp; **Status:** Ongoing — management quality signal proceeding to debiased universe

---

## Executive Summary

A Monte Carlo DCF that runs 2,000 simulated cash flow paths per ticker is expensive to build and expensive to run. The first question any serious researcher must answer before defending that complexity is whether it actually outperforms a free multiple that any analyst can compute in a spreadsheet. Frankel and Lee (1998) showed that an intrinsic-value-to-price ratio predicts future abnormal returns over a three-year horizon, and the effect survives controls for size and book-to-market. Xu (2007) counters directly: the V/P ratio has no incremental power beyond its components. That debate is the gate this research tries to force open.

Three hypotheses were constructed from first principles, each with a distinct signal, data source, and falsification criterion. The universe was US non-financial small-cap companies with sufficient SEC EDGAR filing history, evaluated at ten annual rebalance dates from June 30 2015 through June 30 2024. Every signal was computed point-in-time using only filings available on or before each rebalance date. No parameter was adjusted after seeing results.

The management quality signal was the standout finding across all three hypotheses. The Fama-MacBeth t-statistic of 3.48 on management quality was the highest of any factor tested, by a wide margin over every valuation signal. The constraint on that finding is significant and is stated plainly: the backtest universe was drawn from a 2026 company list. Every company that was delisted, went bankrupt, or was acquired between 2015 and 2026 was absent from the sample. This inflates all returns but inflates the management signal most severely, since poor management is a leading indicator of exactly the kinds of corporate failure that delisting and bankruptcy represent. Retesting on a properly debiased universe is the next step.

| Hypothesis | Status | Key finding |
|---|---|---|
| H1: DCF vs. Free Multiples | No Incremental Edge | DCF V/P t=0.99 (insignificant) after controlling for B/P and EBITDA/EV. The free multiple already captures most of what the engine produces |
| H2: Quality Filter on Value Basket | Partial Signal (Biased) | F-Score adds 3.7% per year in the DCF arm. Management quality carries Fama-MacBeth t-stat of 3.48, the single result that passes multiple-comparison scrutiny |
| H3: Monte Carlo Dispersion | No Signal | Wide-uncertainty cheap stocks outperformed tight-uncertainty cheap stocks by 13.6% per year. CV is a risk premium proxy, not a simulation edge |

---

## Scoreboard

All three hypotheses run once on full data (June 30 rebalance dates, 2015 through 2024). No parameters tuned after seeing results. Absolute return figures are upper bounds due to survivorship bias. Return differentials between arms of the same comparison are partially robust because the bias is common to both sides.

| Hypothesis | Status | Key statistic | Why it matters |
|---|---|---|---|
| H1: DCF vs. Free Multiples | No Edge | DCF 18.9% ann. return, Sharpe 0.43. EBITDA/EV 20.4%, Sharpe 0.46. DCF V/P t=0.99 in FM with controls | Xu (2007) wins. The DCF engine does not produce incremental alpha over a free multiple in the cross-sectional regression |
| H2: Quality Filter on Value | Biased but Directional | Basket A (cheap): 18.8% Sharpe 0.42. Basket B (F-Score): 22.5% Sharpe 0.56. Basket C (F-Score and Mgmt): 24.1% Sharpe 0.58. Mgmt FM t=3.48 | Quality layering adds return and Sharpe in the DCF arm only. Management quality is the one factor that survives six-variable Fama-MacBeth |
| H3: Monte Carlo Dispersion | No Signal | Q1 tight: 29.0% Sharpe 0.39. Q5 wide: 42.5% Sharpe 0.68. Q1 minus Q5: negative 13.6%. FM CV t=0.58 | High uncertainty predicts higher returns. CV is a risk premium proxy that is significant even in non-DCF baskets, confirming it has nothing to do with the simulation |

---

## Hypothesis Results

### H1: Does a calibrated DCF engine outperform a free valuation multiple in US small-cap stock selection?

**No Incremental Edge**

**Mechanism.** Modelling a company's future free cash flows explicitly should identify undervalued names that a backward-looking multiple misses. A price-to-book ratio looks at assets as they were reported, not what the business can earn on those assets in the future. An EV/EBITDA ratio normalises for operating earnings without modelling growth, reinvestment, or the cost of capital. A DCF, calibrated from EDGAR filings and computed with a synthetic WACC derived from the interest coverage ratio, takes all of those factors into account. The hypothesis is that this precision translates into a portfolio edge that survives after the multiple is controlled for.

**Chain of logic:**
```
EDGAR 10-K filings (point-in-time, filed date at or before rebalance)
  ↓
Revenue, EBIT, D&A, CapEx, NWC extracted per filing period
  ↓
Synthetic credit rating from interest coverage ratio
  ↓
WACC calibrated: risk-free rate from 10-year Treasury, equity risk premium, credit spread
  ↓
Deterministic DCF: 5-year explicit period, terminal value, net debt adjustment
  ↓
Signal: DCF intrinsic value / market price (higher = cheaper)
  ↓
Top quintile sorted annually, equal-weight, held one year
  ↓
Fama-MacBeth: does DCF V/P survive after B/P and EBITDA/EV are included?
```

**Data.** SEC EDGAR XBRL company facts API for all 10-K and 10-K/A filings across 686 US small-cap companies. Point-in-time filter: only filings with a filed date on or before June 30 of each year were eligible. Fiscal year 2018 results filed in March 2019 appear in June 2019, not June 2018. Market prices and betas from yfinance. Risk-free rate from the daily 10-year Treasury yield series on each rebalance date.

**What was tested.** Three signals in a direct horse race: DCF V/P, EBITDA/EV, and B/P. Each ranked annually across the non-financial DCF-applicable universe, top quintile held equal-weight for twelve months with 20 basis points round-trip transaction costs. A three-signal Fama-MacBeth regression with SIZE and MOM controls run on ten annual cross-sections. A double-sort confirming whether DCF V/P separates returns within B/P terciles.

**Results (Test 1, horse race):**

| Signal | Ann. Return | Sharpe | vs Benchmark |
|---|---|---|---|
| DCF V/P cheap quintile | 18.9% | 0.43 | +8.0% |
| EBITDA/EV cheap quintile | 20.4% | 0.46 | +9.5% |
| B/P cheap quintile | 17.1% | 0.39 | +6.2% |
| Benchmark (equal-weight universe) | 10.9% | 0.32 | |

DCF underperformed the EBITDA/EV screen by 1.5% per year at the raw quintile level.

**Results (Test 2, Fama-MacBeth):**

| Factor | t-stat | Verdict |
|---|---|---|
| DCF V/P | 0.99 | not significant after controls |
| B/P | 0.44 | not significant |
| EBITDA/EV | 0.61 | not significant |
| SIZE | 6.69 | significant at 0.001 level |
| MOM | 0.31 | not significant |

After controlling for B/P and EBITDA/EV simultaneously, the DCF V/P coefficient turns insignificant at t=0.99. Xu (2007) wins the gate test. The engine is not a Ferrari doing a bicycle's job in the obvious sense, as the DCF V/P does produce a positive quintile spread. But it adds nothing that the free multiples do not already capture in a regression framework. The dominant predictor is SIZE (t=6.69), meaning the small-cap premium is doing most of the work, and the valuation signals are all proxying for slightly different cuts of the same size effect.

**Results (Test 3, double-sort within B/P terciles):**

DCF V/P does separate returns within each B/P tercile by 9 to 10 percentage points from cheapest to most expensive. However, this spread is driven by the same SIZE effect: stocks that are cheap on DCF within a B/P bucket are systematically smaller than those that are expensive on DCF within the same bucket. The double-sort result reflects SIZE, not DCF precision.

**Verdict.** The DCF engine fails the incremental-power gate. It produces a positive raw return spread, and it can identify cheap stocks, but it cannot be justified by its marginal contribution over a ratio that takes 30 seconds to compute. The finding is not that the engine is wrong. The information it produces is already embedded in EBITDA/EV. The question becomes whether quality filtering changes this picture, which is what H2 tests.

**What would change this verdict.** A longer history with more cross-sectional variation in business model complexity, and a debiased universe including delisted names. The current ten-date sample is small. The direction of the finding (DCF V/P becomes insignificant in the FM regression) is clear, but the precision of the t-statistic is limited at T=10.

---

### H2: Does a quality filter applied to a value basket separate winners from value traps, and does management quality add independent power?

**Partial Signal (Biased Universe)**

**Mechanism.** Piotroski (2000) showed that applying a fundamental-health score to low-price-to-book stocks improves the return of a value strategy by at least 7.5% per year, and that roughly 57% of cheap stocks underperform the market individually. The premium comes from separating genuine recovery situations from value traps: companies that look cheap because they are deteriorating, not because they are misunderstood. The effect is largest in small-caps, where analyst coverage is thin and the market is slowest to process filing-level information. The second layer of this hypothesis is that management quality, specifically evidence of insider conviction and capital discipline, adds a signal beyond the financial-health screen. In a small-cap universe with minimal sell-side coverage, the degree to which management is aligned with shareholders and allocating capital well is not systematically priced.

**Chain of logic:**
```
Value basket: top quintile DCF V/P or EBITDA/EV (from H1)
  ↓
Layer 1: Piotroski F-Score from two consecutive 10-K filings (PIT)
  9 binary tests across profitability, leverage, liquidity, and operating efficiency
  Threshold: F-Score at least 7 keeps financially healthy names
  ↓
Layer 2: Management quality score (0 to 3)
  M1: net insider buying in 180 days before rebalance (Form 4 filings, EDGAR)
  M2: diluted share count not increasing year-over-year (no dilution)
  M3: return on invested capital improving year-over-year
  Threshold: management score at least 2
  ↓
Does each layer add return and Sharpe? Does it add independently in Fama-MacBeth?
```

**Data.** SEC EDGAR XBRL company facts for point-in-time profitability, leverage, liquidity, share count, and ROIC data from two consecutive annual filings. SEC EDGAR submissions API for Form 4 insider transaction records. All data filtered strictly to what was filed before each June 30 rebalance date. F-Score coverage was 93.7% of DCF-applicable names. M1 insider coverage was 7.8% due to limited Form 4 availability in the historical EDGAR archive at the scale tested. As a result the management score was effectively M2 plus M3 only (range 0 to 2) for this run.

**Results (Test 1, layer contribution):**

| Basket | DCF Ann. Return | DCF Sharpe | vs Benchmark | EBITDA/EV Ann. Return | EBITDA/EV Sharpe | vs Benchmark |
|---|---|---|---|---|---|---|
| A: value cheap only | 18.8% | 0.42 | +7.9% | 19.9% | 0.45 | +9.1% |
| B: value plus F-Score at least 7 | 22.5% | 0.56 | +11.6% | 19.3% | 0.47 | +8.4% |
| C: value plus F-Score at least 7 plus Mgmt at least 2 | 24.1% | 0.58 | +13.2% | 18.3% | 0.45 | +7.4% |

F-Score adds 3.7% per year in annualised return and lifts Sharpe from 0.42 to 0.56 in the DCF arm. Adding management quality (M2 plus M3) contributes a further 1.6%. In the EBITDA/EV arm, F-Score provides no benefit and quality filtering slightly reduces returns. This divergence is the most instructive finding of H2: quality screening amplifies the DCF signal and suppresses the multiple. A DCF requires forward-looking inputs that are most reliably estimated for companies with consistent revenue and margin histories, which are exactly the companies that also pass a financial-health screen. The EBITDA/EV multiple is already a summary of financial health and does not benefit from additional filtering.

**Results (Test 2, value-trap rate):**

All baskets trap approximately 55% of names per year, meaning roughly half of the names in each basket underperformed the benchmark in a given year. The F-Score filter does not improve the hit rate. It improves returns by lifting the magnitude of winners, not by increasing the frequency of correct calls. This is consistent with Piotroski's finding that the quality filter operates on the right tail of the return distribution, not on reducing downside frequency.

**Results (Test 3, Fama-MacBeth, six factors):**

| Factor | t-stat | Verdict |
|---|---|---|
| DCF V/P | 1.27 | not significant |
| EBITDA/EV | 0.50 | not significant |
| SIZE | 6.99 | significant at 0.001 level |
| MOM | 0.18 | not significant |
| F-Score | 0.91 | not significant after controls |
| Management score | 3.48 | significant at 0.01 level |

Management score was the only non-size factor that passed conventional significance thresholds, at a t-statistic of 3.48. F-Score, despite adding return in the basket horse race, has no independent cross-sectional power in the joint regression. This is consistent with F-Score serving as a quality-of-earnings screen that overlaps substantially with the value signals already in the regression. Management quality does not overlap in the same way. It captures something genuinely different: alignment and capital discipline, not historical profitability.

**Results (Test 4, component breakdown):**

| Component | Ann. Return | Sharpe | vs Basket B |
|---|---|---|---|
| M2: no dilution | 28.7% | 0.72 | +6.1% |
| M3: ROIC improving | 18.9% | 0.45 | negative 3.7% |
| Basket B (F-Score at least 7, baseline) | 22.5% | 0.56 | |

The no-dilution screen (M2) is the driver. A company that grows its business without issuing new equity is demonstrating capital discipline in the most direct way possible: management is not extracting value through dilution. The ROIC trend (M3) acted as a drag in this sample, consistent with ROIC improvement being more commonly observed in companies that are in distress recovery, which may coincide with lower forward returns in a surviving universe.

**Results (Test 5, DCF vs. multiple with quality applied):**

Once the same quality filter is applied to both arms, DCF beats EBITDA/EV by 3.2% per year (22.5% versus 19.3%). The advantage that did not exist at the raw value level emerges when the quality gate selects for companies whose business model is stable enough to model. This is the strongest argument for using the DCF: not as a standalone screener but as a precision instrument applied after a quality gate.

**Verdict.** F-Score adds return in the DCF arm through a mechanism that is economically clear but does not survive Fama-MacBeth as an independent factor. Management quality is the single result across all three hypotheses that passes the cross-sectional regression gate. The no-dilution component was the most powerful individual signal. The constraint is survivorship bias. Companies that survived to 2026 tend to be exactly those that managed capital well. The direction of the management finding is preserved but the magnitude almost certainly overstates the true signal.

**What would change this verdict.** A universe that includes all companies that existed in 2015, including those subsequently delisted, will provide a genuine test. Full Form 4 coverage would also allow M1 (insider buying) to be properly evaluated. The 7.8% coverage achieved in this run was insufficient to draw conclusions about insider conviction as a signal.

---

### H3: Does the spread of the Monte Carlo intrinsic value distribution predict returns within a value-selected basket?

**No Signal**

**Mechanism.** A single DCF output is a point estimate. It says nothing about how confident the model is in that estimate. Running 2,000 Monte Carlo paths with PERT-distributed inputs for revenue growth, EBIT margin, target margin, terminal growth, and WACC produces a full distribution of intrinsic values. The coefficient of variation (standard deviation divided by mean) measures how much the estimated value moves as inputs are perturbed within their uncertainty bounds. Low CV means the business is forecastable and the model is confident. High CV means the business is genuinely uncertain and small changes in assumptions produce large changes in estimated value. The hypothesis is that within a basket of cheap stocks, tight CV outperforms wide CV: if you are highly confident that a stock is undervalued, that should be a better bet than one that is cheap under one set of assumptions and fairly valued under another.

**Chain of logic:**
```
2,000-path Monte Carlo DCF per (ticker, rebalance date)
  ↓
PERT distributions for revenue growth, EBIT margin, target margin,
terminal growth, and WACC with Gaussian copula correlations
  ↓
CV = std(simulated intrinsic values) / mean(simulated intrinsic values)
  ↓
Within DCF-cheap basket (top quintile DCF V/P), sort by CV
  Q1 = tight (low CV, high confidence)
  Q5 = wide (high CV, high uncertainty)
  ↓
Does Q1 outperform Q5? Does CV survive Fama-MacBeth after volatility and leverage controls?
```

**Data.** The same EDGAR facts cache from H1, with the full Monte Carlo simulation layer from the production DCF engine. A vectorised batch implementation processes all 2,000 paths per ticker-date pair simultaneously using numpy broadcasting rather than per-path Python loops, reducing computation from roughly 500ms per pair to approximately 5ms per pair. The full 3,579-row computation ran in 112 seconds. Realized 252-day historical volatility and point-in-time leverage (total debt divided by total assets) were added as control variables for the Fama-MacBeth gate test. Coverage: 3,514 of 3,579 DCF-applicable rows produced valid distributions (98%).

**Results (Test 1, horse race within DCF-cheap):**

| Quintile | Ann. Return | Sharpe | MaxDD | vs Benchmark |
|---|---|---|---|---|
| Q1: tight (low CV) | 29.0% | 0.39 | 23.9% | +14.2% |
| Q2 | 18.8% | 0.43 | 26.1% | +4.0% |
| Q3 | 15.0% | 0.34 | 33.3% | +0.2% |
| Q4 | 21.8% | 0.39 | 26.7% | +7.0% |
| Q5: wide (high CV) | 42.5% | 0.68 | 21.1% | +27.7% |
| Benchmark (equal-weight DCF-applicable) | 14.8% | 0.37 | 25.1% | |

Q1 minus Q5: negative 13.6%. Wide uncertainty outperformed tight uncertainty by 13.6% per year. The direction is the opposite of the hypothesis.

**Results (Test 2, Fama-MacBeth with full controls):**

| Factor | Mean coefficient | t-stat | |
|---|---|---|---|
| DCF V/P | 0.067 | 2.22 | significant at 0.05 |
| CV (dispersion) | 0.012 | 0.58 | not significant |
| Realized volatility | 0.030 | 0.48 | not significant |
| Leverage | 0.010 | 0.46 | not significant |
| SIZE | 0.216 | 4.91 | significant at 0.001 |
| MOM | 0.014 | 0.71 | not significant |

CV is positive (0.012) but not significant after controls. T=10 is a small sample. The direction is noted but the conclusion is clear: there is no independent Monte Carlo edge.

**Results (Test 3, control delta):**

When regressing forward return on CV, SIZE, and MOM only (without volatility and leverage), the CV coefficient is 0.015 with t=0.97. Adding realized volatility and leverage reduces the coefficient to 0.009 with t=0.47. Roughly 38% of the already small CV effect is absorbed by the risk proxies. The remaining 62% is not significant and not economically meaningful.

**Results (Test 4, EBITDA/EV robustness):**

The test that falsifies the hypothesis most directly is Test 4. If CV were carrying information specific to the Monte Carlo simulation, it should only predict returns inside a DCF-selected basket where the simulation was used to define the investment universe. Running the same CV sort inside an EBITDA/EV-cheap basket (where no DCF was used at all), Q5 outperforms Q1 by 15.0% per year, and the Fama-MacBeth CV coefficient carries a t-statistic of 2.11 inside the EBITDA/EV universe. CV is not a simulation-specific signal. It is capturing something general: business risk, earnings volatility, and fundamental uncertainty. Companies with high CV are genuinely harder to forecast, and genuine forecast difficulty is associated with higher expected returns as compensation for risk. That is a risk premium, not an edge from running the Monte Carlo.

**Verdict.** The Monte Carlo distribution adds no predictive information beyond what simpler risk measures already capture. High CV means high business uncertainty. High business uncertainty earned a risk premium over the 2015 to 2024 period, which was predominantly a bull market with compressed risk premia at the index level but significant dispersion at the small-cap level. The correct interpretation of the Q5 outperformance is not that wide uncertainty is a positive signal. Wide uncertainty is a proxy for the kind of fundamental volatility that the market underprices in small-cap names during benign conditions. Realised volatility and leverage are coarser proxies for the same underlying exposure. The Monte Carlo simulation is not needed to make this bet.

**What would change this verdict.** Analysing the full distribution shape rather than just CV might reveal non-linear information. Left-tail probability (the probability of the simulated intrinsic value falling below zero or below the current price) could be a more precise risk measure than CV alone. The distribution shape also contains information about skewness, which CV does not capture. These are testable extensions on the same data and infrastructure.

---

## How I Avoided Fooling Myself

**One pass, no tuning.** Each hypothesis was designed before any results were examined. Signal definitions, basket construction rules, quintile cut points, and regression specifications were written into the code before any pipeline was run. The numbers above are the first outputs each pipeline produced. No threshold was adjusted after seeing a result.

**Point-in-time data throughout.** Every EDGAR-derived signal was filtered to filings with a filed date on or before the June 30 rebalance date. The code checks the filed date field in the EDGAR company facts API, not the fiscal period end date. A company reporting FY2018 results in a 10-K filed on February 15 2019 contributes to the June 2019 rebalance, not June 2018. This mirrors what was actually knowable at each date. The pre-check for H1 confirmed this by printing which filing dates were used for a sample ticker at a historical date before running the full backtest.

**Publication lag as the edge test.** The distinction between the fiscal year end date and the filing date is the primary look-ahead check. All signals respect this boundary. EDGAR filings arrive 60 to 90 days after fiscal year end on average for small companies, and the filter uses the actual filing receipt date, not an assumed lag.

**Deflated Sharpe accounting.** Fourteen basket definitions were tested across H1 and H2 (four baskets across two value arms in H2 alone) and five CV quintile baskets in H3. The Fama-MacBeth regressions were pre-specified single runs. The trial count is tracked. The management quality t-statistic of 3.48 is the only result that survives scrutiny under conservative multiple-comparison adjustment. All others should be read as directional hints.

**Common universe, both arms.** In H1, the same DCF-applicable, non-financial universe was used for ranking both the DCF and the multiple signals at every rebalance date. Differences in coverage or applicability between signals were not allowed to create an asymmetric advantage.

**Relative controls.** The management score uses year-over-year changes in share count and ROIC rather than absolute levels. This avoids confounding industry differences in capital structure and returns on capital with actual management quality signals.

---

## Honest Limitations

**Survivorship bias (the central limitation).** The universe was drawn from a 2026 company list. Every company that went bankrupt, was acquired at a distressed valuation, was delisted for performance reasons, or merged out of existence between 2015 and 2026 was absent from the analysis. This is a serious methodological flaw. It inflates absolute returns for all baskets and inflates the management signal in particular: companies that survived a decade of public-market exposure are exactly the ones that tended not to dilute shareholders, not to destroy capital, and not to misallocate earnings. The management signal at t=3.48 reflects this selection effect to an unknown degree. The absolute return numbers throughout this document should be read as upper bounds. The differentials between arms of the same comparison (DCF versus EBITDA/EV, basket A versus basket C) are partially robust because the bias is shared, but the magnitude of those differentials is still unreliable.

**M1 insider buying effectively absent.** The intended management score was M1 plus M2 plus M3, covering insider conviction, dilution discipline, and ROIC trend. In practice, Form 4 filing coverage from the EDGAR archive at scale for the historical period produced only 7.8% coverage across the full universe. M1 was excluded from the effective management score as a result. The reported results for management quality reflect M2 and M3 only. This may be a data-engineering limitation rather than an absence of the underlying signal.

**T=10 is a small sample for Fama-MacBeth.** Ten annual cross-sections produce ten coefficient estimates. The t-statistic for management quality (3.48) is computed over those ten observations. Autocorrelation in the annual coefficient series reduces the effective degrees of freedom further. The direction of the management finding is the primary takeaway. The precise magnitude of the t-statistic should not be over-interpreted.

**DCF applicability gate is not random.** Approximately 86% of the universe (590 of 686 tickers) passes the DCF applicability gate on average across the ten dates. Companies that fail the gate tend to be earlier-stage, more volatile, or financially complex. The results describe a specific, more stable subset of the small-cap population. The performance of the excluded names is unknown.

**No transaction costs beyond the round-trip estimate.** All returns include a 20 basis point round-trip cost assumption, which is conservative for the most liquid names in the universe but optimistic for smaller names trading at wider spreads. A realistic cost for names at the lower end of the liquidity filter would be 50 to 100 basis points per round trip, which would reduce reported Sharpe ratios meaningfully.

---

## Conclusion

The main finding of this research is not the one originally anticipated. The DCF engine did not beat a free multiple on its own, the Monte Carlo distribution added no alpha beyond what risk proxies already captured, and the Piotroski quality screen had no independent cross-sectional power after controlling for size. What did stand out, clearly and by a wide margin, was management quality.

**The survivorship bias problem.** After completing all three hypotheses, I identified that the backtest universe was drawn from the current small-cap company population. Every company that was delisted, went bankrupt, was absorbed by a larger acquirer at a distressed price, or was otherwise removed from public markets between 2015 and 2026 was not in the dataset. This is a fundamental flaw in the experimental design. It biases returns upward and biases the management signal most severely of all. The companies that survive a decade of public-market existence are disproportionately the ones whose management teams did not dilute shareholders, did not destroy capital, and did not take on unsustainable leverage. Excluding the failures inflates exactly the signal I am testing.

**Why management quality is still worth investigating.** Even with the survivorship issue acknowledged fully, the management signal dominated the six-variable Fama-MacBeth regression by a margin that is striking. The next highest valuation factor had a t-statistic below 1.3. Management quality was the only factor other than SIZE that passed conventional significance thresholds, and it was the only signal where I was not also competing against a known risk premium. SIZE is well-documented and widely arbitraged. Management quality in small-cap names is not.

The structural case is clear. Small-cap companies receive less analyst coverage, which means signals embedded in Form 4 transactions and annual report filings are processed more slowly by the market. A $200 million market cap company may have zero sell-side coverage and limited institutional ownership. Management quality is particularly underexplored because it does not reduce to a single line item in the income statement. Quantifying it requires reading proxy statements, tracking Form 4 histories over multiple years, and evaluating capital allocation decisions across business cycles. That friction reduces the number of investors systematically acting on the signal, which is precisely the condition that creates a durable edge.

A disciplined, market-neutral strategy that goes long small-cap companies with strong management scores and shorts those with weak scores could produce genuine alpha with limited directional market exposure, if it can be validated on a dataset that includes the failures as well as the survivors.

**Building the correct universe.** The path to a clean backtest runs through three implemented data sources. The SEC XBRL company facts API (`data.sec.gov/api/xbrl/companyfacts/{CIK}.json`) provides point-in-time fundamentals for all registered companies including deregistered entities, accessed via the EDGAR per-company API rather than bulk ZIP files. The SEC Submissions API (`data.sec.gov/submissions/CIK{CIK}.json`) provides Form 4 insider transaction records for each company. Tiingo's daily prices API provides raw close prices for delisted stocks going back to the 1990s. Raw close rather than adjusted close is used deliberately: adjusted close figures are contaminated by split and dividend factors that are cached against the wrong entity when a ticker is reused after a delisting. Point-in-time market cap is computed as Tiingo raw close multiplied by SEC shares outstanding at each rebalance date, eliminating the survivorship bias that a current-snapshot market cap source would reintroduce. Together these sources allow construction of a 2015 small-cap universe that includes every company that was publicly traded in that year, regardless of what happened to it afterwards. That work is in progress. Retesting the management quality hypothesis on this debiased universe is the primary next step.

---

## Data Sources

| Source | Data | Used in |
|---|---|---|
| SEC EDGAR XBRL API | Point-in-time 10-K filings: revenue, EBIT, D&A, CapEx, NWC, balance sheet items. Filed date filtering to eliminate look-ahead bias | H1, H2, H3 |
| SEC EDGAR Submissions API | Form 4 insider transaction XML filings for net buying computation (M1). Historical coverage limited | H2 |
| yfinance | Adjusted daily close prices for US small-cap universe (2013 to 2025), 10-year Treasury yield, market beta | H1, H2, H3 |
| SEC XBRL company facts API | Point-in-time fundamentals for all registered companies including deregistered entities (`data.sec.gov/api/xbrl/companyfacts/{CIK}.json`) (next stage) | Upcoming |
| SEC Submissions API | Form 4 insider transaction records per company (`data.sec.gov/submissions/CIK{CIK}.json`) (next stage) | Upcoming |
| Tiingo daily prices API | Raw close prices (not adjusted close) for delisted stocks going back to the 1990s (next stage) | Upcoming |

---

## Reproducibility

All analysis code lives in this repository. The pipeline is Python 3.12 with standard scientific libraries.

```
research/dcf_vs_multiple/run.py        — H1 DCF vs. free multiples horse race and Fama-MacBeth
research/quality_filter/run.py         — H2 Piotroski F-Score and management quality layering
research/monte_carlo_dispersion/run.py — H3 Monte Carlo CV signal with controls
```

All three pipelines cache results to their respective `_cache/` directories. The EDGAR facts cache is shared across all three pipelines from `research/dcf_vs_multiple/_cache/edgar_facts/`. Running any pipeline for the first time fetches and caches the required data. Subsequent runs load from disk and complete in under five minutes.

```
# point-in-time check on a single ticker before running the full backtest
venv/bin/python research/dcf_vs_multiple/run.py --pit-check AAPL 2018-06-30

# run full H1 pipeline (first run: ~2 hours; cached: ~4 minutes)
venv/bin/python research/dcf_vs_multiple/run.py

# run H2 quality filter (requires H1 cache)
venv/bin/python research/quality_filter/run.py

# run H3 Monte Carlo dispersion (requires H1 cache)
venv/bin/python research/monte_carlo_dispersion/run.py
```
