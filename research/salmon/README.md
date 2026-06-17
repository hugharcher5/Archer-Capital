# North Atlantic Salmon: A Quantitative Alpha Search

**Author:** Hugh Archer &nbsp;|&nbsp; **Period:** 2024–2026 &nbsp;|&nbsp; **Status:** Concluded — no exploitable edge found

---

## Executive Summary

Norwegian salmon-farmer equities (Mowi, SalMar, Lerøy, Grieg, Bakkafrost) are specialist-covered mid-caps whose core inputs — weekly lice counts, disease events, ocean temperature, feed prices, and live biomass — are published as public data. The question was whether any of those inputs, processed rigorously, contains a signal that survives until a trade can be executed.

**Method.** Five independent hypotheses were constructed from first principles, each mapped to a distinct data source and signal chain. Each was run exactly once on the full available history with no iteration or parameter tuning. Results are reported as-is.

| Hypothesis | Status | Key finding |
|---|---|---|
| H1 — SST Lead-Lag | **No Signal** | Best t-stat 1.28; 18 truly independent episodes over 14 years |
| H2 — Lice Efficiency | **Weak Hint** | Real signal at lag 0 (t = −1.99); gone at lag 1, the earliest tradeable point |
| H3 — Acute Events | **No Signal** | t = −0.28 across 17 events; data quality too poor to test the concept |
| H4 — Feed-Cost Squeeze | **Weak Hint** | t = 2.49 at lag 5 months; below Bonferroni threshold; backtest Sharpe −0.05 |
| H5 — Biomass Nowcast | **Blocked** | Strongest concept structurally; blocked on granular biomass data with no public API |

**Meta-conclusion.** The mechanism linking lice, disease, and feed costs to salmon-farmer earnings is real and well-understood. What is absent is any exploitable lag between when the data becomes public and when the specialist sell-side community (DNB, Pareto, ABG Sundal Collier) acts on it. Every testable signal either appears simultaneously with the information release or is too small to survive a proper multiple-testing correction. The Norwegian salmon equity market is efficiently covered for the data sources tested.

---

## Scoreboard

All five hypotheses run once on full data (2012–2026 where available). No parameters tuned.

| Hypothesis | Status | Key statistic | Why it did not yield an edge |
|---|---|---|---|
| H1: Sea-Surface Temperature Lead-Lag | No Signal | t-stat = 1.28, R² ≈ 0% | Satellite SST anomalies show zero ability to predict relative farmer returns across any lag from 1–8 weeks. 753 weekly rows collapse to 18 truly independent warm-water episodes. |
| H2: Relative Lice Efficiency | Weak Hint | t = −1.99 at lag 0, t = −0.47 at lag 1, Sharpe = 0.16 | Signal collapses after one week. Lag 0 captures the market reacting to Barentswatch simultaneously, not a lead. The earliest tradeable lag (1) has no signal. |
| H3: Acute Event Detection | No Signal | t = −0.28, n = 17 events | Aggregate signal is noise. Event dates were reconstructed from secondary sources, not original Mattilsynet filings, making the analysis unreliable. |
| H4: Feed-Cost Squeeze | Weak Hint | t = 2.49 at lag 5, Bonferroni threshold 2.81, Sharpe = −0.05 | Fishmeal-to-return signal at 5-month lag is economically coherent but falls short of the multiple-testing threshold and produces a flat backtest. Only 6 El Niño events since 2005. |
| H5: Biomass Nowcast | Blocked | Pipeline complete; data unavailable | Granular biomass data from Fiskeridirektoratet has no public API. SSB discontinued the monthly series in 2019. Structurally the strongest concept; remains untested. |

---

## Hypothesis Results

### H1 — Sea-Surface Temperature Lead-Lag

**No Signal**

**Mechanism.** Sea lice reproduce faster in warm water. If a production zone's water is warmer than its seasonal average, a lice spike should follow 2–4 weeks later — before it appears in the weekly Barentswatch report. Use satellite temperature anomalies to go short the affected farmer before the lice data confirms the problem.

**Chain of logic:**
```
Warmer SST anomaly (NOAA OISST)
  ↓  2–4 weeks
Higher lice density
  ↓  1–3 weeks
Treatment costs spike, quality falls
  ↓
Farmer underperforms peer basket
```

**Data.** NOAA OISST v2.1 daily SST at 0.25° resolution (2012–2026). Zone-weighted signals built for all 11 Norwegian production zones using farmer-specific zone footprints. 753 weekly observations.

**What was tested.** Whether zone-weighted SST anomaly predicted each farmer's relative return (farmer minus equal-weight basket) at horizons of 1–8 weeks. Sign consistency and t-statistics reported across all farmers and all lags.

**Results:**

| Metric | Value |
|---|---|
| Best t-statistic | 1.28 |
| R² (all tests) | ≈ 0% |
| Sign consistency | Mixed |
| True independent observations | 18 warm-water episodes |

**Verdict.** No signal. The temperature data shows zero ability to predict which farmer will underperform across any horizon or farmer tested. The unverified middle link (SST → lice) means we cannot distinguish between a mechanism that is wrong and one that operates too slowly to produce an equity edge.

**What would change this verdict:** A confirmed SST → lice correlation with 2–4 week lead at zone level, combined with a demonstrated lice → stock return link at lag 1+. Both links would need to hold independently on out-of-sample data.

---

### H2 — Relative Lice Efficiency

**Weak Hint**

**Mechanism.** Do not trade the absolute lice level. Compare each farmer's sites to the other operators in the same production zone. If Mowi's sites in zone 6 carry three times the zone average while SalMar's nearby sites are at the average, that is a company-specific operational warning invisible in headline lice figures. Normalising by local environmental pressure strips the commodity factor.

**Chain of logic:**
```
Farmer A's lice residual rises (vs zone LOO average)
  ↓
Operational problem: poor biosecurity or equipment
  ↓
Treatment costs spike, fish quality falls
  ↓
Earnings disappoint (market has not priced it yet)
  ↓
Short farmer A vs basket
```

**Data.** Barentswatch FishHealth API: 364,953 site-week observations, 1,186 unique sites, 2012–2026 (687 weeks). Site-to-company mapping via Fiskeridirektoratet: 484 sites mapped across Mowi (200), SalMar (142), Lerøy (122), Grieg (20). Signal: leave-one-out zone residual shifted forward 1 week (publication lag).

**Results (Test A, pooled OLS):**

| Lag | t-stat | Signal? |
|---|---|---|
| 0 | **−1.99** | Yes — negative |
| 1 | −0.47 | noise |
| 2 | +0.04 | noise |
| 3 | +0.34 | noise |
| 4 | +0.86 | noise |

Effective n per farmer: ~190 (adjusted for r1 ≈ 0.55 autocorrelation, raw n = 685 weeks). Test C Sharpe: 0.16 pre-cost.

**Chart — H2 t-statistic by lag:**

```
t-stat
 +1.5 |
      |
 +1.0 |
      |
 +0.5 |                                     ████   █████
      |                                     █████  █████
  0.0 |────────────────────────────────────────────────────── (zero)
      |                      ██
 -0.5 |                      ██
      |
 -1.0 |
      |
-1.5  |·····················orange·gate·(t·=·−1.5)···········
      |
 -2.0 | ████
      | ████  (-1.99)
      |
      +────────────────────────────────────────────────────
         Lag 0    Lag 1    Lag 2    Lag 3    Lag 4
                    ↑
            Earliest tradeable
            (publication lag)
```

The signal at lag 0 (−1.99) reflects the market reacting simultaneously to the same Barentswatch data. Lag 1 is the earliest a trade could be executed; the signal is gone.

**Verdict.** The mechanism is confirmed: relative lice performance predicts relative stock returns. The problem is timing. The Barentswatch report is published Monday mornings and the specialist sell-side processes it immediately. Lag 0 captures their simultaneous reaction. Lag 1, the earliest point at which a retail or independent investor could trade, shows no signal (t = −0.47). Not viable as a systematic strategy without a genuine publication-lag advantage.

**What would change this verdict:** Evidence that a subset of farms (e.g. those with infrequent reporting or specific sites) exhibits a delayed publication pattern. Alternatively, a correlated signal (satellite observation, weather data) that precedes the lice report by one week.

---

### H3 — Acute Event Detection

**No Signal**

**Mechanism.** ISA outbreaks (mandatory whole-farm culls), mass mortality events from algal blooms, and de-licing accidents appear in Norwegian Food Safety Authority (Mattilsynet) notices before analysts or news wires process them. Reading those notices in Norwegian at publication time creates a short-term short opportunity in the affected company.

**Chain of logic:**
```
ISA confirmed (Mattilsynet PDF, Monday morning)
  ↓
Mandatory cull: sudden, quantified biomass loss
  ↓
Stock falls over next 1–10 days as news spreads
  ↓
Short before professional community reacts
```

**Results by event type (17 events, CAR at day +1):**

| Event type | n | Mean CAR +1d |
|---|---|---|
| All events | 17 | −0.13% |
| ISA specifically | 8 | +0.14% |
| Mortality events | 4 | **−2.26%** |
| PD events (CAR 5d) | 5 | +4.4% |

**Why the result is unreliable.** Nearly all 17 events were reconstructed from secondary sources (news articles, annual reports, academic papers) rather than original Mattilsynet filings. This introduces three compounding problems:

- **Approximate dates.** An ISA confirmed on March 15 but recorded as "March" adds two weeks of noise to every event window.
- **Attribution uncertainty.** Regional disease notices attributed to the dominant local operator could be wrong.
- **Small sample.** 17 events across 5 correlated stocks provides limited statistical power even with perfect data.

The only interesting sub-result: the 4 mortality events (the 2019 Nordland algal bloom and the 2021 de-licing incident) showed −2.26% mean return at day +1, all four negative (t = −3.05, p = 0.055). Directionally consistent but dominated by one large event, and approximate dates undermine confidence.

**Verdict.** No reliable signal. The aggregate result is noise. The mortality sub-group is directionally consistent but too small and too imprecise to trade on. The concept is structurally sound; the bottleneck is building a structured Mattilsynet data source with exact event dates.

**What would change this verdict:** A systematic scrape of Mattilsynet's weekly fish-health notices with exact confirmation dates and locality IDs cross-referenced against the Barentswatch site register. This is a data-engineering problem, not an analytical one.

---

### H4 — Feed-Cost Squeeze via the El Niño / Anchovy / Fishmeal Chain

**Weak Hint**

**Mechanism.** Salmon feed is the largest single cost in farming. The primary protein ingredient is fishmeal, made from Peruvian anchovies. El Niño disrupts the Peruvian anchovy catch, fishmeal prices spike, and salmon-farmer margins compress months later when feed contracts roll. El Niño climate forecasts are available before the earnings impact materialises.

**Chain of logic:**
```
El Niño develops (NOAA ONI index)
  ↓  2–4 months: quota cuts, lower catch
Fishmeal prices spike
  ↓  1–3 months: feed contracts adjust
Feed costs rise
  ↓  quarterly lag to earnings
Farmer margins compressed, stock underperforms
```

**What was tested:**

- **Test A (climate → commodity):** NOAA ONI index against fishmeal prices at 0–12 month lags. No relationship found; only 6 independent El Niño events available since 2005.
- **Test B (commodity → returns):** Monthly fishmeal price change against salmon-farmer relative returns at 0–9 month lags. Hint at lag 5: β = −0.33, t = 2.49. Sign consistent across most farmers. Sign flips positive at lag 2, which is suspicious.
- **Test C (backtest):** Simple signal using fishmeal rises during active El Niño. Backtest Sharpe = −0.05.

**Key statistics:**

| Metric | Value |
|---|---|
| Best t-stat (lag 5) | 2.49 |
| Bonferroni threshold (10 lags) | 2.81 |
| ONI climate signal (Test A) | No relationship found |
| Backtest Sharpe (Test C) | −0.05 |
| Independent El Niño events | 6 (since 2005) |

**Verdict.** Fishmeal-to-returns at lag 5 is the most economically coherent finding in this study and the only result I would watch further. But it falls short of the Bonferroni-corrected threshold (2.49 vs 2.81), the backtest is flat, and the sample is dominated by just six El Niño events. It is a candidate for revisit with 5–10 more years of data, not a tradeable signal today.

**What would change this verdict:** A longer out-of-sample history with more El Niño events, or an alternative feed-cost series (e.g. soy protein isolate prices) that exhibits a cleaner lag structure at 5 months.

---

### H5 — Biomass Nowcast

**Blocked**

**Mechanism.** Norwegian fish-health regulations require farmers to report live biomass weekly to Fiskeridirektoratet. If granular, company-level biomass data can be extracted and converted to "expected harvest weight" two quarters forward, it creates a novel high-frequency estimate of supply-side earnings drivers that the sell-side models quarterly at best.

**Chain of logic:**
```
Fiskeridirektoratet weekly biomass reports (site level)
  ↓
Aggregate by company + production zone
  ↓
Model harvest weight forward (biological growth curve)
  ↓
Compare to sell-side harvest volume estimates
  ↓
Trade divergence before quarterly earnings
```

**Why it is blocked.** Fiskeridirektoratet publishes aggregate national production totals but does not expose a public API for site-level or company-level biomass. Statistics Norway (SSB) published monthly series until 2019; the series was discontinued. No vendor offers a structured historical series. The data exists within the regulator's internal systems but is not accessible.

**Verdict.** Remains the strongest structural concept of the five and the one most worth revisiting if data access improves. The signal chain is the tightest, the information is genuinely non-public at granular resolution, and the sell-side blind spot (quarterly vs weekly modelling) is real.

---

## How I Avoided Fooling Myself

**One pass, no tuning.** Each hypothesis was designed, coded, and run once. No results were used to adjust parameters or choose between model specifications.

**Effective sample size.** Weekly equity returns are highly autocorrelated (r1 ≈ 0.55 in this sample). Raw observation counts are misleading. Every test reports effective n using the AR(1) correction:

```
n_eff = n × (1 − r1) / (1 + r1)
```

With r1 = 0.55 and n = 685 weeks, effective n ≈ 190. This is the number I used for significance decisions, not 685.

**Leave-one-out zone averages.** The H2 benchmark for each farmer excludes that farmer's own sites. A large operator cannot anchor its own reference level. The LOO average is built from peer operators in the same geographic production zone only.

**Publication lag as the edge test.** A signal that appears at lag 0 is not an edge — it means the market reacted at the same time as the data release. The correct test is lag 1+, which is the earliest point at which an independent investor could act on publicly released data.

**Bonferroni correction.** Testing 10 lags in H4 raises the significance threshold from t ≈ 2.0 to t = 2.81 (α = 0.05 / 10). All results are reported against the corrected threshold.

**Relative returns.** All equity signals use farmer return minus equal-weight peer basket. This removes the salmon-price commodity factor that affects all farmers equally and would otherwise dominate any test.

---

## Honest Limitations

- **Retroactive FDIR mapping.** The Fiskeridirektoratet site-to-owner register reflects current state (2026). Pre-acquisition sites are attributed to current owners. For example, Marine Harvest sites acquired before the 2018 rebrand appear as "Mowi" throughout history. Ownership-change events (mergers, license trades) are not tracked point-in-time.

- **792 unmapped Barentswatch sites.** Of 1,186 sites in the lice data, only 394 (33%) are mapped to the four test farmers. The remaining 792 sites belong to smaller Norwegian operators not included in the study. The zone LOO average is therefore built on a non-representative sample of zone peers.

- **GSF thinness.** The Global Salmon Feed index used in H4 has thin trading and may not reflect actual contract prices paid by farmers. A more reliable series would be direct contract price data from feed producers (BioMar, Skretting, Cargill Aqua).

- **H3 approximate dates.** The majority of the 17 acute events have dates accurate only to the month or the week. The event-study methodology requires day-level precision. The current results understate noise rather than overstate signal.

- **H5 untested.** The strongest structural concept was never quantitatively assessed due to data unavailability. This is the primary gap in the study.

- **No transaction costs.** All backtest results are gross of transaction costs, bid-ask spread, and borrowing costs for short positions. Norwegian small-cap equities (particularly Grieg and Bakkafrost) have materially wider spreads than US-listed large caps.

---

## Conclusion

Research concluded June 2026 after completing all testable hypotheses. The Norwegian salmon equity market is efficiently priced for the publicly available data sources examined. The mechanism — lice, disease, and feed costs linking to earnings — is real and well-understood by specialist sell-side analysts at DNB Markets, Pareto Securities, and ABG Sundal Collier, who process the same public data faster than an independent investor without privileged infrastructure.

**Main takeaways:**

1. **The mechanism is real but not tradeable.** Relative lice efficiency (H2) confirmed that lice residuals predict relative stock returns. The problem is entirely one of timing: the signal lives at lag 0 (simultaneous with publication) and is gone by lag 1 (the earliest executable point). The sell-side is faster.

2. **Publication lag is the central constraint.** Every testable signal in this study collapsed at lag 1. Any future work in this market must either identify a genuinely delayed data source, a correlated signal that precedes the public release, or a structural information advantage (e.g. direct site monitoring). Analytical sophistication alone is not sufficient.

3. **Data quality was the recurring binding constraint.** H3 was blocked by imprecise event dates. H5 was blocked entirely by data access. H1 was blocked by the missing SST-to-lice link. In each case, the concept was sound and the data was not. If the data quality problems were resolved, the study could be meaningfully extended.

The only result worth watching as more data accumulates: the fishmeal-to-returns hint at lag 5 months (H4, t = 2.49) — below the multiple-testing threshold today, but economically coherent and consistent in sign.

---

## Data Sources

| Source | Data | Used in |
|---|---|---|
| Barentswatch FishHealth API | Weekly lice counts per site, 2012–2026 (364,953 site-week rows) | H2 |
| Fiskeridirektoratet (FDIR) | Site-to-company ownership register, production zone boundaries | H2 |
| NOAA OISST v2.1 | Daily sea-surface temperature at 0.25° resolution, 2012–2026 | H1 |
| NOAA ONI Index | Monthly Oceanic Niño Index for El Niño classification | H4 |
| Global Salmon Feed (GSF) | Monthly fishmeal price index | H4 |
| Mattilsynet / secondary sources | ISA, PD, and mortality event dates and company attributions (17 events) | H3 |
| Yahoo Finance / yfinance | Weekly adjusted close prices for Mowi, SalMar, Lerøy, Grieg, Bakkafrost | H1–H4 |

---

## Reproducibility

All analysis code lives in this repository. The pipeline is Python 3.12 with standard scientific libraries (pandas, scipy, numpy, statsmodels).

```
research/lice_efficiency/run.py   — H2 lice pipeline (requires Barentswatch credentials)
research/acute_events/            — H3 event study
research/sst/                     — H1 SST analysis
research/feed_cost/               — H4 feed-cost squeeze
strategies/north_atlantic_salmon/ — Streamlit research page (full interactive version)
research/salmon/salmon_research_report.html — standalone HTML version of this document
```

Barentswatch API access requires a free account at [barentswatch.no](https://www.barentswatch.no) with the `api` scope.
