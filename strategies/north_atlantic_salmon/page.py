"""
North Atlantic Salmon Research Hub
Renders the salmon market overview and all five hypothesis write-ups
in the Strategies tab.
"""

from __future__ import annotations
import streamlit as st


# ── status badge helpers ───────────────────────────────────────────────────────

_BADGE = {
    "no_signal":  ("🔴", "No Signal",    "#c0392b"),
    "blocked":    ("🟡", "Blocked",       "#e67e22"),
    "ready":      ("🟡", "Ready to Run",  "#e67e22"),
    "weak":       ("🟠", "Weak Hint",     "#d35400"),
    "research":   ("🔵", "In Research",   "#2980b9"),
}


def _status_pill(key: str) -> None:
    icon, label, colour = _BADGE[key]
    st.markdown(
        f'<span style="background:{colour};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:0.78rem;font-weight:600;">'
        f'{icon} {label}</span>',
        unsafe_allow_html=True,
    )


def _h2(text: str) -> None:
    st.markdown(f"### {text}")


def _h3(text: str) -> None:
    st.markdown(f"#### {text}")


def _divider() -> None:
    st.markdown("---")


# ── main render function ───────────────────────────────────────────────────────

def render_salmon_page() -> None:

    st.markdown("## North Atlantic Salmon")
    st.caption(
        "Five quantitative hypothesis tests on Norwegian salmon-farmer equities. "
        "Each hypothesis was run once on the full available data with no parameter tuning."
    )

    _divider()

    # ── Market Overview ────────────────────────────────────────────────────────
    with st.expander("The Salmon Market: Background", expanded=True):

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("""
**What salmon farmers actually do**

Atlantic salmon farming starts with eggs, which are hatched into juvenile fish (smolts) in freshwater tanks on land. After 12 to 18 months the smolts are transferred to large net pens in a Norwegian fjord and grown out for another 18 to 24 months until they reach roughly 4 to 5 kg. Then they are harvested, processed, and shipped, primarily to Europe and the US, within days.

The regulatory binding constraint is **Maximum Allowed Biomass (MAB)**: the government caps how much live fish weight each company is permitted to have in the water at any time. Growth, disease, and escape incidents all eat into that cap. Running close to the cap means maximum revenue. Losing fish to disease or escape is a direct hit to how much you can sell.

**How profits work**

A salmon farmer's earnings swing on three levers:
1. **The salmon price**, a volatile spot commodity traded daily, similar to oil or copper
2. **Cost per kilogram**, where feed is the largest single cost, followed by labour and medication
3. **Volume harvested**, which is capped by MAB and reduced whenever disease hits

A bad disease year hits all three at once: fewer fish harvested, higher treatment costs, and sometimes lower-quality fish that sell at a discount.
""")

        with col_b:
            st.markdown("""
**Sea lice and ISA: the two main disease risks**

*Sea lice* are tiny parasites that live on the outside of fish and feed on skin and mucus. They do not usually kill the salmon but they:
- Downgrade the fish quality, reducing sale price
- Cost a lot to treat (chemical baths, thermal de-licing machines, cleaner fish like wrasse that physically eat the lice)
- Spread rapidly through crowded sea pens

*ISA (Infectious Salmon Anaemia)* is a virus. When a site tests positive, regulators mandate culling every fish on the farm immediately. This is a sudden, binary event. One day the farm is running normally, the next it is emptied. It creates sharp, fast-moving stock impacts.

**The traffic light system**

Norway divides its coastline into 13 production zones. Every two years, regulators assess the average lice pressure in each zone and assign a colour:
- 🟢 Green: zone can *expand* licensed biomass
- 🟡 Yellow: biomass held flat
- 🔴 Red: biomass must be *cut* by 6%

The biennial cycle makes the red-light outcome slow and telegraphed. What is faster is the **weekly lice report**: every Monday each farm site publishes the number of adult female lice per fish. These weekly numbers are the live trading signal.

**Why this market is hard to beat**

Norway's major salmon farmers (Mowi, SalMar, Lerøy, Bakkafrost, and Grieg) are well-covered mid and large-cap stocks. Dedicated analysts at DNB, Pareto, ABG Sundal Collier, and Kepler Cheuvreux build detailed models using the exact same public weekly lice and biomass data. Specialist seafood funds have every site mapped to every company. The only realistic edge is either connecting a dataset they do not routinely process, such as satellite ocean temperature or climate indices, to the lice data, or finding **post-event drift** where the market is slow to fully price in a bad trend visible in the weekly data.
""")

    _divider()

    # ── Summary scoreboard ─────────────────────────────────────────────────────
    _h2("Research Scoreboard")
    st.caption("Five hypothesis tests, each run once on full available data with no tuning.")

    cols = st.columns(5)
    summaries = [
        ("H1\nSST Lead-Lag",        "no_signal"),
        ("H2\nLice Efficiency",     "weak"),
        ("H3\nAcute Events",        "no_signal"),
        ("H4\nFeed-Cost Squeeze",   "weak"),
        ("H5\nBiomass Nowcast",     "blocked"),
    ]
    for col, (label, status) in zip(cols, summaries):
        with col:
            icon, lbl, colour = _BADGE[status]
            st.markdown(
                f'<div style="border:1px solid {colour};border-radius:8px;'
                f'padding:12px 8px;text-align:center;">'
                f'<div style="font-size:0.75rem;font-weight:600;white-space:pre-line;">{label}</div>'
                f'<div style="margin-top:6px;background:{colour};color:white;border-radius:8px;'
                f'padding:2px 0;font-size:0.7rem;">{icon} {lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 1 — SST LEAD-LAG
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H1: Sea-Surface Temperature as an Early Warning on Lice")
    _status_pill("no_signal")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

Lice reproduce faster in warm water. NOAA publishes daily satellite-measured ocean temperatures for every 0.25 degree square of ocean. The theory: if a Norwegian production zone's water is warmer than its seasonal average, a lice spike should follow 2 to 4 weeks later, before it shows up in the weekly lice report that everyone else reads.

The trade: use satellite temperature anomalies to predict which farmers will be hit with elevated lice, and short those relative to their clean peers, one step earlier than the lice data.

**The chain of logic**
```
Warmer sea temperature
        ↓ (2 to 4 weeks)
More sea lice than usual
        ↓ (1 to 3 weeks)
Higher treatment costs, quality downgrades
        ↓
That farmer underperforms its peers
```

**What we tested**

- Fetched daily NOAA ocean temperature for all 11 Norwegian production zones (PO1 to PO11), from 2012 to present, via satellite data (OISST v2.1)
- Computed each zone's temperature anomaly, measuring how much warmer or colder than the typical temperature for that time of year
- Built a zone-weighted temperature signal for each farmer based on where they farm
- Tested whether that signal predicted each farmer's relative return (performance versus the peer basket) over the next 1 to 8 weeks

**Results**

| What we measured | Result |
|---|---|
| Best statistical signal found | t-stat = 1.28 (threshold to take seriously is above 2.0) |
| R-squared across all tests | approximately 0%, the signal explains none of the return variation |
| Sign consistency | Mixed, roughly half positive and half negative across farmers and horizons |
| True independent observations | 18 warm-water episodes over 14 years, not 753 weekly rows |

The temperature data showed zero ability to predict which farmer would underperform. The results were consistent with random noise across every farmer and every time horizon tested.

**The missing middle link**

The pipeline tested SST directly against returns, skipping the SST to lice step. Even if warm water does cause more lice, the lice to returns connection may be too slow, too noisy, or already priced by the time it appears in the weekly report. The unverified middle link means we cannot distinguish between a mechanism that is wrong and one that operates through a channel too slow for equity returns to reflect.
""")

    with col2:
        st.info("""
**What the numbers mean**

A **t-statistic** tells you how confident to be that a result is real versus random. A rough guide:
- Below 1.5: probably noise
- 1.5 to 2.0: possibly something
- Above 2.0: worth investigating
- Above 3.0: strong

The best result across all tests was **1.28**. That is noise.
""")

        st.info("""
**Why 18 episodes, not 753 weeks?**

The 753 weekly rows of data are not 753 independent observations. This week's temperature is correlated with last week's, and they are almost the same reading. The number of truly independent events is how many distinct warm-water periods were actually observed. That was 18 over 14 years. Statistics based on 18 real events are much weaker than they appear when the calendar says 753 observations.
""")

        st.markdown("""
**Verdict:** No signal found. The satellite temperature data does not predict relative farmer returns across 14 years of data. The mechanism may exist biologically but does not translate into an equity edge on weekly or monthly return horizons.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 2 — LICE EFFICIENCY
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H2: Relative Lice Efficiency: Which Farmer is Better at Managing Disease?")
    _status_pill("weak")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

Do not trade the absolute lice level. That is what everyone is looking at. Instead, compare each farmer to their neighbours in the same zone. If zone 6 (Trøndelag) is having a bad lice season and every farm there is struggling, that is priced in. But if Mowi's sites in zone 6 have three times more lice than the zone average while SalMar's nearby sites are at the zone average, that is a company-specific warning signal that is harder to see from headline lice figures alone.

**The chain of logic**
```
Farmer A's lice count is rising faster than the zone average
        ↓
They have an operational problem (poor biosecurity, older equipment, bad luck)
        ↓
Treatment costs will spike and fish quality will fall
        ↓
Their earnings will disappoint, but the market has not priced it yet
        ↓
Short that farmer relative to the clean peers
```

**What we built and ran**

364,953 site-week observations pulled from Barentswatch (2012 to 2026), covering 1,186 unique farm sites. 484 of those sites were mapped to the four test farmers via the Fiskeridirektoratet public registry:
- Mowi: 200 sites across PO4 to PO8 (pan-Norway)
- SalMar: 142 sites across PO5 to PO12
- Lerøy (LSG): 122 sites across PO3 to PO13
- Grieg (GSF): 20 sites in PO2 (Rogaland)

**Signal construction: leave-one-out zone residual**

The zone benchmark is computed using all other farmers' sites in the same zone that week, then subtracting the current farmer's average. This leave-one-out approach prevents a large farmer like Mowi from anchoring their own benchmark. A farm with 25% of a zone's sites would otherwise deflate their own residual by construction.

**Results**

| Test | What it measured | Result |
|---|---|---|
| **Test A** | Pooled regression across 4 farmers, lags 0 to 4 | beta = -0.012, t = -1.99 at lag 0 only |
| **Test B** | Per-farmer t-stats at each lag | 3 of 4 negative at lag 0, collapses at lag 1 and beyond |
| **Test C** | Short above-zone farmer versus basket | Pooled Sharpe = 0.16 before costs |

**The key finding: the market is already there**

The lag 0 result (t = -1.99) says: in the same week a farmer posts elevated lice, their stock already underperforms. One week later, after the standard publication lag, the effect is gone (t = -0.47 at lag 1, then positive noise at lags 2 to 4).

This is exactly what you would expect if the weekly Barentswatch report is being read in real time by specialist salmon analysts. The signal is real but it is already priced by the time the data is publicly available.

**Effective sample size**

Each farmer's lice residual has a lag-1 autocorrelation of approximately 0.55, meaning this week's reading is half-explained by last week's. Adjusting for this, the approximately 685 weekly observations per farmer shrink to roughly 180 to 200 independent episodes, closer to how many distinct lice events actually occurred rather than how many calendar weeks of data exist.
""")

    with col2:
        st.warning("""
**What "t = -1.99 at lag 0 only" means**

A t-statistic of -1.99 would normally be borderline interesting. The crucial qualifier is lag 0 only.

The Barentswatch data is published weekly on Monday mornings. The earliest you can trade on it is Monday's open, which is what lag 1 tests. At lag 1 the t-stat drops to -0.47. The lag-0 correlation almost certainly reflects the market reacting to the same data at the same time, not evidence of a lead.

The mechanism exists: lice does predict returns. But the market prices it simultaneously, not with a delay that can be exploited.
""")

        st.info("""
**Why the leave-one-out method matters**

An earlier version of the pipeline computed zone averages including each farmer's own sites. A farmer with 25 sites in a zone of 30 total would effectively be compared against themselves, producing a residual that was structurally close to zero.

The leave-one-out fix removes each farmer's contribution before computing their benchmark. Approximately 15% of farmer-site-weeks had undefined leave-one-out averages because the farmer was the sole zone occupant that week. Those observations are excluded from the residual.
""")

        st.markdown("""
**Known data limitations**

- **Retroactive attribution:** The FDIR site-to-owner mapping reflects current ownership. Sites Mowi acquired from Marine Harvest in 2018 are attributed to Mowi throughout the full history, which is a known inaccuracy prior to 2018.
- **Zone coverage:** Only 394 of 1,186 Barentswatch sites have zone assignments (the 4 test farmers). The other 792 sites, smaller Norwegian operators, have no zone data in this pipeline, so they contribute nothing to the zone benchmark.
- **GSF thinness:** Grieg has only 20 sites, all in PO2. Their signal is the noisiest and most sensitive to a single bad site-week.

**Verdict:** The mechanism is real but the market already prices it. Not viable as a systematic strategy.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 3 — ACUTE EVENT DETECTION
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H3: Acute Event Detection: Catching Disasters Before the News Does")
    _status_pill("no_signal")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

ISA outbreaks (mandatory whole-farm culls), mass fish deaths from algal blooms or jellyfish swarms, and de-licing accidents where the treatment itself kills the fish all appear in Norwegian government regulatory databases before they are picked up by analysts or news wires. The Norwegian Food Safety Authority (Mattilsynet) publishes weekly disease-status notices. If you can read those PDFs in Norwegian the moment they are posted, you can short the affected farmer before the professional community processes the news.

**The chain of logic**
```
ISA confirmed at a Mowi site on a Monday morning (Mattilsynet PDF)
        ↓
Mowi must cull the entire farm: sudden biomass loss
        ↓
Mowi stock falls over the next 1 to 10 days as the news spreads
        ↓
You have already shorted it
```

**What we found**

We tested 17 historical events (ISA outbreaks, algal blooms, de-licing accidents, and Pancreas Disease events) going back to 2016. The results were clear but discouraging:

| Event type | Average return at day +1 | Signal |
|---|---|---|
| All 17 events | -0.13% | Noise (t = -0.28) |
| ISA specifically | Flat | Random |
| Algal bloom and de-licing (n=4) | -2.3% | Directionally consistent, but only 4 events |
| Pancreas Disease events (n=5) | +4.4% | Counterintuitive, likely noise |

The aggregate signal showed near-zero effect. At longer horizons of 5, 10, and 21 days the affected companies were actually outperforming their peers on average, which is the opposite of the hypothesis.

**Why the data is unreliable**

Nearly all 17 events were reconstructed from secondary sources (news articles, annual reports, academic papers) rather than the original Mattilsynet filings. This means:

1. **Dates are wrong.** Approximate dates scramble the analysis. An ISA confirmed on March 15 but recorded as "March" produces random noise.
2. **Company attribution is guesswork.** "ISA detected in Nordland" attributed to Mowi because they operate there could easily be wrong.
3. **Small sample.** Even with perfect data, 17 events across 5 correlated stocks is limited statistical power.

**The only interesting sub-result**

The mortality sub-group (4 events: the 2019 Nordland algal bloom and the 2021 de-licing accident) showed -2.3% mean return at day +1, with 4 of 4 events negative (t = -3.05, p = 0.055). This is directionally consistent, but it is 4 events dominated by a single large incident, and the approximate dates mean this could easily be coincidence.
""")

    with col2:
        st.warning("""
**Why the data problem is hard to fix**

Mattilsynet publishes disease confirmations as PDF notices in Norwegian. There is no structured API. Doing this properly would require:

1. Scraping Mattilsynet's weekly notices going back 8 or more years
2. Parsing Norwegian-language PDFs to extract date, locality ID, disease type, and severity
3. Cross-referencing locality IDs against Barentswatch's site register to identify the operator
4. Then running the event study on precise dates

This is substantial engineering work, not analysis.
""")

        st.info("""
**Is the concept valid?**

Yes, if you have exact confirmation dates.

The logic is sound: ISA triggers mandatory culling, which is a sudden and quantified biomass shock. The market likely under-reacts to a Mattilsynet notice buried in Norwegian regulatory documents relative to a Reuters headline.

The 2019 algal bloom destroyed roughly 8 million salmon across multiple farms. That is a material event.

The gap between a plausible concept and a tradeable signal is entirely a data-quality problem, not a theory problem.
""")

        st.markdown("""
**Verdict:** No signal found at the aggregate level. The concept is structurally sound but the 17-event dataset is too small and too imprecise to test it properly. Not worth pursuing without a structured Mattilsynet data source.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 4 — FEED-COST SQUEEZE
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H4: Feed-Cost Squeeze via the El Nino, Anchovy, and Fishmeal Chain")
    _status_pill("weak")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

Salmon feed is the biggest single cost in farming. A major ingredient is fishmeal, made from Peruvian anchovies. The Peruvian anchovy catch is severely disrupted by El Nino. Warm Pacific ocean conditions push the fish deep and the government cuts catch quotas. Less catch means less fishmeal and prices spike.

If you can see a fishmeal price spike coming via El Nino climate forecasts before the salmon farmer reports earnings three to six months later, you can position for the margin hit before the stock reflects it.

**The chain of logic**
```
El Nino develops (NOAA climate index)
        ↓ (2 to 4 months: quota cuts, lower catch)
Fishmeal prices spike
        ↓ (1 to 3 months: feed contracts adjust)
Salmon farmer feed costs rise
        ↓ (quarterly lag to earnings reports)
Farmer margins compressed, stock underperforms
        ↓
You are already short, having seen the El Nino signal months earlier
```

**What we tested**

- **Test A (climate to commodity):** Does El Nino (measured by the NOAA ONI index) predict fishmeal prices 0 to 12 months later?
  - Result: No relationship found. Only 6 independent El Nino events were available since 2005, which gives the test very limited statistical power.

- **Test B (commodity to returns):** Does a fishmeal price rise predict lower salmon-farmer returns 0 to 9 months later?
  - Result: A hint of a signal at 4 to 5 month lags. At lag 5, the coefficient was -0.33 with a t-stat of 2.49. Rising fishmeal in month 0 predicted negative returns in month 5. The sign was correct across most farmers. However, at lag 2 the sign flipped positive, which is suspicious. After correcting for testing 10 different lags simultaneously (Bonferroni correction), the threshold to trust a result rises to t = 2.81. The best result was 2.49, just short.

- **Test C (backtest):** Built a simple signal using fishmeal rises and active El Nino conditions, backtested short positions.
  - Result: Sharpe ratio = -0.05. Not tradeable.

**Key context: sample size**

There have only been 6 distinct El Nino events since 2005. A hint that appears across those 6 events could easily be coincidence. The data does not have the power to confirm or deny the hypothesis with any reliability.
""")

    with col2:
        st.info("""
**What a Sharpe ratio means**

A **Sharpe ratio** compares a strategy's returns to its risk. Rough guide:
- Below 0.5: marginal, hard to justify trading costs
- 0.5 to 1.0: decent
- Above 1.0: good
- Above 2.0: exceptional

The backtest produced **-0.05**. After costs, the strategy made slightly less than nothing while taking risk to do it.
""")

        st.info("""
**The lag-5 result: signal or noise?**

The t-stat of 2.49 at a 5-month lag is the most interesting result across all five hypotheses. It is not random at face value. It is consistent across multiple farmers and the sign makes economic sense: fishmeal goes up and margins go down five months later.

The honest assessment is that we do not know. The multiple-testing correction means we would expect to find one result this strong by chance alone when testing 10 lags. But the consistent direction across tickers makes it slightly more credible than pure luck.

If there is independent evidence that Norwegian farmers forward-contract feed on roughly 5-month cycles, that prior would strengthen the case considerably. Without it, it remains a weak signal.
""")

        st.markdown("""
**Verdict:** Not strong enough to trade. The lag 4 to 5 pattern is the most economically coherent finding of the five hypotheses, but the backtest is flat and the sample size is too small to trust the regression results.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  HYPOTHESIS 5 — BIOMASS NOWCAST
    # ══════════════════════════════════════════════════════════════════════════
    _h2("H5: Biomass Nowcast: Forecasting Salmon Supply Before the Market Does")
    _status_pill("blocked")
    st.markdown("")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**The idea in plain English**

All the salmon currently swimming in Norwegian sea cages will be harvested over the next 6 to 18 months. If you know how much live fish is in the water right now and you know how fast salmon grow, you can estimate future harvest supply quite accurately. More supply means price falls. Less supply means price rises.

Fish Pool runs a market for salmon price futures, similar to a futures market for oil. If the futures price does not reflect your supply forecast, you have a trade: buy futures when your model says supply will be lower than the market expects, and sell when it will be higher.

**The chain of logic**
```
Total live salmon biomass in Norwegian sea cages (measured weekly per site)
        ↓ (apply growth models: temperature, feeding rates)
Forecast harvest supply for the next 6, 12, 18 months
        ↓
Compare to Fish Pool forward curve
        ↓
If your forecast exceeds the forward curve: sell futures (price will fall)
If your forecast is below the forward curve: buy futures (price will rise)
```

**Why this might beat the professionals**

This is the most analytically straightforward of the five. It is essentially a supply forecasting model. The reason it might work is purely about data granularity and speed. The official Norwegian biomass statistics are published monthly by Fiskeridirektoratet. The raw weekly site-level data is published by Barentswatch before the aggregated government statistics come out. A real-time aggregation of that raw data could give you a supply estimate that is weeks ahead of the consensus.

**What we built**

The full pipeline is written and ready:
- Salmon price data: 258 months from Statistics Norway (SSB), confirmed working
- Harvest proxy: weekly export volumes from SSB (258 months), used as a stand-in for actual harvest data
- Equity prices: all 5 Oslo tickers
- Statistical tests A through D all coded and ready to run

**What is blocking it**

The key input, monthly or weekly granular biomass data from Fiskeridirektoratet, has no public API. The Norwegian Fisheries Directorate website is JavaScript-rendered with no data endpoints. SSB used to publish monthly aquaculture statistics but shut down that series in 2019.

The data exists. Fiskeridirektoratet publishes a Biomassestatistikk Excel file on their website. It requires a manual download to obtain.
""")

    with col2:
        st.info("""
**How Fish Pool forwards work**

Fish Pool is the exchange where salmon processors, retailers, and farmers buy and sell salmon price contracts for future delivery, similar to oil futures on the CME. A Fish Pool forward is a contract that locks in a price for a specific volume of salmon at a specific future date.

If you sell a forward at 90 NOK per kg and the actual price at delivery turns out to be 75 NOK per kg, you made 15 NOK per kg profit. If it ends at 100 NOK per kg, you lost 10 NOK per kg.

Fish Pool requires a subscription to access forward prices, which means the final trading test cannot run without one.
""")

        st.success("""
**Structurally the strongest idea of the five**

If the data were available, this would be the most defensible trade:
- Clear supply and demand mechanism
- Quantifiable input (biomass in tonnes)
- Liquid, priced instrument (Fish Pool forwards)
- The competing analysts rely on the same lagged government statistics you are trying to beat

The data constraint is real but it cuts both ways: the absence of an API makes it harder for quant funds to automate as well.
""")

        st.markdown("""
**Verdict:** Not testable without the biomass input file. The pipeline is complete and would run immediately upon obtaining the Fiskeridirektoratet Biomassestatistikk data. Structurally the most compelling of the five hypotheses if the data can be sourced.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  CONCLUSION
    # ══════════════════════════════════════════════════════════════════════════
    _h2("Conclusion")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
**Why this research was concluded**

After running all five hypothesis tests on the full available dataset, no statistically significant and exploitable edge was identified in Norwegian salmon equities using publicly available data. The research was concluded on those grounds.

The Norwegian salmon equity market is among the most efficiently covered specialty markets in Europe. A small number of dedicated sell-side teams at Oslo-based banks have mapped every farm site to every corporate entity, and they process the weekly Barentswatch lice data in real time. The structural advantage that a retail or independent researcher might expect to have, access to granular public data that is difficult to aggregate, turns out to be an advantage the specialist community has already built into their models.

**Main takeaways**

**The mechanism is real, the edge is not.** Across every hypothesis, the underlying biological and economic logic held. Lice do hurt farmers. El Nino does raise fishmeal prices. ISA events do cause stock drops. But in each case, the effect was either already priced at the moment the data became public (H2), too small to be statistically reliable at a tradeable lag (H4), or too noisy to extract cleanly from available data (H1, H3).

**Publication lag is the central constraint.** The clearest finding of the research was H2: lice efficiency showed a real signal at lag 0 (t = -1.99) that completely disappeared at lag 1. That single week gap, the difference between when Barentswatch publishes and when you can execute, is where the edge died. The specialists are already positioned before retail data reaches the market.

**Data quality was a recurring limit.** H3 could not be properly tested because exact disease-event dates were unavailable in structured form. H5 could not be tested because the granular biomass data has no API. In both cases the hypothesis remains open in theory, but the data engineering required to test it properly is substantial.

**The feed-cost chain (H4) is the only result worth watching.** A t-stat of 2.49 at a 5-month lag does not survive a multiple-testing correction, and the backtest Sharpe was -0.05. But the economic mechanism is real and the directionality was consistent across farmers. If independent evidence ever confirmed that Norwegian farmers typically forward-contract feed on 5 to 6 month cycles, that would provide a structural reason to take the lag seriously. Without it, the result is filed as a weak and unconfirmed hint.
""")

    with col2:
        st.info("""
**Summary of findings by hypothesis**

**H1: SST Lead-Lag**
Best t-stat: 1.28. No signal. The temperature signal does not predict relative farmer returns on any horizon tested.

**H2: Lice Efficiency**
t = -1.99 at lag 0, drops to noise at lag 1. Mechanism confirmed, but the market already prices it at publication. No exploitable lag exists.

**H3: Acute Events**
No aggregate signal (t = -0.28). Data too imprecise and sample too small to test the concept properly.

**H4: Feed-Cost Squeeze**
Weak hint at lag 5 months (t = 2.49), does not survive multiple-testing correction. Backtest Sharpe = -0.05.

**H5: Biomass Nowcast**
Not testable without proprietary biomass input data. Structurally the strongest concept of the five.
""")

        st.markdown("""
**On the broader lesson**

The salmon market is a useful case study in the limits of alternative data for equity alpha. The data is public, granular, and timely. The mechanisms connecting it to stock prices are real and well-understood. Yet none of it translates into a tradeable edge, precisely because the same data is consumed in real time by analysts whose sole job is to do exactly this. Alternative data creates an edge when it is hard to access, hard to interpret, or not yet on the radar of specialist coverage. In Norwegian salmon, none of those conditions hold.
""")

    _divider()

    # ══════════════════════════════════════════════════════════════════════════
    #  DATA SOURCES
    # ══════════════════════════════════════════════════════════════════════════
    _h2("Data Sources")
    st.caption("All data sources used across the five hypothesis tests.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
**Market and price data**

**Yahoo Finance via yfinance**
Weekly adjusted closing prices for MOWI.OL, SALM.OL, LSG.OL, GSF.OL, and BAKKA.OL. Coverage: January 2012 to June 2026. 755 weekly observations per ticker.

**World Bank Commodity Price Data (Pink Sheet)**
Monthly fishmeal spot prices in USD per tonne. Used in H4 (Feed-Cost Squeeze) to measure input cost shocks.

**Statistics Norway (SSB)**
Monthly Norwegian salmon export volumes and prices. 258 monthly observations used as a harvest proxy in H5 (Biomass Nowcast).

**Climate and ocean data**

**NOAA OISST v2.1 (via NOAA CoastWatch ERDDAP)**
Daily sea-surface temperature at 0.25-degree resolution. Dataset: ncdcOisst21Agg_LonPM180. Coverage: January 2012 to present. Used in H1 (SST Lead-Lag) to compute weekly zone temperature anomalies for all 11 Norwegian production zones.

**NOAA Oceanic Nino Index (ONI)**
Monthly El Nino and La Nina climate index. Used in H4 to identify active El Nino events since 2005.
""")

    with col_b:
        st.markdown("""
**Norwegian aquaculture data**

**Barentswatch FishHealth API (OAuth2)**
Weekly site-level adult female lice counts per fish for all active Norwegian aquaculture sites. 364,953 site-week observations across 1,186 unique sites from 2012 to 2026. Used in H2 (Lice Efficiency). Requires free application registration at developer.barentswatch.no with FishHealth scope.

**Fiskeridirektoratet (FDIR) Public API**
Norwegian aquaculture site registry. Provides company-to-site mapping, production zone assignments, and entity-level ownership records. 484 sites mapped across four test farmers: Mowi (200 sites), SalMar (142 sites), Lerøy (122 sites), and Grieg Seafood (20 sites). No authentication required.

**Norwegian Food Safety Authority (Mattilsynet)**
Weekly disease-status notices published as PDF documents in Norwegian. Used in H3 (Acute Events) to reconstruct 17 historical disease events from 2016 onward. Data was sourced indirectly from secondary references (news articles, annual reports, academic literature) rather than original filings, which is a known limitation of the H3 analysis.
""")
