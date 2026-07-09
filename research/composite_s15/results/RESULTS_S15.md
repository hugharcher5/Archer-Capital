# S15: Low-Turnover Composite of Validated-IC Signals

**Trial #19, N_TRIALS=19 (DSR).** Registry key: `compositeIC_S15`. Status: `completed_in_sample`, **FAILS DSR**.

New independent construction — a composite of three previously-tested signals with real, correctly-signed IC (GP/A quality — S3-Q, trial #8; residual short-term reversal — S8, trial #14; IVOL-conditioned value — S7, trial #13), each of which failed as a standalone strategy for a different reason. Not a re-tune of any of the three.

## Verdict (one line)

**Diversification did not help.** The composite's Sharpe (-0.070) is worse than the best individual component recomputed on the identical shared universe (GP/A-alone, +0.007), fails DSR at N=19, and misses the promotion bar (CALMAR ≥ 1 or Sharpe ≥ 0.8) by a wide margin — driven mainly by the IVOL-Value component's continuous generalization flipping sign (IC went from +0.036 in S7's original double-sort to -0.015 here) and by GP/A and IVOL-Value being meaningfully correlated (+0.563), which ate into the diversification benefit the trial was designed to test for.

## Headline comparison table

All four portfolios computed on the **identical** per-quarter intersected universe (names with valid GP/A, ResidRev, and IVOL-Value signals that quarter), same dates, same beta-neutral sizing (60-day OLS vs EW-universe adjClose), same cost model — isolating the diversification effect from any universe/cadence difference.

| Metric | **Composite** | GP/A-alone | ResidRev-alone | IVOLValue-alone |
|---|---:|---:|---:|---:|
| Sharpe (net, annualized) | **-0.070** | 0.007 | -0.222 | -1.119 |
| CALMAR | **-0.057** | -0.038 | -0.138 | -0.197 |
| Ann. return (net) | -1.85% | -1.13% | -5.23% | -10.33% |
| Ann. return (gross) | 6.30% | 2.87% | 5.55% | -3.59% |
| Market beta | 0.093 | -0.115 | 0.220 | -0.022 |
| Market beta t-stat | 1.122 | -1.185 | **2.223** | -0.391 |
| Max drawdown | -32.60% | -29.73% | -37.79% | -52.45% |
| Mean IC (Spearman) | 0.0286 | 0.0404 | 0.0290 | **-0.0151** |
| IC t-stat | 1.644 | 2.148 | 1.428 | -0.983 |
| Win rate | 41.7% | 45.8% | 41.7% | 16.7% |
| Mean turnover / quarter | 59% | 23% | 80% | 52% |
| Cost drag (bps/yr) | 801 | 397 | 1074 | 712 |
| Clears DSR (N=19) | **NO** | NO | NO | NO |
| N periods | 24 | 24 | 24 | 24 |

Per-period Sharpe threshold: `norm.ppf(1-1/19)/sqrt(24)` ≈ 0.297 (per-quarter units); composite's per-period SR is well below this.

### Original registry numbers, for reference only

These used each signal's own broader universe (not intersected) and, for S8, monthly cadence (72 periods vs. 24 here) — **not directly comparable** to the table above, which is why the "-alone" columns were recomputed rather than re-quoted:

| Study | Trial | Sharpe | CALMAR | Mean IC | IC t-stat |
|---|---|---:|---:|---:|---:|
| gp_S3-GP | #8 | +0.1338 | +0.0332 | +0.0404 | +2.283 |
| ivolvalue_S7 | #13 | -0.5444 | -0.2095 | +0.0358 | +1.356 |
| residreversal_S8 | #14 | -1.1075 | -0.2154 | +0.0360 | +3.397 |

## Year-by-year returns (net), 2015-2020

| Year | Composite | GP/A-alone | ResidRev-alone | IVOLValue-alone |
|---|---:|---:|---:|---:|
| 2015 | -4.47% | -1.69% | -21.39% | -1.56% |
| 2016 | -8.33% | -11.93% | +7.12% | -10.45% |
| 2017 | -0.32% | +4.65% | -6.95% | -12.32% |
| 2018 | +2.15% | **+45.48%** | -7.66% | -12.67% |
| 2019 | -11.41% | -16.22% | -9.68% | -18.84% |
| 2020 | +13.17% | -15.42% | +10.89% | -5.09% |

No single year dominates the composite's full-period average the way 2018 dominates GP/A-alone (+45% in one year, driven by a small-n quarter — see turnover/guard notes below) — the composite's return profile is smoother year-to-year than any individual component, which is the diversification mechanism working as intended on *volatility*, just not enough to produce a positive risk-adjusted return, since none of the three components had a positive gross-minus-cost edge on this shared universe to diversify *into*.

## Market beta: does staying beta-neutral compound or cancel?

Composite beta = **0.093 (t=1.122)** — not statistically significant, i.e. the composite is beta-neutral within noise. But this is not simply because all three legs are individually flat: **ResidRev-alone has a significant positive beta of 0.220 (t=2.223)** on this universe — the 60-day-OLS beta-neutral sizing does not fully neutralize the reversal leg, likely because names crossing the reversal decile threshold each quarter are systematically higher/lower beta than the beta-computation window captures. GP/A-alone (-0.115) and IVOLValue-alone (-0.022) sit closer to zero. The composite's beta ends up as roughly an average of the three, diluting ResidRev's beta tilt rather than eliminating it — **three beta-neutral legs combined does not guarantee the combination stays beta-neutral**; it landed close to zero here mostly by the negative-beta legs offsetting the one significantly positive leg, not because each leg was individually clean.

## Direct comparison — did diversification help?

**No.** Composite Sharpe (-0.070) < best single component alone (GP/A-alone, +0.007). The composite is not even the best of the four portfolios shown — it's actually second-worst by CALMAR magnitude among GP/A and ResidRev. Diversification reduced volatility/drawdown concentration in any one component (visible in the smoother year-by-year table above) but could not overcome that two of the three components had negative or near-zero net edge on this shared, cost-loaded universe, and the third (IVOL-Value) actively hurt via a wrong-signed continuous generalization (see falsification diagnostics).

## Falsification diagnostics

### 1. Sign-convention audit

| Component | Mean IC (shared universe) | IC t-stat | Original registry sign | Consistent? |
|---|---:|---:|---|---|
| GP/A | +0.0404 | +2.148 | positive | **Yes** |
| ResidRev | +0.0290 | +1.428 | positive | **Yes** (weaker t, fewer/quarterly periods vs. original 72 monthly) |
| IVOL-Value | **-0.0151** | -0.983 | positive | **No — sign flipped** |

The IVOL-Value component is the diagnostic finding of this trial: S7's original double-sort (hard top-tercile IVOL gate, then value-tercile *within* that subset) had positive IC (+0.036, t=1.36). The continuous generalization built for this composite — `ivol_value_raw = rank_pct(IVOL) × rank_pct(E/P)`, scored over the *full* universe with no hard IVOL gate — produces a small, statistically insignificant **negative** IC instead. This confirms the research agent's caution at design time: S7's edge lived specifically in the *conditional/interaction* structure (value only works once you're already in the priciest-to-arbitrage, highest-IVOL names), and a multiplicative rank score computed over the whole universe dilutes that structure rather than preserving it — most of the universe (names in the bottom two-thirds of IVOL) contributes rank_pct(IVOL) mass that has nothing to do with the tested hypothesis. **This is new methodology, not a re-tune of S7, and it does not reproduce S7's edge.**

### 2. Concentration check (composite portfolio)

Per-period share of total |dollar-weighted return contribution| (both legs pooled) from the largest and top-3 names:

- Mean top-1 name share: 6.0% (max 19.3%)
- Mean top-3 name share: 12.8% (max 24.1%)
- Periods with top-3 share > 30%: **0/24**

Clean — the composite's return is not being driven by a handful of names in any period.

### 3. Pairwise correlation of component-alone quarterly net L/S returns (in-sample)

| | GP/A | ResidRev | IVOLValue |
|---|---:|---:|---:|
| **GP/A** | 1.000 | -0.175 | **0.563** |
| **ResidRev** | -0.175 | 1.000 | 0.031 |
| **IVOLValue** | 0.563 | 0.031 | 1.000 |

GP/A and ResidRev are close to the "genuinely uncorrelated" hypothesis stated at pre-registration (-0.175). ResidRev and IVOL-Value are also close to uncorrelated (+0.031). But **GP/A and IVOL-Value are meaningfully correlated at +0.563** — not the "three distinct, roughly-uncorrelated mechanisms" the trial's motivation assumed. This is a partial explanation for the lack of diversification benefit: two of the three legs move together often enough that the effective number of independent bets is closer to two than three, on top of IVOL-Value's own wrong-signed IC on this universe.

## Construction notes (pre-registered deviations, flagged as designed)

- **IVOL-Value continuous scalar**: `rank_pct(IVOL) × rank_pct(E/P)`, both computed cross-sectionally on the full per-quarter universe. S7's original double-sort has no natural continuous scalar (it was designed as a hard-gated two-stage sort, not a score); this is new methodology built specifically for this composite, not a reuse of S7's own construction. It does not reproduce S7's IC sign (see falsification diagnostic #1 above).
- **ResidRev cadence**: evaluated at quarterly dates (`compute_residual_reversal()` called with quarterly `rebalance_date`) rather than S8's native monthly cadence, to match the composite's quarterly reform. This is a valid mechanical reuse of the same function but redefines what's captured (last-month-of-quarter residual only, not a monthly-refreshed series) and drops S8's turnover-discipline carry mechanism entirely (no carry needed — full quarterly reform).
- **Beta source standardized**: all four portfolios (composite + 3 "-alone" comparisons) use the same 60-day-OLS-vs-EW-universe-adjClose beta for leg sizing, rather than each signal's original beta source (S7 originally reused its own IVOL-regression beta). This ensures beta-neutral construction itself cannot explain any Sharpe difference between the four portfolios.
- **"-Alone" universe**: all three standalone comparison portfolios are drawn from the *same* per-quarter intersected universe as the composite (names with all three signals present), not each signal's own broader original universe — a deliberate choice to isolate the diversification effect from universe-composition differences. This is why the "-alone" numbers in the headline table differ from the original registry numbers quoted for reference.
- **Guard**: quintile leg = ceil(n×0.20); widen to terciles if leg < 20 (S5/S10/S11 convention). Fired 0/24 periods for the composite (universe sizes ranged 190-388 names after the three-way intersection, comfortably above the guard threshold throughout).

## Data / infrastructure

No new data sourcing — fully reused existing caches: `mgmt_pit/cache/{tickers.csv, sec_prefilter.csv, prices_all.pkl, adjclose_all.pkl, sec_facts_trimmed.json, gp_facts_trimmed.json}` and `ivol_value/cache/value_facts_trimmed.json`. Runtime: 261s (4.4 min) end-to-end including all cache loads.

Output files: `research/composite_s15/results/{s15_period_returns.csv, s15_universe_audit.csv, s15_concentration.csv, s15_component_correlation.csv, s15_year_by_year_{composite,gp_alone,residrev_alone,ivolvalue_alone}.csv}`.
