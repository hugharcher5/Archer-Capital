# Private Competitor Distress — Research Writeup

*Tested, failed at Layer 1, closed. The thesis, the test, and why it was stopped.*

## Thesis

Go long listed small-caps whose **private** competitors are showing distress in Companies House filings — late accounts, overdue confirmation statements, administration. When a private competitor fails, its revenue redistributes to surviving players; where a listed small-cap is a primary beneficiary, that share gain should precede its appearance in consensus estimates, leaving a window where the stock is mispriced relative to its improving competitive position. Private firms carry no share price and negligible analyst attention, so the distress sits in public filings that few participants screen systematically — the structural condition for an underpriced signal.

## Signal construction

Each private competitor receives a composite distress score in [0, 1]:

| Component | Weight |
|---|---|
| Accounts filed late | 40% |
| Company status (e.g. administration) | 30% |
| Accounts overdue | 20% |
| Confirmation statement late | 10% |

Distressed private entities are mapped to listed competitors via shared SIC codes. When a listed name's average competitor-distress score exceeds 0.35, it is bought and held 90 days, with a half-threshold exit at 0.175. Costs: 30bps round-trip — optimistic for small-caps, so realistic costs only worsen the outcome. Thresholds fixed ex ante; no parameter tuning.

## Test

Layer 1 of the seven-layer pipeline: in-sample edge on the data the signal was built on — the necessary-but-not-sufficient first gate. The metric is abnormal return versus an equal-weighted peer basket, stripping out market and sector moves. Fail here and the remaining six layers do not run.

## Results

| Measure | Result |
|---|---|
| Sharpe | −0.36 |
| CAGR (abnormal) | −6.48% |
| Max drawdown | −74.2% |
| Win rate | 48.4% |
| Trades | 62 |

Cumulative abnormal return drifts steadily down across the sample — the longs underperformed their peers, the opposite of the thesis. The event study (CAR over the 21 days post-signal) is mixed and directionless, with no systematic drift either way.

## Why it failed

1. **Latency.** UK private accounts can be filed up to ~21 months after period end. By the time a late-filing flag triggers, any share migration has largely happened or is already in the listed rival's price. The signal fires on stale information.
2. **Attribution.** SIC codes are too coarse to isolate the true beneficiary. A shared classification doesn't imply genuine competition, and a failed firm's demand disperses across many rivals rather than one listed name — so the book frequently held the wrong stock.
3. **Magnitude.** Share migration is real but gradual and diffuse — negligible against everything else moving a small-cap over a 90-day hold.

## Conclusion

Closed at Layer 1. A signal that can't clear the in-sample gate doesn't earn MCPT, walk-forward, or crisis replay. The recurring lesson across this research: a sound economic mechanism is necessary but not sufficient — to be tradeable, the underlying data has to be both timely and precise. Here it was neither. The discipline worth noting is stopping at the first gate rather than tuning until the backtest flatters.
