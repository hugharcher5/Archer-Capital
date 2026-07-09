# Russell-3: Boundary-Crossing Subset — Results

Family member of **"Russell Reconstitution"** (established by Russell-1,
trial #17) — shares Russell-1's DSR slot (trial_number=17, N_TRIALS
unchanged), logged as `russell3_boundary_crossing`. Reused Russell-1's
existing sourced dataset entirely — no new data acquisition.

## Sample: NOT thin

748 repeat-crosser tickers (appear in 2+ distinct years across the combined
additions+deletions universe, 2016-2023) — **~39% of the full universe**.
This is far larger than the "under 10" scenario flagged as a risk in the
pre-registration; it's consistent with well-documented Russell 2000/Microcap
boundary churn (small/micro-caps bounce across the cutoff on ordinary price
volatility, not fundamental change — which is, if anything, exactly the
"nothing fundamentally different about the company" case the hypothesis
describes).

**Data gap**: no Russell 3000 deletions PDF was recoverable for 2018 or 2019
in the original acquisition (r3000 deletions genuinely absent from the
archive those two years, only Microcap deletions survive for 2018). Any
repeat-crosser whose only second leg would have been an undetected 2018/2019
deletion is missed — the 748-ticker / 1,751-event count is a modest
undercount, not an overcount.

## Results table (event-level, N = crossing events)

| Subset | N events | CAAR | SD | t-stat | p-value | Win rate |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | 1,493 | +0.31% | 19.25% | +0.615 | 0.539 | 46.0% |
| Additions-only | 822 | **-0.60%** | 17.96% | -0.958 | 0.338 | 45.5% |
| Deletions-only | 671 | **+1.42%** | 20.67% | +1.775 | 0.076 | 46.6% |

1,751 total crossing events across 748 tickers; 1,493 priced (85.3%
coverage — 1 ticker needed a fresh Tiingo fetch, the rest were already
cached from Russell-1).

No annualized Sharpe/CALMAR reported, per pre-registration — not meaningful
on non-continuous event samples.

**Sign is backwards in both splits.** The hypothesis predicts additions
drift *up* (anticipated buying) and deletions drift *down* (anticipated
selling). Here additions are slightly *negative* and deletions are slightly
*positive* — the opposite of the predicted direction in both cases. Neither
clears conventional significance (deletions-only is the closest, at
p=0.076, but wrong-signed, so this is not a near-miss for the hypothesis —
it's a marginal result pointing the wrong way).

## Falsification diagnostics

1. **Concentration check**: clean in every subset — top-2 tickers account
   for 2.4% (pooled), 4.0% (additions), 2.8% (deletions) of total |abnormal
   return| mass, all far under the 40% flag threshold. With N in the
   hundreds per subset, no single name's crossings could plausibly dominate,
   and none does.
2. **Statistical-independence caveat** (not in the original falsification
   list, added for honesty): treating each ticker-year crossing as an
   independent observation likely overstates precision somewhat — a
   chronically boundary-hugging stock's multiple crossings aren't fully
   independent draws (shared idiosyncratic volatility character), and
   within-year events share exposure to that year's specific market
   conditions net of the IWM benchmark subtraction. This would make the
   *true* standard errors larger than reported, i.e. the already-null
   t-stats are, if anything, optimistic. Does not change the conclusion.

## Comparison to Russell-1 (the actual point of this trial)

| | N | CAAR | t-stat |
|---|---:|---:|---:|
| Russell-1 (full list, 8 annual cycles) | 8 | +0.64% | +0.416 |
| Russell-3 pooled (repeat-crossers) | 1,493 | +0.31% | +0.615 |
| Russell-3 additions-only | 822 | -0.60% | -0.958 |
| Russell-3 deletions-only | 671 | +1.42% | +1.775 |

The boundary-crossing subset does **not** show a materially different or
cleaner effect than Russell-1's full-list null. If anything, splitting by
direction reveals the two halves point the wrong way relative to the
hypothesis, which the full pooled Russell-1 test couldn't have shown (it
only ever tested additions). The larger N here (1,493 vs. 8) makes this a
*more*, not less, informative null than Russell-1's — with hundreds of
events per subset and clean concentration diagnostics, low power is not a
credible excuse for the null result.

## Verdict

**Null, and where a subset comes closest to significance (deletions-only,
p=0.076), it's wrong-signed.** Two independent attempts within the Russell
Reconstitution family (full addition list, boundary-crossing subset) have
now both failed to find a tradable pre/post-effective-date drift.

**Recommendation: close the "Russell Reconstitution" family.** There is no
basis in either trial to continue searching for a variant that works — per
the pre-registered instruction, a favorable-looking result on a thin sample
would not have been treated as validated, and here the sample isn't even
thin (N=1,493) and still shows nothing supportive.
