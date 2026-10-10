# External Nowcast/Consensus Signal Test (2026-07-18)
Verification of a build decision, not a DSR trial -- not logged to the registry. Methodology locked in README.md before running. Promotion criterion (locked before seeing results): p < 0.05 AND n >= 15 walk-forward test periods.

## GDPNow vs Part 1 growth direction
- n=39 periods tested (walk-forward, anchor applied) | matches=19 | hit rate=48.7% | binomial p=1.0000
- **Verdict: NOT significant (p=1.0000)**

## Cleveland Nowcast vs Part 1 inflation direction
- n=102 periods tested (walk-forward, anchor applied) | matches=67 | hit rate=65.7% | binomial p=0.0020
- **Verdict: SIGNIFICANT (p=0.0020) -- clears the promotion bar**

## SPF CPI-surprise vs Part 1 inflation direction (quarterly proxy)
- n=77 periods tested (walk-forward, anchor applied) | matches=49 | hit rate=63.6% | binomial p=0.0220
- **Verdict: SIGNIFICANT (p=0.0220) -- clears the promotion bar**

## SPF RGDP-surprise vs Part 1 growth direction
- n=85 periods tested (walk-forward, anchor applied) | matches=54 | hit rate=63.5% | binomial p=0.0165
- **Verdict: SIGNIFICANT (p=0.0165) -- clears the promotion bar**

## Summary table

                                                          signal   n  hit_rate_pct  p_value                                            verdict
                               GDPNow vs Part 1 growth direction  39          48.7   1.0000                         NOT significant (p=1.0000)
                 Cleveland Nowcast vs Part 1 inflation direction 102          65.7   0.0020 SIGNIFICANT (p=0.0020) -- clears the promotion bar
SPF CPI-surprise vs Part 1 inflation direction (quarterly proxy)  77          63.6   0.0220 SIGNIFICANT (p=0.0220) -- clears the promotion bar
                    SPF RGDP-surprise vs Part 1 growth direction  85          63.5   0.0165 SIGNIFICANT (p=0.0165) -- clears the promotion bar
