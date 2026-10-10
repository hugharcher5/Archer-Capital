# NY Fed Staff Nowcast Test (2026-07-20)
Verification of a build decision, not a DSR trial -- not logged to the registry. Methodology locked in README.md before running. Last free growth-prediction candidate per the brief.

## NY Fed Nowcast (real-time only, 2016+, excl. 2021-2023 gap) vs Part 1 growth direction
- n=35 | hit rate=40.0% | binomial p=0.3105
- NOT significant (p=0.3105)

## [SUPPLEMENTARY, incl. 2002-2015 reconstructed] NY Fed Nowcast vs Part 1 growth direction
- n=91 | hit rate=45.1% | binomial p=0.4018
- NOT significant (p=0.4018)

## Multiple-testing correction (growth-only family of 5)

Bonferroni(5) = 0.010. Primary result p=0.3105 **does NOT survive**.

## Comparison against every other growth signal tested

                            signal  n  hit_rate_pct  p_value                       verdict
                            GDPNow 39          48.7   1.0000               NOT significant
    SPF RGDP-surprise (full panel) 85          63.5   0.0165 Clears p<0.05, not Bonferroni
    SPF RGDP TOP-1 (point-in-time) 85          60.0   0.0821               NOT significant
SPF RGDP TOP-5-AVG (point-in-time) 40          65.0   0.0807               NOT significant
   NY Fed Nowcast (real-time only) 35          40.0   0.3105    NOT significant (p=0.3105)
