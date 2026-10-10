# Individual-Forecaster Accuracy Within SPF (2026-07-19)
Verification of a build decision, not a DSR trial -- not logged to the registry. Methodology locked in README.md before running.

## Cross-sectional MAE dispersion (150 forecasters with >=20+ submissions)

Mean MAE=4.826 | median=4.434 | std=1.855 | IQR=[3.476, 5.964] | range=[1.834, 11.573]
**Read: MEANINGFUL spread -- forecaster identity is not interchangeable** (coefficient of variation = 0.38)

## Genuine point-in-time signals (the real test)

### SPF TOP-1 forecaster (point-in-time)
- n=85 | hit rate=60.0% | binomial p=0.0821
- NOT significant (p=0.0821)
- vs Bonferroni(6)=0.00833: **does NOT survive**

### SPF TOP-5-AVG forecasters (point-in-time)
- n=40 | hit rate=65.0% | binomial p=0.0807
- NOT significant (p=0.0807)
- vs Bonferroni(6)=0.00833: **does NOT survive**

## Circular/hindsight comparison (the WRONG way -- shown to quantify the bias gap)

### [CIRCULAR/WRONG] TOP-1 by full-sample hindsight MAE
- n=85 | hit rate=61.2% | binomial p=0.0503

### [CIRCULAR/WRONG] TOP-5-AVG by full-sample hindsight MAE
- n=31 | hit rate=71.0% | binomial p=0.0294

## Comparison against the already-tested full-panel consensus
SPF RGDP-surprise (full-panel median consensus, already tested): n=85, hit rate=63.5%, p=0.0165 (did not survive Bonferroni(4)).
