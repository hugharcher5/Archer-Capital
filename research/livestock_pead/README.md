# Livestock Disease -> Animal-Pharma Event Study (H1)

## WAHIS Access Path

Primary: WAHIS REST API at `https://wahis.woah.org/pi/getReportList` (POST, JSON payload).
The API accepts disease filters, date ranges, and returns paginated report lists.
Individual reports: `https://wahis.woah.org/pi/getReport/{report_info_id}` (GET).

**Status**: The WAHIS API returns 403 without a session cookie from the WAHIS portal.
We use a curated event table sourced from four official reporting systems:

## Provenance Audit

### Event counts by source
  - EMPRES: 51 events
  - EU_ADIS: 30 events
  - WAHIS: 20 events
  - APHIS: 17 events

### Date field definitions

| Source   | Date field used as "notification_date" | Definition |
|----------|----------------------------------------|------------|
| APHIS    | Confirmation Date | Date NVSL confirmatory testing completed and flock confirmed positive |
| WAHIS    | Report Date | Date the member state submitted the immediate notification to WOAH |
| EU_ADIS  | Notification Date | Date the outbreak was notified in the EU Animal Disease Information System |
| EMPRES   | Observation Date | Date recorded in FAO EMPRES-i alert; used as proxy for first public knowledge |

### Date discrepancy note

These date fields are NOT identical across sources:
- WAHIS report_date typically lags national confirmation by 1-3 days
- EMPRES observation_date can lag WAHIS by 0-7 days for non-EU events
- EU_ADIS notification_date is same-day or +1d vs national confirmation for EU members
- APHIS confirmation_date is the most precise (tied to lab result)

**Reconciliation**: All dates are aligned to the *public notification/confirmation date* —
the first date a market participant could have known from an official source.
This is conservative for the event study (delays entry, biases against finding drift).
We do NOT use biological onset dates (which would introduce look-ahead bias).

### Spot-check file

A stratified sample of 10 events (2 per disease where possible) is output to
`output/event_audit.csv` for manual verification before running statistics.

## Universe Assumptions

Livestock-exposed tickers: companies with >20% revenue from livestock health products.
Species exposure: mapped from 10-K segment disclosures and company presentations.
- Multi-species: PAHC, ELAN, NEOG (+ VIRP.PA, VETO.PA, DPH.L if Tiingo-available)
- Placebo (companion-only): IDXX, ZTS (mixed but companion-dominant for falsification)

## Matching Rule

An event matches a ticker if:
1. event.species IN ticker.species_exposure (or ticker has "multi" exposure)
2. event.region IN ticker.sales_regions (or ticker is "global")

Coarse mapping only — revenue-weighting deferred to H3.

## Coverage Gaps

- International tickers (VIRP.PA, VETO.PA, DPH.L) may not be available on Tiingo free tier
- Investable US livestock universe may be as thin as 3 names (PAHC, ELAN, NEOG)
- Delisted placebo names (HSKA, KIN, PETX) dropped — insufficient Tiingo coverage

## Event Data

Total events: 118
Date range: 2015-01-20 to 2025-01-15
Diseases: ASF, BTV, FMD, HPAI, NCD, PRRS

## Statistical Structure

Results are reported in three tiers:
- **(a) PRIMARY**: HPAI-only, major vs minor market — cleanest test (29/29 balance)
- **(b) SECONDARY**: all 6 diseases pooled, major vs minor
- **(c) Per-disease**: each disease separately, with major/minor split where N>=20

The verdict distinguishes (a)-only, (a)+(b), or (b)-only support.

Generated: 2026-06-21 20:15
