# Russell-1: Pre-Effective-Date Anticipatory Drift (Index Addition Baseline)

New DSR family: "Russell Reconstitution". This is the first trial in the family.

## Hypothesis

Stocks confirmed for addition to a Russell US Index (Microcap->2000 or 2000->1000
migration, or new-to-index via IPO) drift upward between the preliminary-list
announcement date and the reconstitution effective date, as traders anticipate
forced mechanical buying from index-benchmarked funds. Pure-flow hypothesis, no
fundamental mechanism. Entry after the information is public; exit at the
effective date, before any post-event reversal.

## Data acquisition (done first, per pre-registration)

FTSE Russell / Russell Investments does not host a historical archive on the
live site -- only the current year's addition/deletion PDFs are linked
(confirmed: `ru3000-additions-*.pdf` URLs for years before 2025 all 404 on the
live domain). There is no free API. The one academic replication package found
(`pyndex`, github.com/alemicheli/pyndex) requires CRSP/WRDS access, which we
don't have.

**What we used instead**: the actual primary-source PDFs FTSE Russell / Russell
Investments posted each year (both weekly "preliminary" snapshots during the
reconstitution window and the "final" confirmed list) are archived by the
Wayback Machine. These are real, clean, machine-parseable tables (Company /
Symbol / Sector), not scanned images -- confirmed by parsing a sample with
`pdfplumber` before committing to this approach. See `acquire.py` for the CDX
query logic and `parse.py` for the table extraction.

**Years recovered**: 2016-2023 (8 annual cycles). **2015 excluded**: the only
recoverable documents for that year (`russell.com/documents/indexes/*.pdf`)
are press-release "highlights" summaries, not ticker-level tables -- confirmed
by parsing them (6-7 rows of prose, not ~150-300 rows of a real table).
2024-2025 are the user-specified untouched holdout and were not used for
signal construction (2025's PDFs are in fact still live on the current site
for this same reason -- they haven't been superseded yet).

**Announcement (preliminary-post) and effective dates per year** are sourced
from FTSE Russell's own official schedule press releases / index notices for
each specific historical year (via web search), NOT the current 2026 schedule.
See `cycle_calendar.py` for the full per-year sourcing notes, including two
flagged approximations: 2018 (no PDF explicitly labeled "final" survives in
the archive; we use the last-captured "preliminary"-titled snapshot, whose
capture date trails the official final-close date by 2 days) and 2020 (the
COVID-disrupted 2020 schedule's exact first-preliminary-post date wasn't
directly confirmed by press search; we use the query-period-end date as a
conservative proxy).

## Universe

Russell 3000 = Russell 1000 UNION Russell 2000. An addition to the R3000 list
is therefore, by definition, an addition to either R1000 or R2000 -- exactly
the union the hypothesis calls for. We did NOT need to separately source
Russell-1000-specific, Russell-2000-specific, or Microcap-specific lists for
the primary universe (Microcap lists were downloaded incidentally but are not
used in the primary test).

## Files

- `acquire.py` -- Wayback CDX query + PDF download (Phase 1)
- `parse.py` -- PDF table extraction into `results/parsed_all_rows.csv` (Phase 2)
- `cycle_calendar.py` -- the 8-year cycle calendar with sourced dates (Phase 2b)
- `build_universe.py` -- combines calendar + parsed tables into
  `results/russell1_universe.csv`, one row per (year, ticker) (Phase 2c)
- `fetch_prices.py` -- Tiingo adjClose/close fetch for all tickers + IWM benchmark (Phase 3)
- `run_backtest.py` -- event-study backtest, CAAR stats, diagnostics (Phase 4-5)
- `log_to_main_registry.py` -- completes the "trial-registry logging" step named in
  run_backtest.py's own docstring but not implemented there; logs to the main
  `research/trial_registry.csv` as trial #17, new independent family "Russell
  Reconstitution".

## Addendum: 2015 IS recoverable after all (found in a later session)

The exclusion note above says only "highlights"/press-release PDFs survive for
2015 -- true for the `russell.com/documents/indexes/*.pdf` paths this pipeline
checked. But a **separate URL path** does have real ticker-level data for 2015:
the live tool page `russell.com/indexes/americas/tools-resources/reconstitution/
additions-deletions.page` was archived with server-rendered HTML `<table>`
elements (`id="Russell3000AddTable"`, `"Russell3000DeleteTable"`,
`"RussellMicrocapAddTable"`, presumably `"RussellMicrocapDeleteTable"` too),
not a PDF. Confirmed by direct fetch + `pandas.read_html`:
147 R3000 additions / 153 R3000 deletions / 244 Microcap additions, real
company/symbol/sector rows, at Wayback snapshot `20150627172039` (2015-06-27,
one day after the actual 2015-06-26 effective date -- so this is a FINAL list,
not preliminary; no snapshot exists in the archive between the 2015-06-12
preliminary announcement and the effective date, so no genuine 2015
preliminary list has been found).

Raw snapshot saved at `cache/2015_addendum/additions-deletions_20150627172039.html`
for whoever picks this up. **Not yet wired into `cycle_calendar.py` /
`build_universe.py` / the backtest** -- adding a 9th (2015) cycle changes the
in-sample N and is a deliberate methodology decision, not something to graft
on silently. If you want to add it: 2015 would be FINAL-only (no preliminary
leg), which doesn't fit this trial's entry/exit design without a documented
proxy decision (same class of caveat as the 2016/2018 proxy notes above).
