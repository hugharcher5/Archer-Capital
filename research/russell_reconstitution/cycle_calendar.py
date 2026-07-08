"""
Phase 2b: The Russell-1 cycle calendar -- one row per year, 2016-2023.

Every date below is sourced from FTSE Russell's own official schedule
press releases / index notices for that specific historical year (NOT the
current 2026 schedule), cross-checked against the Wayback-archived PDF
capture dates wherever we could recover an actual PDF (2019-2023 all have
an exact filename-date match to the sourced effective date; 2016/2017/2018
are corroborated by press coverage summarizing FTSE Russell's schedule
announcements for those years). See README.md for the source list.

preliminary_posted : first date FTSE Russell posted the preliminary
    add/delete lists that year (after 6pm ET). This is T0 for entry timing.
effective_date : the date the reconstitution is "final after close" --
    this is exit timing, and (combined with the final list PDF) the
    confirmed-membership date.
final_source_file : which downloaded PDF we treat as the CONFIRMED final
    Russell 3000 additions list for that cycle.
final_is_exact : True if the PDF's own filename date matches the sourced
    effective_date exactly (2019-2023) or is explicitly labeled "final"
    (2017, 2023). False means we are using the best available proxy
    (2016: captured 6 days after the effective date, no drift expected;
    2018: no PDF explicitly labeled "final" exists in the archive -- we
    use the last preliminary-labeled capture, whose capture date (June 24)
    trails the official last-update-and-final date (June 22) by 2 days,
    so it should already reflect the final, not preliminary, membership).
2015 is EXCLUDED: only press-release/"highlights" PDFs (not ticker-level
tables) could be recovered for that year -- see acquire.py docstring.
"""

import pandas as pd

CYCLES = [
    # year, preliminary_posted, effective_date, final_source_file, final_is_exact, prelim_source_note
    (2016, "2016-06-10", "2016-06-24",
     "2016_20160630013131_russell-3000-index-2016-additions", False,
     "Preliminary date per FTSE Russell 2016 schedule press coverage (rank day May 27, "
     "prelim posted June 10, updates June 17/24, final after close June 24). Final-list "
     "capture is dated 6 days after the effective date; the same page (captured June 21, "
     "mid-cycle) and this one differ by exactly 2 names, consistent with normal late drops."),
    (2017, "2017-05-31", "2017-06-23",
     "2017_20180201200920_final-r3000-additions-2017", True,
     "Preliminary date per press coverage of the 2017 schedule (rank day May 12, prelim "
     "'on or before May 31'). Final source file's URL slug is explicitly 'final-r3000-additions-2017'."),
    (2018, "2018-06-08", "2018-06-22",
     "2018_20180624102756_2018-preliminary-russell-3000-index-additions", False,
     "Preliminary/effective dates per press coverage of the 2018 schedule (rank day May 11, "
     "prelim posted June 8, updates June 15/22, final after close June 22). No PDF explicitly "
     "labeled 'final' was recoverable from the archive; the only surviving capture is titled "
     "'preliminary' in its URL slug but was crawled June 24 (2 days after the official final "
     "close), so its table content should already reflect the June 22 final update, not the "
     "original June 8 post. Treated as the confirmed universe with this caveat flagged."),
    (2019, "2019-06-07", "2019-06-28",
     "2019_20221014032449_RU3000_Additions_20190628.pdf", True,
     "Preliminary date per press coverage of the 2019 schedule (rank day May 10, prelim posted "
     "June 7, updates June 14/21, final after close June 28). Final source filename date "
     "(20190628) matches the sourced effective date exactly."),
    (2020, "2020-06-12", "2020-06-26",
     "2020_20210517133719_ru3000_additions_20200626_0.pdf", True,
     "Preliminary date is a proxy: official coverage confirms rank day May 8 and a query "
     "period running May 22-June 12, 2020 (COVID-disrupted schedule) but the exact 'first "
     "preliminary post' date was not directly confirmed by press search; June 12 (query-period "
     "end) is used as the conservative (latest-plausible, i.e. shortest anticipatory window) "
     "estimate. Final source filename date (20200626) matches the sourced effective date exactly."),
    (2021, "2021-06-04", "2021-06-25",
     "2021_20210625230319_ru3000_additions_20210625.pdf", True,
     "Preliminary date per official 2021 schedule press coverage (rank day May 7, prelim "
     "posted June 4, updates June 11/18, final after close June 25). Final source filename "
     "date (20210625) matches the sourced effective date exactly."),
    (2022, "2022-06-03", "2022-06-24",
     "2022_20220627135943_ru3000_additions_20220624.pdf", True,
     "Preliminary date per official 2022 schedule press coverage (rank day May 6, prelim "
     "posted June 3, updates June 10/17, final after close June 24). Final source filename "
     "date (20220624) matches the sourced effective date exactly."),
    (2023, "2023-05-19", "2023-06-23",
     "2023_20230705185546_ru3000_additions_final_20230623.pdf", True,
     "Preliminary date per official 2023 schedule press coverage (rank day April 28, prelim "
     "posted May 19, updates May 26/June 2/9/16, final after close June 23 -- note this year's "
     "effective date is NOT the last Friday of June (June 30) due to a schedule shift). Final "
     "source filename is explicitly 'final' and dated 20230623, matching the sourced effective "
     "date exactly."),
]

COLUMNS = ["year", "preliminary_posted", "effective_date", "final_source_file",
           "final_is_exact", "note"]


def get_calendar() -> pd.DataFrame:
    df = pd.DataFrame(CYCLES, columns=COLUMNS)
    df["preliminary_posted"] = pd.to_datetime(df["preliminary_posted"])
    df["effective_date"] = pd.to_datetime(df["effective_date"])
    return df


if __name__ == "__main__":
    df = get_calendar()
    pd.set_option("display.max_colwidth", 0)
    print(df[["year", "preliminary_posted", "effective_date", "final_is_exact"]])
