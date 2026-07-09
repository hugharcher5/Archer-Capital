"""
Deletions-side companion to cycle_calendar.py, needed for Russell-3
(boundary-crossing subset): a repeat-crosser can flip in via an addition in
one year and flip out via a deletion in another, so we need a "confirmed
final deletions list" per year the same way Russell-1 needed one for
additions.

preliminary_posted / effective_date are IDENTICAL to cycle_calendar.py (same
reconstitution cycle, same two dates apply to both the additions and
deletions lists posted that day) -- only final_source_file differs.

GAPS: no Russell 3000 deletions PDF was recoverable at all for 2018 or 2019
in the original Russell-1 acquisition (acquire.py's CDX pull only found
Russell Microcap deletions for 2018, and nothing for 2019). Per the
prerequisite ("reuse the existing sourced dataset ... do not re-source
data"), these two years are marked unavailable rather than re-fetched --
repeat-crosser detection for 2018/2019 deletions is therefore incomplete;
any ticker whose only "second leg" of a crossing would have been an
undetected 2018 or 2019 deletion will be missed. Flagged in the results.
"""

import pandas as pd

from cycle_calendar import get_calendar

# year -> final_source_file (last/closest-to-effective-date deletions capture
# available), or None if genuinely unavailable in the archive.
FINAL_DELETIONS_SOURCE = {
    2016: "2016_20160630015402_russell-3000-index-2016-deletions",
    2017: "2017_20180201202107_final-r3000-deletions-2017",  # explicitly "final" in slug
    2018: None,  # GAP -- no r3000 deletions PDF recovered for 2018
    2019: None,  # GAP -- no r3000 deletions PDF recovered for 2019
    2020: "2020_20200811174923_ru3000_deletions_20200619.pdf",  # only capture; 1 week before effective (06-26)
    2021: "2021_20210626025026_ru3000_deletions_20210625.pdf",  # dated exactly the effective date
    2022: "2022_20220625012025_ru3000_deletions_20220624.pdf",  # dated exactly the effective date
    2023: "2023_20230621220003_ru3000_deletions_20230616.pdf",  # closest available; 1 week before effective (06-23), no explicit "final" capture exists
}


def get_deletions_calendar() -> pd.DataFrame:
    cal = get_calendar()
    cal = cal.copy()
    cal["final_deletions_source"] = cal["year"].map(FINAL_DELETIONS_SOURCE)
    return cal[["year", "preliminary_posted", "effective_date", "final_deletions_source"]]


if __name__ == "__main__":
    print(get_deletions_calendar())
