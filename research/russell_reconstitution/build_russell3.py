"""
Russell-3: Boundary-Crossing Subset. Family member of "Russell Reconstitution"
(established by Russell-1, trial #17) -- shares its DSR slot.

Reuses the existing Russell-1 sourced dataset (parsed_all_rows.csv, the
1,771-row additions universe, cycle_calendar.py's sourced dates, and the
already-fetched Tiingo price cache) -- no new data acquisition.

Step 1: build a deletions-side universe parallel to russell1_universe.csv
        (build_universe.py only ever built the additions side).
Step 2: combine additions + deletions, find every ticker appearing in 2+
        distinct years -- these are the "repeat crossers."
Step 3: emit one row per CROSSING EVENT (not per ticker) for those tickers.
"""

from pathlib import Path

import pandas as pd

from cycle_calendar import get_calendar
from deletions_calendar import get_deletions_calendar

RESULTS = Path(__file__).resolve().parent / "results"


def build_deletions_universe(parsed: pd.DataFrame) -> pd.DataFrame:
    del_cal = get_deletions_calendar()
    rows = []
    for _, cyc in del_cal.iterrows():
        src = cyc.final_deletions_source
        if src is None:
            print(f"  [deletions] {cyc.year}: NO r3000 deletions source available -- SKIPPED (gap)")
            continue
        sub = parsed[parsed.source_file == src]
        if sub.empty:
            print(f"  [deletions] {cyc.year}: WARNING source file {src} not found in parsed rows")
            continue
        for _, r in sub.iterrows():
            rows.append({
                "year": cyc.year,
                "ticker": r.symbol,
                "company": r.company,
                "sector": r.sector,
                "preliminary_posted": cyc.preliminary_posted,
                "effective_date": cyc.effective_date,
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["year", "ticker"])
    return df


def main():
    parsed = pd.read_csv(RESULTS / "parsed_all_rows.csv")
    add_universe = pd.read_csv(RESULTS / "russell1_universe.csv",
                                parse_dates=["preliminary_posted", "effective_date"])
    add_universe = add_universe.copy()
    add_universe["list_kind"] = "addition"

    print("Building deletions-side universe (years with r3000 deletions data available)...")
    del_universe = build_deletions_universe(parsed)
    del_universe["preliminary_posted"] = pd.to_datetime(del_universe["preliminary_posted"])
    del_universe["effective_date"] = pd.to_datetime(del_universe["effective_date"])
    del_universe["list_kind"] = "deletion"
    del_universe.to_csv(RESULTS / "russell1_deletions_universe.csv", index=False)
    print(f"Deletions universe: {len(del_universe)} (year, ticker) rows across "
          f"{sorted(del_universe.year.unique())}")

    combined = pd.concat([
        add_universe[["year", "ticker", "company", "sector", "preliminary_posted",
                       "effective_date", "list_kind"]],
        del_universe[["year", "ticker", "company", "sector", "preliminary_posted",
                       "effective_date", "list_kind"]],
    ], ignore_index=True)
    combined = combined.drop_duplicates(subset=["year", "ticker", "list_kind"])
    combined.to_csv(RESULTS / "russell3_combined_events.csv", index=False)

    # A ticker's "crossings" = distinct YEARS it appears in (as addition or
    # deletion, doesn't matter which). Repeat crosser = appears in 2+ years.
    years_per_ticker = combined.groupby("ticker")["year"].nunique()
    repeat_tickers = sorted(years_per_ticker[years_per_ticker >= 2].index)

    print(f"\nTotal unique tickers in combined additions+deletions universe: {combined['ticker'].nunique()}")
    print(f"Repeat crossers (appear in 2+ distinct years): {len(repeat_tickers)}")

    crossing_events = combined[combined.ticker.isin(repeat_tickers)].sort_values(["ticker", "year"])
    crossing_events.to_csv(RESULTS / "russell3_crossing_events.csv", index=False)

    print(f"Total crossing events for these tickers: {len(crossing_events)}")
    print(crossing_events.groupby("list_kind").size())
    print("\nPer-ticker year sequences:")
    for t in repeat_tickers:
        seq = crossing_events[crossing_events.ticker == t][["year", "list_kind"]].values.tolist()
        print(f"  {t:6s}: {seq}")

    return crossing_events


if __name__ == "__main__":
    main()
