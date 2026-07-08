"""
Phase 2c: Combine the cycle calendar with parsed PDF tables into the
master Russell-1 addition universe: one row per (year, ticker).
"""

from pathlib import Path

import pandas as pd

from cycle_calendar import get_calendar

RESULTS = Path(__file__).resolve().parent / "results"


def build():
    parsed = pd.read_csv(RESULTS / "parsed_all_rows.csv")
    cal = get_calendar()

    rows = []
    for _, cyc in cal.iterrows():
        sub = parsed[parsed.source_file == cyc.final_source_file]
        if sub.empty:
            print(f"WARNING: no parsed rows for {cyc.year} source {cyc.final_source_file}")
            continue
        for _, r in sub.iterrows():
            rows.append({
                "year": cyc.year,
                "ticker": r.symbol,
                "company": r.company,
                "sector": r.sector,
                "preliminary_posted": cyc.preliminary_posted,
                "effective_date": cyc.effective_date,
                "final_is_exact": cyc.final_is_exact,
            })

    universe = pd.DataFrame(rows).drop_duplicates(subset=["year", "ticker"])
    universe.to_csv(RESULTS / "russell1_universe.csv", index=False)

    print(f"Total (year, ticker) rows: {len(universe)}")
    print(universe.groupby("year").size())
    print(f"\nUnique tickers across all cycles: {universe['ticker'].nunique()}")
    print(f"Wrote -> {RESULTS / 'russell1_universe.csv'}")


if __name__ == "__main__":
    build()
