"""
Phase 2: Parse raw FTSE Russell reconstitution PDFs into clean per-year
ticker tables, and pick the preliminary / final candidate per year.

Format (consistent 2016-2023): a "Company | Symbol | Sector" table, one
company per row, repeated across pages, with a "Russell US Indexes /
Reconstitution / Russell 3000(R) Index - Additions" (or "- Deletions",
or "Russell Microcap(R) - Additions") banner + legal-disclaimer footer on
every page. Sector is drawn from a small fixed vocabulary (Russell's own
9-sector scheme pre-~2020, ICB-style thereafter) -- matching on that fixed
vocabulary as a line SUFFIX is far more reliable than a generic regex,
since company names are themselves in all-caps and can contain tokens that
look like tickers (e.g. "ACNB CORP  ACNB  Financial Services").
"""

import re
from pathlib import Path

import pandas as pd
import pdfplumber

CACHE = Path(__file__).resolve().parent / "cache"
RAW = CACHE / "raw_pdfs"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Fixed sector vocabulary observed across 2016-2023 (Russell's own scheme,
# then ICB-style after the ~2020 taxonomy change). Longest-first so
# "Materials & Processing" isn't shadowed by a shorter partial match.
KNOWN_SECTORS = sorted([
    "Health Care", "Technology", "Consumer Discretionary", "Industrials",
    "Financials", "Financial Services", "Consumer Staples",
    "Producer Durables", "Energy", "Materials & Processing",
    "Basic Materials", "Telecommunications", "Real Estate", "Utilities",
], key=len, reverse=True)

SYMBOL_RE = re.compile(r"^[A-Z]{1,6}(\.[A-Z]{1,2})?$")

NOISE_SUBSTRINGS = [
    "An LSEG Business", "London Stock Exchange", "Frank Russell", "FTSE International",
    "Global Debt Capital", "licensors", "undertake", "authorised", "reliable indicator",
    "For more information", "please visit", "Russell US Indexes", "Reconstitution",
    "Company Symbol", "Russell 3000", "Russell Microcap", "Russell 1000", "Russell 2000",
]


def parse_pdf(path: Path) -> pd.DataFrame:
    """Extract Company/Symbol/Sector rows from one addition/deletion PDF."""
    rows = []
    list_kind = None
    index_kind = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if re.search(r"\bAdditions\b", text):
                list_kind = "additions"
            elif re.search(r"\bDeletions\b", text):
                list_kind = "deletions"
            if "Microcap" in text:
                index_kind = "rmicro"
            elif "3000" in text:
                index_kind = "r3000"

            for raw_line in text.split("\n"):
                line = raw_line.strip()
                if not line or any(n in line for n in NOISE_SUBSTRINGS):
                    continue
                if line.startswith("©"):
                    continue

                matched_sector = None
                for sector in KNOWN_SECTORS:
                    if line.endswith(sector):
                        matched_sector = sector
                        remainder = line[: -len(sector)].strip()
                        break
                if matched_sector is None:
                    continue

                tokens = remainder.split()
                if len(tokens) < 2:
                    continue
                symbol = tokens[-1]
                if not SYMBOL_RE.match(symbol):
                    continue
                company = " ".join(tokens[:-1])
                rows.append({
                    "company": company,
                    "symbol": symbol,
                    "sector": matched_sector,
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol", "company"])
    if not df.empty:
        df["list_kind"] = list_kind
        df["index_kind"] = index_kind
    return df


def main():
    manifest_rows = []
    all_parsed = []
    for path in sorted(RAW.glob("*")):
        try:
            df = parse_pdf(path)
        except Exception as e:
            print(f"FAILED to parse {path.name}: {e}")
            continue
        n = len(df)
        kind = df["list_kind"].iloc[0] if n else None
        idx = df["index_kind"].iloc[0] if n else None
        print(f"{path.name:70s} n={n:4d}  list={kind}  index={idx}")
        if n:
            df["source_file"] = path.name
            all_parsed.append(df)
        manifest_rows.append({"file": path.name, "n_rows": n, "list_kind": kind, "index_kind": idx})

    if all_parsed:
        full = pd.concat(all_parsed, ignore_index=True)
        full.to_csv(RESULTS / "parsed_all_rows.csv", index=False)
        print(f"\nWrote {len(full)} total parsed rows -> {RESULTS / 'parsed_all_rows.csv'}")

    pd.DataFrame(manifest_rows).to_csv(RESULTS / "parse_manifest.csv", index=False)


if __name__ == "__main__":
    main()
