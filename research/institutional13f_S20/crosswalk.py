"""
S20 Phase 1, take 2: free, no-quota-dependency CUSIP/ticker crosswalk via
issuer-name fuzzy matching against SEC's own company_tickers.json, instead of
the FMP CUSIP lookup that hit a hard account-wide quota wall.

No paid vendor, no rate limit -- SEC's own two free datasets cross-referenced
against each other:
  1. company_tickers.json  (ticker -> canonical company name, per SEC)
  2. 13F INFOTABLE.tsv      (issuer name, per 13F filer, as reported)
"""
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

CACHE = Path(__file__).resolve().parent / "cache"
TMP = Path("/tmp")

MATCH_THRESHOLD = 90.0

_SUFFIX_RE = re.compile(
    r"\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LLC|LP|L P|"
    r"PLC|HOLDINGS?|GROUP|TRUST|CLASS\s*[A-Z]|COM|NEW|SPONSORED|ADR|ADS)\b",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.upper()
    n = _PUNCT_RE.sub(" ", n)
    n = _SUFFIX_RE.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_universe_names(univ_csv: str, tickers_json: dict) -> pd.DataFrame:
    univ = pd.read_csv(univ_csv, dtype=str)
    by_ticker = {v["ticker"]: v["title"] for v in tickers_json.values()}
    by_cik = {int(v["cik_str"]): v["title"] for v in tickers_json.values()}

    rows = []
    for _, r in univ.iterrows():
        ticker, cik = r["ticker"], r["cik"]
        name = by_ticker.get(ticker)
        if name is None and cik and cik == cik:
            try:
                name = by_cik.get(int(float(cik)))
            except (ValueError, TypeError):
                name = None
        rows.append({"ticker": ticker, "cik": cik, "sec_name": name,
                      "norm_name": normalize_name(name) if name else None})
    return pd.DataFrame(rows)


def load_13f_accessions(zippaths: list[str], target_date: str) -> set:
    accs = set()
    for zp in zippaths:
        with zipfile.ZipFile(TMP / zp) as zf:
            with zf.open("SUBMISSION.tsv") as f:
                sub = pd.read_csv(f, sep="\t", dtype=str)
        sub["FILING_DATE"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
        sub = sub[sub["FILING_DATE"] <= pd.Timestamp(target_date)]
        sub = sub[sub["SUBMISSIONTYPE"].isin(["13F-HR", "13F-HR/A"])]
        accs.update(sub["ACCESSION_NUMBER"])
    return accs


def load_issuer_names(zippath: str, accessions: set) -> pd.Series:
    """Unique issuer names (NAMEOFISSUER) appearing in the given accessions."""
    names = set()
    with zipfile.ZipFile(TMP / zippath) as zf:
        with zf.open("INFOTABLE.tsv") as f:
            for chunk in pd.read_csv(f, sep="\t", dtype=str,
                                      usecols=["ACCESSION_NUMBER", "NAMEOFISSUER"],
                                      chunksize=500_000):
                sub = chunk[chunk["ACCESSION_NUMBER"].isin(accessions)]
                names.update(sub["NAMEOFISSUER"].dropna().unique())
    return pd.Series(sorted(names))


def fuzzy_match_coverage(universe_df: pd.DataFrame, issuer_names: pd.Series) -> pd.DataFrame:
    """For each universe name, find the best fuzzy match among 13F issuer names."""
    norm_issuers = issuer_names.map(normalize_name)
    issuer_lookup = dict(zip(norm_issuers, issuer_names))
    choices = [n for n in norm_issuers if n]

    results = []
    for _, row in universe_df.iterrows():
        if not row["norm_name"]:
            results.append({"ticker": row["ticker"], "sec_name": row["sec_name"],
                             "matched_issuer": None, "score": 0.0, "matched": False})
            continue
        best = process.extractOne(row["norm_name"], choices, scorer=fuzz.token_sort_ratio)
        if best is None:
            results.append({"ticker": row["ticker"], "sec_name": row["sec_name"],
                             "matched_issuer": None, "score": 0.0, "matched": False})
            continue
        match_norm, score, _ = best
        results.append({
            "ticker": row["ticker"], "sec_name": row["sec_name"],
            "matched_issuer": issuer_lookup.get(match_norm), "score": score,
            "matched": score >= MATCH_THRESHOLD,
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    tickers_json = json.load(open(CACHE / "company_tickers.json"))

    print("=== Loading universe names (SEC canonical titles) ===")
    univ_2015 = load_universe_names("/tmp/univ_2015.csv", tickers_json)
    univ_2020 = load_universe_names("/tmp/univ_2020.csv", tickers_json)
    print(f"2015 universe: {len(univ_2015)} names, {univ_2015['sec_name'].notna().sum()} resolved from company_tickers.json")
    print(f"2020 universe: {len(univ_2020)} names, {univ_2020['sec_name'].notna().sum()} resolved from company_tickers.json")

    print("\n=== Loading 13F accessions (filing-date gated) ===")
    acc_2015 = load_13f_accessions(["2014q4_form13f.zip", "2015q1_form13f.zip"], "2015-01-01")
    acc_2020 = load_13f_accessions(["2019q4_form13f.zip", "2020q1_form13f.zip"], "2020-01-01")
    print(f"2015: {len(acc_2015)} accessions with FILING_DATE<=2015-01-01")
    print(f"2020: {len(acc_2020)} accessions with FILING_DATE<=2020-01-01")

    print("\n=== Loading unique issuer names from INFOTABLE ===")
    issuers_2015 = load_issuer_names("2014q4_form13f.zip", acc_2015)
    issuers_2020 = load_issuer_names("2019q4_form13f.zip", acc_2020)
    print(f"2015: {len(issuers_2015)} unique issuer names held")
    print(f"2020: {len(issuers_2020)} unique issuer names held")

    print("\n=== Fuzzy matching (this may take a couple minutes) ===")
    match_2015 = fuzzy_match_coverage(univ_2015, issuers_2015)
    match_2020 = fuzzy_match_coverage(univ_2020, issuers_2020)

    match_2015.to_csv(CACHE / "match_2015.csv", index=False)
    match_2020.to_csv(CACHE / "match_2020.csv", index=False)

    for label, m, univ in [("2015", match_2015, univ_2015), ("2020", match_2020, univ_2020)]:
        n_matched = int(m["matched"].sum())
        print(f"\n{label}: {n_matched}/{len(m)} universe names matched >= {MATCH_THRESHOLD} "
              f"({n_matched/len(m):.1%}) against {len(m)} names with a resolvable SEC title")
