"""
Russell 3000 / Russell Microcap reconstitution data — Wayback Machine sourcing
================================================================================
Sources historical Russell U.S. index reconstitution addition/deletion lists
(2015-2023) from archived FTSE Russell / Russell Investments pages via the
Internet Archive's Wayback Machine CDX API. See README.md in this directory
for the full narrative of how this data source was discovered and why it's
necessary (the live FTSE Russell site only hosts current-year documents).

This script does NOT run any backtest — it only builds the structured
dataset at research/russell/data/. Backtest logic (Russell-1, Russell-2)
reads from that dataset and never touches the network.

Two source formats, both real and confirmed by direct fetch + parse:
  1. PDF   (2016-2023): tables with columns [Company, Symbol, Sector],
     hosted on ftserussell.com / content.ftserussell.com, parsed with
     pdfplumber.
  2. HTML  (2015 only): a live "additions-deletions.page" tool with
     <table id="Russell3000AddTable"> etc., parsed with pandas.read_html.
     No PDF equivalent was found for 2015 — the two documents that LOOK
     like ticker lists by filename ("2015-reconstitution.pdf" and
     "preliminary-list-2015-reconstitution.pdf") are actually a narrative
     "highlights" doc and a press release with no ticker table at all.
     Confirmed by direct extraction — see README.md.

For each year, we need up to 4 documents: {R3000, RMICRO} x {ADD, DELETE},
at up to 2 points in time: PRELIMINARY (near the announcement) and FINAL
(near the effective date, last Friday of June). Not every slot has a
confirmed distinct preliminary snapshot for every year — see the
PROXY_FLAGS this script emits and the coverage report in README.md.
Where no genuine preliminary snapshot exists, the FINAL list is used as a
proxy and flagged `is_proxy=True` in the output — never fabricated.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
import pdfplumber
import requests

HERE     = Path(__file__).resolve().parent
CACHE    = HERE / "cache"
DATA_OUT = HERE / "data"
CACHE.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

CDX_API   = "http://web.archive.org/cdx/search/cdx"
WEB_ARCHIVE = "https://web.archive.org/web"
_UA = {"User-Agent": "research-archer-capital/1.0 (historical index data study)"}

_LAST_REQ: float = 0.0


def _throttle(min_interval: float = 0.5) -> None:
    global _LAST_REQ
    wait = min_interval - (time.time() - _LAST_REQ)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQ = time.time()


# =============================================================================
# Historical effective dates (last Friday of June) — verified against
# filenames where the source itself embeds the date (2015: "...20150626",
# 2019: "RU3000_Additions_20190628", 2023: "...final20230623"). All others
# computed via the "last Friday of June" rule and cross-checked to be
# internally consistent with the weekly-snapshot filenames found (see
# README.md coverage table).
# =============================================================================

EFFECTIVE_DATES = {
    2015: date(2015, 6, 26),
    2016: date(2016, 6, 24),
    2017: date(2017, 6, 23),
    2018: date(2018, 6, 22),
    2019: date(2019, 6, 28),
    2020: date(2020, 6, 26),
    2021: date(2021, 6, 25),
    2022: date(2022, 6, 24),
    2023: date(2023, 6, 23),
}

# Documented preliminary-announcement dates where an actual press release
# / source document confirms the date. Where absent, the earliest
# confirmed preliminary DATA snapshot date is used as the effective proxy
# announcement date instead (see per-year notes in the coverage report).
DOCUMENTED_ANNOUNCEMENT_DATES = {
    2015: date(2015, 6, 12),   # FTSE Russell press release, confirmed by direct fetch
}


@dataclass
class SourceDoc:
    year: int
    index: str        # "R3000" | "RMICRO"
    direction: str     # "ADD" | "DELETE"
    timing: str        # "PRELIM" | "FINAL"
    url: str
    snapshot_ts: str
    fmt: str           # "pdf" | "html"
    is_proxy: bool = False   # True if this doc is standing in for a missing slot


# =============================================================================
# CDX helpers
# =============================================================================

def cdx_lookup(url: str) -> list[dict]:
    """Query CDX for all snapshots of an exact URL. Returns list of dicts."""
    _throttle()
    try:
        r = requests.get(CDX_API, params={"url": url, "output": "json"}, headers=_UA, timeout=30)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        print(f"  [CDX ERROR] {url}: {e}")
        return []
    if len(rows) < 2:
        return []
    header, *data = rows
    return [dict(zip(header, row)) for row in data]


def cdx_search_domain(domain: str, url_filter: str, limit: int = 300) -> list[dict]:
    """Broader CDX search with a regex filter on urlkey, collapsed to unique URLs."""
    _throttle()
    params = {
        "url": domain, "matchType": "domain",
        "filter": f"urlkey:{url_filter}",
        "collapse": "urlkey", "limit": limit, "output": "json",
    }
    try:
        r = requests.get(CDX_API, params=params, headers=_UA, timeout=30)
        r.raise_for_status()
        rows = r.json()
    except Exception as e:
        print(f"  [CDX ERROR] domain={domain} filter={url_filter}: {e}")
        return []
    if len(rows) < 2:
        return []
    header, *data = rows
    return [dict(zip(header, row)) for row in data]


def best_pdf_snapshot(candidates: list[dict], min_length: int = 5000) -> Optional[dict]:
    """Pick the best real-PDF snapshot from a CDX result list (status 200, application/pdf, non-trivial size)."""
    ok = [c for c in candidates
          if c.get("statuscode") == "200"
          and c.get("mimetype") == "application/pdf"
          and int(c.get("length", 0) or 0) >= min_length]
    if not ok:
        return None
    ok.sort(key=lambda c: c["timestamp"])
    return ok[0]


def fetch_snapshot_bytes(original_url: str, timestamp: str) -> bytes:
    wb_url = f"{WEB_ARCHIVE}/{timestamp}/{original_url}"
    _throttle()
    r = requests.get(wb_url, headers=_UA, timeout=60)
    r.raise_for_status()
    return r.content


# =============================================================================
# Candidate URL generation per (year, index, direction, timing)
# =============================================================================
# Each slot gets a priority-ordered list of exact-URL guesses (based on the
# real naming conventions discovered across the site's several redesigns),
# tried in order via CDX; first confirmed application/pdf hit wins.

def _candidates(year: int, index: str, direction: str, timing: str) -> list[str]:
    y = year
    idx_slug_lower  = {"R3000": "russell-3000", "RMICRO": "russell-microcap"}[index]
    idx_tag_short   = {"R3000": "ru3000", "RMICRO": "rmicro"}[index]
    dir_slug        = {"ADD": "additions", "DELETE": "deletions"}[direction]

    out: list[str] = []

    # 2018-2019 Drupal "support-document(s)" slug patterns
    timing_slug = "preliminary" if timing == "PRELIM" else "final"
    out += [
        f"https://www.ftserussell.com/files/support-document/{y}-{timing_slug}-{idx_slug_lower}-index-{dir_slug}",
        f"http://www.ftserussell.com/files/support-documents/{y}-{timing_slug}-{idx_slug_lower}-index-{dir_slug}",
    ]

    # 2016-2017 "russell-3000-index-YYYY-additions" / "final-r3000-additions-YYYY" patterns
    out += [
        f"http://www.ftserussell.com/files/support-documents/{idx_slug_lower}-index-{y}-{dir_slug}",
        f"http://www.ftserussell.com/files/support-documents/final-{idx_tag_short}-{dir_slug}-{y}",
        f"https://www.ftserussell.com/files/support-documents/preliminary-{idx_tag_short}-{dir_slug}-{y}",
    ]

    # 2020-2023 weekly-dated content.ftserussell.com direct files
    eff = EFFECTIVE_DATES[y]
    if timing == "FINAL":
        out += [
            f"https://content.ftserussell.com/sites/default/files/{idx_tag_short}_{dir_slug}_final_{eff:%Y%m%d}.pdf",
            f"https://content.ftserussell.com/sites/default/files/{idx_tag_short}_{dir_slug}_{eff:%Y%m%d}.pdf",
            f"https://content.ftserussell.com/sites/default/files/{idx_tag_short}_{dir_slug}_{eff:%Y%m%d}_0.pdf",
        ]

    # 2019 explicit RU3000_Additions_YYYYMMDD (support_document, capitalized)
    out += [
        f"https://content.ftserussell.com/sites/default/files/support_document/{'RU3000' if index=='R3000' else 'RUMICRO'}_{dir_slug.capitalize()}_{eff:%Y%m%d}.pdf",
    ]

    # 2021 slug variant with dash
    out += [
        f"https://www.ftserussell.com/files/support-document/{idx_slug_lower}-index-{dir_slug}-{y}",
        f"https://www.ftserussell.com/files/support-document/{idx_slug_lower}-{dir_slug}-{y}",
        f"https://www.ftserussell.com/files/support-document/{idx_slug_lower}-{y}-{timing_slug}-index-{dir_slug}",
        f"https://www.ftserussell.com/files/support-document/{idx_slug_lower}-{y}-{timing_slug}-{dir_slug}",
        f"https://www.ftserussell.com/files/support-document/{idx_slug_lower}-{y}-{timing_slug}-{dir_slug}-0",
    ]

    return out


def discover_weekly_dated_pdfs(year: int, index: str, direction: str) -> list[dict]:
    """
    Broad CDX search for the YYYYMMDD-dated content.ftserussell.com files
    (2020-2023 weekly-cadence pattern). Returns ALL confirmed dated
    snapshots for that (year, index, direction), sorted chronologically —
    caller picks earliest (preliminary) / latest-<=-effective (final).
    """
    idx_tag_short = {"R3000": "ru3000", "RMICRO": "rmicro"}[index]
    dir_slug      = {"ADD": "additions", "DELETE": "deletions"}[direction]
    pattern = rf".*{idx_tag_short}_{dir_slug}_{year}.*|.*{idx_tag_short}_{dir_slug}_2\d{{7}}.*"
    rows = cdx_search_domain(
        "content.ftserussell.com/sites/default/files/",
        pattern, limit=200,
    )
    # keep only rows whose embedded date matches this calendar year
    out = []
    for r in rows:
        orig = r.get("original", "")
        if f"_{year}" not in orig and not any(str(year) in seg for seg in orig.split("_")):
            continue
        out.append(r)
    return out


# =============================================================================
# Slot resolution: find the actual usable document for one (year, index, direction, timing)
# =============================================================================

def resolve_slot(year: int, index: str, direction: str, timing: str) -> Optional[SourceDoc]:
    # 2015 handled entirely separately (HTML page, not PDF) — see resolve_2015()
    if year == 2015:
        return None

    # Try the weekly-dated discovery first for 2020-2023 (richest, most reliable)
    if year >= 2020:
        weekly = discover_weekly_dated_pdfs(year, index, direction)
        weekly_ok = [w for w in weekly
                     if w.get("statuscode") == "200" and w.get("mimetype") == "application/pdf"]
        if weekly_ok:
            weekly_ok.sort(key=lambda w: w["timestamp"])
            eff = EFFECTIVE_DATES[year]
            eff_str = eff.strftime("%Y%m%d")
            if timing == "PRELIM":
                chosen = weekly_ok[0]
            else:
                # final = snapshot whose embedded date is closest to (<=) effective date
                dated = [(w, w["original"]) for w in weekly_ok]
                on_or_before_eff = [w for w, orig in dated if _extract_yyyymmdd(orig) and _extract_yyyymmdd(orig) <= eff_str]
                chosen = (on_or_before_eff[-1] if on_or_before_eff else weekly_ok[-1])
            return SourceDoc(year, index, direction, timing, chosen["original"], chosen["timestamp"], "pdf")

    # Fall back to guessed exact-URL candidates, checked via CDX
    for cand in _candidates(year, index, direction, timing):
        rows = cdx_lookup(cand)
        best = best_pdf_snapshot(rows)
        if best:
            return SourceDoc(year, index, direction, timing, best["original"], best["timestamp"], "pdf")

    return None


def _extract_yyyymmdd(s: str) -> Optional[str]:
    import re
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", s)
    return m.group(0) if m else None


# =============================================================================
# 2015: HTML page method
# =============================================================================

_2015_TABLE_IDS = {
    ("R3000", "ADD"):    "Russell3000AddTable",
    ("R3000", "DELETE"): "Russell3000DeleteTable",
    ("RMICRO", "ADD"):    "RussellMicrocapAddTable",
    ("RMICRO", "DELETE"): "RussellMicrocapDeleteTable",
}

_2015_PAGE_URL = "http://www.russell.com:80/indexes/americas/tools-resources/reconstitution/additions-deletions.page"


def resolve_2015_final_snapshot() -> Optional[dict]:
    rows = cdx_lookup(_2015_PAGE_URL)
    ok = [r for r in rows if r.get("statuscode") == "200"]
    if not ok:
        return None
    ok.sort(key=lambda r: r["timestamp"])
    eff_str = EFFECTIVE_DATES[2015].strftime("%Y%m%d")
    on_or_after = [r for r in ok if r["timestamp"][:8] >= eff_str]
    return on_or_after[0] if on_or_after else ok[-1]


def parse_2015_html_table(html_bytes: bytes, index: str, direction: str) -> pd.DataFrame:
    table_id = _2015_TABLE_IDS[(index, direction)]
    tables = pd.read_html(io.BytesIO(html_bytes), attrs={"id": table_id})
    df = tables[0]
    df.columns = [c.strip() for c in df.columns]
    return df


# =============================================================================
# PDF parsing
# =============================================================================

def parse_pdf_table(pdf_bytes: bytes) -> pd.DataFrame:
    """
    Parse the standard [Company, Symbol, Sector] table repeated across pages.
    Header row ('Company','Symbol','Sector') repeats each page — dropped.
    """
    records: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                for row in t:
                    if row is None or len(row) < 2:
                        continue
                    cells = [c.strip() if isinstance(c, str) else c for c in row]
                    if not cells or cells[0] in (None, "", "Company"):
                        continue
                    company = cells[0]
                    symbol  = cells[1] if len(cells) > 1 else None
                    sector  = cells[2] if len(cells) > 2 else None
                    if not company or not symbol:
                        continue
                    records.append({"Company": company, "Symbol": symbol, "Sector": sector})
    return pd.DataFrame(records)


# =============================================================================
# Orchestration
# =============================================================================

def fetch_and_cache(doc: SourceDoc) -> Optional[bytes]:
    safe_name = f"{doc.year}_{doc.index}_{doc.direction}_{doc.timing}_{doc.snapshot_ts}.{doc.fmt}"
    cache_path = CACHE / safe_name
    if cache_path.exists():
        return cache_path.read_bytes()
    try:
        content = fetch_snapshot_bytes(doc.url, doc.snapshot_ts)
    except Exception as e:
        print(f"  [FETCH ERROR] {doc.url} @ {doc.snapshot_ts}: {e}")
        return None
    cache_path.write_bytes(content)
    return content


def build_dataset(years: list[int]) -> tuple[pd.DataFrame, list[dict]]:
    all_rows: list[dict] = []
    coverage: list[dict] = []

    for year in years:
        eff_date = EFFECTIVE_DATES[year]
        print(f"\n{'='*70}\nYEAR {year}  (effective date: {eff_date})\n{'='*70}")

        if year == 2015:
            snap = resolve_2015_final_snapshot()
            if snap is None:
                print("  [FAIL] No usable 2015 snapshot found for additions-deletions.page.")
                coverage.append({"year": year, "slot": "ALL", "status": "MISSING"})
                continue
            content = fetch_snapshot_bytes(snap["original"], snap["timestamp"])
            cache_path = CACHE / f"2015_page_{snap['timestamp']}.html"
            cache_path.write_bytes(content)

            for index in ("R3000", "RMICRO"):
                for direction in ("ADD", "DELETE"):
                    timing = "FINAL"
                    try:
                        df = parse_2015_html_table(content, index, direction)
                    except Exception as e:
                        print(f"  [PARSE FAIL] 2015 {index} {direction}: {e}")
                        coverage.append({"year": year, "slot": f"{index}-{direction}-{timing}", "status": "PARSE_FAILED"})
                        continue
                    n = len(df)
                    print(f"  2015 {index:6s} {direction:6s} {timing:6s}: {n} rows (HTML page, snapshot {snap['timestamp']})")
                    for _, row in df.iterrows():
                        all_rows.append({
                            "year": year, "index": index, "direction": direction,
                            "timing": timing, "is_proxy": False,
                            "company": row.get("Company"), "ticker": row.get("Symbol"),
                            "sector": row.get("Sector"),
                            "effective_date": eff_date.isoformat(),
                            "preliminary_announcement_date": DOCUMENTED_ANNOUNCEMENT_DATES.get(2015, ""),
                            "source_url": snap["original"], "snapshot_ts": snap["timestamp"],
                        })
                    coverage.append({"year": year, "slot": f"{index}-{direction}-{timing}",
                                      "status": "OK", "n_rows": n, "source": snap["original"]})
                    # No genuine preliminary snapshot found for 2015 — see README.
                    coverage.append({"year": year, "slot": f"{index}-{direction}-PRELIM",
                                      "status": "MISSING_NO_PROXY_APPLIED",
                                      "note": "No archived snapshot exists in the ~2wk window between the "
                                              "documented 2015-06-12 announcement and the 2015-06-26 effective "
                                              "date; Wayback's crawl density for this page did not cover it."})
            continue

        # 2016-2023: PDF slots
        for index in ("R3000", "RMICRO"):
            for direction in ("ADD", "DELETE"):
                slots_found: dict[str, SourceDoc] = {}
                for timing in ("PRELIM", "FINAL"):
                    doc = resolve_slot(year, index, direction, timing)
                    if doc is not None:
                        slots_found[timing] = doc

                if "FINAL" not in slots_found:
                    print(f"  [MISSING] {year} {index} {direction} FINAL — no candidate resolved.")
                    coverage.append({"year": year, "slot": f"{index}-{direction}-FINAL", "status": "MISSING"})
                if "PRELIM" not in slots_found and "FINAL" in slots_found:
                    print(f"  [PROXY] {year} {index} {direction} PRELIM missing — using FINAL as proxy.")
                    slots_found["PRELIM"] = SourceDoc(
                        year, index, direction, "PRELIM",
                        slots_found["FINAL"].url, slots_found["FINAL"].snapshot_ts,
                        slots_found["FINAL"].fmt, is_proxy=True,
                    )
                    coverage.append({"year": year, "slot": f"{index}-{direction}-PRELIM",
                                      "status": "PROXY_FROM_FINAL"})

                for timing, doc in slots_found.items():
                    content = fetch_and_cache(doc)
                    if content is None:
                        coverage.append({"year": year, "slot": f"{index}-{direction}-{timing}", "status": "FETCH_FAILED"})
                        continue
                    try:
                        df = parse_pdf_table(content)
                    except Exception as e:
                        print(f"  [PARSE FAIL] {year} {index} {direction} {timing}: {e}")
                        coverage.append({"year": year, "slot": f"{index}-{direction}-{timing}", "status": "PARSE_FAILED"})
                        continue
                    n = len(df)
                    proxy_tag = " [PROXY]" if doc.is_proxy else ""
                    print(f"  {year} {index:6s} {direction:6s} {timing:6s}: {n:>4} rows{proxy_tag}  "
                          f"({doc.url}  @{doc.snapshot_ts})")
                    for _, row in df.iterrows():
                        all_rows.append({
                            "year": year, "index": index, "direction": direction,
                            "timing": timing, "is_proxy": doc.is_proxy,
                            "company": row.get("Company"), "ticker": row.get("Symbol"),
                            "sector": row.get("Sector"),
                            "effective_date": eff_date.isoformat(),
                            "preliminary_announcement_date": DOCUMENTED_ANNOUNCEMENT_DATES.get(year, ""),
                            "source_url": doc.url, "snapshot_ts": doc.snapshot_ts,
                        })
                    coverage.append({"year": year, "slot": f"{index}-{direction}-{timing}",
                                      "status": "OK", "n_rows": n, "source": doc.url,
                                      "is_proxy": doc.is_proxy})

    return pd.DataFrame(all_rows), coverage


def main() -> None:
    years = list(range(2015, 2024))
    df, coverage = build_dataset(years)

    out_csv = DATA_OUT / "russell_reconstitution_2015_2023.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[Saved] {out_csv}  ({len(df):,} rows)")

    cov_df = pd.DataFrame(coverage)
    cov_csv = DATA_OUT / "coverage_report.csv"
    cov_df.to_csv(cov_csv, index=False)
    print(f"[Saved] {cov_csv}  ({len(cov_df):,} rows)")

    # ── Sanity summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}\nSANITY SUMMARY\n{'='*70}")
    for year in years:
        yd = df[df["year"] == year]
        if yd.empty:
            print(f"  {year}: NO DATA")
            continue
        pivot = yd.groupby(["index", "direction", "timing"]).size().to_dict()
        proxy_flags = yd[yd["is_proxy"]][["index", "direction"]].drop_duplicates()
        proxy_str = (", ".join(f"{r['index']}-{r['direction']}" for _, r in proxy_flags.iterrows())
                     if not proxy_flags.empty else "none")
        print(f"  {year}: {dict(pivot)}  | proxied slots: {proxy_str}")


if __name__ == "__main__":
    main()
