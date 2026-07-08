"""
Phase 1: Data acquisition for Russell-1 (Pre-Effective-Date Anticipatory Drift).

FTSE Russell / Russell Investments does not host a historical archive of
past-year reconstitution addition/deletion lists on the live site (only the
current year is linked). There is no free API or academic dataset either
(the CRSP-based `pyndex` replication package requires paid WRDS access).

Primary-source PDFs that FTSE Russell/Russell Investments posted each year
(both weekly "preliminary" snapshots during the reconstitution window and the
"final" confirmed list) are however archived by the Wayback Machine and are
real, clean, machine-parseable tables (Company / Symbol / Industry) -- not
scanned images.

IMPORTANT: Wayback capture *timestamps* do not track the *content* year --
a page about the 2017 reconstitution may first get crawled in 2018, 2021, or
2026. So this script does NOT restrict CDX queries by capture-date window;
it pulls the full addition/deletion URL listing per domain (unrestricted)
and buckets each hit to a year using the year embedded in the URL slug or
filename itself (e.g. "final-r3000-additions-2017", "ru3000_additions_20210625").

Russell 3000 = Russell 1000 (union) Russell 2000. An addition to the R3000
list is therefore, by definition, an addition to either R1000 or R2000 -- so
these lists ARE the exact universe the hypothesis calls for (no need to
separately source Russell 1000-specific or Russell 2000-specific lists, and
no need for the Microcap lists for the primary universe).
"""

import json
import re
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 Archer Capital Research research@archercapital.test"}
CACHE = Path(__file__).resolve().parent / "cache"
RAW = CACHE / "raw_pdfs"
RAW.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2015, 2024))  # 2015-2023 inclusive (in-sample per user decision)

CDX = "https://web.archive.org/cdx/search/cdx"
SLEEP = 1.0


def _get(url, params=None, retries=5, timeout=60):
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            return r
        except Exception as e:
            wait = 3 * (i + 1)
            print(f"  [retry {i+1}/{retries}] {e} -- sleeping {wait}s")
            time.sleep(wait)
    return None


def _cdx(params):
    r = _get(CDX, params=params)
    if r is None or r.status_code != 200 or not r.text.strip():
        return []
    try:
        return json.loads(r.text)
    except json.JSONDecodeError:
        return []


YEAR_RE = re.compile(r"20(1[5-9]|2[0-9])")


def bucket_year(url: str) -> int:
    """Extract a year 2015-2029 from the URL slug/filename, preferring the
    LAST 4-digit year token (dated filenames put the year at the end,
    e.g. ru3000_additions_20210625.pdf -> 2021)."""
    matches = YEAR_RE.findall(url)
    if not matches:
        return None
    # findall with groups returns the group content; reconstruct full year
    full_years = [int("20" + m) for m in matches]
    return full_years[-1]


def find_all_candidates() -> list:
    rows = []
    for domain in ["ftserussell.com", "content.ftserussell.com"]:
        print(f"Querying CDX for domain={domain} (unrestricted, all-time) ...")
        data = _cdx({
            "url": domain,
            "matchType": "domain",
            "output": "json",
            "filter": [
                "statuscode:200",
                "mimetype:application/pdf",
                "urlkey:.*(ru3000|russell.?3000|r3000|rmicro|microcap).*(addition|deletion).*",
            ],
            "collapse": "digest",
            "limit": 2000,
        })
        print(f"  -> {max(len(data) - 1, 0)} hits")
        if data:
            rows.extend(data[1:])
        time.sleep(SLEEP)
    seen = set()
    uniq = []
    for row in rows:
        key = (row[1], row[2])
        if key not in seen:
            seen.add(key)
            uniq.append(row)
    return uniq


def download(timestamp: str, original: str, year) -> Path:
    safe_name = original.split("/")[-1].split("?")[0]
    out = RAW / f"{year}_{timestamp}_{safe_name}"
    if out.exists() and out.stat().st_size > 0:
        return out
    url = f"https://web.archive.org/web/{timestamp}id_/{original}"
    r = _get(url)
    time.sleep(SLEEP)
    if r is not None and r.status_code == 200 and r.content[:4] == b"%PDF":
        out.write_bytes(r.content)
        return out
    print(f"    FAILED download ({r.status_code if r else 'no-resp'}): {url}")
    return None


def main():
    candidates = find_all_candidates()
    print(f"\nTotal unique candidates across both domains: {len(candidates)}")

    manifest = []
    for row in candidates:
        _, timestamp, original, mimetype, statuscode, digest, length = row
        year = bucket_year(original)
        if year is None or year not in YEARS:
            continue
        path = download(timestamp, original, year)
        manifest.append({
            "year": year,
            "capture_timestamp": timestamp,
            "original_url": original,
            "digest": digest,
            "length": length,
            "local_path": str(path) if path else None,
        })
        print(f"  [{year}] {timestamp}  {original}  -> {'OK' if path else 'FAIL'}")

    import pandas as pd
    man_df = pd.DataFrame(manifest).sort_values(["year", "original_url"])
    man_df.to_csv(CACHE / "acquisition_manifest.csv", index=False)
    print(f"\nManifest written: {CACHE / 'acquisition_manifest.csv'} ({len(man_df)} rows)")
    print(man_df.groupby("year").size())


if __name__ == "__main__":
    main()
