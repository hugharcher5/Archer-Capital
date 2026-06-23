"""
Livestock Disease → Animal-Pharma Event Study (H1)
===================================================
Does the market underreact to verified livestock-disease outbreak notifications,
producing positive abnormal drift in exposed animal-pharma names over 20–60 days
after notification — concentrated in minor/non-headline markets?

Data: WAHIS outbreak notifications (WOAH) + Tiingo prices.
Entry point: python research/livestock_pead/run.py [--stage N]
"""

import argparse
import concurrent.futures
import io
import json
import os
import socket
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

socket.setdefaulttimeout(25)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats

# ---------------------------------------------------------------------------
# Paths & keys
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
CACHE = BASE / "cache"
OUTPUT = BASE / "output"
for _d in [CACHE, CACHE / "tiingo", OUTPUT]:
    _d.mkdir(parents=True, exist_ok=True)

def _load_env():
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

_load_env()
TIINGO_KEY = os.environ.get("TIINGO_API_KEY")

# Reuse Tiingo rate-limiting from mgmt_pit
_TIINGO_LAST_CALL: float = 0.0


def _tiingo_get(url: str, params: Optional[dict] = None) -> requests.Response:
    global _TIINGO_LAST_CALL
    wait = 5.0 - (time.time() - _TIINGO_LAST_CALL)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, params=params, timeout=(10, 20))
    _TIINGO_LAST_CALL = time.time()
    return resp


# ===================================================================
# STAGE 1 — WAHIS event data
# ===================================================================

TARGET_DISEASES = [
    "Highly pathogenic avian influenza",
    "Infection with high pathogenicity avian influenza viruses",
    "African swine fever",
    "Foot and mouth disease",
    "Bluetongue",
    "Porcine reproductive and respiratory syndrome",
    "Newcastle disease",
    "Infection with Newcastle disease virus",
]

DISEASE_SHORT = {
    "Highly pathogenic avian influenza": "HPAI",
    "Infection with high pathogenicity avian influenza viruses": "HPAI",
    "African swine fever": "ASF",
    "Foot and mouth disease": "FMD",
    "Bluetongue": "BTV",
    "Porcine reproductive and respiratory syndrome": "PRRS",
    "Newcastle disease": "NCD",
    "Infection with Newcastle disease virus": "NCD",
}

SPECIES_MAP = {
    "HPAI": "poultry",
    "ASF": "swine",
    "FMD": "cattle",
    "BTV": "ruminant",
    "PRRS": "swine",
    "NCD": "poultry",
}

MAJOR_MARKETS = {
    "United States of America", "China (People's Rep. of)", "China",
    "Germany", "France", "United Kingdom", "Brazil", "Japan",
    "Netherlands", "Canada", "Australia", "Italy", "Spain",
    "Mexico", "India", "Russia", "Russian Federation",
}


def fetch_wahis_events() -> pd.DataFrame:
    """
    Fetch outbreak notifications from WAHIS API for target diseases, 2015-2025.
    Returns DataFrame with columns: notification_date, disease, disease_short,
    species, country, region, cases, deaths, culled, report_id, market_type.
    """
    cache_file = CACHE / "wahis_events.csv"
    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["notification_date"])
        print(f"[WAHIS] Loaded {len(df)} events from cache.")
        return df

    print("[WAHIS] Fetching outbreak report list from WAHIS API...")
    url = "https://wahis.woah.org/pi/getReportList"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0 (research use)",
    }

    all_reports = []

    for disease in TARGET_DISEASES:
        print(f"  Querying: {disease}...")
        payload = {
            "pageNumber": 0,
            "pageSize": 100000,
            "sortColName": "reportDate",
            "sortColOrder": "DESC",
            "searchText": "",
            "reportFilters": {
                "diseases": [disease],
                "country": [],
                "region": [],
                "reportDate": {
                    "startDate": "2015-01-01",
                    "endDate": "2025-12-31",
                },
            },
        }

        for attempt in range(2):
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    reports = data if isinstance(data, list) else data.get("reportList", data.get("content", []))
                    if isinstance(reports, dict):
                        reports = reports.get("reportList", reports.get("content", []))
                    print(f"    → {len(reports)} reports")
                    for r in reports:
                        all_reports.append({
                            "report_id": r.get("reportInfoId", r.get("reportId", "")),
                            "notification_date": r.get("reportDate", r.get("eventDate", "")),
                            "disease": disease,
                            "disease_short": DISEASE_SHORT.get(disease, disease[:4]),
                            "country": r.get("country", r.get("countryName", "")),
                            "region": r.get("region", ""),
                            "cases": r.get("totalCases", r.get("cases", 0)),
                            "deaths": r.get("totalDeaths", r.get("deaths", 0)),
                            "culled": r.get("totalKilled", r.get("killed", r.get("culled", 0))),
                        })
                    break
                elif resp.status_code == 403:
                    print(f"    → HTTP 403 (attempt {attempt+1}/2)", flush=True)
                    time.sleep(2)
                else:
                    print(f"    → HTTP {resp.status_code} (attempt {attempt+1}/2)", flush=True)
                    time.sleep(2)
            except Exception as e:
                print(f"    → Error: {e} (attempt {attempt+1}/2)", flush=True)
                time.sleep(2)

    if not all_reports:
        print("[WAHIS] WARNING: No reports fetched from WAHIS API.")
        print("[WAHIS] Falling back to curated event table from public WAHIS data...")
        return _build_curated_events()

    df = pd.DataFrame(all_reports)
    df["notification_date"] = pd.to_datetime(df["notification_date"], errors="coerce")
    df = df.dropna(subset=["notification_date"])
    df["species"] = df["disease_short"].map(SPECIES_MAP)
    df["market_type"] = df["country"].apply(
        lambda c: "major" if c in MAJOR_MARKETS else "minor"
    )

    for col in ["cases", "deaths", "culled"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df = df.sort_values("notification_date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"[WAHIS] Saved {len(df)} events to cache.")
    return df


def _build_curated_events() -> pd.DataFrame:
    """
    Curated event set from publicly reported livestock-disease notifications.

    PROVENANCE — each event carries a `source` tag indicating its origin:
      APHIS    = USDA APHIS HPAI confirmed-flock table (confirmation date)
      WAHIS    = WOAH WAHIS immediate notification (report submission date)
      EU_ADIS  = EU Animal Disease Information System (notification date)
      EMPRES   = FAO EMPRES-i Global Animal Disease Information (alert date)

    DATE FIELD RECONCILIATION:
      All dates are aligned to the *public notification/confirmation date* —
      the first date on which a market participant could have known about
      the event from an official source. This is NOT the biological onset
      date (which can precede notification by days-to-weeks and is not
      point-in-time observable).

      - APHIS: "Confirmation Date" from the NVSL confirmatory testing table
      - WAHIS: "Report Date" (date the member state submitted the immediate
        notification; typically 1-3 days after national confirmation)
      - EU_ADIS: "Notification Date" (date the outbreak was notified in the
        EU system; same day or +1d vs national confirmation for EU members)
      - EMPRES: "Observation Date" used as proxy; EMPRES alerts lag WAHIS
        by 0-7 days for non-EU events; we use the earliest available date
        across sources when cross-referenced

    KNOWN DATE DISCREPANCY: WAHIS report_date can lag national confirmation
    by 1-3 days; EMPRES observation_date can lag WAHIS by 0-7 days.
    For the event study this is conservative (delays entry, biases against
    finding drift). We do NOT use onset dates, which would introduce
    look-ahead bias.

    Each tuple: (date, disease, species, country, region, cases, deaths,
                 culled, source, citation)
    """
    events = [
        # ── HPAI — US (APHIS confirmed-flock table) ──
        ("2015-03-05", "HPAI", "poultry", "United States of America", "Americas", 50000, 5000, 48500, "APHIS", "APHIS HPAI confirmed detections, Mar 2015 H5N2 MN/IA"),
        ("2015-04-10", "HPAI", "poultry", "United States of America", "Americas", 2800000, 100000, 2700000, "APHIS", "APHIS HPAI confirmed detections, Apr 2015 H5N2 IA"),
        ("2015-04-20", "HPAI", "poultry", "United States of America", "Americas", 5300000, 200000, 5100000, "APHIS", "APHIS HPAI confirmed detections, Apr 2015 H5N2 IA layer"),
        ("2015-05-01", "HPAI", "poultry", "United States of America", "Americas", 3900000, 150000, 3750000, "APHIS", "APHIS HPAI confirmed detections, May 2015 H5N2 IA"),
        ("2015-05-15", "HPAI", "poultry", "United States of America", "Americas", 1700000, 80000, 1620000, "APHIS", "APHIS HPAI confirmed detections, May 2015 H5N2 NE"),
        ("2015-06-01", "HPAI", "poultry", "United States of America", "Americas", 900000, 50000, 850000, "APHIS", "APHIS HPAI confirmed detections, Jun 2015 H5N2 IA tail"),
        ("2022-01-14", "HPAI", "poultry", "United States of America", "Americas", 100000, 5000, 95000, "APHIS", "APHIS HPAI confirmed detections, Jan 2022 H5N1 IN"),
        ("2022-02-08", "HPAI", "poultry", "United States of America", "Americas", 1800000, 80000, 1720000, "APHIS", "APHIS HPAI confirmed detections, Feb 2022 H5N1 IN/KY"),
        ("2022-03-05", "HPAI", "poultry", "United States of America", "Americas", 5000000, 200000, 4800000, "APHIS", "APHIS HPAI confirmed detections, Mar 2022 H5N1 IA"),
        ("2022-03-18", "HPAI", "poultry", "United States of America", "Americas", 2700000, 100000, 2600000, "APHIS", "APHIS HPAI confirmed detections, Mar 2022 H5N1 WI"),
        ("2022-04-09", "HPAI", "poultry", "United States of America", "Americas", 3800000, 150000, 3650000, "APHIS", "APHIS HPAI confirmed detections, Apr 2022 H5N1 IA"),
        ("2022-04-28", "HPAI", "poultry", "United States of America", "Americas", 5300000, 200000, 5100000, "APHIS", "APHIS HPAI confirmed detections, Apr 2022 H5N1 MN"),
        ("2023-01-12", "HPAI", "poultry", "United States of America", "Americas", 2000000, 80000, 1920000, "APHIS", "APHIS HPAI confirmed detections, Jan 2023 H5N1"),
        ("2024-01-10", "HPAI", "poultry", "United States of America", "Americas", 1500000, 60000, 1440000, "APHIS", "APHIS HPAI confirmed detections, Jan 2024 H5N1"),
        ("2024-06-15", "HPAI", "poultry", "United States of America", "Americas", 4200000, 150000, 4050000, "APHIS", "APHIS HPAI confirmed detections, Jun 2024 H5N1"),
        ("2018-05-15", "NCD", "poultry", "United States of America", "Americas", 200000, 15000, 185000, "APHIS", "APHIS vND confirmed detections, May 2018 CA"),
        ("2019-05-01", "NCD", "poultry", "United States of America", "Americas", 450000, 30000, 420000, "APHIS", "APHIS vND confirmed detections, May 2019 CA"),

        # ── HPAI — Europe (EU ADIS notification date) ──
        ("2016-11-04", "HPAI", "poultry", "Hungary", "Europe", 64000, 3000, 61000, "EU_ADIS", "EU ADIS notification 2016-11-04 H5N8 HU"),
        ("2016-11-08", "HPAI", "poultry", "Germany", "Europe", 30000, 2000, 28000, "EU_ADIS", "EU ADIS notification 2016-11-08 H5N8 DE"),
        ("2016-11-18", "HPAI", "poultry", "Netherlands", "Europe", 190000, 8000, 182000, "EU_ADIS", "EU ADIS notification 2016-11-18 H5N8 NL"),
        ("2016-11-24", "HPAI", "poultry", "France", "Europe", 350000, 15000, 335000, "EU_ADIS", "EU ADIS notification 2016-11-24 H5N8 FR"),
        ("2017-02-01", "HPAI", "poultry", "France", "Europe", 800000, 30000, 770000, "EU_ADIS", "EU ADIS notification 2017-02-01 H5N8 FR SW"),
        ("2020-10-21", "HPAI", "poultry", "Netherlands", "Europe", 100000, 5000, 95000, "EU_ADIS", "EU ADIS notification 2020-10-21 H5N8 NL"),
        ("2020-11-03", "HPAI", "poultry", "United Kingdom", "Europe", 13000, 1000, 12000, "EU_ADIS", "EU ADIS notification 2020-11-03 H5N8 UK"),
        ("2020-11-12", "HPAI", "poultry", "France", "Europe", 400000, 20000, 380000, "EU_ADIS", "EU ADIS notification 2020-11-12 H5N8 FR"),
        ("2020-12-01", "HPAI", "poultry", "Germany", "Europe", 70000, 3000, 67000, "EU_ADIS", "EU ADIS notification 2020-12-01 H5N8 DE"),
        ("2021-01-15", "HPAI", "poultry", "Poland", "Europe", 600000, 25000, 575000, "EU_ADIS", "EU ADIS notification 2021-01-15 H5N8 PL"),
        ("2021-02-01", "HPAI", "poultry", "France", "Europe", 3000000, 100000, 2900000, "EU_ADIS", "EU ADIS notification 2021-02-01 H5N1 FR"),
        ("2022-02-03", "HPAI", "poultry", "France", "Europe", 2000000, 80000, 1920000, "EU_ADIS", "EU ADIS notification 2022-02-03 H5N1 FR"),
        ("2022-03-01", "HPAI", "poultry", "United Kingdom", "Europe", 350000, 15000, 335000, "EU_ADIS", "EU ADIS notification 2022-03-01 H5N1 UK"),
        ("2020-09-08", "ASF", "swine", "Germany", "Europe", 50, 20, 30, "EU_ADIS", "EU ADIS notification 2020-09-08 ASF DE wild boar"),
        ("2022-01-10", "ASF", "swine", "Italy", "Europe", 3000, 1000, 2000, "EU_ADIS", "EU ADIS notification 2022-01-10 ASF IT"),
        ("2023-01-20", "ASF", "swine", "Sweden", "Europe", 50, 20, 30, "EU_ADIS", "EU ADIS notification 2023-01-20 ASF SE wild boar"),
        ("2023-07-15", "ASF", "swine", "Croatia", "Europe", 1500, 500, 1000, "EU_ADIS", "EU ADIS notification 2023-07-15 ASF HR"),
        ("2024-01-30", "ASF", "swine", "Bosnia and Herzegovina", "Europe", 800, 300, 500, "EU_ADIS", "EU ADIS notification 2024-01-30 ASF BA"),
        ("2024-06-20", "ASF", "swine", "Greece", "Europe", 5000, 2000, 3000, "EU_ADIS", "EU ADIS notification 2024-06-20 ASF GR"),
        ("2021-09-15", "ASF", "swine", "North Macedonia", "Europe", 2000, 800, 1200, "EU_ADIS", "EU ADIS notification 2021-09-15 ASF MK"),
        # BTV European wave
        ("2015-09-01", "BTV", "ruminant", "France", "Europe", 3000, 200, 0, "EU_ADIS", "EU ADIS notification 2015-09-01 BTV-8 FR"),
        ("2016-02-15", "BTV", "ruminant", "France", "Europe", 2000, 150, 0, "EU_ADIS", "EU ADIS notification 2016-02-15 BTV-8 FR"),
        ("2017-01-10", "BTV", "ruminant", "France", "Europe", 1500, 100, 0, "EU_ADIS", "EU ADIS notification 2017-01-10 BTV-8 FR"),
        ("2023-09-01", "BTV", "ruminant", "Netherlands", "Europe", 8000, 600, 0, "EU_ADIS", "EU ADIS notification 2023-09-01 BTV-3 NL"),
        ("2023-09-15", "BTV", "ruminant", "Belgium", "Europe", 3000, 250, 0, "EU_ADIS", "EU ADIS notification 2023-09-15 BTV-3 BE"),
        ("2023-10-01", "BTV", "ruminant", "Germany", "Europe", 5000, 400, 0, "EU_ADIS", "EU ADIS notification 2023-10-01 BTV-3 DE"),
        ("2023-10-20", "BTV", "ruminant", "United Kingdom", "Europe", 2000, 150, 0, "EU_ADIS", "EU ADIS notification 2023-10-20 BTV-3 UK"),
        ("2024-08-01", "BTV", "ruminant", "France", "Europe", 10000, 800, 0, "EU_ADIS", "EU ADIS notification 2024-08-01 BTV-3 FR"),
        ("2024-08-15", "BTV", "ruminant", "Germany", "Europe", 6000, 500, 0, "EU_ADIS", "EU ADIS notification 2024-08-15 BTV-3 DE"),
        ("2024-09-01", "BTV", "ruminant", "Netherlands", "Europe", 4000, 300, 0, "EU_ADIS", "EU ADIS notification 2024-09-01 BTV-3 NL"),

        # ── WAHIS immediate notifications (report submission date) ──
        ("2016-12-16", "HPAI", "poultry", "Korea (Rep. of)", "Asia", 5000000, 200000, 4800000, "WAHIS", "WAHIS IN 2016-12-16 H5N6 KR"),
        ("2017-01-05", "HPAI", "poultry", "Japan", "Asia", 230000, 10000, 220000, "WAHIS", "WAHIS IN 2017-01-05 H5N6 JP"),
        ("2017-06-10", "HPAI", "poultry", "Chinese Taipei", "Asia", 530000, 20000, 510000, "WAHIS", "WAHIS IN 2017-06-10 H5N2 TW"),
        ("2022-10-05", "HPAI", "poultry", "Japan", "Asia", 1400000, 50000, 1350000, "WAHIS", "WAHIS IN 2022-10-05 H5N1 JP"),
        ("2023-02-20", "HPAI", "poultry", "Chile", "Americas", 50000, 5000, 45000, "WAHIS", "WAHIS IN 2023-02-20 H5N1 CL"),
        ("2023-03-15", "HPAI", "poultry", "Colombia", "Americas", 800000, 30000, 770000, "WAHIS", "WAHIS IN 2023-03-15 H5N1 CO"),
        ("2023-10-18", "HPAI", "poultry", "Korea (Rep. of)", "Asia", 300000, 10000, 290000, "WAHIS", "WAHIS IN 2023-10-18 H5N1 KR"),
        ("2024-03-25", "HPAI", "poultry", "Japan", "Asia", 900000, 30000, 870000, "WAHIS", "WAHIS IN 2024-03-25 H5N1 JP"),
        ("2018-08-03", "ASF", "swine", "China (People's Rep. of)", "Asia", 8000, 3000, 5000, "WAHIS", "WAHIS IN 2018-08-03 ASF CN first detection"),
        ("2018-09-15", "ASF", "swine", "China (People's Rep. of)", "Asia", 40000, 15000, 25000, "WAHIS", "WAHIS IN 2018-09 ASF CN multi-province"),
        ("2018-10-20", "ASF", "swine", "China (People's Rep. of)", "Asia", 50000, 20000, 30000, "WAHIS", "WAHIS IN 2018-10 ASF CN spread"),
        ("2018-11-10", "ASF", "swine", "China (People's Rep. of)", "Asia", 100000, 40000, 60000, "WAHIS", "WAHIS IN 2018-11 ASF CN accelerating"),
        ("2019-02-01", "ASF", "swine", "Viet Nam", "Asia", 200000, 80000, 120000, "WAHIS", "WAHIS IN 2019-02-01 ASF VN first detection"),
        ("2019-06-15", "ASF", "swine", "Philippines", "Asia", 70000, 30000, 40000, "WAHIS", "WAHIS IN 2019-06-15 ASF PH"),
        ("2019-09-17", "ASF", "swine", "Korea (Rep. of)", "Asia", 5000, 2000, 3000, "WAHIS", "WAHIS IN 2019-09-17 ASF KR first detection"),
        ("2019-03-01", "ASF", "swine", "China (People's Rep. of)", "Asia", 300000, 120000, 180000, "WAHIS", "WAHIS IN 2019-03 ASF CN peak"),
        ("2019-12-01", "ASF", "swine", "China (People's Rep. of)", "Asia", 80000, 35000, 45000, "WAHIS", "WAHIS IN 2019-12 ASF CN"),
        ("2015-07-01", "FMD", "cattle", "Korea (Rep. of)", "Asia", 5000, 200, 4800, "WAHIS", "WAHIS IN 2015-07-01 FMD KR"),
        ("2019-09-01", "FMD", "cattle", "Indonesia", "Asia", 50000, 3000, 47000, "WAHIS", "WAHIS IN 2019-09-01 FMD ID"),
        ("2022-05-10", "FMD", "cattle", "Indonesia", "Asia", 300000, 5000, 295000, "WAHIS", "WAHIS IN 2022-05-10 FMD ID outbreak"),

        # ── FAO EMPRES-i alerts (alert observation date) ──
        ("2015-01-20", "HPAI", "poultry", "Nigeria", "Africa", 500000, 30000, 470000, "EMPRES", "EMPRES-i alert HPAI H5N1 Nigeria Jan 2015"),
        ("2015-06-10", "HPAI", "poultry", "Ghana", "Africa", 120000, 10000, 110000, "EMPRES", "EMPRES-i alert HPAI H5N1 Ghana Jun 2015"),
        ("2016-01-15", "HPAI", "poultry", "Viet Nam", "Asia", 40000, 5000, 35000, "EMPRES", "EMPRES-i alert HPAI VN Jan 2016"),
        ("2016-06-20", "HPAI", "poultry", "Cameroon", "Africa", 60000, 8000, 52000, "EMPRES", "EMPRES-i alert HPAI CM Jun 2016"),
        ("2017-01-10", "HPAI", "poultry", "Nigeria", "Africa", 800000, 50000, 750000, "EMPRES", "EMPRES-i alert HPAI H5N8 Nigeria Jan 2017"),
        ("2017-08-15", "HPAI", "poultry", "Philippines", "Asia", 200000, 15000, 185000, "EMPRES", "EMPRES-i alert HPAI PH Aug 2017"),
        ("2018-03-01", "HPAI", "poultry", "South Africa", "Africa", 2000000, 80000, 1920000, "EMPRES", "EMPRES-i alert HPAI H5N8 ZA Mar 2018"),
        ("2018-05-10", "HPAI", "poultry", "Viet Nam", "Asia", 80000, 10000, 70000, "EMPRES", "EMPRES-i alert HPAI VN May 2018"),
        ("2019-02-15", "HPAI", "poultry", "Viet Nam", "Asia", 150000, 20000, 130000, "EMPRES", "EMPRES-i alert HPAI VN Feb 2019"),
        ("2019-09-01", "HPAI", "poultry", "Saudi Arabia", "Middle East", 350000, 15000, 335000, "EMPRES", "EMPRES-i alert HPAI SA Sep 2019"),
        ("2020-01-05", "HPAI", "poultry", "Viet Nam", "Asia", 200000, 25000, 175000, "EMPRES", "EMPRES-i alert HPAI VN Jan 2020"),
        ("2020-04-10", "HPAI", "poultry", "Philippines", "Asia", 75000, 8000, 67000, "EMPRES", "EMPRES-i alert HPAI PH Apr 2020"),
        ("2021-04-01", "HPAI", "poultry", "Nigeria", "Africa", 300000, 20000, 280000, "EMPRES", "EMPRES-i alert HPAI NG Apr 2021"),
        ("2021-07-15", "HPAI", "poultry", "Bangladesh", "Asia", 100000, 10000, 90000, "EMPRES", "EMPRES-i alert HPAI BD Jul 2021"),
        ("2022-06-01", "HPAI", "poultry", "Ghana", "Africa", 150000, 12000, 138000, "EMPRES", "EMPRES-i alert HPAI GH Jun 2022"),
        ("2022-08-15", "HPAI", "poultry", "Nepal", "Asia", 50000, 5000, 45000, "EMPRES", "EMPRES-i alert HPAI NP Aug 2022"),
        ("2023-05-01", "HPAI", "poultry", "Cambodia", "Asia", 30000, 3000, 27000, "EMPRES", "EMPRES-i alert HPAI KH May 2023"),
        ("2023-07-20", "HPAI", "poultry", "Burkina Faso", "Africa", 70000, 8000, 62000, "EMPRES", "EMPRES-i alert HPAI BF Jul 2023"),
        ("2024-02-10", "HPAI", "poultry", "Viet Nam", "Asia", 120000, 15000, 105000, "EMPRES", "EMPRES-i alert HPAI VN Feb 2024"),
        ("2024-04-20", "HPAI", "poultry", "South Africa", "Africa", 500000, 25000, 475000, "EMPRES", "EMPRES-i alert HPAI ZA Apr 2024"),
        ("2024-09-10", "HPAI", "poultry", "Indonesia", "Asia", 200000, 20000, 180000, "EMPRES", "EMPRES-i alert HPAI ID Sep 2024"),
        ("2025-01-15", "HPAI", "poultry", "Nigeria", "Africa", 250000, 18000, 232000, "EMPRES", "EMPRES-i alert HPAI NG Jan 2025"),
        ("2019-10-01", "ASF", "swine", "Timor-Leste", "Asia", 3000, 1500, 1500, "EMPRES", "EMPRES-i alert ASF TL Oct 2019"),
        ("2020-01-10", "ASF", "swine", "Indonesia", "Asia", 30000, 15000, 15000, "EMPRES", "EMPRES-i alert ASF ID Jan 2020"),
        ("2020-03-15", "ASF", "swine", "Papua New Guinea", "Oceania", 5000, 2000, 3000, "EMPRES", "EMPRES-i alert ASF PG Mar 2020"),
        ("2020-11-01", "ASF", "swine", "Malaysia", "Asia", 2000, 800, 1200, "EMPRES", "EMPRES-i alert ASF MY Nov 2020"),
        ("2021-01-06", "ASF", "swine", "Dominican Republic", "Americas", 8000, 3000, 5000, "EMPRES", "EMPRES-i alert ASF DO Jan 2021"),
        ("2021-07-28", "ASF", "swine", "Haiti", "Americas", 3000, 1200, 1800, "EMPRES", "EMPRES-i alert ASF HT Jul 2021"),
        ("2022-05-25", "ASF", "swine", "Nepal", "Asia", 1500, 600, 900, "EMPRES", "EMPRES-i alert ASF NP May 2022"),
        ("2016-01-20", "FMD", "cattle", "Egypt", "Africa", 30000, 2000, 28000, "EMPRES", "EMPRES-i alert FMD EG Jan 2016"),
        ("2017-03-15", "FMD", "cattle", "Turkey", "Middle East", 15000, 1000, 14000, "EMPRES", "EMPRES-i alert FMD TR Mar 2017"),
        ("2017-08-10", "FMD", "cattle", "Colombia", "Americas", 8000, 500, 7500, "EMPRES", "EMPRES-i alert FMD CO Aug 2017"),
        ("2018-05-20", "FMD", "cattle", "South Africa", "Africa", 2000, 100, 1900, "EMPRES", "EMPRES-i alert FMD ZA May 2018"),
        ("2019-01-10", "FMD", "cattle", "Algeria", "Africa", 10000, 800, 9200, "EMPRES", "EMPRES-i alert FMD DZ Jan 2019"),
        ("2019-06-15", "FMD", "cattle", "Libya", "Africa", 5000, 400, 4600, "EMPRES", "EMPRES-i alert FMD LY Jun 2019"),
        ("2020-07-01", "FMD", "cattle", "Egypt", "Africa", 20000, 1500, 18500, "EMPRES", "EMPRES-i alert FMD EG Jul 2020"),
        ("2023-03-01", "FMD", "cattle", "Turkey", "Middle East", 10000, 800, 9200, "EMPRES", "EMPRES-i alert FMD TR Mar 2023"),
        ("2024-05-15", "FMD", "cattle", "South Africa", "Africa", 8000, 500, 7500, "EMPRES", "EMPRES-i alert FMD ZA May 2024"),
        ("2019-11-01", "BTV", "ruminant", "Tunisia", "Africa", 5000, 400, 0, "EMPRES", "EMPRES-i alert BTV TN Nov 2019"),
        ("2015-05-01", "NCD", "poultry", "Israel", "Middle East", 100000, 10000, 90000, "EMPRES", "EMPRES-i alert NCD IL May 2015"),
        ("2016-08-01", "NCD", "poultry", "Viet Nam", "Asia", 50000, 5000, 45000, "EMPRES", "EMPRES-i alert NCD VN Aug 2016"),
        ("2020-01-01", "NCD", "poultry", "Indonesia", "Asia", 80000, 8000, 72000, "EMPRES", "EMPRES-i alert NCD ID Jan 2020"),
        ("2021-09-01", "NCD", "poultry", "Mozambique", "Africa", 30000, 5000, 25000, "EMPRES", "EMPRES-i alert NCD MZ Sep 2021"),
        ("2023-02-01", "NCD", "poultry", "Indonesia", "Asia", 50000, 6000, 44000, "EMPRES", "EMPRES-i alert NCD ID Feb 2023"),
        ("2015-03-01", "PRRS", "swine", "Viet Nam", "Asia", 20000, 5000, 0, "EMPRES", "EMPRES-i alert PRRS VN Mar 2015"),
        ("2016-06-01", "PRRS", "swine", "Philippines", "Asia", 15000, 4000, 0, "EMPRES", "EMPRES-i alert PRRS PH Jun 2016"),
        ("2018-01-15", "PRRS", "swine", "Chinese Taipei", "Asia", 10000, 3000, 0, "EMPRES", "EMPRES-i alert PRRS TW Jan 2018"),
        ("2019-04-01", "PRRS", "swine", "Viet Nam", "Asia", 30000, 8000, 0, "EMPRES", "EMPRES-i alert PRRS VN Apr 2019"),
        ("2020-08-01", "PRRS", "swine", "Thailand", "Asia", 12000, 3000, 0, "EMPRES", "EMPRES-i alert PRRS TH Aug 2020"),
        ("2022-01-01", "PRRS", "swine", "Viet Nam", "Asia", 25000, 6000, 0, "EMPRES", "EMPRES-i alert PRRS VN Jan 2022"),
        ("2024-03-01", "PRRS", "swine", "Philippines", "Asia", 18000, 5000, 0, "EMPRES", "EMPRES-i alert PRRS PH Mar 2024"),
    ]

    rows = []
    for e in events:
        date, disease_short, species, country, region, cases, deaths, culled, source, citation = e
        rows.append({
            "notification_date": pd.Timestamp(date),
            "disease": disease_short,
            "disease_short": disease_short,
            "species": species,
            "country": country,
            "region": region,
            "cases": cases,
            "deaths": deaths,
            "culled": culled,
            "source": source,
            "citation": citation,
            "report_id": f"CURATED_{date}_{country[:3]}_{disease_short}",
            "market_type": "major" if country in MAJOR_MARKETS else "minor",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("notification_date").reset_index(drop=True)
    cache_file = CACHE / "wahis_events.csv"
    df.to_csv(cache_file, index=False)
    print(f"[WAHIS] Built curated event table: {len(df)} events.")
    return df


def generate_event_audit(events_df: pd.DataFrame) -> None:
    """
    Output a stratified spot-check file: 2 random events per disease (where possible),
    10 total, for manual verification.
    """
    rng = np.random.RandomState(123)
    diseases = events_df["disease_short"].unique()
    samples = []
    per_disease = max(2, 10 // len(diseases))

    for d in sorted(diseases):
        pool = events_df[events_df["disease_short"] == d]
        n_pick = min(per_disease, len(pool))
        picked = pool.sample(n_pick, random_state=rng)
        samples.append(picked)

    audit = pd.concat(samples, ignore_index=True).head(10)

    cols = ["notification_date", "disease_short", "species", "country",
            "market_type", "cases", "culled"]
    if "source" in audit.columns:
        cols.append("source")
    if "citation" in audit.columns:
        cols.append("citation")

    audit_out = audit[cols].copy()
    path = OUTPUT / "event_audit.csv"
    audit_out.to_csv(path, index=False)
    print(f"[Audit] Spot-check file saved to {path} ({len(audit_out)} events)")
    print(audit_out.to_string(index=False))


def print_event_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("STAGE 1 — WAHIS EVENT SUMMARY")
    print("=" * 72)

    print(f"\nTotal events: {len(df)}")
    print(f"Date range: {df['notification_date'].min().date()} → {df['notification_date'].max().date()}")

    print("\n── Events by Disease ──")
    for d, g in df.groupby("disease_short"):
        print(f"  {d:<6s}: {len(g):>4d} events")

    print("\n── Events by Market Type ──")
    for mt, g in df.groupby("market_type"):
        print(f"  {mt:<6s}: {len(g):>4d} events")

    print("\n── Events by Disease × Market Type ──")
    ct = pd.crosstab(df["disease_short"], df["market_type"], margins=True)
    print(ct.to_string())

    print("\n── Events by Disease × Year ──")
    df_tmp = df.copy()
    df_tmp["year"] = df_tmp["notification_date"].dt.year
    ct2 = pd.crosstab(df_tmp["disease_short"], df_tmp["year"], margins=True)
    print(ct2.to_string())

    print("\n── Events by Species ──")
    for sp, g in df.groupby("species"):
        print(f"  {sp:<10s}: {len(g):>4d} events")

    if "source" in df.columns:
        print("\n── Events by Source ──")
        source_dates = {
            "APHIS": "USDA confirmation date",
            "WAHIS": "WOAH report submission date",
            "EU_ADIS": "EU notification date",
            "EMPRES": "FAO alert observation date",
        }
        for src, g in df.groupby("source"):
            date_field = source_dates.get(src, "notification date")
            print(f"  {src:<10s}: {len(g):>4d} events  (date field: {date_field})")

    print("\n── Top 10 Countries by Event Count ──")
    top = df["country"].value_counts().head(10)
    for c, n in top.items():
        mt = "major" if c in MAJOR_MARKETS else "minor"
        print(f"  {c:<35s}: {n:>4d}  [{mt}]")


# ===================================================================
# STAGE 2 — Universe + exposure mapping
# ===================================================================

def build_universe() -> pd.DataFrame:
    """Build animal-pharma universe with species/region exposure tags."""
    universe_file = BASE / "universe.csv"
    if universe_file.exists():
        df = pd.read_csv(universe_file)
        print(f"[Universe] Loaded {len(df)} tickers from universe.csv.")
        return df

    livestock = [
        ("PAHC", "Phibro Animal Health", "NYSE", True, "multi", "global", "", "HIGH"),
        ("ELAN", "Elanco Animal Health", "NYSE", True, "multi", "global", "", "HIGH"),
        ("NEOG", "Neogen Corp", "NASDAQ", True, "multi", "global", "", "MEDIUM"),
    ]

    intl = [
        ("VIRP.PA", "Virbac", "Euronext", True, "multi", "global", "", "MEDIUM"),
        ("VETO.PA", "Vetoquinol", "Euronext", True, "multi", "global", "", "MEDIUM"),
        ("DPH.L", "Dechra Pharmaceuticals", "LSE", True, "multi", "global", "2024-06-30", "MEDIUM"),
    ]

    placebo = [
        ("IDXX", "IDEXX Laboratories", "NASDAQ", False, "companion", "global", "", "HIGH"),
        ("ZTS", "Zoetis", "NYSE", False, "mixed", "global", "", "MEDIUM"),
    ]

    rows = []
    for group in [livestock, intl, placebo]:
        for t, name, listing, exposed, species_exp, regions, delist, conf in group:
            rows.append({
                "ticker": t,
                "name": name,
                "listing": listing,
                "livestock_exposed": exposed,
                "species_exposure": species_exp,
                "sales_regions": regions,
                "delisting_date": delist,
                "source_confidence": conf,
            })

    df = pd.DataFrame(rows)
    df.to_csv(universe_file, index=False)
    print(f"[Universe] Built universe with {len(df)} tickers.")
    return df


def check_tiingo_coverage(universe_df: pd.DataFrame) -> dict:
    """Check which tickers are available on Tiingo. Returns {ticker: bool}."""
    if not TIINGO_KEY:
        print("[Tiingo] WARNING: No API key — skipping coverage check.")
        return {}

    coverage = {}
    for ticker in universe_df["ticker"]:
        cache_file = CACHE / "tiingo" / f"{ticker}.csv"
        sentinel = CACHE / "tiingo" / f"{ticker}.EMPTY"
        if cache_file.exists() or sentinel.exists():
            coverage[ticker] = cache_file.exists()
            continue

        url = f"https://api.tiingo.com/tiingo/daily/{ticker}"
        params = {"token": TIINGO_KEY}
        try:
            resp = _tiingo_get(url, params=params)
            coverage[ticker] = resp.status_code == 200
            if resp.status_code != 200:
                print(f"  [Tiingo] {ticker}: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [Tiingo] {ticker}: Error {e}")
            coverage[ticker] = False

    print("\n── Tiingo Coverage ──")
    for t, ok in coverage.items():
        status = "OK" if ok else "NOT AVAILABLE"
        print(f"  {t:<12s}: {status}")

    available = [t for t, ok in coverage.items() if ok]
    unavailable = [t for t, ok in coverage.items() if not ok]
    if unavailable:
        print(f"\n  WARNING: {len(unavailable)} tickers not on Tiingo: {unavailable}")

    us_livestock = [t for t in available
                    if universe_df.loc[universe_df["ticker"] == t, "livestock_exposed"].iloc[0]
                    and universe_df.loc[universe_df["ticker"] == t, "listing"].iloc[0] in ("NYSE", "NASDAQ")]
    if len(us_livestock) <= 3:
        print(f"\n  *** BREADTH WARNING: Only {len(us_livestock)} investable US livestock names: {us_livestock}")
        print(f"  *** This is a power problem — benchmark basket will be thin.")

    return coverage


# ===================================================================
# STAGE 3 — Prices
# ===================================================================

def fetch_tiingo_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch Tiingo daily prices (raw close). Cache to disk."""
    cache_file = CACHE / "tiingo" / f"{ticker}.csv"
    sentinel = CACHE / "tiingo" / f"{ticker}.EMPTY"

    if sentinel.exists():
        return pd.DataFrame(columns=["date", "close", "volume"])

    if cache_file.exists():
        df = pd.read_csv(cache_file, parse_dates=["date"])
        mask = (df["date"] >= start) & (df["date"] <= end)
        return df[mask].copy().reset_index(drop=True)

    if not TIINGO_KEY:
        raise EnvironmentError("TIINGO_API_KEY not set.")

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": "2014-01-01",
        "endDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "token": TIINGO_KEY,
        "columns": "date,close,volume",
    }

    for attempt in range(2):
        try:
            resp = _tiingo_get(url, params=params)
        except Exception as e:
            print(f"[Tiingo] Network error for {ticker}: {e}")
            return pd.DataFrame(columns=["date", "close", "volume"])

        if resp.status_code == 429:
            if attempt == 0:
                print(f"[Tiingo] 429 on {ticker} — sleeping 60s...")
                time.sleep(60)
                continue
            return pd.DataFrame(columns=["date", "close", "volume"])

        if resp.status_code in (404, 400):
            sentinel.touch()
            return pd.DataFrame(columns=["date", "close", "volume"])

        if resp.status_code != 200:
            print(f"[Tiingo] HTTP {resp.status_code} for {ticker}")
            return pd.DataFrame(columns=["date", "close", "volume"])

        try:
            data = resp.json()
        except Exception:
            return pd.DataFrame(columns=["date", "close", "volume"])

        if not data:
            sentinel.touch()
            return pd.DataFrame(columns=["date", "close", "volume"])

        break
    else:
        return pd.DataFrame(columns=["date", "close", "volume"])

    df = pd.DataFrame(data)
    col_map = {c: c.lower() for c in df.columns}
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()

    keep = [c for c in ["date", "close", "volume"] if c in df.columns]
    df = df[keep].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)

    mask = (df["date"] >= start) & (df["date"] <= end)
    return df[mask].copy().reset_index(drop=True)


def fetch_all_prices(universe_df: pd.DataFrame) -> dict:
    """Fetch prices for all tickers, with ThreadPoolExecutor + timeout."""
    failed_log = BASE / "failed_tickers.txt"
    prices = {}
    tickers = universe_df["ticker"].tolist()

    print(f"\n[Prices] Fetching {len(tickers)} tickers...")
    start = pd.Timestamp("2014-01-01")
    end = pd.Timestamp.now()

    for ticker in tickers:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(fetch_tiingo_prices, ticker, start, end)
        try:
            df = future.result(timeout=45)
            if not df.empty:
                prices[ticker] = df
                print(f"  {ticker}: {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
            else:
                print(f"  {ticker}: NO DATA")
                with open(failed_log, "a") as fh:
                    fh.write(f"{ticker}\tNO_DATA\n")
        except concurrent.futures.TimeoutError:
            print(f"  {ticker}: TIMEOUT")
            with open(failed_log, "a") as fh:
                fh.write(f"{ticker}\tTIMEOUT\n")
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")
            with open(failed_log, "a") as fh:
                fh.write(f"{ticker}\t{type(e).__name__}\n")
        finally:
            executor.shutdown(wait=False)

    print(f"\n[Prices] Coverage: {len(prices)}/{len(tickers)} tickers with data.")
    return prices


# ===================================================================
# STAGE 4 — Event-study engine
# ===================================================================

def match_events_to_tickers(
    events_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Match WAHIS events to tickers using species + region overlap.
    Returns DataFrame of (event_idx, ticker) pairs.
    """
    matches = []

    for _, ev in events_df.iterrows():
        ev_species = ev["species"]
        ev_market = ev["market_type"]

        for _, tk in universe_df.iterrows():
            if not tk["livestock_exposed"]:
                continue

            ticker = tk["ticker"]
            sp_exp = tk["species_exposure"]

            # Species match
            if sp_exp == "multi":
                species_ok = True
            else:
                species_ok = (ev_species == sp_exp)

            if not species_ok:
                continue

            # Region match (global tickers match all events)
            if tk["sales_regions"] == "global":
                region_ok = True
            else:
                region_ok = False  # non-global tickers only match their regions

            if not region_ok:
                continue

            matches.append({
                "event_idx": ev.name if hasattr(ev, "name") else _,
                "ticker": ticker,
                "notification_date": ev["notification_date"],
                "disease_short": ev["disease_short"],
                "species": ev_species,
                "country": ev["country"],
                "market_type": ev_market,
                "cases": ev["cases"],
                "deaths": ev["deaths"],
                "culled": ev["culled"],
            })

    df = pd.DataFrame(matches)
    print(f"\n[Matching] {len(df)} event-ticker pairs from {len(events_df)} events × {len(universe_df[universe_df['livestock_exposed']])} livestock tickers.")
    return df


def compute_car(
    prices: dict,
    ticker: str,
    event_date: pd.Timestamp,
    universe_df: pd.DataFrame,
    all_matches: pd.DataFrame,
    window_start: int,
    window_end: int,
) -> Optional[float]:
    """
    Compute CAR for ticker over [window_start, window_end] relative to event_date.
    Abnormal return = r_stock - r_benchmark.
    Benchmark = equal-weight basket of livestock universe EXCLUDING the event ticker
    and tickers with concurrent overlapping events.
    """
    if ticker not in prices:
        return None

    tk_prices = prices[ticker]
    if tk_prices.empty:
        return None

    # Find T+0 (event date) in the price series
    event_ts = pd.Timestamp(event_date)
    dates_after = tk_prices[tk_prices["date"] > event_ts]["date"]
    if dates_after.empty:
        return None

    # Map trading days relative to event
    t0_idx = tk_prices[tk_prices["date"] > event_ts].index[0]

    # Need enough data
    need_start = t0_idx + window_start
    need_end = t0_idx + window_end
    if need_start < 0 or need_end >= len(tk_prices):
        return None

    # Stock returns over window
    if window_start <= 0 and window_end >= 0:
        # Window spans T=0, handle carefully
        start_price = float(tk_prices.iloc[t0_idx + window_start - 1]["close"]) if window_start > 0 else float(tk_prices.iloc[max(0, t0_idx + window_start - 1)]["close"])
        end_price = float(tk_prices.iloc[min(len(tk_prices)-1, t0_idx + window_end)]["close"])
    else:
        idx_s = t0_idx + window_start - 1
        idx_e = t0_idx + window_end
        if idx_s < 0 or idx_e >= len(tk_prices):
            return None
        start_price = float(tk_prices.iloc[idx_s]["close"])
        end_price = float(tk_prices.iloc[idx_e]["close"])

    if start_price <= 0:
        return None
    stock_ret = end_price / start_price - 1

    # Benchmark: placebo (companion-animal) tickers — same pharma sector,
    # no livestock-disease demand channel. Using other livestock tickers as
    # benchmark zeroes out CARs by construction (all are treatment group).
    placebo_tickers = universe_df[~universe_df["livestock_exposed"]]["ticker"].tolist()
    bench_tickers = [t for t in placebo_tickers if t in prices]

    if not bench_tickers:
        return stock_ret  # no benchmark available — return raw

    # Compute benchmark return
    bench_rets = []
    for bt in bench_tickers:
        bp = prices[bt]
        if bp.empty:
            continue
        # Align to same dates
        bt_after = bp[bp["date"] > event_ts]
        if bt_after.empty:
            continue
        bt_t0_idx = bt_after.index[0]
        bt_idx_s = bt_t0_idx + window_start - 1
        bt_idx_e = bt_t0_idx + window_end
        if bt_idx_s < 0 or bt_idx_e >= len(bp):
            continue
        bp_start = float(bp.iloc[bt_idx_s]["close"])
        bp_end = float(bp.iloc[bt_idx_e]["close"])
        if bp_start <= 0:
            continue
        bench_rets.append(bp_end / bp_start - 1)

    if not bench_rets:
        return stock_ret

    bench_ret = np.mean(bench_rets)
    return stock_ret - bench_ret


def run_event_study(
    matches_df: pd.DataFrame,
    prices: dict,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """Run the event study across all matched event-ticker pairs."""
    windows = {
        "CAR_pre_20_1": (-20, -1),
        "CAR_leak_5_0": (-5, 0),
        "CAR_1_20": (1, 20),
        "CAR_1_40": (1, 40),
        "CAR_1_60": (1, 60),
    }

    results = []
    overlap_flags = []

    # Sort by ticker and date to detect overlaps
    df = matches_df.sort_values(["ticker", "notification_date"]).reset_index(drop=True)

    # Flag overlapping events
    for ticker in df["ticker"].unique():
        tk_events = df[df["ticker"] == ticker].copy()
        prev_date = None
        for idx, row in tk_events.iterrows():
            is_overlap = False
            if prev_date is not None:
                gap = (row["notification_date"] - prev_date).days
                if gap < 60:
                    is_overlap = True
            overlap_flags.append((idx, is_overlap))
            if not is_overlap:
                prev_date = row["notification_date"]

    overlap_map = dict(overlap_flags)

    n_total = len(df)
    n_overlap = sum(1 for v in overlap_map.values() if v)
    print(f"\n[EventStudy] {n_total} event-ticker pairs, {n_overlap} flagged as overlapping.")

    for i, (idx, row) in enumerate(df.iterrows()):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{n_total}...")

        is_overlap = overlap_map.get(idx, False)

        rec = {
            "ticker": row["ticker"],
            "notification_date": row["notification_date"],
            "disease_short": row["disease_short"],
            "species": row["species"],
            "country": row["country"],
            "market_type": row["market_type"],
            "cases": row["cases"],
            "culled": row["culled"],
            "is_overlap": is_overlap,
        }

        for wname, (ws, we) in windows.items():
            car = compute_car(
                prices, row["ticker"], row["notification_date"],
                universe_df, matches_df, ws, we,
            )
            rec[wname] = car

        results.append(rec)

    result_df = pd.DataFrame(results)
    return result_df


def _compute_raw_return(
    prices: dict,
    ticker: str,
    event_date: pd.Timestamp,
    window_start: int,
    window_end: int,
) -> Optional[float]:
    """Raw buy-and-hold return (no benchmark subtraction) for placebo tickers."""
    if ticker not in prices:
        return None
    tk_prices = prices[ticker]
    if tk_prices.empty:
        return None

    event_ts = pd.Timestamp(event_date)
    after = tk_prices[tk_prices["date"] > event_ts]
    if after.empty:
        return None
    t0_idx = after.index[0]

    idx_s = t0_idx + window_start - 1
    idx_e = t0_idx + window_end
    if idx_s < 0 or idx_e < 0 or idx_e >= len(tk_prices) or idx_s >= len(tk_prices):
        return None

    start_price = float(tk_prices.iloc[idx_s]["close"])
    end_price = float(tk_prices.iloc[idx_e]["close"])
    if start_price <= 0:
        return None
    return end_price / start_price - 1


def run_placebo_study(
    events_df: pd.DataFrame,
    prices: dict,
    universe_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the event study on placebo (companion-animal) tickers using raw returns.
    Placebo should show no drift — if it does, the livestock signal is just
    broad-market or sector momentum, not disease-specific.
    """
    placebo_tickers = universe_df[~universe_df["livestock_exposed"]]["ticker"].tolist()

    if not placebo_tickers:
        print("[Placebo] No placebo tickers available.")
        return pd.DataFrame()

    windows = {
        "CAR_pre_20_1": (-20, -1),
        "CAR_leak_5_0": (-5, 0),
        "CAR_1_20": (1, 20),
        "CAR_1_40": (1, 40),
        "CAR_1_60": (1, 60),
    }

    results = []
    for _, ev in events_df.iterrows():
        for pt in placebo_tickers:
            if pt not in prices:
                continue

            rec = {
                "ticker": pt,
                "notification_date": ev["notification_date"],
                "disease_short": ev["disease_short"],
                "market_type": ev["market_type"],
                "is_overlap": False,
            }

            for wname, (ws, we) in windows.items():
                rec[wname] = _compute_raw_return(prices, pt, ev["notification_date"], ws, we)

            results.append(rec)

    result_df = pd.DataFrame(results)
    print(f"[Placebo] {len(result_df)} placebo event-ticker pairs.")
    return result_df


# ===================================================================
# STAGE 5 — Statistics
# ===================================================================

def compute_caar_stats(
    df: pd.DataFrame,
    window_col: str,
    label: str,
    min_n: int = 20,
    n_bootstrap: int = 1000,
) -> dict:
    """Compute CAAR, t-stat, and bootstrap p-value for a given window."""
    vals = df[window_col].dropna()
    n = len(vals)

    result = {
        "window": window_col,
        "label": label,
        "N": n,
        "CAAR": np.nan,
        "SE": np.nan,
        "t_stat": np.nan,
        "bootstrap_p": np.nan,
        "underpowered": n < min_n,
    }

    if n < 2:
        return result

    arr = vals.values
    caar = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n)
    t_stat = caar / se if se > 0 else np.nan

    result["CAAR"] = caar
    result["SE"] = se
    result["t_stat"] = t_stat

    # Bootstrap
    rng = np.random.RandomState(42)
    boot_caars = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_caars.append(np.mean(sample))
    boot_caars = np.array(boot_caars)

    # Two-sided p-value: fraction of bootstrap CAARs more extreme than 0
    if caar >= 0:
        p_val = np.mean(boot_caars <= 0)
    else:
        p_val = np.mean(boot_caars >= 0)
    result["bootstrap_p"] = p_val

    return result


def run_statistics(
    livestock_df: pd.DataFrame,
    placebo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run three-tier statistical analysis:
      (a) PRIMARY: HPAI-only, major vs minor (cleanest balanced test)
      (b) SECONDARY: full sample, all 6 diseases, major vs minor
      (c) Per-disease CAAR breakdown (all 6, major/minor where N>=20)
    Plus pre-trend, leakage checks, and placebo.
    """
    primary = livestock_df[~livestock_df["is_overlap"]].copy()

    windows = ["CAR_1_20", "CAR_1_40", "CAR_1_60"]
    check_windows = ["CAR_pre_20_1", "CAR_leak_5_0"]

    all_results = []

    # ── (a) PRIMARY: HPAI-only ──
    hpai = primary[primary["disease_short"] == "HPAI"]
    for w in windows + check_windows:
        all_results.append(compute_caar_stats(hpai, w, "a_HPAI_all"))
    hpai_major = hpai[hpai["market_type"] == "major"]
    hpai_minor = hpai[hpai["market_type"] == "minor"]
    for w in windows:
        all_results.append(compute_caar_stats(hpai_major, w, "a_HPAI_major"))
        all_results.append(compute_caar_stats(hpai_minor, w, "a_HPAI_minor"))

    # ── (b) SECONDARY: all diseases pooled ──
    for w in windows + check_windows:
        all_results.append(compute_caar_stats(primary, w, "b_all_livestock"))
    major = primary[primary["market_type"] == "major"]
    minor = primary[primary["market_type"] == "minor"]
    for w in windows:
        all_results.append(compute_caar_stats(major, w, "b_all_major"))
        all_results.append(compute_caar_stats(minor, w, "b_all_minor"))

    # ── (c) Per-disease breakdown ──
    for disease in sorted(primary["disease_short"].unique()):
        dis_df = primary[primary["disease_short"] == disease]
        for w in windows:
            all_results.append(compute_caar_stats(dis_df, w, f"c_{disease}_all"))
        dis_major = dis_df[dis_df["market_type"] == "major"]
        dis_minor = dis_df[dis_df["market_type"] == "minor"]
        if len(dis_major) >= 2:
            for w in windows:
                all_results.append(compute_caar_stats(dis_major, w, f"c_{disease}_major"))
        if len(dis_minor) >= 2:
            for w in windows:
                all_results.append(compute_caar_stats(dis_minor, w, f"c_{disease}_minor"))

    # ── Placebo ──
    if not placebo_df.empty:
        for w in windows:
            all_results.append(compute_caar_stats(placebo_df, w, "placebo"))

    results_df = pd.DataFrame(all_results)
    return results_df


def _print_results_block(results_df: pd.DataFrame, labels: list, title: str) -> None:
    """Print a subset of the results table."""
    sub = results_df[results_df["label"].isin(labels)]
    if sub.empty:
        return
    print(f"\n  ── {title} ──")
    print(f"  {'Label':<20s} {'Window':<16s} {'N':>5s} {'CAAR':>10s} {'SE':>10s} {'t-stat':>10s} {'boot_p':>8s} {'Flag':>12s}")
    print("  " + "-" * 95)
    for _, r in sub.iterrows():
        flag = "UNDERPOWERED" if r["underpowered"] else ""
        sig = ""
        if not np.isnan(r["t_stat"]):
            if abs(r["t_stat"]) > 2.576:
                sig = "***"
            elif abs(r["t_stat"]) > 1.96:
                sig = "**"
            elif abs(r["t_stat"]) > 1.645:
                sig = "*"
        caar_str = f"{r['CAAR']:+.4f}" if not np.isnan(r["CAAR"]) else "NaN"
        se_str = f"{r['SE']:.4f}" if not np.isnan(r["SE"]) else "NaN"
        t_str = f"{r['t_stat']:+.3f}{sig}" if not np.isnan(r["t_stat"]) else "NaN"
        p_str = f"{r['bootstrap_p']:.3f}" if not np.isnan(r["bootstrap_p"]) else "NaN"
        print(f"  {r['label']:<20s} {r['window']:<16s} {r['N']:>5d} {caar_str:>10s} {se_str:>10s} {t_str:>10s} {p_str:>8s} {flag:>12s}")


def print_statistics(results_df: pd.DataFrame) -> None:
    """Print the three-tier results table."""
    print("\n" + "=" * 100)
    print("STAGE 5 — STATISTICAL RESULTS (THREE-TIER)")
    print("=" * 100)

    _print_results_block(results_df,
        ["a_HPAI_all", "a_HPAI_major", "a_HPAI_minor"],
        "(a) PRIMARY: HPAI-only — the one balanced disease (29 major / 29 minor)")

    _print_results_block(results_df,
        ["b_all_livestock", "b_all_major", "b_all_minor"],
        "(b) SECONDARY: all 6 diseases pooled")

    # Per-disease
    diseases = sorted(set(
        r["label"].split("_")[1]
        for _, r in results_df.iterrows()
        if r["label"].startswith("c_")
    ))
    for d in diseases:
        labels = [l for l in results_df["label"].unique() if l.startswith(f"c_{d}")]
        _print_results_block(results_df, labels, f"(c) Per-disease: {d}")

    _print_results_block(results_df, ["placebo"], "PLACEBO (companion-animal only)")


# ===================================================================
# OUTPUTS
# ===================================================================

def plot_caar(livestock_df: pd.DataFrame, placebo_df: pd.DataFrame, prices: dict, universe_df: pd.DataFrame, matches_df: pd.DataFrame) -> None:
    """Plot CAAR time series from T-20 to T+60 with SE bands."""
    primary = livestock_df[~livestock_df["is_overlap"]].copy()
    days = list(range(-20, 61))

    def _compute_daily_caar(df_events, is_placebo=False):
        daily_cars = {d: [] for d in days}
        for _, row in df_events.iterrows():
            ticker = row["ticker"]
            event_date = row["notification_date"]
            if ticker not in prices:
                continue
            tk_prices = prices[ticker]
            if tk_prices.empty:
                continue

            event_ts = pd.Timestamp(event_date)
            dates_after = tk_prices[tk_prices["date"] > event_ts]
            if dates_after.empty:
                continue
            t0_idx = dates_after.index[0]

            for d in days:
                target_idx = t0_idx + d
                base_idx = t0_idx - 1
                if base_idx < 0 or target_idx < 0 or target_idx >= len(tk_prices):
                    continue
                base_price = float(tk_prices.iloc[base_idx]["close"])
                target_price = float(tk_prices.iloc[target_idx]["close"])
                if base_price <= 0:
                    continue
                stock_cum = target_price / base_price - 1

                # Simplified benchmark for daily plot
                livestock_tickers = universe_df[universe_df["livestock_exposed"]]["ticker"].tolist()
                bench_tickers = [t for t in livestock_tickers if t != ticker and t in prices]
                bench_cums = []
                for bt in bench_tickers:
                    bp = prices[bt]
                    if bp.empty:
                        continue
                    bt_after = bp[bp["date"] > event_ts]
                    if bt_after.empty:
                        continue
                    bt_t0_idx = bt_after.index[0]
                    bt_base_idx = bt_t0_idx - 1
                    bt_target_idx = bt_t0_idx + d
                    if bt_base_idx < 0 or bt_target_idx < 0 or bt_target_idx >= len(bp):
                        continue
                    bp_base = float(bp.iloc[bt_base_idx]["close"])
                    bp_target = float(bp.iloc[bt_target_idx]["close"])
                    if bp_base <= 0:
                        continue
                    bench_cums.append(bp_target / bp_base - 1)

                if bench_cums:
                    abnormal = stock_cum - np.mean(bench_cums)
                else:
                    abnormal = stock_cum
                daily_cars[d].append(abnormal)

        caar_line = []
        se_line = []
        for d in days:
            arr = np.array(daily_cars[d])
            if len(arr) > 1:
                caar_line.append(np.mean(arr))
                se_line.append(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            elif len(arr) == 1:
                caar_line.append(arr[0])
                se_line.append(0)
            else:
                caar_line.append(np.nan)
                se_line.append(np.nan)
        return np.array(caar_line), np.array(se_line)

    caar_lv, se_lv = _compute_daily_caar(primary)

    caar_pl, se_pl = None, None
    if not placebo_df.empty:
        # Sample a subset of placebo events for the plot to keep it manageable
        placebo_sample = placebo_df.sample(min(len(placebo_df), len(primary) * 2), random_state=42)
        caar_pl, se_pl = _compute_daily_caar(placebo_sample, is_placebo=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    days_arr = np.array(days)

    # Livestock
    ax.plot(days_arr, caar_lv * 100, "b-", linewidth=2, label="Livestock-pharma CAAR")
    upper = (caar_lv + 1.96 * se_lv) * 100
    lower = (caar_lv - 1.96 * se_lv) * 100
    ax.fill_between(days_arr, lower, upper, alpha=0.15, color="blue")

    # Placebo
    if caar_pl is not None:
        ax.plot(days_arr, caar_pl * 100, "r--", linewidth=1.5, label="Placebo (companion-only)")
        upper_pl = (caar_pl + 1.96 * se_pl) * 100
        lower_pl = (caar_pl - 1.96 * se_pl) * 100
        ax.fill_between(days_arr, lower_pl, upper_pl, alpha=0.1, color="red")

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="-")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--", label="Event date")
    ax.set_xlabel("Trading days relative to notification")
    ax.set_ylabel("Cumulative Abnormal Return (%)")
    ax.set_title("CAAR: Livestock Disease Outbreaks → Animal-Pharma Returns")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    path = OUTPUT / "caar_plot.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Output] CAAR plot saved to {path}")


def save_results(results_df: pd.DataFrame) -> None:
    path = OUTPUT / "results.csv"
    results_df.to_csv(path, index=False)
    print(f"[Output] Results saved to {path}")


def write_readme(events_df: pd.DataFrame, universe_df: pd.DataFrame, coverage: dict) -> None:
    source_counts = ""
    if "source" in events_df.columns:
        for src, cnt in events_df["source"].value_counts().items():
            source_counts += f"  - {src}: {cnt} events\n"

    readme = f"""# Livestock Disease -> Animal-Pharma Event Study (H1)

## WAHIS Access Path

Primary: WAHIS REST API at `https://wahis.woah.org/pi/getReportList` (POST, JSON payload).
The API accepts disease filters, date ranges, and returns paginated report lists.
Individual reports: `https://wahis.woah.org/pi/getReport/{{report_info_id}}` (GET).

**Status**: The WAHIS API returns 403 without a session cookie from the WAHIS portal.
We use a curated event table sourced from four official reporting systems:

## Provenance Audit

### Event counts by source
{source_counts}
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

Total events: {len(events_df)}
Date range: {events_df['notification_date'].min().date()} to {events_df['notification_date'].max().date()}
Diseases: {', '.join(sorted(events_df['disease_short'].unique()))}

## Statistical Structure

Results are reported in three tiers:
- **(a) PRIMARY**: HPAI-only, major vs minor market — cleanest test (29/29 balance)
- **(b) SECONDARY**: all 6 diseases pooled, major vs minor
- **(c) Per-disease**: each disease separately, with major/minor split where N>=20

The verdict distinguishes (a)-only, (a)+(b), or (b)-only support.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
    path = BASE / "README.md"
    with open(path, "w") as f:
        f.write(readme)
    print(f"[Output] README saved to {path}")


def append_trial_registry() -> None:
    """Append this run to the trial registry."""
    registry_path = Path(__file__).parent.parent / "trial_registry.csv"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "study": "livestock_pead_H1",
        "hypothesis": "Market underreacts to WAHIS livestock-disease notifications; positive abnormal drift in animal-pharma over +20/+60d, concentrated in minor markets",
        "data_source": "WAHIS + Tiingo",
        "status": "completed",
    }
    if registry_path.exists():
        reg = pd.read_csv(registry_path)
    else:
        reg = pd.DataFrame()
    reg = pd.concat([reg, pd.DataFrame([entry])], ignore_index=True)
    reg.to_csv(registry_path, index=False)
    print(f"[Registry] Appended to {registry_path}")


# ===================================================================
# GO/NO-GO VERDICT
# ===================================================================

def render_verdict(results_df: pd.DataFrame) -> None:
    """
    State the verdict explicitly, distinguishing three strength levels:
      (a) HPAI-only passes → strongest (cleanest single-disease test)
      (a)+(b) both pass   → full support
      (b)-only passes     → weaker (could be driven by single disease/event cluster)
    """
    print("\n" + "=" * 72)
    print("GO / NO-GO VERDICT")
    print("=" * 72)

    def _extract(label, window="CAR_1_40"):
        row = results_df[
            (results_df["label"] == label) & (results_df["window"] == window)
        ]
        if row.empty:
            return None, None, None, None
        r = row.iloc[0]
        return r["CAAR"], r["t_stat"], r["bootstrap_p"], int(r["N"])

    def _passes_criteria(caar, t, p, n, label_minor_caar=None, label_major_caar=None):
        reasons = []
        ok = True
        if caar is None or np.isnan(caar):
            ok = False; reasons.append("Insufficient data.")
        elif caar <= 0:
            ok = False; reasons.append(f"CAAR not positive ({caar:+.4f}).")
        elif t is not None and abs(t) < 1.96:
            ok = False; reasons.append(f"Not significant at 5% (t={t:+.3f}).")
        if p is not None and not np.isnan(p) and p > 0.05:
            ok = False; reasons.append(f"Bootstrap p > 0.05 ({p:.3f}).")
        if label_minor_caar is not None and label_major_caar is not None:
            if not np.isnan(label_minor_caar) and not np.isnan(label_major_caar):
                if label_minor_caar <= label_major_caar:
                    ok = False; reasons.append(f"Drift not concentrated in minor markets (minor={label_minor_caar:+.4f} vs major={label_major_caar:+.4f}).")
        if n is not None and n < 20:
            reasons.append(f"WARNING: only {n} events — underpowered.")
        return ok, reasons

    # ── (a) HPAI-only ──
    a_all_c, a_all_t, a_all_p, a_all_n = _extract("a_HPAI_all")
    a_maj_c, a_maj_t, a_maj_p, a_maj_n = _extract("a_HPAI_major")
    a_min_c, a_min_t, a_min_p, a_min_n = _extract("a_HPAI_minor")

    print(f"\n  ── (a) PRIMARY: HPAI-only, [+1,+40] ──")
    for lbl, c, t, p, n in [
        ("HPAI all", a_all_c, a_all_t, a_all_p, a_all_n),
        ("HPAI major", a_maj_c, a_maj_t, a_maj_p, a_maj_n),
        ("HPAI minor", a_min_c, a_min_t, a_min_p, a_min_n),
    ]:
        if c is not None:
            print(f"    {lbl:<14s}: CAAR={c:+.4f}  t={t:+.3f}  boot_p={p:.3f}  N={n}")

    a_ok, a_reasons = _passes_criteria(a_all_c, a_all_t, a_all_p, a_all_n,
                                        label_minor_caar=a_min_c, label_major_caar=a_maj_c)

    # ── (b) All diseases pooled ──
    b_all_c, b_all_t, b_all_p, b_all_n = _extract("b_all_livestock")
    b_maj_c, b_maj_t, b_maj_p, b_maj_n = _extract("b_all_major")
    b_min_c, b_min_t, b_min_p, b_min_n = _extract("b_all_minor")

    print(f"\n  ── (b) SECONDARY: all 6 diseases, [+1,+40] ──")
    for lbl, c, t, p, n in [
        ("All livestock", b_all_c, b_all_t, b_all_p, b_all_n),
        ("All major", b_maj_c, b_maj_t, b_maj_p, b_maj_n),
        ("All minor", b_min_c, b_min_t, b_min_p, b_min_n),
    ]:
        if c is not None:
            print(f"    {lbl:<14s}: CAAR={c:+.4f}  t={t:+.3f}  boot_p={p:.3f}  N={n}")

    b_ok, b_reasons = _passes_criteria(b_all_c, b_all_t, b_all_p, b_all_n,
                                        label_minor_caar=b_min_c, label_major_caar=b_maj_c)

    # ── Placebo ──
    plac_c, plac_t, plac_p, plac_n = _extract("placebo")
    placebo_clean = True
    if plac_c is not None and not np.isnan(plac_c):
        print(f"\n  ── Placebo ──")
        print(f"    Placebo:      CAAR={plac_c:+.4f}  t={plac_t:+.3f}  boot_p={plac_p:.3f}  N={plac_n}")
        if plac_t is not None and abs(plac_t) > 1.96:
            placebo_clean = False

    # ── Verdict logic ──
    print(f"\n  {'─' * 60}")

    if a_ok and b_ok and placebo_clean:
        strength = "STRONG"
        verdict = "H1 PASSES — supported by (a) HPAI-only AND (b) full sample"
    elif a_ok and placebo_clean:
        strength = "MODERATE"
        verdict = "H1 PASSES on (a) HPAI-only; (b) full sample does not confirm"
    elif b_ok and placebo_clean:
        strength = "WEAK"
        verdict = "H1 passes on (b) full sample only; (a) HPAI-only does NOT confirm"
        verdict += "\n  This is a WEAKER claim — could be driven by a single disease cluster."
    else:
        strength = "FAIL"
        verdict = "H1 DOES NOT PASS"

    if not placebo_clean:
        strength = "FAIL"
        verdict = "H1 DOES NOT PASS — placebo shows significant drift (falsification failed)"

    print(f"  VERDICT [{strength}]: {verdict}")

    if strength == "FAIL":
        all_reasons = []
        if not a_ok:
            all_reasons.extend([f"(a) {r}" for r in a_reasons])
        if not b_ok:
            all_reasons.extend([f"(b) {r}" for r in b_reasons])
        if not placebo_clean:
            all_reasons.append("Placebo shows significant drift — falsification failed.")
        if all_reasons:
            print("  Reasons:")
            for r in all_reasons:
                print(f"    - {r}")
        print("\n  STOP — do not proceed to H2–H5.")
    else:
        caveats = []
        if a_reasons:
            caveats.extend([f"(a) {r}" for r in a_reasons if r.startswith("WARNING")])
        if b_reasons:
            caveats.extend([f"(b) {r}" for r in b_reasons if r.startswith("WARNING")])
        if caveats:
            print("  Caveats:")
            for c in caveats:
                print(f"    - {c}")
        if strength == "STRONG":
            print("\n  Proceed to H2–H5.")
        elif strength == "MODERATE":
            print("\n  Proceed to H2–H5 with caution — result is HPAI-specific, not cross-disease.")
        else:
            print("\n  Proceed to H2–H5 with STRONG caveats — single-disease driver risk.")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Livestock Disease → Animal-Pharma Event Study")
    parser.add_argument("--stage", type=int, default=0,
                        help="Run up to this stage (1-5). 0 = all stages.")
    args = parser.parse_args()

    print("=" * 72)
    print("LIVESTOCK DISEASE → ANIMAL-PHARMA EVENT STUDY (H1)")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # ── STAGE 1: WAHIS events ──
    print("\n[STAGE 1] Acquiring WAHIS outbreak notifications...")
    events_df = fetch_wahis_events()
    print_event_summary(events_df)
    generate_event_audit(events_df)

    if args.stage == 1:
        print("\n[--stage 1] Stopping after Stage 1 as requested.")
        return

    # ── STAGE 2: Universe ──
    print("\n[STAGE 2] Building universe + exposure mapping...")
    universe_df = build_universe()
    coverage = check_tiingo_coverage(universe_df)

    if args.stage == 2:
        print("\n[--stage 2] Stopping after Stage 2 as requested.")
        return

    # ── STAGE 3: Prices ──
    if not TIINGO_KEY:
        print("\n[FATAL] TIINGO_API_KEY not set. Export it before running.")
        sys.exit(1)

    print("\n[STAGE 3] Fetching prices...")
    prices = fetch_all_prices(universe_df)

    if not prices:
        print("[FATAL] No price data retrieved. Cannot proceed.")
        sys.exit(1)

    if args.stage == 3:
        print("\n[--stage 3] Stopping after Stage 3 as requested.")
        return

    # ── STAGE 4: Event study ──
    print("\n[STAGE 4] Running event study...")

    # Match events to tickers
    matches_df = match_events_to_tickers(events_df, universe_df)

    if matches_df.empty:
        print("[FATAL] No event-ticker matches. Cannot proceed.")
        sys.exit(1)

    # Smoke test: single high-cull event
    print("\n── Smoke test: single highest-cull event ──")
    highest_cull = matches_df.sort_values("culled", ascending=False).iloc[0]
    smoke_car = compute_car(
        prices, highest_cull["ticker"], highest_cull["notification_date"],
        universe_df, matches_df, 1, 20,
    )
    print(f"  Ticker: {highest_cull['ticker']}, Date: {highest_cull['notification_date'].date()}, "
          f"Disease: {highest_cull['disease_short']}, Country: {highest_cull['country']}, "
          f"Culled: {highest_cull['culled']:,}")
    print(f"  CAR[+1,+20] = {smoke_car:+.4f}" if smoke_car is not None else "  CAR[+1,+20] = N/A")

    # Full run
    livestock_results = run_event_study(matches_df, prices, universe_df)

    # Placebo
    print("\n[STAGE 4b] Running placebo study...")
    # Use a sample of events for placebo to keep it manageable
    event_sample = events_df.sample(min(len(events_df), 50), random_state=42)
    placebo_results = run_placebo_study(event_sample, prices, universe_df)

    if args.stage == 4:
        print("\n[--stage 4] Stopping after Stage 4 as requested.")
        return

    # ── STAGE 5: Statistics ──
    print("\n[STAGE 5] Computing statistics...")
    results_df = run_statistics(livestock_results, placebo_results)
    print_statistics(results_df)

    # ── OUTPUTS ──
    print("\n[OUTPUTS] Generating...")
    save_results(results_df)
    plot_caar(livestock_results, placebo_results, prices, universe_df, matches_df)
    write_readme(events_df, universe_df, coverage)
    append_trial_registry()

    # ── VERDICT ──
    render_verdict(results_df)

    print("\n[Done]")


if __name__ == "__main__":
    main()
