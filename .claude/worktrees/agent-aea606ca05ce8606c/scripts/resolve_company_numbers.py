"""
Resolves company NAMES -> official Companies House numbers using the CH Search API.

Why this instead of hand-searching:
  - The number comes straight from Companies House (official, not a guess).
  - It returns the TOP candidates per name, so YOU pick the correct operating
    entity (full-accounts trading company) and avoid Topco/Midco/Holdco shells.

It does NOT auto-pick an entity. Picking operating-vs-holding is a judgement call
the script deliberately leaves to you (print shows status/type/address to help).

Setup:
  - Uses the SAME key you already have: env var COMPANIES_HOUSE_API_KEY
  - Auth = HTTP Basic (key as username, blank password)
  - Rate limit: 600 requests / 5 min  -> we sleep 0.6s between calls

Run:
  export COMPANIES_HOUSE_API_KEY=your_key   # (or put it in your .env and load it)
  python resolve_company_numbers.py

Output:
  company_numbers_candidates.csv  (one row per candidate, grouped by query)
  Columns: parent, query_name, rank, company_number, title, status, type, address
"""

import csv
import os
import time
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import requests

API_KEY = os.environ.get("COMPANIES_HOUSE_API_KEY")
if not API_KEY:
    sys.exit("Set COMPANIES_HOUSE_API_KEY first (the same key your pipeline uses).")

SEARCH_URL = "https://api.company-information.service.gov.uk/search/companies"
CANDIDATES_PER_NAME = 4          # top N matches to show per name
SLEEP_SECONDS = 0.6              # stay under 600 req / 5 min
OUT = Path(__file__).parent.parent / "company_numbers_candidates.csv"

# -----------------------------------------------------------------------------
# Names to resolve, grouped by the listed company they relate to.
# RED-flagged sectors (Luceco #11, Smiths News #15, McBride #26) are excluded.
# Foreign giants (TUI AG, P&G, Reckitt, etc.) are kept only where a UK entity is
# plausible; they are NOT distress-signal targets, so don't sweat imperfect hits.
# -----------------------------------------------------------------------------
TARGETS = {
    "OTB / On the Beach": [
        "On the Beach Group plc", "Loveholidays", "Jet2holidays", "TUI UK",
        "easyJet holidays", "Hays Travel", "Trailfinders", "Secret Escapes",
        "lastminute.com", "Thomas Cook", "Saga",
    ],
    "MACF / Macfarlane": [
        "Macfarlane Group PLC", "Kite Packaging", "Rajapack", "Antalis",
        "Springpack", "Fencor Packaging Group", "GWP Group", "Robinson plc",
    ],
    "MARS / Marston's": [
        "Marston's PLC", "Stonegate Group", "Greene King", "Mitchells and Butlers",
        "J D Wetherspoon", "Admiral Taverns", "Punch Pubs", "Star Pubs and Bars",
        "Young & Co's Brewery", "Loungers", "Shepherd Neame",
    ],
    "MER / Mears": [
        "Mears Group PLC", "Morgan Sindall Property Services", "Mitie",
        "Kier Group", "Sureserve", "Wates Group", "Pinnacle Group",
        "Fortem", "United Living", "Novus Property Solutions", "Renew Holdings",
    ],
    "CARD / Card Factory": [
        "Card Factory plc", "Esquire Retail", "Scribbler", "Hallmark Cards",
        "Moonpig", "WH Smith", "thortful",
    ],
    "FORT / Forterra": [
        "Forterra plc", "Ibstock", "Michelmersh Brick", "Wienerberger",
        "Aggregate Industries", "Tarmac Trading", "Naylor Industries",
    ],
    "BOOT / Henry Boot": [
        "Henry Boot PLC", "St Modwen", "Harworth Group", "Urban and Civic",
        "Watkin Jones", "Keepmoat",
    ],
    "NXR / Norcros": [
        "Norcros plc", "Aqualisa", "Mira Showers", "Bristan Group",
        "Grohe", "Methven", "Roman Limited", "Ideal Standard",
    ],
    "HFD / Halfords": [
        "Halfords Group plc", "Kwik-Fit", "Euro Car Parts", "GSF Car Parts",
        "ATS Euromaster", "Micheldever Tyre Services", "Evans Cycles",
        "Decathlon", "Blackcircles",
    ],
    "SHI / SIG": [
        "SIG plc", "Travis Perkins", "Grafton Group", "Stark Building Materials",
        "Saint-Gobain Building Distribution", "MKM Building Supplies",
        "Huws Gray", "Encon",
    ],
    "LSL / LSL Property": [
        "LSL Property Services", "Connells", "Savills", "The Property Franchise Group",
        "Spicerhaart", "M Winkworth", "Arun Estate Agencies", "Leaders Romans Group",
    ],
    "SFR / Severfield": [
        "Severfield plc", "William Hare Group", "Billington Holdings",
        "Caunton Engineering", "BHC Limited", "Bourne Group", "EvadX",
    ],
    "GYM / The Gym Group": [
        "The Gym Group plc", "Pure Gym", "JD Gyms", "Anytime Fitness",
        "Energie Fitness", "Snap Fitness", "David Lloyd Leisure",
        "Nuffield Health", "Bannatyne",
    ],
    "DFS / DFS Furniture": [
        "DFS Furniture plc", "ScS Group", "Furniture Village", "Oak Furnitureland",
        "Dreams Limited", "Wren Kitchens", "Dunelm", "IKEA",
    ],
    "VP & SDY / Equipment hire": [
        "Vp plc", "Speedy Hire", "GAP Group", "HSS Hire", "Nationwide Platforms",
        "Sunbelt Rentals", "Ardent Hire Solutions", "Boels Rental",
    ],
    "GLE & CRST / Housebuilders": [
        "MJ Gleeson", "Crest Nicholson", "Persimmon", "Barratt Redrow",
        "Taylor Wimpey", "Bellway", "Vistry Group", "Berkeley Group",
        "Miller Homes", "Keepmoat Homes", "Avant Homes", "Bloor Homes",
        "Hill Group", "Cala Homes",
    ],
    "FOXT / Foxtons": [
        "Foxtons Group", "Knight Frank", "Dexters", "Kinleigh Folkard and Hayward",
        "Chestertons", "Hamptons",
    ],
    "ECEL / Eurocell": [
        "Eurocell plc", "Epwin Group", "VEKA", "Deceuninck", "Liniar",
        "REHAU", "Profile 22",
    ],
    "FSTA / Fuller's": [
        "Fuller Smith and Turner", "City Pub Group", "Hall and Woodhouse",
        "Urban Pubs and Bars",
    ],
    "HEAD / Headlam": [
        "Headlam Group", "Likewise Group", "Victoria plc", "Tapi Carpets",
        "United Carpets", "Carpetright",
    ],
}


def search(name: str):
    """Return up to CANDIDATES_PER_NAME candidate companies for a name."""
    try:
        resp = requests.get(
            SEARCH_URL,
            params={"q": name, "items_per_page": CANDIDATES_PER_NAME},
            auth=(API_KEY, ""),
            timeout=20,
        )
        if resp.status_code == 429:
            # rate limited - back off and retry once
            time.sleep(20)
            resp = requests.get(
                SEARCH_URL,
                params={"q": name, "items_per_page": CANDIDATES_PER_NAME},
                auth=(API_KEY, ""),
                timeout=20,
            )
        resp.raise_for_status()
        return resp.json().get("items", [])
    except Exception as e:
        print(f"  ! error on '{name}': {e}")
        return []


def main():
    rows = []
    total = sum(len(v) for v in TARGETS.values())
    done = 0

    for parent, names in TARGETS.items():
        print(f"\n=== {parent} ===")
        for name in names:
            done += 1
            items = search(name)
            if not items:
                print(f"  [{done}/{total}] {name:40s} -> NO MATCH (check manually)")
                rows.append([parent, name, "", "", "", "", ""])
            else:
                for rank, it in enumerate(items, 1):
                    num = it.get("company_number", "")
                    title = it.get("title", "")
                    status = it.get("company_status", "")
                    ctype = it.get("company_type", "")
                    addr = (it.get("address_snippet", "") or "").replace("\n", " ")
                    rows.append([parent, name, rank, num, title, status, ctype, addr])
                    if rank == 1:
                        print(f"  [{done}/{total}] {name:40s} -> {num}  {title}  ({status})")
            time.sleep(SLEEP_SECONDS)

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["parent", "query_name", "rank", "company_number",
                    "title", "status", "type", "address"])
        w.writerows(rows)

    print(f"\nDone. {total} names searched -> {OUT.resolve()}")
    print("Next: open the CSV, and for each name pick the rank whose title/type/address")
    print("matches the real operating company (full accounts, NOT a Topco/Midco/Holdco).")


if __name__ == "__main__":
    main()
