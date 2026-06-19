"""
Confirmed Companies House numbers for private competitor distress strategy.

Structure
---------
COMPETITORS : dict[group_key, dict]
    group_key  — matches the "parent" labels in company_numbers_candidates.csv
    tickers    — LSE tickers for the listed company/companies in this group
    companies  — list of {ch_number, name, notes}

Curation notes
--------------
- Thomas Cook (OTB group) dropped: no active UK entity post-2019 collapse.
- Star Pubs and Bars (MARS group) dropped: all CH hits are dissolved/liquidation.
- Stark Building Materials (SHI group) dropped: same CH number as Saint-Gobain
  Building Distribution (01647362) after 2022 acquisition — deduplicated.
- Loveholidays: 07839224 We Love Holidays Limited (operating co; avoid parent
  Wednesday Topco Limited 13540640).
- Fortem: 04638969 Fortem Solutions Limited (Willmott Dixon group — not ENGIE).
- St Modwen: reclassified from BOOT group to GLE & CRST / Housebuilders
  (09095920 St. Modwen Homes Limited).
- Carpetright: retained as 02294875 CPR Realisations Limited (administration)
  — live distress signal for HEAD / Headlam group.
- Bannatyne: SC197151 — Scottish-registered, SC prefix is normal.
- Grafton Group, JD Gyms, Snap Fitness, Oak Furnitureland, Hall & Woodhouse
  all confirmed manually from lower-ranked CH search hits.
"""

from __future__ import annotations

COMPETITORS: dict[str, dict] = {
    "OTB / On the Beach": {
        "tickers": ["OTB"],
        "companies": [
            {"ch_number": "09736592", "name": "On the Beach Group plc"},
            {"ch_number": "07839224", "name": "We Love Holidays Limited", "notes": "Loveholidays operating co"},
            {"ch_number": "04472486", "name": "Jet2holidays Limited"},
            {"ch_number": "02830117", "name": "TUI UK Limited"},
            {"ch_number": "11927917", "name": "easyJet Holidays Limited"},
            {"ch_number": "01990682", "name": "Hays Travel Limited"},
            {"ch_number": "01004502", "name": "Trailfinders Limited"},
            {"ch_number": "07026107", "name": "Secret Escapes Limited"},
            {"ch_number": "03852152", "name": "lastminute.com Limited"},
            {"ch_number": "08804263", "name": "Saga plc"},
        ],
    },
    "MACF / Macfarlane": {
        "tickers": ["MACF"],
        "companies": [
            {"ch_number": "SC004221", "name": "Macfarlane Group PLC"},
            {"ch_number": "04680835", "name": "Kite Packaging Limited"},
            {"ch_number": "03110319", "name": "Rajapack Limited"},
            {"ch_number": "01088345", "name": "Antalis Limited"},
            {"ch_number": "08775441", "name": "Springpack Limited"},
            {"ch_number": "03837995", "name": "Fencor Packaging Group Limited"},
            {"ch_number": "05459368", "name": "GWP Group Limited"},
            {"ch_number": "00039811", "name": "Robinson plc"},
        ],
    },
    "MARS / Marston's": {
        "tickers": ["MARS"],
        "companies": [
            {"ch_number": "00031461", "name": "Marston's PLC"},
            {"ch_number": "16893371", "name": "Stonegate Group Limited"},
            {"ch_number": "00024511", "name": "Greene King Limited"},
            {"ch_number": "04659443", "name": "Mitchells and Butlers plc"},
            {"ch_number": "01709784", "name": "J D Wetherspoon plc"},
            {"ch_number": "05438628", "name": "Admiral Taverns Limited"},
            {"ch_number": "14782545", "name": "Punch Pubs & Co Limited"},
            {"ch_number": "00032762", "name": "Young & Co's Brewery PLC"},
            {"ch_number": "11910770", "name": "Loungers plc"},
            {"ch_number": "00138256", "name": "Shepherd Neame Limited"},
        ],
    },
    "MER / Mears": {
        "tickers": ["MER"],
        "companies": [
            {"ch_number": "03232863", "name": "Mears Group PLC"},
            {"ch_number": "04415196", "name": "Morgan Sindall Property Services Limited"},
            {"ch_number": "02938041", "name": "Mitie Group plc"},
            {"ch_number": "02708030", "name": "Kier Group plc"},
            {"ch_number": "14843644", "name": "Sureserve Group plc"},
            {"ch_number": "01824828", "name": "Wates Group Limited"},
            {"ch_number": "04240859", "name": "Pinnacle Group Limited"},
            {"ch_number": "04638969", "name": "Fortem Solutions Limited", "notes": "Willmott Dixon group"},
            {"ch_number": "15574106", "name": "United Living Group Limited"},
            {"ch_number": "02403551", "name": "Novus Property Solutions Limited"},
            {"ch_number": "05485409", "name": "Renew Holdings plc"},
        ],
    },
    "CARD / Card Factory": {
        "tickers": ["CARD"],
        "companies": [
            {"ch_number": "09002747", "name": "Card Factory plc"},
            {"ch_number": "12338587", "name": "Esquire Retail Limited"},
            {"ch_number": "12796326", "name": "Scribbler Cards Limited"},
            {"ch_number": "03414540", "name": "Hallmark Cards Limited"},
            {"ch_number": "03852652", "name": "Moonpig Group plc"},
            {"ch_number": "05202036", "name": "WH Smith PLC"},
            {"ch_number": "09524310", "name": "Thortful Limited"},
        ],
    },
    "FORT / Forterra": {
        "tickers": ["FORT"],
        "companies": [
            {"ch_number": "09963666", "name": "Forterra plc"},
            {"ch_number": "09760850", "name": "Ibstock plc"},
            {"ch_number": "03433257", "name": "Michelmersh Brick Holdings plc"},
            {"ch_number": "05299520", "name": "Wienerberger Limited"},
            {"ch_number": "09634189", "name": "Aggregate Industries UK Limited"},
            {"ch_number": "00453791", "name": "Tarmac Trading Limited"},
            {"ch_number": "14694094", "name": "Naylor Industries Limited"},
        ],
    },
    "BOOT / Henry Boot": {
        "tickers": ["BOOT"],
        "companies": [
            {"ch_number": "00160996", "name": "Henry Boot PLC"},
            {"ch_number": "02649340", "name": "Harworth Group plc"},
            {"ch_number": "09123307", "name": "Urban & Civic plc"},
            {"ch_number": "09791105", "name": "Watkin Jones plc"},
            {"ch_number": "01998780", "name": "Keepmoat Limited"},
        ],
    },
    "NXR / Norcros": {
        "tickers": ["NXR"],
        "companies": [
            {"ch_number": "03691883", "name": "Norcros plc"},
            {"ch_number": "05222773", "name": "Aqualisa Products Limited"},
            {"ch_number": "11451703", "name": "Mira Showers Limited"},
            {"ch_number": "01318148", "name": "Bristan Group Limited"},
            {"ch_number": "00770795", "name": "Grohe Limited"},
            {"ch_number": "05222773", "name": "Methven UK Limited"},
            {"ch_number": "02184168", "name": "Roman Limited"},
            {"ch_number": "00533523", "name": "Ideal Standard Limited"},
        ],
    },
    "HFD / Halfords": {
        "tickers": ["HFD"],
        "companies": [
            {"ch_number": "04457314", "name": "Halfords Group plc"},
            {"ch_number": "14010889", "name": "Kwik Fit (GB) Limited"},
            {"ch_number": "07611358", "name": "Euro Car Parts Limited"},
            {"ch_number": "01779084", "name": "GSF Car Parts Limited"},
            {"ch_number": "04303731", "name": "ATS Euromaster Limited"},
            {"ch_number": "01817398", "name": "Micheldever Tyre Services Limited"},
            {"ch_number": "11577650", "name": "Evans Cycles Limited"},
            {"ch_number": "03140144", "name": "Decathlon UK Limited"},
            {"ch_number": "SC224310", "name": "Blackcircles.com Limited"},
        ],
    },
    "SHI / SIG": {
        "tickers": ["SHI"],
        "companies": [
            {"ch_number": "00998314", "name": "SIG plc"},
            {"ch_number": "00824821", "name": "Travis Perkins plc"},
            {"ch_number": "02886378", "name": "Grafton Group (UK) plc"},
            {"ch_number": "01647362", "name": "Saint-Gobain Building Distribution Limited", "notes": "Now trading as Stark UK after 2022 acquisition"},
            {"ch_number": "11609901", "name": "MKM Building Supplies Limited"},
            {"ch_number": "02506633", "name": "Huws Gray Limited"},
            {"ch_number": "03411533", "name": "Encon Insulation Limited"},
        ],
    },
    "LSL / LSL Property": {
        "tickers": ["LSL"],
        "companies": [
            {"ch_number": "05114014", "name": "LSL Property Services plc"},
            {"ch_number": "03187394", "name": "Connells Limited"},
            {"ch_number": "02122174", "name": "Savills plc"},
            {"ch_number": "08721920", "name": "The Property Franchise Group plc"},
            {"ch_number": "SC605833", "name": "Spicerhaart Limited"},
            {"ch_number": "01189557", "name": "M Winkworth plc"},
            {"ch_number": "02597969", "name": "Arun Estates Limited"},
            {"ch_number": "09939099", "name": "Leaders Romans Group Limited"},
        ],
    },
    "SFR / Severfield": {
        "tickers": ["SFR"],
        "companies": [
            {"ch_number": "01721262", "name": "Severfield plc"},
            {"ch_number": "02616280", "name": "William Hare Group Limited"},
            {"ch_number": "02402219", "name": "Billington Holdings plc"},
            {"ch_number": "00968729", "name": "Caunton Engineering Limited"},
            {"ch_number": "NI029649", "name": "BHC Limited"},
            {"ch_number": "09197897", "name": "Bourne Group Limited"},
            {"ch_number": "01791754", "name": "EvadX Limited"},
        ],
    },
    "GYM / The Gym Group": {
        "tickers": ["GYM"],
        "companies": [
            {"ch_number": "08528493", "name": "The Gym Group plc"},
            {"ch_number": "06690189", "name": "PureGym Limited"},
            {"ch_number": "08770057", "name": "JD Sports Gyms Limited"},
            {"ch_number": "SC339149", "name": "Anytime Fitness UK Limited"},
            {"ch_number": "12659023", "name": "Energie Fitness Group Limited"},
            {"ch_number": "07027212", "name": "Lift Brands UK Limited", "notes": "UK master franchisee for Snap Fitness"},
            {"ch_number": "01516226", "name": "David Lloyd Leisure Limited"},
            {"ch_number": "00576970", "name": "Nuffield Health"},
            {"ch_number": "SC197151", "name": "Bannatyne Ltd"},
        ],
    },
    "DFS / DFS Furniture": {
        "tickers": ["DFS"],
        "companies": [
            {"ch_number": "07236769", "name": "DFS Furniture plc"},
            {"ch_number": "03263435", "name": "ScS Group plc"},
            {"ch_number": "02307708", "name": "Furniture Village Limited"},
            {"ch_number": "12645185", "name": "Oak Furnitureland Group Limited"},
            {"ch_number": "08428347", "name": "Dreams Limited"},
            {"ch_number": "06799478", "name": "Wren Kitchens Limited"},
            {"ch_number": "05045977", "name": "Dunelm Group plc"},
            {"ch_number": "01986283", "name": "IKEA Limited"},
        ],
    },
    "VP & SDY / Equipment hire": {
        "tickers": ["VP", "SDY"],
        "companies": [
            {"ch_number": "00481833", "name": "Vp plc"},
            {"ch_number": "00927680", "name": "Speedy Hire plc"},
            {"ch_number": "00198823", "name": "GAP Group Limited"},
            {"ch_number": "04889012", "name": "HSS Hire Service Group Limited"},
            {"ch_number": "02268921", "name": "Nationwide Platforms Limited"},
            {"ch_number": "00444569", "name": "Sunbelt Rentals Limited"},
            {"ch_number": "03987596", "name": "Ardent Hire Solutions Limited"},
            {"ch_number": "03542206", "name": "Boels Rental Limited"},
        ],
    },
    "GLE & CRST / Housebuilders": {
        "tickers": ["MJG", "CRST"],
        "companies": [
            {"ch_number": "09268016", "name": "MJ Gleeson plc"},
            {"ch_number": "01040616", "name": "Crest Nicholson Holdings plc"},
            {"ch_number": "01818486", "name": "Persimmon plc"},
            {"ch_number": "00604574", "name": "Barratt Redrow plc"},
            {"ch_number": "00296805", "name": "Taylor Wimpey plc"},
            {"ch_number": "01372603", "name": "Bellway plc"},
            {"ch_number": "00306718", "name": "Vistry Group plc"},
            {"ch_number": "01454064", "name": "Berkeley Group Holdings plc"},
            {"ch_number": "SC255429", "name": "Miller Homes Limited"},
            {"ch_number": "02207338", "name": "Keepmoat Homes Limited"},
            {"ch_number": "03215228", "name": "Avant Homes Group Limited"},
            {"ch_number": "02162561", "name": "Bloor Homes Limited"},
            {"ch_number": "12714205", "name": "The Hill Group (Contractors) Limited"},
            {"ch_number": "SC074857", "name": "Cala Homes Limited"},
            {"ch_number": "09095920", "name": "St. Modwen Homes Limited", "notes": "Blackstone-owned since 2021"},
        ],
    },
    "FOXT / Foxtons": {
        "tickers": ["FOXT"],
        "companies": [
            {"ch_number": "07108742", "name": "Foxtons Group plc"},
            {"ch_number": "OC305934", "name": "Knight Frank LLP"},
            {"ch_number": "16796871", "name": "Dexters Estate Agent Group Limited"},
            {"ch_number": "02965708", "name": "Kinleigh Folkard & Hayward Limited"},
            {"ch_number": "12071118", "name": "Chestertons Limited"},
            {"ch_number": "04826001", "name": "Hamptons International Limited"},
        ],
    },
    "ECEL / Eurocell": {
        "tickers": ["ECEL"],
        "companies": [
            {"ch_number": "08654028", "name": "Eurocell plc"},
            {"ch_number": "07742256", "name": "Epwin Group plc"},
            {"ch_number": "01626563", "name": "VEKA UK Limited"},
            {"ch_number": "01565521", "name": "Deceuninck Limited"},
            {"ch_number": "03360857", "name": "Liniar Limited"},
            {"ch_number": "00722004", "name": "REHAU Limited"},
            {"ch_number": "02467789", "name": "Profile 22 Limited"},
        ],
    },
    "FSTA / Fuller's": {
        "tickers": ["FSTA"],
        "companies": [
            {"ch_number": "00241882", "name": "Fuller Smith & Turner PLC"},
            {"ch_number": "07814568", "name": "City Pub Group plc"},
            {"ch_number": "00057696", "name": "Hall & Woodhouse Limited"},
            {"ch_number": "15449818", "name": "Urban Pubs and Bars Limited"},
        ],
    },
    "HEAD / Headlam": {
        "tickers": ["HEAD"],
        "companies": [
            {"ch_number": "00460129", "name": "Headlam Group plc"},
            {"ch_number": "08010067", "name": "Likewise Group plc"},
            {"ch_number": "00282204", "name": "Victoria plc"},
            {"ch_number": "16966223", "name": "Tapi Carpets & Floors Limited"},
            {"ch_number": "08053659", "name": "United Carpets Group plc"},
            {"ch_number": "02294875", "name": "CPR Realisations Limited", "notes": "Carpetright — in administration since 2023; live distress signal"},
        ],
    },
}


def all_company_numbers() -> list[str]:
    """Return deduplicated list of all confirmed CH numbers."""
    seen: set[str] = set()
    result: list[str] = []
    for group in COMPETITORS.values():
        for co in group["companies"]:
            n = co["ch_number"]
            if n not in seen:
                seen.add(n)
                result.append(n)
    return result


def companies_for_ticker(ticker: str) -> list[dict]:
    """Return competitor list for a given LSE ticker."""
    for group in COMPETITORS.values():
        if ticker in group["tickers"]:
            return group["companies"]
    return []
