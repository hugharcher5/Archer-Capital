"""
Industry operating margins used as the mature-margin target for companies that
have no profitable history of their own.

Source: Aswath Damodaran, "Operating and Net Margins by Industry (US)",
January 2026 data set (pre-tax unadjusted operating margin), NYU Stern:
https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/margin.html

Yahoo Finance industry labels are matched to Damodaran's industries by keyword.
Order matters: the first matching keyword wins, so specific labels come before
general ones (e.g. "semiconductor equipment" before "semiconductor").
"""

from __future__ import annotations

# Total US market excluding financials, January 2026.  Used when the industry
# is unknown, unmatched, or its industry margin is not meaningful (≤ 0).
MARKET_MARGIN: float = 0.1314

# (Yahoo industry keyword, Damodaran industry, pre-tax operating margin)
_INDUSTRY_TABLE: list[tuple[str, str, float]] = [
    ("semiconductor equipment",   "Semiconductor Equip",                    0.2617),
    ("semiconductor",             "Semiconductor",                          0.3533),
    ("electronic gaming",         "Software (Entertainment)",               0.3385),
    ("software",                  "Software (System & Application)",        0.3298),
    ("internet content",          "Information Services",                   0.1189),
    ("information technology",    "Computer Services",                      0.0741),
    ("computer hardware",         "Computers/Peripherals",                  0.2248),
    ("communication equipment",   "Telecom. Equipment",                     0.2070),
    ("electronic components",     "Electronics (General)",                  0.1042),
    ("scientific & technical",    "Electronics (General)",                  0.1042),
    ("telecom",                   "Telecom. Services",                      0.2047),
    ("biotechnology",             "Drugs (Biotechnology)",                  0.0897),
    ("drug manufacturers",        "Drugs (Pharmaceutical)",                 0.2954),
    ("health information",        "Heathcare Information and Technology",   0.1471),
    ("medical care facilities",   "Hospitals/Healthcare Facilities",        0.1336),
    ("medical distribution",      "Healthcare Support Services",            0.0300),
    ("healthcare plans",          "Healthcare Support Services",            0.0300),
    ("medical",                   "Healthcare Products",                    0.1534),
    ("diagnostics",               "Healthcare Products",                    0.1534),
    ("auto parts",                "Auto Parts",                             0.0567),
    ("auto manufacturers",        "Auto & Truck",                           0.0232),
    ("auto & truck dealerships",  "Retail (Automotive)",                    0.0624),
    ("airlines",                  "Air Transport",                          0.0532),
    ("aerospace",                 "Aerospace/Defense",                      0.0865),
    ("railroads",                 "Transportation (Railroads)",             0.3741),
    ("trucking",                  "Trucking",                               0.0689),
    ("marine shipping",           "Shipbuilding & Marine",                  0.1260),
    ("freight",                   "Transportation",                         0.0757),
    ("solar",                     "Green & Renewable Energy",               0.1987),
    ("renewable",                 "Green & Renewable Energy",               0.1987),
    ("utilities",                 "Utility (General)",                      0.2349),
    ("oil & gas e&p",             "Oil/Gas (Production and Exploration)",   0.2542),
    ("oil & gas integrated",      "Oil/Gas (Integrated)",                   0.1125),
    ("oil & gas midstream",       "Oil/Gas Distribution",                   0.2578),
    ("oil & gas equipment",       "Oilfield Svcs/Equip.",                   0.0465),
    ("oil & gas refining",        "Oil/Gas (Integrated)",                   0.1125),
    ("gold",                      "Precious Metals",                        0.4039),
    ("silver",                    "Precious Metals",                        0.4039),
    ("steel",                     "Steel",                                  0.0410),
    ("metals",                    "Metals & Mining",                        0.2385),
    ("copper",                    "Metals & Mining",                        0.2385),
    ("aluminum",                  "Metals & Mining",                        0.2385),
    ("specialty chemicals",       "Chemical (Specialty)",                   0.1217),
    ("chemicals",                 "Chemical (Diversified)",                 0.0329),
    ("building materials",        "Building Materials",                     0.1264),
    ("residential construction",  "Homebuilding",                           0.1257),
    ("engineering & construction","Engineering/Construction",               0.0649),
    ("machinery",                 "Machinery",                              0.1586),
    ("electrical equipment",      "Electrical Equipment",                   0.0953),
    ("waste management",          "Environmental & Waste Services",         0.1461),
    ("packaging",                 "Packaging & Container",                  0.0958),
    ("lumber",                    "Paper/Forest Products",                  0.0634),
    ("paper",                     "Paper/Forest Products",                  0.0634),
    ("restaurants",               "Restaurant/Dining",                      0.1579),
    ("grocery",                   "Retail (Grocery and Food)",              0.0229),
    ("discount stores",           "Retail (General)",                       0.0680),
    ("department stores",         "Retail (General)",                       0.0680),
    ("internet retail",           "Retail (General)",                       0.0680),
    ("home improvement",          "Retail (Building Supply)",               0.1194),
    ("specialty retail",          "Retail (Special Lines)",                 0.0773),
    ("apparel",                   "Apparel",                                0.0911),
    ("footwear",                  "Shoe",                                   0.0903),
    ("beverages - non-alcoholic", "Beverage (Soft)",                        0.2053),
    ("beverages",                 "Beverage (Alcoholic)",                   0.2276),
    ("tobacco",                   "Tobacco",                                0.4354),
    ("packaged foods",            "Food Processing",                        0.1063),
    ("confectioners",             "Food Processing",                        0.1063),
    ("food distribution",         "Food Wholesalers",                       0.0261),
    ("farm products",             "Farming/Agriculture",                    0.0545),
    ("household",                 "Household Products",                     0.1862),
    ("furnishings",               "Furn/Home Furnishings",                  0.0659),
    ("entertainment",             "Entertainment",                          0.1060),
    ("broadcasting",              "Broadcasting",                           0.1233),
    ("publishing",                "Publishing & Newspapers",                0.0998),
    ("education",                 "Education",                              0.1401),
    ("lodging",                   "Hotel/Gaming",                           0.1939),
    ("resorts",                   "Hotel/Gaming",                           0.1939),
    ("gambling",                  "Hotel/Gaming",                           0.1939),
    ("travel services",           "Hotel/Gaming",                           0.1939),
    ("leisure",                   "Recreation",                             0.0969),
    ("staffing",                  "Business & Consumer Services",           0.1227),
    ("consulting",                "Business & Consumer Services",           0.1227),
    ("specialty business",        "Business & Consumer Services",           0.1227),
    ("asset management",          "Investments & Asset Management",         0.2550),
    ("credit services",           "Financial Svcs. (Non-bank & Insurance)", 0.1848),
    ("insurance",                 "Insurance (Prop/Cas.)",                  0.1525),
    ("reit",                      "R.E.I.T.",                               0.2464),
    ("real estate",               "Real Estate (General/Diversified)",      0.2145),
]


def industry_margin(industry: str) -> tuple[float, str]:
    """Return (operating margin, label describing where it came from)."""
    key = (industry or "").lower()
    if key:
        for kw, damodaran_name, margin in _INDUSTRY_TABLE:
            if kw in key and margin > 0:
                return margin, f"{damodaran_name} industry average (Damodaran, Jan 2026)"
    return MARKET_MARGIN, "US market average excl. financials (Damodaran, Jan 2026)"
