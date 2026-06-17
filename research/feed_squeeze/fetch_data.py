"""
Data fetchers for the feed-cost-squeeze pipeline.

Each function returns a tidy pandas Series/DataFrame indexed on month-end dates
(or weekly dates for the SSB salmon price, which is resampled later).

Publication lags are NOT applied here; they are applied in build_panel.py so
the logic stays in one auditable place.

Sources
-------
  ONI        — NOAA CPC Oceanic Niño Index (monthly text file, public)
  Fishmeal   — World Bank Pink Sheet (monthly Excel, public)
  Salmon     — Statistics Norway (SSB) PxWeb/JSON-stat2 API (weekly)
  Equities   — yfinance weekly adjusted close
  Anchovy    — STUB: no clean programmatic source in v1
"""

import io
import time
import warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf

from .config import FARMER_TICKERS, REF_TICKER, START_DATE


# ── helpers ───────────────────────────────────────────────────────────────────

def _get(url: str, timeout: int = 60, retries: int = 3, delay: float = 2.0) -> requests.Response:
    """GET with simple retry/backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _post(url: str, payload: dict, timeout: int = 60, retries: int = 3, delay: float = 2.0) -> requests.Response:
    """POST JSON with simple retry/backoff."""
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# ── ONI ──────────────────────────────────────────────────────────────────────

# NOAA labels each row by the 3-month window; we centre it on the middle month.
_SEASON_CENTRE = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
    "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
    "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}


def fetch_oni() -> pd.Series:
    """
    NOAA CPC Oceanic Niño Index (ONI) — monthly series, returned as 'oni'.

    The ONI is a 3-month running mean of ERSST.v5 SST anomalies in the
    Niño 3.4 region (5°N–5°S, 170°W–120°W).

    Publication lag applied in build_panel.py: PUB_LAG['oni'] = 1 month.
    """
    url  = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    resp = _get(url)
    df   = pd.read_csv(io.StringIO(resp.text), sep=r"\s+", engine="python")
    # Normalise column names (file header may include leading space)
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SEAS"].isin(_SEASON_CENTRE)].copy()
    df["month"] = df["SEAS"].map(_SEASON_CENTRE)
    df["date"]  = pd.to_datetime(
        {"year": df["YR"].astype(int), "month": df["month"], "day": 1}
    ) + pd.offsets.MonthEnd(0)
    oni = df.set_index("date")["ANOM"].sort_index()
    oni.name = "oni"
    return oni[oni.index >= START_DATE].astype(float)


# ── FISHMEAL ─────────────────────────────────────────────────────────────────

_WB_PINK_SHEET_URL = (
    "https://thedocs.worldbank.org/en/doc/"
    "5d903e848db1d1b83e0ec8f744e55570-0350012021"
    "/related/CMO-Historical-Data-Monthly.xlsx"
)


def fetch_fishmeal() -> "pd.Series | None":
    """
    World Bank Pink Sheet — monthly fishmeal price (USD/metric ton).

    Downloads the CMO Historical Data Monthly Excel file and extracts the
    'fish meal' column from the 'Monthly Prices' sheet.  The Pink Sheet
    format varies by vintage; we try two layouts (dates-as-rows and
    dates-as-columns) and warn if neither works.

    Returns None with a warning if the file cannot be fetched or parsed.

    Publication lag applied in build_panel.py: PUB_LAG['fishmeal'] = 1 month.
    """
    try:
        resp = _get(_WB_PINK_SHEET_URL)
    except Exception as exc:
        warnings.warn(f"[fishmeal] Could not download World Bank Pink Sheet: {exc}")
        return None

    try:
        return _parse_pink_sheet(resp.content)
    except Exception as exc:
        warnings.warn(f"[fishmeal] Could not parse Pink Sheet: {exc}")
        return None


def _parse_pink_sheet(content: bytes) -> "pd.Series | None":
    """
    Pink Sheet layout (confirmed 2025 vintage):
      Row 0-3 : metadata text
      Row 4   : commodity names in columns 1..N  (col 0 is blank)
      Row 5   : unit labels
      Row 6+  : data; col 0 = "YYYYMmm" (e.g. "1960M01"), cols 1..N = prices
    """
    xl = pd.ExcelFile(io.BytesIO(content))
    sheet = next(
        (s for s in xl.sheet_names
         if "monthly" in s.lower() and "price" in s.lower()),
        xl.sheet_names[0],
    )
    raw = pd.read_excel(io.BytesIO(content), sheet_name=sheet, header=None)

    # Find the commodity-name row (first row whose col-1 is a non-empty string
    # and whose col-0 is NaN — i.e. the header row before data rows).
    header_row_idx = None
    for i, row in raw.iterrows():
        if pd.isna(row.iloc[0]) and isinstance(row.iloc[1], str) and row.iloc[1].strip():
            header_row_idx = i
            break

    if header_row_idx is None:
        warnings.warn("[fishmeal] Could not find commodity header row in Pink Sheet")
        return None

    commodity_names = raw.iloc[header_row_idx]   # Series: col index → name

    # Find the "Fish meal" column
    fish_col_idx = next(
        (idx for idx, val in commodity_names.items()
         if isinstance(val, str) and "fish" in val.lower()),
        None,
    )
    if fish_col_idx is None:
        warnings.warn("[fishmeal] Could not locate fishmeal column in Pink Sheet")
        return None

    # Data rows: col 0 is "YYYYMmm", fish_col_idx is price
    data_start = header_row_idx + 2   # skip header row and units row
    date_strs  = raw.iloc[data_start:, 0].astype(str)
    prices     = pd.to_numeric(raw.iloc[data_start:, fish_col_idx], errors="coerce")

    # Parse "1960M01" → datetime
    def _parse_ym(s: str) -> pd.Timestamp:
        try:
            year, month = s.split("M")
            return pd.Timestamp(int(year), int(month), 1)
        except Exception:
            return pd.NaT

    dates = pd.DatetimeIndex([_parse_ym(d) for d in date_strs])
    s = pd.Series(prices.values, index=dates, name="fishmeal_usd_mt")
    s = s[s.index.notna()].dropna().sort_index()
    s.index = s.index + pd.offsets.MonthEnd(0)
    return s[s.index >= START_DATE]


# ── SALMON PRICE (SSB) ───────────────────────────────────────────────────────

_SSB_TABLE_URL = "https://data.ssb.no/api/v0/en/table/03024/"


def fetch_ssb_salmon() -> "pd.Series | None":
    """
    Statistics Norway (SSB) weekly salmon export price, NOK/kg.
    Table 03024: 'Weekly prices for fresh salmon exported from Norway'.

    Variables (confirmed from API metadata):
      VareGrupper2 = "01"       Fish-farm bred salmon, fresh or chilled
      ContentsCode = "Kilopris" Price per kilo (NOK)
      Tid                       ISO week in Norwegian format "YYYYUww"

    Resampled to month-end (last weekly observation per month).
    Returns None with a warning if the API is unavailable or fails to parse.

    Publication lag applied in build_panel.py: PUB_LAG['salmon_price'] = 0 months.
    """
    query = {
        "query": [
            {"code": "VareGrupper2", "selection": {"filter": "item", "values": ["01"]}},
            {"code": "ContentsCode",  "selection": {"filter": "item", "values": ["Kilopris"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    try:
        resp = _post(_SSB_TABLE_URL, query)
    except Exception as exc:
        warnings.warn(f"[salmon] SSB API unavailable: {exc}")
        return None

    try:
        return _parse_ssb_jsonstat2(resp.json())
    except Exception as exc:
        warnings.warn(f"[salmon] Could not parse SSB response: {exc}")
        return None


def _parse_ssb_jsonstat2(data: dict) -> "pd.Series | None":
    """
    Convert a JSON-stat2 SSB response into a monthly salmon price Series.

    SSB week labels use Norwegian format "YYYYUww" (uke = week).
    We replace 'U' with 'W' and parse as ISO weeks (Friday = day 5).
    """
    dim_ids = data["id"]
    dims    = data["dimension"]
    values  = data["value"]

    # Time dimension is "Tid" in SSB tables
    tid_key = next(
        (k for k in dim_ids if k.lower() in ("tid", "time", "week", "uke")),
        dim_ids[0],
    )
    # Week labels in order
    week_labels = list(dims[tid_key]["category"]["index"].keys())

    def _uke_to_date(w: str) -> pd.Timestamp:
        # "2005U01" → "2005W01-5" → Friday of ISO week 1 of 2005
        iso = w.replace("U", "W")
        try:
            return pd.to_datetime(iso + "-5", format="%GW%V-%u")
        except (ValueError, TypeError):
            return pd.NaT

    dates = pd.DatetimeIndex([_uke_to_date(w) for w in week_labels])
    s = pd.Series(
        [float(v) if v is not None else np.nan for v in values],
        index=dates,
        name="salmon_price_nok_kg",
    )
    s = s[s.index.notna()].sort_index()
    s = s[s.index >= START_DATE]
    return s.resample("ME").last()


# ── EQUITIES ─────────────────────────────────────────────────────────────────

def fetch_equities(
    tickers: "list[str] | None" = None,
) -> "tuple[pd.DataFrame, dict]":
    """
    Fetch weekly adjusted-close prices for salmon farmer equities via yfinance,
    resample to month-end, and return a per-ticker coverage report.

    Tickers with no data are excluded from the returned DataFrame.
    Coverage dict maps ticker → (first_date, last_date, n_weekly_obs) | None.

    Publication lag applied in build_panel.py: PUB_LAG['equity'] = 0 months.
    """
    if tickers is None:
        tickers = FARMER_TICKERS + [REF_TICKER]

    monthly_frames: dict = {}
    coverage:       dict = {}

    for ticker in tickers:
        hist = _fetch_history(ticker)
        if hist is None or hist.empty:
            warnings.warn(f"[equity] No data for {ticker}")
            coverage[ticker] = None
            continue

        close = hist["Close"].dropna()
        # Strip timezone so all series share a tz-naive index
        close.index = close.index.tz_localize(None) if close.index.tz is not None else close.index
        monthly = close.resample("ME").last()
        monthly_frames[ticker] = monthly
        coverage[ticker] = (close.index[0].date(), close.index[-1].date(), len(close))

    if not monthly_frames:
        raise RuntimeError("No equity data could be fetched — check internet connectivity.")

    return pd.DataFrame(monthly_frames), coverage


def _fetch_history(
    ticker: str,
    max_tries: int = 3,
    delay: float   = 2.0,
) -> "pd.DataFrame | None":
    """yfinance history() with retry/backoff (mirrors fast_info convention in CLAUDE.md)."""
    t = yf.Ticker(ticker)
    for attempt in range(max_tries):
        try:
            hist = t.history(start=START_DATE, interval="1wk", auto_adjust=True)
            if not hist.empty:
                return hist
        except Exception:
            if attempt == max_tries - 1:
                return None
            time.sleep(delay)
    return None


# ── ANCHOVY QUOTA — STUB ──────────────────────────────────────────────────────
#
# Peruvian anchovy quota data (IMARPE) is not available via any clean
# programmatic API as of v1.  IMARPE publishes quota decisions through press
# releases; historical data must be hand-compiled from annual reports.
#
# When this stub is filled in, return:
#   pd.Series indexed by the quota ANNOUNCEMENT DATE (not the season start),
#   values in million metric tons, name='anchovy_quota_mmt'.
#   Apply PUB_LAG['anchovy'] = 0 (the quota is public on announcement day).
#   Join to the panel in build_panel.py before running the tests.
#
# The hardcoded table below is a placeholder from secondary sources:
_ANCHOVY_HARDCODED: dict = {
    # (year, season_num) → quota_million_mt
    # Season 1 = Apr–Jul,  Season 2 = Nov–Jan
    (2010, 1): 3.0,
    (2010, 2): 2.5,
    (2012, 1): 0.0,   # season prohibited
    (2014, 1): 1.9,   # reduced by El Niño
    (2023, 1): 1.9,   # reduced by El Niño Costero
}


def stub_anchovy() -> "tuple[None, dict]":
    """Return (None, hardcoded_reference_dict) — quota series unavailable in v1."""
    return None, _ANCHOVY_HARDCODED
