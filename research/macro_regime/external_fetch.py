"""
Data acquisition for the three external nowcast/consensus signals (see
README.md's "External nowcast/consensus signals" section for sourcing and
methodology notes). This replaces a from-scratch custom nowcast build.

- GDPNow: reuses fetch.py's existing ALFRED vintage infra (FRED `GDPNOW`).
- Cleveland Fed Inflation Nowcasting: scraped JSON endpoint (undocumented,
  found via page source, CC-BY 4.0). Cached PERMANENTLY (not the daily-
  refresh convention used elsewhere) because this source is fragile -- if
  the endpoint disappears, the raw file already on disk keeps this analysis
  reproducible.
- SPF: direct xlsx downloads from the Philadelphia Fed, also cached
  permanently (stable published dataset, low risk, but no reason to
  re-download every run either).
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from fetch import fetch_vintage_series, first_release_series

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

CLEVELAND_JSON_URL = (
    "https://www.clevelandfed.org/-/media/files/webcharts/inflationnowcasting/nowcast_month.json"
)
SPF_MEDIAN_LEVEL_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/medianLevel.xlsx"
)
SPF_MEDIAN_GROWTH_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/medianGrowth.xlsx"
)
SPF_MICRODATA_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/historical-data/SPFmicrodata.xlsx"
)
SPF_RELEASE_DATES_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/spf-release-dates.txt"
)
NYFED_HISTORICAL_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/policy/nowcast/"
    "New-York-Fed-Staff-Nowcast_data_2002-2021.xlsx"
)
NYFED_CURRENT_URL = (
    "https://www.newyorkfed.org/medialibrary/Research/Interactives/Data/NowCast/"
    "Downloads/New-York-Fed-Staff-Nowcast_download_data.xlsx"
)
NYFED_REALTIME_START = pd.Timestamp("2016-01-01")  # per NY Fed's own blue/red-shading disclosure
_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_gdpnow_vintages() -> pd.DataFrame:
    """Full ALFRED vintage history for GDPNOW -- same infra as GDP/CPI."""
    return fetch_vintage_series("GDPNOW")


def _download_raw(url: str, cache_file: Path) -> Path:
    if cache_file.exists():
        return cache_file
    r = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=30)
    r.raise_for_status()
    cache_file.write_bytes(r.content)
    time.sleep(0.5)
    return cache_file


def fetch_cleveland_raw() -> Path:
    return _download_raw(CLEVELAND_JSON_URL, CACHE / "cleveland_nowcast_month_raw.json")


def fetch_spf_raw() -> tuple[Path, Path]:
    level = _download_raw(SPF_MEDIAN_LEVEL_URL, CACHE / "spf_medianLevel_raw.xlsx")
    growth = _download_raw(SPF_MEDIAN_GROWTH_URL, CACHE / "spf_medianGrowth_raw.xlsx")
    return level, growth


def fetch_spf_microdata_raw() -> Path:
    return _download_raw(SPF_MICRODATA_URL, CACHE / "spf_microdata_raw.xlsx")


def fetch_spf_release_dates_raw() -> Path:
    return _download_raw(SPF_RELEASE_DATES_URL, CACHE / "spf_release_dates_raw.txt")


def fetch_nyfed_historical_raw() -> Path:
    """2002-2021 file -- frozen/static (the Nowcast was suspended in 2021,
    this file will never be updated again), permanent cache is correct."""
    return _download_raw(NYFED_HISTORICAL_URL, CACHE / "nyfed_historical_raw.xlsx")


def fetch_nyfed_current_raw() -> Path:
    """2022-present file -- updates weekly, so use the daily-freshness cache
    convention (like fetch.py's plain/vintage series) instead of permanent."""
    from fetch import _cache_is_fresh
    cache_file = CACHE / "nyfed_current_raw.xlsx"
    if _cache_is_fresh(cache_file):
        return cache_file
    r = requests.get(NYFED_CURRENT_URL, headers={"User-Agent": _BROWSER_UA}, timeout=30)
    r.raise_for_status()
    cache_file.write_bytes(r.content)
    return cache_file


def _parse_nyfed_sheet(path: Path) -> pd.DataFrame:
    """'Forecasts By Quarter' sheet: wide matrix, rows=weekly forecast
    dates, columns=target quarter, values=nowcast for that quarter as of
    that week. Returns long format: forecast_date, target_quarter, value."""
    raw = pd.read_excel(path, sheet_name="Forecasts By Quarter", engine="calamine", header=None)
    header_row_idx = None
    for i in range(20):
        if "Forecast Date" in raw.iloc[i].values:
            header_row_idx = i
            break
    if header_row_idx is None:
        raise ValueError(f"Could not find 'Forecast Date' header row in {path}")

    df = pd.read_excel(path, sheet_name="Forecasts By Quarter", engine="calamine", header=header_row_idx)
    df = df.rename(columns={"Forecast Date": "forecast_date"})
    quarter_cols = [c for c in df.columns if c != "forecast_date"]

    long = df.melt(id_vars="forecast_date", value_vars=quarter_cols,
                    var_name="target_quarter_str", value_name="value")
    long = long.dropna(subset=["value"])
    long["forecast_date"] = pd.to_datetime(long["forecast_date"])
    long["target_quarter"] = pd.PeriodIndex(long["target_quarter_str"], freq="Q").to_timestamp()
    return long[["forecast_date", "target_quarter", "value"]].sort_values(
        ["target_quarter", "forecast_date"]
    ).reset_index(drop=True)


def parse_nyfed_nowcast() -> pd.DataFrame:
    """
    Combines both files into one long series, tagged by whether each row is
    genuinely real-time-published or a retrospective model reconstruction
    (per the NY Fed's own disclosure: 2016:Q1+ in the historical file, and
    everything in the current file, is real-time; 2002:Q1-2015:Q4 in the
    historical file is a reconstruction). No imputation across the
    2021:Q4-2023:Q3 suspension gap -- it's left as a genuine gap.
    """
    hist_path = fetch_nyfed_historical_raw()
    current_path = fetch_nyfed_current_raw()

    hist = _parse_nyfed_sheet(hist_path)
    current = _parse_nyfed_sheet(current_path)

    combined = pd.concat([hist, current], ignore_index=True)
    combined = combined.drop_duplicates(subset=["forecast_date", "target_quarter"], keep="last")
    combined["is_real_time"] = combined["target_quarter"] >= NYFED_REALTIME_START
    return combined.sort_values(["target_quarter", "forecast_date"]).reset_index(drop=True)


def parse_cleveland_nowcast(raw_path: Path, series_name: str = "Core CPI Inflation") -> pd.DataFrame:
    """
    Each element of the JSON list is one target month's rolling nowcast
    chart (subcaption 'YYYY-M'), tracking that month's MoM inflation
    nowcast daily until the actual print is released (at which point the
    nowcast series goes blank and an 'Actual ...' series fills in on the
    release date). Returns one row per target month with:
      - terminal_nowcast: the last non-empty nowcast value before release
      - terminal_date: the calendar date of that last value
      - actual: the realized value (if present in this file)
    """
    data = json.loads(raw_path.read_text())
    rows = []
    for chart in data:
        subcap = chart["chart"]["subcaption"]
        target_year, target_month = (int(x) for x in subcap.split("-"))

        cats_raw = [c["label"] for c in chart["categories"][0]["category"]]
        dates = []
        year_cursor = target_year
        prev_mm = None
        for label in cats_raw:
            m = re.match(r"^(\d{2})/(\d{2})$", label)
            if m:
                mm, dd = int(m.group(1)), int(m.group(2))
                if prev_mm is not None and mm < prev_mm:
                    year_cursor += 1
                prev_mm = mm
                dates.append(pd.Timestamp(year=year_cursor, month=mm, day=dd))
            else:
                dates.append(None)  # milestone marker (e.g. "CPI Mar"), not a date

        series_map = {ds["seriesname"]: ds["data"] for ds in chart["dataset"]}
        if series_name not in series_map:
            continue
        nowcast_vals = [d.get("value", "") for d in series_map[series_name]]
        actual_series_name = f"Actual {series_name}"
        actual_vals = [d.get("value", "") for d in series_map.get(actual_series_name, [])]

        terminal_date, terminal_val = None, None
        for dt, v in zip(dates, nowcast_vals):
            if dt is not None and v not in ("", None):
                terminal_date, terminal_val = dt, float(v)

        actual_val, actual_date = None, None
        for dt, v in zip(dates, actual_vals):
            if dt is not None and v not in ("", None):
                actual_date, actual_val = dt, float(v)

        rows.append({
            "target_month": pd.Timestamp(year=target_year, month=target_month, day=1),
            "terminal_nowcast": terminal_val,
            "terminal_nowcast_date": terminal_date,
            "actual": actual_val,
            "actual_date": actual_date,
        })

    df = pd.DataFrame(rows).sort_values("target_month").reset_index(drop=True)
    return df


def parse_spf_cpi(level_path: Path) -> pd.DataFrame:
    """SPF CORECPI sheet: annualized quarterly % change forecasts, CPI1 (current
    quarter, i.e. the nowcast made at survey time) through CPI6 (5Q ahead)."""
    df = pd.read_excel(level_path, sheet_name="CORECPI", engine="calamine")
    df["period"] = pd.PeriodIndex.from_fields(year=df["YEAR"], quarter=df["QUARTER"], freq="Q").to_timestamp()
    return df.sort_values("period").reset_index(drop=True)


def parse_spf_rgdp(growth_path: Path) -> pd.DataFrame:
    """SPF RGDP sheet: median real GDP growth forecasts, drgdp2 (1Q ahead of
    the survey's current quarter) through drgdp6."""
    df = pd.read_excel(growth_path, sheet_name="RGDP", engine="calamine")
    df["period"] = pd.PeriodIndex.from_fields(year=df["YEAR"], quarter=df["QUARTER"], freq="Q").to_timestamp()
    return df.sort_values("period").reset_index(drop=True)


def parse_spf_individual_rgdp(microdata_path: Path) -> pd.DataFrame:
    """
    Individual-forecaster RGDP microdata: LEVELS (RGDP1 = current-quarter
    nowcast at survey time, RGDP2 = 1Q-ahead forecast), unlike the
    pre-computed median growth-rate file. Implied 1Q-ahead annualized growth
    forecast = (RGDP2/RGDP1)^4 - 1, computed here per (ID, period) row.
    """
    df = pd.read_excel(microdata_path, sheet_name="RGDP", engine="calamine")
    df["period"] = pd.PeriodIndex.from_fields(year=df["YEAR"], quarter=df["QUARTER"], freq="Q").to_timestamp()
    df["period_forecasted"] = df["period"] + pd.DateOffset(months=3)
    df["forecast_growth_next_q"] = (df["RGDP2"] / df["RGDP1"]) ** 4 - 1.0
    df["forecast_growth_next_q"] *= 100
    out = df[["ID", "period", "period_forecasted", "forecast_growth_next_q"]].dropna()
    return out.sort_values(["ID", "period"]).reset_index(drop=True)


def parse_spf_release_dates(raw_path: Path) -> pd.DataFrame:
    """
    Exact SPF survey deadline/release dates (1990:Q2 onward -- Philly Fed's
    own file states earlier dates aren't known). Used for real date-based
    point-in-time gating of forecaster rankings instead of a quarter-count
    heuristic.
    """
    rows = []
    year = None
    pattern = re.compile(
        r"^\s*(\d{4})?\s+Q(\d)\s+(\d{1,2}/\d{1,2}/\d{2,4})\*{0,3}\s+(\d{1,2}/\d{1,2}/\d{2,4})\*{0,3}\s*$"
    )
    for line in raw_path.read_text().splitlines():
        m = pattern.match(line)
        if not m:
            continue
        if m.group(1):
            year = int(m.group(1))
        rows.append({
            "year": year, "quarter": int(m.group(2)),
            "deadline_date": pd.to_datetime(m.group(3), format="%m/%d/%y"),
            "release_date": pd.to_datetime(m.group(4), format="%m/%d/%y"),
        })
    df = pd.DataFrame(rows)
    df["period"] = pd.PeriodIndex.from_fields(year=df["year"], quarter=df["quarter"], freq="Q").to_timestamp()
    return df.sort_values("period").reset_index(drop=True)


if __name__ == "__main__":
    gdpnow = fetch_gdpnow_vintages()
    print(f"GDPNow vintages: {len(gdpnow)} rows, {gdpnow['date'].nunique()} quarters")

    cleveland_raw = fetch_cleveland_raw()
    cleveland = parse_cleveland_nowcast(cleveland_raw)
    print(f"\nCleveland Nowcast: {len(cleveland)} target months")
    print(cleveland.tail(8).to_string())

    level_path, growth_path = fetch_spf_raw()
    spf_cpi = parse_spf_cpi(level_path)
    spf_rgdp = parse_spf_rgdp(growth_path)
    print(f"\nSPF CORECPI: {len(spf_cpi)} quarters")
    print(spf_cpi.tail(5).to_string())
    print(f"\nSPF RGDP: {len(spf_rgdp)} quarters")
    print(spf_rgdp.tail(5).to_string())
