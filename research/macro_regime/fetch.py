"""
Data acquisition for the Macro Regime Classifier (infrastructure, not a DSR
trial -- see README.md for the locked methodology and series-choice notes).

FRED: GDP (A191RO1Q156NBEA) and Core CPI (CPILFESL) are pulled via ALFRED
vintage endpoints (get_series_all_releases) so the classifier can use each
observation's true first-release value, gated on its own real-world release
date -- no look-ahead. Every other FRED series is same-day/next-day public
with no meaningful revision lag and is pulled as a plain series.

Tiingo: SPY + IEF daily prices, for the stock-bond correlation input to the
Precarious flag (not available on FRED).
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests
from fredapi import Fred

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)


def _load_dotenv() -> None:
    env_path = BASE.parents[1] / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()
TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "").strip()


def _cache_is_fresh(cache_file: Path) -> bool:
    """
    Cache is 'fresh' only if last written today. This project elsewhere
    caches point-in-time historical data forever (it never changes), but a
    live regime read needs today's VIX/DFF/BAA10Y/GDP-vintage etc. -- a
    forever-cache would silently freeze the live read at whatever day it was
    first built. Daily refresh keeps the historical-backtest reuse benefit
    (fast within a session/day) while staying correct for Part 3.
    """
    if not cache_file.exists():
        return False
    mtime = pd.Timestamp.fromtimestamp(cache_file.stat().st_mtime).normalize()
    return mtime == pd.Timestamp.today().normalize()

# Plain (non-vintage) series: same-day/next-day public, no PIT gating needed.
PLAIN_SERIES = {
    "INDPRO": "industrial_production",
    "RSAFS": "retail_sales",
    "CFNAI": "cfnai",             # PMI proxy (NAPM discontinued on FRED)
    "ICSA": "jobless_claims",
    "UMCSENT": "consumer_sentiment",  # confidence proxy (Conference Board stale since 2024)
    "DCOILWTICO": "wti_oil",
    "T10Y2Y": "curve_10y2y",
    "T10Y3M": "curve_10y3m",
    "BAA10Y": "credit_spread_baa10y",   # primary credit spread (HY OAS truncated to 3yr by FRED)
    "VIXCLS": "vix",
    "DFF": "fedfunds_daily",
    "DTWEXBGS": "usd_broad",
    "UNRATE": "unemployment_rate",
    "BAMLH0A0HYM2": "hy_oas_supplementary",  # 2023+ only, disclosed truncation, cross-check only
}

# PIT vintage series: drive the actual coincident regime label. GDPC1 (the
# level series) is used instead of the pre-computed YoY transform
# (A191RO1Q156NBEA) because ALFRED only has genuine per-quarter vintage
# granularity for the YoY transform from 2014-09-26 onward (everything
# earlier is bunched into one artificial "release date") -- GDPC1 has real
# per-quarter vintages from ~1991-12 onward, which comfortably covers the
# 2015+ window. YoY is computed manually from GDPC1's first-release levels.
VINTAGE_SERIES = {
    "GDPC1": "gdp_level_pit",
    "CPILFESL": "core_cpi_index_pit",
}


def fetch_plain_series(series_id: str) -> pd.DataFrame:
    """Latest-revision value, full history. No PIT gating (same-day public data)."""
    cache_file = CACHE / f"plain_{series_id}.csv"
    if _cache_is_fresh(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"])

    fred = Fred(api_key=FRED_API_KEY)
    s = fred.get_series(series_id).dropna()
    df = pd.DataFrame({"date": s.index, "value": s.values})
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  {series_id}: {len(df)} obs, {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def fetch_vintage_series(series_id: str) -> pd.DataFrame:
    """
    Full ALFRED vintage history: one row per (period, realtime_start, value).
    Used to reconstruct the true first-release value for each period, gated
    on the date it actually became public -- no look-ahead. Uses fredapi's
    get_series_all_releases (handles ALFRED's vintage_dates pagination).
    """
    cache_file = CACHE / f"vintage_{series_id}.csv"
    if _cache_is_fresh(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date", "realtime_start"])

    fred = Fred(api_key=FRED_API_KEY)
    df = fred.get_series_all_releases(series_id)
    df = df.rename(columns={"realtime_start": "realtime_start", "date": "date", "value": "value"})
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df = df.sort_values(["date", "realtime_start"]).reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    n_periods = df["date"].nunique()
    print(f"  {series_id}: {len(df)} vintage rows, {n_periods} distinct periods, "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def first_release_series(vintage_df: pd.DataFrame) -> pd.Series:
    """
    Collapse a vintage DataFrame to one row per period: the FIRST published
    value for that period, indexed by realtime_start (the date it became
    public). This is the PIT-correct series to use for classification.
    """
    first = vintage_df.sort_values("realtime_start").groupby("date", as_index=False).first()
    first = first.sort_values("date").reset_index(drop=True)
    s = pd.Series(first["value"].values, index=pd.to_datetime(first["date"]), name="value")
    s.attrs["realtime_start"] = pd.Series(
        first["realtime_start"].values, index=pd.to_datetime(first["date"])
    )
    return s


def _tiingo_get(url, params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=(10, 20))
        except Exception as e:
            print(f"  network error: {e}")
            time.sleep(2)
            continue
        if r.status_code == 429:
            print("  429 rate limit -- sleeping 30s")
            time.sleep(30)
            continue
        return r
    return None


def fetch_tiingo(ticker: str) -> pd.DataFrame:
    cache_file = CACHE / f"tiingo_{ticker}.csv"
    if _cache_is_fresh(cache_file):
        return pd.read_csv(cache_file, parse_dates=["date"])

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": "2002-01-01",
        "endDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "token": TIINGO_KEY,
        "columns": "date,close,adjClose",
    }
    r = _tiingo_get(url, params)
    if r is None or r.status_code != 200:
        raise RuntimeError(f"Tiingo fetch failed for {ticker}: {r.status_code if r else 'no response'}")
    data = r.json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[["date", "close", "adjClose"]].sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    print(f"  {ticker}: {len(df)} obs, {df['date'].min().date()} -> {df['date'].max().date()}")
    return df


def fetch_all() -> dict:
    print("Fetching PIT vintage series (GDP, Core CPI)...")
    vintages = {sid: fetch_vintage_series(sid) for sid in VINTAGE_SERIES}

    print("Fetching plain series...")
    plains = {sid: fetch_plain_series(sid) for sid in PLAIN_SERIES}

    print("Fetching Tiingo SPY/IEF...")
    tiingo = {t: fetch_tiingo(t) for t in ("SPY", "IEF")}

    return {"vintages": vintages, "plains": plains, "tiingo": tiingo}


if __name__ == "__main__":
    fetch_all()
    print("Done.")
