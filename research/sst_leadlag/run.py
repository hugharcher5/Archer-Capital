#!/usr/bin/env python3
"""
Sea-surface-temperature lead-lag signal on salmon farmers.
research/sst_leadlag/run.py

Hypothesis chain (two links tested separately):
  Link A  SST anomaly → sea-lice escalation (2–6 weeks lead)
  Link B  SST/lice signal → farmer RELATIVE return (farmer minus basket)

Relative returns strip the common salmon-price factor; lice is operational
cost, so its signal should appear in cross-sectional dispersion, not levels.

Data sources (all public; degraded gracefully):
  SST       — NOAA OISST v2.1 via ERDDAP (daily 0.25°), 1982–present
  Sea lice  — Barentswatch fish-health API (weekly per locality/zone)
  Price     — SSB table 03024 (weekly, already confirmed working)
  Equities  — yfinance weekly (.OL tickers)

Run once:
    venv/bin/python research/sst_leadlag/run.py

Do NOT tune parameters to improve results.
"""

import sys, os, io, time, warnings, re
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

SST_START      = "2012-01-01"   # align with lice data availability
EQUITY_START   = "2012-01-01"
LAGS_A         = list(range(0, 7))   # weeks: SST anomaly → lice
HORIZONS_B     = list(range(1, 9))   # weeks ahead: signal → relative return
FARMER_TICKERS = ["MOWI.OL", "SALM.OL", "LSG.OL", "BAKKA.OL", "GSF.OL"]

# NOAA OISST ERDDAP dataset
ERDDAP_BASE    = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
SST_DATASET    = "ncdcOisst21Agg_LonPM180"

# Norwegian production zones (PO1–PO11):
# centroid lat/lon placed in coastal marine water (avoids OISST land mask).
# Zones 12-13 (Jan Mayen, Svalbard) have no material salmon farming → excluded.
#
# Lat/lon are snapped below to the nearest OISST 0.25° grid centre (offset 0.125°).
PRODUCTION_ZONES = {
    1:  {"name": "PO1 Rogaland S",         "lat": 58.50, "lon":  5.00},
    2:  {"name": "PO2 Hardanger/Ryfylke",  "lat": 59.75, "lon":  5.00},
    3:  {"name": "PO3 Hordaland",          "lat": 60.50, "lon":  4.75},
    4:  {"name": "PO4 Sogn og Fjordane",   "lat": 61.75, "lon":  4.75},
    5:  {"name": "PO5 Møre og Romsdal",    "lat": 62.50, "lon":  5.75},
    6:  {"name": "PO6 Trøndelag",          "lat": 64.00, "lon":  9.75},
    7:  {"name": "PO7 Helgeland S",        "lat": 65.75, "lon": 12.25},
    8:  {"name": "PO8 Helgeland N",        "lat": 67.25, "lon": 14.50},
    9:  {"name": "PO9 Troms S",            "lat": 69.00, "lon": 17.50},
   10:  {"name": "PO10 Troms N",           "lat": 70.00, "lon": 21.00},
   11:  {"name": "PO11 Finnmark E",        "lat": 70.50, "lon": 26.00},
}

# Farmer-to-zone weights (approximate; based on public annual reports).
# Weights sum to ~1 for each farmer's Norwegian operations.
# BAKKA.OL (Bakkafrost) operates in the Faroe Islands — no Norwegian zone weights.
# It is included in the basket for diversification but excluded from zone signal.
FARMER_ZONES = {
    "MOWI":  {1:0.09, 2:0.09, 3:0.09, 4:0.09, 5:0.09, 6:0.09,
              7:0.09, 8:0.09, 9:0.09,10:0.10},  # diversified national footprint
    "SALM":  {6:0.30, 7:0.35, 8:0.35},           # concentrated Trøndelag/Helgeland
    "LSG":   {3:0.25, 4:0.25, 5:0.25, 6:0.25},   # Western Norway + Trøndelag
    "BAKKA": {},                                    # Faroe Islands — no PO mapping
    "GSF":   {3:0.35, 4:0.35, 5:0.30},            # SW Norway + BC + Shetland (NOR share)
}

# Publication lags (weeks)
# SST:   OISST daily files posted within 1-2 days; lag 0 for weekly signal.
# Lice:  Barentswatch publishes weekly counts ~1 week after collection. Lag 1.
# Price: SSB weekly, within 2 weeks; lag 0.
# Equity: live daily prices; lag 0.
PUB_LAG = {"sst": 0, "lice": 1, "price": 0, "equity": 0}

# Signal thresholds
SST_ANOMALY_THRESHOLD = 0.5    # °C above seasonal baseline
SIGNAL_COST_BPS       = 10     # PLACEHOLDER round-trip cost
WARM_EPISODE_MIN_WEEKS = 3     # min duration to count as independent episode


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _rule(n=72):  print("─" * n)
def _hdr(t):      _rule(); print(f"  {t}"); _rule()
def _note(msg):   print(f"  NOTE: {msg}")
def _warn(msg):   print(f"  WARNING: {msg}")
def _skip(msg):   print(f"  [SKIP] {msg}")


def _snap_oisst(val):
    """Snap a coordinate to the nearest OISST 0.25° grid centre (offset 0.125°)."""
    return round((val - 0.125) * 4) / 4 + 0.125


def _http_get(url, timeout=120):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(10)


def _ols(x: pd.Series, y: pd.Series):
    """OLS with NaN-safe masking. Returns (coef, t_stat, r2, n)."""
    mask = x.notna() & y.notna()
    n    = int(mask.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, n
    sl, ic, rv, pv, se = stats.linregress(x[mask].values, y[mask].values)
    tstat = sl / se if se > 0 else np.nan
    return round(sl, 6), round(tstat, 3), round(rv**2, 4), n


# ─────────────────────────────────────────────────────────────────────────────
# SST FETCH — NOAA OISST via ERDDAP
# ─────────────────────────────────────────────────────────────────────────────

def fetch_sst_zone(zone_id: int, zone_info: dict) -> "pd.Series | None":
    """
    Fetch daily OISST SST for a single production zone centroid.

    Queries the nearest 0.25° OISST grid cell to the zone centroid via ERDDAP.
    Returns a weekly-resampled pd.Series (mean of daily values per week),
    or None if the cell is masked (land) or the request fails.

    Publication lag: 0 weeks (OISST files posted within 1-2 days — immaterial
    for a weekly signal).
    """
    lat_raw = zone_info["lat"]
    lon_raw = zone_info["lon"]
    lat     = _snap_oisst(lat_raw)
    lon     = _snap_oisst(lon_raw)

    # ERDDAP griddap CSV: sst[(time)][(zlev)][(lat)][(lon)]
    # ncdcOisst21Agg_LonPM180 has 4 dimensions: time, zlev, latitude, longitude
    url = (
        f"{ERDDAP_BASE}/{SST_DATASET}.csv"
        f"?sst[({SST_START}T12:00:00Z):(last)]"
        f"[(0.0)]"
        f"[({lat:.3f}):1:({lat:.3f})]"
        f"[({lon:.3f}):1:({lon:.3f})]"
    )

    try:
        resp = _http_get(url, timeout=360)
        # ERDDAP CSV: first row is column headers, second row is units, then data
        lines = resp.text.splitlines()
        # Find header line (contains "time")
        header_idx = next((i for i, l in enumerate(lines) if "time" in l.lower()), 0)
        df = pd.read_csv(
            io.StringIO("\n".join(lines[header_idx:])),
            skiprows=[1],   # skip the units row
        )
        df.columns = [c.strip() for c in df.columns]
        time_col = next(c for c in df.columns if "time" in c.lower())
        sst_col  = next(c for c in df.columns if "sst" in c.lower())

        df[time_col] = pd.to_datetime(df[time_col], utc=True).dt.tz_localize(None)
        df = df.set_index(time_col)[sst_col]
        df = pd.to_numeric(df, errors="coerce")

        # OISST land/ice mask: masked values returned as NaN
        if df.notna().sum() < 10:
            _warn(f"  PO{zone_id} centroid ({lat:.3f}N, {lon:.3f}E) appears to be "
                  f"masked — likely on land or under ice. Trying +0.25° offshore.")
            # nudge offshore (west for Western Norway, north for Northern Norway)
            nudge_lon = lon - 0.25 if lon < 15 else lon
            nudge_lat = lat - 0.25 if lat > 68 else lat
            return fetch_sst_zone(
                zone_id,
                {**zone_info, "lat": nudge_lat, "lon": nudge_lon},
            ) if (nudge_lon != lon or nudge_lat != lat) else None

        # Resample daily → weekly (ISO week ending Sunday)
        weekly = df.resample("W").mean()
        weekly.name = f"sst_po{zone_id}"
        return weekly

    except Exception as exc:
        _warn(f"  SST fetch PO{zone_id} failed: {type(exc).__name__}: {str(exc)[:80]}")
        return None


def fetch_all_sst() -> pd.DataFrame:
    """
    Fetch SST for all production zones. Returns DataFrame with one column per zone.
    Also computes national aggregate (equal-weighted mean across zones with data).
    """
    frames = {}
    print("  Querying OISST for production zone centroids (ERDDAP):")
    for zid, zinfo in PRODUCTION_ZONES.items():
        print(f"    PO{zid:2d} {zinfo['name']:<30s} "
              f"→ centroid ({zinfo['lat']:.2f}N, {zinfo['lon']:.2f}E)", end=" … ", flush=True)
        s = fetch_sst_zone(zid, zinfo)
        if s is not None:
            frames[f"sst_po{zid}"] = s
            print(f"{s.notna().sum()} weekly obs")
        else:
            print("FAILED")

    if not frames:
        _warn("No SST data fetched from ERDDAP — all zones failed.")
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df["sst_national"] = df.mean(axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SST ANOMALY
# ─────────────────────────────────────────────────────────────────────────────

def compute_anomaly(series: pd.Series, label: str = "") -> pd.Series:
    """
    Compute anomaly by subtracting the week-of-year seasonal mean.

    Method: for each ISO week (1–52), compute the grand mean across all
    available years and subtract it from each observation. This removes the
    dominant annual cycle without requiring external climatology files.

    Limitation: the baseline drifts if the full series spans a period with
    long-term warming; a rolling baseline would be more robust for long
    records. For 10–14 years at 0.25°C/decade, the error is ~0.1–0.2°C
    — acceptable for a first-pass test.
    """
    df     = series.dropna().to_frame("val")
    df["woy"] = df.index.isocalendar().week.astype(int)
    baseline  = df.groupby("woy")["val"].mean()
    anomaly   = series.copy()
    for woy, mean_val in baseline.items():
        idx = series.index[series.index.isocalendar().week == woy]
        anomaly.loc[idx] -= mean_val
    anomaly.name = (series.name or label) + "_anom"
    return anomaly


# ─────────────────────────────────────────────────────────────────────────────
# SEA LICE FETCH — Barentswatch
# ─────────────────────────────────────────────────────────────────────────────

_BW_BASE = "https://www.barentswatch.no/bwapi"

# Candidate endpoints, tried in order (Barentswatch has changed API versions).
# We attempt each and use the first that returns parseable lice data.
_BW_LICE_ENDPOINTS = [
    # v1 overview aggregate by week
    f"{_BW_BASE}/v1/geodata/fishhealth/lice",
    # v3 geodata (newer)
    f"{_BW_BASE}/v3/geodata/fishhealth/lice",
    # bare fishhealth
    f"{_BW_BASE}/v1/fishhealth",
    f"{_BW_BASE}/v2/fishhealth/lice/weekly",
]


def _try_bw_endpoint(url: str, params: dict = None) -> "dict | list | None":
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 401:
            _warn(f"  Barentswatch: 401 Unauthorised at {url} — API key required.")
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def fetch_lice_national() -> "pd.Series | None":
    """
    Fetch weekly aggregate sea-lice counts (adult female lice per fish) from
    Barentswatch fish-health API.

    Barentswatch API notes (confirmed 2026-06-15):
      - Base URL: https://www.barentswatch.no/bwapi/
      - Some endpoints require OAuth2 client_credentials.
      - The 'open data' fish-health aggregate may be public; we try without auth.
      - If auth is required we cannot proceed without credentials.

    Publication lag: 1 week (data collected during the week, published ~1 week
    later on the Barentswatch portal).

    Returns weekly pd.Series indexed by ISO week end (Sunday), or None.
    """
    # ── Attempt 1: try known public endpoints ─────────────────────────────────
    for url in _BW_LICE_ENDPOINTS:
        data = _try_bw_endpoint(url)
        if data is None:
            continue
        print(f"  [lice] Barentswatch responded at {url}")
        print(f"    Response type: {type(data).__name__}, length: {len(data) if hasattr(data,'__len__') else 'N/A'}")

        # Parse if it's a list of records with lice count
        if isinstance(data, list) and len(data) > 0:
            df = pd.json_normalize(data)
            print(f"    Columns: {list(df.columns[:10])}")
            # Look for week/year and lice count fields
            lice_col = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in
                        ["adult", "female", "lice", "avglice", "avgadult"])),
                None,
            )
            year_col = next(
                (c for c in df.columns if "year" in c.lower() or "aar" in c.lower()),
                None,
            )
            week_col = next(
                (c for c in df.columns if "week" in c.lower() or "uke" in c.lower()),
                None,
            )
            if lice_col and year_col and week_col:
                df["date"] = df.apply(
                    lambda r: _iso_week_to_date(int(r[year_col]), int(r[week_col])),
                    axis=1,
                )
                s = df.groupby("date")[lice_col].mean()
                s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
                s.name = "lice_adult_female_per_fish"
                print(f"    → {len(s)} weekly obs  "
                      f"({s.index[0].date()} → {s.index[-1].date()})")
                return s

        if isinstance(data, dict):
            print(f"    Dict keys: {list(data.keys())[:8]}")

    # ── Attempt 2: year-by-year weekly query ─────────────────────────────────
    records = []
    start_year = int(SST_START[:4])
    for year in range(start_year, 2027):
        for week in range(1, 53):
            url = f"{_BW_BASE}/v1/geodata/fishhealth/overview/weekly/{year}/{week}"
            data = _try_bw_endpoint(url)
            if data is None:
                break   # auth or not found — stop probing
            if isinstance(data, dict) and "avgAdultFemaleLicePerFish" in data:
                records.append({
                    "date": _iso_week_to_date(year, week),
                    "lice": data["avgAdultFemaleLicePerFish"],
                })
        if not records:
            break   # first year already failed — don't continue

    if records:
        s = pd.DataFrame(records).set_index("date")["lice"]
        s = s.sort_index()
        s.name = "lice_adult_female_per_fish"
        print(f"  [lice] Barentswatch overview/weekly: {len(s)} obs")
        return s

    _warn("[lice] All Barentswatch endpoints failed or returned no parseable lice data.")
    _warn("       The Barentswatch API may require OAuth2 credentials.")
    _warn("       Register at: https://developer.barentswatch.no/")
    _warn("       Gate test (SST → lice) cannot run without this data.")
    return None


def _iso_week_to_date(year: int, week: int) -> pd.Timestamp:
    """Convert ISO year+week to the Sunday (end) of that week."""
    try:
        return pd.to_datetime(f"{year}W{week:02d}-7", format="%GW%V-%u")
    except Exception:
        return pd.NaT


def fetch_lice_by_zone() -> "pd.DataFrame | None":
    """
    Attempt to fetch lice data by production zone (PO1–PO11).
    Falls back to national aggregate if zone-level data is unavailable.
    Returns DataFrame with one column per zone (if available).
    """
    # Try a zone-level endpoint (if it exists, return zone-indexed DataFrame)
    url = f"{_BW_BASE}/v1/geodata/fishhealth/lice/zone"
    data = _try_bw_endpoint(url)
    if data is not None and isinstance(data, list) and len(data) > 0:
        df = pd.json_normalize(data)
        print(f"  [lice zones] Columns: {list(df.columns[:10])}")
        zone_col = next(
            (c for c in df.columns if "zone" in c.lower() or "omraade" in c.lower()),
            None,
        )
        if zone_col:
            # Parse per zone
            frames = {}
            for zone_id in range(1, 12):
                zdf = df[df[zone_col] == zone_id]
                if zdf.empty:
                    continue
                lice_col = next(
                    (c for c in df.columns
                     if any(k in c.lower() for k in ["adult", "female", "lice"])),
                    None,
                )
                if lice_col:
                    frames[f"lice_po{zone_id}"] = (
                        zdf.set_index("date")[lice_col]
                        .sort_index()
                    )
            if frames:
                return pd.DataFrame(frames)
    return None   # zone data unavailable


# ─────────────────────────────────────────────────────────────────────────────
# SALMON PRICE — SSB (known working)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_salmon_price() -> pd.Series:
    """SSB 03024 weekly export price (NOK/kg). Confirmed working from prior runs."""
    query = {
        "query": [
            {"code": "VareGrupper2", "selection": {"filter": "item", "values": ["01"]}},
            {"code": "ContentsCode",  "selection": {"filter": "item", "values": ["Kilopris"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    url = "https://data.ssb.no/api/v0/en/table/03024/"
    for attempt in range(3):
        try:
            r = requests.post(url, json=query, timeout=60)
            r.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

    data        = r.json()
    time_labels = list(data["dimension"]["Tid"]["category"]["index"].keys())
    values      = data["value"]

    def _uke_to_date(w):
        try:
            return pd.to_datetime(w.replace("U", "W") + "-7", format="%GW%V-%u")
        except Exception:
            return pd.NaT

    dates = pd.DatetimeIndex([_uke_to_date(w) for w in time_labels])
    s = pd.Series(
        [float(v) if v is not None else np.nan for v in values],
        index=dates,
        name="salmon_price_nok",
    )
    return s[s.index.notna()].sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# EQUITIES — yfinance
# ─────────────────────────────────────────────────────────────────────────────

def fetch_equities() -> "tuple[pd.DataFrame, dict]":
    """Weekly adjusted-close prices for salmon farmer tickers."""
    frames   = {}
    coverage = {}
    for ticker in FARMER_TICKERS:
        t    = yf.Ticker(ticker)
        hist = pd.DataFrame()
        for attempt in range(3):
            try:
                hist = t.history(start=EQUITY_START, interval="1wk", auto_adjust=True)
                if not hist.empty:
                    break
            except Exception:
                if attempt == 2:
                    break
            time.sleep(2)

        if hist.empty:
            coverage[ticker] = None
            continue

        close = hist["Close"].dropna()
        close.index = (close.index.tz_localize(None)
                       if close.index.tz is not None else close.index)
        # Snap to week-ending Sunday to match SST/lice index
        close = close.resample("W").last()
        frames[ticker]   = close
        coverage[ticker] = (close.index[0].date(), close.index[-1].date(), len(close))

    prices = pd.DataFrame(frames) if frames else pd.DataFrame()
    return prices, coverage


# ─────────────────────────────────────────────────────────────────────────────
# PANEL ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(sst_df, lice_national, lice_zones, price, eq_prices, eq_coverage):
    """
    Apply publication lags, compute anomalies, relative returns, and
    zone-weighted signals. Join all series into a weekly panel.

    Key point-in-time rules (all lags applied via .shift()):
      SST:    lag 0 — OISST posted within 1-2 days of observation
      Lice:   lag 1 week — Barentswatch data reported ~1 week after collection
      Price:  lag 0 — SSB weekly, within 2 weeks (treat as 0 for weekly)
      Equity: lag 0 — live daily prices, weekly close
    """
    parts = []

    # ── SST: national aggregate anomaly ──────────────────────────────────────
    sst_anom_cols = []
    if "sst_national" in sst_df.columns:
        sst_nat = sst_df["sst_national"].shift(PUB_LAG["sst"])
        sst_nat.name = "sst_national"
        sst_nat_anom = compute_anomaly(sst_nat, "sst_national")
        parts += [sst_nat, sst_nat_anom]
        sst_anom_cols.append("sst_national_anom")

    # Per-zone SST anomaly
    for col in [c for c in sst_df.columns if c.startswith("sst_po")]:
        s = sst_df[col].shift(PUB_LAG["sst"])
        anom = compute_anomaly(s, col)
        parts.append(anom)
        sst_anom_cols.append(anom.name)

    # ── Lice: national de-seasonalised ───────────────────────────────────────
    if lice_national is not None:
        lice_pub = lice_national.shift(PUB_LAG["lice"])
        lice_pub.name = "lice_national"
        lice_anom = compute_anomaly(lice_pub, "lice_national")
        parts += [lice_pub, lice_anom]

    # Per-zone lice anomaly (if available)
    if lice_zones is not None:
        for col in lice_zones.columns:
            s    = lice_zones[col].shift(PUB_LAG["lice"])
            anom = compute_anomaly(s, col)
            parts.append(anom)

    # ── Salmon price: weekly log return ──────────────────────────────────────
    price_pub = price.shift(PUB_LAG["price"])
    price_pub.name = "salmon_price"
    price_ret = np.log(price_pub / price_pub.shift(1))
    price_ret.name = "salmon_price_ret"
    parts += [price_pub, price_ret]

    # ── Equity returns and relative returns ──────────────────────────────────
    farmer_cols = [t for t in FARMER_TICKERS if t in eq_prices.columns]
    short_names = [t.split(".")[0] for t in farmer_cols]
    ret_cols    = {}

    if farmer_cols:
        eq = eq_prices[farmer_cols]
        eq_ret = np.log(eq / eq.shift(1))
        eq_ret.columns = [t.split(".")[0] + "_ret" for t in farmer_cols]

        n_valid = eq_ret.notna().sum(axis=1)
        basket  = eq_ret.mean(axis=1)
        basket[n_valid < 2] = np.nan
        basket.name = "basket_ret"

        # Relative returns: farmer minus basket (strips common commodity factor)
        for col in eq_ret.columns:
            fname         = col.replace("_ret", "")
            rel_col       = fname + "_rel"
            eq_ret[rel_col] = eq_ret[col] - basket
            ret_cols[fname] = rel_col

        parts += [basket] + list(eq_ret.values.T if False else [eq_ret[c] for c in eq_ret.columns])

    # ── Zone-weighted SST anomaly per farmer ─────────────────────────────────
    # For each farmer, compute a weighted average SST anomaly across their zones
    for fname, zone_weights in FARMER_ZONES.items():
        if not zone_weights:
            continue
        avail_zones = {z: w for z, w in zone_weights.items()
                       if f"sst_po{z}_anom" in [p.name for p in parts if hasattr(p, 'name')]}
        if not avail_zones:
            # SST cols may be in parts already — let's build later in main after concat
            pass

    # ── Join ─────────────────────────────────────────────────────────────────
    panel = pd.concat(parts, axis=1).sort_index()
    panel = panel[panel.index >= pd.Timestamp(SST_START)]

    # Build zone-weighted SST per farmer POST-concat (easier to index columns)
    for fname, zone_weights in FARMER_ZONES.items():
        if not zone_weights:
            continue
        avail = [(z, w) for z, w in zone_weights.items()
                 if f"sst_po{z}_anom" in panel.columns]
        if avail:
            total_w = sum(w for _, w in avail)
            weighted = sum(panel[f"sst_po{z}_anom"] * w for z, w in avail) / total_w
            weighted.name = f"sst_{fname.lower()}_zone_anom"
            panel = pd.concat([panel, weighted], axis=1)

    return panel, ret_cols


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def count_warm_episodes(sst_anom: pd.Series, threshold: float = SST_ANOMALY_THRESHOLD) -> list:
    """
    Count independent warm SST anomaly episodes.
    An episode = SST anomaly ≥ threshold for ≥ WARM_EPISODE_MIN_WEEKS consecutive weeks.
    """
    above    = (sst_anom >= threshold).astype(int)
    episodes = []
    in_ep    = False
    ep_start = None
    run_len  = 0

    for date, val in above.dropna().items():
        if val:
            if not in_ep:
                ep_start = date
                in_ep    = True
            run_len += 1
        else:
            if in_ep and run_len >= WARM_EPISODE_MIN_WEEKS:
                episodes.append((ep_start, date, run_len))
            in_ep   = False
            run_len = 0
    if in_ep and run_len >= WARM_EPISODE_MIN_WEEKS:
        episodes.append((ep_start, sst_anom.index[-1], run_len))

    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# TEST A — SST anomaly → lice counts
# ─────────────────────────────────────────────────────────────────────────────

def test_a(panel: pd.DataFrame) -> "pd.DataFrame | None":
    """
    Gate test: does de-seasonalised SST anomaly predict future sea-lice counts?

    Regression: lice_national_anom_{t+k} ~ sst_national_anom_t, for k = 0..6 weeks.
    Expected sign: POSITIVE (warmer SST → more lice).

    If no lice data: gate fails, stop.
    If SST data missing: gate fails, stop.
    Reports ALL lags — do not select the best-looking one.
    """
    if "lice_national_anom" not in panel.columns:
        _skip("No lice data — cannot run gate test.")
        return None
    if "sst_national_anom" not in panel.columns:
        _skip("No SST national aggregate — cannot run gate test.")
        return None

    sst   = panel["sst_national_anom"]
    lice  = panel["lice_national_anom"]
    rows  = []

    for lag in LAGS_A:
        future_lice = lice.shift(-lag)
        coef, t, r2, n = _ols(sst, future_lice)
        sign_str = "✓ pos" if (coef or 0) > 0 else "✗ neg (WRONG)"
        rows.append({"lag_weeks": lag, "coef": coef, "t_stat": t,
                     "r2": r2, "n": n, "sign": sign_str})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TEST B — signal → relative farmer returns
# ─────────────────────────────────────────────────────────────────────────────

def test_b(panel: pd.DataFrame, ret_cols: dict) -> "pd.DataFrame | None":
    """
    Does zone-weighted SST anomaly (or lice if available) predict each farmer's
    relative return (farmer return minus basket) over 1–8 weeks ahead?

    Signal options (in priority order):
      1. Zone-weighted lice anomaly per farmer  (if lice data available)
      2. Zone-weighted SST anomaly per farmer   (as SST-direct signal)

    Expected sign: NEGATIVE (high signal → underperformance vs basket).
    Feed/commodity effects wash out in relative returns (they are common shocks).

    Reports ALL horizons per farmer.
    """
    if not ret_cols:
        _skip("No equity return data — cannot run Test B.")
        return None

    rows = []
    for fname, rel_col in ret_cols.items():
        if rel_col not in panel.columns:
            continue

        rel_ret = panel[rel_col]

        # Pick signal: lice-zone if available, else SST-zone, else SST national
        lice_zone_col = f"lice_po{fname.lower()}_anom" if False else None  # zone lice TBD
        sst_zone_col  = f"sst_{fname.lower()}_zone_anom"
        sst_nat_col   = "sst_national_anom"

        if lice_zone_col and lice_zone_col in panel.columns:
            signal     = panel[lice_zone_col]
            signal_src = "lice_zone"
        elif sst_zone_col in panel.columns:
            signal     = panel[sst_zone_col]
            signal_src = "sst_zone_weighted"
        elif sst_nat_col in panel.columns:
            signal     = panel[sst_nat_col]
            signal_src = "sst_national"
        else:
            continue

        for h in HORIZONS_B:
            fwd_ret = rel_ret.shift(-h)
            coef, t, r2, n = _ols(signal, fwd_ret)
            sign_str = "✓ neg" if (coef or 0) < 0 else "✗ POSITIVE (WRONG)"
            rows.append({
                "farmer":       fname,
                "signal_src":   signal_src,
                "horizon_wks":  h,
                "coef":         coef,
                "t_stat":       t,
                "r2":           r2,
                "n":            n,
                "sign":         sign_str,
            })

    return pd.DataFrame(rows) if rows else None


# ─────────────────────────────────────────────────────────────────────────────
# TEST C — signal backtest
# ─────────────────────────────────────────────────────────────────────────────

def test_c(panel: pd.DataFrame, ret_cols: dict) -> "dict | None":
    """
    Simple signal backtest on relative returns.

    Signal per farmer: SST zone anomaly > SST_ANOMALY_THRESHOLD
    → expect underperformance → short relative to basket (short farmer, long basket).
    Entry: next week (shift signal forward 1 week, no same-week look-ahead).

    Strategy return per farmer when signal on: -(farmer_rel_ret)
    Aggregate across farmers equally (every signalled farmer equally weighted).

    Costs: SIGNAL_COST_BPS bps PLACEHOLDER on each transition.
    """
    if not ret_cols:
        return None

    all_returns = []
    for fname, rel_col in ret_cols.items():
        sst_zone_col = f"sst_{fname.lower()}_zone_anom"
        signal_col   = sst_zone_col if sst_zone_col in panel.columns else "sst_national_anom"
        if signal_col not in panel.columns or rel_col not in panel.columns:
            continue

        signal        = (panel[signal_col] > SST_ANOMALY_THRESHOLD).astype(float)
        signal_lagged = signal.shift(1)
        rel_ret       = panel[rel_col]

        strat = pd.Series(
            np.where(signal_lagged.fillna(False).astype(bool), -rel_ret, 0.0),
            index=panel.index,
        )
        transitions = signal_lagged.fillna(0).astype(int).diff().abs().fillna(0)
        strat -= transitions * (SIGNAL_COST_BPS / 10_000)

        valid = strat.notna() & rel_ret.notna() & signal_lagged.notna()
        all_returns.append(strat[valid])

    if not all_returns:
        return None

    # Aggregate across farmers (simple average of strategy returns)
    agg = pd.concat(all_returns, axis=1).mean(axis=1).dropna()
    if len(agg) < 20:
        return None

    sharpe    = agg.mean() / agg.std() * np.sqrt(52) if agg.std() > 0 else np.nan
    hit_rate  = float((agg > 0).mean())
    n_ep      = 0
    if "sst_national_anom" in panel.columns:
        eps = count_warm_episodes(panel["sst_national_anom"])
        n_ep = len(eps)

    return {
        "mean_weekly_ret_strategy": round(float(agg.mean()),  5),
        "hit_rate":                 round(hit_rate,            3),
        "annualised_sharpe":        round(float(sharpe),       3),
        "n_weeks_in_strategy":      int(len(agg)),
        "n_independent_warm_episodes": n_ep,
        "cost":                     f"{SIGNAL_COST_BPS} bps round-trip per transition (PLACEHOLDER)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("default")

    # ── 1. Fetch all data ─────────────────────────────────────────────────────
    _hdr("SST LEAD-LAG SIGNAL — DATA FETCH")

    print("Fetching SST (NOAA OISST v2.1 via ERDDAP) …")
    sst_df = fetch_all_sst()
    n_sst_zones = len([c for c in sst_df.columns if c.startswith("sst_po")])
    if sst_df.empty:
        _warn("No SST data — all ERDDAP queries failed.")
    else:
        print(f"  → {n_sst_zones} zones with data, "
              f"{sst_df['sst_national'].notna().sum() if 'sst_national' in sst_df.columns else 0}"
              f" national weekly obs")

    print("\nFetching sea-lice counts (Barentswatch) …")
    lice_national = fetch_lice_national()
    lice_zones    = fetch_lice_by_zone() if lice_national is not None else None

    print("\nFetching salmon price (SSB 03024) …")
    price = fetch_salmon_price()
    print(f"  → {price.notna().sum()} weekly obs")

    print("\nFetching equities (yfinance) …")
    eq_prices, eq_coverage = fetch_equities()
    for tk, info in eq_coverage.items():
        if info:
            print(f"  {tk:<12s}  {info[0]} → {info[1]}  ({info[2]} weekly obs)")
        else:
            print(f"  {tk:<12s}  NO DATA")

    # ── 2. Assemble panel ─────────────────────────────────────────────────────
    _hdr("PANEL ASSEMBLY")
    panel, ret_cols = build_panel(
        sst_df, lice_national, lice_zones, price, eq_prices, eq_coverage
    )
    print(f"Panel: {panel.shape[0]} weeks × {panel.shape[1]} columns")
    print(f"Range: {panel.index[0].date()} → {panel.index[-1].date()}")
    print("\nNon-null counts per series:")
    print(panel.count().to_string())

    print("\nPublication lags applied:")
    for src, lag in PUB_LAG.items():
        print(f"  {src:<10s} shift({lag}w)  "
              + {"sst": "OISST posted within 1-2 days — 0 lag for weekly signal",
                 "lice": "Barentswatch data ~1 week after collection",
                 "price": "SSB weekly, ≤2 weeks — treat as 0",
                 "equity": "live daily prices"}[src])

    print("\nZone-weighted SST anomaly columns built for:")
    for fname, zone_weights in FARMER_ZONES.items():
        col = f"sst_{fname.lower()}_zone_anom"
        avail = ", ".join(f"PO{z}(w={w:.0%})" for z, w in zone_weights.items()
                          if f"sst_po{z}_anom" in panel.columns)
        status = f"→ {col}" if col in panel.columns else "→ UNAVAILABLE (no zone SST)"
        print(f"  {fname:<8s}  zones: {avail or 'NONE'}  {status}")

    print("\nFarmer-to-zone mapping (approximate; based on public annual reports):")
    print("  Farmer    Core production zones                     Note")
    print("  -------   ----------------------------------------  --------------------------------")
    zone_notes = {
        "MOWI":  "Norway-wide; also Chile, Canada, Scotland",
        "SALM":  "Trøndelag + Helgeland dominant",
        "LSG":   "Western Norway + Trøndelag",
        "BAKKA": "Faroe Islands (not in NOR PO system)",
        "GSF":   "Western Norway + BC Canada + Shetland",
    }
    for fname, zone_weights in FARMER_ZONES.items():
        zone_str = ", ".join(f"PO{z}" for z in sorted(zone_weights)) if zone_weights else "none"
        print(f"  {fname:<8s}  {zone_str:<42s}  {zone_notes.get(fname,'')}")

    # ── 3. Diagnostics ────────────────────────────────────────────────────────
    _hdr("DIAGNOSTIC — EFFECTIVE SAMPLE SIZE")
    print("Weekly rows are autocorrelated — NOT independent observations.")
    print("True signal n = number of independent warm SST anomaly episodes.\n")

    if "sst_national_anom" in panel.columns:
        warm_eps = count_warm_episodes(panel["sst_national_anom"])
        print(f"  SST national anomaly ≥ {SST_ANOMALY_THRESHOLD}°C "
              f"for ≥ {WARM_EPISODE_MIN_WEEKS} consecutive weeks: {len(warm_eps)} episodes")
        for i, (s, e, ln) in enumerate(warm_eps, 1):
            print(f"  {i:2d}.  {s.date()} → {e.date()}  ({ln} weeks)")
    else:
        print("  No SST anomaly data — cannot compute warm episodes.")

    print(f"\n  Cross-section size:    {len(ret_cols)} farmers with relative-return data")
    print(f"  Total weekly obs:      {panel.shape[0]}  (NOT the independent n)")
    print(f"  Lice data available:   {'yes' if lice_national is not None else 'NO — gate test blocked'}")

    # ── 4. Test A ─────────────────────────────────────────────────────────────
    _hdr("TEST A (GATE) — SST ANOMALY → LICE COUNTS?")
    print("Regression: lice_national_anom_{t+k} ~ sst_national_anom_t, k = 0..6 weeks.")
    print("Expected sign: POSITIVE (warmer SST → more lice).\n")

    tA        = test_a(panel)
    gate_pass = False

    if tA is None:
        if lice_national is None:
            print("  GATE STATUS: BLOCKED — no lice data.")
            print("  The chain cannot be validated without Barentswatch lice counts.")
            print("  Tests B and C run below on the SST-ONLY signal (weaker test;")
            print("  no lice validation means we're skipping link A of the chain).")
        elif "sst_national_anom" not in panel.columns:
            print("  GATE STATUS: BLOCKED — no SST data.")
    else:
        print(tA.to_string(index=False))
        max_t_a   = float(tA["t_stat"].abs().max(skipna=True))
        n_pos_sig = int(((tA["t_stat"].abs() > 1.65) & (tA["coef"] > 0)).sum())
        gate_pass = n_pos_sig > 0
        print(f"\n  Max |t| = {max_t_a:.2f}  → gate {'PASSES' if gate_pass else 'FAILS'}")
        if not gate_pass:
            print("  SST anomaly does not reliably predict lice escalation in this data.")
            print("  The first link of the chain is not supported.")
            print("  Proceeding to Test B for completeness — interpret cautiously.")

    # ── 5. Test B ─────────────────────────────────────────────────────────────
    _hdr("TEST B — SST SIGNAL → RELATIVE FARMER RETURNS")
    print("Signal: zone-weighted SST anomaly per farmer (or SST national if no zone SST).")
    print("Target: farmer return MINUS basket return (strips commodity factor).")
    print("Expected sign: NEGATIVE (warmer SST → worse relative performance).\n")

    tB = test_b(panel, ret_cols)
    if tB is None:
        _skip("No equity or SST data for Test B.")
    else:
        if tB.empty:
            print("  No results computed.")
        else:
            # Print per farmer
            for fname in tB["farmer"].unique():
                sub = tB[tB["farmer"] == fname]
                print(f"\n  [{fname}] signal: {sub['signal_src'].iloc[0]}")
                print(sub.drop(columns=["farmer","signal_src"]).to_string(index=False))

            max_t_b    = float(tB["t_stat"].abs().max(skipna=True))
            neg_sig    = tB[(tB["t_stat"].abs() > 1.65) & (tB["coef"] < 0)]
            pos_sig    = tB[(tB["t_stat"].abs() > 1.65) & (tB["coef"] > 0)]
            print(f"\n  Max |t| across all farmers/horizons: {max_t_b:.2f}")
            if not neg_sig.empty:
                print(f"  {len(neg_sig)} (farmer, horizon) combinations significant NEGATIVE ✓")
            if not pos_sig.empty:
                print(f"  ⚠  {len(pos_sig)} combinations significant POSITIVE (WRONG direction)")

    # ── 6. Test C ─────────────────────────────────────────────────────────────
    _hdr("TEST C — SIGNAL BACKTEST")

    if tA is None and lice_national is None:
        _note("Gate incomplete (no lice data) — Test C runs on SST-only signal.")
        _note("This is a weaker test since link A (SST→lice) is not confirmed.")

    b_max_t = float(tB["t_stat"].abs().max(skipna=True)) if tB is not None and not tB.empty else 0.0
    if b_max_t < 1.65 and lice_national is None:
        _skip("Neither gate nor Test B shows evidence — backtest would overfit.")
    else:
        stats_c = test_c(panel, ret_cols)
        if stats_c is None:
            _skip("Insufficient data for backtest.")
        else:
            print()
            for k, v in stats_c.items():
                if isinstance(v, float):
                    print(f"  {k:<42s}  {v:+.4f}")
                else:
                    print(f"  {k:<42s}  {v}")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    _hdr("SUMMARY & DATA-SOURCE REPORT")
    print(f"  SST (NOAA OISST via ERDDAP):    "
          f"{n_sst_zones}/11 zones fetched")
    print(f"  Lice (Barentswatch API):         "
          f"{'✓ ' + str(lice_national.notna().sum()) + ' weekly obs' if lice_national is not None else '✗ unavailable (auth required or API changed)'}")
    print(f"  Salmon price (SSB 03024):        ✓ {price.notna().sum()} weekly obs")
    print(f"  Equities:                        "
          f"{sum(1 for v in eq_coverage.values() if v)}/{len(eq_coverage)} tickers")
    print(f"  Fish Pool forwards:              ✗ (not applicable to this test)")
    print()
    print(f"  Warm SST episodes (≥{SST_ANOMALY_THRESHOLD}°C, ≥{WARM_EPISODE_MIN_WEEKS}w): "
          f"{len(count_warm_episodes(panel['sst_national_anom'])) if 'sst_national_anom' in panel.columns else 'N/A'}")
    print(f"  Cross-section:                   {len(ret_cols)} farmers (BAKKA excl. from signal)")
    print(f"  Test A gate:                     {'PASSED' if gate_pass else 'FAILED or BLOCKED'}")
    print()
    if lice_national is None:
        print("  KEY LIMITATION: Without Barentswatch lice data, link A of the chain")
        print("  (SST→lice) is unverified. Any Test B result should be treated as")
        print("  'SST predicts relative returns via unknown mechanism' — not validated.")
        print("  To unlock: register at developer.barentswatch.no for API credentials.")
    _rule()
    print("  Do not tune any parameter to improve these results.")
    _rule()


if __name__ == "__main__":
    main()
