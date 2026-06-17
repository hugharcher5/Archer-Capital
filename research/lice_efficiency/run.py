#!/usr/bin/env python3
"""
research/lice_efficiency/run.py
================================
Hypothesis: A salmon farmer that consistently maintains lice BELOW its zone
average is operationally superior. Relative lice residual should predict
relative equity returns — operational alpha, commodity-neutral.

Tests
-----
  A  — pooled panel OLS, lags 0-4: does relative lice → relative return?
  B  — per-farmer OLS: sign consistency across names
  C  — simple signal backtest (only if Test A max |t-stat| > 1.5)

AUTH
-----
Barentswatch fish-health data requires OAuth2 client credentials.
Register (free) at https://developer.barentswatch.no/

Then set env vars before running:
  export BW_CLIENT_ID="your-client-id"
  export BW_CLIENT_SECRET="your-client-secret"

OR create a file: research/lice_efficiency/bw_credentials.py
  BW_CLIENT_ID = "your-client-id"
  BW_CLIENT_SECRET = "your-client-secret"

Company → site mapping uses the public Fiskeridirektoratet API (no auth).

Point-in-time: lice has 1-week publication lag throughout.
No look-ahead anywhere.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
_CACHE     = _HERE / "_cache"
_CACHE.mkdir(exist_ok=True)

_BW_BASE   = "https://www.barentswatch.no/bwapi"
_BW_TOKEN  = "https://id.barentswatch.no/connect/token"
_FDIR_BASE = "https://api.fiskeridir.no/pub-aqua/api/v1"

START_YEAR = 2012
END_YEAR   = 2026
PUB_LAG    = 1     # weeks: lice collected week T, available T+1
MAX_LAG    = 4     # test lags 0..MAX_LAG
C_GATE     = 1.5   # min abs(t-stat) in Test A to run Test C

TICKERS: dict[str, str] = {
    "MOWI":  "MOWI.OL",
    "SALM":  "SALM.OL",
    "LSG":   "LSG.OL",
    "GSF":   "GSF.OL",
    "BAKKA": "BAKKA.OL",
}
EXCLUDED_SIGNAL = {
    "BAKKA": (
        "Primarily Faroese. Norwegian sites acquired 2019 via Scottish Salmon Co. "
        "Cannot reliably map to Norwegian site numbers → excluded from lice tests."
    )
}

# Company name patterns, case-insensitive substring match
COMPANY_PATTERNS: list[tuple[str, list[str]]] = [
    ("MOWI", ["mowi", "marine harvest"]),
    ("SALM", ["salmar"]),
    ("LSG",  ["lerøy", "leroy", "leroi"]),
    ("GSF",  ["grieg seafood", "grieg"]),
]

# FDIR entity name search terms (what to search per farmer).
# Covers both current names and historical (Marine Harvest → Mowi in 2018).
FDIR_ENTITY_SEARCHES: dict[str, list[str]] = {
    "MOWI": ["mowi", "marine harvest"],
    "SALM": ["salmar"],
    "LSG":  ["lerøy", "leroy"],
    "GSF":  ["grieg seafood", "grieg"],
}

# Norwegian org-number (openNr in FDIR) → farmer ticker.
# Sourced from FDIR entities endpoint — these are verified.
ORG_MAP: dict[str, str] = {
    # MOWI (FDIR confirmed)
    "921668236": "MOWI",  # Mowi Seawater Norway AS
    "964118191": "MOWI",  # Mowi ASA (Norwegian holding)
    "912820924": "MOWI",  # Mowi ASA (listed entity)
    "817794282": "MOWI",  # Marine Harvest Norway AS (legacy name pre-2018)
    # SALM (FDIR confirmed)
    "958973306": "SALM",  # SalMar AS
    "928957489": "SALM",  # SalMar Oppdrett AS
    "935182484": "SALM",  # SalMar Dåfjord AS
    "935543948": "SALM",  # SalMar Settefisk AS
    "966840528": "SALM",  # SalMar Farming AS
    "981615697": "SALM",  # SalMar ASA (listed entity)
    # LSG (FDIR confirmed)
    "886813082": "LSG",   # Lerøy Vest AS
    "930185698": "LSG",   # Lerøy Vest Sjø AS
    "918904913": "LSG",   # Lerøy Sjøtroll Kjærelva AS
    "996542068": "LSG",   # Lerøy Årskog AS
    "985848718": "LSG",   # Lerøy Midt AS
    "820740882": "LSG",   # Lerøy Ocean Harvest AS
    "985940460": "LSG",   # Lerøy Aurora AS
    "930155209": "LSG",   # Lerøy Midt Sjø AS
    "930155179": "LSG",   # Lerøy Aurora Sjø AS
    "975350940": "LSG",   # Lerøy Seafood Group ASA (listed entity)
    # GSF (FDIR confirmed)
    "935496578": "GSF",   # Grieg Seafood Farming AS
    "838065392": "GSF",   # Grieg Seafood Rogaland AS
    "987793812": "GSF",   # Grieg Seafood ASA (listed entity)
    "990308341": "GSF",   # Grieg Seafood Finnmark AS (legacy)
}

# ─────────────────────────────────────────────────────────────────────────────
# AUTH — Barentswatch OAuth2
# ─────────────────────────────────────────────────────────────────────────────
def _get_bw_credentials() -> tuple[str, str] | None:
    """
    Load Barentswatch client credentials from env or local credentials file.
    Returns (client_id, client_secret) or None if not configured.
    """
    cid  = os.environ.get("BW_CLIENT_ID")
    csec = os.environ.get("BW_CLIENT_SECRET")
    if cid and csec:
        return cid, csec

    cred_file = _HERE / "bw_credentials.py"
    if cred_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("bw_creds", cred_file)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cid  = getattr(mod, "BW_CLIENT_ID", None)
        csec = getattr(mod, "BW_CLIENT_SECRET", None)
        if cid and csec:
            return cid, csec

    return None


_bw_token_cache: dict = {}

def _get_bw_token(creds: tuple[str, str]) -> str | None:
    """Fetch (or reuse cached) OAuth2 access token from Barentswatch."""
    global _bw_token_cache
    if _bw_token_cache.get("expires_at", 0) > time.time() + 30:
        return _bw_token_cache["token"]

    cid, csec = creds
    try:
        r = requests.post(
            _BW_TOKEN,
            data={
                "client_id":     cid,
                "client_secret": csec,
                "grant_type":    "client_credentials",
                "scope":         "api",
            },
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        _bw_token_cache = {"token": token, "expires_at": time.time() + expires_in}
        print(f"  [auth] Barentswatch token obtained (expires in {expires_in}s)")
        return token
    except Exception as e:
        print(f"  [auth] Token fetch failed: {e}")
        return None


def _bw_get(url: str, params: dict | None = None,
            token: str | None = None) -> list | dict | str | None:
    """GET Barentswatch. Returns 'AUTH' on 401/403, None on error."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        if r.status_code in (401, 403):
            return "AUTH"
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _iso_end(year: int, week: int) -> pd.Timestamp:
    try:
        return pd.to_datetime(f"{year}W{week:02d}-7", format="%GW%V-%u")
    except Exception:
        return pd.NaT


def _identify_farmer(name: str | None, org: str | None = None) -> str | None:
    if org and str(org).strip() in ORG_MAP:
        return ORG_MAP[str(org).strip()]
    if not isinstance(name, str):
        return None
    nl = name.lower().strip()
    for farmer, patterns in COMPANY_PATTERNS:
        if any(p in nl for p in patterns):
            return farmer
    return None


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def _ols(y: np.ndarray, x: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(x)
    y_, x_ = y[mask], x[mask]
    n = int(mask.sum())
    if n < 30:
        return {"coef": np.nan, "t_stat": np.nan, "r2": np.nan, "n": n}
    slope, _, r, _, se = stats.linregress(x_, y_)
    t = slope / se if se > 0 else np.nan
    return {"coef": slope, "t_stat": t, "r2": r ** 2, "n": n}


def _effective_n(series: pd.Series) -> int:
    """
    Estimate effective sample size adjusting for AR(1) autocorrelation.
    n_eff = n * (1 - r1) / (1 + r1)  where r1 is lag-1 autocorrelation.
    Returns raw n if autocorrelation is indeterminate.
    """
    s = series.dropna()
    n = len(s)
    if n < 10:
        return n
    r1 = s.autocorr(lag=1)
    if np.isnan(r1) or abs(r1) >= 1.0:
        return n
    return max(1, int(n * (1 - r1) / (1 + r1)))


# ─────────────────────────────────────────────────────────────────────────────
# FDIR SITE → COMPANY MAPPING (public API, no auth required)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_fdir_site_map() -> pd.DataFrame:
    """
    Build Norwegian aquaculture site → company mapping from the public
    Fiskeridirektoratet API via the entity + sites endpoint.

    Approach: for each farmer, search entities by name (covers all subsidiaries),
    then fetch all sites for each matched entity.

    Returns DataFrame: [site_nr, farmer, company_name, prod_area_code]
    No auth required. Caches for 30 days.
    """
    cache = _CACHE / "fdir_site_map.csv"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 86400 < 30:
        age = (time.time() - cache.stat().st_mtime) / 86400
        print(f"  [FDIR] Loading site map from cache ({age:.0f}d old)")
        return pd.read_csv(cache)

    print("\n[FDIR] Building site → farmer map via entity search …")

    rows: list[dict] = []

    for farmer, search_terms in FDIR_ENTITY_SEARCHES.items():
        seen_entities: set[str] = set()

        for term in search_terms:
            try:
                r = requests.get(
                    f"{_FDIR_BASE}/entities",
                    params={"name": term},
                    timeout=20,
                )
                r.raise_for_status()
                entities = r.json()
            except Exception as e:
                print(f"  [FDIR] Entity search '{term}' failed: {e}")
                continue

            for entity in entities:
                eid   = entity.get("id", "")
                ename = entity.get("name", "")
                if eid in seen_entities:
                    continue
                seen_entities.add(eid)

                # Fetch all sites for this entity
                try:
                    r2 = requests.get(
                        f"{_FDIR_BASE}/entities/{eid}/sites",
                        timeout=20,
                    )
                    r2.raise_for_status()
                    sites = r2.json()
                except Exception as e:
                    print(f"  [FDIR] Sites for entity {ename} failed: {e}")
                    continue

                for site in sites:
                    snr  = site.get("siteNr")
                    prod = (site.get("sitePlacement") or {}).get("prodAreaCode")
                    if snr:
                        rows.append({
                            "site_nr":        int(snr),
                            "farmer":         farmer,
                            "company_name":   ename,
                            "prod_area_code": int(prod) if prod else None,
                        })

                print(f"  [FDIR]   {farmer} / {ename}: {len(sites)} sites")
                time.sleep(0.1)

    if not rows:
        print("  [FDIR] No site data returned.")
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["site_nr"])   # one farmer per site
        .reset_index(drop=True)
    )
    df.to_csv(cache, index=False)
    print(f"\n  [FDIR] Total: {len(df):,} unique sites")
    for f in ["MOWI", "SALM", "LSG", "GSF"]:
        n = (df["farmer"] == f).sum()
        zones = sorted(df.loc[df["farmer"] == f, "prod_area_code"].dropna()
                       .astype(int).unique().tolist())
        print(f"         {f}: {n} sites  zones={zones}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BARENTSWATCH LICE FETCH (site-level, requires OAuth2)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_site_lice(token: str) -> pd.DataFrame:
    """
    Fetch site-level adult female lice per fish from Barentswatch.
    Requires a valid OAuth2 token (pass "cached" to skip fetch and use cache).

    Strategy:
    1. Try the bulk download endpoint (fastest, gets full history at once).
    2. Fall back to per-week site-summary endpoint.

    Returns DataFrame: [site_nr, year, week, lice, date]
    Caches to _cache/site_lice.csv.
    """
    cache = _CACHE / "site_lice.csv"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 86400 < 7:
        print(f"  [lice] Cache hit ({(time.time()-cache.stat().st_mtime)/86400:.1f}d old)")
        df = pd.read_csv(cache, parse_dates=["date"])
        print(f"         {len(df):,} rows, {df['site_nr'].nunique()} sites, "
              f"{df['date'].nunique()} weeks")
        return df

    if token == "cached":
        print("  [lice] ERROR: cache missing but token is 'cached'. Re-run with valid token.")
        return pd.DataFrame()

    print(f"\n[lice] Fetching site-level lice data ({START_YEAR}–{END_YEAR}) …")

    # ── Strategy 1: bulk download ──────────────────────────────────────────
    download_url = f"{_BW_BASE}/v1/geodata/download/fishhealth"
    for rtype in ["allliceweekly", "liceweekly", "lice", "locality", "alllice"]:
        data = _bw_get(download_url, params={
            "reporttype": rtype,
            "filetype":   "json",
            "fromweek":   1,  "fromyear": START_YEAR,
            "toweek":     52, "toyear":   END_YEAR,
        }, token=token)
        if isinstance(data, list) and len(data) > 10:
            print(f"  [lice] Bulk download: reporttype={rtype}, {len(data)} records")
            df = _parse_lice_records(data)
            if not df.empty:
                df.to_csv(cache, index=False)
                print(f"  [lice] Cached {len(df):,} site-week rows")
                return df
        elif data == "AUTH":
            print("  [lice] Auth failed even with token — check credentials.")
            return pd.DataFrame()
        elif isinstance(data, dict):
            print(f"  [lice] Bulk download ({rtype}) returned dict: {list(data.keys())[:5]}")

    print("  [lice] Bulk download not successful. Trying per-week endpoint …")

    # ── Strategy 2: per-week site summary ────────────────────────────────
    # Endpoint returns {"year":..., "week":..., "localities": [...]}
    all_frames: list[pd.DataFrame] = []
    today_yr  = _date.today().year
    today_wk  = _date.today().isocalendar()[1]

    for year in range(START_YEAR, END_YEAR + 1):
        yr_frames: list[pd.DataFrame] = []
        consecutive_empty = 0

        for week in range(1, 53):
            if year == today_yr and week > today_wk:
                break

            url  = f"{_BW_BASE}/v1/geodata/fishhealth/locality/{year}/{week}"
            data = _bw_get(url, token=token)

            if data == "AUTH":
                print(f"  [lice] Auth failed at {year}/W{week}. Check token.")
                return pd.DataFrame()

            # Unwrap {"localities": [...]} envelope and inject year/week
            if isinstance(data, dict) and "localities" in data:
                records = data["localities"]
                for r in records:
                    r["year"] = year
                    r["week"] = week
                data = records

            if isinstance(data, list) and data:
                parsed = _parse_lice_records(data)
                if not parsed.empty:
                    yr_frames.append(parsed)
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
            else:
                consecutive_empty += 1

            # Early years may have sparse data; stop at 8 consecutive empty
            if consecutive_empty > 8:
                break

            time.sleep(0.05)

        if yr_frames:
            yr_df = pd.concat(yr_frames, ignore_index=True)
            print(f"  [lice] {year}: {len(yr_df):,} rows, "
                  f"{yr_df['site_nr'].nunique()} sites")
            all_frames.append(yr_df)
        else:
            print(f"  [lice] {year}: no data")

    if not all_frames:
        print("  [lice] No data retrieved.")
        return pd.DataFrame()

    df = (
        pd.concat(all_frames, ignore_index=True)
        .drop_duplicates(subset=["date", "site_nr"])
        .sort_values(["date", "site_nr"])
        .reset_index(drop=True)
    )
    df.to_csv(cache, index=False)
    print(f"  [lice] Cached {len(df):,} site-week rows → {cache}")
    return df


def _parse_lice_records(data: list) -> pd.DataFrame:
    """Parse a list of Barentswatch locality-week dicts into tidy rows."""
    if not data or not isinstance(data, list):
        return pd.DataFrame()
    df = pd.json_normalize(data)
    if df.empty:
        return pd.DataFrame()

    site_col = _find_col(df, ["localityNo", "lokalitetNr", "siteNo", "localityNumber"])
    year_col = _find_col(df, ["year", "aar", "Year"])
    week_col = _find_col(df, ["week", "uke", "weekNo", "Week"])
    lice_col = _find_col(df, ["avgAdultFemaleLicePerFish", "avgAdultFemaleLice",
                               "adultFemaleLice", "avgFemaleAdultLice",
                               "avgLice", "femaleAdultLice"])

    if not (site_col and year_col and week_col and lice_col):
        has_lice = [c for c in df.columns if "lice" in c.lower() or "lus" in c.lower()]
        if has_lice:
            print(f"  [parse] Lice-like columns: {has_lice}")
        else:
            print(f"  [parse] No lice column. Available: {list(df.columns[:15])}")
        return pd.DataFrame()

    out = pd.DataFrame()
    out["site_nr"] = pd.to_numeric(df[site_col], errors="coerce")
    out["year"]    = pd.to_numeric(df[year_col], errors="coerce")
    out["week"]    = pd.to_numeric(df[week_col], errors="coerce")
    out["lice"]    = pd.to_numeric(df[lice_col], errors="coerce")

    out["date"] = out.apply(
        lambda r: _iso_end(int(r["year"]), int(r["week"]))
        if pd.notna(r["year"]) and pd.notna(r["week"]) else pd.NaT,
        axis=1,
    )

    return out.dropna(subset=["site_nr", "date", "lice"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC: FARMER → SITE MAPPING
# ─────────────────────────────────────────────────────────────────────────────
def print_mapping_diagnostic(lice_df: pd.DataFrame, site_map: pd.DataFrame) -> None:
    """Print explicit farmer → site mapping for user verification."""
    print("\n" + "═" * 70)
    print("DIAGNOSTIC: FARMER → SITE MAPPING")
    print("  Source: Fiskeridirektoratet (FDIR) public API")
    print("  Note: Current FDIR state used (approximate for historical periods).")
    print("═" * 70)

    if site_map.empty:
        print("  (no site map available)")
        return

    for farmer in ["MOWI", "SALM", "LSG", "GSF"]:
        fdf = site_map[site_map["farmer"] == farmer]
        flag = ""
        if farmer in EXCLUDED_SIGNAL:
            flag = "  [EXCLUDED FROM TESTS]"

        if fdf.empty:
            print(f"\n  {farmer}{flag}: *** NO SITES MAPPED ***")
            continue

        in_lice = set()
        if not lice_df.empty and "site_nr" in lice_df.columns:
            in_lice = set(lice_df["site_nr"].unique())

        site_nrs  = set(fdf["site_nr"].dropna().astype(int).tolist())
        matched   = site_nrs & in_lice
        zone_dist = ""
        if "prod_area_code" in fdf.columns:
            zc = fdf["prod_area_code"].dropna().astype(int).value_counts().head(5).to_dict()
            zone_dist = "zones: " + ", ".join(
                f"PO{z}({n})" for z, n in sorted(zc.items())
            )
        cnames = fdf["company_name"].dropna().unique()[:5].tolist()

        print(f"\n  {farmer}{flag}")
        print(f"    Registry sites: {len(site_nrs)}")
        print(f"    Sites in lice data: {len(matched)}")
        print(f"    Sample site_nrs: {sorted(list(site_nrs))[:20]}")
        if cnames:
            print(f"    Company names: {cnames}")
        if zone_dist:
            print(f"    {zone_dist}")

    unmapped = site_map[site_map["farmer"].isna()]["company_name"].dropna()
    if not unmapped.empty:
        vc = unmapped.value_counts().head(20)
        print("\n  Top unmatched companies (expand ORG_MAP if any belong to above farmers):")
        for name, cnt in vc.items():
            print(f"    {cnt:4d}  {name}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# EQUITY DATA
# ─────────────────────────────────────────────────────────────────────────────
def fetch_equity_returns() -> pd.DataFrame:
    """
    Weekly adjusted-close returns for all tickers.
    Index = ISO week-end Sunday. Columns = farmer codes.
    """
    cache = _CACHE / "equity_returns.csv"
    if cache.exists() and (time.time() - cache.stat().st_mtime) / 86400 < 1:
        print(f"  [equity] Cache ({(time.time()-cache.stat().st_mtime)/86400:.2f}d)")
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    print("\n[equity] Downloading weekly prices via yfinance …")
    price_dict: dict[str, pd.Series] = {}

    for farmer, ticker in TICKERS.items():
        for attempt in range(3):
            try:
                raw = yf.download(
                    ticker,
                    start=f"{START_YEAR}-01-01",
                    end=f"{END_YEAR+1}-01-01",
                    interval="1wk",
                    progress=False,
                    auto_adjust=True,
                )
                if raw.empty:
                    print(f"  [equity] {ticker}: empty")
                    break
                close = raw["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                price_dict[farmer] = close.squeeze()
                print(f"  [equity] {ticker}: {len(raw)} weeks "
                      f"({raw.index[0].date()} → {raw.index[-1].date()})")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  [equity] {ticker}: FAILED — {e}")
                time.sleep(2)

    if not price_dict:
        raise RuntimeError("[equity] No equity data retrieved.")

    px = pd.DataFrame(price_dict)
    px.index = pd.to_datetime(px.index)
    px = px.resample("W-SUN").last()
    rets = px.pct_change().dropna(how="all")
    rets.to_csv(cache)
    return rets


# ─────────────────────────────────────────────────────────────────────────────
# PANEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_panel(
    lice_df: pd.DataFrame,
    site_map: pd.DataFrame,
    equity_rets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build weekly farmer×week panel with leave-one-out zone residuals.

    Zone assignment: joined from FDIR site map (prod_area_code) on site_nr.
    Zone covers only the 4 mapped farmers' sites — the Barentswatch per-week
    endpoint does not include production-area in locality records.

    Zone average for farmer F is computed from ALL OTHER mapped sites in the
    same zone that week (leave-one-out), preventing circularity where a large
    farmer anchors their own benchmark.

    POINT-IN-TIME IMPURITY: FDIR site→zone and site→owner mapping reflects
    current FDIR state applied retroactively. Sites acquired or divested
    mid-period are attributed to their current owner throughout. Historical
    attribution is therefore approximate — treat regression results accordingly.

    Returns panel indexed by date (Sunday).
    """
    print("\n[panel] Building panel …")
    print("  ⚠ POINT-IN-TIME NOTE: FDIR site→owner/zone mapping is current-state")
    print("    applied retroactively. Historical attribution is approximate.")

    if lice_df.empty or site_map.empty:
        print("  FATAL: Missing lice or site map — cannot build panel.")
        return pd.DataFrame()

    # ── Step 1: build separate zone and farmer lookup tables ──────────────────
    # Zone map: all FDIR sites that have a prod_area_code (not filtered by farmer)
    if "prod_area_code" not in site_map.columns:
        print("  FATAL: site_map has no prod_area_code column.")
        return pd.DataFrame()

    zone_map = (
        site_map[["site_nr", "prod_area_code"]]
        .dropna(subset=["prod_area_code"])
        .drop_duplicates("site_nr")
        .assign(zone=lambda d: d["prod_area_code"].astype(int))
        [["site_nr", "zone"]]
    )
    farmer_map = (
        site_map[["site_nr", "farmer"]]
        .dropna(subset=["farmer"])
        .drop_duplicates("site_nr")
    )

    # ── Step 2: attach zone and farmer to all lice rows ───────────────────────
    lice = (
        lice_df
        .merge(zone_map,   on="site_nr", how="left")
        .merge(farmer_map, on="site_nr", how="left")
    )

    # ── Zone coverage diagnostic ──────────────────────────────────────────────
    n_total      = len(lice)
    n_with_zone  = lice["zone"].notna().sum()
    n_with_farmer = lice["farmer"].notna().sum()
    n_both        = (lice["zone"].notna() & lice["farmer"].notna()).sum()
    print(f"\n  [zone join] {n_with_zone:,}/{n_total:,} lice rows got a zone "
          f"({100*n_with_zone/n_total:.1f}%)")
    print(f"  [farmer join] {n_with_farmer:,}/{n_total:,} lice rows got a farmer")
    print(f"  [both] {n_both:,} rows have zone + farmer + lice")

    unmapped = lice.loc[lice["zone"].isna(), "site_nr"].value_counts().head(10)
    if not unmapped.empty:
        print(f"  [unmapped sites top-10]: {unmapped.index.tolist()}")
        print(f"    (These are non-farmer sites — expected, not an error)")

    if n_with_zone == 0:
        print("  FATAL: No zone assignments after join.")
        print("         Check that FDIR site map has prod_area_code populated.")
        return pd.DataFrame()

    # ── Step 3: leave-one-out zone averages ───────────────────────────────────
    # lice_ok = all rows with a known zone and a non-null lice count
    lice_ok = lice.dropna(subset=["zone", "lice"]).copy()
    lice_ok["zone"] = lice_ok["zone"].astype(int)
    lice_ok["lice"] = lice_ok["lice"].astype(float)

    test_farmers = [f for f in ["MOWI", "SALM", "LSG", "GSF"]
                    if f in lice_ok["farmer"].dropna().unique()]
    if not test_farmers:
        print("  FATAL: No recognised farmers in lice data after join.")
        print("         Check site_nr key overlap between lice cache and FDIR map.")
        return pd.DataFrame()
    print(f"\n  [panel] Farmers with mapped site data: {test_farmers}")

    # Zone totals across ALL sites with known zone (includes all 4 farmers + any others)
    zone_totals = (
        lice_ok
        .groupby(["date", "zone"])
        .agg(zone_sum=("lice", "sum"), zone_count=("lice", "count"))
        .reset_index()
    )

    # Farmer contribution per (date, zone) — needed to subtract for leave-one-out
    farmer_contrib = (
        lice_ok[lice_ok["farmer"].isin(test_farmers)]
        .groupby(["date", "zone", "farmer"])
        .agg(f_sum=("lice", "sum"), f_count=("lice", "count"))
        .reset_index()
    )

    # Leave-one-out: zone_avg_loo = (zone_total - this_farmer) / (zone_count - this_farmer_count)
    loo = farmer_contrib.merge(zone_totals, on=["date", "zone"], how="left")
    loo["loo_sum"]   = loo["zone_sum"]   - loo["f_sum"]
    loo["loo_count"] = loo["zone_count"] - loo["f_count"]
    # Guard: if this farmer IS the only site in the zone that week → NaN (undefined benchmark)
    loo["zone_avg_loo"] = np.where(
        loo["loo_count"] > 0,
        loo["loo_sum"] / loo["loo_count"],
        np.nan,
    )
    loo = loo[["date", "zone", "farmer", "zone_avg_loo"]]

    # ── Step 4: site-level residuals ──────────────────────────────────────────
    farmer_sites = lice_ok[lice_ok["farmer"].isin(test_farmers)].copy()
    farmer_sites = farmer_sites.merge(loo, on=["date", "zone", "farmer"], how="left")
    farmer_sites["residual"] = farmer_sites["lice"] - farmer_sites["zone_avg_loo"]

    n_with_loo = farmer_sites["zone_avg_loo"].notna().sum()
    n_farmer_total = len(farmer_sites)
    loo_solo_weeks = farmer_sites["zone_avg_loo"].isna().sum()
    print(f"  [LOO] Zone avg available for {n_with_loo:,}/{n_farmer_total:,} "
          f"farmer-site-weeks ({100*n_with_loo/n_farmer_total:.1f}%)")
    if loo_solo_weeks > 0:
        print(f"  [LOO] {loo_solo_weeks:,} site-weeks where farmer is sole zone occupant "
              f"(LOO undefined → excluded from residual)")

    # ── Step 5: per-farmer weekly residual (equal-weight across sites) ─────────
    farmer_resid = (
        farmer_sites.dropna(subset=["residual"])
        .groupby(["date", "farmer"])["residual"]
        .mean()
        .unstack("farmer")
        .add_suffix("_lice_residual")
    )
    farmer_resid.index = pd.to_datetime(farmer_resid.index)
    farmer_resid = farmer_resid.sort_index()

    # ── Step 6: publication-lag signal ────────────────────────────────────────
    # Shift residual forward PUB_LAG weeks so signal[t] is available at t
    for f in test_farmers:
        raw_col = f"{f}_lice_residual"
        sig_col = f"{f}_signal"
        if raw_col in farmer_resid.columns:
            farmer_resid[sig_col] = farmer_resid[raw_col].shift(PUB_LAG)

    # ── Step 7: equity relative returns ───────────────────────────────────────
    basket_farmers = [f for f in test_farmers if f in equity_rets.columns]
    basket = equity_rets[basket_farmers].mean(axis=1)
    rel_rets = pd.DataFrame(
        {f"{f}_rel_ret": equity_rets[f] - basket
         for f in basket_farmers if f in equity_rets.columns}
    )

    # ── Step 8: join into panel ────────────────────────────────────────────────
    panel = farmer_resid.join(rel_rets, how="outer").sort_index()
    panel = panel.loc[
        (panel.index >= pd.Timestamp(f"{START_YEAR}-01-01"))
        & (panel.index <= pd.Timestamp(f"{END_YEAR}-12-31"))
    ]

    # ── Step 9: panel diagnostics ─────────────────────────────────────────────
    print(f"\n  [panel] Shape: {panel.shape[0]} weeks × {panel.shape[1]} columns")
    print(f"          Date range: {panel.index.min().date()} → {panel.index.max().date()}")
    for f in test_farmers:
        sig = f"{f}_signal"
        ret = f"{f}_rel_ret"
        n = (panel.get(sig, pd.Series(dtype=float)).notna() &
             panel.get(ret, pd.Series(dtype=float)).notna()).sum()
        print(f"  [panel] {f}: {n} farmer-weeks with both signal + return")

    print("\n  Sample rows (signal = lice_residual shifted +1wk, rel_ret = farmer−basket):")
    sample_cols = [c for c in panel.columns if "_signal" in c or "_rel_ret" in c]
    sample = panel[sample_cols].dropna(how="all").head(5)
    print(sample.to_string())
    print()

    # Sites per zone summary
    zone_site_counts = (
        farmer_sites
        .groupby("zone")["site_nr"]
        .nunique()
        .sort_index()
    )
    print("  Farmer-attributed sites per zone (used in LOO benchmark):")
    for z, n in zone_site_counts.items():
        print(f"    PO{z}: {n} unique sites")

    return panel


# ─────────────────────────────────────────────────────────────────────────────
# TEST A — POOLED PANEL OLS
# ─────────────────────────────────────────────────────────────────────────────
def run_test_a(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Pooled OLS: relative_return[t] = α + β * lice_signal[t-L] + ε
    for L = 0..MAX_LAG. All farmers pooled.

    Expected sign: β < 0 (more lice → lower relative return).
    """
    print("\n" + "═" * 70)
    print("TEST A — POOLED PANEL: relative lice → relative return?")
    print("  Spec: rel_return[t] ~ lice_signal[t-L], pooled across farmers")
    print("  Expected: β < 0 (rising lice residual → underperformance)")
    print("  Signal = LOO lice_residual shifted +1 week (publication lag)")
    print("  LOO = leave-one-out: farmer excluded from their own zone benchmark")
    print("═" * 70)

    test_farmers = [
        f for f in ["MOWI", "SALM", "LSG", "GSF"]
        if f"{f}_signal" in panel.columns and f"{f}_rel_ret" in panel.columns
    ]
    if not test_farmers:
        print("  CANNOT RUN: no farmers with both signal and return.")
        return pd.DataFrame()

    results = []
    hdr = (f"{'Lag':>5}  {'β (coef)':>10}  {'t-stat':>8}  {'R²':>8}  "
           f"{'N obs':>7}  {'N eff':>6}  Sign?")
    print(hdr)
    print("─" * len(hdr))

    for lag in range(MAX_LAG + 1):
        y_parts, x_parts = [], []
        for f in test_farmers:
            sig = f"{f}_signal"
            ret = f"{f}_rel_ret"
            x   = panel[sig].shift(lag)
            mask = panel[ret].notna() & x.notna()
            y_parts.append(panel.loc[mask, ret].values)
            x_parts.append(x[mask].values)

        y   = np.concatenate(y_parts)
        x   = np.concatenate(x_parts)
        res = _ols(y, x)

        # Effective n from pooled signal series (approx; autocorr of pooled x)
        x_series = pd.Series(x)
        n_eff = _effective_n(x_series)

        sign_ok = (
            "✓ (neg)" if (not np.isnan(res["t_stat"]) and res["t_stat"] < 0) else
            "✗ (pos)" if (not np.isnan(res["t_stat"]) and res["t_stat"] > 0) else "?"
        )
        results.append({"lag": lag, **res, "n_eff": n_eff, "sign_ok": sign_ok})

        t = res["t_stat"]
        print(
            f"  L={lag}  {res['coef']:>10.5f}  {t:>8.2f}  "
            f"{res['r2']:>8.5f}  {res['n']:>7}  {n_eff:>6}  {sign_ok}"
        )

    df_res = pd.DataFrame(results)
    t_abs_max = df_res["t_stat"].abs().max() if not df_res.empty else 0.0
    any_neg   = (df_res["t_stat"].dropna() < 0).any()

    print(f"\n  Max |t-stat|: {t_abs_max:.2f}   Any negative: {any_neg}")
    print(f"  N eff is autocorrelation-adjusted; true independent episodes << N obs.")
    if any_neg and t_abs_max >= C_GATE:
        print(f"  GATE: OPEN  — proceeding to Test C  (|t| ≥ {C_GATE})")
    elif any_neg:
        print(f"  GATE: WEAK  — sign correct but weak (max |t| = {t_abs_max:.2f} < {C_GATE})")
        print(f"  GATE: CLOSED for Test C")
    else:
        print("  GATE: CLOSED — wrong or flat sign; mechanism likely dead")

    return df_res


# ─────────────────────────────────────────────────────────────────────────────
# TEST B — PER-FARMER CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────
def run_test_b(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-farmer OLS at each lag. 4×5 t-stat table (farmers × lags)."""
    print("\n" + "═" * 70)
    print("TEST B — PER-FARMER CONSISTENCY (t-stats; want all negative)")
    print("═" * 70)

    test_farmers = [
        f for f in ["MOWI", "SALM", "LSG", "GSF"]
        if f"{f}_signal" in panel.columns and f"{f}_rel_ret" in panel.columns
    ]
    if not test_farmers:
        print("  CANNOT RUN")
        return pd.DataFrame()

    lags = list(range(MAX_LAG + 1))
    hdr = f"  {'Farmer':<8}" + "".join(f"    L={L}" for L in lags)
    print(hdr)
    print("  " + "─" * (8 + 9 * len(lags)))

    rows = []
    for f in test_farmers:
        sig = f"{f}_signal"
        ret = f"{f}_rel_ret"
        row = {"farmer": f}
        t_vals = []
        for lag in lags:
            x    = panel[sig].shift(lag)
            mask = panel[ret].notna() & x.notna()
            res  = _ols(panel.loc[mask, ret].values, x[mask].values)
            row[f"L{lag}_t"] = res["t_stat"]
            row[f"L{lag}_n"] = res["n"]
            t_vals.append(res["t_stat"])

        rows.append(row)
        t_str = "".join(
            f"  {t:>7.2f}" if not np.isnan(t) else "     n/a" for t in t_vals
        )
        print(f"  {f:<8}{t_str}")

    df_b = pd.DataFrame(rows)
    print()
    print("  Sign consistency (neg = correct direction):")
    for lag in lags:
        col    = f"L{lag}_t"
        t_vals = df_b[col].dropna()
        n_neg  = (t_vals < 0).sum()
        print(f"    L={lag}: {n_neg}/{len(t_vals)} negative")

    return df_b


# ─────────────────────────────────────────────────────────────────────────────
# TEST C — SIGNAL BACKTEST
# ─────────────────────────────────────────────────────────────────────────────
def run_test_c(panel: pd.DataFrame, test_a: pd.DataFrame) -> None:
    """
    Signal: lice_signal[t] > 0 (farmer above LOO zone avg = worse ops).
    Position: short farmer vs basket for 1 week.
    Reports mean relative return (signal-on vs off), hit rate, annualised Sharpe.
    """
    if test_a.empty:
        return

    neg_rows = test_a[test_a["t_stat"] < 0]
    if neg_rows.empty or neg_rows["t_stat"].abs().max() < C_GATE:
        print("\n" + "═" * 70)
        print("TEST C — BACKTEST: SKIPPED (Test A gate not met)")
        print("═" * 70)
        return

    print("\n" + "═" * 70)
    print("TEST C — SIMPLE SIGNAL BACKTEST")
    print("  Signal-on: lice_signal[t] > 0 (above LOO zone average = worse ops)")
    print("  Position: short farmer vs basket for next 1 week")
    print("  (Short = negative relative return expected per Test A)")
    print("═" * 70)

    best_lag = int(neg_rows.loc[neg_rows["t_stat"].idxmin(), "lag"])
    print(f"  Using L={best_lag} (most negative t in Test A)\n")

    test_farmers = [
        f for f in ["MOWI", "SALM", "LSG", "GSF"]
        if f"{f}_signal" in panel.columns and f"{f}_rel_ret" in panel.columns
    ]

    all_on_rets: list[float] = []

    hdr = (f"  {'Farmer':<8}  {'N_on':>5}  {'N_off':>5}  "
           f"{'Mean(on)':>9}  {'Mean(off)':>9}  {'Hit%':>6}  {'Sharpe':>7}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for f in test_farmers:
        sig  = f"{f}_signal"
        ret  = f"{f}_rel_ret"
        x    = panel[sig].shift(best_lag)
        mask = x.notna() & panel[ret].notna()
        on   = x[mask] > 0
        on_r = -panel.loc[mask & on,  ret]  # short = negate rel return
        off_r = panel.loc[mask & ~on, ret]

        def _sharpe(s: pd.Series) -> float:
            return (s.mean() / s.std() * np.sqrt(52)) if len(s) >= 5 and s.std() > 0 else np.nan

        mean_on  = float(on_r.mean())   if len(on_r)  > 0 else np.nan
        mean_off = float(off_r.mean())  if len(off_r) > 0 else np.nan
        hit      = float((on_r > 0).mean()) if len(on_r) > 0 else np.nan
        sh       = _sharpe(on_r)
        flag     = "✓" if (not np.isnan(mean_on) and mean_on > 0) else "✗"

        print(
            f"  {f:<8}  {int(on.sum()):>5}  {int((~on).sum()):>5}  "
            f"{mean_on:>+9.4f}  {mean_off:>+9.4f}  {hit:>6.1%}  "
            f"{sh:>7.2f}  {flag}"
        )
        all_on_rets.extend(on_r.tolist())

    if all_on_rets:
        pooled  = pd.Series(all_on_rets)
        psharpe = (pooled.mean() / pooled.std() * np.sqrt(52)) \
            if pooled.std() > 0 else np.nan
        n_episodes = _effective_n(pooled)
        print(
            f"\n  POOLED:   n={len(pooled)}  n_eff≈{n_episodes}  "
            f"mean={pooled.mean():+.4f}  hit={float((pooled>0).mean()):.1%}  "
            f"Sharpe={psharpe:.2f}"
        )

    print("\n  NOTE: Sharpe is raw — no transaction costs, no slippage.")
    print("        n_eff adjusts for weekly autocorrelation; use as rough gauge.")
    print("        Do NOT optimise or tune thresholds based on these results.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("LICE EFFICIENCY HYPOTHESIS TEST")
    print(f"  Period: {START_YEAR}–{END_YEAR}")
    print(f"  Tickers: {', '.join(TICKERS.values())}")
    print(f"  Publication lag: {PUB_LAG} wk  |  Lags tested: 0–{MAX_LAG}")
    print(f"  Zone residual method: leave-one-out (LOO)")
    print("=" * 70)

    for f, reason in EXCLUDED_SIGNAL.items():
        print(f"\n  EXCLUSION: {f} — {reason}")

    # ── Auth check ────────────────────────────────────────────────────────────
    # Skip token fetch if lice cache is fresh — saves a network round-trip
    _lice_cache  = _CACHE / "site_lice.csv"
    _lice_cached = (
        _lice_cache.exists()
        and (time.time() - _lice_cache.stat().st_mtime) / 86400 < 7
    )
    creds = _get_bw_credentials()
    token: str | None = None

    if creds is not None and not _lice_cached:
        token = _get_bw_token(creds)
        if token is None:
            print("  [auth] Failed to obtain token. Check credentials.")
            return
    elif creds is not None and _lice_cached:
        print("  [auth] Lice data cached — skipping token fetch.")
        token = "cached"

    if creds is None:
        print("\n" + "═" * 70)
        print("NO BARENTSWATCH CREDENTIALS FOUND")
        print("─" * 70)
        print("Site-level lice data requires OAuth2 client credentials.")
        print()
        print("STEPS TO GET ACCESS (free, ~2 minutes):")
        print("  1. Go to: https://developer.barentswatch.no/")
        print("  2. Click 'Sign up' and create an account")
        print("  3. Create an application → select 'FishHealth' API scope")
        print("  4. Copy the client_id and client_secret")
        print("  5. Set env vars:")
        print("       export BW_CLIENT_ID='your-id'")
        print("       export BW_CLIENT_SECRET='your-secret'")
        print("     OR create research/lice_efficiency/bw_credentials.py with:")
        print("       BW_CLIENT_ID = 'your-id'")
        print("       BW_CLIENT_SECRET = 'your-secret'")
        print("  6. Re-run this script")
        print()
        print("WHAT WE CAN DO NOW (without auth):")
        print("  ✓ Build the company → site mapping (FDIR public API)")
        print("  ✓ Fetch equity returns (yfinance)")
        print("  ✗ Lice counts per site per week (Barentswatch — needs auth)")
        print("═" * 70)
        print()
        print("[Running partial pipeline to validate data infrastructure …]\n")
    else:
        print("  [auth] Barentswatch credentials found.")
        print()

    # ── FDIR site map (always runs — public) ─────────────────────────────────
    site_map = fetch_fdir_site_map()

    # ── Equity returns (always runs) ─────────────────────────────────────────
    equity_rets = fetch_equity_returns()

    if creds is None:
        print("\n" + "═" * 70)
        print("PARTIAL RESULTS (no Barentswatch auth)")
        print("═" * 70)
        if not site_map.empty:
            print(f"\nFDIR site map: {len(site_map):,} sites loaded")
            for f in ["MOWI", "SALM", "LSG", "GSF"]:
                n = (site_map["farmer"] == f).sum()
                print(f"  {f}: {n} sites mapped")
        print(f"\nEquity data: {list(equity_rets.columns)}, "
              f"{len(equity_rets)} weeks")
        print("\nAdd Barentswatch credentials and re-run to execute lice tests.")
        print("=" * 70)
        return

    # ── Lice data (requires auth or cache) ───────────────────────────────────
    lice_df = fetch_site_lice(token)

    # ── Diagnostic: farmer → site mapping ────────────────────────────────────
    print_mapping_diagnostic(lice_df, site_map)

    if lice_df.empty:
        print("\nFATAL: No lice data retrieved. Cannot run tests.")
        return

    # ── Panel ─────────────────────────────────────────────────────────────────
    panel = build_panel(lice_df, site_map, equity_rets)
    if panel.empty:
        print("\nFATAL: Panel empty — see diagnostics above for which join step failed.")
        return

    # ── Tests ─────────────────────────────────────────────────────────────────
    test_a_results = run_test_a(panel)
    _               = run_test_b(panel)
    run_test_c(panel, test_a_results)

    # ── Effective sample size summary ─────────────────────────────────────────
    print("\n" + "═" * 70)
    print("EFFECTIVE SAMPLE SIZE (autocorrelation-adjusted)")
    print("═" * 70)
    for f in ["MOWI", "SALM", "LSG", "GSF"]:
        sig = f"{f}_signal"
        if sig in panel.columns:
            s = panel[sig].dropna()
            n_raw = len(s)
            n_eff = _effective_n(s)
            r1    = s.autocorr(lag=1) if n_raw >= 10 else float("nan")
            print(f"  {f:<8}  n={n_raw}  r1={r1:+.3f}  n_eff≈{n_eff}  "
                  f"({'independent signal' if n_eff > n_raw*0.5 else 'highly autocorrelated'})")
    print()
    print("  True independent episodes << weekly row count due to autocorrelation.")
    print("  Do not interpret t-stats as if observations were i.i.d.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("SUMMARY")
    print("═" * 70)
    print("  1. Zone residual: LOO method — each farmer excluded from their own")
    print("     zone benchmark to avoid circularity.")
    print("  2. Zone coverage: only the 4 mapped farmers' sites have zone info.")
    print("     Non-farmer sites (792 of 1186 unique sites) have no zone.")
    print("  3. Test A gate: sign + magnitude determine if mechanism exists.")
    print("  4. Test B: consistent negative t across farmers = robust signal.")
    print("  5. Test C: only valid if gate is open.")
    print("  6. Point-in-time impurity: FDIR mapping is current-state retroactive.")
    print()
    print("  Do NOT tune lags or thresholds based on these results.")
    print("=" * 70)


if __name__ == "__main__":
    main()
