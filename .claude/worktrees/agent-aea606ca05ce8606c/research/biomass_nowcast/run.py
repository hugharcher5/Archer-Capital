#!/usr/bin/env python3
"""
Salmon supply-nowcast — first hypothesis test.
research/biomass_nowcast/run.py

Hypothesis: standing salmon biomass in Norwegian waters today predicts future
harvest supply, which predicts future salmon price (more biomass → more supply
→ lower price, 6–18 months forward).

We test two links:
  A (gate)  — does biomass YoY predict future harvest volume YoY?
  B         — does biomass YoY predict future salmon price YoY?
              Expected sign is NEGATIVE. Flag any wrong or flip signs.
  C         — does it predict returns BEYOND the forward curve?
              (Honest alpha test. Fish Pool forwards require a subscription;
               this section is a clearly-labelled skip with commentary.)
  D         — simple signal backtest (only if A and B hold).

Point-in-time: every series is lagged by its real publication delay before it
can inform any signal. All tests run on year-over-year (YoY) log changes to
remove the dominant seasonal cycle — raw levels are unusable here.

Run once:
    venv/bin/python research/biomass_nowcast/run.py

Do NOT tune any parameters to improve results.
"""

import sys, os, io, time, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

START          = "2005-01-01"
HORIZONS       = [6, 9, 12, 15, 18]   # forward prediction horizons in months
FARMER_TICKERS = ["MOWI.OL", "SALM.OL", "LSG.OL", "BAKKA.OL", "GSF.OL"]

# Real publication lags (months) applied to each series.
# At row month-end T, the value shown is what was known as of T.
#   biomass : Fiskeridirektoratet releases ~6 weeks after month-end → 2 months
#   harvest : SSB monthly slaughter table released ~4-6 weeks after → 1 month
#   price   : SSB weekly export data, within 2 weeks → 0 months for monthly panel
#   equity  : Live daily prices → 0 months
PUB_LAG = {"biomass": 2, "harvest": 1, "price": 0, "equity": 0}

SIGNAL_COST_BPS = 10   # PLACEHOLDER round-trip transaction cost
MIN_YOY_WINDOW  = 12   # months needed before YoY series begins


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _rule(n=72):  print("─" * n)
def _hdr(t):      _rule(); print(f"  {t}"); _rule()
def _note(msg):   print(f"  NOTE: {msg}")
def _skip(msg):   print(f"  [SKIP] {msg}")
def _warn(msg):   print(f"  WARNING: {msg}")


def _http_get(url, timeout=30):
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2)


def _ssb_meta(table_id):
    return _http_get(f"https://data.ssb.no/api/v0/en/table/{table_id}/").json()


def _ssb_post(table_id, query, timeout=60):
    url = f"https://data.ssb.no/api/v0/en/table/{table_id}/"
    for attempt in range(3):
        try:
            r = requests.post(url, json=query, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2)


def _parse_date_label(s):
    """
    Parse SSB/Fiskeridirektoratet time labels to pd.Timestamp.
    Handles: 'YYYYMmm', 'YYYYUww' (Norwegian week), 'YYYYWww', ISO strings.
    """
    s = str(s).strip()
    try:
        if len(s) == 7 and "M" in s:          # "2005M01" → monthly
            y, m = s.split("M")
            return pd.Timestamp(int(y), int(m), 1)
        if "U" in s:                           # "2005U01" → Norwegian ISO week
            return pd.to_datetime(s.replace("U", "W") + "-5", format="%GW%V-%u")
        if "W" in s and "-" not in s:          # "2005W01"
            return pd.to_datetime(s + "-5", format="%GW%V-%u")
        return pd.to_datetime(s)               # ISO / fallback
    except Exception:
        return pd.NaT


def _jsonstat2_to_series(data, value_name="value"):
    """
    Convert a JSON-stat2 response to a pd.Series indexed by date.

    Handles 1-D and filtered multi-D responses (where non-time dims have size 1).
    Assumes the time dimension is named 'Tid', 'Time', 'Uke', or similar.
    """
    dim_ids = data["id"]
    dims    = data["dimension"]
    values  = data["value"]
    sizes   = data["size"]

    tid_key = next(
        (k for k in dim_ids if k.lower() in ("tid", "time", "week", "uke")),
        dim_ids[-1],
    )
    tid_idx     = dim_ids.index(tid_key)
    time_labels = list(dims[tid_key]["category"]["index"].keys())
    n_time      = sizes[tid_idx]

    n_total = 1
    for s in sizes:
        n_total *= s
    n_other = n_total // n_time if n_time > 0 else 1

    if tid_idx == 0:
        # Time-first layout: [t0_x0, t0_x1, …, t1_x0, …]  → stride by n_other
        slice_vals = values[::n_other]
    else:
        # Time-last or middle layout: take the first n_time values (first slice
        # of all other dims, which are each size 1 because we filtered them).
        slice_vals = values[:n_time]

    dates = pd.DatetimeIndex([_parse_date_label(w) for w in time_labels])
    s = pd.Series(
        [float(v) if v is not None else np.nan for v in slice_vals],
        index=dates,
        name=value_name,
    )
    return s[s.index.notna()].sort_index()


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_salmon_price():
    """
    SSB table 03024: weekly salmon export price (NOK/kg).
    VareGrupper2='01' (farmed salmon), ContentsCode='Kilopris'.
    Known to work — confirmed from feed-squeeze pipeline.
    Publication lag: 0 months.
    """
    query = {
        "query": [
            {"code": "VareGrupper2", "selection": {"filter": "item", "values": ["01"]}},
            {"code": "ContentsCode",  "selection": {"filter": "item", "values": ["Kilopris"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    data = _ssb_post("03024", query)
    s = _jsonstat2_to_series(data, "salmon_price_nok")
    return s[s.index >= START].resample("ME").last()


def fetch_harvest_volume():
    """
    Weekly salmon export volume from SSB table 03024, ContentCode='Vekt' (tonnes).

    SSB terminated its dedicated monthly aquaculture slaughter statistics in 2019
    and transferred the series to Fiskeridirektoratet (no public API exists there).
    The previous table IDs (03020, 03021) no longer exist in the SSB API.

    Export volume from SSB 03024 is a close proxy for harvest volume: >95% of
    Norwegian farmed salmon is exported, and fresh salmon is exported within
    days of slaughter — so weekly export tonnage tracks weekly slaughter tightly.

    VareGrupper2='01' (fresh/chilled farmed salmon, the dominant product form).
    Publication lag: 0 months (weekly data released within 2 weeks; we treat as
    0 for the monthly panel, same as price).

    Note: if the hypothesis requires SLAUGHTER VOLUME (not export volume), the
    authoritative source is the Fiskeridirektoratet monthly Excel, which must be
    downloaded manually from fiskeridir.no (see stub_biomass_local below).
    """
    query = {
        "query": [
            {"code": "VareGrupper2", "selection": {"filter": "item", "values": ["01"]}},
            {"code": "ContentsCode",  "selection": {"filter": "item", "values": ["Vekt"]}},
        ],
        "response": {"format": "json-stat2"},
    }
    try:
        data = _ssb_post("03024", query)
        s    = _jsonstat2_to_series(data, "export_volume_tonnes")
        s    = s[s.index >= START].resample("ME").sum()   # sum weekly exports per month
        s    = s.replace(0, np.nan)

        if s.notna().sum() < 12:
            _warn(f"[harvest] SSB 03024 Vekt returned only {s.notna().sum()} obs")
            return None

        print(f"  [harvest] SSB 03024 Vekt (export volume proxy):  "
              f"{s.notna().sum()} monthly obs  "
              f"({s.first_valid_index().date()} → {s.last_valid_index().date()})")
        return s

    except Exception as exc:
        _warn(f"[harvest] SSB 03024 Vekt fetch failed: {type(exc).__name__}: {exc}")
        return None


def _try_fiskedir_biomass():
    """
    Attempt to fetch monthly standing biomass from Fiskeridirektoratet PxWeb API.
    Returns (pd.Series, table_description) or (None, None).
    """
    base = "https://statistikk.fiskeridir.no/api/v0/no"

    # Step 1: discover available databases
    r = _http_get(f"{base}/", timeout=15)
    dbs = r.json()

    # Step 2: find the aquaculture database
    akv_db = next(
        (d for d in dbs
         if any(k in d.get("text", "").lower() or k in d.get("id", "").lower()
                for k in ["akvakultur", "aquaculture", "akv"])),
        dbs[0] if dbs else None,
    )
    if akv_db is None:
        return None, None

    akv_id = akv_db["id"]
    print(f"  [biomass] Fiskeridirektoratet DB: '{akv_db.get('text','')}' (id={akv_id})")

    # Step 3: list tables in the aquaculture database
    r2     = _http_get(f"{base}/{akv_id}/", timeout=15)
    tables = r2.json()

    # Step 4: find a biomass / fish-at-sea table
    bio_table = next(
        (t for t in tables
         if any(k in t.get("text", "").lower() or k in t.get("id", "").lower()
                for k in ["biomasse", "biomass", "stående", "fisk i sjø",
                           "beholdning", "standing", "sjø"])),
        None,
    )
    if bio_table is None:
        print(f"  [biomass] No biomass table found in {akv_id}. Tables: "
              f"{[t.get('text','')[:30] for t in tables[:6]]}")
        return None, None

    table_id   = bio_table["id"]
    table_text = bio_table.get("text", "")
    print(f"    → Table '{table_text}' (id={table_id})")

    # Step 5: read table metadata
    r3   = _http_get(f"{base}/{akv_id}/{table_id}/", timeout=15)
    meta = r3.json()

    # Step 6: build query — for non-time dims pick totals or aggregates
    query_filters = []
    for var in meta.get("variables", []):
        code  = var["code"]
        vals  = var.get("values", [])
        vlabs = var.get("valueTexts", [])
        if code.lower() in ("tid", "time"):
            continue
        # For species: prefer salmon
        if any(k in code.lower() or k in " ".join(vlabs).lower()
               for k in ["art", "fisk", "species"]):
            sal_idx = next(
                (i for i, l in enumerate(vlabs)
                 if "laks" in l.lower() or "salmon" in l.lower()),
                None,
            )
            if sal_idx is not None:
                query_filters.append(
                    {"code": code, "selection": {"filter": "item", "values": [vals[sal_idx]]}}
                )
                print(f"    {code} → '{vlabs[sal_idx]}'")
                continue
        # For everything else: pick "total" / first value
        tot_idx = next(
            (i for i, l in enumerate(vlabs)
             if any(k in l.lower() for k in ["total", "alle", "all", "sum", "hele"])),
            0,
        )
        query_filters.append(
            {"code": code, "selection": {"filter": "item", "values": [vals[tot_idx]]}}
        )
        print(f"    {code} → '{vlabs[tot_idx]}'")

    # Step 7: POST the query
    post_url = f"{base}/{akv_id}/{table_id}/"
    for attempt in range(3):
        try:
            r4 = requests.post(
                post_url,
                json={"query": query_filters, "response": {"format": "json-stat2"}},
                timeout=60,
            )
            r4.raise_for_status()
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)

    s = _jsonstat2_to_series(r4.json(), "biomass_tonnes")
    s = s[s.index >= START].resample("ME").last()

    if s.notna().sum() < 12:
        _warn(f"[biomass] Fiskeridirektoratet returned only {s.notna().sum()} obs")
        return None, None

    desc = f"Fiskeridirektoratet — {table_text}"
    print(f"    → {s.notna().sum()} monthly obs "
          f"({s.first_valid_index().date()} → {s.last_valid_index().date()})")
    return s, desc


def _try_ssb_biomass():
    """
    Attempt to find standing salmon biomass in SSB aquaculture tables.
    Tries a set of plausible table IDs and accepts the first with a matching title.
    Returns (pd.Series, table_description) or (None, None).
    """
    candidates = ["03021", "07719", "06367", "06474", "03033"]
    bio_kws    = ["biomass", "fish at sea", "fisk i sjø", "stående", "standing",
                  "beholdning", "at sea", "i sjø"]

    for tid in candidates:
        try:
            meta  = _ssb_meta(tid)
            title = meta.get("title", "").lower()
            if not any(k in title for k in bio_kws):
                continue
            print(f"  [biomass] SSB {tid}: '{meta.get('title','')[:72]}'")

            # Try to filter to salmon where possible
            variables     = meta.get("variables", [])
            query_filters = []
            for var in variables:
                code  = var["code"]
                vals  = var.get("values", [])
                vlabs = var.get("valueTexts", [])
                if code.lower() in ("tid", "time"):
                    continue
                if any(k in code.lower() or k in " ".join(vlabs).lower()
                       for k in ["art", "fisk", "species"]):
                    sal_idx = next(
                        (i for i, l in enumerate(vlabs)
                         if "laks" in l.lower() or "salmon" in l.lower()),
                        None,
                    )
                    if sal_idx is not None:
                        query_filters.append({
                            "code": code,
                            "selection": {"filter": "item", "values": [vals[sal_idx]]},
                        })
                        continue
                # Default: first value (often "total")
                query_filters.append({
                    "code": code,
                    "selection": {"filter": "item", "values": [vals[0]]},
                })

            data = _ssb_post(tid, {"query": query_filters, "response": {"format": "json-stat2"}})
            s    = _jsonstat2_to_series(data, "biomass_tonnes")
            s    = s[s.index >= START].resample("ME").last()

            if s.notna().sum() < 12:
                continue

            desc = f"SSB table {tid} — {meta.get('title','')[:50]}"
            print(f"    → {s.notna().sum()} monthly obs")
            return s, desc

        except Exception as exc:
            print(f"  [biomass] SSB {tid} failed: {type(exc).__name__}: {exc}")
            continue

    return None, None


def _try_local_biomass():
    """
    Load biomass from a manually-downloaded Fiskeridirektoratet Excel file.

    INSTRUCTIONS TO ACTIVATE:
      1. Go to fiskeridir.no → Akvakultur → Statistikk akvakultur
      2. Download the monthly "Biomassestatistikk" Excel file
      3. Save it to: research/biomass_nowcast/data/biomass_fiskeridir.xlsx
      4. Re-run this script — it will parse automatically.

    The Excel typically has columns: Year, Month, Production area, Species,
    Standing biomass (tonnes). We aggregate to national monthly total for salmon.

    Publication lag: 2 months (data for month T published ~6 weeks after T).
    """
    local_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "biomass_fiskeridir.xlsx"
    )
    if not os.path.exists(local_path):
        return None, None

    print(f"  [biomass] Found local file: {local_path}")
    try:
        import io
        raw = pd.read_excel(local_path, header=None)
        # Auto-detect: look for a column with month dates and one with tonnage numbers
        # This is a best-effort parse; adjust column indices if format changes.
        # Expected: column 0 = year (int), col 1 = month (int), col N = biomass (float)
        # Try reading with pandas auto-detection first
        df = pd.read_excel(local_path)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Find date-like columns and tonnage column
        year_col  = next((c for c in df.columns if "year" in str(c).lower() or "år" in str(c).lower()), None)
        month_col = next((c for c in df.columns if "month" in str(c).lower() or "måned" in str(c).lower()), None)
        bm_col    = next((c for c in df.columns if any(k in str(c).lower()
                          for k in ["biomasse", "biomass", "tonn", "tonne", "stående"])), None)

        if year_col and month_col and bm_col:
            df["date"] = pd.to_datetime(
                {"year": df[year_col].astype(int), "month": df[month_col].astype(int), "day": 1}
            ) + pd.offsets.MonthEnd(0)
            s = df.groupby("date")[bm_col].sum()
            s = pd.to_numeric(s, errors="coerce").dropna()
            s.name = "biomass_tonnes"
            s = s[s.index >= START]
            print(f"    → {len(s)} monthly obs  ({s.index[0].date()} → {s.index[-1].date()})")
            return s, f"Fiskeridirektoratet local Excel ({os.path.basename(local_path)})"
        else:
            _warn(f"[biomass] Could not auto-detect columns in local file. Columns: {list(df.columns[:8])}")
            return None, None

    except Exception as exc:
        _warn(f"[biomass] Local file parse failed: {exc}")
        return None, None


def fetch_biomass():
    """
    Fetch monthly standing salmon biomass (tonnes at sea).

    Priority order (each falls through if unavailable):
      1. Local Fiskeridirektoratet Excel  (research/biomass_nowcast/data/biomass_fiskeridir.xlsx)
      2. Fiskeridirektoratet PxWeb API    (statistikk.fiskeridir.no — may not exist publicly)
      3. SSB aquaculture tables           (terminated 2019; annual only after that date)

    DATA SOURCE STATUS (confirmed 2026-06-14):
      Fiskeridirektoratet: 'statistikk.fiskeridir.no' does NOT resolve. Their website
        (fiskeridir.no) is JavaScript-rendered with no programmatic data API.
        Monthly biomass Excels must be downloaded manually.
      SSB: Aquaculture monthly statistics terminated in 2019 and transferred to
        Fiskeridirektoratet. Remaining SSB tables (09259, 07326) are ANNUAL, not
        monthly, and closed after 2019.

    Publication lag: 2 months (Fiskeridirektoratet publishes ~6 weeks after month-end).
    Returns (pd.Series | None, source_description | None).
    """
    # ── Priority 1: local Excel ───────────────────────────────────────────────
    try:
        s, desc = _try_local_biomass()
        if s is not None:
            return s, desc
    except Exception as exc:
        print(f"  [biomass] Local file check failed: {exc}")

    # ── Priority 2: Fiskeridirektoratet API (expected to fail) ────────────────
    try:
        s, desc = _try_fiskedir_biomass()
        if s is not None:
            return s, desc
    except Exception as exc:
        print(f"  [biomass] Fiskeridirektoratet API unreachable: {type(exc).__name__}: {exc}")

    # ── Priority 3: SSB fallback (annual data only, 2005-2019) ───────────────
    try:
        s, desc = _try_ssb_biomass()
        if s is not None:
            return s, desc
    except Exception as exc:
        print(f"  [biomass] SSB fallback failed: {type(exc).__name__}: {exc}")

    _warn("[biomass] All sources failed.")
    _note("To proceed: download 'Biomassestatistikk' Excel from fiskeridir.no")
    _note("and save to: research/biomass_nowcast/data/biomass_fiskeridir.xlsx")
    return None, None


def fetch_forwards():
    """
    Attempt Fish Pool salmon forward curve data.
    Fish Pool is a private exchange; historical forward curves require a
    subscription or API key. This function intentionally fails gracefully.

    Without forward data, Test C (the honest alpha test) cannot run.
    Any Test B result should therefore be interpreted as 'mechanism present
    but likely already priced by professional supply forecasters (Kontali,
    Rabobank, the farmers themselves)' — NOT as tradeable edge.
    """
    # Fish Pool website: https://fishpool.eu — no public historical API
    _warn("[forwards] Fish Pool forward data requires a subscription — SKIP Test C.")
    _note("Without Test C we can only test whether biomass predicts price (mechanism),")
    _note("not whether it predicts price BEYOND what the forward curve already encodes.")
    _note("Treat any Test B result as 'mechanism plausible, edge unproven'.")
    return None


def fetch_equities():
    """
    yfinance monthly equity returns for salmon farmers.
    Publication lag: 0 (live daily prices, monthly resampled).
    """
    frames   = {}
    coverage = {}
    for ticker in FARMER_TICKERS:
        t = yf.Ticker(ticker)
        hist = pd.DataFrame()
        for attempt in range(3):
            try:
                hist = t.history(start=START, interval="1wk", auto_adjust=True)
                if not hist.empty:
                    break
            except Exception:
                if attempt == 2:
                    break
            time.sleep(2)

        if hist.empty:
            _warn(f"[equity] No data for {ticker}")
            coverage[ticker] = None
            continue

        close = hist["Close"].dropna()
        close.index = (close.index.tz_localize(None)
                       if close.index.tz is not None else close.index)
        monthly = close.resample("ME").last()
        frames[ticker] = monthly
        coverage[ticker] = (close.index[0].date(), close.index[-1].date(), len(close))

    prices = pd.DataFrame(frames) if frames else pd.DataFrame()
    return prices, coverage


# ─────────────────────────────────────────────────────────────────────────────
# PANEL ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────

def build_panel(biomass, harvest, price, eq_prices, eq_coverage):
    """
    Apply publication lags, compute YoY log changes for de-seasonalisation,
    and join all series into a single monthly panel.

    De-seasonalisation via YoY log change: yoy_t = ln(x_t / x_{t-12}).
    This removes the dominant seasonal cycle without external libraries.
    Critical: raw levels in salmon data are almost entirely seasonal and will
    produce spurious regression signals if used directly.
    """
    parts = []

    # ── Salmon price (pub lag = 0) ────────────────────────────────────────────
    sp = price.shift(PUB_LAG["price"])       # 0-month shift → no-op
    sp.name = "salmon_price"
    sp_yoy = np.log(sp / sp.shift(12))
    sp_yoy.name = "salmon_price_yoy"
    parts += [sp, sp_yoy]

    # ── Biomass (pub lag = 2 months) ──────────────────────────────────────────
    if biomass is not None:
        bm = biomass.shift(PUB_LAG["biomass"])
        bm.name = "biomass"
        bm_yoy = np.log(bm / bm.shift(12))
        bm_yoy.name = "biomass_yoy"
        parts += [bm, bm_yoy]

    # ── Harvest (pub lag = 1 month) ───────────────────────────────────────────
    if harvest is not None:
        hv = harvest.shift(PUB_LAG["harvest"])
        hv.name = "harvest"
        hv_yoy = np.log(hv / hv.shift(12))
        hv_yoy.name = "harvest_yoy"
        parts += [hv, hv_yoy]

    # ── Equities (pub lag = 0) ────────────────────────────────────────────────
    farmer_cols = [t for t in FARMER_TICKERS if t in eq_prices.columns]
    if farmer_cols:
        eq_px  = eq_prices[farmer_cols]
        eq_ret = np.log(eq_px / eq_px.shift(1))
        eq_ret.columns = [t.split(".")[0] + "_ret" for t in farmer_cols]

        # Basket: equal-weight across tickers with data; require ≥2 tickers
        n_valid = eq_ret.notna().sum(axis=1)
        basket  = eq_ret.mean(axis=1)
        basket[n_valid < 2] = np.nan
        basket.name = "basket_ret"
        parts += [basket] + [eq_ret[c] for c in eq_ret.columns]

    panel = pd.concat(parts, axis=1).sort_index()
    panel = panel[panel.index >= pd.Timestamp(START) + pd.offsets.MonthEnd(0)]
    return panel


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def count_price_cycles(price_yoy: pd.Series):
    """
    Count independent salmon price cycles by identifying sustained phases
    (≥4 consecutive months) in the YoY price change above/below zero.

    One full cycle = one positive phase + one negative phase.
    This gives the TRUE effective sample size for autocorrelated monthly data.
    """
    MIN_RUN = 4
    series  = price_yoy.dropna()
    if len(series) < MIN_RUN:
        return 0, []

    sign = np.sign(series.values)
    runs = []
    cur_sign   = sign[0]
    run_start  = series.index[0]
    run_len    = 1

    for date, s in zip(series.index[1:], sign[1:]):
        if s == cur_sign:
            run_len += 1
        else:
            if run_len >= MIN_RUN:
                runs.append((run_start, date, cur_sign, run_len))
            cur_sign  = s
            run_start = date
            run_len   = 1
    if run_len >= MIN_RUN:
        runs.append((run_start, series.index[-1], cur_sign, run_len))

    n_full_cycles = len(runs) // 2
    return n_full_cycles, runs


# ─────────────────────────────────────────────────────────────────────────────
# REGRESSION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _ols(x: pd.Series, y: pd.Series):
    """OLS with NaN-safe masking. Returns (coef, t_stat, r2, n)."""
    mask = x.notna() & y.notna()
    n    = int(mask.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, n
    sl, ic, rv, pv, se = stats.linregress(x[mask].values, y[mask].values)
    tstat = sl / se if se > 0 else np.nan
    return round(sl, 6), round(tstat, 3), round(rv ** 2, 4), n


# ─────────────────────────────────────────────────────────────────────────────
# TEST A — GATE: biomass → harvest volume
# ─────────────────────────────────────────────────────────────────────────────

def test_a(panel: pd.DataFrame):
    """
    Gate test: does de-seasonalised biomass predict future harvest volume?

    Regression: harvest_yoy_{t+h} ~ biomass_yoy_t
    Expected sign: POSITIVE (more fish in the water → more fish harvested later).
    If the gate fails across all horizons, the supply-chain link is not in the
    data and the premise for Tests B/D is broken.
    """
    if "biomass_yoy" not in panel.columns or "harvest_yoy" not in panel.columns:
        return None

    bm   = panel["biomass_yoy"]
    hv   = panel["harvest_yoy"]
    rows = []

    for h in HORIZONS:
        fwd_hv = hv.shift(-h)
        coef, t, r2, n = _ols(bm, fwd_hv)
        sign_ok = ("✓ pos" if (coef or 0) > 0 else "✗ neg (WRONG)")
        rows.append({"horizon_m": h, "coef": coef, "t_stat": t,
                     "r2": r2, "n": n, "sign": sign_ok})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TEST B — biomass → future salmon price
# ─────────────────────────────────────────────────────────────────────────────

def test_b(panel: pd.DataFrame):
    """
    Does de-seasonalised biomass today predict future salmon price changes?

    Regression: salmon_price_yoy_{t+h} ~ biomass_yoy_t
    Expected sign: NEGATIVE (more biomass → more supply → lower price).
    A POSITIVE sign means the mechanism is not working as hypothesised.
    Report ALL horizons; do not select the best-looking one.
    """
    if "biomass_yoy" not in panel.columns or "salmon_price_yoy" not in panel.columns:
        return None

    bm   = panel["biomass_yoy"]
    pr   = panel["salmon_price_yoy"]
    rows = []

    for h in HORIZONS:
        fwd_pr = pr.shift(-h)
        coef, t, r2, n = _ols(bm, fwd_pr)
        sign_str = ("✓ neg" if (coef or 0) < 0 else "✗ POSITIVE (WRONG)")
        rows.append({"horizon_m": h, "coef": coef, "t_stat": t,
                     "r2": r2, "n": n, "sign": sign_str})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TEST D — simple signal backtest (only if A and B hold)
# ─────────────────────────────────────────────────────────────────────────────

def test_d(panel: pd.DataFrame):
    """
    Signal: biomass_yoy > 0 → more supply coming → short salmon basket.
           biomass_yoy ≤ 0 → short supply → long salmon basket (or flat).

    Entry: next-month (signal at T observed → position entered at T+1 close).
    Hold:  1 month (monthly rebalance).
    Target: equity basket return (1-month forward).
            If no equities, use salmon price YoY as a proxy (clearly flagged).

    Transaction cost: SIGNAL_COST_BPS bps round-trip on each transition
                      (PLACEHOLDER — flag in output).
    """
    if "biomass_yoy" not in panel.columns:
        return None

    if "basket_ret" in panel.columns:
        target_col  = "basket_ret"
        target_desc = "equity basket (1-month return)"
    elif "salmon_price_yoy" in panel.columns:
        target_col  = "salmon_price_yoy"
        target_desc = "salmon price YoY (PROXY — no equities)"
    else:
        return None

    bm            = panel["biomass_yoy"]
    target        = panel[target_col]
    signal        = (bm > 0).astype(float)
    signal_lagged = signal.shift(1)    # enter next month, no same-month look-ahead

    # When signal=1: short the basket (position = -1)
    strat = pd.Series(
        np.where(signal_lagged.fillna(False).astype(bool), -target, 0.0),
        index=target.index,
    )

    # Apply round-trip cost on every signal transition
    transitions = signal_lagged.fillna(0).astype(int).diff().abs().fillna(0)
    strat      -= transitions * (SIGNAL_COST_BPS / 10_000)

    valid  = strat.notna() & target.notna() & signal_lagged.notna()
    strat  = strat[valid]
    sig_v  = signal_lagged[valid].astype(bool)

    if sig_v.sum() < 5:
        return None

    on_ret   = strat[sig_v]
    off_ret  = strat[~sig_v]
    hit_rate = float((on_ret > 0).mean())
    sharpe   = (strat.mean() / strat.std() * np.sqrt(12)
                if strat.std() > 0 else np.nan)
    n_ep = int((signal_lagged.fillna(False).astype(int).diff() > 0).sum())

    return {
        "target":                  target_desc,
        "mean_ret_signal_on":      round(float(on_ret.mean()),  5),
        "mean_ret_signal_off":     round(float(off_ret.mean()), 5),
        "hit_rate_signal_on":      round(hit_rate,              3),
        "annualised_sharpe":       round(float(sharpe),         3),
        "n_signal_on_months":      int(sig_v.sum()),
        "n_total_months":          int(valid.sum()),
        "n_signal_episodes":       n_ep,
        "cost":                    f"{SIGNAL_COST_BPS} bps round-trip (PLACEHOLDER)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("default")

    # ── 1. Fetch data ─────────────────────────────────────────────────────────
    _hdr("SALMON SUPPLY NOWCAST — DATA FETCH")

    print("Fetching salmon price (SSB 03024) …")
    price = fetch_salmon_price()
    print(f"  → {price.notna().sum()} monthly obs  "
          f"({price.first_valid_index().date()} → {price.last_valid_index().date()})")

    print("\nFetching biomass …")
    biomass, biomass_source = fetch_biomass()

    print("\nFetching harvest volume (SSB 03020) …")
    harvest = fetch_harvest_volume()

    print("\nFetching Fish Pool forwards …")
    forwards = fetch_forwards()

    print("\nFetching equities (yfinance) …")
    eq_prices, eq_coverage = fetch_equities()
    for tk, info in eq_coverage.items():
        if info:
            print(f"  {tk:<12s} {info[0]} → {info[1]}  ({info[2]} weekly obs)")
        else:
            print(f"  {tk:<12s}  NO DATA")

    # ── 2. Assemble panel ─────────────────────────────────────────────────────
    _hdr("PANEL ASSEMBLY")
    panel = build_panel(biomass, harvest, price, eq_prices, eq_coverage)

    print(f"Panel: {panel.shape[0]} months × {panel.shape[1]} columns")
    print(f"Range: {panel.index[0].date()} → {panel.index[-1].date()}")
    print("\nNon-null months per series:")
    print(panel.count().to_string())
    print("\nPublication lags applied (point-in-time enforcement):")
    print(f"  biomass   shift({PUB_LAG['biomass']}m)  Fiskeridirektoratet ~6-week delay → 2-month conservative lag")
    print(f"  harvest   shift({PUB_LAG['harvest']}m)  SSB monthly release ~4–6 weeks → 1 month")
    print(f"  price     shift({PUB_LAG['price']}m)  SSB weekly data, within 2 weeks → 0 for monthly")
    print(f"  equity    shift({PUB_LAG['equity']}m)  Live daily prices")
    print("De-seasonalisation: YoY log change ln(x_t / x_{t-12}) on all level series.")

    # ── 3. Effective sample size ───────────────────────────────────────────────
    _hdr("DIAGNOSTIC — EFFECTIVE SAMPLE SIZE")
    print("Monthly rows are heavily autocorrelated — NOT independent observations.")
    print("True n = number of independent salmon price cycles, counted below.\n")

    sp_yoy = panel.get("salmon_price_yoy", pd.Series(dtype=float))
    n_cycles, runs = count_price_cycles(sp_yoy.dropna())

    print(f"  Salmon price YoY — sustained phases (≥4 months):")
    if runs:
        for (s, e, sgn, ln) in runs:
            direction = "+ve (price up vs yr-ago)" if sgn > 0 else "-ve (price down vs yr-ago)"
            print(f"    {s.strftime('%Y-%m')} → {e.strftime('%Y-%m')}  "
                  f"({ln:2d} months)  {direction}")
    print(f"\n  Full cycles (one +ve + one -ve phase): {n_cycles}")
    print(f"  Monthly row count: {panel.shape[0]}  ← this is NOT the sample size")

    if biomass_source:
        print(f"\n  Biomass source: {biomass_source}")
    else:
        print("\n  Biomass: UNAVAILABLE")
    if harvest is not None:
        hv_yoy = panel.get("harvest_yoy", pd.Series(dtype=float))
        print(f"  Harvest data:  {int(hv_yoy.notna().sum())} monthly obs")

    # ── 4. Gate: Test A ───────────────────────────────────────────────────────
    _hdr("TEST A (GATE) — DE-SEASONALISED BIOMASS → FUTURE HARVEST VOLUME")

    gate_pass = False

    if biomass is None:
        _skip("No biomass data — hypothesis cannot be evaluated. Stopping here.")
        _hdr("SUMMARY")
        print("  GATE FAILED: biomass data unavailable from all attempted sources.")
        print("  Cannot test the supply-chain hypothesis without standing biomass.")
        print("  Next step: obtain Fiskeridirektoratet 'biomassestatistikk' data")
        print("  (monthly Excel, page: fiskeridir.no/Akvakultur/Statistikk-akvakultur)")
        _rule()
        return

    if harvest is None:
        _skip("No harvest data — gate test cannot run.")
        print("  Proceeding to Test B without verifying the supply-chain link.")
        print("  Interpret Test B results more cautiously without the gate.\n")
        tA = None
    else:
        print("Regression: harvest_yoy_{t+h} ~ biomass_yoy_t")
        print("biomass is pub-lagged 2m; harvest is pub-lagged 1m.")
        print("Expected coefficient sign: POSITIVE (more fish → more slaughter).\n")
        tA = test_a(panel)
        print(tA.to_string(index=False))

        max_t_a    = float(tA["t_stat"].abs().max(skipna=True))
        n_pos_sig  = ((tA["t_stat"].abs() > 1.65) & (tA["coef"] > 0)).sum()
        gate_pass  = n_pos_sig > 0

        print(f"\n  Max |t| across horizons: {max_t_a:.2f}  "
              f"({'gate PASSES' if gate_pass else 'gate FAILS'})")
        if not gate_pass:
            print("  De-seasonalised biomass does NOT reliably predict future harvest volume.")
            print("  The supply-chain premise is not supported in this data.")
            print("  Test B may still show a price relationship, but without a confirmed")
            print("  biomass→harvest link the mechanism is questionable.")

    # ── 5. Test B ─────────────────────────────────────────────────────────────
    _hdr("TEST B — DE-SEASONALISED BIOMASS → FUTURE SALMON PRICE")
    print("Regression: salmon_price_yoy_{t+h} ~ biomass_yoy_t")
    print("biomass is pub-lagged 2m; price is pub-lagged 0m.")
    print("Expected sign: NEGATIVE (more biomass → more supply → lower future price).")
    print("Report ALL horizons. Flag POSITIVE signs as wrong direction.\n")

    tB = test_b(panel)
    if tB is None:
        _skip("Missing biomass or price data.")
    else:
        print(tB.to_string(index=False))

        neg_sig = tB[(tB["t_stat"].abs() > 1.65) & (tB["coef"] < 0)]
        pos_sig = tB[(tB["t_stat"].abs() > 1.65) & (tB["coef"] > 0)]
        max_t_b = float(tB["t_stat"].abs().max(skipna=True))

        print(f"\n  Max |t| across horizons: {max_t_b:.2f}")
        if neg_sig.empty and pos_sig.empty:
            print("  No horizon clears |t| > 1.65.")
            print("  No significant biomass→price relationship in this sample.")
        if not neg_sig.empty:
            print(f"  {len(neg_sig)} horizon(s) show significant NEGATIVE effect (as expected): "
                  f"{neg_sig['horizon_m'].tolist()}")
        if not pos_sig.empty:
            print(f"  ⚠  {len(pos_sig)} horizon(s) show POSITIVE sign (WRONG direction): "
                  f"{pos_sig['horizon_m'].tolist()}")

    # ── 6. Test C ─────────────────────────────────────────────────────────────
    _hdr("TEST C — BIOMASS BEYOND FORWARD CURVE (ALPHA TEST)")
    _skip("Fish Pool forward data requires a paid subscription.")
    print()
    _note("This is the HONEST bar for tradeable edge. Without it, Test B only shows")
    _note("that the mechanism exists, not that it offers edge vs. professional")
    _note("supply forecasters (Kontali, farm operators, Rabobank aquaculture team)")
    _note("who observe their own biomass and trade the Fish Pool forward curve.")
    _note("Biomass predicting spot may simply reflect what's already in the curve.")

    # ── 7. Test D (conditional) ───────────────────────────────────────────────
    _hdr("TEST D — SIGNAL BACKTEST")

    has_b_signal = tB is not None and float(tB["t_stat"].abs().max(skipna=True)) > 1.65

    if not gate_pass and not has_b_signal:
        print("  Neither gate (Test A) nor Test B shows evidence at |t| > 1.65.")
        _skip("Backtest would overfit to noise — no prior support for the signal.")
    else:
        if not gate_pass:
            _note("Running backtest despite gate failure — interpret very cautiously.")
        if not has_b_signal:
            _note("Test B showed no effect — backtest is speculative.")

        stats_d = test_d(panel)
        if stats_d is None:
            _skip("Insufficient signal-on months (<5) or missing data.")
        else:
            print()
            for k, v in stats_d.items():
                if isinstance(v, float):
                    print(f"  {k:<38s}  {v:+.4f}")
                else:
                    print(f"  {k:<38s}  {v}")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    _hdr("SUMMARY")
    print(f"  Biomass:      {'✓ ' + biomass_source if biomass_source else '✗ unavailable'}")
    print(f"  Harvest:      {'✓ SSB 03020' if harvest is not None else '✗ unavailable'}")
    print(f"  Salmon price: ✓ SSB 03024")
    print(f"  Forwards:     ✗ Fish Pool — requires subscription (Test C skipped)")
    print(f"  Equities:     {sum(1 for v in eq_coverage.values() if v)} / "
          f"{len(eq_coverage)} tickers available")
    print()
    print(f"  Effective sample:  ~{n_cycles} full salmon price cycles (true n)")
    print(f"  Monthly rows:      {panel.shape[0]}  (autocorrelated — NOT independent)")
    print()

    if tB is not None:
        max_t_b = float(tB["t_stat"].abs().max(skipna=True))
        if max_t_b < 1.65:
            print("  VERDICT: No significant biomass→price relationship detected.")
        else:
            print(f"  VERDICT: Test B max |t| = {max_t_b:.2f}.")
            print("  Some horizon(s) show a relationship, but WITHOUT Test C we cannot")
            print("  claim tradeable edge vs. the professional forward market.")
    else:
        print("  VERDICT: Insufficient data to evaluate the hypothesis.")

    _rule()
    print("  Do not tune any parameter to improve these results.")
    _rule()


if __name__ == "__main__":
    main()
