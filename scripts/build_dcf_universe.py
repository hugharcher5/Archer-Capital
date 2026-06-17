#!/usr/bin/env python3
"""
scripts/build_dcf_universe.py
==============================
Build a current-snapshot DCF stock universe and write it to
config/dcf_universe.csv.

SURVIVORSHIP BIAS NOTE: This is a point-in-time snapshot of TODAY's survivors.
Any historical backtest on this static list will overstate returns because
delisted / acquired / failed names are absent. Point-in-time universe
reconstruction (e.g. from CRSP or a historical master file) is a separate
step required before any historical backtest.

DATA SOURCE
-----------
Primary:   FMP /stable/company-screener  (paid plan required)
           FMP /api/v3/stock-screener     (legacy, deprecated Aug 2025)
Fallback:  yfinance screen()             (Yahoo Finance, free)

The script probes FMP first, reports the result, and auto-falls-back
to yfinance if FMP is unavailable. No manual intervention needed.

Usage:
    venv/bin/python scripts/build_dcf_universe.py

FMP key read from .env (same as DCF engine):  FMP_API_KEY=...
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from yfinance import EquityQuery
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
MARKET_CAP_MIN   = 100_000_000     # $100 M  — small-cap floor
MARKET_CAP_MAX   = 2_000_000_000   # $2 B    — small-cap ceiling
DOLLAR_VOL_MIN   = 1_000_000       # $1 M/day — liquidity gate
SECTOR_EXCLUDE   = {"Financial Services", "Real Estate"}
YF_PAGE_SIZE     = 250             # rows per yfinance screen() call
SECTOR_WORKERS   = 20              # threads for sector enrichment
SECTOR_TIMEOUT   = 10              # seconds per ticker info call

# Yahoo Finance exchange codes to keep (NASDAQ tiers + NYSE + NYSE American)
KEEP_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE"}
# Human-readable names for output
EXCHANGE_LABEL = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "NYQ": "NYSE",   "ASE": "AMEX",
}

OUT_PATH = Path(__file__).parent.parent / "config" / "dcf_universe.csv"

# Truncation-detection: these exact totals signal a possible API cap
_SUSPICIOUS_COUNTS = {250, 500, 750, 1000, 1500, 2000, 2500, 3000, 5000}

# ─────────────────────────────────────────────────────────────────────────────
# FMP CONFIG  (mirrors dcf/valuation/sources.py)
# ─────────────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env")
load_dotenv()

_FMP_V3_BASE     = "https://financialmodelingprep.com/api/v3"
_FMP_STABLE_BASE = "https://financialmodelingprep.com/stable"


def _load_fmp_key() -> str:
    return os.getenv("FMP_API_KEY", "").strip()


# ─────────────────────────────────────────────────────────────────────────────
# FMP SCREENER PROBE  (documents why it fails; auto-fallback)
# ─────────────────────────────────────────────────────────────────────────────
def probe_fmp(api_key: str) -> pd.DataFrame | None:
    """
    Attempt FMP screener endpoints in order of preference.
    Returns a DataFrame on success, None if all fail.
    Prints a clear diagnosis for each failure.
    """
    print("\n[FMP] Probing screener endpoints …")

    # ── 1. Stable endpoint (current API, paid plan) ───────────────────────
    url = f"{_FMP_STABLE_BASE}/company-screener"
    params = {
        "marketCapMoreThan":  MARKET_CAP_MIN,
        "marketCapLowerThan": MARKET_CAP_MAX,
        "exchange":           "NASDAQ,NYSE,AMEX",
        "country":            "US",
        "isEtf":              "false",
        "isFund":             "false",
        "isActivelyTrading":  "true",
        "limit":              3000,
        "apikey":             api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                print(f"  FMP stable/company-screener: OK ({len(data)} rows)")
                return _fmp_to_df(data)
        elif r.status_code == 402:
            print(
                "  FMP stable/company-screener: 402 Payment Required.\n"
                "     The company-screener endpoint requires a paid FMP subscription.\n"
                "     Upgrade at https://financialmodelingprep.com/developer/docs/pricing"
            )
        elif r.status_code == 403:
            print(f"  FMP stable/company-screener: 403 Forbidden — {r.json().get('Error Message','')[:80]}")
        else:
            print(f"  FMP stable/company-screener: {r.status_code}")
    except Exception as e:
        print(f"  FMP stable/company-screener: error — {e}")

    # ── 2. Legacy v3 endpoint (deprecated Aug 2025) ───────────────────────
    url = f"{_FMP_V3_BASE}/stock-screener"
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and data:
                print(f"  FMP v3/stock-screener: OK ({len(data)} rows)")
                return _fmp_to_df(data)
        elif r.status_code == 403:
            msg = ""
            try:
                msg = r.json().get("Error Message", "")[:120]
            except Exception:
                pass
            print(
                f"  FMP v3/stock-screener: 403 Legacy endpoint blocked.\n"
                f"     FMP deprecated /api/v3/ endpoints after August 2025.\n"
                f"     Only accounts subscribed before that date can use legacy endpoints.\n"
                f"     Msg: {msg}"
            )
        else:
            print(f"  FMP v3/stock-screener: {r.status_code}")
    except Exception as e:
        print(f"  FMP v3/stock-screener: error — {e}")

    print(
        "\n  FMP screener unavailable with this API key.\n"
        "  To fix: upgrade to FMP paid plan at https://financialmodelingprep.com/developer/docs/pricing\n"
        "  Auto-falling back to yfinance screener (Yahoo Finance, free) …"
    )
    return None


def _fmp_to_df(data: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(data)
    # Normalise to common schema
    df = df.rename(columns={
        "companyName": "companyName",
        "marketCap":   "marketCap",
        "price":       "price",
        "volume":      "volume",
    })
    df["avgDollarVolume"] = (
        pd.to_numeric(df.get("price",   pd.Series()), errors="coerce") *
        pd.to_numeric(df.get("volume",  pd.Series()), errors="coerce")
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# YFINANCE SCREENER  (fallback, free)
# ─────────────────────────────────────────────────────────────────────────────
def yf_screen() -> pd.DataFrame:
    """
    Pull US small-cap equities via yfinance screen().
    Paginates through all results (250 per call).
    Returns a raw DataFrame with market cap, price, volume, exchange fields.
    Sector/industry are NOT in the screener response — enriched separately.
    """
    print("\n[yfinance screener] Querying Yahoo Finance …")
    print(f"  marketCap: ${MARKET_CAP_MIN/1e6:.0f}M – ${MARKET_CAP_MAX/1e6:.0f}M  |  region: US")

    q = EquityQuery("and", [
        EquityQuery("gt", ["intradaymarketcap", MARKET_CAP_MIN]),
        EquityQuery("lt", ["intradaymarketcap", MARKET_CAP_MAX]),
        EquityQuery("eq", ["region", "us"]),
    ])

    all_quotes: list[dict] = []
    offset = 0
    total_reported = None

    while True:
        result = yf.screen(q, size=YF_PAGE_SIZE, offset=offset)
        quotes = result.get("quotes", [])
        if total_reported is None:
            total_reported = result.get("total", 0)
            print(f"  Yahoo Finance reports total = {total_reported}")

        all_quotes.extend(quotes)
        print(f"  Fetched {len(all_quotes):>5} / {total_reported}  (offset={offset})")

        if len(quotes) < YF_PAGE_SIZE:
            break   # last page
        offset += YF_PAGE_SIZE
        time.sleep(0.3)

    n_raw = len(all_quotes)
    print(f"\n  RAW count (before filters): {n_raw}")

    # Truncation check
    if n_raw in _SUSPICIOUS_COUNTS or (total_reported and n_raw < total_reported * 0.9):
        print(
            f"\n  ⚠️  WARNING: Retrieved {n_raw} but total was {total_reported}.\n"
            f"     The result set may be truncated by Yahoo Finance.\n"
            f"     Proceeding with available names — universe may be incomplete."
        )
    else:
        print(f"  ✓ Retrieved {n_raw} of {total_reported} reported — looks complete.")

    df = pd.json_normalize(all_quotes)

    # Rename to canonical column names
    rename = {
        "symbol":                 "symbol",
        "shortName":              "companyName",
        "longName":               "longName",
        "marketCap":              "marketCap",
        "exchange":               "exchange",
        "regularMarketPrice":     "price",
        "regularMarketVolume":    "volume",
        "averageDailyVolume3Month": "avgDailyVol3m",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Dollar volume: use 3-month average (more stable than spot volume)
    df["price_num"]        = pd.to_numeric(df.get("price",        pd.NA), errors="coerce")
    df["avgDailyVol3m_num"] = pd.to_numeric(df.get("avgDailyVol3m", pd.NA), errors="coerce")
    df["avgDollarVolume"]  = df["price_num"] * df["avgDailyVol3m_num"]

    # Volume (spot) for output column
    df["volume"] = pd.to_numeric(df.get("volume", pd.NA), errors="coerce")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR ENRICHMENT  (threaded yfinance, for final filtered set)
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_sector_info(symbol: str) -> dict:
    """Fetch sector + industry + long company name for one ticker."""
    try:
        info = yf.Ticker(symbol).info
        return {
            "symbol":       symbol,
            "sector":       info.get("sector",   ""),
            "industry":     info.get("industry", ""),
            "companyName":  info.get("longName",  info.get("shortName", "")),
        }
    except Exception:
        return {"symbol": symbol, "sector": "", "industry": "", "companyName": ""}


def enrich_sector(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch sector and industry for each ticker using ThreadPoolExecutor.
    Replaces empty sector/industry columns in df.
    """
    symbols = df["symbol"].dropna().unique().tolist()
    n = len(symbols)
    print(f"\n[sector] Fetching sector/industry for {n} tickers "
          f"({SECTOR_WORKERS} threads) …")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=SECTOR_WORKERS) as ex:
        futures = {ex.submit(_fetch_sector_info, sym): sym for sym in symbols}
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 100 == 0 or done == n:
                print(f"  {done}/{n} done", end="\r", flush=True)

    print()
    sector_df = pd.DataFrame(results)
    df = df.merge(sector_df[["symbol", "sector", "industry"]], on="symbol", how="left")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# POST-FILTERS
# ─────────────────────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, source: str) -> pd.DataFrame:
    n0 = len(df)
    print(f"\n[filters]  (source: {source})")
    print(f"  Raw from screener:         {n0:>5}")

    # ── Drop rows missing critical fields ─────────────────────────────────
    df = df.dropna(subset=["symbol", "marketCap", "price_num"])
    n1 = len(df)
    print(f"  After null drop:           {n1:>5}  (-{n0-n1})")

    # ── Exchange filter (NASDAQ / NYSE / AMEX only) ───────────────────────
    if "exchange" in df.columns:
        df = df[df["exchange"].isin(KEEP_EXCHANGES)]
    n2 = len(df)
    print(f"  After exchange filter:     {n2:>5}  (-{n1-n2})")

    # ── Dollar-volume liquidity gate ──────────────────────────────────────
    df = df[df["avgDollarVolume"] >= DOLLAR_VOL_MIN]
    n3 = len(df)
    print(f"  After dollar-vol >${DOLLAR_VOL_MIN/1e6:.0f}M/day:  {n3:>5}  (-{n2-n3})")

    # ── Sector enrichment (now, on reduced set) ───────────────────────────
    if "sector" not in df.columns or df["sector"].isna().all():
        df = enrich_sector(df)

    # ── Drop financial / real-estate sectors ─────────────────────────────
    if "sector" in df.columns:
        mask_fin = df["sector"].isin(SECTOR_EXCLUDE)
        df = df[~mask_fin]
    n4 = len(df)
    print(f"  After excl. {SECTOR_EXCLUDE}:")
    print(f"                             {n4:>5}  (-{n3-n4})")
    print(f"  ──────────────────────────────────")
    print(f"  FINAL:                     {n4:>5}")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
def write_output(df: pd.DataFrame) -> None:
    out = pd.DataFrame({
        "symbol":         df["symbol"],
        "companyName":    df.get("companyName", ""),
        "marketCap":      pd.to_numeric(df.get("marketCap"), errors="coerce"),
        "sector":         df.get("sector",   ""),
        "industry":       df.get("industry", ""),
        "exchange":       df["exchange"].map(EXCHANGE_LABEL).fillna(df["exchange"]),
        "price":          df["price_num"],
        "volume":         pd.to_numeric(df.get("volume"), errors="coerce"),
        "avgDollarVolume": df["avgDollarVolume"],
    })
    out = out.sort_values("marketCap", ascending=False).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\n[output] Wrote {len(out)} tickers → {OUT_PATH}")

    # Sector breakdown
    print("\n[sector breakdown of final universe]")
    breakdown = (
        out.groupby("sector")
        .agg(
            count        = ("symbol", "count"),
            median_cap_M = ("marketCap", lambda x: x.median() / 1e6),
        )
        .sort_values("count", ascending=False)
    )
    for sector, row in breakdown.iterrows():
        bar = "█" * int(row["count"] / max(breakdown["count"]) * 20)
        print(f"  {sector:<35}  {int(row['count']):>4}  "
              f"median_cap=${row['median_cap_M']:.0f}M  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    print("=" * 65)
    print("BUILD DCF UNIVERSE")
    print(f"  Small-cap:  ${MARKET_CAP_MIN/1e6:.0f}M – ${MARKET_CAP_MAX/1e6:.0f}M")
    print(f"  Liquidity:  >${DOLLAR_VOL_MIN/1e6:.0f}M avg daily dollar volume")
    print(f"  Exchanges:  NASDAQ / NYSE / AMEX (US only)")
    print(f"  Excl.:      {SECTOR_EXCLUDE}")
    print(f"  Output:     {OUT_PATH}")
    print("=" * 65)

    api_key = _load_fmp_key()
    if api_key:
        raw_df = probe_fmp(api_key)
        source = "FMP"
    else:
        print("\n[FMP] FMP_API_KEY not set — skipping FMP, using yfinance.")
        raw_df = None
        source = "yfinance"

    if raw_df is None:
        raw_df = yf_screen()
        source = "yfinance"

    filt_df = apply_filters(raw_df, source)
    write_output(filt_df)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
