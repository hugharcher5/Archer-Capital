#!/usr/bin/env python3
"""
research/acute_events/run.py  —  EXPLORATORY, not production.

Hypothesis: acute disease/mortality events (ISA, PD, algal blooms, de-licing accidents)
on Norwegian salmon farmers cause the affected company to underperform its peer basket
(negative CAR) over +1 / +5 / +10 / +21 trading days following the public announcement.

Point-in-time: event date = public confirmation date, not biological onset.
No look-ahead:  first trade at open of the NEXT session after the event is public.
Abnormal return = company cumulative return  minus  equal-weight peer basket return
                  (affected company excluded from basket to avoid self-contamination).
"""

import json
import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TICKERS     = ["MOWI.OL", "SALM.OL", "LSG.OL", "BAKKA.OL", "GSF.OL"]
WINDOWS     = [1, 5, 10, 21]          # trading days
PRICE_START = "2015-01-01"
PRICE_END   = datetime.today().strftime("%Y-%m-%d")
HTTP_TIMEOUT = 12


# ─────────────────────────────────────────────────────────────────────────────
# 1.  EVENT DATA
# ─────────────────────────────────────────────────────────────────────────────

# ── 1a. BarentsWatch ─────────────────────────────────────────────────────────

def _bw_parse_feature(item, zone_type):
    """Extract a minimal event dict from a BW disease-zone feature/object."""
    props = item.get("properties", item)
    date_str = (props.get("designationDate")
                or props.get("date")
                or props.get("startDate")
                or props.get("reportDate"))
    if not date_str:
        return None
    try:
        date = pd.to_datetime(date_str)
    except Exception:
        return None
    if date < pd.Timestamp(PRICE_START):
        return None
    return {
        "date":               date,
        "event_type":         zone_type,
        "region":             props.get("county") or props.get("municipality") or "Norway",
        "ticker":             None,   # locality → listed company mapping is manual
        "source":             "BARENTSWATCH",
        "company_confidence": "NONE",
        "notes":              f"BW: {json.dumps(props)[:100]}",
    }


def fetch_barentswatch_events():
    """
    Try BarentsWatch public fish-health API for PD / ISA zone designations.
    Returns list of event dicts (ticker=None; company mapping is manual).
    Silently returns [] on any failure — this source may require auth.
    """
    base = "https://www.barentswatch.no/bwapi"
    endpoints = [
        ("/v1/geodata/fishhealth/pdzone",  "PD"),
        ("/v2/geodata/fishhealth/pdzone",  "PD"),
        ("/v1/geodata/fishhealth/ilazone", "ISA"),
        ("/v2/geodata/fishhealth/ilazone", "ISA"),
    ]
    events = []
    for path, etype in endpoints:
        try:
            r = requests.get(base + path,
                             headers={"Accept": "application/json"},
                             timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                data = r.json()
                features = data if isinstance(data, list) else data.get("features", [])
                parsed = [e for f in features for e in [_bw_parse_feature(f, etype)] if e]
                print(f"    [BW] {path} → 200 | features={len(features)}, parsed={len(parsed)}")
                events.extend(parsed)
            else:
                print(f"    [BW] {path} → HTTP {r.status_code}")
        except Exception as exc:
            print(f"    [BW] {path} → error: {exc}")
    return events


# ── 1b. Mattilsynet ──────────────────────────────────────────────────────────

def fetch_mattilsynet_events():
    """
    Try Norwegian Food Safety Authority (Mattilsynet) open data.
    Returns list of event dicts or [] on failure.
    Note: hotell.difi.no hosts inspection-report feeds, not disease-zone feeds,
    so even a 200 is unlikely to contain ISA/PD confirmations directly.
    Included as a probe to detect if the endpoint exists for future parsing.
    """
    candidates = [
        "https://hotell.difi.no/api/json/mattilsynet/tilsyn/tilsynsrapporter?page=0",
        "https://data.mattilsynet.no/fiskehelse/sykdomsutbrudd",
    ]
    events = []
    for url in candidates:
        try:
            r = requests.get(url, timeout=HTTP_TIMEOUT)
            print(f"    [MT] {url[:72]} → HTTP {r.status_code}")
            # Would need further parsing if 200; returning [] for now
        except Exception as exc:
            print(f"    [MT] {url[:72]} → error: {exc}")
    return events


# ── 1c. Hardcoded stub ────────────────────────────────────────────────────────
#
# Populated from publicly available reporting:
#   - Mattilsynet weekly disease-zone press releases
#   - Stock-exchange notices (Oslo Børs)
#   - Trade press (Intrafish, SalmonBusiness, iLaks)
#   - Earnings calls and quarterly reports
#
# DATES: best-effort approximations. Use Mattilsynet weekly reports to pin
#        exact confirmation dates before treating this as actionable data.
#
# COMPANY CONFIDENCE levels:
#   HIGH   — named in a stock-exchange notice or earnings call
#   MEDIUM — named in multiple news sources; date may be off
#   LOW    — inferred from regional operations; date approximate
#   NONE   — no company attribution in available reporting
#
# !! ALL entries must be VERIFIED against primary sources before any use. !!

STUB_EVENTS = [

    # ── ALGAL BLOOM / MASS MORTALITY ─────────────────────────────────────────

    # May 2019: Chrysochromulina leadbeateri bloom in Nordland + Troms.
    # Mowi Q2 2019 earnings explicitly cited ~8,000 tonnes biomass lost.
    # SalMar also reported losses in the same quarter (smaller scale).
    # Event first widely reported ≈ 23 May 2019.
    {
        "date":               "2019-05-23",
        "event_type":         "MORTALITY",
        "region":             "Nordland/Troms",
        "ticker":             "MOWI.OL",
        "notes":              "Chrysochromulina leadbeateri algal bloom; Mowi ~8 000t lost — "
                              "cited explicitly in Q2-2019 earnings. Highest confidence event.",
        "source":             "STUB",
        "company_confidence": "HIGH",
    },
    {
        "date":               "2019-05-23",
        "event_type":         "MORTALITY",
        "region":             "Nordland/Troms",
        "ticker":             "SALM.OL",
        "notes":              "Same Chrysochromulina bloom; SalMar Q2-2019 cited losses "
                              "(smaller exposure than Mowi). Date is first press report.",
        "source":             "STUB",
        "company_confidence": "MEDIUM",
    },

    # Feb 2021: algal ingress in Rogaland (dinoflagellates) — Grieg primary area
    {
        "date":               "2021-02-10",
        "event_type":         "MORTALITY",
        "region":             "Rogaland",
        "ticker":             "GSF.OL",
        "notes":              "Algal ingress Rogaland; Grieg Seafood primary operator. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },

    # ── ISA (Infectious Salmon Anaemia) ──────────────────────────────────────

    {
        "date":               "2015-07-14",
        "event_type":         "ISA",
        "region":             "Troms",
        "ticker":             "MOWI.OL",
        "notes":              "ISA confirmed Troms; Marine Harvest (Mowi) widely operated here. "
                              "Date approximate — VERIFY via Mattilsynet weekly report.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2016-04-05",
        "event_type":         "ISA",
        "region":             "Nordland",
        "ticker":             None,
        "notes":              "ISA confirmed Nordland; no company attribution in available reporting.",
        "source":             "STUB",
        "company_confidence": "NONE",
    },
    {
        "date":               "2017-09-25",
        "event_type":         "ISA",
        "region":             "Nordland",
        "ticker":             "SALM.OL",
        "notes":              "ISA at site in Nordland; SalMar operated multiple sites here. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2018-03-12",
        "event_type":         "ISA",
        "region":             "Troms",
        "ticker":             "MOWI.OL",
        "notes":              "ISA confirmed Troms; Mowi (Marine Harvest) primary operator. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2018-11-06",
        "event_type":         "ISA",
        "region":             "Hordaland",
        "ticker":             "LSG.OL",
        "notes":              "ISA at Lerøy Seafood Hordaland site. Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2020-03-03",
        "event_type":         "ISA",
        "region":             "Nordland",
        "ticker":             "MOWI.OL",
        "notes":              "ISA Nordland; Mowi has largest biomass in this county. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2021-08-17",
        "event_type":         "ISA",
        "region":             "Trøndelag",
        "ticker":             "SALM.OL",
        "notes":              "ISA in Trøndelag; SalMar primary operator here. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2022-05-24",
        "event_type":         "ISA",
        "region":             "Finnmark",
        "ticker":             "GSF.OL",
        "notes":              "ISA in Finnmark; Grieg Seafood operates there. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2023-02-28",
        "event_type":         "ISA",
        "region":             "Troms",
        "ticker":             "MOWI.OL",
        "notes":              "ISA in Troms; Mowi primary operator. Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2024-06-10",
        "event_type":         "ISA",
        "region":             "Nordland",
        "ticker":             None,
        "notes":              "ISA Nordland; no specific company attribution from available reports.",
        "source":             "STUB",
        "company_confidence": "NONE",
    },

    # ── PD (Pancreas Disease / Salmon Alpha Virus) ────────────────────────────
    # PD is endemic in the statutory PD zone (Rogaland → parts of Møre og Romsdal).
    # Confirmations are frequent; only larger / newsworthy outbreaks listed here.

    {
        "date":               "2016-09-19",
        "event_type":         "PD",
        "region":             "Rogaland",
        "ticker":             "GSF.OL",
        "notes":              "PD confirmed Rogaland; Grieg Seafood primary operator in county. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2017-05-08",
        "event_type":         "PD",
        "region":             "Hordaland",
        "ticker":             "LSG.OL",
        "notes":              "PD in Hordaland; Lerøy major operator. Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2018-07-30",
        "event_type":         "PD",
        "region":             "Rogaland",
        "ticker":             "GSF.OL",
        "notes":              "PD in Rogaland. Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2019-11-11",
        "event_type":         "PD",
        "region":             "Møre og Romsdal",
        "ticker":             "MOWI.OL",
        "notes":              "PD in Møre; Mowi significant operations here. "
                              "Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
    {
        "date":               "2021-04-06",
        "event_type":         "PD",
        "region":             "Rogaland",
        "ticker":             "GSF.OL",
        "notes":              "PD outbreak in Rogaland. Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },

    # ── DE-LICING / TREATMENT ACCIDENTS ──────────────────────────────────────

    {
        "date":               "2017-12-11",
        "event_type":         "MORTALITY",
        "region":             "Nordland",
        "ticker":             "MOWI.OL",
        "notes":              "De-licing treatment accident (thermolicer); fish mortality "
                              "reported. Mowi primary operator Nordland. Date approximate — VERIFY.",
        "source":             "STUB",
        "company_confidence": "LOW",
    },
]


def load_events():
    """
    Try programmatic sources; merge with stub.
    Returns (all_df, mappable_df).
    """
    print("\n── EVENT DATA ─────────────────────────────────────────────────────")

    print("\n  [1/2] BarentsWatch API ...")
    bw = fetch_barentswatch_events()

    print("\n  [2/2] Mattilsynet open data ...")
    mt = fetch_mattilsynet_events()

    n_prog = len(bw) + len(mt)
    print(f"\n  Programmatic events found: {n_prog}  (BW={len(bw)}, MT={len(mt)})")

    # Build combined frame
    stub_df = pd.DataFrame(STUB_EVENTS)
    stub_df["date"] = pd.to_datetime(stub_df["date"])

    parts = [stub_df]
    if bw:
        bw_df = pd.DataFrame(bw)
        bw_df["date"] = pd.to_datetime(bw_df["date"])
        parts.insert(0, bw_df)
    if mt:
        mt_df = pd.DataFrame(mt)
        mt_df["date"] = pd.to_datetime(mt_df["date"])
        parts.insert(0, mt_df)

    all_df = (pd.concat(parts, ignore_index=True)
                .sort_values("date")
                .reset_index(drop=True))

    mappable = all_df[all_df["ticker"].notna()].copy()
    unmapped  = all_df[all_df["ticker"].isna()]

    print(f"\n  Total events              : {len(all_df)}")
    print(f"  ├─ with ticker (usable)   : {len(mappable)}")
    print(f"  └─ without ticker (unusable): {len(unmapped)}  ← no company attribution")

    print("\n  Source mix (all events):")
    for src, cnt in all_df["source"].value_counts().items():
        tag = "" if src == "STUB" else "  ← programmatic"
        print(f"    {src}: {cnt}{tag}")

    return all_df, mappable


# ─────────────────────────────────────────────────────────────────────────────
# 2.  EQUITY PRICES
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices():
    """Download daily adjusted-close prices for all tickers."""
    print("\n── PRICE DATA ─────────────────────────────────────────────────────")
    print(f"  Tickers : {TICKERS}")
    print(f"  Period  : {PRICE_START} → {PRICE_END}")

    raw = yf.download(
        TICKERS,
        start=PRICE_START,
        end=PRICE_END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance 0.2+ returns MultiIndex (field, ticker) columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    elif "Close" in raw.columns:
        prices = raw[["Close"]].copy()
    else:
        prices = raw.copy()

    prices = prices.ffill().dropna(how="all")

    # Ensure we have all tickers as columns
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(-1)

    available = [t for t in TICKERS if t in prices.columns]
    missing   = [t for t in TICKERS if t not in prices.columns]

    print(f"  Rows    : {len(prices)}")
    print(f"  Got     : {available}")
    if missing:
        print(f"  Missing : {missing}")

    return prices[available]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CAR COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_returns(prices):
    """Simple daily percentage returns; drop the first all-NaN row."""
    return prices.pct_change().iloc[1:]


def first_trading_day_after(idx, date):
    """First date in idx strictly after `date`."""
    ts = pd.Timestamp(date)
    later = idx[idx > ts]
    return later[0] if len(later) > 0 else None


def compute_event_car(returns, event_date, ticker, windows):
    """
    Cumulative Abnormal Return for one event.

    Basket = equal-weight mean of ALL OTHER tickers (affected co. excluded).
    Entry  = first trading day strictly after event_date (no same-day look-ahead).
    CAR_Wd = cumulative company return - cumulative basket return over W days.

    Returns dict {f"CAR_{w}d": float} or None if uncomputable.
    """
    if ticker not in returns.columns:
        return None

    t0 = first_trading_day_after(returns.index, event_date)
    if t0 is None:
        return None

    peers = [t for t in returns.columns if t != ticker]
    if not peers:
        return None

    future = returns.loc[t0:]

    result = {"t0": t0}
    for w in windows:
        chunk = future.iloc[:w]
        if len(chunk) < w:
            result[f"CAR_{w}d"] = np.nan
            continue

        # compound cumulative returns
        co_cum = float((1 + chunk[ticker].fillna(0)).prod() - 1)

        # basket: day-by-day equal-weight average of peers, then compound
        peer_daily = chunk[peers].mean(axis=1, skipna=True)
        bk_cum     = float((1 + peer_daily.fillna(0)).prod() - 1)

        result[f"CAR_{w}d"] = co_cum - bk_cum

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  STATISTICAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def summarise(vals, label, indent="  "):
    """Print mean / median / %neg / t-stat for a list of CAR values."""
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)], dtype=float)
    n   = len(arr)
    if n < 2:
        print(f"{indent}{label}: n={n}  — too few for t-test")
        return
    mean    = np.mean(arr)
    median  = np.median(arr)
    pct_neg = np.mean(arr < 0) * 100
    t, p    = stats.ttest_1samp(arr, 0)
    star    = " *" if p < 0.10 else ""
    print(f"{indent}{label}: n={n:2d}  mean={mean:+.2%}  median={median:+.2%}"
          f"  %neg={pct_neg:.0f}%  t={t:+.2f}  p={p:.3f}{star}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("ACUTE EVENT STUDY — NORWEGIAN SALMON FARMERS — EXPLORATORY")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 72)

    # ── Events ───────────────────────────────────────────────────────────────
    all_df, mappable = load_events()

    # ── Prices ───────────────────────────────────────────────────────────────
    prices  = fetch_prices()
    returns = compute_returns(prices)

    # ── Compute CARs ─────────────────────────────────────────────────────────
    print("\n── COMPUTING CARs ─────────────────────────────────────────────────")

    records  = []
    skipped  = []

    for _, row in mappable.iterrows():
        ticker     = row["ticker"]
        event_date = row["date"]

        if ticker not in returns.columns:
            skipped.append(f"  {str(event_date.date())}  {ticker}  → ticker not in price data")
            continue

        car = compute_event_car(returns, event_date, ticker, WINDOWS)
        if car is None:
            skipped.append(f"  {str(event_date.date())}  {ticker}  → no future price data after event")
            continue

        rec = {
            "date":               event_date,
            "ticker":             ticker,
            "event_type":         row.get("event_type", "?"),
            "region":             row.get("region", ""),
            "source":             row.get("source", "STUB"),
            "company_confidence": row.get("company_confidence", "?"),
            "entry_date":         car["t0"],
        }
        for w in WINDOWS:
            rec[f"CAR_{w}d"] = car.get(f"CAR_{w}d", np.nan)
        records.append(rec)

    if skipped:
        print("  Skipped events:")
        for s in skipped:
            print(s)

    if not records:
        print("\n  !! No usable events after filtering. Cannot compute CARs.")
        return

    df = (pd.DataFrame(records)
            .sort_values("date")
            .reset_index(drop=True))
    n  = len(df)

    # ── Statistical power assessment ──────────────────────────────────────────
    print("\n── STATISTICAL POWER ──────────────────────────────────────────────")
    if n < 10:
        power_level = "CRITICAL"
        power_msg   = (f"Only {n} usable events. Results are purely anecdotal."
                       f" This sample cannot support any statistical inference.")
    elif n < 15:
        power_level = "MINIMAL"
        power_msg   = (f"{n} usable events. Statistical power is MINIMAL."
                       f" This is a curiosity, not a testable edge."
                       f" t-statistics are unreliable; treat results as directional only.")
    elif n < 30:
        power_level = "MARGINAL"
        power_msg   = (f"{n} events — marginal power."
                       f" t-stats are indicative but easily dominated by outliers.")
    else:
        power_level = "REASONABLE"
        power_msg   = f"{n} events — reasonable for a preliminary hypothesis screen."

    print(f"  [{power_level}]  {power_msg}")

    # ── Per-event table ───────────────────────────────────────────────────────
    print("\n── PER-EVENT TABLE ────────────────────────────────────────────────")

    def fmt_pct(x):
        try:
            return f"{x:+.2%}" if not np.isnan(x) else "   NaN"
        except Exception:
            return str(x)

    display_cols = (["date", "ticker", "event_type", "region",
                     "company_confidence"]
                    + [f"CAR_{w}d" for w in WINDOWS])
    disp = df[display_cols].copy()
    for w in WINDOWS:
        disp[f"CAR_{w}d"] = disp[f"CAR_{w}d"].apply(fmt_pct)

    with pd.option_context("display.max_columns", 20,
                           "display.width", 160,
                           "display.max_colwidth", 18):
        print(disp.to_string(index=False))

    # ── Test A: aggregate CARs ────────────────────────────────────────────────
    print("\n── TEST A: AGGREGATE CARs ACROSS ALL EVENTS ───────────────────────")
    print("  H0: mean CAR = 0   H1: mean CAR < 0  (affected farmer underperforms)\n")
    for w in WINDOWS:
        summarise(df[f"CAR_{w}d"].tolist(), f"+{w:>2}d CAR")

    # ── Test B: by event type ─────────────────────────────────────────────────
    print("\n── TEST B: CAR BY EVENT TYPE ──────────────────────────────────────")
    for etype in sorted(df["event_type"].unique()):
        sub = df[df["event_type"] == etype]
        print(f"\n  [{etype}]")
        for w in WINDOWS:
            summarise(sub[f"CAR_{w}d"].tolist(), f"+{w:>2}d", indent="    ")

    # ── Source / confidence breakdown ─────────────────────────────────────────
    print("\n── SOURCE BREAKDOWN (all events, including unmapped) ───────────────")
    for src, cnt in all_df["source"].value_counts().items():
        pct = cnt / len(all_df) * 100
        tag = "" if src == "STUB" else "  ← programmatic"
        print(f"  {src:20s}: {cnt:3d}  ({pct:.0f}%){tag}")

    print("\n── COMPANY ATTRIBUTION CONFIDENCE (usable events only) ─────────────")
    for conf, cnt in df["company_confidence"].value_counts().items():
        print(f"  {conf:8s}: {cnt:3d}")

    # ── Honest conclusion ─────────────────────────────────────────────────────
    print("\n── CONCLUSION ─────────────────────────────────────────────────────")
    print(f"  Total events collected   : {len(all_df)}")
    print(f"  Events with ticker mapped: {len(mappable)}")
    print(f"  Usable events (with price data + future horizon): {n}")
    print(f"\n  [{power_level}]  {power_msg}")

    if n >= 2:
        arr5 = df["CAR_5d"].dropna().values
        if len(arr5) >= 2:
            mean5 = np.mean(arr5)
            t5, p5 = stats.ttest_1samp(arr5, 0)
            direction = "NEGATIVE → supports hypothesis" if mean5 < 0 else "POSITIVE → contradicts hypothesis"
            sig_str   = "p < 0.10 — marginally significant" if p5 < 0.10 else "not significant (p >= 0.10)"
            print(f"\n  Key metric — mean +5d CAR = {mean5:+.2%}  ({direction})")
            print(f"  t = {t5:+.2f},  p = {p5:.3f}  — {sig_str}")

    print("""
  CRITICAL CAVEAT: The stub table is populated from memory + public reporting,
  NOT from a systematic scrape of Mattilsynet disease-confirmation notices.
  Most dates are APPROXIMATE (± weeks); company attributions are INFERRED
  from regional operations, not from primary regulatory documents.

  NEXT STEPS to make this actionable:
  1. Pull exact confirmation dates from Mattilsynet weekly disease reports
     (published as PDFs at mattilsynet.no — or request a data extract).
  2. Map each confirmed locality to a specific operator via BarentsWatch
     locality registers (requires BW API credentials).
  3. Increase sample to ≥30 with HIGH/MEDIUM confidence to support inference.
  4. Consider whether ISA/PD confirmations were preceded by trading halts
     (Oslo Børs) that could distort the next-open entry assumption.
""")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "event_cars.csv")
    df.to_csv(out_path, index=False)
    print(f"  Results saved → {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
