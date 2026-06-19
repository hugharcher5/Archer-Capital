"""
research/quality_filter/run.py

Hypothesis 2: Does codified quality (Piotroski F-Score + Management score) add return
ON TOP of a cheap-value basket — and is the contribution independent, or already baked
into the value signal?

REUSES from H1 (research/dcf_vs_multiple):
- Point-in-time EDGAR layer (same cache, same helper functions)
- Cached signal panel (signals_panel.parquet): dcf_vp, ebitda_ev, size, mom, fwd_ret
- Same 10 annual rebalance dates (June 30, 2015-2024)
- Same shared universe (DCF-able, non-financial — Financial/RE already excluded)

NEW SIGNALS (both point-in-time from EDGAR; no recomputation of DCF):
  Piotroski F-Score (0-9): 9 binary tests from two consecutive PIT 10-K filings
  Management Score (0-3):  M1 insider buying (Form 4), M2 dilution, M3 ROIC

BASKETS (two arms × 4 filter levels = 8 baskets; benchmark = EW whole universe):
  DCF arm:       A=DCF-cheap  B=+F≥7  C=+Mgmt≥2  D=+Mgmt=3
  EBITDA/EV arm: same filter levels applied to EBITDA/EV-cheap

NOTE: T=10 annual periods. Read DIRECTION and CONSISTENCY across tests.
No single t-stat is conclusive. Trials logged for Deflated Sharpe.
Survivorship bias: absolute returns are upper bounds; differentials are robust.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import socket
import numpy as np
import pandas as pd
import urllib.request
import urllib.error
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "dcf"))

H1_CACHE  = REPO / "research" / "dcf_vs_multiple" / "_cache"
H1_PANEL  = H1_CACHE / "signals_panel.parquet"
EDGAR_DIR = H1_CACHE / "edgar_facts"
CIK_CACHE = H1_CACHE / "edgar_tickers.json"

CACHE_DIR = Path(__file__).parent / "_cache"
for _d in [CACHE_DIR, CACHE_DIR / "submissions", CACHE_DIR / "insider"]:
    _d.mkdir(parents=True, exist_ok=True)

QUALITY_CACHE = CACHE_DIR / "quality_scores.parquet"

# ── Constants ─────────────────────────────────────────────────────────────────
N_QUINTILES   = 5
COSTS_BPS     = 20
INSIDER_WINDOW = 180   # days before rebalance date for insider transactions
F_THRESH_B    = 7      # F-Score gate for basket B
M_THRESH_C    = 2      # Mgmt gate for basket C
M_THRESH_D    = 3      # Mgmt gate for basket D

START_YEAR = 2015
END_YEAR   = 2025
REB_MONTH  = 6
REB_DAY    = 30

_EDGAR_HDR = {"User-Agent": "Archer Capital research@archercapital.dev"}


# ── Price helpers ──────────────────────────────────────────────────────────────

def _price_at(hist: pd.DataFrame | None, as_of: date) -> float | None:
    if hist is None or hist.empty:
        return None
    ts  = pd.Timestamp(as_of)
    sub = hist[hist.index <= ts]
    if sub.empty:
        return None
    return float(sub["Close"].iloc[-1])


def add_fwd_ret(panel: pd.DataFrame) -> pd.DataFrame:
    """Download price history for all panel tickers and compute 1-year forward returns."""
    tickers = list(panel["ticker"].unique())
    price_start = f"{START_YEAR - 1}-01-01"
    price_end   = f"{END_YEAR + 1}-01-01"
    print(f"  Downloading price history for {len(tickers)} tickers ({price_start} → {price_end})...")

    CHUNK = 100
    all_close: dict[str, pd.Series] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            raw = yf.download(chunk, start=price_start, end=price_end,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]] if "Close" in raw.columns else raw
            for sym in close.columns:
                s = close[sym].dropna()
                if not s.empty:
                    all_close[sym] = s
        except Exception:
            pass
    price_hists = {sym: s.to_frame("Close") for sym, s in all_close.items()}
    print(f"  Price histories loaded: {len(price_hists)}/{len(tickers)}")

    reb_dates = [date(y, REB_MONTH, REB_DAY) for y in range(START_YEAR, END_YEAR)]
    fwd_map: dict[pd.Timestamp, date] = {}
    for i, d in enumerate(reb_dates[:-1]):
        fwd_map[pd.Timestamp(d)] = date(reb_dates[i + 1].year, reb_dates[i + 1].month, reb_dates[i + 1].day)
    last = reb_dates[-1]
    fwd_map[pd.Timestamp(last)] = date(last.year + 1, last.month, last.day)

    cost = COSTS_BPS / 10000

    def _fwd(row):
        fwd_date = fwd_map.get(row["as_of"])
        if fwd_date is None:
            return float("nan")
        ph = price_hists.get(row["ticker"])
        if ph is None:
            return float("nan")
        as_of_d = row["as_of"].date() if hasattr(row["as_of"], "date") else row["as_of"]
        p0 = _price_at(ph, as_of_d)
        p1 = _price_at(ph, fwd_date)
        if p0 is None or p1 is None or p0 <= 0:
            return float("nan")
        return (p1 - p0) / p0 - cost

    panel = panel.copy()
    panel["fwd_ret"] = panel.apply(_fwd, axis=1)
    return panel

# ═══════════════════════════════════════════════════════════════════════════════
# EDGAR helpers (reuse same PIT layer as H1)
# ═══════════════════════════════════════════════════════════════════════════════

def _edgar_get(url: str) -> dict | list:
    req = urllib.request.Request(url, headers=_EDGAR_HDR)
    with urllib.request.urlopen(req, timeout=20) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode())


_CIK_MAP: dict[str, str] | None = None

def _cik(ticker: str) -> str | None:
    global _CIK_MAP
    if _CIK_MAP is None:
        _CIK_MAP = json.loads(CIK_CACHE.read_text())
    return _CIK_MAP.get(ticker.upper())


def _load_facts(ticker: str) -> dict | None:
    """Load cached EDGAR facts JSON for ticker (H1 already downloaded them)."""
    p = EDGAR_DIR / f"{ticker.upper()}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _pit_series(facts: dict, tags: list[str], as_of: date, abs_val: bool = False) -> pd.Series:
    """PIT annual 10-K series: for each fiscal-year-end, value from most recent filing ≤ as_of."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node  = us_gaap.get(tag, {})
        units = node.get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        best: dict[pd.Timestamp, tuple[date, float]] = {}
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            fs = row.get("filed")
            if not fs:
                continue
            try:
                filed = date.fromisoformat(fs)
            except ValueError:
                continue
            if filed > as_of:
                continue
            end_str, val = row.get("end"), row.get("val")
            if end_str is None or val is None:
                continue
            try:
                v  = abs(float(val)) if abs_val else float(val)
                ts = pd.Timestamp(end_str)
                if ts not in best or filed > best[ts][0]:
                    best[ts] = (filed, v)
            except (ValueError, TypeError):
                pass
        if best:
            s = pd.Series({ts: v for ts, (_, v) in best.items()}).sort_index()
            return s
    return pd.Series(dtype=float)


def _pit_scalar(facts: dict, tags: list[str], as_of: date, abs_val: bool = False) -> float:
    """Most recent 10-K balance-sheet scalar with filed ≤ as_of."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node  = us_gaap.get(tag, {})
        units = node.get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        best_filed: date | None = None
        best_val: float = float("nan")
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            fs = row.get("filed")
            if not fs:
                continue
            try:
                filed = date.fromisoformat(fs)
            except ValueError:
                continue
            if filed > as_of:
                continue
            val = row.get("val")
            if val is None:
                continue
            try:
                v = abs(float(val)) if abs_val else float(val)
                if best_filed is None or filed > best_filed:
                    best_filed, best_val = filed, v
            except (ValueError, TypeError):
                pass
        if not math.isnan(best_val):
            return best_val
    return float("nan")


def _two(facts: dict, tags: list[str], as_of: date, abs_val: bool = False
         ) -> tuple[float, float]:
    """(current, prior) from last two values of PIT series."""
    s = _pit_series(facts, tags, as_of, abs_val)
    if len(s) < 2:
        return float("nan"), float("nan")
    return float(s.iloc[-1]), float(s.iloc[-2])


def _pit_filing_dates(facts: dict, as_of: date) -> list[tuple[str, str]]:
    """Return [(fiscal_year_end, filed_date)] for the last two 10-Ks available as of as_of."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in ["NetIncomeLoss", "OperatingIncomeLoss", "Revenues"]:
        node = us_gaap.get(tag, {})
        rows = node.get("units", {}).get("USD", [])
        best: dict[pd.Timestamp, date] = {}
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            fs, end_str = row.get("filed"), row.get("end")
            if not fs or not end_str:
                continue
            try:
                filed = date.fromisoformat(fs)
                ts    = pd.Timestamp(end_str)
            except ValueError:
                continue
            if filed > as_of:
                continue
            if ts not in best or filed > best[ts]:
                best[ts] = filed
        if len(best) >= 2:
            sorted_entries = sorted(best.items())
            return [(str(ts.date()), str(fd)) for ts, fd in sorted_entries[-2:]]
    return []


# ═══════════════════════════════════════════════════════════════════════════════
# Piotroski F-Score
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fscore(facts: dict, as_of: date) -> tuple[int | None, dict]:
    """
    Piotroski F-Score (0-9) using the two most recent PIT 10-K filings.
    Returns (score, components_dict) or (None, {}) if insufficient data.
    """
    ni_c,  ni_p  = _two(facts, ["NetIncomeLoss"], as_of)
    cfo_c, _     = _two(facts, ["NetCashProvidedByUsedInOperatingActivities"], as_of)
    ast_c, ast_p = _two(facts, ["Assets"], as_of)
    rev_c, rev_p = _two(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    ], as_of)
    gp_c, gp_p   = _two(facts, ["GrossProfit"], as_of)
    ca_c,  ca_p  = _two(facts, ["AssetsCurrent"], as_of)
    cl_c,  cl_p  = _two(facts, ["LiabilitiesCurrent"], as_of)

    # Debt for leverage (try consolidated then long-term)
    ltd_s = _pit_series(facts, ["LongTermDebtNoncurrent", "LongTermDebt"], as_of)
    ltd_c = float(ltd_s.iloc[-1]) if len(ltd_s) >= 1 else float("nan")
    ltd_p = float(ltd_s.iloc[-2]) if len(ltd_s) >= 2 else float("nan")

    # Shares (period-end preferred)
    sh_s = _pit_series(facts, [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ], as_of)
    sh_c = float(sh_s.iloc[-1]) if len(sh_s) >= 1 else float("nan")
    sh_p = float(sh_s.iloc[-2]) if len(sh_s) >= 2 else float("nan")

    # Minimum viable data: need NI, CFO, total assets, revenue (2 years)
    if any(math.isnan(x) for x in [ni_c, cfo_c, ast_c, ast_p, rev_c, rev_p]):
        return None, {}
    if ast_c <= 0 or ast_p <= 0:
        return None, {}

    c: dict[str, int] = {}

    # F1: Profitability — ROA (NI/Assets) > 0
    c["F1"] = 1 if ni_c > 0 else 0

    # F2: Operating cash flow > 0
    c["F2"] = 1 if cfo_c > 0 else 0

    # F3: ROA improving YoY
    if not math.isnan(ni_p):
        c["F3"] = 1 if (ni_c / ast_c > ni_p / ast_p) else 0
    else:
        c["F3"] = 0

    # F4: Accruals quality — CFO > NI (cash earnings beat accrual earnings)
    c["F4"] = 1 if cfo_c > ni_c else 0

    # F5: Leverage falling (long-term debt / assets)
    if not any(math.isnan(x) for x in [ltd_c, ltd_p]):
        c["F5"] = 1 if (ltd_c / ast_c < ltd_p / ast_p) else 0
    else:
        c["F5"] = 0  # conservative when debt data missing

    # F6: Current ratio improving
    if not any(math.isnan(x) for x in [ca_c, ca_p, cl_c, cl_p]) and cl_c > 0 and cl_p > 0:
        c["F6"] = 1 if (ca_c / cl_c > ca_p / cl_p) else 0
    else:
        c["F6"] = 0

    # F7: Shares not increasing (>1% dilution fails)
    if not math.isnan(sh_c) and not math.isnan(sh_p) and sh_p > 0:
        c["F7"] = 1 if (sh_c <= sh_p * 1.01) else 0
    else:
        c["F7"] = 0

    # F8: Gross margin improving (skip = 0 if no gross profit data)
    if not any(math.isnan(x) for x in [gp_c, gp_p]) and rev_c > 0 and rev_p > 0:
        c["F8"] = 1 if (gp_c / rev_c > gp_p / rev_p) else 0
    else:
        c["F8"] = 0  # conservative; marks companies with no COGS disclosure

    # F9: Asset turnover improving
    c["F9"] = 1 if (rev_c / ast_c > rev_p / ast_p) else 0

    return sum(c.values()), c


# ═══════════════════════════════════════════════════════════════════════════════
# Form 4 insider data (M1)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_submissions(cik_padded: str) -> dict | None:
    """Fetch and cache EDGAR submissions JSON (7-day TTL)."""
    cache = CACHE_DIR / "submissions" / f"{cik_padded}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    try:
        data = _edgar_get(f"https://data.sec.gov/submissions/CIK{cik_padded}.json")
        cache.write_text(json.dumps(data))
        return data
    except Exception:
        return None


def _parse_form4(xml_text: str) -> dict:
    """Parse Form 4 XML; return {purchases, sales, is_insider}."""
    out = {"purchases": 0.0, "sales": 0.0, "is_insider": False}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for rel in root.findall(".//reportingOwnerRelationship"):
        if (rel.findtext("isDirector", "0").strip() in ("1", "true") or
                rel.findtext("isOfficer",  "0").strip() in ("1", "true")):
            out["is_insider"] = True
            break
    for txn in root.findall(".//nonDerivativeTransaction"):
        code   = txn.findtext("transactionCoding/transactionCode", "").strip()
        sh_txt = txn.findtext("transactionAmounts/transactionShares/value", "0")
        try:
            shares = abs(float(sh_txt.replace(",", "").strip()))
        except ValueError:
            shares = 0.0
        if code == "P":
            out["purchases"] += shares
        elif code == "S":
            out["sales"] += shares
    return out


def _form4_filings_in_window(submissions: dict, as_of: date) -> list[dict]:
    """List Form 4 filings in [as_of - INSIDER_WINDOW days, as_of]."""
    window_start = as_of - timedelta(days=INSIDER_WINDOW)
    results = []

    def _scan(sec):
        for acc, fd, form, pdoc in zip(
            sec.get("accessionNumber", []),
            sec.get("filingDate", []),
            sec.get("form", []),
            sec.get("primaryDocument", []),
        ):
            if form != "4":
                continue
            try:
                fd_d = date.fromisoformat(fd)
            except ValueError:
                continue
            if window_start <= fd_d <= as_of:
                results.append({"accn": acc, "pdoc": pdoc, "date": fd})

    recent = submissions.get("filings", {}).get("recent", {})
    _scan(recent)

    # If recent section doesn't reach back to window_start, fetch older files
    dates_in_recent = [d for d in recent.get("filingDate", []) if d]
    if dates_in_recent:
        oldest_recent = min(date.fromisoformat(d) for d in dates_in_recent)
        if oldest_recent > window_start:
            for file_info in submissions.get("filings", {}).get("files", []):
                fname     = file_info.get("name", "")
                date_to   = file_info.get("dateTo", "")
                date_from = file_info.get("dateFrom", "")
                try:
                    if date_to and date.fromisoformat(date_to) < window_start:
                        continue
                    if date_from and date.fromisoformat(date_from) > as_of:
                        continue
                except ValueError:
                    pass
                try:
                    older = _edgar_get(f"https://data.sec.gov/submissions/{fname}")
                    _scan(older)
                except Exception:
                    pass

    return results


_INSIDER_RATE = 0.0  # seconds between requests (set dynamically)
_SEC_ARCHIVES_REACHABLE = True   # set to False on first errno-101; skips all subsequent Form 4 fetches


def get_insider_summary(ticker: str, as_of: date, submissions: dict | None) -> dict | None:
    """
    Insider trading summary for the INSIDER_WINDOW days before as_of.
    Returns {purchases, sales, net_buying, n_form4, n_parsed} or None.
    Cached at (ticker, year) level since window is fixed to H1 of each year.
    """
    cache_key = f"{ticker.upper()}_{as_of.year}"
    cache_path = CACHE_DIR / "insider" / f"{cache_key}.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    if submissions is None:
        return None

    form4s = _form4_filings_in_window(submissions, as_of)
    if not form4s:
        # Distinguish "none in window" from "history doesn't go back far enough"
        dates_all = submissions.get("filings", {}).get("recent", {}).get("filingDate", [])
        if dates_all:
            oldest = min(date.fromisoformat(d) for d in dates_all if d)
            window_start = as_of - timedelta(days=INSIDER_WINDOW)
            if oldest > window_start:
                return None  # history doesn't go back far enough → unavailable

    cik_padded = _CIK_MAP.get(ticker.upper(), "0000000000") if _CIK_MAP else "0000000000"
    cik_num    = str(int(cik_padded))

    total_buys  = 0.0
    total_sells = 0.0
    n_parsed    = 0

    global _SEC_ARCHIVES_REACHABLE
    if not _SEC_ARCHIVES_REACHABLE:
        cache_path.write_text(json.dumps({"purchases": 0.0, "sales": 0.0,
                                           "net_buying": False, "n_form4": 0, "n_parsed": 0}))
        return None

    for filing in form4s[:15]:   # cap at 15 Form 4s per window
        acc_clean = filing["accn"].replace("-", "")
        pdoc      = filing["pdoc"]
        url       = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_clean}/{pdoc}"
        try:
            time.sleep(0.12)   # ~8 req/sec, well within SEC 10/sec limit
            req = urllib.request.Request(url, headers=_EDGAR_HDR)
            with urllib.request.urlopen(req, timeout=12) as r:
                xml_text = r.read().decode("utf-8", errors="replace")
            parsed = _parse_form4(xml_text)
            if parsed["is_insider"]:
                total_buys  += parsed["purchases"]
                total_sells += parsed["sales"]
                n_parsed    += 1
        except OSError as e:
            if getattr(e, "errno", None) == 101 or "unreachable" in str(e).lower():
                _SEC_ARCHIVES_REACHABLE = False   # network down — skip all remaining
                break
        except Exception:
            pass

    result = {
        "purchases": total_buys,
        "sales":     total_sells,
        "net_buying": total_buys > total_sells,
        "n_form4":   len(form4s),
        "n_parsed":  n_parsed,
    }
    cache_path.write_text(json.dumps(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Management Score (M1 insider, M2 dilution, M3 ROIC)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mgmt(facts: dict, as_of: date, insider: dict | None) -> dict:
    """
    Management score 0-3. Returns dict with total, m1, m2, m3 (each 0/1 or None).
    """
    # ── M2: shares outstanding < 2 fiscal years ago ───────────────────────────
    sh_s = _pit_series(facts, [
        "CommonStockSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfDilutedSharesOutstanding",
    ], as_of)
    if len(sh_s) >= 3:
        m2 = 1 if float(sh_s.iloc[-1]) <= float(sh_s.iloc[-3]) * 1.02 else 0
    elif len(sh_s) >= 2:
        m2 = 1 if float(sh_s.iloc[-1]) <= float(sh_s.iloc[-2]) * 1.02 else 0
    else:
        m2 = 0

    # ── M3: ROIC improving YoY (NOPAT / Invested Capital) ────────────────────
    ebit_c, ebit_p = _two(facts, ["OperatingIncomeLoss"], as_of)
    ast_c,  ast_p  = _two(facts, ["Assets"], as_of)
    cl_c,   cl_p   = _two(facts, ["LiabilitiesCurrent"], as_of)

    # Tax rate (mean of available history)
    tax_e = _pit_series(facts, ["IncomeTaxExpenseBenefit"], as_of)
    ptax  = _pit_series(facts, [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
    ], as_of)
    tax_rate = 0.25
    if not tax_e.empty and not ptax.empty:
        idx = tax_e.index.intersection(ptax.index)
        pt_ = ptax[idx]; tx_ = tax_e[idx]
        mask = pt_ > 0
        if mask.any():
            tax_rate = float((tx_[mask] / pt_[mask]).clip(0.0, 0.6).mean())

    if not any(math.isnan(x) for x in [ebit_c, ebit_p, ast_c, ast_p, cl_c, cl_p]):
        ic_c = ast_c - cl_c
        ic_p = ast_p - cl_p
        if ic_c > 0 and ic_p > 0:
            roic_c = ebit_c * (1 - tax_rate) / ic_c
            roic_p = ebit_p * (1 - tax_rate) / ic_p
            m3 = 1 if roic_c > roic_p else 0
        else:
            m3 = 0
    else:
        m3 = 0

    # ── M1: net insider buying ────────────────────────────────────────────────
    m1: int | None
    if insider is None:
        m1 = None                    # unavailable — excluded from avg score
    else:
        m1 = 1 if insider.get("net_buying", False) else 0

    # Total: sum available signals; None → not counted
    total = sum(v for v in [m1, m2, m3] if v is not None)

    return {"mgmt_total": total, "m1": m1, "m2": m2, "m3": m3}


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def run_precheck() -> bool:
    print("\n" + "=" * 70)
    print("  PRE-CHECK — Point-in-time for F-Score and Insider Data")
    print("=" * 70)

    ticker     = "AAPL"
    check_date = date(2018, 6, 30)
    facts      = _load_facts(ticker)

    # ── Check 1: Two consecutive PIT 10-K dates ───────────────────────────────
    print(f"\n[1] F-Score PIT: {ticker} as of {check_date}")
    if facts is None:
        print("  [ERROR] EDGAR facts not found — re-run H1 first")
        return False

    filing_dates = _pit_filing_dates(facts, check_date)
    if len(filing_dates) < 2:
        print(f"  [ERROR] Only {len(filing_dates)} 10-K available — need 2")
        return False

    prev_fy, prev_filed = filing_dates[-2]
    curr_fy, curr_filed = filing_dates[-1]
    print(f"  Current 10-K:  fiscal year end {curr_fy}  |  filed {curr_filed}")
    print(f"  Prior   10-K:  fiscal year end {prev_fy}  |  filed {prev_filed}")

    # Verify PIT: both filed dates must be ≤ check_date
    curr_ok = date.fromisoformat(curr_filed) <= check_date
    prev_ok = date.fromisoformat(prev_filed) <= check_date
    print(f"  Both filed ≤ {check_date}: {curr_ok and prev_ok}")

    # Spot-check: compute F-Score
    fscore, comps = compute_fscore(facts, check_date)
    if fscore is not None:
        print(f"  F-Score computed: {fscore}/9  ({comps})")
    else:
        print(f"  [WARN] F-Score returned None — insufficient data")

    check1_pass = len(filing_dates) >= 2 and curr_ok and prev_ok
    print(f"  CHECK 1: {'PASS ✓' if check1_pass else 'FAIL ✗'}")

    # ── Check 2: Form 4 / Insider data ───────────────────────────────────────
    print(f"\n[2] Insider data (M1): {ticker} as of {check_date}")
    print(f"  Window: {check_date - timedelta(days=INSIDER_WINDOW)} → {check_date}")
    print(f"  Fetching EDGAR submissions...")

    cik = _cik(ticker)
    if cik is None:
        print("  [ERROR] CIK not found")
        check2_pass = False
    else:
        subs = fetch_submissions(cik.zfill(10))
        if subs is None:
            print("  [ERROR] Could not fetch submissions JSON")
            check2_pass = False
        else:
            form4s = _form4_filings_in_window(subs, check_date)
            print(f"  Form 4 filings in window: {len(form4s)}")
            if not form4s:
                print("  [WARN] No Form 4 filings found in window")
                print("  M1: unavailable for this (ticker, date) — will score as NaN")
                check2_pass = False
            else:
                print(f"  Attempting to parse first Form 4: {form4s[0]}")
                cik_num   = str(int(cik))
                acc_clean = form4s[0]["accn"].replace("-", "")
                pdoc      = form4s[0]["pdoc"]
                url       = f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc_clean}/{pdoc}"
                try:
                    time.sleep(0.1)
                    req = urllib.request.Request(url, headers=_EDGAR_HDR)
                    with urllib.request.urlopen(req, timeout=15) as r:
                        xml_text = r.read().decode("utf-8", errors="replace")
                    parsed = _parse_form4(xml_text)
                    print(f"  Parsed Form 4: is_insider={parsed['is_insider']}  "
                          f"purchases={parsed['purchases']:.0f}  sales={parsed['sales']:.0f}")

                    # Try the full summary
                    ins = get_insider_summary(ticker, check_date, subs)
                    if ins:
                        print(f"  6-month summary: {ins['n_form4']} Form 4s found, "
                              f"{ins['n_parsed']} parsed (officer/director)")
                        print(f"  Total buys: {ins['purchases']:.0f} shares  |  "
                              f"sells: {ins['sales']:.0f} shares  |  "
                              f"net_buying: {ins['net_buying']}")
                        check2_pass = True
                    else:
                        print("  [WARN] Summary returned None — history may not go back far enough")
                        check2_pass = False
                except Exception as e:
                    print(f"  [ERROR] Could not parse Form 4: {e}")
                    check2_pass = False

    if not check2_pass:
        print("\n  *** M1 CAVEAT ***")
        print("  Insider Form 4 data is NOT reliably available for all (ticker, date) pairs.")
        print("  Likely causes: EDGAR submissions don't cover early dates (2015-2016),")
        print("  rate limits, or sparse filings for some companies.")
        print("  Strategy: M1 will be scored when available; treated as NaN otherwise.")
        print("  Management score will be computed as M2+M3 (0-2) when M1 is absent.")
        print("  Flag: M1 coverage will be reported per rebalance date.")

    print(f"\n  PRE-CHECK OVERALL: {'PASS ✓' if (check1_pass and check2_pass) else 'PARTIAL — see caveats above'}")
    return check1_pass   # F-Score PIT is the gate; M1 is best-effort


# ═══════════════════════════════════════════════════════════════════════════════
# Build quality panel
# ═══════════════════════════════════════════════════════════════════════════════

def build_quality_panel(h1_panel: pd.DataFrame) -> pd.DataFrame:
    """
    For each row in h1_panel (DCF-applicable), compute F-Score and Mgmt score.
    Loads EDGAR facts from H1 cache (no new HTTP requests for F/M2/M3).
    M1 requires Form 4 fetching (best-effort, rate-limited).
    """
    if QUALITY_CACHE.exists():
        print(f"\n  Loading quality scores from cache: {QUALITY_CACHE}")
        q = pd.read_parquet(QUALITY_CACHE)
        return h1_panel.merge(q, on=["ticker", "as_of"], how="left")

    # Work on DCF-applicable rows only
    working = h1_panel[h1_panel["dcf_applicable"]].copy()
    print(f"\n  Computing quality scores for {len(working):,} DCF-applicable rows...")

    # ── Pre-fetch submissions JSONs in parallel (for M1) ─────────────────────
    tickers_needed = working["ticker"].unique().tolist()
    global _CIK_MAP
    if _CIK_MAP is None:
        _CIK_MAP = json.loads(CIK_CACHE.read_text())

    print(f"  Fetching EDGAR submissions for {len(tickers_needed)} tickers (for M1)...")
    subs_map: dict[str, dict | None] = {}

    socket.setdefaulttimeout(15)

    def _fetch_subs(sym):
        cik = _CIK_MAP.get(sym.upper())
        if not cik:
            return sym, None
        return sym, fetch_submissions(cik.zfill(10))

    # Use wait() with a per-batch timeout so hung threads don't block forever.
    BATCH = 20
    PER_BATCH_TIMEOUT = 60   # seconds; each batch of 20 gets at most 60s
    done_count = 0
    with ThreadPoolExecutor(max_workers=5) as pool:
        for batch_start in range(0, len(tickers_needed), BATCH):
            batch = tickers_needed[batch_start:batch_start + BATCH]
            futs  = {pool.submit(_fetch_subs, sym): sym for sym in batch}
            completed, pending = __import__("concurrent.futures", fromlist=["wait"]).wait(
                futs, timeout=PER_BATCH_TIMEOUT
            )
            for f in completed:
                sym = futs[f]
                try:
                    _, subs = f.result()
                except Exception:
                    subs = None
                subs_map[sym] = subs
            for f in pending:
                subs_map[futs[f]] = None   # timed out → treat as unavailable
            done_count += len(batch)
            if done_count % 100 < BATCH:
                print(f"    submissions: {done_count}/{len(tickers_needed)}", flush=True)

    socket.setdefaulttimeout(None)

    # ── Score each row ────────────────────────────────────────────────────────
    rows_out = []
    n_total  = len(working)

    for i, (_, row) in enumerate(working.iterrows()):
        ticker = row["ticker"]
        as_of_ts = row["as_of"]
        as_of_d  = as_of_ts.date() if hasattr(as_of_ts, "date") else as_of_ts

        facts = _load_facts(ticker)
        if facts is None:
            rows_out.append({"ticker": ticker, "as_of": as_of_ts,
                             "fscore": None, "mgmt_total": None,
                             "m1": None, "m2": None, "m3": None,
                             **{f"F{k}": None for k in range(1, 10)}})
            continue

        # F-Score
        fscore, comps = compute_fscore(facts, as_of_d)

        # M1: insider (rate-limited; uses submissions pre-fetched above)
        subs = subs_map.get(ticker)
        insider = get_insider_summary(ticker, as_of_d, subs) if subs is not None else None

        # Mgmt
        mgmt = compute_mgmt(facts, as_of_d, insider)

        r = {
            "ticker":     ticker,
            "as_of":      as_of_ts,
            "fscore":     fscore,
            "mgmt_total": mgmt["mgmt_total"],
            "m1":         mgmt["m1"],
            "m2":         mgmt["m2"],
            "m3":         mgmt["m3"],
        }
        if comps:
            for k in range(1, 10):
                r[f"F{k}"] = comps.get(f"F{k}")
        else:
            for k in range(1, 10):
                r[f"F{k}"] = None
        rows_out.append(r)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_total} rows scored", flush=True)

    q_df = pd.DataFrame(rows_out)
    q_df.to_parquet(QUALITY_CACHE, index=False)
    print(f"  Quality scores saved: {QUALITY_CACHE}")

    return h1_panel.merge(q_df, on=["ticker", "as_of"], how="left")


# ═══════════════════════════════════════════════════════════════════════════════
# Portfolio construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _shared_universe(panel: pd.DataFrame) -> pd.DataFrame:
    """Shared universe: DCF-applicable + fwd_ret + dcf_vp + bp + ebitda_ev available."""
    df = panel[panel["dcf_applicable"] & panel["fwd_ret"].notna()].copy()
    df = df.dropna(subset=["dcf_vp", "bp", "ebitda_ev"]).copy()

    # Cross-sectional winsorise signals at 1/99 pct per date
    rows_out = []
    for _, grp in df.groupby("as_of"):
        if len(grp) < 10:
            continue
        g = grp.copy()
        for col in ["dcf_vp", "ebitda_ev", "size", "mom", "fscore", "mgmt_total"]:
            if col in g.columns:
                vals = pd.to_numeric(g[col], errors="coerce")
                lo, hi = vals.quantile(0.01), vals.quantile(0.99)
                g[col] = vals.clip(lo, hi)
        rows_out.append(g)
    return pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()


def _basket_returns(df: pd.DataFrame, value_col: str, fscore_gate: int | None,
                    mgmt_gate: int | None) -> dict:
    """
    Equal-weight basket returns per date.
    value_col: 'dcf_vp' or 'ebitda_ev'
    fscore_gate: None or minimum F-Score required (AND with top-quintile value)
    mgmt_gate:   None or minimum Mgmt required
    """
    q5_rets, bm_rets, n_stocks_list = [], [], []
    trap_all, trap_bm = [], []   # for value-trap rate

    for as_of, grp in df.groupby("as_of"):
        sub = grp.dropna(subset=[value_col, "fwd_ret"])
        if len(sub) < N_QUINTILES:
            continue

        # Top quintile by value signal
        q_thresh = sub[value_col].quantile(1 - 1 / N_QUINTILES)
        cheap    = sub[sub[value_col] >= q_thresh].copy()

        # Apply quality filters
        if fscore_gate is not None and "fscore" in cheap.columns:
            cheap = cheap.dropna(subset=["fscore"])
            cheap = cheap[cheap["fscore"].astype(float) >= fscore_gate]
        if mgmt_gate is not None and "mgmt_total" in cheap.columns:
            cheap = cheap.dropna(subset=["mgmt_total"])
            cheap = cheap[cheap["mgmt_total"].astype(float) >= mgmt_gate]

        if len(cheap) == 0:
            continue

        bm_ret = sub["fwd_ret"].mean()
        q5_ret = cheap["fwd_ret"].mean() - (COSTS_BPS / 10000)

        q5_rets.append(q5_ret)
        bm_rets.append(bm_ret)
        n_stocks_list.append(len(cheap))

        # Value-trap: fraction of names underperforming benchmark
        trap_all.append(len(cheap))
        trap_bm.append((cheap["fwd_ret"] < bm_ret).sum())

    return {
        "q5": np.array(q5_rets),
        "bm": np.array(bm_rets),
        "n_stocks": np.array(n_stocks_list),
        "trap_count": np.array(trap_bm),
        "trap_total": np.array(trap_all),
    }


def _perf(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {"ret": float("nan"), "sharpe": float("nan"), "maxdd": float("nan"), "n": 0}
    ret    = float(np.mean(returns))
    std    = float(np.std(returns, ddof=1))
    sharpe = ret / std if std > 0 else float("nan")
    cum    = np.cumprod(1 + returns)
    maxdd  = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {"ret": ret, "sharpe": sharpe, "maxdd": maxdd, "n": len(returns)}


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def report_coverage(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  COVERAGE: F-Score and Mgmt per rebalance date")
    print("=" * 70)

    df = panel[panel["dcf_applicable"]].copy()
    print(f"\n  {'Date':<14} {'Universe':>9} {'F-Score':>9} {'Mgmt≥0':>9} {'M1 avail':>10}")
    print(f"  {'-' * 55}")

    for as_of, grp in df.groupby("as_of"):
        n_total  = len(grp)
        n_fscore = grp["fscore"].notna().sum()
        n_mgmt   = grp["mgmt_total"].notna().sum()
        n_m1     = grp["m1"].notna().sum()
        date_str = str(as_of)[:10]
        print(f"  {date_str:<14} {n_total:>9} {n_fscore:>9} {n_mgmt:>9} {n_m1:>10}")


def test1_layer_contribution(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 1 — LAYER CONTRIBUTION  (A→B→C→D)")
    print("  *** Absolute returns are UPPER BOUNDS (survivorship bias) ***")
    print("  *** T=10 annual periods — direction matters more than t-stats ***")
    print("=" * 70)

    DEFLATED_SHARPE_TRIALS = 14   # documented for caller
    print(f"\n  [Deflated Sharpe note: {DEFLATED_SHARPE_TRIALS} basket definitions tested in this run]")

    for arm_label, value_col in [("DCF V/P arm", "dcf_vp"), ("EBITDA/EV arm", "ebitda_ev")]:
        print(f"\n  ── {arm_label} ──────────────────────────────────────────")
        print(f"  {'Basket':<28} {'Ann.Ret':>9} {'Sharpe':>8} {'MaxDD':>8}"
              f" {'vs BM':>8} {'N/date':>7}")
        print(f"  {'-' * 68}")

        baskets = [
            ("A  cheap only",              None,       None),
            (f"B  +F≥{F_THRESH_B}",        F_THRESH_B, None),
            (f"C  +F≥{F_THRESH_B}+Mgmt≥{M_THRESH_C}", F_THRESH_B, M_THRESH_C),
            (f"D  +F≥{F_THRESH_B}+Mgmt={M_THRESH_D}", F_THRESH_B, M_THRESH_D),
        ]

        prev_ret = None
        for label, fg, mg in baskets:
            r = _basket_returns(df, value_col, fg, mg)
            if len(r["q5"]) == 0:
                print(f"  {label:<28}  insufficient data")
                continue
            p = _perf(r["q5"])
            bm_ret = float(np.mean(r["bm"])) if len(r["bm"]) > 0 else float("nan")
            excess = p["ret"] - bm_ret
            n_avg  = float(np.mean(r["n_stocks"])) if len(r["n_stocks"]) > 0 else 0
            delta  = f"  Δ={p['ret']-prev_ret:+.1%}" if prev_ret is not None else ""
            print(f"  {label:<28} {p['ret']:>8.1%} {p['sharpe']:>8.2f} "
                  f"{p['maxdd']:>8.1%} {excess:>+8.1%} {n_avg:>7.0f}{delta}")
            prev_ret = p["ret"]

        print(f"  {'Benchmark':<28} {float(np.mean(_basket_returns(df, value_col, None, None)['bm'])):>8.1%}")


def test2_trap_rate(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 2 — VALUE-TRAP RATE")
    print("  Fraction of basket names that UNDERPERFORM the benchmark per year")
    print("=" * 70)
    print(f"\n  {'Basket':<35} {'DCF arm trap%':>14} {'EBITDA/EV arm trap%':>20}")
    print(f"  {'-' * 72}")

    baskets = [
        ("A  cheap only",              None,       None),
        (f"B  +F≥{F_THRESH_B}",        F_THRESH_B, None),
        (f"C  +F≥{F_THRESH_B}+Mgmt≥{M_THRESH_C}", F_THRESH_B, M_THRESH_C),
        (f"D  +F≥{F_THRESH_B}+Mgmt={M_THRESH_D}", F_THRESH_B, M_THRESH_D),
    ]

    for label, fg, mg in baskets:
        rates = {}
        for arm_col in ["dcf_vp", "ebitda_ev"]:
            r = _basket_returns(df, arm_col, fg, mg)
            if r["trap_total"].sum() > 0:
                rates[arm_col] = float(r["trap_count"].sum() / r["trap_total"].sum())
            else:
                rates[arm_col] = float("nan")
        dcf_r = f"{rates.get('dcf_vp', float('nan')):.1%}" if not math.isnan(rates.get("dcf_vp", float("nan"))) else "n/a"
        ev_r  = f"{rates.get('ebitda_ev', float('nan')):.1%}" if not math.isnan(rates.get("ebitda_ev", float("nan"))) else "n/a"
        print(f"  {label:<35} {dcf_r:>14} {ev_r:>20}")


def test3_fama_macbeth(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 3 — FAMA-MACBETH: Is quality INDEPENDENT of value+SIZE?")
    print("  THE GATE: F-Score and Mgmt positive/significant after value+SIZE?")
    print("=" * 70)

    sig_cols  = ["dcf_vp", "ebitda_ev", "size", "mom", "fscore", "mgmt_total"]
    coef_rows = []

    for as_of, grp in df.groupby("as_of"):
        sub = grp.dropna(subset=sig_cols + ["fwd_ret"]).copy()
        if len(sub) < 20:
            continue
        for col in sig_cols:
            mu  = sub[col].mean()
            std = sub[col].std()
            sub[col + "_z"] = (sub[col] - mu) / std if std > 1e-10 else 0.0
        X_cols = [c + "_z" for c in sig_cols]
        X = sub[X_cols].values
        y = sub["fwd_ret"].values
        try:
            coefs = np.linalg.lstsq(
                np.column_stack([np.ones(len(X)), X]), y, rcond=None
            )[0]
            row = {"as_of": as_of, "intercept": coefs[0]}
            for j, col in enumerate(sig_cols):
                row[col] = coefs[j + 1]
            coef_rows.append(row)
        except Exception:
            continue

    if len(coef_rows) < 3:
        print("  Insufficient periods for Fama-MacBeth (need ≥3)")
        return

    coef_df = pd.DataFrame(coef_rows)
    T = len(coef_df)

    label_map = {
        "dcf_vp": "DCF V/P", "ebitda_ev": "EBITDA/EV",
        "size": "SIZE", "mom": "MOM",
        "fscore": "F-Score", "mgmt_total": "Mgmt score",
    }
    print(f"\n  Periods: {T}  |  Cross-sectional z-scored signals, OLS on forward return")
    print(f"\n  {'Variable':<16} {'Mean coef':>12} {'t-stat':>9}  Verdict")
    print(f"  {'-' * 58}")

    for col in sig_cols:
        if col not in coef_df.columns:
            continue
        mean_c = coef_df[col].mean()
        se_c   = coef_df[col].std(ddof=1) / np.sqrt(T)
        t_stat = mean_c / se_c if se_c > 0 else float("nan")
        stars  = ("***" if abs(t_stat) > 2.576 else
                  "**"  if abs(t_stat) > 1.960 else
                  "*"   if abs(t_stat) > 1.645 else "")
        print(f"  {label_map.get(col, col):<16} {mean_c:>12.4f} {t_stat:>9.2f}  {stars}")

    print(f"\n  *** p<0.01   ** p<0.05   * p<0.10   (T={T}, note small sample)")

    # Gate verdict for F-Score and Mgmt
    print(f"\n  ══ GATE VERDICT ══")
    for col, lbl in [("fscore", "F-Score"), ("mgmt_total", "Mgmt score")]:
        if col not in coef_df.columns:
            continue
        mc  = coef_df[col].mean()
        se  = coef_df[col].std(ddof=1) / np.sqrt(T)
        t   = mc / se if se > 0 else float("nan")
        if mc > 0 and abs(t) >= 1.645:
            verdict = f"POSITIVE & SIGNIFICANT (t={t:.2f}) → adds independent value ✓"
        elif mc > 0:
            verdict = f"positive but weak (t={t:.2f}) → suggestive, not conclusive (T={T} is small)"
        else:
            verdict = f"NON-POSITIVE (t={t:.2f}) → captured by value + SIZE ✗"
        print(f"  {lbl}: coef={mc:.4f}  {verdict}")


def test4_breakdown(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 4 — COMPONENT BREAKDOWN + F-SCORE THRESHOLD SWEEP")
    print("=" * 70)

    # ── Individual Mgmt signals ───────────────────────────────────────────────
    print(f"\n  [4a] Individual management signals (DCF arm, F≥{F_THRESH_B})")
    print(f"  Each = DCF-cheap + F≥{F_THRESH_B} + that signal = 1")
    print(f"\n  {'Signal':<22} {'Ann.Ret':>9} {'Sharpe':>8} {'MaxDD':>8} {'vs B basket':>12}")
    print(f"  {'-' * 62}")

    # Baseline B
    r_B = _basket_returns(df, "dcf_vp", F_THRESH_B, None)
    p_B = _perf(r_B["q5"])

    for m_col, m_label in [("m1", "M1 insider buy"),
                            ("m2", "M2 no dilution"),
                            ("m3", "M3 ROIC improve")]:
        if m_col not in df.columns:
            continue
        # Inject as a filter: DCF-cheap + F≥7 + this m-signal=1
        rets = []
        for as_of, grp in df.groupby("as_of"):
            sub = grp.dropna(subset=["dcf_vp", "fwd_ret", "fscore", m_col])
            if len(sub) < N_QUINTILES:
                continue
            q_thresh = sub["dcf_vp"].quantile(0.8)
            cheap    = sub[sub["dcf_vp"] >= q_thresh]
            cheap    = cheap[cheap["fscore"].astype(float) >= F_THRESH_B]
            cheap    = cheap[cheap[m_col].astype(float) == 1]
            if len(cheap) == 0:
                continue
            rets.append(cheap["fwd_ret"].mean() - COSTS_BPS / 10000)
        if not rets:
            print(f"  {m_label:<22}  no data")
            continue
        p = _perf(np.array(rets))
        delta = p["ret"] - p_B["ret"]
        print(f"  {m_label:<22} {p['ret']:>8.1%} {p['sharpe']:>8.2f} "
              f"{p['maxdd']:>8.1%} {delta:>+12.1%}")

    print(f"  {'Basket B (F≥7 only)':<22} {p_B['ret']:>8.1%} {p_B['sharpe']:>8.2f} "
          f"{p_B['maxdd']:>8.1%}   ← baseline")

    # ── F-Score threshold sweep ───────────────────────────────────────────────
    print(f"\n  [4b] F-Score threshold sweep (DCF arm)")
    print(f"  Is the effect monotone in threshold, or a single lucky cut?")
    print(f"\n  {'Threshold':<22} {'Ann.Ret':>9} {'Sharpe':>8} {'MaxDD':>8} {'N/date':>8}")
    print(f"  {'-' * 58}")

    for thresh in [5, 6, 7, 8, 9]:
        r = _basket_returns(df, "dcf_vp", thresh, None)
        if len(r["q5"]) == 0:
            print(f"  F ≥ {thresh:<18} no data")
            continue
        p = _perf(r["q5"])
        n = float(np.mean(r["n_stocks"])) if len(r["n_stocks"]) > 0 else 0
        print(f"  F ≥ {thresh:<18} {p['ret']:>8.1%} {p['sharpe']:>8.2f} "
              f"{p['maxdd']:>8.1%} {n:>8.0f}")

    print(f"\n  Monotone expected: F≥5 < F≥7 < F≥9 in Sharpe if quality is real.")
    print(f"  If non-monotone: survivorship / small-sample artefact more likely.")


def test5_dcf_vs_multiple_with_quality(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 5 — DCF+QUALITY vs EBITDA/EV+QUALITY")
    print("  Is DCF still adding value once quality is controlled?")
    print("=" * 70)

    print(f"\n  {'Basket':<38} {'Ann.Ret':>9} {'Sharpe':>8} {'MaxDD':>8}")
    print(f"  {'-' * 65}")

    for label, arm, fg, mg in [
        ("DCF V/P  — cheap only (A)",          "dcf_vp",   None,       None),
        (f"DCF V/P  — +F≥{F_THRESH_B} (B)",   "dcf_vp",   F_THRESH_B, None),
        (f"EBITDA/EV — cheap only (A')",        "ebitda_ev", None,      None),
        (f"EBITDA/EV — +F≥{F_THRESH_B} (B')",  "ebitda_ev", F_THRESH_B, None),
        (f"DCF V/P  — +F≥{F_THRESH_B}+Mgmt≥{M_THRESH_C} (C)",  "dcf_vp",   F_THRESH_B, M_THRESH_C),
        (f"EBITDA/EV — +F≥{F_THRESH_B}+Mgmt≥{M_THRESH_C} (C')", "ebitda_ev", F_THRESH_B, M_THRESH_C),
    ]:
        r = _basket_returns(df, arm, fg, mg)
        if len(r["q5"]) == 0:
            print(f"  {label:<38}  no data")
            continue
        p = _perf(r["q5"])
        print(f"  {label:<38} {p['ret']:>8.1%} {p['sharpe']:>8.2f} {p['maxdd']:>8.1%}")

    print(f"\n  VERDICT:")
    r_B  = _basket_returns(df, "dcf_vp",    F_THRESH_B, None)
    r_Bp = _basket_returns(df, "ebitda_ev", F_THRESH_B, None)
    if len(r_B["q5"]) > 0 and len(r_Bp["q5"]) > 0:
        diff = _perf(r_B["q5"])["ret"] - _perf(r_Bp["q5"])["ret"]
        print(f"  B (DCF+F) minus B' (EBITDA/EV+F) return differential: {diff:+.1%}")
        if abs(diff) < 0.02:
            print("  Differential < 2%: DCF and free-multiple are essentially equivalent")
            print("  once quality is layered on. DCF adds no incremental signal here.")
        elif diff > 0:
            print("  DCF outperforms free multiple even with quality filter.")
        else:
            print("  EBITDA/EV outperforms DCF even with quality filter.")
            print("  Free multiple + quality beats DCF + quality.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("  HYPOTHESIS 2: QUALITY FILTER ON VALUE BASKET")
    print("  Piotroski F-Score (0-9) + Management Score (0-3)")
    print("  Reusing H1 signal panel + EDGAR cache — no DCF recomputation")
    print("  SURVIVORSHIP NOTE: absolute returns are upper bounds.")
    print("  DCF-vs-multiple and layer differentials are survivorship-robust.")
    print("  ONE TRIAL set — logged for Deflated Sharpe.")
    print("#" * 70)

    # ── PRE-CHECK ─────────────────────────────────────────────────────────────
    ok = run_precheck()
    if not ok:
        print("\n[STOP] F-Score PIT check failed. Cannot compute quality scores reliably.")
        sys.exit(1)

    # ── Load H1 signal panel ──────────────────────────────────────────────────
    print(f"\n  Loading H1 signal panel: {H1_PANEL}")
    h1 = pd.read_parquet(H1_PANEL)
    h1["as_of"] = pd.to_datetime(h1["as_of"])

    # ── Forward returns (not stored in parquet — compute from price history) ────
    if "fwd_ret" not in h1.columns:
        h1 = add_fwd_ret(h1)
    h1 = h1[h1["fwd_ret"].notna()].copy()
    print(f"  H1 panel rows with fwd_ret: {len(h1):,}")

    # ── Build quality panel ───────────────────────────────────────────────────
    panel = build_quality_panel(h1)
    print(f"  Panel after quality merge: {len(panel):,} rows")

    # ── Shared universe (same definition as H1) ───────────────────────────────
    shared = _shared_universe(panel)
    if shared.empty:
        print("[ERROR] Shared universe empty after filters")
        sys.exit(1)

    print(f"\n  Shared universe rows: {len(shared):,}")
    print(f"  Unique tickers:       {shared['ticker'].nunique():,}")
    print(f"  Dates with F-Score:   {shared.groupby('as_of')['fscore'].apply(lambda x: x.notna().any()).sum()}")
    print(f"  F-Score coverage:     {shared['fscore'].notna().mean():.1%}")
    print(f"  Mgmt coverage:        {shared['mgmt_total'].notna().mean():.1%}")
    print(f"  M1 coverage:          {shared['m1'].notna().mean():.1%}")

    # ── Tests ─────────────────────────────────────────────────────────────────
    report_coverage(panel)
    test1_layer_contribution(shared)
    test2_trap_rate(shared)
    test3_fama_macbeth(shared)
    test4_breakdown(shared)
    test5_dcf_vs_multiple_with_quality(shared)

    print("\n" + "#" * 70)
    print("  RUN COMPLETE")
    print("#" * 70)
    print(f"  Quality cache: {QUALITY_CACHE}")
    print(f"  Delete to recompute: rm {QUALITY_CACHE}")


if __name__ == "__main__":
    main()
