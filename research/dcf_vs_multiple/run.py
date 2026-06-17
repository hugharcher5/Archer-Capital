"""
research/dcf_vs_multiple/run.py

Hypothesis 1: Does the DCF engine beat a free multiple?

Signals (HIGHER = CHEAPER, winsorised cross-sectionally 1/99 pct each date):
  DCF V/P    = point-in-time deterministic DCF intrinsic value / price
  B/P        = book equity per share / price
  EBITDA/EV  = EBITDA / enterprise value

Controls: SIZE = log(mktcap), MOM = 12-1 month return.

Tests:
  1. Horse race: top-quintile annualised return, Sharpe, max DD — DCF vs multiples vs benchmark
  2. Fama-MacBeth: is DCF V/P coefficient positive/significant after controlling for B/P, EBITDA/EV?
  3. Double-sort: B/P tercile × DCF V/P tercile — does DCF separate returns within multiple buckets?

NOTE — SURVIVORSHIP BIAS: config/dcf_universe.csv is a current snapshot.
Absolute returns are UPPER BOUNDS. The DCF-minus-multiple DIFFERENTIAL is survivorship-robust
(bias is common to both arms). Report differential as headline; label absolute Sharpes "upper bound".

NOTE — ONE TRIAL: do not tune. Record for Deflated Sharpe adjustment.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import urllib.request
import urllib.error
from scipy import stats

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "dcf"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")
from valuation.dcf import (
    Assumptions, value as dcf_value,
    TERMINAL_G, HIGH_GROWTH_THRESHOLD, MATURE_MARGIN_DEFAULT,
)

# ── Parameters ────────────────────────────────────────────────────────────────
REBALANCE     = "annual"     # "annual" | "quarterly"
REB_MONTH     = 6
REB_DAY       = 30
START_YEAR    = 2015
END_YEAR      = 2025        # forward returns through June 2025
N_QUINTILES   = 5
COSTS_BPS     = 20          # round-trip basis points
SECTOR_EXCL   = {"Financial Services", "Real Estate"}
BETA_LOOKBACK = 504         # ~2 years trading days for rolling beta
MOM_WINDOW    = 252         # 12 months for momentum
MOM_SKIP      = 21          # skip last month
ERP           = 0.045       # Damodaran implied ERP (mirrors existing engine)

UNIVERSE_CSV  = REPO / "config" / "dcf_universe.csv"
CACHE_DIR     = Path(__file__).parent / "_cache"
for _d in [CACHE_DIR, CACHE_DIR / "edgar_facts"]:
    _d.mkdir(parents=True, exist_ok=True)

SIGNALS_CACHE = CACHE_DIR / "signals_panel.parquet"

_SYNTH_RATING: list[tuple[float, float, float]] = [
    (12.50, float("inf"), 0.0063),
    ( 9.50,       12.50,  0.0078),
    ( 7.50,        9.50,  0.0098),
    ( 6.00,        7.50,  0.0113),
    ( 4.50,        6.00,  0.0128),
    ( 4.00,        4.50,  0.0163),
    ( 3.50,        4.00,  0.0238),
    ( 3.00,        3.50,  0.0313),
    ( 2.50,        3.00,  0.0388),
    ( 2.00,        2.50,  0.0463),
    ( 1.50,        2.00,  0.0563),
    ( 1.25,        1.50,  0.0713),
    ( 0.80,        1.25,  0.0863),
    ( 0.50,        0.80,  0.1113),
    (float("-inf"), 0.50,  0.1413),
]

_EDGAR_HDR   = {"User-Agent": "Archer Capital research@archercapital.dev"}
_TICKERS_CACHE = CACHE_DIR / "edgar_tickers.json"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGAR helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _edgar_get(url: str) -> dict:
    req = urllib.request.Request(url, headers=_EDGAR_HDR)
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
        if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return json.loads(raw.decode())


_CIK_MAP: dict[str, str] | None = None

def _get_cik_map() -> dict[str, str]:
    global _CIK_MAP
    if _CIK_MAP is not None:
        return _CIK_MAP
    if _TICKERS_CACHE.exists() and (time.time() - _TICKERS_CACHE.stat().st_mtime) < 7 * 86400:
        _CIK_MAP = json.loads(_TICKERS_CACHE.read_text())
        return _CIK_MAP
    data = _edgar_get("https://www.sec.gov/files/company_tickers.json")
    _CIK_MAP = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
    _TICKERS_CACHE.write_text(json.dumps(_CIK_MAP))
    return _CIK_MAP


def fetch_edgar_facts(ticker: str) -> dict | None:
    """Fetch + disk-cache EDGAR company facts JSON (7-day TTL)."""
    cache = CACHE_DIR / "edgar_facts" / f"{ticker.upper()}.json"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 7 * 86400:
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    cik_map = _get_cik_map()
    cik = cik_map.get(ticker.upper())
    if cik is None:
        return None
    try:
        facts = _edgar_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json")
        cache.write_text(json.dumps(facts))
        return facts
    except Exception:
        return None


def _pit_series(facts: dict, tags: list[str], as_of: date, abs_val: bool = False) -> pd.Series:
    """
    Point-in-time annual 10-K series.
    For each fiscal-year-end, keeps the value from the most recent filing with filed <= as_of.
    Handles restatements: if a 10-K/A was filed before as_of, its values supersede the original.
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node  = us_gaap.get(tag, {})
        units = node.get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        # best[(fiscal_year_end_ts)] = (filed_date, value)
        best: dict[pd.Timestamp, tuple[date, float]] = {}
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            filed_str = row.get("filed")
            if not filed_str:
                continue
            try:
                filed = date.fromisoformat(filed_str)
            except ValueError:
                continue
            if filed > as_of:
                continue
            end_str = row.get("end")
            val     = row.get("val")
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
    """Most recent 10-K balance-sheet scalar with filed <= as_of."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node  = us_gaap.get(tag, {})
        units = node.get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        best_filed: date | None = None
        best_val:   float       = float("nan")
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            filed_str = row.get("filed")
            if not filed_str:
                continue
            try:
                filed = date.fromisoformat(filed_str)
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
                    best_filed = filed
                    best_val   = v
            except (ValueError, TypeError):
                pass
        if not math.isnan(best_val):
            return best_val
    return float("nan")


def _latest_10k_filed(facts: dict, as_of: date) -> date | None:
    """Most recent 10-K filing date with filed <= as_of (uses revenue as proxy)."""
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    best: date | None = None
    for tag in [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet", "OperatingIncomeLoss",
    ]:
        node = us_gaap.get(tag, {})
        for row in (node.get("units", {}).get("USD", []) or []):
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            fs = row.get("filed")
            if not fs:
                continue
            try:
                f = date.fromisoformat(fs)
            except ValueError:
                continue
            if f <= as_of and (best is None or f > best):
                best = f
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Price / market data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _price_at(hist: pd.DataFrame | None, as_of: date) -> float | None:
    if hist is None or hist.empty:
        return None
    ts  = pd.Timestamp(as_of)
    sub = hist[hist.index <= ts]
    if sub.empty:
        return None
    return float(sub["Close"].iloc[-1])


def _beta_at(stock_hist: pd.DataFrame | None, mkt_hist: pd.DataFrame,
             as_of: date, lookback: int = BETA_LOOKBACK) -> float:
    if stock_hist is None or stock_hist.empty:
        return 1.0
    ts    = pd.Timestamp(as_of)
    s_ret = stock_hist[stock_hist.index <= ts].tail(lookback)["Close"].pct_change().dropna()
    m_ret = mkt_hist  [mkt_hist.index   <= ts].tail(lookback)["Close"].pct_change().dropna()
    idx   = s_ret.index.intersection(m_ret.index)
    if len(idx) < 60:
        return 1.0
    slope, *_ = stats.linregress(m_ret[idx].values, s_ret[idx].values)
    return float(np.clip(slope, 0.1, 4.0))


def _rf_at(tnx_hist: pd.DataFrame | None, as_of: date) -> float:
    if tnx_hist is None or tnx_hist.empty:
        return 0.043
    p = _price_at(tnx_hist, as_of)
    return (p / 100.0) if (p and p > 0) else 0.043


def _mom_at(hist: pd.DataFrame | None, as_of: date) -> float:
    if hist is None or hist.empty:
        return float("nan")
    ts  = pd.Timestamp(as_of)
    sub = hist[hist.index <= ts].tail(MOM_WINDOW)
    if len(sub) < MOM_WINDOW // 2:
        return float("nan")
    p_end   = float(sub["Close"].iloc[-(MOM_SKIP + 1)])
    p_start = float(sub["Close"].iloc[0])
    if p_start <= 0:
        return float("nan")
    return (p_end - p_start) / p_start


def _synth_spread(coverage: float) -> tuple[str, float]:
    for lo, hi, spread in _SYNTH_RATING:
        if lo <= coverage < hi:
            rating = {
                0.0063: "Aaa/AAA", 0.0078: "Aa2/AA", 0.0098: "A1/A+", 0.0113: "A2/A",
                0.0128: "A3/A-",   0.0163: "Baa2/BBB", 0.0238: "Ba1/BB+", 0.0313: "Ba2/BB",
                0.0388: "B1/B+",   0.0463: "B2/B",  0.0563: "B3/B-", 0.0713: "Caa/CCC",
                0.0863: "Ca2/CC",  0.1113: "C2/C",  0.1413: "D2/D",
            }.get(spread, "?")
            return rating, spread
    return "D2/D", 0.1413


# ═══════════════════════════════════════════════════════════════════════════════
# Point-in-time DCF (reuses existing dcf.value() for computation)
# ═══════════════════════════════════════════════════════════════════════════════

def pit_dcf(
    ticker:   str,
    facts:    dict,
    as_of:    date,
    price:    float,
    mktcap:   float,
    beta_raw: float,
    rf:       float,
) -> float | None:
    """
    Point-in-time deterministic DCF value per share (USD).
    Uses EDGAR filings with filed <= as_of.
    Calls the existing dcf.value() (pure, no I/O) for computation.
    Returns None if data insufficient or DCF gate fails.
    """
    revenue = _pit_series(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    ], as_of)
    ebit = _pit_series(facts, ["OperatingIncomeLoss"], as_of)
    da   = _pit_series(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ], as_of, abs_val=True)
    capex = _pit_series(facts, [
        "PaymentsToAcquirePropertyPlantAndEquipment",
    ], as_of, abs_val=True)
    tax_exp = _pit_series(facts, ["IncomeTaxExpenseBenefit"], as_of)
    pretax  = _pit_series(facts, [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
    ], as_of)
    int_exp = _pit_series(facts, [
        "InterestExpense", "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ], as_of, abs_val=True)

    cash       = _pit_scalar(facts, ["CashAndCashEquivalentsAtCarryingValue"], as_of)
    ltd        = _pit_scalar(facts, ["LongTermDebtNoncurrent", "LongTermDebt"], as_of)
    cur_debt   = _pit_scalar(facts, ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"], as_of)
    diluted_sh = _pit_scalar(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding"], as_of)
    curr_assets = _pit_scalar(facts, ["AssetsCurrent"], as_of)
    curr_liab   = _pit_scalar(facts, ["LiabilitiesCurrent"], as_of)

    if revenue.empty or ebit.empty or len(revenue) < 2:
        return None
    if math.isnan(diluted_sh) or diluted_sh <= 0:
        return None

    cash       = 0.0 if math.isnan(cash) else cash
    total_debt = (0.0 if math.isnan(ltd) else ltd) + (0.0 if math.isnan(cur_debt) else cur_debt)

    # Align income series
    df_s = pd.concat({"revenue": revenue, "ebit": ebit}, axis=1).dropna()
    if len(df_s) < 2:
        return None

    rev_growth  = float(np.clip(df_s["revenue"].pct_change().dropna().mean(), -0.50, 0.80))
    ebit_margin = float(np.clip((df_s["ebit"] / df_s["revenue"]).mean(), -0.30, 0.60))
    best_margin = float((df_s["ebit"] / df_s["revenue"]).max())
    target_marg = max(best_margin, MATURE_MARGIN_DEFAULT)

    if da.empty:
        da_pct = 0.03
    else:
        idx_da = df_s.index.intersection(da.index)
        da_pct = float(np.clip((da[idx_da] / df_s["revenue"][idx_da]).mean(), 0.0, 0.30)) if len(idx_da) else 0.03

    if capex.empty:
        capex_pct = 0.03
    else:
        idx_cx = df_s.index.intersection(capex.index)
        capex_pct = float(np.clip((capex[idx_cx] / df_s["revenue"][idx_cx]).mean(), 0.0, 0.50)) if len(idx_cx) else 0.03

    # Tax rate
    tax_rate = 0.25
    if not tax_exp.empty and not pretax.empty:
        idx_t = tax_exp.index.intersection(pretax.index)
        pt_  = pretax[idx_t]
        tx_  = tax_exp[idx_t]
        mask = pt_ > 0
        if mask.any():
            tax_rate = float((tx_[mask] / pt_[mask]).clip(0.0, 0.6).mean())

    # NWC
    nwc_pct = 0.05
    rev_last = float(df_s["revenue"].iloc[-1])
    if not math.isnan(curr_assets) and not math.isnan(curr_liab) and rev_last > 0:
        nwc = (curr_assets - cash) - curr_liab
        nwc_pct = float(np.clip(nwc / rev_last, -0.30, 0.50))

    # WACC
    beta_adj = 0.67 * beta_raw + 0.33
    coe      = rf + beta_adj * ERP
    ebit_last = float(df_s["ebit"].iloc[-1])
    int_last  = float(int_exp.iloc[-1]) if not int_exp.empty else 0.0
    coverage  = ebit_last / int_last if int_last > 0 else float("inf")
    rating, spread = _synth_spread(coverage)
    cod = (rf + spread) * (1 - tax_rate)
    E   = mktcap
    D   = total_debt
    V   = E + D
    eq_w = E / V if V > 0 else 1.0
    dt_w = D / V if V > 0 else 0.0
    wacc = eq_w * coe + dt_w * cod

    # Terminal growth
    avg_g      = (rev_growth + TERMINAL_G) / 2.0
    terminal_g = max(min(avg_g, TERMINAL_G), 0.015)
    if wacc <= terminal_g + 0.01:
        terminal_g = wacc - 0.01
    if wacc <= terminal_g:
        return None

    net_debt     = total_debt - cash
    forecast_yrs = 10 if rev_growth > HIGH_GROWTH_THRESHOLD else 5
    ebitda_est   = ebit_last + (float(da.iloc[-1]) if not da.empty else 0.0)

    a = Assumptions(
        ticker=ticker,             currency="USD",
        start_revenue=rev_last,    revenue_growth=rev_growth,
        ebit_margin=ebit_margin,   target_margin=target_marg,
        tax_rate=tax_rate,         da_pct=da_pct,
        capex_pct=capex_pct,       nwc_pct=nwc_pct,
        terminal_g=terminal_g,     wacc=wacc,
        net_debt=net_debt,         diluted_shares=diluted_sh,
        fx_rate=1.0,               forecast_years=forecast_yrs,
        rf=rf,                     erp=ERP,
        beta_adj=beta_adj,         cost_of_equity=coe,
        cost_of_debt_pretax=rf + spread,
        cost_of_debt_aftertax=cod,
        equity_weight=eq_w,        debt_weight=dt_w,
        implied_rating=rating,
        current_price_usd=price,
        current_ebitda=ebitda_est,
        market_ev_local=mktcap + net_debt,
    )

    try:
        vps = dcf_value(a)
    except Exception:
        return None

    if not math.isfinite(vps) or vps <= 0:
        return None

    # DCF applicability gate (mirrors existing engine)
    terminal_rev  = rev_last * (1 + (rev_growth + terminal_g) / 2) ** forecast_yrs
    terminal_fcff = (terminal_rev * target_marg * (1 - tax_rate)
                     + terminal_rev * da_pct
                     - terminal_rev * min(capex_pct, da_pct))
    if terminal_fcff <= 0:
        return None

    return vps


# ═══════════════════════════════════════════════════════════════════════════════
# Signal computation (all signals for one ticker at one date)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_signals(
    ticker:   str,
    facts:    dict,
    as_of:    date,
    ph:       pd.DataFrame | None,
    mkt_hist: pd.DataFrame,
    tnx_hist: pd.DataFrame,
) -> dict | None:
    price = _price_at(ph, as_of)
    if price is None or price <= 0:
        return None

    diluted_sh = _pit_scalar(facts, [
        "WeightedAverageNumberOfDilutedSharesOutstanding"
    ], as_of)
    if math.isnan(diluted_sh) or diluted_sh <= 0:
        return None

    mktcap = price * diluted_sh

    cash     = _pit_scalar(facts, ["CashAndCashEquivalentsAtCarryingValue"], as_of)
    cash     = 0.0 if math.isnan(cash) else cash
    ltd      = _pit_scalar(facts, ["LongTermDebtNoncurrent", "LongTermDebt"], as_of)
    cur_d    = _pit_scalar(facts, ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"], as_of)
    total_debt = (0.0 if math.isnan(ltd) else ltd) + (0.0 if math.isnan(cur_d) else cur_d)
    net_debt   = total_debt - cash

    book_eq = _pit_scalar(facts, [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ], as_of)

    revenue = _pit_series(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    ], as_of)
    ebit = _pit_series(facts, ["OperatingIncomeLoss"], as_of)
    da   = _pit_series(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ], as_of, abs_val=True)

    # EBITDA (most recent FY with both ebit and da)
    ebitda = float("nan")
    if not ebit.empty:
        ebit_last = float(ebit.iloc[-1])
        da_last   = float(da.iloc[-1]) if not da.empty else 0.0
        ebitda    = ebit_last + da_last

    ev = mktcap + net_debt
    ebitda_ev = (ebitda / ev
                 if (ev > 0 and not math.isnan(ebitda) and ebitda > 0)
                 else float("nan"))

    bp = float("nan")
    if not math.isnan(book_eq) and diluted_sh > 0 and price > 0:
        bp = (book_eq / diluted_sh) / price

    size = math.log(mktcap) if mktcap > 0 else float("nan")
    mom  = _mom_at(ph, as_of)

    rf       = _rf_at(tnx_hist, as_of)
    beta_raw = _beta_at(ph, mkt_hist, as_of)
    vps      = pit_dcf(ticker, facts, as_of, price, mktcap, beta_raw, rf)
    dcf_vp   = vps / price if vps is not None else float("nan")

    last_filed = _latest_10k_filed(facts, as_of)

    return {
        "ticker":         ticker,
        "as_of":          as_of,
        "price":          price,
        "mktcap":         mktcap,
        "net_debt":       net_debt,
        "dcf_vp":         dcf_vp,
        "bp":             bp,
        "ebitda_ev":      ebitda_ev,
        "size":           size,
        "mom":            mom,
        "dcf_applicable": vps is not None,
        "last_filed":     str(last_filed) if last_filed else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — POINT-IN-TIME PRE-CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def run_precheck() -> bool:
    print("\n" + "=" * 70)
    print("  POINT-IN-TIME PRE-CHECK — AAPL as of 2018-06-30")
    print("=" * 70)
    PRECHECK_DATE = date(2018, 6, 30)

    # ── Arm A: existing yfinance engine (always latest) ───────────────────────
    print("\n[A] Existing engine (yfinance — always returns latest data, no date param)")
    try:
        import io, contextlib
        from valuation.data import fetch_raw
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            raw = fetch_raw("AAPL")
        rev_years   = sorted(set(ts.year for ts in raw.revenue.index))
        latest_fy   = max(rev_years)
        latest_rev  = float(raw.revenue.iloc[-1]) / 1e9
        print(f"  Revenue fiscal years in yfinance data: {rev_years}")
        print(f"  Most recent: FY{latest_fy}, revenue = ${latest_rev:.1f}B")
        print(f"  yfinance does NOT expose filing dates — no as_of filtering possible")
        print(f"  → As of 2018-06-30, FY{latest_fy} data did NOT exist yet → LOOK-AHEAD BIAS")
    except Exception as e:
        print(f"  [WARN] Could not run existing engine: {e}")
        latest_fy = 2023

    # ── Arm B: point-in-time EDGAR engine ────────────────────────────────────
    print(f"\n[B] Point-in-time EDGAR engine (filed ≤ {PRECHECK_DATE})")
    facts = fetch_edgar_facts("AAPL")
    if facts is None:
        print("  [ERROR] Could not fetch EDGAR facts for AAPL")
        return False

    # Show all 10-K revenue filings with filed dates
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    rev_rows: list[dict] = []
    for tag in ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"]:
        node = us_gaap.get(tag, {})
        for row in node.get("units", {}).get("USD", []):
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            if not row.get("filed") or not row.get("end"):
                continue
            try:
                rev_rows.append({
                    "fiscal_year_end": row["end"],
                    "filed":           row["filed"],
                    "revenue_B":       float(row["val"]) / 1e9,
                    "available":       date.fromisoformat(row["filed"]) <= PRECHECK_DATE,
                })
            except Exception:
                pass
        if rev_rows:
            break

    rev_rows = sorted(rev_rows, key=lambda r: r["fiscal_year_end"])

    print(f"\n  AAPL 10-K revenue filings (last 8):")
    print(f"  {'Fiscal Year End':<18} {'Filed':<14} {'Revenue':>10}   Available at 2018-06-30?")
    print(f"  {'-' * 62}")
    for r in rev_rows[-8:]:
        avail_str = "✓  YES" if r["available"] else "✗  NO (filed after)"
        print(f"  {r['fiscal_year_end']:<18} {r['filed']:<14} {r['revenue_B']:>8.1f}B   {avail_str}")

    rev_pit = _pit_series(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    ], PRECHECK_DATE)

    if rev_pit.empty:
        print(f"\n  PIT engine found NO revenue data filed ≤ {PRECHECK_DATE}")
        print("  PRE-CHECK: FAIL")
        return False

    pit_fy       = rev_pit.index[-1].year
    last_filed   = _latest_10k_filed(facts, PRECHECK_DATE)
    pit_rev_last = float(rev_pit.iloc[-1]) / 1e9

    print(f"\n  PIT engine result (filed ≤ {PRECHECK_DATE}):")
    pit_years = [ts.year for ts in rev_pit.index]
    print(f"  Revenue fiscal years: {pit_years}")
    print(f"  Most recent: FY{pit_fy}, revenue = ${pit_rev_last:.1f}B")
    print(f"  Most recent 10-K filing used: {last_filed}")

    print(f"\n  COMPARISON:")
    print(f"  Existing engine:  FY{latest_fy}  (no date filtering — always latest)")
    print(f"  PIT engine:       FY{pit_fy}   (correctly filters to filed ≤ {PRECHECK_DATE})")

    pit_ok = pit_fy < latest_fy
    if pit_ok:
        print(f"\n  PRE-CHECK: PASS ✓")
        print(f"  PIT engine uses FY{pit_fy} data — {latest_fy - pit_fy} years behind current.")
        print(f"  Would-be look-ahead if using current engine: {latest_fy - pit_fy} fiscal years.")
        print(f"  Proceeding with point-in-time backtest.")
    else:
        print(f"\n  PRE-CHECK: FAIL ✗  (PIT engine not filtering correctly — STOP)")

    return pit_ok


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Universe prep
# ═══════════════════════════════════════════════════════════════════════════════

def prep_universe() -> list[str]:
    print("\n" + "=" * 70)
    print("  UNIVERSE PREP")
    print("=" * 70)

    df = pd.read_csv(UNIVERSE_CSV)
    n0 = len(df)
    print(f"\n  Snapshot:             {n0:,} tickers")

    # Second-pass sector retry for nulls
    null_mask = df["sector"].isna() | (df["sector"] == "")
    n_null = null_mask.sum()
    print(f"  Null-sector tickers:  {n_null:,} — retrying via yfinance...")

    import yfinance as yf

    def _fetch_sector(sym):
        try:
            info = yf.Ticker(sym).info
            return {"symbol": sym,
                    "sector":   info.get("sector", ""),
                    "industry": info.get("industry", "")}
        except Exception:
            return {"symbol": sym, "sector": "", "industry": ""}

    null_tickers = df[null_mask]["symbol"].tolist()
    updated: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=20) as pool:
        futs = {pool.submit(_fetch_sector, sym): sym for sym in null_tickers}
        done = 0
        for f in as_completed(futs):
            r = f.result()
            updated[r["symbol"]] = r
            done += 1
            if done % 100 == 0:
                print(f"    sector retry: {done}/{len(null_tickers)}", flush=True)

    for sym, r in updated.items():
        mask = df["symbol"] == sym
        df.loc[mask, "sector"]   = r["sector"]
        df.loc[mask, "industry"] = r["industry"]

    # After retry
    still_null = (df["sector"].isna() | (df["sector"] == "")).sum()
    resolved   = n_null - still_null
    print(f"  Resolved:             {resolved:,}  |  Still unknown: {still_null:,}")

    # Exclude Financial Services + Real Estate (known)
    df_excl = df[df["sector"].isin(SECTOR_EXCL)]
    df = df[~df["sector"].isin(SECTOR_EXCL)]
    print(f"\n  After Financial/RE exclusion:")
    print(f"    Removed: {len(df_excl):,}  |  Remaining: {len(df):,}")
    print(f"    Unknown-sector included (not positively identified as excl): {still_null:,}")

    universe = df["symbol"].dropna().str.upper().unique().tolist()
    print(f"\n  Final universe:       {len(universe):,} tickers (entering signal computation)")
    return universe


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Price data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_price_data(universe: list[str]) -> tuple[
    dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame
]:
    import yfinance as yf
    print("\n" + "=" * 70)
    print("  PRICE DATA LOADING")
    print("=" * 70)

    price_start = "2013-01-01"   # extra history for beta / momentum
    price_end   = "2025-08-01"

    # Download SPY and ^TNX first
    print(f"  Downloading SPY (market) and ^TNX (risk-free)...")
    ref = yf.download(["SPY", "^TNX"], start=price_start, end=price_end,
                      auto_adjust=True, progress=False)
    # yf.download returns MultiIndex (Price, Ticker) columns
    def _extract(ref_df, ticker):
        if isinstance(ref_df.columns, pd.MultiIndex):
            return ref_df["Close"][[ticker]].rename(columns={ticker: "Close"}).dropna()
        return ref_df[["Close"]].dropna()

    mkt_hist = _extract(ref, "SPY")
    tnx_hist = _extract(ref, "^TNX")
    print(f"  SPY: {len(mkt_hist)} days  |  ^TNX: {len(tnx_hist)} days")

    # Batch download universe in chunks of 200
    print(f"  Downloading {len(universe)} tickers (may take a few minutes)...")
    CHUNK = 200
    all_close: dict[str, pd.Series] = {}
    for i in range(0, len(universe), CHUNK):
        chunk = universe[i:i + CHUNK]
        try:
            raw = yf.download(chunk, start=price_start, end=price_end,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"]
            else:
                close = raw[["Close"]].rename(columns={"Close": chunk[0]})
            for sym in chunk:
                if sym in close.columns:
                    s = close[sym].dropna()
                    if len(s) > 60:
                        all_close[sym] = s
        except Exception as e:
            print(f"  [WARN] chunk {i}-{i+CHUNK}: {e}")
        if (i // CHUNK + 1) % 5 == 0:
            print(f"  {min(i + CHUNK, len(universe))}/{len(universe)} downloaded", flush=True)

    # Wrap each series as a small DataFrame with 'Close' column
    price_hists = {sym: s.to_frame("Close") for sym, s in all_close.items()}
    print(f"  Price histories loaded: {len(price_hists)}/{len(universe)} tickers")
    return price_hists, mkt_hist, tnx_hist


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Signal computation
# ═══════════════════════════════════════════════════════════════════════════════

def get_rebalance_dates() -> list[date]:
    if REBALANCE == "annual":
        return [date(y, REB_MONTH, REB_DAY) for y in range(START_YEAR, END_YEAR)]
    else:
        dates = []
        for y in range(START_YEAR, END_YEAR):
            for q in [3, 6, 9, 12]:
                last_day = 31 if q in (3, 12) else 30
                dates.append(date(y, q, last_day))
        return dates


def build_signal_panel(
    universe:    list[str],
    reb_dates:   list[date],
    price_hists: dict[str, pd.DataFrame],
    mkt_hist:    pd.DataFrame,
    tnx_hist:    pd.DataFrame,
) -> pd.DataFrame:

    if SIGNALS_CACHE.exists():
        print(f"\n  Loading signals from cache: {SIGNALS_CACHE}")
        return pd.read_parquet(SIGNALS_CACHE)

    n_tickers = len(universe)
    n_dates   = len(reb_dates)
    print(f"\n" + "=" * 70)
    print("  SIGNAL COMPUTATION")
    print("=" * 70)
    print(f"\n[COMPUTE ESTIMATE]")
    print(f"  Tickers in universe:  {n_tickers}")
    print(f"  Rebalance dates:      {n_dates}  ({reb_dates[0]} → {reb_dates[-1]})")
    print(f"  Max DCF computations: {n_tickers * n_dates:,} (deterministic, ~1ms each)")
    print(f"  EDGAR fact fetches:   {n_tickers} (cached on disk after first run)")
    print(f"  Estimate:             ~5-15 min first run, ~1-3 min with caches")
    print()

    rows: list[dict] = []
    n_edgar_ok = 0

    for i, ticker in enumerate(universe):
        facts = fetch_edgar_facts(ticker)
        if facts is None:
            continue
        n_edgar_ok += 1
        ph = price_hists.get(ticker)
        for reb_date in reb_dates:
            sig = compute_signals(ticker, facts, reb_date, ph, mkt_hist, tnx_hist)
            if sig is not None:
                rows.append(sig)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_tickers} tickers done  |  {len(rows)} signal rows so far", flush=True)

    print(f"\n  EDGAR facts loaded: {n_edgar_ok}/{n_tickers}")
    print(f"  Total signal rows:  {len(rows)}")

    panel = pd.DataFrame(rows)
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    panel.to_parquet(SIGNALS_CACHE, index=False)
    print(f"  Saved to cache: {SIGNALS_CACHE}")
    return panel


def add_forward_returns(panel: pd.DataFrame, price_hists: dict, reb_dates: list[date]) -> pd.DataFrame:
    """Add 1-period forward return (as_of → next rebalance date), net of 20bps round-trip costs."""
    fwd_map = {}
    for i, d in enumerate(reb_dates[:-1]):
        fwd_map[pd.Timestamp(d)] = date(reb_dates[i + 1].year, reb_dates[i + 1].month, reb_dates[i + 1].day)
    # Last date: +1 year
    last = reb_dates[-1]
    fwd_map[pd.Timestamp(last)] = date(last.year + 1, last.month, last.day)

    cost_one_way = COSTS_BPS / 10000 / 2  # split round-trip equally buy+sell

    def _fwd(row):
        fwd_date = fwd_map.get(row["as_of"])
        if fwd_date is None:
            return float("nan")
        ph = price_hists.get(row["ticker"])
        if ph is None:
            return float("nan")
        p0 = _price_at(ph, row["as_of"].date() if hasattr(row["as_of"], "date") else row["as_of"])
        p1 = _price_at(ph, fwd_date)
        if p0 is None or p1 is None or p0 <= 0:
            return float("nan")
        gross = (p1 - p0) / p0
        return gross - cost_one_way * 2   # buy + sell

    panel = panel.copy()
    panel["fwd_ret"] = panel.apply(_fwd, axis=1)
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Portfolio construction helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _winsorise(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    p_lo = series.quantile(lo)
    p_hi = series.quantile(hi)
    return series.clip(p_lo, p_hi)


def build_shared_universe_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Shared universe: only dates × tickers where DCF is applicable
    AND both B/P and EBITDA/EV are available AND fwd_ret is available.
    All signals winsorised cross-sectionally (1/99 pct) at each date.
    """
    df = panel.copy()
    df = df[df["dcf_applicable"] & df["fwd_ret"].notna()].copy()

    rows_out = []
    for as_of, grp in df.groupby("as_of"):
        grp = grp.dropna(subset=["dcf_vp", "bp", "ebitda_ev", "fwd_ret"])
        if len(grp) < N_QUINTILES * 3:   # need at least 15 stocks
            continue
        grp = grp.copy()
        for col in ["dcf_vp", "bp", "ebitda_ev", "size", "mom"]:
            if col in grp.columns:
                grp[col] = _winsorise(grp[col])
        rows_out.append(grp)

    if not rows_out:
        return pd.DataFrame()
    return pd.concat(rows_out, ignore_index=True)


def compute_quintile_returns(df: pd.DataFrame, signal: str) -> dict:
    """Equal-weight top-quintile and benchmark returns per date."""
    q5_rets, bm_rets = [], []
    for as_of, grp in df.groupby("as_of"):
        if signal not in grp.columns or grp[signal].isna().all():
            continue
        grp = grp.dropna(subset=[signal, "fwd_ret"])
        if len(grp) < N_QUINTILES:
            continue
        grp["q"] = pd.qcut(grp[signal], N_QUINTILES, labels=False, duplicates="drop")
        q5 = grp[grp["q"] == N_QUINTILES - 1]["fwd_ret"].mean()  # cheapest = highest signal
        bm = grp["fwd_ret"].mean()
        q5_rets.append(q5)
        bm_rets.append(bm)
    return {"q5": np.array(q5_rets), "bm": np.array(bm_rets)}


def perf_stats(returns: np.ndarray, label: str, is_annual: bool = True) -> dict:
    """Annualised return, Sharpe, max drawdown from a return series."""
    if len(returns) == 0:
        return {}
    ret    = float(np.mean(returns))
    std    = float(np.std(returns, ddof=1))
    sharpe = ret / std if std > 0 else float("nan")
    cum    = np.cumprod(1 + returns)
    roll_max = np.maximum.accumulate(cum)
    dd      = cum / roll_max - 1
    max_dd  = float(dd.min())
    return {
        "label":    label,
        "ann_ret":  ret,
        "sharpe":   sharpe,
        "max_dd":   max_dd,
        "n_periods": len(returns),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test1_horse_race(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 1 — HORSE RACE (top-quintile vs benchmark)")
    print("  *** Absolute Sharpes are UPPER BOUNDS (survivorship-biased universe) ***")
    print("=" * 70)

    signals = {"DCF V/P": "dcf_vp", "B/P": "bp", "EBITDA/EV": "ebitda_ev"}
    results = {}

    for label, col in signals.items():
        r = compute_quintile_returns(df, col)
        if len(r["q5"]) == 0:
            continue
        q5_stats = perf_stats(r["q5"], f"{label} Q5 (cheapest)")
        bm_stats = perf_stats(r["bm"], "Benchmark")
        results[label] = {"q5": q5_stats, "bm": bm_stats, "q5_rets": r["q5"]}

    if not results:
        print("  No results — insufficient data")
        return

    # Benchmark (same for all signals — use DCF's benchmark)
    bm_ref = list(results.values())[0]["bm"]

    print(f"\n  {'Signal':<14} {'Q5 Ann.Ret':>12} {'Q5 Sharpe':>11} {'Q5 MaxDD':>10}  {'vs BM':>8}")
    print(f"  {'-' * 58}")

    for label, r in results.items():
        q = r["q5"]
        bm = r["bm"]
        excess = q["ann_ret"] - bm["ann_ret"]
        print(
            f"  {label:<14}"
            f" {q['ann_ret']:>11.1%}"
            f" {q['sharpe']:>11.2f}"
            f" {q['max_dd']:>10.1%}"
            f"  {excess:>+8.1%}"
        )

    print(f"\n  {'Benchmark':<14} {bm_ref['ann_ret']:>11.1%} {bm_ref['sharpe']:>11.2f} {bm_ref['max_dd']:>10.1%}")

    # DCF-minus-best-multiple differential (survivorship-robust)
    if "DCF V/P" in results and len(results) > 1:
        dcf_ret = results["DCF V/P"]["q5"]["ann_ret"]
        best_mult_ret = max(
            results[k]["q5"]["ann_ret"]
            for k in results if k != "DCF V/P"
        )
        best_mult_label = max(
            (k for k in results if k != "DCF V/P"),
            key=lambda k: results[k]["q5"]["ann_ret"],
        )
        diff = dcf_ret - best_mult_ret
        print(f"\n  ═══════════════════════════════════════════════")
        print(f"  DCF V/P Q5 ann. return:             {dcf_ret:>+8.1%}")
        print(f"  Best multiple ({best_mult_label}) Q5 return:  {best_mult_ret:>+8.1%}")
        print(f"  DCF-minus-best-multiple differential: {diff:>+8.1%}  ← survivorship-robust")
        print(f"  ═══════════════════════════════════════════════")
        if diff > 0:
            print(f"  DCF V/P OUTPERFORMS best free multiple by {diff:.1%} per year")
        else:
            print(f"  DCF V/P UNDERPERFORMS best free multiple by {abs(diff):.1%} per year")


def test2_fama_macbeth(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 2 — FAMA-MACBETH (incremental power of DCF V/P)")
    print("  THE GATE: is DCF V/P significant after B/P and EBITDA/EV are included?")
    print("=" * 70)

    signal_cols = ["dcf_vp", "bp", "ebitda_ev", "size", "mom"]
    coef_rows: list[dict] = []

    for as_of, grp in df.groupby("as_of"):
        sub = grp.dropna(subset=signal_cols + ["fwd_ret"]).copy()
        if len(sub) < 20:
            continue
        # Cross-sectional z-score
        for col in signal_cols:
            mu  = sub[col].mean()
            std = sub[col].std()
            sub[col + "_z"] = (sub[col] - mu) / std if std > 1e-10 else 0.0

        X_cols = [c + "_z" for c in signal_cols]
        X = sub[X_cols].values
        y = sub["fwd_ret"].values
        try:
            res = np.linalg.lstsq(
                np.column_stack([np.ones(len(X)), X]), y, rcond=None
            )
            coefs = res[0]  # [intercept, dcf_vp, bp, ebitda_ev, size, mom]
            row = {"as_of": as_of, "intercept": coefs[0]}
            for j, col in enumerate(signal_cols):
                row[col] = coefs[j + 1]
            coef_rows.append(row)
        except Exception:
            continue

    if len(coef_rows) < 3:
        print("  Insufficient periods for Fama-MacBeth (need ≥3)")
        return

    coef_df = pd.DataFrame(coef_rows)
    T = len(coef_df)

    print(f"\n  Periods used: {T}  |  Each cross-section: z-scored signals, OLS on forward return")
    print(f"\n  {'Variable':<14} {'Mean coef':>12} {'t-stat':>9}  {'Verdict'}")
    print(f"  {'-' * 55}")

    for col in signal_cols:
        if col not in coef_df.columns:
            continue
        mean_c = coef_df[col].mean()
        se_c   = coef_df[col].std(ddof=1) / np.sqrt(T)
        t_stat = mean_c / se_c if se_c > 0 else float("nan")
        sig    = ("***" if abs(t_stat) > 2.576 else
                  "**"  if abs(t_stat) > 1.960 else
                  "*"   if abs(t_stat) > 1.645 else "")
        label_map = {
            "dcf_vp": "DCF V/P", "bp": "B/P", "ebitda_ev": "EBITDA/EV",
            "size": "SIZE", "mom": "MOM",
        }
        print(f"  {label_map.get(col, col):<14} {mean_c:>12.4f} {t_stat:>9.2f}  {sig}")

    # Explicit gate verdict
    dcf_mean = coef_df["dcf_vp"].mean() if "dcf_vp" in coef_df else float("nan")
    dcf_se   = coef_df["dcf_vp"].std(ddof=1) / np.sqrt(T) if "dcf_vp" in coef_df else float("nan")
    dcf_t    = dcf_mean / dcf_se if dcf_se > 0 else float("nan")

    print(f"\n  {'-' * 55}")
    print(f"  *** p<0.01   ** p<0.05   * p<0.10")
    print(f"\n  ══ GATE VERDICT ══")
    if not math.isnan(dcf_t):
        if dcf_mean > 0 and abs(dcf_t) >= 1.645:
            print(f"  DCF V/P coefficient: {dcf_mean:.4f}  t = {dcf_t:.2f}  → POSITIVE & SIGNIFICANT")
            print(f"  GATE: PASS ✓  DCF adds information beyond B/P + EBITDA/EV.")
            print(f"  The engine is doing real work, not just replicating a free multiple.")
        elif dcf_mean > 0:
            print(f"  DCF V/P coefficient: {dcf_mean:.4f}  t = {dcf_t:.2f}  → positive but INSIGNIFICANT")
            print(f"  GATE: MARGINAL — DCF direction correct but power weak (small sample?).")
            print(f"  Cannot confidently claim DCF adds beyond multiples at this sample size.")
        else:
            print(f"  DCF V/P coefficient: {dcf_mean:.4f}  t = {dcf_t:.2f}  → NON-POSITIVE")
            print(f"  GATE: FAIL ✗  DCF V/P adds nothing once B/P and EBITDA/EV are in the regression.")
            print(f"  XU (2007) WINS: the engine is a Ferrari doing a bicycle's job.")
            print(f"  A free EV/EBITDA ratio captures everything the DCF adds and more.")


def test3_double_sort(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 3 — DOUBLE-SORT (B/P tercile × DCF V/P tercile)")
    print("  Does DCF V/P separate returns WITHIN B/P buckets?")
    print("=" * 70)

    cell_rets: dict[tuple[int, int], list[float]] = {}
    n_dates   = 0

    for as_of, grp in df.groupby("as_of"):
        sub = grp.dropna(subset=["bp", "dcf_vp", "fwd_ret"])
        if len(sub) < 9:
            continue
        try:
            sub = sub.copy()
            sub["bp_t"]     = pd.qcut(sub["bp"],     3, labels=[1, 2, 3], duplicates="drop")
            sub["dcf_vp_t"] = pd.qcut(sub["dcf_vp"], 3, labels=[1, 2, 3], duplicates="drop")
        except Exception:
            continue
        for (bt, dt), cell in sub.groupby(["bp_t", "dcf_vp_t"]):
            key = (int(bt), int(dt))
            cell_rets.setdefault(key, []).extend(cell["fwd_ret"].tolist())
        n_dates += 1

    if n_dates == 0:
        print("  No dates with sufficient data for double-sort")
        return

    print(f"\n  Mean annual return by (B/P tercile, DCF V/P tercile)")
    print(f"  B/P 1=expensive→3=cheap  |  DCF V/P 1=expensive→3=cheap")
    print(f"  Cell = equal-weight stocks, averaged across {n_dates} rebalance dates")
    print(f"\n  {'':10}", end="")
    for dt in [1, 2, 3]:
        print(f"  DCF V/P T{dt}", end="")
    print(f"  {'Row avg':>10}")
    print(f"  {'-' * 55}")

    for bt in [1, 2, 3]:
        row_vals = []
        print(f"  B/P T{bt}    ", end="")
        for dt in [1, 2, 3]:
            vals = cell_rets.get((bt, dt), [])
            m    = float(np.mean(vals)) if vals else float("nan")
            row_vals.append(m)
            cell_str = f"{m:.1%}" if not math.isnan(m) else " n/a"
            print(f"  {cell_str:>10}", end="")
        row_avg = float(np.nanmean(row_vals))
        print(f"  {row_avg:>10.1%}")

    # Column averages
    print(f"  {'Col avg':10}", end="")
    for dt in [1, 2, 3]:
        col_vals = [np.mean(cell_rets[(bt, dt)]) for bt in [1, 2, 3]
                    if (bt, dt) in cell_rets and cell_rets[(bt, dt)]]
        print(f"  {np.mean(col_vals):>10.1%}" if col_vals else f"  {'n/a':>10}", end="")
    print()

    # Interpretation
    # Does moving from DCF T1 → T3 increase returns within each B/P bucket?
    monotone_count = 0
    for bt in [1, 2, 3]:
        rets_by_dt = [np.mean(cell_rets.get((bt, dt), [float("nan")])) for dt in [1, 2, 3]]
        if all(not math.isnan(x) for x in rets_by_dt):
            if rets_by_dt[0] < rets_by_dt[1] < rets_by_dt[2]:
                monotone_count += 1

    print(f"\n  DCF V/P monotone increasing within B/P buckets: {monotone_count}/3")
    if monotone_count >= 2:
        print("  → DCF V/P separates returns within multiple buckets ✓")
    elif monotone_count == 1:
        print("  → Mixed evidence — DCF V/P partially separates within multiples")
    else:
        print("  → DCF V/P does NOT separate returns within B/P buckets")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("  HYPOTHESIS 1: DCF ENGINE vs FREE MULTIPLE")
    print("  Universe: US small-cap, non-financial, DCF-applicable stocks")
    print("  Rebalance: ANNUAL (June 30)  |  Costs: 20bps round-trip")
    print("  SURVIVORSHIP NOTE: Absolute Sharpes are upper bounds.")
    print("  DCF-minus-multiple differential is survivorship-robust.")
    print("  ONE TRIAL — record for Deflated Sharpe.")
    print("#" * 70)

    # ── PRE-CHECK ─────────────────────────────────────────────────────────────
    pit_ok = run_precheck()
    if not pit_ok:
        print("\n[STOP] Point-in-time pre-check FAILED. Backtest would be look-ahead biased.")
        print("       Fix the PIT data layer before proceeding.")
        sys.exit(1)

    # ── UNIVERSE PREP ─────────────────────────────────────────────────────────
    universe = prep_universe()

    # ── PRICE DATA ────────────────────────────────────────────────────────────
    price_hists, mkt_hist, tnx_hist = load_price_data(universe)

    # ── REBALANCE DATES ───────────────────────────────────────────────────────
    reb_dates = get_rebalance_dates()
    print(f"\n  Rebalance dates ({len(reb_dates)}): {reb_dates[0]} → {reb_dates[-1]}")

    # ── SIGNAL PANEL ──────────────────────────────────────────────────────────
    panel = build_signal_panel(universe, reb_dates, price_hists, mkt_hist, tnx_hist)
    if panel.empty:
        print("[ERROR] No signals computed. Check EDGAR access and price data.")
        sys.exit(1)

    # ── FORWARD RETURNS ───────────────────────────────────────────────────────
    print("\n  Adding forward returns...")
    panel = add_forward_returns(panel, price_hists, reb_dates)

    # ── SHARED UNIVERSE PANEL ─────────────────────────────────────────────────
    shared = build_shared_universe_panel(panel)
    if shared.empty:
        print("[ERROR] Shared universe is empty after DCF-applicable filter.")
        sys.exit(1)

    n_tickers_per_date = shared.groupby("as_of")["ticker"].nunique()
    print(f"\n" + "=" * 70)
    print("  UNIVERSE FUNNEL")
    print("=" * 70)
    print(f"  CSV snapshot:               {len(pd.read_csv(UNIVERSE_CSV)):>6,}")
    print(f"  After sector exclusion:     {len(universe):>6,}")
    print(f"  With price data:            {len(price_hists):>6,}")
    print(f"  Signal rows computed:       {len(panel):>6,}")
    print(f"  DCF-applicable rows:        {panel['dcf_applicable'].sum():>6,}")
    print(f"  Shared universe rows:       {len(shared):>6,}")
    print(f"  Tickers/date (avg):         {n_tickers_per_date.mean():>6.0f}")
    print(f"  Tickers/date (min-max):     {n_tickers_per_date.min()}-{n_tickers_per_date.max()}")

    # ── TESTS ─────────────────────────────────────────────────────────────────
    test1_horse_race(shared)
    test2_fama_macbeth(shared)
    test3_double_sort(shared)

    print("\n" + "#" * 70)
    print("  RUN COMPLETE")
    print("#" * 70)
    print(f"  Signal cache: {SIGNALS_CACHE}")
    print(f"  To rerun with fresh signals: delete {SIGNALS_CACHE}")


if __name__ == "__main__":
    main()
