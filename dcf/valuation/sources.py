"""
Normalized financial data schema and per-source fetchers.

Each fetcher returns a SourceData with the same field names so
reconcile.py can compare them without knowing the origin.
"""

from __future__ import annotations
import math
import os
import json
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


# ── Normalized schema ─────────────────────────────────────────────────────────

@dataclass
class SourceData:
    source_name: str
    currency:    str

    # Annual series — pd.Series, index = fiscal-year-end Timestamp, ascending
    revenue:   pd.Series    # total revenue, in reporting currency
    ebit:      pd.Series    # operating income (EBIT)
    dep_amort: pd.Series    # D&A, positive
    capex:     pd.Series    # capital expenditure, positive

    # Point estimates — most recent fiscal year / balance sheet
    diluted_shares: float   # total diluted shares (actual count, not millions)
    total_debt:     float   # total interest-bearing debt, in reporting currency
    cash:           float   # cash & equivalents, in reporting currency
    tax_rate:       float   # effective rate, decimal (e.g. 0.21)

    missing_fields: list = field(default_factory=list)


# ── Yahoo Finance fetcher (wraps valuation/data.py) ───────────────────────────

def fetch_yahoo(ticker: str) -> SourceData:
    """Delegate to fetch_raw() and re-map fields into SourceData."""
    from .data import fetch_raw

    raw     = fetch_raw(ticker)
    missing = list(raw.missing_fields)

    # Normalize total_debt: subtract operating lease liabilities.
    # Convention: net debt = bonds/notes + finance-lease liabilities only.
    # Rent expense is already in EBIT, so operating leases must not also be
    # counted as debt (that would double-count the cost).
    # yfinance bundles operating leases into Total Debt as "Capital Lease
    # Obligations"; raw.op_lease_liab holds that amount so we can strip it.
    total_debt_normalized = max(0.0, raw.total_debt - raw.op_lease_liab)

    # Effective tax rate from historical series (same logic as drivers.py)
    tax_rate = float('nan')
    if not raw.tax_provision.empty and not raw.pretax_income.empty:
        pt   = raw.pretax_income.reindex(raw.tax_provision.index)
        mask = (pt > 0) & raw.tax_provision.notna()
        if mask.any():
            tax_rate = float((raw.tax_provision[mask] / pt[mask]).clip(0.0, 0.6).mean())
    if math.isnan(tax_rate):
        missing.append('tax_rate')

    return SourceData(
        source_name='Yahoo',
        currency=raw.currency,
        revenue=raw.revenue,
        ebit=raw.ebit,
        dep_amort=raw.da,
        capex=raw.capex,
        diluted_shares=raw.diluted_shares,
        total_debt=total_debt_normalized,
        cash=raw.cash,
        tax_rate=tax_rate,
        missing_fields=missing,
    )


# ── Financial Modeling Prep fetcher ───────────────────────────────────────────
# NOTE: FMP retired all /api/v3/* endpoints for non-legacy accounts on 2025-08-31
# (they now 403 with "Legacy Endpoint ... no longer supported"). This fetcher
# targets the current /stable/ API, which uses `?symbol=` instead of a path
# segment but otherwise returns the same field names for the statements we use.

_FMP_BASE = "https://financialmodelingprep.com/stable"


def _fmp_get(endpoint: str, symbol: str, api_key: str) -> list[dict]:
    q = urllib.parse.urlencode({"symbol": symbol, "limit": 5, "apikey": api_key})
    url = f"{_FMP_BASE}/{endpoint}?{q}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "archer-capital/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode(errors="replace").strip()
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"FMP HTTP {e.code} for {endpoint}{suffix}") from e
    except Exception as e:
        raise RuntimeError(f"FMP request failed ({endpoint}): {e}") from e

    if isinstance(data, dict):
        msg = data.get("Error Message") or data.get("message") or str(data)
        raise RuntimeError(f"FMP API error: {msg}")
    if not isinstance(data, list):
        raise RuntimeError(f"FMP unexpected response type for {endpoint}")
    return data


def _to_series(rows: list[dict], key: str, abs_val: bool = False) -> pd.Series:
    """Parse a list of FMP statement rows into an ascending date-indexed Series."""
    pts: dict[pd.Timestamp, float] = {}
    for row in rows:
        date = row.get("date") or row.get("fillingDate")
        val  = row.get(key)
        if not date or val is None:
            continue
        try:
            pts[pd.Timestamp(date)] = abs(float(val)) if abs_val else float(val)
        except (ValueError, TypeError):
            pass
    if not pts:
        return pd.Series(dtype=float)
    return pd.Series(pts).sort_index()   # ascending: oldest first


def _scalar_first(rows: list[dict], *keys: str) -> float:
    """Return the first non-None numeric value across keys from the most recent row."""
    for row in (rows[:1] if rows else []):
        for k in keys:
            v = row.get(k)
            if v is not None:
                try:
                    f = float(v)
                    if math.isfinite(f):
                        return f
                except (TypeError, ValueError):
                    pass
    return float('nan')


def fetch_edgar(ticker: str) -> SourceData:
    """
    Fetch annual (10-K / FY) financial data from SEC EDGAR XBRL company facts.
    Requires no API key.  Raises RuntimeError with a clear message on failure.
    """
    HEADERS = {"User-Agent": "Archer Capital research@archercapital.dev"}

    def _get(url: str):
        import gzip as _gzip
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
                enc = r.headers.get("Content-Encoding", "")
                if enc == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = _gzip.decompress(raw)
                return json.loads(raw.decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"EDGAR HTTP {e.code} for {url}") from e
        except Exception as e:
            raise RuntimeError(f"EDGAR request failed: {e}") from e

    # ── 1. CIK lookup ─────────────────────────────────────────────────────────
    tickers_url = "https://www.sec.gov/files/company_tickers.json"
    tickers_map = _get(tickers_url)
    cik = None
    ticker_upper = ticker.upper()
    for entry in tickers_map.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            cik = str(entry["cik_str"]).zfill(10)
            break
    if cik is None:
        raise RuntimeError(f"EDGAR: ticker '{ticker}' not found in company_tickers.json")

    print(f"  [EDGAR] CIK for {ticker_upper}: {cik}", end=" ", flush=True)

    # ── 2. Company facts ──────────────────────────────────────────────────────
    facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    facts = _get(facts_url)
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    print("✓")

    def _annual_series(tag_candidates: list[str], abs_val: bool = False) -> tuple[pd.Series, list[str]]:
        """Try each tag in order; return first non-empty annual Series + list of tried tags."""
        tried: list[str] = []
        for tag in tag_candidates:
            tried.append(tag)
            node = us_gaap.get(tag, {})
            units = node.get("units", {})
            rows = units.get("USD") or units.get("shares") or []
            if not rows:
                continue
            # 10-K form guarantees annual; fp filter unnecessary and sometimes absent
            fy_rows = [r for r in rows if r.get("form") == "10-K"]
            if not fy_rows:
                continue
            pts: dict[pd.Timestamp, float] = {}
            for row in fy_rows:
                end = row.get("end")
                val = row.get("val")
                if end is None or val is None:
                    continue
                try:
                    v = abs(float(val)) if abs_val else float(val)
                    pts[pd.Timestamp(end)] = v
                except (ValueError, TypeError):
                    pass
            if not pts:
                continue
            s = pd.Series(pts).sort_index()
            # deduplicate: keep most recent filing per fiscal-year-end
            s = s[~s.index.duplicated(keep="last")]
            return s, tried
        return pd.Series(dtype=float), tried

    def _annual_scalar(tag_candidates: list[str]) -> tuple[float, list[str]]:
        """Most recent FY 10-K value for the first tag that has data."""
        s, tried = _annual_series(tag_candidates)
        if s.empty:
            return float("nan"), tried
        return float(s.iloc[-1]), tried

    missing: list[str] = []

    # ── Series fields ─────────────────────────────────────────────────────────
    revenue, t = _annual_series([
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ])
    if revenue.empty:
        missing.append("revenue")
        print(f"  [EDGAR] revenue: no data (tried {t})")

    ebit, t = _annual_series(["OperatingIncomeLoss"])
    if ebit.empty:
        missing.append("ebit")
        print(f"  [EDGAR] ebit: no data (tried {t})")

    # D&A strategy:
    # 1. Comprehensive tag (ideal — covers all depreciation + amortization)
    # 2. Sum of dep component + intangible amortization (handles MSFT-style split filers)
    # 3. Dep-only fallback
    dep_amort, t = _annual_series([
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
    ], abs_val=True)

    if dep_amort.empty:
        dep_part,  _ = _annual_series(["DepreciationAndAmortization",
                                        "OtherDepreciationAndAmortization",
                                        "Depreciation"], abs_val=True)
        amort_part, _ = _annual_series(["AmortizationOfIntangibleAssets",
                                         "AmortizationOfAcquiredIntangibles"],
                                        abs_val=True)
        lease_amort, _ = _annual_series(["FinanceLeaseRightOfUseAssetAmortization",
                                          "OperatingLeaseRightOfUseAssetAmortization"],
                                         abs_val=True)
        if not dep_part.empty:
            combined = dep_part.copy()
            if not amort_part.empty:
                idx = combined.index.intersection(amort_part.index)
                if len(idx):
                    combined = combined.reindex(idx) + amort_part.reindex(idx)
            if not lease_amort.empty:
                idx = combined.index.intersection(lease_amort.index)
                if len(idx):
                    combined = combined.reindex(idx) + lease_amort.reindex(idx)
            dep_amort = combined.sort_index()
            t.append("components-summed")

    if dep_amort.empty:
        missing.append("dep_amort")
        print(f"  [EDGAR] dep_amort: no data (tried {t})")

    capex, t = _annual_series(
        ["PaymentsToAcquirePropertyPlantAndEquipment"], abs_val=True
    )
    if capex.empty:
        missing.append("capex")
        print(f"  [EDGAR] capex: no data (tried {t})")

    # ── Scalar fields ─────────────────────────────────────────────────────────
    diluted_shares, t = _annual_scalar(["WeightedAverageNumberOfDilutedSharesOutstanding"])
    if math.isnan(diluted_shares):
        missing.append("diluted_shares")
        print(f"  [EDGAR] diluted_shares: no data (tried {t})")

    cash, t = _annual_scalar(["CashAndCashEquivalentsAtCarryingValue"])
    if math.isnan(cash):
        missing.append("cash")
        print(f"  [EDGAR] cash: no data (tried {t})")

    # Total debt = long-term + current debt + finance lease obligations
    ltd, t1    = _annual_scalar(["LongTermDebtNoncurrent", "LongTermDebt"])
    cur, t2    = _annual_scalar(["LongTermDebtCurrent", "ShortTermBorrowings",
                                  "DebtCurrent"])
    fl_lt, t3  = _annual_scalar(["FinanceLeaseLiabilityNoncurrent"])
    fl_cur, t4 = _annual_scalar(["FinanceLeaseLiabilityCurrent"])
    components = [x for x in [ltd, cur, fl_lt, fl_cur] if not math.isnan(x)]
    total_debt = sum(components) if components else float("nan")
    if math.isnan(total_debt):
        missing.append("total_debt")
        print(f"  [EDGAR] total_debt: no data (tried {t1 + t2 + t3 + t4})")

    # Effective tax rate: IncomeTaxExpenseBenefit / IncomeLossFromContinuingOps...
    tax_series, _  = _annual_series(["IncomeTaxExpenseBenefit"])
    pretax_series, _ = _annual_series([
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
    ])
    tax_rate = float("nan")
    if not tax_series.empty and not pretax_series.empty:
        pt   = pretax_series.reindex(tax_series.index)
        mask = (pt > 0) & tax_series.notna()
        if mask.any():
            tax_rate = float((tax_series[mask] / pt[mask]).clip(0.0, 0.6).mean())
    if math.isnan(tax_rate):
        missing.append("tax_rate")

    return SourceData(
        source_name="EDGAR",
        currency="USD",
        revenue=revenue,
        ebit=ebit,
        dep_amort=dep_amort,
        capex=capex,
        diluted_shares=diluted_shares,
        total_debt=total_debt,
        cash=cash,
        tax_rate=tax_rate,
        missing_fields=missing,
    )


def fetch_fmp(ticker: str) -> SourceData:
    """
    Fetch income statement, balance sheet, and cash flow from FMP.
    Requires FMP_API_KEY in .env.  Raises RuntimeError with a clear message on failure.
    """
    api_key = os.getenv("FMP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "FMP_API_KEY not set — add FMP_API_KEY=<key> to your .env file.  "
            "Free keys available at https://financialmodelingprep.com/developer/docs"
        )

    print(f"  [FMP] Fetching {ticker}…", end=" ", flush=True)
    inc = _fmp_get("income-statement", ticker, api_key)
    bs  = _fmp_get("balance-sheet-statement", ticker, api_key)
    cf  = _fmp_get("cash-flow-statement", ticker, api_key)
    print(f"got {len(inc)} income / {len(bs)} balance / {len(cf)} cashflow rows")

    if not inc:
        raise RuntimeError(
            f"FMP returned no income statement rows for '{ticker}'. "
            "Check the ticker symbol and that your API plan covers this endpoint."
        )

    missing: list[str] = []
    currency = (inc[0].get("reportedCurrency") or "USD").upper()

    # ── Series fields ─────────────────────────────────────────────────────────
    revenue   = _to_series(inc, "revenue")
    ebit      = _to_series(inc, "operatingIncome")   # FMP's operatingIncome = EBIT
    dep_amort = _to_series(cf,  "depreciationAndAmortization", abs_val=True)
    capex     = _to_series(cf,  "capitalExpenditure",           abs_val=True)

    for name, s in [("revenue", revenue), ("ebit", ebit),
                    ("dep_amort", dep_amort), ("capex", capex)]:
        if s.empty:
            missing.append(name)

    # ── Balance sheet scalars ─────────────────────────────────────────────────
    cash = _scalar_first(bs, "cashAndCashEquivalents",
                              "cashAndShortTermInvestments")

    total_debt = _scalar_first(bs, "totalDebt")
    if math.isnan(total_debt):
        ltd      = _scalar_first(bs, "longTermDebt",
                                     "longTermDebtAndCapitalLeaseObligation") or 0.0
        cur_debt = _scalar_first(bs, "shortTermDebt",
                                     "shortTermBorrowings",
                                     "currentPortionOfLongTermDebt") or 0.0
        total_debt = ltd + cur_debt

    # Diluted shares from most recent income statement row
    diluted_shares = _scalar_first(inc, "weightedAverageShsOutDil",
                                        "weightedAverageShsOut")

    # ── Effective tax rate across all available years ─────────────────────────
    rates = []
    for row in inc:
        pretax = row.get("incomeBeforeTax")
        tax    = row.get("incomeTaxExpense")
        if pretax and tax:
            try:
                pt, tx = float(pretax), float(tax)
                if pt > 0:
                    r = tx / pt
                    if 0.0 <= r <= 0.6:
                        rates.append(r)
            except (TypeError, ValueError):
                pass
    tax_rate = float(np.mean(rates)) if rates else float('nan')

    for name, val in [("cash", cash), ("total_debt", total_debt),
                      ("diluted_shares", diluted_shares), ("tax_rate", tax_rate)]:
        if math.isnan(val):
            missing.append(name)

    return SourceData(
        source_name="FMP",
        currency=currency,
        revenue=revenue,
        ebit=ebit,
        dep_amort=dep_amort,
        capex=capex,
        diluted_shares=diluted_shares,
        total_debt=total_debt,
        cash=cash,
        tax_rate=tax_rate,
        missing_fields=missing,
    )
