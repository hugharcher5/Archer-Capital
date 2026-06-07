"""
Normalized financial data schema and per-source fetchers.

Each fetcher returns a SourceData with the same field names so
reconcile.py can compare them without knowing the origin.
"""

from __future__ import annotations
import math
import os
import json
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
        total_debt=raw.total_debt,
        cash=raw.cash,
        tax_rate=tax_rate,
        missing_fields=missing,
    )


# ── Financial Modeling Prep fetcher ───────────────────────────────────────────

_FMP_BASE = "https://financialmodelingprep.com/api/v3"


def _fmp_get(endpoint: str, api_key: str) -> list[dict]:
    url = f"{_FMP_BASE}/{endpoint}?limit=5&apikey={api_key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "archer-capital/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"FMP HTTP {e.code} for {endpoint}") from e
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
    inc = _fmp_get(f"income-statement/{ticker}", api_key)
    bs  = _fmp_get(f"balance-sheet-statement/{ticker}", api_key)
    cf  = _fmp_get(f"cash-flow-statement/{ticker}", api_key)
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
