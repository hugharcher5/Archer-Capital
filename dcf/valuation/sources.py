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

    # ── Extra filing fields (populated by fetch_edgar only) ──────────────────
    # Used to build the model from official filings first; other sources leave
    # them empty and the model falls back to Yahoo for each one.
    pretax_income:    pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    tax_provision:    pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    interest_expense: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sbc:              pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    current_assets:      float = float("nan")   # fiscal-year-end balance sheet
    current_liabilities: float = float("nan")
    current_debt:        float = float("nan")
    cash_fy:             float = float("nan")
    balance_date: object = None    # date of the balance sheet behind cash and debt
    shares_date:  object = None    # period end of the share count
    tags_used: dict = field(default_factory=dict)   # field → XBRL tag behind the latest value


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
    Fetch financial data from SEC EDGAR XBRL company facts (the companies' own
    10-K and 10-Q filings).  Requires no API key.  Raises RuntimeError with a
    clear message on failure.

    Flow items (revenue, EBIT, tax, interest, D&A, CapEx, SBC) are full fiscal
    years from 10-K filings.  Bridge items (cash, debt, share count) are read
    from the most recent 10-K or 10-Q balance sheet so they are as current as
    the filings allow.  Working-capital items come from the latest fiscal-year
    balance sheet so they line up with annual revenue.
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
    if not us_gaap:
        # Foreign private issuers filing 20-F under IFRS tag with ifrs-full, which
        # uses different concepts; every field then falls back to Yahoo Finance.
        raise RuntimeError("EDGAR: no US-GAAP data (company likely files under IFRS on Form 20-F)")

    _ANNUAL_FORMS  = {"10-K", "10-K/A", "10-KT"}
    _CURRENT_FORMS = _ANNUAL_FORMS | {"10-Q", "10-Q/A"}

    def _rows(tag: str, unit: str = "USD") -> list[dict]:
        return us_gaap.get(tag, {}).get("units", {}).get(unit, [])

    def _is_full_year(row: dict) -> bool:
        try:
            days = (pd.Timestamp(row["end"]) - pd.Timestamp(row["start"])).days
        except (KeyError, ValueError, TypeError):
            return False
        return 350 <= days <= 380

    def _tag_series(tag: str, abs_val: bool, unit: str = "USD") -> pd.Series:
        """Full-fiscal-year values for one tag, one value per period end.
        Later filings win, so restated figures replace the originals."""
        pts: dict[pd.Timestamp, tuple[str, float]] = {}
        for row in _rows(tag, unit):
            if row.get("form") not in _ANNUAL_FORMS or not _is_full_year(row):
                continue
            try:
                end = pd.Timestamp(row["end"])
                val = float(row["val"])
            except (KeyError, ValueError, TypeError):
                continue
            filed = row.get("filed", "")
            if end not in pts or filed >= pts[end][0]:
                pts[end] = (filed, abs(val) if abs_val else val)
        if not pts:
            return pd.Series(dtype=float)
        return pd.Series({k: v[1] for k, v in pts.items()}).sort_index()

    def _flow(tags: list[str], abs_val: bool = False, combine: str = "first") -> tuple[pd.Series, str]:
        """Annual series built across candidate tags.

        Companies switch tags over time (e.g. SalesRevenueNet → Revenues →
        RevenueFromContractWithCustomer...), so each fiscal year is filled
        from whichever candidate reports it: the first in priority order
        ('first') or the largest ('max', used for revenue, where the total
        line is the largest of the revenue concepts).
        """
        per_tag = {t: _tag_series(t, abs_val) for t in tags}
        per_tag = {t: s for t, s in per_tag.items() if not s.empty}
        if not per_tag:
            return pd.Series(dtype=float), ""
        frame = pd.DataFrame(per_tag)
        if combine == "max":
            out = frame.max(axis=1)
        else:
            out = frame.bfill(axis=1).iloc[:, 0]
        out = out.dropna().sort_index()
        # Name the tag that supplied the latest year, for the provenance table.
        latest = frame.loc[out.index[-1]].dropna()
        used = latest.idxmax() if combine == "max" else latest.index[0]
        return out, used

    def _instant_at(tags: list[str], date: pd.Timestamp | None, forms=_CURRENT_FORMS,
                    unit: str = "USD") -> tuple[float, str, pd.Timestamp | None]:
        """Balance-sheet value for the first tag reported at `date`
        (or, when date is None, the most recent date any tag reports)."""
        best: tuple[pd.Timestamp, str, float] | None = None
        for tag in tags:
            latest: dict[pd.Timestamp, tuple[str, float]] = {}
            for row in _rows(tag, unit):
                if row.get("form") not in forms or "start" in row:
                    continue
                try:
                    end = pd.Timestamp(row["end"])
                    val = float(row["val"])
                except (KeyError, ValueError, TypeError):
                    continue
                filed = row.get("filed", "")
                if end not in latest or filed >= latest[end][0]:
                    latest[end] = (filed, val)
            if not latest:
                continue
            if date is not None:
                if date in latest:
                    return latest[date][1], tag, date
                continue
            end = max(latest)
            if best is None or end > best[0]:
                best = (end, tag, latest[end][1])
        if best is None:
            return float("nan"), "", None
        return best[2], best[1], best[0]

    missing: list[str] = []
    tags_used: dict[str, str] = {}

    # ── Income statement (10-K, full fiscal years) ───────────────────────────
    revenue, tags_used["revenue"] = _flow([
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ], combine="max")
    ebit, tags_used["ebit"] = _flow(["OperatingIncomeLoss"])
    pretax_income, tags_used["pretax_income"] = _flow([
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ])
    tax_provision, tags_used["tax_provision"] = _flow(["IncomeTaxExpenseBenefit"])
    interest_expense, tags_used["interest_expense"] = _flow([
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ], abs_val=True)

    # ── Cash flow statement (10-K) ────────────────────────────────────────────
    # D&A: only a single, comprehensive reported line is trusted.  Filers that
    # split D&A across several tags are left empty here (Yahoo's cash-flow
    # total is used instead), because summing the pieces misses untagged items.
    dep_amort, tags_used["dep_amort"] = _flow([
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ], abs_val=True)
    capex, tags_used["capex"] = _flow([
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ], abs_val=True)
    sbc, tags_used["sbc"] = _flow(["ShareBasedCompensation",
                                   "AllocatedShareBasedCompensationExpense"], abs_val=True)

    for name, s in [("revenue", revenue), ("ebit", ebit), ("pretax_income", pretax_income),
                    ("tax_provision", tax_provision), ("interest_expense", interest_expense),
                    ("dep_amort", dep_amort), ("capex", capex), ("sbc", sbc)]:
        if s.empty:
            missing.append(name)

    # ── Latest balance sheet (10-K or 10-Q): cash, debt ──────────────────────
    _, _, bs_date = _instant_at(["Assets"], None)
    cash, tags_used["cash"], _ = _instant_at(["CashAndCashEquivalentsAtCarryingValue",
                                              "CashCashEquivalentsAndShortTermInvestments"], bs_date)

    # Debt = bonds/notes/loans + commercial paper + finance leases.  Operating
    # lease liabilities are excluded (their cost is already inside EBIT).
    def _bs(tags):
        v, _, _ = _instant_at(tags, bs_date)
        return 0.0 if math.isnan(v) else v
    ltd_noncurrent = _bs(["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"])
    if ltd_noncurrent:
        current = _bs(["DebtCurrent"]) or (
            _bs(["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"])
            + _bs(["ShortTermBorrowings", "CommercialPaper"]))
        total_debt = ltd_noncurrent + current
    else:
        # LongTermDebt already includes the current portion.
        total_debt = _bs(["LongTermDebt"]) + _bs(["ShortTermBorrowings", "CommercialPaper"])
    total_debt += _bs(["FinanceLeaseLiabilityNoncurrent"]) + _bs(["FinanceLeaseLiabilityCurrent"])
    if bs_date is None:
        total_debt = float("nan")
    tags_used["total_debt"] = "LongTermDebt + current debt + finance leases"

    # Share count: diluted weighted average from the most recent 10-Q or 10-K.
    diluted_shares, shares_date = float("nan"), None
    tags_used["diluted_shares"] = ""
    pts = []
    for row in _rows("WeightedAverageNumberOfDilutedSharesOutstanding", "shares"):
        if row.get("form") in _CURRENT_FORMS and row.get("end"):
            pts.append((row["end"], row.get("filed", ""), float(row["val"])))
    if pts:
        end, _, diluted_shares = max(pts)
        shares_date = pd.Timestamp(end)
        tags_used["diluted_shares"] = "WeightedAverageNumberOfDilutedSharesOutstanding"

    # ── Fiscal-year-end balance sheet: operating working capital ─────────────
    fy_date = revenue.index[-1] if not revenue.empty else None
    current_assets, _, _ = _instant_at(["AssetsCurrent"], fy_date, forms=_ANNUAL_FORMS)
    current_liabilities, _, _ = _instant_at(["LiabilitiesCurrent"], fy_date, forms=_ANNUAL_FORMS)
    cash_fy, _, _ = _instant_at(["CashAndCashEquivalentsAtCarryingValue"], fy_date, forms=_ANNUAL_FORMS)
    v, _, _ = _instant_at(["DebtCurrent"], fy_date, forms=_ANNUAL_FORMS)
    if math.isnan(v):
        a, _, _ = _instant_at(["LongTermDebtCurrent"], fy_date, forms=_ANNUAL_FORMS)
        b, _, _ = _instant_at(["ShortTermBorrowings", "CommercialPaper"], fy_date, forms=_ANNUAL_FORMS)
        v = (0.0 if math.isnan(a) else a) + (0.0 if math.isnan(b) else b)
    current_debt_fy = v

    for name, val in [("cash", cash), ("total_debt", total_debt),
                      ("diluted_shares", diluted_shares),
                      ("current_assets", current_assets),
                      ("current_liabilities", current_liabilities)]:
        if math.isnan(val):
            missing.append(name)

    # Effective tax rate: same definition as drivers.py (mean of yearly rates).
    tax_rate = float("nan")
    if not tax_provision.empty and not pretax_income.empty:
        pt   = pretax_income.reindex(tax_provision.index)
        mask = (pt > 0) & tax_provision.notna()
        if mask.any():
            tax_rate = float((tax_provision[mask] / pt[mask]).clip(0.0, 0.6).mean())
    if math.isnan(tax_rate):
        missing.append("tax_rate")

    if missing:
        print(f"  [EDGAR] not found in filings: {', '.join(missing)}")

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
        pretax_income=pretax_income,
        tax_provision=tax_provision,
        interest_expense=interest_expense,
        sbc=sbc,
        current_assets=current_assets,
        current_liabilities=current_liabilities,
        current_debt=current_debt_fy,
        cash_fy=cash_fy,
        balance_date=bs_date,
        shares_date=shares_date,
        tags_used=tags_used,
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
