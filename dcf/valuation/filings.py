"""
Build the model's inputs from official SEC filings first, Yahoo Finance second.

fetch_raw() reads everything from Yahoo Finance.  apply_filing_data() then
replaces each field with the SEC EDGAR (10-K / 10-Q) figure wherever the filing
data is present, current and consistent, and records where every input came
from and why, so the UI can show it.

Fields fall back to Yahoo when:
  * the company does not file with the SEC under US GAAP (foreign 20-F filers),
  * the company does not tag that line in its XBRL, or stopped tagging it,
  * the tagged figure is inconsistent with the reported total (a sign that the
    company used an unusual tag or split the line across several tags).
Market data (price, market cap, beta, exchange rate) is never in filings.
"""

from __future__ import annotations
import math

import pandas as pd

YEARS: int = 5                 # annual statements used (→ 4 years of growth)
DATE_TOLERANCE_DAYS: int = 20  # fiscal-year ends from two sources count as the same year

# Largest gap (vs Yahoo, latest year) before a filing figure is treated as a
# tagging problem rather than a genuine number.  Revenue definitions agree
# closely across providers, so its tolerance is tight.
_MAX_GAP = {"revenue": 0.15}
_DEFAULT_MAX_GAP = 0.50

EDGAR = "SEC EDGAR"
YAHOO = "Yahoo Finance"

# Display labels for the provenance table, in display order.
FIELD_LABELS: dict[str, str] = {
    "revenue":          "Revenue",
    "ebit":             "Operating income (EBIT)",
    "pretax_income":    "Pre-tax income",
    "tax_provision":    "Income tax expense",
    "interest_expense": "Interest expense",
    "da":               "Depreciation & amortisation",
    "capex":            "Capital expenditure",
    "sbc":              "Stock-based compensation",
    "nwc":              "Working capital (current assets & liabilities)",
    "diluted_shares":   "Diluted share count",
    "total_debt":       "Total debt",
    "cash":             "Cash & equivalents",
    "price":            "Share price",
    "market_cap":       "Market capitalisation",
    "beta":             "Beta",
    "fx_rate":          "Exchange rate",
    "industry":         "Industry classification",
}

_NOT_IN_FILINGS = {
    "price":      "Market data. Filings do not contain a live share price.",
    "beta":       "Market data. Beta is estimated from share-price history, which filings do not contain.",
    "fx_rate":    "Market data. Live exchange rates are not in filings.",
    "industry":   "Yahoo's industry label is used to pick the industry margin for unproven companies.",
}


def _align(series: pd.Series, ref: pd.DatetimeIndex) -> pd.Series:
    """Map a series onto reference fiscal-year-end dates (nearest within tolerance)."""
    if series.empty or len(ref) == 0:
        return pd.Series(dtype=float)
    out = {}
    for d in ref:
        gaps = abs(series.index - d)
        i = int(gaps.argmin())
        if gaps[i].days <= DATE_TOLERANCE_DAYS:
            out[d] = float(series.iloc[i])
    return pd.Series(out, dtype=float).sort_index()


def _value_at(series: pd.Series, d) -> float:
    s = _align(series, pd.DatetimeIndex([d]))
    return float(s.iloc[0]) if not s.empty else float("nan")


def _gap(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)) or abs(b) < 1e-9:
        return float("nan")
    return abs(a - b) / abs(b)


def apply_filing_data(raw, reconcile_result) -> dict:
    """
    Overwrite fields on `raw` (a RawData from fetch_raw) with SEC filing data
    where usable.  Returns provenance: field → {"source": ..., "detail": ...}.
    """
    sources = list(getattr(reconcile_result, "sources", []) or [])
    edgar = next((s for s in sources if s.source_name == "EDGAR"), None)
    prov: dict[str, dict] = {}

    def mark(field, source, detail):
        prov[field] = {"source": source, "detail": detail}

    if edgar is not None and (raw.currency or "USD").upper() != "USD":
        edgar_reason = (f"The company reports in {raw.currency}; its SEC filings are not in "
                        "US-GAAP dollar form, so Yahoo's figures are used throughout.")
        edgar = None
    elif edgar is None:
        status = getattr(reconcile_result, "source_status", {}) if reconcile_result else {}
        why = status.get("EDGAR") if isinstance(status, dict) else None
        if why and "IFRS" in why:
            edgar_reason = ("The company files under IFRS (Form 20-F), whose tags differ from "
                            "US GAAP, so Yahoo's figures are used throughout.")
        else:
            edgar_reason = ("SEC EDGAR data was unavailable for this ticker (not an SEC "
                            "filer, or the request failed), so Yahoo's figures are used.")
    else:
        edgar_reason = ""

    # ── Income statement and cash-flow lines ─────────────────────────────────
    yahoo_series = {
        "revenue": raw.revenue, "ebit": raw.ebit, "pretax_income": raw.pretax_income,
        "tax_provision": raw.tax_provision, "interest_expense": raw.interest_expense,
        "da": raw.da, "capex": raw.capex, "sbc": raw.sbc,
    }
    edgar_series = {} if edgar is None else {
        "revenue": edgar.revenue, "ebit": edgar.ebit, "pretax_income": edgar.pretax_income,
        "tax_provision": edgar.tax_provision, "interest_expense": edgar.interest_expense,
        "da": edgar.dep_amort, "capex": edgar.capex, "sbc": edgar.sbc,
    }
    tags = getattr(edgar, "tags_used", {}) if edgar is not None else {}
    tag_key = {"da": "dep_amort"}

    # Revenue sets the fiscal years every other line is aligned to.
    use_edgar_rev = False
    if edgar is not None:
        e_rev, y_rev = edgar.revenue, raw.revenue
        if len(e_rev) < 2:
            rev_reason = "Revenue is not tagged in a usable way in the company's XBRL filings."
        elif not y_rev.empty and e_rev.index[-1] < y_rev.index[-1] - pd.Timedelta(days=DATE_TOLERANCE_DAYS):
            rev_reason = ("The latest 10-K in the XBRL data is older than the latest year "
                          "Yahoo reports.")
        else:
            g = _gap(float(e_rev.iloc[-1]), _value_at(y_rev, e_rev.index[-1]))
            if math.isfinite(g) and g > _MAX_GAP["revenue"]:
                rev_reason = (f"The tagged revenue differs from the reported total by {g:.0%}, "
                              "which usually means the company tags revenue by segment or "
                              "product line rather than as one total.")
            else:
                use_edgar_rev = True
                rev_reason = ""

    if use_edgar_rev:
        ref = edgar.revenue.index[-YEARS:]
        raw.revenue = edgar.revenue.iloc[-YEARS:].astype(float)
        mark("revenue", EDGAR, f"10-K, XBRL tag {tags.get('revenue', '')}")
        for f in ("ebit", "pretax_income", "tax_provision", "interest_expense", "da", "capex", "sbc"):
            e = _align(edgar_series[f], ref)
            y = _align(yahoo_series[f], ref)
            latest = ref[-1]
            reason = ""
            if e.empty or latest not in e.index:
                if f == "da":
                    reason = ("The company does not report D&A as a single tagged line; it "
                              "splits it across several tags, and adding those up misses "
                              "untagged items. Yahoo's cash-flow-statement total is used.")
                elif f == "ebit":
                    reason = ("Operating income is not tagged in the filings (common for banks "
                              "and insurers, which have no operating-income line).")
                elif not edgar_series[f].empty:
                    reason = ("The company stopped tagging this line in its recent filings, so "
                              "the filing data is out of date.")
                else:
                    reason = "This line is not tagged in the company's XBRL filings."
            else:
                g = _gap(float(e[latest]), float(y.get(latest, float("nan"))))
                small = (abs(float(y.get(latest, 0.0))) < 0.01 * abs(float(raw.revenue.iloc[-1])))
                if math.isfinite(g) and g > _MAX_GAP.get(f, _DEFAULT_MAX_GAP) and not small:
                    reason = (f"The tagged figure differs from the reported line by {g:.0%}, a "
                              "sign the company uses an unusual tag or splits the line, so "
                              "Yahoo's reported figure is used.")
            if reason:
                setattr(raw, f, y)
                mark(f, YAHOO, reason)
            else:
                setattr(raw, f, e)
                mark(f, EDGAR, f"10-K, XBRL tag {tags.get(tag_key.get(f, f), '')}")
    else:
        reason = edgar_reason or (rev_reason + " To keep every line on the same fiscal "
                                  "years, the income and cash-flow statements all come from Yahoo.")
        for f in yahoo_series:
            mark(f, YAHOO, reason)

    # ── Working capital: fiscal-year-end balance sheet ───────────────────────
    nwc_items = None
    if use_edgar_rev:
        vals = (edgar.current_assets, edgar.current_liabilities, edgar.cash_fy)
        if all(math.isfinite(v) for v in vals):
            nwc_items = vals
    if nwc_items:
        raw.current_assets, raw.current_liabilities, raw.cash_nwc = nwc_items
        raw.current_debt = edgar.current_debt if math.isfinite(edgar.current_debt) else 0.0
        mark("nwc", EDGAR, f"10-K balance sheet at {raw.revenue.index[-1].date()}")
    else:
        raw.cash_nwc = raw.cash
        mark("nwc", YAHOO, edgar_reason or ("Current assets or liabilities are not tagged "
                                            "(banks and insurers do not split them), so "
                                            "Yahoo's balance sheet is used."))

    # ── Bridge items: latest balance sheet (EDGAR > FMP > Yahoo) ─────────────
    pref = getattr(reconcile_result, "preferred", None)
    if pref is not None and (pref.currency or "").upper() != (raw.currency or "").upper():
        pref = None

    def _pref_detail(p, what):
        if p.source_name == "EDGAR":
            d = p.shares_date if what == "shares" else p.balance_date
            when = f" at {pd.Timestamp(d).date()}" if d is not None else ""
            return f"latest 10-Q/10-K{when}"
        return f"{p.source_name} (SEC data unavailable for this field)"

    # Some filers tag share counts in millions (MCD files 712.3, not 712,300,000).
    # A filing count more than 10x away from Yahoo's is a scaling error, not a
    # real difference, so fall back to Yahoo's count.
    yahoo_shares = raw.diluted_shares
    if (pref is not None and math.isfinite(pref.diluted_shares) and pref.diluted_shares > 0
            and yahoo_shares > 0 and not 0.1 < pref.diluted_shares / yahoo_shares < 10):
        mark("diluted_shares", YAHOO,
             f"The filing share count ({pref.diluted_shares:,.0f}) is more than 10x away from "
             f"Yahoo's ({yahoo_shares:,.0f}), so it looks mis-scaled.")
        shares_from_filing = False
    elif pref is not None and math.isfinite(pref.diluted_shares) and pref.diluted_shares > 0:
        raw.diluted_shares = float(pref.diluted_shares)
        src = EDGAR if pref.source_name == "EDGAR" else pref.source_name
        mark("diluted_shares", src, _pref_detail(pref, "shares") +
             (", diluted weighted-average shares" if pref.source_name == "EDGAR" else ""))
        shares_from_filing = pref.source_name == "EDGAR"
    else:
        mark("diluted_shares", YAHOO, edgar_reason or "Share count not found in the filings.")
        shares_from_filing = False

    if pref is not None and math.isfinite(pref.total_debt) and pref.total_debt >= 0:
        raw.total_debt = float(pref.total_debt)   # already standardised by the source fetcher
        mark("total_debt", EDGAR if pref.source_name == "EDGAR" else pref.source_name,
             _pref_detail(pref, "debt") + "; bonds, loans, commercial paper and finance leases "
             "(operating leases excluded)")
    else:
        raw.total_debt = max(0.0, raw.total_debt - getattr(raw, "op_lease_liab", 0.0))
        mark("total_debt", YAHOO, (edgar_reason or "Debt not found in the filings.") +
             " Operating lease liabilities are removed from Yahoo's total.")

    if pref is not None and math.isfinite(pref.cash) and pref.cash >= 0:
        raw.cash = float(pref.cash)
        mark("cash", EDGAR if pref.source_name == "EDGAR" else pref.source_name,
             _pref_detail(pref, "cash"))
    else:
        mark("cash", YAHOO, edgar_reason or "Cash is not tagged in the filings (common for banks).")

    # ── Market data: always Yahoo ────────────────────────────────────────────
    if shares_from_filing and raw.current_price_usd > 0:
        # Keep market cap consistent with the share count the model divides by.
        raw.market_cap_usd = raw.current_price_usd * raw.diluted_shares
        raw.market_cap_local = (raw.market_cap_usd / raw.fx_rate) if raw.fx_rate > 0 else raw.market_cap_usd
        mark("market_cap", YAHOO, "Yahoo share price × filing share count.")
    else:
        if not raw.market_cap_usd > 0 and raw.current_price_usd > 0 and raw.diluted_shares > 0:
            raw.market_cap_usd = raw.current_price_usd * raw.diluted_shares
            raw.market_cap_local = (raw.market_cap_usd / raw.fx_rate) if raw.fx_rate > 0 else raw.market_cap_usd
        mark("market_cap", YAHOO, "Market data. Filings do not contain a live market value.")
    for f, why in _NOT_IN_FILINGS.items():
        if f == "fx_rate" and (raw.currency or "USD").upper() == "USD":
            continue
        mark(f, YAHOO, why)

    if not raw.current_price_usd > 0:
        raise ValueError(f"No share price available for {raw.ticker} from Yahoo (quote or price history).")
    if not raw.diluted_shares > 0:
        raise ValueError("No share count available from any data source (Yahoo, FMP or SEC EDGAR).")

    return {k: prov[k] for k in FIELD_LABELS if k in prov}
