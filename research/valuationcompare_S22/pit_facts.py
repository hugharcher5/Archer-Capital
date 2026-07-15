"""
Point-in-time SEC XBRL fact extraction, shared by all four S22 signals
(DCF, P/E, P/S, P/B all need PIT-gated revenue/net income/book equity/shares;
the DCF additionally needs EBIT, D&A, capex, interest expense, tax, SBC,
current assets/liabilities, cash, and debt components).

Reuses the full per-CIK SEC company-facts JSON already cached at
research/mgmt_pit/cache/sec_facts/{cik}.json (2,142 files, 6.1GB, built by
earlier trials -- no new SEC fetches). Builds one DCF-specific trimmed cache
(analogous to gp/run.py's gp_facts_trimmed.json) containing only the ~15
concepts needed here, so the full 682-concept JSON files are parsed once,
not on every rebalance date.

Filing-date discipline: every extractor takes an `as_of_str` and only
considers XBRL facts with `filed <= as_of_str` -- matching the exact
convention already used throughout this registry (gp/run.py's
_get_ttm_flow / _get_assets_pit).
"""
import json
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MGMT_PIT = Path(__file__).resolve().parents[1] / "mgmt_pit"
FULL_FACTS_DIR = MGMT_PIT / "cache" / "sec_facts"
TRIMMED_CACHE = Path(__file__).resolve().parent / "cache" / "s22_facts_trimmed.json"
TRIMMED_CACHE.parent.mkdir(parents=True, exist_ok=True)

QTR_MIN_DAYS, QTR_MAX_DAYS = 75, 115
ANN_MIN_DAYS, ANN_MAX_DAYS = 300, 400

# ── XBRL concept hierarchies (tried in order; first match with data wins) ────
REVENUE_CONCEPTS = [
    "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet", "SalesRevenueServicesNet",
]
EBIT_CONCEPTS = ["OperatingIncomeLoss"]
NET_INCOME_CONCEPTS = ["NetIncomeLoss", "ProfitLoss"]
STOCKHOLDERS_EQUITY_CONCEPTS = [
    "StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
]
DA_CONCEPTS_PRIMARY = ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet"]
DA_DEP_FALLBACK = ["DepreciationAndAmortization", "OtherDepreciationAndAmortization", "Depreciation"]
DA_AMORT_FALLBACK = ["AmortizationOfIntangibleAssets", "AmortizationOfAcquiredIntangibles"]
CAPEX_CONCEPTS = ["PaymentsToAcquirePropertyPlantAndEquipment"]
INTEREST_EXPENSE_CONCEPTS = ["InterestExpense", "InterestExpenseDebt"]
TAX_CONCEPTS = ["IncomeTaxExpenseBenefit"]
PRETAX_CONCEPTS = ["IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"]
SBC_CONCEPTS = ["ShareBasedCompensation"]
CASH_CONCEPTS = ["CashAndCashEquivalentsAtCarryingValue"]
CURRENT_ASSETS_CONCEPTS = ["AssetsCurrent"]
CURRENT_LIAB_CONCEPTS = ["LiabilitiesCurrent"]
LTD_CONCEPTS = ["LongTermDebtNoncurrent", "LongTermDebt"]
STD_CONCEPTS = ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"]
FL_LT_CONCEPTS = ["FinanceLeaseLiabilityNoncurrent"]
FL_CUR_CONCEPTS = ["FinanceLeaseLiabilityCurrent"]
SHARES_CONCEPTS = ["WeightedAverageNumberOfDilutedSharesOutstanding",
                   "CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]

ALL_CONCEPTS = list(set(
    REVENUE_CONCEPTS + EBIT_CONCEPTS + NET_INCOME_CONCEPTS + STOCKHOLDERS_EQUITY_CONCEPTS
    + DA_CONCEPTS_PRIMARY + DA_DEP_FALLBACK + DA_AMORT_FALLBACK + CAPEX_CONCEPTS
    + INTEREST_EXPENSE_CONCEPTS + TAX_CONCEPTS + PRETAX_CONCEPTS + SBC_CONCEPTS
    + CASH_CONCEPTS + CURRENT_ASSETS_CONCEPTS + CURRENT_LIAB_CONCEPTS
    + LTD_CONCEPTS + STD_CONCEPTS + FL_LT_CONCEPTS + FL_CUR_CONCEPTS + SHARES_CONCEPTS
))

_FACTS_CACHE: dict[int, Optional[dict]] = {}


def preload_facts(ciks: list[int]) -> None:
    """Load trimmed DCF/valuation facts, building the trimmed cache on first run."""
    import time
    t0 = time.time()
    if TRIMMED_CACHE.exists():
        with open(TRIMMED_CACHE) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _FACTS_CACHE[int(k)] = v
        missing = [c for c in ciks if c not in _FACTS_CACHE]
        if missing:
            _load_from_full_files(missing, save=False)
        loaded = sum(1 for v in _FACTS_CACHE.values() if v is not None)
        print(f"[S22Facts] {loaded:,}/{len(ciks):,} loaded from trimmed cache in {time.time()-t0:.1f}s")
        return

    print(f"[S22Facts] First run -- trimming {len(ciks):,} full fact files (~1-2 min)...")
    _load_from_full_files(ciks, save=True)
    loaded = sum(1 for v in _FACTS_CACHE.values() if v is not None)
    print(f"[S22Facts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


def _load_from_full_files(ciks: list[int], save: bool) -> None:
    for cik in ciks:
        p = FULL_FACTS_DIR / f"{cik}.json"
        if not p.exists():
            _FACTS_CACHE[cik] = None
            continue
        try:
            with open(p) as f:
                full = json.load(f)
            trimmed: dict = {"facts": {"us-gaap": {}}}
            ug = full.get("facts", {}).get("us-gaap", {})
            for concept in ALL_CONCEPTS:
                if concept in ug:
                    trimmed["facts"]["us-gaap"][concept] = ug[concept]
            _FACTS_CACHE[cik] = trimmed
        except Exception:
            _FACTS_CACHE[cik] = None
    if save:
        print("[S22Facts] Saving trimmed cache...")
        serializable = {str(k): v for k, v in _FACTS_CACHE.items()}
        with open(TRIMMED_CACHE, "w") as f:
            json.dump(serializable, f)


def get_facts(cik: int) -> Optional[dict]:
    return _FACTS_CACHE.get(cik)


def _period_days(start_str: str, end_str: str) -> int:
    try:
        return (date.fromisoformat(end_str) - date.fromisoformat(start_str)).days
    except Exception:
        return 0


def annual_series(facts: Optional[dict], concept_candidates: list[str], as_of_str: str,
                   abs_val: bool = False) -> pd.Series:
    """PIT-gated annual (10-K, ~300-400 day period) series, filed <= as_of_str.
    Index = fiscal-period-end Timestamp, ascending. Deduplicated by end-date,
    keeping the most-recently-filed revision (handles restatements correctly --
    a later restatement is only used once ITS OWN filed date has passed)."""
    if facts is None:
        return pd.Series(dtype=float)
    for concept in concept_candidates:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        by_end: dict = {}
        for entries in units_dict.values():
            for e in entries:
                filed, end, val, start = e.get("filed"), e.get("end"), e.get("val"), e.get("start")
                if filed is None or end is None or val is None or filed > as_of_str:
                    continue
                if start is not None:
                    days = _period_days(start, end)
                    if not (ANN_MIN_DAYS <= days <= ANN_MAX_DAYS):
                        continue
                if e.get("form") not in ("10-K", "10-K/A"):
                    continue
                if end not in by_end or filed > by_end[end][0]:
                    by_end[end] = (filed, val)
        if not by_end:
            continue
        pts = {pd.Timestamp(end): (abs(float(v)) if abs_val else float(v))
               for end, (filed, v) in by_end.items()}
        s = pd.Series(pts).sort_index()
        if not s.empty:
            return s
    return pd.Series(dtype=float)


def scalar_pit(facts: Optional[dict], concept_candidates: list[str], as_of_str: str) -> float:
    """Most recent value (any period type -- balance-sheet instant or flow) filed <= as_of_str."""
    if facts is None:
        return float("nan")
    for concept in concept_candidates:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        best = None
        for entries in units_dict.values():
            for e in entries:
                filed, end, val = e.get("filed"), e.get("end"), e.get("val")
                if filed is None or end is None or val is None or filed > as_of_str:
                    continue
                key = (filed, end)
                if best is None or key > best[:2]:
                    best = (filed, end, float(val))
        if best is not None:
            return best[2]
    return float("nan")


def shares_pit(facts: Optional[dict], as_of_str: str) -> float:
    v = scalar_pit(facts, SHARES_CONCEPTS, as_of_str)
    return v if (not np.isnan(v)) and v > 0 else float("nan")


def da_series(facts: Optional[dict], as_of_str: str) -> pd.Series:
    """D&A with the same 3-tier fallback as fetch_edgar(): comprehensive tag,
    then dep+amort components summed, then dep-only."""
    da = annual_series(facts, DA_CONCEPTS_PRIMARY, as_of_str, abs_val=True)
    if not da.empty:
        return da
    dep = annual_series(facts, DA_DEP_FALLBACK, as_of_str, abs_val=True)
    amort = annual_series(facts, DA_AMORT_FALLBACK, as_of_str, abs_val=True)
    if dep.empty:
        return pd.Series(dtype=float)
    combined = dep.copy()
    if not amort.empty:
        idx = combined.index.intersection(amort.index)
        if len(idx):
            combined = combined.reindex(idx) + amort.reindex(idx)
    return combined.sort_index()


def total_debt_pit(facts: Optional[dict], as_of_str: str) -> float:
    ltd = scalar_pit(facts, LTD_CONCEPTS, as_of_str)
    std = scalar_pit(facts, STD_CONCEPTS, as_of_str)
    fl_lt = scalar_pit(facts, FL_LT_CONCEPTS, as_of_str)
    fl_cur = scalar_pit(facts, FL_CUR_CONCEPTS, as_of_str)
    components = [x for x in [ltd, std, fl_lt, fl_cur] if not np.isnan(x)]
    return float(sum(components)) if components else float("nan")
