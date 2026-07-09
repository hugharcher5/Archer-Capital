"""
S13: Opportunistic Insider Cluster Buying  (independent trial #20)
====================================================================
Reformulation of E1 (trial #1, mgmt_pit M1 "insider conviction": naive net
Form-4 buying, IC ~= 0.0002, FAIL). E1 pooled ROUTINE (scheduled/recurring)
and OPPORTUNISTIC (discretionary) insider trades together, which the
literature (Cohen, Malloy, Pomorski 2012, "Decoding Inside Information")
argues destroys most of the signal: routine trades carry little
information content and dilute the opportunistic subset.

Signal (pre-registered, locked before any results are seen):
  - ROUTINE purchase: an insider (reportingOwner CIK, keyed per issuer) who
    purchased in the SAME calendar month in >=2 of the prior 3 calendar
    years at the SAME company, OR any transaction flagged <aff10b5One> true
    if present in the Form 4 XML (added by SEC 2023 rule; expected absent
    for this entire 2015-2021 in-sample window -- checked empirically on a
    2017 Apple Form 4, confirmed no such element exists pre-2023 schema).
  - OPPORTUNISTIC purchase: everything else (open-market buy, action=A,
    price>0).
  - CLUSTER EVENT (core new signal): >=3 DIFFERERENT insiders at the same
    company make OPPORTUNISTIC purchases within a rolling 30-calendar-day
    window, with the triggering (3rd distinct insider's) purchase date
    falling inside the trailing quarter before rebalance_date.
  - Long: companies with a confirmed cluster event in the trailing quarter.
  - Short: companies with ZERO Form-4 acquisition (action=A, any price)
    activity at all in the trailing quarter -- a clean long-tilt-vs-no-
    signal comparison, NOT a short-on-selling design (conceptually
    distinct signal, not conflated here).
  - Naive net-insider-buying (E1's M1, verbatim) is re-run on the SAME
    universe/window as a direct falsification comparison point.

Filing-date gated throughout (same look-ahead-trap note as E1): a Form 4
transaction only counts toward a given rebalance_date if FILED <=
rebalance_date. Transaction date (not filed date) determines calendar-
month / cluster-window membership once a filing is admissible.

Data reuse (per instructions -- do NOT rebuild data infrastructure that
already exists):
  - Universe construction, Tiingo price cache, SEC facts/submissions cache,
    and the existing parsed Form-4 cache (research/mgmt_pit/cache/) are
    100% reused from mgmt_pit (E1's own data pipeline), read-only.
  - The existing parsed Form-4 cache (form4_all.json, ~40k filings) has
    shares/price/action/date per transaction but NOT reporting-owner
    identity (needed to tell "3 different insiders" apart) or the 10b5-1
    flag. Of the ~40k cached filings, only ~10.6k contain at least one
    open-market purchase (action=A, price>0) -- only THOSE need their raw
    Form 4 XML re-fetched (once) to extract owner identity + 10b5-1. This
    trial's own cache (cache/form4_owner/) stores that enrichment; it does
    not duplicate or replace mgmt_pit's cache.

Universe: $100M-2B, survivorship-bias-free, PIT, quarterly rebalance, 1Q
hold -- identical construction and cadence to E1/E2/E3 for comparability.
Beta-neutral leg sizing (60-day OLS vs EW universe return, adjClose).
Name-count guard: if the long (cluster) leg has <20 names in a quarter, that
quarter's return is excluded from the P&L series (not forced), but its
event count is still reported. Fraction of quarters with a usable long leg
is reported explicitly, since cluster events are expected to be rare.

In-sample: 2015-01-01 -> 2021-01-01, quarterly, 24 periods (matches
E1/E2/E3 and the S3/S5/S11 program default).
Trial: independent trial #20. This is a distinct mechanism (routine/
opportunistic classification + clustering) from E1's naive pooled signal,
so it consumes a NEW trial slot rather than reusing E1's (E1 was never
logged into the unified research/trial_registry.csv -- it lives only in
mgmt_pit/results/trial_registry.csv from the old 4-trial-count era -- so
this is the first time the insider-buying mechanism enters the unified
DSR count).
CONCURRENCY NOTE: N_TRIALS_DSR was 17 (Russell family) when this trial's
data pipeline was launched, so it was originally coded/run as trial #18.
Two other trials (analystrevision_S14 -> #18, compositeIC_S15 -> #19) were
logged to the shared registry by other concurrent sessions during this
trial's ~2.5hr runtime (Form-4 owner-identity re-fetch + backtest), which
was only discovered via a trial_number collision at registry-write time.
Corrected post-hoc to the true N=20 (DSR threshold recomputed; verdict
unchanged -- still FAILS DSR either way). Lesson: N_TRIALS_DSR must be
read fresh from the registry immediately before writing results, not just
at task start, when trials may run for hours with other sessions active.

Falsification diagnostics:
  - Sign-convention audit: cluster-flagged names' mean forward return vs
    zero-activity names' mean forward return, checked each usable quarter.
  - Concentration check: single-quarter share of |return| mass.
  - Total cluster-event count across the full in-sample window reported
    explicitly as a sample-size caveat if small (Russell-3 precedent).

Namespacing: all outputs live under research/insider_clusters/ -- fully
separate from any other concurrently running trial.
"""

import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
from scipy import stats

# =============================================================================
# SECTION 1: Paths, config, constants
# =============================================================================

SEC_UA = "Hugh Archer hugharcher5@gmail.com"

MGMT_PIT  = Path(__file__).resolve().parents[1] / "mgmt_pit"
CACHE     = MGMT_PIT / "cache"                              # shared, read-only
OWN_CACHE = Path(__file__).resolve().parent / "cache"        # S13-private cache
OWNER_DIR = OWN_CACHE / "form4_owner"                        # per-accn owner enrichment
RESULTS   = Path(__file__).resolve().parent / "results"
REGISTRY  = Path(__file__).resolve().parents[1] / "trial_registry.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
OWNER_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

MCAP_MIN, MCAP_MAX = 100e6, 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")
_all_dates = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 24 quarterly periods

# ── Signal construction (pre-registered, locked) ─────────────────────────────
ROUTINE_LOOKBACK_YEARS = 3
ROUTINE_MIN_YEARS_HIT  = 2       # same calendar month in >=2 of prior 3 years
CLUSTER_MIN_INSIDERS   = 3       # >=3 different insiders
CLUSTER_WINDOW_DAYS    = 30      # rolling window
TRAILING_MONTHS        = 3       # "trailing quarter"
LONG_LEG_MIN_NAMES     = 20      # name-count guard

# ── Cost model (identical to E1/S3/S5/S11 program) ───────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR ───────────────────────────────────────────────────────────────────────
N_TRIALS = 20   # true N at write-time (see CONCURRENCY NOTE in module docstring)

_TIINGO_LAST: float = 0.0
_SEC_LAST:    float = 0.0


def _sec_get(url: str, timeout: int = 30) -> requests.Response:
    global _SEC_LAST
    wait = 0.2 - (time.time() - _SEC_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=(10, timeout))
    _SEC_LAST = time.time()
    return r


# =============================================================================
# SECTION 2: Shared read-only caches (Tiingo tickers, prefilter, prices, adj)
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
    print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
    return df


def load_prefilter() -> pd.DataFrame:
    path = CACHE / "sec_prefilter.csv"
    df = pd.read_csv(path)
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors.")
    return df


_PRICE_CACHE: dict[str, pd.DataFrame] = {}
_PRICE_EMPTY = pd.DataFrame(columns=["date", "close", "volume"])


def preload_all_prices() -> None:
    t0 = time.time()
    with open(CACHE / "prices_all.pkl", "rb") as f:
        data = pickle.load(f)
    _PRICE_CACHE.update(data.get("prices", {}))
    print(f"[Preload] Raw prices: {len(_PRICE_CACHE):,} tickers in {time.time()-t0:.1f}s")


def _last_close(ticker: str, as_of: pd.Timestamp) -> Optional[float]:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return None
    sub = df[df["date"] <= as_of]
    return float(sub.iloc[-1]["close"]) if not sub.empty else None


def _avg_dollar_vol(ticker: str, as_of: pd.Timestamp, lookback: int = 60) -> float:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return 0.0
    sub = df[df["date"] <= as_of].tail(lookback)
    return float((sub["close"] * sub["volume"]).mean()) if not sub.empty else 0.0


def _get_price_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
    if df.empty:
        return _PRICE_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


_ADJ_CACHE: dict[str, pd.DataFrame] = {}
_ADJ_EMPTY = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices() -> None:
    t0 = time.time()
    adj_pkl = CACHE / "adjclose_all.pkl"
    if not adj_pkl.exists():
        print("[AdjPreload] No pickle -- beta will use raw close fallback.")
        return
    with open(adj_pkl, "rb") as f:
        data = pickle.load(f)
    _ADJ_CACHE.update(data.get("adj", {}))
    adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    print(f"[AdjPreload] {adj_loaded:,} tickers in {time.time()-t0:.1f}s")


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 3: Shares-outstanding (reused verbatim from mgmt_pit's trimmed facts)
# =============================================================================

_FACTS_CACHE: dict[int, Optional[dict]] = {}


def preload_shares_facts() -> None:
    """mgmt_pit's sec_facts_trimmed.json already includes CommonStockSharesOutstanding
    / EntityCommonStockSharesOutstanding (its own NEEDED_CONCEPTS) -- reused as-is,
    read-only, no new concepts required for this trial."""
    t0 = time.time()
    with open(CACHE / "sec_facts_trimmed.json") as f:
        raw = json.load(f)
    for k, v in raw.items():
        _FACTS_CACHE[int(k)] = v
    loaded = sum(1 for v in _FACTS_CACHE.values() if v is not None)
    print(f"[Preload] SEC facts (shares): {loaded:,} CIKs in {time.time()-t0:.1f}s")


def get_gaap_value_pit(facts: dict, concept: str, as_of_date: pd.Timestamp,
                        namespace: str = "us-gaap") -> tuple:
    """Verbatim reuse of mgmt_pit.run.get_gaap_value_pit's logic."""
    try:
        units_dict = facts["facts"][namespace][concept]["units"]
    except (KeyError, TypeError):
        return (None, None, None)

    as_of_str = as_of_date.strftime("%Y-%m-%d")
    best_any = None
    best_annual = None
    has_duration = False

    for entries in units_dict.values():
        for entry in entries:
            filed = entry.get("filed")
            end = entry.get("end")
            val = entry.get("val")
            if filed is None or end is None or val is None:
                continue
            if filed > as_of_str:
                continue
            start = entry.get("start")
            period_days = 0
            if start:
                has_duration = True
                try:
                    y1, m1, d1 = int(end[:4]), int(end[5:7]), int(end[8:10])
                    y0, m0, d0 = int(start[:4]), int(start[5:7]), int(start[8:10])
                    period_days = (y1 - y0) * 365 + (m1 - m0) * 30 + (d1 - d0)
                except Exception:
                    pass
            sort_key = (filed, end)
            if best_any is None or sort_key > best_any[0]:
                best_any = (sort_key, float(val), end, filed)
            if 300 <= period_days <= 400:
                if best_annual is None or sort_key > best_annual[0]:
                    best_annual = (sort_key, float(val), end, filed)

    winner = best_annual if (has_duration and best_annual is not None) else best_any
    if winner is None:
        return (None, None, None)
    return (winner[1], pd.Timestamp(winner[2]), pd.Timestamp(winner[3]))


def get_shares_pit(cik: int, as_of: pd.Timestamp) -> Optional[float]:
    facts = _FACTS_CACHE.get(cik)
    if facts is None:
        return None
    val, _, _ = get_gaap_value_pit(facts, "CommonStockSharesOutstanding", as_of)
    if val is None or val <= 0:
        val, _, _ = get_gaap_value_pit(facts, "EntityCommonStockSharesOutstanding", as_of)
    return val if (val is not None and val > 0) else None


# =============================================================================
# SECTION 4: SEC submissions (reused, read-only, per-CIK cache)
# =============================================================================

_SUBS_CACHE: dict[int, Optional[dict]] = {}


def fetch_submissions(cik: int) -> Optional[dict]:
    if cik in _SUBS_CACHE:
        return _SUBS_CACHE[cik]
    cache_file = CACHE / "sec_subs" / f"{cik}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            data = json.load(f)
        _SUBS_CACHE[cik] = data
        return data
    # Not expected: E1 already built this universe. Network fallback kept for safety.
    url = f"https://data.sec.gov/submissions/CIK{str(cik).zfill(10)}.json"
    try:
        resp = _sec_get(url)
    except Exception:
        _SUBS_CACHE[cik] = None
        return None
    if resp.status_code != 200:
        _SUBS_CACHE[cik] = None
        return None
    data = resp.json()
    _SUBS_CACHE[cik] = data
    return data


def fetch_form4_filings(cik: int, submissions: dict) -> list:
    """Verbatim reuse of mgmt_pit.run.fetch_form4_filings."""
    try:
        recent = submissions["filings"]["recent"]
        forms = recent.get("form", [])
        accns = recent.get("accessionNumber", [])
        filed_dates = recent.get("filedDate", recent.get("filingDate", []))
        primary_docs = recent.get("primaryDocument", [])
    except (KeyError, TypeError):
        return []
    results = []
    for i, form in enumerate(forms):
        if form == "4":
            try:
                results.append((accns[i], filed_dates[i],
                                 primary_docs[i] if i < len(primary_docs) else ""))
            except IndexError:
                continue
    return results


# =============================================================================
# SECTION 5: Existing parsed Form-4 cache (mgmt_pit, read-only) -- has
# date/shares/price/action but NOT owner identity or 10b5-1.
# =============================================================================

_FORM4_BASIC: dict[str, list] = {}


def load_form4_basic_cache() -> None:
    t0 = time.time()
    with open(CACHE / "form4_all.json") as f:
        _FORM4_BASIC.update(json.load(f))
    print(f"[Form4] Loaded {len(_FORM4_BASIC):,} existing parsed filings "
          f"(shares/price/action, no owner identity) in {time.time()-t0:.1f}s")


# =============================================================================
# SECTION 6: Owner-identity + 10b5-1 enrichment (NEW for this trial)
# =============================================================================

def _find_all(root, tag_suffix: str):
    return [el for el in root.iter() if el.tag.endswith(tag_suffix)]


def _find_first_text(el, tag_suffix: str) -> Optional[str]:
    """Recursively search descendants of el for the first element whose tag
    ends with tag_suffix and return its text (direct text, not wrapped)."""
    for desc in el.iter():
        if desc is el:
            continue
        if desc.tag.endswith(tag_suffix):
            return desc.text
    return None


def _child_value(el, tag_suffix: str) -> Optional[str]:
    for child in el:
        if child.tag.endswith(tag_suffix):
            for sub in child:
                if sub.tag.endswith("value"):
                    return sub.text
            return child.text
    return None


def parse_form4_owner_xml(cik: int, accn: str, doc: str) -> Optional[dict]:
    """
    Fetch (or load cached) raw Form 4 XML and extract:
      - owner_key: tuple of reporting-owner CIKs (fallback: names) -- identity
        used to tell "different insiders" apart.
      - aff10b5one: True/False/None (None = element absent, expected for the
        entire 2015-2021 in-sample window -- confirmed empirically absent in
        pre-2023-schema filings).
      - transactions: list of {date, shares, price, action} (nonDerivative
        purchases/sales only) -- re-extracted here so this cache is fully
        self-contained (doesn't need to cross-reference mgmt_pit's cache).
    """
    accn_nodash = accn.replace("-", "")
    cache_file = OWNER_DIR / f"{accn_nodash}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    if not doc:
        return None
    doc_clean = doc.split("/")[-1] if "/" in doc else doc
    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{doc_clean}"
    try:
        resp = _sec_get(url)
    except Exception as e:
        print(f"  [Form4Owner] network error {accn}: {e}")
        return None
    if resp.status_code != 200:
        return None
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return None

    owner_ciks = []
    owner_names = []
    for ro in _find_all(root, "reportingOwner"):
        rci = _find_first_text(ro, "rptOwnerCik")
        rname = _find_first_text(ro, "rptOwnerName")
        if rci:
            owner_ciks.append(rci.strip().lstrip("0") or "0")
        if rname:
            owner_names.append(rname.strip())

    aff10b5one = None
    for el in root.iter():
        if el.tag.endswith("aff10b5One"):
            txt = (el.text or "").strip().lower()
            aff10b5one = txt in ("1", "true")
            break

    transactions = []
    for nd_tx in root.iter():
        if not nd_tx.tag.endswith("nonDerivativeTransaction"):
            continue
        tx_amounts = None
        date_el_text = None
        for child in nd_tx:
            if child.tag.endswith("transactionDate"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        date_el_text = sub.text
            if child.tag.endswith("transactionAmounts"):
                tx_amounts = child
        if tx_amounts is None:
            continue
        shares = price = action = None
        for child in tx_amounts:
            if child.tag.endswith("transactionShares"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        shares = sub.text
            elif child.tag.endswith("transactionPricePerShare"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        price = sub.text
            elif child.tag.endswith("transactionAcquiredDisposedCode"):
                for sub in child:
                    if sub.tag.endswith("value"):
                        action = sub.text
        try:
            if date_el_text and shares is not None and action in ("A", "D"):
                transactions.append({
                    "date": date_el_text,
                    "shares": float(shares),
                    "price": float(price) if price is not None else 0.0,
                    "action": action.strip(),
                })
        except (ValueError, AttributeError):
            continue

    owner_key = owner_ciks if owner_ciks else owner_names
    result = {
        "cik": cik,
        "owner_key": owner_key,
        "aff10b5one": aff10b5one,
        "transactions": transactions,
    }
    with open(cache_file, "w") as f:
        json.dump(result, f)
    return result


# =============================================================================
# SECTION 7: Universe builder (identical construction to mgmt_pit/E1)
# =============================================================================

def build_universe(rebalance_date: pd.Timestamp, tiingo_tickers: pd.DataFrame,
                    cik_sic_map: dict) -> pd.DataFrame:
    survivors_set = set(cik_sic_map.keys())
    active = tiingo_tickers[
        (tiingo_tickers["startDate"] <= rebalance_date)
        & (tiingo_tickers["endDate"].isna() | (tiingo_tickers["endDate"] >= rebalance_date))
        & tiingo_tickers["ticker"].isin(survivors_set)
    ]
    rows = []
    for _, row in active.iterrows():
        ticker = row["ticker"]
        cik, sic = cik_sic_map.get(ticker, (None, None))
        if cik is None:
            continue
        price = _last_close(ticker, rebalance_date)
        if price is None or price < PRICE_MIN:
            continue
        shares = get_shares_pit(cik, rebalance_date)
        if shares is None:
            continue
        mcap = shares * price
        if not (MCAP_MIN <= mcap <= MCAP_MAX):
            continue
        dvol = _avg_dollar_vol(ticker, rebalance_date)
        if dvol < DOLLAR_VOL_MIN:
            continue
        rows.append({"ticker": ticker, "cik": cik, "sic": sic, "mcap": mcap,
                      "price": price, "shares": shares, "avg_dvol": dvol})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["ticker", "cik", "sic", "mcap", "price", "shares", "avg_dvol"])
    df = df.drop_duplicates(subset=["ticker"], keep="first")
    print(f"  [Universe] {rebalance_date.date()} -> {len(df):,} names")
    return df


# =============================================================================
# SECTION 8: Naive M1 (E1 verbatim) -- direct comparison point
# =============================================================================

def compute_m1_insider_buying(cik: int, submissions: dict, as_of_date: pd.Timestamp,
                               shares_outstanding: float) -> float:
    """Verbatim reuse of mgmt_pit.run.compute_m1_insider_buying (6-month window,
    net A-D shares / shares outstanding)."""
    if shares_outstanding is None or shares_outstanding <= 0:
        return 0.0
    window_start_str = (as_of_date - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
    as_of_str = as_of_date.strftime("%Y-%m-%d")
    filings = fetch_form4_filings(cik, submissions)

    net_shares = 0.0
    parsed_any = False
    for accn, filed_date_str, doc in filings:
        if not (window_start_str <= filed_date_str <= as_of_str):
            continue
        accn_nodash = accn.replace("-", "")
        txns = _FORM4_BASIC.get(accn_nodash)
        if txns is None:
            continue
        for txn in txns:
            txn_date = txn.get("date", "")
            if not (window_start_str <= txn_date <= as_of_str):
                continue
            parsed_any = True
            if txn["action"] == "A":
                net_shares += txn["shares"]
            elif txn["action"] == "D":
                net_shares -= txn["shares"]
    if not parsed_any:
        return 0.0
    return net_shares / shares_outstanding


if __name__ == "__main__":
    print("This module is imported by fetch_owners.py and backtest.py.")
