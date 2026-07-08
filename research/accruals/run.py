"""
S11: Accruals Quality (Sloan 1996)  (independent trial #16)
=============================================================
Signal   : Total Accruals = (Net Income − Operating Cash Flow) / Total Assets
           NI and CFO are TTM (sum of 4 most recent quarterly filed periods,
           annual-period fallback), Total Assets is the most recent PIT
           balance-sheet value — same TTM-over-Assets shape as GP/A (trial
           #8), but built from earnings/cash-flow-statement concepts instead
           of gross profit. This is the only untested signal in the program
           that shares GP/A's core structural property: slow-moving, purely
           fundamental, XBRL-derived, no price-window or event-timing
           component. GP/A (semiannual variant) is the program's sole DSR
           pass to date — S11 is the highest-probability candidate to
           replicate that structural profile.

Hypothesis: high-accruals firms (earnings driven more by accounting
           adjustments than cash generation) subsequently underperform;
           low-accruals firms (earnings closely backed by cash flow)
           subsequently outperform. Distinct mechanism from GP/A
           (profitability LEVEL) and S5 asset growth (balance-sheet
           EXPANSION) — this is about earnings QUALITY.

Look-ahead discipline (same-filing gating — specific to accruals):
  - NI and CFO for a given fiscal quarter must come from the SAME as-filed
    report (same SEC accession number). XBRL entries for a period can be
    revised/restated at different times for different concepts; picking
    each concept's most-recently-filed value independently risks mixing a
    restated NI with an original (or differently-restated) CFO figure for
    the same nominal period. We gate on accn equality per matched period.
  - TTM: sum the 4 most recent FILED quarterly periods (75-115 days each,
    matched by accession number between NI and CFO), non-overlapping,
    deduplicated by period end-date (most-recently-filed revision, but only
    among the matched-accn pool). If <4 quarterly-matched periods exist,
    fall back to the most recent annual-period (300-400 days) matched pair
    as a TTM proxy (GP/A precedent).
  - Total Assets: most recent Assets entry filed ≤ as_of (instant,
    balance-sheet concept — no duration matching needed, per S5 precedent).
  - Every lookup gated on filed ≤ as_of (= rebalance_date).

Portfolio: Long BOTTOM quintile (lowest/most negative accruals — cash-backed
           earnings), short TOP quintile (highest accruals — earnings least
           backed by cash). Sign convention: long = low accruals.
           Beta-neutral: 60-day OLS vs EW universe return (adjClose layer).
           Equal-weight within each leg.
Name-count guard: if the natural quintile leg (ceil(n*0.20)) would hold
           fewer than 20 names, widen to TERCILES for that period only
           (construction rule, not a new trial — frequency logged).
Rebalance: Quarterly, pre-registered (no sweep — S3 lesson). Hold: 1 quarter.
           24 periods over 2015-2021.
In-sample: 2015-01-01 -> 2021-01-01. Walk-forward 2021-2025 stays reserved.
Trial    : #16 — independent family (does not overlap S3/GP-quality [#8],
           S1-S4/IVOL-MAX [#7], S6-S9/momentum [#6], S5/asset-growth [#12],
           S7/ivol-value [#13], S8/residual-reversal [#14], S10/amihud [#15]).
           DSR threshold computed at N_TRIALS=16.

Falsification diagnostics (embedded in primary design, not bolted on after):
  - Sign-convention audit: assert mean accruals of the long leg (low
    accruals) is below the mean accruals of the short leg (high accruals)
    every period before results are interpreted.
  - Concentration check: single quarter + single SIC major-group share of
    cumulative L/S return, flagged if disproportionate.
  - Plausibility filter: any accruals ratio beyond ±3 of the period's
    cross-sectional mean/std is excluded (shell-company / pre-merger
    artifact risk — same LGIH-style precedent as S5's ASSETS_FLOOR fix).
  - 2020-window diagnostic: cumulative return and share of total |return|
    mass contributed by the four 2020 rebalance dates reported explicitly,
    given how many other trials in this registry (S1, S2, S7, E2) had
    their result determined largely by the 2020 window.

Coverage note: unlike Assets (near-universal, S5) or GP (needs a Revenue/
COGS split many service firms don't report), the primary failure mode here
is the SAME-ACCN MATCH ITSELF — NI and CFO can each individually have great
coverage but fail to share an accession number for a given period (e.g. a
CFO restatement filed separately from the NI restatement). This is tracked
as its own coverage category (n_accn_mismatch), distinct from n_no_data
(no overlapping period at all). Direct (TTM-quarterly) vs annual-fallback
match rates reported each period and in aggregate; fallback use > 25% of
computed signals is flagged (IVOL/S5 precedent).

No-survivorship guarantee:
  - Delisted tickers retained with returns to last trading day.
  - Universe rebuilt PIT from SEC filings each quarter.

Namespacing: all outputs (results, coverage/cache) live under
research/accruals/ — kept fully separate from any other concurrently
running trial (S3-SA walk-forward may still be running).
"""

import json
import os
import pickle
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy import stats


# =============================================================================
# SECTION 1: Paths, config, constants
# =============================================================================

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()
TIINGO_KEY = os.environ.get("TIINGO_API_KEY", "")
SEC_UA     = "Hugh Archer hugharcher5@gmail.com"

MGMT_PIT = Path(__file__).resolve().parents[2] / "research" / "mgmt_pit"
CACHE    = MGMT_PIT / "cache"                              # shared, read-only
OWN_CACHE = Path(__file__).resolve().parent / "cache"      # S11-private cache
RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

OWN_CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (same as all prior experiments) ─────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── Beta window (for beta-neutral sizing) ─────────────────────────────────────
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

# ── Accruals XBRL concepts ────────────────────────────────────────────────────
NI_CONCEPTS  = ["NetIncomeLoss", "ProfitLoss"]
CFO_CONCEPTS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
ASSETS_CONCEPT  = "Assets"
SHARES_CONCEPTS = [
    "CommonStockSharesOutstanding",
    "EntityCommonStockSharesOutstanding",
]
ACCRUALS_NEEDED = set(NI_CONCEPTS) | set(CFO_CONCEPTS) | {ASSETS_CONCEPT} | set(SHARES_CONCEPTS)

# ── TTM flow-matching windows (calendar days between period start/end) ───────
TTM_QUARTERS = 4
QTR_MIN_DAYS = 75
QTR_MAX_DAYS = 115
ANN_MIN_DAYS = 300
ANN_MAX_DAYS = 400

# Plausibility floor on any Assets value used as the accruals denominator.
# Same shell-company / pre-reverse-merger placeholder filter as S5's
# ASSETS_FLOOR (LGIH-style artifact: a real operating company's CIK
# carrying a prior near-zero "shell" Assets entry that lands inside the
# PIT window and produces a nonsensical accruals ratio).
ASSETS_FLOOR = 1e6

# Cross-sectional outlier filter on the accruals ratio itself (distinct from
# ASSETS_FLOOR, which screens the denominator). Applied per-period after all
# signals are computed: any accruals_signal beyond ±3 of that period's
# cross-sectional mean/std is excluded before quintile construction.
OUTLIER_Z_THRESH = 3.0

# ── Backtest dates ────────────────────────────────────────────────────────────
IS_START        = pd.Timestamp("2015-01-01")
IS_END          = pd.Timestamp("2021-01-01")
_all_dates      = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 24 quarterly periods

# 2020-window diagnostic: the four rebalance dates whose HOLD period falls
# inside calendar-year 2020 (rebalance on/after 2020-01-01, before 2021-01-01).
WINDOW_2020_START = pd.Timestamp("2020-01-01")
WINDOW_2020_END   = pd.Timestamp("2021-01-01")

# ── Cost model ───────────────────────────────────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020

# ── DSR: independent trial #16 ────────────────────────────────────────────────
N_TRIALS = 16


# =============================================================================
# SECTION 2: Rate-limited HTTP
# =============================================================================

_TIINGO_LAST: float = 0.0
_SEC_LAST:    float = 0.0


def _tiingo_get(url: str, params: dict = None) -> requests.Response:
    global _TIINGO_LAST
    wait = 1.5 - (time.time() - _TIINGO_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, params=params, timeout=(10, 20))
    _TIINGO_LAST = time.time()
    return r


def _sec_get(url: str) -> requests.Response:
    global _SEC_LAST
    wait = 0.2 - (time.time() - _SEC_LAST)
    if wait > 0:
        time.sleep(wait)
    r = requests.get(url, headers={"User-Agent": SEC_UA}, timeout=(10, 30))
    _SEC_LAST = time.time()
    return r


# =============================================================================
# SECTION 3: Tiingo metadata helpers
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
        return df
    if not TIINGO_KEY:
        raise RuntimeError("No TIINGO_API_KEY and no cached tickers.csv.")
    import io, zipfile
    url  = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(name) as f:
            df_all = pd.read_csv(f)
    df_all.columns = [c.strip() for c in df_all.columns]
    df = df_all[
        df_all["exchange"].isin(US_EXCHANGES) & (df_all["assetType"] == "Stock")
    ].copy()
    for col in ["startDate", "endDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["startDate"]).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


# =============================================================================
# SECTION 4: SEC prefilter
# =============================================================================

def load_prefilter() -> pd.DataFrame:
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}. Run mgmt_pit/run.py first.")
    df = pd.read_csv(path)
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors.")
    return df


# =============================================================================
# SECTION 5: Raw price cache (return layer)
# =============================================================================

_PRICES_PKL   = CACHE / "prices_all.pkl"
_PRICE_CACHE: dict[str, pd.DataFrame] = {}
_PRICE_EMPTY  = pd.DataFrame(columns=["date", "close", "volume"])


def preload_all_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if not _PRICES_PKL.exists():
        raise FileNotFoundError(f"Missing {_PRICES_PKL}. Run research/ivol/run.py first.")
    with open(_PRICES_PKL, "rb") as f:
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


# =============================================================================
# SECTION 6: AdjClose cache (beta layer)
# =============================================================================

_ADJ_PKL      = CACHE / "adjclose_all.pkl"
_ADJ_CACHE:   dict[str, pd.DataFrame] = {}
_ADJ_SENTINEL: set[str] = set()
_ADJ_EMPTY    = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices(tickers: list[str]) -> None:
    t0 = time.time()
    if not _ADJ_PKL.exists():
        print("[AdjPreload] No pickle — beta will use raw close fallback.")
        return
    with open(_ADJ_PKL, "rb") as f:
        data = pickle.load(f)
    _ADJ_CACHE.update(data.get("adj", {}))
    _ADJ_SENTINEL.update(data.get("sentinels", set()))
    adj_loaded = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    print(f"[AdjPreload] {adj_loaded:,} tickers in {time.time()-t0:.1f}s")


def _adj_slice(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
    if df.empty:
        return _ADJ_EMPTY
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask]


# =============================================================================
# SECTION 7: SEC Accruals facts cache (S11-private, trimmed)
# =============================================================================
#
# We only need NetIncomeLoss/ProfitLoss, the CFO concepts, Assets, and
# shares-outstanding per CIK. Own trimmed cache under research/accruals/cache/
# so this trial's cache lifecycle is fully independent of any concurrently
# running trial. The raw per-CIK source (mgmt_pit/cache/sec_facts/*.json) is
# shared, read-only infrastructure — safe to read concurrently.

_AQ_FACTS_CACHE: dict[int, Optional[dict]] = {}
_AQ_FACTS_PKL   = OWN_CACHE / "accruals_facts_trimmed.json"


def preload_accruals_facts(ciks: list[int]) -> None:
    t0 = time.time()

    if _AQ_FACTS_PKL.exists():
        with open(_AQ_FACTS_PKL) as f:
            raw = json.load(f)
        for k, v in raw.items():
            _AQ_FACTS_CACHE[int(k)] = v
        missing = [c for c in ciks if c not in _AQ_FACTS_CACHE]
        if missing:
            _load_accruals_facts_from_files(missing, save=False)
        loaded = sum(1 for v in _AQ_FACTS_CACHE.values() if v is not None)
        print(f"[AQFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")
        return

    print(f"[AQFacts] First run — loading {len(ciks):,} full fact files "
          f"(this takes ~1 min)...")
    _load_accruals_facts_from_files(ciks, save=True)
    loaded = sum(1 for v in _AQ_FACTS_CACHE.values() if v is not None)
    print(f"[AQFacts] {loaded:,}/{len(ciks):,} loaded in {time.time()-t0:.1f}s")


def _load_accruals_facts_from_files(ciks: list[int], save: bool = True) -> None:
    for cik in ciks:
        p = CACHE / "sec_facts" / f"{cik}.json"
        if not p.exists():
            _AQ_FACTS_CACHE[cik] = None
            continue
        try:
            with open(p) as f:
                full = json.load(f)
            trimmed: dict = {"facts": {"us-gaap": {}}}
            ug = full.get("facts", {}).get("us-gaap", {})
            for concept in ACCRUALS_NEEDED:
                if concept in ug:
                    trimmed["facts"]["us-gaap"][concept] = ug[concept]
            _AQ_FACTS_CACHE[cik] = trimmed
        except Exception:
            _AQ_FACTS_CACHE[cik] = None
    if save:
        print("[AQFacts] Saving accruals facts trimmed cache...")
        serializable = {str(k): v for k, v in _AQ_FACTS_CACHE.items()}
        with open(_AQ_FACTS_PKL, "w") as f:
            json.dump(serializable, f)


def _get_shares_pit(facts: Optional[dict], as_of_str: str) -> Optional[float]:
    """Most recent shares outstanding where filed ≤ as_of."""
    if facts is None:
        return None
    for concept in SHARES_CONCEPTS:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        best: Optional[tuple] = None
        for entries in units_dict.values():
            for e in entries:
                filed = e.get("filed")
                end   = e.get("end")
                val   = e.get("val")
                if filed is None or end is None or val is None:
                    continue
                if filed > as_of_str:
                    continue
                key = (filed, end)
                if best is None or key > best[:2]:
                    best = (filed, end, float(val))
        if best is not None and best[2] > 0:
            return best[2]
    return None


# =============================================================================
# SECTION 8: Universe builder (identical to S-IVOL / S-MAX / S3-GP / S5)
# =============================================================================

def build_universe(
    rebalance_date: pd.Timestamp,
    tiingo_tickers: pd.DataFrame,
    cik_sic_map: dict,
) -> pd.DataFrame:
    """
    PIT investable universe at rebalance_date.
    Shares-outstanding sourced from _AQ_FACTS_CACHE (already loaded).
    """
    as_of_str     = rebalance_date.strftime("%Y-%m-%d")
    survivors_set = set(cik_sic_map.keys())
    active = tiingo_tickers[
        (tiingo_tickers["startDate"] <= rebalance_date)
        & (
            tiingo_tickers["endDate"].isna()
            | (tiingo_tickers["endDate"] >= rebalance_date)
        )
        & tiingo_tickers["ticker"].isin(survivors_set)
    ]
    rows = []
    for _, row in active.iterrows():
        ticker   = row["ticker"]
        cik, sic = cik_sic_map.get(ticker, (None, None))
        if cik is None:
            continue

        price = _last_close(ticker, rebalance_date)
        if price is None or price < PRICE_MIN:
            continue

        facts  = _AQ_FACTS_CACHE.get(cik)
        shares = _get_shares_pit(facts, as_of_str)
        if shares is None or shares <= 0:
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
    print(f"  [Universe] {rebalance_date.date()} → {len(df):,} names", end="")
    return df


# =============================================================================
# SECTION 9: Accruals signal computation (purely fundamental)
# =============================================================================

def _period_days(start_str: str, end_str: str) -> int:
    try:
        s = date.fromisoformat(start_str)
        e = date.fromisoformat(end_str)
        return (e - s).days
    except Exception:
        return 0


def _flow_all_entries_by_end(
    facts: Optional[dict],
    concept_list: list[str],
    as_of_str: str,
    min_days: int,
    max_days: int,
) -> dict:
    """
    end_date -> list of (filed, start, val, accn) for EVERY qualifying entry
    (all accessions, not just the most-recently-filed) within [min_days,
    max_days] duration, filed <= as_of. Tries each concept in concept_list
    in order; the FIRST concept with any qualifying data wins entirely
    (never mixes values across concepts for one company).

    Keeping every accession (not just the latest) is required for
    same-filing matching: NI is routinely re-disclosed as a comparative
    figure in many subsequent filings (high accn churn), while CFO is
    rarely re-disclosed at discrete-quarter granularity (10-Qs mostly
    show cumulative YTD cash flow, not discrete-quarter comparatives).
    Picking each concept's independently latest-filed accn would almost
    never land on a shared accession even when an internally-consistent
    matching pair exists earlier in the history — so we must search
    across the full accession history for a common accn instead.
    """
    if facts is None:
        return {}
    for concept in concept_list:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        by_end: dict = {}
        found_any = False
        for entries in units_dict.values():
            for e in entries:
                filed = e.get("filed")
                end   = e.get("end")
                start = e.get("start")
                val   = e.get("val")
                accn  = e.get("accn")
                if filed is None or end is None or start is None or val is None or accn is None:
                    continue
                if filed > as_of_str:
                    continue
                days = _period_days(start, end)
                if not (min_days <= days <= max_days):
                    continue
                found_any = True
                by_end.setdefault(end, []).append((filed, start, float(val), accn))
        if found_any:
            return by_end
    return {}


def _match_same_filing(
    ni_by_end: dict,
    cfo_by_end: dict,
) -> tuple:
    """
    For each period-end present in both dicts, find accession numbers
    common to BOTH NI's and CFO's entry lists for that period (i.e. a
    single filing that reported both concepts together for that period —
    the current quarter's original filing, or a later filing's comparative
    column, as long as both concepts appear under the SAME accn). Among
    common accns, the most-recently-filed one wins.

    Returns (matched_list, n_accn_mismatch): matched_list is
    [(end, start, ni_val, cfo_val, filed), ...]; n_accn_mismatch counts
    periods present in both dicts with NO common accn at all — a genuine
    restatement-timing mismatch (NI and CFO never co-reported for that
    period under one filing), correctly excluded rather than silently
    paired across inconsistent revisions.
    """
    matched = []
    n_mismatch = 0
    for end, ni_list in ni_by_end.items():
        cfo_list = cfo_by_end.get(end)
        if cfo_list is None:
            continue
        ni_by_accn  = {accn: (filed, start, val) for filed, start, val, accn in ni_list}
        cfo_by_accn = {accn: (filed, start, val) for filed, start, val, accn in cfo_list}
        common = set(ni_by_accn) & set(cfo_by_accn)
        if common:
            best_accn = max(common, key=lambda a: ni_by_accn[a][0])   # latest filed
            filed_ni, start_ni, val_ni = ni_by_accn[best_accn]
            _,        _,        val_cfo = cfo_by_accn[best_accn]
            matched.append((end, start_ni, val_ni, val_cfo, filed_ni))
        else:
            n_mismatch += 1
    return matched, n_mismatch


def _select_non_overlapping(matched: list, n_needed: int) -> list:
    """Sort by end-date descending; greedily pick non-overlapping periods."""
    sorted_m = sorted(matched, key=lambda x: x[0], reverse=True)
    selected: list = []
    for m in sorted_m:
        if len(selected) >= n_needed:
            break
        end, start = m[0], m[1]
        overlap = any(not (end <= s[1] or start >= s[0]) for s in selected)
        if not overlap:
            selected.append(m)
    return selected


def _get_ttm_accrual_components(facts: Optional[dict], as_of_str: str) -> tuple:
    """
    TTM (Net Income, Operating Cash Flow), same-filing (accn) matched.

    Returns (ttm_ni, ttm_cfo, method, n_accn_mismatch):
      method ∈ {"quarterly", "annual", None}
      n_accn_mismatch: periods where NI and CFO both existed for the same
        period-end but were filed under different accession numbers (a
        restatement-timing mismatch — excluded, not used).
    """
    if facts is None:
        return None, None, None, 0

    ni_q  = _flow_all_entries_by_end(facts, NI_CONCEPTS,  as_of_str, QTR_MIN_DAYS, QTR_MAX_DAYS)
    cfo_q = _flow_all_entries_by_end(facts, CFO_CONCEPTS, as_of_str, QTR_MIN_DAYS, QTR_MAX_DAYS)
    matched_q, n_mismatch_q = _match_same_filing(ni_q, cfo_q)
    selected = _select_non_overlapping(matched_q, TTM_QUARTERS)

    if len(selected) == TTM_QUARTERS:
        ttm_ni  = sum(s[2] for s in selected)
        ttm_cfo = sum(s[3] for s in selected)
        return ttm_ni, ttm_cfo, "quarterly", n_mismatch_q

    ni_a  = _flow_all_entries_by_end(facts, NI_CONCEPTS,  as_of_str, ANN_MIN_DAYS, ANN_MAX_DAYS)
    cfo_a = _flow_all_entries_by_end(facts, CFO_CONCEPTS, as_of_str, ANN_MIN_DAYS, ANN_MAX_DAYS)
    matched_a, n_mismatch_a = _match_same_filing(ni_a, cfo_a)
    n_mismatch = n_mismatch_q + n_mismatch_a
    if matched_a:
        best = max(matched_a, key=lambda x: (x[4], x[0]))   # most recent filed, then end
        return best[2], best[3], "annual", n_mismatch

    return None, None, None, n_mismatch


def _get_assets_pit(facts: Optional[dict], as_of_str: str) -> Optional[float]:
    """Most recent Total Assets (balance-sheet concept, filed ≤ as_of)."""
    if facts is None:
        return None
    try:
        units_dict = facts["facts"]["us-gaap"][ASSETS_CONCEPT]["units"]
    except (KeyError, TypeError):
        return None
    best: Optional[tuple] = None
    for entries in units_dict.values():
        for e in entries:
            filed = e.get("filed")
            end   = e.get("end")
            val   = e.get("val")
            if filed is None or end is None or val is None:
                continue
            if filed > as_of_str:
                continue
            key = (filed, end)
            if best is None or key > best[:2]:
                best = (filed, end, float(val))
    if best is not None and best[2] >= ASSETS_FLOOR:
        return best[2]
    return None


def compute_accruals_signals(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> tuple:
    """
    Compute total accruals = (TTM NI - TTM CFO) / Assets for each universe
    member, then apply the cross-sectional ±3-std outlier filter.

    Coverage returns:
      n_signal        : names with valid accruals_signal (post-outlier-filter)
      n_quarterly      : TTM built from 4 matched quarterly periods
      n_annual_fb      : TTM built from a single matched annual period (fallback)
      n_accn_mismatch  : periods with both NI & CFO but differing accn (excluded)
      n_no_overlap     : no NI/CFO period overlap at all (no signal possible)
      n_no_assets      : NI/CFO computed but no usable Assets denominator
      n_outliers       : signals computed but excluded by the ±3-std filter
    """
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    raw_records = []
    n_quarterly = n_annual_fb = n_no_overlap = n_no_assets = 0
    n_accn_mismatch_total = 0

    for _, row in universe_df.iterrows():
        cik   = int(row["cik"])
        facts = _AQ_FACTS_CACHE.get(cik)

        ttm_ni, ttm_cfo, method, n_mismatch = _get_ttm_accrual_components(facts, as_of_str)
        n_accn_mismatch_total += n_mismatch

        if ttm_ni is None or ttm_cfo is None:
            n_no_overlap += 1
            continue

        assets = _get_assets_pit(facts, as_of_str)
        if assets is None or assets <= 0:
            n_no_assets += 1
            continue

        if method == "quarterly":
            n_quarterly += 1
        else:
            n_annual_fb += 1

        raw_records.append({
            "ticker":           row["ticker"],
            "accruals_signal":  (ttm_ni - ttm_cfo) / assets,
            "accruals_method":  method,
        })

    raw_df = (
        pd.DataFrame(raw_records)
        if raw_records
        else pd.DataFrame(columns=["ticker", "accruals_signal", "accruals_method"])
    )

    # ── Cross-sectional outlier filter (±3 std) ─────────────────────────────
    n_outliers = 0
    if len(raw_df) >= 10:
        mu, sigma = raw_df["accruals_signal"].mean(), raw_df["accruals_signal"].std(ddof=1)
        if sigma > 0:
            z = (raw_df["accruals_signal"] - mu) / sigma
            outlier_mask = z.abs() > OUTLIER_Z_THRESH
            n_outliers = int(outlier_mask.sum())
            signals_df = raw_df.loc[~outlier_mask].reset_index(drop=True)
        else:
            signals_df = raw_df
    else:
        signals_df = raw_df

    coverage = {
        "n_universe":       len(universe_df),
        "n_signal":         len(signals_df),
        "n_quarterly":      n_quarterly,
        "n_annual_fb":      n_annual_fb,
        "n_accn_mismatch":  n_accn_mismatch_total,
        "n_no_overlap":     n_no_overlap,
        "n_no_assets":      n_no_assets,
        "n_outliers":       n_outliers,
        "pct_dropped":      100 * (len(universe_df) - len(signals_df)) / max(len(universe_df), 1),
    }
    return signals_df, coverage


# =============================================================================
# SECTION 10: Beta computation (60-day EW adjClose market model)
# =============================================================================

def compute_betas(
    universe_df: pd.DataFrame,
    rebalance_date: pd.Timestamp,
) -> dict:
    """
    Compute per-ticker 60-day OLS beta vs EW universe return.
    Returns dict: ticker → beta (np.nan if insufficient data).
    """
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)

    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj    = _adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            raw_df = _PRICE_CACHE.get(ticker, _ADJ_EMPTY)
            if raw_df.empty:
                continue
            raw_w = raw_df[(raw_df["date"] >= beta_start)
                           & (raw_df["date"] <= rebalance_date)]
            if raw_w.empty:
                continue
            s = raw_w.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()
        else:
            s   = adj.set_index("date")["adjClose"]
            s   = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()

        if len(ret) >= BETA_MIN_DAYS:
            adj_series[ticker] = ret

    if not adj_series:
        return {}

    ret_mat = pd.DataFrame(adj_series).sort_index()
    ew_mkt  = ret_mat.clip(-0.5, 0.5).mean(axis=1)

    betas: dict[str, float] = {}
    for ticker, ret_s in adj_series.items():
        y_s = ret_s
        x_s = ew_mkt.reindex(y_s.index).dropna()
        y_s = y_s.reindex(x_s.index).dropna()
        x_s = x_s.reindex(y_s.index)
        n   = len(y_s)
        if n < BETA_MIN_DAYS:
            betas[ticker] = np.nan
            continue
        X = np.column_stack([np.ones(n), x_s.values])
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(X, y_s.values, rcond=None)
            betas[ticker] = float(coeffs[1]) if rank >= 2 else np.nan
        except Exception:
            betas[ticker] = np.nan

    return betas


# =============================================================================
# SECTION 11: Return calculation (raw close, survivorship-bias-free)
# =============================================================================

def compute_returns(
    universe_df: pd.DataFrame,
    hold_start: pd.Timestamp,
    hold_end: pd.Timestamp,
    delisted_map: dict,
) -> pd.DataFrame:
    """
    Entry: first close on or after hold_start.
    Exit : last close on or before hold_end (or delist_date if earlier).
    """
    df = universe_df.copy()
    entry_prices, exit_prices, raw_returns, delisted_flags = [], [], [], []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        prices = _get_price_slice(ticker, hold_start, hold_end + pd.Timedelta(days=5))

        if prices.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan);  delisted_flags.append(False)
            continue

        entry_cands = prices[prices["date"] >= hold_start]
        if entry_cands.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan);  delisted_flags.append(False)
            continue

        entry_price = float(entry_cands.iloc[0]["close"])
        delist_date = delisted_map.get(ticker, pd.NaT)
        is_delisted = pd.notna(delist_date) and (delist_date < hold_end)

        if is_delisted:
            exit_cands  = prices[prices["date"] <= delist_date]
            exit_price  = float(
                (exit_cands if not exit_cands.empty else prices).iloc[-1]["close"])
            delisted_flags.append(True)
        else:
            exit_cands  = prices[prices["date"] <= hold_end]
            exit_price  = float(exit_cands.iloc[-1]["close"]) if not exit_cands.empty else entry_price
            delisted_flags.append(False)

        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        raw_returns.append(exit_price / entry_price - 1)

    df["entry_price"] = entry_prices
    df["exit_price"]  = exit_prices
    df["raw_return"]  = raw_returns
    df["delisted"]    = delisted_flags
    return df


# =============================================================================
# SECTION 12: Portfolio construction + cost model
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame, betas: dict) -> tuple:
    """
    Long BOTTOM 20% accruals (lowest/most cash-backed — long leg), short
    TOP 20% (highest accruals — short leg). Min structural leg size 20 via
    quintiles; if fewer than 20 names would land in a quintile leg, widen to
    TERCILES for this period only (name-count guard, logged not a new trial).
    """
    df = signals_df.dropna(subset=["accruals_signal", "raw_return"]).copy()
    n  = len(df)
    if n < 6:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, False

    leg_quintile = int(np.ceil(n * 0.20))
    guard_fired  = leg_quintile < 20
    leg          = int(np.ceil(n / 3.0)) if guard_fired else leg_quintile
    leg          = min(leg, n // 2)
    if leg < 1:
        return [], [], {}, {}, 1.0, 1.0, np.nan, np.nan, guard_fired

    ranked = df.sort_values("accruals_signal", ascending=True)   # low → high
    longs  = ranked.iloc[:leg]["ticker"].tolist()    # LOW accruals  = long
    shorts = ranked.iloc[-leg:]["ticker"].tolist()   # HIGH accruals = short

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    mean_beta_long  = float(np.nanmean([betas.get(t, np.nan) for t in longs]))
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return longs, shorts, lw, sw, long_scale, short_scale, mean_beta_long, mean_beta_short, guard_fired


def _leg_return_detail(
    returns_df: pd.DataFrame,
    tickers: list,
    weights: dict,
    is_short: bool,
    turnover: float,
) -> tuple:
    """Quarterly cost model: borrow /4, bid-ask x actual turnover."""
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (MCAP_MIN + MCAP_MAX) / 2

    gross = ba_cost = borrow_cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross   += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / 4.0) * w

    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    else:
        return ( gross - total_cost),   gross,  ba_cost, borrow_cost


# =============================================================================
# SECTION 13: Backtest engine
# =============================================================================

def run_backtest(
    tiingo_tickers: pd.DataFrame,
    prefilter_df: pd.DataFrame,
) -> dict:
    t_total = time.time()

    survivors   = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list   = []
    coverage_records: list = []
    universe_audit: list   = []
    sector_contrib: dict   = {}   # sic major group -> cumulative $ contribution
    sign_violations: int   = 0
    guard_fire_count: int  = 0

    prev_long_set:  set = set()
    prev_short_set: set = set()

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        hold_end = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END

        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} → {hold_end.date()}")

        # ── 1. Universe ───────────────────────────────────────────────────
        try:
            univ = build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        # ── 2. Accruals signal (fundamental — no price layer) ──────────────
        signals_df, cov = compute_accruals_signals(univ, t)
        n_dropped = cov["n_universe"] - cov["n_signal"]
        pct_drop  = cov["pct_dropped"]
        fallback_frac = 100 * cov["n_annual_fb"] / max(cov["n_signal"], 1)
        print(f"  ACC: {cov['n_signal']}/{cov['n_universe']} with signal "
              f"({pct_drop:.0f}% dropped) | qtrly={cov['n_quarterly']} "
              f"annualFB={cov['n_annual_fb']} ({fallback_frac:.0f}% of signal) "
              f"accnMismatch={cov['n_accn_mismatch']} noOverlap={cov['n_no_overlap']} "
              f"noAssets={cov['n_no_assets']} outliers={cov['n_outliers']}")
        if fallback_frac > 25:
            print(f"  *** WARNING: annual-fallback TTM used for "
                  f"{fallback_frac:.0f}% of signals (>25% threshold). ***")

        coverage_records.append({"rebalance_date": t, **cov})

        # ── 3. Merge signals; compute hold-period returns ───────────────────
        univ_sig = univ.merge(signals_df[["ticker", "accruals_signal"]], on="ticker", how="left")
        try:
            returns_df = compute_returns(univ_sig, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        n_delisted = int(returns_df["delisted"].sum())

        universe_audit.append({
            "rebalance_date":  t,
            "hold_end":        hold_end,
            "n_universe":      len(returns_df),
            "n_with_signal":   int(returns_df["accruals_signal"].notna().sum()),
            "n_delisted":      n_delisted,
            "n_valid_returns": int(returns_df["raw_return"].notna().sum()),
        })

        # ── 4. Beta computation (60-day EW adjClose) ────────────────────────
        betas = compute_betas(univ, t)

        # ── 5. Portfolio (beta-neutral, name-count guard) ───────────────────
        (longs, shorts, lw, sw,
         long_scale, short_scale,
         mean_beta_long, mean_beta_short, guard_fired) = build_portfolio(returns_df, betas)
        if not longs:
            print(f"  [SKIP] n<6 after signal filter")
            continue
        if guard_fired:
            guard_fire_count += 1
            print(f"  [GUARD] quintile leg <20 names → widened to terciles "
                  f"(leg={len(longs)})")

        # ── 5b. Sign-convention audit ────────────────────────────────────────
        mean_acc_long  = float(signals_df.set_index("ticker")["accruals_signal"]
                                 .reindex(longs).mean())
        mean_acc_short = float(signals_df.set_index("ticker")["accruals_signal"]
                                 .reindex(shorts).mean())
        sign_ok = mean_acc_long < mean_acc_short
        if not sign_ok:
            sign_violations += 1
            print(f"  *** SIGN VIOLATION: long-leg accruals ({mean_acc_long:.3f}) "
                  f">= short-leg accruals ({mean_acc_short:.3f}) ***")

        # ── 6. Turnover ───────────────────────────────────────────────────────
        if prev_long_set:
            long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1)
            short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0

        prev_long_set  = set(longs)
        prev_short_set = set(shorts)
        avg_turnover   = (long_turnover + short_turnover) / 2.0

        # ── 7. Leg returns (quarterly cost model) ─────────────────────────────
        long_net,  long_gross,  long_ba,  long_bw  = _leg_return_detail(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover)

        ls_net   = long_scale * long_net   + short_scale * short_net
        ls_gross = long_scale * long_gross + short_scale * short_gross
        ls_ba    = long_scale * long_ba    + short_scale * short_ba
        ls_bw    = long_scale * long_bw    + short_scale * short_bw
        ls_cost  = ls_ba + ls_bw

        # ── 7b. Sector contribution (concentration check) ────────────────────
        sic_map  = returns_df.set_index("ticker")["sic"].to_dict()
        ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
        for tk in longs:
            r = ret_map.get(tk, np.nan)
            if np.isnan(r):
                continue
            sic = sic_map.get(tk, np.nan)
            grp = int(sic // 100) if pd.notna(sic) else -1
            sector_contrib[grp] = sector_contrib.get(grp, 0.0) + long_scale * lw.get(tk, 0.0) * r
        for tk in shorts:
            r = ret_map.get(tk, np.nan)
            if np.isnan(r):
                continue
            sic = sic_map.get(tk, np.nan)
            grp = int(sic // 100) if pd.notna(sic) else -1
            sector_contrib[grp] = sector_contrib.get(grp, 0.0) - short_scale * sw.get(tk, 0.0) * r

        # ── 8. EW market return ───────────────────────────────────────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret   = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        # ── 9. IC: Spearman(accruals_signal, realized_return) ─────────────────
        # IC < 0 expected: high accruals -> low subsequent return (hypothesis).
        valid_sig = returns_df.dropna(subset=["accruals_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["accruals_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} "
              f"to={avg_turnover:.0%} β_L={mean_beta_long:.2f} β_S={mean_beta_short:.2f} "
              f"IC={ic:.3f} Dlist={n_delisted} accL={mean_acc_long:.3f} accS={mean_acc_short:.3f}")

        period_records.append({
            "rebalance_date":   t,
            "hold_end":         hold_end,
            "n_universe":       len(returns_df),
            "n_with_signal":    int(returns_df["accruals_signal"].notna().sum()),
            "n_delisted":       n_delisted,
            "pct_dropped":      pct_drop,
            "long_ret":         long_net,
            "short_ret":        short_net,
            "ls_ret":           ls_net,
            "ls_gross":         ls_gross,
            "ls_cost":          ls_cost,
            "ls_ba_cost":       ls_ba,
            "ls_borrow_cost":   ls_bw,
            "long_scale":       long_scale,
            "short_scale":      short_scale,
            "mean_beta_long":   mean_beta_long,
            "mean_beta_short":  mean_beta_short,
            "long_turnover":    long_turnover,
            "short_turnover":   short_turnover,
            "avg_turnover":     avg_turnover,
            "mkt_ret":          mkt_ret,
            "ic":               ic,
            "n_long":           len(longs),
            "n_short":          len(shorts),
            "guard_fired":      guard_fired,
            "mean_acc_long":    mean_acc_long,
            "mean_acc_short":   mean_acc_short,
            "sign_ok":          sign_ok,
        })

    elapsed = time.time() - t_total
    return {
        "period_records":   period_records,
        "universe_audit":   universe_audit,
        "coverage_records": coverage_records,
        "sector_contrib":   sector_contrib,
        "sign_violations":  sign_violations,
        "guard_fire_count": guard_fire_count,
        "elapsed_s":        elapsed,
    }


# =============================================================================
# SECTION 14: Metrics
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
    """Quarterly series → annualized (4 periods/year)."""
    ls = ls_series.dropna().reset_index(drop=True)
    n  = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "max_dd": np.nan, "t_stat": np.nan,
                "win_rate": np.nan, "n": n}

    cum     = (1 + ls).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (4.0 / n) - 1)
    mu      = float(ls.mean())
    sigma   = float(ls.std(ddof=1))
    sharpe  = float((mu / sigma) * np.sqrt(4)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s    = (cum - roll_mx) / roll_mx
    max_dd  = float(dd_s.min())
    calmar  = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat  = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt  = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "t_stat": t_stat,
            "win_rate": win_rt, "n": n}


def compute_market_beta(ls_returns: pd.Series, mkt_returns: pd.Series) -> dict:
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < 5:
        return {"beta": np.nan, "beta_t": np.nan}
    x = combined["mkt"].values
    y = combined["ls"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"beta": np.nan, "beta_t": np.nan}
    beta   = float(coeffs[1])
    y_hat  = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n      = len(y)
    se     = (np.sqrt(ss_res / max(n - 2, 1))
              / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    """Per-period SR threshold (Bailey & Lopez de Prado 2014)."""
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# SECTION 15: Output & registry
# =============================================================================

def print_results(results: dict) -> None:
    records  = results["period_records"]
    coverage = results["coverage_records"]
    elapsed  = results["elapsed_s"]

    if not records:
        print("\n[ERROR] No periods completed.")
        return

    df   = pd.DataFrame(records)
    cov  = pd.DataFrame(coverage)

    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()

    m          = compute_metrics(ls_series)
    m_gross    = compute_metrics(gross_series)
    beta_stats = compute_market_beta(ls_series, mkt_series)

    per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
    dsr_thr       = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr    = (bool(per_period_sr > dsr_thr)
                     if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False)

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    mean_monthly_ba_bps = float(df["ls_ba_cost"].mean())   * 4 * 10_000
    mean_monthly_bw_bps = float(df["ls_borrow_cost"].mean()) * 4 * 10_000
    annual_cost_drag    = (mean_monthly_ba_bps + mean_monthly_bw_bps)
    mean_turnover       = float(df["avg_turnover"].mean())
    mean_long_scale     = float(df["long_scale"].mean())
    mean_short_scale    = float(df["short_scale"].mean())
    mean_bl             = float(df["mean_beta_long"].mean())
    mean_bs             = float(df["mean_beta_short"].mean())
    mean_drop           = float(df["pct_dropped"].mean())

    guard_fire_count = int(results["guard_fire_count"])
    sign_violations  = int(results["sign_violations"])
    n_periods_run    = len(df)

    # Coverage summary across all periods
    total_univ    = int(cov["n_universe"].sum())
    total_sig     = int(cov["n_signal"].sum())
    total_qtrly   = int(cov["n_quarterly"].sum())
    total_annfb   = int(cov["n_annual_fb"].sum())
    total_mismatch = int(cov["n_accn_mismatch"].sum())
    total_outliers = int(cov["n_outliers"].sum())
    overall_drop  = 100 * (total_univ - total_sig) / max(total_univ, 1)
    overall_fb_frac = 100 * total_annfb / max(total_sig, 1)

    sep = "=" * 78

    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT")
    print(sep)
    print(f"  Periods where long-leg accruals >= short-leg accruals (violation): "
          f"{sign_violations}/{n_periods_run}")
    if sign_violations == 0:
        print("  PASS — long leg is lowest-accruals every period, as intended.")
    else:
        print("  *** FAIL — sign convention broken in at least one period. "
              "Results below may be inverted. ***")

    print(f"\n{sep}")
    print("ACCRUALS COVERAGE AUDIT (NI/CFO same-filing TTM match)")
    print(sep)
    print(f"  Total stock-period slots (universe × quarters):  {total_univ:,}")
    print(f"  With valid accruals signal:                      {total_sig:,} ({100*total_sig/max(total_univ,1):.1f}%)")
    print(f"    — TTM from 4 matched quarterly periods:         {total_qtrly:,}")
    print(f"    — TTM from matched annual period (fallback):    {total_annfb:,} "
          f"({overall_fb_frac:.1f}% of signals)")
    print(f"  Accn mismatches excluded (NI/CFO restated separately): {total_mismatch:,}")
    print(f"  Outliers excluded (±{OUTLIER_Z_THRESH:.0f} std cross-sectional):  {total_outliers:,}")
    print(f"  Dropped (no signal):                              {total_univ-total_sig:,} ({overall_drop:.1f}%)")
    if overall_fb_frac > 25:
        print(f"\n  *** WARNING: annual-fallback TTM used for {overall_fb_frac:.0f}% of signals "
              f"(>25% threshold — IVOL/S5 data-quality precedent). ***")
    else:
        print(f"\n  Fallback usage acceptable ({overall_fb_frac:.1f}% < 25% threshold).")

    print(f"\n  Name-count guard (quintile leg <20 -> widen to terciles) fired: "
          f"{guard_fire_count}/{n_periods_run} periods "
          f"({100*guard_fire_count/max(n_periods_run,1):.0f}%)")

    print(f"\n{sep}")
    print("S11: ACCRUALS QUALITY (SLOAN 1996) — IN-SAMPLE RESULTS  (independent trial #16)")
    print("Signal: Total Accruals = (TTM NetIncome - TTM CFO) / Assets  (filed <= t, same-accn matched)")
    print("Long BOTTOM 20% accruals | Short TOP 20% accruals | Beta-neutral | Quarterly")
    print(sep)

    print(f"\n  *** INFORMATION COEFFICIENT (cost-independent signal quality) ***")
    print(f"  Mean IC (Spearman, accruals vs ret)    : {mean_ic:>12.4f}")
    print(f"  IC t-stat                              : {ic_tstat:>12.3f}")
    print(f"  IC < 0 = high accruals underperformed (hypothesis confirmed).")
    print(f"  |IC| t > 2 = signal has statistically significant predictive power.")

    print(f"\n  {'Metric':<46} {'Value':>14}")
    print(f"  {'-'*46} {'-'*14}")
    print(f"  {'Total cumulative return (net)':<46} {m['total_ret']:>13.2%}")
    print(f"  {'Annualized return (net)':<46} {m['ann_ret']:>13.2%}")
    print(f"  {'Annualized return (GROSS)':<46} {m_gross['ann_ret']:>13.2%}")
    print(f"  {'Cost drag (annualized bps)':<46} {annual_cost_drag:>13.0f}")
    print(f"    {'  bid-ask (bps/yr)':<44} {mean_monthly_ba_bps:>13.0f}")
    print(f"    {'  borrow  (bps/yr)':<44} {mean_monthly_bw_bps:>13.0f}")
    print(f"  {'Mean actual turnover per quarter':<46} {mean_turnover:>13.0%}")
    print(f"  {'  (reference: GP/A turnover ≈ 23%/quarter)':<46} {'':>13}")
    print(f"  {'Sharpe ratio (annualized, net)':<46} {m['sharpe']:>13.3f}")
    print(f"  {'CALMAR ratio':<46} {m['calmar']:>13.3f}")
    print(f"  {'Max drawdown':<46} {m['max_dd']:>13.2%}")
    print(f"  {'t-stat (mean quarterly net return)':<46} {m['t_stat']:>13.3f}")
    print(f"  {'Market beta (vs EW universe)':<46} {beta_stats['beta']:>13.3f}")
    print(f"    {'beta t-stat (should be ~0)':<44} {beta_stats['beta_t']:>13.3f}")
    print(f"  {'Mean gross long exposure ($)':<46} {mean_long_scale:>13.3f}")
    print(f"  {'Mean gross short exposure ($)':<46} {mean_short_scale:>13.3f}")
    print(f"  {'Mean ex-ante beta long leg':<46} {mean_bl:>13.3f}")
    print(f"  {'Mean ex-ante beta short leg':<46} {mean_bs:>13.3f}")
    print(f"  {'Win rate':<46} {m['win_rate']:>13.1%}")
    print(f"  {'Periods (N)':<46} {m['n']:>13d}")
    print(f"  {'Mean universe names dropped (%)':<46} {mean_drop:>12.1f}%")

    print(f"\n  --- DSR Check (N_trials={N_TRIALS}, quarterly, per-period SR units) ---")
    print(f"  {'Per-period Sharpe':<46} {per_period_sr:>13.4f}")
    print(f"  {'DSR threshold (per-quarter SR)':<46} {dsr_thr:>13.4f}")
    print(f"  {'Annualized Sharpe threshold':<46} {dsr_thr*2:>13.4f}")
    print(f"  {'Clears DSR':<46} {'YES' if clears_dsr else 'NO':>13}")

    print(f"\n  Raw quarterly returns (compute alpha vs T-bill Rf externally):")
    print(f"  Mean quarterly L/S (net) : {ls_series.mean():.4%}")
    print(f"  Std  quarterly L/S (net) : {ls_series.std(ddof=1):.4%}")

    # ── Concentration check: single-quarter contribution ──────────────────────
    cum_final   = float((1 + ls_series.reset_index(drop=True)).cumprod().iloc[-1] - 1)
    abs_period_sum = float(ls_series.abs().sum())
    print(f"\n{sep}")
    print("CONCENTRATION CHECK (single-episode artifact risk)")
    print(sep)
    period_share = (ls_series.abs() / abs_period_sum) if abs_period_sum > 0 else ls_series * 0
    top_idx   = period_share.idxmax()
    top_share = float(period_share.loc[top_idx])
    top_date  = df.loc[top_idx, "rebalance_date"]
    print(f"  Largest single-quarter share of total |return| mass: "
          f"{top_date.date()} = {top_share:.1%}")
    if top_share > 0.30:
        print(f"  *** WARNING: single quarter drives >{30:.0f}% of return mass — "
              f"possible single-episode artifact. ***")
    else:
        print(f"  No single quarter dominates (<30% threshold).")

    # ── Concentration check: sector contribution ───────────────────────────────
    sector_contrib = results.get("sector_contrib", {})
    if sector_contrib:
        total_abs = sum(abs(v) for v in sector_contrib.values())
        if total_abs > 0:
            top_sic, top_val = max(sector_contrib.items(), key=lambda kv: abs(kv[1]))
            top_sec_share = abs(top_val) / total_abs
            print(f"  Largest single SIC major-group (2-digit) contribution: "
                  f"SIC {top_sic:02d} = {top_sec_share:.1%} of total |contribution|")
            if top_sec_share > 0.40:
                print(f"  *** WARNING: single sector drives >{40:.0f}% of return — "
                      f"possible sector concentration artifact. ***")
            else:
                print(f"  No single sector dominates (<40% threshold).")

    # ── 2020-window diagnostic ──────────────────────────────────────────────
    print(f"\n{sep}")
    print("2020-WINDOW DIAGNOSTIC (S1/S2/S7/E2 precedent — check before, not after)")
    print(sep)
    in_2020  = df[(df["rebalance_date"] >= WINDOW_2020_START) & (df["rebalance_date"] < WINDOW_2020_END)]
    out_2020 = df[~df.index.isin(in_2020.index)]
    if not in_2020.empty:
        ret_2020     = float(in_2020["ls_ret"].sum())
        share_2020   = float(in_2020["ls_ret"].abs().sum() / abs_period_sum) if abs_period_sum > 0 else np.nan
        mean_2020    = float(in_2020["ls_ret"].mean())
        mean_out     = float(out_2020["ls_ret"].mean()) if not out_2020.empty else np.nan
        print(f"  2020 rebalance dates (n={len(in_2020)}): "
              f"{[str(d.date()) for d in in_2020['rebalance_date']]}")
        print(f"  Sum of 2020 quarterly L/S returns          : {ret_2020:>10.2%}")
        print(f"  2020 share of total |return| mass           : {share_2020:>10.1%}")
        print(f"  Mean quarterly L/S return, 2020              : {mean_2020:>10.2%}")
        print(f"  Mean quarterly L/S return, ex-2020            : {mean_out:>10.2%}")
        if share_2020 > 0.40:
            print(f"  *** WARNING: 2020 drives >{40:.0f}% of total |return| mass — "
                  f"result may be a 2020-specific artifact, not a persistent signal. ***")
        else:
            print(f"  2020 does not disproportionately dominate (<40% threshold).")
    else:
        print("  No rebalance dates fall in the 2020 window.")

    # Quarterly returns log
    print(f"\n{sep}")
    print("QUARTERLY RETURNS LOG")
    print(f"  {'Date':<11} {'Net':>7} {'Gross':>7} {'Cost':>6} {'TOver':>6} "
          f"{'LScl':>5} {'SScl':>5} {'Mkt':>7} {'IC':>6} {'nL':>3} {'Drop%':>5} {'Guard':>5}")
    print(f"  {'-'*11} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*5} {'-'*5} "
          f"{'-'*7} {'-'*6} {'-'*3} {'-'*5} {'-'*5}")
    for _, r in df.iterrows():
        print(f"  {str(r['rebalance_date'].date()):<11} "
              f"{r['ls_ret']:>6.2%} {r['ls_gross']:>6.2%} {r['ls_cost']:>5.2%} "
              f"{r['avg_turnover']:>5.0%} "
              f"{r['long_scale']:>4.2f}x {r['short_scale']:>4.2f}x "
              f"{r['mkt_ret']:>6.2%} {r['ic']:>6.3f} "
              f"{int(r['n_long']):>3} {r['pct_dropped']:>4.0f}% "
              f"{'Y' if r['guard_fired'] else '':>5}")

    # Drawdown series
    cum      = (1 + ls_series.reset_index(drop=True)).cumprod()
    roll_max = cum.cummax()
    dd_s     = (cum - roll_max) / roll_max
    print(f"\n  Drawdown series (quarterly):")
    for i, (dt, val) in enumerate(zip(df["rebalance_date"], dd_s)):
        marker = " <- MAX DD" if abs(val - m["max_dd"]) < 1e-9 else ""
        print(f"    {str(dt.date()):<11}  {val:>7.2%}{marker}")

    print(f"\n  Runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")


def save_results(results: dict) -> None:
    records  = results["period_records"]
    audit    = results["universe_audit"]
    coverage = results["coverage_records"]
    if not records:
        return
    pd.DataFrame(records).to_csv(RESULTS / "accruals_period_returns.csv", index=False)
    pd.DataFrame(audit).to_csv(RESULTS / "accruals_universe_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(RESULTS / "accruals_coverage_audit.csv", index=False)
    sector_df = pd.DataFrame(
        [{"sic_major_group": k, "contribution": v}
         for k, v in results.get("sector_contrib", {}).items()]
    ).sort_values("contribution", ascending=False)
    sector_df.to_csv(RESULTS / "accruals_sector_contribution.csv", index=False)
    print(f"\n[Saved] Results → {RESULTS}/")


def log_registry(results: dict) -> None:
    records = results["period_records"]
    if not records:
        return

    df           = pd.DataFrame(records)
    ls_series    = df["ls_ret"]
    gross_series = df["ls_gross"]
    mkt_series   = df["mkt_ret"]
    ic_series    = df["ic"].dropna()
    m            = compute_metrics(ls_series)
    m_gross      = compute_metrics(gross_series)
    beta_stats   = compute_market_beta(ls_series, mkt_series)

    per_period_sr = float(ls_series.mean() / ls_series.std(ddof=1)) if ls_series.std(ddof=1) > 0 else np.nan
    dsr_thr       = deflated_sharpe_threshold(N_TRIALS, m["n"])
    clears_dsr    = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) else False

    mean_ic  = float(ic_series.mean()) if len(ic_series) > 0 else np.nan
    ic_tstat = (
        ic_series.mean() / (ic_series.std(ddof=1) / np.sqrt(len(ic_series)))
        if len(ic_series) > 1 and ic_series.std(ddof=1) > 0 else np.nan
    )

    annual_cost_drag = (float(df["ls_ba_cost"].mean()) + float(df["ls_borrow_cost"].mean())) * 4 * 10_000
    cov = pd.DataFrame(results["coverage_records"])
    total_univ  = int(cov["n_universe"].sum())
    total_sig   = int(cov["n_signal"].sum())
    total_annfb = int(cov["n_annual_fb"].sum())
    overall_fb_frac = 100 * total_annfb / max(total_sig, 1)

    sign_violations  = int(results["sign_violations"])
    guard_fire_count = int(results["guard_fire_count"])
    n_periods_run    = len(df)

    in_2020 = df[(df["rebalance_date"] >= WINDOW_2020_START) & (df["rebalance_date"] < WINDOW_2020_END)]
    abs_period_sum = float(ls_series.abs().sum())
    share_2020 = float(in_2020["ls_ret"].abs().sum() / abs_period_sum) if abs_period_sum > 0 and not in_2020.empty else np.nan

    verdict_bits = []
    verdict_bits.append("CLEARS DSR" if clears_dsr else "FAILS DSR")
    if sign_violations > 0:
        verdict_bits.append(f"SIGN AUDIT FAILED ({sign_violations} periods)")
    if overall_fb_frac > 25:
        verdict_bits.append(f"FALLBACK USAGE {overall_fb_frac:.0f}% (>25% threshold)")
    if not np.isnan(share_2020) and share_2020 > 0.40:
        verdict_bits.append(f"2020-DOMINATED ({share_2020:.0%} of |return| mass)")
    mean_turnover = float(df["avg_turnover"].mean())
    if not clears_dsr and mean_ic is not None and not np.isnan(mean_ic) and mean_ic < 0 and m_gross["ann_ret"] > 0:
        verdict_bits.append(
            f"real IC but cost-killed — turnover {mean_turnover:.0%}/qtr "
            f"vs GP/A's ~23%/qtr"
        )
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp":            datetime.now().isoformat(),
        "study":                "accruals_S11",
        "hypothesis":           (
            "Total accruals (Sloan 1996): (TTM NetIncome - TTM OperatingCashFlow) "
            "/ Assets. High-accruals firms (earnings driven by accounting "
            "adjustments) subsequently underperform; low-accruals firms "
            "(cash-backed earnings) subsequently outperform. Long bottom "
            "quintile (lowest accruals), short top quintile (highest accruals). "
            "Independent family — shares GP/A's slow-moving fundamental-only "
            "structural profile but tests earnings QUALITY, not LEVEL (GP/A) "
            "or balance-sheet GROWTH (S5)."
        ),
        "data_source":          "Tiingo (raw close for returns) + SEC XBRL (NetIncomeLoss, CFO, Assets tags, PIT, same-accn matched)",
        "status":               "completed_in_sample",
        "trial_number":         16,
        "trial_note":           "Independent trial #16. Not a family member of any prior trial.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "quarterly",
        "rebalance_note":       "Pre-registered single frequency, no sweep (S3 lesson).",
        "portfolio_note":       (
            f"Beta-neutral (EW market proxy, 60-day adjClose). Name-count guard "
            f"(quintile leg<20 -> terciles) fired {guard_fire_count}/{n_periods_run} periods."
        ),
        "signal_note":          (
            "Accruals = (TTM NetIncomeLoss - TTM CFO)/Assets, filed <= t, NI/CFO "
            "matched by SEC accession number per period (same-filing gating); "
            f"annual-TTM-fallback used for {overall_fb_frac:.1f}% of signals."
        ),
        "ann_return_net":       m["ann_ret"],
        "ann_return_gross":     m_gross["ann_ret"],
        "annual_cost_drag_bps": annual_cost_drag,
        "mean_turnover":        mean_turnover,
        "sharpe":               m["sharpe"],
        "calmar":               m["calmar"],
        "max_drawdown":         m["max_dd"],
        "t_stat":               m["t_stat"],
        "market_beta":          beta_stats["beta"],
        "market_beta_tstat":    beta_stats["beta_t"],
        "mean_long_scale":      float(df["long_scale"].mean()),
        "mean_short_scale":     float(df["short_scale"].mean()),
        "mean_beta_long":       float(df["mean_beta_long"].mean()),
        "mean_beta_short":      float(df["mean_beta_short"].mean()),
        "mean_ic":              mean_ic,
        "ic_t_stat":            ic_tstat,
        "win_rate":             m["win_rate"],
        "n_periods":            m["n"],
        "per_period_sr":        per_period_sr,
        "dsr_threshold":        dsr_thr,
        "clears_dsr":           clears_dsr,
        "mean_pct_dropped":     float(df["pct_dropped"].mean()),
        "adj_fallback_pct":     overall_fb_frac,
        "notes":                verdict,
    }

    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "accruals_S11"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] accruals_S11 logged (trial #16) → {REGISTRY}")
    print(f"[Verdict] {verdict}")


# =============================================================================
# SECTION 16: Main
# =============================================================================

def main() -> None:
    print("=" * 78)
    print("S11: ACCRUALS QUALITY (SLOAN 1996)  (independent trial #16)")
    print("Signal: Total Accruals = (TTM NetIncome - TTM CFO)/Assets (SEC XBRL, PIT, same-accn matched)")
    print("Long BOTTOM 20% accruals | Short TOP 20% accruals | Beta-neutral")
    print("Quarterly rebalance | In-sample: 2015-01-01 -> 2021-01-01 | 24 periods")
    print(f"DSR: N_TRIALS={N_TRIALS} | independent family, not folded into any prior trial")
    print("=" * 78)

    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = load_tiingo_tickers()

    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = load_prefilter()
    survivors        = prefilter_df[prefilter_df["passed"]]
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 3b] Loading adjClose pickle (beta layer)...")
    preload_all_adj_prices(survivor_tickers)

    print(f"\n[Step 4] Loading Accruals XBRL facts ({len(survivor_ciks):,} CIKs)...")
    preload_accruals_facts(survivor_ciks)

    print(f"\n[Step 5] Running backtest ({len(REBALANCE_DATES)} quarterly periods)...")
    results = run_backtest(tiingo_tickers, prefilter_df)

    print_results(results)
    save_results(results)
    log_registry(results)


if __name__ == "__main__":
    main()
