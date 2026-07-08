"""
S2: Post-Earnings Announcement Drift (PEAD)  —  independent trial #11
======================================================================
Signal (surprise proxy — no analyst estimates, small-cap appropriate):
  Surprise = 3-day abnormal adjClose return around the SEC filing date.
  Abnormal = stock's 3-day adjClose return minus the EW-universe 3-day
  return over the same window  [D-1, D0, D+1].

  Filing date = earliest PIT earnings date available:
    1. 8-K Item 2.02 date from sec_subs (earnings release)
    2. 10-Q / 10-K filing date from sec_facts (fallback)
  Amended filings (/A) ignored.  Restated dates never used.

Look-ahead discipline:
  - Entry strictly at D+2 (first close after 3-day window closes).
    No price from on or after entry date is used in the surprise.
  - All filing dates are PIT (filed ≤ D before using).
  - adjClose used for surprise and CAAR (avoids split artifacts, S9 lesson).
  - Raw close used for hold-period P&L and delisting returns.

Portfolio (calendar-time, beta-neutral):
  - Hold period: 45 trading days (pre-registered, not swept).
  - Long = names in top-quintile surprise within trailing 60-cal-day window;
    short = bottom quintile.  Name-count guard: widen to tercile if <20 per leg.
  - Beta: 60-day OLS vs EW universe return at time of filing.
  - Equal-weight within each leg; leg-level beta-neutral scaling.

Cost model: bid-ask by cap tier (20–100bps) + borrow on shorts (0.5–2%/yr).
  Bid-ask: paid at entry and exit.  Borrow: accrued per trading day on short leg.

DSR: N_TRIALS=11.  n_obs = monthly periods in IS window ≈ 72.
  Threshold = norm.ppf(1 − 1/11) / sqrt(72) ≈ 0.157 per-month SR.
  Annualized: × sqrt(12) ≈ 0.545.

CAAR diagnostic: cumulative avg abnormal return (adjClose) by quintile over
45 post-entry days.  Printed as text table.  If top-quintile doesn't drift
up and bottom down, the drift mechanism is absent — reported as such.
"""

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import date, timedelta
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
CACHE    = MGMT_PIT / "cache"
RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"

RESULTS.mkdir(parents=True, exist_ok=True)

# ── Universe filters (same as all prior experiments) ─────────────────────────
MCAP_MIN       = 100e6
MCAP_MAX       = 2e9
DOLLAR_VOL_MIN = 1e6
PRICE_MIN      = 1.0
US_EXCHANGES   = {"NYSE", "NASDAQ", "AMEX"}

# ── PEAD signal parameters (pre-registered, never tuned) ─────────────────────
HOLD_DAYS       = 45       # trading days
SURPRISE_WINDOW = 1        # days each side of filing date (D-1, D0, D+1)
ENTRY_OFFSET    = 2        # trading days after filing date
MIN_LEG         = 20       # min names per leg before quintile→tercile guard
QUINTILE_WINDOW_DAYS = 60  # calendar days for rolling quintile assignment
BETA_WINDOW_DAYS     = 60  # calendar days for OLS beta lookback
BETA_MIN_OBS         = 20  # min daily observations for beta
BETA_LOOKBACK_CAL    = 95  # calendar day span for beta window

# ── DSR ───────────────────────────────────────────────────────────────────────
N_TRIALS = 11

# ── Backtest period ───────────────────────────────────────────────────────────
IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")

# ── Cost model (same as all prior experiments) ────────────────────────────────
SPREAD_MIN_BPS    = 20
SPREAD_MAX_BPS    = 100
BORROW_MIN_ANNUAL = 0.005
BORROW_MAX_ANNUAL = 0.020


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


# =============================================================================
# SECTION 3: Tiingo metadata
# =============================================================================

def load_tiingo_tickers() -> pd.DataFrame:
    path = CACHE / "tickers.csv"
    if path.exists():
        df = pd.read_csv(path, parse_dates=["startDate", "endDate"])
        print(f"[Tiingo] Loaded {len(df):,} tickers from cache.")
        return df
    raise FileNotFoundError(f"Missing {path}. Run research/mgmt_pit/run.py first.")


# =============================================================================
# SECTION 4: SEC prefilter
# =============================================================================

def load_prefilter() -> pd.DataFrame:
    path = CACHE / "sec_prefilter.csv"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}.")
    df = pd.read_csv(path)
    print(f"[PreFilter] {len(df):,} screened, {df['passed'].sum():,} survivors.")
    return df


# =============================================================================
# SECTION 5: Raw close cache (hold-period P&L, delisting-safe)
# =============================================================================

_PRICES_PKL = CACHE / "prices_all.pkl"
_PRICE_CACHE: dict = {}
_PRICE_EMPTY = pd.DataFrame(columns=["date", "close", "volume"])


def preload_all_prices(tickers: list) -> None:
    t0 = time.time()
    if not _PRICES_PKL.exists():
        raise FileNotFoundError(f"Missing {_PRICES_PKL}.")
    with open(_PRICES_PKL, "rb") as f:
        data = pickle.load(f)
    _PRICE_CACHE.update(data.get("prices", {}))
    print(f"[Preload] Raw prices: {len(_PRICE_CACHE):,} tickers in {time.time()-t0:.1f}s")


# =============================================================================
# SECTION 6: AdjClose cache (surprise window, CAAR, beta)
# =============================================================================

_ADJ_PKL      = CACHE / "adjclose_all.pkl"
_ADJ_CACHE:   dict = {}
_ADJ_EMPTY    = pd.DataFrame(columns=["date", "adjClose"])


def preload_all_adj_prices(tickers: list) -> None:
    t0 = time.time()
    if not _ADJ_PKL.exists():
        print("[AdjPreload] No adjClose pickle — surprise will be degraded.")
        return
    with open(_ADJ_PKL, "rb") as f:
        data = pickle.load(f)
    _ADJ_CACHE.update(data.get("adj", {}))
    adj_ok = sum(1 for v in _ADJ_CACHE.values() if not v.empty)
    print(f"[AdjPreload] {adj_ok:,} tickers in {time.time()-t0:.1f}s")


# =============================================================================
# SECTION 7: Earnings calendar (8-K Item 2.02 primary, 10-Q fallback)
# =============================================================================

_SEC_SUBS_DIR  = CACHE / "sec_subs"
_SEC_FACTS_DIR = CACHE / "sec_facts"

# Trimmed GP facts also has filing info; use full sec_facts for raw dates
_SEC_FACTS_TRIMMED = CACHE / "sec_facts_trimmed.json"
_SHARES_CONCEPTS   = ["CommonStockSharesOutstanding", "EntityCommonStockSharesOutstanding"]

# All facts already loaded? If GP run was done before PEAD, gp_facts_trimmed.json
# exists.  We only need shares for universe building here.
_SHARES_CACHE: dict = {}


def _load_shares_from_facts_trimmed() -> None:
    """Load shares-outstanding from the existing sec_facts_trimmed.json."""
    p = _SEC_FACTS_TRIMMED
    if not p.exists():
        return
    with open(p) as f:
        raw = json.load(f)
    for k, v in raw.items():
        if v:
            _SHARES_CACHE[int(k)] = v
    print(f"[Shares] Loaded {len(_SHARES_CACHE):,} entries from sec_facts_trimmed.")


def _get_shares_pit(cik: int, as_of_str: str) -> Optional[float]:
    facts = _SHARES_CACHE.get(cik)
    if facts is None:
        return None
    for concept in _SHARES_CONCEPTS:
        try:
            units_dict = facts["facts"]["us-gaap"][concept]["units"]
        except (KeyError, TypeError):
            continue
        best = None
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


def _extract_8k_dates(cik: int) -> list:
    """Return list of (filing_date_str, 'primary-8K') for 8-K Item 2.02 events."""
    p = _SEC_SUBS_DIR / f"{cik}.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return []
    recent = d.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    items  = recent.get("items", [])
    dates  = recent.get("filingDate", [])
    results = []
    for fm, it, dt in zip(forms, items, dates):
        if IS_START.strftime("%Y-%m-%d") <= dt < IS_END.strftime("%Y-%m-%d"):
            # Filter amended/restated forms
            if fm == "8-K" and "2.02" in str(it):
                results.append((dt, "8-K"))
    return results


def _extract_10q_dates_from_facts(cik: int) -> list:
    """
    Return list of (filing_date_str, '10-Q') from sec_facts for 10-Q/10-K forms.
    Used as fallback for CIKs where sec_subs recent block misses early IS years.
    """
    p = _SEC_FACTS_DIR / f"{cik}.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return []
    seen_dates: set = set()
    ug = d.get("facts", {}).get("us-gaap", {})
    for concept_data in ug.values():
        for unit_data in concept_data.get("units", {}).values():
            for e in unit_data:
                fm   = e.get("form", "")
                filed = e.get("filed", "")
                if fm in ("10-Q", "10-K") and (
                        IS_START.strftime("%Y-%m-%d") <= filed < IS_END.strftime("%Y-%m-%d")):
                    seen_dates.add((filed, "10-Q-fb"))
    return sorted(seen_dates)


def _subs_oldest_date(cik: int) -> Optional[str]:
    """Oldest filingDate in the recent block of sec_subs for this CIK."""
    p = _SEC_SUBS_DIR / f"{cik}.json"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            d = json.load(f)
    except Exception:
        return None
    dates = d.get("filings", {}).get("recent", {}).get("filingDate", [])
    return min(dates) if dates else None


def build_earnings_calendar(survivor_ciks: list, ticker_cik_map: dict) -> pd.DataFrame:
    """
    Build PIT earnings event table.

    For each CIK:
      1. Extract 8-K Item 2.02 dates from sec_subs (primary).
      2. If sec_subs recent block starts after IS_START, also extract 10-Q/10-K dates
         from sec_facts for the gap window; deduplicate by suppressing a 10-Q date
         if an 8-K exists within ±60 calendar days.
    """
    t0 = time.time()
    cik_to_tickers: dict = defaultdict(list)
    for ticker, cik in ticker_cik_map.items():
        cik_to_tickers[cik].append(ticker)

    is_start_str = IS_START.strftime("%Y-%m-%d")
    rows = []
    n_8k = n_10q = n_dedup = 0

    for cik in survivor_ciks:
        tickers = cik_to_tickers.get(cik, [])
        if not tickers:
            continue
        ticker = sorted(tickers)[0]  # pick first alphabetically if multiple

        # Primary: 8-K Item 2.02
        events_8k  = _extract_8k_dates(cik)
        dates_8k   = set(dt for dt, _ in events_8k)
        n_8k      += len(events_8k)

        # Fallback: 10-Q dates from sec_facts if subs gap
        events_10q: list = []
        oldest_subs = _subs_oldest_date(cik)
        if oldest_subs is None or oldest_subs > is_start_str:
            for dt, src in _extract_10q_dates_from_facts(cik):
                # Suppress if within 60 days of any 8-K event (same quarter)
                dt_d = date.fromisoformat(dt)
                near_8k = any(
                    abs((dt_d - date.fromisoformat(bk)).days) <= 60
                    for bk in dates_8k
                )
                if near_8k:
                    n_dedup += 1
                else:
                    events_10q.append((dt, "10-Q-fb"))
        n_10q += len(events_10q)

        for dt, src in events_8k + events_10q:
            rows.append({
                "cik":         cik,
                "ticker":      ticker,
                "filing_date": pd.Timestamp(dt),
                "source":      src,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["cik", "filing_date"])
    df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    elapsed = time.time() - t0
    print(f"[EarningsCalendar] {len(df):,} events | "
          f"8-K: {n_8k:,} | 10-Q-fb: {n_10q:,} | deduped: {n_dedup:,} "
          f"({len(df['cik'].unique()):,} CIKs) in {elapsed:.1f}s")
    return df


# =============================================================================
# SECTION 8: Daily return matrices (for IS window)
# =============================================================================

def build_daily_return_matrices(
    tickers: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple:
    """
    Build:
      raw_ret  : DataFrame(date, ticker) of daily raw-close % changes
      adj_ret  : DataFrame(date, ticker) of daily adjClose % changes
      ew_ret   : Series(date) of EW-universe daily adjClose return
      trading_days : sorted list of pd.Timestamp (trading days in [start, end])
    """
    t0 = time.time()
    # Raw close
    raw_series: dict = {}
    adj_series: dict = {}
    extended_end = end + pd.Timedelta(days=HOLD_DAYS * 2)  # extra for hold returns

    for ticker in tickers:
        # Raw close
        df_r = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
        if not df_r.empty:
            mask = (df_r["date"] >= start - pd.Timedelta(days=5)) & (df_r["date"] <= extended_end)
            s    = df_r.loc[mask].set_index("date")["close"]
            s    = s[~s.index.duplicated(keep="last")].sort_index()
            raw_series[ticker] = s.pct_change().dropna()

        # AdjClose
        df_a = _ADJ_CACHE.get(ticker, _ADJ_EMPTY)
        if not df_a.empty:
            mask_a = (df_a["date"] >= start - pd.Timedelta(days=120)) & (df_a["date"] <= extended_end)
            sa     = df_a.loc[mask_a].set_index("date")["adjClose"]
            sa     = sa[~sa.index.duplicated(keep="last")].sort_index()
            adj_series[ticker] = sa

    raw_ret = pd.DataFrame(raw_series).sort_index()
    adj_lvl = pd.DataFrame(adj_series).sort_index()
    adj_ret = adj_lvl.pct_change().sort_index()

    # EW market return from adjClose (>= IS_START)
    adj_is = adj_ret[adj_ret.index >= start]
    ew_ret = adj_is.clip(-0.5, 0.5).mean(axis=1)

    # Trading day list from raw close dates in IS window + hold buffer
    tdays_full = sorted(raw_ret[(raw_ret.index >= start) & (raw_ret.index < extended_end)].index.tolist())

    print(f"[DailyMatrix] raw_ret: {raw_ret.shape}, adj_ret: {adj_ret.shape}, "
          f"ew_ret: {len(ew_ret):,} days in {time.time()-t0:.1f}s")
    return raw_ret, adj_lvl, adj_ret, ew_ret, tdays_full


# =============================================================================
# SECTION 9: Surprise computation  (3-day adj abnormal return)
# =============================================================================

def compute_surprises(
    events_df: pd.DataFrame,
    adj_lvl: pd.DataFrame,
    ew_ret: pd.Series,
    tdays_full: list,
) -> pd.DataFrame:
    """
    For each event:
      surprise = (adjClose[D+1] / adjClose[D-2] − 1)
                 − (EW_market return over same 3-day window)

    D = filing_date.
    adjClose used (not raw close) to avoid split artifacts within the window.
    All prices are strictly from before the entry date (D+2); no look-ahead.

    Returns events_df with 'surprise' column (NaN if insufficient price data).
    """
    tdays_set = sorted(set(tdays_full))
    tdays_arr = np.array(tdays_set)

    def _nth_tday_after(d: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
        """nth trading day on or after d."""
        idx = np.searchsorted(tdays_arr, d)
        target = idx + n
        if target < len(tdays_arr):
            return pd.Timestamp(tdays_arr[target])
        return None

    def _nth_tday_before(d: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
        """nth trading day before d (n=1 means previous trading day)."""
        idx = np.searchsorted(tdays_arr, d, side="left") - n
        if idx >= 0:
            return pd.Timestamp(tdays_arr[idx])
        return None

    surprises    = []
    entry_dates  = []
    exit_dates   = []
    valid        = []

    for _, row in events_df.iterrows():
        fd = row["filing_date"]
        ticker = row["ticker"]

        # Key dates for 3-day window: [D-1, D0, D+1]
        d_minus2 = _nth_tday_before(fd, 2)   # close BEFORE the window
        d_plus1  = _nth_tday_after(fd,  1)   # close at END of window
        d_plus2  = _nth_tday_after(fd,  2)   # entry date

        if d_minus2 is None or d_plus1 is None or d_plus2 is None:
            surprises.append(np.nan)
            entry_dates.append(pd.NaT)
            exit_dates.append(pd.NaT)
            valid.append(False)
            continue

        # Stock adjClose levels
        if ticker not in adj_lvl.columns:
            surprises.append(np.nan)
            entry_dates.append(pd.NaT)
            exit_dates.append(pd.NaT)
            valid.append(False)
            continue

        p_before = adj_lvl[ticker].get(d_minus2, np.nan)
        p_after  = adj_lvl[ticker].get(d_plus1,  np.nan)
        if np.isnan(p_before) or np.isnan(p_after) or p_before <= 0:
            surprises.append(np.nan)
            entry_dates.append(pd.NaT)
            exit_dates.append(pd.NaT)
            valid.append(False)
            continue

        stock_3d = p_after / p_before - 1.0

        # EW market return over same 3-day window
        # Sum of EW daily returns on D-1, D0, D+1
        window_days = tdays_arr[
            np.searchsorted(tdays_arr, d_minus2, side="right") :
            np.searchsorted(tdays_arr, d_plus1,  side="right")
        ]
        mkt_rets = ew_ret.reindex(pd.DatetimeIndex(window_days)).dropna()
        mkt_3d   = float((1 + mkt_rets).prod() - 1.0) if len(mkt_rets) > 0 else np.nan

        if np.isnan(mkt_3d):
            surprises.append(np.nan)
            entry_dates.append(pd.NaT)
            exit_dates.append(pd.NaT)
            valid.append(False)
            continue

        abret = stock_3d - mkt_3d

        # Exit date: entry_date + HOLD_DAYS trading days
        entry_idx = np.searchsorted(tdays_arr, d_plus2)
        exit_idx  = entry_idx + HOLD_DAYS
        if exit_idx < len(tdays_arr):
            exit_d = pd.Timestamp(tdays_arr[exit_idx])
        else:
            exit_d = pd.NaT

        surprises.append(abret)
        entry_dates.append(d_plus2)
        exit_dates.append(exit_d)
        valid.append(True)

    events_df = events_df.copy()
    events_df["surprise"]   = surprises
    events_df["entry_date"] = entry_dates
    events_df["exit_date"]  = exit_dates
    events_df["valid"]      = valid
    return events_df


# =============================================================================
# SECTION 10: Universe membership (PIT monthly check)
# =============================================================================

def build_universe_membership(
    tiingo_tickers: pd.DataFrame,
    ticker_cik_map: dict,
    trading_days: list,
) -> dict:
    """
    Build a PIT universe membership index.
    Returns dict: {date → set of tickers in universe on that date}.
    Recomputed monthly (first trading day of each month) to control runtime.
    """
    t0   = time.time()
    months_done: dict = {}
    membership: dict  = {}
    is_start_str = IS_START.strftime("%Y-%m-%d")
    is_end_str   = IS_END.strftime("%Y-%m-%d")

    for d in trading_days:
        if not (IS_START <= d < IS_END):
            continue
        month_key = d.strftime("%Y-%m")
        if month_key in months_done:
            membership[d] = months_done[month_key]
            continue

        as_of_str = d.strftime("%Y-%m-%d")
        active = tiingo_tickers[
            (tiingo_tickers["startDate"] <= d)
            & (tiingo_tickers["endDate"].isna() | (tiingo_tickers["endDate"] >= d))
        ]
        univ_set = set()
        for _, row in active.iterrows():
            t = row["ticker"]
            if t not in ticker_cik_map:
                continue
            cik = ticker_cik_map[t]
            df_p = _PRICE_CACHE.get(t, _PRICE_EMPTY)
            if df_p.empty:
                continue
            sub = df_p[df_p["date"] <= d]
            if sub.empty:
                continue
            price = float(sub.iloc[-1]["close"])
            if price < PRICE_MIN:
                continue
            shares = _get_shares_pit(cik, as_of_str)
            if shares is None or shares <= 0:
                continue
            mcap = shares * price
            if not (MCAP_MIN <= mcap <= MCAP_MAX):
                continue
            dvol_sub = sub.tail(60)
            dvol = float((dvol_sub["close"] * dvol_sub["volume"]).mean()) if not dvol_sub.empty else 0.0
            if dvol < DOLLAR_VOL_MIN:
                continue
            univ_set.add(t)
        months_done[month_key] = univ_set
        membership[d] = univ_set

    n_months = len(months_done)
    print(f"[Universe] Membership built for {n_months} months in {time.time()-t0:.1f}s")
    return membership


# =============================================================================
# SECTION 11: Quintile assignment (rolling 60-cal-day PIT window)
# =============================================================================

def assign_quintiles(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each event E filed on date F:
      - Look at all events filed in [F - 60 cal days, F] as the comparison group.
      - Assign quintile 1 (top 20%) through 5 (bottom 20%) based on surprise rank.
      - Fully PIT: comparison group only includes events that already have their
        3-day window closed (filed ≤ F means their surprise is known by F).

    quintile 1 = high surprise = long
    quintile 5 = low surprise  = short
    """
    df = events_df[events_df["valid"] & events_df["surprise"].notna()].copy()
    df = df.sort_values("filing_date").reset_index(drop=True)

    quintiles = []
    for i, row in df.iterrows():
        fd    = row["filing_date"]
        surp  = row["surprise"]
        cutoff = fd - pd.Timedelta(days=QUINTILE_WINDOW_DAYS)
        # Comparison: all valid events in the 60-day window up to and including F
        mask = (df["filing_date"] >= cutoff) & (df["filing_date"] <= fd)
        comp = df.loc[mask, "surprise"].dropna().values
        if len(comp) == 0:
            quintiles.append(3)
            continue
        pct_rank = np.mean(comp <= surp)  # what fraction is ≤ this surprise
        if pct_rank >= 0.80:
            quintiles.append(1)
        elif pct_rank >= 0.60:
            quintiles.append(2)
        elif pct_rank >= 0.40:
            quintiles.append(3)
        elif pct_rank >= 0.20:
            quintiles.append(4)
        else:
            quintiles.append(5)

    df["quintile"] = quintiles
    return df


# =============================================================================
# SECTION 12: Beta computation (60-day OLS vs EW universe at filing date)
# =============================================================================

def compute_event_betas(events_df: pd.DataFrame, adj_ret: pd.DataFrame,
                        ew_ret: pd.Series) -> pd.DataFrame:
    """
    Per-event beta using 60-day adjClose OLS vs EW universe.
    Computed at filing_date (before entry) — fully PIT.
    """
    df     = events_df.copy()
    betas  = []
    mcaps  = []

    for _, row in df.iterrows():
        ticker = row["ticker"]
        fd     = row["filing_date"]
        cik    = row["cik"]

        # Beta window: 60 trading days ending AT filing_date
        beta_start = fd - pd.Timedelta(days=BETA_LOOKBACK_CAL)
        if ticker in adj_ret.columns:
            y_all = adj_ret[ticker]
        else:
            betas.append(1.0)
            mcaps.append(0.0)
            continue

        mask = (y_all.index >= beta_start) & (y_all.index <= fd)
        y_s  = y_all[mask].dropna()
        x_s  = ew_ret.reindex(y_s.index).dropna()
        y_s  = y_s.reindex(x_s.index).dropna()
        x_s  = x_s.reindex(y_s.index)
        n    = len(y_s)
        if n < BETA_MIN_OBS:
            betas.append(1.0)
        else:
            X = np.column_stack([np.ones(n), x_s.values])
            try:
                coeffs, _, rank, _ = np.linalg.lstsq(X, y_s.values, rcond=None)
                b = float(coeffs[1]) if rank >= 2 else 1.0
                b = np.clip(b, 0.2, 3.0)
                betas.append(float(b))
            except Exception:
                betas.append(1.0)

        # PIT mcap at filing_date
        df_p = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
        price = np.nan
        if not df_p.empty:
            sub = df_p[df_p["date"] <= fd]
            if not sub.empty:
                price = float(sub.iloc[-1]["close"])
        shares = _get_shares_pit(int(cik), fd.strftime("%Y-%m-%d"))
        mcap = shares * price if (shares and not np.isnan(price)) else 0.0
        mcaps.append(mcap)

    df["beta"] = betas
    df["mcap"] = mcaps
    return df


# =============================================================================
# SECTION 13: Cost model helpers
# =============================================================================

def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / max(MCAP_MAX - MCAP_MIN, 1)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow_daily(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / max(MCAP_MAX - MCAP_MIN, 1)))
    annual = BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac
    return annual / 252


# =============================================================================
# SECTION 14: Calendar-time portfolio engine
# =============================================================================

def run_portfolio(
    events_df: pd.DataFrame,
    raw_ret: pd.DataFrame,
    adj_ret: pd.DataFrame,
    ew_ret: pd.Series,
    tdays_full: list,
    membership: dict,
) -> dict:
    """
    Daily calendar-time L/S PEAD portfolio.

    For each IS trading day d:
      1. Find active events (entry_date ≤ d ≤ exit_date) in-universe at d.
      2. Sort by surprise quintile → long Q1, short Q5.
         Name-count guard: if |Q1| or |Q5| < MIN_LEG, widen to tercile.
      3. EW within each leg; beta-neutral leg-level scaling.
      4. Gross daily P&L from raw close returns.
      5. Costs: bid-ask at entry and exit (per event), daily borrow on shorts.

    Returns dict with:
      daily_records    : per-day portfolio results
      event_records    : per-event drift tracking (for CAAR)
      guard_fire_count : how many days the min-leg guard fired
    """
    # Index events by entry/exit for fast lookup
    ev = events_df[
        events_df["valid"]
        & events_df["surprise"].notna()
        & events_df["entry_date"].notna()
        & events_df["exit_date"].notna()
    ].copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    ev["exit_date"]  = pd.to_datetime(ev["exit_date"])

    # Ensure trading day array is sorted Timestamps
    tdays_arr = sorted(set(tdays_full))

    # Filter to IS window trading days only
    is_tdays = [d for d in tdays_arr if IS_START <= d < IS_END]

    daily_records: list = []
    event_daily:   dict = defaultdict(dict)   # {event_idx: {day: adj_abret}}
    guard_days: int = 0

    # Build raw_ret and adj_ret indexed by date for fast lookup
    raw_idx = raw_ret.index.tolist()
    raw_idx_set = set(raw_idx)

    mid_mcap = (MCAP_MIN + MCAP_MAX) / 2

    for d in is_tdays:
        # Active events on this day
        active = ev[(ev["entry_date"] <= d) & (ev["exit_date"] >= d)].copy()

        # Filter to in-universe at d (use monthly membership)
        in_univ = membership.get(d, set())
        active  = active[active["ticker"].isin(in_univ)] if in_univ else active

        # Deduplicate: if ticker appears multiple times, keep most-recent entry
        if active.duplicated(subset=["ticker"], keep=False).any():
            active = (active.sort_values("entry_date", ascending=False)
                            .drop_duplicates(subset=["ticker"], keep="first"))

        if len(active) < 4:
            daily_records.append({"date": d, "ls_gross": 0.0, "ls_net": 0.0,
                                   "ls_ba": 0.0, "ls_bw": 0.0, "n_long": 0,
                                   "n_short": 0, "skipped": True,
                                   "guard_fired": False, "mkt_ret": ew_ret.get(d, np.nan)})
            continue

        # Sort by surprise descending; assign legs
        active_sorted = active.sort_values("surprise", ascending=False)
        n = len(active_sorted)
        q_cut = max(1, int(n * 0.20))
        guard_fired = False
        sort_label  = "quintile"

        if q_cut < MIN_LEG:
            t_cut = max(1, int(n * 0.33))
            if t_cut >= MIN_LEG:
                q_cut = t_cut
                sort_label = "tercile"
            else:
                q_cut = max(1, n // 2)  # use best available
                sort_label = "all"
            guard_fired = True
            guard_days += 1

        long_ev  = active_sorted.iloc[:q_cut]
        short_ev = active_sorted.iloc[-q_cut:]

        # Beta-neutral leg scaling
        bl = float(np.clip(long_ev["beta"].mean(),  0.2, 3.0))
        bs = float(np.clip(short_ev["beta"].mean(), 0.2, 3.0))
        denom = bl + bs
        if denom > 0.05:
            long_scale  = float(np.clip(2 * bs / denom, 0.2, 3.0))
            short_scale = float(np.clip(2 * bl / denom, 0.2, 3.0))
        else:
            long_scale = short_scale = 1.0

        n_long  = len(long_ev)
        n_short = len(short_ev)

        # Gross daily returns (raw close)
        long_tickers  = long_ev["ticker"].tolist()
        short_tickers = short_ev["ticker"].tolist()

        def _leg_raw_ret(tickers_list: list) -> float:
            if d not in raw_idx_set:
                return np.nan
            vals = raw_ret.loc[d, [t for t in tickers_list if t in raw_ret.columns]]
            v    = vals.dropna()
            return float(v.mean()) if len(v) > 0 else np.nan

        long_gross_d  = _leg_raw_ret(long_tickers)
        short_gross_d = _leg_raw_ret(short_tickers)

        if np.isnan(long_gross_d) or np.isnan(short_gross_d):
            daily_records.append({"date": d, "ls_gross": np.nan, "ls_net": np.nan,
                                   "ls_ba": 0.0, "ls_bw": 0.0,
                                   "n_long": n_long, "n_short": n_short,
                                   "skipped": True, "guard_fired": guard_fired,
                                   "mkt_ret": ew_ret.get(d, np.nan)})
            continue

        ls_gross_d = long_scale * long_gross_d - short_scale * short_gross_d

        # ── Costs ──────────────────────────────────────────────────────────
        ba_cost_d  = 0.0
        bw_cost_d  = 0.0

        # Bid-ask at ENTRY (position opened today)
        entering_long  = long_ev[long_ev["entry_date"] == d]
        entering_short = short_ev[short_ev["entry_date"] == d]

        for _, er in entering_long.iterrows():
            mc   = er["mcap"] if er["mcap"] > 0 else mid_mcap
            wt   = 1.0 / n_long
            ba_cost_d += 2.0 * _spread(mc) * wt * long_scale

        for _, er in entering_short.iterrows():
            mc   = er["mcap"] if er["mcap"] > 0 else mid_mcap
            wt   = 1.0 / n_short
            ba_cost_d += 2.0 * _spread(mc) * wt * short_scale

        # Bid-ask at EXIT (position closed today)
        exiting_long  = long_ev[long_ev["exit_date"] == d]
        exiting_short = short_ev[short_ev["exit_date"] == d]

        for _, er in exiting_long.iterrows():
            mc   = er["mcap"] if er["mcap"] > 0 else mid_mcap
            wt   = 1.0 / n_long
            ba_cost_d += 2.0 * _spread(mc) * wt * long_scale

        for _, er in exiting_short.iterrows():
            mc   = er["mcap"] if er["mcap"] > 0 else mid_mcap
            wt   = 1.0 / n_short
            ba_cost_d += 2.0 * _spread(mc) * wt * short_scale

        # Daily borrow on short positions
        for _, er in short_ev.iterrows():
            mc   = er["mcap"] if er["mcap"] > 0 else mid_mcap
            wt   = 1.0 / n_short
            bw_cost_d += _borrow_daily(mc) * wt * short_scale

        ls_net_d = ls_gross_d - ba_cost_d - bw_cost_d

        # ── CAAR tracking (adj abnormal return) ────────────────────────────
        if d in adj_ret.index:
            mkt_d = ew_ret.get(d, np.nan)
            for _, er in long_ev.iterrows():
                idx_key = int(er.name)
                t = er["ticker"]
                if t in adj_ret.columns:
                    ar = adj_ret.loc[d, t]
                    if not np.isnan(ar) and not np.isnan(mkt_d):
                        event_daily[idx_key][d] = ar - mkt_d
            for _, er in short_ev.iterrows():
                idx_key = int(er.name)
                t = er["ticker"]
                if t in adj_ret.columns:
                    ar = adj_ret.loc[d, t]
                    if not np.isnan(ar) and not np.isnan(mkt_d):
                        event_daily[idx_key][d] = -(ar - mkt_d)  # inverted for short

        daily_records.append({
            "date":         d,
            "ls_gross":     ls_gross_d,
            "ls_net":       ls_net_d,
            "ls_ba":        ba_cost_d,
            "ls_bw":        bw_cost_d,
            "n_long":       n_long,
            "n_short":      n_short,
            "long_scale":   long_scale,
            "short_scale":  short_scale,
            "beta_long":    bl,
            "beta_short":   bs,
            "skipped":      False,
            "guard_fired":  guard_fired,
            "sort_label":   sort_label,
            "mkt_ret":      ew_ret.get(d, np.nan),
        })

    print(f"[Portfolio] {len(is_tdays):,} trading days processed. "
          f"Guard fired: {guard_days:,} days ({100*guard_days/max(len(is_tdays),1):.1f}%).")
    return {
        "daily_records":    daily_records,
        "event_daily":      dict(event_daily),
        "guard_fire_count": guard_days,
        "n_tdays":          len(is_tdays),
    }


# =============================================================================
# SECTION 15: IC computation (Spearman — cost-independent signal check)
# =============================================================================

def compute_ic(events_df: pd.DataFrame, raw_ret: pd.DataFrame) -> dict:
    """
    IC = Spearman(surprise at entry, realized 45-day raw-close return).
    Computed across all valid events in IS window.
    """
    rows = []
    for _, row in events_df.iterrows():
        if not row["valid"] or np.isnan(row["surprise"]):
            continue
        if pd.isna(row["entry_date"]) or pd.isna(row["exit_date"]):
            continue
        ticker    = row["ticker"]
        entry_d   = row["entry_date"]
        exit_d    = row["exit_date"]
        if ticker not in raw_ret.columns:
            continue
        # Raw close at entry and exit
        df_r  = _PRICE_CACHE.get(ticker, _PRICE_EMPTY)
        sub_e = df_r[df_r["date"] <= entry_d]
        sub_x = df_r[(df_r["date"] >= entry_d) & (df_r["date"] <= exit_d)]
        if sub_e.empty or sub_x.empty:
            continue
        p_entry = float(sub_e.iloc[-1]["close"])
        p_exit  = float(sub_x.iloc[-1]["close"])
        if p_entry <= 0:
            continue
        hold_ret = p_exit / p_entry - 1.0
        rows.append({"surprise": row["surprise"], "hold_ret": hold_ret,
                     "quintile": row.get("quintile", np.nan)})

    if len(rows) < 20:
        return {"ic": np.nan, "ic_tstat": np.nan, "n_events": len(rows)}

    df = pd.DataFrame(rows)
    ic, _ = stats.spearmanr(df["surprise"], df["hold_ret"])
    n = len(df)
    ic_se = np.sqrt((1 - ic**2)**2 / (n - 2)) if n > 2 else np.nan
    ic_t  = ic / ic_se if ic_se and ic_se > 0 else np.nan
    return {"ic": float(ic), "ic_tstat": float(ic_t), "n_events": n,
            "surprise_std": float(df["surprise"].std()),
            "mean_hold_ret_by_q": df.groupby("quintile")["hold_ret"].mean().to_dict()}


# =============================================================================
# SECTION 16: CAAR by quintile (cumulative avg abnormal return curve)
# =============================================================================

def compute_caar(
    events_df: pd.DataFrame,
    adj_lvl: pd.DataFrame,
    adj_ret: pd.DataFrame,
    ew_ret: pd.Series,
    tdays_full: list,
) -> dict:
    """
    For each event in the top and bottom quintile, track the cumulative
    abnormal return (adjClose) for each of the 45 trading days post-entry.

    Returns dict: {quintile → list of CAAR values for days 1..45}
    """
    tdays_arr = sorted(set(tdays_full))

    ev_q = events_df[
        events_df["valid"]
        & events_df["quintile"].notna()
        & events_df["entry_date"].notna()
    ].copy()

    # Separate by quintile
    by_q: dict = {1: [], 5: []}  # will store per-event daily AR lists

    for _, row in ev_q.iterrows():
        q = int(row["quintile"])
        if q not in (1, 5):
            continue
        ticker   = row["ticker"]
        entry_d  = row["entry_date"]
        if ticker not in adj_lvl.columns:
            continue

        # Entry index in trading days
        entry_idx = np.searchsorted(tdays_arr, entry_d, side="left")
        if entry_idx >= len(tdays_arr):
            continue

        # Collect 45 daily abnormal returns post-entry
        car = 0.0
        car_series = []
        for k in range(1, HOLD_DAYS + 1):
            d_idx = entry_idx + k
            if d_idx >= len(tdays_arr):
                break
            d = tdays_arr[d_idx]
            if ticker in adj_ret.columns:
                ar_stock = adj_ret.loc[d, ticker] if d in adj_ret.index else np.nan
            else:
                ar_stock = np.nan
            mkt_d = ew_ret.get(d, np.nan)
            if not np.isnan(ar_stock) and not np.isnan(mkt_d):
                car += (ar_stock - mkt_d)
            else:
                car += 0.0  # fill gaps with zero
            car_series.append(car)

        if len(car_series) > 0:
            by_q[q].append(car_series)

    # Average across events per day
    caar: dict = {}
    for q, series_list in by_q.items():
        if not series_list:
            caar[q] = [np.nan] * HOLD_DAYS
            continue
        max_len = max(len(s) for s in series_list)
        mat = np.full((len(series_list), max_len), np.nan)
        for i, s in enumerate(series_list):
            mat[i, :len(s)] = s
        caar[q] = np.nanmean(mat, axis=0).tolist()

    return caar


# =============================================================================
# SECTION 17: Metrics
# =============================================================================

def _deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


def compute_metrics_monthly(daily_records: list) -> dict:
    """
    Aggregate daily returns to monthly, then compute all standard metrics.
    Monthly aggregation chosen for DSR (n_obs ≈ 72, more stable than daily).
    """
    df = pd.DataFrame(daily_records)
    df = df[~df["skipped"] & df["ls_net"].notna()].copy()
    if df.empty:
        return {}

    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    # Monthly L/S returns (compound daily)
    monthly = (
        df.groupby("month")["ls_net"].apply(lambda x: (1 + x).prod() - 1)
    )
    monthly_gross = (
        df.groupby("month")["ls_gross"].apply(lambda x: (1 + x).prod() - 1)
    )
    monthly_mkt = (
        df.groupby("month")["mkt_ret"].apply(lambda x: (1 + x.dropna()).prod() - 1)
    )

    ls      = monthly.values
    ls_gr   = monthly_gross.values
    n_obs   = len(ls)
    if n_obs < 2:
        return {}

    cum     = (1 + pd.Series(ls)).cumprod()
    total   = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (12.0 / n_obs) - 1)
    ann_gr  = float(((1 + pd.Series(ls_gr)).cumprod().iloc[-1]) ** (12.0 / n_obs) - 1)

    mu    = float(np.mean(ls))
    sigma = float(np.std(ls, ddof=1))
    sharpe = float((mu / sigma) * np.sqrt(12)) if sigma > 0 else np.nan

    roll_mx = cum.cummax()
    dd_s    = (cum - roll_mx) / roll_mx
    max_dd  = float(dd_s.min())
    calmar  = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan

    t_stat  = float(mu / (sigma / np.sqrt(n_obs))) if sigma > 0 else np.nan
    win_rt  = float((pd.Series(ls) > 0).mean())
    per_sr  = float(mu / sigma) if sigma > 0 else np.nan

    # DSR
    dsr_thr = _deflated_sharpe_threshold(N_TRIALS, n_obs)
    clears  = bool(per_sr > dsr_thr) if not np.isnan(per_sr) else False

    # Market beta (monthly L/S vs monthly EW market)
    mkt_arr = monthly_mkt.values
    valid   = ~(np.isnan(ls) | np.isnan(mkt_arr))
    if valid.sum() >= 5:
        X  = np.column_stack([np.ones(valid.sum()), mkt_arr[valid]])
        y  = ls[valid]
        try:
            c, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
            beta_mkt = float(c[1])
            y_hat    = X @ c
            ss_res   = float(np.dot(y - y_hat, y - y_hat))
            se       = (np.sqrt(ss_res / max(valid.sum() - 2, 1))
                        / (np.std(mkt_arr[valid], ddof=1) * np.sqrt(valid.sum() - 1)))
            beta_t   = float(beta_mkt / se) if se > 0 else np.nan
        except Exception:
            beta_mkt = beta_t = np.nan
    else:
        beta_mkt = beta_t = np.nan

    # Annual cost drag
    ba_ann  = float(df["ls_ba"].mean()) * 252 if "ls_ba" in df.columns else np.nan
    bw_ann  = float(df["ls_bw"].mean()) * 252 if "ls_bw" in df.columns else np.nan
    cost_ann_bps = (ba_ann + bw_ann) * 10_000 if not np.isnan(ba_ann) else np.nan

    # Turnover: fraction of long/short book that turns over each day
    n_long_mean  = float(df["n_long"].mean())  if "n_long"  in df.columns else np.nan
    entering_days = (df[df["ls_ba"] > 0]["ls_ba"] > 0).sum() if "ls_ba" in df.columns else 0

    return {
        "n_obs_monthly":    n_obs,
        "total_ret":        total,
        "ann_ret_net":      ann_ret,
        "ann_ret_gross":    ann_gr,
        "cost_drag_bps":    cost_ann_bps,
        "ba_bps":           ba_ann * 10_000 if not np.isnan(ba_ann) else np.nan,
        "bw_bps":           bw_ann * 10_000 if not np.isnan(bw_ann) else np.nan,
        "sharpe_net":       sharpe,
        "calmar":           calmar,
        "max_dd":           max_dd,
        "t_stat_monthly":   t_stat,
        "win_rate":         win_rt,
        "beta_mkt":         beta_mkt,
        "beta_mkt_tstat":   beta_t,
        "per_month_sr":     per_sr,
        "dsr_threshold":    dsr_thr,
        "dsr_thr_ann":      dsr_thr * np.sqrt(12) if not np.isnan(dsr_thr) else np.nan,
        "clears_dsr":       clears,
        "mean_n_long":      n_long_mean,
    }


# =============================================================================
# SECTION 18: Output
# =============================================================================

def print_caar(caar: dict) -> None:
    sep = "─" * 68
    print(f"\n{sep}")
    print("CAAR BY SURPRISE QUINTILE  (cumulative avg abnormal adjClose return)")
    print("Day 0 = entry date (D+2 after filing).  + = long earns abnormal return.")
    print(f"{'Day':>4}  {'Q1 (top)':>10}  {'Q5 (bot)':>10}  {'Q1−Q5':>10}  Spread")
    print(f"{'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*6}")
    q1 = caar.get(1, [])
    q5 = caar.get(5, [])
    for i in range(min(HOLD_DAYS, len(q1), len(q5))):
        q1v  = q1[i] if i < len(q1) else np.nan
        q5v  = q5[i] if i < len(q5) else np.nan
        diff = q1v - q5v if not (np.isnan(q1v) or np.isnan(q5v)) else np.nan
        # ASCII bar (scale = 0.5% per char)
        bar_len = int(diff / 0.005) if not np.isnan(diff) else 0
        bar = ("+" * max(0, bar_len) if bar_len >= 0 else "-" * max(0, -bar_len))[:12]
        print(f"{i+1:>4}  {q1v:>10.4%}  {q5v:>10.4%}  {diff:>10.4%}  {bar}")
    # Verdict
    if q1 and q5:
        final_q1 = q1[min(HOLD_DAYS-1, len(q1)-1)]
        final_q5 = q5[min(HOLD_DAYS-1, len(q5)-1)]
        spread   = final_q1 - final_q5
        print(f"\n  Final CAAR spread (Q1 − Q5) at day {HOLD_DAYS}: {spread:+.4%}")
        if spread > 0.01:
            print("  → Drift mechanism PRESENT: Q1 drifts up, Q5 drifts down post-entry.")
        elif spread > 0:
            print("  → Drift mechanism WEAK: positive spread but < 1% over 45 days.")
        else:
            print("  → Drift mechanism ABSENT: spread ≤ 0. Drift not visible in this data.")


def print_results(
    metrics: dict,
    ic_stats: dict,
    portfolio_results: dict,
    events_df: pd.DataFrame,
) -> None:
    sep = "=" * 78
    daily_df = pd.DataFrame(portfolio_results["daily_records"])
    daily_act = daily_df[~daily_df["skipped"]]

    guard_days   = portfolio_results["guard_fire_count"]
    n_tdays      = portfolio_results["n_tdays"]
    guard_pct    = 100 * guard_days / max(n_tdays, 1)

    n_events     = ic_stats.get("n_events", 0)
    mean_n_long  = metrics.get("mean_n_long", np.nan)
    n_q1 = (events_df["quintile"] == 1).sum()
    n_q5 = (events_df["quintile"] == 5).sum()

    # Turnover: mean daily active count as fraction of avg_portfolio entering each day
    entering_mask = (daily_act["ls_ba"] > 0) if "ls_ba" in daily_act.columns else pd.Series(dtype=bool)

    print(f"\n{sep}")
    print("S2: POST-EARNINGS ANNOUNCEMENT DRIFT — IN-SAMPLE RESULTS  (trial #11)")
    print("Signal: 3-day adj abnormal return (vs EW market) around SEC filing date")
    print("Long top-quintile surprise  |  Short bottom-quintile  |  Beta-neutral")
    print(f"Hold: {HOLD_DAYS} trading days  |  Calendar-time daily portfolio  |  IS: 2015-2021")
    print(sep)

    print(f"\n  *** INFORMATION COEFFICIENT (cost-independent signal quality) ***")
    print(f"  Mean IC (Spearman, surprise vs 45-day hold return) : {ic_stats.get('ic', np.nan):>10.4f}")
    print(f"  IC t-stat                                          : {ic_stats.get('ic_tstat', np.nan):>10.3f}")
    print(f"  Events evaluated for IC                            : {ic_stats.get('n_events', 0):>10,}")
    print(f"  IC > 0: high surprise predicts positive 45-day return.")
    print(f"  IC t > 2: signal statistically significant.")

    print(f"\n  Event calendar:")
    print(f"    Total PEAD events (IS window)                    : {len(events_df):>10,}")
    print(f"    Valid (surprise computed, entered portfolio)      : {int(events_df['valid'].sum()):>10,}")
    print(f"    Long (Q1 surprise)                               : {n_q1:>10,}")
    print(f"    Short (Q5 surprise)                              : {n_q5:>10,}")
    print(f"    Mean surprise (std)                              : {events_df['surprise'].std():.4f}")

    w = 46
    print(f"\n  {'Metric':<{w}} {'Value':>14}")
    print(f"  {'─'*w} {'─'*14}")
    def r(label, val, fmt="{:.3f}"):
        fmtd = fmt.format(val) if isinstance(val, float) and not np.isnan(val) else ("—" if not isinstance(val, bool) else str(val))
        if isinstance(val, bool):
            fmtd = "YES" if val else "NO"
        print(f"  {label:<{w}} {fmtd:>14}")

    r("Total cumulative return (net)", metrics.get("total_ret", np.nan), "{:.2%}")
    r("Annualized return (net)",       metrics.get("ann_ret_net", np.nan), "{:.2%}")
    r("Annualized return (GROSS)",     metrics.get("ann_ret_gross", np.nan), "{:.2%}")
    r("Cost drag (bps/yr)",            metrics.get("cost_drag_bps", np.nan), "{:.0f}")
    r("  bid-ask (bps/yr)",            metrics.get("ba_bps", np.nan), "{:.0f}")
    r("  borrow  (bps/yr)",            metrics.get("bw_bps", np.nan), "{:.0f}")
    r("Sharpe ratio (annualized, net)", metrics.get("sharpe_net", np.nan))
    r("CALMAR ratio",                  metrics.get("calmar", np.nan))
    r("Max drawdown",                  metrics.get("max_dd", np.nan), "{:.2%}")
    r("t-stat (net monthly returns)",  metrics.get("t_stat_monthly", np.nan))
    r("Market beta (vs EW universe)",  metrics.get("beta_mkt", np.nan))
    r("  beta t-stat (should be ≈0)", metrics.get("beta_mkt_tstat", np.nan))
    r("Win rate (monthly)",            metrics.get("win_rate", np.nan), "{:.1%}")
    r("N (monthly periods in IS)",     float(metrics.get("n_obs_monthly", 0)), "{:.0f}")
    r("Mean long-leg name count/day",  metrics.get("mean_n_long", np.nan), "{:.1f}")
    r("Name-count guard fire rate",    guard_pct, "{:.1f}%")
    r("Long-leg gross exposure ($)",   daily_act["long_scale"].mean() if "long_scale" in daily_act.columns else np.nan)
    r("Short-leg gross exposure ($)",  daily_act["short_scale"].mean() if "short_scale" in daily_act.columns else np.nan)

    print(f"\n  --- DSR Check (N_trials={N_TRIALS}, monthly periods) ---")
    print(f"  Per-month Sharpe                          : {metrics.get('per_month_sr', np.nan):>12.4f}")
    print(f"  DSR threshold (per-month SR)              : {metrics.get('dsr_threshold', np.nan):>12.4f}")
    print(f"  DSR threshold (annualized SR)             : {metrics.get('dsr_thr_ann', np.nan):>12.4f}")
    print(f"  Clears DSR                                : {'YES' if metrics.get('clears_dsr') else 'NO':>12}")

    # Monthly return log (abbreviated to years)
    print(f"\n  Monthly net returns (annualized by calendar year):")
    daily_df2 = pd.DataFrame(portfolio_results["daily_records"])
    daily_df2 = daily_df2[~daily_df2["skipped"] & daily_df2["ls_net"].notna()].copy()
    if not daily_df2.empty:
        daily_df2["date"] = pd.to_datetime(daily_df2["date"])
        daily_df2["year"] = daily_df2["date"].dt.year
        for yr, grp in daily_df2.groupby("year"):
            yr_cum = (1 + grp["ls_net"]).prod() - 1
            n_days = len(grp)
            yr_ann = (1 + yr_cum) ** (252 / n_days) - 1
            print(f"    {yr}: {yr_ann:+.2%} (cumulative {yr_cum:+.2%}, {n_days}d)")


def save_results(portfolio_results: dict, events_df: pd.DataFrame,
                 caar: dict, metrics: dict) -> None:
    daily_df = pd.DataFrame(portfolio_results["daily_records"])
    daily_df.to_csv(RESULTS / "pead_daily_returns.csv", index=False)
    events_df.to_csv(RESULTS / "pead_events.csv", index=False)
    # CAAR
    caar_rows = []
    q1 = caar.get(1, [])
    q5 = caar.get(5, [])
    for i in range(min(HOLD_DAYS, len(q1), len(q5))):
        caar_rows.append({"day": i+1, "caar_q1": q1[i] if i < len(q1) else np.nan,
                          "caar_q5": q5[i] if i < len(q5) else np.nan})
    pd.DataFrame(caar_rows).to_csv(RESULTS / "pead_caar.csv", index=False)
    print(f"[Saved] Results → {RESULTS}/")


def log_registry(metrics: dict, ic_stats: dict,
                 events_df: pd.DataFrame, guard_days: int) -> None:
    row = {
        "timestamp":            __import__("datetime").datetime.now().isoformat(),
        "study":                "pead_S2",
        "hypothesis":           (
            "Post-earnings announcement drift in small/mid-cap stocks. "
            "High (low) earnings surprise stocks continue to drift up (down) "
            "for 45 trading days post-announcement. Surprise = 3-day adj abnormal "
            "return around SEC filing date. Beta-neutral. Long Q1, short Q5."
        ),
        "data_source":          "Tiingo (raw close P&L) + adjClose (surprise/CAAR) + SEC 8-K 2.02 / 10-Q",
        "status":               "completed_in_sample",
        "trial_number":         11,
        "trial_note":           "Independent trial #11. Not a family member of any prior trial.",
        "n_trials_dsr":         N_TRIALS,
        "rebalance_freq":       "daily_calendar_time",
        "hold_days":            HOLD_DAYS,
        "portfolio_note":       "Beta-neutral (EW market proxy, 60-day adjClose). Quintile guard → tercile if <20/leg.",
        "signal_note":          "3-day adj abnormal return around 8-K Item 2.02 (10-Q fallback). Entry D+2.",
        "ann_return_net":       metrics.get("ann_ret_net"),
        "ann_return_gross":     metrics.get("ann_ret_gross"),
        "annual_cost_drag_bps": metrics.get("cost_drag_bps"),
        "sharpe":               metrics.get("sharpe_net"),
        "calmar":               metrics.get("calmar"),
        "max_drawdown":         metrics.get("max_dd"),
        "t_stat":               metrics.get("t_stat_monthly"),
        "market_beta":          metrics.get("beta_mkt"),
        "market_beta_tstat":    metrics.get("beta_mkt_tstat"),
        "win_rate":             metrics.get("win_rate"),
        "n_periods":            metrics.get("n_obs_monthly"),
        "per_period_sr":        metrics.get("per_month_sr"),
        "dsr_threshold":        metrics.get("dsr_threshold"),
        "clears_dsr":           metrics.get("clears_dsr"),
        "mean_ic":              ic_stats.get("ic"),
        "ic_t_stat":            ic_stats.get("ic_tstat"),
        "n_events":             ic_stats.get("n_events"),
        "guard_fire_days":      guard_days,
    }
    reg_df = pd.read_csv(REGISTRY) if REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "pead_S2"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(REGISTRY, index=False)
    print(f"[Registry] pead_S2 logged (trial #11) → {REGISTRY}")


# =============================================================================
# SECTION 19: Main
# =============================================================================

def main() -> None:
    t_total = time.time()
    sep = "=" * 78
    print(sep)
    print("S2: POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD)  —  trial #11")
    print("Signal: 3-day adj abnormal return (vs EW universe) around SEC filing date")
    print("Entry D+2 | Hold 45 trading days | Beta-neutral | Calendar-time daily")
    print(f"DSR: N_TRIALS={N_TRIALS} | IS: 2015-01-01 → 2021-01-01")
    print(sep)

    # ── Step 1: Metadata ──────────────────────────────────────────────────────
    print("\n[Step 1] Loading metadata...")
    tiingo_tickers = load_tiingo_tickers()
    prefilter_df   = load_prefilter()

    survivors         = prefilter_df[prefilter_df["passed"]]
    survivor_tickers  = survivors["ticker"].tolist()
    survivor_ciks     = survivors["cik"].dropna().astype(int).tolist()

    # Build ticker → cik map (and cik → ticker)
    ticker_cik_map = {r["ticker"]: int(r["cik"]) for _, r in survivors.iterrows()
                      if pd.notna(r["cik"])}

    # ── Step 2: Price caches ──────────────────────────────────────────────────
    print(f"\n[Step 2a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    preload_all_prices(survivor_tickers)

    print(f"\n[Step 2b] Loading adjClose pickle...")
    preload_all_adj_prices(survivor_tickers)

    # ── Step 3: Shares for universe membership ────────────────────────────────
    print("\n[Step 3] Loading shares-outstanding (for universe check)...")
    _load_shares_from_facts_trimmed()

    # ── Step 4: Earnings calendar ─────────────────────────────────────────────
    print(f"\n[Step 4] Building earnings calendar ({len(survivor_ciks):,} CIKs)...")
    events_raw = build_earnings_calendar(survivor_ciks, ticker_cik_map)

    # ── Step 5: Daily return matrices ─────────────────────────────────────────
    print(f"\n[Step 5] Building daily return matrices (IS + hold buffer)...")
    raw_ret, adj_lvl, adj_ret, ew_ret, tdays_full = build_daily_return_matrices(
        survivor_tickers, IS_START, IS_END)

    # ── Step 6: Universe membership (PIT monthly) ─────────────────────────────
    print("\n[Step 6] Building universe membership (monthly PIT check)...")
    is_tdays = [d for d in tdays_full if IS_START <= d < IS_END]
    membership = build_universe_membership(tiingo_tickers, ticker_cik_map, is_tdays)

    # ── Step 7: Compute surprises ─────────────────────────────────────────────
    print(f"\n[Step 7] Computing 3-day adj abnormal returns ({len(events_raw):,} events)...")
    events_surp = compute_surprises(events_raw, adj_lvl, ew_ret, tdays_full)
    n_valid = int(events_surp["valid"].sum())
    n_na    = events_surp["surprise"].isna().sum()
    print(f"  Valid surprises: {n_valid:,} | NaN (insufficient price data): {n_na:,}")

    # ── Step 8: Quintile assignment ────────────────────────────────────────────
    print(f"\n[Step 8] Assigning quintiles (rolling 60-cal-day window)...")
    events_q = assign_quintiles(events_surp)
    q_counts = events_q["quintile"].value_counts().sort_index()
    print(f"  Quintile distribution: {q_counts.to_dict()}")

    # ── Step 9: Beta + mcap at filing date ────────────────────────────────────
    print(f"\n[Step 9] Computing per-event betas...")
    events_full = compute_event_betas(events_q, adj_ret, ew_ret)

    # ── Step 10: IC (cost-independent) ───────────────────────────────────────
    print(f"\n[Step 10] Computing IC (Spearman: surprise vs 45-day hold return)...")
    ic_stats = compute_ic(events_full, raw_ret)
    print(f"  IC = {ic_stats['ic']:.4f}  t = {ic_stats['ic_tstat']:.3f}  N = {ic_stats['n_events']:,}")

    # ── Step 11: Calendar-time portfolio ─────────────────────────────────────
    print(f"\n[Step 11] Running calendar-time portfolio ({len(is_tdays):,} trading days)...")
    portfolio_results = run_portfolio(
        events_full, raw_ret, adj_ret, ew_ret, tdays_full, membership)

    # ── Step 12: Metrics ──────────────────────────────────────────────────────
    print("\n[Step 12] Computing metrics...")
    metrics = compute_metrics_monthly(portfolio_results["daily_records"])
    if not metrics:
        print("[ERROR] No metrics computed — no valid daily returns.")
        return

    # ── Step 13: CAAR ─────────────────────────────────────────────────────────
    print("\n[Step 13] Computing CAAR by quintile...")
    caar = compute_caar(events_full, adj_lvl, adj_ret, ew_ret, tdays_full)

    # ── Step 14: Output ───────────────────────────────────────────────────────
    print_results(metrics, ic_stats, portfolio_results, events_full)
    print_caar(caar)

    # Verdict line
    print(f"\n{'='*78}")
    ic_sig   = abs(ic_stats.get("ic_tstat", 0) or 0) > 2
    sr_ok    = (metrics.get("sharpe_net", 0) or 0) > (metrics.get("dsr_thr_ann", 99) or 99)
    drift_ok = False
    q1_v = caar.get(1, [])
    q5_v = caar.get(5, [])
    if q1_v and q5_v:
        final_q1 = q1_v[min(HOLD_DAYS-1, len(q1_v)-1)]
        final_q5 = q5_v[min(HOLD_DAYS-1, len(q5_v)-1)]
        drift_ok = (final_q1 - final_q5) > 0.01

    print(f"VERDICT: IC {'significant' if ic_sig else 'not significant'} | "
          f"DSR {'CLEARS' if metrics.get('clears_dsr') else 'fails'} | "
          f"Drift {'visible' if drift_ok else 'weak/absent'} | "
          f"Net Sharpe = {metrics.get('sharpe_net', np.nan):.3f}")
    if not sr_ok:
        print("  Family CLOSES as standalone. If IC is significant and drift is visible")
        print("  but net Sharpe fails DSR, cost structure or regime issues are the culprit.")
        print("  A revenue/cash-flow surprise variant (not accruals) could be tested as S2b.")
    else:
        print("  Trial PASSES in-sample gate. Log as pead_S2.")

    # ── Step 15: Save & registry ──────────────────────────────────────────────
    print("\n[Step 15] Saving results...")
    elapsed = time.time() - t_total
    print(f"  Total runtime: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    save_results(portfolio_results, events_full, caar, metrics)
    log_registry(metrics, ic_stats, events_full,
                 portfolio_results["guard_fire_count"])


if __name__ == "__main__":
    main()
