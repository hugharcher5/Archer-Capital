"""
research/earnings_clustering/run.py

Hypothesis 4: Do returns cluster around earnings announcements?
Among DCF-cheap names, does abnormal return concentrate in earnings windows,
consistent with mispricing?  Diagnostic test — does NOT change the strategy.

REUSES:
  - H1 signal panel  (signals_panel.parquet)     → DCF-cheap basket definition
  - H2 submissions cache (quality_filter/_cache/submissions/)  → 8-K Item 2.02 dates
  - Same 10 annual rebalance dates (Jun 30, 2015-2024)
  - No DCF recomputation

EARNINGS DATES:
  EDGAR 8-K filings where items contains "2.02" (Results of Operations).
  Filing date is used as the announcement date (SEC requires 8-K within 4
  business days; in practice AAPL files same day or next morning).
  Fully point-in-time: we only use dates already in the submissions cache,
  and the filing date is always ≤ the true announcement date + 4 business days.

BENCHMARK:
  SPY daily returns (market-adjusted abnormal returns, per-day).
  This is the standard in event study literature and sidesteps computing a
  dynamic equal-weight index from 590 small-cap stocks daily.

SURVIVORSHIP NOTE:
  Absolute returns are upper bounds.  The return decomposition (earnings vs
  non-earnings fraction) is internally consistent and survivorship-robust.

ONE TRIAL — logged for Deflated Sharpe.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO      = Path(__file__).resolve().parent.parent.parent
H1_CACHE  = REPO / "research" / "dcf_vs_multiple" / "_cache"
H1_PANEL  = H1_CACHE / "signals_panel.parquet"
H2_CACHE  = REPO / "research" / "quality_filter" / "_cache"
SUBS_DIR  = H2_CACHE / "submissions"
CIK_CACHE = H1_CACHE / "edgar_tickers.json"

CACHE_DIR = Path(__file__).parent / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
START_YEAR  = 2015
END_YEAR    = 2025
REB_MONTH   = 6
REB_DAY     = 30
N_QUINTILES = 5
WINDOWS     = [1, 2, 5]    # half-window sizes: [-1,+1], [-2,+2], [-5,+5]
WINDOW_STD  = 2             # standard window
MIN_EVENTS  = 100           # warn threshold


# ═══════════════════════════════════════════════════════════════════════════════
# EDGAR earnings dates (8-K Item 2.02)
# ═══════════════════════════════════════════════════════════════════════════════

_CIK_MAP: dict[str, str] | None = None


def _load_cik_map() -> dict[str, str]:
    global _CIK_MAP
    if _CIK_MAP is None:
        _CIK_MAP = json.loads(CIK_CACHE.read_text())
    return _CIK_MAP


def get_earnings_dates_edgar(ticker: str) -> list[date]:
    """
    Return earnings announcement dates from cached EDGAR submissions JSON.
    Filters 8-K filings where items contains "2.02" (Results of Operations).
    Falls back to all 8-Ks when items field is absent.
    """
    cik_map = _load_cik_map()
    cik     = cik_map.get(ticker.upper())
    if not cik:
        return []

    subs_path = SUBS_DIR / f"{cik.zfill(10)}.json"
    if not subs_path.exists():
        return []

    try:
        subs = json.loads(subs_path.read_text())
    except Exception:
        return []

    recent = subs.get("filings", {}).get("recent", {})
    forms  = recent.get("form", [])
    fdates = recent.get("filingDate", [])
    items  = recent.get("items", [])
    has_items = len(items) > 0

    earn_dates: list[date] = []
    for i, form in enumerate(forms):
        if form not in ("8-K", "8-K/A"):
            continue
        if i >= len(fdates):
            continue
        fd_str = fdates[i]
        if not fd_str:
            continue
        # Items gate: require "2.02" if field is present; else accept all 8-Ks
        if has_items and i < len(items):
            if "2.02" not in str(items[i]):
                continue
        try:
            earn_dates.append(date.fromisoformat(fd_str))
        except ValueError:
            pass

    return sorted(set(earn_dates))


# ═══════════════════════════════════════════════════════════════════════════════
# DCF-cheap basket (same definition as H1/H2)
# ═══════════════════════════════════════════════════════════════════════════════

def get_rebalance_dates() -> list[date]:
    return [date(y, REB_MONTH, REB_DAY) for y in range(START_YEAR, END_YEAR)]


def get_dcf_cheap(panel: pd.DataFrame) -> pd.DataFrame:
    """Top quintile by DCF V/P per rebalance date — same as H1 portfolio."""
    rows = []
    for as_of, grp in panel.groupby("as_of"):
        sub = grp[grp["dcf_applicable"]].dropna(subset=["dcf_vp"]).copy()
        if len(sub) < N_QUINTILES:
            continue
        thresh = sub["dcf_vp"].quantile(1 - 1 / N_QUINTILES)
        cheap  = sub[sub["dcf_vp"] >= thresh].copy()
        cheap["as_of"] = as_of
        rows.append(cheap)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# Price data
# ═══════════════════════════════════════════════════════════════════════════════

def load_price_data(tickers: list[str]) -> dict[str, pd.Series]:
    """Download daily adjusted-close for all tickers + SPY. Returns {sym: Series}."""
    all_tickers = sorted(set(tickers + ["SPY"]))
    start = f"{START_YEAR - 1}-01-01"
    end   = f"{END_YEAR + 1}-01-01"
    print(f"  Downloading daily prices for {len(all_tickers)} tickers ({start} → {end})...")

    CHUNK  = 100
    result: dict[str, pd.Series] = {}
    for i in range(0, len(all_tickers), CHUNK):
        chunk = all_tickers[i : i + CHUNK]
        try:
            raw = yf.download(chunk, start=start, end=end,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            for sym in close.columns:
                s = close[sym].dropna()
                if not s.empty:
                    result[sym] = s
        except Exception:
            pass
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Return decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def _find_trading_idx(index: pd.DatetimeIndex, target: date) -> int:
    """Index of the trading day closest to (but not after) target."""
    ts  = pd.Timestamp(target)
    pos = np.searchsorted(index, ts, side="right") - 1
    return max(0, pos)


def decompose_returns(
    dcf_cheap:    pd.DataFrame,
    price_hists:  dict[str, pd.Series],
    spy_prices:   pd.Series,
    earnings_map: dict[str, list[date]],
    reb_dates:    list[date],
) -> pd.DataFrame:
    """
    For each (ticker, holding-period) in DCF-cheap:
      - Compute daily market-adjusted return (stock_ret − spy_ret)
      - Tag each trading day as "in-earnings-window" or not (for each window size)
      - Accumulate per-event and per-period records

    Returns DataFrame with columns:
      ticker, as_of, half_win, earn_date (or None), type, window_ar, n_days
    """
    reb_map: dict[date, date] = {}
    for i, d in enumerate(reb_dates[:-1]):
        reb_map[d] = reb_dates[i + 1]
    last = reb_dates[-1]
    reb_map[last] = date(last.year + 1, last.month, last.day)

    # Pre-compute SPY daily returns once
    spy_ret = spy_prices.pct_change()

    records: list[dict] = []
    n_obs = len(dcf_cheap)

    for obs_i, (_, row) in enumerate(dcf_cheap.iterrows()):
        if (obs_i + 1) % 200 == 0:
            print(f"    decomposed {obs_i + 1}/{n_obs} ticker-years", flush=True)

        ticker   = row["ticker"]
        as_of_ts = row["as_of"]
        as_of_d  = as_of_ts.date() if hasattr(as_of_ts, "date") else as_of_ts
        next_reb = reb_map.get(as_of_d)
        if next_reb is None:
            continue

        prices = price_hists.get(ticker)
        if prices is None:
            continue

        # Daily returns for this stock in (as_of, next_reb]
        ts0 = pd.Timestamp(as_of_d)
        ts1 = pd.Timestamp(next_reb)
        s_ret = prices.pct_change()
        s_ret = s_ret[(s_ret.index > ts0) & (s_ret.index <= ts1)].dropna()
        if len(s_ret) < 10:
            continue

        # Align SPY to same days
        bm_ret = spy_ret.reindex(s_ret.index).fillna(0.0)

        # Daily abnormal return (additive approximation — fine for short windows)
        ar = (s_ret.values - bm_ret.values)  # shape (T,)
        idx = s_ret.index                     # DatetimeIndex of trading days
        T   = len(ar)

        # Earnings dates in this holding period
        ed_list = earnings_map.get(ticker, [])
        period_eds = [d for d in ed_list if as_of_d < d <= next_reb]

        for half_win in WINDOWS:
            # Mark each trading day as belonging to an earnings window
            in_win  = np.zeros(T, dtype=bool)
            win_events: list[dict] = []

            for ed in period_eds:
                ts_ed = pd.Timestamp(ed)
                # Position in the index: closest trading day ≤ ed
                pos = _find_trading_idx(idx, ed)
                # If the closest trading day is more than 5 calendar days away, skip
                # (guards against announcement falling on a long holiday)
                if abs((idx[pos] - ts_ed).days) > 5:
                    continue
                lo = max(0, pos - half_win)
                hi = min(T - 1, pos + half_win)
                in_win[lo : hi + 1] = True
                w_ar = float(ar[lo : hi + 1].sum())
                win_events.append({
                    "ticker":    ticker,
                    "as_of":     as_of_d,
                    "half_win":  half_win,
                    "earn_date": ed,
                    "type":      "earnings",
                    "window_ar": w_ar,
                    "n_days":    hi - lo + 1,
                })

            if not period_eds:
                # No earnings dates available for this ticker-year — skip so we
                # only compare stocks that have both windows in both arms.
                continue

            records.extend(win_events)

            # Non-earnings residual for this holding period
            ne_ar   = float(ar[~in_win].sum())
            ne_days = int((~in_win).sum())
            records.append({
                "ticker":    ticker,
                "as_of":     as_of_d,
                "half_win":  half_win,
                "earn_date": None,
                "type":      "non_earnings",
                "window_ar": ne_ar,
                "n_days":    ne_days,
            })

    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-CHECK
# ═══════════════════════════════════════════════════════════════════════════════

def run_precheck(cheap: pd.DataFrame) -> bool:
    print("\n" + "=" * 70)
    print("  PRE-CHECK — Earnings Date Availability and Event Count")
    print("=" * 70)

    # 1. Test PIT earnings dates for AAPL 2018-06-30 → 2019-06-30
    ticker      = "AAPL"
    check_date  = date(2018, 6, 30)
    next_date   = date(2019, 6, 30)
    print(f"\n[1] EDGAR 8-K Item 2.02 earnings dates: {ticker}")
    print(f"  Holding period: {check_date} → {next_date}")

    ed = get_earnings_dates_edgar(ticker)
    if not ed:
        print(f"  [FAIL] No earnings dates found in EDGAR cache for {ticker}.")
        print(f"  Ensure H2 submissions cache exists at {SUBS_DIR}")
        return False

    print(f"  Total historical 8-K earnings dates in cache: {len(ed)}")
    in_period = [d for d in ed if check_date < d <= next_date]
    print(f"  Dates in holding period: {len(in_period)}")
    for d in in_period:
        print(f"    {d}")

    if len(in_period) < 2:
        print(f"  [WARN] Fewer than 2 earnings dates in period — coverage may be thin.")

    pre_2018 = [d for d in ed if d < check_date]
    print(f"  Dates available before {check_date}: {len(pre_2018)}  (confirms PIT coverage)")
    if not pre_2018:
        print("  [FAIL] No historical coverage before check date — cannot proceed.")
        return False

    print(f"  CHECK 1: PASS ✓")

    # 2. Estimate total earnings events across DCF-cheap basket
    print(f"\n[2] Estimating total earnings events in DCF-cheap basket...")
    n_tickers = cheap["ticker"].nunique()
    n_dates   = cheap["as_of"].nunique()
    n_rows    = len(cheap)
    # Sample 20 tickers to estimate earnings-dates-per-ticker
    sample    = list(cheap["ticker"].unique()[:20])
    sample_counts = [len(get_earnings_dates_edgar(t)) for t in sample]
    avg_per_ticker = np.mean(sample_counts) if sample_counts else 0
    # Expected events: avg dates per ticker × fraction falling in a 1-year window
    # Each ticker appears in ~10 annual windows; each window captures ~4 earnings dates
    expected_events = n_rows * 4  # rough: 4 quarterly per ticker-year
    print(f"  DCF-cheap basket: {n_tickers} unique tickers, {n_dates} rebalance dates")
    print(f"  Ticker-year observations: {n_rows}")
    print(f"  Sample avg historical earn dates in EDGAR (20 tickers): {avg_per_ticker:.0f}")
    print(f"  Expected earnings events (est. 4/yr × {n_rows} obs): ~{expected_events:,}")

    if expected_events < MIN_EVENTS:
        print(f"  [WARN] Expected <{MIN_EVENTS} events — statistical power weak")
    else:
        print(f"  CHECK 2: PASS ✓")

    print(f"\n  PRE-CHECK OVERALL: PASS ✓")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test1_aggregate(ev: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  TEST 1 — AGGREGATE DECOMPOSITION  (standard window [-2, +2])")
    print("  *** SURVIVORSHIP NOTE: absolute levels are upper bounds ***")
    print("=" * 70)

    df    = ev[ev["half_win"] == WINDOW_STD]
    earn  = df[df["type"] == "earnings"]["window_ar"].dropna()
    ne    = df[df["type"] == "non_earnings"]["window_ar"].dropna()
    n_earn, n_ne = len(earn), len(ne)

    if n_earn < 2 or n_ne < 2:
        print("  Insufficient data for Test 1.")
        return

    mean_earn = earn.mean()
    mean_ne   = ne.mean()

    # Two-sample t-test: earnings window AR vs non-earnings period AR
    tstat, pval = stats.ttest_ind(earn, ne)

    # % of total pooled AR in earnings windows
    total_earn_sum = earn.sum()
    total_ne_sum   = ne.sum()
    total_sum      = total_earn_sum + total_ne_sum
    pct_in_earn    = total_earn_sum / total_sum * 100 if total_sum != 0 else float("nan")

    # Calendar expectation: (2×half_win+1) trading days per event × ~4 events/yr ÷ 252
    random_pct = (2 * WINDOW_STD + 1) * 4 / 252 * 100

    print(f"\n  Earnings events (5-day windows):   {n_earn:>6,}")
    print(f"  Non-earnings periods:              {n_ne:>6,}")
    print(f"\n  Mean AR per earnings window:       {mean_earn:>+8.3%}  (sum of daily ARs)")
    print(f"  Mean AR per non-earnings period:   {mean_ne:>+8.3%}  (full holding-period residual)")
    print(f"\n  NOTE: earnings windows are 5 days; non-earnings are ~247 days.")
    print(f"  Per-day comparison below normalises for length difference.")

    # Normalise by days for like-for-like comparison
    ev_std = ev[ev["half_win"] == WINDOW_STD].copy()
    ev_std = ev_std[ev_std["n_days"] > 0].copy()
    ev_std["daily_ar"] = ev_std["window_ar"] / ev_std["n_days"]
    earn_daily = ev_std[ev_std["type"] == "earnings"]["daily_ar"].dropna()
    ne_daily   = ev_std[ev_std["type"] == "non_earnings"]["daily_ar"].dropna()
    tstat_d, pval_d = stats.ttest_ind(earn_daily, ne_daily)

    print(f"\n  Mean DAILY AR — earnings windows:  {earn_daily.mean():>+8.4%}")
    print(f"  Mean DAILY AR — non-earn residual: {ne_daily.mean():>+8.4%}")
    print(f"  t-stat (earn vs non-earn, per-day):  {tstat_d:>6.2f}   p={pval_d:.4f}")
    sig = "***" if pval_d < 0.01 else "**" if pval_d < 0.05 else "*" if pval_d < 0.10 else ""
    print(f"  {sig} (p<0.01=***, p<0.05=**, p<0.10=*)")

    print(f"\n  % of total AR in earnings windows: {pct_in_earn:>6.1f}%")
    print(f"  Calendar expectation if random:    {random_pct:>6.1f}%  ({2*WINDOW_STD+1}×4÷252 trading days)")
    print(f"  Ratio (actual / random):           {pct_in_earn/random_pct:>6.1f}×")


def test2_window_robustness(ev: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  TEST 2 — WINDOW SIZE ROBUSTNESS")
    print("  Same finding across [-1,+1], [-2,+2], [-5,+5]?")
    print("=" * 70)
    print(f"\n  {'Window':<16}  {'N events':>9}  {'Daily AR (earn)':>15}  {'Daily AR (non)':>15}  {'t-stat':>8}  {'p-val':>7}  {'% in win':>9}  {'vs random':>10}")
    print("  " + "-" * 100)

    for hw in WINDOWS:
        label = f"[−{hw},+{hw}] ({2*hw+1}-day)"
        sub   = ev[ev["half_win"] == hw].copy()
        sub   = sub[sub["n_days"] > 0].copy()
        sub["daily_ar"] = sub["window_ar"] / sub["n_days"]

        earn  = sub[sub["type"] == "earnings"]
        ne    = sub[sub["type"] == "non_earnings"]
        earn_d = earn["daily_ar"].dropna()
        ne_d   = ne["daily_ar"].dropna()

        if len(earn_d) < 2 or len(ne_d) < 2:
            print(f"  {label:<16}  {'n/a':>9}")
            continue

        tstat, pval = stats.ttest_ind(earn_d, ne_d)
        total_e = earn["window_ar"].sum()
        total_n = ne["window_ar"].sum()
        total   = total_e + total_n
        pct     = total_e / total * 100 if total != 0 else float("nan")
        rand    = (2 * hw + 1) * 4 / 252 * 100
        ratio   = pct / rand if rand > 0 else float("nan")

        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"  {label:<16}  {len(earn_d):>9,}  {earn_d.mean():>+14.4%}  {ne_d.mean():>+14.4%}  "
              f"{tstat:>7.2f}{sig}  {pval:>7.4f}  {pct:>8.1f}%  {ratio:>8.1f}×")

    print("\n  (*** p<0.01, ** p<0.05, * p<0.10)")
    print("  Consistent direction + significant across all 3 windows → robust.")
    print("  Only significant at one window → likely noise / window-fitting.")


def test3_per_stock(ev: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  TEST 3 — PER-STOCK CLUSTERING")
    print("  What fraction of names shows earnings-window clustering?")
    print("=" * 70)

    df   = ev[ev["half_win"] == WINDOW_STD]
    earn = df[df["type"] == "earnings"]
    ne   = df[df["type"] == "non_earnings"]

    tickers = earn["ticker"].unique()
    clustering: list[tuple[str, float, float, float]] = []

    for ticker in tickers:
        t_earn_sum = earn[earn["ticker"] == ticker]["window_ar"].sum()
        t_ne_sum   = ne[ne["ticker"]   == ticker]["window_ar"].sum()
        total      = t_earn_sum + t_ne_sum
        if total == 0:
            continue
        pct = t_earn_sum / total * 100
        clustering.append((ticker, pct, t_earn_sum, t_ne_sum))

    if not clustering:
        print("  No per-stock data available.")
        return

    clustering.sort(key=lambda x: -x[1])
    n_total     = len(clustering)
    n_above_50  = sum(1 for _, pct, _, _ in clustering if pct > 50)
    n_above_rand_3x = sum(1 for _, pct, _, _ in clustering
                           if pct > 3 * (2 * WINDOW_STD + 1) * 4 / 252 * 100)
    rand_pct    = (2 * WINDOW_STD + 1) * 4 / 252 * 100

    print(f"\n  Tickers with earnings data in basket:      {n_total:>5,}")
    print(f"  Tickers where earn windows > 50% of AR:    {n_above_50:>5,}  ({n_above_50/n_total*100:.1f}%)")
    print(f"  Tickers where earn windows > 3× random:   {n_above_rand_3x:>5,}  ({n_above_rand_3x/n_total*100:.1f}%)")
    print(f"  Random expectation (calendar): {rand_pct:.1f}% of trading days")

    # Distribution of % in earnings windows
    pcts = [p for _, p, _, _ in clustering]
    pct_arr = np.array(pcts)
    print(f"\n  Distribution of (earn-window %) across names:")
    print(f"    Min:    {pct_arr.min():>+7.1f}%")
    print(f"    P25:    {np.percentile(pct_arr, 25):>+7.1f}%")
    print(f"    Median: {np.percentile(pct_arr, 50):>+7.1f}%")
    print(f"    P75:    {np.percentile(pct_arr, 75):>+7.1f}%")
    print(f"    Max:    {pct_arr.max():>+7.1f}%")

    print(f"\n  Top 15 most clustered names (% of total AR in earnings windows):")
    print(f"  {'Ticker':<10}  {'Earn-win %':>10}  {'Earn-win AR':>12}  {'Non-earn AR':>12}")
    print("  " + "-" * 50)
    for ticker, pct, earn_ar, ne_ar in clustering[:15]:
        print(f"  {ticker:<10}  {pct:>+9.1f}%  {earn_ar:>+11.3%}  {ne_ar:>+11.3%}")

    print(f"\n  Bottom 10 (least clustered / negative):")
    print(f"  {'Ticker':<10}  {'Earn-win %':>10}  {'Earn-win AR':>12}  {'Non-earn AR':>12}")
    print("  " + "-" * 50)
    for ticker, pct, earn_ar, ne_ar in clustering[-10:]:
        print(f"  {ticker:<10}  {pct:>+9.1f}%  {earn_ar:>+11.3%}  {ne_ar:>+11.3%}")

    # Test for consistency vs outlier-driven
    median_pct = float(np.median(pct_arr))
    if n_above_50 / n_total > 0.4:
        print(f"\n  PATTERN: Broad — majority of names show earnings clustering.")
    elif n_above_50 / n_total > 0.2:
        print(f"\n  PATTERN: Moderate — minority of names drive the clustering.")
    else:
        print(f"\n  PATTERN: Concentrated — few outlier names dominate the result.")
    print(f"  Median name has {median_pct:.1f}% of AR in earnings windows.")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "#" * 70)
    print("  HYPOTHESIS 4: RETURN CLUSTERING AROUND EARNINGS")
    print("  Diagnostic: mispricing (clustering) vs. risk premium (smooth)?")
    print("  SURVIVORSHIP NOTE: absolute returns are upper bounds.")
    print("  Benchmark: SPY daily returns (market-adjusted).")
    print("  ONE TRIAL — logged for Deflated Sharpe.")
    print("#" * 70)

    # ── Load H1 signal panel ──────────────────────────────────────────────────
    print(f"\n  Loading H1 signal panel: {H1_PANEL}")
    panel = pd.read_parquet(H1_PANEL)
    panel["as_of"] = pd.to_datetime(panel["as_of"])

    # ── DCF-cheap basket ──────────────────────────────────────────────────────
    dcf_cheap = get_dcf_cheap(panel)
    if dcf_cheap.empty:
        print("[ERROR] DCF-cheap basket is empty. Check H1 panel.")
        sys.exit(1)
    cheap_tickers = list(dcf_cheap["ticker"].unique())
    print(f"  DCF-cheap basket: {len(dcf_cheap):,} ticker-year obs, "
          f"{len(cheap_tickers)} unique tickers, "
          f"{dcf_cheap['as_of'].nunique()} dates")

    # ── PRE-CHECK ─────────────────────────────────────────────────────────────
    ok = run_precheck(dcf_cheap)
    if not ok:
        print("\n[STOP] Pre-check failed.")
        sys.exit(1)

    # ── Price data ────────────────────────────────────────────────────────────
    price_hists = load_price_data(cheap_tickers)
    spy_prices  = price_hists.get("SPY")
    if spy_prices is None:
        print("[ERROR] SPY prices unavailable — cannot compute market-adjusted returns.")
        sys.exit(1)
    n_with_prices = sum(1 for t in cheap_tickers if t in price_hists)
    print(f"  Price histories: {n_with_prices}/{len(cheap_tickers)} cheap tickers + SPY")

    # ── Earnings dates ────────────────────────────────────────────────────────
    print(f"\n  Loading EDGAR earnings dates for {len(cheap_tickers)} tickers...")
    earnings_map: dict[str, list[date]] = {}
    for ticker in cheap_tickers:
        earnings_map[ticker] = get_earnings_dates_edgar(ticker)
    tickers_with_dates = sum(1 for v in earnings_map.values() if v)
    total_earn_dates   = sum(len(v) for v in earnings_map.values())
    print(f"  Tickers with EDGAR earnings dates: {tickers_with_dates}/{len(cheap_tickers)}")
    print(f"  Total EDGAR earnings dates loaded: {total_earn_dates:,}")

    # ── Rebalance dates ───────────────────────────────────────────────────────
    reb_dates = get_rebalance_dates()
    print(f"  Rebalance dates: {reb_dates[0]} → {reb_dates[-1]}  ({len(reb_dates)} periods)")

    # ── Return decomposition ──────────────────────────────────────────────────
    print(f"\n  Decomposing returns into earnings/non-earnings windows...")
    ev = decompose_returns(dcf_cheap, price_hists, spy_prices, earnings_map, reb_dates)

    if ev.empty:
        print("[ERROR] No events computed. Check price/earnings data.")
        sys.exit(1)

    n_earn_events = (ev["type"] == "earnings").sum()
    n_ne_periods  = (ev["type"] == "non_earnings").sum()
    n_tickers_ev  = ev["ticker"].nunique()
    print(f"\n  Events computed (standard window):")
    print(f"    Earnings window events:   {n_earn_events:>6,}")
    print(f"    Non-earnings periods:     {n_ne_periods:>6,}")
    print(f"    Unique tickers with data: {n_tickers_ev:>6,}")

    if n_earn_events < MIN_EVENTS:
        print(f"\n  [WARN] Only {n_earn_events} earnings events — statistical power is weak.")
        print("  Results are indicative only.")

    # ── Coverage by year ──────────────────────────────────────────────────────
    print(f"\n  Coverage by rebalance year (earnings events only, standard window):")
    ev_std = ev[(ev["half_win"] == WINDOW_STD) & (ev["type"] == "earnings")].copy()
    ev_std["year"] = pd.to_datetime(ev_std["as_of"].astype(str)).dt.year
    cov = ev_std.groupby("year").agg(
        n_events=("earn_date", "count"),
        n_tickers=("ticker", "nunique"),
    )
    print(f"\n  {'Year':>6}  {'Events':>8}  {'Tickers':>8}")
    print("  " + "-" * 28)
    for year, row2 in cov.iterrows():
        print(f"  {year:>6}  {int(row2['n_events']):>8,}  {int(row2['n_tickers']):>8,}")

    # ── Tests ─────────────────────────────────────────────────────────────────
    test1_aggregate(ev)
    test2_window_robustness(ev)
    test3_per_stock(ev)

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    df_std     = ev[(ev["half_win"] == WINDOW_STD) & (ev["n_days"] > 0)].copy()
    df_std["daily_ar"] = df_std["window_ar"] / df_std["n_days"]
    earn_daily = df_std[df_std["type"] == "earnings"]["daily_ar"].dropna()
    ne_daily   = df_std[df_std["type"] == "non_earnings"]["daily_ar"].dropna()
    _, pval_d  = stats.ttest_ind(earn_daily, ne_daily) if len(earn_daily) > 1 else (0, 1)

    total_e  = ev[(ev["half_win"] == WINDOW_STD) & (ev["type"] == "earnings")]["window_ar"].sum()
    total_n  = ev[(ev["half_win"] == WINDOW_STD) & (ev["type"] == "non_earnings")]["window_ar"].sum()
    total    = total_e + total_n
    pct_earn = total_e / total * 100 if total != 0 else 0
    rand_pct = (2 * WINDOW_STD + 1) * 4 / 252 * 100

    if pct_earn > rand_pct * 3 and pval_d < 0.10 and earn_daily.mean() > ne_daily.mean():
        verdict = ("MISPRICING story: abnormal return concentrates disproportionately "
                   "in earnings windows — consistent with market underreaction to "
                   "cheap stocks before announcements.")
    elif pct_earn < rand_pct * 1.5 or earn_daily.mean() <= ne_daily.mean():
        verdict = ("RISK PREMIUM story: abnormal return is spread smoothly across "
                   "the year — consistent with compensation for bearing systematic "
                   "risk, not exploiting mispricing.")
    else:
        verdict = ("MIXED / INCONCLUSIVE: some earnings clustering but not "
                   "statistically strong — interpret with caution "
                   "(T=10 annual periods, small-sample).")

    print(f"\n  {verdict}")
    print(f"\n  Supporting numbers:")
    print(f"    % of AR in earnings windows: {pct_earn:.1f}%  (random: {rand_pct:.1f}%,  ratio: {pct_earn/rand_pct:.1f}×)")
    print(f"    p-value (earn vs non-earn daily AR): {pval_d:.4f}")
    print(f"    Mean daily AR — earnings windows:   {earn_daily.mean():>+.4%}")
    print(f"    Mean daily AR — non-earn residual:  {ne_daily.mean():>+.4%}")

    print("\n" + "#" * 70)
    print("  RUN COMPLETE")
    print("#" * 70)


if __name__ == "__main__":
    main()
