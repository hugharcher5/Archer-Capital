"""
Event study and threshold backtest for Private Competitor Distress strategy.
This is the Layer 1 implementation — real data replacing the placeholder.

Two event sources:
  1. KNOWN_EVENTS  — hardcoded dated list (user should review dates/mappings)
  2. Auto-detected — rising-edge threshold crossings in max_distress time series

Produces:
  results/event_study_results.csv   — per-event CAR table + aggregate stats
  results/layer1_equity_curve.csv   — weekly abnormal equity curve
  results/layer1_metrics.json       — Sharpe, MDD, win-rate, trades, CAGR

CONSTRAINTS (enforced):
  - No look-ahead: entry at NEXT trading day after event, never same day.
  - All returns are ABNORMAL (excess vs peer basket), never raw.
  - Threshold=0.35, hold=90 days are fixed parameters; no tuning in this build.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# Make repo root importable
_REPO = Path(__file__).parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance not installed: pip install yfinance")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paths ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"

# ── Parameters (fixed — tuning belongs in walk-forward, not here) ──────────────

THRESHOLD      = 0.35   # rising-edge entry
HALF_THRESHOLD = 0.175  # mean-reversion exit
HOLD_DAYS      = 90     # calendar-day max hold
TC_RT_BPS      = 30     # round-trip transaction cost in bps (placeholder)
TC_RT          = TC_RT_BPS / 10_000

CAR_WINDOWS    = (21, 63, 126)   # trading-day horizons for event study

# ── LSE ticker mapping ─────────────────────────────────────────────────────────
# Internal strategy codes → real LSE ticker symbols

_LSE_REMAP: dict[str, str] = {
    # Internal code → yfinance LSE symbol (without .L).
    # Marston's trades as MARS.L on yfinance; no remap needed.
    # Add entries here only when the internal code genuinely differs from LSE ticker.
}

_ALL_TICKERS: list[str] = [
    "OTB", "MACF", "MARS", "MER", "CARD", "FORT", "BOOT", "NXR",
    "HFD", "SHI", "LSL", "SFR", "GYM", "DFS", "VP", "SDY",
    "MJG", "CRST", "FOXT", "ECEL", "FSTA", "HEAD",
]


def _lse(t: str) -> str:
    """Internal ticker → LSE symbol (with .L suffix)."""
    return _LSE_REMAP.get(t, t) + ".L"


def _internal(lse: str) -> str:
    """LSE symbol → internal ticker."""
    sym = lse.replace(".L", "")
    inv = {v: k for k, v in _LSE_REMAP.items()}
    return inv.get(sym, sym)


# ── Known events (user should review these dates and mappings) ─────────────────
# Format: (event_date, listed_ticker, trigger_name, note)
# event_date = date the distress event became public knowledge
# listed_ticker = internal code of the listed company expected to benefit
# Enter at NEXT TRADING DAY after event_date — enforced in CAR computation.

KNOWN_EVENTS: list[tuple[date, str, str, str]] = [
    # Paperchase: UK stationery/greetings chain → direct rival to Card Factory
    (date(2023, 1, 31),  "CARD",  "Paperchase",
     "Paperchase administration Jan 2023 — stationery/greetings sector"),

    # Carpetright: flooring retailer → benefit to Headlam (wholesale supply)
    (date(2023, 6,  1),  "HEAD",  "CPR Realisations Ltd",
     "Carpetright administration Jun 2023 — flooring/carpets sector"),

    # CTD Tiles: tile specialist → Headlam flooring adjacent; NXR bathrooms
    (date(2024, 8,  1),  "HEAD",  "CTD Tiles",
     "CTD Tiles administration Aug 2024 — tiles/flooring sector"),

    # Loungers: café-bar chain taken private by Fortress → pub sector benefits
    (date(2025, 1,  1),  "MARS",  "Loungers plc",
     "Loungers go-private Jan 2025 — café-bar removes listed competitor"),
    (date(2025, 1,  1),  "FSTA",  "Loungers plc",
     "Loungers go-private Jan 2025 — pub/bar sector"),
]


# ── Price data ─────────────────────────────────────────────────────────────────

def pull_price_data(
    tickers: list[str] | None = None,
    start: str = "2009-01-01",
) -> pd.DataFrame:
    """
    Pull daily adjusted close from yfinance for all listed tickers.
    Returns wide DataFrame indexed by date, columns = INTERNAL ticker codes.
    Missing / unresolvable tickers are logged and omitted.
    """
    if tickers is None:
        tickers = _ALL_TICKERS

    lse_symbols = [_lse(t) for t in tickers]
    log.info("Downloading price data for %d tickers …", len(lse_symbols))

    try:
        raw = yf.download(
            lse_symbols,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    except Exception as exc:
        log.error("yfinance download failed: %s", exc)
        return pd.DataFrame()

    # Extract 'Close' — handle both single and multi-ticker outputs
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        # Single ticker returned flat
        close = raw[["Close"]].copy()
        close.columns = [lse_symbols[0]]

    # Remap columns back to internal codes and drop empty columns
    internal_cols: dict[str, str] = {_lse(t): t for t in tickers}
    close.rename(columns=internal_cols, inplace=True)
    close = close[[c for c in close.columns if c in set(tickers)]]
    close.index = pd.to_datetime(close.index)

    valid = close.columns[close.notna().sum() > 0].tolist()
    missing = [t for t in tickers if t not in valid]
    if missing:
        log.warning("No price data for: %s", missing)

    log.info("Price data: %d tickers, %s → %s",
             len(valid), close.index.min().date(), close.index.max().date())
    return close[valid]


# ── Peer basket ────────────────────────────────────────────────────────────────

def peer_basket_returns(
    daily_returns: pd.DataFrame,
    exclude: str,
) -> pd.Series:
    """Equal-weighted daily returns of all tickers EXCEPT `exclude`."""
    peers = [c for c in daily_returns.columns if c != exclude]
    if not peers:
        return pd.Series(0.0, index=daily_returns.index)
    return daily_returns[peers].mean(axis=1)


# ── CAR computation ────────────────────────────────────────────────────────────

def compute_car(
    ticker: str,
    event_date: date,
    daily_prices: pd.DataFrame,
    windows: tuple[int, ...] = CAR_WINDOWS,
) -> dict:
    """
    Compute cumulative abnormal return at each window horizon.

    Entry: NEXT trading day after event_date (no same-day look-ahead).
    Abnormal = cumulative ticker return − cumulative peer basket return.

    Returns dict with keys:
        entry_date, n_days_available, skipped, skip_reason,
        car_21, car_63, car_126  (NaN if insufficient data)
    """
    result: dict = {
        "entry_date":       None,
        "n_days_available": 0,
        "skipped":          False,
        "skip_reason":      "",
        **{f"car_{w}": np.nan for w in windows},
    }

    if ticker not in daily_prices.columns:
        result["skipped"]     = True
        result["skip_reason"] = f"no price data for {ticker}"
        return result

    # Find first trading day STRICTLY AFTER event_date
    event_ts = pd.Timestamp(event_date)
    future_dates = daily_prices.index[daily_prices.index > event_ts]
    if future_dates.empty:
        result["skipped"]     = True
        result["skip_reason"] = "event date after price data ends"
        return result

    # Guard: ticker must have actual (non-NaN) price data within 20 calendar
    # days of the event.  The price index covers 2009-today for ALL tickers
    # (NaN before IPO), so we cannot use daily_prices.index > event_ts alone —
    # we must check the first non-NaN price date for this specific ticker.
    returns   = daily_prices.pct_change()
    first_valid = returns[ticker].loc[future_dates[0]:].first_valid_index()

    if first_valid is None:
        result["skipped"]     = True
        result["skip_reason"] = f"no valid return data for {ticker} after event"
        return result

    if (first_valid - event_ts).days > 20:
        result["skipped"]     = True
        result["skip_reason"] = (
            f"{ticker} not listed at event date "
            f"(first non-NaN data {first_valid.date()} is "
            f"{(first_valid - event_ts).days}d after event)"
        )
        return result

    entry_ts = first_valid
    result["entry_date"] = entry_ts.date()

    ticker_rets = returns[ticker].loc[entry_ts:]
    basket_rets = peer_basket_returns(returns, exclude=ticker).loc[entry_ts:]

    abnormal  = ticker_rets - basket_rets
    valid_abn = abnormal.dropna()
    result["n_days_available"] = int(len(valid_abn))

    if len(valid_abn) == 0:
        result["skipped"]     = True
        result["skip_reason"] = f"no valid return data for {ticker} after entry"
        return result

    # Cumulative abnormal return over days with valid data only
    cum_abn = (1 + valid_abn).cumprod() - 1

    for w in windows:
        if len(cum_abn) >= w:
            result[f"car_{w}"] = round(float(cum_abn.iloc[w - 1]), 6)
        # else stays NaN (insufficient history for this window)

    return result


# ── Auto-detected threshold crossings ─────────────────────────────────────────

def detect_threshold_crossings(
    ts: pd.DataFrame,
    ticker_series: dict[str, pd.Series],  # ticker → weekly max_distress Series
    threshold: float = THRESHOLD,
) -> list[dict]:
    """
    Detect rising-edge crossings: was < threshold last week, >= threshold this week.
    Returns list of event dicts.
    """
    events: list[dict] = []

    for ticker, series in ticker_series.items():
        series = series.sort_index().dropna()
        if len(series) < 2:
            continue
        prev = series.shift(1)
        crossings = series[(prev < threshold) & (series >= threshold)]
        for ts_date, score in crossings.items():
            events.append({
                "event_date":  ts_date.date(),
                "ticker":      ticker,
                "score":       round(float(score), 4),
                "prev_score":  round(float(prev.loc[ts_date]), 4),
                "source":      "auto",
                "trigger":     f"max_distress crossed {threshold}",
                "note":        "",
            })

    return sorted(events, key=lambda e: e["event_date"])


# ── Full event study ───────────────────────────────────────────────────────────

def run_event_study(
    ts: pd.DataFrame,
    daily_prices: pd.DataFrame,
    known_events: list[tuple] = KNOWN_EVENTS,
    threshold: float = THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines known + auto-detected events, computes CARs for each.

    Returns
    -------
    per_event_df  : one row per event with event metadata + CARs
    aggregate_df  : mean/median CAR, % positive, t-stat per CAR window
    """
    # Build per-ticker weekly series from ts
    ticker_series: dict[str, pd.Series] = {}
    for ticker in ts["ticker"].unique():
        sub = ts[ts["ticker"] == ticker].set_index("date")["max_distress"].sort_index()
        ticker_series[ticker] = sub

    # Auto-detected events
    auto_events = detect_threshold_crossings(ts, ticker_series, threshold)
    log.info("Auto-detected %d threshold crossings.", len(auto_events))

    # Combine all events
    all_events: list[dict] = []

    for ev_date, ticker, trigger, note in known_events:
        all_events.append({
            "event_date": ev_date,
            "ticker":     ticker,
            "trigger":    trigger,
            "note":       note,
            "source":     "known",
            "score":      float(ticker_series.get(ticker, pd.Series()).get(
                pd.Timestamp(ev_date), np.nan) or np.nan),
            "prev_score": np.nan,
        })

    for ae in auto_events:
        all_events.append(ae)

    # Compute CAR for each event
    rows: list[dict] = []
    skipped: list[str] = []

    for ev in all_events:
        ev_date = ev["event_date"]
        ticker  = ev["ticker"]
        car     = compute_car(ticker, ev_date, daily_prices)

        if car["skipped"]:
            skipped.append(
                f"{ev_date} {ticker} ({ev['trigger']}): {car['skip_reason']}"
            )
            continue

        rows.append({
            "source":      ev["source"],
            "event_date":  ev_date,
            "ticker":      ticker,
            "trigger":     ev.get("trigger", ""),
            "note":        ev.get("note", ""),
            "score_at_event": round(ev.get("score", np.nan), 4) if not np.isnan(ev.get("score", np.nan)) else np.nan,
            "entry_date":  car["entry_date"],
            "n_days":      car["n_days_available"],
            **{f"car_{w}": car[f"car_{w}"] for w in CAR_WINDOWS},
        })

    if skipped:
        print(f"\n{'='*60}")
        print(f"  EVENTS DROPPED — missing price data ({len(skipped)})")
        print(f"{'='*60}")
        for s in skipped:
            print(f"  {s}")

    per_event_df = pd.DataFrame(rows)

    # Aggregate stats per CAR window
    agg_rows: list[dict] = []
    for w in CAR_WINDOWS:
        col = f"car_{w}"
        if col not in per_event_df.columns:
            continue
        vals = per_event_df[col].dropna()
        if len(vals) == 0:
            continue
        from scipy import stats as sp_stats
        tstat, pval = (np.nan, np.nan)
        if len(vals) >= 2:
            tstat, pval = sp_stats.ttest_1samp(vals, 0.0)
        agg_rows.append({
            "window_days":   w,
            "n_events":      len(vals),
            "mean_car":      round(float(vals.mean()), 4),
            "median_car":    round(float(vals.median()), 4),
            "pct_positive":  round(float((vals > 0).mean() * 100), 1),
            "t_stat":        round(float(tstat), 3) if not np.isnan(tstat) else np.nan,
            "p_value":       round(float(pval), 4) if not np.isnan(pval) else np.nan,
            "min_car":       round(float(vals.min()), 4),
            "max_car":       round(float(vals.max()), 4),
        })

    aggregate_df = pd.DataFrame(agg_rows)
    return per_event_df, aggregate_df


# ── Threshold portfolio backtest ───────────────────────────────────────────────

def run_threshold_backtest(
    ts: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    Weekly-rebalanced long portfolio.

    Entry  : rising-edge crossing (prev < THRESHOLD, curr >= THRESHOLD)
             entered at NEXT week's Friday close (no same-week look-ahead).
    Exit   : after HOLD_DAYS calendar days OR max_distress drops below HALF_THRESHOLD.
    Weight : equal across all open positions.
    Cost   : TC_RT applied as half on entry, half on exit (per position).

    Abnormal return per position = position_return − peer_basket_return.
    Portfolio return = equal-weighted mean of all abnormal position returns.
    When no positions open → abnormal return = 0.

    Returns
    -------
    equity_df  : DataFrame [date, n_positions, portfolio_return,
                             basket_return, abnormal_return, equity]
    metrics    : dict {sharpe, max_drawdown, win_rate, total_trades, cagr,
                       annotation, tc_bps}
    """
    # Build weekly returns from daily prices (Friday close)
    weekly_prices = daily_prices.resample("W-FRI").last()
    weekly_returns = weekly_prices.pct_change()

    # Build per-ticker weekly max_distress pivot
    ts_pivot = ts.pivot(index="date", columns="ticker", values="max_distress")
    ts_pivot.index = pd.to_datetime(ts_pivot.index)
    ts_pivot = ts_pivot.sort_index()

    # Align to weekly Friday schedule
    all_fridays = weekly_returns.index.intersection(ts_pivot.index)
    if len(all_fridays) == 0:
        log.error("No overlapping dates between price data and distress timeseries.")
        return pd.DataFrame(), {}

    ts_aligned    = ts_pivot.reindex(all_fridays)
    price_aligned = weekly_prices.reindex(all_fridays)
    ret_aligned   = weekly_returns.reindex(all_fridays)

    # Track open positions: ticker → (entry_date, entry_price)
    open_positions: dict[str, dict] = {}
    prev_distress: dict[str, float] = {}

    equity   = 1.0
    equity_rows: list[dict] = []
    trade_returns: list[float] = []   # per-trade abnormal return

    tc_half = TC_RT / 2  # cost per entry or exit leg

    for i, friday in enumerate(all_fridays):
        if i == 0:
            # Initialise previous distress
            for ticker in ts_aligned.columns:
                prev_distress[ticker] = float(ts_aligned[ticker].iloc[0] or 0)
            equity_rows.append({
                "date": friday, "n_positions": 0,
                "portfolio_return": 0.0, "basket_return": 0.0,
                "abnormal_return": 0.0, "equity": equity,
            })
            continue

        curr_dist: dict[str, float] = {
            ticker: float(val) if not np.isnan(val) else 0.0
            for ticker, val in ts_aligned.loc[friday].items()
            if ticker in ts_aligned.columns
        }

        # ── Process exits ────────────────────────────────────────────────────
        for ticker in list(open_positions.keys()):
            pos    = open_positions[ticker]
            held   = (friday - pos["entry_date"]).days

            # Compute this position's return (raw, for exit tracking)
            if ticker in ret_aligned.columns and not np.isnan(ret_aligned[ticker].loc[friday]):
                pos_ret = float(ret_aligned[ticker].loc[friday])
            else:
                pos_ret = 0.0

            basket_ret = float(peer_basket_returns(ret_aligned, exclude=ticker).loc[friday])
            abn_ret    = pos_ret - basket_ret

            exit_triggered = (
                held >= HOLD_DAYS
                or curr_dist.get(ticker, 0.0) < HALF_THRESHOLD
            )
            if exit_triggered:
                # Apply exit TC
                abn_ret -= tc_half
                pos["cumulative_abn"] = pos.get("cumulative_abn", 0.0) + abn_ret
                trade_returns.append(pos["cumulative_abn"])
                del open_positions[ticker]
            else:
                pos["cumulative_abn"] = pos.get("cumulative_abn", 0.0) + abn_ret

        # ── Process entries ──────────────────────────────────────────────────
        for ticker, cur_score in curr_dist.items():
            if ticker in open_positions:
                continue
            prev_score = prev_distress.get(ticker, 0.0)
            if prev_score < THRESHOLD and cur_score >= THRESHOLD:
                # Rising-edge: enter at THIS week's close (next session after signal)
                # Signal was on prev Friday, we enter this Friday — one period lag
                open_positions[ticker] = {
                    "entry_date":    friday,
                    "entry_score":   cur_score,
                    "cumulative_abn": -tc_half,  # entry TC applied immediately
                }

        # ── Portfolio return this week ────────────────────────────────────────
        n_pos = len(open_positions)
        if n_pos > 0:
            abn_rets = []
            for ticker in open_positions:
                if ticker in ret_aligned.columns and not np.isnan(ret_aligned[ticker].loc[friday]):
                    pos_ret = float(ret_aligned[ticker].loc[friday])
                else:
                    pos_ret = 0.0
                basket = float(peer_basket_returns(ret_aligned, exclude=ticker).loc[friday])
                abn_rets.append(pos_ret - basket)
            port_abn = float(np.mean(abn_rets))
        else:
            port_abn = 0.0

        # Compute basket return vs equal-weight of all tickers (for reporting)
        all_ret = ret_aligned.mean(axis=1).loc[friday]
        basket_all = float(all_ret) if not np.isnan(all_ret) else 0.0

        equity *= (1 + port_abn)
        equity_rows.append({
            "date":             friday,
            "n_positions":      n_pos,
            "portfolio_return": round(port_abn, 6),
            "basket_return":    round(basket_all, 6),
            "abnormal_return":  round(port_abn, 6),
            "equity":           round(equity, 6),
        })

        # Update prev_distress
        prev_distress = curr_dist

    equity_df = pd.DataFrame(equity_rows)
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    # ── Metrics ───────────────────────────────────────────────────────────────
    abn = equity_df["abnormal_return"]
    active = abn[equity_df["n_positions"] > 0]

    sharpe = float(
        active.mean() / active.std() * np.sqrt(52)
    ) if len(active) > 1 and active.std() > 0 else np.nan

    eq_vals = equity_df["equity"].values
    peak    = np.maximum.accumulate(eq_vals)
    dd      = (eq_vals - peak) / peak
    max_dd  = float(dd.min())

    total_trades = len(trade_returns)
    win_rate = float(np.mean([r > 0 for r in trade_returns])) if trade_returns else np.nan

    # CAGR: from first date to last
    first_date = equity_df["date"].iloc[0]
    last_date  = equity_df["date"].iloc[-1]
    years = (last_date - first_date).days / 365.25
    cagr = float(equity_df["equity"].iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan

    metrics = {
        "sharpe":          round(sharpe, 3) if not np.isnan(sharpe) else None,
        "max_drawdown":    round(max_dd * 100, 2),
        "win_rate":        round(win_rate * 100, 1) if not np.isnan(win_rate) else None,
        "total_trades":    total_trades,
        "cagr":            round(cagr * 100, 2) if not np.isnan(cagr) else None,
        "tc_bps":          TC_RT_BPS,
        "annotation":      (
            f"ABNORMAL returns vs equal-weighted peer basket. "
            f"TC={TC_RT_BPS}bps round-trip (placeholder). "
            f"Threshold={THRESHOLD}, hold={HOLD_DAYS}d, half-threshold exit={HALF_THRESHOLD}. "
            f"No parameter tuning — fixed as specified."
        ),
    }

    return equity_df, metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load or build distress timeseries ─────────────────────────────────
    ts_path = RESULTS_DIR / "distress_timeseries.csv"
    if ts_path.exists():
        log.info("Loading distress timeseries from %s", ts_path)
        ts = pd.read_csv(ts_path, parse_dates=["date"])
    else:
        log.info("distress_timeseries.csv not found — building now …")
        from data_pipelines.distress_timeseries import run_timeseries
        ts = run_timeseries()

    log.info("Timeseries loaded: %d rows, %s → %s",
             len(ts), ts["date"].min().date(), ts["date"].max().date())

    # ── 2. Pull price data ────────────────────────────────────────────────────
    daily_prices = pull_price_data()
    if daily_prices.empty:
        log.error("No price data retrieved — cannot run backtest.")
        return

    # ── 3. Event study ────────────────────────────────────────────────────────
    per_event, aggregate = run_event_study(ts, daily_prices)

    per_event.to_csv(RESULTS_DIR / "event_study_results.csv", index=False)
    log.info("Wrote event study → %s", RESULTS_DIR / "event_study_results.csv")

    # ── 4. Threshold backtest ─────────────────────────────────────────────────
    equity_df, metrics = run_threshold_backtest(ts, daily_prices)

    equity_df.to_csv(RESULTS_DIR / "layer1_equity_curve.csv", index=False)
    with open(RESULTS_DIR / "layer1_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    log.info("Wrote equity curve + metrics → %s", RESULTS_DIR)

    # ── 5. Print results ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  EVENT STUDY — PER EVENT")
    print(f"{'='*70}")
    if not per_event.empty:
        cols = ["event_date", "ticker", "source", "trigger",
                "entry_date", "n_days", "car_21", "car_63", "car_126"]
        cols = [c for c in cols if c in per_event.columns]
        print(per_event[cols].to_string(index=False))
    else:
        print("  (no events with sufficient price data)")

    print(f"\n{'='*70}")
    print("  EVENT STUDY — AGGREGATE STATS")
    print(f"{'='*70}")
    if not aggregate.empty:
        print(aggregate.to_string(index=False))
    else:
        print("  (no aggregate data)")

    print(f"\n{'='*70}")
    print("  LAYER 1 METRICS")
    print(f"{'='*70}")
    for k, v in metrics.items():
        if k != "annotation":
            unit = "%" if k in ("max_drawdown", "win_rate", "cagr") else ""
            print(f"  {k:<18} {v}{unit}")
    print(f"\n  Note: {metrics.get('annotation', '')}")

    # Print dormant exclusions if they exist
    excl_path = RESULTS_DIR / "dormant_shell_exclusions.csv"
    if excl_path.exists():
        excl = pd.read_csv(excl_path)
        if not excl.empty:
            print(f"\n{'='*70}")
            print(f"  DORMANT SHELL EXCLUSIONS ({len(excl)} companies)")
            print(f"{'='*70}")
            for _, row in excl.iterrows():
                print(f"  {row['ch_number']:<12}  {row['name'][:40]:<41}  [{row['tickers']}]")


if __name__ == "__main__":
    main()
