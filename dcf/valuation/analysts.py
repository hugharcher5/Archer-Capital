"""
External benchmarks for a valuation: individual analyst price targets and an
independent DCF.  Free sources only, no LLM.

- Analyst targets: Yahoo Finance rating changes (yfinance upgrades_downgrades).
  Each row is one firm's action with its current price target; the latest
  target per firm from the last 12 months is kept.  These are 12-month price
  targets, which analysts set with a mix of DCF and multiples.
- Independent DCF: Financial Modeling Prep's free discounted-cash-flow
  endpoint (FMP_API_KEY).  One call per ticker.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
import requests

TARGET_LOOKBACK_DAYS = 365


@dataclass
class AnalystBenchmarks:
    ticker: str
    targets: pd.DataFrame = field(default_factory=pd.DataFrame)  # Firm, Target, Date, Rating
    summary: dict = field(default_factory=dict)  # Yahoo consensus: low / high / mean / median
    fmp_dcf: float | None = None
    errors: dict = field(default_factory=dict)


def analyst_targets(ticker: str) -> tuple[pd.DataFrame, dict]:
    """Latest price target per firm over the lookback window, plus Yahoo's consensus."""
    import yfinance as yf

    t = yf.Ticker(ticker)
    summary = dict(t.analyst_price_targets or {})
    ud = t.upgrades_downgrades
    if ud is None or ud.empty or "currentPriceTarget" not in ud.columns:
        return pd.DataFrame(columns=["Firm", "Target", "Date", "Rating"]), summary

    ud = ud[ud["currentPriceTarget"] > 0]
    idx = pd.to_datetime(ud.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    ud = ud.set_axis(idx)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=TARGET_LOOKBACK_DAYS)
    ud = ud[ud.index >= cutoff].sort_index(ascending=False)
    latest = ud.groupby("Firm").head(1)
    df = pd.DataFrame({
        "Firm":   latest["Firm"].values,
        "Target": latest["currentPriceTarget"].astype(float).values,
        "Date":   latest.index.date,
        "Rating": latest["ToGrade"].values,
    }).sort_values("Target", ascending=False, ignore_index=True)
    return df, summary


def fmp_dcf(ticker: str) -> float | None:
    """FMP's free unlevered DCF value per share, or None if unavailable."""
    key = os.getenv("FMP_API_KEY", "").strip()
    if not key:
        return None
    r = requests.get(
        "https://financialmodelingprep.com/stable/discounted-cash-flow",
        params={"symbol": ticker, "apikey": key}, timeout=20,
    )
    r.raise_for_status()
    rows = r.json()
    if isinstance(rows, list) and rows and rows[0].get("dcf") is not None:
        return float(rows[0]["dcf"])
    return None


def fetch_benchmarks(ticker: str) -> AnalystBenchmarks:
    out = AnalystBenchmarks(ticker=ticker)
    try:
        out.targets, out.summary = analyst_targets(ticker)
    except Exception as e:
        out.errors["Yahoo analyst targets"] = str(e)
    try:
        out.fmp_dcf = fmp_dcf(ticker)
    except Exception as e:
        out.errors["FMP DCF"] = str(e)
    return out
