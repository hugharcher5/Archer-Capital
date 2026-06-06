import time
import yaml
import yfinance as yf

CONFIG_PATH = "config/positions.yaml"
MAX_RETRIES = 3
RETRY_DELAY = 2


def _load_positions(path=CONFIG_PATH):
    with open(path) as f:
        return yaml.safe_load(f)["positions"]


def _fetch_price(ticker):
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).fast_info
            last = info.last_price
            prev = info.previous_close
            if last and prev:
                return last, prev
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    return None, None


def fetch_portfolio(config_path=CONFIG_PATH):
    """Return (rows, totals) for all positions. Skips tickers that fail to fetch."""
    positions = _load_positions(config_path)
    rows = []

    for pos in positions:
        ticker    = pos["ticker"]
        shares    = pos["shares"]
        avg_price = pos["avg_price"]

        last, prev = _fetch_price(ticker)
        if last is None:
            continue

        rows.append({
            "ticker":       ticker,
            "shares":       shares,
            "avg_price":    avg_price,
            "last":         last,
            "daily_pct":    (last - prev) / prev * 100,
            "market_value": shares * last,
            "cost_basis":   shares * avg_price,
            "unreal_pnl":   shares * last - shares * avg_price,
            "day_pnl":      shares * (last - prev),
        })

    total_mv = sum(r["market_value"] for r in rows)
    for r in rows:
        r["weight"] = r["market_value"] / total_mv * 100

    totals = {
        "market_value": total_mv,
        "cost_basis":   sum(r["cost_basis"]  for r in rows),
        "unreal_pnl":   sum(r["unreal_pnl"]  for r in rows),
        "day_pnl":      sum(r["day_pnl"]     for r in rows),
    }

    return rows, totals
