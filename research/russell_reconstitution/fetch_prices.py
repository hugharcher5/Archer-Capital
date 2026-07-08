"""
Phase 3: Fetch Tiingo daily prices (adjClose primary, raw close fallback)
for every ticker in the Russell-1 addition universe, plus IWM (iShares
Russell 2000 ETF) as the small-cap benchmark for each cycle's window.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache" / "tiingo"
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS = BASE / "results"


def _load_dotenv() -> None:
    env_path = BASE.parents[1] / ".env"
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

_LAST_CALL = 0.0


def _tiingo_get(url, params, retries=2):
    global _LAST_CALL
    for attempt in range(retries):
        wait = 1.3 - (time.time() - _LAST_CALL)
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, params=params, timeout=(10, 20))
        except Exception as e:
            print(f"  network error: {e}")
            _LAST_CALL = time.time()
            continue
        _LAST_CALL = time.time()
        if r.status_code == 429:
            print("  429 rate limit -- sleeping 60s")
            time.sleep(60)
            continue
        return r
    return None


def fetch_one(ticker: str) -> pd.DataFrame:
    cache_file = CACHE / f"{ticker}.csv"
    sentinel = CACHE / f"{ticker}.EMPTY"
    if sentinel.exists():
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])
    if cache_file.exists():
        return pd.read_csv(cache_file, parse_dates=["date"])

    url = f"https://api.tiingo.com/tiingo/daily/{ticker}/prices"
    params = {
        "startDate": "2015-01-01",
        "endDate": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "token": TIINGO_KEY,
        "columns": "date,close,adjClose,volume",
    }
    r = _tiingo_get(url, params)
    if r is None or r.status_code in (404, 400):
        sentinel.touch()
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])
    if r.status_code != 200:
        print(f"  HTTP {r.status_code} for {ticker}")
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])
    try:
        data = r.json()
    except Exception:
        sentinel.touch()
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])
    if not data:
        sentinel.touch()
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])

    df = pd.DataFrame(data)
    df.columns = [c.lower() if c.lower() != "adjclose" else "adjClose" for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    keep = [c for c in ["date", "close", "adjClose", "volume"] if c in df.columns]
    df = df[keep].drop_duplicates("date").sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    return df


def main():
    universe = pd.read_csv(RESULTS / "russell1_universe.csv")
    tickers = sorted(universe["ticker"].unique())
    print(f"Fetching {len(tickers)} unique tickers + IWM benchmark ...")

    n_ok, n_empty = 0, 0
    for i, t in enumerate(tickers + ["IWM"]):
        df = fetch_one(t)
        if df.empty:
            n_empty += 1
        else:
            n_ok += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)+1}  ok={n_ok} empty={n_empty}")

    print(f"\nDone. ok={n_ok} empty={n_empty} out of {len(tickers)+1}")


if __name__ == "__main__":
    main()
