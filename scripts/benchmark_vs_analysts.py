"""
Value 100 stocks with the live Monte Carlo DCF and compare each with analyst
price targets and FMP's free DCF.  Writes dcf/data/analyst_benchmark.csv, which
the "Analyst estimates" tab of the app reads.

    venv/bin/python scripts/benchmark_vs_analysts.py            # full run
    venv/bin/python scripts/benchmark_vs_analysts.py --no-fmp   # keep FMP values from the last CSV

Reconciles Yahoo and SEC EDGAR only.  FMP's free plan allows 250 calls a day
and covers only some symbols, so the run spends one FMP call per ticker on its
DCF rather than three on financial statements.
"""
import argparse
import contextlib
import io
import os
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO), str(REPO / "dcf")]
OUT = REPO / "dcf" / "data" / "analyst_benchmark.csv"

TICKERS = """
AAPL MSFT NVDA GOOGL AMZN META TSLA AVGO ORCL ADBE CRM AMD INTC CSCO IBM QCOM TXN
KO PEP PG WMT COST MCD NKE SBUX HD LOW TGT DIS NFLX
JNJ PFE MRK ABBV LLY UNH AMGN TMO ABT
CAT DE BA GE HON UPS LMT MMM UNP
XOM CVX COP SLB OXY
LIN NEE DUK SO DOW NEM FCX
T VZ TMUS
JPM BAC GS V MA BRK-B
ETSY ROKU PLTR SNOW UBER ABNB DOCU ZM CROX DECK POOL
ASML TSM NVO SAP TM SONY BABA SHOP UL BP
RIVN LCID
ADP INTU NOW PANW CMG ORLY AZO ISRG
""".split()


def value_one(args) -> dict:
    ticker, use_fmp = args
    from valuation.sources import fetch_yahoo, fetch_edgar
    from valuation.reconcile import reconcile
    from valuation.montecarlo import run_valuation
    from valuation.analysts import analyst_targets, fmp_dcf

    row = {"ticker": ticker}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sources = []
            for fetch in (fetch_yahoo, fetch_edgar):
                try:
                    sources.append(fetch(ticker))
                except Exception:
                    pass
            if len(sources) >= 2:
                rr = reconcile(sources)
                r = run_valuation(ticker, n_sims=3000, sigma_cross=rr.sigma_cross, reconcile_result=rr)
            else:
                r = run_valuation(ticker, n_sims=3000)
        row.update(price=r.current_price_usd)
        if r.dcf_applicable:
            row.update(status="valued", p10=r.p10, p50=r.p50, p90=r.p90,
                       note="unproven company" if r.unproven else "")
        else:
            row.update(status="not applicable", note="terminal-year free cash flow is negative")
    except Exception as e:
        row.update(status="failed", note=str(e).split(". ")[0][:160])

    for _ in range(3):   # Yahoo occasionally returns an empty frame under load
        try:
            targets, summary = analyst_targets(ticker)
        except Exception:
            targets, summary = pd.DataFrame(), {}
        if not targets.empty:
            break
    if not targets.empty:
        row.update(analyst_mean=targets["Target"].mean(), analyst_median=targets["Target"].median(),
                   analyst_sd=targets["Target"].std(ddof=1), analyst_n=len(targets))
    elif summary.get("mean"):
        # No firm-by-firm targets in the last year: fall back to Yahoo's consensus
        row.update(analyst_mean=summary["mean"], analyst_median=summary.get("median"))
    row.setdefault("price", summary.get("current"))
    if use_fmp:
        try:
            row["fmp_dcf"] = fmp_dcf(ticker)
        except Exception:
            row["fmp_dcf"] = None
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fmp", action="store_true", help="reuse FMP DCF values from the existing CSV")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(REPO / ".env")

    rows = []
    with Pool(4) as pool:
        for row in pool.imap_unordered(value_one, [(t, not args.no_fmp) for t in TICKERS]):
            rows.append(row)
            print(f"{row['ticker']:6s} {row.get('status')}", flush=True)

    df = pd.DataFrame(rows).set_index("ticker").reindex(TICKERS)
    if args.no_fmp and OUT.exists():
        df["fmp_dcf"] = pd.read_csv(OUT).set_index("ticker")["fmp_dcf"].reindex(df.index)
    df["run_date"] = date.today().isoformat()
    cols = ["price", "p10", "p50", "p90", "analyst_mean", "analyst_median", "analyst_sd",
            "analyst_n", "fmp_dcf", "status", "note", "run_date"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.reindex(columns=cols).round(2).rename_axis("ticker").to_csv(OUT)
    print(f"wrote {OUT} ({(df['status'] == 'valued').sum()} valued)")


if __name__ == "__main__":
    main()
