"""
Monte Carlo DCF runner.

Usage:
    venv/bin/python run_mc.py MSFT
    venv/bin/python run_mc.py ZS 20000
"""

import sys
from valuation.montecarlo import run_monte_carlo

if __name__ == '__main__':
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'MSFT'
    n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000
    run_monte_carlo(ticker, n_sims)
