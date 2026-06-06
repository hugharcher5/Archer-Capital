# Archer Capital

A personal portfolio tracking tool built in Python.

## Project Overview

Archer Capital is a stock portfolio dashboard. It reads positions from a config file, pulls live prices from yfinance, and prints a formatted table with P&L, daily moves, and portfolio weights.

## Phase 1 — Portfolio Dashboard (complete)

- `config/positions.yaml` — position definitions (ticker, shares, avg_price)
- `portfolio/dashboard.py` — main script: loads positions, fetches live prices with retry logic, prints a Rich table with last price, daily %, market value, cost basis, unrealized P&L, daily P&L, and portfolio weight per position plus totals

## Commands

Run the dashboard:
```bash
venv/bin/python portfolio/dashboard.py
```

Install dependencies (first time or after changes to requirements.txt):
```bash
venv/bin/pip install -r requirements.txt
```

## Tech Stack

- Python 3.12
- yfinance — live price data
- PyYAML — position config
- Rich — terminal table rendering
- venv — dependency isolation

## Key Conventions

- Positions are defined in `config/positions.yaml`, not hardcoded
- yfinance calls use `fast_info` (lighter than `info`) wrapped in try/except with up to 3 retries and 2s backoff
- Console width is fixed at 120 with `no_wrap=True` on all columns to prevent truncation
