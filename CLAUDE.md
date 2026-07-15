# Archer Capital

A personal quantitative research and DCF valuation tool built in Python.

## Repository Structure

```
Archer-Capital/
├── dcf/                  — DCF valuation engine + Streamlit UI (complete)
│   ├── valuation/        — core DCF, Monte Carlo, WACC, data ingestion
│   ├── portfolio/        — dashboard (terminal) and web UI (Streamlit)
│   ├── config/           — positions.yaml
│   ├── output/           — saved Monte Carlo charts
│   ├── tests/            — unit and consistency tests
│   ├── run_dcf.py        — CLI: deterministic DCF for a ticker
│   ├── run_mc.py         — CLI: Monte Carlo DCF for a ticker
│   └── run_reconcile.py  — CLI: multi-source data reconciliation
├── data_pipelines/       — (in progress) market data ingestion
├── backtester/           — (in progress) strategy backtesting
├── signals/              — (in progress) alpha signal research
├── paper_trader/         — (in progress) paper trading execution
├── requirements.txt
└── venv/
```

## Commands

Run the Streamlit valuation UI:
```bash
venv/bin/streamlit run dcf/portfolio/web.py
```

Run the terminal portfolio dashboard:
```bash
venv/bin/python dcf/portfolio/dashboard.py
```

Run a deterministic DCF:
```bash
venv/bin/python dcf/run_dcf.py MSFT
```

Run the Monte Carlo DCF:
```bash
venv/bin/python dcf/run_mc.py MSFT
```

Run tests:
```bash
venv/bin/python -m pytest dcf/tests/
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
- Streamlit + Plotly — web UI
- scipy — Monte Carlo, root-finding (brentq)
- venv — dependency isolation

## Key Conventions

- Positions are defined in `dcf/config/positions.yaml`, not hardcoded
- yfinance calls use `fast_info` wrapped in try/except with up to 3 retries and 2s backoff
- All DCF computation is pure (no I/O) — safe to call in a Monte Carlo loop
- `dcf/conftest.py` adds `dcf/` to sys.path so pytest finds `valuation/` when run from the repo root

## Research Registry (open-ended Phase 3 trials)

- `research/trial_registry.csv` is the **machine-readable ground truth** for every research trial (independent DSR count, all metrics). It is written to concurrently by multiple sessions — always re-read its current max `trial_number` immediately before logging a new trial, not just at task start.
- `research/CORRECTED_TRIAL_REGISTRY.md` is the **one canonical narrative registry** — full per-trial write-ups, the DSR accounting table, and the master results table. Every trial in the CSV should have a corresponding detail section here.
- **Before creating any new registry-style file** (anything like `*_TRIAL_REGISTRY.md`, `RESULTS_SUMMARY.md`, or a fresh per-trial `RESULTS_*.md` meant to stand alone as a registry), check whether `CORRECTED_TRIAL_REGISTRY.md` already covers it — add a section there instead of starting a new file. This project has twice fragmented its registry across multiple files (`PHASE3_TRIAL_REGISTRY.md`, `RESULTS_SUMMARY.md`, both superseded and merged back in 2026-07-12); don't create a third.
- Individual per-trial detail files under a trial's own results directory (e.g. `research/composite_s15/results/RESULTS_S15.md`) are fine as source documents that the canonical registry points to — the rule is against a second file that duplicates the *registry's own role* (cross-trial DSR accounting, master results table).
