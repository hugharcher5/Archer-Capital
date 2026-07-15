"""
Point-in-time RawData constructor for S22's DCF signal.

Reuses the DCF engine's existing PURE computation chain completely unmodified:
    raw = build_pit_rawdata(...)          # <- the only new piece
    drvrs = compute_drivers(raw)          # existing, unmodified
    wacc_r = compute_wacc(raw, drvrs, as_of=t)  # existing + additive as_of param
    base = _build_base(raw, drvrs, wacc_r)      # existing, unmodified
    dcf_value = dcf.value(base)                 # existing, unmodified

Deterministic margin-of-safety is used as the ranking signal (not the full
Monte Carlo probability-of-undervaluation) -- confirmed with the user as the
computationally feasible choice: one clean calc per (ticker, quarter), no
10,000-path simulation loop.
"""
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pit_facts as PF

_DCF_DIR = Path(__file__).resolve().parents[2] / "dcf"
sys.path.insert(0, str(_DCF_DIR))
from valuation.data import RawData          # noqa: E402
from valuation.drivers import compute_drivers  # noqa: E402
from valuation.wacc import compute_wacc     # noqa: E402
from valuation.montecarlo import _build_base  # noqa: E402
from valuation import dcf as dcf_module     # noqa: E402


def build_pit_rawdata(ticker: str, cik: int, as_of: pd.Timestamp,
                       price: float, shares: float, beta: float) -> RawData | None:
    as_of_str = as_of.strftime("%Y-%m-%d")
    facts = PF.get_facts(cik)
    if facts is None:
        return None

    revenue = PF.annual_series(facts, PF.REVENUE_CONCEPTS, as_of_str)
    ebit = PF.annual_series(facts, PF.EBIT_CONCEPTS, as_of_str)
    interest_expense = PF.annual_series(facts, PF.INTEREST_EXPENSE_CONCEPTS, as_of_str, abs_val=True)
    pretax_income = PF.annual_series(facts, PF.PRETAX_CONCEPTS, as_of_str)
    tax_provision = PF.annual_series(facts, PF.TAX_CONCEPTS, as_of_str, abs_val=True)
    sbc = PF.annual_series(facts, PF.SBC_CONCEPTS, as_of_str, abs_val=True)
    da = PF.da_series(facts, as_of_str)
    capex = PF.annual_series(facts, PF.CAPEX_CONCEPTS, as_of_str, abs_val=True)

    if revenue.empty or ebit.empty or len(revenue) < 2 or len(ebit) < 2:
        return None   # compute_drivers() requires >=2 years of revenue+ebit history

    cash = PF.scalar_pit(facts, PF.CASH_CONCEPTS, as_of_str)
    total_debt = PF.total_debt_pit(facts, as_of_str)
    current_assets = PF.scalar_pit(facts, PF.CURRENT_ASSETS_CONCEPTS, as_of_str)
    current_liabilities = PF.scalar_pit(facts, PF.CURRENT_LIAB_CONCEPTS, as_of_str)
    current_debt = PF.scalar_pit(facts, PF.STD_CONCEPTS, as_of_str)

    # Balance-sheet scalars: 0.0 fallback when a smaller filer doesn't tag the
    # concept at all (matches fetch_edgar()'s missing-field tolerance -- a
    # missing current_debt/current_liabilities is far more common than a
    # genuinely-zero one for small caps, but treating as 0 is the same
    # convention this codebase already uses rather than dropping the name).
    cash = 0.0 if np.isnan(cash) else cash
    total_debt = 0.0 if np.isnan(total_debt) else total_debt
    current_assets = 0.0 if np.isnan(current_assets) else current_assets
    current_liabilities = 0.0 if np.isnan(current_liabilities) else current_liabilities
    current_debt = 0.0 if np.isnan(current_debt) else current_debt

    if np.isnan(price) or price <= 0 or np.isnan(shares) or shares <= 0:
        return None
    market_cap = price * shares

    return RawData(
        ticker=ticker, currency="USD",
        current_price_usd=float(price),
        market_cap_usd=float(market_cap),
        market_cap_local=float(market_cap),
        fx_rate=1.0,
        beta=float(beta) if not np.isnan(beta) else 1.0,
        diluted_shares=float(shares),
        revenue=revenue, ebit=ebit,
        interest_expense=interest_expense,
        pretax_income=pretax_income,
        tax_provision=tax_provision,
        sbc=sbc, da=da, capex=capex,
        cash=float(cash), total_debt=float(total_debt),
        current_assets=float(current_assets),
        current_liabilities=float(current_liabilities),
        current_debt=float(current_debt),
        op_lease_liab=0.0, price_vol=0.0,
        missing_fields=[],
    )


def pit_dcf_margin_of_safety(ticker: str, cik: int, as_of: pd.Timestamp,
                              price: float, shares: float, beta: float) -> float | None:
    """Returns (dcf_value_per_share - price) / price, or None if infeasible or
    if the DCF's own applicability gate (reused verbatim from montecarlo.py's
    run_valuation -- terminal FCFF > 0 and base-case intrinsic > 0) fails."""
    raw = build_pit_rawdata(ticker, cik, as_of, price, shares, beta)
    if raw is None:
        return None
    try:
        # compute_drivers/compute_wacc print verbose diagnostic tables (their
        # existing, unmodified behavior) -- suppressed here at the call site
        # only, since this runs thousands of times across the backtest.
        with contextlib.redirect_stdout(io.StringIO()):
            drvrs = compute_drivers(raw)
            wacc_r = compute_wacc(raw, drvrs, as_of=as_of)
            base = _build_base(raw, drvrs, wacc_r)
            dcf_r = dcf_module.detailed_value(base)
    except Exception:
        return None

    terminal_fcff = float(dcf_r.forecast["FCFF"].iloc[-1])
    dcf_value = dcf_r.value_per_share_usd
    dcf_applicable = terminal_fcff > 0 and dcf_value > 0
    if not dcf_applicable or not np.isfinite(dcf_value):
        return None
    return (dcf_value - price) / price
