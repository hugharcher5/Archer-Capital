"""Fetch and structure raw financial inputs from yfinance for a single ticker."""

from __future__ import annotations
import warnings
import yfinance as yf
import pandas as pd
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")


@dataclass
class RawData:
    ticker: str
    currency: str               # reporting (financial) currency, e.g. 'USD', 'TWD'
    current_price_usd: float    # stock price in USD
    market_cap_usd: float       # market cap in USD
    market_cap_local: float     # market cap in reporting currency
    fx_rate: float              # 1 unit local → USD
    beta: float
    diluted_shares: float       # actual share count

    # Annual series — index is fiscal-year-end date, ascending (oldest first)
    revenue: pd.Series          # in reporting currency
    ebit: pd.Series
    interest_expense: pd.Series # stored as positive
    pretax_income: pd.Series
    tax_provision: pd.Series    # stored as positive
    sbc: pd.Series              # stock-based compensation, positive
    da: pd.Series               # D&A, positive
    capex: pd.Series            # capex, stored as positive

    # Balance sheet — most recent fiscal year (in reporting currency)
    cash: float
    total_debt: float
    current_assets: float
    current_liabilities: float
    current_debt: float         # current portion of debt (excluded from operating NWC)

    missing_fields: list = field(default_factory=list)


def _safe_row(df: pd.DataFrame, candidates: list[str],
              missing: list[str], label: str) -> pd.Series:
    """Return first matching row as a date-indexed Series, or empty Series."""
    for name in candidates:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors='coerce').dropna()
            if not s.empty:
                return s.sort_index()   # ascending: oldest first
    missing.append(label)
    return pd.Series(dtype=float)


def _safe_scalar(d: dict, candidates: list[str],
                 missing: list[str], label: str,
                 default: float = float('nan')) -> float:
    for k in candidates:
        v = d.get(k)
        if v is not None:
            try:
                f = float(v)
                if f == f:      # not NaN
                    return f
            except (TypeError, ValueError):
                pass
    missing.append(label)
    return default


def _bs_latest(bs: pd.DataFrame, candidates: list[str],
               missing: list[str], label: str) -> float:
    """Most recent balance-sheet value for the first matching row."""
    for name in candidates:
        if name in bs.index:
            s = pd.to_numeric(bs.loc[name], errors='coerce').dropna()
            if not s.empty:
                return float(s.iloc[0])   # columns are descending → iloc[0] = most recent
    missing.append(label)
    return 0.0


def get_fx_rate(ccy: str) -> float:
    """Return USD per 1 unit of ccy. Returns 1.0 for USD or on failure."""
    if ccy.upper() == 'USD':
        return 1.0
    try:
        rate = float(yf.Ticker(f"{ccy}USD=X").fast_info.last_price)
        if rate > 0:
            return rate
    except Exception:
        pass
    print(f"  [DATA] Warning: could not fetch FX rate for {ccy}/USD; assuming 1.0")
    return 1.0


def fetch_raw(ticker: str, years: int = 5) -> RawData:
    t    = yf.Ticker(ticker)
    info = t.info or {}
    missing: list[str] = []

    # ── Currency ──────────────────────────────────────────────────────────────
    currency = (info.get('financialCurrency') or info.get('currency') or 'USD').upper()
    fx_rate  = get_fx_rate(currency)

    # ── Price & market cap ────────────────────────────────────────────────────
    price_usd   = _safe_scalar(info, ['currentPrice', 'regularMarketPrice', 'previousClose'],
                               missing, 'current_price_usd')
    mktcap_usd  = _safe_scalar(info, ['marketCap'], missing, 'market_cap_usd')
    mktcap_local = mktcap_usd / fx_rate if fx_rate > 0 else mktcap_usd

    # ── Beta ──────────────────────────────────────────────────────────────────
    beta = _safe_scalar(info, ['beta'], [], 'beta', default=float('nan'))
    if beta != beta:
        beta = 1.0
        print(f"  [DATA] beta missing for {ticker}; defaulting to 1.0")

    # ── Diluted shares ────────────────────────────────────────────────────────
    diluted_shares = _safe_scalar(
        info, ['impliedSharesOutstanding', 'sharesOutstanding'],
        [], 'diluted_shares', default=float('nan'))
    if diluted_shares != diluted_shares:
        if price_usd > 0 and mktcap_usd > 0:
            diluted_shares = mktcap_usd / price_usd
            print(f"  [DATA] diluted_shares inferred from market_cap / price")
        else:
            diluted_shares = 0.0
            missing.append('diluted_shares')

    # ── Financial statements ──────────────────────────────────────────────────
    def _fetch(primary: str, fallback: str) -> pd.DataFrame:
        for attr in (primary, fallback):
            try:
                df = getattr(t, attr)
                if df is not None and not df.empty:
                    return df
            except Exception:
                pass
        return pd.DataFrame()

    inc = _fetch('income_stmt', 'financials')
    cf  = _fetch('cashflow', 'cash_flow')
    bs  = _fetch('balance_sheet', 'quarterly_balance_sheet')

    def _trim(df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:, :years] if not df.empty else df

    inc, cf, bs = _trim(inc), _trim(cf), _trim(bs)

    # ── Income statement series ───────────────────────────────────────────────
    revenue      = _safe_row(inc, ['Total Revenue', 'Revenue'],                              missing, 'revenue')
    ebit         = _safe_row(inc, ['EBIT', 'Operating Income', 'Ebit'],                      missing, 'ebit')
    interest_exp = _safe_row(inc, ['Interest Expense', 'Net Non Operating Interest Income Expense'],
                             missing, 'interest_expense').abs()
    pretax_inc   = _safe_row(inc, ['Pretax Income', 'Income Before Tax'],                    missing, 'pretax_income')
    tax_prov     = _safe_row(inc, ['Tax Provision', 'Income Tax Expense'],                   missing, 'tax_provision').abs()

    # ── Cash flow series ──────────────────────────────────────────────────────
    da    = _safe_row(cf, ['Depreciation And Amortization',
                           'Depreciation Depletion And Amortization',
                           'Depreciation'],                                                  missing, 'da').abs()
    capex = _safe_row(cf, ['Capital Expenditure', 'Capital Expenditures',
                           'Purchase Of Property Plant And Equipment'],                      missing, 'capex').abs()
    sbc   = _safe_row(cf, ['Stock Based Compensation', 'Share Based Compensation Expense'],  [],      'sbc').abs()
    if sbc.empty:
        sbc = _safe_row(inc, ['Stock Based Compensation'],                                   missing, 'sbc').abs()

    # ── Balance sheet scalars (most recent year) ──────────────────────────────
    cash = _bs_latest(bs, ['Cash And Cash Equivalents',
                           'Cash Cash Equivalents And Short Term Investments'],              missing, 'cash')

    # Total debt: try direct field, then sum LTD + current debt
    total_debt = _bs_latest(bs, ['Total Debt'], [], 'total_debt_direct')
    if total_debt == 0:
        ltd      = _bs_latest(bs, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'], [], 'ltd')
        cur_debt = _bs_latest(bs, ['Current Debt', 'Short Term Debt',
                                   'Current Debt And Capital Lease Obligation'],             [], 'cdt')
        total_debt = ltd + cur_debt
        if total_debt == 0:
            missing.append('total_debt')

    curr_assets = _bs_latest(bs, ['Current Assets'],                                        missing, 'current_assets')
    curr_liab   = _bs_latest(bs, ['Current Liabilities'],                                   missing, 'current_liabilities')
    curr_debt   = _bs_latest(bs, ['Current Debt', 'Short Term Debt',
                                  'Current Debt And Capital Lease Obligation'],              [],      'current_debt')

    # ── Summary print ─────────────────────────────────────────────────────────
    uniq_missing = sorted(set(missing))
    print(f"\n{'='*60}")
    print(f"  RAW DATA — {ticker}")
    print(f"{'='*60}")
    if uniq_missing:
        print(f"  Missing fields: {', '.join(uniq_missing)}")
    else:
        print(f"  All fields fetched successfully.")
    print(f"\n  Currency (reporting)  : {currency}  (1 {currency} = {fx_rate:.6f} USD)")
    print(f"  Current price (USD)   : ${price_usd:,.2f}")
    print(f"  Market cap (USD)      : ${mktcap_usd/1e9:.2f}B")
    print(f"  Market cap ({currency:3s})      : {mktcap_local/1e9:.2f}B {currency}")
    print(f"  Beta                  : {beta:.3f}")
    print(f"  Diluted shares        : {diluted_shares/1e9:.3f}B")
    print(f"  Annual revenue years  : {len(revenue)}")
    print(f"  Cash                  : {cash/1e9:.3f}B {currency}")
    print(f"  Total debt            : {total_debt/1e9:.3f}B {currency}")
    nwc = (curr_assets - cash) - (curr_liab - curr_debt)
    print(f"  Operating NWC         : {nwc/1e9:.3f}B {currency}  [(CA−Cash)−(CL−CurrDebt)]")

    return RawData(
        ticker=ticker,
        currency=currency,
        current_price_usd=price_usd,
        market_cap_usd=mktcap_usd,
        market_cap_local=mktcap_local,
        fx_rate=fx_rate,
        beta=beta,
        diluted_shares=diluted_shares,
        revenue=revenue,
        ebit=ebit,
        interest_expense=interest_exp,
        pretax_income=pretax_inc,
        tax_provision=tax_prov,
        sbc=sbc,
        da=da,
        capex=capex,
        cash=cash,
        total_debt=total_debt,
        current_assets=curr_assets,
        current_liabilities=curr_liab,
        current_debt=curr_debt,
        missing_fields=uniq_missing,
    )
