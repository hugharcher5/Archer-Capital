"""
research/monte_carlo_dispersion/run.py

Hypothesis 3: Does the Monte Carlo distribution carry information?

Within DCF-cheap names (top-quintile DCF V/P), does tight intrinsic-value
dispersion (low CV = std/mean of simulated IV) outperform wide dispersion
(high CV)? Does this survive controls for known risk factors — realized
volatility and leverage?

CV is computed fresh via 2,000-path Monte Carlo DCF for every DCF-applicable
(ticker, as_of_date), using the same EDGAR facts and PERT mechanics as the
production engine. Results are cached to _cache/mc_stats.parquet.

T = 10 annual cross-sections (2015-06-30 → 2024-06-30). Small sample —
flag all results for Deflated Sharpe. Read direction and magnitude, not
significance alone.

Tests:
  1. Horse race: Q1–Q5 by CV within DCF-cheap. Q1=tight, Q5=wide.
     Headline: Q1-minus-Q5 differential.
  2. Fama-MacBeth: CV coefficient WITH all controls [DCF V/P, CV, vol,
     leverage, SIZE, MOM]. Gate: is CV positive and significant after
     vol and leverage are in?
  3. Coefficient delta: CV in [CV, SIZE, MOM] vs [CV, SIZE, MOM, vol, lev].
     Delta = how much of CV effect is explained by risk proxies.
  4. EBITDA/EV robustness: same tests using EBITDA/EV-cheap instead of
     DCF-cheap. Should be null — dispersion of DCF IV should not predict
     returns inside non-DCF-selected baskets.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "dcf"))
from dotenv import load_dotenv
load_dotenv(REPO / ".env")
from valuation.dcf import (
    Assumptions, value as dcf_value,
    TERMINAL_G, HIGH_GROWTH_THRESHOLD, MATURE_MARGIN_DEFAULT,
    G_CEILING_A, G_CEILING_B,
)

# ── Study parameters ───────────────────────────────────────────────────────────
START_YEAR    = 2015
END_YEAR      = 2025
REB_MONTH     = 6
REB_DAY       = 30
N_QUINTILES   = 5
COSTS_BPS     = 20          # round-trip
ERP           = 0.045       # Damodaran implied ERP
BETA_LOOKBACK = 504         # ~2 years of trading days
VOL_WINDOW    = 252         # 252-day realized volatility
MOM_WINDOW    = 252         # 12-month momentum
MOM_SKIP      = 21          # skip last month

# MC parameters (mirrors montecarlo.py)
N_PATHS                  = 2_000   # paths per (ticker, date)
N_WORKERS                = 12      # parallel worker threads
SPREAD_SIGMA             = 3.0
WACC_SPREAD_ABS          = 0.015
TERM_G_FALLBACK          = 0.020   # macro GDP vol fallback (from montecarlo.py)
TERM_G_MIN               = 0.015
TERM_G_MAX_ABS           = 0.040
WACC_TG_GAP              = 0.010
TARGET_MARGIN_SPREAD_MIN = 0.03
TARGET_MARGIN_SPREAD_MAX = 0.20
MC_SEED                  = 42

# Correlation matrix for 5 MC inputs: [rev_g, ebit_margin, terminal_g, wacc, target_margin]
_CORR = np.array([
    [1.00,  0.40,  0.30,  0.20,  0.30],
    [0.40,  1.00,  0.10, -0.10,  0.60],
    [0.30,  0.10,  1.00,  0.10,  0.10],
    [0.20, -0.10,  0.10,  1.00, -0.15],
    [0.30,  0.60,  0.10, -0.15,  1.00],
], dtype=float)

_SYNTH_TABLE: list[tuple[float, float, float]] = [
    (12.50, float("inf"), 0.0063),
    ( 9.50,       12.50,  0.0078),
    ( 7.50,        9.50,  0.0098),
    ( 6.00,        7.50,  0.0113),
    ( 4.50,        6.00,  0.0128),
    ( 4.00,        4.50,  0.0163),
    ( 3.50,        4.00,  0.0238),
    ( 3.00,        3.50,  0.0313),
    ( 2.50,        3.00,  0.0388),
    ( 2.00,        2.50,  0.0463),
    ( 1.50,        2.00,  0.0563),
    ( 1.25,        1.50,  0.0713),
    ( 0.80,        1.25,  0.0863),
    ( 0.50,        0.80,  0.1113),
    (float("-inf"), 0.50, 0.1413),
]

_EDGAR_HDR = {"User-Agent": "Archer Capital research@archercapital.dev"}

CACHE_DIR   = Path(__file__).parent / "_cache"
DCF_SIGNALS = REPO / "research" / "dcf_vs_multiple" / "_cache" / "signals_panel.parquet"
EDGAR_CACHE = REPO / "research" / "dcf_vs_multiple" / "_cache" / "edgar_facts"
MC_CACHE    = CACHE_DIR / "mc_stats.parquet"
PRICE_CACHE = CACHE_DIR / "price_data.pkl"

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# EDGAR helpers (reuse dcf_vs_multiple cache; no new SEC calls)
# ═══════════════════════════════════════════════════════════════════════════════

def load_edgar_facts(ticker: str) -> dict | None:
    path = EDGAR_CACHE / f"{ticker.upper()}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _pit_series(facts: dict, tags: list[str], as_of: date, abs_val: bool = False) -> pd.Series:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node  = us_gaap.get(tag, {})
        units = node.get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        best: dict[pd.Timestamp, tuple[date, float]] = {}
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            fs = row.get("filed")
            if not fs:
                continue
            try:
                filed = date.fromisoformat(fs)
            except ValueError:
                continue
            if filed > as_of:
                continue
            end_str = row.get("end")
            val     = row.get("val")
            if end_str is None or val is None:
                continue
            try:
                v  = abs(float(val)) if abs_val else float(val)
                ts = pd.Timestamp(end_str)
                if ts not in best or filed > best[ts][0]:
                    best[ts] = (filed, v)
            except (ValueError, TypeError):
                pass
        if best:
            return pd.Series({ts: v for ts, (_, v) in best.items()}).sort_index()
    return pd.Series(dtype=float)


def _pit_scalar(facts: dict, tags: list[str], as_of: date, abs_val: bool = False) -> float:
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node  = us_gaap.get(tag, {})
        units = node.get("units", {})
        rows  = units.get("USD") or units.get("shares") or []
        best_filed: date | None = None
        best_val:   float       = float("nan")
        for row in rows:
            if row.get("form") not in ("10-K", "10-K/A"):
                continue
            fs = row.get("filed")
            if not fs:
                continue
            try:
                filed = date.fromisoformat(fs)
            except ValueError:
                continue
            if filed > as_of:
                continue
            val = row.get("val")
            if val is None:
                continue
            try:
                v = abs(float(val)) if abs_val else float(val)
                if best_filed is None or filed > best_filed:
                    best_filed = filed
                    best_val   = v
            except (ValueError, TypeError):
                pass
        if not math.isnan(best_val):
            return best_val
    return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# PERT + copula utilities (inlined from montecarlo.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _pert_ppf(u: np.ndarray, lo: float, mode: float, hi: float) -> np.ndarray:
    if abs(hi - lo) < 1e-12:
        return np.full(len(u), mode, dtype=float)
    mode = min(max(mode, lo), hi)
    r    = hi - lo
    a    = 1.0 + 4.0 * (mode - lo) / r
    b    = 1.0 + 4.0 * (hi - mode) / r
    return lo + r * scipy.stats.beta.ppf(np.clip(u, 1e-9, 1.0 - 1e-9), a, b)


def _synth_spread(coverage: float) -> tuple[str, float]:
    _LABELS = {
        0.0063: "Aaa/AAA", 0.0078: "Aa2/AA", 0.0098: "A1/A+",  0.0113: "A2/A",
        0.0128: "A3/A-",   0.0163: "Baa2/BBB",0.0238: "Ba1/BB+",0.0313: "Ba2/BB",
        0.0388: "B1/B+",   0.0463: "B2/B",   0.0563: "B3/B-",  0.0713: "Caa/CCC",
        0.0863: "Ca2/CC",  0.1113: "C2/C",   0.1413: "D2/D",
    }
    for lo, hi, sp in _SYNTH_TABLE:
        if lo <= coverage < hi:
            return _LABELS.get(sp, "?"), sp
    return "D2/D", 0.1413


def _ensure_psd(C: np.ndarray) -> np.ndarray:
    C = (C + C.T) / 2.0
    eigvals = np.linalg.eigvalsh(C)
    if np.all(eigvals >= -1e-10):
        return C
    lam, V = np.linalg.eigh(C)
    C2 = V @ np.diag(np.maximum(lam, 1e-8)) @ V.T
    d  = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorised batch DCF — replaces per-path Python loop
# ═══════════════════════════════════════════════════════════════════════════════

def _dcf_batch(
    a:    Assumptions,
    rg_s: np.ndarray,   # (N,) revenue growth per path
    em_s: np.ndarray,   # (N,) EBIT margin per path
    tm_s: np.ndarray,   # (N,) target margin per path
    tg_s: np.ndarray,   # (N,) terminal growth per path
    wa_s: np.ndarray,   # (N,) WACC per path
) -> np.ndarray:
    """
    Fully vectorised DCF over N paths (USD companies, fx_rate=1).
    Only rev_growth, ebit_margin, target_margin, terminal_g, wacc vary by path.
    All scalar parameters (da_pct, capex_pct, nwc_pct, tax_rate, net_debt,
    diluted_shares) come from the base-case Assumptions object.

    Mirrors _compute() in dcf.py, including size-dependent growth ceiling,
    without per-path Python loops or pandas DataFrames.
    Returns (N,) array of intrinsic values per share (USD).
    """
    N = len(rg_s)
    T = a.forecast_years

    idx  = np.arange(T, dtype=float)
    frac = idx / max(T - 1, 1)        # linear fade positions 0 → 1

    # Revenue growth and margin schedules: (N, T)
    g_sched  = rg_s[:, None] + (tg_s - rg_s)[:, None] * frac[None, :]
    em_sched = em_s[:, None] + (tm_s - em_s)[:, None] * frac[None, :]

    # CapEx schedule: fade to da_pct when capex > da; else constant
    if a.capex_pct > a.da_pct:
        cp_fade = a.capex_pct + (a.da_pct - a.capex_pct) * frac   # (T,)
    else:
        cp_fade = np.full(T, a.capex_pct)
    cp_sched = np.broadcast_to(cp_fade[None, :], (N, T)).copy()   # (N, T)

    # Sequential revenue path — needed for size ceiling (each year depends on prev)
    rev   = np.empty((N, T))
    fcff  = np.empty((N, T))
    disc  = np.empty((N, T))
    prev_rev = np.full(N, a.start_revenue)
    prev_nwc = np.full(N, a.start_revenue * a.nwc_pct)

    for t in range(T):
        r_usd_b = prev_rev / 1e9
        g_ceil  = np.maximum(tg_s, G_CEILING_A / np.maximum(r_usd_b, 1e-9) ** G_CEILING_B)
        g_eff   = np.minimum(g_sched[:, t], g_ceil)

        curr_rev = prev_rev * (1.0 + g_eff)
        rev[:, t] = curr_rev

        ebit  = curr_rev * em_sched[:, t]
        nopat = ebit * (1.0 - a.tax_rate)
        da    = curr_rev * a.da_pct
        capex = curr_rev * cp_sched[:, t]
        nwc   = curr_rev * a.nwc_pct
        dnwc  = nwc - prev_nwc
        fcff[:, t] = nopat + da - capex - dnwc
        disc[:, t] = 1.0 / (1.0 + wa_s) ** (t + 1)

        prev_rev = curr_rev
        prev_nwc = nwc

    pv_explicit = (fcff * disc).sum(axis=1)               # (N,)

    # Terminal value via Gordon Growth
    last_fcff   = fcff[:, -1]
    gap         = wa_s - tg_s
    tv          = np.where(gap > 1e-4, last_fcff * (1.0 + tg_s) / gap, np.nan)
    pv_tv       = tv / (1.0 + wa_s) ** T

    equity = pv_explicit + pv_tv - a.net_debt
    if a.diluted_shares <= 0:
        return np.full(N, np.nan)
    return equity / a.diluted_shares   # USD, fx_rate=1


# ═══════════════════════════════════════════════════════════════════════════════
# Point-in-time Monte Carlo
# ═══════════════════════════════════════════════════════════════════════════════

def pit_mc(
    ticker:  str,
    facts:   dict,
    as_of:   date,
    price:   float,
    mktcap:  float,
    beta_raw: float,
    rf:       float,
    n_paths: int = N_PATHS,
) -> tuple[float, float, float] | None:
    """
    Point-in-time Monte Carlo DCF.
    Returns (mean_iv, std_iv, cv) where cv = std/mean, or None if not applicable.
    Uses vectorised _dcf_batch for speed (~2 ms per call at N=2000).
    """
    # ── Fundamental data ──────────────────────────────────────────────────────
    revenue = _pit_series(facts, [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues", "SalesRevenueNet",
    ], as_of)
    ebit = _pit_series(facts, ["OperatingIncomeLoss"], as_of)
    da   = _pit_series(facts, [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
    ], as_of, abs_val=True)
    capex  = _pit_series(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"],
                          as_of, abs_val=True)
    taxexp = _pit_series(facts, ["IncomeTaxExpenseBenefit"], as_of)
    pretax = _pit_series(facts, [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"
    ], as_of)
    intexp = _pit_series(facts, [
        "InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense",
    ], as_of, abs_val=True)

    cash      = _pit_scalar(facts, ["CashAndCashEquivalentsAtCarryingValue"], as_of)
    ltd       = _pit_scalar(facts, ["LongTermDebtNoncurrent", "LongTermDebt"], as_of)
    cur_debt  = _pit_scalar(facts, ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"], as_of)
    dil_sh    = _pit_scalar(facts, ["WeightedAverageNumberOfDilutedSharesOutstanding"], as_of)
    curr_a    = _pit_scalar(facts, ["AssetsCurrent"], as_of)
    curr_l    = _pit_scalar(facts, ["LiabilitiesCurrent"], as_of)

    if revenue.empty or ebit.empty or len(revenue) < 2:
        return None
    if math.isnan(dil_sh) or dil_sh <= 0:
        return None

    cash       = 0.0 if math.isnan(cash) else cash
    total_debt = (0.0 if math.isnan(ltd) else ltd) + (0.0 if math.isnan(cur_debt) else cur_debt)

    df_s = pd.concat({"revenue": revenue, "ebit": ebit}, axis=1).dropna()
    if len(df_s) < 2:
        return None

    rev_last    = float(df_s["revenue"].iloc[-1])
    rev_g_ser   = df_s["revenue"].pct_change().dropna()
    margin_ser  = df_s["ebit"] / df_s["revenue"]

    rev_growth  = float(np.clip(rev_g_ser.mean(), -0.50, 0.80))
    ebit_margin = float(np.clip(margin_ser.mean(), -0.30, 0.60))
    best_margin = float(margin_ser.max())
    target_marg = max(best_margin, MATURE_MARGIN_DEFAULT)

    sg  = max(float(rev_g_ser.std())   if len(rev_g_ser)  >= 2 else 0.10, 0.03)
    sm  = max(float(margin_ser.std())  if len(margin_ser) >= 2 else 0.05, 0.02)
    stm = max(sm, 0.02)

    idx_da = df_s.index.intersection(da.index)
    da_pct = float(np.clip((da[idx_da] / df_s["revenue"][idx_da]).mean(), 0.0, 0.30)) \
             if len(idx_da) else 0.03
    idx_cx = df_s.index.intersection(capex.index)
    capex_pct = float(np.clip((capex[idx_cx] / df_s["revenue"][idx_cx]).mean(), 0.0, 0.50)) \
                if len(idx_cx) else 0.03

    tax_rate = 0.25
    if not taxexp.empty and not pretax.empty:
        idx_t = taxexp.index.intersection(pretax.index)
        pt_ = pretax[idx_t]; tx_ = taxexp[idx_t]
        mask = pt_ > 0
        if mask.any():
            tax_rate = float((tx_[mask] / pt_[mask]).clip(0.0, 0.6).mean())

    nwc_pct = 0.05
    if not math.isnan(curr_a) and not math.isnan(curr_l) and rev_last > 0:
        nwc_pct = float(np.clip(((curr_a - cash) - curr_l) / rev_last, -0.30, 0.50))

    beta_adj = 0.67 * beta_raw + 0.33
    coe      = rf + beta_adj * ERP
    ebit_last = float(df_s["ebit"].iloc[-1])
    int_last  = float(intexp.iloc[-1]) if not intexp.empty else 0.0
    coverage  = ebit_last / int_last if int_last > 0 else float("inf")
    rating, spread = _synth_spread(coverage)
    cod  = (rf + spread) * (1.0 - tax_rate)
    E, D, V = mktcap, total_debt, mktcap + total_debt
    eq_w = E / V if V > 0 else 1.0
    dt_w = D / V if V > 0 else 0.0
    wacc = eq_w * coe + dt_w * cod

    avg_g      = (rev_growth + TERMINAL_G) / 2.0
    terminal_g = float(np.clip(min(avg_g, TERMINAL_G), TERM_G_MIN, TERMINAL_G))
    if wacc <= terminal_g + 0.01:
        terminal_g = wacc - 0.01
    if wacc <= terminal_g:
        return None

    net_debt     = total_debt - cash
    forecast_yrs = 10 if rev_growth > HIGH_GROWTH_THRESHOLD else 5
    ebitda_est   = ebit_last + (float(da.iloc[-1]) if not da.empty else 0.0)

    a = Assumptions(
        ticker=ticker,              currency="USD",
        start_revenue=rev_last,     revenue_growth=rev_growth,
        ebit_margin=ebit_margin,    target_margin=target_marg,
        tax_rate=tax_rate,          da_pct=da_pct,
        capex_pct=capex_pct,        nwc_pct=nwc_pct,
        terminal_g=terminal_g,      wacc=wacc,
        net_debt=net_debt,          diluted_shares=dil_sh,
        fx_rate=1.0,                forecast_years=forecast_yrs,
        rf=rf,                      erp=ERP,
        beta_adj=beta_adj,          cost_of_equity=coe,
        cost_of_debt_pretax=rf + spread,
        cost_of_debt_aftertax=cod,
        equity_weight=eq_w,         debt_weight=dt_w,
        implied_rating=rating,
        current_price_usd=price,    current_ebitda=ebitda_est,
        market_ev_local=mktcap + net_debt,
    )

    # DCF applicability gate (mirrors pit_dcf from dcf_vs_multiple)
    try:
        det_val = dcf_value(a)
    except Exception:
        return None
    if not math.isfinite(det_val) or det_val <= 0:
        return None
    term_rev  = rev_last * (1 + (rev_growth + terminal_g) / 2) ** forecast_yrs
    term_fcff = (term_rev * target_marg * (1.0 - tax_rate)
                 + term_rev * da_pct - term_rev * min(capex_pct, da_pct))
    if term_fcff <= 0:
        return None

    # ── PERT bounds ───────────────────────────────────────────────────────────
    rg_lo = max(rev_growth  - SPREAD_SIGMA * sg,  -0.30)
    rg_hi = min(rev_growth  + SPREAD_SIGMA * sg,   1.50)
    em_lo = max(ebit_margin - SPREAD_SIGMA * sm,  -0.20)
    em_hi = min(ebit_margin + SPREAD_SIGMA * sm,   0.75)

    tg_cap   = min(TERM_G_MAX_ABS, wacc - WACC_TG_GAP, avg_g + 0.01)
    tg_floor = max(-0.02, 0.30 * avg_g)
    tg_half  = SPREAD_SIGMA * TERM_G_FALLBACK
    tg_lo    = max(terminal_g - tg_half, tg_floor)
    tg_hi    = max(tg_lo + 1e-6, min(terminal_g + tg_half, tg_cap, wacc - WACC_TG_GAP))

    wa_lo = max(wacc - SPREAD_SIGMA * WACC_SPREAD_ABS, 0.04)
    wa_hi = wacc + SPREAD_SIGMA * WACC_SPREAD_ABS

    tm_half = float(np.clip(SPREAD_SIGMA * stm, TARGET_MARGIN_SPREAD_MIN, TARGET_MARGIN_SPREAD_MAX))
    tm_lo = max(target_marg - tm_half, -0.10)
    tm_hi = min(target_marg + tm_half,  0.75)

    # ── Correlated draws (Gaussian copula) ───────────────────────────────────
    rng = np.random.default_rng(MC_SEED)
    C   = _ensure_psd(_CORR.copy())
    L   = np.linalg.cholesky(C)
    ZC  = rng.standard_normal((n_paths, 5)) @ L.T
    U   = scipy.stats.norm.cdf(ZC)

    rg_s = _pert_ppf(U[:, 0], rg_lo, rev_growth,  rg_hi)
    em_s = _pert_ppf(U[:, 1], em_lo, ebit_margin,  em_hi)
    tg_s = _pert_ppf(U[:, 2], tg_lo, terminal_g,   tg_hi)
    wa_s = _pert_ppf(U[:, 3], wa_lo, wacc,          wa_hi)
    tm_s = _pert_ppf(U[:, 4], tm_lo, target_marg,   tm_hi)
    tg_s = np.minimum(tg_s, wa_s - WACC_TG_GAP)

    # ── Vectorised DCF ────────────────────────────────────────────────────────
    vps = _dcf_batch(a, rg_s, em_s, tm_s, tg_s, wa_s)
    valid = vps[np.isfinite(vps) & (vps > 0)]
    if len(valid) < 10:
        return None

    mean_iv = float(np.mean(valid))
    std_iv  = float(np.std(valid))
    cv      = std_iv / mean_iv if mean_iv > 0 else float("nan")
    return mean_iv, std_iv, cv


# ═══════════════════════════════════════════════════════════════════════════════
# Leverage helper
# ═══════════════════════════════════════════════════════════════════════════════

def pit_leverage(facts: dict | None, as_of: date) -> float:
    if facts is None:
        return float("nan")
    total_assets = _pit_scalar(facts, ["Assets"], as_of)
    ltd     = _pit_scalar(facts, ["LongTermDebtNoncurrent", "LongTermDebt"], as_of)
    curdebt = _pit_scalar(facts, ["LongTermDebtCurrent", "ShortTermBorrowings", "DebtCurrent"], as_of)
    debt    = (0.0 if math.isnan(ltd) else ltd) + (0.0 if math.isnan(curdebt) else curdebt)
    if math.isnan(total_assets) or total_assets <= 0:
        return float("nan")
    return float(np.clip(debt / total_assets, 0.0, 1.5))


# ═══════════════════════════════════════════════════════════════════════════════
# Price / market data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _price_at(hist: pd.DataFrame | None, as_of: date) -> float | None:
    if hist is None or hist.empty:
        return None
    sub = hist[hist.index <= pd.Timestamp(as_of)]
    return float(sub["Close"].iloc[-1]) if not sub.empty else None


def _beta_at(stock_hist: pd.DataFrame | None, mkt_hist: pd.DataFrame,
             as_of: date, lookback: int = BETA_LOOKBACK) -> float:
    if stock_hist is None or stock_hist.empty:
        return 1.0
    ts    = pd.Timestamp(as_of)
    s_ret = stock_hist[stock_hist.index <= ts].tail(lookback)["Close"].pct_change().dropna()
    m_ret = mkt_hist  [mkt_hist.index   <= ts].tail(lookback)["Close"].pct_change().dropna()
    idx   = s_ret.index.intersection(m_ret.index)
    if len(idx) < 60:
        return 1.0
    slope, *_ = stats.linregress(m_ret[idx].values, s_ret[idx].values)
    return float(np.clip(slope, 0.1, 4.0))


def _rf_at(tnx_hist: pd.DataFrame | None, as_of: date) -> float:
    if tnx_hist is None or tnx_hist.empty:
        return 0.043
    p = _price_at(tnx_hist, as_of)
    return (p / 100.0) if (p and p > 0) else 0.043


def _realized_vol(hist: pd.DataFrame | None, as_of: date) -> float:
    if hist is None or hist.empty:
        return float("nan")
    sub = hist[hist.index <= pd.Timestamp(as_of)].tail(VOL_WINDOW + 1)
    if len(sub) < VOL_WINDOW // 2:
        return float("nan")
    log_ret = np.log(sub["Close"] / sub["Close"].shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(252))


def _fwd_return(hist: pd.DataFrame | None, as_of: date, next_date: date) -> float:
    if hist is None or hist.empty:
        return float("nan")
    p0 = _price_at(hist, as_of)
    p1 = _price_at(hist, next_date)
    if p0 is None or p1 is None or p0 <= 0:
        return float("nan")
    return (p1 - p0) / p0 - COSTS_BPS / 10_000.0


# ═══════════════════════════════════════════════════════════════════════════════
# Price data loading (with disk cache)
# ═══════════════════════════════════════════════════════════════════════════════

def load_price_data(
    tickers: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    import pickle
    if PRICE_CACHE.exists():
        print(f"  Loading price cache: {PRICE_CACHE}")
        with open(PRICE_CACHE, "rb") as f:
            return pickle.load(f)

    import yfinance as yf
    print(f"  Downloading price data for {len(tickers)} tickers + SPY + ^TNX...")
    price_start = "2013-01-01"
    price_end   = "2025-08-01"

    ref = yf.download(["SPY", "^TNX"], start=price_start, end=price_end,
                      auto_adjust=True, progress=False)

    def _col(df, sym):
        if isinstance(df.columns, pd.MultiIndex):
            return df["Close"][[sym]].rename(columns={sym: "Close"}).dropna()
        return df[["Close"]].dropna()

    mkt_hist = _col(ref, "SPY")
    tnx_hist = _col(ref, "^TNX")
    print(f"  SPY: {len(mkt_hist)} days  |  ^TNX: {len(tnx_hist)} days")

    CHUNK = 200
    all_close: dict[str, pd.Series] = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        try:
            raw = yf.download(chunk, start=price_start, end=price_end,
                              auto_adjust=True, progress=False, threads=True)
            if raw.empty:
                continue
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) \
                    else raw[["Close"]].rename(columns={"Close": chunk[0]})
            for sym in chunk:
                if sym in close.columns:
                    s = close[sym].dropna()
                    if len(s) > 60:
                        all_close[sym] = s
        except Exception as e:
            print(f"  [WARN] chunk {i}-{i + CHUNK}: {e}")
        if (i // CHUNK + 1) % 5 == 0:
            print(f"  {min(i + CHUNK, len(tickers))}/{len(tickers)} downloaded", flush=True)

    price_hists = {sym: s.to_frame("Close") for sym, s in all_close.items()}
    print(f"  Price histories loaded: {len(price_hists)}/{len(tickers)}")

    result = (price_hists, mkt_hist, tnx_hist)
    with open(PRICE_CACHE, "wb") as f:
        pickle.dump(result, f)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MC panel builder (with disk cache)
# ═══════════════════════════════════════════════════════════════════════════════

def build_mc_panel(
    signals_df:  pd.DataFrame,
    mkt_hist:    pd.DataFrame,
    tnx_hist:    pd.DataFrame,
    price_hists: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if MC_CACHE.exists():
        print(f"  Loading MC cache: {MC_CACHE}")
        return pd.read_parquet(MC_CACHE)

    applicable = signals_df[signals_df["dcf_applicable"]].copy()
    n_tickers  = applicable["ticker"].nunique()
    n_rows     = len(applicable)
    print(f"\n  MCF COMPUTATION: {n_rows:,} (ticker, date) pairs, {n_tickers:,} unique tickers")
    print(f"  N_PATHS={N_PATHS:,}, N_WORKERS={N_WORKERS}, vectorised batch DCF")

    def _process_ticker(ticker: str, grp: pd.DataFrame) -> list[dict]:
        facts = load_edgar_facts(ticker)
        if facts is None:
            return []
        ph  = price_hists.get(ticker)
        out = []
        for _, row in grp.iterrows():
            as_of  = row["as_of"].date() if hasattr(row["as_of"], "date") \
                     else date.fromisoformat(str(row["as_of"])[:10])
            br     = _beta_at(ph, mkt_hist, as_of)
            rf     = _rf_at(tnx_hist, as_of)
            result = pit_mc(ticker, facts, as_of, float(row["price"]),
                            float(row["mktcap"]), br, rf, N_PATHS)
            if result is not None:
                mean_iv, std_iv, cv = result
                out.append({"ticker": ticker, "as_of": pd.Timestamp(as_of),
                            "mean_iv": mean_iv, "std_iv": std_iv, "cv": cv})
        return out

    ticker_groups = list(applicable.groupby("ticker"))
    rows: list[dict] = []
    n_done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(_process_ticker, t, g): t for t, g in ticker_groups}
        for f in as_completed(futs):
            n_done += 1
            rows.extend(f.result())
            if n_done % 50 == 0 or n_done == len(ticker_groups):
                elapsed = time.time() - t0
                print(f"  {n_done}/{len(ticker_groups)} tickers  |  "
                      f"{len(rows):,} MC rows  |  {elapsed:.0f}s", flush=True)

    if not rows:
        print("  [WARN] No MC results — returning empty DataFrame")
        return pd.DataFrame(columns=["ticker", "as_of", "mean_iv", "std_iv", "cv"])

    mc_df = pd.DataFrame(rows)
    mc_df["as_of"] = pd.to_datetime(mc_df["as_of"])
    mc_df.to_parquet(MC_CACHE, index=False)
    print(f"  MC stats saved → {MC_CACHE}  ({len(mc_df):,} rows)")
    return mc_df


# ═══════════════════════════════════════════════════════════════════════════════
# Full analysis panel assembly
# ═══════════════════════════════════════════════════════════════════════════════

def get_rebalance_dates() -> list[date]:
    return [date(y, REB_MONTH, REB_DAY) for y in range(START_YEAR, END_YEAR)]


def build_full_panel(
    signals_df:  pd.DataFrame,
    mc_df:       pd.DataFrame,
    price_hists: dict[str, pd.DataFrame],
    mkt_hist:    pd.DataFrame,
    tnx_hist:    pd.DataFrame,
) -> pd.DataFrame:
    print("\n  Assembling full panel (realized vol, leverage, forward returns)...")

    panel = signals_df.copy()
    panel["as_of"] = pd.to_datetime(panel["as_of"])

    if not mc_df.empty:
        mc_df = mc_df.copy()
        mc_df["as_of"] = pd.to_datetime(mc_df["as_of"])
        panel = panel.merge(mc_df[["ticker", "as_of", "mean_iv", "std_iv", "cv"]],
                            on=["ticker", "as_of"], how="left")
    else:
        for col in ("mean_iv", "std_iv", "cv"):
            panel[col] = float("nan")

    reb_dates = get_rebalance_dates()
    fwd_map: dict[pd.Timestamp, date] = {}
    for i, d in enumerate(reb_dates[:-1]):
        fwd_map[pd.Timestamp(d)] = reb_dates[i + 1]
    fwd_map[pd.Timestamp(reb_dates[-1])] = date(
        reb_dates[-1].year + 1, reb_dates[-1].month, reb_dates[-1].day
    )

    realized_vols: list[float] = []
    leverages:     list[float] = []
    fwd_rets:      list[float] = []

    for ticker, grp in panel.groupby("ticker"):
        facts = load_edgar_facts(ticker)
        ph    = price_hists.get(ticker)
        for idx, row in grp.iterrows():
            as_of = row["as_of"].date()
            realized_vols.append((_realized_vol(ph, as_of), idx))
            leverages.append((pit_leverage(facts, as_of), idx))
            next_d = fwd_map.get(pd.Timestamp(as_of))
            fwd_rets.append((_fwd_return(ph, as_of, next_d) if next_d else float("nan"), idx))

    vol_map = {idx: v for v, idx in realized_vols}
    lev_map = {idx: v for v, idx in leverages}
    fwd_map_out = {idx: v for v, idx in fwd_rets}

    panel["realized_vol"] = panel.index.map(vol_map)
    panel["leverage"]     = panel.index.map(lev_map)
    panel["fwd_ret"]      = panel.index.map(fwd_map_out)

    # Coverage report
    dcf_ok  = panel["dcf_applicable"].sum()
    cv_ok   = panel["cv"].notna().sum()
    vol_ok  = panel["realized_vol"].notna().sum()
    lev_ok  = panel["leverage"].notna().sum()
    fwd_ok  = panel["fwd_ret"].notna().sum()
    full_ok = panel[panel["dcf_applicable"] & panel["cv"].notna()
                    & panel["realized_vol"].notna() & panel["leverage"].notna()
                    & panel["fwd_ret"].notna()].shape[0]

    print(f"\n  Panel coverage:")
    print(f"    Total rows:          {len(panel):,}")
    print(f"    DCF-applicable:      {dcf_ok:,}")
    print(f"    With MC CV:          {cv_ok:,}  ({cv_ok/dcf_ok*100:.0f}% of DCF-applicable)")
    print(f"    With realized vol:   {vol_ok:,}")
    print(f"    With leverage:       {lev_ok:,}")
    print(f"    With forward return: {fwd_ok:,}")
    print(f"    Fully complete rows: {full_ok:,}  (all 5 fields present)")
    return panel


# ═══════════════════════════════════════════════════════════════════════════════
# Analysis utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _winsorise(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    return s.clip(s.quantile(lo), s.quantile(hi))


def perf_stats(returns: np.ndarray) -> dict:
    if len(returns) == 0:
        return {"ann_ret": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"), "n": 0}
    ret    = float(np.mean(returns))
    std    = float(np.std(returns, ddof=1))
    sharpe = ret / std if std > 0 else float("nan")
    cum    = np.cumprod(1 + returns)
    roll_max = np.maximum.accumulate(cum)
    max_dd = float((cum / roll_max - 1).min())
    return {"ann_ret": ret, "sharpe": sharpe, "max_dd": max_dd, "n": len(returns)}


def fama_macbeth(
    df:          pd.DataFrame,
    factor_cols: list[str],
    outcome_col: str = "fwd_ret",
) -> pd.DataFrame:
    """
    Fama-MacBeth: cross-sectional z-score, OLS at each date, average across dates.
    Returns DataFrame with columns [factor, mean_coef, t_stat, T].
    """
    coef_rows: list[dict] = []
    for _, grp in df.groupby("as_of"):
        sub = grp.dropna(subset=factor_cols + [outcome_col]).copy()
        if len(sub) < 20:
            continue
        for col in factor_cols:
            mu  = sub[col].mean()
            std = sub[col].std()
            sub[f"{col}_z"] = (sub[col] - mu) / std if std > 1e-10 else 0.0
        X_cols = [f"{col}_z" for col in factor_cols]
        X = np.column_stack([np.ones(len(sub))] + [sub[c].values for c in X_cols])
        y = sub[outcome_col].values
        try:
            coefs = np.linalg.lstsq(X, y, rcond=None)[0]
            row = {}
            for j, col in enumerate(factor_cols):
                row[col] = coefs[j + 1]
            coef_rows.append(row)
        except Exception:
            continue

    if len(coef_rows) < 3:
        return pd.DataFrame()

    coef_df = pd.DataFrame(coef_rows)
    T = len(coef_df)
    out_rows = []
    for col in factor_cols:
        if col not in coef_df.columns:
            continue
        mc  = coef_df[col].mean()
        se  = coef_df[col].std(ddof=1) / np.sqrt(T)
        t   = mc / se if se > 0 else float("nan")
        out_rows.append({"factor": col, "mean_coef": mc, "t_stat": t, "T": T})
    return pd.DataFrame(out_rows)


def _sig_stars(t: float) -> str:
    at = abs(t)
    if at > 2.576: return "***"
    if at > 1.960: return "**"
    if at > 1.645: return "*"
    return ""


def _print_fm_table(fm_df: pd.DataFrame, factors: list[str], label_map: dict) -> None:
    T = int(fm_df["T"].iloc[0]) if len(fm_df) else 0
    print(f"\n  Periods: T={T}  (each cross-section z-scored, OLS → forward return)")
    print(f"  {'Factor':<18} {'Mean coef':>12} {'t-stat':>9}  {'Sig'}")
    print(f"  {'-' * 48}")
    for _, r in fm_df.iterrows():
        label = label_map.get(r["factor"], r["factor"])
        sig   = _sig_stars(r["t_stat"])
        print(f"  {label:<18} {r['mean_coef']:>12.4f} {r['t_stat']:>9.2f}  {sig}")
    print(f"  *** p<0.01  ** p<0.05  * p<0.10  (two-tailed)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Pre-check
# ═══════════════════════════════════════════════════════════════════════════════

def run_precheck(
    signals_df:  pd.DataFrame,
    price_hists: dict[str, pd.DataFrame],
    mkt_hist:    pd.DataFrame,
    tnx_hist:    pd.DataFrame,
) -> bool:
    print("\n" + "=" * 70)
    print("  PRE-CHECK")
    print("=" * 70)

    applicable = signals_df[signals_df["dcf_applicable"]].copy()
    sample = applicable.sample(min(5, len(applicable)), random_state=42)

    # ── 1a. Distribution stats ─────────────────────────────────────────────────
    print("\n[1] Distribution stats (mean_iv, std_iv, CV) — 5 sample rows")
    print(f"  {'Ticker':<8} {'As-of':<13} {'Mean IV':>10} {'Std IV':>10} {'CV':>8}")
    print(f"  {'-' * 54}")
    mc_ok = 0
    for _, row in sample.iterrows():
        ticker = row["ticker"]
        as_of  = row["as_of"].date() if hasattr(row["as_of"], "date") \
                 else date.fromisoformat(str(row["as_of"])[:10])
        facts  = load_edgar_facts(ticker)
        br     = _beta_at(price_hists.get(ticker), mkt_hist, as_of)
        rf     = _rf_at(tnx_hist, as_of)
        result = pit_mc(ticker, facts, as_of, float(row["price"]), float(row["mktcap"]),
                        br, rf, n_paths=300)
        if result is not None:
            mean_iv, std_iv, cv = result
            print(f"  {ticker:<8} {str(as_of):<13} {mean_iv:>10.2f} {std_iv:>10.2f} {cv:>8.3f}")
            mc_ok += 1
        else:
            print(f"  {ticker:<8} {str(as_of):<13} {'GATE FAIL':>32}")

    if mc_ok == 0:
        print("  PRE-CHECK [1]: FAIL — no MC results produced")
        return False
    print(f"  PRE-CHECK [1]: PASS ✓  ({mc_ok}/{len(sample)} rows returned distribution stats)")

    # ── 1b. Realized volatility coverage ──────────────────────────────────────
    print("\n[2] Realized vol (252-day) coverage — sample 20 DCF-applicable tickers")
    sample20 = applicable.sample(min(20, len(applicable)), random_state=99)
    vol_avail = 0
    for _, row in sample20.iterrows():
        as_of = row["as_of"].date() if hasattr(row["as_of"], "date") \
                else date.fromisoformat(str(row["as_of"])[:10])
        ph = price_hists.get(row["ticker"])
        v  = _realized_vol(ph, as_of)
        if not math.isnan(v):
            vol_avail += 1
    pct = vol_avail / len(sample20) * 100
    print(f"  Vol available: {vol_avail}/{len(sample20)} = {pct:.0f}%")
    if pct < 50:
        print("  PRE-CHECK [2]: FAIL — vol coverage below 50%")
        return False
    print(f"  PRE-CHECK [2]: PASS ✓")

    # ── 1c. Leverage coverage ─────────────────────────────────────────────────
    print("\n[3] Leverage (total_debt / total_assets) coverage — same 20 tickers")
    lev_avail = 0
    for _, row in sample20.iterrows():
        as_of  = row["as_of"].date() if hasattr(row["as_of"], "date") \
                 else date.fromisoformat(str(row["as_of"])[:10])
        facts  = load_edgar_facts(row["ticker"])
        lev    = pit_leverage(facts, as_of)
        if not math.isnan(lev):
            lev_avail += 1
    pct_lev = lev_avail / len(sample20) * 100
    print(f"  Leverage available: {lev_avail}/{len(sample20)} = {pct_lev:.0f}%")
    if pct_lev < 50:
        print("  PRE-CHECK [3]: WARN — leverage coverage below 50%; results may be noisy")
    else:
        print(f"  PRE-CHECK [3]: PASS ✓")

    print("\n  PRE-CHECK: ALL PASS — proceeding to full panel build")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Test 1: Horse race (CV quintiles within DCF-cheap)
# ═══════════════════════════════════════════════════════════════════════════════

def run_test1(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 1 — HORSE RACE: CV quintiles within DCF-cheap basket")
    print("  Q1 = tight (low CV, high confidence)  Q5 = wide (high CV)")
    print("  Benchmark = equal-weight DCF-applicable universe each date")
    print("  *** Absolute Sharpes are UPPER BOUNDS (survivorship bias) ***")
    print("  *** T=10: read direction + magnitude, not significance alone ***")
    print("=" * 70)

    need = ["dcf_vp", "cv", "fwd_ret"]
    df   = panel[panel["dcf_applicable"] & panel["cv"].notna()
                 & panel["fwd_ret"].notna()].copy()

    q_rets: dict[int, list[float]] = {q: [] for q in range(N_QUINTILES)}
    bm_rets: list[float] = []
    n_dates = 0

    for as_of, grp in df.groupby("as_of"):
        grp = grp.dropna(subset=need)
        if len(grp) < N_QUINTILES * 4:
            continue
        grp = grp.copy()
        for col in ["dcf_vp", "cv"]:
            grp[col] = _winsorise(grp[col])

        # DCF-cheap: top quintile of dcf_vp
        try:
            grp["dcf_q"] = pd.qcut(grp["dcf_vp"], N_QUINTILES, labels=False,
                                    duplicates="drop")
        except Exception:
            continue
        cheap = grp[grp["dcf_q"] == N_QUINTILES - 1].dropna(subset=["cv"])
        if len(cheap) < N_QUINTILES:
            continue

        # Sort cheap by CV into quintiles
        try:
            cheap = cheap.copy()
            cheap["cv_q"] = pd.qcut(cheap["cv"], min(N_QUINTILES, len(cheap)),
                                     labels=False, duplicates="drop")
        except Exception:
            continue

        for q in range(N_QUINTILES):
            cell = cheap[cheap["cv_q"] == q]
            if len(cell) > 0:
                q_rets[q].append(float(cell["fwd_ret"].mean()))

        bm_rets.append(float(grp["fwd_ret"].mean()))
        n_dates += 1

    if n_dates == 0:
        print("  No dates with sufficient data.")
        return

    print(f"\n  Dates used: {n_dates}")
    print(f"\n  {'Quintile':<12} {'Ann. Return':>12} {'Sharpe':>9} {'MaxDD':>9}  {'vs BM':>8}")
    print(f"  {'-' * 56}")

    q_stats = {}
    bm      = perf_stats(np.array(bm_rets))
    for q in range(N_QUINTILES):
        arr  = np.array(q_rets[q])
        if len(arr) == 0:
            continue
        s    = perf_stats(arr)
        lab  = f"Q{q+1} (tight)" if q == 0 else (f"Q{q+1} (wide)" if q == N_QUINTILES-1 else f"Q{q+1}")
        diff = s["ann_ret"] - bm["ann_ret"]
        print(f"  {lab:<12} {s['ann_ret']:>11.1%} {s['sharpe']:>9.2f} {s['max_dd']:>9.1%}  {diff:>+8.1%}")
        q_stats[q] = s

    print(f"\n  {'Benchmark':<12} {bm['ann_ret']:>11.1%} {bm['sharpe']:>9.2f} {bm['max_dd']:>9.1%}")

    if 0 in q_stats and (N_QUINTILES - 1) in q_stats:
        diff_q1q5 = q_stats[0]["ann_ret"] - q_stats[N_QUINTILES - 1]["ann_ret"]
        print(f"\n  ══ HEADLINE: Q1 (tight) minus Q5 (wide) differential ══")
        print(f"  Q1 ann. return:  {q_stats[0]['ann_ret']:>+8.1%}")
        print(f"  Q5 ann. return:  {q_stats[N_QUINTILES-1]['ann_ret']:>+8.1%}")
        print(f"  Q1-minus-Q5:     {diff_q1q5:>+8.1%}")
        if diff_q1q5 > 0:
            print(f"  → Tight-cheap OUTPERFORMS wide-cheap by {diff_q1q5:.1%} per year")
            print(f"    (but T=10; confirm via Fama-MacBeth — Test 2)")
        elif diff_q1q5 > -0.02:
            print(f"  → Near-zero differential — CV carries little horse-race signal")
        else:
            print(f"  → Wide-cheap OUTPERFORMS tight-cheap by {abs(diff_q1q5):.1%}/yr — unexpected direction")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Test 2: Fama-MacBeth with full controls
# ═══════════════════════════════════════════════════════════════════════════════

def run_test2(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 2 — FAMA-MACBETH: CV with full controls")
    print("  THE GATE: is CV positive and significant AFTER vol + leverage?")
    print("=" * 70)

    factors = ["dcf_vp", "cv", "realized_vol", "leverage", "size", "mom"]
    label_map = {
        "dcf_vp": "DCF V/P", "cv": "CV (dispersion)",
        "realized_vol": "Vol (252d)", "leverage": "Leverage",
        "size": "SIZE", "mom": "MOM",
    }

    df = panel[panel["dcf_applicable"] & panel["cv"].notna()
               & panel["fwd_ret"].notna()].copy()
    for col in factors:
        if col in df.columns:
            df[col] = df.groupby("as_of")[col].transform(_winsorise)

    fm_df = fama_macbeth(df, factors)
    if fm_df.empty:
        print("  Insufficient data for Fama-MacBeth (need ≥3 dates with ≥20 stocks)")
        return

    _print_fm_table(fm_df, factors, label_map)

    # Gate verdict
    cv_row = fm_df[fm_df["factor"] == "cv"]
    if cv_row.empty:
        print("\n  GATE: INDETERMINATE (CV factor absent from regression)")
        return

    cv_coef = float(cv_row["mean_coef"].iloc[0])
    cv_t    = float(cv_row["t_stat"].iloc[0])
    T       = int(cv_row["T"].iloc[0])

    print(f"\n  ══ GATE VERDICT (T={T}) ══")
    print(f"  CV coefficient:  {cv_coef:.4f}   t-stat: {cv_t:.2f}")
    if cv_coef > 0 and abs(cv_t) >= 1.645:
        print(f"  GATE: PASS ✓  CV is POSITIVE and SIGNIFICANT after vol + leverage.")
        print(f"  Interpretation: tight uncertainty (low CV) within cheap stocks")
        print(f"  predicts higher returns beyond known risk factors.")
        print(f"  → The Monte Carlo captures information the point estimate cannot.")
    elif cv_coef > 0:
        print(f"  GATE: MARGINAL — CV is positive but insignificant (t={cv_t:.2f}).")
        print(f"  T=10 is a small sample; the direction is correct but power is weak.")
        print(f"  Cannot confidently assert Monte Carlo edge over risk premium.")
    elif cv_coef < 0:
        print(f"  GATE: NOTE — CV coefficient is negative after controls.")
        print(f"  Higher CV (wider) predicts lower returns; direction consistent with Q1>Q5")
        print(f"  but requires significance to claim Monte Carlo edge.")
        if abs(cv_t) >= 1.645:
            print(f"  CV is NEGATIVE and SIGNIFICANT: tighter DCF uncertainty reliably")
            print(f"  predicts higher returns even controlling for vol and leverage.")
        else:
            print(f"  Insignificant — not enough evidence to act on.")
    else:
        print(f"  CV ≈ 0 after controls. The Monte Carlo adds no predictive information")
        print(f"  beyond what vol and leverage already capture.")

    print(f"\n  ── INTERPRETATION GUIDE ──")
    print(f"  CV positive and significant → MC captures INFORMATION beyond known risk")
    print(f"  CV positive but loses significance → risk premium, not MC edge")
    print(f"  CV flat or negative → dispersion carries no predictive power (complexity tax)")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Test 3: Per-quintile breakdown and control delta
# ═══════════════════════════════════════════════════════════════════════════════

def run_test3(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 3 — COEFFICIENT DELTA: how much of CV is explained by risk?")
    print("  Reg A: [CV, SIZE, MOM]  → forward return")
    print("  Reg B: [CV, SIZE, MOM, vol, leverage]  → forward return")
    print("  Delta = Reg A coef − Reg B coef (portion explained by risk controls)")
    print("=" * 70)

    df = panel[panel["dcf_applicable"] & panel["cv"].notna()
               & panel["fwd_ret"].notna()].copy()
    for col in ["cv", "size", "mom", "realized_vol", "leverage"]:
        if col in df.columns:
            df[col] = df.groupby("as_of")[col].transform(_winsorise)

    label_map = {
        "cv": "CV (dispersion)", "size": "SIZE", "mom": "MOM",
        "realized_vol": "Vol (252d)", "leverage": "Leverage",
    }

    factors_a = ["cv", "size", "mom"]
    factors_b = ["cv", "size", "mom", "realized_vol", "leverage"]

    fm_a = fama_macbeth(df, factors_a)
    fm_b = fama_macbeth(df, factors_b)

    def _coef(fm, factor):
        r = fm[fm["factor"] == factor]
        if r.empty:
            return float("nan"), float("nan")
        return float(r["mean_coef"].iloc[0]), float(r["t_stat"].iloc[0])

    cv_a_c, cv_a_t = _coef(fm_a, "cv")
    cv_b_c, cv_b_t = _coef(fm_b, "cv")

    print(f"\n  Regression A: [CV, SIZE, MOM]")
    if not fm_a.empty:
        _print_fm_table(fm_a, factors_a, label_map)

    print(f"\n  Regression B: [CV, SIZE, MOM, vol, leverage]")
    if not fm_b.empty:
        _print_fm_table(fm_b, factors_b, label_map)

    print(f"\n  ══ CV CONTROL DELTA ══")
    print(f"  CV in Reg A (no risk controls):  {cv_a_c:>+8.4f}   t = {cv_a_t:.2f}")
    print(f"  CV in Reg B (with vol/leverage): {cv_b_c:>+8.4f}   t = {cv_b_t:.2f}")
    if not math.isnan(cv_a_c) and not math.isnan(cv_b_c) and abs(cv_a_c) > 1e-8:
        delta     = cv_a_c - cv_b_c
        pct_expl  = delta / cv_a_c * 100
        print(f"  Delta (A − B):                   {delta:>+8.4f}")
        print(f"  % of CV effect explained by risk:  {pct_expl:.0f}%")
        if pct_expl > 80:
            print(f"  → Most of the CV signal is risk premium (vol/leverage absorb it).")
            print(f"    Adding the Monte Carlo replaces, not adds to, known risk factors.")
        elif pct_expl > 40:
            print(f"  → CV is partially a risk premium ({pct_expl:.0f}% absorbed) but")
            print(f"    meaningful residual ({100-pct_expl:.0f}%) remains after controls.")
        else:
            print(f"  → Small risk-premium share. CV effect is mostly independent")
            print(f"    of volatility and leverage — consistent with a genuine MC edge.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Test 4: EBITDA/EV robustness
# ═══════════════════════════════════════════════════════════════════════════════

def run_test4(panel: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    print("  TEST 4 — EBITDA/EV ROBUSTNESS")
    print("  Do the same tests hold using EBITDA/EV-cheap instead of DCF-cheap?")
    print("  Prediction: dispersion of DCF IV should NOT predict returns")
    print("  inside a basket selected without DCF — null result expected.")
    print("=" * 70)

    df_all = panel[panel["dcf_applicable"] & panel["cv"].notna()
                   & panel["fwd_ret"].notna() & panel["ebitda_ev"].notna()].copy()
    for col in ["ebitda_ev", "cv", "size", "mom", "realized_vol", "leverage"]:
        if col in df_all.columns:
            df_all[col] = df_all.groupby("as_of")[col].transform(_winsorise)

    # ── 4a. CV quintile horse race within EBITDA/EV-cheap ─────────────────────
    print(f"\n[4a] CV quintile horse race within EBITDA/EV-cheap basket")
    q_rets: dict[int, list[float]] = {q: [] for q in range(N_QUINTILES)}
    bm_rets: list[float] = []
    n_dates = 0

    for as_of, grp in df_all.groupby("as_of"):
        grp = grp.dropna(subset=["ebitda_ev", "cv", "fwd_ret"])
        if len(grp) < N_QUINTILES * 4:
            continue
        try:
            grp = grp.copy()
            grp["ev_q"] = pd.qcut(grp["ebitda_ev"], N_QUINTILES, labels=False,
                                   duplicates="drop")
        except Exception:
            continue
        cheap = grp[grp["ev_q"] == N_QUINTILES - 1].dropna(subset=["cv"])
        if len(cheap) < N_QUINTILES:
            continue
        try:
            cheap = cheap.copy()
            cheap["cv_q"] = pd.qcut(cheap["cv"], min(N_QUINTILES, len(cheap)),
                                     labels=False, duplicates="drop")
        except Exception:
            continue
        for q in range(N_QUINTILES):
            cell = cheap[cheap["cv_q"] == q]
            if len(cell) > 0:
                q_rets[q].append(float(cell["fwd_ret"].mean()))
        bm_rets.append(float(grp["fwd_ret"].mean()))
        n_dates += 1

    if n_dates == 0:
        print("  No dates with sufficient data for EBITDA/EV horse race.")
    else:
        bm = perf_stats(np.array(bm_rets))
        print(f"  Dates used: {n_dates}")
        print(f"\n  {'Quintile':<12} {'Ann. Return':>12} {'Sharpe':>9}  {'vs BM':>8}")
        print(f"  {'-' * 46}")
        q_stats = {}
        for q in range(N_QUINTILES):
            arr = np.array(q_rets[q])
            if len(arr) == 0:
                continue
            s   = perf_stats(arr)
            lab = f"Q{q+1} (tight)" if q == 0 else (f"Q{q+1} (wide)" if q == N_QUINTILES-1 else f"Q{q+1}")
            print(f"  {lab:<12} {s['ann_ret']:>11.1%} {s['sharpe']:>9.2f}  {s['ann_ret']-bm['ann_ret']:>+8.1%}")
            q_stats[q] = s
        print(f"  {'Benchmark':<12} {bm['ann_ret']:>11.1%} {bm['sharpe']:>9.2f}")

        if 0 in q_stats and (N_QUINTILES - 1) in q_stats:
            diff = q_stats[0]["ann_ret"] - q_stats[N_QUINTILES - 1]["ann_ret"]
            print(f"\n  Q1-minus-Q5 (EBITDA/EV cheap):  {diff:>+8.1%}")
            if abs(diff) < 0.03:
                print(f"  → Near-zero — DCF CV adds no signal in EBITDA/EV-cheap basket.")
                print(f"    Supports H3: dispersion matters specifically via the DCF channel.")
            else:
                print(f"  → Non-trivial differential. CV may contain general value-within-cheap")
                print(f"    information not specific to the DCF engine.")

    # ── 4b. Fama-MacBeth within EBITDA/EV-selected universe ───────────────────
    print(f"\n[4b] Fama-MacBeth: CV coefficient within EBITDA/EV-cheap")
    factors_ev = ["ebitda_ev", "cv", "realized_vol", "leverage", "size", "mom"]
    label_map_ev = {
        "ebitda_ev": "EBITDA/EV", "cv": "CV (dispersion)",
        "realized_vol": "Vol (252d)", "leverage": "Leverage",
        "size": "SIZE", "mom": "MOM",
    }
    fm_ev = fama_macbeth(df_all, factors_ev)
    if fm_ev.empty:
        print("  Insufficient data for FM regression.")
    else:
        _print_fm_table(fm_ev, factors_ev, label_map_ev)
        cv_row = fm_ev[fm_ev["factor"] == "cv"]
        if not cv_row.empty:
            cv_c = float(cv_row["mean_coef"].iloc[0])
            cv_t = float(cv_row["t_stat"].iloc[0])
            print(f"\n  ══ ROBUSTNESS VERDICT ══")
            if abs(cv_t) >= 1.645:
                print(f"  CV is significant within EBITDA/EV universe (t={cv_t:.2f})")
                print(f"  → CV carries general value-stock information, not DCF-specific.")
                print(f"    Supports 'risk premium' interpretation over 'MC edge'.")
            else:
                print(f"  CV is NOT significant within EBITDA/EV universe (t={cv_t:.2f})")
                print(f"  → Null result confirmed. DCF CV only matters when you're doing DCF.")
                print(f"    Consistent with a genuine Monte Carlo-specific information channel.")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 70)
    print("  HYPOTHESIS 3: DOES MONTE CARLO DISPERSION CARRY INFORMATION?")
    print("  CV = std(simulated IV) / mean(simulated IV) from 2,000-path DCF MC")
    print(f"  Universe: DCF-applicable US stocks, {START_YEAR}–{END_YEAR}")
    print(f"  T = {END_YEAR - START_YEAR} annual cross-sections  |  Rebalance: Jun 30")
    print(f"  Trial count: 4 tests → Deflated Sharpe applies")
    print("=" * 70)

    # ── Load signals panel ────────────────────────────────────────────────────
    print(f"\n  Loading signals panel: {DCF_SIGNALS}")
    if not DCF_SIGNALS.exists():
        print(f"  ERROR: signals_panel.parquet not found at {DCF_SIGNALS}")
        print(f"  Run research/dcf_vs_multiple/run.py first to build the panel.")
        sys.exit(1)
    signals_df = pd.read_parquet(DCF_SIGNALS)
    signals_df["as_of"] = pd.to_datetime(signals_df["as_of"])
    print(f"  Loaded: {len(signals_df):,} rows, {signals_df['ticker'].nunique():,} tickers")
    print(f"  DCF-applicable: {signals_df['dcf_applicable'].sum():,} rows")
    print(f"  Dates: {sorted(signals_df['as_of'].dt.date.unique())}")

    # ── Load price data ───────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    tickers = signals_df["ticker"].unique().tolist()
    price_hists, mkt_hist, tnx_hist = load_price_data(tickers)

    # ── Pre-check ─────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    ok = run_precheck(signals_df, price_hists, mkt_hist, tnx_hist)
    if not ok:
        print("\n  ABORTING: pre-check failed.")
        sys.exit(1)

    # ── Build MC panel ────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    mc_df = build_mc_panel(signals_df, mkt_hist, tnx_hist, price_hists)

    # ── Build full analysis panel ─────────────────────────────────────────────
    print("\n" + "-" * 70)
    panel = build_full_panel(signals_df, mc_df, price_hists, mkt_hist, tnx_hist)

    # ── Run tests ─────────────────────────────────────────────────────────────
    run_test1(panel)
    run_test2(panel)
    run_test3(panel)
    run_test4(panel)

    print("\n" + "=" * 70)
    print("  DONE")
    print(f"  MC cache: {MC_CACHE}")
    print(f"  Price cache: {PRICE_CACHE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
