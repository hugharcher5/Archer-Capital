"""
Monte Carlo simulation layer for the DCF engine.

Reuses data.py, drivers.py, wacc.py, and the pure value() function.
No valuation logic is duplicated here.
"""

from __future__ import annotations
import copy, math, os
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')   # non-interactive; must precede pyplot
import matplotlib.pyplot as plt

from .data    import fetch_raw
from .drivers import compute_drivers
from .wacc    import compute_wacc
from .dcf     import Assumptions, value, TERMINAL_G, HIGH_GROWTH_THRESHOLD, MATURE_MARGIN_DEFAULT


# ─────────────────────────── Editable constants ───────────────────────────────

SPREAD_SIGMA: float            = 3.0     # PERT half-width = N × historical σ
HIGH_VOL_THRESHOLD: float      = 0.20    # Student-t copula when σ_growth exceeds this
STUDENT_T_DF: int              = 5       # degrees of freedom for Student-t copula
WACC_SPREAD_ABS: float         = 0.015   # WACC PERT half-width (absolute, e.g. ±1.5%)
TERM_G_MIN: float              = 0.015   # 1.5% floor on terminal g
TERM_G_MAX_ABS: float          = 0.040   # 4.0% ceiling (also bounded by WACC − 1%)
TERM_G_MODE: float             = TERMINAL_G          # 2.5%
WACC_TG_GAP: float             = 0.010   # minimum enforced gap: WACC − terminal_g
PROFITABLE_MARGIN_SPREAD: float   = 0.08  # target_margin PERT half-width for profitable cos
UNPROFITABLE_MARGIN_SPREAD: float = 0.15  # wider spread for currently-unprofitable cos

# Annual FX volatility by reporting currency (GBM σ)
FX_ANNUAL_VOL: dict[str, float] = {
    'USD': 0.00, 'EUR': 0.08, 'GBP': 0.09,
    'JPY': 0.10, 'TWD': 0.12, 'KRW': 0.13,
    'CNY': 0.08, 'HKD': 0.03, '_default': 0.15,
}

# Correlation matrix: [rev_growth, ebit_margin, terminal_g, wacc, target_margin]
# Growth ↔ margin: +0.40 (tech operating leverage)
# Growth ↔ terminal_g: +0.30 (faster growers sustain higher perpetuity rates)
# Growth ↔ wacc: +0.20 (higher growth → higher perceived risk)
# Margin ↔ wacc: −0.10 (higher profitability → slightly lower risk)
# Growth ↔ target_margin: +0.30 (fast growers tend toward higher long-run margins)
# margin ↔ target_margin: +0.60 (current margin is strongest predictor of maturity)
# wacc ↔ target_margin: −0.15 (more profitable at maturity → lower perceived risk)
CORR: np.ndarray = np.array([
    #  rev_g   margin  term_g   wacc  tgt_mgn
    [  1.00,    0.40,   0.30,   0.20,   0.30],  # rev_growth
    [  0.40,    1.00,   0.10,  -0.10,   0.60],  # ebit_margin
    [  0.30,    0.10,   1.00,   0.10,   0.10],  # terminal_g
    [  0.20,   -0.10,   0.10,   1.00,  -0.15],  # wacc
    [  0.30,    0.60,   0.10,  -0.15,   1.00],  # target_margin
], dtype=float)


# ─────────────────────────────── PSD check ────────────────────────────────────

def _ensure_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Verify C is positive semi-definite.
    If any eigenvalue < 0, apply nearest-PSD correction (clip + rescale to
    correlation matrix) and report the fix.
    """
    C = (C + C.T) / 2.0
    eigvals = np.linalg.eigvalsh(C)
    if np.all(eigvals >= -1e-10):
        print(f"  PSD check: all {len(eigvals)} eigenvalues ≥ 0  ✓"
              f"  (min λ = {eigvals.min():.2e})")
        return C
    n_neg = int((eigvals < 0).sum())
    print(f"  PSD check: {n_neg} negative eigenvalue(s) — applying nearest-PSD fix.")
    lam, V = np.linalg.eigh(C)
    C2 = V @ np.diag(np.maximum(lam, eps)) @ V.T
    d  = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


# ──────────────────────── PERT inverse CDF ────────────────────────────────────

def _pert_ppf(u: np.ndarray, lo: float, mode: float, hi: float) -> np.ndarray:
    """
    Inverse PERT CDF at uniform quantiles u ∈ (0,1).
    PERT is a scaled Beta: α = 1 + 4(mode−lo)/(hi−lo), β = 1 + 4(hi−mode)/(hi−lo).
    Degenerates to a point mass at mode when lo == hi.
    """
    if abs(hi - lo) < 1e-12:
        return np.full(len(u), mode, dtype=float)
    r = hi - lo
    a = 1.0 + 4.0 * (mode - lo) / r
    b = 1.0 + 4.0 * (hi - mode) / r
    return lo + r * scipy.stats.beta.ppf(np.clip(u, 1e-9, 1.0 - 1e-9), a, b)


# ──────────────────────────── Copula draws ────────────────────────────────────

def _gaussian_uniforms(n: int, C: np.ndarray,
                        rng: np.random.Generator) -> np.ndarray:
    """(n, k) correlated uniform draws via Gaussian copula."""
    L  = np.linalg.cholesky(C)
    ZC = rng.standard_normal((n, C.shape[0])) @ L.T
    return scipy.stats.norm.cdf(ZC)


def _student_t_uniforms(n: int, C: np.ndarray, df: int,
                         rng: np.random.Generator) -> np.ndarray:
    """(n, k) correlated uniform draws via Student-t copula (fatter tails)."""
    L  = np.linalg.cholesky(C)
    ZC = rng.standard_normal((n, C.shape[0])) @ L.T
    W  = rng.chisquare(df, size=n)             # chi-squared scaling factor
    T  = ZC / np.sqrt(W[:, None] / df)        # multivariate-t
    return scipy.stats.t.cdf(T, df=df)


# ──────────────── Build base Assumptions (mirrors run_dcf._assemble) ──────────

def _build_base(raw, drivers, wacc_r) -> Assumptions:
    net_debt      = raw.total_debt - raw.cash
    fy            = 10 if drivers.revenue_growth > HIGH_GROWTH_THRESHOLD else 5
    ebitda        = ((float(raw.ebit.iloc[-1]) if not raw.ebit.empty else 0.0) +
                     (float(raw.da.iloc[-1])   if not raw.da.empty   else 0.0))
    target_margin = max(drivers.best_ebit_margin, MATURE_MARGIN_DEFAULT)
    return Assumptions(
        ticker=raw.ticker,        currency=raw.currency,
        start_revenue=float(raw.revenue.iloc[-1]),
        revenue_growth=drivers.revenue_growth,
        ebit_margin=drivers.ebit_margin,
        target_margin=target_margin,
        tax_rate=drivers.tax_rate,
        da_pct=drivers.da_pct,
        capex_pct=drivers.capex_pct,
        nwc_pct=drivers.nwc_pct,
        terminal_g=TERM_G_MODE,
        wacc=wacc_r.wacc,
        net_debt=net_debt,
        diluted_shares=raw.diluted_shares,
        fx_rate=raw.fx_rate,
        forecast_years=fy,
        rf=wacc_r.rf,                       erp=wacc_r.erp,
        beta_adj=wacc_r.beta_adj,           cost_of_equity=wacc_r.cost_of_equity,
        cost_of_debt_pretax=wacc_r.cost_of_debt_pretax,
        cost_of_debt_aftertax=wacc_r.cost_of_debt_aftertax,
        equity_weight=wacc_r.equity_weight, debt_weight=wacc_r.debt_weight,
        implied_rating=wacc_r.implied_rating,
        current_price_usd=raw.current_price_usd,
        current_ebitda=ebitda,
        market_ev_local=raw.market_cap_local + net_debt,
    )


# ──────────────────────────── Core simulation ─────────────────────────────────

def _run_sims(
    base: Assumptions,
    sg: float,              # drivers.std_revenue_growth
    sm: float,              # drivers.std_ebit_margin
    corr: np.ndarray,
    n: int,
    spread_sigma: float,
    rng: np.random.Generator,
    copula: str,            # 'gaussian' | 'student-t'
) -> np.ndarray:
    """
    Draw n per-share intrinsic values (USD).
    spread_sigma=0 → all inputs pinned at base-case mode (consistency check).
    """
    w = base.wacc

    if spread_sigma < 1e-9:
        # Degenerate distributions: point masses at the base-case modes
        rg_lo = rg_hi = base.revenue_growth
        em_lo = em_hi = base.ebit_margin
        tg_lo = tg_hi = TERM_G_MODE
        wa_lo = wa_hi = base.wacc
        tm_lo = tm_hi = base.target_margin
    else:
        rg_lo = max(base.revenue_growth - spread_sigma * sg, -0.30)
        rg_hi = min(base.revenue_growth + spread_sigma * sg,  1.50)
        em_lo = max(base.ebit_margin    - spread_sigma * sm, -0.20)
        em_hi = min(base.ebit_margin    + spread_sigma * sm,  0.75)
        tg_lo = TERM_G_MIN
        tg_hi = max(TERM_G_MIN + 1e-6, min(TERM_G_MAX_ABS, w - WACC_TG_GAP))
        wa_lo = max(w - WACC_SPREAD_ABS, 0.04)
        wa_hi = w + WACC_SPREAD_ABS
        tm_spread = (UNPROFITABLE_MARGIN_SPREAD if base.ebit_margin < 0
                     else PROFITABLE_MARGIN_SPREAD)
        tm_lo = max(base.target_margin - tm_spread, -0.10)
        tm_hi = min(base.target_margin + tm_spread,  0.60)

    # Correlated uniform draws
    if copula == 'student-t':
        U = _student_t_uniforms(n, corr, STUDENT_T_DF, rng)
    else:
        U = _gaussian_uniforms(n, corr, rng)

    # Map uniforms → PERT marginals
    rg_s = _pert_ppf(U[:, 0], rg_lo, base.revenue_growth, rg_hi)
    em_s = _pert_ppf(U[:, 1], em_lo, base.ebit_margin,    em_hi)
    tg_s = _pert_ppf(U[:, 2], tg_lo, TERM_G_MODE,         tg_hi)
    wa_s = _pert_ppf(U[:, 3], wa_lo, base.wacc,           wa_hi)
    tm_s = _pert_ppf(U[:, 4], tm_lo, base.target_margin,  tm_hi)

    # Hard clamp: enforce WACC − terminal_g ≥ WACC_TG_GAP at all times
    tg_s = np.minimum(tg_s, wa_s - WACC_TG_GAP)

    # FX path — GBM for non-USD, only when spread > 0
    if base.currency != 'USD' and spread_sigma > 1e-9:
        vol = FX_ANNUAL_VOL.get(base.currency, FX_ANNUAL_VOL['_default'])
        T   = base.forecast_years
        Z   = rng.standard_normal(n)
        fx_s = base.fx_rate * np.exp(-0.5 * vol**2 * T + vol * math.sqrt(T) * Z)
    else:
        fx_s = np.full(n, base.fx_rate)

    # Inner loop: one copy, mutate per draw, call pure value()
    results = np.empty(n)
    skipped = 0
    a = copy.copy(base)

    for i in range(n):
        a.revenue_growth = float(rg_s[i])
        a.ebit_margin    = float(em_s[i])
        a.target_margin  = float(tm_s[i])
        a.terminal_g     = float(tg_s[i])
        a.wacc           = float(wa_s[i])
        a.fx_rate        = float(fx_s[i])
        try:
            results[i] = value(a)
        except Exception:
            results[i] = float('nan')
            skipped    += 1

    if skipped:
        print(f"  ⚠  {skipped}/{n} draws skipped (unexpected error); excluded from results.")
    return results[~np.isnan(results)]


# ───────────────────────────── Histogram ──────────────────────────────────────

def _save_histogram(ticker, results, price, n_valid, copula_label,
                    det_val, p10, p50, p90) -> str:
    os.makedirs('output', exist_ok=True)
    path = f'output/{ticker}_montecarlo.png'

    # Winsorise display only (P1–P99) so extreme tails don't squash the body
    p1, p99 = np.percentile(results, [1, 99])
    display  = results[(results >= p1) & (results <= p99)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.hist(display, bins=120, color='steelblue', edgecolor='none', alpha=0.75)
    ax.axvline(price,    color='crimson',    lw=2.0,          label=f'Market price   ${price:.2f}')
    ax.axvline(p50,      color='seagreen',   lw=2.0, ls='--', label=f'P50            ${p50:.2f}')
    ax.axvline(det_val,  color='gold',       lw=1.5, ls='-.', label=f'Base case      ${det_val:.2f}')
    ax.axvline(p10,      color='darkorange', lw=1.2, ls=':',  label=f'P10 / P90     ${p10:.2f} / ${p90:.2f}')
    ax.axvline(p90,      color='darkorange', lw=1.2, ls=':')
    ax.set_xlabel('Intrinsic Value per Share (USD)', fontsize=12)
    ax.set_ylabel('Frequency',                       fontsize=12)
    ax.set_title(
        f'{ticker} — Monte Carlo DCF  ({n_valid:,} simulations, {copula_label} copula)',
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ──────────────────────────── Public entry point ──────────────────────────────

def run_monte_carlo(ticker: str, n_sims: int = 10_000) -> None:
    # ── 1. Fetch and build base case ──────────────────────────────────────────
    raw    = fetch_raw(ticker)
    drvrs  = compute_drivers(raw)
    wacc_r = compute_wacc(raw, drvrs)
    base   = _build_base(raw, drvrs, wacc_r)
    det_val = value(base)

    sg = drvrs.std_revenue_growth
    sm = drvrs.std_ebit_margin
    w  = base.wacc

    # ── 2. Setup ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SETUP — {ticker.upper()}")
    print(f"{'='*60}")

    corr = _ensure_psd(CORR.copy())

    use_t        = sg > HIGH_VOL_THRESHOLD
    copula_key   = 'student-t' if use_t else 'gaussian'
    copula_label = f'Student-t (df={STUDENT_T_DF})' if use_t else 'Gaussian'
    print(f"\n  Copula : {copula_label}"
          f"   (σ_growth = {sg:.2%},  threshold = {HIGH_VOL_THRESHOLD:.0%})")

    tg_hi = max(TERM_G_MIN + 1e-6, min(TERM_G_MAX_ABS, w - WACC_TG_GAP))
    print(f"\n  PERT bounds (±{SPREAD_SIGMA:.0f}σ):")
    print(f"  {'Input':<22} {'Min':>10}  {'Mode':>10}  {'Max':>10}")
    print(f"  {'─'*56}")
    print(f"  {'Revenue growth':<22}"
          f" {max(base.revenue_growth - SPREAD_SIGMA*sg, -0.30):>10.2%}"
          f"  {base.revenue_growth:>10.2%}"
          f"  {min(base.revenue_growth + SPREAD_SIGMA*sg, 1.50):>10.2%}")
    print(f"  {'EBIT margin':<22}"
          f" {max(base.ebit_margin - SPREAD_SIGMA*sm, -0.20):>10.2%}"
          f"  {base.ebit_margin:>10.2%}"
          f"  {min(base.ebit_margin + SPREAD_SIGMA*sm, 0.75):>10.2%}")
    print(f"  {'Terminal g':<22}"
          f" {TERM_G_MIN:>10.2%}  {TERM_G_MODE:>10.2%}  {tg_hi:>10.2%}")
    print(f"  {'WACC':<22}"
          f" {max(w - WACC_SPREAD_ABS, 0.04):>10.2%}"
          f"  {w:>10.2%}"
          f"  {w + WACC_SPREAD_ABS:>10.2%}")
    tm_spread_disp = (UNPROFITABLE_MARGIN_SPREAD if base.ebit_margin < 0
                      else PROFITABLE_MARGIN_SPREAD)
    print(f"  {'Target margin':<22}"
          f" {max(base.target_margin - tm_spread_disp, -0.10):>10.2%}"
          f"  {base.target_margin:>10.2%}"
          f"  {min(base.target_margin + tm_spread_disp, 0.60):>10.2%}")
    if base.currency != 'USD':
        fxv = FX_ANNUAL_VOL.get(base.currency, FX_ANNUAL_VOL['_default'])
        print(f"  {'FX (' + base.currency + '/USD)':<22}"
              f"  {'GBM (lognormal)':>10}  {base.fx_rate:>10.5f}  {'σ=' + f'{fxv:.0%}':>10}")

    # ── 3. Consistency check: zero-variance run ───────────────────────────────
    print(f"\n  Consistency check (zero-variance run, n=50)...")
    rng_cc  = np.random.default_rng(0)
    cc_sims = _run_sims(base, sg, sm, corr, 50, 0.0, rng_cc, copula_key)
    cc_p50  = float(np.median(cc_sims))
    cc_ok   = abs(cc_p50 - det_val) < 0.01
    print(f"  Deterministic value()  = ${det_val:.4f}")
    print(f"  Zero-variance MC P50   = ${cc_p50:.4f}")
    print(f"  {'✓  PASS' if cc_ok else '✗  FAIL — check simulation centring'}")

    # ── 4. Full simulation ────────────────────────────────────────────────────
    print(f"\n  Running {n_sims:,} simulations ({copula_label} copula)...")
    rng  = np.random.default_rng(seed=42)
    sims = _run_sims(base, sg, sm, corr, n_sims, SPREAD_SIGMA, rng, copula_key)
    n_v  = len(sims)

    # ── 5. Statistics ─────────────────────────────────────────────────────────
    pcts         = np.percentile(sims, [10, 25, 50, 75, 90])
    p10, p25, p50, p75, p90 = pcts
    mean_v       = float(np.mean(sims))
    std_v        = float(np.std(sims))
    pct_under    = float(np.mean(sims > base.current_price_usd)) * 100
    price        = base.current_price_usd

    print(f"\n{'#'*60}")
    print(f"  MONTE CARLO RESULTS — {ticker.upper()}  ({n_v:,} valid simulations)")
    print(f"{'#'*60}")
    print(f"  Copula        : {copula_label}")
    print(f"  Current price : ${price:.2f}")
    print(f"  Base case     : ${det_val:.2f}")
    print(f"\n  Intrinsic value per share (USD):")
    print(f"  {'─'*36}")
    print(f"  P10   : ${p10:>9.2f}")
    print(f"  P25   : ${p25:>9.2f}")
    print(f"  P50   : ${p50:>9.2f}   ← median")
    print(f"  P75   : ${p75:>9.2f}")
    print(f"  P90   : ${p90:>9.2f}")
    print(f"  {'─'*36}")
    print(f"  Mean  : ${mean_v:>9.2f}")
    print(f"  Stdev : ${std_v:>9.2f}")
    print(f"  Min   : ${np.min(sims):>9.2f}   Max : ${np.max(sims):.2f}")
    print(f"\n  P(undervalued) : {pct_under:.1f}%"
          f"   (draws where intrinsic > ${price:.2f})")

    # ── 6. Histogram ──────────────────────────────────────────────────────────
    path = _save_histogram(
        ticker, sims, price, n_v, copula_label,
        det_val, float(p10), float(p50), float(p90),
    )
    print(f"\n  Histogram → {path}")
