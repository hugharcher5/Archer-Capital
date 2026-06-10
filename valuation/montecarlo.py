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

from .data      import fetch_raw
from .drivers   import compute_drivers
from .wacc      import compute_wacc
from .dcf       import Assumptions, value, detailed_value, TERMINAL_G, HIGH_GROWTH_THRESHOLD, MATURE_MARGIN_DEFAULT
from .result    import ValuationResult
from .reconcile import FIELD_TYPE, ALL_FIELDS


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

    Defensive: mode is clamped to [lo, hi] so that the Beta parameters α and β
    are always ≥ 1.  A caller that passes mode outside [lo, hi] (e.g. because a
    hard cap clips hi below the base-case value) would otherwise get β < 0, which
    causes scipy to return NaN for every draw and silently kills all simulations.
    """
    if abs(hi - lo) < 1e-12:
        return np.full(len(u), mode, dtype=float)
    mode = min(max(mode, lo), hi)   # guarantee lo ≤ mode ≤ hi
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
    sg: float,              # σ_eff for revenue growth (historical + cross-source in quadrature)
    sm: float,              # σ_eff for ebit margin (historical + cross-source in quadrature)
    corr: np.ndarray,
    n: int,
    spread_sigma: float,
    rng: np.random.Generator,
    copula: str,                 # 'gaussian' | 'student-t'
    sw: float = WACC_SPREAD_ABS,    # σ_eff for WACC PERT half-width
    shares_cross: float = 0.0,  # absolute σ_cross for diluted_shares (0 = not promoted)
    shares_promoted: bool = False,
    nd_cross: float = 0.0,      # absolute σ_cross for net_debt (0 = not promoted)
    nd_promoted: bool = False,
) -> np.ndarray:
    """
    Draw n per-share intrinsic values (USD).
    spread_sigma=0 → all inputs pinned at base-case mode (consistency check).
    When spread_sigma=0 the promoted-variable flags are also ignored, so the
    zero-variance run exactly reproduces the deterministic value().
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
        wa_lo = max(w - spread_sigma * sw, 0.04)
        wa_hi = w + spread_sigma * sw
        tm_spread = (UNPROFITABLE_MARGIN_SPREAD if base.ebit_margin < 0
                     else PROFITABLE_MARGIN_SPREAD)
        tm_lo = max(base.target_margin - tm_spread, -0.10)
        # Cap at 0.75 (matches em_hi) so that high-margin companies such as NVDA
        # (target_margin ≈ 66%) don't produce tm_hi < mode, which would give a
        # negative Beta parameter and silently NaN-out every simulation draw.
        tm_hi = min(base.target_margin + tm_spread,  0.75)

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

    # Promoted balance-sheet variables: diluted_shares and net_debt.
    # Sampled independently (not through the copula — balance-sheet uncertainty
    # is orthogonal to operating-driver forecast uncertainty).
    # Narrow PERT centred on preferred-source value, half-width = σ_cross.
    # Only active when spread_sigma > 0 (not in the consistency-check run).
    if shares_promoted and spread_sigma > 1e-9:
        sh_lo = max(0.0, base.diluted_shares - shares_cross)
        sh_hi = base.diluted_shares + shares_cross
        sh_s  = _pert_ppf(rng.uniform(0.0, 1.0, n), sh_lo, base.diluted_shares, sh_hi)
    else:
        sh_s = np.full(n, base.diluted_shares)

    if nd_promoted and spread_sigma > 1e-9:
        nd_lo = base.net_debt - nd_cross
        nd_hi = base.net_debt + nd_cross
        nd_s  = _pert_ppf(rng.uniform(0.0, 1.0, n), nd_lo, base.net_debt, nd_hi)
    else:
        nd_s = np.full(n, base.net_debt)

    # Inner loop: one copy, mutate per draw, call pure value()
    results  = np.empty(n)
    n_except = 0
    a = copy.copy(base)

    for i in range(n):
        a.revenue_growth = float(rg_s[i])
        a.ebit_margin    = float(em_s[i])
        a.target_margin  = float(tm_s[i])
        a.terminal_g     = float(tg_s[i])
        a.wacc           = float(wa_s[i])
        a.fx_rate        = float(fx_s[i])
        a.diluted_shares = float(sh_s[i])
        a.net_debt       = float(nd_s[i])
        try:
            results[i] = value(a)
        except Exception:
            results[i] = float('nan')
            n_except   += 1

    valid     = results[~np.isnan(results)]
    n_nan     = n - len(valid)
    if n_nan:
        # n_nan covers both exception-thrown draws and draws where value() returned
        # NaN (e.g. from invalid PERT parameters producing NaN sampled inputs).
        print(f"  ⚠  {n_nan}/{n} draws discarded ({n_except} exceptions, "
              f"{n_nan - n_except} silent NaN); excluded from results.")
    return valid


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


# ─────────────────────────── Public functions ─────────────────────────────────

def run_valuation(
    ticker: str,
    n_sims: int = 10_000,
    sigma_cross: dict[str, float] | None = None,
    reconcile_result=None,
) -> ValuationResult:
    """
    All computation, zero I/O.  n_sims=0 → skip Monte Carlo.

    sigma_cross: dict from ReconcileResult.sigma_cross.  When provided, the
    effective per-variable σ is combined with the historical σ in quadrature:
        σ_eff = sqrt(σ_hist² + σ_cross²)
    Only DATA-classified fields appear in sigma_cross; DEFINITIONAL fields do
    not widen anything.  Share count and net debt are promoted to sampled
    variables if their relative disagreement exceeds 2%.

    reconcile_result: ReconcileResult from reconcile().  When provided, recon_fields
    and recon_sigma are populated on the returned ValuationResult for UI display.
    """
    if sigma_cross is None:
        sigma_cross = {}

    raw    = fetch_raw(ticker)
    drvrs  = compute_drivers(raw)
    wacc_r = compute_wacc(raw, drvrs)
    sw = wacc_r.std_wacc   # historical WACC σ (or fallback)
    base   = _build_base(raw, drvrs, wacc_r)
    dcf_r  = detailed_value(base)

    sg_hist = drvrs.std_revenue_growth
    sm_hist = drvrs.std_ebit_margin

    # Effective σ: combine historical spread with cross-source uncertainty in quadrature.
    # σ_eff can only grow, never shrink — adding a zero σ_cross leaves σ_hist unchanged.
    sg = math.sqrt(sg_hist**2 + sigma_cross.get("revenue_growth", 0.0)**2)
    sm = math.sqrt(sm_hist**2 + sigma_cross.get("ebit_margin",    0.0)**2)

    # Promoted balance-sheet variables.
    # A DATA variable is promoted when its relative cross-source spread > 2%.
    sc_abs = sigma_cross.get("diluted_shares", 0.0)
    shares_promoted = (base.diluted_shares > 0 and
                       sc_abs / base.diluted_shares > 0.02)

    nd_abs = sigma_cross.get("net_debt", 0.0)
    nd_promoted = (abs(base.net_debt) > 1e-9 and
                   nd_abs / abs(base.net_debt) > 0.02)

    # ── Reconciliation metadata for UI display ────────────────────────────────
    recon_fields: dict = {}
    recon_sigma:  dict = {}
    if reconcile_result is not None:
        for f in ALL_FIELDS:
            d = reconcile_result.disagreement.get(f, float('nan'))
            recon_fields[f] = {
                "source":       reconcile_result.field_sources.get(f, "Yahoo"),
                "disagree_pct": d * 100 if math.isfinite(d) else None,
                "field_type":   FIELD_TYPE.get(f, "DATA"),
            }
        recon_sigma["revenue_growth"] = {
            "sigma_hist":  sg_hist,
            "sigma_cross": sigma_cross.get("revenue_growth", 0.0),
            "sigma_eff":   sg,
        }
        recon_sigma["ebit_margin"] = {
            "sigma_hist":  sm_hist,
            "sigma_cross": sigma_cross.get("ebit_margin", 0.0),
            "sigma_eff":   sm,
        }
        if shares_promoted:
            recon_sigma["diluted_shares"] = {
                "sigma_hist": 0.0, "sigma_cross": sc_abs, "sigma_eff": sc_abs,
            }
        if nd_promoted:
            recon_sigma["net_debt"] = {
                "sigma_hist": 0.0, "sigma_cross": nd_abs, "sigma_eff": nd_abs,
            }

    nan = float('nan')
    if n_sims <= 0:
        return ValuationResult(
            ticker=ticker.upper(), currency=base.currency,
            current_price_usd=base.current_price_usd,
            drivers=drvrs, wacc_result=wacc_r,
            assumptions=base, dcf=dcf_r,
            sims=np.array([]), n_valid=0,
            copula_label='', consistency_pass=False, cc_p50=nan,
            p10=nan, p25=nan, p50=nan, p75=nan, p90=nan,
            mean_val=nan, std_val=nan, pct_undervalued=nan,
            sigma_cross=sigma_cross,
            recon_fields=recon_fields, recon_sigma=recon_sigma,
            sw=sw,
        )

    corr       = _ensure_psd(CORR.copy())
    use_t      = sg > HIGH_VOL_THRESHOLD
    copula_key = 'student-t' if use_t else 'gaussian'
    copula_lbl = f'Student-t (df={STUDENT_T_DF})' if use_t else 'Gaussian'

    # Consistency check: zero spread_sigma pins all variables to mode.
    # Pass sg/sm (with any σ_cross baked in) but spread_sigma=0 overrides bounds to
    # point masses, so σ_eff doesn't matter — output must equal deterministic value().
    cc_sims = _run_sims(base, sg, sm, corr, 50, 0.0,
                        np.random.default_rng(0), copula_key, sw=sw)
    cc_p50  = float(np.median(cc_sims))
    cc_ok   = abs(cc_p50 - dcf_r.value_per_share_usd) < 0.01

    # Full simulation
    sims = _run_sims(
        base, sg, sm, corr, n_sims, SPREAD_SIGMA,
        np.random.default_rng(seed=42), copula_key,
        sw=sw,
        shares_cross=sc_abs,   shares_promoted=shares_promoted,
        nd_cross=nd_abs,       nd_promoted=nd_promoted,
    )
    n_v  = len(sims)

    if n_v == 0:
        raise ValueError(
            f"{ticker.upper()}: all {n_sims:,} Monte Carlo draws produced NaN — "
            "simulation aborted.  This usually means a PERT parameter was invalid "
            "(mode outside [lo, hi]) or every DCF evaluation overflowed.  "
            "Check that WACC > terminal_g and that all driver assumptions are finite."
        )

    # Stats computed on the full un-winsorized array
    p10, p25, p50, p75, p90 = (float(x) for x in np.percentile(sims, [10, 25, 50, 75, 90]))
    mean_v    = float(np.mean(sims))
    std_v     = float(np.std(sims))
    pct_under = float(np.mean(sims > base.current_price_usd)) * 100

    return ValuationResult(
        ticker=ticker.upper(), currency=base.currency,
        current_price_usd=base.current_price_usd,
        drivers=drvrs, wacc_result=wacc_r,
        assumptions=base, dcf=dcf_r,
        sims=sims, n_valid=n_v,
        copula_label=copula_lbl, consistency_pass=cc_ok, cc_p50=cc_p50,
        p10=p10, p25=p25, p50=p50, p75=p75, p90=p90,
        mean_val=mean_v, std_val=std_v, pct_undervalued=pct_under,
        sigma_cross=sigma_cross,
        recon_fields=recon_fields, recon_sigma=recon_sigma,
        sw=sw,
    )


def print_valuation(result: ValuationResult) -> None:
    """Terminal formatter — consumes a ValuationResult, produces the existing output + PNG."""
    base  = result.assumptions
    drvrs = result.drivers
    sc    = result.sigma_cross
    sg_hist = drvrs.std_revenue_growth
    sm_hist = drvrs.std_ebit_margin
    sg    = math.sqrt(sg_hist**2 + sc.get("revenue_growth", 0.0)**2)  # σ_eff
    sm    = math.sqrt(sm_hist**2 + sc.get("ebit_margin",    0.0)**2)  # σ_eff
    w     = base.wacc
    det   = result.dcf.value_per_share_usd

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SETUP — {result.ticker}")
    print(f"{'='*60}")
    cross_note = ""
    if sc:
        parts = [f"σ_cross({k})={v:.2%}" for k, v in sc.items()
                 if k in ("revenue_growth", "ebit_margin")]
        if parts:
            cross_note = "  cross-source: " + ", ".join(parts)
    print(f"\n  Copula : {result.copula_label}"
          f"   (σ_growth_eff = {sg:.2%},  threshold = {HIGH_VOL_THRESHOLD:.0%})"
          + cross_note)

    tg_hi = max(TERM_G_MIN + 1e-6, min(TERM_G_MAX_ABS, w - WACC_TG_GAP))
    print(f"\n  PERT bounds (±{SPREAD_SIGMA:.0f}σ_eff):")
    print(f"  {'Input':<22} {'Min':>10}  {'Mode':>10}  {'Max':>10}  {'σ_eff':>8}")
    print(f"  {'─'*65}")
    print(f"  {'Revenue growth':<22}"
          f" {max(base.revenue_growth - SPREAD_SIGMA*sg, -0.30):>10.2%}"
          f"  {base.revenue_growth:>10.2%}"
          f"  {min(base.revenue_growth + SPREAD_SIGMA*sg, 1.50):>10.2%}"
          f"  {sg:>7.2%}")
    print(f"  {'EBIT margin':<22}"
          f" {max(base.ebit_margin - SPREAD_SIGMA*sm, -0.20):>10.2%}"
          f"  {base.ebit_margin:>10.2%}"
          f"  {min(base.ebit_margin + SPREAD_SIGMA*sm, 0.75):>10.2%}"
          f"  {sm:>7.2%}")
    print(f"  {'Terminal g':<22}"
          f" {TERM_G_MIN:>10.2%}  {TERM_G_MODE:>10.2%}  {tg_hi:>10.2%}")
    sw  = result.sw
    print(f"  {'WACC':<22}"
          f" {max(w - SPREAD_SIGMA*sw, 0.04):>10.2%}"
          f"  {w:>10.2%}  {w + SPREAD_SIGMA*sw:>10.2%}"
          f"  {sw:>7.2%}")
    tm_sp = UNPROFITABLE_MARGIN_SPREAD if base.ebit_margin < 0 else PROFITABLE_MARGIN_SPREAD
    print(f"  {'Target margin':<22}"
          f" {max(base.target_margin - tm_sp, -0.10):>10.2%}"
          f"  {base.target_margin:>10.2%}"
          f"  {min(base.target_margin + tm_sp, 0.75):>10.2%}")
    if sc.get("diluted_shares", 0) / max(base.diluted_shares, 1) > 0.02:
        sc_abs = sc["diluted_shares"]
        print(f"  {'Diluted shares*':<22}"
              f" {(base.diluted_shares-sc_abs)/1e9:>10.3f}B"
              f"  {base.diluted_shares/1e9:>9.3f}B"
              f"  {(base.diluted_shares+sc_abs)/1e9:>9.3f}B  [promoted]")
    if sc.get("net_debt", 0) / max(abs(base.net_debt), 1e-9) > 0.02:
        nd_abs = sc["net_debt"]
        print(f"  {'Net debt*':<22}"
              f" {(base.net_debt-nd_abs)/1e9:>10.3f}B"
              f"  {base.net_debt/1e9:>9.3f}B"
              f"  {(base.net_debt+nd_abs)/1e9:>9.3f}B  [promoted]")
    if base.currency != 'USD':
        fxv = FX_ANNUAL_VOL.get(base.currency, FX_ANNUAL_VOL['_default'])
        print(f"  {'FX (' + base.currency + '/USD)':<22}"
              f"  {'GBM (lognormal)':>10}  {base.fx_rate:>10.5f}  {'σ=' + f'{fxv:.0%}':>10}")

    print(f"\n  Consistency check (zero-variance run, n=50)...")
    print(f"  Deterministic value()  = ${det:.4f}")
    print(f"  Zero-variance MC P50   = ${result.cc_p50:.4f}")
    print(f"  {'✓  PASS' if result.consistency_pass else '✗  FAIL — check simulation centring'}")

    sims  = result.sims
    n_v   = result.n_valid
    price = base.current_price_usd

    print(f"\n{'#'*60}")
    print(f"  MONTE CARLO RESULTS — {result.ticker}  ({n_v:,} valid simulations)")
    print(f"{'#'*60}")
    print(f"  Copula        : {result.copula_label}")
    print(f"  Current price : ${price:.2f}")
    print(f"  Base case     : ${det:.2f}")
    print(f"\n  Intrinsic value per share (USD):")
    print(f"  {'─'*36}")
    print(f"  P10   : ${result.p10:>9.2f}")
    print(f"  P25   : ${result.p25:>9.2f}")
    print(f"  P50   : ${result.p50:>9.2f}   ← median")
    print(f"  P75   : ${result.p75:>9.2f}")
    print(f"  P90   : ${result.p90:>9.2f}")
    print(f"  {'─'*36}")
    print(f"  Mean  : ${result.mean_val:>9.2f}")
    print(f"  Stdev : ${result.std_val:>9.2f}")
    print(f"  Min   : ${np.min(sims):>9.2f}   Max : ${np.max(sims):.2f}")
    print(f"\n  P(undervalued) : {result.pct_undervalued:.1f}%"
          f"   (draws where intrinsic > ${price:.2f})")

    path = _save_histogram(
        result.ticker, sims, price, n_v, result.copula_label,
        det, result.p10, result.p50, result.p90,
    )
    print(f"\n  Histogram → {path}")


def run_monte_carlo(ticker: str, n_sims: int = 10_000) -> None:
    print_valuation(run_valuation(ticker, n_sims))
