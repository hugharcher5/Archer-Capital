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
from .dcf       import Assumptions, value, value_and_ceiling_hits, detailed_value, TERMINAL_G, HIGH_GROWTH_THRESHOLD, MATURE_MARGIN_DEFAULT, G_CEILING_A, G_CEILING_B
from .result    import ValuationResult
from .reconcile import FIELD_TYPE, ALL_FIELDS


# ─────────────────────────── Editable constants ───────────────────────────────

SPREAD_SIGMA: float            = 3.0     # PERT half-width = N × historical σ
MC_RANDOM_SEED: int            = 42      # reproducible main simulation seed
HIGH_VOL_THRESHOLD: float      = 0.20    # σ_eff revenue growth — Student-t copula trigger
EBIT_VOL_THRESHOLD: float      = 0.05    # σ_eff EBIT margin (5pp) — Student-t copula trigger
FCF_VOL_THRESHOLD: float       = 0.08    # σ FCF/revenue (8pp) — Student-t copula trigger
SIGMA_CROSS_THRESHOLD: float   = 0.15    # max rate σ_cross (15pp) — Student-t copula trigger
STUDENT_T_DF: int              = 5       # degrees of freedom for Student-t copula
WACC_SPREAD_ABS: float         = 0.015   # WACC PERT half-width (absolute, e.g. ±1.5%)
TERM_G_MIN: float              = 0.015   # 1.5% floor on terminal g
TERM_G_MAX_ABS: float          = 0.040   # 4.0% ceiling (also bounded by WACC − 1%)
TERM_G_MODE: float             = TERMINAL_G          # 2.5%
WACC_TG_GAP: float             = 0.010   # minimum enforced gap: WACC − terminal_g
PROFITABLE_MARGIN_SPREAD: float   = 0.08  # target_margin PERT half-width for profitable cos
UNPROFITABLE_MARGIN_SPREAD: float = 0.15  # wider spread for currently-unprofitable cos
TARGET_MARGIN_SPREAD_MIN: float = 0.03   # 3pp floor: prevents degenerate zero-width PERT
TARGET_MARGIN_SPREAD_MAX: float = 0.20   # 20pp cap: avoids absurd ranges for tiny samples

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


def _terminal_g_params(revenue_growth: float, wacc: float) -> tuple[float, float, float]:
    """Company-specific terminal growth (tg_mode, tg_floor, tg_cap).

    avg_g = mean of linspace(revenue_growth, TERM_G_MODE, N)
          = (revenue_growth + TERM_G_MODE) / 2  [analytically, any N]

    tg_floor = max(-2%, 0.3 × avg_g)         — allow decline, not below 30% of avg
    tg_mode  = min(avg_g, 2.5%)              — anchor at forecast avg, cap at long-run GDP
    tg_cap   = min(4%, WACC-1%, avg_g+1pp)  — can't exceed GDP ceiling or WACC gap

    Safety clamps guarantee floor < mode < cap so PERT construction never fails.
    """
    avg_g    = (revenue_growth + TERM_G_MODE) / 2.0
    tg_mode  = min(avg_g, TERM_G_MODE)
    tg_cap   = min(TERM_G_MAX_ABS, wacc - WACC_TG_GAP, avg_g + 0.01)
    tg_floor = max(-0.02, 0.30 * avg_g)
    tg_floor = min(tg_floor, tg_mode - 1e-4)   # safety: floor must sit below mode
    tg_cap   = max(tg_cap,   tg_mode + 1e-4)   # safety: cap must sit above mode
    return float(tg_mode), float(tg_floor), float(tg_cap)


_TERM_G_SIGMA_CACHE: float | None = None   # computed once per process


def _get_term_g_sigma() -> tuple[float, bool]:
    """
    Return (sigma, used_fallback).

    Terminal growth is a convergence-to-long-run-GDP assumption, not a
    company-specific forecast, so its uncertainty is derived from MACRO history:
    the historical volatility of US annual nominal GDP growth (FRED series 'GDPA').
    The same sigma is therefore used for every company by design.

    Optional refinement: sigma is modestly widened for companies whose revenue
    growth has been highly volatile, reflecting greater uncertainty about when
    (and at what rate) they reach steady state.  The adjustment is bounded so it
    cannot dominate the macro base.

    Falls back to 2pp (consistent with observed US nominal GDP volatility) when
    FRED data is unavailable.
    """
    global _TERM_G_SIGMA_CACHE
    if _TERM_G_SIGMA_CACHE is not None:
        return _TERM_G_SIGMA_CACHE, False

    _FALLBACK = 0.020   # ~2pp — conservative estimate of US nominal GDP growth volatility

    api_key = os.getenv('FRED_API_KEY', '').strip()
    if not api_key or api_key == 'your_api_key_here':
        _TERM_G_SIGMA_CACHE = _FALLBACK
        return _FALLBACK, True

    try:
        from fredapi import Fred
        # Annual nominal GDP levels (billions USD, current prices, SAAR)
        gdp = Fred(api_key=api_key).get_series('GDPA').dropna()
        g   = gdp.pct_change().dropna()
        if len(g) >= 10:
            sigma = float(g.std())
            _TERM_G_SIGMA_CACHE = sigma
            return sigma, False
    except Exception as e:
        print(f"  [MC] FRED GDPA fetch failed ({e}); terminal-g sigma using fallback.")

    _TERM_G_SIGMA_CACHE = _FALLBACK
    return _FALLBACK, True


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
        terminal_g=_terminal_g_params(drivers.revenue_growth, wacc_r.wacc)[0],
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
    stm: float = 0.08,   # σ_eff for target-margin PERT half-width (company hist; 8pp fallback)
    stg: float = 0.020,  # σ for terminal-g PERT; macro-derived (GDP history), same for all companies
    shares_cross: float = 0.0,  # absolute σ_cross for diluted_shares (0 = not promoted)
    shares_promoted: bool = False,
    nd_cross: float = 0.0,      # absolute σ_cross for net_debt (0 = not promoted)
    nd_promoted: bool = False,
    _collect: bool = False,     # when True return (valid, extras_dict) for transparency payload
):
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
        tg_lo = tg_hi = base.terminal_g
        wa_lo = wa_hi = base.wacc
        tm_lo = tm_hi = base.target_margin
    else:
        rg_lo = max(base.revenue_growth - spread_sigma * sg, -0.30)
        rg_hi = min(base.revenue_growth + spread_sigma * sg,  1.50)
        em_lo = max(base.ebit_margin    - spread_sigma * sm, -0.20)
        em_hi = min(base.ebit_margin    + spread_sigma * sm,  0.75)
        # terminal_g PERT: spread from macro GDP history, bounded by company-specific floor/cap.
        _, tg_floor_p, tg_cap_p = _terminal_g_params(base.revenue_growth, w)
        tg_half = spread_sigma * stg
        tg_lo   = max(base.terminal_g - tg_half, tg_floor_p)
        tg_hi   = max(tg_lo + 1e-6, min(base.terminal_g + tg_half, tg_cap_p, w - WACC_TG_GAP))
        wa_lo = max(w - spread_sigma * sw, 0.04)
        wa_hi = w + spread_sigma * sw
        # target_margin PERT: spread driven by company historical EBIT-margin σ.
        # Clamped so small samples don't produce zero or absurdly wide ranges.
        tm_half = float(np.clip(spread_sigma * stm, TARGET_MARGIN_SPREAD_MIN, TARGET_MARGIN_SPREAD_MAX))
        tm_lo = max(base.target_margin - tm_half, -0.10)
        tm_hi = min(base.target_margin + tm_half,  0.75)

    # Correlated uniform draws
    if copula == 'student-t':
        U = _student_t_uniforms(n, corr, STUDENT_T_DF, rng)
    else:
        U = _gaussian_uniforms(n, corr, rng)

    # Map uniforms → PERT marginals
    rg_s = _pert_ppf(U[:, 0], rg_lo, base.revenue_growth, rg_hi)
    em_s = _pert_ppf(U[:, 1], em_lo, base.ebit_margin,    em_hi)
    tg_s = _pert_ppf(U[:, 2], tg_lo, base.terminal_g,      tg_hi)
    wa_s = _pert_ppf(U[:, 3], wa_lo, base.wacc,           wa_hi)
    tm_s = _pert_ppf(U[:, 4], tm_lo, base.target_margin,  tm_hi)

    # Hard clamp: enforce WACC − terminal_g ≥ WACC_TG_GAP at all times
    tg_s = np.minimum(tg_s, wa_s - WACC_TG_GAP)

    # FX GBM path — per-year rates for non-USD, independent of fundamental draws.
    # fx_paths shape (n, T): rate_t = rate_0 × exp(cumsum[-0.5σ² + σZ_s] for s=1..t)
    # USD companies: fx_paths stays None; a.fx_path is never set → byte-identical output.
    T = base.forecast_years
    if base.currency != 'USD' and spread_sigma > 1e-9:
        vol      = FX_ANNUAL_VOL.get(base.currency, FX_ANNUAL_VOL['_default'])
        Z_fx     = rng.standard_normal((n, T))                           # (n, T) independent
        log_inc  = -0.5 * vol**2 + vol * Z_fx                           # annual log-increments
        fx_paths = base.fx_rate * np.exp(np.cumsum(log_inc, axis=1))    # (n, T) cumulative path
    else:
        fx_paths = None   # USD or zero-spread: leave a.fx_path=None, use a.fx_rate

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
    results      = np.empty(n)
    ceiling_hits = np.zeros(n, dtype=int)   # forecast years ceiling was binding, per draw
    n_except     = 0
    a = copy.copy(base)

    for i in range(n):
        a.revenue_growth = float(rg_s[i])
        a.ebit_margin    = float(em_s[i])
        a.target_margin  = float(tm_s[i])
        a.terminal_g     = float(tg_s[i])
        a.wacc           = float(wa_s[i])
        if fx_paths is not None:
            a.fx_path = fx_paths[i]   # per-year path; a.fx_rate stays as current spot
        a.diluted_shares = float(sh_s[i])
        a.net_debt       = float(nd_s[i])
        try:
            if _collect:
                results[i], ceiling_hits[i] = value_and_ceiling_hits(a)
            else:
                results[i] = value(a)
        except Exception:
            results[i]      = float('nan')
            ceiling_hits[i] = 0
            n_except        += 1

    valid_mask = ~np.isnan(results)
    valid      = results[valid_mask]
    n_nan      = n - len(valid)
    if n_nan:
        # n_nan covers both exception-thrown draws and draws where value() returned
        # NaN (e.g. from invalid PERT parameters producing NaN sampled inputs).
        print(f"  ⚠  {n_nan}/{n} draws discarded ({n_except} exceptions, "
              f"{n_nan - n_except} silent NaN); excluded from results.")

    if not _collect:
        return valid

    # Ceiling stats across valid draws
    ch_valid = ceiling_hits[valid_mask]
    ceiling_stats = {
        'pct_paths_bound': float(np.mean(ch_valid > 0)) * 100.0,
        'avg_years_bound': float(np.mean(ch_valid)),
    }

    # Transparency extras: sampled inputs + PERT bounds + FX paths (filtered to valid draws)
    samples = {
        'revenue_growth': rg_s[valid_mask],
        'ebit_margin':    em_s[valid_mask],
        'target_margin':  tm_s[valid_mask],
        'terminal_g':     tg_s[valid_mask],
        'wacc':           wa_s[valid_mask],
    }
    if shares_promoted:
        samples['diluted_shares'] = sh_s[valid_mask]
    if nd_promoted:
        samples['net_debt'] = nd_s[valid_mask]

    pert_params = {
        'revenue_growth': {'lo': rg_lo, 'mode': base.revenue_growth, 'hi': rg_hi},
        'ebit_margin':    {'lo': em_lo, 'mode': base.ebit_margin,    'hi': em_hi},
        'target_margin':  {'lo': tm_lo, 'mode': base.target_margin,  'hi': tm_hi},
        'terminal_g':     {'lo': tg_lo, 'mode': base.terminal_g,     'hi': tg_hi},
        'wacc':           {'lo': wa_lo, 'mode': base.wacc,           'hi': wa_hi},
    }

    if fx_paths is not None:
        fx_extra = {
            'per_year': [
                {
                    'year': t + 1,
                    'p10': float(np.percentile(fx_paths[:, t], 10)),
                    'p50': float(np.percentile(fx_paths[:, t], 50)),
                    'p90': float(np.percentile(fx_paths[:, t], 90)),
                }
                for t in range(T)
            ],
            'sample_paths': fx_paths[:min(100, n)],
        }
    else:
        fx_extra = None

    return valid, {'samples': samples, 'pert_params': pert_params, 'fx_extra': fx_extra,
                   'ceiling_stats': ceiling_stats}


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
    sw = wacc_r.std_wacc   # historical WACC σ (or fallback) — no σ_cross (WACC inputs are market-derived)
    base   = _build_base(raw, drvrs, wacc_r)
    dcf_r  = detailed_value(base)

    # ── DCF applicability gate ────────────────────────────────────────────────
    # HARD GATE: terminal-year FCFF <= 0 → Gordon Growth TV is degenerate.
    # Secondary diagnostics are recorded for transparency but do not gate alone.
    _terminal_fcff     = float(dcf_r.forecast['FCFF'].iloc[-1])
    _current_ebit      = base.ebit_margin * base.start_revenue
    _frac_neg_fcff     = float((dcf_r.forecast['FCFF'] < 0).mean())
    _bc_intrinsic      = dcf_r.value_per_share_usd
    # Gate trips on either condition: terminal FCFF degenerate OR base-case
    # intrinsic negative (the latter catches companies where years of negative
    # FCFFs overwhelm a positive terminal value, e.g. IONQ-style pre-profit).
    dcf_applicable     = _terminal_fcff > 0 and _bc_intrinsic > 0

    _gate_triggers: list[str] = []
    if _terminal_fcff <= 0:
        _gate_triggers.append('terminal_fcff_non_positive')
    if _current_ebit < 0:
        _gate_triggers.append('current_ebit_negative')
    if _frac_neg_fcff > 0.0:
        n_neg = round(_frac_neg_fcff * base.forecast_years)
        _gate_triggers.append(f'forecast_fcff_negative_in_{n_neg}_of_{base.forecast_years}_years')
    if _bc_intrinsic < 0:
        _gate_triggers.append('base_case_intrinsic_negative')

    sg_hist = drvrs.std_revenue_growth
    sm_hist = drvrs.std_ebit_margin

    # Effective σ: combine historical spread with cross-source uncertainty in quadrature.
    # σ_eff can only grow, never shrink — adding a zero σ_cross leaves σ_hist unchanged.
    sg = math.sqrt(sg_hist**2 + sigma_cross.get("revenue_growth", 0.0)**2)
    sm = math.sqrt(sm_hist**2 + sigma_cross.get("ebit_margin",    0.0)**2)

    # target_margin uses the same σ_cross source as ebit_margin (derived from same data).
    stm = math.sqrt(drvrs.std_target_margin**2 + sigma_cross.get("ebit_margin", 0.0)**2)

    # terminal_g is anchored at avg_g = (revenue_growth + TERM_G_MODE) / 2, so revenue_growth
    # uncertainty propagates at half-weight: σ_cross_tg = 0.5 × σ_cross["revenue_growth"].
    stg_macro, tg_fallback = _get_term_g_sigma()
    if tg_fallback:
        print("  [MC] Terminal-g spread: FRED unavailable — using fixed 2pp fallback.")
    stg = math.sqrt(stg_macro**2 + (0.5 * sigma_cross.get("revenue_growth", 0.0))**2)

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
        recon_sigma["target_margin"] = {
            "sigma_hist":  drvrs.std_target_margin,
            "sigma_cross": sigma_cross.get("ebit_margin", 0.0),
            "sigma_eff":   stm,
        }
        recon_sigma["terminal_g"] = {
            "sigma_hist":  stg_macro,
            "sigma_cross": 0.5 * sigma_cross.get("revenue_growth", 0.0),
            "sigma_eff":   stg,
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
            sw=sw, stm=stm, stg=stg,
            dcf_applicable=dcf_applicable,
        )

    corr = _ensure_psd(CORR.copy())

    # Student-t copula triggers (4 fundamental): any one is sufficient.
    # Stock-price volatility is deliberately excluded — it is market-derived, not an
    # operating fundamental, and would conflate valuation uncertainty with market pricing.
    _max_sc = max(sigma_cross.get('revenue_growth', 0.0), sigma_cross.get('ebit_margin', 0.0))
    _triggers: list[str] = []
    if sg                > HIGH_VOL_THRESHOLD:    _triggers.append(f"σ_growth={sg:.1%} > {HIGH_VOL_THRESHOLD:.0%}")
    if sm                > EBIT_VOL_THRESHOLD:    _triggers.append(f"σ_margin={sm:.1%} > {EBIT_VOL_THRESHOLD:.0%}")
    if drvrs.std_fcf_pct > FCF_VOL_THRESHOLD:     _triggers.append(f"σ_FCF={drvrs.std_fcf_pct:.1%} > {FCF_VOL_THRESHOLD:.0%}")
    if _max_sc           > SIGMA_CROSS_THRESHOLD: _triggers.append(f"max_σ_cross={_max_sc:.1%} > {SIGMA_CROSS_THRESHOLD:.0%}")

    use_t      = bool(_triggers)
    copula_key = 'student-t' if use_t else 'gaussian'
    copula_lbl = f'Student-t (df={STUDENT_T_DF})' if use_t else 'Gaussian'
    if use_t:
        print(f"  [MC] Student-t copula: {'; '.join(_triggers)}")

    # Consistency check: zero spread_sigma pins all variables to mode.
    # Pass sg/sm (with any σ_cross baked in) but spread_sigma=0 overrides bounds to
    # point masses, so σ_eff doesn't matter — output must equal deterministic value().
    cc_sims = _run_sims(base, sg, sm, corr, 50, 0.0,
                        np.random.default_rng(0), copula_key, sw=sw, stm=stm, stg=stg)
    cc_p50  = float(np.median(cc_sims))
    cc_ok   = abs(cc_p50 - dcf_r.value_per_share_usd) < 0.01

    # Full simulation — collect samples and PERT bounds for the transparency payload.
    sims, _extras = _run_sims(
        base, sg, sm, corr, n_sims, SPREAD_SIGMA,
        np.random.default_rng(seed=MC_RANDOM_SEED), copula_key,
        sw=sw, stm=stm, stg=stg,
        shares_cross=sc_abs,   shares_promoted=shares_promoted,
        nd_cross=nd_abs,       nd_promoted=nd_promoted,
        _collect=True,
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

    # ── Transparency payload ──────────────────────────────────────────────────
    _, tg_floor_main, tg_cap_main = _terminal_g_params(base.revenue_growth, base.wacc)
    _pp = _extras['pert_params']

    _sigma_meta = {
        'revenue_growth': {
            'sigma_hist':  sg_hist,
            'sigma_cross': sigma_cross.get('revenue_growth', 0.0),
            'sigma_eff':   sg,
            'clamp_floor': -0.30,
            'clamp_cap':    1.50,
            'rationale_key': 'historical_mean_growth',
        },
        'ebit_margin': {
            'sigma_hist':  sm_hist,
            'sigma_cross': sigma_cross.get('ebit_margin', 0.0),
            'sigma_eff':   sm,
            'clamp_floor': -0.20,
            'clamp_cap':    0.75,
            'rationale_key': 'historical_mean_ebit_margin',
        },
        'target_margin': {
            'sigma_hist':  drvrs.std_target_margin,
            'sigma_cross': sigma_cross.get('ebit_margin', 0.0),
            'sigma_eff':   stm,
            'clamp_floor': -0.10,
            'clamp_cap':    0.75,
            'rationale_key': 'best_historical_ebit_margin',
        },
        'terminal_g': {
            'sigma_hist':  stg_macro,
            'sigma_cross': 0.5 * sigma_cross.get('revenue_growth', 0.0),
            'sigma_eff':   stg,
            'clamp_floor': tg_floor_main,
            'clamp_cap':   tg_cap_main,
            'rationale_key': 'avg_growth_anchored_to_gdp',
        },
        'wacc': {
            'sigma_hist':  sw,
            'sigma_cross': 0.0,
            'sigma_eff':   sw,
            'clamp_floor': 0.04,
            'clamp_cap':   float('inf'),
            'rationale_key': 'capm_synthetic_rating_blume_beta',
        },
    }

    def _clamp_binding(lo: float, hi: float, raw_lo: float, raw_hi: float) -> str:
        floor_hit = lo > raw_lo + 1e-9
        cap_hit   = hi < raw_hi - 1e-9
        if floor_hit and cap_hit:
            return 'both'
        return 'floor' if floor_hit else ('cap' if cap_hit else 'none')

    distribution_params: dict = {}
    for var, info in _sigma_meta.items():
        lo    = _pp[var]['lo']
        hi    = _pp[var]['hi']
        mode  = _pp[var]['mode']
        raw_lo = mode - SPREAD_SIGMA * info['sigma_eff']
        raw_hi = mode + SPREAD_SIGMA * info['sigma_eff']
        distribution_params[var] = {
            'pert_min':      lo,
            'pert_mode':     mode,
            'pert_max':      hi,
            'sigma_hist':    info['sigma_hist'],
            'sigma_cross':   info['sigma_cross'],
            'sigma_eff':     info['sigma_eff'],
            'clamp_floor':   info['clamp_floor'],
            'clamp_cap':     info['clamp_cap'],
            'clamp_binding': _clamp_binding(lo, hi, raw_lo, raw_hi),
            'rationale_key': info['rationale_key'],
        }

    # Realised correlation from the 5 primary sampled inputs
    _smp = _extras['samples']
    _corr_mat = np.row_stack([
        _smp['revenue_growth'],
        _smp['ebit_margin'],
        _smp['terminal_g'],
        _smp['wacc'],
        _smp['target_margin'],
    ])
    corr_realized = np.corrcoef(_corr_mat) if _corr_mat.shape[1] >= 2 else np.eye(5)

    # FX info
    _fx_extra = _extras.get('fx_extra')
    fx_vol = FX_ANNUAL_VOL.get(base.currency, FX_ANNUAL_VOL['_default'])
    fx_info = {
        'is_foreign':   base.currency != 'USD',
        'sigma':        fx_vol,
        'per_year':     _fx_extra['per_year']    if _fx_extra else [],
        'sample_paths': _fx_extra['sample_paths'] if _fx_extra else np.array([]),
    }

    # Copula info including all 4 fundamental trigger specs
    _trigger_specs = [
        {
            'name':         'revenue_growth_vol',
            'threshold':    HIGH_VOL_THRESHOLD,
            'actual_value': sg,
            'fired':        sg > HIGH_VOL_THRESHOLD,
        },
        {
            'name':         'ebit_margin_vol',
            'threshold':    EBIT_VOL_THRESHOLD,
            'actual_value': sm,
            'fired':        sm > EBIT_VOL_THRESHOLD,
        },
        {
            'name':         'fcf_vol',
            'threshold':    FCF_VOL_THRESHOLD,
            'actual_value': drvrs.std_fcf_pct,
            'fired':        drvrs.std_fcf_pct > FCF_VOL_THRESHOLD,
        },
        {
            'name':         'sigma_cross_max',
            'threshold':    SIGMA_CROSS_THRESHOLD,
            'actual_value': _max_sc,
            'fired':        _max_sc > SIGMA_CROSS_THRESHOLD,
        },
    ]
    copula_info = {
        'type':     copula_key,
        'df':       STUDENT_T_DF if use_t else None,
        'triggers': _trigger_specs,
        'note':     (
            'Stock-price volatility deliberately excluded — market-derived input, not an '
            'operating fundamental; would conflate valuation uncertainty with market pricing.'
        ),
    }

    _cs = _extras.get('ceiling_stats', {})
    transparency = {
        'samples':              _smp,
        'distribution_params':  distribution_params,
        'copula_info':          copula_info,
        'correlation_target':   corr.copy(),
        'correlation_realized': corr_realized,
        'fx_info':              fx_info,
        'dcf_applicability': {
            'dcf_applicable':     dcf_applicable,
            'terminal_fcff':      _terminal_fcff,
            'current_ebit':       _current_ebit,
            'frac_negative_fcff': _frac_neg_fcff,
            'base_case_intrinsic': _bc_intrinsic,
            'triggers':           _gate_triggers,
        },
        'g_ceiling': {
            'A':               G_CEILING_A,
            'b':               G_CEILING_B,
            'pct_paths_bound': _cs.get('pct_paths_bound', 0.0),
            'avg_years_bound': _cs.get('avg_years_bound', 0.0),
        },
        'validation': {
            'zero_width_test_passed': cc_ok,
            'mc_p50_at_zero_width':   cc_p50,
            'deterministic_base':     dcf_r.value_per_share_usd,
            'abs_diff':               abs(cc_p50 - dcf_r.value_per_share_usd),
        },
        'n_sims':      n_sims,
        'random_seed': MC_RANDOM_SEED,
    }

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
        sw=sw, stm=stm, stg=stg,
        transparency=transparency,
        dcf_applicable=dcf_applicable,
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
          f"   (σ_growth={sg:.2%}  σ_margin={sm:.2%}  σ_FCF={drvrs.std_fcf_pct:.2%})"
          + cross_note)

    stg          = result.stg
    spread_sigma = SPREAD_SIGMA
    _, tg_floor_p, tg_cap_p = _terminal_g_params(base.revenue_growth, w)
    tg_h  = spread_sigma * stg
    tg_lo = max(base.terminal_g - tg_h, tg_floor_p)
    tg_hi = max(tg_lo + 1e-6, min(base.terminal_g + tg_h, tg_cap_p, w - WACC_TG_GAP))
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
          f" {tg_lo:>10.2%}  {base.terminal_g:>10.2%}  {tg_hi:>10.2%}"
          f"  {stg:>7.2%}")
    sw  = result.sw
    print(f"  {'WACC':<22}"
          f" {max(w - SPREAD_SIGMA*sw, 0.04):>10.2%}"
          f"  {w:>10.2%}  {w + SPREAD_SIGMA*sw:>10.2%}"
          f"  {sw:>7.2%}")
    stm  = result.stm
    tm_h = float(np.clip(SPREAD_SIGMA * stm, TARGET_MARGIN_SPREAD_MIN, TARGET_MARGIN_SPREAD_MAX))
    print(f"  {'Target margin':<22}"
          f" {max(base.target_margin - tm_h, -0.10):>10.2%}"
          f"  {base.target_margin:>10.2%}"
          f"  {min(base.target_margin + tm_h, 0.75):>10.2%}"
          f"  {stm:>7.2%}")
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
              f"  {'GBM path/yr':>10}  {base.fx_rate:>10.5f}  {'σ=' + f'{fxv:.0%}':>10}")

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
