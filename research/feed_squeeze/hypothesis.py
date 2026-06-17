"""
Hypothesis tests for the feed-cost-squeeze signal.

Test A  ─ Does ONI lead fishmeal price?
Test B  ─ Does fishmeal predict salmon-farmer residual returns
          (returns net of the salmon spot-price channel)?
Test C  ─ Simple signal backtest (only run when A or B show something).

All series passed in are already point-in-time (publication lags baked in by
build_panel.py).  Within each test, additional lags are pure statistical
choices, not further publication-lag adjustments.
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    ENSO_THRESHOLD, ENSO_MIN_MONTHS,
    TEST_A_MAX_LAG, TEST_B_MAX_LAG,
    SIGNAL_FM_LOOKBACK, SIGNAL_ONI_THRESHOLD, SIGNAL_COST_BPS,
    FARMER_TICKERS,
)


# ── El Niño cycle counter ─────────────────────────────────────────────────────

def count_enso_cycles(oni_raw: pd.Series) -> "list[tuple]":
    """
    Count independent El Niño episodes in the RAW (un-lagged) ONI series.

    Uses the NOAA standard definition: ONI ≥ +0.5 °C for ≥ 5 consecutive
    overlapping 3-month periods.

    Pass in the raw ONI so we count calendar cycles, not the lagged signal.

    Returns a list of (start_date, end_date) tuples.
    """
    above    = (oni_raw >= ENSO_THRESHOLD).astype(int)
    episodes = []
    in_ep    = False
    ep_start = None
    run_len  = 0

    for date, val in above.items():
        if val:
            if not in_ep:
                ep_start = date
                in_ep    = True
            run_len += 1
        else:
            if in_ep and run_len >= ENSO_MIN_MONTHS:
                episodes.append((ep_start, date - pd.offsets.MonthEnd(1)))
            in_ep   = False
            run_len = 0

    if in_ep and run_len >= ENSO_MIN_MONTHS:
        episodes.append((ep_start, oni_raw.index[-1]))

    return episodes


# ── Test A ───────────────────────────────────────────────────────────────────

def test_a_oni_vs_fishmeal(
    panel: pd.DataFrame,
    max_lag: int = TEST_A_MAX_LAG,
) -> pd.DataFrame:
    """
    Test A: does ONI level predict forward fishmeal price changes?

    For lag k we compute Pearson r between ONI_pub_t and d_fishmeal_{t+k}:
    "does today's ONI predict fishmeal k months from now?"

    Because oni_pub already carries a 1-month publication lag, a table row
    at lag=k corresponds to a real-world information lag of 1+k months.

    ALL lags 0..max_lag are reported — do not cherry-pick.
    """
    if "d_fishmeal" not in panel.columns or panel["d_fishmeal"].notna().sum() < 20:
        warnings.warn("[Test A] Fishmeal data missing or too short.")
        return pd.DataFrame()

    oni = panel["oni_pub"]
    dfm = panel["d_fishmeal"]
    rows = []

    for lag in range(max_lag + 1):
        future_fm = dfm.shift(-lag)          # fishmeal k months ahead of ONI
        mask      = oni.notna() & future_fm.notna()
        n         = int(mask.sum())
        if n < 10:
            rows.append({"lag_k": lag, "pearson_r": np.nan, "p_value": np.nan, "n": n})
            continue
        r, p = stats.pearsonr(oni[mask].values, future_fm[mask].values)
        rows.append({"lag_k": lag, "pearson_r": round(r, 4), "p_value": round(p, 4), "n": n})

    return pd.DataFrame(rows)


# ── Test B ───────────────────────────────────────────────────────────────────

def test_b_fishmeal_vs_residual(
    panel: pd.DataFrame,
    max_lag: int = TEST_B_MAX_LAG,
) -> "tuple[pd.DataFrame, pd.Series, pd.DataFrame, float, float]":
    """
    Test B: does lagged fishmeal price change predict salmon-farmer returns
    after stripping out the salmon spot-price channel?

    Stage 1
    -------
    Regress basket_ret on contemporaneous salmon_ret (salmon spot return).
    The residual ε is the margin/feed/operational channel.

    Stage 2
    -------
    For lags k = 0..max_lag, regress ε_t on d_fishmeal_{t-k}.
    d_fishmeal already carries a 1-month pub lag, so the real-world
    information gap is 1+k months.

    ALL lags are reported.  The most common failure mode in this test is
    selecting the best-looking lag and presenting only that.

    Returns
    -------
    results_basket  : lag table for the equal-weighted basket
    resid           : the basket residual series (used in Test C)
    results_ind     : long-form lag table for individual tickers
    beta_salmon     : β from the stage-1 regression
    r2_stage1       : R² from the stage-1 regression
    """
    basket = panel.get("basket_ret", pd.Series(dtype=float))
    salmon = panel.get("salmon_ret", pd.Series(dtype=float))
    dfm    = panel.get("d_fishmeal", pd.Series(dtype=float))

    # ── Stage 1: residualise on salmon spot ──────────────────────────────────
    if salmon is not None and salmon.notna().sum() > 10:
        mask1 = basket.notna() & salmon.notna()
        sl1, ic1, r1, _, se1 = stats.linregress(salmon[mask1].values, basket[mask1].values)
        resid   = basket - (ic1 + sl1 * salmon)
        r2_s1   = r1 ** 2
        beta    = sl1
    else:
        warnings.warn("[Test B] Salmon price data missing — using raw basket returns.")
        resid = basket.copy()
        r2_s1 = np.nan
        beta  = np.nan

    resid.name = "basket_resid"

    # ── Stage 2: lag regressions ─────────────────────────────────────────────
    def _lag_table(y: pd.Series, label: str) -> list:
        rows = []
        for lag in range(max_lag + 1):
            x    = dfm.shift(lag)            # fishmeal observed lag months ago
            mask = y.notna() & x.notna()
            n    = int(mask.sum())
            if n < 10:
                rows.append({"ticker": label, "lag_k": lag,
                             "coef": np.nan, "t_stat": np.nan, "r2": np.nan, "n": n})
                continue
            sl, ic, rv, _, se = stats.linregress(x[mask].values, y[mask].values)
            rows.append({
                "ticker": label, "lag_k": lag,
                "coef":   round(sl, 6),
                "t_stat": round(sl / se, 3) if se > 0 else np.nan,
                "r2":     round(rv ** 2, 5),
                "n":      n,
            })
        return rows

    basket_rows = _lag_table(resid, "basket")

    # Individual tickers (each residualised on salmon separately)
    ind_rows = []
    ind_cols = [c for c in panel.columns
                if c.endswith("_ret")
                and c not in ("basket_ret", "salmon_ret", "BMK_ret")]
    for col in ind_cols:
        label = col.replace("_ret", "")
        raw_y = panel[col]
        if raw_y.notna().sum() < 20:
            continue
        if salmon is not None and salmon.notna().sum() > 10:
            mask_i = raw_y.notna() & salmon.notna()
            if mask_i.sum() < 10:
                continue
            sl_i, ic_i, _, _, _ = stats.linregress(salmon[mask_i].values, raw_y[mask_i].values)
            ind_resid = raw_y - (ic_i + sl_i * salmon)
        else:
            ind_resid = raw_y.copy()
        ind_rows.extend(_lag_table(ind_resid, label))

    results_basket = pd.DataFrame(basket_rows)
    results_ind    = pd.DataFrame(ind_rows) if ind_rows else pd.DataFrame()

    return results_basket, resid, results_ind, beta, r2_s1


# ── Test C ───────────────────────────────────────────────────────────────────

def test_c_backtest(
    panel: pd.DataFrame,
    resid: pd.Series,
) -> "dict | None":
    """
    Test C: simple signal backtest on the residual return.

    Signal (all inputs are already pub-lagged in the panel):
      ON  when  12m fishmeal log-change > 0  AND  ONI ≥ SIGNAL_ONI_THRESHOLD
      OFF otherwise

    Trade: signal ON → short basket residual next month (entry at T+1 close).
           signal OFF → flat.

    Cost: SIGNAL_COST_BPS bps round-trip applied on every transition.
    This is a PLACEHOLDER — mark any results with the cost assumption.

    Returns a stats dict, or None if there's insufficient data or signal
    observations to evaluate.
    """
    fm_pub  = panel["fishmeal_pub"] if "fishmeal_pub" in panel.columns else None
    oni_pub = panel["oni_pub"]      if "oni_pub"      in panel.columns else None

    if fm_pub is None or fm_pub.notna().sum() < SIGNAL_FM_LOOKBACK + 12:
        warnings.warn("[Test C] Insufficient fishmeal history — skipping backtest.")
        return None

    # 12-month log change of the already-pub-lagged fishmeal level
    fm_12m = np.log(fm_pub / fm_pub.shift(SIGNAL_FM_LOOKBACK))

    if oni_pub is not None and oni_pub.notna().sum() > 10:
        signal = (fm_12m > 0) & (oni_pub >= SIGNAL_ONI_THRESHOLD)
    else:
        warnings.warn("[Test C] ONI data unavailable — using fishmeal-only signal.")
        signal = fm_12m > 0

    # Enter next month: shift signal forward 1 so we can't trade on the same
    # month we observe the signal (no same-month look-ahead).
    signal_lagged = signal.shift(1).reindex(resid.index)
    aligned_resid = resid.reindex(signal_lagged.index)

    # Strategy return: short (−1) when signal on, flat (0) otherwise
    strat = pd.Series(
        np.where(signal_lagged.fillna(False), -aligned_resid, 0.0),
        index=aligned_resid.index,
    )

    # Round-trip cost applied on each signal transition
    transitions = signal_lagged.fillna(False).astype(int).diff().abs().fillna(0)
    strat -= transitions * (SIGNAL_COST_BPS / 10_000)

    valid = strat.notna() & aligned_resid.notna() & signal_lagged.notna()
    strat = strat[valid]
    sig_v = signal_lagged[valid].astype(bool)
    res_v = aligned_resid[valid]

    if strat.empty or sig_v.sum() < 5:
        warnings.warn("[Test C] Too few signal-on months to evaluate.")
        return None

    on_ret  = strat[sig_v]
    off_ret = strat[~sig_v]

    sharpe = strat.mean() / strat.std() * np.sqrt(12) if strat.std() > 0 else np.nan

    # Count distinct signal episodes (rising edges of the lagged signal)
    n_episodes = int((signal_lagged.fillna(False).astype(int).diff() > 0).sum())

    return {
        "mean_monthly_ret_signal_on":  round(float(on_ret.mean()),   5),
        "mean_monthly_ret_signal_off": round(float(off_ret.mean()),  5),
        "hit_rate_signal_on":          round(float((on_ret > 0).mean()), 3),
        "annualised_sharpe":           round(float(sharpe),          3),
        "n_signal_on_months":          int(sig_v.sum()),
        "n_total_months":              int(valid.sum()),
        "n_signal_episodes":           n_episodes,
        "cost_assumption":             f"{SIGNAL_COST_BPS} bps round-trip (PLACEHOLDER)",
    }
