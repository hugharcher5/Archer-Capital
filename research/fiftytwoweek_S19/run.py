"""
S19: 52-Week High Anchoring (George & Hwang 2004)
====================================================
New independent trial. Genuinely distinct mechanism from S1/S4 (IVOL/MAX):
anchoring to a slow-moving REFERENCE PRICE (the 52-week high), not extreme
recent-return behavior. Requires only Tiingo price data + the already-cached,
shares-outstanding-only SEC facts used for PIT market cap (same minimal
dependency as S9/S1/S4) -- no GP/A-style financial-statement facts, no new
data sourcing, avoiding the feasibility risk that stopped S14 / constrained S18.

HYPOTHESIS: investors anchor on a stock's 52-week high as a psychological
reference point / perceived ceiling. Stocks trading NEAR their 52-week high
see good news underreacted to (reluctance to bid "through" the anchor),
producing subsequent positive drift as the bias corrects. The original
finding: nearness-to-52-week-high predicts returns BETTER than raw momentum,
and the two effects are largely independent -- tested explicitly below, not
assumed.

SIGNAL: nearness_t = adjClose_t / max(adjClose over the trailing 252 trading
days ending at/before t). Computed using only data available as of t (the
window ends at t, no look-ahead). Long top quintile (nearest to 52w high),
short bottom quintile (furthest below). MIN_52W_DAYS=200 trading days of
history required (same order as S9's 180-day momentum threshold, slightly
higher since the ratio specifically wants near-full-year coverage).

CONSTRUCTION: reuses momentum_jt (S9)'s universe builder, PIT market-cap
(SEC shares-outstanding facts, already cached, no new fetches), price/adjClose
caches, and 12-1 momentum signal function VERBATIM (imported by file path, not
reimplemented) -- same $100M-$2B universe, same survivorship-bias-free design.
Rebalance: MONTHLY, pre-registered before running (matches the original
literature's convention; not swept). Beta-neutral leg sizing (60-day OLS vs
EW-universe adjClose, gp/run.py's convention, adapted here rather than
cross-importing gp's separate cache namespace -- see compute_betas below).
Cost model: standard bid-ask by cap tier + borrow on the short leg, ACTUAL
turnover between consecutive monthly holdings (not an assumed fixed rate).

SECONDARY TEST -- INDEPENDENCE FROM MOMENTUM (built into the primary design):
at every rebalance date, also compute 12-1 momentum (S9's exact function) on
the same universe, report the cross-sectional Spearman correlation between
the two signals, and run a Fama-MacBeth two-step regression of forward return
on BOTH signals' cross-sectional ranks simultaneously -- this is the actual
test of whether nearness predicts returns after controlling for momentum, or
washes out once momentum is accounted for.
"""

import sys
import json
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Reuse momentum_jt (S9)'s infra by file path (avoids the run.py/run.py
# module-name collision discovered during S18 -- both files are named run.py) ──
_M9_PATH = Path(__file__).resolve().parents[1] / "momentum_jt" / "run.py"
_spec = importlib.util.spec_from_file_location("momentum_jt_run", _M9_PATH)
M9 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M9)

RESULTS  = Path(__file__).resolve().parent / "results"
REGISTRY = Path(__file__).resolve().parents[1] / "trial_registry.csv"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Pre-registered parameters (locked before running) ────────────────────────
MIN_52W_DAYS      = 200     # min trailing trading days required for the 52w-high ratio
LOOKBACK_DAYS     = 252     # trailing trading days defining "52 weeks"
BETA_WINDOW_DAYS  = 60
BETA_MIN_DAYS     = 20
BETA_LOOKBACK_CAL = 95

IS_START = pd.Timestamp("2015-01-01")
IS_END   = pd.Timestamp("2021-01-01")
_all_dates = pd.date_range(IS_START, IS_END, freq="MS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 72 monthly periods

SPREAD_MIN_BPS, SPREAD_MAX_BPS = 20, 100
BORROW_MIN_ANNUAL, BORROW_MAX_ANNUAL = 0.005, 0.020
MCAP_MIN, MCAP_MAX = M9.MCAP_MIN, M9.MCAP_MAX


# =============================================================================
# 52-week-high nearness signal
# =============================================================================

def compute_52w_signal(ticker: str, rebalance_date: pd.Timestamp) -> float | None:
    window_start = rebalance_date - pd.Timedelta(days=400)   # calendar buffer for 252 trading days
    adj = M9._adj_slice(ticker, window_start, rebalance_date)
    if len(adj) < MIN_52W_DAYS:
        return None
    window = adj.tail(LOOKBACK_DAYS)
    high = float(window["adjClose"].max())
    current = float(window.iloc[-1]["adjClose"])
    if high <= 0 or not np.isfinite(high):
        return None
    return current / high


# =============================================================================
# Beta-neutral leg sizing (adapted from gp/run.py's convention; uses M9's own
# adjClose/raw-close caches directly rather than importing a second, separately
# populated cache namespace from gp/run.py)
# =============================================================================

def compute_betas(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict:
    beta_start = rebalance_date - pd.Timedelta(days=BETA_LOOKBACK_CAL)
    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj = M9._adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            raw_df = M9._PRICE_CACHE.get(ticker, M9._PRICE_EMPTY)
            if raw_df.empty:
                continue
            raw_w = raw_df[(raw_df["date"] >= beta_start) & (raw_df["date"] <= rebalance_date)]
            if raw_w.empty:
                continue
            s = raw_w.set_index("date")["close"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()
        else:
            s = adj.set_index("date")["adjClose"]
            s = s[~s.index.duplicated(keep="last")].sort_index()
            ret = s.pct_change().dropna()
        if len(ret) >= BETA_MIN_DAYS:
            adj_series[ticker] = ret

    if not adj_series:
        return {}
    ret_mat = pd.DataFrame(adj_series).sort_index()
    ew_mkt = ret_mat.clip(-0.5, 0.5).mean(axis=1)

    betas: dict[str, float] = {}
    for ticker, ret_s in adj_series.items():
        x_s = ew_mkt.reindex(ret_s.index).dropna()
        y_s = ret_s.reindex(x_s.index).dropna()
        x_s = x_s.reindex(y_s.index)
        n = len(y_s)
        if n < BETA_MIN_DAYS:
            betas[ticker] = np.nan
            continue
        X = np.column_stack([np.ones(n), x_s.values])
        try:
            coeffs, _, rank, _ = np.linalg.lstsq(X, y_s.values, rcond=None)
            betas[ticker] = float(coeffs[1]) if rank >= 2 else np.nan
        except Exception:
            betas[ticker] = np.nan
    return betas


def _spread(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return (SPREAD_MAX_BPS + (SPREAD_MIN_BPS - SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - MCAP_MIN) / (MCAP_MAX - MCAP_MIN)))
    return BORROW_MAX_ANNUAL + (BORROW_MIN_ANNUAL - BORROW_MAX_ANNUAL) * frac


def build_portfolio(signals_df: pd.DataFrame, betas: dict, signal_col: str) -> dict:
    df = signals_df.dropna(subset=[signal_col, "raw_return"]).copy()
    n = len(df)
    if n < 40:
        return {"longs": [], "shorts": [], "lw": {}, "sw": {}, "long_scale": 1.0,
                "short_scale": 1.0, "mean_beta_long": np.nan, "mean_beta_short": np.nan}

    leg = max(int(np.ceil(n * 0.20)), 20)
    leg = min(leg, n // 2)

    ranked = df.sort_values(signal_col, ascending=False)
    longs  = ranked.iloc[:leg]["ticker"].tolist()   # nearest to 52w high = long
    shorts = ranked.iloc[-leg:]["ticker"].tolist()  # furthest below = short

    lw = {t: 1.0 / leg for t in longs}
    sw = {t: 1.0 / leg for t in shorts}

    mean_beta_long  = float(np.nanmean([betas.get(t, np.nan) for t in longs]))
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in shorts]))
    denom = mean_beta_long + mean_beta_short
    if np.isfinite(denom) and denom > 0.05:
        long_scale  = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long  / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return {"longs": longs, "shorts": shorts, "lw": lw, "sw": sw,
            "long_scale": long_scale, "short_scale": short_scale,
            "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short}


def _leg_return_detail(returns_df, tickers, weights, is_short, turnover):
    ret_map  = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (MCAP_MIN + MCAP_MAX) / 2
    gross = ba_cost = borrow_cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w  = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross   += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / 12.0) * w   # MONTHLY borrow = annual / 12
    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    return (gross - total_cost), gross, ba_cost, borrow_cost


# =============================================================================
# Metrics (monthly periods -> annualize with sqrt(12) / 12)
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n = len(ls)
    if n < 2:
        return {"total_ret": np.nan, "ann_ret": np.nan, "sharpe": np.nan,
                "calmar": np.nan, "max_dd": np.nan, "t_stat": np.nan,
                "win_rate": np.nan, "n": n}
    cum = (1 + ls).cumprod()
    total = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (12.0 / n) - 1)
    mu, sigma = float(ls.mean()), float(ls.std(ddof=1))
    sharpe = float((mu / sigma) * np.sqrt(12)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s = (cum - roll_mx) / roll_mx
    max_dd = float(dd_s.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    t_stat = float(mu / (sigma / np.sqrt(n))) if sigma > 0 else np.nan
    win_rt = float((ls > 0).mean())
    return {"total_ret": total, "ann_ret": ann_ret, "sharpe": sharpe,
            "calmar": calmar, "max_dd": max_dd, "t_stat": t_stat,
            "win_rate": win_rt, "n": n}


def compute_market_beta(ls_returns: pd.Series, mkt_returns: pd.Series) -> dict:
    combined = pd.DataFrame({"ls": ls_returns, "mkt": mkt_returns}).dropna()
    if len(combined) < 5:
        return {"beta": np.nan, "beta_t": np.nan}
    x, y = combined["mkt"].values, combined["ls"].values
    X = np.column_stack([np.ones(len(x)), x])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return {"beta": np.nan, "beta_t": np.nan}
    beta = float(coeffs[1])
    y_hat = X @ coeffs
    ss_res = float(np.dot(y - y_hat, y - y_hat))
    n = len(y)
    se = (np.sqrt(ss_res / max(n - 2, 1)) / (np.std(x, ddof=1) * np.sqrt(n - 1)))
    beta_t = float(beta / se) if se > 0 else np.nan
    return {"beta": beta, "beta_t": beta_t}


def deflated_sharpe_threshold(n_trials: int, n_obs: int) -> float:
    if n_trials <= 1 or n_obs <= 1:
        return np.nan
    return float(stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs))


# =============================================================================
# Backtest engine
# =============================================================================

def run_backtest() -> dict:
    print("[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = M9.load_tiingo_tickers()
    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = M9.load_prefilter()
    survivors = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    M9.preload_all_prices(survivor_tickers)
    print(f"\n[Step 3b] Preloading adjClose prices ({len(survivor_tickers):,} tickers)...")
    M9.preload_all_adj_prices(survivor_tickers)
    print(f"\n[Step 4] Preloading SEC shares-outstanding facts ({len(survivor_ciks):,} CIKs)...")
    M9.preload_sec_facts(survivor_ciks)

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list = []
    fm_records: list = []       # Fama-MacBeth per-period cross-sectional regression coefficients
    corr_records: list = []     # per-period rank correlation between nearness and momentum

    prev_long_set: set  = set()
    prev_short_set: set = set()

    n_periods = len(REBALANCE_DATES)
    for i, t in enumerate(REBALANCE_DATES):
        hold_end = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        try:
            univ = M9.build_universe(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe: {e}")
            continue
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        univ["nearness"] = [compute_52w_signal(r["ticker"], t) for _, r in univ.iterrows()]
        univ["momentum"] = [M9.compute_momentum_signal(r["ticker"], t) for _, r in univ.iterrows()]
        n_near = univ["nearness"].notna().sum()
        n_mom  = univ["momentum"].notna().sum()
        n_both = univ[["nearness", "momentum"]].notna().all(axis=1).sum()
        print(f"  [Signal] nearness={n_near}/{len(univ)} momentum={n_mom}/{len(univ)} both={n_both}")

        try:
            returns_df = M9.compute_returns(univ, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        betas = compute_betas(univ, t)
        port = build_portfolio(returns_df, betas, "nearness")
        longs, shorts, lw, sw = port["longs"], port["shorts"], port["lw"], port["sw"]
        if not longs:
            print("  [SKIP] n<40 after signal filter")
            continue

        if prev_long_set:
            long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1)
            short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0
        prev_long_set, prev_short_set = set(longs), set(shorts)
        avg_turnover = (long_turnover + short_turnover) / 2.0

        long_net, long_gross, long_ba, long_bw = _leg_return_detail(
            returns_df, longs, lw, is_short=False, turnover=long_turnover)
        short_net, short_gross, short_ba, short_bw = _leg_return_detail(
            returns_df, shorts, sw, is_short=True, turnover=short_turnover)

        ls_net   = port["long_scale"] * long_net   + port["short_scale"] * short_net
        ls_gross = port["long_scale"] * long_gross + port["short_scale"] * short_gross
        ls_ba    = port["long_scale"] * long_ba    + port["short_scale"] * short_ba
        ls_bw    = port["long_scale"] * long_bw    + port["short_scale"] * short_bw
        ls_cost  = ls_ba + ls_bw

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        valid_sig = returns_df.dropna(subset=["nearness", "raw_return"])
        ic = (stats.spearmanr(valid_sig["nearness"], valid_sig["raw_return"])[0]
              if len(valid_sig) >= 10 else np.nan)

        # ── Momentum independence: correlation + Fama-MacBeth cross-sectional reg ──
        both = returns_df.dropna(subset=["nearness", "momentum", "raw_return"])
        if len(both) >= 20:
            near_mom_corr, _ = stats.spearmanr(both["nearness"], both["momentum"])
            near_rank = both["nearness"].rank(pct=True).values
            mom_rank  = both["momentum"].rank(pct=True).values
            y = both["raw_return"].values
            X = np.column_stack([np.ones(len(y)), near_rank, mom_rank])
            try:
                coeffs, _, rank_, _ = np.linalg.lstsq(X, y, rcond=None)
                b_near, b_mom = float(coeffs[1]), float(coeffs[2])
            except Exception:
                b_near, b_mom = np.nan, np.nan
        else:
            near_mom_corr, b_near, b_mom = np.nan, np.nan, np.nan

        corr_records.append({"rebalance_date": t, "corr": near_mom_corr, "n": len(both)})
        fm_records.append({"rebalance_date": t, "b_nearness": b_near, "b_momentum": b_mom, "n": len(both)})

        print(f"  L/S={ls_net:+.2%} gr={ls_gross:+.2%} cost={ls_cost:.3%} "
              f"to={avg_turnover:.0%} IC={ic:.3f} corr(near,mom)={near_mom_corr:.3f} "
              f"n_long={len(longs)}")

        period_records.append({
            "rebalance_date": t, "hold_end": hold_end,
            "n_universe": len(returns_df), "n_with_signal": n_near,
            "ls_ret": ls_net, "ls_gross": ls_gross, "ls_cost": ls_cost,
            "ls_ba_cost": ls_ba, "ls_borrow_cost": ls_bw,
            "long_scale": port["long_scale"], "short_scale": port["short_scale"],
            "mean_beta_long": port["mean_beta_long"], "mean_beta_short": port["mean_beta_short"],
            "avg_turnover": avg_turnover, "mkt_ret": mkt_ret, "ic": ic,
            "n_long": len(longs), "n_short": len(shorts),
        })

    return {
        "period_records": period_records,
        "fm_records": fm_records,
        "corr_records": corr_records,
    }


if __name__ == "__main__":
    results = run_backtest()
    pd.DataFrame(results["period_records"]).to_csv(RESULTS / "s19_period_returns.csv", index=False)
    pd.DataFrame(results["fm_records"]).to_csv(RESULTS / "s19_famamacbeth.csv", index=False)
    pd.DataFrame(results["corr_records"]).to_csv(RESULTS / "s19_correlation.csv", index=False)
    print(f"\n[Saved] -> {RESULTS}/s19_period_returns.csv, s19_famamacbeth.csv, s19_correlation.csv")
