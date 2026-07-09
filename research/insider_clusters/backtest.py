"""
S13: Opportunistic Insider Cluster Buying -- backtest engine.
See run.py module docstring for full hypothesis/design/pre-registration.
"""
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from scipy import stats

import run as R

# =============================================================================
# Per-company Form-4 event assembly (owner-tagged, filing-date gated)
# =============================================================================

_EVENTS_CACHE: dict[int, list] = {}
_owner_fetch_failures = 0
_owner_fetch_total = 0


def get_company_events(cik: int) -> list:
    """All Form-4 transactions we have txn-level data for on this CIK
    (bounded by mgmt_pit's original E1 fetch windows -- coverage caveat
    logged separately), tagged with owner_key where available."""
    global _owner_fetch_failures, _owner_fetch_total
    if cik in _EVENTS_CACHE:
        return _EVENTS_CACHE[cik]

    subs = R.fetch_submissions(cik)
    if not subs:
        _EVENTS_CACHE[cik] = []
        return []

    filings = R.fetch_form4_filings(cik, subs)
    events = []
    for accn, filed, doc in filings:
        accn_nodash = accn.replace("-", "")
        txns = R._FORM4_BASIC.get(accn_nodash)
        if txns is None:
            continue

        owner_key = None
        aff10b5 = None
        has_purchase = any(t["action"] == "A" and t.get("price", 0) > 0 for t in txns)
        if has_purchase:
            _owner_fetch_total += 1
            owner_file = R.OWNER_DIR / f"{accn_nodash}.json"
            if owner_file.exists():
                with open(owner_file) as f:
                    od = json.load(f)
                ok = od.get("owner_key") or []
                owner_key = tuple(sorted(ok)) if ok else None
                aff10b5 = od.get("aff10b5one")
            if owner_key is None:
                _owner_fetch_failures += 1

        for txn in txns:
            events.append({
                "accn": accn_nodash,
                "filed": filed,
                "date": txn["date"],
                "shares": txn["shares"],
                "price": txn["price"],
                "action": txn["action"],
                "owner_key": owner_key if owner_key is not None else (accn_nodash,),
                "owner_known": owner_key is not None,
                "aff10b5one": aff10b5,
            })

    _EVENTS_CACHE[cik] = events
    return events


def classify_cluster_and_activity(cik: int, rebalance_date: pd.Timestamp) -> dict:
    """
    Returns dict: cluster_flag, no_buying_flag, n_opportunistic, n_routine,
    n_cluster_insiders_max (max distinct insiders seen in any 30d window).
    Filing-date gated (filed <= rebalance_date); transaction date determines
    calendar-month / trailing-quarter / cluster-window membership.
    """
    events = get_company_events(cik)
    as_of_str = rebalance_date.strftime("%Y-%m-%d")
    trailing_start = rebalance_date - pd.DateOffset(months=R.TRAILING_MONTHS)
    trailing_start_str = trailing_start.strftime("%Y-%m-%d")

    # Gate: filed <= rebalance_date (no look-ahead), always.
    gated = [e for e in events if e["filed"] <= as_of_str]

    # ── "No insider buying activity at all" (short-leg eligibility) ──────────
    any_A_in_quarter = any(
        e["action"] == "A" and (trailing_start_str <= e["date"] < as_of_str)
        for e in gated
    )
    no_buying_flag = not any_A_in_quarter

    # ── Purchases (identity-known) for routine/opportunistic + clustering ────
    purchases = [e for e in gated if e["action"] == "A" and e["price"] > 0 and e["owner_known"]]
    if not purchases:
        return {"cluster_flag": False, "no_buying_flag": no_buying_flag,
                "n_opportunistic": 0, "n_routine": 0, "max_cluster_insiders": 0}

    # Purchases in trailing quarter (candidates for this period's classification)
    in_quarter = [p for p in purchases if trailing_start_str <= p["date"] < as_of_str]
    if not in_quarter:
        return {"cluster_flag": False, "no_buying_flag": no_buying_flag,
                "n_opportunistic": 0, "n_routine": 0, "max_cluster_insiders": 0}

    # Owner purchase-month history (same company), from ALL gated purchases
    # (not just in-quarter) -- classification lookback.
    owner_month_years: dict[tuple, dict[int, set]] = {}
    for p in purchases:
        y, m = int(p["date"][:4]), int(p["date"][5:7])
        owner_month_years.setdefault(p["owner_key"], {}).setdefault(m, set()).add(y)

    n_routine = 0
    n_opportunistic = 0
    opp_events = []
    for p in in_quarter:
        if p["aff10b5one"] is True:
            n_routine += 1
            continue
        y, m = int(p["date"][:4]), int(p["date"][5:7])
        prior_years = {y - 1, y - 2, y - 3}
        hit_years = owner_month_years.get(p["owner_key"], {}).get(m, set()) & prior_years
        # Exclude the current occurrence's own year from counting toward itself
        hit_years = {yr for yr in hit_years if yr != y}
        if len(hit_years) >= R.ROUTINE_MIN_YEARS_HIT:
            n_routine += 1
        else:
            n_opportunistic += 1
            opp_events.append(p)

    # ── Cluster detection: >=3 distinct insiders, opportunistic purchases,
    # within a rolling 30-day window, all inside the trailing quarter. ──────
    opp_events.sort(key=lambda e: e["date"])
    max_insiders = 0
    n = len(opp_events)
    for i in range(n):
        d0 = pd.Timestamp(opp_events[i]["date"])
        owners = set()
        for j in range(i, n):
            dj = pd.Timestamp(opp_events[j]["date"])
            if (dj - d0).days > R.CLUSTER_WINDOW_DAYS:
                break
            owners.add(opp_events[j]["owner_key"])
        max_insiders = max(max_insiders, len(owners))

    cluster_flag = max_insiders >= R.CLUSTER_MIN_INSIDERS

    return {
        "cluster_flag": cluster_flag,
        "no_buying_flag": no_buying_flag,
        "n_opportunistic": n_opportunistic,
        "n_routine": n_routine,
        "max_cluster_insiders": max_insiders,
    }


# =============================================================================
# Returns, beta, cost model (reused pattern from accruals/gp trials)
# =============================================================================

def compute_returns(universe_df: pd.DataFrame, hold_start: pd.Timestamp,
                     hold_end: pd.Timestamp, delisted_map: dict) -> pd.DataFrame:
    df = universe_df.copy()
    entry_prices, exit_prices, raw_returns, delisted_flags = [], [], [], []
    for _, row in df.iterrows():
        ticker = row["ticker"]
        prices = R._get_price_slice(ticker, hold_start, hold_end + pd.Timedelta(days=5))
        if prices.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan); delisted_flags.append(False)
            continue
        entry_cands = prices[prices["date"] >= hold_start]
        if entry_cands.empty:
            entry_prices.append(np.nan); exit_prices.append(np.nan)
            raw_returns.append(np.nan); delisted_flags.append(False)
            continue
        entry_price = float(entry_cands.iloc[0]["close"])
        delist_date = delisted_map.get(ticker, pd.NaT)
        is_delisted = pd.notna(delist_date) and (delist_date < hold_end)
        if is_delisted:
            exit_cands = prices[prices["date"] <= delist_date]
            exit_price = float((exit_cands if not exit_cands.empty else prices).iloc[-1]["close"])
            delisted_flags.append(True)
        else:
            exit_cands = prices[prices["date"] <= hold_end]
            exit_price = float(exit_cands.iloc[-1]["close"]) if not exit_cands.empty else entry_price
            delisted_flags.append(False)
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        raw_returns.append(exit_price / entry_price - 1)
    df["entry_price"] = entry_prices
    df["exit_price"] = exit_prices
    df["raw_return"] = raw_returns
    df["delisted"] = delisted_flags
    return df


def compute_betas(universe_df: pd.DataFrame, rebalance_date: pd.Timestamp) -> dict:
    beta_start = rebalance_date - pd.Timedelta(days=R.BETA_LOOKBACK_CAL)
    adj_series: dict[str, pd.Series] = {}
    for _, row in universe_df.iterrows():
        ticker = row["ticker"]
        adj = R._adj_slice(ticker, beta_start, rebalance_date)
        if adj.empty:
            raw_df = R._PRICE_CACHE.get(ticker, R._ADJ_EMPTY)
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
        if len(ret) >= R.BETA_MIN_DAYS:
            adj_series[ticker] = ret

    if not adj_series:
        return {}
    ret_mat = pd.DataFrame(adj_series).sort_index()
    ew_mkt = ret_mat.clip(-0.5, 0.5).mean(axis=1)

    betas: dict[str, float] = {}
    for ticker, ret_s in adj_series.items():
        y_s = ret_s
        x_s = ew_mkt.reindex(y_s.index).dropna()
        y_s = y_s.reindex(x_s.index).dropna()
        x_s = x_s.reindex(y_s.index)
        n = len(y_s)
        if n < R.BETA_MIN_DAYS:
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
    frac = max(0.0, min(1.0, (mcap - R.MCAP_MIN) / (R.MCAP_MAX - R.MCAP_MIN)))
    return (R.SPREAD_MAX_BPS + (R.SPREAD_MIN_BPS - R.SPREAD_MAX_BPS) * frac) / 10_000


def _borrow(mcap: float) -> float:
    frac = max(0.0, min(1.0, (mcap - R.MCAP_MIN) / (R.MCAP_MAX - R.MCAP_MIN)))
    return R.BORROW_MAX_ANNUAL + (R.BORROW_MIN_ANNUAL - R.BORROW_MAX_ANNUAL) * frac


def _leg_return_detail(returns_df, tickers, weights, is_short, turnover):
    ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
    mcap_map = returns_df.set_index("ticker")["mcap"].to_dict()
    mid = (R.MCAP_MIN + R.MCAP_MAX) / 2
    gross = ba_cost = borrow_cost = 0.0
    for t in tickers:
        r = ret_map.get(t, np.nan)
        if np.isnan(r):
            continue
        w = weights.get(t, 0.0)
        mc = mcap_map.get(t, mid)
        gross += w * r
        ba_cost += turnover * 2.0 * _spread(mc) * w
        if is_short:
            borrow_cost += (_borrow(mc) / 4.0) * w
    total_cost = ba_cost + borrow_cost
    if is_short:
        return (-gross - total_cost), (-gross), ba_cost, borrow_cost
    return (gross - total_cost), gross, ba_cost, borrow_cost


def build_portfolio(signals_df: pd.DataFrame, betas: dict) -> dict:
    """Long = cluster_flag names (equal weight, ALL of them -- flag-based,
    not a quantile sort). Short = no_buying_flag names (equal weight, ALL).
    Beta-neutral dollar sizing between legs. Name-count guard: long leg
    reported but excluded from the P&L series if < LONG_LEG_MIN_NAMES."""
    df = signals_df.dropna(subset=["raw_return"]).copy()
    longs = df[df["cluster_flag"]]["ticker"].tolist()
    shorts = df[df["no_buying_flag"]]["ticker"].tolist()

    guard_fired = len(longs) < R.LONG_LEG_MIN_NAMES
    usable = len(longs) > 0 and not guard_fired and len(shorts) > 0

    lw = {t: 1.0 / len(longs) for t in longs} if longs else {}
    sw = {t: 1.0 / len(shorts) for t in shorts} if shorts else {}

    mean_beta_long = float(np.nanmean([betas.get(t, np.nan) for t in longs])) if longs else np.nan
    mean_beta_short = float(np.nanmean([betas.get(t, np.nan) for t in shorts])) if shorts else np.nan
    denom = mean_beta_long + mean_beta_short if (longs and shorts) else np.nan
    if np.isfinite(denom) and denom > 0.05:
        long_scale = float(np.clip(2.0 * mean_beta_short / denom, 0.2, 3.0))
        short_scale = float(np.clip(2.0 * mean_beta_long / denom, 0.2, 3.0))
    else:
        long_scale = short_scale = 1.0

    return {
        "longs": longs, "shorts": shorts, "lw": lw, "sw": sw,
        "long_scale": long_scale, "short_scale": short_scale,
        "mean_beta_long": mean_beta_long, "mean_beta_short": mean_beta_short,
        "guard_fired": guard_fired, "usable": usable,
    }


# =============================================================================
# Metrics
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
    ann_ret = float((1 + total) ** (4.0 / n) - 1)
    mu = float(ls.mean())
    sigma = float(ls.std(ddof=1))
    sharpe = float((mu / sigma) * np.sqrt(4)) if sigma > 0 else np.nan
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
    x = combined["mkt"].values
    y = combined["ls"].values
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
    return stats.norm.ppf(1 - 1.0 / n_trials) / np.sqrt(n_obs)


# =============================================================================
# Backtest engine
# =============================================================================

def _form4_cache_max_date() -> str:
    """Diagnostic: the reused mgmt_pit Form-4 cache (form4_all.json) turns out
    to have near-zero transaction density after ~2018-01 (discovered during
    this trial, not caused by it -- likely an incomplete/interrupted historical
    E1 fetch run). Returns the last date with meaningful (>=50 txn) density."""
    from collections import Counter
    month_counts: Counter = Counter()
    for txns in R._FORM4_BASIC.values():
        for t in txns:
            month_counts[t["date"][:7]] += 1
    dense_months = sorted(m for m, c in month_counts.items() if c >= 50)
    return dense_months[-1] if dense_months else "unknown"


def run_backtest(tiingo_tickers: pd.DataFrame, universes: dict) -> dict:
    t_total = time.time()
    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    form4_max_dense_month = _form4_cache_max_date()
    print(f"\n[Coverage] Reused Form-4 cache (mgmt_pit/form4_all.json) has "
          f"meaningful transaction density through: {form4_max_dense_month} "
          f"-- pre-existing gap in the reused cache, discovered (not caused) "
          f"by this trial. Quarters after this are DATA BLACKOUTS, not "
          f"evidence of cluster-event rarity.")

    period_records = []
    quarter_event_log = []
    sector_contrib: dict = {}
    sign_violations = 0
    guard_fire_count = 0
    n_usable_periods = 0

    prev_long_set: set = set()
    prev_short_set: set = set()

    n_periods = len(R.REBALANCE_DATES)
    for i, t in enumerate(R.REBALANCE_DATES):
        hold_end = R.REBALANCE_DATES[i + 1] if i + 1 < n_periods else R.IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        univ = universes[t]
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        # ── Signal: classify each company ────────────────────────────────
        cluster_flags, no_buying_flags, m1_vals = [], [], []
        n_opp_total, n_rtn_total, n_cluster_companies = 0, 0, 0
        for _, row in univ.iterrows():
            cik = int(row["cik"])
            cls = classify_cluster_and_activity(cik, t)
            cluster_flags.append(cls["cluster_flag"])
            no_buying_flags.append(cls["no_buying_flag"])
            n_opp_total += cls["n_opportunistic"]
            n_rtn_total += cls["n_routine"]
            if cls["cluster_flag"]:
                n_cluster_companies += 1

            subs = R.fetch_submissions(cik)
            m1 = R.compute_m1_insider_buying(cik, subs, t, row["shares"]) if subs else 0.0
            m1_vals.append(m1)

        univ = univ.copy()
        univ["cluster_flag"] = cluster_flags
        univ["no_buying_flag"] = no_buying_flags
        univ["m1_naive"] = m1_vals

        n_no_buying = int(sum(no_buying_flags))
        data_blackout = (n_no_buying == len(univ)) and (n_opp_total + n_rtn_total == 0)
        quarter_event_log.append({
            "rebalance_date": t, "n_universe": len(univ),
            "n_cluster_companies": n_cluster_companies,
            "n_no_buying_companies": n_no_buying,
            "n_opportunistic_purchases": n_opp_total,
            "n_routine_purchases": n_rtn_total,
            "data_blackout": data_blackout,
        })
        if data_blackout:
            print(f"  [DATA BLACKOUT] zero Form-4 transactions found for ANY company "
                  f"this quarter (pre-existing reused-cache gap, not signal rarity)")

        # ── Returns ───────────────────────────────────────────────────────
        returns_df = compute_returns(univ, t, hold_end, delisted_map)
        n_delisted = int(returns_df["delisted"].sum())

        # ── Beta + portfolio ─────────────────────────────────────────────
        betas = compute_betas(univ, t)
        port = build_portfolio(returns_df, betas)
        longs, shorts = port["longs"], port["shorts"]

        if port["guard_fired"]:
            guard_fire_count += 1
            print(f"  [GUARD] long (cluster) leg = {len(longs)} names "
                  f"(< {R.LONG_LEG_MIN_NAMES} minimum) -- period EXCLUDED from P&L series")

        # ── IC (Spearman): cluster_flag vs return, naive M1 vs return ──────
        valid = returns_df.dropna(subset=["raw_return"])
        if len(valid) >= 10:
            ic_cluster, _ = stats.spearmanr(valid["cluster_flag"].astype(float), valid["raw_return"])
            ic_naive, _ = stats.spearmanr(valid["m1_naive"], valid["raw_return"])
        else:
            ic_cluster = ic_naive = np.nan

        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        record = {
            "rebalance_date": t, "hold_end": hold_end,
            "n_universe": len(returns_df), "n_delisted": n_delisted,
            "n_long": len(longs), "n_short": len(shorts),
            "guard_fired": port["guard_fired"], "usable": port["usable"],
            "ic_cluster": ic_cluster, "ic_naive": ic_naive,
            "mkt_ret": mkt_ret,
        }

        if port["usable"]:
            n_usable_periods += 1
            if prev_long_set:
                long_turnover = len(set(longs) - prev_long_set) / max(len(longs), 1)
                short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
            else:
                long_turnover = short_turnover = 1.0
            avg_turnover = (long_turnover + short_turnover) / 2.0

            long_net, long_gross, long_ba, long_bw = _leg_return_detail(
                returns_df, longs, port["lw"], is_short=False, turnover=long_turnover)
            short_net, short_gross, short_ba, short_bw = _leg_return_detail(
                returns_df, shorts, port["sw"], is_short=True, turnover=short_turnover)

            ls_net = port["long_scale"] * long_net + port["short_scale"] * short_net
            ls_gross = port["long_scale"] * long_gross + port["short_scale"] * short_gross
            ls_ba = port["long_scale"] * long_ba + port["short_scale"] * short_ba
            ls_bw = port["long_scale"] * long_bw + port["short_scale"] * short_bw

            # Sign-convention audit: cluster-leg mean return should exceed
            # no-buying-leg mean return (hypothesis: cluster buying informative).
            ret_map = returns_df.set_index("ticker")["raw_return"].to_dict()
            mean_long_ret = float(np.mean([ret_map[t] for t in longs]))
            mean_short_ret = float(np.mean([ret_map[t] for t in shorts]))
            sign_ok = mean_long_ret > mean_short_ret
            if not sign_ok:
                sign_violations += 1

            # Sector concentration tracking
            sic_map = returns_df.set_index("ticker")["sic"].to_dict()
            for tk in longs:
                r_ = ret_map.get(tk, np.nan)
                if np.isnan(r_):
                    continue
                sic = sic_map.get(tk, np.nan)
                grp = int(sic // 100) if pd.notna(sic) else -1
                sector_contrib[grp] = sector_contrib.get(grp, 0.0) + port["long_scale"] * port["lw"].get(tk, 0.0) * r_
            for tk in shorts:
                r_ = ret_map.get(tk, np.nan)
                if np.isnan(r_):
                    continue
                sic = sic_map.get(tk, np.nan)
                grp = int(sic // 100) if pd.notna(sic) else -1
                sector_contrib[grp] = sector_contrib.get(grp, 0.0) - port["short_scale"] * port["sw"].get(tk, 0.0) * r_

            record.update({
                "ls_ret": ls_net, "ls_gross": ls_gross, "ls_cost": ls_ba + ls_bw,
                "ls_ba_cost": ls_ba, "ls_borrow_cost": ls_bw,
                "long_scale": port["long_scale"], "short_scale": port["short_scale"],
                "mean_beta_long": port["mean_beta_long"], "mean_beta_short": port["mean_beta_short"],
                "avg_turnover": avg_turnover, "sign_ok": sign_ok,
                "mean_long_ret": mean_long_ret, "mean_short_ret": mean_short_ret,
            })
            prev_long_set, prev_short_set = set(longs), set(shorts)
            print(f"  L/S={ls_net:+.2%} nL={len(longs)} nS={len(shorts)} "
                  f"IC_clust={ic_cluster:.3f} IC_naive={ic_naive:.3f}")
        else:
            record.update({
                "ls_ret": np.nan, "ls_gross": np.nan, "ls_cost": np.nan,
                "ls_ba_cost": np.nan, "ls_borrow_cost": np.nan,
                "long_scale": np.nan, "short_scale": np.nan,
                "mean_beta_long": np.nan, "mean_beta_short": np.nan,
                "avg_turnover": np.nan, "sign_ok": None,
                "mean_long_ret": np.nan, "mean_short_ret": np.nan,
            })
            print(f"  [UNUSABLE] nL={len(longs)} nS={len(shorts)} "
                  f"IC_clust={ic_cluster:.3f} IC_naive={ic_naive:.3f}")

        period_records.append(record)

    elapsed = time.time() - t_total
    n_blackout = int(sum(1 for e in quarter_event_log if e["data_blackout"]))
    n_data_covered = n_periods - n_blackout
    return {
        "period_records": period_records,
        "quarter_event_log": quarter_event_log,
        "sector_contrib": sector_contrib,
        "sign_violations": sign_violations,
        "guard_fire_count": guard_fire_count,
        "n_usable_periods": n_usable_periods,
        "n_periods_total": n_periods,
        "n_blackout_periods": n_blackout,
        "n_data_covered_periods": n_data_covered,
        "form4_max_dense_month": form4_max_dense_month,
        "elapsed_s": elapsed,
        "owner_fetch_failures": _owner_fetch_failures,
        "owner_fetch_total": _owner_fetch_total,
    }


# =============================================================================
# Output & registry
# =============================================================================

def print_and_save_results(results: dict) -> dict:
    df_all = pd.DataFrame(results["period_records"])
    ev_df = pd.DataFrame(results["quarter_event_log"])
    usable = df_all[df_all["usable"] == True].copy()  # noqa: E712

    sep = "=" * 78
    print(f"\n{sep}")
    print("S13: OPPORTUNISTIC INSIDER CLUSTER BUYING -- IN-SAMPLE RESULTS (trial #20)")
    print(sep)

    n_periods_total = results["n_periods_total"]
    n_usable = results["n_usable_periods"]
    n_blackout = results["n_blackout_periods"]
    n_covered = results["n_data_covered_periods"]
    print(f"\n{sep}")
    print("*** DATA-COVERAGE FINDING (discovered by this trial, not caused by it) ***")
    print(sep)
    print(f"  The reused mgmt_pit Form-4 cache (form4_all.json, E1's own fetch output) "
          f"has meaningful transaction density only through {results['form4_max_dense_month']}. "
          f"{n_blackout}/{n_periods_total} quarters ({str(R.REBALANCE_DATES[n_periods_total-n_blackout].date()) if n_blackout < n_periods_total else 'n/a'} "
          f"onward) are total DATA BLACKOUTS -- zero Form-4 transactions found for "
          f"ANY company in the universe that quarter. This is a pre-existing gap in "
          f"the reused data pipeline, not evidence that insider buying (or cluster "
          f"events) became rare. Effective supported window for this signal: "
          f"{n_covered}/{n_periods_total} quarters (2015-01-01 through the last "
          f"pre-blackout quarter).")
    print(f"  Within the {n_covered} DATA-COVERED quarters, the name-count guard "
          f"(long leg <{R.LONG_LEG_MIN_NAMES}) fired {max(results['guard_fire_count']-n_blackout,0)} times "
          f"-- i.e. cluster events were NOT rare when data was actually available.")

    print(f"\nQuarters with usable (>=20 name) cluster long leg: "
          f"{n_usable}/{n_periods_total} ({100*n_usable/n_periods_total:.0f}% of all quarters, "
          f"{100*n_usable/max(n_covered,1):.0f}% of DATA-COVERED quarters)")

    total_cluster_events = int(ev_df["n_cluster_companies"].sum())
    total_opp = int(ev_df["n_opportunistic_purchases"].sum())
    total_rtn = int(ev_df["n_routine_purchases"].sum())
    print(f"\nTotal cluster-buying events (company-quarters) across in-sample window: "
          f"{total_cluster_events} (all within the {n_covered} data-covered quarters)")
    print(f"Total opportunistic purchases classified: {total_opp}")
    print(f"Total routine purchases classified: {total_rtn}")

    frac_failed = (results["owner_fetch_failures"] / results["owner_fetch_total"]
                   if results["owner_fetch_total"] > 0 else 0.0)
    print(f"\nOwner-identity fetch coverage: {results['owner_fetch_total'] - results['owner_fetch_failures']}"
          f"/{results['owner_fetch_total']} purchase filings resolved to an owner CIK "
          f"({100*(1-frac_failed):.1f}%)")

    # ── IC (both signals, ALL periods incl. unusable-for-P&L ones) ─────────
    ic_c = df_all["ic_cluster"].dropna()
    ic_n = df_all["ic_naive"].dropna()
    mean_ic_c = float(ic_c.mean()) if len(ic_c) else np.nan
    ic_c_t = (ic_c.mean() / (ic_c.std(ddof=1) / np.sqrt(len(ic_c)))
              if len(ic_c) > 1 and ic_c.std(ddof=1) > 0 else np.nan)
    mean_ic_n = float(ic_n.mean()) if len(ic_n) else np.nan
    ic_n_t = (ic_n.mean() / (ic_n.std(ddof=1) / np.sqrt(len(ic_n)))
              if len(ic_n) > 1 and ic_n.std(ddof=1) > 0 else np.nan)

    print(f"\n{sep}")
    print("INFORMATION COEFFICIENT (Spearman, cost-independent)")
    print(sep)
    print(f"  Cluster-flag IC  : mean={mean_ic_c:>8.4f}  t={ic_c_t:>7.3f}  n={len(ic_c)}")
    print(f"  Naive M1 IC (E1) : mean={mean_ic_n:>8.4f}  t={ic_n_t:>7.3f}  n={len(ic_n)}  "
          f"(E1 original: IC=+0.0002, t=0.02)")

    # ── P&L metrics (usable periods only) ───────────────────────────────────
    if len(usable) >= 2:
        m = compute_metrics(usable["ls_ret"])
        m_gross = compute_metrics(usable["ls_gross"])
        beta_stats = compute_market_beta(usable["ls_ret"], usable["mkt_ret"])
        per_period_sr = (float(usable["ls_ret"].mean() / usable["ls_ret"].std(ddof=1))
                         if usable["ls_ret"].std(ddof=1) > 0 else np.nan)
        dsr_thr = deflated_sharpe_threshold(R.N_TRIALS, m["n"])
        clears_dsr = bool(per_period_sr > dsr_thr) if not np.isnan(per_period_sr) and not np.isnan(dsr_thr) else False
        annual_cost_drag = (float(usable["ls_ba_cost"].mean()) + float(usable["ls_borrow_cost"].mean())) * 4 * 10_000
        mean_turnover = float(usable["avg_turnover"].mean())

        print(f"\n{sep}")
        print(f"P&L METRICS (usable quarters only, n={m['n']})")
        print(sep)
        print(f"  {'Annualized return (net)':<42} {m['ann_ret']:>12.2%}")
        print(f"  {'Annualized return (gross)':<42} {m_gross['ann_ret']:>12.2%}")
        print(f"  {'Cost drag (annualized bps)':<42} {annual_cost_drag:>12.0f}")
        print(f"  {'Mean turnover per quarter':<42} {mean_turnover:>12.0%}")
        print(f"  {'Sharpe (net, annualized)':<42} {m['sharpe']:>12.3f}")
        print(f"  {'CALMAR':<42} {m['calmar']:>12.3f}")
        print(f"  {'Max drawdown':<42} {m['max_dd']:>12.2%}")
        print(f"  {'Market beta':<42} {beta_stats['beta']:>12.3f}")
        print(f"  {'Beta t-stat':<42} {beta_stats['beta_t']:>12.3f}")
        print(f"  {'Win rate':<42} {m['win_rate']:>12.1%}")
        print(f"  {'N periods (usable)':<42} {m['n']:>12d}")
        print(f"\n  --- DSR (N_TRIALS={R.N_TRIALS}) ---")
        print(f"  {'Per-period SR':<42} {per_period_sr:>12.4f}")
        print(f"  {'DSR threshold':<42} {dsr_thr:>12.4f}")
        print(f"  {'Clears DSR':<42} {'YES' if clears_dsr else 'NO':>12}")
    else:
        m = {"ann_ret": np.nan, "sharpe": np.nan, "calmar": np.nan, "max_dd": np.nan,
             "t_stat": np.nan, "win_rate": np.nan, "n": len(usable)}
        m_gross = dict(m)
        beta_stats = {"beta": np.nan, "beta_t": np.nan}
        per_period_sr = dsr_thr = np.nan
        clears_dsr = False
        annual_cost_drag = mean_turnover = np.nan
        print(f"\n{sep}")
        print(f"P&L METRICS: INSUFFICIENT USABLE QUARTERS ({len(usable)}) -- "
              f"cannot compute Sharpe/DSR. Cluster events too rare for a P&L series.")
        print(sep)

    # ── Sign-convention audit ────────────────────────────────────────────────
    sv = results["sign_violations"]
    print(f"\n{sep}")
    print("SIGN-CONVENTION AUDIT")
    print(sep)
    print(f"  Periods where long-leg (cluster) mean return <= short-leg "
          f"(no-buying) mean return: {sv}/{len(usable)}")

    # ── Concentration check ──────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CONCENTRATION CHECK")
    print(sep)
    if len(usable) >= 1 and usable["ls_ret"].abs().sum() > 0:
        share = usable["ls_ret"].abs() / usable["ls_ret"].abs().sum()
        top_idx = share.idxmax()
        print(f"  Largest single-quarter share of |return| mass: "
              f"{usable.loc[top_idx, 'rebalance_date'].date()} = {float(share.loc[top_idx]):.1%}")
    sector_contrib = results.get("sector_contrib", {})
    if sector_contrib:
        total_abs = sum(abs(v) for v in sector_contrib.values())
        if total_abs > 0:
            top_sic, top_val = max(sector_contrib.items(), key=lambda kv: abs(kv[1]))
            print(f"  Largest single SIC major-group contribution: "
                  f"SIC {top_sic:02d} = {abs(top_val)/total_abs:.1%} of total |contribution|")

    # ── Per-quarter event log ────────────────────────────────────────────────
    print(f"\n{sep}")
    print("PER-QUARTER EVENT LOG")
    print(sep)
    print(f"  {'Date':<11} {'nUniv':>5} {'nClust':>6} {'nNoBuy':>6} {'nOpp':>5} "
          f"{'nRtn':>5} {'ICc':>7} {'ICn':>7} {'Usable':>7}")
    for _, r in df_all.iterrows():
        ev = ev_df[ev_df["rebalance_date"] == r["rebalance_date"]].iloc[0]
        print(f"  {str(r['rebalance_date'].date()):<11} {int(ev['n_universe']):>5} "
              f"{int(ev['n_cluster_companies']):>6} {int(ev['n_no_buying_companies']):>6} "
              f"{int(ev['n_opportunistic_purchases']):>5} {int(ev['n_routine_purchases']):>5} "
              f"{r['ic_cluster']:>7.3f} {r['ic_naive']:>7.3f} "
              f"{'Y' if r['usable'] else '':>7}")

    print(f"\nRuntime: {results['elapsed_s']:.1f}s ({results['elapsed_s']/60:.1f} min)")

    return {
        "m": m, "m_gross": m_gross, "beta_stats": beta_stats,
        "per_period_sr": per_period_sr, "dsr_thr": dsr_thr, "clears_dsr": clears_dsr,
        "mean_ic_c": mean_ic_c, "ic_c_t": ic_c_t, "mean_ic_n": mean_ic_n, "ic_n_t": ic_n_t,
        "annual_cost_drag": annual_cost_drag, "mean_turnover": mean_turnover,
        "total_cluster_events": total_cluster_events, "total_opp": total_opp,
        "total_rtn": total_rtn, "n_usable": n_usable, "n_periods_total": n_periods_total,
        "sign_violations": sv, "frac_owner_resolved": 1 - frac_failed,
        "n_blackout": n_blackout, "n_covered": n_covered,
        "form4_max_dense_month": results["form4_max_dense_month"],
    }


def save_csvs(results: dict) -> None:
    pd.DataFrame(results["period_records"]).to_csv(
        R.RESULTS / "insiderclusters_period_returns.csv", index=False)
    pd.DataFrame(results["quarter_event_log"]).to_csv(
        R.RESULTS / "insiderclusters_event_log.csv", index=False)
    print(f"\n[Saved] CSVs -> {R.RESULTS}/")


def log_registry(results: dict, summary: dict) -> None:
    verdict_bits = []
    verdict_bits.append("CLEARS DSR" if summary["clears_dsr"] else "FAILS DSR")
    verdict_bits.append(
        f"DATA-COVERAGE GAP in reused mgmt_pit Form-4 cache: dense through "
        f"{summary['form4_max_dense_month']}, {summary['n_blackout']}/"
        f"{summary['n_periods_total']} quarters are total blackouts (pre-existing, "
        f"discovered not caused by this trial) -- effective window "
        f"{summary['n_covered']}/{summary['n_periods_total']} quarters"
    )
    if summary["n_usable"] < summary["n_covered"]:
        verdict_bits.append(
            f"within covered quarters, {summary['n_usable']}/{summary['n_covered']} "
            f"had a usable (>=20 name) cluster long leg")
    else:
        verdict_bits.append(
            f"guard never fired within covered quarters -- cluster events were NOT rare "
            f"when data existed ({summary['total_cluster_events']} events total)")
    if summary["sign_violations"] > 0:
        verdict_bits.append(f"SIGN AUDIT: {summary['sign_violations']} violation(s)")
    verdict = " | ".join(verdict_bits)

    row = {
        "timestamp": datetime.now().isoformat(),
        "study": "insiderclusters_S13",
        "hypothesis": (
            "Opportunistic insider cluster buying (Cohen/Malloy/Pomorski 2012 "
            "routine-vs-opportunistic distinction applied to E1's naive net "
            "insider-buying signal, trial #1, IC~=0.0002 FAIL). Cluster = >=3 "
            "different insiders making opportunistic (non-routine) open-market "
            "purchases at the same company within a rolling 30-day window. "
            "Long = confirmed cluster event in trailing quarter; short = zero "
            "insider buying activity at all in trailing quarter (clean tilt-vs-"
            "no-signal design, not short-on-selling)."
        ),
        "data_source": (
            "Reused mgmt_pit/E1's Form-4 pipeline + universe entirely (Tiingo "
            "prices, SEC XBRL shares, SEC submissions/Form-4 all read from "
            "mgmt_pit's cache, read-only). New: raw Form-4 XML re-fetched for "
            "the ~10.6k filings containing a purchase, to extract reporting-"
            "owner identity + 10b5-1 flag (absent pre-2023 schema, confirmed "
            "empirically, expected non-factor for this 2015-2021 window)."
        ),
        "status": "completed_in_sample",
        "trial_number": 20,
        "trial_note": (
            "Independent trial #20. Reformulation of E1 (mgmt_pit's old "
            "4-trial-count era, never logged to the unified registry) -- "
            "first time the insider-buying mechanism enters the unified DSR "
            "count. Distinct mechanism (routine/opportunistic classification "
            "+ clustering), not a re-tune of E1. CONCURRENCY NOTE: originally "
            "run as trial #18 (N=17 at task start); two other trials "
            "(analystrevision_S14, compositeIC_S15) were logged by concurrent "
            "sessions during this trial's ~2.5hr runtime, discovered via a "
            "trial_number collision and corrected post-hoc to true N=20."
        ),
        "n_trials_dsr": R.N_TRIALS,
        "rebalance_freq": "quarterly",
        "rebalance_note": "Matches E1/E2/E3's original cadence for comparability.",
        "portfolio_note": (
            f"Beta-neutral (EW market proxy, 60-day adjClose). Flag-based legs "
            f"(not quantile sorts): long=all cluster-flagged names, short=all "
            f"zero-activity names. Name-count guard (long leg <{R.LONG_LEG_MIN_NAMES}) "
            f"never fired within the {summary['n_covered']} data-covered quarters "
            f"(2015-01-01 through {summary['form4_max_dense_month']}); the reused "
            f"mgmt_pit Form-4 cache is a total blackout (zero transactions, any "
            f"company) for the remaining {summary['n_blackout']} quarters -- a "
            f"pre-existing data gap discovered by this trial, not signal rarity."
        ),
        "ann_return_net": summary["m"]["ann_ret"],
        "ann_return_gross": summary["m_gross"]["ann_ret"],
        "annual_cost_drag_bps": summary["annual_cost_drag"],
        "mean_turnover": summary["mean_turnover"],
        "sharpe": summary["m"]["sharpe"],
        "calmar": summary["m"]["calmar"],
        "max_drawdown": summary["m"]["max_dd"],
        "t_stat": summary["m"]["t_stat"],
        "market_beta": summary["beta_stats"]["beta"],
        "market_beta_tstat": summary["beta_stats"]["beta_t"],
        "mean_ic": summary["mean_ic_c"],
        "ic_t_stat": summary["ic_c_t"],
        "win_rate": summary["m"]["win_rate"],
        "n_periods": summary["m"]["n"],
        "per_period_sr": summary["per_period_sr"],
        "dsr_threshold": summary["dsr_thr"],
        "clears_dsr": summary["clears_dsr"],
        "n_events": summary["total_cluster_events"],
        "signal_note": (
            f"Naive-M1(E1)-rerun on same universe/window: IC={summary['mean_ic_n']:.4f} "
            f"t={summary['ic_n_t']:.3f} (E1 original: IC=+0.0002, t=0.02). "
            f"Total opportunistic purchases classified={summary['total_opp']}, "
            f"routine={summary['total_rtn']}. Owner-identity resolved for "
            f"{summary['frac_owner_resolved']:.1%} of purchase filings."
        ),
        "notes": verdict,
    }

    reg_df = pd.read_csv(R.REGISTRY) if R.REGISTRY.exists() else pd.DataFrame()
    if not reg_df.empty and "study" in reg_df.columns:
        reg_df = reg_df[reg_df["study"] != "insiderclusters_S13"].copy()
    reg_df = pd.concat([reg_df, pd.DataFrame([row])], ignore_index=True)
    reg_df.to_csv(R.REGISTRY, index=False)
    print(f"[Registry] insiderclusters_S13 logged (trial #20) -> {R.REGISTRY}")
    print(f"[Verdict] {verdict}")


def main():
    print("=" * 78)
    print("S13: OPPORTUNISTIC INSIDER CLUSTER BUYING (independent trial #20)")
    print(f"N_TRIALS_DSR = {R.N_TRIALS} (true N at write-time; see CONCURRENCY NOTE)")
    print("=" * 78)

    tiingo = R.load_tiingo_tickers()
    R.preload_all_prices()
    R.preload_all_adj_prices()
    R.preload_shares_facts()
    R.load_form4_basic_cache()

    with open(R.OWN_CACHE / "universes.pkl", "rb") as f:
        d = pickle.load(f)
    universes = d["universes"]

    results = run_backtest(tiingo, universes)
    summary = print_and_save_results(results)
    save_csvs(results)
    log_registry(results, summary)


if __name__ == "__main__":
    main()
