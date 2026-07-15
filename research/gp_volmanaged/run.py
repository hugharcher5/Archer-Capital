"""
Volatility-Managed GP/A  (Moreira & Muir, "Volatility-Managed Portfolios," 2017)
=================================================================================
New independent trial. NOT a re-tune of S3-Q's GP/A signal -- this is a pure
construction-level overlay tested on top of the identical signal, identical
quintile long-short, identical beta-neutral leg sizing.

BASE SIGNAL (unchanged from S3-Q / gp/run.py, reused via import, not reimplemented):
  GP/A = TTM Gross Profit / Total Assets, SEC XBRL, filed <= t.
  Long top 20% GP/A, short bottom 20%, min 20 names/leg, equal-weight within leg.
  Beta-neutral sizing: long_$ x beta_L = short_$ x beta_S, gross = 2x (before overlay).
  Quarterly rebalance, adjClose for beta, raw close for hold-period returns.

UNIVERSE UPGRADE (this trial's first change vs S3-Q):
  S3-Q's universe is the standard SEC-prefilter + Tiingo + $100M-$2B mcap + $1M
  dollar-vol filter, with NO index-membership discipline at all. This trial adds
  a "confirmed-exit overlay": at each rebalance date, any ticker whose most
  recent Russell 3000 event (from the already-built Wayback-archived
  additions/deletions data in research/russell_reconstitution/results/) is a
  DELETION is excluded from the tradable universe from that point forward.

  IMPORTANT LIMITATION, confirmed before building (see conversation record) and
  approved by the user as the "confirmed-exit overlay" option:
  The archived Russell data is additions/deletions EVENTS per annual cycle, not
  an absolute full-membership snapshot, and does not separately tag Russell 1000
  vs Russell 2000 (only 'r3000' and 'rmicro'). There is no seed snapshot anywhere
  in this repo to reconstruct true point-in-time Russell 2000 membership from
  scratch. This overlay is therefore NOT "actual Russell 2000 membership" -- it
  is the standard mcap-filtered universe with CONFIRMED R3000 exits removed.
  Confirmed inclusion is not established for names that never appear in an
  event; they remain in the universe by default by the same logic as before.
  Additionally: R3000 deletions data has a GAP for 2018 and 2019 (no PDF was
  recoverable in the original Russell-1 acquisition) -- any ticker whose only
  true R3000 exit happened in 2018/2019 will be MISSED by this overlay and will
  incorrectly remain in the universe until any later recorded event (if any).

  Window: 2016-2023 (8 annual R3000 cycles) is what the archive covers; 2015 is
  explicitly unwired in this repo's own README ("a deliberate methodology
  decision, not something to graft on silently"). This trial therefore uses
  2016-01-01 -> 2024-01-01 (32 quarterly periods) as the "closest achievable
  ~8-year window," not 2015-2023.

VOLATILITY-TARGETING OVERLAY (this trial's actual hypothesis under test):
  At each rebalance, before applying that period's trade, compute the trailing
  realized volatility of the UNSCALED (beta-neutral-only) GP/A portfolio's DAILY
  returns (adjClose-based) over up to the trailing 126 trading days (~6 months).
  Requires >= 60 trading days of prior history before any scaling activates
  (first ~1-2 quarters of the backtest use vol_scale = 1.0, flagged as
  "insufficient trailing history").

  vol_scale = clip(TARGET_VOL / trailing_realized_vol, 0.2, 3.0)

  PRE-REGISTERED, LOCKED BEFORE RUNNING (not tuned after seeing results):
    TARGET_VOL      = 10% annualized
    VOL_WINDOW_DAYS = 126 trading days (~6 months)
    VOL_MIN_DAYS    = 60 trading days minimum before scaling activates
    SCALE_CLIP      = [0.2x, 3.0x]  (same bounds already used for beta-neutral
                       leg scaling elsewhere in this codebase, reused here for
                       consistency rather than inventing a new convention)

  final_long_scale  = long_scale_beta_neutral  x vol_scale
  final_short_scale = short_scale_beta_neutral x vol_scale
  (gross exposure of BOTH legs scaled proportionally, ratio between legs --
  i.e. the beta-neutral property -- is preserved; only the overall gross
  level moves.)

  KNOWN COST-MODEL SIMPLIFICATION (disclosed, not silently absorbed):
  the "standard cost model" (per instructions) charges transaction costs only
  on the fraction of NAMES that rotate in/out of a leg each quarter, not on the
  incremental notional traded when vol_scale changes gross exposure on unchanged
  names. This likely UNDERSTATES real-world cost drag for the vol-managed
  variant specifically (Moreira & Muir's own paper flags vol-timing as adding
  extra turnover). Not fixed here -- flagged as an open item.

DIRECT COMPARISON (isolates the vol-scaling effect from the universe/window
change): this script runs TWO parallel series over the IDENTICAL 2016-2023
window and IDENTICAL Russell-overlaid universe --
  "base"        = beta-neutral GP/A, no vol-targeting (the control)
  "vol-managed" = base + vol-targeting overlay (the treatment)
plus reports the ORIGINAL S3-Q trial (#8, 2015-2021, no Russell overlay) as a
separate historical reference (different window/universe -- not a clean A/B,
shown for context only).
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Reuse S3-Q's infrastructure by import, not reimplementation ──────────────
_GP_DIR = Path(__file__).resolve().parents[1] / "gp"
sys.path.insert(0, str(_GP_DIR))
import run as gp  # noqa: E402  (research/gp/run.py)

RUSSELL_DIR = Path(__file__).resolve().parents[1] / "russell_reconstitution" / "results"
RESULTS     = Path(__file__).resolve().parent / "results"
REGISTRY    = Path(__file__).resolve().parents[1] / "trial_registry.csv"
RESULTS.mkdir(parents=True, exist_ok=True)

# ── Pre-registered vol-targeting parameters (locked before running) ─────────
TARGET_VOL      = 0.10
VOL_WINDOW_DAYS = 126
VOL_MIN_DAYS    = 60
SCALE_CLIP_LOW  = 0.2
SCALE_CLIP_HIGH = 3.0

# ── Window: closest achievable ~8yr window given Russell archive coverage ────
IS_START   = pd.Timestamp("2016-01-01")
IS_END     = pd.Timestamp("2024-01-01")
_all_dates = pd.date_range(IS_START, IS_END, freq="QS")
REBALANCE_DATES = list(_all_dates[_all_dates < IS_END])   # 32 quarterly periods


# =============================================================================
# Russell R3000 confirmed-exit overlay (reuses existing Russell-1/Russell-3
# output files verbatim -- no re-sourcing, no re-parsing of raw PDFs)
# =============================================================================

def load_russell_events() -> pd.DataFrame:
    add = pd.read_csv(RUSSELL_DIR / "russell1_universe.csv", parse_dates=["effective_date"])
    dele = pd.read_csv(RUSSELL_DIR / "russell1_deletions_universe.csv", parse_dates=["effective_date"])
    add  = add[["ticker", "effective_date"]].copy();  add["event"]  = "addition"
    dele = dele[["ticker", "effective_date"]].copy(); dele["event"] = "deletion"
    ev = pd.concat([add, dele], ignore_index=True).sort_values(["ticker", "effective_date"])
    return ev


RUSSELL_EVENTS = load_russell_events()
print(f"[Russell overlay] {len(RUSSELL_EVENTS):,} R3000 events loaded "
      f"({(RUSSELL_EVENTS['event']=='addition').sum():,} additions, "
      f"{(RUSSELL_EVENTS['event']=='deletion').sum():,} deletions), "
      f"years {sorted(pd.concat([pd.read_csv(RUSSELL_DIR/'russell1_universe.csv')['year'], pd.read_csv(RUSSELL_DIR/'russell1_deletions_universe.csv')['year']]).unique())}")


def excluded_tickers_asof(rebalance_date: pd.Timestamp) -> set:
    """Tickers whose most recent confirmed R3000 event at/before this date is a deletion."""
    past = RUSSELL_EVENTS[RUSSELL_EVENTS["effective_date"] <= rebalance_date]
    if past.empty:
        return set()
    latest = past.groupby("ticker").tail(1)
    return set(latest.loc[latest["event"] == "deletion", "ticker"])


def build_universe_overlay(rebalance_date, tiingo_tickers, cik_sic_map):
    base = gp.build_universe(rebalance_date, tiingo_tickers, cik_sic_map)
    excl = excluded_tickers_asof(rebalance_date)
    n_before = len(base)
    filtered = base[~base["ticker"].isin(excl)].copy()
    return filtered, n_before, len(filtered), len(excl)


# =============================================================================
# Daily unscaled-basket return path (adjClose) -- feeds trailing-vol estimate
# =============================================================================

def daily_basket_returns(longs, shorts, lw, sw, long_scale, short_scale,
                          start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    names = sorted(set(longs) | set(shorts))
    frames = {}
    for t in names:
        adj = gp._adj_slice(t, start, end)
        if adj.empty:
            continue
        s = adj.set_index("date")["adjClose"]
        s = s[~s.index.duplicated(keep="last")].sort_index()
        r = s.pct_change().dropna()
        if not r.empty:
            frames[t] = r
    if not frames:
        return pd.Series(dtype=float)
    mat = pd.DataFrame(frames).sort_index()
    w = pd.Series(0.0, index=mat.columns)
    for t in longs:
        if t in w.index:
            w[t] += lw.get(t, 0.0) * long_scale
    for t in shorts:
        if t in w.index:
            w[t] -= sw.get(t, 0.0) * short_scale
    # Missing-day fill is a simplification for the vol PROXY only (not P&L).
    port = (mat.fillna(0.0) * w).sum(axis=1)
    return port


# =============================================================================
# Backtest engine
# =============================================================================

def run_backtest() -> dict:
    print("\n[Step 1] Loading Tiingo universe metadata...")
    tiingo_tickers = gp.load_tiingo_tickers()

    print("\n[Step 2] Loading SEC pre-filter...")
    prefilter_df = gp.load_prefilter()
    survivors        = prefilter_df[prefilter_df["passed"]][["ticker", "cik", "sic"]].copy()
    cik_sic_map      = {r["ticker"]: (int(r["cik"]), r["sic"]) for _, r in survivors.iterrows()}
    survivor_tickers = survivors["ticker"].tolist()
    survivor_ciks    = survivors["cik"].dropna().astype(int).tolist()

    print(f"\n[Step 3a] Preloading raw close prices ({len(survivor_tickers):,} tickers)...")
    gp.preload_all_prices(survivor_tickers)
    print(f"\n[Step 3b] Loading adjClose pickle (beta + vol layer)...")
    gp.preload_all_adj_prices(survivor_tickers)
    print(f"\n[Step 4] Loading GP XBRL facts ({len(survivor_ciks):,} CIKs)...")
    gp.preload_gp_facts(survivor_ciks)

    delisted_map: dict = {}
    for _, r in tiingo_tickers.iterrows():
        if pd.notna(r["endDate"]):
            delisted_map[r["ticker"]] = r["endDate"]

    period_records: list = []
    running_daily = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    prev_long_set: set  = set()
    prev_short_set: set = set()

    n_periods = len(REBALANCE_DATES)
    print(f"\n[Step 5] Running backtest ({n_periods} quarterly periods, "
          f"{IS_START.date()} -> {IS_END.date()})...")

    for i, t in enumerate(REBALANCE_DATES):
        hold_end = REBALANCE_DATES[i + 1] if i + 1 < n_periods else IS_END
        print(f"\n[{i+1:02d}/{n_periods}] {t.date()} -> {hold_end.date()}")

        # ── 1. Universe: base filter + Russell confirmed-exit overlay ────────
        try:
            univ, n_before, n_after, n_excl = build_universe_overlay(t, tiingo_tickers, cik_sic_map)
        except Exception as e:
            print(f"  [ERR] build_universe_overlay: {e}")
            continue
        print(f"  [Universe] base={n_before:,} -> russell_overlay={n_after:,} "
              f"(excluded {n_excl:,} confirmed R3000 exits)")
        if univ.empty:
            print("  [SKIP] empty universe")
            continue

        # ── 2. GP signal (unchanged) ──────────────────────────────────────────
        signals_df, cov = gp.compute_gp_signals(univ, t)
        pct_drop = cov["pct_dropped"]
        print(f"  GP: {cov['n_signal']}/{cov['n_universe']} with signal ({pct_drop:.0f}% dropped)")

        # ── 3. Returns ─────────────────────────────────────────────────────────
        univ_sig = univ.merge(signals_df[["ticker", "gp_signal"]], on="ticker", how="left")
        try:
            returns_df = gp.compute_returns(univ_sig, t, hold_end, delisted_map)
        except Exception as e:
            print(f"  [ERR] compute_returns: {e}")
            continue

        # ── 4. Beta (unchanged) ───────────────────────────────────────────────
        betas = gp.compute_betas(univ, t)

        # ── 5. Portfolio: beta-neutral only (BASE, pre-vol-overlay) ───────────
        (longs, shorts, lw, sw,
         long_scale_bn, short_scale_bn,
         mean_beta_long, mean_beta_short) = gp.build_portfolio(returns_df, betas)
        if not longs:
            print(f"  [SKIP] n<40 after signal filter")
            continue

        # ── 6. Turnover ───────────────────────────────────────────────────────
        if prev_long_set:
            long_turnover  = len(set(longs)  - prev_long_set)  / max(len(longs),  1)
            short_turnover = len(set(shorts) - prev_short_set) / max(len(shorts), 1)
        else:
            long_turnover = short_turnover = 1.0
        prev_long_set, prev_short_set = set(longs), set(shorts)
        avg_turnover = (long_turnover + short_turnover) / 2.0

        # ── 7. Leg returns (raw close, standard cost model) ───────────────────
        long_net,  long_gross,  long_ba,  long_bw  = gp._leg_return_detail(
            returns_df, longs,  lw, is_short=False, turnover=long_turnover)
        short_net, short_gross, short_ba, short_bw = gp._leg_return_detail(
            returns_df, shorts, sw, is_short=True,  turnover=short_turnover)

        # ── 8. BASE combined series (beta-neutral only, no vol overlay) ───────
        ls_net_base   = long_scale_bn * long_net   + short_scale_bn * short_net
        ls_gross_base = long_scale_bn * long_gross + short_scale_bn * short_gross
        ls_cost_base  = long_scale_bn * (long_ba + long_bw) + short_scale_bn * (short_ba + short_bw)

        # ── 9. Trailing realized vol of the UNSCALED basket (strictly < t) ────
        hist = running_daily[running_daily.index < t].tail(VOL_WINDOW_DAYS)
        if len(hist) >= VOL_MIN_DAYS:
            trailing_vol = float(hist.std(ddof=1) * np.sqrt(252))
            vol_scale = (float(np.clip(TARGET_VOL / trailing_vol, SCALE_CLIP_LOW, SCALE_CLIP_HIGH))
                         if trailing_vol > 0 else 1.0)
            vol_scale_active = True
        else:
            trailing_vol = np.nan
            vol_scale = 1.0
            vol_scale_active = False

        final_long_scale  = long_scale_bn  * vol_scale
        final_short_scale = short_scale_bn * vol_scale

        ls_net_vm   = final_long_scale * long_net   + final_short_scale * short_net
        ls_gross_vm = final_long_scale * long_gross + final_short_scale * short_gross
        ls_cost_vm  = final_long_scale * (long_ba + long_bw) + final_short_scale * (short_ba + short_bw)

        # ── 10. EW market return + IC (unaffected by vol overlay) ─────────────
        valid_all = returns_df.dropna(subset=["raw_return"])
        mkt_ret = float(valid_all["raw_return"].mean()) if not valid_all.empty else np.nan

        valid_sig = returns_df.dropna(subset=["gp_signal", "raw_return"])
        if len(valid_sig) >= 10:
            ic, _ = stats.spearmanr(valid_sig["gp_signal"], valid_sig["raw_return"])
        else:
            ic = np.nan

        print(f"  BASE L/S={ls_net_base:+.2%}  VOL-MGD L/S={ls_net_vm:+.2%}  "
              f"trail_vol={trailing_vol if not np.isnan(trailing_vol) else float('nan'):.1%} "
              f"vol_scale={vol_scale:.2f}x  IC={ic:.3f}")

        # ── 11. Advance the daily buffer with THIS period's unscaled basket ────
        daily_q = daily_basket_returns(longs, shorts, lw, sw, long_scale_bn, short_scale_bn, t, hold_end)
        if not daily_q.empty:
            running_daily = pd.concat([running_daily, daily_q]).sort_index()
            running_daily = running_daily[~running_daily.index.duplicated(keep="last")]

        period_records.append({
            "rebalance_date":     t,
            "hold_end":           hold_end,
            "year":               t.year,
            "n_universe_base":    n_before,
            "n_universe_russell": n_after,
            "n_russell_excluded": n_excl,
            "pct_dropped":        pct_drop,
            "long_ret":           long_net,
            "short_ret":          short_net,
            "ls_ret_base":        ls_net_base,
            "ls_gross_base":      ls_gross_base,
            "ls_cost_base":       ls_cost_base,
            "ls_ret_vm":          ls_net_vm,
            "ls_gross_vm":        ls_gross_vm,
            "ls_cost_vm":         ls_cost_vm,
            "trailing_vol":       trailing_vol,
            "vol_scale":          vol_scale,
            "vol_scale_active":   vol_scale_active,
            "long_scale_bn":      long_scale_bn,
            "short_scale_bn":     short_scale_bn,
            "final_long_scale":   final_long_scale,
            "final_short_scale":  final_short_scale,
            "mean_beta_long":     mean_beta_long,
            "mean_beta_short":    mean_beta_short,
            "long_turnover":      long_turnover,
            "short_turnover":     short_turnover,
            "avg_turnover":       avg_turnover,
            "mkt_ret":            mkt_ret,
            "ic":                 ic,
            "n_long":             len(longs),
            "n_short":            len(shorts),
        })

    return {"period_records": period_records}


# =============================================================================
# Metrics (reuse gp.compute_metrics / gp.compute_market_beta / gp.deflated_sharpe_threshold)
# =============================================================================

def compute_metrics(ls_series: pd.Series) -> dict:
    return gp.compute_metrics(ls_series)


def compute_market_beta(ls: pd.Series, mkt: pd.Series) -> dict:
    return gp.compute_market_beta(ls, mkt)


if __name__ == "__main__":
    import json
    results = run_backtest()
    with open(RESULTS / "_raw_period_records.json", "w") as f:
        json.dump([{k: (v.isoformat() if isinstance(v, pd.Timestamp) else
                         (None if isinstance(v, float) and np.isnan(v) else v))
                    for k, v in r.items()} for r in results["period_records"]], f, indent=2)
    print(f"\n[Saved] raw period records -> {RESULTS}/_raw_period_records.json")
