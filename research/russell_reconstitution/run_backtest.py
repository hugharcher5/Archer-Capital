"""
Phase 4-5: Russell-1 event study backtest, CAAR stats, falsification
diagnostics, and trial-registry logging.

Design (event-study, not a Sharpe/DSR-gated L/S strategy -- same family of
design as H1 livestock event study):
  - One cycle per year, 2016-2023 (8 cycles, in-sample; 2024-2025 reserved
    untouched holdout per pre-registration).
  - Entry: first trading day strictly after the preliminary list was
    publicly posted that year (T0 sourced per-year in cycle_calendar.py).
  - Exit: the actual historical effective date for that year.
  - Position: equal-weight, long-only basket of ALL confirmed Russell 3000
    additions that cycle (Russell 3000 = R1000 union R2000).
  - Benchmark: IWM (iShares Russell 2000 ETF) adjClose return over the
    identical [entry, exit] window -- isolates whether additions beat the
    broad small-cap market over that specific window, not just rode a
    rising market. (Cap-weighted, not equal-weighted -- see README caveat.)
  - Cost model: ONE entry + ONE exit per name per cycle (buy-and-hold, no
    rebalancing) -- flat 25bps/leg (50bps round-trip) institutional
    bid-ask assumption. Russell-eligible names are index-liquidity-screened,
    so this is deliberately lighter than the $100M-$2B cap-tier formula's
    100bps ceiling used elsewhere in the program; see README for rationale.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
TIINGO_CACHE = BASE / "cache" / "tiingo"

SPREAD_PER_LEG = 0.0025  # 25 bps; flat, see module docstring


def load_price(ticker: str) -> pd.DataFrame:
    f = TIINGO_CACHE / f"{ticker}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])
    df = pd.read_csv(f, parse_dates=["date"])
    return df


def price_on_or_after(df: pd.DataFrame, target: pd.Timestamp, col: str):
    sub = df[df["date"] > target].sort_values("date")
    if sub.empty or pd.isna(sub.iloc[0][col]):
        return None, None
    return sub.iloc[0]["date"], sub.iloc[0][col]


def price_on_or_before(df: pd.DataFrame, target: pd.Timestamp, col: str):
    sub = df[df["date"] <= target].sort_values("date")
    if sub.empty or pd.isna(sub.iloc[-1][col]):
        return None, None
    return sub.iloc[-1]["date"], sub.iloc[-1][col]


def ticker_return(ticker: str, entry_target: pd.Timestamp, exit_target: pd.Timestamp):
    """Return (ret, entry_date_used, exit_date_used, col_used) or (None,...) if unpriceable."""
    df = load_price(ticker)
    if df.empty:
        return None, None, None, None
    for col in ["adjClose", "close"]:
        if col not in df.columns:
            continue
        e_date, e_px = price_on_or_after(df, entry_target, col)
        x_date, x_px = price_on_or_before(df, exit_target, col)
        if e_date is None or x_date is None or x_date <= e_date:
            continue
        if e_px is None or e_px <= 0:
            continue
        return (x_px / e_px - 1.0), e_date, x_date, col
    return None, None, None, None


def main():
    from cycle_calendar import get_calendar
    universe = pd.read_csv(RESULTS / "russell1_universe.csv", parse_dates=["preliminary_posted", "effective_date"])
    cal = get_calendar()

    prefilter = pd.read_csv(BASE.parents[1] / "research" / "mgmt_pit" / "cache" / "sec_prefilter.csv")
    core_universe_tickers = set(prefilter.loc[prefilter["passed"], "ticker"])

    iwm = load_price("IWM")

    cycle_rows = []
    ticker_rows = []
    fallback_close_count = 0

    for _, cyc in cal.iterrows():
        year = cyc.year
        sub = universe[universe.year == year]
        entry_target = cyc.preliminary_posted
        exit_target = cyc.effective_date

        # Benchmark window: use IWM's own trading calendar for entry/exit dates
        iwm_e_date, iwm_e_px = price_on_or_after(iwm, entry_target, "adjClose")
        iwm_x_date, iwm_x_px = price_on_or_before(iwm, exit_target, "adjClose")
        bench_ret = (iwm_x_px / iwm_e_px - 1.0) if (iwm_e_px and iwm_x_px) else np.nan

        rets = []
        n_total = len(sub)
        n_priced = 0
        n_in_core = 0
        for _, row in sub.iterrows():
            ret, e_date, x_date, col = ticker_return(row.ticker, entry_target, exit_target)
            if row.ticker in core_universe_tickers:
                n_in_core += 1
            if ret is None:
                continue
            n_priced += 1
            if col == "close":
                fallback_close_count += 1
            rets.append(ret)
            ticker_rows.append({
                "year": year, "ticker": row.ticker, "company": row.company,
                "entry_date": e_date, "exit_date": x_date, "return": ret,
                "price_col": col,
            })

        gross_basket_ret = float(np.mean(rets)) if rets else np.nan
        net_basket_ret = gross_basket_ret - 2 * SPREAD_PER_LEG if not np.isnan(gross_basket_ret) else np.nan
        abnormal_gross = gross_basket_ret - bench_ret if not np.isnan(gross_basket_ret) else np.nan
        abnormal_net = net_basket_ret - bench_ret if not np.isnan(net_basket_ret) else np.nan

        cycle_rows.append({
            "year": year,
            "entry_date": iwm_e_date, "exit_date": iwm_x_date,
            "n_confirmed_additions": n_total,
            "n_priced": n_priced,
            "coverage_pct": n_priced / n_total if n_total else np.nan,
            "n_in_core_100m_2b_universe": n_in_core,
            "core_universe_frac": n_in_core / n_total if n_total else np.nan,
            "basket_gross_ret": gross_basket_ret,
            "basket_net_ret": net_basket_ret,
            "benchmark_ret_IWM": bench_ret,
            "abnormal_ret_gross": abnormal_gross,
            "abnormal_ret_net": abnormal_net,
            "hold_calendar_days": (iwm_x_date - iwm_e_date).days if (iwm_e_date is not None and iwm_x_date is not None) else np.nan,
        })

    cycles = pd.DataFrame(cycle_rows)
    tickers_df = pd.DataFrame(ticker_rows)
    cycles.to_csv(RESULTS / "russell1_cycle_results.csv", index=False)
    tickers_df.to_csv(RESULTS / "russell1_ticker_returns.csv", index=False)

    print("=" * 100)
    print("PER-CYCLE RESULTS")
    print("=" * 100)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(cycles[["year", "entry_date", "exit_date", "n_confirmed_additions", "n_priced",
                       "coverage_pct", "basket_gross_ret", "benchmark_ret_IWM",
                       "abnormal_ret_gross", "abnormal_ret_net"]].to_string(index=False))

    valid = cycles.dropna(subset=["abnormal_ret_gross"])
    n = len(valid)
    caar_gross = valid["abnormal_ret_gross"].mean()
    caar_net = valid["abnormal_ret_net"].mean()
    sd_gross = valid["abnormal_ret_gross"].std(ddof=1)
    t_stat_gross = caar_gross / (sd_gross / np.sqrt(n)) if sd_gross > 0 else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat_gross), df=n - 1)) if not np.isnan(t_stat_gross) else np.nan
    win_rate = (valid["abnormal_ret_gross"] > 0).mean()

    print("\n" + "=" * 100)
    print("CAAR SUMMARY (across %d cycles)" % n)
    print("=" * 100)
    print(f"CAAR (gross, abnormal return)  : {caar_gross:+.4%}")
    print(f"CAAR (net of 50bps round-trip) : {caar_net:+.4%}")
    print(f"Cross-cycle std dev            : {sd_gross:.4%}")
    print(f"t-stat (gross)                 : {t_stat_gross:+.3f}")
    print(f"two-sided p-value              : {p_value:.4f}")
    print(f"Win rate                       : {win_rate:.1%} ({int((valid['abnormal_ret_gross']>0).sum())}/{n})")
    print(f"Fallback-to-raw-close tickers  : {fallback_close_count}")
    print(f"Mean coverage (priced/confirmed): {valid['coverage_pct'].mean():.1%}")
    print(f"Mean fraction inside $100M-$2B core universe: {valid['core_universe_frac'].mean():.1%}")

    # ---- Falsification diagnostics ----
    print("\n" + "=" * 100)
    print("FALSIFICATION DIAGNOSTICS")
    print("=" * 100)

    # (1) Concentration check
    abs_ab = valid["abnormal_ret_gross"].abs()
    total_abs = abs_ab.sum()
    max_year = valid.loc[abs_ab.idxmax(), "year"]
    max_frac = abs_ab.max() / total_abs if total_abs > 0 else np.nan
    print(f"Concentration: largest single-cycle |abnormal return| is {max_frac:.1%} of total "
          f"(year {max_year}). {'FLAG >40%' if max_frac > 0.40 else 'OK (<40%)'}")

    # (2) Decay-over-time check: correlation of abnormal return with year
    if n >= 4:
        corr = np.corrcoef(valid["year"], valid["abnormal_ret_gross"])[0, 1]
        print(f"Decay check: corr(year, abnormal_ret) = {corr:+.3f} "
              f"({'declining over time (crowded out)' if corr < -0.2 else 'flat/no clear decay' if abs(corr) <= 0.2 else 'INCREASING over time'})")

    # (3) Sign-convention audit
    print("Sign-convention audit: basket is built exclusively from *additions* files "
          "(list_kind='additions' filter applied at build_universe.py); no deletions "
          "were used to construct entries. PASS by construction.")

    # (4) Turnover / cost-structure comparison
    print("\n" + "=" * 100)
    print("COST-STRUCTURE COMPARISON vs. prior high-turnover trials")
    print("=" * 100)
    print(f"Russell-1: 1 entry + 1 exit per name per YEAR (buy-and-hold the window).")
    print(f"  Flat cost assumption: {2*SPREAD_PER_LEG*10000:.0f} bps round-trip per name, charged once.")
    print(f"  Prior comparables (annualized cost drag): S2 PEAD 4,079 bps/yr, S4 MAX 2,202 bps/yr, "
          f"S8 residual reversal 761 bps/yr (at 21%/mo turnover). Russell-1's structural turnover "
          f"is a small fraction of these by design (one holding period per year, not continuous rebalancing).")

    verdict_lines = []
    real_effect = t_stat_gross > 1.96 and caar_gross > 0
    verdict_lines.append(f"CAAR = {caar_gross:+.4%} across {n} cycles, t-stat = {t_stat_gross:+.3f}, "
                          f"win rate = {win_rate:.1%}.")
    if real_effect:
        verdict_lines.append("VERDICT: statistically significant positive pre-effective-date drift found.")
    else:
        verdict_lines.append("VERDICT: NOT statistically significant at the 5% level "
                              "(or wrong-signed) -- no reliable pre-effective-date drift detected in-sample.")
    if real_effect and corr < -0.2:
        verdict_lines.append("Effect shows signs of decaying over the in-sample years (consistent with crowding-out).")
    elif real_effect:
        verdict_lines.append("Effect does not show a clear decay pattern over the in-sample years.")
    if real_effect:
        verdict_lines.append("RECOMMENDATION: unlock the 2024-2025 holdout as the next step before any live use.")
    else:
        verdict_lines.append("RECOMMENDATION: do NOT unlock the 2024-2025 holdout -- baseline shows nothing.")

    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    for line in verdict_lines:
        print(line)

    return {
        "n": n, "caar_gross": caar_gross, "caar_net": caar_net, "t_stat": t_stat_gross,
        "p_value": p_value, "win_rate": win_rate, "sd_gross": sd_gross,
        "concentration_frac": max_frac, "decay_corr": corr if n >= 4 else np.nan,
        "verdict_lines": verdict_lines,
    }


if __name__ == "__main__":
    main()
