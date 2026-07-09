"""
Russell-3: Boundary-Crossing Subset -- event-study backtest.

Family member of "Russell Reconstitution" (Russell-1, trial #17) -- shares
its DSR slot, does NOT increment N_TRIALS independently. Reuses Russell-1's
entry/exit logic and IWM benchmark unchanged -- this trial tests a different
subset of names, not a re-tuned version of Russell-1.

Per-crossing-event, not per-cycle: unlike Russell-1 (one basket per YEAR),
here each repeat-crosser ticker contributes one event PER YEAR it crosses
(as an addition or a deletion). Entry/exit dates and benchmark window are
still taken from the same year's cycle_calendar (same sourced dates as
Russell-1). Return sign convention is NOT flipped for deletions: we compute
the same long-only-window return (px[exit]/px[entry] - 1) minus the IWM
benchmark for every event, whether addition or deletion. The hypothesis
predicts this abnormal return is POSITIVE for addition events (anticipated
buying) and NEGATIVE for deletion events (anticipated selling) -- reported
separately for exactly this reason.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
TIINGO_CACHE = BASE / "cache" / "tiingo"


def load_price(ticker: str) -> pd.DataFrame:
    f = TIINGO_CACHE / f"{ticker}.csv"
    if not f.exists():
        return pd.DataFrame(columns=["date", "close", "adjClose", "volume"])
    return pd.read_csv(f, parse_dates=["date"])


def price_on_or_after(df, target, col):
    sub = df[df["date"] > target].sort_values("date")
    if sub.empty or pd.isna(sub.iloc[0][col]):
        return None, None
    return sub.iloc[0]["date"], sub.iloc[0][col]


def price_on_or_before(df, target, col):
    sub = df[df["date"] <= target].sort_values("date")
    if sub.empty or pd.isna(sub.iloc[-1][col]):
        return None, None
    return sub.iloc[-1]["date"], sub.iloc[-1][col]


def event_return(ticker, entry_target, exit_target):
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


def caar_stats(abnormal: pd.Series) -> dict:
    n = len(abnormal)
    if n == 0:
        return {"n": 0, "caar": np.nan, "sd": np.nan, "t_stat": np.nan, "p_value": np.nan, "win_rate": np.nan}
    caar = abnormal.mean()
    sd = abnormal.std(ddof=1) if n > 1 else np.nan
    t_stat = caar / (sd / np.sqrt(n)) if (n > 1 and sd > 0) else np.nan
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1)) if (n > 1 and not np.isnan(t_stat)) else np.nan
    win_rate = (abnormal > 0).mean()
    return {"n": n, "caar": caar, "sd": sd, "t_stat": t_stat, "p_value": p_value, "win_rate": win_rate}


def main():
    from cycle_calendar import get_calendar
    cal = get_calendar().set_index("year")
    iwm = load_price("IWM")

    crossing_events = pd.read_csv(RESULTS / "russell3_crossing_events.csv",
                                   parse_dates=["preliminary_posted", "effective_date"])

    event_rows = []
    for _, row in crossing_events.iterrows():
        year = row.year
        entry_target = cal.loc[year, "preliminary_posted"]
        exit_target = cal.loc[year, "effective_date"]

        ret, e_date, x_date, col = event_return(row.ticker, entry_target, exit_target)

        iwm_e_date, iwm_e_px = price_on_or_after(iwm, entry_target, "adjClose")
        iwm_x_date, iwm_x_px = price_on_or_before(iwm, exit_target, "adjClose")
        bench_ret = (iwm_x_px / iwm_e_px - 1.0) if (iwm_e_px and iwm_x_px) else np.nan

        if ret is None or np.isnan(bench_ret):
            event_rows.append({
                "year": year, "ticker": row.ticker, "list_kind": row.list_kind,
                "priced": False, "return": np.nan, "benchmark_ret": np.nan, "abnormal_ret": np.nan,
            })
            continue

        event_rows.append({
            "year": year, "ticker": row.ticker, "list_kind": row.list_kind,
            "priced": True, "entry_date": e_date, "exit_date": x_date, "price_col": col,
            "return": ret, "benchmark_ret": bench_ret, "abnormal_ret": ret - bench_ret,
        })

    events = pd.DataFrame(event_rows)
    events.to_csv(RESULTS / "russell3_event_returns.csv", index=False)

    priced = events[events.priced].copy()
    n_total = len(events)
    n_priced = len(priced)
    print(f"Total crossing events: {n_total} | priced: {n_priced} ({n_priced/n_total:.1%} coverage)")
    print(f"Unique repeat-crosser tickers: {events.ticker.nunique()}")
    print(events.groupby("list_kind").size())

    pooled = caar_stats(priced["abnormal_ret"])
    additions = caar_stats(priced.loc[priced.list_kind == "addition", "abnormal_ret"])
    deletions = caar_stats(priced.loc[priced.list_kind == "deletion", "abnormal_ret"])

    print("\n" + "=" * 90)
    print("RUSSELL-3 RESULTS TABLE (event-study, no annualized Sharpe/Calmar -- non-continuous small-N events)")
    print("=" * 90)
    header = f"{'Subset':<16s}{'N events':>10s}{'CAAR':>10s}{'SD':>10s}{'t-stat':>10s}{'p-value':>10s}{'Win rate':>10s}"
    print(header)

    def _fmt(v, fmt):
        return "n/a" if (v is None or (isinstance(v, float) and np.isnan(v))) else format(v, fmt)

    for label, s in [("Pooled", pooled), ("Additions-only", additions), ("Deletions-only", deletions)]:
        t_str = _fmt(s['t_stat'], "+.3f")
        p_str = _fmt(s['p_value'], ".3f")
        print(f"{label:<16s}{s['n']:>10d}{s['caar']:>+10.4%}{s['sd']:>10.4%}"
              f"{t_str:>10s}{p_str:>10s}{s['win_rate']:>10.1%}")

    # ---- Falsification diagnostics ----
    print("\n" + "=" * 90)
    print("FALSIFICATION DIAGNOSTICS")
    print("=" * 90)

    def concentration_check(df, label):
        if df.empty:
            print(f"{label}: no priced events.")
            return
        by_ticker = df.groupby("ticker")["abnormal_ret"].apply(lambda s: s.abs().sum())
        total = by_ticker.sum()
        top2 = by_ticker.sort_values(ascending=False).head(2)
        frac = top2.sum() / total if total > 0 else np.nan
        print(f"{label}: top-2 tickers ({', '.join(top2.index)}) = {frac:.1%} of total |abnormal return| "
              f"mass ({df.ticker.nunique()} unique tickers, {len(df)} events). "
              f"{'FLAG >40%' if frac > 0.40 else 'OK (<40%)'}")

    concentration_check(priced, "Pooled")
    concentration_check(priced[priced.list_kind == "addition"], "Additions-only")
    concentration_check(priced[priced.list_kind == "deletion"], "Deletions-only")

    n_years_missing_deletions = 2  # 2018, 2019 -- see deletions_calendar.py
    print(f"\nData completeness caveat: no Russell 3000 DELETIONS PDF was recoverable for "
          f"{n_years_missing_deletions} of the 8 in-sample years (2018, 2019) in the original "
          f"Russell-1 acquisition. Any repeat-crosser whose only second leg would have been an "
          f"undetected 2018/2019 deletion is missed -- the true repeat-crosser count and event "
          f"count are both likely modest undercounts.")

    # ---- Comparison to Russell-1 ----
    print("\n" + "=" * 90)
    print("COMPARISON TO RUSSELL-1 (full addition-list, 8 cycles)")
    print("=" * 90)
    r1_caar_gross = 0.006374  # from russell1_cycle_results.csv / RESULTS.md
    r1_t_stat = 0.416
    r1_n = 8
    print(f"Russell-1 (full list, N=8 cycles):        CAAR = {r1_caar_gross:+.4%}, t = {r1_t_stat:+.3f}")
    print(f"Russell-3 pooled (repeat-crossers, N={pooled['n']}): CAAR = {pooled['caar']:+.4%}, "
          f"t = {_fmt(pooled['t_stat'], '+.3f')}")
    print(f"Russell-3 additions-only (N={additions['n']}):        CAAR = {additions['caar']:+.4%}, "
          f"t = {_fmt(additions['t_stat'], '+.3f')}")
    print(f"Russell-3 deletions-only (N={deletions['n']}):        CAAR = {deletions['caar']:+.4%}, "
          f"t = {_fmt(deletions['t_stat'], '+.3f')}")

    materially_different = (
        not np.isnan(additions['t_stat']) and abs(additions['t_stat']) > 1.96
    ) or (
        not np.isnan(deletions['t_stat']) and abs(deletions['t_stat']) > 1.96
    )
    print(f"\n{'Materially different (a subset clears |t|>1.96) from Russell-1 null' if materially_different else 'Looks SIMILARLY NULL to Russell-1 -- boundary-crossing subset does not isolate a cleaner effect'}")

    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"N = {pooled['n']} crossing events across {events.ticker.nunique()} unique repeat-crosser "
          f"tickers (additions: {additions['n']}, deletions: {deletions['n']}).")
    if materially_different:
        print("A subset shows a statistically significant effect -- but see honest-framing note below.")
    else:
        print("No subset clears conventional significance. Boundary-crossing subset does NOT show a "
              "materially different or cleaner effect than Russell-1's full-list null.")
    print("RECOMMENDATION: close the 'Russell Reconstitution' family -- two independent attempts "
          "(full list, boundary-crossing subset) both null/inconclusive; no basis to continue "
          "searching for a variant that works.")

    return {
        "pooled": pooled, "additions": additions, "deletions": deletions,
        "n_unique_tickers": events.ticker.nunique(),
        "materially_different": materially_different,
    }


if __name__ == "__main__":
    main()
