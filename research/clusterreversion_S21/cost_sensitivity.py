"""
Cost-model sensitivity check across the registry's cost-nonviable trials
(S2 PEAD, S4 MAX, S8 residual reversal, S21 cluster reversion).

For each trial, isolate the BID-ASK SPREAD cost component specifically
(holding borrow/securities-lending cost fixed -- that's a separate rate
assumption, not what this investigation is about), rescale it by a
multiplier k in {0.25, 0.5, 1.0, 1.5, 2.0}, and recompute net Sharpe/CALMAR
at each k to see whether a plausible bid-ask recalibration would change any
trial's FAIL verdict.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"


def compute_metrics(ls_series: pd.Series, periods_per_year: int = 12) -> dict:
    ls = ls_series.dropna().reset_index(drop=True)
    n = len(ls)
    if n < 2:
        return {"ann_ret": np.nan, "sharpe": np.nan, "calmar": np.nan, "max_dd": np.nan, "n": n}
    cum = (1 + ls).cumprod()
    total = float(cum.iloc[-1] - 1)
    ann_ret = float((1 + total) ** (periods_per_year / n) - 1)
    mu, sigma = float(ls.mean()), float(ls.std(ddof=1))
    sharpe = float((mu / sigma) * np.sqrt(periods_per_year)) if sigma > 0 else np.nan
    roll_mx = cum.cummax()
    dd_s = (cum - roll_mx) / roll_mx
    max_dd = float(dd_s.min())
    calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else np.nan
    return {"ann_ret": ann_ret, "sharpe": sharpe, "calmar": calmar, "max_dd": max_dd, "n": n}


def sensitivity(tag: str, period_df: pd.DataFrame, periods_per_year: int, ks=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0)) -> pd.DataFrame:
    """period_df must have columns: gross, ba, bw (per period, already at the
    target compounding frequency -- one row per period)."""
    rows = []
    for k in ks:
        net_k = period_df["gross"] - k * period_df["ba"] - period_df["bw"]
        m = compute_metrics(net_k, periods_per_year)
        rows.append({"trial": tag, "k": k, **m})
    return pd.DataFrame(rows)


# =============================================================================
# S2 PEAD -- daily series, resample to monthly (matches registry's own n=72)
# =============================================================================
pead_daily = pd.read_csv("research/pead/results/pead_daily_returns.csv")
pead_daily["date"] = pd.to_datetime(pead_daily["date"])
pead_daily["month"] = pead_daily["date"].dt.to_period("M")
pead_monthly = pead_daily.groupby("month").agg(
    gross=("ls_gross", lambda s: float((1 + s).prod() - 1)),
    ba=("ls_ba", "sum"), bw=("ls_bw", "sum")).reset_index()

# =============================================================================
# S4 MAX -- already monthly
# =============================================================================
max_df = pd.read_csv("research/max/results/max_period_returns.csv")
max_monthly = max_df.rename(columns={"ls_gross": "gross", "ls_ba_cost": "ba", "ls_borrow_cost": "bw"})[["gross", "ba", "bw"]]

# =============================================================================
# S8 residual reversal -- already monthly
# =============================================================================
s8_df = pd.read_csv("research/residual_reversal/results/s8_period_returns.csv")
s8_monthly = s8_df.rename(columns={"ls_gross": "gross", "ls_ba_cost": "ba", "ls_borrow_cost": "bw"})[["gross", "ba", "bw"]]

# =============================================================================
# S21 cluster reversion -- daily book, resample to monthly. Borrow is
# negligible here (5-day holds; verified: R._borrow(mcap)/365*0.5 <=~0.0000274
# per day) so "cost" is treated as ~100% bid-ask for the sensitivity split --
# ba = cost, bw = 0. This is a close, disclosed approximation, not exact.
# =============================================================================
s21_daily = pd.read_csv(RESULTS / "daily_book_returns_basket.csv", parse_dates=["date"])
s21_daily["month"] = s21_daily["date"].dt.to_period("M")
s21_monthly = s21_daily.groupby("month").agg(
    gross=("book_gross", lambda s: float((1 + s).prod() - 1)),
    ba=("book_cost", "sum")).reset_index()
s21_monthly["bw"] = 0.0

print("=" * 100)
print("COST-MODEL SENSITIVITY: net Sharpe/CALMAR at k x current bid-ask spread assumption (borrow held fixed)")
print("=" * 100)

all_results = []
for tag, df in [("S2-PEAD", pead_monthly), ("S4-MAX", max_monthly), ("S8-ResidRev", s8_monthly), ("S21-basket", s21_monthly)]:
    res = sensitivity(tag, df, periods_per_year=12)
    all_results.append(res)
    print(f"\n--- {tag} (n={res['n'].iloc[0]} monthly periods) ---")
    print(res[["k", "ann_ret", "sharpe", "calmar", "max_dd"]].to_string(index=False,
          formatters={"ann_ret": "{:.2%}".format, "sharpe": "{:.3f}".format,
                      "calmar": "{:.3f}".format, "max_dd": "{:.2%}".format}))

full = pd.concat(all_results, ignore_index=True)
full.to_csv(RESULTS / "cost_sensitivity.csv", index=False)
print(f"\n[Saved] -> {RESULTS}/cost_sensitivity.csv")

print("\n" + "=" * 100)
print("VERDICT CHECK: does any trial cross Sharpe>=0.8 at ANY plausible k (0.25x-2x)?")
print("=" * 100)
for tag in full["trial"].unique():
    sub = full[full["trial"] == tag]
    max_sharpe = sub["sharpe"].max()
    k_at_max = sub.loc[sub["sharpe"].idxmax(), "k"]
    crosses = bool((sub["sharpe"] >= 0.8).any())
    print(f"{tag}: max Sharpe across k-range = {max_sharpe:.3f} (at k={k_at_max}) "
          f"-- {'CROSSES 0.8' if crosses else 'never crosses 0.8, even at k=0.25x (spread 1/4 of current)'}")
