"""
Tests whether the three external nowcast/consensus signals meaningfully
lead Part 1's coincident regime label. Methodology locked in README.md's
"External nowcast/consensus signals" section BEFORE this was run. Verifies
existing infrastructure / evaluates a build decision -- NOT a DSR trial,
not logged to trial_registry.csv or CORRECTED_TRIAL_REGISTRY.md.

Four tests (GDPNow vs growth, Cleveland Nowcast vs inflation, SPF-CPI
surprise vs inflation, SPF-RGDP surprise vs growth), each an expanding-
window walk-forward with an exact two-sided binomial test against p=0.5
(a direct forecast of the same quantity -> per-period binary match, not
Part 2's onset/lead-time framing, so the permutation machinery there
doesn't apply here -- binomial is the correct, simpler tool).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fetch import fetch_all, first_release_series
from classify import build_growth_direction, build_inflation_direction
from external_fetch import (
    fetch_gdpnow_vintages, fetch_cleveland_raw, parse_cleveland_nowcast,
    fetch_spf_raw, parse_spf_cpi, parse_spf_rgdp,
)

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

GDPNOW_ANCHOR = pd.Timestamp("2015-01-01")
CLEVELAND_ANCHOR = pd.Timestamp("2018-01-01")
SPF_ANCHOR = pd.Timestamp("2005-01-01")
MIN_TEST_SIZE = 15
SIG_LEVEL = 0.05


def _direction_from_level(s: pd.Series) -> pd.Series:
    """accel if value > prior value, decel if <, NaN if tie/first obs -- a
    plain period-over-period direction call, the same convention Part 1
    uses for its own smoothed series, applied here to each external
    source's own value sequence."""
    diff = s.diff()
    return diff.apply(lambda x: "accel" if x > 0 else ("decel" if x < 0 else np.nan))


def binomial_walkforward(df: pd.DataFrame, anchor: pd.Timestamp, label: str) -> dict:
    """df must have columns: period, external_direction, part1_direction."""
    d = df[df["period"] >= anchor].dropna(subset=["external_direction", "part1_direction"]).copy()
    d = d[d["external_direction"].isin(["accel", "decel"]) & d["part1_direction"].isin(["accel", "decel"])]
    n = len(d)
    if n == 0:
        return {"label": label, "n": 0, "matches": 0, "hit_rate": None,
                "p_value": None, "verdict": "NO DATA", "detail": d}
    matches = int((d["external_direction"] == d["part1_direction"]).sum())
    hit_rate = matches / n
    p_value = stats.binomtest(matches, n, p=0.5, alternative="two-sided").pvalue
    if n < MIN_TEST_SIZE:
        verdict = f"INCONCLUSIVE (n={n} < {MIN_TEST_SIZE} minimum test size)"
    elif p_value < SIG_LEVEL:
        verdict = f"SIGNIFICANT (p={p_value:.4f}) -- clears the promotion bar"
    else:
        verdict = f"NOT significant (p={p_value:.4f})"
    return {"label": label, "n": n, "matches": matches, "hit_rate": hit_rate,
            "p_value": p_value, "verdict": verdict, "detail": d}


# ── GDPNow vs Part 1 growth direction ─────────────────────────────────────

def test_gdpnow(data: dict) -> dict:
    gdpnow_v = fetch_gdpnow_vintages()
    gdp_first = first_release_series(data["vintages"]["GDPC1"])
    gdp_release_dates = gdp_first.attrs["realtime_start"]

    rows = []
    for q, release_dt in gdp_release_dates.items():
        sub = gdpnow_v[(gdpnow_v["date"] == q) & (gdpnow_v["realtime_start"] < release_dt)]
        if sub.empty:
            continue
        last = sub.sort_values("realtime_start").iloc[-1]
        rows.append({"period": q, "terminal_gdpnow": last["value"]})
    terminal_df = pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    terminal_df["external_direction"] = _direction_from_level(terminal_df["terminal_gdpnow"])

    part1 = build_growth_direction(data["vintages"]["GDPC1"])[["period", "growth_direction"]]
    part1 = part1.rename(columns={"growth_direction": "part1_direction"})

    merged = terminal_df.merge(part1, on="period", how="inner")
    return binomial_walkforward(merged, GDPNOW_ANCHOR, "GDPNow vs Part 1 growth direction")


# ── Cleveland Nowcast vs Part 1 inflation direction ───────────────────────

def test_cleveland(data: dict) -> dict:
    raw = fetch_cleveland_raw()
    cleveland = parse_cleveland_nowcast(raw, series_name="Core CPI Inflation")
    cleveland = cleveland.rename(columns={"target_month": "period"})
    cleveland["external_direction"] = _direction_from_level(cleveland["terminal_nowcast"])

    part1 = build_inflation_direction(data["vintages"]["CPILFESL"])[["period", "inflation_direction"]]
    part1 = part1.rename(columns={"inflation_direction": "part1_direction"})

    merged = cleveland.merge(part1, on="period", how="inner")
    return binomial_walkforward(merged, CLEVELAND_ANCHOR, "Cleveland Nowcast vs Part 1 inflation direction")


# ── SPF consensus surprise vs Part 1 (both CPI and RGDP) ──────────────────

def _quarterly_avg_index(vintage_df: pd.DataFrame) -> pd.Series:
    """Quarterly-average level of a monthly first-release index series
    (SPF's own CPI convention is an annualized q/q rate on the quarterly-
    average index, not a single month's level)."""
    s = first_release_series(vintage_df)
    q = s.resample("QS").mean()
    return q


def test_spf_cpi(data: dict) -> dict:
    level_path, _ = fetch_spf_raw()
    spf = parse_spf_cpi(level_path)[["period", "CORECPI2"]].copy()
    spf["forecast_for_next_q"] = spf["CORECPI2"]
    spf["period_forecasted"] = spf["period"] + pd.DateOffset(months=3)

    cpi_q_avg = _quarterly_avg_index(data["vintages"]["CPILFESL"])
    actual_qoq_ann = (cpi_q_avg / cpi_q_avg.shift(1)) ** 4 - 1.0
    actual_df = pd.DataFrame({"period": actual_qoq_ann.index, "actual": actual_qoq_ann.values * 100})

    merged = spf.merge(actual_df, left_on="period_forecasted", right_on="period", how="inner")
    merged["surprise"] = merged["actual"] - merged["forecast_for_next_q"]
    merged["external_direction"] = merged["surprise"].apply(
        lambda x: "accel" if x > 0 else ("decel" if x < 0 else np.nan)
    )
    merged["period"] = merged["period_forecasted"]

    # Part 1's inflation_direction is monthly -- use the quarter's last month
    # as a quarterly proxy (disclosed simplification, see README).
    part1_monthly = build_inflation_direction(data["vintages"]["CPILFESL"])[["period", "inflation_direction"]]
    part1_q = part1_monthly.sort_values("period").groupby(
        part1_monthly["period"].dt.to_period("Q")
    ).last()[["period", "inflation_direction"]].reset_index(drop=True)
    part1_q = part1_q.rename(columns={"inflation_direction": "part1_direction"})
    part1_q["period"] = part1_q["period"].dt.to_period("Q").dt.to_timestamp()

    merged["period"] = merged["period"].dt.to_period("Q").dt.to_timestamp()
    merged = merged.merge(part1_q, on="period", how="inner")
    return binomial_walkforward(merged, SPF_ANCHOR, "SPF CPI-surprise vs Part 1 inflation direction (quarterly proxy)")


def test_spf_rgdp(data: dict) -> dict:
    _, growth_path = fetch_spf_raw()
    spf = parse_spf_rgdp(growth_path)[["period", "drgdp2"]].copy()
    spf["period_forecasted"] = spf["period"] + pd.DateOffset(months=3)

    gdp_first = first_release_series(data["vintages"]["GDPC1"])
    actual_qoq_ann = (gdp_first / gdp_first.shift(1)) ** 4 - 1.0
    actual_df = pd.DataFrame({"period": actual_qoq_ann.index, "actual": actual_qoq_ann.values * 100})

    merged = spf.merge(actual_df, left_on="period_forecasted", right_on="period", how="inner")
    merged["surprise"] = merged["actual"] - merged["drgdp2"]
    merged["external_direction"] = merged["surprise"].apply(
        lambda x: "accel" if x > 0 else ("decel" if x < 0 else np.nan)
    )
    merged["period"] = merged["period_forecasted"]

    part1 = build_growth_direction(data["vintages"]["GDPC1"])[["period", "growth_direction"]]
    part1 = part1.rename(columns={"growth_direction": "part1_direction"})

    merged = merged.merge(part1, on="period", how="inner")
    return binomial_walkforward(merged, SPF_ANCHOR, "SPF RGDP-surprise vs Part 1 growth direction")


def main():
    data = fetch_all()

    results = [
        test_gdpnow(data),
        test_cleveland(data),
        test_spf_cpi(data),
        test_spf_rgdp(data),
    ]

    lines = ["# External Nowcast/Consensus Signal Test (2026-07-18)\n"]
    lines.append(
        "Verification of a build decision, not a DSR trial -- not logged to "
        "the registry. Methodology locked in README.md before running. "
        "Promotion criterion (locked before seeing results): p < 0.05 AND "
        "n >= 15 walk-forward test periods.\n"
    )

    summary_rows = []
    for r in results:
        lines.append(f"\n## {r['label']}\n")
        if r["n"] == 0:
            lines.append("No overlapping data after merging with Part 1 -- cannot test.\n")
            summary_rows.append({"signal": r["label"], "n": 0, "hit_rate": None,
                                  "p_value": None, "verdict": r["verdict"]})
            continue
        lines.append(
            f"- n={r['n']} periods tested (walk-forward, anchor applied) | "
            f"matches={r['matches']} | hit rate={r['hit_rate']*100:.1f}% | "
            f"binomial p={r['p_value']:.4f}\n"
        )
        lines.append(f"- **Verdict: {r['verdict']}**\n")
        summary_rows.append({
            "signal": r["label"], "n": r["n"],
            "hit_rate_pct": round(r["hit_rate"] * 100, 1),
            "p_value": round(r["p_value"], 4), "verdict": r["verdict"],
        })

    lines.append("\n## Summary table\n\n")
    lines.append(pd.DataFrame(summary_rows).to_string(index=False) + "\n")

    out_path = RESULTS / "external_signals_report.md"
    out_path.write_text("".join(lines))
    print("".join(lines))
    print(f"\nSaved -> {out_path}")
    return results


if __name__ == "__main__":
    main()
