"""
Tests whether following the top-ranked SPF individual forecaster(s) beats
the full-panel consensus already tested (SPF RGDP-surprise: n=85, hit rate
63.5%, p=0.0165, did NOT survive Bonferroni(4)). Methodology locked in
README.md's "Individual-forecaster accuracy within SPF" section BEFORE
this was run. Verification of a build decision, not a DSR trial -- not
logged to trial_registry.csv or CORRECTED_TRIAL_REGISTRY.md.

The core risk this guards against: picking "the best forecaster" using
full-sample (hindsight) accuracy and then backtesting them is circular.
Every ranking used to pick a forecaster for quarter T here uses ONLY
(forecaster, forecast, actual) pairs whose actual outcome was already
publicly released, strictly before quarter T's own forecast-setting
survey's TRUE DEADLINE DATE (real dates from Philly Fed's own release-date
file, not an assumed quarter-count lag).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fetch import fetch_all, first_release_series
from classify import build_growth_direction
from external_fetch import (
    fetch_spf_microdata_raw, parse_spf_individual_rgdp,
    fetch_spf_release_dates_raw, parse_spf_release_dates,
)
from external_test import binomial_walkforward, SPF_ANCHOR

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MIN_SUBMISSIONS = 20
TOP_N_SINGLE = 1
TOP_N_GROUP = 5
FAMILY_SIZE = 6  # GDPNow, Cleveland, SPF-CPI, SPF-RGDP, SPF-TOP1, SPF-TOP5AVG
BONFERRONI_6 = 0.05 / FAMILY_SIZE


def _actual_growth_by_quarter(data: dict) -> pd.DataFrame:
    gdp_first = first_release_series(data["vintages"]["GDPC1"])
    actual_qoq_ann = ((gdp_first / gdp_first.shift(1)) ** 4 - 1.0) * 100
    release_dates = gdp_first.attrs["realtime_start"]
    return pd.DataFrame({
        "period": actual_qoq_ann.index,
        "actual": actual_qoq_ann.values,
        "actual_release_date": release_dates.values,
    })


def _build_resolved_errors(indiv: pd.DataFrame, actual_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (ID, period_forecasted): forecast, actual, error, and the
    date the actual became public (needed for point-in-time ranking gates)."""
    merged = indiv.merge(
        actual_df, left_on="period_forecasted", right_on="period", how="inner",
        suffixes=("", "_actual"),
    )
    merged["error"] = merged["actual"] - merged["forecast_growth_next_q"]
    merged["abs_error"] = merged["error"].abs()
    return merged[["ID", "period", "period_forecasted", "forecast_growth_next_q",
                   "actual", "abs_error", "actual_release_date"]]


def point_in_time_ranking(resolved: pd.DataFrame, survey_deadline: pd.Timestamp) -> pd.DataFrame:
    """MAE per forecaster using ONLY pairs whose actual was public before
    `survey_deadline` -- the real no-look-ahead gate."""
    eligible = resolved[resolved["actual_release_date"] < survey_deadline]
    counts = eligible.groupby("ID").size()
    qualifying_ids = counts[counts >= MIN_SUBMISSIONS].index
    mae = eligible[eligible["ID"].isin(qualifying_ids)].groupby("ID")["abs_error"].mean()
    return mae.sort_values()  # ascending: best (lowest MAE) first


def full_sample_hindsight_ranking(resolved: pd.DataFrame) -> pd.Series:
    """The WRONG, circular way: rank by MAE computed over ALL history,
    including periods that hadn't happened yet at the time of any given
    pick. Computed explicitly to show the hindsight-bias gap, not hidden."""
    counts = resolved.groupby("ID").size()
    qualifying_ids = counts[counts >= MIN_SUBMISSIONS].index
    mae = resolved[resolved["ID"].isin(qualifying_ids)].groupby("ID")["abs_error"].mean()
    return mae.sort_values()


def build_signal(indiv: pd.DataFrame, resolved: pd.DataFrame, release_dates: pd.DataFrame,
                  part1: pd.DataFrame, use_hindsight: bool, static_ranking: pd.Series = None) -> dict:
    """
    For each test quarter T (= period_forecasted), determine the survey
    quarter (T's period_forecasted minus 3mo = the survey's own period),
    look up its true deadline date, rank forecasters (point-in-time unless
    use_hindsight), and build the TOP-1 and TOP-5-AVG signals.
    """
    test_quarters = sorted(part1["period"].unique())
    top1_rows, top5_rows = [], []

    for T in test_quarters:
        survey_period = pd.Timestamp(T) - pd.DateOffset(months=3)
        deadline_row = release_dates[release_dates["period"] == survey_period]
        if deadline_row.empty:
            continue  # no confirmed survey date this far back/forward
        deadline = deadline_row.iloc[0]["deadline_date"]

        if use_hindsight:
            ranking = static_ranking
        else:
            ranking = point_in_time_ranking(resolved, deadline)
        if ranking is None or ranking.empty:
            continue

        this_q_submissions = indiv[indiv["period"] == survey_period].set_index("ID")["forecast_growth_next_q"]

        # TOP-1: walk down the ranking until a forecaster who submitted this quarter.
        top1_forecast = None
        for fid in ranking.index:
            if fid in this_q_submissions.index:
                top1_forecast = this_q_submissions[fid]
                break
        if top1_forecast is not None:
            top1_rows.append({"period": T, "forecast": top1_forecast})

        # TOP-5-AVG: average whichever of the top-5-ranked forecasters submitted.
        top5_ids = [fid for fid in ranking.index[:TOP_N_GROUP] if fid in this_q_submissions.index]
        if top5_ids:
            top5_rows.append({"period": T, "forecast": this_q_submissions[top5_ids].mean()})

    return {"top1": pd.DataFrame(top1_rows), "top5": pd.DataFrame(top5_rows)}


def main():
    data = fetch_all()

    microdata_path = fetch_spf_microdata_raw()
    indiv = parse_spf_individual_rgdp(microdata_path)

    release_dates_path = fetch_spf_release_dates_raw()
    release_dates = parse_spf_release_dates(release_dates_path)

    actual_df = _actual_growth_by_quarter(data)
    resolved = _build_resolved_errors(indiv, actual_df)

    part1_dir = build_growth_direction(data["vintages"]["GDPC1"])[["period", "growth_direction"]]
    part1_dir = part1_dir.rename(columns={"growth_direction": "part1_direction"})

    lines = ["# Individual-Forecaster Accuracy Within SPF (2026-07-19)\n"]
    lines.append(
        "Verification of a build decision, not a DSR trial -- not logged to "
        "the registry. Methodology locked in README.md before running.\n"
    )

    # ── Dispersion check: how much does forecaster identity vary? ─────────
    full_mae = full_sample_hindsight_ranking(resolved)
    lines.append(f"\n## Cross-sectional MAE dispersion ({len(full_mae)} forecasters with >={MIN_SUBMISSIONS}+ submissions)\n")
    lines.append(
        f"\nMean MAE={full_mae.mean():.3f} | median={full_mae.median():.3f} | "
        f"std={full_mae.std():.3f} | IQR=[{full_mae.quantile(.25):.3f}, {full_mae.quantile(.75):.3f}] | "
        f"range=[{full_mae.min():.3f}, {full_mae.max():.3f}]\n"
    )
    spread_verdict = (
        "MEANINGFUL spread -- forecaster identity is not interchangeable"
        if full_mae.std() / full_mae.mean() > 0.15
        else "TIGHT clustering -- forecaster identity barely matters, most are similarly accurate"
    )
    lines.append(f"**Read: {spread_verdict}** (coefficient of variation = {full_mae.std()/full_mae.mean():.2f})\n")

    # ── Genuine point-in-time signals (the real test) ──────────────────────
    pit_signals = build_signal(indiv, resolved, release_dates, part1_dir, use_hindsight=False)

    results = {}
    for name, sig_df in [("SPF TOP-1 forecaster (point-in-time)", pit_signals["top1"]),
                          ("SPF TOP-5-AVG forecasters (point-in-time)", pit_signals["top5"])]:
        if sig_df.empty:
            results[name] = {"label": name, "n": 0, "verdict": "NO DATA"}
            continue
        merged = sig_df.merge(actual_df[["period", "actual"]], on="period", how="left")
        merged["surprise"] = merged["actual"] - merged["forecast"]
        merged["external_direction"] = merged["surprise"].apply(
            lambda x: "accel" if x > 0 else ("decel" if x < 0 else np.nan)
        )
        merged = merged.merge(part1_dir, on="period", how="left")
        results[name] = binomial_walkforward(merged, SPF_ANCHOR, name)

    # ── The WRONG, circular comparison (hindsight-ranked, static) ──────────
    hindsight_signals = build_signal(indiv, resolved, release_dates, part1_dir,
                                      use_hindsight=True, static_ranking=full_mae)
    hindsight_results = {}
    for name, sig_df in [("[CIRCULAR/WRONG] TOP-1 by full-sample hindsight MAE", hindsight_signals["top1"]),
                          ("[CIRCULAR/WRONG] TOP-5-AVG by full-sample hindsight MAE", hindsight_signals["top5"])]:
        if sig_df.empty:
            hindsight_results[name] = {"label": name, "n": 0, "verdict": "NO DATA"}
            continue
        merged = sig_df.merge(actual_df[["period", "actual"]], on="period", how="left")
        merged["surprise"] = merged["actual"] - merged["forecast"]
        merged["external_direction"] = merged["surprise"].apply(
            lambda x: "accel" if x > 0 else ("decel" if x < 0 else np.nan)
        )
        merged = merged.merge(part1_dir, on="period", how="left")
        hindsight_results[name] = binomial_walkforward(merged, SPF_ANCHOR, name)

    lines.append("\n## Genuine point-in-time signals (the real test)\n")
    for name, r in results.items():
        lines.append(f"\n### {name}\n")
        if r["n"] == 0:
            lines.append("No qualifying data (no forecaster cleared the minimum-submission bar at any test quarter, or no submissions that quarter).\n")
            continue
        bonf_note = "survives" if r["p_value"] is not None and r["p_value"] < BONFERRONI_6 else "does NOT survive"
        lines.append(
            f"- n={r['n']} | hit rate={r['hit_rate']*100:.1f}% | binomial p={r['p_value']:.4f}\n"
            f"- {r['verdict']}\n"
            f"- vs Bonferroni(6)={BONFERRONI_6:.5f}: **{bonf_note}**\n"
        )

    lines.append("\n## Circular/hindsight comparison (the WRONG way -- shown to quantify the bias gap)\n")
    for name, r in hindsight_results.items():
        lines.append(f"\n### {name}\n")
        if r["n"] == 0:
            lines.append("No qualifying data.\n")
            continue
        lines.append(f"- n={r['n']} | hit rate={r['hit_rate']*100:.1f}% | binomial p={r['p_value']:.4f}\n")

    lines.append("\n## Comparison against the already-tested full-panel consensus\n")
    lines.append(
        "SPF RGDP-surprise (full-panel median consensus, already tested): "
        "n=85, hit rate=63.5%, p=0.0165 (did not survive Bonferroni(4)).\n"
    )

    out_path = RESULTS / "individual_forecaster_report.md"
    out_path.write_text("".join(lines))
    print("".join(lines))
    print(f"\nSaved -> {out_path}")
    return results, hindsight_results, full_mae


if __name__ == "__main__":
    main()
