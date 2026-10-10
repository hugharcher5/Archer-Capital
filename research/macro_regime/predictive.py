"""
Part 2 -- Predictive / Early Layer. Reported SEPARATELY from Part 1's
coincident label, never blended into it (methodology locked in README.md).

(a) Base-effect model: a "flat" (zero-growth) scenario for GDP/CPI's next
    print, computable today since the year-ago comparison base is already
    known -- reports a directional bias, not a forecast.
(b) Early-warning flags (curve inversion onset, credit-spread-widening
    onset) and an honest lead-time test against when Part 1's coincident
    label actually changed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from fetch import fetch_all, first_release_series
from classify import build_monthly_timeline, REGIME_MAP

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

DECEL_REGIMES = {"Stagflation", "Deflation/Risk-off"}

# Early-warning thresholds -- locked in README.md, calibrated only against
# each series' own historical distribution, before any lead-time result existed.
CREDIT_WIDEN_THRESH = 0.50   # BAA10Y pp above its own trailing 252d rolling min
CREDIT_RESET_BAND = 0.25     # must return within this of the trailing low to "reset"
CREDIT_RESET_DAYS = 20       # ...for at least this many trading days
ROLLING_LOW_WINDOW = 252     # trading days


def base_effect_gdp(gdp_vintage: pd.DataFrame) -> pd.DataFrame:
    """
    forward_yoy_flat = level(t) / level(t-3q) - 1  (the YoY comparison next
    quarter's print will use, under a flat-growth scenario, using only data
    already public today). base_effect_bias = forward_yoy_flat - YoY(t).
    """
    s = first_release_series(gdp_vintage)
    yoy = s / s.shift(4) - 1.0
    forward_flat = s / s.shift(3) - 1.0
    bias = forward_flat - yoy
    out = pd.DataFrame({
        "period": s.index,
        "gdp_yoy": yoy.values,
        "gdp_forward_yoy_flat": forward_flat.values,
        "gdp_base_effect_bias": bias.values,
        "gdp_implied_direction": np.where(bias > 0, "accel-bias",
                                   np.where(bias < 0, "decel-bias", "flat")),
    })
    return out


def base_effect_cpi(cpi_vintage: pd.DataFrame) -> pd.DataFrame:
    """forward_yoy_flat = idx(t) / idx(t-11mo) - 1, same logic as GDP."""
    s = first_release_series(cpi_vintage)
    yoy = s / s.shift(12) - 1.0
    forward_flat = s / s.shift(11) - 1.0
    bias = forward_flat - yoy
    out = pd.DataFrame({
        "period": s.index,
        "cpi_yoy": yoy.values,
        "cpi_forward_yoy_flat": forward_flat.values,
        "cpi_base_effect_bias": bias.values,
        "cpi_implied_direction": np.where(bias > 0, "accel-bias",
                                   np.where(bias < 0, "decel-bias", "flat")),
    })
    return out


def _onset_dates(flag: pd.Series, min_false_run: int = CREDIT_RESET_DAYS) -> list:
    """
    First day of each True-episode ('onset'), debounced: after firing, the
    flag must run continuously False for >= min_false_run trading days before
    a new onset can fire -- otherwise noisy zero-crossings during one choppy
    episode (e.g. a curve bouncing a few bps either side of 0 for months)
    each get double-counted as separate onsets. Same 20-trading-day
    reset convention as the credit-spread-widening flag, for consistency.
    """
    flag = flag.fillna(False).astype(bool)
    onsets = []
    armed = True
    false_streak = 0
    for dt, v in flag.items():
        if v and armed:
            onsets.append(dt)
            armed = False
        if v:
            false_streak = 0
        else:
            false_streak += 1
            if false_streak >= min_false_run:
                armed = True
    return onsets


def curve_inversion_onsets(plains: dict) -> dict:
    out = {}
    for sid, name in (("T10Y2Y", "curve_10y2y_inversion"), ("T10Y3M", "curve_10y3m_inversion")):
        s = plains[sid].set_index("date")["value"].sort_index()
        flag = s < 0
        out[name] = _onset_dates(flag)
    return out


def credit_widening_onsets(plains: dict) -> list:
    s = plains["BAA10Y"].set_index("date")["value"].sort_index()
    rolling_min = s.rolling(ROLLING_LOW_WINDOW, min_periods=ROLLING_LOW_WINDOW).min()
    widened = (s - rolling_min) >= CREDIT_WIDEN_THRESH
    near_low = (s - rolling_min) <= CREDIT_RESET_BAND

    onsets = []
    armed = True  # can fire
    reset_counter = 0
    prev_widened = False
    for dt, is_widened, is_near_low in zip(s.index, widened, near_low):
        if pd.isna(is_widened):
            continue
        if armed and is_widened and not prev_widened:
            onsets.append(dt)
            armed = False
        if not armed:
            if is_near_low:
                reset_counter += 1
            else:
                reset_counter = 0
            if reset_counter >= CREDIT_RESET_DAYS:
                armed = True
        prev_widened = bool(is_widened)
    return onsets


def regime_change_dates(timeline: pd.DataFrame) -> pd.DataFrame:
    """Every month where the coincident regime label differs from the prior month."""
    t = timeline.sort_values("month").reset_index(drop=True)
    changed = t["regime"] != t["regime"].shift(1)
    changes = t[changed & t["regime"].shift(1).notna()][["month", "regime"]].copy()
    changes = changes.rename(columns={"regime": "new_regime"})
    return changes


def lead_time_report(onsets: list, changes: pd.DataFrame, label: str,
                      timeline_start: pd.Timestamp, baseline_gaps: np.ndarray) -> dict:
    """
    For each onset, find the next regime-label change strictly after it.
    Reports lead time to (a) ANY change, (b) specifically a change INTO a
    decelerating regime (Stagflation/Deflation-Risk-off).

    Onsets before the regime timeline's own coverage start are EXCLUDED --
    there is no regime baseline to measure a "next change" against before
    the timeline exists, so including them would inflate lead times with an
    artifact of timeline coverage, not a real early-warning signal. This
    means, honestly, that most of history's real inversion/widening episodes
    (1970s-1990s) can't be evaluated here given the ALFRED vintage coverage
    constraint documented in README.md -- disclosed, not hidden.

    Significance: a one-sided Mann-Whitney U test (H1: lead times are
    stochastically SHORTER than the unconditioned baseline gap between
    regime changes) is reported alongside the descriptive stats. At n=8-13
    onsets this test has very limited power -- do not read a non-significant
    p-value as "no effect," only as "not established at this sample size."
    """
    n_excluded = sum(1 for o in onsets if o < timeline_start)
    onsets = [o for o in onsets if o >= timeline_start]

    any_leads, decel_leads = [], []
    for onset in onsets:
        after = changes[changes["month"] > onset]
        if not after.empty:
            lead = (after.iloc[0]["month"] - onset).days
            any_leads.append(lead)
        decel_after = after[after["new_regime"].isin(DECEL_REGIMES)]
        if not decel_after.empty:
            lead = (decel_after.iloc[0]["month"] - onset).days
            decel_leads.append(lead)

    def _stats(leads):
        if not leads:
            return {"n_onsets_with_subsequent_change": 0, "mean_days": None, "median_days": None,
                     "mwu_p_value": None}
        p_value = None
        if len(leads) >= 2:
            _, p_value = stats.mannwhitneyu(leads, baseline_gaps, alternative="less")
        return {
            "n_onsets_with_subsequent_change": len(leads),
            "mean_days": float(np.mean(leads)),
            "median_days": float(np.median(leads)),
            "mwu_p_value": float(p_value) if p_value is not None else None,
        }

    return {
        "flag": label,
        "n_onsets_total": len(onsets),
        "n_onsets_excluded_pre_timeline": n_excluded,
        "any_regime_change": _stats(any_leads),
        "change_into_decel_regime": _stats(decel_leads),
    }


def baseline_change_frequency(changes: pd.DataFrame) -> dict:
    """
    Unconditional gap between consecutive regime-label changes -- the
    honest null/base-rate comparison. If a flag's lead times aren't
    meaningfully shorter than this, "the onset preceded a change" is likely
    just base-rate luck (changes happen often enough that almost anything
    precedes one eventually), not a real early-warning signal.
    """
    gaps = changes["month"].diff().dt.days.dropna()
    return {"mean_days": float(gaps.mean()), "median_days": float(gaps.median())}


def main():
    data = fetch_all()
    timeline = build_monthly_timeline(data)
    changes = regime_change_dates(timeline)
    baseline = baseline_change_frequency(changes)

    gdp_base = base_effect_gdp(data["vintages"]["GDPC1"])
    cpi_base = base_effect_cpi(data["vintages"]["CPILFESL"])
    gdp_base.to_csv(RESULTS / "part2_gdp_base_effect.csv", index=False)
    cpi_base.to_csv(RESULTS / "part2_cpi_base_effect.csv", index=False)

    curve_onsets = curve_inversion_onsets(data["plains"])
    credit_onsets = credit_widening_onsets(data["plains"])

    timeline_start = pd.Timestamp(timeline["month"].min())
    baseline_gaps = changes["month"].diff().dt.days.dropna().values

    reports = []
    for name, onsets in curve_onsets.items():
        reports.append(lead_time_report(onsets, changes, name, timeline_start, baseline_gaps))
    reports.append(lead_time_report(credit_onsets, changes, "credit_spread_widening", timeline_start, baseline_gaps))

    # ── Write a plain-text/markdown report, honestly, including weak results ──
    lines = ["# Part 2 -- Predictive/Early Layer: Lead-Time Report\n"]
    lines.append(
        "Reported separately from Part 1's coincident label per README. "
        "This is the harder ask; a weak result here is reported as-is, not hidden.\n"
    )
    lines.append(f"\nTotal coincident regime-label changes in history: {len(changes)}\n")
    lines.append(f"Regime timeline coverage: {timeline_start.date()} -> {timeline['month'].max().date()}\n")
    lines.append(
        f"\n**Sample-size caveat**: each flag has only 8-13 onsets within the "
        f"timeline's coverage window. A one-sided Mann-Whitney U test (H1: lead "
        f"times are stochastically shorter than the baseline gap between regime "
        f"changes) is reported below for each flag, but at this sample size the "
        f"test has very limited power -- a non-significant p-value means the "
        f"effect isn't ESTABLISHED at this sample size, not that it's disproven. "
        f"Treat all of Part 2 as suggestive, not conclusive.\n"
    )
    lines.append(
        f"\n**Base-rate baseline (read this before the flags below):** regime "
        f"labels change on average every {baseline['mean_days']:.0f} days "
        f"(median {baseline['median_days']:.0f}d) with NO conditioning on any "
        f"flag at all. Because changes happen this often, a flag finding "
        f"'100% of onsets preceded a subsequent change' is a weak claim by "
        f"itself -- almost any onset will precede *some* change within a few "
        f"months just by base rate. The only honest test is whether a flag's "
        f"lead time is meaningfully **shorter** than this baseline (concentration), "
        f"not merely nonzero -- which is exactly what the Mann-Whitney U test below "
        f"checks formally, rather than an eyeballed threshold.\n"
    )

    for r in reports:
        lines.append(f"\n## {r['flag']}\n")
        lines.append(
            f"- Onsets within timeline coverage: {r['n_onsets_total']} "
            f"({r['n_onsets_excluded_pre_timeline']} earlier onsets excluded -- "
            f"predate the regime timeline, no valid lead-time baseline)\n"
        )
        a = r["any_regime_change"]
        d = r["change_into_decel_regime"]
        lines.append(
            f"- Lead time to ANY subsequent regime-label change: "
            f"{a['n_onsets_with_subsequent_change']}/{r['n_onsets_total']} onsets had one; "
            f"mean {a['mean_days']:.0f}d, median {a['median_days']:.0f}d\n"
            if a["mean_days"] is not None else
            f"- Lead time to ANY subsequent regime-label change: no onsets had one.\n"
        )
        lines.append(
            f"- Lead time to a change specifically INTO Stagflation/Deflation-Risk-off: "
            f"{d['n_onsets_with_subsequent_change']}/{r['n_onsets_total']} onsets had one; "
            f"mean {d['mean_days']:.0f}d, median {d['median_days']:.0f}d\n"
            if d["mean_days"] is not None else
            f"- Lead time to a change specifically INTO Stagflation/Deflation-Risk-off: "
            f"no onsets had one.\n"
        )
        if a["mwu_p_value"] is not None:
            p = a["mwu_p_value"]
            if p < 0.05:
                verdict = f"statistically significant at p={p:.3f} (n={a['n_onsets_with_subsequent_change']})"
            elif p < 0.10:
                verdict = (
                    f"marginal, p={p:.3f} (n={a['n_onsets_with_subsequent_change']}) -- "
                    f"directionally suggestive but NOT significant at conventional levels; "
                    f"do not call this 'concentrated' or 'confirmed'"
                )
            else:
                verdict = f"NOT significant, p={p:.3f} (n={a['n_onsets_with_subsequent_change']})"
            lines.append(
                f"- Mann-Whitney U vs. base-rate baseline (lead times shorter than "
                f"baseline gaps?): **{verdict}**\n"
            )

    lines.append("\n## Base-effect model -- most recent readings\n")
    lines.append(f"\nGDP (last 4 quarters):\n\n{gdp_base.tail(4).to_string(index=False)}\n")
    lines.append(f"\nCore CPI (last 6 months):\n\n{cpi_base.tail(6).to_string(index=False)}\n")

    report_path = RESULTS / "part2_leadtime_report.md"
    report_path.write_text("".join(lines))
    print("".join(lines))
    print(f"\nSaved -> {report_path}")


if __name__ == "__main__":
    main()
