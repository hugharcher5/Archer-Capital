"""
Walk-forward (expanding-window) validation of Part 2's early-warning flags.
This is verification of existing infrastructure, NOT a new DSR trial -- not
logged to trial_registry.csv or CORRECTED_TRIAL_REGISTRY.md. Does not touch
Part 1's regime_timeline.csv or the live classifier; only re-tests Part 2's
lead-time claims with out-of-sample discipline.

Design (locked before running):
- ANCHOR_CUTOFF = 2010-01-01. Onsets before this are "training" evidence
  only (establishing the initial belief a flag might work) and are not
  themselves scored as walk-forward test cases. Regime changes before this
  are not scored as walk-forward test outcomes either.
- Expansion steps = each regime-change date strictly after ANCHOR_CUTOFF, in
  chronological order ("one regime-cycle at a time" per the brief). At each
  step, "does a change get anticipated" is evaluated using ONLY onsets that
  occurred strictly before it (the anchor is cumulative/expanding -- no
  look-ahead into onsets that haven't happened yet).
- HORIZON_DAYS = 365: the maximum gap between an onset and a subsequent
  regime change for the onset to count as having "anticipated" it. Locked
  BEFORE running by checking it is more generous than every lead time
  observed in the ORIGINAL in-sample analysis (max there was 358 days, for
  credit-widening) -- so this choice cannot penalize any case the in-sample
  analysis called a "hit," while still imposing a real cutoff (a warning
  that takes over a year to pay off is not a useful early-warning signal in
  practice).
- Null comparison: because regime changes recur often (median gap ~92-150
  days) and HORIZON_DAYS=365 is generous relative to that, a high raw "hit
  rate" is close to guaranteed by base rate alone, not evidence a flag is
  informative -- exactly the same trap the original in-sample Mann-Whitney
  check existed to catch. A null hit-rate (computed over ALL calendar days
  in the walk-forward window, not just onset dates) is reported alongside
  for honest comparison.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fetch import fetch_all
from classify import build_monthly_timeline
from predictive import curve_inversion_onsets, credit_widening_onsets, regime_change_dates

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

ANCHOR_CUTOFF = pd.Timestamp("2010-01-01")
HORIZON_DAYS = 365


def walk_forward_flag(onsets: list, changes: pd.DataFrame, timeline_start: pd.Timestamp,
                       today: pd.Timestamp) -> dict:
    onsets = sorted(o for o in onsets if o >= timeline_start)
    changes_sorted = changes.sort_values("month")["month"].tolist()

    # ── Per-transition scorecard: each regime change after the anchor is one
    # expansion step. Was it anticipated by any onset already known by then? ──
    steps = [c for c in changes_sorted if c > ANCHOR_CUTOFF]
    step_rows = []
    for c in steps:
        candidates = [o for o in onsets if o < c and (c - o).days <= HORIZON_DAYS]
        if candidates:
            best_onset = max(candidates)
            step_rows.append({
                "step_change_date": c, "anticipated_by_onset": best_onset,
                "lead_time_days": (c - best_onset).days, "verdict": "HIT",
            })
        else:
            step_rows.append({
                "step_change_date": c, "anticipated_by_onset": None,
                "lead_time_days": None, "verdict": "MISSED",
            })

    # ── Per-onset scorecard: each onset from the anchor cutoff onward (i.e.
    # the flags actually being "walk-forward tested", not pre-anchor training
    # evidence) -- did it lead to a change within the horizon, too late, or is
    # its outcome not yet knowable? ──
    onset_rows = []
    for o in onsets:
        if o < ANCHOR_CUTOFF:
            continue
        after = [c for c in changes_sorted if c > o]
        if after:
            next_c = after[0]
            lead = (next_c - o).days
            status = "HIT" if lead <= HORIZON_DAYS else "FALSE POSITIVE (change came, too late)"
        elif (today - o).days > HORIZON_DAYS:
            next_c, lead, status = None, None, "FALSE POSITIVE (no change within horizon)"
        else:
            next_c, lead, status = None, None, "PENDING (horizon not yet elapsed)"
        onset_rows.append({
            "onset_date": o, "next_change": next_c, "lead_time_days": lead, "status": status,
        })

    # ── Null/base-rate comparison: for every CALENDAR MONTH in the
    # walk-forward window (not just onset dates), what fraction have a
    # regime change within HORIZON_DAYS after them? If a flag's hit rate
    # isn't meaningfully above this, "mostly hits" is base rate, not signal. ──
    calendar_points = pd.date_range(ANCHOR_CUTOFF, today - pd.Timedelta(days=HORIZON_DAYS), freq="MS")
    null_hits = 0
    for d in calendar_points:
        after = [c for c in changes_sorted if c > d]
        if after and (after[0] - d).days <= HORIZON_DAYS:
            null_hits += 1
    null_hit_rate = null_hits / len(calendar_points) if len(calendar_points) else None

    n_hit = sum(1 for r in step_rows if r["verdict"] == "HIT")
    n_missed = sum(1 for r in step_rows if r["verdict"] == "MISSED")
    n_onset_hit = sum(1 for r in onset_rows if r["status"] == "HIT")
    n_onset_fp = sum(1 for r in onset_rows if r["status"].startswith("FALSE POSITIVE"))
    n_onset_pending = sum(1 for r in onset_rows if r["status"].startswith("PENDING"))

    return {
        "step_rows": step_rows, "onset_rows": onset_rows,
        "n_transitions_tested": len(steps), "n_transitions_hit": n_hit, "n_transitions_missed": n_missed,
        "n_onsets_tested": len(onset_rows), "n_onsets_hit": n_onset_hit,
        "n_onsets_false_positive": n_onset_fp, "n_onsets_pending": n_onset_pending,
        "null_hit_rate": null_hit_rate, "n_null_calendar_points": len(calendar_points),
    }


def permutation_test_coverage(n_onsets: int, observed_coverage: int, steps: list,
                                window_start: pd.Timestamp, window_end: pd.Timestamp,
                                horizon_days: int, n_sims: int = 20000, seed: int = 42) -> dict:
    """
    The per-onset 'hit rate' is close to guaranteed by base rate alone (onsets
    are rare, transitions are frequent, horizon is generous -- almost any
    onset will have SOME transition within a year). The per-transition
    COVERAGE rate (what fraction of actual transitions had a preceding onset)
    is the metric that actually reflects onset TIMING, so it's what gets a
    real null test here: place n_onsets random dates uniformly in the test
    window, count how many of the real transitions they'd "cover" under the
    same rule, repeat n_sims times, and see where the real flag's observed
    coverage count falls in that null distribution (empirical p-value =
    fraction of simulations with coverage >= observed).
    """
    if n_onsets == 0 or not steps:
        return {"observed_coverage": observed_coverage, "null_mean": None,
                "null_median": None, "p_value": None, "n_sims": n_sims}

    rng = np.random.default_rng(seed)
    window_days = (window_end - window_start).days
    step_arr = np.array([(c - window_start).days for c in steps])

    null_coverage = np.empty(n_sims, dtype=int)
    for i in range(n_sims):
        random_offsets = rng.integers(0, window_days, size=n_onsets)
        random_offsets.sort()
        covered = 0
        for c_off in step_arr:
            prior = random_offsets[random_offsets < c_off]
            if prior.size and (c_off - prior[-1]) <= horizon_days:
                covered += 1
        null_coverage[i] = covered

    p_value = float((null_coverage >= observed_coverage).mean())
    return {
        "observed_coverage": observed_coverage,
        "null_mean": float(null_coverage.mean()),
        "null_median": float(np.median(null_coverage)),
        "p_value": p_value,
        "n_sims": n_sims,
    }


def main():
    data = fetch_all()
    timeline = build_monthly_timeline(data)
    changes = regime_change_dates(timeline)
    timeline_start = pd.Timestamp(timeline["month"].min())
    today = pd.Timestamp(timeline["month"].max())

    curve_onsets = curve_inversion_onsets(data["plains"])
    credit_onsets = credit_widening_onsets(data["plains"])

    flags = {
        "curve_10y2y_inversion": curve_onsets["curve_10y2y_inversion"],
        "curve_10y3m_inversion": curve_onsets["curve_10y3m_inversion"],
        "credit_spread_widening": credit_onsets,
    }

    lines = ["# Part 2 Walk-Forward Validation (2026-07-17)\n"]
    lines.append(
        "Out-of-sample follow-up to the original in-sample lead-time test "
        "(see 'Verification audit' in README.md). This is verification of "
        "existing infrastructure, not a DSR trial -- not logged to the "
        "registry. Does not change Part 1's regime_timeline.csv or the live "
        "classifier.\n"
    )
    lines.append(
        f"\nDesign: expanding-window anchor starting {ANCHOR_CUTOFF.date()}, one "
        f"expansion step per subsequent regime change, HORIZON_DAYS={HORIZON_DAYS} "
        f"(locked before running -- more generous than every in-sample lead time "
        f"observed, so it cannot retroactively penalize a prior 'hit').\n"
    )
    lines.append(
        "\n**Read this before the tables below**: sample sizes here are "
        "extremely small (3-8 onsets being walk-forward tested per flag, "
        "since the anchor already consumes the pre-2010 onsets as training "
        "evidence). This is a low-confidence read regardless of outcome -- "
        "not proof a flag works, and not proof it doesn't.\n"
    )

    steps_all = [c for c in changes.sort_values("month")["month"].tolist() if c > ANCHOR_CUTOFF]

    summary_rows = []
    for name, onsets in flags.items():
        r = walk_forward_flag(onsets, changes, timeline_start, today)
        perm = permutation_test_coverage(
            n_onsets=r["n_onsets_tested"], observed_coverage=r["n_transitions_hit"],
            steps=steps_all, window_start=ANCHOR_CUTOFF, window_end=today,
            horizon_days=HORIZON_DAYS,
        )
        lines.append(f"\n## {name}\n")

        lines.append(f"\n### Per-transition scorecard ({r['n_transitions_tested']} expansion steps)\n\n")
        step_df = pd.DataFrame(r["step_rows"])
        if not step_df.empty:
            step_df_display = step_df.copy()
            step_df_display["step_change_date"] = step_df_display["step_change_date"].dt.date
            step_df_display["anticipated_by_onset"] = step_df_display["anticipated_by_onset"].apply(
                lambda x: x.date() if pd.notna(x) else None
            )
            lines.append(step_df_display.to_string(index=False) + "\n")
        else:
            lines.append("No regime changes occurred after the anchor cutoff to test.\n")

        lines.append(f"\n### Per-onset scorecard ({r['n_onsets_tested']} onsets tested, i.e. onsets >= anchor cutoff)\n\n")
        onset_df = pd.DataFrame(r["onset_rows"])
        if not onset_df.empty:
            onset_df_display = onset_df.copy()
            onset_df_display["onset_date"] = onset_df_display["onset_date"].dt.date
            onset_df_display["next_change"] = onset_df_display["next_change"].apply(
                lambda x: x.date() if pd.notna(x) else None
            )
            lines.append(onset_df_display.to_string(index=False) + "\n")
        else:
            lines.append("No onsets occurred at/after the anchor cutoff to test.\n")

        lines.append(
            f"\n**Summary**: {r['n_transitions_hit']}/{r['n_transitions_tested']} "
            f"transitions correctly anticipated, {r['n_transitions_missed']}/"
            f"{r['n_transitions_tested']} missed (no prior flag). "
            f"{r['n_onsets_hit']}/{r['n_onsets_tested']} onsets led to a timely "
            f"change (hit), {r['n_onsets_false_positive']}/{r['n_onsets_tested']} "
            f"confirmed false positives, {r['n_onsets_pending']}/{r['n_onsets_tested']} "
            f"still pending (too recent to resolve).\n"
        )
        lines.append(
            f"\n**Null/base-rate comparison (per-onset framing)**: {r['null_hit_rate']*100:.0f}% "
            f"of ALL calendar months in the walk-forward window (n={r['n_null_calendar_points']}) "
            f"already had a regime change within {HORIZON_DAYS} days, with NO "
            f"conditioning on any flag -- this is why the per-onset '100% hit "
            f"rate' above is NOT evidence of skill: it's close to guaranteed by "
            f"base rate given how frequently regime changes occur.\n"
        )
        if perm["p_value"] is not None:
            lines.append(
                f"\n**Permutation test (the real test — per-transition COVERAGE "
                f"framing)**: placing {r['n_onsets_tested']} random dates in the "
                f"same window {perm['n_sims']} times, the null distribution covers "
                f"a mean of {perm['null_mean']:.1f} / {r['n_transitions_tested']} "
                f"transitions (median {perm['null_median']:.0f}) purely by chance "
                f"placement. The real flag covered {r['n_transitions_hit']} / "
                f"{r['n_transitions_tested']} -- empirical p-value = "
                f"{perm['p_value']:.3f} (fraction of random placements that did "
                f"at least as well).\n"
            )
        else:
            lines.append("\n**Permutation test**: not computable (zero onsets tested).\n")

        summary_rows.append({
            "flag": name,
            "transitions_hit": f"{r['n_transitions_hit']}/{r['n_transitions_tested']}",
            "onsets_confirmed_fp": f"{r['n_onsets_false_positive']}/{r['n_onsets_tested']}",
            "onsets_pending": r["n_onsets_pending"],
            "null_hit_rate_pct (per-onset)": round(r["null_hit_rate"] * 100, 0) if r["null_hit_rate"] is not None else None,
            "perm_test_p_value (per-transition)": round(perm["p_value"], 3) if perm["p_value"] is not None else None,
        })

    lines.append("\n## Overall summary table\n\n")
    lines.append(pd.DataFrame(summary_rows).to_string(index=False) + "\n")

    lines.append(
        "\n## Verdict\n\n"
        "See README.md's Verification audit section (2026-07-17 entry) for "
        "the full honest verdict on whether any flag survives this "
        "out-of-sample test.\n"
    )

    out_path = RESULTS / "walkforward_report.md"
    out_path.write_text("".join(lines))
    print("".join(lines))
    print(f"\nSaved -> {out_path}")
    return summary_rows


if __name__ == "__main__":
    main()
