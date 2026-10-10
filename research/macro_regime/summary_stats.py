"""
Part 1 output summary: time spent in each regime, Precarious co-occurrence
per regime, and regime duration (consecutive-month streak) distribution.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
TIMELINE_PATH = RESULTS / "regime_timeline.csv"


def compute_summary_stats(timeline: pd.DataFrame) -> dict:
    t = timeline.sort_values("month").reset_index(drop=True)

    time_in_regime = t["regime"].value_counts()
    time_in_regime_pct = (time_in_regime / len(t) * 100).round(1)

    precarious_by_regime = t.groupby("regime")["precarious"].mean().mul(100).round(1)

    # Regime duration: consecutive-month streak lengths.
    streak_id = (t["regime"] != t["regime"].shift(1)).cumsum()
    streak_id.name = "streak_id"
    streaks = t.groupby(streak_id).agg(regime=("regime", "first"), n_months=("regime", "size"))
    duration_by_regime = streaks.groupby("regime")["n_months"].describe()[["count", "mean", "min", "max"]]

    return {
        "n_months_total": len(t),
        "time_in_regime_counts": time_in_regime,
        "time_in_regime_pct": time_in_regime_pct,
        "precarious_pct_by_regime": precarious_by_regime,
        "precarious_pct_overall": round(t["precarious"].mean() * 100, 1),
        "regime_duration_stats": duration_by_regime,
        "n_regime_changes": int((t["regime"] != t["regime"].shift(1)).sum() - 1),
    }


def main():
    timeline = pd.read_csv(TIMELINE_PATH, parse_dates=["month"])
    stats = compute_summary_stats(timeline)

    lines = ["# Macro Regime Classifier -- Summary Stats\n"]
    lines.append(f"\nCoverage: {timeline['month'].min().date()} -> {timeline['month'].max().date()} "
                 f"({stats['n_months_total']} months)\n")

    lines.append("\n## Time spent in each regime\n")
    df1 = pd.DataFrame({
        "months": stats["time_in_regime_counts"],
        "pct_of_history": stats["time_in_regime_pct"],
    })
    lines.append(f"\n{df1.to_string()}\n")

    lines.append(f"\n## Precarious flag co-occurrence\n")
    lines.append(f"\nOverall: Precarious in {stats['precarious_pct_overall']}% of months\n")
    lines.append(f"\nBy regime (% of that regime's months where Precarious was also true):\n")
    lines.append(f"\n{stats['precarious_pct_by_regime'].to_string()}\n")

    lines.append(f"\n## Regime duration distribution (consecutive-month streaks)\n")
    lines.append(f"\n{stats['regime_duration_stats'].to_string()}\n")
    lines.append(f"\nTotal regime changes across history: {stats['n_regime_changes']}\n")

    out_path = RESULTS / "summary_stats.md"
    out_path.write_text("".join(lines))
    print("".join(lines))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
