"""
Tests the NY Fed Staff Nowcast as a growth-direction signal, using the same
walk-forward + binomial framework as GDPNow (external_test.py) -- no new
test machinery built. Methodology locked in README.md's "NY Fed Staff
Nowcast" section BEFORE this was run. Verification of a build decision, not
a DSR trial -- not logged to trial_registry.csv or CORRECTED_TRIAL_REGISTRY.md.

This is the last free growth-prediction candidate per the brief: after this,
GDP growth prediction is either found or documented as unsolved with
currently available free data, not re-attempted from scratch.
"""

from pathlib import Path

import pandas as pd

from fetch import fetch_all, first_release_series
from classify import build_growth_direction
from external_fetch import parse_nyfed_nowcast, NYFED_REALTIME_START
from external_test import binomial_walkforward, _direction_from_level

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FAMILY_SIZE_GROWTH = 5  # GDPNow, SPF-RGDP consensus, SPF-RGDP TOP-1, SPF-RGDP TOP-5-avg, NY Fed
BONFERRONI_GROWTH = 0.05 / FAMILY_SIZE_GROWTH


def _terminal_by_quarter(nyfed: pd.DataFrame, gdp_release_dates: pd.Series, real_time_only: bool) -> pd.DataFrame:
    rows = []
    for q, release_dt in gdp_release_dates.items():
        sub = nyfed[(nyfed["target_quarter"] == q) & (nyfed["forecast_date"] < release_dt)]
        if real_time_only:
            sub = sub[sub["is_real_time"]]
        if sub.empty:
            continue
        last = sub.sort_values("forecast_date").iloc[-1]
        rows.append({"period": q, "terminal_nyfed": last["value"]})
    return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)


def test_nyfed(data: dict, real_time_only: bool, label: str) -> dict:
    nyfed = parse_nyfed_nowcast()
    gdp_first = first_release_series(data["vintages"]["GDPC1"])
    gdp_release_dates = gdp_first.attrs["realtime_start"]

    terminal_df = _terminal_by_quarter(nyfed, gdp_release_dates, real_time_only)
    terminal_df["external_direction"] = _direction_from_level(terminal_df["terminal_nyfed"])

    part1 = build_growth_direction(data["vintages"]["GDPC1"])[["period", "growth_direction"]]
    part1 = part1.rename(columns={"growth_direction": "part1_direction"})

    merged = terminal_df.merge(part1, on="period", how="inner")
    anchor = NYFED_REALTIME_START if real_time_only else pd.Timestamp("2002-01-01")
    return binomial_walkforward(merged, anchor, label)


def main():
    data = fetch_all()

    primary = test_nyfed(data, real_time_only=True,
                          label="NY Fed Nowcast (real-time only, 2016+, excl. 2021-2023 gap) vs Part 1 growth direction")
    supplementary = test_nyfed(data, real_time_only=False,
                                label="[SUPPLEMENTARY, incl. 2002-2015 reconstructed] NY Fed Nowcast vs Part 1 growth direction")

    lines = ["# NY Fed Staff Nowcast Test (2026-07-20)\n"]
    lines.append(
        "Verification of a build decision, not a DSR trial -- not logged to "
        "the registry. Methodology locked in README.md before running. "
        "Last free growth-prediction candidate per the brief.\n"
    )

    for r in (primary, supplementary):
        lines.append(f"\n## {r['label']}\n")
        if r["n"] == 0:
            lines.append("No qualifying data.\n")
            continue
        lines.append(
            f"- n={r['n']} | hit rate={r['hit_rate']*100:.1f}% | binomial p={r['p_value']:.4f}\n"
            f"- {r['verdict']}\n"
        )

    if primary["n"] > 0:
        bonf_note = ("survives" if primary["p_value"] is not None and primary["p_value"] < BONFERRONI_GROWTH
                      else "does NOT survive")
        lines.append(
            f"\n## Multiple-testing correction (growth-only family of {FAMILY_SIZE_GROWTH})\n"
            f"\nBonferroni({FAMILY_SIZE_GROWTH}) = {BONFERRONI_GROWTH:.3f}. "
            f"Primary result p={primary['p_value']:.4f} **{bonf_note}**.\n"
        )

    lines.append("\n## Comparison against every other growth signal tested\n\n")
    comparison = pd.DataFrame([
        {"signal": "GDPNow", "n": 39, "hit_rate_pct": 48.7, "p_value": 1.0000, "verdict": "NOT significant"},
        {"signal": "SPF RGDP-surprise (full panel)", "n": 85, "hit_rate_pct": 63.5, "p_value": 0.0165, "verdict": "Clears p<0.05, not Bonferroni"},
        {"signal": "SPF RGDP TOP-1 (point-in-time)", "n": 85, "hit_rate_pct": 60.0, "p_value": 0.0821, "verdict": "NOT significant"},
        {"signal": "SPF RGDP TOP-5-AVG (point-in-time)", "n": 40, "hit_rate_pct": 65.0, "p_value": 0.0807, "verdict": "NOT significant"},
        {"signal": "NY Fed Nowcast (real-time only)", "n": primary["n"],
         "hit_rate_pct": round(primary["hit_rate"] * 100, 1) if primary["hit_rate"] is not None else None,
         "p_value": round(primary["p_value"], 4) if primary["p_value"] is not None else None,
         "verdict": primary["verdict"]},
    ])
    lines.append(comparison.to_string(index=False) + "\n")

    out_path = RESULTS / "nyfed_report.md"
    out_path.write_text("".join(lines))
    print("".join(lines))
    print(f"\nSaved -> {out_path}")
    return primary, supplementary


if __name__ == "__main__":
    main()
