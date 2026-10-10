"""
Part 1 -- Coincident Regime Classifier. Methodology locked in README.md
before this was written; do not tune thresholds against the output below.

Produces a monthly labeled timeline: regime (Goldilocks / Reflation /
Stagflation / Deflation-Risk-off), the independent Precarious fragility
flag, and the Hawkish/Dovish/Neutral policy overlay. Point-in-time gated
for GDP/CPI via their ALFRED first-release vintage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from fetch import fetch_all, first_release_series

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

REGIME_MAP = {
    ("accel", "decel"): "Goldilocks",
    ("accel", "accel"): "Reflation",
    ("decel", "accel"): "Stagflation",
    ("decel", "decel"): "Deflation/Risk-off",
}

# Precarious flag thresholds -- locked in README.md against each series' own
# historical distribution, before any regime/flag output existed.
CREDIT_SPREAD_COMPRESSED = 1.75   # BAA10Y < this (pp)
VIX_LOW = 15.0                    # VIXCLS < this
CURVE_FLAT_OR_INVERTED = 0.25     # T10Y2Y <= this (pp)
CORR_WINDOW = 60                  # trading days, SPY/IEF rolling correlation

# Policy overlay
POLICY_WINDOW_DAYS = 126          # ~6 calendar months of trading days
POLICY_HAWKISH_THRESH = 0.25      # DFF change over window (pp)
POLICY_DOVISH_THRESH = -0.25


def _direction_series(smoothed: pd.Series, lag: int) -> pd.Series:
    """
    Sign of the change in an already-smoothed series over `lag` periods.
    Ties (exactly zero) carry forward the previous direction -- per README,
    to avoid oscillation from a knife-edge zero.
    """
    diff = smoothed - smoothed.shift(lag)
    direction = diff.apply(lambda x: "accel" if x > 0 else ("decel" if x < 0 else np.nan))
    direction = direction.ffill()
    return direction


def build_growth_direction(gdp_vintage: pd.DataFrame) -> pd.DataFrame:
    """
    GDP YoY computed from GDPC1's first-release levels (both the current and
    year-ago level use their own first-release vintage -- same convention as
    core CPI). GDPC1 (not the pre-computed A191RO1Q156NBEA transform) is used
    because it has genuine per-quarter ALFRED vintages from ~1991 onward; the
    transform series bunches everything before 2014-09-26 into one
    artificial release date. 2-quarter SMA, direction = QoQ change in the
    smoothed series.
    """
    s = first_release_series(gdp_vintage)
    realtime = s.attrs["realtime_start"]
    yoy = s / s.shift(4) - 1.0
    smoothed = yoy.rolling(2, min_periods=2).mean()
    direction = _direction_series(smoothed, lag=1)
    out = pd.DataFrame({
        "period": s.index,
        "gdp_yoy_raw": yoy.values,
        "gdp_yoy_smoothed": smoothed.values,
        "growth_direction": direction.values,
        "realtime_start": realtime.values,
    })
    return out


def build_inflation_direction(cpi_vintage: pd.DataFrame) -> pd.DataFrame:
    """
    Core CPI YoY computed from first-release index vintages (both current and
    year-ago level use their own first-release value, per README). 6-month
    SMA, direction = quarter-equivalent (3-month) change in the smoothed series.
    """
    s = first_release_series(cpi_vintage)
    realtime = s.attrs["realtime_start"]
    yoy = s / s.shift(12) - 1.0
    smoothed = yoy.rolling(6, min_periods=6).mean()
    direction = _direction_series(smoothed, lag=3)
    out = pd.DataFrame({
        "period": s.index,
        "core_cpi_yoy_raw": yoy.values,
        "core_cpi_yoy_smoothed": smoothed.values,
        "inflation_direction": direction.values,
        "realtime_start": realtime.values,
    })
    return out


def build_precarious_flag(plains: dict, tiingo: dict) -> pd.DataFrame:
    """Daily Precarious flag: all 4 locked conditions must hold simultaneously."""
    baa = plains["BAA10Y"].set_index("date")["value"].rename("baa10y")
    vix = plains["VIXCLS"].set_index("date")["value"].rename("vix")
    curve = plains["T10Y2Y"].set_index("date")["value"].rename("curve")

    spy = tiingo["SPY"].set_index("date")["adjClose"].rename("spy")
    ief = tiingo["IEF"].set_index("date")["adjClose"].rename("ief")
    px = pd.concat([spy, ief], axis=1).dropna()
    rets = px.pct_change()
    corr = rets["spy"].rolling(CORR_WINDOW).corr(rets["ief"]).rename("sb_corr")

    df = pd.concat([baa, vix, curve, corr], axis=1, sort=True).ffill()

    cond_credit = df["baa10y"] < CREDIT_SPREAD_COMPRESSED
    cond_vix = df["vix"] < VIX_LOW
    cond_curve = df["curve"] <= CURVE_FLAT_OR_INVERTED
    corr_rising = df["sb_corr"] > df["sb_corr"].shift(CORR_WINDOW)
    cond_corr = (df["sb_corr"] > 0) & corr_rising

    df["precarious"] = cond_credit & cond_vix & cond_curve & cond_corr
    df["cond_credit_compressed"] = cond_credit
    df["cond_vix_low"] = cond_vix
    df["cond_curve_flat_inverted"] = cond_curve
    df["cond_stockbond_corr_rising"] = cond_corr
    return df.reset_index().rename(columns={"index": "date"})


def build_policy_overlay(plains: dict) -> pd.DataFrame:
    """Hawkish/Dovish/Neutral from trailing-window DFF change."""
    dff = plains["DFF"].set_index("date")["value"].rename("dff").sort_index()
    change = dff - dff.shift(POLICY_WINDOW_DAYS)

    def _label(x):
        if pd.isna(x):
            return np.nan
        if x > POLICY_HAWKISH_THRESH:
            return "Hawkish"
        if x < POLICY_DOVISH_THRESH:
            return "Dovish"
        return "Neutral"

    label = change.apply(_label)
    out = pd.DataFrame({"date": dff.index, "dff": dff.values,
                         "dff_change_6m": change.values, "policy_label": label.values})
    return out


def build_monthly_timeline(data: dict) -> pd.DataFrame:
    growth = build_growth_direction(data["vintages"]["GDPC1"])
    inflation = build_inflation_direction(data["vintages"]["CPILFESL"])
    precarious_daily = build_precarious_flag(data["plains"], data["tiingo"])
    policy_daily = build_policy_overlay(data["plains"])

    # Monthly index spanning the earliest to latest available data.
    start = min(growth["realtime_start"].min(), inflation["realtime_start"].min())
    end = pd.Timestamp.today().normalize()
    months = pd.date_range(start.to_period("M").to_timestamp(), end, freq="MS")

    rows = []
    for m in months:
        # Filing/release-date gating: only use a GDP/CPI reading whose real
        # publication date (realtime_start) has already passed as of month m.
        g_avail = growth[growth["realtime_start"] <= m]
        i_avail = inflation[inflation["realtime_start"] <= m]
        if g_avail.empty or i_avail.empty:
            continue
        g_row = g_avail.iloc[-1]
        i_row = i_avail.iloc[-1]

        if pd.isna(g_row["growth_direction"]) or pd.isna(i_row["inflation_direction"]):
            continue

        regime = REGIME_MAP[(g_row["growth_direction"], i_row["inflation_direction"])]

        # Daily overlays: use the last available observation within the month.
        month_end = m + pd.offsets.MonthEnd(0)
        prec_month = precarious_daily[
            (precarious_daily["date"] >= m) & (precarious_daily["date"] <= month_end)
        ]
        pol_month = policy_daily[
            (policy_daily["date"] >= m) & (policy_daily["date"] <= month_end)
        ]
        precarious = bool(prec_month["precarious"].iloc[-1]) if not prec_month.empty else np.nan
        policy = pol_month["policy_label"].iloc[-1] if not pol_month.empty else np.nan

        rows.append({
            "month": m,
            "regime": regime,
            "growth_direction": g_row["growth_direction"],
            "inflation_direction": i_row["inflation_direction"],
            "gdp_yoy_smoothed": g_row["gdp_yoy_smoothed"],
            "core_cpi_yoy_smoothed": i_row["core_cpi_yoy_smoothed"],
            "gdp_realtime_start": g_row["realtime_start"],
            "cpi_realtime_start": i_row["realtime_start"],
            "precarious": precarious,
            "policy_label": policy,
        })

    timeline = pd.DataFrame(rows)
    return timeline


def main():
    data = fetch_all()
    timeline = build_monthly_timeline(data)
    out_path = RESULTS / "regime_timeline.csv"
    timeline.to_csv(out_path, index=False)
    print(f"\nSaved {len(timeline)} monthly rows -> {out_path}")
    print(f"Coverage: {timeline['month'].min().date()} -> {timeline['month'].max().date()}")
    print("\nRegime value counts:")
    print(timeline["regime"].value_counts())
    print(f"\nPrecarious flag: {timeline['precarious'].sum()} / {timeline['precarious'].notna().sum()} months")
    return timeline


if __name__ == "__main__":
    main()
