"""
Part 3 -- Live Regime Read. Informational only (not a backtest): current
regime / Precarious flag / policy label from the most recent available
data. Imported by strategies/macro_trading/page.py.
"""

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
TIMELINE_PATH = BASE / "results" / "regime_timeline.csv"


def get_live_read(rebuild: bool = False) -> dict:
    """
    Returns the most recent row of the regime timeline as a plain dict,
    plus daily-granularity Precarious component values (since those move
    faster than the monthly regime label) computed fresh from cache.
    """
    if rebuild or not TIMELINE_PATH.exists():
        from classify import main as build_timeline
        build_timeline()

    df = pd.read_csv(TIMELINE_PATH, parse_dates=["month"])
    last = df.sort_values("month").iloc[-1]

    from fetch import fetch_all
    from classify import build_precarious_flag, build_policy_overlay
    data = fetch_all()
    prec = build_precarious_flag(data["plains"], data["tiingo"]).sort_values("date")
    prec_last = prec.iloc[-1]
    policy = build_policy_overlay(data["plains"]).sort_values("date")
    policy_last = policy.iloc[-1]

    return {
        "as_of_month": last["month"],
        "regime": last["regime"],
        "growth_direction": last["growth_direction"],
        "inflation_direction": last["inflation_direction"],
        "gdp_yoy_smoothed": last["gdp_yoy_smoothed"],
        "core_cpi_yoy_smoothed": last["core_cpi_yoy_smoothed"],
        "gdp_realtime_start": last["gdp_realtime_start"],
        "cpi_realtime_start": last["cpi_realtime_start"],
        "precarious_as_of_month_end": bool(last["precarious"]) if pd.notna(last["precarious"]) else None,
        "precarious_daily_asof": prec_last["date"],
        "precarious_daily": bool(prec_last["precarious"]),
        "precarious_conditions": {
            "credit_compressed": bool(prec_last["cond_credit_compressed"]),
            "vix_low": bool(prec_last["cond_vix_low"]),
            "curve_flat_or_inverted": bool(prec_last["cond_curve_flat_inverted"]),
            "stockbond_corr_rising": bool(prec_last["cond_stockbond_corr_rising"]),
        },
        "policy_label": policy_last["policy_label"],
        "policy_daily_asof": policy_last["date"],
        "dff_level": policy_last["dff"],
        "dff_change_6m": policy_last["dff_change_6m"],
    }


def get_external_signals() -> dict:
    """
    Current readings of the external nowcast/consensus signals tested in
    README.md (2026-07-18 through 2026-07-20 results). Only Cleveland
    Nowcast survives its own Bonferroni correction (inflation family of 2).
    GDP GROWTH PREDICTION IS CLOSED, UNSOLVED with free data: five
    candidates tested (GDPNow, SPF-RGDP consensus/TOP-1/TOP-5-avg, NY Fed
    Staff Nowcast), none clears Bonferroni(5)=0.01 -- see
    "growth_prediction_status" below and README's "Final verdict on GDP
    growth prediction" section. Do not re-attempt a 6th growth candidate
    without reading that section first. These are all external-model-
    derived and informational only: kept structurally separate from Part
    1's coincident regime label, never blended into it.
    """
    from external_fetch import (
        fetch_cleveland_raw, parse_cleveland_nowcast, fetch_spf_raw,
        parse_spf_cpi, parse_spf_rgdp,
    )

    cleveland_raw = fetch_cleveland_raw()
    cleveland = parse_cleveland_nowcast(cleveland_raw, series_name="Core CPI Inflation")
    cleveland = cleveland.sort_values("target_month").reset_index(drop=True)
    cleveland["direction"] = cleveland["terminal_nowcast"].diff().apply(
        lambda x: "accel" if x > 0 else ("decel" if x < 0 else None)
    )
    clev_last = cleveland.iloc[-1]

    level_path, growth_path = fetch_spf_raw()
    spf_cpi = parse_spf_cpi(level_path).sort_values("period").reset_index(drop=True)
    spf_rgdp = parse_spf_rgdp(growth_path).sort_values("period").reset_index(drop=True)
    cpi_last = spf_cpi.iloc[-1]
    rgdp_last = spf_rgdp.iloc[-1]

    return {
        "cleveland_nowcast": {
            "label": "Cleveland Fed Inflation Nowcast (Core CPI) [external, informational]",
            "target_month": clev_last["target_month"],
            "terminal_nowcast_mom_pct": clev_last["terminal_nowcast"],
            "terminal_nowcast_date": clev_last["terminal_nowcast_date"],
            "direction_vs_prior_month": clev_last["direction"],
            "walk_forward_result": "n=102, hit rate 65.7%, binomial p=0.0020 (survives Bonferroni for 4 tests)",
        },
        "spf_consensus": {
            "label": "Philadelphia Fed Survey of Professional Forecasters [external, informational]",
            "survey_quarter": cpi_last["period"],
            "median_core_cpi_forecast_next_q_pct": cpi_last["CORECPI2"],
            "median_rgdp_growth_forecast_next_q_pct": rgdp_last["drgdp2"],
            "note": (
                "Median forecast (annualized) for the quarter AFTER the survey "
                "quarter. The historically-tested signal is the SURPRISE (actual "
                "vs. this forecast, known only in hindsight) -- this is the live "
                "forward-looking reading, not a surprise, since the outcome "
                "hasn't happened yet."
            ),
            "walk_forward_result": (
                "CPI-surprise: n=77, hit rate 63.6%, p=0.0220 (does not survive Bonferroni). "
                "RGDP-surprise: n=85, hit rate 63.5%, p=0.0165 (does not survive Bonferroni(5) "
                "for the growth family -- see growth_prediction_status)."
            ),
        },
        "growth_prediction_status": {
            "label": "GDP growth-direction prediction: NO VALIDATED LEADING SIGNAL [closed search]",
            "summary": (
                "Five free candidates tested, none cleared Bonferroni(5)=0.01: "
                "GDPNow (n=39, 48.7%, p=1.0000, FAILS), SPF RGDP consensus "
                "(n=85, 63.5%, p=0.0165, clears p<0.05 but fails correction), "
                "SPF RGDP TOP-1 forecaster (n=85, 60.0%, p=0.0821, FAILS), "
                "SPF RGDP TOP-5-avg forecasters (n=40, 65.0%, p=0.0807, FAILS), "
                "NY Fed Staff Nowcast (n=35, 40.0%, p=0.3105, FAILS)."
            ),
            "note": (
                "Documented as UNSOLVED with currently available free data, not "
                "merely 'not yet found' -- this search is closed. See README.md's "
                "'Final verdict on GDP growth prediction' section before "
                "attempting a new growth-nowcast/consensus candidate."
            ),
        },
    }


if __name__ == "__main__":
    read = get_live_read()
    for k, v in read.items():
        print(f"{k}: {v}")
    print()
    for k, v in get_external_signals().items():
        print(f"{k}: {v}")
