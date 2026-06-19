"""
Entry point for the feed-cost-squeeze hypothesis pipeline.

Usage
-----
    venv/bin/python research/feed_squeeze/run.py

Prints Test A, the critical El Niño cycle diagnostic, Test B, and (only when
A or B show a meaningful signal) Test C backtest statistics.

No files are written — this is an exploratory screen, not a production run.
Do not tune anything in this file to improve results.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

# Allow running as a script from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from research.feed_squeeze.build_panel  import build_panel
from research.feed_squeeze.fetch_data   import fetch_oni   # raw series for cycle counting
from research.feed_squeeze.hypothesis   import (
    count_enso_cycles,
    test_a_oni_vs_fishmeal,
    test_b_fishmeal_vs_residual,
    test_c_backtest,
)


# ── formatting helpers ────────────────────────────────────────────────────────

def _rule(n: int = 72) -> None:
    print("─" * n)


def _header(title: str) -> None:
    _rule()
    print(f"  {title}")
    _rule()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    warnings.filterwarnings("default")

    # ── 1. Assemble panel ────────────────────────────────────────────────────
    _header("FEED-COST-SQUEEZE HYPOTHESIS — DATA ASSEMBLY")
    panel, coverage = build_panel(verbose=True)

    # ── 2. CRITICAL DIAGNOSTIC: independent El Niño cycles ──────────────────
    _header("CRITICAL DIAGNOSTIC — INDEPENDENT EL NIÑO CYCLES")
    print("The ONI leg of the hypothesis has ~5–7 independent events over 20 years.")
    print("Monthly row counts (≈240) are autocorrelated — they are NOT the true n.")
    print("Interpret all ONI-linked correlations in light of the count below.\n")

    oni_raw  = fetch_oni()
    cycles   = count_enso_cycles(oni_raw)
    n_cycles = len(cycles)
    print(f"  El Niño episodes detected (ONI ≥ +0.5 for ≥ 5 consecutive months): "
          f"{'*** ' if n_cycles < 8 else ''}{n_cycles}{'  ← small sample ***' if n_cycles < 8 else ''}\n")
    for i, (start, end) in enumerate(cycles, 1):
        print(f"  {i:2d}.  {start.strftime('%Y-%m')} → {end.strftime('%Y-%m')}")

    # ── 3. Test A ────────────────────────────────────────────────────────────
    _header("TEST A — ONI leads fishmeal price?")
    print("Pearson r between ONI_pub_t and d_fishmeal_{t+k}, for k = 0 … 12.")
    print("oni_pub carries a 1-month publication lag; real-world lag = 1+k months.")
    print("ALL lags reported — do not interpret the strongest lag as the signal.\n")

    tA = test_a_oni_vs_fishmeal(panel)
    if tA.empty:
        print("  [SKIP] Fishmeal data not available.")
    else:
        print(tA.to_string(index=False))

    # ── 4. Test B ────────────────────────────────────────────────────────────
    _header("TEST B — Fishmeal predicts residual farmer returns?")
    print("Basket return residualised on salmon spot return (stage 1), then")
    print("regressed on d_fishmeal lagged k months (stage 2, k = 0 … 9).")
    print("d_fishmeal carries a 1-month pub lag; real-world lag = 1+k months.\n")

    tB_basket, resid, tB_ind, beta_salmon, r2_s1 = test_b_fishmeal_vs_residual(panel)

    if tB_basket.empty:
        print("  [SKIP] Insufficient data for Test B.")
        resid = None
    else:
        if not np.isnan(beta_salmon):
            print(f"  Stage 1 — β(salmon spot) = {beta_salmon:.3f},  R² = {r2_s1:.3f}")
            print(f"  Residual = basket return net of salmon spot exposure.\n")
        print("  [Basket residual — all lags]")
        print(tB_basket.drop(columns="ticker").to_string(index=False))

        if not tB_ind.empty:
            print("\n  [Individual tickers — all lags]")
            print(tB_ind.to_string(index=False))

    # ── 5. Test C (conditional) ───────────────────────────────────────────────
    _header("TEST C — Signal backtest")

    if resid is None or resid.notna().sum() < 30:
        print("  [SKIP] Insufficient residual data.")
    else:
        a_max_r = float(tA["pearson_r"].abs().max()) if not tA.empty else 0.0
        b_max_t = float(tB_basket["t_stat"].abs().max()) if not tB_basket.empty else 0.0

        threshold_r = 0.20   # modest hurdle: at least one lag with |r| > 0.20
        threshold_t = 1.65   # at least one lag with |t| > 1.65 (~10% one-tail)

        print(f"  Test A  max |Pearson r| = {a_max_r:.3f}  (hurdle {threshold_r})")
        print(f"  Test B  max |t-stat|    = {b_max_t:.3f}  (hurdle {threshold_t})\n")

        if (np.isnan(a_max_r) or a_max_r < threshold_r) and (np.isnan(b_max_t) or b_max_t < threshold_t):
            print("  Neither test clears its hurdle.")
            print("  Running the backtest anyway would overfit to noise — SKIPPING.")
            print("  Conclusion: no evidence for the feed-squeeze signal in this sample.")
        else:
            print("  At least one test clears its hurdle — running backtest.\n")
            stats_c = test_c_backtest(panel, resid)
            if stats_c is None:
                print("  [SKIP] Test C returned no result (see warnings above).")
            else:
                for k, v in stats_c.items():
                    if isinstance(v, float):
                        print(f"  {k:<38s} {v:+.4f}")
                    else:
                        print(f"  {k:<38s} {v}")

    # ── 6. Summary ───────────────────────────────────────────────────────────
    _rule()
    print(f"  Independent El Niño cycles in sample: {n_cycles}  "
          f"← anchor all ONI-leg conclusions to this number.")
    print("  This was a first-pass screen. Do not tune parameters to improve results.")
    _rule()


if __name__ == "__main__":
    main()
