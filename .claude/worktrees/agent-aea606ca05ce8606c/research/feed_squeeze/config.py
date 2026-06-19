"""
Configuration for the feed-cost-squeeze hypothesis pipeline.

Hypothesis chain:
  El Niño (warm Pacific)
    → Peruvian anchovy catch falls
    → fishmeal price rises
    → salmon farmer feed costs rise (lagged — farmers forward-contract)
    → margin squeeze
    → salmon equities underperform

We test the two tradeable links:
  A. Does ONI lead fishmeal price?
  B. Does fishmeal price predict salmon-farmer returns after stripping salmon spot?
"""

START_DATE = "2005-01-01"
END_DATE   = None   # None → today

# Salmon farmer equities (Oslo Børs = .OL; London = .L)
FARMER_TICKERS = ["MOWI.OL", "SALM.OL", "LSG.OL", "BAKKA.OL", "GSF.OL"]
REF_TICKER     = "BMK.L"   # Benchmark Holdings: upstream genetics / feed tech

# ── Publication lags (months) ─────────────────────────────────────────────────
# Applied centrally in build_panel.py so the logic is auditable in one place.
# A shift of k means: at row T, the value shown is the one observed at T-k
# (because the data for T wasn't publicly available until T+k).
PUB_LAG = {
    # NOAA releases the 3-month running mean ~1 month after the centre month.
    # e.g. the DJF (Jan-centred) value for Jan-2010 appears in NOAA's
    # February-2010 release.
    "oni":          1,
    # World Bank Pink Sheet is released with roughly a 1-month delay.
    "fishmeal":     1,
    # SSB weekly salmon prices are published within 1-2 weeks; for a monthly
    # panel the effective lag rounds to zero.
    "salmon_price": 0,
    # Daily equity prices; no publication lag.
    "equity":       0,
}

# ── El Niño identification (NOAA standard) ────────────────────────────────────
ENSO_THRESHOLD  = 0.5   # ONI ≥ +0.5 °C to qualify
ENSO_MIN_MONTHS = 5     # must persist for ≥ 5 consecutive months

# ── Test A: ONI vs fishmeal ───────────────────────────────────────────────────
TEST_A_MAX_LAG = 12   # lags (months) to test in the forward correlation grid

# ── Test B: fishmeal vs residual return ──────────────────────────────────────
TEST_B_MAX_LAG      = 9     # lags (months) to test in the regression grid
TEST_B_MIN_TICKERS  = 2     # minimum simultaneous tickers required for basket

# ── Test C: signal construction ──────────────────────────────────────────────
SIGNAL_FM_LOOKBACK   = 12   # months for the 12m fishmeal change signal
SIGNAL_ONI_THRESHOLD = 0.5  # ONI level that flags El Niño pressure
SIGNAL_COST_BPS      = 10   # PLACEHOLDER round-trip transaction cost (bps)
