"""
Signal logic for the Private Competitor Distress strategy.

TODO: Implement the following pipeline:

1. DATA INGESTION
   - Pull Companies House bulk filing data (accounts, confirmations, charges)
   - Filter for late/overdue accounts (filing_date > deadline + grace_period)
   - Flag winding-up petitions and county court judgements
   - Cross-reference SIC codes against listed UK/US small-cap universe

2. DISTRESS SCORING
   - For each private company: compute a composite distress score from
     (a) filing delay in days
     (b) interest coverage trend (if accounts available)
     (c) net asset deterioration year-on-year
     (d) director resignation frequency
   - Normalise scores to [0, 1] per SIC cohort

3. LISTED COMPANY MAPPING
   - For each distressed private entity: identify listed competitors
     sharing 3-digit SIC code with market cap < $2B
   - Weight by revenue overlap proxy (private revenue / listed revenue)
   - Aggregate distress exposure score per listed ticker

4. SIGNAL GENERATION
   - Long signal fires when:
     (a) listed ticker's composite distress-exposure score > threshold
     (b) listed ticker shows positive trailing revenue momentum (QoQ > 0)
     (c) analyst consensus has NOT yet revised estimates upward
   - Position size: proportional to distress score, capped at 5% per name
   - Hold period: 90 calendar days or until distress score reverts, whichever first

5. EXIT LOGIC
   - Exit on: hold period expiry, distress score reversion below half threshold,
     or listed company's own accounts showing margin deterioration

6. RISK FILTERS
   - Exclude: micro-caps < $50M market cap, stocks with bid-ask > 1.5%,
     positions during earnings blackout window (±5 days around earnings date)

Parameters to tune during backtesting:
    DISTRESS_THRESHOLD: float = 0.65     # minimum distress score to open position
    FILING_DELAY_DAYS:  int   = 30       # late filing trigger (days past deadline)
    HOLD_PERIOD:        int   = 90       # maximum holding period in calendar days
    MAX_POSITION_PCT:   float = 0.05     # maximum portfolio weight per name
    MIN_MARKET_CAP_USD: float = 50e6     # micro-cap filter
    MAX_SPREAD_PCT:     float = 0.015    # liquidity filter
"""
