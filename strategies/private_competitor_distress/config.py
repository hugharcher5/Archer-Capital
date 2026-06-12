"""Strategy metadata for Private Competitor Distress."""

from pathlib import Path

STRATEGY_CONFIG = {
    "id":   "private_competitor_distress",
    "name": "Private Competitor Distress",
    "description": (
        "This strategy goes long listed small-cap companies whose private competitors "
        "are showing signs of financial distress in public filings — specifically late "
        "accounts, winding-up petitions, and deteriorating balance sheets visible in "
        "Companies House data. The thesis is that when a private competitor enters "
        "distress, its customers and revenue migrate toward the surviving listed player. "
        "That market share transfer rarely shows up immediately in analyst estimates or "
        "consensus earnings, creating a window where the stock is mispriced relative to "
        "its improving competitive position. The signal is constructed from a combination "
        "of filing-delay flags, coverage ratio deterioration, and cross-referencing "
        "SIC codes to identify which listed companies share a competitive addressable "
        "market with the distressed private entity."
    ),
    "results_dir": Path(__file__).parent / "results",
    "signal_file": Path(__file__).parent / "signal.py",
    "validation_layers": [
        {
            "id":          "in_sample_backtest",
            "number":      1,
            "name":        "In-Sample Backtest",
            "description": (
                "Runs the signal on historical data within the training window. "
                "Proves the strategy has edge on data it was built on. A necessary "
                "but not sufficient condition — results here can reflect overfitting."
            ),
            "status":      "not_started",
        },
        {
            "id":          "in_sample_mcpt",
            "number":      2,
            "name":        "In-Sample MCPT (1,000 Permutations)",
            "description": (
                "Monte Carlo Permutation Test: randomises the signal label sequence "
                "1,000 times and measures what Sharpe ratio the strategy achieves by "
                "luck alone. The real Sharpe must sit in the top 5% of the permuted "
                "distribution to pass."
            ),
            "status":      "not_started",
        },
        {
            "id":          "walk_forward",
            "number":      3,
            "name":        "Walk-Forward Analysis",
            "description": (
                "Repeatedly re-trains on a rolling in-sample window and tests on the "
                "immediately following out-of-sample period. The out-of-sample Sharpe "
                "across all windows should be consistently positive and not much lower "
                "than the in-sample Sharpe."
            ),
            "status":      "not_started",
        },
        {
            "id":          "walk_forward_mcpt",
            "number":      4,
            "name":        "Walk-Forward MCPT",
            "description": (
                "Same permutation test applied specifically to the out-of-sample "
                "walk-forward returns. Guards against the possibility that the "
                "walk-forward windows themselves were inadvertently over-selected."
            ),
            "status":      "not_started",
        },
        {
            "id":          "deflated_sharpe",
            "number":      5,
            "name":        "Deflated Sharpe Ratio",
            "description": (
                "Corrects the observed Sharpe ratio for the number of strategy "
                "configurations trialled during development. The more variants you "
                "tested, the higher the probability that the best one outperformed "
                "by chance. DSR adjusts for this multiple-testing bias."
            ),
            "status":      "not_started",
        },
        {
            "id":          "crisis_replay",
            "number":      6,
            "name":        "Crisis Replay",
            "description": (
                "Re-runs the strategy through the three most severe recent market "
                "crises: 2008 Global Financial Crisis, 2020 COVID crash, and the "
                "2022 rate-shock drawdown. Checks that the strategy does not have "
                "a hidden structural short-volatility or long-beta exposure that "
                "only surfaces in tail events."
            ),
            "status":      "not_started",
        },
        {
            "id":          "paper_trading",
            "number":      7,
            "name":        "Paper Trading",
            "description": (
                "Live signal generation with paper (simulated) execution. Confirms "
                "the signal continues to fire on new data after development is "
                "complete, and that execution assumptions are realistic."
            ),
            "status":      "not_started",
        },
    ],
}
