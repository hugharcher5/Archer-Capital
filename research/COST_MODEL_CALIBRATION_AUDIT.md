# Cost Model Calibration Audit

**Triggered by:** S21 (clusterreversion_S21, trial #27) reporting a ~110%/yr cost drag against a genuinely positive gross Sharpe (0.95) — extreme enough to warrant checking whether the registry's standard cost model (`_spread()`/`_borrow()`, identical across every trial since S1) is realistically calibrated, or whether S21's result is an artifact of misapplying a flat cost tier to a trade set that happens to skew illiquid.

**One-line answer:** The cost model is **reasonably calibrated** against external evidence — if anything conservative (safely higher, not underestimating) at the large end of the $100M–$2B range. S21's own trades are **not** concentrated in illiquid names (roughly uniform across market-cap deciles, median entry $790–826M — solidly mid-range). The extreme cost drag is a genuine consequence of trading frequency (4.9-day average holds, thousands of round trips), not miscalibration. **A sensitivity check confirms this is robust**: even cutting the spread assumption to 1/4 of its current value leaves every cost-flagged FAIL trial (S2, S4, S8, S21) still failing the Sharpe≥0.8 bar. **No adjustment to the standard cost model is recommended.**

---

## 1. S21's actual liquidity-tier breakdown

Reconstructed each of S21's 10 formation-cycle universes (reusing the exact same cached mcap/dollar-volume data the backtest itself used — no new fetching) and classified every traded member by market-cap decile and dollar-volume decile of the pooled $100M–$2B universe.

| | Basket (4,683 trades) | Peer (4,858 trades) |
|---|---:|---:|
| Bottom-3 mcap deciles (smallest names) | 28.2% | 24.1% |
| Top-3 mcap deciles (largest names) | 30.0% | 32.7% |
| Bottom-3 dollar-volume deciles (least liquid) | 32.6% | 30.7% |
| Top-3 dollar-volume deciles (most liquid) | 29.5% | 33.1% |
| Median entry market cap | $790M | $826M |
| Median entry avg. dollar volume | $5.83M/day | $6.48M/day |

Both the market-cap and dollar-volume decile distributions are close to **uniform** (each decile ≈7–14% of trades, vs. an expected 10% under uniformity) — there is no meaningful skew toward the illiquid end. The median traded name sits at ~$800M market cap, roughly two-thirds of the way up the $100M–$2B range, not near the illiquid floor. **S21's high average cost is not an artifact of the flat cost-tier model being misapplied to a systematically-illiquid trade selection — the trade set is a representative cross-section of the universe.** The cost driver is trading *frequency* (see §5), not trade *composition*.

## 2. Does Tiingo carry historical bid/ask data?

**No.** Confirmed two ways:
- Direct inspection of the cached price data (`research/mgmt_pit/cache/prices_all.pkl`, `adjclose_all.pkl`): columns are `date, open, high, low, close, volume` (and `adjClose` in the adjusted cache) — no bid, ask, or spread field anywhere.
- Every price-fetching script in this project (`research/ivol/run.py`, `research/russell_reconstitution/fetch_prices.py`) calls Tiingo's `/tiingo/daily/{ticker}/prices` **End-of-Day** endpoint exclusively. Per Tiingo's own API documentation, this endpoint returns OHLCV (+ dividend/split-adjusted variants) only. Tiingo's National Best Bid/Offer (bid/ask) data exists only on their separate **IEX real-time/websocket** product — a live/recent-quotes feed, not a historical daily archive reaching back to 2015 for small-cap names, and not something this project has ever fetched or cached.

**Conclusion: realized historical spreads cannot be computed directly from this project's data.** The cost model's spread assumption has always been, and must remain, an external calibration input rather than something back-tested against this project's own price history.

## 3. External estimates (since Tiingo has no bid/ask)

Multiple sources triangulate on modern (post-decimalization, i.e. post-2001) US small-cap spread levels:

- **SEC, "A characterization of market quality for small capitalization US equities" (2013)** — the single most directly comparable source: its market-cap bins (**<$100M, $100M–$250M, $250M–$500M, $500M–$1B, $1B–$2B, $2B–$5B**) match this project's own $100M–$2B universe almost exactly. Headline finding: the fraction of ticker-days with a quoted spread ≤2 cents rises monotonically with market cap — 0.3% (<$100M) → 1.9% ($100M–$250M) → 7.2% ($250M–$500M) → 34.9% ($500M–$1B) → **72.9%** ($1B–$2B). For a typical $20–40 small-cap price, 2 cents ≈ 5–10bps — meaning a clear majority of $1B–$2B names trade inside ~10bps on a typical day, while the great majority of sub-$250M names trade at meaningfully wider spreads. *(Full PDF blocked by SEC's bot-protection from automated fetch; figures triangulated via search-indexed citations of the same document.)*
- **Factor Investor / O'Shaughnessy Asset Management (2017), citing 1982–2016 data** — all-in cost estimate (spread + impact + commission combined) for a $10M trade: **~5bps** in the most liquid large-cap tier vs. **~220bps** in the least-liquid micro-cap tier. Note this is for a comparatively large $10M trade size relative to typical micro-cap ADV — the impact component would be smaller for the position sizes actually used in this project's backtests (equal-weighted legs of a modest research book, not $10M single-name clips), so this figure is best read as an upper bound on realistic spread-only cost, not a direct spread estimate.
- **SEC ATS-N filing data (FY2019)** — average NBBO spread for negotiated dark-pool/ATS executions in small/mid-cap names: **14.94bps**. This is institutional block-crossing (often midpoint-executed), so it understates lit-market retail-size quoted spreads, but confirms modern small/mid-cap spreads are an order of magnitude tighter than pre-decimalization estimates.
- **Loeb (1983)** — the classic reference (0.52% large-cap to 6.55% small-cap) is **not usable as a modern comparison**: it predates 2001 decimalization, after which US equity spreads collapsed by roughly 5–10x across the board. Cited here only to flag that any cost-model discussion citing Loeb-era numbers directly (without a decimalization adjustment) would be badly miscalibrated.

**Reading these against the model's tiers** (`SPREAD_MIN_BPS=20` at $2B, `SPREAD_MAX_BPS=100` at $100M, linear in between, one-way):
- At the **top of the range** ($1B–$2B, model ≈20bps): the SEC data's "≤2 cents on 73% of days" for a typical-priced name in this tier implies many names see spreads *tighter* than 20bps most days — the model's 20bps floor reads as reasonably calibrated to **slightly conservative** (safely at or above what a majority of names in this tier actually see, not underestimating cost).
- At the **bottom of the range** ($100M–$250M, model ≈100bps): only 1.9% of ticker-days clear the "≤2 cents" bar, and 2 cents represents a much larger bps figure at typical lower small-cap prices — consistent with spreads commonly exceeding 100bps in this tier on a meaningful share of days. The model's 100bps ceiling is in the right neighborhood, arguably still on the moderate side relative to the widest tail (worst days, thinnest names) in this bucket.

**Overall: the 20–100bps one-way tiered assumption is a realistic, if anything mildly conservative, calibration for this universe** — not evidence of systematic overstatement or understatement in either direction.

## 4. IBKR commission — confirmed not conflated with the cost model

IBKR's tiered commission (~$0.0035/share) is negligible relative to spread costs at any realistic small-cap share price — e.g. a $20 stock at $0.0035/share ≈ **1.75bps**, roughly 1/10th to 1/50th of the model's 20–100bps spread assumption. Directly confirmed by reading `_spread()`/`_borrow()` in `research/gp/run.py` (and identically in every other trial's cost model, including S21's): both functions are pure linear-interpolation-by-market-cap formulas returning a bps-of-notional figure — **there is no per-share or per-trade commission term anywhere in the code**. The cost model is exclusively a market-microstructure (spread + securities-lending) model; it does not conflate or double-count brokerage commission, and commission would not be a material addition even if it were included.

## 5. Would a corrected cost assumption flip any FAIL verdict?

Recomputed net Sharpe/CALMAR for the four trials explicitly flagged as cost-driven FAILs (S2 PEAD, S4 MAX, S8 residual reversal, S21 cluster reversion) at k ∈ {0, 0.25, 0.5, 1.0, 1.5, 2.0} × the current bid-ask spread assumption, holding the separate borrow/securities-lending rate fixed (that's a different assumption, not what this audit is about). Monthly-aggregated series, same Sharpe/CALMAR formula used throughout the registry.

| Trial | k=0 (no spread cost) | k=0.25 | k=0.5 | k=1.0 (current) | k=2.0 | Crosses Sharpe≥0.8 at any k>0? |
|---|---:|---:|---:|---:|---:|---|
| S2-PEAD | Sharpe −0.366 | −1.052 | −1.713 | −2.843 | −4.170 | **No** — gross itself is negative |
| S4-MAX | Sharpe **+0.330** | −0.068 | −0.466 | −1.264 | −2.855 | **No** — even zero cost falls short of 0.8 |
| S8-ResidRev | Sharpe −0.446 | −0.612 | −0.778 | −1.107 | −1.746 | **No** — gross itself is negative |
| S21-basket | Sharpe **+0.954** | −3.750 | −8.349 | −16.589 | −27.663 | **No** (only at the degenerate k=0) |

**Two distinct sub-findings:**

- **S2 (PEAD) and S8 (residual reversal) are not primarily cost stories at all** — their gross returns are already negative before any spread cost is applied (k=0 Sharpe still −0.37 and −0.45 respectively). No plausible spread recalibration rescues a strategy with negative gross expectancy; these fails are about signal economics, not cost-model calibration.
- **S4 (MAX) and S21 (cluster reversion) do have genuinely positive gross Sharpe** (0.33 and 0.95), so cost calibration is the actually relevant question for these two — and the answer is the same either way: **no realistic (non-zero) recalibration changes the verdict.** S21 in particular is instructive: even cutting the spread assumption to a quarter of its current value (5–25bps one-way, below even the SEC data's tightest-tier evidence) still produces a Sharpe of −3.75, because the problem is trading *frequency* (round trips roughly weekly), not the per-trade cost rate — a lower bps assumption gets multiplied by the same enormous turnover and still compounds catastrophically. **The actual lever that would matter for S21 is a lower-turnover construction (a genuinely new trial), not a different cost assumption on the same construction.**

## Recommendation

**No change to the standard cost model is warranted.** It is reasonably calibrated against the best available external evidence (SEC's own small-cap market-quality study, using nearly identical market-cap bins to this project's universe), does not conflate spread/impact with brokerage commission, and — most importantly — every registry FAIL verdict currently attributed to cost is robust across a wide, externally-plausible range of alternative cost assumptions. If the model is imprecise anywhere, the available evidence points to it being mildly conservative (safely higher, not lower) at the liquid end of the range, which if anything means genuine trading costs for the $1B–$2B tier could be modestly *lower* than currently modeled — a direction that would make the registry's FAIL verdicts *more* likely to hold, not less.

## Data / methods

- Liquidity-tier audit: `research/clusterreversion_S21/liquidity_audit.py` → `research/clusterreversion_S21/results/{liquidity_audit_summary.json, liquidity_audit_basket.csv, liquidity_audit_peer.csv}`. Reused the exact same universe-construction code (`gp_industry_neutral.build_universe`) and cached price/fact data the S21 backtest itself used — no new data sourcing.
- Cost sensitivity: `research/clusterreversion_S21/cost_sensitivity.py` → `research/clusterreversion_S21/results/cost_sensitivity.csv`. Reused each trial's own already-saved period-return CSVs (`research/pead/results/pead_daily_returns.csv`, `research/max/results/max_period_returns.csv`, `research/residual_reversal/results/s8_period_returns.csv`, S21's `daily_book_returns_basket.csv`), isolating the `ls_ba`/`ls_ba_cost` (bid-ask) columns from `ls_bw`/`ls_borrow_cost` (borrow) where both are separately saved (S2, S4, S8); S21's daily cost series wasn't split at capture time, but borrow is provably negligible there (5-day holds; verified borrow accrual ≤~0.0000274/day), so its `cost` column is treated as ~100% bid-ask for this check — a disclosed, minor approximation.

Sources: [SEC — A characterization of market quality for small capitalization US equities](https://www.sec.gov/marketstructure/research/small_cap_liquidity.pdf) · [Factor Investor — Micro Caps, Factor Spreads, Structural Biases](https://www.factorinvestor.com/blog/micro-caps-factor-spreads-structural-biases-and-the-institutional-imperative) · [Tiingo — End-of-Day Stock Price API Documentation](https://www.tiingo.com/documentation/end-of-day) · [Tiingo — Real-time & Historical IEX API Documentation](https://www.tiingo.com/documentation/iex)
