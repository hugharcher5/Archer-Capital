"""
Private Competitor Distress Pipeline.

Fetches Companies House data for all confirmed competitors and computes a
composite distress score per company, then aggregates to listed tickers.

Scoring components (weights sum to 1.0):
    accounts_delay  (0.40)  — days late / 365, clamped [0, 1]
    status          (0.30)  — 0=active, 0.5=dissolved, 0.8=admin, 1.0=liquidation
    overdue         (0.20)  — 1.0 if accounts deadline passed with no filing
    confirmation    (0.10)  — late confirmation-statement (same delay logic)

Usage:
    python data_pipelines/distress_pipeline.py
    # writes results to strategies/private_competitor_distress/results/
"""

from __future__ import annotations

import csv
import logging
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from data_pipelines.companies_house import CompaniesHouseClient
from data_pipelines.confirmed_companies import COMPETITORS, all_company_numbers

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

_RESULTS_DIR = Path(__file__).parent.parent / "strategies" / "private_competitor_distress" / "results"

# ── Scoring constants ─────────────────────────────────────────────────────────

_STATUS_SCORE: dict[str, float] = {
    "active":       0.00,
    "dissolved":    0.50,
    "administration": 0.80,
    "liquidation":  1.00,
    "receivership": 1.00,
    "converted-closed": 0.40,
    "closed":       0.40,
    "closed-on":    0.40,
    "registered":   0.00,
}

def _accounts_deadline(period_end: date, company_type: str) -> date:
    """Statutory filing deadline: 6 months (plc) or 9 months (private) after period end."""
    months = 6 if company_type in ("plc", "public-limited-company") else 9
    month = period_end.month + months
    year = period_end.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    try:
        return period_end.replace(year=year, month=month)
    except ValueError:
        # e.g. Feb 29 → Feb 28 in non-leap year
        import calendar
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, min(period_end.day, last_day))


def compute_distress_score(
    profile: dict | None,
    filings: list[dict],
    today: date | None = None,
) -> dict:
    """
    Return a score dict for one company.

    Keys: score, accounts_delay, status_score, overdue_score, confirmation_score,
          days_late, period_end, filed_at, company_status, accounts_type, flags
    """
    today = today or date.today()

    # Defaults
    status_score        = 0.0
    accounts_delay      = 0.0
    overdue_score       = 0.0
    confirmation_score  = 0.0
    days_late           = 0
    period_end          = None
    filed_at            = None
    accounts_type       = ""
    company_status      = ""
    company_type        = ""
    flags: list[str]    = []

    if profile:
        company_status = profile.get("status", "")
        company_type   = profile.get("type", "")
        status_score   = _STATUS_SCORE.get(company_status, 0.0)
        if company_status in ("administration", "liquidation", "receivership"):
            flags.append(company_status.upper())

    # Accounts component
    accounts_filings = [f for f in filings if f.get("category") == "accounts" and f.get("made_up_date")]
    if accounts_filings:
        latest = accounts_filings[0]  # CH returns newest first
        period_end    = latest["made_up_date"]
        filed_at      = latest["date"]
        accounts_type = latest.get("type", "")
        deadline      = _accounts_deadline(period_end, company_type)

        if filed_at:
            raw_late   = (filed_at - deadline).days
            days_late  = max(raw_late, 0)
            accounts_delay = min(days_late / 365, 1.0)
            if days_late > 30:
                flags.append(f"ACCOUNTS_LATE_{days_late}d")
        else:
            # Filed but no date recorded — treat as on-time
            pass

    elif profile and profile.get("incorporation_date"):
        # Company old enough to have filed at least once but hasn't
        inc = profile["incorporation_date"]
        if inc and (today - inc).days > 550:
            overdue_score = 0.9
            flags.append("NO_ACCOUNTS_EVER")

    # Overdue component: deadline passed and still no filing for the expected period
    if period_end and not filed_at:
        deadline = _accounts_deadline(period_end, company_type)
        if today > deadline:
            overdue_score = 1.0
            flags.append(f"OVERDUE_SINCE_{deadline}")
    elif period_end:
        # Check if next period is now overdue (period_end was >12 months ago)
        next_period_end = period_end.replace(year=period_end.year + 1)
        if next_period_end < today:
            next_deadline = _accounts_deadline(next_period_end, company_type)
            if today > next_deadline:
                overdue_days = (today - next_deadline).days
                overdue_score = min(overdue_days / 180, 1.0)
                if overdue_days > 14:
                    flags.append(f"NEXT_PERIOD_OVERDUE_{overdue_days}d")

    # Confirmation statement component
    conf_filings = [f for f in filings if f.get("category") == "confirmation-statement" and f.get("date")]
    if conf_filings:
        latest_conf = conf_filings[0]
        conf_date   = latest_conf["date"]
        if conf_date:
            conf_overdue_days = (today - conf_date).days - 365 - 14  # annual + 14d grace
            if conf_overdue_days > 0:
                confirmation_score = min(conf_overdue_days / 180, 1.0)
                flags.append(f"CONF_OVERDUE_{conf_overdue_days}d")

    composite = (
        0.40 * accounts_delay
        + 0.30 * status_score
        + 0.20 * overdue_score
        + 0.10 * confirmation_score
    )

    return {
        "score":               round(composite, 4),
        "accounts_delay":      round(accounts_delay, 4),
        "status_score":        round(status_score, 4),
        "overdue_score":       round(overdue_score, 4),
        "confirmation_score":  round(confirmation_score, 4),
        "days_late":           days_late,
        "period_end":          str(period_end) if period_end else "",
        "filed_at":            str(filed_at) if filed_at else "",
        "company_status":      company_status,
        "company_type":        company_type,
        "accounts_type":       accounts_type,
        "flags":               "|".join(flags),
    }


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(dry_run: bool = False) -> dict[str, list[dict]]:
    """
    Fetch CH data for all confirmed competitors and compute distress scores.

    Returns
    -------
    {
        "company_scores":  list[dict],   # one row per company
        "ticker_signals":  list[dict],   # one row per listed ticker
    }
    Writes two CSVs to strategies/private_competitor_distress/results/.
    """
    client  = CompaniesHouseClient()
    today   = date.today()

    # Build reverse lookup: ch_number → (group_key, query_name, tickers, notes)
    ch_meta: dict[str, dict] = {}
    for group_key, group in COMPETITORS.items():
        tickers = group["tickers"]
        for co in group["companies"]:
            num = co["ch_number"]
            if num not in ch_meta:
                ch_meta[num] = {
                    "group":    group_key,
                    "name":     co["name"],
                    "tickers":  tickers,
                    "notes":    co.get("notes", ""),
                }

    all_nums = list(ch_meta.keys())
    log.info("Fetching CH data for %d companies …", len(all_nums))

    company_rows: list[dict] = []

    for i, num in enumerate(all_nums, 1):
        meta = ch_meta[num]
        log.info("[%d/%d] %s  %s", i, len(all_nums), num, meta["name"])

        if dry_run:
            profile  = None
            filings  = []
        else:
            profile   = client.get_company_profile(num)
            # Fetch accounts and confirmations separately — unfiltered history
            # only returns 20 items and may miss accounts for prolific filers.
            accounts_filings = client.get_filing_history(num, category="accounts", max_items=5)
            conf_filings     = client.get_filing_history(num, category="confirmation-statement", max_items=3)
            filings          = accounts_filings + conf_filings

        scores = compute_distress_score(profile, filings, today)

        company_rows.append({
            "ch_number":    num,
            "name":         meta["name"],
            "group":        meta["group"],
            "tickers":      ",".join(meta["tickers"]),
            "notes":        meta["notes"],
            **scores,
        })

    # ── Aggregate by ticker ───────────────────────────────────────────────────

    from collections import defaultdict
    ticker_buckets: dict[str, list[dict]] = defaultdict(list)
    for row in company_rows:
        for ticker in row["tickers"].split(","):
            ticker_buckets[ticker.strip()].append(row)

    DISTRESS_THRESHOLD = 0.35

    ticker_rows: list[dict] = []
    for ticker, rows in sorted(ticker_buckets.items()):
        distressed = [r for r in rows if r["score"] >= DISTRESS_THRESHOLD]
        if not rows:
            continue
        avg_score = sum(r["score"] for r in rows) / len(rows)
        max_score = max(r["score"] for r in rows)
        max_co    = max(rows, key=lambda r: r["score"])
        ticker_rows.append({
            "ticker":               ticker,
            "n_competitors":        len(rows),
            "n_distressed":         len(distressed),
            "pct_distressed":       round(len(distressed) / len(rows), 3),
            "avg_distress_score":   round(avg_score, 4),
            "max_distress_score":   round(max_score, 4),
            "most_distressed_co":   max_co["name"],
            "most_distressed_num":  max_co["ch_number"],
            "signal":               "LONG" if avg_score >= DISTRESS_THRESHOLD else "",
        })

    # ── Write results ─────────────────────────────────────────────────────────

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    co_path = _RESULTS_DIR / "distress_scores.csv"
    tk_path = _RESULTS_DIR / "ticker_signals.csv"

    _write_csv(co_path, company_rows)
    _write_csv(tk_path, ticker_rows)

    log.info("Results written to %s", _RESULTS_DIR)
    return {"company_scores": company_rows, "ticker_signals": ticker_rows}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info("Wrote %d rows → %s", len(rows), path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run private competitor distress pipeline")
    p.add_argument("--dry-run", action="store_true", help="Skip API calls; score with empty data")
    args = p.parse_args()

    results = run_pipeline(dry_run=args.dry_run)

    print("\n=== TOP DISTRESSED COMPETITORS ===")
    sorted_cos = sorted(results["company_scores"], key=lambda r: r["score"], reverse=True)
    for r in sorted_cos[:20]:
        print(f"  {r['score']:.3f}  {r['ch_number']:<12}  {r['name'][:40]:<41}  [{r['tickers']}]  {r['flags']}")

    print("\n=== TICKER SIGNALS ===")
    sorted_tks = sorted(results["ticker_signals"], key=lambda r: r["avg_distress_score"], reverse=True)
    for r in sorted_tks:
        sig = f"  *** SIGNAL: {r['signal']} ***" if r["signal"] else ""
        print(f"  {r['ticker']:<6}  avg={r['avg_distress_score']:.3f}  max={r['max_distress_score']:.3f}"
              f"  {r['n_distressed']}/{r['n_competitors']} distressed{sig}")
