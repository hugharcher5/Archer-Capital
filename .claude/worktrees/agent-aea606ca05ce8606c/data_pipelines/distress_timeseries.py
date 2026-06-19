"""
Point-in-time weekly distress score for every private competitor in
confirmed_companies.COMPETITORS, aggregated to listed LSE tickers.

CRITICAL — NO LOOK-AHEAD
    pit_score_on_date(cd, d) uses ONLY accounts whose filed_at <= d.
    The score on any date D reflects only information that was publicly
    available at Companies House on that date.

Usage:
    python data_pipelines/distress_timeseries.py
    # Writes:
    #   strategies/private_competitor_distress/results/distress_timeseries.csv
    #   strategies/private_competitor_distress/results/dormant_shell_exclusions.csv
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from data_pipelines.companies_house import CompaniesHouseClient
from data_pipelines.confirmed_companies import COMPETITORS

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Constants ──────────────────────────────────────────────────────────────────

SERIES_START = date(2010, 1, 1)
DEFUNCT_WINDOW_DAYS = 180      # days of elevated distress after insolvency
SANITY_GATE_DAYS = 540         # >540 days overdue + active → exclude as dormant shell
SCORE_HALF_LIFE = 180.0        # delay_to_score(180) ≈ 0.63, delay_to_score(365) ≈ 0.87

CACHE_DIR = (
    Path(__file__).parent.parent
    / "strategies"
    / "private_competitor_distress"
    / "results"
    / "cache"
)
RESULTS_DIR = (
    Path(__file__).parent.parent
    / "strategies"
    / "private_competitor_distress"
    / "results"
)


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class AccountsFiling:
    period_end: date
    filed_at: date | None        # None means not yet filed (outstanding)
    due_date: date               # computed: period_end + 9mo (private) or 6mo (plc)


@dataclass
class CompanyData:
    ch_number: str
    name: str
    company_type: str
    current_status: str
    incorporation_date: date | None
    accounts: list[AccountsFiling] = field(default_factory=list)  # sorted by period_end ASC
    insolvency_date: date | None = None   # date of first insolvency/liquidation filing
    excluded_dormant: bool = False


# ── Deadline helper ────────────────────────────────────────────────────────────

def _accounts_deadline(period_end: date, company_type: str) -> date:
    """Statutory filing deadline: 6 months (plc) or 9 months (private)."""
    months = 6 if company_type in ("plc", "public-limited-company") else 9
    month = period_end.month + months
    year = period_end.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    try:
        return period_end.replace(year=year, month=month)
    except ValueError:
        import calendar
        return date(year, month, calendar.monthrange(year, month)[1])


def _add_one_year(d: date) -> date:
    """Add one year, handling Feb 29 safely."""
    try:
        return d.replace(year=d.year + 1)
    except ValueError:
        # Feb 29 in a leap year → Feb 28 in next year
        return d.replace(year=d.year + 1, day=28)


# ── Score function ─────────────────────────────────────────────────────────────

def delay_to_score(days: float) -> float:
    """Saturating exponential. delay_to_score(180) ≈ 0.63, (365) ≈ 0.87."""
    return float(1 - np.exp(-max(days, 0) / SCORE_HALF_LIFE))


# ── JSON serialisation helpers ─────────────────────────────────────────────────

def _date_to_str(d: date | None) -> str | None:
    return d.isoformat() if d is not None else None


def _str_to_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


# ── API fetch + disk cache ─────────────────────────────────────────────────────

def fetch_company_data(ch_number: str, client: CompaniesHouseClient) -> CompanyData | None:
    """
    Fetch company profile + full accounts history + insolvency/liquidation filings.
    Saves raw data to CACHE_DIR/{ch_number}.json to avoid re-fetching.
    Returns None if the company profile cannot be retrieved.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{ch_number}.json"

    # ── Try loading from disk cache first ─────────────────────────────────────
    if cache_path.exists():
        try:
            with cache_path.open(encoding="utf-8") as fh:
                raw = json.load(fh)
            return _deserialise_company_data(raw)
        except Exception as exc:
            log.warning("Cache read failed for %s (%s); re-fetching.", ch_number, exc)

    # ── Fetch from API ────────────────────────────────────────────────────────
    profile = client.get_company_profile(ch_number)
    if profile is None:
        log.warning("Could not fetch profile for %s — skipping.", ch_number)
        return None

    company_type   = profile.get("type", "")
    current_status = profile.get("status", "")
    name           = profile.get("name", "")
    inc_date       = profile.get("incorporation_date")  # already a date object

    # ── Accounts filings (all historical — up to 50) ──────────────────────────
    raw_accounts = client.get_filing_history(
        ch_number, category="accounts", max_items=50
    )

    accounts: list[AccountsFiling] = []
    for item in raw_accounts:
        period_end = item.get("made_up_date")
        filed_at   = item.get("date")
        if period_end is None:
            continue
        due = _accounts_deadline(period_end, company_type)
        accounts.append(AccountsFiling(
            period_end=period_end,
            filed_at=filed_at,
            due_date=due,
        ))

    # Sort ascending by period_end
    accounts.sort(key=lambda a: a.period_end)

    # ── Insolvency / liquidation filings ──────────────────────────────────────
    insolvency_filings = client.get_filing_history(
        ch_number, category="insolvency", max_items=10
    )
    liquidation_filings = client.get_filing_history(
        ch_number, category="liquidation", max_items=10
    )

    all_insolvency = insolvency_filings + liquidation_filings
    insolvency_date: date | None = None

    if all_insolvency:
        dates = [item["date"] for item in all_insolvency if item.get("date") is not None]
        if dates:
            insolvency_date = min(dates)

    # Fallback: if status implies insolvency but no filings found, use today
    if insolvency_date is None and current_status in ("administration", "liquidation"):
        insolvency_date = date.today()

    # ── Build cache payload and write ─────────────────────────────────────────
    cache_payload = {
        "ch_number":         ch_number,
        "name":              name,
        "company_type":      company_type,
        "current_status":    current_status,
        "incorporation_date": _date_to_str(inc_date),
        "accounts": [
            {
                "period_end": _date_to_str(a.period_end),
                "filed_at":   _date_to_str(a.filed_at),
                "due_date":   _date_to_str(a.due_date),
            }
            for a in accounts
        ],
        "insolvency_date": _date_to_str(insolvency_date),
    }
    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(cache_payload, fh, indent=2)
    except OSError as exc:
        log.warning("Could not write cache for %s: %s", ch_number, exc)

    return CompanyData(
        ch_number=ch_number,
        name=name,
        company_type=company_type,
        current_status=current_status,
        incorporation_date=inc_date,
        accounts=accounts,
        insolvency_date=insolvency_date,
    )


def _deserialise_company_data(raw: dict) -> CompanyData:
    """Reconstruct CompanyData from the JSON cache dict."""
    accounts: list[AccountsFiling] = []
    for a in raw.get("accounts", []):
        pe = _str_to_date(a.get("period_end"))
        if pe is None:
            continue
        filed = _str_to_date(a.get("filed_at"))
        due   = _str_to_date(a.get("due_date"))
        if due is None:
            # Recompute if missing from cache
            ctype = raw.get("company_type", "")
            due = _accounts_deadline(pe, ctype)
        accounts.append(AccountsFiling(period_end=pe, filed_at=filed, due_date=due))
    accounts.sort(key=lambda a: a.period_end)

    return CompanyData(
        ch_number=raw.get("ch_number", ""),
        name=raw.get("name", ""),
        company_type=raw.get("company_type", ""),
        current_status=raw.get("current_status", ""),
        incorporation_date=_str_to_date(raw.get("incorporation_date")),
        accounts=accounts,
        insolvency_date=_str_to_date(raw.get("insolvency_date")),
    )


# ── Sanity gate ────────────────────────────────────────────────────────────────

def check_sanity_gate(cd: CompanyData, today: date) -> bool:
    """
    Returns True → exclude as likely dormant shell.
    Trigger: status == 'active' AND the NEXT expected accounting period is
    more than SANITY_GATE_DAYS days overdue with no filing yet filed.
    """
    if cd.current_status != "active":
        return False
    if not cd.accounts:
        return False

    last = max(cd.accounts, key=lambda a: a.period_end)
    next_pe  = _add_one_year(last.period_end)
    next_due = _accounts_deadline(next_pe, cd.company_type)

    # Has a filing already been made for the next period?
    has_next = any(
        a.period_end >= next_pe and a.filed_at is not None
        for a in cd.accounts
    )
    if has_next:
        return False

    overdue = (today - next_due).days
    return overdue > SANITY_GATE_DAYS


# ── Point-in-time score ────────────────────────────────────────────────────────

def pit_score_on_date(cd: CompanyData, d: date) -> float:
    """
    Compute the distress score for company cd as of date d.

    CRITICAL: uses ONLY filings where filed_at <= d.
    No look-ahead: the score on d reflects only what was known at d.

    Returns a float in [0.0, 1.0].
    """
    # 1. Before incorporation → no distress signal possible
    if cd.incorporation_date is not None and d < cd.incorporation_date:
        return 0.0

    # 2. Insolvency handling
    if cd.insolvency_date is not None:
        if d < cd.insolvency_date:
            pass  # fall through to normal scoring
        elif d <= cd.insolvency_date + timedelta(days=DEFUNCT_WINDOW_DAYS):
            return 1.0
        else:
            # Company defunct and window has passed
            return 0.0

    # 3. Known filings up to date d (only those already filed by d)
    known_filings: list[AccountsFiling] = [
        a for a in cd.accounts
        if a.filed_at is not None and a.filed_at <= d
    ]

    # 4. No known filings yet — but company is old enough to have filed
    if not known_filings:
        if cd.incorporation_date is None:
            return 0.0
        age_days = (d - cd.incorporation_date).days
        if age_days > 550:
            estimated_first_due = cd.incorporation_date + timedelta(days=550)
            overdue_days = max(0, (d - estimated_first_due).days)
            return delay_to_score(overdue_days)
        return 0.0

    # 5. Most recent known filing (by period_end)
    latest = max(known_filings, key=lambda f: f.period_end)

    # Score for lateness of the most recent known filing
    late_days = max(0, (latest.filed_at - latest.due_date).days)  # type: ignore[operator]
    score_filed = delay_to_score(late_days)

    # 6. Check whether the NEXT expected period is now overdue as of d
    next_pe  = _add_one_year(latest.period_end)
    next_due = _accounts_deadline(next_pe, cd.company_type)

    if next_due <= d:
        has_next_filed = any(
            f.period_end >= next_pe and f.filed_at is not None and f.filed_at <= d
            for f in known_filings
        )
        if not has_next_filed:
            score_overdue = delay_to_score((d - next_due).days)
            return max(score_filed, score_overdue)

    # 7. No overdue next period — return score based on timeliness of latest filing
    return score_filed


# ── Per-company score series ───────────────────────────────────────────────────

def build_company_series(
    cd: CompanyData,
    weekly_dates: list[date],
) -> list[float]:
    """Build a PIT score for each weekly date. Returns list aligned to weekly_dates."""
    return [pit_score_on_date(cd, d) for d in weekly_dates]


# ── Weekly date generator ──────────────────────────────────────────────────────

def weekly_fridays(start: date, end: date) -> list[date]:
    """Return every Friday from start to end (inclusive)."""
    # Fast-forward to first Friday
    days_ahead = (4 - start.weekday()) % 7   # 4 = Friday
    current = start + timedelta(days=days_ahead)
    fridays: list[date] = []
    while current <= end:
        fridays.append(current)
        current += timedelta(weeks=1)
    return fridays


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_timeseries(force_refetch: bool = False) -> pd.DataFrame:
    """
    Full pipeline. Fetches (or loads from cache) CH data for all confirmed
    competitors, applies the sanity gate, builds PIT weekly distress scores,
    and aggregates to listed LSE tickers.

    Returns a long-format DataFrame with columns:
        date, ticker, max_distress, equal_weight_distress

    Note: 'equal_weight_distress' is written as 'weighted_distress' in the
    output CSV as a placeholder column name; it is equal-weighted mean, NOT
    revenue-weighted. Revenue weighting requires iXBRL parsing and is future work.

    Also writes:
        results/distress_timeseries.csv
        results/dormant_shell_exclusions.csv
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if force_refetch:
        # Wipe cache so everything is re-fetched
        for cache_file in CACHE_DIR.glob("*.json"):
            cache_file.unlink()
        log.info("Cache cleared (force_refetch=True).")

    client = CompaniesHouseClient()
    today  = date.today()

    # ── Build ch_number → tickers mapping (deduplicated) ─────────────────────
    # A ch_number can appear in multiple groups (e.g. if curated from different sectors)
    ch_to_tickers: dict[str, set[str]] = {}
    ch_to_name: dict[str, str] = {}

    for group in COMPETITORS.values():
        tickers = group["tickers"]
        for co in group["companies"]:
            num = co["ch_number"]
            ch_to_tickers.setdefault(num, set()).update(tickers)
            if num not in ch_to_name:
                ch_to_name[num] = co["name"]

    all_ch_numbers = list(ch_to_tickers.keys())
    log.info("Loading data for %d unique companies …", len(all_ch_numbers))

    # ── Fetch / load from cache ───────────────────────────────────────────────
    company_data: dict[str, CompanyData] = {}
    failed: list[str] = []

    for i, ch_number in enumerate(all_ch_numbers, 1):
        cache_path = CACHE_DIR / f"{ch_number}.json"
        if cache_path.exists():
            try:
                with cache_path.open(encoding="utf-8") as fh:
                    raw = json.load(fh)
                cd = _deserialise_company_data(raw)
                company_data[ch_number] = cd
                log.debug("[%d/%d] Loaded from cache: %s", i, len(all_ch_numbers), ch_number)
                continue
            except Exception as exc:
                log.warning("Cache corrupt for %s (%s); re-fetching.", ch_number, exc)

        log.info("[%d/%d] Fetching: %s  %s", i, len(all_ch_numbers), ch_number, ch_to_name.get(ch_number, ""))
        cd = fetch_company_data(ch_number, client)
        if cd is None:
            failed.append(ch_number)
            continue
        company_data[ch_number] = cd

    if failed:
        log.warning("Failed to fetch %d companies: %s", len(failed), failed)

    # ── Apply sanity gate ─────────────────────────────────────────────────────
    dormant_exclusions: list[dict] = []
    for ch_number, cd in company_data.items():
        if check_sanity_gate(cd, today):
            cd.excluded_dormant = True
            log.info("DORMANT SHELL EXCLUDED: %s  %s", ch_number, cd.name)
            dormant_exclusions.append({
                "ch_number": ch_number,
                "name":      cd.name,
                "status":    cd.current_status,
                "tickers":   ",".join(sorted(ch_to_tickers.get(ch_number, set()))),
                "reason":    "sanity_gate: next period >540d overdue, status=active",
            })

    # Print exclusions clearly
    if dormant_exclusions:
        print(f"\n{'='*60}")
        print(f"  DORMANT SHELL EXCLUSIONS ({len(dormant_exclusions)} companies)")
        print(f"{'='*60}")
        for exc in dormant_exclusions:
            print(f"  {exc['ch_number']:<12}  {exc['name'][:40]:<41}  [{exc['tickers']}]")
        print()

    # ── Write dormant exclusions CSV ──────────────────────────────────────────
    excl_path = RESULTS_DIR / "dormant_shell_exclusions.csv"
    if dormant_exclusions:
        excl_df = pd.DataFrame(dormant_exclusions)
        excl_df.to_csv(excl_path, index=False)
        log.info("Wrote %d dormant exclusions → %s", len(dormant_exclusions), excl_path)
    else:
        # Write empty file with headers so downstream code can always load it
        pd.DataFrame(columns=["ch_number", "name", "status", "tickers", "reason"]).to_csv(
            excl_path, index=False
        )

    # ── Build weekly date range ───────────────────────────────────────────────
    weekly_dates = weekly_fridays(SERIES_START, today)
    log.info("Building timeseries: %d weekly dates from %s to %s",
             len(weekly_dates), weekly_dates[0], weekly_dates[-1])

    # ── Compute per-company PIT series ────────────────────────────────────────
    # Structure: ch_number → list[float] aligned to weekly_dates
    company_series: dict[str, list[float]] = {}

    for ch_number, cd in company_data.items():
        if cd.excluded_dormant:
            continue
        company_series[ch_number] = build_company_series(cd, weekly_dates)
        log.debug("Built series for %s  %s", ch_number, cd.name)

    # ── Aggregate per (date, ticker) ──────────────────────────────────────────
    # Build ticker → list of ch_numbers (non-excluded)
    ticker_to_chs: dict[str, list[str]] = {}
    for ch_number in company_series:
        for ticker in ch_to_tickers.get(ch_number, set()):
            ticker_to_chs.setdefault(ticker, []).append(ch_number)

    rows: list[dict] = []

    for i, d in enumerate(weekly_dates):
        for ticker, chs in ticker_to_chs.items():
            scores = [company_series[ch][i] for ch in chs if ch in company_series]
            if not scores:
                continue
            max_dist   = float(np.max(scores))
            eq_w_dist  = float(np.mean(scores))
            rows.append({
                "date":                d,
                "ticker":              ticker,
                "max_distress":        round(max_dist, 6),
                # NOTE: named 'weighted_distress' in output but is equal-weight mean.
                # Revenue weighting requires iXBRL parsing — future work.
                "weighted_distress":   round(eq_w_dist, 6),
            })

    ts = pd.DataFrame(rows)
    if not ts.empty:
        ts["date"] = pd.to_datetime(ts["date"])
        ts = ts.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Write timeseries CSV ──────────────────────────────────────────────────
    ts_path = RESULTS_DIR / "distress_timeseries.csv"
    ts.to_csv(ts_path, index=False)
    log.info("Wrote %d rows → %s", len(ts), ts_path)

    return ts


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PIT weekly distress timeseries")
    parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Wipe local cache and re-fetch all Companies House data",
    )
    args = parser.parse_args()

    ts = run_timeseries(force_refetch=args.force_refetch)

    print(f"\n=== TIMESERIES SUMMARY ===")
    print(f"  Rows:    {len(ts):,}")
    if not ts.empty:
        print(f"  Dates:   {ts['date'].min().date()} → {ts['date'].max().date()}")
        print(f"  Tickers: {sorted(ts['ticker'].unique().tolist())}")
        print(f"\n  Latest week (tail):")
        latest = ts[ts["date"] == ts["date"].max()].copy()
        latest = latest.sort_values("max_distress", ascending=False)
        print(latest.to_string(index=False))
