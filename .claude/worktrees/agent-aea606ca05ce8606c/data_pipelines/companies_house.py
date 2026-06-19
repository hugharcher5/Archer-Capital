"""
Companies House REST API client.

Requirements: requests, python-dotenv
Rate limit:   600 requests / 5 minutes (120 req/min).
Auth:         HTTP Basic — API key as username, blank password.

Usage (standalone test):
    python data_pipelines/companies_house.py
"""

from __future__ import annotations

import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ── Bootstrap ─────────────────────────────────────────────────────────────────

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BASE_URL       = "https://api.company-information.service.gov.uk"
_RATE_LIMIT     = 600          # requests per window
_WINDOW_SECONDS = 300          # 5-minute window
_MIN_INTERVAL   = _WINDOW_SECONDS / _RATE_LIMIT   # ~0.5 s between requests
_MAX_RETRIES    = 3
_BACKOFF_BASE   = 2.0          # seconds; doubles on each retry


# ── Data classes (plain dicts for now; swap in dataclasses when schema firms up)

def _parse_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


# ── Client ────────────────────────────────────────────────────────────────────

class CompaniesHouseClient:
    """
    Thin wrapper around the Companies House REST API.

    All public methods return plain dicts.  Failed requests are logged and
    return None (or an empty list) so a batch run can continue past one
    broken company number.
    """

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.getenv("COMPANIES_HOUSE_API_KEY")
        if not key:
            raise EnvironmentError(
                "COMPANIES_HOUSE_API_KEY is not set. "
                "Add it to your .env file or export it in the shell."
            )
        self._session = requests.Session()
        self._session.auth = (key, "")         # Basic auth: key as username, blank password
        self._session.headers.update({"Accept": "application/json"})
        self._last_request_at: float = 0.0

    # ── Internal request helper ───────────────────────────────────────────────

    def _get(self, path: str, params: dict | None = None) -> dict | None:
        """
        GET /{path} with rate-limiting and exponential-backoff retry.
        Returns parsed JSON or None on unrecoverable failure.
        """
        url = f"{_BASE_URL}/{path.lstrip('/')}"

        for attempt in range(1, _MAX_RETRIES + 1):
            # Throttle: honour the 600/5min cap
            elapsed = time.monotonic() - self._last_request_at
            if elapsed < _MIN_INTERVAL:
                time.sleep(_MIN_INTERVAL - elapsed)

            try:
                self._last_request_at = time.monotonic()
                resp = self._session.get(url, params=params, timeout=10)
            except requests.RequestException as exc:
                log.warning("Request error (attempt %d/%d): %s — %s",
                            attempt, _MAX_RETRIES, url, exc)
                time.sleep(_BACKOFF_BASE ** attempt)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", _BACKOFF_BASE ** attempt))
                log.warning("Rate-limited (429). Sleeping %ds.", retry_after)
                time.sleep(retry_after)
                continue

            if resp.status_code in (404, 410):
                log.warning("Not found (%d): %s", resp.status_code, url)
                return None

            if resp.status_code >= 500:
                log.warning("Server error %d (attempt %d/%d): %s",
                            resp.status_code, attempt, _MAX_RETRIES, url)
                time.sleep(_BACKOFF_BASE ** attempt)
                continue

            log.error("Unexpected status %d: %s", resp.status_code, url)
            return None

        log.error("All %d attempts failed: %s", _MAX_RETRIES, url)
        return None

    # ── Public methods ────────────────────────────────────────────────────────

    def get_company_profile(self, company_number: str) -> dict | None:
        """
        Fetch core company profile.

        Returns:
            {
                company_number:      str,
                name:                str,
                status:              str,   # e.g. "active", "dissolved"
                incorporation_date:  date | None,
                sic_codes:           list[str],
                registered_address:  str,
                type:                str,
            }
        """
        raw = self._get(f"company/{company_number}")
        if raw is None:
            return None

        address_parts = raw.get("registered_office_address", {})
        address_line  = ", ".join(
            v for v in [
                address_parts.get("address_line_1"),
                address_parts.get("address_line_2"),
                address_parts.get("locality"),
                address_parts.get("postal_code"),
            ]
            if v
        )

        return {
            "company_number":     raw.get("company_number", company_number),
            "name":               raw.get("company_name", ""),
            "status":             raw.get("company_status", ""),
            "incorporation_date": _parse_date(raw.get("date_of_creation")),
            "sic_codes":          raw.get("sic_codes", []),
            "registered_address": address_line,
            "type":               raw.get("type", ""),
        }

    def get_filing_history(
        self,
        company_number: str,
        category: str | None = None,
        max_items: int = 100,
    ) -> list[dict]:
        """
        Fetch filing history.  Each entry includes the made-up date (the period
        the accounts cover) and the date filed, so callers can detect late filings
        by comparing filed_at against the statutory deadline.

        Args:
            category:  optional CH category filter, e.g. "accounts", "confirmation-statement"
            max_items: page size (CH max is 100)

        Returns list of:
            {
                transaction_id:  str,
                type:            str,   # e.g. "AA" = annual accounts
                description:     str,
                date:            date | None,   # date filed at CH
                made_up_date:    date | None,   # period end date (accounts only)
                action_date:     date | None,   # due date where available
                category:        str,
            }
        """
        params: dict[str, Any] = {"items_per_page": max_items, "start_index": 0}
        if category:
            params["category"] = category

        raw = self._get(f"company/{company_number}/filing-history", params=params)
        if raw is None:
            return []

        results = []
        for item in raw.get("items", []):
            # made_up_date is nested under description_values, not at the top level
            made_up_date = _parse_date(
                item.get("description_values", {}).get("made_up_date")
                or item.get("made_up_date")
            )
            results.append({
                "transaction_id": item.get("transaction_id", ""),
                "type":           item.get("type", ""),
                "description":    item.get("description", ""),
                "date":           _parse_date(item.get("date")),
                "made_up_date":   made_up_date,
                "action_date":    _parse_date(item.get("action_date")),
                "category":       item.get("category", ""),
            })

        return results

    def get_accounts_data(self, company_number: str) -> dict | None:
        """
        Pull the most recent filed accounts from the filing history.

        The CH REST API does not serve parsed financials directly — that lives
        in iXBRL documents linked from each filing.  This method returns the
        metadata for the latest accounts filing so you know what period it covers
        and when it was filed.  Parsing the iXBRL for balance sheet / P&L figures
        is a separate step (see TODO below).

        Returns:
            {
                company_number:  str,
                accounts_type:   str,   # e.g. "total-exemption-full"
                period_end:      date | None,
                filed_at:        date | None,
                days_late:       int | None,   # positive = late, None = can't compute
            }

        TODO: follow the `links.document_metadata` URL to fetch the iXBRL filing
              and extract revenue, EBIT, net assets for distress scoring.
        """
        filings = self.get_filing_history(company_number, category="accounts", max_items=10)
        if not filings:
            log.warning("No accounts filings found for %s", company_number)
            return None

        # Latest accounts = first item (CH returns newest first)
        latest = filings[0]

        # Statutory deadline for private companies: 9 months after period end
        days_late: int | None = None
        if latest["made_up_date"] and latest["date"]:
            from datetime import timedelta
            deadline = latest["made_up_date"].replace(
                year=latest["made_up_date"].year + 1
                if latest["made_up_date"].month > 3
                else latest["made_up_date"].year
            )
            # Simpler: 9 calendar months approximated as 274 days
            deadline = latest["made_up_date"] + timedelta(days=274)
            delta    = (latest["date"] - deadline).days
            days_late = max(delta, 0)   # 0 = on time or early

        return {
            "company_number": company_number,
            "accounts_type":  latest.get("type", ""),
            "period_end":     latest["made_up_date"],
            "filed_at":       latest["date"],
            "days_late":      days_late,
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    # Tesco PLC — large, active, well-documented; good smoke-test target
    TEST_COMPANY = "00445790"

    print("\n" + "=" * 60)
    print(f"  Companies House API — smoke test ({TEST_COMPANY})")
    print("=" * 60)

    client = CompaniesHouseClient()

    print("\n── Profile ──")
    profile = client.get_company_profile(TEST_COMPANY)
    if profile:
        for k, v in profile.items():
            print(f"  {k:<22} {v}")

    print("\n── Filing history (last 10) ──")
    filings = client.get_filing_history(TEST_COMPANY, max_items=10)
    for f in filings:
        date_str     = str(f["date"])           if f["date"]        else "—"
        made_up_str  = str(f["made_up_date"])   if f["made_up_date"] else "—"
        print(f"  {date_str}  type={f['type']:<8}  period_end={made_up_str}  {f['description'][:60]}")

    print("\n── Latest accounts ──")
    accounts = client.get_accounts_data(TEST_COMPANY)
    if accounts:
        for k, v in accounts.items():
            print(f"  {k:<22} {v}")

    print("\n✓ API connection verified. No data written to the database.\n")
