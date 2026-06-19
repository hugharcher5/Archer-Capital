"""
PostgreSQL storage layer for Companies House data.

Requirements: psycopg2-binary, python-dotenv

Tables (auto-created on first run):
    ch_company_profiles   — one row per company number
    ch_filings            — one row per filing transaction
    ch_accounts           — one row per accounts period

All writes use upsert (INSERT ... ON CONFLICT DO UPDATE) so re-running
the pipeline never creates duplicates.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

log = logging.getLogger(__name__)

# ── Connection ────────────────────────────────────────────────────────────────

def _connection_kwargs() -> dict:
    return {
        "host":     os.getenv("DB_HOST",     "localhost"),
        "port":     int(os.getenv("DB_PORT", "5432")),
        "dbname":   os.getenv("DB_NAME",     "archer_capital"),
        "user":     os.getenv("DB_USER",     "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
    }


@contextmanager
def _conn() -> Generator[psycopg2.extensions.connection, None, None]:
    con = psycopg2.connect(**_connection_kwargs())
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema bootstrap ──────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS ch_company_profiles (
    company_number      TEXT PRIMARY KEY,
    name                TEXT,
    status              TEXT,
    incorporation_date  DATE,
    sic_codes           TEXT[],
    registered_address  TEXT,
    company_type        TEXT,
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ch_filings (
    transaction_id  TEXT PRIMARY KEY,
    company_number  TEXT NOT NULL,
    filing_type     TEXT,
    description     TEXT,
    date_filed      DATE,
    made_up_date    DATE,
    action_date     DATE,
    category        TEXT,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ch_accounts (
    company_number  TEXT,
    period_end      DATE,
    accounts_type   TEXT,
    filed_at        DATE,
    days_late       INTEGER,
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (company_number, period_end)
);
"""


def ensure_schema() -> None:
    """Create tables if they don't exist. Safe to call on every run."""
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(_DDL)
    log.info("Schema verified.")


# ── Upsert helpers ────────────────────────────────────────────────────────────

def upsert_company_profile(profile: dict) -> None:
    """
    Insert or update a company profile row.

    Expects the dict returned by CompaniesHouseClient.get_company_profile().
    """
    sql = """
        INSERT INTO ch_company_profiles
            (company_number, name, status, incorporation_date,
             sic_codes, registered_address, company_type, updated_at)
        VALUES
            (%(company_number)s, %(name)s, %(status)s, %(incorporation_date)s,
             %(sic_codes)s, %(registered_address)s, %(type)s, NOW())
        ON CONFLICT (company_number) DO UPDATE SET
            name               = EXCLUDED.name,
            status             = EXCLUDED.status,
            incorporation_date = EXCLUDED.incorporation_date,
            sic_codes          = EXCLUDED.sic_codes,
            registered_address = EXCLUDED.registered_address,
            company_type       = EXCLUDED.company_type,
            updated_at         = NOW()
    """
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(sql, profile)
    log.info("Upserted profile: %s", profile.get("company_number"))


def upsert_filings(filings: list[dict]) -> None:
    """
    Bulk-upsert a list of filing records.

    Expects the list returned by CompaniesHouseClient.get_filing_history().
    company_number must be added to each dict by the caller before passing in.
    """
    if not filings:
        return

    sql = """
        INSERT INTO ch_filings
            (transaction_id, company_number, filing_type, description,
             date_filed, made_up_date, action_date, category, updated_at)
        VALUES
            (%(transaction_id)s, %(company_number)s, %(type)s, %(description)s,
             %(date)s, %(made_up_date)s, %(action_date)s, %(category)s, NOW())
        ON CONFLICT (transaction_id) DO UPDATE SET
            filing_type  = EXCLUDED.filing_type,
            description  = EXCLUDED.description,
            date_filed   = EXCLUDED.date_filed,
            made_up_date = EXCLUDED.made_up_date,
            action_date  = EXCLUDED.action_date,
            category     = EXCLUDED.category,
            updated_at   = NOW()
    """
    with _conn() as con:
        with con.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, filings)
    log.info("Upserted %d filings.", len(filings))


def upsert_accounts(accounts: dict) -> None:
    """
    Insert or update an accounts record.

    Expects the dict returned by CompaniesHouseClient.get_accounts_data().
    """
    sql = """
        INSERT INTO ch_accounts
            (company_number, period_end, accounts_type, filed_at, days_late, updated_at)
        VALUES
            (%(company_number)s, %(period_end)s, %(accounts_type)s,
             %(filed_at)s, %(days_late)s, NOW())
        ON CONFLICT (company_number, period_end) DO UPDATE SET
            accounts_type = EXCLUDED.accounts_type,
            filed_at      = EXCLUDED.filed_at,
            days_late     = EXCLUDED.days_late,
            updated_at    = NOW()
    """
    with _conn() as con:
        with con.cursor() as cur:
            cur.execute(sql, accounts)
    log.info("Upserted accounts: %s / period_end=%s",
             accounts.get("company_number"), accounts.get("period_end"))
