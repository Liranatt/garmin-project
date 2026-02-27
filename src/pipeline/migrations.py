"""Startup migration and audit helpers for pipeline reliability."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import psycopg2
from dotenv import load_dotenv

from enhanced_schema import upgrade_database

log = logging.getLogger("pipeline.migrations")


def _resolve_conn_str(conn_str: str | None) -> str:
    load_dotenv()
    return (conn_str or os.getenv("POSTGRES_CONNECTION_STRING") or os.getenv("DATABASE_URL") or "").strip()


def ensure_startup_schema(conn_str: str | None = None) -> None:
    """Run idempotent startup migrations before pipeline execution."""
    cs = _resolve_conn_str(conn_str)
    if not cs:
        raise RuntimeError("POSTGRES_CONNECTION_STRING (or DATABASE_URL) is not configured")

    upgrade_database(cs)

    conn = psycopg2.connect(cs)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE IF EXISTS weekly_summaries
                ADD COLUMN IF NOT EXISTS analysis_status TEXT DEFAULT 'unknown'
                """
            )
            cur.execute(
                """
                ALTER TABLE IF EXISTS weekly_summaries
                ADD COLUMN IF NOT EXISTS pipeline_status_json JSONB
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_agent_rec_week
                ON agent_recommendations(week_date DESC)
                """
            )
    finally:
        conn.close()

    log.info("Startup migrations completed.")


def schema_audit(conn_str: str | None = None) -> Dict[str, Any]:
    """Return table/column/index audit data for runtime inspection."""
    cs = _resolve_conn_str(conn_str)
    if not cs:
        return {
            "ok": False,
            "error": "POSTGRES_CONNECTION_STRING (or DATABASE_URL) is not configured",
            "tables": {},
            "missing_tables": [],
        }

    required_tables: List[str] = [
        "daily_metrics",
        "activities",
        "weekly_summaries",
        "agent_recommendations",
        "matrix_summaries",
        "correlation_results",
    ]

    required_columns = {
        "weekly_summaries": ["week_start_date", "analysis_status", "pipeline_status_json"],
        "agent_recommendations": ["week_date", "recommendation", "agent_name"],
    }

    out: Dict[str, Any] = {"ok": True, "tables": {}, "missing_tables": []}
    conn = psycopg2.connect(cs)
    try:
        with conn.cursor() as cur:
            for table in required_tables:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public' AND table_name = %s
                    )
                    """,
                    (table,),
                )
                exists = bool(cur.fetchone()[0])
                table_info: Dict[str, Any] = {"exists": exists, "columns": [], "missing_columns": []}
                if not exists:
                    out["missing_tables"].append(table)
                    out["tables"][table] = table_info
                    continue

                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position
                    """,
                    (table,),
                )
                cols = [r[0] for r in cur.fetchall()]
                table_info["columns"] = cols
                expected = required_columns.get(table, [])
                table_info["missing_columns"] = [c for c in expected if c not in cols]
                out["tables"][table] = table_info

        out["ok"] = not out["missing_tables"] and not any(
            info.get("missing_columns") for info in out["tables"].values()
        )
        return out
    finally:
        conn.close()

