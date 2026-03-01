"""
Shared database utilities.
Single source of truth for PostgreSQL connection-string resolution.
"""

from __future__ import annotations

import os


def get_conn_str() -> str:
    """Return PostgreSQL connection string.

    Checks POSTGRES_CONNECTION_STRING first, falls back to DATABASE_URL
    (Heroku standard).  Normalises postgres:// to postgresql:// for psycopg2.
    """
    url = os.getenv("POSTGRES_CONNECTION_STRING") or os.getenv("DATABASE_URL") or ""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url
