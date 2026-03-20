#!/usr/bin/env python3
"""
Seed Garth Session to PostgreSQL
=================================
Run this ONCE from your LOCAL machine (where you have a valid ~/.garth session)
to bootstrap the garth OAuth2 token into the production DB.

After this runs successfully, the GitHub Actions pipeline will be able to
resume the session from the DB instead of re-logging in every run.

Usage:
    cd garmin-project
    pip install garth psycopg2-binary python-dotenv
    python scripts/seed_garth_session.py

Requires in your .env (or env vars):
    POSTGRES_CONNECTION_STRING=postgresql://...
    GARTH_HOME=~/.garth   (default)
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg2
import garth

load_dotenv()

GARTH_HOME = os.path.expanduser(os.getenv("GARTH_HOME", "~/.garth"))
CONN_STR = os.getenv("POSTGRES_CONNECTION_STRING", "")

if not CONN_STR:
    print("ERROR: POSTGRES_CONNECTION_STRING is not set. Check your .env file.")
    sys.exit(1)

# ── Step 1: Resume local garth session ───────────────────────────────────────
print(f"Attempting to resume garth session from: {GARTH_HOME}")
try:
    garth.resume(GARTH_HOME)
    username = garth.client.username
    print(f"OK: Resumed session for {username}")
except Exception as e:
    print(f"ERROR: Could not resume garth session from {GARTH_HOME}: {e}")
    print("Make sure you have logged in locally with garth first:")
    print("  python -c \"import garth; garth.login('your@email.com', 'password'); garth.save('~/.garth')\"")
    sys.exit(1)

# ── Step 2: Read the token file ───────────────────────────────────────────────
token_path = Path(GARTH_HOME) / "oauth2_token.json"
if not token_path.exists():
    print(f"ERROR: {token_path} does not exist. Garth session may be OAuth1-only.")
    # Try oauth1_token.json as well
    oauth1_path = Path(GARTH_HOME) / "oauth1_token.json"
    if not oauth1_path.exists():
        print("No oauth1_token.json either. Cannot proceed.")
        sys.exit(1)
    print("Found oauth1_token.json — garth uses both for resume.")

token_data = token_path.read_text()
print(f"Read {len(token_data)} bytes from {token_path}")

# ── Step 3: Ensure app_config table exists ────────────────────────────────────
print("Connecting to PostgreSQL...")
try:
    conn = psycopg2.connect(CONN_STR, sslmode="require")
except Exception:
    # Try without sslmode for local DBs
    conn = psycopg2.connect(CONN_STR)

conn.autocommit = True
cur = conn.cursor()

cur.execute("""
    CREATE TABLE IF NOT EXISTS app_config (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    )
""")
print("OK: app_config table ready.")

# ── Step 4: Upsert the token ──────────────────────────────────────────────────
cur.execute("""
    INSERT INTO app_config (key, value, updated_at)
    VALUES ('garth_oauth2_token', %s, NOW())
    ON CONFLICT (key) DO UPDATE
        SET value      = EXCLUDED.value,
            updated_at = NOW()
""", (token_data,))

# Verify
cur.execute("SELECT updated_at FROM app_config WHERE key = 'garth_oauth2_token'")
row = cur.fetchone()
print(f"OK: Token stored in DB (updated_at={row[0]})")

cur.close()
conn.close()

print()
print("SUCCESS: Garth session seeded to DB.")
print("The GitHub Actions daily sync should now be able to resume without re-logging in.")
print("The token will auto-refresh on each successful sync run.")
