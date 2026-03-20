#!/usr/bin/env python3
"""
Seed Garth Session to PostgreSQL
=================================
Run this ONCE from your LOCAL machine (where you have a valid ~/.garth session)
to bootstrap BOTH garth tokens (OAuth1 + OAuth2) into the production DB.

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

# -- Step 1: Resume local garth session --
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

# -- Step 2: Check which token files exist --
TOKEN_FILES = [
    ("garth_oauth2_token", "oauth2_token.json"),
    ("garth_oauth1_token", "oauth1_token.json"),
]

for db_key, filename in TOKEN_FILES:
    path = Path(GARTH_HOME) / filename
    if path.exists():
        print(f"Found: {path} ({path.stat().st_size} bytes)")
    else:
        print(f"WARNING: {path} not found - will skip this token.")

# -- Step 3: Ensure app_config table exists --
print("Connecting to PostgreSQL...")
try:
    conn = psycopg2.connect(CONN_STR, sslmode="require")
except Exception:
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

# -- Step 4: Upsert both tokens --
seeded = 0
for db_key, filename in TOKEN_FILES:
    token_path = Path(GARTH_HOME) / filename
    if not token_path.exists():
        print(f"SKIP: {filename} not found.")
        continue
    token_data = token_path.read_text()
    cur.execute("""
        INSERT INTO app_config (key, value, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (key) DO UPDATE
            SET value      = EXCLUDED.value,
                updated_at = NOW()
    """, (db_key, token_data))
    print(f"OK: Seeded {db_key} ({len(token_data)} bytes)")
    seeded += 1

# Verify
cur.execute("SELECT key, updated_at FROM app_config WHERE key LIKE 'garth_%' ORDER BY key")
rows = cur.fetchall()
print("\nCurrent garth tokens in DB:")
for row in rows:
    print(f"  {row[0]:35s}  updated_at={row[1]}")

cur.close()
conn.close()

if seeded == 0:
    print("\nERROR: No token files were found to seed. Check your GARTH_HOME.")
    sys.exit(1)

print(f"\nSUCCESS: {seeded} garth token(s) seeded to DB.")
print("The GitHub Actions daily sync should now resume without re-logging in.")
print("Both tokens (OAuth1 + OAuth2) are needed for a full year-long session.")
