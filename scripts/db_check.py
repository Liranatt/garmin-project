"""
One-off DB check script for Heroku.
Prints Postgres version, whether daily_metrics exists, row count, and latest row.
Safe: does not echo connection strings.
"""
import os
import sys
try:
    import psycopg2
except Exception as e:
    print('ERROR: psycopg2 not installed:', e)
    sys.exit(2)

cs = os.getenv('POSTGRES_CONNECTION_STRING') or os.getenv('DATABASE_URL')
if not cs:
    print('NO_CONN_STRING')
    sys.exit(2)

try:
    conn = psycopg2.connect(cs)
    cur = conn.cursor()
    cur.execute('SELECT version()')
    print('pg_version:', cur.fetchone()[0])
    cur.execute("SELECT to_regclass('public.daily_metrics')")
    exists = cur.fetchone()[0]
    print('daily_metrics_exists:', bool(exists))
    if exists:
        cur.execute('SELECT COUNT(*) FROM public.daily_metrics')
        print('daily_metrics_count:', cur.fetchone()[0])
        cur.execute("SELECT date, resting_hr FROM public.daily_metrics ORDER BY date DESC LIMIT 1")
        print('latest_row:', cur.fetchone())
    cur.close()
    conn.close()
    print('OK')
except Exception as e:
    print('ERROR:', repr(e))
    sys.exit(1)
