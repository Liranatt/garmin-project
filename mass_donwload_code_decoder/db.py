"""Database connection helper."""

import psycopg2
from config import PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASS


def get_connection():
    """Return a new psycopg2 connection."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS,
    )


def init_schema(schema_path: str = "schema.sql"):
    """Execute the DDL script to create / recreate all tables."""
    conn = get_connection()
    cur = conn.cursor()
    with open(schema_path, encoding="utf-8") as f:
        cur.execute(f.read())
    conn.commit()
    cur.close()
    conn.close()
    print("[✓] Schema initialised.")


if __name__ == "__main__":
    init_schema()
