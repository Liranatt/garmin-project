"""Configuration loaded from .env"""

import os
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# PostgreSQL
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
PG_DB = os.getenv("POSTGRES_DB", "garmin")
PG_USER = os.getenv("POSTGRES_USER", "postgres")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "")

# Data paths
DATA_DIR = Path(os.getenv("GARMIN_DATA_DIR", "."))
CONNECT_DIR = DATA_DIR / "DI_CONNECT"

# Date filter
DATE_FROM = date.fromisoformat(os.getenv("DATE_FROM", "2026-02-01"))
