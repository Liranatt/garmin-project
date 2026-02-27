"""
Garmin Weekly Sync - CLI entrypoint.

Run from src/:
    python weekly_sync.py            # Full pipeline
    python weekly_sync.py --fetch    # Fetch only (skip AI)
    python weekly_sync.py --analyze  # Analyze only (skip fetch)
"""

from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from pipeline.weekly_pipeline import WeeklySyncPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("weekly_sync")


def main() -> None:
    parser = argparse.ArgumentParser(description="Garmin Weekly Sync Pipeline")
    parser.add_argument("--fetch", action="store_true", help="Fetch only - skip AI analysis")
    parser.add_argument("--analyze", action="store_true", help="Analyze only - skip Garmin fetch")
    parser.add_argument("--days", type=int, default=7, help="Days to fetch (default: 7)")
    parser.add_argument("--bulk-import", action="store_true", help="Run bulk import automation (email + zip)")
    args = parser.parse_args()

    if args.bulk_import:
        try:
            from bulk_import import run_bulk_import

            log.info("Starting BULK IMPORT...")
            if not run_bulk_import(auto=True):
                log.error("Bulk import failed.")
                sys.exit(1)

            if not args.analyze and not args.fetch:
                log.info("Bulk import finished. Exiting (no other flags set).")
                sys.exit(0)
        except ImportError:
            log.error("Could not import src.bulk_import. Check project structure.")
            sys.exit(1)

    pipeline = WeeklySyncPipeline(fetch_days=args.days)
    success = pipeline.run(skip_fetch=args.analyze, skip_ai=args.fetch)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
