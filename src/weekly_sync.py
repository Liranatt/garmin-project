"""
Garmin Weekly Sync — Automated Health Data Pipeline
=====================================================
Standalone orchestrator.  Run once a week to:
  1. Fetch last 7 days from Garmin → PostgreSQL (upserts)
  2. Classify training intensity / body-part split
  3. Compute correlation matrices + agent summary
  4. Run CrewAI agents with pre-computed context
  5. Store AI insights

Usage:
    python weekly_sync.py            # Full pipeline
    python weekly_sync.py --fetch    # Fetch only (skip AI)
    python weekly_sync.py --analyze  # Analyze only (skip fetch)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from datetime import date, timedelta

import psycopg2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("weekly_sync")

# Local modules (all standalone — no garmin_health_tracker)
from enhanced_fetcher import EnhancedGarminDataFetcher
from correlation_engine import CorrelationEngine
from enhanced_agents import AdvancedHealthAgents
from email_notifier import send_weekly_email


class WeeklySyncPipeline:
    """Automated weekly data sync and analysis — fully standalone."""

    def __init__(self, fetch_days: int = 7):
        self.conn_str = os.getenv("POSTGRES_CONNECTION_STRING", "")
        self.fetch_days = fetch_days

    # ─── Main entry ───────────────────────────────────────────

    def run(self, skip_fetch: bool = False, skip_ai: bool = False) -> bool:
        """Execute the full weekly pipeline."""
        log.info("="*60)
        log.info("  WEEKLY GARMIN SYNC STARTED")
        log.info("  Date: %s", date.today())
        log.info("="*60)

        try:
            counts = {}
            training = {}
            corr_result = {}

            # Step 1: Fetch
            if not skip_fetch:
                log.info("Step 1/4: Fetching last 7 days from Garmin…")
                counts, training = self._fetch_data()
            else:
                log.info("Step 1/4: SKIPPED (--analyze mode)")

            # Step 2: Correlation matrices
            log.info("Step 2/4: Computing correlation matrices…")
            corr_result = self._compute_correlations(training or None)

            if not skip_ai:
                # Step 3: AI analysis
                log.info("Step 3/4: Running AI analysis…")
                insights = self._analyze_week(corr_result)

                # Step 4: Save insights
                log.info("Step 4/4: Saving insights…")
                self._save_insights(insights)

                # Step 5: Email notification
                log.info("Step 5: Sending email notification…")
                self._send_email(insights)
            else:
                insights = "(AI analysis skipped)"
                log.info("Step 3-4/4: SKIPPED (--fetch mode)")

            log.info("="*60)
            log.info("  WEEKLY SYNC COMPLETE")
            log.info("="*60)

            self._print_summary(counts, insights)
            return True

        except Exception as e:
            log.error("Pipeline failed: %s", e)
            traceback.print_exc()
            return False

    # ─── Step 1: Fetch ────────────────────────────────────────

    def _fetch_data(self):
        """Fetch Garmin data and write to PostgreSQL."""
        fetcher = EnhancedGarminDataFetcher()
        if not fetcher.authenticate():
            raise RuntimeError("Garmin authentication failed")
        counts = fetcher.fetch_and_store(days=self.fetch_days)
        training = fetcher.classify_training_intensity()

        log.info("Fetched: %d day-rows, %d activities, %d BB events",
                 counts.get('daily_metrics', 0),
                 counts.get('activities', 0),
                 counts.get('bb_events', 0))

        # Print training classification
        if training:
            for d in sorted(training.keys()):
                info = training[d]
                log.info("%s: %s (%d min hard)", d, info['intensity'],
                         info['hard_minutes'])

        return counts, training

    # ─── Step 2: Correlations ─────────────────────────────────

    def _compute_correlations(self, training_days=None) -> dict:
        """Run the correlation engine for all benchmark periods."""
        try:
            engine = CorrelationEngine(self.conn_str)
            result = engine.compute_benchmarks(training_days=training_days)
            return result
        except Exception as e:
            log.warning("Correlation engine error: %s", e)
            traceback.print_exc()
            return {"benchmarks": {}, "available": [], "longest": None,
                    "comparison": "", "data_days": 0}

    # ─── Step 3: AI analysis ──────────────────────────────────

    def _analyze_week(self, corr_result: dict = None) -> str:
        """Run CrewAI agents with pre-computed matrix context."""
        if corr_result is None:
            corr_result = {}
        agents = AdvancedHealthAgents()

        # Use the longest benchmark as the primary matrix context
        benchmarks = corr_result.get("benchmarks", {})
        longest = corr_result.get("longest")
        matrix_ctx = benchmarks.get(longest, "") if longest else ""

        # run_weekly_summary handles:
        #   - All 5 agents in sequence
        #   - Past recommendation context injection
        #   - Parsing + persisting new recommendations
        result = agents.run_weekly_summary(
            matrix_context=matrix_ctx,
            comparison_context=corr_result.get("comparison", ""),
        )
        return str(result)

    # ─── Step 4: Save insights ────────────────────────────────

    def _save_insights(self, insights: str):
        """Persist AI insights to weekly_summaries table."""
        week_start = date.today() - timedelta(days=date.today().weekday())
        concise = self._build_concise_summary(insights)

        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = True
        cur = conn.cursor()

        # Ensure the table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS weekly_summaries (
                summary_id       SERIAL PRIMARY KEY,
                week_start_date  DATE NOT NULL UNIQUE,
                avg_resting_hr   NUMERIC(5,1),
                avg_hrv          NUMERIC(5,1),
                avg_sleep_score  NUMERIC(4,1),
                avg_stress       NUMERIC(4,1),
                avg_readiness    NUMERIC(4,1),
                total_training_load INTEGER,
                total_steps      INTEGER,
                total_activities INTEGER,
                key_insights     TEXT,
                recommendations  TEXT,
                vs_previous_week JSONB,
                created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Compute weekly stats using REAL column names
        cur.execute("""
            SELECT
                AVG(resting_hr)          AS avg_rhr,
                AVG(hrv_last_night)      AS avg_hrv,
                AVG(sleep_score)         AS avg_sleep,
                AVG(stress_level)        AS avg_stress,
                AVG(training_readiness)  AS avg_readiness,
                SUM(bb_drained)          AS total_load,
                SUM(total_steps)         AS total_steps
            FROM daily_metrics
            WHERE date >= %s
        """, (week_start,))
        stats = cur.fetchone()

        # Count activities this week
        cur.execute(
            "SELECT COUNT(*) FROM activities WHERE date >= %s",
            (week_start,),
        )
        activity_count = cur.fetchone()[0]

        cur.execute("""
            INSERT INTO weekly_summaries
                (week_start_date, avg_resting_hr, avg_hrv, avg_sleep_score,
                 avg_stress, avg_readiness, total_training_load, total_steps,
                 total_activities, key_insights, recommendations)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (week_start_date) DO UPDATE SET
                avg_resting_hr    = EXCLUDED.avg_resting_hr,
                avg_hrv           = EXCLUDED.avg_hrv,
                avg_sleep_score   = EXCLUDED.avg_sleep_score,
                avg_stress        = EXCLUDED.avg_stress,
                avg_readiness     = EXCLUDED.avg_readiness,
                total_training_load = EXCLUDED.total_training_load,
                total_steps       = EXCLUDED.total_steps,
                total_activities  = EXCLUDED.total_activities,
                key_insights      = EXCLUDED.key_insights,
                recommendations   = EXCLUDED.recommendations,
                created_at        = CURRENT_TIMESTAMP
        """, (
            week_start,
            stats[0], stats[1], stats[2], stats[3], stats[4],
            stats[5], stats[6], activity_count,
            insights, concise,
        ))

        cur.close()
        conn.close()
        log.info("Insights saved for week %s", week_start)

    def _build_concise_summary(self, insights: str) -> str:
        """Create a strict 3-bullet, human-friendly summary for UI cards."""
        text = str(insights or "").replace("\r", "").strip()
        if not text:
            return (
                "- What changed: Insufficient data in this run.\n"
                "- Why it matters: Without stable signal, training decisions should stay conservative.\n"
                "- Next 24-48h: Keep effort moderate and reassess after tomorrow's sync."
            )

        candidates = []
        for raw in text.splitlines():
            line = raw.strip().lstrip("-* ").strip()
            if not line:
                continue
            if line.startswith("|") and line.endswith("|"):
                continue
            if set(line) <= {"=", "-", ":"}:
                continue
            candidates.append(line)

        def clip(s: str, limit: int = 260) -> str:
            s = s.replace("\n", " ").strip()
            if len(s) <= limit:
                return s
            return s[: limit - 3].rstrip() + "..."

        def bullet(label: str, value: str) -> str:
            prefix = f"- {label}: "
            allowed = max(48, 280 - len(prefix))
            return prefix + clip(value, allowed)

        what_changed = ""
        why_it_matters = ""
        next_24_48h = ""

        for line in candidates:
            low = line.lower()
            if not what_changed and any(t in low for t in ("trend", "up", "down", "increase", "decrease", "improved", "declined", "drop", "rise", "stable")):
                what_changed = line
                continue
            if not why_it_matters and any(t in low for t in ("because", "means", "risk", "impact", "matters", "recovery", "fatigue", "stress", "readiness")):
                why_it_matters = line
                continue
            if not next_24_48h and any(t in low for t in ("recommend", "should", "next", "today", "tomorrow", "focus", "avoid", "keep", "do")):
                next_24_48h = line

        for line in candidates:
            if not what_changed:
                what_changed = line
                continue
            if not why_it_matters and line != what_changed:
                why_it_matters = line
                continue
            if not next_24_48h and line not in (what_changed, why_it_matters):
                next_24_48h = line

        what_changed = clip(what_changed or "Week-over-week signal changed, but data density is limited.")
        why_it_matters = clip(why_it_matters or "This affects recovery quality and your expected response to training load.")
        next_24_48h = clip(next_24_48h or "Keep intensity moderate, prioritize sleep timing, and reassess after the next daily run.")

        return (
            f"{bullet('What changed', what_changed)}\n"
            f"{bullet('Why it matters', why_it_matters)}\n"
            f"{bullet('Next 24-48h', next_24_48h)}"
        )

    # ─── Step 5: Email ────────────────────────────────────────

    def _send_email(self, insights: str):
        """Send weekly recommendations email."""
        try:
            week_start = date.today() - timedelta(days=date.today().weekday())
            send_weekly_email(insights, week_date=week_start)
        except Exception as e:
            log.warning("Email notification failed (non-fatal): %s", e)

    # ─── Summary ──────────────────────────────────────────────

    def _print_summary(self, counts, insights):
        """Print execution summary."""
        log.info("SYNC SUMMARY:")
        if counts:
            log.info("  Daily rows:  %d", counts.get('daily_metrics', 0))
            log.info("  Activities:  %d", counts.get('activities', 0))
            log.info("  BB events:   %d", counts.get('bb_events', 0))

        preview = str(insights)[:500]
        log.info("AI INSIGHTS (preview): %s…", preview)
        log.info("Full insights saved to database")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Garmin Weekly Sync Pipeline"
    )
    parser.add_argument("--fetch", action="store_true",
                        help="Fetch only — skip AI analysis")
    parser.add_argument("--analyze", action="store_true",
                        help="Analyze only — skip Garmin fetch")
    parser.add_argument("--days", type=int, default=7,
                        help="Days to fetch (default: 7)")
    parser.add_argument("--bulk-import", action="store_true",
                        help="Run bulk import automation (email + zip)")
    args = parser.parse_args()

    if args.bulk_import:
        # Import here to avoid circular dependencies if any
        try:
            from bulk_import import run_bulk_import
            log.info("Starting BULK IMPORT...")
            if not run_bulk_import(auto=True):
                log.error("Bulk import failed.")
                sys.exit(1)
            # If bulk import succeeds, we might want to continue with sync or stop.
            # Usually bulk import is heavy, so maybe stop? 
            # Or continue to analyze the new data.
            # Let's continue to analyze if --analyze is set, otherwise exit.
            if not args.analyze and not args.fetch:
                log.info("Bulk import finished. Exiting (no other flags set).")
                sys.exit(0)
        except ImportError:
            log.error("Could not import src.bulk_import. Check project structure.")
            sys.exit(1)

    pipeline = WeeklySyncPipeline(fetch_days=args.days)
    success = pipeline.run(
        skip_fetch=args.analyze,
        skip_ai=args.fetch,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
