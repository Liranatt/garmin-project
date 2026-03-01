"""Weekly pipeline orchestration with explicit health signaling."""

from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import psycopg2

from correlation_engine import CorrelationEngine
from email_notifier import send_weekly_email
from enhanced_agents import AdvancedHealthAgents
from enhanced_fetcher import EnhancedGarminDataFetcher
from pipeline.migrations import ensure_startup_schema
from pipeline.summary_builder import build_concise_summary

log = logging.getLogger("weekly_sync")


class WeeklySyncPipeline:
    """Automated weekly data sync and analysis."""

    def __init__(self, fetch_days: int = 7):
        self.conn_str = os.getenv("POSTGRES_CONNECTION_STRING", "")
        self.fetch_days = fetch_days

    def run(self, skip_fetch: bool = False, skip_ai: bool = False) -> bool:
        """Execute the full weekly pipeline and persist machine-readable status."""
        pipeline_status: Dict[str, Any] = {
            "run_date": date.today().isoformat(),
            "run_started_at": datetime.utcnow().isoformat() + "Z",
            "fetch_ok": bool(skip_fetch),
            "correlation_ok": False,
            "agents_ok": bool(skip_ai),
            "insights_ok": bool(skip_ai),
            "email_ok": False,
            "analysis_status": "unknown",
            "degraded_reasons": [],
        }

        log.info("=" * 60)
        log.info("  WEEKLY GARMIN SYNC STARTED")
        log.info("  Date: %s", date.today())
        log.info("=" * 60)

        counts: Dict[str, Any] = {}
        training: Dict[str, Any] = {}
        corr_result: Dict[str, Any] = {}
        insights: str = "(no insights generated)"

        try:
            log.info("Step 0/5: Running startup migrations...")
            ensure_startup_schema(self.conn_str)

            if not skip_fetch:
                log.info("Step 1/5: Fetching last %s days from Garmin...", self.fetch_days)
                counts, training = self._fetch_data()
                pipeline_status["fetch_ok"] = True
            else:
                log.info("Step 1/5: SKIPPED (--analyze mode)")

            log.info("Step 2/5: Computing correlation matrices...")
            corr_result = self._compute_correlations(training or None)
            pipeline_status["analysis_status"] = corr_result.get("analysis_status", "unknown")
            pipeline_status["degraded_reasons"] = corr_result.get("degraded_reasons", [])
            pipeline_status["correlation_ok"] = pipeline_status["analysis_status"] == "success"

            if pipeline_status["analysis_status"] == "failed":
                log.error("Correlation analysis failed. Continuing only to persist degraded status.")
            elif pipeline_status["analysis_status"] == "degraded":
                log.warning(
                    "Correlation analysis degraded: %s",
                    ", ".join(pipeline_status["degraded_reasons"]) or "no details",
                )

            if not skip_ai:
                log.info("Step 3/5: Running AI analysis...")
                insights = self._analyze_week(corr_result)
                pipeline_status["agents_ok"] = True

                log.info("Step 4/5: Saving insights...")
                self._save_insights(
                    insights=insights,
                    analysis_status=pipeline_status["analysis_status"],
                    pipeline_status=pipeline_status,
                )
                pipeline_status["insights_ok"] = True

                # Email disabled — was sending weekly insights on every daily run.
                # TODO: re-enable when moved to a true weekly-only schedule.
                log.info("Step 5/5: Email notification SKIPPED (disabled)")
                pipeline_status["email_ok"] = True
            else:
                insights = "(AI analysis skipped)"
                log.info("Step 3-5/5: SKIPPED (--fetch mode)")

        except Exception as e:
            pipeline_status["analysis_status"] = "failed"
            pipeline_status["degraded_reasons"] = ["pipeline_exception"]
            log.error("Pipeline failed: %s", e)
            traceback.print_exc()
        finally:
            pipeline_status["run_finished_at"] = datetime.utcnow().isoformat() + "Z"
            pipeline_status["overall_status"] = self._overall_status(pipeline_status)
            self._write_pipeline_status_file(pipeline_status)
            self._print_summary(counts, insights, pipeline_status)
            log.info("=" * 60)
            log.info("  WEEKLY SYNC COMPLETE (status=%s)", pipeline_status["overall_status"])
            log.info("=" * 60)

        strict_health = os.getenv("STRICT_PIPELINE_HEALTH", "0").strip() == "1"
        if strict_health:
            return pipeline_status["overall_status"] == "success"
        return pipeline_status["overall_status"] != "failed"

    def _fetch_data(self):
        """Fetch Garmin data and write to PostgreSQL."""
        fetcher = EnhancedGarminDataFetcher()
        if not fetcher.authenticate():
            raise RuntimeError("Garmin authentication failed")
        counts = fetcher.fetch_and_store(days=self.fetch_days)
        training = fetcher.classify_training_intensity()

        log.info(
            "Fetched: %d day-rows, %d activities, %d BB events",
            counts.get("daily_metrics", 0),
            counts.get("activities", 0),
            counts.get("bb_events", 0),
        )

        if training:
            for d in sorted(training.keys()):
                info = training[d]
                log.info("%s: %s (%d min hard)", d, info["intensity"], info["hard_minutes"])

        return counts, training

    def _compute_correlations(self, training_days=None) -> Dict[str, Any]:
        """Run the correlation engine for all benchmark periods."""
        try:
            engine = CorrelationEngine(self.conn_str)
            result = engine.compute_benchmarks(training_days=training_days)
            if "analysis_status" not in result:
                result["analysis_status"] = "success"
            if "degraded_reasons" not in result:
                result["degraded_reasons"] = []
            return result
        except Exception as e:
            log.warning("Correlation engine error: %s", e)
            traceback.print_exc()
            return {
                "benchmarks": {},
                "available": [],
                "longest": None,
                "comparison": "",
                "data_days": 0,
                "analysis_status": "failed",
                "degraded_reasons": ["correlation_engine_exception"],
            }

    def _analyze_week(self, corr_result: Optional[Dict[str, Any]] = None) -> str:
        """Run CrewAI agents with pre-computed matrix context."""
        corr_result = corr_result or {}
        agents = AdvancedHealthAgents()

        benchmarks = corr_result.get("benchmarks", {})
        longest = corr_result.get("longest")
        matrix_ctx = benchmarks.get(longest, "") if longest else ""

        result = agents.run_weekly_summary(
            matrix_context=matrix_ctx,
            comparison_context=corr_result.get("comparison", ""),
        )
        return str(result)

    def _save_insights(self, insights: str, analysis_status: str, pipeline_status: Dict[str, Any]):
        """Persist AI insights to weekly_summaries with status metadata."""
        week_start = date.today() - timedelta(days=date.today().weekday())
        concise = build_concise_summary(insights)

        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = True
        cur = conn.cursor()

        # Schema is managed by migrations.py ensure_startup_schema()

        cur.execute(
            """
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
            """,
            (week_start,),
        )
        stats = cur.fetchone()

        cur.execute("SELECT COUNT(*) FROM activities WHERE date >= %s", (week_start,))
        activity_count = cur.fetchone()[0]

        pipeline_status_json = json.dumps(pipeline_status, ensure_ascii=False)
        cur.execute(
            """
            INSERT INTO weekly_summaries
                (week_start_date, avg_resting_hr, avg_hrv, avg_sleep_score,
                 avg_stress, avg_readiness, total_training_load, total_steps,
                 total_activities, key_insights, recommendations, analysis_status,
                 pipeline_status_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (week_start_date) DO UPDATE SET
                avg_resting_hr      = EXCLUDED.avg_resting_hr,
                avg_hrv             = EXCLUDED.avg_hrv,
                avg_sleep_score     = EXCLUDED.avg_sleep_score,
                avg_stress          = EXCLUDED.avg_stress,
                avg_readiness       = EXCLUDED.avg_readiness,
                total_training_load = EXCLUDED.total_training_load,
                total_steps         = EXCLUDED.total_steps,
                total_activities    = EXCLUDED.total_activities,
                key_insights        = EXCLUDED.key_insights,
                recommendations     = EXCLUDED.recommendations,
                analysis_status     = EXCLUDED.analysis_status,
                pipeline_status_json = EXCLUDED.pipeline_status_json,
                created_at          = CURRENT_TIMESTAMP
            """,
            (
                week_start,
                stats[0],
                stats[1],
                stats[2],
                stats[3],
                stats[4],
                stats[5],
                stats[6],
                activity_count,
                insights,
                concise,
                analysis_status,
                pipeline_status_json,
            ),
        )

        cur.close()
        conn.close()
        log.info("Insights saved for week %s with analysis_status=%s", week_start, analysis_status)

    def _send_email(self, insights: str) -> bool:
        """Send weekly recommendations email."""
        try:
            week_start = date.today() - timedelta(days=date.today().weekday())
            send_weekly_email(insights, week_date=week_start)
            return True
        except Exception as e:
            log.warning("Email notification failed (non-fatal): %s", e)
            return False

    @staticmethod
    def _overall_status(status: Dict[str, Any]) -> str:
        if not status.get("fetch_ok", False):
            return "failed"
        if status.get("analysis_status") == "failed":
            return "failed"
        if not status.get("agents_ok", False) or not status.get("insights_ok", False):
            return "failed"
        if status.get("analysis_status") == "degraded":
            return "degraded"
        return "success"

    @staticmethod
    def _write_pipeline_status_file(status: Dict[str, Any]) -> None:
        today = date.today().isoformat()
        default_path = f"pipeline_status_{today}.json"
        path = os.getenv("PIPELINE_STATUS_PATH", default_path)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(status, fh, indent=2, ensure_ascii=False)
            log.info("Pipeline status written to %s", path)
        except Exception as e:
            log.warning("Failed to write pipeline status file: %s", e)

    def _print_summary(self, counts: Dict[str, Any], insights: str, status: Dict[str, Any]):
        """Print execution summary."""
        log.info("SYNC SUMMARY:")
        if counts:
            log.info("  Daily rows:  %d", counts.get("daily_metrics", 0))
            log.info("  Activities:  %d", counts.get("activities", 0))
            log.info("  BB events:   %d", counts.get("bb_events", 0))

        log.info("  Correlation status: %s", status.get("analysis_status"))
        reasons = status.get("degraded_reasons") or []
        if reasons:
            log.info("  Degraded reasons: %s", ", ".join(reasons))
        log.info("  Overall status: %s", status.get("overall_status"))

        preview = str(insights)[:500]
        log.info("AI INSIGHTS (preview): %s...", preview)
        log.info("Full insights saved to database")

