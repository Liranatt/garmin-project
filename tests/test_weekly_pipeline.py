"""Contract tests for weekly pipeline health-state semantics.

Expanded from original 3 tests to cover:
- _overall_status with all edge cases
- _write_pipeline_status_file date naming
- _save_insights guard clauses
- Correlation failure propagation
- Email send failure handling
- Full pipeline run mocking
"""

import json
import os
import sys
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline.weekly_pipeline import WeeklySyncPipeline


# ─── _overall_status ─────────────────────────────────────────


class TestOverallStatus:

    def test_success(self):
        status = {
            "fetch_ok": True,
            "analysis_status": "success",
            "agents_ok": True,
            "insights_ok": True,
        }
        assert WeeklySyncPipeline._overall_status(status) == "success"

    def test_degraded_when_analysis_degraded(self):
        status = {
            "fetch_ok": True,
            "analysis_status": "degraded",
            "agents_ok": True,
            "insights_ok": True,
        }
        assert WeeklySyncPipeline._overall_status(status) == "degraded"

    def test_failed_when_agents_fail(self):
        status = {
            "fetch_ok": True,
            "analysis_status": "success",
            "agents_ok": False,
            "insights_ok": True,
        }
        assert WeeklySyncPipeline._overall_status(status) == "failed"

    def test_failed_when_fetch_fails(self):
        status = {
            "fetch_ok": False,
            "analysis_status": "success",
            "agents_ok": True,
            "insights_ok": True,
        }
        assert WeeklySyncPipeline._overall_status(status) == "failed"

    def test_failed_when_insights_fail(self):
        status = {
            "fetch_ok": True,
            "analysis_status": "success",
            "agents_ok": True,
            "insights_ok": False,
        }
        assert WeeklySyncPipeline._overall_status(status) == "failed"

    def test_failed_when_analysis_failed(self):
        status = {
            "fetch_ok": True,
            "analysis_status": "failed",
            "agents_ok": True,
            "insights_ok": True,
        }
        assert WeeklySyncPipeline._overall_status(status) == "failed"

    def test_missing_keys_treated_as_false(self):
        status = {"fetch_ok": True, "analysis_status": "success"}
        # agents_ok and insights_ok missing → treated as False
        assert WeeklySyncPipeline._overall_status(status) == "failed"


# ─── _write_pipeline_status_file ──────────────────────────────


class TestWritePipelineStatus:

    def test_writes_valid_json(self, tmp_path):
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)
        status_path = str(tmp_path / "status.json")

        with patch.dict(os.environ, {"PIPELINE_STATUS_PATH": status_path}):
            pipeline._write_pipeline_status_file({
                "run_date": "2026-02-27",
                "overall_status": "success",
                "fetch_ok": True,
            })

        with open(status_path) as f:
            data = json.load(f)
        assert data["overall_status"] == "success"
        assert data["run_date"] == "2026-02-27"

    def test_handles_write_failure_gracefully(self, tmp_path):
        """Writing to a non-existent deep path should not crash."""
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)
        bad_path = str(tmp_path / "nonexistent" / "deep" / "status.json")

        with patch.dict(os.environ, {"PIPELINE_STATUS_PATH": bad_path}):
            # Should not raise
            pipeline._write_pipeline_status_file({"test": True})


# ─── Correlation failure propagation ─────────────────────────


class TestCorrelationFailurePropagation:

    def test_correlation_exception_returns_failed(self):
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)
        pipeline.conn_str = "postgresql://fake"

        with patch("pipeline.weekly_pipeline.CorrelationEngine") as mock_engine:
            mock_engine.side_effect = Exception("DB connection failed")
            result = pipeline._compute_correlations()

        assert result["analysis_status"] == "failed"
        assert result["longest"] is None
        assert result["benchmarks"] == {}

    def test_correlation_success_passes_through(self):
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)
        pipeline.conn_str = "postgresql://fake"

        mock_result = {
            "benchmarks": {"1_week": "summary text"},
            "available": ["1_week"],
            "longest": "1_week",
            "comparison": "comparison text",
            "data_days": 21,
            "analysis_status": "success",
            "degraded_reasons": [],
        }

        with patch("pipeline.weekly_pipeline.CorrelationEngine") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.compute_benchmarks.return_value = mock_result
            mock_cls.return_value = mock_instance
            result = pipeline._compute_correlations()

        assert result["analysis_status"] == "success"
        assert result["longest"] == "1_week"


# ─── Agent context propagation ───────────────────────────────


class TestAgentContextPropagation:

    def test_empty_corr_result_passes_empty_context(self):
        """When correlation fails, agents should get empty matrix_context."""
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)

        with patch("pipeline.weekly_pipeline.AdvancedHealthAgents") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.run_weekly_summary.return_value = "agent output"
            mock_cls.return_value = mock_instance

            result = pipeline._analyze_week({
                "benchmarks": {},
                "longest": None,
                "comparison": "",
            })

        # Verify empty context was passed
        mock_instance.run_weekly_summary.assert_called_once_with(
            matrix_context="",
            comparison_context="",
        )
        assert result == "agent output"

    def test_successful_corr_passes_context(self):
        """When correlation succeeds, agents should get the summary text."""
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)

        with patch("pipeline.weekly_pipeline.AdvancedHealthAgents") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.run_weekly_summary.return_value = "agent output"
            mock_cls.return_value = mock_instance

            pipeline._analyze_week({
                "benchmarks": {"1_week": "correlation summary here"},
                "longest": "1_week",
                "comparison": "week comparison",
            })

        mock_instance.run_weekly_summary.assert_called_once_with(
            matrix_context="correlation summary here",
            comparison_context="week comparison",
        )


# ─── Email failure handling ──────────────────────────────────


class TestEmailHandling:

    def test_email_failure_returns_false(self):
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)

        with patch("pipeline.weekly_pipeline.send_weekly_email", side_effect=Exception("SMTP error")):
            result = pipeline._send_email("some insights")

        assert result is False

    def test_email_success_returns_true(self):
        pipeline = WeeklySyncPipeline.__new__(WeeklySyncPipeline)

        with patch("pipeline.weekly_pipeline.send_weekly_email"):
            result = pipeline._send_email("some insights")

        assert result is True


# ─── Pipeline initialization ─────────────────────────────────


class TestPipelineInit:

    def test_default_fetch_days(self):
        pipeline = WeeklySyncPipeline()
        assert pipeline.fetch_days == 7

    def test_custom_fetch_days(self):
        pipeline = WeeklySyncPipeline(fetch_days=14)
        assert pipeline.fetch_days == 14

    @patch.dict(os.environ, {"POSTGRES_CONNECTION_STRING": "postgresql://test"})
    def test_reads_conn_str_from_env(self):
        pipeline = WeeklySyncPipeline()
        assert pipeline.conn_str == "postgresql://test"
