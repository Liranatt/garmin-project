"""
Tests for the CrewAI agent tools (SQL, correlation, pattern).

All DB calls are mocked — no real Postgres required.
"""
import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ─── Import the raw functions behind @tool decorators ─────────
# CrewAI's @tool wraps functions in StructuredTool objects.
# We import the module and grab the underlying _run methods.
import enhanced_agents as _agents_mod

# The @tool decorator stores the original function; calling .run() invokes it.
# We'll define thin wrappers that call the right interface.

def run_sql_query(query: str) -> str:
    return _agents_mod.run_sql_query.run(query)

def calculate_correlation(m1: str, m2: str, days: int = 30) -> str:
    return _agents_mod.calculate_correlation.run(
        metric1=m1, metric2=m2, days=days
    )

def analyze_pattern(metric: str, days: int = 30) -> str:
    return _agents_mod.analyze_pattern.run(metric=metric, days=days)


class TestRunSqlQuery:
    """Test the SQL query tool."""

    def test_blocks_insert(self):
        result = run_sql_query("INSERT INTO daily_metrics VALUES (1,2)")
        assert "INSERT" in result and "not allowed" in result

    def test_blocks_drop(self):
        result = run_sql_query("DROP TABLE daily_metrics")
        assert "DROP" in result and "not allowed" in result

    def test_blocks_delete(self):
        result = run_sql_query("DELETE FROM daily_metrics")
        assert "DELETE" in result and "not allowed" in result

    @patch("enhanced_agents.psycopg2")
    def test_returns_dataframe_string(self, mock_pg):
        """A valid SELECT should return formatted table text."""
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        df = pd.DataFrame({"date": ["2026-01-01"], "resting_hr": [55]})
        with patch("enhanced_agents.pd.read_sql_query", return_value=df):
            result = run_sql_query("SELECT * FROM daily_metrics LIMIT 1")
        assert "55" in result
        assert "resting_hr" in result

    @patch("enhanced_agents.psycopg2")
    def test_empty_result(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        with patch("enhanced_agents.pd.read_sql_query", return_value=pd.DataFrame()):
            result = run_sql_query("SELECT * FROM daily_metrics WHERE 1=0")
        assert "no results" in result.lower()


# ─── calculate_correlation ────────────────────────────────────


class TestCalculateCorrelation:
    """Test the correlation tool."""

    @patch("enhanced_agents.psycopg2")
    def test_strong_positive(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        # Perfectly correlated data
        n = 20
        df = pd.DataFrame({"hrv": np.arange(n, dtype=float),
                           "sleep": np.arange(n, dtype=float)})
        with patch("enhanced_agents.pd.read_sql_query", return_value=df):
            result = calculate_correlation("hrv", "sleep", 30)
        assert "1.000" in result
        assert "Strong" in result

    @patch("enhanced_agents.psycopg2")
    def test_insufficient_data(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        df = pd.DataFrame({"hrv": [1.0, 2.0], "sleep": [3.0, 4.0]})
        with patch("enhanced_agents.pd.read_sql_query", return_value=df):
            result = calculate_correlation("hrv", "sleep", 30)
        assert "Not enough" in result


# ─── analyze_pattern ──────────────────────────────────────────


class TestAnalyzePattern:
    """Test the pattern analysis tool."""

    @patch("enhanced_agents.psycopg2")
    def test_returns_stats(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        vals = list(range(10, 25))  # 15 days of data
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=len(vals)),
            "stress_level": vals,
        })
        with patch("enhanced_agents.pd.read_sql_query", return_value=df):
            result = analyze_pattern("stress_level", 30)
        assert "Mean" in result
        assert "Trend" in result

    @patch("enhanced_agents.psycopg2")
    def test_insufficient_data(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        df = pd.DataFrame({"date": ["2026-01-01"], "stress_level": [50]})
        with patch("enhanced_agents.pd.read_sql_query", return_value=df):
            result = analyze_pattern("stress_level", 30)
        assert "Not enough" in result


# ─── get_past_recommendations ─────────────────────────────────


def get_past_recommendations(weeks: int = 4) -> str:
    return _agents_mod.get_past_recommendations.run(weeks=weeks)


class TestGetPastRecommendations:
    """Test the past recommendations retrieval tool."""

    @patch("enhanced_agents.psycopg2")
    def test_no_recommendations(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        with patch("enhanced_agents.pd.read_sql_query", return_value=pd.DataFrame()):
            result = get_past_recommendations(4)
        assert "No past recommendations" in result or "first analysis" in result

    @patch("enhanced_agents.psycopg2")
    def test_returns_past_recs(self, mock_pg):
        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn

        df = pd.DataFrame({
            "week_date": ["2026-02-10"],
            "agent_name": ["synthesizer"],
            "recommendation": ["Increase sleep duration"],
            "target_metric": ["sleep_score"],
            "expected_direction": ["IMPROVE"],
            "status": ["pending"],
            "outcome_notes": [None],
        })
        with patch("enhanced_agents.pd.read_sql_query", return_value=df):
            result = get_past_recommendations(4)
        assert "Increase sleep duration" in result
        assert "sleep_score" in result


# ─── _parse_recommendations ──────────────────────────────────


class TestParseRecommendations:
    """Test the recommendation parser."""

    def test_parses_structured_output(self):
        from enhanced_agents import AdvancedHealthAgents
        text = """
        Here are your quick wins:

        RECOMMENDATION: Aim for 7+ hours of sleep on training days
        TARGET_METRIC: sleep_score
        EXPECTED_DIRECTION: IMPROVE

        RECOMMENDATION: Reduce evening screen time
        TARGET_METRIC: deep_sleep_sec
        EXPECTED_DIRECTION: IMPROVE
        """
        recs = AdvancedHealthAgents._parse_recommendations(text)
        assert len(recs) == 2
        assert recs[0]["recommendation"] == "Aim for 7+ hours of sleep on training days"
        assert recs[0]["target_metric"] == "sleep_score"
        assert recs[0]["expected_direction"] == "IMPROVE"
        assert recs[1]["target_metric"] == "deep_sleep_sec"

    def test_empty_output(self):
        from enhanced_agents import AdvancedHealthAgents
        recs = AdvancedHealthAgents._parse_recommendations("No structured recommendations here.")
        assert recs == []


# ─── save_recommendations_to_db ──────────────────────────────


class TestSaveRecommendations:
    """Test saving recommendations to DB."""

    @patch("enhanced_agents.psycopg2")
    def test_saves_to_db(self, mock_pg):
        from enhanced_agents import save_recommendations_to_db
        from datetime import date

        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cur = MagicMock()
        mock_conn.cursor.return_value = mock_cur

        recs = [
            {"recommendation": "Sleep more", "target_metric": "sleep_score",
             "expected_direction": "IMPROVE"},
        ]
        save_recommendations_to_db(recs, week_date=date(2026, 2, 10))

        # Should have called execute for CREATE TABLE + INSERT
        assert mock_cur.execute.call_count >= 2

    @patch("enhanced_agents.psycopg2")
    def test_empty_list_does_nothing(self, mock_pg):
        from enhanced_agents import save_recommendations_to_db
        save_recommendations_to_db([])
        mock_pg.connect.assert_not_called()

