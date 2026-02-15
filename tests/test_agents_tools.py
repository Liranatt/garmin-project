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
