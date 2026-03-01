"""
Tests for the enhanced Garmin data fetcher.

Covers: deep_vars, _safe_val, _apply_quality_fixes.
"""
import math

import numpy as np
import pandas as pd
import pytest

from enhanced_fetcher import deep_vars, _safe_val, EnhancedGarminDataFetcher


# ─── deep_vars ────────────────────────────────────────────────


class SimpleObj:
    """Minimal attribute-holding object for testing deep_vars."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestDeepVars:
    """Test recursive attribute extraction."""

    def test_flat_object(self):
        obj = SimpleObj(a=1, b="hello")
        result = deep_vars(obj)
        assert result["a"] == 1
        assert result["b"] == "hello"

    def test_nested_object(self):
        inner = SimpleObj(x=42)
        outer = SimpleObj(child=inner, y=99)
        result = deep_vars(outer)
        # deep_vars uses dot notation for nested keys
        assert result["child.x"] == 42
        assert result["y"] == 99

    def test_none_value_kept(self):
        obj = SimpleObj(val=None)
        result = deep_vars(obj)
        assert "val" in result
        assert result["val"] is None

    def test_small_list_kept(self):
        """Small lists (<=10 items) are kept as-is."""
        obj = SimpleObj(items=[1, 2, 3])
        result = deep_vars(obj)
        assert result["items"] == [1, 2, 3]

    def test_large_list_summarized(self):
        """Lists >10 items are replaced with a summary string."""
        obj = SimpleObj(items=list(range(20)))
        result = deep_vars(obj)
        assert "20 items" in result["items"]

    def test_prefix_applied(self):
        obj = SimpleObj(v=10)
        result = deep_vars(obj, prefix="pre")
        # With prefix set, keys become prefix.key
        assert "pre.v" in result
        assert result["pre.v"] == 10


# ─── _safe_val ────────────────────────────────────────────────


class TestSafeVal:
    """Test PostgreSQL-safe value conversions."""

    def test_nan_to_none(self):
        assert _safe_val(float("nan")) is None

    def test_nat_to_none(self):
        assert _safe_val(pd.NaT) is None

    def test_none_stays_none(self):
        assert _safe_val(None) is None

    def test_numpy_int_to_python_int(self):
        val = _safe_val(np.int64(42))
        assert val == 42
        assert isinstance(val, int)

    def test_numpy_float_to_python_float(self):
        val = _safe_val(np.float64(3.14))
        assert abs(val - 3.14) < 1e-10
        assert isinstance(val, float)

    def test_normal_string_unchanged(self):
        assert _safe_val("hello") == "hello"

    def test_normal_int_unchanged(self):
        assert _safe_val(7) == 7


# ─── _apply_quality_fixes ────────────────────────────────────


class TestApplyQualityFixes:
    """Test data quality clamping rules."""

    @staticmethod
    def _make_fetcher():
        """Create an EnhancedGarminDataFetcher without actual DB connection."""
        fetcher = EnhancedGarminDataFetcher.__new__(EnhancedGarminDataFetcher)
        return fetcher

    def test_hrv_511_nullified(self):
        """HRV weekly average of 511 (sensor sentinel) should become None."""
        fetcher = self._make_fetcher()
        daily = {"2026-01-01": {"date": "2026-01-01", "tr_hrv_weekly_avg": 511, "hydration_value_ml": 2000}}
        fixed = fetcher._apply_quality_fixes(daily)
        assert fixed["2026-01-01"]["tr_hrv_weekly_avg"] is None

    def test_hrv_normal_unchanged(self):
        fetcher = self._make_fetcher()
        daily = {"2026-01-01": {"date": "2026-01-01", "tr_hrv_weekly_avg": 45.0, "hydration_value_ml": 2000}}
        fixed = fetcher._apply_quality_fixes(daily)
        assert fixed["2026-01-01"]["tr_hrv_weekly_avg"] == 45.0


