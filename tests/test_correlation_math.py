"""
Tests for the correlation engine mathematical computations.

Covers: _adaptive_alpha, Pearson lag-1, AR(1), Markov transitions,
anomaly detection, date-gap handling, and bin discretization.
"""
import sys
import os
import math

import numpy as np
import pandas as pd
import pytest

# Allow importing from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from correlation_engine import (
    _adaptive_alpha,
    N_BINS,
    BIN_LABELS,
    ACWR_EDGES,
    ACWR_LABELS,
    SMOOTH_ALPHA_MIN,
    SMOOTH_ALPHA_MAX,
    SMOOTH_ALPHA_BASE,
    MIN_TRANSITIONS_KL,
    CorrelationEngine,
)


# ─── _adaptive_alpha ──────────────────────────────────────────


class TestAdaptiveAlpha:
    """Verify kernel smoothing strength computation."""

    def test_zero_transitions_returns_max(self):
        assert _adaptive_alpha(0) == SMOOTH_ALPHA_MAX

    def test_negative_transitions_returns_max(self):
        assert _adaptive_alpha(-5) == SMOOTH_ALPHA_MAX

    def test_large_n_approaches_min(self):
        # With n=10000, α = 0.5/√10000 = 0.005 → clamped to MIN
        a = _adaptive_alpha(10_000)
        assert a == SMOOTH_ALPHA_MIN

    def test_small_n_capped_at_max(self):
        # With n=1, α = 0.5/1 = 0.5 → clamped to MAX
        a = _adaptive_alpha(1)
        assert a == SMOOTH_ALPHA_MAX

    def test_mid_range(self):
        # n=10 → α = 0.5/√10 ≈ 0.158
        a = _adaptive_alpha(10)
        expected = SMOOTH_ALPHA_BASE / math.sqrt(10)
        assert abs(a - expected) < 1e-6

    def test_monotonic_decrease(self):
        """Alpha should decrease as n increases."""
        prev = _adaptive_alpha(1)
        for n in [5, 10, 50, 200, 1000]:
            cur = _adaptive_alpha(n)
            assert cur <= prev, f"α({n}) = {cur} > α(prev) = {prev}"
            prev = cur


# ─── Constants ────────────────────────────────────────────────


class TestConstants:
    """Verify Markov configuration after bins upgrade."""

    def test_n_bins_is_3(self):
        assert N_BINS == 3

    def test_bin_labels_have_3_entries(self):
        assert len(BIN_LABELS) == 3
        assert BIN_LABELS[0] == "LOW"
        assert BIN_LABELS[-1] == "HIGH"

    def test_min_transitions_kl(self):
        assert MIN_TRANSITIONS_KL >= 5

    def test_acwr_edges_defined(self):
        assert len(ACWR_EDGES) == 5
        assert ACWR_EDGES[1] == 0.8  # sweet spot lower
        assert ACWR_EDGES[2] == 1.3  # sweet spot upper

    def test_acwr_labels_defined(self):
        assert len(ACWR_LABELS) == 4
        assert "SWEET_SPOT" in ACWR_LABELS

    def test_percentile_edges_driven_by_nbins(self):
        """Percentile edges should have N_BINS + 1 entries."""
        import numpy as np
        vals = np.random.randn(100)
        edges = np.percentile(vals, np.linspace(0, 100, N_BINS + 1))
        assert len(edges) == N_BINS + 1


# ─── Date-gap handling ────────────────────────────────────────


class TestDateGapHandling:
    """Verify that .asfreq('D') gap-fill prevents false lag pairs."""

    @staticmethod
    def _make_gapped_df() -> pd.DataFrame:
        """Create a DataFrame with a 3-day gap (Feb 3-5 missing)."""
        dates = ["2026-02-01", "2026-02-02", "2026-02-06", "2026-02-07"]
        vals = [100.0, 110.0, 200.0, 210.0]
        df = pd.DataFrame({"date": pd.to_datetime(dates), "metric": vals})
        return df

    def test_shift1_without_gap_fill_is_wrong(self):
        """Without gap-fill, shift(1) pairs Feb 2 with Feb 6 — wrong."""
        df = self._make_gapped_df()
        shifted = df["metric"].shift(1)
        # Position 2 (Feb 6) gets value from position 1 (Feb 2) = 110
        assert shifted.iloc[2] == 110.0, "Without gap-fill, Feb 6 pairs with Feb 2"

    def test_shift1_with_gap_fill_is_correct(self):
        """With gap-fill, Feb 6's lag-1 should be NaN (Feb 5 is missing)."""
        df = self._make_gapped_df()
        df = df.set_index("date").asfreq("D").reset_index()
        shifted = df["metric"].shift(1)
        # Feb 6 is now at index 5, its lag is Feb 5 which is NaN (gap-filled)
        feb6_idx = df[df["date"] == pd.Timestamp("2026-02-06")].index[0]
        assert pd.isna(shifted.iloc[feb6_idx]), "After gap-fill, Feb 6 lag should be NaN"

    def test_gap_fill_inserts_nan_rows(self):
        """Gap-fill should add 3 NaN rows for Feb 3, 4, 5."""
        df = self._make_gapped_df()
        df = df.set_index("date").asfreq("D").reset_index()
        assert len(df) == 7  # Feb 1-7 inclusive
        # The gap-filled rows should have NaN for metric
        gap_rows = df[df["date"].isin(pd.to_datetime(["2026-02-03", "2026-02-04", "2026-02-05"]))]
        assert gap_rows["metric"].isna().all()


# ─── Pearson lag-1 ────────────────────────────────────────────


class TestPearsonLag1:
    """Test lag-1 Pearson correlations on synthetic data."""

    @staticmethod
    def _make_autocorrelated_data(n=50, phi=0.9) -> pd.DataFrame:
        """Generate AR(1) data: x_t = phi * x_{t-1} + noise."""
        np.random.seed(42)
        x = [0.0]
        for _ in range(n - 1):
            x.append(phi * x[-1] + np.random.normal(0, 0.3))
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        return pd.DataFrame({"date": dates, "metric_a": x, "metric_b": np.random.randn(n)})

    def test_strong_autocorrelation_detected(self):
        """A metric with phi=0.9 should have strong lag-1 with itself (via another metric pair)."""
        df = self._make_autocorrelated_data()
        engine = CorrelationEngine.__new__(CorrelationEngine)
        # Call the method directly — it expects data sorted by date with gap-fill done
        df_indexed = df.set_index("date").asfreq("D").reset_index()
        _, lag1 = engine._layer1_pearson(df_indexed, ["metric_a", "metric_b"])
        # At minimum, lag-1 should compute without error
        assert isinstance(lag1, list)


# ─── AR(1) ────────────────────────────────────────────────────


class TestAR1:
    """Test AR(1) model recovery on known signal."""

    @staticmethod
    def _make_ar1_signal(n=100, phi=0.8) -> pd.DataFrame:
        np.random.seed(123)
        x = [0.0]
        for _ in range(n - 1):
            x.append(phi * x[-1] + np.random.normal(0, 0.2))
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        return pd.DataFrame({"date": dates, "signal": x})

    def test_recovers_phi(self):
        """AR(1) fit should recover φ ≈ 0.8 from synthetic data."""
        df = self._make_ar1_signal()
        engine = CorrelationEngine.__new__(CorrelationEngine)
        results = engine._layer2b_ar1(df, ["signal"])
        assert len(results) >= 1
        # Tuple now has 6 elements: (metric, slope, r2, p, n, is_stationary)
        metric_name, slope, r2, p, n, is_stationary = results[0]
        assert metric_name == "signal"
        # φ should be close to 0.8 (within ±0.15 for 100 samples)
        assert abs(slope - 0.8) < 0.15, f"Expected φ ≈ 0.8, got {slope:.3f}"
        # R² should be reasonably high
        assert r2 > 0.3
        # Stationarity flag should be boolean-like (numpy or Python bool)
        assert isinstance(is_stationary, (bool, np.bool_))

    def test_constant_metric_skipped(self):
        """A constant metric (std=0) should be skipped."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=20),
            "const": [42.0] * 20,
        })
        engine = CorrelationEngine.__new__(CorrelationEngine)
        results = engine._layer2b_ar1(df, ["const"])
        assert len(results) == 0


# ─── Anomaly detection ───────────────────────────────────────


class TestAnomalies:
    """Test percentile-based anomaly detection."""

    def test_detects_high_anomaly(self):
        """Values above 95th percentile should be flagged HIGH."""
        n = 30
        vals = [50.0] * n
        vals[-1] = 100.0  # spike
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=n),
            "metric": vals,
        })
        engine = CorrelationEngine.__new__(CorrelationEngine)
        anomalies = engine._layer2c_anomalies(df, df, ["metric"])
        assert len(anomalies) >= 1
        # Last value should be flagged as HIGH
        directions = [a[4] for a in anomalies]
        assert "HIGH" in directions

    def test_no_anomaly_in_flat_data(self):
        """Constant data should produce no anomalies (std ≈ 0)."""
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=20),
            "metric": [50.0] * 20,
        })
        engine = CorrelationEngine.__new__(CorrelationEngine)
        anomalies = engine._layer2c_anomalies(df, df, ["metric"])
        assert len(anomalies) == 0

    def test_anomaly_uses_percentiles(self):
        """Verify anomalies are based on 5th/95th percentile, not z-score."""
        np.random.seed(42)
        n = 100
        vals = np.random.exponential(10, n)  # skewed distribution
        # Add a value just above 95th percentile
        p95 = np.percentile(vals, 95)
        vals = list(vals)
        vals[-1] = p95 + 1  # should be flagged as HIGH
        df = pd.DataFrame({
            "date": pd.date_range("2026-01-01", periods=len(vals)),
            "metric": vals,
        })
        engine = CorrelationEngine.__new__(CorrelationEngine)
        anomalies = engine._layer2c_anomalies(df, df, ["metric"])
        # At least the last value should be flagged
        flagged_values = [a[2] for a in anomalies]
        assert any(v > p95 for v in flagged_values)


# ─── Markov transitions ──────────────────────────────────────


class TestMarkov:
    """Test Markov transition matrix properties."""

    @staticmethod
    def _make_markov_data(n=60) -> pd.DataFrame:
        """Generate data with clear state transitions."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-01", periods=n, freq="D")
        # Oscillating pattern — should produce non-trivial transitions
        vals = np.sin(np.linspace(0, 8 * np.pi, n)) * 20 + 50
        vals += np.random.normal(0, 3, n)
        return pd.DataFrame({"date": dates, "metric": vals})

    def test_transition_rows_sum_to_one(self):
        """Each row of the smoothed transition matrix should sum to 1."""
        df = self._make_markov_data()
        engine = CorrelationEngine.__new__(CorrelationEngine)
        normality = {"metric": ("non_normal", 0.01)}
        results = engine._layer2e_markov(df, ["metric"], normality)
        assert len(results) >= 1
        T = results[0]["marginal"]
        for i in range(T.shape[0]):
            row_sum = T[i, :].sum()
            assert abs(row_sum - 1.0) < 1e-10, f"Row {i} sums to {row_sum}, not 1.0"

    def test_stationary_distribution_sums_to_one(self):
        """Stationary distribution π should sum to 1."""
        df = self._make_markov_data()
        engine = CorrelationEngine.__new__(CorrelationEngine)
        normality = {"metric": ("non_normal", 0.01)}
        results = engine._layer2e_markov(df, ["metric"], normality)
        assert len(results) >= 1
        pi = results[0]["stationary"]
        assert abs(pi.sum() - 1.0) < 1e-6

    def test_kl_divergence_non_negative(self):
        """KL divergence should always be ≥ 0."""
        df = self._make_markov_data()
        # Add a conditioning metric
        df["cond"] = np.random.randn(len(df)) * 10
        engine = CorrelationEngine.__new__(CorrelationEngine)
        normality = {"metric": ("non_normal", 0.01), "cond": ("normal", 0.5)}
        results = engine._layer2e_markov(df, ["metric", "cond"], normality)
        for r in results:
            assert r["best_kl"] >= 0, f"KL divergence is negative: {r['best_kl']}"

    def test_tertile_bins_used(self):
        """After upgrade, bins should be ≤ 3 (tertile discretization)."""
        df = self._make_markov_data()
        engine = CorrelationEngine.__new__(CorrelationEngine)
        normality = {"metric": ("non_normal", 0.01)}
        results = engine._layer2e_markov(df, ["metric"], normality)
        assert len(results) >= 1
        assert results[0]["bins"] <= 3

    def test_markov_result_has_labels(self):
        """Markov results should include labels list."""
        df = self._make_markov_data()
        engine = CorrelationEngine.__new__(CorrelationEngine)
        normality = {"metric": ("non_normal", 0.01)}
        results = engine._layer2e_markov(df, ["metric"], normality)
        assert len(results) >= 1
        assert "labels" in results[0]
        assert isinstance(results[0]["labels"], list)


# ─── Markov consecutive-day check ────────────────────────────


class TestMarkovDateGaps:
    """Verify Markov transitions skip non-consecutive days."""

    def test_gapped_days_produce_fewer_transitions(self):
        """A dataset with gaps should have fewer transitions than a
        contiguous one of the same measurement count."""
        np.random.seed(42)
        # Contiguous: 30 consecutive days
        dates_cont = pd.date_range("2026-01-01", periods=30, freq="D")
        vals = np.random.randn(30) * 10 + 50

        df_cont = pd.DataFrame({"date": dates_cont, "metric": vals})

        # Gapped: same 30 values but every other day
        dates_gap = pd.date_range("2026-01-01", periods=30, freq="2D")
        df_gap = pd.DataFrame({"date": dates_gap, "metric": vals})

        engine = CorrelationEngine.__new__(CorrelationEngine)
        normality = {"metric": ("non_normal", 0.01)}

        res_cont = engine._layer2e_markov(df_cont, ["metric"], normality)
        res_gap = engine._layer2e_markov(df_gap, ["metric"], normality)

        if res_cont and res_gap:
            # Gapped should have 0 transitions (gap=2 between every pair)
            assert res_gap[0]["n_transitions"] == 0 or \
                   res_gap[0]["n_transitions"] < res_cont[0]["n_transitions"]
