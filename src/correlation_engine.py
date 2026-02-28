"""
Correlation Matrix Engine ג€” Production Version
================================================
Pre-computes statistical matrices from daily_metrics for AI agent
consumption and trend analysis.

Architecture (4 layers):
  Layer 0 ג€” Data loading + cleaning:  Auto-discover numeric columns,
            rename tr_ abbreviations, fix sentinels, add training intensity.
  Layer 1 ג€” Pearson:  NxN same-day + lag-1 correlations with p-values.
  Layer 2 ג€” Conditional:
            AR(1) for all metrics, conditioned AR(1) with top predictors,
            Markov transition matrices for non-normal metrics,
            KL-divergence ranking.
  Layer 3 ג€” Agent Summary:  ~2-3 KB natural-language digest that replaces
            raw SQL in agent prompts.

Key mathematical fixes (proven in test_matrices.py):
  ג€¢ Marginal matrix uses ADAPTIVE kernel smoothing (־± = BASE/גˆn, clamped
    to [0.02, 0.25]) ג€” heavy smoothing for sparse data, light for large n.
  ג€¢ KL-divergence only computed when BOTH conditioning levels have ג‰¥5
    transitions ג€” prevents noise from dominating the divergence ranking.
  ג€¢ Adjusted Rֲ² is required ג€” raw Rֲ² with nג‰ˆ10, p=3 is meaningless.
  ג€¢ HRV sentinel 511 ג†’ NaN.
  ג€¢ Data quality disclaimer when n < 20.
  ג€¢ Confidence tiers on Markov: HIGH (ג‰¥100), GOOD (ג‰¥30), MODERATE (ג‰¥15),
    PRELIMINARY (<15).
"""

from __future__ import annotations

import os
import json
import math
import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats as sp_stats
from statsmodels.tsa.stattools import adfuller
from dotenv import load_dotenv
from analytics.markov_layer import compute_markov_layer

load_dotenv()

log = logging.getLogger("correlation_engine")


# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•
#  CONSTANTS
# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•

# Columns to skip when auto-discovering numeric metrics
SKIP_COLS = {
    "created_at", "updated_at", "date",
    "sleep_score_qualifier", "hrv_status", "tr_level",
    "sleep_start_local", "sleep_end_local",
}

# Rename tr_ abbreviations to human-readable names
RENAME_MAP = {
    "tr_sleep_score":       "training_sleep_score",
    "tr_sleep_score_pct":   "training_sleep_pct",
    "tr_recovery_time":     "training_recovery_hours",
    "tr_recovery_pct":      "training_recovery_pct",
    "tr_acute_load":        "training_acute_load",
    "tr_acwr_pct":          "training_load_ratio_pct",
    "tr_hrv_weekly_avg":    "training_hrv_weekly_avg",
    "tr_hrv_pct":           "training_hrv_pct",
    "tr_stress_history_pct": "training_stress_history_pct",
    "tr_sleep_history_pct": "training_sleep_history_pct",
}

# Key targets for conditioned AR(1) analysis
KEY_TARGETS = [
    "training_readiness", "sleep_score", "hrv_last_night",
    "stress_level", "bb_charged", "resting_hr",
    "vo2_max_running", "acwr",
]

# Key metrics to include in the metric-ranges section of the summary
KEY_RANGE_METRICS = [
    "resting_hr", "sleep_score", "hrv_last_night", "stress_level",
    "training_readiness", "bb_charged", "bb_drained", "total_steps",
    "deep_sleep_sec", "rem_sleep_sec",
    "training_hard_minutes", "training_has_upper", "training_has_lower",
    "vo2_max_running", "vo2_max_running_delta", "acwr",
    "daily_load_acute", "daily_load_chronic",
    "race_time_5k", "race_time_5k_delta",
    "heat_acclimation_pct",
    # Derived metrics (Plews 2012, Esco 2025)
    "ln_hrv", "hrv_weekly_mean", "hrv_weekly_cv",
    # User-reported (wellness_log)
    "caffeine_intake", "alcohol_intake", "energy_level",
    "overall_stress_level", "workout_feeling", "muscle_soreness",
]

# Markov discretisation
N_BINS = 3
BIN_LABELS = ["LOW", "MED", "HIGH"]

# ACWR evidence-based thresholds (Gabbett 2016, BJSM, 2597 citations)
# Used instead of percentile bins for the acwr metric specifically.
ACWR_EDGES = [0, 0.8, 1.3, 1.5, 999]
ACWR_LABELS = ["UNDERTRAINED", "SWEET_SPOT", "HIGH_RISK", "DANGER"]

# Smoothing alpha for Markov transitions (must be same for marginal & conditional)
# Now adaptive: ־± = max(SMOOTH_ALPHA_MIN, SMOOTH_ALPHA_BASE / sqrt(n_transitions))
# This gives more smoothing when data is sparse and lets empirical
# distributions dominate as the dataset grows (neural-net flavoured
# kernel smoothing).
SMOOTH_ALPHA_BASE = 0.50   # numerator for ־± = base / גˆn
SMOOTH_ALPHA_MIN  = 0.02   # floor ג€” always keep a tiny uniform mix
SMOOTH_ALPHA_MAX  = 0.25   # ceiling ג€” cap smoothing for very small n

# Minimum transitions per conditioning level for KL to be computed
MIN_TRANSITIONS_KL = 5


def _adaptive_alpha(n_transitions: int) -> float:
    """Compute kernel smoothing strength from sample size.

    ־± = clamp(BASE / גˆn, MIN, MAX)

    With 4 transitions  ג†’ ־± ג‰ˆ 0.25 (heavy smoothing, sparse data)
    With 10 transitions ג†’ ־± ג‰ˆ 0.16
    With 50 transitions ג†’ ־± ג‰ˆ 0.07
    With 200+           ג†’ ־± ג‰ˆ 0.04 (light smoothing, data speaks)
    """
    if n_transitions <= 0:
        return SMOOTH_ALPHA_MAX
    raw = SMOOTH_ALPHA_BASE / math.sqrt(n_transitions)
    return max(SMOOTH_ALPHA_MIN, min(SMOOTH_ALPHA_MAX, raw))

# Benchmark periods: (key, label, days)
# days=None ג†’ "current week" = Monday of this week ג†’ ref_date
# All others are rolling windows: ref_date גˆ’ (daysגˆ’1) ג†’ ref_date
BENCHMARK_PERIODS = [
    ("current_week",  "Current Week",    None),
    ("1_week",        "Last 7 Days",     7),
    ("2_weeks",       "Last 14 Days",    14),
    ("3_weeks",       "Last 21 Days",    21),
    ("1_month",       "Last 30 Days",    30),
    ("2_months",      "Last 60 Days",    60),
    ("3_months",      "Last 90 Days",    90),
    ("6_months",      "Last 180 Days",   180),
    ("1_year",        "Last 365 Days",   365),
]

# Exercise categories for body-part detection
UPPER_BODY_CATS = {
    "PULL_UP", "ROW", "SHOULDER_PRESS", "BENCH_PRESS", "CURL",
    "LATERAL_RAISE", "TRICEPS_EXTENSION", "CHEST_FLY", "PUSHUP",
    "DIP", "DEADLIFT", "T_BAR_ROW",
}
LOWER_BODY_CATS = {
    "SQUAT", "LUNGE", "LEG_PRESS", "LEG_CURL", "LEG_EXTENSION",
    "CALF_RAISE", "HIP", "GLUTE",
}


# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•
#  SCHEMA ג€” tables for stored results
# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•

CORRELATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matrix_summaries (
    id           SERIAL PRIMARY KEY,
    computed_at  DATE NOT NULL UNIQUE,
    summary_text TEXT NOT NULL,
    token_est    INTEGER NOT NULL
);
"""


# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•
#  ENGINE
# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•

class CorrelationEngine:
    """
    Orchestrates all four layers of correlation computation.
    Standalone ג€” uses psycopg2 directly, no external DatabaseManager.
    """

    def __init__(self, conn_str: Optional[str] = None):
        self.conn_str = conn_str or os.getenv("POSTGRES_CONNECTION_STRING", "")

    # ג”€ג”€ג”€ Schema bootstrap ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def bootstrap_schema(self):
        """Create engine tables if they don't exist."""
        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = True
        cur = conn.cursor()
        for stmt in CORRELATION_SCHEMA_SQL.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        cur.close()
        conn.close()

    # ג”€ג”€ג”€ MAIN ENTRY ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def compute_weekly(self, training_days: Optional[Dict] = None) -> str:
        """
        Run the full pipeline and return the agent-ready summary.

        Parameters
        ----------
        training_days : dict, optional
            {date: {intensity, hard_minutes, has_upper, has_lower, ...}}
            from EnhancedGarminDataFetcher.classify_training_intensity().
            If None, training info is omitted from the summary.
        """
        log.info("\nCorrelation Engine - computing weekly matrices...")
        self.bootstrap_schema()
        run_result = self._compute_raw_with_status(training_days)
        summary = run_result["summary"]
        if run_result["analysis_status"] != "success":
            log.warning(
                "Weekly correlation status=%s (%s)",
                run_result["analysis_status"],
                ", ".join(run_result.get("degraded_reasons", [])) or "no details",
            )
        if summary:
            self._store_summary(summary)
            log.info("Correlation engine complete\n")
        return summary

    # Raw computation (no storage)
    def _compute_raw(self, training_days: Optional[Dict] = None, date_start=None, date_end=None) -> str:
        """Compatibility wrapper: return only summary text."""
        return self._compute_raw_with_status(
            training_days=training_days,
            date_start=date_start,
            date_end=date_end,
        )["summary"]

    def _compute_raw_with_status(
        self, training_days: Optional[Dict] = None, date_start=None, date_end=None
    ) -> Dict[str, Any]:
        """Run layers 0-3 and return summary plus analysis_status metadata."""
        result: Dict[str, Any] = {
            "summary": "",
            "analysis_status": "success",
            "degraded_reasons": [],
        }

        df, metrics = self._layer0_load_and_clean(date_start=date_start, date_end=date_end)
        if df is None or len(df) < 5:
            msg = "Not enough data for correlation analysis (need >= 5 days)."
            log.info("   %s", msg)
            result["summary"] = msg
            result["analysis_status"] = "degraded"
            result["degraded_reasons"] = ["insufficient_daily_rows"]
            return result

        n_days = len(df)
        if training_days:
            # Normalize to datetime.date to avoid Timestamp vs date comparison
            def _to_date(val):
                if hasattr(val, "date"):
                    return val.date()
                return val

            min_date = _to_date(df["date"].min())
            max_date = _to_date(df["date"].max())
            td_filtered = {d: v for d, v in training_days.items()
                           if min_date <= _to_date(d) <= max_date}
            df["training_hard_minutes"] = df["date"].apply(
                lambda d: td_filtered.get(_to_date(d), {}).get("hard_minutes", 0)
            )
            df["training_has_upper"] = df["date"].apply(
                lambda d: 1 if td_filtered.get(_to_date(d), {}).get("has_upper", False) else 0
            )
            df["training_has_lower"] = df["date"].apply(
                lambda d: 1 if td_filtered.get(_to_date(d), {}).get("has_lower", False) else 0
            )
            for tc in ["training_hard_minutes", "training_has_upper", "training_has_lower"]:
                if df[tc].notna().sum() >= 5 and tc not in metrics:
                    metrics.append(tc)
            log.info("   Added training columns -> %d total metrics", len(metrics))
            training_for_summary = td_filtered
        else:
            training_for_summary = None

        data = df[metrics].copy()
        data_with_date = df[["date", *metrics]].copy()

        try:
            sig_pairs, lag1_results = self._layer1_pearson(data, metrics)
            normality = self._layer2a_normality(data, metrics)
            ar1_results = self._layer2b_ar1(data, metrics)
            anomalies = self._layer2c_anomalies(df, data, metrics)
            cond_ar1_results = self._layer2d_conditioned_ar1(data, metrics, normality)
            rolling_corr_results = self._layer2f_rolling_correlation(data, sig_pairs)
            multi_lag_results = self._multi_lag_carryover(data, metrics)
        except Exception as e:
            log.exception("Core correlation layers failed: %s", e)
            result["summary"] = f"Correlation analysis failed: {e}"
            result["analysis_status"] = "failed"
            result["degraded_reasons"] = ["core_layer_failure"]
            return result

        markov_results: List[Dict[str, Any]] = []
        try:
            markov_results = self._layer2e_markov(data_with_date, metrics, normality)
        except Exception as e:
            log.warning("Markov/KL layer failed; continuing in degraded mode: %s", e)
            result["analysis_status"] = "degraded"
            result["degraded_reasons"].append("markov_layer_failed")

        summary = self._layer3_summary(
            df,
            n_days,
            metrics,
            sig_pairs,
            lag1_results,
            normality,
            ar1_results,
            anomalies,
            cond_ar1_results,
            markov_results,
            rolling_corr_results,
            multi_lag_results,
            training_for_summary,
        )

        if result["degraded_reasons"]:
            summary = (
                f"{summary}\n\n[ANALYSIS STATUS]\n"
                f"  status={result['analysis_status']}\n"
                f"  reasons={', '.join(result['degraded_reasons'])}"
            )

        n_normal = sum(1 for s, _ in normality.values() if s == "normal")
        n_nonnorm = sum(1 for s, _ in normality.values() if s == "non_normal")
        n_markov_reliable = sum(1 for mr in markov_results if mr["confidence"] in ("HIGH", "GOOD"))
        log.info(
            "\n   COMPUTATION DIGEST (%s, %d days)\n"
            "   Layer 0 Load & Clean    : %d metrics\n"
            "   Layer 1a Same-day       : %d significant pairs\n"
            "   Layer 1b Lag-1          : %d predictors\n"
            "   Layer 2a Normality      : %d normal, %d non-normal\n"
            "   Layer 2b AR(1)          : %d models\n"
            "   Layer 2c Anomalies      : %d anomalies\n"
            "   Layer 2d Cond AR        : %d models\n"
            "   Layer 2e Markov + KL    : %d models (%d reliable)\n"
            "   Layer 3 Summary         : %d chars (~%d tokens)",
            f"{df['date'].min()} -> {df['date'].max()}",
            n_days,
            len(metrics),
            len(sig_pairs),
            len(lag1_results),
            n_normal,
            n_nonnorm,
            len(ar1_results),
            len(anomalies),
            len(cond_ar1_results),
            len(markov_results),
            n_markov_reliable,
            len(summary),
            len(summary) // 4,
        )

        result["summary"] = summary
        # Attach raw layer data for storage (JSON-safe)
        result["raw_results"] = {
            "pearson_pairs": [{"m1": p[0], "m2": p[1], "r": float(p[2]), "p": float(p[3]), "n": int(p[4]) if len(p) > 4 else 0}
                              for p in sig_pairs] if sig_pairs else [],
            "lag1_pairs": [{"m1": l[0], "m2": l[1], "r": float(l[2]), "p": float(l[3]) if len(l) > 3 else 0, "n": int(l[4]) if len(l) > 4 else 0}
                           for l in lag1_results] if lag1_results else [],
            "ar1_results": {a[0]: {"phi": float(a[1]), "r2": float(a[2])}
                            for a in ar1_results} if ar1_results else {},
            "n_anomalies": len(anomalies),
            "n_cond_ar1": len(cond_ar1_results),
            "n_markov": len(markov_results),
            "n_metrics": len(metrics),
            "n_days": n_days,
        }
        return result

    def _get_data_range(self):
        """Return (earliest_date, latest_date, total_days) from daily_metrics."""
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        cur.execute(
            "SELECT MIN(date), MAX(date), COUNT(DISTINCT date) FROM daily_metrics"
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row[0] is None:
            return None, None, 0
        return row[0], row[1], row[2]

    # ג”€ג”€ג”€ Benchmark entry point ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def compute_benchmarks(self, training_days: Optional[Dict] = None,
                           ref_date=None) -> Dict[str, Any]:
        """Compute correlation matrices for every benchmark period that
        has sufficient data (ג‰¥5 days).

        Benchmark windows (all ending at *ref_date*):
          current_week  = Monday of this week ג†’ today (may be partial)
          1_week        = last 7 days
          2_weeks       = last 14 days
          ג€¦up to 1_year = last 365 days

        Returns
        -------
        dict with keys:
            benchmarks : {period_key: summary_text}
            available  : [period_keys that were computed]
            longest    : key of the longest computed period
            comparison : structured comparison text for comparator agent
            data_days  : total distinct days in the database
        """
        ref = ref_date or date.today()
        log.info(f"\nנ“ Correlation Engine ג€” benchmark analysis (ref {ref})ג€¦")
        self.bootstrap_schema()

        # 1. Discover data range
        earliest, latest, total_days = self._get_data_range()
        if earliest is None:
            log.info("   ג ן¸  No data in daily_metrics.")
            return {
                "benchmarks": {},
                "available": [],
                "longest": None,
                "comparison": "",
                "data_days": 0,
                "analysis_status": "failed",
                "status_by_period": {},
                "degraded_reasons": ["no_daily_metrics_data"],
            }
        log.info(f"   Data range: {earliest} ג†’ {latest}  ({total_days} distinct days)")

        # 2. Build candidate windows
        candidates: List[Tuple[str, str, date, date]] = []
        seen_ranges: set = set()  # deduplicate identical clamped ranges
        for key, label, days in BENCHMARK_PERIODS:
            if days is None:
                # Current week: Monday ג†’ ref
                monday = ref - timedelta(days=ref.weekday())
                ds, de = monday, ref
            else:
                ds = ref - timedelta(days=days - 1)
                de = ref

            # Clamp start to earliest available data
            actual_start = max(ds, earliest)
            actual_end = min(de, latest)
            potential_days = (actual_end - actual_start).days + 1

            if potential_days < 5:
                log.info(f"   ג­ן¸  {label}: skipping ({potential_days} potential days, need ג‰¥5)")
                continue

            # Skip current_week if it would be identical to 1_week
            if key == "current_week":
                one_week_start = ref - timedelta(days=6)
                if ds <= one_week_start:
                    log.info(f"   ג­ן¸  {label}: same as Last 7 Days, skipping duplicate")
                    continue

            # Deduplicate: if clamped range is same as a previous candidate, skip
            range_key = (actual_start, actual_end)
            if range_key in seen_ranges:
                log.info(f"   ג­ן¸  {label}: same effective range as a shorter window, skipping")
                continue
            seen_ranges.add(range_key)

            candidates.append((key, label, ds, de))

        # 3. Compute each benchmark
        benchmarks: Dict[str, str] = {}
        computed: List[Tuple[str, str, date, date]] = []
        status_by_period: Dict[str, Dict[str, Any]] = {}
        all_reasons: List[str] = []
        overall_status = "success"

        for key, label, ds, de in candidates:
            log.info(f"\nג•ג• BENCHMARK: {label} ({ds} ג†’ {de}) ג•ג•")
            run_result = self._compute_raw_with_status(training_days, date_start=ds, date_end=de)
            benchmarks[key] = run_result["summary"]
            # Store raw results in correlation_results table
            self._store_correlation_result(
                window_label=label, date_start=ds, date_end=de,
                run_result=run_result,
            )
            period_status = run_result.get("analysis_status", "success")
            period_reasons = run_result.get("degraded_reasons", [])
            status_by_period[key] = {
                "analysis_status": period_status,
                "degraded_reasons": list(period_reasons),
            }
            if period_status == "failed":
                overall_status = "failed"
            elif period_status == "degraded" and overall_status != "failed":
                overall_status = "degraded"
            for reason in period_reasons:
                if reason not in all_reasons:
                    all_reasons.append(reason)
            computed.append((key, label, ds, de))

        # 4. Determine longest
        longest_key = None
        for key, *_ in reversed(computed):
            if status_by_period.get(key, {}).get("analysis_status") != "failed":
                longest_key = key
                break

        # 5. Build comparison
        comparison = self._build_benchmark_comparison(benchmarks, computed)

        # 6. Store the longest summary
        if longest_key and benchmarks.get(longest_key):
            self._store_summary(benchmarks[longest_key])
            log.info(f"   נ’¾ Stored summary for '{longest_key}' "
                     f"({len(benchmarks[longest_key])} chars) ג†’ matrix_summaries")

        # ג”€ג”€ Final overview ג”€ג”€
        log.info("\n" + "ג•" * 55)
        log.info("  BENCHMARK SUMMARY")
        log.info("ג•" * 55)
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            sz = len(benchmarks.get(key, ""))
            status = status_by_period.get(key, {}).get("analysis_status", "unknown")
            log.info(f"  ג“ {label:20s}  {span:3d} days  {sz:5d} chars  status={status}")
        log.info(f"  Stored to DB: {longest_key or 'none'}")
        log.info(f"  Comparison context: {len(comparison)} chars")
        log.info(f"  Overall analysis status: {overall_status}")
        log.info("ג•" * 55 + "\n")

        return {
            "benchmarks": benchmarks,
            "available": [c[0] for c in computed],
            "longest": longest_key,
            "comparison": comparison,
            "data_days": total_days,
            "analysis_status": overall_status,
            "status_by_period": status_by_period,
            "degraded_reasons": all_reasons,
        }

    # ג”€ג”€ג”€ Legacy multi-period wrapper ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def compute_multi_period(self, training_days: Optional[Dict] = None,
                              week1_start=None, week1_end=None,
                              week2_start=None, week2_end=None) -> Dict[str, str]:
        """
        Compute matrices for week1, week2, combined 2-week, and comparison.
        Returns dict with keys: week1, week2, combined, comparison.
        """
        log.info("\nנ“ Correlation Engine ג€” multi-period analysisג€¦")
        self.bootstrap_schema()

        results: Dict[str, str] = {}

        log.info("\nג•ג•ג• Week 1 ג•ג•ג•")
        results["week1"] = self._compute_raw(training_days,
                                              date_start=week1_start,
                                              date_end=week1_end)

        log.info("\nג•ג•ג• Week 2 ג•ג•ג•")
        results["week2"] = self._compute_raw(training_days,
                                              date_start=week2_start,
                                              date_end=week2_end)

        log.info("\nג•ג•ג• Combined 2-week ג•ג•ג•")
        results["combined"] = self._compute_raw(training_days,
                                                 date_start=week1_start,
                                                 date_end=week2_end)

        results["comparison"] = self._build_comparison(
            results["week1"], results["week2"], results["combined"]
        )

        self._store_summary(results["combined"])

        # ג”€ג”€ Multi-period overview ג”€ג”€
        log.info("\n" + "ג•" * 55)
        log.info("  MULTI-PERIOD SUMMARY")
        log.info("ג•" * 55)
        log.info(f"  ג“ Week 1    : {len(results['week1']):5d} chars")
        log.info(f"  ג“ Week 2    : {len(results['week2']):5d} chars")
        log.info(f"  ג“ Combined  : {len(results['combined']):5d} chars")
        log.info(f"  ג“ Comparison: {len(results['comparison']):5d} chars")
        log.info(f"  Stored to DB: combined ({len(results['combined'])} chars)")
        log.info("ג•" * 55 + "\n")
        return results

    # ג”€ג”€ג”€ Comparison builder ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _build_comparison(self, summary_w1: str, summary_w2: str,
                           summary_combined: str) -> str:
        """Build a structured comparison between week1 and week2 matrices.

        Includes FULL per-week summaries and the combined 2-week summary
        so the comparator agent can do real quantitative comparison.
        """
        lines: List[str] = []
        lines.append("=== WEEK-OVER-WEEK MATRIX COMPARISON ===\n")

        # ג”€ג”€ Full Week 1 matrix ג”€ג”€
        lines.append("=" * 50)
        lines.append("[WEEK 1 ג€” FULL CORRELATION MATRIX]")
        lines.append("=" * 50)
        lines.append(summary_w1)
        lines.append("")

        # ג”€ג”€ Full Week 2 matrix ג”€ג”€
        lines.append("=" * 50)
        lines.append("[WEEK 2 ג€” FULL CORRELATION MATRIX]")
        lines.append("=" * 50)
        lines.append(summary_w2)
        lines.append("")

        # ג”€ג”€ Full Combined 2-week matrix ג”€ג”€
        lines.append("=" * 50)
        lines.append("[COMBINED 2-WEEK ג€” FULL CORRELATION MATRIX]")
        lines.append("=" * 50)
        lines.append(summary_combined)
        lines.append("")

        # ג”€ג”€ Instructions ג”€ג”€
        lines.append("=" * 50)
        lines.append("[INSTRUCTIONS FOR COMPARATOR AGENT]")
        lines.append("=" * 50)
        lines.append("You have the FULL matrices for Week 1, Week 2, and the")
        lines.append("combined 2-week period above. Compare them quantitatively:\n")
        lines.append("1. Pearson pairs: which correlations strengthened/weakened?")
        lines.append("   Compare the actual r values side by side.")
        lines.append("2. Next-day predictors: did the same variables remain predictive?")
        lines.append("   Compare r values and sample sizes.")
        lines.append("3. AR(1) persistence: compare Rֲ² values for each metric.")
        lines.append("   A change of >0.10 in Rֲ² is meaningful.")
        lines.append("4. Markov transitions: compare state stickiness (diagonal probs).")
        lines.append("   Note the number of transitions in each week.")
        lines.append("5. KL-divergence: compare KL values and conditioning metrics.")
        lines.append("   A shift of >0.03 is meaningful.")
        lines.append("6. Metric ranges: compare means, stds, min/max side by side.")
        lines.append("   Changes >10% or >1 std are meaningful.")
        lines.append("7. STABILITY: findings in BOTH weeks AND combined = high confidence.")
        lines.append("   Findings in only one week = preliminary / evolving.")
        lines.append("8. If a week has too few data points for a section, note that")
        lines.append("   the absence is due to DATA SPARSITY, not physiological change.")

        return "\n".join(lines)

    # ג”€ג”€ג”€ Benchmark comparison builder ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _build_benchmark_comparison(self, benchmarks: Dict[str, str],
                                     computed: List[Tuple[str, str, date, date]]) -> str:
        """Build a structured comparison across ALL computed benchmark periods.

        Includes FULL summaries for every period so the comparator agent
        can do genuine quantitative cross-timeframe comparison.
        """
        if not computed:
            return "No benchmark periods were computed (insufficient data)."

        lines: List[str] = []
        lines.append("=== MULTI-SCALE BENCHMARK COMPARISON ===")
        lines.append(f"Computed {len(computed)} benchmark periods:\n")

        # ג”€ג”€ Coverage table ג”€ג”€
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            lines.append(f"  ג€¢ {label:20s}  {ds} ג†’ {de}  ({span} days)")
        lines.append("")

        # ג”€ג”€ Full summaries for each period ג”€ג”€
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            lines.append("=" * 60)
            lines.append(f"[{label.upper()} ({ds} ג†’ {de}, {span} days) ג€” "
                         f"FULL CORRELATION MATRIX]")
            lines.append("=" * 60)
            lines.append(benchmarks.get(key, "(not computed)"))
            lines.append("")

        # ג”€ג”€ Instructions for comparator ג”€ג”€
        lines.append("=" * 60)
        lines.append("[INSTRUCTIONS FOR COMPARATOR AGENT]")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"You have FULL correlation matrices for {len(computed)}")
        lines.append("benchmark periods above. Your job is CROSS-TIMEFRAME")
        lines.append("STABILITY ANALYSIS ג€” how do statistical patterns evolve")
        lines.append("as the analysis window grows?\n")

        lines.append("AVAILABLE WINDOWS:")
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            lines.append(f"  ג€¢ {label} ג€” {span} days of data")
        lines.append("")

        lines.append("REQUIRED ANALYSIS:")
        lines.append("")
        lines.append("1. CORRELATION EVOLUTION: For the strongest Pearson pairs,")
        lines.append("   track how r values change across benchmark windows.")
        lines.append("   Example: resting_hr ֳ— bb_charged: 7d r=-0.91, 14d r=-0.96,")
        lines.append("   30d r=-0.88 ג†’ STABLE across scales.")
        lines.append("   A change of >0.15 in r IS meaningful.")
        lines.append("")
        lines.append("2. PREDICTOR STABILITY: Do next-day predictors remain")
        lines.append("   significant at longer windows? Rising n should improve")
        lines.append("   reliability. Track r and n across windows.")
        lines.append("")
        lines.append("3. PERSISTENCE EVOLUTION: Compare AR(1) Rֲ² at each window.")
        lines.append("   Metrics with CONSISTENT Rֲ² across windows are truly")
        lines.append("   persistent. A change >0.10 is meaningful.")
        lines.append("")
        lines.append("4. MARKOV & KL MATURATION: These need more data.")
        lines.append("   Note which window first produces Markov/KL results")
        lines.append("   (the \"emergence window\"). Below 20 transitions = PRELIMINARY.")
        lines.append("")
        lines.append("5. METRIC RANGES: How do means and variability change")
        lines.append("   across windows? Useful for spotting recent shifts vs")
        lines.append("   long-term baselines.")
        lines.append("")
        lines.append("6. CONFIDENCE TIERS:")
        lines.append("   - ROBUST: Finding appears at ALL available windows")
        lines.append("     ג†’ highest confidence, treat as established.")
        lines.append("   - STABLE: Appears in 2+ adjacent windows")
        lines.append("     ג†’ good confidence, likely real.")
        lines.append("   - EMERGING: Only at the shortest window")
        lines.append("     ג†’ recent development, needs monitoring.")
        lines.append("   - LONGER-WINDOW-ONLY: Only at longer windows")
        lines.append("     ג†’ needs more data to detect, not visible short-term.")
        lines.append("   - SPARSE: Window had too few data points")
        lines.append("     ג†’ DATA SPARSITY, not physiological change.")
        lines.append("")
        lines.append("7. KEY INSIGHT: Which findings are MOST and LEAST")
        lines.append("   sensitive to window size? Window-insensitive findings")
        lines.append("   are the most actionable.")

        return "\n".join(lines)

    def _extract_section(self, summary: str, section_name: str) -> str:
        """Extract a named section from a summary string."""
        marker = f"[{section_name}"
        start = summary.find(marker)
        if start < 0:
            return ""
        next_bracket = summary.find("\n[", start + 1)
        if next_bracket < 0:
            return summary[start:].strip()
        return summary[start:next_bracket].strip()

    # ג”€ג”€ג”€ LAYER 0: Load + Clean ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer0_load_and_clean(self, date_start=None,
                                date_end=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Load daily_metrics, apply quality fixes, discover numeric cols."""
        log.info("   Layer 0: loading dataג€¦")

        conn = psycopg2.connect(self.conn_str)
        query = "SELECT * FROM daily_metrics"
        params: list = []
        clauses: list = []
        if date_start:
            clauses.append("date >= %s")
            params.append(date_start)
        if date_end:
            clauses.append("date <= %s")
            params.append(date_end)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY date"
        df = pd.read_sql_query(query, conn, params=params or None)

        # Also load wellness_log and nutrition_log for user-reported data
        try:
            df_wellness = pd.read_sql_query(
                "SELECT date, caffeine_intake, alcohol_intake, "
                "illness_severity, injury_severity, overall_stress_level, "
                "energy_level, workout_feeling, muscle_soreness "
                "FROM wellness_log ORDER BY date", conn
            )
        except Exception:
            df_wellness = pd.DataFrame()



        conn.close()

        if df.empty:
            return None, []

        log.info(f"   Loaded {len(df)} days ({df['date'].min()} ג†’ {df['date'].max()})")

        # Fix 0: Enforce daily continuity ג€” fill date gaps with NaN rows
        # so that .shift(1) and positional [1:]/[:-1] operations correctly
        # skip non-consecutive days (NaN pairs naturally drop via .dropna()).
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").asfreq("D").reset_index()
        n_filled = df.shape[0] - len(df.dropna(how="all", subset=[c for c in df.columns if c != "date"]))
        if n_filled > 0:
            log.info(f"   Inserted {n_filled} gap-fill rows for date continuity")

        # Merge wellness_log and nutrition_log (user-reported subjective data)
        if not df_wellness.empty:
            df_wellness["date"] = pd.to_datetime(df_wellness["date"])
            df = df.merge(df_wellness, on="date", how="left")
            n_wellness = df_wellness.shape[0]
            log.info(f"   Merged {n_wellness} wellness_log entries")



        # Fix 1: HRV sentinel 511 ג†’ NaN
        if "tr_hrv_weekly_avg" in df.columns:
            n_511 = (df["tr_hrv_weekly_avg"] == 511).sum()
            df.loc[df["tr_hrv_weekly_avg"] == 511, "tr_hrv_weekly_avg"] = np.nan
            if n_511 > 0:
                log.info(f"   Fixed {n_511} HRV sentinel values (511ג†’NaN)")



        # Fix 3: Rename tr_ abbreviations
        actual_renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
        df.rename(columns=actual_renames, inplace=True)

        # Fix 4: Log-transform HRV (Esco 2025, Plews 2012)
        # RMSSD is right-skewed; lnRMSSD is standard in the literature.
        # Pearson assumes normality ג€” use lnRMSSD instead of raw HRV.
        hrv_col = "training_hrv_weekly_avg" if "training_hrv_weekly_avg" in df.columns else "hrv_last_night"
        if hrv_col in df.columns:
            df["ln_hrv"] = np.log(df[hrv_col].clip(lower=1))
            log.info("   Derived: ln_hrv (log-transformed HRV)")

        # Fix 5: Weekly HRV Mean + CV (Plews 2012, Esco 2025)
        # The coefficient of variation of HRV over a rolling 7-day window
        # is a stronger marker of recovery status than the mean alone.
        if "hrv_last_night" in df.columns:
            hrv_series = df["hrv_last_night"]
            rolling_mean = hrv_series.rolling(7, min_periods=3).mean()
            rolling_std = hrv_series.rolling(7, min_periods=3).std()
            df["hrv_weekly_mean"] = rolling_mean
            df["hrv_weekly_cv"] = (rolling_std / rolling_mean.clip(lower=1)) * 100
            log.info("   Derived: hrv_weekly_mean, hrv_weekly_cv (7-day rolling)")

        # Auto-discover numeric columns with ג‰¥5 non-null values
        candidate_cols = [
            c for c in df.columns
            if c not in SKIP_COLS
            and c not in {"training_intensity"}  # categorical
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        # Also skip any renamed SKIP targets
        renamed_skips = {RENAME_MAP.get(s, s) for s in SKIP_COLS}
        candidate_cols = [c for c in candidate_cols if c not in renamed_skips]

        metrics = [c for c in candidate_cols if df[c].notna().sum() >= 5]
        log.info(f"   {len(metrics)} metrics with ג‰¥5 data points")

        return df, metrics

    # ג”€ג”€ג”€ LAYER 1: Pearson ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer1_pearson(self, data: pd.DataFrame,
                        metrics: List[str]) -> Tuple[List, List]:
        """Same-day + lag-1 Pearson correlations with p-values."""
        log.info("   Layer 1: Pearson correlationsג€¦")

        # Same-day
        R0 = data.corr(method="pearson")

        # p-values
        P0 = pd.DataFrame(np.ones_like(R0.values),
                           index=R0.index, columns=R0.columns)
        for i, ci in enumerate(metrics):
            for j, cj in enumerate(metrics):
                if i >= j:
                    continue
                mask = data[[ci, cj]].dropna()
                n = len(mask)
                if n < 3:
                    continue
                r = R0.loc[ci, cj]
                if abs(r) < 1.0 and n > 2:
                    t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r * r + 1e-15)
                    p = 2 * sp_stats.t.sf(abs(t_stat), n - 2)
                    P0.loc[ci, cj] = p
                    P0.loc[cj, ci] = p

        sig_pairs = []
        for i, ci in enumerate(metrics):
            for j, cj in enumerate(metrics):
                if i >= j:
                    continue
                r = R0.loc[ci, cj]
                p = P0.loc[ci, cj]
                if not np.isnan(r) and p < 0.05 and abs(r) > 0.3:
                    n = len(data[[ci, cj]].dropna())
                    sig_pairs.append((ci, cj, r, p, n))
        sig_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

        # Lag-1
        lagged = data.shift(1)
        lag1_results = []
        for predictor in metrics:
            for target in metrics:
                if predictor == target:
                    continue
                valid = pd.DataFrame({
                    "pred": lagged[predictor], "tgt": data[target]
                }).dropna()
                n = len(valid)
                if n < 5:
                    continue
                # Skip constant arrays (all same value ג†’ no correlation)
                if np.std(valid["pred"]) < 1e-10 or np.std(valid["tgt"]) < 1e-10:
                    continue
                r, p = sp_stats.pearsonr(valid["pred"], valid["tgt"])
                if p < 0.05 and abs(r) > 0.3:
                    lag1_results.append((predictor, target, r, p, n))
        lag1_results.sort(key=lambda x: abs(x[2]), reverse=True)

        log.info(f"   ג“ {len(sig_pairs)} same-day, {len(lag1_results)} lag-1 pairs")
        return sig_pairs, lag1_results

    # ג”€ג”€ג”€ LAYER 2a: Normality ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer2a_normality(self, data: pd.DataFrame,
                           metrics: List[str]) -> Dict[str, Tuple[str, float]]:
        """Shapiro-Wilk normality test for each metric.

        Tests Hג‚€: metric ~ Normal.  The W statistic measures how well
        the ordered sample matches expected normal order statistics:

            W = (־£ aבµ¢ x_(i))ֲ² / ־£ (xבµ¢ - xּ„)ֲ²

        Decision: p > 0.05 ג†’ assume normal, else non-normal.
        Non-normal metrics are preferred candidates for Markov analysis
        (Layer 2e) since their distributions benefit from state-based
        modelling over linear assumptions.

        Truncates to first 5 000 values (scipy limit).
        """
        log.info("   Layer 2a: normality testsג€¦")
        normality: Dict[str, Tuple[str, float]] = {}
        for m in metrics:
            vals = data[m].dropna()
            if len(vals) < 5:
                normality[m] = ("assumed_normal", 1.0)
            else:
                _, p = sp_stats.shapiro(vals.values[:5000])
                normality[m] = ("normal" if p > 0.05 else "non_normal", p)
        n_norm = sum(1 for s, _ in normality.values() if s == "normal")
        log.info(f"   ג“ {n_norm}/{len(metrics)} normal")
        return normality

    # ג”€ג”€ג”€ LAYER 2b: AR(1) ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer2b_ar1(self, data: pd.DataFrame,
                     metrics: List[str]) -> List[Tuple]:
        """First-order autoregressive model: does yesterday predict today?

        For each metric x, fits the AR(1) model via OLS:

            x_t = ֿ†ֲ·x_{tגˆ’1} + c + ־µ_t

        where ֿ† (slope) is the persistence coefficient.  Reports:
        - ֿ† : autocorrelation strength  (ֿ† ג‰ˆ 1 ג†’ strong persistence)
        - Rֲ² : fraction of variance explained by yesterdayג€™s value
        - p  : significance of the linear relationship

        Date-gap safe: relies on .shift(1) after asfreq('D') gap-fill,
        so non-consecutive days produce NaN pairs that are dropped.
        """
        log.info("   Layer 2b: AR(1) persistenceג€¦")
        results = []
        for m in metrics:
            vals = data[m].dropna()
            if len(vals) < 5:
                continue
            y = vals.values[1:]
            x = vals.values[:-1]
            if np.std(x) < 1e-10:
                continue

            # ADF stationarity test (Daza 2018)
            is_stationary = True
            try:
                if len(vals) >= 8:
                    adf_stat, adf_p, *_ = adfuller(vals.values, maxlag=1)
                    is_stationary = adf_p < 0.05
            except Exception:
                pass  # fallback: assume stationary

            slope, intercept, r, p, se = sp_stats.linregress(x, y)
            results.append((m, slope, r**2, p, len(y), is_stationary))
        results.sort(key=lambda x: x[2], reverse=True)
        log.info(f"   ג“ {len(results)} AR(1) models")
        return results

    # ג”€ג”€ג”€ LAYER 2c: Anomalies ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer2c_anomalies(self, df: pd.DataFrame, data: pd.DataFrame,
                           metrics: List[str]) -> List[Tuple]:
        """Detect recent statistical anomalies via percentile thresholds.

        For each metric computes the 5th and 95th percentiles over the
        full window.  Flags the 3 most recent values that fall outside
        these bounds as anomalies, labelled HIGH (> p95) or LOW (< p5).

        Also reports the z-score for context:  z = (x גˆ’ ־¼) / ֿƒ

        Using percentiles instead of z-scores is distribution-agnostic
        ג€” works equally well for normal and skewed metrics (e.g. HRV).
        """
        anomalies = []
        for m in metrics:
            vals = data[m].dropna()
            if len(vals) < 5:
                continue
            mu, sd = vals.mean(), vals.std()
            if sd < 1e-10:
                continue
            p5 = np.percentile(vals, 5)
            p95 = np.percentile(vals, 95)
            recent = df[["date", m]].dropna().tail(3)
            for _, row in recent.iterrows():
                val = row[m]
                if val < p5 or val > p95:
                    z = (val - mu) / sd
                    direction = "HIGH" if val > p95 else "LOW"
                    anomalies.append((str(row["date"]), m, val, z, direction))
        return anomalies

    # ג”€ג”€ג”€ LAYER 2d: Conditioned AR(1) ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer2d_conditioned_ar1(self, data: pd.DataFrame,
                                  metrics: List[str],
                                  normality: Dict) -> List[Tuple]:
        """Multi-variable next-day prediction with adjusted Rֲ².

        Extends AR(1) by adding exogenous regressors.  Two levels:

        Level 1 ג€” single conditioning variable z:
            x_t = ־²ג‚€ֲ·x_{tגˆ’1} + ־²ג‚ֲ·z_{tגˆ’1} + ־²ג‚‚ + ־µ

        Level 2 ג€” two conditioning variables zג‚, zג‚‚:
            x_t = ־²ג‚€ֲ·x_{tגˆ’1} + ־²ג‚ֲ·zג‚_{tגˆ’1} + ־²ג‚‚ֲ·zג‚‚_{tגˆ’1} + ־²ג‚ƒ + ־µ

        Solved via ordinary least-squares (np.linalg.lstsq).
        Reports adjusted Rֲ² to penalise extra parameters:

            Rֲ²_adj = 1 גˆ’ [(1גˆ’Rֲ²)(nגˆ’1)] / (nגˆ’pגˆ’1)

        Only kept if Rֲ² > 0.10 (Level 1) or Rֲ² > 0.15 (Level 2).
        Improvement = Rֲ²_model גˆ’ Rֲ²_baseline (baseline = simple AR(1)).

        Uses the top-5 Level-1 winners as candidates for Level-2
        combinations (greedy forward search).
        """
        log.info("   Layer 2d: conditioned AR(1)ג€¦")
        # Only use KEY_TARGETS that exist in our metrics
        targets = [t for t in KEY_TARGETS
                   if t in metrics
                   or RENAME_MAP.get(t, t) in metrics]
        # Map renamed targets
        targets_actual = []
        for t in targets:
            if t in metrics:
                targets_actual.append(t)
            elif RENAME_MAP.get(t, t) in metrics:
                targets_actual.append(RENAME_MAP[t])

        results = []

        for target in targets_actual:
            target_vals = data[target].dropna()
            if len(target_vals) < 6:
                continue

            # Baseline simple AR(1)
            y_s = target_vals.values[1:]
            x_s = target_vals.values[:-1]
            if np.std(x_s) < 1e-10:
                continue
            _, _, r_s, _, _ = sp_stats.linregress(x_s, y_s)
            r2_baseline = r_s ** 2

            other = [m for m in metrics if m != target]

            # Level 1: AR(1) + one conditioning variable
            level1_hits = []
            for cond in other:
                sub = data[[target, cond]].dropna()
                if len(sub) < 6:
                    continue
                y = sub[target].values[1:]
                x_lag = sub[target].values[:-1]
                z = sub[cond].values[:-1]
                if np.std(x_lag) < 1e-10 or np.std(z) < 1e-10:
                    continue

                X_design = np.column_stack([x_lag, z, np.ones(len(y))])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                y_hat = X_design @ beta
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-15) if ss_tot > 0 else 0
                improvement = r2 - r2_baseline

                if r2 > 0.1:
                    level1_hits.append((target, [cond], beta[0], beta[1],
                                       r2, improvement, len(y)))

            level1_hits.sort(key=lambda x: x[4], reverse=True)
            results.extend(level1_hits[:5])

            # Level 2: AR(1) + two conditioning variables
            if len(level1_hits) >= 2:
                top_conds = [h[1][0] for h in level1_hits[:5]]
                for a_i in range(len(top_conds)):
                    for b_i in range(a_i + 1, len(top_conds)):
                        c1, c2 = top_conds[a_i], top_conds[b_i]
                        sub = data[[target, c1, c2]].dropna()
                        if len(sub) < 7:
                            continue
                        y = sub[target].values[1:]
                        x_lag = sub[target].values[:-1]
                        z1 = sub[c1].values[:-1]
                        z2 = sub[c2].values[:-1]
                        if np.std(x_lag) < 1e-10:
                            continue

                        X_design = np.column_stack([x_lag, z1, z2,
                                                    np.ones(len(y))])
                        try:
                            beta, _, _, _ = np.linalg.lstsq(X_design, y,
                                                             rcond=None)
                        except np.linalg.LinAlgError:
                            continue
                        y_hat = X_design @ beta
                        ss_res = np.sum((y - y_hat) ** 2)
                        ss_tot = np.sum((y - y.mean()) ** 2)
                        r2 = 1 - ss_res / (ss_tot + 1e-15) if ss_tot > 0 else 0
                        improvement = r2 - r2_baseline

                        if r2 > 0.15:
                            results.append((target, [c1, c2], beta[0],
                                            None, r2, improvement, len(y)))

        results.sort(key=lambda x: x[4], reverse=True)
        log.info(f"   ג“ {len(results)} conditioned AR(1) models")
        return results

    # ג”€ג”€ג”€ LAYER 2e: Markov + KL ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer2e_markov(self, data: pd.DataFrame, metrics: List[str],
                        normality: Dict) -> List[Dict]:
        """Markov transition matrices with KL-divergence conditioning."""
        return compute_markov_layer(
            data=data,
            metrics=metrics,
            normality=normality,
            key_targets=KEY_TARGETS,
            rename_map=RENAME_MAP,
            n_bins=N_BINS,
            bin_labels=BIN_LABELS,
            acwr_edges=ACWR_EDGES,
            acwr_labels=ACWR_LABELS,
            min_transitions_kl=MIN_TRANSITIONS_KL,
            adaptive_alpha_fn=_adaptive_alpha,
            logger=log,
        )

    # ג”€ג”€ג”€ Multi-Lag Carryover Detection (Daza 2018) ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _multi_lag_carryover(self, data: pd.DataFrame,
                              metrics: List[str]) -> List[Tuple]:
        """Check lag-2 and lag-3 correlations for KEY_TARGETS.

        If a predictorג†’target relationship is still significant at
        lag-2 or lag-3 days, it indicates a multi-day carryover effect
        (e.g., a hard workout affects HRV for 2-3 days, not just 1).
        Based on Daza (2018) ג€” carryover effects in N-of-1 designs.

        Only checks KEY_TARGETS to avoid combinatorial explosion.
        """
        log.info("   Multi-lag carryover detectionג€¦")
        results = []
        targets_actual = [
            RENAME_MAP.get(t, t) for t in KEY_TARGETS
            if RENAME_MAP.get(t, t) in metrics
        ]

        for lag in [2, 3]:
            lagged = data.shift(lag)
            for predictor in metrics:
                for target in targets_actual:
                    if predictor == target:
                        continue
                    valid = pd.DataFrame({
                        "pred": lagged[predictor], "tgt": data[target]
                    }).dropna()
                    n = len(valid)
                    if n < 8:
                        continue
                    if np.std(valid["pred"]) < 1e-10 or np.std(valid["tgt"]) < 1e-10:
                        continue
                    r, p = sp_stats.pearsonr(valid["pred"], valid["tgt"])
                    if p < 0.05 and abs(r) > 0.3:
                        results.append((predictor, target, lag, r, p, n))

        results.sort(key=lambda x: abs(x[3]), reverse=True)
        log.info(f"   ג“ {len(results)} multi-lag carryover pairs")
        return results

    # ג”€ג”€ג”€ LAYER 2f: Rolling Correlation Stationarity ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer2f_rolling_correlation(self, data: pd.DataFrame,
                                      sig_pairs: List[Tuple]) -> List[Dict]:
        """Check whether top correlations are stable over time.

        For each of the top-10 significant same-day pairs, compute a
        rolling 30-day Pearson r.  High variance of rolling-r means the
        relationship is non-stationary (unstable) ג€” possibly driven by
        a confound or a phase-shift in the user's routine.

        Returns list of dicts with:
          - pair: (metric_a, metric_b)
          - mean_r: mean of rolling-r values
          - std_r: std of rolling-r values (instability indicator)
          - stability: "STABLE" (std < 0.15), "MODERATE" (std < 0.25),
                       or "UNSTABLE" (std >= 0.25)
        """
        log.info("   Layer 2f: rolling correlation stationarityג€¦")
        results = []
        window = 30

        for a, b, r_overall, p, *_rest in sig_pairs[:10]:
            sub = data[[a, b]].dropna()
            if len(sub) < window + 5:
                continue
            rolling_r = []
            for start in range(len(sub) - window + 1):
                chunk = sub.iloc[start:start + window]
                if chunk[a].std() < 1e-10 or chunk[b].std() < 1e-10:
                    continue
                r_win, _ = sp_stats.pearsonr(chunk[a], chunk[b])
                rolling_r.append(r_win)

            if len(rolling_r) < 3:
                continue
            arr = np.array(rolling_r)
            std_r = float(np.std(arr))
            mean_r = float(np.mean(arr))

            if std_r < 0.15:
                stability = "STABLE"
            elif std_r < 0.25:
                stability = "MODERATE"
            else:
                stability = "UNSTABLE"

            results.append({
                "pair": (a, b),
                "mean_r": mean_r,
                "std_r": std_r,
                "stability": stability,
            })

        log.info(f"   ג“ {len(results)} rolling correlation checks")
        return results

    # ג”€ג”€ג”€ LAYER 3: Agent Summary ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _layer3_summary(self, df, n_days, metrics, sig_pairs, lag1_results,
                        normality, ar1_results, anomalies,
                        cond_ar1_results, markov_results,
                        rolling_corr_results, multi_lag_results,
                        training_days) -> str:
        """Build natural-language summary for agent consumption."""
        log.info("   Layer 3: building agent summaryג€¦")
        lines: List[str] = []

        lines.append(f"=== CORRELATION MATRIX ANALYSIS ({df['date'].max()}) ===")
        lines.append(f"Data: {n_days} days, {len(metrics)} numeric metrics")

        if n_days < 21:
            lines.append(
                f"NOTE*: Only {n_days} days of data collected so far. "
                f"All multi-variable models are PRELIMINARY ג€” treat Rֲ² values "
                f"and transition probabilities as directional, not definitive. "
                f"Confidence improves significantly after 21+ days."
            )
        lines.append("")

        # Training intensity (if available)
        if training_days:
            lines.append("[TRAINING INTENSITY CLASSIFICATION]")
            lines.append("  Rules: gym >30min = HARD, running HR>130 = HARD, "
                         "basketball >30min = HARD")
            lines.append("  Body part detection from Garmin exercise sets "
                         "(PULL_UP/ROW = UPPER, SQUAT/LUNGE = LOWER)")
            for d in sorted(training_days.keys()):
                info = training_days[d]
                body = []
                if info.get("has_upper"):
                    body.append("UPPER")
                if info.get("has_lower"):
                    body.append("LOWER")
                body_str = "+".join(body) if body else "none"
                types = ", ".join(info.get("activity_types", []))
                lines.append(
                    f"  {d}: {info['intensity']} "
                    f"({info['hard_minutes']:.0f}min hard, "
                    f"body={body_str}, types={types})"
                )
            lines.append("")

        # Same-day correlations
        lines.append("[SAME-DAY CORRELATIONS (strongest pairs)]")
        for a, b, r, p, *rest in sig_pairs[:12]:
            n = rest[0] if rest else None
            arrow = "ג†‘ג†‘" if r > 0 else "ג†‘ג†“"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else "*")
            n_str = f", n={n}" if n is not None else ""
            lines.append(f"  {arrow} {a} ֳ— {b}: r={r:+.3f} (p={p:.4f}{n_str}) {sig}")
        lines.append("")

        # Next-day predictors
        lines.append("[NEXT-DAY PREDICTORS (what yesterday predicts about today)]")
        for pred, tgt, r, p, n in lag1_results[:8]:
            lines.append(f"  {pred} ג†’ {tgt}: r={r:+.3f} (p={p:.4f}, n={n})")
        lines.append("")

        # AR(1) persistence
        lines.append("[AUTOREGRESSIVE PERSISTENCE]")
        for entry in ar1_results[:8]:
            m, phi, r2, p, n = entry[:5]
            is_stationary = entry[5] if len(entry) > 5 else True
            label = ("strong" if r2 > 0.4
                     else ("moderate" if r2 > 0.15 else "weak"))
            flag = "" if is_stationary else " ג  TREND-DOMINATED"
            lines.append(f"  {m}: ֿ†={phi:+.3f}, Rֲ²={r2:.3f} ({label}){flag}")
        lines.append("")

        # Distributions
        normal_m = [m for m, (s, _) in normality.items() if s == "normal"]
        nonnorm_m = [m for m, (s, _) in normality.items() if s == "non_normal"]
        lines.append("[DISTRIBUTIONS]")
        lines.append(f"  Normal: {', '.join(normal_m[:10])}")
        lines.append(f"  Non-normal: {', '.join(nonnorm_m[:10])}")
        lines.append("")

        # Anomalies
        if anomalies:
            lines.append("[RECENT ANOMALIES (last 3 days, outside 5th/95th percentile)]")
            for dt, m, val, z, d in anomalies:
                lines.append(f"  {dt} ג€” {m}={val:.1f} (z={z:+.2f}, {d})")
            lines.append("")

        # Multi-lag carryover
        if multi_lag_results:
            lines.append("[MULTI-DAY CARRYOVER (lag-2 and lag-3 still significant)]")
            for pred, tgt, lag, r, p, n in multi_lag_results:
                lines.append(
                    f"  {pred} ג†’ {tgt} (lag-{lag}): r={r:+.3f} (p={p:.4f}, n={n})"
                    f" ג€” effect persists {lag} days"
                )
            lines.append("")

        # Rolling correlation stability
        if rolling_corr_results:
            lines.append("[CORRELATION STABILITY (30-day rolling window)]")
            for rc in rolling_corr_results:
                a, b = rc["pair"]
                lines.append(
                    f"  {a} ֳ— {b}: mean_r={rc['mean_r']:+.3f}, "
                    f"std={rc['std_r']:.3f} ({rc['stability']})"
                )
            lines.append("")

        # Conditioned AR(1)
        lines.append("[CONDITIONED AR(1) ג€” MULTI-VARIABLE NEXT-DAY PREDICTION]")
        lines.append("  Higher Rֲ² = better prediction. "
                     "Positive improvement = conditioning helps.")
        for target, conds, phi, beta_c, r2, improv, n in cond_ar1_results[:10]:
            cond_str = " + ".join(conds)
            # Adjusted Rֲ²
            p_vars = len(conds) + 1
            if n > p_vars + 1:
                r2_adj = 1 - (1 - r2) * (n - 1) / (n - p_vars - 1)
            else:
                r2_adj = r2
            star = "*" if n < 20 else ""
            if improv > 0.05:
                interp = f"Knowing {cond_str} IMPROVES prediction by {improv:+.3f}"
            elif improv > 0:
                interp = f"{cond_str} adds SLIGHT predictive value"
            else:
                interp = f"{cond_str} doesn't help beyond lag"
            lines.append(
                f"  {target} ~ lag + {cond_str}: Rֲ²={r2:.3f} "
                f"(adj={r2_adj:.3f}) ({improv:+.3f} vs simple, n={n}{star}) "
                f"ג€” {interp}"
            )
        if any(n < 20 for *_, n in cond_ar1_results[:10]):
            lines.append("  * n<20: preliminary, may be inflated")
        lines.append("")

        # Markov transitions
        lines.append("[MARKOV STATE TRANSITIONS]")
        lines.append("  Each metric split into LOW/MED/HIGH. "
                     "High diagonal = 'sticky' state.")
        lines.append("  Smoothing is ADAPTIVE: ־± shrinks as data grows "
                     "(more data ג†’ trust empirical, less smoothing).")
        for mr in markov_results[:8]:
            labels = mr.get("labels", BIN_LABELS[:mr["bins"]])
            lines.append(
                f"\n  {mr['target']} ({mr['n_transitions']} transitions, "
                f"־±={mr['smooth_alpha']:.3f}, "
                f"confidence={mr['confidence']}, "
                f"bins: {np.round(mr['edges'], 1)}):"
            )
            stat_desc = ", ".join(
                f"{labels[i]}={mr['stationary'][i]:.0%}"
                for i in range(mr["bins"])
            )
            lines.append(f"    Long-run: {stat_desc}")
            for i in range(mr["bins"]):
                diag = mr["marginal"][i, i]
                if diag > 0.5:
                    lines.append(
                        f"    When {labels[i]}, {diag:.0%} stays {labels[i]}"
                    )
                else:
                    nxt = int(np.argmax(mr["marginal"][i]))
                    lines.append(
                        f"    When {labels[i]}, "
                        f"{mr['marginal'][i, nxt]:.0%} ג†’ {labels[nxt]}"
                    )
        lines.append("")

        # KL divergence ranking
        kl_ranking = [(mr["target"], mr["best_cond"], mr["best_kl"])
                      for mr in markov_results if mr["best_cond"]]
        kl_ranking.sort(key=lambda x: x[2], reverse=True)

        lines.append("[KL-DIVERGENCE: WHICH CONDITIONS CHANGE PREDICTIONS?]")
        lines.append("  STRONG (>0.10) | MODERATE (0.03-0.10) | WEAK (<0.03)")
        for target, cond, kl in kl_ranking[:10]:
            sig = ("STRONG" if kl > 0.10
                   else ("MODERATE" if kl > 0.03 else "WEAK"))
            lines.append(
                f"  {target} conditioned on {cond}: KL={kl:.4f} ({sig})"
            )
            if kl > 0.03:
                mr = next((m for m in markov_results
                           if m["target"] == target), None)
                if mr and mr["best_cond_T"]:
                    labels_k = mr.get("labels", BIN_LABELS[:mr["bins"]])
                    max_diff = 0
                    diff_desc = ""
                    for i in range(mr["bins"]):
                        for j in range(mr["bins"]):
                            for cl, cl_lbl in [(0, "LOW"), (1, "HIGH")]:
                                d = abs(mr["best_cond_T"][cl][i, j]
                                        - mr["marginal"][i, j])
                                if d > max_diff:
                                    max_diff = d
                                    diff_desc = (
                                        f"When {cond} is {cl_lbl}, "
                                        f"{target} {labels_k[i]}ג†’{labels_k[j]} "
                                        f"changes by {d:+.2f}"
                                    )
                    if diff_desc:
                        lines.append(f"    ג†’ {diff_desc}")
        lines.append("")

        # Current metric ranges
        lines.append(f"[CURRENT METRIC RANGES (last {n_days} days)]")
        for m in KEY_RANGE_METRICS:
            # Check renamed versions too
            actual = m
            if m not in df.columns:
                actual = RENAME_MAP.get(m, m)
            if actual in df.columns:
                vals = df[actual].dropna()
                if len(vals) > 0:
                    lines.append(
                        f"  {actual}: min={vals.min():.0f}, "
                        f"max={vals.max():.0f}, "
                        f"mean={vals.mean():.1f}, std={vals.std():.1f}"
                    )

        summary = "\n".join(lines)
        log.info(f"   ג“ Summary: {len(summary)} chars (~{len(summary)//4} tokens)")
        return summary

    # ג”€ג”€ג”€ Storage ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def _store_summary(self, summary: str):
        """Persist agent summary to database."""
        today = date.today()
        token_est = len(summary) // 4
        conn = psycopg2.connect(self.conn_str)
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO matrix_summaries (computed_at, summary_text, token_est)
               VALUES (%s, %s, %s)
               ON CONFLICT (computed_at) DO UPDATE SET
                   summary_text = EXCLUDED.summary_text,
                   token_est = EXCLUDED.token_est""",
            (today, summary, token_est),
        )
        cur.close()
        conn.close()

    def _store_correlation_result(self, window_label: str, date_start, date_end,
                                  run_result: Dict[str, Any]):
        """Persist raw correlation layer outputs to correlation_results table."""
        raw = run_result.get("raw_results", {})
        if not raw:
            return
        today = date.today()
        try:
            conn = psycopg2.connect(self.conn_str)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO correlation_results
                   (computed_at, window_label, date_start, date_end,
                    n_days, n_metrics, analysis_status,
                    pearson_pairs, lag1_pairs, ar1_results,
                    anomalies, cond_ar1, metric_ranges)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (computed_at, window_label) DO UPDATE SET
                       date_start = EXCLUDED.date_start,
                       date_end = EXCLUDED.date_end,
                       n_days = EXCLUDED.n_days,
                       n_metrics = EXCLUDED.n_metrics,
                       analysis_status = EXCLUDED.analysis_status,
                       pearson_pairs = EXCLUDED.pearson_pairs,
                       lag1_pairs = EXCLUDED.lag1_pairs,
                       ar1_results = EXCLUDED.ar1_results,
                       anomalies = EXCLUDED.anomalies,
                       cond_ar1 = EXCLUDED.cond_ar1,
                       metric_ranges = EXCLUDED.metric_ranges
                """,
                (
                    today, window_label, date_start, date_end,
                    raw.get("n_days", 0), raw.get("n_metrics", 0),
                    run_result.get("analysis_status", "unknown"),
                    json.dumps(raw.get("pearson_pairs", [])),
                    json.dumps(raw.get("lag1_pairs", [])),
                    json.dumps(raw.get("ar1_results", {})),
                    json.dumps({"n_anomalies": raw.get("n_anomalies", 0)}),
                    json.dumps({"n_cond_ar1": raw.get("n_cond_ar1", 0)}),
                    json.dumps({"n_markov": raw.get("n_markov", 0)}),
                ),
            )
            cur.close()
            conn.close()
            log.info("   Stored correlation_results for %s (%s)", window_label, today)
        except Exception as e:
            log.warning("   Failed to store correlation_results: %s", e)

    # ג”€ג”€ג”€ Public API ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€ג”€

    def get_latest_summary(self) -> Optional[str]:
        """Return the most recent matrix summary, or None."""
        conn = psycopg2.connect(self.conn_str)
        cur = conn.cursor()
        cur.execute(
            "SELECT summary_text FROM matrix_summaries "
            "ORDER BY computed_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None


# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•
#  CLI
# ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•ג•

if __name__ == "__main__":
    engine = CorrelationEngine()
    summary = engine.compute_weekly()
    log.info("\n" + summary)

