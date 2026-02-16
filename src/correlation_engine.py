"""
Correlation Matrix Engine — Production Version
================================================
Pre-computes statistical matrices from daily_metrics for AI agent
consumption and trend analysis.

Architecture (4 layers):
  Layer 0 — Data loading + cleaning:  Auto-discover numeric columns,
            rename tr_ abbreviations, fix sentinels, add training intensity.
  Layer 1 — Pearson:  NxN same-day + lag-1 correlations with p-values.
  Layer 2 — Conditional:
            AR(1) for all metrics, conditioned AR(1) with top predictors,
            Markov transition matrices for non-normal metrics,
            KL-divergence ranking.
  Layer 3 — Agent Summary:  ~2-3 KB natural-language digest that replaces
            raw SQL in agent prompts.

Key mathematical fixes (proven in test_matrices.py):
  • Marginal matrix uses ADAPTIVE kernel smoothing (α = BASE/√n, clamped
    to [0.02, 0.25]) — heavy smoothing for sparse data, light for large n.
  • KL-divergence only computed when BOTH conditioning levels have ≥5
    transitions — prevents noise from dominating the divergence ranking.
  • Adjusted R² is required — raw R² with n≈10, p=3 is meaningless.
  • HRV sentinel 511 → NaN.
  • Data quality disclaimer when n < 20.
  • Confidence tiers on Markov: HIGH (≥100), GOOD (≥30), MODERATE (≥15),
    PRELIMINARY (<15).
"""

from __future__ import annotations

import os
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

load_dotenv()

log = logging.getLogger("correlation_engine")


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Columns to skip when auto-discovering numeric metrics
SKIP_COLS = {
    "created_at", "updated_at", "date",
    "sleep_score_qualifier", "hrv_status", "tr_level",
    "sleep_start_local", "sleep_end_local",
    # Garbage columns (always 0 or garbage)
    "hydration_goal_ml", "sweat_loss_ml",
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
    "deep_sleep_sec", "rem_sleep_sec", "hydration_value_ml",
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
    # Nutrition (nutrition_log aggregated daily)
    "total_calories", "total_protein", "total_carbs", "total_fat",
]

# Markov discretisation
N_BINS = 3
BIN_LABELS = ["LOW", "MED", "HIGH"]

# ACWR evidence-based thresholds (Gabbett 2016, BJSM, 2597 citations)
# Used instead of percentile bins for the acwr metric specifically.
ACWR_EDGES = [0, 0.8, 1.3, 1.5, 999]
ACWR_LABELS = ["UNDERTRAINED", "SWEET_SPOT", "HIGH_RISK", "DANGER"]

# Smoothing alpha for Markov transitions (must be same for marginal & conditional)
# Now adaptive: α = max(SMOOTH_ALPHA_MIN, SMOOTH_ALPHA_BASE / sqrt(n_transitions))
# This gives more smoothing when data is sparse and lets empirical
# distributions dominate as the dataset grows (neural-net flavoured
# kernel smoothing).
SMOOTH_ALPHA_BASE = 0.50   # numerator for α = base / √n
SMOOTH_ALPHA_MIN  = 0.02   # floor — always keep a tiny uniform mix
SMOOTH_ALPHA_MAX  = 0.25   # ceiling — cap smoothing for very small n

# Minimum transitions per conditioning level for KL to be computed
MIN_TRANSITIONS_KL = 5


def _adaptive_alpha(n_transitions: int) -> float:
    """Compute kernel smoothing strength from sample size.

    α = clamp(BASE / √n, MIN, MAX)

    With 4 transitions  → α ≈ 0.25 (heavy smoothing, sparse data)
    With 10 transitions → α ≈ 0.16
    With 50 transitions → α ≈ 0.07
    With 200+           → α ≈ 0.04 (light smoothing, data speaks)
    """
    if n_transitions <= 0:
        return SMOOTH_ALPHA_MAX
    raw = SMOOTH_ALPHA_BASE / math.sqrt(n_transitions)
    return max(SMOOTH_ALPHA_MIN, min(SMOOTH_ALPHA_MAX, raw))

# Benchmark periods: (key, label, days)
# days=None → "current week" = Monday of this week → ref_date
# All others are rolling windows: ref_date − (days−1) → ref_date
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


# ═══════════════════════════════════════════════════════════════
#  SCHEMA — tables for stored results
# ═══════════════════════════════════════════════════════════════

CORRELATION_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS matrix_summaries (
    id           SERIAL PRIMARY KEY,
    computed_at  DATE NOT NULL UNIQUE,
    summary_text TEXT NOT NULL,
    token_est    INTEGER NOT NULL
);
"""


# ═══════════════════════════════════════════════════════════════
#  ENGINE
# ═══════════════════════════════════════════════════════════════

class CorrelationEngine:
    """
    Orchestrates all four layers of correlation computation.
    Standalone — uses psycopg2 directly, no external DatabaseManager.
    """

    def __init__(self, conn_str: Optional[str] = None):
        self.conn_str = conn_str or os.getenv("POSTGRES_CONNECTION_STRING", "")

    # ─── Schema bootstrap ─────────────────────────────────────

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

    # ─── MAIN ENTRY ───────────────────────────────────────────

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
        log.info("\n📐 Correlation Engine — computing weekly matrices…")
        self.bootstrap_schema()
        summary = self._compute_raw(training_days)
        if summary:
            self._store_summary(summary)
            log.info("✅ Correlation engine complete\n")
        return summary

    # ─── Raw computation (no storage) ─────────────────────────

    def _compute_raw(self, training_days: Optional[Dict] = None,
                     date_start=None, date_end=None) -> str:
        """Run layers 0-3 for a date range and return the summary string."""
        df, metrics = self._layer0_load_and_clean(date_start=date_start,
                                                   date_end=date_end)
        if df is None or len(df) < 5:
            msg = "Not enough data for correlation analysis (need ≥ 5 days)."
            log.info(f"   ⚠️  {msg}")
            return msg

        n_days = len(df)

        if training_days:
            min_date = df["date"].min()
            max_date = df["date"].max()
            td_filtered = {d: v for d, v in training_days.items()
                           if min_date <= d <= max_date}
            df["training_hard_minutes"] = df["date"].apply(
                lambda d: td_filtered.get(d, {}).get("hard_minutes", 0)
            )
            df["training_has_upper"] = df["date"].apply(
                lambda d: 1 if td_filtered.get(d, {}).get("has_upper", False) else 0
            )
            df["training_has_lower"] = df["date"].apply(
                lambda d: 1 if td_filtered.get(d, {}).get("has_lower", False) else 0
            )
            for tc in ["training_hard_minutes", "training_has_upper",
                       "training_has_lower"]:
                if df[tc].notna().sum() >= 5 and tc not in metrics:
                    metrics.append(tc)
            log.info(f"   Added training columns → {len(metrics)} total metrics")
            training_for_summary = td_filtered
        else:
            training_for_summary = None

        data = df[metrics].copy()

        sig_pairs, lag1_results = self._layer1_pearson(data, metrics)
        normality = self._layer2a_normality(data, metrics)
        ar1_results = self._layer2b_ar1(data, metrics)
        anomalies = self._layer2c_anomalies(df, data, metrics)
        cond_ar1_results = self._layer2d_conditioned_ar1(data, metrics, normality)
        markov_results = self._layer2e_markov(data, metrics, normality)
        rolling_corr_results = self._layer2f_rolling_correlation(data, sig_pairs)
        multi_lag_results = self._multi_lag_carryover(data, metrics)

        summary = self._layer3_summary(
            df, n_days, metrics, sig_pairs, lag1_results,
            normality, ar1_results, anomalies,
            cond_ar1_results, markov_results,
            rolling_corr_results, multi_lag_results,
            training_for_summary,
        )

        # ── Per-window computation digest ──
        n_normal = sum(1 for s, _ in normality.values() if s == "normal")
        n_nonnorm = sum(1 for s, _ in normality.values() if s == "non_normal")
        n_markov_reliable = sum(
            1 for mr in markov_results if mr["confidence"] in ("HIGH", "GOOD")
        )
        date_range = f"{df['date'].min()} → {df['date'].max()}"
        log.info(
            "\n   ┌─── COMPUTATION DIGEST (%s, %d days) ───\n"
            "   │ Layer 0  Load & Clean    : %d metrics auto-discovered\n"
            "   │ Layer 1a Same-day Pearson: %d significant pairs (p<0.05)\n"
            "   │ Layer 1b Lag-1 Pearson   : %d next-day predictors (p<0.05)\n"
            "   │ Layer 2a Normality       : %d normal, %d non-normal\n"
            "   │ Layer 2b AR(1)           : %d persistence models\n"
            "   │ Layer 2c Anomalies       : %d recent anomalies (|z|>1.5)\n"
            "   │ Layer 2d Conditioned AR  : %d multi-var prediction models\n"
            "   │ Layer 2e Markov + KL     : %d transition models (%d reliable)\n"
            "   │ Layer 3  Summary         : %d chars (~%d tokens)\n"
            "   └──────────────────────────────────────────────",
            date_range, n_days,
            len(metrics),
            len(sig_pairs),
            len(lag1_results),
            n_normal, n_nonnorm,
            len(ar1_results),
            len(anomalies),
            len(cond_ar1_results),
            len(markov_results), n_markov_reliable,
            len(summary), len(summary) // 4,
        )

        return summary

    # ─── Data range helper ─────────────────────────────────────

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

    # ─── Benchmark entry point ────────────────────────────────

    def compute_benchmarks(self, training_days: Optional[Dict] = None,
                           ref_date=None) -> Dict[str, Any]:
        """Compute correlation matrices for every benchmark period that
        has sufficient data (≥5 days).

        Benchmark windows (all ending at *ref_date*):
          current_week  = Monday of this week → today (may be partial)
          1_week        = last 7 days
          2_weeks       = last 14 days
          …up to 1_year = last 365 days

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
        log.info(f"\n📐 Correlation Engine — benchmark analysis (ref {ref})…")
        self.bootstrap_schema()

        # 1. Discover data range
        earliest, latest, total_days = self._get_data_range()
        if earliest is None:
            log.info("   ⚠️  No data in daily_metrics.")
            return {"benchmarks": {}, "available": [], "longest": None,
                    "comparison": "", "data_days": 0}
        log.info(f"   Data range: {earliest} → {latest}  ({total_days} distinct days)")

        # 2. Build candidate windows
        candidates: List[Tuple[str, str, date, date]] = []
        seen_ranges: set = set()  # deduplicate identical clamped ranges
        for key, label, days in BENCHMARK_PERIODS:
            if days is None:
                # Current week: Monday → ref
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
                log.info(f"   ⏭️  {label}: skipping ({potential_days} potential days, need ≥5)")
                continue

            # Skip current_week if it would be identical to 1_week
            if key == "current_week":
                one_week_start = ref - timedelta(days=6)
                if ds <= one_week_start:
                    log.info(f"   ⏭️  {label}: same as Last 7 Days, skipping duplicate")
                    continue

            # Deduplicate: if clamped range is same as a previous candidate, skip
            range_key = (actual_start, actual_end)
            if range_key in seen_ranges:
                log.info(f"   ⏭️  {label}: same effective range as a shorter window, skipping")
                continue
            seen_ranges.add(range_key)

            candidates.append((key, label, ds, de))

        # 3. Compute each benchmark
        benchmarks: Dict[str, str] = {}
        computed: List[Tuple[str, str, date, date]] = []

        for key, label, ds, de in candidates:
            log.info(f"\n══ BENCHMARK: {label} ({ds} → {de}) ══")
            summary = self._compute_raw(training_days, date_start=ds, date_end=de)
            benchmarks[key] = summary
            computed.append((key, label, ds, de))

        # 4. Determine longest
        longest_key = computed[-1][0] if computed else None

        # 5. Build comparison
        comparison = self._build_benchmark_comparison(benchmarks, computed)

        # 6. Store the longest summary
        if longest_key and benchmarks.get(longest_key):
            self._store_summary(benchmarks[longest_key])
            log.info(f"   💾 Stored summary for '{longest_key}' "
                     f"({len(benchmarks[longest_key])} chars) → matrix_summaries")

        # ── Final overview ──
        log.info("\n" + "═" * 55)
        log.info("  BENCHMARK SUMMARY")
        log.info("═" * 55)
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            sz = len(benchmarks.get(key, ""))
            log.info(f"  ✓ {label:20s}  {span:3d} days  {sz:5d} chars")
        log.info(f"  Stored to DB: {longest_key or 'none'}")
        log.info(f"  Comparison context: {len(comparison)} chars")
        log.info("═" * 55 + "\n")

        return {
            "benchmarks": benchmarks,
            "available": [c[0] for c in computed],
            "longest": longest_key,
            "comparison": comparison,
            "data_days": total_days,
        }

    # ─── Legacy multi-period wrapper ──────────────────────────

    def compute_multi_period(self, training_days: Optional[Dict] = None,
                              week1_start=None, week1_end=None,
                              week2_start=None, week2_end=None) -> Dict[str, str]:
        """
        Compute matrices for week1, week2, combined 2-week, and comparison.
        Returns dict with keys: week1, week2, combined, comparison.
        """
        log.info("\n📐 Correlation Engine — multi-period analysis…")
        self.bootstrap_schema()

        results: Dict[str, str] = {}

        log.info("\n═══ Week 1 ═══")
        results["week1"] = self._compute_raw(training_days,
                                              date_start=week1_start,
                                              date_end=week1_end)

        log.info("\n═══ Week 2 ═══")
        results["week2"] = self._compute_raw(training_days,
                                              date_start=week2_start,
                                              date_end=week2_end)

        log.info("\n═══ Combined 2-week ═══")
        results["combined"] = self._compute_raw(training_days,
                                                 date_start=week1_start,
                                                 date_end=week2_end)

        results["comparison"] = self._build_comparison(
            results["week1"], results["week2"], results["combined"]
        )

        self._store_summary(results["combined"])

        # ── Multi-period overview ──
        log.info("\n" + "═" * 55)
        log.info("  MULTI-PERIOD SUMMARY")
        log.info("═" * 55)
        log.info(f"  ✓ Week 1    : {len(results['week1']):5d} chars")
        log.info(f"  ✓ Week 2    : {len(results['week2']):5d} chars")
        log.info(f"  ✓ Combined  : {len(results['combined']):5d} chars")
        log.info(f"  ✓ Comparison: {len(results['comparison']):5d} chars")
        log.info(f"  Stored to DB: combined ({len(results['combined'])} chars)")
        log.info("═" * 55 + "\n")
        return results

    # ─── Comparison builder ───────────────────────────────────

    def _build_comparison(self, summary_w1: str, summary_w2: str,
                           summary_combined: str) -> str:
        """Build a structured comparison between week1 and week2 matrices.

        Includes FULL per-week summaries and the combined 2-week summary
        so the comparator agent can do real quantitative comparison.
        """
        lines: List[str] = []
        lines.append("=== WEEK-OVER-WEEK MATRIX COMPARISON ===\n")

        # ── Full Week 1 matrix ──
        lines.append("=" * 50)
        lines.append("[WEEK 1 — FULL CORRELATION MATRIX]")
        lines.append("=" * 50)
        lines.append(summary_w1)
        lines.append("")

        # ── Full Week 2 matrix ──
        lines.append("=" * 50)
        lines.append("[WEEK 2 — FULL CORRELATION MATRIX]")
        lines.append("=" * 50)
        lines.append(summary_w2)
        lines.append("")

        # ── Full Combined 2-week matrix ──
        lines.append("=" * 50)
        lines.append("[COMBINED 2-WEEK — FULL CORRELATION MATRIX]")
        lines.append("=" * 50)
        lines.append(summary_combined)
        lines.append("")

        # ── Instructions ──
        lines.append("=" * 50)
        lines.append("[INSTRUCTIONS FOR COMPARATOR AGENT]")
        lines.append("=" * 50)
        lines.append("You have the FULL matrices for Week 1, Week 2, and the")
        lines.append("combined 2-week period above. Compare them quantitatively:\n")
        lines.append("1. Pearson pairs: which correlations strengthened/weakened?")
        lines.append("   Compare the actual r values side by side.")
        lines.append("2. Next-day predictors: did the same variables remain predictive?")
        lines.append("   Compare r values and sample sizes.")
        lines.append("3. AR(1) persistence: compare R² values for each metric.")
        lines.append("   A change of >0.10 in R² is meaningful.")
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

    # ─── Benchmark comparison builder ─────────────────────────

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

        # ── Coverage table ──
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            lines.append(f"  • {label:20s}  {ds} → {de}  ({span} days)")
        lines.append("")

        # ── Full summaries for each period ──
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            lines.append("=" * 60)
            lines.append(f"[{label.upper()} ({ds} → {de}, {span} days) — "
                         f"FULL CORRELATION MATRIX]")
            lines.append("=" * 60)
            lines.append(benchmarks.get(key, "(not computed)"))
            lines.append("")

        # ── Instructions for comparator ──
        lines.append("=" * 60)
        lines.append("[INSTRUCTIONS FOR COMPARATOR AGENT]")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"You have FULL correlation matrices for {len(computed)}")
        lines.append("benchmark periods above. Your job is CROSS-TIMEFRAME")
        lines.append("STABILITY ANALYSIS — how do statistical patterns evolve")
        lines.append("as the analysis window grows?\n")

        lines.append("AVAILABLE WINDOWS:")
        for key, label, ds, de in computed:
            span = (de - ds).days + 1
            lines.append(f"  • {label} — {span} days of data")
        lines.append("")

        lines.append("REQUIRED ANALYSIS:")
        lines.append("")
        lines.append("1. CORRELATION EVOLUTION: For the strongest Pearson pairs,")
        lines.append("   track how r values change across benchmark windows.")
        lines.append("   Example: resting_hr × bb_charged: 7d r=-0.91, 14d r=-0.96,")
        lines.append("   30d r=-0.88 → STABLE across scales.")
        lines.append("   A change of >0.15 in r IS meaningful.")
        lines.append("")
        lines.append("2. PREDICTOR STABILITY: Do next-day predictors remain")
        lines.append("   significant at longer windows? Rising n should improve")
        lines.append("   reliability. Track r and n across windows.")
        lines.append("")
        lines.append("3. PERSISTENCE EVOLUTION: Compare AR(1) R² at each window.")
        lines.append("   Metrics with CONSISTENT R² across windows are truly")
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
        lines.append("     → highest confidence, treat as established.")
        lines.append("   - STABLE: Appears in 2+ adjacent windows")
        lines.append("     → good confidence, likely real.")
        lines.append("   - EMERGING: Only at the shortest window")
        lines.append("     → recent development, needs monitoring.")
        lines.append("   - LONGER-WINDOW-ONLY: Only at longer windows")
        lines.append("     → needs more data to detect, not visible short-term.")
        lines.append("   - SPARSE: Window had too few data points")
        lines.append("     → DATA SPARSITY, not physiological change.")
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

    # ─── LAYER 0: Load + Clean ────────────────────────────────

    def _layer0_load_and_clean(self, date_start=None,
                                date_end=None) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """Load daily_metrics, apply quality fixes, discover numeric cols."""
        log.info("   Layer 0: loading data…")

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

        try:
            df_nutrition = pd.read_sql_query(
                "SELECT date, "
                "SUM(calories) as total_calories, "
                "SUM(protein_grams) as total_protein, "
                "SUM(carbs_grams) as total_carbs, "
                "SUM(fat_grams) as total_fat, "
                "SUM(fiber_grams) as total_fiber, "
                "SUM(water_ml) as total_water_ml "
                "FROM nutrition_log GROUP BY date ORDER BY date", conn
            )
        except Exception:
            df_nutrition = pd.DataFrame()

        conn.close()

        if df.empty:
            return None, []

        log.info(f"   Loaded {len(df)} days ({df['date'].min()} → {df['date'].max()})")

        # Fix 0: Enforce daily continuity — fill date gaps with NaN rows
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

        if not df_nutrition.empty:
            df_nutrition["date"] = pd.to_datetime(df_nutrition["date"])
            df = df.merge(df_nutrition, on="date", how="left")
            n_nutrition = df_nutrition.shape[0]
            log.info(f"   Merged {n_nutrition} nutrition_log entries")

        # Fix 1: HRV sentinel 511 → NaN
        if "tr_hrv_weekly_avg" in df.columns:
            n_511 = (df["tr_hrv_weekly_avg"] == 511).sum()
            df.loc[df["tr_hrv_weekly_avg"] == 511, "tr_hrv_weekly_avg"] = np.nan
            if n_511 > 0:
                log.info(f"   Fixed {n_511} HRV sentinel values (511→NaN)")

        # Fix 2: Drop garbage columns that are always 0
        for gc in ["hydration_goal_ml", "sweat_loss_ml"]:
            if gc in df.columns:
                df.drop(columns=[gc], inplace=True)

        # Fix 3: Rename tr_ abbreviations
        actual_renames = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
        df.rename(columns=actual_renames, inplace=True)

        # Fix 4: Log-transform HRV (Esco 2025, Plews 2012)
        # RMSSD is right-skewed; lnRMSSD is standard in the literature.
        # Pearson assumes normality — use lnRMSSD instead of raw HRV.
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

        # Auto-discover numeric columns with ≥5 non-null values
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
        log.info(f"   {len(metrics)} metrics with ≥5 data points")

        return df, metrics

    # ─── LAYER 1: Pearson ─────────────────────────────────────

    def _layer1_pearson(self, data: pd.DataFrame,
                        metrics: List[str]) -> Tuple[List, List]:
        """Same-day + lag-1 Pearson correlations with p-values."""
        log.info("   Layer 1: Pearson correlations…")

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
                    sig_pairs.append((ci, cj, r, p))
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
                # Skip constant arrays (all same value → no correlation)
                if np.std(valid["pred"]) < 1e-10 or np.std(valid["tgt"]) < 1e-10:
                    continue
                r, p = sp_stats.pearsonr(valid["pred"], valid["tgt"])
                if p < 0.05 and abs(r) > 0.3:
                    lag1_results.append((predictor, target, r, p, n))
        lag1_results.sort(key=lambda x: abs(x[2]), reverse=True)

        log.info(f"   ✓ {len(sig_pairs)} same-day, {len(lag1_results)} lag-1 pairs")
        return sig_pairs, lag1_results

    # ─── LAYER 2a: Normality ─────────────────────────────────

    def _layer2a_normality(self, data: pd.DataFrame,
                           metrics: List[str]) -> Dict[str, Tuple[str, float]]:
        """Shapiro-Wilk normality test for each metric.

        Tests H₀: metric ~ Normal.  The W statistic measures how well
        the ordered sample matches expected normal order statistics:

            W = (Σ aᵢ x_(i))² / Σ (xᵢ - x̄)²

        Decision: p > 0.05 → assume normal, else non-normal.
        Non-normal metrics are preferred candidates for Markov analysis
        (Layer 2e) since their distributions benefit from state-based
        modelling over linear assumptions.

        Truncates to first 5 000 values (scipy limit).
        """
        log.info("   Layer 2a: normality tests…")
        normality: Dict[str, Tuple[str, float]] = {}
        for m in metrics:
            vals = data[m].dropna()
            if len(vals) < 5:
                normality[m] = ("assumed_normal", 1.0)
            else:
                _, p = sp_stats.shapiro(vals.values[:5000])
                normality[m] = ("normal" if p > 0.05 else "non_normal", p)
        n_norm = sum(1 for s, _ in normality.values() if s == "normal")
        log.info(f"   ✓ {n_norm}/{len(metrics)} normal")
        return normality

    # ─── LAYER 2b: AR(1) ─────────────────────────────────────

    def _layer2b_ar1(self, data: pd.DataFrame,
                     metrics: List[str]) -> List[Tuple]:
        """First-order autoregressive model: does yesterday predict today?

        For each metric x, fits the AR(1) model via OLS:

            x_t = φ·x_{t−1} + c + ε_t

        where φ (slope) is the persistence coefficient.  Reports:
        - φ : autocorrelation strength  (φ ≈ 1 → strong persistence)
        - R² : fraction of variance explained by yesterday’s value
        - p  : significance of the linear relationship

        Date-gap safe: relies on .shift(1) after asfreq('D') gap-fill,
        so non-consecutive days produce NaN pairs that are dropped.
        """
        log.info("   Layer 2b: AR(1) persistence…")
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
        log.info(f"   ✓ {len(results)} AR(1) models")
        return results

    # ─── LAYER 2c: Anomalies ─────────────────────────────────

    def _layer2c_anomalies(self, df: pd.DataFrame, data: pd.DataFrame,
                           metrics: List[str]) -> List[Tuple]:
        """Detect recent statistical anomalies via percentile thresholds.

        For each metric computes the 5th and 95th percentiles over the
        full window.  Flags the 3 most recent values that fall outside
        these bounds as anomalies, labelled HIGH (> p95) or LOW (< p5).

        Also reports the z-score for context:  z = (x − μ) / σ

        Using percentiles instead of z-scores is distribution-agnostic
        — works equally well for normal and skewed metrics (e.g. HRV).
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

    # ─── LAYER 2d: Conditioned AR(1) ─────────────────────────

    def _layer2d_conditioned_ar1(self, data: pd.DataFrame,
                                  metrics: List[str],
                                  normality: Dict) -> List[Tuple]:
        """Multi-variable next-day prediction with adjusted R².

        Extends AR(1) by adding exogenous regressors.  Two levels:

        Level 1 — single conditioning variable z:
            x_t = β₀·x_{t−1} + β₁·z_{t−1} + β₂ + ε

        Level 2 — two conditioning variables z₁, z₂:
            x_t = β₀·x_{t−1} + β₁·z₁_{t−1} + β₂·z₂_{t−1} + β₃ + ε

        Solved via ordinary least-squares (np.linalg.lstsq).
        Reports adjusted R² to penalise extra parameters:

            R²_adj = 1 − [(1−R²)(n−1)] / (n−p−1)

        Only kept if R² > 0.10 (Level 1) or R² > 0.15 (Level 2).
        Improvement = R²_model − R²_baseline (baseline = simple AR(1)).

        Uses the top-5 Level-1 winners as candidates for Level-2
        combinations (greedy forward search).
        """
        log.info("   Layer 2d: conditioned AR(1)…")
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
        log.info(f"   ✓ {len(results)} conditioned AR(1) models")
        return results

    # ─── LAYER 2e: Markov + KL ───────────────────────────────

    def _layer2e_markov(self, data: pd.DataFrame, metrics: List[str],
                        normality: Dict) -> List[Dict]:
        """Markov transition matrices with KL-divergence conditioning.

        For each target metric:

        1. **Discretization** — values are binned into N_BINS = 3 tertile
           states (LOW / MED / HIGH) using percentile edges driven by
           N_BINS.  Exception: the `acwr` metric uses evidence-based
           thresholds from Gabbett (2016, BJSM): <0.8 = UNDERTRAINED,
           0.8-1.3 = SWEET_SPOT, 1.3-1.5 = HIGH_RISK, >1.5 = DANGER.

        2. **Marginal transition matrix** T[i,j] = P(state_j | state_i).
           Only transitions between truly consecutive days are counted
           (gap-aware: checks |date_{t+1} − date_t| = 1).

        3. **Adaptive kernel smoothing** — regularises sparse rows:
               T' = (1−α)·T + α·U
           where U is uniform(1/N) and α = clamp(BASE/√n, MIN, MAX)
           decreases with more data (see _adaptive_alpha).

        4. **Stationary distribution** π from the left eigen-vector
           of T' corresponding to eigenvalue 1:
               π T' = π ,  Σ π_i = 1

        5. **Conditional split** — for each candidate conditioning
           metric c, the population is split by median:
               c_level = 0 if c ≤ median(c) else 1

        6. **KL divergence** measures how much the conditional
           transition matrix differs from the marginal:
               D_KL(P ∥ Q) = Σ_j P[i,j] · ln(P[i,j] / Q[i,j])
           averaged over rows and both conditioning levels.
           Higher KL → the conditioning metric genuinely changes
           the day-to-day dynamics of the target.

        Gated by MIN_TRANSITIONS_KL per conditioning level to
        avoid noisy KL estimates from tiny samples.
        """
        log.info("   Layer 2e: Markov transitions + KL…")

        # Targets: non-normal + key health metrics
        markov_targets = [m for m in metrics
                          if normality.get(m, ("normal",))[0] == "non_normal"
                          and data[m].notna().sum() >= 8]
        for m in KEY_TARGETS:
            actual = RENAME_MAP.get(m, m)
            if actual in metrics and actual not in markov_targets:
                if data[actual].notna().sum() >= 8:
                    markov_targets.append(actual)

        results = []

        for target in markov_targets:
            non_null = data[["date", target]].dropna()
            vals = non_null[target].values
            dates = pd.to_datetime(non_null["date"]).values
            if len(vals) < 6:
                continue

            # Use ACWR evidence-based bins (Gabbett 2016) for acwr metric,
            # otherwise use N_BINS-driven percentile bins
            if target == "acwr":
                edges = np.array(ACWR_EDGES, dtype=np.float64)
                actual_labels = ACWR_LABELS
            else:
                edges = np.percentile(vals, np.linspace(0, 100, N_BINS + 1))
                edges[0] -= 1
                edges[-1] += 1
                actual_labels = BIN_LABELS
            edges = np.unique(edges)
            actual_bins = len(edges) - 1
            if actual_bins < 2:
                continue

            bins = np.clip(np.digitize(vals, edges[1:-1]), 0, actual_bins - 1)

            # Marginal transition matrix — only count consecutive-day pairs
            T_marginal = np.zeros((actual_bins, actual_bins), dtype=np.float64)
            for t in range(len(bins) - 1):
                day_gap = (dates[t + 1] - dates[t]) / np.timedelta64(1, "D")
                if day_gap == 1:  # only truly consecutive days
                    T_marginal[bins[t], bins[t + 1]] += 1

            n_trans = int(T_marginal.sum())
            alpha = _adaptive_alpha(n_trans)

            # Normalize + ADAPTIVE KERNEL SMOOTH (must match conditional!)
            rs = T_marginal.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            T_marginal_norm = T_marginal / rs
            uniform = np.ones_like(T_marginal_norm) / actual_bins
            T_marginal_norm = ((1 - alpha) * T_marginal_norm
                               + alpha * uniform)
            rs = T_marginal_norm.sum(axis=1, keepdims=True)
            T_marginal_norm = T_marginal_norm / rs

            # Stationary distribution
            try:
                eigvals, eigvecs = np.linalg.eig(T_marginal_norm.T)
                idx = np.argmin(np.abs(eigvals - 1.0))
                stationary = np.real(eigvecs[:, idx])
                stationary = stationary / stationary.sum()
            except Exception:
                stationary = np.ones(actual_bins) / actual_bins

            # Find best conditioning metric by KL
            best_kl = 0
            best_cond = None
            best_cond_T = None

            cond_cands = [m for m in metrics
                          if m != target and data[m].notna().sum() >= 8]

            for cond_metric in cond_cands:
                sub = data[["date", target, cond_metric]].dropna()
                if len(sub) < 6:
                    continue
                t_vals = sub[target].values
                c_vals = sub[cond_metric].values
                sub_dates = pd.to_datetime(sub["date"]).values

                t_bins = np.clip(np.digitize(t_vals, edges[1:-1]),
                                 0, actual_bins - 1)
                c_med = np.median(c_vals)
                c_bins = (c_vals > c_med).astype(int)

                T_cond = {}
                level_trans = {}
                for c_level in [0, 1]:
                    T_c = np.zeros((actual_bins, actual_bins), dtype=np.float64)
                    for t in range(len(t_bins) - 1):
                        day_gap = (sub_dates[t + 1] - sub_dates[t]) / np.timedelta64(1, "D")
                        if c_bins[t] == c_level and day_gap == 1:
                            T_c[t_bins[t], t_bins[t + 1]] += 1
                    n_trans_c = int(T_c.sum())
                    level_trans[c_level] = n_trans_c
                    rs = T_c.sum(axis=1, keepdims=True)
                    rs[rs == 0] = 1
                    T_c = T_c / rs
                    # Same adaptive kernel smoothing as marginal
                    alpha_c = _adaptive_alpha(n_trans_c)
                    T_c = (1 - alpha_c) * T_c + alpha_c * uniform
                    rs = T_c.sum(axis=1, keepdims=True)
                    T_c = T_c / rs
                    T_cond[c_level] = T_c

                # Gate: require minimum transitions per conditioning level
                if min(level_trans.values()) < MIN_TRANSITIONS_KL:
                    continue

                # KL divergence (both sides smoothed → no blowup)
                eps = 1e-12
                kl_total = 0
                for c_level in [0, 1]:
                    P = np.clip(T_cond[c_level], eps, None)
                    Q = np.clip(T_marginal_norm, eps, None)
                    kl = float(np.mean(np.sum(P * np.log(P / Q), axis=1)))
                    kl_total += kl
                kl_avg = kl_total / 2

                if kl_avg > best_kl:
                    best_kl = kl_avg
                    best_cond = cond_metric
                    best_cond_T = T_cond

            n_total = int(T_marginal.sum())
            # Confidence tier based on transition count
            if n_total >= 100:
                conf_tier = "HIGH"
            elif n_total >= 30:
                conf_tier = "GOOD"
            elif n_total >= 15:
                conf_tier = "MODERATE"
            else:
                conf_tier = "PRELIMINARY"

            results.append({
                "target": target,
                "bins": actual_bins,
                "labels": actual_labels[:actual_bins],
                "edges": edges,
                "marginal": T_marginal_norm,
                "stationary": stationary,
                "n_transitions": n_total,
                "smooth_alpha": alpha,
                "confidence": conf_tier,
                "best_cond": best_cond,
                "best_cond_T": best_cond_T,
                "best_kl": best_kl,
            })

        results.sort(key=lambda x: x["best_kl"], reverse=True)
        log.info(f"   ✓ {len(results)} Markov models")
        return results

    # ─── Multi-Lag Carryover Detection (Daza 2018) ────────────

    def _multi_lag_carryover(self, data: pd.DataFrame,
                              metrics: List[str]) -> List[Tuple]:
        """Check lag-2 and lag-3 correlations for KEY_TARGETS.

        If a predictor→target relationship is still significant at
        lag-2 or lag-3 days, it indicates a multi-day carryover effect
        (e.g., a hard workout affects HRV for 2-3 days, not just 1).
        Based on Daza (2018) — carryover effects in N-of-1 designs.

        Only checks KEY_TARGETS to avoid combinatorial explosion.
        """
        log.info("   Multi-lag carryover detection…")
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
        log.info(f"   ✓ {len(results)} multi-lag carryover pairs")
        return results

    # ─── LAYER 2f: Rolling Correlation Stationarity ───────────

    def _layer2f_rolling_correlation(self, data: pd.DataFrame,
                                      sig_pairs: List[Tuple]) -> List[Dict]:
        """Check whether top correlations are stable over time.

        For each of the top-10 significant same-day pairs, compute a
        rolling 30-day Pearson r.  High variance of rolling-r means the
        relationship is non-stationary (unstable) — possibly driven by
        a confound or a phase-shift in the user's routine.

        Returns list of dicts with:
          - pair: (metric_a, metric_b)
          - mean_r: mean of rolling-r values
          - std_r: std of rolling-r values (instability indicator)
          - stability: "STABLE" (std < 0.15), "MODERATE" (std < 0.25),
                       or "UNSTABLE" (std >= 0.25)
        """
        log.info("   Layer 2f: rolling correlation stationarity…")
        results = []
        window = 30

        for a, b, r_overall, p in sig_pairs[:10]:
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

        log.info(f"   ✓ {len(results)} rolling correlation checks")
        return results

    # ─── LAYER 3: Agent Summary ───────────────────────────────

    def _layer3_summary(self, df, n_days, metrics, sig_pairs, lag1_results,
                        normality, ar1_results, anomalies,
                        cond_ar1_results, markov_results,
                        rolling_corr_results, multi_lag_results,
                        training_days) -> str:
        """Build natural-language summary for agent consumption."""
        log.info("   Layer 3: building agent summary…")
        lines: List[str] = []

        lines.append(f"=== CORRELATION MATRIX ANALYSIS ({df['date'].max()}) ===")
        lines.append(f"Data: {n_days} days, {len(metrics)} numeric metrics")

        if n_days < 21:
            lines.append(
                f"NOTE*: Only {n_days} days of data collected so far. "
                f"All multi-variable models are PRELIMINARY — treat R² values "
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
        for a, b, r, p in sig_pairs[:12]:
            arrow = "↑↑" if r > 0 else "↑↓"
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else "*")
            lines.append(f"  {arrow} {a} × {b}: r={r:+.3f} (p={p:.4f}) {sig}")
        lines.append("")

        # Next-day predictors
        lines.append("[NEXT-DAY PREDICTORS (what yesterday predicts about today)]")
        for pred, tgt, r, p, n in lag1_results[:8]:
            lines.append(f"  {pred} → {tgt}: r={r:+.3f} (p={p:.4f}, n={n})")
        lines.append("")

        # AR(1) persistence
        lines.append("[AUTOREGRESSIVE PERSISTENCE]")
        for entry in ar1_results[:8]:
            m, phi, r2, p, n = entry[:5]
            is_stationary = entry[5] if len(entry) > 5 else True
            label = ("strong" if r2 > 0.4
                     else ("moderate" if r2 > 0.15 else "weak"))
            flag = "" if is_stationary else " ⚠ TREND-DOMINATED"
            lines.append(f"  {m}: φ={phi:+.3f}, R²={r2:.3f} ({label}){flag}")
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
                lines.append(f"  {dt} — {m}={val:.1f} (z={z:+.2f}, {d})")
            lines.append("")

        # Multi-lag carryover
        if multi_lag_results:
            lines.append("[MULTI-DAY CARRYOVER (lag-2 and lag-3 still significant)]")
            for pred, tgt, lag, r, p, n in multi_lag_results:
                lines.append(
                    f"  {pred} → {tgt} (lag-{lag}): r={r:+.3f} (p={p:.4f}, n={n})"
                    f" — effect persists {lag} days"
                )
            lines.append("")

        # Rolling correlation stability
        if rolling_corr_results:
            lines.append("[CORRELATION STABILITY (30-day rolling window)]")
            for rc in rolling_corr_results:
                a, b = rc["pair"]
                lines.append(
                    f"  {a} × {b}: mean_r={rc['mean_r']:+.3f}, "
                    f"std={rc['std_r']:.3f} ({rc['stability']})"
                )
            lines.append("")

        # Conditioned AR(1)
        lines.append("[CONDITIONED AR(1) — MULTI-VARIABLE NEXT-DAY PREDICTION]")
        lines.append("  Higher R² = better prediction. "
                     "Positive improvement = conditioning helps.")
        for target, conds, phi, beta_c, r2, improv, n in cond_ar1_results[:10]:
            cond_str = " + ".join(conds)
            # Adjusted R²
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
                f"  {target} ~ lag + {cond_str}: R²={r2:.3f} "
                f"(adj={r2_adj:.3f}) ({improv:+.3f} vs simple, n={n}{star}) "
                f"— {interp}"
            )
        if any(n < 20 for *_, n in cond_ar1_results[:10]):
            lines.append("  * n<20: preliminary, may be inflated")
        lines.append("")

        # Markov transitions
        lines.append("[MARKOV STATE TRANSITIONS]")
        lines.append("  Each metric split into LOW/MED/HIGH. "
                     "High diagonal = 'sticky' state.")
        lines.append("  Smoothing is ADAPTIVE: α shrinks as data grows "
                     "(more data → trust empirical, less smoothing).")
        for mr in markov_results[:8]:
            labels = mr.get("labels", BIN_LABELS[:mr["bins"]])
            lines.append(
                f"\n  {mr['target']} ({mr['n_transitions']} transitions, "
                f"α={mr['smooth_alpha']:.3f}, "
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
                        f"{mr['marginal'][i, nxt]:.0%} → {labels[nxt]}"
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
                                        f"{target} {labels_k[i]}→{labels_k[j]} "
                                        f"changes by {d:+.2f}"
                                    )
                    if diff_desc:
                        lines.append(f"    → {diff_desc}")
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
        log.info(f"   ✓ Summary: {len(summary)} chars (~{len(summary)//4} tokens)")
        return summary

    # ─── Storage ──────────────────────────────────────────────

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

    # ─── Public API ───────────────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = CorrelationEngine()
    summary = engine.compute_weekly()
    log.info("\n" + summary)
