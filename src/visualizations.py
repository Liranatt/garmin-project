"""
Garmin Data Visualization & Analysis
=====================================
Creates comprehensive Plotly charts from daily_metrics.
Standalone — uses psycopg2 directly, no DatabaseManager.

All column names match the real PostgreSQL schema:
    resting_hr, hrv_last_night, sleep_score, stress_level,
    training_readiness, bb_charged, bb_drained, bb_peak, bb_low,
    total_steps, deep_sleep_sec, rem_sleep_sec, sleep_seconds,
    avg_respiration, body_battery_change, tr_acute_load, weight_kg …
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import psycopg2
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("visualizations")


class HealthDataVisualizer:
    """Creates Plotly visualisations of health data."""

    def __init__(self, conn_str: Optional[str] = None):
        self.conn_str = conn_str or os.getenv("POSTGRES_CONNECTION_STRING", "")
        self.colors = {
            "primary": "#00B8A9",
            "secondary": "#F8B400",
            "danger": "#E63946",
            "success": "#06FFA5",
            "info": "#5C7CFA",
            "warning": "#FFA94D",
        }

    # ── helper ────────────────────────────────────────────────

    def _query(self, sql: str) -> pd.DataFrame:
        conn = psycopg2.connect(self.conn_str)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df

    # ── 1. Weekly dashboard ───────────────────────────────────

    def create_weekly_dashboard(self, weeks_back: int = 4) -> Optional[go.Figure]:
        """All key metrics in one 4×2 figure."""
        df = self._query(f"""
            SELECT * FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '{weeks_back * 7} days'
            ORDER BY date
        """)
        if df.empty:
            log.info("No data available for visualization")
            return None

        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=(
                "Resting Heart Rate Trend",
                "Sleep Quality & Duration",
                "HRV (Recovery Indicator)",
                "Stress Levels",
                "Body Battery (Charged vs Drained)",
                "Body Battery Range",
                "Steps & Activity",
                "Training Readiness",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": True}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.15,
        )

        # 1. Resting HR + MA
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["resting_hr"],
                       name="Resting HR", mode="lines+markers",
                       line=dict(color=self.colors["primary"], width=2)),
            row=1, col=1,
        )
        rhr_ma = df["resting_hr"].rolling(window=7, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=df["date"], y=rhr_ma,
                       name="7-day Avg", mode="lines",
                       line=dict(color=self.colors["secondary"], width=2, dash="dash")),
            row=1, col=1,
        )

        # 2. Sleep score bar + duration line
        fig.add_trace(
            go.Bar(x=df["date"], y=df["sleep_score"],
                   name="Sleep Score", marker_color=self.colors["info"]),
            row=1, col=2,
        )
        if "sleep_seconds" in df.columns:
            sleep_hrs = df["sleep_seconds"] / 3600.0
            fig.add_trace(
                go.Scatter(x=df["date"], y=sleep_hrs,
                           name="Sleep Hours", mode="lines+markers",
                           line=dict(color=self.colors["warning"], width=2)),
                row=1, col=2, secondary_y=True,
            )

        # 3. HRV
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["hrv_last_night"],
                       name="HRV (ms)", mode="lines+markers",
                       line=dict(color=self.colors["success"], width=2),
                       fill="tozeroy",
                       fillcolor="rgba(6, 255, 165, 0.1)"),
            row=2, col=1,
        )

        # 4. Stress
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["stress_level"],
                       name="Avg Stress", mode="lines",
                       line=dict(color=self.colors["danger"], width=2),
                       fill="tozeroy",
                       fillcolor="rgba(230, 57, 70, 0.1)"),
            row=2, col=2,
        )

        # 5. Body Battery charged/drained
        fig.add_trace(
            go.Bar(x=df["date"], y=df["bb_charged"],
                   name="BB Charged", marker_color=self.colors["success"]),
            row=3, col=1,
        )
        fig.add_trace(
            go.Bar(x=df["date"], y=-df["bb_drained"].fillna(0),
                   name="BB Drained", marker_color=self.colors["danger"]),
            row=3, col=1,
        )

        # 6. Body Battery range (peak / low)
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["bb_peak"],
                       name="BB Peak", mode="lines",
                       line=dict(color=self.colors["success"], width=2)),
            row=3, col=2,
        )
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["bb_low"],
                       name="BB Low", mode="lines",
                       line=dict(color=self.colors["danger"], width=2)),
            row=3, col=2,
        )

        # 7. Steps
        fig.add_trace(
            go.Bar(x=df["date"], y=df["total_steps"],
                   name="Steps", marker_color=self.colors["info"]),
            row=4, col=1,
        )

        # 8. Training readiness
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["training_readiness"],
                       name="Readiness", mode="lines+markers",
                       line=dict(color=self.colors["primary"], width=3)),
            row=4, col=2,
        )

        fig.update_layout(
            height=1400, showlegend=True,
            title_text=f"{weeks_back}-Week Health Dashboard",
            title_font_size=24, hovermode="x unified",
        )
        return fig

    # ── 2. Recovery analysis ──────────────────────────────────

    def create_recovery_analysis(self, days: int = 30) -> Optional[go.Figure]:
        """Sleep/HRV/stress/BB relationships."""
        df = self._query(f"""
            SELECT date, sleep_score, sleep_seconds, hrv_last_night,
                   stress_level, bb_peak, training_readiness, resting_hr
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY date
        """)
        if df.empty:
            return None

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Sleep Quality vs Next-Day Readiness",
                "HRV vs Resting Heart Rate",
                "Stress Impact on Sleep",
                "Weekly Recovery Score",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # 1. Sleep → next-day readiness
        df["next_day_readiness"] = df["training_readiness"].shift(-1)
        fig.add_trace(
            go.Scatter(
                x=df["sleep_score"], y=df["next_day_readiness"],
                mode="markers", name="Sleep→Readiness",
                marker=dict(
                    size=10, color=df["hrv_last_night"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="HRV"),
                ),
                text=df["date"],
            ),
            row=1, col=1,
        )

        # 2. HRV vs RHR
        fig.add_trace(
            go.Scatter(x=df["hrv_last_night"], y=df["resting_hr"],
                       mode="markers", name="HRV vs RHR",
                       marker=dict(size=10, color=self.colors["primary"])),
            row=1, col=2,
        )

        # 3. Stress → sleep
        fig.add_trace(
            go.Scatter(x=df["stress_level"], y=df["sleep_score"],
                       mode="markers", name="Stress→Sleep",
                       marker=dict(size=10, color=self.colors["danger"])),
            row=2, col=1,
        )

        # 4. Weekly recovery composite
        df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
        weekly = df.groupby("week").agg({
            "hrv_last_night": "mean",
            "sleep_score": "mean",
            "stress_level": "mean",
            "bb_peak": "mean",
        }).dropna(how="all").reset_index()

        if not weekly.empty:
            hrv_max = weekly["hrv_last_night"].max() or 1
            weekly["recovery_score"] = (
                (weekly["hrv_last_night"] / hrv_max * 25)
                + (weekly["sleep_score"] / 100 * 25)
                + ((100 - weekly["stress_level"]) / 100 * 25)
                + (weekly["bb_peak"] / 100 * 25)
            )
            fig.add_trace(
                go.Bar(x=weekly["week"], y=weekly["recovery_score"],
                       name="Recovery Score",
                       marker_color=self.colors["success"]),
                row=2, col=2,
            )

        fig.update_layout(
            height=900,
            title_text="Recovery Analysis — Find What Works",
            title_font_size=22, showlegend=True,
        )
        return fig

    # ── 3. Training analysis ──────────────────────────────────

    def create_training_analysis(self, days: int = 60) -> Optional[go.Figure]:
        """Training patterns, load, and performance."""
        activities = self._query(f"""
            SELECT date, activity_type,
                   duration_sec / 3600.0 AS hours,
                   distance_m / 1000.0 AS km,
                   average_hr AS avg_hr, calories,
                   aerobic_training_effect,
                   anaerobic_training_effect
            FROM activities
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY date
        """)

        metrics = self._query(f"""
            SELECT date, tr_acute_load, training_readiness,
                   bb_charged, bb_drained
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            ORDER BY date
        """)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Training Acute Load & Readiness",
                "Activity Distribution",
                "Body Battery Charged vs Drained",
                "Training Effect by Activity",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "pie"}],
                [{"secondary_y": False}, {"type": "scatter"}],
            ],
        )

        # 1. Acute load + readiness
        if not metrics.empty:
            fig.add_trace(
                go.Scatter(x=metrics["date"], y=metrics["tr_acute_load"],
                           name="Acute Load",
                           line=dict(color=self.colors["danger"], width=2)),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=metrics["date"], y=metrics["training_readiness"],
                           name="Readiness",
                           line=dict(color=self.colors["info"], width=2,
                                     dash="dash")),
                row=1, col=1, secondary_y=True,
            )

        # 2. Activity distribution pie
        if not activities.empty:
            counts = activities["activity_type"].value_counts()
            fig.add_trace(
                go.Pie(labels=counts.index, values=counts.values,
                       name="Activities"),
                row=1, col=2,
            )

        # 3. BB charged vs drained
        if not metrics.empty:
            fig.add_trace(
                go.Bar(x=metrics["date"], y=metrics["bb_charged"],
                       name="BB Charged",
                       marker_color=self.colors["success"]),
                row=2, col=1,
            )
            fig.add_trace(
                go.Bar(x=metrics["date"], y=metrics["bb_drained"],
                       name="BB Drained",
                       marker_color=self.colors["danger"]),
                row=2, col=1,
            )

        # 4. Training effect scatter
        if not activities.empty:
            fig.add_trace(
                go.Scatter(
                    x=activities["aerobic_training_effect"],
                    y=activities["anaerobic_training_effect"],
                    mode="markers", name="Training Effects",
                    marker=dict(
                        size=activities["hours"] * 10,
                        color=activities["avg_hr"],
                        colorscale="RdYlGn_r", showscale=True,
                    ),
                    text=activities["activity_type"],
                ),
                row=2, col=2,
            )

        fig.update_layout(
            height=900,
            title_text="Training Analysis — Are You Improving?",
            title_font_size=22, showlegend=True,
        )
        return fig

    # ── 4. Correlation heatmap ────────────────────────────────

    def create_correlation_heatmap(self, days: int = 60) -> Optional[go.Figure]:
        """NxN Pearson heatmap of numeric columns."""
        df = self._query(f"""
            SELECT resting_hr, total_steps, sleep_score,
                   stress_level, bb_peak, bb_drained,
                   hrv_last_night, avg_respiration,
                   training_readiness, deep_sleep_sec, rem_sleep_sec
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
        """)
        if df.empty:
            return None

        corr = df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu", zmid=0,
            text=corr.values.round(2), texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation"),
        ))
        fig.update_layout(
            title="Metric Correlations — Find Hidden Patterns",
            title_font_size=22, height=800, xaxis_tickangle=-45,
        )
        return fig

    # ── 5. Progress report ────────────────────────────────────

    def create_progress_report(self, weeks_back: int = 4) -> Dict:
        """Week-over-week comparison dict."""
        df = self._query(f"""
            WITH weekly_stats AS (
                SELECT
                    DATE_TRUNC('week', date) AS week_start,
                    AVG(resting_hr) AS avg_rhr,
                    AVG(sleep_score) AS avg_sleep,
                    AVG(hrv_last_night) AS avg_hrv,
                    AVG(stress_level) AS avg_stress,
                    AVG(total_steps) AS avg_steps,
                    SUM(bb_drained) AS total_training_load,
                    AVG(training_readiness) AS avg_readiness,
                    COUNT(*) AS days_logged
                FROM daily_metrics
                WHERE date >= CURRENT_DATE - INTERVAL '{weeks_back * 7} days'
                GROUP BY week_start
                ORDER BY week_start
            )
            SELECT * FROM weekly_stats
        """)

        if len(df) < 2:
            return {"error": "Not enough data for comparison"}

        current = df.iloc[-1]
        previous = df.iloc[-2]

        def _cmp(metric):
            cur = current[metric]
            prev = previous[metric]
            chg = cur - prev if pd.notna(cur) and pd.notna(prev) else 0
            pct = (chg / prev * 100) if prev else 0
            return {"current": cur, "previous": prev,
                    "change": chg, "change_pct": pct}

        return {
            "current_week": current["week_start"],
            "comparisons": {
                "resting_hr": _cmp("avg_rhr"),
                "sleep_score": _cmp("avg_sleep"),
                "hrv": _cmp("avg_hrv"),
                "stress": _cmp("avg_stress"),
                "steps": _cmp("avg_steps"),
                "training_load": _cmp("total_training_load"),
            },
            "trends": self._calculate_trends(df),
        }

    def _calculate_trends(self, df: pd.DataFrame) -> Dict:
        trends = {}
        for col in ["avg_rhr", "avg_sleep", "avg_hrv", "avg_stress"]:
            if col in df.columns and len(df) >= 3:
                y = df[col].values
                mask = ~np.isnan(y)
                if mask.sum() >= 2:
                    x = np.arange(len(df))
                    slope, _ = np.polyfit(x[mask], y[mask], 1)
                    direction = (
                        "improving" if (slope < 0 and "stress" in col)
                                       or (slope > 0 and "stress" not in col)
                        else "declining"
                    )
                    trends[col] = {"slope": slope, "direction": direction}
        return trends

    # ── 6. Calendar heatmap ───────────────────────────────────

    def create_calendar_heatmap(self, year: int = None) -> Optional[go.Figure]:
        """GitHub-style calendar heatmap of body-battery drain."""
        if year is None:
            year = datetime.now().year

        df = self._query(f"""
            SELECT date, bb_drained, total_steps, sleep_score
            FROM daily_metrics
            WHERE EXTRACT(YEAR FROM date) = {year}
            ORDER BY date
        """)
        if df.empty:
            return None

        df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
        df["week_of_year"] = (
            pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
        )

        pivot = df.pivot_table(
            values="bb_drained", index="day_of_week",
            columns="week_of_year", aggfunc="mean",
        )

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns,
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale="YlOrRd",
            colorbar=dict(title="BB Drained"),
        ))
        fig.update_layout(
            title=f"{year} Training Calendar",
            xaxis_title="Week of Year", yaxis_title="Day of Week",
            height=400,
        )
        return fig

    # ── Export ────────────────────────────────────────────────

    def export_all_charts(self, output_dir: str = "health_reports"):
        """Generate and save all charts to HTML files."""
        os.makedirs(output_dir, exist_ok=True)
        log.info("Generating all visualizations…")

        charts = {
            "dashboard": self.create_weekly_dashboard(),
            "recovery": self.create_recovery_analysis(),
            "training": self.create_training_analysis(),
            "correlations": self.create_correlation_heatmap(),
            "calendar": self.create_calendar_heatmap(),
        }

        for name, fig in charts.items():
            if fig:
                filepath = os.path.join(output_dir, f"{name}.html")
                fig.write_html(filepath)
                log.info(f"   Saved {name}.html")

        log.info(f"\nAll charts saved to {output_dir}/")
        log.info("   Open any HTML file in your browser to view!")
