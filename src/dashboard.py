"""
Garmin Health Dashboard
=======================
Interactive Streamlit web interface for exploring health data.
Standalone — uses psycopg2 directly.

Run with: streamlit run dashboard.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

from dotenv import load_dotenv
import psycopg2

load_dotenv()

# ── Connection string: Streamlit Cloud secrets → env var fallback ──
def _get_conn_str() -> str:
    try:
        return st.secrets["POSTGRES_CONNECTION_STRING"]
    except Exception:
        return os.getenv("POSTGRES_CONNECTION_STRING", "")

# ── Local imports (all standalone) ────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualizations import HealthDataVisualizer
from enhanced_agents import AdvancedHealthAgents

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Health Dashboard",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #00B8A9; margin-bottom: 2rem; }
    .metric-card  { background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #00B8A9; }
    .improvement  { color: #06FFA5; font-weight: bold; }
    .decline      { color: #E63946; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────

CONN_STR = _get_conn_str()


@st.cache_resource
def get_visualizer():
    return HealthDataVisualizer(CONN_STR)


@st.cache_resource
def get_agents():
    return AdvancedHealthAgents()


def query(sql: str) -> pd.DataFrame:
    conn = psycopg2.connect(CONN_STR)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


def load_data(days: int = 30) -> pd.DataFrame:
    return query(f"""
        SELECT * FROM daily_metrics
        WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
        ORDER BY date DESC
    """)


def calculate_trends(df: pd.DataFrame, metric: str):
    if metric not in df.columns or len(df) < 7:
        return 0, "→"
    recent = df[metric].head(7).mean()
    previous = df[metric].iloc[7:14].mean() if len(df) >= 14 else df[metric].mean()
    if pd.isna(recent) or pd.isna(previous):
        return 0, "→"
    change = ((recent - previous) / previous * 100) if previous != 0 else 0
    arrow = "↑" if change > 0 else ("↓" if change < 0 else "→")
    return change, arrow


# ═══════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════

def main():
    st.sidebar.title("🏃‍♂️ Health Dashboard")
    st.sidebar.markdown("---")

    page = st.sidebar.selectbox(
        "Navigate",
        ["📊 Overview", "📈 Trends & Analysis", "🔍 Deep Dive",
         "🤖 AI Insights", "🎯 Goals & Progress", "⚙️ Settings"],
    )
    st.sidebar.markdown("---")
    days_back = st.sidebar.slider("Days to display", 7, 90, 30)

    try:
        df = load_data(days_back)
        if df.empty:
            st.error("No data available. Run `python weekly_sync.py` to sync.")
            return

        if page == "📊 Overview":
            show_overview(df)
        elif page == "📈 Trends & Analysis":
            show_trends()
        elif page == "🔍 Deep Dive":
            show_deep_dive(df)
        elif page == "🤖 AI Insights":
            show_ai_insights()
        elif page == "🎯 Goals & Progress":
            show_goals()
        elif page == "⚙️ Settings":
            show_settings()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure your database is set up and contains data.")


# ── Overview ──────────────────────────────────────────────────

def show_overview(df: pd.DataFrame):
    st.markdown('<p class="main-header">💪 Your Health Overview</p>',
                unsafe_allow_html=True)

    latest = df.iloc[0]
    st.markdown(f"### Latest Data: {latest['date']}")

    col1, col2, col3, col4 = st.columns(4)

    rhr_c, rhr_a = calculate_trends(df, "resting_hr")
    hrv_c, hrv_a = calculate_trends(df, "hrv_last_night")
    slp_c, slp_a = calculate_trends(df, "sleep_score")
    str_c, str_a = calculate_trends(df, "stress_level")

    with col1:
        v = latest.get("resting_hr")
        st.metric("Resting Heart Rate",
                  f"{v:.0f} bpm" if pd.notna(v) else "N/A",
                  f"{rhr_a} {abs(rhr_c):.1f}%", delta_color="inverse")
    with col2:
        v = latest.get("hrv_last_night")
        st.metric("HRV",
                  f"{v:.0f} ms" if pd.notna(v) else "N/A",
                  f"{hrv_a} {abs(hrv_c):.1f}%", delta_color="normal")
    with col3:
        v = latest.get("sleep_score")
        st.metric("Sleep Score",
                  f"{v:.0f}" if pd.notna(v) else "N/A",
                  f"{slp_a} {abs(slp_c):.1f}%", delta_color="normal")
    with col4:
        v = latest.get("stress_level")
        st.metric("Stress Level",
                  f"{v:.0f}" if pd.notna(v) else "N/A",
                  f"{str_a} {abs(str_c):.1f}%", delta_color="inverse")

    st.markdown("---")

    # Recovery status
    st.markdown("### 🔋 Recovery Status")
    c1, c2, c3 = st.columns(3)

    with c1:
        r = latest.get("training_readiness")
        if pd.notna(r):
            icon = "🟢" if r >= 75 else ("🟡" if r >= 50 else "🔴")
            st.markdown(f"{icon} **Training Readiness:** {r:.0f}/100")
        else:
            st.markdown("**Training Readiness:** N/A")

    with c2:
        bb = latest.get("bb_peak")
        if pd.notna(bb):
            icon = "🟢" if bb >= 80 else ("🟡" if bb >= 50 else "🔴")
            st.markdown(f"{icon} **Body Battery Peak:** {bb:.0f}/100")
        else:
            st.markdown("**Body Battery:** N/A")

    with c3:
        hs = latest.get("hrv_status")
        if pd.notna(hs) and hs:
            icon = "🟢" if "balanced" in str(hs).lower() else "🟡"
            st.markdown(f"{icon} **HRV Status:** {hs}")
        else:
            st.markdown("**HRV Status:** N/A")

    st.markdown("---")

    # Last 7 days chart
    st.markdown("### 📊 Last 7 Days")
    last_7 = df.head(7).sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_7["date"], y=last_7["training_readiness"],
                             name="Readiness", mode="lines+markers",
                             line=dict(color="#00B8A9", width=3)))
    fig.add_trace(go.Scatter(x=last_7["date"], y=last_7["sleep_score"],
                             name="Sleep Score", mode="lines+markers",
                             line=dict(color="#5C7CFA", width=2)))
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=True, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Recent activities
    st.markdown("### 🏃‍♂️ Recent Activities")
    try:
        activities = query("""
            SELECT date, activity_name, activity_type,
                   duration_sec / 60 AS minutes,
                   distance_m / 1000.0 AS km,
                   average_hr AS avg_hr, calories
            FROM activities
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            ORDER BY date DESC LIMIT 5
        """)
        if not activities.empty:
            st.dataframe(activities, use_container_width=True, hide_index=True)
        else:
            st.info("No recent activities logged.")
    except Exception:
        st.info("Activities table not available yet.")


# ── Trends ────────────────────────────────────────────────────

def show_trends():
    st.markdown('<p class="main-header">📈 Trends & Analysis</p>',
                unsafe_allow_html=True)

    viz = get_visualizer()
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "💤 Recovery", "🏋️ Training", "🔗 Correlations"])

    with tab1:
        fig = viz.create_weekly_dashboard(weeks_back=4)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = viz.create_recovery_analysis(days=30)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = viz.create_training_analysis(days=60)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = viz.create_correlation_heatmap(days=60)
        if fig:
            st.plotly_chart(fig, use_container_width=True)


# ── Deep Dive ─────────────────────────────────────────────────

def show_deep_dive(df: pd.DataFrame):
    st.markdown('<p class="main-header">🔍 Deep Dive Analysis</p>',
                unsafe_allow_html=True)

    available = [
        "resting_hr", "hrv_last_night", "sleep_score", "stress_level",
        "bb_peak", "bb_drained", "training_readiness", "total_steps",
    ]
    metric = st.selectbox("Select metric to analyze", available)

    if metric not in df.columns:
        st.error(f"Metric {metric} not available in data")
        return

    st.markdown(f"### {metric.replace('_', ' ').title()} Over Time")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df[metric],
                             name="Actual", mode="lines+markers",
                             line=dict(color="#00B8A9", width=2)))
    ma = df[metric].rolling(window=7, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=df["date"], y=ma,
                             name="7-day Average", mode="lines",
                             line=dict(color="#F8B400", width=2, dash="dash")))
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    vals = df[metric].dropna()
    with c1:
        st.metric("Mean", f"{vals.mean():.1f}" if len(vals) else "N/A")
    with c2:
        st.metric("Median", f"{vals.median():.1f}" if len(vals) else "N/A")
    with c3:
        st.metric("Min", f"{vals.min():.1f}" if len(vals) else "N/A")
    with c4:
        st.metric("Max", f"{vals.max():.1f}" if len(vals) else "N/A")

    st.markdown("### Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[metric], nbinsx=20,
                               marker_color="#00B8A9"))
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


# ── AI Insights ───────────────────────────────────────────────

def show_ai_insights():
    st.markdown('<p class="main-header">🤖 AI-Powered Insights</p>',
                unsafe_allow_html=True)
    st.info("💡 These insights go far beyond what Garmin Connect offers!")

    analysis_type = st.selectbox(
        "Select analysis type",
        ["Comprehensive Analysis", "Weekly Summary", "Goal Progress"],
    )

    if analysis_type == "Comprehensive Analysis":
        st.markdown("### Deep Pattern Analysis")
        period = st.slider("Analysis period (days)", 14, 90, 30)
        if st.button("🚀 Run Analysis", type="primary"):
            with st.spinner("AI agents analyzing your data…"):
                agents = get_agents()
                result = agents.run_comprehensive_analysis(period)
                st.markdown("### Analysis Results")
                st.markdown(str(result))

    elif analysis_type == "Weekly Summary":
        st.markdown("### Weekly Performance Summary")
        if st.button("📊 Generate Summary", type="primary"):
            with st.spinner("Generating weekly summary…"):
                agents = get_agents()
                result = agents.run_weekly_summary()
                st.markdown("### This Week vs Last Week")
                st.markdown(str(result))

    elif analysis_type == "Goal Progress":
        st.markdown("### Goal Progress Tracking")
        goal = st.text_input("Enter your goal", "Improve HRV by 10%")
        if st.button("🎯 Analyze Progress", type="primary"):
            with st.spinner("Analyzing goal progress…"):
                agents = get_agents()
                result = agents.run_goal_analysis(goal)
                st.markdown("### Progress Report")
                st.markdown(str(result))


# ── Goals ─────────────────────────────────────────────────────

def show_goals():
    st.markdown('<p class="main-header">🎯 Goals & Progress</p>',
                unsafe_allow_html=True)
    st.info("🚧 Custom goals tracking coming soon!  "
            "Use AI Insights to track progress.")

    weekly = query("""
        WITH weekly_stats AS (
            SELECT
                DATE_TRUNC('week', date) AS week,
                AVG(resting_hr) AS avg_rhr,
                AVG(hrv_last_night) AS avg_hrv,
                AVG(sleep_score) AS avg_sleep,
                AVG(training_readiness) AS avg_readiness
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '28 days'
            GROUP BY week ORDER BY week
        )
        SELECT * FROM weekly_stats
    """)

    if not weekly.empty:
        st.markdown("### 4-Week Progress")
        for metric, name in [
            ("avg_rhr", "Resting HR"), ("avg_hrv", "HRV"),
            ("avg_sleep", "Sleep Score"), ("avg_readiness", "Readiness"),
        ]:
            if metric in weekly.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=weekly["week"], y=weekly[metric],
                    mode="lines+markers", line=dict(width=3), name=name))
                fig.update_layout(title=name, height=250,
                                  margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)


# ── Settings ──────────────────────────────────────────────────

def show_settings():
    st.markdown('<p class="main-header">⚙️ Settings</p>',
                unsafe_allow_html=True)

    st.markdown("### Data Management")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Sync Data from Garmin", type="primary"):
            st.info("Run `python weekly_sync.py` to sync data")
    with c2:
        if st.button("📥 Export All Charts"):
            viz = get_visualizer()
            viz.export_all_charts()
            st.success("Charts exported to health_reports/ directory!")

    st.markdown("---")
    st.markdown("### Database Info")
    try:
        mc = query("SELECT COUNT(*) AS count FROM daily_metrics")
        ac = query("SELECT COUNT(*) AS count FROM activities")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Daily Metrics Records", mc["count"].iloc[0])
        with c2:
            st.metric("Activities Logged", ac["count"].iloc[0])
    except Exception:
        st.warning("Could not query database.")


if __name__ == "__main__":
    main()
