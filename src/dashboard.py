"""
Garmin Health Intelligence — Dashboard v2
==========================================
2026 glass-morphism UI · 9-agent parallel AI · chat interface
Standalone — psycopg2 only, no ORM.

Run:  streamlit run src/dashboard.py
"""
from __future__ import annotations

import os, sys, time, threading, html
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv
import psycopg2

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Garmin Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _conn_str() -> str:
    try:
        return st.secrets["POSTGRES_CONNECTION_STRING"]
    except Exception:
        return os.getenv("POSTGRES_CONNECTION_STRING", "")


def _api_key() -> str:
    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        return os.getenv("GOOGLE_API_KEY", "")


CONN = _conn_str()

# ═══════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 40%, #0a0f1a 100%);
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(13,17,23,0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Glass card */
.gc {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}
.gc:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

/* Metric tile */
.mt {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.mt::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.mt.ag::before { background: linear-gradient(90deg, #00f5a0, #00d9f5); }
.mt.ab::before { background: linear-gradient(90deg, #667eea, #764ba2); }
.mt.aa::before { background: linear-gradient(90deg, #f093fb, #f5576c); }
.mt.ac::before { background: linear-gradient(90deg, #4facfe, #00f2fe); }

.ml {
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: rgba(255,255,255,0.4);
    font-weight: 600;
    margin-bottom: .35rem;
}
.mv {
    font-size: 2rem;
    font-weight: 800;
    color: #fff;
    line-height: 1;
    margin-bottom: .25rem;
    letter-spacing: -0.02em;
}
.mu { font-size: .85rem; font-weight: 400; color: rgba(255,255,255,0.35); }
.md { font-size: .75rem; font-weight: 600; margin-top: .4rem; }
.md.up { color: #00f5a0; }
.md.dn { color: #f5576c; }
.md.fl { color: rgba(255,255,255,0.3); }

/* Page titles — centered, bold, proud */
.pt {
    font-size: 2.2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
    color: #fff;
    text-align: center;
    margin-bottom: .1rem;
}
.ps {
    font-size: .8rem;
    color: rgba(255,255,255,0.3);
    font-weight: 400;
    text-align: center;
    margin-bottom: 2.5rem;
}

/* Section header */
.sh {
    font-size: .7rem;
    text-transform: uppercase;
    letter-spacing: .12em;
    color: rgba(255,255,255,0.3);
    font-weight: 700;
    margin: 2.5rem 0 1rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* Status dot */
.sd {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pg 2s ease-in-out infinite;
}
.sd.g { background: #00f5a0; box-shadow: 0 0 8px rgba(0,245,160,0.5); }
.sd.y { background: #f5c542; box-shadow: 0 0 8px rgba(245,197,66,0.5); }
.sd.r { background: #f5576c; box-shadow: 0 0 8px rgba(245,87,108,0.5); }
@keyframes pg { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

/* Recovery bar */
.rb-bg {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 6px;
    width: 100%;
    margin-top: .5rem;
    overflow: hidden;
}
.rb-f {
    height: 100%;
    border-radius: 8px;
    transition: width 1s ease;
}

/* Chat */
.cm {
    padding: 1rem 1.25rem;
    border-radius: 12px;
    margin-bottom: .75rem;
    font-size: .9rem;
    line-height: 1.6;
}
.cu {
    background: rgba(102,126,234,0.15);
    border: 1px solid rgba(102,126,234,0.2);
    margin-left: 3rem;
}
.ca {
    background: rgba(0,245,160,0.06);
    border: 1px solid rgba(0,245,160,0.1);
    margin-right: 3rem;
}
.cn {
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: #00f5a0;
    font-weight: 700;
    margin-bottom: .35rem;
}

/* Activity row */
.ar {
    display: flex;
    align-items: center;
    padding: .75rem 1rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    margin-bottom: .5rem;
    transition: background .2s;
}
.ar:hover { background: rgba(255,255,255,0.05); }

/* Insight card */
.ic {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: .75rem;
}
.ic-t {
    font-size: .65rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    font-weight: 700;
    margin-bottom: .5rem;
}
.ic-b {
    font-size: .85rem;
    color: rgba(255,255,255,0.75);
    line-height: 1.7;
    white-space: pre-wrap;
}

/* Streamlit overrides */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%) !important;
    color: #0a0a0f !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    padding: .5rem 1.5rem !important;
    font-size: .85rem !important;
    letter-spacing: .02em !important;
    transition: all .3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(0,245,160,0.3) !important;
}
div[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
}
.stPlotlyChart { border-radius: 12px; overflow: hidden; }
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.03);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: .5rem 1.25rem;
    font-size: .8rem;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.08) !important;
    color: #fff !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _q(sql: str) -> pd.DataFrame:
    """Run SQL query and return DataFrame."""
    conn = psycopg2.connect(CONN)
    try:
        df = pd.read_sql_query(sql, conn)
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300)
def _load(days: int = 30) -> pd.DataFrame:
    return _q(
        f"SELECT * FROM daily_metrics "
        f"WHERE date >= CURRENT_DATE - INTERVAL '{days} days' "
        f"ORDER BY date DESC"
    )


# Plotly layout template
PL = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="rgba(255,255,255,0.7)", size=12),
    margin=dict(l=0, r=0, t=30, b=0),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="rgba(255,255,255,0.5)"),
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
    ),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)"),
)


def _trend(df, m):
    """Return (pct, arrow, css_class) for week-over-week change."""
    if m not in df.columns or len(df) < 7:
        return 0, "→", "fl"
    recent = df[m].head(7).mean()
    prev = df[m].iloc[7:14].mean() if len(df) >= 14 else df[m].mean()
    if pd.isna(recent) or pd.isna(prev) or prev == 0:
        return 0, "→", "fl"
    pct = (recent - prev) / abs(prev) * 100
    if abs(pct) < 1:
        return pct, "→", "fl"
    return pct, ("↑" if pct > 0 else "↓"), ("up" if pct > 0 else "dn")


def _tile(label, value, unit="", dpct=0, arrow="→", cls="fl", accent="g", invert=False):
    """Render a metric tile HTML."""
    if invert:
        cls = {"up": "dn", "dn": "up"}.get(cls, cls)
    dh = ""
    if abs(dpct) >= 0.5:
        dh = f'<div class="md {cls}">{arrow} {abs(dpct):.1f}% vs prev week</div>'
    return (
        f'<div class="mt a{accent}">'
        f'<div class="ml">{label}</div>'
        f'<div class="mv">{value}<span class="mu"> {unit}</span></div>'
        f'{dh}</div>'
    )


def _bar(label, value, mx=100):
    """Render a recovery bar HTML."""
    pct = min(value / mx * 100, 100) if mx else 0
    if pct >= 70:
        clr = "linear-gradient(90deg,#00f5a0,#00d9f5)"
    elif pct >= 40:
        clr = "linear-gradient(90deg,#f5c542,#f5a623)"
    else:
        clr = "linear-gradient(90deg,#f5576c,#c62828)"
    return (
        f'<div style="margin-bottom:1rem;">'
        f'<div style="display:flex;justify-content:space-between;margin-bottom:.25rem;">'
        f'<span style="font-size:.75rem;color:rgba(255,255,255,0.5);font-weight:500;">{label}</span>'
        f'<span style="font-size:.75rem;color:#fff;font-weight:700;">{value:.0f}</span></div>'
        f'<div class="rb-bg"><div class="rb-f" style="width:{pct}%;background:{clr};"></div></div></div>'
    )


def _dot(v, hi=70, lo=40):
    """Render a status dot (green/yellow/red)."""
    c = "g" if v >= hi else ("y" if v >= lo else "r")
    return f'<span class="sd {c}"></span>'


# ═══════════════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════════════


def main():
    with st.sidebar:
        st.markdown(
            '<div style="padding:1.5rem .5rem 1rem;">'
            '<div style="font-size:1.1rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">'
            '⚡ GARMIN<span style="color:#00f5a0;"> INTELLIGENCE</span></div>'
            '<div style="font-size:.6rem;color:rgba(255,255,255,0.25);text-transform:uppercase;'
            'letter-spacing:.15em;margin-top:.25rem;">Personal Health Analytics</div></div>',
            unsafe_allow_html=True,
        )
        page = st.selectbox(
            "Nav",
            ["Overview", "Trends", "Deep Dive", "AI Chat", "AI Analysis", "Goals"],
            label_visibility="collapsed",
        )
        days = st.slider("Days", 7, 90, 30, label_visibility="collapsed")

    try:
        df = _load(days)
        if df.empty:
            st.error("No data. Run `python src/weekly_sync.py` first.")
            return
        {
            "Overview": pg_overview,
            "Trends": pg_trends,
            "Deep Dive": pg_dive,
            "AI Chat": pg_chat,
            "AI Analysis": pg_ai,
            "Goals": pg_goals,
        }[page](df)
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check database connection and data.")


# ══════════════════════════════════════════════════════════════
#  1. OVERVIEW
# ══════════════════════════════════════════════════════════════


def pg_overview(df):
    lat = df.iloc[0]
    st.markdown(
        f'<div class="pt">Dashboard</div>'
        f'<div class="ps">Latest: {lat.get("date", "")}</div>',
        unsafe_allow_html=True,
    )

    # ── Hero metric tiles ──
    cols = st.columns(4, gap="medium")
    hero_metrics = [
        ("Resting HR", "resting_hr", "bpm", "g", True),
        ("HRV", "hrv_last_night", "ms", "b", False),
        ("Sleep Score", "sleep_score", "", "c", False),
        ("Stress", "stress_level", "", "a", True),
    ]
    for col, (lb, key, un, acc, inv) in zip(cols, hero_metrics):
        v = lat.get(key)
        p, a, c = _trend(df, key)
        with col:
            st.markdown(
                _tile(lb, f"{v:.0f}" if pd.notna(v) else "—", un, p, a, c, acc, inv),
                unsafe_allow_html=True,
            )

    # ── Recovery status ──
    st.markdown('<div class="sh">Recovery Status</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1], gap="medium")
    with c1:
        bars_html = ""
        for lb, k in [
            ("Training Readiness", "training_readiness"),
            ("Body Battery Peak", "bb_peak"),
            ("Sleep Score", "sleep_score"),
        ]:
            v = lat.get(k)
            if pd.notna(v):
                bars_html += _bar(lb, v)
        st.markdown(f'<div class="gc">{bars_html}</div>', unsafe_allow_html=True)

    with c2:
        tr = lat.get("training_readiness")
        if pd.notna(tr):
            if tr >= 70:
                rec = "Ready to train hard"
            elif tr >= 40:
                rec = "Moderate intensity OK"
            else:
                rec = "Recovery day recommended"
            st.markdown(
                f'<div class="gc">'
                f'<div class="ml">Today\'s Recommendation</div>'
                f'<div style="margin:1rem 0;">{_dot(tr)}'
                f'<span style="color:#fff;font-weight:600;font-size:.9rem;">{rec}</span></div>'
                f'<div style="font-size:.75rem;color:rgba(255,255,255,0.35);">'
                f'Based on readiness {tr:.0f}, HRV, and recovery metrics</div></div>',
                unsafe_allow_html=True,
            )

    # ── 7-Day Trend chart ──
    st.markdown('<div class="sh">7-Day Trend</div>', unsafe_allow_html=True)
    l7 = df.head(7).sort_values("date")
    fig = go.Figure()
    for cn, nm, clr in [
        ("training_readiness", "Readiness", "#00f5a0"),
        ("sleep_score", "Sleep", "#667eea"),
        ("hrv_last_night", "HRV", "#4facfe"),
    ]:
        if cn in l7.columns:
            fig.add_trace(go.Scatter(
                x=l7["date"], y=l7[cn], name=nm, mode="lines+markers",
                line=dict(color=clr, width=2.5), marker=dict(size=5),
            ))
    fig.update_layout(**PL, height=280)
    st.plotly_chart(fig, use_container_width=True)

    # ── Recent activities ──
    st.markdown('<div class="sh">Recent Activities</div>', unsafe_allow_html=True)
    try:
        acts = _q(
            "SELECT date, activity_name, activity_type, "
            "ROUND(duration_sec/60.0) AS min, ROUND(distance_m/1000.0,1) AS km, "
            "average_hr AS hr, calories AS cal "
            "FROM activities WHERE date >= CURRENT_DATE - INTERVAL '7 days' "
            "ORDER BY date DESC LIMIT 5"
        )
        if not acts.empty:
            rows_html = ""
            for _, r in acts.iterrows():
                nm = r.get("activity_name") or r.get("activity_type") or "Activity"
                parts = []
                for k, fmt in [("min", "{:.0f}m"), ("km", "{:.1f}km"), ("hr", "♥{:.0f}"), ("cal", "{:.0f}cal")]:
                    v = r.get(k)
                    if pd.notna(v) and v:
                        parts.append(fmt.format(v))
                rows_html += (
                    f'<div class="ar">'
                    f'<div style="flex:1;">'
                    f'<div style="color:#fff;font-weight:600;font-size:.85rem;">{nm}</div>'
                    f'<div style="color:rgba(255,255,255,0.3);font-size:.7rem;">{r.get("date", "")}</div></div>'
                    f'<div style="color:rgba(255,255,255,0.5);font-size:.8rem;">{" · ".join(parts)}</div></div>'
                )
            st.markdown(
                f'<div class="gc" style="padding:.75rem;">{rows_html}</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
#  2. TRENDS — rebuilt from scratch, no more awful heatmap
# ══════════════════════════════════════════════════════════════


def pg_trends(df):
    st.markdown(
        '<div class="pt">Trends & Analysis</div>'
        '<div class="ps">Your body\'s story, told through data</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["Recovery", "Training", "Sleep"])

    # ── Recovery Tab ──
    with tab1:
        d = _load(30).sort_values("date")
        if d.empty:
            st.info("No data.")
            return

        # Dual-axis: HRV (area) + Resting HR (line)
        st.markdown('<div class="sh">HRV vs Resting Heart Rate</div>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "hrv_last_night" in d.columns:
            fig.add_trace(
                go.Scatter(
                    x=d["date"], y=d["hrv_last_night"], name="HRV",
                    fill="tozeroy", fillcolor="rgba(0,245,160,0.08)",
                    line=dict(color="#00f5a0", width=2),
                ),
                secondary_y=False,
            )
        if "resting_hr" in d.columns:
            fig.add_trace(
                go.Scatter(
                    x=d["date"], y=d["resting_hr"], name="Resting HR",
                    line=dict(color="#f5576c", width=2, dash="dot"),
                ),
                secondary_y=True,
            )
        fig.update_layout(**PL, height=300)
        fig.update_yaxes(title_text="HRV (ms)", secondary_y=False, gridcolor="rgba(255,255,255,0.04)")
        fig.update_yaxes(title_text="RHR (bpm)", secondary_y=True, gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, use_container_width=True)

        # Stress vs Body Battery
        st.markdown('<div class="sh">Stress vs Body Battery</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if "stress_level" in d.columns:
            fig.add_trace(go.Scatter(
                x=d["date"], y=d["stress_level"], name="Stress",
                fill="tozeroy", fillcolor="rgba(245,87,108,0.08)",
                line=dict(color="#f5576c", width=2),
            ))
        for col_name, label, color, dash in [
            ("bb_peak", "BB Peak", "#00f5a0", None),
            ("bb_low", "BB Low", "#f5c542", "dot"),
        ]:
            if col_name in d.columns:
                fig.add_trace(go.Scatter(
                    x=d["date"], y=d[col_name], name=label,
                    line=dict(color=color, width=2, dash=dash),
                ))
        fig.update_layout(**PL, height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Training Readiness
        if "training_readiness" in d.columns:
            st.markdown('<div class="sh">Training Readiness</div>', unsafe_allow_html=True)
            fig = go.Figure()
            colors = [
                "#f5576c" if v < 40 else ("#f5c542" if v < 70 else "#00f5a0")
                for v in d["training_readiness"].fillna(0)
            ]
            fig.add_trace(go.Bar(
                x=d["date"], y=d["training_readiness"],
                marker_color=colors, name="Readiness",
            ))
            ma = d["training_readiness"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=d["date"], y=ma, name="7d avg",
                line=dict(color="#fff", width=2, dash="dot"),
            ))
            fig.update_layout(**PL, height=250)
            st.plotly_chart(fig, use_container_width=True)

    # ── Training Tab ──
    with tab2:
        try:
            acts = _q(
                "SELECT date, activity_type, activity_name, "
                "duration_sec/60.0 AS mins, distance_m/1000.0 AS km, "
                "average_hr, max_hr, calories, "
                "aerobic_training_effect AS aero_te, "
                "anaerobic_training_effect AS anaero_te "
                "FROM activities "
                "WHERE date >= CURRENT_DATE - INTERVAL '60 days' "
                "ORDER BY date"
            )
        except Exception:
            acts = pd.DataFrame()

        if acts.empty:
            st.info("No activity data yet.")
        else:
            # Weekly volume stacked by type
            st.markdown('<div class="sh">Training Volume</div>', unsafe_allow_html=True)
            acts["week"] = pd.to_datetime(acts["date"]).dt.to_period("W").dt.start_time
            weekly = acts.groupby(["week", "activity_type"])["mins"].sum().reset_index()
            types = weekly["activity_type"].unique()
            palette = [
                "#00f5a0", "#667eea", "#f5576c", "#4facfe",
                "#f5c542", "#f093fb", "#764ba2", "#00d9f5",
            ]
            fig = go.Figure()
            for i, atype in enumerate(types):
                sub = weekly[weekly["activity_type"] == atype]
                fig.add_trace(go.Bar(
                    x=sub["week"], y=sub["mins"], name=str(atype),
                    marker_color=palette[i % len(palette)],
                ))
            fig.update_layout(
                **PL, height=300, barmode="stack",
                yaxis_title="Minutes",
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # HR by activity — bubble chart
            st.markdown('<div class="sh">Heart Rate by Activity</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=acts["mins"], y=acts["average_hr"], mode="markers",
                marker=dict(
                    size=8,
                    color=acts["calories"].fillna(0),
                    colorscale=[[0, "#667eea"], [1, "#f5576c"]],
                    showscale=True,
                    colorbar=dict(
                        title="Cal",
                        titlefont=dict(color="rgba(255,255,255,0.5)"),
                    ),
                ),
                text=acts["activity_name"],
                hovertemplate=(
                    "%{text}<br>%{x:.0f}min · ♥%{y:.0f}"
                    "<br>%{marker.color:.0f} cal<extra></extra>"
                ),
            ))
            fig.update_layout(
                **PL, height=300,
                xaxis_title="Duration (min)", yaxis_title="Avg HR (bpm)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Training effects — horizontal bars
            if "aero_te" in acts.columns:
                st.markdown('<div class="sh">Training Effects</div>', unsafe_allow_html=True)
                recent = acts.dropna(subset=["aero_te"]).tail(10).sort_values("date", ascending=True)
                if not recent.empty:
                    labels = (
                        recent["activity_name"].fillna(recent["activity_type"].fillna("Activity")).astype(str)
                        + "  " + recent["date"].astype(str)
                    )
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        y=labels, x=recent["aero_te"], name="Aerobic",
                        orientation="h", marker_color="#00f5a0",
                    ))
                    fig.add_trace(go.Bar(
                        y=labels, x=recent["anaero_te"], name="Anaerobic",
                        orientation="h", marker_color="#667eea",
                    ))
                    fig.update_layout(
                        **PL, height=max(250, len(recent) * 40), barmode="group",
                        xaxis_title="Training Effect (0-5)",
                        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ── Sleep Tab ──
    with tab3:
        d = _load(30).sort_values("date")
        has_sleep = all(
            c in d.columns for c in ["deep_sleep_sec", "rem_sleep_sec", "sleep_seconds"]
        )
        if not has_sleep:
            st.info("Sleep breakdown data not available.")
            return

        # Sleep architecture stacked bar
        st.markdown('<div class="sh">Sleep Architecture</div>', unsafe_allow_html=True)
        sd = d[["date", "deep_sleep_sec", "rem_sleep_sec", "sleep_seconds"]].dropna()
        if not sd.empty:
            sd = sd.copy()
            sd["deep_h"] = sd["deep_sleep_sec"] / 3600
            sd["rem_h"] = sd["rem_sleep_sec"] / 3600
            sd["light_h"] = (
                (sd["sleep_seconds"] - sd["deep_sleep_sec"] - sd["rem_sleep_sec"]) / 3600
            ).clip(lower=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sd["date"], y=sd["deep_h"], name="Deep", marker_color="#667eea"))
            fig.add_trace(go.Bar(x=sd["date"], y=sd["rem_h"], name="REM", marker_color="#f093fb"))
            fig.add_trace(go.Bar(
                x=sd["date"], y=sd["light_h"], name="Light",
                marker_color="rgba(255,255,255,0.15)",
            ))
            fig.update_layout(**PL, height=300, barmode="stack", yaxis_title="Hours")
            st.plotly_chart(fig, use_container_width=True)

        # Sleep score trend
        if "sleep_score" in d.columns:
            st.markdown('<div class="sh">Sleep Score</div>', unsafe_allow_html=True)
            fig = go.Figure()
            colors = [
                "#f5576c" if v < 60 else ("#f5c542" if v < 80 else "#00f5a0")
                for v in d["sleep_score"].fillna(0)
            ]
            fig.add_trace(go.Bar(
                x=d["date"], y=d["sleep_score"], marker_color=colors, name="Score",
            ))
            ma = d["sleep_score"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=d["date"], y=ma, name="7d avg",
                line=dict(color="#fff", width=2, dash="dot"),
            ))
            fig.update_layout(**PL, height=250)
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  3. DEEP DIVE
# ══════════════════════════════════════════════════════════════


def pg_dive(df):
    st.markdown(
        '<div class="pt">Deep Dive</div>'
        '<div class="ps">Single-metric analysis with distribution</div>',
        unsafe_allow_html=True,
    )

    avail = [
        c for c in [
            "resting_hr", "hrv_last_night", "sleep_score", "stress_level",
            "bb_peak", "bb_drained", "training_readiness", "total_steps",
            "deep_sleep_sec", "rem_sleep_sec", "bb_charged",
        ]
        if c in df.columns
    ]
    if not avail:
        st.warning("No metrics available.")
        return

    metric = st.selectbox("Metric", avail, label_visibility="collapsed")
    vals = df[metric].dropna()
    if vals.empty:
        st.warning("No data for this metric.")
        return

    # Stats tiles
    cols = st.columns(4, gap="medium")
    for col, (lb, v), ac in zip(
        cols,
        [("Mean", vals.mean()), ("Median", vals.median()), ("Min", vals.min()), ("Max", vals.max())],
        ["g", "b", "a", "c"],
    ):
        with col:
            st.markdown(_tile(lb, f"{v:.1f}", accent=ac), unsafe_allow_html=True)

    # Time series
    cdf = df[["date", metric]].dropna().sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf[metric], mode="lines+markers", name="Actual",
        line=dict(color="#00f5a0", width=2), marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=cdf["date"], y=cdf[metric].rolling(7, min_periods=1).mean(),
        mode="lines", name="7d avg",
        line=dict(color="#667eea", width=2, dash="dot"),
    ))
    fig.update_layout(**PL, height=320)
    st.plotly_chart(fig, use_container_width=True)

    # Distribution
    st.markdown('<div class="sh">Distribution</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=vals, nbinsx=20,
        marker=dict(color="rgba(0,245,160,0.4)", line=dict(color="#00f5a0", width=1)),
    ))
    fig.update_layout(**PL, height=220, bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  4. AI CHAT
# ══════════════════════════════════════════════════════════════


def pg_chat(df):
    st.markdown(
        '<div class="pt">Ask Your Data</div>'
        '<div class="ps">Chat with an AI agent that has live access to your health database</div>',
        unsafe_allow_html=True,
    )

    if not _api_key():
        st.warning("Set GOOGLE_API_KEY in Streamlit secrets to enable AI features.")
        return

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Render history
    for m in st.session_state.chat:
        if m["role"] == "user":
            st.markdown(
                f'<div class="cm cu">{html.escape(m["text"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="cm ca"><div class="cn">⚡ Health Intelligence</div>{m["text"]}</div>',
                unsafe_allow_html=True,
            )

    inp = st.chat_input("Ask anything about your health data...")
    if inp:
        st.session_state.chat.append({"role": "user", "text": inp})
        st.markdown(
            f'<div class="cm cu">{html.escape(inp)}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Thinking..."):
            try:
                resp = _chat_agent(inp)
                st.session_state.chat.append({"role": "agent", "text": resp})
                st.markdown(
                    f'<div class="cm ca"><div class="cn">⚡ Health Intelligence</div>{resp}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"Agent error: {e}")


def _chat_agent(question: str) -> str:
    """Single agent that answers health questions via SQL."""
    from crewai import Agent, Task, Crew, Process
    from enhanced_agents import (
        run_sql_query, calculate_correlation, find_best_days,
        analyze_pattern, _get_llm,
    )

    agent = Agent(
        role="Health Data Analyst",
        goal="Answer the user's health question accurately using their Garmin data",
        backstory=(
            "You are a concise health data analyst with direct SQL access "
            "to the user's Garmin health database (PostgreSQL). Tables: daily_metrics "
            "(date, resting_hr, hrv_last_night, sleep_score, stress_level, "
            "training_readiness, bb_charged, bb_drained, bb_peak, total_steps, "
            "deep_sleep_sec, rem_sleep_sec, sleep_seconds, weight_kg, etc.), "
            "activities (date, activity_name, activity_type, duration_sec, distance_m, "
            "average_hr, max_hr, calories), body_battery_events. "
            "Use PostgreSQL syntax. Be specific with numbers. Keep answers concise."
        ),
        verbose=False,
        allow_delegation=False,
        tools=[run_sql_query, calculate_correlation, find_best_days, analyze_pattern],
        llm=_get_llm(),
    )
    task = Task(
        description=f"Answer this question about the user's health data: {question}",
        agent=agent,
        expected_output="A concise, data-backed answer",
    )
    result = Crew(
        agents=[agent], tasks=[task],
        process=Process.sequential, verbose=False,
    ).kickoff()
    raw = getattr(result, "raw", str(result))
    return raw.replace("\n", "<br>")


# ══════════════════════════════════════════════════════════════
#  5. AI ANALYSIS — all 9 agents, parallel
# ══════════════════════════════════════════════════════════════


def pg_ai(df):
    st.markdown(
        '<div class="pt">AI Analysis</div>'
        '<div class="ps">9 specialized agents analyze your data in parallel</div>',
        unsafe_allow_html=True,
    )

    if not _api_key():
        st.warning("Set GOOGLE_API_KEY in Streamlit secrets to enable AI features.")
        return

    # Show last saved summary
    try:
        row = _q(
            "SELECT summary_text, created_at "
            "FROM weekly_summaries ORDER BY created_at DESC LIMIT 1"
        )
        if not row.empty:
            ts = row.iloc[0]["created_at"]
            txt = row.iloc[0]["summary_text"][:5000]
            with st.expander(f"Last saved summary — {ts}", expanded=False):
                st.markdown(
                    f'<div style="color:rgba(255,255,255,0.75);font-size:.82rem;'
                    f'line-height:1.7;white-space:pre-wrap;">{txt}</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    st.markdown('<div class="sh">Run New Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.75rem;color:rgba(255,255,255,0.35);margin-bottom:1rem;">'
        '<strong style="color:rgba(255,255,255,0.5);">Cost:</strong> ~$0.02 per run '
        '(Gemini 2.5 Flash). Even daily that\'s $0.60/month.</div>',
        unsafe_allow_html=True,
    )

    mode = st.selectbox(
        "Mode",
        [
            "Full Analysis — 9 agents in parallel (~30s)",
            "Goal Analysis — 2 agents (~15s)",
        ],
        label_visibility="collapsed",
    )

    goal = ""
    if "Goal" in mode:
        goal = st.text_input("Your goal", "Improve HRV by 10%")

    if st.button("Run Analysis", type="primary"):
        t0 = time.time()
        agents_obj = _get_agents()

        if "Full" in mode:
            with st.spinner("Running all 9 agents in parallel..."):
                result = _run_9_parallel(agents_obj)
        else:
            with st.spinner("Analyzing goal..."):
                result = str(agents_obj.run_goal_analysis(goal))

        elapsed = time.time() - t0

        st.markdown(
            f'<div style="text-align:center;font-size:.7rem;'
            f'color:rgba(255,255,255,0.3);margin:1rem 0;">'
            f'Completed in {elapsed:.0f}s</div>',
            unsafe_allow_html=True,
        )

        _render_analysis(result)


def _render_analysis(result: str):
    """Render analysis result as styled insight cards."""
    if "=" * 50 in result:
        sections = result.split("=" * 50)
        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue
            lines = sec.split("\n", 1)
            title = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            if not title:
                continue
            upper = title.upper()
            if "PATTERN" in upper or "HIDDEN" in upper:
                color = "#00f5a0"
            elif "MATRIX" in upper or "CORRELATION" in upper:
                color = "#667eea"
            elif "RECOVERY" in upper:
                color = "#f5576c"
            elif "TREND" in upper or "FORECAST" in upper:
                color = "#4facfe"
            elif "BOTTLENECK" in upper or "WEAKNESS" in upper or "ACTION" in upper:
                color = "#f5c542"
            elif "SLEEP" in upper:
                color = "#f093fb"
            else:
                color = "rgba(255,255,255,0.5)"
            st.markdown(
                f'<div class="ic">'
                f'<div class="ic-t" style="color:{color};">{html.escape(title)}</div>'
                f'<div class="ic-b">{html.escape(body)}</div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            f'<div class="gc"><div style="color:rgba(255,255,255,0.85);font-size:.85rem;'
            f'line-height:1.7;white-space:pre-wrap;">{html.escape(str(result))}</div></div>',
            unsafe_allow_html=True,
        )


@st.cache_resource
def _get_agents():
    from enhanced_agents import AdvancedHealthAgents
    return AdvancedHealthAgents()


def _run_9_parallel(agents_obj) -> str:
    """Run all 9 agents grouped in 3 batches of 3 threads each."""
    from crewai import Task, Crew, Process

    results: dict[str, str] = {}

    def _run(name: str, agent, desc: str, exp: str):
        try:
            task = Task(description=desc, agent=agent, expected_output=exp)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            results[name] = str(crew.kickoff())
        except Exception as e:
            results[name] = f"Error: {e}"

    batch1 = [
        (
            "CORRELATION MATRIX INTERPRETATION",
            agents_obj.matrix_analyst,
            "Interpret the latest correlation data. Query daily_metrics for the last 14 days. "
            "Which metric pairs show strong Pearson r (>0.5)? What's real vs noise?",
            "Key correlations with interpretation",
        ),
        (
            "CROSS-TIMEFRAME STABILITY",
            agents_obj.matrix_comparator,
            "Compare metric relationships across 7d, 14d, and 30d windows. "
            "Which correlations are stable (ROBUST) vs just recent noise (EMERGING)?",
            "Stability classification of key patterns",
        ),
        (
            "HIDDEN PATTERNS",
            agents_obj.pattern_detective,
            "Find the top 3 non-obvious patterns in the last 14 days. "
            "Look for day-of-week effects, delayed correlations, unexpected metric interactions.",
            "3 hidden patterns with evidence",
        ),
    ]

    batch2 = [
        (
            "PERFORMANCE OPTIMIZATION",
            agents_obj.performance_optimizer,
            "Analyze peak vs poor performance days in the last 14 days. "
            "What conditions predict great training days? Provide specific recommendations.",
            "Peak performance conditions + recommendations",
        ),
        (
            "RECOVERY ASSESSMENT",
            agents_obj.recovery_specialist,
            "Assess current recovery status: HRV trend, sleep quality, body battery patterns, "
            "resting HR. Signs of overtraining? Fatigue accumulation?",
            "Recovery status with actionable recommendation",
        ),
        (
            "TRENDS & FORECASTS",
            agents_obj.trend_forecaster,
            "Project current trends forward. Is HRV improving or declining? "
            "Sleep quality? Stress? Any early warning signs? What happens if trends continue?",
            "Trend projections with early warnings",
        ),
    ]

    batch3 = [
        (
            "LIFESTYLE CONNECTIONS",
            agents_obj.lifestyle_analyst,
            "Which lifestyle factors (steps, stress patterns, training timing) most impact "
            "health outcomes? What differentiates good days from bad days?",
            "Lifestyle impact analysis",
        ),
        (
            "SLEEP ANALYSIS",
            agents_obj.sleep_analyst,
            "Analyze sleep architecture: deep/REM/light as % of total vs clinical norms. "
            "Consistency (CV>15% = inconsistent). Impact on next-day readiness and body battery.",
            "Sleep architecture analysis with clinical comparison",
        ),
        (
            "BOTTLENECK & QUICK WINS",
            agents_obj.weakness_identifier,
            "Synthesize everything: What is the #1 limiting factor right now? "
            "What are 3 quick wins for the next week? Be specific and actionable.",
            "1 bottleneck + 3 quick wins",
        ),
    ]

    for batch in [batch1, batch2, batch3]:
        threads = [threading.Thread(target=_run, args=args) for args in batch]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)

    labels = [b[0] for b in batch1 + batch2 + batch3]
    sections = []
    for label in labels:
        if label in results:
            sections.append(f"{'=' * 50}\n  {label}\n{'=' * 50}\n\n{results[label]}")

    return "\n\n".join(sections) if sections else "Analysis timed out."


# ══════════════════════════════════════════════════════════════
#  6. GOALS — informative with real data and benchmarks
# ══════════════════════════════════════════════════════════════


def pg_goals(df):
    st.markdown(
        '<div class="pt">Goals & Progress</div>'
        '<div class="ps">8-week trend analysis with benchmarks</div>',
        unsafe_allow_html=True,
    )

    try:
        weekly = _q("""
            SELECT DATE_TRUNC('week', date) AS week,
                   AVG(resting_hr) AS rhr,
                   AVG(hrv_last_night) AS hrv,
                   AVG(sleep_score) AS sleep,
                   AVG(training_readiness) AS readiness,
                   AVG(stress_level) AS stress,
                   AVG(bb_peak) AS bb_peak,
                   AVG(total_steps) AS steps,
                   COUNT(*) AS days
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '56 days'
            GROUP BY week ORDER BY week
        """)
    except Exception:
        st.info("Not enough data yet.")
        return

    if len(weekly) < 2:
        st.info("Need at least 2 weeks of data.")
        return

    # ── This Week vs 8 Weeks Ago ──
    latest = weekly.iloc[-1]
    first = weekly.iloc[0]

    st.markdown('<div class="sh">This Week vs 8 Weeks Ago</div>', unsafe_allow_html=True)
    cols = st.columns(4, gap="medium")
    comparisons = [
        ("HRV", "hrv", "ms", "b", False),
        ("Sleep", "sleep", "", "c", False),
        ("Readiness", "readiness", "", "g", False),
        ("Resting HR", "rhr", "bpm", "a", True),
    ]
    for col, (lb, key, un, ac, inv) in zip(cols, comparisons):
        lv = latest.get(key)
        fv = first.get(key)
        if pd.notna(lv) and pd.notna(fv) and fv != 0:
            pct = (lv - fv) / abs(fv) * 100
            arrow = "↑" if pct > 0 else ("↓" if pct < 0 else "→")
            cls = "up" if pct > 0 else ("dn" if pct < 0 else "fl")
        else:
            pct, arrow, cls = 0, "→", "fl"
        with col:
            st.markdown(
                _tile(lb, f"{lv:.1f}" if pd.notna(lv) else "—", un, pct, arrow, cls, ac, inv),
                unsafe_allow_html=True,
            )

    # ── Weekly trend sparklines with benchmarks ──
    st.markdown('<div class="sh">Weekly Trends</div>', unsafe_allow_html=True)

    metrics_cfg = [
        ("hrv", "HRV", "#00f5a0", "ms",
         "Higher = better recovery. Top athletes: 50-100ms.", False),
        ("sleep", "Sleep Score", "#667eea", "/100",
         "Clinical target: >80. <60 = poor quality.", False),
        ("readiness", "Training Readiness", "#4facfe", "/100",
         ">70 = go hard. 40-70 = moderate. <40 = rest.", False),
        ("rhr", "Resting HR", "#f5576c", "bpm",
         "Lower = fitter. Watch for sudden increases (illness/overtraining).", True),
        ("stress", "Stress", "#f5c542", "",
         "Garmin avg: 25-50. <25 = calm. >50 = high stress.", True),
        ("bb_peak", "Body Battery Peak", "#f093fb", "/100",
         ">80 = fully recharged. <50 = inadequate recovery.", False),
        ("steps", "Daily Steps", "#00d9f5", "",
         "WHO target: 8,000-10,000. Affects BB drain pattern.", False),
    ]

    for key, label, color, unit, note, lower_better in metrics_cfg:
        if key not in weekly.columns:
            continue
        vals = weekly[key].dropna()
        if vals.empty:
            continue

        latest_v = vals.iloc[-1]
        first_v = vals.iloc[0]
        change = latest_v - first_v
        improving = (change < 0) if lower_better else (change > 0)
        trend_word = "improving" if improving else ("declining" if change != 0 else "stable")
        trend_color = (
            "#00f5a0" if improving
            else ("#f5576c" if change != 0 else "rgba(255,255,255,0.3)")
        )

        c1, c2 = st.columns([3, 1], gap="medium")
        with c1:
            # Compute fillcolor with transparency
            r_hex = int(color[1:3], 16)
            g_hex = int(color[3:5], 16)
            b_hex = int(color[5:7], 16)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly["week"], y=weekly[key],
                mode="lines+markers", name=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
                fill="tozeroy",
                fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.05)",
            ))
            fig.update_layout(**PL, height=160, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(
                f'<div class="gc" style="height:100%;display:flex;flex-direction:column;'
                f'justify-content:center;">'
                f'<div class="ml">{label}</div>'
                f'<div style="font-size:1.5rem;font-weight:800;color:#fff;">'
                f'{latest_v:.1f}<span style="font-size:.7rem;color:rgba(255,255,255,0.35);"> '
                f'{unit}</span></div>'
                f'<div style="font-size:.7rem;font-weight:600;color:{trend_color};margin-top:.3rem;">'
                f'{trend_word} · {"+" if change >= 0 else ""}{change:.1f}</div>'
                f'<div style="font-size:.65rem;color:rgba(255,255,255,0.25);margin-top:.4rem;">'
                f'{note}</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
