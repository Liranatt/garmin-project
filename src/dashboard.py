"""
Garmin Health Intelligence — Dashboard v3
==========================================
Robust multi-page analytics dashboard with:
  • Multi-timeframe trend explorer (7d/14d/21d/30d/60d/90d)
  • Correlation heatmaps, lag-1 matrices, Markov diagrams
  • Date deep-dive with full-day report
  • Real-time agent chat with 9 specialists
  • Parallel 9-agent analysis
  • Goals & benchmark tracking

Run:  streamlit run src/dashboard.py
"""
from __future__ import annotations

import os, sys, time, threading, html, json, traceback
from datetime import datetime, timedelta, date as date_type

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
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
    initial_sidebar_state="expanded",
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
#  DESIGN SYSTEM — v3 glass-morphism
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
    background: rgba(13,17,23,0.97);
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255,255,255,0.5) !important;
    font-size: .7rem !important;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-weight: 600;
}

/* Glass card */
.gc {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}
.gc:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(255,255,255,0.12);
    transform: translateY(-1px);
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
    margin-bottom: .75rem;
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
.mt.ad::before { background: linear-gradient(90deg, #f5c542, #f5a623); }

.ml { font-size: .65rem; text-transform: uppercase; letter-spacing: .12em;
      color: rgba(255,255,255,0.4); font-weight: 600; margin-bottom: .35rem; }
.mv { font-size: 2rem; font-weight: 800; color: #fff; line-height: 1;
      margin-bottom: .25rem; letter-spacing: -0.02em; }
.mu { font-size: .85rem; font-weight: 400; color: rgba(255,255,255,0.35); }
.md { font-size: .75rem; font-weight: 600; margin-top: .4rem; }
.md.up { color: #00f5a0; }
.md.dn { color: #f5576c; }
.md.fl { color: rgba(255,255,255,0.3); }

/* Page titles */
.pt { font-size: 2.2rem; font-weight: 900; letter-spacing: -0.04em;
      color: #fff; text-align: center; margin-bottom: .1rem; }
.ps { font-size: .8rem; color: rgba(255,255,255,0.3); font-weight: 400;
      text-align: center; margin-bottom: 2rem; }

/* Section header */
.sh { font-size: .7rem; text-transform: uppercase; letter-spacing: .12em;
      color: rgba(255,255,255,0.3); font-weight: 700; margin: 2rem 0 1rem;
      padding-bottom: .5rem; border-bottom: 1px solid rgba(255,255,255,0.06); }

/* Status dot */
.sd { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
      margin-right: 6px; animation: pg 2s ease-in-out infinite; }
.sd.g { background: #00f5a0; box-shadow: 0 0 8px rgba(0,245,160,0.5); }
.sd.y { background: #f5c542; box-shadow: 0 0 8px rgba(245,197,66,0.5); }
.sd.r { background: #f5576c; box-shadow: 0 0 8px rgba(245,87,108,0.5); }
@keyframes pg { 0%,100%{opacity:1;} 50%{opacity:0.6;} }

/* Recovery bar */
.rb-bg { background: rgba(255,255,255,0.06); border-radius: 8px;
         height: 6px; width: 100%; margin-top: .5rem; overflow: hidden; }
.rb-f { height: 100%; border-radius: 8px; transition: width 1s ease; }

/* Chat */
.cm { padding: 1rem 1.25rem; border-radius: 12px; margin-bottom: .75rem;
      font-size: .9rem; line-height: 1.6; }
.cu { background: rgba(102,126,234,0.15); border: 1px solid rgba(102,126,234,0.2);
      margin-left: 3rem; }
.ca { background: rgba(0,245,160,0.06); border: 1px solid rgba(0,245,160,0.1);
      margin-right: 3rem; }
.cn { font-size: .65rem; text-transform: uppercase; letter-spacing: .1em;
      color: #00f5a0; font-weight: 700; margin-bottom: .35rem; }

/* Activity row */
.ar { display: flex; align-items: center; padding: .75rem 1rem;
      border-radius: 10px; background: rgba(255,255,255,0.02);
      border: 1px solid rgba(255,255,255,0.04); margin-bottom: .5rem;
      transition: background .2s; }
.ar:hover { background: rgba(255,255,255,0.05); }

/* Insight card */
.ic { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.06);
      border-radius: 14px; padding: 1.25rem; margin-bottom: .75rem; }
.ic-t { font-size: .65rem; text-transform: uppercase; letter-spacing: .1em;
        font-weight: 700; margin-bottom: .5rem; }
.ic-b { font-size: .85rem; color: rgba(255,255,255,0.75); line-height: 1.7;
        white-space: pre-wrap; }

/* Heatmap annotation override */
.hm-label { font-size: .7rem; color: rgba(255,255,255,0.5); margin-bottom: .25rem; }

/* Streamlit overrides */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
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
.stSlider > div { padding-left: .5rem; }
div[data-testid="stDateInput"] > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  DATA ACCESS LAYER
# ═══════════════════════════════════════════════════════════════


def _q(sql: str, params=None) -> pd.DataFrame:
    """Run read-only SQL query and return DataFrame."""
    conn = psycopg2.connect(CONN)
    try:
        df = pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
    return df


@st.cache_data(ttl=300)
def _load(days: int = 30) -> pd.DataFrame:
    return _q(
        "SELECT * FROM daily_metrics "
        "WHERE date >= CURRENT_DATE - INTERVAL '%s days' "
        "ORDER BY date DESC",
        params=(days,),
    )


@st.cache_data(ttl=300)
def _load_range(start: str, end: str) -> pd.DataFrame:
    return _q(
        "SELECT * FROM daily_metrics WHERE date >= %s AND date <= %s ORDER BY date",
        params=(start, end),
    )


@st.cache_data(ttl=300)
def _load_activities(days: int = 60) -> pd.DataFrame:
    return _q(
        "SELECT * FROM activities WHERE date >= CURRENT_DATE - INTERVAL '%s days' "
        "ORDER BY date DESC",
        params=(days,),
    )


@st.cache_data(ttl=600)
def _load_all_columns() -> list:
    """Get all numeric column names from daily_metrics."""
    sample = _q("SELECT * FROM daily_metrics LIMIT 1")
    return [c for c in sample.columns
            if c not in ("date", "created_at", "updated_at")
            and pd.api.types.is_numeric_dtype(sample[c])]


@st.cache_data(ttl=300)
def _get_date_range() -> tuple:
    """Return (min_date, max_date) from daily_metrics."""
    r = _q("SELECT MIN(date) AS mn, MAX(date) AS mx FROM daily_metrics")
    if r.empty:
        return None, None
    return r.iloc[0]["mn"], r.iloc[0]["mx"]


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

PALETTE = ["#00f5a0", "#667eea", "#f5576c", "#4facfe", "#f5c542",
           "#f093fb", "#764ba2", "#00d9f5", "#ff6b6b", "#48dbfb"]


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


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
    c = "g" if v >= hi else ("y" if v >= lo else "r")
    return f'<span class="sd {c}"></span>'


def _safe_fmt(v, fmt=".0f"):
    """Safely format a numeric value; return '—' if NaN."""
    if pd.isna(v):
        return "—"
    return f"{v:{fmt}}"


# ═══════════════════════════════════════════════════════════════
#  ROUTING
# ═══════════════════════════════════════════════════════════════

PAGES = [
    "Overview",
    "Trends",
    "Correlations",
    "Deep Dive",
    "Date Explorer",
    "Agent Chat",
    "Agent Analysis",
    "Goals",
]


def main():
    with st.sidebar:
        st.markdown(
            '<div style="padding:1.5rem .5rem 1rem;">'
            '<div style="font-size:1.1rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">'
            '⚡ GARMIN<span style="color:#00f5a0;"> INTELLIGENCE</span></div>'
            '<div style="font-size:.6rem;color:rgba(255,255,255,0.25);text-transform:uppercase;'
            'letter-spacing:.15em;margin-top:.25rem;">Personal Health Analytics v3</div></div>',
            unsafe_allow_html=True,
        )
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")

        st.markdown('<div class="sh">Settings</div>', unsafe_allow_html=True)
        days = st.slider("Default window (days)", 7, 180, 30)

    try:
        df = _load(days)
        if df.empty:
            st.error("No data. Run `python src/weekly_sync.py` first.")
            return
        {
            "Overview": pg_overview,
            "Trends": pg_trends,
            "Correlations": pg_correlations,
            "Deep Dive": pg_dive,
            "Date Explorer": pg_date_explorer,
            "Agent Chat": pg_chat,
            "Agent Analysis": pg_ai,
            "Goals": pg_goals,
        }[page](df, days)
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        with st.expander("Details"):
            st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════
#  1. OVERVIEW
# ══════════════════════════════════════════════════════════════


def pg_overview(df, days):
    lat = df.iloc[0]
    st.markdown(
        f'<div class="pt">Dashboard</div>'
        f'<div class="ps">Latest: {lat.get("date", "")}</div>',
        unsafe_allow_html=True,
    )

    # ── Hero tiles ──
    cols = st.columns(5, gap="medium")
    hero = [
        ("Resting HR", "resting_hr", "bpm", "g", True),
        ("HRV", "hrv_last_night", "ms", "b", False),
        ("Sleep", "sleep_score", "", "c", False),
        ("Stress", "stress_level", "", "a", True),
        ("Readiness", "training_readiness", "", "d", False),
    ]
    for col, (lb, key, un, acc, inv) in zip(cols, hero):
        v = lat.get(key)
        p, a, c = _trend(df, key)
        with col:
            st.markdown(
                _tile(lb, _safe_fmt(v), un, p, a, c, acc, inv),
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
            ("HRV", "hrv_last_night"),
        ]:
            v = lat.get(k)
            mx = 100 if k != "hrv_last_night" else 120
            if pd.notna(v):
                bars_html += _bar(lb, v, mx)
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
            stress = lat.get("stress_level", 0)
            acwr = lat.get("acwr")
            acwr_txt = ""
            if pd.notna(acwr):
                if acwr < 0.8:
                    acwr_txt = f"ACWR {acwr:.2f} — detraining risk"
                elif acwr <= 1.3:
                    acwr_txt = f"ACWR {acwr:.2f} — optimal zone"
                else:
                    acwr_txt = f"ACWR {acwr:.2f} — injury risk"
            st.markdown(
                f'<div class="gc">'
                f'<div class="ml">Today\'s Recommendation</div>'
                f'<div style="margin:.75rem 0;">{_dot(tr)}'
                f'<span style="color:#fff;font-weight:600;">{rec}</span></div>'
                f'<div style="font-size:.72rem;color:rgba(255,255,255,0.4);">'
                f'Readiness {tr:.0f} · Stress {_safe_fmt(stress)}'
                f'{"<br>" + acwr_txt if acwr_txt else ""}</div></div>',
                unsafe_allow_html=True,
            )

    # ── 7-Day Trend ──
    st.markdown('<div class="sh">7-Day Trend</div>', unsafe_allow_html=True)
    l7 = df.head(7).sort_values("date")
    fig = go.Figure()
    for cn, nm, clr in [
        ("training_readiness", "Readiness", "#00f5a0"),
        ("sleep_score", "Sleep", "#667eea"),
        ("hrv_last_night", "HRV", "#4facfe"),
        ("stress_level", "Stress", "#f5576c"),
    ]:
        if cn in l7.columns:
            fig.add_trace(go.Scatter(
                x=l7["date"], y=l7[cn], name=nm, mode="lines+markers",
                line=dict(color=clr, width=2.5), marker=dict(size=5),
            ))
    fig.update_layout(**PL, height=300)
    st.plotly_chart(fig, use_container_width=True)

    # ── Recent activities ──
    st.markdown('<div class="sh">Recent Activities</div>', unsafe_allow_html=True)
    try:
        acts = _q(
            "SELECT date, activity_name, activity_type, "
            "ROUND(duration_sec/60.0) AS min, ROUND(distance_m/1000.0,1) AS km, "
            "average_hr AS hr, calories AS cal "
            "FROM activities WHERE date >= CURRENT_DATE - INTERVAL '7 days' "
            "ORDER BY date DESC LIMIT 8"
        )
        if not acts.empty:
            rows_html = ""
            for _, r in acts.iterrows():
                nm = r.get("activity_name") or r.get("activity_type") or "Activity"
                parts = []
                for k, fmt in [("min", "{:.0f}m"), ("km", "{:.1f}km"),
                               ("hr", "♥{:.0f}"), ("cal", "{:.0f}cal")]:
                    v = r.get(k)
                    if pd.notna(v) and v:
                        parts.append(fmt.format(v))
                rows_html += (
                    f'<div class="ar">'
                    f'<div style="flex:1;">'
                    f'<div style="color:#fff;font-weight:600;font-size:.85rem;">{nm}</div>'
                    f'<div style="color:rgba(255,255,255,0.3);font-size:.7rem;">{r.get("date","")}</div></div>'
                    f'<div style="color:rgba(255,255,255,0.5);font-size:.8rem;">{" · ".join(parts)}</div></div>'
                )
            st.markdown(f'<div class="gc" style="padding:.75rem;">{rows_html}</div>',
                        unsafe_allow_html=True)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
#  2. TRENDS — multi-timeframe explorer
# ══════════════════════════════════════════════════════════════


def pg_trends(df, days):
    st.markdown(
        '<div class="pt">Trends</div>'
        '<div class="ps">Multi-timeframe trend explorer</div>',
        unsafe_allow_html=True,
    )

    # Timeframe selector
    windows = [7, 14, 21, 30, 60, 90]
    available_windows = [w for w in windows if w <= days]
    if not available_windows:
        available_windows = [7]

    tabs = st.tabs([f"{w}d" for w in available_windows] + ["Compare"])

    # Individual window tabs
    for i, w in enumerate(available_windows):
        with tabs[i]:
            wd = _load(w).sort_values("date")
            if wd.empty:
                st.info(f"No data for {w}-day window.")
                continue
            _render_trend_window(wd, w)

    # Comparison tab — overlay multiple windows
    with tabs[-1]:
        st.markdown('<div class="sh">Multi-Window Comparison</div>', unsafe_allow_html=True)
        metric_opts = [c for c in [
            "resting_hr", "hrv_last_night", "sleep_score", "stress_level",
            "training_readiness", "bb_charged", "bb_drained", "total_steps",
            "bb_peak", "vo2_max_running", "acwr",
        ] if c in df.columns]
        sel_metric = st.selectbox("Metric", metric_opts, key="trend_cmp_metric")

        # Stats per window
        stats_data = []
        for w in available_windows:
            wd = _load(w)
            if sel_metric in wd.columns:
                vals = wd[sel_metric].dropna()
                if not vals.empty:
                    stats_data.append({
                        "Window": f"{w}d",
                        "Mean": f"{vals.mean():.1f}",
                        "Std": f"{vals.std():.1f}",
                        "Min": f"{vals.min():.0f}",
                        "Max": f"{vals.max():.0f}",
                        "CV%": f"{(vals.std()/vals.mean()*100):.1f}" if vals.mean() != 0 else "—",
                    })
        if stats_data:
            st.dataframe(
                pd.DataFrame(stats_data), use_container_width=True, hide_index=True,
            )

        # Overlay chart
        fig = go.Figure()
        for i, w in enumerate(available_windows):
            wd = _load(w).sort_values("date")
            if sel_metric in wd.columns:
                fig.add_trace(go.Scatter(
                    x=wd["date"], y=wd[sel_metric],
                    name=f"{w}d", mode="lines",
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                ))
        fig.update_layout(**PL, height=350, title=sel_metric.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)


def _render_trend_window(wd, window_days):
    """Render trend charts for a single time window."""
    tab_r, tab_t, tab_s = st.tabs(["Recovery", "Training", "Sleep"])

    with tab_r:
        # HRV vs RHR dual-axis
        st.markdown('<div class="sh">HRV vs Resting Heart Rate</div>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if "hrv_last_night" in wd.columns:
            fig.add_trace(go.Scatter(
                x=wd["date"], y=wd["hrv_last_night"], name="HRV",
                fill="tozeroy", fillcolor="rgba(0,245,160,0.08)",
                line=dict(color="#00f5a0", width=2),
            ), secondary_y=False)
        if "resting_hr" in wd.columns:
            fig.add_trace(go.Scatter(
                x=wd["date"], y=wd["resting_hr"], name="RHR",
                line=dict(color="#f5576c", width=2, dash="dot"),
            ), secondary_y=True)
        fig.update_layout(**PL, height=280)
        fig.update_yaxes(title_text="HRV (ms)", secondary_y=False,
                         gridcolor="rgba(255,255,255,0.04)")
        fig.update_yaxes(title_text="RHR (bpm)", secondary_y=True,
                         gridcolor="rgba(255,255,255,0.04)")
        st.plotly_chart(fig, use_container_width=True)

        # Stress vs Body Battery
        st.markdown('<div class="sh">Stress vs Body Battery</div>', unsafe_allow_html=True)
        fig = go.Figure()
        if "stress_level" in wd.columns:
            fig.add_trace(go.Scatter(
                x=wd["date"], y=wd["stress_level"], name="Stress",
                fill="tozeroy", fillcolor="rgba(245,87,108,0.08)",
                line=dict(color="#f5576c", width=2),
            ))
        for cn, lb, clr, dash in [
            ("bb_peak", "BB Peak", "#00f5a0", None),
            ("bb_low", "BB Low", "#f5c542", "dot"),
        ]:
            if cn in wd.columns:
                fig.add_trace(go.Scatter(
                    x=wd["date"], y=wd[cn], name=lb,
                    line=dict(color=clr, width=2, dash=dash),
                ))
        fig.update_layout(**PL, height=280)
        st.plotly_chart(fig, use_container_width=True)

        # Training readiness bars
        if "training_readiness" in wd.columns:
            st.markdown('<div class="sh">Training Readiness</div>', unsafe_allow_html=True)
            fig = go.Figure()
            colors = ["#f5576c" if v < 40 else ("#f5c542" if v < 70 else "#00f5a0")
                      for v in wd["training_readiness"].fillna(0)]
            fig.add_trace(go.Bar(x=wd["date"], y=wd["training_readiness"],
                                 marker_color=colors, name="Readiness"))
            ma = wd["training_readiness"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=wd["date"], y=ma, name="7d avg",
                                     line=dict(color="#fff", width=2, dash="dot")))
            fig.update_layout(**PL, height=250)
            st.plotly_chart(fig, use_container_width=True)

    with tab_t:
        try:
            acts = _load_activities(window_days)
        except Exception:
            acts = pd.DataFrame()

        if acts.empty:
            st.info("No activity data.")
        else:
            # Weekly volume
            st.markdown('<div class="sh">Training Volume</div>', unsafe_allow_html=True)
            acts_c = acts.copy()
            if "duration_sec" in acts_c.columns:
                acts_c["mins"] = acts_c["duration_sec"] / 60.0
            else:
                acts_c["mins"] = 0
            acts_c["week"] = pd.to_datetime(acts_c["date"]).dt.to_period("W").dt.start_time
            weekly = acts_c.groupby(["week", "activity_type"])["mins"].sum().reset_index()
            types = weekly["activity_type"].unique()
            fig = go.Figure()
            for i, atype in enumerate(types):
                sub = weekly[weekly["activity_type"] == atype]
                fig.add_trace(go.Bar(x=sub["week"], y=sub["mins"], name=str(atype),
                                     marker_color=PALETTE[i % len(PALETTE)]))
            fig.update_layout(**PL, height=280, barmode="stack", yaxis_title="Minutes")
            st.plotly_chart(fig, use_container_width=True)

            # HR by activity
            st.markdown('<div class="sh">Heart Rate by Activity</div>', unsafe_allow_html=True)
            if "average_hr" in acts_c.columns and "mins" in acts_c.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=acts_c["mins"], y=acts_c["average_hr"], mode="markers",
                    marker=dict(size=8,
                                color=acts_c["calories"].fillna(0) if "calories" in acts_c.columns else 0,
                                colorscale=[[0, "#667eea"], [1, "#f5576c"]],
                                showscale=True, colorbar=dict(title="Cal")),
                    text=acts_c.get("activity_name", acts_c.get("activity_type", "")),
                    hovertemplate="%{text}<br>%{x:.0f}min · ♥%{y:.0f}<extra></extra>",
                ))
                fig.update_layout(**PL, height=280,
                                  xaxis_title="Duration (min)", yaxis_title="Avg HR")
                st.plotly_chart(fig, use_container_width=True)

            # Training effects
            if "aerobic_training_effect" in acts_c.columns:
                st.markdown('<div class="sh">Training Effects (recent 10)</div>',
                            unsafe_allow_html=True)
                recent = acts_c.dropna(subset=["aerobic_training_effect"]).tail(10).sort_values("date")
                if not recent.empty:
                    labels = (recent.get("activity_name", recent.get("activity_type", "")).astype(str)
                              + "  " + recent["date"].astype(str))
                    fig = go.Figure()
                    fig.add_trace(go.Bar(y=labels, x=recent["aerobic_training_effect"],
                                         name="Aerobic", orientation="h", marker_color="#00f5a0"))
                    if "anaerobic_training_effect" in recent.columns:
                        fig.add_trace(go.Bar(y=labels, x=recent["anaerobic_training_effect"],
                                             name="Anaerobic", orientation="h", marker_color="#667eea"))
                    fig.update_layout(**PL, height=max(250, len(recent)*40), barmode="group")
                    st.plotly_chart(fig, use_container_width=True)

    with tab_s:
        has_sleep = all(c in wd.columns for c in ["deep_sleep_sec", "rem_sleep_sec", "sleep_seconds"])
        if not has_sleep:
            st.info("No sleep architecture data.")
            return

        # Architecture stacked bar
        st.markdown('<div class="sh">Sleep Architecture</div>', unsafe_allow_html=True)
        sd = wd[["date", "deep_sleep_sec", "rem_sleep_sec", "sleep_seconds"]].dropna().copy()
        if not sd.empty:
            sd["deep_h"] = sd["deep_sleep_sec"] / 3600
            sd["rem_h"] = sd["rem_sleep_sec"] / 3600
            sd["light_h"] = ((sd["sleep_seconds"] - sd["deep_sleep_sec"] - sd["rem_sleep_sec"]) / 3600).clip(lower=0)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=sd["date"], y=sd["deep_h"], name="Deep", marker_color="#667eea"))
            fig.add_trace(go.Bar(x=sd["date"], y=sd["rem_h"], name="REM", marker_color="#f093fb"))
            fig.add_trace(go.Bar(x=sd["date"], y=sd["light_h"], name="Light",
                                 marker_color="rgba(255,255,255,0.15)"))
            fig.update_layout(**PL, height=280, barmode="stack", yaxis_title="Hours")
            st.plotly_chart(fig, use_container_width=True)

        # Sleep score trend
        if "sleep_score" in wd.columns:
            st.markdown('<div class="sh">Sleep Score</div>', unsafe_allow_html=True)
            fig = go.Figure()
            colors = ["#f5576c" if v < 60 else ("#f5c542" if v < 80 else "#00f5a0")
                      for v in wd["sleep_score"].fillna(0)]
            fig.add_trace(go.Bar(x=wd["date"], y=wd["sleep_score"], marker_color=colors))
            ma = wd["sleep_score"].rolling(7, min_periods=1).mean()
            fig.add_trace(go.Scatter(x=wd["date"], y=ma, name="7d avg",
                                     line=dict(color="#fff", width=2, dash="dot")))
            fig.update_layout(**PL, height=250)
            st.plotly_chart(fig, use_container_width=True)

        # Sleep stats
        st.markdown('<div class="sh">Sleep Stats</div>', unsafe_allow_html=True)
        if not sd.empty:
            total_h = sd["deep_h"] + sd["rem_h"] + sd["light_h"]
            deep_pct = (sd["deep_h"] / total_h * 100).mean()
            rem_pct = (sd["rem_h"] / total_h * 100).mean()
            light_pct = (sd["light_h"] / total_h * 100).mean()
            cols = st.columns(4)
            for col, (lb, v, norm) in zip(cols, [
                ("Deep %", deep_pct, "15-20%"),
                ("REM %", rem_pct, "20-25%"),
                ("Light %", light_pct, "50-60%"),
                ("Avg Hours", total_h.mean(), "7-9h"),
            ]):
                with col:
                    st.markdown(
                        f'<div class="mt ag">'
                        f'<div class="ml">{lb}</div>'
                        f'<div class="mv">{v:.1f}</div>'
                        f'<div style="font-size:.65rem;color:rgba(255,255,255,0.3);">norm: {norm}</div></div>',
                        unsafe_allow_html=True,
                    )


# ══════════════════════════════════════════════════════════════
#  3. CORRELATIONS — heatmaps, lag-1, Markov
# ══════════════════════════════════════════════════════════════


def pg_correlations(df, days):
    st.markdown(
        '<div class="pt">Correlations</div>'
        '<div class="ps">Statistical relationships between your metrics</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(["Heatmap", "Top Pairs", "Lag-1 Predictors", "Computed Summary"])

    # Key metrics for heatmap
    hm_metrics = [c for c in [
        "resting_hr", "hrv_last_night", "sleep_score", "stress_level",
        "training_readiness", "bb_charged", "bb_drained", "bb_peak",
        "total_steps", "deep_sleep_sec", "rem_sleep_sec",
        "vo2_max_running", "acwr", "daily_load_acute",
    ] if c in df.columns]

    with tab1:
        st.markdown('<div class="sh">Same-Day Correlation Matrix</div>', unsafe_allow_html=True)
        if len(hm_metrics) < 3:
            st.info("Need at least 3 metrics for heatmap.")
        else:
            d_sorted = df[hm_metrics].dropna(how="all")
            corr = d_sorted.corr(method="pearson")

            # Pretty labels
            labels = [m.replace("_", " ").title()[:18] for m in hm_metrics]
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=labels, y=labels,
                colorscale=[[0, "#f5576c"], [0.5, "#0d1117"], [1, "#00f5a0"]],
                zmin=-1, zmax=1,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=9, color="rgba(255,255,255,0.6)"),
                hovertemplate="%{x} × %{y}: r=%{z:.3f}<extra></extra>",
            ))
            fig.update_layout(
                **PL, height=max(450, len(hm_metrics) * 35),
                xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
                yaxis=dict(tickfont=dict(size=9)),
                margin=dict(l=120, b=120, t=30, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="sh">Top Correlated Pairs</div>', unsafe_allow_html=True)
        all_metrics = [c for c in df.columns
                       if pd.api.types.is_numeric_dtype(df[c]) and c not in ("date",)]
        pairs = []
        for i, a in enumerate(all_metrics):
            for b in all_metrics[i+1:]:
                clean = df[[a, b]].dropna()
                if len(clean) < 5:
                    continue
                from scipy import stats as _sp
                r, p = _sp.pearsonr(clean[a], clean[b])
                if abs(r) >= 0.3 and p < 0.05:
                    pairs.append({"Metric A": a, "Metric B": b,
                                  "r": round(r, 3), "p": round(p, 4),
                                  "n": len(clean),
                                  "Strength": "Strong" if abs(r) > 0.7 else (
                                      "Moderate" if abs(r) > 0.5 else "Weak")})
        if pairs:
            pairs_df = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False).head(25)
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No significant correlations found.")

    with tab3:
        st.markdown('<div class="sh">Lag-1: What Yesterday Predicts About Today</div>',
                    unsafe_allow_html=True)
        lag_results = []
        d_sorted = df.sort_values("date")
        for predictor in hm_metrics:
            for target in hm_metrics:
                if predictor == target:
                    continue
                sub = d_sorted[[predictor, target]].copy()
                sub["pred_lag"] = sub[predictor].shift(1)
                clean = sub[["pred_lag", target]].dropna()
                if len(clean) < 5:
                    continue
                from scipy import stats as _sp
                if clean["pred_lag"].std() < 1e-10 or clean[target].std() < 1e-10:
                    continue
                r, p = _sp.pearsonr(clean["pred_lag"], clean[target])
                if abs(r) >= 0.25 and p < 0.1:
                    lag_results.append({
                        "Yesterday's": predictor,
                        "→ Today's": target,
                        "r": round(r, 3),
                        "p": round(p, 4),
                        "n": len(clean),
                        "Direction": "positive" if r > 0 else "negative",
                    })
        if lag_results:
            lag_df = pd.DataFrame(lag_results).sort_values("r", key=abs, ascending=False).head(20)
            st.dataframe(lag_df, use_container_width=True, hide_index=True)

            # Visualize top 5 lag-1
            st.markdown('<div class="sh">Top 5 Predictive Relationships</div>',
                        unsafe_allow_html=True)
            for _, row in lag_df.head(5).iterrows():
                pred, tgt = row["Yesterday's"], row["→ Today's"]
                sub = d_sorted[[pred, tgt, "date"]].copy()
                sub["pred_lag"] = sub[pred].shift(1)
                clean = sub[["date", "pred_lag", tgt]].dropna()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=clean["pred_lag"], y=clean[tgt], mode="markers",
                                         marker=dict(color="#00f5a0", size=6),
                                         hovertemplate=f"Yesterday {pred}: %{{x:.1f}}<br>Today {tgt}: %{{y:.1f}}<extra></extra>"))
                fig.update_layout(**PL, height=220,
                                  xaxis_title=f"Yesterday's {pred}",
                                  yaxis_title=f"Today's {tgt}",
                                  title=f"r={row['r']:.3f}, n={row['n']}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No significant lag-1 predictors found.")

    with tab4:
        st.markdown('<div class="sh">Latest Computed Correlation Summary</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.72rem;color:rgba(255,255,255,0.35);margin-bottom:1rem;">'
            'This is the pre-computed Layer 3 summary from the correlation engine — '
            'the same data your AI agents read.</div>',
            unsafe_allow_html=True,
        )
        try:
            row = _q("SELECT summary_text, computed_at FROM matrix_summaries "
                     "ORDER BY computed_at DESC LIMIT 1")
            if not row.empty:
                ts = row.iloc[0]["computed_at"]
                txt = row.iloc[0]["summary_text"]
                st.markdown(f'**Computed:** {ts}')
                st.code(txt, language=None)
            else:
                st.info("No matrix summary found. Run `python src/weekly_sync.py --analyze` first.")
        except Exception as e:
            st.info(f"Could not load summary: {e}")


# ══════════════════════════════════════════════════════════════
#  4. DEEP DIVE — single metric exploration
# ══════════════════════════════════════════════════════════════


def pg_dive(df, days):
    st.markdown(
        '<div class="pt">Deep Dive</div>'
        '<div class="ps">Single-metric analysis with statistics & distribution</div>',
        unsafe_allow_html=True,
    )

    avail = [c for c in df.columns
             if pd.api.types.is_numeric_dtype(df[c])
             and c not in ("date",) and df[c].notna().sum() >= 3]
    if not avail:
        st.warning("No metrics available.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        metric = st.selectbox("Choose metric", sorted(avail), key="dive_metric")
    with c2:
        window = st.selectbox("Window", [7, 14, 21, 30, 60, 90, 180],
                              index=3, key="dive_window")

    wd = _load(window).sort_values("date")
    vals = wd[metric].dropna()
    if vals.empty:
        st.warning("No data for this metric.")
        return

    # Stats tiles
    cols = st.columns(5, gap="medium")
    mean_v, std_v, min_v, max_v = vals.mean(), vals.std(), vals.min(), vals.max()
    cv = (std_v / mean_v * 100) if mean_v != 0 else 0
    for col, (lb, v), ac in zip(cols,
        [("Mean", mean_v), ("Std Dev", std_v), ("Min", min_v),
         ("Max", max_v), ("CV%", cv)],
        ["g", "b", "a", "c", "d"]):
        with col:
            st.markdown(_tile(lb, f"{v:.1f}", accent=ac), unsafe_allow_html=True)

    # Time series + moving averages
    st.markdown('<div class="sh">Time Series</div>', unsafe_allow_html=True)
    cdf = wd[["date", metric]].dropna().sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf[metric], mode="lines+markers",
                             name="Actual", line=dict(color="#00f5a0", width=2),
                             marker=dict(size=4)))
    fig.add_trace(go.Scatter(x=cdf["date"], y=cdf[metric].rolling(7, min_periods=1).mean(),
                             mode="lines", name="7d MA",
                             line=dict(color="#667eea", width=2, dash="dot")))
    if len(cdf) >= 14:
        fig.add_trace(go.Scatter(x=cdf["date"], y=cdf[metric].rolling(14, min_periods=1).mean(),
                                 mode="lines", name="14d MA",
                                 line=dict(color="#f5c542", width=1.5, dash="dash")))
    fig.update_layout(**PL, height=320)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Distribution
        st.markdown('<div class="sh">Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=vals, nbinsx=20,
                                   marker=dict(color="rgba(0,245,160,0.4)",
                                               line=dict(color="#00f5a0", width=1))))
        fig.update_layout(**PL, height=250, bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Day-of-week analysis
        st.markdown('<div class="sh">Day of Week Pattern</div>', unsafe_allow_html=True)
        dow = wd[["date", metric]].copy()
        dow["date"] = pd.to_datetime(dow["date"])
        dow["dow"] = dow["date"].dt.day_name()
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_means = dow.groupby("dow")[metric].mean().reindex(dow_order).dropna()
        if not dow_means.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dow_means.index, y=dow_means.values,
                                 marker_color=PALETTE[:len(dow_means)]))
            fig.update_layout(**PL, height=250)
            st.plotly_chart(fig, use_container_width=True)

    # Correlations with this metric
    st.markdown('<div class="sh">Correlates With...</div>', unsafe_allow_html=True)
    corr_list = []
    for other in avail:
        if other == metric:
            continue
        clean = wd[[metric, other]].dropna()
        if len(clean) < 5:
            continue
        from scipy import stats as _sp
        r, p = _sp.pearsonr(clean[metric], clean[other])
        if p < 0.1:
            corr_list.append({"Metric": other, "r": round(r, 3), "p": round(p, 4)})
    if corr_list:
        corr_df = pd.DataFrame(corr_list).sort_values("r", key=abs, ascending=False).head(10)
        st.dataframe(corr_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
#  5. DATE EXPLORER — pick any day, see everything
# ══════════════════════════════════════════════════════════════


def pg_date_explorer(df, days):
    st.markdown(
        '<div class="pt">Date Explorer</div>'
        '<div class="ps">Pick any day — see every metric, activity, and context</div>',
        unsafe_allow_html=True,
    )

    mn, mx = _get_date_range()
    if mn is None:
        st.info("No data available.")
        return

    # Date picker
    selected_date = st.date_input(
        "Select a date",
        value=mx if mx else date_type.today(),
        min_value=mn, max_value=mx,
        key="date_explorer_date",
    )

    sd_str = str(selected_date)
    day_data = _q(
        "SELECT * FROM daily_metrics WHERE date = %s",
        params=(sd_str,),
    )

    if day_data.empty:
        st.warning(f"No data for {sd_str}. You may not have worn the watch that day.")
        return

    day = day_data.iloc[0]

    # Hero tiles for the day
    st.markdown(f'<div class="sh">{sd_str} — Key Metrics</div>', unsafe_allow_html=True)
    cols = st.columns(6, gap="small")
    day_metrics = [
        ("RHR", "resting_hr", "bpm", "g"),
        ("HRV", "hrv_last_night", "ms", "b"),
        ("Sleep", "sleep_score", "", "c"),
        ("Stress", "stress_level", "", "a"),
        ("Readiness", "training_readiness", "", "d"),
        ("Steps", "total_steps", "", "g"),
    ]
    for col, (lb, key, un, ac) in zip(cols, day_metrics):
        v = day.get(key)
        with col:
            st.markdown(
                f'<div class="mt a{ac}">'
                f'<div class="ml">{lb}</div>'
                f'<div class="mv" style="font-size:1.5rem;">{_safe_fmt(v)}'
                f'<span class="mu"> {un}</span></div></div>',
                unsafe_allow_html=True,
            )

    # Full metrics table
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sh">Body & Recovery</div>', unsafe_allow_html=True)
        body_metrics = [
            ("Resting HR", "resting_hr"), ("HRV", "hrv_last_night"),
            ("BB Charged", "bb_charged"), ("BB Drained", "bb_drained"),
            ("BB Peak", "bb_peak"), ("BB Low", "bb_low"),
            ("Weight", "weight_kg"), ("Hydration", "hydration_value_ml"),
            ("ACWR", "acwr"), ("Acute Load", "daily_load_acute"),
            ("Chronic Load", "daily_load_chronic"),
            ("VO2 Max Running", "vo2_max_running"),
        ]
        rows = []
        for lb, key in body_metrics:
            v = day.get(key)
            if pd.notna(v):
                rows.append({"Metric": lb, "Value": f"{v:.1f}" if isinstance(v, float) else str(v)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with c2:
        st.markdown('<div class="sh">Sleep Architecture</div>', unsafe_allow_html=True)
        sleep_metrics = [
            ("Sleep Score", "sleep_score"), ("Total Sleep", "sleep_seconds"),
            ("Deep Sleep", "deep_sleep_sec"), ("REM Sleep", "rem_sleep_sec"),
            ("Light Sleep", "light_sleep_sec"), ("Avg Respiration", "avg_respiration"),
            ("BB Change (sleep)", "body_battery_change"),
            ("Sleep HR", "resting_hr_sleep"),
        ]
        rows = []
        for lb, key in sleep_metrics:
            v = day.get(key)
            if pd.notna(v):
                if key in ("sleep_seconds", "deep_sleep_sec", "rem_sleep_sec", "light_sleep_sec"):
                    rows.append({"Metric": lb, "Value": f"{v/3600:.1f} hrs ({v/60:.0f} min)"})
                else:
                    rows.append({"Metric": lb, "Value": f"{v:.1f}" if isinstance(v, float) else str(v)})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Activities that day
    st.markdown(f'<div class="sh">Activities on {sd_str}</div>', unsafe_allow_html=True)
    try:
        acts = _q(
            "SELECT activity_name, activity_type, sport_type, "
            "ROUND(duration_sec/60.0) AS min, ROUND(distance_m/1000.0,2) AS km, "
            "average_hr, max_hr, calories, "
            "aerobic_training_effect AS aero_te, "
            "anaerobic_training_effect AS anaero_te, training_load "
            "FROM activities WHERE date = %s ORDER BY duration_sec DESC",
            params=(sd_str,),
        )
        if not acts.empty:
            st.dataframe(acts, use_container_width=True, hide_index=True)
        else:
            st.info("No activities recorded.")
    except Exception:
        st.info("No activities table or no data.")

    # ── Context: 3 days before and after ──
    st.markdown('<div class="sh">Context: ±3 Days</div>', unsafe_allow_html=True)
    context_start = selected_date - timedelta(days=3)
    context_end = selected_date + timedelta(days=3)
    ctx_data = _load_range(str(context_start), str(context_end))
    if not ctx_data.empty:
        ctx = ctx_data.sort_values("date")
        ctx_metrics = [c for c in ["date", "resting_hr", "hrv_last_night", "sleep_score",
                                    "stress_level", "training_readiness", "bb_charged",
                                    "bb_peak", "total_steps"] if c in ctx.columns]
        st.dataframe(ctx[ctx_metrics], use_container_width=True, hide_index=True)

        # Mini chart
        fig = go.Figure()
        for cn, nm, clr in [
            ("training_readiness", "Readiness", "#00f5a0"),
            ("sleep_score", "Sleep", "#667eea"),
            ("hrv_last_night", "HRV", "#4facfe"),
        ]:
            if cn in ctx.columns:
                fig.add_trace(go.Scatter(x=ctx["date"], y=ctx[cn], name=nm,
                                         mode="lines+markers",
                                         line=dict(color=clr, width=2),
                                         marker=dict(size=5)))
        # Mark selected date
        fig.add_vline(x=sd_str, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.update_layout(**PL, height=250)
        st.plotly_chart(fig, use_container_width=True)

    # ── Ask agents about this day ──
    st.markdown('<div class="sh">Ask Agents About This Day</div>', unsafe_allow_html=True)
    if _api_key():
        day_question = st.text_input(
            "Ask about this day",
            placeholder=f"e.g. Why did I feel great on {sd_str}? What contributed to my high readiness?",
            key="date_q",
        )
        if day_question and st.button("Ask", key="date_ask_btn"):
            with st.spinner("Agents analyzing..."):
                enriched_q = (
                    f"The user is asking about {sd_str} specifically. "
                    f"Context from that day: "
                    f"RHR={_safe_fmt(day.get('resting_hr'))}, "
                    f"HRV={_safe_fmt(day.get('hrv_last_night'))}, "
                    f"Sleep={_safe_fmt(day.get('sleep_score'))}, "
                    f"Stress={_safe_fmt(day.get('stress_level'))}, "
                    f"Readiness={_safe_fmt(day.get('training_readiness'))}, "
                    f"BB charged={_safe_fmt(day.get('bb_charged'))}, "
                    f"Steps={_safe_fmt(day.get('total_steps'))}. "
                    f"User question: {day_question}"
                )
                resp = _chat_agent(enriched_q)
                st.markdown(
                    f'<div class="cm ca"><div class="cn">⚡ Analysis</div>{resp}</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Set GOOGLE_API_KEY to enable agent queries.")


# ══════════════════════════════════════════════════════════════
#  6. AGENT CHAT — real-time conversation with all 9 specialists
# ══════════════════════════════════════════════════════════════


def pg_chat(df, days):
    st.markdown(
        '<div class="pt">Agent Chat</div>'
        '<div class="ps">Talk to 9 specialized AI agents — they have full database access + correlation data</div>',
        unsafe_allow_html=True,
    )

    if not _api_key():
        st.warning("Set GOOGLE_API_KEY in Streamlit secrets or .env to enable AI features.")
        return

    # Chat mode selection
    mode = st.radio(
        "Agent mode",
        ["Single Agent (fast)", "Multi-Agent (comprehensive)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown(
        '<div style="font-size:.72rem;color:rgba(255,255,255,0.3);margin-bottom:.5rem;">'
        '<b>Single:</b> one analyst answers quickly. '
        '<b>Multi:</b> dispatches to relevant specialists who work in parallel, '
        'then synthesizes their responses. Slower but deeper.</div>',
        unsafe_allow_html=True,
    )

    # Example questions
    with st.expander("Example questions", expanded=False):
        st.markdown("""
- Do you think last week's leg day affected this week's running?
- Can you give me a summary of [date] — calories, body battery, everything?
- What predicts my best training days?
- Is my HRV improving or declining? What's driving it?
- What's my biggest bottleneck right now?
- How does my sleep architecture compare to clinical norms?
- What should I focus on this week?
- Show me the correlation between stress and sleep quality
- Did my high step days lead to better recovery the next day?
        """)

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
                f'<div class="cm ca"><div class="cn">{m.get("agent_name", "⚡ Health Intelligence")}</div>{m["text"]}</div>',
                unsafe_allow_html=True,
            )

    inp = st.chat_input("Ask anything about your health data...")
    if inp:
        st.session_state.chat.append({"role": "user", "text": inp})
        st.markdown(f'<div class="cm cu">{html.escape(inp)}</div>', unsafe_allow_html=True)

        # Load matrix context to give agents correlation awareness
        matrix_ctx = ""
        try:
            row = _q("SELECT summary_text FROM matrix_summaries ORDER BY computed_at DESC LIMIT 1")
            if not row.empty:
                matrix_ctx = row.iloc[0]["summary_text"]
        except Exception:
            pass

        if "Single" in mode:
            with st.spinner("Agent thinking..."):
                try:
                    resp = _chat_agent(inp, matrix_context=matrix_ctx)
                    st.session_state.chat.append({
                        "role": "agent", "text": resp,
                        "agent_name": "⚡ Health Analyst",
                    })
                    st.markdown(
                        f'<div class="cm ca"><div class="cn">⚡ Health Analyst</div>{resp}</div>',
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.error(f"Agent error: {e}")
        else:
            with st.spinner("Dispatching to specialists — this may take 30-60s..."):
                try:
                    results = _multi_agent_chat(inp, matrix_context=matrix_ctx)
                    for agent_name, resp in results.items():
                        st.session_state.chat.append({
                            "role": "agent", "text": resp,
                            "agent_name": agent_name,
                        })
                        st.markdown(
                            f'<div class="cm ca"><div class="cn">{agent_name}</div>{resp}</div>',
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Multi-agent error: {e}")

    # Clear chat button
    if st.session_state.chat:
        if st.button("Clear chat", key="clear_chat"):
            st.session_state.chat = []
            st.rerun()


def _chat_agent(question: str, matrix_context: str = "") -> str:
    """Single agent that answers health questions via SQL + correlation data."""
    from crewai import Agent, Task, Crew, Process
    from enhanced_agents import (
        run_sql_query, calculate_correlation, find_best_days,
        analyze_pattern, _get_llm,
    )

    ctx_block = ""
    if matrix_context:
        ctx_block = (
            "\n\nYou also have access to PRE-COMPUTED CORRELATION DATA. "
            "Use this to back up your answers with statistical evidence:\n"
            f"{matrix_context[:4000]}\n"
        )

    agent = Agent(
        role="Health Data Analyst",
        goal="Answer the user's health question accurately using their Garmin data and correlation analysis",
        backstory=(
            "You are a concise health data analyst with direct SQL access "
            "to the user's Garmin health database (PostgreSQL) AND pre-computed "
            "correlation matrices. You can cite specific Pearson r values, "
            "lag-1 predictors, AR(1) persistence, Markov transitions, and "
            "KL-divergence results. Tables: daily_metrics "
            "(date, resting_hr, hrv_last_night, sleep_score, stress_level, "
            "training_readiness, bb_charged, bb_drained, bb_peak, bb_low, "
            "total_steps, deep_sleep_sec, rem_sleep_sec, sleep_seconds, "
            "weight_kg, acwr, daily_load_acute, daily_load_chronic, "
            "vo2_max_running, avg_respiration, body_battery_change, etc.), "
            "activities (date, activity_name, activity_type, duration_sec, "
            "distance_m, average_hr, max_hr, calories, aerobic_training_effect, "
            "anaerobic_training_effect, training_load, sport_type). "
            "Use PostgreSQL syntax. Be specific with numbers. "
            "When the user asks about relationships between metrics, cite "
            "the correlation data. When they ask about specific days, query "
            "the database. Keep answers concise but data-rich."
            f"{ctx_block}"
        ),
        verbose=False,
        allow_delegation=False,
        tools=[run_sql_query, calculate_correlation, find_best_days, analyze_pattern],
        llm=_get_llm(),
    )
    task = Task(
        description=f"Answer this question about the user's health data: {question}",
        agent=agent,
        expected_output="A concise, data-backed answer with specific numbers and correlation evidence",
    )
    result = Crew(
        agents=[agent], tasks=[task], process=Process.sequential, verbose=False,
    ).kickoff()
    raw = getattr(result, "raw", str(result))
    return raw.replace("\n", "<br>")


def _multi_agent_chat(question: str, matrix_context: str = "") -> dict:
    """Dispatch question to relevant specialists in parallel.
    
    Returns dict of {agent_name: response}.
    """
    from crewai import Agent, Task, Crew, Process
    from enhanced_agents import (
        run_sql_query, calculate_correlation, find_best_days,
        analyze_pattern, _get_llm,
    )

    tools = [run_sql_query, calculate_correlation, find_best_days, analyze_pattern]
    llm = _get_llm()

    ctx_block = ""
    if matrix_context:
        ctx_block = (
            "\n\nPRE-COMPUTED CORRELATION DATA (cite these when relevant):\n"
            f"{matrix_context[:3000]}\n"
        )

    # Create specialist agents relevant to most questions
    specialists = {
        "🔬 Matrix Analyst": Agent(
            role="Correlation Analyst", verbose=False, allow_delegation=False,
            goal="Provide correlation-backed statistical evidence for the user's question",
            backstory=f"You interpret correlations, lag-1 predictors, AR(1) persistence, and Markov transitions. {ctx_block}",
            tools=[], llm=llm,
        ),
        "🏋️ Performance": Agent(
            role="Performance Analyst", verbose=False, allow_delegation=False,
            goal="Answer from a training and performance perspective",
            backstory="You analyze activities, training load, ACWR, and performance trends. Use PostgreSQL for daily_metrics and activities tables.",
            tools=tools, llm=llm,
        ),
        "😴 Sleep Analyst": Agent(
            role="Sleep Specialist", verbose=False, allow_delegation=False,
            goal="Answer from a sleep quality perspective",
            backstory="You analyze sleep architecture (deep/REM/light), sleep score, and how sleep affects next-day metrics. Use PostgreSQL.",
            tools=tools, llm=llm,
        ),
        "🎯 Synthesizer": Agent(
            role="Health Synthesizer", verbose=False, allow_delegation=False,
            goal="Synthesize insights from other agents into a clear, actionable answer",
            backstory=f"You bring together all perspectives and provide the definitive answer. {ctx_block}",
            tools=tools, llm=llm,
        ),
    }

    results = {}

    def _run_specialist(name, agent):
        try:
            task = Task(
                description=(
                    f"User asked: {question}\n\n"
                    f"Answer from your specialist perspective. "
                    f"Query the database if needed. Cite specific numbers."
                ),
                agent=agent,
                expected_output="Specialist perspective with data",
            )
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            raw = getattr(crew.kickoff(), "raw", "")
            results[name] = raw.replace("\n", "<br>")
        except Exception as e:
            results[name] = f"Error: {e}"

    # Run first 3 in parallel, then synthesizer
    threads = []
    for name, agent in list(specialists.items())[:3]:
        t = threading.Thread(target=_run_specialist, args=(name, agent))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=90)

    # Synthesizer gets context from others
    synth_agent = specialists["🎯 Synthesizer"]
    other_responses = "\n\n".join(f"[{k}]: {v}" for k, v in results.items())
    try:
        task = Task(
            description=(
                f"User asked: {question}\n\n"
                f"Other specialists have already analyzed:\n{other_responses}\n\n"
                f"Synthesize the BEST answer. Resolve any disagreements. "
                f"If one agent provided stronger evidence, favor that view. "
                f"Be specific and actionable."
            ),
            agent=synth_agent,
            expected_output="Synthesized answer combining all specialist perspectives",
        )
        crew = Crew(agents=[synth_agent], tasks=[task], process=Process.sequential, verbose=False)
        raw = getattr(crew.kickoff(), "raw", "")
        results["🎯 Synthesizer"] = raw.replace("\n", "<br>")
    except Exception as e:
        results["🎯 Synthesizer"] = f"Synthesis error: {e}"

    return results


# ══════════════════════════════════════════════════════════════
#  7. AI ANALYSIS — full 9-agent parallel analysis
# ══════════════════════════════════════════════════════════════


def pg_ai(df, days):
    st.markdown(
        '<div class="pt">Full AI Analysis</div>'
        '<div class="ps">All 9 specialized agents analyze your data in parallel</div>',
        unsafe_allow_html=True,
    )

    if not _api_key():
        st.warning("Set GOOGLE_API_KEY to enable AI features.")
        return

    # Show saved summaries
    try:
        summaries = _q(
            "SELECT summary_text, created_at FROM weekly_summaries "
            "ORDER BY created_at DESC LIMIT 5"
        )
        if not summaries.empty:
            st.markdown('<div class="sh">Recent AI Reports</div>', unsafe_allow_html=True)
            for _, row in summaries.iterrows():
                ts = row["created_at"]
                txt = str(row["summary_text"])[:200] + "..."
                with st.expander(f"Report — {ts}"):
                    st.markdown(
                        f'<div style="color:rgba(255,255,255,0.75);'
                        f'font-size:.82rem;line-height:1.7;white-space:pre-wrap;">'
                        f'{html.escape(str(row["summary_text"]))}</div>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        pass

    # Run new analysis
    st.markdown('<div class="sh">Run New Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.72rem;color:rgba(255,255,255,0.35);margin-bottom:1rem;">'
        'Cost: ~$0.02 per run (Gemini 2.5 Flash)</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        mode = st.selectbox("Analysis mode", [
            "Full 9-Agent Analysis (~30s)",
            "Goal Analysis — 2 agents (~15s)",
            "Quick Check — 3 agents (~15s)",
        ], key="ai_mode")
    with c2:
        if "Goal" in mode:
            goal = st.text_input("Your goal", "Improve HRV by 10%", key="ai_goal")
        else:
            goal = ""

    if st.button("Run Analysis", type="primary", key="run_ai"):
        t0 = time.time()
        agents_obj = _get_agents()

        if "Full" in mode:
            with st.spinner("Running all 9 agents in parallel (3 batches × 3 threads)..."):
                result = _run_9_parallel(agents_obj)
        elif "Goal" in mode:
            with st.spinner("Analyzing goal..."):
                result = str(agents_obj.run_goal_analysis(goal))
        else:
            with st.spinner("Running quick check (3 agents)..."):
                result = _run_quick_check(agents_obj)

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
            elif "PERFORMANCE" in upper:
                color = "#00d9f5"
            elif "LIFESTYLE" in upper:
                color = "#764ba2"
            elif "STABILITY" in upper or "BENCHMARK" in upper:
                color = "#48dbfb"
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
    """Run all 9 agents in 3 batches of 3 threads."""
    from crewai import Task, Crew, Process

    # Load matrix context
    matrix_ctx = ""
    try:
        row = _q("SELECT summary_text FROM matrix_summaries ORDER BY computed_at DESC LIMIT 1")
        if not row.empty:
            matrix_ctx = row.iloc[0]["summary_text"]
    except Exception:
        pass

    ctx_suffix = ""
    if matrix_ctx:
        ctx_suffix = f"\n\nPRE-COMPUTED CORRELATION DATA:\n{matrix_ctx[:3000]}"

    results: dict[str, str] = {}

    def _run(name, agent, desc, exp):
        try:
            task = Task(description=desc + ctx_suffix, agent=agent, expected_output=exp)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            results[name] = str(crew.kickoff())
        except Exception as e:
            results[name] = f"Error: {e}"

    batch1 = [
        ("CORRELATION MATRIX INTERPRETATION", agents_obj.matrix_analyst,
         "Interpret the correlation data. Which metric pairs show strong Pearson r? "
         "What are the key lag-1 predictors? What do the Markov transitions show?",
         "Key correlations with interpretation"),
        ("CROSS-TIMEFRAME STABILITY", agents_obj.matrix_comparator,
         "Compare metric relationships across time windows. "
         "Which correlations are ROBUST vs just recent noise?",
         "Stability classification of key patterns"),
        ("HIDDEN PATTERNS", agents_obj.pattern_detective,
         "Find the top 3 non-obvious patterns. Look for day-of-week effects, "
         "delayed correlations, unexpected interactions. Query the database for evidence.",
         "3 hidden patterns with evidence"),
    ]
    batch2 = [
        ("PERFORMANCE OPTIMIZATION", agents_obj.performance_optimizer,
         "Analyze peak vs poor performance days. What conditions predict great training days?",
         "Peak performance conditions + recommendations"),
        ("RECOVERY ASSESSMENT", agents_obj.recovery_specialist,
         "Assess recovery: HRV trend, sleep quality, body battery, RHR. "
         "Signs of overtraining? Bounce-back speed?",
         "Recovery status with recommendation"),
        ("TRENDS & FORECASTS", agents_obj.trend_forecaster,
         "Project current trends. Which metrics improving/declining? Early warnings?",
         "Trend projections with early warnings"),
    ]
    batch3 = [
        ("LIFESTYLE CONNECTIONS", agents_obj.lifestyle_analyst,
         "Which lifestyle factors most impact health outcomes? "
         "What differentiates good days from bad days?",
         "Lifestyle impact analysis"),
        ("SLEEP ANALYSIS", agents_obj.sleep_analyst,
         "Deep/REM/light as % of total vs clinical norms. Consistency. "
         "Next-day impact of sleep quality.",
         "Sleep architecture analysis"),
        ("BOTTLENECK & QUICK WINS", agents_obj.weakness_identifier,
         "What is the #1 bottleneck? What are 3 quick wins for next week?",
         "1 bottleneck + 3 quick wins"),
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
            sections.append(f"{'='*50}\n  {label}\n{'='*50}\n\n{results[label]}")
    return "\n\n".join(sections) if sections else "Analysis timed out."


def _run_quick_check(agents_obj) -> str:
    """Run 3 key agents for a quick check."""
    from crewai import Task, Crew, Process

    results: dict[str, str] = {}

    def _run(name, agent, desc, exp):
        try:
            task = Task(description=desc, agent=agent, expected_output=exp)
            crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
            results[name] = str(crew.kickoff())
        except Exception as e:
            results[name] = f"Error: {e}"

    batch = [
        ("PERFORMANCE SUMMARY", agents_obj.performance_optimizer,
         "Quick week-over-week comparison: key metrics this week vs last week.",
         "Week-over-week comparison"),
        ("RECOVERY STATUS", agents_obj.recovery_specialist,
         "Quick recovery assessment: are we good to train or need rest?",
         "Recovery status"),
        ("TOP PRIORITIES", agents_obj.weakness_identifier,
         "What's the #1 thing to focus on this week?",
         "Top priority with evidence"),
    ]

    threads = [threading.Thread(target=_run, args=args) for args in batch]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=90)

    sections = []
    for name, _, _, _ in batch:
        if name in results:
            sections.append(f"{'='*50}\n  {name}\n{'='*50}\n\n{results[name]}")
    return "\n\n".join(sections) if sections else "Analysis timed out."


# ══════════════════════════════════════════════════════════════
#  8. GOALS — weekly trends with clinical benchmarks
# ══════════════════════════════════════════════════════════════


def pg_goals(df, days):
    st.markdown(
        '<div class="pt">Goals & Progress</div>'
        '<div class="ps">Multi-week trends with benchmarks</div>',
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
                   AVG(acwr) AS acwr,
                   AVG(vo2_max_running) AS vo2max,
                   COUNT(*) AS days
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY week ORDER BY week
        """)
    except Exception:
        st.info("Not enough data yet.")
        return

    if len(weekly) < 2:
        st.info("Need at least 2 weeks of data.")
        return

    latest = weekly.iloc[-1]
    first = weekly.iloc[0]

    # This week vs start
    st.markdown('<div class="sh">This Week vs Earliest</div>', unsafe_allow_html=True)
    cols = st.columns(5, gap="small")
    comparisons = [
        ("HRV", "hrv", "ms", "b", False),
        ("Sleep", "sleep", "", "c", False),
        ("Readiness", "readiness", "", "g", False),
        ("RHR", "rhr", "bpm", "a", True),
        ("VO2Max", "vo2max", "", "d", False),
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

    # Weekly trend sparklines
    st.markdown('<div class="sh">Weekly Trends</div>', unsafe_allow_html=True)

    metrics_cfg = [
        ("hrv", "HRV", "#00f5a0", "ms", "Higher = better recovery. 50-100ms typical.", False),
        ("sleep", "Sleep Score", "#667eea", "/100", "Clinical target: >80.", False),
        ("readiness", "Training Readiness", "#4facfe", "/100", ">70 = go hard. <40 = rest.", False),
        ("rhr", "Resting HR", "#f5576c", "bpm", "Lower = fitter.", True),
        ("stress", "Stress", "#f5c542", "", "<25 = calm. >50 = high.", True),
        ("bb_peak", "BB Peak", "#f093fb", "/100", ">80 = fully recharged.", False),
        ("steps", "Daily Steps", "#00d9f5", "", "WHO: 8,000-10,000.", False),
        ("acwr", "ACWR", "#764ba2", "", "0.8-1.3 optimal.", False),
        ("vo2max", "VO2 Max", "#48dbfb", "", "Higher = fitter.", False),
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
        trend_color = "#00f5a0" if improving else ("#f5576c" if change != 0 else "rgba(255,255,255,0.3)")

        c1, c2 = st.columns([3, 1], gap="medium")
        with c1:
            r_hex = int(color[1:3], 16)
            g_hex = int(color[3:5], 16)
            b_hex = int(color[5:7], 16)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly["week"], y=weekly[key], mode="lines+markers", name=label,
                line=dict(color=color, width=2.5), marker=dict(size=5),
                fill="tozeroy", fillcolor=f"rgba({r_hex},{g_hex},{b_hex},0.05)",
            ))
            fig.update_layout(**PL, height=150, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(
                f'<div class="gc" style="height:100%;display:flex;flex-direction:column;'
                f'justify-content:center;padding:1rem;">'
                f'<div class="ml">{label}</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:#fff;">'
                f'{latest_v:.1f}<span style="font-size:.65rem;color:rgba(255,255,255,0.35);"> '
                f'{unit}</span></div>'
                f'<div style="font-size:.68rem;font-weight:600;color:{trend_color};margin-top:.2rem;">'
                f'{trend_word} · {"+" if change >= 0 else ""}{change:.1f}</div>'
                f'<div style="font-size:.6rem;color:rgba(255,255,255,0.25);margin-top:.3rem;">'
                f'{note}</div></div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
