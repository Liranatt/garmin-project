"""
Shared helpers for API routes.
Extracted from the monolithic api.py for maintainability.
Contains: DB access, type coercion, text formatting, insight structuring.
"""

from __future__ import annotations

import logging
import os
import json
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

log = logging.getLogger("api")


# ─── DB helpers ─────────────────────────────────────────────

def _normalize_db_url(value: str) -> str:
    db_url = (value or "").strip()
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url


def _conn_str() -> str:
    return _normalize_db_url(
        os.getenv("POSTGRES_CONNECTION_STRING") or os.getenv("DATABASE_URL") or ""
    )


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _fetch_all(query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
    cs = _conn_str()
    if not cs:
        raise RuntimeError("POSTGRES_CONNECTION_STRING is not set")
    conn = psycopg2.connect(cs)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params or ())
            rows = cur.fetchall()
            out: List[Dict[str, Any]] = []
            for row in rows:
                out.append({k: _to_jsonable(v) for k, v in dict(row).items()})
            return out
    finally:
        conn.close()


def _fetch_one(query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
    rows = _fetch_all(query, params=params)
    return rows[0] if rows else None


@lru_cache(maxsize=64)
def _table_columns(table_name: str) -> List[str]:
    rows = _fetch_all(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """,
        (table_name,),
    )
    return [r["column_name"] for r in rows]


def _first_existing(table_name: str, candidates: List[str]) -> Optional[str]:
    cols = set(_table_columns(table_name))
    for c in candidates:
        if c in cols:
            return c
    return None


def _pick_row_value(row: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key and row.get(key) is not None:
            return row.get(key)
    return None


# ─── Type coercion ──────────────────────────────────────────

def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


# ─── Text formatting ───────────────────────────────────────

def _window_note(values: List[float], higher_is_better: bool = True) -> str:
    clean = [v for v in values if v is not None]
    if len(clean) < 4:
        return "Not enough data points yet."
    mid = len(clean) // 2
    first = sum(clean[:mid]) / max(mid, 1)
    second = sum(clean[mid:]) / max(len(clean) - mid, 1)
    if first == 0:
        return "Not enough data points yet."
    pct = ((second - first) / abs(first)) * 100.0
    improving = pct > 0 if higher_is_better else pct < 0
    direction = "improved" if improving else "declined"
    return f"{direction} {abs(pct):.1f}% between early and recent window."


def _clip_text(text: str, max_len: int = 280) -> str:
    s = _text(text).replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


# ─── Insight structuring ───────────────────────────────────

def _structured_insight(text: str) -> Dict[str, Any]:
    """Enforce short, human-readable insight schema for UI cards."""
    cleaned = _text(text)
    if not cleaned:
        empty = "Insufficient data to generate this section yet."
        return {
            "headline": "Daily Health Insight",
            "what_changed": empty,
            "why_it_matters": empty,
            "next_24_48h": "Prioritize recovery basics and reassess on the next daily sync.",
            "bullets": [
                f"What changed: {empty}",
                f"Why it matters: {empty}",
                "Next 24-48h: Prioritize recovery basics and reassess on the next daily sync.",
            ],
            "summary": f"- What changed: {empty}\n- Why it matters: {empty}\n- Next 24-48h: Prioritize recovery basics and reassess on the next daily sync.",
        }

    candidates: List[str] = []
    for raw in cleaned.splitlines():
        line = raw.strip().lstrip("-* ").strip()
        if not line:
            continue
        if line.startswith("|") and line.endswith("|"):
            continue
        if set(line) <= {"=", "-", ":"}:
            continue
        candidates.append(line)

    what_changed = ""
    why_it_matters = ""
    next_24_48h = ""

    for line in candidates:
        low = line.lower()
        if not what_changed and any(t in low for t in ("trend", "up", "down", "increase", "decrease", "improved", "declined", "drop", "rise", "stable")):
            what_changed = line
            continue
        if not why_it_matters and any(t in low for t in ("because", "means", "risk", "impact", "matters", "recovery", "fatigue", "stress", "readiness")):
            why_it_matters = line
            continue
        if not next_24_48h and any(t in low for t in ("recommend", "should", "next", "today", "tomorrow", "focus", "avoid", "keep", "do")):
            next_24_48h = line

    for line in candidates:
        if not what_changed:
            what_changed = line
            continue
        if not why_it_matters and line != what_changed:
            why_it_matters = line
            break
    if not next_24_48h:
        next_24_48h = "Keep effort moderate today, protect sleep quality tonight, and reassess after tomorrow's readiness."

    what_changed = _clip_text(what_changed or "Weekly pattern changed, but details are limited.", 260)
    why_it_matters = _clip_text(why_it_matters or "This pattern can affect recovery quality and training response.", 260)
    next_24_48h = _clip_text(next_24_48h, 260)
    headline = _clip_text(what_changed, 84)

    def bullet(label: str, value: str) -> str:
        prefix = f"{label}: "
        allowed = max(48, 280 - len(prefix))
        return prefix + _clip_text(value, allowed)

    bullets = [
        bullet("What changed", what_changed),
        bullet("Why it matters", why_it_matters),
        bullet("Next 24-48h", next_24_48h),
    ]
    summary = "\n".join(f"- {b}" for b in bullets)
    return {
        "headline": headline,
        "what_changed": what_changed,
        "why_it_matters": why_it_matters,
        "next_24_48h": next_24_48h,
        "bullets": bullets,
        "summary": summary,
    }


_LOW_SIGNAL_PATTERNS = (
    "data not provided",
    "not provided",
    "insufficient data",
    "cannot be generated",
    "could not be generated",
    "not available",
    "no past recommendations",
    "please provide the pre-computed correlation analysis",
)


def _is_low_signal_text(text: str) -> bool:
    low = _text(text).lower()
    if not low:
        return True
    return any(pat in low for pat in _LOW_SIGNAL_PATTERNS)


def _insight_rank(item: Dict[str, Any]) -> tuple:
    summary = _text(item.get("summary"))
    detail = _text(item.get("detail")) or _text(item.get("content"))
    merged = f"{summary}\n{detail}"
    low_signal = _is_low_signal_text(merged)
    actionable = any(
        k in merged.lower()
        for k in ("recommend", "next 24-48h", "focus", "keep", "avoid", "target", "expected")
    )
    src = _text(item.get("source"))
    src_priority = 1 if src == "agent_recommendations" else 0
    ts = _text(item.get("timestamp") or item.get("created_at"))
    return (
        0 if low_signal else 1,
        1 if actionable else 0,
        src_priority,
        ts,
    )


# ─── Activity helpers ──────────────────────────────────────

def _matches_sport(activity_type: str, sport: str) -> bool:
    s = _text(sport).lower().strip()
    if s in ("", "all"):
        return True
    val = _text(activity_type).lower()
    buckets = {
        "running": ("run", "jog"),
        "cycling": ("cycl", "bike"),
        "swimming": ("swim"),
        "skiing": ("ski",),
        "gym": ("strength", "weight", "gym", "crossfit", "workout"),
    }
    tokens = buckets.get(s, (s,))
    return any(t in val for t in tokens)


def _sport_bucket(activity_type: str) -> str:
    val = _text(activity_type).lower()
    if any(t in val for t in ("run", "jog")):
        return "running"
    if any(t in val for t in ("cycl", "bike")):
        return "cycling"
    if "swim" in val:
        return "swimming"
    if "ski" in val:
        return "skiing"
    if any(t in val for t in ("strength", "weight", "gym", "crossfit", "workout")):
        return "gym"
    if any(t in val for t in ("basketball", "bball")):
        return "basketball"
    return "other"


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 5:
        return None
    x_vals = [p[0] for p in pairs]
    y_vals = [p[1] for p in pairs]
    x_mean = sum(x_vals) / len(x_vals)
    y_mean = sum(y_vals) / len(y_vals)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x, y in pairs:
        dx = x - x_mean
        dy = y - y_mean
        num += dx * dy
        den_x += dx * dx
        den_y += dy * dy
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x ** 0.5 * den_y ** 0.5)


def _readiness_state(value: Any) -> Optional[str]:
    v = _num(value)
    if v is None:
        return None
    if v <= 25:
        return "poor"
    if v <= 50:
        return "low"
    if v <= 75:
        return "moderate"
    if v <= 95:
        return "high"
    return "effective"


def _parse_activity_date(value: Any) -> Any:
    if value is None:
        return None
    s = str(value)
    return s[:10] if len(s) >= 10 else s


def _speed_to_kph(value: Any) -> Optional[float]:
    v = _num(value)
    if v is None:
        return None
    # Garmin exports are usually m/s; convert if plausible.
    if 0 < v < 25:
        return round(v * 3.6, 3)
    return round(v, 3)


# ─── Payload builders ──────────────────────────────────────

def _build_snapshot_payload(row: Dict[str, Any], field_map: Dict[str, Optional[str]]) -> Dict[str, Any]:
    date_col = field_map["date"]
    rhr_col = field_map["rhr"]
    hrv_col = field_map["hrv"]
    sleep_col = field_map["sleep"]
    bb_col = field_map["battery"]
    stress_col = field_map["stress"]
    load_col = field_map["load"]
    readiness_col = field_map["readiness"]

    timestamp = _pick_row_value(row, date_col or "")
    resting_hr = _num(_pick_row_value(row, rhr_col or ""))
    hrv = _num(_pick_row_value(row, hrv_col or ""))
    sleep = _num(_pick_row_value(row, sleep_col or ""))
    battery = _num(_pick_row_value(row, bb_col or ""))
    stress = _num(_pick_row_value(row, stress_col or ""))
    load = _num(_pick_row_value(row, load_col or ""))
    readiness = _num(_pick_row_value(row, readiness_col or ""))

    snapshot = {
        "resting_hr": resting_hr,
        "restingHeartRate": resting_hr,
        "rhr": resting_hr,
        "resting_hr_avg": resting_hr,
        "hrv": hrv,
        "hrv_ms": hrv,
        "rmssd": hrv,
        "sleep_score": sleep,
        "sleepScore": sleep,
        "sleep": sleep,
        "body_battery": battery,
        "bodyBattery": battery,
        "battery": battery,
        "stress": stress,
        "stress_score": stress,
        "avg_stress": stress,
        "training_load": load,
        "load": load,
        "acute_load": load,
        "seven_day_load": load,
        "training_readiness": readiness,
        "readiness": readiness,
        "readiness_score": readiness,
        "timestamp": timestamp,
        "created_at": timestamp,
        "date": timestamp,
        "as_of": timestamp,
    }
    return snapshot


def _build_history_rows(rows: List[Dict[str, Any]], field_map: Dict[str, Optional[str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        date_val = _pick_row_value(row, field_map["date"] or "")
        rhr = _num(_pick_row_value(row, field_map["rhr"] or ""))
        hrv = _num(_pick_row_value(row, field_map["hrv"] or ""))
        sleep = _num(_pick_row_value(row, field_map["sleep"] or ""))
        stress = _num(_pick_row_value(row, field_map["stress"] or ""))
        battery = _num(_pick_row_value(row, field_map["battery"] or ""))
        load = _num(_pick_row_value(row, field_map["load"] or ""))
        readiness = _num(_pick_row_value(row, field_map["readiness"] or ""))

        out.append(
            {
                "date": date_val,
                "timestamp": date_val,
                "as_of": date_val,
                "resting_hr": rhr,
                "restingHr": rhr,
                "rhr": rhr,
                "hrv": hrv,
                "hrv_last_night": hrv,
                "hrv_ms": hrv,
                "rmssd": hrv,
                "sleep_score": sleep,
                "sleepScore": sleep,
                "sleep": sleep,
                "stress": stress,
                "stress_level": stress,
                "stress_score": stress,
                "avg_stress": stress,
                "battery": battery,
                "body_battery": battery,
                "bb_peak": battery,
                "training_load": load,
                "daily_load_acute": load,
                "load": load,
                "acute_load": load,
                "training_readiness": readiness,
                "readiness": readiness,
                "readiness_score": readiness,
            }
        )
    return out


def _workout_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    types = sorted(
        {
            _text(r.get("activity_type"))
            for r in rows
            if r.get("activity_type") not in (None, "")
        }
    )

    running = [r for r in rows if "run" in _text(r.get("activity_type")).lower()]
    running_speeds = [_num(r.get("speed_kph")) for r in running]
    running_cadence = [_num(r.get("avg_cadence")) for r in running]

    strength = [
        r
        for r in rows
        if any(
            token in _text(r.get("activity_type")).lower()
            for token in ("strength", "weight", "gym", "crossfit", "workout")
        )
    ]
    strength_load = [_num(r.get("training_load")) for r in strength]
    strength_duration = [_num(r.get("duration_min")) for r in strength]

    if any(v is not None for v in strength_load):
        strength_note = "Strength load " + _window_note(
            [v for v in strength_load if v is not None], higher_is_better=True
        )
    elif any(v is not None for v in strength_duration):
        strength_note = "Strength duration " + _window_note(
            [v for v in strength_duration if v is not None], higher_is_better=True
        )
    else:
        strength_note = "No strength-specific signal found in available columns."

    running_note = "Not enough running data points yet."
    if any(v is not None for v in running_speeds):
        running_note = "Running speed " + _window_note(
            [v for v in running_speeds if v is not None], higher_is_better=True
        )
    if any(v is not None for v in running_cadence):
        running_note += " Running cadence " + _window_note(
            [v for v in running_cadence if v is not None], higher_is_better=True
        )

    return {
        "activity_types": types,
        "strength_proxy_trend": {"note": strength_note},
        "running_trend": {"note": running_note},
    }


def _fallback_chat(message: str) -> str:
    try:
        row = _fetch_one(
            """
            SELECT
                AVG(resting_hr) AS avg_rhr,
                AVG(hrv_last_night) AS avg_hrv,
                AVG(sleep_score) AS avg_sleep,
                AVG(stress_level) AS avg_stress
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '7 days'
            """
        ) or {}
        return (
            "AI chat model is unavailable right now. "
            f"Last 7d averages: RHR={_num(row.get('avg_rhr'))}, "
            f"HRV={_num(row.get('avg_hrv'))}, "
            f"Sleep={_num(row.get('avg_sleep'))}, "
            f"Stress={_num(row.get('avg_stress'))}. "
            f"Your question was: {message}"
        )
    except Exception:
        return (
            "AI chat model is unavailable right now and database summary could not be loaded. "
            f"Your question was: {message}"
        )
