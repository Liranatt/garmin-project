"""
FastAPI backend contract for gps_presentation frontend integration.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from decimal import Decimal
from functools import lru_cache
from typing import Any, Dict, List, Optional

import psycopg2
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor

load_dotenv()

log = logging.getLogger("api")


def _normalize_db_url(value: str) -> str:
    db_url = (value or "").strip()
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url


def _conn_str() -> str:
    return _normalize_db_url(
        os.getenv("POSTGRES_CONNECTION_STRING") or os.getenv("DATABASE_URL") or ""
    )


if not os.getenv("POSTGRES_CONNECTION_STRING"):
    fallback_db_url = _normalize_db_url(os.getenv("DATABASE_URL", ""))
    if fallback_db_url:
        # CrewAI tools in src.enhanced_agents read this env var directly.
        os.environ["POSTGRES_CONNECTION_STRING"] = fallback_db_url


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _num(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


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
    if v >= 70:
        return "high"
    if v >= 50:
        return "medium"
    return "low"


app = FastAPI(title="Garmin Health API", version="1.0.0")

_origin_env = os.getenv("FRONTEND_ORIGINS", "")
_origins = [o.strip() for o in _origin_env.split(",") if o.strip()] or [
    "https://liranattar.dev",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


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


@app.get("/")
def root() -> Dict[str, Any]:
    return {"service": "garmin-health-api", "status": "ok"}


@app.get("/health-check")
def health_check() -> JSONResponse:
    try:
        _fetch_one("SELECT 1 AS ok")
        return JSONResponse({"status": "Online", "message": "Online"})
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={
                "status": "Waking up",
                "message": f"Service starting or DB unavailable: {e}",
            },
        )


@app.get("/api/v1/snapshot/latest")
def snapshot_latest() -> Dict[str, Any]:
    try:
        date_col = _first_existing("daily_metrics", ["date", "created_at"])
        if not date_col:
            return {"snapshot": {}, "data": {}}

        field_map = {
            "date": date_col,
            "rhr": _first_existing("daily_metrics", ["resting_hr"]),
            "hrv": _first_existing("daily_metrics", ["hrv_last_night", "tr_hrv_weekly_avg", "hrv_weekly_avg"]),
            "sleep": _first_existing("daily_metrics", ["sleep_score"]),
            "battery": _first_existing("daily_metrics", ["bb_peak", "bb_charged", "bb_low"]),
            "stress": _first_existing("daily_metrics", ["stress_level", "avg_stress_level"]),
            "load": _first_existing("daily_metrics", ["tr_acute_load", "daily_load_acute", "bb_drained"]),
            "readiness": _first_existing("daily_metrics", ["training_readiness", "readiness"]),
        }

        select_cols = [c for c in field_map.values() if c]
        row = _fetch_one(
            f"""
            SELECT {", ".join(dict.fromkeys(select_cols))}
            FROM daily_metrics
            ORDER BY {date_col} DESC
            LIMIT 1
            """
        )
        if not row:
            return {"snapshot": {}, "data": {}}

        snapshot = _build_snapshot_payload(row, field_map)
        out = {"snapshot": snapshot, "data": snapshot}
        out.update(snapshot)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/metrics/history")
def metrics_history(days: int = Query(default=60, ge=1, le=3650)) -> Dict[str, Any]:
    try:
        date_col = _first_existing("daily_metrics", ["date", "created_at"])
        if not date_col:
            return {"data": [], "history": []}

        field_map = {
            "date": date_col,
            "rhr": _first_existing("daily_metrics", ["resting_hr"]),
            "hrv": _first_existing("daily_metrics", ["hrv_last_night", "tr_hrv_weekly_avg", "hrv_weekly_avg"]),
            "sleep": _first_existing("daily_metrics", ["sleep_score"]),
            "stress": _first_existing("daily_metrics", ["stress_level", "avg_stress_level"]),
            "battery": _first_existing("daily_metrics", ["bb_peak", "bb_charged", "bb_low"]),
            "load": _first_existing("daily_metrics", ["daily_load_acute", "tr_acute_load", "bb_drained"]),
            "readiness": _first_existing("daily_metrics", ["training_readiness", "readiness"]),
        }
        select_cols = [c for c in field_map.values() if c]
        rows = _fetch_all(
            f"""
            SELECT {", ".join(dict.fromkeys(select_cols))}
            FROM daily_metrics
            WHERE {date_col} >= CURRENT_DATE - (%s * INTERVAL '1 day')
            ORDER BY {date_col} ASC
            """,
            (days,),
        )
        items = _build_history_rows(rows, field_map)
        return {"data": items, "history": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/workouts/progress")
def workouts_progress(
    days: int = Query(default=30, ge=1, le=3650),
    sport: str = Query(default="all"),
) -> Dict[str, Any]:
    try:
        date_col = _first_existing("activities", ["date", "start_time_local", "start_time_gmt"])
        if not date_col:
            return {
                "data": [],
                "summary": {
                    "activity_types": [],
                    "strength_proxy_trend": {"note": "No activities date column found."},
                },
            }

        date_expr = "date" if date_col == "date" else f"({date_col})::timestamp::date"
        selected = {
            "date_raw": date_col,
            "activity_name": _first_existing("activities", ["activity_name"]),
            "activity_type": _first_existing("activities", ["activity_type", "sport_type"]),
            "duration_sec": _first_existing("activities", ["duration_sec", "elapsed_duration_sec", "moving_duration_sec"]),
            "distance_m": _first_existing("activities", ["distance_m"]),
            "average_hr": _first_existing("activities", ["average_hr", "avg_hr"]),
            "average_speed": _first_existing("activities", ["average_speed", "avg_speed"]),
            "avg_cadence": _first_existing("activities", ["avg_cadence"]),
            "training_load": _first_existing("activities", ["training_load"]),
        }

        select_parts = [f"{date_expr} AS workout_date"]
        for alias, col in selected.items():
            if col:
                select_parts.append(f"{col} AS {alias}")

        rows = _fetch_all(
            f"""
            SELECT {", ".join(select_parts)}
            FROM activities
            WHERE {date_expr} >= CURRENT_DATE - (%s * INTERVAL '1 day')
            ORDER BY workout_date ASC
            """,
            (days,),
        )

        items: List[Dict[str, Any]] = []
        for row in rows:
            duration_sec = _num(row.get("duration_sec"))
            distance_m = _num(row.get("distance_m"))
            duration_min = round(duration_sec / 60.0, 2) if duration_sec is not None else None
            distance_km = round(distance_m / 1000.0, 3) if distance_m is not None else None
            speed_kph = _speed_to_kph(row.get("average_speed"))
            avg_hr = _num(row.get("average_hr"))
            avg_cadence = _num(row.get("avg_cadence"))
            training_load = _num(row.get("training_load"))
            when = _parse_activity_date(row.get("workout_date"))
            activity_name = row.get("activity_name")
            activity_type = row.get("activity_type")

            item = {
                "date": when,
                "timestamp": when,
                "activity_name": activity_name,
                "activityName": activity_name,
                "activity_type": activity_type,
                "activityType": activity_type,
                "sport_type": activity_type,
                "sportType": activity_type,
                "duration_min": duration_min,
                "durationMin": duration_min,
                "duration_sec": duration_sec,
                "durationSec": duration_sec,
                "distance_km": distance_km,
                "distanceKm": distance_km,
                "distance_m": distance_m,
                "distanceM": distance_m,
                "average_hr": avg_hr,
                "avg_hr": avg_hr,
                "avgHr": avg_hr,
                "speed_kph": speed_kph,
                "speedKph": speed_kph,
                "avg_cadence": avg_cadence,
                "avgCadence": avg_cadence,
                "cadence": avg_cadence,
                "training_load": training_load,
                "trainingLoad": training_load,
            }
            items.append(item)

        filtered = [r for r in items if _matches_sport(_text(r.get("activity_type")), sport)]
        summary = _workout_summary(filtered)
        return {"data": filtered, "summary": summary, "selected_sport": sport}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/cross-effects")
def analytics_cross_effects(days: int = Query(default=90, ge=14, le=3650)) -> Dict[str, Any]:
    """Cross-activity effect estimates (Pearson + readiness-state transitions)."""
    try:
        act_date_col = _first_existing("activities", ["date", "start_time_local", "start_time_gmt"])
        dm_date_col = _first_existing("daily_metrics", ["date", "created_at"])
        if not act_date_col or not dm_date_col:
            return {"pearson_effects": [], "markov_effects": [], "summary": "Missing required date columns."}

        act_load_col = _first_existing("activities", ["training_load"])
        act_duration_col = _first_existing("activities", ["duration_sec", "elapsed_duration_sec", "moving_duration_sec"])
        act_type_col = _first_existing("activities", ["activity_type", "sport_type"])
        dm_cols = {
            "readiness": _first_existing("daily_metrics", ["training_readiness", "readiness"]),
            "sleep": _first_existing("daily_metrics", ["sleep_score"]),
            "stress": _first_existing("daily_metrics", ["stress_level", "avg_stress_level"]),
            "hrv": _first_existing("daily_metrics", ["hrv_last_night", "tr_hrv_weekly_avg", "hrv_weekly_avg"]),
            "rhr": _first_existing("daily_metrics", ["resting_hr"]),
        }
        if not act_type_col:
            return {"pearson_effects": [], "markov_effects": [], "summary": "Activity type column missing."}

        act_date_expr = "date" if act_date_col == "date" else f"({act_date_col})::timestamp::date"
        act_select = [f"{act_date_expr} AS d", f"{act_type_col} AS activity_type"]
        if act_load_col:
            act_select.append(f"{act_load_col} AS training_load")
        if act_duration_col:
            act_select.append(f"{act_duration_col} AS duration_sec")
        act_rows = _fetch_all(
            f"""
            SELECT {", ".join(act_select)}
            FROM activities
            WHERE {act_date_expr} >= CURRENT_DATE - (%s * INTERVAL '1 day')
            """,
            (days + 2,),
        )

        dm_select = [f"{dm_date_col} AS d"]
        for alias, col in dm_cols.items():
            if col:
                dm_select.append(f"{col} AS {alias}")
        dm_rows = _fetch_all(
            f"""
            SELECT {", ".join(dm_select)}
            FROM daily_metrics
            WHERE {dm_date_col} >= CURRENT_DATE - (%s * INTERVAL '1 day')
            ORDER BY {dm_date_col} ASC
            """,
            (days + 2,),
        )

        # Aggregate activity stimulus per day and sport bucket.
        stimulus_by_day: Dict[str, Dict[str, float]] = {}
        for row in act_rows:
            day = _parse_activity_date(row.get("d"))
            if not day:
                continue
            bucket = _sport_bucket(_text(row.get("activity_type")))
            load = _num(row.get("training_load"))
            if load is None:
                duration = _num(row.get("duration_sec"))
                load = (duration / 60.0) if duration is not None else None
            if load is None:
                continue
            stimulus_by_day.setdefault(day, {})
            stimulus_by_day[day][bucket] = stimulus_by_day[day].get(bucket, 0.0) + float(load)

        metrics_by_day: Dict[str, Dict[str, Optional[float]]] = {}
        for row in dm_rows:
            day = _parse_activity_date(row.get("d"))
            if not day:
                continue
            metrics_by_day[day] = {
                "readiness": _num(row.get("readiness")),
                "sleep": _num(row.get("sleep")),
                "stress": _num(row.get("stress")),
                "hrv": _num(row.get("hrv")),
                "rhr": _num(row.get("rhr")),
            }

        days_sorted = sorted(metrics_by_day.keys())
        if len(days_sorted) < 6:
            return {"pearson_effects": [], "markov_effects": [], "summary": "Not enough daily rows for cross-effects."}

        buckets = ["running", "cycling", "swimming", "skiing", "gym", "basketball"]
        outcomes = [
            ("readiness", "Training Readiness"),
            ("sleep", "Sleep Score"),
            ("stress", "Stress"),
            ("hrv", "HRV"),
            ("rhr", "Resting HR"),
        ]

        pearson_effects: List[Dict[str, Any]] = []
        for bucket in buckets:
            x_vals: List[float] = []
            outcome_series: Dict[str, List[Optional[float]]] = {k: [] for k, _ in outcomes}
            for idx in range(len(days_sorted) - 1):
                day = days_sorted[idx]
                nxt = days_sorted[idx + 1]
                x = stimulus_by_day.get(day, {}).get(bucket, 0.0)
                x_vals.append(float(x))
                for outcome_key, _ in outcomes:
                    outcome_series[outcome_key].append(metrics_by_day.get(nxt, {}).get(outcome_key))

            for outcome_key, outcome_label in outcomes:
                corr = _pearson(x_vals, outcome_series[outcome_key])
                if corr is None:
                    continue
                pearson_effects.append(
                    {
                        "sport": bucket,
                        "outcome": outcome_key,
                        "outcome_label": outcome_label,
                        "r": round(float(corr), 4),
                        "window_days": days,
                        "mode": "next_day",
                    }
                )
        pearson_effects.sort(key=lambda item: abs(item["r"]), reverse=True)

        # Markov-style readiness transitions conditioned on sport stimulus.
        markov_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        for idx in range(len(days_sorted) - 1):
            day = days_sorted[idx]
            nxt = days_sorted[idx + 1]
            from_state = _readiness_state(metrics_by_day.get(day, {}).get("readiness"))
            to_state = _readiness_state(metrics_by_day.get(nxt, {}).get("readiness"))
            if not from_state or not to_state:
                continue
            for bucket, load in stimulus_by_day.get(day, {}).items():
                if load <= 0:
                    continue
                markov_counts.setdefault(bucket, {}).setdefault(from_state, {})
                markov_counts[bucket][from_state][to_state] = markov_counts[bucket][from_state].get(to_state, 0) + 1

        markov_effects: List[Dict[str, Any]] = []
        for bucket, from_map in markov_counts.items():
            for from_state, to_map in from_map.items():
                total = sum(to_map.values())
                if total <= 0:
                    continue
                for to_state, count in to_map.items():
                    markov_effects.append(
                        {
                            "sport": bucket,
                            "from_state": from_state,
                            "to_state": to_state,
                            "probability": round(count / total, 4),
                            "count": count,
                        }
                    )
        markov_effects.sort(key=lambda item: item["probability"], reverse=True)

        top = pearson_effects[:3]
        if top:
            summary = " | ".join(
                f"{e['sport']} -> {e['outcome_label']} ({'+' if e['r'] >= 0 else ''}{e['r']:.2f})"
                for e in top
            )
        else:
            summary = "No reliable cross-activity Pearson effects yet."

        return {
            "pearson_effects": pearson_effects[:18],
            "markov_effects": markov_effects[:18],
            "summary": summary,
            "window_days": days,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/insights/latest")
def insights_latest() -> Dict[str, Any]:
    try:
        insights: List[Dict[str, Any]] = []
        weekly_cols = set(_table_columns("weekly_summaries"))
        if weekly_cols and "week_start_date" in weekly_cols:
            key_text_col = "key_insights" if "key_insights" in weekly_cols else None
            rec_col = "recommendations" if "recommendations" in weekly_cols else None
            created_col = "created_at" if "created_at" in weekly_cols else None
            select_parts = ["week_start_date"]
            if created_col:
                select_parts.append(created_col)
            if key_text_col:
                select_parts.append(key_text_col)
            if rec_col:
                select_parts.append(rec_col)
            rows = _fetch_all(
                f"""
                SELECT {", ".join(select_parts)}
                FROM weekly_summaries
                ORDER BY {created_col if created_col else "week_start_date"} DESC
                LIMIT 5
                """
            )
            for r in rows:
                summary_text = _text(r.get(key_text_col)) if key_text_col else ""
                rec_text = _text(r.get(rec_col)) if rec_col else ""
                text = rec_text or summary_text
                structured = _structured_insight(text)
                ts = r.get(created_col) if created_col else r.get("week_start_date")
                if text:
                    insights.append(
                        {
                            "summary": structured["summary"],
                            "insight": structured["summary"],
                            "message": structured["summary"],
                            "text": structured["summary"],
                            "headline": structured["headline"],
                            "what_changed": structured["what_changed"],
                            "why_it_matters": structured["why_it_matters"],
                            "next_24_48h": structured["next_24_48h"],
                            "bullets": structured["bullets"],
                            "agent": "synthesizer",
                            "agent_name": "synthesizer",
                            "source": "weekly_summaries",
                            "timestamp": ts,
                            "created_at": ts,
                            "detail": summary_text or rec_text,
                            "content": summary_text or rec_text,
                        }
                    )

        rec_cols = set(_table_columns("agent_recommendations"))
        if rec_cols and "recommendation" in rec_cols:
            rows = _fetch_all(
                """
                SELECT
                    recommendation,
                    COALESCE(agent_name, 'synthesizer') AS agent_name,
                    week_date,
                    expected_direction,
                    target_metric
                FROM agent_recommendations
                ORDER BY week_date DESC, id DESC
                LIMIT 8
                """
            )
            for r in rows:
                txt = _text(r.get("recommendation"))
                if not txt:
                    continue
                details = f"target={r.get('target_metric')} expected={r.get('expected_direction')}"
                structured = _structured_insight(txt)
                insights.append(
                    {
                        "summary": structured["summary"],
                        "insight": structured["summary"],
                        "message": structured["summary"],
                        "text": structured["summary"],
                        "headline": structured["headline"],
                        "what_changed": structured["what_changed"],
                        "why_it_matters": structured["why_it_matters"],
                        "next_24_48h": structured["next_24_48h"],
                        "bullets": structured["bullets"],
                        "agent": r.get("agent_name"),
                        "agent_name": r.get("agent_name"),
                        "source": "agent_recommendations",
                        "timestamp": r.get("week_date"),
                        "created_at": r.get("week_date"),
                        "detail": details,
                        "content": details,
                    }
                )

        return {"insights": insights, "data": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/chat")
def chat(body: ChatRequest) -> Dict[str, Any]:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        # Optional AI path. If unavailable, fallback returns a deterministic response.
        from crewai import Agent, Crew, Process, Task
        try:
            from src.enhanced_agents import (
                _get_llm,
                analyze_pattern,
                calculate_correlation,
                find_best_days,
                run_sql_query,
            )
        except Exception:
            from enhanced_agents import (
                _get_llm,
                analyze_pattern,
                calculate_correlation,
                find_best_days,
                run_sql_query,
            )

        agent = Agent(
            role="Health Data Analyst",
            goal="Answer with concise, data-backed insights from Garmin data.",
            backstory=(
                "You analyze daily_metrics and activities in PostgreSQL. "
                "Use SQL evidence and keep the answer concise."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[run_sql_query, calculate_correlation, find_best_days, analyze_pattern],
            llm=_get_llm(),
        )
        task = Task(
            description=f"Answer this user question with concrete numbers when possible: {message}",
            agent=agent,
            expected_output="Concise answer with concrete evidence.",
        )
        result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
        answer = getattr(result, "raw", str(result))
    except Exception:
        answer = _fallback_chat(message)

    return {
        "answer": answer,
        "response": answer,
        "message": answer,
        "text": answer,
        "output": answer,
    }
