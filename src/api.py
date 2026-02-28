"""
FastAPI backend contract for gps_presentation frontend integration.

Route handlers are defined here; shared utilities live in routes/helpers.py.
"""

from __future__ import annotations
import uuid
from fastapi import BackgroundTasks
import logging
import os
os.environ["CREWAI_DISABLE_SIGTERM"] = "true"
import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from routes.helpers import (
    _conn_str, _fetch_all, _fetch_one, _first_existing,
    _num, _text, _pick_row_value,
    _build_snapshot_payload, _build_history_rows,
    _workout_summary, _structured_insight,
    _is_low_signal_text, _insight_rank,
    _matches_sport, _sport_bucket, _pearson, _readiness_state,
    _parse_activity_date, _speed_to_kph,
    _clip_text, _window_note, _fallback_chat,
)

try:
    from pipeline.migrations import schema_audit
except ImportError:
    schema_audit = None

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    _limiter = Limiter(key_func=get_remote_address)
    _HAS_SLOWAPI = True
except ImportError:
    _HAS_SLOWAPI = False
    _limiter = None

log = logging.getLogger("api")


# ─── App setup ─────────────────────────────────────────────

app = FastAPI(title="Garmin Health API", version="1.0.0")

if _HAS_SLOWAPI and _limiter:
    app.state.limiter = _limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


# ─── Routes ────────────────────────────────────────────────

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

        # Build daily_metrics lookup
        dm_by_day: Dict[str, Dict[str, Any]] = {}
        for row in dm_rows:
            day = _parse_activity_date(row.get("d"))
            if day:
                dm_by_day[day] = row

        sorted_days = sorted(dm_by_day.keys())

        # Pearson effects: for each sport bucket, correlate stimulus with next-day outcomes
        sport_buckets = sorted({b for stim in stimulus_by_day.values() for b in stim})
        pearson_effects: List[Dict[str, Any]] = []
        for bucket in sport_buckets:
            for outcome_key in ("readiness", "sleep", "stress", "hrv", "rhr"):
                xs: List[float] = []
                ys: List[float] = []
                for i, day in enumerate(sorted_days[:-1]):
                    next_day = sorted_days[i + 1]
                    stim = (stimulus_by_day.get(day) or {}).get(bucket)
                    if stim is None:
                        continue
                    outcome = _num((dm_by_day.get(next_day) or {}).get(outcome_key))
                    if outcome is None:
                        continue
                    xs.append(stim)
                    ys.append(outcome)
                r = _pearson(xs, ys)
                if r is not None:
                    pearson_effects.append({
                        "sport": bucket,
                        "outcome": outcome_key,
                        "r": round(r, 3),
                        "n": len(xs),
                        "interpretation": (
                            "strong" if abs(r) > 0.7 else
                            "moderate" if abs(r) > 0.4 else
                            "weak" if abs(r) > 0.2 else "negligible"
                        ),
                    })

        # Markov effects: readiness state transitions after each sport
        markov_effects: List[Dict[str, Any]] = []
        readiness_key = "readiness"
        for bucket in sport_buckets:
            transitions: Dict[str, Dict[str, int]] = {}
            for i, day in enumerate(sorted_days[:-1]):
                next_day = sorted_days[i + 1]
                stim = (stimulus_by_day.get(day) or {}).get(bucket)
                if stim is None:
                    continue
                state_today = _readiness_state((dm_by_day.get(day) or {}).get(readiness_key))
                state_tomorrow = _readiness_state((dm_by_day.get(next_day) or {}).get(readiness_key))
                if state_today is None or state_tomorrow is None:
                    continue
                transitions.setdefault(state_today, {})
                transitions[state_today][state_tomorrow] = transitions[state_today].get(state_tomorrow, 0) + 1

            if transitions:
                markov_effects.append({
                    "sport": bucket,
                    "transitions": transitions,
                    "total_observations": sum(
                        sum(to_counts.values())
                        for to_counts in transitions.values()
                    ),
                })

        summary_parts = []
        for pe in sorted(pearson_effects, key=lambda x: abs(x["r"]), reverse=True)[:5]:
            summary_parts.append(
                f"{pe['sport']} → next-day {pe['outcome']}: r={pe['r']:.2f} ({pe['interpretation']}, n={pe['n']})"
            )
        cross_summary = "; ".join(summary_parts) if summary_parts else "Not enough cross-effect data yet."

        return {
            "pearson_effects": pearson_effects,
            "markov_effects": markov_effects,
            "summary": cross_summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/insights/latest")
def insights_latest() -> Dict[str, Any]:
    try:
        insights: List[Dict[str, Any]] = []

        # Source 1: weekly_summaries
        try:
            weekly_cols = {r["column_name"] for r in _fetch_all(
                """SELECT column_name FROM information_schema.columns
                   WHERE table_schema='public' AND table_name='weekly_summaries'"""
            )}
        except Exception:
            weekly_cols = set()

        if weekly_cols:
            pipeline_status_col = "pipeline_status_json" if "pipeline_status_json" in weekly_cols else None
            select_parts = ["week_start_date"]
            for col in ("key_insights", "recommendations", "analysis_status"):
                if col in weekly_cols:
                    select_parts.append(col)
            if pipeline_status_col:
                select_parts.append(pipeline_status_col)

            rows = _fetch_all(
                f"""SELECT {', '.join(select_parts)} FROM weekly_summaries
                    ORDER BY week_start_date DESC LIMIT 4"""
            )

            for r in rows:
                raw_pipeline_status = r.get("pipeline_status_json")
                pipeline_status = raw_pipeline_status
                if isinstance(raw_pipeline_status, str):
                    try:
                        pipeline_status = json.loads(raw_pipeline_status)
                    except Exception:
                        pipeline_status = {"raw": raw_pipeline_status}

                raw_insights = _text(r.get("key_insights"))
                raw_recs = _text(r.get("recommendations"))

                if raw_insights and not _is_low_signal_text(raw_insights):
                    structured = _structured_insight(raw_insights)
                    item = {
                        "source": "weekly_summaries",
                        "type": "weekly_insight",
                        "timestamp": r.get("week_start_date"),
                        "summary": structured["headline"],
                        "detail": structured["summary"],
                        "content": raw_insights,
                        "structured": structured,
                        "analysis_status": r.get("analysis_status", "unknown"),
                        "pipeline_status": pipeline_status,
                    }
                    item.update(structured)
                    insights.append(item)

                if raw_recs and not _is_low_signal_text(raw_recs):
                    structured_rec = _structured_insight(raw_recs)
                    item = {
                        "source": "weekly_summaries",
                        "type": "recommendation",
                        "timestamp": r.get("week_start_date"),
                        "summary": structured_rec["headline"],
                        "detail": structured_rec["summary"],
                        "content": raw_recs,
                        "structured": structured_rec,
                        "analysis_status": r.get("analysis_status", "unknown"),
                        "pipeline_status": pipeline_status,
                    }
                    item.update(structured_rec)
                    insights.append(item)

        # Source 2: agent_recommendations
        try:
            rec_rows = _fetch_all(
                """SELECT week_date, agent_name, recommendation,
                          target_metric, expected_direction, status, outcome_notes
                   FROM agent_recommendations
                   ORDER BY week_date DESC, id DESC LIMIT 20"""
            )
            for rec in rec_rows:
                text = _text(rec.get("recommendation"))
                if text and not _is_low_signal_text(text):
                    structured = _structured_insight(text)
                    item = {
                        "source": "agent_recommendations",
                        "type": "recommendation",
                        "timestamp": rec.get("week_date"),
                        "summary": _clip_text(text, 120),
                        "detail": text,
                        "content": text,
                        "structured": structured,
                        "agent_name": rec.get("agent_name"),
                        "target_metric": rec.get("target_metric"),
                        "expected_direction": rec.get("expected_direction"),
                        "status": rec.get("status"),
                        "outcome_notes": rec.get("outcome_notes"),
                    }
                    item.update(structured)
                    insights.append(item)
        except Exception:
            pass

        # Deduplicate by content hash
        if insights:
            dedup: Dict[str, Dict[str, Any]] = {}
            for item in insights:
                key = _text(item.get("content"))[:200]
                if key not in dedup:
                    dedup[key] = item
            insights = list(dedup.values()) if dedup else insights

            insights.sort(key=_insight_rank, reverse=True)
            insights = insights[:12]

        return {"insights": insights, "data": insights}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Rate limit decorator (no-op if slowapi not installed)
_rate_limit = _limiter.limit("5/minute") if _HAS_SLOWAPI and _limiter else lambda f: f

# מילון בזיכרון לשמירת מצב המשימות והתשובות
active_chat_jobs: Dict[str, Dict[str, Any]] = {}

def run_chat_agent_background(job_id: str, message: str):
    """
    פונקציה זו רצה ברקע, מפעילה את הסוכנים, ושומרת את התוצאה במילון.
    """
    try:
        from crewai import Agent, Crew, Process, Task
        try:
            from src.enhanced_agents import (
                _get_llm, analyze_pattern, calculate_correlation,
                find_best_days, run_sql_query, AdvancedHealthAgents
            )
        except Exception:
            from enhanced_agents import (
                _get_llm, analyze_pattern, calculate_correlation,
                find_best_days, run_sql_query, AdvancedHealthAgents
            )

        # ── Part 1: Complexity Routing ──
        COMPLEX_KEYWORDS = [
            "report", "compare", "analysis", "deep dive", "breakdown",
            "trend", "over time", "getting better", "getting worse",
            "overtraining", "recovery pattern", "what affects",
            "cross-training", "all my workouts", "weekly", "summarize",
            "correlate", "in depth", "in-depth"
        ]
        is_complex = any(kw in message.lower() for kw in COMPLEX_KEYWORDS)

        answer = ""

        if is_complex:
            try:
                import time
                crew_builder = AdvancedHealthAgents()
                tasks = crew_builder.create_deep_analysis_tasks(analysis_period=30)
                
                synth_task = tasks[-1]
                synth_task.description = f"USER QUESTION: {message}\n\n" + synth_task.description
                
                agents = [t.agent for t in tasks]
                accumulated_text = ""
                
                for idx, (agent, task) in enumerate(zip(agents, tasks)):
                    if "Synthesizer" in agent.role:
                        task.description += f"\n\n=== PREVIOUS AGENT FINDINGS ===\n{accumulated_text}"
                        
                    single_crew = Crew(
                        agents=[agent],
                        tasks=[task],
                        process=Process.sequential,
                        verbose=False
                    )
                    
                    out = single_crew.kickoff()
                    raw_text = getattr(out, "raw", str(out))
                    
                    accumulated_text += f"\n--- {agent.role} ---\n{raw_text}\n"
                    
                    if "Synthesizer" in agent.role:
                        answer = raw_text
                        
                    if idx < len(agents) - 1:
                        time.sleep(1) # הורדנו את ההמתנה הארוכה כי אנחנו ברקע!
                        
            except Exception:
                pass # יפול לסוכן הפשוט

        # ── Part 2: Upgraded Simple Agent ──
        if not answer:
            try:
                from routes.helpers import _fetch_one
                baselines = _fetch_one("""
                    SELECT 
                        ROUND(AVG(sleep_score)::numeric, 1) AS avg_sleep,
                        ROUND(AVG(training_readiness)::numeric, 1) AS avg_readiness,
                        ROUND(AVG(hrv_last_night)::numeric, 1) AS avg_hrv,
                        ROUND(AVG(stress_level)::numeric, 1) AS avg_stress,
                        ROUND(AVG(resting_hr)::numeric, 1) AS avg_rhr,
                        ROUND(AVG(total_steps)::numeric, 0) AS avg_steps,
                        ROUND(AVG(bb_charged)::numeric, 1) AS avg_bb_charged
                    FROM daily_metrics
                    WHERE date >= CURRENT_DATE - INTERVAL '30 days'
                """)
                baseline_ctx = (
                    f"\n\nUSER'S 30-DAY BASELINES: "
                    f"sleep_score={baselines.get('avg_sleep')}, "
                    f"training_readiness={baselines.get('avg_readiness')}, "
                    f"HRV={baselines.get('avg_hrv')}ms, "
                    f"stress={baselines.get('avg_stress')}, "
                    f"RHR={baselines.get('avg_rhr')}bpm, "
                    f"steps={baselines.get('avg_steps')}, "
                    f"BB_charged={baselines.get('avg_bb_charged')}"
                )
            except Exception:
                baseline_ctx = ""

            try:
                temp_crew = AdvancedHealthAgents()
                full_rules_ctx = temp_crew.ctx
            except Exception:
                full_rules_ctx = "RULES: Maintain a strict evidentiary diagnostic approach."

            agent = Agent(
                role="Diagnostic Health Analyst",
                goal="Answer user questions using strict 3-layer diagnostic reasoning.",
                backstory=(
                    "You are an elite diagnostic health analyst.\n"
                    f"{full_rules_ctx}\n\n"
                    "CHAT RULES:\n"
                    "1. Keep answers conversational but deeply analytical.\n"
                    "2. NO generic advice. NEVER say 'likely due to'. Use empirical data.\n"
                    "3. Convert raw seconds to hours/minutes."
                ),
                verbose=False,
                allow_delegation=False,
                tools=[run_sql_query, calculate_correlation, find_best_days, analyze_pattern],
                llm=_get_llm(),
            )

            task = Task(
                description=(
                    f"User question: {message}"
                    f"{baseline_ctx}"
                    "\n\nSTEPS:\n"
                    "1. Query the relevant data from daily_metrics (include today, yesterday AND the day before)\n"
                    "2. LAYER 1: State what happened (the surface fact)\n"
                    "3. LAYER 2: Evaluate adjacent variables (training_load, BB drained, stress, prep data)\n"
                    "4. LAYER 3: Investigate mechanics (HRV, sleeping HR, deep sleep, respiration)\n"
                    "5. Follow the timeline: For causal questions ('what caused X'), do not stop at yesterday. Trace the data chronologically backwards to find where the shift originated.\n"
                    "6. Write your final answer.\n\n"
                    "MANDATORY OUTPUT FORMAT:\n"
                    "You MUST format your output using these exact headers:\n"
                    "### The Facts (Layer 1)\n"
                    "### Surface Drivers (Layer 2)\n"
                    "### Deep Mechanics (Layer 3)\n"
                    "### Conclusion\n\n"
                    "FORBIDDEN: 'likely due to', 'probably because', 'caused by' without naming the exact metric."
                ),
                expected_output="A diagnostic answer strictly following the 4 mandatory markdown headers."
            )

            result = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            ).kickoff()
            answer = getattr(result, "raw", str(result))

        # עדכון המילון כשהכל הסתיים בהצלחה
        active_chat_jobs[job_id] = {
            "status": "completed",
            "answer": answer,
            "response": answer,
            "message": answer,
            "text": answer,
            "output": answer,
        }

    except Exception as e:
        # במקרה של קריסה מלאה, נשתמש בסוכן הגיבוי ונשמור את תשובתו
        fallback = _fallback_chat(message)
        active_chat_jobs[job_id] = {
            "status": "completed",
            "answer": fallback,
            "response": fallback,
            "message": fallback,
            "text": fallback,
            "output": fallback,
            "error": str(e) # לשם דיבוג אם תצטרך
        }

@app.post("/api/v1/chat")
@_rate_limit
def chat(body: ChatRequest, request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    # 1. יצירת מזהה ייחודי לבקשה
    job_id = str(uuid.uuid4())
    
    # 2. רישום המשימה במילון כ"מעובדת"
    active_chat_jobs[job_id] = {"status": "processing"}

    # 3. שיגור הפונקציה הכבדה לריצה ב-Background Thread
    background_tasks.add_task(run_chat_agent_background, job_id, message)

    # 4. החזרת תשובה ל-Frontend תוך חלקיק שנייה! (Heroku מרוצה)
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Agent is thinking..."
    }

@app.get("/api/v1/admin/migration-audit")
def migration_audit() -> Dict[str, Any]:
    if schema_audit is None:
        raise HTTPException(status_code=500, detail="migration module not available")
    try:
        return schema_audit(_conn_str())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/chat/status/{job_id}")
def get_chat_status(job_id: str) -> Dict[str, Any]:
    job_data = active_chat_jobs.get(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    
    return job_data
