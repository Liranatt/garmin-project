from __future__ import annotations

import os
from contextlib import closing
from datetime import datetime
from typing import Any

import psycopg2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Garmin Health API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://liranattar.dev",
        "https://www.liranattar.dev",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _db_url() -> str:
    db_url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_CONNECTION_STRING") or ""
    db_url = db_url.strip()
    # Heroku may provide postgres://; psycopg2 prefers postgresql://
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
    return db_url


if not os.getenv("POSTGRES_CONNECTION_STRING"):
    resolved_db_url = _db_url()
    if resolved_db_url:
        # CrewAI SQL tools in enhanced_agents read this env var directly.
        os.environ["POSTGRES_CONNECTION_STRING"] = resolved_db_url


def get_db_connection():
    db_url = _db_url()
    if not db_url:
        raise RuntimeError("DATABASE_URL / POSTGRES_CONNECTION_STRING is not configured")
    return psycopg2.connect(db_url)


def _to_serializable_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    if "timestamp" in out and out["timestamp"] is not None:
        out["timestamp"] = str(out["timestamp"])
    return out


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _deterministic_chat_reply(message: str) -> str:
    lower_q = message.lower()
    focus = "general"
    if any(k in lower_q for k in ["sleep", "rem", "deep"]):
        focus = "sleep"
    elif any(k in lower_q for k in ["stress", "recover", "readiness"]):
        focus = "recovery"
    elif any(k in lower_q for k in ["train", "workout", "load", "performance"]):
        focus = "training"

    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT date AS timestamp,
                           resting_hr,
                           hrv_last_night AS hrv,
                           sleep_score,
                           bb_peak AS battery,
                           stress_level AS stress,
                           daily_load_acute AS load
                    FROM daily_metrics
                    ORDER BY date DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()
    except Exception:
        row = None

    if row:
        summary = (
            f"Latest snapshot: RHR {row.get('resting_hr')}, "
            f"HRV {row.get('hrv')}, Sleep {row.get('sleep_score')}, "
            f"Battery {row.get('battery')}, Stress {row.get('stress')}, "
            f"Load {row.get('load')}."
        )
    else:
        summary = "No latest snapshot row is available yet."

    if focus == "sleep":
        return f"{summary} For sleep: keep timing consistent and reduce late-night training intensity."
    if focus == "recovery":
        return f"{summary} For recovery: aim for lower stress trend with stable sleep score and HRV."
    if focus == "training":
        return f"{summary} For training: increase load gradually and verify next-day recovery response."
    return f"{summary} Ask specifically about sleep, stress, recovery, or training for a focused analysis."


def _ai_chat_reply(message: str) -> str | None:
    # Keep API boot stable even if AI stack has dependency/runtime issues.
    if not os.getenv("GOOGLE_API_KEY", "").strip():
        return None

    try:
        from crewai import Agent, Crew, Process, Task
        from src.enhanced_agents import (
            _get_llm,
            run_sql_query,
            calculate_correlation,
            find_best_days,
            analyze_pattern,
        )
    except Exception:
        return None

    try:
        agent = Agent(
            role="Health Data Analyst",
            goal="Answer the user with concise, data-backed health insights.",
            backstory=(
                "You analyze the Garmin health PostgreSQL database. "
                "Use SQL tools and provide concrete, short answers."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[run_sql_query, calculate_correlation, find_best_days, analyze_pattern],
            llm=_get_llm(),
        )
        task = Task(
            description=f"User question: {message}",
            agent=agent,
            expected_output="A concise, actionable, data-backed answer.",
        )
        result = Crew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=False,
        ).kickoff()
        raw = getattr(result, "raw", str(result)).strip()
        return raw or None
    except Exception:
        return None


@app.get("/health-check")
def health_check():
    db_state = "not_configured"
    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        db_state = "connected"
    except Exception:
        if _db_url():
            db_state = "unavailable"

    return {
        "status": "Online",
        "database": db_state,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/v1/snapshot/latest")
def get_latest_snapshot():
    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT date AS timestamp,
                           resting_hr,
                           hrv_last_night AS hrv,
                           sleep_score,
                           bb_peak AS battery,
                           stress_level AS stress,
                           daily_load_acute AS load
                    FROM daily_metrics
                    ORDER BY date DESC
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

        if row:
            return {"snapshot": _to_serializable_snapshot(dict(row))}
        return {"snapshot": {}}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/metrics/history")
def get_metrics_history(days: int = 90):
    safe_days = max(7, min(int(days), 365))
    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT date,
                           resting_hr,
                           hrv_last_night AS hrv,
                           sleep_score,
                           stress_level AS stress,
                           bb_peak AS battery,
                           daily_load_acute AS training_load
                    FROM daily_metrics
                    WHERE date >= CURRENT_DATE - (%s * INTERVAL '1 day')
                    ORDER BY date ASC
                    """,
                    (safe_days,),
                )
                rows = cur.fetchall()

        data = []
        for row in rows:
            item = dict(row)
            if item.get("date") is not None:
                item["date"] = str(item["date"])
            data.append(item)

        return {"days": safe_days, "data": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/insights/latest")
def get_latest_insights():
    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                row = None
                # weekly_summaries in this project stores key_insights + recommendations.
                try:
                    cur.execute(
                        """
                        SELECT week_start_date AS timestamp, key_insights, recommendations
                        FROM weekly_summaries
                        ORDER BY week_start_date DESC
                        LIMIT 1
                        """
                    )
                    row = cur.fetchone()
                except Exception:
                    # Backward-compatible fallback if older column name exists.
                    conn.rollback()
                    cur.execute(
                        """
                        SELECT week_start_date AS timestamp, summary_text AS key_insights, recommendations
                        FROM weekly_summaries
                        ORDER BY week_start_date DESC
                        LIMIT 1
                        """
                    )
                    row = cur.fetchone()

        insights = []
        if row:
            ts = str(row.get("timestamp")) if row.get("timestamp") is not None else ""
            key_insights = _safe_text(row.get("key_insights"))
            recommendations = _safe_text(row.get("recommendations"))

            if key_insights:
                insights.append(
                    {
                        "agent": "Medical Synthesizer",
                        "text": key_insights,
                        "timestamp": ts,
                    }
                )
            if recommendations:
                insights.append(
                    {
                        "agent": "Action Agent",
                        "text": recommendations,
                        "timestamp": ts,
                    }
                )

        return {"insights": insights}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ChatQuery(BaseModel):
    message: str


@app.post("/api/v1/chat")
def chat_endpoint(query: ChatQuery):
    message = query.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")

    try:
        ai_reply = _ai_chat_reply(message)
        if ai_reply:
            lower = ai_reply.lower()
            failure_markers = [
                "cannot connect",
                "unable to connect",
                "database connection error",
                "sql error",
                "connection error",
            ]
            if not any(marker in lower for marker in failure_markers):
                return {"answer": ai_reply}

        return {"answer": _deterministic_chat_reply(message)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
