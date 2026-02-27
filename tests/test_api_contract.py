"""
Contract/behavior tests for src/api.py.

These tests mock DB access and validate:
- strict insight summary structure + limits
- snapshot readiness aliases
- workouts sport filtering
- cross-effects endpoint payload (Pearson + Markov)
"""

import os
import sys
from datetime import date, timedelta

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import api as api_mod
import routes.helpers as helpers_mod


def test_structured_insight_has_required_sections_and_limits():
    raw = """
    Sleep score improved this week, while stress spikes were still present on two evenings.
    This matters because unstable stress can reduce readiness and next-day session quality.
    Recommendation: keep training moderate tomorrow and protect sleep timing tonight.
    """
    parsed = api_mod._structured_insight(raw)

    assert parsed["headline"]
    assert parsed["what_changed"]
    assert parsed["why_it_matters"]
    assert parsed["next_24_48h"]
    assert len(parsed["bullets"]) == 3
    assert all(len(item) <= 280 for item in parsed["bullets"])
    assert parsed["summary"].count("\n") == 2


def test_snapshot_latest_includes_readiness_aliases(monkeypatch):
    mapping = {
        ("daily_metrics", "date"): "date",
        ("daily_metrics", "resting_hr"): "resting_hr",
        ("daily_metrics", "hrv_last_night"): "hrv_last_night",
        ("daily_metrics", "sleep_score"): "sleep_score",
        ("daily_metrics", "bb_peak"): "bb_peak",
        ("daily_metrics", "stress_level"): "stress_level",
        ("daily_metrics", "daily_load_acute"): "daily_load_acute",
        ("daily_metrics", "training_readiness"): "training_readiness",
    }

    def fake_first_existing(table, candidates):
        for c in candidates:
            if (table, c) in mapping:
                return c
        return None

    monkeypatch.setattr(api_mod, "_first_existing", fake_first_existing)
    monkeypatch.setattr(
        api_mod,
        "_fetch_one",
        lambda _q, params=None: {
            "date": "2026-02-20",
            "resting_hr": 54,
            "hrv_last_night": 49,
            "sleep_score": 80,
            "bb_peak": 79,
            "stress_level": 19,
            "daily_load_acute": 795,
            "training_readiness": 72,
        },
    )

    out = api_mod.snapshot_latest()
    assert out["snapshot"]["training_readiness"] == 72
    assert out["snapshot"]["readiness"] == 72
    assert out["snapshot"]["readiness_score"] == 72


def test_workouts_progress_filters_by_sport(monkeypatch):
    col_map = {
        ("activities", "date"): "date",
        ("activities", "activity_name"): "activity_name",
        ("activities", "activity_type"): "activity_type",
        ("activities", "duration_sec"): "duration_sec",
        ("activities", "distance_m"): "distance_m",
        ("activities", "average_hr"): "average_hr",
        ("activities", "average_speed"): "average_speed",
        ("activities", "avg_cadence"): "avg_cadence",
        ("activities", "training_load"): "training_load",
    }

    def fake_first_existing(table, candidates):
        for c in candidates:
            if (table, c) in col_map:
                return c
        return None

    monkeypatch.setattr(api_mod, "_first_existing", fake_first_existing)
    monkeypatch.setattr(
        api_mod,
        "_fetch_all",
        lambda _q, params=None: [
            {
                "workout_date": "2026-02-18",
                "activity_name": "Easy Run",
                "activity_type": "running",
                "duration_sec": 1800,
                "distance_m": 5200,
                "average_hr": 143,
                "average_speed": 3.1,
                "avg_cadence": 165,
                "training_load": 70,
            },
            {
                "workout_date": "2026-02-19",
                "activity_name": "Cycling Session",
                "activity_type": "cycling",
                "duration_sec": 2400,
                "distance_m": 14000,
                "average_hr": 141,
                "average_speed": 5.8,
                "avg_cadence": 82,
                "training_load": 84,
            },
        ],
    )

    out = api_mod.workouts_progress(days=30, sport="running")
    assert out["selected_sport"] == "running"
    assert len(out["data"]) == 1
    assert out["data"][0]["activity_type"] == "running"


def test_cross_effects_returns_pearson_and_markov(monkeypatch):
    def fake_first_existing(table, candidates):
        defaults = {
            "activities": {
                "date",
                "training_load",
                "duration_sec",
                "activity_type",
            },
            "daily_metrics": {
                "date",
                "training_readiness",
                "sleep_score",
                "stress_level",
                "hrv_last_night",
                "resting_hr",
            },
        }
        for c in candidates:
            if c in defaults.get(table, set()):
                return c
        return None

    start = date(2026, 2, 1)
    act_rows = []
    dm_rows = []
    for i in range(10):
        day = (start + timedelta(days=i)).isoformat()
        act_rows.append(
            {
                "d": day,
                "activity_type": "running",
                "training_load": 40 + (i * 6),
                "duration_sec": 2000 + (i * 60),
            }
        )
        dm_rows.append(
            {
                "d": day,
                "readiness": 45 + (i * 4),
                "sleep": 68 + (i * 1.5),
                "stress": 35 - (i * 1.2),
                "hrv": 42 + (i * 1.8),
                "rhr": 58 - (i * 0.6),
            }
        )

    def fake_fetch_all(query, params=None):
        if "FROM activities" in query:
            return act_rows
        if "FROM daily_metrics" in query:
            return dm_rows
        return []

    monkeypatch.setattr(api_mod, "_first_existing", fake_first_existing)
    monkeypatch.setattr(api_mod, "_fetch_all", fake_fetch_all)

    out = api_mod.analytics_cross_effects(days=30)
    assert out["pearson_effects"]
    assert out["markov_effects"]
    assert "running" in out["summary"]


def test_insights_latest_returns_structured_fields(monkeypatch):
    def fake_fetch_all(query, params=None):
        if "information_schema.columns" in query:
            return [
                {"column_name": "week_start_date"},
                {"column_name": "created_at"},
                {"column_name": "key_insights"},
                {"column_name": "recommendations"},
            ]
        if "FROM weekly_summaries" in query:
            return [
                {
                    "week_start_date": "2026-02-17",
                    "created_at": "2026-02-20T06:00:00",
                    "key_insights": (
                        "Sleep improved compared with the previous window.\n"
                        "Stress spikes were observed after late-day hard sessions.\n"
                        "Recommendation: use a lighter day tomorrow and cap effort to moderate intensity."
                    ),
                    "recommendations": "",
                }
            ]
        if "FROM agent_recommendations" in query:
            return []
        return []

    monkeypatch.setattr(api_mod, "_fetch_all", fake_fetch_all)

    out = api_mod.insights_latest()
    assert out["insights"]
    first = out["insights"][0]
    assert first["headline"]
    assert first["what_changed"]
    assert first["why_it_matters"]
    assert first["next_24_48h"]
    assert len(first["bullets"]) == 3
    assert all(len(item) <= 280 for item in first["bullets"])

