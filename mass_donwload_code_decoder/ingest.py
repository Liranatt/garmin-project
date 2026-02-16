"""
Garmin JSON → PostgreSQL ingestion pipeline.

Run:  python ingest.py          (full pipeline)
      python ingest.py --init   (recreate schema first)

Only records from DATE_FROM (default 2026-02-01) onwards are inserted.
"""

import json
import sys
import os
import glob
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from config import CONNECT_DIR, DATE_FROM
from db import get_connection, init_schema

# ─────────────────────── helpers ───────────────────────

def load_json(path: str | Path) -> Any:
    """Load and return JSON from a file path."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_date(value: str | int | None) -> date | None:
    """Convert a string like '2026-02-11' or epoch-ms to a date object."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.utcfromtimestamp(value / 1000).date()
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def after_cutoff(d: date | None) -> bool:
    """Return True if *d* is on or after DATE_FROM."""
    return d is not None and d >= DATE_FROM


def glob_files(subdir: str, pattern: str) -> list[Path]:
    """Return matching JSON files under CONNECT_DIR/subdir."""
    base = CONNECT_DIR / subdir
    return sorted(base.glob(pattern))


def upsert(cur, sql: str, params: tuple):
    """Execute an upsert statement, silently skip duplicates."""
    try:
        cur.execute(sql, params)
    except psycopg2.IntegrityError:
        cur.connection.rollback()


# ─────────────────────── ingest functions ───────────────────────

def ingest_user_profile(cur):
    path = CONNECT_DIR / "DI-Connect-User" / "user_profile.json"
    if not path.exists():
        return
    d = load_json(path)
    cur.execute("""
        INSERT INTO garmin.user_profile (user_name, first_name, last_name, email, gender, birth_date)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_name) DO UPDATE SET
            first_name=EXCLUDED.first_name, last_name=EXCLUDED.last_name,
            email=EXCLUDED.email, gender=EXCLUDED.gender, birth_date=EXCLUDED.birth_date
    """, (d.get("userName"), d.get("firstName"), d.get("lastName"),
          d.get("emailAddress"), d.get("gender"), d.get("birthDate")))
    print(f"  [user_profile] 1 row")


def ingest_user_settings(cur):
    path = CONNECT_DIR / "DI-Connect-User" / "user_settings.json"
    if not path.exists():
        return
    d = load_json(path)
    cur.execute("DELETE FROM garmin.user_settings")
    cur.execute("""
        INSERT INTO garmin.user_settings (preferred_locale, handedness, use_fixed_location)
        VALUES (%s, %s, %s)
    """, (d.get("preferredLocale"), d.get("handedness"), d.get("useFixedLocation")))
    print(f"  [user_settings] 1 row")


def ingest_social_profile(cur):
    for p in glob_files("DI-Connect-User", "*social-profile.json"):
        d = load_json(p)
        cur.execute("""
            INSERT INTO garmin.social_profile
                (display_name, profile_image_url_large, profile_image_url_medium,
                 profile_image_url_small, user_profile_number, garmin_guid,
                 level, level_update_date, level_is_upper_cap, raw_json)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (display_name) DO UPDATE SET raw_json=EXCLUDED.raw_json
        """, (d.get("displayName"),
              d.get("profileImageUrlLarge"), d.get("profileImageUrlMedium"),
              d.get("profileImageUrlSmall"), d.get("userProfileNumber"),
              d.get("garminGUID"), d.get("level"), str(d.get("levelUpdateDate")) if d.get("levelUpdateDate") else None,
              d.get("levelIsUpperCap"), json.dumps(d)))
    print(f"  [social_profile] done")


def ingest_device_backup(cur):
    for p in glob_files("DI-Connect-Device", "*DeviceBackups.json"):
        rows = load_json(p)
        for d in rows:
            uuid_val = d.get("primaryUUID", {}).get("uuid") if isinstance(d.get("primaryUUID"), dict) else None
            cur.execute("""
                INSERT INTO garmin.device_backup (device_backup_id, user_id, device_id, primary_uuid, create_date, pending_restore)
                VALUES (%s,%s,%s,%s,%s,%s)
                ON CONFLICT (device_backup_id) DO NOTHING
            """, (d.get("deviceBackupId"), d.get("userId"), d.get("deviceId"),
                  uuid_val, d.get("createDate"), d.get("pendingRestore")))
    print(f"  [device_backup] done")


def ingest_sleep(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Wellness", "*sleepData.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue

            # Extract score details safely
            scores = d.get("sleepScores", {}) or {}
            overall = scores.get("overall", {}) or {}
            rem_sc = scores.get("rem", {}) or {}
            deep_sc = scores.get("deep", {}) or {}
            light_sc = scores.get("light", {}) or {}

            cur.execute("""
                INSERT INTO garmin.sleep (
                    calendar_date, sleep_start_ts_gmt, sleep_end_ts_gmt,
                    sleep_start_ts_local, sleep_end_ts_local,
                    unmeasurable_sleep_seconds, deep_sleep_seconds, light_sleep_seconds,
                    rem_sleep_seconds, awake_sleep_seconds,
                    average_spo2_value, lowest_spo2_value, highest_spo2_value,
                    average_spo2_hr_sleep,
                    average_respiration_value, lowest_respiration_value, highest_respiration_value,
                    avg_sleep_stress, sleep_score_feedback,
                    overall_score_value, overall_score_qualifier,
                    rem_percentage_value, rem_optimal_start, rem_optimal_end,
                    deep_percentage_value, deep_optimal_start, deep_optimal_end,
                    light_percentage_value, light_optimal_start, light_optimal_end,
                    duration_in_ms, awakenings_count_value,
                    average_stress_during_sleep, age_group, sleep_result_type,
                    resting_heart_rate, raw_json
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s
                ) ON CONFLICT (calendar_date) DO UPDATE SET raw_json=EXCLUDED.raw_json
            """, (
                str(cd),
                str(d.get("sleepStartTimestampGMT")) if d.get("sleepStartTimestampGMT") is not None else None,
                str(d.get("sleepEndTimestampGMT")) if d.get("sleepEndTimestampGMT") is not None else None,
                str(d.get("sleepStartTimestampLocal")) if d.get("sleepStartTimestampLocal") is not None else None,
                str(d.get("sleepEndTimestampLocal")) if d.get("sleepEndTimestampLocal") is not None else None,
                d.get("unmeasurableSleepSeconds"), d.get("deepSleepSeconds"),
                d.get("lightSleepSeconds"), d.get("remSleepSeconds"),
                d.get("awakeSleepSeconds"),
                d.get("averageSpO2Value"), d.get("lowestSpO2Value"),
                d.get("highestSpO2Value"), d.get("averageSpO2HRSleep"),
                d.get("averageRespirationValue"), d.get("lowestRespirationValue"),
                d.get("highestRespirationValue"), d.get("avgSleepStress"),
                d.get("sleepScoreFeedback"),
                overall.get("value"), overall.get("qualifierKey"),
                rem_sc.get("value"), rem_sc.get("optimalStart"), rem_sc.get("optimalEnd"),
                deep_sc.get("value"), deep_sc.get("optimalStart"), deep_sc.get("optimalEnd"),
                light_sc.get("value"), light_sc.get("optimalStart"), light_sc.get("optimalEnd"),
                d.get("sleepDurationInMillis"),
                (d.get("sleepScores", {}) or {}).get("awakeningsCount", {}).get("value") if isinstance((d.get("sleepScores", {}) or {}).get("awakeningsCount"), dict) else None,
                d.get("averageStressDuringSleep"), d.get("ageGroup"),
                d.get("sleepResultType"), d.get("restingHeartRate"),
                json.dumps(d),
            ))
            cnt += 1
    print(f"  [sleep] {cnt} rows")


def ingest_daily_summary(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Aggregator", "UDSFile_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.daily_summary (
                    calendar_date, user_profile_pk,
                    total_kilocalories, active_kilocalories, bmr_kilocalories,
                    total_steps, total_distance_meters,
                    highly_active_seconds, active_seconds, sedentary_seconds, sleeping_seconds,
                    moderate_intensity_minutes, vigorous_intensity_minutes,
                    floors_ascended, floors_descended,
                    min_heart_rate, max_heart_rate, resting_heart_rate,
                    avg_stress_level, max_stress_level, stress_duration,
                    rest_stress_duration, activity_stress_duration,
                    low_stress_duration, medium_stress_duration, high_stress_duration,
                    body_battery_charged_value, body_battery_drained_value,
                    body_battery_highest_value, body_battery_lowest_value, body_battery_most_recent,
                    average_spo2, lowest_spo2, latest_spo2,
                    avg_waking_respiration_value, highest_respiration_value, lowest_respiration_value,
                    raw_json
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s
                ) ON CONFLICT (calendar_date) DO UPDATE SET raw_json=EXCLUDED.raw_json
            """, (
                str(cd), d.get("userProfilePK"),
                d.get("totalKilocalories"), d.get("activeKilocalories"), d.get("bmrKilocalories"),
                d.get("totalSteps"), d.get("totalDistanceMeters"),
                d.get("highlyActiveSeconds"), d.get("activeSeconds"),
                d.get("sedentarySeconds"), d.get("sleepingSeconds"),
                d.get("moderateIntensityMinutes"), d.get("vigorousIntensityMinutes"),
                d.get("floorsAscended"), d.get("floorsDescended"),
                d.get("minHeartRate"), d.get("maxHeartRate"), d.get("restingHeartRate"),
                d.get("averageStressLevel"), d.get("maxStressLevel"),
                d.get("stressDuration"), d.get("restStressDuration"),
                d.get("activityStressDuration"), d.get("lowStressDuration"),
                d.get("mediumStressDuration"), d.get("highStressDuration"),
                d.get("bodyBatteryChargedValue"), d.get("bodyBatteryDrainedValue"),
                d.get("bodyBatteryHighestValue"), d.get("bodyBatteryLowestValue"),
                d.get("bodyBatteryMostRecentValue"),
                d.get("averageSpo2"), d.get("lowestSpo2"), d.get("latestSpo2"),
                d.get("avgWakingRespirationValue"), d.get("highestRespirationValue"),
                d.get("lowestRespirationValue"),
                json.dumps(d),
            ))
            cnt += 1
    print(f"  [daily_summary] {cnt} rows")


def ingest_activities(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Fitness", "*summarizedActivities.json"):
        raw = load_json(p)
        # File wraps activities in [{"summarizedActivitiesExport": [...]}]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "summarizedActivitiesExport" in raw[0]:
            rows = raw[0]["summarizedActivitiesExport"]
        else:
            rows = raw if isinstance(raw, list) else [raw]
        for d in rows:
            # Timestamps are epoch-ms in this export format
            cd = parse_date(d.get("startTimeLocal") or d.get("beginTimestamp"))
            if not after_cutoff(cd):
                continue
            # activityType is a plain string here, not a dict
            at = d.get("activityType", "")
            if isinstance(at, dict):
                at = at.get("typeKey", "")
            cur.execute("""
                INSERT INTO garmin.activity (
                    activity_id, activity_name, start_time_local, start_time_gmt,
                    activity_type, sport_type, event_type,
                    distance, duration, elapsed_duration, moving_duration,
                    elevation_gain, elevation_loss,
                    average_speed, max_speed, average_hr, max_hr,
                    average_running_cadence, max_running_cadence,
                    calories, bmr_calories, steps, average_stride_length,
                    training_effect_label, aerobic_training_effect, anaerobic_training_effect,
                    training_effect_label_anaerobic,
                    vo2_max_value, lactate_threshold,
                    min_temperature, max_temperature,
                    start_latitude, start_longitude, end_latitude, end_longitude,
                    location_name,
                    moderate_intensity_minutes, vigorous_intensity_minutes,
                    pr_flag, manual_activity, auto_calc_calories,
                    average_respiration_rate, training_load, water_estimated,
                    begin_timestamp, raw_json
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s
                ) ON CONFLICT (activity_id) DO UPDATE SET raw_json=EXCLUDED.raw_json
            """, (
                d.get("activityId"), d.get("name"),
                str(d.get("startTimeLocal")) if d.get("startTimeLocal") else None,
                str(d.get("startTimeGmt")) if d.get("startTimeGmt") else None,
                at, d.get("sportType"), d.get("eventTypeId"),
                d.get("distance"), d.get("duration"),
                d.get("elapsedDuration"), d.get("movingDuration"),
                d.get("elevationGain"), d.get("elevationLoss"),
                d.get("avgSpeed"), d.get("maxSpeed"),
                d.get("avgHr"), d.get("maxHr"),
                d.get("avgRunningCadenceInStepsPerMinute"),
                d.get("maxRunningCadenceInStepsPerMinute"),
                d.get("calories"), d.get("bmrCalories"),
                d.get("steps"), d.get("avgStrideLength"),
                d.get("trainingEffectLabel"), d.get("aerobicTrainingEffect"),
                d.get("anaerobicTrainingEffect"),
                d.get("anaerobicTrainingEffectMessage"),
                d.get("vO2MaxValue"), str(d.get("lactateThreshold")) if d.get("lactateThreshold") else None,
                d.get("minTemperature"), d.get("maxTemperature"),
                d.get("startLatitude"), d.get("startLongitude"),
                d.get("endLatitude"), d.get("endLongitude"),
                d.get("locationName"),
                d.get("moderateIntensityMinutes"), d.get("vigorousIntensityMinutes"),
                d.get("pr"), d.get("manualActivity"), d.get("autoCalcCalories"),
                d.get("averageRespirationRate"), d.get("activityTrainingLoad"),
                d.get("waterEstimated"),
                d.get("beginTimestamp"),
                json.dumps(d),
            ))
            cnt += 1
    print(f"  [activity] {cnt} rows")


def ingest_training_history(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "TrainingHistory_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.training_history (
                    calendar_date, user_profile_pk, device_id, update_timestamp,
                    training_load_balance, weekly_acute_chronic_workload_ratio,
                    weekly_aerobic_excess_post, weekly_anaerobic_excess_post,
                    daily_training_load_acute, daily_training_load_chronic,
                    training_load_balance_feedback, raw_json
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date) DO UPDATE SET raw_json=EXCLUDED.raw_json
            """, (
                str(cd), d.get("userProfilePK"), d.get("deviceId"),
                d.get("updateTimestamp"),
                d.get("trainingLoadBalance"),
                d.get("weeklyAcuteChronicWorkloadRatio"),
                d.get("weeklyAerobicExcessPost"), d.get("weeklyAnaerobicExcessPost"),
                d.get("dailyTrainingLoadAcute"), d.get("dailyTrainingLoadChronic"),
                d.get("trainingLoadBalanceFeedback"),
                json.dumps(d),
            ))
            cnt += 1
    print(f"  [training_history] {cnt} rows")


def ingest_training_readiness(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "TrainingReadinessDTO_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            # "score" is a plain number (or None), not a dict
            overall_score = d.get("score")
            # componentScores may be None or a list of dicts
            cs = d.get("componentScores") or []
            components = {}
            if isinstance(cs, list):
                for c in cs:
                    if isinstance(c, dict):
                        components[c.get("readinessComponent")] = c.get("score", 0)
            cur.execute("""
                INSERT INTO garmin.training_readiness (
                    calendar_date, user_profile_pk, device_id, timestamp,
                    training_readiness_level, overall_score,
                    recovery_score, training_load_score, sleep_score,
                    hrv_score, stress_history_score, raw_json
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date) DO UPDATE SET raw_json=EXCLUDED.raw_json
            """, (
                str(cd), d.get("userProfilePK"), d.get("deviceId"),
                d.get("timestamp"),
                d.get("level"), overall_score,
                components.get("RECOVERY_TIME"), components.get("TRAINING_LOAD_BALANCE"),
                components.get("SLEEP"), components.get("HRV"),
                components.get("STRESS_HISTORY"),
                json.dumps(d),
            ))
            cnt += 1
    print(f"  [training_readiness] {cnt} rows")


def ingest_vo2max(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "MetricsMaxMetData_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.vo2max (
                    calendar_date, user_profile_pk, device_id, update_timestamp,
                    sport, sub_sport, vo2_max_value, max_met, max_met_category, calibrated_data
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date, sport) DO UPDATE SET
                    vo2_max_value=EXCLUDED.vo2_max_value, max_met=EXCLUDED.max_met
            """, (
                str(cd), d.get("userProfilePK"), d.get("deviceId"),
                d.get("updateTimestamp"),
                d.get("sport"), d.get("subSport"),
                d.get("vo2MaxValue"), d.get("maxMet"),
                d.get("maxMetCategory"), d.get("calibratedData"),
            ))
            cnt += 1
    print(f"  [vo2max] {cnt} rows")


def ingest_run_race_predictions(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "RunRacePredictions_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.run_race_predictions (
                    calendar_date, user_profile_pk, device_id, timestamp,
                    race_time_5k, race_time_10k, race_time_half, race_time_marathon
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date) DO UPDATE SET
                    race_time_5k=EXCLUDED.race_time_5k, race_time_10k=EXCLUDED.race_time_10k,
                    race_time_half=EXCLUDED.race_time_half, race_time_marathon=EXCLUDED.race_time_marathon
            """, (
                str(cd), d.get("userProfilePK"), d.get("deviceId"),
                d.get("timestamp"),
                d.get("raceTime5K"), d.get("raceTime10K"),
                d.get("raceTimeHalf"), d.get("raceTimeMarathon"),
            ))
            cnt += 1
    print(f"  [run_race_predictions] {cnt} rows")


def ingest_cycling_ability(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "CyclingAbility_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.cycling_ability (
                    calendar_date, user_profile_pk, device_id, timestamp,
                    aerobic_endurance, aerobic_capacity, anaerobic_capacity,
                    profile_type, profile_type_feedback,
                    aerobic_endurance_feedback, aerobic_capacity_feedback, anaerobic_capacity_feedback
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date) DO UPDATE SET
                    aerobic_endurance=EXCLUDED.aerobic_endurance
            """, (
                str(cd), d.get("userProfilePk"), d.get("deviceId"),
                d.get("timestamp"),
                d.get("aerobicEndurance"), d.get("aerobicCapacity"),
                d.get("anaerobicCapacity"), d.get("profileType"),
                d.get("profileTypeFeedback"),
                d.get("aerobicEnduranceFeedback"), d.get("aerobicCapacityFeedback"),
                d.get("anaerobicCapacityFeedback"),
            ))
            cnt += 1
    print(f"  [cycling_ability] {cnt} rows")


def ingest_heat_altitude_acclimation(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "MetricsHeatAltitudeAcclimation_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.heat_altitude_acclimation (
                    calendar_date, user_profile_pk, device_id, timestamp,
                    altitude_acclimation, previous_altitude_acclimation,
                    heat_acclimation_percentage, previous_heat_acclimation_pct,
                    current_altitude, prev_altitude,
                    acclimation_percentage, prev_acclimation_percentage
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date) DO UPDATE SET
                    altitude_acclimation=EXCLUDED.altitude_acclimation
            """, (
                str(cd), d.get("userProfilePK"), d.get("deviceId"),
                d.get("timestamp"),
                d.get("altitudeAcclimation"), d.get("previousAltitudeAcclimation"),
                d.get("heatAcclimationPercentage"), d.get("previousHeatAcclimationPercentage"),
                d.get("currentAltitude"), d.get("prevAltitude"),
                d.get("acclimationPercentage"), d.get("prevAcclimationPercentage"),
            ))
            cnt += 1
    print(f"  [heat_altitude_acclimation] {cnt} rows")


def ingest_acute_training_load(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Metrics", "MetricsAcuteTrainingLoad_*.json"):
        rows = load_json(p)
        for d in rows:
            # calendarDate may be epoch-ms here
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.acute_training_load (
                    calendar_date, user_profile_pk, device_id, timestamp,
                    acwr_status, acwr_status_feedback,
                    daily_training_load_acute, daily_training_load_chronic
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (calendar_date) DO UPDATE SET
                    daily_training_load_acute=EXCLUDED.daily_training_load_acute,
                    daily_training_load_chronic=EXCLUDED.daily_training_load_chronic
            """, (
                str(cd), d.get("userProfilePK"), d.get("deviceId"),
                d.get("timestamp"),
                d.get("acwrStatus"), d.get("acwrStatusFeedback"),
                d.get("dailyTrainingLoadAcute"), d.get("dailyTrainingLoadChronic"),
            ))
            cnt += 1
    print(f"  [acute_training_load] {cnt} rows")


def ingest_hydration(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Aggregator", "HydrationLogFile_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("calendarDate"))
            if not after_cutoff(cd):
                continue
            uuid_val = d.get("uuid", {}).get("uuid") if isinstance(d.get("uuid"), dict) else None
            cur.execute("""
                INSERT INTO garmin.hydration (
                    calendar_date, user_profile_pk, timestamp_local, hydration_source,
                    value_in_ml, activity_id, estimated_sweat_loss_ml, duration, uuid
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                str(cd), d.get("userProfilePK"), d.get("timestampLocal"),
                d.get("hydrationSource"), d.get("valueInML"),
                d.get("activityId"), d.get("estimatedSweatLossInML"),
                d.get("duration"), uuid_val,
            ))
            cnt += 1
    print(f"  [hydration] {cnt} rows")


def ingest_heart_rate_zones(cur):
    for p in glob_files("DI-Connect-Wellness", "*heartRateZones.json"):
        rows = load_json(p)
        cur.execute("DELETE FROM garmin.heart_rate_zones")
        for d in rows:
            cur.execute("""
                INSERT INTO garmin.heart_rate_zones (
                    training_method, resting_hr_used, lactate_threshold_hr_used,
                    zone1_floor, zone2_floor, zone3_floor, zone4_floor, zone5_floor,
                    max_hr_used, resting_hr_auto_update, sport
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                d.get("trainingMethod"), d.get("restingHeartRateUsed"),
                d.get("lactateThresholdHeartRateUsed"),
                d.get("zone1Floor"), d.get("zone2Floor"), d.get("zone3Floor"),
                d.get("zone4Floor"), d.get("zone5Floor"),
                d.get("maxHeartRateUsed"), d.get("restingHrAutoUpdateUsed"),
                d.get("sport"),
            ))
    print(f"  [heart_rate_zones] done")


def ingest_fitness_age(cur):
    for p in glob_files("DI-Connect-Wellness", "*fitnessAgeData.json"):
        rows = load_json(p)
        cur.execute("DELETE FROM garmin.fitness_age")
        for d in (rows if isinstance(rows, list) else [rows]):
            cur.execute("""
                INSERT INTO garmin.fitness_age (raw_json) VALUES (%s)
            """, (json.dumps(d),))
    print(f"  [fitness_age] done")


def ingest_personal_records(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Fitness", "*personalRecord.json"):
        rows = load_json(p)
        for d in rows:
            cur.execute("""
                INSERT INTO garmin.personal_record (
                    type_id, activity_id, pr_start_time_gmt, pr_type_name, value, raw_json
                ) VALUES (%s,%s,%s,%s,%s,%s)
            """, (
                d.get("typeId"), d.get("activityId"),
                d.get("prStartTimeGMT"), d.get("prTypeName"),
                d.get("value"), json.dumps(d),
            ))
            cnt += 1
    print(f"  [personal_record] {cnt} rows")


def ingest_pacebands(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Routing", "*pacebands*.json"):
        raw = load_json(p)
        items = raw if isinstance(raw, list) else [raw]
        for item in items:
            # Each item has paceBandSummary and paceBandSplits
            summary = item.get("paceBandSummary", {}) if isinstance(item, dict) else {}
            cur.execute("""
                INSERT INTO garmin.paceband (
                    course_id, course_name, total_time_seconds, total_distance_meters,
                    num_segments, raw_json
                ) VALUES (%s,%s,%s,%s,%s,%s)
            """, (
                summary.get("courseId"), summary.get("courseName"),
                summary.get("totalTimeInSeconds"), summary.get("totalDistanceInMeters"),
                summary.get("numberOfSegments"), json.dumps(item),
            ))
            cnt += 1
    print(f"  [paceband] {cnt} rows")


def ingest_ecg(cur):
    cnt = 0
    for p in glob_files("DI-Connect-Health-ECG", "*.json"):
        rows = load_json(p)
        for d in rows:
            summary = d.get("summary", {})
            reading = d.get("reading", {})
            device_info = summary.get("deviceInfo", {})
            # check date filter
            cd = parse_date(summary.get("startTime"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.ecg_reading (
                    detail_id, start_time, start_time_local, rhythm_classification,
                    mounting_side, rmssd_hrv, heart_rate_average, ecg_app_version,
                    device_product_name, device_product_id, device_firmware_version,
                    sample_rate, duration_seconds, lead_type, samples, symptoms
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (detail_id) DO NOTHING
            """, (
                summary.get("detailId"), summary.get("startTime"),
                summary.get("startTimeLocal"), summary.get("rhythmClassification"),
                summary.get("mountingSide"), summary.get("rmssdHrv"),
                summary.get("heartRateAverage"), summary.get("ecgAppVersion"),
                device_info.get("productName"), device_info.get("productId"),
                device_info.get("firmwareVersion"),
                reading.get("sampleRate"), reading.get("durationInSeconds"),
                reading.get("leadType"), reading.get("samples"),
                summary.get("symptoms", []),
            ))
            cnt += 1
    print(f"  [ecg_reading] {cnt} rows")


def ingest_user_goals(cur):
    cnt = 0
    for p in glob_files("DI-Connect-User", "UserGoal_*.json"):
        rows = load_json(p)
        for d in rows:
            cd = parse_date(d.get("startDate"))
            if not after_cutoff(cd):
                continue
            cur.execute("""
                INSERT INTO garmin.user_goal (
                    goal_id, user_profile_pk, goal_type, goal_sub_type,
                    goal_value, goal_source, start_date, create_date, raw_json
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                d.get("goalId"), d.get("userProfilePK"),
                d.get("goalType"), d.get("goalSubType"),
                d.get("goalValue"), d.get("goalSource"),
                d.get("startDate"), d.get("createDate"),
                json.dumps(d),
            ))
            cnt += 1
    print(f"  [user_goal] {cnt} rows")



# ─────────────────────── FIT file ingestion ───────────────────────

def _semicircles_to_deg(val):
    """Convert Garmin semicircle coordinates to degrees."""
    if val is None:
        return None
    return val * (180 / 2**31)


def _field_val(msg, name):
    """Safely extract a field value from a FIT message."""
    field = msg.get(name)
    if field is None:
        return None
    return field.value if hasattr(field, 'value') else field


def ingest_fit_files(cur):
    """Parse FIT files and insert track points, laps, and sessions."""
    import fitparse
    from psycopg2.extras import execute_values

    fit_dirs = [
        CONNECT_DIR / "DI-Connect-Uploaded-Files" / "UploadedFiles_0-_Part1",
        CONNECT_DIR / "DI-Connect-Uploaded-Files" / "UploadedFiles_0-_Part2",
    ]

    total_records = 0
    total_laps = 0
    total_sessions = 0
    total_files = 0
    skipped = 0

    for fit_dir in fit_dirs:
        if not fit_dir.exists():
            continue
        fit_files = sorted(fit_dir.glob("*.fit"))
        print(f"  [fit] scanning {len(fit_files)} files in {fit_dir.name}...")

        for fp in fit_files:
            try:
                fitfile = fitparse.FitFile(str(fp))
                messages = list(fitfile.get_messages())
            except Exception:
                skipped += 1
                continue

            # Extract activity ID from filename: lirattar@gmail.com_<id>.fit
            fname = fp.stem
            activity_id = None
            if "_" in fname:
                try:
                    activity_id = int(fname.rsplit("_", 1)[-1])
                except ValueError:
                    pass

            # Check session date to filter
            sessions = [m for m in messages if m.name == "session"]
            if sessions:
                # Get earliest session start_time
                start_times = []
                for s in sessions:
                    for f in s.fields:
                        if f.name == "start_time" and f.value:
                            start_times.append(f.value)
                if start_times:
                    earliest = min(start_times)
                    if hasattr(earliest, 'date'):
                        session_date = earliest.date() if callable(earliest.date) else earliest.date
                    else:
                        session_date = parse_date(str(earliest))
                    if session_date and not after_cutoff(session_date):
                        skipped += 1
                        continue

            fit_name = fp.name
            total_files += 1

            # --- Sessions ---
            for s in sessions:
                fields = {f.name: f.value for f in s.fields}
                cur.execute("""
                    INSERT INTO garmin.fit_session (
                        fit_file, activity_id, sport, sub_sport,
                        start_time, timestamp,
                        total_elapsed_time, total_timer_time, total_distance, total_calories,
                        avg_heart_rate, max_heart_rate, avg_speed, max_speed,
                        avg_cadence, max_cadence, total_ascent, total_descent, avg_temperature
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    fit_name, activity_id,
                    str(fields.get("sport")) if fields.get("sport") else None,
                    str(fields.get("sub_sport")) if fields.get("sub_sport") else None,
                    fields.get("start_time"), fields.get("timestamp"),
                    fields.get("total_elapsed_time"), fields.get("total_timer_time"),
                    fields.get("total_distance"), fields.get("total_calories"),
                    fields.get("avg_heart_rate"), fields.get("max_heart_rate"),
                    fields.get("avg_speed") or fields.get("enhanced_avg_speed"),
                    fields.get("max_speed") or fields.get("enhanced_max_speed"),
                    fields.get("avg_cadence"), fields.get("max_cadence"),
                    fields.get("total_ascent"), fields.get("total_descent"),
                    fields.get("avg_temperature"),
                ))
                total_sessions += 1

            # --- Laps ---
            laps = [m for m in messages if m.name == "lap"]
            for lap in laps:
                fields = {f.name: f.value for f in lap.fields}
                cur.execute("""
                    INSERT INTO garmin.fit_lap (
                        fit_file, activity_id, start_time,
                        total_elapsed_time, total_timer_time, total_distance, total_calories,
                        avg_heart_rate, max_heart_rate, avg_speed, max_speed,
                        avg_cadence, max_cadence, total_ascent, total_descent,
                        sport, sub_sport
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    fit_name, activity_id, fields.get("start_time"),
                    fields.get("total_elapsed_time"), fields.get("total_timer_time"),
                    fields.get("total_distance"), fields.get("total_calories"),
                    fields.get("avg_heart_rate"), fields.get("max_heart_rate"),
                    fields.get("avg_speed") or fields.get("enhanced_avg_speed"),
                    fields.get("max_speed") or fields.get("enhanced_max_speed"),
                    fields.get("avg_cadence"), fields.get("max_cadence"),
                    fields.get("total_ascent"), fields.get("total_descent"),
                    str(fields.get("sport")) if fields.get("sport") else None,
                    str(fields.get("sub_sport")) if fields.get("sub_sport") else None,
                ))
                total_laps += 1

            # --- Records (GPS/HR track points) — batch insert ---
            records = [m for m in messages if m.name == "record"]
            if records:
                batch = []
                for rec in records:
                    fields = {f.name: f.value for f in rec.fields}
                    lat = _semicircles_to_deg(fields.get("position_lat"))
                    lon = _semicircles_to_deg(fields.get("position_long"))
                    batch.append((
                        fit_name, activity_id,
                        fields.get("timestamp"),
                        lat, lon,
                        fields.get("distance"),
                        fields.get("enhanced_altitude") or fields.get("altitude"),
                        fields.get("enhanced_speed") or fields.get("speed"),
                        fields.get("heart_rate"),
                        fields.get("cadence"),
                        fields.get("temperature"),
                        fields.get("fractional_cadence"),
                        fields.get("power"),
                    ))
                execute_values(cur, """
                    INSERT INTO garmin.fit_record (
                        fit_file, activity_id, timestamp,
                        position_lat, position_long, distance, altitude, speed,
                        heart_rate, cadence, temperature, fractional_cadence, power
                    ) VALUES %s
                """, batch, page_size=500)
                total_records += len(batch)

            # Commit per file to avoid huge transactions
            cur.connection.commit()

    print(f"  [fit] {total_files} files processed ({skipped} skipped/pre-cutoff)")
    print(f"  [fit] {total_sessions} sessions, {total_laps} laps, {total_records} track points")


# ─────────────────────── main ───────────────────────

ALL_INGESTORS = [
    ingest_user_profile,
    ingest_user_settings,
    ingest_social_profile,
    ingest_device_backup,
    ingest_sleep,
    ingest_daily_summary,
    ingest_activities,
    ingest_training_history,
    ingest_training_readiness,
    ingest_vo2max,
    ingest_run_race_predictions,
    ingest_cycling_ability,
    ingest_heat_altitude_acclimation,
    ingest_acute_training_load,
    ingest_hydration,
    ingest_heart_rate_zones,
    ingest_fitness_age,
    ingest_personal_records,
    ingest_pacebands,
    ingest_ecg,
    ingest_user_goals,
    ingest_fit_files,
]


def main():
    if "--init" in sys.argv:
        init_schema()

    conn = get_connection()
    cur = conn.cursor()

    print(f"\n{'='*50}")
    print(f"Garmin Ingestion Pipeline")
    print(f"Data dir : {CONNECT_DIR}")
    print(f"Date from: {DATE_FROM}")
    print(f"{'='*50}\n")

    for fn in ALL_INGESTORS:
        try:
            fn(cur)
            conn.commit()
        except Exception as exc:
            conn.rollback()
            print(f"  [ERROR] {fn.__name__}: {exc}")

    cur.close()
    conn.close()
    print(f"\n{'='*50}")
    print("Done.")


if __name__ == "__main__":
    main()
