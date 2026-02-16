-- Recreate garmin schema if it doesn't exist (safe for production DB)
CREATE SCHEMA IF NOT EXISTS garmin;

-- 
-- User / Device / Social
--
CREATE TABLE IF NOT EXISTS garmin.user_profile (
    user_name TEXT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    email TEXT,
    gender TEXT,
    birth_date DATE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS garmin.user_settings (
    preferred_locale TEXT,
    handedness TEXT,
    use_fixed_location BOOLEAN
);

CREATE TABLE IF NOT EXISTS garmin.social_profile (
    display_name TEXT PRIMARY KEY,
    profile_image_url_large TEXT,
    profile_image_url_medium TEXT,
    profile_image_url_small TEXT,
    user_profile_number BIGINT,
    garmin_guid TEXT,
    level INTEGER,
    level_update_date TIMESTAMP,
    level_is_upper_cap BOOLEAN,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.device_backup (
    device_backup_id BIGINT PRIMARY KEY,
    user_id BIGINT,
    device_id BIGINT,
    primary_uuid TEXT,
    create_date TIMESTAMP,
    pending_restore BOOLEAN
);

--
-- Wellness / health
--
CREATE TABLE IF NOT EXISTS garmin.sleep (
    calendar_date DATE PRIMARY KEY,
    sleep_start_ts_gmt TIMESTAMP,
    sleep_end_ts_gmt TIMESTAMP,
    sleep_start_ts_local TIMESTAMP,
    sleep_end_ts_local TIMESTAMP,
    unmeasurable_sleep_seconds INTEGER,
    deep_sleep_seconds INTEGER,
    light_sleep_seconds INTEGER,
    rem_sleep_seconds INTEGER,
    awake_sleep_seconds INTEGER,
    average_spo2_value NUMERIC,
    lowest_spo2_value NUMERIC,
    highest_spo2_value NUMERIC,
    average_spo2_hr_sleep NUMERIC,
    average_respiration_value NUMERIC,
    lowest_respiration_value NUMERIC,
    highest_respiration_value NUMERIC,
    avg_sleep_stress NUMERIC,
    sleep_score_feedback TEXT,
    overall_score_value INTEGER,
    overall_score_qualifier TEXT,
    rem_percentage_value NUMERIC,
    rem_optimal_start NUMERIC,
    rem_optimal_end NUMERIC,
    deep_percentage_value NUMERIC,
    deep_optimal_start NUMERIC,
    deep_optimal_end NUMERIC,
    light_percentage_value NUMERIC,
    light_optimal_start NUMERIC,
    light_optimal_end NUMERIC,
    duration_in_ms BIGINT,
    awakenings_count_value INTEGER,
    average_stress_during_sleep NUMERIC,
    age_group TEXT,
    sleep_result_type TEXT,
    resting_heart_rate INTEGER,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.daily_summary (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    total_kilocalories NUMERIC,
    active_kilocalories NUMERIC,
    bmr_kilocalories NUMERIC,
    total_steps INTEGER,
    total_distance_meters NUMERIC,
    highly_active_seconds INTEGER,
    active_seconds INTEGER,
    sedentary_seconds INTEGER,
    sleeping_seconds INTEGER,
    moderate_intensity_minutes INTEGER,
    vigorous_intensity_minutes INTEGER,
    floors_ascended NUMERIC,
    floors_descended NUMERIC,
    min_heart_rate INTEGER,
    max_heart_rate INTEGER,
    resting_heart_rate INTEGER,
    avg_stress_level NUMERIC,
    max_stress_level NUMERIC,
    stress_duration INTEGER,
    rest_stress_duration INTEGER,
    activity_stress_duration INTEGER,
    low_stress_duration INTEGER,
    medium_stress_duration INTEGER,
    high_stress_duration INTEGER,
    body_battery_charged_value INTEGER,
    body_battery_drained_value INTEGER,
    body_battery_highest_value INTEGER,
    body_battery_lowest_value INTEGER,
    body_battery_most_recent INTEGER,
    average_spo2 NUMERIC,
    lowest_spo2 NUMERIC,
    latest_spo2 NUMERIC,
    avg_waking_respiration_value NUMERIC,
    highest_respiration_value NUMERIC,
    lowest_respiration_value NUMERIC,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.hydration (
    calendar_date DATE,
    user_profile_pk BIGINT,
    timestamp_local TIMESTAMP,
    hydration_source TEXT,
    value_in_ml INTEGER,
    activity_id BIGINT,
    estimated_sweat_loss_ml INTEGER,
    duration INTEGER,
    uuid TEXT
);

CREATE TABLE IF NOT EXISTS garmin.heart_rate_zones (
    training_method TEXT,
    resting_hr_used INTEGER,
    lactate_threshold_hr_used INTEGER,
    zone1_floor INTEGER,
    zone2_floor INTEGER,
    zone3_floor INTEGER,
    zone4_floor INTEGER,
    zone5_floor INTEGER,
    max_hr_used INTEGER,
    resting_hr_auto_update BOOLEAN,
    sport TEXT
);

CREATE TABLE IF NOT EXISTS garmin.fitness_age (
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.ecg_reading (
    detail_id TEXT PRIMARY KEY,
    start_time TIMESTAMP,
    start_time_local TIMESTAMP,
    rhythm_classification TEXT,
    mounting_side TEXT,
    rmssd_hrv NUMERIC,
    heart_rate_average INTEGER,
    ecg_app_version TEXT,
    device_product_name TEXT,
    device_product_id BIGINT,
    device_firmware_version TEXT,
    sample_rate NUMERIC,
    duration_seconds NUMERIC,
    lead_type TEXT,
    samples TEXT,
    symptoms TEXT[]
);

CREATE TABLE IF NOT EXISTS garmin.user_goal (
    goal_id BIGINT,
    user_profile_pk BIGINT,
    goal_type TEXT,
    goal_sub_type TEXT,
    goal_value NUMERIC,
    goal_source TEXT,
    start_date DATE,
    create_date TIMESTAMP,
    raw_json JSONB
);

--
-- Fitness / Metrics
--
CREATE TABLE IF NOT EXISTS garmin.activity (
    activity_id BIGINT PRIMARY KEY,
    activity_name TEXT,
    start_time_local TIMESTAMP,
    start_time_gmt TIMESTAMP,
    activity_type TEXT,
    sport_type TEXT,
    event_type TEXT,
    distance NUMERIC,
    duration NUMERIC,
    elapsed_duration NUMERIC,
    moving_duration NUMERIC,
    elevation_gain NUMERIC,
    elevation_loss NUMERIC,
    average_speed NUMERIC,
    max_speed NUMERIC,
    average_hr INTEGER,
    max_hr INTEGER,
    average_running_cadence NUMERIC,
    max_running_cadence NUMERIC,
    calories NUMERIC,
    bmr_calories NUMERIC,
    steps INTEGER,
    average_stride_length NUMERIC,
    training_effect_label TEXT,
    aerobic_training_effect NUMERIC,
    anaerobic_training_effect NUMERIC,
    training_effect_label_anaerobic TEXT,
    vo2_max_value NUMERIC,
    lactate_threshold NUMERIC,
    min_temperature NUMERIC,
    max_temperature NUMERIC,
    start_latitude NUMERIC,
    start_longitude NUMERIC,
    end_latitude NUMERIC,
    end_longitude NUMERIC,
    location_name TEXT,
    moderate_intensity_minutes INTEGER,
    vigorous_intensity_minutes INTEGER,
    pr_flag BOOLEAN,
    manual_activity BOOLEAN,
    auto_calc_calories BOOLEAN,
    average_respiration_rate NUMERIC,
    training_load NUMERIC,
    water_estimated NUMERIC,
    begin_timestamp BIGINT,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.training_history (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    device_id BIGINT,
    update_timestamp TIMESTAMP,
    training_load_balance NUMERIC,
    weekly_acute_chronic_workload_ratio NUMERIC,
    weekly_aerobic_excess_post NUMERIC,
    weekly_anaerobic_excess_post NUMERIC,
    daily_training_load_acute NUMERIC,
    daily_training_load_chronic NUMERIC,
    training_load_balance_feedback TEXT,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.training_readiness (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    device_id BIGINT,
    timestamp TIMESTAMP,
    training_readiness_level TEXT,
    overall_score INTEGER,
    recovery_score INTEGER,
    training_load_score INTEGER,
    sleep_score INTEGER,
    hrv_score INTEGER,
    stress_history_score INTEGER,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.vo2max (
    calendar_date DATE,
    user_profile_pk BIGINT,
    device_id BIGINT,
    update_timestamp TIMESTAMP,
    sport TEXT,
    sub_sport TEXT,
    vo2_max_value NUMERIC,
    max_met NUMERIC,
    max_met_category TEXT,
    calibrated_data BOOLEAN,
    PRIMARY KEY (calendar_date, sport)
);

CREATE TABLE IF NOT EXISTS garmin.run_race_predictions (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    device_id BIGINT,
    timestamp TIMESTAMP,
    race_time_5k NUMERIC,
    race_time_10k NUMERIC,
    race_time_half NUMERIC,
    race_time_marathon NUMERIC
);

CREATE TABLE IF NOT EXISTS garmin.cycling_ability (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    device_id BIGINT,
    timestamp TIMESTAMP,
    aerobic_endurance NUMERIC,
    aerobic_capacity NUMERIC,
    anaerobic_capacity NUMERIC,
    profile_type TEXT,
    profile_type_feedback TEXT,
    aerobic_endurance_feedback TEXT,
    aerobic_capacity_feedback TEXT,
    anaerobic_capacity_feedback TEXT
);

CREATE TABLE IF NOT EXISTS garmin.heat_altitude_acclimation (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    device_id BIGINT,
    timestamp TIMESTAMP,
    altitude_acclimation NUMERIC,
    previous_altitude_acclimation NUMERIC,
    heat_acclimation_percentage NUMERIC,
    previous_heat_acclimation_pct NUMERIC,
    current_altitude NUMERIC,
    prev_altitude NUMERIC,
    acclimation_percentage NUMERIC,
    prev_acclimation_percentage NUMERIC
);

CREATE TABLE IF NOT EXISTS garmin.acute_training_load (
    calendar_date DATE PRIMARY KEY,
    user_profile_pk BIGINT,
    device_id BIGINT,
    timestamp TIMESTAMP,
    acwr_status TEXT,
    acwr_status_feedback TEXT,
    daily_training_load_acute NUMERIC,
    daily_training_load_chronic NUMERIC
);

CREATE TABLE IF NOT EXISTS garmin.personal_record (
    type_id INTEGER,
    activity_id BIGINT,
    pr_start_time_gmt TIMESTAMP,
    pr_type_name TEXT,
    value NUMERIC,
    raw_json JSONB
);

CREATE TABLE IF NOT EXISTS garmin.paceband (
    course_id BIGINT,
    course_name TEXT,
    total_time_seconds NUMERIC,
    total_distance_meters NUMERIC,
    num_segments INTEGER,
    raw_json JSONB
);

--
-- FIT Data
--
CREATE TABLE IF NOT EXISTS garmin.fit_session (
    fit_file TEXT,
    activity_id BIGINT,
    sport TEXT,
    sub_sport TEXT,
    start_time TIMESTAMP,
    timestamp TIMESTAMP,
    total_elapsed_time NUMERIC,
    total_timer_time NUMERIC,
    total_distance NUMERIC,
    total_calories NUMERIC,
    avg_heart_rate INTEGER,
    max_heart_rate INTEGER,
    avg_speed NUMERIC,
    max_speed NUMERIC,
    avg_cadence INTEGER,
    max_cadence INTEGER,
    total_ascent NUMERIC,
    total_descent NUMERIC,
    avg_temperature NUMERIC
);

CREATE TABLE IF NOT EXISTS garmin.fit_lap (
    fit_file TEXT,
    activity_id BIGINT,
    start_time TIMESTAMP,
    total_elapsed_time NUMERIC,
    total_timer_time NUMERIC,
    total_distance NUMERIC,
    total_calories NUMERIC,
    avg_heart_rate INTEGER,
    max_heart_rate INTEGER,
    avg_speed NUMERIC,
    max_speed NUMERIC,
    avg_cadence INTEGER,
    max_cadence INTEGER,
    total_ascent NUMERIC,
    total_descent NUMERIC,
    sport TEXT,
    sub_sport TEXT
);

CREATE TABLE IF NOT EXISTS garmin.fit_record (
    fit_file TEXT,
    activity_id BIGINT,
    timestamp TIMESTAMP,
    position_lat NUMERIC,
    position_long NUMERIC,
    distance NUMERIC,
    altitude NUMERIC,
    speed NUMERIC,
    heart_rate INTEGER,
    cadence INTEGER,
    temperature INTEGER,
    fractional_cadence NUMERIC,
    power INTEGER
);
