"""
Enhanced Database Schema
=========================
Adds supplementary tables on top of the core schema (daily_metrics,
activities) which lives in garmin_health_tracker.py.

Tables added here:
  - wellness_log          (daily questionnaire data)
  - nutrition_log         (meal tracking)
  - personal_records      (PRs)
  - goals                 (custom goals)
  - weekly_summaries      (cached AI insights)

Views:
  - weekly_comparison     (week-over-week analysis)
  - metric_correlations   (lag analysis convenience view)

NOTE: correlation engine tables live in correlation_engine.py.
"""
import logging
logger = logging.getLogger("enhanced_schema")

ENHANCED_SCHEMA_SQL = """
-- Add wellness_log table for daily questionnaire data
CREATE TABLE IF NOT EXISTS wellness_log (
    log_id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    
    -- Substances
    caffeine_intake INTEGER,  -- mg or servings
    alcohol_intake INTEGER,   -- drinks
    
    -- Health Status
    illness_severity INTEGER,  -- 0-5 scale
    injury_severity INTEGER,   -- 0-5 scale
    
    -- Subjective Feelings
    overall_stress_level INTEGER,  -- 0-100
    energy_level INTEGER,          -- 0-100
    workout_feeling INTEGER,       -- 0-100 (how workout felt)
    muscle_soreness INTEGER,       -- 0-5 scale
    
    -- Menstrual Cycle (if applicable)
    cycle_phase TEXT,  -- menstruation, follicular, ovulation, luteal
    cycle_day INTEGER,
    
    -- Notes
    notes TEXT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for faster date queries
CREATE INDEX IF NOT EXISTS idx_wellness_log_date ON wellness_log(date DESC);

-- Add nutrition_log table for meal tracking
CREATE TABLE IF NOT EXISTS nutrition_log (
    log_id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    
    -- Macros
    calories INTEGER,
    protein_grams NUMERIC(6,1),
    carbs_grams NUMERIC(6,1),
    fat_grams NUMERIC(6,1),
    fiber_grams NUMERIC(5,1),
    
    -- Hydration
    water_ml INTEGER,
    
    -- Meal timing
    meal_type TEXT,  -- breakfast, lunch, dinner, snack
    meal_time TIME,
    
    -- Notes
    meal_description TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_nutrition_date ON nutrition_log(date DESC);

-- Add personal_records table for tracking PRs
CREATE TABLE IF NOT EXISTS personal_records (
    record_id SERIAL PRIMARY KEY,
    activity_type TEXT NOT NULL,
    record_type TEXT NOT NULL,  -- fastest_5k, longest_run, etc.
    
    value NUMERIC(10,2),
    unit TEXT,  -- seconds, meters, etc.
    
    date_achieved DATE,
    activity_id BIGINT,  -- Reference to activities table
    
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(activity_type, record_type)
);

-- Add goals table for custom goal tracking
CREATE TABLE IF NOT EXISTS goals (
    goal_id SERIAL PRIMARY KEY,
    goal_name TEXT NOT NULL,
    goal_type TEXT,  -- fitness, weight, sleep, etc.
    
    -- Target
    target_value NUMERIC(10,2),
    target_unit TEXT,
    target_date DATE,
    
    -- Current progress
    current_value NUMERIC(10,2),
    progress_pct NUMERIC(5,2),
    
    -- Status
    status TEXT,  -- active, achieved, abandoned
    
    -- Dates
    start_date DATE,
    end_date DATE,
    
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add weekly_summaries table for caching AI analysis
CREATE TABLE IF NOT EXISTS weekly_summaries (
    summary_id SERIAL PRIMARY KEY,
    week_start_date DATE NOT NULL UNIQUE,
    
    -- Key metrics averages
    avg_resting_hr NUMERIC(5,1),
    avg_hrv NUMERIC(5,1),
    avg_sleep_score NUMERIC(4,1),
    avg_stress NUMERIC(4,1),
    avg_readiness NUMERIC(4,1),
    
    total_training_load INTEGER,
    total_steps INTEGER,
    total_activities INTEGER,
    
    -- AI-generated insights
    key_insights TEXT,
    recommendations TEXT,
    
    -- Comparison to previous week
    vs_previous_week JSONB,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add comparison view for easy week-over-week analysis
CREATE OR REPLACE VIEW weekly_comparison AS
WITH weekly_stats AS (
    SELECT 
        DATE_TRUNC('week', date)::DATE as week_start,
        COUNT(*) as days_logged,
        AVG(resting_hr) as avg_rhr,
        AVG(hrv_last_night) as avg_hrv,
        AVG(sleep_score) as avg_sleep,
        AVG(sleep_seconds / 3600.0) as avg_sleep_hours,
        AVG(stress_level) as avg_stress,
        SUM(bb_drained) as total_load,
        AVG(training_readiness) as avg_readiness,
        SUM(total_steps) as total_steps,
        AVG(bb_peak) as avg_body_battery
    FROM daily_metrics
    WHERE date >= CURRENT_DATE - INTERVAL '12 weeks'
    GROUP BY week_start
)
SELECT 
    ws1.week_start,
    ws1.days_logged,
    ws1.avg_rhr,
    ws1.avg_hrv,
    ws1.avg_sleep,
    ws1.avg_stress,
    ws1.total_load,
    ws1.avg_readiness,
    -- Comparison to previous week
    ws1.avg_rhr - ws2.avg_rhr as rhr_change,
    ws1.avg_hrv - ws2.avg_hrv as hrv_change,
    ws1.avg_sleep - ws2.avg_sleep as sleep_change,
    ws1.total_load - ws2.total_load as load_change
FROM weekly_stats ws1
LEFT JOIN weekly_stats ws2 ON ws2.week_start = ws1.week_start - INTERVAL '7 days'
ORDER BY ws1.week_start DESC;

-- Add correlation analysis view
CREATE OR REPLACE VIEW metric_correlations AS
SELECT 
    date,
    sleep_score,
    LAG(sleep_score, 1) OVER (ORDER BY date) as prev_day_sleep,
    training_readiness,
    LAG(training_readiness, 1) OVER (ORDER BY date) as prev_day_readiness,
    hrv_last_night,
    stress_level,
    bb_peak,
    bb_drained,
    resting_hr
FROM daily_metrics
WHERE date >= CURRENT_DATE - INTERVAL '90 days'
ORDER BY date;
"""

def upgrade_database(conn_str: str = None):
    """
    Apply enhanced schema to existing database.
    Also ensures correlation engine tables exist.
    Safe to run multiple times (uses IF NOT EXISTS).

    Parameters
    ----------
    conn_str : str, optional
        PostgreSQL connection string.  Falls back to
        POSTGRES_CONNECTION_STRING env var.
    """
    import os
    import psycopg2
    from dotenv import load_dotenv
    load_dotenv()

    conn_str = conn_str or os.getenv("POSTGRES_CONNECTION_STRING", "")

    try:
        conn = psycopg2.connect(conn_str)
        conn.autocommit = True
        cur = conn.cursor()

        # Enhanced tables + views
        for statement in ENHANCED_SCHEMA_SQL.split(';'):
            stmt = statement.strip()
            if stmt:
                cur.execute(stmt)

        cur.close()
        conn.close()

        # Correlation engine tables
        from correlation_engine import CorrelationEngine
        CorrelationEngine(conn_str).bootstrap_schema()

        logger.info("Database schema upgraded successfully!")
        logger.info("   Enhanced tables: wellness_log, nutrition_log,")
        logger.info("                    personal_records, goals,")
        logger.info("                    weekly_summaries")
        logger.info("   Correlation:     matrix_summaries")
        logger.info("   Views:           weekly_comparison,")
        logger.info("                    metric_correlations")

    except Exception as e:
        logger.info(f"Schema upgrade failed: {e}")
        raise


if __name__ == "__main__":
    upgrade_database()
