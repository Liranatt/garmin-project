"""
Garmin Bulk-Download → Production Merge
========================================
Reads data from the garmin DB (bulk JSON/FIT export) and merges
enriching columns into the production postgres DB.

What gets merged:
  daily_metrics  ← VO2 max, race predictions, acute training load,
                    heat/altitude acclimation (+ delta columns)
  activities     ← training effects, training load, VO2 max per activity,
                    stride length

Run:
    python garmin_merge.py                   # full merge
    python garmin_merge.py --validate-only   # cross-validate, no writes
    python garmin_merge.py --since 2026-02-10
"""

import os, sys, argparse, logging
from datetime import date, timedelta
from collections import defaultdict

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("garmin_merge")

GARMIN_CS = os.getenv("GARMIN_CONNECTION_STRING", "")
PROD_CS   = os.getenv("POSTGRES_CONNECTION_STRING", "")

if not GARMIN_CS:
    sys.exit("❌ GARMIN_CONNECTION_STRING not set in .env")
if not PROD_CS:
    sys.exit("❌ POSTGRES_CONNECTION_STRING not set in .env")


# ═══════════════════════════════════════════════════════════════
#  SCHEMA MIGRATION  — add new columns if they don't exist
# ═══════════════════════════════════════════════════════════════

MIGRATION_SQL = """
-- ── daily_metrics: new columns from garmin bulk data ──
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS vo2_max_running        REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS vo2_max_cycling        REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS vo2_max_running_delta  REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS race_time_5k           REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS race_time_10k          REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS race_time_half         REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS race_time_5k_delta     REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS daily_load_acute       REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS daily_load_chronic     REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS acwr                   REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS heat_acclimation_pct   REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS altitude_acclimation   REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS cycling_aerobic_endurance  REAL;
ALTER TABLE daily_metrics ADD COLUMN IF NOT EXISTS cycling_anaerobic_capacity REAL;

-- ── activities: enrichment columns ──
ALTER TABLE activities ADD COLUMN IF NOT EXISTS date                      DATE;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS vo2_max_value             REAL;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS aerobic_training_effect   REAL;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS anaerobic_training_effect REAL;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS training_effect_label     VARCHAR(50);
ALTER TABLE activities ADD COLUMN IF NOT EXISTS training_load             REAL;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS avg_stride_length         REAL;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS avg_respiration_rate      REAL;
ALTER TABLE activities ADD COLUMN IF NOT EXISTS sport_type                VARCHAR(50);
"""


def run_migration(prod_conn):
    """Add new columns to daily_metrics and activities tables."""
    log.info("Running schema migration…")
    # Each ALTER TABLE needs its own transaction
    for stmt in MIGRATION_SQL.strip().split(";"):
        stmt = stmt.strip()
        if stmt and not stmt.startswith("--"):
            cur = prod_conn.cursor()
            try:
                cur.execute(stmt)
                prod_conn.commit()
            except Exception as e:
                prod_conn.rollback()
                # "already exists" is expected — suppress it
                if "already exists" not in str(e):
                    log.info(f"    ⚠️  Migration: {e}")
            cur.close()
    log.info("  ✅ Schema migration complete")


# ═══════════════════════════════════════════════════════════════
#  DATA EXTRACTION FROM GARMIN DB
# ═══════════════════════════════════════════════════════════════

def detect_schema(garmin_cur):
    """Detect whether garmin DB uses 'garmin' or 'public' schema."""
    garmin_cur.execute("""
        SELECT COUNT(*) FROM information_schema.tables
        WHERE table_schema = 'garmin' AND table_type = 'BASE TABLE'
    """)
    if garmin_cur.fetchone()[0] > 0:
        return "garmin"
    return "public"


def extract_vo2max(garmin_cur, schema, since):
    """Extract VO2 Max values, pivoted by sport → running/cycling."""
    garmin_cur.execute(f"""
        SELECT calendar_date, sport, vo2_max_value
        FROM {schema}.vo2max
        WHERE calendar_date >= %s AND vo2_max_value IS NOT NULL
        ORDER BY calendar_date
    """, (since,))

    by_date = defaultdict(dict)
    for d, sport, val in garmin_cur.fetchall():
        sport_key = str(sport).upper() if sport else "UNKNOWN"
        if "RUN" in sport_key:
            by_date[d]["vo2_max_running"] = float(val)
        elif "CYCL" in sport_key:
            by_date[d]["vo2_max_cycling"] = float(val)
        else:
            # Default to running if unknown sport
            by_date[d].setdefault("vo2_max_running", float(val))

    return dict(by_date)


def extract_race_predictions(garmin_cur, schema, since):
    """Extract race time predictions."""
    garmin_cur.execute(f"""
        SELECT calendar_date, race_time_5k, race_time_10k, race_time_half
        FROM {schema}.run_race_predictions
        WHERE calendar_date >= %s
        ORDER BY calendar_date
    """, (since,))

    by_date = {}
    for d, t5k, t10k, thalf in garmin_cur.fetchall():
        by_date[d] = {
            "race_time_5k":   float(t5k) if t5k else None,
            "race_time_10k":  float(t10k) if t10k else None,
            "race_time_half": float(thalf) if thalf else None,
        }
    return by_date


def extract_acute_load(garmin_cur, schema, since):
    """Extract acute/chronic training load + ACWR."""
    garmin_cur.execute(f"""
        SELECT calendar_date,
               daily_training_load_acute,
               daily_training_load_chronic,
               acwr_status
        FROM {schema}.acute_training_load
        WHERE calendar_date >= %s
        ORDER BY calendar_date
    """, (since,))

    by_date = {}
    for d, acute, chronic, acwr_status in garmin_cur.fetchall():
        acwr_val = None
        if acute and chronic and chronic > 0:
            acwr_val = round(float(acute) / float(chronic), 3)
        by_date[d] = {
            "daily_load_acute":  float(acute) if acute else None,
            "daily_load_chronic": float(chronic) if chronic else None,
            "acwr":              acwr_val,
        }
    return by_date


def extract_heat_altitude(garmin_cur, schema, since):
    """Extract heat/altitude acclimation."""
    garmin_cur.execute(f"""
        SELECT calendar_date,
               heat_acclimation_percentage,
               altitude_acclimation
        FROM {schema}.heat_altitude_acclimation
        WHERE calendar_date >= %s
        ORDER BY calendar_date
    """, (since,))

    by_date = {}
    for d, heat, alt in garmin_cur.fetchall():
        by_date[d] = {
            "heat_acclimation_pct": float(heat) if heat else None,
            "altitude_acclimation": float(alt) if alt else None,
        }
    return by_date


def extract_cycling_ability(garmin_cur, schema, since):
    """Extract cycling ability metrics."""
    garmin_cur.execute(f"""
        SELECT calendar_date, aerobic_endurance, anaerobic_capacity
        FROM {schema}.cycling_ability
        WHERE calendar_date >= %s
        ORDER BY calendar_date
    """, (since,))

    by_date = {}
    for d, aero, anaero in garmin_cur.fetchall():
        by_date[d] = {
            "cycling_aerobic_endurance":  float(aero) if aero else None,
            "cycling_anaerobic_capacity": float(anaero) if anaero else None,
        }
    return by_date


def extract_activities(garmin_cur, schema, since):
    """Extract enriched activity fields."""
    garmin_cur.execute(f"""
        SELECT activity_id,
               vo2_max_value,
               aerobic_training_effect,
               anaerobic_training_effect,
               training_effect_label,
               training_load,
               average_stride_length,
               average_respiration_rate,
               sport_type
        FROM {schema}.activity
        WHERE activity_id IS NOT NULL
        ORDER BY activity_id
    """, )  # No date filter — activity_id is the key

    activities = {}
    for row in garmin_cur.fetchall():
        aid = row[0]
        activities[aid] = {
            "vo2_max_value":             float(row[1]) if row[1] else None,
            "aerobic_training_effect":   float(row[2]) if row[2] else None,
            "anaerobic_training_effect": float(row[3]) if row[3] else None,
            "training_effect_label":     row[4],
            "training_load":             float(row[5]) if row[5] else None,
            "avg_stride_length":         float(row[6]) if row[6] else None,
            "avg_respiration_rate":      float(row[7]) if row[7] else None,
            "sport_type":               row[8],
        }
    return activities


# ═══════════════════════════════════════════════════════════════
#  DELTA COMPUTATION (7-day rolling change)
# ═══════════════════════════════════════════════════════════════

def compute_deltas(daily_data, all_dates):
    """
    For slow-moving metrics (VO2 max, race times), compute the
    change over 7 days.  This is what correlates with daily biometrics,
    not the raw (near-constant) value.

    Adds _delta keys to daily_data in-place.
    """
    sorted_dates = sorted(all_dates)

    # Build lookup: date → metric value
    vo2_by_date = {}
    r5k_by_date = {}
    for d in sorted_dates:
        if d in daily_data:
            v = daily_data[d].get("vo2_max_running")
            if v:
                vo2_by_date[d] = v
            r = daily_data[d].get("race_time_5k")
            if r:
                r5k_by_date[d] = r

    # Forward-fill: carry last known value forward
    last_vo2 = None
    last_r5k = None
    vo2_filled = {}
    r5k_filled = {}
    for d in sorted_dates:
        if d in vo2_by_date:
            last_vo2 = vo2_by_date[d]
        if last_vo2 is not None:
            vo2_filled[d] = last_vo2

        if d in r5k_by_date:
            last_r5k = r5k_by_date[d]
        if last_r5k is not None:
            r5k_filled[d] = last_r5k

    # Compute 7-day delta
    for d in sorted_dates:
        d7 = d - timedelta(days=7)
        if d in daily_data:
            if d in vo2_filled and d7 in vo2_filled:
                daily_data[d]["vo2_max_running_delta"] = round(
                    vo2_filled[d] - vo2_filled[d7], 2)
            if d in r5k_filled and d7 in r5k_filled:
                daily_data[d]["race_time_5k_delta"] = round(
                    r5k_filled[d] - r5k_filled[d7], 2)


# ═══════════════════════════════════════════════════════════════
#  UPSERT TO PRODUCTION DB
# ═══════════════════════════════════════════════════════════════

# New columns we write to daily_metrics
NEW_DAILY_COLS = [
    "vo2_max_running", "vo2_max_cycling", "vo2_max_running_delta",
    "race_time_5k", "race_time_10k", "race_time_half", "race_time_5k_delta",
    "daily_load_acute", "daily_load_chronic", "acwr",
    "heat_acclimation_pct", "altitude_acclimation",
    "cycling_aerobic_endurance", "cycling_anaerobic_capacity",
]

# New columns we write to activities
NEW_ACTIVITY_COLS = [
    "vo2_max_value", "aerobic_training_effect", "anaerobic_training_effect",
    "training_effect_label", "training_load", "avg_stride_length",
    "avg_respiration_rate", "sport_type",
]


def upsert_daily_metrics(prod_cur, daily_data, dry_run=False):
    """Upsert new columns into daily_metrics. Uses COALESCE to not overwrite."""
    if not daily_data:
        log.info("  No daily data to merge")
        return 0

    # Build UPSERT SQL
    coalesce_parts = ", ".join(
        f"{c} = COALESCE(EXCLUDED.{c}, daily_metrics.{c})"
        for c in NEW_DAILY_COLS
    )
    sql = f"""
        INSERT INTO daily_metrics (date, {', '.join(NEW_DAILY_COLS)})
        VALUES (%s, {', '.join(['%s'] * len(NEW_DAILY_COLS))})
        ON CONFLICT (date) DO UPDATE SET {coalesce_parts}
    """

    count = 0
    for d in sorted(daily_data.keys()):
        row = daily_data[d]
        values = [d] + [row.get(c) for c in NEW_DAILY_COLS]

        # Skip if all values are None
        if all(v is None for v in values[1:]):
            continue

        if dry_run:
            non_null = {c: row[c] for c in NEW_DAILY_COLS if row.get(c) is not None}
            log.info(f"    [DRY] {d}: would set {non_null}")
        else:
            try:
                prod_cur.execute(sql, values)
                count += 1
            except Exception as e:
                prod_cur.connection.rollback()
                log.info(f"    ⚠️  {d}: {e}")

    if not dry_run:
        prod_cur.connection.commit()
    return count


def upsert_activities(prod_cur, activities, dry_run=False):
    """Enrich existing activities with training effects, VO2 max, etc."""
    if not activities:
        log.info("  No activity enrichment data")
        return 0

    # Use COALESCE so we don't overwrite API-fetched values
    coalesce_parts = ", ".join(
        f"{c} = COALESCE(EXCLUDED.{c}, activities.{c})"
        for c in NEW_ACTIVITY_COLS
    )
    sql = f"""
        INSERT INTO activities (activity_id, {', '.join(NEW_ACTIVITY_COLS)})
        VALUES (%s, {', '.join(['%s'] * len(NEW_ACTIVITY_COLS))})
        ON CONFLICT (activity_id) DO UPDATE SET {coalesce_parts}
    """

    count = 0
    for aid, data in activities.items():
        values = [aid] + [data.get(c) for c in NEW_ACTIVITY_COLS]

        # Skip if all enrichment values are None
        if all(v is None for v in values[1:]):
            continue

        if dry_run:
            non_null = {c: data[c] for c in NEW_ACTIVITY_COLS if data.get(c) is not None}
            log.info(f"    [DRY] activity {aid}: would set {non_null}")
        else:
            try:
                prod_cur.execute(sql, values)
                count += 1
            except Exception as e:
                prod_cur.connection.rollback()
                log.info(f"    ⚠️  activity {aid}: {e}")

    if not dry_run:
        prod_cur.connection.commit()
    return count


# ═══════════════════════════════════════════════════════════════
#  VALIDATION REPORT
# ═══════════════════════════════════════════════════════════════

def validate_merge(prod_cur, daily_data, activities):
    """Post-merge validation: check data actually landed."""
    log.info("\n  Post-merge validation:")

    if daily_data:
        dates = sorted(daily_data.keys())
        prod_cur.execute("""
            SELECT date, vo2_max_running, daily_load_acute, acwr,
                   race_time_5k, heat_acclimation_pct
            FROM daily_metrics
            WHERE date >= %s AND date <= %s
            ORDER BY date
        """, (dates[0], dates[-1]))

        rows = prod_cur.fetchall()
        filled = 0
        for r in rows:
            non_null = sum(1 for v in r[1:] if v is not None)
            if non_null > 0:
                filled += 1

        log.info(f"    daily_metrics: {filled}/{len(rows)} rows have "
              f"new garmin columns populated")

    if activities:
        prod_cur.execute("""
            SELECT COUNT(*) FILTER (WHERE aerobic_training_effect IS NOT NULL),
                   COUNT(*) FILTER (WHERE training_load IS NOT NULL),
                   COUNT(*) FILTER (WHERE vo2_max_value IS NOT NULL),
                   COUNT(*)
            FROM activities
        """)
        aero, tl, vo2, total = prod_cur.fetchone()
        log.info(f"    activities: {total} total — "
              f"aero_effect={aero}, training_load={tl}, vo2_max={vo2}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Merge garmin DB → production")
    parser.add_argument("--validate-only", action="store_true",
                        help="Cross-validate only, no writes")
    parser.add_argument("--since", type=str, default=None,
                        help="Only process data from this date onward (default: 30 days ago)")
    args = parser.parse_args()

    since = date.fromisoformat(args.since) if args.since else (date.today() - timedelta(days=30))
    dry_run = args.validate_only

    log.info(f"\n{'═' * 60}")
    log.info(f"  Garmin Bulk-Download → Production Merge")
    log.info(f"  Since: {since}")
    log.info(f"  Mode:  {'VALIDATE ONLY (no writes)' if dry_run else 'FULL MERGE'}")
    log.info(f"{'═' * 60}\n")

    # Connect to both databases
    garmin_conn = psycopg2.connect(GARMIN_CS)
    garmin_conn.autocommit = True
    garmin_cur = garmin_conn.cursor()

    prod_conn = psycopg2.connect(PROD_CS)
    prod_cur = prod_conn.cursor()

    schema = detect_schema(garmin_cur)
    log.info(f"  Garmin DB schema: {schema}")

    # ── Step 1: Migrate schema ──
    if not dry_run:
        run_migration(prod_conn)

    # ── Step 2: Extract from garmin DB ──
    log.info("\nExtracting from garmin DB…")

    vo2_data      = extract_vo2max(garmin_cur, schema, since)
    race_data     = extract_race_predictions(garmin_cur, schema, since)
    load_data     = extract_acute_load(garmin_cur, schema, since)
    heat_data     = extract_heat_altitude(garmin_cur, schema, since)
    cycling_data  = extract_cycling_ability(garmin_cur, schema, since)
    activity_data = extract_activities(garmin_cur, schema, since)

    log.info(f"  VO2 Max:          {len(vo2_data)} dates")
    log.info(f"  Race predictions: {len(race_data)} dates")
    log.info(f"  Acute load:       {len(load_data)} dates")
    log.info(f"  Heat/altitude:    {len(heat_data)} dates")
    log.info(f"  Cycling ability:  {len(cycling_data)} dates")
    log.info(f"  Activities:       {len(activity_data)} activities")

    # ── Step 3: Merge daily data into a unified dict ──
    all_dates = set()
    all_dates.update(vo2_data.keys(), race_data.keys(), load_data.keys(),
                     heat_data.keys(), cycling_data.keys())

    daily_data = {}
    for d in all_dates:
        row = {}
        row.update(vo2_data.get(d, {}))
        row.update(race_data.get(d, {}))
        row.update(load_data.get(d, {}))
        row.update(heat_data.get(d, {}))
        row.update(cycling_data.get(d, {}))
        daily_data[d] = row

    # ── Step 4: Compute deltas for slow-moving metrics ──
    compute_deltas(daily_data, all_dates)

    # Count non-null delta values
    deltas_computed = sum(
        1 for d in daily_data.values()
        if d.get("vo2_max_running_delta") is not None
        or d.get("race_time_5k_delta") is not None
    )
    log.info(f"\n  Delta columns computed: {deltas_computed} dates "
          f"(need ≥7 days of history)")

    # ── Step 5: Upsert to production ──
    log.info(f"\n{'─' * 60}")
    log.info(f"  {'VALIDATION PREVIEW' if dry_run else 'MERGING TO PRODUCTION'}…")
    log.info(f"{'─' * 60}")

    daily_count = upsert_daily_metrics(prod_cur, daily_data, dry_run=dry_run)
    activity_count = upsert_activities(prod_cur, activity_data, dry_run=dry_run)

    if dry_run:
        log.info(f"\n  [DRY RUN] Would upsert {len(daily_data)} daily rows, "
              f"{len(activity_data)} activities")
    else:
        log.info(f"\n  ✅ daily_metrics:  {daily_count} rows upserted")
        log.info(f"  ✅ activities:     {activity_count} rows enriched")

        # Validate
        validate_merge(prod_cur, daily_data, activity_data)

    # Cleanup
    garmin_cur.close()
    garmin_conn.close()
    prod_cur.close()
    prod_conn.close()

    log.info(f"\n{'═' * 60}")
    log.info("  Done.")
    log.info(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
