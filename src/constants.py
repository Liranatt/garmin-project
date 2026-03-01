"""
Shared constants used across multiple modules.
Single source of truth for body-part exercise categories.
"""

# Exercise categories for body-part detection (from Garmin exercise-sets API)
UPPER_BODY_CATS = {
    "PULL_UP", "ROW", "SHOULDER_PRESS", "BENCH_PRESS", "CURL",
    "LATERAL_RAISE", "TRICEPS_EXTENSION", "CHEST_FLY", "PUSHUP",
    "DIP", "DEADLIFT", "T_BAR_ROW",
}
LOWER_BODY_CATS = {
    "SQUAT", "LUNGE", "LEG_PRESS", "LEG_CURL", "LEG_EXTENSION",
    "CALF_RAISE", "HIP", "GLUTE",
}
