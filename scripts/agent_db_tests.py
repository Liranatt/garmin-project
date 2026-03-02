"""
Run a set of read-only agent-tool queries against the production DB
This script imports the tool functions from `src.enhanced_agents` and calls them.
Do NOT instantiate any Agent/LLM to avoid external calls.
"""
from src.enhanced_agents import (
    run_sql_query,
    calculate_correlation,
    find_best_days,
    analyze_pattern,
    get_past_recommendations,
)

queries = [
    ("run_sql_query", "SELECT date, resting_hr FROM daily_metrics ORDER BY date DESC LIMIT 5"),
    ("run_sql_query", "SELECT date, resting_hr FROM daily_metrics WHERE date > CURRENT_DATE ORDER BY date DESC LIMIT 5"),
    ("calculate_correlation", ("hrv_last_night", "sleep_score", 30)),
    ("calculate_correlation", ("resting_hr", "sleep_score", 90)),
    ("find_best_days", ("training_readiness", 5)),
    ("find_best_days", ("sleep_score", 3)),
    ("analyze_pattern", ("stress_level", 30)),
    ("analyze_pattern", ("sleep_score", 14)),
    ("get_past_recommendations", (4,)),
    ("run_sql_query", "SELECT activity_type, COUNT(*) AS cnt FROM activities GROUP BY activity_type ORDER BY cnt DESC LIMIT 10"),
]

print("Starting agent DB tests...\n")

for i, item in enumerate(queries, start=1):
    name, payload = item
    print(f"--- Test {i}: {name} ---")
    try:
        if name == "run_sql_query":
            out = run_sql_query(payload)
        elif name == "calculate_correlation":
            out = calculate_correlation(*payload)
        elif name == "find_best_days":
            out = find_best_days(*payload)
        elif name == "analyze_pattern":
            out = analyze_pattern(*payload)
        elif name == "get_past_recommendations":
            out = get_past_recommendations(*payload)
        else:
            out = f"Unknown test {name}"
    except Exception as e:
        out = f"EXCEPTION: {e!r}"
    print(out)
    print()

print("Agent DB tests complete.")
