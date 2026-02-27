# Stability Baseline (2026-02-21)

## Scope
- Repository snapshot date: 2026-02-21
- Baseline target: daily sync reliability, correlation pipeline status signaling, schema consistency, and frontend insight/correlation semantics.

## Local Verification Baseline
- Test command: `.venv\\Scripts\\python.exe -m pytest -q`
- Result: `65 passed, 1 warning`
- Warning observed: `.pytest_cache` write permission warning (non-blocking for runtime behavior).

## Key File Size Baseline (before this slice)
- `src/correlation_engine.py`: 1647 lines
- `src/enhanced_agents.py`: 1074 lines
- `src/api.py`: 1089 lines
- `gps_presentation_repo/garmin/index.html`: 2817 lines

## Reliability Gaps Confirmed
- Markov layer expected a `date` column while `_compute_raw()` passed metrics-only data.
- Pipeline return status did not explicitly separate `success` vs `degraded` vs `failed`.
- Weekly summary rows did not consistently include machine-readable pipeline status.

## Baseline Rollback Marker
- Use git commit/tag before rollout for rollback.
- Keep additive schema updates only (no destructive migrations in this slice).

