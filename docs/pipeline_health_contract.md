# Pipeline Health Contract

## Status File
- Path: `src/pipeline_status.json` (configurable via `PIPELINE_STATUS_PATH`)
- Produced by: `WeeklySyncPipeline.run()`

## Required Fields
- `fetch_ok` (bool)
- `correlation_ok` (bool)
- `agents_ok` (bool)
- `insights_ok` (bool)
- `email_ok` (bool)
- `analysis_status` (`success|degraded|failed`)
- `degraded_reasons` (string list)
- `overall_status` (`success|degraded|failed`)
- `run_started_at` / `run_finished_at` (UTC ISO)

## Semantics
- `success`: all critical stages succeeded and correlation analysis is healthy.
- `degraded`: pipeline completed but analysis quality is reduced (for example Markov layer unavailable).
- `failed`: critical stage failed (fetch, core correlation layers, or insight persistence).

## Persistence
- Weekly row in `weekly_summaries` includes:
  - `analysis_status`
  - `pipeline_status_json` (JSONB)

## CI/Workflow Integration
- `daily_sync.yml` uploads `pipeline_status.json` as workflow artifact.
- Workflow log prints status JSON for quick triage.

