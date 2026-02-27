# Production Migration Runbook

## Purpose
Run idempotent schema upgrades before pipeline execution and verify readiness.

## Preconditions
- `POSTGRES_CONNECTION_STRING` is configured.
- Deployment uses additive DDL only (`IF NOT EXISTS`, `ALTER ... ADD COLUMN IF NOT EXISTS`).

## Startup Migration Entry
- Python module: `src/pipeline/migrations.py`
- Function: `ensure_startup_schema()`

## What It Applies
- Enhanced tables/views via `enhanced_schema.upgrade_database()`
- Correlation engine table bootstrap (`matrix_summaries`)
- Additional reliability columns:
  - `weekly_summaries.analysis_status`
  - `weekly_summaries.pipeline_status_json`
- `agent_recommendations` weekly index guard.

## Verification
1. Call API endpoint: `GET /api/v1/admin/migration-audit`
2. Confirm:
   - required tables exist
   - required columns exist
   - `ok=true`

## Rollback Strategy
- This slice is additive; rollback by deploying previous app version.
- If needed, ignore new columns at application layer first.
- Do not drop new columns/tables during incident response unless explicitly approved.

