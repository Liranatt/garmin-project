# Hands-Free Deployment Readiness Audit (2026-02-21)

## Verdict
**Not ready for hands-free deployment yet.**

This audit was run on **2026-02-21** across backend, frontend, GitHub Actions, and Heroku (live app + repo state).

## Findings (Ordered by Severity)

### P0 - Production schema drift: `agent_recommendations` table missing
- Impact:
  - Recommendation memory loop is inactive in production.
  - Any production query that expects `agent_recommendations` will fail.
- Evidence:
  - Live DB column check showed `agent_recommendations_columns=` (empty).
  - Direct query to `agent_recommendations` failed with `psycopg2.errors.UndefinedTable`.
- Relevant code paths expecting this table:
  - `src/enhanced_agents.py:174`
  - `src/enhanced_agents.py:245`
  - `src/enhanced_agents.py:278`
  - `src/api.py:970`

### P0 - Correlation context appears stale in production while sync reports success
- Impact:
  - Insight quality is degraded while pipeline health appears successful.
  - Agents produce placeholder-like output instead of data-grounded correlation interpretation.
- Evidence:
  - `matrix_summaries` latest row is dated **2026-02-15**.
  - Latest weekly summary row created **2026-02-21** contains text indicating pre-computed correlation analysis was not provided.
  - Live endpoint `/api/v1/insights/latest` currently returns that degraded narrative.
  - Latest scheduled `Daily Health Sync` run on GitHub Actions completed `success` on **2026-02-21**.

### P1 - Data quality issue in production history payload (many null-only days)
- Impact:
  - Frontend trends/correlation interpretation can be skewed by empty-day density.
  - User-facing analytics confidence is reduced.
- Evidence:
  - Live DB query over last 30 days:
    - `total_rows_30d=31`
    - `all_null_core_rows_30d=14`
  - Sample null-only dates: `2026-01-22` through `2026-02-04`.
  - Live endpoint `/api/v1/metrics/history?days=30` reflects these null-heavy rows.

### P1 - Bulk import automation is currently failing on schedule
- Impact:
  - Hands-free ingestion is not reliable.
- Evidence:
  - Latest `Daily Bulk Import Poller` scheduled run failed:
    - Run ID: `22250040483`
    - Date: **2026-02-21T04:08:44Z**
    - Failed step: `Run bulk import poller`
- Limitation:
  - Full logs were not retrievable via unauthenticated GitHub job-log API (403 admin-rights requirement).

### P1 - Workflow reliability is noisy (multiple reruns)
- Impact:
  - “Unattended-safe” confidence is not yet established.
- Evidence:
  - `Daily Health Sync` run ID `22214619169` shows `run_attempt=6`.
  - Another recent run showed elevated retries historically.

### P2 - Deployment/doc drift around required secrets
- Impact:
  - Setup drift can cause avoidable misconfigurations.
- Evidence:
  - Bulk workflow now expects `EMAIL_IMAP_USER`:
    - `.github/workflows/daily_bulk_import.yml:31`
  - README required secrets list does **not** include it:
    - `README.md:75`
    - `README.md:80`
  - `.env.example` does not document email vars used by automation (`EMAIL_APP_PASSWORD`, `EMAIL_RECIPIENT`, `EMAIL_IMAP_USER`).

### P2 - Production Heroku app does not yet include new audit/status endpoints
- Impact:
  - Operational visibility remains limited in current deployed slug.
- Evidence:
  - `/api/v1/admin/migration-audit` returns `404` in production.
  - This endpoint exists in local repo at `src/api.py:1114`.

### P2 - Local/frontend refactor is not yet reflected in production site
- Impact:
  - Production is still serving monolithic inline HTML/JS/CSS.
- Evidence:
  - Local refactor references split assets:
    - `gps_presentation_repo/garmin/index.html:11`
    - `gps_presentation_repo/garmin/index.html:224`
  - Live site `https://liranattar.dev/garmin/` still returns inline `<style>`/inline script content.

## Environment Checks Per Surface

### Backend (Local)
- `pytest`: **71 passed**
  - Command: `.venv\\Scripts\\python.exe -B -m pytest -q -p no:cacheprovider`
- `weekly_sync` CLI import/help: OK
- API import: OK

### Frontend (Local)
- JavaScript syntax check: OK
  - Command: `node --check gps_presentation_repo/garmin/frontend/js/app.js`

### GitHub Actions (Remote)
- Workflow YAML parse: OK for all `.github/workflows/*.yml`
- Latest runs:
  - Daily sync: success (2026-02-21)
  - Daily bulk import: failure (2026-02-21)

### Heroku (Remote)
- App: `garmin-health-liran`
- Dynos: `web: 1` up
- Add-ons: Postgres + Scheduler present
- Config keys present: `DATABASE_URL`, `GARMIN_*`, `GOOGLE_API_KEY`, `EMAIL_*`
- Config caveat:
  - `POSTGRES_CONNECTION_STRING` not present on Heroku (current web API handles fallback from `DATABASE_URL`, but not every historical path did).

## Readiness Decision
Hands-free deployment should be considered **blocked** until these are resolved in production:
1. `agent_recommendations` table existence and write path validation.
2. Correlation pipeline freshness/status (matrix summary should update daily when sync succeeds).
3. Bulk import scheduled run stability.
4. Null-only daily rows root-cause analysis and mitigation.

## What I changed during this audit
- I did **not** apply fixes.
- I created this report file only:
  - `docs/hands_free_deployment_readiness_2026-02-21.md`

Note: the repository already had pre-existing modified/untracked files before this audit began.
