# Request Log

Updated: 2026-02-20

## Active Thread Requests

| ID | Request | Status | Notes |
|---|---|---|---|
| 1 | Read project and acknowledge frontend integration contract | DONE | Contract acknowledged from `docs/frontend_integration_contract.md`. |
| 2 | Finish frontend integration contract (backend side) | DONE | Implemented in `src/api.py` (`/health-check`, `/api/v1/snapshot/latest`, `/api/v1/metrics/history`, `/api/v1/workouts/progress`, `/api/v1/insights/latest`, `/api/v1/chat`) with CORS for `https://liranattar.dev`. |
| 3 | Fix/check workout progress endpoint and verify data availability | DONE | Endpoint implemented and DB checked: running/speed/cadence/training-load fields exist (sparse sample size). |
| 4 | Clarify weekly vs daily sync naming in docs | DONE | Updated README wording and workflow filename reference to daily sync. |
| 5 | Add CI test execution | DONE | Added `pytest -q` step to `.github/workflows/daily_sync.yml`. |
| 6 | Keep a running request log in project root and mark done when finished | DONE | This file is now the source of truth and will be updated each conversation. |
| 7 | Bulk export automation follow-up (after API is done) | PARKED | Waiting for your detailed requirements after API completion. |
| 8 | Run tests and verify everything works | DONE | Full suite passed with project venv: `60 passed` (`.venv\\Scripts\\python.exe -m pytest -q`). |
| 9 | Implement bulk export automation mission before testing | DONE | Implemented robust Gmail export detection + freshness gating + staged ingest/merge flow in `src/bulk_import.py`; reminder recipient override added. |
| 10 | Try bulk export flow end-to-end with current mailbox data | DONE | Pipeline runs and finds latest Garmin email, but current Feb 17 pre-signed S3 download link is expired; run exits cleanly and waits for fresh export. |
| 11 | Ensure automation does not depend on local computer being on | DONE | Added bulk import poller step to GitHub Actions in `.github/workflows/daily_sync.yml` so it runs on CI runners. |
| 12 | Move bulk import automation to Heroku execution | DONE | Removed bulk-import step from GitHub Actions, added Heroku process commands in `Procfile`, and documented Heroku Scheduler setup in `README.md`. |
| 13 | Move scheduling to GitHub Actions (weekly reminder + daily import poller) | DONE | Added `.github/workflows/weekly_export_reminder.yml` and `.github/workflows/daily_bulk_import.yml`; updated README schedule docs. |
| 14 | Confirm backend still matches frontend integration contract | DONE | Local route smoke-check returned HTTP 200 on all contract endpoints and chat response alias fields. |
| 15 | Fix non-fast-forward GitHub push on `main` | DONE | Rebased local commits onto `origin/main`, resolved `src/api.py` conflict, and pushed successfully (`81ff9d9`). |

## Environment / Access Notes

- DB access used by API: `POSTGRES_CONNECTION_STRING` (or `DATABASE_URL` fallback).
- Local verification in this workspace succeeded via `.env` and reached Heroku Postgres.
- System Python is missing `statsmodels`; use project venv for tests.

## Next To-Do

- Validate first scheduled GitHub Actions runs (weekly reminder workflow + daily bulk import poller) and confirm Heroku deploy health endpoint.
