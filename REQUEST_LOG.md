# Request Log

Updated: 2026-02-21

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
| 16 | Check frontend chat rendering and fix corruption if needed | DONE | Patched `gps_presentation/garmin/index.html` `formatChatReply()` to preserve backend fallback/short replies (no forced "Key takeaways" rewrite); pushed commit `f13923c` to frontend `main`. |
| 17 | Investigate live chat fallback and patch backend AI import path | DONE | Updated `src/api.py` chat endpoint to import tools via `src.enhanced_agents` first (with fallback), fixing module-path issues in Heroku runtime. |
| 18 | Fix agent DB reachability in Heroku chat path | DONE | Added DB URL normalization + automatic `DATABASE_URL` -> `POSTGRES_CONNECTION_STRING` env alias in `src/api.py` so CrewAI tools can connect consistently. |
| 19 | Verify production chat and insights quality after deploy | DONE | Live website now returns valid AI responses (not fallback) and renders insights sections correctly. |
| 20 | Redesign Agent Insights to be concise and actually insightful | DONE | Frontend now renders compact action bullets instead of raw log-style blocks; backend now prefers concise recommendation text in `/api/v1/insights/latest`. |
| 21 | Remove Streamlit-migration wording and align page content to project narrative | DONE | Updated hero/footer copy in frontend to project-centric health intelligence language. |
| 22 | Make correlation heatmap explain meaning (not just color) | DONE | Added textual interpretation summary + per-cell meaning tooltips with direction/strength (`r`) in correlations view. |
| 23 | Fix window dropdown white-on-white unreadable options | DONE | Styled select/option foreground/background for readable contrast in analytics toolbar. |
| 24 | Add workout progress sport filters (running/cycling/swimming/skiing/gym) | DONE | Added sport filter UI and backend `sport` query support in `/api/v1/workouts/progress`; trends/workout summaries now scope to selected sport. |
| 25 | Expand matrices (including Pearson + Markov) for cross-activity effects | DONE | Added backend `/api/v1/analytics/cross-effects` with next-day sport-to-metric Pearson effects and readiness-state Markov transitions; integrated key effects into correlations UI summary. |
| 26 | Finish remaining strict insight schema + add dedicated tests, then push | DONE | Enforced structured insight fields (`headline`, `what_changed`, `why_it_matters`, `next_24_48h`, 3 bullets <=280 chars), updated agent output guidance, and added `tests/test_api_contract.py` for snapshot/workout/cross-effects/insight contracts. |
| 27 | Verify deployed Heroku API against new contracts | DONE | Live checks on 2026-02-20 passed for `/health-check`, snapshot/history/workouts/cross-effects/insights endpoints and `POST /api/v1/chat` (all HTTP 200 with expected structured fields). |
| 28 | Define agent-memory logging contract (question/context/output) | TODO | Needed before reinforcement phase: currently only recommendation snippets are persisted; add explicit run log model and storage policy. |
| 29 | P0: Fix production `agent_recommendations` schema drift | TODO | Production DB currently missing `agent_recommendations`; this blocks memory loop and causes insight-quality degradation. |
| 30 | P0: Ensure correlation context freshness for latest insights | TODO | `matrix_summaries` in production is stale vs latest sync date; must guarantee fresh pre-computed context or degrade explicitly. |
| 31 | P1: Explain and stabilize workflow retry/noise behavior | TODO | Investigate high `run_attempt` counts/retries and add actionable diagnostics + reliability guardrails. Evidence: run `22214619169` created `2026-02-20T06:56:31Z` later shows `run_attempt=6`; reruns can overwrite `run_started_at`, so scheduling analysis must key on `created_at` + attempt history. |
| 32 | P1: Bulk import IMAP auth failure tracking | PARKED | 2026-02-21 failed with `imaplib.IMAP4.error: [ALERT] Application-specific password required`; user will fix credentials later. |
| 33 | P1: Add README note on data collection start date | TODO | Clarify data starts at 2026-02-01 so initial null/sparse windows are expected. |
| 34 | P2: Document frontend deployment mismatch expectations | TODO | Clarify what changed in local frontend split and what should be visible only after deploy. |
| 35 | P2: Document/setup auto deploy to Heroku from GitHub | TODO | Reduce manual post-push deploy steps; low priority until blockers are solved. |
| 36 | Clarify GitHub Actions schedule vs local time (06:00 confusion) | TODO | Daily sync cron is UTC and can start late due queue; decide desired local-time trigger (06:00 local -> 04:00 UTC winter / 03:00 UTC summer) and update cron/expectations docs. |
| 37 | Improve insight quality pipeline design (reduce generic/placeholder outputs) | TODO | Add freshness gate for matrix context (fail/degrade if stale), require structured synthesizer block only for top-card, and stop publishing placeholder text as primary insight. |
| 38 | Explain and document chat-agent execution model | TODO | Add technical note on `POST /api/v1/chat`: single CrewAI agent, DB tools, no long-term memory, fallback path, and limits. |
| 39 | Design agent run-log model for future reinforcement | TODO | Add bounded `agent_run_logs` table: question, context hash/window, per-agent outputs, parsed recs, token/cost metadata, and retention policy to avoid noisy memory. |
| 40 | P0: Fix matrix context stale/missing path (insights freshness gate) | MERGED | Duplicate of #30 with implementation detail; keep #30 as canonical tracking item. |
| 41 | P0: Fix missing recommendations table in production | MERGED | Duplicate of #29 with implementation detail; keep #29 as canonical tracking item. |
| 42 | P1: Fix sparse/null-heavy data window effect on insight quality | TODO | How to fix: enforce minimum non-null observations per metric/window, filter all-null dates from scoring/correlation inputs, and surface confidence labels (`insufficient`, `preliminary`, `reliable`) in insight payloads. |
| 43 | Clarify "reliability/freshness in production, not intent" | DONE | Clarified: code path intends daily matrix compute (`weekly_pipeline -> compute_benchmarks -> matrix_summaries`), but production freshness/consistency is currently unreliable. |
| 44 | Document which CrewAI agents are active in chat vs insight pipeline | DONE | Clarified: chat uses one ad-hoc `Health Data Analyst` agent (`/api/v1/chat`); weekly insights use five agents in `AdvancedHealthAgents` (statistical_interpreter, health_pattern_analyst, performance_recovery, sleep_lifestyle, synthesizer). |
| 45 | P1: Add anti-hallucination guardrails for chat answers | TODO | How to fix: require evidence block per answer (queried window + metrics), enforce minimum sample-size rules for correlations, and downgrade to deterministic fallback when evidence quality is low. |
| 46 | Defer implementation of latest fixes to next cycle | PARKED | User requested logging-only for now; no code fixes applied in this step. |
| 47 | Deduplicate open backlog to avoid rework | DONE | Reviewed done vs pending items and collapsed duplicate P0 entries; `Next To-Do` now references canonical IDs only. |

## Environment / Access Notes

- DB access used by API: `POSTGRES_CONNECTION_STRING` (or `DATABASE_URL` fallback).
- Local verification in this workspace succeeded via `.env` and reached Heroku Postgres.
- System Python is missing `statsmodels`; use project venv for tests.

## Next To-Do

1. `#29` P0: Fix production `agent_recommendations` schema drift and verify write path from pipeline.
2. `#30` P0: Ensure correlation context freshness for latest insights (fresh matrix gate + degraded status if stale).
3. `#31` P1: Stabilize workflow retry/noise behavior with concrete diagnostics from run history.
4. `#36` P1: Clarify UTC schedule vs local 06:00 expectation and monitor trigger jitter.
5. `#37` + `#42` P1: Improve insight quality with sparse/null guards and confidence tiering.
6. `#45` P1: Add anti-hallucination guardrails for chat answers (evidence basis + min sample rules).
7. `#38` P2: Document chat execution architecture and limits.
8. `#28` + `#39` P2: Define agent run-log/memory contract for future reinforcement.
9. `#33` P1: Add README note that tracked data starts at `2026-02-01`.
10. `#32` PARKED: Bulk import IMAP auth fix after credential update.
11. `#34` + `#35` P2: Deployment/docs polish (frontend deploy expectations, optional GitHub -> Heroku auto-deploy).

## Implementation Ideas (Not Applied Yet)

1. Schedule reliability: keep cron in UTC, but set explicit target for your local 06:00 and monitor `run_started_at - scheduled_at` jitter over 14 days; alert if delay exceeds 20 minutes.
2. Daily matrix freshness gate: before publishing insights, verify `matrix_summaries.max(date) >= latest daily_metrics.date`; if stale, mark insight as degraded and show reason.
3. Insight quality control: rank outputs by evidence density (mentions of concrete metric + date + direction), and demote generic advice blocks.
4. Agent run logging (for future reinforcement): store each run’s question, context window/hash, raw agent outputs, parsed recommendations, and timestamp with retention limits.
5. Chat transparency: expose a lightweight “answer basis” block in API responses (which tool/period was used) so weak answers are easier to debug quickly.

## Captured Failure Logs

### 2026-02-21 - Daily Bulk Import Poller (IMAP auth)

```text
Run python src/bulk_import.py --auto --overlap-days 7
2026-02-21 04:09:44 [INFO] Bulk export request remains manual.
2026-02-21 04:09:44 [INFO] Please trigger export at: https://www.garmin.com/en-US/account/datamanagement/exportdata/
Traceback (most recent call last):
  File "/home/runner/work/garmin-project/garmin-project/src/bulk_import.py", line 638, in <module>
    main()
  File "/home/runner/work/garmin-project/garmin-project/src/bulk_import.py", line 626, in main
    success = run_bulk_import(
              ^^^^^^^^^^^^^^^^
  File "/home/runner/work/garmin-project/garmin-project/src/bulk_import.py", line 556, in run_bulk_import
    link, candidate, reason = monitor_email_for_link(
                              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/garmin-project/garmin-project/src/bulk_import.py", line 334, in monitor_email_for_link
    candidate = find_latest_export_email(lookback_days=lookback_days)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/runner/work/garmin-project/garmin-project/src/bulk_import.py", line 287, in find_latest_export_email
    mail.login(user, password)
  File "/opt/hostedtoolcache/Python/3.12.12/x64/lib/python3.12/imaplib.py", line 621, in login
    raise self.error(dat[-1])
imaplib.IMAP4.error: b'[ALERT] Application-specific password required: https://support.google.com/accounts/answer/185833 (Failure)'
Error: Process completed with exit code 1.
```
