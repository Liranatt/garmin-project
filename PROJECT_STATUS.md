# Garmin Health Analytics — Project Status

*Last updated: February 15, 2026*

## What This Project Is

A personal health analytics platform that pulls data from a Garmin watch, runs statistical analysis and AI-powered insights using 8 specialized agents, and displays everything on an interactive dashboard.

### Architecture

```
Garmin Watch
    ↓
Garmin Connect Cloud
    ↓ (garth API)
EnhancedGarminDataFetcher  →  PostgreSQL on Heroku (daily_metrics, activities, body_battery_events)
    ↓                              ↓
GarminMerge (enrichment)    CorrelationEngine (Pearson, lag-1, AR(1), Markov, KL-divergence)
    ↓                              ↓
    └──────────────────→  8 AI Agents (CrewAI + Gemini 2.5 Flash)
                                    ↓
                            Streamlit Dashboard (6 pages)
```

### The 8 Agents

| # | Agent | What It Does |
|---|-------|-------------|
| 1 | **Matrix Analyst** | Interprets pre-computed correlation matrices — same-day correlations, next-day predictors, AR(1) persistence, Markov transitions, KL-divergence, anomalies |
| 2 | **Pattern Detective** | Finds 3–5 hidden day-by-day patterns beyond correlations, flags suspect data (e.g. `rem_sleep_sec=0`), examines bounce-back sequences |
| 3 | **Performance Optimizer** | Week-over-week comparison table for every metric — change %, best/worst day, ACWR analysis, activity summary |
| 4 | **Recovery Specialist** | Assesses recovery capacity — HRV bounce-back, RHR normalization, BB recharge after hard days, overtraining checklist |
| 5 | **Trend Forecaster** | Per-metric trend analysis with CV, directional trend detection (3+ consecutive), outlier removal test, early warnings |
| 6 | **Lifestyle Analyst** | Connects specific activities to next-day biometric responses, classifies activities (commute vs training), ACWR trajectory |
| 7 | **Weakness Identifier** | Fact-checks all previous agents, identifies #1 bottleneck with evidence strength, proposes 3 quick wins with confidence ratings |
| 8 | **Multi-Scale Benchmark Analyst** | Compares correlation matrices across time windows (7d→365d), classifies findings as ROBUST/STABLE/EMERGING/FRAGILE |

### Tech Stack

- **Data Fetching**: `garth` (Garmin Connect API)
- **Database**: PostgreSQL on Heroku (Essential-0, $5/month)
- **Statistics**: `scipy`, `numpy`, `pandas` — Pearson, lag-1, AR(1), conditioned AR(1), kernel-smoothed Markov, KL-divergence, Shapiro-Wilk
- **AI Agents**: `crewai` + `langchain-google-genai` + Gemini 2.5 Flash
- **Dashboard**: `streamlit` + `plotly`
- **Sync**: `weekly_sync.py` CLI pipeline

---

## What Has Been Done ✅

### Core Pipeline (working end-to-end)
- [x] **Garmin data fetching** (`src/enhanced_fetcher.py` — 842 lines) — Authenticates via garth, fetches 11 data types, builds unified daily rows, UPSERTs into PostgreSQL
- [x] **Training classification** — Auto-classifies training intensity and body parts from exercise-sets API
- [x] **Correlation engine** (`src/correlation_engine.py` — 1198 lines) — 4-layer statistical analysis + multi-period benchmarks (7d→365d). Stores to `matrix_summaries` table
- [x] **8 AI agents** (`src/enhanced_agents.py` — 1090 lines) — All 8 agents defined with custom SQL tools (read-only guarded), 3 execution modes
- [x] **Dashboard** (`src/dashboard.py` — 427 lines) — Streamlit app with 6 pages
- [x] **Weekly sync pipeline** (`src/weekly_sync.py` — 270 lines) — Orchestrates: Fetch → Classify → Benchmark Correlations → AI Agents → Save insights
- [x] **Data enrichment** (`src/garmin_merge.py` — 555 lines) — Merges VO2 max, race predictions, ACWR from GarminDB bulk export
- [x] **Visualizations** (`src/visualizations.py` — 539 lines) — Plotly charts with HTML export
- [x] **Database schema** (`src/enhanced_schema.py` — 263 lines) — Supplementary tables and views

### Infrastructure (completed)
- [x] **Heroku Postgres** — Database migrated from local to Heroku Essential-0 ($5/month). App: `garmin-health-liran`
- [x] **`.env` updated** — Points to Heroku Postgres (local connection string preserved as comment)
- [x] **`.env.example` created** — Template with dummy values for all required env vars
- [x] **`.gitignore` updated** — Covers `.env`, `data/`, `archive/`, `agent_output*.txt`, `*.dump`, `.garth/`
- [x] **`requirements.txt` cleaned** — Removed unused `langchain-openai`, `schedule`, `matplotlib`, `seaborn`. Added `langchain-google-genai`
- [x] **`README.md` created** — 587-line comprehensive README with architecture, setup, usage, deployment options

### Bug Fixes (completed)
- [x] **SQL column mismatches fixed** — `dashboard.py` and `visualizations.py` now use `duration_sec` and `distance_m` (matching production schema)
- [x] **Hardcoded dates removed** — `enhanced_fetcher.py` uses `today - 14 days`, `garmin_merge.py` uses `today - 30 days` as defaults
- [x] **Fake hydration data removed** — Now stores `NULL` instead of random values, so agents won't analyze fabricated data
- [x] **SQL read-only guard added** — `run_sql_query` in `enhanced_agents.py` rejects INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE

### Exploratory / Testing
- [x] **Manual test scripts** in `tests/` — 9 test files covering connections, API, agents, correlations, E2E
- [x] **Debug & analysis scripts** in `scripts/` — 5 utility scripts
- [x] **Agent output samples** — `agent_output_7agents.txt`, `agent_output_8agents_full.txt`, `agent_output_8agents_v2.txt`

---

## Known Issues / Remaining Work 🟡

### Stubs (ship as-is)

1. **Goals page** — Shows "coming soon" message, only displays weekly stats
2. **Settings sync button** — Shows instructions to run CLI instead of actually syncing
3. **Schema tables never populated** — `wellness_log`, `nutrition_log`, `personal_records`, `goals` tables exist but have no input mechanism
4. **ConstantInputWarning spam** — `pearsonr()` warns on constant-input columns (harmless, only with < 30 days of data). Will self-resolve as more data accumulates

---

## Pre-Deployment Checklist — All Done ✅

### Phase 1: Security & Cleanup
- [x] **`.env.example` created** — Template with dummy values
- [x] **`.gitignore` updated** — Covers secrets, data, dumps
- [x] **`requirements.txt` cleaned** — Removed unused deps, added `langchain-google-genai`

### Phase 2: Bug Fixes
- [x] **SQL column mismatches** — `duration_sec` / `distance_m` everywhere
- [x] **Hardcoded dates** — Replaced with dynamic `today - N days` defaults
- [x] **Fake hydration data** — Now stores `NULL` instead of `np.random.uniform()`
- [x] **SQL read-only guard** — Agent SQL tool rejects INSERT/UPDATE/DELETE/DROP/ALTER/TRUNCATE/CREATE

### Phase 3: Infrastructure
- [x] **GitHub Actions workflow** — `.github/workflows/weekly_sync.yml`, cron every Sunday 06:00 UTC, garth session persistence
- [x] **Streamlit Cloud config** — `.streamlit/config.toml`, `dashboard.py` reads `st.secrets` with fallback to `os.getenv()`
- [x] **Print→logging migration** — 160 print() calls replaced with structured `logging` across all 8 modules
- [x] **Legacy module archived** — `garmin_health_tracker.py` → `archive/garmin_health_tracker_OLD.py`, `__init__.py` cleaned

### Phase 4: Verification
- [x] **End-to-end verification** — `scripts/verify_deployment.py` passes all 5 checks:
  - Heroku DB connection & row counts (28/17/35/1/1)
  - All 8 module imports
  - `CorrelationEngine.compute_benchmarks()` — 4 periods computed
  - 0 residual `print()` calls in `src/`
  - Structured logging confirmed on all modules

---

## What To Do Next — Deploy 🚀

### Push to GitHub
- [ ] `git init && git add -A && git commit -m "Initial commit"`
- [ ] Create GitHub repo and push
- [ ] Configure GitHub Secrets: `POSTGRES_CONNECTION_STRING`, `GARMIN_EMAIL`, `GARMIN_PASSWORD`, `GOOGLE_API_KEY`, `GARTH_SESSION` (base64-encoded garth auth)
- [ ] Test workflow with manual trigger on Actions tab
- [ ] Enable cron schedule

### Connect Streamlit Community Cloud
- [ ] Link GitHub repo to Streamlit Cloud
- [ ] Set `POSTGRES_CONNECTION_STRING` and `GOOGLE_API_KEY` in Streamlit Secrets
- [ ] Verify all 6 dashboard pages load

---

## Deployment Target

| Component | Where | Cost |
|-----------|-------|------|
| Code | GitHub (Pro) | Free |
| Database | Heroku Postgres Essential-0 | $5/month |
| Weekly Sync | GitHub Actions (cron) | Free (2,000 min/month) |
| Dashboard | Streamlit Community Cloud | Free |
| **Total** | | **$5/month** |

### Environment Variables

| Variable | Where to Set | Used By |
|----------|-------------|---------|
| `GARMIN_EMAIL` | GitHub Secrets | weekly_sync (Actions) |
| `GARMIN_PASSWORD` | GitHub Secrets | weekly_sync (Actions) |
| `POSTGRES_CONNECTION_STRING` | GitHub Secrets + Streamlit Secrets | Both |
| `GOOGLE_API_KEY` | GitHub Secrets + Streamlit Secrets | Both |
| `GARTH_HOME` | GitHub Actions env | weekly_sync (Actions) |
| `GARMIN_CONNECTION_STRING` | Local `.env` only | garmin_merge (local runs only) |

---

## File Map

```
garmin_project/
├── .env                          # Local secrets (never committed)
├── .env.example                  # Template for env vars
├── .gitignore                    # Git exclusions
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── PROJECT_STATUS.md             # This file
│
├── src/
│   ├── weekly_sync.py            # Entry point — orchestrates full pipeline
│   ├── enhanced_fetcher.py       # Garmin API → PostgreSQL
│   ├── enhanced_agents.py        # 8 CrewAI agents + read-only SQL tools
│   ├── correlation_engine.py     # Statistical analysis + multi-period benchmarks
│   ├── dashboard.py              # Streamlit app (6 pages)
│   ├── visualizations.py         # Plotly chart generators
│   ├── enhanced_schema.py        # Supplementary DB tables/views
│   ├── garmin_merge.py           # VO2max/ACWR enrichment from GarminDB
│   ├── garmin_health_tracker.py  # Legacy module (to be deleted)
│   └── __init__.py
│
├── tests/                        # Manual test scripts (not pytest)
├── scripts/                      # Debug & exploration utilities
├── data/                         # Exploratory CSV exports (gitignored)
└── archive/                      # Old module versions (gitignored)
```
