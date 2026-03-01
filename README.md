# Garmin Health Intelligence 🏃‍♂️📊

A personal health analytics platform that turns Garmin wearable data into actionable insights. A **FastAPI REST backend** on Heroku serves a **static frontend** on GitHub Pages, backed by a custom-built statistical engine and a team of 5 specialized AI agents.

## 🎯 Project Goal

Garmin Connect provides excellent data tracking, but I wanted more depth. I wanted to understand the *relationships* between my metrics, not just see the numbers.

**Garmin Health Intelligence** answers "Why?":
* Why is my recovery low today despite sleeping 8 hours?
* How does my daily stress actually impact my training readiness?
* What is the single most impactful factor affecting my sleep quality?

Instead of relying on generic algorithms, this system uses custom-built mathematics to analyze *my* specific data patterns and provide personalized feedback — delivered through a live web dashboard with natural-language AI chat.

## 📸 Live Dashboard

The frontend is a static site hosted on **GitHub Pages** at [`gps_presentation_repo/garmin/index.html`](https://liranattar.dev/garmin/). It communicates with the Heroku backend via REST API calls.

![Agent Chat](docs/screenshots/agent_chat.png)
*Chatting with the AI agents to get data-backed answers to health questions.*

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AUTOMATED PIPELINE (GitHub Actions)               │
│                                                                         │
│  Step 1: FETCH         Garmin Connect API → PostgreSQL (upserts)        │
│  Step 2: CLASSIFY      Training intensity & body-part split             │
│  Step 3: COMPUTE       Correlation Engine (13 statistical layers)       │
│  Step 4: ANALYZE       5 AI Agents interpret the math (CrewAI)          │
│  Step 5: SAVE          Insights → DB + Recommendations → Memory         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      LIVE WEB APPLICATION                               │
│                                                                         │
│  Frontend (GitHub Pages)  ──REST──▶  FastAPI Backend (Heroku)           │
│  • Dashboard cards                  • 10 REST endpoints                 │
│  • AI Chat (async polling)          • Async agent chat (job_id)         │
│  • Metric history charts            • PostgreSQL on Heroku              │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | File | Lines | Role |
|---|---|---|---|
| **REST API** | `src/api.py` | 759 | FastAPI backend — 10 routes, async chat, health-check |
| **Route Helpers** | `src/routes/helpers.py` | 526 | DB queries, snapshot builder, fallback chat |
| **Data Fetcher** | `src/enhanced_fetcher.py` | 941 | Pulls data from Garmin Connect via the `garth` API |
| **Statistical Engine** | `src/correlation_engine.py` | 1,722 | Deterministic math — Pearson, AR(1), Markov chains, anomaly detection |
| **AI Agents** | `src/enhanced_agents.py` | 1,257 | 5 constrained specialist agents that interpret the math |
| **Pipeline** | `src/weekly_sync.py` | — | Orchestrates the automated pipeline end-to-end |
| **Summary Builder** | `src/pipeline/summary_builder.py` | — | 3-bullet concise summaries for UI cards |
| **Email** | `src/email_notifier.py` | — | Sends HTML email with recommendations |
| **Schema** | `src/enhanced_schema.py` | — | Database table definitions |
| **Bulk Import** | `src/bulk_import.py` + `src/garmin_merge.py` | — | FIT/ZIP file ingestion and DB merge |

> [!TIP]
> **For a deep dive into the math and agent design, see [Why the Math Is Mathing & Why Agents Aren't Hallucinating](why_the_math_mathing_and_why_agents_arent_hallucinating.md)** — a 600+ line technical document explaining every formula, every statistical test, and every anti-hallucination guardrail.

## 🌐 REST API

The backend exposes 10 endpoints via FastAPI, deployed on Heroku:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health-check` | Returns `{"status": "Online"}` — used by frontend to show connection status |
| GET | `/api/v1/snapshot/latest` | Today's metrics snapshot (body battery, sleep, HRV, readiness, etc.) |
| GET | `/api/v1/metrics/history` | 30-day daily metrics for charting |
| GET | `/api/v1/workouts/progress` | Recent activities with intensity & body-part classification |
| GET | `/api/v1/analytics/cross-effects` | Cross-metric correlation analysis from the statistical engine |
| GET | `/api/v1/insights/latest` | Most recent AI-generated insight summary |
| POST | `/api/v1/chat` | Submit a question → returns `{"job_id": "..."}` for async polling |
| GET | `/api/v1/chat/status/{job_id}` | Poll chat result → `{"status": "running"/"done"/"error", "answer": "..."}` |
| GET | `/api/v1/admin/migration-audit` | Database schema migration status |

### Chat Flow (Async)

1. Frontend sends `POST /api/v1/chat` with `{"message": "..."}`
2. Backend returns `{"job_id": "abc123"}` immediately
3. Frontend polls `GET /api/v1/chat/status/abc123` every 3 seconds
4. Backend runs agent analysis in a `BackgroundTask` — complexity routing decides multi-agent crew vs. single agent
5. When done, poll returns `{"status": "done", "answer": "..."}`
6. Fallback chain: multi-agent → simple agent → deterministic DB-only answer

### Deployment

- **Backend**: Heroku (`web: uvicorn src.api:app --host 0.0.0.0 --port $PORT`)
- **Frontend**: GitHub Pages (static HTML/JS)
- **Database**: PostgreSQL on Heroku (`DATABASE_URL` auto-provisioned)
- Connection string normalization: `postgres://` → `postgresql://` for psycopg2 compatibility

### Automation Schedule

| Task | Frequency | Platform | Purpose |
|---|---|---|---|
| **Daily Sync** | Daily (06:00 UTC) | GitHub Actions | Fetches recent Garmin data into PostgreSQL |
| **Export Reminder** | Weekly (Sun 08:00 UTC) | GitHub Actions | Sends reminder email to trigger Garmin export |
| **Bulk Import** | Daily (02:00 UTC) | GitHub Actions | Polls mailbox for fresh export link and imports |

### GitHub Actions Setup

Required GitHub Secrets:
- `POSTGRES_CONNECTION_STRING` / `DATABASE_URL`
- `GARMIN_EMAIL`, `GARMIN_PASSWORD`
- `EMAIL_APP_PASSWORD`, `EMAIL_RECIPIENT`

## 💾 Database Structure

PostgreSQL tables designed for time-series health analysis:

| Table | Purpose |
|---|---|
| `daily_metrics` | Core table — RHR, HRV, Sleep Score, Body Battery, Stress, Readiness, VO2Max, etc. |
| `activities` | Every workout — type, duration, intensity, physiological load |
| `wellness_log` | Self-reported data — energy, soreness, caffeine, nutrition |
| `matrix_summaries` | Pre-computed statistical context for the AI agents |
| `agent_recommendations` | Parsed recommendations with target metrics & expected directions |
| `weekly_summaries` | Aggregated weekly stats + full AI insight text |

## 🤖 The AI Agent Team

Instead of a single generic LLM, the system employs **5 specialized AI agents** (CrewAI + Google Gemini 2.5 Flash), each with a distinct role:

| # | Agent | Role | SQL? |
|---|---|---|---|
| 1 | **Statistical Interpreter** | Translates correlation matrices into plain English. Works purely on pre-computed math. | No |
| 2 | **Health Pattern Analyst** | Detects day-by-day patterns, flags outliers and data quality issues. | Yes |
| 3 | **Performance & Recovery** | Analyzes ACWR trajectory, training load, recovery capacity. | Yes |
| 4 | **Sleep & Lifestyle** | Sleep architecture deep-dive, connects lifestyle ↔ biometrics. | Yes |
| 5 | **Synthesizer** | Resolves conflicts, checks past recommendations (long-term memory), produces final insights. | Yes |

**Key design decisions:**
- Agents **never compute statistics** — all math is pre-computed by `CorrelationEngine`
- SQL access is **read-only** — `INSERT/UPDATE/DELETE/DROP` are blocked at the tool level
- 14 analytical rules are injected into every agent prompt (sample size, outlier checks, metric disambiguation, etc.)
- Partial-day data guardrail: agents are warned that today's row (synced at 06:00 UTC) contains only morning values
- `.ctx` property exposes analysis rules + physiological guide + DB schema hint for the chat agent path

> [!IMPORTANT]
> **The agents never compute statistics.** All math is performed by the deterministic `CorrelationEngine` *before* agents are invoked. Agents only interpret pre-validated results. See the [full technical explanation](why_the_math_mathing_and_why_agents_arent_hallucinating.md).

## 📧 Email Notifications

After each automated pipeline run, the system can send an HTML-formatted email with:
- 🎯 **Recommendation cards** extracted from the Synthesizer's output
- 📊 **Synthesizer analysis** summary (bottleneck + quick wins)
- Color-coded badges (IMPROVE / DECLINE / STABLE) for each recommendation

The email degrades gracefully — if `EMAIL_APP_PASSWORD` isn't set, the pipeline skips email with a log warning.

## 🚀 Key Features

* **Fully Automated:** Runs daily via GitHub Actions — zero manual intervention
* **Live Web Dashboard:** Real-time metrics, history charts, and AI chat via GitHub Pages frontend
* **Natural Language Chat:** Ask questions like *"How did my marathon training affect my sleep this month?"* and get answers grounded in your actual database — async processing with job polling
* **Long-Term Memory:** The Synthesizer tracks past recommendations and adjusts advice based on outcomes
* **Privacy Focused:** Personal data stays in your private database; only statistical summaries go to the LLM
* **Bulk Import Automation:** Garmin Data Export → ZIP Download → DB Ingest → Production Merge
* **Partial-Day Guardrail:** Agents are warned that today's data is incomplete (synced morning-only), preventing misinterpretation of accumulative metrics

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.12** | Core language |
| **FastAPI + uvicorn** | REST API backend |
| **PostgreSQL** | Time-series health database (Heroku) |
| **CrewAI** | Agent orchestration framework |
| **Google Gemini 2.5 Flash** | LLM for agent reasoning (low-cost, high-throughput) |
| **GitHub Pages** | Static frontend hosting |
| **GitHub Actions** | Automated daily/weekly pipelines |
| **Gmail SMTP** | Email notifications |
| **Heroku** | Backend hosting + managed PostgreSQL |

## 📂 Project Structure

```
garmin_project/
├── src/
│   ├── api.py                  # FastAPI backend (10 REST endpoints)
│   ├── correlation_engine.py   # Statistical engine (13 analysis layers)
│   ├── enhanced_agents.py      # 5 AI agents + tools + memory
│   ├── enhanced_fetcher.py     # Garmin API data fetcher
│   ├── weekly_sync.py          # Pipeline orchestrator
│   ├── email_notifier.py       # Email sender
│   ├── enhanced_schema.py      # DB schema definitions
│   ├── bulk_import.py          # ZIP/FIT file ingestion
│   ├── garmin_merge.py         # DB merge logic
│   ├── db_utils.py             # Shared DB connection helper
│   ├── constants.py            # Shared constants (body-part categories)
│   ├── routes/
│   │   └── helpers.py          # DB query helpers, fallback chat
│   └── pipeline/
│       └── summary_builder.py  # Concise summary generation
├── tests/                      # 109 pytest tests
├── docs/
│   ├── frontend_integration_contract.md
│   ├── technical_report.md
│   └── pipeline_health_contract.md
├── gps_presentation_repo/
│   └── garmin/index.html       # GitHub Pages frontend
├── .github/workflows/
│   ├── daily_sync.yml
│   ├── daily_bulk_import.yml
│   └── weekly_export_reminder.yml
├── why_the_math_mathing_and_why_agents_arent_hallucinating.md
├── Procfile                    # Heroku: uvicorn src.api:app
├── requirements.txt
└── .env                        # Credentials (not committed)
```

---

*This project is for educational and personal use, exploring the intersection of Data Engineering, Statistical Analysis, and Generative AI.*
