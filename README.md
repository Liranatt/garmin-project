# Garmin Health Intelligence

AI-powered health analytics that goes far beyond what Garmin Connect shows you. Fetches biometric data from Garmin, computes statistical correlations, and runs 9 specialized AI agents to deliver actionable health insights — all stored in PostgreSQL (Heroku) and presented via an interactive dashboard.

---

## What This Does vs Garmin Connect

| Capability | Garmin Connect | This Project |
|---|---|---|
| Raw metrics display | Yes | Yes |
| Week-over-week trends | Limited | Full comparison |
| Cross-metric correlations | No | Pearson + lagged |
| AI-powered insights | No | 9 specialized agents |
| Statistical anomaly detection | No | Z-scores + Shapiro-Wilk |
| Training load analysis (ACWR) | Basic | Acute/chronic ratio |
| Custom SQL queries | No | Full database access |
| Automated weekly reports | No | Cron / GitHub Actions |
| Interactive Plotly charts | No | Dashboard |
| Body battery patterns | Basic | Charge/drain/transitions |
| Recovery optimization | No | Sleep↔readiness analysis |
| Overtraining detection | No | HRV + load patterns |

---

## Architecture

```
Garmin Watch → Garmin Connect → Garmin API
                                     ↓
                           garth library (OAuth)
                                     ↓
                     EnhancedGarminDataFetcher
                        (src/enhanced_fetcher.py)
                                     ↓
                           PostgreSQL on Heroku (UPSERT)
                          ┌──────────┴──────────┐
                          ↓                     ↓
                CorrelationEngine        garmin_merge.py
               (src/correlation_engine.py)   (bulk enrichment)
                          ↓
                 AdvancedHealthAgents
                (src/enhanced_agents.py)
                9 CrewAI agents + Gemini
                          ↓
                weekly_summaries table
                          ↓
           ┌──────────────┴──────────────┐
           ↓                             ↓
   Streamlit Dashboard          CLI / Direct SQL
    (src/dashboard.py)
           ↓
  HealthDataVisualizer
   (src/visualizations.py)
```

---

## Project Structure

```
garmin_project/
├── README.md               # This file
├── .env.example            # Template for environment variables
├── .gitignore
├── requirements.txt        # Python dependencies
├── PROJECT_STATUS.md       # Development status & roadmap
│
├── src/                    # Core application modules
│   ├── __init__.py
│   ├── enhanced_fetcher.py     # Garmin data collection via garth
│   ├── correlation_engine.py   # Statistical analysis (4-layer) + adaptive kernel smoothing
│   ├── enhanced_agents.py      # 9 AI agents (CrewAI + Gemini)
│   ├── enhanced_schema.py      # Extended database schema
│   ├── garmin_health_tracker.py # Config, DatabaseManager, core classes
│   ├── garmin_merge.py         # Bulk Garmin DB merge utility
│   ├── visualizations.py       # Plotly chart generation
│   ├── dashboard.py            # Streamlit web dashboard
│   └── weekly_sync.py          # Weekly automation pipeline
│
├── .github/workflows/      # CI/CD
│   └── weekly_sync.yml         # Sunday cron automation
│
└── .streamlit/             # Dashboard config
    └── config.toml             # Dark theme, green accent
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (local or Heroku)
- Garmin Connect account (with watch syncing data)
- Google AI API key (for Gemini — free tier works)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/garmin-health-tracker.git
cd garmin-health-tracker
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```env
# Garmin Connect credentials
GARMIN_EMAIL=your@email.com
GARMIN_PASSWORD=your_password

# PostgreSQL
POSTGRES_CONNECTION_STRING=postgresql://user:pass@localhost:5432/postgres

# Google Gemini API (https://aistudio.google.com/apikey)
GOOGLE_API_KEY=your_gemini_api_key

# Garth OAuth token storage
GARTH_HOME=~/.garth

# Optional: Garmin bulk download DB (for garmin_merge.py)
# GARMIN_CONNECTION_STRING=postgresql://user:pass@localhost:5432/garmin
```

### 3. First Run

```bash
# Full pipeline: fetch data → compute correlations → run AI agents
python src/weekly_sync.py

# Or step by step:
python src/weekly_sync.py --fetch     # Fetch only (skip AI)
python src/weekly_sync.py --analyze   # Analyze only (skip fetch)
python src/weekly_sync.py --days 14   # Custom date range
```

On first run, Garmin may prompt for MFA — enter the code in the terminal.

### 4. Launch Dashboard

```bash
streamlit run src/dashboard.py
```

---

## Core Modules

### Data Fetcher (`src/enhanced_fetcher.py`)

Collects **all** available Garmin metrics using the `garth` library with OAuth:

- **Heart rate**: resting, max, min
- **HRV**: last night, 5-min high, weekly avg, status
- **Stress**: average + duration breakdowns (rest/low/medium/high)
- **Sleep**: duration, deep/light/REM/awake, score, respiration, skin temp
- **Body battery**: charged, drained, peak, low, intraday events
- **Steps**: total daily count
- **Training readiness**: overall score + 10 sub-scores
- **Intensity minutes**: moderate + vigorous
- **Weight & hydration**
- **Activities**: type, duration, distance, HR, cadence, elevation, calories

All data is written to PostgreSQL with **UPSERT** (idempotent — safe to re-run).

Training days are classified as **HARD / MODERATE / EASY / REST** based on activity type, duration, and heart rate. Upper/lower body detection via exercise-sets API.

### Correlation Engine (`src/correlation_engine.py`)

4-layer statistical analysis computed from `daily_metrics`:

| Layer | Analysis | Output |
|-------|----------|--------|
| **0** | Data loading, cleaning, auto-discovery of numeric columns | Clean DataFrame |
| **1** | NxN Pearson correlations (same-day + lag-1), p-values | Correlation matrices |
| **2** | AR(1) persistence, normality (Shapiro-Wilk), anomaly detection (z-scores), conditioned AR(1), Markov transition matrices, KL-divergence | Statistical summaries |
| **3** | Natural-language summary (~2-3 KB) | Text for AI agents |

Uses **adaptive kernel-smoothed** Markov matrices (α = BASE/√n, clamped to [0.02, 0.25] — heavy smoothing when sparse, light when data is large) and adjusted R². KL-divergence gated: requires ≥5 transitions per conditioning level. Results stored in `matrix_summaries` table.

### AI Agents (`src/enhanced_agents.py`)

9 specialized CrewAI agents powered by Google Gemini (`gemini-2.5-flash`):

| Agent | Role |
|-------|------|
| **Matrix Analyst** | Interprets correlation matrices and statistical patterns |
| **Multi-Scale Benchmark Analyst** | Compares patterns across time windows (7d→365d), classifies as ROBUST/STABLE/EMERGING/FRAGILE |
| **Pattern Detective** | Discovers hidden relationships in the data |
| **Performance Optimizer** | Recommends training and recovery strategies |
| **Recovery Specialist** | Analyzes recovery bounce-back and overtraining signals |
| **Trend Forecaster** | Projects trends and flags concerning directions |
| **Lifestyle Analyst** | Links lifestyle choices to health outcomes |
| **Sleep Analyst** | Deep sleep architecture analysis — duration, deep/REM/light breakdown, consistency, next-day impact |
| **Weakness Identifier** | Synthesizes all findings into #1 bottleneck + quick wins |

Each agent has access to 4 tools for direct database queries:
- `run_sql_query` — Execute SQL against the health database (read-only guard)
- `calculate_correlation` — Compute correlation between any two metrics
- `find_best_days` — Find days with optimal metric values
- `analyze_pattern` — Detect patterns over time windows

**Analysis modes:**

```python
from enhanced_agents import AdvancedHealthAgents

agents = AdvancedHealthAgents(db_manager=None)

# Weekly summary (all 9 agents)
result = agents.run_weekly_summary(matrix_context="...")

# Comprehensive deep analysis (6 agents)
result = agents.run_comprehensive_analysis(days=30)

# Goal-specific analysis (2 agents)
result = agents.run_goal_analysis(goal="improve sleep quality")
```

### Dashboard (`src/dashboard.py`)

Interactive Streamlit web app with:

- **Overview**: Key metrics at a glance + trend sparklines
- **Trends & Analysis**: Recovery analysis, training load, correlation heatmaps
- **Deep Dive**: Single-metric historical analysis
- **AI Insights**: Run agents on demand from the UI
- **Goals & Progress**: Track custom health goals
- **Settings**: Configure date ranges, visualization preferences

### Visualizations (`src/visualizations.py`)

Plotly-based chart library:

- Weekly dashboard (4×2 subplot grid)
- Recovery analysis (sleep→readiness, HRV vs RHR, stress→sleep)
- Training analysis (load curves, activity distribution, body battery, training effects)
- Correlation heatmap
- Progress report (week-over-week comparison)
- Calendar heatmap (GitHub-style)
- Export all charts to HTML

### Weekly Sync (`src/weekly_sync.py`)

Main automation entry point. Orchestrates the full pipeline:

1. **Fetch** — Last 7 days of Garmin data → PostgreSQL
2. **Correlate** — Compute statistical matrices + multi-period benchmarks
3. **Analyze** — Run 9 AI agents on the data
4. **Store** — Save insights to `weekly_summaries` table

```bash
python src/weekly_sync.py              # Full pipeline
python src/weekly_sync.py --fetch      # Fetch only
python src/weekly_sync.py --analyze    # Analyze only
python src/weekly_sync.py --days 14    # Custom range
```

### Garmin Merge (`src/garmin_merge.py`)

Enriches the production database with data from a separate Garmin bulk-download database:

- VO2 max (running/cycling)
- Race predictions (5K, 10K, half marathon)
- Acute/chronic training load + ACWR
- Heat/altitude acclimation status
- Cycling abilities (FTP, etc.)
- Activity-level training effects and load

```bash
python src/garmin_merge.py                  # Full merge
python src/garmin_merge.py --validate-only  # Dry run
python src/garmin_merge.py --since 2025-01-01
```

---

## Database Schema

### Main Tables

**`daily_metrics`** — One row per day (~65+ columns):
- Date (PK), heart rate (resting/max/min), HRV (last_night/5min_high/weekly_avg/status)
- Stress (level + rest/low/medium/high duration), sleep (seconds/deep/light/REM/awake/score/respiration/stress/skin_temp)
- Steps, body battery (charged/drained/peak/low), intensity minutes (moderate/vigorous)
- Training readiness (score + 10 sub-scores), weight, hydration
- Extended: VO2 max, race predictions, ACWR, acclimation status

**`activities`** — One row per activity:
- Activity ID (PK), name, type, timestamps, distance, duration, HR, speed, cadence, calories, elevation
- Extended: training effects (aerobic/anaerobic), training load, VO2 max, stride length, sport type

**`body_battery_events`** — Intraday body battery events:
- Type, start time, duration, impact, feedback type

### Supplementary Tables

| Table | Purpose |
|-------|---------|
| `wellness_log` | Daily subjective wellness questionnaire |
| `nutrition_log` | Nutrition tracking |
| `personal_records` | Personal bests and records |
| `goals` | Health and fitness goals |
| `weekly_summaries` | AI-generated weekly analysis reports |
| `matrix_summaries` | Computed statistical summaries |

---

## Usage Patterns

### Weekly Health Review

```bash
# Run the full pipeline every Sunday
python src/weekly_sync.py

# Check the dashboard
streamlit run src/dashboard.py
```

### Training Decision

```bash
# Quick look at recent metrics
python scripts/raw_analysis.py

# Or query directly
python -c "
import psycopg2, os
from dotenv import load_dotenv; load_dotenv()
conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
cur = conn.cursor()
cur.execute('''
    SELECT date, training_readiness, hrv_last_night, bb_charged, stress_level
    FROM daily_metrics ORDER BY date DESC LIMIT 7
''')
for row in cur.fetchall():
    print(row)
"
```

### Read Latest AI Insights

```bash
python scripts/read_agents.py
```

### Verify AI Claims

```bash
python scripts/verify_insights.py
```

---

## Automation & Deployment

### Option 1: Local Cron (Free)

```bash
# Linux/macOS crontab
0 8 * * 0 cd /path/to/garmin-health-tracker && python src/weekly_sync.py >> sync.log 2>&1

# Windows Task Scheduler
# Action: python src/weekly_sync.py
# Trigger: Weekly, Sunday 8:00 AM
```

**Pros:** Simplest setup, free, full control.
**Cons:** Requires computer to be on at scheduled time.

### Option 2: Cloud Server ($5–10/month)

Deploy on DigitalOcean, AWS, or any VPS:

```bash
ssh your-server
git clone https://github.com/YOU/garmin-health-tracker.git
cd garmin-health-tracker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env

crontab -e
0 8 * * 0 cd ~/garmin-health-tracker && .venv/bin/python src/weekly_sync.py >> sync.log 2>&1
```

### Option 3: GitHub Actions (Free with Hosted DB)

Use a free hosted PostgreSQL (e.g., Supabase, Neon) + GitHub Actions:

1. Push code to GitHub (ensure `.env` is gitignored)
2. Add secrets in GitHub → Settings → Secrets:
   - `POSTGRES_CONNECTION_STRING`
   - `GARMIN_EMAIL`, `GARMIN_PASSWORD`
   - `GOOGLE_API_KEY`
3. Create `.github/workflows/weekly_sync.yml`:

```yaml
name: Weekly Health Sync
on:
  schedule:
    - cron: '0 8 * * 0'  # Every Sunday 8 AM UTC
  workflow_dispatch:       # Manual trigger

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: python src/weekly_sync.py
        env:
          POSTGRES_CONNECTION_STRING: ${{ secrets.POSTGRES_CONNECTION_STRING }}
          GARMIN_EMAIL: ${{ secrets.GARMIN_EMAIL }}
          GARMIN_PASSWORD: ${{ secrets.GARMIN_PASSWORD }}
          GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
```

**Estimated cost:** Supabase (free) + GitHub Actions ≈ $0.12/month.

---

## GitHub Setup

### Initial Push

```bash
cd garmin-health-tracker

# Verify .env is NOT committed
cat .gitignore | grep ".env"

git init
git add .
git commit -m "Initial commit: Garmin Health Intelligence"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/garmin-health-tracker.git
git push -u origin main
```

### Security Checklist

- [ ] `.env` is in `.gitignore`
- [ ] No credentials in committed files
- [ ] `archive/` is gitignored
- [ ] `data/` (CSV exports) is gitignored
- [ ] GitHub secrets configured for Actions (if using)

---

## Real-World Use Cases

### Overtraining Detection

The system identifies overtraining patterns by cross-referencing:
- Declining HRV trend + rising resting HR
- High training load (ACWR > 1.5)
- Dropping body battery recovery
- Rising stress levels during sleep

### Sleep Optimization

Discovers what actually affects your sleep:
- Which training intensities lead to better deep sleep
- Optimal hydration levels for sleep quality
- Stress management impact on REM sleep
- Correlation between training timing and sleep score

### Finding Peak Performance Days

```sql
SELECT date, training_readiness, hrv_last_night, sleep_score,
       bb_charged, stress_level
FROM daily_metrics
WHERE training_readiness > 70
ORDER BY training_readiness DESC;
```

### Goal Tracking

```python
from enhanced_agents import AdvancedHealthAgents

agents = AdvancedHealthAgents(db_manager=None)
result = agents.run_goal_analysis(goal="improve HRV from 40 to 55")
print(result)
```

---

## Costs

| Component | Cost |
|-----------|------|
| Garmin API | Free (your data) |
| PostgreSQL (Heroku Essential-0) | $5/month |
| Google Gemini API (free tier) | Free (up to 15 RPM) |
| GitHub Actions | ~$0.12/month |
| Streamlit Community Cloud | Free |
| **Total** | **~$5/month** |

---

## Troubleshooting

### Authentication Issues

```
MFA Required: Enter the code from your Garmin Connect app when prompted.
Session expired: Delete ~/.garth/ and re-authenticate.
```

### Database Connection

```bash
# Test PostgreSQL connection
psql $POSTGRES_CONNECTION_STRING -c "SELECT 1;"

# Verify tables exist
psql $POSTGRES_CONNECTION_STRING -c "\dt"
```

### Agent Errors

```bash
# Test Gemini API key
python tests/test_gemini.py

# Test agents with real data
python tests/test_agents.py
```

### Empty Data

If metrics show NULL, check:
1. Garmin watch was worn and synced to the phone app
2. Sufficient time has passed for Garmin to process data (up to 24h)
3. Run `python scripts/debug_fields.py` to inspect available API fields

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `crewai` | AI agent orchestration framework (uses Gemini via litellm) |
| `garth` | Garmin Connect API client (OAuth) |
| `psycopg2-binary` | PostgreSQL driver |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `scipy` | Statistical analysis |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard |
| `python-dotenv` | Environment variable management |

---

## License

MIT
