# Garmin Health Intelligence

A personal health analytics system that applies **real statistical methods** — not just averages and charts — to Garmin biometric data, then feeds the results to **9 specialized AI agents** for interpretation.

> **The core idea:** raw wearable data → 8-step statistical pipeline → structured text summary → 9 AI agents that actually understand the math they're reading.

---

## The Math: 8-Step Computation Pipeline

Every weekly run executes the full pipeline from `correlation_engine.py` (~1,300 lines of pure computation, zero AI):

### Layer 0 — Data Ingestion

Load `daily_metrics` from PostgreSQL, auto-discover all numeric columns (~55–65 metrics depending on available data), drop columns with insufficient variance.

### Layer 1 — Correlation Matrices

| Step | Method | What it produces |
|------|--------|-----------------|
| **1a** | Pearson $r$ (same-day) | $N \times N$ correlation matrix across all metric pairs. Each cell: $(r, p)$. Filters: $\|r\| \geq 0.3$, $p < 0.05$ |
| **1b** | Pearson $r$ (lag-1) | Shifted matrix — does yesterday's metric $X_{t-1}$ predict today's metric $Y_t$? Same thresholds. These are the **predictive** correlations |

### Layer 2 — Statistical Depth

| Step | Method | What it produces |
|------|--------|-----------------|
| **2a** | Shapiro-Wilk | Normality test per metric. $W$ statistic + $p$-value. Flags non-normal distributions where Pearson assumptions may not hold |
| **2b** | AR(1) persistence | Autoregressive coefficient $\phi_1$ for each metric: $X_t = \phi_1 X_{t-1} + \varepsilon_t$. High $\phi_1$ → metric has momentum; low → noisy/random |
| **2c** | Z-score anomalies | $z_i = (x_i - \bar{x}) / s$. Any $\|z\| > 2.0$ flagged with date, metric, direction. These are your "unusual days" |
| **2d** | Conditioned AR(1) | $Y_t = \beta_0 + \beta_1 Y_{t-1} + \beta_2 X_t + \varepsilon$, reported with **adjusted $R^2$**. Does adding a second metric improve the autoregressive fit? Filters: adj. $R^2 > 0.10$, $p < 0.05$ |
| **2e** | Markov transition matrices | Discretize each metric into {LOW, MID, HIGH} via terciles. Build $3 \times 3$ transition matrices $P_{ij}$ conditioned on a second metric's state. Apply **KL-divergence** to detect state-dependent dynamics |

### Adaptive Kernel Smoothing (Markov)

Raw transition counts are sparse early on. Smoothing prevents the matrix from being dominated by sampling noise:

$$\alpha = \frac{0.50}{\sqrt{n}}, \quad \text{clamped to } [0.02,\; 0.25]$$

- $n = 27$ (one month) → $\alpha \approx 0.096$ — moderate smoothing
- $n = 365$ (one year) → $\alpha \approx 0.026$ — near-raw counts
- KL-divergence gated: requires $\geq 5$ transitions per conditioning level (`MIN_TRANSITIONS_KL = 5`)

**Confidence tiers** on every Markov result: HIGH ($\geq 100$ transitions), GOOD ($\geq 30$), MODERATE ($\geq 15$), PRELIMINARY ($< 15$).

### Layer 3 — Natural Language Summary

Pure Python `f-string` formatting (no AI). Converts all numeric results into ~2–3 KB of structured text using hardcoded thresholds:
- $|r| > 0.7$ → "strong", $> 0.5$ → "moderate"
- $R^2 > 0.4$ → "strong model", $> 0.2$ → "moderate"
- KL $> 0.10$ → "STRONG divergence", $> 0.05$ → "moderate"

This text becomes the **context window** for all 9 agents — they read the summary, not the raw numbers.

### Multi-Period Benchmarks

The pipeline runs across sliding windows: **7d → 14d → 30d → 60d → 90d → 180d → 365d** (as data permits). Each window gets its own Layer 1–3 output. Benchmark comparison classifies each correlation as:

| Pattern | Definition |
|---------|-----------|
| **ROBUST** | Present across $\geq 3$ windows with consistent sign |
| **STABLE** | Present in longest + one mid-range window |
| **EMERGING** | Only appears in recent windows |
| **FRAGILE** | Appears/disappears inconsistently |

---

## The Agents: 9 Specialists on Gemini 2.5 Flash

Each agent is a [CrewAI](https://github.com/crewAIInc/crewAI) agent backed by `gemini-2.5-flash` (temperature 0.3). They share 4 tools for live database queries and receive the Layer 3 summary + benchmark comparison as context.

| # | Agent | What it does |
|---|-------|-------------|
| 1 | **Matrix Analyst** | Reads the correlation matrices and statistical summaries. Identifies which relationships are real vs. noise |
| 2 | **Benchmark Comparator** | Compares patterns across time windows (7d→365d). Flags what's ROBUST vs. EMERGING |
| 3 | **Pattern Detective** | Discovers hidden or non-obvious relationships in the data |
| 4 | **Performance Optimizer** | Translates statistical findings into training recommendations |
| 5 | **Recovery Specialist** | Analyzes recovery dynamics — body battery recharge, HRV bounce-back, overtraining signals |
| 6 | **Trend Forecaster** | Projects current trends forward, flags concerning directions |
| 7 | **Lifestyle Analyst** | Links lifestyle factors (hydration, stress, steps) to outcomes |
| 8 | **Sleep Analyst** | Deep sleep architecture — duration, deep/REM/light as % of total vs. clinical norms, consistency (CV > 15% = inconsistent), next-day impact on readiness and body battery |
| 9 | **Weakness Identifier** | Synthesizes all findings into the #1 bottleneck + actionable quick wins |

**Agent tools:**
- `run_sql_query` — Execute SQL against the health DB (read-only guard: rejects INSERT/UPDATE/DELETE)
- `calculate_correlation` — Compute Pearson $r$ between any two metrics on the fly
- `find_best_days` — Retrieve days with optimal metric values
- `analyze_pattern` — Detect patterns over configurable time windows

---

## Architecture

```
Garmin Watch → Garmin Connect → Garmin API (garth OAuth)
                                       ↓
                         EnhancedGarminDataFetcher → PostgreSQL (Heroku, UPSERT)
                                                          ↓
                    ┌─────────────────────────────────────┴──────────────┐
                    ↓                                                    ↓
          CorrelationEngine (8 steps)                           garmin_merge.py
          Layer 0 → 1a,1b → 2a–2e → 3                        (bulk enrichment)
          × 7 benchmark windows                                         
                    ↓                                                    
      Layer 3 summary + benchmark comparison                             
                    ↓                                                    
          9 CrewAI Agents (Gemini 2.5 Flash)                             
                    ↓                                                    
          weekly_summaries table → Streamlit Dashboard / CLI
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

See **[The Math: 8-Step Computation Pipeline](#the-math-8-step-computation-pipeline)** above for full details.

~1,300 lines of pure statistical computation. No AI. Results stored in `matrix_summaries` table (1 row/day via UPSERT).

### AI Agents (`src/enhanced_agents.py`)

See **[The Agents: 9 Specialists](#the-agents-9-specialists-on-gemini-25-flash)** above for the full roster and tools.

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
