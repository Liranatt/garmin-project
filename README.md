# Garmin Health Intelligence

A personal health analytics system that applies **real statistical methods** to Garmin wearable data, then feeds the results to **9 specialized AI agents** for interpretation humans can act on.

> **Core idea:** raw wearable data → 8-step statistical pipeline → structured text summary → 9 AI agents that actually understand the math they're reading.

---

## The Problem

Garmin Connect gives you charts and numbers, but it doesn't tell you **why** your training readiness dropped, **whether** the pattern is real or noise, or **what** actually predicts your good days vs. bad days. Specifically:

1. **No cross-metric analysis.** Garmin shows each metric in isolation. It never asks: "does your HRV predict tomorrow's training readiness?" or "does high stress today predict poor sleep tonight?"

2. **No statistical rigor.** When Garmin says a trend is "improving", it's comparing two averages with no significance test, no confidence interval, no control for sample size. With 7 data points, almost anything looks like a trend.

3. **No temporal dynamics.** Lag-1 effects (yesterday→today), autoregressive persistence (does a metric have momentum?), and state-dependent transitions (does being in a "low HRV" state make it likely you stay there?) are invisible in Garmin's UI.

4. **Date gaps corrupt the math.** If you don't wear the watch for 3 days, Garmin stores nothing for those days. A naive `.shift(1)` then pairs Monday's data with Friday's data as if they were consecutive — producing **silently wrong** correlations, AR(1) fits, and Markov transitions. This is not a Garmin bug; it's a fundamental data-processing pitfall that most wearable analytics tools ignore.

5. **No actionable synthesis.** Even if you could see all the statistics, you'd still need someone (or something) to read them and say: "your #1 bottleneck right now is poor deep sleep, and here's what to do about it."

---

## How I Tackled It

### Architecture

```
Garmin Watch → Garmin Connect → Garmin API (garth OAuth)
                                       ↓
                      EnhancedGarminDataFetcher → PostgreSQL (Heroku, UPSERT)
                         ↑ tenacity retry ×3                    ↓
                    ┌─────────────────────────────────────┴──────────────┐
                    ↓                                                    ↓
          CorrelationEngine (8 steps)                           garmin_merge.py
          Layer 0 → 1a,1b → 2a–2e → 3                        (bulk enrichment)
          × 9 benchmark windows
                    ↓
      Layer 3 summary + benchmark comparison
                    ↓
          9 CrewAI Agents (Gemini 2.5 Flash)
          3 batches × 3 threads (parallel)
                    ↓
          weekly_summaries table → Streamlit Dashboard / CLI
```

### Layer-by-layer approach

Instead of throwing all the data at an LLM and hoping for the best, the system separates **computation** from **interpretation**:

- **Layers 0–3** (`correlation_engine.py`, ~1,400 lines of pure Python + numpy + scipy) do all the math. No AI.  Zero LLM calls. Every number is reproducible and testable.
- **Layer 4** (9 CrewAI agents on Gemini 2.5 Flash) reads the structured text summary produced by Layer 3 and interprets it with domain knowledge and live SQL access to the raw data.

This means the AI agents never hallucinate the statistics — they're reading pre-computed, validated results. They can only hallucinate the **interpretation**, and even that is anchored to real numbers.

### The date-gap fix

The hardest bug to find. `.shift(1)` assumes consecutive rows = consecutive days, but Garmin data has gaps (watch off, travel, charging). The fix has two parts:

1. **In Layer 0:** after loading from PostgreSQL, enforce daily continuity with `.asfreq('D')`. This inserts NaN placeholder rows for every missing date, so `.shift(1)` on Feb 6 correctly sees NaN (from the missing Feb 5), not Feb 2's real value.

2. **In Markov transitions:** even with gap-fill, Markov transition counting must check `|date_{t+1} − date_t| = 1 day` before incrementing the transition matrix. Values separated by >1 day are **not** state transitions — they're disconnected observations.

### Retry logic

Garmin's API occasionally drops connections mid-sync. All 9 API endpoints are now wrapped with `tenacity`:
- 3 attempts per call
- Exponential backoff: 2s → 4s → 8s (capped at 30s)
- Only retries transient errors (ConnectionError, TimeoutError, OSError)
- Non-transient errors (404, auth failure) raise immediately

---

## The Solution: 8-Step Computation Pipeline

Every weekly run executes the full pipeline from `correlation_engine.py`:

### Layer 0 — Data Ingestion

Load `daily_metrics` from PostgreSQL, auto-discover all numeric columns (~55–65 metrics depending on available data), drop columns with insufficient variance. **Enforce daily continuity** with `.asfreq('D')` gap-fill.

### Layer 1 — Correlation Matrices

| Step | Method | What it produces |
|------|--------|-----------------|
| **1a** | Pearson *r* (same-day) | N×N correlation matrix across all metric pairs. Each cell: (*r*, *p*). Filters: \|*r*\| ≥ 0.3, *p* < 0.05 |
| **1b** | Pearson *r* (lag-1) | Shifted matrix — does yesterday's metric X predict today's metric Y? Same thresholds. These are the **predictive** correlations |

### Layer 2 — Statistical Depth

| Step | Method | What it produces |
|------|--------|-----------------|
| **2a** | Shapiro-Wilk | Normality test per metric. W statistic: $W = (\sum a_i x_{(i)})^2 \;/\; \sum (x_i - \bar{x})^2$. Flags non-normal distributions where Pearson assumptions weaken |
| **2b** | AR(1) persistence | Autoregressive coefficient: $x_t = \varphi \cdot x_{t-1} + c + \varepsilon_t$. High φ → metric has momentum; low → noisy/random. Date-gap safe via `.asfreq('D')` |
| **2c** | Z-score anomalies | $z = (x - \mu) / \sigma$. Any \|z\| > 1.5 flagged with date, metric, direction (HIGH/LOW). The 1.5σ threshold captures the outer ~13% of a normal distribution |
| **2d** | Conditioned AR(1) | Multivariate OLS: $x_t = \beta_0 x_{t-1} + \beta_1 z_{t-1} + \beta_2 + \varepsilon$, reported with **adjusted R²**: $R^2_{adj} = 1 - [(1-R^2)(n-1)]/(n-p-1)$. Two-level forward search: Level 1 (single predictor) → Level 2 (pairs from top-5 Level-1 winners) |
| **2e** | Markov transition matrices | See detailed breakdown below |

### Layer 2e — Markov Analysis (detailed)

Each target metric goes through a 6-step Markov pipeline:

1. **Discretization** — Values binned into 5 quintile states (VERY_LOW, LOW, MED, HIGH, VERY_HIGH) via percentile edges [0, 20, 40, 60, 80, 100]. Upgraded from 3 bins (terciles) to 5 for higher resolution.

2. **Marginal transition matrix** — $T[i,j] = P(\text{state}_j \mid \text{state}_i)$. Only transitions between truly consecutive days are counted (gap-aware: checks $|date_{t+1} - date_t| = 1$).

3. **Adaptive kernel smoothing** — Regularises sparse rows:

$$T' = (1-\alpha) \cdot T + \alpha \cdot U$$

   where $U$ is uniform($1/N$) and $\alpha = \text{clamp}(\text{BASE}/\sqrt{n},\; \text{MIN},\; \text{MAX})$ decreases with more data:

| Transitions | α | Effect |
|-------------|---|--------|
| 4 | 0.25 | Heavy smoothing (sparse data) |
| 27 (1 month) | 0.096 | Moderate |
| 200 | 0.035 | Light (data speaks) |
| 1000+ | 0.02 | Near-raw counts |

4. **Stationary distribution** — $\pi$ from the left eigenvector of $T'$ corresponding to eigenvalue 1: $\pi T' = \pi$, $\sum \pi_i = 1$.

5. **Conditional split** — For each candidate conditioning metric $c$, population is split by median into two groups. Each group gets its own transition matrix.

6. **KL divergence** — Measures how much the conditional transition matrix differs from the marginal:

$$D_{KL}(P \| Q) = \sum_j P[i,j] \cdot \ln\!\left(\frac{P[i,j]}{Q[i,j]}\right)$$

   Averaged over rows and both conditioning levels. Higher KL → the conditioning metric genuinely changes the day-to-day dynamics. Gated by `MIN_TRANSITIONS_KL = 8` per conditioning level (raised from 5 to compensate for 5-bin sparsity).

**Confidence tiers** on every Markov result: HIGH (≥100 transitions), GOOD (≥30), MODERATE (≥15), PRELIMINARY (<15).

### Layer 3 — Natural Language Summary

Pure Python `f-string` formatting (no AI). Converts all numeric results into ~2–3 KB of structured text using hardcoded thresholds:
- |*r*| > 0.7 → "strong", > 0.5 → "moderate"
- R² > 0.4 → "strong model", > 0.2 → "moderate"
- KL > 0.10 → "STRONG divergence", > 0.05 → "moderate"

This text becomes the **context window** for all 9 agents.

### Multi-Period Benchmarks

The pipeline runs across sliding windows: **current week → 7d → 14d → 21d → 30d → 60d → 90d → 180d → 365d** (as data permits). Benchmark comparison classifies each correlation as:

| Pattern | Definition |
|---------|-----------|
| **ROBUST** | Present across ≥3 windows with consistent sign |
| **STABLE** | Present in longest + one mid-range window |
| **EMERGING** | Only appears in recent windows |
| **FRAGILE** | Appears/disappears inconsistently |

---

## The Agents: 9 Specialists on Gemini 2.5 Flash

Each agent is a [CrewAI](https://github.com/crewAIInc/crewAI) agent backed by `gemini-2.5-flash` (temperature 0.3). They share 4 tools for live database queries and receive the Layer 3 summary + benchmark comparison as context.

| # | Agent | Role |
|---|-------|------|
| 1 | **Matrix Analyst** | Reads correlation matrices and statistical summaries. Separates real signal from noise |
| 2 | **Benchmark Comparator** | Compares patterns across time windows (7d→365d). Flags ROBUST vs. EMERGING |
| 3 | **Pattern Detective** | Discovers hidden or non-obvious relationships (day-of-week effects, delayed correlations) |
| 4 | **Performance Optimizer** | Translates statistical findings into training recommendations |
| 5 | **Recovery Specialist** | Analyzes body battery recharge, HRV bounce-back, overtraining signals |
| 6 | **Trend Forecaster** | Projects current trends forward, flags concerning directions |
| 7 | **Lifestyle Analyst** | Links lifestyle factors (hydration, stress, steps) to outcomes |
| 8 | **Sleep Analyst** | Deep/REM/light as % of total vs. clinical norms, consistency (CV > 15% = inconsistent), next-day impact |
| 9 | **Weakness Identifier** | Synthesizes all findings into the #1 bottleneck + 3 actionable quick wins |

**Parallel execution:** Agents run in 3 batches of 3 threads each (~30s total instead of ~90s sequential). Total cost per full run: **~$0.02** on Gemini 2.5 Flash.

**Agent tools:**
- `run_sql_query` — Execute read-only SQL against the health DB (rejects INSERT/UPDATE/DELETE/DROP)
- `calculate_correlation` — Compute Pearson *r* between any two metrics on the fly
- `find_best_days` — Retrieve days with optimal metric values
- `analyze_pattern` — Detect patterns and compute trend/volatility over configurable windows

---

## Database

**PostgreSQL on Heroku** (Essential-0, $5/month). All writes use UPSERT — idempotent, safe to re-run.

### Main Tables

| Table | Rows | Description |
|-------|------|-------------|
| `daily_metrics` | 1/day, ~65 columns | All biometric data: HR, HRV, stress, sleep architecture, body battery, steps, training readiness (10 sub-scores), weight, hydration |
| `activities` | 1/workout | Name, type, duration, distance, HR, cadence, calories, elevation, training effects (aerobic/anaerobic), VO2 max |
| `body_battery_events` | N/day | Intraday body battery changes with event type, impact, and feedback |
| `weekly_summaries` | 1/week | AI-generated weekly analysis reports |
| `matrix_summaries` | 1/day | Computed Layer 3 statistical summaries |

### Data Quality Rules (baked into fetcher)

| Rule | Why |
|------|-----|
| HRV weekly avg = 511 → NULL | Garmin sentinel for "not enough baseline data" |
| Hydration = 0 mL → NULL | User never logged → don't let agents analyze fabricated zeros |
| Training readiness: only AFTER_WAKEUP_RESET entries | Other entries are partial/stale |
| `.asfreq('D')` gap-fill in correlation engine | Prevents false lag-1 pairs from date gaps |
| Markov consecutive-day check | Only count transitions between truly consecutive days |

---

## Automation: GitHub Actions

The weekly pipeline runs automatically every Sunday at 06:00 UTC via `.github/workflows/weekly_sync.yml`:

```yaml
on:
  schedule:
    - cron: '0 6 * * 0'    # Every Sunday
  workflow_dispatch:         # Manual trigger from GitHub UI
```

**What it does:**
1. Checks out the repo
2. Installs Python 3.12 + dependencies
3. Restores garth OAuth session from GitHub Secrets
4. Runs `python weekly_sync.py --days 7` (fetch + correlate + 9 agents)
5. Saves updated garth session back

**Required GitHub Secrets:**
- `POSTGRES_CONNECTION_STRING` — Heroku Postgres URL
- `GARMIN_EMAIL` / `GARMIN_PASSWORD` — Garmin Connect credentials
- `GOOGLE_API_KEY` — Gemini API key
- `GARTH_SESSION` / `GARTH_SESSION_OAUTH2` — Base64-encoded OAuth tokens

---

## Dashboard

Interactive Streamlit web app with glass-morphism UI (dark theme, green accent).

| Page | What it shows |
|------|--------------|
| **Overview** | Hero metric tiles (RHR, HRV, Sleep, Stress) with week-over-week arrows, recovery bars, 7-day trend chart, recent activities |
| **Trends** | 3 tabs: Recovery (HRV vs RHR dual-axis, stress vs body battery, readiness bars), Training (weekly volume stacked by type, HR bubble chart, training effects), Sleep (architecture stacked bars, score trend) |
| **Deep Dive** | Single-metric analysis: stats tiles, time series + 7d moving average, distribution histogram |
| **AI Chat** | Free-form chat interface backed by a CrewAI agent with live SQL access |
| **AI Analysis** | Run all 9 agents in parallel (3 batches × 3 threads). Results rendered as color-coded insight cards |
| **Goals** | 8-week trend sparklines for 7 key metrics with clinical benchmarks |

---

## Testing

**49 tests**, all passing (`pytest tests/ -v`):

| File | Tests | Coverage |
|------|-------|----------|
| `test_correlation_math.py` | 22 | `_adaptive_alpha` boundary values + monotonicity, constants (5 bins, 8 min transitions), date-gap `.asfreq('D')` fix, Pearson lag-1, AR(1) φ recovery, z-score anomaly detection, Markov row-sum/stationary/KL/quintile, consecutive-day skip |
| `test_fetcher.py` | 18 | `deep_vars` (flat/nested/lists/prefix), `_safe_val` (NaN→None, NaT→None, numpy→Python), `_apply_quality_fixes` (HRV 511, hydration 0) |
| `test_agents_tools.py` | 9 | SQL injection guard (INSERT/DROP/DELETE blocked), `run_sql_query` output format, `calculate_correlation` strong/insufficient, `analyze_pattern` stats/insufficient. All DB calls mocked — no Postgres required |

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_correlation_math.py -v
```

---

## Project Structure

```
garmin_project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── correlation_engine.py    # 8-step statistical pipeline (~1,400 lines)
│   ├── enhanced_agents.py       # 9 CrewAI agents + 4 tools (~1,200 lines)
│   ├── enhanced_fetcher.py      # Garmin API fetcher with tenacity retry (~860 lines)
│   ├── dashboard.py             # Streamlit v2 glass-morphism UI (~1,300 lines)
│   ├── weekly_sync.py           # Main automation: fetch → correlate → agents
│   ├── enhanced_schema.py       # Extended database schema
│   ├── garmin_health_tracker.py # Config, DatabaseManager, core classes
│   ├── garmin_merge.py          # Bulk merge from Garmin DB export
│   └── visualizations.py        # Plotly chart generation library
│
├── tests/
│   ├── test_correlation_math.py # 22 tests — math pipeline
│   ├── test_fetcher.py          # 18 tests — data utilities
│   └── test_agents_tools.py     # 9 tests — agent tools (mocked DB)
│
├── .github/workflows/
│   └── weekly_sync.yml          # Sunday cron automation
│
└── .streamlit/
    └── config.toml              # Dark theme, #00f5a0 accent
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- PostgreSQL 14+ (local or Heroku)
- Garmin Connect account with watch syncing
- Google AI API key (Gemini — free tier works)

### 1. Clone & Install

```bash
git clone https://github.com/Liranatt/garmin-project.git
cd garmin-project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
GARMIN_EMAIL=your@email.com
GARMIN_PASSWORD=your_password
POSTGRES_CONNECTION_STRING=postgresql://user:pass@host:5432/db
GOOGLE_API_KEY=your_gemini_api_key
GARTH_HOME=~/.garth
```

### 3. First Run

```bash
# Full pipeline: fetch → compute correlations → run 9 AI agents
python src/weekly_sync.py

# Or step by step:
python src/weekly_sync.py --fetch     # Fetch only
python src/weekly_sync.py --analyze   # Analyze only (skip fetch)
python src/weekly_sync.py --days 14   # Custom date range
```

### 4. Dashboard

```bash
streamlit run src/dashboard.py
```

---

## Costs

| Component | Cost |
|-----------|------|
| Garmin API | Free |
| PostgreSQL (Heroku Essential-0) | $5/month |
| Gemini 2.5 Flash (9 agents/week) | ~$0.08/month |
| GitHub Actions | ~$0.12/month |
| Streamlit Community Cloud | Free |
| **Total** | **~$5.20/month** |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `crewai[google-genai]` | AI agent orchestration (uses Gemini via litellm) |
| `garth` | Garmin Connect API client (OAuth) |
| `psycopg2-binary` | PostgreSQL driver |
| `pandas` / `numpy` | Data manipulation |
| `scipy` | Statistical analysis (Shapiro-Wilk, OLS) |
| `plotly` | Interactive charts |
| `streamlit` | Web dashboard |
| `tenacity` | Retry logic with exponential backoff |
| `pytest` | Test framework |
| `python-dotenv` | Environment variable management |

---

## License

MIT
