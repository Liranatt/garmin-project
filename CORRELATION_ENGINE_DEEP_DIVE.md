# Correlation Engine — Deep Dive

> For: Liran  
> Readable by: Humans 🧠, Mathematicians 📐, and Coders 💻

---

## 1. What Does This Engine Do?

**One sentence**: Takes your daily health numbers from `daily_metrics`, runs statistical analysis across multiple time scales, and produces a text summary that tells the AI agents "here's what statistically matters in your body right now."

**The output** is stored in `matrix_summaries.summary_text` — the one you saw from Feb 15. That's the only time it fully succeeded.

---

## 2. Architecture — 4 Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 0: Load & Clean                                         │
│  daily_metrics → DataFrame → fix sentinels → derive metrics    │
│  Output: clean DataFrame + list of valid numeric metrics       │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1: Pearson Correlations                                 │
│  1a: Same-day NxN correlation matrix (r, p-value)              │
│  1b: Lag-1 (yesterday→today) correlations                      │
│  Output: significant pairs (|r| > 0.3, p < 0.05)              │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: Advanced Statistics                                  │
│  2a: Shapiro-Wilk normality tests                              │
│  2b: AR(1) autoregressive persistence                          │
│  2c: Anomaly detection (percentile-based)                      │
│  2d: Conditioned AR(1) (multi-variable prediction)             │
│  2e: Markov state transitions + KL-divergence                  │
│  2f: Rolling correlation stationarity (30-day window)          │
│  2g: Multi-lag carryover (2-3 day delayed effects)             │
│  Output: per-metric models, transition matrices, anomalies     │
└────────────────────────┬────────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 3: Agent Summary                                        │
│  All results → 2-3 KB natural-language text                    │
│  Stored in matrix_summaries → passed to AI agents              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer-by-Layer Explanation

### Layer 0 — Load & Clean

**Code**: `_layer0_load_and_clean()` (lines 741-864)

**What it does**:
1. Loads `SELECT * FROM daily_metrics` (optionally filtered by date)
2. Merges `wellness_log` and `nutrition_log` if they have data (left join on date)
3. Fixes data:
   - HRV sentinel value 511 → NULL (Garmin returns this when not enough baseline)  
   - Drops `hydration_goal_ml`, `sweat_loss_ml` (always garbage)
   - Renames `tr_*` abbreviations to readable names (e.g., `tr_sleep_score` → `training_sleep_score`)
4. **Derives new metrics** (this is important math):
   - `ln_hrv` = ln(HRV) — log-transform because RMSSD is right-skewed (Plews 2012, Esco 2025). Pearson correlation assumes normality, so lnRMSSD is standard in sports science.
   - `hrv_weekly_mean` = 7-day rolling mean of `hrv_last_night`
   - `hrv_weekly_cv` = 7-day rolling coefficient of variation (CV) = (std / mean) × 100. This is a **stronger marker of recovery** than mean alone — high CV = unstable autonomic system.
5. **Fills date gaps** with NaN rows using `asfreq('D')` — critical for `.shift(1)` to work correctly for lag calculations
6. Auto-discovers all numeric columns with ≥5 non-null values → these become the metrics list

**🔑 Key decision**: The engine uses whatever columns have data. If VO2 Max or ACWR are NULL (because bulk import isn't working), they're simply excluded from analysis. **This is why your matrix_summaries from Feb 15 had those columns but recent runs don't**.

---

### Layer 1 — Pearson Correlations

**Code**: `_layer1_pearson()` (lines 868-927)

#### 1a: Same-Day Correlations

For every pair of metrics (A, B), computes:

$$r_{AB} = \frac{\sum (A_i - \bar{A})(B_i - \bar{B})}{\sqrt{\sum(A_i-\bar{A})^2 \cdot \sum(B_i-\bar{B})^2}}$$

Then computes the p-value using t-statistic:

$$t = r \cdot \sqrt{\frac{n-2}{1-r^2}}$$

**Kept if**: |r| > 0.3 AND p < 0.05

**What your data showed (Feb 15)**:
- `sleep_score × training_sleep_score`: r = +0.997 — nearly identical metrics (makes sense, one derives from the other)
- `bb_charged × training_acute_load`: r = -0.972 — **when you train harder, body battery recharges less** (real insight!)
- `resting_hr × bb_charged`: r = -0.961 — **lower resting HR = better body battery recovery**

#### 1b: Lag-1 (Next-Day Predictors)

Same Pearson formula, but between yesterday's metric A and today's metric B:

$$r_{AB}^{lag1} = \text{pearsonr}(A_{t-1}, B_t)$$

Tells you: **what yesterday predicts about today**.

**From your data**: `sleep_score → training_acute_load`: r = +0.989 — better sleep yesterday → higher acute load capacity today.

---

### Layer 2a — Normality Tests

**Code**: `_layer2a_normality()` (lines 931-958)

Uses **Shapiro-Wilk test** on each metric:

$$W = \frac{(\sum a_i x_{(i)})^2}{\sum (x_i - \bar{x})^2}$$

- p > 0.05 → assume **normal** distribution → Pearson is valid
- p ≤ 0.05 → **non-normal** → Markov analysis is preferred (Layer 2e)

**Why it matters**: Pearson assumes linear relationships between normally-distributed variables. For non-normal metrics (like body battery), Markov state-based modeling captures the actual dynamics better.

---

### Layer 2b — AR(1) Persistence

**Code**: `_layer2b_ar1()` (lines 962-1002)

Fits first-order autoregressive model for each metric:

$$x_t = \phi \cdot x_{t-1} + c + \varepsilon_t$$

Where:
- **φ (phi)** = autocorrelation strength (close to 1 = yesterday strongly predicts today)
- **R²** = how much of today's value is explained by yesterday's value
- Includes **ADF stationarity test** — if metric has a trend (non-stationary), it's flagged

**From your data**: 
- `training_load_ratio_pct`: φ = +0.875, R² = 0.652 — **very persistent**, yesterday's load ratio strongly determines today's
- `training_acute_load`: φ = -0.231, R² = 0.496 — **rebounds** (high yesterday → lower today)

**🧠 Human interpretation**: R² > 0.4 = "your yesterday strongly predicts your today for this metric." R² < 0.15 = "this metric resets daily, yesterday doesn't matter much."

---

### Layer 2c — Anomaly Detection

**Code**: `_layer2c_anomalies()` (lines 1006-1036)

For each metric, finds the 5th and 95th percentiles. If your last 3 days include values outside those bounds → flagged as anomaly with z-score.

Uses **percentile-based detection** (not just z-scores) because it works for both normal and skewed distributions.

---

### Layer 2d — Conditioned AR(1)

**Code**: `_layer2d_conditioned_ar1()` (lines 1040-1159)

This is the most sophisticated layer. It asks: **"Can I predict tomorrow's metric better if I also know another variable?"**

**Level 1** — One conditioning variable:

$$x_t = \beta_0 \cdot x_{t-1} + \beta_1 \cdot z_{t-1} + \beta_2 + \varepsilon$$

**Level 2** — Two conditioning variables:

$$x_t = \beta_0 \cdot x_{t-1} + \beta_1 \cdot z_{1,t-1} + \beta_2 \cdot z_{2,t-1} + \beta_3 + \varepsilon$$

Solved via **OLS (ordinary least-squares)** using `np.linalg.lstsq`.

Uses **adjusted R²** to penalize model complexity:

$$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$$

**From your data (Feb 15)**:
- `training_readiness ~ lag + highest_respiration + acwr`: R²_adj = 0.969 — knowing your respiration rate and ACWR nearly perfectly predicts tomorrow's training readiness
- Only n=6 though — **preliminary**!

**🔑 The asterisk warnings matter**: When n < 20, R² values are inflated. This is why the engine prints `*` after small samples.

---

### Layer 2e — Markov State Transitions + KL-Divergence

**Code**: `_layer2e_markov()` (lines 1163-1179) → delegates to `analytics/markov_layer.py`

**Step 1: Discretize** each metric into 3 bins: LOW / MED / HIGH (using terciles)

**Step 2: Build transition matrix** — count how often you go from state A to state B:

$$P_{ij} = \frac{\text{count}(A \rightarrow B) + \alpha/k}{\text{count}(A \rightarrow *) + \alpha}$$

Where α is **adaptive smoothing**: $\alpha = \text{clamp}\left(\frac{0.5}{\sqrt{n}}, 0.02, 0.25\right)$

- Few transitions → heavy smoothing (don't trust sparse data)
- Many transitions → light smoothing (let the data speak)

**Step 3: Stationary distribution** — solve $\pi P = \pi$ to find long-run probabilities.

**Step 4: KL-Divergence conditioning** — for each metric, split data by another metric's state (HIGH vs LOW), build separate transition matrices, and compute:

$$D_{KL}(P_{cond} \| P_{marginal}) = \sum_i \sum_j P_{cond}(i,j) \cdot \ln\frac{P_{cond}(i,j)}{P_{marginal}(i,j)}$$

**What this tells you**: "Does knowing my stress is HIGH change how my body battery transitions between states?" If KL is large → YES, that condition changes the dynamics.

**From your data**:
- `bb_charged` conditioned on `rest_stress_sec`: KL = 0.356 (STRONG) — **when rest stress is HIGH, body battery's LOW→HIGH transition changes by +0.61**. Translation: more rest stress → your BB is more likely to jump from LOW to HIGH (counter-intuitive? Maybe you rest more after stressful days → bigger BB recharge)

**Confidence tiers** (based on transition count):
- HIGH: ≥100 transitions
- GOOD: ≥30
- MODERATE: ≥15
- PRELIMINARY: <15

With only 28 days of data, most are PRELIMINARY.

---

### Layer 2f — Rolling Correlation Stationarity

**Code**: `_layer2f_rolling_correlation()` (lines 1225-1278)

Takes the top-10 significant same-day pairs and checks: **is the correlation stable over time?**

Uses a **30-day rolling window** of Pearson-r values:
- std(rolling-r) < 0.15 → STABLE (reliable relationship)
- std(rolling-r) < 0.25 → MODERATE
- std(rolling-r) ≥ 0.25 → UNSTABLE (may be driven by confound or lifestyle change)

**Currently**: You need ≥35 days of data for this to work (30-day window + 5 minimum). With 28 days (Feb 15), no results are produced.

---

### Layer 2g — Multi-Lag Carryover

**Code**: `_multi_lag_carryover()` (lines 1183-1221)

Checks if a predictor→target relationship persists at **lag-2 and lag-3** (2-3 days later).

Based on **Daza (2018)** — carryover effects in N-of-1 designs.

Example: if a hard workout still shows up in your HRV **3 days later**, that's a carryover effect.

---

### Layer 3 — Agent Summary

**Code**: `_layer3_summary()` (lines 1282-1502)

Takes ALL outputs from Layers 0-2 and builds a **structured text summary** with sections:
1. Training Intensity Classification
2. Same-Day Correlations (top 12)
3. Next-Day Predictors (top 8)
4. Autoregressive Persistence (top 8)
5. Distributions (normal vs non-normal)
6. Recent Anomalies
7. Multi-Day Carryover
8. Rolling Correlation Stability
9. Conditioned AR(1) (top 10)
10. Markov State Transitions (top 8)
11. KL-Divergence (top 10)
12. Current Metric Ranges

This ~2-3 KB text is what goes into `matrix_summaries.summary_text` and gets passed to the AI agents.

---

## 4. Entry Points — How The Engine Is Called

There are **3 ways** to call the engine:

| Method | What it does | When used |
|---|---|---|
| `compute_weekly()` | Runs on ALL data, stores to DB | Daily pipeline |
| `compute_benchmarks()` | Runs on 7d, 14d, 30d, 60d, 90d, 6mo, 1yr windows | Pipeline (benchmark mode) |
| `compute_multi_period()` | Week1 vs Week2 vs Combined 2-week | Legacy, not currently used |

The **daily pipeline** calls `compute_benchmarks()`, which:
1. Checks what date range exists in `daily_metrics`
2. Builds candidate windows (skips windows with < 5 days)
3. Runs `_compute_raw_with_status()` for each window
4. Stores the **longest** period's summary to `matrix_summaries`
5. Builds a cross-timeframe comparison text

---

## 5. What's Stored vs What's Computed In-Memory

| What | Stored in DB? | Details |
|---|---|---|
| Text summary | ✅ `matrix_summaries.summary_text` | ~2-3 KB natural language |
| Token estimate | ✅ `matrix_summaries.token_est` | For LLM cost tracking |
| Computed date | ✅ `matrix_summaries.computed_at` | One per day, UPSERT |
| Raw Pearson matrix (NxN) | ❌ | Computed in-memory each run |
| Lag-1 results | ❌ | Computed in-memory |
| AR(1) coefficients | ❌ | Computed in-memory |
| Markov transition matrices | ❌ | Computed in-memory |
| KL-divergence values | ❌ | Computed in-memory |
| Anomaly list | ❌ | Computed in-memory |
| Conditioned AR(1) models | ❌ | Computed in-memory |

**Your question**: "Don't we want to store raw matrices to verify the math?"

**Yes.** Storing at least the key raw outputs (Pearson pairs + coefficients, AR(1) φ values, Markov transition matrices) would let us:
- Verify computations are correct
- Track statistical evolution over time
- Debug when things look wrong
- Build trend-over-time visualizations on the frontend

See **Section 7** for the proposed schema.

---

## 6. Why It Stopped Working After Feb 15

**The pipeline flow:**
```
weekly_pipeline._run_correlations()
  → CorrelationEngine(conn_str).compute_benchmarks(training_days)
```

If `compute_benchmarks()` returns `analysis_status = "failed"` or `longest = None`:
```python
matrix_ctx = benchmarks.get(longest, "") if longest else ""
# → empty string!
```

Then `_analyze_week(corr_result)` passes this empty string to the agents:
```python
result = agents.run_weekly_summary(
    matrix_context=matrix_ctx,       # ← EMPTY STRING
    comparison_context="",            # ← EMPTY STRING
)
```

The **Interpreter** agent specifically needs this data. Without it, it says "I require the pre-computed correlation analysis data." The other agents (Pattern, Performance, Sleep, Synthesizer) work fine because they use SQL tools to query the DB directly.

**Most likely cause**: A computation failure in one of the layers (possibly Markov due to data shape changes, or an import error in `analytics/markov_layer.py`). The error gets caught, status = "degraded" or "failed", longest = None, and no summary gets stored.

---

## 7. Proposed: Storing Raw Matrices in DB

New table to add:

```sql
CREATE TABLE IF NOT EXISTS correlation_results (
    id              SERIAL PRIMARY KEY,
    computed_at     DATE NOT NULL,
    period_key      TEXT NOT NULL,              -- 'weekly', '7d', '14d', '30d', etc.
    date_start      DATE NOT NULL,
    date_end        DATE NOT NULL,
    n_days          INTEGER NOT NULL,
    n_metrics       INTEGER NOT NULL,
    
    -- Layer 1 results
    sig_pairs_json  JSONB,                      -- [{a, b, r, p}, ...]
    lag1_pairs_json JSONB,                      -- [{pred, tgt, r, p, n}, ...]
    
    -- Layer 2 results  
    ar1_json        JSONB,                      -- [{metric, phi, r2, p, n, stationary}, ...]
    normality_json  JSONB,                      -- {metric: [status, p_value]}
    anomalies_json  JSONB,                      -- [{date, metric, value, z, direction}]
    cond_ar1_json   JSONB,                      -- [{target, conditions, r2, adj_r2, improvement, n}]
    markov_json     JSONB,                      -- [{target, transitions, edges, marginal, ...}]
    kl_json         JSONB,                      -- [{target, condition, kl_value}]
    
    -- Meta
    analysis_status TEXT NOT NULL DEFAULT 'success',
    degraded_reasons TEXT[],
    summary_text    TEXT,
    token_est       INTEGER,
    
    UNIQUE(computed_at, period_key)
);
```

This would let you run queries like:
```sql
-- Track how a specific correlation evolves over time
SELECT computed_at, elem->>'r' as r_value
FROM correlation_results, 
     jsonb_array_elements(sig_pairs_json) elem
WHERE elem->>'a' = 'resting_hr' AND elem->>'b' = 'bb_charged'
ORDER BY computed_at;
```

---

## 8. Academic References Used

| Citation | Used For | Layer |
|---|---|---|
| Plews 2012 | ln(RMSSD) transformation, weekly HRV CV | 0 |
| Esco 2025 | HRV normalization best practices | 0 |
| Gabbett 2016 (BJSM) | ACWR thresholds (0.8–1.3 sweet spot) | 2e |
| Daza 2018 | Multi-lag carryover in N-of-1 designs | 2g |
| Shapiro-Wilk (1965) | Normality testing | 2a |
| Augmented Dickey-Fuller | Stationarity testing for AR(1) | 2b |

---

## 9. Key Code Locations

| Component | File | Lines |
|---|---|---|
| Constants & config | `src/correlation_engine.py` | 56-165 |
| Schema DDL | `src/correlation_engine.py` | 172-179 |
| Engine class | `src/correlation_engine.py` | 186-1522 |
| Markov layer | `src/analytics/markov_layer.py` | (separate module) |
| Pipeline integration | `src/pipeline/weekly_pipeline.py` | 163-176 |
| Storage (`_store_summary`) | `src/correlation_engine.py` | 1506-1522 |

---

## 10. What Needs Fixing

1. **Store raw matrices** — not just text summary (Section 7 above)
2. **Debug why it stopped after Feb 15** — likely a Layer 2e failure or import error
3. **Interpreter agent needs the context** — it's the only agent that fails when correlation data is missing
4. **More data = better results** — with only 18 days (Feb 10-27), most results are PRELIMINARY. After 30+ days, the engine becomes much more useful. After 90+ days, the rolling correlation and multi-lag analysis start working.
