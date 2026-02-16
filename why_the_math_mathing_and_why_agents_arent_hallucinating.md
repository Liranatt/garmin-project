# Why the Math Is Mathing & Why Agents Aren't Hallucinating 🧠📐

This document deconstructs — to the last line of code — the mathematical and algorithmic infrastructure of the **Garmin Health Intelligence** system.

The core engineering philosophy is **Determinism before Semantics**: every number the AI agents cite was computed by a deterministic mathematical engine *before* any LLM was ever invoked. The agents cannot invent correlations, fabricate statistics, or hallucinate trends — they can only *interpret* pre-validated facts.

The system is split into two strictly separated components:

| Component | Role | Can it lie? |
|---|---|---|
| **Mathematical Engine** (`correlation_engine.py`, 1 640 lines) | Computes statistical truth from raw sensor data | No — deterministic math |
| **AI Agents** (`enhanced_agents.py`, 1 038 lines) | Interprets meaning and gives recommendations | Constrained — read-only access to pre-computed results |

> [!IMPORTANT]
> Inspired by N-of-1 experimental design principles (Daza 2018), Deep Learning regularisation techniques (Prof. Oren Freifeld, Intro to Deep Learning, Ben-Gurion University), and sports science workload models (Gabbett 2016).

---

## Part I: The Mathematical Engine

The `CorrelationEngine` class in [`correlation_engine.py`](src/correlation_engine.py) operates as a deterministic **4-layer pipeline**. Each layer feeds the next. No AI is involved until Layer 3 produces the final text summary.

```
Layer 0 → Data Loading & Cleaning (auto-discovery, sentinels, derived metrics)
Layer 1 → Pearson Correlations (same-day NxN + lag-1 next-day predictors)
Layer 2 → Conditional Analysis (6 sub-layers: normality, AR(1), anomalies,
          conditioned AR(1), Markov + KL divergence, rolling stability)
Layer 3 → Natural-Language Summary (≈2-3 KB text digest for agent consumption)
```

---

### 1. Pearson Correlation Coefficient (*r*)

#### What it is (for someone with no math background)

Imagine you plotted two things against each other — say, your stress level and your sleep score — on a scatter plot. If high stress *always* comes with low sleep, the dots form a tight downward line. The **Pearson correlation coefficient** (denoted *r*) is a number between **−1** and **+1** that measures how tightly those dots form a line:

| *r* value | Meaning |
|---|---|
| *r* = +1.0 | Perfect positive line — when one goes up, the other **always** goes up |
| *r* = +0.7 | Strong positive — they tend to move together |
| *r* = 0.0 | No linear relationship at all |
| *r* = −0.7 | Strong negative — when one goes up, the other tends to go **down** |
| *r* = −1.0 | Perfect inverse line |

#### Formal definition

For two variables *X* and *Y* measured over *n* days:

$$r = \frac{\sum_{i=1}^{n}(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n}(X_i - \bar{X})^2 \cdot \sum_{i=1}^{n}(Y_i - \bar{Y})^2}}$$

Where $\bar{X}$ and $\bar{Y}$ are the means (averages) of each variable.

**In words:** for each day, ask "how far is today's stress from the average stress?" and "how far is today's sleep from the average sleep?" Multiply those two deviations together. If they're *both* high on the same days (or *both* low), the product is large and positive → positive *r*. If one is high when the other is low, the product is negative → negative *r*.

#### What is a *p*-value?

The *p*-value answers: **"If there were truly NO relationship between these two metrics, how likely would I be to see a correlation this strong just by random chance?"**

- *p* < 0.05 → "There's less than a 5% chance this is a coincidence." We call this **statistically significant**.
- *p* < 0.01 → Stronger evidence (< 1% chance of coincidence).
- *p* = 0.50 → Basically a coin flip — the correlation is probably noise.

#### How the code computes it

**Same-day correlations** (Layer 1a in `_layer1_pearson`, line 782):

The code computes the *N×N* Pearson matrix across all auto-discovered numeric metrics. For each pair, it computes the *p*-value using a *t*-test:

$$t = \frac{r \cdot \sqrt{n - 2}}{\sqrt{1 - r^2}}$$

This *t*-statistic follows a Student's *t*-distribution with *n − 2* degrees of freedom. The code only keeps pairs where **|*r*| > 0.3 AND *p* < 0.05** — filtering out both weak and statistically insignificant correlations.

**Next-day predictors (lag-1)** (Layer 1b, line 819):

The same Pearson formula, but shifted by one day: "Does *yesterday's* value of metric A predict *today's* value of metric B?" The code shifts the entire dataset by 1 row (`data.shift(1)`) and runs Pearson on the shifted predictor vs the current target. Same thresholds: |*r*| > 0.3 and *p* < 0.05.

> [!NOTE]
> **Constant-input guard**: Before calling `pearsonr`, the code checks `np.std(x) < 1e-10`. If a metric is constant (e.g., all zeros), Pearson is undefined. The code skips it silently rather than producing a warning.

---

### 2. Log-Transform of HRV (lnRMSSD)

#### The problem

Heart Rate Variability (HRV) as measured by Garmin is the RMSSD (Root Mean Square of Successive Differences between heartbeats). RMSSD values are **right-skewed** — most days cluster around 40-70 ms, but occasional nights produce values of 100+ ms. Pearson correlation assumes the data is roughly **normally distributed** (bell-shaped). Applying Pearson to skewed data can inflate or deflate correlations.

#### The solution (Plews 2012, Esco 2025)

The established practice in sports-science literature is to use **lnRMSSD** — the natural logarithm of the HRV value:

$$\text{lnHRV} = \ln(\text{HRV})$$

The logarithm compresses large values and stretches small ones, pulling the distribution closer to a bell curve. After this transform, Pearson correlations become meaningful.

**Code:** `_layer0_load_and_clean`, line 750 — `df["ln_hrv"] = np.log(df[hrv_col].clip(lower=1))`

---

### 3. HRV Weekly Coefficient of Variation (CV)

#### What it is

A single night's HRV can vary wildly. A *mean* over a rolling 7-day window is more stable. But even more informative is the **Coefficient of Variation**:

$$\text{CV} = \frac{\sigma}{\mu} \times 100\%$$

Where $\sigma$ is the standard deviation and $\mu$ is the mean of HRV over the past 7 days.

**What it tells you:** If your *mean* HRV is 60 ms but your CV is 40%, your HRV is bouncing wildly day-to-day — a sign of inconsistent recovery or lifestyle stressors. A CV < 10% means your nervous system is stable.

**Code:** `_layer0_load_and_clean`, lines 756–762 — computes `hrv_weekly_mean` and `hrv_weekly_cv` using a rolling 7-day window with `min_periods=3`.

**Reference:** Plews et al. (2012) showed that the CV of lnRMSSD is a stronger marker of training adaptation than the mean alone.

---

### 4. Shapiro-Wilk Normality Test

#### What is a "normal distribution"?

The normal (Gaussian) distribution is the classic bell curve — most values cluster near the mean, with symmetric tails. Many statistical tests (like Pearson) assume data is approximately normal. If the data is heavily skewed (e.g., sleep duration: most nights 7h, but occasional 3h nights), linear methods can be misleading.

#### The Shapiro-Wilk test

The Shapiro-Wilk test asks: **"Does this data look like it came from a bell curve?"**

It produces a *W* statistic:

$$W = \frac{\left(\sum_{i=1}^{n} a_i \cdot x_{(i)}\right)^2}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

Where $x_{(i)}$ are the sorted values and $a_i$ are pre-computed weights based on what a normal distribution *should* look like.

- *p* > 0.05 → The data is consistent with a normal distribution.
- *p* ≤ 0.05 → The data is **non-normal** (skewed, multi-modal, or heavy-tailed).

**Why it matters in this system:** Non-normal metrics are flagged as preferred candidates for **Markov state-based modelling** (Section 7) instead of linear regression, because their distributions benefit from discretisation into states.

**Code:** `_layer2a_normality`, line 845 — runs `scipy.stats.shapiro()` on each metric.

---

### 5. Autoregressive Models — AR(1) & Stationarity

#### What is AR(1)?

"Autoregressive" means **a variable predicts itself**. The AR(1) model says: *today's value is a linear function of yesterday's value, plus noise*:

$$X_t = c + \phi \cdot X_{t-1} + \epsilon_t$$

| Symbol | Meaning |
|---|---|
| $X_t$ | Today's value (e.g., today's HRV) |
| $X_{t-1}$ | Yesterday's value |
| $\phi$ | **Persistence coefficient** — how much of yesterday carries over |
| $c$ | Intercept (baseline) |
| $\epsilon_t$ | Random noise (what yesterday can't explain) |

**Interpretation of $\phi$:**

- $\phi$ ≈ 0.9 → Strong persistence. Today's HRV is 90% determined by yesterday's. The metric is "sticky" — it changes slowly. Small interventions won't move the needle quickly.
- $\phi$ ≈ 0.3 → Weak persistence. Today's value is mostly independent of yesterday's. The metric resets daily — interventions have immediate impact.
- $\phi$ ≈ 0.0 → No persistence. Knowing yesterday tells you nothing about today.

**What is *R*² ?**

*R*² (R-squared, "coefficient of determination") answers: **"What fraction of today's variation can be explained by yesterday's value?"**

$$R^2 = 1 - \frac{\sum(X_t - \hat{X}_t)^2}{\sum(X_t - \bar{X})^2}$$

Where $\hat{X}_t$ is the predicted value from the model and $\bar{X}$ is the overall mean.

- *R*² = 0.80 → 80% of the variation in today's value is explained by yesterday's value. The metric is highly persistent.
- *R*² = 0.10 → Only 10% explained. Most of the variation comes from factors other than yesterday.

#### What is stationarity? (ADF test)

A **stationary** series fluctuates around a stable mean — it doesn't drift upward or downward over time. A **non-stationary** series has a trend (e.g., your fitness is steadily improving over months).

**Why it matters:** If your HRV has been rising steadily for 60 days, an AR(1) model will find $\phi$ ≈ 1.0 — but that's not because "yesterday predicts today." It's because *both* yesterday and today are on the same upward trend. The AR(1) is measuring the *trend*, not daily dependency.

The **Augmented Dickey-Fuller (ADF)** test detects this:
- *p* < 0.05 → The series is stationary (no trend). AR(1) is valid.
- *p* ≥ 0.05 → The series is trend-dominated. AR(1) is flagged as `⚠ TREND-DOMINATED`.

**Code:** `_layer2b_ar1`, line 876 — fits OLS regression of the series against its 1-day lag. ADF test runs via `statsmodels.tsa.stattools.adfuller()` with `maxlag=1` when *n* ≥ 8.

**Scientific justification:** Daza (2018) establishes AR(1) as a valid counterfactual framework for N-of-1 trials, allowing us to subtract the "expected" natural drift of a metric to isolate the impact of daily interventions.

---

### 6. Percentile-Based Anomaly Detection

#### What it is

For each metric, the code computes the **5th and 95th percentiles** over the entire window. Any of the 3 most recent daily values that falls outside this range is flagged as an anomaly.

**Example:** If your RHR has a 5th percentile of 52 bpm and 95th of 62 bpm, and today your RHR is 65 bpm → **HIGH anomaly**.

The code also reports the **z-score** for context:

$$z = \frac{x - \mu}{\sigma}$$

A z-score tells you how many standard deviations a value is from the mean:
- |*z*| > 2 → Unusual (occurs < 5% of the time in a normal distribution)
- |*z*| > 3 → Very rare (< 0.3%)

**Why percentiles instead of pure z-scores?** Percentiles are **distribution-agnostic** — they work for skewed data (like HRV or sleep duration) just as well as for bell-shaped data. z-scores assume normality.

**Code:** `_layer2c_anomalies`, line 920 — uses `np.percentile(vals, 5)` and `np.percentile(vals, 95)`.

---

### 7. Adaptive Markov Transition Matrices (Kernel Smoothing)

#### What is a Markov Chain?

A **Markov Chain** models a system that moves between discrete **states**, where the probability of the next state depends *only* on the current state (not on deeper history). In our case:

- The system discretises each metric into **3 states**: LOW, MED, HIGH (using percentile-based tertile bins).
- The **transition matrix** $T$ records the probability of moving from state $i$ to state $j$:

$$T_{ij} = P(S_{t+1} = j \mid S_t = i)$$

**Example transition matrix for Sleep Score:**

| From ↓ / To → | LOW | MED | HIGH |
|---|---|---|---|
| **LOW** | 0.55 | 0.35 | 0.10 |
| **MED** | 0.20 | 0.50 | 0.30 |
| **HIGH** | 0.10 | 0.30 | 0.60 |

**Reading this:** If you had LOW sleep today, there's a 55% chance tomorrow is also LOW (sleep is "sticky" in bad states), 35% chance of MED, and only 10% chance of jumping to HIGH.

High diagonal values = the state is **sticky** (hard to escape). Low diagonal values = the state is **volatile** (changes easily).

**Special case — ACWR:** Instead of percentile bins, the `acwr` metric uses **evidence-based thresholds** from Gabbett (2016, BJSM — 2 597 citations):
- < 0.8 → UNDERTRAINED
- 0.8–1.3 → SWEET_SPOT (optimal)
- 1.3–1.5 → HIGH_RISK
- \> 1.5 → DANGER (injury risk)

#### The "Small Data" Problem

In an N-of-1 dataset (one person, a few weeks), some state transitions may be observed only once or twice. A naive frequency count (Maximum Likelihood Estimation) would assign extreme probabilities: "LOW→HIGH happened 1 out of 1 times = 100% probability." This is statistically reckless.

#### The Solution: Adaptive Kernel Smoothing

Inspired by **regularisation techniques from Deep Learning** (as taught by **Prof. Oren Freifeld**, Introduction to Deep Learning, Ben-Gurion University of the Negev), we smooth the empirical transition matrix by mixing it with a **uniform prior**:

$$T'_{smoothed} = (1 - \alpha) \cdot T_{empirical} + \alpha \cdot U$$

Where $U$ is the uniform matrix $U_{ij} = 1/N_{states}$ (equal probability of going anywhere).

**The key innovation:** $\alpha$ is not constant — it is **adaptive**, computed from the number of observed transitions:

$$\alpha = \text{clamp}\left(\frac{0.50}{\sqrt{n_{transitions}}}, 0.02, 0.25\right)$$

| Observed transitions (*n*) | α | Effect |
|---|---|---|
| 4 | 0.25 (max) | Heavy smoothing — we don't trust sparse data |
| 10 | 0.16 | Moderate smoothing |
| 50 | 0.07 | Light smoothing — data starts to dominate |
| 200+ | 0.04 (near min) | Minimal smoothing — empirical data speaks |

**Why this works:** With very few observations, α is high, pulling all probabilities toward uniform (maximum uncertainty). As data accumulates, α decays, letting the empirical distribution dominate. This is the same principle as weight decay / L2 regularisation in neural networks — a Bayesian prior that fades as evidence grows.

**Code:** `_adaptive_alpha()` at line 125, applied in `_layer2e_markov()` at line 1077.

#### Confidence Tiers

Each Markov model is assigned a confidence tier based on the number of observed transitions:

| Transitions | Tier | Interpretation |
|---|---|---|
| ≥ 100 | HIGH | Trustworthy estimates |
| ≥ 30 | GOOD | Usable for recommendations |
| ≥ 15 | MODERATE | Directionally correct, handle with care |
| < 15 | PRELIMINARY | Treat as hypothesis only |

#### Stationary Distribution

From the smoothed transition matrix, the code computes the **stationary distribution** $\pi$ — the long-run probability of being in each state. Mathematically, $\pi$ is the left eigenvector of $T'$ corresponding to eigenvalue 1:

$$\pi \cdot T' = \pi, \quad \sum_i \pi_i = 1$$

**In plain language:** "If your current patterns continue forever, what fraction of your days will be LOW / MED / HIGH?" This tells the agents your physiological baseline.

**Code:** Computed via `np.linalg.eig(T'.T)` at line 1176.

---

### 8. Kullback-Leibler (KL) Divergence

#### What it is

KL Divergence measures how different two probability distributions are. In our system, it quantifies **conditional causality**: "Does knowing about a condition (e.g., high stress) *change* what happens to a metric tomorrow?"

$$D_{KL}(P \parallel Q) = \sum_j P_j \cdot \ln\left(\frac{P_j}{Q_j}\right)$$

Where:
- **$Q$ (the marginal)** = your *normal* transition probabilities (how your body moves between states regardless of any condition)
- **$P$ (the conditional)** = your transition probabilities *when a specific condition is present* (e.g., "days when stress was above the median")

#### How to interpret it

- **High $D_{KL}$** (e.g., 0.15+) → The condition fundamentally alters your day-to-day dynamics. Example: *"High stress breaks your recovery physics — when stressed, you're 3× more likely to stay stuck in LOW sleep."*
- **Low $D_{KL}$** (e.g., < 0.03) → The condition is noise. Your transition dynamics are unchanged regardless.

The system ranks all candidate conditioning metrics by their KL divergence and reports the **most influential** one for each target.

#### Safety gates

1. **Minimum transitions:** KL is only computed when **both** conditioning levels (above/below median) have ≥ 5 transitions each (`MIN_TRANSITIONS_KL = 5`). With fewer, the KL estimate is noise.
2. **Smoothing:** Both the marginal and conditional matrices use the same adaptive kernel smoothing (Section 7), preventing division-by-zero and dampening noise in sparse cells.
3. **Epsilon clipping:** All probabilities are clipped to a floor of $10^{-12}$ before computing $\ln(P/Q)$ to avoid $\ln(0)$.

**Code:** `_layer2e_markov`, starting at line 1183 — the KL loop iterates over candidate conditioning metrics, splits by median, and computes `np.sum(P * np.log(P / Q))` averaged over rows and both conditioning levels.

---

### 9. Conditioned AR(1) & Adjusted *R*²

#### What it is

While basic AR(1) (Section 5) asks "does yesterday's HRV predict today's HRV?", **conditioned AR(1)** asks: *"does yesterday's HRV PLUS yesterday's stress level predict today's HRV better than HRV alone?"*

Two levels:

**Level 1** — one conditioning variable:
$$X_t = \beta_0 \cdot X_{t-1} + \beta_1 \cdot Z_{t-1} + \beta_2 + \epsilon$$

**Level 2** — two conditioning variables (greedy forward selection from top-5 Level 1 winners):
$$X_t = \beta_0 \cdot X_{t-1} + \beta_1 \cdot Z_{1,t-1} + \beta_2 \cdot Z_{2,t-1} + \beta_3 + \epsilon$$

Solved via **Ordinary Least Squares**: `np.linalg.lstsq()`.

#### Why Adjusted *R*², not raw *R*²

In small datasets (N-of-1, sometimes only 10–20 data points), adding *any* extra variable will mechanically increase raw *R*². This is a well-known bias called **overfitting**.

**Adjusted *R*²** penalises complexity:

$$R^2_{adj} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}$$

Where *p* is the number of predictors and *n* is the number of observations.

**What this means:** The system only credits a predictor if it improves prediction power *after* the complexity penalty. Adding a useless variable will *decrease* adjusted *R*²  — making it a rigorous filter against coincidental correlations.

**Thresholds:**
- Level 1 models are kept only if *R*² > 0.10
- Level 2 models are kept only if *R*² > 0.15

**Improvement metric:** Each model reports `ΔR² = R²_conditioned − R²_baseline` — how much the conditioning variable(s) improved over simple AR(1). An improvement of 0.05+ is meaningful; 0.10+ is strong evidence that the conditioning variable genuinely helps predict the target.

**Code:** `_layer2d_conditioned_ar1`, line 954.

---

### 10. Multi-Lag Carryover Detection

#### What it is

Standard lag-1 analysis asks: "Does yesterday's training predict today's HRV?" But some effects take **2-3 days** to manifest. A hard workout might suppress HRV not just tomorrow, but for 48-72 hours.

The code checks **lag-2** and **lag-3** Pearson correlations for all key target metrics:

$$r(\text{predictor}_{t-k}, \text{target}_t) \quad \text{for } k \in \{2, 3\}$$

If a predictor→target relationship is still significant at lag-2 or lag-3, it indicates a **multi-day carryover effect**.

**Thresholds:** Same as lag-1 (|*r*| > 0.3, *p* < 0.05), with a stricter sample size requirement (*n* ≥ 8).

**Scientific basis:** Daza (2018) — carryover effects in N-of-1 experimental designs.

**Code:** `_multi_lag_carryover`, line 1275.

---

### 11. Rolling Correlation Stationarity

#### What it is

A correlation between two metrics might be *r* = −0.85 over 90 days. But what if that relationship only existed during weeks 1-4, and then vanished? A single *r* value over the full window would hide this instability.

The system computes a **rolling 30-day Pearson *r*** for the top-10 strongest correlations. It then measures the **standard deviation** of those rolling *r* values:

| std(*r*) | Stability | Interpretation |
|---|---|---|
| < 0.15 | STABLE | Trustworthy — the relationship is consistent over time |
| 0.15–0.25 | MODERATE | Somewhat variable — possibly influenced by a confound |
| ≥ 0.25 | UNSTABLE | Unreliable — the relationship may be driven by a specific phase or lifestyle change |

**Why this matters:** Agents are instructed to weight STABLE correlations more heavily in their recommendations. An UNSTABLE correlation may be real, but needs more investigation before acting on it.

**Code:** `_layer2f_rolling_correlation`, line 1317.

---

### 12. Acute:Chronic Workload Ratio (ACWR)

#### What it is

ACWR is the ratio of your **short-term** training load (acute, 7-day exponential moving average) to your **long-term** training load (chronic, 28-day average). It captures load spikes:

$$\text{ACWR} = \frac{\text{Acute Load (7d)}}{\text{Chronic Load (28d)}}$$

#### Evidence-based thresholds (Gabbett 2016, BJSM)

| ACWR | Zone | Meaning |
|---|---|---|
| < 0.8 | Undertrained | Doing less than your body is adapted to — detraining risk |
| 0.8 – 1.3 | Sweet Spot | Optimal loading — fitness gains with minimal injury risk |
| 1.3 – 1.5 | High Risk | Approaching overload — monitor recovery closely |
| > 1.5 | Danger Zone | Injury risk significantly elevated |

**Key point:** These thresholds are **not learned by AI**. They are hard-coded domain knowledge from sports science, injected directly into the feature set and the Markov state bins for ACWR.

**Code:** `ACWR_EDGES` and `ACWR_LABELS` at line 109, used in `_layer2e_markov`.

---

### 13. Multi-Scale Benchmark System

#### What it is

Rather than computing correlations over a single window, the engine runs the **entire 4-layer pipeline** over multiple time scales simultaneously:

| Window | Purpose |
|---|---|
| Current Week | Detect acute changes |
| 7, 14, 21, 30 days | Short-to-medium term patterns |
| 60, 90 days | Medium-term baselines |
| 180, 365 days | Long-term physiological trends |

Each window produces a full correlation matrix. The system then builds a **cross-timeframe stability analysis**: findings that appear at ALL windows are classified as **ROBUST** (highest confidence). Findings that only appear at one window are **EMERGING** (needs monitoring).

**Code:** `compute_benchmarks()`, line 336.

---

## Part II: Why Agents Aren't Hallucinating

The primary risk with Generative AI in data analysis is the **"Stochastic Parrot" effect** — LLMs are probabilistic token predictors, not calculators. They can produce plausible-sounding but mathematically wrong conclusions. We solve this through a strict **Separation of Concerns** architecture.

### 1. The "Pre-Computed Context" Protocol

The AI agents **never see raw CSV files or database tables directly for their analysis.** They do not perform arithmetic.

**What they receive:** A ~2-3 KB natural-language **summary** generated by Layer 3 of the `CorrelationEngine`. This summary contains *statistically validated facts* such as:

```
↑↓ stress_level × sleep_score: r=-0.670 (p=0.0042) ***
```

**What they do:** The agent translates `r=-0.670` into *"High stress is significantly degrading your sleep quality."* The agent cannot "invent" a correlation because it doesn't have the raw numbers to calculate one — it only has the validated results.

> [!CAUTION]
> Layer 3 (`_layer3_summary`, line 1374) includes an explicit **data quality disclaimer** when *n* < 21 days: *"All multi-variable models are PRELIMINARY — treat R² values and transition probabilities as directional, not definitive."* This disclaimer is injected directly into the agent's context so it knows to hedge its conclusions.

### 2. Read-Only SQL Guardrails

Agents are given a tool `run_sql_query` to explore raw data (e.g., "What activities happened last Tuesday?"), but it is **sandboxed**.

**How it works** (`enhanced_agents.py`, lines 56–61):

```python
forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE', 'CREATE']
query_upper = query.strip().upper()
for keyword in forbidden:
    if keyword in query_upper:
        return f"Error: {keyword} operations are not allowed. This tool is read-only."
```

**Effect:** The agent has **SELECT-only** capability. It can observe the database but cannot corrupt the ground truth. Any attempt to modify data is blocked before reaching the database.

### 3. Six Purpose-Built Tools

Each tool constrains what an agent can do:

| Tool | Purpose | Can it modify data? |
|---|---|---|
| `run_sql_query` | Explore raw data via SQL | No — read-only guard |
| `calculate_correlation` | Compute Pearson *r* between two metrics | No — read-only |
| `find_best_days` | Find top-*N* days for a metric | No — read-only |
| `analyze_pattern` | Compute trend, volatility (CV), direction | No — read-only |
| `get_past_recommendations` | Retrieve previous AI advice from DB | No — read-only |
| `save_recommendations_to_db` | Persist new recommendations | **Pipeline-only** — not exposed as an agent tool. Called by the orchestration code after parsing Synthesizer output. |

> [!IMPORTANT]
> `save_recommendations_to_db` is NOT in the agent's tool list. It is called only by the pipeline code (`run_weekly_summary`, line 943) after the Synthesizer finishes. The agent *asks* for recommendations to be saved by emitting structured blocks (RECOMMENDATION / TARGET_METRIC / EXPECTED_DIRECTION), and the pipeline code parses and saves them.

### 4. 10 Analytical Rules (Anti-Hallucination Constraints)

Every agent task includes an injected `analysis_rules` block (lines 611–648) with 10 mandatory rules:

1. **SAMPLE SIZE**: Any finding with *n* < 20 must be labelled PRELIMINARY. No recommendations on *n* < 10.
2. **OUTLIER CHECK**: Before calling something a "trend", remove the single best and worst day. If the conclusion changes → it's an outlier effect, not a trend.
3. **DATA QUALITY**: Values of 0 for REM/deep sleep or sleep_score < 40 = suspected tracking failure. Report averages with and without suspect nights.
4. **METRIC DISAMBIGUATION**: `tr_acute_load`, `daily_load_acute`, and `training_load` are THREE DIFFERENT metrics. Never conflate them.
5. **DAY-BY-DAY VERIFICATION**: When citing a correlation, query actual daily values and confirm the relationship holds for specific day pairs.
6. **BOUNCE-BACK MATTERS**: After a hard day, look at the *response*. If HRV normalised within 24h, that's a positive signal (good recovery capacity).
7. **VARIANCE ≠ TREND**: Sleep bouncing 52→89→72→83→70 is *variability*, not a decline. Report the range and standard deviation.
8. **USE ALL DATA**: Don't ignore new columns (ACWR, VO2Max, race times).
9. **POSITIVE FINDINGS**: Not everything needs fixing. If metrics are improving, say so prominently.
10. **NO GENERIC ADVICE**: Every recommendation must cite a specific correlation, day pair, or metric value. *"Get better sleep"* is not a recommendation.

### 5. Long-Term Memory (The "Synthesizer" Feedback Loop)

To prevent the "Goldfish Memory" problem — where an AI gives the same generic advice every week regardless of whether it worked — the **Synthesizer** agent has a dedicated memory system.

**How it works:**

1. **Before generating output**, the Synthesizer calls `get_past_recommendations(weeks=4)` to retrieve all previous recommendations from the `agent_recommendations` table in PostgreSQL.
2. **Cross-reference:** It compares past advice with current data. For each past recommendation:
   - Was it followed?
   - Did the target metric change in the expected direction?
   - Should it be reinforced, revised, or dropped?
3. **Structured output:** New recommendations are emitted in a parseable format:
   ```
   RECOMMENDATION: Increase sleep consistency (bedtime within 30-min window)
   TARGET_METRIC: sleep_score
   EXPECTED_DIRECTION: IMPROVE
   ```
4. **Pipeline saves:** The orchestration code (`_parse_recommendations`, line 1004) parses these blocks and calls `save_recommendations_to_db` to persist them.

**Result:** A **closed feedback loop**. The Synthesizer can say: *"Last week I recommended reducing evening screen time (Target: sleep_score). Your sleep score dropped 5%. This advice isn't working; let's pivot to sleep timing consistency instead."*

### 6. Specialized Agent Hierarchy

Instead of one "God Mode" agent that tries to do everything, the system splits the cognitive load among **5 narrow specialists**. This reduces the context window noise for each agent, minimising the risk of hallucination.

| # | Agent | Role | Has SQL access? |
|---|---|---|---|
| 1 | **Statistical Interpreter** | Translates every section of the pre-computed correlation data into plain language. No decision making — only interprets numbers. | No |
| 2 | **Health Pattern Analyst** | Detects non-obvious day-by-day patterns. Verifies correlations against actual data. Flags suspect data. Rigorously evaluates trends. | Yes |
| 3 | **Performance & Recovery** | Analyses week-over-week performance changes, ACWR trajectory, recovery capacity (bounce-back speed after hard days). | Yes |
| 4 | **Sleep & Lifestyle** | Deep-dives into sleep architecture (deep/REM/light %). Connects daily activities to biometric outcomes. Classifies activities (7-min bike commute ≠ cycling workout). | Yes |
| 5 | **Synthesizer & Fact-Checker** | The ONLY agent allowed to make recommendations. Cross-checks all other agents' findings. References past recommendations. Identifies the #1 bottleneck (if one exists). | Yes |

**Why the Statistical Interpreter has no SQL access:** It operates purely on the pre-computed summary. Giving it SQL access would risk it running its own (potentially flawed) statistics that contradict the deterministic engine — exactly the problem we're solving.

**Why the Synthesizer has memory but the others don't:** Only the final recommender needs historical context. Giving all agents memory would inflate context windows, increasing hallucination risk and API costs, with no analytical benefit.

This architecture ensures that no single entity has enough "creative license" to drift away from the hard data provided by the Mathematical Engine.

---

## References

| Source | Used for | Where in code |
|---|---|---|
| **Daza (2018)** — "Causal analysis of self-tracked time series data," N-of-1 methods | AR(1) as counterfactual framework, multi-lag carryover effects | `_layer2b_ar1`, `_multi_lag_carryover` |
| **Gabbett (2016)** — BJSM, "The training–injury prevention paradox" (2 597 citations) | ACWR evidence-based thresholds | `ACWR_EDGES`, `_layer2e_markov` |
| **Plews et al. (2012)** — "Training adaptation monitoring with HRV" | lnRMSSD transform, HRV weekly CV as recovery marker | `_layer0_load_and_clean` (ln_hrv, hrv_weekly_cv) |
| **Esco et al. (2025)** — HRV monitoring in athletes | lnRMSSD standard in sports science | `_layer0_load_and_clean` |
| **Prof. Oren Freifeld** — Intro to Deep Learning, Ben-Gurion University | Regularisation philosophy: Bayesian priors, weight decay, kernel smoothing as overfitting prevention | `_adaptive_alpha`, Markov smoothing |
