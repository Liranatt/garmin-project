# Why the Math Mathing & Why Agents Aren't Hallucinating 🧠📐

This document deconstructs the mathematical and algorithmic infrastructure of the **Garmin Health Intelligence** system.
The core engineering philosophy here is **Determinism before Semantics**: I tried to solve the "Black Box" problem inherent in Large Language Models (LLMs) by strictly separating the **Mathematical Engine** (which calculates truth) from the **AI Agents** (which interpret meaning).

---

## Part I: The Mathematical Engine

The `CorrelationEngine` (`src/correlation_engine.py`) operates as a deterministic 4-layer perceptron for statistical truth. It computes hard metrics before any AI involvement. Below is the rigorous definition, implementation, and justification for each component.

### 1. Autoregressive Models (AR-1) & Stationarity

#### Definition
The **Autoregressive (AR-1)** model posits that the value of a biometric metric at day $t$ is a linear function of its value at day $t-1$, plus noise:
$$X_t = c + \phi X_{t-1} + \epsilon_t$$
Here, $\phi$ represents the **Persistence Coefficient** (or "physiological momentum").

#### Implementation Details
* **Code:** `_layer2b_ar1` in `correlation_engine.py`.
* **Logic:** The system performs an Ordinary Least Squares (OLS) regression of the series against its own 1-day lag.
* **Stationarity Guard:** Before fitting, the system runs an **Augmented Dickey-Fuller (ADF)** test (`adfuller` from `statsmodels`). If $p > 0.05$, the series is flagged as non-stationary (trend-dominated), and the AR-1 validity is downgraded. This prevents the model from mistaking a long-term trend for daily dependency.

#### Scientific Justification
* **Daza (2018)** establishes AR-1 as a valid counterfactual framework for N-of-1 trials, allowing us to subtract the "expected" natural drift of a metric to isolate the impact of daily interventions.

---

### 2. Adaptive Markov Transition Matrices (Kernel Smoothing)

#### Definition
We model health states as a **Markov Chain** with 3 discrete states (Low, Medium, High). The transition matrix $T$ defines the probability of moving from state $i$ to state $j$:
$$T_{ij} = P(S_{t+1} = j \mid S_t = i)$$

#### The "Small Data" Problem
In an N-of-1 dataset, observations are sparse. A naive frequency count (Maximum Likelihood Estimation) is unstable. If you observed "Low Recovery" only twice, and once it led to "High", a naive model assumes a 50% probability, which is statistically reckless.

#### The Solution: Adaptive Kernel Smoothing
To solve this, we implement a smoothing technique inspired by **Deep Learning priors** (as taught by **Prof. Oren Freifeld**, BGU). We regularize the empirical transition matrix $T_{emp}$ by mixing it with a uniform prior $U$:

$$T'_{smoothed} = (1 - \alpha) \cdot T_{emp} + \alpha \cdot U$$

Where $\alpha$ is not constant, but **adaptive** based on sample size $N$:
$$\alpha = \text{clamp}\left( \frac{0.5}{\sqrt{N}}, 0.02, 0.25 \right)$$

* **Logic (from `_adaptive_alpha`):** When $N$ is small (sparse data), $\alpha$ is high (~0.25), pulling probabilities toward uniformity (maximum entropy/uncertainty). As $N$ grows, $\alpha$ decays, allowing the empirical data to dominate.
* **Justification:** This prevents the "Overfitting" of probability estimates on sparse physiological data.

---

### 3. Kullback-Leibler (KL) Divergence

#### Definition
KL Divergence measures the "information distance" between two probability distributions $P$ and $Q$. In our context, it quantifies **Conditional Causality**:
$$D_{KL}(P \parallel Q) = \sum P(x) \ln \left( \frac{P(x)}{Q(x)} \right)$$

#### Implementation
* **$Q$ (Marginal):** How your body transitions between states *normally*.
* **$P$ (Conditional):** How your body transitions *when specific condition $Z$ is present* (e.g., "Alcohol Intake > 0").
* **Interpretation:**
    * High $D_{KL}$: The condition fundamentally alters your physiological dynamics (e.g., "Alcohol breaks your recovery physics").
    * Low $D_{KL}$: The condition is noise; the transition dynamics remain unchanged.

#### Scientific Justification
This extends Information Theory into physiological monitoring, allowing the agents to rank "Feature Importance" quantitatively rather than heuristically.

---

### 4. Conditioned AR(1) & Adjusted $R^2$

#### Definition
A multivariate extension of the AR model that introduces exogenous regressors ($Z$):
$$X_t = \beta_0 X_{t-1} + \beta_1 Z_{t-1} + \epsilon$$

#### Implementation
* **Code:** `_layer2d_conditioned_ar1`.
* **Metric:** We use **Adjusted $R^2$** ($R^2_{adj}$), not raw $R^2$.
    $$R^2_{adj} = 1 - (1-R^2) \frac{n-1}{n-p-1}$$
* **Why:** In small datasets (N-of-1), adding *any* variable increases raw $R^2$. Adjusted $R^2$ penalizes complexity ($p$). The system only accepts a predictor if it improves prediction power *after* the penalty for adding complexity. This rigorously filters out coincidental correlations.

---

### 5. Acute:Chronic Workload Ratio (ACWR)

#### Definition
The ratio of the acute training load (7-day exponential moving average) to the chronic training load (28-day average).

#### Implementation
* **Thresholds:** The code explicitly defines bins based on **Gabbett (2016)**:
    * `< 0.8`: Undertrained
    * `0.8 - 1.3`: Sweet Spot (Optimal)
    * `> 1.5`: Danger Zone (Injury Risk)
* **Logic:** This is not learned by AI; it is hard-coded domain knowledge injected into the feature set.

---

## Part II: Why Agents Aren't Hallucinating

The primary risk with GenAI in data analysis is the "Stochastic Parrot" effect—LLMs are probabilistic token predictors, not calculators. We solve this via a strict **Separation of Concerns** architecture defined in `enhanced_agents.py`.

### 1. The "Pre-Computed Context" Protocol
The AI agents **never see raw CSV files**. They do not perform arithmetic.
* **Input:** They receive a text-based `summary` generated by Layer 3 of the `CorrelationEngine`.
* **Content:** This summary contains *statistically validated facts* (e.g., "Pearson r=-0.67 (p<0.01) between Stress and Sleep").
* **Task:** The Agent's role is strictly **Semantic Interpretation**. It translates "r=-0.67" into "High stress is significantly degrading your sleep quality." It cannot "invent" a correlation because it doesn't have the raw numbers to calculate one—it only has the validated results.

### 2. Read-Only SQL Guardrails
Agents are given a tool `run_sql_query` to explore the data (e.g., "What happened last Tuesday?"), but it is sandboxed.
* **Code:** `enhanced_agents.py` lines ~60-70.
* **Logic:** The function explicitly scans for keywords: `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`. If found, it raises a `PermissionError`.
* **Effect:** The Agent has "Select-Only" capability. It can observe the database but cannot corrupt the ground truth.

### 3. Long-Term Memory (The "Synthesizer" Loop)
To prevent the "Goldfish Memory" problem where an AI gives the same generic advice every week, the `Synthesizer` agent has a dedicated memory loop.
* **Mechanism:** It queries the `agent_recommendations` table in PostgreSQL.
* **Logic:** Before generating output, it executes: `get_past_recommendations(weeks=4)`.
* **Outcome:** It cross-references past advice with current data. "Last week I told you to increase sleep (Target: Sleep Score). Your score dropped by 5%. This advice is not working; let's pivot." This creates a **Closed Feedback Loop** grounded in reality.

### 4. Specialized Agent Hierarchy
Instead of one "God Mode" agent, we split the cognitive load among 5 narrow specialists. This reduces the context window noise for each agent, minimizing hallucinations.
1.  **Statistical Interpreter:** Translates math to English. (No decision making).
2.  **Health Pattern Analyst:** Detects N-of-1 outliers.
3.  **Performance & Recovery:** Focuses purely on load (ACWR) and recovery physics.
4.  **Sleep & Lifestyle:** Focuses on sleep architecture.
5.  **Synthesizer:** The only agent allowed to make recommendations, based on the filtered inputs of the others.

This architecture ensures that no single entity has enough "creative license" to drift away from the hard data provided by the Mathematical Engine.
