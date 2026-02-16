# Garmin Health Intelligence 🏃‍♂️📊

A personal data analytics project designed to dig deeper into wearable health metrics. The goal: take raw Garmin data, apply rigorous statistical analysis, and use specialized AI agents to interpret the results into actionable insights.

## Why I Built This

Garmin Connect is excellent for tracking *what* happened today or this week, but I often felt it lacked the depth to explain *why*. I found myself asking questions that the standard dashboard couldn't answer:

* Does yesterday's high stress *actually* impact my sleep quality tonight?
* Is this drop in HRV just random noise, or the start of a negative trend?
* What is the strongest statistical predictor for a high-performance training day?

Instead of feeding raw CSV files into an LLM and hoping for the best (which often leads to hallucinations), I built a pipeline that separates **mathematical calculation** from **AI interpretation**.

## Dashboard Preview

![Overview Page](docs/screenshots/overview.png)
*Real-time metrics with 7-day trends and recovery analysis*

![Correlation Matrix](docs/screenshots/correlations.png)
*Statistical relationships between sleep, HRV, and performance*

![Agent Chat](docs/screenshots/agent_chat.png)
*Ask natural language questions grounded in your statistical data*

## Example Insights

After analyzing several months of data, the system discovered:

* **Strong negative correlation (r = -0.67, p < 0.01)** between stress score and next-day training readiness
* **Markov analysis** revealed that after "Poor Sleep" nights, there's only a 23% probability of reaching "High Recovery" the next day
* **AR(1) persistence** showed my HRV has low momentum (φ = 0.31), suggesting high day-to-day variability — single-night sleep quality matters more than accumulated trends
* **Conditioned AR(1)** found that adding `daily_load_acute` as a predictor improves next-day `training_readiness` prediction from R² = 0.82 to R² = 0.94

## How It Works

The system runs as a weekly automated pipeline:

1.  **Data Collection** — Fetches raw data from the Garmin API (via `garth`) and upserts it into PostgreSQL. Handles API instability with retry logic.

2.  **Correlation Engine** — Before any AI touches the data, this engine calculates:
    * **Pearson Correlations** — linear relationships between metrics (same-day + lag-1, p < 0.05)
    * **AR(1) Models** — "momentum" / persistence of specific metrics with ADF stationarity checks
    * **Markov Transitions** — probability of state changes (e.g., "Poor Sleep" → "High Recovery")
    * **Conditioned AR(1)** — which variable *combinations* best predict next-day outcomes
    * **Percentile-Based Anomaly Detection** — distribution-agnostic alerting (5th/95th percentile)
    * **Rolling Correlation Stability** — 30-day rolling window to flag non-stationary relationships
    * **Multi-Lag Carryover** — lag-2 and lag-3 analysis for multi-day physiological effects

3.  **AI Agents** — A team of 5 specialized agents (CrewAI + Gemini 2.5 Flash) with **long-term memory**. Agents receive the *statistical summary*, not raw numbers — and the Synthesizer reviews past recommendations to track what worked.

4.  **Dashboard** — Streamlit interface to visualize trends, explore data, and chat with the agents using natural language.

5.  **Daily Input** — A self-reporting page where I log subjective data daily (stress, energy, caffeine, soreness, nutrition). These are automatically correlated with Garmin metrics in the next analysis cycle.

## Scientific Foundation

Every statistical method in this project is grounded in published, peer-reviewed research. This isn't guesswork — it's an implementation of established techniques from sports science, biostatistics, and digital health.

| Project Method | Published Basis | Key Finding |
|---|---|---|
| Pearson + lag-1 correlations | Standard biostatistics; validated for wearables by Seshadri et al. (2019) | Linear associations between wearable metrics are foundational for load monitoring |
| AR(1) persistence models | Daza (2018) — counterfactual N-of-1 framework | Autocorrelation in single-subject time series is a legitimate and established analytical approach |
| ADF stationarity testing | Daza (2018) — stationarization requirement | Non-stationary series must be flagged to avoid AR(1) capturing trends instead of true persistence |
| Markov transition matrices | Matias et al. (2022) — N-of-1 nighttime HR study | State-transition modeling of health metrics validated in 3 subjects over 4 years of wearable data |
| ACWR (acute:chronic workload ratio) | Gabbett (2016, BJSM, 2597 citations) | The "training-injury prevention paradox" — ACWR 0.8–1.3 is the evidence-based sweet spot |
| Conditioned AR(1) with exogenous predictors | Extension of Daza's framework; standard multivariate AR in epidemiology | Multi-variable prediction with adjusted R² prevents overfitting |
| Adaptive kernel smoothing | Bayesian-frequentist compromise for sparse data | Prevents sparse Markov rows from producing misleading probability estimates |
| KL-divergence for conditioning impact | Standard information theory; applied in Daza (2018) | Quantifies how much a conditioning variable changes day-to-day dynamics |
| Weekly HRV CV (coefficient of variation) | Plews et al. (2012, 500+ citations) | HRV variability, not just the mean, detects functional overreaching in athletes |
| Log-transformed HRV (lnRMSSD) | Esco et al. (2025, Sensors) — comprehensive HRV review | RMSSD is right-skewed; log transformation is the standard before parametric analysis |
| N-of-1 individual analysis | Hekler et al. (2019, BMC Medicine) — "Why We Need a Small Data Paradigm" | Individual-level data is more actionable for personal health than population averages |
| HRV-guided training prescription | Javaloyes et al. (2019, IJSPP, RCT, 147 citations) | HRV-guided training outperformed predefined plans in trained cyclists |
| Multi-day carryover detection | Daza (2018) — washout and carryover effects | Lag-2 and lag-3 analysis detects multi-day physiological effects (e.g., hard workout → 3-day HRV suppression) |
| Rolling correlation stability | Time-varying coefficient analysis (standard econometrics) | Identifies whether relationships between metrics are stable or phase-dependent |
| User-reported subjective data | Esco et al. (2025) — HRV alone is insufficient | Combining objective wearable data with subjective wellness reports improves recovery assessment |

*This project does not claim to replicate these studies — it implements their validated methods on personal wearable data, combining them into a single analytical pipeline. All statistical outputs include confidence tiers, significance thresholds, and explicit "correlation ≠ causation" disclaimers.*

### References

1. Daza, E.J. (2018). Causal Analysis of Self-tracked Time Series Data Using a Counterfactual Framework for N-of-1 Trials. *Methods of Information in Medicine*, 57(1), 10–21. PMID: [29621835](https://pubmed.ncbi.nlm.nih.gov/29621835/)
2. Esco, M.R. et al. (2025). Monitoring Training Adaptation and Recovery Status in Athletes Using Heart Rate Variability via Mobile Devices: A Narrative Review. *Sensors*, 26(1), 3. DOI: [10.3390/s26010003](https://doi.org/10.3390/s26010003)
3. Gabbett, T.J. (2016). The training–injury prevention paradox: should athletes be training smarter *and* harder? *British Journal of Sports Medicine*, 50(5), 273–280. DOI: [10.1136/bjsports-2015-095788](https://doi.org/10.1136/bjsports-2015-095788)
4. Plews, D.J. et al. (2012). Heart rate variability in elite triathletes, is variation in variability the key to effective training? *European Journal of Applied Physiology*, 112, 3729–3741. DOI: [10.1007/s00421-012-2354-4](https://doi.org/10.1007/s00421-012-2354-4)
5. Hekler, E.B. et al. (2019). Why we need a small data paradigm. *BMC Medicine*, 17, 133. PMID: [31311528](https://pubmed.ncbi.nlm.nih.gov/31311528/)
6. Javaloyes, A. et al. (2019). Training Prescription Guided by Heart-Rate-Variability in Cycling. *International Journal of Sports Physiology and Performance*, 14(1), 23–32. DOI: [10.1123/ijspp.2018-0122](https://doi.org/10.1123/ijspp.2018-0122)
7. Matias, R. et al. (2022). What possibly affects nighttime heart rate? Conclusions from N-of-1 observational data. *Digital Health*, 8. PMID: [36046637](https://pubmed.ncbi.nlm.nih.gov/36046637/)
8. Seshadri, D.R. et al. (2019). Wearable sensors for monitoring the internal and external workload of the athlete. *NPJ Digital Medicine*, 2, 71. DOI: [10.1038/s41746-019-0149-2](https://doi.org/10.1038/s41746-019-0149-2)

## Technical Challenges & Solutions

### The "Date Gap" Problem
Standard `.shift(1)` is dangerous with wearable data — if I skip wearing the watch for a weekend, a naive shift compares Friday to Monday as if they're consecutive, corrupting correlations.
**Solution:** Strict date enforcement (`.asfreq('D')`) inserts `NaN` for missing days, so models only analyze true consecutive sequences.

### Preventing AI Hallucinations
LLMs are confident but bad at math. The AI never sees raw CSVs — only a structured "Context Window" generated *after* statistical significance is proven ($p < 0.05$). Anomaly detection uses percentiles (5th/95th) instead of z-scores — works equally well for normal and skewed metrics like HRV.

### Signal vs. Noise
Wearable data is noisy. Adaptive Kernel Smoothing on Markov transition matrices prevents a single outlier from skewing the probability model.

## How It Differs From Existing Tools

| Feature | Garmin Connect | Whoop | This Project |
|---------|---------------|-------|--------------|
| Correlation Analysis | ❌ | ⚠️ Black box | ✅ Transparent |
| Markov Transitions | ❌ | ❌ | ✅ Yes |
| AR(1) / Conditioned AR(1) | ❌ | ❌ | ✅ Yes |
| Custom Queries | ❌ | ❌ | ✅ Natural language + SQL |
| Cost | Free* | $30/mo | ~$5/mo |
| AI Interpretation | Basic | ⚠️ Proprietary | ✅ Open, auditable agents |

*\*Requires watch purchase*

## Tech Stack

**Core Pipeline:** Python 3.12 → PostgreSQL → Gemini 2.5 Flash
**Statistics:** NumPy, SciPy (Pearson, AR(1), Markov chains)
**Visualization:** Streamlit + Plotly
**Automation:** GitHub Actions (weekly cron)

## The Agents

The system runs 5 specialized agents in sequence:

| Agent | Role |
|-------|------|
| **Statistical Interpreter** | Reads the full correlation matrix and translates every section into plain-language findings with confidence ratings |
| **Health Pattern Analyst** | Finds day-by-day patterns and trends — verifies correlations against actual data, flags outliers |
| **Performance & Recovery** | Week-over-week comparison + bounce-back analysis after hard training days |
| **Sleep & Lifestyle** | Sleep architecture deep-dive + connects specific activities to next-day outcomes |
| **Synthesizer** | Fact-checks all previous agents, reviews past recommendations via long-term memory, identifies the #1 bottleneck, and gives evidence-based quick wins |

## Known Limitations

* **Sample size:** Built for individual use (n=1). Patterns may not generalize to other users.
* **Data sparsity:** Markov transitions require ≥100 days for stability. With <30 days, results are marked as preliminary. Plews et al. (2014) recommend a minimum of 3–5 HRV recordings per week for a valid weekly profile.
* **Correlation ≠ Causation:** Statistical relationships don't prove cause-effect. The system flags this explicitly.
* **No clinical validation:** Statistical methods are unit-tested for mathematical correctness (60 tests across 3 test suites), but no prospective validation has been performed against health outcomes. This is an exploratory tool, not a validated clinical instrument.
* **User-reported data:** Wellness and nutrition entries are self-reported — subject to recall bias and inconsistent logging.
* **AI interpretation boundaries:** Commercial wearable platforms (Garmin, Whoop, Oura) intentionally limit AI-driven health feedback due to regulatory classification as Software as a Medical Device (SaMD) and liability concerns. This project operates outside that scope as a personal, non-commercial, educational tool.

## Disclaimer

This project applies published statistical methods to personal wearable data for educational exploration. **It is not a medical device, does not provide diagnoses, and should not replace professional medical advice.**

The AI agents interpret statistical patterns — they do not have clinical training, context about your medical history, or the ability to perform physical examination. Recommendations generated by this system are data-driven observations, not medical guidance.

If you notice concerning health trends, consult a qualified healthcare professional.

---

*This project is for educational and personal use, exploring the intersection of Data Engineering, Statistics, and GenAI.*
