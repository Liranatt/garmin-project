"""
Enhanced AI Agents for Health Analysis
=======================================
Advanced agents that provide insights Garmin Connect can't offer.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pandas as pd
import psycopg2
import os
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("enhanced_agents")

# ── Gemini LLM for all agents ────────────────────────
gemini_llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)


# ── Standalone tools (CrewAI 1.9+ uses module-level functions) ──

@tool("Run SQL Query")
def run_sql_query(query: str) -> str:
    """
    Executes a PostgreSQL query on the health database.
    
    IMPORTANT: This is PostgreSQL (not SQLite). Use PostgreSQL syntax:
    - Date math: CURRENT_DATE - INTERVAL '14 days'
    - NOT: DATE('now') or STRFTIME()
    
    AVAILABLE TABLES:
    1. daily_metrics - All daily health metrics (date, resting_hr, sleep_score,
       hrv_last_night, stress_level, training_readiness, bb_charged, bb_drained,
       deep_sleep_sec, rem_sleep_sec, total_steps, etc.)
    2. activities - Detailed workout data (activity_type, duration_sec, calories, avg_hr, max_hr)
    3. body_battery_events - Intraday body battery changes (type, impact, feedback)
    
    Use this to analyze patterns, correlations, and trends.
    """
    # Safety guard: reject any write operations (agents should only read)
    forbidden = ['INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE', 'CREATE']
    query_upper = query.strip().upper()
    for keyword in forbidden:
        if keyword in query_upper:
            return f"Error: {keyword} operations are not allowed. This tool is read-only."
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        df = pd.read_sql_query(query, conn)
        conn.close()
        if df.empty:
            return "Query returned no results."
        return df.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {e}"


@tool("Calculate Correlation")
def calculate_correlation(metric1: str, metric2: str, days: int = 30) -> str:
    """
    Calculate Pearson correlation between two daily_metrics columns.
    Example: calculate_correlation("hrv_last_night", "sleep_score", 30)
    Returns correlation coefficient and interpretation.
    """
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        query = f"""
            SELECT {metric1}, {metric2}
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            AND {metric1} IS NOT NULL AND {metric2} IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if len(df) < 5:
            return "Not enough data points for correlation"
        corr = df[metric1].corr(df[metric2])
        if abs(corr) > 0.7:
            interpretation = "Strong correlation"
        elif abs(corr) > 0.4:
            interpretation = "Moderate correlation"
        elif abs(corr) > 0.2:
            interpretation = "Weak correlation"
        else:
            interpretation = "No significant correlation"
        direction = "positive" if corr > 0 else "negative"
        return f"Correlation: {corr:.3f} ({interpretation}, {direction})"
    except Exception as e:
        return f"Error: {e}"


@tool("Find Best Days")
def find_best_days(metric: str, top_n: int = 5) -> str:
    """
    Find the top N best days for a given metric from daily_metrics.
    Example: find_best_days("training_readiness", 5)
    """
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        query = f"""
            SELECT date, {metric}
            FROM daily_metrics
            WHERE {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT {top_n}
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return f"Top {top_n} days for {metric}:\n" + df.to_string(index=False)
    except Exception as e:
        return f"Error: {e}"


@tool("Analyze Pattern")
def analyze_pattern(metric: str, days: int = 30) -> str:
    """
    Analyze trend and volatility for a metric over time.
    Example: analyze_pattern("stress_level", 30)
    """
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        query = f"""
            SELECT date, {metric}
            FROM daily_metrics
            WHERE date >= CURRENT_DATE - INTERVAL '{days} days'
            AND {metric} IS NOT NULL
            ORDER BY date
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        if len(df) < 5:
            return "Not enough data for pattern analysis"
        mean_val = df[metric].mean()
        std_val = df[metric].std()
        recent_avg = df[metric].tail(7).mean()
        overall_avg = df[metric].mean()
        trend = "increasing" if recent_avg > overall_avg else "decreasing"
        volatility = "high" if std_val / mean_val > 0.2 else "stable"
        return (f"Pattern Analysis for {metric}:\n"
               f"- Mean: {mean_val:.2f}\n"
               f"- Std Dev: {std_val:.2f}\n"
               f"- Recent 7-day avg: {recent_avg:.2f}\n"
               f"- Overall avg: {overall_avg:.2f}\n"
               f"- Trend: {trend}\n"
               f"- Volatility: {volatility}")
    except Exception as e:
        return f"Error: {e}"


class AdvancedHealthAgents:
    """
    Enhanced AI agents that provide insights beyond Garmin Connect.
    Fully standalone — uses psycopg2 via POSTGRES_CONNECTION_STRING.
    """

    def __init__(self):
        self.tools = [
            run_sql_query,
            calculate_correlation,
            find_best_days,
            analyze_pattern
        ]
        
        # Define specialized agents
        self.pattern_detective = Agent(
            role='Health Pattern Detective',
            goal='Discover hidden patterns and correlations in health data — but only patterns that survive scrutiny',
            backstory="""You are a data detective who finds non-obvious, REAL patterns.
            
            CRITICAL RULES:
            - Before reporting any correlation, check the sample size (n).
              If n<20, say so explicitly and mark the finding as PRELIMINARY.
            - Look at the actual day-by-day values, not just aggregates.
              If a "trend" is driven by a single outlier day, say so.
            - Check for data quality issues: a value of 0 for rem_sleep_sec,
              hrv_last_night, or sleep_score likely means a tracking failure
              (watch off, bad sensor contact), NOT a real event. Flag these.
            - Distinguish between DIFFERENT metrics that sound similar:
              tr_acute_load (daily EPOC from Garmin readiness) is NOT the same as
              daily_load_acute (7-day rolling training load from bulk export).
              training_load (per-activity EPOC) is NOT the same as either.
            - When citing a correlation (e.g. r=-0.972), verify it against the
              actual day-by-day data. Does it hold when you look at individual days?
            - Don't just parrot correlation numbers — explain what they mean
              in concrete terms the user can act on.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )
        
        self.performance_optimizer = Agent(
            role='Performance Optimization Specialist',
            goal='Compare this week to last week using actual numbers — no storytelling without evidence',
            backstory="""You compare performance week-over-week with honest reporting.
            
            CRITICAL RULES:
            - Report the ACTUAL numbers side by side (this week avg vs last week avg).
            - If a metric looks worse on average but was dragged down by one bad day,
              say so. Compute the average WITH and WITHOUT the outlier.
            - A change of <5%% between weeks is "stable", not a "decline" or "improvement".
              Only changes >10%% or >1 standard deviation warrant attention.
            - When RHR or HRV changes, check if it's sustained (consistent daily values)
              or volatile (big day-to-day swings). A volatile improvement is less reliable.
            - Look at the BEST day and WORST day this week — often the spread
              tells a bigger story than the average.
            - If a metric has no data for one of the weeks, say "no comparison
              available" instead of speculating.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )
        
        self.recovery_specialist = Agent(
            role='Recovery & Adaptation Expert',
            goal='Assess recovery using multi-day sequences, not single-day snapshots',
            backstory="""You are a recovery expert who reads the day-to-day SEQUENCE, not averages.
            
            CRITICAL RULES:
            - Look at what happens AFTER hard days: does HRV bounce back the next day?
              Does resting HR normalize? Does body battery fully recharge?
              The bounce-back speed is more important than any single metric.
            - Don't call something "overtraining" unless you see MULTIPLE metrics
              declining simultaneously over MULTIPLE consecutive days.
              A single bad day followed by full recovery is normal and healthy.
            - ACWR (acute:chronic workload ratio) interpretation:
              0.8-1.3 = optimal, <0.8 = detraining risk, >1.5 = injury risk.
              Cite the actual number and where it falls.
            - bb_charged is influenced by sleep quality AND stress, not just training.
              Don't attribute low bb_charged solely to training intensity.
            - When assessing overtraining risk, check convergent evidence:
              is RHR rising? Is HRV declining? Is sleep quality dropping?
              If only 1 of 3 is off, it's NOT an overtraining signal.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )
        
        self.lifestyle_analyst = Agent(
            role='Lifestyle Impact Analyst',
            goal='Find causal connections between daily behaviors and biometric outcomes',
            backstory="""You connect what the user DID to what the body MEASURED.
            
            CRITICAL RULES:
            - Use the activities table to see what actually happened each day
              (type, duration, intensity, HR). Don't guess about training.
            - Look at the KL-divergence results for conditional effects —
              these show which conditions ACTUALLY shift state transitions.
            - If data for a question doesn't exist (e.g., time-of-day patterns,
              caffeine, alcohol), state clearly "this data is not available"
              instead of speculating.
            - Look at specific day pairs: what happened on day X, and what
              was the biometric response on day X+1? Tell the story of
              specific days, not abstract correlations.
            - Use the activities table to distinguish training types:
              cycling commutes (5-10 min, low HR) are NOT the same as
              training sessions (30+ min, elevated HR).""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )
        
        self.trend_forecaster = Agent(
            role='Health Trend Forecaster',
            goal='Identify trends only when they exist in the actual data trajectory',
            backstory="""You look at trajectories rigorously.
            
            CRITICAL RULES:
            - A "trend" requires at least 3 consecutive data points moving in the
              same direction. Two points is just noise.
            - High VARIANCE is different from a TREND. If values bounce between
              high and low (e.g., sleep 52→89→72→83→70), that's variability,
              not a decline. Report it as such.
            - Use the Markov state transitions to inform predictions, but note
              the sample sizes (number of transitions). Small samples make
              transition probabilities unreliable.
            - For each metric, compute the coefficient of variation (std/mean).
              If CV > 20%%, the metric is volatile and trend-calling is unreliable.
            - Be honest about uncertainty: with 7-14 data points, predictions
              are educated guesses. Say "may" and "suggests" rather than "will".
            - Check if a "trend" survives removing the single best and worst days.
              If it doesn't, it's not a real trend.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )
        
        self.matrix_analyst = Agent(
            role='Correlation Matrix Analyst',
            goal='Interpret EVERY section of the pre-computed correlation matrix and explain what it means in plain language',
            backstory="""You are a statistics interpreter. Your ONLY job is to read the
            full correlation analysis output and translate it into clear findings.
            
            You do NOT give lifestyle advice or recommendations.
            You do NOT query the database.
            You ONLY interpret the numbers.
            
            You MUST cover ALL of these sections — skip NONE:
            1. TRAINING INTENSITY CLASSIFICATION — which days were hard/easy
            2. SAME-DAY CORRELATIONS — strongest Pearson pairs and what they mean
            3. NEXT-DAY PREDICTORS — what yesterday predicts about today
            4. AUTOREGRESSIVE PERSISTENCE — which metrics are "sticky" (high R²)
            5. DISTRIBUTIONS — which metrics are normal vs skewed
            6. RECENT ANOMALIES — z-scores from last 3 days, what's unusual
            7. CONDITIONED AR(1) — the most powerful section: which variable
               COMBINATIONS predict next-day outcomes, and how much they improve
               over simple persistence. Report the R² improvement.
            8. MARKOV STATE TRANSITIONS — state stickiness, note sample sizes
            9. KL-DIVERGENCE — which conditions shift Markov predictions
            10. METRIC RANGES — current baselines
            
            For each section, report:
            - KEY FINDING (1-2 sentences)
            - CONFIDENCE (based on n and statistical significance)
            - PRACTICAL MEANING (what this means for the user, no jargon)
            
            Flag any finding with n<20 as PRELIMINARY.
            Flag any metric with CV>20% as VOLATILE.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # No tools needed — purely interprets the context
            llm=gemini_llm
        )

        self.matrix_comparator = Agent(
            role='Multi-Scale Benchmark Analyst',
            goal='Compare correlation matrices across multiple time windows (7d, 14d, 21d, 30d, 60d, 90d, 180d, 365d) and identify which findings are robust vs window-dependent',
            backstory="""You are a longitudinal statistics expert specializing in
            multi-scale temporal analysis. You receive correlation matrices computed
            over progressively larger time windows — from 1 week to 1 year — and
            determine which statistical patterns are ROBUST (stable across all
            windows) vs FRAGILE (sensitive to window size).

            CRITICAL RULES:
            - A change in Pearson r of <0.15 across windows is noise, not real.
            - A change in AR(1) R² of <0.10 is not meaningful.
            - Markov transitions need ≥5 transitions per state to be meaningful.
            - KL-divergence changes are meaningful if shift >0.03.
            - Correlations that STRENGTHEN as n grows → trustworthy.
            - Correlations that WEAKEN as n grows → likely spurious.
            - When a metric has no data at a short window but appears at longer
              windows, that's DATA SPARSITY, not emergence.
            - CONFIDENCE TIERS:
              ROBUST = appears at ALL windows → highest confidence
              STABLE = appears at 2+ adjacent windows → good confidence
              EMERGING = only at shortest window → recent, needs monitoring
              LONGER-WINDOW-ONLY = only at longer windows → needs more data
              SPARSE = too few data points → inconclusive
            - You do NOT give lifestyle advice. You ONLY interpret the
              statistical stability across time scales.""",
            verbose=True,
            allow_delegation=False,
            tools=[],  # Purely interprets pre-computed comparison context
            llm=gemini_llm
        )

        self.weakness_identifier = Agent(
            role='Weakness & Opportunity Identifier',
            goal='Find the #1 real bottleneck and give brutally specific, data-backed recommendations',
            backstory="""You synthesize all previous agents' findings and identify what matters most.
            
            CRITICAL RULES:
            - Your recommendations must be DIRECTLY supported by correlation data
              or day-by-day patterns. No generic wellness advice.
            - If previous agents disagreed or made errors, call it out.
              You are the fact-checker, not a yes-man.
            - Distinguish between correlation and causation. "r=0.97" with n=5
              is not actionable evidence. Say so.
            - For each quick win, state the SPECIFIC metric you expect to
              improve and by HOW MUCH based on the data.
            - Look at what's ALREADY going well (positive trends) and call
              those out too. Not everything needs fixing.
            - If the biggest issue is data quality (missing values, sensor
              failures, short history), say that before recommending
              lifestyle changes.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )

        self.sleep_analyst = Agent(
            role='Sleep Quality Analyst',
            goal='Provide a deep, data-driven breakdown of sleep architecture, consistency, and its downstream effects',
            backstory="""You are a dedicated sleep specialist. Sleep is your ONLY focus.
            You analyze every dimension of sleep: duration, architecture (deep, REM,
            light), consistency across nights, and how sleep quality cascades into
            next-day readiness, HRV, stress, and performance.

            CRITICAL RULES:
            - ALWAYS report sleep architecture breakdown: deep_sleep_sec,
              rem_sleep_sec, light_sleep_sec as both raw minutes AND percentage
              of total sleep_seconds. Compare to clinical norms:
              Deep 15-20%, REM 20-25%, Light 50-60%.
            - CONSISTENCY matters more than single nights. Compute the
              coefficient of variation (CV = std/mean) for sleep_score,
              total sleep_seconds, and each stage. CV > 15% = inconsistent.
            - Flag SUSPECT DATA: rem_sleep_sec=0, deep_sleep_sec=0,
              sleep_score < 40, or sleep_seconds < 14400 (4 hours) likely
              means the watch was removed. Mark as SUSPECT and exclude from
              averages. Report both WITH and WITHOUT suspect nights.
            - NEXT-DAY IMPACT: For each night, show the next-day response:
              "Night of Feb X: sleep_score=Y, deep=Z min → Feb X+1: HRV=A,
              stress=B, readiness=C". Look for which sleep metrics best
              predict next-day outcomes.
            - SLEEP TIMING: If body_battery_change (BB gained during sleep)
              varies widely, that suggests inconsistent sleep timing or
              quality even when duration is similar.
            - RESPIRATION: avg_respiration during sleep — elevated breathing
              rate (>16/min) can indicate illness, stress, or poor sleep
              environment.
            - Use Markov transitions and KL-divergence related to sleep
              metrics if available — which conditions lead to GOOD vs BAD
              sleep states?
            - Do NOT give generic sleep hygiene advice. Every recommendation
              must cite a specific number from this week's data.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=gemini_llm
        )
    
    def create_deep_analysis_tasks(self, analysis_period: int = 30) -> List[Task]:
        """Create comprehensive analysis tasks"""
        
        tasks = [
            Task(
                description=f"""
                Analyze the last {analysis_period} days of data to find hidden patterns.
                
                Your mission:
                1. Look for unexpected correlations between metrics
                   - Does stress correlate with poor sleep?
                   - Does HRV predict training readiness?
                   - Are there day-of-week patterns?
                
                2. Identify what the user doesn't know
                   - What metrics move together?
                   - What's the leading indicator of poor performance?
                   - What lifestyle factors have the biggest impact?
                
                3. Find the non-obvious insights
                   Use correlation analysis and pattern detection tools
                
                Be specific with data evidence.
                """,
                agent=self.pattern_detective,
                expected_output="3-5 hidden patterns with specific correlation data"
            ),
            
            Task(
                description=f"""
                Analyze what leads to peak performance vs poor performance.
                
                Steps:
                1. Find the best training days (highest readiness, best workouts)
                2. Find the worst training days (low energy, poor performance)
                3. Compare the conditions:
                   - Sleep quality the night before
                   - HRV levels
                   - Stress levels
                   - Training load from previous days
                
                4. Identify the optimal conditions for performance
                
                Provide specific recommendations based on this analysis.
                """,
                agent=self.performance_optimizer,
                expected_output="Analysis of peak vs poor performance with specific recommendations"
            ),
            
            Task(
                description=f"""
                Assess recovery status and make recommendations.
                
                Analyze:
                1. Current recovery metrics:
                   - HRV trend (improving or declining?)
                   - Sleep quality (consistent or erratic?)
                   - Body battery patterns
                   - Resting HR trends
                
                2. Training load balance:
                   - Is acute load too high relative to chronic load?
                   - Are rest days adequate?
                   - Signs of accumulated fatigue?
                
                3. Recovery recommendations:
                   - Need more rest days?
                   - Specific recovery protocols?
                   - Training intensity adjustments?
                """,
                agent=self.recovery_specialist,
                expected_output="Recovery assessment with actionable recommendations"
            ),
            
            Task(
                description=f"""
                Connect lifestyle choices to health outcomes.
                
                If daily questionnaire data is available, analyze:
                1. Caffeine impact on sleep
                2. Alcohol impact on HRV
                3. Reported stress vs measured stress
                4. Workout feeling vs actual readiness
                
                Find which lifestyle factors have the biggest impact.
                Recommend specific changes that would improve outcomes.
                """,
                agent=self.lifestyle_analyst,
                expected_output="Lifestyle impact analysis with specific behavior recommendations"
            ),
            
            Task(
                description=f"""
                Identify trends and forecast future direction.
                
                For each key metric, determine:
                1. Is it trending up, down, or stable?
                2. Is the trend accelerating or slowing?
                3. What's likely to happen if trends continue?
                4. Any early warning signs of problems?
                
                Focus on:
                - Fitness trends (VO2 max, training load adaptation)
                - Recovery trends (HRV, sleep quality over time)
                - Stress trends (accumulation or improvement)
                
                Provide forward-looking guidance.
                """,
                agent=self.trend_forecaster,
                expected_output="Trend analysis with predictions and early warnings"
            ),
            
            Task(
                description=f"""
                Identify the biggest weaknesses and opportunities.
                
                Find:
                1. What's the limiting factor?
                   - Is it sleep quality?
                   - Is it accumulated stress?
                   - Is it inadequate recovery?
                   - Is it training consistency?
                
                2. What would have the biggest impact if improved?
                
                3. Prioritize recommendations:
                   - Quick wins (can improve immediately)
                   - Medium-term improvements (2-4 weeks)
                   - Long-term developments (1-3 months)
                
                Be specific and prioritized.
                """,
                agent=self.weakness_identifier,
                expected_output="Prioritized list of weaknesses and improvement opportunities"
            )
        ]
        
        return tasks
    
    def create_weekly_summary_tasks(self, matrix_context: str = "",
                                     comparison_context: str = "") -> List[Task]:
        """Create tasks for weekly summary report.
        If matrix_context is provided, agents receive pre-computed
        correlation data so they don't need to query for it.
        If comparison_context is provided, the Matrix Comparator agent
        receives week-over-week comparison data.

        Uses 9 agents: matrix_analyst, matrix_comparator,
        pattern_detective, performance_optimizer, recovery_specialist,
        trend_forecaster, lifestyle_analyst, sleep_analyst,
        weakness_identifier."""

        corr_block = ""
        if matrix_context:
            corr_block = (
                "\n\nPRE-COMPUTED CORRELATION ANALYSIS (use this data — it is "
                "comprehensive and saves tokens vs raw SQL queries):\n"
                f"{matrix_context}\n"
            )

        comparison_block = ""
        if comparison_context:
            comparison_block = (
                "\n\nWEEK-OVER-WEEK COMPARISON DATA:\n"
                f"{comparison_context}\n"
            )

        # ── Shared analytical rules that go into every task ──
        analysis_rules = (
            "\n\n══════ ANALYTICAL RULES (MUST FOLLOW) ══════\n"
            "1. SAMPLE SIZE: Any correlation or prediction with n<20 is PRELIMINARY.\n"
            "   Say so explicitly. Do NOT build recommendations on n<10 findings.\n"
            "2. OUTLIER CHECK: Before calling something a 'trend', check if removing\n"
            "   the single best and worst day changes the conclusion. If it does,\n"
            "   it's an outlier effect, not a trend. Report both the raw average and\n"
            "   the average excluding the outlier.\n"
            "3. DATA QUALITY: A value of 0 for rem_sleep_sec or deep_sleep_sec,\n"
            "   or a sleep_score below 40, likely indicates a tracking failure\n"
            "   (watch removed during sleep, sensor malfunction). Flag it as\n"
            "   SUSPECT DATA — do not treat it as a real physiological event.\n"
            "4. METRIC DISAMBIGUATION:\n"
            "   - tr_acute_load (DB) → displayed as 'training_acute_load' = Garmin's\n"
            "     readiness-derived EPOC accumulation (changes even on rest days)\n"
            "   - daily_load_acute = 7-day rolling training load from Garmin bulk data\n"
            "   - training_load (activities table) = per-workout EPOC score\n"
            "   These are THREE DIFFERENT metrics. Never conflate them.\n"
            "5. DAY-BY-DAY VERIFICATION: When citing a correlation (e.g. r=-0.97),\n"
            "   query the actual daily values and check: does the relationship\n"
            "   hold when you look at specific day pairs? Name the days.\n"
            "6. BOUNCE-BACK MATTERS: After a hard day, look at what happened NEXT.\n"
            "   If HRV/RHR/stress normalized within 24h, that's a POSITIVE signal\n"
            "   (good recovery capacity), not a negative one.\n"
            "7. VARIANCE vs TREND: High day-to-day swings (e.g., sleep bouncing\n"
            "   52→89→72→83→70) is VARIABILITY, not a decline. Report the range\n"
            "   and standard deviation, not just the average.\n"
            "8. USE ALL AVAILABLE DATA: You have ACWR, daily_load_acute,\n"
            "   daily_load_chronic, vo2_max_running, race_time_5k. Analyze them.\n"
            "   Don't ignore new columns just because they weren't there before.\n"
            "9. POSITIVE FINDINGS: Not everything is a problem. If RHR is improving,\n"
            "   HRV is up, or ACWR is in the optimal zone — say so prominently.\n"
            "10. NO GENERIC ADVICE: Every recommendation must cite a specific\n"
            "    correlation, day pair, or metric value that supports it.\n"
            "    'Get better sleep' is not a recommendation. 'Your REM variance\n"
            "    (0-108 min, CV=58%%) suggests inconsistent sleep timing'\n"
            "    is a recommendation.\n"
            "══════════════════════════════════════════════\n"
        )

        db_hint = (
            "\n\nDATABASE COLUMN REFERENCE — daily_metrics table\n"
            "(use ONLY these exact column names in SQL queries):\n\n"
            "  Heart & HRV:\n"
            "    resting_hr — resting heart rate (bpm, lower=fitter)\n"
            "    hrv_last_night — overnight HRV in ms (higher=better recovery)\n"
            "    resting_hr_sleep — HR during sleep\n\n"
            "  Stress:\n"
            "    stress_level — daily avg stress 0-100 (lower=calmer)\n\n"
            "  Sleep:\n"
            "    sleep_score — Garmin sleep quality 0-100\n"
            "    sleep_seconds, deep_sleep_sec, rem_sleep_sec, light_sleep_sec\n"
            "    avg_respiration — breaths/min during sleep\n"
            "    body_battery_change — BB gained during sleep\n\n"
            "  Body Battery:\n"
            "    bb_charged — total energy recharged (higher=better rest)\n"
            "    bb_drained — total energy spent (proxy for daily load)\n"
            "    bb_peak, bb_low — daily max/min BB level\n\n"
            "  Activity & Steps:\n"
            "    total_steps, moderate_intensity_min, vigorous_intensity_min\n\n"
            "  Training Readiness:\n"
            "    training_readiness — Garmin readiness score 0-100\n"
            "    tr_acute_load — short-term training load (EPOC)\n\n"
            "  Cardio Fitness (from Garmin bulk data):\n"
            "    vo2_max_running — VO2Max for running (mL/kg/min, higher=fitter)\n"
            "    vo2_max_running_delta — 7-day change in VO2Max (positive=improving)\n"
            "    vo2_max_cycling — VO2Max for cycling\n"
            "    race_time_5k — predicted 5K time in seconds (lower=faster)\n"
            "    race_time_10k — predicted 10K time in seconds\n"
            "    race_time_5k_delta — 7-day change (negative=getting faster)\n\n"
            "  Training Load:\n"
            "    daily_load_acute — short-term load (7-day)\n"
            "    daily_load_chronic — long-term load (28-day)\n"
            "    acwr — acute:chronic workload ratio (<0.8 detraining, "
            "0.8-1.3 optimal, >1.5 injury risk)\n\n"
            "  Environment:\n"
            "    heat_acclimation_pct — heat adaptation 0-100%\n"
            "    altitude_acclimation — altitude adaptation score\n\n"
            "  Weight: weight_kg\n"
            "  Hydration: hydration_value_ml\n\n"
            "  activities table columns:\n"
            "    activity_id, activity_type, distance_m, duration_sec,\n"
            "    average_hr, max_hr, calories, elevation_gain_m,\n"
            "    aerobic_training_effect (1.0-5.0), "
            "anaerobic_training_effect (1.0-5.0),\n"
            "    training_load (EPOC-based), vo2_max_value (per activity),\n"
            "    avg_stride_length, sport_type\n\n"
            "  NOTE: There is NO column named hrv_ms, stress_avg, "
            "sleep_hours, body_battery_max, or training_load_balance.\n"
        )

        ctx = f"{corr_block}{analysis_rules}{db_hint}"

        return [
            Task(
                description=f"""
                You are given a full pre-computed correlation analysis.
                Your job is to interpret EVERY section of it — skip NOTHING.

                Go through each section in order and for each one report:
                - SECTION NAME
                - KEY FINDINGS (the 2-3 most important numbers)
                - SAMPLE SIZE & CONFIDENCE
                - WHAT IT MEANS IN PLAIN LANGUAGE

                MANDATORY SECTIONS (you must cover ALL of these):
                1. Training Intensity Classification
                2. Same-Day Correlations (top 5 most meaningful pairs)
                3. Next-Day Predictors (top 5 — these show causality)
                4. Autoregressive Persistence (which metrics persist day-to-day)
                5. Distributions (normal vs skewed — affects which stats are valid)
                6. Recent Anomalies (what's unusual in the last 3 days)
                7. Conditioned AR(1) — THIS IS THE MOST IMPORTANT SECTION.
                   For each model listed, explain:
                   - What variables predict what
                   - The R² and how much it improves over simple persistence
                   - What this means practically
                   - Sample size and whether it's reliable
                8. Markov State Transitions (state stickiness + sample sizes)
                9. KL-Divergence (which conditions shift predictions)
                10. Metric Ranges (current baselines and context)

                DO NOT give advice or recommendations.
                DO NOT query the database.
                ONLY interpret the numbers.
                {ctx}
                """,
                agent=self.matrix_analyst,
                expected_output="Section-by-section interpretation of the full correlation matrix with confidence ratings"
            ),

            Task(
                description=f"""
                Perform a CROSS-TIMEFRAME STABILITY ANALYSIS using the
                benchmark comparison data below. You have full correlation
                matrices for EVERY available time window (the number of
                windows depends on how much data exists — could be 2-9
                windows ranging from 7 days to 365 days).

                {comparison_block}

                REQUIRED SECTIONS:

                1. CORRELATION EVOLUTION: For the top Pearson pairs, track
                   r values across ALL available windows (7d, 14d, 21d, 30d, etc).
                   Show them in a table:
                   | Metric Pair | 7d r | 14d r | 21d r | 30d r | Verdict |
                   Classify: ROBUST, STABLE, EMERGING, WEAKENING, SPARSE.

                2. PREDICTOR STABILITY: Same for next-day predictors.
                   Track r and n across windows. Rising n + stable r = reliable.

                3. PERSISTENCE EVOLUTION: AR(1) R² across windows.
                   | Metric | 7d R² | 14d R² | 21d R² | 30d R² | Verdict |

                4. MARKOV & KL MATURATION: Note which window first produces
                   Markov/KL results. Below 20 transitions = PRELIMINARY.
                   Are KL findings consistent once they appear?

                5. METRIC RANGES ACROSS WINDOWS: For key metrics, show how
                   mean and std change as the window grows. Recent-window
                   means that differ from long-window means indicate trends.

                6. CONFIDENCE SUMMARY: For every major finding, assign:
                   ROBUST / STABLE / EMERGING / LONGER-WINDOW-ONLY / SPARSE

                If a window has too few data points for a section, say
                'DATA SPARSITY' — don't say the pattern 'disappeared'.
                {ctx}
                """,
                agent=self.matrix_comparator,
                expected_output="Cross-timeframe stability analysis with per-window numbers and confidence tiers"
            ),

            Task(
                description=f"""
                Find 3-5 non-obvious patterns in this week's data.
                The Matrix Analyst has already interpreted the correlation
                tables — do NOT repeat that analysis. Focus on DAY-BY-DAY
                patterns in the actual data that go BEYOND what correlations show.

                REQUIRED STEPS:
                1. First, query the day-by-day values for this week:
                   SELECT date, resting_hr, hrv_last_night, sleep_score,
                   stress_level, training_readiness, bb_charged, bb_drained,
                   total_steps, rem_sleep_sec, deep_sleep_sec, acwr,
                   daily_load_acute, daily_load_chronic
                   FROM daily_metrics WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                   ORDER BY date

                2. FLAG any suspect data: rem_sleep=0, sleep_score<40, or
                   any metric that's a clear outlier (>2σ from the week's mean).
                   Mark these as SUSPECT and note how they affect averages.

                3. Look at DAY PAIRS (what happened on day X → day X+1):
                   - After the hardest training day, did recovery metrics
                     bounce back the next day? How quickly?
                   - After the worst sleep night, what happened to stress
                     and readiness the next day?

                4. Examine the pre-computed correlations, but VERIFY each one
                   by checking the n (sample size) and whether the actual
                   day-by-day data supports the claimed relationship.

                5. Report each pattern as:
                   PATTERN: [name]
                   EVIDENCE: [specific days and numbers]
                   SAMPLE SIZE: n=[number]
                   CONFIDENCE: [HIGH if n>=20, PRELIMINARY if n<20, SUSPECT if data quality issues]
                   SO WHAT: [what this means for the user in plain language]
                {ctx}
                """,
                agent=self.pattern_detective,
                expected_output="3-5 patterns with day-by-day evidence and confidence ratings"
            ),

            Task(
                description=f"""
                Create a week-over-week performance comparison.

                REQUIRED FORMAT — for each metric, report:
                | Metric | Last Week | This Week | Change | Verdict |

                REQUIRED STEPS:
                1. Query both weeks' daily values (not just averages):
                   SELECT date, resting_hr, hrv_last_night, sleep_score,
                   stress_level, training_readiness, bb_charged, bb_drained, total_steps
                   FROM daily_metrics
                   WHERE date >= CURRENT_DATE - INTERVAL '14 days'
                   ORDER BY date

                2. For each metric, report:
                   - Raw averages for both weeks
                   - If any day is a clear outlier (>2σ), compute the average
                     WITH and WITHOUT that day. Show both.
                   - Day range (min-max) for this week to show variability
                   - Verdict: IMPROVED (>10%% better), STABLE (<10%% change),
                     DECLINED (>10%% worse), or VOLATILE (high variance,
                     trend unclear)

                3. Highlight the BEST single day and WORST single day this week.
                   What made them different?

                4. ACWR and training load analysis:
                   - What is the current ACWR? Where does it fall
                     (optimal/detraining/injury risk)?
                   - Show how daily_load_acute and daily_load_chronic
                     are trending.

                5. Activities summary: how many workouts, what types,
                   total training load this week vs last.
                {ctx}
                """,
                agent=self.performance_optimizer,
                expected_output="Week-over-week table with outlier-adjusted averages and variability analysis"
            ),

            Task(
                description=f"""
                Assess recovery capacity — not just current state,
                but HOW WELL the body recovers from hard efforts.

                REQUIRED STEPS:
                1. Identify the 2-3 hardest days this week (by training_load
                   from activities table or bb_drained from daily_metrics).

                2. For each hard day, analyze the NEXT-DAY response:
                   - Did HRV go up or down?
                   - Did resting HR go up or down?
                   - Did bb_charged recover?
                   - How was the sleep_score?
                   Format: "Feb X (HARD): bb_drained=Y → Feb X+1: HRV=Z (+/-),
                   RHR=W (+/-), bb_charged=V"

                3. OVERTRAINING CHECKLIST (answer each yes/no with evidence):
                   □ Is resting HR trending upward over 5+ consecutive days?
                   □ Is HRV trending downward over 5+ consecutive days?
                   □ Is sleep quality declining over 5+ consecutive days?
                   □ Is the user reporting higher stress levels consistently?
                   □ Is ACWR > 1.5?
                   Overtraining requires YES on 3+ of these. Otherwise, NOT overtraining.

                4. Based on the bounce-back analysis, what is the user's
                   recovery SPEED? Fast (<24h to normalize), moderate (24-48h),
                   or slow (>48h)?

                5. Intensity recommendation with SPECIFIC reasoning.
                {ctx}
                """,
                agent=self.recovery_specialist,
                expected_output="Bounce-back analysis for each hard day + overtraining checklist + recovery speed assessment"
            ),

            Task(
                description=f"""
                Analyze trends RIGOROUSLY — only report trends that survive scrutiny.

                REQUIRED STEPS:
                For each of these metrics: HRV, resting HR, sleep_score,
                stress_level, training_readiness, ACWR:

                1. List the daily values for the last 7 days in a row.
                2. Compute: mean, std, coefficient of variation (CV = std/mean).
                3. Is there a DIRECTIONAL trend (3+ consecutive days moving
                   in the same direction)? Or is it oscillating?
                4. OUTLIER TEST: remove the single highest and lowest values.
                   Does the "trend" still hold? If not, it's variability.
                5. Markov transition relevance: cite the transition probabilities
                   BUT note the sample size (number of transitions observed).
                   If <10 transitions, call it unreliable.

                REPORT FORMAT per metric:
                   [METRIC_NAME]
                   Values: [day1, day2, ... day7]
                   Mean: X, Std: Y, CV: Z%%
                   Directional trend: YES (direction) / NO (oscillating)
                   Survives outlier removal: YES / NO
                   Verdict: IMPROVING / DECLINING / STABLE / VOLATILE

                Finally, list the TOP 2 EARLY WARNINGS (things that could
                worsen if unchecked) and TOP 2 POSITIVE SIGNALS.
                {ctx}
                """,
                agent=self.trend_forecaster,
                expected_output="Per-metric trend table with CV, outlier test, and directional analysis"
            ),

            Task(
                description=f"""
                Connect specific activities and behaviors to next-day outcomes.

                REQUIRED STEPS:
                1. Query this week's activities:
                   SELECT date, activity_type, sport_type,
                   duration_sec, average_hr, max_hr, training_load
                   FROM activities WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                   ORDER BY date

                2. For each training day, build a DAY STORY:
                   "Feb X: [what they did] → Feb X+1: [how body responded]"
                   Include specific numbers for HRV, RHR, sleep_score, stress, BB.

                3. Classify activities properly:
                   - Short cycling (5-10 min, low HR) = commute, not training
                   - Running 30+ min at HR>150 = hard cardio
                   - Strength 60+ min = hard resistance
                   - Walking = active recovery
                   Don't treat a 7-min bike commute as a "cycling workout."

                4. ACWR trajectory: show the daily ACWR values and explain
                   what the trend means (ramping up? pulling back? stable?).

                5. Use KL-divergence findings to answer: which CONDITIONS
                   (e.g., high steps, low stress) actually SHIFT the
                   probability of good outcomes? Cite the specific KL values
                   and what they mean.

                6. State clearly what data you DON'T have (time-of-day,
                   nutrition, caffeine, alcohol) and don't speculate about it.
                {ctx}
                """,
                agent=self.lifestyle_analyst,
                expected_output="Day-by-day activity stories with next-day biometric responses"
            ),

            Task(
                description=f"""
                Perform a DEDICATED SLEEP ANALYSIS for this week.

                REQUIRED STEPS:
                1. Query sleep data for the last 7 days:
                   SELECT date, sleep_score, sleep_seconds,
                   deep_sleep_sec, rem_sleep_sec, light_sleep_sec,
                   avg_respiration, body_battery_change,
                   resting_hr_sleep
                   FROM daily_metrics
                   WHERE date >= CURRENT_DATE - INTERVAL '7 days'
                   ORDER BY date

                2. ARCHITECTURE TABLE — for each night:
                   | Date | Score | Total hrs | Deep min (%%) | REM min (%%) | Light min (%%) | Resp | BB Change |
                   Flag any night with deep=0, REM=0, or score<40 as SUSPECT.

                3. AVERAGES — compute with and without suspect nights:
                   - Mean sleep_score, total hours, deep%%, REM%%, light%%
                   - Compare to norms: Deep 15-20%%, REM 20-25%%, Light 50-60%%
                   - Is the user getting enough deep sleep? Enough REM?

                4. CONSISTENCY:
                   - CV (std/mean) for sleep_score and total sleep time
                   - CV > 15%% = inconsistent — this is a problem even if
                     the average is fine. Report the spread (min→max).

                5. NEXT-DAY IMPACT — for each night, show:
                   "Night of [date]: score=X, deep=Y min →
                    Next day: HRV=A, stress=B, readiness=C"
                   Which sleep metric is the BEST predictor of next-day
                   readiness? Is it total time, deep%%, REM%%, or score?

                6. BODY BATTERY RECHARGE:
                   - body_battery_change during sleep — how much energy
                     was recovered? Does it correlate with sleep quality?
                   - Nights with low BB recharge despite decent sleep_score
                     suggest other factors (stress, late meals, timing).

                7. RESPIRATION: any nights with avg_respiration > 16?
                   Elevated respiratory rate during sleep can indicate
                   illness, nasal congestion, or poor sleep environment.

                8. Use any sleep-related Markov/KL findings from the
                   correlation matrix to answer: what CONDITIONS lead to
                   good vs poor sleep? (e.g., high steps, low stress,
                   specific training patterns)

                9. WEEK-OVER-WEEK: Query last week's sleep for comparison:
                   SELECT date, sleep_score, sleep_seconds, deep_sleep_sec,
                   rem_sleep_sec FROM daily_metrics
                   WHERE date >= CURRENT_DATE - INTERVAL '14 days'
                   AND date < CURRENT_DATE - INTERVAL '7 days'
                   ORDER BY date
                   Compare this week vs last week averages.

                10. TOP INSIGHT: What is the single most important sleep
                    finding this week? Support with specific numbers.
                {ctx}
                """,
                agent=self.sleep_analyst,
                expected_output="Sleep architecture table + consistency analysis + next-day impact mapping + week-over-week comparison"
            ),

            Task(
                description=f"""
                Synthesize all previous findings into the #1 bottleneck
                and top 3 actionable quick wins.

                REQUIRED STEPS:
                1. FACT-CHECK previous agents: Did they flag outliers?
                   Did they note sample sizes? Did they conflate metrics?
                   If any agent made an error, correct it here.

                2. BOTTLENECK IDENTIFICATION:
                   - List the top 3 candidate bottlenecks with supporting data.
                   - For each, assess: is the data STRONG (n>=20, consistent
                     daily pattern) or WEAK (n<20, outlier-driven, volatile)?
                   - Pick the one with the strongest evidence.
                   - If the data is too sparse to confidently pick a bottleneck,
                     say so. "Insufficient data to determine" is a valid answer.

                3. QUICK WINS — each must have:
                   - THE SPECIFIC metric you expect to improve
                   - THE EVIDENCE (correlation value, day-pair example,
                     KL-divergence shift) that supports this action
                   - THE CONFIDENCE LEVEL (high/medium/low based on n)
                   - HOW to verify it worked (what to look for next week)

                4. WHAT'S ALREADY WORKING: name 2-3 things going well
                   that should be continued. Not everything is broken.

                5. HONEST LIMITATIONS: what can't we determine yet?
                   (too few days of data, missing VO2max history,
                   no nutrition data, etc.)
                {ctx}
                """,
                agent=self.weakness_identifier,
                expected_output="Fact-checked synthesis with confidence-rated bottleneck + evidence-based quick wins + honest limitations"
            ),
        ]
    
    def create_goal_progress_tasks(self, goal_description: str) -> List[Task]:
        """Create tasks for tracking progress toward specific goals"""
        
        return [
            Task(
                description=f"""
                Analyze progress toward this goal: {goal_description}
                
                Assess:
                1. What metrics are relevant to this goal?
                2. Are those metrics improving?
                3. What's working well?
                4. What's not working?
                5. What adjustments are needed?
                
                Provide specific progress update with data.
                """,
                agent=self.performance_optimizer,
                expected_output="Goal progress report with specific metrics"
            ),
            
            Task(
                description=f"""
                Based on the goal: {goal_description}
                
                Recommend specific actions for the next 2 weeks to accelerate progress.
                Be concrete and measurable.
                """,
                agent=self.weakness_identifier,
                expected_output="2-week action plan for goal achievement"
            )
        ]
    
    def run_comprehensive_analysis(self, analysis_period: int = 30) -> str:
        """Run full analysis with all agents"""
        
        log.info("\n" + "="*60)
        log.info("🤖 RUNNING COMPREHENSIVE HEALTH ANALYSIS")
        log.info("   This will take a few minutes...")
        log.info("="*60 + "\n")
        
        tasks = self.create_deep_analysis_tasks(analysis_period)
        
        crew = Crew(
            agents=[
                self.pattern_detective,
                self.performance_optimizer,
                self.recovery_specialist,
                self.lifestyle_analyst,
                self.trend_forecaster,
                self.weakness_identifier
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        
        log.info("\n" + "="*60)
        log.info("✅ ANALYSIS COMPLETE")
        log.info("="*60 + "\n")
        
        return result
    
    def run_weekly_summary(self, matrix_context: str = "",
                            comparison_context: str = "") -> str:
        """Generate weekly summary report using 8 agents.
        Returns the COMBINED output from all 8 agents, not just the last one.
        """

        log.info("\n  Generating Weekly Summary (9 agents)…\n")

        tasks = self.create_weekly_summary_tasks(
            matrix_context=matrix_context,
            comparison_context=comparison_context,
        )

        agent_labels = [
            "CORRELATION MATRIX INTERPRETATION (Matrix Analyst)",
            "CROSS-TIMEFRAME STABILITY (Benchmark Analyst)",
            "HIDDEN PATTERNS (Pattern Detective)",
            "PERFORMANCE SUMMARY (Performance Optimizer)",
            "RECOVERY ASSESSMENT (Recovery Specialist)",
            "TRENDS & FORECASTS (Trend Forecaster)",
            "LIFESTYLE CONNECTIONS (Lifestyle Analyst)",
            "SLEEP ANALYSIS (Sleep Analyst)",
            "BOTTLENECK & QUICK WINS (Weakness Identifier)",
        ]

        crew = Crew(
            agents=[
                self.matrix_analyst,
                self.matrix_comparator,
                self.pattern_detective,
                self.performance_optimizer,
                self.recovery_specialist,
                self.trend_forecaster,
                self.lifestyle_analyst,
                self.sleep_analyst,
                self.weakness_identifier,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        # Combine ALL agent outputs, not just the last one
        sections = []
        try:
            for label, task_out in zip(agent_labels, result.tasks_output):
                raw = getattr(task_out, "raw", str(task_out))
                sections.append(f"{'='*60}\n  {label}\n{'='*60}\n\n{raw}")
        except Exception:
            # Fallback: if tasks_output isn't available, return raw result
            return str(result)

        combined = "\n\n".join(sections)
        log.info(f"\n  Combined output: {len(combined)} chars from {len(sections)} agents")
        return combined
    
    def run_goal_analysis(self, goal: str) -> str:
        """Analyze progress toward specific goal"""
        
        log.info(f"\n🎯 Analyzing progress toward: {goal}\n")
        
        tasks = self.create_goal_progress_tasks(goal)
        
        crew = Crew(
            agents=[self.performance_optimizer, self.weakness_identifier],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        return crew.kickoff()
