"""
Enhanced AI Agents for Health Analysis
=======================================
Advanced agents that provide insights Garmin Connect can't offer.
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pandas as pd
import psycopg2
import json
import os
import logging
from datetime import date, timedelta
from typing import Any, List
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("enhanced_agents")

# ג”€ג”€ Gemini LLM for all agents (lazyג€‘init to avoid import-time crashes) ג”€ג”€
_gemini_llm = None

def _get_llm() -> LLM:
    global _gemini_llm
    if _gemini_llm is None:
        _gemini_llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3,
        )
    return _gemini_llm


# ג”€ג”€ Standalone tools (CrewAI 1.9+ uses module-level functions) ג”€ג”€

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


@tool("Get Past Recommendations")
def get_past_recommendations(weeks: int = 4) -> str:
    """
    Retrieve past AI recommendations from the last N weeks.
    Shows what was recommended, the target metric, and current status.
    Use this to review what was suggested before and whether it worked.
    """
    weeks = max(1, int(weeks))
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        _ensure_agent_recommendations_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT week_date, agent_name, recommendation,
                       target_metric, expected_direction, status, outcome_notes
                FROM agent_recommendations
                WHERE week_date >= CURRENT_DATE - (%s * INTERVAL '1 week')
                ORDER BY week_date DESC, id
                """,
                (weeks,),
            )
            rows = cur.fetchall()
        df = pd.DataFrame(
            rows,
            columns=[
                "week_date",
                "agent_name",
                "recommendation",
                "target_metric",
                "expected_direction",
                "status",
                "outcome_notes",
            ],
        )
        conn.close()
        if df.empty:
            return "No past recommendations found. This is the first analysis."
        return df.to_string(index=False)
    except Exception as e:
        return f"No past recommendations available: {e}"


def _ensure_agent_recommendations_table(conn) -> None:
    # Table created by enhanced_schema.py via ensure_startup_schema()
    pass


def save_recommendations_to_db(recommendations: list, week_date=None):
    """
    Save parsed recommendations to agent_recommendations table.
    Called by the pipeline after Synthesizer output is parsed.

    Parameters
    ----------
    recommendations : list of dict
        Each dict has keys: recommendation, target_metric,
        expected_direction, agent_name (default: 'synthesizer')
    week_date : date, optional
        Defaults to start of current week.
    """
    if not recommendations:
        return
    if week_date is None:
        today = date.today()
        week_date = today - timedelta(days=today.weekday())

    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        conn.autocommit = True
        _ensure_agent_recommendations_table(conn)
        cur = conn.cursor()

        for rec in recommendations:
            cur.execute(
                "INSERT INTO agent_recommendations "
                "(week_date, agent_name, recommendation, target_metric, expected_direction) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    week_date,
                    rec.get('agent_name', 'synthesizer'),
                    rec['recommendation'],
                    rec.get('target_metric', ''),
                    rec.get('expected_direction', ''),
                ),
            )

        cur.close()
        conn.close()
        log.info("Saved %d recommendations to agent_recommendations", len(recommendations))
    except Exception as e:
        log.warning("Failed to save recommendations: %s", e)


def load_past_recommendations_context(weeks: int = 4) -> str:
    """
    Load past recommendations as a text block for agent context injection.
    Returns a formatted string or empty string if no recommendations.
    """
    weeks = max(1, int(weeks))
    try:
        conn = psycopg2.connect(os.getenv('POSTGRES_CONNECTION_STRING'))
        _ensure_agent_recommendations_table(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT week_date, recommendation, target_metric,
                       expected_direction, status, outcome_notes
                FROM agent_recommendations
                WHERE week_date >= CURRENT_DATE - (%s * INTERVAL '1 week')
                ORDER BY week_date DESC, id
                """,
                (weeks,),
            )
            rows = cur.fetchall()
        conn.close()
        if not rows:
            return ""
        lines = ["\n\nPAST RECOMMENDATIONS (last {} weeks):".format(weeks)]
        for row in rows:
            week_date, recommendation, target_metric, expected_direction, status, outcome = row
            status = status or 'pending'
            outcome = outcome or ''
            lines.append(
                f"  [{week_date}] {recommendation} "
                f"(target: {target_metric}, expected: {expected_direction}, "
                f"status: {status})"
            )
            if outcome:
                lines.append(f"    → Outcome: {outcome}")
        lines.append("")
        return "\n".join(lines)
    except Exception:
        return ""


class AdvancedHealthAgents:
    """
    Enhanced AI agents that provide insights beyond Garmin Connect.
    Fully standalone ג€” uses psycopg2 via POSTGRES_CONNECTION_STRING.
    """

    def __init__(self) -> None:
        self.tools = [
            run_sql_query,
            calculate_correlation,
            find_best_days,
            analyze_pattern,
            get_past_recommendations
        ]
        
        # ── 5 Consolidated Agents (merged from original 9 for cost efficiency) ──

        self.statistical_interpreter = Agent(
            role='Statistical Interpreter',
            goal='Interpret EVERY section of the pre-computed correlation data and assess cross-timeframe stability',
            backstory="""You are a statistics interpreter and longitudinal analysis expert.
            Your job has two parts:

            PART 1 — SINGLE-WINDOW INTERPRETATION:
            Read the full correlation analysis and translate it into clear findings.
            You MUST cover ALL sections — skip NONE:
            1. TRAINING INTENSITY CLASSIFICATION
            2. SAME-DAY CORRELATIONS — strongest Pearson pairs
            3. NEXT-DAY PREDICTORS — what yesterday predicts
            4. AUTOREGRESSIVE PERSISTENCE — which metrics are "sticky"
            5. DISTRIBUTIONS — normal vs skewed
            6. RECENT ANOMALIES — z-scores from last 3 days
            7. CONDITIONED AR(1) — which variable COMBINATIONS predict
               next-day outcomes, R² improvement over simple persistence
            8. MARKOV STATE TRANSITIONS — state stickiness + sample sizes
            9. KL-DIVERGENCE — which conditions shift predictions
            10. METRIC RANGES — current baselines

            PART 2 — CROSS-TIMEFRAME STABILITY (if multi-window data provided):
            Compare across time windows and classify:
            ROBUST = stable across ALL windows
            STABLE = appears at 2+ adjacent windows
            EMERGING = only at shortest window
            SPARSE = too few data points
            Correlations that STRENGTHEN as n grows = trustworthy.
            Correlations that WEAKEN as n grows = likely spurious.

            For each section, report:
            - KEY FINDING (1-2 sentences)
            - CONFIDENCE (based on n and statistical significance)
            - PRACTICAL MEANING (no jargon)

            You do NOT give lifestyle advice. You ONLY interpret numbers.
            Flag any finding with n<20 as PRELIMINARY.
            Flag any metric with CV>20%% as VOLATILE.

            INVESTIGATION RULE: For every anomaly or strong correlation you
            report, note what the surface inputs (Layer 2: stress, workload,
            BB drain) and underlying mechanics (Layer 3: sleeping HR,
            deep sleep, respiration, HRV) say about it. Flag any
            contradictions between surface metrics and deeper mechanics.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=_get_llm()
        )

        self.health_pattern_analyst = Agent(
            role='Health Pattern & Trend Analyst',
            goal='Discover hidden patterns and rigorously assess trends — only report findings that survive scrutiny',
            backstory="""You find non-obvious patterns AND rigorously evaluate trends.

            PATTERN DETECTION RULES:
            - Check sample size (n). If n<20, mark as PRELIMINARY.
            - Look at actual day-by-day values, not just aggregates.
              If a "trend" is driven by a single outlier, say so.
            - Data quality: 0 for rem_sleep_sec/hrv_last_night or
              sleep_score < 40 = likely tracking failure. Flag these.
            - Distinguish metrics: tr_acute_load, daily_load_acute,
              and training_load (per-activity) are THREE DIFFERENT things.
            - When citing a correlation, verify against day-by-day data.

            TREND ANALYSIS RULES:
            - A "trend" needs 3+ consecutive points in same direction.
            - High VARIANCE is not a TREND. Bouncing = variability.
            - Compute CV (std/mean). CV > 20%% = volatile.
            - Check if trend survives removing best and worst days.
            - Use "may" and "suggests" with limited data.

            Report each finding as:
            FINDING: [name]
            EVIDENCE: [specific days and numbers]
            SAMPLE SIZE: n=[number]
            CONFIDENCE: HIGH / PRELIMINARY / SUSPECT
            VERDICT: IMPROVING / DECLINING / STABLE / VOLATILE

            INVESTIGATION RULE: When identifying patterns, always check if
            surface-level patterns (stress, workload) are confirmed or
            contradicted by underlying mechanics (sleeping HR, deep sleep,
            respiration). If they contradict, dig deeper and report both.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=_get_llm()
        )

        self.performance_recovery = Agent(
            role='Performance & Recovery Analyst',
            goal='Compare performance week-over-week AND assess recovery capacity with honest evidence',
            backstory="""You analyze both performance trends and recovery quality.

            PERFORMANCE RULES:
            - Report ACTUAL numbers (this week avg vs last week avg).
            - If average is dragged by one bad day, compute WITH and WITHOUT.
            - Change <5%% = "stable". Only >10%% warrants attention.
            - Check if changes are sustained or volatile.
            - Report BEST and WORST day — spread matters more than average.

           RECOVERY RULES:
            - After hard days: does HRV bounce back? Does RHR normalize?
              Bounce-back speed > any single metric.
            - HIGH LOAD ALERT: If a specific day had very high load (e.g. >1.5x average or 'HARD'),
              assume it likely impacted sleep/recovery even if the general correlation is weak.
              Check that specific day's next-day sleep score explicitly.
            - Don't call "overtraining" unless MULTIPLE metrics decline
              simultaneously over MULTIPLE consecutive days.
            - ACWR: 0.8-1.3 = optimal, <0.8 = detraining, >1.5 = injury risk.
            - Training readiness scale (use exact buckets):
              0-25 = poor (red), 26-50 = low (orange), 51-75 = moderate (green),
              76-95 = high (blue), 96-100 = effective (purple).
            - bb_charged depends on sleep AND stress, not just training.
            - Overtraining needs convergent evidence: RHR up + HRV down +
              sleep worse. If only 1 of 3, NOT overtraining.
            - Report next-day response after hard days with specific numbers.
            - Recovery speed: Fast (<24h), Moderate (24-48h), Slow (>48h).

            INVESTIGATION RULE: Rest days must be validated — check stress,
            HRV, sleeping HR, not just the absence of activity. A hard
            workout day followed by excellent sleeping HR tells a different
            recovery story than one followed by poor sleeping HR. Always
            check the underlying mechanics before labeling recovery speed.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=_get_llm()
        )

        self.sleep_lifestyle = Agent(
            role='Sleep & Lifestyle Analyst',
            goal='Deep-dive into sleep architecture AND connect daily activities to biometric outcomes',
            backstory="""You cover sleep quality and lifestyle impact.

            SLEEP RULES:
            - Report architecture: deep/rem/light as minutes AND %%.
              Norms: Deep 15-20%%, REM 20-25%%, Light 50-60%%.
            - Consistency (CV) > single nights. CV > 15%% = inconsistent.
            - SUSPECT DATA: rem=0, deep=0, score<40, sleep<4hrs.
              Report averages WITH and WITHOUT suspect nights.
            - Show NEXT-DAY IMPACT per night with specific numbers.
            - avg_respiration > 16/min during sleep = flag.
            - Compare this week vs last week sleep metrics.

            LIFESTYLE RULES:
            - Use activities table for actual data (type, duration, HR).
            - Classify properly: 7-min bike commute != cycling workout.
            - Build DAY STORIES: "Feb X: [activity] -> Feb X+1: [response]"
            - Use KL-divergence for which conditions shift transitions.
            - If data doesn't exist (caffeine, alcohol), say "not available".
            - Every finding must cite specific numbers.

            INVESTIGATION RULE: Always decompose sleep quality through
            all three layers. Surface factors (stress, BB drain, hard
            workout) may suggest bad sleep, but underlying mechanics
            (sleeping HR, respiration rate, sleep architecture percentages)
            tell the true story. When they contradict, report the full
            picture and let the deeper metrics take priority.""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=_get_llm()
        )

        self.synthesizer = Agent(
            role='Synthesizer & Fact-Checker',
            goal='Find the #1 real bottleneck and give brutally specific, data-backed recommendations',
            backstory="""You synthesize all previous agents' findings and identify
            what matters most.

            CRITICAL RULES:
            - Recommendations must be DIRECTLY supported by correlation data
              or day-by-day patterns. No generic wellness advice.
            - If previous agents disagreed or made errors, call it out.
            - Distinguish correlation from causation. r=0.97 with n=5(!) = weak.
            - For each quick win, state the SPECIFIC metric you expect to
              improve and by HOW MUCH based on the data.
            - Call out what's ALREADY going well. Not everything needs fixing.
            - If the biggest issue is data quality, say that first.

            INVESTIGATION RULE: When previous agents claim a causal
            relationship, verify they checked all three layers:
            Layer 1 (outcome), Layer 2 (surface inputs), Layer 3
            (underlying mechanics). If they stopped at Layer 2 without
            checking Layer 3, call it out. Never let a surface-level
            explanation stand when deeper mechanics tell a different story.

            LONG-TERM MEMORY RULES:
            - ALWAYS check past recommendations using the 'Get Past Recommendations' tool.
            - If a recommendation from a previous week was followed and the target
              metric improved, REINFORCE it ("keep doing X — it's working").
            - If a recommendation was followed but the metric didn't improve,
              REVISE it with a new approach or acknowledge it may take more time.
            - If a recommendation was ignored, decide if it's still relevant.
            - Reference specific past recommendations by date and content.
            - Structure each new recommendation as:
              RECOMMENDATION: [what to do]
              TARGET_METRIC: [which metric should change]
              EXPECTED_DIRECTION: [IMPROVE/DECLINE/STABLE]""",
            verbose=True,
            allow_delegation=False,
            tools=self.tools,
            llm=_get_llm()
        )

        # ── Backward-compatible aliases for dashboard/weekly_sync ──
        self.matrix_analyst = self.statistical_interpreter
        self.matrix_comparator = self.statistical_interpreter
        self.pattern_detective = self.health_pattern_analyst
        self.trend_forecaster = self.health_pattern_analyst
        self.performance_optimizer = self.performance_recovery
        self.recovery_specialist = self.performance_recovery
        self.lifestyle_analyst = self.sleep_lifestyle
        self.sleep_analyst = self.sleep_lifestyle
        self.weakness_identifier = self.synthesizer

    def create_deep_analysis_tasks(self, analysis_period: int = 30) -> List[Task]:
        """Create comprehensive analysis tasks (4 agents)"""

        tasks = [
            Task(
                description=f"""
                Analyze the last {analysis_period} days to find hidden patterns
                AND rigorously evaluate trends.

                PATTERNS:
                1. Look for unexpected correlations (stress vs sleep, HRV vs readiness)
                2. Day-of-week patterns, delayed correlations
                3. Verify each pattern against day-by-day data

                TRENDS — for HRV, resting HR, sleep_score, stress, readiness, ACWR:
                1. List daily values, compute mean/std/CV
                2. Is there a directional trend (3+ consecutive days)?
                3. Does the trend survive removing best/worst days?
                4. Verdicts: IMPROVING / DECLINING / STABLE / VOLATILE

                Be specific with data evidence.
                """,
                agent=self.health_pattern_analyst,
                expected_output="3-5 hidden patterns + per-metric trend analysis with confidence"
            ),

            Task(
                description=f"""
                Week-over-week performance comparison + recovery assessment.

                PERFORMANCE:
                1. Key metrics this week vs last week (averages + ranges)
                2. Best/worst days and what made them different
                3. ACWR and training load analysis

                RECOVERY:
                1. Find 2-3 hardest days (by training_load or bb_drained)
                2. Analyze next-day response for each (HRV, RHR, BB, sleep)
                3. Overtraining checklist (RHR up? HRV down? Sleep worse?)
                4. Recovery speed assessment
                """,
                agent=self.performance_recovery,
                expected_output="Week-over-week table + recovery bounce-back analysis"
            ),

            Task(
                description=f"""
                Sleep architecture deep-dive + lifestyle-to-outcome connections.

                SLEEP:
                1. Architecture breakdown per night (deep/REM/light as %%)
                2. Consistency (CV for sleep_score and duration)
                3. Next-day impact per night
                4. Flag suspect data (score<40, rem=0)

                LIFESTYLE:
                1. Query activities table for this period
                2. Build day stories: activity -> next-day biometric response
                3. Classify activities properly (commute vs training)
                4. What data is NOT available (caffeine, alcohol, etc.)
                """,
                agent=self.sleep_lifestyle,
                expected_output="Sleep architecture + lifestyle impact analysis"
            ),

            Task(
                description=f"""
                Synthesize all findings into the #1 bottleneck and top 3 quick wins.

                1. Fact-check previous agents
                2. Identify bottleneck with strongest evidence
                3. Quick wins with specific metrics and confidence levels
                4. What's already working well
                5. Honest limitations
                """,
                agent=self.synthesizer,
                expected_output="Prioritized bottleneck + evidence-based quick wins"
            )
        ]

        return tasks

    def _get_past_recs_context(self) -> str:
        """Load past recommendations for injection into agent task descriptions."""
        return load_past_recommendations_context(weeks=4)

    def create_weekly_summary_tasks(self, matrix_context: str = "",
                                     comparison_context: str = "") -> List[Task]:
        """Create tasks for weekly summary report.
        If matrix_context is provided, agents receive pre-computed
        correlation data so they don't need to query for it.
        If comparison_context is provided, the Matrix Comparator agent
        receives week-over-week comparison data.

        Uses 5 consolidated agents: statistical_interpreter,
        health_pattern_analyst, performance_recovery,
        sleep_lifestyle, synthesizer."""

        corr_block = ""
        if matrix_context:
            corr_block = (
                "\n\nPRE-COMPUTED CORRELATION ANALYSIS (use this data ג€” it is "
                "comprehensive and saves tokens vs raw SQL queries):\n"
                f"{matrix_context}\n"
            )

        comparison_block = ""
        if comparison_context:
            # Extract 'same_week_last_year' if present
            import re
            swly_block = ""
            match = re.search(r"\[SAME WEEK LAST YEAR\](.*?)(\n\[|$)", comparison_context, re.DOTALL)
            if match:
                swly_block = f"\n\nSAME WEEK LAST YEAR BENCHMARK:\n{match.group(1).strip()}\n"
            comparison_block = (
                "\n\nWEEK-OVER-WEEK COMPARISON DATA:\n"
                f"{comparison_context}\n"
                f"{swly_block}"
            )

        # == Shared analytical rules that go into every task ==
        analysis_rules = (
            "\n\n====== ANALYTICAL RULES (MUST FOLLOW) ======\n"
            "1. SAMPLE SIZE: Any correlation or prediction with n<20 is PRELIMINARY.\n"
            "   Say so explicitly. Do NOT build recommendations on n<10 findings.\n"
            "2. OUTLIER CHECK: Before calling something a 'trend', check if removing\n"
            "   the single best and worst day changes the conclusion. If it does,\n"
            "   it's an outlier effect, not a trend. Report both the raw average and\n"
            "   the average excluding the outlier.\n"
            "3. DATA QUALITY: A value of 0 for rem_sleep_sec or deep_sleep_sec,\n"
            "   or a sleep_score below 40, likely indicates a tracking failure\n"
            "   (watch removed during sleep, sensor malfunction). Flag it as\n"
            "   SUSPECT DATA -- do not treat it as a real physiological event.\n"
            "4. METRIC DISAMBIGUATION:\n"
            "   - tr_acute_load (DB) = displayed as 'training_acute_load' = Garmin's\n"
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
            "   52->89->72->83->70) is VARIABILITY, not a decline. Report the range\n"
            "   and standard deviation, not just the average.\n"
            "8. USE ALL AVAILABLE DATA: You have ACWR, daily_load_acute,\n"
            "   daily_load_chronic, vo2_max_running, race_time_5k. Analyze them.\n"
            "   Don't ignore new columns just because they weren't there before.\n"
            "9. POSITIVE FINDINGS: Not everything is a problem. If RHR is improving,\n"
            "   HRV is up, or ACWR is in the optimal zone -- say so prominently.\n"
            "10. NO GENERIC ADVICE: Every recommendation must cite a specific\n"
            "    correlation, day pair, or metric value that supports it.\n"
            "    'Get better sleep' is not a recommendation. 'Your REM variance\n"
            "    (0-108 min, CV=58%%) suggests inconsistent sleep timing'\n"
            "    is a recommendation.\n"
            "11. TRAINING READINESS BANDS (use exact labels):\n"
            "    0-25 poor/red, 26-50 low/orange, 51-75 moderate/green,\n"
            "    76-95 high/blue, 96-100 effective/purple.\n"
            "12. NO UNSUBSTANTIATED CAUSALITY: Never say 'likely due to',\n"
            "    'because of', or 'caused by' without citing specific supporting\n"
            "    metrics. If underlying data is missing or ambiguous, say\n"
            "    'reason unclear from available data'.\n"
            "13. HYPOTHESIZE FROM DATA & USE THE OFF-SWITCH:\n"
            "    Form your initial hypothesis ONLY from the correlation matrices\n"
            "    or SQL text data provided. Do NOT use your pre-trained biases.\n"
            "    Once you have an empirical hypothesis from the text, verify it\n"
            "    using the 3-Layer Diagnostic process. OFF-SWITCH: If the data\n"
            "    does not show a clear link, it is 100% acceptable to state:\n"
            "    'I checked the deep metrics and surface inputs, and there is\n"
            "    no clear causal link in the data provided.' Do not hallucinate\n"
            "    a connection if the data is empty or ambiguous.\n"
            "14. DIAGNOSTIC INVESTIGATION (MANDATORY FOR EVERY FINDING):\n"
            "    Work like a doctor doing differential diagnosis. For EVERY\n"
            "    metric you analyze, follow this layered process:\n"
            "    LAYER 1 -- OUTCOME: What actually happened? State the fact.\n"
            "    LAYER 2 -- SURFACE INPUTS: What factors COULD have influenced it?\n"
            "      Check: stress, workout load, bb_drained, previous day activities,\n"
            "      sleep duration. Note which align and which CONTRADICT the outcome.\n"
            "    LAYER 3 -- UNDERLYING MECHANICS: What actually drove it?\n"
            "      When surface inputs contradict the outcome, dig into deeper\n"
            "      metrics: resting_hr_sleep, deep_sleep_sec, rem_sleep_sec,\n"
            "      avg_respiration, hrv_last_night. These tell the real story.\n"
            "    PRIORITY RULE: When Layer 2 contradicts Layer 1, you MUST\n"
            "      check Layer 3 before drawing conclusions.\n"
            "    COMPLETENESS RULE: Never stop at the first explanation that\n"
            "      fits. Always check ALL available metrics before concluding.\n"
            "      A single supporting metric is not enough -- look for\n"
            "      convergent evidence across multiple metrics.\n"
            "==============================================\n"
        )

        db_hint = (
            "\n\nDATABASE COLUMN REFERENCE -- daily_metrics table\n"
            "(use ONLY these exact column names in SQL queries):\n\n"
            "  Heart & HRV:\n"
            "    resting_hr -- resting heart rate (bpm, lower=fitter)\n"
            "    hrv_last_night -- overnight HRV in ms (higher=better recovery)\n"
            "    resting_hr_sleep -- HR during sleep\n\n"
            "  Stress:\n"
            "    stress_level -- daily avg stress 0-100 (lower=calmer)\n\n"
            "  Sleep:\n"
            "    sleep_score -- Garmin sleep quality 0-100\n"
            "    sleep_seconds, deep_sleep_sec, rem_sleep_sec, light_sleep_sec\n"
            "    avg_respiration -- breaths/min during sleep\n"
            "    body_battery_change -- BB gained during sleep\n\n"
            "  Body Battery:\n"
            "    bb_charged -- total energy recharged (higher=better rest)\n"
            "    bb_drained -- total energy spent (proxy for daily load)\n"
            "    bb_peak, bb_low -- daily max/min BB level\n\n"
            "  Activity & Steps:\n"
            "    total_steps, moderate_intensity_min, vigorous_intensity_min\n\n"
            "  Training Readiness:\n"
            "    training_readiness -- Garmin readiness score 0-100\n"
            "    Readiness bands: 0-25 poor/red, 26-50 low/orange,\n"
            "    51-75 moderate/green, 76-95 high/blue, 96-100 effective/purple\n"
            "    tr_acute_load -- short-term training load (EPOC)\n\n"
            "  Cardio Fitness (from Garmin bulk data):\n"
            "    vo2_max_running -- VO2Max for running (mL/kg/min, higher=fitter)\n"
            "    vo2_max_running_delta -- 7-day change in VO2Max (positive=improving)\n"
            "    vo2_max_cycling -- VO2Max for cycling\n"
            "    race_time_5k -- predicted 5K time in seconds (lower=faster)\n"
            "    race_time_10k -- predicted 10K time in seconds\n"
            "    race_time_5k_delta -- 7-day change (negative=getting faster)\n\n"
            "  Training Load:\n"
            "    daily_load_acute -- short-term load (7-day)\n"
            "    daily_load_chronic -- long-term load (28-day)\n"
            "    acwr -- acute:chronic workload ratio (<0.8 detraining, "
            "0.8-1.3 optimal, >1.5 injury risk)\n\n"
            "  Environment:\n"
            "    heat_acclimation_pct -- heat adaptation 0-100%%\n"
            "    altitude_acclimation -- altitude adaptation score\n\n"
            "  Weight: weight_kg\n\n"
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

        # == Physiological investigation guide (hypothesis-driven) ==
        physio_rules = (
            "\n\n====== PHYSIOLOGICAL INVESTIGATION GUIDE ======\n"
            "These are KNOWN RELATIONSHIPS to investigate. Each tells you\n"
            "what CAN affect what. You MUST check the correlation matrices\n"
            "and raw data to verify if it actually DOES for this user.\n"
            "Never presume -- always verify.\n\n"
            "A. RECOVERY & ANS -- WHAT TO INVESTIGATE:\n"
            "- Body Battery can be affected by HRV, stress, AND sleep -- not\n"
            "  just 'rest'. Check: does a rest day with high stress actually\n"
            "  show BB recovery in the data?\n"
            "- Lower RHR + higher HRV + lower stress can indicate\n"
            "  parasympathetic dominance. Verify: do the same-day correlations\n"
            "  confirm this for this user?\n"
            "- Training Readiness is a composite (sleep, recovery time, HRV,\n"
            "  stress, load). When it changes, decompose: which sub-metric\n"
            "  actually drove the change? Check conditioned AR(1) data.\n"
            "- HRV weekly trend (mean + CV) can be more informative than\n"
            "  single readings. Check: does AR(1) persistence support this?\n\n"
            "B. SLEEP -- WHAT TO INVESTIGATE:\n"
            "- Deep sleep supports physical restoration; REM supports cognitive\n"
            "  processing. Check: does deep_sleep_sec correlate with next-day\n"
            "  training_readiness? Does rem_sleep_sec correlate with stress?\n"
            "- Sleep score can predict next-day HRV, stress, readiness.\n"
            "  Check: what does NEXT-DAY PREDICTORS actually show for sleep?\n"
            "- Respiration >16/min during sleep can indicate elevated\n"
            "  sympathetic tone. Check: does avg_respiration correlate with\n"
            "  stress_level or hrv_last_night in the matrices?\n\n"
            "C. CROSS-TRAINING -- WHAT TO INVESTIGATE:\n"
            "When you see improvement in one sport, check if ANY of these\n"
            "are supported by the data:\n"
            "- Running can build cardiovascular fitness (check: running\n"
            "  frequency vs resting_hr or vo2_max_running trends)\n"
            "- Cycling provides aerobic base with low joint impact\n"
            "  (check: cycling sessions vs next-day recovery metrics)\n"
            "- Swimming can improve respiratory efficiency\n"
            "  (check: swim sessions vs avg_respiration trends)\n"
            "- Skiing demands eccentric strength + cardiovascular endurance\n"
            "  (check: ski sessions vs bb_drained and recovery timeline)\n"
            "- Gym/Strength can improve muscular economy across endurance\n"
            "  sports (check: do weeks with gym show lower exercising HR\n"
            "  in other sports? Check activities table avg_hr trends)\n"
            "- Basketball/Team Sports provide natural interval training\n"
            "  (check: sessions vs vo2_max changes)\n"
            "- Yoga can enhance parasympathetic recovery (check: do yoga\n"
            "  days show better next-day HRV or lower stress in lag-1?)\n"
            "- Pilates can improve core stability and movement economy\n"
            "  (check: sessions vs soreness in wellness_log)\n\n"
            "D. WORKOUT QUALITY -- WHAT TO INVESTIGATE:\n"
            "- Improving fitness shows as same/lower HR for same output.\n"
            "  Check: query activities for SAME sport_type over time --\n"
            "  is avg_hr trending down for similar distances?\n"
            "- ACWR 0.8-1.3 is optimal. Check: where is ACWR now?\n"
            "- Training effect per workout reflects stimulus quality.\n"
            "  Check: how does aerobic_training_effect correlate with\n"
            "  next-day metrics in this user's data?\n\n"
            "E. STRESS & LIFESTYLE -- WHAT TO INVESTIGATE:\n"
            "- Stress can impair sleep, HRV, and recovery. Check: what\n"
            "  does same-day and lag-1 data show for stress_level?\n"
            "- Mental stress + hard workout can compound. Check: do days\n"
            "  with both high stress AND high bb_drained show worse\n"
            "  next-day recovery than days with only one?\n"
            "- Steps and stairs can promote active recovery.\n"
            "  Check: does total_steps correlate with bb_charged?\n"
            "- If caffeine/alcohol data exists in wellness_log,\n"
            "  check their lag-1 impact on sleep_score and HRV.\n\n"
            "F. MULTI-DAY EFFECTS -- WHAT TO INVESTIGATE:\n"
            "- Recovery timing is individual. Check: MULTI-DAY CARRYOVER\n"
            "  shows which lag (1, 2, or 3 days) is significant. This is\n"
            "  the user's ACTUAL recovery timeline -- do NOT assume textbook.\n"
            "- If lag-2 is significant but lag-1 is not, the effect takes\n"
            "  2 days to manifest. Report that finding.\n"
            "- Present raw day-by-day numbers and reference which lag the\n"
            "  matrices say matters for each metric pair.\n"
            "- Weekly patterns may exist but only report them if the sample\n"
            "  size and correlation strength support them.\n"
            "==============================================\n"
        )

        ctx = f"{corr_block}{analysis_rules}{physio_rules}{db_hint}"


        return [
            Task(
                description=f"""
                You are given pre-computed correlation analysis AND optionally
                multi-window comparison data. Your job is to interpret ALL of it.

                PART 1 — SINGLE-WINDOW (cover ALL 10 sections):
                1. Training Intensity Classification
                2. Same-Day Correlations (top 5 pairs)
                3. Next-Day Predictors (top 5)
                4. Autoregressive Persistence
                5. Distributions (normal vs skewed)
                6. Recent Anomalies (last 3 days z-scores)
                7. Conditioned AR(1) — MOST IMPORTANT: R² improvements
                8. Markov State Transitions (+ sample sizes)
                9. KL-Divergence
                10. Metric Ranges

                PART 2 — CROSS-TIMEFRAME (if comparison data available):
                Track key findings across ALL windows (7d to 365d).
                Classify: ROBUST / STABLE / EMERGING / SPARSE.
                Show tables: | Metric Pair | 7d r | 14d r | 30d r | Verdict |

                DO NOT give advice. ONLY interpret numbers.
                {ctx}
                {comparison_block}
                """,
                agent=self.statistical_interpreter,
                expected_output="Full statistical interpretation with cross-timeframe stability"
            ),

            Task(
                description=f"""
                Find 3-5 non-obvious patterns AND rigorously assess trends.
                The Statistical Interpreter has covered the correlation tables —
                focus on DAY-BY-DAY patterns that go BEYOND what correlations show.

                PATTERNS:
                1. Query day-by-day values for this week
                2. FLAG suspect data (rem=0, score<40, outliers >2σ)
                3. Analyze DAY PAIRS (day X → day X+1 responses)
                4. Verify pre-computed correlations against actual data

                TRENDS — for HRV, RHR, sleep_score, stress, readiness, ACWR:
                1. List daily values for last 7 days
                2. Compute mean, std, CV
                3. Directional trend? (3+ consecutive days)
                4. Survives outlier removal?
                5. Verdict per metric

                Report as: FINDING / EVIDENCE / SAMPLE SIZE / CONFIDENCE / VERDICT
                {ctx}
                """,
                agent=self.health_pattern_analyst,
                expected_output="3-5 patterns + per-metric trend table with confidence"
            ),

            Task(
                description=f"""
                Week-over-week performance comparison + recovery assessment.

                PERFORMANCE:
                1. Query both weeks daily values (14 days)
                2. Per metric: raw averages, outlier-adjusted, range, verdict
                   (IMPROVED >10%% / STABLE <10%% / DECLINED >10%% / VOLATILE)
                3. Best and worst day this week — what made them different
                4. ACWR + training load analysis
                5. Activities summary: workouts, types, total load

                RECOVERY:
                1. Find 2-3 hardest days (training_load or bb_drained)
                2. Next-day response for each hard day with specific numbers
                3. Overtraining checklist (5 criteria, yes/no with evidence)
                4. Recovery speed: Fast / Moderate / Slow
                5. Intensity recommendation with reasoning
                {ctx}
                """,
                agent=self.performance_recovery,
                expected_output="Week-over-week table + bounce-back analysis + overtraining checklist"
            ),

            Task(
                description=f"""
                SLEEP deep-dive + LIFESTYLE connections for this week.

                SLEEP:
                1. Query 7 days of sleep data
                2. Architecture table per night: Score | Total | Deep%% | REM%% | Light%% | Resp | BB Change
                3. Flag suspect nights, compute averages with/without
                4. Consistency: CV for score and duration
                5. Next-day impact mapping per night
                6. Week-over-week sleep comparison (query last week too)
                7. Top sleep insight

                LIFESTYLE:
                1. Query activities for this week
                2. Day stories: "Feb X: [activity] → Feb X+1: [biometrics]"
                3. Classify activities (commute vs training)
                4. ACWR trajectory with daily values
                5. KL-divergence: which conditions shift outcomes?
                6. State clearly what data you DON'T have
                {ctx}
                """,
                agent=self.sleep_lifestyle,
                expected_output="Sleep architecture + day-by-day lifestyle stories"
            ),

            Task(
                description=f"""
                Synthesize all previous findings into #1 bottleneck + top 3 quick wins.

                STEP 0 — MEMORY CHECK:
                Use the 'Get Past Recommendations' tool to retrieve past recommendations.
                Review what was recommended before and what happened.
                {self._get_past_recs_context()}

                1. FACT-CHECK: Did previous agents flag outliers? Note sample sizes?
                   Conflate metrics? Correct any errors.
                2. PAST RECOMMENDATIONS FOLLOW-UP:
                   - For each past recommendation, assess: was it followed? Did the
                     target metric change in the expected direction?
                   - Explicitly state: "Last week I recommended X. Result: Y."
                3. BOTTLENECK: Top 3 candidates with supporting data.
                   Assess evidence strength (n>=20 consistent vs n<20 outlier-driven).
                   Pick the one with strongest evidence (or say "insufficient data").
                4. QUICK WINS — each must have:
                   - Specific metric expected to improve
                   - Evidence (correlation, day-pair, KL shift)
                   - Confidence level (high/medium/low)
                   - How to verify next week
                   Format each as:
                   RECOMMENDATION: [what to do]
                   TARGET_METRIC: [metric name]
                   EXPECTED_DIRECTION: [IMPROVE/DECLINE/STABLE]
                5. WHAT'S WORKING: 2-3 positive things to continue
                6. HONEST LIMITATIONS: what can't we determine yet?
                7. FINAL UI SUMMARY (MANDATORY):
                   End your response with EXACTLY three bullets, each <= 280 chars:
                   - What changed: ...
                   - Why it matters: ...
                   - Next 24-48h: ...
                   Use plain language, avoid tables, SQL, or dense metric dumps in this block.
                {ctx}
                """,
                agent=self.synthesizer,
                expected_output="Fact-checked synthesis with past rec follow-up + bottleneck + structured quick wins + limitations"
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
                agent=self.performance_recovery,
                expected_output="Goal progress report with specific metrics"
            ),
            
            Task(
                description=f"""
                Based on the goal: {goal_description}
                
                Recommend specific actions for the next 2 weeks to accelerate progress.
                Be concrete and measurable.
                """,
                agent=self.synthesizer,
                expected_output="2-week action plan for goal achievement"
            )
        ]
    
    def run_comprehensive_analysis(self, analysis_period: int = 30) -> str:
        """Run full analysis with all agents"""
        
        log.info("\n" + "="*60)
        log.info("נ₪– RUNNING COMPREHENSIVE HEALTH ANALYSIS")
        log.info("   This will take a few minutes...")
        log.info("="*60 + "\n")
        
        tasks = self.create_deep_analysis_tasks(analysis_period)

        agent_labels = [
            "STATISTICAL INTERPRETATION (Interpreter)",
            "PATTERNS & TRENDS (Pattern Analyst)",
            "PERFORMANCE & RECOVERY (Performance/Recovery)",
            "SLEEP & LIFESTYLE (Sleep/Lifestyle)",
            "BOTTLENECK & QUICK WINS (Synthesizer)",
        ]

        crew = Crew(
            agents=[
                self.statistical_interpreter,
                self.health_pattern_analyst,
                self.performance_recovery,
                self.sleep_lifestyle,
                self.synthesizer,
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
        
        log.info(f"\nנ¯ Analyzing progress toward: {goal}\n")
        
        tasks = self.create_goal_progress_tasks(goal)
        
        crew = Crew(
            agents=[self.performance_recovery, self.synthesizer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        return crew.kickoff()

    def run_weekly_summary(self, matrix_context: str = "",
                           comparison_context: str = "") -> str:
        """Run the 5-agent weekly summary pipeline.

        Returns the combined text output from all agents.
        Also parses Synthesizer recommendations for DB persistence.
        """
        log.info("\n" + "="*60)
        log.info("  RUNNING WEEKLY SUMMARY (5 agents)")
        log.info("="*60 + "\n")

        tasks = self.create_weekly_summary_tasks(
            matrix_context=matrix_context,
            comparison_context=comparison_context,
        )

        agent_labels = [
            "STATISTICAL INTERPRETATION (Interpreter)",
            "PATTERNS & TRENDS (Pattern Analyst)",
            "PERFORMANCE & RECOVERY (Performance/Recovery)",
            "SLEEP & LIFESTYLE (Sleep/Lifestyle)",
            "BOTTLENECK & QUICK WINS (Synthesizer)",
        ]

        crew = Crew(
            agents=[
                self.statistical_interpreter,
                self.health_pattern_analyst,
                self.performance_recovery,
                self.sleep_lifestyle,
                self.synthesizer,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        result = crew.kickoff()

        # Combine ALL agent outputs
        sections = []
        synthesizer_output = ""
        try:
            for label, task_out in zip(agent_labels, result.tasks_output):
                raw = getattr(task_out, "raw", str(task_out))
                sections.append(f"{'='*60}\n  {label}\n{'='*60}\n\n{raw}")
                if "Synthesizer" in label:
                    synthesizer_output = raw
        except Exception:
            return str(result)

        # Parse and save structured recommendations from Synthesizer
        recs = self._parse_recommendations(synthesizer_output)
        if recs:
            save_recommendations_to_db(recs)
            log.info("Saved %d recommendations to long-term memory", len(recs))

        combined = "\n\n".join(sections)
        log.info(f"\n  Combined output: {len(combined)} chars from {len(sections)} agents")
        return combined

    @staticmethod
    def _parse_recommendations(synthesizer_text: str) -> list:
        """Parse structured recommendations from Synthesizer output.

        Looks for blocks like:
            RECOMMENDATION: [text]
            TARGET_METRIC: [metric]
            EXPECTED_DIRECTION: [IMPROVE/DECLINE/STABLE]
        """
        recs = []
        current_rec = None
        for line in synthesizer_text.splitlines():
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("RECOMMENDATION:"):
                # Save previous if exists
                if current_rec and current_rec.get("recommendation"):
                    recs.append(current_rec)
                current_rec = {
                    "recommendation": stripped.split(":", 1)[1].strip(),
                    "target_metric": "",
                    "expected_direction": "",
                    "agent_name": "synthesizer",
                }
            elif upper.startswith("TARGET_METRIC:") and current_rec is not None:
                current_rec["target_metric"] = stripped.split(":", 1)[1].strip()
            elif upper.startswith("EXPECTED_DIRECTION:") and current_rec is not None:
                val = stripped.split(":", 1)[1].strip().upper()
                if val in ("IMPROVE", "DECLINE", "STABLE"):
                    current_rec["expected_direction"] = val
        # Don't forget the last one
        if current_rec and current_rec.get("recommendation"):
            recs.append(current_rec)
        return recs
