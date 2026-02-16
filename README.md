# Garmin Health Intelligence 🏃‍♂️📊

A personal data analytics platform designed to unlock actionable insights from wearable health metrics. This project bridges the gap between raw data collection and meaningful interpretation by combining a robust data pipeline with a team of specialized AI agents.

## 🎯 Project Goal

Garmin Connect provides excellent data tracking, but I wanted more depth. I wanted to understand the *relationships* between my metrics, not just see the numbers.

The goal of **Garmin Health Intelligence** is to answer "Why?":
* Why is my recovery low today despite sleeping 8 hours?
* How does my daily stress actually impact my training readiness?
* What is the single most impactful factor affecting my sleep quality?

Instead of relying on generic algorithms, this system uses a custom-built pipeline to analyze *my* specific data patterns and provide personalized feedback.

## 📸 Dashboard Preview

![Overview Page](docs/screenshots/overview.png)
*Real-time metrics, recovery status, and 7-day trends at a glance.*

![Correlation Matrix](docs/screenshots/correlations.png)
*Discovering hidden relationships between sleep, HRV, stress, and activity.*

![Agent Chat](docs/screenshots/agent_chat.png)
*Chatting with the AI agents to get data-backed answers to health questions.*

## 🏗️ System Architecture

The project operates as an automated weekly pipeline:

1.  **Data Ingestion:** Automatically fetches raw data from Garmin (Heart Rate, Sleep, Stress, Activities, etc.) using the `garth` API.
2.  **Database Storage:** Upserts all data into a structured **PostgreSQL** database tailored for time-series analysis.
3.  **Statistical Engine:** Runs a comprehensive analysis layer (correlations, trends, anomalies) to prepare a "Context Window" for the AI.
4.  **AI Analysis:** A team of specialized Agents (CrewAI) interprets the statistical findings.
5.  **Visualization:** A Streamlit dashboard presents the data and allows for interactive exploration.

## 💾 Database Structure

The heart of the system is a PostgreSQL database designed to capture the full spectrum of health data. Key tables include:

* **`daily_metrics`**: The core table containing daily summaries (Resting HR, HRV, Sleep Score, Body Battery, Stress Level, Training Readiness, etc.).
* **`activities`**: Detailed logs of every workout (Running, Cycling, Gym), including duration, intensity, and physiological load.
* **`wellness_log`**: A custom table for self-reported subjective data (Energy Level, Soreness, Caffeine Intake, Nutrition). This allows the system to correlate *feelings* with *biometrics*.
* **`matrix_summaries`**: Stores the pre-computed statistical correlations to give the AI agents long-term context.

## 🤖 The AI Agents Team

Instead of a single generic LLM, the system employs **5 specialized AI Agents**, each with a distinct role and expertise. They work together to analyze the data:

| Agent | Role & Responsibility |
| :--- | :--- |
| **1. Statistical Interpreter** | The "Data Scientist". Reads the raw correlation matrices and statistical outputs. Its job is to translate numbers into plain English findings (e.g., "Strong negative correlation found between Stress and Deep Sleep"). |
| **2. Health Pattern Analyst** | The "Detective". Looks for day-to-day patterns and anomalies that pure statistics might miss. It flags outliers (like a bad night's sleep due to a sensor error) and identifies trends (e.g., "HRV has been trending down for 3 days"). |
| **3. Performance & Recovery** | The "Coach". Focuses purely on training metrics. It analyzes Training Load, ACWR (Acute:Chronic Workload Ratio), and recovery bounce-back after hard sessions. It answers: "Are you ready to train hard today?" |
| **4. Sleep & Lifestyle** | The "Wellness Expert". Deep-dives into sleep architecture (REM/Deep/Light) and connects lifestyle factors (like late workouts or high stress) to sleep outcomes. |
| **5. The Synthesizer** | The "Team Lead". This is the most crucial agent. It reads the reports from all other agents, resolves conflicts, checks against **long-term memory** (past recommendations), and produces the final actionable insights and the "Top 3 Quick Wins" for the week. |

## 🚀 Key Features

* **Automated Sync:** Runs weekly via GitHub Actions to keep data fresh.
* **Interactive Dashboard:** Explore trends, deep-dive into specific days, and visualize correlations.
* **Natural Language Chat:** Ask questions like "How did my marathon training affect my sleep this month?" and get answers based on your actual database.
* **Daily Input Form:** Easily log subjective data to enrich the analysis.
* **Privacy Focused:** Personal data stays in your private database; only statistical summaries are sent to the LLM for analysis.

## 🛠️ Tech Stack

* **Python 3.12**
* **PostgreSQL** (Database)
* **Streamlit** (Dashboard UI)
* **CrewAI** (Agent Orchestration)
* **Google Gemini 2.5 Flash** (LLM)
* **Plotly** (Visualizations)
* **GitHub Actions** (CI/CD & Automation)

---

*This project is for educational and personal use, exploring the intersection of Data Engineering and GenAI.*
