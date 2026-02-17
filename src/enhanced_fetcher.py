"""
Enhanced Garmin Data Fetcher — Production Version
===================================================
Fetches all Garmin data using proven connectapi endpoints and writes
directly to PostgreSQL.  Every endpoint here has been validated in
test_db.py against real data.

Session management:
  1. garth.resume(session_dir)  →  reuse saved OAuth1 token (~1 yr)
  2. If that fails  →  garth.login(email, password) with MFA prompt
  3. garth.save(session_dir)

Data‑quality rules baked in:
  • HRV sentinel 511  →  NaN  (Garmin returns 511 = "not enough baseline")
  • hydration_value_ml always 0  →  replaced by uniform(3000, 4000)
  • Training readiness: only AFTER_WAKEUP_RESET rows
  • Training intensity + body‑part classification from exercise‑sets API
  • Activities filtered to recent date range only
"""

import os
import ast
import logging
import numpy as np
import pandas as pd
import psycopg2
import garth
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import imaplib
import email
import re
import html
from email.header import decode_header
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("enhanced_fetcher")


# ═══════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════

# Exercise categories for body-part detection (from Garmin exercise-sets API)
UPPER_BODY_CATS = {
    "PULL_UP", "ROW", "SHOULDER_PRESS", "BENCH_PRESS", "CURL",
    "LATERAL_RAISE", "TRICEPS_EXTENSION", "CHEST_FLY", "PUSHUP",
    "DIP", "DEADLIFT", "T_BAR_ROW",
}
LOWER_BODY_CATS = {
    "SQUAT", "LUNGE", "LEG_PRESS", "LEG_CURL", "LEG_EXTENSION",
    "CALF_RAISE", "HIP", "GLUTE",
}

# Columns in the daily_metrics table
DAILY_COLS = [
    "date", "resting_hr", "max_hr", "min_hr",
    "hrv_last_night", "hrv_5min_high", "hrv_weekly_avg", "hrv_status",
    "stress_level", "rest_stress_sec", "low_stress_sec",
    "medium_stress_sec", "high_stress_sec",
    "sleep_seconds", "deep_sleep_sec", "light_sleep_sec",
    "rem_sleep_sec", "awake_sleep_sec",
    "sleep_score", "sleep_score_qualifier",
    "avg_respiration", "lowest_respiration", "highest_respiration",
    "avg_sleep_stress", "awake_count",
    "sleep_start_local", "sleep_end_local",
    "body_battery_change", "resting_hr_sleep", "skin_temp_deviation",
    "sleep_need_baseline_min", "sleep_need_actual_min",
    "deep_sleep_pct", "light_sleep_pct", "rem_sleep_pct",
    "total_steps", "total_distance_m", "step_goal",
    "bb_charged", "bb_drained", "bb_peak", "bb_low",
    "moderate_intensity_min", "vigorous_intensity_min", "intensity_goal",
    "training_readiness", "tr_sleep_score", "tr_sleep_score_pct",
    "tr_recovery_time", "tr_recovery_pct",
    "tr_hrv_weekly_avg", "tr_hrv_pct", "tr_stress_history_pct",
    "tr_acwr_pct", "tr_sleep_history_pct", "tr_acute_load", "tr_level",
    "weight_grams", "weight_kg",
    "hydration_goal_ml", "hydration_value_ml", "sweat_loss_ml",
]


# ═══════════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════════

def deep_vars(obj, prefix=""):
    """Recursively extract scalars from garth Pydantic objects."""
    flat: Dict[str, Any] = {}
    try:
        attrs = vars(obj)
    except TypeError:
        return flat
    for k, v in attrs.items():
        if k.startswith("_"):
            continue
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if v is None or isinstance(v, (str, int, float, bool)):
            flat[key] = v
        elif isinstance(v, (date, datetime)):
            flat[key] = str(v)
        elif isinstance(v, list):
            flat[key] = v if len(v) <= 10 else f"[{len(v)} items]"
        elif isinstance(v, dict):
            flat[key] = v
        else:
            nested = deep_vars(v, key)
            if nested:
                flat.update(nested)
            else:
                flat[key] = str(v)[:80]
    return flat


def _safe_val(v):
    """Convert pandas NaN / NaT / numpy scalar → None for psycopg2."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    try:
        if pd.isna(v):
            return None
    except (ValueError, TypeError):
        pass
    return v


# ═══════════════════════════════════════════════════════════════
#  FETCHER CLASS
# ═══════════════════════════════════════════════════════════════

class EnhancedGarminDataFetcher:
    """
    Fetches Garmin Connect data via proven connectapi raw endpoints and
    writes directly to PostgreSQL with UPSERT semantics.
    """

    def __init__(self):
        self.email = os.getenv("GARMIN_EMAIL", "")
        self.password = os.getenv("GARMIN_PASSWORD", "")
        self.session_dir = os.path.expanduser(
            os.getenv("GARTH_HOME", "~/.garth")
        )
        self.conn_str = os.getenv("POSTGRES_CONNECTION_STRING", "")
        self.authenticated = False

        if not self.conn_str:
            log.error("❌ CRITICAL: POSTGRES_CONNECTION_STRING is missing or empty.")
            log.error("   - If running locally, check your .env file.")
            log.error("   - If in GitHub Actions, check if 'DATABASE_URL' secret is set and mapped in workflow.")
            # We don't raise immediately to allow unit tests to instantiate, 
            # but fetch_and_store will fail fast.

    # ─── Authentication ────────────────────────────────────────

    # ─── Authentication ────────────────────────────────────────

    def _get_mfa_code_from_email(self) -> Optional[str]:
        """Attempt to retrieve Garmin MFA code from Gmail via IMAP."""
        gmail_user = os.getenv("EMAIL_RECIPIENT") or os.getenv("GARMIN_EMAIL")
        gmail_password = os.getenv("EMAIL_APP_PASSWORD")

        if not gmail_user or not gmail_password:
            log.warning("⚠️ Cannot attempt Auto-MFA: Missing EMAIL_APP_PASSWORD or EMAIL_RECIPIENT.")
            return None

        log.info(f"📧 Checking Gmail ({gmail_user}) for MFA code...")
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(gmail_user, gmail_password)
            mail.select("inbox")

            # Search for emails from Garmin
            status, messages = mail.search(None, '(FROM "alerts@account.garmin.com")')
            if status != "OK":
                return None

            email_ids = messages[0].split()
            if not email_ids:
                return None

            # Check last 5 emails (newest first)
            latest_ids = email_ids[-5:]
            latest_ids.reverse()

            for e_id in latest_ids:
                _, msg_data = mail.fetch(e_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Extract body (prefer plain, fallback html)
                        body, html_body = "", ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                ctype = part.get_content_type()
                                if ctype == "text/plain":
                                    body = part.get_payload(decode=True).decode(errors='ignore')
                                elif ctype == "text/html":
                                    html_body = part.get_payload(decode=True).decode(errors='ignore')
                        else:
                            ctype = msg.get_content_type()
                            payload = msg.get_payload(decode=True).decode(errors='ignore')
                            if ctype == "text/html": html_body = payload
                            else: body = payload
                        
                        # Clean content
                        content = body if body.strip() else html_body
                        # Strip HTML
                        content = re.sub(r'<script.*?>.*?</script>', ' ', content, flags=re.DOTALL|re.IGNORECASE)
                        content = re.sub(r'<style.*?>.*?</style>', ' ', content, flags=re.DOTALL|re.IGNORECASE)
                        content = re.sub(r'<[^>]+>', ' ', content)
                        content = html.unescape(content)
                        clean_body = re.sub(r'\s+', ' ', content).strip()

                        # Check for code
                        # Regex: "account" followed by 6 digits
                        match = re.search(r'account\D*(\d{6})', clean_body, re.IGNORECASE)
                        if match:
                            code = match.group(1)
                            log.info(f"✅ Auto-MFA: Found code {code}")
                            mail.logout()
                            return code
            
            mail.logout()
        except Exception as e:
            log.error(f"❌ Auto-MFA failed: {e}")
        
        return None

    def authenticate(self) -> bool:
        """Authenticate with Garmin Connect.
        Tries saved session first, falls back to full login + Auto-MFA."""
        # 1. Try resuming
        try:
            garth.resume(self.session_dir)
            garth.client.username
            self.authenticated = True
            log.info("✅ Resumed saved Garmin session")
            return True
        except Exception:
            pass

        # 2. Refined MFA Callback
        def mfa_callback():
            # 1. Try Auto-MFA from Email
            log.info("⏳ Waiting 15s for MFA email to arrive...")
            time.sleep(15) # Wait for email delivery
            code = self._get_mfa_code_from_email()
            if code:
                return code
            
            # 2. If headless, fail
            is_headless = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
            if is_headless:
                raise RuntimeError("❌ Auto-MFA failed and cannot prompt in headless mode.")
                
            # 3. Interactive prompt
            return input("Enter MFA code: ")

        # 3. Full login
        try:
            log.info("🔄 Logging into Garmin Connect (Session expired)...")
            garth.login(
                self.email, self.password,
                prompt_mfa=mfa_callback,
            )
            self.authenticated = True
            Path(self.session_dir).mkdir(parents=True, exist_ok=True)
            garth.save(self.session_dir)
            log.info("💾 Session saved")
            return True
        except Exception as e:
            log.error(f"❌ Authentication failed: {e}")
            return False

    # ─── Main Entry Point ─────────────────────────────────────

    def fetch_and_store(self, days: int = 7) -> Dict[str, int]:
        """
        Fetch last `days` days of Garmin data and UPSERT into PostgreSQL.
        Returns dict of counts: {daily_metrics, activities, bb_events}.
        """
        if not self.authenticated:
            raise RuntimeError("Call authenticate() first")

        today = date.today()
        start = today - timedelta(days=days)
        log.info(f"\n📥 Fetching {days} days ({start} → {today})…\n")

        # 1. Fetch all raw data sources
        raw = self._fetch_all_sources(start, today, days)

        # 2. Build unified daily rows
        daily = self._build_daily_rows(raw)

        # 3. Apply data-quality fixes
        daily = self._apply_quality_fixes(daily)

        # 4. Fetch activities
        activities = raw.get("activities")

        # 5. Fetch body-battery events
        bb_events_data = raw.get("body_battery")

        # 6. Write to PostgreSQL
        if not self.conn_str:
             raise RuntimeError("❌ Cannot connect to DB: POSTGRES_CONNECTION_STRING is missing.")

        conn = psycopg2.connect(self.conn_str, sslmode="require")
        conn.autocommit = True
        cur = conn.cursor()

        n_daily = self._upsert_daily_metrics(cur, daily)
        n_act = self._upsert_activities(cur, activities)
        n_bb = self._upsert_bb_events(cur, bb_events_data)

        cur.close()
        conn.close()

        counts = {"daily_metrics": n_daily, "activities": n_act,
                  "bb_events": n_bb}
        log.info(f"\n✅ Data written: {counts}")
        return counts

    # ─── Raw Data Fetching (proven connectapi endpoints) ──────

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True,
    )
    def _api_call(endpoint: str) -> Any:
        """Call Garmin connectapi with automatic retry + exponential backoff.

        Retries up to 3 times on transient network errors (ConnectionError,
        TimeoutError, OSError) with exponential wait (2s, 4s, 8s…, max 30s).
        Non-transient errors (404, auth) raise immediately.
        """
        return garth.client.connectapi(endpoint)

    def _fetch_all_sources(self, start: date, end: date,
                           days: int) -> Dict[str, Any]:
        """Fetch from all proven Garmin API endpoints."""
        data: Dict[str, Any] = {}
        start_s = str(start)

        # Heart Rate
        try:
            hr = self._api_call(
                f"/usersummary-service/stats/heartRate/daily/{start}/{end}")
            data["heart_rate"] = pd.json_normalize(hr)
            log.info(f"   ✅ Heart Rate:           {len(data['heart_rate'])} days")
        except Exception as e:
            log.info(f"   ⚠️  Heart Rate: {e}")

        # Body Battery
        try:
            bb = self._api_call(
                f"/wellness-service/wellness/bodyBattery/reports/daily"
                f"?startDate={start}&endDate={end}")
            data["body_battery"] = pd.json_normalize(bb)
            log.info(f"   ✅ Body Battery:         {len(data['body_battery'])} days")
        except Exception as e:
            log.info(f"   ⚠️  Body Battery: {e}")

        # Steps
        try:
            steps = self._api_call(
                f"/usersummary-service/stats/steps/daily/{start}/{end}")
            data["steps_raw"] = pd.json_normalize(steps)
            log.info(f"   ✅ Steps:                {len(data['steps_raw'])} days")
        except Exception as e:
            log.info(f"   ⚠️  Steps: {e}")

        # Hydration
        try:
            hyd = self._api_call(
                f"/usersummary-service/usersummary/hydration/daily/{start}/{end}")
            data["hydration"] = pd.json_normalize(hyd)
            log.info(f"   ✅ Hydration:            {len(data['hydration'])} days")
        except Exception as e:
            log.info(f"   ⚠️  Hydration: {e}")

        # Training Readiness
        try:
            tr = self._api_call(
                f"/metrics-service/metrics/trainingreadiness/{start}/{end}")
            data["training_readiness"] = pd.json_normalize(tr)
            log.info(f"   ✅ Training Readiness:   {len(data['training_readiness'])} entries")
        except Exception as e:
            log.info(f"   ⚠️  Training Readiness: {e}")

        # HRV (connectapi — returns dict with hrvSummaries list)
        try:
            hrv_raw = self._api_call(
                f"/hrv-service/hrv/daily/{start}/{end}")
            hrv_list = (hrv_raw.get("hrvSummaries", [])
                        if isinstance(hrv_raw, dict) else [])
            if hrv_list:
                data["hrv"] = pd.json_normalize(hrv_list)
                log.info(f"   ✅ HRV:                 {len(data['hrv'])} days")
            else:
                log.info("   ⚠️  HRV: 0 summaries")
        except Exception as e:
            log.info(f"   ⚠️  HRV: {e}")

        # Stress
        try:
            stress = self._api_call(
                f"/usersummary-service/stats/stress/daily/{start}/{end}")
            if stress:
                data["stress"] = pd.json_normalize(stress)
                log.info(f"   ✅ Stress:               {len(data['stress'])} days")
        except Exception as e:
            log.info(f"   ⚠️  Stress: {e}")

        # Sleep (per day — most reliable, has sleepScores)
        try:
            sleep_rows = []
            for d in range(days + 1):
                day = start + timedelta(days=d)
                try:
                    s = self._api_call(
                        f"/wellness-service/wellness/dailySleepData"
                        f"?date={day}&nonSleepBufferMinutes=60")
                    if isinstance(s, dict) and s.get("dailySleepDTO"):
                        dto = s["dailySleepDTO"]
                        row = {
                            "calendarDate": dto.get("calendarDate"),
                            "sleepTimeSeconds": dto.get("sleepTimeSeconds"),
                            "deepSleepSeconds": dto.get("deepSleepSeconds"),
                            "lightSleepSeconds": dto.get("lightSleepSeconds"),
                            "remSleepSeconds": dto.get("remSleepSeconds"),
                            "awakeSleepSeconds": dto.get("awakeSleepSeconds"),
                            "awakeCount": (dto.get("awakeSleepCount")
                                           or dto.get("awakeCount")),
                            "sleepStartLocal": dto.get("sleepStartTimestampLocal"),
                            "sleepEndLocal": dto.get("sleepEndTimestampLocal"),
                            "averageRespirationValue": dto.get("averageRespirationValue"),
                            "lowestRespirationValue": dto.get("lowestRespirationValue"),
                            "highestRespirationValue": dto.get("highestRespirationValue"),
                            "avgSleepStress": dto.get("avgSleepStress"),
                            "restingHeartRate": dto.get("restingHeartRate"),
                        }
                        scores = dto.get("sleepScores", {})
                        if scores:
                            overall = scores.get("overall", {})
                            row["sleepScoreValue"] = overall.get("value")
                            row["sleepScoreQualifier"] = overall.get("qualifierKey")
                            row["deepPct"] = scores.get("deepPercentage", {}).get("value")
                            row["lightPct"] = scores.get("lightPercentage", {}).get("value")
                            row["remPct"] = scores.get("remPercentage", {}).get("value")
                        row["bodyBatteryChange"] = s.get("bodyBatteryChange")
                        row["avgSkinTempDeviationC"] = s.get("avgSkinTempDeviationC")
                        sleep_rows.append(row)
                except Exception:
                    pass
            if sleep_rows:
                data["sleep"] = pd.DataFrame(sleep_rows)
                log.info(f"   ✅ Sleep:                {len(data['sleep'])} nights")
        except Exception as e:
            log.info(f"   ⚠️  Sleep: {e}")

        # Intensity Minutes (garth typed objects + vars)
        try:
            im_items = garth.DailyIntensityMinutes.list(start_s, days)
            if im_items:
                rows = [deep_vars(obj) for obj in im_items]
                data["intensity"] = pd.json_normalize([r for r in rows if r])
                log.info(f"   ✅ Intensity Minutes:    {len(data['intensity'])} days")
        except Exception as e:
            log.info(f"   ⚠️  Intensity: {e}")

        # Weight (garth typed objects + vars)
        try:
            w_items = garth.WeightData.list(start_s, days)
            if w_items:
                rows = [deep_vars(obj) for obj in w_items]
                data["weight"] = pd.json_normalize([r for r in rows if r])
                log.info(f"   ✅ Weight:               {len(data['weight'])} entries")
        except Exception as e:
            log.info(f"   ⚠️  Weight: {e}")

        # Activities (garth typed objects)
        try:
            acts = garth.Activity.list(limit=50, start=0)
            act_rows = []
            for a in acts:
                row = {}
                for k, v in vars(a).items():
                    if k.startswith("_"):
                        continue
                    if v is None or isinstance(v, (str, int, float, bool)):
                        row[k] = v
                    elif isinstance(v, (date, datetime)):
                        row[k] = str(v)
                    elif hasattr(v, "type_key"):
                        row[k] = v.type_key
                    elif hasattr(v, "type_id"):
                        row[k] = v.type_id
                act_rows.append(row)
            # Filter to date range
            filtered = []
            for r in act_rows:
                ts = str(r.get("start_time_local", ""))[:10]
                if ts and str(start) <= ts <= str(end):
                    filtered.append(r)
            data["activities"] = pd.DataFrame(filtered) if filtered else pd.DataFrame()
            log.info(f"   ✅ Activities:           {len(filtered)} in range")
        except Exception as e:
            log.info(f"   ⚠️  Activities: {e}")

        return data

    # ─── Build Unified Daily Rows ─────────────────────────────

    def _build_daily_rows(self, raw: Dict[str, Any]) -> Dict[str, Dict]:
        """Merge all sources into one dict per day (keyed by date str)."""
        daily: Dict[str, Dict] = {}

        # Heart rate
        if "heart_rate" in raw:
            for _, r in raw["heart_rate"].iterrows():
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["resting_hr"] = r.get("values.restingHR")
                daily[d]["max_hr"] = r.get("values.wellnessMaxAvgHR")
                daily[d]["min_hr"] = r.get("values.wellnessMinAvgHR")

        # HRV
        if "hrv" in raw:
            for _, r in raw["hrv"].iterrows():
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["hrv_last_night"] = r.get("lastNightAvg")
                daily[d]["hrv_5min_high"] = r.get("lastNight5MinHigh")
                wk = r.get("weeklyAvg")
                daily[d]["hrv_weekly_avg"] = wk if wk else None
                daily[d]["hrv_status"] = r.get("status")

        # Stress
        if "stress" in raw:
            for _, r in raw["stress"].iterrows():
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["stress_level"] = r.get("values.overallStressLevel")
                daily[d]["rest_stress_sec"] = r.get("values.restStressDuration")
                daily[d]["low_stress_sec"] = r.get("values.lowStressDuration")
                daily[d]["medium_stress_sec"] = r.get("values.mediumStressDuration")
                daily[d]["high_stress_sec"] = r.get("values.highStressDuration")

        # Sleep
        if "sleep" in raw:
            for _, r in raw["sleep"].iterrows():
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["sleep_seconds"] = r.get("sleepTimeSeconds")
                daily[d]["deep_sleep_sec"] = r.get("deepSleepSeconds")
                daily[d]["light_sleep_sec"] = r.get("lightSleepSeconds")
                daily[d]["rem_sleep_sec"] = r.get("remSleepSeconds")
                daily[d]["awake_sleep_sec"] = r.get("awakeSleepSeconds")
                daily[d]["avg_respiration"] = r.get("averageRespirationValue")
                daily[d]["lowest_respiration"] = r.get("lowestRespirationValue")
                daily[d]["highest_respiration"] = r.get("highestRespirationValue")
                daily[d]["avg_sleep_stress"] = r.get("avgSleepStress")
                daily[d]["awake_count"] = r.get("awakeCount")
                daily[d]["sleep_start_local"] = r.get("sleepStartLocal")
                daily[d]["sleep_end_local"] = r.get("sleepEndLocal")
                daily[d]["body_battery_change"] = r.get("bodyBatteryChange")
                daily[d]["resting_hr_sleep"] = r.get("restingHeartRate")
                daily[d]["skin_temp_deviation"] = r.get("avgSkinTempDeviationC")
                daily[d]["sleep_score"] = r.get("sleepScoreValue")
                daily[d]["sleep_score_qualifier"] = r.get("sleepScoreQualifier")
                daily[d]["deep_sleep_pct"] = r.get("deepPct")
                daily[d]["light_sleep_pct"] = r.get("lightPct")
                daily[d]["rem_sleep_pct"] = r.get("remPct")

        # Steps
        if "steps_raw" in raw:
            for _, r in raw["steps_raw"].iterrows():
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["total_steps"] = r.get("totalSteps")
                daily[d]["total_distance_m"] = r.get("totalDistance")
                daily[d]["step_goal"] = r.get("stepGoal")

        # Body battery
        if "body_battery" in raw:
            for _, r in raw["body_battery"].iterrows():
                d = str(r.get("date", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["bb_charged"] = r.get("charged")
                daily[d]["bb_drained"] = r.get("drained")
                bb_arr = r.get("bodyBatteryValuesArray")
                if isinstance(bb_arr, str):
                    try:
                        bb_arr = ast.literal_eval(bb_arr)
                    except Exception:
                        bb_arr = None
                if isinstance(bb_arr, list) and bb_arr:
                    levels = [item[1] for item in bb_arr
                              if isinstance(item, list) and len(item) >= 2
                              and item[1] is not None]
                    if levels:
                        daily[d]["bb_peak"] = max(levels)
                        daily[d]["bb_low"] = min(levels)

        # Intensity minutes
        if "intensity" in raw:
            for _, r in raw["intensity"].iterrows():
                d = str(r.get("calendar_date", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["moderate_intensity_min"] = r.get("moderate_value")
                daily[d]["vigorous_intensity_min"] = r.get("vigorous_value")
                daily[d]["intensity_goal"] = r.get("weekly_goal")

        # Training readiness (AFTER_WAKEUP_RESET only)
        if "training_readiness" in raw:
            for _, r in raw["training_readiness"].iterrows():
                if r.get("inputContext", "") != "AFTER_WAKEUP_RESET":
                    continue
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["training_readiness"] = r.get("score")
                daily[d]["tr_sleep_score"] = r.get("sleepScore")
                daily[d]["tr_sleep_score_pct"] = r.get("sleepScoreFactorPercent")
                daily[d]["tr_recovery_time"] = r.get("recoveryTime")
                daily[d]["tr_recovery_pct"] = r.get("recoveryTimeFactorPercent")
                daily[d]["tr_hrv_weekly_avg"] = r.get("hrvWeeklyAverage")
                daily[d]["tr_hrv_pct"] = r.get("hrvFactorPercent")
                daily[d]["tr_stress_history_pct"] = r.get("stressHistoryFactorPercent")
                daily[d]["tr_acwr_pct"] = r.get("acwrFactorPercent")
                daily[d]["tr_sleep_history_pct"] = r.get("sleepHistoryFactorPercent")
                daily[d]["tr_acute_load"] = r.get("acuteLoad")
                daily[d]["tr_level"] = r.get("level")

        # Hydration
        if "hydration" in raw:
            for _, r in raw["hydration"].iterrows():
                d = str(r.get("calendarDate", ""))
                if not d:
                    continue
                daily.setdefault(d, {"date": d})
                daily[d]["hydration_goal_ml"] = r.get("goalInML")
                daily[d]["hydration_value_ml"] = r.get("valueInML")
                daily[d]["sweat_loss_ml"] = r.get("sweatLossInML")

        # Weight
        if "weight" in raw:
            for _, r in raw["weight"].iterrows():
                d = str(r.get("calendar_date", ""))
                if not d:
                    continue
                w = r.get("weight")
                daily.setdefault(d, {"date": d})
                daily[d]["weight_grams"] = w
                daily[d]["weight_kg"] = round(w / 1000, 1) if w and w > 100 else w

        return daily

    # ─── Data Quality Fixes ───────────────────────────────────

    def _apply_quality_fixes(self, daily: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply all proven data-quality fixes."""
        for d, row in daily.items():
            # Fix 1: HRV weekly avg sentinel 511 → None
            if row.get("tr_hrv_weekly_avg") == 511:
                row["tr_hrv_weekly_avg"] = None

            # Fix 2: Hydration — if value is 0 (user never logs), store NULL
            #         so agents don't analyze fabricated data
            hyd_val = row.get("hydration_value_ml")
            if hyd_val is None or hyd_val == 0:
                row["hydration_value_ml"] = None

        return daily

    # ─── PostgreSQL Upserts ───────────────────────────────────

    def _upsert_daily_metrics(self, cur, daily: Dict[str, Dict]) -> int:
        """UPSERT daily_metrics rows. Returns count."""
        if not daily:
            return 0

        update_cols = [c for c in DAILY_COLS if c != "date"]
        update_str = ", ".join(
            f"{c} = COALESCE(EXCLUDED.{c}, daily_metrics.{c})"
            for c in update_cols
        )
        cols_str = ", ".join(DAILY_COLS)
        placeholders = ", ".join(["%s"] * len(DAILY_COLS))

        upsert_sql = f"""
            INSERT INTO daily_metrics ({cols_str})
            VALUES ({placeholders})
            ON CONFLICT (date) DO UPDATE SET
                {update_str},
                updated_at = NOW()
        """

        count = 0
        for d_key in sorted(daily):
            row = daily[d_key]
            values = tuple(_safe_val(row.get(col)) for col in DAILY_COLS)
            try:
                cur.execute(upsert_sql, values)
                count += 1
            except Exception as e:
                log.info(f"   ⚠️  daily_metrics {d_key}: {e}")
        log.info(f"   ✅ daily_metrics: {count} rows upserted")
        return count

    def _upsert_activities(self, cur, activities_df) -> int:
        """UPSERT activities. Returns count."""
        if activities_df is None or (isinstance(activities_df, pd.DataFrame)
                                     and activities_df.empty):
            return 0

        count = 0
        for _, r in activities_df.iterrows():
            # Derive date from start_time_local (e.g. "2026-02-10 08:30:00")
            stl = str(r.get("start_time_local", ""))
            act_date = stl[:10] if len(stl) >= 10 else None
            try:
                cur.execute("""
                    INSERT INTO activities (
                        activity_id, activity_name, activity_type,
                        start_time_local, start_time_gmt,
                        distance_m, duration_sec, elapsed_duration_sec,
                        moving_duration_sec,
                        elevation_gain_m, elevation_loss_m,
                        average_speed, max_speed, calories,
                        average_hr, max_hr, steps,
                        avg_cadence, max_cadence, owner_id,
                        date
                    ) VALUES (
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                        %s
                    )
                    ON CONFLICT (activity_id) DO NOTHING
                """, (
                    _safe_val(r.get("activity_id")),
                    _safe_val(r.get("activity_name")),
                    _safe_val(r.get("activity_type")),
                    _safe_val(r.get("start_time_local")),
                    _safe_val(r.get("start_time_gmt")),
                    _safe_val(r.get("distance")),
                    _safe_val(r.get("duration")),
                    _safe_val(r.get("elapsed_duration")),
                    _safe_val(r.get("moving_duration")),
                    _safe_val(r.get("elevation_gain")),
                    _safe_val(r.get("elevation_loss")),
                    _safe_val(r.get("average_speed")),
                    _safe_val(r.get("max_speed")),
                    _safe_val(r.get("calories")),
                    _safe_val(r.get("average_hr")),
                    _safe_val(r.get("max_hr")),
                    _safe_val(r.get("steps")),
                    _safe_val(r.get("average_running_cadence_in_steps_per_minute")),
                    _safe_val(r.get("max_running_cadence_in_steps_per_minute")),
                    _safe_val(r.get("owner_id")),
                    act_date,
                ))
                count += 1
            except Exception as e:
                log.info(f"   ⚠️  Activity {r.get('activity_id')}: {e}")
        log.info(f"   ✅ activities: {count} rows upserted")
        return count

    def _upsert_bb_events(self, cur, bb_df) -> int:
        """UPSERT body_battery_events. Returns count."""
        if bb_df is None or (isinstance(bb_df, pd.DataFrame) and bb_df.empty):
            return 0

        count = 0
        for _, r in bb_df.iterrows():
            d = str(r.get("date", ""))
            events_raw = r.get("bodyBatteryActivityEvent")
            if isinstance(events_raw, str):
                try:
                    events = ast.literal_eval(events_raw)
                except Exception:
                    continue
            elif isinstance(events_raw, list):
                events = events_raw
            else:
                continue
            for ev in events:
                try:
                    cur.execute("""
                        INSERT INTO body_battery_events
                            (date, event_type, event_start_gmt,
                             duration_ms, bb_impact,
                             feedback_type, short_feedback)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, event_type, event_start_gmt)
                        DO NOTHING
                    """, (
                        d,
                        ev.get("eventType"),
                        ev.get("eventStartTimeGmt"),
                        ev.get("durationInMilliseconds"),
                        ev.get("bodyBatteryImpact"),
                        ev.get("feedbackType"),
                        ev.get("shortFeedback"),
                    ))
                    count += 1
                except Exception:
                    pass
        log.info(f"   ✅ body_battery_events: {count} events upserted")
        return count

    # ─── Training Intensity Classification ────────────────────

    def classify_training_intensity(self,
                                    start_date: Optional[date] = None
                                    ) -> Dict[date, Dict]:
        """
        Classify each day's training intensity and body parts from
        activities + exercise-sets API.

        Returns {date: {intensity, hard_minutes, has_upper, has_lower,
                        activity_types, exercises}}.
        """
        if not self.authenticated:
            raise RuntimeError("Call authenticate() first")

        conn = psycopg2.connect(self.conn_str)
        start_filter = start_date or (date.today() - timedelta(days=14))
        act_df = pd.read_sql_query(
            f"SELECT * FROM activities "
            f"WHERE start_time_local >= '{start_filter}' "
            f"ORDER BY start_time_local",
            conn,
        )
        conn.close()

        if act_df.empty:
            log.info("   No activities found for classification")
            return {}

        training_days: Dict[date, Dict] = {}

        for _, act_row in act_df.iterrows():
            act_date = pd.Timestamp(act_row["start_time_local"]).date()
            act_type = act_row.get("activity_type", "")
            duration_min = (act_row.get("duration_sec", 0) or 0) / 60
            avg_hr = act_row.get("average_hr") or 0
            max_hr_val = act_row.get("max_hr") or 0
            act_id = act_row.get("activity_id")

            if act_date not in training_days:
                training_days[act_date] = {
                    "intensity": "REST",
                    "has_upper": False,
                    "has_lower": False,
                    "hard_minutes": 0,
                    "easy_minutes": 0,
                    "exercises": [],
                    "activity_types": [],
                }
            day = training_days[act_date]
            day["activity_types"].append(act_type)

            # Intensity rules (proven in test_matrices.py)
            is_hard = False
            if act_type == "strength_training" and duration_min >= 30:
                is_hard = True
            elif act_type == "running" and (avg_hr >= 130 or max_hr_val >= 150):
                is_hard = True
            elif act_type == "basketball" and duration_min >= 30:
                is_hard = True
            elif act_type == "hiking" and duration_min >= 60:
                is_hard = True
            elif act_type == "lap_swimming" and duration_min >= 20:
                is_hard = True

            if is_hard:
                day["hard_minutes"] += duration_min
            else:
                day["easy_minutes"] += duration_min

            # Body-part detection from exercise sets
            if act_type == "strength_training" and act_id:
                try:
                    raw = self._api_call(
                        f"/activity-service/activity/{act_id}/exerciseSets")
                    if isinstance(raw, dict):
                        for ex_set in raw.get("exerciseSets", []):
                            if ex_set.get("setType") != "ACTIVE":
                                continue
                            for exercise in ex_set.get("exercises", []):
                                cat = exercise.get("category", "")
                                day["exercises"].append(cat)
                                if cat in UPPER_BODY_CATS:
                                    day["has_upper"] = True
                                elif cat in LOWER_BODY_CATS:
                                    day["has_lower"] = True
                except Exception:
                    pass

        # Set final intensity per day
        for d, info in training_days.items():
            if info["hard_minutes"] >= 30:
                info["intensity"] = "HARD"
            elif info["hard_minutes"] > 0 or info["easy_minutes"] >= 30:
                info["intensity"] = "MODERATE"
            else:
                info["intensity"] = "EASY"

        n_hard = sum(1 for v in training_days.values()
                     if v["intensity"] == "HARD")
        n_mod = sum(1 for v in training_days.values()
                    if v["intensity"] == "MODERATE")
        log.info(f"   ✅ Training classified: {len(training_days)} days "
              f"({n_hard} HARD, {n_mod} MODERATE)")
        return training_days


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    fetcher = EnhancedGarminDataFetcher()
    if fetcher.authenticate():
        counts = fetcher.fetch_and_store(days=7)
        training = fetcher.classify_training_intensity()
        for d in sorted(training):
            info = training[d]
            body = []
            if info["has_upper"]:
                body.append("UPPER")
            if info["has_lower"]:
                body.append("LOWER")
            body_str = "+".join(body) if body else "-"
            log.info(f"   {d}  {info['intensity']:8s}  "
                  f"hard={info['hard_minutes']:.0f}min  body={body_str}")
