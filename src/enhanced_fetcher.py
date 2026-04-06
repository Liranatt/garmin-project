"""
Enhanced Garmin Data Fetcher — Production Version
===================================================
Fetches all Garmin data using proven connectapi endpoints and writes
directly to PostgreSQL with fail-fast architecture.
"""
import os
import ast
import logging
import numpy as np
import pandas as pd
import psycopg2
import garth
import imaplib
import email
import re
import html
import time
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from email.header import decode_header
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger("enhanced_fetcher")

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ═══════════════════════════════════════════════════════════════
try:
    from .constants import UPPER_BODY_CATS, LOWER_BODY_CATS
except ImportError:
    from constants import UPPER_BODY_CATS, LOWER_BODY_CATS

DAILY_COLS = [
    "date", "resting_hr", "max_hr", "min_hr", "hrv_last_night", "hrv_5min_high",
    "hrv_weekly_avg", "hrv_status", "stress_level", "rest_stress_sec",
    "low_stress_sec", "medium_stress_sec", "high_stress_sec", "sleep_seconds",
    "deep_sleep_sec", "light_sleep_sec", "rem_sleep_sec", "awake_sleep_sec",
    "sleep_score", "sleep_score_qualifier", "avg_respiration", "lowest_respiration",
    "highest_respiration", "avg_sleep_stress", "awake_count", "sleep_start_local",
    "sleep_end_local", "body_battery_change", "resting_hr_sleep", "skin_temp_deviation",
    "sleep_need_baseline_min", "sleep_need_actual_min", "deep_sleep_pct",
    "light_sleep_pct", "rem_sleep_pct", "total_steps", "total_distance_m",
    "step_goal", "bb_charged", "bb_drained", "bb_peak", "bb_low",
    "moderate_intensity_min", "vigorous_intensity_min", "intensity_goal",
    "training_readiness", "tr_sleep_score", "tr_sleep_score_pct", "tr_recovery_time",
    "tr_recovery_pct", "tr_hrv_weekly_avg", "tr_hrv_pct", "tr_stress_history_pct",
    "tr_acwr_pct", "tr_sleep_history_pct", "tr_acute_load", "tr_level",
    "weight_grams", "weight_kg", "hydration_goal_ml", "hydration_value_ml", "sweat_loss_ml",
]

# ═══════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════
class GarminFetcherError(Exception):
    pass

class AuthenticationError(GarminFetcherError):
    pass

class DatabaseError(GarminFetcherError):
    pass

class MFAError(GarminFetcherError):
    pass

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def deep_vars(obj: Any, prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    try:
        attrs = vars(obj)
    except TypeError:
        return flat
    for k, v in attrs.items():
        if k.startswith("_"): continue
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
            if nested: flat.update(nested)
            else: flat[key] = str(v)[:80]
    return flat

def _safe_val(v: Any) -> Any:
    if v is None: return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return float(v)
    if hasattr(pd, "isna") and pd.isna(v): return None
    return v

# ═══════════════════════════════════════════════════════════════
# FETCHER CLASS
# ═══════════════════════════════════════════════════════════════
class EnhancedGarminDataFetcher:
    def __init__(self):
        self.email = self._get_env("GARMIN_EMAIL")
        self.password = self._get_env("GARMIN_PASSWORD")
        self.session_dir = os.path.expanduser(os.getenv("GARTH_HOME", "~/.garth"))
        self.conn_str = self._get_env("POSTGRES_CONNECTION_STRING")
        self.authenticated = False
        self._last_data = {}

    @staticmethod
    def _get_env(key: str) -> str:
        val = os.getenv(key)
        if not val:
            raise EnvironmentError(f"Missing required environment variable: {key}")
        return val

    def _get_mfa_code_from_email(self) -> str:
        gmail_user = os.getenv("EMAIL_RECIPIENT") or self.email
        gmail_password = self._get_env("EMAIL_APP_PASSWORD")

        log.info("Checking Gmail (%s) for MFA code...", gmail_user)
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(gmail_user, gmail_password)
        mail.select("inbox")

        status, messages = mail.search(None, '(FROM "alerts@account.garmin.com")')
        if status != "OK":
            raise MFAError(f"IMAP search failed with status: {status}")

        email_ids = messages[0].split()
        if not email_ids:
            raise MFAError("No MFA emails found from alerts@account.garmin.com")

        for e_id in email_ids[-5:][::-1]:
            _, data = mail.fetch(e_id, "(RFC822)")
            msg = email.message_from_bytes(data[0][1])

            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            match = re.search(r'account\D*(\d{6})', body, re.IGNORECASE)
            if match:
                code = match.group(1)
                log.info("MFA code extracted from email.")
                mail.logout()
                return code

        mail.logout()
        raise MFAError("MFA code not found in recent Garmin emails.")

    def _restore_session_from_db(self) -> int:
        log.info("Attempting session restoration from database...")
        with psycopg2.connect(self.conn_str, sslmode="require") as conn:
            with conn.cursor() as cur:
                Path(self.session_dir).mkdir(parents=True, exist_ok=True)
                restored = 0
                for key, fname in [
                    ("garth_oauth2_token", "oauth2_token.json"),
                    ("garth_oauth1_token", "oauth1_token.json"),
                ]:
                    cur.execute("SELECT value FROM app_config WHERE key = %s", (key,))
                    row = cur.fetchone()
                    if row:
                        (Path(self.session_dir) / fname).write_text(row[0])
                        restored += 1
                return restored

    def _persist_session_to_db(self) -> None:
        log.info("Persisting fresh session to database...")
        with psycopg2.connect(self.conn_str, sslmode="require") as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                for key, fname in [
                    ("garth_oauth2_token", "oauth2_token.json"),
                    ("garth_oauth1_token", "oauth1_token.json"),
                ]:
                    token_path = Path(self.session_dir) / fname
                    if not token_path.exists():
                        raise DatabaseError(f"Token file missing after login: {fname}")
                    cur.execute("""
                        INSERT INTO app_config (key, value, updated_at)
                        VALUES (%s, %s, NOW())
                        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()
                    """, (key, token_path.read_text()))

    def authenticate(self) -> None:
        """
        Two explicit, non-overlapping paths:

        Path A — tokens exist in DB (token_count > 0):
            Restore tokens to disk -> garth.resume() -> validate with network call.
            Any failure raises AuthenticationError. No fallback to fresh login.
            A bad/expired stored token requires manual re-bootstrap.

        Path B — no tokens in DB (token_count == 0):
            Fresh login with MFA -> garth.save() -> persist tokens to DB.
            This is the first-run / manual bootstrap path only.
        """
        token_count = self._restore_session_from_db()

        if token_count > 0:
            try:
                garth.resume(self.session_dir)
                _ = garth.client.username  # network call — validates token is alive
            except Exception as e:
                raise AuthenticationError(
                    f"Stored tokens exist but are invalid or expired. "
                    f"Re-bootstrap locally to refresh DB tokens. "
                    f"Cause: {type(e).__name__}: {e}"
                ) from e
            self.authenticated = True
            log.info("Garmin session resumed successfully.")
            return

        # Path B: no tokens in DB — first run or after manual token wipe
        log.info("No tokens found in DB. Performing fresh login with MFA.")

        def mfa_callback() -> str:
            log.info("Waiting 15s for MFA email delivery...")
            time.sleep(15)
            return self._get_mfa_code_from_email()

        try:
            garth.login(self.email, self.password, prompt_mfa=mfa_callback)
        except Exception as e:
            raise AuthenticationError(
                f"Fresh Garmin login failed: {type(e).__name__}: {e}"
            ) from e

        self.authenticated = True
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)
        garth.save(self.session_dir)
        self._persist_session_to_db()
        log.info("Fresh login complete. Tokens persisted to DB.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        reraise=True
    )
    def _api_call(self, endpoint: str) -> Any:
        return garth.client.connectapi(endpoint)

    def _apply_quality_fixes(self, daily: Dict[str, Dict]) -> Dict[str, Dict]:
        for d, row in daily.items():
            if row.get("tr_hrv_weekly_avg") == 511:
                row["tr_hrv_weekly_avg"] = None
        return daily

    def fetch_and_store(self, days: int = 7) -> Dict[str, int]:
        if not self.authenticated:
            raise AuthenticationError("Fetcher not authenticated.")

        start = date.today() - timedelta(days=days)
        end = date.today()

        log.info("Fetching data for %d days (%s to %s)", days, start, end)
        self._last_data = self._fetch_all_sources(start, end, days)
        daily = self._build_daily_rows(self._last_data)
        daily = self._apply_quality_fixes(daily)

        log.info("Writing results to database...")
        with psycopg2.connect(self.conn_str, sslmode="require") as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                return {
                    "daily_metrics": self._upsert_daily_metrics(cur, daily),
                    "activities": self._upsert_activities(cur, self._last_data.get("activities")),
                    "bb_events": self._upsert_bb_events(cur, self._last_data.get("body_battery"))
                }

    def classify_training_intensity(self) -> Dict[str, Any]:
        if not self._last_data or "activities" not in self._last_data:
            return {}

        df = self._last_data["activities"]
        if df.empty: return {}

        training = {}
        for _, r in df.iterrows():
            d = str(r.get("start_time_local", ""))[:10]
            if not d: continue

            dur = r.get("duration", 0) / 60.0
            hr_avg = r.get("average_hr", 0)

            intensity = "recovery"
            if hr_avg > 150: intensity = "high"
            elif hr_avg > 130: intensity = "moderate"

            if d not in training:
                training[d] = {"intensity": intensity, "hard_minutes": 0}

            if intensity != "recovery":
                training[d]["hard_minutes"] += int(dur)

            if intensity == "high": training[d]["intensity"] = "high"
            elif intensity == "moderate" and training[d]["intensity"] == "recovery":
                training[d]["intensity"] = "moderate"
        return training

    def _fetch_all_sources(self, start: date, end: date, days: int) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        core_endpoints = {
            "heart_rate": f"/usersummary-service/stats/heartRate/daily/{start}/{end}",
            "body_battery": f"/wellness-service/wellness/bodyBattery/reports/daily?startDate={start}&endDate={end}",
            "steps_raw": f"/usersummary-service/stats/steps/daily/{start}/{end}",
            "training_readiness": f"/metrics-service/metrics/trainingreadiness/{start}/{end}",
            "stress": f"/usersummary-service/stats/stress/daily/{start}/{end}",
            "hrv": f"/hrv-service/hrv/daily/{start}/{end}"
        }

        for name, url in core_endpoints.items():
            res = self._api_call(url)
            if name == "hrv":
                results[name] = pd.json_normalize(res.get("hrvSummaries", []))
            else:
                results[name] = pd.json_normalize(res)

        results["intensity"] = pd.json_normalize([deep_vars(o) for o in garth.DailyIntensityMinutes.list(str(start), days)])
        results["weight"] = pd.json_normalize([deep_vars(o) for o in garth.WeightData.list(str(start), days)])

        acts = garth.Activity.list(limit=50)
        act_rows = [
            {k: v for k, v in vars(a).items() if not k.startswith("_")}
            for a in acts
            if "start_time_local" in vars(a) and isinstance(vars(a)["start_time_local"], str)
        ]
        df = pd.DataFrame(act_rows)
        if not df.empty:
            results["activities"] = df[df["start_time_local"].str[:10].between(str(start), str(end))]

        return results

    def _build_daily_rows(self, raw: Dict[str, Any]) -> Dict[str, Dict]:
        daily: Dict[str, Dict] = {}
        for name, df in raw.items():
            if not isinstance(df, pd.DataFrame) or df.empty: continue

            date_col = next((c for c in ["calendarDate", "date", "calendar_date"] if c in df.columns), None)
            if not date_col: continue

            for _, r in df.iterrows():
                d = str(r[date_col])
                daily.setdefault(d, {"date": d})

                if name == "heart_rate":
                    daily[d].update({
                        "resting_hr": r.get("values.restingHR"),
                        "max_hr": r.get("values.wellnessMaxAvgHR"),
                        "min_hr": r.get("values.wellnessMinAvgHR")
                    })
                elif name == "hrv":
                    daily[d].update({"hrv_last_night": r.get("lastNightAvg"), "hrv_status": r.get("status")})
                elif name == "training_readiness" and r.get("inputContext") == "AFTER_WAKEUP_RESET":
                    daily[d].update({"training_readiness": r.get("score"), "tr_recovery_time": r.get("recoveryTime")})

        if not daily:
            raise GarminFetcherError("No daily metrics could be constructed from raw data.")
        return daily

    def _upsert_daily_metrics(self, cur: Any, daily: Dict[str, Dict]) -> int:
        count = 0
        for d_key in sorted(daily):
            row = daily[d_key]
            cols = [c for c in DAILY_COLS if c in row and row[c] is not None]
            if not cols or (len(cols) == 1 and cols[0] == "date"):
                continue

            vals = [_safe_val(row[c]) for c in cols]
            upd = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols if c != "date"])
            sql = (
                f"INSERT INTO daily_metrics ({','.join(cols)}) "
                f"VALUES ({','.join(['%s'] * len(cols))}) "
                f"ON CONFLICT (date) DO UPDATE SET {upd}, updated_at = NOW()"
            )
            cur.execute(sql, vals)
            count += 1
        return count

    def _upsert_activities(self, cur: Any, df: pd.DataFrame) -> int:
        if df is None or df.empty: return 0
        for _, r in df.iterrows():
            cur.execute(
                "INSERT INTO activities (activity_id, date) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (r.get("activity_id"), str(r.get("start_time_local", ""))[:10])
            )
        return len(df)

    def _upsert_bb_events(self, cur: Any, df: pd.DataFrame) -> int:
        return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    fetcher = EnhancedGarminDataFetcher()
    fetcher.authenticate()
    res = fetcher.fetch_and_store(days=2)
    log.info("Sync complete: %s", res)
