
"""
Garmin Bulk Import Orchestrator
===============================

Automation flow:
1. Weekly reminder email asks user to trigger Garmin export manually.
2. Daily job scans Gmail for latest Garmin export-ready email.
3. If newest matching email is fresh (<= 7 days), download ZIP.
4. Extract nested ZIPs and ingest into staging schema (`garmin`).
5. Merge incremental deltas to production tables.
6. Optional staging cleanup (disabled by default).
"""

import sys
import os
import shutil
import zipfile
import re
import imaplib
import email
import logging
import argparse
import traceback
import html
import importlib
from dataclasses import dataclass
from email.header import decode_header
from email.utils import parsedate_to_datetime
import requests
import psycopg2
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path to import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import config (will be monkeypatched)
sys.path.append(str(PROJECT_ROOT / "mass_donwload_code_decoder"))
import config
import db

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("bulk_import")

# Constants
GARMIN_EXPORT_URL = "https://www.garmin.com/en-US/account/datamanagement/exportdata/"
IMAP_URL = "imap.gmail.com"
DEFAULT_LOOKBACK_DAYS = 45
DEFAULT_MAX_EMAIL_AGE_DAYS = 7

TEMP_DIR = Path(os.environ.get("TEMP", "/tmp")) / "garmin_bulk_import"
DOWNLOAD_DIR = TEMP_DIR / "downloads"
EXTRACT_DIR = TEMP_DIR / "extracted"
URL_RE = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)


@dataclass
class ExportEmailCandidate:
    uid: bytes
    sender: str
    subject: str
    received_at: datetime
    download_url: str | None

def setup_env():
    """Normalize environment variables for Heroku/local execution."""
    # Ensure POSTGRES connection string exists.
    if not os.getenv("POSTGRES_CONNECTION_STRING") and os.getenv("DATABASE_URL"):
        os.environ["POSTGRES_CONNECTION_STRING"] = os.getenv("DATABASE_URL")

    if not os.getenv("POSTGRES_CONNECTION_STRING"):
        raise RuntimeError("POSTGRES_CONNECTION_STRING (or DATABASE_URL) is required")

    # Keep staging and production in same Heroku DB, different schemas.
    if not os.getenv("GARMIN_CONNECTION_STRING"):
        os.environ["GARMIN_CONNECTION_STRING"] = os.environ["POSTGRES_CONNECTION_STRING"]

    if not os.getenv("EMAIL_APP_PASSWORD"):
        raise RuntimeError("EMAIL_APP_PASSWORD is required for Gmail IMAP access")

    # IMAP login identity (separate from notification recipient).
    imap_user = (
        os.getenv("EMAIL_IMAP_USER")
        or os.getenv("EMAIL_RECIPIENT")
        or os.getenv("GARMIN_EMAIL")
    )
    if not imap_user:
        raise RuntimeError(
            "EMAIL_IMAP_USER (or fallback EMAIL_RECIPIENT/GARMIN_EMAIL) is required for Gmail IMAP access"
        )
    os.environ["EMAIL_IMAP_USER"] = imap_user

    # Recipient for notifications, independent from IMAP login account.
    if not os.getenv("EMAIL_RECIPIENT"):
        os.environ["EMAIL_RECIPIENT"] = os.getenv("GARMIN_EMAIL", imap_user)

def request_export_from_garmin():
    """Manual trigger reminder (Garmin export request is user-initiated)."""
    log.info("Bulk export request remains manual.")
    log.info("Please trigger export at: %s", GARMIN_EXPORT_URL)
    return True

def _decode_header_value(value: str | None) -> str:
    if not value:
        return ""
    out = []
    for chunk, enc in decode_header(value):
        if isinstance(chunk, bytes):
            out.append(chunk.decode(enc or "utf-8", errors="ignore"))
        else:
            out.append(chunk)
    return "".join(out).strip()


def _message_datetime(msg: email.message.Message) -> datetime:
    raw = msg.get("Date")
    if raw:
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _extract_text_and_html(msg: email.message.Message) -> tuple[str, str]:
    text_parts: list[str] = []
    html_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = (part.get_content_type() or "").lower()
            if ctype not in ("text/plain", "text/html"):
                continue
            payload = part.get_payload(decode=True)
            if payload is None:
                continue
            charset = part.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="ignore")
            if ctype == "text/plain":
                text_parts.append(decoded)
            else:
                html_parts.append(decoded)
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="ignore")
            if (msg.get_content_type() or "").lower() == "text/html":
                html_parts.append(decoded)
            else:
                text_parts.append(decoded)

    return "\n".join(text_parts), "\n".join(html_parts)


def _html_to_text(raw_html: str) -> str:
    no_script = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
    no_style = re.sub(r"(?is)<style.*?>.*?</style>", " ", no_script)
    stripped = re.sub(r"(?is)<[^>]+>", " ", no_style)
    return html.unescape(re.sub(r"\s+", " ", stripped)).strip()


def _looks_like_export_ready(subject: str, plain_text: str, html_text: str) -> bool:
    blob = f"{subject}\n{plain_text}\n{_html_to_text(html_text)}".lower()
    if "download your data" in blob:
        return True
    if "your garmin data export request" in blob:
        return True
    if "garmin" in blob and "export" in blob and "download" in blob:
        return True
    return False


def _extract_download_url(subject: str, plain_text: str, html_text: str) -> str | None:
    candidates = set()
    preferred = set()

    if html_text:
        decoded_html = html.unescape(html_text)
        for href, inner in re.findall(r"(?is)<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", decoded_html):
            h = href.strip(").,;\"'")
            candidates.add(h)
            inner_text = _html_to_text(inner).lower()
            if "download" in inner_text:
                preferred.add(h)

    for src in (subject, plain_text, html_text):
        if not src:
            continue
        decoded_src = html.unescape(src)
        for m in URL_RE.findall(decoded_src):
            candidates.add(m.strip(").,;\"'"))
    if not candidates:
        return None

    s3_zip_links = [u for u in candidates if "s3.amazonaws.com" in u.lower() and ".zip" in u.lower()]
    if s3_zip_links:
        # Prefer active pre-signed links if available; else return newest-looking one.
        active = [u for u in s3_zip_links if not _is_presigned_expired(u)]
        if active:
            return active[0]
        return s3_zip_links[0]

    def is_non_download_asset(url: str) -> bool:
        u = url.lower().split("?", 1)[0]
        if u.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".woff", ".woff2", ".css", ".js")):
            return True
        if "fonts.gstatic.com" in u or "fonts.googleapis.com" in u:
            return True
        if "res.garmin.com/email/garmin/assets/" in u:
            return True
        return False

    def score(url: str) -> int:
        u = url.lower()
        if is_non_download_asset(url):
            return -100
        s = 0
        if url in preferred:
            s += 12
        if "s3.amazonaws.com" in u and ".zip" in u:
            s += 12
        if "it-gdpr-bucket" in u:
            s += 8
        if ".zip?" in u or u.endswith(".zip"):
            s += 6
        if "garmin.com" in u:
            s += 4
        if "download" in u:
            s += 3
        if "exportdata" in u:
            s += 2
        if "download_request" in u:
            s += 2
        if "/account/datamanagement/exportdata/" in u and "download" not in u and ".zip" not in u:
            s -= 6
        if "utm_campaign" in u and ".zip" not in u and "download_request" not in u:
            s -= 3
        if "fonts.gstatic" in u or "fonts.googleapis" in u:
            s -= 4
        return s

    ranked = sorted(candidates, key=score, reverse=True)
    for u in ranked:
        if score(u) < 3:
            continue
        if not _is_presigned_expired(u):
            return u

    best = ranked[0]
    return best if score(best) >= 3 else None


def _is_presigned_expired(url: str) -> bool:
    """Detect expiry for AWS-style pre-signed URLs."""
    low = url.lower()
    if "x-amz-date=" not in low or "x-amz-expires=" not in low:
        return False
    try:
        from urllib.parse import urlparse, parse_qs

        qs = parse_qs(urlparse(url).query)
        amz_date = (qs.get("X-Amz-Date") or qs.get("x-amz-date") or [""])[0]
        amz_expires = (qs.get("X-Amz-Expires") or qs.get("x-amz-expires") or [""])[0]
        if not amz_date or not amz_expires:
            return False

        signed_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        expiry = signed_at + timedelta(seconds=int(amz_expires))
        return datetime.now(timezone.utc) > expiry
    except Exception:
        return False


def find_latest_export_email(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> ExportEmailCandidate | None:
    """Return newest Garmin export-ready email candidate, if any."""
    user = os.getenv("EMAIL_IMAP_USER") or os.getenv("EMAIL_RECIPIENT")
    password = os.getenv("EMAIL_APP_PASSWORD")
    if not user or not password:
        raise RuntimeError("EMAIL_IMAP_USER (or EMAIL_RECIPIENT) and EMAIL_APP_PASSWORD are required")

    since_str = (date.today() - timedelta(days=lookback_days)).strftime("%d-%b-%Y")
    mail = imaplib.IMAP4_SSL(IMAP_URL)
    candidates: list[ExportEmailCandidate] = []
    try:
        try:
            mail.login(user, password)
        except imaplib.IMAP4.error as e:
            raise RuntimeError(
                "IMAP login failed. Verify EMAIL_IMAP_USER matches the Gmail account "
                "that owns EMAIL_APP_PASSWORD, and ensure the app password is active."
            ) from e
        mail.select("INBOX")
        status, msg_ids = mail.search(None, f'(SINCE "{since_str}")')
        if status != "OK" or not msg_ids or not msg_ids[0]:
            return None

        for uid in reversed(msg_ids[0].split()):
            status, data = mail.fetch(uid, "(RFC822)")
            if status != "OK" or not data or not data[0]:
                continue
            msg = email.message_from_bytes(data[0][1])
            sender = _decode_header_value(msg.get("From"))
            if "garmin" not in sender.lower():
                continue
            subject = _decode_header_value(msg.get("Subject"))
            plain_text, html_text = _extract_text_and_html(msg)
            if not _looks_like_export_ready(subject, plain_text, html_text):
                continue
            candidates.append(
                ExportEmailCandidate(
                    uid=uid,
                    sender=sender,
                    subject=subject,
                    received_at=_message_datetime(msg),
                    download_url=_extract_download_url(subject, plain_text, html_text),
                )
            )
    finally:
        try:
            mail.close()
        except Exception:
            pass
        try:
            mail.logout()
        except Exception:
            pass

    if not candidates:
        return None
    return max(candidates, key=lambda c: c.received_at)


def monitor_email_for_link(
    max_email_age_days: int = DEFAULT_MAX_EMAIL_AGE_DAYS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[str | None, ExportEmailCandidate | None, str]:
    """Return (url, candidate, reason). url is None when no fresh export is ready."""
    candidate = find_latest_export_email(lookback_days=lookback_days)
    if candidate is None:
        return None, None, "No Garmin export-ready email found."

    age_days = (datetime.now(timezone.utc) - candidate.received_at).total_seconds() / 86400.0
    if age_days > max_email_age_days:
        return (
            None,
            candidate,
            f"Latest export email is stale ({age_days:.1f} days old). Waiting for a fresh one.",
        )
    if not candidate.download_url:
        return None, candidate, "Latest matching email has no detectable download URL."
    if _is_presigned_expired(candidate.download_url):
        return None, candidate, "Latest Garmin download link is already expired. Request a new export email."
    return candidate.download_url, candidate, ""

def download_file(url: str, session=None) -> Path:
    """Download Garmin export ZIP and validate it."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    local_filename = DOWNLOAD_DIR / f"garmin_export_{int(datetime.now().timestamp())}.zip"
    log.info("Downloading Garmin export to %s ...", local_filename)

    client = session or requests.Session()
    with client.get(url, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    if not zipfile.is_zipfile(local_filename):
        raise ValueError(f"Downloaded file is not a zip: {local_filename}")

    log.info("Download complete.")
    return local_filename

def extract_zip_recursively(zip_path: Path) -> Path:
    """Extract root ZIP and all nested ZIPs, then locate Garmin data root."""
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Extracting %s to %s ...", zip_path, EXTRACT_DIR)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_DIR)

    processed = set()
    while True:
        found_new = False
        for nested in EXTRACT_DIR.rglob("*.zip"):
            key = str(nested.resolve())
            if key in processed:
                continue
            processed.add(key)
            found_new = True
            try:
                with zipfile.ZipFile(nested, "r") as z:
                    z.extractall(nested.parent)
            except Exception as e:
                log.warning("Failed nested unzip (%s): %s", nested.name, e)
        if not found_new:
            break

    for p in EXTRACT_DIR.rglob("DI_CONNECT"):
        if p.is_dir():
            log.info("Found Garmin data root: %s", p)
            return p

    for p in EXTRACT_DIR.rglob("*"):
        if p.is_dir() and ((p / "DI-Connect-User").exists() or (p / "DI-Connect-Fitness").exists()):
            log.info("Found Garmin data root: %s", p)
            return p

    raise ValueError("Could not find DI_CONNECT payload in extracted files.")

def get_latest_db_date():
    """Get the latest date from production DB to determine incremental start."""
    try:
        conn = psycopg2.connect(os.getenv("POSTGRES_CONNECTION_STRING"))
        cur = conn.cursor()
        cur.execute("SELECT MAX(date) FROM daily_metrics")
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0]:
            return row[0]
    except Exception as e:
        log.warning(f"Could not fetch latest DB date: {e}")
    return None

def run_ingest(connect_dir: Path, overlap_days: int = 7):
    """Run ingest pipeline into staging schema with incremental cutoff."""
    log.info("Starting INGEST pipeline...")
    
    # 1. Determine Start Date (Lazy Incremental)
    latest_date = get_latest_db_date()
    start_date = date(2020, 1, 1)  # Default
    
    if latest_date:
        start_date = latest_date - timedelta(days=overlap_days)
        log.info(
            "🔄 INCREMENTAL MODE: Latest DB date is %s. Ingesting from %s ...",
            latest_date,
            start_date,
        )
    else:
        log.info("🆕 FULL IMPORT MODE: No existing data found. Ingesting from 2020-01-01.")

    # Monkeypatch config
    config.CONNECT_DIR = connect_dir
    
    # Initialize DB (garmin schema)
    schema_path = PROJECT_ROOT / "mass_donwload_code_decoder" / "schema_garmin.sql"
    if schema_path.exists():
        log.info("Ensuring 'garmin' schema exists...")
        conn = db.get_connection()
        cur = conn.cursor()
        try:
            with open(schema_path, encoding="utf-8") as f:
                cur.execute(f.read())
            conn.commit()
            log.info("Schema init OK.")
        except Exception as e:
            conn.rollback()
            log.warning(f"Schema init warning: {e}")
        finally:
            cur.close()
            conn.close()
            
    # Run ingestors
    import ingest as ingest_mod
    ingest = importlib.reload(ingest_mod)

    ingest.CONNECT_DIR = connect_dir
    ingest.DATE_FROM = start_date
    config.DATE_FROM = start_date
    
    conn = db.get_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    for fn in ingest.ALL_INGESTORS:
        try:
            fn(cur)
        except Exception as e:
            log.error(f"Ingestor {fn.__name__} failed: {e}")
            
    cur.close()
    conn.close()
    log.info("Ingest complete.")
    return start_date

def run_merge(start_date=None):
    """Run the merge to production pipeline."""
    log.info("Starting MERGE pipeline...")
    import garmin_merge
    
    # Use subprocess to call proper args or mock sys.argv
    args = ["garmin_merge.py"]
    if start_date:
        args.extend(["--since", str(start_date)])
        
    old_argv = sys.argv
    sys.argv = args
    try:
        garmin_merge.main()
    except SystemExit as e:
        if e.code != 0:
            log.error("Merge failed.")
            raise
    except Exception as e:
        log.error(f"Merge error: {e}")
        raise
    finally:
        sys.argv = old_argv
        
    log.info("Merge complete.")

def cleanup_staging_db():
    """Drop the garmin schema to free up space."""
    log.info("Cleaning up staging database (dropping 'garmin' schema)...")
    conn = db.get_connection()
    cur = conn.cursor()
    try:
        cur.execute("DROP SCHEMA IF EXISTS garmin CASCADE")
        conn.commit()
        log.info("Staging data cleaned up.")
    except Exception as e:
        conn.rollback()
        log.error(f"Failed to cleanup staging DB: {e}")
    finally:
        cur.close()
        conn.close()

def run_bulk_import(
    auto: bool = False,
    zip_path: str | None = None,
    cleanup_after: bool = False,
    max_email_age_days: int = DEFAULT_MAX_EMAIL_AGE_DAYS,
    overlap_days: int = 7,
    fail_on_pending: bool = False,
):
    """Run the bulk import pipeline.

    In auto mode:
    - if no fresh Garmin download email is available, returns True (no-op)
      unless fail_on_pending=True.
    """
    setup_env()
    
    local_zip = None
    if zip_path:
        local_zip = Path(zip_path)
        if not local_zip.exists():
            log.error("Provided zip path does not exist: %s", local_zip)
            return False
    elif auto:
        # 1. Manual trigger reminder
        request_export_from_garmin()
        
        # 2. Find newest export-ready email and validate freshness
        link, candidate, reason = monitor_email_for_link(
            max_email_age_days=max_email_age_days,
            lookback_days=max(30, max_email_age_days * 5),
        )
        if not link:
            if candidate is not None:
                log.info(
                    "Newest Garmin email candidate: subject='%s', date=%s",
                    candidate.subject,
                    candidate.received_at.isoformat(),
                )
            log.warning(reason)
            return not fail_on_pending
            
        # 3. Download
        try:
            local_zip = download_file(link)
        except Exception as e:
            log.error(f"Download failed: {e}")
            return False
    else:
        log.error("Must provide either auto=True or zip_path.")
        return False

    if not local_zip or not local_zip.exists():
        log.error("Invalid zip path.")
        return False
        
    # 4. Extract
    try:
        connect_dir = extract_zip_recursively(local_zip)
        
        # 5. Ingest
        start_date = run_ingest(connect_dir, overlap_days=overlap_days)
        
        # 6. Merge
        run_merge(start_date)
        
        # 7. Optional cleanup DB
        if cleanup_after:
            cleanup_staging_db()
        else:
            log.info("Staging cleanup skipped (cleanup_after=False).")
        
        # Cleanup Files
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            log.info("Cleaned up temp files.")
            
        return True
            
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Garmin Bulk Import")
    parser.add_argument("--auto", action="store_true", help="Find fresh Garmin export email and process automatically")
    parser.add_argument("--zip-path", help="Path to local zip file")
    parser.add_argument("--cleanup-staging", action="store_true", help="Drop garmin staging schema after successful merge")
    parser.add_argument("--max-email-age-days", type=int, default=DEFAULT_MAX_EMAIL_AGE_DAYS, help="Ignore Garmin export emails older than this")
    parser.add_argument("--overlap-days", type=int, default=7, help="Re-ingest this many days before latest production date")
    parser.add_argument("--strict-pending", action="store_true", help="Exit non-zero when no fresh Garmin export email is available")
    args = parser.parse_args()
    
    if not args.auto and not args.zip_path:
        parser.print_help()
        sys.exit(1)
        
    success = run_bulk_import(
        auto=args.auto,
        zip_path=args.zip_path,
        cleanup_after=args.cleanup_staging,
        max_email_age_days=args.max_email_age_days,
        overlap_days=args.overlap_days,
        fail_on_pending=args.strict_pending,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
