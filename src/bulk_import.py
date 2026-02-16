
"""
Garmin Bulk Import Orchestrator
===============================

Automates the end-to-end flow:
1. (Auto) Request data export from Garmin (if possible via garth)
2. (Auto) Monitor Gmail for download link
3. Download zip file
4. Extract nested zips
5. Ingest JSON/FIT into 'garmin' schema (staging)
6. Merge into public schema (production)

Usage:
    python src/bulk_import.py --auto         # Full automation
    python src/bulk_import.py --zip-path X   # Manual zip provided
"""

import sys
import os
import time
import shutil
import zipfile
import re
import imaplib
import email
import logging
import argparse
import requests
from datetime import date, datetime
from pathlib import Path

# Add project root to path to import modules
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import config (will be monkeypatched)
sys.path.append(str(PROJECT_ROOT / "mass_donwload_code_decoder"))
import config
import db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("bulk_import")

# Constants
GARMIN_EXPORT_URL = "https://www.garmin.com/en-US/account/datamanagement/exportdata/" # UI URL, API might differ
# For now, we rely on the user manually dealing with the export request if garth fails, 
# or we implement a simple request if we can reverse engineer it.
# The user wants "sending a request... getting to my email... download... straight to heroku".

TEMP_DIR = Path(os.environ.get("TEMP", "/tmp")) / "garmin_bulk_import"

def setup_env():
    """Ensure environment variables are set."""
    required = ["GARMIN_EMAIL", "GARMIN_PASSWORD", "EMAIL_APP_PASSWORD", "EMAIL_RECIPIENT", "DATABASE_URL"]
    missing = [v for v in required if not os.getenv(v) and not os.getenv("POSTGRES_CONNECTION_STRING")]
    if missing:
        log.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)
    
    # Ensure POSTGRES_CONNECTION_STRING is set (ingest/merge rely on it or similar)
    if not os.getenv("POSTGRES_CONNECTION_STRING") and os.getenv("DATABASE_URL"):
        os.environ["POSTGRES_CONNECTION_STRING"] = os.getenv("DATABASE_URL")
    
    # Ensure GARMIN_CONNECTION_STRING is set to the SAME database for this simplified flow
    if not os.getenv("GARMIN_CONNECTION_STRING"):
        os.environ["GARMIN_CONNECTION_STRING"] = os.environ["POSTGRES_CONNECTION_STRING"]

def request_export_from_garmin():
    """Attempt to request data export via garth."""
    import garth
    try:
        garth.login(os.getenv("GARMIN_EMAIL"), os.getenv("GARMIN_PASSWORD"))
        # This is the tricky part. The export is usually a UI action.
        # We'll try to hit the endpoint if known, otherwise we might warn the user.
        # Research suggests there is no public API. 
        # However, we can try to use the sso session.
        # For now, we'll log that this step is experimental.
        log.info("Logged in to Garmin via garth.")
        
        # Todo: Implement actual POST if we find the endpoint. 
        # For now, we will assume the user has clicked the button OR we are polling for an existing email.
        # If the user wants full automation, we might need a browser tool (selenium) but that's heavy.
        # Let's check if there's a simpler endpoint.
        # Based on research, it's not simple. 
        # WE WILL SKIP THE REQUEST STEP FOR NOW AND JUST MONITOR EMAIL.
        # The user said "connect sending a request", implying they want it. 
        # But without a known API, we can't do it reliably without a browser.
        log.warning("Automated export request is not fully implemented (requires browser automation).")
        log.warning("Please trigger the export manually at: https://www.garmin.com/en-US/account/datamanagement/exportdata/")
        log.info("Proceeding to monitor email...")
        return True
    except Exception as e:
        log.error(f"Garth login failed: {e}")
        return False

def monitor_email_for_link(timeout_hours=2):
    """Poll Gmail for the Garmin export link."""
    user = os.getenv("EMAIL_RECIPIENT")
    password = os.getenv("EMAIL_APP_PASSWORD")
    imap_url = 'imap.gmail.com'
    
    log.info(f"Connecting to Gmail ({user})... polling for Garmin email.")
    
    start_time = time.time()
    while (time.time() - start_time) < (timeout_hours * 3600):
        try:
            mail = imaplib.IMAP4_SSL(imap_url)
            mail.login(user, password)
            mail.select("INBOX")
            
            # Search for emails from Garmin with "export" or "download" in subject
            # filtering for recent emails (last 24h)
            status, messages = mail.search(None, '(FROM "noreply@garmin.com" SUBJECT "export" SINCE "{}")'.format(
                date.today().strftime("%d-%b-%Y")
            ))
            
            if status == "OK":
                for num in reversed(messages[0].split()):
                    _, data = mail.fetch(num, '(RFC822)')
                    msg = email.message_from_bytes(data[0][1])
                    subject = msg["subject"]
                    log.info(f"Found email: {subject}")
                    
                    # Extract link
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                body = part.get_payload(decode=True).decode()
                                break
                    else:
                        body = msg.get_payload(decode=True).decode()
                    
                    # Look for url
                    # Usually: https://www.garmin.com/en-US/account/datamanagement/exportdata/download/...
                    match = re.search(r'(https://[^ \n"]*garmin\.com[^ \n"]*export[^ \n"]*)', body)
                    if match:
                        url = match.group(1)
                        log.info(f"Found download URL: {url}")
                        return url
            
            mail.close()
            mail.logout()
            
        except Exception as e:
            log.warning(f"IMAP poll error: {e}")
        
        log.info("Waiting 60s...")
        time.sleep(60)
        
    return None

def download_file(url, session=None):
    """Download the zip file to a temp path."""
    if not TEMP_DIR.exists():
        TEMP_DIR.mkdir(parents=True)
    
    local_filename = TEMP_DIR / "garmin_export.zip"
    log.info(f"Downloading to {local_filename}...")
    
    # We might need authentication for the download link if it's protected
    # Usually the link in email works directly or redirects to login.
    # If it redirects, we might need garth session.
    
    if session:
        r = session.get(url, stream=True)
    else:
        r = requests.get(url, stream=True)
        
    r.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            
    log.info("Download complete.")
    return local_filename

def extract_zip_recursively(zip_path):
    """Extract zip and any nested zips, looking for DI_CONNECT."""
    extract_root = TEMP_DIR / "extracted"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir()
    
    log.info(f"Extracting {zip_path} to {extract_root}...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_root)
        
    # Find nested zips (Garmin exports often have a big zip with `_Part1.zip` inside)
    # Recursively extract them
    for root, dirs, files in os.walk(extract_root):
        for file in files:
            if file.lower().endswith(".zip"):
                fp = Path(root) / file
                log.info(f"Extracting nested zip: {file}")
                try:
                    with zipfile.ZipFile(fp, 'r') as z:
                        z.extractall(Path(root))
                    # Check if it created a folder or just dumped files
                except Exception as e:
                    log.warning(f"Failed to extract {file}: {e}")
                    
    # Locate DI_CONNECT
    connect_dir = None
    for root, dirs, files in os.walk(extract_root):
        if "DI_CONNECT" in dirs: # Found the parent of DI_CONNECT? No, usually DI_CONNECT is the folder
            connect_dir = Path(root) / "DI_CONNECT"
            break
        if Path(root).name == "DI_CONNECT":
            connect_dir = Path(root)
            break
            
    # Fallback: look for generic Garmin structure
    if not connect_dir:
        # Try to find DI-Connect-User as a proxy
        for root, dirs, files in os.walk(extract_root):
            if "DI-Connect-User" in dirs:
                connect_dir = Path(root)
                break
                
    if not connect_dir:
        raise ValueError("Could not find DI_CONNECT directory in extracted files.")
        
    log.info(f"Found Garmin data at: {connect_dir}")
    return connect_dir

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

def run_ingest(connect_dir):
    """Run the ingest pipeline with INCREMENTAL filtering."""
    log.info("Starting INGEST pipeline...")
    
    # 1. Determine Start Date (Lazy Incremental)
    latest_date = get_latest_db_date()
    start_date = date(2020, 1, 1)  # Default
    
    if latest_date:
        # Buffer of 14 days to catch updates/corrections
        start_date = latest_date - timedelta(days=14)
        log.info(f"🔄 INCREMENTAL MODE: Latest DB date is {latest_date}. Ingesting from {start_date}...")
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
    import ingest
    # Filter by date!
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

def run_bulk_import(auto=False, zip_path=None):
    """Run the bulk import pipeline."""
    setup_env()
    
    local_zip = None
    if zip_path:
        local_zip = Path(zip_path)
    elif auto:
        # 1. Request (Optional/Manual for now)
        request_export_from_garmin()
        
        # 2. Monitor Email
        link = monitor_email_for_link() # Blocks until email found
        if not link:
            log.error("No email link found. Exiting.")
            return False
            
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
        start_date = run_ingest(connect_dir)
        
        # 6. Merge
        run_merge(start_date)
        
        # 7. Cleanup DB
        cleanup_staging_db()
        
        # Cleanup Files
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR)
            log.info("Cleaned up temp files.")
            
        return True
            
    except Exception as e:
        log.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Garmin Bulk Import")
    parser.add_argument("--auto", action="store_true", help="Monitor email and download automatically")
    parser.add_argument("--zip-path", help="Path to local zip file")
    args = parser.parse_args()
    
    if not args.auto and not args.zip_path:
        parser.print_help()
        sys.exit(1)
        
    success = run_bulk_import(auto=args.auto, zip_path=args.zip_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
