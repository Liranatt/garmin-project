# Deep Dive: Garmin Automated Data Pipeline

This report details the inner workings of your automated health data system. It explains *why* it works, *how* it handles your private data, and the *safety mechanisms* that prevent data loss.

## 1. The Core Architecture

The system operates on a "Push-Pull" hybrid model due to Garmin's privacy constraints.

### The "Trigger" (You)
Garmin blocks automated bots from requesting full data exports. This is a security feature. Therefore, the **only manual step** is the authorization click.
*   **Why**: Prove you are human.
*   **When**: Weekly (recommended) or whenever you want deep historical data.

### The "Watcher" (The Automation)
Once you click that button, the system takes over completely.

## 2. Component Analysis

### A. The Reminder (`src/send_export_reminder.py`)
*   **Role**: The "Alarm Clock".
*   **Mechanism**: A simple Python script scheduled on Heroku. It uses `smtplib` to connect to Gmail via your App Password.
*   **Why it works**: It's stateless. It just wakes up at the scheduled time (e.g., Sunday 8am) and fires an email to your inbox.

### B. The Orchestrator (`src/bulk_import.py`)
This is the heavy lifter. It runs daily (or as scheduled) and performs 4 critical phases:

#### Phase 1: The Lookout (Privacy-Focused)
*   **Code**: `monitor_email_for_link()`
*   **Logic**: Connects to your Gmail via IMAP.
*   **The Filter**: It searches **only** for:
    *   Sender: `noreply@garmin.com`
    *   Subject: contains "export"
    *   Time: Received in the last 24 hours.
*   **Safety**: It *ignores* all other emails. It does not read your personal mail. It only extracts the `https://.../download_request/` link.

#### Phase 2: The Heist (Download & Extract)
*   **Code**: `download_file()` & `extract_zip_recursively()`
*   **Logic**:
    1.  Downloads the multi-gigabyte zip file to a temporary folder (`/tmp`).
    2.  Garmin zips are nested (zips inside zips). The script recursively hunts for the `DI_CONNECT` folder, which contains the actual JSON and FIT files.
    
#### Phase 3: The Staging (Ingestion)
*   **Code**: `run_ingest()`
*   **Logic**: It reuses your existing `ingest.py` logic.
*   **The Staging Area**: It creates a **temporary schema** called `garmin`.
    *   Why? Raw data is messy. We don't want to dump raw, messy data directly into your clean `public` production tables.
    *   It populates `garmin.activities`, `garmin.sleep`, etc., in strict isolation.

#### Phase 4: The Merger & Cleanup (Safety First)
*   **Code**: `run_merge()` & `cleanup_staging_db()`
*   **The Merge**: It executes SQL to select clean data from `garmin` and **UPSERT** (Update/Insert) it into `public`.
    *   *Conflict Handling*: If a record exists, it updates it. If not, it creates it.
*   **The Cleanup**: Once the merge confirms success, it runs `DROP SCHEMA garmin CASCADE`.
    *   *Result*: The temporary 2GB of data vanishes. Your production database stays lean, containing only the high-value metrics.

### C. The Daily Sync (`src/weekly_sync.py`)
*   **Role**: The "Autopilot".
*   **Mechanism**: Uses `enhanced_fetcher.py` and `garth`.
*   **Why it works**: Garmin *does* provide an API for recent data. This script logs in (using saved session tokens) and fetches the last 7 days of data directly.
*   **Usage**: This covers 95% of your needs automatically. The Bulk Import is only for the deep files (GPS traces, raw FIT files) that the API sometimes skips or for backfilling years of history.

## 3. Data Flow Diagram

```mermaid
graph TD
    User((You)) -->|1. Click Export| Garmin[Garmin Servers]
    Scheduler[Heroku Scheduler] -->|2. Runs Daily| Orchestrator[src/bulk_import.py]
    
    Garmin -->|3. Emails Link| Gmail[Your Inbox]
    
    Orchestrator -->|4. Checks IMAP| Gmail
    Orchestrator -->|5. Downloads Zip| Temp[/tmp Storage]
    
    Temp -->|6. Extracts| JSON[JSON/FIT Files]
    
    JSON -->|7. Ingests| Staging[(Staging DB: 'garmin')]
    
    Staging -->|8. Merges| Prod[(Production DB: 'public')]
    
    Orchestrator -->|9. Drops Schema| Staging
```

## 4. Why This is Safe
1.  **Isolation**: Staging data never touches production data until the final merge.
2.  **No Deletes**: The merge process only *adds* or *updates* rows. It never deletes your existing history.
3.  **Ephemeral**: The massive zip files and temporary tables exist for only minutes, preventing Heroku storage overflow.

## 5. Security Summary
*   **Credentials**: Your Gmail App Password and Garmin credentials act as the keys. They are stored in Heroku Config Vars (encrypted), never in the code.
*   **Scope**: The code is strictly scoped to `garmin.com` domains and specific email subjects.

This system gives you the power of a raw data engineer's pipeline with the convenience of a "set it and forget it" automation.
