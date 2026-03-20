# Garmin Project — Audit & Fix Plan
_Date: 2026-03-20_

---

## Part 1: Root Causes of the Daily Sync Failures

### Bug 1 — `time` not imported in `enhanced_fetcher.py` (caused Mar 18 crash)

Inside `authenticate()`, the `mfa_callback` closure calls `time.sleep(15)`
but `import time` is missing from the file entirely.

Fix: Add to the top of enhanced_fetcher.py:

    import time

---

### Bug 2 — Garth session persistence is fundamentally broken (causes all 429 errors)

The workflow writes the garth session to $GITHUB_ENV in the "Save garth session" step,
NOT back to the GitHub secret. $GITHUB_ENV is ephemeral — it dies when the run ends.

So every run:
  1. "Restore garth session" reads the stale/empty secret
  2. garth.resume() fails
  3. Falls through to garth.login(email, password)
  4. Garmin 429s you for repeated logins from the same CI IP

Fix: Store the garth OAuth2 token in PostgreSQL instead.

---

#### Step 1 — Add app_config table to enhanced_schema.py

    CREATE TABLE IF NOT EXISTS app_config (
        key        TEXT PRIMARY KEY,
        value      TEXT NOT NULL,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );

Also add this to src/pipeline/migrations.py so it's created on startup.

---

#### Step 2 — Add two methods to EnhancedGarminDataFetcher in enhanced_fetcher.py

    def _restore_session_from_db(self) -> None:
        """Pull saved garth OAuth2 token from DB and write to disk."""
        if not self.conn_str:
            return
        try:
            conn = psycopg2.connect(self.conn_str, sslmode="require")
            cur = conn.cursor()
            cur.execute("SELECT value FROM app_config WHERE key = 'garth_oauth2_token'")
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row:
                Path(self.session_dir).mkdir(parents=True, exist_ok=True)
                (Path(self.session_dir) / "oauth2_token.json").write_text(row[0])
                log.info("Restored garth session from DB")
        except Exception as e:
            log.warning("Could not restore session from DB: %s", e)


    def _persist_session_to_db(self) -> None:
        """Save current garth OAuth2 token to DB for next run."""
        token_path = Path(self.session_dir) / "oauth2_token.json"
        if not token_path.exists():
            return
        try:
            token_data = token_path.read_text()
            conn = psycopg2.connect(self.conn_str, sslmode="require")
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO app_config (key, value, updated_at)
                VALUES ('garth_oauth2_token', %s, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value,
                        updated_at = NOW()
            """, (token_data,))
            cur.close()
            conn.close()
            log.info("Garth session persisted to DB")
        except Exception as e:
            log.warning("Could not persist session to DB: %s", e)

---

#### Step 3 — Wire them up inside authenticate()

At the very start of authenticate(), BEFORE garth.resume():

    self._restore_session_from_db()

After garth.save(self.session_dir) on successful login:

    self._persist_session_to_db()

---

#### Step 4 — New daily_sync.yml (remove both garth session steps entirely)

    name: Daily Health Sync
    on:
      schedule:
        - cron: '0 6 * * *'
      workflow_dispatch:

    jobs:
      sync:
        runs-on: ubuntu-latest
        timeout-minutes: 30
        steps:
          - uses: actions/checkout@v4

          - uses: actions/setup-python@v5
            with:
              python-version: '3.12'
              cache: 'pip'

          - name: Install dependencies
            run: pip install -r requirements.txt

          - name: Run daily sync pipeline
            env:
              POSTGRES_CONNECTION_STRING: ${{ secrets.POSTGRES_CONNECTION_STRING }}
              GARMIN_EMAIL: ${{ secrets.GARMIN_EMAIL }}
              GARMIN_PASSWORD: ${{ secrets.GARMIN_PASSWORD }}
              GOOGLE_API_KEY: ${{ secrets.GOOGLE_API_KEY }}
              EMAIL_APP_PASSWORD: ${{ secrets.EMAIL_APP_PASSWORD }}
              EMAIL_RECIPIENT: ${{ secrets.EMAIL_RECIPIENT }}
              PIPELINE_STATUS_PATH: pipeline_status.json
              GARTH_HOME: ~/.garth
            working-directory: src
            run: |
              echo "Debugging Environment Variables:"
              if [ -z "${POSTGRES_CONNECTION_STRING}" ]; then
                echo "POSTGRES_CONNECTION_STRING is EMPTY"
              else
                echo "POSTGRES_CONNECTION_STRING is SET (length: ${#POSTGRES_CONNECTION_STRING})"
              fi
              python weekly_sync.py --days 2

          - name: Show pipeline status
            if: always()
            working-directory: src
            run: |
              if [ -f pipeline_status.json ]; then cat pipeline_status.json
              else echo "pipeline_status.json not generated"; fi

          - name: Upload pipeline status artifact
            if: always()
            uses: actions/upload-artifact@v4
            with:
              name: pipeline-status
              path: src/pipeline_status.json
              if-no-files-found: warn

---

### Bug 3 — ensure_startup_schema() called with no args in _save_insights()

In src/pipeline/weekly_pipeline.py inside _save_insights():

    ensure_startup_schema()   # BUG: missing conn_str

Schema is already ensured in Step 0 of run(). Just DELETE this line.

---

## Part 2: Files to Touch (Priority Order)

| Priority | File                              | Action                                                                 |
|----------|-----------------------------------|------------------------------------------------------------------------|
| RED 1    | src/enhanced_fetcher.py           | Add `import time`                                                      |
| RED 2    | src/enhanced_schema.py            | Add app_config table                                                   |
| RED 3    | src/pipeline/migrations.py        | Add app_config to ensure_startup_schema()                              |
| RED 4    | src/enhanced_fetcher.py           | Add _restore_session_from_db() + _persist_session_to_db(), wire them  |
| RED 5    | .github/workflows/daily_sync.yml  | Remove garth session steps, use clean version above                    |
| RED 6    | src/pipeline/weekly_pipeline.py   | Delete orphan ensure_startup_schema() call in _save_insights()         |
| YELLOW 7 | src/enhanced_fetcher.py           | Refactor _fetch_all_sources() with _safe_fetch() helper (~100 lines)   |
| YELLOW 8 | .github/workflows/daily_sync.yml  | Move pytest out of daily cron — run tests only on push/PR              |
| GREEN 9  | src/enhanced_agents.py            | Extract prompt strings to prompts.py, remove boilerplate               |
| GREEN 10 | root directory                    | Delete PLAN.md, REQUEST_LOG.md, CORRELATION_ENGINE_DEEP_DIVE.md,      |
|          |                                   | garmin_project.code-workspace                                          |
| GREEN 11 | mass_donwload_code_decoder/       | Fix typo in folder name or delete if unused                            |

---

## Part 3: _fetch_all_sources() Refactor Reference

Current pattern (repeated 12 times):

    try:
        data["heart_rate"] = pd.json_normalize(self._api_call("/some/endpoint"))
        log.info(" heart_rate fetched")
    except Exception as e:
        log.info(" heart_rate failed: %s", e)

Add this helper method to the class:

    def _safe_fetch(self, name: str, endpoint: str):
        try:
            result = self._api_call(endpoint)
            log.info("  %s fetched", name)
            return result
        except Exception as e:
            log.warning("  %s failed: %s", name, e)
            return None

Then each source in _fetch_all_sources() becomes one line.

---

## Summary: The 6 Changes That Fix Everything

    1. enhanced_fetcher.py       -> add `import time`
    2. enhanced_schema.py        -> add app_config table
    3. pipeline/migrations.py    -> include app_config in ensure_startup_schema()
    4. enhanced_fetcher.py       -> add _restore_session_from_db() + _persist_session_to_db()
                                    call _restore at START of authenticate()
                                    call _persist after garth.save()
    5. daily_sync.yml            -> remove the two broken garth session steps
    6. weekly_pipeline.py        -> delete orphan ensure_startup_schema() in _save_insights()

Once these 6 changes are in, the sync will be reliable.
Everything after that is cleanup.
