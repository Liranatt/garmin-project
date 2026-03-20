# Garmin Project — Phase 2: Cleanup Plan
# (Critical bugs from Phase 1 are DONE)
# Date: 2026-03-20

---

## Status

DONE:
  1. import time added to enhanced_fetcher.py
  2. app_config table added to schema + migrations
  3. _restore_session_from_db() + _persist_session_to_db() added and wired in authenticate()
  4. Broken garth session steps removed from daily_sync.yml
  5. Orphan ensure_startup_schema() deleted from _save_insights()

NOW: Cosmetic cleanup — reduce noise, make the codebase readable.

---

## File 1: src/enhanced_fetcher.py  (941 lines -> ~650)

### Problem A: _fetch_all_sources() has 12 identical try/except blocks

Current code (repeated 12 times, ~8 lines each = ~96 lines of noise):

    try:
        hr = self._api_call(f"/usersummary-service/stats/heartRate/daily/{start}/{end}")
        data["heart_rate"] = pd.json_normalize(hr)
        log.info(f" heart_rate: {len(data['heart_rate'])} days")
    except Exception as e:
        log.info(f" heart_rate failed: {e}")

    # ... 11 more of these exact same blocks

Fix — add this helper method to the class (place it right after _api_call):

    def _safe_fetch(self, name: str, endpoint: str) -> Optional[Any]:
        """Call a Garmin API endpoint, returning None on any error."""
        try:
            result = self._api_call(endpoint)
            log.info("  fetched: %s", name)
            return result
        except Exception as e:
            log.warning("  failed: %s — %s", name, e)
            return None

Then replace the 12 repeated blocks in _fetch_all_sources() with clean one-liners:

    # BEFORE (8 lines per source):
    try:
        hr = self._api_call(f"/usersummary-service/stats/heartRate/daily/{start}/{end}")
        data["heart_rate"] = pd.json_normalize(hr)
        log.info(f" heart_rate fetched")
    except Exception as e:
        log.info(f" heart_rate failed: {e}")

    # AFTER (3 lines):
    raw = self._safe_fetch("heart_rate", f"/usersummary-service/stats/heartRate/daily/{start}/{end}")
    if raw is not None:
        data["heart_rate"] = pd.json_normalize(raw)

Apply this to every endpoint EXCEPT sleep.
Sleep has its own per-day loop with custom dict building — leave that block alone.

Endpoints to convert (12 total):
  heart_rate, body_battery, steps, training_readiness,
  hrv, stress, intensity, weight, activities
  (sleep loop stays as-is)

---

### Problem B: deep_vars() — check if it can be deleted

deep_vars() is a ~30-line recursive introspection function used to extract
fields from garth Pydantic objects (DailyIntensityMinutes, WeightData).

Step 1 — check usages in PyCharm:
  Right-click deep_vars -> Find Usages
  You should see it only in the intensity and weight sections of _fetch_all_sources()

Step 2 — test in a Python console (after authenticating garth):

    import garth
    items = garth.DailyIntensityMinutes.list("2026-03-13", 7)
    print(hasattr(items[0], 'model_dump'))   # True = Pydantic v2
    print(hasattr(items[0], 'dict'))          # True = Pydantic v1

If model_dump() exists, replace:

    rows = [deep_vars(obj) for obj in im_items]

with:

    rows = [obj.model_dump() for obj in im_items]

Then delete the deep_vars() function entirely.

If neither method exists, leave deep_vars() in place — it is doing real work.

---

## File 2: src/enhanced_agents.py  (1257 lines -> ~350 logic + prompts.py for text)

### What is actually in this file

  ~100 lines  imports, LLM singleton, 5 @tool functions     GOOD — keep as-is
  ~200 lines  AdvancedHealthAgents class, agent definitions  GOOD — keep as-is
  ~800 lines  prompt strings hardcoded inline:               THIS IS THE NOISE
                - analysis_rules  (~350 lines)
                - physio_rules    (~150 lines)
                - db_hint         (~100 lines)
                - 5 agent backstory strings (~30 lines each)
                - 5 task description strings (~30 lines each)

The agents are well designed. The noise is all the text living inline in the code.

### Fix — create src/prompts.py and move all strings there

Create new file: src/prompts.py

Cut and paste these out of enhanced_agents.py into prompts.py:

    # src/prompts.py

    ANALYSIS_RULES = """
    ====== ANALYTICAL RULES (MUST FOLLOW) ======
    1. SAMPLE SIZE: Any correlation or prediction with n<20 is PRELIMINARY...
    [paste the full analysis_rules string here]
    ==============================================
    """

    PHYSIO_RULES = """
    ====== PHYSIOLOGICAL INVESTIGATION GUIDE ======
    [paste the full physio_rules string here]
    ==============================================
    """

    DB_HINT = """
    DATABASE COLUMN REFERENCE -- daily_metrics table
    [paste the full db_hint string here]
    """

    STATISTICAL_INTERPRETER_BACKSTORY = """
    You are a statistics interpreter and longitudinal analysis expert...
    [paste the full backstory from self.statistical_interpreter]
    """

    HEALTH_PATTERN_ANALYST_BACKSTORY = """
    [paste the full backstory from self.health_pattern_analyst]
    """

    PERFORMANCE_RECOVERY_BACKSTORY = """
    [paste the full backstory from self.performance_recovery]
    """

    SLEEP_LIFESTYLE_BACKSTORY = """
    [paste the full backstory from self.sleep_lifestyle]
    """

    SYNTHESIZER_BACKSTORY = """
    [paste the full backstory from self.synthesizer]
    """

Then in enhanced_agents.py, replace the import block at top with:

    try:
        from .prompts import (
            ANALYSIS_RULES, PHYSIO_RULES, DB_HINT,
            STATISTICAL_INTERPRETER_BACKSTORY,
            HEALTH_PATTERN_ANALYST_BACKSTORY,
            PERFORMANCE_RECOVERY_BACKSTORY,
            SLEEP_LIFESTYLE_BACKSTORY,
            SYNTHESIZER_BACKSTORY,
        )
    except ImportError:
        from prompts import (
            ANALYSIS_RULES, PHYSIO_RULES, DB_HINT,
            STATISTICAL_INTERPRETER_BACKSTORY,
            HEALTH_PATTERN_ANALYST_BACKSTORY,
            PERFORMANCE_RECOVERY_BACKSTORY,
            SLEEP_LIFESTYLE_BACKSTORY,
            SYNTHESIZER_BACKSTORY,
        )

And each agent definition becomes clean:

    self.statistical_interpreter = Agent(
        role='Statistical Interpreter',
        goal='Interpret EVERY section of the pre-computed correlation data...',
        backstory=STATISTICAL_INTERPRETER_BACKSTORY,
        verbose=True,
        allow_delegation=False,
        tools=self.tools,
        llm=_get_llm()
    )

And in create_weekly_summary_tasks(), instead of building the massive ctx string inline:

    # BEFORE:
    analysis_rules = ("\n\n====== ANALYTICAL RULES..." ... 350 lines ...)
    physio_rules = ("\n\n====== PHYSIOLOGICAL..." ... 150 lines ...)
    db_hint = ("\n\nDATABASE COLUMN REFERENCE..." ... 100 lines ...)
    ctx = f"{corr_block}{analysis_rules}{physio_rules}{db_hint}"

    # AFTER:
    ctx = f"{corr_block}{ANALYSIS_RULES}{PHYSIO_RULES}{DB_HINT}"

Zero functional change. File goes from 1257 -> ~350 lines.
prompts.py holds the ~800 lines of text, isolated and easy to edit.

---

## File 3: Root directory — delete junk files

Delete these (right-click -> Delete in PyCharm, then git commit):

  PLAN.md                                                      Claude's planning notes
  REQUEST_LOG.md                                               Claude's request log
  CORRELATION_ENGINE_DEEP_DIVE.md                              Duplicates code comments
  why_the_math_mathing_and_why_agents_arent_hallucinating.md   Same
  garmin_project.code-workspace                                VSCode file, you use PyCharm

Keep: README.md (it is accurate and useful)

---

## File 4: mass_donwload_code_decoder/ — rename or delete

Folder name has a typo: donwload should be download.

Before touching it:
  In PyCharm: right-click the folder -> Find Usages
  Check if any workflow .yml file references it by path

If nothing uses it: delete the whole folder
If something uses it: rename to mass_download_code_decoder
  PyCharm will offer to update all references automatically

---

## File 5: .github/workflows/daily_sync.yml — remove pytest from cron

Currently the daily cron job runs pytest -q every morning.
That means you are running your full test suite every single day
just to fetch Garmin data. Wasted CI minutes.

Remove this step from daily_sync.yml:

    # DELETE THIS from daily_sync.yml:
    - name: Run tests
      run: pytest -q

Tests should only run on push and pull_request, not on schedule.
If you want to keep tests in CI, create a separate workflow:

    # .github/workflows/tests.yml
    name: Tests
    on:
      push:
        branches: [main]
      pull_request:
    jobs:
      test:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with:
              python-version: '3.12'
              cache: 'pip'
          - run: pip install -r requirements.txt
          - run: pytest -q

---

## Priority Order for This Phase

  1. enhanced_fetcher.py    add _safe_fetch(), refactor _fetch_all_sources()
                            check deep_vars() — delete if .model_dump() works
  2. enhanced_agents.py     create prompts.py, move all string literals there
  3. daily_sync.yml         remove pytest step, create separate tests.yml
  4. Root directory         delete the 5 junk files
  5. mass_donwload folder   check usages, rename or delete

---

## What NOT to touch yet

  src/correlation_engine.py   1722 lines   Math is correct, not causing failures
  src/routes/helpers.py        526 lines   Not causing failures
  src/api.py                   759 lines   Working fine
  tests/                       109 tests   Leave alone
