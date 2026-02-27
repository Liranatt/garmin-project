# Garmin Pipeline — Action Plan

> Last updated: 2026-02-27

## ✅ Done

| Fix | Files Changed |
|-----|---------------|
| Correlation engine: `Timestamp vs date` TypeError | `correlation_engine.py` |
| Interpreter agent: added SQL tools fallback | `enhanced_agents.py` |
| Export reminder: fixed broken import | `send_export_reminder.py` |
| Removed hydration fetch + processing | `enhanced_fetcher.py` |
| Removed nutrition_log from correlation engine | `correlation_engine.py` |
| Removed hydration from agent DB hints | `enhanced_agents.py` |
| Added `correlation_results` table + wired storage | `enhanced_schema.py`, `correlation_engine.py` |
| Added `matrix_summaries` to central schema | `enhanced_schema.py` |
| Pipeline status filename now includes date | `pipeline/weekly_pipeline.py` |
| API rate limiting (slowapi, 5/min chat) | `api.py`, `requirements.txt` |
| Dead code archived | `dashboard.py`, `visualizations.py` → `archive/` |
| Schema dedup: removed DDL from pipeline+agents | `weekly_pipeline.py`, `enhanced_agents.py` |
| Split `api.py` (1134→480L) | `api.py` + `routes/helpers.py` (430L) |
| Test suites: email (17), summary (9), pipeline (20+) | `tests/` |

---

## � Next Up

| # | Task | Effort |
|---|------|--------|
| 1 | **Push & verify** — deploy, check next weekly pipeline run | 10min |
| 2 | **Test weekly email** — trigger manually with real secrets | 30min |
| 3 | **Trace bulk import failure** — add verbose logging | 3h |

## 🟢 Future Refactoring

| # | File | Lines | Notes |
|---|------|-------|-------|
| 4 | `enhanced_agents.py` | 1060 | Split if agent count grows |
| 5 | `correlation_engine.py` | 1598 | layers already factored out (markov_layer.py) |
| 6 | `enhanced_fetcher.py` | 945 | Stable, low churn |

---

> **Deep dives**: `CORRELATION_ENGINE_DEEP_DIVE.md` has full math + architecture
