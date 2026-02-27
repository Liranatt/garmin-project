"""
API Routes Package
==================
Split from the monolithic api.py for maintainability.
The original api.py re-exports the FastAPI `app` for backward compatibility.

Modules:
  helpers  - DB utilities, formatting, text processing
  data     - /snapshot, /metrics, /workouts endpoints
  analytics- /analytics/cross-effects endpoint
  insights - /insights/latest endpoint
  chat     - /chat endpoint (with rate limiting)
"""
