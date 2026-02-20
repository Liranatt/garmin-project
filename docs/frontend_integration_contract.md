# Frontend Integration Contract (gps_presentation)

Last verified: 2026-02-20

## Source of Truth

- Frontend repo: `https://github.com/Liranatt/gps_presentation`
- Active app file: `garmin/index.html`
- All live backend calls are made from `garmin/index.html` only.

## Frontend Runtime Assumptions

- API base URL is hardcoded in frontend:
  - `https://garmin-health-liran-f16faec73fc5.herokuapp.com`
- Requests use `fetch` with CORS mode.
- No auth headers are sent (public endpoints).

## Required Endpoints

1. `GET /health-check`
- Purpose: status badge ("Online"/"Waking up"/"Offline").
- Frontend reads one of:
  - `status`
  - `message`
- If response text includes `wake|boot|starting`, UI shows "Waking up".

2. `GET /api/v1/snapshot/latest`
- Purpose: hero KPIs + snapshot cards.
- Frontend accepts:
  - object with `snapshot` key, or
  - object with `data` key, or
  - flat object.
- Fields consumed (any alias below works):
  - resting HR: `resting_hr | restingHeartRate | rhr | resting_hr_avg`
  - HRV: `hrv | hrv_ms | rmssd`
  - sleep: `sleep_score | sleepScore | sleep`
  - body battery: `body_battery | bodyBattery | battery`
  - stress: `stress | stress_score | avg_stress`
  - load: `training_load | load | acute_load | seven_day_load`
  - timestamp: `timestamp | created_at | date | as_of`

3. `GET /api/v1/metrics/history?days=<int>`
- Purpose: trends + correlation matrix (correlations computed on frontend).
- Frontend accepts:
  - array directly, or
  - `{ data: [...] }`, or
  - `{ history: [...] }`.
- Row fields consumed (aliases):
  - date: `date | timestamp | as_of`
  - resting HR: `resting_hr | restingHr | rhr`
  - HRV: `hrv | hrv_last_night | hrv_ms | rmssd`
  - sleep: `sleep_score | sleepScore | sleep`
  - stress: `stress | stress_level | stress_score | avg_stress`
  - battery: `battery | body_battery | bb_peak`
  - load: `training_load | daily_load_acute | load | acute_load`
- Notes:
  - frontend filters out rows before `2026-02-01`.
  - frontend sorts by date ascending.

4. `GET /api/v1/workouts/progress?days=<int>`
- Purpose: workout progress tab (cards + table).
- Frontend accepts:
  - array directly, or
  - `{ data: [...] }`.
- Optional summary object used at:
  - `summary.activity_types` (array)
  - `summary.strength_proxy_trend.note` (string)
- Row fields consumed (aliases):
  - date: `date | timestamp`
  - activity name: `activity_name | activityName`
  - activity type: `activity_type | activityType | sport_type | sportType`
  - duration min: `duration_min | durationMin` or derived from `duration_sec | durationSec`
  - distance km: `distance_km | distanceKm` or derived from `distance_m | distanceM`
  - avg HR: `average_hr | avg_hr | avgHr`
  - speed: `speed_kph | speedKph`
  - cadence: `avg_cadence | avgCadence | cadence`
  - load: `training_load | trainingLoad`

5. `GET /api/v1/insights/latest`
- Purpose: main insight + additional agent cards.
- Frontend accepts:
  - array directly, or
  - `{ insights: [...] }`, or
  - `{ data: [...] }`, or
  - single object with insight/message/text.
- Item fields consumed (aliases):
  - text/summary: `summary | insight | message | text | content`
  - agent/source: `agent | agent_name | source | role`
  - timestamp: `timestamp | created_at | date | time`
  - detail: `detail | text | content | raw`

6. `POST /api/v1/chat`
- Request JSON:
  - `{ "message": "<string>" }`
- Frontend expects response field in priority:
  - `answer | response | message | text | output`
- Any non-2xx returns error text to UI.

## CORS Requirements

Frontend origin is served from GitHub Pages/custom domain.
Backend must allow:

- `Origin: https://liranattar.dev`
- Methods: `GET, POST, OPTIONS` (and others if desired)
- Headers: at minimum `content-type` for chat preflight

Observed live backend already returns correct CORS headers for these checks.

## Current Integration Gap (Observed)

- `GET /api/v1/workouts/progress?days=30` returns `404 Not Found` on live backend.
- This breaks the Workout Progress tab and forces error state in UI.

## Cross-Repo Change Rule

If backend changes any of these, a frontend PR in `gps_presentation` is required:

- Base URL
- Endpoint paths
- Query parameter names
- Response key names not covered by existing aliases
- Auth model (if endpoints become protected)

If backend keeps the same contract, frontend changes are not required.
