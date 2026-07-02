# Session Storage (Save Session)

Persistent SQLite storage for biomechanical and kinematic data collected during
StableWalk video playback.

## Overview

When you select degrees of freedom (DOFs) and play a video, the dashboard records
frame-by-frame samples (coordinates, joint angles, velocities, and derived deltas).
The **Save Session** button writes those samples to a local SQLite database without
blocking the UI.

## Usage

1. Load and analyze a video (not the presentation demo).
2. Select one or more DOFs (e.g. right knee).
3. Play the video with **Tracking History** view mode (default).
4. Click **Save Session** in the Selected DOF Positions panel, or use
   **File → Save Session to database…**.

If no playback samples exist yet, Save Session falls back to exporting all frames
from the loaded recording for the currently selected DOFs.

Saved sessions are stored at:

```
data/output/sessions/stablewalk_sessions.db
```

## Database schema

### `sessions`

| Column         | Type    | Description                          |
|----------------|---------|--------------------------------------|
| session_id     | TEXT PK | UUID for the saved session           |
| video_source   | TEXT    | Video filename or URL                |
| created_at     | TEXT    | UTC ISO-8601 timestamp               |
| fps            | REAL    | Video frame rate                     |
| sample_count   | INTEGER | Number of kinematic rows saved       |
| selected_dofs  | TEXT    | Comma-separated GUI DOF item IDs     |
| notes          | TEXT    | Optional notes (reserved)            |

### `kinematic_samples`

| Column           | Type    | Description                              |
|------------------|---------|------------------------------------------|
| id               | INTEGER | Auto-increment primary key               |
| session_id       | TEXT FK | Parent session                           |
| frame_number     | INTEGER | 1-based frame index (matches UI table)   |
| time_s           | REAL    | Time position in seconds                 |
| dof_name         | TEXT    | Display label for the DOF                |
| joint_name       | TEXT    | Anatomical joint name                    |
| x, y, z          | REAL    | Joint position (meters)                  |
| angle_deg        | REAL    | Joint / DOF angle (degrees)            |
| velocity         | REAL    | Linear speed (m/s) when available        |
| velocity_deg_s   | REAL    | Angular rate (deg/s) when available      |
| next_angle_deg   | REAL    | Angle on the next frame                  |
| delta_angle_deg  | REAL    | Frame-to-frame angle change              |

Indexes: `(session_id)`, `(session_id, frame_number)`.

Schema version is tracked in `schema_version` (currently **v1**).

## Architecture

```
Playback tick (_append_dof_table_history_tick)
    ├── DofPositionTableHistory  → UI table (last 200 rows)
    └── SessionKinematicCollector → unlimited structured samples

Save Session (background thread)
    └── SessionStorageService
            └── SessionRepository → SQLite
```

### New modules

| File | Role |
|------|------|
| `stablewalk/storage/models.py` | `AnalysisSession`, `KinematicSample` dataclasses |
| `stablewalk/storage/schema.py` | DB creation and migrations |
| `stablewalk/storage/repository.py` | SQLite read/write |
| `stablewalk/storage/service.py` | Application service API |
| `stablewalk/storage/collector.py` | In-memory playback sample buffer |

### Modified modules

| File | Change |
|------|--------|
| `stablewalk/config.py` | `SESSIONS_DB_PATH`, output dir bootstrap |
| `stablewalk/ui/dof_position_table.py` | `kinematic_sample_for_item()` |
| `stablewalk/ui/tk/dashboard_layout.py` | **Save Session** button |
| `stablewalk/ui/tk/app.py` | Collector wiring, async save handler, menu item |

## Error handling and logging

- Validation errors (empty source, no samples) show an info dialog before any I/O.
- Database failures are logged with `logger.exception` and surfaced via an error dialog.
- Save runs on a daemon background thread; the button is disabled until completion.

## Querying saved data

Example SQLite query:

```sql
SELECT s.session_id, s.video_source, s.created_at, k.frame_number,
       k.time_s, k.dof_name, k.angle_deg, k.x, k.y, k.z
FROM sessions s
JOIN kinematic_samples k ON k.session_id = s.session_id
ORDER BY s.created_at DESC, k.frame_number
LIMIT 20;
```

Programmatic access:

```python
from stablewalk.storage import SessionStorageService

service = SessionStorageService()
for session in service.list_recent_sessions(limit=10):
    print(session.session_id, session.video_source, session.sample_count)
```
