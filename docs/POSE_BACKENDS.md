# Pose / Human Mesh Recovery backend architecture

StableWalk supports **multiple video-to-3D reconstruction backends** through a
modular adapter layer. The default remains **MediaPipe** — no PyTorch or GPU
research dependencies are required for the main application or OpenSim pipeline.

## Design goals

1. **MediaPipe stays default** — existing gait analysis, GUI, and OpenSim export unchanged.
2. **Optional HMR backends** — ROMP, HybrIK, WHAM are experimental adapters only.
3. **Canonical skeleton** — all backends map to the same joint ids (`pelvis`, `left_hip`, …).
4. **Safe fallback** — optional backends fail loudly; fallback to MediaPipe only when allowed.
5. **Research comparison** — structured metrics, no automatic “winner”.

## Architecture

```
Video
  │
  ▼
HumanMotionBackend  (mediapipe | romp | hybrik | wham)
  │
  ▼
HumanMotionSequence / HumanMotionFrame
  │  joint_positions_3d → canonical ids
  │  landmark_confidence, coordinate_system_metadata
  │
  ├─► human_motion_sequence_to_pose_sequence()  → legacy PoseSequence
  ├─► human_motion_sequence_to_gait_motion()  → GaitMotionRecording
  └─► OpenSim marker reconstruction (via canonical_to_trc_landmarks)
```

### Key modules

| Module | Purpose |
|--------|---------|
| `stablewalk/pose/backends/base.py` | `HumanMotionBackend` abstract interface |
| `stablewalk/pose/backends/types.py` | `HumanMotionFrame`, `HumanMotionSequence` |
| `stablewalk/pose/backends/mediapipe_backend.py` | Default production backend |
| `stablewalk/pose/backends/romp_backend.py` | ROMP/SMPL placeholder |
| `stablewalk/pose/backends/hybrik_backend.py` | HybrIK placeholder |
| `stablewalk/pose/backends/wham_backend.py` | WHAM placeholder |
| `stablewalk/pose/backends/registry.py` | Factory + availability |
| `stablewalk/pose/backends/canonical.py` | Landmark → canonical conversion |
| `stablewalk/pose/backends/comparison.py` | Research comparison metrics |
| `stablewalk/pose/backends/environment.py` | Runtime inspection (no installs) |

## Configuration

Environment variables (also documented in `stablewalk/config.py`):

```bash
# Default — do not change for production OpenSim workflow
set POSE_BACKEND=mediapipe

# Future experimental values (require separate env + dependencies)
set POSE_BACKEND=romp
set POSE_BACKEND=hybrik
set POSE_BACKEND=wham

# When an optional backend is unavailable:
set POSE_BACKEND_ALLOW_FALLBACK=true   # log + fall back to MediaPipe
set POSE_BACKEND_ALLOW_FALLBACK=false  # raise BackendUnavailableError
```

**Important:** StableWalk does **not** silently switch backends. If `POSE_BACKEND=wham`
and WHAM is missing, you see:

```
Backend unavailable: WHAM — WHAM dependencies are not installed
```

Fallback occurs only when `POSE_BACKEND_ALLOW_FALLBACK=true`.

## Canonical joints

All backends must populate these ids where possible:

`pelvis`, `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`,
`right_ankle`, `left_heel`, `right_heel`, `left_toe`, `right_toe`, `spine`,
`neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`,
`left_wrist`, `right_wrist`

Defined in `models/joint_registry.py` and `pose/backends/canonical.py`.

## Recommended environments

### `stablewalk-opensim` (main app)

- Python 3.10+
- MediaPipe Tasks API
- OpenSim SDK (optional, for IK)
- **No PyTorch required**

### `stablewalk-hmr` (research only)

Use a **separate conda environment** for HMR experiments. Do not install ROMP,
HybrIK, or WHAM into the OpenSim conda stack — PyTorch/CUDA pins may conflict.

Before installing, inspect the current machine:

```python
from stablewalk.pose.backends import inspect_runtime_environment
print(inspect_runtime_environment().to_dict())
```

Official references (install in `stablewalk-hmr` only):

- ROMP: https://github.com/Arthur151/ROMP
- HybrIK: https://github.com/Jeff-sjtu/HybrIK
- WHAM: https://github.com/yohanshin/WHAM

StableWalk provides **adapter placeholders** only — full inference wiring is
future research work.

## Comparison script

```bash
python scripts/compare_pose_backends.py --video data/videos/my_walk.mp4
python scripts/compare_pose_backends.py --video walk.mp4 --max-frames 120 --json out.json
```

Metrics (informational):

- Valid frame percentage
- Landmark trajectory smoothness
- Joint-angle consistency
- Foot-ground jitter
- Pelvis trajectory consistency
- OpenSim marker reconstruction confidence / IK readiness score

The script runs even when only MediaPipe is installed and lists unavailable backends.

## Programmatic usage

```python
from stablewalk.pose.backends import create_pose_backend
from stablewalk.pose.backends.canonical import human_motion_sequence_to_gait_motion

backend = create_pose_backend("mediapipe", video_mode=True)
sequence = backend.process_video("walk.mp4")
recording = human_motion_sequence_to_gait_motion(sequence)
```

## OpenSim pipeline

The OpenSim export path is **unchanged**. It still consumes MediaPipe-exported TRC
landmarks and anatomical marker reconstruction. Backend comparison uses canonical
joints only for **research scoring**, not for production IK unless explicitly wired.

## Inspiration disclaimer

This extension is inspired by monocular Human Mesh Recovery approaches (SMPL,
ROMP, HybrIK, WHAM). StableWalk does **not** ship those models or claim integration
until adapters are fully implemented and validated.
