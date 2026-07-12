# StableWalk coordinate systems

This document is generated from `stablewalk.coordinates.coordinate_map.FRAME_REGISTRY`
and the conversion functions in `stablewalk/coordinates/coordinate_map.py`.

## Canonical convention (SW_CANONICAL)

All gait metrics, 3D skeleton reconstruction, foot clearance, contact detection,
joint path graphs, and `GaitMotionRecording` use this frame unless noted.

| Property | Value |
|----------|-------|
| **Frame ID** | `sw_canonical_y_up` |
| **Origin** | Pelvis (`mid_hip` / `pelvis`) at (0, 0, 0) |
| **+X** | Mediolateral — subject's right |
| **+Y** | Vertical — up |
| **+Z** | Forward — anterior / toward camera |
| **Units** | Body-normalized meters (~1.0 body height) |
| **Handedness** | Right-handed |

**Transform from MediaPipe image landmarks:**

```
x' = (x - cx) * scale
y' = -(y - cy) * scale     # image Y-down → canonical Y-up
z' = (depth_layer - mp.z * 0.2) * scale
scale → TARGET_SKELETON_HEIGHT
```

---

## MediaPipe image landmarks (`mediapipe_normalized_image`)

| Property | Value |
|----------|-------|
| **Source** | `pose_landmarks` from MediaPipe Pose Landmarker |
| **Origin** | Image top-left |
| **+X** | Right (0–1 normalized width) |
| **+Y** | Down (0–1 normalized height) |
| **+Z** | Toward camera (relative depth) |
| **Units** | Normalized |
| **Used by** | Raw keypoints, 2D kinematics, **OpenSim TRC export input** |

---

## MediaPipe world landmarks (`mediapipe_world_landmarks_m`)

| Property | Value |
|----------|-------|
| **Source** | `pose_world_landmarks` (optional, meters) |
| **+Y** | Up in MediaPipe world space |
| **Used by** | Not the default StableWalk pipeline; documented for future lab export |

---

## Canonical StableWalk joints (`GaitMotionRecording`)

| Property | Value |
|----------|-------|
| **Source** | `pose_adapter.pose_sequence_to_gait_motion` |
| **Frame** | SW_CANONICAL (same as 3D skeleton) |
| **Joint IDs** | `pelvis`, `left_knee`, `left_ankle`, … (see `joint_registry`) |

Positions come from `positions_normalized` in pose JSON when present, else
`skeleton_3d.reconstruct_skeleton_3d`.

---

## 3D skeleton visualization

| Property | Value |
|----------|-------|
| **Data frame** | SW_CANONICAL |
| **Display projection** | Oblique: `(x + 0.22*z, y)` via `canonical_to_visualization_oblique` |
| **Vertical for clearance overlay** | +Y (same as contact detector) |

---

## Foot clearance & gait contact

| Property | Value |
|----------|-------|
| **Vertical axis** | +Y (`CANONICAL_VERTICAL_AXIS`) |
| **Floor reference** | `floor_y` = robust low percentile of ankle heights |
| **Clearance** | `point_Y - floor_y` |
| **Module** | `stablewalk.analysis.ground_reference` |

---

## Joint movement 3D path graph

| Property | Value |
|----------|-------|
| **Data frame** | SW_CANONICAL joint positions from `GaitMotionRecording` |
| **Axis labels** | X — Mediolateral, Y — Vertical, Z — Forward |

---

## OpenSim TRC export (`opensim_trc_export_mm`)

| Property | Value |
|----------|-------|
| **Source** | Raw MediaPipe **image** landmarks (not hip-centered) |
| **Origin** | Image corner (scaled), **not** pelvis-centered |
| **+X** | Image horizontal × scale |
| **+Y** | Up: `(1 - y) * scale` |
| **+Z** | `-z * scale` |
| **Units** | Millimeters in `.trc` file |
| **Function** | `mediapipe_to_opensim_trc_mm` |

Mapped anatomical markers (`marker_reconstruction.py`) assume this TRC frame.

---

## OpenSim IK / MOT coordinates

| Property | Value |
|----------|-------|
| **Source** | OpenSim Inverse Kinematics solver |
| **Content** | Generalized coordinates (joint angles), not positions |
| **Units** | Degrees (`inDegrees=yes`) |

---

## Conversion API

```python
from stablewalk.coordinates import (
    mediapipe_to_canonical,
    mediapipe_to_opensim_trc_mm,
    canonical_to_visualization_oblique,
    check_anatomical_ordering,
    audit_recording_anatomy,
    debug_canonical_joint_positions,
)
```

## Debug command

```bash
python scripts/debug_coordinates.py abnormal_gait --frames 0 50 100
```

## Automated checks

On pose → gait conversion, `audit_recording_anatomy` samples frames and warns if:

- Head is not above pelvis on +Y
- Knees are not below pelvis
- Ankles are not below knees
- Feet float far above the estimated floor

Warnings are stored in `GaitMotionRecording.metadata["coordinate_warnings"]`.
