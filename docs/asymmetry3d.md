# Dynamic 3D reconstruction and gait asymmetry

Requirement 4 analysis lives in `stablewalk/analysis/asymmetry3d/`.
It **reuses** MediaPipe canonical 3D (`GaitMotionRecording`), existing
gait-cycle events, and the OpenSim Inverse Dynamics readiness gate.
It does **not** add a second 3D solver or a second skeleton viewer.

## Coordinate provenance

| Label | Meaning |
|-------|---------|
| `inferred_monocular_3d` | Default. Pelvis-centred `sw_canonical_y_up`, body-normalized metres (~1.0 body height). Heuristic depth. |
| `calibrated_metric` | Only when the session is explicitly marked calibrated. Then spatial step length may be reported in metres. |

Playback, the existing 3D skeleton, and OpenSim IK/TRC export are unchanged.

## Gait events

Heel strike, toe-off, left/right cycles, stance, swing, step and stride
timing come from `analyze_gait_cycles` (not a parallel detector). View
classification (`gait_view_analysis`) sets reliability:

- Frontal / unknown cameras: **step length unreliable**
- Unknown + very low view confidence: camera **unsuitable** for events
- Timing can remain usable when spatial measures are not

## Symmetry indices

Documented in `indices.py` and exported with every report:

| Index | Formula | Symmetric value |
|-------|---------|-----------------|
| Robinson SI (signed %) | \(200(L-R)/(L+R)\) | 0 |
| ASI (unsigned %) | \(200\|L-R\|/(\|L\|+\|R\|)\) | 0 |
| SI_norm | \(2\min(L,R)/(L+R)\) for non-negative quantities | 1 |

Left and right gait cycles are resampled to 0–100% (`GAIT_CYCLE_SAMPLE_COUNT` = 101)
**before** waveform comparison. Unaligned frames are never subtracted.

## Joint moments — three kinds

1. **Kinematic asymmetry** — statistical 1st/2nd moments of 3D joint paths (not torques).
2. **Estimated joint moment** — optional \(M = I\alpha\) with Dempster/Winter segment inertia. Requires subject mass (kg). Experimental and model-dependent.
3. **Measured joint moment** — OpenSim Inverse Dynamics with **measured GRF**. Disabled when force plates / ExternalLoads are missing (`opensim_id_readiness.py`). Video vGRF proxies are never treated as ID loads.

## GUI

Motion inner tab **Asymmetry**: L/R table, phase-aligned knee and foot plots,
ROM bars, stance/swing bars, optional I·α plot. The 3D skeleton remains on
Overview / Kinematics.
