# Requirement 4 — dynamic 3D reconstruction and gait asymmetry

**Date:** 2026-10-12  
**Status:** implemented (additive). Existing 3D skeleton, playback, and OpenSim IK were not replaced.

## What was reused

- `GaitMotionRecording` snapshots (MediaPipe → canonical 3D)
- `pose/reconstruction.py` assumptions (documented on every report)
- `analyze_gait_cycles` heel strike / toe-off / stance / swing
- `resample_cycle_trajectory` (0–100% gait phase, 101 samples)
- `gait_view_analysis` camera-view reliability
- `opensim_id_readiness.assess_opensim_id_readiness` (measured moments stay off without GRF)
- Overview / Kinematics 3D skeleton and playback clock

## New modules

| Path | Role |
|------|------|
| `stablewalk/analysis/asymmetry3d/kinematics.py` | Positions, bone orientations, angles, ω, α, bone-length / low-confidence flags |
| `events.py` | Wrap existing gait events + view reliability |
| `indices.py` | Robinson SI, ASI, SI_norm |
| `trajectories.py` | Phase-aligned L/R curves |
| `compare.py` | Bilateral scalars (ROM, ω, timing, clearance, waveform amplitude) |
| `moments.py` | Kinematic vs estimated I·α vs measured ID |
| `report.py` / `export.py` | Orchestration and CSV/JSON |
| `stablewalk/ui/tk/dashboard_asymmetry.py` | Motion **Asymmetry** page (no second 3D viewer) |

## Validation

`tests/test_asymmetry3d.py`, `tests/test_asymmetry3d_moments.py`, `tests/test_asymmetry_workspace.py`

- Synthetic symmetric gait → SI/ASI near 0
- Synthetic asymmetric knee ROM → ASI rises, aligned RMSE rises
- Different walking speeds → phase-aligned waveforms still match; stride times differ
- Missing landmarks → no crash, no invented infinities
- Frontal / unknown views → spatial step length / events gated
- No mass → estimated moments disabled; measured ID never claimed without GRF

## Limitations (thesis-facing)

- Monocular Z is heuristic; body-normalized metres are not laboratory mocap.
- Step length in metres is disabled unless the session is calibrated **and** the view supports spatial gait.
- \(M=I\alpha\) ignores GRF, two-joint coupling, and soft tissue; it is an experimental proxy.
- OpenSim Inverse Dynamics is **not executed**. Readiness is reported only.
- Indices are not clinical diagnoses.
