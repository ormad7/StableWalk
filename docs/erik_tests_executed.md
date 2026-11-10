# Automated tests executed — requirements 1–4

**Date:** 2026-10-13  
**Environment:** Windows, project `StableWalk`, `python -m pytest`

## Command A

```
python -m pytest tests/test_wire_model_inventory.py tests/test_spectral_analysis.py tests/test_spectral_features.py tests/test_spectral_workspace.py tests/test_asymmetry3d.py tests/test_asymmetry3d_moments.py tests/test_asymmetry_workspace.py tests/test_scientific_gait_validation.py tests/test_opensim_id_readiness.py tests/test_opensim_ik_status.py tests/test_overview_clinical_summary.py tests/test_gait_event_sufficiency.py tests/test_real_video_gait_validation.py -q --tb=short
```

**Result: 98 passed in 22.71 s**

Coverage of that command:

- Wire joint vs DOF inventory
- ACF, Welch/FFT, R(0)/R(τ), eigh refusal on lagged matrices, L/R xcorr
- Spectral features, FPS invariance, missing landmarks, reproducibility, CSV/JSON
- GUI channel mapping (no live window required)
- Asymmetry SI/ASI, speed-normalized phase, frontal view gate, moment kinds
- COM–BoS unreportable, cadence from HS cycles, synthetic cadence GT
- OpenSim ID readiness (not ready without GRF); IK status honesty
- Overview N/A / confidence vocabulary
- Event sufficiency; real-video file checks (pose skipped where JSON missing)

## Command B

```
python -m pytest tests/test_scientific_interpretation.py tests/test_motion_analysis_tabs.py tests/test_opensim_ik_status.py -q --tb=line
```

**Result: 18 passed in 2.00 s**

## Not executed

- Full Tk GUI click-through
- OpenSim ScaleTool / live IK on a subject
- Force-plate Inverse Dynamics
- Requirement 5 simulator
- Screenshot capture

No test was edited to force `normal_gait.mp4` to a high stability score.
