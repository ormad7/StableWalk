# Research requirements compliance — Erik 1–4

**Date:** 2026-10-13  
**Project:** StableWalk M.Sc. (Computer Science)  
**Rule:** a GUI tab is not proof of a method. Status below is from source
audit plus **executed pytest**. Requirement 5 (RL simulator) is out of
scope and is **not implemented**.

Executed this pass:

```
tests/test_wire_model_inventory.py
tests/test_spectral_analysis.py
tests/test_spectral_features.py
tests/test_spectral_workspace.py
tests/test_asymmetry3d.py
tests/test_asymmetry3d_moments.py
tests/test_asymmetry_workspace.py
tests/test_scientific_gait_validation.py
tests/test_opensim_id_readiness.py
tests/test_opensim_ik_status.py
tests/test_overview_clinical_summary.py
tests/test_gait_event_sufficiency.py
tests/test_real_video_gait_validation.py
→ 98 passed

tests/test_scientific_interpretation.py
tests/test_motion_analysis_tabs.py
tests/test_opensim_ik_status.py
→ 18 passed
```

Interactive Tk playback, live OpenSim ScaleTool, and force-plate Inverse
Dynamics were **not** run in this session.

---

## Biomechanical model (requirement 1)

Joints and degrees of freedom are **not** interchangeable.

| Quantity | Count | Source |
|----------|------:|--------|
| Trackable joints | 21 | `JOINT_IDS` in `stablewalk/models/joint_registry.py` |
| Root (not in `JOINT_IDS`) | 1 pelvis | `ROOT_JOINT_ID` |
| Sagittal wire DOFs | 13 | `DOF_IDS` |
| Exported pose angles | 15 | `GAIT_ANGLE_FIELDS` in `stablewalk/pose/dof.py` |

The 21 joints are: head, neck, spine, L/R shoulder, elbow, wrist, hip,
knee, ankle, heel, toe, foot. Supervisor asked 12–20; this existing model
keeps extra foot landmarks because contact and clearance need them. It was
not reduced to fit the range.

The 13 DOFs are one sagittal angle each (neck, spine, L/R shoulder, elbow,
hip, knee, ankle). Extra exported fields `torso_tilt` and `pelvis_rotation`
are pose-derived, not extra musculoskeletal coordinates.

This is a **monocular kinematic wire figure**, not OpenSim Gait2392 (which
has on the order of 23 body segments and many more coordinates). MediaPipe
landmarks drive the figure. Depth Z is heuristic.

Test: `tests/test_wire_model_inventory.py`.

---

## Compliance matrix

| Requirement | Source files | Actual implementation | Tests | Missing / gated | Validation status |
|-------------|--------------|----------------------|-------|-----------------|-------------------|
| 1 Wire gait model (12–20 joints) | `models/joint_registry.py`, `pose/reconstruction.py`, `pose/skeleton_3d.py`, `pose/dof.py` | 21 joints + pelvis; 13 sagittal DOFs; monocular 3D wire | `test_wire_model_inventory.py` | Not a scaled subject-specific OpenSim model | **Implemented** (existing; not rebuilt) |
| 2 Normalized ACF | `analysis/spectral/autocorrelation.py` | ρ(k) via `np.correlate` after mean removal; stride/step lag windows | `test_spectral_analysis.py` | — | **Implemented**, synthetic GT |
| 2 Time-lag / cross-correlation | `analysis/spectral/matrices.py` | Lagged Pearson ρ_LR(τ); peak lag | `test_left_right_cross_corr_peaks_at_half_period` | — | **Implemented** |
| 2 Multivariate R(0), R(τ) | `matrices.py` | Pairwise-complete covariance and correlation; R(τ) not assumed symmetric | `test_r0_is_symmetric_and_eigh_matches_numpy`, `test_lagged_matrix_is_not_fed_to_eigh` | — | **Implemented** |
| 2 Eigenanalysis | `analysis/spectral/eigen.py` | `eigh` on symmetric R(0) and optional Hermitian band CSD | same + eigen errors | Lagged R(τ) **refuses** symmetric eig | **Implemented** |
| 2 FFT / Welch PSD | `analysis/spectral/spectrum.py` | rFFT periodogram; numpy Welch; SciPy Welch if installed; Hann window | `test_sine_spectrum_dominant_frequency`, `test_numpy_welch_matches_scipy_when_available` | — | **Implemented** |
| 2 Sampling rate / missing values / confidence | `analysis/spectral/series.py`, `config.py` | FPS grid; NaN not zero-fill; interp ≤ `max_gap_s`; visibility mask; `mean_visibility` exported | `test_missing_joint_is_nan_not_zero_fill`, `test_sampling_rate_does_not_shift_known_frequency` | Visibility is a hard mask, not a weighted covariance | **Implemented** |
| 2 GUI + export | `ui/tk/dashboard_spectral.py`, `ui/viewers/spectral_charts.py`, `analysis/spectral/export.py` | Motion → Spectral Analysis; async `analyze_spectral(recording)`; CSV/JSON | `test_spectral_workspace.py`, export tests | Live GUI not re-driven this pass | **Implemented** (code + unit) |
| 3 Spectral features | `analysis/spectral/features.py` | f0, entropy, centroid, bandwidth, harmonics, ACF peak, stride CV, L/R phase, R(0) eigenvalues | `test_spectral_features.py` | Clinical meaning **not assigned** | **Implemented** |
| 3 Feature vectors / export | `features.py`, `export.py`, `dashboard_spectral_features.py` | `SESSION_VECTOR_KEYS`; JSON schema `stablewalk-spectral-features`; CSV section | `test_feature_export_includes_definitions`, `test_repeated_analysis_is_reproducible` | — | **Implemented** |
| 4 Dynamic 3D reconstruction | `pose/reconstruction.py`, `models/gait_motion.py`, `analysis/angle_filter.py` | Video → MediaPipe → hip-centred Y-up wire; despike isolated jumps | `test_scientific_gait_validation.py` despike; reconstruction notes in asymmetry tests | Heuristic Z; not mocap / not SMPL | **Implemented** (monocular) |
| 4 Angular kinematics | `analysis/asymmetry3d/kinematics.py` | Positions, flexion, ω, α | `test_asymmetry3d.py` | Ankle convention is interior angle, not clean dorsi/plantar | **Implemented** |
| 4 OpenSim retarget / IK | `opensim_ik_status.py`, TRC/MOT export, IK tool | Export operational; IK **Complete** only if `ik_state == Completed` | `test_opensim_ik_status.py` | Demo Gait2392, typically **Partial** without ScaleTool | **Operational, gated** |
| 4 L/R asymmetry | `analysis/asymmetry3d/` | ROM, timing, stance/swing, clearance, phase-aligned waveforms; Robinson SI / ASI | `test_asymmetry3d.py` | Spatial metres need calibration + suitable view | **Implemented** |
| 4 Asymmetric **joint moments** | `asymmetry3d/moments.py`, `opensim_id_readiness.py` | Three kinds: (1) statistical trajectory moments, (2) optional M≈Iα if mass given, (3) measured ID | `test_asymmetry3d_moments.py`, `test_opensim_id_readiness.py` | **Measured ID unavailable** without force plates | **Honest: ID off** |
| 8 Reporting honesty | `ui/overview_clinical_summary.py`, `stability_margin.py` (`reportable=False`), gait quality omits unreportable BoS | Stability KPI N/A for monocular COM–BoS; pose vs interpretation confidence split | `test_overview_clinical_summary.py`, `test_scientific_gait_validation.py` | Composite Gait Quality still mixes movement domains (labelled as composite) | **Fixed for COM–BoS / cadence withhold** |
| 5 RL / Isaac simulator | `docs/erik_requirement_5_plan.md`, stub `isaac_lab_integration.py` | Plan only | — | Entire requirement | **Not implemented** — do not demonstrate as done |

---

## What the GUI actually computes

Motion inner notebook:

- **Kinematics** — existing 3D skeleton + joint charts from `GaitMotionRecording`.
- **Spectral Analysis** — calls `analyze_spectral` on the **loaded video’s**
  recording (worker thread). Plots: signal, ACF, PSD, L/R, R(0) heatmap,
  eigenvalues.
- **Spectral Features** — `extract_spectral_features` from that same report.
- **Asymmetry** — `analyze_asymmetry3d` on the same recording and detected
  cycles. No second 3D viewer.

Overview 3D is the same reconstruction used for analysis. Compare Research
runs independent pose pipelines for Video A and B.

---

## Real-video honesty

A filename such as `normal_gait.mp4` is **not** clinical ground truth.
Monocular COM vs base of support is **experimental and unreportable** as a
session stability score. Cadence is reported only from detected heel-strike
cycles. OpenSim “Complete · IK Partial” means files were exported, not that
Inverse Kinematics validated the gait.

See also `docs/scientific_investigation_normal_gait.md`.
