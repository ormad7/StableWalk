# Erik requirements — audit, architecture, and implementation plan

**Status:** audit complete. Requirement 2 implemented 2026-10-12 (`docs/erik_requirement_2_report.md`). Requirement 3 implemented 2026-10-12 (`docs/spectral_features.md`). Requirement 4 implemented 2026-10-12 (`docs/erik_requirement_4_report.md`, `docs/asymmetry3d.md`). Requirement 5 **plan complete, not implemented** 2026-10-12 (`docs/erik_requirement_5_plan.md`).  
**Date:** 2026-10-12  
**Scope for next implementation:** supervisor requirements **2–4** only (validate before any simulator code).  
**Requirement 1:** deferred (see §8). **Requirement 5:** architecture/feasibility only — no physics engine or RL trainer.

---

## 1. Supervisor requirements (as given)

| # | Requirement | This plan |
|---|-------------|-----------|
| 1 | Human wire gait model with ~12–20 joints / DOF | **Mostly already present.** Do not rebuild. Document mapping; optional later alignment. |
| 2 | Spectral characteristics via **autocorrelation matrix** analysis | **Not implemented.** Design in §6–7. |
| 3 | **Spectral feature extraction** | **Not implemented.** Design in §6–7. |
| 4 | **Dynamic 3D reconstruction** and **asymmetric moments** | Reconstruction **exists** (heuristic monocular). Asymmetric **moments** as a 3D/spectral statistic **do not exist**. Design in §6–7. |
| 5 | Reinforcement / deep-learning simulator for unstable gait | **Planned, not implemented.** See `docs/erik_requirement_5_plan.md`. AMP export and contact-sync formulas exist; Isaac/MuJoCo/Gym are not installed. Do not call this an RL system. |

This document claims an algorithm exists **only** when source files prove it.

---

## 2. Checkpoint (safe rollback)

| Item | Value |
|------|--------|
| Git root | `StableWalk/` |
| Last committed HEAD | `a2d63c5cca0cba4fc8d45d522e1db3211e91232e` |
| HEAD message | `Polish StableWalk UI for laboratory presentation quality.` |
| Annotated tag | `erik-req-2-4-audit-baseline` (created after this audit) |
| Working tree | **Dirty.** Many uncommitted UI/analysis edits from prior sessions. **Not committed** by this task. |
| Do **not** checkpoint | `data/demo_videos/_gavd_cache/` and other generated demo caches |

**Rollback of committed history:** `git checkout erik-req-2-4-audit-baseline` (or `a2d63c5`).  
**Rollback of uncommitted work:** restore from this tag only after a later commit, or keep a local stash; this audit does not squash or reset the dirty tree.

This task adds **only** this plan file. It must not overwrite pose math, OpenSim IK, playback, or exports.

---

## 3. Repository inventory (what already exists)

### 3.1 Pipeline (end to end)

```
Video (OpenCV) → PoseSequence JSON (MediaPipe / optional SMPL)
       → GaitMotionRecording (canonical 3D + DOF)
       → gait cycles, contact, vGRF proxy, biomechanics
       → Tk dashboard + CSV/JSON/NPZ/PDF exports
       → optional OpenSim TRC/MOT + experimental IK
       → optional Real-to-Sim AMP NPZ (external Isaac Lab)
```

Entry: `python main.py --gui`. Canonical motion conversion: `stablewalk.adapters.pose_adapter.pose_sequence_to_gait_motion`.

### 3.2 Pose and 3D

| Module | Role | Proven |
|--------|------|--------|
| `stablewalk/pose/video.py` | OpenCV decode; `timestamp_s = index / fps` | Yes |
| `stablewalk/pose/backends/mediapipe_backend.py` | Default 2D+relative-z landmarks | Yes |
| `stablewalk/models/pose_data.py` | `Keypoint` (normalized image x,y,z, visibility), `PoseFrame`, `PoseSequence` | Yes |
| `stablewalk/pose/skeleton_3d.py` | Stick figure, 15 visualization joints, heuristic depth layers | Yes |
| `stablewalk/pose/reconstruction.py` | `SkeletonSequence3D`: `(T,J,3)` positions, visibility, root trajectory, fps | Yes — **not SMPL, not triangulation** |
| `stablewalk/coordinates/coordinate_map.py` | Frame IDs, units, MediaPipe → canonical / OpenSim TRC | Yes |
| `stablewalk/adapters/pose_adapter.py` | `PoseSequence` → `GaitMotionRecording` | Yes |
| `stablewalk/pose/backends/smpl_joint_map.py` | Optional SMPL; uses scipy Rotation **if present** | Optional path only |

`reconstruction.py` states explicitly: simplified 3D from 2D, hip-centered, body-normalized, heuristic Z. That is the current “dynamic 3D reconstruction.” It is **not** laboratory photogrammetry.

### 3.3 Data models (authoritative time series)

| Container | Location | Use |
|-----------|----------|-----|
| **`GaitMotionRecording` / `JointTimeSeries`** | `stablewalk/models/gait_motion.py` | Authoritative synchronized timeline for GUI, OpenSim bridge, gait features |
| `PoseSequence` / `PoseFrame` | `stablewalk/models/pose_data.py` | Raw pipeline product (all frames, including undetected) |
| `MotionSeriesIndex` | `stablewalk/models/motion_series.py` | GUI-oriented angles + XYZ from pose frames |
| `JointMotionBundle` / `JointMotionSeries` | `stablewalk/ui/viewers/joint_motion_analysis_chart.py` | Motion tab kinematics (NaN-aware) |
| `SkeletonSequence3D` | `stablewalk/pose/reconstruction.py` | Dense numpy `(T,J,3)` reconstruction |

**Primary input for requirements 2–4:** `GaitMotionRecording.build_time_series()` plus snapshot `metadata["landmark_visibility"]`. Do not invent a second motion store.

### 3.4 Wire / joint / DOF model (requirement 1 — already close)

| Registry | Count | File |
|----------|-------|------|
| Canonical joints `JOINT_IDS` + pelvis | 21 + `pelvis` | `models/joint_registry.py` |
| Revolute DOF `DOF_IDS` | **13** flexion-style angles | `joint_registry.py` |
| Exported gait angles `GAIT_ANGLE_FIELDS` | **15** fields | `pose/dof.py` (`MIN_GAIT_DOF = 13`) |
| GUI selectable items `GUI_DOF_ITEM_IDS` | **16** (hips→toes, shoulders→wrists) | `ui/dof_selection.py` |
| 3D vis joints `SKELETON_3D_JOINTS` | **15** | `pose/skeleton_3d.py` |
| Hierarchy `JOINT_PARENTS` | Tree from pelvis | `joint_registry.py` |

The stick-figure + 13–16 DOF already satisfies “approximately 12–20 joints or degrees of freedom.” Next work should **reuse** these IDs, not introduce a parallel skeleton.

### 3.5 Gait metrics, “features,” and asymmetry (existing ≠ supervisor spectral)

**Exists (scalar / spatiotemporal / ROM):**

- `analysis/gait_cycle_analysis.py`, `gait_cycle_phase.py`, `gait_feature_analysis.py`
- `analysis/biomechanical/symmetry_metrics.py` — L/R step, stance, swing, ROM ratios
- `analysis/advanced/features.py` — fixed-length **gait-lab style** vector (`cadence_hz`, `symmetry_score`, …). **Not FFT / not autocorrelation.**
- `analysis/gait_feature_analysis.py` uses `np.corrcoef` on **mean left vs right knee curves over gait-cycle %**. That is a **scalar phase-consistency score**, not an autocorrelation matrix of a multivariate time series.
- `analysis/advanced/ml_stability.py` — optional Ridge on those gait features (sklearn if installed). **Not** an RL gait simulator.

**Does not exist (repo-wide search of `*.py` and `*.md`):**

- `fft` / `rfft` / `periodogram` / `welch`
- `autocorr` / autocorrelation matrix / Toeplitz lag matrix
- Eigen-analysis of a lag-covariance for spectral gait modes

### 3.6 “Moments” today vs requirement 4

| Existing | What it actually is |
|----------|---------------------|
| L/R symmetry ratios | Scalar spatiotemporal / ROM comparison |
| Estimated vGRF / `analysis/forces.py` | Vertical **force proxy** from CoM acceleration, not instrumented GRF |
| `analysis/opensim_id_readiness.py` | Documents that **OpenSim Inverse Dynamics joint moments are not available** (no measured GRF/COP) |
| COM / BoS / stability margin | Estimated kinematics, not kinetic moments |

**Not in source:** 3D trajectory moment tensors, lag-moment spectral features, or inverse-dynamics joint torques.

For requirement 4 this plan uses **kinematic asymmetric moments** (left vs right 3D position/velocity statistics and second-moment spatial spread), computed on the existing reconstruction. It will **not** call these OpenSim ID joint moments.

### 3.7 GUI (must stay additive)

Four dashboard tabs in `ui/tk/`: Overview, Motion Analysis, Biomechanics, Advanced & Export. Playback is shared. Motion charts already plot angle, ω, α, speed, XYZ vs `time_s`.

Spectral / 3D-moment UI, if added later, belongs as **new charts or a new Advanced subsection**, not a rewrite of Motion or Overview.

### 3.8 OpenSim and exports

| Piece | Status |
|-------|--------|
| TRC/MOT export | Present (`opensim_integration.py`, `biomechanics/service.py`) |
| Experimental IK | Present; status separated from TRC/MOT pipeline (`opensim_ik_status.py`) |
| Inverse dynamics | Readiness audit only — **do not feed proxy GRF into ID** |
| Session DB / JSON / CSV / PDF / NPZ | `io/`, `storage/` |
| AMP / Isaac Lab | Export artifacts only (`real_to_sim/`, `analysis/isaac_lab_integration.py`). `ISAAC_LAB_AVAILABLE = False` |

### 3.9 Tests

`tests/` is a large pytest suite (layout, playback, gait, OpenSim status, biomechanics, reconstruction helpers). There is **no** spectral/autocorrelation test module. New work must add tests without weakening existing ones.

### 3.10 Dependencies

Required (`requirements.txt`): OpenCV, MediaPipe, **numpy**, matplotlib, Pillow, reportlab, pytest.

**Do not add** scipy, sklearn, PyTorch, Isaac, or OpenSim as required packages for requirements 2–4. Numpy is sufficient (`np.correlate`, `np.fft.rfft`, `np.linalg.eigh`). Optional scipy, if already installed, may be used behind a try/import — same pattern as `smpl_joint_map.py`.

---

## 4. Synchronized joint time series — facts from source

### 4.1 How the series is built

1. Video FPS from `cv2.CAP_PROP_FPS` (fallback **30** if missing). Frame time: `index / fps` (`pose/video.py`).
2. `PoseFrame.timestamp_s` / `timestamp_ms`; `PoseFrame.time_seconds(fps)` prefers timestamp, else `frame_index / fps`.
3. `pose_sequence_to_gait_motion(..., detected_only=True)` **drops** frames that are not `detected` or have no keypoints (aligned with GUI `pose_indices`).
4. Each kept frame becomes a `SkeletonSnapshot` with `frame_index` and `time_s=frame.time_seconds(fps)`.
5. `GaitMotionRecording.build_time_series()` walks snapshots in order into:
   - `times_s: list[float]`
   - `positions[joint_id]: list[Vec3]`
   - `angles`, `velocities`, `dof_angles`, `dof_velocities` (`float | None`)

GUI conversion sites: `ui/tk/app.py`, `ui/tk/session_cache.py`.

### 4.2 Coordinate conventions and units

Documented in `docs/COORDINATE_SYSTEMS.md` and `coordinates/coordinate_map.py`.

| Frame | ID | Origin | Axes | Units | Who uses it |
|-------|-----|--------|------|-------|-------------|
| **Canonical (analysis / 3D GUI)** | `sw_canonical_y_up` (`pelvis_centered_y_up_z_forward` alias) | Pelvis / mid-hip | +X subject right, +Y up, +Z anterior / toward camera | **Body-normalized meters** (~1.0 = nose-to-ankle span) | `GaitMotionRecording`, skeleton, gait metrics, Motion XYZ |
| MediaPipe image | `mediapipe_normalized_image` | Image top-left | +X right, +Y **down**, z relative depth | **Normalized 0–1** (not pixels) | `Keypoint`, overlay, OpenSim TRC **input** |
| OpenSim TRC | `opensim_trc_export_mm` | Lab-style from **image** landmarks | Separate from canonical | **millimetres** | IK export only |
| MediaPipe world | `mediapipe_world_landmarks_m` | Optional | MP world +Y up | metres | **Not** default pipeline |

`Vec3` docstring says “meters (hip-centered).” That is **model-relative / body-normalized metres**, not world mocap metres. Hip centering **removes** walking translation from joint XYZ; `SkeletonSequence3D.root_trajectory` holds a weak pre-centering pelvis path.

**Spectral / 3D-moment code must tag every array with `coordinate_system` and units.** Default: canonical `sw_canonical_y_up`, body-normalized m, angles in degrees.

Do **not** mix OpenSim TRC mm into the autocorrelation matrix unless the user explicitly selects the IK MOT path (later optional).

### 4.3 Timestamps, FPS, gaps

- Nominal sampling: `recording.fps` (video FPS).
- `detected_only=True` **omits** failed detections → `times_s` can have **gaps** (not a strictly uniform grid).
- Consecutive snapshot `dt` may therefore be `k / fps` for integer k, not always `1/fps`.
- `attach_snapshot_kinematic_rates` uses actual `time_s` delta, with fallback `1/fps` if `dt <= 0`.
- Video quality already flags `fps < 20` (`analysis/biomechanical/video_quality.py`).

**Implication for spectra:** resample onto a uniform grid (`np.arange(t0, t1, 1/fps)`) before FFT or lag-covariance. Never assume `len(times) * 1/fps == duration` without checking `np.diff(times_s)`.

### 4.4 Missing values (critical)

| Layer | Missing-joint behavior |
|-------|------------------------|
| `DofSample.angle_deg` | `None` allowed; `build_time_series` keeps `None` |
| `JointTimeSeries.positions` | **Missing joint filled with `Vec3(0,0,0)`** (pelvis origin) |
| `JointMotionSeries` (Motion tab) | **NaN** if joint absent on the snapshot |
| `MotionSeriesIndex.joint_xyz` | `(None, None, None)` then skipped in trajectories |

**Do not feed `JointTimeSeries.positions` zeros into FFT/autocorrelation.** Treat `(0,0,0)` as missing when the joint is absent from the snapshot **or** visibility is below threshold. Prefer building arrays from snapshots the same way `build_joint_motion_series` does (NaN), not from the zero-filled `JointTimeSeries.positions` lists.

### 4.5 Pose confidence

| Source | Range | Notes |
|--------|-------|-------|
| `Keypoint.visibility` | 0–1 | MediaPipe landmark score |
| `SkeletonSnapshot.metadata["landmark_visibility"]` | canonical joint → float | Filled in `pose_adapter` from keypoints |
| `SkeletonSequence3D.visibility` | `(T, J)` | Reconstruction path |
| `confidence_for_joint_at_frame` | 0–1 or None | Matches `keypoint.name == joint_id` — **can miss** GUI ids (`head` vs `nose`, `left_toe` vs `left_foot_index`) |

Spectral masking should use **snapshot `landmark_visibility`** (canonical ids) with a documented threshold (propose **0.5**, configurable). Samples below threshold → NaN, then interpolate or drop before lag estimation.

`_extract_image_xy` already skips keypoints with `visibility < 0.3` for the 2D overlay only; that is **not** a spectral mask.

---

## 5. What can be reused vs what must be new

### Reuse (do not rewrite)

- `GaitMotionRecording`, `SkeletonSnapshot`, `joint_registry` IDs
- `pose_adapter.pose_sequence_to_gait_motion`
- `coordinate_map` frame IDs / units
- `build_joint_motion_series` NaN + finite-difference pattern
- Gait cycle indices from `gait_cycle_analysis` (optional cycle-normalized spectra later)
- Export helpers in `io/` (add a new JSON/NPZ writer; do not change existing metric JSON schema meaning)
- Scientific labeling pattern in `ui/scientific_labels.py` (`estimated` / `derived`)

### Do not reuse as if they were spectral

- `analysis/advanced/features.py` (spatiotemporal vector)
- `symmetry_metrics.py` (scalar L/R)
- `np.corrcoef` knee phase score
- vGRF / OpenSim ID readiness
- AMP / Isaac Lab exports (requirement 5)

### New packages (additive)

```
stablewalk/analysis/spectral/          # req 2–3
  __init__.py
  series.py          # uniform grid, masks, confidence, channel packing
  autocorrelation.py # lag-covariance / autocorrelation matrix + eigenmodes
  features.py        # spectral feature vector + provenance
  report.py          # dataclass for GUI/export

stablewalk/analysis/asymmetry3d/       # req 4
  __init__.py
  reconstruction.py  # thin adapter over existing 3D (no second reconstructor)
  moments.py         # L/R 3D kinematic moments
  compare.py         # asymmetry indices with units and caveats
```

GUI and export come in a **later implementation phase**, after unit tests on synthetic signals. No change to IK math, playback clocks, or existing gait metric formulas.

---

## 6. Modular architecture (requirements 2–4)

### 6.1 Data contract (single ingress)

```text
GaitMotionRecording
    → SpectralSeriesBuilder
         channels: named 1-D float arrays on a uniform time grid
         mask: bool (finite & visibility ≥ τ)
         meta: fps, coordinate_system, units, joint_ids, channel_kind
```

**Default channel set (wire-model aligned, ~16–20):**

- Positions: selected joints × `{x,y,z}` in canonical m (body-normalized)
- Optional: DOF angles (deg) for hips, knees, ankles, shoulders, elbows
- Optional: pelvis `root_trajectory` from `SkeletonSequence3D` if translation modes are wanted (must be labeled separately; not hip-centered)

Start with **lower-limb XYZ + sagittal flexion DOF** (hips, knees, ankles) so the autocorrelation matrix stays interpretable (~18–24 channels). Full 16 GUI joints × 3 axes is allowed but document condition number / regularization.

### 6.2 Requirement 2 — autocorrelation matrix

**Definition to implement (standard DSP, numpy-only):**

1. Detrend each valid channel (subtract mean; optional linear detrend).
2. Optional z-score per channel (document; default **on** for mixed units if angles and metres are stacked — **or** keep separate matrices for position vs angle).
3. For lags `k = 0 … K` (`K` default: `min(2 s * fps, T//2)`):
   - biased or unbiased sample autocorrelation / lag-covariance  
   - **Primary supervisor object:** multivariate **lag-0 autocorrelation / covariance matrix** `R = (1/T) X_cᵀ X_c` of size `(C, C)`, plus optional **block-Toeplitz** of stacked lags if the supervisor expects “the” autocorrelation matrix in the Yule–Walker sense.
4. Symmetric eigendecomposition: `R v = λ v` (`np.linalg.eigh`).
5. Outputs: eigenvalues (spectrum of R), explained variance ratio, leading eigenvectors (joint-space modes), optional per-channel `r_xx(k)` for cadence peak.

**Not claimed as existing.** Closest code is `np.corrcoef` on **cycle-averaged knees**, which is a 2-vector correlation, not `R`.

**Numerical rules:**

- Minimum valid samples after masking (propose `T_eff ≥ 3 * fps` and `≥ 64`).
- Regularize `R += ε I` with small `ε` if `cond(R)` is large.
- If gaps remain after interpolation, use pairwise complete observations for lag-0 only and flag `method="pairwise"`; do not silently zero-fill.

### 6.3 Requirement 3 — spectral feature extraction

From the **same** uniform series (and from `R`’s eigenvalues):

| Feature | Source | Unit / note |
|---------|--------|-------------|
| Dominant frequency per channel | `rfft` peak in 0.3–4 Hz (gait band) | Hz |
| Spectral centroid / bandwidth | magnitude spectrum in gait band | Hz |
| Band power | 0.3–1.5 Hz (stride), 1.5–3 Hz (step/harmonics) | fraction of total |
| Harmonic ratio | power at 2f₀ / f₀ | dimensionless |
| Autocorr lag of first secondary peak | `r_xx(k)` | seconds (~stride time) |
| Eigenvalue entropy / participation | spectrum of `R` | derived |
| Condition number of `R` | `λ_max/λ_min` | quality flag |

Store as a dataclass with `kind="derived"`, `coordinate_system`, `channel_set_id`, and warnings (short clip, low FPS, high missingness).

**Do not replace** `analysis/advanced/features.py`. Optionally **append** a few spectral keys later under a namespaced export (`spectral.*`) so old CSV columns stay stable.

### 6.4 Requirement 4 — dynamic 3D reconstruction and asymmetric moments

**Reconstruction:** call existing `build_skeleton_sequence_3d` / `GaitMotionRecording` joints. Do **not** add a second depth heuristic. Document `ASSUMPTIONS` from `reconstruction.py` on every 3D-moment report.

**Asymmetric moments (kinematic, estimated):**

For paired joints `(left_*, right_*)` in canonical frame, over the same masked interval:

1. **First moments:** mean position `(μx, μy, μz)` left vs right; difference Δμ.
2. **Second moments:** covariance of the 3D trajectory (spatial spread ellipsoid); compare trace, principal axes, and volume proxy `det(Σ)^{1/6}`.
3. **Velocity second moments:** covariance of linear velocity (from finite differences on uniform grid).
4. **Time-series moment asymmetry:** for each axis, compare 2nd/3rd/4th statistical moments (variance, skewness, kurtosis) left vs right.
5. **Optional coupling to req 2:** left-block vs right-block of `R`, or `‖R_LR − R_RLᵀ‖` as a coherence/asymmetry residual.

**Explicit non-goals for this phase:**

- Inverse-dynamics joint **torques** (OpenSim ID)
- Passing vGRF proxies into ID
- Claiming millimetre-accurate world reconstruction

If the supervisor later wants kinetic moments, that is a **new** workstream gated on measured GRF (`opensim_id_readiness.py`).

### 6.5 Validation architecture

| Layer | Method |
|-------|--------|
| Unit (synthetic) | Pure sinusoid at known f₀ and fps → peak within 1 bin; two-channel identical signals → `R` rank-1; left=right trajectories → moment Δ ≈ 0 |
| Unit (gaps) | Punch holes + visibility mask → features flagged, no crash, no zero-fill FFT |
| Regression | Existing pytest suite must stay green |
| Optional demo JSON | `scripts/validate_biomechanical_demos.py` pattern: run spectral report on a checked-in pose JSON, snapshot selected scalars with loose tolerances |
| Scientific | Label all outputs estimated/derived; cite monocular 3D limits |

---

## 7. Implementation plan (after this audit)

**Constraint:** additive modules + tests + optional export/GUI. No overwrite of working pose, IK, playback, or gait-metric code.

### Phase A — series adapter (no GUI)

- `SpectralSeriesBuilder.from_recording(recording, joint_ids, …)`
- Uniform resample, visibility mask, refuse zero-filled positions
- Tests: timestamp gaps, FPS fallback, NaN policy

### Phase B — autocorrelation matrix (req 2)

- Lag-0 `R`, eigenvalues/vectors, optional per-channel `r_xx`
- Tests: known covariance, symmetry of `R`, eigenvalue sum vs trace

### Phase C — spectral features (req 3)

- rFFT band features + lag-peak stride time
- Tests: sinusoid f₀, harmonic ratio on 2-tone signal

### Phase D — 3D asymmetric moments (req 4)

- Adapter over existing reconstruction
- L/R moment tables + warnings
- Tests: mirrored synthetic walk → near-zero asymmetry; one-sided offset → detected Δ

### Phase E — export (optional, still no behavior change to old files)

- `spectral_report.json` / `.npz` beside session outputs
- Provenance block (fps, frame, units, mask stats)

### Phase F — GUI (optional, last)

- Read-only charts: eigenvalue scree, autocorr lag, L/R moment bars
- Advanced or Motion subsection; do not auto-mark pipeline stages complete

Requirement 1 (wire model polish) and 5 (RL simulator) stay **after** A–D.

---

## 8. Deferred: requirements 1 and 5

**Req 1.** Already 13 DOF + 16 GUI joints + hierarchical wire model. Later: pick one documented 12–20 node subset for papers (propose **16 GUI joints + pelvis** = 17 nodes) and freeze IDs in the spectral channel list.

**Req 5.** Plan: `docs/erik_requirement_5_plan.md`. Contracts: `stablewalk/sim/` (flags stay False). `real_to_sim/` exports AMP clips and contact-sync **scores**; Isaac Lab / MuJoCo / Gymnasium are **not** installed. A DL/RL unstable-gait simulator would be stages S3–S5 after requirements 2–4 validate. Do not pretend AMP export is that simulator. Simulation reward ≠ predicted instability ≠ clinical diagnosis.

---

## 9. Acceptance criteria (implementation phases A–D)

### Global

- [ ] No edits to OpenSim IK math, gait-cycle timing formulas, vGRF equations, or playback clocks.
- [ ] No new required dependencies beyond numpy (already required).
- [ ] Existing pytest suite remains green (or only pre-existing failures documented).
- [ ] Every public result includes `coordinate_system`, units, `fps`, `n_samples`, `missing_fraction`, `kind` (derived/estimated).
- [ ] Documentation does not claim FFT/autocorrelation existed before these modules.

### Requirement 2

- [ ] Given `GaitMotionRecording`, code produces a symmetric `(C,C)` autocorrelation/covariance matrix `R`.
- [ ] Eigendecomposition returns `C` real eigenvalues, descending, `sum(λ) ≈ trace(R)` within numerical tolerance.
- [ ] Synthetic two-channel identical series → leading eigenvalue dominates; off-diagonal of correlation form ≈ 1.
- [ ] Zero-filled missing joints are **not** treated as true (0,0,0) motion.

### Requirement 3

- [ ] Feature vector includes at least: dominant frequency, band-power ratio, first autocorr lag peak (s), eigenvalue entropy or participation ratio.
- [ ] Unit sinusoid at 1.0 Hz, fps=30, T≥6 s → dominant frequency **1.0 ± 1 bin**.
- [ ] Short or empty series returns structured “unavailable” + warnings, not exceptions or fake numbers.

### Requirement 4

- [x] 3D positions come from existing reconstruction / gait motion (no second solver).
- [x] For each paired joint, report L/R first and second kinematic moments and a scalar asymmetry index with units.
- [x] Symmetric synthetic trajectories → indices near 0; documented threshold.
- [x] Report text states reconstruction is monocular-heuristic, not mocap, and moments are **not** inverse-dynamics torques.

### Validation

- [ ] `tests/test_spectral_series.py`, `tests/test_spectral_autocorrelation.py`, `tests/test_spectral_features.py`, `tests/test_asymmetry3d_moments.py`
- [ ] No GUI required for A–D to pass CI.

---

## 10. Risks and supervisor-facing caveats

1. **Monocular Z** is anatomical layers + MediaPipe relative depth. Spectral power on Z is weaker and more biased than X/Y. Prefer sagittal DOF + X/Y, or down-weight Z.
2. **Hip-centered coordinates** remove progression; “stride spatial second moments” in joint XYZ are **relative to pelvis**, not lab-frame step length. Use `root_trajectory` or image-space step metrics if world progression is required — and label it.
3. **Dropped detections** bias spectra toward well-tracked frames (often mid-stance). Report `missing_fraction`.
4. **Low FPS** (<20) aliases gait harmonics; reuse video-quality warnings.
5. **“Asymmetric moments”** must be defined in the thesis glossary as kinematic/statistical, unless a force plate is added.

---

## 11. Existing tests (this audit run)

Filled after `python -m pytest tests -q --tb=line` from `StableWalk/` on 2026-10-12. See the closing section of this file once the run finishes.

---

## 12. Next action after plan sign-off

Implement **Phase A** only (`stablewalk/analysis/spectral/series.py` + tests). Do not wire GUI until A–D tests pass.
