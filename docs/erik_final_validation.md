# Erik final validation — requirements 2–4

**Project:** StableWalk M.Sc.  
**Supervisor:** Erik  
**Audit date:** 2026-10-14  
**Focus:** (2) spectral characteristics / autocorrelation matrices, (3) spectral feature extraction, (4) dynamic 3D reconstruction and asymmetric moments.  
**Out of scope for a pass/fail close-out:** (5) reinforcement-learning gait simulator (later stage). Requirement 1 is noted only as context.

**Rule:** GUI tabs are not evidence. Status is taken from source functions, formulas as coded, and tests **executed in this session**. Full implementation of 2–4 as laboratory biomechanics is **not** claimed.

**This session**

| Item | Result |
|------|--------|
| NumPy / SciPy | 2.0.0 / 1.14.0 |
| Pytest | **73 passed** in 22.34 s (`test_spectral_analysis`, `test_spectral_features`, `test_erik_req24_validation`, `test_asymmetry3d`, `test_asymmetry3d_moments`, `test_opensim_id_readiness`, `test_wire_model_inventory`, `test_final_scientific_acceptance`) |
| Real videos | `data/demo_videos/normal_gait.mp4`, `abnormal_gait.mp4` present; analysis on `data/output/poses/{normal,abnormal}_gait_poses.json` |
| OpenSim ID | `ready_for_traditional_id = False` on all four recordings |
| Feature export | JSON schema `stablewalk-spectral-analysis`; CSV contains `# autocorrelation` and `# session_vector` |

Interactive Tk was not used as scientific evidence. OpenSim InverseKinematicsTool was **not re-solved** in this script; `.mot` files on disk (120 rows) and a same-day GUI IK run (120 frames, 449 ms) are cited separately.

---

## Acceptance summary

| Requirement | Verdict | Meaning |
|-------------|---------|---------|
| 1. Wire model (12–20) | **PARTIAL** | 21 trackable joints + pelvis; 13 sagittal DOFs. Joints ≠ DOFs. |
| 2. Spectral / autocorrelation matrix | **PASS** | Method validated vs NumPy/SciPy. Demo clips yield no ACF stride peak. |
| 3. Spectral feature extraction | **PASS** | Features coded, exported, and checked. Several ACF keys are correctly `None` on short videos. |
| 4a. Dynamic 3D reconstruction | **PARTIAL** | Per-frame XYZ exist; Z is heuristic; bone-length flags fail on 100% of frames. |
| 4b. Asymmetric moments | **PARTIAL** | Kinematic SI/ASI implemented. Physical joint torques correctly **absent** (no GRF). |
| 5. RL simulator | **NOT TESTED** | Later stage; not implemented as a deliverable. |

**Overall 2–4: PARTIAL.** DSP requirements 2–3 pass. Requirement 4 does not meet a force-plate / mocap reading of “dynamic 3D and asymmetric moments.”

---

## Requirement 2 — Spectral characteristics (autocorrelation matrix)

**Verdict: PASS** (method). Operational result on ~4 s demo videos: ACF stride estimator **insufficient**.

### Implementation

| Piece | File | Function |
|-------|------|----------|
| Normalized ACF | `analysis/spectral/autocorrelation.py` | `normalized_autocorrelation`, `analyze_autocorrelation` |
| Multivariate lagged matrices | `analysis/spectral/matrices.py` | `lagged_matrix`, `lagged_cross_cov`, `lagged_cross_correlation` |
| FFT / Welch | `analysis/spectral/spectrum.py` | `rfft_periodogram`, `welch_psd`, `welch_psd_numpy` |
| Eigenanalysis | `analysis/spectral/eigen.py` | `eigh_symmetric`, `decompose_r0` |
| Series from video pose | `analysis/spectral/series.py`, `report.py` | `build_spectral_series`, `analyze_spectral` |
| Orchestration | `analysis/spectral/report.py` | `analyze_spectral` |

Input: `GaitMotionRecording` (video → MediaPipe → adapter). Default univariate channel: left knee flexion. Default matrix: six sagittal DOFs.

### Mathematical formulas (as coded)

Normalized ACF after mean removal on the longest finite run:

\[
\rho(k)=\frac{\sum_t x[t]\,x[t+k]}{\sum_t x[t]^2},\qquad \rho(0)=1
\]

(`np.correlate` on the centered series, then lag-0 normalize.)

Lagged covariance:

\[
R_{ij}(\tau)=\mathrm{cov}\big(x_i(t),\,x_j(t+\tau)\big)
\]

For \(\tau\neq 0\), \(R(\tau)\) is not symmetric; \(R(\tau)=R(-\tau)^\top\). `eigh_symmetric` **refuses** lagged matrices (`NonsymmetricEigError`).

\(R(0)\) correlation is Pearson \(D^{-1/2} R_{\mathrm{cov}}(0)\,D^{-1/2}\). Eigenvalues: `numpy.linalg.eigh` on the symmetrized finite matrix, sorted descending.

Welch / rFFT: one-sided density (\(V^2/\mathrm{Hz}\)), Hann window; SciPy used when installed (`scaling="density"`). Walking band for peaks: 0.4–4 Hz.

Missing samples stay NaN (gap-limited interpolate). They are not zero-filled.

### Tests and numerical results (this session)

| Test | Result |
|------|--------|
| \(\rho(k)\) vs `np.correlate` (1.25 Hz sine, 400 samples, 50 Hz) | max abs error **0** → **PASS** |
| \(R(\tau)=R(-\tau)^\top\) at \(\tau=10\) | max abs error **0** → **PASS** |
| `welch_psd_numpy` vs `scipy.signal.welch` | max relative error **1e-10** → **PASS** |
| Active Welch backend | `scipy.signal.welch` |
| rFFT dominant frequency vs 1.25 Hz | **1.25 Hz**, error **0** → **PASS** |
| SciPy Welch dominant vs 1.25 Hz | **1.171875 Hz**, error **0.078125 Hz** (nperseg=128 bin) → **PASS** |
| Synthetic regular gait ACF stride | **1.00 s** (GT 1.00 s), \(\rho(0)=1\), reliability moderate → **PASS** |
| \(R(0)\) 6×6 symmetric on all four recordings | True → **PASS** |

| Dataset | ACF \(\rho(0)\) | ACF stride (s) | Reliability | Derived from video series? |
|---------|-----------------|----------------|-------------|----------------------------|
| Synthetic regular 1 Hz / 8 s | 1.0 | **1.00** | moderate | No (analytic wire) |
| Synthetic asymmetric | 1.0 | **1.00** | moderate | No (analytic wire) |
| `normal_gait` poses (120 fr, 25 Hz, 4.76 s) | 1.0 | **None** | insufficient | **Yes** — `normal_gait_poses.json` from `normal_gait.mp4` |
| `abnormal_gait` poses (120 fr, 29.97 Hz, 3.97 s) | 1.0 | **None** | insufficient | **Yes** — `abnormal_gait_poses.json` from `abnormal_gait.mp4` |

Pixel-level pose was not re-extracted in this session. The pose JSON is the project’s video-derived time series.

### Remaining limitations

- Demo clips are too short for the configured ACF stride-lag search.
- Eigenanalysis is only on symmetric \(R(0)\), by design.
- Default matrices use six sagittal DOFs, not every named joint pair.
- Results are DSP estimates, not diagnoses.

---

## Requirement 3 — Spectral feature extraction

**Verdict: PASS** (extraction and export). Several keys are **correctly unavailable** on the demo videos.

### Implementation

| Piece | File | Function |
|-------|------|----------|
| Feature set | `analysis/spectral/features.py` | `spectral_entropy`, `band_power`, `spectral_bandwidth`, session-vector assembly |
| Dominant frequency | `analysis/spectral/spectrum.py` | `_dominant_in_band` (used from features) |
| Export | `analysis/spectral/export.py` | `export_spectral_json`, `export_spectral_csv` |
| Keys | `SESSION_VECTOR_KEYS` (22 names, schema order) | |

GUI (`dashboard_spectral_features.py`) only displays these outputs.

### Mathematical formulas (as coded)

Let \(P(f)\) be the one-sided PSD and \(B=[0.4,4]\,\mathrm{Hz}\). \(p_i=P_i/\sum P_j\) on bins in \(B\) with \(P>0\).

| Feature | Formula |
|---------|---------|
| Dominant frequency | \(f_0=\arg\max_{f\in B} P(f)\) |
| Band power | \(\sum_{i\in B} P_i\,\Delta f\) |
| Spectral entropy | \(H=(-\sum p_i\log p_i)/\log N\in[0,1]\) |
| Centroid | \(c=\sum f_i p_i\) |
| Bandwidth | \(\sqrt{\sum p_i(f_i-c)^2}\) |
| Harmonic ratio | \(P(2f_0)/P(f_0)\) at nearest bin in \(B\) |
| Regularity | \(\rho\) at ACF stride lag (None if no peak) |
| L/R phase | \(\mathrm{wrap}_{180}(360^\circ\cdot\tau_{\mathrm{peak}}/T_{\mathrm{stride}})\) |
| L/R deltas | frequency, entropy, RMS amplitude |

Unavailable inputs → `value=None`, `available=False`. Zeros are not fabricated.

### Tests and numerical results

| Test | Result |
|------|--------|
| Entropy vs NumPy Shannon definition | identity (unit test) → **PASS** |
| Band power vs \(\sum P_i\Delta f\) | identity → **PASS** |
| JSON + CSV session vector | schema present; `# session_vector` in CSV → **PASS** |
| Opposite-phase L/R sines | xcorr lag ≈ 0.5 s, phase ≈ 180° (unit test) → **PASS** |

**Synthetic regular (GT stride 1.0 Hz)**

| Key | Value | Note |
|-----|-------|------|
| dominant_frequency_hz | 0.976562 | Welch bin; error 0.023 Hz |
| spectral_entropy | 0.300081 | |
| spectral_centroid_hz | 1.000019 | |
| spectral_bandwidth_hz | 0.11271 | |
| harmonic_ratio | ~0 | pure fundamental |
| acf_stride_peak / lag | 0.873 / **1.00 s** | |
| stride_cv | 0 | |
| lr_amplitude_delta | 0 | symmetric amps |
| lr_xcorr_lag_s | −0.48 | right knee generated at \(+\pi\) |

**Synthetic asymmetric (20° / 40° knees):** same primary-channel spectrum; `lr_amplitude_delta` **14.07**.

**`normal_gait` (video-derived)**

| Key | Value | Available |
|-----|-------|-----------|
| dominant_frequency_hz | 0.417 | yes |
| spectral_entropy | 0.695 | yes |
| spectral_centroid_hz | 1.017 | yes |
| spectral_bandwidth_hz | 0.746 | yes |
| harmonic_ratio | 0.164 | yes |
| acf_stride_peak / lag / regularity | null | **no** |
| stride_cv | 0.228 | yes (2 cycles) |
| lr_relative_phase_deg | null | **no** |

**`abnormal_gait` (video-derived)**

| Key | Value | Available |
|-----|-------|-----------|
| dominant_frequency_hz | 0.500 | yes |
| spectral_entropy | 0.800 | yes |
| spectral_centroid_hz | 1.249 | yes |
| harmonic_ratio | 0.789 | yes |
| acf_stride_peak | null | **no** |
| stride_cv | 0.660 | yes |

Dominant frequencies ~0.4–0.5 Hz on the demo clips are **not** asserted to be true stride frequency.

### Remaining limitations

- Not clinically validated as pathology markers.
- Harmonic ratio on a pure sine is degenerate (~0).
- Phase requires a stride period; demo ACF peak is missing, so phase is None.

---

## Requirement 4 — Dynamic 3D reconstruction and asymmetric moments

Split because the wording mixes geometry and kinetics.

### 4a. Dynamic 3D reconstruction — PARTIAL

#### Implementation

| Piece | File | Function |
|-------|------|----------|
| Monocular 3D wire | `pose/reconstruction.py` | `build_skeleton_sequence_3d`, `reconstruct_frame_3d` |
| 15-joint MediaPipe subset | `pose/skeleton_3d.py` | `SKELETON_3D_JOINTS`, `reconstruct_skeleton_3d` |
| Analysis kinematics (22 nodes) | `analysis/asymmetry3d/kinematics.py` | `build_kinematics`, `_central_diff` |
| Interior angles | `pose/kinematics.py` | `angle_at_joint`, `compute_joint_angles` |
| OpenSim IK | `opensim_sdk.py` | `run_stablewalk_ik_experimental`, `InverseKinematicsTool` |

#### Formulas / conventions

- Reconstruction frame: `hip_centered_y_up_z_forward`. Image \(x\to X\), image \(y\) down \(\to -Y\). Depth \(Z\) = anatomical layer + scaled MediaPipe \(z\). Height normalized ~1 body height. **Not** SMPL / mocap.
- Analysis frame: `sw_canonical_y_up` (pelvis origin, +X right, +Y up, +Z forward), body-normalized metres. Provenance `inferred_monocular_3d`.
- Interior angle: \(\theta=\arccos(\hat{v}_1\cdot\hat{v}_2)\) in \([0,180]^\circ\); 3D if depth varies, else image plane.
- Velocity / acceleration: forward/backward difference at ends, central difference interior:

\[
\omega_i=\frac{y_{i+1}-y_{i-1}}{2\Delta t}
\]

Visibility gate 0.50; short gaps interpolated; remaining NaN.

#### Tests and numerical results

| Test | Result |
|------|--------|
| Central difference vs analytic sine | unit test passed → **PASS** |
| `normal_gait` reconstruction | shape **(120, 15, 3)**, 120 detected → **PASS** (heuristic 3D) |
| `normal_gait` kinematics | shape **(120, 22, 3)**, `sw_canonical_y_up` → **PASS** (inferred) |
| `abnormal_gait` reconstruction | **(120, 15, 3)**, 120 detected → **PASS** (heuristic) |
| Bone-length / rate sanity | `unrealistic_frac = 1.0` on synthetic **and** both videos → **PARTIAL** |
| OpenSim IK `.mot` | `normal_gait_ik.mot` and `abnormal_gait_ik.mot`: exist, **nRows=120**, time column present. Latest experimental run `run_787305e91e7251c7_1791971901863_ik.mot` also 120 rows. Same-day GUI log: IK completed 120 frames in 449 ms. → **PASS** as experimental IK file production |
| Inverse Dynamics | `ready_for_traditional_id = False` (no ExternalLoads / GRF / COP) → correctly **refused** |

This audit **did not** call `InverseKinematicsTool.run()` again. IK correctness vs a marker lab was **not** scored.

#### Remaining limitations

- Two joint inventories (15 vs 22). Labels agree on Y-up / Z-forward; operators are not proven identical.
- Heuristic Z; left/right and forward/back ambiguity from one camera.
- `unrealistic_frac=1.0` means every frame failed bone-length or rate checks. Demo 3D is not laboratory trajectories.
- Gait-cycle phase sums printed during the dump: DS+SS+flight = 72%, 52%, 70% (not ~100%).
- IK is experimental (mapped MediaPipe markers on a demo Gait2392 model), not a subject-specific ScaleTool study.

---

### 4b. Asymmetry and “asymmetric moments” — PARTIAL

#### What the phrase means in this codebase

Three **kinds** are labeled and not collapsed (`analysis/asymmetry3d/moments.py`):

| Kind | Meaning | This session |
|------|---------|--------------|
| `kinematic_asymmetry` | Statistical L/R comparison (ROM, timing, SI/ASI). **Not torques.** | Populated |
| `estimated_joint_moment` | \(M\approx I\alpha\), Dempster/Winter \(I=mL^2(k^2+d^2)\). Needs mass. Not OpenSim ID. | Not populated (mass not passed) |
| `measured_joint_moment` | Inverse Dynamics with measured GRF/COP/`ExternalLoads.xml` | **Empty.** Gate refuses without force plates |

**Physical joint moments were not invented.**

#### Implementation

| Piece | File | Function |
|-------|------|----------|
| Indices | `analysis/asymmetry3d/indices.py` | `robinson_si`, `absolute_symmetry_index` |
| Scalars | `analysis/asymmetry3d/compare.py` | ROM, stance, clearance, step length |
| Phase alignment | `analysis/asymmetry3d/trajectories.py` | `compare_phase_aligned` (0–100% grid) |
| Report | `analysis/asymmetry3d/report.py` | `analyze_asymmetry3d` |
| ID gate | `analysis/opensim_id_readiness.py` | `assess_opensim_id_readiness` |

#### Formulas

\[
\mathrm{SI}=200\frac{L-R}{L+R},\qquad
\mathrm{ASI}=200\frac{|L-R|}{|L|+|R|}
\]

Phase-aligned RMSE/Pearson only after resampling to a shared gait-phase grid. Unaligned frames are not subtracted.

#### Tests and numerical results

| Test | Result |
|------|--------|
| `_walking_recording` L/R alignment + asymmetric ROM | `test_asymmetry3d` passed → **PASS** (synthetic walking generator) |
| Sine wire 20°/20° knee ASI | **0.000** (GT 0) → **PASS** |
| Sine wire 20°/40° knee ASI | **66.667** (GT 66.667) → **PASS** |
| Phase-aligned knee on sine / demo videos | `aligned=false`, `n_right_cycles=0` → **FAIL** on these inputs (method exists) |
| Measured moments | false on all four → **PASS** (honest refusal) |
| ID ready | false on all four → **PASS** (honest refusal) |

**`normal_gait`** (scalars `reliable=False`; do not read as clinical laterality)

| Metric | L | R | ASI % |
|--------|---|---|-------|
| knee_flexion_rom (deg) | 4.58 | 9.25 | **67.64** |
| Cadence (cycle module) | 64.5 steps/min from 2 cycles | | |

Knee ROM magnitudes are far below typical overground walking. This is **not** treated as a healthy/pathological label. `normal_gait` is **not** more symmetric than `abnormal_gait` on knee ASI (67.6% vs 45.1%).

**`abnormal_gait`:** knee ROM 8.91° / 14.10°, ASI 45.10%; cadence 51.4 from 2 cycles.

Step length: unavailable (session not calibrated). Foot clearance: body-normalized heel Y span, not millimetre lab clearance.

### Remaining limitations (4b)

- No force-plate joint torques.
- Phase-aligned L/R comparison did not activate on the two demo videos or the sine generator.
- Estimated \(M\approx I\alpha\) is unused unless mass is supplied.
- Required inputs for measured moments: scaled OpenSim model, IK kinematics, **measured GRF + COP + ExternalLoads.xml**. Video vGRF is explicitly not a substitute.

---

## Requirement 1 (context only)

**PARTIAL.** `models/joint_registry.py`: 21 `JOINT_IDS` + pelvis; 13 `DOF_IDS`; `WIRE_MODEL_KIND = monocular_wire_skeleton`. Supervisor range 12–20 is met by DOFs, not by joint count. Not reduced in this audit.

---

## Requirement 5

**NOT TESTED / not delivered.** Later-stage RL/DL simulator. Architecture notes elsewhere are not acceptance evidence.

---

## Validation matrix

| Condition | Spectral | Asymmetry | OpenSim ID | Notes |
|-----------|----------|-----------|------------|-------|
| Synthetic regular 1 Hz | ACF period error 0 s; \(f_0\) error 0.023 Hz | Knee ASI 0 | Not ready | `unrealistic_frac=1.0`; 0 right cycles |
| Synthetic asymmetric 20°/40° | L/R RMS delta 14.07 | Knee ASI 66.667 | Not ready | Spatial wire still symmetric |
| `normal_gait.mp4` | Features available; ACF stride **missing** | Knee ASI 67.6%; aligned RMSE **missing** | Not ready; IK `.mot` 120 rows | 4.76 s, 2 cycles |
| `abnormal_gait.mp4` | Features available; ACF stride **missing** | Knee ASI 45.1% | Not ready; IK `.mot` 120 rows | 3.97 s |

No clinically desirable result was forced.

---

## What was implemented in prior work (still present)

CSV `# session_vector` so the 22-key feature vector is exported, not only JSON / per-row records. This session did **not** add GRF torques or a second reconstruction.

## What would be required to claim full 2–4 compliance

1. Longer calibrated recordings (or IMUs) so ACF stride peaks and bilateral gait cycles exist on **real** video.
2. Metric 3D (known scale, non-heuristic Z) before ROM/clearance are treated as biomechanical magnitudes.
3. Force-plate GRF/COP and ExternalLoads before any **measured** joint moment is reported.
4. A single joint inventory shared by reconstruction and analysis.

Until then: StableWalk **implements validated DSP (2–3) and kinematic asymmetry on a monocular wire**, with OpenSim Inverse Dynamics correctly **blocked**. That is a **PARTIAL** close of Erik’s requirements 2–4, not a full research acceptance.
