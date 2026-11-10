# Erik research acceptance report — StableWalk (requirements 1–4)

**Project:** StableWalk M.Sc.  
**Supervisor:** Erik  
**Audit date:** 2026-10-14  
**Scope:** original requirements 1–4. Requirement 5 (reinforcement / deep-learning gait simulator) is later-stage and is **not** treated as a deliverable here.  
**Rule:** a GUI tab is not evidence. Status is taken from source functions, mathematical definitions, inputs/outputs, and tests **executed in this session**. Full compliance is **not** claimed.

**This session (executed):** NumPy 2.0.0, SciPy 1.14.0. `67 passed` in 22.37 s:

```
tests/test_wire_model_inventory.py
tests/test_spectral_analysis.py
tests/test_spectral_features.py
tests/test_erik_req24_validation.py
tests/test_asymmetry3d.py
tests/test_asymmetry3d_moments.py
tests/test_opensim_id_readiness.py
tests/test_opensim_ik_status.py
```

Independent numerical checks in this session (synthetic 1.25 Hz sinusoid unless noted):

| Check | Result |
|-------|--------|
| rFFT dominant frequency vs 1.25 Hz | error **0 Hz** |
| SciPy Welch dominant frequency vs 1.25 Hz | **1.171875 Hz** (error 0.078125 Hz; nperseg=128 bin) |
| `normalized_autocorrelation` vs `np.correlate` (centered, lag-0 normalized) | max abs error **0** |
| `welch_psd_numpy` vs `scipy.signal.welch` (`scaling="density"`, Hann, nperseg=128) | max relative error **1e-10** |
| Active Welch backend | `scipy.signal.welch` |
| Lagged covariance \(R(\tau)=R(-\tau)^\top\) at \(\tau=10\) | max abs error **0** |
| Spectral entropy of the 1.25 Hz periodogram in 0.4–4 Hz | **0.258336** (matches NumPy Shannon definition in unit test) |

Interactive Tk playback and live OpenSim ScaleTool were **not** re-run in this scientific pass.

---

## Acceptance table

| Requirement | Code classification | Acceptance | Notes |
|-------------|---------------------|------------|-------|
| 1. Wire human gait model (12–20) | **PARTIAL** | **PARTIAL** | Hierarchical wire exists and is used. Trackable joints = **21** (+ pelvis root), above 12–20. Sagittal DOFs = **13**, inside the range. Joints ≠ DOFs. Not OpenSim Gait2392. Reconstruction uses a **15**-joint MediaPipe subset. |
| 2. Spectral characteristics, including autocorrelation matrix | **IMPLEMENTED** | **PASS** | Normalized ACF, \(R(0)\), lagged \(R(\tau)\), L/R cross-correlation, and symmetric eigenanalysis are coded, defined, and numerically validated. Short real clips do **not** yield ACF stride peaks. |
| 3. Spectral feature extraction | **IMPLEMENTED** | **PASS** | FFT/Welch, dominant frequency, entropy, harmonics, band power, periodicity, L/R coordination, and a 22-key session vector (JSON + CSV) are implemented and checked against NumPy/SciPy. Features are **derived DSP scalars**, not disease labels. |
| 4a. Dynamic 3D reconstruction | **PARTIAL** | **PARTIAL** | Per-frame XYZ, interior angles, \(\omega/\alpha\) central differences, missing-data / visibility handling, and experimental OpenSim IK `.mot` exist. Depth is **heuristic**. Bone-length flags are **unrealistic on 100% of frames** for both demo clips and the sine wire. |
| 4b. Asymmetric moments | **PARTIAL** | **PARTIAL** | Kinematic L/R asymmetry (SI / ASI / SI_norm) is implemented. Physical joint **torques are correctly refused** without GRF. Estimated \(M\approx I\alpha\) exists only when mass is supplied (not used on the demo dumps). Phase-aligned L/R cycles were **not available** on the demo videos. |
| 5. RL / DL gait simulator | **MISSING** | **NOT TESTED** | Explicitly out of scope for this audit. |

**Overall requirements 1–4: PARTIAL.** Methods 2 and 3 pass as signal-processing implementations. Methods 1 and 4 do not meet a laboratory biomechanics reading of the original wording.

---

## Status vocabulary

| Label | Meaning |
|-------|---------|
| **IMPLEMENTED** | Function exists, equations match the code, and tests passed this session |
| **PARTIAL** | Core path exists, but a required input, convention, count, or validation is incomplete |
| **MISSING** | Not present |
| **UNVERIFIED** | Code exists; this session did not execute the relevant live or external check |
| **PASS / PARTIAL / FAIL / NOT TESTED** | Acceptance of the *requirement*, not of a GUI screen |

---

## Requirement 1 — Wire human gait model (12–20)

**Classification: PARTIAL**

### What actually works

A named hierarchical wire is the session kinematics source. `JOINT_IDS` has 21 trackable landmarks plus `ROOT_JOINT_ID = "pelvis"`. `DOF_IDS` has 13 sagittal flexion/orientation coordinates. Parent tree is explicit in `JOINT_PARENTS`. Lower-limb landmarks (hip, knee, ankle, heel, toe, foot, both sides) are retained for contact and clearance. Kind is documented as `monocular_wire_skeleton`, not a scaled OpenSim subject model.

### Mapping

| Item | Location |
|------|----------|
| Source modules | `stablewalk/models/joint_registry.py`, `stablewalk/pose/reconstruction.py`, `stablewalk/pose/skeleton_3d.py`, `stablewalk/pose/dof.py`, `stablewalk/adapters/pose_adapter.py` |
| Functions | inventory constants; `build_skeleton_sequence_3d`; `reconstruct_skeleton_3d`; `pose_sequence_to_gait_motion` |
| Mathematical definition | Kinematic tree: pelvis origin; child joints attached by `JOINT_PARENTS`. Sagittal DOFs are interior / flexion angles, not generalized OpenSim coordinates unless an IK `.mot` is loaded separately. |
| Input | MediaPipe-style landmarks (or synthetic snapshots) |
| Output | `GaitMotionRecording` snapshots with 22 position nodes (21 + pelvis) and 13 DOF samples; `SkeletonSequence3D` with **15** joints |
| GUI | Overview / Motion kinematics 3D wire. Tab presence is **not** taken as proof. |
| Tests | `tests/test_wire_model_inventory.py` (passed) |

### Evidence

```
WIRE_TRACKABLE_JOINT_COUNT == 21
WIRE_SAGITTAL_DOF_COUNT == 13
WIRE_MODEL_KIND == "monocular_wire_skeleton"
ROOT_JOINT_ID == "pelvis" and is not in JOINT_IDS
```

`SKELETON_3D_JOINTS` (reconstruction) = 15 names (`nose`, `mid_shoulder`, `mid_hip`, limb chain to ankles). Heel, toe, foot, head, neck, and spine are **not** in that 15-joint array; they appear on the gait-motion wire via the adapter.

### What is still missing

- The supervisor range **12–20** is not met by the 21-joint inventory. The count was **not** reduced in this audit (heel/toe/foot are used for contact).
- Reconstruction (15) and analysis wire (22) are **two inventories**.
- Not a subject-specific OpenSim Gait2392 / Rajagopal model.

### Scientifically uncertain

Whether Erik’s “12–20” meant **joints**, **DOFs**, or **either**. Thirteen sagittal DOFs fit; twenty-one joints do not. The two skeleton lists are consistent enough for a stick figure, not proven identical in every frame.

---

## Requirement 2 — Spectral characteristics, including autocorrelation matrix

**Classification: IMPLEMENTED**  
**Acceptance: PASS** (method). Operational result on the short demo videos: ACF stride peak **unavailable**.

### What actually works

Univariate normalized autocorrelation, pairwise lagged covariance/correlation matrices, left–right cross-correlation, and eigenanalysis of **symmetric** \(R(0)\) only.

### Mapping

| Item | Location |
|------|----------|
| Source modules | `analysis/spectral/autocorrelation.py`, `matrices.py`, `eigen.py`, `spectrum.py`, `series.py`, `report.py`, `channels.py`, `config.py` |
| Functions | `normalized_autocorrelation`, `analyze_autocorrelation`, `lagged_matrix`, `lagged_cross_cov`, `lagged_cross_correlation`, `eigh_symmetric`, `decompose_r0`, `band_averaged_csd`, `analyze_spectral` |
| Input | `GaitMotionRecording` channels (default univariate: left knee flexion; default matrix: 6 sagittal DOFs) |
| Output | ACF \(\rho(k)\), stride/step lag estimates, \(R(0)\) cov/corr, \(R(\tau)\) at configured lags, L/R \(\rho_{LR}(\tau)\), eigenvalues/vectors of \(R(0)\) |
| GUI | Motion → Spectral Analysis (`dashboard_spectral.py` calls `analyze_spectral`) |
| Tests | `test_spectral_analysis.py`, `test_erik_req24_validation.py` (passed, including SciPy Welch) |

### Mathematical definitions (as implemented)

Normalized ACF after mean removal on the longest finite run:

\[
\rho(k)=\frac{\sum_t x[t]\,x[t+k]}{\sum_t x[t]^2},\qquad \rho(0)=1
\]

This is the biased Pearson form matching `numpy.correlate` on the centered series, then lag-0 normalization. **This session: max abs error vs NumPy = 0.**

Lagged covariance:

\[
R_{ij}(\tau)=\mathrm{cov}\big(x_i(t),\,x_j(t+\tau)\big)
\]

For \(\tau\neq 0\), \(R(\tau)\) is **not** symmetric. The code documents and enforces \(R(\tau)=R(-\tau)^\top\) and **refuses** `numpy.linalg.eigh` on lagged matrices (`NonsymmetricEigError`). **This session: transpose residual = 0.**

\(R(0)\) correlation is Pearson \(D^{-1/2} R_{\mathrm{cov}}(0)\,D^{-1/2}\). Eigenanalysis uses `numpy.linalg.eigh` on the symmetrized finite matrix, eigenvalues sorted **descending**.

Time delays: \(\tau\) in samples; \(\tau_s=\tau/f_s\). Negative lags use \(x[-\tau:n]\) vs \(y[:n+\tau]\).

Missing samples: series interpolation is gap-limited (`max_gap_s`); remaining gaps stay NaN. They are **not** zero-filled.

### Validation evidence

| Dataset | ACF \(\rho(0)\) | ACF stride period | ACF reliability | \(R(0)\) shape | \(R(0)\) symmetric |
|---------|-----------------|-------------------|-----------------|---------------|---------------------|
| Synthetic regular gait (1.0 Hz, 8 s, 50 fps) | 1.0 | **1.00 s** (error 0 vs GT) | moderate | 6×6 | True |
| Synthetic asymmetric gait | 1.0 | **1.00 s** | moderate | 6×6 | True |
| `normal_gait` poses (120 frames, 25 fps, 4.76 s) | 1.0 | **None** (0 peaks) | insufficient | 6×6 | True |
| `abnormal_gait` poses (120 frames, 29.97 fps, 3.97 s) | 1.0 | **None** (0 peaks) | insufficient | 6×6 | True |

Video-derived computation: analysis is run on `data/output/poses/{normal,abnormal}_gait_poses.json`, which are the pose time series extracted from `data/demo_videos/{normal,abnormal}_gait.mp4` (both files present). Pose extraction was **not** re-run from pixels in this session.

### What is still missing

- ACF stride-peak estimation on the two demo clips (clips are ~4 s; walking-band lag search finds no peak).
- Multivariate lagged eigenanalysis of \(R(\tau)\) for \(\tau\neq 0\) (correctly omitted; SVD is not implemented as a substitute).
- Cross-correlation of **all** joint pairs in the GUI default (matrices use 6 sagittal DOFs; L/R xcorr uses configured pairs).

### Scientifically uncertain

Whether a ~4 s monocular clip is long enough for a stable walking autocorrelation. The method is validated on 8 s synthetic sinusoids; the demo videos are too short for the ACF stride estimator as configured.

---

## Requirement 3 — Spectral feature extraction

**Classification: IMPLEMENTED**  
**Acceptance: PASS** (extraction pipeline). Several ACF-derived keys are **correctly None** on the demo videos.

### What actually works

From the Welch (or rFFT) PSD and ACF:

- dominant frequency in the walking band 0.4–4 Hz  
- dominant power, gait-band power \(\sum P_i\Delta f\)  
- normalized spectral entropy \(H=(-\sum p_i\log p_i)/\log N\)  
- centroid, bandwidth  
- harmonic ratio \(P(2f_0)/P(f_0)\), stride/harmonic band-power ratio  
- ACF regularity / stride lag / prominence  
- stride CV from detected cycles (nullable)  
- L/R frequency, entropy, amplitude deltas; xcorr lag; relative phase  
- 22-key `session_vector` in JSON and, as of this session, a CSV `# session_vector` section  

### Mapping

| Item | Location |
|------|----------|
| Source | `analysis/spectral/features.py`, `spectrum.py`, `export.py` |
| Functions | `spectral_entropy`, `band_power`, `extract_features` / session vector assembly, `rfft_periodogram`, `welch_psd` / `welch_psd_numpy`, `export_spectral_json`, `export_spectral_csv` |
| Input | `SpectralAnalysisReport` (+ optional gait cycles) |
| Output | `SpectralFeatureSet.records` and `session_vector`; CSV/JSON |
| GUI | Motion → Spectral Features (`dashboard_spectral_features.py`) |
| Tests | `test_spectral_features.py`, `test_erik_req24_validation.py` (passed) |

### Mathematical checks this session

- Spectral entropy matches the NumPy Shannon definition on a Gaussian bump PSD (`test_spectral_entropy_matches_numpy_shannon_definition`).
- Band power matches \(\bigl(\sum P_i\bigr)\Delta f\) (`test_band_power_matches_numpy_bin_sum`).
- Welch NumPy path matches SciPy 1.14.0 to \(10^{-10}\) relative.
- Missing features stay `value=None` / `available=False` (no fabricated zeros).

### Session vectors (actual values)

**Synthetic regular (GT stride 1.0 Hz).** Dominant frequency **0.976562 Hz** (Welch bin; error 0.023 Hz). Spectral entropy **0.300**. ACF stride peak **0.873** at **1.00 s**. `stride_cv` **0**. L/R amplitude delta **0**. L/R xcorr lag **−0.48 s** (~ opposite-phase right knee, as the sine generator uses \(+\pi\)). Harmonic ratio ~**0** (pure fundamental).

**Synthetic asymmetric (L/R knee amp 20°/40°).** Spectral keys match the regular case on the **primary (left knee)** channel. L/R amplitude delta **14.07** (RMS difference). Expected.

**`normal_gait` (video-derived).**

| Key | Value | Available |
|-----|-------|-----------|
| dominant_frequency_hz | 0.417 | yes |
| spectral_entropy | 0.695 | yes |
| harmonic_ratio | 0.164 | yes |
| acf_stride_peak / lag / regularity | null | **no** |
| stride_duration_s | null | **no** |
| stride_cv | 0.228 | yes (from 2 detected cycles) |
| lr_xcorr_lag_s | −1.16 s | yes |
| lr_relative_phase_deg | null | **no** |

**`abnormal_gait` (video-derived).**

| Key | Value | Available |
|-----|-------|-----------|
| dominant_frequency_hz | 0.500 | yes |
| spectral_entropy | 0.800 | yes |
| harmonic_ratio | 0.789 | yes |
| acf_stride_peak / lag / regularity | null | **no** |
| stride_cv | 0.660 | yes |
| lr_xcorr_lag_s | −1.23 s | yes |
| lr_relative_phase_deg | null | **no** |

Dominant frequencies ~0.4–0.5 Hz on the demo clips are **not** asserted to be true stride frequency. They are argmax of a short, noisy, monocular PSD in 0.4–4 Hz.

### Code change in this session

`export_spectral_csv` now writes `# session_vector`. That was the only confirmed missing **export** piece of requirement 3. Tests assert the section is present.

### What is still missing

- Clinical validation of any feature as a pathology marker (not claimed).
- Harmonic structure on a pure sine is degenerate (ratio ~0); that is mathematically consistent, not a gait harmonic model.
- SciPy `entropy` helper is unused; the implementation is the textbook Shannon form.

### Scientifically uncertain

Whether 0.42 Hz on `normal_gait` is stride, step, or drift. The clip is 4.76 s with 2 counted cycles (~0.42 Hz if those cycles were strides). That numerical coincidence does **not** validate the PSD against a metronome or IMU.

---

## Requirement 4 — Dynamic 3D reconstruction and asymmetric moments

Split because the original wording mixes geometry and kinetics.

### 4a. Dynamic 3D reconstruction — PARTIAL

#### What actually works

- Per-frame 3D joint coordinates on two paths:
  - `build_skeleton_sequence_3d`: shape **(120, 15, 3)** on both demo pose files; all 120 frames detected; coordinate label `hip_centered_y_up_z_forward`.
  - `build_kinematics`: shape **(T, 22, 3)** including pelvis; coordinate label `sw_canonical_y_up`; provenance `inferred_monocular_3d`.
- Interior joint angles 0–180° from 3D (fallback 2D) in `pose/kinematics.py` (`angle_at_joint`).
- Velocity / acceleration: first-order edges, central difference interior (`_central_diff`). Unit test vs analytic sine passed.
- Visibility gate `VISIBILITY_MIN = 0.50`; short gaps interpolated; remaining samples NaN.
- OpenSim IK: `normal_gait_ik.mot` and `abnormal_gait_ik.mot` **present** on disk. Readiness still **not** traditional ID (see 4b).

#### Mapping

| Item | Location |
|------|----------|
| Reconstruction | `pose/reconstruction.py` (`ASSUMPTIONS`, `COORDINATE_SYSTEM`) |
| Kinematics reuse | `analysis/asymmetry3d/kinematics.py` |
| Angles | `pose/kinematics.py` |
| IK status | `tests/test_opensim_ik_status.py`; `.mot` under `data/output/opensim/` |

#### Assumptions (from source, not restated as fact)

Single camera; hip-centered; image \(x\to X\), image \(y\) down \(\to -Y\); depth Z = anatomical layer + scaled MediaPipe \(z\) (~0.2); height normalized to ~1 body height per frame. **Not** SMPL, not multi-view triangulation, not metric mocap.

#### Actual reconstruction errors / flags (this session)

| Dataset | missing_fraction | low_confidence_frac | unrealistic_frac | coordinate system |
|---------|------------------|---------------------|------------------|-------------------|
| Synthetic regular | 0.0 | 0.0 | **1.0** | `sw_canonical_y_up` |
| Synthetic asymmetric | 0.0 | 0.0 | **1.0** | `sw_canonical_y_up` |
| `normal_gait` | 0.0 | 0.0 | **1.0** | `sw_canonical_y_up` |
| `abnormal_gait` | 0.0 | 0.0 | **1.0** | `sw_canonical_y_up` |

`unrealistic_frac = 1.0` means **every frame** failed bone-length or angular-rate sanity on this pipeline. The sine generator places joints on a non-anatomical wire, so the synthetic 1.0 is expected. The same flag on **both real clips** is a reconstruction-quality finding, not a GUI issue. Do not treat demo 3D as laboratory trajectories.

Gait-cycle phase sums printed during the dump:

- Double support + single support + flight = **72.00%**, **51.92%**, **70.45%** (not ~100%).

That is a **scientific** event-model inconsistency, not hidden.

#### What is still missing

- Metric, calibrated 3D (force-plate lab / mocap / multi-view).
- Proof that `hip_centered_y_up_z_forward` and `sw_canonical_y_up` are the same operator on every joint (labels agree on Y-up / Z-forward; the 15- vs 22-joint inventories differ).
- Reliable OpenSim IK as a **validated** inverse-kinematics study (`.mot` files exist; this session did not re-solve IK).

#### Scientifically uncertain

Monocular Z, left/right and forward/back ambiguity, and self-occlusion. Joint angles on foreshortened views. Trajectory consistency of heuristic depth.

---

### 4b. Asymmetry and “asymmetric moments” — PARTIAL

#### Investigation: what “asymmetric moments” means in this codebase

The code **distinguishes three kinds** and does not collapse them:

| Kind | Meaning | Status on demo data |
|------|---------|---------------------|
| `kinematic_asymmetry` | Statistical L/R comparison of trajectories and events (means, ROM, SI/ASI). **Not torques.** | Computed |
| `estimated_joint_moment` | Planar \(M\approx I\alpha\) with Dempster/Winter inertia. Requires subject mass. **Not** OpenSim ID. | **Not populated** in this dump (mass not passed into `analyze_asymmetry3d`) |
| `measured_joint_moment` | OpenSim Inverse Dynamics with measured GRF/COP/`ExternalLoads.xml` | **Refused.** `ready_for_traditional_id is False` on all four recordings |

**Physical joint moments were not invented.** Inverse Dynamics remains gated. Video vGRF proxies are documented as **not** ID input.

Formulas for kinematic indices (`analysis/asymmetry3d/indices.py`):

\[
\mathrm{SI}=200\frac{L-R}{L+R},\qquad
\mathrm{ASI}=200\frac{|L-R|}{|L|+|R|},\qquad
\mathrm{SI_{norm}}=\frac{2\min(L,R)}{L+R}
\]

(SI_norm only for non-negative quantities; 1 = symmetry.)

Gait-cycle alignment: left and right cycles are resampled to a 0–100% grid **before** RMSE/Pearson (`trajectories.py`). Unaligned frames are not subtracted. **This is implemented and unit-tested** on `_walking_recording`. **It did not activate** on the sine generator or on the two demo pose files (`n_right_cycles = 0`, `aligned = false`).

#### Mapping

| Item | Location |
|------|----------|
| Compare / ROM / clearance / timing | `analysis/asymmetry3d/compare.py`, `events.py`, `indices.py` |
| Moments | `analysis/asymmetry3d/moments.py` |
| ID gate | `analysis/opensim_id_readiness.py` |
| GUI | Motion → Asymmetry (`dashboard_asymmetry.py`) |
| Tests | `test_asymmetry3d.py`, `test_asymmetry3d_moments.py`, `test_opensim_id_readiness.py` |

#### Actual asymmetry numbers (this session)

All demo/sine scalars below have `reliable = False` in the report (incomplete bilateral cycle sets), even when SI/ASI are numeric. **Do not read them as clinical laterality.**

**Synthetic regular (GT ASI = 0).** Knee ROM L=R = 39.29° (≈ peak-to-peak of 20° sine). ASI **0**. Stride duration L=R = 1.00 s. Stance duration 0.28 s (too short for real walking contact; the sine heel is not a gait event model). Foot clearance L=R = 0.295 body-normalized units. Step length **unavailable** (not calibrated). Knee trajectory **not aligned** (7 left cycles, 0 right).

**Synthetic asymmetric (GT ASI from amplitudes = 66.667%).** Knee ROM 39.29° vs 78.57°. SI **−66.667%**, ASI **66.667%** — matches the amplitude formula exactly. L/R amplitude spectral delta 14.07. Trajectory still **not aligned** (`n_right_cycles = 0`). Stance/clearance remain symmetric because the generator only scales **knee DOF**, not the spatial wire.

**`normal_gait.mp4` → poses (do not treat as “healthy” ground truth).**

| Metric | Left | Right | ASI % | Available |
|--------|------|-------|-------|-----------|
| stride_duration (s) | 1.86 | 2.00 | 7.25 | yes |
| stance_duration (s) | 1.12 | 0.90 | 21.78 | yes |
| knee_flexion_rom (deg) | 4.58 | 9.25 | **67.64** | yes |
| hip_flexion_rom (deg) | 2.85 | 3.24 | 12.98 | yes |
| ankle_flexion_rom (deg) | 16.37 | 15.46 | 5.67 | yes |
| foot_clearance (body-norm. Y span) | 0.102 | 0.109 | 6.76 | yes |
| step_length | — | — | — | **no** |
| phase-aligned knee RMSE | — | — | — | **no** (`n_right_cycles=0`) |

Cadence **64.5 steps/min** from **2** cycles (`metrics_reliable=True` in the cycle module; asymmetry layer still marks scalars unreliable). Knee ROM magnitudes are **far below** typical overground walking (~60°). That is a monocular / clip / view limitation, not a clinical “stiff knee” claim.

**`abnormal_gait.mp4` → poses.**

| Metric | Left | Right | ASI % | Available |
|--------|------|-------|-------|-----------|
| stride_duration (s) | 1.75 | 1.94 | 9.96 | yes |
| stance_duration (s) | 1.02 | 1.53 | 40.52 | yes |
| knee_flexion_rom (deg) | 8.91 | 14.10 | 45.10 | yes |
| hip_flexion_rom (deg) | 3.54 | 5.63 | 45.62 | yes |
| ankle_flexion_rom (deg) | 27.07 | 13.38 | 67.70 | yes |
| foot_clearance | 0.067 | 0.045 | 40.13 | yes |
| step_length | — | — | — | **no** |
| phase-aligned knee RMSE | — | — | — | **no** |

Cadence **51.4 steps/min** from 2 cycles. ACF stride peak unavailable.

OpenSim ID checks (both demo runs): scaled model **partial** (demo `.osim`, not session ScaleTool); subject mass **partial** (70 kg pose estimate); IK `.mot` **present**; ExternalLoads / GRF / COP **missing**. `ready_for_traditional_id = False`.

#### What is still missing

- Measured joint torques (force plates required).
- Phase-aligned L/R comparison on the two demo videos.
- Calibrated foot clearance (mm) and step length.
- Estimated \(M\approx I\alpha\) on the default analysis path (mass not supplied).
- Bilateral gait-event completeness (right HS cycles = 0 on demo + sine generator).

#### Scientifically uncertain

Whether “asymmetric moments” in the original brief meant (i) statistical motion moments, (ii) \(I\alpha\) estimates, or (iii) inverse-dynamics torques. The implementation answers all three **as labeled kinds**. Only (i) is populated on the demo data. (iii) must remain empty without GRF.

The large knee ASI on `normal_gait` (**67.6%**, similar to the synthetic 20°/40° case) is **not** evidence that the clip is pathological, and it is **not** suppressed to look “normal.”

---

## Phase 6 — Validation summary

| Condition | Source | Spectral pipeline | Asymmetry pipeline | OpenSim ID | Errors / availability |
|-----------|--------|-------------------|--------------------|------------|------------------------|
| Synthetic regular gait | `make_sine_gait_recording` 1 Hz, L=R 20° | Available. ACF period error **0 s**. Welch \(f_0\) error **0.023 Hz**. Entropy 0.300 | Knee ASI **0** (GT 0). Trajectory **not** phase-aligned | Not ready | `unrealistic_frac=1.0`; stance 0.28 s not a real contact model |
| Synthetic asymmetric gait | same, L/R 20°/40° | Same primary-channel spectrum; L/R RMS delta 14.07 | Knee ASI **66.667** (GT 66.667) | Not ready | Right cycles = 0; spatial wire still symmetric |
| Real normal gait video | `data/demo_videos/normal_gait.mp4` via `normal_gait_poses.json` | Available; ACF stride peak **missing** | Scalars numeric but `reliable=False`; aligned RMSE **missing** | Not ready; IK `.mot` present | DS+SS+flight ≠ 100%; knee ROM ~5–9°; 120 frames / 4.76 s |
| Real abnormal gait video | `data/demo_videos/abnormal_gait.mp4` via `abnormal_gait_poses.json` | Available; ACF stride peak **missing** | Same pattern; stance ASI 40.5%; ankle ROM ASI 67.7% | Not ready; IK `.mot` present | 120 frames / 3.97 s; cadence 51.4 from 2 cycles |

No clinically desirable result was forced. `normal_gait` is **not** reported as more symmetric than `abnormal_gait` on knee ROM ASI (67.6% vs 45.1% under this pipeline).

Ground-truth **pixel-repose** from the MP4 files was not repeated this session; analysis used the existing pose JSON (the project’s video-derived time series).

---

## Requirement 5 — Reinforcement / deep-learning gait simulator

**MISSING / NOT TESTED.** Later-stage. Not implemented as an Isaac / RL / learned gait controller in this audit. Architecture notes exist in other docs; they are not acceptance evidence.

---

## Tests passed this session (scientific)

| File | Role |
|------|------|
| `test_wire_model_inventory.py` | Joint vs DOF counts |
| `test_spectral_analysis.py` | ACF vs NumPy, Welch vs SciPy, \(R(0)\) eigh, lagged transpose, export |
| `test_spectral_features.py` | Entropy/band-power vs NumPy, session vector, L/R phase on sines, CSV vector |
| `test_erik_req24_validation.py` | Known-frequency recovery; demo poses; ID not ready; synthetic asymmetric ASI |
| `test_asymmetry3d.py` | Central difference, walking-recording alignment, asymmetric ROM detection |
| `test_asymmetry3d_moments.py` | Kind separation; ID refusal without GRF |
| `test_opensim_id_readiness.py` | Gate logic |
| `test_opensim_ik_status.py` | IK status mapping |

**67 passed.** Interactive GUI tests were not part of this scientific pass.

---

## Changes made in this audit (small, tested)

1. **CSV session vector** — `export_spectral_csv` now writes `# session_vector` so the 22-key feature vector is exported, not only per-row records / JSON.
2. **Tests** — \(R(\tau)=R(-\tau)^\top\); spectral entropy and band power vs NumPy; CSV vector assertion; synthetic asymmetric ASI vs `SyntheticGaitSpec.expected_asi_knee_amp`.

No GRF, no fabricated inverse-dynamics torques, no rewrite of pose/IK/playback.

---

## What would be required to claim full compliance

1. A documented 12–20 **or** explicit supervisor acceptance of 21 joints + 13 DOFs, and a single joint inventory shared by reconstruction and analysis.
2. Longer, calibrated recordings (or IMUs) so ACF stride peaks and phase-aligned L/R cycles exist on **real** video, not only on synthetic walking recordings.
3. Metric 3D (known scale, non-heuristic Z) before ROM/clearance are treated as biomechanical magnitudes.
4. Force-plate GRF/COP and ExternalLoads before any **measured** joint moment is reported.
5. Requirement 5 remains future work.

Until those exist, StableWalk **implements DSP and kinematic-asymmetry methods on a monocular wire**, with OpenSim ID correctly **blocked**. That is a **PARTIAL** acceptance of Erik’s original requirements 1–4, not a full research close-out.
