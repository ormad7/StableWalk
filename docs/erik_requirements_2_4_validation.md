# Erik requirements 2–4 — evidence-based scientific validation

**Date:** 2026-10-14  
**Workspace:** `StableWalk/`  
**Auditor role:** gait analysis / DSP / computer vision / biomechanics  
**Verdict:** Requirements 2 and 3 are **implemented and synthetically validated**. Requirement 4 is **partial**. This is **not** scientific completion, clinical validation, or laboratory inverse-dynamics.

Every displayed or exported scalar cited below was produced by executed code on synthetic signals or on pose JSON that exists in this workspace. Missing or unreliable quantities are `None` / `unavailable` / `reliable=false`. They were not replaced with zeros.

**Libraries this run:** NumPy 2.0.0, SciPy 1.14.0 (`scipy.signal.welch`).

**Tests this session:** `104 passed` (`tests/test_dimensional_consistency.py`, `test_joint_motion_analysis_chart.py`, `test_motion_analysis_tabs.py`, `test_spectral_analysis.py`, `test_spectral_features.py`, `test_erik_req24_validation.py`, `test_asymmetry3d.py`, `test_asymmetry3d_moments.py`, `test_opensim_id_readiness.py`, `test_perf_caching_and_lazy_paint.py`, `test_motion_graph_first_paint.py`, `test_motion_analysis_layout.py`). Clip dump: `python scripts/erik_req24_audit.py`.

---

## Incremental defects fixed in this session

| Defect | Evidence | Fix |
|--------|----------|-----|
| Angular jerk was not computed. Linear jerk `jerk_m_s3` is \(d^3\|v\|/dt^3\) in m/s³. Erik asked for angular jerk in angular units per cubic second. | `JointMotionSeries` had no `angular_jerk_deg_s3`; CSV and Motion tabs only labeled “Jerk (m/s³)”. | `angular_jerk_deg_s3 = dα/dt` in `build_joint_motion_series`. Motion tab **Angular jerk (°/s³)**. CSV column `angular_jerk_deg_s3`. Live kinematics cell with unit `°/s³`. |
| PSD / eigen plot labels omitted units; results table quoted covariance while `eigh` is applied to Pearson \(R(0)\). | `spectral_charts.py` ylabel `"PSD"`; `dashboard_spectral.py` string `R(0) = XᵀX/(n−1)`. | Ylabels `PSD (units²/Hz)`, `ρ (normalized)`, `λ (R(0) correlation)`. Table now states biased ACF and “eigh of Pearson \(R(0)\); lagged \(R(τ)\) is not used”. |

Analytic cubic trajectory (\(j_\text{lin}=12\,\mathrm{m/s}^3\), \(j_\text{ang}=48\,\mathrm{deg/s}^3\)): interior-sample means **11.999999999999025 m/s³** and **47.999999999999815 deg/s³**.

---

## What “asymmetric moments” means here

Erik’s wording is ambiguous (biomechanical joint torques vs statistical moments of motion). The implementation distinguishes **three labeled kinds** and does **not** invent force-plate torques.

| Kind | Meaning | When available |
|------|---------|----------------|
| `kinematic_asymmetry` | 1st/2nd **statistical** moments of 3D joint paths (mean XYZ, covariance trace) plus Robinson SI / ASI | Always, from reconstructed kinematics |
| `estimated_joint_moment` | Sagittal proxy \(M \approx I\alpha\) with Dempster/Winter inertia | Only if subject mass (kg) is supplied |
| `measured_joint_moment` | OpenSim Inverse Dynamics with measured GRF + COP + `ExternalLoads.xml` | **Never** on current sessions (`ready_for_traditional_id is False`) |

Source: `stablewalk/analysis/asymmetry3d/moments.py` (`KIND_KINEMATIC`, `KIND_ESTIMATED`, `KIND_MEASURED`).  
Gate: `stablewalk/analysis/opensim_id_readiness.py` — video vGRF / contact masks are **not** passed to `InverseDynamicsTool`.

---

## Requirement 2 — spectral characteristics of the wire gait model

**Overall: PASS** on methods and synthetic ground truth. **PARTIAL** on demo-video periodicity (ACF stride correctly withheld).

### 2.1 Normalized autocorrelation

| Field | Content |
|-------|---------|
| Source | `stablewalk/analysis/spectral/autocorrelation.py` — `normalized_autocorrelation`, `analyze_autocorrelation` |
| Formula | Longest contiguous finite run; demean; \(\rho(k)=\sum_t x[t]x[t+k]/\sum_t x[t]^2\). \(\rho(0)=1\). Stride peak only if a **positive** local max lies in the configured lag window. |
| Numerical tests | `test_normalized_acf_matches_numpy_correlate`: max \(\lvert\rho-\rho_\text{numpy}\rvert = 2.22\times10^{-16}\). Sine 1 Hz, 8 s, 50 Hz: ACF stride lag **1.00 s**. Negative troughs rejected (`test_acf_stride_peak_rejects_negative_rho`). |
| Status | **PASS** |
| Remaining | Biased (not unbiased \(1/(N-k)\)) estimator. Demo clips have **no** ACF stride peak (see condition table). |

### 2.2 Multivariate lagged covariance

| Field | Content |
|-------|---------|
| Source | `stablewalk/analysis/spectral/matrices.py` — `covariance_matrix`, `lagged_matrix`, `lagged_cross_cov` |
| Formula | \(R_{\mathrm{cov},ij}(0)=\frac{1}{n-1}\sum(x_i-\bar x_i)(x_j-\bar x_j)\) pairwise-complete. \(R_{ij}(\tau)=\mathrm{cov}(x_i(t),x_j(t+\tau))\). \(R(\tau)=R(-\tau)^\top\), **not** symmetric for \(\tau\neq 0\). |
| Numerical tests | `test_lagged_matrix_obeys_time_reversal_transpose` (atol \(10^{-10}\)). Demo \(R(0)\) is \(6\times 6\) and symmetric. |
| Status | **PASS** |
| Remaining | Mixed-unit channels are z-scored before the matrix when `zscore_for_matrices=True`. Covariance units are then dimensionless. |

### 2.3 Cross-correlation between joint signals

| Field | Content |
|-------|---------|
| Source | `lagged_cross_correlation`, `build_matrix_bundle` (default L/R name pairs) |
| Formula | Pearson \(\rho_{ij}(\tau)\) on overlapping finite samples. Peak = arg max among bins \(\ge 0.98\max\rho\), smallest \(\lvert\tau\rvert\). |
| Numerical tests | Opposite-phase 1 Hz sines: peak lag **0.50 s** (`test_left_right_cross_corr_peaks_at_half_period`). Feature `lr_relative_phase_deg` ≈ **180°** (`test_left_right_phase_for_opposite_sines`). |
| Status | **PASS** |
| Remaining | Demo `lr_relative_phase_deg` is `None` because ACF stride period is `None` (phase needs \(T_\text{stride}\)). |

### 2.4 Correct matrix normalization

| Field | Content |
|-------|---------|
| Source | `correlation_from_covariance`, `_lagged_corr` |
| Formula | Pearson \(R_\text{corr}(0)=D^{-1/2}R_\text{cov}(0)D^{-1/2}\), \(D=\mathrm{diag}(R_\text{cov})\). Lagged: \(\mathrm{corrcoef}\) of the lagged pair. Diagonal of \(R(0)\) forced to 1. |
| Numerical tests | Identical channels → \(\rho=1\) (`test_identical_channels_unit_correlation`). `test_pearson_r0_has_unit_diagonal`. |
| Status | **PASS** |
| Remaining | Zero-variance channels: off-diagonal NaN, diagonal still 1. |

### 2.5 Eigenvalue analysis (only where appropriate)

| Field | Content |
|-------|---------|
| Source | `stablewalk/analysis/spectral/eigen.py` — `eigh_symmetric`, `decompose_r0`, optional `band_averaged_csd` |
| Formula | `numpy.linalg.eigh` on symmetric \(R(0)\) covariance and Pearson correlation after `0.5(A+A^\top)`. Lagged \(R(\tau)\) raises `NonsymmetricEigError`. Optional Hermitian CSD: \(S=\langle X(f)X(f)^H\rangle\) in the gait band. |
| Numerical tests | `test_r0_is_symmetric_and_eigh_matches_numpy` (atol \(10^{-8}\)). `test_lagged_matrix_is_not_fed_to_eigh`. Demo leading correlation eigenvalues: normal_gait **2.038**, abnormal_gait **2.170**. |
| Status | **PASS** |
| Remaining | NaNs in \(R(0)\) are replaced by 0 before `eigh` (defined operator, not a statistical imputation of the missing pair). |

### 2.6 FFT and Welch PSD

| Field | Content |
|-------|---------|
| Source | `stablewalk/analysis/spectral/spectrum.py` — `rfft_periodogram`, `welch_psd`, `welch_psd_numpy` |
| Formula | One-sided density. Hann; scale \(1/(f_s\sum w^2)\); double non-DC (and non-Nyquist if even). Welch prefers `scipy.signal.welch(..., scaling="density", detrend="constant")`. |
| Numerical tests | 1.25 Hz, \(N=400\), \(f_s=50\): FFT peak within one bin (`test_synthetic_sine_recovers_known_frequency_fft_and_welch`). NumPy Welch vs SciPy: freq error **0**, PSD max rel. error **3.29×10⁻¹⁵**. Runtime method: `scipy.signal.welch`. |
| Status | **PASS** |
| Remaining | Plot ylabel is now `PSD (units²/Hz)` (channel units squared). Absolute PSD scale depends on the processed series units (degrees or body-normalized metres). |

### 2.7 Gait periodicity extraction

| Field | Content |
|-------|---------|
| Source | `analyze_autocorrelation` + `analyze_channel_spectrum` dominant frequency in band \(0.4\)–\(4\) Hz |
| Formula | Stride period = lag of the positive ACF peak in `stride_lag_s`. Step period similarly. Reliability score is a **data-quality** number, not a clinical finding. |
| Numerical tests | Synthetic 1 Hz: ACF lag **1.00 s**, session `dominant_frequency_hz` **0.9765625 Hz** (Welch bin). Irregular noise: entropy **0.934** vs sine **0.300**. Short clip 0.6 s: `stride_period_s is None`, reliability `insufficient`. |
| Status | **PASS** (synthetic). **PARTIAL** (demo videos: ACF stride `None` on normal_gait and abnormal_gait). |
| Remaining | Demo dominant frequencies sit near the **lower band edge** (0.42 Hz, 0.50 Hz) and must not be read as cadence. Cadence, when emitted, comes from heel-strike events, not from this ACF. |

---

## Requirement 3 — spectral feature extraction

**Overall: PASS** for derived features, structured vectors, and export. Displayed values come from processed series. Unreliable features stay `None`.

Session vector keys (`SESSION_VECTOR_KEYS`, order is schema):

`dominant_frequency_hz`, `dominant_power`, `gait_band_power`, `spectral_entropy`, `spectral_centroid_hz`, `spectral_bandwidth_hz`, `harmonic_ratio`, `stride_harmonic_power_ratio`, `acf_stride_peak`, `acf_stride_lag_s`, `acf_peak_prominence`, `stride_duration_s`, `stride_cv`, `regularity_acf`, `rms_amplitude`, `leading_eigenvalue`, `explained_variance_mode1`, `lr_freq_delta_hz`, `lr_entropy_delta`, `lr_amplitude_delta`, `lr_xcorr_lag_s`, `lr_relative_phase_deg`.

Source: `stablewalk/analysis/spectral/features.py` — `extract_spectral_features`.  
Export: `stablewalk/analysis/spectral/export.py` — `export_spectral_json`, `export_spectral_csv` (sections `# parameters`, `# time_series` with units, `# autocorrelation`, `# welch_psd`, `# r0_correlation`, `# eigenvalues`, `# spectral_features`, `# session_vector`).  
GUI: `dashboard_spectral.py`, `dashboard_spectral_features.py`, `ui/viewers/spectral_charts.py`.

| Feature | Formula (code) | Test / number | Status |
|---------|----------------|---------------|--------|
| Dominant frequency | \(f_0=\arg\max_{f\in B} P(f)\), \(B=[0.4,4]\) Hz | Synthetic 1.00 / 1.20 / 1.25 Hz recovered within bin or 0.15 Hz. FPS 30 vs 50 does not shift 1.2 Hz (`test_sampling_rate_does_not_shift_known_frequency`). Athletic 27-frame clip: **`None`**. | **PASS** |
| Spectral entropy | \(H=(-\sum p_i\log p_i)/\log N\), \(p=P/\sum P\) on \(B\) | Matches independent NumPy Shannon (rel \(10^{-12}\)). Sine \(H=0.300\) < noise \(H=0.934\). | **PASS** |
| Spectral centroid | \(\sum f_i p_i\) on \(B\) | Computed on processed PSD (`_centroid`). Synthetic sine session uses same \(P(f)\). | **PASS** |
| Spectral bandwidth | \(\sqrt{\sum p_i(f_i-c)^2}\) | Narrow 1 Hz line: \(H<0.2\), BW \(<0.3\) Hz (`test_entropy_bandwidth_helpers`). | **PASS** |
| Harmonic power | \(P(2f_0)/P(f_0)\) if \(2f_0\in B\); also stride-band / harmonic-band power | Keys `harmonic_ratio`, `stride_harmonic_power_ratio` in session vector. Demo clips emit finite ratios from the **trajectory** PSD, not from a validated harmonic gait model. | **PASS** (definition). Treat demo ratios as descriptive only. |
| ACF peak strength | \(\rho\) at stride lag; prominence \(\rho_\text{peak}-\min\rho_{[0,\text{peak}]}\) | Synthetic sine: peak \(>0.05\), lag 1.00 s. Demo: all ACF stride fields **`None`**. | **PASS** |
| L/R spectral differences | \(\Delta f\), \(\Delta H\), \(\Delta\) RMS, xcorr lag, relative phase \(360^\circ\,\tau/T\) | Synthetic opposite knees: \(\lvert\tau\rvert\approx0.5\) s, phase \(\approx180^\circ\). | **PASS** |
| Structured feature vectors | `FeatureRecord` with `value`, `units`, `available`, `quality`, `definition` | Unavailable → `value is None`, `available=False` (`test_missing_landmarks_do_not_fabricate_amplitude`). | **PASS** |
| CSV / JSON export | JSON schema `stablewalk-spectral-analysis` / `stablewalk-spectral-features`; CSV multi-section | `test_export_json_and_csv`, `test_feature_export_includes_definitions`. Disclaimer: not a clinical diagnosis. | **PASS** |

**Remaining (req 3):** Demo `dominant_frequency_hz` near 0.4 Hz is the band-edge argmax, not stride frequency. `stride_cv` can be filled from heel-strike cycles while ACF stride stays `None` — do not mix the two as one “periodicity” number. GUI results table still does not list every session-vector key (entropy / bandwidth live in the features workspace).

---

## Requirement 4 — dynamic 3D reconstruction and asymmetric moments

**Overall: PARTIAL.** Reconstruction, per-frame kinematics, angles, velocity/acceleration, L/R indices, and cycle resampling exist. Metric spatial gait and measured joint torques do **not**.

### 4.1 3D joint reconstruction

| Field | Content |
|-------|---------|
| Source | `stablewalk/pose/reconstruction.py` — `reconstruct_frame_3d`, `build_skeleton_sequence_3d`; `stablewalk/pose/skeleton_3d.py` — `reconstruct_skeleton_3d`; adapter `pose_adapter.py` → `GaitMotionRecording` |
| Formula / model | Monocular wire skeleton. Hip-centered, Y up, X left–right, Z heuristic (anatomical layer + scaled MediaPipe \(z\)). Height normalized to ~1 body height. **Not** SMPL, **not** multi-view, **not** mocap metres. Assumptions string: `ASSUMPTIONS` in `reconstruction.py`. |
| Numerical tests | `tests/test_asymmetry3d.py`, `tests/test_wire_model_inventory.py`, joint-path scientific tests. Provenance `inferred_monocular_3d`. |
| Status | **PARTIAL** |
| Remaining | Depth is heuristic. Bone-length / rate sanity flags fire on both demo clips (“Frequent unrealistic bone lengths or angular rates”). |

### 4.2 Per-frame joint trajectories

| Field | Content |
|-------|---------|
| Source | `GaitMotionRecording.snapshot_at(i).joints[jid].position`; `analysis/asymmetry3d/kinematics.py` — `build_kinematics`; path math `analysis/joint_path_trace.py` |
| Formula | Positions stored per snapshot. Path length is gap-aware (segments with \(\Delta t>1.5/f_s\) are not connected). Display centimetres are stature-scaled estimates, not laboratory metres. |
| Numerical tests | Joint-path scientific suite (prior session). Kinematic bundle units: `body_normalized_m` unless calibrated. |
| Status | **PASS** (traceability). **PARTIAL** (metric scale). |
| Remaining | No wand / force-plate scale. |

### 4.3 Joint angles

| Field | Content |
|-------|---------|
| Source | Snapshot `angle_deg` / DOF `angle_deg`; Motion Analysis `angle_value_for_item` |
| Formula | Geometric interior angle at the landmark (hip–knee–ankle for knees). **Not** anatomical flexion and **not** an OpenSim model coordinate. Documented in `KNEE_ANGLE_CONVENTION_*`. |
| Numerical tests | `test_motion_angle_convention_marks_raw_knee_interior`. |
| Status | **PASS** as pose geometry. **PARTIAL** vs clinical / OpenSim angles. |
| Remaining | Do not compare these numbers to OpenSim `knee_angle_r` without an explicit mapping. |

### 4.4 Velocity and acceleration

| Field | Content |
|-------|---------|
| Source | `build_joint_motion_series`; `build_kinematics` (`omega_deg_s`, `alpha_deg_s2`) |
| Formula | \(\omega\) from stored DOF velocity or \(d\theta/dt\). \(\alpha=d\omega/dt\). Linear speed \(\lvert\mathbf{v}\rvert\). `accel_m_s2` = \(d\lvert\mathbf{v}\rvert/dt\) (**tangential rate of path speed**, not \(\lVert\mathbf{a}\rVert\)). Linear jerk \(d^2\lvert\mathbf{v}\rvert/dt^2\). Angular jerk \(d\alpha/dt\). |
| Numerical tests | Analytic cubic: linear jerk 12.000 m/s³, angular jerk 48.000 deg/s³; holds at 25 / 50 / 100 Hz (`test_fps_does_not_change_analytic_jerk_units`). |
| Status | **PASS** after this session’s angular-jerk fix. |
| Remaining | `accel_m_s2` is not Euclidean acceleration magnitude. Label is dimensionally correct (m/s²) but the quantity is \(d\lvert v\rvert/dt\). |

### 4.5 Left–right movement symmetry

| Field | Content |
|-------|---------|
| Source | `analysis/asymmetry3d/indices.py`, `compare.py` |
| Formula | \(\mathrm{SI}=200(L-R)/(L+R)\); \(\mathrm{ASI}=200\lvert L-R\rvert/(\lvert L\rvert+\lvert R\rvert)\); \(\mathrm{SI}_\text{norm}=2\min(L,R)/(L+R)\) for non-negative quantities. |
| Numerical tests | Synthetic knees 20° vs 40°: ASI matches expected amplitude ASI within 25 points (`test_synthetic_asymmetric_gait_reports_nonzero_knee_asi`). Robinson helpers unit-tested. |
| Status | **PASS** (indices). |
| Remaining | Demo ROM ASI uses the **whole session** when L/R cycles are incomplete (warning on every ROM row). That is not a phase-aligned bilateral comparison. |

### 4.6 Cycle alignment

| Field | Content |
|-------|---------|
| Source | `analysis/asymmetry3d/trajectories.py` — resample to 0–100 % (`GAIT_CYCLE_SAMPLE_COUNT=101`) before RMSE / Pearson |
| Formula | Pointwise L−R is never taken on raw unaligned frames. |
| Numerical tests | Alignment helpers in `test_asymmetry3d.py`. Demo `traj_amp_*` all **`None` / reliable=false** (incomplete bilateral cycles). |
| Status | **PASS** (algorithm). **PARTIAL** (demo data cannot populate aligned amplitudes). |
| Remaining | Need ≥2 complete cycles **per side** before aligned waveform ASI is emitted. |

### 4.7 Temporal and spatial asymmetry

| Field | Content |
|-------|---------|
| Source | `compare.py` scalars: step/stride/stance/swing duration; step length; foot clearance |
| Formula | Temporal: event intervals. Spatial metric step length **disabled** unless calibrated + suitable view. Body-normalized proxy may still be emitted with a warning. |
| Numerical tests | normal_gait: metric `step_length` **null**; body-normalized step L/R 0.146 / 0.207, ASI 34.1 %. abnormal_gait frontal: body-normalized step marked **`reliable=false`**. |
| Status | **PARTIAL** |
| Remaining | Contact partition on both demos fails \(\mathrm{DS}+\mathrm{SS}+\text{flight}\approx 100\%\) (51.9 % and 70.6 %). Cadence is therefore event-based and should be treated cautiously even when `metrics_reliable` is true. |

### 4.8 OpenSim integration

| Field | Content |
|-------|---------|
| Source | IK status `tests/test_opensim_ik_status.py`; ID audit `opensim_id_readiness.py`; experimental IK `.mot` export under `data/output/opensim/` |
| Formula | Traditional ID requires scaled `.osim`, IK kinematics, subject mass in the model, measured GRF+COP, `ExternalLoads.xml`. |
| Numerical tests | `test_id_readiness_not_ready_without_external_loads`. All three clips this run: `opensim_id_ready: false`. Even if flags were ready, `build_moment_report` **does not execute** `InverseDynamicsTool` (`measured_available` stays False). |
| Status | **PARTIAL** (IK export / status). **FAIL** as “OpenSim inverse-dynamics moments”. |
| Remaining | Do not claim joint torques. Do not feed video GRF proxies into ID. |

### 4.9 Asymmetric moments (resolved)

| Field | Content |
|-------|---------|
| Source | `moments.py` — `kinematic_trajectory_moments`, `proximal_inertia_kg_m2`, `build_moment_report` |
| Formula | Statistical: mean and \(\mathrm{tr}(\mathrm{Cov})\) of XYZ. Estimated: \(I=m L^2(k_\text{com}^2+d_\text{com}^2)\), \(M=I\alpha\) with \(\alpha\) in **rad/s²** (`np.deg2rad`). Measured: refused. |
| Numerical tests | No mass → estimated disabled (`test_moments_disabled_without_mass`). Mass 70 kg, height 1.70 m → \(M\) matches \(I\alpha\) at atol \(10^{-12}\) (`test_estimated_moment_is_I_times_alpha_rad_and_labeled_nm`). Demo dump: estimated disabled (“subject mass is required”). GUI estimated moments labeled `N·m (experimental)` (`dashboard_asymmetry.py`). |
| Status | **PASS** as a labeled three-kind API. **NOT** measured torques. |
| Remaining | \(M\approx I\alpha\) ignores gravity, GRF, and multi-segment coupling. Enable only with recorded mass and treat as experimental. |

---

## Dimensional consistency

| Quantity | Code field | Unit | Plot / live label | Export column | Check |
|----------|------------|------|-------------------|---------------|--------|
| Angle | `angle_deg` | deg | Angle (°) | `angle_deg` | PASS |
| Angular velocity | `omega_deg_s` | deg/s | ω / Angular velocity (°/s) | `angular_velocity_deg_s` | PASS |
| Angular acceleration | `alpha_deg_s2` | deg/s² | α (°/s²) | `angular_acceleration_deg_s2` | PASS |
| Angular jerk | `angular_jerk_deg_s3` | deg/s³ | Angular jerk (°/s³) | `angular_jerk_deg_s3` | **PASS (fixed this session)** |
| Linear speed | `speed_m_s` | m/s | Velocity (m/s) | `linear_speed_m_s` | PASS |
| Linear “accel” | `accel_m_s2` | m/s² | Acceleration (m/s²) | `linear_accel_m_s2` | PASS (unit). Quantity is \(d\lvert v\rvert/dt\). |
| Linear jerk | `jerk_m_s3` | m/s³ | Linear jerk (m/s³) | `jerk_m_s3` | PASS |
| Position | `x_m`,`y_m`,`z_m` | m (body-normalized unless calibrated) | Position (m); path HUD “cm est.” | `x_m`,`y_m`,`z_m` | PARTIAL (scale) |
| Frequency | `dominant_hz` | Hz | Hz | `dominant_frequency_hz` | PASS |
| ACF | `rho` | 1 | ρ (normalized) | `rho_normalized` | PASS |
| PSD | `psd` | units²/Hz | PSD (units²/Hz) | `psd` | PASS (label fixed) |
| Estimated moment | `moment_nm` | N·m | N·m (experimental) | estimated JSON `rms_nm` / `peak_nm` | PASS as \(I\alpha\); not ID |
| Robinson SI / ASI | `si_percent`, `asi_percent` | % | percent | CSV/JSON | PASS |

Occluded frame: missing joint stays NaN, not 0 (`test_occluded_joint_keeps_nan_not_zero`).

---

## Condition tests

| Condition | Stimulus | Result | Status |
|-----------|----------|--------|--------|
| Normal walking | `data/demo_videos/normal_gait.mp4` + `normal_gait_poses.json` | 120 frames, 25 Hz, 4.76 s. Spectral available. ACF stride **None**. Dominant \(f=0.417\) Hz (band edge). \(R(0)\) 6×6 symmetric. Cycles=2, cadence 64.5 steps/min (event-based; DS+SS+flight = 51.9 %). Metric step length **null**. OpenSim ID false. Estimated \(I\alpha\) disabled (no mass). | **PARTIAL** — methods run; periodicity and metric spatials withheld or flagged |
| Asymmetric walking | `SyntheticGaitSpec` L/R knee 20° vs 40° | ASI on knee ROM follows the amplitude difference (`test_synthetic_asymmetric_gait_reports_nonzero_knee_asi`). | **PASS** (synthetic). No dedicated limping pose JSON with usable length was re-processed this run. |
| Assistive device | `abnormal_gait.mp4` + `abnormal_gait_poses.json` (walking-aid clip) | 120 frames, 29.97 Hz, 3.97 s. Spectral available. ACF stride **None**. Dominant \(f=0.499\) Hz. Frontal view: metric and body-normalized step length unreliable. OpenSim ID false. | **PARTIAL** — features from the signal; gait-event partition remains suspect (70.6 %) |
| Different frame rates | Synthetic 1.2 Hz at 30 Hz and 50 Hz; analytic jerk at 25 / 50 / 100 Hz | Dominant frequency stays within 0.2 Hz of 1.2 Hz. Jerk units unchanged. | **PASS** |
| Occluded joints | Dropped landmark / empty snapshot joints | Raw series keep NaN; features with no finite samples stay `available=False`. Motion series XYZ/angle NaN at the hole. | **PASS** |
| Short recordings | 0.5–0.6 s sine; `athletic_walking` 27 frames, 23.98 Hz | ACF stride `None`. Athletic: cycle_count=1, cadence `None`, `event_sufficiency=input_insufficient`, `dominant_frequency_hz=None`, entropy still computed (0.677) from the short spectrum. | **PASS** (honest withhold) |
| Synthetic reference | `_sine_recording`, `known_frequency_vector`, `make_sine_gait_recording` | FFT/Welch/ACF/R(0)/eigh/features/export as above. | **PASS** |

`event_sufficiency="none"` on the two demo clips that have two cycles is a **token hygiene** issue (empty reason). It is not a fabricated cadence.

---

## Requirement scoreboard

| ID | Item | Status |
|----|------|--------|
| 2.1 | Normalized ACF | **PASS** |
| 2.2 | Lagged covariance | **PASS** |
| 2.3 | Joint cross-correlation | **PASS** |
| 2.4 | Pearson normalization | **PASS** |
| 2.5 | Eigenanalysis of \(R(0)\) only | **PASS** |
| 2.6 | FFT / Welch PSD | **PASS** |
| 2.7 | Periodicity | **PASS** synthetic / **PARTIAL** demo |
| 3.1–3.7 | Spectral features + L/R | **PASS** |
| 3.8–3.9 | Vectors + CSV/JSON | **PASS** |
| 4.1 | 3D reconstruction | **PARTIAL** (monocular heuristic) |
| 4.2 | Trajectories | **PASS** / scale **PARTIAL** |
| 4.3 | Joint angles | **PASS** (pose interior) |
| 4.4 | Velocity / acceleration / jerk | **PASS** (angular jerk added) |
| 4.5 | L/R symmetry indices | **PASS** |
| 4.6 | Cycle alignment | **PASS** algorithm / **PARTIAL** demo |
| 4.7 | Temporal / spatial asymmetry | **PARTIAL** |
| 4.8 | OpenSim | **PARTIAL** IK / **FAIL** ID moments |
| 4.9 | Asymmetric moments | **PASS** as three labeled kinds; measured torque **not claimed** |

---

## Remaining work (do not mark complete until done)

1. Longer, calibrated recordings (≥6 stride cycles per side, known \(f_s\), known stature/mass) before claiming ACF stride or metric step length on real video.  
2. Clarify `event_sufficiency="none"` when cycles exist; keep the DS+SS+flight partition failure as a reliability flag.  
3. Optionally plot Euclidean \(\lVert\mathbf{a}\rVert\) as a separate series so `accel_m_s2` is not misread.  
4. Subject mass UI → gated \(I\alpha\) only, still labeled experimental.  
5. OpenSim ID remains blocked until measured GRF/COP and `ExternalLoads.xml` exist. Do not invent them.  
6. Map pose interior knee angle to anatomical flexion / OpenSim coordinates if those are required for comparison.  
7. Requirement 5 (RL / Isaac) is out of scope and unimplemented.

**Scientific completion is not claimed.** Methods for requirements 2–3 are in place and match independent NumPy/SciPy identities on synthetic signals. Requirement 4 reconstructs a monocular wire skeleton and reports statistical (and optional \(I\alpha\)) asymmetry. It does not measure joint torques.
