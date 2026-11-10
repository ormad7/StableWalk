# Erik research compliance — StableWalk requirements 1–4

**Project:** StableWalk M.Sc.  
**Supervisor requirements in scope:** (1) wire gait model, (2) spectral characteristics including autocorrelation matrices, (3) spectral features / extraction, (4) dynamic 3D reconstruction and asymmetric moments.  
**Out of scope:** requirement 5 (reinforcement / deep-learning gait simulator) — **not implemented**.  
**Date of this audit:** 2026-10-13  
**Rule used throughout:** a GUI tab is not evidence of a method. Status is taken from source functions, mathematical definitions, inputs/outputs, and **executed pytest**. An algorithm is marked **IMPLEMENTED** only if it is coded **and** validated in this pass.

Interactive Tk playback, live OpenSim ScaleTool, and force-plate Inverse Dynamics were **not** run in this session.

---

## Verdict

Requirements 1–4 are implemented as **analysis modules** that consume the existing `GaitMotionRecording` (video → MediaPipe → monocular wire). Spectral Analysis, Spectral Features, and Asymmetry display those results; they do not recompute pose.

Measured OpenSim Inverse Dynamics joint moments are **missing** (no force plates / ExternalLoads). Requirement 5 is **not implemented**. Statistical trajectory moments are **not** joint torques.

---

## Status vocabulary

| Label | Meaning |
|-------|---------|
| **IMPLEMENTED** | Function exists, equations match the code, unit tests passed this pass |
| **PARTIAL** | Core path exists but a required input, convention, or validation is incomplete |
| **MISSING** | Not present in this install |
| **UNVERIFIED** | Code exists; this session did not execute the relevant live or external check |

---

## STEP 1 — Compliance matrix

| # | Supervisor item | Files / functions | Mathematics (see later sections) | Inputs | Outputs | GUI (display only) | Tests | Status |
|---|-----------------|-------------------|----------------------------------|--------|---------|--------------------|-------|--------|
| 1 | Wire gait model ~12–20 elements **or** DOFs | `models/joint_registry.py` (`JOINT_IDS`, `DOF_IDS`, `ROOT_JOINT_ID`); `pose/reconstruction.py`; `pose/skeleton_3d.py`; `pose/dof.py` | 21 trackable joints + pelvis root; **13** sagittal DOFs. Joints ≠ DOFs | MediaPipe landmarks, video FPS | Per-frame XYZ + sagittal angles on `GaitMotionRecording` | Overview / Motion **Kinematics** 3D wire | `tests/test_wire_model_inventory.py` | **IMPLEMENTED** as existing model. Joint count **21** is above 12–20; sagittal DOF count **13** is inside the range. Not reduced to fit. Not OpenSim Gait2392 |
| 2a | Normalized autocorrelation | `analysis/spectral/autocorrelation.py` `normalized_autocorrelation`, `analyze_autocorrelation` | Biased Pearson ρ(k) after mean removal | Uniform-grid joint / DOF series | ρ(k), stride/step lag, reliability | Motion → Spectral Analysis, ACF panel | `test_spectral_analysis.py` | **IMPLEMENTED** |
| 2b | L/R cross-correlation | `analysis/spectral/matrices.py` `lagged_cross_correlation` | Pearson ρ_LR(τ) on pairwise-complete samples | Left/right channel pairs | ρ(τ), peak lag, peak value | corr-vs-lag via ACF + L/R PSD; numeric in table / export | `test_left_right_cross_corr_peaks_at_half_period` | **IMPLEMENTED** |
| 2c | Multivariate R(0), R(τ) | `matrices.py` `covariance_matrix`, `lagged_matrix`, `build_matrix_bundle` | Pairwise-complete cov; R(τ)=cov(x_i(t), x_j(t+τ)); R(τ) not symmetric for τ≠0 | Default 6 sagittal DOFs | R(0) cov/corr + lagged matrices | R(0) heatmap | `test_r0_is_symmetric_and_eigh_matches_numpy`, `test_lagged_matrix_is_not_fed_to_eigh` | **IMPLEMENTED** |
| 2d | Eigenanalysis | `analysis/spectral/eigen.py` `eigh_symmetric`, `decompose_r0` | `numpy.linalg.eigh` **only** on symmetric R(0) / Hermitian CSD | R(0) correlation/covariance | λ, v, explained variance | Eigenvalue bar chart | same + `NonsymmetricEigError` | **IMPLEMENTED** |
| 2e | Missing samples, lag, FPS | `analysis/spectral/series.py`, `config.py` | FPS grid; NaN not zero-fill; linear fill only for gaps ≤ `max_gap_s` (0.15 s); lag_s = k / fps | Recording FPS, visibility | Processed series + warnings | Warnings strip | `test_missing_joint_is_nan_not_zero_fill`, `test_sampling_rate_does_not_shift_known_frequency` | **IMPLEMENTED**. Visibility is a **hard mask**, not a weighted covariance |
| 2f | Heatmap + corr-vs-lag plots | `ui/viewers/spectral_charts.py` `draw_correlation_heatmap`, `draw_autocorr` | Display of computed ρ and R(0) | `SpectralAnalysisReport` | Matplotlib panels | Spectral Analysis 3×2 figure | `test_spectral_workspace.py` (wiring) | **IMPLEMENTED** (display). Live Tk **UNVERIFIED** this pass |
| 3a | FFT, Welch PSD, f0, harmonics, energy, L/R | `analysis/spectral/spectrum.py` | rFFT density; numpy Welch (SciPy if installed); f0 in 0.4–4 Hz | Processed series, FPS | Periodogram, PSD, f0, harmonics, Δf_LR | Signal, Welch, L/R PSD | `test_sine_spectrum_dominant_frequency`, `test_noisy_sine_still_finds_frequency` | **IMPLEMENTED**. SciPy identity test **skipped** in this env (`scipy not installed`); numpy Welch is the operational path |
| 3b | Feature extraction | `analysis/spectral/features.py` `extract_spectral_features` | See § STEP 4 | Same report + optional cycles | `SESSION_VECTOR_KEYS` (22 scalars) + per-channel / L/R / cycle records | Motion → **Spectral Features** | `tests/test_spectral_features.py` | **IMPLEMENTED**. Clinical meaning **not assigned** |
| 3c | CSV / JSON export | `analysis/spectral/export.py`; GUI `export_spectral` | Schema `stablewalk-spectral-features` | `SpectralAnalysisReport` | Multi-section CSV + JSON | CSV / JSON buttons on both spectral pages | `test_feature_export_includes_definitions` | **IMPLEMENTED**. Both tabs export the **same** report |
| 4a | Dynamic 3D reconstruction | `pose/reconstruction.py` `build_skeleton_sequence_3d`; `models/gait_motion.py`; `analysis/asymmetry3d/kinematics.py` `build_kinematics` | Hip-centred Y-up wire; per-frame X,Y,Z | Pose sequence | Positions (T,J,3), visibility, angles, ω, α | Overview / Kinematics 3D; Asymmetry uses **same** recording | `test_asymmetry3d.py`, `test_scientific_gait_validation.py` | **IMPLEMENTED** as **monocular inferred 3D**. Heuristic Z. Not mocap / not SMPL |
| 4b | Angular kinematics | `kinematics.py` `_central_diff` | Interior central difference; endpoints forward/back; NaNs propagate | Angle series, dt = 1/fps | ω (°/s), α (°/s²) | Asymmetry ω RMS / peak rows | `test_central_diff_matches_analytic_sine_velocity_and_acceleration` (this pass) | **IMPLEMENTED**. Ankle is interior flexion, **not** a clean dorsi/plantar convention |
| 4c | OpenSim IK | TRC/MOT export, IK tool, `opensim_ik_status.py` | Experimental IK on exported markers | Mapped markers, optional ScaleTool | MOT / status | OpenSim panel | `test_opensim_ik_status.py` | **PARTIAL / gated**. “Complete” only if `ik_state == Completed`. Demo Gait2392 typically Partial without ScaleTool. Live ScaleTool **UNVERIFIED** |
| 4d | L/R asymmetry | `analysis/asymmetry3d/` `compare_bilateral`, `compare_phase_aligned`, `indices.py` | Robinson SI, ASI, SI_norm; 0–100% phase grid | Recording + `analyze_gait_cycles` | Scalars + aligned waveforms | Motion → **Asymmetry** | `tests/test_asymmetry3d.py` | **IMPLEMENTED** for kinematics. Metric step length **gated** on calibration |
| 4e | Asymmetric **moments** | `asymmetry3d/moments.py`; `opensim_id_readiness.py` | **Three kinds** — see STEP 7. Not silently equated | Positions / α / optional mass / ID readiness | Kinematic stats; optional M≈Iα; ID refused without GRF | Asymmetry table + I·α plot only if mass entered | `test_asymmetry3d_moments.py`, `test_opensim_id_readiness.py` | **PARTIAL**. Statistical moments **IMPLEMENTED**. Estimated M≈Iα **IMPLEMENTED** only with mass. Measured ID **MISSING** |
| 5 | RL / Isaac simulator | `docs/erik_requirement_5_plan.md`, stub `analysis/isaac_lab_integration.py` | — | — | — | Must not be demonstrated as done | `test_sim_architecture_spec.py` (availability false) | **MISSING / not implemented** |

### What the GUI actually computes (not evidence by itself)

Motion inner notebook (`dashboard_layout.py`):

| Tab | Computation | Duplicate pipeline? |
|-----|-------------|---------------------|
| Kinematics | Existing 3D skeleton + joint charts from `gait_motion` | No — original pose path |
| Spectral Analysis | Worker: `analyze_spectral(rec, sequence, cycles, univariate_channel_id)` | **No** — single spectral package |
| Spectral Features | `report.features` from that **same** `analyze_spectral` result | **No** |
| Asymmetry | Worker: `analyze_asymmetry3d(rec, cycles, sequence, subject_mass_kg, …)` | **No** second 3D solver (`dashboard_asymmetry.py` docstring: does not create a second 3D viewer) |

Empty states now show **report warnings** when analysis ran but data are insufficient (`insufficient_spectral_copy`, `insufficient_features_copy`, `insufficient_asymmetry_copy`). Missing ROM / stance values are **not** plotted as 0°.

---

## STEP 2 — Autocorrelation matrix

**Source:** `stablewalk/analysis/spectral/autocorrelation.py`, `matrices.py`, `eigen.py`, `series.py`.

### Time series

Signals come from synchronized `GaitMotionRecording` snapshots (video-derived joints / sagittal DOFs). They are placed on a uniform grid at `dt = 1/fps`. Missing or low-visibility samples stay **NaN**. They are **never** replaced with `(0,0,0)`. Interior gaps shorter than `max_gap_s = 0.15` s may be linearly interpolated; edges are not extrapolated. Optional Savitzky–Golay (~0.12 s, polyorder 3) and linear detrend run on the processed copy.

### Normalized autocorrelation

For the longest contiguous finite run \(x[t]\), after mean removal:

\[
\rho(k)=\frac{\sum_t x[t]\,x[t+k]}{\sum_t x[t]^2},\qquad k=0,\ldots,k_{\max}
\]

Implemented with `numpy.correlate` then lag-0 normalization. Stride lag is searched in 0.60–1.80 s; step lag in 0.30–0.90 s. Reliability is a 0–1 **data-quality** score (peak strength, cycle count, missingness) — **not** an instability diagnosis.

### Left–right cross-correlation

For pair \((x_L, x_R)\):

\[
\rho_{LR}(\tau)=\mathrm{corr}\bigl(x_L[t],\,x_R[t+\tau]\bigr)
\]

on pairwise-complete samples (need ≥ 4). Peak lag is the highest ρ at the **smallest** \(|\tau|\) among near-maxima (98 % of max).

### Multivariate lagged covariance

Let \(X(t)\in\mathbb{R}^{C}\) be the default matrix channels (6 sagittal DOFs: L/R hip, knee, ankle flexion; optionally z-scored because units mix).

\[
R_{ij}(0)=\widehat{\mathrm{cov}}\bigl(x_i(t),x_j(t)\bigr)
\quad\text{(pairwise complete, }n-1\text{)}
\]

\[
R_{ij}(\tau)=\widehat{\mathrm{cov}}\bigl(x_i(t),x_j(t+\tau)\bigr)
\]

Default lags: \(\tau\in\{0,\;\mathrm{round}(0.5\,\mathrm{fps}),\;-\mathrm{round}(0.5\,\mathrm{fps}),\;\mathrm{round}(\mathrm{fps})\}\) samples, i.e. 0, ±0.5 s, 1 s. Pearson correlation is \(D^{-1/2}R_{\mathrm{cov}}D^{-1/2}\). **\(R(\tau)\) for \(\tau\neq 0\) is not symmetric** (\(R(\tau)=R(-\tau)^\top\)).

### Eigenanalysis

`eigh_symmetric` requires \(\|M-M^\top\|_\infty \le 10^{-8}\). It then symmetrizes \(M\leftarrow\tfrac12(M+M^\top)\) and calls `numpy.linalg.eigh`. Lagged \(R(\tau)\) raises `NonsymmetricEigError`. Optional gait-band averaged CSD is treated as Hermitian.

### Plots

- Autocorrelation vs lag (`draw_autocorr`) with stride/step markers  
- R(0) Pearson heatmap (`draw_correlation_heatmap`, `vmin=-1`, `vmax=1`)  
- Eigenvalue / explained-variance bars  

---

## STEP 3 — Spectral analysis

**Source:** `stablewalk/analysis/spectral/spectrum.py`.

Walking-band search window \(B=[0.40,\,4.00]\) Hz (`SpectralAnalysisConfig.gait_band_hz`). This is a **search window**, not a claim that energy outside \(B\) is pathology.

### FFT periodogram

One-sided density of the finite (NaN-stripped) demeaned series, Hann window, `numpy.fft.rfft`, scaling \(1/(f_s\sum w^2)\), doubled on non-Nyquist positive bins. Units: \(\mathrm{units}^2/\mathrm{Hz}\).

### Welch PSD

Prefer `scipy.signal.welch` when importable; else `welch_psd_numpy` (periodic Hann, same density scaling). This environment: **SciPy not installed** → operational method is `numpy_welch`. Segment length defaults to `min(256, n)` with 50 % overlap.

### Dominant frequency and harmonics

\[
f_0=\arg\max_{f\in B} P(f)
\]

Harmonic candidates at \(2f_0, 3f_0,\ldots\) if they remain in \(B\). Harmonic ratio \(P(2f_0)/P(f_0)\) at the nearest bin. If gait-band power fraction \(< 0.15\), a warning is attached (dominant peak may be noise/drift).

### Spectral energy

Band power \(G=\sum_{f\in B} P(f)\,\Delta f\) (implemented as bin sum × Δf in features). Left/right comparison reports \(\Delta f_0\) and centroid difference (`compare_left_right_spectra`).

### Noise / short clips

- Irregular synthetic walks get low ACF reliability  
- Duration \(< 2\) s or \(N< 64\) → warnings; stride peak marked insufficient  
- FPS \(< 20\) → aliasing warning  
- Unavailable features export `value=None`, never a fabricated 0  

---

## STEP 4 — Feature extraction

**Source:** `stablewalk/analysis/spectral/features.py`.  
**Orchestrator:** `analyze_spectral` always attaches `extract_spectral_features`.  
**GUI:** Motion → Spectral Features (`dashboard_spectral_features.py`) reads `gui._spectral_report.features`.

Let \(p_i=P_i/\sum P_j\) over positive bins in \(B\).

| Key (`SESSION_VECTOR_KEYS`) | Definition |
|-----------------------------|------------|
| `dominant_frequency_hz` | \(f_0=\arg\max_{f\in B} P(f)\) |
| `dominant_power` | \(P(f_0)\) |
| `gait_band_power` | \(\sum_{B} P_i\Delta f\) |
| `spectral_entropy` | \(H=(-\sum p_i\log p_i)/\log N\in[0,1]\) |
| `spectral_centroid_hz` | \(\sum f_i p_i\) |
| `spectral_bandwidth_hz` | \(\sqrt{\sum p_i(f_i-c)^2}\) |
| `harmonic_ratio` | \(P(2f_0)/P(f_0)\) if \(2f_0\in B\) |
| `stride_harmonic_power_ratio` | band power \([0.4,1.5]\) / \([1.5,3.0]\) Hz |
| `acf_stride_peak`, `acf_stride_lag_s`, `acf_peak_prominence` | ρ and lag of stride ACF peak |
| `stride_duration_s` | that lag (s) |
| `stride_cv` | CV of detected cycle durations (need ≥ 2 cycles) |
| `regularity_acf` | ρ at stride lag |
| `rms_amplitude` | RMS of finite processed samples |
| `leading_eigenvalue`, `explained_variance_mode1` | from symmetric R(0) correlation eigh |
| `lr_freq_delta_hz`, `lr_entropy_delta`, `lr_amplitude_delta` | left–right spectral / amplitude differences |
| `lr_xcorr_lag_s` | peak lag of ρ_LR |
| `lr_relative_phase_deg` | \(\mathrm{wrap}_{180}(360^\circ\cdot\tau_{\mathrm{peak}}/T_{\mathrm{stride}})\) |

Unavailable scalars stay `None` with `available=False` and `quality="insufficient"`.

Export: JSON schema `stablewalk-spectral-features` plus CSV section `spectral_features` (same file as ACF/PSD/R(0)).

These features describe periodicity, regularity, and coordination of **pose time series**. They are **not** clinically validated disease indicators.

---

## STEP 5 — Dynamic 3D reconstruction

**Source:** `pose/reconstruction.py` (`ASSUMPTIONS`), `analysis/asymmetry3d/kinematics.py`.

### What is actually computed

Per-frame joint estimates \( (X,Y,Z) \) on the wire:

- Origin at pelvis (mid-hip); **+Y up, +X right, +Z forward** (`sw_canonical_y_up` / hip-centred).  
- Image \(x\) → skeleton \(X\); image \(y\) (down) → \(-Y\).  
- Depth \(Z\) = anatomical layer offsets + scaled MediaPipe relative \(z\) (~0.2) — **heuristic, not metric depth**.  
- Body height normalized ~1.0 (nose-to-ankle); limbs pulled toward population segment ratios.  
- Visibility / detection flags: joints below `VISIBILITY_MIN` stay NaN.

Provenance labels (`kinematics.py` `infer_coordinate_provenance`):

| Label | Meaning |
|-------|---------|
| `inferred_monocular_3d` | Default. Body-normalized metres. **Not** laboratory XYZ |
| `calibrated_metric` | Only if the session is explicitly marked calibrated |

### Angular quantities

Sagittal DOFs from `DOF_IDS` (degrees). Angular velocity prefers snapshot velocity when present, else `_central_diff`. Acceleration is `_central_diff` of ω.

Interior:

\[
\dot y[i]=\frac{y[i+1]-y[i-1]}{2\,\Delta t}
\]

Endpoints: first-order one-sided differences. A NaN at index \(i\) makes \(\dot y[i-1]\) and \(\dot y[i+1]\) NaN (no interpolation through missing landmarks).

### Visualization / IK

- 3D trajectory display: Overview / Motion Kinematics (existing viewer). Asymmetry plots **phase-aligned** 2-D curves, not a second skeleton.  
- OpenSim IK: export operational; coverage is **gated**. Inverse kinematics is not a substitute for calibrated mocap.

**Do not treat monocular Z or body-normalized metres as physically calibrated measurements.**

---

## STEP 6 — Asymmetry analysis

**Source:** `analysis/asymmetry3d/compare.py`, `trajectories.py`, `events.py`, `indices.py`.  
Events reuse `analyze_gait_cycles` — no parallel event detector.

### Left/right joints

| Quantity | Left series | Right series | Notes |
|----------|-------------|--------------|-------|
| Knee / hip / ankle ROM | P5–P95 of flexion on that side’s cycles | same | If cycles missing, whole-session ROM with a warning |
| Knee / hip / ankle ω RMS and peak | filtered ω on cycles | same | derived |
| Foot clearance | P95−P5 of heel \(Y\) | same | **not** millimetre laboratory clearance |
| Step duration | HS → contralateral HS | same | temporal |
| Stride duration | ipsilateral HS → HS | same | temporal |
| Stance | HS → ipsilateral toe-off | same | temporal |
| Swing | toe-off → next ipsilateral HS | same | temporal |
| Step length (metric m) | contralateral foot ΔZ at HS | same | **None** unless `calibrated_metric` |
| Step length (body-normalized) | same geometry | same | estimated proxy |

### Cycle-normalized trajectories

Left and right cycles are resampled to a common 0–100 % grid (`GAIT_CYCLE_SAMPLE_COUNT = 101`) **before** RMSE / Pearson. Unaligned frames are never subtracted. If either side lacks cycles, `aligned=False` and the GUI shows “Need left and right gait cycles”.

### Indices

\[
\mathrm{SI}=200\frac{L-R}{L+R}\quad\text{(signed percent; 0 = symmetry)}
\]

\[
\mathrm{ASI}=200\frac{|L-R|}{|L|+|R|}\quad\text{(unsigned percent; 0 = symmetry)}
\]

\[
\mathrm{SI_{norm}}=\frac{2\min(L,R)}{L+R}\quad\text{(non-negative only; 1 = symmetry)}
\]

Undefined when the denominator is ~0. **Not a clinical diagnosis.**

Frontal / unknown cameras: spatial step length unreliable. Timing can remain usable when spatial measures are not.

---

## STEP 7 — Asymmetric moments (not silently equated)

Supervisor wording (“asymmetric moments”) can mean **biomechanical joint moments** (torques, N·m) or **statistical moments** of trajectories. The code **distinguishes three kinds** (`moments.py`):

| Kind | What it is | When available | Units |
|------|------------|----------------|-------|
| `kinematic_asymmetry` | 1st/2nd statistical moments of 3D joint paths (mean XYZ, covariance trace, L/R Δ) | Whenever kinematics exist | metres (body-normalized unless calibrated) — **not N·m** |
| `estimated_joint_moment` | Planar rigid-segment \(M_i(t)=I_i\alpha_i(t)\) | Subject **mass (kg)** required; stature defaulted if missing | N·m, **experimental** |
| `measured_joint_moment` | OpenSim Inverse Dynamics with **measured GRF + COP + ExternalLoads.xml** | `assess_opensim_id_readiness` `ready_for_traditional_id` | N·m measured — **unavailable** on typical StableWalk sessions |

### Estimated \(M\approx I\alpha\) (only if mass is entered)

Winter / Dempster sagittal inertia about the proximal joint:

\[
I_i=m_i L_i^2\bigl(k_{\mathrm{com}}^2+d_{\mathrm{com}}^2\bigr),\qquad M_i=I_i\alpha_i
\]

with \(\alpha\) in rad/s². **Ground reaction forces are not used.** Hip-centred monocular \(\alpha\) is noisy. This is **not** OpenSim ID.

### Measured ID — missing data (reported, not fabricated)

Traditional OpenSim ID requires:

1. Subject-specific scaled model (`ScaleTool`)  
2. IK kinematics (`InverseKinematicsTool`)  
3. **Measured** external loads (Fx, Fy, Fz, COP) referenced in `ExternalLoads.xml`  
4. Subject mass in the model  

StableWalk does **not** provide force-plate GRF/COP. Video vGRF proxies and foot-contact masks are **not** passed into `InverseDynamicsTool`. `OPENSIM_ID_REQUIRES_MEASURED_FORCES` is the recorded reason. No physical joint-moment time series is invented.

---

## STEP 8 — Scientific validation (executed this pass)

Command (conda env `stablewalk-opensim`, cwd `StableWalk`):

```
python -m pytest
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
  tests/test_final_scientific_acceptance.py
```

**Result: 115 passed, 1 skipped** (`tests/test_spectral_analysis.py::test_numpy_welch_matches_scipy_when_available` — `scipy not installed`).

| Validation asked | Test | Outcome |
|------------------|------|---------|
| Known sinusoidal frequencies | `test_sine_spectrum_dominant_frequency`, `test_recording_stride_from_sine_knees`, `test_sampling_rate_does_not_shift_known_frequency` | Pass — f0 / stride recovered at 30 and 50 FPS |
| Periodic vs irregular multivariate | `test_irregular_signal_has_low_reliability`; features entropy vs noise | Pass |
| Symmetric vs asymmetric gait-like motion | `test_asymmetry3d.py` (amp mismatch raises ASI); `test_symmetric_paths_have_small_kinematic_moment_delta` | Pass |
| Missing landmarks | `test_missing_joint_is_nan_not_zero_fill`, `test_missing_landmarks_do_not_fabricate_amplitude`, central-diff NaN neighbours | Pass |
| Different video FPS | 30 vs 50 Hz sine features; FPS metadata on demo files | Pass |
| Very short clips | `test_insufficient_clip_warns_and_skips_stride`, `test_short_clip_marks_features_unavailable` | Pass — warnings, no invented stride |
| Numerical derivative accuracy | `test_central_diff_matches_analytic_sine_velocity_and_acceleration` (200 Hz, 1.5 Hz sine; interior ω rtol 3 %, α rtol 8 %) | Pass (added this pass) |
| Demo video vs documented GT | `test_real_video_gait_validation.py` | **Container checks pass.** Pose JSON, when present, must not fabricate cycles. A filename such as `normal_gait.mp4` is **not** clinical ground truth. No optical-mocap reference in this repo |

---

## STEP 9 — GUI integration

| Requirement | Implementation |
|-------------|----------------|
| Spectral Analysis shows current video | `schedule_spectral_analysis` uses `gui.gait_motion` |
| Spectral Features shows current video | Same `_spectral_report`; no second FFT |
| Asymmetry shows current video | `schedule_asymmetry_analysis` uses `gui.gait_motion` + `_gait_cycle` |
| Insufficient-data copy | Empty states include report warnings; table cells use “unavailable” / “—” not 0 |
| No duplicate tabs / pipelines | Four Motion inner pages; one spectral package; one asymmetry3d package; one 3D viewer |

Live GUI click-through of a loaded video was **not** executed in this session (**UNVERIFIED** interactively). Unit wiring tests passed.

---

## STEP 10 — Missing data, limitations, unfinished supervisor items

### Missing data (do not invent)

- Force-plate GRF / COP / `ExternalLoads.xml` → measured joint moments **off**  
- Subject-specific scaled OpenSim model on every session (ScaleTool)  
- Calibrated metric camera for laboratory metres  
- Optical-mocap or marker-based GT for `normal_gait.mp4`  
- SciPy in this conda env (numpy Welch used)  
- Isaac Lab / Gym / MuJoCo / SB3 for requirement 5  

### Known limitations

- 21 joints (plus pelvis) vs supervisor “approximately 12–20 elements”; 13 sagittal DOFs fit the range — **joints are not DOFs**  
- Monocular Z and scale are heuristic; L/R and fore/aft can be ambiguous from one view  
- Ankle angle is geometric interior flexion, not ISB dorsi/plantar  
- Visibility gating is binary, not a weighted covariance  
- Estimated \(M=I\alpha\) ignores GRF, two-joint muscles, and 3-D inertia  
- Spectral “reliability” is DSP quality, not a fall-risk score  
- Composite gait-quality scores elsewhere in the app are labelled composite and must not be read as a validated clinical index  
- Monocular COM vs base-of-support stability remains **unreportable** as a session KPI  

### Unfinished supervisor requirements

| Item | Status |
|------|--------|
| Requirement 5 — reinforcement / deep-learning gait simulator | **Not implemented.** Plan only (`docs/erik_requirement_5_plan.md`). Do not demonstrate as done |
| Measured OpenSim Inverse Dynamics | **Not implemented** pending instrumented kinetics |
| Reduction of the wire to 12–20 **joints** | **Not done** — foot landmarks kept for contact/clearance |
| Calibrated 3-D reconstruction | **Not the default path** |
| Clinical validation of spectral features | **Not claimed** |

---

## This pass — code changes (incremental)

No pose / IK / playback / gait-cycle math was rewritten.

1. `tests/test_asymmetry3d.py` — analytic-sine `_central_diff` test; kinematics ω sanity on known knee flexion.  
2. GUI insufficient-data copy on Spectral Analysis, Spectral Features, and Asymmetry (warnings from the **computed** report).  
3. Asymmetry ROM / stance–swing charts no longer coerce missing values to 0°.  
4. Wiring tests for those empty-state strings.  
5. This document.

---

## Related documents (do not supersede this honesty rule)

- `docs/erik_requirements_1_4_compliance.md` — earlier matrix  
- `docs/erik_requirement_2_report.md` — spectral methods  
- `docs/spectral_features.md`, `docs/asymmetry3d.md`  
- `docs/erik_requirement_5_plan.md` — RL **plan only**
