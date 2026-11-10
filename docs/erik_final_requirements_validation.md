# Erik requirements 2–4 — final validation

**Project:** StableWalk (M.Sc.)  
**Date of this execution:** 2026-10-14  
**Scope:** Spectral characteristics (req. 2), spectral feature extraction (req. 3), dynamic 3D reconstruction and asymmetric moments (req. 4). Gait-event detection is included because cycles feed features and asymmetry.  
**Out of scope:** requirement 5 (reinforcement-learning simulator) — **not implemented**.  
**Rule:** a GUI tab is not evidence. Status is taken from source functions, equations, and **executed tests**. Unavailable quantities stay `None`; they are never written as 0.

Interactive Tk click-through of the Spectral Features tree, live OpenSim ScaleTool, and force-plate Inverse Dynamics were **not** run in this pass. Feature scalars below were produced by `analyze_spectral` / `analyze_gait_cycles` / `analyze_asymmetry3d` on disk pose JSON (the same objects the Motion tabs display).

---

## Verdict

| Requirement | Status | One-line result |
|-------------|--------|-----------------|
| 1 Wire model ~12–20 joints **or** DOFs | **IMPLEMENTED** (existing) | 21 trackable joints + pelvis root; 13 sagittal DOFs. Joints ≠ DOFs. Not reduced to fit. |
| 2 Autocorrelation / R(0) / R(τ) / FFT / Welch / eigen | **IMPLEMENTED** | Unit tests + synthetic 1 Hz recovery. Demo clips: R(0) 6×6 symmetric; stride ACF peak **unavailable** (no positive ρ in 0.60–1.80 s). |
| 3 Feature vectors on Spectral Features tab | **IMPLEMENTED** | 22-key `session_vector` with real numbers or `null`. This pass: do not report a **negative** ACF “stride”. |
| 4 Dynamic 3D, ω, α, L/R, cycle alignment | **IMPLEMENTED** (monocular) | Same `GaitMotionRecording`. Metric step length **off**. Phase-aligned L/R curves **unavailable** on both demos (incomplete bilateral cycles). |
| 4 Asymmetric **joint moments** (N·m) | **PARTIAL** | Statistical trajectory moments yes. \(M\approx I\alpha\) only with entered mass. OpenSim ID **missing** (no GRF / ExternalLoads). |
| Gait events HS / TO | **IMPLEMENTED with limits** | Contact FSM + kinematic supplement. Supplement **blocked** when contact > 85% (aids / shuffle). Toe-off is leave-floor, not mid-swing peak. |
| 5 RL simulator | **MISSING** | `docs/erik_requirement_5_plan.md` only. |

---

## What already existed (not duplicated)

The spectral package, feature extractor, Asymmetry workspace, OpenSim ID gate, and camera-view classifier were already in tree. This pass **did not** add a second FFT, a second 3D reconstructor, or a fake ID solver.

| Area | Canonical implementation |
|------|--------------------------|
| Wire joints / DOFs | `stablewalk/models/joint_registry.py` |
| Monocular 3D | `stablewalk/pose/reconstruction.py`, `models/gait_motion.py` |
| Normalized ACF | `analysis/spectral/autocorrelation.py` |
| R(0), R(τ), L/R xcorr | `analysis/spectral/matrices.py` |
| Eigen of R(0) only | `analysis/spectral/eigen.py` (`NonsymmetricEigError` on lagged R) |
| FFT / Welch | `analysis/spectral/spectrum.py` |
| Missing samples | `analysis/spectral/series.py` (NaN, not zero-fill) |
| Feature vector | `analysis/spectral/features.py` (`SESSION_VECTOR_KEYS`) |
| GUI Spectral Features | `ui/tk/dashboard_spectral_features.py` reads `gui._spectral_report.features` |
| Asymmetry / ω / α | `analysis/asymmetry3d/` |
| OpenSim ID honesty | `analysis/opensim_id_readiness.py` |
| Camera view | `analysis/gait_view_analysis.py` |

### Changes in this pass (gaps only)

1. **Positive ACF peaks only** — `_best_peak_in_band` rejects ρ ≤ 0.05 so a trough is not exported as stride duration.  
2. **Cycle list accepted** — `extract_spectral_features` now takes either `GaitCycleAnalysisResult` or a list of `DetectedGaitCycle` (stride CV was previously dropped on a plain list).  
3. **Kinematic HS/TO** — heel-strike = prominent clearance **minimum** (+ velocity sign change when velocities exist). Toe-off = first **leave-floor** after HS, not the swing-height maximum.  
4. **No kinematic invention under saturated contact** — `kinematic_supplement_allowed` is false when contact fraction > 0.85.  
5. **View-aware prominence** — frontal / unknown cameras use a higher clearance threshold; sagittal slightly lower. Optional `sequence` on `analyze_gait_cycles`.  
6. **Synthetic GT module** — `analysis/synthetic_gait.py` (known \(f_0\), known L/R amplitudes).

---

## Requirement 2 — autocorrelation and spectral analysis

### Mathematics (as coded)

**Normalized autocorrelation** (`normalized_autocorrelation`), longest finite run, mean removed:

\[
\rho(k)=\frac{\sum_t x[t]\,x[t+k]}{\sum_t x[t]^2},\qquad \rho(0)=1
\]

Stride lag is the largest **positive** local maximum of ρ in \([0.60,1.80]\) s with \(\rho>0.05\). Step lag uses \([0.30,0.90]\) s. Missing: `stride_period_s = None`.

**L/R cross-correlation:** pairwise-complete Pearson \(\rho_{LR}(\tau)\).

**Multivariate R(0):** pairwise-complete covariance of the default 6 sagittal DOFs (L/R hip, knee, ankle flexion), optionally z-scored. Pearson form \(D^{-1/2}R_{\mathrm{cov}}D^{-1/2}\).

**Lagged R(τ):** \(R_{ij}(\tau)=\widehat{\mathrm{cov}}(x_i(t),x_j(t+\tau))\). For \(\tau\neq 0\), \(R(\tau)\) is **not** symmetric; it is **not** passed to `eigh`.

**Eigenanalysis:** `numpy.linalg.eigh` only if \(\|M-M^\top\|_\infty\le 10^{-8}\).

**FFT:** one-sided Hann periodogram, density scaling \(1/(f_s\sum w^2)\).

**Welch:** `scipy.signal.welch` if importable, else `welch_psd_numpy` (periodic Hann, 50% overlap, `nperseg=min(256,N)`). This environment: SciPy **not** installed → `numpy_welch`.

**Missing values:** landmarks below visibility stay NaN. Gaps ≤ 0.15 s may be linearly interpolated. Edges are not extrapolated. `(0,0,0)` is not used as a pose.

### Executed tests (this pass)

```
python -m pytest tests/test_wire_model_inventory.py tests/test_spectral_analysis.py
  tests/test_spectral_features.py tests/test_spectral_workspace.py tests/test_asymmetry3d.py
  tests/test_asymmetry3d_moments.py tests/test_asymmetry_workspace.py
  tests/test_scientific_gait_validation.py tests/test_opensim_id_readiness.py
  tests/test_opensim_ik_status.py tests/test_gait_event_sufficiency.py
  tests/test_real_video_gait_validation.py tests/test_erik_req24_validation.py
  tests/test_gait_view_analysis.py -q
```

**Result: 96 passed** (2026-10-14). SciPy identity test is **skipped** unless SciPy is installed (`test_numpy_welch_matches_scipy_when_available`).

Synthetic 1 Hz recording (`_sine_recording` / `SyntheticGaitSpec`):

| Quantity | Ground truth | Measured |
|----------|--------------|----------|
| Dominant frequency | 1.00 Hz | 0.977 Hz (Welch bin) |
| ACF stride lag | 1.00 s | 1.00 s |
| Spectral entropy (periodic) | low vs noise | 0.300 |
| Spectral entropy (irregular) | high | 0.934 |
| R(0) | symmetric | `r0_symmetric: true`, shape 6×6 |

---

## Requirement 3 — feature extraction

**File:** `analysis/spectral/features.py`.  
**GUI:** Motion → **Spectral Features** (`dashboard_spectral_features.py`). Every row is `FeatureRecord.value` or “—” when `available=False`.

| Key | Formula |
|-----|---------|
| `dominant_frequency_hz` | \(\arg\max_{f\in B} P(f)\), \(B=[0.40,4.00]\) Hz |
| `spectral_entropy` | \(H=(-\sum p_i\log p_i)/\log N\in[0,1]\) |
| `harmonic_ratio` | \(P(2f_0)/P(f_0)\) if \(2f_0\in B\) |
| `acf_stride_peak` / `acf_stride_lag_s` | positive ρ peak in the stride window |
| `stride_cv` | \(\mathrm{std}(T_i)/\mathrm{mean}(T_i)\) of detected cycles (≥ 2) |
| `lr_*` | right − left spectral / amplitude / xcorr |

### Demo session vectors (executed 2026-10-14)

Channel: `dof:left_knee_flexion`. Source: `data/output/poses/normal_gait_poses.json`, `abnormal_gait_poses.json`.

| Feature | `normal_gait` | `abnormal_gait` |
|---------|---------------|-----------------|
| Frames / FPS / duration | 120 / 25.0 / 4.76 s | 120 / 29.97 / 3.97 s |
| `dominant_frequency_hz` | **0.417** | **0.499** |
| `spectral_entropy` | 0.695 | 0.800 |
| `spectral_centroid_hz` | 1.017 | 1.249 |
| `harmonic_ratio` | 0.164 | 0.789 |
| `acf_stride_peak` | **null** | **null** |
| `stride_duration_s` | **null** | **null** |
| `stride_cv` | 0.228 (2 HS cycles) | 0.660 (2 HS cycles) |
| `rms_amplitude` | 1.53 deg | 2.71 deg |
| `leading_eigenvalue` (R(0) corr.) | 2.038 | 2.170 |
| `explained_variance_mode1` | 0.340 | 0.362 |
| `lr_freq_delta_hz` | 0.0 | 0.0 |
| `lr_xcorr_lag_s` | −1.16 s | −1.23 s |
| `lr_relative_phase_deg` | **null** (no stride T) | **null** |

**Interpretation limits (required):**

- Both clips are **shorter than 5 s**. Frequency resolution is coarse; 0.42 Hz sits on the **lower edge of B** and must not be read as cadence.  
- ACF stride is **unavailable** because no **positive** peak exists in the walking lag window. That is honest: the left-knee series on these monocular clips is not a clean periodic stride.  
- `stride_cv` uses **event** cycles, not ACF. Two cycles on a 4 s clip is a weak sample.  
- Features are **not** disease labels.

---

## Requirement 4 — dynamic reconstruction and asymmetry

### What is computed

Per-frame hip-centred \((X,Y,Z)\): **+Y up**, body height ~1. Depth \(Z\) is a **heuristic** (anatomical layers + scaled MediaPipe \(z\)). Provenance: `inferred_monocular_3d`. Not mocap, not SMPL, not calibrated metres.

Sagittal angles from `DOF_IDS`. Angular velocity / acceleration: interior central difference

\[
\dot y[i]=\frac{y[i+1]-y[i-1]}{2\Delta t}
\]

NaNs propagate; no interpolation through missing landmarks.

**Asymmetry indices** (`analysis/asymmetry3d/indices.py`):

\[
\mathrm{SI}=200\frac{L-R}{L+R},\qquad
\mathrm{ASI}=200\frac{|L-R|}{|L|+|R|}
\]

Phase alignment resamples each cycle to 0–100% **before** L/R RMSE. If either side lacks complete cycles, `aligned=False` and trajectory amplitudes stay `null`.

### Demo results (same pose files)

| | `normal_gait` | `abnormal_gait` |
|--|---------------|-----------------|
| Estimated view | Oblique (conf. 0.56) | **Frontal** (conf. 0.70) |
| Contact fraction | 0.53 | 0.75 |
| HS (L/R) | 3 / 2 | 3 / 2 |
| TO (L/R) | 2 / 2 | 2 / 1 |
| HS times (s) | 0.04, 0.88, 1.60, 2.88, 3.76 | 0.067, 0.067, 1.00, 2.00, 3.57 |
| Complete cycles | 2 | 2 |
| Cadence (steps/min) | 64.5 | 51.4 |
| Metrics reliable flag | true | true |
| Support accounting warning | DS+SS+flight = 51.9% ≠ 100% | 70.5% ≠ 100% |
| Knee ROM ASI (%) | 67.6 (session ROM, **not** phase-aligned) | 45.1 (same caveat) |
| Step length (m) | **null** (uncalibrated) | **null** |
| Step length (body-norm.) | 0.146 / 0.206 | 0.033 / 0.073 (**unreliable**, frontal) |
| Phase-aligned traj. amp. | all **null** | all **null** |
| OpenSim ID ready | **false** | **false** |

Cadence and ASI on these clips are **monocular estimates on ~4 s of video**. They are not laboratory walking-speed ground truth. `normal_gait.mp4` in the filename is **not** a clinical normal label.

Simultaneous HS at \(t=0.067\) s on both sides of `abnormal_gait` is a detector collision at clip start, not a biomechanical double hit.

### Asymmetric moments — three kinds (not silently equated)

| Kind | Meaning | This session |
|------|---------|--------------|
| `kinematic_asymmetry` | Statistical 1st/2nd moments of trajectories | Computed |
| `estimated_joint_moment` | \(M_i=I_i\alpha_i\) (Winter/Dempster) | **Disabled** — subject mass not entered |
| `measured_joint_moment` | OpenSim ID + measured GRF/COP + ExternalLoads.xml | **Missing** — `ready_for_traditional_id=False` |

Video contact masks and vGRF proxies are **not** passed into Inverse Dynamics.

---

## Gait event detection

**Primary:** contact FSM — heel strike = 0→1, toe-off = 1→0 (`detect_gait_events`).  
**Supplement:** kinematics only if the FSM found fewer than two HS or zero TO **and** contact fraction ≤ 0.85.

| Situation | Behaviour |
|-----------|-----------|
| Clearance prominence too small | No kinematic events (`test_flat_clearance_does_not_fabricate_events`) |
| Walking aid / shuffle, contact > 85% | Kinematic supplement **off**; cadence withheld if partition untrustworthy |
| Occluded foot (`visibility < 0.35`) | Sample ignored |
| Frontal camera | Higher prominence; spatial step length marked unreliable |
| Sagittal camera | Slightly lower prominence (clearance is more meaningful) |

Toe-off is **not** the mid-swing clearance peak (`test_kinematic_events_match_labeled_periodic_minima`).

Multiple camera **views** are classified automatically (`FRONTAL`, `SAGITTAL_LEFT/RIGHT`, `OBLIQUE`). This is one monocular camera per clip, not multi-camera fusion.

---

## Missing / incomplete functionality

| Item | Status |
|------|--------|
| Requirement 5 RL / Isaac | Not implemented |
| Measured joint moments (N·m) | Missing GRF, COP, ExternalLoads, typically unscaled Gait2392 |
| Calibrated metric step length / clearance | Needs explicit session calibration |
| Phase-aligned L/R trajectories on the two demo clips | Incomplete bilateral cycles |
| Positive ACF stride on the two demo clips | Unavailable (signal not periodic enough) |
| SciPy Welch vs numpy identity | Skipped here (SciPy not installed); numpy Welch is operational |
| Optical mocap / force-plate GT for `normal_gait.mp4` / `abnormal_gait.mp4` | Not in this repository |
| Multi-camera triangulation | Not implemented |
| Walking-aid kinematics as valid gait cycles | Intentionally **not** fabricated |

---

## Evidence files

| Artifact | Path |
|----------|------|
| This report | `docs/erik_final_requirements_validation.md` |
| Live clip + synthetic dump | `data/output/erik_req24_audit.json` (from `scripts/erik_req24_audit.py`) |
| Pose inputs | `data/output/poses/normal_gait_poses.json`, `abnormal_gait_poses.json` |
| Videos | `data/demo_videos/normal_gait.mp4`, `abnormal_gait.mp4` |
| Pytest (96 passed) | commands in § Requirement 2 |

The working Tk application is unchanged except for the analysis gaps listed above. Pose estimation algorithms were not modified to “fix” display or cadence.
