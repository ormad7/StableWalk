# Erik scientific acceptance — requirements 2–4

**Date:** 2026-10-15  
**Application:** StableWalk (M.Sc. gait-stability research)  
**Task:** 5 of 6  
**Tests this pass:** `77 passed` (`test_spectral_analysis`, `test_spectral_features`, `test_erik_req24_validation`, `test_asymmetry3d`, `test_asymmetry3d_moments`, `test_gait_event_sufficiency`, `test_spectral_workspace`, `test_opensim_ik_status`).  
**Clip dump:** `python scripts/erik_scientific_acceptance_dump.py` → `docs/_erik_acceptance_dump.json`.  
**Figures:** `docs/erik_acceptance/normal_gait_spectral.png`, `docs/erik_acceptance/abnormal_gait_spectral.png`.

**Overall:** Requirements **2 and 3 are implemented and synthetically validated**. Requirement **4 is partial**. This is **not** clinical validation, laboratory inverse-dynamics, or a claim that demo videos are optically calibrated gait.

No second spectral pipeline, reconstruction, or decorative tab was added. Work is in the existing Spectral Analysis / Spectral Features / Asymmetry pages and `stablewalk/analysis/spectral` + `asymmetry3d`.

---

## Requirements-to-code matrix

Status: **PASS** / **PARTIAL** / **FAIL** / **NOT TESTED**.

| Req | Item | Primary source | Status | Evidence this pass |
|-----|------|----------------|--------|--------------------|
| **2** | Spectral characteristics of the wire model | | **PASS** methods; **PARTIAL** demo periodicity | |
| 2.1 | Normalized ACF | `analysis/spectral/autocorrelation.py` | **PASS** | Synthetic 1 Hz: stride lag **1.00 s**. Demo: stride peak **unavailable** (honest). |
| 2.2 | Multivariate lagged matrices | `analysis/spectral/matrices.py` `lagged_matrix` | **PASS** | \(R(\tau)=R(-\tau)^\top\) test. Demo lags **±0.48 s, 1.00 s** (25 Hz). CSV now exports each \(R(\tau)\). |
| 2.3 | L/R cross-correlation | `lagged_cross_correlation` | **PASS** | Synthetic opposite sines: peak **0.50 s**. Demo hip pair \(\rho^\star=0.473\) at \(-1.16\) s. |
| 2.4 | Pearson normalization | `correlation_from_covariance` | **PASS** | Unit diagonal of \(R(0)\). Demo \(R(0)\) is **6×6**. |
| 2.5 | Eigenanalysis | `analysis/spectral/eigen.py` `eigh_symmetric` | **PASS** | Lagged \(R(\tau)\) refused. Demo \(\lambda_1=\) **2.038** (normal), **2.170** (abnormal). |
| 2.6 | FFT / Welch PSD | `analysis/spectral/spectrum.py` | **PASS** | FFT 1.25 Hz error **0**. Welch 1.172 Hz error **0.078 Hz** (`scipy.signal.welch`). |
| 2.7 | Dominant frequency, harmonics, periodicity | spectrum + ACF | **PASS** synthetic / **PARTIAL** demo | Demo \(f_0\) at **band edge** 0.42 / 0.50 Hz — not cadence. Harmonics drawn on PSD. |
| **3** | Spectral feature extraction | `analysis/spectral/features.py` | **PASS** | |
| 3.1 | Spectral entropy | `spectral_entropy` | **PASS** | Matches Shannon (rel \(10^{-12}\)). Demo **0.695** / **0.800**. |
| 3.2 | Spectral centroid | `_centroid` / session vector | **PASS** | Demo **1.017 Hz** / **1.249 Hz**. |
| 3.3 | Spectral bandwidth | `spectral_bandwidth` | **PASS** | Demo **0.746 Hz** / **0.791 Hz**. |
| 3.4 | Dominant frequency | \(\arg\max_B P(f)\) | **PASS** | Synthetic session \(f_0=0.977\) Hz vs 1.00 Hz. |
| 3.5 | Harmonic power | \(P(2f_0)/P(f_0)\); stride/harmonic bands | **PASS** (definition) | Demo ratios **0.164** / **0.789** — descriptive only. |
| 3.6 | ACF peak features | stride lag / prominence / regularity | **PASS** | Synthetic available. Demo **unavailable**. |
| 3.7 | L/R spectral differences | session vector `lr_*` | **PASS** | \(\Delta H\), \(\Delta\) RMS, xcorr lag. Relative phase **unavailable** without ACF \(T\). |
| 3.8 | Feature vectors + export | `export.py`; Features workspace | **PASS** | JSON schema `stablewalk-spectral-analysis`. CSV includes session vector, \(R(\tau)\), L/R xcorr. |
| **4** | Dynamic 3D + asymmetric moments | `analysis/asymmetry3d/` | **PARTIAL** | |
| 4.1 | Per-frame XYZ | `pose/reconstruction.py`, `pose/skeleton_3d.py`, adapter | **PARTIAL** | Provenance `inferred_monocular_3d`, units `body_normalized_m`. Depth heuristic. |
| 4.2 | Coordinate consistency | `coordinates/coordinate_map.py` `sw_canonical_y_up` | **PASS** (frame) / **PARTIAL** (scale) | Hip-centered +Y up. Not lab metres. |
| 4.3 | Temporal continuity | `joint_path_trace.py` gap \(1.5/f_s\); spectral `max_gap_s=0.15` | **PASS** | Missing = NaN, not zero. Demo missing fraction **0**. |
| 4.4 | Video / skeleton / 3D sync | playback + `SessionAnalysisCache` | **PASS** | Same recording timestamps. **NOT TESTED** as a timed GUI capture this pass. |
| 4.5 | Joint angles | pose interior angle | **PASS** as geometry | Not anatomical flexion / not OpenSim `knee_angle_*`. |
| 4.6 | Velocity / acceleration | `build_kinematics`; Motion series | **PASS** | \(\omega\), \(\alpha\); linear jerk \(\lVert d^3\mathbf{r}/dt^3\rVert\). |
| 4.7 | Confidence propagation | landmark visibility; `VISIBILITY_MIN=0.50` | **PARTIAL** | Spectral masks vis &lt; 0.50. Event confidence = foot visibility. No full 3D covariance. |
| 4.8 | OpenSim IK | experimental `.mot` / IK status | **PARTIAL** | IK export can exist. **FAIL** as Inverse Dynamics (no measured GRF). |
| 4.9 | L/R trajectories, ROM, timing, stance/swing | `compare.py`, `trajectories.py` | **PASS** algorithms / **PARTIAL** demo | Cycle alignment needs ≥2 cycles **per side**. |
| 4.10 | Biomechanical vs statistical moments | `moments.py` three kinds | **PASS** API | Measured ID **unavailable**. \(I\alpha\) **disabled** (no subject mass). Statistical moments of XYZ **available**. |

---

## Sampling, lags, missing data (requirement 2)

| Parameter | Value | Where |
|-----------|-------|-------|
| Demo \(f_s\) | **25.00 Hz** (`normal_gait`), **29.97 Hz** (`abnormal_gait`) | pose JSON / recording |
| Duration | **4.76 s** / **3.97 s** | 120 frames each |
| Visibility gate | 0.50 | `SpectralAnalysisConfig.visibility_min` |
| Max interpolable gap | 0.15 s | longer gaps stay NaN |
| Smooth window | 0.12 s, SG poly 3 | processed series (not a second pose solver) |
| Detrend | linear | ACF/PSD of residual |
| Walking band \(B\) | 0.40–4.00 Hz | dominant frequency search |
| ACF stride window | 0.60–1.80 s | positive local max only |
| ACF max lag | 2.50 s | plots |
| Matrix lags | \(0\), \(\pm 0.5\,f_s\), \(+f_s\) samples | demo **0.48 / −0.48 / 1.00 s** at 25 Hz |
| Welch | Hann, density, SciPy when present | method string recorded |
| Missing fill | **never 0** | `test_synthetic_missing_samples_do_not_zero_fill` |

---

## Formulas (as coded)

**Normalized ACF** (biased, demeaned, longest finite run):

\[
\rho(k)=\frac{\sum_t x[t]x[t+k]}{\sum_t x[t]^2},\quad \rho(0)=1.
\]

**Pearson \(R(0)\)**: \(D^{-1/2} R_{\mathrm{cov}}(0) D^{-1/2}\).  
**Lagged covariance**: \(R_{ij}(\tau)=\mathrm{cov}(x_i(t),x_j(t+\tau))\), \(R(\tau)=R(-\tau)^\top\). **Not** passed to `eigh`.

**Welch / rFFT**: one-sided density, Hann, scale \(1/(f_s\sum w^2)\), double non-DC.

**Entropy** on \(B\): \(H=(-\sum p_i\log p_i)/\log N\), \(p=P/\sum P\).

**Centroid / bandwidth**: \(\sum f_i p_i\), \(\sqrt{\sum p_i(f_i-c)^2}\).

**Robinson / ASI**: \(SI=200(L-R)/(L+R)\), \(ASI=200|L-R|/(|L|+|R|)\).

**Estimated moment** (only with mass): \(M=I\alpha\) with \(\alpha\) in rad/s², Dempster/Winter \(I=m L^2(k_{\mathrm{com}}^2+d_{\mathrm{com}}^2)\). **Not** GRF inverse dynamics.

**Measured moment**: OpenSim ID — **refused** without force plates + `ExternalLoads.xml`.

---

## Requirement 2 — plots of computed values

Spectral Analysis (existing 3×2, not a new tab) draws the **actual** processed series, \(\rho(\tau)\), Welch \(P(f)\) with \(f_0\) and \(2f_0,3f_0\) markers, L/R PSD, \(R(0)\) heatmap, and \(\lambda\) of Pearson \(R(0)\).

![normal_gait Spectral Analysis](erik_acceptance/normal_gait_spectral.png)

*Figure: `normal_gait` — 25 Hz, 4.76 s. \(f_0=0.42\) Hz (band edge). ACF has a lobe near 2.0 s, **outside** the 0.60–1.80 s stride window, so stride period stays unavailable. \(\lambda_1\approx 2.04\).*

![abnormal_gait Spectral Analysis](erik_acceptance/abnormal_gait_spectral.png)

*Figure: `abnormal_gait` — 29.97 Hz, 3.97 s. \(f_0=0.50\) Hz. Right-knee PSD energy ≫ left (display of a computed L/R difference, not a diagnosis).*

The Numerical results table now also lists entropy, bandwidth, missing/interpolated fractions, lagged \(\tau\), and L/R xcorr peak from the same report.

---

## Requirement 3 — feature vectors (executed)

Session vector keys are `SESSION_VECTOR_KEYS` in `features.py`. Unavailable fields are JSON `null`, never 0.

| Key | Synthetic 1 Hz, 8 s, 50 Hz | `normal_gait` | `abnormal_gait` |
|-----|---------------------------:|--------------:|----------------:|
| `dominant_frequency_hz` | 0.977 | 0.417 | 0.499 |
| `spectral_entropy` | 0.300 | 0.695 | 0.800 |
| `spectral_centroid_hz` | (session) | 1.017 | 1.249 |
| `spectral_bandwidth_hz` | (session) | 0.746 | 0.791 |
| `harmonic_ratio` | (session) | 0.164 | 0.789 |
| `acf_stride_lag_s` | **1.00** | **unavailable** | **unavailable** |
| `stride_cv` (from HS cycles) | — | 0.228 | 0.653 |
| `leading_eigenvalue` | — | 2.038 | 2.170 |
| `lr_xcorr_lag_s` | ~0.5 (opposite knees) | −1.16 | −1.23 |
| `lr_relative_phase_deg` | ~180° when \(T\) known | **unavailable** | **unavailable** |

Do **not** mix `stride_cv` (heel-strike cycles) with ACF stride: they are different estimators. Demo \(f_0\) near 0.4 Hz is the walking-band argmax of a short residual, **not** step rate (event cadence 64.5 / 51.9 steps/min).

Export: Motion → Spectral Analysis **CSV/JSON**; Features workspace table + export.

---

## Requirement 4 — 3D reconstruction and moments

### Reconstruction

| Fact | Value |
|------|--------|
| Solver | Monocular wire (`reconstruct_skeleton_3d` / MediaPipe → `GaitMotionRecording`) |
| Provenance (both demos) | `inferred_monocular_3d` |
| Units | `body_normalized_m` (~1 body height) |
| Depth | Anatomical layers + scaled MediaPipe \(z\) — **not physically measured** |
| Sync | Snapshot `time_s` shared by skeleton, 3D path, spectral series, events |

Assumptions string: `stablewalk/pose/reconstruction.py` `ASSUMPTIONS`.

### Asymmetry (demo)

Algorithms: Robinson SI, ASI, SI_norm; phase resampling to 0–100% (`GAIT_CYCLE_SAMPLE_COUNT=101`) **before** waveform RMSE.  
Demo clips have only **2** same-side cycles total — aligned L/R waveforms are often **unreliable / None**. Metric step length is **null** unless calibrated.

### Moments — three labeled kinds

| Kind | Demo result |
|------|-------------|
| `kinematic_asymmetry` (mean XYZ, cov trace, SI/ASI) | **Available** from kinematics |
| `estimated_joint_moment` \(I\alpha\) | **Unavailable** — subject mass not supplied |
| `measured_joint_moment` OpenSim ID | **Unavailable** — `opensim_id_ready: false`; no GRF/COP/`ExternalLoads.xml` |

Video vGRF proxies are **not** passed to `InverseDynamicsTool`.

---

## Numerical validation errors (synthetic ground truth)

| Test | Truth | Estimate | Error |
|------|------:|---------:|------:|
| rFFT peak, 1.25 Hz, \(N=400\), \(f_s=50\) | 1.25 Hz | 1.25 Hz | **0 Hz** |
| Welch peak (same) | 1.25 Hz | 1.171875 Hz | **0.078 Hz** (one Welch bin) |
| ACF stride, 1.00 Hz, 8 s | 1.00 s | 1.00 s | **0 s** |
| Knee ASI, amplitudes 20° vs 40° | \(200\cdot20/60=66.667\%\) | 66.667% | **0** |
| ACF vs `numpy.correlate` | — | — | max \(\lvert\Delta\rho\rvert=2.22\times10^{-16}\) (unit test) |
| Analytic cubic jerk (Task 4) | 12 m/s³, 48 deg/s³ | interior SG | within test rtol |

---

## Condition tests (provided videos)

| Condition | Result | Status |
|-----------|--------|--------|
| `normal_gait.mp4` | Spectral **available**. ACF stride **None**. \(f_0=0.417\) Hz. \(R(0)\) 6×6. Cycles=2, cadence 64.5 spm. ID false. \(I\alpha\) false. | **PARTIAL** |
| `abnormal_gait.mp4` (walking aid) | Spectral **available**. ACF stride **None**. \(f_0=0.499\) Hz. Cadence 51.9 spm. Strong L/R PSD imbalance on the plot. ID false. | **PARTIAL** |
| Known-frequency sine | FFT/Welch/ACF recover \(f\) and \(T\) | **PASS** |
| Known-asymmetry sine knees | ASI matches amplitude formula | **PASS** |
| Occlusion / missing samples | NaN retained; features `available=false` | **PASS** |
| Short clip / athletic 27 frames | ACF and cadence withheld | **PASS** (honest) |

`event_sufficiency="none"` on the two-cycle demos means **no sufficiency failure**, not High confidence. Reason string: *“2 complete same-side cycle(s) — no event-sufficiency failure. Measurement reliability is a separate flag.”*

---

## Scientific limitations

1. Monocular depth is heuristic. Spatial metres are **not** laboratory scale.  
2. Demo recordings are too short for a reliable ACF stride (lobe ~2 s sits outside 0.60–1.80 s).  
3. Dominant frequency at the 0.4 Hz band edge is **not** cadence.  
4. Pose knee angle is a geometric interior angle, not OpenSim flexion.  
5. Contact partition DS+SS+flight ≠ 100% on both demos (51.9% / 70.6%) — stance **percent** is internally inconsistent even when HS cadence is emitted.  
6. No force plates → **no measured joint torques**. \(I\alpha\) is a planar proxy and is off without mass.  
7. Spectral features are **not** disease classifiers.  
8. GUI video/skeleton/3D playhead sync is implemented; this pass did **not** capture a timed GUI movie (**NOT TESTED** as an instrumented timing trial).

---

## Remaining tasks (do not mark scientifically complete)

1. Longer calibrated walks (≥6 strides per side, known stature/mass, known \(f_s\)) before claiming ACF stride or metric step length on real video.  
2. Subject-mass input if experimental \(I\alpha\) is to be shown, still labeled experimental.  
3. OpenSim ID stays blocked until measured GRF/COP and `ExternalLoads.xml` exist. Do not invent them.  
4. Optional map of pose interior knee angle → anatomical flexion / OpenSim coordinates.  
5. Instrumented GUI sync timing test (frame-accurate).  
6. **Requirement 5 (RL / Isaac)** is out of scope for Task 5 and unimplemented.

**Full scientific / clinical compliance is not claimed.** Requirements 2–3 exist as coded DSP on pose time series and match independent NumPy/SciPy identities on synthetic signals. Requirement 4 reconstructs a monocular wire skeleton and reports statistical (not force-plate) asymmetry.
