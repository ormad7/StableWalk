# Requirement 2 implementation report — spectral characteristics

**Date:** 2026-10-12  
**Scope:** Autocorrelation, lagged cross-correlation, R(0)/R(τ) matrices, FFT/Welch PSD, symmetric eigen-analysis, GUI, CSV/JSON export.  
**Non-goals:** Pose/reconstruction rewrite, OpenSim IK/ID, clinical instability labels, requirement 3/4/5.

---

## Changed / added files

### New analysis package (`stablewalk/analysis/spectral/`)

| File | Role |
|------|------|
| `config.py` | `SpectralAnalysisConfig` (FPS-aware gaps, smoothing, gait band, lags) |
| `channels.py` | Named L/R hip/knee/ankle DOF, pelvis translation/yaw, heel/toe, 3D trajectories |
| `series.py` | Uniform-grid extract, confidence mask, short-gap interp, Savitzky–Golay, detrend |
| `autocorrelation.py` | Normalized ACF ρ(k), stride/step lag peaks, reliability score |
| `matrices.py` | R(0) covariance/correlation, R(τ), lagged L/R cross-correlation |
| `spectrum.py` | rFFT periodogram, numpy Welch (+ SciPy when installed), harmonics, L/R Δf |
| `eigen.py` | `eigh` on symmetric R(0) and Hermitian band-averaged CSD only |
| `report.py` | `analyze_spectral()` orchestrator |
| `export.py` | Multi-section CSV + JSON with parameters, units, disclaimer |
| `__init__.py` | Public API |

### GUI / integration

| File | Change |
|------|--------|
| `ui/tk/dashboard_spectral.py` | Spectral Analysis workspace, async worker, table, export |
| `ui/viewers/spectral_charts.py` | Signal, ACF, PSD, L/R, heatmap, eigenvalue charts + playhead |
| `ui/tk/dashboard_layout.py` | Motion inner notebook: Kinematics \| Spectral Analysis |
| `ui/tk/app.py` | Schedule on gait attach; playhead; joint-selector sync |
| `ui/tk/playback_active_tab.py` | `spectral` playback group on Motion tab |
| `ui/tk/dashboard_page_frame.py` | Motion subtitle mentions spectral analysis |
| `ui/scientific_labels.py` | Derived-tier spectral labels + disclaimer |
| `analysis/__init__.py` | Lazy export of `analyze_spectral` |

### Tests / docs

| File | Change |
|------|--------|
| `tests/test_spectral_analysis.py` | Synthetic periodic / noisy / irregular / short / gapped signals; NumPy/SciPy checks |
| `tests/test_spectral_workspace.py` | Channel mapping + playback group |
| `docs/erik_requirement_2_report.md` | This report |

**Not modified:** pose backends, `reconstruction.py`, OpenSim IK math, gait-cycle timing, vGRF, existing Motion kinematics graphs (they remain on the Kinematics sub-page).

---

## Implemented methods (scientific)

1. **Time series** from `GaitMotionRecording` snapshots (NaN for missing/low-visibility joints; never `Vec3(0,0,0)` fillers). Resample to `1/fps`. Linear fill only for interior gaps ≤ `max_gap_s` (default 0.15 s). No edge extrapolation.
2. **Smoothing:** centered Savitzky–Golay, window ≈ 0.12 s, polyorder 3 — short vs a gait cycle so ~1–3 Hz content remains.
3. **Detrend:** configurable `none` / `mean` / `linear` (default linear) on processed copies; raw is retained.
4. **Normalized ACF:** ρ(k) = ⟨x[t] x[t+k]⟩ / ⟨x²⟩ after mean removal (`np.correlate`). Stride lag searched in 0.60–1.80 s; step lag in 0.30–0.90 s.
5. **Lagged cross-correlation** of L/R pairs; peak = highest ρ at the smallest |lag|.
6. **R(0)** sample covariance and Pearson correlation (symmetric). **R(τ)** lagged cross-covariance (not assumed symmetric).
7. **FFT** one-sided density periodogram; **Welch PSD** (`scipy.signal.welch` if present, else equivalent numpy). Dominant frequency only inside 0.4–4 Hz.
8. **Eigenanalysis:** `np.linalg.eigh` on R(0) covariance/correlation and optional Hermitian gait-band CSD. Lagged R(τ) raises `NonsymmetricEigError`.
9. **Reliability** is a 0–1 data-quality score (peak strength, cycle count, missingness). It is **not** an instability diagnosis.
10. **GUI** (Motion → Spectral Analysis): channel selector, six charts, results table, playhead on the processed signal, CSV/JSON export. Computation runs on a daemon thread.

---

## Test results

```
python -m pytest tests/test_spectral_analysis.py tests/test_spectral_workspace.py tests/test_playback_active_tab.py -q
26 passed
```

Related regression files (`test_joint_motion_analysis_chart.py`, `test_opensim_ik_status.py`, `test_motion_analysis_tabs.py`, `test_gait_cycle_phase.py`): **24 passed**.

`tests/test_dashboard_notebook.py::test_install_dashboard_notebook_creates_visible_tabs` failed in this environment with `TclError: Can't find a usable tk.tcl` (Tk runtime, not spectral math). Other layout tests that skip on `TclError` passed.

Spectral checks include: `np.correlate` ACF identity, SciPy Welch match when SciPy is installed, `np.linalg.eigh` on R(0), refusal of eigh on lagged R(τ), stride recovery on a 1 Hz synthetic walk, insufficient-data warnings, and CSV/JSON export.

---

## Remaining limitations

- Monocular Z and hip-centered joint XYZ are estimated; pelvis translation uses the pose **root trajectory** when a `PoseSequence` is passed, not the zeroed hip-centered pelvis.
- Ankle/heel/toe channels are omitted from the report when landmarks are missing or low-confidence.
- Welch on very short clips is coarse (nperseg capped by length).
- R(0) z-scores mixed units before the multivariate matrix (documented). Univariate ACF/PSD stay in physical units after detrend.
- No claim of laboratory photogrammetry, inverse-dynamics moments, or clinical interpretation.
- Requirement 3 (standalone spectral feature vector export into `advanced/features.py`) and requirement 4 (3D asymmetric moments) are not implemented here.
