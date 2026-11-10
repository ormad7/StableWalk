# Spectral feature extraction

Derived kinematic descriptors from the requirement-2 spectral pipeline
(`SpectralAnalysisReport`). They are **not** clinically validated disease
indicators.

## Inputs

- Processed joint/DOF time series (uniform FPS grid, NaN for missing landmarks)
- Welch / rFFT PSD confined to a walking band (default 0.4–4 Hz)
- Normalized autocorrelation ρ(k)
- Symmetric R(0) correlation eigenvalues
- Left/right lagged cross-correlation
- Optional detected gait cycles (`DetectedGaitCycle`)
- Optional video-quality score from `assess_video_quality`

Missing inputs → `available=false`, `value=null`. Zeros are never invented.

## Definitions

| Feature | Formula / rule | Units |
|---------|----------------|-------|
| Dominant frequency \(f_0\) | \(\arg\max_{f\in B} P(f)\) | Hz |
| Dominant power | \(P(f_0)\) | units²/Hz |
| Gait-band power | \(\sum_{f\in B} P(f)\,\Delta f\) | units² |
| Spectral entropy | \(H=-\sum p_i\log p_i/\log N\), \(p=P/\sum P\) on \(B\) | 1 |
| Centroid | \(\sum f_i p_i\) | Hz |
| Bandwidth | \(\sqrt{\sum p_i(f_i-c)^2}\) | Hz |
| Harmonic ratio | \(P(2f_0)/P(f_0)\) if \(2f_0\in B\) | 1 |
| Stride/harmonic power ratio | band power \(0.4\)–\(1.5\) Hz / \(1.5\)–\(3\) Hz | 1 |
| ACF stride peak / lag / prominence | max ρ in 0.60–1.80 s | 1, s, 1 |
| Stride duration | ACF stride lag | s |
| Stride CV | \(\mathrm{std}(T_i)/\mathrm{mean}(T_i)\) of detected cycles | 1 |
| Regularity | ρ at stride lag | 1 |
| RMS amplitude | \(\sqrt{\mathrm{mean}(x^2)}\) of processed samples | deg or m |
| Eigenvalues / explained variance | `eigh` of R(0) correlation | 1 |
| L/R Δf, ΔH, ΔRMS | right − left | Hz, 1, units |
| Xcorr lag / relative phase | \(\tau^\star\); \(360^\circ \tau^\star/T\) wrapped ±180° | s, deg |

\(B\) is configurable (`gait_band_hz`).

## Feature vectors

- **Session:** `session_vector` keyed by `SESSION_VECTOR_KEYS`
- **Left / right:** same scalars scoped to each side of the active pair
- **Bilateral:** differences and coordination
- **Cycle:** RMS and a coarse rFFT when the slice has ≥ `min_cycle_samples`

Quality labels (`insufficient` / `low` / `moderate`) are data-quality scores,
not diagnoses.
