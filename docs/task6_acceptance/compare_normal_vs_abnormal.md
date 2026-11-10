# Gait video comparison report

- Video A: `normal_gait` (normal_gait)
- Video B: `abnormal_gait` (abnormal_gait)
- Independent pipelines: **yes**
- Δ definition: B − A
- Exported: 2026-10-15T06:55:09.061656+00:00

## Data quality

- A: 120 frames @ 25 Hz · detected 100%
- B: 120 frames @ 30 Hz · detected 100%
- A cycles: 2 · B cycles: 2

## Numerical differences

| Metric | A | B | Δ (B−A) | Units | Family |
|---|---:|---:|---:|---|---|
| Cadence | 64.52 | 51.87 | -12.64 | steps/min | temporal |
| Stride time | 1.86 | 1.735 | -0.1249 | s | temporal |
| Knee flexion ROM (mean L/R) | 6.913 | 11.52 | 4.606 | deg | kinematic |
| Knee ROM ASI | 67.64 | 44.79 | -22.85 | % | asymmetry |
| Dominant frequency | 0.4167 | 0.4995 | 0.08283 | Hz | spectral |
| Spectral entropy | 0.6947 | 0.7998 | 0.105 | 1 | spectral |
| ACF stride lag | — | — | — | s | autocorrelation |
| ACF regularity | — | — | — | 1 | autocorrelation |
| Pose detection | 100 | 100 | 0 | % | quality |
| Analyzed frames | 120 | 120 | 0 | 1 | quality |
| Frame rate | 25 | 29.97 | 4.97 | Hz | quality |
| Phase-aligned knee RMSE | — | 2.378 | 2.378 | deg | joint_angle |

## Waveforms

- knee_flexion: aligned=True, RMSE=2.378 deg. Each video's cycles resampled to 0–100% independently, then RMSE.
- hip_flexion: aligned=True, RMSE=0.8934 deg. Each video's cycles resampled to 0–100% independently, then RMSE.

## Limitations

- Monocular 3D is body-normalized and heuristic — not laboratory mocap.
- Δ = B − A on derived kinematic/spectral quantities, not clinical scores.
- Unequal FPS or duration is handled by gait-phase resampling when cycles exist.
- Do not label a recording as dementia-related (or any clinical diagnosis) unless catalog metadata includes a verified diagnosis and a reliable source. Gait category tags are research organization only.

## Clinical labeling

Do not label a recording as dementia-related (or any clinical diagnosis) unless catalog metadata includes a verified diagnosis and a reliable source. Gait category tags are research organization only.
