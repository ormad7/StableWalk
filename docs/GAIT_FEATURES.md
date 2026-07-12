# Normalized gait features

See `stablewalk/analysis/gait_feature_analysis.py` for the implementation.

## Feature normalization tiers

| Tier | Meaning | Examples |
|------|---------|----------|
| **RAW** | Absolute units (m, s, °) | `step_length_m`, `pelvis_mediolateral_range_m` |
| **BODY_NORMALIZED** | Divided by robust median segment size | `normalized_step_length` (= step / avg leg length) |
| **GAIT_CYCLE_NORMALIZED** | Resampled to 0–100% gait cycle (101 samples) | Cycle knee trajectories, RMSE, repeatability |

## Body segment dimensions

Estimated from **multiple high-confidence frames** (median, not single-frame):

- Hip width, shoulder width
- Left/right leg, thigh, shank length
- Average leg length

## GUI

Video panel knee chart: switch **Time** ↔ **Gait Cycle %**.

- **Time** — knee angles vs seconds
- **Gait Cycle %** — mean cycle trajectory + optional ±1 SD envelope (≥3 cycles)

## Stability integration

Stability v2 uses:

- BODY_NORMALIZED spatial features for pelvis/trunk/step domains
- GAIT_CYCLE_NORMALIZED metrics for cycle consistency domain
- OpenSim IK joint angles when `{run}_ik.mot` exists (otherwise pose-derived, labeled explicitly)
