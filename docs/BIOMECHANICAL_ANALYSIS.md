# Biomechanical Analysis

StableWalk's advanced biomechanical analysis extends the pose → gait → contact pipeline with clinically meaningful **estimated** and **derived** metrics. It does not replace existing stability scoring, gait-cycle detection, contact masks, virtual GRF, OpenSim export, or Real-to-Sim retargeting.

## Terminology

| Label | Meaning |
|-------|---------|
| **Measured** | Instrumented data (force plates, IMU, mocap). Not available from monocular video alone. |
| **Estimated** | Pose-derived proxy using anthropometric or heuristic models (COM, BoS, ROM). |
| **Derived** | Computed from estimated inputs (symmetry index, stability margin, gait quality score). |

## Pipeline position

```
Video → Pose (MediaPipe/SMPL) → Gait cycles → Foot contact → Biomechanical analysis → Exports + GUI
```

Entry point: `stablewalk.analysis.biomechanical.run_biomechanical_analysis()`

## 1. Center of Mass (COM)

**Method:** Segment-weighted whole-body COM (de Leva-style mass fractions).

For segment \(s\) with mass fraction \(m_s\) and COM position \(\mathbf{r}_s\):

\[
\mathbf{r}_{COM} = \frac{\sum_s m_s \mathbf{r}_s}{\sum_s m_s}
\]

Segment COMs are placed along each segment at literature COM fractions (e.g. thigh 43.3% from hip).

**Velocity / acceleration:** Finite differences on \(\mathbf{r}_{COM}(t)\) using actual timestamps.

**Confidence:** Fraction of segments visible × mean segment confidence (capped 0–1).

**Export:** `center_of_mass.npz` — `com_x/y/z`, `com_v*`, `com_a*`, `timestamps`, `frame_indices`, `confidence`.

**Kind:** `estimated`

## 2. Base of Support (BoS)

**Method:** Horizontal-plane (X–Z) convex hull of foot landmarks (heel, ankle, toe) when `contact_binary = 1`.

| Phase | Polygon |
|-------|---------|
| Left stance | Left foot points |
| Right stance | Right foot points |
| Double support | Hull of both feet |
| Swing | Empty / low confidence |

**Export:** `base_of_support.npz` — `polygon_vertices`, `support_type`, `centroid_x/z`, `area_m2`.

**Kind:** `estimated`

## 3. Stability Margin

**Method:** Project COM onto horizontal plane; compute signed distance to BoS polygon edge.

\[
d_{margin} = \begin{cases}
+d & \text{COM inside BoS} \\
-d & \text{COM outside BoS}
\end{cases}
\]

**States:**

| State | Condition |
|-------|-----------|
| Stable | \(d_{margin} \ge 0.04\) m |
| Reduced Stability | \(0 \le d_{margin} < 0.04\) m |
| Unstable | \(d_{margin} < 0\) m |

Stored in COM NPZ as `stability_margin_m`, `stability_state`, `stability_confidence`.

**Kind:** `derived`

## 4. Symmetry

Uses existing `symmetry_ratio()` and `symmetry_index()` from gait analysis:

\[
SI = \frac{2 \min(L,R)}{L + R + \epsilon}
\]

Metrics: step/stride length, stance/swing duration, cadence consistency, knee/hip/ankle ROM symmetry.

**Overall symmetry %:** Weighted mean of available indices × 100.

**Kind:** `derived`

## 5. Joint ROM

Per-side min, max, ROM, mean, std from pose joint angles (or OpenSim IK DOFs when available).

**Kind:** `estimated` (pose) or `measured` (OpenSim IK from calibrated model)

## 6. Advanced Gait Metrics

Reuses gait-cycle temporal metrics plus pelvis horizontal speed:

| Metric | Source |
|--------|--------|
| Cadence, step/stride time | Heel-strike intervals |
| Stance/swing/double-support % | Contact masks |
| Walking speed | Median pelvis horizontal displacement / Δt |
| Step width | Median \|x_left_ankle − x_right_ankle\| |

Each metric includes a **confidence** field (0–1).

## 7. Gait Quality Score (0–100)

Weighted composite:

| Component | Weight |
|-----------|--------|
| Stability score (v2) | 0.22 |
| Stability margin % stable | 0.15 |
| Symmetry | 0.20 |
| Cadence consistency | 0.12 |
| Cycle consistency | 0.12 |
| Contact reliability | 0.10 |
| Movement quality | 0.09 |

Returns `explanation` and `dominant_factors` for interpretability.

**Kind:** `derived`

## 8. Video Quality

Heuristic checks before analysis:

- Body truncation (missing feet/head)
- Low FPS (< 20)
- Low pose confidence
- Camera movement (pelvis image jitter)
- Motion blur proxy (large inter-frame landmark jumps)

**Export:** `video_quality.json` with `overall_quality_score` and `warnings`.

## 9. Exports

Under `data/output/motion_reference/<run_name>/`:

| File | Content |
|------|---------|
| `center_of_mass.npz` | COM trajectory + stability margin arrays |
| `base_of_support.npz` | Support polygons |
| `video_quality.json` | Input quality assessment |
| `biomechanical_report.json` | Full metrics, abnormalities, interpretation |

Real-to-Sim report (`real_to_sim_pipeline_report.json`) includes `biomechanical` summary and stage `5_biomechanical`.

## 10. GUI

**Biomechanics** tab: summary cards + synchronized timeline plots.

**3D Overview overlays:** COM marker (red), support polygon (blue), gait direction arrow (green), foot contact labels.

## 11. Validation

```bash
python scripts/validate_biomechanical_demos.py
```

Compares Normal / Athletic / Abnormal when pose JSON is available. Does not force expected ordering if data contradicts labels.

## Limitations

- Monocular scale: absolute meters are approximate.
- BoS polygons are 2D projections, not foot pressure maps.
- Stability margin is a kinematic proxy, not capture-point or ZMP from force data.
- OpenSim inverse dynamics still requires measured external loads for true kinetics.

## Module map

```
stablewalk/analysis/biomechanical/
  com_estimation.py
  base_of_support.py
  stability_margin.py
  symmetry_metrics.py
  joint_rom.py
  advanced_gait_metrics.py
  gait_quality_score.py
  video_quality.py
  orchestrator.py
stablewalk/io/biomechanical_export.py
```
