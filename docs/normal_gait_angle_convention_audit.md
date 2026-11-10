# normal_gait.mp4 angle-convention audit

Generated: 2026-10-21T15:15Z from live code on two pose caches of the same video.
The filename is not a diagnosis. Thresholds, ACF, and f0 were not changed to look successful.

## What the GUI is actually showing

The running session is **autosave poses**, not `data/output/poses/normal_gait_poses.json`.

| Cache | Path | Pose mtime / export | Complete HS–HS | Primary f0 | ACF stride | Scores in that bundle |
|---|---|---|---|---|---|---|
| **GUI autosave** | `data/output/sessions/autosave/stablewalk_autosave/` | 2026-10-21 17:53 | **3** left, 1.16 s each | **0.4167 Hz** | unavailable | **92.5** gait-quality composite, **82.5** pelvis stability |
| July catalog JSON | `data/output/poses/normal_gait_poses.json` | 2026-07-13 | **2** left (1.56 s, 2.16 s) | **0.4167 Hz** | unavailable | live recompute 75.4 / 80.7 / 66.3 — not 92.5 / 82.5 |

These are **two different landmark reconstructions** of `normal_gait.mp4` (25 Hz, 120 frames, 4.76–4.80 s). Spectral on the primary channel is current in both. Cycle counts and summary scores are not interchangeable between the two files.

## Observed symptoms, traced

| Symptom | Cause | Not a cause |
|---|---|---|
| Dominant frequency 1.042 → 0.4167 Hz | Primary channel is `dof:left_knee_flexion`. Current f0 is the max Welch bin in 0.40–4.00 Hz on the **3D HKA flexion** series. 0.4167 Hz = 2 Δf (Δf = 25/120 = 0.2083 Hz). **1.0417 Hz still exists on catalog hip channels**, not on current left knee. | 180°−θ label flip (affine; PSD/ACF unchanged). No frequency prior was applied. |
| ACF period unavailable | No **positive local maximum** of ρ(k) inside 0.60–1.80 s with ρ > 0.05. Knee window max ρ sits at the **1.80 s edge** (not a local peak). | Cycle detector (independent). Thresholds were not raised to create a peak. |
| “3 valid cycles” | Autosave `analyze_gait_cycles`: 4 left HS + 3 right HS → **3 complete left HS–HS**. Spectral features report 3 valid / 3 detected on that list. | ACF `n_stride_cycles` (unavailable). Catalog-JSON detector (2 cycles). |
| Scores remain 92.5 and 82.5 | Autosave `results_snapshot.json` `biomech_analysis.gait_quality.score = 92.5` and `pelvis_stability.score = 82.5`. Those formulas **do not read f0 or ACF**. | Spectral recompute. Live catalog-JSON biomech (different numbers). |

## Two pose versions (inconsistent signal sources)

Autosave stored left-knee DOF is **99–156°** (median 137°). Catalog stored left-knee DOF is **172–180°** (median 179°). Current analysis does **not** use those stored values for the knee channel.

| Left knee series | Autosave | Catalog JSON |
|---|---|---|
| Stored interior DOF (pose JSON / `left_knee_flexion`) | 98.6–156.0° | 172.5–180.0° |
| 180° − stored | 24.0–81.4° | 0.02–7.53° |
| 3D HKA flexion (spectral raw / kinematics) | 2.0–32.9° (median 7.0°) | 2.9–35.7° (median 7.7°) |
| MAE(HKA flexion, 180° − stored) | **35.7°** | **8.1°** |
| Spectral raw vs HKA flexion | match (0°) | match (0°) |

`analysis_knee_flexion_deg` prefers 3D Hip–Knee–Ankle flexion and only falls back to 180° − stored if HKA landmarks are missing. Stored pose DOFs used `angle_at_joint`, which can drop to 2D. That is a **different geometry**, not a rename.

A pure 180° − θ map is affine. After linear detrend + z-score, PSD and ACF of flexion equal those of the same interior series. The 1.042 → 0.4167 change therefore **cannot** come from the 180° − θ label alone.

## Formulas

**Knee interior θ**  
θ = acos(clip( (hip−knee)·(ankle−knee) / (|hip−knee| |ankle−knee|), −1, 1 )) in degrees. Always XYZ for HKA; stored DOF may use 2D if depth is negligible.

**Knee flexion (canonical analysis / display)**  
knee_flexion_deg = 180° − θ. 0° = extension.

**Dominant frequency**  
f0 = arg max_{f ∈ [0.40, 4.00] Hz} P(f), P = Welch PSD of the z-scored processed series. Hann window. `nperseg` defaults to ~N (120) so Δf = fps/N = 0.2083 Hz.

**ACF**  
ρ(k) = Σ x[t] x[t+k] / Σ x[t]² after mean removal on the longest finite run.  
Stride period = lag of the largest **positive local max** of ρ in [0.60, 1.80] s with ρ > 0.05. A trough or a window-edge maximum is not a period.

**Contact cycle**  
Complete ipsilateral heel-strike → next same-side heel-strike with duration in [0.35, 3.0] s. Partial lead/trail intervals are dropped. Uses foot clearance + contact FSM. Does not use joint-angle convention, PSD, or ACF.

**Heuristic gait-quality composite (92.5 in autosave)**  
score = Σ(component_k · weight_k) / Σ(weight_k) over active k.  
Weights: stability 0.22, stability_margin 0.15, symmetry 0.20, cadence_consistency 0.12, cycle_consistency 0.12, contact_reliability 0.10, motion_quality 0.09. Missing components omitted; weights renormalized. Not a validated instrument.

**Pelvis stability (82.5 in autosave)**  
Stability v2 domain from gait-frame pelvis sway / oscillation / jerk. Not f0.

**Overview movement-stability / gait-quality domain groups**  
Different algorithms: movement = pelvis + trunk + smoothness; overview gait-quality domains = temporal/spatial symmetry + clearance + cycle + contact. Do not substitute one for the other.

## Spectral settings (unchanged)

| Setting | Value |
|---|---|
| visibility_min | 0.50 |
| max_gap_s | 0.15 (linear fill only; no edge extrapolation) |
| smooth_window_s / poly | 0.12 s Savitzky–Golay / 3 |
| detrend | linear |
| zscore_for_matrices | true (ACF/FFT use `normalized`) |
| gait_band_hz | (0.40, 4.00) |
| stride_lag_s | (0.60, 1.80) |
| step_lag_s | (0.30, 0.90) |
| max_acf_lag_s | 2.50 |
| min_samples / min_duration_s | 64 / 2.00 |
| min_acf_peak | 0.05 |
| coordinate_system | `sw_canonical_y_up` |
| primary channel | `dof:left_knee_flexion` |
| knee storage tag | `pose_interior_hka` |

## Preprocessing chain (spectral)

1. Snapshot DOF → knee converted via HKA flexion; hip/ankle passed through stored DOF.  
2. Resample to uniform FPS grid.  
3. Drop samples with landmark visibility < 0.50.  
4. Linear-fill interior gaps ≤ 0.15 s.  
5. Short SG smooth (keeps ~1–3 Hz).  
6. Linear detrend → `processed` (deg, signed about 0).  
7. Z-score → `normalized` (dimensionless) for ACF/FFT/PSD.

Autosave primary raw: 2.0–32.9° (median 7.0°). Catalog primary raw: 2.9–35.7° (median 7.7°).

## Cycle methods (confidence is not interchangeable)

### 1. Contact HS–HS (`analyze_gait_cycles`)

| | Autosave (GUI) | Catalog JSON |
|---|---|---|
| Left HS times (s) | 0.04, 1.20, 2.36, 3.52 | 0.04, 1.60, 3.76 |
| Right HS times (s) | 0.04, 0.96, 2.32 | 0.88, 2.88 |
| Complete cycles | **3** left @ 1.16 s | **2** left @ 1.56, 2.16 s |
| Cadence | 128.6 steps/min | 64.5 steps/min |
| Contact confidence | 1.0 HIGH | 1.0 HIGH |
| Reliability reason | Computed from 3 complete same-side heel-strike cycles | Computed from 2 complete same-side heel-strike cycles |
| Support-time warning | stance+swing+flight = 83.33% | 51.92% |

Contact confidence is **not** spectral reliability and **not** pose visibility.

### 2. Left + right ipsilateral cycles (`cycles_from_heel_strikes`)

Same HS events, counted per side. Catalog: L2 + R1 = **3**. Autosave: L3 + R2 = **5**.  
If a UI label says “3 valid cycles” on the catalog file, it is this **sum**, not 3 left strides. On the GUI autosave, “3” is the primary detector’s left HS–HS count.

### 3. Spectral feature cycle count

Reuses method 1 (`cycles.cycles`), then counts durations > 0.  
Autosave: **3 valid / 3 detected**. Catalog: **2 valid / 2 detected**.  
Per-cycle PSD slices use those HS–HS bounds. This is not an ACF estimate.

### 4. ACF period

Separate estimator on the z-scored primary knee series.  
Autosave and catalog: **unavailable** — “No clear autocorrelation peak in the stride lag window.”  
Must not be filled from methods 1–3.

### Snapshot vs stability usable-cycle count (autosave inconsistency)

`results_snapshot.json` stores **3** cycles from method 1, but `biomech_stability` in the same file says **usable gait cycles: 0** and marks joint smoothness / cycle consistency UNAVAILABLE. Gait quality 92.5 still used `cycle_consistency = 100` from the cycle-analysis object. Those two layers were not written from the same cycle evidence.

## Hip and ankle conventions

| Joint | Pose storage | Spectral raw | Kinematics table | Joint ROM |
|---|---|---|---|---|
| Knee | interior HKA via `angle_at_joint` (2D fallback allowed), tagged `pose_interior_hka` | **3D HKA flexion** `180°−θ` | 3D HKA flexion | `180°−interior` |
| Hip | interior shoulder–hip–knee, **named** `*_hip_flexion` | **stored interior** (no convert) | stored interior | `180°−interior` |
| Ankle | interior knee–ankle–foot_index, named `*_ankle_flexion` | stored interior | stored interior | interior span (not 180°−θ) |

Hip is labeled flexion in Spectral/Asymmetry ids but is still the pose interior angle. ROM hip and spectral hip are **different conventions**. Ankle is consistently interior (foot landmarks are not a clean dorsi/plantar axis).

Catalog hip still peaks at **1.0417 Hz** with ACF lag 1.04 s (ρ ≈ 0.16–0.24, reliability `low`). Autosave hip peaks at **0.4167 Hz** with ACF unavailable. Switching the displayed channel from hip to left knee on the catalog file looks exactly like “1.042 → 0.4167”.

## Composite-score dependencies

Autosave `gait_quality` 92.5 components: stability 89.95, symmetry 82.52, cadence_consistency 100, cycle_consistency 100, contact_reliability 100 (5/7; stability_margin and motion_quality omitted).  
Autosave pelvis domain 82.5. Symmetry component 82.52 is a third nearby number, not f0.

Live catalog-JSON recompute (same code, other pose file): movement stability 75.4, overview gait-quality domains 60.8, heuristic composite 80.7, legacy 8-domain 66.3. Domain scores there: pelvis 80.4, trunk 95.3, contact 87.5, smoothness 51.6.

None of these read spectral f0 or ACF. Recomputing Spectral cannot change 92.5 / 82.5. Recomputing biomech on the **catalog** JSON also cannot reproduce 92.5 / 82.5 because that cache is a different skeleton.

## Stale / mixed versions

1. **Two pose files** for one video (autosave 2026-10-21 vs catalog 2026-07-13). Cycle detector and hip spectra disagree.  
2. **GUI `gui._spectral_report`** is kept until `force=True`. A session started before the knee conversion can show an old f0 until Spectral is recomputed. After recompute, leftover session JSON can still hold old biomech scores.  
3. **Autosave snapshot internal split**: 3 HS–HS cycles vs stability “0 usable cycles”.  
4. **E2E dump** `docs/_session_consistency_e2e/recording_a/spectral.json` is the catalog file: primary 0.4167 Hz, hip 1.0417 Hz, ACF null, 2 cycles. It is current for that file, not a restored 1.042 knee result.

## Quality reasons (left unavailable on purpose)

- Knee ACF: no in-window local max; ρ at 1.80 s is an edge, not a stride peak.  
- 0.4167 Hz is the first in-band bin with the most power on a short, weakly periodic knee series (centroid ~0.76 Hz on catalog). It is not “the walking rate.”  
- Support+flight ≠ 100% is bookkeeping, not a fabricated cycle count.  
- Catalog cadence 64.5 steps/min from two long left strides is thin evidence; autosave 128.6 steps/min from three 1.16 s strides is a different detector input, not a corrected diagnosis.

## What was not done

- No expected-frequency override.  
- ACF period left unavailable.  
- No threshold edits to create a stride peak or to force cycle counts to match.  
- Catalog and autosave were not merged.

Machine-readable dump: `docs/_normal_gait_angle_convention_audit/audit.json` (catalog-JSON live run). Re-run `python scripts/normal_gait_angle_convention_audit.py` for a fresh catalog dump.
