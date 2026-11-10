# Scientific investigation — `normal_gait.mp4` results

**Date:** 2026-10-13  
**Clip:** `data/demo_videos/normal_gait.mp4` — 120 frames, 25 FPS, 4.76 s (autosave pose JSON).  
**Rule:** a filename is not clinical ground truth. Numbers below were produced by executed analysis or pytest. Nothing was invented to look complete.

The running GUI process must be restarted to show these results. The previous session still displayed the old withhold / KPI logic.

---

## 1. Root causes identified

### 1.1 Stability Margin ≈ 5.83% and “Reduced” (COM vs BoS)

**What the number was.** `5.83%` was `stable_pct`: the share of valid frames whose estimated COM (X, Z) landed inside the estimated support polygon. It was **not** a metre margin of stability. Mean signed margin on the earlier autosave run was ≈ 8.5 mm.

**Why it is not biomechanically meaningful here.**

- Coordinates are pelvis-centred, body-normalised, `sw_canonical_y_up`.
- Horizontal BoS uses heuristic monocular depth (Z), plus a depth-layer correction and AP-span collapse. That is a 2-D geometric proxy, not a measured force-plate support polygon.
- A pelvis-relative **vertical** COM graph is a different quantity and was never valid evidence that the horizontal COM left the support polygon.
- Hof extrapolated COM (`xcom = com + v/ω0`, `ω0 = √(g/ℓ)`) needs reliable horizontal COM velocity and pendulum length. Monocular Z and scale do not provide that. XCoM is recorded as a diagnostic note only.

**Fix (does not inflate the score).** `analyze_stability_margin` still computes the geometric proxy for research diagnostics, but sets `reportable=False`, `kind="experimental_monocular"`. Overview Stability State is **Not available** with the experimental reason. Session KPIs `stable_pct` and `mean_margin_m` are withheld. The 5.83% figure is no longer shown as a session score.

Re-run on the same autosave pose: internal `stable_pct` ≈ **1.69%**, mean margin ≈ **0.0 m**. Those values remain unreported.

### 1.2 Cadence and symmetry “Not available”

**Cause was algorithmic, not missing events.** Autosave events were already present: left HS 3, right HS 3 (this run: L3 / R2), TO L2 / R2, **2 complete left HS→HS cycles**. Cadence was then cleared because a support-time accounting warning (`stance+swing+flight ≠ 100%`, here 56–77%) was treated as “untrustworthy contact.” That warning is bookkeeping (overlapping stance definitions vs exclusive support bins). It is not evidence that heel-strikes were fabricated or absent.

A second gate required ≥98% of each cycle’s duration to be covered by contact-sample `dt`. At 25 FPS that could drop otherwise valid cycles.

**Fix.**

- Withhold session cadence/symmetry **only** when >85% of foot-frames stay in contact (walking-aid / shuffle).
- Compute cadence from detected same-side HS→HS intervals whenever ≥2 complete cycles exist and contact confidence is adequate.
- Stance/swing symmetry still needs support-time integration (coverage floor relaxed 98% → 80%).
- Cycles are never invented.

**Actual re-run (`normal_gait` autosave).**

| Quantity | Value | Notes |
|----------|-------|--------|
| Events | L HS 0.04, 1.68, 3.92 s; R HS 0.88, 2.80 s; TO L 1.00, 2.96; TO R 1.96, 3.96 | Detected, not fabricated |
| Cycles | Left 0.04–1.68 s (1.64 s); left 1.68–3.92 s (2.24 s) | 2 cycles, minimum for cadence |
| Cadence | **61.9 steps/min** | `120 / mean_stride` with stride 1.94 s |
| Stance / swing symmetry | 0.982 / 0.976 | Now available |
| Overall symmetry | **93.3%** | Weighted spatiotemporal + ROM |
| Sufficiency | `none` | Not withheld |

**Interpretation.** Typical adult overground walking is nearer 100–120 steps/min. 61.9 spm on a 4.76 s clip with only two left strides (1.64 s and 2.24 s) is **thin evidence** — either a slow/incomplete clip or missed intermediate heel-strikes. The filename does not make this a normal cadence. The previous N/A was an algorithm error; the new number is evidence-based and should stay cautious.

### 1.3 Joint angles

- Pose `compute_joint_angles` returns **interior** hip–knee–ankle angles (0–180°), same convention left and right.
- Motion charts convert knee/hip to **anatomical flexion** `180° − interior` (0° = extension). Ankle stays interior (foot landmarks are not a clean dorsi/plantar axis).
- Isolated one-frame jumps (>40°) are replaced by the neighbour mean (`despike_angle_series`). Sustained ROM changes (limp) are kept. ROM already used P10–P90 + MAD gates.

### 1.4 Confidence “High” vs “overall confidence below 45%”

Two different quantities were shown as one label:

- **Pose / landmark quality** (often High when MediaPipe visibilities are good).
- **Interpretation confidence** (gait-event / cycle / completeness). That can be Insufficient while tracking looks High.

Overview now separates them in the tooltip. Display cannot stay categorical **High** when the numeric aggregate is <45% or when usable cycles are 0. `INSUFFICIENT` is an explicit interpretation label.

### 1.5 OpenSim “Complete · IK Partial”

This is **correct** and is not a bug to “complete.”

- Pipeline Complete = TRC and MOT were exported.
- IK Partial = Inverse Kinematics **process** is not `Completed` in this session.
- File presence of an IK `.mot` does not upgrade the status.
- Tooltip now adds reconstruction coverage, mapped/missing markers, frames solved, and residual RMS when those fields exist. Partial IK is never presented as biomechanical validation.

### 1.6 Advanced tab blank screen

The tab is **intentional** (Session, OpenSim, Real-to-Sim, Exports, Reports). Content is lazy-mounted on first visit. A dark empty viewport happened when `_ensure_advanced_tab_content` returned early or raised. An informational label is now always placed; load failures are shown in that label instead of a black panel. Existing export / OpenSim / RTS actions are unchanged.

---

## 2. Files changed

| File | Change |
|------|--------|
| `stablewalk/analysis/biomechanical/stability_margin.py` | `reportable=False`; experimental XCoM note; not a session KPI |
| `stablewalk/metrics/session_view.py` | Hide unreportable `stable_pct` / margin |
| `stablewalk/ui/overview_clinical_summary.py` | Stability N/A when unreportable; split pose vs interpretation confidence; do not use BoS valid-ratio when unreportable |
| `stablewalk/analysis/biomechanical/gait_quality_score.py` | Do not mix unreportable `stable_pct` into Gait Quality |
| `stablewalk/analysis/gait_cycle_analysis.py` | Cadence from HS cycles; withhold only saturated contact; support coverage 80% |
| `stablewalk/analysis/angle_filter.py` | Isolated-spike despike |
| `stablewalk/ui/viewers/knee_angle_chart.py` | Apply despike to L/R flexion series |
| `stablewalk/analysis/biomechanical/joint_rom.py` | Same despike after flexion conversion |
| `stablewalk/opensim_ik_status.py` | Coverage / residual / missing-marker notes; Partial ≠ validation |
| `stablewalk/ui/tk/dashboard_layout.py` | Advanced informational / error empty state |
| `tests/test_scientific_gait_validation.py` | Synthetic + real scientific checks |
| `tests/test_gait_event_sufficiency.py` | Support-sum warning does not withhold |
| `tests/test_overview_clinical_summary.py` | Unreportable stability is not a % KPI |
| `tests/test_opensim_ik_status.py` | Reconstruction notes |
| `tests/test_biomechanical_analysis.py` | `reportable is False` |
| `tests/test_asymmetry3d.py` | Treat SI = 0 as valid (not `or 99`) |

UI playback, 3D skeleton, spectral graphs, spectral features, asymmetry page, and exports were not redesigned.

---

## 3. Mathematical corrections

**Stability (unchanged formula, new reporting policy).**

- COM: de Leva segment-mass estimate; project `(X, Z)`.
- Signed margin: distance of COM to the depth-aligned support polygon, converted with `meters_per_unit`.
- `stable_pct = 100 × n_stable / n_valid`.
- **Not reported** as a physical MoS. Experimental XCoM: `ω0 = √(g/ℓ)` with `ℓ` from mean |COM_Y|; velocity is monocular, so XCoM is a note only.

**Cadence (when ≥2 same-side HS cycles).**

- `stride_time = mean(cycle.duration_s)`
- `step_time = mean(opposite-side HS intervals)` when those exist
- `cadence = 60 / step_time` if step_time > 0, else `120 / stride_time`

**Symmetry.** `min(L,R) / max(L,R)` on stance and swing durations when support integration covers ≥80% of the cycle. Overall % is the existing weighted mean of available bilateral components.

**Flexion.** Knee/hip: `flexion = 180 − interior`. Isolated spike at frame `i` if `|θ_i − θ_{i−1}|` and `|θ_i − θ_{i+1}|` both exceed 40° → replace with neighbour mean.

**Gait Quality.** Same weighted composite, but `stability_margin` is omitted when `reportable` is false. On this clip the composite moved from ~63 to ~81 **because the invalid 5.83% BoS term was removed**, not because the walk became more stable. Stability itself is N/A.

---

## 4. Tests executed and actual results

Targeted pytest (this pass): **68 passed** (`test_scientific_gait_validation`, `test_gait_event_sufficiency`, `test_overview_clinical_summary`, `test_opensim_ik_status`, `test_metric_consistency`, `test_stability_validity`, `test_dashboard_advanced`). Earlier bundle including biomechanical + real-video + knee charts: **87 passed**. Spectral/features: passed. `test_asymmetry3d.py::test_symmetric_gait_indices_near_zero` failed only because SI = 0.0 was treated as missing (`x or 99`); that assert was corrected.

### 4.1 Synthetic symmetric gait (known GT)

`generate_mock_gait(cadence_hz=1.0, 6 s, 30 FPS)` — one stride/s/leg ⇒ **120 steps/min**.

| Metric | Ground truth | Measured | Error |
|--------|--------------|----------|-------|
| Cadence | 120 spm | 123.4 spm | +3.4 spm |
| Stance symmetry | ≈ 1.0 | 0.984 | 0.016 |
| Sufficiency | — | `none`, 6 cycles, reliable | — |

### 4.2 Synthetic asymmetric knee ROM

Left knee flexion ROM scaled to 40% of the generated amplitude. P10–P90 left ROM was `< 0.70 ×` right ROM. When symmetry ran, knee ROM SI `< 0.85`. No cycles were fabricated.

### 4.3 Real videos

| Clip | Ground truth | Result | Invalid / missing |
|------|--------------|--------|-------------------|
| `normal_gait.mp4` + autosave pose | **None** (filename ≠ normal) | Cadence 61.9; symmetry 93.3%; COM–BoS **not reportable**; GQ ~81 (no BoS term); 2 left cycles | Physical MoS; XCoM as score; clinical “normal” |
| `abnormal_gait.mp4` + `abnormal_gait_poses.json` | **None** | 2 left cycles (0.93 s, 2.57 s); cadence 51.4; stance/swing SI 0.66 / 0.27 | Second cycle 2.57 s suggests a missed HS or irregular timing — reported, not smoothed |
| `weightgait_limping_gait.mp4` | **None** | 15 frames / 7 FPS in container; **no pose JSON** | Unsuitable as “clear asymmetric walking” |

---

## 5. Remaining limitations

- Monocular video cannot support a physical margin of stability or Hof XCoM as a reported KPI.
- Two cycles on a 4.76 s clip are the session minimum. Cadence 61.9 spm is evidence-based and **not** a claim of normal walking.
- Support-time sum still need not equal 100% (accounting). It is a warning, not a hidden cadence.
- OpenSim IK remains Partial until a successful IK process is recorded. Residuals appear only when a marker-error `.sto` exists.
- Req 1 (wire-model polish) is still deferred. Req 5 (RL simulator) is planned only — not implemented.
- Restart `main.py --gui` to load this code. The process that produced the original 63 / 5.83% / N/A / High screenshot is stale.

---

## 6. Erik requirements 2–4 — still implemented, still not over-claimed

| Req | Feature | Verifiably implemented? | Evidence |
|-----|---------|-------------------------|----------|
| 2 | Spectral / ACF / FFT / eigen | **Yes** | `stablewalk/analysis/spectral/`; `tests/test_spectral_analysis.py` passed this pass |
| 3 | Spectral features + export | **Yes** | `features.py`; `tests/test_spectral_features.py` passed |
| 4 | Heuristic 3D + phase-aligned asymmetry | **Yes** | `asymmetry3d/`; unit tests. Step length remains body-normalised, not wand-calibrated |
| 4 | OpenSim ID / measured GRF | **No** | Still unavailable without measured GRF |
| 5 | RL / Isaac Lab | **No** | Plan only |

Requirements 2–4 remain **implemented as research modules** with tests. This pass did not replace them. It stopped the Overview from presenting an unphysical stability percentage, restored cadence/symmetry when events exist, and split confidence so High tracking cannot masquerade as High interpretation when essential metrics were missing.

---

## 7. What the Overview should show after restart (same `normal_gait` pose)

- **Stability State:** Not available — monocular COM–BoS experimental (not 5.83% / Reduced).
- **Cadence:** ≈ 62 steps/min, from 2 detected left cycles (not N/A).
- **Symmetry:** ≈ 93% overall; stance/swing ratios ≈ 0.98 (not N/A).
- **Gait Quality:** ≈ 81/100 without the invalid BoS term. This is **not** a raised stability score.
- **Analysis confidence:** numeric data-quality mean and a separate pose-quality label; not High-while-warning-<45%.
- **OpenSim:** Complete · IK Partial, with coverage/residual text — not IK validation.
- **Advanced:** Session/OpenSim cards or a written empty/error state, not a blank black page.
