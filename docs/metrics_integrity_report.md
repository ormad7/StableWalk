# Metrics integrity audit — `normal_gait.mp4`

**Date:** 2026-10-13  
**Clip:** `data/demo_videos/normal_gait.mp4` — **120 frames, 25 FPS, 4.76 s**.  
**Pose source:** `data/output/sessions/autosave/stablewalk_autosave/poses.json` (same video).  
**Rule:** a filename is not clinical ground truth. Numbers were not tuned to make this clip look “normal.”

Two result sets exist. They must not be mixed.

| Set | Origin | Cadence | Symmetry | Gait Quality | COM–BoS | Analysis confidence |
|-----|--------|---------|----------|--------------|---------|---------------------|
| **A — GUI snapshot the user saw** | `results_snapshot.json` exported 2026-10-13 10:49 | **63 steps/min** | **87%** | **84/100** | Snapshot *analysis_summary* said “Reduced (6% stable frames)” while biomech `reportable=false` | **High** (pose-weighted 95%) |
| **B — live recompute, this pass** | `run_biomechanical_analysis` + `build_analysis_summary` on the same poses | **61.86 → 62 steps/min** | **93.3%** | **80.95 → 81/100** | **Not available** | **Moderate** |

Set A is what Overview/Results showed in that session. Set B is what current code produces. Event timestamps differ (see §2). **Neither set is optical-mocap ground truth.** No heel-strike file for this clip was found in the repo.

---

## 1. Source and definition of every reported number

### Gait Quality ≈ 84/100 (snapshot) / 81/100 (live)

**Function:** `compute_gait_quality_score` in `stablewalk/analysis/biomechanical/gait_quality_score.py`.  
**Kind:** derived composite 0–100. **Not** a clinical grade.

\[
Q=\sum_k w_k c_k \Big/ \sum_k w_k
\]

over *active* components only (missing components are omitted, not zero-filled). Nominal weights: stability 0.22, stability_margin 0.15, symmetry 0.20, cadence_consistency 0.12, cycle_consistency 0.12, contact_reliability 0.10, motion_quality 0.09.

`stability_margin` is **omitted** when `reportable=False`. The BoS 5.83% / 1.69% is **not** inside Q.

Live components (set B):

| Component | Value | Role |
|-----------|------:|------|
| movement stability | 73.76 | pelvis/trunk/smoothness domain score — **not** COM–BoS |
| motion quality | 40.34 | joint smoothness (weakest term) |
| overall symmetry | 93.31 | same composite as the Symmetry card |
| cadence consistency | 84.54 | \(1-\mathrm{CV}\) of cycle duration |
| cycle consistency | 84.54 | same CV transform |
| contact reliability | 100 | contact_confidence × 100 |

\(Q \approx 80.95\) → displayed **81/100**. Snapshot Q **83.6 → 84/100** used symmetry 87.23 and cadence/cycle consistency ~99 because that session’s two right-side strides were more even (1.88 s and 1.92 s).

A **second** “Gait Quality” of **58.3** exists in `gait_analysis_summary` (temporal/spatial symmetry, clearance, cycle consistency, contact). Overview / Results **84 or 81** come from `compute_gait_quality_score`, not that 58.3. They are different algorithms.

### Symmetry ≈ 87% (snapshot) / 93% (live)

**Function:** `analyze_symmetry` in `stablewalk/analysis/biomechanical/symmetry_metrics.py`.  
**Display:** `overall_symmetry_pct`.

Each bilateral ratio is \(\min(L,R)/\max(L,R)\) (stance, swing) or \(2\min(L,R)/(L+R)\) (step/stride length, ROM). Overall:

\[
S=100\cdot\frac{\sum_i w_i r_i}{\sum_i w_i}
\]

weights: stride 1.4, step 1.3, stance 1.2, swing 1.2, cadence consistency 1.1, knee ROM 1.0, hip 0.9, ankle 0.9. Need ≥2 components.

**87% is not clinical symmetry.** It is this weighted mean. Snapshot swing ratio **0.630** pulled S down to 87.23. Live swing ratio **0.976** gives **93.31**. ROM spans on this clip are **&lt; 8°** for hip/knee (oblique monocular sagittal estimates); that ROM SI is noisy. A warning is now attached. The composite was **not** retuned.

### Cadence ≈ 63 (snapshot) / 62 (live) steps/min

**Function:** `_compute_temporal_metrics` in `gait_cycle_analysis.py`.

- Stride time = mean of complete **same-side** HS→HS cycle durations.  
- Step time = mean of **alternating** HS intervals that fall inside those cycles.  
- Cadence (**step frequency**):

\[
\mathrm{cadence}=\begin{cases}
60/t_{\mathrm{step}} & t_{\mathrm{step}}>0\\
120/t_{\mathrm{stride}} & \text{otherwise}
\end{cases}
\quad\text{steps/min}
\]

Stride frequency is \(60/t_{\mathrm{stride}}\) **strides/min** (one-half of cadence for a two-step stride). Typical adult overground walking is nearer 100–120 steps/min. **62–63 spm on 4.76 s with two cycles is thin evidence**, not “normal walking.”

### Physical COM–BoS

**Function:** `analyze_stability_margin`. Coordinates are pelvis-centred, body-normalised, heuristic Z. `reportable=False`. Internal `stable_pct` (snapshot 5.83%, live 1.69%) is a 2-D geometric proxy, **not** Hof margin of stability. Overview must show **Not available**. Snapshot `analysis_summary` incorrectly showed “Reduced (6% stable frames)” while the biomech blob already had `reportable=false`. Current `build_analysis_summary` withholds it.

### OpenSim IK Partial

**Function:** `resolve_opensim_ik_status` / `_pair_short_label`.  
**Complete · IK Partial** means TRC/MOT export finished; Inverse Kinematics **process** is not `Completed`. File presence of `.mot` does not upgrade the status. No marker-error `.sto` was found → **residuals unavailable**.

### Analysis Confidence High (snapshot)

**Bug (fixed this pass).** `build_analysis_summary._analysis_confidence` averaged pose/video quality (~95/100) with contact confidence and gait-quality.confidence → **95% → High**. That is **execution / tracking success**, not measurement reliability. `gait_analysis_summary` already said **MODERATE** (2 cycles). Results showed High; Overview could show High or a high % because `_usable_cycle_quality_pct` scored **2 cycles as 100%**.

**Documented High (now):** ≥3 complete same-side HS cycles, duration ≥6 s, contact/cycle score ≥72%, contact tier not LOW. Pose tracking is a **separate** field. This clip cannot be High.

---

## 2. Cadence — heel strikes, timestamps, step vs stride

### Set A (snapshot, 10:49) — matches the user’s 63 steps/min

| Event | Frame | Time (s) |
|-------|------:|---------:|
| left HS | 1 | 0.04 |
| right HS | 22 | 0.88 |
| left TO | 30 | 1.20 |
| left HS | 39 | 1.56 |
| right TO | 53 | 2.12 |
| right HS | 69 | 2.76 |
| left TO | 81 | 3.24 |
| left HS | 89 | 3.56 |
| right TO | 106 | 4.24 |
| right HS | 117 | 4.68 |

Cycles used for cadence: **right** HS→HS 0.88–2.76 s (1.88 s) and 2.76–4.68 s (1.92 s). Mean stride **1.90 s**. Alternating HS inside those cycles: mean step **0.95 s**. Cadence \(60/0.95 = 63.16\) steps/min. Stride frequency \(60/1.90 = 31.6\) strides/min.

### Set B (live recompute, this pass)

| Event | Frame | Time (s) |
|-------|------:|---------:|
| left HS | 1 | 0.04 |
| right HS | 22 | 0.88 |
| left TO | 25 | 1.00 |
| left HS | 42 | 1.68 |
| right TO | 49 | 1.96 |
| right HS | 70 | 2.80 |
| left TO | 74 | 2.96 |
| left HS | 98 | 3.92 |
| right TO | 99 | 3.96 |

**No right HS at 4.68 s.** Cycles: **left** HS→HS 0.04–1.68 s (1.64 s) and 1.68–3.92 s (2.24 s). Mean stride **1.94 s**. Mean step **0.97 s**. Cadence \(60/0.97 = 61.86\) → **62 steps/min**. Stride frequency \(60/1.94 = 30.9\) strides/min.

The two left strides (1.64 s vs 2.24 s) are inconsistent. That may be a missed intermediate HS or a genuinely irregular clip. **Events were not adjusted to raise cadence.**

**Manual reference annotations:** none in this repository. Container metadata only (120 frames, 25 FPS).

**Short-recording reliability:** 4.76 s and 2 cycles meet the cadence *minimum* (`MIN_CYCLES_FOR_CADENCE = 2`) and fail the **High** confidence floor (3 cycles, 6 s). Support-time sum is not 100% (warning; accounting, not a fabricated cadence).

---

## 3. Symmetry — what 87% actually is

Snapshot components (set A):

| Component | Ratio | Weight |
|-----------|------:|-------:|
| stride length | 0.953 | 1.4 |
| step length | 0.953 | 1.3 |
| stance duration | **0.872** | 1.2 |
| swing duration | **0.630** | 1.2 |
| cadence consistency | 0.989 | 1.1 |
| knee ROM | 0.821 | 1.0 |
| hip ROM | 0.993 | 0.9 |
| ankle ROM | 0.747 | 0.9 |

Weighted mean × 100 = **87.23%**. Stance L/R times 1.56 s / 1.36 s; swing 0.34 s / 0.54 s. The 87% figure is dominated by **swing asymmetry in that event set**, plus noisy 2–3° knee ROM SI.

Live components (set B): stance 0.982, swing 0.976, step/stride 0.894, cadence consistency 0.845, knee 0.981, hip 0.994, ankle 0.920 → **93.31%**. Hip/knee ROM still &lt; 8°; warning added. Still **not** a clinical laterality score.

The Gait Quality explanation no longer says “typical acceptable range.”

---

## 4. Foot clearance

**Ground plane:** `estimate_ground_plane` — horizontal \(Y=\mathrm{floor}_y\), robust low percentile of stance-weighted heel/toe heights. Snapshot `floor_y = -1.7545`. Scale mode **`body_normalized`**.

**Clearance:** \(\max(0,\,Y_{\mathrm{foot}}-\mathrm{floor}_y)\) in body-scale metres; UI ×100 → “cm estimated.” **Not calibrated laboratory centimetres.** No wand / known-height metric scale on this clip.

Autosave `bilateral_foot_clearance.json` states: *“Estimated body-scale clearance from monocular pose; centimeter values are not clinical ground truth.”*

Live feature clearances (session, body-normalised m): left **0.064**, right **0.074** (same order as snapshot 0.064 / 0.074). Sanity cap 0.35 m. Confidence: estimated / body-scale, not measured.

---

## 5. OpenSim — IK Partial

Export `run_787305e91e7251c7_1791882999321`:

- Detected frames **120/120**  
- Mapped markers **17** (L/R shoulder, hip, knee, ankle, heel, toe, elbow, wrist, HEAD; pelvis/thorax in mapping dict)  
- Units mm, `y_up_meters_to_mm`  
- Many marker **Z = 0** in the TRC JSON (heuristic / collapsed depth)  
- **IK residual RMS: not available** (no completed IK `.sto`)  
- SDK available in that export; process state not `Completed`

**IK Partial** = Inverse Kinematics did not complete in-session. TRC/MOT export is **not** a successful IK solve and is **not** biomechanical validation. Tooltip: `format_ik_reconstruction_notes`.

---

## 6. COM / BoS

Physical stability remains **unavailable**. Internal proxy is still computed for research diagnostics (`reportable=False`). Overview findings no longer receive `stable_pct` when unreportable. Illustrative COM/BoS graphs are not a diagnosis. Session KPI `stable_pct` is withheld.

---

## 7. Confidence — execution vs measurement

| Quantity | What it is | This clip |
|----------|------------|-----------|
| Tracking / video quality | MediaPipe visibility | ~95–100/100 (High **pose**) |
| Contact confidence tier | Floor + visibility + HS count | HIGH (contact bookkeeping) |
| Pipeline confidence | Stages completed | Moderate (~64%, 4/8) |
| **Analysis confidence** | Measurement reliability | Snapshot **High** (wrong). Live **Moderate** |

**High definition (code constant `HIGH_CONFIDENCE_DEFINITION`):** ≥3 complete same-side HS cycles, duration ≥6 s, contact/cycle score ≥72%, contact not LOW. Pose tracking is reported separately.

Per-metric quality on live summary: cadence available but 2-cycle/4.8 s; symmetry composite available; COM–BoS unavailable; IK Partial.

---

## 8. Report consistency

- Overview, Biomechanics, Results, and JSON must use `gui._biomech_analysis` / `_gait_cycle` / `_analysis_summary_cache` from the **current** video.  
- `_reset_gait_cycle_panel` now clears `_analysis_summary_cache`, `_spectral_report`, `_asymmetry_report` and bumps worker tokens so in-flight jobs cannot paint the next video.  
- Unavailable gait quality is `score=None`, `available=False` — **not 0/100**.  
- Snapshot set A mixed an old `analysis_summary` (BoS 6% shown) with biomech `reportable=false`. Current builder matches biomech.  
- **Restart the GUI** after this pass; a live session still holding the 10:49 snapshot will keep 63 / 87 / 84 / High until re-analysed.

---

## 9. Joint-angle discontinuities

`despike_angle_series`: isolated one-frame jumps &gt; 40° on **both** neighbours are replaced by the neighbour mean. Sustained ROM (limp) is kept. Knee charts and joint ROM now **count** replacements and flag them. Hip/knee flexion ROM on this clip is ~2–5° (oblique monocular) — a reconstruction limit, not despike flattening a real 40° gait arc. Cycle-normalised *interior* knee in the snapshot still spans ~106–150°, which is a different convention (interior vs 180−interior flexion).

---

## 10. Tests executed (actual outcomes)

```
python -m pytest
  tests/test_metrics_integrity.py
  tests/test_analysis_summary.py
  tests/test_overview_clinical_summary.py
  tests/test_joint_rom_flexion.py
  tests/test_gait_event_sufficiency.py
  tests/test_scientific_gait_validation.py
  tests/test_opensim_ik_status.py
→ 57 passed

  tests/test_biomechanical_analysis.py
  tests/test_metric_consistency.py
  tests/test_overview_clinical_display.py
  tests/test_opensim_ik_status.py
→ 53 passed
```

`tests/test_metrics_integrity.py` (new):

| Test | Result |
|------|--------|
| High definition documented (3 cycles, 6 s) | passed |
| 2 cycles + 4.8 s cannot be High (95% pose still Moderate) | passed |
| Despike isolated jump; keep sustained change | passed |
| Gait quality unavailable is not 0 | passed |
| Live normal_gait: cadence formula, 2-run reproducibility, BoS withheld, confidence ≠ High | passed |
| Unreportable BoS not passed into Overview findings | passed |
| Overview 2-cycle display is not High | passed |
| New-video reset clears spectral / asymmetry / summary cache | passed |
| Event list + cadence identical on two `analyze_gait_cycles` runs | passed |

No cadence or symmetry target was fitted to “look like normal walking.”

---

## Changes this pass (integrity only)

- `_analysis_confidence`: measurement reliability; High floor documented; pose quality excluded from the level.  
- Overview confidence: 2 cycles ≠ 100% evidence; tracking not averaged into the %; &lt;3 cycles cannot display High.  
- Unreportable COM–BoS excluded from Overview finding inputs.  
- Session reset clears spectral/asymmetry reports and summary cache.  
- Isolated angle spikes counted/flagged; ROM SI &lt; 8° warned.  
- Gait Quality copy no longer calls 87–93% a “typical acceptable range.”

---

## Remaining limitations

- No marker-based or force-plate ground truth for this file.  
- Event detector still differs between the 10:49 snapshot and current code; both are pose-derived.  
- Two cycles on 4.76 s remain the scientific minimum for reporting cadence, not a reliable session.  
- Monocular Z and body-normalised metres cannot support physical MoS or metric foot clearance.  
- OpenSim IK residuals stay unavailable until a completed IK process writes `.sto`.  
- Dual Gait Quality algorithms (81 vs domain-group 58) remain; Overview uses the 0–100 composite and labels it derived.
