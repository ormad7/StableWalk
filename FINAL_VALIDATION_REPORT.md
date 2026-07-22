# StableWalk — Final Validation Report

**Date:** 13 July 2026  
**Scope:** Cross-tab metric consistency, calculation grounding, GUI–backend alignment, and automated test coverage.  
**Status:** Honest assessment for master's thesis presentation — not all checks pass.

---

## Executive Summary

StableWalk computes gait and biomechanical metrics from **monocular MediaPipe pose estimation**. The calculation pipeline is implemented, tested at unit/integration level, and wired into the dashboard. Presentation has been polished with consistent **measured / estimated / derived / calculated** tier labeling.

**What is validated:**
- Core biomechanical orchestrator produces coherent outputs on demo pose data
- Results Summary, Biomechanics tab, and export paths share the same underlying `BiomechanicalAnalysisResult`
- Frame/playhead synchronization updates Biomechanics state and Results Summary together
- Walking speed is withheld when confidence or plausibility checks fail (e.g. athletic demo)

**What is not fully validated:**
- End-to-end GUI visual QA on every tab (slow; partially covered by unit tests)
- Clinical accuracy against force-plate or optical mocap gold standard (out of scope)
- Three failing stability-validity tests (classification drift vs. demo pose data)
- `final_scientific_validation.py` state-leakage segment crashes on MediaPipe timestamp error

---

## 1. Metric Validation Matrix

| Metric | Primary module | Tier | GUI tabs | Backend verified |
|--------|----------------|------|----------|------------------|
| **Walking Speed** | `walking_speed.py` → `advanced_gait_metrics.py` | estimated | Biomechanics, Results Summary | Yes — plausibility gate tested; athletic demo correctly unavailable |
| **Cadence** | `gait_cycle_analysis.py` → `advanced_gait_metrics.py` | calculated | Overview (gait cycle), Biomechanics, Results Summary | Yes — same `cadence_steps_per_min` source |
| **Symmetry** | `symmetry_metrics.py` | derived | Biomechanics, Results Summary | Yes — overall % index; **distinct** from Overview stance/swing ratios |
| **ROM** | `joint_rom.py` (Biomechanics); `knee_angle_chart.py` (Motion) | estimated | Biomechanics sidebar, Motion knee chart | Partial — same joint family; knee chart uses flexion convention, full ROM uses pose kinematics |
| **Center of Mass** | `com_estimation.py` | estimated | 3D overlay, Biomechanics chart, Results Summary | Yes — segment-weighted model; not force-plate COM |
| **Stability Margin** | `stability_margin.py` + `base_of_support.py` | derived | Biomechanics, Results Summary, 3D overlay | Yes — COM–BoS signed distance; frame-synced state |
| **Virtual GRF** | `estimated_vgrf_analysis.py` | estimated | Overview physics panel, Motion chart, Results Summary | Yes — pelvis acceleration proxy; disclaimers present |
| **Gait Events** | `foot_contact_analysis.py`, `gait_cycle_analysis.py` | derived | Overview cards, Motion timeline, Results Summary | Yes — HS/TO/double-support from contact FSM |
| **Video Quality** | `video_quality.py` | derived | Biomechanics, Results Summary | Yes — pose-visibility heuristics, not pixel QA |

---

## 2. Cross-Tab Consistency

### Shared session object
All tabs operate on the **same analyzed video** via shared session state in `app.py`:
- `_biomech_analysis`, `_foot_contact`, `_gait_cycle`, `_estimated_vgrf`
- `_build_analysis_summary()` assembles Results Summary from these objects
- Playhead updates call `_update_biomechanics_panel(frame_index=…)` and `_update_analysis_summary_panel(frame_index=…, timestamp_s=…)` together

### Verified numeric agreement (normal gait demo poses)

| Field | Biomechanics / cycles | Results Summary | Match |
|-------|----------------------|-----------------|-------|
| Cadence | 64.5 steps/min | 65 steps/min | Yes (rounding) |
| Symmetry | 83.84% | 84% | Yes (rounding) |
| Walking speed | 0.561 m/s | 0.56 m/s (Estimated) | Yes |
| Video quality | 95/100 | 95/100 | Yes |

Demo validation scripts (`scripts/validate_biomechanical_demos.py`, `scripts/validate_analysis_summary_demos.py`) ran successfully on `data/output/poses/` for abnormal, normal, and athletic clips.

### Intentional naming distinctions (post-polish)

| UI label | Score source | Notes |
|----------|--------------|-------|
| **Movement Stability (derived)** | `stability_scoring` domain group | Overview sidebar — pelvis/trunk/smoothness |
| **Gait Coordination (derived)** | `gait_analysis_summary` domain group | Overview sidebar — timing/clearance domains; *not* biomechanical composite |
| **Composite Gait Quality (derived)** | `gait_quality_score.py` | Biomechanics + Results Summary |
| **Stance ratio / Swing ratio** | `GaitTemporalMetrics` L/R ratios | Overview gait-cycle panel; *not* overall symmetry % |
| **Gait Symmetry (derived)** | `symmetry_metrics.py` overall index | Biomechanics + Results Summary |
| **Temporal Symmetry (derived)** | `stability_scoring` metric | Advanced tab only |

These are **different constructs** with distinct formulas. They are now labeled to avoid implying a single “symmetry” or “gait quality” number.

---

## 3. Timestamp Synchronization

| Component | Sync behavior | Status |
|-----------|---------------|--------|
| Overview video / skeleton | `_overview_video_frame_index` set on playhead | OK |
| Gait phase / contact cards | `_gait_contact_frame_index`, `_gait_phase_frame_index` | OK |
| Biomechanics stability state | Per-frame lookup by `frame_index` | OK |
| Biomechanics charts | `playhead_time_s` vertical marker | OK |
| Results Summary playhead line | `frame_index` + `timestamp_s` in summary | OK |
| COM/BoS 3D overlays | Frame cache keyed by `frame_index` | OK |

`tests/test_overview_frame_consistency.py` verifies frame-index alignment across overview subsystems when demo poses are available.

---

## 4. Automated Test Results

### Metric-focused suite (13 July 2026)

```
pytest tests/test_biomechanical_analysis.py
     tests/test_gait_cycle_analysis.py
     tests/test_foot_contact_analysis.py
     tests/test_virtual_grf.py
     tests/test_analysis_summary.py
     tests/test_overview_frame_consistency.py
     tests/test_knee_chart_interpretation.py
     tests/test_contact_clearance_consistency.py
     tests/test_stability_validity.py
     tests/test_gait_analysis_summary.py
```

**Result:** 70 passed, **3 failed** (`test_stability_validity.py`)

Failures indicate **validity tier classification drift** on demo data:
- `test_demo_normal_insufficient_data` — expects `INSUFFICIENT_DATA`, got `PROVISIONAL`
- `test_demo_athletic_provisional` — expects `PROVISIONAL`, got `INSUFFICIENT_DATA`
- `test_stability_result_includes_validity_dict` — normal demo status mismatch

These do **not** invalidate calculations; they indicate tests or thresholds need realignment with current validity logic.

### Additional passing areas
- Pipeline status diagram aggregation (`test_pipeline_status.py`)
- Scientific interpretation sentence bounds (`test_scientific_interpretation.py`)
- BoS visualization (`test_bos_visualization.py`)
- Dashboard notebook structure (`test_dashboard_notebook.py`)

---

## 5. Validated Calculations (by category)

### Calculated
- **Cadence** — mean inter-heel-strike interval → steps/min; requires ≥1 HS event
- **Gait phase %** — contact FSM macro phases
- **Double-support %** — bilateral contact frame fraction

### Estimated
- **Walking speed** — best of pelvis trajectory, cadence×step length, image-plane drift; gated 0.25–3.5 m/s, confidence ≥0.35
- **COM** — de Leva-style segment fractions along pose segments
- **Virtual GRF** — F = m(a_z + g) from pelvis vertical acceleration; split by contact
- **Joint ROM** — max − min pose-derived angles over clip
- **Foot clearance** — height above estimated floor plane (body-normalized scale)

### Derived
- **Gait symmetry index** — weighted L/R agreement (stance, swing, step, stride, ROM, cadence consistency)
- **Stability margin** — signed COM distance to BoS polygon; Stable / Reduced / Unstable thresholds
- **Composite gait quality** — weighted blend of stability score, margin, symmetry, cadence/cycle consistency
- **Video quality** — heuristic deductions from pose detection rate, jitter, visibility
- **Gait events** — HS/TO from contact edges; double support from bilateral contact
- **Movement stability / gait coordination** — domain-group scores from `stability_scoring`

### Measured
- **None from video alone.** OpenSim IK outputs are experimental mapped-marker motion, not clinical mocap. Force plates are not integrated.

---

## 6. Known Assumptions

1. **Default stature 1.70 m** scales monocular displacements to meters for walking speed and clearance display.
2. **Hip-centered canonical frame** removes global translation; overground speed requires separate estimators.
3. **Vertical axis +Y** for COM, BoS projection, and vGRF.
4. **Adult walking speed** plausible range 0.25–3.5 m/s.
5. **Stability margin threshold** 0.04 m (~fraction of foot length at body scale).
6. **Contact detection** uses clearance + velocity hysteresis; not force-plate ground truth.
7. **OpenSim IK** uses direct + synthetic marker mapping (~20+ of 31 Gait2392 tasks).

---

## 7. Current Limitations

| Limitation | Impact |
|------------|--------|
| Monocular scale ambiguity | Athletic demo: walking speed correctly withheld |
| Short demo clips (≤120 frames in validation) | Low gait-cycle count; cadence/stability validity often `INSUFFICIENT_DATA` or `PROVISIONAL` |
| Pose jitter | Stability margin often shows low % stable frames even on “normal” demos |
| vGRF is kinematic proxy | Cannot validate against force plates; peak BW labels are qualitative |
| Two “gait quality” concepts | Now labeled **Gait Coordination** vs **Composite Gait Quality** |
| ROM not in Results Summary cards | Available in Biomechanics and Motion Analysis only |
| GUI visual QA tests | Slow; not run in this validation pass |
| Stability validity tests | 3/8 failing — needs threshold/test update |

---

## 8. Future Improvements

1. Realign `stability_validity` tests with current demo pose classifications.
2. Gold-standard comparison dataset (force plate + mocap) for walking speed and vGRF correlation bounds.
3. Per-cycle ROM in Results Summary export.
4. Unified symmetry glossary tooltip linking Overview ratios vs overall symmetry index.
5. Fix MediaPipe monotonic timestamp issue in `final_scientific_validation.py` leakage test.
6. Optional OpenSim IK angle overlay vs pose angles in Motion Analysis.
7. Automated cross-tab numeric regression test (GUI-free) on frozen pose fixtures.

---

## 9. Presentation Polish Applied (this review)

- Centralized tier badges via `scientific_labels.format_tier_badge()`
- Biomechanics sidebar: removed duplicate video-quality row and duplicated scientific interpretation
- Consistent **steps/min** units (replaced mixed `spm`)
- Overview: **Gait Coordination (derived)** vs Results **Composite Gait Quality (derived)**
- Gait-cycle panel: **Stance ratio / Swing ratio** (not “symmetry”)
- Exported reports: methodology preamble, grouped sections, pipeline diagram, tier definitions
- README: data-tier table and link to this report

---

## 10. Conclusion

StableWalk is **suitable for academic demonstration** as pose-based gait analysis research software, provided presenters:

1. State clearly that metrics are **estimated or derived from video pose**, not measured kinetics.
2. Explain the **dual score families** (movement/coordination domains vs biomechanical composite).
3. Treat low stability-margin percentages on short monocular clips as **expected sensitivity**, not proof of pathology.
4. Cite this report and the embedded tier definitions in thesis methodology.

**Validation verdict:** Calculations are implemented and internally consistent across tabs; clinical measurement claims are appropriately withheld. Remaining work is test maintenance, gold-standard benchmarking, and optional GUI visual regression — not core algorithm replacement.

---

*Generated as part of the StableWalk final review. Re-run `scripts/validate_biomechanical_demos.py` and `scripts/validate_analysis_summary_demos.py` after processing new videos.*
