# StableWalk final acceptance — Task 6 of 6

**Date:** 2026-10-15  
**Role:** end-to-end QA (Python, computer vision, scientific integration)  
**Rule:** a tab that opens is not a method. PASS requires a performed test. This document does **not** declare the project fully validated.

**Interpreter used this run:** `C:\Users\ormad\AppData\Local\Programs\Python\Python310\python.exe`  
**Harness:** `python scripts/stablewalk_task6_acceptance.py`  
**Machine log:** `docs/_task6_acceptance_run.json`  
**Related pytest (this run):** 215 passed, **2 failed**, 3 warnings (198 s)

Automated GUI-row counts from the harness: **PASS 66, FAIL 3, NOT TESTED 4**. Several of those PASS rows are qualified below (screenshot grab, Compare GQ mismatch, collapsed Advanced buttons).

**Overall verdict:** the two demo videos load and analyze as independent sessions; Compare research export is independent and numerically consistent with the orchestrator for cadence; OpenSim **file** export is valid; Inverse Kinematics was **not** executed in this interpreter; 3D playhead refresh is too slow; screenshots did not capture the Tk window. Core scientific and OpenSim-IK requirements remain incomplete. **Do not treat StableWalk as fully accepted.**

---

## Tests actually executed

| ID | Scope | How it was run | Result |
|----|--------|----------------|--------|
| TEST 1 | Video analysis of `normal_gait.mp4` then `abnormal_gait.mp4` | Opened with OpenCV; full pipeline from cached pose JSON (`fresh` GUI loads + headless orchestrator); object-id isolation + repeat of normal after abnormal | **PASS** for cached-pose analysis / isolation. **NOT TESTED:** MediaPipe re-estimation of the MP4s |
| TEST 2 | Compare mode | `try_build_demo_snapshot`, `SessionCompareCache`, `compare_independent_clips` export, live Compare tab | **PASS** independence / IDs / export. **PARTIAL** GUI CompareMetrics vs orchestrator |
| TEST 3 | OpenSim | SDK probe, TRC/MOT write+parse, `run_inverse_kinematics_if_available`, ID readiness | **PASS** export + honest Partial. **NOT TESTED:** live IK / residuals / solved-frame coverage |
| TEST 4 | Advanced | Workspace categories, mapped Session/OpenSim controls, inspector Toplevel, joint-data dialog, generated-files scan | **PASS** Session/OpenSim/inspector. **PARTIAL** collapsed Real-to-Sim / Exports / Reports; GUI OpenSim export files dict stayed `None` |
| TEST 5 | Six tabs, playback widgets, resize, load | Real `StableWalkGUI` | **PASS** for tab presence, mapping, 1280×720 resize, sequential loads. **FAIL** screenshot evidence (wrong window) |
| TEST 6 | Performance | Wall-clock analysis, `show_pose_at` loop, tracemalloc | **PASS** analysis time / memory. **FAIL** pose-refresh rate |
| TEST 7 | Session traceability | `SessionMetricView` vs biomech/compare; live GUI source | **PASS** headless view vs orchestrator. **FAIL** `_current_source` empty after Advanced/inspector |
| TEST 8 | This report | Written from the JSON + pytest + artifacts of this run | — |

---

## TEST 1 — Video analysis

Containers opened:

| File | FPS | Frames (container) | Pose frames | Size | First frame |
|------|-----|--------------------|-------------|------|-------------|
| `data/demo_videos/normal_gait.mp4` | 25.00 | 120 | 120 | 3840×2160 | OK |
| `data/demo_videos/abnormal_gait.mp4` | 29.97 | **150** | **120** | 480×360 | OK |

`abnormal_gait` pose JSON covers 120 / 150 container frames. Analysis used the pose recording, not a silent pad of the extra 30 frames.

Headless pipeline (`load_pose_sequence` → `pose_sequence_to_gait_motion` → cycles, spectral, asymmetry3d, `run_biomechanical_analysis`):

| Clip | Q | Cadence (spm) | Cycles | Symmetry % | Time | Provenance |
|------|--:|--------------:|-------:|-----------:|-----:|------------|
| normal_gait | **80.72** | **64.52** | 2 | 86.3 | 2.59 s | `inferred_monocular_3d` |
| abnormal_gait | **70.64** | **51.87** | 2 | 76.9 | 1.45 s | `inferred_monocular_3d` |

Sequential analyses used **distinct** Python objects (`object_alias=False`). Repeating `normal_gait` after `abnormal_gait` reproduced Q **80.719934985081** and cadence **64.5161**. GUI `load_poses(..., fresh=True)` sources: `normal_gait.mp4` then `abnormal_gait.mp4` (did not keep the first path).

Live GUI loads: 21.0 s / 17.3 s (includes Tk bind). Overview video, skeleton, and 3D path widgets mapped for both.

**NOT TESTED:** running MediaPipe on the MP4s again.

Gait-cycle warnings (logged, not hidden): DS+SS+flight **51.9%** (normal) and **70.6%** (abnormal) — not 100%. Contact-fraction warning on one pass.

---

## TEST 2 — Compare mode

Independent snapshots: keys `normal` / `abnormal`, distinct `sequence` ids, filenames `normal_gait.mp4` / `abnormal_gait.mp4`. Joint dictionaries are the same 22 names (registry 21 + pelvis/root). t0 `left_knee` XYZ differs (A `[0.089, -0.812, -0.009]`, B `[0.115, -0.554, 0.057]`, body-normalized). Compare slots `state=ready`.

Research compare (`compare_independent_clips`, Δ = B − A), exported:

- `docs/task6_acceptance/compare_normal_vs_abnormal.json` (6741 B)
- `docs/task6_acceptance/compare_normal_vs_abnormal.md`

| Metric | A | B | Δ | Status |
|--------|--:|--:|--:|--------|
| Cadence (spm) | 64.52 | 51.87 | **−12.64** | Matches orchestrator |
| Dominant frequency (Hz) | 0.417 | 0.500 | 0.083 | Matches Task 5 |
| Spectral entropy | 0.695 | 0.800 | 0.105 | Matches Task 5 |
| ACF stride lag | — | — | — | Honest unavailable |
| Phase-aligned knee RMSE | — | 2.38° | aligned | Cycles resampled 0–100% |

`independent=True`. Missing ACF fields did **not** become 0.

**PARTIAL / defect:** GUI `compare_session_metrics` on demo snapshots reported `gait_quality` Δ **−19.05** and **cadence_delta=None**, while the orchestrator ΔQ is **−10.08** and cadence exists on both clips. `tests/test_session_compare.py::test_load_or_analyze_demo_session_is_compare_ready` failed: `stability_margin_m is None`. Compare **research** export is the numerically correct path; the **session-cache CompareMetrics** path is not identical to Overview/Biomech.

Live Compare tab mapped after both loads; no freeze in this harness.

---

## TEST 3 — OpenSim

| Check | Result | Evidence |
|-------|--------|----------|
| SDK availability | **NOT TESTED as installed** (probe **PASS**) | `available=False` — this Python 3.10 env. Compatible export only |
| Model on disk | **NOT TESTED** | `find_usable_opensim_model()` empty in this env |
| TRC validity | **PASS** | PathFileType, 120 frames, **17 markers**, mm, rows match, 70442 B (`docs/task6_acceptance/opensim_normal_gait/normal_gait.trc`) |
| MOT validity | **PASS** | `nRows=120`, `nColumns=16`, `inDegrees=yes`, time first (`normal_gait.mot`) |
| Actual IK | **NOT TESTED** | `run_inverse_kinematics_if_available` → `ran=False`, message *OpenSim SDK not installed — compatible export only* |
| Frame / marker coverage (export) | **PASS** (export only) | 120/120 TRC rows; 17 mapped markers. Not a full Gait2392 markerset |
| IK residuals | **NOT TESTED** | No completed `InverseKinematicsTool` |
| IK status honesty | **PASS** | Resolved **Partial**, not Complete. File presence did not upgrade the label |
| Inverse Dynamics | **PASS** (blocked) | `ready_for_traditional_id=False`; GRF / ExternalLoads absent. No forces invented |

An older terminal log (2026-10-14) in another environment printed “IK completed successfully” with an `_ik.mot`. **That is not this run.** This Task 6 interpreter has no SDK.

---

## TEST 4 — Advanced

Categories present: Session, OpenSim, Real-to-Sim, Exports, Reports.

| Action | Result |
|--------|--------|
| Save Session button | Mapped **PASS** (folder dialog not driven — would block) |
| View Joint Data | Dialog invoked **PASS** |
| 3D Coordinate Inspector | `Toplevel` opened **PASS** |
| Generated Files scanner | **PASS** (scan API) |
| Export OpenSim Files | Button mapped; quiet invoke `files=None` — **PARTIAL** (async not awaited) |
| Real-to-Sim / Exports / Reports | Widgets exist; **not mapped** while sections are collapsed — expected chrome, not a click test of the pipelines |
| Real-to-Sim pipeline execution | **NOT TESTED** (no RL/Isaac; requirement 5 out of scope) |

---

## TEST 5 — UI

Six tabs present and selectable: Overview, Motion Analysis, Biomechanics, Results, Compare, Advanced. Navigation walk did not freeze. Play + timeline mapped. Motion chart host mapped after analysis. Window **1280×720** applied.

**Screenshot FAIL:** `ImageGrab` of the Tk `winfo_root*` box captured the **Cursor IDE**, not StableWalk (same image for Overview and Compare). Visual proof of skeleton/3D/sync is therefore **NOT TESTED** as a photograph. Widget `winfo_ismapped` is the evidence that exists.

---

## TEST 6 — Performance

| Measurement | Value | Status |
|-------------|-------|--------|
| Headless analysis | 2.59 s / 1.45 s (cached poses) | **PASS** |
| GUI `load_poses` | 21.0 s / 17.3 s | Recorded (includes Tk) |
| GUI launch | 9.03 s | Recorded |
| `show_pose_at` 12 frames | 7.95 s → **1.5** updates/s (normal); 6.70 s → **1.8**/s (abnormal) | **FAIL** vs interactive playback |
| tracemalloc peak (repeat analysis) | 12.2 MiB | **PASS** (analysis only, not GUI retained bitmaps) |

The 1.5 updates/s figure is wall time inside `_show_pose_at` + a 20 ms pump, not video decode FPS. Per-frame 3D/skeleton work is still too heavy for smooth playback on this host.

Pytest `test_perf_caching_and_lazy_paint` / `test_ui_playback_performance` / `test_gui_memory` passed in the 215-pass bundle.

---

## TEST 7 — Research traceability

Headless `SessionMetricView` for `normal_gait` matches the orchestrator: Q **80.7**, cadence **64.5**, symmetry **86.3**, cycles **2**. `analysis_confidence` in the view is **unavailable (—)** while cycle sufficiency is `"none"` (no sufficiency failure) — two different flags.

`abnormal_gait` view: Q **70.6**, cadence **51.9**, symmetry **76.9**.

Live GUI after loading abnormal then opening Advanced/inspector: `live_Q=72.38` (same session family as abnormal ~70.6, not normal 80.7) but `_current_source` was **empty**. **FAIL** for the source-string check. Analysis object still held a finite Q.

Step length / speed were **non-null** (0.110 m / 0.561 m/s normal; 0.133 m / 0.907 m/s abnormal). These are **stature-scaled monocular estimates**, not lab metres. They were not zero-filled from missing data.

---

## Pytest executed this pass

```
python -m pytest tests/test_compare_workflow.py tests/test_session_compare.py
  tests/test_compare_loading.py tests/test_opensim_ik_status.py
  tests/test_opensim_id_readiness.py tests/test_advanced_workspace.py
  tests/test_dashboard_advanced.py tests/test_generated_files_panel.py
  tests/test_dashboard_notebook.py tests/test_metric_consistency.py
  tests/test_research_pipeline_independence.py tests/test_perf_caching_and_lazy_paint.py
  tests/test_ui_playback_performance.py tests/test_gui_memory.py
  tests/test_playback_sync.py tests/test_real_video_gait_validation.py
  tests/test_advanced_tab_open.py tests/test_chart_interactions.py
  tests/test_unavailable_labels.py -q
```

**215 passed, 2 failed:**

1. `test_session_compare.py::test_load_or_analyze_demo_session_is_compare_ready` — `stability_margin_m is None` on the abnormal demo snapshot.  
2. `test_dashboard_notebook.py::test_tab_switch_stress_singletons` — `_comparison_mode is None` after 50 tab switches (Compare controller not created).

---

## Error / warning log (this run)

Harness `errors: []` (no uncaught exception after the Vec3/summary signature fixes in the script).

Logged scientific warnings:

- `Double support + single support + flight is 51.92% / 70.63% / 72.77%, not approximately 100%.`
- `94% of foot-frames are in contact (>85%) — floor may be set too high or swing clearance is underestimated.`
- `Heel strikes present but no complete gait cycles segmented` / `Only 0 complete same-side …` (transient on a short/failed partition, not the 2-cycle demos’ final cadence).

---

## Screenshots

Attempted paths:

- `docs/task6_acceptance/screenshots/overview_normal_gait.png`
- `docs/task6_acceptance/screenshots/overview_abnormal_gait.png`
- `docs/task6_acceptance/screenshots/compare_tab.png`

**Status: FAIL as product evidence.** PIL `ImageGrab` recorded the IDE, not the Tk dashboard. Do not cite these PNGs as proof of skeleton, 3D, or Compare visuals.

---

## Unresolved scientific limitations

1. Monocular depth is inferred (`inferred_monocular_3d`). Spatial metres are stature-scaled estimates.  
2. Demo clips are ~4–5 s / 2 cycles; ACF stride remains unavailable (Task 5).  
3. Contact partition ≠ 100%.  
4. OpenSim IK **not run** in this Python 3.10 process; Inverse Dynamics blocked (no measured GRF).  
5. Requirement 5 (RL / Isaac) unimplemented.  
6. MediaPipe was not re-run on the MP4s.  
7. GUI CompareMetrics GQ/cadence do not match the orchestrator on demo snapshots.  
8. `analysis_confidence` can be unavailable in `session_view` while cadence is published.

---

## Remaining GUI defects

1. Pose/3D `show_pose_at` ≈ **1.5–1.8** GUI updates/s for 12 cached frames.  
2. Tab-switch stress: Compare controller stays `None` (`test_tab_switch_stress_singletons`).  
3. `_current_source` empty after Advanced + inspector even though `gait_motion` is loaded.  
4. Screenshot grab of `winfo_root*` does not isolate the Tk window on this desktop.  
5. Demo snapshot `stability_margin_m` None → Compare-ready test fails.  
6. Collapsed Advanced sections: Real-to-Sim / Exports / Reports buttons not mapped until expanded (by design; click-paths not exercised).  
7. Quiet OpenSim GUI export did not populate `_opensim_export_files` within the pump window.

---

## Exact files changed (Task 6)

No application redesign. Task 6 added the harness, artifacts, and this report:

| Path | Role |
|------|------|
| `scripts/stablewalk_task6_acceptance.py` | E2E harness (new) |
| `docs/_task6_acceptance_run.json` | Machine-readable results |
| `docs/task6_acceptance/compare_normal_vs_abnormal.json` | Compare export |
| `docs/task6_acceptance/compare_normal_vs_abnormal.md` | Compare export |
| `docs/task6_acceptance/opensim_normal_gait/normal_gait.trc` | TRC evidence |
| `docs/task6_acceptance/opensim_normal_gait/normal_gait.mot` | MOT evidence |
| `docs/task6_acceptance/opensim_normal_gait/normal_gait_opensim.json` | OpenSim JSON bundle |
| `docs/task6_acceptance/screenshots/*.png` | Unusable IDE grabs |
| `docs/stablewalk_final_acceptance.md` | This report |

The git working tree also contains earlier Task 1–5 work (metrics, spectral, skeleton, layout, etc.). Those are **not** Task 6 product changes.

---

## Stop

Task 6 of 6 is the acceptance record above. **The project is not fully validated:** MediaPipe re-run, live OpenSim IK, Inverse Dynamics, requirement 5, GUI CompareMetrics identity, and playable 3D refresh remain open.
