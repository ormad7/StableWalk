# StableWalk final acceptance report

**Date:** 2026-10-13  
**Purpose:** End-to-end quality assurance and scientific validation for an M.Sc. research demonstration.  
**Rule:** a tab that opens is not proof of a method. PASS requires a performed test. This document does not claim clinical validity, force-plate stability, OpenSim Inverse Dynamics, or requirement 5 (RL / Isaac).

**Verdict:** The desktop application launched, every main tab and inner analysis tab was exercised, cached pose sessions for `normal_gait.mp4` and `abnormal_gait.mp4` were analysed, COM–BoS is not reported as reduced stability, exported summary JSON matches the Overview stability card, and automated suites used in this pass completed without failures. Remaining gaps are listed as **NOT TESTED**, not as silent passes.

---

## Environment (actually used)

| Item | Value |
| --- | --- |
| Host | Windows, conda env `stablewalk-opensim` |
| Interpreter | `C:\Users\ormad\anaconda3\envs\stablewalk-opensim\python.exe` |
| Screen | **1536 × 864** (Tk `winfo_screenwidth/height`) |
| Tk DPI scale | **1.00** (process scale factor; OS 125/150/175% was not changed) |
| GUI launch | `StableWalkGUI` via `python scripts/final_acceptance_qa.py` (real Tk, then destroyed) |
| Machine-readable GUI log | `docs/_acceptance_run.json` |

Live GUI counts after defect retest: **PASS 62, FAIL 0, NOT TESTED 1** (OS DPI setting).

---

## What was executed

1. Inventory of demo videos and pose JSON on disk.
2. Launch of the real Tk desktop GUI (not a mock).
3. Selection of every main tab: Overview, Motion Analysis, Biomechanics, Results, Compare, Advanced.
4. Selection of every Motion inner tab: Kinematics, Spectral Analysis, Spectral Features, Asymmetry.
5. Selection of every Biomechanics inner tab: COM, Ground Reaction Force, Balance, Contact, Gait Phase.
6. Logical reflow at 1366×768, 1600×900, 1920×1080, 2560×1440, 3840×2160. Physical geometry was clamped to **1528×816** whenever the requested size exceeded the 1536×864 display.
7. Load of cached poses for `normal_gait.mp4` and `abnormal_gait.mp4` (`load_poses(..., fresh=True)` — MediaPipe was not re-run).
8. Scientific pytest on those pose files (ACF, spectral export, features, 3D provenance, asymmetry, joint-angle convention, gait events, COM–BoS honesty, session isolation).
9. Invalid / missing / extremely short video gates used by the GUI (`check_source_format`, `quick_validate_source`).
10. Automated suites listed under [Test commands](#test-commands).

---

## Videos actually available

| File | Role in this pass |
| --- | --- |
| `data/demo_videos/normal_gait.mp4` | Opened (cv2); analysed from `normal_gait_poses.json` in GUI + pytest |
| `data/demo_videos/abnormal_gait.mp4` | Opened (cv2); analysed from `abnormal_gait_poses.json` in GUI + pytest |
| `data/demo_videos/athletic_walking.mp4` | File present; pytest opens container if present; **not** loaded in the GUI session |
| `data/demo_videos/candidates/weightgait_limping_gait.mp4` | Extremely short: **15 frames, 2.14 s** |
| Other candidate / backup MP4s | Present on disk; **not** analysed in this pass |

Filenames are **not** treated as ground truth. `abnormal_gait.mp4` is a research demo clip, not a diagnosed pathology label.

Pose-level analysis used existing JSON under `data/output/poses/`. Full MediaPipe re-estimation of the MP4s was **not** repeated in this session.

---

## Defects found, corrected, and retested

New product features were not added. The following are confirmed defects required for scientific honesty or for an existing supervisor tab to show its results.

### F1 — Unreportable COM–BoS could still be exported as reduced stability

| | |
| --- | --- |
| **Status before fix** | FAIL (Results / interpretation / PDF) |
| **Symptom** | Overview already showed **Not available** when `stability_margin.reportable is False`, but `build_analysis_summary` still wrote `"{state} ({stable_pct}% stable frames)"` as an available KPI. `build_scientific_interpretation` then produced “markedly reduced stability” / reduced-margin wording from that percentage. The session PDF dumped **Stable-frame %** from the same object. |
| **Root cause** | `reportable=False` was honoured on the Overview card and Balance enrichment, but not on the Results summary builder, scientific-interpretation paragraph, or PDF stability table. |
| **Changes** | `stablewalk/analysis/analysis_summary.py` — unreportable margin → `Not available`, `available=False`. `stablewalk/analysis/scientific_interpretation.py` — do not parse `stable_pct` or dominant Unstable/Reduced labels when unreportable. `stablewalk/io/session_pdf_report.py` — print reportability note instead of Stable-frame %. |
| **Retest** | `tests/test_analysis_summary.py::test_unreportable_com_bos_is_not_exported_as_stability_kpi` PASS. `tests/test_scientific_interpretation.py::test_unreportable_monocular_stability_is_not_a_reduced_conclusion` PASS. `tests/test_final_scientific_acceptance.py` COM–BoS + export tests PASS. GUI: Overview **Not available**, export JSON `available=False` / `Not available` for both demo clips. |

### F2 — Spectral Analysis and Asymmetry reports never bound in the live GUI

| | |
| --- | --- |
| **Status before fix** | FAIL (GUI bind) |
| **Symptom** | First live GUI run: `Spectral report bound … available=None canvas=False`; Asymmetry `provenance=None`. Pytest `analyze_spectral` / `analyze_asymmetry3d` on the same pose files succeeded. |
| **Root cause** | Background workers called `gui.root.after(0, apply)` from a non-Tk thread. On this Windows/Tk build that callback did not apply, so `_spectral_report` / `_asymmetry_report` stayed `None`. |
| **Changes** | Workers now only store `(token, report, err)` on the GUI object. A main-thread `after(50, drain)` poller copies the result and renders. |
| **Retest** | GUI retest: Spectral `available=True canvas=True` on both clips; Asymmetry `provenance=inferred_monocular_3d` on both clips. |

### F3 — Resize harness false FAIL on Motion while Asymmetry was selected

| | |
| --- | --- |
| **Status** | Harness error, not a product clip/overlap bug |
| **Symptom** | `_collect_issues(..., tab="Motion")` requires `motion_graph_host` (Kinematics). After walking inner tabs the notebook was left on Asymmetry, so the host was unmapped at every size. |
| **Change** | Acceptance script selects Kinematics before the Motion clip/overlap check, and treats Spectral empty-state as valid when no report exists yet. |
| **Retest** | All five requested sizes PASS (`docs/_acceptance_run.json`). |

---

## Feature table

Status key: **PASS** = test was performed and met the criterion. **FAIL** = performed and did not (none remain after retest). **NOT TESTED** = not performed in this environment.

### Desktop shell and navigation

| Feature | Status | Evidence |
| --- | --- | --- |
| Launch desktop GUI | PASS | `StableWalkGUI` created; screen 1536×864; Tk DPI 1.00 |
| Main tabs present | PASS | Overview, Motion Analysis, Biomechanics, Results, Compare, Advanced |
| Select Overview | PASS | mapped 1346×503 |
| Select Motion Analysis | PASS | mapped |
| Select Biomechanics | PASS | mapped |
| Select Results | PASS | mapped |
| Select Compare | PASS | mapped |
| Select Advanced | PASS | mapped; workspace mapped |
| Motion / Kinematics | PASS | inner notebook select |
| Motion / Spectral Analysis | PASS | select + later bound report |
| Motion / Spectral Features | PASS | select + canvas mapped after load |
| Motion / Asymmetry | PASS | select + `inferred_monocular_3d` |
| Biomechanics / COM | PASS | inner select + panel 824×428 after load |
| Biomechanics / Ground Reaction Force | PASS | same |
| Biomechanics / Balance | PASS | same; copy is experimental / unreportable |
| Biomechanics / Contact | PASS | same |
| Biomechanics / Gait Phase | PASS | same |
| File / View / Help menus | PASS | also Simulation, Demo |
| Play + timeline | PASS | mapped on Overview |
| GUI did not freeze during reflow | PASS | reflow wall 1.03 s on retest |

### Layout, resize, plots, video, 3D

| Feature | Status | Evidence |
| --- | --- | --- |
| Resize 1366×768 | PASS | physical window fit; no clip/overlap issues from the layout collector |
| Resize 1600×900 | PASS | logical reflow; **physical clamp 1528×816** |
| Resize 1920×1080 | PASS | logical reflow; physical clamp 1528×816 |
| Resize 2560×1440 | PASS | logical reflow; physical clamp 1528×816 |
| Resize 3840×2160 | PASS | logical reflow; physical clamp 1528×816 |
| Windows OS DPI 125 / 150 / 175% | NOT TESTED | process scale remained 1.00; display scaling was not changed in Settings |
| Physical 4K pixel desktop | NOT TESTED | no 3840×2160 monitor; only logical width/height were applied |
| Unintentional clipping / overlap (automated) | PASS | `_collect_issues` at each size after selecting Kinematics for Motion |
| Matplotlib legend-on-data overlap (human) | NOT TESTED | no screenshot / ImageGrab assertion |
| Plots readable (widget mapped, size floor) | PASS | kinematics host, spectral canvas after bind, features canvas, biomech panel |
| Video contain-fit / aspect ratio | PASS | letterbox math `min(box_w/src_w, box_h/src_h)` for 16:9, 1:1 box, and 9:16 sources; Overview video panel mapped after load |
| 3D views resize | PASS | Overview skeleton + 3D path mapped after both clips |
| Scrolling (automated) | PASS | layout collector allows biomech rail scroll; no unexpected nested page scrollbars flagged |
| Dual-session Compare with two analysed videos | NOT TESTED | Compare tab and empty workspace were exercised; A/B analyse-and-diff was not run |
| Before/after screenshots | NOT TESTED | none captured |

Automated layout regression (separate from the live acceptance script): `tests/test_layout_scale.py`, `test_responsive_layout_regression.py`, `test_responsive_window_sizes.py`, `test_overview_layout.py`, `test_motion_analysis_tabs.py`, `test_biomech_layout.py`, `test_results_layout.py`, `test_compare_layout.py`, `test_dashboard_advanced.py` — included in the 114 passed / 2 skipped run below. Two skips are existing Tk init/`skipTest` cases, not silent passes.

### Video handling

| Feature | Status | Evidence |
| --- | --- | --- |
| `normal_gait.mp4` opens | PASS | cv2 open; fps>1; frames≥2 |
| `abnormal_gait.mp4` opens | PASS | same |
| `athletic_walking.mp4` opens | PASS | pytest `test_athletic_walking_opens_if_present` |
| Missing path rejected | PASS | `check_source_format` “Video file not found”; GUI `_load_video` stayed alive, `processing=False` |
| Invalid MP4 container | PASS | junk bytes → “Could not open video source.” |
| Extremely short clip | PASS | limp candidate 15 frames / 2.14 s; still readable |
| Low-quality (blur / occlusion / labelled poor video) | NOT TESTED | no dedicated low-quality fixture was scored as “low quality” in this pass |
| Live MediaPipe re-run on demo MP4s | NOT TESTED | GUI used cached `*_poses.json` |

### Session analysis, isolation, confidence, exports

| Feature | Status | Evidence |
| --- | --- | --- |
| Load `normal_gait` poses | PASS | `load_poses=True`; recording present |
| Load `abnormal_gait` poses | PASS | same |
| Session isolation | PASS | sources switched from `normal_gait.mp4` to `abnormal_gait.mp4`; spectral objects are distinct in pytest |
| Overview stability (normal) | PASS | **Not available** |
| Overview stability (abnormal) | PASS | **Not available** |
| Unreportable stability ≠ reduced | PASS | neither clip showed `%` or “reduced” on the stability card |
| Analysis confidence (normal) | PASS | **89%** (tracking/analysis confidence, not a stability claim) |
| Analysis confidence (abnormal) | PASS | **Insufficient** |
| Results JSON vs UI stability | PASS | both clips: UI `Not available`, export `Not available`, `available=False` |
| PDF binary vs on-screen widgets | NOT TESTED | JSON summary path was compared; a full PDF pixel/text dump vs the Results tab was not diffed in the GUI |
| Gait-cycle warnings on demo poses | PASS (honesty) | engine logged incomplete cycles / contact-sum ≠ 100% — cadence is not fabricated |

### Scientific methods (supervisor requirements 1–4)

| Feature | Status | Evidence |
| --- | --- | --- |
| Autocorrelation ρ(0) = 1 | PASS | `test_autocorrelation_lag0_is_one_on_real_pose`; `test_normalized_acf_matches_numpy_correlate` |
| Spectral analysis (Welch/FFT, R(0), eigh) | PASS | `tests/test_spectral_analysis.py`; GUI Spectral `available=True` on both clips |
| Feature extraction / session vector | PASS | `SESSION_VECTOR_KEYS` present; export JSON `session_vector` equals in-memory dict; features warn they are not disease labels |
| 3D reconstruction provenance | PASS | Asymmetry kinematics `inferred_monocular_3d`; reconstruction assumptions remain monocular (not mocap) |
| Gait asymmetry (Robinson SI / ASI) | PASS | `tests/test_asymmetry3d.py` definitions + synthetic detectability; real pose report built and JSON export matches `to_dict` provenance |
| Joint angle convention | PASS | interior 180° → flexion 0°; real `left_knee_flexion` samples in (−30°, 180°) |
| Gait event timing | PASS | sufficiency tests; when ≥2 events exist they are time-sorted; HS interval gates when present |
| COM–BoS not a physical stability KPI | PASS | `reportable is False` on real `normal_gait` poses; Overview N/A; gait quality must not treat margin as a confident reduced-stability conclusion |
| OpenSim Inverse Dynamics with measured GRF | NOT TESTED | ID remains blocked without measured GRF (unit tests of the gate only) |
| Subject-specific OpenSim ScaleTool | NOT TESTED | not run |
| Requirement 5 RL / Isaac simulator | NOT TESTED | **not implemented** |

---

## Measured demo-session notes (not diagnoses)

These are observations from **this** cached-pose run. They are not clinical findings and do not imply that `normal_gait.mp4` is “stable”.

| Clip | Overview stability | Confidence | Spectral GUI | Asymmetry provenance |
| --- | --- | --- | --- | --- |
| `normal_gait.mp4` | Not available | 89% | available=True | `inferred_monocular_3d` |
| `abnormal_gait.mp4` | Not available | Insufficient | available=True | `inferred_monocular_3d` |

Engine messages during GUI load included incomplete gait-cycle segmentation and contact fractions that do not sum to 100%. That is consistent with monocular contact estimation. Cadence and similar cycle metrics must remain unavailable when sufficiency fails — they were not forced to a demo-friendly number.

---

## Test commands

```text
python scripts/final_acceptance_qa.py
→ PASS 62, FAIL 0, NOT TESTED 1

python -m pytest tests/test_final_scientific_acceptance.py tests/test_analysis_summary.py tests/test_scientific_interpretation.py tests/test_biomech_chart_enrichment.py tests/test_overview_clinical_summary.py -q --tb=short
→ 53 passed

python -m pytest tests/test_wire_model_inventory.py tests/test_spectral_analysis.py tests/test_spectral_features.py tests/test_spectral_workspace.py tests/test_asymmetry3d.py tests/test_asymmetry3d_moments.py tests/test_asymmetry_workspace.py tests/test_scientific_gait_validation.py tests/test_opensim_id_readiness.py tests/test_opensim_ik_status.py tests/test_gait_event_sufficiency.py tests/test_real_video_gait_validation.py tests/test_knee_chart_interpretation.py tests/test_metric_consistency.py tests/test_session_pdf_report.py -q --tb=line
→ 102 passed, 1 skipped

python -m pytest tests/test_layout_scale.py tests/test_responsive_layout_regression.py tests/test_responsive_window_sizes.py tests/test_overview_layout.py tests/test_motion_analysis_tabs.py tests/test_biomech_layout.py tests/test_results_layout.py tests/test_compare_layout.py tests/test_dashboard_advanced.py tests/test_final_scientific_acceptance.py tests/test_analysis_summary.py tests/test_scientific_interpretation.py tests/test_biomech_chart_enrichment.py -q --tb=line
→ 114 passed, 2 skipped
```

No test was edited to inflate `normal_gait.mp4` stability. Joints and DOFs remain distinct (requirement 1 inventory tests).

---

## What this demonstration may and may not claim

**May claim**

- A working Tk research GUI with the listed tabs, responsive reflow on the tested logical sizes, and cached-pose analysis of the two demo walking clips.
- Signal-processing products (ACF, PSD, spectral features) derived from pose time series, with export matching in-memory values.
- Monocular 3D / left–right asymmetry labelled `inferred_monocular_3d`, not calibrated mocap.
- Honest unavailability of physical COM–BoS stability from monocular video.

**Must not claim**

- Clinical diagnosis, reduced balance, or force-plate margin of stability.
- That `normal_gait.mp4` has high physical stability (the KPI is **Not available**).
- That OpenSim Inverse Dynamics or ScaleTool was executed on a subject.
- That requirement 5 (RL simulator) exists in the product.
- That 4K or Windows 150% DPI was visually certified on this machine.
- That Compare was validated with two fully analysed live sessions.

---

## Files touched for confirmed defects

- `stablewalk/analysis/analysis_summary.py`
- `stablewalk/analysis/scientific_interpretation.py`
- `stablewalk/io/session_pdf_report.py`
- `stablewalk/ui/tk/dashboard_spectral.py`
- `stablewalk/ui/tk/dashboard_asymmetry.py`
- `tests/test_analysis_summary.py`
- `tests/test_scientific_interpretation.py`
- `tests/test_final_scientific_acceptance.py`
- `scripts/final_acceptance_qa.py`

Supporting honesty work from the same research line (Balance tab copy / enrichment) remains in `dashboard_biomechanics.py` and `biomech_chart_enrichment.py`.
