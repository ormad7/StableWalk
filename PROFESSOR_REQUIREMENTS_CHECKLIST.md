# StableWalk — Professor Requirements Checklist

This document maps the **current StableWalk implementation** against the professor’s requirement:

> *The project should analyze walking stability using OpenSim. It should receive a walking video, extract frames, analyze the person’s movement, calculate joint coordinates, angles, degrees of freedom, and stability metrics, and present the results visually.*

**Legend**

| Status | Meaning |
| --- | --- |
| **Done** | Implemented and usable in the dashboard or pipeline without extra setup |
| **Partially done** | Implemented but limited, optional, environment-dependent, or not fully validated end-to-end |
| **Missing** | Not implemented or not usable in the current project |

**Review method:** Static inspection of `stablewalk/` (pipeline, analysis, OpenSim modules, Tk dashboard), plus user-verified OpenSim Demo IK and experimental StableWalk IK on Windows with conda OpenSim SDK.

**Review date:** June 2026

---

## Checklist

### 1. Video input from file or URL

**Status: Done**

- Local file: toolbar **File…** → `_browse_video()` → **Analyze** runs `run_gait_pipeline()`.
- URL: URL field, presets, custom URL dialog; `VideoReader` / `VideoProcessor` handle streamable URLs.

---

### 2. Frame extraction from video

**Status: Done**

- Frames are read and processed from video via OpenCV + MediaPipe.
- Optional legacy mode can cache JPG frames to disk. Default path processes frames in the pipeline without requiring manual extraction.

---

### 3. Pose detection from each frame

**Status: Done**

- MediaPipe Pose detects body landmarks per frame.
- Results stored in `PoseSequence` / `PoseFrame` with visibility and detection flags.
- Skeleton overlay rendered on the main video panel.

---

### 4. Joint coordinates per frame

**Status: Done**

- Per-frame keypoints and enriched positions in `PoseFrame`.
- 3D skeleton reconstruction via `pose/skeleton_3d.py` and `pose/enrichment.py`.
- Real-time position table shows X/Y/Z for selected body points.

**Note:** Coordinates are **estimated from monocular video**, not laboratory motion capture.

---

### 5. Joint angles per frame

**Status: Done**

- Computed in `stablewalk/pose/kinematics.py`; stored on `PoseFrame.joint_angles`.
- Shown in DOF tree, real-time table, selected-point cards, charts, and exported to `.mot`/JSON.

---

### 6. Degrees of freedom analysis

**Status: Done**

- Canonical DOF registry: `stablewalk/models/joint_registry.py`.
- 16-point multi-select checklist + skeleton click selection.
- Step preview, angle hooks, velocity readouts, and per-point inspection in the dashboard.

---

### 7. Real-time dashboard visualization

**Status: Done**

- Tkinter dashboard: video overlay, tables, charts, stability panel, session overview, playback controls.
- Real-time position table throttled during playback; frame/time slider synchronized with skeleton player.

---

### 8. Skeleton visualization

**Status: Done**

- Interactive 2D stick-figure skeleton with left/right color coding and selection highlights.
- Optional robot simulation panel.

---

### 9. 3D trajectory visualization

**Status: Done**

- Matplotlib 3D trajectory plot with center-of-mass path and per-joint trajectories.

---

### 10. Multi-point joint selection

**Status: Done**

- Multi-select via checklist, add-point dropdown, skeleton click toggle, remove/clear controls.
- Supports hips, knees, ankles, heels, toes, shoulders, elbows, wrists (16 points).

---

### 11. Stability score

**Status: Done**

- Transparent 0–100 score with **Stable / Moderate / Unstable** classification.
- Weighted metrics: gait symmetry (30%), ROM (25%), step consistency (25%), center of mass (20%).

**Important:** Score is computed from **MediaPipe pose kinematics**, not from OpenSim solver output.

---

### 12. Stability explanation

**Status: Done**

- Human-readable findings, primary issue, weighted scoring notes in `StabilityResult`.
- **Full explanation** button opens complete text.
- Designed to be explainable for academic review, not a black box.

---

### 13. OpenSim-compatible TRC export

**Status: Done**

- `export_trc_file()` writes standard OpenSim `.trc` (frame, time, marker X/Y/Z in mm).
- MediaPipe → OpenSim marker mapping (`L_KNEE`, `R_HIP`, …).
- Verified output: `data/output/opensim/walk_stream/walk_stream.trc` (17 markers).
- Works **without** the OpenSim SDK installed.

---

### 14. OpenSim-compatible MOT export

**Status: Done**

- `export_motion_file()` writes OpenSim Storage `.mot` (or `.csv` fallback).
- Verified output: `data/output/opensim/walk_stream/walk_stream.mot`.
- Works without the OpenSim SDK installed.

---

### 15. JSON biomechanical export

**Status: Done**

- `export_opensim_ready_json()` bundles markers, joint angles, metadata, stability report, selected DOF.
- Verified output: `data/output/opensim/walk_stream/walk_stream_opensim.json`.

---

### 16. Real OpenSim SDK availability

**Status: Done**

| What works | Notes |
| --- | --- |
| Safe optional import in `opensim_sdk.py` | `OPENSIM_AVAILABLE` + fresh probe |
| GUI shows **OpenSim SDK: Installed** | When conda `opensim` package is active |
| Startup logging | SDK detected, module path, version |

---

### 17. Loading `.osim` model

**Status: Done**

| What works | Notes |
| --- | --- |
| `load_opensim_model()` / `validate_opensim_model()` | Real `osim.Model` + `initSystem()` |
| GUI **Load .osim Model** + **Select Local Model** | File dialog + `models/opensim/` discovery |
| Gait2392 demo model | `Gait2392_Pipeline/subject01_simbody.osim` (39 markers) |

---

### 18. Running OpenSim inverse kinematics

**Status: Partially done** (Demo IK done; StableWalk IK experimental with partial mapping)

| Workflow | Status | Output |
| --- | --- | --- |
| **OpenSim Demo IK** | **Done** | `models/opensim/Gait2392_Pipeline/subject01_walk1_ik.mot` |
| **StableWalk IK (experimental)** | **Partially done** | `data/output/opensim/walk_stream/walk_stream_ik.mot` |
| Marker mapping | **Done** | `marker_mapping.json`, `marker_mapping_report.txt`, mapped TRC |
| No fake IK | **Done** | Success only when `*_ik.mot` exists on disk |

**StableWalk IK details:**

- Original TRC uses MediaPipe names (`L_KNEE`, `R_HIP`) — **0 direct matches** with Gait2392 model.
- Mapping renames 9 markers to OpenSim names (`L.ASIS`, `R.Heel`, …).
- IK runs via `stablewalk_setup_ik.xml` + `subject01_simbody.osim`.
- **Limitation:** only 9 of 31+ model markers are covered; thigh/shank/pelvis markers are missing from MediaPipe.

**Stability metrics still use MediaPipe**, not OpenSim IK output.

---

### 19. Clear explanation of what is MediaPipe and what is OpenSim

**Status: Done**

| Where | Status |
| --- | --- |
| **README.md** | MediaPipe = pose; OpenSim = export + real IK |
| **OPENSIM_STATUS.md** | Professor-facing explanation + two/three workflows |
| **OpenSim panel** | Separate Demo IK vs StableWalk IK statuses |
| **marker_mapping.json** / **marker_mapping_report.txt** | Documents name mapping honestly |

---

### 20. README explanation for professor

**Status: Done**

- `README.md` covers project goal, pipeline, stability methodology, MediaPipe vs OpenSim roles, export formats, OpenSim IK workflows, and limitations.
- Suitable as the primary written explanation for academic review.

---

## Summary scorecard

| # | Requirement | Status |
| --- | --- | --- |
| 1 | Video input from file or URL | **Done** |
| 2 | Frame extraction from video | **Done** |
| 3 | Pose detection from each frame | **Done** |
| 4 | Joint coordinates per frame | **Done** |
| 5 | Joint angles per frame | **Done** |
| 6 | Degrees of freedom analysis | **Done** |
| 7 | Real-time dashboard visualization | **Done** |
| 8 | Skeleton visualization | **Done** |
| 9 | 3D trajectory visualization | **Done** |
| 10 | Multi-point joint selection | **Done** |
| 11 | Stability score | **Done** |
| 12 | Stability explanation | **Done** |
| 13 | OpenSim-compatible TRC export | **Done** |
| 14 | OpenSim-compatible MOT export | **Done** |
| 15 | JSON biomechanical export | **Done** |
| 16 | Real OpenSim SDK availability | **Done** |
| 17 | Loading `.osim` model | **Done** |
| 18 | Running OpenSim inverse kinematics | **Partially done** |
| 19 | Clear MediaPipe vs OpenSim explanation | **Done** |
| 20 | README explanation for professor | **Done** |

**Counts:** 19 Done · 1 Partially done · 0 Missing (as standalone features)

---

## Final conclusion

### Is the current project enough for the professor’s OpenSim requirement?

**Yes for integration proof and export; partially yes for video-driven IK.**

StableWalk **fully satisfies** the overall project shape:

- Walking video in (file or URL)
- Frame processing and pose detection
- Joint coordinates, angles, and degrees of freedom
- Transparent stability score with plain-language explanation
- Professional visual dashboard

For the **OpenSim-specific** part:

| Claim | Accurate today? |
| --- | --- |
| “Exports valid OpenSim-compatible files (.trc, .mot, JSON)” | **Yes** |
| “OpenSim SDK installed and detected” | **Yes** |
| “Runs official OpenSim Demo IK end-to-end” | **Yes** |
| “Runs StableWalk IK on mapped MediaPipe TRC” | **Yes** (experimental) |
| “StableWalk IK is fully validated biomechanics” | **No** — partial marker mapping |
| “Analyzes walking stability **using OpenSim**” | **No** — stability is from **MediaPipe** |

The project **does not fake OpenSim execution**. Demo IK and StableWalk IK success both require a real `*_ik.mot` file on disk.

### What would improve StableWalk IK (future work)

1. Add more marker correspondences (knee/ankle → shank proxies) with documented uncertainty.
2. Model scaling (`ScaleTool`) for subject-specific geometry.
3. Compare OpenSim IK joint angles with MediaPipe angles in the report.
4. Optional: use OpenSim IK output to inform stability metrics (currently separate).

### One-sentence answer for the professor

> StableWalk delivers a complete video-to-gait-analysis dashboard with explainable stability metrics from MediaPipe, valid OpenSim-compatible exports, verified real OpenSim inverse kinematics via the official Gait2392 demo, and experimental StableWalk IK on mapped MediaPipe markers — with an honest acknowledgment that partial marker mapping limits biomechanical reliability.
