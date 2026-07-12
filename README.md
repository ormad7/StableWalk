# StableWalk

**Analyzing walking stability from video using MediaPipe pose estimation and an OpenSim-compatible biomechanical layer.**

StableWalk is a final-year Computer Science project that takes an ordinary
walking video and determines whether the walking pattern is **stable**,
**moderate**, or **unstable**. It combines modern computer-vision pose
estimation (MediaPipe) with a biomechanics-oriented representation layer
(OpenSim) so the extracted human motion can be analyzed, visualized, and
exported in formats used by the biomechanics community.

---

## 1. Project goal

The goal of StableWalk is to **quantify and explain walking stability** from a
single monocular video. Given a clip of a person walking, the system:

- detects the human skeleton frame by frame,
- computes joint coordinates, joint angles, and degrees of freedom,
- derives transparent, explainable stability metrics, and
- presents everything in an interactive desktop dashboard.

The stability result is intentionally **not a black box**: every score is built
from simple, documented biomechanical formulas with plain-language reasons.

---

## 2. Current pipeline

```
Video (file or URL)
      │
      ▼
Frame extraction (OpenCV)
      │
      ▼
MediaPipe Pose  ──►  Body landmarks (x, y, z, visibility)
      │
      ▼
Joint coordinates  ──►  Joint angles (knee, hip, ankle, shoulder, …)  ──►  Degrees of freedom
      │
      ▼
Stability analysis (symmetry · range of motion · step consistency · center of mass)
      │
      ▼
Desktop dashboard (video overlay, skeleton, 3D trajectory, tables, stability score)
      │
      ▼
OpenSim-compatible export (.trc markers · .mot/.csv angles · biomechanical JSON)
```

The dashboard provides the video and pose overlay, skeleton visualization, 3D
trajectory, real-time position table, selectable body points / DOF, a stability
breakdown, and playback controls.

---

## 3. Why OpenSim is used

[OpenSim](https://opensim.stanford.edu/) is an open-source platform for modeling
and simulating the musculoskeletal system. In StableWalk it serves as the
**biomechanics-oriented representation and export layer**. Rather than keeping
the motion only as raw image landmarks, StableWalk expresses it using concepts
that are standard in biomechanics:

- **Markers** — named anatomical points tracked over time.
- **Joints and coordinates** — anatomical joints and their generalized
  coordinates (e.g. `knee_angle_l`, `hip_flexion_r`).
- **Motion data** — time-indexed marker trajectories and joint-angle tables.

This makes the extracted motion compatible with established biomechanics tools
and prepares it for future musculoskeletal simulation.

---

## 4. Important clarification: MediaPipe vs OpenSim

These two libraries play **different, complementary roles** and should not be
confused:

| Library | Role in StableWalk |
| --- | --- |
| **MediaPipe** | Performs **pose estimation** — detects the human skeleton and landmark coordinates from each video frame. All **stability metrics** are computed from MediaPipe kinematics. |
| **OpenSim** | Provides the **biomechanical representation**, **export formats**, and **real Inverse Kinematics** when the OpenSim SDK is installed. |

In short: **MediaPipe sees the body; OpenSim describes and simulates the body biomechanically.**

### Professor-facing summary

1. **MediaPipe** extracts pose from video.
2. **StableWalk** converts pose landmarks to OpenSim-compatible TRC / MOT / JSON.
3. The **OpenSim SDK** runs real Inverse Kinematics (`InverseKinematicsTool`).
4. **Demo IK** (official Gait2392 pipeline) proves OpenSim integration works.
5. **Experimental StableWalk IK** runs on mapped MediaPipe markers from the user's walking session.
6. **Limitation:** direct video-to-OpenSim IK is **experimental** — MediaPipe landmarks are approximated with direct mappings and synthetic markers. Coverage is much improved versus the earlier 9-marker mapping, but this is not clinical-grade motion capture.

---

## 5. What the project currently supports

The OpenSim integration layer is fully functional. Export works **without**
requiring OpenSim to be installed; real IK requires the OpenSim SDK.

### Export (always available)

- **Mapping MediaPipe landmarks to OpenSim-style markers**
  (e.g. `left_knee → L_KNEE`, `right_heel → R_HEEL`).
- **Exporting marker trajectories** as an OpenSim `.trc` file.
- **Exporting joint-angle / motion data** as an OpenSim `.mot` Storage file
  (or `.csv` fallback).
- **Exporting a biomechanical JSON bundle** with marker mapping, per-frame
  data, stability report, and metadata.

Verified export for a typical session (`walk_stream`):

| File | Path |
| --- | --- |
| TRC | `data/output/opensim/walk_stream/walk_stream.trc` |
| MOT | `data/output/opensim/walk_stream/walk_stream.mot` |
| JSON | `data/output/opensim/walk_stream/walk_stream_opensim.json` |

### MediaPipe → OpenSim export marker names (17 markers)

| MediaPipe landmark | Exported TRC name |
| --- | --- |
| shoulders | `L_SHOULDER` / `R_SHOULDER` |
| hips | `L_HIP` / `R_HIP` |
| knees | `L_KNEE` / `R_KNEE` |
| ankles | `L_ANKLE` / `R_ANKLE` |
| heels | `L_HEEL` / `R_HEEL` |
| toes | `L_TOE` / `R_TOE` |
| elbows, wrists, head | `L_ELBOW`, `R_ELBOW`, `L_WRIST`, `R_WRIST`, `HEAD` |

### Marker mapping for OpenSim IK (direct + synthetic)

MediaPipe export names differ from the Gait2392 OpenSim model (31 IK marker tasks).
StableWalk applies **direct mappings** via `models/opensim/marker_mapping.json` and
generates **synthetic markers** from hips, knees, ankles, and feet in
`opensim_marker_mapping.py`.

**Direct mappings (11 landmarks → OpenSim names):**

| StableWalk TRC | OpenSim (Gait2392) |
| --- | --- |
| `R_SHOULDER` / `L_SHOULDER` | `R.Acromium` / `L.Acromium` |
| `HEAD` | `Top.Head` |
| `R_HIP` / `L_HIP` | `R.ASIS` / `L.ASIS` |
| `R_ANKLE` / `L_ANKLE` | `R.Shank.Front` / `L.Shank.Front` |
| `R_HEEL` / `L_HEEL` | `R.Heel` / `L.Heel` |
| `R_TOE` / `L_TOE` | `R.Toe.Tip` / `L.Toe.Tip` |

**Synthetic markers (approximated from MediaPipe landmarks):**

| OpenSim marker | Approximation |
| --- | --- |
| `V.Sacral` | Midpoint of `L_HIP` and `R_HIP` |
| `Sternum` | Between shoulder midpoint and `HEAD` |
| `R/L.Thigh.Upper` | 35% from hip toward knee |
| `R/L.Shank.Upper` | 35% from knee toward ankle |
| `R/L.Midfoot.Sup` | Centroid of ankle, heel, toe |
| `R/L.Midfoot.Lat` | Midpoint of ankle and heel |

`L_KNEE` / `R_KNEE` feed synthetic thigh/shank markers (Gait2392 has no `R.Knee.Lat`).
Elbow/wrist landmarks are exported but not used in the lower-body Gait2392 IK set.

This produces a richer mapped TRC (~21 markers):
`data/output/opensim/walk_stream/walk_stream_mapped_for_opensim.trc`

See `models/opensim/marker_mapping_report.txt` for coverage, synthetic marker notes,
and missing OpenSim markers.

---

## 6. OpenSim integration (real IK)

StableWalk integrates with the **OpenSim Python SDK** for real Inverse Kinematics.
Two separate workflows exist:

### A. OpenSim Demo IK (proves SDK integration)

Uses the official Gait2392 pipeline bundled under `models/opensim/Gait2392_Pipeline/`:

```bash
python main.py --run-opensim-demo-ik
```

| Input | Output |
| --- | --- |
| `subject01_Setup_IK.xml` | `subject01_walk1_ik.mot` |

Or click **Run OpenSim Demo IK** in the dashboard OpenSim panel.

### B. Experimental StableWalk IK (mapped MediaPipe markers)

Uses the user's walking session after export and marker name mapping:

```bash
python main.py --run-ik --opensim-model models/opensim/Gait2392_Pipeline/subject01_simbody.osim
```

| File | Path |
| --- | --- |
| Original TRC | `data/output/opensim/walk_stream/walk_stream.trc` |
| Mapped TRC | `data/output/opensim/walk_stream/walk_stream_mapped_for_opensim.trc` |
| IK setup XML | `data/output/opensim/walk_stream/stablewalk_setup_ik.xml` |
| Model | `models/opensim/Gait2392_Pipeline/subject01_simbody.osim` |
| IK output | `data/output/opensim/walk_stream/walk_stream_ik.mot` |

Or click **Run StableWalk IK Experimental** in the dashboard.

> **Experimental only:** StableWalk IK uses mapped **and synthetic** markers (~20+ of 31
> Gait2392 IK tasks). It runs and produces a real `.mot` file when marker coverage is
> sufficient, but biomechanical reliability remains **moderate at best** because MediaPipe
> is not equivalent to full optical motion capture.

Success is never faked — IK is reported complete only when `*_ik.mot` exists on disk.
Before IK, the console reports original/mapped/synthetic marker counts, model coverage
percentage, readiness tier, and reliability (limited / moderate / high).

### Installing the OpenSim SDK

```bash
conda create -n stablewalk-opensim python=3.11
conda activate stablewalk-opensim
conda install -c opensim-org opensim
pip install -r requirements.txt
```

### Downloading sample models

```bash
python download_opensim_model.py              # Gait2392_Simbody bundle
python download_opensim_model.py --list       # list local models
```

Place the Gait2392 **Pipeline** folder (with `subject01_Setup_IK.xml`, demo TRC,
and `subject01_simbody.osim`) under `models/opensim/Gait2392_Pipeline/` for Demo IK.

See [OPENSIM_STATUS.md](OPENSIM_STATUS.md) for the full verified status report.

---

## 7. Future work

Implemented today: OpenSim SDK integration, Demo IK, improved marker mapping (direct +
synthetic), mapped TRC generation, weighted IK setup XML, experimental StableWalk IK,
model discovery, and honest status reporting.

Possible next steps:

- **Model scaling** (`ScaleTool`) before IK for subject-specific geometry.
- **Compare OpenSim IK joint angles** with MediaPipe angles in the stability report.
- **Upper-body markers** from elbow/wrist if a full-body OpenSim model is added.
- **Muscle and joint-force analysis** (static optimization, joint reaction analysis).

### Optional OpenSim SDK (export works without it)

```python
try:
    import opensim as osim
    OPENSIM_AVAILABLE = True
except ImportError:
    OPENSIM_AVAILABLE = False
```

- If the SDK **is** installed, StableWalk runs real Inverse Kinematics.
- If it is **not** installed, StableWalk still exports valid OpenSim-compatible files.

---

## 8. Stability analysis

Stability is computed from four transparent, weighted metric groups
(`stablewalk/analysis/biomech_stability.py`):

| Metric | What it measures | Weight |
| --- | --- | --- |
| **Gait symmetry** | Left-vs-right knee, hip, and ankle angle agreement. | 30% |
| **Joint range of motion** | Consistency of knee/hip/ankle ROM between sides. | 25% |
| **Step consistency** | Step-timing regularity and left/right step balance. | 25% |
| **Center of mass** | Side-to-side sway of a hip/shoulder body-center proxy. | 20% |

The final **0–100 score** is the weighted average, classified as:

- **Stable** (≥ 70), **Moderate** (45–69), or **Unstable** (< 45).

Each result includes a plain-language explanation, e.g.
*"Right knee range of motion is significantly different from the left knee."*

---

## Installation

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Core dependencies: `opencv-python`, `mediapipe`, `numpy`, `matplotlib`,
`Pillow`. (OpenSim is optional — see above.)

---

## Usage

```bash
# Desktop dashboard (recommended)
python main.py --gui

# OpenSim Demo IK (official Gait2392 pipeline — proves SDK integration)
python main.py --run-opensim-demo-ik

# Experimental StableWalk IK (mapped MediaPipe TRC — requires prior export)
python main.py --run-ik --opensim-model models/opensim/Gait2392_Pipeline/subject01_simbody.osim

# Process a local clip (auto-exports OpenSim files after analysis)
python main.py data/videos/my_walk.mp4

# Stream from a URL
python main.py --url https://example.com/walk.mp4
```

In the dashboard:

- **Export OpenSim Files** — writes `.trc`, `.mot`, JSON for the current session.
- **Save Session** — persists kinematic DOF samples to `data/output/sessions/stablewalk_sessions.db` (SQLite).
- **Run OpenSim Demo IK** — runs the official Gait2392 demo pipeline.
- **Run StableWalk IK Experimental** — runs IK on mapped MediaPipe markers.
- **Stability Breakdown** + **Full explanation** — see why a walk is stable or unstable.

Exports are written to `data/output/opensim/<session>/`.
Session kinematic data is stored in `data/output/sessions/stablewalk_sessions.db` (see [docs/SESSION_STORAGE.md](docs/SESSION_STORAGE.md)).
See [PROFESSOR_REQUIREMENTS_CHECKLIST.md](PROFESSOR_REQUIREMENTS_CHECKLIST.md) for requirement mapping.

---

## Project structure (key modules)

```
stablewalk/
├── opensim_integration.py        # MediaPipe → OpenSim markers + .trc/.mot/JSON export
├── opensim_sdk.py                # OpenSim SDK probe, Demo IK, experimental StableWalk IK
├── opensim_marker_mapping.py     # Marker name mapping + mapped TRC generation
├── opensim_models.py             # Discover .osim files under models/opensim/
models/
├── opensim/
│   ├── marker_mapping.json       # StableWalk → OpenSim marker name mapping
│   ├── marker_mapping_report.txt # Matching / missing marker comparison
│   └── Gait2392_Pipeline/        # Official demo IK bundle
├── analysis/biomech_stability.py # Transparent, explainable stability metrics
├── core/pipeline.py              # Video → frames → pose → JSON pipeline
├── pose/                         # MediaPipe estimation, kinematics, gait events
└── ui/tk/                        # Tkinter dashboard
download_opensim_model.py         # Download official opensim-org sample models
OPENSIM_STATUS.md                 # Verified OpenSim integration status
PROFESSOR_REQUIREMENTS_CHECKLIST.md
```

---

## Demo Video Sources

Research-oriented demo protocol: **[data/demo_videos/DEMO_VIDEO_SOURCES.md](data/demo_videos/DEMO_VIDEO_SOURCES.md)**

| UI label | Internal key | Target source |
|----------|--------------|---------------|
| **Abnormal** | `abnormal` | [GAVD](https://github.com/Rahmyyy/GAVD) — annotations + YouTube URL |
| **Normal** | `normal` | [Health&Gait](https://zenodo.org/records/14039922) UGS |
| **Performance** | `athletic` | Health&Gait FGS (fast gait speed) |

```bash
python scripts/demo_video_selection_workflow.py
python scripts/select_gavd_abnormal_candidate.py
python scripts/inspect_healthgait_samples.py
python scripts/validate_demo_candidate.py --video <candidate.mp4>
```

Legacy Pexels / Utah walker-assisted clips are **deprecated** and fail the new validator. Install replacements only after `ACCEPT` or `ACCEPT_WITH_LIMITATIONS`.

---

## A note for the evaluation

> StableWalk uses **MediaPipe** to extract human pose from a walking video and
> converts the detected motion into **OpenSim-compatible** marker trajectories (TRC),
> joint-angle tables (MOT), and a biomechanical JSON bundle. When the OpenSim SDK
> is installed, StableWalk runs **real Inverse Kinematics**: the official **Demo IK**
> (Gait2392 pipeline) proves the integration, and **experimental StableWalk IK**
> runs on mapped + synthetic MediaPipe markers from the user's session. Stability metrics are
> computed from MediaPipe kinematics, not from OpenSim. Direct video-to-OpenSim IK is
> experimental — coverage is improved (~20+ markers) but not clinical-grade; the project
> documents this honestly and never fakes IK success.
