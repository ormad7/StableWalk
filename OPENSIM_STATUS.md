# OpenSim Status — Demo IK + StableWalk IK

**Last updated:** 2026-06-04 (auto-generated)

---

## Runtime verification

| Check | Status |
| --- | --- |
| OpenSim SDK detected | yes |
| OpenSim Demo IK | completed |
| Demo IK output file | `C:\Users\ormad\Downloads\Stable__Walk\StableWalk\models\opensim\Gait2392_Pipeline\subject01_walk1_ik.mot` |
| StableWalk TRC export | completed |
| Mapped TRC | `C:\Users\ormad\Downloads\Stable__Walk\StableWalk\data\output\opensim\walk_stream\walk_stream_mapped_for_opensim.trc` |
| Marker mapping | improved |
| Mapped markers matching OpenSim | 21 / 31 (67.7%) |
| Direct mapped markers | 11 |
| Synthetic markers | 10 |
| IK readiness tier | reliable |
| Reliability | high (not clinical-grade) |
| StableWalk IK | completed (experimental — not fully validated) |
| StableWalk IK output file | `C:\Users\ormad\Downloads\Stable__Walk\StableWalk\data\output\opensim\walk_stream\walk_stream_ik.mot` |

StableWalk IK note:

> MediaPipe pose landmarks are approximated for OpenSim IK. Direct mappings use exported landmarks; synthetic markers are interpolated from hips, knees, ankles, and feet. This improves coverage versus raw export but is not clinical-grade mocap.

---

## Two separate OpenSim workflows

### Part A — OpenSim Demo IK (proves SDK integration)

Uses the **official OpenSim Gait2392 sample files** (not MediaPipe):

| File | Path |
| --- | --- |
| IK setup XML | `models/opensim/Gait2392_Pipeline/subject01_Setup_IK.xml` |
| Demo TRC | `models/opensim/Gait2392_Pipeline/subject01_walk1.trc` |
| Static TRC | `models/opensim/Gait2392_Pipeline/subject01_static.trc` |
| Model | `models/opensim/Gait2392_Pipeline/subject01_simbody.osim` |
| IK output | `models/opensim/Gait2392_Pipeline/subject01_walk1_ik.mot` |

**GUI:** Click **Run OpenSim Demo IK**

**CLI:** `python main.py --run-opensim-demo-ik`

**Code:** `run_opensim_demo_ik()` → `InverseKinematicsTool(setup_xml).run()`

Success is reported **only** when a real `*ik*.mot` file exists on disk.

### Part B — StableWalk IK (MediaPipe → OpenSim)

Uses **StableWalk-exported** files from video analysis:

| File | Path |
| --- | --- |
| StableWalk TRC | `data/output/opensim/walk_stream/walk_stream.trc` |
| Mapped TRC | `data/output/opensim/walk_stream/walk_stream_mapped_for_opensim.trc` |
| StableWalk MOT | `data/output/opensim/walk_stream/walk_stream.mot` (MediaPipe angles) |
| JSON | `data/output/opensim/walk_stream/walk_stream_opensim.json` |
| Marker mapping | `models/opensim/marker_mapping.json` |
| StableWalk IK output | `data/output/opensim/walk_stream/walk_stream_ik.mot` (only if IK runs) |

**GUI:** **Run StableWalk IK Experimental** (mapped + synthetic markers — experimental, not clinical-grade)

StableWalk exports 17 MediaPipe landmarks (`L_KNEE`, `R_HIP`, …). The mapped TRC renames direct matches (`R.ASIS`, `R.Heel`, …) and adds **synthetic** thigh, shank, sacral, midfoot, and sternum markers interpolated from hips/knees/ankles/feet. This improves coverage versus the previous 9-marker mapping, but MediaPipe is still not equivalent to full optical motion capture.

---

## Quick verification

```bash
# Run demo IK immediately (requires OpenSim SDK in conda env)
python main.py --run-opensim-demo-ik

# Or from Python
python -c "from stablewalk.opensim_sdk import run_opensim_demo_ik; r=run_opensim_demo_ik(); print(r)"
```
