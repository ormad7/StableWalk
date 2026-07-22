# StableWalk — Project Architecture

StableWalk is a **master's research prototype** for monocular-video gait analysis and offline Real-to-Sim preparation. It is not a clinical diagnostic device or a replacement for instrumented gait laboratories.

## System overview

```
Video / pose JSON
       │
       ▼
┌──────────────────┐
│ Pose backends    │  MediaPipe (default), optional SMPL/ROMP
└────────┬─────────┘
         ▼
┌──────────────────┐
│ GaitMotionRecording │  Canonical hip-centered 3D skeleton
└────────┬─────────┘
         ▼
┌──────────────────┐     ┌─────────────────────┐
│ Gait cycles      │────▶│ Foot contact + vGRF │
│ Gait features    │     │ (estimated timing)  │
└────────┬─────────┘     └──────────┬──────────┘
         │                          │
         ▼                          ▼
┌──────────────────┐     ┌─────────────────────┐
│ Biomechanical    │     │ Real-to-Sim pipeline │
│ analysis         │     │ (4 stages + exports) │
└────────┬─────────┘     └──────────┬──────────┘
         │                          │
         ▼                          ▼
   GUI dashboard              NPZ / JSON artifacts
   (4 tabs)                   → future Isaac Lab
```

## GUI dashboard (four tabs)

| Tab | Purpose |
|-----|---------|
| **Overview** | Video + 3D skeleton, gait summary cards, contact & Estimated Virtual GRF charts |
| **Motion Analysis** | Knee flexion, joint trajectories, DOF selection |
| **Biomechanics** | Estimated biomechanical parameters, ROM, video quality checklist |
| **Advanced & Export** | Real-to-Sim stages, session export, OpenSim hooks |

Playback transport is shared; charts receive the current playhead timestamp so plots stay synchronized with the loaded video.

## Analysis layers

| Layer | Package | Data tier |
|-------|---------|-----------|
| Pose / skeleton | `stablewalk/pose/`, `adapters/` | Estimated |
| Gait cycles & features | `stablewalk/analysis/gait_*` | Calculated / derived |
| Foot contact | `stablewalk/analysis/foot_contact_analysis.py` | Estimated |
| Estimated Virtual GRF | `stablewalk/analysis/estimated_vgrf_analysis.py` | Estimated |
| Biomechanics | `stablewalk/analysis/biomechanical/` | Estimated + derived |
| Movement stability score | `stablewalk/analysis/biomech_stability.py` | Derived |
| Real-to-Sim | `stablewalk/real_to_sim/` | Mixed (exports for external sim) |

Terminology constants: `stablewalk/ui/scientific_labels.py`.

## Coordinate systems

- **SW_CANONICAL:** Hip-centered, body-normalized meters — used for gait analysis and 3D skeleton.
- **Image space:** MediaPipe 0–1 coordinates — used for video overlay and walking-speed scaling.
- **OpenSim TRC:** Separate lab-style export frame when OpenSim IK is run.

See `docs/COORDINATE_SYSTEMS.md`.

## Storage & exports

- Session DB: `data/output/sessions/stablewalk_sessions.db`
- Motion reference runs: `data/output/motion_reference/<run_name>/`
- Biomechanical artifacts: COM/BoS NPZ, `biomechanical_report.json`, `video_quality.json`

Exports embed a `terminology` block distinguishing **measured**, **estimated**, **derived**, and **calculated** quantities.

## Current capabilities

- Monocular walking video → pose → gait metrics → biomechanical dashboard
- Foot contact timing, gait events, Estimated Virtual GRF (pose proxy)
- COM, base of support, stability margin, symmetry, ROM, gait quality score
- Offline Real-to-Sim artifact generation (motion, retarget, AMP reference, contact-sync reward)
- Tkinter GUI with tabbed dashboard and matplotlib charts

## Limitations & assumptions

- **No force plates:** All GRF values are Estimated Virtual GRF, not measured kinetics.
- **Monocular scale:** Absolute walking speed and step length are approximate; hip-centered coordinates do not preserve world translation.
- **No SMPL by default:** MediaPipe BlazePose is the default perception backend.
- **Isaac Lab training** is external; StableWalk exports reference clips only.
- **OpenSim IK** requires marker mapping and model setup; not automatic from arbitrary video.

## Future Isaac Lab integration

Planned external workflow (not implemented inside StableWalk):

1. Load `retargeted_motion.npz` / `amp_reference_motion.npz` in Isaac Lab.
2. Run GMR or built-in retargeting to Unitree G1/H1 URDF.
3. Train AMP policy using exported contact masks and contact-sync rewards.
4. Replace `LegacyPoseProxyForceEstimator` with `PhysicsSimulationForceEstimator` reading PhysX ContactSensor data during training.

See `docs/SPEC_COMPLIANCE.md` and `docs/REAL_TO_SIM_PIPELINE.md`.

## Key entry points

| Command | Action |
|---------|--------|
| `python main.py --gui` | Launch dashboard |
| `python main.py --real-to-sim <video>` | Offline Real-to-Sim pipeline |
| `python scripts/validate_biomechanical_demos.py` | Biomechanics smoke test on demo JSON |

## Module index

```
stablewalk/
  analysis/          # Gait, contact, vGRF, biomechanical, stability
  real_to_sim/       # Pipeline, retargeting, AMP export, rewards
  ui/                # Tk dashboard, charts, scientific_labels
  io/                # NPZ/JSON/session exports
  pose/              # Backends, enrichment, kinematics
  models/            # PoseSequence, GaitMotionRecording
docs/                # Architecture, biomechanics, spec compliance
tests/               # Pytest suite (~38 modules)
```
