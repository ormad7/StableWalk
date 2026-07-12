# Virtual Ground Reaction Force (vGRF) — research architecture

StableWalk does **not** measure ground reaction forces from monocular video.

Use the terminology:

- **Estimated Virtual Ground Reaction Force (vGRF)**, or
- **Simulated Ground Reaction Force**

when referring to physics-based or learned proxies intended for comparative research.

## Foot contact vs virtual force

| Signal | Meaning | Is it GRF? |
|--------|---------|------------|
| **Foot contact mask** | Heuristic timing (clearance, velocity, persistence) | **No** |
| **Pose-based vertical proxy** (`GRFAnalyzer` in `forces.py`) | Newton-inspired vertical estimate from CoM acceleration | Approximation only — not instrumented |
| **Virtual GRF** (`virtual_grf.py`) | Future physics / learned estimators | Simulated — not measured |

The GUI **Physics Force Estimation** panel reports virtual-force status separately from **Gait Cycle** contact indicators.

## OpenSim Inverse Dynamics — what is missing

Traditional OpenSim **Inverse Dynamics** requires:

1. Subject-specific **scaled** `.osim` model (`ScaleTool`)
2. **Joint kinematics** from IK (`.mot`)
3. **Measured external loads** — force-plate GRF (Fx, Fy, Fz) and center of pressure
4. **`ExternalLoads.xml`** referencing those measured forces

StableWalk today:

| Requirement | Status |
|-------------|--------|
| Scaled OpenSim model | Demo model only; no per-subject `ScaleTool` |
| Subject mass in model | Estimated mass for pose analysis only |
| Joint kinematics | Pose `.mot` + optional `{run}_ik.mot` |
| External loads XML | **Not present** |
| Measured GRF / COP | **Not present** |

**Do not** invent external forces and pass them to `InverseDynamicsTool` as measured data.

Run `assess_opensim_id_readiness(run_name)` (see `opensim_id_readiness.py`) for an honest audit.

## Virtual force estimator interface

Module: `stablewalk/analysis/virtual_grf.py`

```text
VirtualForceEstimator
├── UnavailableForceEstimator          (default — research placeholder)
├── PhysicsSimulationForceEstimator    (future Isaac Lab contact forces)
└── LearnedForceEstimator              (future ML model)
```

### Input contract (`VirtualForceEstimatorInput`)

- Normalized gait-cycle kinematics (`CycleConsistencyResult`)
- Joint trajectories (canonical 3D positions)
- Pelvis / root trajectory
- Foot contact mask (timing only)
- Foot contact events
- Estimated body mass and segment dimensions
- Video FPS

### Output contract (`VirtualForceResult`)

- Left / right vertical vGRF trajectories (N)
- Optional 3D force vectors per foot
- Body-weight normalization: `vGRF_BW = Force / (mass × 9.81)`
- Estimation method, confidence, scientific disclaimer

## Real-to-Sim pipeline (planned)

```text
Video
  → Pose / Human Mesh Recovery
  → Canonical 3D Motion (GaitMotionRecording)
  → Gait Contact Mask (timing — not force)
  → Motion Retargeting
  → Physics-Based Humanoid Simulation (Isaac Lab — separate env)
  → Contact Forces (ContactSensor / contact-force APIs)
  → Virtual GRF Analysis
```

Isaac Lab is **not** installed in the OpenSim / MediaPipe environment. See `isaac_lab_integration.py`.

## Motion reference export

Export a retargeting dataset:

```bash
python main.py --export-motion-reference data/videos/my_walk.mp4
python main.py --export-motion-reference walk_stream   # existing pose run name
```

Output: `data/output/motion_reference/{run}/stablewalk_motion.npz`

### NPZ contents

| Array | Shape | Description |
|-------|-------|-------------|
| `timestamps` | (N,) | Time in seconds |
| `fps` | scalar | Frame rate |
| `root_positions` | (N, 3) | Pelvis-centered root |
| `root_orientations` | empty | Reserved (not from MediaPipe) |
| `canonical_joint_positions` | (N, J, 3) | Canonical skeleton |
| `left_contact_mask` | (N,) | Contact timing — **not force** |
| `right_contact_mask` | (N,) | Contact timing — **not force** |
| `body_scale_metadata_json` | string | Segment dimensions + notes |

## Related modules

| Path | Role |
|------|------|
| `stablewalk/analysis/virtual_grf.py` | Estimator interface + data contract |
| `stablewalk/analysis/opensim_id_readiness.py` | ID prerequisite audit |
| `stablewalk/analysis/isaac_lab_integration.py` | Isaac Lab placeholder |
| `stablewalk/io/motion_reference_export.py` | `stablewalk_motion.npz` export |
| `stablewalk/analysis/forces.py` | Legacy pose-based vertical proxy (separate from vGRF) |
