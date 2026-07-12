# Real-to-Sim Gait Pipeline

StableWalk implements the **4-stage Real-to-Sim workflow** described in the project
research specification: extract gait style from human video, retarget to a humanoid
morphology, prepare AMP reference motion for Isaac Lab, and evaluate virtual ground
reaction forces with contact-mask synchronization.

## Pipeline overview

```
Walking video
      │
      ▼
1. Perception — MediaPipe pose → gait style fingerprint
      │           (stride, cadence, hip sway, arm swing)
      ▼
2. Retargeting — scale motion to Unitree G1 proportions
      │
      ▼
3. Simulation prep — export AMP reference motion (.npz)
      │                (Isaac Lab AMP training in separate env)
      ▼
4. Physics — virtual GRF proxy + contact-sync reward
```

## Stage 1: Perception (Video → 4D motion)

**Goal:** Capture the visual "fingerprint" of the walker's gait.

| Characteristic | Source |
|----------------|--------|
| Stride / step length | Pelvis displacement between heel strikes |
| Cadence | Gait cycle temporal metrics |
| Hip sway | Pelvis mediolateral range |
| Arm swing | Wrist–shoulder displacement range |

**Module:** `stablewalk/real_to_sim/gait_style_extraction.py`

**Output:** Gait style fingerprint embedded in `stablewalk_motion.npz`

## Stage 2: Retargeting (Morphology alignment)

Human limb lengths differ from simulated humanoids (Unitree G1/H1). StableWalk applies
**uniform scale retargeting** using median leg length from the video vs a robot
reference length.

**Config:** `models/real_to_sim/unitree_g1_retarget.json`

**Module:** `stablewalk/real_to_sim/retargeting.py`

Full GMR (General Motion Retargeting) in Isaac Lab requires a separate simulation
environment with the robot URDF.

## Stage 3: Simulation (AMP reference)

**Goal:** Prepare a reference motion clip for Adversarial Motion Priors (AMP) training.

Isaac Lab is **not** bundled with StableWalk. The offline pipeline exports:

- `data/output/motion_reference/{run}/stablewalk_motion.npz`
- `data/output/motion_reference/{run}/amp_reference_motion.npz`
- `data/output/motion_reference/{run}/amp_reference_manifest.json`

**Module:** `stablewalk/real_to_sim/amp_reference_export.py`

### Isaac Lab setup (separate environment)

1. Install NVIDIA Isaac Sim + Isaac Lab
2. Copy `amp_reference_motion.npz` into your training workspace
3. Configure `ContactSensor` on foot links (see `unitree_g1_retarget.json`)
4. Run AMP training (e.g. Minimal G1 AMP repository)

## Stage 4: Physics & virtual GRF

**Contact masks** (`left_contact_mask`, `right_contact_mask`) indicate *when* each
foot is on the ground — timing only, not force magnitude.

**Virtual GRF** uses a pose-based vertical proxy (`GRFAnalyzer`) until physics
simulation forces are available:

```
F_z(t) ≈ m · (g + a_com,z)   allocated to feet in contact
```

**Contact-sync reward** (for AMP gait reward design):

```python
reward[t] = video_contact_mask[t] * (simulated_force_z[t] > threshold)
```

**Module:** `stablewalk/real_to_sim/contact_sync_reward.py`

## CLI usage

```powershell
# Full 4-stage pipeline (latest pose run)
python main.py --real-to-sim

# From a specific video
python main.py --real-to-sim data/demo_videos/normal_gait.mp4

# From existing pose run
python main.py --real-to-sim walk_stream

# Motion reference only
python main.py --export-motion-reference walk_stream
```

## GUI usage

1. Load and analyze a walking video
2. Open **Advanced & Export** tab
3. Click **Real-to-Sim Pipeline** (runs all 4 stages)
4. Or **Export AMP Reference** for Isaac Lab only

The **Physics Force Estimation** sidebar shows virtual GRF status and gait style summary.

## Output files

| File | Purpose |
|------|---------|
| `stablewalk_motion.npz` | Canonical human motion + contact masks |
| `amp_reference_motion.npz` | Scaled motion for AMP training |
| `amp_reference_manifest.json` | Robot config + gait style metadata |
| `real_to_sim_pipeline_report.json` | Stage status + contact-sync summary |

## Scientific limitations

- MediaPipe pose is not SMPL — joint rotations are estimated, not recovered mesh params
- Virtual GRF is **not measured kinetics** — compare timing and shape only
- Isaac Lab AMP training requires GPU simulation hardware
- Uniform scale retargeting is a first step; clinical accuracy needs GMR + force plates

## Related docs

- `docs/VIRTUAL_GRF.md` — force estimation terminology
- `docs/POSE_BACKENDS.md` — future ROMP/HybrIK/SMPL backends
- `stablewalk/analysis/isaac_lab_integration.py` — Isaac Lab probe + stage map
