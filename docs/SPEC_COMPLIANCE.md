# Spec compliance — Real-to-Sim gait pipeline

This document maps the research specification in  
`Use 3d gait walking simulator of human by means of human video walking (1).docx`  
to the StableWalk implementation.

## 4-stage pipeline overview

| Stage | Spec requirement | StableWalk module | Status |
|-------|------------------|-------------------|--------|
| **1. Perception** | Video → 4D motion; stride, cadence, hip sway, arm swing | `stablewalk/real_to_sim/gait_style_extraction.py` | **Implemented** (MediaPipe pose) |
| **2. Retargeting** | GMR to Unitree G1/H1 URDF morphology | `stablewalk/real_to_sim/retargeting.py` | **Partial** (uniform scale; not full GMR) |
| **3. Simulation** | Isaac Lab AMP training from reference clip | `stablewalk/real_to_sim/amp_reference_export.py` | **Partial** (export only; training external) |
| **4. Physics** | Virtual GRF + contact-mask sync reward | `virtual_grf.py`, `contact_sync_reward.py`, `estimated_vgrf_analysis.py` | **Implemented** (Estimated Virtual GRF offline) |
| **5. Biomechanics** | COM, BoS, stability margin, symmetry, ROM, gait quality | `analysis/biomechanical/` | **Implemented** (pose-based estimates) |

**Offline entry point:**

```powershell
python main.py --real-to-sim data/demo_videos/normal_gait.mp4
```

**Outputs per run** (`output/motion_reference/<run_name>/`):

| File | Stage | Description |
|------|-------|-------------|
| `stablewalk_motion.npz` | 1 | Human motion reference + gait style metadata |
| `retargeted_motion.npz` | 2 | Scaled trajectories for Unitree G1 proportions |
| `amp_reference_motion.npz` | 3 | AMP training reference (root + contacts) |
| `contact_sync_reward.npz` | 4 | Per-frame contact–force synchronization rewards |
| `real_to_sim_pipeline_report.json` | — | Stage statuses and summary metrics |

---

## Stage 1 — Perception layer

**Spec:** Extract joint rotations and root trajectory; capture stride length, cadence, hip sway, and arm swing. Ideal tools: SMPL/SMPL-X, ROMP, HybrIK.

**StableWalk:**

- MediaPipe BlazePose → `GaitMotionRecording` (canonical joint positions, not SMPL mesh).
- `extract_gait_style_fingerprint()` computes stride, cadence, hip sway, arm swing, trunk sway.
- Exported in `stablewalk_motion.npz` via `gait_style_json` metadata.

**Gaps:**

- No SMPL body model or true joint rotation quaternions from video.
- Monocular depth is approximate; short or occluded clips may yield `partial` stage status.

**Workaround:** Use side-view walking clips with ≥2 usable gait cycles for best fingerprint quality.

---

## Stage 2 — Retargeting layer

**Spec:** GMR (General Motion Retargeting) maps human limb lengths to robot URDF (Unitree G1/H1).

**StableWalk:**

- `retarget_motion_reference()` applies uniform scale from median human leg length vs robot reference (`models/real_to_sim/unitree_g1_retarget.json`).
- Writes dedicated `retargeted_motion.npz` (not only the human motion file).

**Gaps:**

- No Isaac Lab GMR or inverse kinematics to robot DOF angles.
- `joint_positions` remain canonical scaled positions; map to URDF in Isaac Lab before AMP training.

**Next step in Isaac Lab:** Load `retargeted_motion.npz` → run built-in GMR scripts with Unitree G1 URDF.

---

## Stage 3 — Simulation layer (AMP)

**Spec:** Adversarial Motion Priors (AMP) in Isaac Lab trains a policy that mimics the video gait style.

**StableWalk:**

- `export_amp_reference()` produces `amp_reference_motion.npz` with root positions, estimated root quaternions, scaled joint positions, and bilateral contact masks.
- `check_isaac_lab_available()` probes for Isaac Lab without importing it at startup.
- Stage status is `partial` when Isaac Lab is not installed (expected in default env).

**Gaps:**

- RL training is not run inside StableWalk; requires separate Isaac Sim + Isaac Lab environment.
- Reference clip uses estimated root orientation from displacement, not mocap-quality rotations.

**Example loader:** `scripts/load_amp_reference_example.py`

---

## Stage 4 — Physics & virtual GRF

**Spec:** ContactSensor / PhysX foot forces; reward when `video_contact_mask[t] × (simulated_force_z[t] > threshold)`.

**StableWalk:**

- `estimate_virtual_grf()` — `LegacyPoseProxyForceEstimator` from pose-derived kinematics (no Isaac Sim required).
- `contact_force_sync_reward()` implements the spec reward per frame.
- `export_contact_sync_reward_npz()` writes `left_reward`, `right_reward`, `combined_reward` arrays.
- `summarize_contact_sync()` reports mean alignment and interpretation text.

**Gaps:**

- `PhysicsSimulationForceEstimator` is a placeholder for Isaac Lab ContactSensor forces during training.
- Virtual GRF from pose proxy is approximate, not PhysX ground truth.

**Validation:** Compare `contact_sync_reward.npz` mean reward against clinical double-peak GRF literature after Isaac training.

---

## GUI integration

| Feature | Location |
|---------|----------|
| 4-stage summary panel | Advanced & Export tab |
| Run full pipeline | **Real-to-Sim Pipeline** button |
| Biomechanics tab | Motion Analysis notebook + Biomechanics tab |
| Terminology tiers | `scientific_labels.py` + export `terminology` blocks |

---

## Compliance summary

| Area | Meets spec? | Notes |
|------|-------------|-------|
| End-to-end offline pipeline | Yes | All 4 stages produce artifacts |
| Gait style fingerprint | Yes | Stride, cadence, sway, arm swing |
| Robot morphology alignment | Partial | Scale only, not GMR |
| AMP reference export | Yes | Ready for external Isaac Lab |
| Contact-sync reward | Yes | Per-frame NPZ + summary in report |
| Isaac Lab AMP training | No | External environment required |
| SMPL / ROMP perception | No | MediaPipe substitute |

StableWalk fulfills the **research architecture and data contracts** described in the spec for an offline Real-to-Sim workflow. Full physics-based imitation learning requires completing stages 2–3 inside Isaac Lab with the exported NPZ files.

See also: `docs/REAL_TO_SIM_PIPELINE.md`
