# Requirement 5 — gait dynamics simulation and reinforcement learning

**Status:** research and planning only (2026-10-12).  
**Implementation status:** **not implemented.** No physics engine, Gym/Isaac environment, or trained policy is operational in this StableWalk install.  
**Prerequisite:** supervisor requirements 2–4 must be validated on real video-derived motion before any simulator code is trained or wired into the GUI.  
**This document does not replace** the pose, gait-cycle, spectral, or asymmetry pipelines.

Claim language used below:

| Phrase | Meaning here |
|--------|----------------|
| **Operational** | Importable in this Python environment and able to step a physics or RL loop |
| **Export / formula / placeholder** | Code exists; it does not run a simulator |
| **Planned** | Architecture only; do not treat as delivered capability |

---

## 0. Verdict

Simulated human gait dynamics and deep reinforcement learning **could** become a research instrument for studying gait *instability hypotheses* after a separate physics environment exists and after simulated trajectories are compared to video-derived kinematics. They **cannot** currently support that study inside StableWalk, and they **must not** be described as a clinically validated diagnostic system.

Evidence from this workspace (probed 2026-10-12):

| Package / subsystem | Present in this environment |
|---------------------|-----------------------------|
| `isaaclab` / `isaacsim` | **No** |
| `mujoco` / `gymnasium` / `pybullet` | **No** |
| `stable_baselines3` / `skrl` / `rsl_rl` | **No** |
| OpenSim Python SDK (`opensim`) | **No** |
| `ISAAC_LAB_AVAILABLE` | **False** (hard-coded plus import probe) |
| `PhysicsSimulationForceEstimator.available` | **False** (placeholder) |
| `LearnedForceEstimator.available` | **False** (placeholder) |
| `requirements.txt` RL/physics extras | **None** (OpenCV, MediaPipe, numpy, matplotlib, Pillow, reportlab, pytest) |
| `models/real_to_sim/*.json` / URDF | **Missing** (retargeting uses a built-in fallback map) |

Real-to-Sim today is an **offline export and scoring pipeline**, not a simulator.

---

## 1. Audit of existing simulation-related code

### 1.1 OpenSim integration

OpenSim in StableWalk is a **biomechanics export / experimental IK** path, not a forward-dynamics gait simulator.

| Piece | Module | Operational? | Role vs requirement 5 |
|-------|--------|--------------|------------------------|
| TRC / MOT export | `opensim_integration.py`, `biomechanics/service.py` | Export yes; SDK not installed here | Kinematics for OpenSim tools |
| Experimental IK | `opensim_ik_status.py` | Status-gated | Inverse kinematics, not control |
| Inverse dynamics | `opensim_id_readiness.py` | **Not ready** | Needs measured GRF / COP / ExternalLoads |
| Scaled subject model | Demo `.osim` only | No per-subject ScaleTool | Cannot treat as digital twin |

**Implication:** OpenSim Inverse Dynamics must not be fed pose-proxy vGRF. OpenSim is **not** the planned RL physics engine. Isaac Lab notes already separate the two (`docs/VIRTUAL_GRF.md`).

### 1.2 Real-to-Sim pipeline

Four **offline** stages (`stablewalk/real_to_sim/pipeline.py`, `docs/REAL_TO_SIM_PIPELINE.md`):

| Stage | Code | What it actually does | What it does not do |
|-------|------|------------------------|---------------------|
| 1 Perception | `gait_style_extraction.py` | Stride, cadence, hip/arm/trunk sway fingerprint from `GaitMotionRecording` | Not HMR/ROMP/HybrIK in-app |
| 2 Retargeting | `retargeting.py`, `retarget_kinematics.py` | Uniform (and optional segment) scale toward Unitree G1 joint names | Not GMR, not IK onto robot DOF, no URDF in repo |
| 3 AMP export | `amp_reference_export.py` | Writes `amp_reference_motion.npz` (schema 1.0) | Does **not** launch Isaac; status is `partial` when Isaac is missing; even if Isaac imports, **training is not auto-run** |
| 4 Contact-sync | `contact_sync_reward.py` + pose vGRF | Formula `reward[t] = video_mask[t] * (Fz > thr)` vs **estimated** Fz | Not PhysX/MuJoCo contact forces |

CLI: `python main.py --real-to-sim`. Monitoring keys `isaac_training_running` / `isaac_training_complete` stay **unavailable** (`pipeline_status.py`).

### 1.3 Motion retargeting

- Config path: `models/real_to_sim/unitree_g1_retarget.json` — **file not in the tree**; `load_retarget_config()` falls back to a small human→G1 name map.
- Scale is morphology alignment (median leg length vs 0.85 m reference), not a physics-consistent retargeter.
- `retarget_kinematics.py` can compute flexion angles and joint limits **if** a skeleton JSON exists; default skeleton JSON is also absent.
- AMP metadata itself says joint positions must be mapped to robot DOF via **Isaac Lab GMR before training**.

### 1.4 Simulation files inventory

| Path | Kind |
|------|------|
| `stablewalk/real_to_sim/*` | Export / fingerprint / contact-sync formulas |
| `stablewalk/analysis/isaac_lab_integration.py` | Stub; `ISAAC_LAB_AVAILABLE = False`; `PIPELINE_STAGES` are labels |
| `stablewalk/analysis/virtual_grf.py` | Pose proxy + two research placeholders |
| `docs/REAL_TO_SIM_PIPELINE.md`, `REAL_TO_SIM_UPGRADE_PLAN.md`, `VIRTUAL_GRF.md` | Design notes; upgrade plan already recorded Isaac/GMR as not executable in-app |
| `stablewalk/sim/` | **This requirement:** observation/objective **contracts only** (no env step) |

There is no `env.py`, no MuJoCo XML, no Isaac task, no policy checkpoint, and no training loop.

### 1.5 Virtual ground reaction forces

Three estimators:

1. **`LegacyPoseProxyForceEstimator`** — operational *as a formula*: \(F_z \approx m(g + a_{\mathrm{CoM},z})\) gated by video contact masks (`GRFAnalyzer`). Timing/shape research only.
2. **`PhysicsSimulationForceEstimator`** — **placeholder**; `available=False`; notes point at Isaac Lab in a separate conda env.
3. **`LearnedForceEstimator`** — **placeholder**; requires labeled force-plate data; not trained.

Contact masks are **when** the foot is likely down. They are **not** force magnitude.

### 1.6 Contact synchronization logic

`contact_sync_reward.py` (schema 1.1) compares:

- video / retargeted contact masks, or
- video masks vs estimated vertical force above a Newton threshold (default 10 N).

This is a **research reward / agreement score**. It is not a physics residual, not a fall detector, and not a clinical score. Stage 4 of Real-to-Sim currently syncs **video masks to pose-proxy Fz**, which is circular for proving a simulator: both sides originate in the same monocular recording.

---

## 2. Is any physics engine or RL environment operational?

**No.**

| Question | Answer |
|----------|--------|
| Can this process `import isaaclab` / `mujoco` / `gymnasium`? | No |
| Does StableWalk `env.reset()` / `env.step()` exist? | No |
| Does AMP export equal AMP training? | No |
| Does contact-sync mean a policy was rewarded in simulation? | No |
| Does `ISAAC_LAB_AVAILABLE` become true if a user later installs Isaac? | Import probe only; still no in-app runner (`pipeline_status`: training never launched from StableWalk) |

`stablewalk.sim.status` records the same facts as code constants so GUI or papers cannot infer a live trainer from an import.

---

## 3. Comparison of feasible approaches

All four can *support research questions*; none is a diagnostic device. Ranking is for **this thesis after requirements 2–4**, not for a general robotics lab.

| Approach | What it is | Fit to StableWalk | Cost / risk | Thesis usefulness | Recommendation |
|----------|------------|-------------------|-------------|-------------------|----------------|
| **Physics-based gait simulation** | Articulated body + contacts (MuJoCo or Isaac PhysX); replay or PD-track video kinematics; read simulated GRF | Needs URDF/MJCF, retarget to DOF, GPU (Isaac) or CPU (MuJoCo) | High setup; OpenSim ID still not solved without measured loads | Strongest path to *independent* contact forces | **Stage S3 target** (MuJoCo first if no Isaac workstation) |
| **Imitation learning** | AMP / adversarial motion prior or behavior cloning on AMP NPZ | Export already sketched; GMR + env missing | Training instability; robot morphology ≠ human | Tests whether video style is reproducible | **Stage S4**, external env only |
| **Reinforcement learning** | Policy maximizes a designed reward (alive, track, contact-sync, energy) | No env, no baselines, no GPU deps | Reward hacking; sim–reality gap | Can probe *hypothesized* instability costs | **Stage S5**, after kinematic replay matches video |
| **Hybrid physics + learned** | Physics contacts + learned residual (or learned vGRF with plate labels) | Matches existing placeholder split | Needs plate data for the learned part | Best long-term research design | **Preferred end-state**; not stage-0 |

**Do not** train a large model now. **Do not** swap the MediaPipe → `GaitMotionRecording` pipeline for a policy.

**Preferred sequence if 2–4 validate:** hybrid *instrumentation* — keep the video pipeline as ground-truth kinematics; add an **optional** physics replay; only then consider imitation/RL in a **separate** environment. MuJoCo is the realistic in-thesis engine if Isaac Sim is unavailable. Isaac Lab remains the documented AMP target but is not required for a kinematics-only comparison study.

---

## 4. Optional simulator architecture

The simulator is **optional and side-car**. The GUI gait pipeline remains the source of truth.

```
Existing (do not replace)
  Video → MediaPipe → GaitMotionRecording
       → gait cycles, spectral (req 2–3), asymmetry3d (req 4)
       → pose-proxy vGRF, OpenSim TRC/MOT (export only)

Optional sidecar (planned, not implemented)
  ObservationAdapter(recording, cycles, spectral, asymmetry)
       → obs_t  (see §5)
  OptionalGaitSimEnv          # gymnasium-like; engine = none | mujoco | isaac
       action: joint PD targets or residual
       reward: research terms only (§8)
       info: tracking errors, contact IoU — never a diagnosis
  SimulatedTrajectory
       → SimVsVideoCompare (§7)
```

Code contract (no stepping): `stablewalk/sim/` (`architecture.py`, `observation_spec.py`, `objectives.py`, `status.py`).

**Inputs the adapter may accept** (all already produced by StableWalk):

- `GaitMotionRecording` joint positions, velocities, angles
- `GaitCycleAnalysisResult` events and contact masks
- Spectral feature vector (req 3)
- Asymmetry indices and phase-normalized trajectories (req 4)
- `GaitStyleFingerprint` (Real-to-Sim stage 1)
- Optional AMP NPZ for external trainers

**Outputs it must never write into:** Overview clinical labels, OpenSim ExternalLoads, Inverse Dynamics, or Compare-mode dementia fields.

**Engine policy:** default `engine="none"` (spec + observation packing only). Enabling MuJoCo or Isaac is an explicit, documented extra environment — not a `requirements.txt` dependency for the desktop app.

---

## 5. Simulator observations

Each channel is defined so a future env can consume **the same quantities the thesis already computes**, with provenance.

| Observation | Source in StableWalk today | Units / notes | Sim use |
|-------------|----------------------------|---------------|---------|
| Joint positions | `GaitMotionRecording` / `JointTimeSeries.positions` | metres, `sw_canonical_y_up`, pelvis-origin, ~body-normalized | Tracking target / state |
| Joint velocities | `JointTimeSeries.velocities` | m/s, same frame | State; finite-diff in sim |
| Joint angles / DOF | `angles`, `dof_angles`, `dof_velocities` | rad, rad/s | Robot PD targets after retarget |
| Center of mass estimate | Hip/pelvis trajectory used in vGRF CoM proxy | metres; **estimate**, not inertial measurement | Balance / capture-point research |
| Ground contact state | Gait-cycle contact masks | `{0,1}` timing; **not GRF** | Reward gating; IoU vs sim contacts |
| Foot clearance | Gait feature / swing-foot height | metres | Trip-risk *hypothesis* term |
| Bilateral asymmetry | `analysis/asymmetry3d` Robinson SI / ASI / SI_norm, 0–100% phase curves | dimensionless indices | Style / instability *features*, not diagnosis |
| Spectral & temporal gait features | `analysis/spectral` A–H; cadence, stride, cycle time | Hz, s, ratios | Periodicity reward; compare spectra §7 |

Packaging rules (planned adapter):

- Missing joints stay **NaN** (same as spectral: never `Vec3(0,0,0)` fillers).
- Each vector carries `fps`, `n_samples`, `missing_fraction`, `coordinate_system`.
- Contact is a **boolean/timing** channel; force, if any, is a separate physics readout with method = `physics_simulation` only when a real engine produced it.

---

## 6. Measurable objectives and research validation criteria

### 6.1 Scientific question

> Can an optional physics or RL sidecar, driven by video-derived gait features, reproduce kinematic and contact-timing patterns closely enough that *simulated* perturbations become a useful way to study gait instability **in silico**?

That is a **methods** question. A “yes” does not imply clinical detection of disease.

### 6.2 Quantitative objectives (simulator quality — not patient outcome)

| ID | Objective | Metric | Provisional pass (research, not clinical) |
|----|-----------|--------|-------------------------------------------|
| O1 | Kinematic tracking | RMSE of pelvis and ankle trajectories (m), hip/knee/ankle angle RMSE (deg), after Procrustes/time-align | Report means + 95% CI; no hardcoded “healthy” cutoff in GUI |
| O2 | Event timing | Heel-strike / toe-off lag (s and % cycle) | Median \|Δt\| vs video events |
| O3 | Contact agreement | Mask IoU; contact-sync reward mean | Compare **physics** contacts to video masks; do not score pose-Fz against the same masks as proof of physics |
| O4 | Spectral fidelity | Δ dominant frequency, band-power ratio, ACF peak lag | Same feature code as req 3 on both streams |
| O5 | Asymmetry fidelity | Δ SI / ASI on phase-normalized curves | Same req 4 indices |
| O6 | Physics sanity (if engine exists) | Simulated vertical GRF: two peaks in walking, impulse ~ body weight × time | Shape checks only until force plates exist |
| O7 | RL (only if trained) | Episode return **and** O1–O5; abort if return rises while O1 worsens (reward hacking) | Checkpoint + seed recorded; not a GUI number |

### 6.3 Validation protocol

1. **Same clip, two paths:** video pipeline (frozen) vs simulator rollout from that clip’s observations/reference.
2. **Held-out clips** from the research catalog (`stablewalk/research/`), independent A/B slots — no shared `PoseSequence`.
3. **Synthetic control:** periodic pendulum / prescribed sinusoid gait in-engine → O4 recovers the known frequency (mirrors req 3 unit tests).
4. **Negative control:** shuffled contact masks must collapse O3.
5. **Do not** validate against unverified disease labels (same rule as the research compare module).

### 6.4 What would *not* count as validation

- AMP NPZ file written
- Contact-sync score using pose-proxy Fz
- A high RL reward
- A threshold on SI or spectral entropy labeled as “unstable gait disorder”

---

## 7. Comparing simulated motion to real video-derived motion

Use the **same canonical frame** (`sw_canonical_y_up`, pelvis origin, body-normalized metres) on both sides. Retargeted robot metres must be inverse-scaled before error in human units, or errors reported in **normalized** units with the scale factor.

| Layer | Video-derived | Simulated | Comparison |
|-------|---------------|-----------|------------|
| Trajectories | `GaitMotionRecording` positions | Sim link positions mapped to joint registry | RMSE / MAE after time offset search or DTW |
| Angles | `dof_angles` | Sim joint q | RMSE (deg) |
| Events | `analyze_gait_cycles` | Threshold on sim foot height/force | Δt, %GC |
| Contact | Video masks | Sim contact sensors (only if engine) | IoU, precision/recall |
| Spectral | Req 3 features | Same functions on sim time series | Paired Δ |
| Asymmetry | Req 4 SI/ASI | Same on sim L/R | Paired Δ |
| Style | `GaitStyleFingerprint` | Cadence/stride from sim | Relative error |

Reporting: JSON/CSV alongside existing research compare reports — **simulator vs video**, not “patient A vs patient B diagnosis.”

If the engine is `none`, comparison is limited to **observation packing round-trip** (adapter reconstructs channels that match the recording). That is a software test, not a physics result.

---

## 8. Three distinct quantities (do not conflate)

| Quantity | Definition | Produced by | Allowed language | Not allowed |
|----------|------------|-------------|------------------|-------------|
| **Simulation reward** | Scalar designed for learning or tracking (alive bonus, pose error penalty, contact-sync, energy) | Env `reward` / `contact_sync_reward.py` | “The policy’s return”, “contact-sync score” | “Fall risk”, “Parkinson’s”, “unstable patient” |
| **Predicted instability** | Research **hypothesis score** from features (e.g. high SI + low clearance + spectral dispersion) | Optional later scorer, clearly labeled `kind=predicted_instability` | “Predicted instability index (unvalidated)” | “Diagnosis”, “clinically unstable” |
| **Clinically validated diagnosis** | Patient-level label from a qualified clinician / gold-standard protocol, with ethics and labeled data | **Out of scope** unless a labeled clinical dataset is added later | Only if such data exist and a separate validation study is run | Any GUI badge, PDF conclusion, or AMP reward presented as diagnosis |

`stablewalk.sim.objectives.ClaimLevel` encodes this split. `clinical_diagnosis` is **never** an output of the planned env.

Contact-sync `mean_reward` is **simulation reward** even when computed on real video, because it is a mask–force agreement formula.

---

## 9. Clinical diagnostic accuracy

StableWalk **must not claim** clinical diagnostic accuracy from:

- monocular pose,
- spectral features,
- asymmetry indices,
- pose-proxy vGRF,
- simulation rewards,
- or a future RL policy,

without **patient-level labeled validation** (clinician labels or accepted clinical instruments, held-out subjects, prevalence-aware metrics).

That labeled corpus is **not** in this repository. Unverified catalog tags (e.g. dementia without source) remain forbidden. Requirement 5 therefore studies **whether simulation can support instability research**, not whether it detects disease.

---

## 10. Feasibility assessment

| Dimension | Assessment |
|-----------|------------|
| **In-app RL this term** | **Not feasible** without new heavy deps, GPU Isaac/MuJoCo, robot assets, and a training loop that would threaten the desktop pipeline |
| **Observation sidecar** | **Feasible** immediately after 2–4 validation: pack existing arrays (S1) |
| **Kinematic replay comparison (S2)** | **Feasible** with numpy only: treat video series as “open-loop sim” and reuse req 3–4 metrics |
| **Physics GRF (S3)** | **Conditionally feasible** if a separate env provides MuJoCo or Isaac + a humanoid model; **not feasible** in the current `requirements.txt` |
| **AMP/RL (S4–S5)** | **Research-feasible** only externally; StableWalk stays the motion provider |
| **Clinical claims** | **Not feasible** (no labeled patients) |
| **OpenSim as the RL engine** | **Poor fit**; keep OpenSim for IK/ID readiness, not policy training |
| **Risk to existing pipeline** | High if Isaac/PyTorch are added to the GUI env; keep sim optional and isolated |

**Honest thesis sentence:** *StableWalk prepares motion, contact timing, spectral and asymmetry features, and an AMP-style reference export for a future physics/RL environment; it does not currently simulate gait dynamics or train a reinforcement learning agent.*

---

## 11. Staged implementation plan

Implement **only after** requirements 2–4 are validated. Each stage has a stop rule. Do not skip to training.

| Stage | Goal | In-repo work | Out of repo | Stop / pass | Status |
|-------|------|--------------|-------------|-------------|--------|
| **S0** | Plan and freeze claims | This document, `stablewalk/sim` contracts, tests that engines are absent | — | Tests assert `SIMULATOR_IMPLEMENTED is False` | **This delivery** |
| **S1** | Observation adapter | Map §5 channels from `GaitMotionRecording` + spectral + asymmetry3d; NaN policy; no GUI change | — | Round-trip tests; no new deps | Planned |
| **S2** | Sim-vs-video metrics without physics | Compare two recordings (video vs kinematic replay / second clip) using O1–O5 | — | Report Δ; still no “RL implemented” | Planned |
| **S3** | Optional physics contacts | Thin adapter behind `engine` flag; read contact forces if present | MuJoCo or Isaac + URDF/MJCF | `PhysicsSimulationForceEstimator.available` true **only** when engine returns forces | Planned |
| **S4** | Imitation in external env | Consume existing AMP NPZ; document GMR; do not auto-train from GUI | Isaac Lab / AMP | Qualitative tracking + O1 | Planned |
| **S5** | RL research rewards | Reward terms from §8; log O1–O5 vs return | GPU trainer | Detect reward hacking | Planned |
| **S6** | Clinical-grade language | — | Labeled patient study | Only then discuss diagnostic metrics — still not a device claim | Not in scope |

**S0 acceptance (this stage):**

- [x] Audit of OpenSim, Real-to-Sim, retargeting, sim files, vGRF, contact-sync
- [x] Explicit statement: no operational physics/RL env
- [x] Approach comparison and optional architecture
- [x] Observation list tied to existing modules
- [x] Objectives and sim-vs-video comparison
- [x] Reward ≠ predicted instability ≠ diagnosis
- [x] No large-model training; gait pipeline unchanged

---

## 12. Pointers

| Document / module | Use |
|-------------------|-----|
| `docs/erik_requirements_plan.md` | Reqs 1–5 map; req 5 remains unimplemented |
| `docs/REAL_TO_SIM_PIPELINE.md` | Four-stage export workflow |
| `docs/REAL_TO_SIM_UPGRADE_PLAN.md` | Earlier inventory: Isaac/GMR not executable in-app |
| `docs/VIRTUAL_GRF.md` | Terminology; ID must not use proxy GRF |
| `stablewalk/sim/` | Planned contracts |
| `stablewalk/real_to_sim/` | Offline export |
| `stablewalk/analysis/isaac_lab_integration.py` | Availability stub |
