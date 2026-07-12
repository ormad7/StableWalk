# Real-to-Sim Upgrade Plan — Technical Review & Staged Implementation

**Document status:** Review complete — no code changes in this step  
**Date:** 2026-07-12  
**Scope:** Upgrade path from current video-to-gait pipeline toward  
`Video → 3D Human Motion → Retargeting → Isaac Lab AMP → Contact Forces / Virtual GRF`

---

## Executive summary

StableWalk already implements a **working offline Real-to-Sim data pipeline** on top of the MediaPipe pose stack. Stages 1 and 4 produce tested artifacts; stages 2 and 3 produce **export-only** retargeting and AMP reference files. **Isaac Lab AMP training, GMR retargeting, and physics-based GRF are not executable inside StableWalk today** — they require separate simulation environments.

This plan preserves:

- The existing MediaPipe pipeline (default, production path)
- GUI, OpenSim export/IK demos, session storage, and all current tests
- Honest separation of **kinematic estimation**, **physics simulation**, and **measured vs estimated** biomechanics

SMPL and related HMR backends will be added as **optional** backends with MediaPipe fallback — not replacements.

---

## 1. MediaPipe pose extraction — where it lives

| Layer | Path | Role |
|-------|------|------|
| **Production estimator (CLI/GUI default)** | `stablewalk/pose/estimation.py` — `PoseEstimator` | MediaPipe Pose Landmarker (Tasks API), VIDEO mode, writes `{run}_poses.json` |
| **Backend adapter (modular)** | `stablewalk/pose/backends/mediapipe_backend.py` — `MediaPipePoseBackend` | Same Tasks API, outputs `HumanMotionFrame` / canonical skeleton |
| **Skeleton overlay** | `stablewalk/pose/skeleton.py` | Draws MediaPipe connections on BGR frames |
| **Model assets** | `stablewalk/model_loader.py`, `models/pose_landmarker_*.task` | Lite/full BlazePose task files |
| **Legacy re-exports** | `stablewalk/pose_estimation.py` | Thin compatibility wrapper |
| **Core orchestration** | `stablewalk/core/pipeline.py`, `main.py` `run_step2()` | Video → frames → pose JSON; still uses `PoseEstimator` directly (not backend registry) |

**Important architectural note:** The backend registry (`stablewalk/pose/backends/registry.py`) exists for future HMR backends, but the **main GUI/CLI path still calls `PoseEstimator` in `estimation.py`**. Wiring the registry into the core pipeline is a planned upgrade (Stage 2 below), not yet done.

---

## 2. Input / output formats

### 2.1 Landmarks (MediaPipe → StableWalk)

| Field | Format | Notes |
|-------|--------|-------|
| Per-landmark coords | `x, y ∈ [0,1]` image-normalized; `z` relative depth | MediaPipe image landmarks |
| Confidence | `visibility ∈ [0,1]` | Stored as `Keypoint.visibility` |
| Names | Lowercase MediaPipe enum names + synthetic `mid_hip` | See `LANDMARK_NAMES` in `estimation.py` |
| Canonical mapping | `pelvis`, `left_hip`, … `right_toe` | `pose/backends/canonical.py` |

### 2.2 Joint positions

| Representation | Units / frame | Module |
|----------------|---------------|--------|
| Raw keypoints | Normalized image | `models/pose_data.py` `Keypoint` |
| Canonical 3D | Pelvis-centered meters (heuristic depth) | `coordinates/coordinate_map.py`, `GaitMotionRecording` |
| OpenSim TRC markers | Millimeters, 17 markers (`L_KNEE`, `R_HIP`, …) | `opensim_integration.py` |
| Motion reference NPZ | `(N, J, 3)` canonical positions + root | `io/motion_reference_export.py` |

Coordinate systems are documented in `docs/COORDINATE_SYSTEMS.md`.

### 2.3 Joint angles

| Source | Format | Notes |
|--------|--------|-------|
| MediaPipe kinematics | Degrees, 14+ DOF, **2D image-plane** angles | `pose/kinematics.py` → `JointAngles` |
| Pose JSON export | Schema `2.0`, per-frame `joint_angles` dict | `models/pose_data.py` |
| OpenSim `.mot` | Storage table, OpenSim coordinate names (`hip_flexion_l`, …) | `opensim_integration.py` `ANGLE_TO_OPENSIM_COORDINATE` |
| OpenSim IK output | `{run}_ik.mot` — **model joint angles** when SDK runs | `opensim_sdk.py` |
| SMPL / robot DOF | **Not present** | Future optional backends + Isaac Lab GMR |

### 2.4 Timestamps

| Location | Field | Derivation |
|----------|-------|------------|
| `PoseFrame` | `timestamp_s`, `timestamp_ms` | `frame_index / fps` or video decode timestamps |
| NPZ exports | `timestamps` `(N,)` float64 seconds | Aligned to gait snapshots |
| OpenSim TRC/MOT | `DataRate`, `DataStartTime`, per-row time column | From sequence FPS |

### 2.5 OpenSim files

| File | Path pattern | Requires SDK? |
|------|--------------|---------------|
| Marker TRC | `data/output/opensim/{run}/{run}.trc` | No (pure Python writer) |
| Mapped TRC (IK) | `{run}_mapped_for_opensim.trc` | No |
| Joint MOT | `{run}.mot` (MediaPipe angles) | No |
| JSON bundle | `{run}_opensim.json` | No |
| IK setup / output | `stablewalk_setup_ik.xml`, `{run}_ik.mot` | Yes — OpenSim Python SDK |
| Demo IK | `models/opensim/Gait2392_Pipeline/*` | Yes |

Export runs automatically after every successful video analysis (`main.py`).

---

## 3. Module inventory — Real-to-Sim related

### 3.1 Gait style extraction

| Item | Detail |
|------|--------|
| **Module** | `stablewalk/real_to_sim/gait_style_extraction.py` |
| **Status** | **Fully implemented** (offline, tested) |
| **Inputs** | `GaitMotionRecording`, `GaitCycleAnalysisResult` |
| **Outputs** | `GaitStyleFingerprint` (stride, cadence, hip/trunk sway, arm swing, confidence) |
| **Embedded in** | `stablewalk_motion.npz` → `gait_style_characteristics_json` |

### 3.2 OpenSim integration

| Item | Detail |
|------|--------|
| **Export** | `opensim_integration.py`, `opensim_marker_mapping.py`, `biomechanics/marker_reconstruction.py` |
| **SDK / IK** | `opensim_sdk.py` |
| **Status** | Export **fully implemented** without SDK; Demo IK **implemented** when SDK installed; StableWalk experimental IK **partial** (approximate markers, not clinical mocap) |
| **Docs** | `OPENSIM_STATUS.md`, README §4–6 |

### 3.3 Retargeting

| Item | Detail |
|------|--------|
| **Module** | `stablewalk/real_to_sim/retargeting.py` |
| **Config** | `models/real_to_sim/unitree_g1_retarget.json` |
| **Status** | **Partial** — uniform scale from human leg length → robot reference length |
| **Missing** | GMR, IK to robot DOF, URDF pose replay, Isaac Lab `retarget` scripts |

### 3.4 Contact detection

| Item | Detail |
|------|--------|
| **Module** | `stablewalk/analysis/gait_cycle_analysis.py` (+ phase helpers) |
| **Status** | **Fully implemented** (heuristic) |
| **Method** | Foot clearance + vertical velocity + Schmitt trigger + morphological debounce |
| **Outputs** | Per-frame `left_contact` / `right_contact`, gait events, `contact_confidence` |
| **Not** | Force-plate contact, pressure insoles, or PhysX contact |

Also: `analysis/gait_contact_debug.py`, `analysis/foot_clearance_debug.py` for QA.

### 3.5 Virtual GRF

| Item | Detail |
|------|--------|
| **Architecture** | `stablewalk/analysis/virtual_grf.py` |
| **Legacy proxy** | `stablewalk/analysis/forces.py` — `GRFAnalyzer` (CoM acceleration heuristic) |
| **Status** | **Partial** |
| | ✅ `LegacyPoseProxyForceEstimator` — wired in Real-to-Sim pipeline when `PoseSequence` available |
| | ⬜ `PhysicsSimulationForceEstimator` — placeholder (Isaac Lab ContactSensor) |
| | ⬜ `LearnedForceEstimator` — placeholder |
| **Docs** | `docs/VIRTUAL_GRF.md` |

**Terminology (required):** Contact masks = timing only. vGRF from pose = **estimated**, not measured. OpenSim ID requires measured external loads — see `opensim_id_readiness.py`.

### 3.6 Isaac Lab integration

| Item | Detail |
|------|--------|
| **Module** | `stablewalk/analysis/isaac_lab_integration.py` |
| **Status** | **Stub / documentation only** |
| **Provides** | Stage map, `check_isaac_lab_available()`, runner notes text |
| **Does not provide** | Isaac Sim install, scene spawn, AMP training, ContactSensor reads |
| **Verified in this environment** | `check_isaac_lab_available()` → `(False, "Isaac Lab not installed…")` |

### 3.7 AMP reference export

| Item | Detail |
|------|--------|
| **Module** | `stablewalk/real_to_sim/amp_reference_export.py` |
| **Orchestrator** | `stablewalk/real_to_sim/pipeline.py` |
| **CLI** | `python main.py --real-to-sim [video\|run]` |
| **GUI** | Advanced tab → Real-to-Sim Pipeline (`ui/tk/app.py`, `dashboard_sections.py`) |
| **Status** | **Export fully implemented and tested**; **AMP training not implemented** |
| **Example loader** | `scripts/load_amp_reference_example.py` |

### 3.8 Pose / HMR backend layer (future SMPL path)

| Backend | File | Status |
|---------|------|--------|
| MediaPipe | `mediapipe_backend.py` | **Production** |
| ROMP (SMPL) | `romp_backend.py` | **Placeholder** — import probe only |
| HybrIK (SMPL) | `hybrik_backend.py` | **Placeholder** |
| WHAM (SMPL) | `wham_backend.py` | **Placeholder** |
| Dedicated SMPL backend | — | **Missing** (ROMP/HybrIK/WHAM are indirect SMPL paths) |

Docs: `docs/POSE_BACKENDS.md`, `pose/backends/environment.py`

---

## 4. Implementation maturity matrix

| Component | Maturity | Runnable locally? | Notes |
|-----------|----------|-------------------|-------|
| MediaPipe pose pipeline | ✅ Full | Yes (CPU) | Default path |
| Pose JSON / overlay export | ✅ Full | Yes | |
| OpenSim TRC/MOT/JSON export | ✅ Full | Yes | No SDK required |
| OpenSim Demo IK | ✅ Full | If SDK installed | Conda `opensim-org` |
| StableWalk experimental IK | ⚠️ Partial | If SDK installed | Approximate markers |
| Gait contact detection | ✅ Full | Yes | Heuristic |
| Gait style fingerprint | ✅ Full | Yes | |
| `stablewalk_motion.npz` | ✅ Full | Yes | Schema 1.1 |
| Uniform scale retargeting | ⚠️ Partial | Yes | Not GMR |
| `amp_reference_motion.npz` | ✅ Full | Yes | Reference clip only |
| Contact-sync reward NPZ | ✅ Full | Yes | Uses pose-proxy forces |
| Virtual GRF (pose proxy) | ⚠️ Partial | Yes | Estimated, not measured |
| Virtual GRF (physics) | ⬜ Missing | No | Needs Isaac Lab |
| Isaac Lab import / training | ⬜ Missing | No | Separate env + GPU |
| SMPL / ROMP / HybrIK / WHAM | ⬜ Mocked | No | Placeholder adapters |
| Core pipeline → backend registry | ⬜ Missing | — | Registry exists but unused in main path |

---

## 5. Environment & dependencies (verified 2026-07-12)

### 5.1 Current machine snapshot

| Item | Value |
|------|-------|
| OS | Windows 10 (10.0.26200), AMD64 |
| Python | **3.10.1** |
| MediaPipe | 0.10.35 ✅ |
| PyTorch | Not installed |
| CUDA | N/A (no PyTorch) |
| OpenSim SDK | Not installed (`No module named 'opensim'`) |
| Isaac Lab | Not installed |

### 5.2 Declared dependencies (`requirements.txt`)

```
opencv-python>=4.8.0
mediapipe>=0.10.9
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=10.0.0
```

OpenSim, PyTorch, Isaac Sim/Lab, and SMPL model files are **intentionally excluded**.

### 5.3 Recommended environments

| Environment | Python | GPU | Purpose |
|-------------|--------|-----|---------|
| `stablewalk-opensim` | 3.10–3.11 | Optional | MediaPipe + GUI + OpenSim IK (CPU OK) |
| `stablewalk-hmr` | 3.10–3.11 | **Strongly recommended** | ROMP / HybrIK / WHAM / SMPL experiments |
| `stablewalk-isaac` | 3.10 (Isaac pins) | **Required** (NVIDIA RTX) | Isaac Sim + Isaac Lab + AMP training |

### 5.4 Dependency conflict risks

| Conflict | Mitigation |
|----------|------------|
| OpenSim conda stack vs PyTorch CUDA pins | Keep HMR and Isaac in **separate** envs (documented in `POSE_BACKENDS.md`) |
| Python 3.12+ vs older HMR wheels | Use 3.10/3.11 for `stablewalk-hmr` |
| MediaPipe vs heavy CUDA PyTorch in one env | Do not install PyTorch into main app env |
| Isaac Sim OS/GPU requirements | Linux preferred for Isaac; Windows supported with RTX + recent drivers — verify against NVIDIA Isaac Sim release notes for target version |

### 5.5 Can these run in the **current** environment?

| Tool | Realistic today? | Requirement |
|------|------------------|-------------|
| MediaPipe | ✅ Yes | Already installed |
| OpenSim export | ✅ Yes | Pure Python |
| OpenSim IK | ⚠️ After conda install | `conda install -c opensim-org opensim` |
| SMPL body model | ❌ No | Licensed model files + PyTorch stack |
| ROMP / HybrIK / WHAM | ❌ No | PyTorch, CUDA, repo-specific pins, adapter wiring |
| Isaac Sim / Isaac Lab | ❌ No | Separate install, NVIDIA GPU, ~tens of GB disk |
| AMP training | ❌ No | Isaac Lab + reference motion + training scripts |

### 5.6 Test status (Real-to-Sim related)

Command run: `pytest tests/test_real_to_sim_pipeline.py tests/test_virtual_grf.py tests/test_motion_reference_export.py tests/test_pose_backends.py`

**Result: 24 passed** (offline pipeline, export schemas, contact-sync math, backend registry).

Full project has **34 test modules** — preserve all during upgrades.

---

## 6. Kinematic vs physics vs measured — labeling rules

All new UI, exports, and docs must preserve these distinctions:

| Category | Examples in StableWalk | Label |
|----------|------------------------|-------|
| **Kinematic estimation** | MediaPipe landmarks, canonical joint positions, gait angles, IK joint angles | "Estimated kinematics" |
| **Heuristic timing** | Contact masks, gait phases | "Contact timing (not force)" |
| **Estimated kinetics** | `GRFAnalyzer`, `LegacyPoseProxyForceEstimator` | "Estimated Virtual GRF" |
| **Simulated kinetics** | Future Isaac ContactSensor | "Simulated GRF (physics)" |
| **Measured kinetics** | Force plates | **Not available** — do not imply |

OpenSim Inverse Dynamics with invented GRF is **explicitly out of scope** until measured `ExternalLoads.xml` exists.

---

## 7. Staged implementation plan

Each stage is independently mergeable, keeps MediaPipe as default, and defines fallback behavior.

---

### Stage 0 — Baseline lock & documentation (no behavior change)

**Goal:** Freeze current contracts before adding backends.

| Action | Files |
|--------|-------|
| Confirm NPZ schemas | `io/motion_reference_export.py`, `real_to_sim/motion_reference_loader.py` |
| Document stage status in GUI | Already in `dashboard_sections.py` |
| Add CI smoke for Real-to-Sim | `.github/workflows` or local script invoking pytest subset |

**Inputs:** Existing repo  
**Outputs:** Unchanged artifacts; baseline test green list  
**Dependencies:** None new  
**Risks:** None  
**Tests:** `tests/test_real_to_sim_pipeline.py`, `tests/test_motion_reference_export.py`  
**Fallback:** N/A  
**Runs:** Local only  

---

### Stage 1 — Wire backend registry into core pipeline (MediaPipe unchanged)

**Goal:** Single code path: `create_pose_backend("mediapipe")` → same JSON/GUI output as today.

| Create / modify | Purpose |
|-----------------|---------|
| Modify `stablewalk/core/pipeline.py` | Optional backend-driven pose pass |
| Modify `main.py` `run_step2()` | Delegate to backend or shared helper |
| Create `stablewalk/pose/backends/pipeline_adapter.py` | `HumanMotionSequence` → `PoseSequence` |
| Modify `stablewalk/config.py` | Document `POSE_BACKEND` default `mediapipe` |

**Inputs:** Video frames or BGR stream  
**Outputs:** Identical `{run}_poses.json`, overlays, OpenSim auto-export  
**Dependencies:** None beyond current `requirements.txt`  
**Risks:** Subtle timestamp/FPS drift if adapter mapping wrong  
**Tests:** Extend `tests/test_pose_backends.py`; golden JSON comparison on demo clip  
**Fallback:** `POSE_BACKEND=mediapipe` or `POSE_BACKEND_ALLOW_FALLBACK=true`  
**Runs:** Local CPU  

**Acceptance:** Bit-for-bit or tolerance-based equivalence vs current MediaPipe JSON on `data/demo_videos/normal_gait.mp4`.

---

### Stage 2 — Optional SMPL backend (ROMP first, MediaPipe fallback)

**Goal:** Add SMPL-class motion as an **opt-in** perception backend without removing MediaPipe.

| Create / modify | Purpose |
|-----------------|---------|
| Create `stablewalk/pose/backends/smpl_backend.py` | Abstract SMPL output → canonical joints + optional pose params |
| Modify `romp_backend.py` | Implement `process_video` / `process_frame` when ROMP installed |
| Modify `registry.py`, `config.py` | Register `smpl` alias → ROMP or future VIBE/etc. |
| Create `docs/SMPL_BACKEND_SETUP.md` | Conda env, model download, license note |
| Modify `real_to_sim/pipeline.py` | Record `perception_backend` in report JSON |
| Modify `io/motion_reference_export.py` | Optional `joint_rotations` from SMPL θ (if available) |

**Inputs:** Video file  
**Outputs:** Same canonical NPZ schema + optional SMPL-specific sidecar `{run}_smpl.npz` (pose β, θ, root)  
**Dependencies:** PyTorch, ROMP, SMPL model files (external `stablewalk-hmr` env)  
**Risks:** License/compliance for SMPL; coordinate frame mismatches; env breakage if mixed with OpenSim conda  
**Tests:** `tests/test_pose_backends.py` with mocked ROMP; integration test skipped unless `STABLEWALK_RUN_HMR=1`  
**Fallback:** On any backend error → log + MediaPipe if `POSE_BACKEND_ALLOW_FALLBACK=true`; else fail loud  
**Runs:** HMR env local/GPU; main app stays MediaPipe-only  

**Not in this stage:** Changing OpenSim IK to consume SMPL markers by default (research flag only).

---

### Stage 3 — HybrIK / WHAM adapters (optional, same contract)

**Goal:** Additional HMR options behind registry; comparison tooling.

| Modify | `hybrik_backend.py`, `wham_backend.py`, `pose/backends/comparison.py` |
| Modify | `scripts/compare_pose_backends.py` |

**Inputs:** Video  
**Outputs:** Canonical motion + comparison JSON  
**Dependencies:** Per-repo PyTorch/CUDA pins  
**Risks:** Maintenance burden; incompatible CUDA stacks  
**Tests:** Availability probes; comparison metrics on short clip  
**Fallback:** MediaPipe  
**Runs:** `stablewalk-hmr` env  

---

### Stage 4 — Enhanced retargeting (offline GMR prep)

**Goal:** Move from uniform scale toward GMR-ready robot joint trajectories.

| Create | `stablewalk/real_to_sim/gmr_prep.py` — end-effector targets from canonical skeleton |
| Create | `stablewalk/real_to_sim/robot_ik_offline.py` — optional PyBullet/trimesh IK (no Isaac) |
| Modify | `retargeting.py` — add method enum: `uniform_scale` \| `segment_scale` \| `gmr_targets` |
| Modify | `unitree_g1_retarget.json` — link names, end-effector map, URDF checksum field |
| Create | `scripts/export_gmr_targets.py` |

**Inputs:** `stablewalk_motion.npz` or `retargeted_motion.npz`  
**Outputs:** `gmr_targets.npz` (foot/hand positions per frame), updated `retargeted_motion.npz` metadata  
**Dependencies:** Optional PyBullet (lightweight offline IK); Isaac not required for this stage  
**Risks:** Offline IK ≠ Isaac GMR quality; wrong URDF version  
**Tests:** Scale factor sanity; foot target continuity; no NaN in targets  
**Fallback:** Keep `uniform_scale` as default method  
**Runs:** Local CPU (PyBullet) or export-only for external GMR  

---

### Stage 5 — Isaac Lab bridge (export → ingest, not training)

**Goal:** Reliable handoff from StableWalk NPZ to Isaac Lab reference loader — still **no claim of training inside StableWalk**.

| Create | `stablewalk/isaac_lab/export_validator.py` — schema checks for Isaac consumer |
| Create | `scripts/isaac_lab/load_amp_reference.py` — **runs in Isaac env**, documented copy |
| Modify | `isaac_lab_integration.py` — version pin table, URDF path validation |
| Create | `docs/ISAAC_LAB_HANDOFF.md` |

**Inputs:** `amp_reference_motion.npz`, manifest JSON  
**Outputs:** Validated bundle + human-readable handoff checklist  
**Dependencies:** Isaac Lab (external env) for validation script only  
**Risks:** Isaac API drift between releases  
**Tests:** NPZ schema unit tests (local); Isaac import test marked `@pytest.mark.isaac` skipped by default  
**Fallback:** Offline export remains valid even if Isaac missing  
**Runs:** Validator local; loader script external GPU env  

---

### Stage 6 — Physics-based virtual GRF (ContactSensor replay)

**Goal:** Replace pose-proxy vGRF with **simulated** forces when Isaac available — keep proxy as fallback.

| Modify | `virtual_grf.py` — implement `PhysicsSimulationForceEstimator.estimate()` via Isaac API |
| Create | `stablewalk/isaac_lab/contact_force_reader.py` |
| Create | `scripts/isaac_lab/replay_motion_contact_forces.py` |

**Inputs:** Retargeted motion + Unitree G1 scene  
**Outputs:** `simulated_grf.npz` (Fx,Fy,Fz per foot), updated `contact_sync_reward.npz` using simulated forces  
**Dependencies:** Isaac Sim + Isaac Lab + GPU  
**Risks:** Sim ≠ human ground truth; sim-to-real gap  
**Tests:** Compare contact-sync metric pose-proxy vs simulated on same clip; document expected divergence  
**Fallback:** `LegacyPoseProxyForceEstimator` when Isaac unavailable (current behavior)  
**Runs:** External `stablewalk-isaac` env only  

**Labeling:** GUI must show method = `physics_simulation` vs `legacy_pose_proxy`.

---

### Stage 7 — AMP training integration (external, verified)

**Goal:** Document and optionally invoke **external** AMP training repo — only mark "implemented" after executable E2E test.

| Create | `external/isaac_amp/README.md` — pin Minimal G1 AMP or official Isaac Lab AMP task |
| Create | `scripts/isaac_lab/run_amp_training.sh` — wrapper, not bundled trainer |
| Modify | `real_to_sim/pipeline.py` — stage 3 status: `export_only` \| `training_submitted` \| `training_complete` |

**Inputs:** `amp_reference_motion.npz`  
**Outputs:** Policy checkpoint path (external), training logs  
**Dependencies:** Isaac Lab AMP stack, GPU hours  
**Risks:** Training instability; license for Unitree assets  
**Tests:** Manual E2E on short clip; CI does **not** run training  
**Fallback:** Export-only mode (current)  
**Runs:** External GPU cluster / workstation  

**Honesty rule:** Do not set stage 3 status to `complete` until a pinned training script produces a checkpoint and foot contact forces are logged.

---

### Stage 8 — GUI & OpenSim preservation pass

**Goal:** Surface backend choice and force-method labels without breaking existing tabs.

| Modify | `ui/tk/app.py`, `dashboard_sections.py` | Backend selector (advanced, default MediaPipe) |
| Modify | Physics Force Estimation panel | Show estimator method + disclaimer |
| Keep | OpenSim export buttons, Demo IK, session DB | No removal |

**Tests:** `tests/test_dashboard_shell.py`, `tests/test_gui_visual_qa.py` (if applicable)  
**Fallback:** Hidden advanced settings; defaults unchanged  

---

## 8. Artifact contract summary (post-upgrade target)

```
data/output/motion_reference/{run}/
├── stablewalk_motion.npz          # Stage 1 — human canonical motion
├── retargeted_motion.npz          # Stage 2 — scaled / GMR-prep motion
├── gmr_targets.npz                # Stage 4 — optional end-effector targets
├── amp_reference_motion.npz       # Stage 3 — AMP reference clip
├── amp_reference_manifest.json
├── contact_sync_reward.npz        # Stage 4/6 — sync vs estimated or simulated GRF
├── simulated_grf.npz              # Stage 6 — optional Isaac physics output
├── {run}_smpl.npz                 # Stage 2 — optional SMPL params
└── real_to_sim_pipeline_report.json
```

Existing OpenSim and pose paths under `data/output/opensim/` and `data/output/poses/` remain unchanged.

---

## 9. Priority recommendation

| Priority | Stage | Rationale |
|----------|-------|-----------|
| P0 | Stage 0 | Lock tests |
| P1 | Stage 1 | Enables all backends without breaking MediaPipe |
| P2 | Stage 2 | SMPL optional path (research spec) |
| P3 | Stage 4 | Better retargeting before Isaac |
| P4 | Stage 5–6 | Isaac handoff + simulated GRF |
| P5 | Stage 7 | AMP training (external) |
| P6 | Stage 3, 8 | Extra HMR backends + GUI polish |

---

## 10. References (in-repo)

| Document | Topic |
|----------|-------|
| `docs/REAL_TO_SIM_PIPELINE.md` | Current 4-stage overview |
| `docs/SPEC_COMPLIANCE.md` | Research spec mapping |
| `docs/POSE_BACKENDS.md` | Backend architecture |
| `docs/VIRTUAL_GRF.md` | Force terminology |
| `docs/COORDINATE_SYSTEMS.md` | Spatial conventions |
| `OPENSIM_STATUS.md` | OpenSim IK verification |
| `README.md` | User-facing pipeline |

---

## 11. Out of scope for this upgrade (explicit)

- Removing or replacing MediaPipe as default
- Claiming Isaac Lab AMP training is built into StableWalk before Stage 7 E2E proof
- Using estimated vGRF as OpenSim `ExternalLoads.xml`
- Clinical-grade accuracy claims for monocular video
- Bundling SMPL licensed model weights in the repository

---

*End of review. Implementation should begin at Stage 0/1 only after explicit approval.*
