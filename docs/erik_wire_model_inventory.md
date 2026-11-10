# Wire gait model inventory (supervisor 12–20)

This is an audit of the **existing** StableWalk kinematic wire. It is **not** OpenSim Gait2392, not a scaled subject model, and not 33 independent degrees of freedom.

**Classification:** the supervisor range **12–20 is satisfied** by independent sagittal DOFs (**13**) and by unique anatomical joints / unique connected segments (**19**). It is **not** satisfied if 21 named nodes (including two derived midfoot aliases) or 33 MediaPipe landmarks are counted as independent DOFs. Landmarks ≠ joints ≠ segments ≠ DOFs ≠ OpenSim coordinates.

## Separate counts

| Inventory | Count | What it is | In 12–20? |
|-----------|------:|------------|-----------|
| Detected pose landmarks | **33** | MediaPipe BlazePose `PoseLandmark` names | No — detections, not DOFs |
| Reconstructed 3D joints | **15** | `SKELETON_3D_JOINTS` (nose, mid_shoulder, mid_hip, limbs to ankles) | Yes as a reconstruction subset |
| Trackable named joints | **21** | `JOINT_IDS` (includes derived `left_foot` / `right_foot`) | Above range if counted as independent components |
| Pelvis root | **1** | `ROOT_JOINT_ID = pelvis` (not listed in `JOINT_IDS`) | Root, not an extra DOF |
| Connected skeleton bones | **21** | `SKELETON_BONES` (one parent edge per named joint) | Inflated by two midfoot aliases |
| Unique anatomical joints | **19** | `JOINT_IDS` minus derived midfoot | Yes |
| Unique connected segments | **19** | Kinematic tree without midfoot aliases | Yes |
| Independently modeled DOFs | **13** | Sagittal `DOF_IDS` | **Yes** |
| OpenSim Gait2392 coordinates | **23** | Published `Gait2392_Simbody` CoordinateSet | Separate model; not this wire |
| Mapped OpenSim aliases | **13** | `OPENSIM_COORDINATE_ALIASES` onto the 13 wire DOFs | Aliases, not extra DOFs |

Exported pose angle fields (`GAIT_ANGLE_FIELDS`) are **15** names. `torso_tilt`, `pelvis_rotation`, and `spine` all map to **`spine_flexion`**. They are not three extra independent DOFs.

## Supervisor 12–20

| Reading of “12–20” | Value | Result |
|--------------------|------:|--------|
| Independent sagittal DOFs | 13 | **Satisfied** |
| Unique anatomical joints (no duplicate midfoot) | 19 | **Satisfied** |
| Unique connected segments | 19 | **Satisfied** |
| 15-joint reconstruction | 15 | **Satisfied** |
| 21 `JOINT_IDS` + pelvis | 22 | Above range — two nodes are contact aliases |
| 33 MediaPipe landmarks | 33 | Must not be used |

**Do not confuse landmarks with independent degrees of freedom.**

## Inventory table

Joint or segment name | Parent | Child | X/Y/Z availability | DOF | Anatomical interpretation | Confidence
---|---|---|---|---|---|---
pelvis | — | pelvis | XYZ reconstructed (mid_hip) or L/R hip midpoint | none on wire (Gait2392 pelvis_tilt/list/rotation aliased → spine_flexion) | Pelvic origin | high as mid-hip
pelvis→spine | pelvis | spine | XYZ derived (pelvis/shoulder/head) | spine_flexion | Thoracolumbar trunk node | medium — interpolated
spine→neck | spine | neck | XYZ derived | neck | Cervical node | medium — interpolated
neck→head | neck | head | XYZ reconstructed (nose) | head_neck | Cranial landmark | medium — nose mapping
spine→left_shoulder | spine | left_shoulder | XYZ reconstructed; Z heuristic | left_shoulder_flexion | Left glenohumeral joint | medium
spine→right_shoulder | spine | right_shoulder | XYZ reconstructed; Z heuristic | right_shoulder_flexion | Right glenohumeral joint | medium
left_shoulder→left_elbow | left_shoulder | left_elbow | XYZ reconstructed; Z heuristic | left_elbow_flexion | Left elbow | medium
right_shoulder→right_elbow | right_shoulder | right_elbow | XYZ reconstructed; Z heuristic | right_elbow_flexion | Right elbow | medium
left_elbow→left_wrist | left_elbow | left_wrist | XYZ reconstructed; Z heuristic | none | Left wrist | medium
right_elbow→right_wrist | right_elbow | right_wrist | XYZ reconstructed; Z heuristic | none | Right wrist | medium
pelvis→left_hip | pelvis | left_hip | XYZ reconstructed; Z heuristic | left_hip_flexion | Left hip | medium
pelvis→right_hip | pelvis | right_hip | XYZ reconstructed; Z heuristic | right_hip_flexion | Right hip | medium
left_hip→left_knee | left_hip | left_knee | XYZ reconstructed; Z heuristic | left_knee_flexion | Left knee | medium
right_hip→right_knee | right_hip | right_knee | XYZ reconstructed; Z heuristic | right_knee_flexion | Right knee | medium
left_knee→left_ankle | left_knee | left_ankle | XYZ reconstructed; Z heuristic | left_ankle_flexion | Left ankle | medium
right_knee→right_ankle | right_knee | right_ankle | XYZ reconstructed; Z heuristic | right_ankle_flexion | Right ankle | medium
left_ankle→left_heel | left_ankle | left_heel | XYZ from MediaPipe heel; ankle-offset fallback | none | Left calcaneus | medium
right_ankle→right_heel | right_ankle | right_heel | XYZ from MediaPipe heel; ankle-offset fallback | none | Right calcaneus | medium
left_heel→left_toe | left_heel | left_toe | XYZ from MediaPipe foot_index; fallback | none | Left hallux / foot index | medium
right_heel→right_toe | right_heel | right_toe | XYZ from MediaPipe foot_index; fallback | none | Right hallux / foot index | medium
left_foot | left_ankle | left_foot | XYZ derived (heel–toe midpoint) | none | Left midfoot contact alias | high as alias; **not an independent DOF**
right_foot | right_ankle | right_foot | XYZ derived (heel–toe midpoint) | none | Right midfoot contact alias | high as alias; **not an independent DOF**

Source of truth: `wire_model_inventory_rows()` in `stablewalk/models/joint_registry.py`.

## Missing and duplicated

### Missing from the 15-joint reconstruction array (present on the gait wire)

- Head (stored as `nose` in reconstruction, mapped to `head`)
- Neck, spine (derived)
- Heel, toe / foot_index (now reconstructed in the same hip-centered frame, then mapped)
- MediaPipe eyes, ears, mouth, thumbs, pinkies — unused on purpose

### Missing from OpenSim Gait2392 mapping

- `pelvis_tx`, `pelvis_ty`, `pelvis_tz`
- Hip adduction / rotation (L/R)
- Subtalar and MTP (L/R)
- Lumbar bending / rotation (extension is the closest analog of `spine_flexion`)
- Elbow/wrist markers are not in the lower-body Gait2392 IK task set

`pelvis_tilt`, `pelvis_list`, and `pelvis_rotation` are **all aliased to `spine_flexion`**. That is one wire DOF, not three.

### Duplicates (corrected)

- **`left_foot` / `right_foot`** were a third ankle-offset node overlapping heel and toe. They are now the **heel–toe midpoint** (contact alias), not an independent segment.
- Display **hip→shoulder rails** plus filled torso plus neck→shoulder sticks overlapped. Torso sticks remain skipped; the filled torso is driven by L/R shoulder and L/R hip.
- Display **ankle–heel–toe cylinders** overlapped the anatomical foot plate. Those sticks are skipped; the sole is locked to reconstructed heel and toe.
- Image-normalized heel/toe (`frame.positions`, Y down) were mixed into hip-centered XYZ, which placed feet above the ankle so they looked missing or fused. Image-frame feet are no longer used as 3D.

## Bilateral completeness

| Region | Present on wire | Driven by |
|--------|-----------------|-----------|
| Pelvis | yes | mid_hip or L/R hip midpoint |
| Torso | yes (`spine`) | derived from pelvis and shoulder mid |
| Head / neck | yes | nose → head; neck interpolated |
| L/R hips | yes | MediaPipe hips |
| L/R knees | yes | MediaPipe knees |
| L/R ankles | yes | MediaPipe ankles |
| L/R shoulders | yes | MediaPipe shoulders |
| L/R elbows | yes | MediaPipe elbows |
| L/R wrists | yes | MediaPipe wrists |
| Feet | yes (heel + toe; midfoot alias) | MediaPipe heel / foot_index, same 3D frame as the limbs |

## Why limbs overlapped or looked missing

1. **3/4 walking camera (35°)** projects both legs onto nearby screen positions during double support. That is a view, not a missing joint.
2. **Duplicate foot paint:** sole plate + ankle–heel–toe limb cylinders occupied the same pixels. The extra cylinders are no longer drawn.
3. **Mixed coordinate frames:** Y-down image heels sat above hip-centered ankles and were then replaced by a short synthetic offset, so feet collapsed into the ankle.
4. **Unprojected leftover joints** in the walking view used raw XY while limbs used rotated XZ. Every snapshot joint is now projected with the same camera.
5. **Very short foot bones** could fail the generic minimum-length gate. Foot segments use a lower length floor.

## Displayed segments are driven

Walking / biomechanical view:

- **Torso volume** — L/R shoulder and L/R hip (measured).
- **Head** — reconstructed `head` (nose).
- **Arms** — shoulder→elbow→wrist (measured).
- **Legs** — hip→knee→ankle (measured).
- **Feet** — sole locked to heel and toe (reconstructed in the canonical frame; midfoot is their midpoint).

No displayed segment is an undriven placeholder bone. Z remains a **heuristic** monocular depth, not optical motion capture.

## Corrections applied in this pass

- Reconstruct MediaPipe heel / foot_index in the same hip-centered, scaled frame as the 15-joint core; keep the 15-column reconstruction arrays unchanged.
- Drive `left_foot` / `right_foot` from heel and toe.
- Do not copy image-frame feet into canonical XYZ.
- Project every snapshot joint with the walking camera.
- Skip duplicate torso rails and duplicate foot cylinders in the viewer.

Gait math, IK, playback, and the 13 sagittal DOFs were not rewritten. Physical joint torques are still refused without GRF. This wire is still `monocular_wire_skeleton`, not Gait2392.
