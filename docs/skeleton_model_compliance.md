# Skeleton model compliance (Task 3)

**Date:** 2026-10-14  
**Model kind:** `monocular_wire_skeleton` (`WIRE_MODEL_KIND`)  
**Not this model:** OpenSim Gait2392 / Gait2354, scaled subject `.osim`, 33 independent DOFs, RL/Isaac.

This audit counts **what the code actually stores, connects, solves, and draws**. MediaPipe detecting 33 landmarks is **not** a 12–20 DOF gait model.

Joints were not fabricated to hit 12–20. Midfoot `left_foot` / `right_foot` remain **contact aliases** (heel–toe midpoint), not extra DOFs.

## 1. Separate counts

Source: `wire_model_count_summary()` in `stablewalk/models/joint_registry.py` (2026-10-14).

| Inventory | Count | What it is | Treat as 12–20? |
|---|---:|---|---|
| Input pose landmarks | **33** | MediaPipe BlazePose `PoseLandmark` | **No** — detections |
| Unique 3D reconstructed joints | **15** | `SKELETON_3D_JOINTS` (nose, mid_shoulder, mid_hip, limbs to ankles) | Yes as reconstruction core |
| Extra feet in the same 3D frame | **4** | `FOOT_3D_LANDMARKS` (heels + foot_index); not packed into the 15-column array | Landmarks, not extra DOFs |
| Named trackable joints | **21** | `JOINT_IDS` (includes 2 midfoot aliases) | **Above** range if counted as independent components |
| Pelvis root | **1** | `ROOT_JOINT_ID = pelvis` (not in `JOINT_IDS`) | Root, not an extra DOF |
| Connected skeleton bones (`SKELETON_BONES`) | **21** | One parent edge per named joint | Inflated by two aliases |
| Unique anatomical joints | **19** | `JOINT_IDS` minus `left_foot` / `right_foot` | **Yes** |
| Unique connected segments | **19** | Kinematic tree without midfoot aliases | **Yes** |
| Independently modeled DOFs | **13** | Sagittal `DOF_IDS` | **Yes** |
| OpenSim Gait2392 coordinates | **23** | Published `Gait2392_Simbody` CoordinateSet | **Separate model** |
| Mapped OpenSim name aliases | **13** | `OPENSIM_COORDINATE_ALIASES` onto the 13 wire DOFs | Aliases, not extra DOFs |
| Video-driven joints | **18** | MediaPipe (or mid_hip / L–R hip midpoint); not spine/neck/midfoot | Driven nodes |
| Joints visualized (Overview gait markers) | **14** | `WIRE_VISUALIZED_GAIT_MARKERS` (shoulders–heels) | Display glyphs |
| Limb sticks visualized | **8** | Hip–knee–ankle and shoulder–elbow–wrist, both sides | Display segments |
| GUI-selectable analysis joints | **16** | `GUI_DOF_ITEM_IDS` (14 markers + toes) | UI pick list, not DOFs |

`GAIT_ANGLE_FIELDS` in `stablewalk/pose/dof.py` lists **15 names**. `torso_tilt`, `pelvis_rotation`, and `spine` all map to **`spine_flexion`**. They are not three extra independent DOFs.

Z is **heuristic monocular depth**, not optical motion capture.

## 2. Real skeletal hierarchy

`JOINT_PARENTS` (`stablewalk/models/joint_registry.py`):

```
pelvis (root)
├─ spine          [interpolated from pelvis + shoulder mid]
│  ├─ neck        [interpolated from spine + head]
│  │  └─ head     ← MediaPipe nose
│  ├─ left_shoulder → left_elbow → left_wrist
│  └─ right_shoulder → right_elbow → right_wrist
├─ left_hip → left_knee → left_ankle
│     ├─ left_heel → left_toe
│     └─ left_foot   [heel–toe midpoint alias]
└─ right_hip → right_knee → right_ankle
      ├─ right_heel → right_toe
      └─ right_foot  [heel–toe midpoint alias]
```

Reconstruction graph (`SKELETON_3D_CONNECTIONS`, 14 edges) uses `mid_hip` / `mid_shoulder` / `nose` instead of pelvis / spine / head. The adapter maps those names onto the canonical tree (`stablewalk/adapters/pose_adapter.py`).

## 3. Component table

Each row is one connected unique segment (or the pelvis root, or a midfoot alias). Columns match the Skeleton Model Inspector.

| Name | Parent | Child | Position source | Wire rotational DOF | OpenSim rotational (Gait2392) | Video-driven | Visualized | Confidence |
|---|---|---|---|---|---|---|---|---|
| pelvis | — | pelvis | XYZ reconstructed (mid_hip) or derived (L/R hip midpoint) | none on wire (Gait2392 pelvis_tilt/list/rotation aliased → spine_flexion) | pelvis_tilt, pelvis_list, pelvis_rotation (not independent wire DOFs) | yes — MediaPipe mid_hip or L/R hip midpoint | torso volume (stick rail skipped in gait view) | high as mid_hip / L/R hip midpoint |
| pelvis→spine | pelvis | spine | XYZ derived (pelvis/shoulder/head) | spine_flexion | lumbar_extension, lumbar_bending, lumbar_rotation | no — interpolated | torso volume | medium — interpolated |
| spine→neck | spine | neck | XYZ derived | neck | none in Gait2392 | no — interpolated | torso volume | medium — interpolated |
| neck→head | neck | head | XYZ reconstructed (Z heuristic) | head_neck | none in Gait2392 | yes — MediaPipe nose | cranial oval | medium — nose mapping |
| spine→left_shoulder | spine | left_shoulder | XYZ reconstructed (Z heuristic) | left_shoulder_flexion | none in Gait2392 | yes | gait marker + limb cylinder | medium |
| spine→right_shoulder | spine | right_shoulder | XYZ reconstructed (Z heuristic) | right_shoulder_flexion | none in Gait2392 | yes | gait marker + limb cylinder | medium |
| left_shoulder→left_elbow | left_shoulder | left_elbow | XYZ reconstructed (Z heuristic) | left_elbow_flexion | none in Gait2392 | yes | gait marker + limb cylinder | medium |
| right_shoulder→right_elbow | right_shoulder | right_elbow | XYZ reconstructed (Z heuristic) | right_elbow_flexion | none in Gait2392 | yes | gait marker + limb cylinder | medium |
| left_elbow→left_wrist | left_elbow | left_wrist | XYZ reconstructed (Z heuristic) | none (endpoint, not an independent DOF) | none in Gait2392 | yes | gait marker + limb cylinder | medium |
| right_elbow→right_wrist | right_elbow | right_wrist | XYZ reconstructed (Z heuristic) | none | none in Gait2392 | yes | gait marker + limb cylinder | medium |
| pelvis→left_hip | pelvis | left_hip | XYZ reconstructed (Z heuristic) | left_hip_flexion | hip_flexion_l, hip_adduction_l, hip_rotation_l | yes | gait marker + limb cylinder | medium |
| pelvis→right_hip | pelvis | right_hip | XYZ reconstructed (Z heuristic) | right_hip_flexion | hip_flexion_r, hip_adduction_r, hip_rotation_r | yes | gait marker + limb cylinder | medium |
| left_hip→left_knee | left_hip | left_knee | XYZ reconstructed (Z heuristic) | left_knee_flexion | knee_angle_l | yes | gait marker + limb cylinder | medium |
| right_hip→right_knee | right_hip | right_knee | XYZ reconstructed (Z heuristic) | right_knee_flexion | knee_angle_r | yes | gait marker + limb cylinder | medium |
| left_knee→left_ankle | left_knee | left_ankle | XYZ reconstructed (Z heuristic) | left_ankle_flexion | ankle_angle_l, subtalar_angle_l, mtp_angle_l | yes | gait marker + limb cylinder | medium |
| right_knee→right_ankle | right_knee | right_ankle | XYZ reconstructed (Z heuristic) | right_ankle_flexion | ankle_angle_r, subtalar_angle_r, mtp_angle_r | yes | gait marker + limb cylinder | medium |
| left_ankle→left_heel | left_ankle | left_heel | MediaPipe heel; ankle-offset fallback | none | none in Gait2392 | yes (fallback if invalid) | gait marker | medium |
| right_ankle→right_heel | right_ankle | right_heel | MediaPipe heel; ankle-offset fallback | none | none in Gait2392 | yes (fallback if invalid) | gait marker | medium |
| left_heel→left_toe | left_heel | left_toe | MediaPipe foot_index; ankle-offset fallback | none | none in Gait2392 | yes (fallback if invalid) | foot plate (not a default gait marker) | medium |
| right_heel→right_toe | right_heel | right_toe | MediaPipe foot_index; ankle-offset fallback | none | none in Gait2392 | yes (fallback if invalid) | foot plate (not a default gait marker) | medium |
| left_foot | left_ankle | left_foot | XYZ derived (heel–toe midpoint) | none | none in Gait2392 | **no** — alias | no stick | high as alias; **not a DOF** |
| right_foot | right_ankle | right_foot | XYZ derived (heel–toe midpoint) | none | none in Gait2392 | **no** — alias | no stick | high as alias; **not a DOF** |

**Wire rotational DOFs (exactly 13):** `head_neck`, `neck`, `spine_flexion`, `left/right_shoulder_flexion`, `left/right_elbow_flexion`, `left/right_hip_flexion`, `left/right_knee_flexion`, `left/right_ankle_flexion`. Each is **one sagittal angle**. Wrists, heels, and toes have positions but no independent wire DOF.

**OpenSim rotational columns** in the table are *anatomical analogues* on Gait2392. They are **not** solved on this wire unless a separate IK `.mot` is loaded. Gait2392 has **no** shoulder/elbow/wrist coordinates. `pelvis_tx/ty/tz` exist on Gait2392 (translations) and are **not** listed as rotational DOFs here.

## 4. Both sides of the body

Required bilateral set (`WIRE_BILATERAL_REQUIRED`): pelvis, spine, neck, head, and L/R hip, knee, ankle, shoulder, elbow, wrist, heel, toe.

| Region | On the wire | Driven by |
|---|---|---|
| Pelvis | yes | mid_hip or L/R hip midpoint |
| Torso | yes (`spine`) | interpolated |
| Head / neck | yes | nose → head; neck interpolated |
| L/R shoulders, elbows, wrists | yes | MediaPipe |
| L/R hips, knees, ankles | yes | MediaPipe |
| L/R heels, toes | yes | MediaPipe heel / foot_index (canonical 3D frame) |

Tested on synthetic **asymmetric** wrists (left raised, right lowered) and on `normal_gait` / `abnormal_gait` at frames 0 / 25% / 50% / 75% / last. All required left and right nodes were present; both arm and both leg chains were connected.

## 5. Arms and legs do not disappear from bad connections

Causes of “missing” limbs in earlier work (coordinate-frame mix, duplicate foot sticks, unprojected leftover joints) are documented in `docs/erik_wire_model_inventory.md`. This audit checked the **current** mapping:

- Adapter skips image-frame Y-down feet when filling canonical XYZ (`pose_adapter._extract_joint_positions`).
- Unique stick segments in Overview gait view are the **eight** limb cylinders; torso is a filled volume; feet are sole plates locked to heel/toe.
- `_build_display_coords` produced distinct left and right wrist points on the asymmetric pose, and all eight limb sticks on every demo quartile frame.

A 3/4 walking camera can still **project** both legs onto nearby pixels in double support. That is viewpoint overlap, not a missing joint.

## 6. Skeleton Model Inspector

The inspector shows the **same counts as this report**, plus the current frame’s live joint/segment/DOF presence and bilateral completeness.

| Where | How |
|---|---|
| View menu | **Skeleton Model Inspector** |
| Skeleton Advanced panel | **Model** |
| Data | `stablewalk/ui/skeleton_model_inspector.py` |
| Window | `stablewalk/ui/tk/skeleton_model_inspector_window.py` |

Tabs: **Components** (parent/child, position source, wire DOF, Gait2392 analogue, video-driven, visualized, confidence), **Wire DOFs** (13 sagittal angles for the playhead), **Hierarchy**, **12–20 readings**.

Overview splitter layout (Task 1) is unchanged. The inspector is a detachable window, refreshed on playback with the Coordinate Inspector.

## 7. Changing input coordinates updates the skeleton and export

Perturbing `left_knee` XYZ on a `normal_gait` snapshot (frame 30):

- Display hip–knee and knee–ankle endpoints **changed** (`_build_display_coords`).
- OpenSim marker table `LKNE` **changed** to the new XYZ (`export_to_opensim_format(..., include_markers=True)`).
- Geometric angle at the knee (hip–knee–ankle) **changed** by > 0.5°.

**Limitation:** `.mot`-style angle columns from `motion_to_opensim_table` use **stored** `snapshot.dofs`. Pose JSON angles are **not** automatically recomputed when a marker is moved. Position-based export (TRC / marker table / drawn segments) follows the new coordinates; stored sagittal DOFs do not unless `calculate_degrees_of_freedom` fills a missing angle (and that helper currently estimates the angle at the **parent** using grandparent–parent–child, which is not a Gait2392 IK step).

## 8. Tests

```
python -m pytest tests/test_skeleton_model_compliance.py tests/test_wire_model_inventory.py -q
```

**Result (2026-10-14): 22 passed.**

| Test | What it checks |
|---|---|
| `test_inventories_are_separate_and_not_fabricated` | 33 ≠ 15 ≠ 19 ≠ 13 ≠ 23 |
| `test_supervisor_verdicts_are_honest` | 12–20 yes/no per interpretation |
| `test_inventory_rows_include_drive_and_opensim_columns` | video-driven / OpenSim columns |
| `test_hierarchy_is_a_tree_from_pelvis` | `JOINT_PARENTS` |
| `test_asymmetric_pose_keeps_both_arms_and_legs` | synthetic raised/lowered wrists |
| `test_perturbing_knee_moves_segment_and_exported_markers` | display + `LKNE` export |
| `test_real_demo_frames_are_bilateral_and_connected` | `normal_gait` / `abnormal_gait` quartiles |
| `test_inspector_fields_show_separate_counts` | inspector numbers |
| existing `test_wire_model_inventory.py` | reconstruction 15-core + driven feet |

## 9. Fabrication

No joints or DOFs were added to inflate the count. `left_foot` / `right_foot` stay aliases. Eyes, ears, mouth, thumbs, and pinkies remain unused on purpose.

## 10. Does this satisfy “approximately 12–20 components or degrees of freedom”?

The phrase is **ambiguous**. Verdicts from `supervisor_12_20_verdicts()`:

| If 12–20 means… | Count | Result |
|---|---:|---|
| Independent sagittal DOFs | 13 | **Satisfies** |
| Unique anatomical joints (no midfoot aliases) | 19 | **Satisfies** |
| Unique connected segments (no midfoot aliases) | 19 | **Satisfies** |
| 15-joint 3D reconstruction core | 15 | **Satisfies** |
| Overview gait joint markers | 14 | **Satisfies** |
| GUI-selectable analysis joints | 16 | **Satisfies** |
| Mapped OpenSim **name** aliases | 13 | **Satisfies as aliases only** — not extra DOFs |
| Limb stick segments drawn in gait view | 8 | **Below** 12–20 |
| Named `JOINT_IDS` (with midfoot aliases) | 21 | **Above** — do not use |
| `JOINT_IDS` + pelvis | 22 | **Above** — do not use |
| MediaPipe Pose landmarks | 33 | **Above** — detections, not DOFs |
| OpenSim Gait2392 CoordinateSet | 23 | **Above** — different model |

**Recommended reading for a wire gait figure:** **13 independent sagittal DOFs** on **19 unique anatomical joints / segments**. That is inside 12–20 and matches a stick-figure gait model, not Gait2392.

### Clarification question for the supervisor

Should **12–20** count:

1. **Independent sagittal degrees of freedom** on the wire (currently **13**), or  
2. **Unique anatomical joints / connected segments** (currently **19**), or  
3. **OpenSim generalized coordinates** on a scaled Gait2392-class model (currently **23** published coordinates, **not** solved here)?

Please do **not** count MediaPipe’s **33** landmarks as degrees of freedom.

If (3) is required, this repository still needs a scaled `.osim`, marker IK, and a CoordinateSet that is actually integrated — that work is **not** implied by the current wire.

## Source files

| Role | Path |
|---|---|
| Canonical joints, parents, DOFs, counts | `stablewalk/models/joint_registry.py` |
| Pose → snapshot mapping | `stablewalk/adapters/pose_adapter.py` |
| 15-joint + foot reconstruction | `stablewalk/pose/skeleton_3d.py` |
| Angle field names (15 labels → 13 DOFs) | `stablewalk/pose/dof.py` |
| GUI pick list (16) / primary markers (14) | `stablewalk/ui/dof_selection.py` |
| Drawn sticks / soles / torso | `stablewalk/ui/viewers/gait_skeleton_renderer.py` |
| OpenSim angle table / marker placeholder | `stablewalk/biomechanics/service.py`, `adapters/opensim_schema.py` |
| Gait2392 marker names (export, not this wire’s DOFs) | `stablewalk/biomechanics/marker_reconstruction.py` |
| Inspector | `stablewalk/ui/skeleton_model_inspector.py`, `stablewalk/ui/tk/skeleton_model_inspector_window.py` |
| Tests | `tests/test_skeleton_model_compliance.py`, `tests/test_wire_model_inventory.py` |

## Known limitations

- Depth Z is heuristic. Do not claim metric 3D or calibrated millimetres.
- One sagittal angle per listed DOF. No hip abduction/rotation, no lumbar bending/rotation, no subtalar/MTP **on the wire**.
- Gait2392 coordinates are a **published list and name aliases**, not a live OpenSim `State`.
- Overview gait markers omit toes (toes remain in the GUI dropdown and on the sole).
- Stored pose-JSON DOF angles do not auto-update when XYZ is edited; marker/segment geometry does.
- `BiomechanicsService._estimate_joint_angle` is a parent-centered vector heuristic, not IK.

Parent Overview layout (Task 1) and viewport fitting (Task 2) were not redesigned for this audit.
