"""Task 3: skeleton / DOF audit — counts, laterality, drive-through, inspector."""

from __future__ import annotations

from copy import deepcopy

import pytest

from stablewalk.adapters.pose_adapter import pose_frame_to_snapshot, pose_sequence_to_gait_motion
from stablewalk.biomechanics.service import export_to_opensim_format
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, Vec3
from stablewalk.models.joint_registry import (
    DOF_IDS,
    JOINT_IDS,
    JOINT_PARENTS,
    MEDIAPIPE_POSE_LANDMARK_COUNT,
    ROOT_JOINT_ID,
    WIRE_BILATERAL_REQUIRED,
    WIRE_DISPLAY_STICK_SEGMENTS,
    WIRE_INDEPENDENT_DOF_COUNT,
    WIRE_UNIQUE_ANATOMICAL_JOINT_COUNT,
    WIRE_UNIQUE_CONNECTED_SEGMENT_COUNT,
    WIRE_VIDEO_DRIVEN_JOINTS,
    WIRE_VISUALIZED_GAIT_MARKERS,
    live_snapshot_counts,
    supervisor_12_20_verdicts,
    wire_hierarchy_lines,
    wire_model_count_summary,
    wire_model_inventory_rows,
)
from stablewalk.models.pose_data import Keypoint, PoseFrame
from stablewalk.ui.skeleton_model_inspector import (
    build_skeleton_model_inspector_fields,
    inspector_component_rows,
    limbs_connected,
)
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    DEFAULT_SKELETON_DISPLAY_MODE,
    _build_display_coords,
)
from stablewalk import config


def _standing_keypoints(*, left_wrist_y: float = 0.56, right_wrist_y: float = 0.56) -> list[Keypoint]:
    pts = {
        "nose": (0.50, 0.18, 0.02),
        "left_shoulder": (0.42, 0.32, 0.01),
        "right_shoulder": (0.58, 0.32, 0.01),
        "left_elbow": (0.38, 0.44, 0.03),
        "right_elbow": (0.62, 0.44, 0.03),
        "left_wrist": (0.36, left_wrist_y, 0.04),
        "right_wrist": (0.64, right_wrist_y, 0.04),
        "left_hip": (0.46, 0.55, 0.00),
        "right_hip": (0.54, 0.55, 0.00),
        "left_knee": (0.46, 0.72, -0.02),
        "right_knee": (0.54, 0.72, -0.02),
        "left_ankle": (0.46, 0.88, -0.04),
        "right_ankle": (0.54, 0.88, -0.04),
        "left_heel": (0.45, 0.91, -0.05),
        "right_heel": (0.55, 0.91, -0.05),
        "left_foot_index": (0.47, 0.94, -0.03),
        "right_foot_index": (0.53, 0.94, -0.03),
    }
    return [
        Keypoint(name=n, x=x, y=y, z=z, visibility=0.95)
        for n, (x, y, z) in pts.items()
    ]


def _pose_frame(**kwargs) -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        image_path="synthetic",
        keypoints=_standing_keypoints(**kwargs),
        detected=True,
    )


def _pose_rec(stem: str):
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return pose_sequence_to_gait_motion(load_pose_sequence(path))


def test_inventories_are_separate_and_not_fabricated() -> None:
    counts = wire_model_count_summary()
    assert counts["detected_pose_landmarks"] == MEDIAPIPE_POSE_LANDMARK_COUNT == 33
    assert counts["independent_sagittal_dofs"] == WIRE_INDEPENDENT_DOF_COUNT == 13
    assert counts["unique_anatomical_joints"] == 19
    assert counts["unique_connected_segments"] == 19
    assert counts["visualized_limb_stick_segments"] == len(WIRE_DISPLAY_STICK_SEGMENTS) == 8
    assert counts["visualized_gait_markers"] == len(WIRE_VISUALIZED_GAIT_MARKERS) == 14
    assert counts["opensim_gait2392_coordinates"] == 23
    # Do not add these into one fake “model size”.
    assert len({counts[k] for k in (
        "detected_pose_landmarks",
        "reconstructed_3d_joints",
        "unique_connected_segments",
        "independent_sagittal_dofs",
        "opensim_gait2392_coordinates",
    )}) == 5


def test_supervisor_verdicts_are_honest() -> None:
    by_name = {row["interpretation"]: row for row in supervisor_12_20_verdicts()}
    assert "satisfies" in by_name["Independent sagittal DOFs"]["verdict"]
    assert "satisfies" in by_name["Unique anatomical joints (no midfoot aliases)"]["verdict"]
    assert "above" in by_name["MediaPipe Pose landmarks"]["verdict"]
    assert "above" in by_name["OpenSim Gait2392 coordinates"]["verdict"]
    assert "above" in by_name["Named JOINT_IDS (includes 2 midfoot aliases)"]["verdict"]


def test_inventory_rows_include_drive_and_opensim_columns() -> None:
    rows = inspector_component_rows()
    extra = {"video_driven", "opensim_rotational", "visualized"}
    assert all(extra <= row.keys() for row in rows)
    left_knee = next(r for r in rows if r["child"] == "left_knee")
    assert "yes" in left_knee["video_driven"]
    assert "left_knee_flexion" in left_knee["dof"]
    assert "knee_angle_l" in left_knee["opensim_rotational"]
    foot = next(r for r in rows if r["name"] == "left_foot")
    assert "no" in foot["video_driven"]
    assert "not an independent" in foot["dof"].lower() or "none" in foot["dof"].lower()


def test_hierarchy_is_a_tree_from_pelvis() -> None:
    text = "\n".join(wire_hierarchy_lines())
    assert text.startswith("pelvis")
    assert JOINT_PARENTS["left_knee"] == "left_hip"
    assert JOINT_PARENTS["left_hip"] == ROOT_JOINT_ID
    for jid in JOINT_IDS:
        assert jid in JOINT_PARENTS
        parent = JOINT_PARENTS[jid]
        if parent is not None:
            assert parent == ROOT_JOINT_ID or parent in JOINT_IDS or parent in JOINT_PARENTS


def test_asymmetric_pose_keeps_both_arms_and_legs() -> None:
    snap = pose_frame_to_snapshot(
        _pose_frame(left_wrist_y=0.22, right_wrist_y=0.70),
        fps=30.0,
    )
    chains = limbs_connected(snap)
    assert all(chains.values()), chains
    for jid in WIRE_BILATERAL_REQUIRED:
        assert jid in snap.joints, jid
    lw = snap.joints["left_wrist"].position
    rw = snap.joints["right_wrist"].position
    display = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
    assert "left_wrist" in display and "right_wrist" in display
    # Raised left wrist stays distinct from the lowered right wrist.
    assert abs(lw.y - rw.y) > 0.05
    assert abs(display["left_wrist"][1] - display["right_wrist"][1]) > 1e-6


def test_perturbing_knee_moves_segment_and_exported_markers() -> None:
    rec = _pose_rec("normal_gait")
    src = rec.snapshot_at(30)
    assert src is not None
    knee = src.joints["left_knee"]
    moved = Vec3(knee.position.x + 0.08, knee.position.y + 0.05, knee.position.z - 0.03)
    new_joints = dict(src.joints)
    new_joints["left_knee"] = JointSample(
        joint_id="left_knee",
        position=moved,
        parent_id=knee.parent_id,
        angle_deg=None,
        velocity=knee.velocity,
        velocity_vector=knee.velocity_vector,
    )
    new_snap = deepcopy(src)
    new_snap.joints = new_joints
    new_snap.dofs = {
        k: type(v)(dof_id=v.dof_id, angle_deg=None, velocity_deg_s=v.velocity_deg_s, joint_id=v.joint_id)
        for k, v in src.dofs.items()
    }

    before = _build_display_coords(src, DEFAULT_SKELETON_DISPLAY_MODE)
    after = _build_display_coords(new_snap, DEFAULT_SKELETON_DISPLAY_MODE)
    assert before["left_knee"] != after["left_knee"]
    hip_b, kn_b, an_b = before["left_hip"], before["left_knee"], before["left_ankle"]
    hip_a, kn_a, an_a = after["left_hip"], after["left_knee"], after["left_ankle"]
    # Distal and proximal sticks both change because the shared vertex moved.
    assert (hip_b, kn_b) != (hip_a, kn_a)
    assert (kn_b, an_b) != (kn_a, an_a)

    rec_before = GaitMotionRecording(
        source="audit-before", fps=rec.fps, snapshots=[src], source_kind="pose_estimation"
    )
    rec_after = GaitMotionRecording(
        source="audit-after", fps=rec.fps, snapshots=[new_snap], source_kind="pose_estimation"
    )
    exp_b = export_to_opensim_format(rec_before, include_markers=True)
    exp_a = export_to_opensim_format(rec_after, include_markers=True)
    assert exp_b.marker_table is not None and exp_a.marker_table is not None
    mk = "LKNE"
    assert mk in exp_b.marker_table.positions
    assert exp_b.marker_table.positions[mk][0] != exp_a.marker_table.positions[mk][0]
    assert exp_a.marker_table.positions[mk][0] == (moved.x, moved.y, moved.z)

    # Geometry at the knee (hip–knee–ankle) changes; stored JSON DOF may stay None here.
    from stablewalk.biomechanics.service import _angle_between_segments

    ang_b = _angle_between_segments(
        src.joints["left_hip"].position,
        src.joints["left_knee"].position,
        src.joints["left_ankle"].position,
    )
    ang_a = _angle_between_segments(
        new_snap.joints["left_hip"].position,
        new_snap.joints["left_knee"].position,
        new_snap.joints["left_ankle"].position,
    )
    assert ang_b is not None and ang_a is not None
    assert abs(ang_a - ang_b) > 0.5


@pytest.mark.parametrize("stem", ("normal_gait", "abnormal_gait"))
def test_real_demo_frames_are_bilateral_and_connected(stem: str) -> None:
    rec = _pose_rec(stem)
    last = rec.frame_count - 1
    for idx in (0, last // 4, last // 2, (3 * last) // 4, last):
        snap = rec.snapshot_at(idx)
        assert snap is not None
        live = live_snapshot_counts(snap)
        assert live["snapshot_left_required"] == live["snapshot_left_required_total"]
        assert live["snapshot_right_required"] == live["snapshot_right_required_total"]
        chains = limbs_connected(snap)
        assert all(chains.values()), (stem, idx, chains)
        display = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
        for parent, child in WIRE_DISPLAY_STICK_SEGMENTS:
            assert parent in display and child in display
            assert display[parent] != display[child]


def test_inspector_fields_show_separate_counts() -> None:
    rec = _pose_rec("normal_gait")
    snap = rec.snapshot_at(0)
    fields = build_skeleton_model_inspector_fields(snapshot=snap, recording=rec)
    assert fields["independent_sagittal_dofs"] == "13"
    assert fields["detected_pose_landmarks"] == "33"
    assert fields["opensim_gait2392_coordinates"] == "23"
    assert int(fields["live_joints"]) >= 19
    assert "complete" in fields["bilateral"]


def test_video_driven_set_excludes_interpolated_and_aliases() -> None:
    assert "spine" not in WIRE_VIDEO_DRIVEN_JOINTS
    assert "neck" not in WIRE_VIDEO_DRIVEN_JOINTS
    assert "left_foot" not in WIRE_VIDEO_DRIVEN_JOINTS
    assert "right_foot" not in WIRE_VIDEO_DRIVEN_JOINTS
    assert "left_knee" in WIRE_VIDEO_DRIVEN_JOINTS
    assert len(DOF_IDS) == 13
