"""Requirement 1: joints, landmarks, segments, and DOFs are different counts."""

from __future__ import annotations

from stablewalk.adapters.pose_adapter import pose_frame_to_snapshot
from stablewalk.models.joint_registry import (
    DOF_IDS,
    DOF_JOINT,
    GAIT2392_SIMBODY_COORDINATE_COUNT,
    GAIT2392_SIMBODY_COORDINATE_NAMES,
    JOINT_IDS,
    JOINT_PARENTS,
    MEDIAPIPE_POSE_LANDMARK_COUNT,
    OPENSIM_COORDINATE_ALIASES,
    ROOT_JOINT_ID,
    WIRE_BILATERAL_REQUIRED,
    WIRE_CONNECTED_SEGMENT_COUNT,
    WIRE_DERIVED_CONTACT_JOINTS,
    WIRE_INDEPENDENT_DOF_COUNT,
    WIRE_LOWER_LIMB_JOINTS,
    WIRE_MODEL_KIND,
    WIRE_RECONSTRUCTED_JOINT_COUNT,
    WIRE_SAGITTAL_DOF_COUNT,
    WIRE_TRACKABLE_JOINT_COUNT,
    WIRE_UNIQUE_ANATOMICAL_JOINT_COUNT,
    WIRE_UNIQUE_ANATOMICAL_JOINTS,
    WIRE_UNIQUE_CONNECTED_SEGMENT_COUNT,
    WIRE_UNIQUE_SKELETON_BONES,
    wire_model_count_summary,
    wire_model_inventory_rows,
)
from stablewalk.models.pose_data import Keypoint, PoseFrame
from stablewalk.pose.dof import GAIT_ANGLE_FIELDS
from stablewalk.pose.skeleton_3d import (
    FOOT_3D_LANDMARKS,
    SKELETON_3D_CONNECTIONS,
    SKELETON_3D_JOINTS,
    reconstruct_skeleton_3d,
)
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    _build_display_xy,
    _is_foot_structure_edge,
    _is_torso_structure_edge,
)


def test_wire_model_counts_joints_separately_from_dofs() -> None:
    assert WIRE_TRACKABLE_JOINT_COUNT == len(JOINT_IDS) == 21
    assert ROOT_JOINT_ID == "pelvis"
    assert ROOT_JOINT_ID not in JOINT_IDS
    assert WIRE_SAGITTAL_DOF_COUNT == len(DOF_IDS) == 13
    assert WIRE_INDEPENDENT_DOF_COUNT == 13
    assert WIRE_TRACKABLE_JOINT_COUNT != WIRE_SAGITTAL_DOF_COUNT
    assert WIRE_MODEL_KIND == "monocular_wire_skeleton"
    for jid in WIRE_LOWER_LIMB_JOINTS:
        assert jid in JOINT_IDS
        assert JOINT_PARENTS[jid] is not None or jid == ROOT_JOINT_ID
    assert len(GAIT_ANGLE_FIELDS) >= 14
    assert "left_knee" in GAIT_ANGLE_FIELDS
    assert "pelvis_rotation" in GAIT_ANGLE_FIELDS


def test_mediapipe_landmark_enum_is_33() -> None:
    from stablewalk.pose.estimation import LANDMARK_NAMES

    assert len(LANDMARK_NAMES) == MEDIAPIPE_POSE_LANDMARK_COUNT == 33
    assert "left_heel" in LANDMARK_NAMES
    assert "left_foot_index" in LANDMARK_NAMES


def test_inventories_are_not_interchangeable() -> None:
    counts = wire_model_count_summary()
    assert counts["detected_pose_landmarks"] == MEDIAPIPE_POSE_LANDMARK_COUNT == 33
    assert counts["reconstructed_3d_joints"] == WIRE_RECONSTRUCTED_JOINT_COUNT == 15
    assert len(SKELETON_3D_JOINTS) == 15
    assert len(SKELETON_3D_CONNECTIONS) == 14
    assert counts["unique_anatomical_joints"] == WIRE_UNIQUE_ANATOMICAL_JOINT_COUNT == 19
    assert counts["unique_connected_segments"] == WIRE_UNIQUE_CONNECTED_SEGMENT_COUNT == 19
    assert counts["connected_skeleton_bones"] == WIRE_CONNECTED_SEGMENT_COUNT == 21
    assert counts["independent_sagittal_dofs"] == 13
    assert counts["opensim_gait2392_coordinates"] == GAIT2392_SIMBODY_COORDINATE_COUNT == 23
    assert len(GAIT2392_SIMBODY_COORDINATE_NAMES) == 23
    assert counts["mapped_opensim_coordinate_aliases"] == len(OPENSIM_COORDINATE_ALIASES)
    # Landmarks, reconstructed joints, unique segments, and DOFs are four lists.
    assert len({
        counts["detected_pose_landmarks"],
        counts["reconstructed_3d_joints"],
        counts["unique_connected_segments"],
        counts["independent_sagittal_dofs"],
        counts["opensim_gait2392_coordinates"],
    }) == 5


def test_derived_midfoot_is_not_an_independent_dof() -> None:
    assert WIRE_DERIVED_CONTACT_JOINTS == ("left_foot", "right_foot")
    for jid in WIRE_DERIVED_CONTACT_JOINTS:
        assert jid in JOINT_IDS
        assert jid not in WIRE_UNIQUE_ANATOMICAL_JOINTS
        assert jid not in DOF_JOINT.values()
    assert WIRE_UNIQUE_ANATOMICAL_JOINT_COUNT == 19
    assert 12 <= WIRE_INDEPENDENT_DOF_COUNT <= 20
    assert 12 <= WIRE_UNIQUE_ANATOMICAL_JOINT_COUNT <= 20
    assert 12 <= WIRE_UNIQUE_CONNECTED_SEGMENT_COUNT <= 20
    assert 12 <= WIRE_RECONSTRUCTED_JOINT_COUNT <= 20


def test_supervisor_12_20_uses_dofs_or_unique_components_not_landmarks() -> None:
    """33 landmarks and 21 named nodes are not the 12–20 metric."""
    assert MEDIAPIPE_POSE_LANDMARK_COUNT > 20
    assert WIRE_TRACKABLE_JOINT_COUNT > 20
    assert 12 <= WIRE_INDEPENDENT_DOF_COUNT <= 20
    assert 12 <= WIRE_UNIQUE_ANATOMICAL_JOINT_COUNT <= 20


def test_inventory_table_rows_are_complete() -> None:
    rows = wire_model_inventory_rows()
    keys = {"name", "parent", "child", "xyz", "dof", "anatomy", "confidence"}
    assert all(keys <= row.keys() for row in rows)
    names = {row["name"] for row in rows}
    assert ROOT_JOINT_ID in names
    assert "left_hip→left_knee" in names
    assert "right_hip→right_knee" in names
    assert "left_foot" in names
    children = {row["child"] for row in rows}
    for jid in WIRE_BILATERAL_REQUIRED:
        assert jid in children or jid == ROOT_JOINT_ID


def test_bilateral_anatomical_completeness() -> None:
    required = {
        "pelvis": ROOT_JOINT_ID,
        "torso": "spine",
        "head": "head",
        "neck": "neck",
        "left_hip": "left_hip",
        "right_hip": "right_hip",
        "left_knee": "left_knee",
        "right_knee": "right_knee",
        "left_ankle": "left_ankle",
        "right_ankle": "right_ankle",
        "left_shoulder": "left_shoulder",
        "right_shoulder": "right_shoulder",
        "left_elbow": "left_elbow",
        "right_elbow": "right_elbow",
        "left_wrist": "left_wrist",
        "right_wrist": "right_wrist",
        "left_foot_heel": "left_heel",
        "right_foot_heel": "right_heel",
        "left_foot_toe": "left_toe",
        "right_foot_toe": "right_toe",
    }
    for label, jid in required.items():
        assert jid in JOINT_IDS or jid == ROOT_JOINT_ID, label
        assert jid in JOINT_PARENTS, label


def test_reconstruction_keeps_15_column_core_and_adds_driven_feet() -> None:
    kps = _standing_keypoints()
    skel = reconstruct_skeleton_3d(kps)
    for name in SKELETON_3D_JOINTS:
        assert name in skel.joints
    for name in FOOT_3D_LANDMARKS:
        assert name in skel.joints
    assert len(SKELETON_3D_JOINTS) == 15


def test_midfoot_is_driven_by_heel_and_toe() -> None:
    snap = pose_frame_to_snapshot(_standing_pose_frame(), fps=30.0)
    for side in ("left", "right"):
        heel = snap.joints[f"{side}_heel"].position
        toe = snap.joints[f"{side}_toe"].position
        foot = snap.joints[f"{side}_foot"].position
        assert abs(foot.x - (heel.x + toe.x) / 2.0) < 1e-9
        assert abs(foot.y - (heel.y + toe.y) / 2.0) < 1e-9
        assert abs(foot.z - (heel.z + toe.z) / 2.0) < 1e-9
        ankle = snap.joints[f"{side}_ankle"].position
        assert heel.y <= ankle.y + 0.015
        assert toe.y <= ankle.y + 0.015


def test_image_frame_feet_are_not_mixed_into_canonical_xyz() -> None:
    frame = _standing_pose_frame()
    # Poison image-frame Y-down leftovers that previously sat above the ankle.
    frame.positions["left_heel"] = {"x": 0.4, "y": 0.99}
    frame.positions["left_toe"] = {"x": 0.42, "y": 0.99}
    snap = pose_frame_to_snapshot(frame, fps=30.0)
    ankle = snap.joints["left_ankle"].position
    heel = snap.joints["left_heel"].position
    assert heel.y <= ankle.y + 0.015


def test_displayed_limb_segments_are_driven_not_duplicate_torso_or_foot_sticks() -> None:
    assert _is_torso_structure_edge("left_hip", "left_shoulder")
    assert _is_torso_structure_edge(ROOT_JOINT_ID, "spine")
    assert _is_foot_structure_edge("left_ankle", "left_heel")
    assert _is_foot_structure_edge("left_heel", "left_toe")
    assert not _is_foot_structure_edge("left_hip", "left_knee")
    assert not _is_torso_structure_edge("left_hip", "left_knee")
    for parent, child in WIRE_UNIQUE_SKELETON_BONES:
        if child in ("left_heel", "right_heel", "left_toe", "right_toe"):
            continue
        if _is_torso_structure_edge(parent, child):
            continue
        assert {parent, child} <= set(JOINT_IDS) | {ROOT_JOINT_ID}


def test_walking_view_projects_every_snapshot_joint() -> None:
    snap = pose_frame_to_snapshot(_standing_pose_frame(), fps=30.0)
    display = _build_display_xy(snap)
    for jid in snap.joints:
        assert jid in display


def _standing_keypoints() -> list[Keypoint]:
    pts = {
        "nose": (0.50, 0.18, 0.02),
        "left_shoulder": (0.42, 0.32, 0.01),
        "right_shoulder": (0.58, 0.32, 0.01),
        "left_elbow": (0.38, 0.44, 0.03),
        "right_elbow": (0.62, 0.44, 0.03),
        "left_wrist": (0.36, 0.56, 0.04),
        "right_wrist": (0.64, 0.56, 0.04),
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


def _standing_pose_frame() -> PoseFrame:
    return PoseFrame(
        frame_index=0,
        image_path="synthetic",
        keypoints=_standing_keypoints(),
        detected=True,
    )
