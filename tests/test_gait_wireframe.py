"""16-joint gait wireframe: store, normalize, smoothing reuse, overlay + 3D."""

from __future__ import annotations

import math

from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait
from stablewalk.models.gait_wireframe import (
    FULL_SKELETON_LABEL,
    GAIT16_SKELETON_LABEL,
    GAIT_WIRE_BONES,
    GAIT_WIRE_DRAW_JOINTS,
    GAIT_WIRE_JOINT_COUNT,
    GAIT_WIRE_JOINT_IDS,
    GAIT_WIRE_JOINT_NAMES,
    GaitWireJoint,
    build_gait_wireframe,
    canonical_gait_wire_model,
    extract_gait_wire_positions,
    gait_wire_included_joint_lines,
    overlay_keypoints_for_gait_wire,
    project_snapshot_to_gait_wire,
    session_body_scale,
)
from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
from stablewalk.models.joint_registry import JOINT_IDS, ROOT_JOINT_ID
from stablewalk.models.pose_data import Keypoint
from stablewalk.ui.skeleton_display_filter import SkeletonDisplayFilter
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    SKELETON_DISPLAY_MODE_OPTIONS,
    SKELETON_MODE_LABELS,
    gait16_visible_joints_and_bones,
)


def _kp(name: str, x: float, y: float, vis: float = 1.0) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, z=0.0, visibility=vis)


def _gait16_overlay_keypoints() -> list[Keypoint]:
    return [
        _kp("left_shoulder", 0.40, 0.28),
        _kp("right_shoulder", 0.60, 0.28),
        _kp("left_elbow", 0.36, 0.40),
        _kp("right_elbow", 0.64, 0.40),
        _kp("left_wrist", 0.34, 0.50),
        _kp("right_wrist", 0.66, 0.50),
        _kp("left_hip", 0.44, 0.55),
        _kp("right_hip", 0.56, 0.55),
        _kp("left_knee", 0.43, 0.72),
        _kp("right_knee", 0.57, 0.72),
        _kp("left_ankle", 0.42, 0.88),
        _kp("right_ankle", 0.58, 0.88),
        _kp("left_heel", 0.41, 0.92),
        _kp("right_heel", 0.59, 0.92),
        _kp("nose", 0.50, 0.12),
        _kp("left_foot_index", 0.42, 0.96),
        _kp("right_foot_index", 0.58, 0.96),
    ]


def test_canonical_gait_wire_model_has_stable_ids_and_names() -> None:
    model = canonical_gait_wire_model()
    assert model["joint_count"] == 16
    assert model["joint_ids"] == list(GAIT_WIRE_JOINT_IDS)
    assert len(model["joints"]) == 16
    ids = [row["id"] for row in model["joints"]]
    names = [row["name"] for row in model["joints"]]
    assert ids == list(GAIT_WIRE_JOINT_IDS)
    assert len(set(ids)) == 16
    assert len(names) == 16
    assert all(name.strip() for name in names)
    assert GAIT_WIRE_JOINT_NAMES["left_knee"] == "Left knee"
    assert GAIT_WIRE_JOINT_NAMES["pelvis_center"] == "Pelvis center"
    lines = gait_wire_included_joint_lines()
    assert len(lines) == 16
    assert any("left_knee" in line and "Left knee" in line for line in lines)


def test_gait_wire_has_exactly_16_joints() -> None:
    assert GAIT_WIRE_JOINT_COUNT == 16
    assert len(GAIT_WIRE_JOINT_IDS) == 16
    assert len(GAIT_WIRE_DRAW_JOINTS) == 16
    required = {
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "shoulder_center",
        "pelvis_center",
    }
    assert set(GAIT_WIRE_JOINT_IDS) == required
    assert len(GAIT_WIRE_BONES) >= 15


def test_full_skeleton_registry_is_unchanged() -> None:
    assert len(JOINT_IDS) == 21
    assert ROOT_JOINT_ID == "pelvis"
    assert "left_toe" in JOINT_IDS
    assert "head" in JOINT_IDS


def test_existing_skeleton_modes_are_preserved() -> None:
    keys = [k for k, _ in SKELETON_DISPLAY_MODE_OPTIONS]
    labels = list(SKELETON_MODE_LABELS)
    for key in ("gait", "2d_pose", "3d_normalized", "side", "biomechanical"):
        assert key in keys
    assert "Walking Skeleton" in labels
    assert "2D Pose Skeleton" in labels
    assert "3D Normalized Skeleton" in labels
    assert "Side Skeleton" in labels
    assert "Biomechanical Skeleton" in labels
    assert ("full", FULL_SKELETON_LABEL) in SKELETON_DISPLAY_MODE_OPTIONS
    assert ("gait_16", GAIT16_SKELETON_LABEL) in SKELETON_DISPLAY_MODE_OPTIONS


def test_store_has_time_frame_xyz_visibility_confidence() -> None:
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=0.5, cadence_hz=1.0))
    wire = build_gait_wireframe(recording)
    assert len(wire.frames) == len(recording.snapshots)
    assert wire.body_scale > 1e-5
    frame = wire.frames[0]
    assert frame.frame_index == recording.snapshots[0].frame_index
    assert frame.time_s == recording.snapshots[0].time_s
    assert len(frame.joints) == 16
    for jid in GAIT_WIRE_JOINT_IDS:
        joint = frame.joints[jid]
        assert isinstance(joint, GaitWireJoint)
        assert joint.joint_id == jid
        assert joint.frame_index == frame.frame_index
        assert joint.time_s == frame.time_s
        assert math.isfinite(joint.x)
        assert math.isfinite(joint.y)
        assert math.isfinite(joint.z)
        assert 0.0 <= joint.visibility <= 1.0
        assert 0.0 <= joint.confidence <= 1.0
    pelvis = frame.joints["pelvis_center"]
    assert abs(pelvis.x) < 1e-9
    assert abs(pelvis.y) < 1e-9
    assert abs(pelvis.z) < 1e-9


def test_normalize_is_pelvis_centered_and_body_scaled() -> None:
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=0.4, cadence_hz=1.0))
    scale = session_body_scale(recording)
    raw = recording.snapshots[2]
    projected = project_snapshot_to_gait_wire(raw, body_scale=scale)
    pelvis = projected.joints[ROOT_JOINT_ID].position
    assert abs(pelvis.x) < 1e-9
    assert abs(pelvis.y) < 1e-9
    assert abs(pelvis.z) < 1e-9
    lh = projected.joints["left_hip"].position
    la = projected.joints["left_ankle"].position
    leg = math.sqrt((lh.x - la.x) ** 2 + (lh.y - la.y) ** 2 + (lh.z - la.z) ** 2)
    assert 0.7 <= leg <= 1.3
    # Source full skeleton is not mutated.
    assert ROOT_JOINT_ID in raw.joints
    assert "head" in raw.joints
    assert "left_toe" in raw.joints
    assert len(projected.joints) == 16


def test_smoothing_runs_on_full_skeleton_before_projection() -> None:
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    filt = SkeletonDisplayFilter()
    scale = session_body_scale(recording)
    for snap in recording.snapshots[:8]:
        filtered = filt.filter_snapshot(snap)
        assert "head" in filtered.joints
        assert "left_toe" in filtered.joints
        assert filtered.metadata.get("display_filtered") is True
        wire = project_snapshot_to_gait_wire(filtered, body_scale=scale)
        joints, bones = gait16_visible_joints_and_bones(wire)
        assert len(joints) == 16
        assert len(bones) == len(GAIT_WIRE_BONES)
        assert snap.metadata.get("display_filtered") is not True


def test_overlay_exposes_all_16_joints_and_connections() -> None:
    from stablewalk.pose.video_overlay_compose import (
        VideoOverlayLayers,
        compose_video_skeleton_overlay,
        gait_wire_overlay_drawable,
    )
    import numpy as np

    kps = _gait16_overlay_keypoints()
    named = overlay_keypoints_for_gait_wire(kps)
    assert set(named) >= set(GAIT_WIRE_JOINT_IDS)
    joints, edges = gait_wire_overlay_drawable(kps)
    assert set(joints) == set(GAIT_WIRE_JOINT_IDS)
    assert len(edges) == len(GAIT_WIRE_BONES)
    base = np.full((160, 120, 3), 40, dtype=np.uint8)
    out = compose_video_skeleton_overlay(
        base,
        kps,
        layers=VideoOverlayLayers(
            skeleton=True,
            joint_labels=True,
            com=False,
            ground_reaction=False,
            joint_trajectory=False,
            ground_plane=False,
        ),
        opacity=1.0,
        skeleton_model="gait_16",
    )
    assert out.shape == base.shape
    assert not np.array_equal(out, base)
    full = compose_video_skeleton_overlay(
        base,
        kps,
        layers=VideoOverlayLayers(
            skeleton=True,
            joint_labels=False,
            com=False,
            ground_reaction=False,
            joint_trajectory=False,
            ground_plane=False,
        ),
        opacity=1.0,
        skeleton_model="full",
    )
    assert not np.array_equal(full, out)


def test_3d_view_draws_all_16_joints_and_bones() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.gait_skeleton_renderer import (
        SKELETON_DISPLAY_MODE_GAIT16,
        draw_gait_skeleton,
        setup_skeleton_axes,
    )

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=0.4, cadence_hz=1.0))
    filt = SkeletonDisplayFilter()
    raw = recording.snapshots[2]
    filtered = filt.filter_snapshot(raw)
    scale = session_body_scale(recording)
    wire_snap = project_snapshot_to_gait_wire(filtered, body_scale=scale)
    joints, bones = gait16_visible_joints_and_bones(wire_snap)
    assert len(joints) == 16
    assert set(joints) == set(GAIT_WIRE_DRAW_JOINTS)
    assert len(bones) == len(GAIT_WIRE_BONES)

    fig = Figure(figsize=(5.0, 4.0), dpi=80)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax, display_mode=SKELETON_DISPLAY_MODE_GAIT16)
    draw_gait_skeleton(
        ax,
        wire_snap,
        clear=True,
        display_mode=SKELETON_DISPLAY_MODE_GAIT16,
        show_com=False,
        show_support_polygon=False,
        show_gait_direction=False,
        show_contact_points=False,
        show_ground_plane=False,
        lite_draw=False,
    )
    canvas.draw()
    pool = getattr(ax, "_sw_skel_pool", {}) or {}
    used = getattr(ax, "_sw_skel_used", set()) or set()
    for jid in GAIT_WIRE_DRAW_JOINTS:
        fill_key = f"joint/{jid}/fill"
        assert fill_key in pool, f"missing 3D joint marker {jid}"
        assert fill_key in used
    for parent, child in bones:
        outline = f"bone/{parent}>{child}/outline"
        assert outline in pool, f"missing 3D bone {parent}->{child}"
        assert outline in used


def test_missing_pelvis_is_omitted_not_zero_filled() -> None:
    snap = SkeletonSnapshot(
        frame_index=0,
        time_s=0.0,
        joints={
            "left_shoulder": JointSample("left_shoulder", Vec3(0.40, 1.40, 0.10)),
            "right_shoulder": JointSample("right_shoulder", Vec3(0.60, 1.40, 0.10)),
            "left_knee": JointSample("left_knee", Vec3(0.42, 0.70, 0.05)),
        },
    )
    raw = extract_gait_wire_positions(snap)
    assert "pelvis_center" not in raw
    projected = project_snapshot_to_gait_wire(snap, body_scale=1.0)
    assert ROOT_JOINT_ID not in projected.joints
    assert "left_knee" in projected.joints
    knee = projected.joints["left_knee"].position
    assert abs(knee.x - 0.42) < 1e-9
    assert abs(knee.y - 0.70) < 1e-9
    assert abs(knee.z - 0.05) < 1e-9


def test_low_confidence_landmarks_are_omitted() -> None:
    snap = SkeletonSnapshot(
        frame_index=0,
        time_s=0.0,
        joints={
            "left_hip": JointSample("left_hip", Vec3(-0.10, 1.00, 0.0)),
            "right_hip": JointSample("right_hip", Vec3(0.10, 1.00, 0.0)),
            "left_knee": JointSample("left_knee", Vec3(-0.10, 0.60, 0.0)),
            "right_knee": JointSample("right_knee", Vec3(0.10, 0.60, 0.0)),
        },
        metadata={
            "landmark_visibility": {
                "left_hip": 0.9,
                "right_hip": 0.9,
                "left_knee": 0.12,
                "right_knee": 0.95,
            }
        },
    )
    raw = extract_gait_wire_positions(snap)
    assert "pelvis_center" in raw
    assert "left_knee" not in raw
    assert "right_knee" in raw
    overlay = overlay_keypoints_for_gait_wire(
        [
            _kp("left_hip", 0.44, 0.55, vis=0.9),
            _kp("right_hip", 0.56, 0.55, vis=0.9),
            _kp("left_knee", 0.43, 0.72, vis=0.1),
            _kp("right_knee", 0.57, 0.72, vis=0.95),
        ]
    )
    assert "pelvis_center" in overlay
    assert "left_knee" not in overlay
    assert "right_knee" in overlay
