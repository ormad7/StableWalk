"""Skeleton panel auto-fit: body should fill ~70–80% of the viewport."""

from __future__ import annotations

import math

from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    DEFAULT_SKELETON_DISPLAY_MODE,
    _SKELETON_VIEW_FILL,
    _WALKING_VIEW_AZIMUTH_DEG,
    _anatomical_segment_profile,
    _apply_view_limits,
    _build_display_coords,
    _collect_body_view_points,
    compute_skeleton_view_box,
    draw_gait_skeleton,
    setup_skeleton_axes,
)



def test_compute_skeleton_view_box_handles_lateral_drift() -> None:
    from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=2.0, cadence_hz=1.0))
    for i, snap in enumerate(recording.snapshots):
        drift = 0.35 * (i / max(len(recording.snapshots) - 1, 1) - 0.5) * 2
        new_joints = {
            jid: JointSample(
                joint_id=jid,
                position=Vec3(
                    sample.position.x + drift,
                    sample.position.y,
                    sample.position.z,
                ),
                parent_id=sample.parent_id,
                angle_deg=sample.angle_deg,
                velocity=sample.velocity,
                velocity_vector=sample.velocity_vector,
            )
            for jid, sample in snap.joints.items()
        }
        recording.snapshots[i] = SkeletonSnapshot(
            frame_index=snap.frame_index,
            time_s=snap.time_s,
            joints=new_joints,
            dofs=snap.dofs,
            metadata=snap.metadata,
        )

    box = compute_skeleton_view_box(recording, DEFAULT_SKELETON_DISPLAY_MODE)
    assert box is not None
    _cx, _cy, half_x, half_y, tight_half_x, tight_half_y = box
    # Legacy global min/max span width was ~1.10 on this drift; auto-fit stays near body width.
    assert half_x < 0.98
    assert half_y < 0.95
    # Tight (unpadded) halves are enclosed by the padded content halves.
    assert 0.0 < tight_half_x <= half_x
    assert 0.0 < tight_half_y <= half_y


def test_apply_view_limits_targets_large_readable_fill() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=2.0, cadence_hz=1.0))
    snap = recording.snapshots[0]
    coords = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
    snap._sw_display_xy = coords
    points = _collect_body_view_points(snap)
    ys = [p[1] for p in points]
    body_h = max(ys) - min(ys)

    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    ax._sw_fixed_view_box = compute_skeleton_view_box(recording, DEFAULT_SKELETON_DISPLAY_MODE)

    _apply_view_limits(ax, snap)
    canvas.draw()
    bbox = ax.get_window_extent(canvas.get_renderer())
    ylim = ax.get_ylim()
    height_fill = (body_h / (ylim[1] - ylim[0])) * bbox.height / bbox.height
    # Target ~65–75% fill so the body is readable but head/feet remain uncropped.
    data_height_fill = body_h / (ylim[1] - ylim[0])
    assert 0.65 <= data_height_fill <= 0.75, (
        f"expected skeleton body fill ~65–75%, got {data_height_fill:.1%}"
    )
    assert bbox.height > 100
    assert 0.65 <= height_fill <= 0.75
    assert 0.72 <= _SKELETON_VIEW_FILL <= 0.78


def test_walking_view_projects_measured_joints_without_repositioning() -> None:
    """Walking view uses camera projection only; tracked joints stay authoritative."""
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    snap = recording.snapshots[0]
    ls0 = snap.joints["left_shoulder"].position
    rs0 = snap.joints["right_shoulder"].position
    lh0 = snap.joints["left_hip"].position
    rh0 = snap.joints["right_hip"].position

    coords = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
    angle = math.radians(_WALKING_VIEW_AZIMUTH_DEG)
    for joint_id in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
        pos = snap.joints[joint_id].position
        assert coords[joint_id][0] == math.cos(angle) * pos.x + math.sin(angle) * pos.z
        assert coords[joint_id][1] == pos.y
    # Analysis samples unchanged.
    assert snap.joints["left_shoulder"].position.x == ls0.x
    assert snap.joints["right_hip"].position.x == rh0.x
    assert snap.joints["left_hip"].position.x == lh0.x
    assert snap.joints["right_shoulder"].position.x == rs0.x


def test_anatomical_segments_taper_without_changing_joint_positions() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Polygon

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    snap = recording.snapshots[6]
    before = {
        joint_id: (sample.position.x, sample.position.y, sample.position.z)
        for joint_id, sample in snap.joints.items()
    }

    thigh = _anatomical_segment_profile("left_hip", "left_knee", 1.8)
    forearm = _anatomical_segment_profile("left_elbow", "left_wrist", 1.8)
    assert thigh is not None and thigh[0] > 0 and thigh[2] > 0 and thigh[1] >= thigh[0]
    assert forearm is not None and forearm[0] > forearm[2] > 0 and forearm[1] >= forearm[2]

    fig = Figure(figsize=(4, 6), dpi=100)
    ax = fig.add_subplot(111)
    draw_gait_skeleton(ax, snap, display_mode="gait")
    anatomical_patches = [patch for patch in ax.patches if isinstance(patch, Polygon)]
    assert len(anatomical_patches) >= 8

    after = {
        joint_id: (sample.position.x, sample.position.y, sample.position.z)
        for joint_id, sample in snap.joints.items()
    }
    assert after == before


def test_side_view_uses_measured_forward_and_vertical_coordinates() -> None:
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    snap = recording.snapshots[0]
    coords = _build_display_coords(snap, "side")
    for joint_id in ("left_shoulder", "right_hip", "left_knee", "right_ankle"):
        pos = snap.joints[joint_id].position
        assert coords[joint_id] == (pos.z, pos.y)
