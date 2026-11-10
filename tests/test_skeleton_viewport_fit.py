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
    # Legacy global min/max span width was ~1.10 on this drift; auto-fit stays near body width
    # plus a small sequence-global pad.
    assert half_x < 1.20
    assert half_y < 1.12
    # Tight (unpadded) halves are enclosed by the padded content halves.
    assert 0.0 < tight_half_x <= half_x
    assert 0.0 < tight_half_y <= half_y


def test_apply_view_limits_keeps_head_and_feet_inside() -> None:
    """Sequence-global fit must leave every frame's soles inside ylim."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=2.0, cadence_hz=1.0))
    fig = Figure(figsize=(4.0, 2.5), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    ax._sw_fixed_view_box = compute_skeleton_view_box(
        recording, DEFAULT_SKELETON_DISPLAY_MODE
    )
    indices = (
        0,
        len(recording.snapshots) // 4,
        len(recording.snapshots) // 2,
        (3 * len(recording.snapshots)) // 4,
        len(recording.snapshots) - 1,
    )
    for idx in indices:
        snap = recording.snapshots[idx]
        snap._sw_display_xy = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
        _apply_view_limits(ax, snap)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        points = _collect_body_view_points(snap)
        assert points, f"frame {idx} produced no body points"
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        assert xlim[0] < min(xs) and xlim[1] > max(xs)
        assert ylim[0] < min(ys) and ylim[1] > max(ys)


def test_apply_view_limits_skips_missing_foot_landmarks() -> None:
    from matplotlib.figure import Figure

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    snap = recording.snapshots[3]
    for jid in ("left_toe", "right_toe", "left_heel", "right_heel"):
        snap.joints.pop(jid, None)
    snap._sw_display_xy = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    _apply_view_limits(ax, snap)
    ylim = ax.get_ylim()
    points = _collect_body_view_points(snap)
    ys = [p[1] for p in points]
    assert ylim[0] < min(ys)
    assert ylim[1] > max(ys)


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
    # Target a large readable fill so the body is clear but head/feet remain uncropped.
    data_height_fill = body_h / (ylim[1] - ylim[0])
    assert 0.70 <= data_height_fill <= 0.96, (
        f"expected skeleton body fill ~70–96%, got {data_height_fill:.1%}"
    )
    assert bbox.height > 100
    assert 0.70 <= height_fill <= 0.96
    assert 0.76 <= _SKELETON_VIEW_FILL <= 0.93


def test_apply_view_limits_keeps_scale_stable_across_frames() -> None:
    """Sequence-global box must not zoom in or out from frame to frame."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=2.0, cadence_hz=1.0))
    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    ax._sw_fixed_view_box = compute_skeleton_view_box(
        recording, DEFAULT_SKELETON_DISPLAY_MODE
    )
    spans: list[tuple[float, float]] = []
    for snap in recording.snapshots:
        snap._sw_display_xy = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
        _apply_view_limits(ax, snap)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        spans.append((xlim[1] - xlim[0], ylim[1] - ylim[0]))
    assert spans
    span_x0, span_y0 = spans[0]
    for span_x, span_y in spans[1:]:
        assert abs(span_x - span_x0) < 1e-9
        assert abs(span_y - span_y0) < 1e-9


def test_apply_view_limits_centers_current_frame() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3

    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    shifted = recording.snapshots[-1]
    new_joints = {
        jid: JointSample(
            joint_id=jid,
            position=Vec3(
                sample.position.x + 0.12,
                sample.position.y,
                sample.position.z,
            ),
            parent_id=sample.parent_id,
            angle_deg=sample.angle_deg,
            velocity=sample.velocity,
            velocity_vector=sample.velocity_vector,
        )
        for jid, sample in shifted.joints.items()
    }
    shifted = SkeletonSnapshot(
        frame_index=shifted.frame_index,
        time_s=shifted.time_s,
        joints=new_joints,
        dofs=shifted.dofs,
        metadata=shifted.metadata,
    )
    coords = _build_display_coords(shifted, DEFAULT_SKELETON_DISPLAY_MODE)
    shifted._sw_display_xy = coords

    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    ax._sw_fixed_view_box = compute_skeleton_view_box(recording, DEFAULT_SKELETON_DISPLAY_MODE)
    _apply_view_limits(ax, shifted)
    xlim = ax.get_xlim()
    center_x = 0.5 * (xlim[0] + xlim[1])
    points = _collect_body_view_points(shifted)
    body_cx = 0.5 * (min(p[0] for p in points) + max(p[0] for p in points))
    assert abs(center_x - body_cx) < 0.08
    ys = [p[1] for p in points]
    ylim = ax.get_ylim()
    assert ylim[0] < min(ys)
    assert ylim[1] > max(ys)
    xs = [p[0] for p in points]
    assert xlim[0] < min(xs)
    assert xlim[1] > max(xs)


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


def test_polished_manikin_keeps_coordinates_and_visible_feet() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle

    from stablewalk.ui.viewers import gait_skeleton_renderer as gsr

    recording = generate_mock_gait(
        MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0)
    )
    snap = recording.snapshots[4]
    before = {
        joint_id: (sample.position.x, sample.position.y, sample.position.z)
        for joint_id, sample in snap.joints.items()
    }

    fig = Figure(figsize=(4, 6), dpi=100)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    draw_gait_skeleton(
        ax,
        snap,
        display_mode="gait",
        highlight_joints={"right_knee"},
    )

    pool = getattr(ax, "_sw_skel_pool", {})
    assert "foot/left/sole" in pool
    assert "foot/right/sole" in pool
    selected = [patch for patch in ax.patches if isinstance(patch, Circle)]
    height = gsr._robust_body_height(snap)
    assert any(circle.get_radius() >= height * 0.014 for circle in selected)
    assert gsr._LIMB_VOLUME_SCALE > 1.0
    assert gsr._JOINT_MARKER_SCALE > 1.0

    after = {
        joint_id: (sample.position.x, sample.position.y, sample.position.z)
        for joint_id, sample in snap.joints.items()
    }
    assert after == before
