"""Task 2: skeleton head-to-feet fit and 3D path viewport vs Coordinate Inspector."""

from __future__ import annotations

import pytest

from stablewalk import config
from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.ui.overview_coordinate_inspector import _fmt_cm, tip_xyz_cm_from_cache
from stablewalk.ui.viewers.dof_trajectory_3d import (
    _joint_path_with_times,
    _viewport_for_overview_dock,
    draw_single_dof_trajectory_3d,
    meters_to_display_cm,
)
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    DEFAULT_SKELETON_DISPLAY_MODE,
    _apply_view_limits,
    _build_display_coords,
    _collect_body_view_points,
    compute_skeleton_view_box,
    setup_skeleton_axes,
)


def _pose_rec(stem: str):
    path = config.POSES_DIR / f"{stem}_poses.json"
    if not path.is_file():
        pytest.skip(f"missing {path}")
    return pose_sequence_to_gait_motion(load_pose_sequence(path))


def _quartile_indices(n: int) -> tuple[int, ...]:
    last = max(n - 1, 0)
    return (0, last // 4, last // 2, (3 * last) // 4, last)


@pytest.mark.parametrize("stem", ("normal_gait", "abnormal_gait"))
def test_skeleton_fit_contains_full_body_on_demo_videos(stem: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rec = _pose_rec(stem)
    assert rec.frame_count >= 8
    fig = Figure(figsize=(4.0, 2.4), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    setup_skeleton_axes(ax)
    box = compute_skeleton_view_box(rec, DEFAULT_SKELETON_DISPLAY_MODE)
    assert box is not None
    ax._sw_fixed_view_box = box
    for idx in _quartile_indices(rec.frame_count):
        snap = rec.snapshot_at(idx)
        assert snap is not None
        snap._sw_display_xy = _build_display_coords(snap, DEFAULT_SKELETON_DISPLAY_MODE)
        _apply_view_limits(ax, snap)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        points = _collect_body_view_points(snap)
        assert len(points) >= 3
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        assert xlim[0] < min(xs) < max(xs) < xlim[1]
        assert ylim[0] < min(ys) < max(ys) < ylim[1]


@pytest.mark.parametrize("stem", ("normal_gait", "abnormal_gait"))
def test_trajectory_viewport_contains_full_knee_path(stem: str) -> None:
    rec = _pose_rec(stem)
    path = _joint_path_with_times(rec, "right_knee", rec.frame_count - 1)
    assert len(path) >= 8
    xs = [p.x for p, _t in path]
    ys = [p.y for p, _t in path]
    zs = [p.z for p, _t in path]
    vp = _viewport_for_overview_dock(
        xs, ys, zs, joint_id="right_knee", fill_panel=True
    )
    for x, y, z in zip(xs, ys, zs):
        assert vp.xlim[0] <= x <= vp.xlim[1]
        assert vp.ylim[0] <= y <= vp.ylim[1]
        assert vp.zlim[0] <= z <= vp.zlim[1]
    # Equal laboratory scale (do not exaggerate one axis).
    sx = vp.xlim[1] - vp.xlim[0]
    sy = vp.ylim[1] - vp.ylim[0]
    sz = vp.zlim[1] - vp.zlim[0]
    assert abs(sx - sy) < 1e-9
    assert abs(sy - sz) < 1e-9


@pytest.mark.parametrize("stem", ("normal_gait", "abnormal_gait"))
def test_path_tip_matches_inspector_cm_at_quartile_frames(stem: str) -> None:
    """Plotted point at frame N matches inspector XYZ at the 0.1 cm display quantum.

    Coordinates are the stature-scaled path used by Overview (not world-calibrated
    millimetres). Do not treat the cm values as metrically accurate without a
    calibration object.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rec = _pose_rec(stem)
    joint_id = "right_knee"
    fig = Figure(figsize=(4.2, 3.6), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax._stablewalk_overview_dock = True
    ax._stablewalk_overview_cm_ticks = True
    for idx in _quartile_indices(rec.frame_count):
        snap = rec.snapshot_at(idx)
        assert snap is not None
        ok, _status = draw_single_dof_trajectory_3d(
            ax,
            rec,
            joint_id,
            end_frame_float=float(idx),
            tip_snapshot=snap,
            clear=True,
        )
        assert ok
        tip_m = getattr(ax, "_stablewalk_tip_xyz_m", None)
        cached = tip_xyz_cm_from_cache(ax)
        assert tip_m is not None and cached is not None
        # Inspector reads the same cache the red marker publishes.
        for meters, cm in zip(tip_m, cached):
            assert abs(cm - meters_to_display_cm(meters)) < 1e-9
        for axis_cm in cached:
            label = _fmt_cm(axis_cm)
            parsed = float(label.replace(" cm", ""))
            assert abs(parsed - axis_cm) < 0.06  # 0.1 cm display rounding
        xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        x, y, z = tip_m
        assert xlim[0] <= x <= xlim[1]
        assert ylim[0] <= y <= ylim[1]
        assert zlim[0] <= z <= zlim[1]
        full = getattr(ax, "_stablewalk_full_path_xyz", None)
        if isinstance(full, tuple) and len(full) >= 4:
            xs, ys, zs = full[1], full[2], full[3]
            assert xlim[0] <= min(xs) and max(xs) <= xlim[1]
            assert ylim[0] <= min(ys) and max(ys) <= ylim[1]
            assert zlim[0] <= min(zs) and max(zs) <= zlim[1]


def test_overview_box_zoom_never_exceeds_unity() -> None:
    from stablewalk.ui.viewers.dof_trajectory_3d import _single_traj_box_zoom

    for w, h in ((2.0, 2.0), (3.2, 2.5), (4.0, 5.0), (6.0, 3.0)):
        zoom = _single_traj_box_zoom(w, h, dpi=100.0, overview_dock=True)
        assert 0.80 <= zoom <= 1.0
