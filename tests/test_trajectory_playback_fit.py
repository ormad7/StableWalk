"""Tests for 3D trajectory autofit + playback tip/path updates."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
from stablewalk.ui.viewers.dof_trajectory_3d import (
    _TRAJ_AUTOFIT_PAD_FRAC,
    _OVERVIEW_ABS_MIN_SPAN_M,
    _get_cached_stable_viewport,
    _viewport_for_overview_dock,
    _viewport_for_single_dof_trajectory,
    clear_trajectory_axis_cache,
    fit_trajectory_viewport,
    update_trajectory_current_tip,
)


def test_autofit_pad_is_within_10_to_15_percent() -> None:
    assert 0.10 <= _TRAJ_AUTOFIT_PAD_FRAC <= 0.15


def test_single_dof_viewport_equal_scale_and_padding() -> None:
    xs = [0.0, 0.04]
    ys = [0.80, 0.84]
    zs = [0.0, 0.02]
    vp = _viewport_for_single_dof_trajectory(xs, ys, zs)
    sx = vp.xlim[1] - vp.xlim[0]
    sy = vp.ylim[1] - vp.ylim[0]
    sz = vp.zlim[1] - vp.zlim[0]
    assert abs(sx - sy) < 1e-9
    assert abs(sy - sz) < 1e-9
    # Data range on X is 0.04; padded span must exceed raw span.
    assert sx > 0.04
    # Equal cube aspect.
    assert abs(vp.box_aspect[0] - vp.box_aspect[1]) < 1e-9


def test_tiny_trajectory_enforces_minimum_span_without_changing_data() -> None:
    xs = [0.10, 0.1005]
    ys = [0.80, 0.8004]
    zs = [0.20, 0.2003]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_knee", motion_dock=True)
    side = vp.xlim[1] - vp.xlim[0]
    assert side >= _OVERVIEW_ABS_MIN_SPAN_M - 1e-9
    # Equal physical scale.
    assert abs((vp.ylim[1] - vp.ylim[0]) - side) < 1e-9
    assert abs((vp.zlim[1] - vp.zlim[0]) - side) < 1e-9


def test_overview_autofit_preserves_equal_scale_and_full_xyz_extent() -> None:
    xs = [-0.03, -0.01, 0.08, 0.16]
    ys = [0.80, 0.805, 0.81, 0.815]
    zs = [-0.12, -0.04, 0.03, 0.07]
    original = (list(xs), list(ys), list(zs))

    vp = _viewport_for_overview_dock(
        xs,
        ys,
        zs,
        joint_id="right_knee",
        fill_panel=True,
    )

    spans = (
        vp.xlim[1] - vp.xlim[0],
        vp.ylim[1] - vp.ylim[0],
        vp.zlim[1] - vp.zlim[0],
    )
    assert max(spans) - min(spans) < 1e-9
    assert vp.box_aspect[0] == vp.box_aspect[1] == vp.box_aspect[2]
    for values, limits in zip((xs, ys, zs), (vp.xlim, vp.ylim, vp.zlim), strict=True):
        assert limits[0] < min(values)
        assert limits[1] > max(values)
    assert (xs, ys, zs) == original


def test_playback_tip_updates_path_without_touching_limits() -> None:
    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")
    ax._stablewalk_motion_dock = True
    ax._stablewalk_overview_cm_ticks = True
    # Pretend a full draw already cached the static path + locked limits.
    full_xs = [0.0, 0.01, 0.02, 0.03, 0.04]
    full_ys = [0.80, 0.81, 0.82, 0.81, 0.80]
    full_zs = [0.0, 0.01, 0.02, 0.03, 0.04]
    ax._stablewalk_full_path_xyz = (
        ("key",),
        full_xs,
        full_ys,
        full_zs,
        [Vec3(x, y, z) for x, y, z in zip(full_xs, full_ys, full_zs)],
    )
    ax._stablewalk_limits_locked = True
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(0.7, 0.9)
    ax.set_zlim(-0.1, 0.1)
    locked = (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())

    # Seed current marker artists.
    tip = Vec3(0.02, 0.82, 0.02)
    from stablewalk.ui.viewers.dof_trajectory_3d import _ensure_current_marker

    _ensure_current_marker(ax, tip, marker_size=40)

    snap = SkeletonSnapshot(
        frame_index=2,
        time_s=0.08,
        joints={
            "right_knee": JointSample(
                joint_id="right_knee",
                position=tip,
                angle_deg=20.0,
                velocity=None,
            )
        },
    )
    recording = SimpleNamespace(frame_count=5, fps=25.0)
    ok = update_trajectory_current_tip(
        ax,
        None,
        tip_snapshot=snap,
        joint_id="right_knee",
        recording=recording,
        list_pos=2,
        update_path=True,
    )
    assert ok is True
    assert (ax.get_xlim(), ax.get_ylim(), ax.get_zlim()) == locked
    state = ax._stablewalk_traj_artists
    assert state.path_line is not None
    xs = list(state.path_line.get_xdata())
    assert 2 <= len(xs) <= 5


def test_clear_trajectory_axis_cache_drops_lock() -> None:
    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")
    ax._stablewalk_limits_locked = True
    ax._stablewalk_stable_viewport = ("k", object())
    clear_trajectory_axis_cache(ax)
    assert not hasattr(ax, "_stablewalk_limits_locked")
    assert not hasattr(ax, "_stablewalk_stable_viewport")


def test_fit_trajectory_viewport_keeps_orbit() -> None:
    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")
    ax._stablewalk_limits_locked = True
    ax._stablewalk_stable_viewport = ("k", object())
    ax._stablewalk_user_camera = (22.0, -48.0)
    ax._stablewalk_pan_offset = (0.1, 0.0, 0.0)
    fit_trajectory_viewport(ax, keep_camera=True)
    assert not hasattr(ax, "_stablewalk_limits_locked")
    assert not hasattr(ax, "_stablewalk_stable_viewport")
    assert ax._stablewalk_user_camera == (22.0, -48.0)
    assert ax._stablewalk_pan_offset == (0.1, 0.0, 0.0)
