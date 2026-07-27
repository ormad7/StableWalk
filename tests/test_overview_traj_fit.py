"""Overview 3D Joint Path — margins, box zoom, and column weights (no clipping)."""

from __future__ import annotations

import pytest

from stablewalk.ui.tk.dashboard_sections import (
    SEC1_TRAJ_PATH_WEIGHT,
    SEC1_TRAJ_PANEL_MINWIDTH,
    SEC1_TRAJ_SKELETON_WEIGHT,
    SEC1_TRAJ_VIDEO_WEIGHT,
)
from stablewalk.ui.viewers.dof_trajectory_3d import (
    _CURRENT_DOT_SIZE,
    _END_DOT_SIZE,
    _START_DOT_SIZE,
    _TRAJECTORY_TARGET_FILL,
    _balanced_box_aspect,
    _overview_camera_for_spans,
    _overview_dof_figure_margins,
    _overview_tick_values,
    _percentile_axis_limits,
    _path_segment_styles,
    _single_dof_figure_margins,
    _single_traj_box_zoom,
    setup_single_dof_trajectory_axes,
    _viewport_for_overview_dock,
)


def test_traj_column_weights_are_balanced() -> None:
    assert SEC1_TRAJ_VIDEO_WEIGHT == 34
    assert SEC1_TRAJ_SKELETON_WEIGHT == 36
    assert SEC1_TRAJ_PATH_WEIGHT == 30
    assert (
        SEC1_TRAJ_VIDEO_WEIGHT
        + SEC1_TRAJ_SKELETON_WEIGHT
        + SEC1_TRAJ_PATH_WEIGHT
        == 100
    )
    assert SEC1_TRAJ_PANEL_MINWIDTH >= 240


def test_overview_margins_are_not_flush() -> None:
    left, bottom, right, top = _overview_dof_figure_margins(3.0, 3.0, dpi=100.0)
    assert left > 0.0
    assert bottom > 0.05
    assert right < 1.0
    assert top < 1.0
    assert right - left > 0.7
    assert top - bottom > 0.7


def test_motion_axes_use_most_of_available_graph_area() -> None:
    left, bottom, right, top = _single_dof_figure_margins(
        6.5, 5.2, dpi=100.0
    )
    area_fraction = (right - left) * (top - bottom)
    assert 0.80 <= area_fraction <= 0.95


def test_overview_box_zoom_shrinks_on_short_panel() -> None:
    short = _single_traj_box_zoom(2.0, 2.0, dpi=100.0, overview_dock=True)
    tall = _single_traj_box_zoom(4.0, 5.0, dpi=100.0, overview_dock=True)
    assert short < 1.0
    assert tall <= 1.0
    assert short <= tall


def test_overview_viewport_pads_and_ignores_nan() -> None:
    xs = [0.0, 0.01, 0.02, float("nan"), 0.03]
    ys = [0.8, 0.81, 0.79, 0.82, 0.805]
    zs = [0.0, 0.01, 0.02, 0.015, 0.025]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_hip")
    assert vp.xlim[0] < min(v for v in xs if v == v)
    assert vp.xlim[1] > max(v for v in xs if v == v)
    span = vp.xlim[1] - vp.xlim[0]
    # ~10% pad on each side of a robust span → total span > raw span
    raw = max(v for v in xs if v == v) - min(v for v in xs if v == v)
    assert span >= raw
    # Box aspect follows data spans (equal units); mild floor keeps thin axes visible.
    bx, by, bz = vp.box_aspect
    longest = max(bx, by, bz)
    assert min(bx, by, bz) / longest >= 0.20


def test_percentile_limits_ignore_spikes() -> None:
    vals = [0.0] * 20 + [5.0]  # spike
    lo, hi = _percentile_axis_limits(
        vals, pad_frac=0.10, low_pct=0.02, high_pct=0.98
    )
    assert hi < 2.0


def test_percentile_limits_ignore_all_nonfinite_values() -> None:
    lo, hi = _percentile_axis_limits(
        [float("nan"), float("inf"), float("-inf"), 0.10, 0.11, 0.12],
        pad_frac=0.12,
        min_span=0.004,
    )
    assert lo < 0.10
    assert hi > 0.12


def test_compact_axes_have_adaptive_readable_ticks() -> None:
    from stablewalk.ui.viewers.dof_trajectory_3d import _format_overview_cm_tick

    ticks = _overview_tick_values(-0.012, 0.018, use_cm=True, target=2)
    assert 2 <= len(ticks) <= 3
    assert ticks == sorted(ticks)
    assert ticks == sorted(set(ticks))
    labels = [_format_overview_cm_tick(t, 0) for t in ticks]
    assert len(labels) == len(set(labels))
    # Nice spacing: adjacent steps are nearly equal.
    if len(ticks) >= 3:
        steps = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
        assert max(steps) / max(min(steps), 1e-12) < 2.5


def test_short_y_axis_avoids_stacked_cm_ticks() -> None:
    """Knee Y ≈ 2 cm must not render -43/-44/-45 stacked on one edge."""
    from stablewalk.ui.viewers.dof_trajectory_3d import _format_overview_cm_tick

    ticks = _overview_tick_values(-0.455, -0.435, use_cm=True, target=2)
    labels = [_format_overview_cm_tick(t, 0) for t in ticks]
    assert len(ticks) == 2
    assert len(labels) == len(set(labels))
    assert abs(ticks[-1] - ticks[0]) >= 0.015


def test_box_aspect_clamps_nearly_flat_axes() -> None:
    aspect = _balanced_box_aspect((0.001, 0.20, 0.004))
    assert min(aspect) / max(aspect) >= 0.25


def test_selected_trajectory_axes_use_scientific_cm_labels() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111, projection="3d")
    ax._stablewalk_motion_dock = True
    ax._stablewalk_overview_cm_ticks = True
    setup_single_dof_trajectory_axes(ax)
    assert ax.get_xlabel() == "X (cm)"
    assert ax.get_ylabel() == "Y (cm)"
    assert ax.get_zlabel() == "Z (cm)"


def test_adaptive_ticks_avoid_duplicate_labels() -> None:
    from stablewalk.ui.viewers.dof_trajectory_3d import _format_overview_cm_tick

    ticks = _overview_tick_values(-0.05, 0.05, use_cm=True, target=3)
    labels = [_format_overview_cm_tick(t, 0) for t in ticks]
    assert len(labels) == len(set(labels))
    assert len(ticks) >= 2


def test_overview_viewport_gives_room_around_tiny_rom() -> None:
    """Near-flat knee Y must expand so markers/ticks are not clipped/stacked."""
    xs = [i * 0.001 for i in range(40)]
    ys = [-0.445 + (i % 3) * 0.0004 for i in range(40)]
    zs = [i * 0.0008 for i in range(40)]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_knee")
    assert vp.ylim[1] - vp.ylim[0] >= 0.055
    # Path stays centred — not stuck in a corner of an empty cube.
    med_x = 0.5 * (min(xs) + max(xs))
    assert vp.xlim[0] < med_x < vp.xlim[1]
    assert (vp.xlim[1] - vp.xlim[0]) <= 0.40


def test_overview_toe_path_fills_cube_not_corner_speck() -> None:
    """Foot tip with a long noisy axis must still fill a tight centred cube."""
    xs = [0.06 + (i % 5) * 0.002 for i in range(50)]
    ys = [-0.95 + (i % 4) * 0.0015 for i in range(50)]
    zs = [0.07 + (i % 6) * 0.002 for i in range(50)]
    # Spike that previously emptied the cube (false forward progression).
    xs = xs + [0.90]
    zs = zs + [0.85]
    ys = ys + [-0.95]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="left_toe")
    side = max(
        vp.xlim[1] - vp.xlim[0],
        vp.ylim[1] - vp.ylim[0],
        vp.zlim[1] - vp.zlim[0],
    )
    assert side <= 0.42 + 1e-9
    # Real cluster (~6–8 cm) must sit near the cube centre.
    assert vp.xlim[0] < 0.065 < vp.xlim[1]
    assert vp.ylim[0] < -0.95 < vp.ylim[1]
    assert vp.zlim[0] < 0.075 < vp.zlim[1]
    # Equal scale cube.
    assert abs((vp.xlim[1] - vp.xlim[0]) - (vp.ylim[1] - vp.ylim[0])) < 1e-9


def test_overview_knee_keeps_tip_inside_cube() -> None:
    """Live tip near Y=-44 cm must stay inside the fitted cube (no floor clip)."""
    xs = [-0.06 + (i % 4) * 0.003 for i in range(40)]
    ys = [-0.445 + (i % 5) * 0.002 for i in range(40)]
    zs = [-0.02 + (i % 3) * 0.002 for i in range(40)]
    # Tip slightly outside the bulk (matches footer Y -44.3).
    xs.append(-0.066)
    ys.append(-0.443)
    zs.append(-0.020)
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_knee")
    tip_x, tip_y, tip_z = -0.066, -0.443, -0.020
    assert vp.xlim[0] <= tip_x <= vp.xlim[1]
    assert vp.ylim[0] <= tip_y <= vp.ylim[1]
    assert vp.zlim[0] <= tip_z <= vp.zlim[1]
    # Equal cube; tip has air (not flush against a face).
    assert (tip_y - vp.ylim[0]) > 0.01
    assert (vp.ylim[1] - tip_y) > 0.01


def test_trajectory_targets_seventy_percent_fill_with_equal_axis_scale() -> None:
    xs = [i / 1000.0 for i in range(100)]
    ys = [0.4 + (i % 8) / 1000.0 for i in range(100)]
    zs = [0.2 + i / 2000.0 for i in range(100)]
    vp = _viewport_for_overview_dock(xs, ys, zs)

    robust_x_span = xs[97] - xs[1]
    displayed_x_span = vp.xlim[1] - vp.xlim[0]
    fill = robust_x_span / displayed_x_span
    # Centred cube: path fills a comfortable fraction of each equal axis.
    assert 0.25 <= fill <= 0.85

    limit_spans = (
        vp.xlim[1] - vp.xlim[0],
        vp.ylim[1] - vp.ylim[0],
        vp.zlim[1] - vp.zlim[0],
    )
    assert max(limit_spans) - min(limit_spans) < 1e-9


def test_path_gradient_and_markers_emphasize_current_frame() -> None:
    colors, widths = _path_segment_styles(12, "#63d8ff")
    assert colors[0][:3] != colors[-1][:3]
    assert all(a[3] <= b[3] for a, b in zip(colors, colors[1:]))
    assert all(a <= b for a, b in zip(widths, widths[1:]))
    assert _CURRENT_DOT_SIZE > _START_DOT_SIZE > _END_DOT_SIZE


def test_expand_viewport_keeps_tip_inside_cube() -> None:
    from stablewalk.models.gait_motion import Vec3
    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _SingleTrajViewport,
        _expand_viewport_to_include,
    )

    vp = _SingleTrajViewport(
        xlim=(-0.02, 0.01),
        ylim=(-0.45, -0.43),
        zlim=(-0.06, 0.0),
        box_aspect=(0.03, 0.02, 0.06),
        elev=25.0,
        azim=-55.0,
    )
    tip = Vec3(-0.059, -0.445, -0.008)
    expanded = _expand_viewport_to_include(vp, [tip], joint_id="right_knee")
    assert expanded.xlim[0] <= tip.x <= expanded.xlim[1]
    assert expanded.ylim[0] <= tip.y <= expanded.ylim[1]
    assert expanded.zlim[0] <= tip.z <= expanded.zlim[1]


def test_expand_viewport_keeps_full_path_inside_cube() -> None:
    from stablewalk.models.gait_motion import Vec3
    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _SingleTrajViewport,
        _expand_viewport_to_include,
    )

    vp = _SingleTrajViewport(
        xlim=(-0.02, 0.01),
        ylim=(-0.45, -0.43),
        zlim=(-0.06, 0.0),
        box_aspect=(0.03, 0.02, 0.06),
        elev=22.0,
        azim=-48.0,
    )
    path = [
        Vec3(-0.02, -0.44, -0.05),
        Vec3(-0.055, -0.448, -0.02),
        Vec3(0.005, -0.435, -0.005),
    ]
    expanded = _expand_viewport_to_include(
        vp, path, joint_id="right_knee", pad_frac=0.28
    )
    for tip in path:
        assert expanded.xlim[0] <= tip.x <= expanded.xlim[1]
        assert expanded.ylim[0] <= tip.y <= expanded.ylim[1]
        assert expanded.zlim[0] <= tip.z <= expanded.zlim[1]


def test_percentile_limits_center_on_median() -> None:
    vals = [0.55] + [1.0 + (i % 3) * 0.01 for i in range(40)]
    lo, hi = _percentile_axis_limits(
        vals, pad_frac=0.12, low_pct=0.02, high_pct=0.98
    )
    med = 1.0
    rel = (med - lo) / (hi - lo)
    assert 0.35 <= rel <= 0.65


def test_overview_cm_ticks_match_negative_knee_height() -> None:
    """Footer Y ≈ −44 cm must sit inside Y-axis ticks (minus signs preserved)."""
    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _format_overview_cm_tick,
        _overview_tick_values,
        meters_to_display_cm,
    )

    xs = [-0.05 + i * 0.001 for i in range(25)]
    ys = [-0.45 + 0.004 * ((i % 5) - 2) for i in range(25)]
    zs = [-0.02 + i * 0.001 for i in range(25)]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_knee")
    tip_y_cm = meters_to_display_cm(ys[-1])
    assert vp.ylim[0] <= ys[-1] <= vp.ylim[1]
    assert -55.0 <= tip_y_cm <= -30.0
    ticks = _overview_tick_values(vp.ylim[0], vp.ylim[1], use_cm=True, target=3)
    labels = [_format_overview_cm_tick(t, 0) for t in ticks]
    assert any(lab.startswith("-") for lab in labels)
    assert all(float(lab) <= 0 for lab in labels if lab not in ("0",))


def test_overview_viewport_centers_path() -> None:
    xs = [0.0 + i * 0.001 for i in range(40)]
    ys = [0.006] + [0.010 + (i % 4) * 0.0002 for i in range(39)]
    zs = [0.0 + i * 0.0008 for i in range(40)]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_hip")
    med_y = 0.010
    rel = (med_y - vp.ylim[0]) / (vp.ylim[1] - vp.ylim[0])
    assert 0.35 <= rel <= 0.65


def test_overview_camera_keeps_y_up_for_planar_path() -> None:
    elev_flat, _ = _overview_camera_for_spans((0.02, 0.003, 0.015))
    elev_tall, _ = _overview_camera_for_spans((0.01, 0.04, 0.01))
    assert elev_flat <= 26.0
    assert elev_tall >= elev_flat
