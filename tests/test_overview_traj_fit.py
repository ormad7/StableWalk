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


def test_compact_axes_have_four_readable_ticks() -> None:
    ticks = _overview_tick_values(-0.012, 0.018, use_cm=True, target=4)
    assert len(ticks) == 4
    assert ticks == sorted(set(ticks))


def test_box_aspect_clamps_nearly_flat_axes() -> None:
    aspect = _balanced_box_aspect((0.001, 0.20, 0.004))
    assert min(aspect) / max(aspect) >= 0.30


def test_selected_trajectory_axes_use_scientific_cm_labels() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111, projection="3d")
    ax._stablewalk_motion_dock = True
    ax._stablewalk_overview_cm_ticks = True
    setup_single_dof_trajectory_axes(ax)
    assert ax.get_xlabel() == "X – Lateral (cm)"
    assert ax.get_ylabel() == "Y – Vertical (cm)"
    assert ax.get_zlabel() == "Z – Forward (cm)"


def test_percentile_limits_center_on_median() -> None:
    # Bulk near 1.0 with a one-sided low tail — median must sit mid-range.
    vals = [0.55] + [1.0 + (i % 3) * 0.01 for i in range(40)]
    lo, hi = _percentile_axis_limits(
        vals, pad_frac=0.12, low_pct=0.02, high_pct=0.98
    )
    med = 1.0
    rel = (med - lo) / (hi - lo)
    assert 0.35 <= rel <= 0.65


def test_overview_viewport_ignores_outlier_spike() -> None:
    xs = [0.0 + i * 0.002 for i in range(30)] + [3.0]
    ys = [0.01 + (i % 5) * 0.001 for i in range(30)] + [0.01]
    zs = [0.0 + i * 0.001 for i in range(30)] + [0.0]
    vp = _viewport_for_overview_dock(xs, ys, zs, joint_id="right_knee")
    assert vp.xlim[1] - vp.xlim[0] < 1.0


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
    # Planar walk paths use the modest Perspective elev so +Y stays screen-up.
    assert elev_flat <= 26.0
    assert elev_tall >= elev_flat


def test_trajectory_targets_seventy_percent_fill_with_equal_axis_scale() -> None:
    xs = [i / 1000.0 for i in range(100)]
    ys = [0.4 + (i % 8) / 1000.0 for i in range(100)]
    zs = [0.2 + i / 2000.0 for i in range(100)]
    vp = _viewport_for_overview_dock(xs, ys, zs)

    robust_x_span = xs[97] - xs[1]
    displayed_x_span = vp.xlim[1] - vp.xlim[0]
    assert robust_x_span / displayed_x_span == pytest.approx(
        _TRAJECTORY_TARGET_FILL, abs=0.03
    )

    limit_spans = (
        vp.xlim[1] - vp.xlim[0],
        vp.ylim[1] - vp.ylim[0],
        vp.zlim[1] - vp.zlim[0],
    )
    unit_scales = [
        aspect / span for aspect, span in zip(vp.box_aspect, limit_spans, strict=True)
    ]
    assert max(unit_scales) - min(unit_scales) < 1e-9


def test_path_gradient_and_markers_emphasize_current_frame() -> None:
    colors, widths = _path_segment_styles(12, "#63d8ff")
    assert colors[0][:3] != colors[-1][:3]
    assert all(a[3] <= b[3] for a, b in zip(colors, colors[1:]))
    assert all(a <= b for a, b in zip(widths, widths[1:]))
    assert _CURRENT_DOT_SIZE > _START_DOT_SIZE > _END_DOT_SIZE
