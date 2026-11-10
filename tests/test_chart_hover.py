"""Tests for shared chart hover tooltips."""

from __future__ import annotations

from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_hover import (
    ChartHoverPoint,
    FIG_HOVER_POINTS_ATTR,
    append_line_hover_points,
    format_chart_tooltip,
    format_inspect_crosshair_chip,
    hover_display_frame,
    set_figure_hover_points,
)


def test_format_chart_tooltip_includes_required_fields() -> None:
    text = format_chart_tooltip(
        ChartHoverPoint(
            ax=None,  # type: ignore[arg-type]
            x=1.0,
            y=42.0,
            frame_index=12,
            timestamp_s=0.4,
            value=42.0,
            joint_name="LEFT KNEE",
            metric_name="Knee flexion",
            unit="deg",
            velocity=1.5,
            acceleration=0.25,
            velocity_unit="°/s",
            acceleration_unit="°/s²",
        )
    )
    assert text.splitlines()[0] == "Inspect"
    assert "Frame: 12" in text
    assert "Time: 0.400 s" in text
    assert "Joint: LEFT KNEE" in text
    assert "Current value: 42 deg" in text
    assert "Velocity: 1.5 °/s" in text
    assert "Acceleration: 0.25 °/s²" in text
    assert "Metric: Knee flexion" in text


def test_hover_display_frame_prefers_list_pos() -> None:
    point = ChartHoverPoint(
        ax=None,  # type: ignore[arg-type]
        x=13.48,
        y=1.0,
        frame_index=323,
        list_pos=323,
        timestamp_s=13.48,
    )
    assert hover_display_frame(point) == 324
    text = format_chart_tooltip(point)
    assert text.startswith("Inspect")
    assert "Frame: 324" in text
    assert "Inspect · 13.48s" == format_inspect_crosshair_chip(13.48)


def test_format_chart_tooltip_cycle_mode_timestamp() -> None:
    text = format_chart_tooltip(
        ChartHoverPoint(
            ax=None,  # type: ignore[arg-type]
            x=50.0,
            y=30.0,
            timestamp_s=50.0,
            value=30.0,
            joint_name="LEFT KNEE",
            metric_name="Knee flexion (cycle mean)",
            unit="deg",
        )
    )
    assert text.splitlines()[0] == "Inspect"
    assert "50.0% gait cycle" in text
    assert "Time:" in text
    assert "Velocity: —" in text
    assert "Acceleration: —" in text


def test_append_line_hover_points_skips_nan() -> None:
    fig = Figure()
    ax = fig.add_subplot(111)
    points: list[ChartHoverPoint] = []
    append_line_hover_points(
        ax,
        [0.0, 0.1],
        [1.0, float("nan")],
        metric_name="Test metric",
        joint_name="Test joint",
        unit="m",
        frame_indices=[0, 1],
        list_positions=[0, 1],
        timestamps=[0.0, 0.1],
        hover_points=points,
    )
    assert len(points) == 1
    assert points[0].frame_index == 0
    set_figure_hover_points(fig, points)
    assert len(getattr(fig, FIG_HOVER_POINTS_ATTR)) == 1


def test_hover_offset_flips_left_near_right_edge() -> None:
    from stablewalk.ui.viewers.chart_hover import _edge_aware_hover_offset_pts

    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.canvas.draw()

    dx_right, _dy_right = _edge_aware_hover_offset_pts(ax, 0.98, 0.50)
    assert dx_right < 0

    dx_left, _dy_left = _edge_aware_hover_offset_pts(ax, 0.08, 0.50)
    assert dx_left > 0

    _dx_top, dy_top = _edge_aware_hover_offset_pts(ax, 0.50, 0.98)
    assert dy_top < 0


def test_hover_offset_keeps_tooltip_inside_figure() -> None:
    from stablewalk.ui.viewers.chart_hover import (
        _edge_aware_hover_offset_pts,
        _estimate_hover_size_px,
    )

    fig = Figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.92, bottom=0.12)
    fig.canvas.draw()

    text = format_chart_tooltip(
        ChartHoverPoint(
            ax=ax,
            x=0.98,
            y=0.50,
            frame_index=40,
            timestamp_s=1.95,
            value=135.75,
            joint_name="RIGHT KNEE",
            metric_name="Knee flexion",
            unit="deg",
            velocity=12.5,
            acceleration=0.25,
            velocity_unit="°/s",
            acceleration_unit="°/s²",
        )
    )
    dx_pts, dy_pts = _edge_aware_hover_offset_pts(ax, 0.98, 0.50, text=text)
    dpi = 100.0
    dx_px = dx_pts * (dpi / 72.0)
    dy_px = dy_pts * (dpi / 72.0)
    tw, th = _estimate_hover_size_px(text, dpi=dpi)
    disp = ax.transData.transform((0.98, 0.50))
    left = disp[0] + dx_px
    right = left + tw
    bottom = disp[1] + dy_px
    top = bottom + th
    pad = 7.0
    assert left >= fig.bbox.x0 - pad
    assert right <= fig.bbox.x1 + pad
    assert bottom >= fig.bbox.y0 - pad
    assert top <= fig.bbox.y1 + pad
