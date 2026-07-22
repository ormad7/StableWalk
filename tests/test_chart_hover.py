"""Tests for shared chart hover tooltips."""

from __future__ import annotations

from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_hover import (
    ChartHoverPoint,
    FIG_HOVER_POINTS_ATTR,
    append_line_hover_points,
    format_chart_tooltip,
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
        )
    )
    assert "Frame: 12" in text
    assert "Timestamp: 0.400 s" in text
    assert "Value: 42 deg" in text
    assert "Joint: LEFT KNEE" in text
    assert "Metric: Knee flexion" in text


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
    assert "50.0% gait cycle" in text


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
