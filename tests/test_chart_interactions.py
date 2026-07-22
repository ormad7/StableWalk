"""Tests for unified 2D chart interactions (grid, nav finalize, hover, export)."""

from __future__ import annotations

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_hover import ChartHoverPoint, FIG_HOVER_POINTS_ATTR
from stablewalk.ui.viewers.chart_interactions import (
    attach_chart_crosshair,
    finalize_chart_interactions,
)
from stablewalk.ui.viewers.chart_navigation import (
    ChartNavState,
    FIG_NAV_STATE_ATTR,
    apply_chart_xlim,
    capture_chart_home_xlim,
)


def test_finalize_restores_zoom_and_registers_hover() -> None:
    fig = Figure(figsize=(4, 2), dpi=80)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    ax.set_xlim(0.0, 2.0)
    capture_chart_home_xlim(fig)
    apply_chart_xlim(fig, (0.5, 1.5))

    ax.clear()
    ax.plot([0.0, 1.0, 2.0], [0.0, 1.0, 0.5])
    ax.set_xlim(0.0, 2.0)
    points = [
        ChartHoverPoint(
            ax=ax,
            x=1.0,
            y=1.0,
            frame_index=3,
            timestamp_s=1.0,
            value=1.0,
            joint_name="L",
            metric_name="Test",
            unit="deg",
        )
    ]
    finalize_chart_interactions(fig, None, hover_points=points)
    assert abs(ax.get_xlim()[0] - 0.5) < 1e-6
    assert abs(ax.get_xlim()[1] - 1.5) < 1e-6
    assert len(getattr(fig, FIG_HOVER_POINTS_ATTR)) == 1
    del canvas


def test_crosshair_attach_is_idempotent() -> None:
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    fig.add_subplot(111)
    attach_chart_crosshair(fig, canvas)  # type: ignore[arg-type]
    attach_chart_crosshair(fig, canvas)  # type: ignore[arg-type]
    assert getattr(fig, "_chart_crosshair_cid", None) is not None


def test_nav_state_survives_clear() -> None:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    capture_chart_home_xlim(fig)
    state: ChartNavState = getattr(fig, FIG_NAV_STATE_ATTR)
    assert state.home_xlim is not None
    assert state.home_xlim[1] > state.home_xlim[0]
