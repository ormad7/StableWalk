"""Tests for dashboard chart zoom/pan navigation helpers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_navigation import (
    ChartNavState,
    apply_chart_xlim,
    capture_chart_home_xlim,
    finalize_chart_view,
    is_chart_zoomed_out,
    reset_chart_navigation,
    update_secondary_annotation_visibility,
    zoom_span_fraction,
    zoom_xlim_at,
)
from stablewalk.ui.viewers.chart_style import mark_secondary_annotation


def test_zoom_xlim_at_centers_on_cursor() -> None:
    home = (0.0, 10.0)
    zoomed = zoom_xlim_at((0.0, 10.0), center=5.0, scale=2.0, home=home)
    assert zoomed[0] == 2.5
    assert zoomed[1] == 7.5


def test_apply_chart_xlim_persists_view() -> None:
    fig = Figure(figsize=(5, 4))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    ax1.plot([0.0, 5.0, 10.0], [1.0, 2.0, 1.5])
    ax2.plot([0.0, 5.0, 10.0], [0.2, 0.4, 0.3])
    capture_chart_home_xlim(fig)
    apply_chart_xlim(fig, (2.0, 8.0))
    assert ax1.get_xlim() == (2.0, 8.0)
    assert ax2.get_xlim() == (2.0, 8.0)
    state = getattr(fig, "_chart_nav_state")
    assert isinstance(state, ChartNavState)
    assert state.view_xlim == (2.0, 8.0)


def test_reset_chart_navigation_clears_saved_view() -> None:
    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0], [0.0, 1.0])
    capture_chart_home_xlim(fig)
    apply_chart_xlim(fig, (0.2, 0.8))
    reset_chart_navigation(fig)
    state = getattr(fig, "_chart_nav_state")
    assert state.view_xlim is None
    assert state.home_xlim is None


def test_secondary_annotations_hide_when_zoomed_out() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 10.0], [0.0, 1.0])
    marker = mark_secondary_annotation(ax.axvline(5.0, color="white", alpha=0.5))
    capture_chart_home_xlim(fig)
    finalize_chart_view(fig)
    assert is_chart_zoomed_out(fig)
    assert zoom_span_fraction(fig) >= 0.9
    assert marker.get_visible() is False

    apply_chart_xlim(fig, (3.0, 6.0))
    assert is_chart_zoomed_out(fig) is False
    update_secondary_annotation_visibility(fig)
    assert marker.get_visible() is True
