"""Tests for dashboard structural shell diagnostics."""

from __future__ import annotations

from unittest import mock

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stablewalk.ui.tk.dashboard_shell import (
    assert_dashboard_widget_singletons,
    build_structural_diagnostic_report,
)


class _FakeTkWidget:
    def __init__(self, master=None):
        self.master = master


class _FakeFigureCanvas:
    def __init__(self):
        self._widget = _FakeTkWidget()

    def get_tk_widget(self):
        return self._widget


def test_build_structural_diagnostic_report_sections():
    gui = mock.Mock()
    scroll_content = mock.Mock()
    scroll_content.__eq__ = lambda self, other: other is scroll_content

    video = mock.Mock()
    video.master = scroll_content

    class _Chain:
        master = scroll_content

    chart = _FakeFigureCanvas()
    chart.get_tk_widget().master = scroll_content

    gui.video_label = video
    gui.chart_canvas = chart
    gui.canvas_dof_traj = _FakeFigureCanvas()
    gui.canvas_dof_traj.get_tk_widget().master = scroll_content
    gui.canvas_3d = _FakeFigureCanvas()
    gui.canvas_3d.get_tk_widget().master = scroll_content
    gui._dash_scroll_content = scroll_content
    gui._dashboard_body = mock.Mock()
    gui.root = mock.Mock()
    gui.root.winfo_children.return_value = []

    report = build_structural_diagnostic_report(gui)
    assert "Why widgets visually overlap" in report
    assert "scroll content frame" in report.lower()


def test_assert_dashboard_widget_singletons_passes():
    gui = mock.Mock()
    gui.video_label = object()
    gui.chart_canvas = mock.Mock(spec=FigureCanvasTkAgg)
    gui.canvas_dof_traj = mock.Mock(spec=FigureCanvasTkAgg)
    gui.canvas_3d = mock.Mock(spec=FigureCanvasTkAgg)
    transport = mock.Mock()
    transport.master = mock.Mock()
    gui._transport_row = transport
    assert_dashboard_widget_singletons(gui)


def test_assert_dashboard_widget_singletons_fails():
    gui = mock.Mock()
    gui.video_label = None
    gui.chart_canvas = None
    gui.canvas_dof_traj = None
    gui.canvas_3d = None
    gui._transport_row = None
    try:
        assert_dashboard_widget_singletons(gui)
        assert False, "expected AssertionError"
    except AssertionError as exc:
        assert "singleton check failed" in str(exc)
