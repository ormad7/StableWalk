"""Tests for GUI layout debug helpers."""

from __future__ import annotations

from unittest import mock

from stablewalk.ui.tk.gui_layout_debug import audit_gui_widget_counts, format_gui_layout_audit


class _FakeCanvas:
    pass


def test_audit_gui_widget_counts_singletons():
    gui = mock.Mock()
    gui.canvas_3d = _FakeCanvas()
    gui.chart_canvas = _FakeCanvas()
    gui.canvas_dof_traj = _FakeCanvas()
    gui.canvas_robot = None
    gui.video_label = object()
    gui.root = mock.Mock()
    gui.root.winfo_children.return_value = []

    counts = audit_gui_widget_counts(gui)
    assert counts["canvas_3d"] == 0  # not FigureCanvasTkAgg instance
    assert counts["video_label_instances"] == 1

    text = format_gui_layout_audit(gui)
    assert "Video canvas instances" in text
    assert "Main scroll canvas" in text
