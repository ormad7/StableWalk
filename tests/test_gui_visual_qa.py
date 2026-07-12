"""Unit tests for GUI visual QA regression helpers."""

from __future__ import annotations

from unittest import mock

from stablewalk.ui.tk.gui_visual_qa import (
    GuiRegressionCheck,
    format_qa_report,
    run_gui_regression_assertions,
)


class _FakeWidget:
    def __init__(self, *, mapped: bool = True, text: str = "42 / 100", master=None):
        self._mapped = mapped
        self._text = text
        self.master = master
        self._geom = (100, 200, 120, 28)

    def winfo_ismapped(self) -> bool:
        return self._mapped

    def winfo_rootx(self) -> int:
        return self._geom[0]

    def winfo_rooty(self) -> int:
        return self._geom[1]

    def winfo_width(self) -> int:
        return self._geom[2]

    def winfo_height(self) -> int:
        return self._geom[3]

    def update_idletasks(self) -> None:
        return

    def cget(self, key: str) -> str:
        if key == "text":
            return self._text
        return ""


def test_run_gui_regression_detects_missing_transport():
    gui = mock.Mock()
    gui.root = _FakeWidget()
    gui.video_label = object()
    gui.canvas_3d = None
    gui.chart_canvas = None
    gui.canvas_dof_traj = None
    gui._transport_row = None
    gui._dash_scroll_canvas = None
    gui._analysis_scroll_canvas = None
    gui._dash_scroll_bottom_pad = 40

    checks = run_gui_regression_assertions(gui)
    names = {c.name for c in checks}
    assert "singleton__transport_row" in names
    transport_check = next(c for c in checks if c.name == "singleton__transport_row")
    assert not transport_check.passed


def test_format_qa_report_includes_conclusion():
    from stablewalk.ui.tk.gui_visual_qa import GuiVisualQAResult

    result = GuiVisualQAResult(
        category="Normal",
        resolution="1920x1080",
        video="normal_gait.mp4",
        checks=[GuiRegressionCheck("test", True, "ok")],
    )
    text = format_qa_report([result])
    assert "GUI Visual QA Report" in text
    assert "Normal" in text
    assert "## Conclusion" in text
