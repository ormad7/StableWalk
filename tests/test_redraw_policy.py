"""Visible-canvas and playback/resize redraw gates."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.redraw_policy import (
    full_tab_redraw_blocked,
    mark_resize_active,
    mpl_canvas_is_mapped,
    skip_unmapped_canvas_paint,
    widget_is_mapped,
)


def test_full_tab_redraw_blocked_while_playing_or_resizing() -> None:
    gui = SimpleNamespace(playing=False, _ui_resize_active=False)
    assert full_tab_redraw_blocked(gui) is False
    gui.playing = True
    assert full_tab_redraw_blocked(gui) is True
    gui.playing = False
    mark_resize_active(gui, True)
    assert full_tab_redraw_blocked(gui) is True
    mark_resize_active(gui, False)
    assert full_tab_redraw_blocked(gui) is False


def test_skip_unmapped_canvas_without_tk() -> None:
    assert skip_unmapped_canvas_paint(None) is True
    assert widget_is_mapped(None) is False
    assert mpl_canvas_is_mapped(None) is False


def test_paint_graphs_skips_full_tab_rebuild_while_playing() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI

    class _Gui:
        playing = True
        calls: list[str] = []

        def _group_visible(self, group: str) -> bool:
            return group in {"overview", "video", "skeleton"}

        def _update_chart(self) -> None:
            self.calls.append("chart")

        def _update_joint_motion_analysis_chart(self, **_k) -> None:
            self.calls.append("joint")

        def refresh_motion_analysis_tab(self) -> None:
            self.calls.append("full_tab")

        def _rebuild_knee_angle_series(self) -> None:
            self.calls.append("rebuild")

        def _apply_default_knee_chart_mode(self) -> None:
            return None

        def _mark_analysis_panels_dirty(self, *groups: str) -> None:
            self.calls.append("dirty")

        def _update_compare_chart(self) -> None:
            self.calls.append("compare")

    gui = _Gui()
    StableWalkGUI._paint_graphs_panels(gui)  # type: ignore[arg-type]
    assert "full_tab" not in gui.calls
    assert "rebuild" not in gui.calls


def test_chart_updates_skip_hidden_groups() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI

    class _Gui:
        playing = True
        chart_canvas = object()
        ax_chart = object()
        sequence = object()
        pose_indices = [0]
        canvas_biomech = object()
        fig_biomech = object()
        _biomech_analysis = object()
        canvas_contact_gait = object()
        fig_contact_gait = object()
        _foot_contact = object()
        _gait_cycle = object()
        _estimated_vgrf = object()

        def _group_visible(self, group: str) -> bool:
            return group == "overview"

    gui = _Gui()
    StableWalkGUI._update_chart(gui)  # type: ignore[arg-type]
    StableWalkGUI._update_biomechanics_chart(gui)  # type: ignore[arg-type]
    StableWalkGUI._update_contact_gait_chart(gui)  # type: ignore[arg-type]
    StableWalkGUI._update_joint_motion_analysis_chart(gui)  # type: ignore[arg-type]
