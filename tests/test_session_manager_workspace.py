"""Unit tests for session workspace camera/graph helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from stablewalk.ui.tk.session_manager import (
    capture_workspace_state,
    restore_workspace_state,
)


class _FakeAx:
    def __init__(self) -> None:
        self.elev = 20.0
        self.azim = -60.0
        self._stablewalk_camera_zoom = 1.5
        self._stablewalk_pan_offset = (0.1, 0.0, -0.2)
        self._stablewalk_user_camera = (20.0, -60.0)

    def view_init(self, elev=None, azim=None) -> None:
        if elev is not None:
            self.elev = elev
        if azim is not None:
            self.azim = azim


class SessionManagerWorkspaceTests(unittest.TestCase):
    def test_capture_and_restore_camera(self) -> None:
        ax = _FakeAx()
        gui = SimpleNamespace(
            ax_dof_traj=ax,
            ax_dof_traj_overview=None,
            ax_3d=None,
            fig=None,
            fig_biomech=None,
            fig_contact_gait=None,
            chart_canvas=None,
            canvas_biomech=None,
            canvas_contact_gait=None,
            playing=False,
            var_overlay_direction=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            var_overlay_contact=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            var_overlay_com=SimpleNamespace(get=lambda: False, set=lambda _v: None),
            var_overlay_bos=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            var_overlay_com_velocity=SimpleNamespace(get=lambda: False, set=lambda _v: None),
            var_overlay_ground=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            var_knee_chart_axis=SimpleNamespace(get=lambda: "Gait Cycle %", set=lambda _v: None),
            var_knee_angle_source=SimpleNamespace(get=lambda: "Auto", set=lambda _v: None),
            var_dof_projection=SimpleNamespace(get=lambda: "3D", set=lambda _v: None),
            var_dof_coord_mode=SimpleNamespace(get=lambda: "ROOT-RELATIVE", set=lambda _v: None),
            smooth_motion=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            show_skeleton=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            highlight_dof=SimpleNamespace(get=lambda: True, set=lambda _v: None),
            dof_table_display_mode=SimpleNamespace(get=lambda: "Tracking History", set=lambda _v: None),
            skeleton_display_mode=SimpleNamespace(get=lambda: "Solid", set=lambda _v: None),
        )
        with mock.patch(
            "stablewalk.ui.tk.dashboard_overview_view_mode.current_overview_view_mode",
            return_value="side_by_side",
        ):
            state = capture_workspace_state(gui)

        self.assertEqual(state["graphs"]["knee_chart_axis"], "Gait Cycle %")
        cam = state["cameras"]["trajectory"]
        self.assertEqual(cam["elev"], 20.0)
        self.assertEqual(cam["zoom"], 1.5)

        ax2 = _FakeAx()
        ax2.elev = 0.0
        ax2.azim = 0.0
        gui2 = SimpleNamespace(
            ax_dof_traj=ax2,
            ax_dof_traj_overview=None,
            ax_3d=None,
            fig=None,
            fig_biomech=None,
            fig_contact_gait=None,
            chart_canvas=None,
            canvas_biomech=None,
            canvas_contact_gait=None,
            skeleton_display_mode=SimpleNamespace(set=lambda _v: None),
            var_knee_chart_axis=SimpleNamespace(set=lambda _v: None),
            var_dof_projection=SimpleNamespace(set=lambda _v: None),
            var_knee_angle_source=SimpleNamespace(set=lambda _v: None),
            var_dof_coord_mode=SimpleNamespace(set=lambda _v: None),
            dof_table_display_mode=SimpleNamespace(set=lambda _v: None),
            smooth_motion=SimpleNamespace(set=lambda _v: None),
            show_skeleton=SimpleNamespace(set=lambda _v: None),
            highlight_dof=SimpleNamespace(set=lambda _v: None),
            var_overlay_direction=SimpleNamespace(set=lambda _v: None),
            var_overlay_contact=SimpleNamespace(set=lambda _v: None),
            var_overlay_com=SimpleNamespace(set=lambda _v: None),
            var_overlay_bos=SimpleNamespace(set=lambda _v: None),
            var_overlay_com_velocity=SimpleNamespace(set=lambda _v: None),
            var_overlay_ground=SimpleNamespace(set=lambda _v: None),
        )
        with mock.patch(
            "stablewalk.ui.tk.dashboard_overview_view_mode.apply_overview_view_mode"
        ):
            restore_workspace_state(gui2, state)
        self.assertAlmostEqual(ax2.elev, 20.0)
        self.assertAlmostEqual(ax2._stablewalk_camera_zoom, 1.5)


if __name__ == "__main__":
    unittest.main()
