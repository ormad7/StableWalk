"""Tests for the continuous bottom dashboard status bar."""

from __future__ import annotations

import unittest
from types import SimpleNamespace


class DashboardStatusBarTests(unittest.TestCase):
    def test_analysis_state_playing(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _analysis_state

        gui = SimpleNamespace(
            _status_bar_flash_text=None,
            _status_bar_flash_until=0.0,
            _presentation_mode=False,
            _pipeline_callback=None,
            _pending_video_load=None,
            playing=True,
            skeleton_player=SimpleNamespace(frame_count=10, state=SimpleNamespace(stopped=False)),
            sequence=object(),
        )
        text, _fg = _analysis_state(gui)
        self.assertEqual(text, "Playing")

    def test_selected_joint_none(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _selected_joint_label

        gui = SimpleNamespace(_active_dof_item_id=lambda: None)
        self.assertEqual(_selected_joint_label(gui), "None")

    def test_playback_speed_formatting(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _playback_speed_text

        gui = SimpleNamespace(play_speed=1.25)
        self.assertEqual(_playback_speed_text(gui), "1.25×")

    def test_status_proxy_sets_flash(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import StatusMessageProxy, set_status_flash

        gui = SimpleNamespace(
            _status_bar_labels={},
            _status_bar_flash_text=None,
            _status_bar_flash_until=0.0,
        )
        proxy = StatusMessageProxy(gui)
        proxy.configure(text="Loading demo…")
        self.assertEqual(gui._status_bar_flash_text, "Loading demo…")
        self.assertGreater(gui._status_bar_flash_until, 0.0)


if __name__ == "__main__":
    unittest.main()
