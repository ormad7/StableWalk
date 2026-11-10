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
        self.assertEqual(text, "Playback · Playing")

    def test_analysis_state_preview_paused(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _analysis_state

        gui = SimpleNamespace(
            _status_bar_flash_text=None,
            _status_bar_flash_until=0.0,
            _presentation_mode=False,
            _pipeline_callback=None,
            _pending_video_load=None,
            _analysis_mode="preview",
            _analysis_max_frames=900,
            playing=False,
            skeleton_player=SimpleNamespace(frame_count=10, state=SimpleNamespace(stopped=False)),
            sequence=object(),
        )
        text, _fg = _analysis_state(gui)
        self.assertEqual(text, "Preview · Paused")

    def test_analysis_state_full_playing(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _analysis_state

        gui = SimpleNamespace(
            _status_bar_flash_text=None,
            _status_bar_flash_until=0.0,
            _presentation_mode=False,
            _pipeline_callback=None,
            _pending_video_load=None,
            _analysis_mode="full",
            _analysis_max_frames=None,
            playing=True,
            skeleton_player=SimpleNamespace(frame_count=10, state=SimpleNamespace(stopped=False)),
            sequence=object(),
        )
        text, _fg = _analysis_state(gui)
        self.assertEqual(text, "Full analysis · Playing")

    def test_analysis_state_analyzing_not_playing(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _analysis_state

        gui = SimpleNamespace(
            _status_bar_flash_text=None,
            _status_bar_flash_until=0.0,
            _presentation_mode=False,
            _pipeline_callback=None,
            _pending_video_load=None,
            _analysis_compute_active=True,
            _analysis_running=True,
            _processing=False,
            playing=True,
            skeleton_player=SimpleNamespace(frame_count=10, state=SimpleNamespace(stopped=False)),
            sequence=object(),
        )
        text, _fg = _analysis_state(gui)
        self.assertEqual(text, "Analyzing")

    def test_selected_joint_none(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _selected_joint_label

        gui = SimpleNamespace(_active_dof_item_id=lambda: None)
        self.assertEqual(_selected_joint_label(gui), "—")

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

    def test_opensim_footer_uses_canonical_short_statuses(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _opensim_status_text

        base = {
            "_presentation_mode": False,
            "_opensim_status_override": None,
            "_opensim_ik_state": None,
            "_opensim_export_completed": False,
            "sequence": None,
            "gait_motion": None,
        }
        not_started = SimpleNamespace(
            **base,
            _opensim_sdk_available=True,
            _opensim_model_valid=True,
        )
        export_partial = SimpleNamespace(
            **{**base, "_opensim_export_completed": True, "sequence": object()},
            _opensim_sdk_available=True,
            _opensim_model_valid=True,
        )
        unavailable = SimpleNamespace(
            **base,
            _opensim_sdk_available=False,
            _opensim_model_valid=False,
        )
        ik_complete = SimpleNamespace(
            **{**base, "_opensim_ik_state": "Completed"},
            _opensim_sdk_available=True,
            _opensim_model_valid=True,
        )
        ik_failed = SimpleNamespace(
            **{**base, "_opensim_ik_state": "Failed"},
            _opensim_sdk_available=True,
            _opensim_model_valid=True,
        )

        self.assertEqual(_opensim_status_text(not_started)[0], "Not Started")
        self.assertEqual(_opensim_status_text(export_partial)[0], "Export Complete · IK unavailable")
        self.assertEqual(_opensim_status_text(ik_complete)[0], "IK Complete")
        self.assertEqual(_opensim_status_text(ik_failed)[0], "IK Failed")
        self.assertEqual(_opensim_status_text(unavailable)[0], "Not Started")

    def test_opensim_footer_does_not_infer_complete_from_files_or_override(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _opensim_status_text

        gui = SimpleNamespace(
            _presentation_mode=False,
            _opensim_status_override="StableWalk IK: Completed\nleftover text",
            _opensim_ik_state=None,
            _opensim_export_completed=True,
            _opensim_sdk_available=True,
            _opensim_model_valid=True,
            _opensim_export_files={"TRC": True, "MOT": True, "IK_MOT": True, "MAPPED": True},
            sequence=object(),
            gait_motion=None,
        )
        self.assertEqual(_opensim_status_text(gui)[0], "Export Complete · IK unavailable")

    def test_opensim_tooltip_explains_partial(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _opensim_tooltip_text

        gui = SimpleNamespace(
            _opensim_ik_state=None,
            _opensim_export_completed=True,
            _opensim_sdk_available=True,
            _opensim_model_valid=True,
            sequence=object(),
            gait_motion=None,
        )
        tip = _opensim_tooltip_text(gui, "Partial")
        self.assertIn("OpenSim Pipeline: Complete", tip)
        self.assertIn("IK: Partial", tip)

    def test_status_bar_reflow_hides_low_priority_fields_first(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _reflow_status_bar

        class _Widget:
            def __init__(self) -> None:
                self.visible = True
                self.width = None

            def grid(self) -> None:
                self.visible = True

            def grid_remove(self) -> None:
                self.visible = False

            def configure(self, **kwargs) -> None:
                self.width = kwargs.get("width", self.width)

        class _Host:
            def __init__(self) -> None:
                self.columns = {}

            def columnconfigure(self, index, **kwargs) -> None:
                self.columns[index] = kwargs

        keys = ("frame", "time", "fps", "joint", "analysis", "opensim")
        cells = {key: _Widget() for key in keys}
        labels = {key: _Widget() for key in keys}
        science = _Widget()
        gui = SimpleNamespace(
            _status_bar_cells=cells,
            _status_bar_labels=labels,
            _status_bar_fields_host=_Host(),
            lbl_status_science=science,
        )

        _reflow_status_bar(gui, width=700)

        self.assertFalse(cells["fps"].visible)
        self.assertFalse(cells["joint"].visible)
        self.assertTrue(cells["analysis"].visible)
        self.assertTrue(cells["opensim"].visible)
        self.assertFalse(science.visible)

    def test_status_field_fits_width_and_keeps_full_tooltip(self) -> None:
        from stablewalk.ui.tk.dashboard_status_bar import _set_field

        class _Label:
            def __init__(self) -> None:
                self.values = {"text": "", "fg": "", "width": 8}

            def cget(self, key):
                return self.values[key]

            def configure(self, **kwargs) -> None:
                self.values.update(kwargs)

        label = _Label()
        gui = SimpleNamespace(_status_bar_labels={"joint": label})
        _set_field(gui, "joint", "Left Knee Flexion")

        self.assertEqual(label.values["text"], "Left Kn…")
        self.assertIn("Left Knee Flexion", label._tooltip_text)

    def test_narrow_transport_yields_width_to_timeline(self) -> None:
        from stablewalk.ui.tk.dashboard_responsive import _apply_transport_layout

        class _Widget:
            def __init__(self) -> None:
                self.visible = True

            def grid(self, **_kwargs) -> None:
                self.visible = True

            def grid_remove(self) -> None:
                self.visible = False

        analysis = _Widget()
        frame = _Widget()
        gui = SimpleNamespace(
            _transport_analysis=analysis,
            _transport_frame=frame,
            _status_bar_cells={},
            _status_bar_labels={},
        )
        _apply_transport_layout(gui, 960)
        # Compact layout restacks sampling/speed instead of hiding them.
        self.assertTrue(analysis.visible)
        self.assertTrue(frame.visible)
        _apply_transport_layout(gui, 1366)
        self.assertTrue(analysis.visible)
        self.assertTrue(frame.visible)


if __name__ == "__main__":
    unittest.main()
