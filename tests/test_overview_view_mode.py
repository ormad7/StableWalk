"""Tests for Overview professional view modes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from stablewalk.ui.tk.dashboard_overview_view_mode import (
    DEFAULT_VIEW_MODE,
    VIEW_MODE_IDS,
    VIEW_MODE_OVERLAY,
    VIEW_MODE_RECONSTRUCTION_FULL,
    VIEW_MODE_SIDE_BY_SIDE,
    VIEW_MODE_SKELETON_ONLY,
    VIEW_MODE_TOGGLE_OPTIONS,
    VIEW_MODE_VIDEO_ONLY,
    current_overview_view_mode,
    load_overview_view_mode,
    preferences_path,
    save_overview_view_mode,
)


class OverviewViewModeTests(unittest.TestCase):
    def test_known_mode_ids(self) -> None:
        self.assertIn(VIEW_MODE_VIDEO_ONLY, VIEW_MODE_IDS)
        self.assertIn(VIEW_MODE_SKELETON_ONLY, VIEW_MODE_IDS)
        self.assertIn(VIEW_MODE_SIDE_BY_SIDE, VIEW_MODE_IDS)
        self.assertIn(VIEW_MODE_OVERLAY, VIEW_MODE_IDS)
        self.assertIn(VIEW_MODE_RECONSTRUCTION_FULL, VIEW_MODE_IDS)

    def test_persist_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefs = Path(tmp) / "ui_preferences.json"
            with mock.patch(
                "stablewalk.ui.tk.dashboard_overview_view_mode.preferences_path",
                return_value=prefs,
            ):
                save_overview_view_mode(VIEW_MODE_OVERLAY)
                self.assertEqual(load_overview_view_mode(), VIEW_MODE_OVERLAY)
                data = json.loads(prefs.read_text(encoding="utf-8"))
                self.assertEqual(data["overview_view_mode"], VIEW_MODE_OVERLAY)

    def test_invalid_pref_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefs = Path(tmp) / "ui_preferences.json"
            prefs.write_text('{"overview_view_mode": "not_a_mode"}', encoding="utf-8")
            with mock.patch(
                "stablewalk.ui.tk.dashboard_overview_view_mode.preferences_path",
                return_value=prefs,
            ):
                self.assertEqual(load_overview_view_mode(), DEFAULT_VIEW_MODE)

    def test_current_mode_prefers_var(self) -> None:
        gui = SimpleNamespace(
            _overview_view_mode_var=SimpleNamespace(get=lambda: VIEW_MODE_VIDEO_ONLY),
            _overview_view_mode=VIEW_MODE_SIDE_BY_SIDE,
        )
        self.assertEqual(current_overview_view_mode(gui), VIEW_MODE_VIDEO_ONLY)

    def test_preferences_path_under_output(self) -> None:
        path = preferences_path()
        self.assertTrue(str(path).endswith("ui_preferences.json"))

    def test_default_view_mode_is_side_by_side(self) -> None:
        self.assertEqual(DEFAULT_VIEW_MODE, VIEW_MODE_SIDE_BY_SIDE)

    def test_primary_toggle_is_split_overlay_video_skeleton_3d_full(self) -> None:
        labels = [label for _mid, label in VIEW_MODE_TOGGLE_OPTIONS]
        self.assertEqual(labels, ["Split", "Overlay", "Video", "Skeleton", "3D Full"])
        ids = [mid for mid, _label in VIEW_MODE_TOGGLE_OPTIONS]
        self.assertEqual(
            ids,
            [
                VIEW_MODE_SIDE_BY_SIDE,
                VIEW_MODE_OVERLAY,
                VIEW_MODE_VIDEO_ONLY,
                VIEW_MODE_SKELETON_ONLY,
                VIEW_MODE_RECONSTRUCTION_FULL,
            ],
        )

    def test_missing_prefs_open_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefs = Path(tmp) / "ui_preferences.json"
            with mock.patch(
                "stablewalk.ui.tk.dashboard_overview_view_mode.preferences_path",
                return_value=prefs,
            ):
                self.assertEqual(load_overview_view_mode(), VIEW_MODE_SIDE_BY_SIDE)

    def test_reset_layout_does_not_reload_analysis(self) -> None:
        import inspect

        from stablewalk.ui.tk.dashboard_overview_panes import reset_overview_layout
        from stablewalk.ui.tk.overview_responsive import (
            apply_overview_viz_arrangement,
            reflow_overview_viz_panels,
        )

        for fn in (
            reset_overview_layout,
            apply_overview_viz_arrangement,
            reflow_overview_viz_panels,
        ):
            src = inspect.getsource(fn).lower()
            assert "process_video" not in src
            assert "_load_demo" not in src
            assert "run_pipeline" not in src


if __name__ == "__main__":
    unittest.main()
