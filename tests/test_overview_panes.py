"""Tests for Overview paned layout sash fractions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from stablewalk.ui.tk.dashboard_overview_panes import (
    DEFAULT_SASH_FRACTIONS,
    DEFAULT_TRAJ_SASH_FRACTIONS,
    MIN_PANE_SIDE,
    MIN_PANE_SKELETON,
    MIN_PANE_VIDEO,
    _clamp_fractions,
    _clamp_sash_pixels,
    clear_sash_fraction_prefs,
    load_sash_fractions,
    save_sash_fractions,
)


class OverviewPanesTests(unittest.TestCase):
    def test_default_fractions_are_ordered(self) -> None:
        self.assertLess(DEFAULT_SASH_FRACTIONS[0], DEFAULT_SASH_FRACTIONS[1])
        self.assertLess(DEFAULT_TRAJ_SASH_FRACTIONS[0], DEFAULT_TRAJ_SASH_FRACTIONS[1])
        path_frac = 1.0 - DEFAULT_TRAJ_SASH_FRACTIONS[1]
        video_frac = DEFAULT_TRAJ_SASH_FRACTIONS[0]
        skel_frac = DEFAULT_TRAJ_SASH_FRACTIONS[1] - DEFAULT_TRAJ_SASH_FRACTIONS[0]
        self.assertAlmostEqual(video_frac, 0.40, places=2)
        self.assertAlmostEqual(skel_frac, 0.30, places=2)
        self.assertAlmostEqual(path_frac, 0.30, places=2)
        self.assertGreater(video_frac, path_frac)

    def test_default_traj_fractions_survive_clamp(self) -> None:
        width = 1200
        x0, x1 = _clamp_sash_pixels(
            width,
            int(round(DEFAULT_TRAJ_SASH_FRACTIONS[0] * width)),
            int(round(DEFAULT_TRAJ_SASH_FRACTIONS[1] * width)),
        )
        video = x0 / width
        skeleton = (x1 - x0) / width
        path = (width - x1) / width
        self.assertGreaterEqual(video, 0.35)
        self.assertGreaterEqual(skeleton, 0.24)
        self.assertGreaterEqual(path, 0.24)
        self.assertGreaterEqual(video, skeleton)
        self.assertGreaterEqual(video, path)
        self.assertLessEqual(path, 0.40)

    def test_min_pane_sizes_positive(self) -> None:
        self.assertGreaterEqual(MIN_PANE_VIDEO, 160)
        self.assertGreaterEqual(MIN_PANE_SKELETON, 160)
        self.assertGreaterEqual(MIN_PANE_SIDE, 120)

    def test_clamp_fractions(self) -> None:
        f0, f1 = _clamp_fractions((0.01, 0.02))
        self.assertGreaterEqual(f1 - f0, 0.12)
        f0, f1 = _clamp_fractions((0.9, 0.95))
        self.assertLessEqual(f1, 0.88)

    def test_clamp_sash_pixels_respects_mins(self) -> None:
        width = 900
        x0, x1 = _clamp_sash_pixels(width, 10, 20)
        self.assertGreaterEqual(x0, MIN_PANE_VIDEO)
        self.assertGreaterEqual(x1 - x0, MIN_PANE_SKELETON)
        self.assertGreaterEqual(width - x1, MIN_PANE_SIDE)

    def test_clamp_sash_pixels_caps_path_and_keeps_video_floor(self) -> None:
        width = 1200
        # Saved prefs previously left Path ~2× Video (sash 1 too far left).
        x0, x1 = _clamp_sash_pixels(width, 180, 420)
        self.assertLessEqual(width - x1, int(width * 0.40) + 1)
        self.assertGreaterEqual(x0, min(int(width * 0.32), MIN_PANE_VIDEO))
        self.assertGreaterEqual(x1 - x0, MIN_PANE_SKELETON)
        self.assertGreaterEqual(width - x1, MIN_PANE_SIDE)

    def test_clamp_overflow_keeps_usable_3d_path(self) -> None:
        width = 720
        x0, x1 = _clamp_sash_pixels(
            width,
            240,
            480,
            min_video=280,
            min_skeleton=280,
            min_path=320,
        )
        self.assertEqual(x0 + (x1 - x0) + (width - x1), width)
        self.assertLessEqual(x1, width)
        self.assertGreaterEqual(width - x1, 180)
        self.assertGreaterEqual(x0, 120)
        self.assertGreaterEqual(x1 - x0, 120)

    def test_defaults_keep_usable_3d_at_laptop_and_fhd(self) -> None:
        for width in (1366, 1920):
            x0, x1 = _clamp_sash_pixels(
                width,
                int(round(DEFAULT_TRAJ_SASH_FRACTIONS[0] * width)),
                int(round(DEFAULT_TRAJ_SASH_FRACTIONS[1] * width)),
            )
            path = width - x1
            self.assertGreaterEqual(path, MIN_PANE_SIDE)
            self.assertGreaterEqual(x0, MIN_PANE_VIDEO)
            self.assertGreaterEqual(x1 - x0, MIN_PANE_SKELETON)
            self.assertAlmostEqual(x0 / width, 0.40, delta=0.06)

    def test_clamp_keeps_3d_floor_at_target_scales(self) -> None:
        from stablewalk.ui.tk.analysis_split import overview_pane_minsizes

        for width, scale in ((1366, 1.0), (1920, 1.0), (1366, 1.25), (1920, 1.50)):
            min_v, min_s, min_p = overview_pane_minsizes(scale)
            x0, x1 = _clamp_sash_pixels(
                width,
                int(round(DEFAULT_TRAJ_SASH_FRACTIONS[0] * width)),
                int(round(DEFAULT_TRAJ_SASH_FRACTIONS[1] * width)),
                min_video=min_v,
                min_skeleton=min_s,
                min_path=min_p,
            )
            self.assertEqual(x0 + (x1 - x0) + (width - x1), width)
            if min_v + min_s + min_p <= width:
                self.assertGreaterEqual(width - x1, min_p)
                self.assertGreaterEqual(x0, min_v)
                self.assertGreaterEqual(x1 - x0, min_s)
            else:
                self.assertGreaterEqual(width - x1, 160)

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            prefs = Path(tmp) / "ui_preferences.json"
            with mock.patch(
                "stablewalk.ui.tk.dashboard_overview_panes.preferences_path",
                return_value=prefs,
            ):
                save_sash_fractions((0.25, 0.60))
                self.assertEqual(load_sash_fractions(), (0.25, 0.60))
                data = json.loads(prefs.read_text(encoding="utf-8"))
                self.assertEqual(data["overview_sash_fractions"], [0.25, 0.60])
                save_sash_fractions((0.28, 0.72), traj=True)
                self.assertEqual(load_sash_fractions(traj=True), (0.28, 0.72))
                clear_sash_fraction_prefs()
                self.assertEqual(load_sash_fractions(), DEFAULT_SASH_FRACTIONS)
                self.assertEqual(load_sash_fractions(traj=True), DEFAULT_TRAJ_SASH_FRACTIONS)


if __name__ == "__main__":
    unittest.main()
