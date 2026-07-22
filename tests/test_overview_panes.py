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
