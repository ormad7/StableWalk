"""Unit tests for mocap-style 3D trajectory chrome helpers."""

from __future__ import annotations

import unittest

from stablewalk.ui.viewers.dof_trajectory_3d import (
    _cumulative_xyz_distances_m,
    _fmt_path_xyz_line,
    _nice_scale_length_m,
    _path_metrics_hud_text,
    _tip_pulse_scale,
)


class TrajectoryMocapChromeTests(unittest.TestCase):
    def test_cumulative_distances(self) -> None:
        xs = [0.0, 3.0, 3.0]
        ys = [0.0, 0.0, 4.0]
        zs = [0.0, 0.0, 0.0]
        cum = _cumulative_xyz_distances_m(xs, ys, zs)
        self.assertEqual(len(cum), 3)
        self.assertAlmostEqual(cum[0], 0.0)
        self.assertAlmostEqual(cum[1], 3.0)
        self.assertAlmostEqual(cum[2], 7.0)

    def test_nice_scale_length(self) -> None:
        self.assertGreater(_nice_scale_length_m(0.20), 0.0)
        # Near 1–2–5 progression.
        self.assertIn(_nice_scale_length_m(0.10), (0.02, 0.05, 0.1, 0.2))

    def test_tip_pulse_scale_range(self) -> None:
        s = _tip_pulse_scale(animate=True)
        self.assertGreaterEqual(s, 1.0)
        self.assertLessEqual(s, 1.25)
        self.assertEqual(_tip_pulse_scale(animate=False), 1.0)

    def test_path_hud_includes_xyz_and_lengths(self) -> None:
        text = _path_metrics_hud_text(
            total_m=0.421,
            traveled_m=0.183,
            xyz_cm=(12.4, -3.1, 91.2),
            use_cm=True,
        )
        self.assertIn("Path length", text)
        self.assertIn("Displacement", text)
        self.assertIn("XYZ", text)
        self.assertIn("cm", text)
        self.assertIn("12.4", _fmt_path_xyz_line((12.4, -3.1, 91.2), use_cm=True))


if __name__ == "__main__":
    unittest.main()
