"""3D trajectory path styling — fade, tail, and camera helpers."""

from __future__ import annotations

import unittest

from stablewalk.ui.viewers.dof_trajectory_3d import (
    _PATH_FADE_ALPHA_MAX,
    _PATH_FADE_ALPHA_MIN,
    _path_segment_styles,
    _tail_segment_slice,
    remember_trajectory_camera,
    reset_trajectory_camera,
    zoom_trajectory_camera,
)


class TrajectoryVisualTests(unittest.TestCase):
    def test_segment_alpha_increases_toward_current(self) -> None:
        colors, _widths = _path_segment_styles(12, "#4dabf7")
        alphas = [c[3] for c in colors]
        self.assertGreater(alphas[-1], alphas[0])
        self.assertGreaterEqual(alphas[0], _PATH_FADE_ALPHA_MIN * 0.9)
        self.assertLessEqual(alphas[-1], _PATH_FADE_ALPHA_MAX)

    def test_segment_width_increases_toward_current(self) -> None:
        _colors, widths = _path_segment_styles(10, "#4dabf7")
        self.assertGreater(widths[-1], widths[0])

    def test_tail_slice_covers_recent_fraction(self) -> None:
        seg_count = 20
        tail_start = _tail_segment_slice(seg_count)
        self.assertGreater(tail_start, 0)
        self.assertLess(tail_start, seg_count - 1)

    def test_previous_samples_are_strongly_faded(self) -> None:
        self.assertLess(_PATH_FADE_ALPHA_MIN, 0.35)
        colors, _ = _path_segment_styles(8, "#4dabf7")
        self.assertLess(colors[0][3], colors[-1][3] * 0.55)


class TrajectoryCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from stablewalk.ui.viewers.dof_trajectory_3d import setup_single_dof_trajectory_axes

        cls.fig = Figure(figsize=(4, 3), dpi=80)
        cls.ax = cls.fig.add_subplot(111, projection="3d")
        setup_single_dof_trajectory_axes(cls.ax)

    def test_zoom_changes_camera_zoom_factor(self) -> None:
        zoom_trajectory_camera(self.ax, 1.25)
        self.assertGreater(float(self.ax._stablewalk_camera_zoom), 1.0)
        remember_trajectory_camera(self.ax)
        self.assertIsNotNone(getattr(self.ax, "_stablewalk_user_camera", None))
        reset_trajectory_camera(self.ax)
        self.assertIsNone(getattr(self.ax, "_stablewalk_user_camera", None))


if __name__ == "__main__":
    unittest.main()
