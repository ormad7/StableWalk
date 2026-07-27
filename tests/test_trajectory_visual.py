"""3D trajectory path styling — fade, tail, and camera helpers."""

from __future__ import annotations

import unittest

from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
from stablewalk.ui.viewers.dof_trajectory_3d import (
    _PATH_FADE_ALPHA_MAX,
    _PATH_FADE_ALPHA_MIN,
    _PATH_LINE_WIDTH,
    _path_segment_styles,
    _tail_segment_slice,
    estimate_body_height_m,
    meters_to_display_cm,
    remember_trajectory_camera,
    reset_trajectory_camera,
    stature_display_scale,
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
        self.assertLessEqual(_PATH_FADE_ALPHA_MIN, 0.40)
        colors, _ = _path_segment_styles(8, "#4dabf7")
        self.assertLess(colors[0][3], colors[-1][3] * 0.70)

    def test_path_line_is_not_a_fat_tube(self) -> None:
        # Thick enough to read at a glance; still a line, not a solid tube.
        self.assertLessEqual(_PATH_LINE_WIDTH, 4.0)
        self.assertGreaterEqual(_PATH_LINE_WIDTH, 2.5)
        _colors, widths = _path_segment_styles(8, "#4dabf7", line_width=_PATH_LINE_WIDTH)
        self.assertLessEqual(max(widths), _PATH_LINE_WIDTH * 1.4)


class StatureDisplayScaleTests(unittest.TestCase):
    def test_inflated_normalized_pose_maps_to_adult_stature(self) -> None:
        from stablewalk.models.gait_motion import GaitMotionRecording

        # Demo "normalized" poses often span ~2.9 m vertically.
        snap = SkeletonSnapshot(
            frame_index=0,
            time_s=0.0,
            joints={
                "head": JointSample("head", Vec3(0.0, 1.45, 0.0)),
                "pelvis": JointSample("pelvis", Vec3(0.0, 0.0, 0.0)),
                "left_ankle": JointSample("left_ankle", Vec3(0.0, -1.45, 0.0)),
            },
        )
        recording = GaitMotionRecording(source="test", fps=25.0, snapshots=[snap])
        height = estimate_body_height_m(recording)
        self.assertIsNotNone(height)
        assert height is not None
        self.assertGreater(height, 2.0)
        scale = stature_display_scale(recording)
        # Mid-shank knee (~−0.75 raw) → ~−44 cm after adult stature mapping.
        knee_raw = -0.75
        self.assertAlmostEqual(
            meters_to_display_cm(knee_raw, scale=scale),
            knee_raw * (1.70 / height) * 100.0,
            places=3,
        )
        self.assertLess(abs(meters_to_display_cm(knee_raw, scale=scale)), 55.0)

    def test_already_normalized_pose_keeps_scale_one(self) -> None:
        from stablewalk.models.gait_motion import GaitMotionRecording

        snap = SkeletonSnapshot(
            frame_index=0,
            time_s=0.0,
            joints={
                "head": JointSample("head", Vec3(0.0, 0.55, 0.0)),
                "pelvis": JointSample("pelvis", Vec3(0.0, 0.0, 0.0)),
                "left_ankle": JointSample("left_ankle", Vec3(0.0, -0.45, 0.0)),
            },
        )
        recording = GaitMotionRecording(source="test", fps=25.0, snapshots=[snap])
        self.assertEqual(stature_display_scale(recording), 1.0)

    def test_viewport_cache_uses_same_stature_scale_as_path(self) -> None:
        """Axis limits must use scaled meters so ticks match the cm readout."""
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from stablewalk.models.gait_motion import GaitMotionRecording
        from stablewalk.ui.viewers.dof_trajectory_3d import _get_cached_stable_viewport

        snaps = []
        for i in range(8):
            snaps.append(
                SkeletonSnapshot(
                    frame_index=i,
                    time_s=i * 0.04,
                    joints={
                        "head": JointSample("head", Vec3(0.0, 1.45, 0.0)),
                        "pelvis": JointSample("pelvis", Vec3(0.0, 0.0, 0.0)),
                        "right_knee": JointSample(
                            "right_knee", Vec3(0.05, -0.75 + 0.01 * i, 0.02 * i)
                        ),
                        "left_ankle": JointSample("left_ankle", Vec3(0.0, -1.45, 0.0)),
                    },
                )
            )
        recording = GaitMotionRecording(source="test", fps=25.0, snapshots=snaps)
        scale = stature_display_scale(recording)
        self.assertNotEqual(scale, 1.0)
        fig = Figure()
        ax = fig.add_subplot(111, projection="3d")
        ax._stablewalk_overview_dock = True
        vp = _get_cached_stable_viewport(
            ax,
            recording,
            "right_knee",
            coord_mode="ROOT-RELATIVE",
            motion_series=None,
            floor_y=None,
            position_scale=scale,
        )
        self.assertIsNotNone(vp)
        assert vp is not None
        # Scaled knee ~−0.44 m → ylim centre near that, not raw −0.75.
        mid_y = 0.5 * (vp.ylim[0] + vp.ylim[1])
        self.assertLess(abs(mid_y - (-0.75 * scale)), 0.12)


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
