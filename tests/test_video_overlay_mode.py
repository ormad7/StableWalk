"""Tests for Overlay Mode video+skeleton composition."""

from __future__ import annotations

import unittest

import numpy as np

from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.video_overlay_compose import (
    JointTrailBuffer,
    VideoOverlayLayers,
    compose_video_skeleton_overlay,
    estimate_image_com,
)


def _kp(name: str, x: float, y: float, vis: float = 1.0) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, z=0.0, visibility=vis)


def _sample_keypoints() -> list[Keypoint]:
    return [
        _kp("nose", 0.50, 0.12),
        _kp("left_shoulder", 0.42, 0.28),
        _kp("right_shoulder", 0.58, 0.28),
        _kp("left_elbow", 0.38, 0.40),
        _kp("right_elbow", 0.62, 0.40),
        _kp("left_hip", 0.45, 0.55),
        _kp("right_hip", 0.55, 0.55),
        _kp("left_knee", 0.44, 0.72),
        _kp("right_knee", 0.56, 0.72),
        _kp("left_ankle", 0.43, 0.90),
        _kp("right_ankle", 0.57, 0.90),
        _kp("left_foot_index", 0.42, 0.94),
        _kp("right_foot_index", 0.58, 0.94),
    ]


class VideoOverlayComposeTests(unittest.TestCase):
    def test_compose_changes_pixels_with_skeleton(self) -> None:
        base = np.full((120, 80, 3), 40, dtype=np.uint8)
        out = compose_video_skeleton_overlay(
            base,
            _sample_keypoints(),
            layers=VideoOverlayLayers(
                skeleton=True,
                joint_labels=False,
                com=False,
                ground_reaction=False,
                joint_trajectory=False,
            ),
            opacity=1.0,
        )
        self.assertEqual(out.shape, base.shape)
        self.assertFalse(np.array_equal(out, base))

    def test_opacity_zero_preserves_video(self) -> None:
        base = np.full((100, 80, 3), 55, dtype=np.uint8)
        out = compose_video_skeleton_overlay(
            base,
            _sample_keypoints(),
            layers=VideoOverlayLayers(skeleton=True),
            opacity=0.0,
        )
        np.testing.assert_array_equal(out, base)

    def test_all_layers_off_returns_original(self) -> None:
        base = np.full((60, 60, 3), 12, dtype=np.uint8)
        out = compose_video_skeleton_overlay(
            base,
            _sample_keypoints(),
            layers=VideoOverlayLayers(
                skeleton=False,
                joint_labels=False,
                com=False,
                ground_reaction=False,
                joint_trajectory=False,
                ground_plane=False,
            ),
            opacity=1.0,
        )
        np.testing.assert_array_equal(out, base)

    def test_estimate_com_near_hips(self) -> None:
        com = estimate_image_com(_sample_keypoints())
        self.assertIsNotNone(com)
        assert com is not None
        self.assertAlmostEqual(com[0], 0.5, delta=0.08)
        self.assertGreater(com[1], 0.35)
        self.assertLess(com[1], 0.75)

    def test_trail_buffer_builds_paths(self) -> None:
        buf = JointTrailBuffer(max_frames=10)
        for i in range(5):
            kps = [
                _kp("left_ankle", 0.4 + i * 0.01, 0.9),
                _kp("right_ankle", 0.6 + i * 0.01, 0.9),
            ]
            buf.update(i, kps)
        trails = buf.trails()
        self.assertIn("left_ankle", trails)
        self.assertEqual(len(trails["left_ankle"]), 5)

    def test_trail_clears_on_scrub_back(self) -> None:
        buf = JointTrailBuffer(max_frames=10)
        buf.update(5, [_kp("left_ankle", 0.4, 0.9)])
        buf.update(6, [_kp("left_ankle", 0.41, 0.9)])
        buf.update(3, [_kp("left_ankle", 0.39, 0.9)])
        trails = buf.trails()
        self.assertEqual(trails, {})

    def test_ground_line_uses_measured_y_only(self) -> None:
        from stablewalk.pose.video_overlay_compose import (
            estimate_image_ground_y,
            image_y_from_vertical_correspondence,
        )

        self.assertIsNone(estimate_image_ground_y([]))
        self.assertIsNone(
            estimate_image_ground_y(
                [_kp("nose", 0.5, 0.1)],
                foot_contact={"left": True, "right": False},
            )
        )
        y = estimate_image_ground_y(
            [
                _kp("left_heel", 0.4, 0.91),
                _kp("left_foot_index", 0.42, 0.93),
            ],
            foot_contact={"left": True, "right": False},
        )
        self.assertIsNotNone(y)
        assert y is not None
        self.assertAlmostEqual(y, 0.92, delta=0.02)

        mapped = image_y_from_vertical_correspondence(0.55, 0.90, 0.50, 0.02, 0.02)
        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertAlmostEqual(mapped, 0.90, delta=0.01)
        self.assertIsNone(
            image_y_from_vertical_correspondence(0.5, 0.9, 0.5, 0.5, 0.0)
        )

    def test_ground_plane_layer_paints_when_y_known(self) -> None:
        base = np.full((100, 80, 3), 40, dtype=np.uint8)
        out = compose_video_skeleton_overlay(
            base,
            [],
            layers=VideoOverlayLayers(
                skeleton=False,
                joint_labels=False,
                com=False,
                ground_reaction=False,
                joint_trajectory=False,
                ground_plane=True,
            ),
            opacity=1.0,
            ground_y_norm=0.80,
        )
        self.assertFalse(np.array_equal(out, base))

    def test_ground_plane_not_invented_without_landmarks(self) -> None:
        base = np.full((60, 60, 3), 12, dtype=np.uint8)
        out = compose_video_skeleton_overlay(
            base,
            [],
            layers=VideoOverlayLayers(
                skeleton=False,
                joint_labels=False,
                com=False,
                ground_reaction=False,
                joint_trajectory=False,
                ground_plane=True,
            ),
            opacity=1.0,
            ground_y_norm=None,
        )
        np.testing.assert_array_equal(out, base)

    def test_foot_clearance_hud_draws_bottom_labels(self) -> None:
        from stablewalk.pose.video_overlay_compose import draw_foot_clearance_hud

        base = np.full((160, 120, 3), 30, dtype=np.uint8)
        out = draw_foot_clearance_hud(base, left_cm=4.2, right_cm=2.8)
        self.assertEqual(out.shape, base.shape)
        self.assertFalse(np.array_equal(out, base))
        # Original buffer must stay untouched (shared decode/cache safety).
        self.assertTrue(np.array_equal(base, np.full((160, 120, 3), 30, dtype=np.uint8)))
        # Labels sit in the lower band of the frame.
        top = out[:80, :, :]
        bottom = out[100:, :, :]
        self.assertTrue(np.array_equal(top, np.full_like(top, 30)))
        self.assertFalse(np.array_equal(bottom, np.full_like(bottom, 30)))

    def test_foot_clearance_hud_unavailable_still_labels(self) -> None:
        from stablewalk.pose.video_overlay_compose import draw_foot_clearance_hud

        base = np.zeros((100, 80, 3), dtype=np.uint8)
        out = draw_foot_clearance_hud(base, left_cm=None, right_cm=None)
        self.assertFalse(np.array_equal(out, base))

    def test_foot_ground_drop_indicator_for_selected_sides(self) -> None:
        from stablewalk.pose.video_overlay_compose import draw_foot_ground_drop_indicators

        base = np.full((160, 120, 3), 25, dtype=np.uint8)
        out = draw_foot_ground_drop_indicators(
            base,
            _sample_keypoints(),
            ground_y_norm=0.96,
            sides=("left", "right"),
            left_cm=4.2,
            right_cm=2.8,
        )
        self.assertEqual(out.shape, base.shape)
        self.assertFalse(np.array_equal(out, base))
        # Source buffer unchanged.
        self.assertTrue(np.array_equal(base, np.full((160, 120, 3), 25, dtype=np.uint8)))

    def test_foot_ground_drop_skips_without_sides(self) -> None:
        from stablewalk.pose.video_overlay_compose import draw_foot_ground_drop_indicators

        base = np.full((80, 60, 3), 18, dtype=np.uint8)
        out = draw_foot_ground_drop_indicators(
            base,
            _sample_keypoints(),
            ground_y_norm=0.9,
            sides=(),
            left_cm=3.0,
            right_cm=3.0,
        )
        np.testing.assert_array_equal(out, base)


class OverlayModeControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import tkinter as tk
        from tkinter import ttk

        from stablewalk.ui.theme import apply_theme

        cls.root = tk.Tk()
        cls.root.withdraw()
        apply_theme(cls.root, ttk.Style(cls.root))

    @classmethod
    def tearDownClass(cls) -> None:
        cls.root.destroy()

    def test_controls_roundtrip_layers_and_opacity(self) -> None:
        import tkinter as tk
        from types import SimpleNamespace

        from stablewalk.ui.tk.overlay_mode_controls import (
            build_overlay_mode_controls,
            overlay_layers_from_gui,
            overlay_opacity_from_gui,
            set_overlay_mode_controls_visible,
        )

        parent = tk.Frame(self.root)
        parent.pack()
        gui = SimpleNamespace()
        changed = {"n": 0}

        build_overlay_mode_controls(
            gui, parent, row=0, on_change=lambda: changed.__setitem__("n", changed["n"] + 1)
        )
        set_overlay_mode_controls_visible(gui, True)
        self.root.update_idletasks()

        gui.var_video_overlay_opacity.set(40.0)
        self.assertAlmostEqual(overlay_opacity_from_gui(gui), 0.4, places=2)

        gui.var_video_overlay_skeleton.set(False)
        gui.var_video_overlay_labels.set(True)
        layers = overlay_layers_from_gui(gui)
        self.assertFalse(layers.skeleton)
        self.assertTrue(layers.joint_labels)
        self.assertTrue(layers.com)

        set_overlay_mode_controls_visible(gui, False)
        parent.destroy()


if __name__ == "__main__":
    unittest.main()
