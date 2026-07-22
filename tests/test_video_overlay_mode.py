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
