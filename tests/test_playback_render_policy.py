"""Playback render policy — live surfaces vs frozen chrome."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from stablewalk.ui.tk.playback_render_policy import (
    LIVE_PLAY_SURFACES,
    MAX_LIVE_WIDGET_HZ,
    PLAY_CHART_HZ,
    STATIC_WHILE_PLAYING,
    TARGET_PLAY_HZ,
    TYPICAL_SESSION_RSS_BUDGET_MB,
    allow_graph_geometry_rebuild,
    allow_play_surface,
    clear_paint_signatures,
    configure_text_if_changed,
    live_widgets_due,
    mark_analysis_dirty,
    paint_signature_changed,
    playing_skips_static,
    rate_limited_due,
)


class PlaybackRenderPolicyTests(unittest.TestCase):
    def test_target_hz_is_sixty(self) -> None:
        self.assertEqual(TARGET_PLAY_HZ, 60)

    def test_live_widget_cap_is_thirty(self) -> None:
        self.assertEqual(MAX_LIVE_WIDGET_HZ, 30)
        self.assertEqual(PLAY_CHART_HZ, 30)
        self.assertLess(MAX_LIVE_WIDGET_HZ, TARGET_PLAY_HZ)
        self.assertEqual(TYPICAL_SESSION_RSS_BUDGET_MB, 500.0)

    def test_graph_geometry_frozen_while_playing_or_scrubbing(self) -> None:
        gui = SimpleNamespace(playing=True, _scrubbing=False)
        self.assertFalse(allow_graph_geometry_rebuild(gui))
        gui.playing = False
        gui._scrubbing = True
        self.assertFalse(allow_graph_geometry_rebuild(gui))
        gui._scrubbing = False
        self.assertTrue(allow_graph_geometry_rebuild(gui))
        self.assertTrue(allow_graph_geometry_rebuild(gui, force=True))

    def test_live_widgets_rate_limit(self) -> None:
        gui = SimpleNamespace()
        self.assertTrue(live_widgets_due(gui, now=1.0))
        self.assertFalse(live_widgets_due(gui, now=1.01))
        self.assertTrue(live_widgets_due(gui, now=1.05))
        self.assertTrue(rate_limited_due(gui, "scrub_panels", max_hz=30.0, now=2.0))
        self.assertFalse(rate_limited_due(gui, "scrub_panels", max_hz=30.0, now=2.01))

    def test_live_surfaces_match_requirements(self) -> None:
        self.assertEqual(
            LIVE_PLAY_SURFACES,
            frozenset(
                {
                    "video",
                    "skeleton",
                    "trajectory_tip",
                    "timeline_cursor",
                    "coordinate_inspector",
                    "gait_phase_indicator",
                }
            ),
        )

    def test_static_surfaces_blocked_while_playing(self) -> None:
        for name in (
            "metric_cards",
            "graphs",
            "tabs",
            "labels",
            "legends",
            "statistics",
            "layouts",
            "joint_information",
            "clinical_hud",
            "status_bar",
            "panels_trees",
            "motion_stats",
        ):
            self.assertIn(name, STATIC_WHILE_PLAYING)
            self.assertFalse(allow_play_surface(name, playing=True))
            self.assertTrue(allow_play_surface(name, playing=False))

    def test_live_surfaces_allowed_while_playing(self) -> None:
        for name in LIVE_PLAY_SURFACES:
            self.assertTrue(allow_play_surface(name, playing=True))

    def test_playing_skips_static_helper(self) -> None:
        gui = SimpleNamespace(playing=True)
        self.assertTrue(playing_skips_static(gui, "graphs"))
        self.assertFalse(playing_skips_static(gui, "video"))
        gui.playing = False
        self.assertFalse(playing_skips_static(gui, "graphs"))

    def test_mark_analysis_dirty(self) -> None:
        gui = SimpleNamespace()
        mark_analysis_dirty(gui, "knee", "joint_motion")
        self.assertEqual(gui._analysis_panel_dirty, {"knee", "joint_motion"})

    def test_configure_text_if_changed(self) -> None:
        class _Lbl:
            def __init__(self) -> None:
                self.text = "a"
                self.calls = 0

            def cget(self, key: str) -> str:
                assert key == "text"
                return self.text

            def configure(self, **kwargs) -> None:
                self.calls += 1
                self.text = kwargs["text"]

        lbl = _Lbl()
        self.assertTrue(configure_text_if_changed(lbl, "b"))
        self.assertEqual(lbl.calls, 1)
        self.assertFalse(configure_text_if_changed(lbl, "b"))
        self.assertEqual(lbl.calls, 1)

    def test_paint_signature_skips_unchanged(self) -> None:
        gui = SimpleNamespace()
        self.assertTrue(paint_signature_changed(gui, "video", ("f0", 640, 480)))
        self.assertFalse(paint_signature_changed(gui, "video", ("f0", 640, 480)))
        self.assertTrue(paint_signature_changed(gui, "video", ("f1", 640, 480)))
        self.assertTrue(paint_signature_changed(gui, "skeleton", 12))
        clear_paint_signatures(gui, "video")
        self.assertTrue(paint_signature_changed(gui, "video", ("f1", 640, 480)))
        self.assertFalse(paint_signature_changed(gui, "skeleton", 12))
        clear_paint_signatures(gui)
        self.assertTrue(paint_signature_changed(gui, "skeleton", 12))


if __name__ == "__main__":
    unittest.main()
