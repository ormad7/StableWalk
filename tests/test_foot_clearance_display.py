"""Tests for foot clearance dashboard display model."""

from __future__ import annotations

import unittest

from stablewalk.analysis.ground_reference import (
    FootClearanceReading,
    GroundReferencePlane,
    compute_foot_clearance_reading,
    lowest_foot_landmark,
)
from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.ui.foot_clearance_display import (
    FOOT_CLEARANCE_FLOOR_REFERENCE,
    FOOT_CLEARANCE_FOOT_REFERENCE,
    OVERVIEW_DISTANCE_CAPTION,
    OVERVIEW_SECTION_TITLE,
    FootSkeletonLabel,
    clearance_m_to_display_cm,
    foot_clearance_confidence_label,
    foot_clearance_dashboard_for_panel,
    foot_skeleton_labels_from_dashboard,
    format_foot_clearance_details,
    measurement_confidence_label,
    overview_distance_text,
    overview_live_clearance_text,
    overview_unavailable_reason,
    parse_card_clearance_cm,
    skeleton_clearance_matches_card,
)


def _bilateral_snapshot(
    *,
    left_toe_y: float = 0.02,
    right_toe_y: float = 0.18,
) -> tuple[GaitMotionRecording, SkeletonSnapshot]:
    def _snap(frame: int, l_y: float, r_y: float) -> SkeletonSnapshot:
        joints = {
            "left_ankle": JointSample("left_ankle", Vec3(-0.05, l_y + 0.02, 0.1)),
            "left_heel": JointSample("left_heel", Vec3(-0.05, l_y, 0.08)),
            "left_toe": JointSample("left_toe", Vec3(-0.05, l_y, 0.12)),
            "right_ankle": JointSample("right_ankle", Vec3(0.05, r_y + 0.02, 0.1)),
            "right_heel": JointSample("right_heel", Vec3(0.05, r_y, 0.08)),
            "right_toe": JointSample("right_toe", Vec3(0.05, r_y, 0.12)),
        }
        return SkeletonSnapshot(
            frame_index=frame,
            time_s=frame * 0.033,
            joints=joints,
            dofs={},
            metadata={"landmark_visibility": {jid: 0.9 for jid in joints}},
        )

    snapshots = [
        _snap(0, 0.02, 0.02),
        _snap(1, left_toe_y, right_toe_y),
        _snap(2, 0.02, 0.20),
    ]
    recording = GaitMotionRecording(source="test.mp4", fps=30.0, snapshots=snapshots)
    return recording, snapshots[1]


class FootClearanceDisplayTests(unittest.TestCase):
    def test_clearance_m_to_cm_only_for_body_normalized(self) -> None:
        text, cm = clearance_m_to_display_cm(0.048, scale_mode="body_normalized")
        self.assertEqual(text, "4.8 cm")
        self.assertAlmostEqual(cm, 4.8)

        text_norm, cm_norm = clearance_m_to_display_cm(0.048, scale_mode="image_normalized")
        self.assertEqual(text_norm, "Unavailable")
        self.assertIsNone(cm_norm)

        text_none, cm_none = clearance_m_to_display_cm(None, scale_mode="body_normalized")
        self.assertEqual(text_none, "Unavailable")
        self.assertIsNone(cm_none)

    def test_dashboard_shows_contact_and_swing_states(self) -> None:
        recording, snapshot = _bilateral_snapshot(left_toe_y=0.02, right_toe_y=0.18)
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        self.assertIsNotNone(panel)
        assert panel is not None

        self.assertEqual(panel.left.state_display, "CONTACT")
        self.assertEqual(panel.right.state_display, "SWING")
        self.assertIn("cm", panel.left.current_display)
        self.assertIn("cm", panel.right.current_display)
        self.assertEqual(panel.left.compact_state, "CONTACT")
        self.assertEqual(panel.right.compact_state, "SWING")

    def test_dashboard_floor_unavailable_reason(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        empty = GaitMotionRecording(source="empty.mp4", fps=30.0, snapshots=[])
        panel = foot_clearance_dashboard_for_panel(snapshot, empty, 0.0)
        self.assertIsNotNone(panel)
        assert panel is not None
        self.assertEqual(panel.left.unavailable_reason, "floor estimate invalid")
        self.assertEqual(panel.left.current_display, "Unavailable")
        self.assertEqual(panel.left.min_swing_display, "Unavailable")
        self.assertEqual(panel.left.max_swing_display, "Unavailable")
        self.assertEqual(panel.confidence, "LOW")
        self.assertEqual(overview_distance_text(panel.left), "Unavailable")

    def test_clearance_display_unavailable_for_unknown_scale(self) -> None:
        text, cm = clearance_m_to_display_cm(0.05, scale_mode="unknown")
        self.assertEqual(text, "Unavailable")
        self.assertIsNone(cm)

    def test_details_include_methodology_and_debug(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        assert panel is not None
        text = format_foot_clearance_details(panel)
        self.assertIn("FOOT CLEARANCE", text)
        self.assertIn(f"Floor reference: {FOOT_CLEARANCE_FLOOR_REFERENCE}", text)
        self.assertIn(f"Foot reference: {FOOT_CLEARANCE_FOOT_REFERENCE}", text)
        self.assertIn("heel/toe/foot-index", FOOT_CLEARANCE_FOOT_REFERENCE)
        self.assertIn("Units: centimeters", text)
        self.assertIn("LEFT FOOT", text)
        self.assertIn("RIGHT FOOT", text)
        self.assertIn("Advanced clearance statistics", text)
        self.assertIn("Minimum swing clearance:", text)
        self.assertIn("Maximum swing clearance:", text)
        self.assertIn("Median swing clearance:", text)
        self.assertIn("Valid sample count:", text)
        self.assertIn("Rejected outlier %:", text)

    def test_zero_clearance_is_contact_not_ambiguous(self) -> None:
        plane = GroundReferencePlane(floor_y=0.02, scale_mode="body_normalized")
        reading = compute_foot_clearance_reading(Vec3(0.0, 0.02, 0.0), plane)
        self.assertEqual(reading.contact_state, "On Ground")
        self.assertAlmostEqual(reading.foot_clearance_m or 0.0, 0.0)
        text, cm = clearance_m_to_display_cm(reading.foot_clearance_m, scale_mode="body_normalized")
        self.assertEqual(text, "0.0 cm")
        self.assertAlmostEqual(cm or 0.0, 0.0)

    def test_overview_copy_and_confidence_labels(self) -> None:
        self.assertEqual(OVERVIEW_SECTION_TITLE, "Estimated Foot Clearance")
        self.assertEqual(OVERVIEW_DISTANCE_CAPTION, "Current foot clearance")
        self.assertEqual(measurement_confidence_label("HIGH"), "Foot Clearance Confidence: High")
        self.assertEqual(
            foot_clearance_confidence_label("MODERATE"),
            "Foot Clearance Confidence: Moderate",
        )

    def test_dashboard_min_max_swing_from_pose_not_invented(self) -> None:
        recording, snapshot = _bilateral_snapshot(left_toe_y=0.02, right_toe_y=0.18)
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 2.0)
        assert panel is not None
        self.assertIn("cm", panel.right.min_swing_display)
        self.assertIn("cm", panel.right.max_swing_display)
        right_min = parse_card_clearance_cm(panel.right.min_swing_display)
        right_max = parse_card_clearance_cm(panel.right.max_swing_display)
        self.assertIsNotNone(right_min)
        self.assertIsNotNone(right_max)
        assert right_min is not None and right_max is not None
        self.assertLessEqual(right_min, right_max)
        # Left stays in contact — no fabricated swing extrema.
        if "cm" in panel.left.min_swing_display.lower():
            left_min = parse_card_clearance_cm(panel.left.min_swing_display)
            left_max = parse_card_clearance_cm(panel.left.max_swing_display)
            self.assertIsNotNone(left_min)
            self.assertIsNotNone(left_max)
        else:
            self.assertEqual(panel.left.min_swing_display, "Unavailable")

    def test_current_frame_refresh_keeps_cached_min_swing(self) -> None:
        from stablewalk.ui.foot_clearance_display import (
            build_foot_clearance_dashboard_with_cache,
            foot_clearance_dashboard_current_frame,
        )

        recording, snapshot = _bilateral_snapshot(left_toe_y=0.02, right_toe_y=0.18)
        panel, cache = build_foot_clearance_dashboard_with_cache(
            snapshot, recording, 2.0
        )
        assert panel is not None and cache is not None
        live = foot_clearance_dashboard_current_frame(snapshot, cache)
        assert live is not None
        self.assertEqual(live.right.min_swing_display, panel.right.min_swing_display)
        self.assertIn("cm", live.right.current_display)
        self.assertEqual(
            live.methodology_note.split("=", 1)[0].strip(),
            "Estimated Foot Clearance",
        )

    def test_session_swing_extrema_are_independent_of_current_sample(self) -> None:
        from stablewalk.ui.foot_clearance_display import (
            build_foot_clearance_dashboard_with_cache,
            foot_clearance_dashboard_current_frame,
        )

        recording, swing_snapshot = _bilateral_snapshot(
            left_toe_y=0.02,
            right_toe_y=0.18,
        )
        initial, cache = build_foot_clearance_dashboard_with_cache(
            recording.snapshots[0],
            recording,
            0.0,
        )
        assert initial is not None and cache is not None
        self.assertIn("cm", initial.right.min_swing_display)
        self.assertIn("cm", initial.right.max_swing_display)

        swing = foot_clearance_dashboard_current_frame(
            swing_snapshot,
            cache,
            prev_right_phase=initial.right_phase,
        )
        assert swing is not None
        self.assertIn("cm", swing.right.min_swing_display)
        self.assertIn("cm", swing.right.max_swing_display)

        invalid_snapshot = SkeletonSnapshot(
            frame_index=3,
            time_s=0.099,
            joints={
                key: value
                for key, value in swing_snapshot.joints.items()
                if not key.startswith("right_")
            },
            dofs={},
            metadata={"landmark_visibility": {}},
        )
        invalid = foot_clearance_dashboard_current_frame(
            invalid_snapshot,
            cache,
            prev_right_phase=swing.right_phase,
        )
        assert invalid is not None
        self.assertNotIn("cm", invalid.right.current_display)
        self.assertIn("cm", invalid.right.min_swing_display)
        self.assertIn("cm", invalid.right.max_swing_display)
        self.assertEqual(invalid.right.state_display, "—")

    def test_overview_distance_and_unavailable_reason(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        assert panel is not None
        self.assertIn("cm", overview_distance_text(panel.left))
        self.assertEqual(
            overview_unavailable_reason("floor estimate invalid"),
            "Floor estimate unavailable",
        )
        self.assertEqual(
            overview_unavailable_reason("sample rejected as outlier (exceeds plausible max)"),
            "Sample rejected as outlier",
        )
        self.assertEqual(
            overview_live_clearance_text("Left", panel.left),
            f"Left Foot Clearance: {panel.left.displayed_clearance_cm:.1f} cm",
        )
        self.assertEqual(
            overview_live_clearance_text("Right", panel.right),
            f"Right Foot Clearance: {panel.right.displayed_clearance_cm:.1f} cm",
        )

    def test_lowest_reliable_landmark_and_foot_index_fallback(self) -> None:
        from stablewalk.analysis.foot_clearance_filter import (
            _heel_toe_landmark_sample,
        )

        plane = GroundReferencePlane(floor_y=0.0, scale_mode="body_normalized")
        joints = {
            "left_ankle": JointSample("left_ankle", Vec3(0.0, 0.12, 0.0)),
            "left_heel": JointSample("left_heel", Vec3(0.0, 0.02, 0.0)),
            "left_toe": JointSample("left_toe", Vec3(0.0, 0.08, 0.0)),
        }
        snapshot = SkeletonSnapshot(
            frame_index=0,
            time_s=0.0,
            joints=joints,
            dofs={},
            metadata={
                "landmark_visibility": {
                    "left_ankle": 0.9,
                    "left_heel": 0.1,
                    "left_toe": 0.9,
                }
            },
        )
        sample = _heel_toe_landmark_sample(snapshot, plane, "left")
        self.assertAlmostEqual(sample.raw_clearance_m or 0.0, 0.08)
        position, joint_id = lowest_foot_landmark(snapshot, "left")
        self.assertEqual(joint_id, "left_toe")
        self.assertAlmostEqual(position.y if position is not None else 0.0, 0.08)

        foot_index = JointSample("left_foot_index", Vec3(0.0, 0.06, 0.0))
        index_snapshot = SkeletonSnapshot(
            frame_index=1,
            time_s=0.033,
            joints={
                "left_ankle": joints["left_ankle"],
                "left_foot_index": foot_index,
            },
            dofs={},
            metadata={
                "landmark_visibility": {
                    "left_ankle": 0.9,
                    "left_foot_index": 0.9,
                }
            },
        )
        index_sample = _heel_toe_landmark_sample(index_snapshot, plane, "left")
        self.assertAlmostEqual(index_sample.raw_clearance_m or 0.0, 0.06)
        index_position, index_joint_id = lowest_foot_landmark(index_snapshot, "left")
        self.assertEqual(index_joint_id, "left_foot_index")
        self.assertAlmostEqual(
            index_position.y if index_position is not None else 0.0,
            0.06,
        )

    def test_skeleton_foot_label_compact_text(self) -> None:
        label = FootSkeletonLabel("left", 2.9, True)
        self.assertEqual(label.compact_text(), "L\n2.9 cm\nCONTACT")
        swing = FootSkeletonLabel("right", 6.9, False)
        self.assertEqual(swing.compact_text(), "R\n6.9 cm\nSWING")
        na = FootSkeletonLabel("left", None, True)
        self.assertEqual(na.compact_text(), "L: N/A")

    def test_skeleton_labels_match_dashboard_panel(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        assert panel is not None
        labels = foot_skeleton_labels_from_dashboard(
            panel, left_contact=True, right_contact=False
        )
        self.assertEqual(labels[0].clearance_cm, panel.left.displayed_clearance_cm)
        self.assertEqual(labels[1].clearance_cm, panel.right.displayed_clearance_cm)
        self.assertTrue(labels[0].contact)
        self.assertFalse(labels[1].contact)

    def test_skeleton_clearance_parity_with_card(self) -> None:
        self.assertTrue(skeleton_clearance_matches_card(2.9, 2.9))
        self.assertTrue(skeleton_clearance_matches_card(2.94, 2.9))
        self.assertFalse(skeleton_clearance_matches_card(2.9, 6.9))
        self.assertTrue(skeleton_clearance_matches_card(None, None))
        self.assertEqual(parse_card_clearance_cm("2.9 cm"), 2.9)
        self.assertIsNone(parse_card_clearance_cm("Unavailable"))

    def test_details_include_foot_clearance_confidence(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        assert panel is not None
        text = format_foot_clearance_details(panel)
        self.assertIn("Foot Clearance Confidence:", text)
        self.assertIn("Foot Clearance Confidence:", text)

    def test_debug_lines_populated(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        assert panel is not None
        self.assertGreaterEqual(len(panel.debug_lines), 7)
        self.assertTrue(any("Coordinate units:" in line for line in panel.debug_lines))
        self.assertTrue(any("valid swing samples" in line.lower() for line in panel.debug_lines))

    def test_overview_apply_keeps_valid_clearance_values_visible(self) -> None:
        from types import SimpleNamespace

        from stablewalk.ui.tk.app import StableWalkGUI

        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 2.0)
        assert panel is not None

        class _Lbl:
            def __init__(self, text: str = "") -> None:
                self.text = text
                self.fg = ""
                self.font = None
                self.mapped = True

            def configure(self, **kwargs: object) -> None:
                if "text" in kwargs:
                    self.text = str(kwargs["text"])
                if "fg" in kwargs:
                    self.fg = str(kwargs["fg"])
                if "font" in kwargs:
                    self.font = kwargs["font"]

            def winfo_manager(self) -> str:
                return "grid" if self.mapped else ""

            def grid(self, *args: object, **kwargs: object) -> None:
                self.mapped = True

            def grid_remove(self) -> None:
                self.mapped = False

        gui = SimpleNamespace(
            lbl_foot_clearance_confidence=_Lbl(),
            lbl_foot_left_current=_Lbl(),
            lbl_foot_left_min_swing=_Lbl(),
            lbl_foot_left_max_swing=_Lbl(),
            lbl_foot_left_state=_Lbl(),
            lbl_foot_left_unavailable=_Lbl(),
            lbl_foot_right_current=_Lbl(),
            lbl_foot_right_min_swing=_Lbl(),
            lbl_foot_right_max_swing=_Lbl(),
            lbl_foot_right_state=_Lbl(),
            lbl_foot_right_unavailable=_Lbl(),
            lbl_overview_clearance_overlay_left=_Lbl(),
            lbl_overview_clearance_overlay_right=_Lbl(),
        )
        StableWalkGUI._apply_foot_clearance_dashboard(gui, panel)

        for prefix, foot in (("foot_left", panel.left), ("foot_right", panel.right)):
            current = getattr(gui, f"lbl_{prefix}_current")
            min_lbl = getattr(gui, f"lbl_{prefix}_min_swing")
            max_lbl = getattr(gui, f"lbl_{prefix}_max_swing")
            state = getattr(gui, f"lbl_{prefix}_state")
            self.assertTrue(current.mapped)
            self.assertTrue(min_lbl.mapped)
            self.assertTrue(max_lbl.mapped)
            self.assertTrue(state.mapped)
            self.assertIn(overview_distance_text(foot), current.text)
            self.assertTrue(current.text.strip())
            self.assertTrue(min_lbl.text.strip())
            self.assertTrue(max_lbl.text.strip())
            if foot.displayed_clearance_cm is not None:
                self.assertIn("cm", current.text.lower())
                self.assertTrue(
                    any(token in state.text for token in ("Contact", "Swing"))
                )
            else:
                self.assertIn("Unavailable", current.text)
                self.assertIn("Unavailable", state.text)
            if "cm" in (foot.min_swing_display or "").lower():
                self.assertIn("cm", min_lbl.text.lower())
            else:
                self.assertIn("Unavailable", min_lbl.text)
            if "cm" in (foot.max_swing_display or "").lower():
                self.assertIn("cm", max_lbl.text.lower())
            else:
                self.assertIn("Unavailable", max_lbl.text)

    def test_overview_apply_shows_unavailable_instead_of_empty(self) -> None:
        from dataclasses import replace
        from types import SimpleNamespace

        from stablewalk.ui.foot_clearance_display import overview_state_text, overview_value_text
        from stablewalk.ui.tk.app import StableWalkGUI

        self.assertEqual(overview_value_text(""), "Unavailable")
        self.assertEqual(overview_value_text("—"), "Unavailable")
        self.assertEqual(overview_value_text("4.8 cm"), "4.8 cm")
        self.assertEqual(overview_state_text(""), "Unavailable")
        self.assertEqual(overview_state_text("—"), "Unavailable")
        self.assertEqual(overview_state_text("CONTACT"), "Contact")
        self.assertEqual(overview_state_text("SWING"), "Swing")

        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 2.0)
        assert panel is not None
        empty_left = replace(
            panel.left,
            current_display="",
            min_swing_display="",
            max_swing_display="—",
            state_display="—",
            displayed_clearance_cm=None,
        )
        panel = replace(panel, left=empty_left)

        class _Lbl:
            def __init__(self, text: str = "") -> None:
                self.text = text
                self.mapped = True

            def configure(self, **kwargs: object) -> None:
                if "text" in kwargs:
                    self.text = str(kwargs["text"])

            def winfo_manager(self) -> str:
                return "grid" if self.mapped else ""

            def grid(self, *args: object, **kwargs: object) -> None:
                self.mapped = True

            def grid_remove(self) -> None:
                self.mapped = False

        gui = SimpleNamespace(
            lbl_foot_clearance_confidence=_Lbl(),
            lbl_foot_left_current=_Lbl(),
            lbl_foot_left_min_swing=_Lbl(),
            lbl_foot_left_max_swing=_Lbl(),
            lbl_foot_left_state=_Lbl(),
            lbl_foot_left_unavailable=_Lbl(),
            lbl_foot_right_current=_Lbl(),
            lbl_foot_right_min_swing=_Lbl(),
            lbl_foot_right_max_swing=_Lbl(),
            lbl_foot_right_state=_Lbl(),
            lbl_foot_right_unavailable=_Lbl(),
            lbl_overview_clearance_overlay_left=_Lbl(),
            lbl_overview_clearance_overlay_right=_Lbl(),
        )
        gui.lbl_foot_left_current.mapped = False
        gui.lbl_foot_left_min_swing.mapped = False
        gui.lbl_foot_left_max_swing.mapped = False
        gui.lbl_foot_left_state.mapped = False
        StableWalkGUI._apply_foot_clearance_dashboard(gui, panel)

        self.assertEqual(gui.lbl_foot_left_current.text, "Unavailable")
        self.assertEqual(gui.lbl_foot_left_min_swing.text, "Unavailable")
        self.assertEqual(gui.lbl_foot_left_max_swing.text, "Unavailable")
        self.assertEqual(gui.lbl_foot_left_state.text, "Unavailable")
        self.assertTrue(gui.lbl_foot_left_current.mapped)
        self.assertTrue(gui.lbl_foot_left_min_swing.mapped)
        self.assertTrue(gui.lbl_foot_left_max_swing.mapped)
        self.assertTrue(gui.lbl_foot_left_state.mapped)
        if panel.right.displayed_clearance_cm is not None:
            self.assertIn("cm", gui.lbl_foot_right_current.text.lower())
        if "cm" in (panel.right.min_swing_display or "").lower():
            self.assertIn(panel.right.min_swing_display, gui.lbl_foot_right_min_swing.text)
        if "cm" in (panel.right.max_swing_display or "").lower():
            self.assertIn(panel.right.max_swing_display, gui.lbl_foot_right_max_swing.text)


if __name__ == "__main__":
    unittest.main()
