"""Tests for foot clearance dashboard display model."""

from __future__ import annotations

import unittest

from stablewalk.analysis.ground_reference import (
    FootClearanceReading,
    GroundReferencePlane,
    compute_foot_clearance_reading,
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
        self.assertIn("Unavailable", panel.left.current_display)
        self.assertEqual(panel.confidence, "LOW")

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
        self.assertIn("min(heel, toe)", FOOT_CLEARANCE_FOOT_REFERENCE)
        self.assertIn("Units: centimeters", text)
        self.assertIn("LEFT FOOT", text)
        self.assertIn("RIGHT FOOT", text)
        self.assertIn("Advanced clearance statistics", text)
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
        self.assertEqual(OVERVIEW_SECTION_TITLE, "FOOT-TO-FLOOR DISTANCE")
        self.assertEqual(OVERVIEW_DISTANCE_CAPTION, "DISTANCE FROM FLOOR")
        self.assertEqual(measurement_confidence_label("HIGH"), "Measurement Confidence: High")
        self.assertEqual(
            foot_clearance_confidence_label("MODERATE"),
            "Foot Clearance Confidence: Moderate",
        )

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
        self.assertIn("Measurement Confidence:", text)

    def test_debug_lines_populated(self) -> None:
        recording, snapshot = _bilateral_snapshot()
        panel = foot_clearance_dashboard_for_panel(snapshot, recording, 1.0)
        assert panel is not None
        self.assertGreaterEqual(len(panel.debug_lines), 7)
        self.assertTrue(any("Coordinate units:" in line for line in panel.debug_lines))
        self.assertTrue(any("valid swing samples" in line.lower() for line in panel.debug_lines))


if __name__ == "__main__":
    unittest.main()
