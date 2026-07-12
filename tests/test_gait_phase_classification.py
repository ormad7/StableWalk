"""Tests for central gait-phase classification."""

from __future__ import annotations

import unittest

from stablewalk.analysis.gait_phase_classification import (
    classify_gait_phase_from_contacts,
    contact_to_display_state,
    format_gait_phase_display,
    validate_phase_contact_consistency,
)


class GaitPhaseClassificationTests(unittest.TestCase):
    def test_single_support_phases(self) -> None:
        self.assertEqual(
            classify_gait_phase_from_contacts(1, 0), "LEFT_STANCE"
        )
        self.assertEqual(
            classify_gait_phase_from_contacts(0, 1), "RIGHT_STANCE"
        )
        self.assertEqual(
            classify_gait_phase_from_contacts(1, 1), "DOUBLE_SUPPORT"
        )

    def test_bilateral_swing_uncertain_when_low_confidence(self) -> None:
        phase = classify_gait_phase_from_contacts(
            0,
            0,
            contact_confidence=0.3,
            confidence_tier="LOW_CONFIDENCE",
        )
        self.assertEqual(phase, "UNCERTAIN")

    def test_bilateral_swing_flight_when_airborne(self) -> None:
        phase = classify_gait_phase_from_contacts(
            0,
            0,
            contact_confidence=0.8,
            confidence_tier="HIGH",
            left_foot_clearance_m=0.15,
            right_foot_clearance_m=0.12,
        )
        self.assertEqual(phase, "FLIGHT")

    def test_contact_display_mapping(self) -> None:
        self.assertEqual(contact_to_display_state(1), "CONTACT")
        self.assertEqual(contact_to_display_state(0), "SWING")

    def test_consistency_left_stance(self) -> None:
        ok, expected = validate_phase_contact_consistency(1, 0, "LEFT_STANCE")
        self.assertTrue(ok)
        self.assertEqual(expected, "LEFT_STANCE")

    def test_consistency_rejects_blank_phase(self) -> None:
        ok, _ = validate_phase_contact_consistency(1, 0, "—")
        self.assertFalse(ok)

    def test_format_display(self) -> None:
        self.assertEqual(format_gait_phase_display("LEFT_STANCE"), "LEFT STANCE")
        self.assertEqual(format_gait_phase_display("FLIGHT"), "FLIGHT")


if __name__ == "__main__":
    unittest.main()
