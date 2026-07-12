"""Tests for contact vs displayed foot-clearance consistency."""

from __future__ import annotations

import unittest

from stablewalk.analysis.contact_clearance_consistency import (
    ContactClearanceConsistencyConfig,
    assess_foot_contact_clearance,
    find_inconsistent_frames,
    pipeline_documentation_lines,
)
from stablewalk.analysis.gait_cycle_analysis import (
    ContactThresholds,
    FootLandmarkSample,
    raw_contact_desire,
)


def _thresholds(*, leg: float = 0.45) -> ContactThresholds:
    entry = leg * 0.055
    exit_c = leg * 0.14
    max_disp = max(exit_c, leg * 0.10)
    return ContactThresholds(
        entry_clearance_m=entry,
        exit_clearance_m=exit_c,
        entry_max_vel_m_s=0.28,
        exit_min_vel_m_s=0.42,
        leg_length_m=leg,
        entry_normalized=entry / leg,
        exit_normalized=exit_c / leg,
        clearance_p10_m=leg * 0.02,
        clearance_p50_m=leg * 0.08,
        clearance_p90_m=leg * 0.18,
        max_display_exit_clearance_m=max_disp,
        max_display_exit_normalized=max_disp / leg,
    )


class ContactClearanceConsistencyTests(unittest.TestCase):
    def test_pipeline_documentation_has_seven_items(self) -> None:
        lines = pipeline_documentation_lines()
        self.assertEqual(len(lines), 7)

    def test_consistent_contact_near_floor(self) -> None:
        t = _thresholds()
        result = assess_foot_contact_clearance(
            side="left",
            contact=True,
            heel_clearance_m=0.02,
            toe_clearance_m=0.025,
            ankle_clearance_m=0.018,
            contact_foot_clearance_m=0.018,
            visibility=0.9,
            leg_length_m=t.leg_length_m,
            entry_clearance_m=t.entry_clearance_m,
            exit_clearance_m=t.exit_clearance_m,
        )
        self.assertEqual(result.verdict, "CONSISTENT")

    def test_inconsistent_high_displayed_with_contact(self) -> None:
        t = _thresholds()
        result = assess_foot_contact_clearance(
            side="right",
            contact=True,
            heel_clearance_m=0.069,
            toe_clearance_m=0.070,
            ankle_clearance_m=0.020,
            contact_foot_clearance_m=0.020,
            visibility=0.9,
            leg_length_m=t.leg_length_m,
            entry_clearance_m=t.entry_clearance_m,
            exit_clearance_m=t.exit_clearance_m,
        )
        self.assertEqual(result.verdict, "INCONSISTENT")

    def test_raw_contact_exits_when_displayed_clearance_high(self) -> None:
        t = _thresholds()
        sample = FootLandmarkSample(
            heel_clearance_m=0.069,
            toe_clearance_m=0.070,
            ankle_clearance_m=0.020,
            foot_clearance_m=0.020,
            heel_velocity_m_s=0.15,
            toe_velocity_m_s=0.15,
            ankle_velocity_m_s=0.10,
            visibility=0.9,
        )
        self.assertFalse(
            raw_contact_desire(sample, currently_in_contact=True, thresholds=t)
        )

    def test_enforce_displayed_clearance_contact_exit(self) -> None:
        from stablewalk.analysis.gait_cycle_analysis import (
            enforce_displayed_clearance_contact_exit,
        )

        t = _thresholds()
        samples = [
            FootLandmarkSample(0.02, 0.02, 0.02, 0.02, 0, 0, 0, 0.9),
            FootLandmarkSample(0.40, 0.41, 0.05, 0.40, 0.3, 0.3, 0.1, 0.9),
        ]
        out = enforce_displayed_clearance_contact_exit([1, 1], samples, t)
        self.assertEqual(out, [1, 0])

    def test_find_inconsistent_frames(self) -> None:
        rows = [
            {
                "frame": 1,
                "left_contact_state": "CONTACT",
                "left_consistency": "INCONSISTENT",
            },
            {
                "frame": 2,
                "right_contact_state": "SWING",
                "right_consistency": "CONSISTENT",
            },
        ]
        hits = find_inconsistent_frames(rows)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["inconsistent_side"], "left")


if __name__ == "__main__":
    unittest.main()
