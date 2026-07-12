"""Unit tests for gait contact and cycle analysis pure functions."""

from __future__ import annotations

import unittest

from stablewalk.analysis.gait_cycle_analysis import (
    FootLandmarkSample,
    FrameContactState,
    GaitEvent,
    apply_contact_state_machine,
    assess_contact_confidence,
    classify_gait_phase,
    compute_temporal_metrics,
    detect_gait_events,
    enforce_min_run_length,
    raw_contact_desire,
    resolve_vertical_axis,
    symmetry_ratio,
)
from stablewalk.analysis.ground_reference import GroundReferencePlane


def _sample(
    clearance: float,
    vel: float = 0.05,
    *,
    in_contact: bool = False,
) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance + 0.005,
        ankle_clearance_m=clearance + 0.008,
        foot_clearance_m=clearance,
        heel_velocity_m_s=vel,
        toe_velocity_m_s=vel,
        ankle_velocity_m_s=vel,
        visibility=0.9,
    )


class GaitContactPureFunctionTests(unittest.TestCase):
    def test_resolve_vertical_axis_defaults_to_y(self) -> None:
        self.assertEqual(resolve_vertical_axis(None), "y")
        self.assertEqual(
            resolve_vertical_axis(GroundReferencePlane(floor_y=0.0, vertical_axis="y")),
            "y",
        )

    def test_classify_gait_phase(self) -> None:
        self.assertEqual(classify_gait_phase(1, 1), "DOUBLE_SUPPORT")
        self.assertEqual(classify_gait_phase(1, 0), "LEFT_STANCE")
        self.assertEqual(classify_gait_phase(0, 1), "RIGHT_STANCE")
        self.assertEqual(classify_gait_phase(0, 0), "UNCERTAIN")
        self.assertEqual(
            classify_gait_phase(
                0,
                0,
                contact_confidence=0.9,
                confidence_tier="HIGH",
                left_foot_clearance_m=0.15,
                right_foot_clearance_m=0.12,
            ),
            "FLIGHT",
        )

    def test_raw_contact_desire_hysteresis(self) -> None:
        low = _sample(0.01, vel=0.05)
        self.assertTrue(raw_contact_desire(low, currently_in_contact=False))
        high = _sample(0.12, vel=0.05)
        self.assertFalse(raw_contact_desire(high, currently_in_contact=False))
        mid = _sample(0.04, vel=0.05)
        self.assertTrue(raw_contact_desire(mid, currently_in_contact=True))
        fast = _sample(0.04, vel=0.55)
        self.assertFalse(raw_contact_desire(fast, currently_in_contact=True))

    def test_raw_contact_exits_on_displayed_clearance_above_ankle(self) -> None:
        from stablewalk.analysis.gait_cycle_analysis import ContactThresholds

        leg = 0.45
        entry = leg * 0.055
        exit_c = leg * 0.14
        max_disp = max(exit_c, leg * 0.10)
        thresholds = ContactThresholds(
            entry_clearance_m=entry,
            exit_clearance_m=exit_c,
            entry_max_vel_m_s=0.28,
            exit_min_vel_m_s=0.42,
            leg_length_m=leg,
            entry_normalized=entry / leg,
            exit_normalized=exit_c / leg,
            clearance_p10_m=0.01,
            clearance_p50_m=0.04,
            clearance_p90_m=0.08,
            max_display_exit_clearance_m=max_disp,
            max_display_exit_normalized=max_disp / leg,
        )
        sample = FootLandmarkSample(
            heel_clearance_m=0.069,
            toe_clearance_m=0.070,
            ankle_clearance_m=0.020,
            foot_clearance_m=0.020,
            heel_velocity_m_s=0.1,
            toe_velocity_m_s=0.1,
            ankle_velocity_m_s=0.05,
            visibility=0.9,
        )
        self.assertFalse(
            raw_contact_desire(sample, currently_in_contact=True, thresholds=thresholds)
        )

    def test_apply_contact_state_machine_debounces(self) -> None:
        desires = [False, True, True, True, False, False, False]
        states = apply_contact_state_machine(
            desires, min_hold_frames=2, min_run_frames=1
        )
        self.assertEqual(states[0], 0)
        self.assertEqual(states[1], 0)
        self.assertEqual(states[2], 1)
        self.assertEqual(states[-1], 0)

    def test_enforce_min_run_length_removes_blips(self) -> None:
        states = [0, 0, 1, 0, 0]
        cleaned = enforce_min_run_length(states, min_len=2)
        self.assertEqual(cleaned, [0, 0, 0, 0, 0])

    def test_detect_gait_events_from_contacts(self) -> None:
        frames = [
            FrameContactState(0, 0.0, 0, 0, "UNCERTAIN", _sample(0.1), _sample(0.1)),
            FrameContactState(1, 0.1, 1, 0, "LEFT_STANCE", _sample(0.01), _sample(0.1)),
            FrameContactState(2, 0.2, 1, 0, "LEFT_STANCE", _sample(0.01), _sample(0.1)),
            FrameContactState(3, 0.3, 0, 0, "UNCERTAIN", _sample(0.12), _sample(0.1)),
        ]
        events = detect_gait_events(frames)
        types = [e.event_type for e in events]
        self.assertIn("left_heel_strike", types)
        self.assertIn("left_toe_off", types)

    def test_symmetry_ratio(self) -> None:
        self.assertAlmostEqual(symmetry_ratio(0.5, 0.5), 1.0)
        self.assertAlmostEqual(symmetry_ratio(0.4, 0.8), 0.5)
        self.assertIsNone(symmetry_ratio(None, 0.5))

    def test_assess_contact_confidence_low_without_floor(self) -> None:
        score, tier = assess_contact_confidence(
            plane=None,
            foot_visibility_mean=0.3,
            valid_frame_ratio=0.5,
            heel_strike_count=0,
            duration_s=1.0,
            scale_mode="unknown",
        )
        self.assertLess(score, 0.5)
        self.assertEqual(tier, "LOW_CONFIDENCE")

    def test_assess_contact_confidence_high_with_floor(self) -> None:
        plane = GroundReferencePlane(floor_y=0.0)
        score, tier = assess_contact_confidence(
            plane=plane,
            foot_visibility_mean=0.9,
            valid_frame_ratio=0.95,
            heel_strike_count=4,
            duration_s=3.0,
            scale_mode="body_normalized",
        )
        self.assertGreaterEqual(score, 0.72)
        self.assertEqual(tier, "HIGH")

    def test_compute_temporal_metrics_cadence(self) -> None:
        per_frame = []
        t = 0.0
        for i in range(60):
            left = 1 if (i // 15) % 2 == 0 else 0
            right = 1 if (i // 15) % 2 == 1 else 0
            phase = classify_gait_phase(left, right)
            per_frame.append(
                FrameContactState(
                    i,
                    t,
                    left,
                    right,
                    phase,
                    _sample(0.01 if left else 0.1),
                    _sample(0.01 if right else 0.1),
                )
            )
            t += 1 / 30.0
        events = detect_gait_events(per_frame)
        metrics = compute_temporal_metrics(
            per_frame,
            events,
            cycles=[],
            fps=30.0,
            contact_confidence=0.8,
            confidence_tier="HIGH",
        )
        self.assertIsNotNone(metrics.left_stance_time_s)
        self.assertIsNotNone(metrics.cadence_steps_per_min)


if __name__ == "__main__":
    unittest.main()
