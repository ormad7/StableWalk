"""0.100 m gate: body-scale metres, +Y / floor_y, Unknown — not stance."""

from __future__ import annotations

from stablewalk.analysis.foot_clearance_filter import _phase_from_clearance as filter_phase
from stablewalk.analysis.gait_phase_classification import FLIGHT_CLEARANCE_THRESHOLD_M
from stablewalk.analysis.ground_reference import (
    CLEARANCE_GATE_RATIONALE,
    NEAR_GROUND_THRESHOLD_M,
    ON_GROUND_THRESHOLD_M,
    VERTICAL_AXIS,
    _phase_from_clearance,
    foot_contact_state,
)


def test_gate_is_body_scale_metres_not_clinical() -> None:
    assert NEAR_GROUND_THRESHOLD_M == 0.100
    assert FLIGHT_CLEARANCE_THRESHOLD_M == NEAR_GROUND_THRESHOLD_M
    assert VERTICAL_AXIS == "y"
    assert "body-scale" in CLEARANCE_GATE_RATIONALE
    assert "floor_y" in CLEARANCE_GATE_RATIONALE
    assert "not classify stance" in CLEARANCE_GATE_RATIONALE.lower() or "does not classify stance" in CLEARANCE_GATE_RATIONALE


def test_near_ground_is_not_stance_or_swing() -> None:
    mid = 0.5 * (ON_GROUND_THRESHOLD_M + NEAR_GROUND_THRESHOLD_M)
    assert foot_contact_state(mid) == "Near Ground"
    assert _phase_from_clearance(mid) == "unknown"
    assert filter_phase(mid) == "unknown"
    assert _phase_from_clearance(ON_GROUND_THRESHOLD_M - 0.001) == "stance"
    assert _phase_from_clearance(NEAR_GROUND_THRESHOLD_M) == "swing"


def test_hysteresis_can_hold_prior_phase_in_near_ground_band() -> None:
    mid = 0.5 * (ON_GROUND_THRESHOLD_M + NEAR_GROUND_THRESHOLD_M)
    assert _phase_from_clearance(mid, prev_phase="stance") == "stance"
    assert _phase_from_clearance(mid, prev_phase="swing") == "swing"
