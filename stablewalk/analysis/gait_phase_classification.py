"""
Central gait-phase classification from smoothed bilateral contact states.

Gait phase is derived only from the final left/right contact mask plus session
contact confidence. GUI layers must consume these values — never re-derive phase
logic in dashboard widgets.
"""

from __future__ import annotations

import logging
from typing import Literal

from stablewalk.analysis.ground_reference import NEAR_GROUND_THRESHOLD_M

logger = logging.getLogger(__name__)

GaitPhaseName = Literal[
    "LEFT_STANCE",
    "RIGHT_STANCE",
    "DOUBLE_SUPPORT",
    "FLIGHT",
    "UNCERTAIN",
]

# Session contact confidence below this → bilateral swing reads as UNCERTAIN.
CONTACT_CONFIDENCE_LOW_THRESHOLD = 0.48

# Both feet must exceed this clearance (body-scale m) to classify FLIGHT.
FLIGHT_CLEARANCE_THRESHOLD_M = NEAR_GROUND_THRESHOLD_M

_PHASE_DISPLAY: dict[str, str] = {
    "LEFT_STANCE": "LEFT STANCE",
    "RIGHT_STANCE": "RIGHT STANCE",
    "DOUBLE_SUPPORT": "DOUBLE SUPPORT",
    "FLIGHT": "FLIGHT",
    "UNCERTAIN": "UNCERTAIN",
    # Legacy alias from older pipeline builds
    "FLIGHT_OR_UNCERTAIN": "UNCERTAIN",
}


def contact_to_display_state(contact: int | bool | None) -> str:
    """Map binary contact mask to Overview contact label."""
    if contact is None:
        return "—"
    return "CONTACT" if bool(contact) else "SWING"


def both_feet_clearly_airborne(
    left_clearance_m: float | None,
    right_clearance_m: float | None,
    *,
    threshold_m: float = FLIGHT_CLEARANCE_THRESHOLD_M,
) -> bool:
    """True when both feet are clearly above the floor reference."""
    if left_clearance_m is None or right_clearance_m is None:
        return False
    return left_clearance_m >= threshold_m and right_clearance_m >= threshold_m


def classify_gait_phase_from_contacts(
    left_contact: int | bool,
    right_contact: int | bool,
    *,
    contact_confidence: float | None = None,
    confidence_tier: str | None = None,
    left_foot_clearance_m: float | None = None,
    right_foot_clearance_m: float | None = None,
) -> GaitPhaseName:
    """
    Derive gait phase from the final smoothed contact mask.

    Rules:
        L=1, R=0 → LEFT_STANCE
        L=0, R=1 → RIGHT_STANCE
        L=1, R=1 → DOUBLE_SUPPORT
        L=0, R=0 → FLIGHT if confidence OK and both feet clearly airborne,
                    else UNCERTAIN
    """
    left = bool(left_contact)
    right = bool(right_contact)

    if left and right:
        return "DOUBLE_SUPPORT"
    if left and not right:
        return "LEFT_STANCE"
    if right and not left:
        return "RIGHT_STANCE"

    low_confidence = confidence_tier == "LOW_CONFIDENCE" or (
        contact_confidence is not None
        and contact_confidence < CONTACT_CONFIDENCE_LOW_THRESHOLD
    )
    if low_confidence:
        return "UNCERTAIN"

    if both_feet_clearly_airborne(left_foot_clearance_m, right_foot_clearance_m):
        return "FLIGHT"

    return "UNCERTAIN"


def format_gait_phase_display(phase: str | None) -> str:
    """Human-readable gait phase for dashboard labels."""
    if not phase:
        return "—"
    key = phase.strip().upper()
    return _PHASE_DISPLAY.get(key, key.replace("_", " "))


def expected_phase_for_contacts(
    left_contact: int | bool,
    right_contact: int | bool,
) -> GaitPhaseName | Literal["FLIGHT_OR_UNCERTAIN"]:
    """Deterministic phase for single-support and double-support (no flight split)."""
    left = bool(left_contact)
    right = bool(right_contact)
    if left and right:
        return "DOUBLE_SUPPORT"
    if left and not right:
        return "LEFT_STANCE"
    if right and not left:
        return "RIGHT_STANCE"
    return "FLIGHT_OR_UNCERTAIN"


def validate_phase_contact_consistency(
    left_contact: int | bool,
    right_contact: int | bool,
    phase: str | None,
) -> tuple[bool, str]:
    """
    Return whether ``phase`` is consistent with contact states.

    Bilateral swing allows FLIGHT or UNCERTAIN but never blank/unknown.
    """
    if not phase or phase.strip() in ("—", "", "UNKNOWN"):
        return False, "missing_phase"

    left = bool(left_contact)
    right = bool(right_contact)
    phase_up = phase.strip().upper()

    if left and right:
        return phase_up == "DOUBLE_SUPPORT", "DOUBLE_SUPPORT"
    if left and not right:
        return phase_up == "LEFT_STANCE", "LEFT_STANCE"
    if right and not left:
        return phase_up == "RIGHT_STANCE", "RIGHT_STANCE"

    # Both feet off — must be FLIGHT, UNCERTAIN, or legacy FLIGHT_OR_UNCERTAIN.
    if phase_up in ("FLIGHT", "UNCERTAIN", "FLIGHT_OR_UNCERTAIN"):
        return True, phase_up
    return False, "FLIGHT or UNCERTAIN"


def log_phase_consistency_warning(
    *,
    frame_index: int,
    left_contact: int | bool,
    right_contact: int | bool,
    contact_confidence: float | None,
    phase: str | None,
    displayed_phase: str | None = None,
) -> None:
    """Emit a debug warning when displayed phase disagrees with contact mask."""
    ok, expected = validate_phase_contact_consistency(
        left_contact, right_contact, phase
    )
    if not ok:
        logger.warning(
            "Gait phase inconsistency frame=%s left=%s right=%s "
            "confidence=%.2f derived_phase=%s expected=%s displayed=%s",
            frame_index,
            int(bool(left_contact)),
            int(bool(right_contact)),
            contact_confidence if contact_confidence is not None else -1.0,
            phase,
            expected,
            displayed_phase,
        )
