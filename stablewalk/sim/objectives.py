"""
Claim levels for simulator outputs.

Simulation reward, predicted instability, and clinical diagnosis are different
objects. The planned environment may emit the first; may later emit the second
as an unvalidated research index; must never emit the third.
"""

from __future__ import annotations

from enum import Enum


class ClaimLevel(str, Enum):
    SIMULATION_REWARD = "simulation_reward"
    PREDICTED_INSTABILITY = "predicted_instability"
    CLINICAL_DIAGNOSIS = "clinical_diagnosis"


CLAIM_DEFINITIONS: dict[ClaimLevel, str] = {
    ClaimLevel.SIMULATION_REWARD: (
        "Scalar used to train or score a controller (tracking error, "
        "contact-sync, energy). Not a clinical finding."
    ),
    ClaimLevel.PREDICTED_INSTABILITY: (
        "Unvalidated research hypothesis score from gait features or simulated "
        "perturbations. Requires explicit kind=predicted_instability labeling."
    ),
    ClaimLevel.CLINICAL_DIAGNOSIS: (
        "Patient-level diagnostic statement. Not an output of StableWalk or "
        "the planned simulator without a separate labeled clinical study."
    ),
}


def claim_allowed(
    level: ClaimLevel,
    *,
    has_patient_level_labeled_validation: bool = False,
) -> bool:
    """
    Whether this architecture may emit ``level``.

    Clinical diagnosis stays False unless a future study supplies patient-level
    labels **and** this function is revisited in that study's code — even then
    the desktop GUI must not present a diagnosis badge by default.
    """
    if level is ClaimLevel.SIMULATION_REWARD:
        return True
    if level is ClaimLevel.PREDICTED_INSTABILITY:
        return True  # allowed as labeled research output only
    if level is ClaimLevel.CLINICAL_DIAGNOSIS:
        return bool(has_patient_level_labeled_validation)
    return False


def contact_sync_claim_level() -> ClaimLevel:
    """Contact-sync mean_reward is always a simulation reward."""
    return ClaimLevel.SIMULATION_REWARD
