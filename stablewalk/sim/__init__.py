"""
Optional gait simulator — architecture contracts only.

Requirement 5 is not implemented. Importing this package does not create a
physics or RL environment.
"""

from stablewalk.sim.architecture import OptionalGaitSimulatorSpec, architecture_summary
from stablewalk.sim.objectives import ClaimLevel, claim_allowed
from stablewalk.sim.observation_spec import OBSERVATION_KEYS
from stablewalk.sim.status import (
    PHYSICS_ENGINE_OPERATIONAL,
    RL_ENVIRONMENT_OPERATIONAL,
    RL_POLICY_TRAINED,
    SIMULATOR_IMPLEMENTED,
    STATUS_NOTE,
)

__all__ = [
    "ClaimLevel",
    "OBSERVATION_KEYS",
    "OptionalGaitSimulatorSpec",
    "PHYSICS_ENGINE_OPERATIONAL",
    "RL_ENVIRONMENT_OPERATIONAL",
    "RL_POLICY_TRAINED",
    "SIMULATOR_IMPLEMENTED",
    "STATUS_NOTE",
    "architecture_summary",
    "claim_allowed",
]
