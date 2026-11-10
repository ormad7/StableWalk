"""
Operational flags for the optional gait simulator.

Requirement 5 is **planned**. These constants must stay False until a physics
engine actually steps and a documented runner exists. Do not set them True
because AMP NPZ export or contact-sync formulas exist.
"""

from __future__ import annotations

# Entire optional sidecar (observation adapter + env + trainer).
SIMULATOR_IMPLEMENTED = False

# MuJoCo / Isaac / Bullet (or similar) can step contacts in this process.
PHYSICS_ENGINE_OPERATIONAL = False

# gymnasium-style reset/step with a humanoid task.
RL_ENVIRONMENT_OPERATIONAL = False

# A trained policy checkpoint is loaded and producing actions.
RL_POLICY_TRAINED = False

STATUS_NOTE = (
    "StableWalk does not run a physics engine or reinforcement-learning "
    "environment in this install. Real-to-Sim writes reference clips and "
    "contact-sync scores; that is not simulation or training. See "
    "docs/erik_requirement_5_plan.md."
)


def simulator_status_dict() -> dict[str, bool | str]:
    return {
        "simulator_implemented": SIMULATOR_IMPLEMENTED,
        "physics_engine_operational": PHYSICS_ENGINE_OPERATIONAL,
        "rl_environment_operational": RL_ENVIRONMENT_OPERATIONAL,
        "rl_policy_trained": RL_POLICY_TRAINED,
        "note": STATUS_NOTE,
    }
