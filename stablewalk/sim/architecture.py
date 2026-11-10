"""
Optional gait-simulator architecture (contract only).

Does not construct MuJoCo, Isaac Lab, or Gym environments. Engine defaults to
``none``. Callers that need a live env must add it in a separate install after
requirements 2–4 are validated (docs/erik_requirement_5_plan.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from stablewalk.sim.observation_spec import OBSERVATION_KEYS
from stablewalk.sim.status import (
    PHYSICS_ENGINE_OPERATIONAL,
    RL_ENVIRONMENT_OPERATIONAL,
    SIMULATOR_IMPLEMENTED,
    STATUS_NOTE,
)

EngineName = Literal["none", "mujoco", "isaac_lab"]


@dataclass(frozen=True)
class OptionalGaitSimulatorSpec:
    """Sidecar design: observations in, research rewards out, pipeline unchanged."""

    engine: EngineName = "none"
    accepts_gait_motion_recording: bool = True
    observation_keys: tuple[str, ...] = OBSERVATION_KEYS
    replaces_video_pipeline: bool = False
    emits_clinical_diagnosis: bool = False
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            STATUS_NOTE,
            "Keep MediaPipe → GaitMotionRecording as the source of truth.",
            "OpenSim remains export/IK; it is not the RL physics engine.",
        )
    )

    def is_operational(self) -> bool:
        if self.engine == "none":
            return False
        return PHYSICS_ENGINE_OPERATIONAL and RL_ENVIRONMENT_OPERATIONAL


def default_simulator_spec() -> OptionalGaitSimulatorSpec:
    return OptionalGaitSimulatorSpec()


def architecture_summary() -> dict[str, object]:
    spec = default_simulator_spec()
    return {
        "simulator_implemented": SIMULATOR_IMPLEMENTED,
        "engine": spec.engine,
        "operational": spec.is_operational(),
        "observation_keys": list(spec.observation_keys),
        "replaces_video_pipeline": spec.replaces_video_pipeline,
        "emits_clinical_diagnosis": spec.emits_clinical_diagnosis,
        "notes": list(spec.notes),
    }
