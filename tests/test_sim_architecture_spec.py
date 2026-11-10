"""Requirement 5: architecture contracts exist; no simulator is implemented."""

from __future__ import annotations

import importlib.util

from stablewalk.analysis.isaac_lab_integration import (
    ISAAC_LAB_AVAILABLE,
    check_isaac_lab_available,
)
from stablewalk.analysis.virtual_grf import (
    LearnedForceEstimator,
    PhysicsSimulationForceEstimator,
    build_virtual_force_input,
)
from stablewalk.sim.architecture import default_simulator_spec
from stablewalk.sim.objectives import ClaimLevel, claim_allowed, contact_sync_claim_level
from stablewalk.sim.observation_spec import OBSERVATION_KEYS, observation_channel
from stablewalk.sim.status import (
    PHYSICS_ENGINE_OPERATIONAL,
    RL_ENVIRONMENT_OPERATIONAL,
    RL_POLICY_TRAINED,
    SIMULATOR_IMPLEMENTED,
)
from tests.test_virtual_grf import _cycles, _recording


def test_simulator_flags_are_not_implemented() -> None:
    assert SIMULATOR_IMPLEMENTED is False
    assert PHYSICS_ENGINE_OPERATIONAL is False
    assert RL_ENVIRONMENT_OPERATIONAL is False
    assert RL_POLICY_TRAINED is False


def test_isaac_lab_not_operational_in_this_environment() -> None:
    assert ISAAC_LAB_AVAILABLE is False
    ok, message = check_isaac_lab_available()
    assert ok is False
    assert "not installed" in message.lower() or "Isaac Lab" in message


def test_rl_and_physics_packages_absent() -> None:
    for name in (
        "isaaclab",
        "isaacsim",
        "mujoco",
        "gymnasium",
        "pybullet",
        "stable_baselines3",
        "skrl",
        "rsl_rl",
    ):
        assert importlib.util.find_spec(name) is None, f"{name} must not be required yet"


def test_physics_and_learned_vgrf_placeholders_unavailable() -> None:
    rec = _recording()
    data = build_virtual_force_input(rec, _cycles(rec))
    phys = PhysicsSimulationForceEstimator().estimate(data)
    learned = LearnedForceEstimator().estimate(data)
    assert phys.available is False
    assert learned.available is False


def test_observation_keys_cover_requirement_5_list() -> None:
    required = {
        "joint_positions",
        "joint_velocities",
        "joint_angles",
        "center_of_mass_estimate",
        "ground_contact_state",
        "foot_clearance",
        "bilateral_asymmetry",
        "spectral_temporal_gait_features",
    }
    assert required <= set(OBSERVATION_KEYS)
    contact = observation_channel("ground_contact_state")
    assert "not ground reaction force" in contact.notes.lower()


def test_default_spec_does_not_replace_pipeline_or_diagnose() -> None:
    spec = default_simulator_spec()
    assert spec.engine == "none"
    assert spec.is_operational() is False
    assert spec.replaces_video_pipeline is False
    assert spec.emits_clinical_diagnosis is False
    assert spec.accepts_gait_motion_recording is True


def test_claim_levels_are_distinct_and_diagnosis_blocked() -> None:
    assert claim_allowed(ClaimLevel.SIMULATION_REWARD) is True
    assert claim_allowed(ClaimLevel.PREDICTED_INSTABILITY) is True
    assert claim_allowed(ClaimLevel.CLINICAL_DIAGNOSIS) is False
    assert (
        claim_allowed(
            ClaimLevel.CLINICAL_DIAGNOSIS,
            has_patient_level_labeled_validation=False,
        )
        is False
    )
    assert contact_sync_claim_level() is ClaimLevel.SIMULATION_REWARD
    assert ClaimLevel.SIMULATION_REWARD != ClaimLevel.PREDICTED_INSTABILITY
    assert ClaimLevel.PREDICTED_INSTABILITY != ClaimLevel.CLINICAL_DIAGNOSIS
