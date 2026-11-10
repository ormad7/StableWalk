"""
Planned observation channels for an optional gait simulator.

Values are produced today by the video pipeline. This module only names them
and records provenance. It does not allocate a Gym observation space or step
a physics model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationChannel:
    """One simulator observation, mapped to an existing StableWalk artifact."""

    key: str
    description: str
    source_module: str
    units: str
    notes: str


# Keys requested for requirement 5 — keep in sync with docs/erik_requirement_5_plan.md §5.
OBSERVATION_CHANNELS: tuple[ObservationChannel, ...] = (
    ObservationChannel(
        key="joint_positions",
        description="Canonical 3D joint positions",
        source_module="stablewalk.models.gait_motion.GaitMotionRecording",
        units="m",
        notes="sw_canonical_y_up, pelvis origin; missing joints are NaN, never (0,0,0).",
    ),
    ObservationChannel(
        key="joint_velocities",
        description="Joint linear velocities",
        source_module="stablewalk.models.gait_motion.JointTimeSeries",
        units="m/s",
        notes="Same frame as positions.",
    ),
    ObservationChannel(
        key="joint_angles",
        description="Joint / DOF angles and angular velocities",
        source_module="stablewalk.models.gait_motion.JointTimeSeries",
        units="rad, rad/s",
        notes="dof_angles / dof_velocities; retarget before robot PD targets.",
    ),
    ObservationChannel(
        key="center_of_mass_estimate",
        description="Pelvis/hip CoM proxy used in virtual GRF",
        source_module="stablewalk.analysis.virtual_grf",
        units="m",
        notes="Estimate from kinematics, not measured inertial CoM.",
    ),
    ObservationChannel(
        key="ground_contact_state",
        description="Left/right foot contact masks",
        source_module="stablewalk.analysis.gait_cycle_analysis",
        units="binary",
        notes="Timing only — not ground reaction force.",
    ),
    ObservationChannel(
        key="foot_clearance",
        description="Swing-foot height / clearance features",
        source_module="stablewalk.analysis.gait_feature_analysis",
        units="m",
        notes="Monocular heuristic clearance.",
    ),
    ObservationChannel(
        key="bilateral_asymmetry",
        description="Robinson SI / ASI / phase-normalized L–R trajectories",
        source_module="stablewalk.analysis.asymmetry3d",
        units="index / m",
        notes="Research asymmetry features, not a diagnosis.",
    ),
    ObservationChannel(
        key="spectral_temporal_gait_features",
        description="Spectral A–H plus cadence, stride, cycle timing",
        source_module="stablewalk.analysis.spectral",
        units="Hz, s, ratio",
        notes="Requirement 3 feature vector on the same recording.",
    ),
)

OBSERVATION_KEYS: tuple[str, ...] = tuple(ch.key for ch in OBSERVATION_CHANNELS)


def observation_channel(key: str) -> ObservationChannel:
    for ch in OBSERVATION_CHANNELS:
        if ch.key == key:
            return ch
    raise KeyError(key)
