"""Adapters between external motion formats and ``GaitMotionRecording``."""

from stablewalk.adapters.opensim_schema import (
    canonical_from_opensim,
    motion_to_opensim_table,
    opensim_column_name,
)
from stablewalk.adapters.pose_adapter import pose_frame_to_snapshot, pose_sequence_to_gait_motion

__all__ = [
    "canonical_from_opensim",
    "motion_to_opensim_table",
    "opensim_column_name",
    "pose_frame_to_snapshot",
    "pose_sequence_to_gait_motion",
]

# Biomechanics / OpenSim service layer lives in ``stablewalk.biomechanics`` to
# avoid circular imports (biomechanics.service imports opensim_schema directly).
