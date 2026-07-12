"""
Pluggable pose / human mesh recovery backends for StableWalk.

Default: ``MediaPipePoseBackend`` (``POSE_BACKEND=mediapipe``).

Experimental optional backends (ROMP, HybrIK, WHAM) are adapters only —
they are not installed with the main application.
"""

from stablewalk.pose.backends.base import BackendUnavailableError, HumanMotionBackend
from stablewalk.pose.backends.canonical import (
    CANONICAL_GAIT_JOINTS,
    human_motion_sequence_to_gait_motion,
    human_motion_sequence_to_pose_sequence,
)
from stablewalk.pose.backends.environment import HMR_COMPATIBILITY_NOTES, inspect_runtime_environment
from stablewalk.pose.backends.registry import (
    BACKEND_REGISTRY,
    SUPPORTED_BACKENDS,
    create_pose_backend,
    list_backend_diagnostics,
    unavailable_message,
)
from stablewalk.pose.backends.types import HumanMotionFrame, HumanMotionSequence

__all__ = [
    "BackendUnavailableError",
    "HumanMotionBackend",
    "HumanMotionFrame",
    "HumanMotionSequence",
    "CANONICAL_GAIT_JOINTS",
    "BACKEND_REGISTRY",
    "SUPPORTED_BACKENDS",
    "create_pose_backend",
    "list_backend_diagnostics",
    "unavailable_message",
    "inspect_runtime_environment",
    "HMR_COMPATIBILITY_NOTES",
    "human_motion_sequence_to_gait_motion",
    "human_motion_sequence_to_pose_sequence",
]
