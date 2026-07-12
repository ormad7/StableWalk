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
from stablewalk.pose.backends.pipeline_runner import (
    BackendResolution,
    PoseExtractionResult,
    extract_pose_from_frames_dir,
    extract_pose_from_video,
    extract_pose_video_with_frame_cache,
    get_backend_diagnostics,
    normalize_backend_mode,
    resolve_pose_backend,
)
from stablewalk.pose.backends.smpl_backend import SMPLPoseBackend
from stablewalk.pose.backends.smpl_validation import validate_smpl_assets
from stablewalk.pose.backends.unified_motion import (
    UnifiedHumanMotion,
    human_motion_sequence_to_unified,
)
from stablewalk.pose.backends.registry import (
    BACKEND_REGISTRY,
    PRIMARY_BACKENDS,
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
    "UnifiedHumanMotion",
    "BackendResolution",
    "PoseExtractionResult",
    "CANONICAL_GAIT_JOINTS",
    "BACKEND_REGISTRY",
    "PRIMARY_BACKENDS",
    "SUPPORTED_BACKENDS",
    "SMPLPoseBackend",
    "create_pose_backend",
    "list_backend_diagnostics",
    "unavailable_message",
    "validate_smpl_assets",
    "normalize_backend_mode",
    "resolve_pose_backend",
    "extract_pose_from_video",
    "extract_pose_from_frames_dir",
    "get_backend_diagnostics",
    "inspect_runtime_environment",
    "HMR_COMPATIBILITY_NOTES",
    "human_motion_sequence_to_gait_motion",
    "human_motion_sequence_to_pose_sequence",
    "human_motion_sequence_to_unified",
]
