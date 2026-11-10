"""
Pluggable pose / human mesh recovery backends for StableWalk.

Default: ``MediaPipePoseBackend`` (``POSE_BACKEND=mediapipe``).

Experimental optional backends (ROMP, HybrIK, WHAM) are adapters only —
they are not installed with the main application.

Exports are lazy so GUI startup can import ``canonical`` / types without
paying for MediaPipe or optional HMR stacks.
"""

from __future__ import annotations

from typing import Any

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
    "extract_pose_video_with_frame_cache",
    "get_backend_diagnostics",
    "inspect_runtime_environment",
    "HMR_COMPATIBILITY_NOTES",
    "human_motion_sequence_to_gait_motion",
    "human_motion_sequence_to_pose_sequence",
    "human_motion_sequence_to_unified",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "BackendUnavailableError": ("stablewalk.pose.backends.base", "BackendUnavailableError"),
    "HumanMotionBackend": ("stablewalk.pose.backends.base", "HumanMotionBackend"),
    "HumanMotionFrame": ("stablewalk.pose.backends.types", "HumanMotionFrame"),
    "HumanMotionSequence": ("stablewalk.pose.backends.types", "HumanMotionSequence"),
    "UnifiedHumanMotion": ("stablewalk.pose.backends.unified_motion", "UnifiedHumanMotion"),
    "human_motion_sequence_to_unified": (
        "stablewalk.pose.backends.unified_motion",
        "human_motion_sequence_to_unified",
    ),
    "CANONICAL_GAIT_JOINTS": ("stablewalk.pose.backends.canonical", "CANONICAL_GAIT_JOINTS"),
    "human_motion_sequence_to_gait_motion": (
        "stablewalk.pose.backends.canonical",
        "human_motion_sequence_to_gait_motion",
    ),
    "human_motion_sequence_to_pose_sequence": (
        "stablewalk.pose.backends.canonical",
        "human_motion_sequence_to_pose_sequence",
    ),
    "HMR_COMPATIBILITY_NOTES": (
        "stablewalk.pose.backends.environment",
        "HMR_COMPATIBILITY_NOTES",
    ),
    "inspect_runtime_environment": (
        "stablewalk.pose.backends.environment",
        "inspect_runtime_environment",
    ),
    "BackendResolution": ("stablewalk.pose.backends.pipeline_runner", "BackendResolution"),
    "PoseExtractionResult": ("stablewalk.pose.backends.pipeline_runner", "PoseExtractionResult"),
    "normalize_backend_mode": (
        "stablewalk.pose.backends.pipeline_runner",
        "normalize_backend_mode",
    ),
    "resolve_pose_backend": (
        "stablewalk.pose.backends.pipeline_runner",
        "resolve_pose_backend",
    ),
    "extract_pose_from_video": (
        "stablewalk.pose.backends.pipeline_runner",
        "extract_pose_from_video",
    ),
    "extract_pose_from_frames_dir": (
        "stablewalk.pose.backends.pipeline_runner",
        "extract_pose_from_frames_dir",
    ),
    "extract_pose_video_with_frame_cache": (
        "stablewalk.pose.backends.pipeline_runner",
        "extract_pose_video_with_frame_cache",
    ),
    "get_backend_diagnostics": (
        "stablewalk.pose.backends.pipeline_runner",
        "get_backend_diagnostics",
    ),
    "SMPLPoseBackend": ("stablewalk.pose.backends.smpl_backend", "SMPLPoseBackend"),
    "validate_smpl_assets": (
        "stablewalk.pose.backends.smpl_validation",
        "validate_smpl_assets",
    ),
    "BACKEND_REGISTRY": ("stablewalk.pose.backends.registry", "BACKEND_REGISTRY"),
    "PRIMARY_BACKENDS": ("stablewalk.pose.backends.registry", "PRIMARY_BACKENDS"),
    "SUPPORTED_BACKENDS": ("stablewalk.pose.backends.registry", "SUPPORTED_BACKENDS"),
    "create_pose_backend": ("stablewalk.pose.backends.registry", "create_pose_backend"),
    "list_backend_diagnostics": (
        "stablewalk.pose.backends.registry",
        "list_backend_diagnostics",
    ),
    "unavailable_message": ("stablewalk.pose.backends.registry", "unavailable_message"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    from importlib import import_module

    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
