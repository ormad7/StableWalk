"""Adapters between external motion formats and ``GaitMotionRecording``.

Exports are lazy so importing ``pose_adapter`` for GUI session load does not
pull MediaPipe / backend registry through this package init.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "canonical_from_opensim",
    "motion_to_opensim_table",
    "opensim_column_name",
    "pose_frame_to_snapshot",
    "pose_sequence_to_gait_motion",
    "human_motion_sequence_to_gait_motion",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "canonical_from_opensim": (
        "stablewalk.adapters.opensim_schema",
        "canonical_from_opensim",
    ),
    "motion_to_opensim_table": (
        "stablewalk.adapters.opensim_schema",
        "motion_to_opensim_table",
    ),
    "opensim_column_name": (
        "stablewalk.adapters.opensim_schema",
        "opensim_column_name",
    ),
    "pose_frame_to_snapshot": (
        "stablewalk.adapters.pose_adapter",
        "pose_frame_to_snapshot",
    ),
    "pose_sequence_to_gait_motion": (
        "stablewalk.adapters.pose_adapter",
        "pose_sequence_to_gait_motion",
    ),
    "human_motion_sequence_to_gait_motion": (
        "stablewalk.pose.backends.canonical",
        "human_motion_sequence_to_gait_motion",
    ),
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

# Biomechanics / OpenSim service layer lives in ``stablewalk.biomechanics`` to
# avoid circular imports (biomechanics.service imports opensim_schema directly).
