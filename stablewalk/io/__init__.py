"""I/O helpers for pose sequences and exports (lazy package exports)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "detected_frame_indices",
    "playback_frame_indices",
    "load_pose_sequence",
    "sequence_needs_enrichment",
]

_LAZY = {
    "detected_frame_indices": ("stablewalk.io.pose_loader", "detected_frame_indices"),
    "playback_frame_indices": ("stablewalk.io.pose_loader", "playback_frame_indices"),
    "load_pose_sequence": ("stablewalk.io.pose_loader", "load_pose_sequence"),
    "sequence_needs_enrichment": (
        "stablewalk.io.pose_loader",
        "sequence_needs_enrichment",
    ),
}


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod_name, attr = target
    import importlib

    mod = importlib.import_module(mod_name)
    value = getattr(mod, attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
