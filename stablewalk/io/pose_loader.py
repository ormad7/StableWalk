"""
Load pose JSON and list detected frames (re-exports legacy visualization helpers).
"""

from __future__ import annotations

from pathlib import Path

from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.visualization import detected_frame_indices, load_pose_sequence

__all__ = [
    "detected_frame_indices",
    "load_pose_sequence",
    "sequence_needs_enrichment",
]


def sequence_needs_enrichment(sequence: PoseSequence) -> bool:
    """True if detected frames lack velocity or 3D skeleton data."""
    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        if not frame.velocity_scalar and not frame.velocities:
            return True
        sk = frame.skeleton_3d or {}
        if not sk.get("joints"):
            return True
    return False
