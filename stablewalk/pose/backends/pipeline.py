"""
High-level video processing via configured pose backend.

The main StableWalk app continues to use ``PoseEstimator`` directly.
This module is for research scripts and future backend switching.
"""

from __future__ import annotations

from pathlib import Path

from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.backends.canonical import human_motion_sequence_to_pose_sequence
from stablewalk.pose.backends.registry import create_pose_backend
from stablewalk.pose.enrichment import enrich_pose_sequence


def process_video_with_configured_backend(
    video_path: str | Path,
    *,
    max_frames: int | None = None,
    enrich: bool = True,
    backend_name: str | None = None,
    allow_fallback: bool | None = None,
) -> PoseSequence:
    """
    Run the configured ``HumanMotionBackend`` and return legacy ``PoseSequence``.

    Default backend is MediaPipe — behaviour matches existing pipeline when
    ``POSE_BACKEND=mediapipe``.
    """
    backend = create_pose_backend(backend_name, allow_fallback=allow_fallback)
    try:
        hm_sequence = backend.process_video(str(video_path), max_frames=max_frames)
        sequence = human_motion_sequence_to_pose_sequence(hm_sequence)
        if enrich:
            enrich_pose_sequence(sequence)
        return sequence
    finally:
        if hasattr(backend, "close"):
            backend.close()
