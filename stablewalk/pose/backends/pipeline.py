"""
High-level video processing via configured pose backend.

Delegates to ``pipeline_runner`` for mediapipe / smpl / auto resolution.
"""

from __future__ import annotations

from pathlib import Path

from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.backends.pipeline_runner import extract_pose_from_video


def process_video_with_configured_backend(
    video_path: str | Path,
    *,
    max_frames: int | None = None,
    enrich: bool = True,
    backend_name: str | None = None,
    allow_fallback: bool | None = None,
) -> PoseSequence:
    """Run the configured backend and return legacy ``PoseSequence``."""
    result = extract_pose_from_video(
        video_path,
        backend_name=backend_name,
        max_frames=max_frames,
        enrich=enrich,
        allow_fallback=allow_fallback,
    )
    return result.sequence
