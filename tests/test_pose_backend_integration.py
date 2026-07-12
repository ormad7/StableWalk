"""
Integration test: short synthetic video → pose extraction with auto backend.

Uses auto mode so CI passes without SMPL dependencies (falls back to MediaPipe).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from stablewalk.pose.backends.pipeline_runner import extract_pose_from_video


def _write_synthetic_walk_video(path: Path, *, frames: int = 12, fps: float = 30.0) -> None:
    """Minimal MP4 with a moving rectangle (person proxy)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter unavailable on this platform")
    try:
        for i in range(frames):
            frame = np.full((240, 320, 3), 220, dtype=np.uint8)
            # Moving vertical bar simulates walking subject
            x = 80 + i * 8
            cv2.rectangle(frame, (x, 40), (x + 40, 200), (60, 90, 140), -1)
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.integration
def test_auto_backend_video_extraction(tmp_path: Path) -> None:
    video = tmp_path / "synthetic_walk.mp4"
    _write_synthetic_walk_video(video, frames=15)

    result = extract_pose_from_video(
        video,
        backend_name="auto",
        max_frames=15,
        enrich=False,
        allow_fallback=True,
    )

    assert result.sequence.fps > 0
    assert len(result.sequence.frames) == 15
    assert result.unified_motion.frame_count == 15
    assert result.resolution.used in ("mediapipe", "smpl")

    if not result.resolution.fallback:
        assert result.resolution.used == "smpl"
    else:
        assert result.resolution.used == "mediapipe"
        assert result.resolution.fallback_reason

    # smpl_motion.npz must not exist for MediaPipe fallback
    if result.resolution.used == "mediapipe":
        from stablewalk.io.smpl_motion_export import maybe_export_smpl_motion

        assert maybe_export_smpl_motion(result.unified_motion, tmp_path) is None
