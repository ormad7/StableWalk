"""
Force-clear pipeline artifacts and release OpenCV captures between videos.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

# Serialize FFmpeg/OpenCV decode — concurrent VideoCapture use can crash libavcodec.
_video_decode_lock = threading.Lock()

# Track captures opened during validation (released on reset)
_active_captures: list[cv2.VideoCapture] = []


def video_decode_lock() -> threading.Lock:
    """Return the global lock for video decode / VideoCapture operations."""
    return _video_decode_lock


def register_capture(cap: cv2.VideoCapture | None) -> None:
    if cap is not None and cap.isOpened():
        _active_captures.append(cap)


def unregister_capture(cap: cv2.VideoCapture | None) -> None:
    if cap is None:
        return
    try:
        _active_captures.remove(cap)
    except ValueError:
        pass


def release_all_captures() -> None:
    """Release any VideoCapture instances still held."""
    global _active_captures
    with _video_decode_lock:
        for cap in _active_captures:
            try:
                if cap.isOpened():
                    cap.release()
            except Exception:
                pass
        _active_captures = []
    logger.debug("Released all video captures")


def clear_run_artifacts(
    frames_dir: Path | None,
    poses_path: Path,
    *,
    retries: int = 3,
) -> None:
    """
    Delete frames directory and pose JSON before a fresh run.

    Raises OSError if artifacts cannot be removed (no silent reuse).
    """
    release_all_captures()

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            if frames_dir is not None and frames_dir.is_dir():
                shutil.rmtree(frames_dir)
            if poses_path.is_file():
                poses_path.unlink()
            if frames_dir is not None:
                frames_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared run artifacts: %s", frames_dir.name)
            elif poses_path.is_file():
                logger.info("Cleared pose artifact: %s", poses_path.name)
            return
        except OSError as exc:
            last_err = exc
            time.sleep(0.15 * (attempt + 1))
    raise OSError(
        f"Could not clear previous run data at {frames_dir}. "
        "Close any program using those files and try again."
    ) from last_err
