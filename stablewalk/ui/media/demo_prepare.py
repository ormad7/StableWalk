"""Prepare demo videos: trim to motion segments and re-encode for StableWalk."""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import cv2

from stablewalk.pose.estimation import PoseEstimator

logger = logging.getLogger(__name__)


def _write_segment(
    source: Path,
    dest: Path,
    *,
    start_frame: int,
    max_frames: int,
) -> bool:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part.mp4")
    writer = cv2.VideoWriter(
        str(tmp),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    while written < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    writer.release()
    cap.release()
    if written < 10:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return dest.is_file()


def find_first_gait_frame(
    source: Path,
    *,
    scan_frames: int = 250,
    min_consecutive: int = 3,
) -> int | None:
    """Return the first frame index with consecutive gait-detected frames."""
    with PoseEstimator(video_mode=True) as estimator:
        sequence, _ = estimator.process_video_with_frame_cache(
            str(source),
            None,
            max_frames=scan_frames,
        )
    streak = 0
    for index, frame in enumerate(sequence.frames):
        if frame.detected:
            streak += 1
            if streak >= min_consecutive:
                return index - min_consecutive + 1
        else:
            streak = 0
    return None


def find_best_walking_segment(
    source: Path,
    *,
    output_frames: int = 120,
    scan_frames: int | None = None,
    window_step: int = 10,
) -> int | None:
    """
    Return the start frame of the highest-motion continuous walking window.

    Uses bilateral ankle/knee vertical motion while pose is detected.
    """
    import numpy as np

    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    limit = scan_frames if scan_frames is not None else total
    if limit <= 0:
        limit = 1400

    with PoseEstimator(video_mode=True) as estimator:
        sequence, _ = estimator.process_video_with_frame_cache(
            str(source),
            None,
            max_frames=limit,
        )

    track_names = ("right_ankle", "left_ankle", "right_knee", "left_knee")
    tracks: dict[str, list[float | None]] = {name: [] for name in track_names}
    detected: list[bool] = []
    for frame in sequence.frames:
        detected.append(bool(frame.detected))
        present = {
            kp.name: kp.y
            for kp in frame.keypoints
            if kp.name in track_names and kp.visibility >= 0.4
        }
        for name in track_names:
            tracks[name].append(present.get(name))

    n = len(sequence.frames)
    if n < output_frames:
        return 0 if any(detected) else None

    max_start = max(0, n - output_frames - max(1, n // 10))

    best_start = 0
    best_score = -1.0
    for start in range(0, max_start + 1, window_step):
        end = start + output_frames
        window_detected = detected[start:end]
        detect_rate = sum(window_detected) / len(window_detected)
        if detect_rate < 0.9:
            continue
        stds: list[float] = []
        for name in track_names:
            vals = [v for v in tracks[name][start:end] if v is not None]
            if len(vals) >= max(8, output_frames // 4):
                stds.append(float(np.std(vals)))
        if len(stds) < 2:
            continue
        motion = float(np.mean(stds))
        score = motion * detect_rate
        if score > best_score:
            best_score = score
            best_start = start

    if best_score < 0:
        return find_first_gait_frame(source, scan_frames=limit)
    return best_start


def trim_to_gait_segment(
    source: Path,
    dest: Path,
    *,
    start_frame: int | None = None,
    output_frames: int = 120,
    scan_frames: int = 250,
    prefer_best_motion: bool = False,
) -> tuple[bool, int]:
    """
    Trim ``source`` to a continuous gait segment and save to ``dest``.

    Returns ``(success, start_frame_used)``.
    """
    if start_frame is None:
        if prefer_best_motion:
            start_frame = find_best_walking_segment(
                source,
                output_frames=output_frames,
                scan_frames=scan_frames if scan_frames > 250 else None,
            )
        else:
            start_frame = find_first_gait_frame(source, scan_frames=scan_frames)
    if start_frame is None:
        logger.warning("No gait segment found in %s", source)
        return False, -1
    ok = _write_segment(source, dest, start_frame=start_frame, max_frames=output_frames)
    return ok, start_frame


def prepare_pexels_demo(
    raw_path: Path,
    dest_path: Path,
    *,
    start_frame: int | None = None,
    output_frames: int = 120,
) -> tuple[bool, str]:
    """
    Trim a downloaded Pexels clip to its usable motion segment.

    Falls back to copying the raw file when trimming fails.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trimmed = Path(tmpdir) / "trimmed.mp4"
        ok, used_start = trim_to_gait_segment(
            raw_path,
            trimmed,
            start_frame=start_frame,
            output_frames=output_frames,
            prefer_best_motion=start_frame is None,
        )
        if ok:
            shutil.copy2(trimmed, dest_path)
            return True, f"trimmed from frame {used_start}, {output_frames} frames"
        shutil.copy2(raw_path, dest_path)
        return False, "trim failed; kept full download"
