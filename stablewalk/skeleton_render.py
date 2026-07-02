"""Shared skeleton drawing for matplotlib viewer and Tk GUI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from stablewalk.gait_dof import GAIT_DOF_LANDMARKS
from stablewalk.models.pose_data import Keypoint
from stablewalk.pose_estimation import LANDMARK_NAMES, PoseLandmarksConnections

MIN_VISIBILITY = 0.5
DISPLAY_MAX_WIDTH = 1280

SKELETON_COLOR = (0, 255, 136)  # BGR green
JOINT_COLOR = (71, 71, 255)  # BGR red-ish
DOF_COLOR = (255, 211, 0)  # BGR yellow


def load_frame_rgb(path: str | Path) -> np.ndarray:
    """Load a frame as RGB uint8; return placeholder if missing."""
    resolved = Path(path)
    if not resolved.is_file():
        return np.zeros((480, 640, 3), dtype=np.uint8)
    img = cv2.imread(str(resolved))
    if img is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    if w > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / w
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def render_frame_with_skeleton(
    image_rgb: np.ndarray,
    keypoints: list[Keypoint],
    *,
    show_skeleton: bool = True,
    highlight_dof: bool = True,
    min_visibility: float = MIN_VISIBILITY,
    gait_events: list[str] | None = None,
) -> np.ndarray:
    """Return RGB image with skeleton overlay drawn on a copy."""
    if image_rgb.size == 0:
        return image_rgb

    out = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = out.shape[:2]
    by_name = {kp.name: kp for kp in keypoints}
    index_to_kp = {
        i: by_name.get(LANDMARK_NAMES[i])
        for i in range(len(LANDMARK_NAMES))
    }

    if show_skeleton:
        for conn in PoseLandmarksConnections.POSE_LANDMARKS:
            start = index_to_kp.get(conn.start)
            end = index_to_kp.get(conn.end)
            if not start or not end:
                continue
            if start.visibility < min_visibility or end.visibility < min_visibility:
                continue
            pt1 = (int(start.x * w), int(start.y * h))
            pt2 = (int(end.x * w), int(end.y * h))
            cv2.line(out, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

    for kp in keypoints:
        if kp.name == "mid_hip" or kp.visibility < min_visibility:
            continue
        cx, cy = int(kp.x * w), int(kp.y * h)
        if highlight_dof and kp.name in GAIT_DOF_LANDMARKS:
            cv2.circle(out, (cx, cy), 8, DOF_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
        elif show_skeleton:
            cv2.circle(out, (cx, cy), 4, JOINT_COLOR, -1, cv2.LINE_AA)

    if gait_events:
        y_off = 28
        for event in gait_events[:4]:
            label = event.replace("_", " ").upper()
            color = (0, 220, 255) if "heel" in event else (255, 180, 0)
            cv2.putText(
                out,
                label,
                (12, y_off),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            y_off += 22

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def skeleton_line_segments(
    keypoints: list[Keypoint],
    min_visibility: float = MIN_VISIBILITY,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Normalized (x,y) segments for matplotlib plotting."""
    by_name = {kp.name: kp for kp in keypoints}
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        if conn.start >= len(LANDMARK_NAMES) or conn.end >= len(LANDMARK_NAMES):
            continue
        a = by_name.get(LANDMARK_NAMES[conn.start])
        b = by_name.get(LANDMARK_NAMES[conn.end])
        if not a or not b or a.visibility < min_visibility or b.visibility < min_visibility:
            continue
        segments.append(
            (
                (clamp01(a.x), clamp01(a.y)),
                (clamp01(b.x), clamp01(b.y)),
            )
        )
    return segments
