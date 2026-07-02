"""Shared skeleton drawing and frame loading for GUI, overlays, and exports."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib

from stablewalk.pose.dof import GAIT_DOF_LANDMARKS
from stablewalk.models.pose_data import Keypoint

LANDMARK_NAMES = [e.name.lower() for e in pose_landmarker_lib.PoseLandmark]
PoseLandmarksConnections = pose_landmarker_lib.PoseLandmarksConnections

MIN_VISIBILITY = 0.5
DISPLAY_MAX_WIDTH = 1280

SKELETON_COLOR = (0, 255, 136)  # BGR green
JOINT_COLOR = (71, 71, 255)  # BGR red-ish
DOF_COLOR = (255, 211, 0)  # BGR yellow


class FrameRgbCache:
    """LRU cache for decoded frame images (speeds up GUI playback)."""

    def __init__(self, max_entries: int = 128) -> None:
        self._max = max(16, max_entries)
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()

    def clear(self) -> None:
        self._store.clear()

    def get_rgb(self, path: str | Path) -> np.ndarray:
        key = str(Path(path).resolve()) if Path(path).is_file() else str(path)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        rgb = _load_frame_rgb_uncached(path)
        self._store[key] = rgb
        if len(self._store) > self._max:
            self._store.popitem(last=False)
        return rgb


def _load_frame_rgb_uncached(path: str | Path) -> np.ndarray:
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


def load_frame_rgb(path: str | Path) -> np.ndarray:
    """Load a frame as RGB uint8; return placeholder if missing."""
    return _load_frame_rgb_uncached(path)


def draw_skeleton_on_bgr(
    bgr: np.ndarray,
    keypoints: list[Keypoint],
    *,
    min_visibility: float = MIN_VISIBILITY,
    highlight_dof: bool = False,
    highlight_joints: set[str] | None = None,
    foot_contact: dict[str, bool] | None = None,
) -> np.ndarray:
    """Draw MediaPipe skeleton on a BGR image copy (shared by pose overlay + GUI)."""
    if bgr.size == 0:
        return bgr
    out = bgr.copy()
    h, w = out.shape[:2]
    by_name = {kp.name: kp for kp in keypoints}
    index_to_kp = {i: by_name.get(LANDMARK_NAMES[i]) for i in range(len(LANDMARK_NAMES))}

    highlight_set: set[str] | None
    if highlight_joints is not None:
        highlight_set = highlight_joints
    elif highlight_dof:
        highlight_set = set(GAIT_DOF_LANDMARKS)
    else:
        highlight_set = set()

    for conn in PoseLandmarksConnections.POSE_LANDMARKS:
        start = index_to_kp.get(conn.start)
        end = index_to_kp.get(conn.end)
        if not start or not end:
            continue
        if start.visibility < min_visibility or end.visibility < min_visibility:
            continue
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        line_color = SKELETON_COLOR
        thickness = 2
        if highlight_set and (
            start.name in highlight_set or end.name in highlight_set
        ):
            line_color = DOF_COLOR
            thickness = 3
        cv2.line(out, pt1, pt2, line_color, thickness, cv2.LINE_AA)

    for kp in keypoints:
        if kp.name == "mid_hip" or kp.visibility < min_visibility:
            continue
        cx, cy = int(kp.x * w), int(kp.y * h)
        if highlight_set and kp.name in highlight_set:
            cv2.circle(out, (cx, cy), 8, DOF_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (cx, cy), 4, JOINT_COLOR, -1, cv2.LINE_AA)

    if foot_contact:
        from stablewalk.pose.contact import ContactDetector

        out = ContactDetector.highlight_contact_on_frame(
            out, keypoints, foot_contact, min_visibility=min_visibility
        )
    return out


def render_frame_with_skeleton(
    image_rgb: np.ndarray,
    keypoints: list[Keypoint],
    *,
    show_skeleton: bool = True,
    highlight_dof: bool = True,
    highlight_joints: set[str] | None = None,
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

    highlight_set: set[str] | None
    if highlight_joints is not None:
        highlight_set = highlight_joints
    elif highlight_dof:
        highlight_set = set(GAIT_DOF_LANDMARKS)
    else:
        highlight_set = set()

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
            line_color = SKELETON_COLOR
            thickness = 2
            if highlight_set and (
                start.name in highlight_set or end.name in highlight_set
            ):
                line_color = DOF_COLOR
                thickness = 3
            cv2.line(out, pt1, pt2, line_color, thickness, cv2.LINE_AA)

    for kp in keypoints:
        if kp.name == "mid_hip" or kp.visibility < min_visibility:
            continue
        cx, cy = int(kp.x * w), int(kp.y * h)
        if highlight_set and kp.name in highlight_set:
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
