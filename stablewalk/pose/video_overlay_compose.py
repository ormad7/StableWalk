"""Compose reconstructed pose overlays onto original video frames.

High-performance single-bitmap path: draw layers on a working buffer, then blend
only painted pixels by opacity. Used by Overview Overlay Mode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

import cv2
import numpy as np

from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib

from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.dof import GAIT_DOF_LANDMARKS
from stablewalk.pose.skeleton import (
    DOF_COLOR,
    JOINT_COLOR,
    LANDMARK_NAMES,
    MIN_VISIBILITY,
    SKELETON_COLOR,
)

PoseLandmarksConnections = pose_landmarker_lib.PoseLandmarksConnections

_JOINT_SHORT_LABELS: dict[str, str] = {
    "nose": "N",
    "left_shoulder": "LS",
    "right_shoulder": "RS",
    "left_elbow": "LE",
    "right_elbow": "RE",
    "left_wrist": "LW",
    "right_wrist": "RW",
    "left_hip": "LH",
    "right_hip": "RH",
    "left_knee": "LK",
    "right_knee": "RK",
    "left_ankle": "LA",
    "right_ankle": "RA",
    "left_heel": "LHe",
    "right_heel": "RHe",
    "left_foot_index": "LF",
    "right_foot_index": "RF",
    "mid_hip": "COM",
}

_LABEL_JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

_TRAIL_JOINTS: tuple[str, ...] = (
    "left_ankle",
    "right_ankle",
    "left_knee",
    "right_knee",
    "left_wrist",
    "right_wrist",
)

_COM_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("left_hip", 0.22),
    ("right_hip", 0.22),
    ("left_shoulder", 0.14),
    ("right_shoulder", 0.14),
    ("left_knee", 0.08),
    ("right_knee", 0.08),
    ("nose", 0.06),
    ("left_ankle", 0.03),
    ("right_ankle", 0.03),
)

COM_COLOR = (0, 220, 255)
TRAIL_COLOR_L = (80, 220, 120)
TRAIL_COLOR_R = (220, 140, 60)
GRF_COLOR_L = (255, 180, 50)
GRF_COLOR_R = (50, 100, 255)


@dataclass(frozen=True)
class VideoOverlayLayers:
    """Which overlay layers to draw on the video."""

    skeleton: bool = True
    joint_labels: bool = False
    com: bool = True
    ground_reaction: bool = True
    joint_trajectory: bool = True


@dataclass
class JointTrailBuffer:
    """Ring buffer of normalized joint positions for trajectory trails."""

    max_frames: int = 48
    _points: dict[str, Deque[tuple[float, float]]] = field(default_factory=dict)
    _last_frame_index: int | None = None

    def clear(self) -> None:
        self._points.clear()
        self._last_frame_index = None

    def update(
        self,
        frame_index: int,
        keypoints: Sequence[Keypoint],
        *,
        min_visibility: float = MIN_VISIBILITY,
        joints: Iterable[str] = _TRAIL_JOINTS,
    ) -> None:
        if self._last_frame_index is not None:
            if frame_index == self._last_frame_index:
                return
            # Scrub jump — reset so trails stay continuous with playback.
            if abs(int(frame_index) - int(self._last_frame_index)) > 2:
                self.clear()
            elif frame_index < self._last_frame_index:
                self.clear()

        by_name = {kp.name: kp for kp in keypoints}
        for name in joints:
            kp = by_name.get(name)
            if kp is None or kp.visibility < min_visibility:
                continue
            buf = self._points.get(name)
            if buf is None:
                buf = deque(maxlen=self.max_frames)
                self._points[name] = buf
            buf.append((float(kp.x), float(kp.y)))
        self._last_frame_index = int(frame_index)

    def trails(self) -> dict[str, list[tuple[float, float]]]:
        return {name: list(pts) for name, pts in self._points.items() if len(pts) >= 2}


def estimate_image_com(
    keypoints: Sequence[Keypoint],
    *,
    min_visibility: float = MIN_VISIBILITY,
) -> tuple[float, float] | None:
    """Estimate COM in normalized image coordinates from pose landmarks."""
    by_name = {kp.name: kp for kp in keypoints}
    sx = sy = wsum = 0.0
    for name, weight in _COM_WEIGHTS:
        kp = by_name.get(name)
        if kp is None or kp.visibility < min_visibility:
            continue
        sx += kp.x * weight
        sy += kp.y * weight
        wsum += weight
    if wsum <= 1e-6:
        mid = by_name.get("mid_hip")
        if mid is not None and mid.visibility >= min_visibility:
            return float(mid.x), float(mid.y)
        lh, rh = by_name.get("left_hip"), by_name.get("right_hip")
        if (
            lh is not None
            and rh is not None
            and lh.visibility >= min_visibility
            and rh.visibility >= min_visibility
        ):
            return (lh.x + rh.x) * 0.5, (lh.y + rh.y) * 0.5
        return None
    return sx / wsum, sy / wsum


def _blend_painted(
    base_bgr: np.ndarray,
    painted_bgr: np.ndarray,
    opacity: float,
) -> np.ndarray:
    """Blend only pixels that differ from the base (overlay opacity 0–1)."""
    opacity = float(np.clip(opacity, 0.0, 1.0))
    if opacity <= 0.0:
        return base_bgr
    if opacity >= 1.0:
        return painted_bgr
    diff = cv2.absdiff(painted_bgr, base_bgr)
    mask = np.any(diff > 0, axis=2)
    if not np.any(mask):
        return base_bgr
    out = base_bgr.copy()
    blended = cv2.addWeighted(painted_bgr, opacity, base_bgr, 1.0 - opacity, 0.0)
    out[mask] = blended[mask]
    return out


def _draw_ground_reaction(
    bgr: np.ndarray,
    by_name: dict[str, Keypoint],
    *,
    left_bw: float,
    right_bw: float,
    left_on: bool,
    right_on: bool,
    min_visibility: float,
) -> None:
    h, w = bgr.shape[:2]
    for side, on, bw, color in (
        ("left", left_on, left_bw, GRF_COLOR_L),
        ("right", right_on, right_bw, GRF_COLOR_R),
    ):
        if not on and bw <= 0.05:
            continue
        foot = by_name.get(f"{side}_foot_index") or by_name.get(f"{side}_ankle")
        kp = foot
        if kp is None or kp.visibility < min_visibility:
            continue
        cx, cy = int(kp.x * w), int(kp.y * h)
        cv2.circle(bgr, (cx, cy), 12, color, 2, cv2.LINE_AA)
        cv2.circle(bgr, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
        magnitude = max(0.15, min(1.8, float(bw) if bw > 0 else 0.6))
        arrow_px = int(28 + 42 * magnitude)
        tip = (cx, max(4, cy - arrow_px))
        cv2.arrowedLine(bgr, (cx, cy), tip, color, 2, cv2.LINE_AA, tipLength=0.28)
        label = f"{bw:.1f}BW" if bw > 0.05 else "GRF"
        cv2.putText(
            bgr,
            label,
            (cx + 8, tip[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_skeleton_layers(
    bgr: np.ndarray,
    keypoints: Sequence[Keypoint],
    *,
    layers: VideoOverlayLayers,
    min_visibility: float,
    highlight_dof: bool,
    highlight_joints: set[str] | None,
    gait_events: list[str] | None,
    foot_contact: dict[str, bool] | None,
    grf_bw: tuple[float, float] | None,
    trails: dict[str, list[tuple[float, float]]] | None,
) -> np.ndarray:
    out = bgr
    h, w = out.shape[:2]
    by_name = {kp.name: kp for kp in keypoints}
    index_to_kp = {
        i: by_name.get(LANDMARK_NAMES[i]) for i in range(len(LANDMARK_NAMES))
    }

    if highlight_joints is not None:
        highlight_set = highlight_joints
    elif highlight_dof:
        highlight_set = set(GAIT_DOF_LANDMARKS)
    else:
        highlight_set = set()

    if layers.joint_trajectory and trails:
        for name, pts in trails.items():
            if len(pts) < 2:
                continue
            color = TRAIL_COLOR_L if name.startswith("left_") else TRAIL_COLOR_R
            poly = np.array(
                [(int(x * w), int(y * h)) for x, y in pts],
                dtype=np.int32,
            )
            cv2.polylines(out, [poly], False, color, 2, cv2.LINE_AA)
            tip = tuple(int(v) for v in poly[-1])
            cv2.circle(out, tip, 3, (255, 255, 255), -1, cv2.LINE_AA)

    if layers.skeleton:
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
                cv2.circle(out, (cx, cy), 7, DOF_COLOR, -1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 9, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(out, (cx, cy), 4, JOINT_COLOR, -1, cv2.LINE_AA)

    if layers.joint_labels:
        for name in _LABEL_JOINTS:
            kp = by_name.get(name)
            if kp is None or kp.visibility < min_visibility:
                continue
            label = _JOINT_SHORT_LABELS.get(name, name[:2].upper())
            cx, cy = int(kp.x * w), int(kp.y * h)
            cv2.putText(
                out,
                label,
                (cx + 6, cy - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (240, 240, 240),
                1,
                cv2.LINE_AA,
            )

    if layers.com:
        com = estimate_image_com(keypoints, min_visibility=min_visibility)
        if com is not None:
            cx, cy = int(com[0] * w), int(com[1] * h)
            cv2.drawMarker(
                out,
                (cx, cy),
                COM_COLOR,
                markerType=cv2.MARKER_CROSS,
                markerSize=18,
                thickness=2,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(out, (cx, cy), 10, COM_COLOR, 2, cv2.LINE_AA)
            cv2.putText(
                out,
                "COM",
                (cx + 12, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COM_COLOR,
                1,
                cv2.LINE_AA,
            )

    if layers.ground_reaction:
        contact = foot_contact or {}
        left_bw = (
            float(grf_bw[0])
            if grf_bw is not None
            else (1.0 if contact.get("left") else 0.0)
        )
        right_bw = (
            float(grf_bw[1])
            if grf_bw is not None
            else (1.0 if contact.get("right") else 0.0)
        )
        _draw_ground_reaction(
            out,
            by_name,
            left_bw=left_bw,
            right_bw=right_bw,
            left_on=bool(contact.get("left")) or left_bw > 0.05,
            right_on=bool(contact.get("right")) or right_bw > 0.05,
            min_visibility=min_visibility,
        )

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

    return out


def compose_video_skeleton_overlay(
    image_rgb: np.ndarray,
    keypoints: Sequence[Keypoint],
    *,
    layers: VideoOverlayLayers | None = None,
    opacity: float = 1.0,
    highlight_dof: bool = False,
    highlight_joints: set[str] | None = None,
    min_visibility: float = MIN_VISIBILITY,
    gait_events: list[str] | None = None,
    foot_contact: dict[str, bool] | None = None,
    grf_bw: tuple[float, float] | None = None,
    trails: dict[str, list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    """
    Render reconstructed skeleton overlays on the original video frame.

    ``opacity`` is 0–1 (UI may expose 0–100%). Only painted overlay pixels are
    blended so the underlying video stays crisp at low opacity.
    """
    if image_rgb.size == 0:
        return image_rgb
    layers = layers or VideoOverlayLayers()
    if not (
        layers.skeleton
        or layers.joint_labels
        or layers.com
        or layers.ground_reaction
        or layers.joint_trajectory
    ):
        return image_rgb

    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    painted = base.copy()
    _draw_skeleton_layers(
        painted,
        keypoints,
        layers=layers,
        min_visibility=min_visibility,
        highlight_dof=highlight_dof,
        highlight_joints=highlight_joints,
        gait_events=gait_events,
        foot_contact=foot_contact,
        grf_bw=grf_bw,
        trails=trails,
    )
    blended = _blend_painted(base, painted, opacity)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


__all__ = [
    "JointTrailBuffer",
    "VideoOverlayLayers",
    "compose_video_skeleton_overlay",
    "estimate_image_com",
]
