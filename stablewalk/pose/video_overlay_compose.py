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

from stablewalk.models.gait_wireframe import (
    GAIT_WIRE_BONES,
    GAIT_WIRE_JOINT_IDS,
    overlay_keypoints_for_gait_wire,
)
from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.dof import GAIT_DOF_LANDMARKS
from stablewalk.pose.skeleton import (
    DOF_COLOR,
    JOINT_COLOR,
    MIN_VISIBILITY,
    SKELETON_COLOR,
)

_LANDMARK_NAMES: list[str] | None = None
_POSE_LANDMARKS_CONNECTIONS = None


def _overlay_landmark_tables():
    """Lazy MediaPipe landmark tables — keep GUI layout off the MediaPipe import."""
    global _LANDMARK_NAMES, _POSE_LANDMARKS_CONNECTIONS
    if _LANDMARK_NAMES is None:
        from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib

        _LANDMARK_NAMES = [e.name.lower() for e in pose_landmarker_lib.PoseLandmark]
        _POSE_LANDMARKS_CONNECTIONS = pose_landmarker_lib.PoseLandmarksConnections
    return _LANDMARK_NAMES, _POSE_LANDMARKS_CONNECTIONS

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
    "pelvis_center": "Pel",
    "shoulder_center": "ShC",
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
GROUND_LINE_COLOR = (186, 198, 210)
GROUND_LINE_LABEL = "Estimated ground"
FOOT_CLEARANCE_HUD_COLOR = (245, 248, 252)
FOOT_CLEARANCE_HUD_SHADOW = (28, 32, 38)
FOOT_DROP_LINE_COLOR = (92, 188, 220)
FOOT_DROP_LABEL_COLOR = (235, 245, 250)

# Image-space candidates for the measuring landmark (y increases downward).
_FOOT_DROP_LANDMARKS: dict[str, tuple[str, ...]] = {
    "left": ("left_heel", "left_foot_index", "left_toe", "left_ankle"),
    "right": ("right_heel", "right_foot_index", "right_toe", "right_ankle"),
}

_STANCE_FOOT_LANDMARKS: dict[str, tuple[str, ...]] = {
    "left": ("left_heel", "left_foot_index", "left_ankle"),
    "right": ("right_heel", "right_foot_index", "right_ankle"),
}


@dataclass(frozen=True)
class VideoOverlayLayers:
    """Which overlay layers to draw on the video."""

    skeleton: bool = True
    joint_labels: bool = False
    com: bool = True
    ground_reaction: bool = True
    joint_trajectory: bool = True
    ground_plane: bool = True


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


def image_y_from_vertical_correspondence(
    hip_img_y: float,
    foot_img_y: float,
    hip_world_y: float,
    foot_world_y: float,
    floor_y: float,
) -> float | None:
    """Map a 3D floor Y onto image Y using measured hip and foot landmarks.

    Returns None when the correspondence is degenerate or would invent a
    line far outside the frame.
    """
    span = float(hip_world_y) - float(foot_world_y)
    if abs(span) < 1e-4:
        return None
    t = (float(hip_world_y) - float(floor_y)) / span
    y = float(hip_img_y) + t * (float(foot_img_y) - float(hip_img_y))
    if y < -0.05 or y > 1.15:
        return None
    return y


def estimate_image_ground_y(
    keypoints: Sequence[Keypoint],
    *,
    foot_contact: dict[str, bool] | None = None,
    min_visibility: float = MIN_VISIBILITY,
) -> float | None:
    """Normalized image Y of estimated ground from stance-foot landmarks.

    Image Y increases downward. Uses heel / foot-index of feet currently in
    stance when contact is known. Returns None when no measured landmarks are
    available — never fabricates a floor.
    """
    by_name = {kp.name: kp for kp in keypoints}
    ys: list[float] = []
    sides = ("left", "right")
    use_contact = foot_contact is not None
    for side in sides:
        if use_contact and not bool(foot_contact.get(side, False)):
            continue
        names = _STANCE_FOOT_LANDMARKS[side]
        if use_contact:
            names = names[:2]
        for name in names:
            kp = by_name.get(name)
            if kp is None or kp.visibility < min_visibility:
                continue
            ys.append(float(kp.y))
    if not ys:
        return None
    if use_contact:
        ys.sort()
        mid = len(ys) // 2
        if len(ys) % 2:
            return ys[mid]
        return 0.5 * (ys[mid - 1] + ys[mid])
    # Contact unknown: the lowest measured foot landmark in the image.
    return max(ys)


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


def _format_foot_clearance_hud_line(side: str, clearance_cm: float | None) -> str:
    label = "Left foot" if side == "left" else "Right foot"
    if clearance_cm is None:
        return f"{label}: —"
    return f"{label}: {float(clearance_cm):.1f} cm"


def draw_foot_clearance_hud(
    image_rgb: np.ndarray,
    *,
    left_cm: float | None,
    right_cm: float | None,
) -> np.ndarray:
    """Small Estimated Foot Clearance labels near the bottom of the video frame.

    Always returns a copy so shared decode / overlay caches are not mutated.
    """
    if image_rgb.size == 0:
        return image_rgb
    out = np.array(image_rgb, copy=True)
    h, w = out.shape[:2]
    if h < 24 or w < 48:
        return out

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    lines = (
        _format_foot_clearance_hud_line("left", left_cm),
        _format_foot_clearance_hud_line("right", right_cm),
    )
    # Keep type small and tucked into the lower-left corner.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42 if w >= 320 else 0.36
    thickness = 1
    line_gap = 16 if h >= 180 else 14
    x = 10
    y0 = h - 10 - line_gap
    for i, text in enumerate(lines):
        y = y0 + i * line_gap
        cv2.putText(
            bgr,
            text,
            (x + 1, y + 1),
            font,
            scale,
            FOOT_CLEARANCE_HUD_SHADOW,
            thickness + 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            text,
            (x, y),
            font,
            scale,
            FOOT_CLEARANCE_HUD_COLOR,
            thickness,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _foot_image_anchor(
    keypoints: Sequence[Keypoint],
    side: str,
    *,
    preferred_joint: str | None = None,
    min_visibility: float = MIN_VISIBILITY,
) -> tuple[float, float] | None:
    """Normalized (x, y) for the measuring foot landmark in image space."""
    by_name = {kp.name: kp for kp in keypoints}
    aliases = {
        f"{side}_toe": (f"{side}_toe", f"{side}_foot_index"),
        f"{side}_foot_index": (f"{side}_foot_index", f"{side}_toe"),
        f"{side}_heel": (f"{side}_heel",),
        f"{side}_ankle": (f"{side}_ankle",),
    }
    if preferred_joint:
        for name in aliases.get(preferred_joint, (preferred_joint,)):
            kp = by_name.get(name)
            if kp is not None and kp.visibility >= min_visibility:
                return float(kp.x), float(kp.y)
    best: tuple[float, float] | None = None
    best_y = -1.0
    for name in _FOOT_DROP_LANDMARKS.get(side, ()):
        kp = by_name.get(name)
        if kp is None or kp.visibility < min_visibility:
            continue
        # Prefer the lowest visible heel/toe (matches min clearance).
        if name.endswith("ankle") and best is not None:
            continue
        if float(kp.y) >= best_y:
            best_y = float(kp.y)
            best = (float(kp.x), float(kp.y))
    return best


def _draw_dashed_vertical(
    bgr: np.ndarray,
    x: int,
    y0: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    dash: int = 5,
    gap: int = 4,
) -> None:
    if y1 < y0:
        y0, y1 = y1, y0
    y = y0
    while y < y1:
        y_end = min(y + dash, y1)
        cv2.line(bgr, (x, y), (x, y_end), color, 1, cv2.LINE_AA)
        y = y_end + gap


def draw_foot_ground_drop_indicators(
    image_rgb: np.ndarray,
    keypoints: Sequence[Keypoint],
    *,
    ground_y_norm: float,
    sides: Sequence[str],
    left_cm: float | None = None,
    right_cm: float | None = None,
    preferred_joints: dict[str, str | None] | None = None,
    min_visibility: float = MIN_VISIBILITY,
) -> np.ndarray:
    """Vertical indicators from selected feet to the estimated ground line.

    Draws a simple dashed drop + cm label. Always returns a copy.
    """
    if image_rgb.size == 0 or not sides or not keypoints:
        return image_rgb
    out = np.array(image_rgb, copy=True)
    h, w = out.shape[:2]
    if h < 24 or w < 48:
        return out
    gy = int(round(float(ground_y_norm) * h))
    gy = max(1, min(h - 2, gy))
    preferred_joints = preferred_joints or {}
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.40 if w >= 320 else 0.34

    for side in sides:
        if side not in ("left", "right"):
            continue
        anchor = _foot_image_anchor(
            keypoints,
            side,
            preferred_joint=preferred_joints.get(side),
            min_visibility=min_visibility,
        )
        if anchor is None:
            continue
        fx_n, fy_n = anchor
        fx = int(round(fx_n * w))
        fy = int(round(fy_n * h))
        fx = max(1, min(w - 2, fx))
        fy = max(1, min(h - 2, fy))
        # Only draw when the foot is above the ground line in image space.
        if fy >= gy - 1:
            # Near/on ground: short tick + label still useful at contact.
            cv2.circle(bgr, (fx, gy), 3, FOOT_DROP_LINE_COLOR, 1, cv2.LINE_AA)
        else:
            _draw_dashed_vertical(bgr, fx, fy, gy, FOOT_DROP_LINE_COLOR)
            cv2.circle(bgr, (fx, fy), 3, FOOT_DROP_LINE_COLOR, 1, cv2.LINE_AA)
            cv2.circle(bgr, (fx, gy), 3, FOOT_DROP_LINE_COLOR, 1, cv2.LINE_AA)

        cm = left_cm if side == "left" else right_cm
        label = f"{float(cm):.1f} cm" if cm is not None else "—"
        mid_y = (fy + gy) // 2 if fy < gy else max(12, gy - 10)
        # Place label to the outside of each foot to reduce overlap.
        if side == "left":
            tx = max(4, fx - 58)
        else:
            tx = min(w - 56, fx + 8)
        cv2.putText(
            bgr,
            label,
            (tx + 1, mid_y + 1),
            font,
            scale,
            FOOT_CLEARANCE_HUD_SHADOW,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            bgr,
            label,
            (tx, mid_y),
            font,
            scale,
            FOOT_DROP_LABEL_COLOR,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _draw_estimated_ground_line(
    bgr: np.ndarray,
    *,
    ground_y_norm: float,
) -> None:
    """Horizontal estimated-ground line in image space (y downward)."""
    h, w = bgr.shape[:2]
    if h < 4 or w < 8:
        return
    y = int(round(float(ground_y_norm) * h))
    y = max(1, min(h - 2, y))
    cv2.line(bgr, (8, y), (w - 8, y), GROUND_LINE_COLOR, 1, cv2.LINE_AA)
    label_y = y - 8 if y > 18 else y + 16
    cv2.putText(
        bgr,
        GROUND_LINE_LABEL,
        (12, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        GROUND_LINE_COLOR,
        1,
        cv2.LINE_AA,
    )


def gait_wire_overlay_drawable(
    keypoints: Sequence[Keypoint],
    *,
    min_visibility: float = MIN_VISIBILITY,
) -> tuple[dict[str, Keypoint], list[tuple[str, str]]]:
    """Visible 16-joint overlay landmarks and bone pairs (image space)."""
    named = overlay_keypoints_for_gait_wire(keypoints)
    joints = {
        jid: kp
        for jid, kp in named.items()
        if kp is not None and float(kp.visibility) >= min_visibility
    }
    edges = [
        (parent, child)
        for parent, child in GAIT_WIRE_BONES
        if parent in joints and child in joints
    ]
    return joints, edges


def _draw_gait16_overlay(
    out: np.ndarray,
    keypoints: Sequence[Keypoint],
    *,
    min_visibility: float,
    highlight_set: set[str],
    highlight_bgr: tuple[int, int, int] | None = None,
) -> None:
    h, w = out.shape[:2]
    accent = highlight_bgr or DOF_COLOR
    joints, edges = gait_wire_overlay_drawable(
        keypoints, min_visibility=min_visibility
    )
    for parent, child in edges:
        start = joints[parent]
        end = joints[child]
        pt1 = (int(start.x * w), int(start.y * h))
        pt2 = (int(end.x * w), int(end.y * h))
        line_color = SKELETON_COLOR
        thickness = 2
        if highlight_set and (parent in highlight_set or child in highlight_set):
            line_color = accent
            thickness = 3
        cv2.line(out, pt1, pt2, line_color, thickness, cv2.LINE_AA)
    for name, kp in joints.items():
        cx, cy = int(kp.x * w), int(kp.y * h)
        if highlight_set and name in highlight_set:
            cv2.circle(out, (cx, cy), 8, accent, -1, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 10, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(out, (cx, cy), 6, JOINT_COLOR, -1, cv2.LINE_AA)


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
    ground_y_norm: float | None = None,
    skeleton_model: str = "full",
    highlight_bgr: tuple[int, int, int] | None = None,
) -> np.ndarray:
    out = bgr
    h, w = out.shape[:2]
    by_name = {kp.name: kp for kp in keypoints}

    if layers.ground_plane:
        y_norm = ground_y_norm
        if y_norm is None:
            y_norm = estimate_image_ground_y(
                keypoints,
                foot_contact=foot_contact,
                min_visibility=min_visibility,
            )
        if y_norm is not None:
            _draw_estimated_ground_line(out, ground_y_norm=y_norm)

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
        if skeleton_model == "gait_16":
            _draw_gait16_overlay(
                out,
                keypoints,
                min_visibility=min_visibility,
                highlight_set=highlight_set,
                highlight_bgr=highlight_bgr,
            )
        else:
            landmark_names, pose_conns = _overlay_landmark_tables()
            index_to_kp = {
                i: by_name.get(landmark_names[i]) for i in range(len(landmark_names))
            }
            accent = highlight_bgr or DOF_COLOR
            for conn in pose_conns.POSE_LANDMARKS:
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
                    line_color = accent
                    thickness = 3
                cv2.line(out, pt1, pt2, line_color, thickness, cv2.LINE_AA)

            for kp in keypoints:
                if kp.name == "mid_hip" or kp.visibility < min_visibility:
                    continue
                cx, cy = int(kp.x * w), int(kp.y * h)
                if highlight_set and kp.name in highlight_set:
                    cv2.circle(out, (cx, cy), 7, accent, -1, cv2.LINE_AA)
                    cv2.circle(out, (cx, cy), 9, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(out, (cx, cy), 4, JOINT_COLOR, -1, cv2.LINE_AA)

    if layers.joint_labels:
        label_names: tuple[str, ...] = (
            GAIT_WIRE_JOINT_IDS if skeleton_model == "gait_16" else _LABEL_JOINTS
        )
        label_lookup = (
            overlay_keypoints_for_gait_wire(keypoints)
            if skeleton_model == "gait_16"
            else by_name
        )
        for name in label_names:
            kp = label_lookup.get(name)
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
    ground_y_norm: float | None = None,
    skeleton_model: str = "full",
    highlight_bgr: tuple[int, int, int] | None = None,
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
        or layers.ground_plane
    ):
        return image_rgb

    base = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    # Opaque overlays: paint in-place (one BGR buffer + one RGB convert).
    if opacity >= 1.0:
        _draw_skeleton_layers(
            base,
            keypoints,
            layers=layers,
            min_visibility=min_visibility,
            highlight_dof=highlight_dof,
            highlight_joints=highlight_joints,
            gait_events=gait_events,
            foot_contact=foot_contact,
            grf_bw=grf_bw,
            trails=trails,
            ground_y_norm=ground_y_norm,
            skeleton_model=skeleton_model,
            highlight_bgr=highlight_bgr,
        )
        return cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
    if opacity <= 0.0:
        return image_rgb

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
        ground_y_norm=ground_y_norm,
        skeleton_model=skeleton_model,
        highlight_bgr=highlight_bgr,
    )
    blended = _blend_painted(base, painted, opacity)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


__all__ = [
    "GROUND_LINE_LABEL",
    "JointTrailBuffer",
    "VideoOverlayLayers",
    "compose_video_skeleton_overlay",
    "draw_foot_clearance_hud",
    "draw_foot_ground_drop_indicators",
    "estimate_image_com",
    "estimate_image_ground_y",
    "gait_wire_overlay_drawable",
    "image_y_from_vertical_correspondence",
]
