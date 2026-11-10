"""Shared skeleton drawing and frame loading for GUI, overlays, and exports."""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from stablewalk.pose.dof import GAIT_DOF_LANDMARKS
from stablewalk.models.pose_data import Keypoint

# MediaPipe landmark enums are loaded lazily — importing mediapipe at module
# import time costs ~1–2s on cold GUI startup.
_LANDMARK_NAMES: list[str] | None = None
_POSE_LANDMARKS_CONNECTIONS = None

MIN_VISIBILITY = 0.5
DISPLAY_MAX_WIDTH = 1280

SKELETON_COLOR = (0, 255, 136)  # BGR green
JOINT_COLOR = (71, 71, 255)  # BGR red-ish
# Selected-joint accent — keep in sync with theme.SELECTED_JOINT_ACCENT (#ffd666).
try:
    from stablewalk.ui.theme import SELECTED_JOINT_ACCENT_BGR as DOF_COLOR
except Exception:  # pragma: no cover - theme optional at import time
    DOF_COLOR = (102, 214, 255)  # BGR amber


def _mediapipe_landmarks():
    global _LANDMARK_NAMES, _POSE_LANDMARKS_CONNECTIONS
    if _LANDMARK_NAMES is None:
        from mediapipe.tasks.python.vision import pose_landmarker as pose_landmarker_lib

        _LANDMARK_NAMES = [e.name.lower() for e in pose_landmarker_lib.PoseLandmark]
        _POSE_LANDMARKS_CONNECTIONS = pose_landmarker_lib.PoseLandmarksConnections
    return _LANDMARK_NAMES, _POSE_LANDMARKS_CONNECTIONS


def __getattr__(name: str):
    if name == "LANDMARK_NAMES":
        names, _ = _mediapipe_landmarks()
        return names
    if name == "PoseLandmarksConnections":
        _, conns = _mediapipe_landmarks()
        return conns
    raise AttributeError(name)


class FrameRgbCache:
    """LRU cache for decoded frame images (speeds up GUI playback)."""

    def __init__(self, max_entries: int = 128) -> None:
        self._max = max(16, max_entries)
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self._prefetch_pending: set[str] = set()

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._prefetch_pending.clear()

    def trim(self, max_entries: int | None = None) -> int:
        """Drop oldest entries down to ``max_entries`` (or the cache cap)."""
        if max_entries is None:
            limit = self._max
        else:
            limit = max(1, int(max_entries))
        removed = 0
        with self._lock:
            while len(self._store) > limit:
                self._store.popitem(last=False)
                removed += 1
        return removed

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def contains(self, path: str | Path) -> bool:
        key = str(path)
        with self._lock:
            return key in self._store

    def get_rgb(self, path: str | Path) -> np.ndarray:
        # Key on the raw path string so cache hits avoid a filesystem stat and
        # Path allocation every frame during playback. Callers pass a stable
        # ``frame.image_path`` so this remains a reliable key.
        key = str(path)
        with self._lock:
            cached = self._store.get(key)
            if cached is not None:
                self._store.move_to_end(key)
                return cached
        rgb = _load_frame_rgb_uncached(path)
        with self._lock:
            self._store[key] = rgb
            self._prefetch_pending.discard(key)
            if len(self._store) > self._max:
                self._store.popitem(last=False)
        return rgb

    def prefetch(self, path: str | Path) -> None:
        """Decode ``path`` on a background thread if not already cached."""
        key = str(path)
        with self._lock:
            if key in self._store or key in self._prefetch_pending:
                return
            self._prefetch_pending.add(key)

        def _worker() -> None:
            try:
                rgb = _load_frame_rgb_uncached(path)
                with self._lock:
                    self._store[key] = rgb
                    if len(self._store) > self._max:
                        self._store.popitem(last=False)
            except Exception:
                pass
            finally:
                with self._lock:
                    self._prefetch_pending.discard(key)

        threading.Thread(
            target=_worker, name="stablewalk-frame-prefetch", daemon=True
        ).start()


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
    landmark_names, pose_conns = _mediapipe_landmarks()
    by_name = {kp.name: kp for kp in keypoints}
    index_to_kp = {i: by_name.get(landmark_names[i]) for i in range(len(landmark_names))}

    highlight_set: set[str] | None
    if highlight_joints is not None:
        highlight_set = highlight_joints
    elif highlight_dof:
        highlight_set = set(GAIT_DOF_LANDMARKS)
    else:
        highlight_set = set()

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


SUBJECT_ROI_COLOR = (80, 220, 255)  # BGR amber-cyan
OTHER_ROI_COLOR = (90, 90, 180)
LOCK_LOST_COLOR = (40, 40, 220)
MISSING_POSE_COLOR = (210, 210, 210)


def _draw_norm_bbox(
    bgr: np.ndarray,
    bbox: tuple[float, float, float, float],
    color: tuple[int, int, int],
    *,
    thickness: int = 2,
    label: str = "",
) -> None:
    h, w = bgr.shape[:2]
    x0, y0, x1, y1 = bbox
    p1 = (int(max(0, min(w - 1, x0 * w))), int(max(0, min(h - 1, y0 * h))))
    p2 = (int(max(0, min(w - 1, x1 * w))), int(max(0, min(h - 1, y1 * h))))
    cv2.rectangle(bgr, p1, p2, color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(
            bgr,
            label,
            (p1[0], max(18, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_identity_on_bgr(bgr: np.ndarray, frame) -> np.ndarray:
    """Draw the analyzed person's box from the same keypoints used in analysis."""
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return bgr
    out = bgr if bgr.flags.writeable else bgr.copy()
    status = ""
    reason = str(getattr(frame, "rejection_reason", "") or "")
    try:
        status = frame.processing_status()
    except Exception:
        if getattr(frame, "detected", False):
            status = "pose_ok"
        elif getattr(frame, "skipped", False):
            status = "filter_skip"
        else:
            status = "undetected_unclassified"
    n_people = int(getattr(frame, "n_people", 0) or 0)
    tracking = str(getattr(frame, "tracking_status", "") or "")
    identity_locked = bool(getattr(frame, "identity_locked", False))
    other_bboxes = list(getattr(frame, "other_bboxes", None) or [])
    kps = list(getattr(frame, "keypoints", None) or [])
    subject_bbox = None
    from stablewalk.pose.subject_lock import pose_bbox_norm

    if kps and (status == "pose_ok" or identity_locked):
        subject_bbox = pose_bbox_norm(kps)
    for box in other_bboxes:
        if box and len(box) == 4:
            _draw_norm_bbox(
                out,
                tuple(float(v) for v in box),
                OTHER_ROI_COLOR,
                thickness=1,
                label="Other",
            )
    if subject_bbox and len(subject_bbox) == 4:
        _draw_norm_bbox(
            out,
            tuple(float(v) for v in subject_bbox),
            SUBJECT_ROI_COLOR,
            thickness=2,
            label="SUBJECT",
        )
    src = int(getattr(frame, "frame_index", 0) or 0)
    if tracking == "unavailable" or status == "subject_lock_skip":
        track_label = "TRACKING UNAVAILABLE"
        hud_color = LOCK_LOST_COLOR
    elif identity_locked:
        track_label = "tracking=locked"
        if status and status != "pose_ok":
            track_label += f"  {status}"
        hud_color = SUBJECT_ROI_COLOR if status == "pose_ok" else MISSING_POSE_COLOR
    else:
        track_label = f"tracking={tracking or status}"
        hud_color = MISSING_POSE_COLOR
    hud = f"src={src}  people={n_people}  {track_label}"
    if reason:
        hud += f"  {reason}"
    cv2.putText(
        out,
        hud,
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        hud_color,
        2,
        cv2.LINE_AA,
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
    landmark_names, pose_conns = _mediapipe_landmarks()
    by_name = {kp.name: kp for kp in keypoints}
    index_to_kp = {
        i: by_name.get(landmark_names[i])
        for i in range(len(landmark_names))
    }

    highlight_set: set[str] | None
    if highlight_joints is not None:
        highlight_set = highlight_joints
    elif highlight_dof:
        highlight_set = set(GAIT_DOF_LANDMARKS)
    else:
        highlight_set = set()

    if show_skeleton:
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
    landmark_names, pose_conns = _mediapipe_landmarks()
    by_name = {kp.name: kp for kp in keypoints}
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for conn in pose_conns.POSE_LANDMARKS:
        if conn.start >= len(landmark_names) or conn.end >= len(landmark_names):
            continue
        a = by_name.get(landmark_names[conn.start])
        b = by_name.get(landmark_names[conn.end])
        if not a or not b or a.visibility < min_visibility or b.visibility < min_visibility:
            continue
        segments.append(
            (
                (clamp01(a.x), clamp01(a.y)),
                (clamp01(b.x), clamp01(b.y)),
            )
        )
    return segments
