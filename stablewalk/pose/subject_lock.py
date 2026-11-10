"""Persistent subject selection among MediaPipe multi-person detections.

The lock follows one chosen person. During overlap or a fused single
detection it returns no pose instead of switching identity or combining
people. The SUBJECT overlay must be derived from the same keypoints that
enter analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from stablewalk.models.pose_data import Keypoint
from stablewalk.pose.validation import has_full_body_pose

_LOCK_MAX_MOVE = 0.18
_LOCK_BBOX_RATIO = 0.55
_OVERLAP_IOU = 0.08
_OVERLAP_GAP = 0.0
_FUSED_AREA_RATIO = 1.35
_AMBIGUOUS_SCORE_GAP = 0.45
_SEED_MAX_DIST = 0.30
_PREDICT_MAX_DT_S = 2.0

TRACKING_LOCKED = "locked"
TRACKING_UNAVAILABLE = "unavailable"
TRACKING_INITIALIZING = "initializing"

REASON_AMBIGUOUS = "subject_identity_ambiguous"
REASON_LOCK_LOST = "subject_lock_lost"


@dataclass
class SubjectSelection:
    """How to choose the analyzed person. Seed is used only to initialize."""

    seed_xy: tuple[float, float] | None = None
    seed_bbox: tuple[float, float, float, float] | None = None
    policy: str = "lock"
    overlap_iou: float = _OVERLAP_IOU
    overlap_gap: float = _OVERLAP_GAP
    fused_area_ratio: float = _FUSED_AREA_RATIO

    def fingerprint(self) -> str:
        if self.seed_xy is not None:
            return f"xy:{self.seed_xy[0]:.3f},{self.seed_xy[1]:.3f}"
        if self.seed_bbox is not None:
            box = ",".join(f"{v:.3f}" for v in self.seed_bbox)
            return f"bbox:{box}"
        return f"policy:{self.policy or 'lock'}"

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_xy": list(self.seed_xy) if self.seed_xy else None,
            "seed_bbox": list(self.seed_bbox) if self.seed_bbox else None,
            "policy": self.policy,
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, data: object | None) -> SubjectSelection:
        if not isinstance(data, dict):
            return cls()
        seed_xy = data.get("seed_xy")
        seed_bbox = data.get("seed_bbox")
        xy = None
        if isinstance(seed_xy, (list, tuple)) and len(seed_xy) == 2:
            xy = (float(seed_xy[0]), float(seed_xy[1]))
        box = None
        if isinstance(seed_bbox, (list, tuple)) and len(seed_bbox) == 4:
            box = tuple(float(v) for v in seed_bbox)
        return cls(
            seed_xy=xy,
            seed_bbox=box,  # type: ignore[arg-type]
            policy=str(data.get("policy") or "lock"),
        )


@dataclass
class SubjectLock:
    """Last accepted subject centroid, size, and motion."""

    cx: float
    cy: float
    bbox_h: float
    initialized: bool = False
    last_bbox: tuple[float, float, float, float] | None = None
    vx: float = 0.0
    vy: float = 0.0
    last_time_s: float = 0.0
    last_n_people: int = 1


@dataclass(frozen=True)
class SubjectPick:
    keypoints: list[Keypoint] | None
    lock: SubjectLock | None
    tracking_status: str
    identity_locked: bool
    reason: str
    subject_bbox: tuple[float, float, float, float] | None
    other_bboxes: list[tuple[float, float, float, float]]


def _xy(keypoints: Sequence[Keypoint], name: str) -> tuple[float, float] | None:
    for kp in keypoints:
        if kp.name == name:
            return (float(kp.x), float(kp.y))
    return None


def pose_bbox_norm(
    keypoints: Sequence[Keypoint],
    *,
    pad: float = 0.04,
) -> tuple[float, float, float, float] | None:
    """Normalized image-space bbox (x0, y0, x1, y1) for overlay / ROI."""
    xs = [float(kp.x) for kp in keypoints if kp.visibility > 0.2]
    ys = [float(kp.y) for kp in keypoints if kp.visibility > 0.2]
    if len(xs) < 4 or len(ys) < 4:
        return None
    x0 = max(0.0, min(xs) - pad)
    y0 = max(0.0, min(ys) - pad)
    x1 = min(1.0, max(xs) + pad)
    y1 = min(1.0, max(ys) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def bbox_area(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_iou(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 1e-12:
        return 0.0
    return inter / union


def bbox_contains(
    bbox: tuple[float, float, float, float] | None,
    x: float,
    y: float,
) -> bool:
    if bbox is None:
        return False
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def lock_bbox_norm(lock: SubjectLock | None) -> tuple[float, float, float, float] | None:
    """Approximate subject ROI from the persistent lock centroid and height."""
    if lock is None or not lock.initialized:
        return None
    if lock.last_bbox is not None:
        return lock.last_bbox
    height = max(float(lock.bbox_h), 0.12)
    width = height * 0.45
    x0 = max(0.0, lock.cx - width * 0.5)
    x1 = min(1.0, lock.cx + width * 0.5)
    y0 = max(0.0, lock.cy - height * 0.45)
    y1 = min(1.0, lock.cy + height * 0.55)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def pose_bbox_height(keypoints: Sequence[Keypoint]) -> float | None:
    ys = [float(kp.y) for kp in keypoints if kp.visibility > 0.2]
    if len(ys) < 4:
        return None
    return max(ys) - min(ys)


def pose_centroid(keypoints: Sequence[Keypoint]) -> tuple[float, float] | None:
    hips = []
    for name in ("mid_hip", "left_hip", "right_hip"):
        pt = _xy(keypoints, name)
        if pt is not None:
            hips.append(pt)
    if hips:
        cx = sum(p[0] for p in hips) / len(hips)
        cy = sum(p[1] for p in hips) / len(hips)
        return (cx, cy)
    xs = [float(kp.x) for kp in keypoints]
    ys = [float(kp.y) for kp in keypoints]
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _predicted_lock(lock: SubjectLock, time_s: float) -> tuple[float, float]:
    dt = float(time_s) - float(lock.last_time_s or 0.0)
    if dt <= 0.0:
        return (lock.cx, lock.cy)
    dt = min(dt, _PREDICT_MAX_DT_S)
    return (lock.cx + lock.vx * dt, lock.cy + lock.vy * dt)


def _score_candidate(
    keypoints: Sequence[Keypoint],
    lock: SubjectLock | None,
    *,
    time_s: float = 0.0,
) -> float:
    """Higher is better. Prefers a locked match, else a large full-body pose."""
    height = pose_bbox_height(keypoints) or 0.0
    centroid = pose_centroid(keypoints)
    full = 1.0 if has_full_body_pose(list(keypoints)) else 0.0
    if lock is None or not lock.initialized or centroid is None:
        return height * (1.0 + 0.5 * full)
    px, py = _predicted_lock(lock, time_s)
    dx = centroid[0] - px
    dy = centroid[1] - py
    dist = (dx * dx + dy * dy) ** 0.5
    size = abs(height - lock.bbox_h) / max(lock.bbox_h, 1e-6)
    if dist > _LOCK_MAX_MOVE:
        return -1.0
    if size > _LOCK_BBOX_RATIO and dist > 0.08:
        return -1.0
    return (2.0 - dist / max(_LOCK_MAX_MOVE, 1e-6)) + full + height - 0.35 * min(size, 2.0)


def _candidate_boxes(
    candidates: Sequence[Sequence[Keypoint]],
) -> list[tuple[float, float, float, float] | None]:
    return [pose_bbox_norm(kps) for kps in candidates]


def bbox_width(bbox: tuple[float, float, float, float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0])


def bbox_x_gap(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> float:
    """Horizontal separation; 0 when the x-ranges overlap."""
    if a is None or b is None:
        return 1.0
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0


def _people_too_close(
    boxes: Sequence[tuple[float, float, float, float] | None],
    *,
    min_iou: float,
    min_gap: float,
) -> bool:
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if bbox_iou(a, b) >= min_iou:
                return True
            if bbox_x_gap(a, b) <= min_gap:
                return True
    return False


def _predicted_bbox(
    lock: SubjectLock, time_s: float
) -> tuple[float, float, float, float] | None:
    box = lock.last_bbox or lock_bbox_norm(lock)
    if box is None:
        return None
    px, py = _predicted_lock(lock, time_s)
    dx = px - lock.cx
    dy = py - lock.cy
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def _seed_index(
    candidates: Sequence[Sequence[Keypoint]],
    boxes: Sequence[tuple[float, float, float, float] | None],
    selection: SubjectSelection | None,
) -> int | None:
    if selection is None:
        return None
    if selection.seed_bbox is not None:
        best_i = None
        best_iou = 0.0
        for i, box in enumerate(boxes):
            iou = bbox_iou(box, selection.seed_bbox)
            if iou > best_iou:
                best_iou = iou
                best_i = i
        if best_i is not None and best_iou >= 0.08:
            return best_i
    if selection.seed_xy is None:
        return None
    sx, sy = selection.seed_xy
    for i, box in enumerate(boxes):
        if bbox_contains(box, sx, sy):
            return i
    best_i = None
    best_d = 1e9
    for i, kps in enumerate(candidates):
        centroid = pose_centroid(kps)
        if centroid is None:
            continue
        dist = ((centroid[0] - sx) ** 2 + (centroid[1] - sy) ** 2) ** 0.5
        if dist < best_d:
            best_d = dist
            best_i = i
    if best_i is not None and best_d <= _SEED_MAX_DIST:
        return best_i
    return None


def _unavailable(
    lock: SubjectLock | None,
    boxes: Sequence[tuple[float, float, float, float] | None],
    reason: str,
    *,
    initializing: bool = False,
) -> SubjectPick:
    others = [b for b in boxes if b is not None]
    return SubjectPick(
        keypoints=None,
        lock=lock,
        tracking_status=(
            TRACKING_INITIALIZING if initializing else TRACKING_UNAVAILABLE
        ),
        identity_locked=False,
        reason=reason,
        subject_bbox=None,
        other_bboxes=others,
    )


def _accept(
    keypoints: Sequence[Keypoint],
    lock: SubjectLock | None,
    boxes: Sequence[tuple[float, float, float, float] | None],
    chosen_box: tuple[float, float, float, float] | None,
    *,
    time_s: float,
) -> SubjectPick:
    centroid = pose_centroid(keypoints)
    height = pose_bbox_height(keypoints) or (lock.bbox_h if lock else 0.3)
    vx = lock.vx if lock else 0.0
    vy = lock.vy if lock else 0.0
    if lock is not None and lock.initialized and centroid is not None:
        dt = float(time_s) - float(lock.last_time_s or 0.0)
        if dt > 1e-3:
            vx = (centroid[0] - lock.cx) / dt
            vy = (centroid[1] - lock.cy) / dt
    if centroid is None:
        new_lock = lock
    else:
        new_lock = SubjectLock(
            cx=centroid[0],
            cy=centroid[1],
            bbox_h=height,
            initialized=True,
            last_bbox=chosen_box,
            vx=vx,
            vy=vy,
            last_time_s=float(time_s),
            last_n_people=len(boxes),
        )
    others = [b for b in boxes if b is not None and b != chosen_box]
    return SubjectPick(
        keypoints=list(keypoints),
        lock=new_lock,
        tracking_status=TRACKING_LOCKED,
        identity_locked=True,
        reason="",
        subject_bbox=chosen_box,
        other_bboxes=others,
    )


def pick_subject(
    candidates: Sequence[Sequence[Keypoint]],
    lock: SubjectLock | None,
    *,
    selection: SubjectSelection | None = None,
    time_s: float = 0.0,
) -> SubjectPick:
    """Choose one person or mark tracking unavailable. Never merges people."""
    boxes = _candidate_boxes(candidates)
    selection = selection or SubjectSelection()
    if not candidates:
        return _unavailable(lock, boxes, REASON_LOCK_LOST)

    if _people_too_close(
        boxes,
        min_iou=selection.overlap_iou,
        min_gap=selection.overlap_gap,
    ):
        return _unavailable(lock, boxes, REASON_AMBIGUOUS)

    if lock is not None and lock.initialized and len(candidates) == 1:
        only_box = boxes[0]
        lock_box = _predicted_bbox(lock, time_s) or lock.last_bbox or lock_bbox_norm(lock)
        area_fused = (
            only_box is not None
            and lock_box is not None
            and bbox_area(only_box)
            > selection.fused_area_ratio * max(bbox_area(lock_box), 1e-6)
        )
        if lock.last_n_people >= 2 or area_fused:
            return _unavailable(lock, boxes, REASON_AMBIGUOUS)

    if lock is None or not lock.initialized:
        seed_i = _seed_index(candidates, boxes, selection)
        if seed_i is not None:
            return _accept(
                candidates[seed_i],
                lock,
                boxes,
                boxes[seed_i],
                time_s=time_s,
            )
        if selection.seed_xy is not None or selection.seed_bbox is not None:
            return _unavailable(
                lock, boxes, REASON_LOCK_LOST, initializing=True
            )
        scored = [
            (_score_candidate(kps, None, time_s=time_s), i)
            for i, kps in enumerate(candidates)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_i = scored[0][1]
        return _accept(
            candidates[best_i], lock, boxes, boxes[best_i], time_s=time_s
        )

    scored = [
        (_score_candidate(kps, lock, time_s=time_s), i)
        for i, kps in enumerate(candidates)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_i = scored[0]
    if best_score < 0:
        reacquired = _reacquire_after_gap(
            candidates, boxes, lock, time_s=time_s
        )
        if reacquired is not None:
            return reacquired
        return _unavailable(lock, boxes, REASON_LOCK_LOST)
    if len(scored) > 1 and scored[1][0] >= 0:
        if best_score - scored[1][0] < _AMBIGUOUS_SCORE_GAP:
            return _unavailable(lock, boxes, REASON_AMBIGUOUS)
    return _accept(
        candidates[best_i], lock, boxes, boxes[best_i], time_s=time_s
    )


def _reacquire_after_gap(
    candidates: Sequence[Sequence[Keypoint]],
    boxes: Sequence[tuple[float, float, float, float] | None],
    lock: SubjectLock,
    *,
    time_s: float,
) -> SubjectPick | None:
    """After occlusion, pick the uniquely nearest person to the predicted lock."""
    dt = float(time_s) - float(lock.last_time_s or 0.0)
    if dt < 0.25 or len(candidates) < 2:
        return None
    px, py = _predicted_lock(lock, time_s)
    dists: list[tuple[float, int]] = []
    for i, kps in enumerate(candidates):
        centroid = pose_centroid(kps)
        if centroid is None:
            continue
        dist = ((centroid[0] - px) ** 2 + (centroid[1] - py) ** 2) ** 0.5
        dists.append((dist, i))
    if not dists:
        return None
    dists.sort()
    best_d, best_i = dists[0]
    if best_d > 0.35:
        return None
    if len(dists) > 1 and dists[1][0] - best_d < 0.12:
        return None
    return _accept(candidates[best_i], lock, boxes, boxes[best_i], time_s=time_s)


def select_subject_keypoints(
    candidates: Sequence[Sequence[Keypoint]],
    lock: SubjectLock | None,
    *,
    selection: SubjectSelection | None = None,
    time_s: float = 0.0,
) -> tuple[list[Keypoint] | None, SubjectLock | None]:
    """Pick the locked subject, or initialize on the seed / largest pose."""
    picked = pick_subject(
        candidates, lock, selection=selection, time_s=time_s
    )
    return picked.keypoints, picked.lock


def selection_for_source(
    source: str | None,
    override: SubjectSelection | None = None,
) -> SubjectSelection:
    """Demo seeds plus an optional GUI/click override."""
    if override is not None and (
        override.seed_xy is not None or override.seed_bbox is not None
    ):
        return override
    name = str(source or "").replace("\\", "/").rsplit("/", 1)[-1].lower()
    if name == "normal_gait_v2.mp4":
        # White-shirt side-view walker, entering from the left at frame 0.
        return SubjectSelection(seed_xy=(0.12, 0.55), policy="seed")
    if override is not None:
        return override
    return SubjectSelection()


def pose_dump_matches_selection(payload: dict, selection: SubjectSelection) -> bool:
    """False when a cached pose dump was built for a different subject."""
    stored = payload.get("subject_selection")
    stored_fp = ""
    if isinstance(stored, dict):
        stored_fp = str(stored.get("fingerprint") or "")
    explicit = selection.seed_xy is not None or selection.seed_bbox is not None
    if not explicit:
        return (not stored_fp) or stored_fp == selection.fingerprint()
    return bool(stored_fp) and stored_fp == selection.fingerprint()
