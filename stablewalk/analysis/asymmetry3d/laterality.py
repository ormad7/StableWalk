"""Anatomical left/right identity from reconstructed 3D joints.

Canonical +X follows image +X (not always subject-right). A person facing
the camera has subject-left on image-right, so ``left_hip.x > right_hip.x``
is the expected MediaPipe labelling — not a swap.

Facing is inferred from travel when the pelvis actually moves, otherwise
from head/nose Z versus the pelvis. Unknown facing plus an internally
consistent left/right chain is a pass, not a fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_MIN_SAMPLES = 4
_SEP = 0.02
_TRAVEL_MIN = 0.08
_HEAD_Z_MIN = 0.03


@dataclass
class LateralityAssessment:
    status: str  # pass | warn | fail | unavailable
    facing: str  # camera | away | travel | unknown
    consistent: bool
    swapped: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "facing": self.facing,
            "consistent": self.consistent,
            "swapped": self.swapped,
            "detail": self.detail,
        }


def _median_xyz(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < _MIN_SAMPLES or arr.shape[1] < 3:
        return None
    finite = np.isfinite(arr).all(axis=1)
    if int(finite.sum()) < _MIN_SAMPLES:
        return None
    return np.median(arr[finite], axis=0)


def _median_x(points: np.ndarray | None) -> float | None:
    mid = _median_xyz(points)
    if mid is None:
        return None
    val = float(mid[0])
    return val if np.isfinite(val) else None


def infer_facing(
    *,
    head: np.ndarray | None = None,
    pelvis: np.ndarray | None = None,
    left_hip: np.ndarray | None = None,
    right_hip: np.ndarray | None = None,
) -> tuple[str, np.ndarray | None]:
    """Return (facing_label, unit forward vector) in the canonical frame."""
    if pelvis is not None:
        mid = np.asarray(pelvis, dtype=float)
    elif left_hip is not None and right_hip is not None:
        mid = 0.5 * (
            np.asarray(left_hip, dtype=float) + np.asarray(right_hip, dtype=float)
        )
    else:
        mid = None
    if mid is not None and mid.ndim == 2 and mid.shape[0] >= _MIN_SAMPLES:
        ok = np.isfinite(mid[:, 0]) & np.isfinite(mid[:, 2])
        if int(ok.sum()) >= _MIN_SAMPLES:
            path = mid[ok]
            travel = path[-1] - path[0]
            horiz = float(np.hypot(travel[0], travel[2]))
            if horiz >= _TRAVEL_MIN:
                forward = np.array([travel[0], 0.0, travel[2]], dtype=float)
                n = float(np.linalg.norm(forward))
                if n > 1e-9:
                    return "travel", forward / n
    if head is not None and mid is not None:
        head_arr = np.asarray(head, dtype=float)
        mid_arr = np.asarray(mid, dtype=float)
        n = min(head_arr.shape[0], mid_arr.shape[0])
        if n >= _MIN_SAMPLES:
            rel = head_arr[:n, 2] - mid_arr[:n, 2]
            finite = rel[np.isfinite(rel)]
            if finite.size >= _MIN_SAMPLES:
                z = float(np.median(finite))
                if z >= _HEAD_Z_MIN:
                    return "camera", np.array([0.0, 0.0, 1.0], dtype=float)
                if z <= -_HEAD_Z_MIN:
                    return "away", np.array([0.0, 0.0, -1.0], dtype=float)
    return "unknown", None


def predicted_right_axis(forward: np.ndarray | None) -> np.ndarray | None:
    """Body-right = forward × up (+Y) in a right-handed frame."""
    if forward is None:
        return None
    fwd = np.asarray(forward, dtype=float).reshape(3)
    n = float(np.linalg.norm(fwd))
    if n < 1e-9:
        return None
    fwd = fwd / n
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0], dtype=float))
    rn = float(np.linalg.norm(right))
    if rn < 1e-9:
        return None
    return right / rn


def assess_laterality(
    left_hip: np.ndarray | None,
    right_hip: np.ndarray | None,
    *,
    left_knee: np.ndarray | None = None,
    right_knee: np.ndarray | None = None,
    left_ankle: np.ndarray | None = None,
    right_ankle: np.ndarray | None = None,
    head: np.ndarray | None = None,
    pelvis: np.ndarray | None = None,
) -> LateralityAssessment:
    """Decide whether labelled left/right joints are anatomically consistent."""
    pairs: list[tuple[str, float, float]] = []
    for name, left, right in (
        ("hip", left_hip, right_hip),
        ("knee", left_knee, right_knee),
        ("ankle", left_ankle, right_ankle),
    ):
        lx = _median_x(left)
        rx = _median_x(right)
        if lx is None or rx is None:
            continue
        pairs.append((name, lx, rx))
    if not pairs:
        return LateralityAssessment(
            status="unavailable",
            facing="unknown",
            consistent=False,
            swapped=False,
            detail="Not enough hip/knee/ankle samples to confirm left vs right identity.",
        )

    signs = []
    for _name, lx, rx in pairs:
        delta = rx - lx
        if abs(delta) <= _SEP:
            signs.append(0.0)
        else:
            signs.append(float(np.sign(delta)))
    nonzero = [s for s in signs if s != 0.0]
    consistent = len(set(nonzero)) <= 1 and bool(nonzero)

    facing, forward = infer_facing(
        head=head,
        pelvis=pelvis,
        left_hip=left_hip,
        right_hip=right_hip,
    )
    pred_right = predicted_right_axis(forward)
    hip = next(((lx, rx) for name, lx, rx in pairs if name == "hip"), None)
    labeled = None
    if hip is not None:
        labeled = np.array([hip[1] - hip[0], 0.0, 0.0], dtype=float)

    swapped = False
    if pred_right is not None and labeled is not None:
        dot = float(np.dot(labeled, pred_right))
        if dot < -_SEP:
            swapped = True
        elif not consistent:
            swapped = True
    elif not consistent:
        swapped = True

    if swapped:
        status = "fail"
        relation = "disagree with the reconstructed facing / intra-limb chain"
    elif not consistent:
        status = "warn"
        relation = "are only weakly separated or disagree across joints"
    elif facing == "unknown" and labeled is not None and labeled[0] < -_SEP:
        # Consistent chain with subject-left on +X. Common when facing the
        # camera; not a swap without an independent facing estimate.
        status = "pass"
        relation = (
            "are internally consistent (subject-left on image +X). "
            "Facing is unknown so this is not scored as a landmark swap"
        )
    else:
        status = "pass"
        relation = "are internally consistent"

    pair_txt = ", ".join(
        f"{name} Lx={lx:+.3f} Rx={rx:+.3f}" for name, lx, rx in pairs
    )
    detail = (
        f"Facing={facing}. Median {pair_txt}. Left/right labels {relation}. "
        "Canonical +X is image-right; facing the camera puts subject-left on +X. "
        "Labels follow the pose landmark map."
    )
    return LateralityAssessment(
        status=status,
        facing=facing,
        consistent=consistent,
        swapped=swapped,
        detail=detail,
    )


def joint_series_from_recording(
    recording: Any,
    joint_id: str,
    *,
    max_frames: int = 120,
) -> np.ndarray | None:
    """Collect (T, 3) positions for one joint from gait-motion snapshots."""
    snaps = list(getattr(recording, "snapshots", None) or [])[: max(1, int(max_frames))]
    if not snaps:
        return None
    out = np.full((len(snaps), 3), np.nan, dtype=float)
    n_ok = 0
    for i, snap in enumerate(snaps):
        joints = getattr(snap, "joints", None) or {}
        sample = joints.get(joint_id)
        if sample is None:
            continue
        try:
            out[i, 0] = float(sample.position.x)
            out[i, 1] = float(sample.position.y)
            out[i, 2] = float(sample.position.z)
        except (TypeError, AttributeError, ValueError):
            continue
        if np.isfinite(out[i]).all():
            n_ok += 1
    if n_ok < _MIN_SAMPLES:
        return None
    return out


def assess_recording_laterality(recording: Any) -> LateralityAssessment:
    head = joint_series_from_recording(recording, "head")
    if head is None:
        head = joint_series_from_recording(recording, "nose")
    pelvis = joint_series_from_recording(recording, "pelvis")
    if pelvis is None:
        pelvis = joint_series_from_recording(recording, "mid_hip")
    return assess_laterality(
        joint_series_from_recording(recording, "left_hip"),
        joint_series_from_recording(recording, "right_hip"),
        left_knee=joint_series_from_recording(recording, "left_knee"),
        right_knee=joint_series_from_recording(recording, "right_knee"),
        left_ankle=joint_series_from_recording(recording, "left_ankle"),
        right_ankle=joint_series_from_recording(recording, "right_ankle"),
        head=head,
        pelvis=pelvis,
    )
