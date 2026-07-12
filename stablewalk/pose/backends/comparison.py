"""
Metrics for comparing pose / HMR backends on the same video.

Produces structured, non-prescriptive comparison data — no automatic
"winner" declaration.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any

from stablewalk.biomechanics.marker_reconstruction import reconstruct_markers_from_trc_frames
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.pose.backends.canonical import canonical_to_trc_landmarks
from stablewalk.pose.backends.types import HumanMotionSequence
from stablewalk.pose.kinematics import compute_joint_angles
from stablewalk.models.pose_data import Keypoint


@dataclass
class BackendComparisonMetrics:
    backend_name: str
    available: bool
    availability_message: str = ""
    frame_count: int = 0
    valid_frame_ratio: float = 0.0
    mean_landmark_confidence: float = 0.0
    trajectory_smoothness_score: float = 0.0
    joint_angle_consistency_score: float = 0.0
    foot_ground_jitter_mm: float | None = None
    pelvis_trajectory_consistency: float = 0.0
    opensim_marker_confidence_mean: float | None = None
    opensim_ik_readiness_score: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "available": self.available,
            "availability_message": self.availability_message,
            "frame_count": self.frame_count,
            "valid_frame_ratio_percent": round(self.valid_frame_ratio * 100.0, 1),
            "mean_landmark_confidence": round(self.mean_landmark_confidence, 3),
            "trajectory_smoothness_score": round(self.trajectory_smoothness_score, 1),
            "joint_angle_consistency_score": round(self.joint_angle_consistency_score, 1),
            "foot_ground_jitter_mm": self.foot_ground_jitter_mm,
            "pelvis_trajectory_consistency": round(self.pelvis_trajectory_consistency, 3),
            "opensim_marker_confidence_mean": self.opensim_marker_confidence_mean,
            "opensim_ik_readiness_score": self.opensim_ik_readiness_score,
            "notes": self.notes,
        }


def _joint_series(
    sequence: HumanMotionSequence,
    joint_id: str,
) -> list[tuple[float, float, float] | None]:
    out: list[tuple[float, float, float] | None] = []
    for frame in sequence.frames:
        if not frame.detected:
            out.append(None)
            continue
        out.append(frame.joint_positions_3d.get(joint_id))
    return out


def _velocity_jerk_score(positions: list[tuple[float, float, float] | None]) -> float:
    """Higher = smoother (lower normalized jerk). Scale 0–100."""
    speeds: list[float] = []
    accels: list[float] = []
    prev = None
    prev_v = None
    for p in positions:
        if p is None or prev is None:
            prev = p
            prev_v = None
            continue
        v = math.sqrt(sum((a - b) ** 2 for a, b in zip(p, prev)))
        speeds.append(v)
        if prev_v is not None:
            accels.append(abs(v - prev_v))
        prev_v = v
        prev = p
    if not accels:
        return 50.0
    jerk = statistics.mean(accels)
    return max(0.0, min(100.0, 100.0 - jerk * 400.0))


def _angle_consistency(sequence: HumanMotionSequence) -> float:
    """Lower std of hip/knee angles → higher score (0–100)."""
    hip_angles: list[float] = []
    knee_angles: list[float] = []
    for frame in sequence.frames:
        if not frame.detected:
            continue
        kps = [
            Keypoint(name=n, x=p[0], y=p[1], z=p[2], visibility=1.0)
            for n, p in (frame.raw_landmarks or {}).items()
        ]
        if not kps:
            continue
        angles = compute_joint_angles(kps)
        if angles.left_hip is not None:
            hip_angles.append(angles.left_hip)
        if angles.right_hip is not None:
            hip_angles.append(angles.right_hip)
        if angles.left_knee is not None:
            knee_angles.append(angles.left_knee)
        if angles.right_knee is not None:
            knee_angles.append(angles.right_knee)
    stds: list[float] = []
    for arr in (hip_angles, knee_angles):
        if len(arr) >= 3:
            stds.append(statistics.pstdev(arr))
    if not stds:
        return 50.0
    mean_std = statistics.mean(stds)
    return max(0.0, min(100.0, 100.0 - mean_std * 0.8))


def _foot_ground_jitter(sequence: HumanMotionSequence) -> float | None:
    """Vertical jitter of foot landmarks (normalized units, scaled to pseudo-mm)."""
    ys: list[float] = []
    for jid in ("left_heel", "right_heel", "left_toe", "right_toe"):
        series = _joint_series(sequence, jid)
        vals = [p[1] for p in series if p is not None]
        if len(vals) >= 3:
            ys.append(statistics.pstdev(vals))
    if not ys:
        return None
    return statistics.mean(ys) * 1000.0


def _pelvis_consistency(sequence: HumanMotionSequence) -> float:
    series = _joint_series(sequence, ROOT_JOINT_ID)
    xs = [p[0] for p in series if p is not None]
    ys = [p[1] for p in series if p is not None]
    if len(xs) < 3:
        return 0.0
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)
    path = math.sqrt(dx * dx + dy * dy)
    if path < 1e-6:
        return 1.0
    jitter = statistics.pstdev(xs) + statistics.pstdev(ys)
    return max(0.0, min(1.0, 1.0 - jitter / max(path, 1e-6)))


def _opensim_marker_confidence(sequence: HumanMotionSequence, fps: float) -> tuple[float | None, float | None]:
    frames_data = []
    for i, frame in enumerate(sequence.frames):
        if not frame.detected:
            continue
        lm = canonical_to_trc_landmarks(frame.joint_positions_3d, scale_to_mm=1000.0)
        if len(lm) < 8:
            continue
        frames_data.append((i, frame.timestamp_s, lm))
    if len(frames_data) < 2:
        return None, None
    result = reconstruct_markers_from_trc_frames(frames_data, fps=fps)
    confs: list[float] = []
    for _fn, _t, markers in result.frames:
        for m in markers.values():
            if m.position is not None:
                confs.append(m.confidence)
    if not confs:
        return None, result.ik_readiness_score
    return statistics.mean(confs), result.ik_readiness_score


def compute_backend_metrics(
    sequence: HumanMotionSequence | None,
    *,
    backend_name: str,
    available: bool,
    availability_message: str = "",
) -> BackendComparisonMetrics:
    """Compute comparison metrics for one backend run."""
    if not available or sequence is None or not sequence.frames:
        return BackendComparisonMetrics(
            backend_name=backend_name,
            available=available,
            availability_message=availability_message,
            notes=["Backend did not produce a motion sequence"],
        )

    confs: list[float] = []
    for frame in sequence.frames:
        if frame.detected:
            confs.extend(frame.landmark_confidence.values())

    pelvis = _joint_series(sequence, ROOT_JOINT_ID)
    smooth = _velocity_jerk_score(pelvis)

    osim_conf, ik_score = _opensim_marker_confidence(sequence, sequence.fps)

    return BackendComparisonMetrics(
        backend_name=backend_name,
        available=True,
        frame_count=len(sequence.frames),
        valid_frame_ratio=sequence.valid_frame_ratio,
        mean_landmark_confidence=statistics.mean(confs) if confs else 0.0,
        trajectory_smoothness_score=smooth,
        joint_angle_consistency_score=_angle_consistency(sequence),
        foot_ground_jitter_mm=_foot_ground_jitter(sequence),
        pelvis_trajectory_consistency=_pelvis_consistency(sequence),
        opensim_marker_confidence_mean=round(osim_conf, 3) if osim_conf is not None else None,
        opensim_ik_readiness_score=ik_score,
    )
