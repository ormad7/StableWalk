"""
Whole-body Center of Mass estimation from segment anthropometry.

Uses de Leva-style segment mass fractions with joint midpoints as segment COM
proxies. Values are **estimated** — not force-plate or OpenSim ID output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from stablewalk.analysis.biomechanical.types import BiomechanicalFrameCOM
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3

logger = logging.getLogger(__name__)

# Segment mass as fraction of total body mass; COM at fraction from proximal joint.
_SEGMENT_SPECS: tuple[tuple[str, str, str, float, float], ...] = (
    ("head", "neck", "head", 0.081, 0.50),
    ("trunk_upper", "spine", "neck", 0.150, 0.50),
    ("trunk_lower", "left_hip", "right_hip", 0.347, 0.50),
    ("left_thigh", "left_hip", "left_knee", 0.100, 0.433),
    ("right_thigh", "right_hip", "right_knee", 0.100, 0.433),
    ("left_shank", "left_knee", "left_ankle", 0.0465, 0.433),
    ("right_shank", "right_knee", "right_ankle", 0.0465, 0.433),
    ("left_foot", "left_ankle", "left_toe", 0.0145, 0.50),
    ("right_foot", "right_ankle", "right_toe", 0.0145, 0.50),
    ("left_arm", "left_shoulder", "left_wrist", 0.028, 0.436),
    ("right_arm", "right_shoulder", "right_wrist", 0.028, 0.436),
)


def _joint_pos(snap: SkeletonSnapshot, jid: str) -> Vec3 | None:
    sample = snap.joints.get(jid)
    if sample is None:
        return None
    return sample.position


def _segment_com(
    snap: SkeletonSnapshot,
    proximal: str,
    distal: str,
    com_fraction: float,
) -> tuple[float, float, float, float] | None:
    """Return (x, y, z, confidence) for one segment COM."""
    p = _joint_pos(snap, proximal)
    d = _joint_pos(snap, distal)
    if p is None or d is None:
        return None
    t = max(0.0, min(1.0, com_fraction))
    x = p.x + t * (d.x - p.x)
    y = p.y + t * (d.y - p.y)
    z = p.z + t * (d.z - p.z)
    return x, y, z, 0.85


def estimate_frame_com(snap: SkeletonSnapshot) -> tuple[tuple[float, float, float], float]:
    """
    Estimate whole-body COM for one skeleton snapshot.

    Returns ((x,y,z), confidence).
    """
    masses: list[float] = []
    positions: list[tuple[float, float, float]] = []
    confidences: list[float] = []

    for _name, prox, dist, mass_frac, com_frac in _SEGMENT_SPECS:
        seg = _segment_com(snap, prox, dist, com_frac)
        if seg is None:
            continue
        x, y, z, conf = seg
        masses.append(mass_frac)
        positions.append((x, y, z))
        confidences.append(conf)

    if not masses:
        pelvis = _joint_pos(snap, "left_hip")
        rh = _joint_pos(snap, "right_hip")
        if pelvis and rh:
            cx = (pelvis.x + rh.x) * 0.5
            cy = (pelvis.y + rh.y) * 0.5
            cz = (pelvis.z + rh.z) * 0.5
            return (cx, cy, cz), 0.35
        return (0.0, 0.0, 0.0), 0.0

    total_m = sum(masses)
    cx = sum(p[0] * m for p, m in zip(positions, masses)) / total_m
    cy = sum(p[1] * m for p, m in zip(positions, masses)) / total_m
    cz = sum(p[2] * m for p, m in zip(positions, masses)) / total_m
    conf = float(np.mean(confidences)) * min(1.0, len(masses) / len(_SEGMENT_SPECS))
    return (cx, cy, cz), conf


def _kinematics_1d(values: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    vel = np.zeros(n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)
    if n < 2:
        return vel, acc
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 1e-9:
            continue
        vel[i] = (values[i] - values[i - 1]) / dt
    for i in range(2, n):
        dt = times[i] - times[i - 1]
        if dt <= 1e-9:
            continue
        acc[i] = (vel[i] - vel[i - 1]) / dt
    return vel, acc


@dataclass
class CenterOfMassAnalysis:
    """Per-frame COM trajectories (estimated)."""

    per_frame: list[BiomechanicalFrameCOM] = field(default_factory=list)
    fps: float = 30.0
    kind: str = "estimated"

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([f.time_s for f in self.per_frame], dtype=np.float64)

    @property
    def positions(self) -> np.ndarray:
        return np.array([f.position for f in self.per_frame], dtype=np.float64)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "fps": self.fps,
            "frame_count": len(self.per_frame),
            "note": (
                "Segment-weighted COM from pose landmarks — estimated, not measured."
            ),
        }


def analyze_center_of_mass(
    recording: GaitMotionRecording,
    contact: FootContactAnalysisResult | None = None,
) -> CenterOfMassAnalysis:
    """Estimate COM position, velocity, and acceleration for every frame."""
    fps = max(recording.fps, 1e-6)
    frame_indices: list[int] = []
    times: list[float] = []
    positions: list[tuple[float, float, float]] = []
    confidences: list[float] = []

    if contact and contact.per_frame:
        for cf in contact.per_frame:
            snap = recording.snapshot_at(cf.frame_index)
            if snap is None:
                continue
            pos, conf = estimate_frame_com(snap)
            frame_indices.append(cf.frame_index)
            times.append(cf.time_s)
            positions.append(pos)
            confidences.append(conf)
    else:
        for i in range(recording.frame_count):
            snap = recording.snapshot_at(i)
            if snap is None:
                continue
            pos, conf = estimate_frame_com(snap)
            frame_indices.append(snap.frame_index)
            times.append(snap.time_s)
            positions.append(pos)
            confidences.append(conf)

    if not times:
        return CenterOfMassAnalysis(fps=fps)

    t_arr = np.array(times, dtype=np.float64)
    xs = np.array([p[0] for p in positions], dtype=np.float64)
    ys = np.array([p[1] for p in positions], dtype=np.float64)
    zs = np.array([p[2] for p in positions], dtype=np.float64)

    vx, ax = _kinematics_1d(xs, t_arr)
    vy, ay = _kinematics_1d(ys, t_arr)
    vz, az = _kinematics_1d(zs, t_arr)

    per_frame: list[BiomechanicalFrameCOM] = []
    for i, fi in enumerate(frame_indices):
        per_frame.append(
            BiomechanicalFrameCOM(
                frame_index=fi,
                time_s=float(times[i]),
                position=(float(xs[i]), float(ys[i]), float(zs[i])),
                velocity=(float(vx[i]), float(vy[i]), float(vz[i])),
                acceleration=(float(ax[i]), float(ay[i]), float(az[i])),
                confidence=float(confidences[i]),
            )
        )

    return CenterOfMassAnalysis(per_frame=per_frame, fps=fps)


__all__ = [
    "CenterOfMassAnalysis",
    "analyze_center_of_mass",
    "estimate_frame_com",
]
