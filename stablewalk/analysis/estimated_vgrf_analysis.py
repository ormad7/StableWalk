"""
Estimated virtual ground reaction force (vGRF) analysis.

Produces pose-based vertical force proxies labeled ``estimated_vgrf`` — not
instrumented kinetics, force-plate measurements, or PhysX ContactSensor output.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.ground_reference import vertical_coordinate
from stablewalk.analysis.virtual_grf import G, SCIENTIFIC_DISCLAIMER, VGRF_TERMINOLOGY
from stablewalk.models.gait_motion import GaitMotionRecording

logger = logging.getLogger(__name__)

METHOD_NAME = "estimated_vgrf"


@dataclass
class EstimatedVGRFMetrics:
    peak_force_n: float = 0.0
    peak_force_bw: float = 0.0
    loading_rate_n_per_s: float = 0.0
    impulse_n_s: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EstimatedVGRFResult:
    """Time-series estimated vertical GRF with summary metrics."""

    method_name: str = METHOD_NAME
    terminology: str = VGRF_TERMINOLOGY
    scientific_disclaimer: str = SCIENTIFIC_DISCLAIMER
    body_mass_kg: float = 70.0
    body_weight_n: float = 70.0 * G
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    left_vgrf_vertical: np.ndarray = field(default_factory=lambda: np.array([]))
    right_vgrf_vertical: np.ndarray = field(default_factory=lambda: np.array([]))
    total_vgrf_vertical: np.ndarray = field(default_factory=lambda: np.array([]))
    left_vgrf_bw: np.ndarray = field(default_factory=lambda: np.array([]))
    right_vgrf_bw: np.ndarray = field(default_factory=lambda: np.array([]))
    com_accel_z: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence: np.ndarray = field(default_factory=lambda: np.array([]))
    metrics: EstimatedVGRFMetrics = field(default_factory=EstimatedVGRFMetrics)
    available: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "terminology": self.terminology,
            "scientific_disclaimer": self.scientific_disclaimer,
            "body_mass_kg": self.body_mass_kg,
            "available": self.available,
            "metrics": self.metrics.to_dict(),
            "sample_count": int(len(self.timestamps)),
            "notes": list(self.notes),
        }


def _pelvis_height_m(snapshot, *, axis: str = "y") -> float | None:
    lh = snapshot.joints.get("left_hip")
    rh = snapshot.joints.get("right_hip")
    if lh is None or rh is None:
        root = snapshot.joints.get("pelvis")
        if root is None:
            return None
        return vertical_coordinate(root.position, axis=axis)
    mid_y = (lh.position.y + rh.position.y) * 0.5
    if axis == "y":
        return mid_y
    if axis == "z":
        return (lh.position.z + rh.position.z) * 0.5
    return (lh.position.x + rh.position.x) * 0.5


def _vertical_kinematics(
    heights: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(heights)
    vel = np.zeros(n, dtype=np.float64)
    acc = np.zeros(n, dtype=np.float64)
    if n < 2:
        return vel, acc
    for i in range(1, n):
        dt = times[i] - times[i - 1]
        if dt <= 1e-9:
            continue
        vel[i] = (heights[i] - heights[i - 1]) / dt
    for i in range(2, n):
        dt = times[i] - times[i - 1]
        if dt <= 1e-9:
            continue
        acc[i] = (vel[i] - vel[i - 1]) / dt
    return vel, acc


def _allocate_total_force(
    f_total: float,
    left_contact: bool,
    right_contact: bool,
) -> tuple[float, float]:
    if f_total <= 0.0:
        return 0.0, 0.0
    if left_contact and right_contact:
        return f_total * 0.5, f_total * 0.5
    if left_contact:
        return f_total, 0.0
    if right_contact:
        return 0.0, f_total
    return 0.0, 0.0


def _loading_rate(force: np.ndarray, times: np.ndarray) -> float:
    if len(force) < 2:
        return 0.0
    best = 0.0
    for i in range(1, len(force)):
        dt = times[i] - times[i - 1]
        if dt <= 1e-9:
            continue
        rate = (force[i] - force[i - 1]) / dt
        if rate > best:
            best = rate
    return float(best)


def analyze_estimated_vgrf(
    recording: GaitMotionRecording,
    contact: FootContactAnalysisResult,
    *,
    body_mass_kg: float = 70.0,
    vertical_axis: str = "y",
) -> EstimatedVGRFResult:
    """
    Estimate vertical vGRF from body mass, gravity, CoM acceleration, and contact.
    """
    notes: list[str] = [
        "Estimated Virtual GRF — not measured kinetics or PhysX contact forces.",
    ]
    bw = body_mass_kg * G

    if not contact.per_frame:
        return EstimatedVGRFResult(
            body_mass_kg=body_mass_kg,
            body_weight_n=bw,
            notes=notes + ["No contact frames — vGRF unavailable."],
        )

    frame_indices = [f.frame_index for f in contact.per_frame]
    times = np.array([f.time_s for f in contact.per_frame], dtype=np.float64)
    heights: list[float] = []
    for fi in frame_indices:
        snap = recording.snapshot_at(fi)
        if snap is None:
            heights.append(np.nan)
            continue
        h = _pelvis_height_m(snap, axis=vertical_axis)
        heights.append(float(h) if h is not None else np.nan)

    h_arr = np.array(heights, dtype=np.float64)
    valid = np.isfinite(h_arr)
    if valid.sum() < 3:
        return EstimatedVGRFResult(
            body_mass_kg=body_mass_kg,
            body_weight_n=bw,
            timestamps=times,
            notes=notes + ["Insufficient pelvis height for CoM proxy."],
        )

    # Fill NaNs for gradient stability
    if not valid.all():
        idx = np.arange(len(h_arr))
        h_arr = np.interp(idx, idx[valid], h_arr[valid])

    _, acc_z = _vertical_kinematics(h_arr, times)

    left_c = contact.left_contact_binary.astype(bool)
    right_c = contact.right_contact_binary.astype(bool)
    left_prob = contact.left_contact_probability
    right_prob = contact.right_contact_probability

    n = len(times)
    left_n = np.zeros(n, dtype=np.float64)
    right_n = np.zeros(n, dtype=np.float64)
    conf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if not left_c[i] and not right_c[i]:
            continue
        f_total = body_mass_kg * (float(acc_z[i]) + G)
        f_total = max(0.0, min(f_total, 2.5 * bw))
        fl, fr = _allocate_total_force(f_total, bool(left_c[i]), bool(right_c[i]))
        left_n[i] = fl
        right_n[i] = fr
        frame_conf = contact.per_frame[i]
        conf[i] = min(
            1.0,
            contact.metrics.contact_confidence
            * 0.5
            + frame_conf.left_confidence * left_prob[i] * 0.25
            + frame_conf.right_confidence * right_prob[i] * 0.25,
        )

    total_n = left_n + right_n
    left_bw = left_n / bw
    right_bw = right_n / bw

    dt_mean = float(np.mean(np.diff(times))) if len(times) > 1 else 1.0 / max(contact.fps, 1e-6)
    impulse = float(np.sum(total_n) * dt_mean)
    peak_n = float(np.max(total_n)) if len(total_n) else 0.0
    load_rate = _loading_rate(total_n, times)
    mean_conf = float(np.mean(conf[conf > 0])) if (conf > 0).any() else contact.metrics.contact_confidence * 0.5

    metrics = EstimatedVGRFMetrics(
        peak_force_n=peak_n,
        peak_force_bw=peak_n / bw if bw > 0 else 0.0,
        loading_rate_n_per_s=load_rate,
        impulse_n_s=impulse,
        confidence=mean_conf,
    )

    return EstimatedVGRFResult(
        method_name=METHOD_NAME,
        body_mass_kg=body_mass_kg,
        body_weight_n=bw,
        timestamps=times,
        left_vgrf_vertical=left_n,
        right_vgrf_vertical=right_n,
        total_vgrf_vertical=total_n,
        left_vgrf_bw=left_bw,
        right_vgrf_bw=right_bw,
        com_accel_z=acc_z,
        confidence=conf,
        metrics=metrics,
        available=True,
        notes=notes,
    )


__all__ = [
    "METHOD_NAME",
    "EstimatedVGRFMetrics",
    "EstimatedVGRFResult",
    "analyze_estimated_vgrf",
]
