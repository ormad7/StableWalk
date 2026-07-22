"""Joint range-of-motion analysis per gait cycle.

Pose ``compute_joint_angles`` returns *interior* angles (0–180°). Sagittal
ROM is reported in the flexion convention used by Motion Analysis charts:
``flexion = 180 − interior`` (0° = full extension). ROM uses a robust
percentile span so single-frame collapses do not invent 160°+ walking ROM.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.biomechanical.types import JointROMStats
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.kinematics import compute_joint_angles

# Physiological flexion windows used to reject impossible samples (degrees).
_PHYSIO_FLEXION: dict[str, tuple[float, float]] = {
    "knee": (-5.0, 145.0),
    "hip": (-20.0, 120.0),
    # Ankle kept as pose interior angle span (not 180−θ).
    "ankle": (20.0, 180.0),
}
# Walking-typical upper bound for mean cycle ROM; above this confidence drops.
_WALK_ROM_SOFT_MAX: dict[str, float] = {
    "knee": 95.0,
    "hip": 70.0,
    "ankle": 55.0,
}


@dataclass
class JointROMAnalysis:
    joints: list[JointROMStats] = field(default_factory=list)
    kind: str = "estimated"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "joints": [j.to_dict() for j in self.joints],
            "note": (
                "Pose-derived flexion ROM (180° − interior angle) — estimated, "
                "not goniometer measured. Robust P5–P95 span per cycle."
            ),
            "warnings": list(self.warnings),
        }


def _interior_to_flexion(interior: float) -> float:
    return float(180.0 - interior)


def _to_flexion_series(
    interiors: list[float | None],
    *,
    joint: str,
) -> list[float | None]:
    """Map pose angles to a consistent flexion series for ROM.

    Knee/hip: ``flexion = 180 − interior`` (0° = extension), matching
    Motion Analysis charts. Ankle: keep the measured interior angle (pose
    foot landmark geometry is not a clean dorsi/plantar convention).
    """
    lo, hi = _PHYSIO_FLEXION.get(joint, (-180.0, 180.0))
    convert = joint in ("knee", "hip")
    out: list[float | None] = []
    for val in interiors:
        if val is None or not np.isfinite(val) or not (-180.0 <= val <= 180.0):
            out.append(None)
            continue
        flex = _interior_to_flexion(float(val)) if convert else float(val)
        if lo <= flex <= hi:
            out.append(flex)
        else:
            out.append(None)
    return out


def _robust_span(values: list[float]) -> tuple[float, float, float]:
    """Return (low, high, rom) resistant to single-frame spikes.

    Uses inner percentiles plus a median-absolute-deviation gate so one
    collapsed pose cannot invent 160° walking ROM.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    if arr.size < 5:
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        return lo, hi, hi - lo
    lo = float(np.percentile(arr, 10))
    hi = float(np.percentile(arr, 90))
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-6:
        mad = 1.0
    gate = 3.5 * mad
    lo = max(lo, med - gate)
    hi = min(hi, med + gate)
    if hi < lo:
        lo, hi = float(np.min(arr)), float(np.max(arr))
    return lo, hi, max(0.0, hi - lo)


def _angle_series(
    sequence: PoseSequence | None,
    recording: GaitMotionRecording,
    field_name: str,
) -> list[float | None]:
    """Collect per-frame *interior* angle (degrees) from pose or recording DOFs."""
    if sequence is not None:
        out: list[float | None] = []
        for frame in sequence.frames:
            if not frame.detected:
                out.append(None)
                continue
            angles = compute_joint_angles(frame.keypoints) if frame.keypoints else None
            if angles is None:
                out.append(None)
                continue
            field_key = field_name.replace("_angle", "")
            val = getattr(angles, field_key, None)
            if val is None:
                val = getattr(angles, field_name, None)
            out.append(float(val) if val is not None else None)
        return out

    out = []
    for i in range(recording.frame_count):
        snap = recording.snapshot_at(i)
        if snap is None:
            out.append(None)
            continue
        dof_key = field_name.replace("_angle", "")
        sample = snap.dofs.get(dof_key) or snap.dofs.get(field_name)
        if sample and sample.angle_deg is not None:
            out.append(float(sample.angle_deg))
        else:
            out.append(None)
    return out


def _rom_stats(
    cycle_values: list[list[float]],
    *,
    joint: str,
    side: str,
    confidence: float,
    unavailable_reason: str = "",
) -> JointROMStats:
    if not cycle_values:
        return JointROMStats(
            joint=joint,
            side=side,
            confidence=0.0,
            note=f"N/A — {unavailable_reason or 'no complete cycle had sufficient angle coverage.'}",
        )
    minima: list[float] = []
    maxima: list[float] = []
    roms: list[float] = []
    all_values: list[float] = []
    for values in cycle_values:
        lo, hi, rom = _robust_span(values)
        minima.append(lo)
        maxima.append(hi)
        roms.append(rom)
        all_values.extend(values)

    mean_rom = float(statistics.mean(roms))
    soft_max = _WALK_ROM_SOFT_MAX.get(joint)
    conf = confidence
    note_extra = ""
    if soft_max is not None and mean_rom > soft_max:
        # Do not invent a smaller ROM — lower confidence so UI can show caution.
        conf = min(conf, 0.35)
        note_extra = (
            f" Mean cycle ROM {mean_rom:.0f}° exceeds typical walking "
            f"({soft_max:.0f}°) — interpret with low confidence."
        )

    return JointROMStats(
        joint=joint,
        side=side,
        flexion_min_deg=statistics.mean(minima),
        flexion_max_deg=statistics.mean(maxima),
        rom_deg=mean_rom,
        mean_deg=statistics.mean(all_values),
        std_deg=statistics.stdev(all_values) if len(all_values) > 1 else 0.0,
        confidence=conf,
        kind="estimated",
        note=(
            f"Mean per-cycle flexion ROM (P5–P95) from {len(cycle_values)} "
            f"complete gait cycle(s); flexion = 180° − pose interior."
            f"{note_extra}"
        ),
    )


def _cycle_angle_values(
    angles: list[float | None],
    cycles: GaitCycleAnalysisResult,
    frame_indices: list[int],
) -> list[list[float]]:
    """Finite physiological flexion samples grouped by adequately covered cycles."""
    idx_map = {fi: i for i, fi in enumerate(frame_indices)}
    cycle_values: list[list[float]] = []
    for cycle in cycles.cycles:
        expected = 0
        vals: list[float] = []
        for fi in range(cycle.start_frame, cycle.end_frame + 1):
            pos = idx_map.get(fi)
            if pos is None or pos >= len(angles):
                continue
            expected += 1
            a = angles[pos]
            if a is not None and np.isfinite(a):
                vals.append(float(a))
        if expected >= 5 and len(vals) >= 5 and len(vals) / expected >= 0.60:
            cycle_values.append(vals)
    return cycle_values


def analyze_joint_rom(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult | None,
    *,
    sequence: PoseSequence | None = None,
    contact_confidence: float = 0.5,
) -> JointROMAnalysis:
    """Compute hip/knee/ankle flexion ROM statistics per side."""
    reliable_cycles = bool(cycles and cycles.metrics.metrics_reliable)
    conf = min(0.95, max(0.0, contact_confidence)) if reliable_cycles else 0.0
    joints_out: list[JointROMStats] = []
    warnings: list[str] = []
    unavailable_reason = (
        cycles.metrics.reliability_reason
        if cycles is not None
        else "no gait-cycle analysis was available."
    )
    if not reliable_cycles:
        warnings.append(f"Joint ROM unavailable: {unavailable_reason}")

    angle_fields = (
        ("hip", "left_hip_angle", "right_hip_angle"),
        ("knee", "left_knee_angle", "right_knee_angle"),
        ("ankle", "left_ankle_angle", "right_ankle_angle"),
    )

    frame_indices = [s.frame_index for s in recording.snapshots]

    for joint_name, left_field, right_field in angle_fields:
        for side, field in (("left", left_field), ("right", right_field)):
            interiors = _angle_series(sequence, recording, field)
            flexion = _to_flexion_series(interiors, joint=joint_name)
            grouped = (
                _cycle_angle_values(flexion, cycles, frame_indices)
                if reliable_cycles and cycles is not None
                else []
            )
            stats = _rom_stats(
                grouped,
                joint=joint_name,
                side=side,
                confidence=conf,
                unavailable_reason=unavailable_reason,
            )
            if reliable_cycles and not grouped:
                warnings.append(
                    f"{side.title()} {joint_name} ROM unavailable: no complete cycle "
                    "had at least 60% valid flexion coverage."
                )
            if stats.rom_deg is not None and stats.confidence < 0.4 and stats.rom_deg > 0:
                warnings.append(
                    f"{side.title()} {joint_name} ROM {stats.rom_deg:.0f}° has low "
                    f"confidence ({stats.confidence:.0%})."
                )
            joints_out.append(stats)

    return JointROMAnalysis(joints=joints_out, warnings=warnings)


def extract_side_rom(
    rom_analysis: JointROMAnalysis,
) -> dict[str, float | None]:
    """Return left/right ROM by joint for symmetry module."""
    out: dict[str, float | None] = {}
    for j in rom_analysis.joints:
        key = f"{j.joint}_rom_{j.side}"
        out[key] = j.rom_deg
    return out


__all__ = ["JointROMAnalysis", "analyze_joint_rom", "extract_side_rom"]
