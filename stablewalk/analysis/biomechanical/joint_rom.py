"""Joint range-of-motion analysis per gait cycle."""

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


@dataclass
class JointROMAnalysis:
    joints: list[JointROMStats] = field(default_factory=list)
    kind: str = "estimated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "joints": [j.to_dict() for j in self.joints],
            "note": "Pose-derived joint angles — estimated, not goniometer measured.",
        }


def _angle_series(
    sequence: PoseSequence | None,
    recording: GaitMotionRecording,
    field_name: str,
) -> list[float | None]:
    """Collect per-frame angle (degrees) from pose sequence or recording DOFs."""
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
            val = getattr(angles, field_key, None) or getattr(angles, field_name, None)
            out.append(float(val) if val is not None else None)
        return out

    out = []
    for i in range(recording.frame_count):
        snap = recording.snapshot_at(i)
        if snap is None:
            out.append(None)
            continue
        # Try DOF samples
        dof_key = field_name.replace("_angle", "")
        sample = snap.dofs.get(dof_key) or snap.dofs.get(field_name)
        if sample and sample.angle_deg is not None:
            out.append(float(sample.angle_deg))
        else:
            out.append(None)
    return out


def _rom_stats(values: list[float], *, joint: str, side: str, confidence: float) -> JointROMStats:
    if not values:
        return JointROMStats(joint=joint, side=side, confidence=0.0)
    vmin, vmax = min(values), max(values)
    rom = vmax - vmin
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    # Flexion/extension labeling: positive flexion convention for sagittal joints
    return JointROMStats(
        joint=joint,
        side=side,
        flexion_min_deg=vmin,
        flexion_max_deg=vmax,
        rom_deg=rom,
        mean_deg=mean,
        std_deg=std,
        confidence=confidence,
        kind="estimated",
    )


def _cycle_rom_values(
    angles: list[float | None],
    cycles: GaitCycleAnalysisResult,
    frame_indices: list[int],
) -> list[float]:
    """ROM within each detected gait cycle."""
    idx_map = {fi: i for i, fi in enumerate(frame_indices)}
    roms: list[float] = []
    for cycle in cycles.cycles:
        vals: list[float] = []
        for fi in range(cycle.start_frame, cycle.end_frame + 1):
            pos = idx_map.get(fi)
            if pos is None or pos >= len(angles):
                continue
            a = angles[pos]
            if a is not None:
                vals.append(a)
        if len(vals) >= 3:
            roms.append(max(vals) - min(vals))
    return roms


def analyze_joint_rom(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult | None,
    *,
    sequence: PoseSequence | None = None,
    contact_confidence: float = 0.5,
) -> JointROMAnalysis:
    """Compute hip/knee/ankle ROM statistics per side."""
    conf = max(0.35, min(0.95, contact_confidence))
    joints_out: list[JointROMStats] = []

    angle_fields = (
        ("hip", "left_hip_angle", "right_hip_angle"),
        ("knee", "left_knee_angle", "right_knee_angle"),
        ("ankle", "left_ankle_angle", "right_ankle_angle"),
    )

    frame_indices = [s.frame_index for s in recording.snapshots]

    for joint_name, left_field, right_field in angle_fields:
        for side, field in (("left", left_field), ("right", right_field)):
            series = _angle_series(sequence, recording, field)
            valid = [v for v in series if v is not None]
            stats = _rom_stats(valid, joint=joint_name, side=side, confidence=conf)
            joints_out.append(stats)

    return JointROMAnalysis(joints=joints_out)


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
