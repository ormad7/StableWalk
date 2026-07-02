"""
Export gait analysis: keypoints, angles, velocities (JSON / CSV).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from stablewalk.gait_dof import GAIT_ANGLE_FIELDS, GAIT_VELOCITY_JOINTS
from stablewalk.kinematics import dof_angular_velocities, velocity_between_frames
from stablewalk.models.pose_data import PoseSequence
from stablewalk.skeleton_3d_model import SKELETON_3D_JOINTS, skeleton_from_frame_data
from stablewalk.visualization import detected_frame_indices


def export_analysis_json(sequence: PoseSequence, path: str | Path) -> Path:
    """Full structured export (same schema as pipeline JSON)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sequence.to_dict(), f, indent=2)
    return path


def export_analysis_csv(sequence: PoseSequence, path: str | Path) -> Path:
    """
    Flat CSV: one row per detected frame with angles and velocities.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    angle_cols = list(GAIT_ANGLE_FIELDS)
    omega_cols = [f"omega_{n}" for n in GAIT_ANGLE_FIELDS]
    vel_cols = [f"vel_{j}" for j in GAIT_VELOCITY_JOINTS]
    pos_cols = [f"{j}_{ax}" for j in SKELETON_3D_JOINTS for ax in ("x", "y", "z")]

    fieldnames = (
        ["frame_index", "detected", "gait_phase_left", "gait_phase_right", "gait_events"]
        + angle_cols
        + omega_cols
        + vel_cols
        + pos_cols
    )

    indices = detected_frame_indices(sequence)
    rows: list[dict[str, object]] = []
    prev = None
    for idx in indices:
        f = sequence.frames[idx]
        row: dict[str, object] = {
            "frame_index": f.frame_index,
            "detected": f.detected,
            "gait_phase_left": f.gait_phase.get("left", ""),
            "gait_phase_right": f.gait_phase.get("right", ""),
            "gait_events": ", ".join(f.gait_events),
        }
        if f.joint_angles:
            for name in angle_cols:
                row[name] = getattr(f.joint_angles, name, None)
        if prev and prev.detected and f.detected:
            for name, omega in dof_angular_velocities(prev, f, sequence.fps).items():
                row[f"omega_{name}"] = omega
            _, scalar = velocity_between_frames(prev, f, sequence.fps)
            for j in GAIT_VELOCITY_JOINTS:
                row[f"vel_{j}"] = scalar.get(j)
        else:
            for j in GAIT_VELOCITY_JOINTS:
                row[f"vel_{j}"] = f.velocity_scalar.get(j)
        if f.keypoints:
            skel = skeleton_from_frame_data(f.keypoints, f.skeleton_3d, 1.0)
            for jname, j in skel.joints.items():
                row[f"{jname}_x"] = j.x
                row[f"{jname}_y"] = j.y
                row[f"{jname}_z"] = j.z
        rows.append(row)
        if f.detected:
            prev = f

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_analysis(
    sequence: PoseSequence,
    path: str | Path,
    *,
    fmt: str = "json",
) -> Path:
    """
    Export analysis to JSON or CSV based on extension or fmt argument.
    """
    path = Path(path)
    ext = path.suffix.lower()
    if fmt == "csv" or ext == ".csv":
        if ext != ".csv":
            path = path.with_suffix(".csv")
        return export_analysis_csv(sequence, path)
    if ext != ".json":
        path = path.with_suffix(".json")
    return export_analysis_json(sequence, path)
