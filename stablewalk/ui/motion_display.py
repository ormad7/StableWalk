"""
Format per-frame motion data for GUI tables and reports (3D positions, DoF, velocities).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stablewalk.pose.dof import DOF_LABELS, GAIT_ANGLE_FIELDS, GAIT_VELOCITY_JOINTS
from stablewalk.pose.kinematics import dof_angular_velocities, velocity_between_frames
from stablewalk.models.pose_data import PoseFrame
from stablewalk.pose.skeleton_3d import SKELETON_3D_JOINTS, Skeleton3D


@dataclass
class FrameMotionSnapshot:
    """Structured motion state for one pose frame."""

    frame_index: int
    positions_3d: list[tuple[str, float, float, float]]
    dof_rows: list[tuple[str, str, str, str, str]]  # label, angle, omega, rom, speed
    velocity_rows: list[tuple[str, str, str, str, str]]  # joint, vx, vy, speed, dir
    summary_lines: list[str]


def _fmt(v: float | None, *, unit: str = "", decimals: int = 2) -> str:
    if v is None:
        return "—"
    return f"{v:.{decimals}f}{unit}"


def _vel_direction_deg(vx: float, vy: float) -> float | None:
    if abs(vx) < 1e-8 and abs(vy) < 1e-8:
        return None
    # Image plane: x right, y down → compass-style degrees (0 = right, 90 = down)
    return math.degrees(math.atan2(vy, vx))


def compute_sequence_rom(
    frames: list[PoseFrame],
    pose_indices: list[int],
) -> dict[str, tuple[float, float]]:
    """Min/max angle (°) per DoF across all detected pose frames."""
    rom: dict[str, tuple[float, float]] = {}
    for name in GAIT_ANGLE_FIELDS:
        vals: list[float] = []
        for idx in pose_indices:
            f = frames[idx]
            if not f.detected or not f.joint_angles:
                continue
            v = getattr(f.joint_angles, name, None)
            if v is not None:
                vals.append(float(v))
        if vals:
            rom[name] = (min(vals), max(vals))
    return rom


def _collect_joint_positions(
    frame: PoseFrame,
    skeleton: Skeleton3D | None,
    *,
    prefer_frame: bool = True,
) -> list[tuple[str, float, float, float]]:
    """Joint XYZ for tables — frame-enriched data first (stable during playback)."""
    positions: list[tuple[str, float, float, float]] = []

    def _from_dict(source: dict) -> None:
        for name in SKELETON_3D_JOINTS:
            raw = source.get(name)
            if raw is None:
                continue
            if isinstance(raw, dict):
                positions.append(
                    (
                        name,
                        float(raw.get("x", 0)),
                        float(raw.get("y", 0)),
                        float(raw.get("z", 0)),
                    )
                )

    def _from_skeleton() -> None:
        if not skeleton or not skeleton.joints:
            return
        for name in SKELETON_3D_JOINTS:
            j = skeleton.joints.get(name)
            if j:
                positions.append((name, j.x, j.y, j.z))

    if prefer_frame:
        if frame.positions_normalized:
            _from_dict(frame.positions_normalized)
        if positions:
            return positions
        sk = frame.skeleton_3d or {}
        joints_blob = sk.get("joints") if isinstance(sk, dict) else None
        if joints_blob:
            _from_dict(joints_blob)
        if positions:
            return positions
        if frame.positions:
            _from_dict(frame.positions)
        if positions:
            return positions

    _from_skeleton()
    if positions:
        return positions

    if frame.positions_normalized:
        _from_dict(frame.positions_normalized)
    if positions:
        return positions

    sk = frame.skeleton_3d or {}
    joints_blob = sk.get("joints") if isinstance(sk, dict) else None
    if joints_blob:
        _from_dict(joints_blob)
    if positions:
        return positions

    if frame.positions:
        _from_dict(frame.positions)
    return positions


def build_frame_motion_snapshot(
    frame: PoseFrame,
    *,
    skeleton: Skeleton3D | None,
    prev_frame: PoseFrame | None,
    fps: float,
    rom_cache: dict[str, tuple[float, float]] | None = None,
) -> FrameMotionSnapshot:
    """Collect 3D joint positions, DoF angles/velocities, and biomechanical summary."""
    positions = _collect_joint_positions(frame, skeleton)

    ang_vel: dict[str, float] = {}
    lin_scalar: dict[str, float] = {}
    lin_vec: dict[str, dict[str, float]] = {}
    if prev_frame and prev_frame.detected and frame.detected:
        ang_vel = dof_angular_velocities(prev_frame, frame, fps)
        lin_vec, lin_scalar = velocity_between_frames(prev_frame, frame, fps)
    elif frame.velocity_scalar:
        lin_scalar = dict(frame.velocity_scalar)
        lin_vec = dict(frame.velocities)

    rom_cache = rom_cache or {}

    dof_rows: list[tuple[str, str, str, str, str]] = []
    angles = frame.joint_angles
    if angles:
        for name in GAIT_ANGLE_FIELDS:
            label = DOF_LABELS.get(name, name)
            angle = getattr(angles, name, None)
            omega = ang_vel.get(name)
            lo, hi = rom_cache.get(name, (None, None))
            if lo is not None and hi is not None:
                rom_txt = f"{lo:.0f}–{hi:.0f}°"
            else:
                rom_txt = "—"
            lin = lin_scalar.get(name) if name in GAIT_VELOCITY_JOINTS else None
            if name in ("left_ankle_flexion", "right_ankle_flexion"):
                lin = lin_scalar.get(name.replace("_flexion", ""), lin)
            dof_rows.append(
                (
                    label,
                    _fmt(angle, unit="°", decimals=1),
                    _fmt(omega, unit="°/s", decimals=1),
                    rom_txt,
                    _fmt(lin, decimals=3),
                )
            )

    velocity_rows: list[tuple[str, str, str, str, str]] = []
    for jname in SKELETON_3D_JOINTS:
        vel = lin_vec.get(jname)
        if not vel:
            continue
        vx = float(vel.get("vx", 0.0))
        vy = float(vel.get("vy", 0.0))
        spd = float(vel.get("speed", 0.0))
        direction = _vel_direction_deg(vx, vy)
        velocity_rows.append(
            (
                jname.replace("_", " "),
                _fmt(vx, decimals=3),
                _fmt(vy, decimals=3),
                _fmt(spd, decimals=3),
                _fmt(direction, unit="°", decimals=0),
            )
        )

    phase = frame.gait_phase or {}
    events = ", ".join(frame.gait_events) if frame.gait_events else "none"

    move_dir = "—"
    hip_vel = lin_vec.get("mid_hip") or lin_vec.get("left_hip")
    if hip_vel:
        vx = float(hip_vel.get("vx", 0.0))
        vy = float(hip_vel.get("vy", 0.0))
        deg = _vel_direction_deg(vx, vy)
        spd = float(hip_vel.get("speed", 0.0))
        if deg is not None:
            move_dir = f"{deg:.0f}° in image plane  |  speed {spd:.3f}"

    summary_lines = [
        f"Frame {frame.frame_index + 1}  (analyzed independently)",
        f"Gait phase   L: {phase.get('left', '—')}   R: {phase.get('right', '—')}",
        f"Events: {events}",
        f"Movement direction (pelvis): {move_dir}",
    ]
    if angles:
        summary_lines.append(
            f"Degrees of freedom: {angles.gait_dof_count()}/{len(GAIT_ANGLE_FIELDS)} measured"
        )
    if skeleton:
        summary_lines.append(f"3D joints: {len(positions)}  ·  coords hip-centered, Y up")
    if rom_cache:
        summary_lines.append(f"ROM spans full walk ({len(rom_cache)} DoF tracked)")

    return FrameMotionSnapshot(
        frame_index=frame.frame_index,
        positions_3d=positions,
        dof_rows=dof_rows,
        velocity_rows=velocity_rows,
        summary_lines=summary_lines,
    )
