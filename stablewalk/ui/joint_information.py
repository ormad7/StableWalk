"""
Live Joint Information panel — display-only kinematics synchronized with playback.

Does not alter trajectory or biomechanics data; formats existing samples for UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.dof_position_table import (
    NA_VALUE,
    _ITEM_DOF_ID,
    _foot_table_cells,
    angle_value_for_item,
)
from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item
from stablewalk.ui.joint_data_table import confidence_for_joint_at_frame
from stablewalk.ui.selected_point_analysis import is_foot_analysis_point
from stablewalk.ui.viewers.dof_trajectory_3d import _joint_angle_window_stats

if TYPE_CHECKING:
    from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot
    from stablewalk.models.pose_data import PoseSequence

_DASH = "—"

JOINT_INFO_FIELD_ORDER: tuple[str, ...] = (
    "joint_name",
    "x",
    "y",
    "z",
    "angle",
    "angular_velocity",
    "angular_acceleration",
    "rom",
    "frame",
    "time",
    "confidence",
    "contact",
    "gait_phase",
    "foot_clearance",
)

JOINT_INFO_TITLES: dict[str, str] = {
    "joint_name": "Joint name",
    "x": "Current X",
    "y": "Current Y",
    "z": "Current Z",
    "angle": "Angle",
    "angular_velocity": "Angular velocity",
    "angular_acceleration": "Angular acceleration",
    "rom": "Range of Motion",
    "frame": "Frame number",
    "time": "Time",
    "confidence": "Tracking confidence",
    "contact": "Contact state",
    "gait_phase": "Current gait phase",
    "foot_clearance": "Foot clearance",
}


@dataclass(frozen=True)
class JointInformationSnapshot:
    """Formatted live readouts for the Joint Information panel."""

    joint_name: str = _DASH
    x: str = _DASH
    y: str = _DASH
    z: str = _DASH
    angle: str = _DASH
    angular_velocity: str = _DASH
    angular_acceleration: str = _DASH
    rom: str = _DASH
    frame: str = _DASH
    time: str = _DASH
    confidence: str = _DASH
    contact: str = _DASH
    gait_phase: str = _DASH
    foot_clearance: str = _DASH

    def as_field_map(self) -> dict[str, str]:
        return {
            "joint_name": self.joint_name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "angular_acceleration": self.angular_acceleration,
            "rom": self.rom,
            "frame": self.frame,
            "time": self.time,
            "confidence": self.confidence,
            "contact": self.contact,
            "gait_phase": self.gait_phase,
            "foot_clearance": self.foot_clearance,
        }


def empty_joint_information() -> JointInformationSnapshot:
    """Placeholder values when no joint is selected or data is unavailable."""
    return JointInformationSnapshot()


def _fmt_m(value: float | None) -> str:
    if value is None:
        return _DASH
    return f"{value:.3f} m"


def _fmt_angle(value: float | None) -> str:
    if value is None:
        return _DASH
    return f"{value:.1f}°"


def _fmt_deg_s(value: float | None) -> str:
    if value is None:
        return _DASH
    return f"{value:.2f} °/s"


def _fmt_deg_s2(value: float | None) -> str:
    if value is None:
        return _DASH
    return f"{value:.2f} °/s²"


def _fmt_confidence(value: float | None) -> str:
    if value is None:
        return _DASH
    pct = value * 100.0 if value <= 1.0 + 1e-9 else value
    return f"{pct:.0f}%"


def _angular_velocity_deg_s(
    item_id: str,
    snapshot: SkeletonSnapshot,
) -> float | None:
    dof_id = _ITEM_DOF_ID.get(item_id)
    dof = snapshot.get_dof(dof_id) if dof_id else None
    if dof is not None and dof.velocity_deg_s is not None:
        return float(dof.velocity_deg_s)
    return None


def _angular_acceleration_deg_s2(
    item_id: str,
    snapshot: SkeletonSnapshot,
    prev_snapshot: SkeletonSnapshot | None,
    *,
    fps: float = 0.0,
) -> float | None:
    """Δω/Δt from consecutive DoF angular velocities (°/s²)."""
    omega = _angular_velocity_deg_s(item_id, snapshot)
    if omega is None or prev_snapshot is None:
        return None
    omega_prev = _angular_velocity_deg_s(item_id, prev_snapshot)
    if omega_prev is None:
        return None
    dt = float(snapshot.time_s) - float(prev_snapshot.time_s)
    if dt <= 1e-6:
        dt = 1.0 / fps if fps > 1e-6 else 0.0
    if dt <= 1e-6:
        return None
    return (omega - omega_prev) / dt


def _rom_display(
    item_id: str,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
) -> str:
    """Prefer angular ROM (min–max) from joint or DoF angle history."""
    if recording is None or recording.frame_count <= 0:
        return _DASH

    anchor = anchor_joint_for_item(item_id)
    if anchor:
        stats = _joint_angle_window_stats(recording, anchor, end_frame_float)
        if stats is not None:
            _current, amin, amax = stats
            return f"{amin:.1f}–{amax:.1f}°"

    dof_id = _ITEM_DOF_ID.get(item_id)
    if not dof_id:
        return _DASH
    ts = recording.build_time_series()
    series = ts.dof_angles.get(dof_id, [])
    if not series:
        return _DASH
    last_i = int(min(max(0, end_frame_float), len(series) - 1))
    window = [float(a) for a in series[: last_i + 1] if a is not None]
    if not window:
        return _DASH
    return f"{min(window):.1f}–{max(window):.1f}°"


def build_joint_information(
    item_id: str | None,
    snapshot: SkeletonSnapshot | None,
    *,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_phase: str | None = None,
    end_frame_float: float | None = None,
) -> JointInformationSnapshot:
    """Build live Joint Information fields for the active joint and frame."""
    if not item_id or snapshot is None:
        return empty_joint_information()

    anchor = anchor_joint_for_item(item_id)
    joint = snapshot.joints.get(anchor) if anchor else None
    frame_index = int(snapshot.frame_index)
    frame_f = (
        float(end_frame_float)
        if end_frame_float is not None
        else float(frame_index)
    )

    prev_snapshot = None
    if recording is not None and frame_index > 0:
        prev_snapshot = recording.snapshot_at(frame_index - 1)

    fps = float(getattr(recording, "fps", 0.0) or 0.0) if recording else 0.0

    x = joint.position.x if joint is not None else None
    y = joint.position.y if joint is not None else None
    z = joint.position.z if joint is not None else None

    angle = angle_value_for_item(item_id, snapshot)
    omega = _angular_velocity_deg_s(item_id, snapshot)
    alpha = _angular_acceleration_deg_s2(
        item_id, snapshot, prev_snapshot, fps=fps
    )

    confidence = confidence_for_joint_at_frame(sequence, anchor, frame_index)

    clearance_raw, contact_raw = _foot_table_cells(item_id, snapshot, recording)
    if is_foot_analysis_point(item_id):
        foot_clearance = (
            clearance_raw if clearance_raw and clearance_raw != NA_VALUE else _DASH
        )
        contact = contact_raw if contact_raw and contact_raw != NA_VALUE else _DASH
    else:
        foot_clearance = _DASH
        contact = _DASH

    phase = (gait_phase or "").strip() or _DASH
    if phase in ("", "—", "-"):
        phase = _DASH

    name = label_for_item(item_id)
    if anchor:
        name = JOINT_DISPLAY_NAMES.get(anchor, name)

    return JointInformationSnapshot(
        joint_name=name,
        x=_fmt_m(x),
        y=_fmt_m(y),
        z=_fmt_m(z),
        angle=_fmt_angle(angle),
        angular_velocity=_fmt_deg_s(omega),
        angular_acceleration=_fmt_deg_s2(alpha),
        rom=_rom_display(item_id, recording, frame_f),
        frame=str(frame_index + 1),
        time=f"{float(snapshot.time_s):.2f} s",
        confidence=_fmt_confidence(confidence),
        contact=contact,
        gait_phase=phase,
        foot_clearance=foot_clearance,
    )
