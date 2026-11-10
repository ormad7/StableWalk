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
    "angle",
    "rom",
    "x",
    "y",
    "z",
    "angular_velocity",
    "confidence",
    "contact",
    "foot_clearance",
    "gait_phase",
)

JOINT_INFO_TITLES: dict[str, str] = {
    "joint_name": "Joint",
    "x": "X (cm, root-rel.)",
    "y": "Y (cm, root-rel.)",
    "z": "Z (cm, root-rel.)",
    "angle": "Current angle (est., °)",
    "angular_velocity": "ω (°/s, derived)",
    "angular_acceleration": "α (°/s², derived)",
    "rom": "Session ROM (°)",
    "frame": "Frame",
    "time": "Time",
    "confidence": "Tracking confidence",
    "contact": "Contact state",
    "gait_phase": "Phase (derived)",
    "foot_clearance": "Clearance (est., cm)",
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


def _fmt_deg_s(value: float | None) -> str:
    from stablewalk.ui.value_provenance import format_na

    if value is None:
        return format_na("angular velocity not available")
    if abs(float(value)) > 400.0:
        return format_na("angular velocity spike (|ω|>400 °/s)")
    return f"{value:.1f} °/s"


def _fmt_angle(value: float | None) -> str:
    from stablewalk.ui.value_provenance import format_na

    if value is None:
        return format_na("joint/DOF angle not available")
    return f"{value:.1f}°"


def _fmt_cm(value: float | None, *, scale: float = 1.0) -> str:
    from stablewalk.ui.value_provenance import format_na

    if value is None:
        return format_na("position not available")
    return f"{value * scale * 100.0:.1f} cm"


def _fmt_m(value: float | None) -> str:
    from stablewalk.ui.value_provenance import format_na

    if value is None:
        return format_na("position not available")
    return f"{value:.3f} m"


def _fmt_deg_s2(value: float | None) -> str:
    from stablewalk.ui.value_provenance import format_na

    if value is None:
        return format_na("angular acceleration not available")
    # Frame-to-frame Δω/Δt is noisy on monocular tracking; hide extreme spikes.
    if abs(float(value)) > 500.0:
        return format_na("angular acceleration spike (|α|>500 °/s²)")
    return f"{value:.1f} °/s²"


def _fmt_confidence(value: float | None) -> str:
    from stablewalk.ui.value_provenance import format_na

    if value is None:
        return format_na("keypoint visibility unavailable")
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
    from stablewalk.ui.value_provenance import format_na

    if recording is None or recording.frame_count <= 0:
        return format_na("no recording for angle window")

    anchor = anchor_joint_for_item(item_id)
    if anchor:
        stats = _joint_angle_window_stats(recording, anchor, end_frame_float)
        if stats is not None:
            _current, amin, amax = stats
            return f"{amin:.1f}–{amax:.1f}°"

    dof_id = _ITEM_DOF_ID.get(item_id)
    if not dof_id:
        return format_na("no DOF angle series for item")
    ts = recording.build_time_series()
    series = ts.dof_angles.get(dof_id, [])
    if not series:
        return format_na("empty DOF angle series")
    last_i = int(min(max(0, end_frame_float), len(series) - 1))
    window = [float(a) for a in series[: last_i + 1] if a is not None]
    if not window:
        return format_na("insufficient angle samples in window")
    return f"{min(window):.1f}–{max(window):.1f}°"


def build_joint_information(
    item_id: str | None,
    snapshot: SkeletonSnapshot | None,
    *,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_phase: str | None = None,
    end_frame_float: float | None = None,
    provenance_sink: dict | None = None,
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
    list_i = int(max(0.0, frame_f))
    if recording is not None and list_i > 0:
        prev_snapshot = recording.snapshot_at(list_i - 1)

    fps = float(getattr(recording, "fps", 0.0) or 0.0) if recording else 0.0

    x = joint.position.x if joint is not None else None
    y = joint.position.y if joint is not None else None
    z = joint.position.z if joint is not None else None

    from stablewalk.ui.viewers.dof_trajectory_3d import stature_display_scale

    pos_scale = stature_display_scale(recording)

    angle = angle_value_for_item(item_id, snapshot)
    omega = _angular_velocity_deg_s(item_id, snapshot)
    alpha = _angular_acceleration_deg_s2(
        item_id, snapshot, prev_snapshot, fps=fps
    )

    confidence = confidence_for_joint_at_frame(sequence, anchor, frame_index)

    clearance_raw, contact_raw = _foot_table_cells(item_id, snapshot, recording)
    from stablewalk.ui.value_provenance import format_na

    if is_foot_analysis_point(item_id):
        foot_clearance = (
            clearance_raw
            if clearance_raw and clearance_raw != NA_VALUE
            else format_na("foot clearance unavailable")
        )
        contact = (
            contact_raw
            if contact_raw and contact_raw != NA_VALUE
            else format_na("contact state unavailable")
        )
    else:
        foot_clearance = format_na("not a foot analysis point")
        contact = format_na("not a foot analysis point")

    phase = (gait_phase or "").strip() or ""
    if phase in ("", "—", "-"):
        phase = format_na("gait phase not available at this frame")

    name = label_for_item(item_id)
    if anchor:
        name = JOINT_DISPLAY_NAMES.get(anchor, name)

    # Prefer playback list position (end_frame_float) for display — never the
    # raw source-video id alone, which can diverge after filtered poses.
    if end_frame_float is not None:
        display_frame = int(max(0, float(end_frame_float))) + 1
        list_pos = int(max(0, float(end_frame_float)))
    else:
        display_frame = int(frame_index) + 1
        list_pos = None

    x_disp = _fmt_cm(x, scale=pos_scale)
    y_disp = _fmt_cm(y, scale=pos_scale)
    z_disp = _fmt_cm(z, scale=pos_scale)
    angle_disp = _fmt_angle(angle)
    omega_disp = _fmt_deg_s(omega)
    rom_disp = _rom_display(item_id, recording, frame_f)
    conf_disp = _fmt_confidence(confidence)

    info = JointInformationSnapshot(
        joint_name=name,
        x=x_disp,
        y=y_disp,
        z=z_disp,
        angle=angle_disp,
        angular_velocity=omega_disp,
        angular_acceleration=_fmt_deg_s2(alpha),
        rom=rom_disp,
        frame=str(display_frame),
        time=f"{float(snapshot.time_s):.2f} s",
        confidence=conf_disp,
        contact=contact,
        gait_phase=phase,
        foot_clearance=foot_clearance,
    )

    if provenance_sink is not None:
        from stablewalk.ui.value_provenance import build_joint_info_provenance

        provenance_sink.update(
            build_joint_info_provenance(
                item_id=item_id,
                list_pos=list_pos,
                source_video_frame=frame_index,
                x_raw=x,
                y_raw=y,
                z_raw=z,
                x_display=x_disp,
                y_display=y_disp,
                z_display=z_disp,
                angle_raw=angle,
                angle_display=angle_disp,
                omega_raw=omega,
                omega_display=omega_disp,
                rom_display=rom_disp,
                confidence=confidence,
                confidence_display=conf_disp,
                contact_display=contact,
                phase_display=phase,
                pos_scale=pos_scale,
                anchor=anchor,
            )
        )
    return info
