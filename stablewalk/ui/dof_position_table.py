"""
Real-time position table rows for GUI-selected degrees of freedom.

Builds formatted table rows from ``SkeletonSnapshot`` data and maintains a
rolling history buffer during playback.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
)
from stablewalk.storage.models import KinematicSample
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.dof_selection import (
    GUI_DOF_ITEM_IDS,
    anchor_joint_for_item,
    label_for_item,
)

NA_VALUE = "—"

# General columns always shown; foot columns appended when a foot point is active.
DOF_TABLE_GENERAL_COLUMNS: tuple[str, ...] = (
    "time",
    "frame",
    "dof",
    "x",
    "y",
    "z",
    "speed",
)
DOF_TABLE_FOOT_COLUMNS: tuple[str, ...] = (
    "foot_clearance",
    "contact_status",
)
DOF_TABLE_COLUMNS: tuple[str, ...] = (
    DOF_TABLE_GENERAL_COLUMNS + DOF_TABLE_FOOT_COLUMNS
)

DofTableRow = tuple[str, ...]

DOF_TABLE_HEADINGS: dict[str, str] = {
    "time": "Time",
    "frame": "Frame",
    "dof": "Selected Point",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "speed": "Speed",
    "foot_clearance": "Foot Clearance",
    "contact_status": "Contact State",
}

DOF_TABLE_WIDTHS: dict[str, int] = {
    "time": 44,
    "frame": 40,
    "dof": 88,
    "x": 48,
    "y": 48,
    "z": 48,
    "speed": 52,
    "foot_clearance": 76,
    "contact_status": 72,
}

MAX_RECENT_TABLE_ROWS = 200

DOF_TABLE_MODE_CURRENT = "Current Frame Only"
DOF_TABLE_MODE_HISTORY = "Tracking History"
DOF_TABLE_MODE_LABELS: tuple[str, ...] = (
    DOF_TABLE_MODE_CURRENT,
    DOF_TABLE_MODE_HISTORY,
)
DOF_TABLE_MODE_DEFAULT = DOF_TABLE_MODE_HISTORY

# Primary flexion DOF for single-joint GUI items
_ITEM_DOF_ID: dict[str, str | None] = {
    "right_hip": "right_hip_flexion",
    "left_hip": "left_hip_flexion",
    "right_knee": "right_knee_flexion",
    "left_knee": "left_knee_flexion",
    "right_ankle": "right_ankle_flexion",
    "left_ankle": "left_ankle_flexion",
    "right_heel": "right_ankle_flexion",
    "left_heel": "left_ankle_flexion",
    "right_toe": "right_ankle_flexion",
    "left_toe": "left_ankle_flexion",
    "right_shoulder": "right_shoulder_flexion",
    "left_shoulder": "left_shoulder_flexion",
    "right_elbow": "right_elbow_flexion",
    "left_elbow": "left_elbow_flexion",
    "right_wrist": "right_elbow_flexion",
    "left_wrist": "left_elbow_flexion",
}


def _dash(value: float | None, *, fmt: str = ".3f", suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:{fmt}}{suffix}"


def angle_value_for_item(item_id: str, snapshot: SkeletonSnapshot) -> float | None:
    """Numeric flexion angle (degrees) for a GUI DOF item."""
    anchor = anchor_joint_for_item(item_id)
    dof_id = _ITEM_DOF_ID.get(item_id)
    dof = snapshot.get_dof(dof_id) if dof_id else None
    joint = snapshot.joints.get(anchor) if anchor else None
    if dof is not None and dof.angle_deg is not None:
        return float(dof.angle_deg)
    if joint is not None and joint.angle_deg is not None:
        return float(joint.angle_deg)
    return None


def _format_table_speed(joint: JointSample | None, dof: DofSample | None) -> str:
    """Linear or angular rate for the position table (compact numeric)."""
    if joint is not None and joint.velocity is not None:
        return f"{joint.velocity:.3f}"
    if dof is not None and dof.velocity_deg_s is not None:
        return f"{dof.velocity_deg_s:.3f} °/s"
    return NA_VALUE


def table_row_cell(row: DofTableRow, column: str) -> str:
    """Read one cell from a table row by column id."""
    try:
        index = DOF_TABLE_COLUMNS.index(column)
    except ValueError:
        return NA_VALUE
    if index >= len(row):
        return NA_VALUE
    return row[index]


def _format_detail_velocity(joint: JointSample | None, dof: DofSample | None) -> str:
    """Prefer angular rate for the DOF details card; fall back to linear speed."""
    if dof is not None and dof.velocity_deg_s is not None:
        return f"{dof.velocity_deg_s:.3f} °/s"
    if joint is not None and joint.velocity is not None:
        return f"{joint.velocity:.3f} m/s"
    return NA_VALUE


def _format_angle(joint: JointSample | None, dof: DofSample | None) -> str:
    if dof is not None and dof.angle_deg is not None:
        return f"{dof.angle_deg:.1f}°"
    if joint is not None and joint.angle_deg is not None:
        return f"{joint.angle_deg:.1f}°"
    return NA_VALUE


def _format_delta_angle(value: float | None) -> str:
    if value is None:
        return NA_VALUE
    return f"{value:+.1f}°"


@dataclass(frozen=True)
class DofDetailMetrics:
    """Formatted values for the selected-DOF sidebar detail card."""

    angle: str
    next_angle: str
    delta_angle: str
    velocity: str


def dof_detail_metrics(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    next_snapshot: SkeletonSnapshot | None = None,
) -> DofDetailMetrics:
    """Build angle / velocity strings for the DOF details panel."""
    anchor = anchor_joint_for_item(item_id)
    dof_id = _ITEM_DOF_ID.get(item_id)
    joint = snapshot.joints.get(anchor) if anchor else None
    dof = snapshot.get_dof(dof_id) if dof_id else None

    angle = _format_angle(joint, dof)
    velocity = _format_detail_velocity(joint, dof)

    next_angle = NA_VALUE
    delta_angle = NA_VALUE
    if next_snapshot is not None:
        next_anchor = anchor_joint_for_item(item_id)
        next_dof = next_snapshot.get_dof(dof_id) if dof_id else None
        next_joint = (
            next_snapshot.joints.get(next_anchor) if next_anchor else None
        )
        next_angle = _format_angle(next_joint, next_dof)
        current_val = angle_value_for_item(item_id, snapshot)
        next_val = angle_value_for_item(item_id, next_snapshot)
        if current_val is not None and next_val is not None:
            delta_angle = _format_delta_angle(next_val - current_val)

    return DofDetailMetrics(
        angle=angle,
        next_angle=next_angle,
        delta_angle=delta_angle,
        velocity=velocity,
    )


def snapshot_for_next_frame(
    recording: GaitMotionRecording,
    snapshot: SkeletonSnapshot,
) -> SkeletonSnapshot | None:
    """Next discrete frame snapshot for next-angle / delta-angle columns."""
    if recording.frame_count <= 1:
        return None
    next_index = snapshot.frame_index + 1
    if next_index >= recording.frame_count:
        return None
    return recording.snapshot_at(next_index)


def _next_and_delta_angle(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    next_snapshot: SkeletonSnapshot | None,
) -> tuple[str, str]:
    next_angle = NA_VALUE
    delta_angle = NA_VALUE
    if next_snapshot is None:
        return next_angle, delta_angle
    anchor = anchor_joint_for_item(item_id)
    dof_id = _ITEM_DOF_ID.get(item_id)
    next_joint = next_snapshot.joints.get(anchor) if anchor else None
    next_dof = next_snapshot.get_dof(dof_id) if dof_id else None
    next_angle = _format_angle(next_joint, next_dof)
    current_val = angle_value_for_item(item_id, snapshot)
    next_val = angle_value_for_item(item_id, next_snapshot)
    if current_val is not None and next_val is not None:
        delta_angle = _format_delta_angle(next_val - current_val)
    return next_angle, delta_angle


def kinematic_sample_for_item(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    next_snapshot: SkeletonSnapshot | None = None,
) -> KinematicSample:
    """Build a structured kinematic sample for persistence (numeric fields)."""
    anchor = anchor_joint_for_item(item_id)
    dof_label = label_for_item(item_id)
    joint_name = "—"
    if anchor:
        joint_name = JOINT_DISPLAY_NAMES.get(anchor, anchor.replace("_", " ").title())

    joint = snapshot.joints.get(anchor) if anchor else None
    dof_id = _ITEM_DOF_ID.get(item_id)
    dof = snapshot.get_dof(dof_id) if dof_id else None

    x = joint.position.x if joint is not None else None
    y = joint.position.y if joint is not None else None
    z = joint.position.z if joint is not None else None

    angle_deg = None
    if dof is not None and dof.angle_deg is not None:
        angle_deg = float(dof.angle_deg)
    elif joint is not None and joint.angle_deg is not None:
        angle_deg = float(joint.angle_deg)

    velocity = None
    if joint is not None and joint.velocity is not None:
        velocity = float(joint.velocity)

    velocity_deg_s = None
    if dof is not None and dof.velocity_deg_s is not None:
        velocity_deg_s = float(dof.velocity_deg_s)

    next_angle_deg = None
    delta_angle_deg = None
    current_val = angle_value_for_item(item_id, snapshot)
    if next_snapshot is not None:
        next_val = angle_value_for_item(item_id, next_snapshot)
        if next_val is not None:
            next_angle_deg = float(next_val)
        if current_val is not None and next_val is not None:
            delta_angle_deg = float(next_val - current_val)

    return KinematicSample(
        frame_number=snapshot.frame_index + 1,
        time_s=float(snapshot.time_s),
        dof_name=dof_label,
        joint_name=joint_name,
        x=x,
        y=y,
        z=z,
        angle_deg=angle_deg,
        velocity=velocity,
        velocity_deg_s=velocity_deg_s,
        next_angle_deg=next_angle_deg,
        delta_angle_deg=delta_angle_deg,
    )


def selection_includes_foot_points(selected_item_ids: set[str]) -> bool:
    """True when at least one selected GUI item is a foot-related point."""
    from stablewalk.analysis.ground_reference import FOOT_POINT_IDS

    return bool(selected_item_ids & FOOT_POINT_IDS)


def is_foot_table_item(item_id: str | None) -> bool:
    """True when the active table row should include foot clearance columns."""
    if not item_id:
        return False
    from stablewalk.analysis.ground_reference import FOOT_POINT_IDS

    return item_id in FOOT_POINT_IDS


def table_display_columns_for_selection(selected_item_ids: set[str]) -> tuple[str, ...]:
    """Treeview columns — foot metrics only for foot analysis points."""
    if selection_includes_foot_points(selected_item_ids):
        return DOF_TABLE_COLUMNS
    return DOF_TABLE_GENERAL_COLUMNS


def table_display_columns_for_item(item_id: str | None) -> tuple[str, ...]:
    """Display columns for a single active analysis point."""
    if is_foot_table_item(item_id):
        return DOF_TABLE_COLUMNS
    return DOF_TABLE_GENERAL_COLUMNS


def _foot_table_cells(
    item_id: str,
    snapshot: SkeletonSnapshot,
    recording: GaitMotionRecording | None,
) -> tuple[str, str]:
    """Foot clearance (cm) and contact status for one table row."""
    from stablewalk.analysis.ground_reference import (
        FOOT_POINT_IDS,
        compute_foot_clearance_reading,
        estimate_ground_plane,
        format_supporting_clearance_cm,
    )

    if recording is None or item_id not in FOOT_POINT_IDS:
        return NA_VALUE, NA_VALUE
    anchor = anchor_joint_for_item(item_id)
    joint = snapshot.joints.get(anchor) if anchor else None
    if joint is None:
        return NA_VALUE, NA_VALUE
    plane = estimate_ground_plane(recording, float(snapshot.frame_index))
    if plane is None:
        return NA_VALUE, NA_VALUE
    reading = compute_foot_clearance_reading(joint.position, plane)
    clearance_cm = format_supporting_clearance_cm(
        reading.foot_clearance_m,
        calibration_check=False,
    )
    return clearance_cm, reading.contact_state


def table_row_for_item(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    next_snapshot: SkeletonSnapshot | None = None,
    recording: GaitMotionRecording | None = None,
) -> DofTableRow:
    """Build one table row for a selected GUI DOF at the given snapshot."""
    anchor = anchor_joint_for_item(item_id)
    dof_label = label_for_item(item_id)

    joint = snapshot.joints.get(anchor) if anchor else None
    dof_id = _ITEM_DOF_ID.get(item_id)
    dof = snapshot.get_dof(dof_id) if dof_id else None

    if joint is not None:
        x = _dash(joint.position.x)
        y = _dash(joint.position.y)
        z = _dash(joint.position.z)
    else:
        x = y = z = NA_VALUE

    foot_clearance, contact_status = _foot_table_cells(
        item_id,
        snapshot,
        recording,
    )

    return (
        f"{snapshot.time_s:.2f}",
        str(snapshot.frame_index + 1),
        dof_label,
        x,
        y,
        z,
        _format_table_speed(joint, dof),
        foot_clearance,
        contact_status,
    )


def table_rows_for_selection(
    snapshot: SkeletonSnapshot,
    selected_item_ids: set[str],
    *,
    next_snapshot: SkeletonSnapshot | None = None,
    recording: GaitMotionRecording | None = None,
) -> list[DofTableRow]:
    """One row per selected DOF in stable panel order."""
    ordered = [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids]
    return [
        table_row_for_item(
            item_id,
            snapshot,
            next_snapshot=next_snapshot,
            recording=recording,
        )
        for item_id in ordered
    ]


@dataclass
class DofPositionTableHistory:
    """Rolling row buffer while animation plays; current-frame rows when paused."""

    max_rows: int = MAX_RECENT_TABLE_ROWS
    rows: list[DofTableRow] = field(default_factory=list)
    _last_frame_by_item: dict[str, int] = field(default_factory=dict, repr=False)

    def clear(self) -> None:
        self.rows.clear()
        self._last_frame_by_item.clear()

    def append_tick(
        self,
        snapshot: SkeletonSnapshot,
        selected_item_ids: set[str],
        *,
        next_snapshot: SkeletonSnapshot | None = None,
        recording: GaitMotionRecording | None = None,
    ) -> bool:
        """
        Append one sample per selected DOF during playback.

        Skips only when the same DOF was already recorded for this frame index.
        Returns True when at least one new row was added.
        """
        frame_index = snapshot.frame_index
        added = False
        ordered = [
            item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids
        ]
        for item_id in ordered:
            if self._last_frame_by_item.get(item_id) == frame_index:
                continue
            self._last_frame_by_item[item_id] = frame_index
            self.rows.append(
                table_row_for_item(
                    item_id,
                    snapshot,
                    next_snapshot=next_snapshot,
                    recording=recording,
                )
            )
            added = True
        if len(self.rows) > self.max_rows:
            self.rows = self.rows[-self.max_rows :]
        return added


def write_dof_history_csv(rows: list[DofTableRow], path: str | Path) -> Path:
    """Write tracked position-table rows to a CSV file."""
    out = Path(path)
    headers = [DOF_TABLE_HEADINGS[col] for col in DOF_TABLE_COLUMNS]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    return out
