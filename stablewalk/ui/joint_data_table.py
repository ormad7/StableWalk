"""
Presentation layer for the Detailed Joint Data table.

Enriches existing kinematic samples and history rows with derived fields
(acceleration, confidence, contact state) at display and export time without
changing playback collection.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.models.pose_data import PoseSequence
from stablewalk.storage.models import KinematicSample
from stablewalk.ui.dof_position_table import (
    DofTableRow,
    NA_VALUE,
    _ITEM_DOF_ID,
    _foot_table_cells,
    angle_value_for_item,
)
from stablewalk.ui.dof_selection import (
    GUI_DOF_ITEM_IDS,
    anchor_joint_for_item,
    label_for_item,
)

JOINT_DATA_TABLE_COLUMNS: tuple[str, ...] = (
    "frame",
    "timestamp",
    "joint",
    "x",
    "y",
    "z",
    "angle",
    "velocity",
    "acceleration",
    "confidence",
    "contact_state",
    "gait_phase",
)

JOINT_DATA_TABLE_HEADINGS: dict[str, str] = {
    "frame": "Frame",
    "timestamp": "Time (s)",
    "joint": "Joint",
    "x": "X (m)",
    "y": "Y (m)",
    "z": "Z (m)",
    "angle": "Angle (°)",
    "velocity": "Velocity (m/s or °/s)",
    "acceleration": "Acceleration (m/s² or °/s²)",
    "confidence": "Confidence (detector visibility 0–1)",
    "contact_state": "Contact State",
    "gait_phase": "Gait Phase",
}

# Confidence is MediaPipe/backend landmark visibility, not presence and not a
# derived tracking-accuracy score. Missing visibility stays unavailable.
CONFIDENCE_SOURCE = "detector_visibility"
CONFIDENCE_UNAVAILABLE_REASON = "detector visibility unavailable"
JOINT_DATA_PAGE_SIZE = 250

JOINT_DATA_TABLE_WIDTHS: dict[str, int] = {
    "frame": 56,
    "timestamp": 72,
    "joint": 112,
    "x": 68,
    "y": 68,
    "z": 68,
    "angle": 72,
    "velocity": 128,
    "acceleration": 150,
    "confidence": 200,
    "contact_state": 104,
    "gait_phase": 110,
}

_EXPORT_TEXT_COLUMNS = frozenset({"joint", "contact_state", "gait_phase"})

LOW_CONFIDENCE_THRESHOLD = 0.45

JOINT_FILTER_ALL = "All joints"

_LABEL_TO_ITEM: dict[str, str] = {
    label_for_item(item_id): item_id for item_id in GUI_DOF_ITEM_IDS
}


def _dash(value: float | None, *, fmt: str = ".3f", suffix: str = "") -> str:
    if value is None:
        return NA_VALUE
    return f"{value:{fmt}}{suffix}"


def _parse_float_cell(value: str) -> float | None:
    if not value or value == NA_VALUE:
        return None
    cleaned = value.strip().replace("°/s", "").replace("m/s", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def gait_phase_label_for_frame(
    frame_index: int,
    *,
    gait_cycle: Any | None = None,
) -> str:
    """Session gait-phase label for one frame, or — when unavailable."""
    if gait_cycle is None:
        return NA_VALUE
    try:
        state = gait_cycle.frame_at(int(frame_index))
    except Exception:
        return NA_VALUE
    if state is None:
        return NA_VALUE
    phase = getattr(state, "phase", None)
    if not phase:
        return NA_VALUE
    from stablewalk.analysis.gait_phase_classification import format_gait_phase_display

    label = format_gait_phase_display(str(phase))
    return label if label else NA_VALUE


def _joint_display_name(item_id: str) -> str:
    anchor = anchor_joint_for_item(item_id)
    if anchor:
        return JOINT_DISPLAY_NAMES.get(anchor, anchor.replace("_", " ").title())
    return label_for_item(item_id)


def confidence_for_joint_at_frame(
    sequence: PoseSequence | None,
    joint_id: str | None,
    frame_index: int,
) -> float | None:
    """Pose keypoint visibility for one joint at a frame (0–1)."""
    if sequence is None or not joint_id:
        return None
    if frame_index < 0 or frame_index >= len(sequence.frames):
        return None
    frame = sequence.frames[frame_index]
    for keypoint in frame.keypoints:
        if keypoint.name == joint_id:
            return float(keypoint.visibility)
    return None


def _compute_acceleration(
    velocity_now: float | None,
    velocity_prev: float | None,
    dt: float | None,
) -> float | None:
    if velocity_now is None or velocity_prev is None or dt is None or dt <= 0:
        return None
    return (velocity_now - velocity_prev) / dt


def _linear_velocity(
    item_id: str,
    snapshot: SkeletonSnapshot,
) -> tuple[float | None, str | None]:
    """Linear joint speed (m/s) or angular DOF speed (°/s). Never invents a value."""
    anchor = anchor_joint_for_item(item_id)
    joint = snapshot.joints.get(anchor) if anchor else None
    if joint is not None and joint.velocity is not None:
        return float(joint.velocity), "m/s"
    dof_id = _ITEM_DOF_ID.get(item_id)
    dof = snapshot.get_dof(dof_id) if dof_id else None
    if dof is not None and dof.velocity_deg_s is not None:
        return float(dof.velocity_deg_s), "deg/s"
    return None, None


@dataclass(frozen=True)
class JointDataRow:
    """One enriched joint-data table row."""

    frame: int
    timestamp: float
    joint: str
    item_id: str
    x: float | None
    y: float | None
    z: float | None
    angle: float | None
    velocity: float | None
    acceleration: float | None
    confidence: float | None
    contact_state: str
    gait_phase: str = "—"
    confidence_source: str | None = None
    confidence_unavailable_reason: str | None = None
    velocity_unit: str | None = None

    def is_low_confidence(self) -> bool:
        return (
            self.confidence is not None
            and self.confidence < LOW_CONFIDENCE_THRESHOLD
        )

    def as_display_tuple(self) -> tuple[str, ...]:
        return (
            str(self.frame),
            f"{self.timestamp:.2f}",
            self.joint,
            _dash(self.x),
            _dash(self.y),
            _dash(self.z),
            _dash(self.angle, fmt=".2f"),
            (
                _dash(self.velocity)
                if self.velocity is None or not self.velocity_unit
                else f"{self.velocity:.3f} {self.velocity_unit}"
            ),
            _dash(self.acceleration),
            _dash(self.confidence, fmt=".2f"),
            self.contact_state,
            self.gait_phase or "—",
        )

    def sort_value(self, column: str) -> tuple[int, str | float]:
        """Return a comparable value for stable sorting."""
        if column == "frame":
            return (0, float(self.frame))
        if column == "timestamp":
            return (0, float(self.timestamp))
        if column == "joint":
            return (1, self.joint.lower())
        if column == "x":
            return (0, self.x if self.x is not None else float("-inf"))
        if column == "y":
            return (0, self.y if self.y is not None else float("-inf"))
        if column == "z":
            return (0, self.z if self.z is not None else float("-inf"))
        if column == "angle":
            return (0, self.angle if self.angle is not None else float("-inf"))
        if column == "velocity":
            return (0, self.velocity if self.velocity is not None else float("-inf"))
        if column == "acceleration":
            return (
                0,
                self.acceleration if self.acceleration is not None else float("-inf"),
            )
        if column == "confidence":
            return (0, self.confidence if self.confidence is not None else float("-inf"))
        if column == "contact_state":
            return (1, self.contact_state.lower())
        if column == "gait_phase":
            return (1, (self.gait_phase or "").lower())
        return (1, "")


def joint_data_row_from_snapshot(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    prev_snapshot: SkeletonSnapshot | None = None,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
) -> JointDataRow:
    """Build one enriched row from a recording snapshot."""
    anchor = anchor_joint_for_item(item_id)
    joint = snapshot.joints.get(anchor) if anchor else None

    x = joint.position.x if joint is not None else None
    y = joint.position.y if joint is not None else None
    z = joint.position.z if joint is not None else None

    velocity, velocity_unit = _linear_velocity(item_id, snapshot)
    prev_velocity, _prev_unit = (
        _linear_velocity(item_id, prev_snapshot)
        if prev_snapshot is not None
        else (None, None)
    )
    dt = None
    if prev_snapshot is not None:
        dt = float(snapshot.time_s) - float(prev_snapshot.time_s)

    acceleration = _compute_acceleration(velocity, prev_velocity, dt)
    confidence = confidence_for_joint_at_frame(
        sequence,
        anchor,
        snapshot.frame_index,
    )
    _, contact_state = _foot_table_cells(item_id, snapshot, recording)
    if contact_state == NA_VALUE:
        contact_state = "—"
    angle = angle_value_for_item(item_id, snapshot)

    return JointDataRow(
        frame=snapshot.frame_index + 1,
        timestamp=float(snapshot.time_s),
        joint=_joint_display_name(item_id),
        item_id=item_id,
        x=x,
        y=y,
        z=z,
        angle=angle,
        velocity=velocity,
        acceleration=acceleration,
        confidence=confidence,
        contact_state=contact_state,
        gait_phase=gait_phase_label_for_frame(
            snapshot.frame_index,
            gait_cycle=gait_cycle,
        ),
        confidence_source=CONFIDENCE_SOURCE if confidence is not None else None,
        confidence_unavailable_reason=(
            None if confidence is not None else CONFIDENCE_UNAVAILABLE_REASON
        ),
        velocity_unit=velocity_unit,
    )


def joint_data_row_from_sample(
    sample: KinematicSample,
    *,
    prev_sample: KinematicSample | None,
    item_id: str | None = None,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
) -> JointDataRow:
    """Build one enriched row from a collected kinematic sample."""
    resolved_item_id = item_id or _LABEL_TO_ITEM.get(sample.dof_name, "")
    frame_index = max(0, int(sample.frame_number) - 1)

    velocity = sample.velocity
    if velocity is None and sample.velocity_deg_s is not None:
        velocity = float(sample.velocity_deg_s)

    prev_velocity = None
    dt = None
    if prev_sample is not None:
        if prev_sample.velocity is not None:
            prev_velocity = float(prev_sample.velocity)
        elif prev_sample.velocity_deg_s is not None:
            prev_velocity = float(prev_sample.velocity_deg_s)
        dt = float(sample.time_s) - float(prev_sample.time_s)

    acceleration = _compute_acceleration(velocity, prev_velocity, dt)

    confidence = None
    contact_state = "—"
    angle = float(sample.angle_deg) if sample.angle_deg is not None else None
    if resolved_item_id and recording is not None:
        snapshot = recording.snapshot_at(frame_index)
        if snapshot is not None:
            anchor = anchor_joint_for_item(resolved_item_id)
            confidence = confidence_for_joint_at_frame(
                sequence,
                anchor,
                frame_index,
            )
            _, contact_state = _foot_table_cells(
                resolved_item_id,
                snapshot,
                recording,
            )
            if contact_state == NA_VALUE:
                contact_state = "—"
            if angle is None:
                angle = angle_value_for_item(resolved_item_id, snapshot)

    return JointDataRow(
        frame=int(sample.frame_number),
        timestamp=float(sample.time_s),
        joint=sample.joint_name or sample.dof_name,
        item_id=resolved_item_id,
        x=sample.x,
        y=sample.y,
        z=sample.z,
        angle=angle,
        velocity=velocity,
        acceleration=acceleration,
        confidence=confidence,
        contact_state=contact_state,
        gait_phase=gait_phase_label_for_frame(
            frame_index,
            gait_cycle=gait_cycle,
        ),
        confidence_source=CONFIDENCE_SOURCE if confidence is not None else None,
        confidence_unavailable_reason=(
            None if confidence is not None else CONFIDENCE_UNAVAILABLE_REASON
        ),
        velocity_unit="m/s" if velocity is not None else None,
    )


def joint_data_rows_from_samples(
    samples: list[KinematicSample],
    *,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
) -> list[JointDataRow]:
    """Enrich collected kinematic samples for display or export."""
    if not samples:
        return []

    rows: list[JointDataRow] = []
    prev_by_dof: dict[str, KinematicSample] = {}
    for sample in samples:
        prev = prev_by_dof.get(sample.dof_name)
        item_id = _LABEL_TO_ITEM.get(sample.dof_name)
        rows.append(
            joint_data_row_from_sample(
                sample,
                prev_sample=prev,
                item_id=item_id,
                recording=recording,
                sequence=sequence,
                gait_cycle=gait_cycle,
            )
        )
        prev_by_dof[sample.dof_name] = sample
    return rows


def joint_data_rows_from_history(
    history_rows: list[DofTableRow],
    *,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
) -> list[JointDataRow]:
    """Enrich legacy position-table history rows."""
    if not history_rows:
        return []

    parsed: list[tuple[int, str, DofTableRow]] = []
    for row in history_rows:
        if len(row) < 3:
            continue
        try:
            frame = int(row[1])
        except ValueError:
            continue
        item_id = _LABEL_TO_ITEM.get(row[2], "")
        if not item_id:
            continue
        parsed.append((frame, item_id, row))

    parsed.sort(key=lambda item: (item[1], item[0]))

    rows: list[JointDataRow] = []
    prev_by_item: dict[str, JointDataRow] = {}
    for frame, item_id, legacy_row in parsed:
        frame_index = frame - 1
        snapshot = (
            recording.snapshot_at(frame_index)
            if recording is not None
            else None
        )
        prev_snapshot = (
            recording.snapshot_at(frame_index - 1)
            if recording is not None and frame_index > 0
            else None
        )

        if snapshot is not None:
            row = joint_data_row_from_snapshot(
                item_id,
                snapshot,
                prev_snapshot=prev_snapshot,
                recording=recording,
                sequence=sequence,
                gait_cycle=gait_cycle,
            )
        else:
            velocity = _parse_float_cell(legacy_row[6] if len(legacy_row) > 6 else NA_VALUE)
            prev_row = prev_by_item.get(item_id)
            dt = None
            prev_velocity = None
            if prev_row is not None:
                prev_velocity = prev_row.velocity
                dt = float(legacy_row[0]) - prev_row.timestamp
            acceleration = _compute_acceleration(velocity, prev_velocity, dt)
            contact_state = legacy_row[8] if len(legacy_row) > 8 else "—"
            if contact_state == NA_VALUE:
                contact_state = "—"
            confidence = confidence_for_joint_at_frame(
                sequence,
                anchor_joint_for_item(item_id),
                frame_index,
            )
            row = JointDataRow(
                frame=frame,
                timestamp=float(legacy_row[0]),
                joint=_joint_display_name(item_id),
                item_id=item_id,
                x=_parse_float_cell(legacy_row[3] if len(legacy_row) > 3 else NA_VALUE),
                y=_parse_float_cell(legacy_row[4] if len(legacy_row) > 4 else NA_VALUE),
                z=_parse_float_cell(legacy_row[5] if len(legacy_row) > 5 else NA_VALUE),
                angle=None,
                velocity=velocity,
                acceleration=acceleration,
                confidence=confidence,
                contact_state=contact_state,
                gait_phase=gait_phase_label_for_frame(
                    frame_index,
                    gait_cycle=gait_cycle,
                ),
                confidence_source=CONFIDENCE_SOURCE if confidence is not None else None,
                confidence_unavailable_reason=(
                    None if confidence is not None else CONFIDENCE_UNAVAILABLE_REASON
                ),
                velocity_unit="m/s" if velocity is not None else None,
            )

        rows.append(row)
        prev_by_item[item_id] = row
    return rows


def joint_data_rows_for_selection(
    snapshot: SkeletonSnapshot,
    selected_item_ids: set[str],
    *,
    recording: GaitMotionRecording | None = None,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
) -> list[JointDataRow]:
    """One enriched row per selected item at the current snapshot."""
    ordered = [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids]
    prev_index = snapshot.frame_index - 1
    prev_snapshot = (
        recording.snapshot_at(prev_index)
        if recording is not None and prev_index >= 0
        else None
    )
    return [
        joint_data_row_from_snapshot(
            item_id,
            snapshot,
            prev_snapshot=prev_snapshot,
            recording=recording,
            sequence=sequence,
            gait_cycle=gait_cycle,
        )
        for item_id in ordered
    ]


def joint_data_rows_from_recording(
    recording: GaitMotionRecording,
    item_ids: set[str] | list[str] | tuple[str, ...],
    *,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
) -> list[JointDataRow]:
    """One row per selected joint for every recording frame."""
    wanted = set(item_ids)
    ordered = [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in wanted]
    if not ordered or recording.frame_count <= 0:
        return []
    rows: list[JointDataRow] = []
    prev_by_item: dict[str, SkeletonSnapshot] = {}
    for index in range(int(recording.frame_count)):
        snapshot = recording.snapshot_at(index)
        if snapshot is None:
            continue
        for item_id in ordered:
            rows.append(
                joint_data_row_from_snapshot(
                    item_id,
                    snapshot,
                    prev_snapshot=prev_by_item.get(item_id),
                    recording=recording,
                    sequence=sequence,
                    gait_cycle=gait_cycle,
                )
            )
            prev_by_item[item_id] = snapshot
    return rows


def available_joint_item_ids(
    recording: GaitMotionRecording | None,
) -> list[str]:
    """GUI joints that actually appear in the recording (or the full checklist)."""
    from stablewalk.ui.dof_selection import item_for_joint

    if recording is None or recording.frame_count <= 0:
        return list(GUI_DOF_ITEM_IDS)
    present: set[str] = set()
    for snapshot in recording.snapshots:
        for joint_id in snapshot.joints:
            item_id = item_for_joint(joint_id)
            if item_id:
                present.add(item_id)
    ordered = [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in present]
    return ordered or list(GUI_DOF_ITEM_IDS)


def collect_joint_data_rows(
    *,
    recording: GaitMotionRecording | None,
    item_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    sequence: PoseSequence | None = None,
    gait_cycle: Any | None = None,
    samples: list[KinematicSample] | None = None,
) -> list[JointDataRow]:
    """Prefer full recording frames; fall back to playback samples."""
    wanted = set(item_ids) if item_ids else set()
    if recording is not None and recording.frame_count > 0:
        if not wanted:
            wanted = set(available_joint_item_ids(recording))
        return joint_data_rows_from_recording(
            recording,
            wanted,
            sequence=sequence,
            gait_cycle=gait_cycle,
        )
    if samples:
        return joint_data_rows_from_samples(
            samples,
            recording=recording,
            sequence=sequence,
            gait_cycle=gait_cycle,
        )
    return []


def filter_joint_data_rows(
    rows: list[JointDataRow],
    joint_filter: str | None,
) -> list[JointDataRow]:
    """Filter rows by joint label or item id. All joints keeps every row."""
    if not joint_filter or joint_filter == JOINT_FILTER_ALL:
        return list(rows)
    return [
        row
        for row in rows
        if row.joint == joint_filter
        or row.item_id == joint_filter
        or label_for_item(row.item_id) == joint_filter
    ]


def paginate_joint_data_rows(
    rows: list[JointDataRow],
    page: int,
    *,
    page_size: int = JOINT_DATA_PAGE_SIZE,
) -> tuple[list[JointDataRow], int, int]:
    """Return (page_rows, page_index, page_count). page_index is 0-based."""
    size = max(1, int(page_size))
    total = len(rows)
    page_count = max(1, (total + size - 1) // size)
    index = max(0, min(int(page), page_count - 1))
    start = index * size
    return rows[start : start + size], index, page_count


def sort_joint_data_rows(
    rows: list[JointDataRow],
    column: str,
    *,
    reverse: bool = False,
) -> list[JointDataRow]:
    """Sort rows by a table column id."""
    if column not in JOINT_DATA_TABLE_COLUMNS:
        return rows
    return sorted(
        rows,
        key=lambda row: row.sort_value(column),
        reverse=reverse,
    )


def joint_filter_options(rows: list[JointDataRow]) -> list[str]:
    """Combobox values: all joints plus joints present in the current rows."""
    joints = sorted({row.joint for row in rows if row.joint})
    return [JOINT_FILTER_ALL, *joints]


def joint_data_export_stem(*, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"joint_data_{stamp}"


def _format_export_value(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_joint_data_csv(rows: list[JointDataRow], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [JOINT_DATA_TABLE_HEADINGS[col] for col in JOINT_DATA_TABLE_COLUMNS]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            payload = asdict(row)
            writer.writerow(
                [
                    (
                        str(payload.get(col) or "")
                        if col in _EXPORT_TEXT_COLUMNS
                        else _format_export_value(payload.get(col))
                    )
                    for col in JOINT_DATA_TABLE_COLUMNS
                ]
            )
    return out


def _row_to_json_dict(row: JointDataRow) -> dict[str, Any]:
    payload = asdict(row)
    out = {col: payload.get(col) for col in JOINT_DATA_TABLE_COLUMNS}
    out["time"] = payload.get("timestamp")
    out["confidence_source"] = payload.get("confidence_source")
    out["confidence_unavailable_reason"] = payload.get("confidence_unavailable_reason")
    out["velocity_unit"] = payload.get("velocity_unit")
    return out


def write_joint_data_json(
    rows: list[JointDataRow],
    path: str | Path,
    *,
    video_source: str | None = None,
    selected_joints: list[str] | None = None,
    fps: float | None = None,
    exported_at: str | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_at": exported_at or datetime.now().isoformat(timespec="seconds"),
        "video_source": video_source,
        "fps": fps,
        "selected_joints": selected_joints or [],
        "row_count": len(rows),
        "columns": list(JOINT_DATA_TABLE_COLUMNS),
        "column_headings": {
            col: JOINT_DATA_TABLE_HEADINGS[col] for col in JOINT_DATA_TABLE_COLUMNS
        },
        "confidence": {
            "source": CONFIDENCE_SOURCE,
            "meaning": "Detector landmark visibility (0–1), not presence and not a derived accuracy score.",
            "missing": CONFIDENCE_UNAVAILABLE_REASON,
        },
        "rows": [_row_to_json_dict(row) for row in rows],
    }
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out


def export_joint_data_bundle(
    rows: list[JointDataRow],
    output_dir: str | Path,
    *,
    video_source: str | None = None,
    selected_joints: list[str] | None = None,
    fps: float | None = None,
    include_json: bool = True,
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No joint data rows to export")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = joint_data_export_stem()
    written: dict[str, Path] = {
        "csv": write_joint_data_csv(rows, directory / f"{stem}.csv"),
    }
    if include_json:
        written["json"] = write_joint_data_json(
            rows,
            directory / f"{stem}.json",
            video_source=video_source,
            selected_joints=selected_joints,
            fps=fps,
        )
    return written


__all__ = [
    "JOINT_DATA_TABLE_COLUMNS",
    "JOINT_DATA_TABLE_HEADINGS",
    "JOINT_DATA_TABLE_WIDTHS",
    "CONFIDENCE_SOURCE",
    "CONFIDENCE_UNAVAILABLE_REASON",
    "JOINT_DATA_PAGE_SIZE",
    "JOINT_FILTER_ALL",
    "LOW_CONFIDENCE_THRESHOLD",
    "JointDataRow",
    "available_joint_item_ids",
    "collect_joint_data_rows",
    "confidence_for_joint_at_frame",
    "export_joint_data_bundle",
    "filter_joint_data_rows",
    "gait_phase_label_for_frame",
    "joint_data_rows_for_selection",
    "joint_data_rows_from_history",
    "joint_data_rows_from_recording",
    "joint_data_rows_from_samples",
    "joint_filter_options",
    "paginate_joint_data_rows",
    "sort_joint_data_rows",
    "write_joint_data_csv",
    "write_joint_data_json",
]
