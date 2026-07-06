"""
Export and import complete gait analysis session bundles.

A session bundle is a folder containing:

- ``session_metadata.json`` — video path, playback frame, UI state
- ``selected_points.json`` — selected body points
- ``tracking_history.csv`` — per-frame table history for all selected points
- ``analysis_summary.json`` — full per-point analysis time series (JSON)
- ``selected_point_summary.json`` — compact current-frame summary for the charted point
- ``gait_motion.json`` — optional full motion recording for offline restore
- ``bilateral_foot_clearance.json`` — left/right foot ground distance time series
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from stablewalk.io.analysis_export import (
    FILE_SELECTED_POINT_SUMMARY,
    analysis_export_columns_for_item,
    project_analysis_export_row,
)
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.storage.models import KinematicSample
from stablewalk.ui.dof_selection import GUI_DOF_LABELS, label_for_item
from stablewalk.ui.selected_point_analysis import (
    analysis_mode_for_item,
    bilateral_foot_clearance_export_rows,
    build_current_point_export_summary,
    collect_analysis_export_rows,
)

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA = "stablewalk-session-bundle"
BUNDLE_VERSION = "1.0"

FILE_SESSION_METADATA = "session_metadata.json"
FILE_SELECTED_POINTS = "selected_points.json"
FILE_TRACKING_HISTORY = "tracking_history.csv"
FILE_ANALYSIS_SUMMARY = "analysis_summary.json"
FILE_GAIT_MOTION = "gait_motion.json"
FILE_BILATERAL_FOOT_CLEARANCE = "bilateral_foot_clearance.json"

TRACKING_HISTORY_GENERAL_COLUMNS: tuple[str, ...] = (
    "time_sec",
    "frame",
    "selected_point",
    "x_m",
    "y_m",
    "z_m",
    "speed_mps",
    "path_length_m",
    "delta_x_m",
    "delta_y_m",
    "delta_z_m",
    "vertical_position_m",
)

TRACKING_HISTORY_FOOT_COLUMNS: tuple[str, ...] = (
    "foot_clearance_m",
    "foot_clearance_cm",
    "contact_status",
    "min_clearance_m",
    "max_clearance_m",
    "average_clearance_m",
)

# Full superset for reading legacy bundles
TRACKING_HISTORY_COLUMNS: tuple[str, ...] = (
    *TRACKING_HISTORY_GENERAL_COLUMNS,
    *TRACKING_HISTORY_FOOT_COLUMNS,
    "calibration_status",
    "time",
    "ground_level_m",
    "ground_level_reference",
    "avg_clearance_m",
    "vertical_distance_axis",
    "unit_scaling_method",
    "clearance_is_estimated",
    "measurement_note",
    "time_s",
    "item_id",
    "joint_name",
    "speed_m_s",
    "distance_from_ground_m",
    "distance_from_ground_cm",
    "angle_deg",
    "delta_angle_deg",
    "ground_distance_m",
)

TRACKING_HISTORY_HEADINGS: dict[str, str] = {
    "time_sec": "Time (s)",
    "frame": "Frame",
    "selected_point": "Selected Point",
    "x_m": "X (m)",
    "y_m": "Y (m)",
    "z_m": "Z (m)",
    "speed_mps": "Speed (m/s)",
    "path_length_m": "Path Length (m)",
    "delta_x_m": "Delta X (m)",
    "delta_y_m": "Delta Y (m)",
    "delta_z_m": "Delta Z (m)",
    "vertical_position_m": "Vertical Position (m)",
    "foot_clearance_m": "Foot Clearance (m)",
    "foot_clearance_cm": "Foot Clearance (cm)",
    "ground_level_m": "Ground Level (m)",
    "contact_status": "Contact Status",
    "min_clearance_m": "Min Clearance (m)",
    "max_clearance_m": "Max Clearance (m)",
    "average_clearance_m": "Average Clearance (m)",
    "calibration_status": "Calibration Status",
    "time": "Time (s)",
    "ground_level_reference": "Ground Level Reference (m)",
    "avg_clearance_m": "Average Clearance (m)",
    "vertical_distance_axis": "Vertical Distance Axis",
    "unit_scaling_method": "Unit / Scaling Method",
    "clearance_is_estimated": "Clearance Is Estimated",
    "measurement_note": "Measurement Note",
    "time_s": "Time (s)",
    "item_id": "Item ID",
    "joint_name": "Joint",
    "speed_m_s": "Speed (m/s)",
    "distance_from_ground_m": "Distance From Ground (m)",
    "distance_from_ground_cm": "Distance From Ground (cm)",
    "angle_deg": "Angle (deg)",
    "delta_angle_deg": "Delta Angle (deg)",
    "ground_distance_m": "Ground Distance (m)",
}


@dataclass
class SessionBundleSnapshot:
    """Everything needed to export or restore one analysis session."""

    video_source: str
    poses_json_path: str | None = None
    fps: float | None = None
    frame_count: int = 0
    selected_item_ids: set[str] = field(default_factory=set)
    last_selected: str | None = None
    charted_item_id: str | None = None
    active_item_id: str | None = None
    analysis_mode: str | None = None
    frame_index: int = 0
    frame_float: float = 0.0
    time_s: float = 0.0
    dof_table_display_mode: str | None = None
    smooth_motion: bool = True
    tracking_samples: list[KinematicSample] = field(default_factory=list)
    recording: GaitMotionRecording | None = None
    notes: str | None = None

    @property
    def export_item_id(self) -> str | None:
        """Active analysis point used for exported tables and summaries."""
        active = self.active_item_id or self.charted_item_id
        if active and active in self.selected_item_ids:
            return active
        ordered = _ordered_selected(self.selected_item_ids)
        return ordered[0] if ordered else None


@dataclass
class SessionBundleLoadResult:
    """Parsed session bundle ready for GUI restore."""

    bundle_dir: Path
    metadata: dict[str, Any]
    selected_points: dict[str, Any]
    tracking_rows: list[dict[str, str]]
    analysis_summary: dict[str, Any]
    recording: GaitMotionRecording | None
    warnings: list[str] = field(default_factory=list)


class SessionBundleError(Exception):
    """Raised when a bundle cannot be read or written."""


def session_bundle_stem(*, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"stablewalk_session_{stamp}"


def _utc_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ordered_selected(item_ids: set[str]) -> list[str]:
    from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS

    return [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in item_ids]


def tracking_history_columns_for_item(item_id: str | None) -> tuple[str, ...]:
    """CSV columns for tracking history — foot metrics only for foot points."""
    return analysis_export_columns_for_item(item_id)


def build_tracking_history_rows(
    recording: GaitMotionRecording,
    selected_item_ids: set[str],
    *,
    active_item_id: str | None = None,
) -> list[dict[str, str]]:
    """Per-frame rows for the active analysis point (mode-aware columns)."""
    export_id = active_item_id
    if not export_id or export_id not in selected_item_ids:
        ordered = _ordered_selected(selected_item_ids)
        export_id = ordered[0] if ordered else None
    if export_id is None:
        return []

    rows = collect_analysis_export_rows(export_id, recording)
    for row in rows:
        row.setdefault("item_id", export_id)
    rows.sort(key=lambda r: int(r.get("frame") or 0))
    return rows


def build_analysis_summary(
    recording: GaitMotionRecording,
    selected_item_ids: set[str],
    *,
    charted_item_id: str | None = None,
    active_item_id: str | None = None,
    analysis_mode: str | None = None,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Structured JSON summary for the active analysis point."""
    primary = active_item_id or charted_item_id
    if not primary or primary not in selected_item_ids:
        ordered = _ordered_selected(selected_item_ids)
        primary = ordered[0] if ordered else None

    mode = analysis_mode or (analysis_mode_for_item(primary) if primary else None)
    by_point: dict[str, Any] = {}
    if primary:
        rows = collect_analysis_export_rows(primary, recording)
        current_summary: dict[str, str] = {}
        target_frame = str(frame_index + 1)
        for row in rows:
            if row.get("frame") == target_frame:
                current_summary = dict(row)
                break
        if not current_summary and rows:
            current_summary = dict(rows[-1])
        by_point[primary] = {
            "label": label_for_item(primary),
            "analysis_mode": mode,
            "frame_count": len(rows),
            "current_frame_summary": current_summary,
            "rows": rows,
        }

    current_frame_analysis: dict[str, Any] | None = None
    if primary:
        try:
            current_frame_analysis = build_current_point_export_summary(
                primary,
                recording,
                frame_index=frame_index,
            )
        except (ValueError, IndexError):
            current_frame_analysis = None

    return {
        "schema": "stablewalk-analysis-summary",
        "version": BUNDLE_VERSION,
        "exported_at": _utc_now(),
        "fps": recording.fps,
        "frame_count": recording.frame_count,
        "active_item_id": primary,
        "charted_item_id": primary,
        "analysis_mode": mode,
        "playback_frame_index": frame_index,
        "selected_item_ids": _ordered_selected(selected_item_ids),
        "points_analyzed": [primary] if primary else [],
        "current_frame_analysis": current_frame_analysis,
        "by_point": by_point,
    }


def build_selected_points_payload(
    selected_item_ids: set[str],
    *,
    last_selected: str | None = None,
    charted_item_id: str | None = None,
    active_item_id: str | None = None,
    analysis_mode: str | None = None,
) -> dict[str, Any]:
    ordered = _ordered_selected(selected_item_ids)
    active = active_item_id or charted_item_id
    if active and active not in selected_item_ids:
        active = None
    return {
        "schema": "stablewalk-selected-points",
        "version": BUNDLE_VERSION,
        "exported_at": _utc_now(),
        "selected_item_ids": ordered,
        "last_selected": last_selected if last_selected in selected_item_ids else (
            ordered[0] if ordered else None
        ),
        "charted_item_id": active,
        "active_item_id": active,
        "analysis_mode": analysis_mode,
        "labels": {item_id: label_for_item(item_id) for item_id in ordered},
    }


def build_session_metadata(
    snapshot: SessionBundleSnapshot,
    *,
    bundle_dir_name: str,
    exported_at: str | None = None,
) -> dict[str, Any]:
    exported = exported_at or _utc_now()
    return {
        "schema": BUNDLE_SCHEMA,
        "version": BUNDLE_VERSION,
        "exported_at": exported,
        "bundle_name": bundle_dir_name,
        "video_source": snapshot.video_source,
        "poses_json_path": snapshot.poses_json_path,
        "fps": snapshot.fps,
        "frame_count": snapshot.frame_count,
        "playback": {
            "frame_index": snapshot.frame_index,
            "frame_float": snapshot.frame_float,
            "time_s": snapshot.time_s,
        },
        "active_item_id": snapshot.export_item_id,
        "analysis_mode": snapshot.analysis_mode,
        "ui": {
            "dof_table_display_mode": snapshot.dof_table_display_mode,
            "smooth_motion": snapshot.smooth_motion,
            "charted_item_id": snapshot.export_item_id,
            "active_item_id": snapshot.export_item_id,
            "analysis_mode": snapshot.analysis_mode,
            "last_selected": snapshot.last_selected,
        },
        "tracking_sample_count": len(snapshot.tracking_samples),
        "notes": snapshot.notes,
        "files": {
            "tracking_history": FILE_TRACKING_HISTORY,
            "analysis_summary": FILE_ANALYSIS_SUMMARY,
            "selected_point_summary": FILE_SELECTED_POINT_SUMMARY,
            "selected_points": FILE_SELECTED_POINTS,
            "gait_motion": FILE_GAIT_MOTION,
        },
    }


def write_tracking_history_csv(
    rows: list[dict[str, str]],
    path: Path,
    *,
    columns: tuple[str, ...] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    item_id = rows[0].get("item_id") if rows else None
    cols = columns or tracking_history_columns_for_item(item_id)
    headers = [TRACKING_HISTORY_HEADINGS.get(col, col) for col in cols]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(col, "") for col in cols])


def write_json_payload(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def export_session_bundle(
    snapshot: SessionBundleSnapshot,
    output_dir: str | Path,
    *,
    include_gait_motion: bool = True,
) -> Path:
    """
    Write a complete session bundle folder under ``output_dir``.

    Returns the path to the created bundle directory.
    """
    if not snapshot.selected_item_ids:
        raise SessionBundleError("Select at least one body point before exporting.")

    recording = snapshot.recording
    if recording is None or recording.frame_count <= 0:
        raise SessionBundleError("No motion recording available to export.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    bundle_name = session_bundle_stem()
    bundle_dir = root / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=False)

    exported_at = _utc_now()
    export_item_id = snapshot.export_item_id
    if export_item_id is None:
        raise SessionBundleError("Select an active body point before exporting.")

    tracking_rows = build_tracking_history_rows(
        recording,
        snapshot.selected_item_ids,
        active_item_id=export_item_id,
    )
    if not tracking_rows and snapshot.tracking_samples:
        tracking_rows = _tracking_rows_from_samples(
            snapshot.tracking_samples,
            active_item_id=export_item_id,
        )

    write_tracking_history_csv(
        tracking_rows,
        bundle_dir / FILE_TRACKING_HISTORY,
        columns=tracking_history_columns_for_item(export_item_id),
    )
    write_json_payload(
        build_selected_points_payload(
            snapshot.selected_item_ids,
            last_selected=snapshot.last_selected,
            charted_item_id=export_item_id,
            active_item_id=export_item_id,
            analysis_mode=snapshot.analysis_mode,
        ),
        bundle_dir / FILE_SELECTED_POINTS,
    )
    write_json_payload(
        build_analysis_summary(
            recording,
            snapshot.selected_item_ids,
            charted_item_id=export_item_id,
            active_item_id=export_item_id,
            analysis_mode=snapshot.analysis_mode,
            frame_index=snapshot.frame_index,
        ),
        bundle_dir / FILE_ANALYSIS_SUMMARY,
    )

    if export_item_id in snapshot.selected_item_ids:
        try:
            point_summary = build_current_point_export_summary(
                export_item_id,
                recording,
                frame_index=snapshot.frame_index,
            )
            write_json_payload(point_summary, bundle_dir / FILE_SELECTED_POINT_SUMMARY)
        except (ValueError, IndexError) as exc:
            logger.warning("Could not write %s: %s", FILE_SELECTED_POINT_SUMMARY, exc)

    write_json_payload(
        build_session_metadata(snapshot, bundle_dir_name=bundle_name, exported_at=exported_at),
        bundle_dir / FILE_SESSION_METADATA,
    )

    write_json_payload(
        {
            "schema": "stablewalk-bilateral-foot-clearance",
            "version": BUNDLE_VERSION,
            "exported_at": exported_at,
            "measurement_note": (
                "Estimated body-scale clearance from monocular pose; "
                "centimeter values are not clinical ground truth."
            ),
            "rows": bilateral_foot_clearance_export_rows(
                recording,
                float(recording.frame_count - 1),
            ),
        },
        bundle_dir / FILE_BILATERAL_FOOT_CLEARANCE,
    )

    if include_gait_motion:
        write_json_payload(recording.to_dict(), bundle_dir / FILE_GAIT_MOTION)

    logger.info("Exported session bundle to %s", bundle_dir.resolve())
    return bundle_dir


def _tracking_rows_from_samples(
    samples: list[KinematicSample],
    *,
    active_item_id: str | None = None,
) -> list[dict[str, str]]:
    """Fallback CSV rows when full recording analysis rows are unavailable."""
    from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS

    label_to_id = {label_for_item(item_id): item_id for item_id in GUI_DOF_ITEM_IDS}
    active_label = label_for_item(active_item_id) if active_item_id else None
    rows: list[dict[str, str]] = []
    for sample in samples:
        if active_label and sample.dof_name != active_label:
            continue
        item_id = label_to_id.get(sample.dof_name, active_item_id or "")
        row = {
            "time": f"{sample.time_s:.3f}",
            "time_sec": f"{sample.time_s:.3f}",
            "frame": str(sample.frame_number),
            "item_id": item_id,
            "selected_point": sample.dof_name,
            "x_m": _fmt(sample.x),
            "y_m": _fmt(sample.y),
            "z_m": _fmt(sample.z),
            "speed_mps": _fmt(sample.velocity),
            "path_length_m": "",
            "delta_x_m": "",
            "delta_y_m": "",
            "delta_z_m": "",
            "vertical_position_m": _fmt(sample.y),
            "foot_clearance_m": "",
            "foot_clearance_cm": "",
            "contact_status": "",
            "min_clearance_m": "",
            "max_clearance_m": "",
            "average_clearance_m": "",
        }
        rows.append(project_analysis_export_row(row, item_id or None))
    return rows


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def resolve_bundle_dir(path: str | Path) -> Path:
    """Accept a bundle directory or a file inside one (e.g. session_metadata.json)."""
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate.parent
    if candidate.is_dir():
        return candidate
    raise SessionBundleError(f"Path not found: {candidate}")


def load_session_bundle(path: str | Path) -> SessionBundleLoadResult:
    """Load and validate a session bundle from disk."""
    bundle_dir = resolve_bundle_dir(path)
    warnings: list[str] = []

    metadata_path = bundle_dir / FILE_SESSION_METADATA
    if not metadata_path.is_file():
        raise SessionBundleError(
            f"Missing {FILE_SESSION_METADATA} in {bundle_dir}\n"
            "Choose a folder exported by StableWalk."
        )

    try:
        metadata = _read_json(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionBundleError(f"Could not read {FILE_SESSION_METADATA}: {exc}") from exc

    selected_path = bundle_dir / FILE_SELECTED_POINTS
    if selected_path.is_file():
        try:
            selected_points = _read_json(selected_path)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read {FILE_SELECTED_POINTS}: {exc}")
            selected_points = {}
    else:
        warnings.append(f"Missing {FILE_SELECTED_POINTS}; selection may be incomplete.")
        selected_points = {}

    analysis_path = bundle_dir / FILE_ANALYSIS_SUMMARY
    if analysis_path.is_file():
        try:
            analysis_summary = _read_json(analysis_path)
        except (OSError, json.JSONDecodeError) as exc:
            warnings.append(f"Could not read {FILE_ANALYSIS_SUMMARY}: {exc}")
            analysis_summary = {}
    else:
        warnings.append(f"Missing {FILE_ANALYSIS_SUMMARY}.")
        analysis_summary = {}

    tracking_rows: list[dict[str, str]] = []
    tracking_path = bundle_dir / FILE_TRACKING_HISTORY
    if tracking_path.is_file():
        try:
            tracking_rows = _read_tracking_history_csv(tracking_path)
        except OSError as exc:
            warnings.append(f"Could not read {FILE_TRACKING_HISTORY}: {exc}")
    else:
        warnings.append(f"Missing {FILE_TRACKING_HISTORY}.")

    recording: GaitMotionRecording | None = None
    motion_path = bundle_dir / FILE_GAIT_MOTION
    if motion_path.is_file():
        try:
            motion_data = _read_json(motion_path)
            recording = GaitMotionRecording.from_dict(motion_data)
        except Exception as exc:
            warnings.append(f"Could not load {FILE_GAIT_MOTION}: {exc}")
    else:
        warnings.append(
            f"Missing {FILE_GAIT_MOTION}. Video re-analysis may be required for graphs."
        )

    return SessionBundleLoadResult(
        bundle_dir=bundle_dir,
        metadata=metadata,
        selected_points=selected_points,
        tracking_rows=tracking_rows,
        analysis_summary=analysis_summary,
        recording=recording,
        warnings=warnings,
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SessionBundleError(f"Expected JSON object in {path.name}")
    return data


def _read_tracking_history_csv(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return rows
        heading_to_key = {v: k for k, v in TRACKING_HISTORY_HEADINGS.items()}
        for raw in reader:
            row: dict[str, str] = {col: "" for col in TRACKING_HISTORY_COLUMNS}
            for heading, value in raw.items():
                key = heading_to_key.get(heading) or heading
                if key in row:
                    row[key] = (value or "").strip()
            rows.append(row)
    return rows


def tracking_rows_to_kinematic_samples(rows: list[dict[str, str]]) -> list[KinematicSample]:
    """Rebuild structured samples from tracking CSV rows."""
    samples: list[KinematicSample] = []
    for row in rows:
        samples.append(
            KinematicSample(
                frame_number=_int_or(row.get("frame"), 0),
                time_s=_float_or(row.get("time_s"), 0.0),
                dof_name=row.get("selected_point") or row.get("item_id") or "—",
                joint_name=row.get("joint_name") or "—",
                x=_float_or_none(row.get("x_m")),
                y=_float_or_none(row.get("y_m")),
                z=_float_or_none(row.get("z_m")),
                angle_deg=_float_or_none(row.get("angle_deg")),
                velocity=_float_or_none(row.get("speed_m_s")),
                delta_angle_deg=_float_or_none(row.get("delta_angle_deg")),
            )
        )
    return samples


def tracking_rows_to_table_rows(rows: list[dict[str, str]]) -> list[tuple[str, ...]]:
    """Convert tracking CSV rows into DOF table display tuples."""
    from stablewalk.ui.dof_position_table import NA_VALUE

    table_rows: list[tuple[str, ...]] = []
    for row in rows:
        clearance = row.get("foot_clearance_cm") or row.get("foot_clearance_m") or ""
        contact = row.get("contact_status") or ""
        table_rows.append(
            (
                row.get("time_sec") or row.get("time") or NA_VALUE,
                row.get("frame") or NA_VALUE,
                row.get("selected_point") or NA_VALUE,
                row.get("x_m") or NA_VALUE,
                row.get("y_m") or NA_VALUE,
                row.get("z_m") or NA_VALUE,
                row.get("speed_mps") or row.get("speed_m_s") or NA_VALUE,
                clearance if clearance else NA_VALUE,
                contact if contact else NA_VALUE,
            )
        )
    return table_rows


def selected_ids_from_payload(payload: dict[str, Any]) -> set[str]:
    raw = payload.get("selected_item_ids") or payload.get("selected") or []
    if isinstance(raw, str):
        raw = [part.strip() for part in raw.split(",") if part.strip()]
    return {item_id for item_id in raw if item_id in GUI_DOF_LABELS}


def _int_or(raw: str | None, default: int) -> int:
    try:
        return int(float(raw)) if raw else default
    except (TypeError, ValueError):
        return default


def _float_or(raw: str | None, default: float) -> float:
    try:
        return float(raw) if raw else default
    except (TypeError, ValueError):
        return default


def _float_or_none(raw: str | None) -> float | None:
    if not raw or raw == "—":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None
