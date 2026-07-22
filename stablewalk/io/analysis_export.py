"""Export selected-point analysis time series to CSV and JSON."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

# Per-frame columns shared by all analysis points
ANALYSIS_EXPORT_GENERAL_COLUMNS: tuple[str, ...] = (
    "time",
    "selected_point",
    "frame",
    "time_sec",
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

# Foot analysis only — appended after general columns
ANALYSIS_EXPORT_FOOT_ONLY_COLUMNS: tuple[str, ...] = (
    "foot_clearance_m",
    "foot_clearance_cm",
    "contact_status",
    "min_clearance_m",
    "max_clearance_m",
    "average_clearance_m",
)

ANALYSIS_EXPORT_CORE_COLUMNS: tuple[str, ...] = ANALYSIS_EXPORT_GENERAL_COLUMNS

ANALYSIS_EXPORT_FOOT_COLUMNS: tuple[str, ...] = (
    *ANALYSIS_EXPORT_GENERAL_COLUMNS,
    *ANALYSIS_EXPORT_FOOT_ONLY_COLUMNS,
)

ANALYSIS_EXPORT_LEGACY_COLUMNS: tuple[str, ...] = (
    "joint_name",
    "time_s",
    "speed_m_s",
    "ground_distance_m",
    "height_above_ground_m",
    "min_foot_clearance_m",
    "max_foot_clearance_m",
    "avg_ground_distance_m",
    "contact_state",
    "angle_deg",
    "delta_angle_deg",
    "trajectory_length_m",
    "range_x_m",
    "range_y_m",
    "range_z_m",
    "vertical_range_m",
    "ground_level_reference",
    "ground_level_m",
    "vertical_distance_axis",
    "unit_scaling_method",
    "clearance_is_estimated",
    "measurement_note",
    "distance_from_ground_m",
    "distance_from_ground_cm",
    "calibration_status",
    "avg_clearance_m",
)

ANALYSIS_EXPORT_COLUMNS: tuple[str, ...] = (
    *ANALYSIS_EXPORT_GENERAL_COLUMNS,
    *ANALYSIS_EXPORT_FOOT_ONLY_COLUMNS,
    *ANALYSIS_EXPORT_LEGACY_COLUMNS,
)

ANALYSIS_CSV_HEADINGS: dict[str, str] = {
    "time": "Time (s)",
    "selected_point": "Selected Point",
    "frame": "Frame",
    "time_sec": "Time (s)",
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
    "contact_status": "Contact State",
    "min_clearance_m": "Min Clearance (m)",
    "max_clearance_m": "Max Clearance (m)",
    "average_clearance_m": "Average Clearance (m)",
    "ground_level_reference": "Ground Level Reference (m)",
    "ground_level_m": "Ground Level (m)",
    "vertical_distance_axis": "Vertical Distance Axis",
    "unit_scaling_method": "Unit / Scaling Method",
    "clearance_is_estimated": "Clearance Is Estimated",
    "measurement_note": "Measurement Note",
    "calibration_status": "Calibration Status",
    "distance_from_ground_m": "Distance From Ground (m)",
    "distance_from_ground_cm": "Distance From Ground (cm)",
    "joint_name": "Joint",
    "time_s": "Time (s)",
    "speed_m_s": "Speed (m/s)",
    "ground_distance_m": "Ground Distance (m)",
    "height_above_ground_m": "Ground Distance (m)",
    "min_foot_clearance_m": "Min Ground Distance (m)",
    "max_foot_clearance_m": "Max Ground Distance (m)",
    "avg_ground_distance_m": "Avg Ground Distance (m)",
    "contact_state": "Contact State",
    "angle_deg": "Angle (deg)",
    "delta_angle_deg": "Delta Angle (deg)",
    "vertical_position_m": "Vertical Position (m)",
    "trajectory_length_m": "Trajectory Length (m)",
    "range_x_m": "Range X (m)",
    "range_y_m": "Range Y (m)",
    "range_z_m": "Range Z (m)",
    "vertical_range_m": "Vertical Range (m)",
    "avg_clearance_m": "Average Clearance (m)",
}

FILE_SELECTED_POINT_SUMMARY = "selected_point_summary.json"


def analysis_export_columns_for_item(item_id: str | None) -> tuple[str, ...]:
    """Return export columns for the active analysis point (general vs foot)."""
    if not item_id:
        return ANALYSIS_EXPORT_GENERAL_COLUMNS
    from stablewalk.ui.selected_point_analysis import is_foot_analysis_point

    if is_foot_analysis_point(item_id):
        return ANALYSIS_EXPORT_FOOT_COLUMNS
    return ANALYSIS_EXPORT_GENERAL_COLUMNS


def project_analysis_export_row(
    row: dict[str, str],
    item_id: str | None,
) -> dict[str, str]:
    """Keep only mode-relevant fields for the active selected point."""
    columns = analysis_export_columns_for_item(item_id)
    projected = {col: row.get(col, "") for col in columns}
    if item_id:
        projected["item_id"] = item_id
    return projected


def analysis_export_stem(*, now: datetime | None = None) -> str:
    """Return a timestamped basename, e.g. ``selected_point_analysis_20250615_143022``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"selected_point_analysis_{stamp}"


def write_analysis_csv(
    rows: list[dict[str, str]],
    path: str | Path,
    *,
    columns: tuple[str, ...] | None = None,
) -> Path:
    """Write analysis rows to a CSV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or ANALYSIS_EXPORT_GENERAL_COLUMNS
    headers = [ANALYSIS_CSV_HEADINGS.get(col, col) for col in cols]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(col, "") for col in cols])
    return out


def write_analysis_json(
    rows: list[dict[str, str]],
    path: str | Path,
    *,
    item_id: str,
    video_source: str | None = None,
    fps: float | None = None,
    exported_at: str | None = None,
    summary: dict[str, Any] | None = None,
    columns: tuple[str, ...] | None = None,
    analysis_mode: str | None = None,
) -> Path:
    """Write analysis rows and export metadata to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = columns or analysis_export_columns_for_item(item_id)
    payload: dict[str, Any] = {
        "exported_at": exported_at or datetime.now().isoformat(timespec="seconds"),
        "item_id": item_id,
        "selected_point": rows[0].get("selected_point") if rows else "",
        "analysis_mode": analysis_mode,
        "video_source": video_source,
        "fps": fps,
        "frame_count": len(rows),
        "columns": list(cols),
        "rows": rows,
    }
    if summary is not None:
        payload["summary"] = summary
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out


def write_selected_point_summary_json(
    summary: dict[str, Any],
    path: str | Path,
) -> Path:
    """Write the compact current-frame summary for the charted selected point."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return out


def export_analysis_bundle(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    *,
    item_id: str,
    video_source: str | None = None,
    fps: float | None = None,
    include_json: bool = True,
    summary: dict[str, Any] | None = None,
    analysis_mode: str | None = None,
) -> dict[str, Path]:
    """
    Export selected-point analysis to timestamped files under ``output_dir``.

    Returns a mapping of format name to written path (always includes ``csv``;
    includes ``json`` when ``include_json`` is True).
    """
    if not rows:
        raise ValueError("No analysis rows to export")

    columns = analysis_export_columns_for_item(item_id)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = analysis_export_stem()
    written: dict[str, Path] = {
        "csv": write_analysis_csv(rows, directory / f"{stem}.csv", columns=columns),
    }
    if include_json:
        written["json"] = write_analysis_json(
            rows,
            directory / f"{stem}.json",
            item_id=item_id,
            video_source=video_source,
            fps=fps,
            summary=summary,
            columns=columns,
            analysis_mode=analysis_mode,
        )
    if summary is not None:
        written["summary"] = write_selected_point_summary_json(
            summary,
            directory / FILE_SELECTED_POINT_SUMMARY,
        )
    return written
