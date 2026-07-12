"""Per-frame foot clearance debug export for scientific audit."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from stablewalk.analysis.foot_clearance_filter import (
    COORDINATE_UNITS,
    build_filtered_foot_clearance_series,
)
from stablewalk.analysis.ground_reference import (
    estimate_ground_plane,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import GaitMotionRecording

DEBUG_CSV_COLUMNS = [
    "frame_index",
    "timestamp_s",
    "vertical_axis",
    "coordinate_units",
    "floor_height_m",
    "left_heel_y_m",
    "left_toe_y_m",
    "right_heel_y_m",
    "right_toe_y_m",
    "left_heel_visibility",
    "left_toe_visibility",
    "right_heel_visibility",
    "right_toe_visibility",
    "left_raw_clearance_m",
    "right_raw_clearance_m",
    "left_filtered_clearance_m",
    "right_filtered_clearance_m",
    "left_displayed_clearance_cm",
    "right_displayed_clearance_cm",
    "left_phase",
    "right_phase",
    "left_display_state",
    "right_display_state",
    "left_valid",
    "right_valid",
    "left_reject_reason",
    "right_reject_reason",
]


def _fmt_m(value: float | None) -> str | float:
    return round(value, 5) if value is not None else ""


def _fmt_cm(value: float | None) -> str | float:
    return round(value * 100.0, 2) if value is not None else ""


def build_foot_clearance_debug_rows(
    recording: GaitMotionRecording,
) -> list[dict[str, Any]]:
    """Build per-frame bilateral foot clearance audit rows."""
    plane = estimate_ground_plane(recording, float(max(recording.frame_count - 1, 0)))
    left_series = build_filtered_foot_clearance_series(recording, plane, "left")
    right_series = build_filtered_foot_clearance_series(recording, plane, "right")

    left_by_frame = {s.frame_index: s for s in left_series.samples}
    right_by_frame = {s.frame_index: s for s in right_series.samples}

    rows: list[dict[str, Any]] = []
    for index in range(recording.frame_count):
        left = left_by_frame.get(index)
        right = right_by_frame.get(index)
        if left is None and right is None:
            continue
        ts = left.timestamp_s if left else (right.timestamp_s if right else 0.0)
        axis = (
            left.vertical_axis
            if left
            else (right.vertical_axis if right else "y")
        )
        floor_y = (
            left.floor_y_m
            if left
            else (right.floor_y_m if right else (plane.floor_y if plane else None))
        )

        def _side_row(side_sample, side: str) -> dict[str, Any]:
            if side_sample is None:
                return {
                    f"{side}_heel_y_m": "",
                    f"{side}_toe_y_m": "",
                    f"{side}_heel_visibility": "",
                    f"{side}_toe_visibility": "",
                    f"{side}_raw_clearance_m": "",
                    f"{side}_filtered_clearance_m": "",
                    f"{side}_displayed_clearance_cm": "",
                    f"{side}_phase": "",
                    f"{side}_display_state": "",
                    f"{side}_valid": "",
                    f"{side}_reject_reason": "",
                }
            lm = side_sample.landmark
            filtered = side_sample.filtered_clearance_m
            return {
                f"{side}_heel_y_m": _fmt_m(lm.heel_y_m),
                f"{side}_toe_y_m": _fmt_m(lm.toe_y_m),
                f"{side}_heel_visibility": round(lm.heel_visibility, 3),
                f"{side}_toe_visibility": round(lm.toe_visibility, 3),
                f"{side}_raw_clearance_m": _fmt_m(side_sample.raw_clearance_m),
                f"{side}_filtered_clearance_m": _fmt_m(filtered),
                f"{side}_displayed_clearance_cm": _fmt_m(filtered),
                f"{side}_phase": side_sample.phase,
                f"{side}_display_state": side_sample.display_state,
                f"{side}_valid": side_sample.is_valid,
                f"{side}_reject_reason": side_sample.reject_reason,
            }

        row = {
            "frame_index": index,
            "timestamp_s": round(ts, 4),
            "vertical_axis": axis,
            "coordinate_units": COORDINATE_UNITS,
            "floor_height_m": _fmt_m(floor_y),
        }
        row.update(_side_row(left, "left"))
        row.update(_side_row(right, "right"))
        rows.append(row)

    return rows


def export_foot_clearance_debug_csv(
    recording: GaitMotionRecording,
    path: Path,
) -> Path:
    """Write per-frame foot clearance debug CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_foot_clearance_debug_rows(recording)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=DEBUG_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def floor_stability_summary(recording: GaitMotionRecording) -> dict[str, Any]:
    """Report floor estimate stability metrics for audit."""
    plane = estimate_ground_plane(recording, float(max(recording.frame_count - 1, 0)))
    if plane is None:
        return {
            "floor_y_m": None,
            "candidate_min_m": None,
            "candidate_max_m": None,
            "candidate_std_m": None,
            "coordinate_units": COORDINATE_UNITS,
            "scale_mode": "unknown",
        }
    return {
        "floor_y_m": plane.floor_y,
        "candidate_min_m": plane.floor_candidate_min,
        "candidate_max_m": plane.floor_candidate_max,
        "candidate_std_m": plane.floor_candidate_std,
        "coordinate_units": plane.coordinate_units,
        "scale_mode": plane.scale_mode,
        "body_vertical_span": plane.body_vertical_span,
    }
