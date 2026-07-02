"""Export full playback tracking history to CSV and JSON."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from stablewalk.storage.models import KinematicSample

TRACKING_CSV_COLUMNS: tuple[str, ...] = (
    "time_s",
    "frame_number",
    "dof_name",
    "joint_name",
    "x",
    "y",
    "z",
    "velocity",
    "angle_deg",
    "velocity_deg_s",
    "next_angle_deg",
    "delta_angle_deg",
)

TRACKING_CSV_HEADINGS: dict[str, str] = {
    "time_s": "Timestamp (s)",
    "frame_number": "Frame",
    "dof_name": "DOF",
    "joint_name": "Joint",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "velocity": "Velocity (m/s)",
    "angle_deg": "Angle (deg)",
    "velocity_deg_s": "Angular velocity (deg/s)",
    "next_angle_deg": "Next angle (deg)",
    "delta_angle_deg": "Delta angle (deg)",
}


def tracking_export_stem(*, now: datetime | None = None) -> str:
    """Return a timestamped basename, e.g. ``tracking_data_20250615_143022``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"tracking_data_{stamp}"


def _format_csv_value(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_tracking_csv(samples: list[KinematicSample], path: str | Path) -> Path:
    """Write structured kinematic samples to a CSV file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    headers = [TRACKING_CSV_HEADINGS[col] for col in TRACKING_CSV_COLUMNS]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for sample in samples:
            row = asdict(sample)
            writer.writerow(
                [_format_csv_value(row.get(col)) for col in TRACKING_CSV_COLUMNS]
            )
    return out


def _sample_to_json_dict(sample: KinematicSample) -> dict[str, Any]:
    return asdict(sample)


def write_tracking_json(
    samples: list[KinematicSample],
    path: str | Path,
    *,
    video_source: str | None = None,
    selected_dofs: list[str] | None = None,
    fps: float | None = None,
    exported_at: str | None = None,
) -> Path:
    """Write structured kinematic samples and export metadata to JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_at": exported_at or datetime.now().isoformat(timespec="seconds"),
        "video_source": video_source,
        "fps": fps,
        "selected_dofs": selected_dofs or [],
        "sample_count": len(samples),
        "samples": [_sample_to_json_dict(sample) for sample in samples],
    }
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out


def export_tracking_bundle(
    samples: list[KinematicSample],
    output_dir: str | Path,
    *,
    video_source: str | None = None,
    selected_dofs: list[str] | None = None,
    fps: float | None = None,
    include_json: bool = True,
) -> dict[str, Path]:
    """
    Export tracking data to timestamped files under ``output_dir``.

    Returns a mapping of format name to written path (always includes ``csv``;
    includes ``json`` when ``include_json`` is True).
    """
    if not samples:
        raise ValueError("No tracking samples to export")

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = tracking_export_stem()
    written: dict[str, Path] = {
        "csv": write_tracking_csv(samples, directory / f"{stem}.csv"),
    }
    if include_json:
        written["json"] = write_tracking_json(
            samples,
            directory / f"{stem}.json",
            video_source=video_source,
            selected_dofs=selected_dofs,
            fps=fps,
        )
    return written
