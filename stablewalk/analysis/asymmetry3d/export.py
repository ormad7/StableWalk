"""CSV / JSON export for the 3D asymmetry report."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stablewalk.analysis.asymmetry3d.report import Asymmetry3DReport


def export_asymmetry_json(report: Asymmetry3DReport, path: str | Path) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["exported_at_utc"] = datetime.now(timezone.utc).isoformat()
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def export_asymmetry_csv(
    report: Asymmetry3DReport,
    path: str | Path,
    *,
    append: bool = False,
) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = []

    def section(title: str, headers: list[str], data: list[list[Any]]) -> None:
        rows.append([f"# {title}"])
        rows.append(headers)
        rows.extend(data)
        rows.append([])

    kin = report.kinematics
    events = report.events
    section(
        "parameters",
        ["key", "value", "units_or_note"],
        [
            ["fps", kin.fps, "Hz"],
            ["coordinate_system", kin.coordinate_system, ""],
            ["provenance", kin.provenance, "inferred_monocular_3d vs calibrated_metric"],
            ["units", kin.units, ""],
            ["n_frames", kin.n_frames, ""],
            ["missing_fraction", kin.missing_fraction, "0-1"],
            ["view_type", events.view_type, ""],
            ["events_reliable", events.events_reliable, ""],
            ["spatial_reliable", events.spatial_reliable, ""],
            ["kind", "derived", "not a clinical diagnosis"],
        ],
    )
    section(
        "index_definitions",
        ["name", "formula"],
        [[k, v] for k, v in report.index_definitions.items()],
    )
    section(
        "scalars",
        [
            "name",
            "display",
            "units",
            "left",
            "right",
            "si_percent",
            "asi_percent",
            "si_norm",
            "difference",
            "difference_units",
            "n_left_cycles",
            "n_right_cycles",
            "quality",
            "reliable",
            "comparison_kind",
            "alignment_event",
            "formula",
            "kind",
            "note",
        ],
        [
            [
                s.name,
                s.display,
                s.units,
                s.left,
                s.right,
                s.si_percent,
                s.asi_percent,
                s.si_norm,
                s.difference,
                s.difference_units,
                s.n_left_cycles,
                s.n_right_cycles,
                s.quality,
                s.reliable,
                s.comparison_kind,
                s.alignment_event,
                s.formula,
                s.kind,
                s.note,
            ]
            for s in report.scalars
        ],
    )
    traj_rows: list[list[Any]] = []
    for pair in report.trajectories:
        for i, pct in enumerate(pair.phase_pct):
            traj_rows.append(
                [
                    pair.name,
                    pair.units,
                    pct,
                    None if not _fin(pair.left_mean[i]) else float(pair.left_mean[i]),
                    None if not _fin(pair.right_mean[i]) else float(pair.right_mean[i]),
                    pair.aligned,
                ]
            )
    section(
        "phase_aligned_trajectories",
        ["name", "units", "gait_pct", "left_mean", "right_mean", "aligned"],
        traj_rows,
    )
    if report.moments is not None:
        mom = report.moments
        section(
            "moments",
            ["key", "value", "note"],
            [
                ["estimated_available", mom.estimated_available, mom.estimated_disabled_reason],
                ["measured_available", mom.measured_available, mom.measured_disabled_reason],
                ["subject_mass_kg", mom.subject_mass_kg, "required for I*α"],
                ["kind_kinematic", "kinematic_asymmetry", "statistical 3D moments"],
                ["kind_estimated", "estimated_joint_moment", "model-dependent"],
                ["kind_measured", "measured_joint_moment", "OpenSim ID + GRF"],
            ],
        )
        section(
            "estimated_moments",
            ["joint", "side", "inertia_kg_m2", "rms_nm", "peak_nm", "available"],
            [
                [
                    e.joint,
                    e.side,
                    e.inertia_kg_m2,
                    e.to_dict()["rms_nm"],
                    e.to_dict()["peak_nm"],
                    e.available,
                ]
                for e in mom.estimated
            ],
        )

    mode = "a" if append else "w"
    with dest.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        if append:
            writer.writerow([])
            writer.writerow(["# asymmetry_summary"])
        writer.writerows(rows)
    return dest


def _fin(value: Any) -> bool:
    try:
        return bool(value == value and abs(float(value)) != float("inf"))
    except Exception:
        return False
