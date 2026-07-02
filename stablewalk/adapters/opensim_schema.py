"""
OpenSim-compatible motion schema helpers.

Provides column naming, coordinate mapping, and export layout for future
``.sto`` / ``.mot`` integration without requiring the OpenSim SDK at runtime.

For the full biomechanics pipeline (export bundle, joint mapping, angle/DOF
computation), use ``stablewalk.biomechanics.BiomechanicsService`` or the
module-level helpers in ``stablewalk.biomechanics.service``.
"""

from __future__ import annotations

from typing import Any

from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.joint_registry import (
    DOF_IDS,
    OPENSIM_COORDINATE_ALIASES,
    all_dof_definitions,
    normalize_dof_id,
)

OPENSIM_EXPORT_VERSION = "1.0"


def opensim_column_name(dof_id: str) -> str:
    """Canonical DOF id → preferred OpenSim coordinate label."""
    dof_id = normalize_dof_id(dof_id)
    definition = next((d for d in all_dof_definitions() if d.id == dof_id), None)
    if definition and definition.opensim_coordinate:
        return definition.opensim_coordinate
    return dof_id


def canonical_from_opensim(name: str) -> str:
    """OpenSim coordinate label → canonical DOF id."""
    return normalize_dof_id(name)


def motion_to_opensim_table(recording: GaitMotionRecording) -> dict[str, Any]:
    """
    Export motion as an OpenSim-style column table (in-memory).

    Returns:
        dict with keys ``time``, ``columns`` (name list), ``data`` (rows of floats).
        Positions are not included here — angles only, matching typical ``.sto`` files.
    """
    ts = recording.build_time_series()
    columns = ["time"] + [opensim_column_name(d) for d in DOF_IDS]
    rows: list[list[float]] = []

    for i, t in enumerate(ts.times_s):
        row: list[float] = [t]
        for dof_id in DOF_IDS:
            series = ts.dof_angles.get(dof_id, [])
            val = series[i] if i < len(series) else None
            row.append(float(val) if val is not None else 0.0)
        rows.append(row)

    return {
        "version": OPENSIM_EXPORT_VERSION,
        "coordinate_system": recording.coordinate_system,
        "source": recording.source,
        "columns": columns,
        "data": rows,
        "coordinate_map": dict(OPENSIM_COORDINATE_ALIASES),
    }
