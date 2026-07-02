"""
Data types for the biomechanics / OpenSim adapter layer.

These structures describe motion in a form that can be exchanged with OpenSim
(``.sto`` / ``.mot`` coordinate files, marker trajectories, joint maps) while
remaining independent of the OpenSim SDK at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

OpenSimFileKind = Literal["sto", "mot", "json", "unknown"]


@dataclass(frozen=True)
class OpenSimJointMapping:
    """One canonical skeleton joint mapped to an OpenSim body / marker name."""

    canonical_joint_id: str
    opensim_body_name: str
    opensim_marker_name: str | None = None
    notes: str = ""


@dataclass
class OpenSimMotionTable:
    """
    In-memory OpenSim-style coordinate table (angle degrees over time).

    Matches the column layout of a typical OpenSim ``.sto`` file:
    ``time`` followed by generalized coordinate names.
    """

    columns: list[str]
    rows: list[list[float]]
    coordinate_system: str = "pelvis_centered_y_up_z_forward"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenSimMarkerTable:
    """
    Placeholder for OpenSim marker / body-fixed point trajectories (XYZ over time).

    Real OpenSim integration will populate this from inverse kinematics or
    experimental marker files (``.trc``). Not required for angle-only ``.sto`` export.
    """

    marker_names: list[str] = field(default_factory=list)
    times_s: list[float] = field(default_factory=list)
    positions: dict[str, list[tuple[float, float, float]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenSimExportBundle:
    """
    Complete export payload produced by ``export_to_opensim_format``.

    Suitable for JSON serialization today; future work writes ``.sto`` / ``.mot``
    directly via the OpenSim API or file writers.
    """

    version: str
    coordinate_table: OpenSimMotionTable
    marker_table: OpenSimMarkerTable | None = None
    joint_mappings: list[OpenSimJointMapping] = field(default_factory=list)
    coordinate_map: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenSimMotionData:
    """
    Normalized motion loaded from an OpenSim-style source.

    Populated by ``load_opensim_motion_data`` once ``.sto`` / ``.mot`` parsing
    or SDK import is implemented. Until then, round-trip JSON exports are supported.
    """

    source_path: str
    file_kind: OpenSimFileKind
    fps: float
    coordinate_table: OpenSimMotionTable
    marker_table: OpenSimMarkerTable | None = None
    joint_mappings: list[OpenSimJointMapping] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
