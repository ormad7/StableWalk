"""
Biomechanics service layer — skeleton motion ↔ OpenSim-ready formats.

This module is the integration boundary for future OpenSim work:

  - **Today:** accepts ``GaitMotionRecording`` / ``SkeletonSnapshot`` from mock
    data or pose-estimation adapters; exports in-memory OpenSim-style tables;
    computes angles/DOFs from existing samples.
  - **Future:** replace placeholder loaders with ``.sto`` / ``.mot`` parsers or
    the OpenSim Python/API bindings; run IK/FK through an OpenSim ``Model``;
    write marker ``.trc`` and model-scaled coordinates.

The GUI and playback pipeline should depend on ``GaitMotionRecording`` only.
Call this service when exporting, importing, or validating biomechanical data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from stablewalk.adapters.opensim_schema import (
    OPENSIM_EXPORT_VERSION,
    motion_to_opensim_table,
    opensim_column_name,
)
from stablewalk.biomechanics.mappings import (
    CANONICAL_TO_OPENSIM_BODY,
    default_joint_mappings,
)
from stablewalk.biomechanics.types import (
    OpenSimExportBundle,
    OpenSimJointMapping,
    OpenSimMarkerTable,
    OpenSimMotionData,
    OpenSimMotionTable,
)
from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, DOF_JOINT, JOINT_IDS, ROOT_JOINT_ID

# ---------------------------------------------------------------------------
# Placeholder hooks — replace with OpenSim SDK calls when available
# ---------------------------------------------------------------------------

_OPENSIM_SDK_AVAILABLE = False
"""
Set to ``True`` when ``opensim`` Python bindings are installed and a Model can
be loaded. Search the codebase for this flag to find integration points.
"""


class BiomechanicsService:
    """
    Adapter/service for biomechanical motion data and OpenSim interchange.

    All public methods are safe to call without the OpenSim SDK installed.
    """

    def export_to_opensim_format(
        self,
        recording: GaitMotionRecording,
        *,
        include_markers: bool = False,
    ) -> OpenSimExportBundle:
        """
        Convert a ``GaitMotionRecording`` into an OpenSim-style export bundle.

        **Purpose:** produce motion data that can later be written to OpenSim
        ``.sto`` / ``.mot`` coordinate files or fed to an OpenSim ``Model``.

        **Current behaviour:**
        - Builds an angle table (time + generalized coordinates) via
          ``motion_to_opensim_table``.
        - Attaches canonical → OpenSim joint mappings.
        - Optionally includes a marker trajectory placeholder (empty until IK).

        **Future OpenSim integration:**
        - Instantiate ``opensim.Model`` from ``model.osim``.
        - Use ``StatesTrajectory`` / ``Storage`` to write native ``.sto``.
        - Populate ``marker_table`` from body-fixed stations or experimental markers.
        """
        raw_table = motion_to_opensim_table(recording)
        coordinate_table = OpenSimMotionTable(
            columns=list(raw_table["columns"]),
            rows=[list(row) for row in raw_table["data"]],
            coordinate_system=str(raw_table.get("coordinate_system", "")),
            source=str(raw_table.get("source", recording.source)),
            metadata={"export_version": raw_table.get("version", OPENSIM_EXPORT_VERSION)},
        )

        marker_table: OpenSimMarkerTable | None = None
        if include_markers:
            marker_table = self._build_marker_table_placeholder(recording)

        return OpenSimExportBundle(
            version=OPENSIM_EXPORT_VERSION,
            coordinate_table=coordinate_table,
            marker_table=marker_table,
            joint_mappings=default_joint_mappings(),
            coordinate_map=dict(raw_table.get("coordinate_map", {})),
            metadata={
                "frame_count": recording.frame_count,
                "fps": recording.fps,
                "duration_s": recording.duration_s,
                "source_kind": recording.source_kind,
                "opensim_sdk_available": _OPENSIM_SDK_AVAILABLE,
            },
        )

    def load_opensim_motion_data(
        self,
        path: str | Path,
    ) -> OpenSimMotionData:
        """
        Load motion from an OpenSim-style file into ``OpenSimMotionData``.

        **Purpose:** import experimental or simulated OpenSim motion for playback,
        comparison, or re-export through the StableWalk skeleton pipeline.

        **Current behaviour:**
        - Supports JSON files previously written by ``export_to_opensim_format``
          (round-trip stub for development).
        - Raises ``NotImplementedError`` for native ``.sto`` / ``.mot`` until a
          parser or OpenSim ``Storage`` reader is wired in.

        **Future OpenSim integration:**
        - ``opensim.Storage(path)`` for ``.sto`` / ``.mot``.
        - Map column names through ``OPENSIM_COORDINATE_ALIASES``.
        - Convert to ``GaitMotionRecording`` via a dedicated inverse adapter.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".json":
            return self._load_export_json(path)

        if suffix in (".sto", ".mot"):
            raise NotImplementedError(
                f"Native OpenSim '{suffix}' import is not implemented yet. "
                "Future work: use opensim.Storage or a .sto parser, then convert "
                "columns to GaitMotionRecording via BiomechanicsService."
            )

        raise NotImplementedError(
            f"Unsupported motion file '{path.name}'. "
            "Expected .json (StableWalk export), .sto, or .mot."
        )

    def map_skeleton_to_opensim_joints(
        self,
        snapshot: SkeletonSnapshot,
    ) -> list[OpenSimJointMapping]:
        """
        Map joints present in a skeleton snapshot to OpenSim body/marker names.

        **Purpose:** document how each detected canonical joint corresponds to
        an OpenSim model segment before IK, visualization, or muscle analysis.

        **Current behaviour:**
        - Returns default mappings for joints that exist in ``snapshot.joints``.
        - Does not require OpenSim runtime.

        **Future OpenSim integration:**
        - Resolve names from the loaded ``Model.getBodySet()`` / ``MarkerSet``.
        - Validate that required bodies exist for the chosen model (gait2392, etc.).
        """
        mappings = default_joint_mappings()
        present = set(snapshot.joints.keys()) | {ROOT_JOINT_ID}
        return [m for m in mappings if m.canonical_joint_id in present]

    def calculate_joint_angles(
        self,
        snapshot: SkeletonSnapshot,
    ) -> dict[str, float | None]:
        """
        Compute or retrieve flexion/extension angles (degrees) for each joint.

        **Purpose:** provide joint angles for analysis tables, OpenSim coordinate
        columns, and GUI DOF panels when only positions are available.

        **Current behaviour:**
        - Uses ``JointSample.angle_deg`` when already populated (mock / pose JSON).
        - Estimates knee and elbow angles from segment vectors when missing.

        **Future OpenSim integration:**
        - Delegate to ``opensim.Model`` joint coordinates or an IK solver.
        - Replace heuristic geometry with model-specific joint definitions.
        """
        angles: dict[str, float | None] = {}

        for joint_id in JOINT_IDS:
            sample = snapshot.joints.get(joint_id)
            if sample and sample.angle_deg is not None:
                angles[joint_id] = sample.angle_deg
                continue
            angles[joint_id] = self._estimate_joint_angle(snapshot, joint_id)

        return angles

    def calculate_degrees_of_freedom(
        self,
        snapshot: SkeletonSnapshot,
    ) -> dict[str, DofSample]:
        """
        Build canonical DOF samples (angles and optional angular rates).

        **Purpose:** unify GUI DOF selection, real-time tables, and OpenSim
        coordinate export on a single DOF dictionary.

        **Current behaviour:**
        - Prefers existing ``snapshot.dofs`` entries.
        - Fills missing DOFs from ``calculate_joint_angles`` via ``DOF_JOINT`` map.

        **Future OpenSim integration:**
        - Read all coordinates from OpenSim ``State`` (not only flexion).
        - Include pelvis translations/rotations as separate DOFs when model requires.
        """
        dofs: dict[str, DofSample] = dict(snapshot.dofs)

        joint_angles = self.calculate_joint_angles(snapshot)
        for dof_id in DOF_IDS:
            if dof_id in dofs and dofs[dof_id].angle_deg is not None:
                continue
            anchor = DOF_JOINT.get(dof_id)
            angle = joint_angles.get(anchor) if anchor else None
            existing = dofs.get(dof_id)
            dofs[dof_id] = DofSample(
                dof_id=dof_id,
                angle_deg=angle,
                velocity_deg_s=existing.velocity_deg_s if existing else None,
                joint_id=anchor,
            )

        return dofs

    def recording_to_opensim_ready(
        self,
        recording: GaitMotionRecording,
    ) -> GaitMotionRecording:
        """
        Enrich a recording with computed DOFs/angles without changing the GUI contract.

        **Purpose:** ensure every snapshot has consistent DOF data before export.

        Returns a new ``GaitMotionRecording`` with enriched snapshots (non-destructive).
        """
        enriched_snapshots: list[SkeletonSnapshot] = []
        for snap in recording.snapshots:
            dofs = self.calculate_degrees_of_freedom(snap)
            joints = dict(snap.joints)
            for joint_id, angle in self.calculate_joint_angles(snap).items():
                if angle is None or joint_id not in joints:
                    continue
                old = joints[joint_id]
                if old.angle_deg is None:
                    joints[joint_id] = JointSample(
                        joint_id=old.joint_id,
                        position=old.position,
                        parent_id=old.parent_id,
                        angle_deg=angle,
                        velocity=old.velocity,
                        velocity_vector=old.velocity_vector,
                    )
            enriched_snapshots.append(
                SkeletonSnapshot(
                    frame_index=snap.frame_index,
                    time_s=snap.time_s,
                    joints=joints,
                    dofs=dofs,
                    metadata=dict(snap.metadata),
                )
            )

        out = GaitMotionRecording(
            source=recording.source,
            fps=recording.fps,
            snapshots=enriched_snapshots,
            source_kind=recording.source_kind,
            coordinate_system=recording.coordinate_system,
            metadata=dict(recording.metadata),
        )
        out.build_time_series()
        return out

    def write_opensim_json(
        self,
        bundle: OpenSimExportBundle,
        path: str | Path,
    ) -> Path:
        """
        Write ``export_to_opensim_format`` output as JSON (development interchange).

        **Future:** add ``write_opensim_sto`` using OpenSim ``Storage`` or a file writer.
        """
        path = Path(path)
        payload = {
            "version": bundle.version,
            "coordinate_table": {
                "columns": bundle.coordinate_table.columns,
                "rows": bundle.coordinate_table.rows,
                "coordinate_system": bundle.coordinate_table.coordinate_system,
                "source": bundle.coordinate_table.source,
            },
            "joint_mappings": [
                {
                    "canonical_joint_id": m.canonical_joint_id,
                    "opensim_body_name": m.opensim_body_name,
                    "opensim_marker_name": m.opensim_marker_name,
                }
                for m in bundle.joint_mappings
            ],
            "coordinate_map": bundle.coordinate_map,
            "metadata": bundle.metadata,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_export_json(self, path: Path) -> OpenSimMotionData:
        data = json.loads(path.read_text(encoding="utf-8"))
        table_raw = data.get("coordinate_table", {})
        coordinate_table = OpenSimMotionTable(
            columns=list(table_raw.get("columns", [])),
            rows=[list(r) for r in table_raw.get("rows", [])],
            coordinate_system=str(table_raw.get("coordinate_system", "")),
            source=str(table_raw.get("source", str(path))),
        )
        mappings = [
            OpenSimJointMapping(
                canonical_joint_id=str(m["canonical_joint_id"]),
                opensim_body_name=str(m["opensim_body_name"]),
                opensim_marker_name=m.get("opensim_marker_name"),
            )
            for m in data.get("joint_mappings", [])
        ]
        fps = float(data.get("metadata", {}).get("fps", 30.0))
        return OpenSimMotionData(
            source_path=str(path),
            file_kind="json",
            fps=fps,
            coordinate_table=coordinate_table,
            joint_mappings=mappings,
            metadata=dict(data.get("metadata", {})),
        )

    def _build_marker_table_placeholder(
        self,
        recording: GaitMotionRecording,
    ) -> OpenSimMarkerTable:
        """
        Placeholder marker trajectories from canonical joint positions.

        Future: replace with OpenSim ``MarkerSet`` / experimental ``.trc`` data.
        """
        ts = recording.build_time_series()
        marker_names = sorted(
            {
                m.opensim_marker_name
                for m in default_joint_mappings()
                if m.opensim_marker_name
            }
        )
        positions: dict[str, list[tuple[float, float, float]]] = {
            name: [] for name in marker_names
        }
        name_to_joint = {
            m.opensim_marker_name: m.canonical_joint_id
            for m in default_joint_mappings()
            if m.opensim_marker_name
        }
        for marker_name, joint_id in name_to_joint.items():
            series = ts.positions.get(joint_id, [])
            positions[marker_name] = [(p.x, p.y, p.z) for p in series]

        return OpenSimMarkerTable(
            marker_names=marker_names,
            times_s=list(ts.times_s),
            positions=positions,
            metadata={"placeholder": True, "note": "Replace with OpenSim IK markers"},
        )

    @staticmethod
    def _estimate_joint_angle(
        snapshot: SkeletonSnapshot,
        joint_id: str,
    ) -> float | None:
        """Simple geometric angle estimate from parent-child segment vectors."""
        from stablewalk.models.joint_registry import JOINT_PARENTS

        parent_id = JOINT_PARENTS.get(joint_id)
        if not parent_id:
            return None

        child = snapshot.joints.get(joint_id)
        parent = snapshot.joints.get(parent_id)
        if not child or not parent:
            return None

        # Grandparent for joint angle at knee/elbow
        grandparent_id = JOINT_PARENTS.get(parent_id)
        if not grandparent_id:
            return None
        grandparent = snapshot.joints.get(grandparent_id)
        if not grandparent:
            return None

        return _angle_between_segments(
            grandparent.position,
            parent.position,
            child.position,
        )


def _angle_between_segments(a: Vec3, b: Vec3, c: Vec3) -> float | None:
    """Angle at point ``b`` formed by segments ba and bc, in degrees."""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    n1 = math.sqrt(sum(x * x for x in v1))
    n2 = math.sqrt(sum(x * x for x in v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    dot = sum(v1[i] * v2[i] for i in range(3)) / (n1 * n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


# Module-level singleton for convenience (GUI, scripts, tests)
_default_service = BiomechanicsService()


def export_to_opensim_format(
    recording: GaitMotionRecording,
    **kwargs: Any,
) -> OpenSimExportBundle:
    """See ``BiomechanicsService.export_to_opensim_format``."""
    return _default_service.export_to_opensim_format(recording, **kwargs)


def load_opensim_motion_data(path: str | Path) -> OpenSimMotionData:
    """See ``BiomechanicsService.load_opensim_motion_data``."""
    return _default_service.load_opensim_motion_data(path)


def map_skeleton_to_opensim_joints(
    snapshot: SkeletonSnapshot,
) -> list[OpenSimJointMapping]:
    """See ``BiomechanicsService.map_skeleton_to_opensim_joints``."""
    return _default_service.map_skeleton_to_opensim_joints(snapshot)


def calculate_joint_angles(snapshot: SkeletonSnapshot) -> dict[str, float | None]:
    """See ``BiomechanicsService.calculate_joint_angles``."""
    return _default_service.calculate_joint_angles(snapshot)


def calculate_degrees_of_freedom(snapshot: SkeletonSnapshot) -> dict[str, DofSample]:
    """See ``BiomechanicsService.calculate_degrees_of_freedom``."""
    return _default_service.calculate_degrees_of_freedom(snapshot)


# CamelCase aliases (OpenSim / JS interoperability naming)
exportToOpenSimFormat = export_to_opensim_format
loadOpenSimMotionData = load_opensim_motion_data
mapSkeletonToOpenSimJoints = map_skeleton_to_opensim_joints
calculateJointAngles = calculate_joint_angles
calculateDegreesOfFreedom = calculate_degrees_of_freedom
