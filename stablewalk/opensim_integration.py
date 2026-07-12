"""
OpenSim-compatible export layer for StableWalk.
===============================================

Pipeline context
-----------------
* **MediaPipe** (``stablewalk/pose/``) extracts body landmarks from the walking
  video. Joint angles, DOF, stability, and skeleton visualization all come from
  that pose-estimation pipeline.
* **OpenSim** represents the same motion *biomechanically*. This module
  converts MediaPipe pose data into OpenSim-compatible artifacts that can be
  opened in the OpenSim application:
    - marker trajectory file (``.trc``),
    - joint-angle motion table (``.mot`` Storage format, or ``.csv``),
    - self-describing JSON bundle with biomechanical metadata.

This module never replaces MediaPipe — it only *reads* the ``PoseSequence`` the
rest of the project already produces.

Export vs real OpenSim SDK
--------------------------
* **Always available (no SDK):** ``export_trc_file``, ``export_motion_file``,
  ``export_opensim_ready_json`` — pure Python writers, no ``opensim`` import
  required at runtime for export.
* **Requires SDK + .osim model:** model loading and inverse kinematics live in
  ``stablewalk/opensim_sdk.py``. See :func:`run_inverse_kinematics` below for the
  optional SDK hook in this module.

OpenSim SDK is optional
-----------------------
The OpenSim Python SDK (``opensim``) ships through conda
(``conda install -c opensim-org opensim``) and is **not** a pip wheel, so it may
not be installed. This module uses a guarded import:

    try:
        import opensim as osim
        OPENSIM_AVAILABLE = True
    except ImportError:
        OPENSIM_AVAILABLE = False

If the SDK is **not** available, export still works and the GUI shows
"Compatible export only". The app never crashes when OpenSim is missing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

from stablewalk.opensim_sdk import (  # noqa: E402 — shared SDK probe
    OPENSIM_AVAILABLE,
    check_opensim_available,
    check_opensim_sdk,
    osim,
)


# ---------------------------------------------------------------------------
# MediaPipe landmark name -> OpenSim-style marker name (requirement #3)
# ---------------------------------------------------------------------------
# These names follow the convention requested for StableWalk. They map the
# MediaPipe Pose landmark names (lowercase) onto short anatomical marker labels
# used in the exported .trc file. Extend freely to match a specific OpenSim
# model's MarkerSet.
MEDIAPIPE_TO_OPENSIM_MARKERS: dict[str, str] = {
    # Core gait markers (explicitly requested)
    "left_shoulder": "L_SHOULDER",
    "right_shoulder": "R_SHOULDER",
    "left_hip": "L_HIP",
    "right_hip": "R_HIP",
    "left_knee": "L_KNEE",
    "right_knee": "R_KNEE",
    "left_ankle": "L_ANKLE",
    "right_ankle": "R_ANKLE",
    "left_heel": "L_HEEL",
    "right_heel": "R_HEEL",
    "left_foot_index": "L_TOE",
    "right_foot_index": "R_TOE",
    # Useful extras (upper limb + head/pelvis) for a fuller skeleton
    "left_elbow": "L_ELBOW",
    "right_elbow": "R_ELBOW",
    "left_wrist": "L_WRIST",
    "right_wrist": "R_WRIST",
    "nose": "HEAD",
    "mid_hip": "PELVIS",
    "mid_shoulder": "THORAX",
}

# JointAngles field name -> OpenSim coordinate label for the .mot table.
# Mirrors OpenSim gait-model coordinate naming (``*_l`` / ``*_r`` suffixes).
ANGLE_TO_OPENSIM_COORDINATE: dict[str, str] = {
    "left_hip": "hip_flexion_l",
    "right_hip": "hip_flexion_r",
    "left_knee": "knee_angle_l",
    "right_knee": "knee_angle_r",
    "left_ankle": "ankle_angle_l",
    "right_ankle": "ankle_angle_r",
    "left_ankle_flexion": "ankle_angle_l",
    "right_ankle_flexion": "ankle_angle_r",
    "left_shoulder": "arm_flex_l",
    "right_shoulder": "arm_flex_r",
    "left_elbow": "elbow_flex_l",
    "right_elbow": "elbow_flex_r",
    "left_wrist": "wrist_flex_l",
    "right_wrist": "wrist_flex_r",
    "neck": "neck_flexion",
    "head_neck": "head_flexion",
    "torso_tilt": "lumbar_extension",
    "pelvis_rotation": "pelvis_rotation",
    "spine": "lumbar_bending",
}

TRC_UNITS = "mm"  # OpenSim .trc convention is millimeters
DEFAULT_SUBJECT_HEIGHT_M = 1.70


# ---------------------------------------------------------------------------
# Normalized data structures (decoupled from the rest of the project)
# ---------------------------------------------------------------------------
@dataclass
class MarkerFrame:
    """One time step of marker data, already in OpenSim marker naming."""

    frame: int
    time_s: float
    markers: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    angles: dict[str, float] = field(default_factory=dict)


@dataclass
class OpenSimMotion:
    """A full processed recording, ready to write to .trc / .mot / .json."""

    name: str
    fps: float
    marker_order: list[str]
    frames: list[MarkerFrame]
    units: str = TRC_UNITS
    coordinate_system: str = "y_up_meters_to_mm"
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Availability check (requirement #7)
# ---------------------------------------------------------------------------
def check_opensim_availability(*, refresh: bool = True) -> dict[str, Any]:
    """
    Report whether the OpenSim Python SDK is importable.

    Returns a small dict so callers (CLI/GUI) can show a clear message. Never
    raises, even on a broken/partial OpenSim install.
    """
    status = check_opensim_sdk(refresh=refresh)
    info: dict[str, Any] = {
        "available": status.available,
        "version": status.version,
        "message": status.message,
        "module_path": None,
    }
    if status.available:
        _, osim_mod = check_opensim_available()
        if osim_mod is not None and not isinstance(osim_mod, str):
            info["module_path"] = getattr(osim_mod, "__file__", None)
    else:
        info["hint"] = (
            "OpenSim SDK not installed — compatible export only. "
            "To enable the live API: conda install -c opensim-org opensim"
        )
    return info


# ---------------------------------------------------------------------------
# Marker mapping (requirement #7: map_mediapipe_to_opensim_markers)
# ---------------------------------------------------------------------------
def map_mediapipe_to_opensim_markers(
    landmarks: Mapping[str, Sequence[float]],
    *,
    mapping: Mapping[str, str] | None = None,
) -> dict[str, tuple[float, float, float]]:
    """
    Map one frame of MediaPipe landmarks to OpenSim marker names.

    Args:
        landmarks: ``{mediapipe_landmark_name: (x, y, z)}`` for a single frame.
            Coordinates may be normalized (MediaPipe) or already metric.
        mapping: Optional override for :data:`MEDIAPIPE_TO_OPENSIM_MARKERS`.

    Returns:
        ``{opensim_marker_name: (x, y, z)}``. Landmarks without a mapping entry
        are skipped (so unrelated MediaPipe points do not pollute the markerset).
    """
    table = mapping or MEDIAPIPE_TO_OPENSIM_MARKERS
    out: dict[str, tuple[float, float, float]] = {}
    for name, xyz in landmarks.items():
        marker = table.get(name)
        if marker is None or xyz is None:
            continue
        x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2] if len(xyz) > 2 else 0.0))
        out[marker] = (x, y, z)
    return out


from stablewalk.coordinates.coordinate_map import (
    estimate_mediapipe_to_meter_scale,
    mediapipe_to_opensim_trc_m,
)


def _convert_axes(x: float, y: float, z: float, scale: float) -> tuple[float, float, float]:
    """Delegate to centralized OpenSim TRC conversion (see ``coordinate_map``)."""
    return mediapipe_to_opensim_trc_m(x, y, z, scale)


def _estimate_scale(frames: Iterable[MarkerFrame], subject_height_m: float) -> float:
    """Estimate normalized→meter scale from marker vertical span in image space."""
    max_span = 0.0
    for f in frames:
        ys = [xyz[1] for xyz in f.markers.values()]
        if len(ys) >= 2:
            max_span = max(max_span, max(ys) - min(ys))
    return estimate_mediapipe_to_meter_scale(
        [0.0, max_span] if max_span > 1e-6 else [],
        subject_height_m=subject_height_m,
    )


# ---------------------------------------------------------------------------
# Build the normalized OpenSimMotion from the project's PoseSequence
# ---------------------------------------------------------------------------
def build_motion_from_pose_sequence(
    sequence: Any,
    *,
    name: str = "stablewalk_motion",
    subject_height_m: float = DEFAULT_SUBJECT_HEIGHT_M,
    detected_only: bool = True,
    to_millimeters: bool = True,
) -> OpenSimMotion:
    """
    Convert a StableWalk ``PoseSequence`` into an :class:`OpenSimMotion`.

    The ``PoseSequence`` / ``PoseFrame`` objects come straight from the existing
    MediaPipe pipeline (``stablewalk.models.pose_data``). This function reads
    them without modifying them, so the dashboard is unaffected.

    Each frame contributes:
        - marker positions (mapped + axis/units converted),
        - joint angles (degrees) keyed by OpenSim coordinate names.
    """
    fps = float(getattr(sequence, "fps", 30.0)) or 30.0
    raw_frames = list(getattr(sequence, "frames", []) or [])

    # First pass: gather mapped markers in normalized space (for scale estimate).
    pre: list[MarkerFrame] = []
    for pf in raw_frames:
        if detected_only and not getattr(pf, "detected", False):
            continue
        landmark_xyz: dict[str, tuple[float, float, float]] = {}
        for kp in getattr(pf, "keypoints", []) or []:
            landmark_xyz[kp.name] = (kp.x, kp.y, kp.z)
        markers = map_mediapipe_to_opensim_markers(landmark_xyz)
        angles = _extract_angles(getattr(pf, "joint_angles", None))
        pre.append(
            MarkerFrame(
                frame=int(getattr(pf, "frame_index", len(pre))),
                time_s=_frame_time(pf, fps),
                markers=markers,
                angles=angles,
            )
        )

    scale = _estimate_scale(pre, subject_height_m)
    mm = 1000.0 if to_millimeters else 1.0

    # Second pass: apply axis + unit conversion.
    converted: list[MarkerFrame] = []
    for mf in pre:
        markers = {
            marker: tuple(c * mm for c in _convert_axes(x, y, z, scale))  # type: ignore[misc]
            for marker, (x, y, z) in mf.markers.items()
        }
        converted.append(
            MarkerFrame(frame=mf.frame, time_s=mf.time_s, markers=markers, angles=mf.angles)
        )

    marker_order = _ordered_marker_names(converted)
    return OpenSimMotion(
        name=name,
        fps=fps,
        marker_order=marker_order,
        frames=converted,
        units=TRC_UNITS if to_millimeters else "m",
        source=str(getattr(sequence, "source_video", "")),
        metadata={
            "subject_height_m": subject_height_m,
            "normalized_to_meter_scale": scale,
            "detected_frames": len(converted),
            "total_frames": len(raw_frames),
            "opensim_sdk_available": OPENSIM_AVAILABLE,
        },
    )


def _frame_time(pose_frame: Any, fps: float) -> float:
    fn = getattr(pose_frame, "time_seconds", None)
    if callable(fn):
        try:
            return float(fn(fps))
        except Exception:
            pass
    return float(getattr(pose_frame, "frame_index", 0)) / max(fps, 1e-6)


def _extract_angles(joint_angles: Any) -> dict[str, float]:
    """Pull populated joint angles and key them by OpenSim coordinate name."""
    if joint_angles is None:
        return {}
    out: dict[str, float] = {}
    for field_name, coord in ANGLE_TO_OPENSIM_COORDINATE.items():
        val = getattr(joint_angles, field_name, None)
        if val is None:
            continue
        # Prefer the first non-null mapping for each coordinate.
        out.setdefault(coord, float(val))
    return out


def _ordered_marker_names(frames: Sequence[MarkerFrame]) -> list[str]:
    """Stable marker column order: requested gait order first, then extras."""
    seen: set[str] = set()
    for f in frames:
        seen.update(f.markers.keys())
    preferred = list(MEDIAPIPE_TO_OPENSIM_MARKERS.values())
    ordered = [m for m in preferred if m in seen]
    ordered += sorted(m for m in seen if m not in set(ordered))
    return ordered


# ---------------------------------------------------------------------------
# .trc writer (requirement #7: export_trc_file)
# ---------------------------------------------------------------------------
def export_trc_file(motion: OpenSimMotion, path: str | Path) -> Path:
    """
    Write marker trajectories as an OpenSim ``.trc`` file.

    The ``.trc`` (Track Row Column) format is a tab-delimited text file with a
    fixed 5-line header. It opens directly in the OpenSim GUI and is the input
    to the Inverse Kinematics tool. No SDK required to write it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    markers = motion.marker_order
    n_markers = len(markers)
    n_frames = len(motion.frames)
    rate = motion.fps

    lines: list[str] = []
    # Header line 1
    lines.append(f"PathFileType\t4\t(X/Y/Z)\t{path.name}")
    # Header line 2 (column titles)
    lines.append(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    )
    # Header line 3 (values)
    lines.append(
        f"{rate:.6f}\t{rate:.6f}\t{n_frames}\t{n_markers}\t{motion.units}\t"
        f"{rate:.6f}\t1\t{n_frames}"
    )
    # Header line 4: Frame#, Time, then each marker name spanning 3 columns
    marker_header = "Frame#\tTime\t" + "\t\t\t".join(markers) + "\t\t"
    lines.append(marker_header)
    # Header line 5: per-axis sub-columns X1 Y1 Z1 X2 Y2 Z2 ...
    axis_header = "\t\t" + "\t".join(
        f"X{i}\tY{i}\tZ{i}" for i in range(1, n_markers + 1)
    )
    lines.append(axis_header)
    lines.append("")  # blank line before data, per TRC convention

    for i, mf in enumerate(motion.frames, start=1):
        cells: list[str] = [str(i), f"{mf.time_s:.6f}"]
        for marker in markers:
            xyz = mf.markers.get(marker)
            if xyz is None:
                cells += ["", "", ""]  # missing marker -> blank (gap)
            else:
                cells += [f"{xyz[0]:.6f}", f"{xyz[1]:.6f}", f"{xyz[2]:.6f}"]
        lines.append("\t".join(cells))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote OpenSim marker file (.trc) -> %s", path)
    return path


# ---------------------------------------------------------------------------
# .mot / CSV writer (requirement #7: export_motion_file)
# ---------------------------------------------------------------------------
def export_motion_file(motion: OpenSimMotion, path: str | Path) -> Path:
    """
    Write joint angles as an OpenSim ``.mot`` Storage file (or CSV).

    If ``path`` ends in ``.csv`` a comma-separated table is written instead of
    the native Storage format (requirement #4 fallback). The ``.mot`` header
    follows OpenSim's Storage convention with ``inDegrees=yes``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    coord_names = _ordered_coordinate_names(motion.frames)
    columns = ["time"] + coord_names

    rows: list[list[float]] = []
    for mf in motion.frames:
        row = [mf.time_s]
        for coord in coord_names:
            row.append(float(mf.angles.get(coord, 0.0)))
        rows.append(row)

    if path.suffix.lower() == ".csv":
        import csv

        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([f"{v:.6f}" for v in row])
        logger.info("Wrote joint-angle table (.csv) -> %s", path)
        return path

    # Native OpenSim Storage (.mot) format
    lines: list[str] = []
    lines.append(motion.name)
    lines.append("version=1")
    lines.append(f"nRows={len(rows)}")
    lines.append(f"nColumns={len(columns)}")
    lines.append("inDegrees=yes")
    lines.append(
        "% StableWalk export: joint angles from MediaPipe pose, "
        "OpenSim-compatible coordinate names."
    )
    lines.append("endheader")
    lines.append("\t".join(columns))
    for row in rows:
        lines.append("\t".join(f"{v:.6f}" for v in row))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote OpenSim motion file (.mot) -> %s", path)
    return path


def _ordered_coordinate_names(frames: Sequence[MarkerFrame]) -> list[str]:
    seen: set[str] = set()
    for f in frames:
        seen.update(f.angles.keys())
    preferred = list(dict.fromkeys(ANGLE_TO_OPENSIM_COORDINATE.values()))
    ordered = [c for c in preferred if c in seen]
    ordered += sorted(c for c in seen if c not in set(ordered))
    return ordered


# ---------------------------------------------------------------------------
# JSON writer (requirement #7: export_opensim_ready_json)
# ---------------------------------------------------------------------------
def export_opensim_ready_json(
    motion: OpenSimMotion,
    path: str | Path,
    *,
    stability: dict[str, Any] | None = None,
    selected_dof: Sequence[str] | None = None,
) -> Path:
    """
    Write a self-describing JSON bundle with all processed biomechanical data.

    Contains the marker mapping, per-frame marker positions, per-frame joint
    angles (OpenSim coordinate names), units, and metadata. Useful for review,
    re-import, or feeding a custom OpenSim script.

    Args:
        stability: Optional walking-stability report (e.g. from
            ``stablewalk.analysis.biomech_stability``) to embed alongside the
            motion data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema": "stablewalk-opensim-export",
        "version": "1.0",
        "name": motion.name,
        "source_video": motion.source,
        "fps": motion.fps,
        "units": motion.units,
        "coordinate_system": motion.coordinate_system,
        "pose_engine": "mediapipe",
        "biomechanics_layer": "opensim",
        "opensim_sdk_available": OPENSIM_AVAILABLE,
        "marker_mapping": MEDIAPIPE_TO_OPENSIM_MARKERS,
        "coordinate_mapping": ANGLE_TO_OPENSIM_COORDINATE,
        "marker_order": motion.marker_order,
        "selected_dof": list(selected_dof) if selected_dof else [],
        "stability": stability,
        "metadata": motion.metadata,
        "frames": [
            {
                "frame": mf.frame,
                "time_s": round(mf.time_s, 6),
                "markers": {
                    m: [round(c, 6) for c in xyz] for m, xyz in mf.markers.items()
                },
                "joint_angles_deg": {k: round(v, 4) for k, v in mf.angles.items()},
            }
            for mf in motion.frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote OpenSim-ready JSON -> %s", path)
    return path


# ---------------------------------------------------------------------------
# High-level orchestration: one call from the pipeline / GUI
# ---------------------------------------------------------------------------
def export_opensim_files(
    sequence: Any,
    output_dir: str | Path,
    *,
    name: str = "stablewalk_motion",
    subject_height_m: float = DEFAULT_SUBJECT_HEIGHT_M,
    motion_as_csv: bool = False,
    stability: dict[str, Any] | None = None,
    selected_dof: Sequence[str] | None = None,
) -> dict[str, Path]:
    """
    Convert a ``PoseSequence`` and write all OpenSim artifacts to ``output_dir``.

    Produces:
        - ``<name>.trc``         marker trajectories
        - ``<name>.mot``/``.csv`` joint-angle coordinate table
        - ``<name>_opensim.json`` full biomechanical bundle

    Returns a dict of the written paths. Safe to call whether or not the OpenSim
    SDK is installed.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    motion = build_motion_from_pose_sequence(
        sequence, name=name, subject_height_m=subject_height_m
    )

    trc_path = export_trc_file(motion, output_dir / f"{name}.trc")
    motion_ext = ".csv" if motion_as_csv else ".mot"
    mot_path = export_motion_file(motion, output_dir / f"{name}{motion_ext}")
    json_path = export_opensim_ready_json(
        motion,
        output_dir / f"{name}_opensim.json",
        stability=stability,
        selected_dof=selected_dof,
    )

    return {"trc": trc_path, "motion": mot_path, "json": json_path}


def export_from_pose_json(
    pose_json_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    name: str | None = None,
    motion_as_csv: bool = False,
) -> dict[str, Path]:
    """
    Convenience: load a pose JSON produced by the pipeline and export OpenSim files.

    Lets you run the OpenSim layer standalone on any previously processed video.
    """
    from stablewalk.visualization import load_pose_sequence

    pose_json_path = Path(pose_json_path)
    sequence = load_pose_sequence(pose_json_path)
    if name is None:
        name = pose_json_path.stem.replace("_poses", "") or "stablewalk_motion"
    if output_dir is None:
        output_dir = pose_json_path.parent / "opensim"
    return export_opensim_files(
        sequence, output_dir, name=name, motion_as_csv=motion_as_csv
    )


# ---------------------------------------------------------------------------
# Optional: real OpenSim API hook (only used when SDK is installed)
# ---------------------------------------------------------------------------
def run_inverse_kinematics(
    model_path: str | Path,
    trc_path: str | Path,
    output_mot_path: str | Path,
    *,
    time_range: tuple[float, float] | None = None,
) -> Path:
    """
    Run OpenSim Inverse Kinematics on an exported .trc (requires the SDK).

    This is the integration point for a real OpenSim environment. It needs a
    scaled ``.osim`` model whose MarkerSet matches the marker names produced by
    :data:`MEDIAPIPE_TO_OPENSIM_MARKERS`.

    Raises:
        RuntimeError: if the OpenSim SDK is not available.
    """
    if not OPENSIM_AVAILABLE:
        raise RuntimeError(
            "OpenSim SDK not installed. Install with "
            "'conda install -c opensim-org opensim' to run Inverse Kinematics. "
            "The .trc/.mot/.json files were still exported and can be opened in "
            "the OpenSim GUI manually."
        )

    # pragma: no cover - requires the OpenSim runtime
    model = osim.Model(str(model_path))  # type: ignore[union-attr]
    ik_tool = osim.InverseKinematicsTool()  # type: ignore[union-attr]
    ik_tool.setModel(model)
    ik_tool.setMarkerDataFileName(str(trc_path))
    ik_tool.setOutputMotionFileName(str(output_mot_path))
    if time_range is not None:
        ik_tool.setStartTime(time_range[0])
        ik_tool.setEndTime(time_range[1])
    ik_tool.run()
    logger.info("OpenSim IK complete -> %s", output_mot_path)
    return Path(output_mot_path)


__all__ = [
    "OPENSIM_AVAILABLE",
    "MEDIAPIPE_TO_OPENSIM_MARKERS",
    "ANGLE_TO_OPENSIM_COORDINATE",
    "MarkerFrame",
    "OpenSimMotion",
    "check_opensim_availability",
    "check_opensim_available",
    "map_mediapipe_to_opensim_markers",
    "build_motion_from_pose_sequence",
    "export_trc_file",
    "export_motion_file",
    "export_opensim_ready_json",
    "export_opensim_files",
    "export_from_pose_json",
    "run_inverse_kinematics",
]
