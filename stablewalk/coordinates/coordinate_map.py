"""
StableWalk coordinate frames, conversions, and anatomical consistency checks.

Canonical biomechanical frame (SW_CANONICAL)
--------------------------------------------
Used by gait analysis, 3D skeleton reconstruction, foot clearance, contact
detection, joint path graphs, and ``GaitMotionRecording``.

  Origin:     pelvis (mid_hip) at (0, 0, 0)
  +X:         mediolateral — subject's right (image +x when facing the camera)
  +Y:         vertical — up
  +Z:         forward — anterior / toward camera (depth heuristic from MediaPipe)
  Units:      body-normalized meters (~1.0 = full body height span)
  Handedness: right-handed (X cross Y = Z)

OpenSim TRC export frame (OSIM_TRC) is a *separate* lab-style frame derived
directly from MediaPipe **image** landmarks (not hip-centered). See
``mediapipe_to_opensim_trc_mm``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical convention
# ---------------------------------------------------------------------------
CANONICAL_FRAME_ID = "sw_canonical_y_up"
CANONICAL_VERTICAL_AXIS = "y"
CANONICAL_UNITS = "m"
CANONICAL_ORIGIN = "pelvis_mid_hip"
CANONICAL_HANDEDNESS = "right_handed"

AXIS_X_MEANING = "mediolateral (subject right +)"
AXIS_Y_MEANING = "vertical (up +)"
AXIS_Z_MEANING = "forward (anterior / toward camera +)"

# Legacy alias used in GaitMotionRecording metadata
DEFAULT_COORDINATE_SYSTEM = "pelvis_centered_y_up_z_forward"

MP_IMAGE_FRAME_ID = "mediapipe_normalized_image"
MP_IMAGE_UNITS = "normalized"

OSIM_TRC_FRAME_ID = "opensim_trc_export_mm"
OSIM_TRC_UNITS = "mm"

# Depth layers for canonical Z (single source — skeleton_3d imports this table)
DEPTH_LAYERS: dict[str, float] = {
    "mid_hip": 0.0,
    "left_hip": 0.0,
    "right_hip": 0.0,
    "left_knee": -0.06,
    "right_knee": -0.06,
    "left_ankle": -0.12,
    "right_ankle": -0.12,
    "left_heel": -0.14,
    "right_heel": -0.14,
    "left_foot_index": -0.13,
    "right_foot_index": -0.13,
    "left_toe": -0.13,
    "right_toe": -0.13,
    "mid_shoulder": 0.04,
    "left_shoulder": 0.05,
    "right_shoulder": 0.05,
    "left_elbow": 0.08,
    "right_elbow": 0.08,
    "left_wrist": 0.10,
    "right_wrist": 0.10,
    "nose": 0.12,
    "head": 0.12,
}

TARGET_SKELETON_HEIGHT = 1.0
DEFAULT_SUBJECT_HEIGHT_M = 1.70
OBLIQUE_DISPLAY_SHEAR = 0.22


class CoordinateFrame(str, Enum):
    """Major coordinate representations in StableWalk."""

    MP_IMAGE = MP_IMAGE_FRAME_ID
    MP_WORLD = "mediapipe_world_landmarks_m"
    SW_CANONICAL = CANONICAL_FRAME_ID
    OSIM_TRC = OSIM_TRC_FRAME_ID
    OSIM_IK_MOT = "opensim_ik_generalized_coordinates"


@dataclass(frozen=True)
class FrameSpec:
    """Documented frame metadata for audits and exports."""

    frame_id: str
    coordinate_source: str
    x_axis: str
    y_axis: str
    z_axis: str
    vertical_axis: str
    forward_axis: str
    mediolateral_axis: str
    units: str
    origin: str
    handedness: str
    transformations: tuple[str, ...] = ()


@dataclass
class AnatomicalCheckResult:
    """Outcome of vertical ordering checks on one frame."""

    frame_index: int
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    joint_y: dict[str, float] = field(default_factory=dict)


FRAME_REGISTRY: dict[CoordinateFrame, FrameSpec] = {
    CoordinateFrame.MP_IMAGE: FrameSpec(
        frame_id=MP_IMAGE_FRAME_ID,
        coordinate_source="MediaPipe pose_landmarks (image, normalized)",
        x_axis="+X right (0–1 image width)",
        y_axis="+Y down (0–1 image height)",
        z_axis="+Z toward camera (relative depth)",
        vertical_axis="+Y down (image — not biomechanical up)",
        forward_axis="+Z toward camera",
        mediolateral_axis="+X right",
        units=MP_IMAGE_UNITS,
        origin="image top-left",
        handedness="right_handed",
        transformations=("none at detection",),
    ),
    CoordinateFrame.MP_WORLD: FrameSpec(
        frame_id="mediapipe_world_landmarks_m",
        coordinate_source="MediaPipe pose_world_landmarks (optional, meters)",
        x_axis="+X right in world",
        y_axis="+Y up in world",
        z_axis="+Z toward camera (MediaPipe world convention)",
        vertical_axis="+Y up",
        forward_axis="+Z toward camera",
        mediolateral_axis="+X right",
        units="m",
        origin="hip-centered in MediaPipe world space",
        handedness="right_handed",
        transformations=(
            "Not used by default StableWalk pipeline; documented for future lab export.",
        ),
    ),
    CoordinateFrame.SW_CANONICAL: FrameSpec(
        frame_id=CANONICAL_FRAME_ID,
        coordinate_source="Hip-centered 3D reconstruction (skeleton_3d / pose_adapter)",
        x_axis=f"+X {AXIS_X_MEANING}",
        y_axis=f"+Y {AXIS_Y_MEANING}",
        z_axis=f"+Z {AXIS_Z_MEANING}",
        vertical_axis=f"+{CANONICAL_VERTICAL_AXIS.upper()} {AXIS_Y_MEANING}",
        forward_axis=f"+Z {AXIS_Z_MEANING}",
        mediolateral_axis=f"+X {AXIS_X_MEANING}",
        units=f"{CANONICAL_UNITS} (body-normalized)",
        origin=CANONICAL_ORIGIN,
        handedness=CANONICAL_HANDEDNESS,
        transformations=(
            "x' = (x - cx) * scale",
            "y' = -(y - cy) * scale  # image Y-down → canonical Y-up",
            "z' = (depth_layer - mp.z * 0.2) * scale",
            "scale → TARGET_SKELETON_HEIGHT (~1 m body span)",
        ),
    ),
    CoordinateFrame.OSIM_TRC: FrameSpec(
        frame_id=OSIM_TRC_FRAME_ID,
        coordinate_source="OpenSim marker export (opensim_integration.export_trc_file)",
        x_axis="+X image horizontal scaled (not pelvis-centered)",
        y_axis="+Y up (flipped from image)",
        z_axis="+Z = -MediaPipe z scaled (depth sign flip)",
        vertical_axis="+Y up",
        forward_axis="approximate -MP z (export convention)",
        mediolateral_axis="+X image horizontal",
        units=OSIM_TRC_UNITS,
        origin="image corner (not mid_hip)",
        handedness="right_handed",
        transformations=(
            "x_os = x * scale_m * 1000",
            "y_os = (1 - y) * scale_m * 1000",
            "z_os = -z * scale_m * 1000",
            "scale_m from subject height / vertical image span",
        ),
    ),
    CoordinateFrame.OSIM_IK_MOT: FrameSpec(
        frame_id="opensim_ik_generalized_coordinates",
        coordinate_source="OpenSim IK .mot generalized coordinates",
        x_axis="N/A (joint angles, not positions)",
        y_axis="N/A",
        z_axis="N/A",
        vertical_axis="N/A",
        forward_axis="N/A",
        mediolateral_axis="N/A",
        units="deg (inDegrees=yes)",
        origin="model-defined",
        handedness="model-defined",
        transformations=("IK solver on TRC markers",),
    ),
}


def frame_spec(frame: CoordinateFrame) -> FrameSpec:
    return FRAME_REGISTRY[frame]


def axis_labels_canonical() -> dict[str, str]:
    """Matplotlib axis labels matching SW_CANONICAL."""
    return {
        "x": "X — Mediolateral (m)",
        "y": "Y — Vertical (m)",
        "z": "Z — Forward (m)",
    }


def compact_axis_legend() -> str:
    return "X mediolateral · Y vertical · Z forward"


def vertical_coordinate(
    position: Any,
    *,
    axis: str = CANONICAL_VERTICAL_AXIS,
) -> float:
    """Read height along the analysis vertical axis (+Y default)."""
    if axis == "y":
        return float(position.y)
    if axis == "z":
        return float(position.z)
    if axis == "x":
        return float(position.x)
    return float(getattr(position, axis))


def depth_from_mediapipe_landmark(joint_name: str, mp_z: float) -> float:
    """Heuristic depth layer + inverted MediaPipe relative z."""
    layer = DEPTH_LAYERS.get(joint_name, 0.0)
    return layer + (-float(mp_z) * 0.2)


def mediapipe_to_canonical(
    x: float,
    y: float,
    z: float,
    *,
    cx: float,
    cy: float,
    joint_name: str = "",
    scale: float = 1.0,
) -> tuple[float, float, float]:
    """
    Convert one MediaPipe image landmark to pelvis-centered canonical meters.

    Matches ``skeleton_3d.reconstruct_skeleton_3d`` joint placement (pre-limb ratios).
    """
    x_c = (float(x) - cx) * scale
    y_c = -(float(y) - cy) * scale
    z_c = depth_from_mediapipe_landmark(joint_name, z) * scale
    return x_c, y_c, z_c


def mediapipe_to_canonical_oblique_xy(
    x: float,
    y: float,
    z: float,
    *,
    shear: float = OBLIQUE_DISPLAY_SHEAR,
) -> tuple[float, float]:
    """Oblique 2D projection used by skeleton renderer."""
    return x + shear * z, y


def mediapipe_to_opensim_trc_m(
    x: float,
    y: float,
    z: float,
    scale_m: float,
) -> tuple[float, float, float]:
    """
    MediaPipe image landmark → OpenSim TRC meters (pre-mm).

    Single definition of the export axis transform (was duplicated in
    ``opensim_integration._convert_axes``).
    """
    x_os = float(x) * scale_m
    y_os = (1.0 - float(y)) * scale_m
    z_os = -float(z) * scale_m
    return x_os, y_os, z_os


def mediapipe_to_opensim_trc_mm(
    x: float,
    y: float,
    z: float,
    scale_m: float,
) -> tuple[float, float, float]:
    """MediaPipe image landmark → OpenSim TRC millimeters."""
    xm, ym, zm = mediapipe_to_opensim_trc_m(x, y, z, scale_m)
    return xm * 1000.0, ym * 1000.0, zm * 1000.0


def estimate_mediapipe_to_meter_scale(
    y_values: Sequence[float],
    subject_height_m: float = DEFAULT_SUBJECT_HEIGHT_M,
) -> float:
    """Scale normalized image Y span to subject height (TRC export)."""
    if len(y_values) < 2:
        return 1.0
    span = max(y_values) - min(y_values)
    if span <= 1e-6:
        return 1.0
    return subject_height_m / span


def canonical_to_visualization_oblique(
    x: float,
    y: float,
    z: float,
    *,
    shear: float = OBLIQUE_DISPLAY_SHEAR,
) -> tuple[float, float]:
    """Oblique 3D skeleton display (X–Y with depth cue)."""
    return x + shear * z, y


def canonical_to_visualization_frontal(x: float, y: float, z: float) -> tuple[float, float]:
    """Frontal projection (mediolateral × vertical)."""
    return x, y


def check_anatomical_ordering(
    joints: Mapping[str, Any],
    *,
    frame_index: int = 0,
    floor_y: float | None = None,
    tolerance_m: float = 0.02,
    stance_clearance_max_m: float = 0.08,
) -> AnatomicalCheckResult:
    """
    Verify vertical ordering on canonical +Y (head above pelvis above knee above ankle).

    When ``floor_y`` is supplied, flags feet far above the floor (stance sanity).
    """
    result = AnatomicalCheckResult(frame_index=frame_index, passed=True)

    def _y(jid: str) -> float | None:
        j = joints.get(jid)
        if j is None:
            return None
        if isinstance(j, dict):
            return float(j.get("y", j.get("Y", 0)))
        return vertical_coordinate(j)

    for jid in (
        "pelvis",
        "mid_hip",
        "head",
        "nose",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_toe",
        "right_toe",
    ):
        y = _y(jid)
        if y is not None:
            key = "pelvis" if jid in ("pelvis", "mid_hip") else ("head" if jid == "nose" else jid)
            result.joint_y.setdefault(key, y)

    pelvis_y = result.joint_y.get("pelvis")
    head_y = result.joint_y.get("head")

    def _fail(msg: str) -> None:
        result.passed = False
        result.failures.append(msg)

    def _warn(msg: str) -> None:
        result.warnings.append(msg)

    if pelvis_y is not None and head_y is not None:
        if head_y <= pelvis_y + tolerance_m:
            _fail(f"head Y ({head_y:.3f}) not above pelvis Y ({pelvis_y:.3f})")

    for side in ("left", "right"):
        knee_y = result.joint_y.get(f"{side}_knee")
        ankle_y = result.joint_y.get(f"{side}_ankle")
        if pelvis_y is not None and knee_y is not None:
            if knee_y >= pelvis_y - tolerance_m:
                _fail(f"{side} knee Y ({knee_y:.3f}) not below pelvis Y ({pelvis_y:.3f})")
        if knee_y is not None and ankle_y is not None:
            if ankle_y >= knee_y - tolerance_m:
                _fail(f"{side} ankle Y ({ankle_y:.3f}) not below {side} knee Y ({knee_y:.3f})")

    for foot, ankle_key in (("left_heel", "left_ankle"), ("left_toe", "left_ankle")):
        fy = result.joint_y.get(foot)
        ay = result.joint_y.get(ankle_key)
        if fy is not None and ay is not None and fy > ay + 0.015:
            _warn(f"{foot} Y ({fy:.3f}) above {ankle_key} Y ({ay:.3f}) — mixed coordinate source?")

    if floor_y is not None:
        foot_ys = [
            result.joint_y[k]
            for k in ("left_heel", "left_toe", "right_heel", "right_toe", "left_ankle", "right_ankle")
            if k in result.joint_y
        ]
        if foot_ys:
            lowest = min(foot_ys)
            clearance = lowest - floor_y
            if clearance > stance_clearance_max_m + 0.15:
                _warn(
                    f"lowest foot Y ({lowest:.3f}) far above floor Y ({floor_y:.3f}) "
                    f"(clearance {clearance:.3f} m) — vertical axis or floor mismatch?"
                )

    return result


def audit_recording_anatomy(
    recording: Any,
    *,
    sample_stride: int = 10,
    max_fail_frac: float = 0.25,
) -> list[str]:
    """
    Sample frames from a GaitMotionRecording; return coordinate-system warnings.
    """
    warnings: list[str] = []
    frame_count = getattr(recording, "frame_count", 0)
    if not frame_count:
        return warnings

    floor_y: float | None = None
    try:
        from stablewalk.analysis.ground_reference import estimate_ground_plane

        plane = estimate_ground_plane(recording, float(frame_count - 1))
        if plane is not None:
            floor_y = plane.floor_y
    except Exception:
        pass

    fails = 0
    checked = 0
    stride = max(1, sample_stride)
    from stablewalk.models.joint_registry import ROOT_JOINT_ID

    for index in range(0, frame_count, stride):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        joints = {jid: js.position for jid, js in snap.joints.items()}
        if ROOT_JOINT_ID in snap.joints:
            joints["pelvis"] = snap.joints[ROOT_JOINT_ID].position
        result = check_anatomical_ordering(joints, frame_index=index, floor_y=floor_y)
        checked += 1
        if not result.passed:
            fails += 1
            if len(warnings) < 5:
                warnings.append(
                    f"Frame {index}: anatomical ordering failed — "
                    + "; ".join(result.failures[:2])
                )

    if checked and fails / checked > max_fail_frac:
        warnings.insert(
            0,
            f"Coordinate-system warning: {fails}/{checked} sampled frames fail "
            f"anatomical vertical ordering (canonical +Y up). "
            f"Check mixed landmark sources or wrong vertical_axis.",
        )
    return warnings


def debug_canonical_joint_positions(
    recording: Any,
    frame_indices: Sequence[int],
) -> str:
    """Print canonical XYZ for key lower-limb joints at selected frames."""
    lines = [
        f"Canonical frame: {CANONICAL_FRAME_ID}",
        f"Vertical axis: +{CANONICAL_VERTICAL_AXIS.upper()} ({AXIS_Y_MEANING})",
        f"Mediolateral: +X ({AXIS_X_MEANING})",
        f"Forward: +Z ({AXIS_Z_MEANING})",
        f"Units: {CANONICAL_UNITS} (body-normalized)",
        "",
    ]
    joints_of_interest = (
        "pelvis",
        "left_knee",
        "left_ankle",
        "left_heel",
        "left_toe",
    )
    from stablewalk.models.joint_registry import ROOT_JOINT_ID

    for index in frame_indices:
        snap = recording.snapshot_at(index)
        if snap is None:
            lines.append(f"Frame {index}: (missing snapshot)")
            continue
        lines.append(f"Frame {index}  t={snap.time_s:.3f}s")
        for jid in joints_of_interest:
            key = ROOT_JOINT_ID if jid == "pelvis" else jid
            sample = snap.joints.get(key)
            if sample is None:
                lines.append(f"  {jid:12s}  —")
                continue
            p = sample.position
            lines.append(f"  {jid:12s}  X={p.x:+.4f}  Y={p.y:+.4f}  Z={p.z:+.4f}")
        lines.append("")
    return "\n".join(lines)


def coordinate_system_map_markdown() -> str:
    """Generate markdown documentation of all registered frames."""
    parts = ["# StableWalk coordinate system map", ""]
    parts.extend(
        [
            "## Canonical convention (SW_CANONICAL)",
            "",
            f"- **+X:** {AXIS_X_MEANING}",
            f"- **+Y:** {AXIS_Y_MEANING}",
            f"- **+Z:** {AXIS_Z_MEANING}",
            f"- **Origin:** {CANONICAL_ORIGIN}",
            f"- **Units:** {CANONICAL_UNITS} body-normalized",
            "",
        ]
    )
    for frame in CoordinateFrame:
        spec = FRAME_REGISTRY[frame]
        parts.extend(
            [
                f"## {spec.frame_id}",
                "",
                f"- **Source:** {spec.coordinate_source}",
                f"- **Origin:** {spec.origin}",
                f"- **Units:** {spec.units}",
                f"- **Handedness:** {spec.handedness}",
                f"- **+X:** {spec.x_axis}",
                f"- **+Y:** {spec.y_axis}",
                f"- **+Z:** {spec.z_axis}",
                f"- **Vertical:** {spec.vertical_axis}",
                f"- **Forward:** {spec.forward_axis}",
                f"- **Mediolateral:** {spec.mediolateral_axis}",
                "- **Transforms:**",
            ]
        )
        for t in spec.transformations:
            parts.append(f"  - {t}")
        parts.append("")
    return "\n".join(parts)


__all__ = [
    "CANONICAL_FRAME_ID",
    "CANONICAL_VERTICAL_AXIS",
    "DEFAULT_COORDINATE_SYSTEM",
    "DEPTH_LAYERS",
    "OBLIQUE_DISPLAY_SHEAR",
    "TARGET_SKELETON_HEIGHT",
    "CoordinateFrame",
    "FrameSpec",
    "AnatomicalCheckResult",
    "FRAME_REGISTRY",
    "axis_labels_canonical",
    "compact_axis_legend",
    "vertical_coordinate",
    "depth_from_mediapipe_landmark",
    "mediapipe_to_canonical",
    "mediapipe_to_canonical_oblique_xy",
    "mediapipe_to_opensim_trc_m",
    "mediapipe_to_opensim_trc_mm",
    "estimate_mediapipe_to_meter_scale",
    "canonical_to_visualization_oblique",
    "canonical_to_visualization_frontal",
    "check_anatomical_ordering",
    "audit_recording_anatomy",
    "debug_canonical_joint_positions",
    "coordinate_system_map_markdown",
]
