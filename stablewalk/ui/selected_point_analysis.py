"""
Joint-specific biomechanical readouts for the Selected Point Analysis panel.

Uses the project's hip-centered 3D coordinate system (+Y vertical, meters).
Does not alter gait scoring or stability calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from stablewalk.analysis.ground_reference import (
    CALIBRATION_CHECK_LABEL,
    FOOT_POINT_IDS,
    GroundReferencePlane,
    FootClearanceReading,
    FootClearanceSessionStats,
    clearance_sanity_flag,
    compute_foot_clearance_reading,
    compute_session_foot_clearance_stats,
    estimate_ground_plane,
    format_clearance_cm,
    format_supporting_clearance_cm,
    ground_distance_m,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.ui.dof_position_table import kinematic_sample_for_item
from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item

# Re-export for callers that already import FOOT_ITEM_IDS from this module.
FOOT_ITEM_IDS = FOOT_POINT_IDS

KNEE_ITEM_IDS: frozenset[str] = frozenset({"left_knee", "right_knee"})
HIP_ITEM_IDS: frozenset[str] = frozenset({"left_hip", "right_hip"})

AnalysisMetric = tuple[str, str]
AnalysisMode = Literal["general", "foot"]


def is_foot_analysis_point(item_id: str | None) -> bool:
    """True when the point uses Foot Analysis mode (clearance / ground distance)."""
    return bool(item_id and item_id in FOOT_ITEM_IDS)


def analysis_mode_for_item(item_id: str | None) -> AnalysisMode:
    return "foot" if is_foot_analysis_point(item_id) else "general"


@dataclass(frozen=True)
class SelectedPointAnalysis:
    """Formatted universal + derived metrics for the analysis panel grid."""

    metrics: tuple[AnalysisMetric, ...]


@dataclass(frozen=True)
class FootAnalysisCardMetrics:
    """Formatted readouts for the dedicated Foot Analysis summary card."""

    selected_point: str
    time_s: str
    frame: str
    foot_clearance_cm: str
    contact_status: str
    min_clearance_cm: str
    max_clearance_cm: str
    avg_clearance_cm: str
    current_speed_mps: str
    position_display: str
    clearance_value_cm: str = "—"
    clearance_quality: str = "—"
    ground_note: str = ""


@dataclass(frozen=True)
class AnalysisPanelMetrics:
    """Compact metrics rows for the Selected Point Analysis header."""

    identity: tuple[AnalysisMetric, ...]
    kinematics: tuple[AnalysisMetric, ...]
    derived: tuple[AnalysisMetric, ...] = ()
    foot_card: FootAnalysisCardMetrics | None = None
    mode: AnalysisMode = "general"


def _dash(value: float | None, *, fmt: str = ".3f", suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value:{fmt}}{suffix}"


def _vertical_component(position: Vec3) -> float:
    """Height along +Y (meters); used for non-foot vertical range metrics."""
    return position.y


def _joint_positions_in_segment(
    recording: GaitMotionRecording,
    joint_id: str,
    end_frame_float: float,
) -> list[Vec3]:
    if recording.frame_count <= 0:
        return []
    last_index = int(max(0, min(end_frame_float, recording.frame_count - 1)))
    positions: list[Vec3] = []
    for index in range(last_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        sample = snap.joints.get(joint_id)
        if sample is None:
            continue
        positions.append(sample.position)
    return positions


def _path_length(positions: list[Vec3]) -> float | None:
    if len(positions) < 2:
        return 0.0 if positions else None
    total = 0.0
    for prev, curr in zip(positions, positions[1:], strict=False):
        dx = curr.x - prev.x
        dy = curr.y - prev.y
        dz = curr.z - prev.z
        total += math.sqrt(dx * dx + dy * dy + dz * dz)
    return total


def _axis_range(positions: list[Vec3], axis: str) -> float | None:
    if not positions:
        return None
    vals = [getattr(p, axis) for p in positions]
    return max(vals) - min(vals)


def _foot_metrics(
    sample,
    positions: list[Vec3],
    plane: GroundReferencePlane | None,
) -> list[AnalysisMetric]:
    """
    Foot / ankle / toe / heel metrics.

    Ground Distance (m) = vertical coordinate minus estimated floor_y (+Y up).
    Foot Clearance (m) = max(0, ground distance). Session min/max/avg use the
    same ground plane and axis as the current frame (see ground_reference).
    """
    speed_label, speed = _speed_metric(sample)
    metrics: list[AnalysisMetric] = [
        ("Joint", sample.joint_name),
    ]
    metrics.extend(_position_block(sample))
    metrics.append((speed_label, speed))

    stats = compute_session_foot_clearance_stats(positions, plane)
    if stats is None:
        metrics.extend(
            [
                ("Ground Distance (m)", "—"),
                ("Foot Clearance (m)", "—"),
                ("Min Ground Distance (m)", "—"),
                ("Max Ground Distance (m)", "—"),
                ("Average Ground Distance (m)", "—"),
                ("Contact Status", "—"),
            ]
        )
        return metrics

    reading = stats.current
    metrics.extend(
        [
            ("Ground Distance (m)", _dash(reading.ground_distance_m)),
            ("Foot Clearance (m)", _dash(reading.foot_clearance_m)),
            ("Min Ground Distance (m)", _dash(stats.min_clearance_m)),
            ("Max Ground Distance (m)", _dash(stats.max_clearance_m)),
            ("Average Ground Distance (m)", _dash(stats.avg_clearance_m)),
            ("Contact Status", reading.contact_state),
        ]
    )
    return metrics


def _format_delta_angle(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}°"


def _speed_metric(sample) -> tuple[str, str]:
    if sample.velocity is not None:
        return "Current Speed (m/s)", f"{sample.velocity:.3f} m/s"
    if sample.velocity_deg_s is not None:
        return "Current Speed (°/s)", f"{sample.velocity_deg_s:.3f} °/s"
    return "Current Speed (m/s)", "—"


def _position_block(sample) -> list[AnalysisMetric]:
    return [
        ("X (m)", _dash(sample.x)),
        ("Y (m)", _dash(sample.y)),
        ("Z (m)", _dash(sample.z)),
    ]


def _knee_metrics(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    next_snapshot: SkeletonSnapshot | None,
    positions: list[Vec3],
) -> list[AnalysisMetric]:
    """Knee — emphasize flexion angle and frame-to-frame change."""
    sample = kinematic_sample_for_item(
        item_id,
        snapshot,
        next_snapshot=next_snapshot,
    )
    speed_label, speed = _speed_metric(sample)
    metrics: list[AnalysisMetric] = [
        ("Current Angle (°)", _dash(sample.angle_deg, fmt=".1f")),
        ("Delta Angle (°)", _format_delta_angle(sample.delta_angle_deg)),
        ("Joint", sample.joint_name),
    ]
    metrics.extend(_position_block(sample))
    metrics.append((speed_label, speed))
    path_len = _path_length(positions)
    if path_len is not None:
        metrics.append(("Trajectory Length (m)", _dash(path_len)))
    y_range = _axis_range(positions, "y")
    if y_range is not None:
        metrics.append(("Vertical Range (m)", _dash(y_range)))
    return metrics


def _hip_metrics(
    sample,
    positions: list[Vec3],
) -> list[AnalysisMetric]:
    """Hip — emphasize position, speed, vertical height, and trajectory span."""
    speed_label, speed = _speed_metric(sample)
    metrics: list[AnalysisMetric] = [
        ("Joint", sample.joint_name),
    ]
    metrics.extend(_position_block(sample))
    metrics.extend(
        [
            (speed_label, speed),
            ("Vertical Position (m)", _dash(sample.y)),
        ]
    )
    path_len = _path_length(positions)
    if path_len is not None:
        metrics.append(("Trajectory Length (m)", _dash(path_len)))
    for axis, label in (("x", "Range X (m)"), ("y", "Range Y (m)"), ("z", "Range Z (m)")):
        span = _axis_range(positions, axis)
        if span is not None and span > 1e-6:
            metrics.append((label, _dash(span)))
    return metrics


def _generic_metrics(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    next_snapshot: SkeletonSnapshot | None,
    positions: list[Vec3],
) -> list[AnalysisMetric]:
    sample = kinematic_sample_for_item(
        item_id,
        snapshot,
        next_snapshot=next_snapshot,
    )
    speed_label, speed = _speed_metric(sample)
    metrics: list[AnalysisMetric] = [
        ("Joint", sample.joint_name),
    ]
    metrics.extend(_position_block(sample))
    metrics.append((speed_label, speed))
    metrics.append(("Vertical Position (m)", _dash(sample.y)))
    if sample.angle_deg is not None:
        metrics.append(("Current Angle (°)", _dash(sample.angle_deg, fmt=".1f")))
    if sample.delta_angle_deg is not None:
        metrics.append(("Delta Angle (°)", _format_delta_angle(sample.delta_angle_deg)))
    path_len = _path_length(positions)
    if path_len is not None:
        metrics.append(("Trajectory Length (m)", _dash(path_len)))
    return metrics


def build_selected_point_analysis(
    item_id: str,
    snapshot: SkeletonSnapshot,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
    *,
    next_snapshot: SkeletonSnapshot | None = None,
) -> SelectedPointAnalysis:
    """Build context-aware formatted metrics for the analysis panel grid."""
    sample = kinematic_sample_for_item(
        item_id,
        snapshot,
        next_snapshot=next_snapshot,
    )

    anchor = anchor_joint_for_item(item_id)
    positions: list[Vec3] = []
    plane: GroundReferencePlane | None = None
    if recording is not None and anchor:
        positions = _joint_positions_in_segment(recording, anchor, end_frame_float)
        if item_id in FOOT_ITEM_IDS:
            plane = estimate_ground_plane(recording, end_frame_float)

    if item_id in FOOT_ITEM_IDS:
        metrics = _foot_metrics(sample, positions, plane)
    elif item_id in KNEE_ITEM_IDS:
        metrics = _knee_metrics(
            item_id,
            snapshot,
            next_snapshot=next_snapshot,
            positions=positions,
        )
    elif item_id in HIP_ITEM_IDS:
        metrics = _hip_metrics(sample, positions)
    else:
        metrics = _generic_metrics(
            item_id,
            snapshot,
            next_snapshot=next_snapshot,
            positions=positions,
        )

    return SelectedPointAnalysis(metrics=tuple(metrics))


# Monocular video has no real-world scale, so the 3D skeleton is body-normalized
# (its size in "units" varies per clip). Reporting height above the floor as a
# fraction of the subject's own body height is therefore scale-invariant and
# reliable; the centimeter value is only an estimate derived from an average
# adult stature and is labelled as such in the UI.
_ASSUMED_BODY_HEIGHT_CM: float = 170.0

# Anthropometric heights of the highest tracked upper-body joints as a fraction
# of full standing height (stature), from Drillis & Contini / Winter averages.
# IMPORTANT: in this project the joint named "head" is the MediaPipe *nose*
# landmark (see ``pose_adapter._POSE_LANDMARK_TO_JOINT``: "nose" -> "head"); it
# is NOT the crown of the head. Both "head" and "nose" therefore sit at the
# nose/eye line (~93.6% of stature), well below the true top of the head. The
# raw "top-joint-to-floor" span underestimates stature, so we divide by the
# joint's fraction to recover full body height. Without this, a point's height
# read as a fraction of the nose line, biasing every value high by ~7%.
_TOP_JOINT_STATURE_FRACTION: tuple[tuple[str, float], ...] = (
    ("crown", 1.000),  # true top of head, if a real crown joint is ever added
    ("head", 0.936),   # MediaPipe nose landmark (aliased to "head") ≈ 93.6% stature
    ("nose", 0.936),   # raw nose landmark ≈ 93.6% of standing height
)
_FALLBACK_TOP_STATURE_FRACTION: float = 0.936


def _subject_body_height(
    snapshot: SkeletonSnapshot,
    floor_y: float,
    *,
    axis: str = "y",
) -> float | None:
    """Estimated standing height (floor to crown) in body-normalized units.

    The skeleton's highest tracked joint is the nose, which is anatomically only
    ~93.6% of standing height. We take the highest available top-of-body joint,
    measure its height above the estimated floor, and divide by that joint's
    anthropometric fraction of stature to estimate the full body height. This
    keeps "height as a fraction of body height" anatomically accurate (a hip at
    ~53% of stature reads ~53%, not an inflated ~57%) while staying robust to the
    per-clip scale of the body-normalized skeleton.
    """
    best_y: float | None = None
    best_fraction: float | None = None
    for jid, fraction in _TOP_JOINT_STATURE_FRACTION:
        sample = snapshot.joints.get(jid)
        if sample is None:
            continue
        y = vertical_coordinate(sample.position, axis=axis)
        if best_y is None or y > best_y:
            best_y = y
            best_fraction = fraction
    if best_y is None:
        ys = [
            vertical_coordinate(sample.position, axis=axis)
            for sample in snapshot.joints.values()
        ]
        if not ys:
            return None
        best_y = max(ys)
        best_fraction = _FALLBACK_TOP_STATURE_FRACTION

    span = best_y - floor_y
    if span <= 1e-6:
        return None
    return span / (best_fraction or _FALLBACK_TOP_STATURE_FRACTION)


# Session-stable stature per recording. A subject's height does not change during
# a walk, so the denominator for "% of body height" should be a single robust
# value rather than a per-frame estimate that jitters with nose detection (or
# drops to a raised wrist on frames where the head is briefly missed). This
# mirrors how the ground plane is estimated once per recording.
_session_stature_cache: dict[tuple[int, int, str, str, int], float] = {}


def _session_body_height(
    recording: GaitMotionRecording,
    floor_y: float,
    *,
    axis: str = "y",
) -> float | None:
    """Robust standing-height estimate over the whole recording (normalized units).

    Estimates stature on every frame via :func:`_subject_body_height` and returns
    the median, so the value is stable across playback and resistant to per-frame
    pose noise and occasional missing head detections.
    """
    if recording.frame_count <= 0:
        return None
    key = (
        id(recording),
        recording.frame_count,
        recording.source or "",
        axis,
        int(round(floor_y * 1e6)),
    )
    cached = _session_stature_cache.get(key)
    if cached is not None:
        return cached

    statures: list[float] = []
    for index in range(recording.frame_count):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        height = _subject_body_height(snap, floor_y, axis=axis)
        if height is not None and height > 1e-6:
            statures.append(height)
    if not statures:
        return None

    import statistics

    value = statistics.median(statures)
    _session_stature_cache[key] = value
    return value


def floor_distance_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot | None,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
) -> str:
    """Height of the selected point above the estimated floor.

    Works for any leg point (hip, knee, ankle, heel, toe): it uses the same
    ground plane and vertical axis as the foot-clearance analysis, so the value
    answers "how far is this point off the floor right now?".

    Because the skeleton is body-normalized (monocular video has no metric
    scale), the height is reported as a fraction of the subject's body height —
    which is scale-invariant and anatomically meaningful — together with an
    estimated centimeter value based on an average adult stature.
    """
    primary, secondary = floor_distance_parts_for_panel(
        item_id, snapshot, recording, end_frame_float
    )
    if secondary:
        return f"{primary}  \u00b7  {secondary}"
    return primary


def _floor_distance_fraction(
    item_id: str,
    snapshot: SkeletonSnapshot | None,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
) -> tuple[str, float] | None:
    """Core height-above-floor result for the selected point.

    Returns ``(kind, value)`` where *kind* is:
      * ``"frac"``  – ``value`` is the height as a fraction of standing height
      * ``"units"`` – no body scale this frame; ``value`` is in body-normalized units
      * ``"floor"`` – the point is at/below the estimated floor (``value`` is 0.0)
    or ``None`` when there is no point/plane/data to measure.
    """
    if not item_id or snapshot is None or recording is None:
        return None
    anchor = anchor_joint_for_item(item_id)
    if not anchor:
        return None
    joint = snapshot.joints.get(anchor)
    if joint is None:
        return None
    plane = estimate_ground_plane(recording, end_frame_float)
    if plane is None:
        return None
    distance = ground_distance_m(joint.position, plane)
    if distance is None:
        return None

    # Stable, session-wide body height (height is constant during a walk); fall
    # back to the current frame only if the session estimate is unavailable.
    body_height = _session_body_height(
        recording, plane.floor_y, axis=plane.vertical_axis
    )
    if body_height is None:
        body_height = _subject_body_height(
            snapshot, plane.floor_y, axis=plane.vertical_axis
        )
    if body_height is None:
        # No reliable body scale this frame — relative units, not a fake cm value.
        return ("units", distance)

    frac = distance / body_height
    if frac <= 0.0:
        return ("floor", 0.0)
    return ("frac", frac)


def floor_distance_parts_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot | None,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
) -> tuple[str, str]:
    """Two-part floor-distance readout for the analysis card.

    Returns ``(primary, secondary)``:
      * *primary*   – the reliable, scale-invariant figure (``% of body height``),
        which is anatomically meaningful regardless of the unknown video scale.
      * *secondary* – the centimeter value, explicitly flagged as an estimate
        that assumes an average ~1.70 m stature (monocular video has no metric
        scale, so the absolute centimeters depend on the subject's real height).
    """
    core = _floor_distance_fraction(item_id, snapshot, recording, end_frame_float)
    if core is None:
        return ("\u2014", "")
    kind, value = core
    if kind == "units":
        return (f"{value:.2f} body-units", "real-world scale unavailable")
    if kind == "floor":
        return ("At floor level", "")
    frac = value
    cm_est = frac * _ASSUMED_BODY_HEIGHT_CM
    primary = f"{frac * 100.0:.0f}% of body height"
    secondary = f"\u2248 {cm_est:.0f} cm (assumes 1.70 m tall)"
    return (primary, secondary)


def floor_distance_range_parts_for_panel(
    item_id: str,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
) -> tuple[str, str]:
    """Lowest\u2192highest height above the floor over the clip so far.

    Turns the single live readout into a gait insight: for a foot it is the
    step-clearance range, for the hip it is the vertical bob. Returns
    ``(primary, secondary)``:
      * *primary*   – ``"Range: 51\u201354% of body height"`` (reliable, scale-free),
      * *secondary* – ``"\u2248 87\u201392 cm"`` (estimate, same ~1.70 m assumption).
    Both are empty strings when there is not enough motion/data to be useful.
    """
    if not item_id or recording is None:
        return ("", "")
    anchor = anchor_joint_for_item(item_id)
    if not anchor:
        return ("", "")
    plane = estimate_ground_plane(recording, end_frame_float)
    if plane is None:
        return ("", "")
    positions = _joint_positions_in_segment(recording, anchor, end_frame_float)
    if len(positions) < 2:
        return ("", "")
    body_height = _session_body_height(
        recording, plane.floor_y, axis=plane.vertical_axis
    )
    if body_height is None or body_height <= 1e-6:
        return ("", "")

    fractions: list[float] = []
    for pos in positions:
        distance = ground_distance_m(pos, plane)
        if distance is None:
            continue
        fractions.append(max(0.0, distance) / body_height)
    if len(fractions) < 2:
        return ("", "")

    lo = min(fractions)
    hi = max(fractions)
    # Only worth showing when the point actually changed height appreciably
    # (>= ~0.5% of body height); otherwise a range adds noise, not insight.
    if (hi - lo) < 0.005:
        return ("", "")

    primary = f"Range: {lo * 100.0:.0f}\u2013{hi * 100.0:.0f}% of body height"
    lo_cm = lo * _ASSUMED_BODY_HEIGHT_CM
    hi_cm = hi * _ASSUMED_BODY_HEIGHT_CM
    secondary = f"\u2248 {lo_cm:.0f}\u2013{hi_cm:.0f} cm"
    return (primary, secondary)


def analysis_point_header(
    item_id: str,
    *,
    multi_selected: bool = False,
) -> str:
    """Point name for the analysis panel (time/frame live on the playback bar)."""
    point = label_for_item(item_id)
    if multi_selected:
        return f"Selected: {point} (+more)"
    return f"Selected: {point}"


def analysis_status_line(
    item_id: str,
    snapshot: SkeletonSnapshot,
    *,
    multi_selected: bool = False,
) -> str:
    """Single status line for time, frame, and selected point (no duplication in grid)."""
    point = label_for_item(item_id)
    if multi_selected:
        point = f"{point} (last selected)"
    return (
        f"Time: {snapshot.time_s:.2f} s  ·  "
        f"Frame: {snapshot.frame_index + 1}  ·  "
        f"Point: {point}"
    )


def analysis_metrics_for_display(
    analysis: SelectedPointAnalysis,
) -> tuple[AnalysisMetric, ...]:
    """Return full grid metrics (legacy / export)."""
    return analysis.metrics


def _metric_lookup(analysis: SelectedPointAnalysis) -> dict[str, str]:
    return {title: value for title, value in analysis.metrics}


def _strip_speed_mps(lookup: dict[str, str]) -> str:
    """Linear speed for the always-visible Speed (m/s) cell."""
    raw = lookup.get("Current Speed (m/s)")
    if raw is None or raw == "—":
        angular = lookup.get("Current Speed (°/s)")
        if angular and angular != "—":
            if angular.endswith(" °/s"):
                return angular[:-4].strip()
            return angular
        return "—"
    if raw.endswith(" m/s"):
        return raw[:-4].strip()
    return raw


def _strip_speed_angular(lookup: dict[str, str]) -> str:
    """Angular speed (3 decimals) for knee derived metrics."""
    raw = lookup.get("Current Speed (°/s)")
    if raw is None or raw == "—":
        return "—"
    if raw.endswith(" °/s"):
        return raw[:-4].strip()
    return raw


def context_metric_for_panel(
    item_id: str,
    analysis: SelectedPointAnalysis,
) -> AnalysisMetric:
    """One context-aware metric for the compact kinematics row."""
    lookup = _metric_lookup(analysis)

    if item_id in FOOT_ITEM_IDS:
        value = lookup.get("Ground Distance (m)") or lookup.get("Foot Clearance (m)", "—")
        return ("Ground Distance (m)", value or "—")

    if item_id in KNEE_ITEM_IDS:
        angle = lookup.get("Current Angle (°)", "—")
        delta = lookup.get("Delta Angle (°)", "—")
        if angle != "—" and delta != "—":
            return ("Angle (°)", f"{angle} / {delta}")
        if angle != "—":
            return ("Angle (°)", angle)
        if delta != "—":
            return ("Delta Angle (°)", delta)
        return ("Angle (°)", "—")

    if item_id in HIP_ITEM_IDS:
        vertical = lookup.get("Vertical Position (m)", "—")
        trajectory = lookup.get("Trajectory Length (m)", "—")
        if vertical != "—" and trajectory != "—":
            return ("Vertical / Trajectory (m)", f"{vertical} / {trajectory}")
        if vertical != "—":
            return ("Vertical Position (m)", vertical)
        if trajectory != "—":
            return ("Trajectory Length (m)", trajectory)
        return ("Vertical Position (m)", "—")

    return ("Context", "—")


def derived_metrics_for_panel(
    item_id: str,
    analysis: SelectedPointAnalysis,
) -> tuple[AnalysisMetric, ...]:
    """Joint-specific measurements shown below the universal kinematics row."""
    lookup = _metric_lookup(analysis)

    if item_id in FOOT_ITEM_IDS:
        return (
            ("Ground Distance (m)", lookup.get("Ground Distance (m)", "—")),
            ("Foot Clearance (m)", lookup.get("Foot Clearance (m)", "—")),
            ("Min Ground Distance (m)", lookup.get("Min Ground Distance (m)", "—")),
            ("Max Ground Distance (m)", lookup.get("Max Ground Distance (m)", "—")),
            ("Avg Ground Distance (m)", lookup.get("Average Ground Distance (m)", "—")),
            ("Contact Status", lookup.get("Contact Status", "—")),
        )

    if item_id in KNEE_ITEM_IDS:
        return (
            ("Angle (°)", lookup.get("Current Angle (°)", "—")),
            ("Delta Angle (°)", lookup.get("Delta Angle (°)", "—")),
            ("Speed (m/s)", _strip_speed_mps(lookup)),
        )

    if item_id in HIP_ITEM_IDS:
        return (
            (
                "Vertical Position (m)",
                lookup.get("Vertical Position (m)", lookup.get("Y (m)", "—")),
            ),
            ("Trajectory Length (m)", lookup.get("Trajectory Length (m)", "—")),
            ("Range X (m)", lookup.get("Range X (m)", "—")),
            ("Range Y (m)", lookup.get("Range Y (m)", "—")),
            ("Range Z (m)", lookup.get("Range Z (m)", "—")),
        )

    candidates: list[AnalysisMetric] = []
    for key, label in (
        ("Current Angle (°)", "Angle (°)"),
        ("Trajectory Length (m)", "Trajectory Length (m)"),
        ("Vertical Position (m)", "Vertical Position (m)"),
        ("Range Y (m)", "Range Y (m)"),
    ):
        value = lookup.get(key)
        if value and value != "—":
            candidates.append((label, value))
    return tuple(candidates[:4])


def _format_panel_coordinate(raw: str | None) -> str:
    """Display X/Y/Z with three decimal places."""
    if not raw or raw == "—":
        return "—"
    text = raw.replace(" m", "").strip()
    try:
        return f"{float(text):.3f}"
    except ValueError:
        return raw


def _format_panel_speed(raw: str | None) -> str:
    """Display linear speed with three decimal places."""
    if not raw or raw == "—":
        return "—"
    text = raw.replace(" m/s", "").replace(" °/s", "").strip()
    try:
        return f"{float(text):.3f}"
    except ValueError:
        return raw


def _format_panel_path_meters(raw: str | None) -> str:
    """Display path/vertical distances with three decimal places and m suffix."""
    if not raw or raw == "—":
        return "—"
    text = raw.replace(" m", "").strip()
    try:
        return f"{float(text):.3f} m"
    except ValueError:
        return raw


def panel_movement_metrics_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> tuple[AnalysisMetric, ...]:
    """General-mode motion readouts — never used for foot analysis points."""
    if is_foot_analysis_point(item_id):
        return ()
    lookup = _metric_lookup(analysis)
    by_title = dict(
        movement_summary_rows_for_panel(
            item_id,
            snapshot,
            analysis,
            recording,
            end_frame_float,
        )
    )
    vertical = lookup.get("Vertical Position (m)") or lookup.get("Y (m)", "—")
    return (
        ("Path Length", _format_panel_path_meters(by_title.get("Path length so far"))),
        ("Delta from Start", by_title.get("Change from start", "—")),
        ("Vertical Position", _format_panel_path_meters(vertical)),
    )


def panel_foot_secondary_metrics_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
    *,
    foot_card: FootAnalysisCardMetrics | None = None,
) -> tuple[AnalysisMetric, ...]:
    """Foot-mode secondary row — clearance, contact, path, vertical height."""
    lookup = _metric_lookup(analysis)
    by_title = dict(
        movement_summary_rows_for_panel(
            item_id,
            snapshot,
            analysis,
            recording,
            end_frame_float,
        )
    )
    if foot_card is not None:
        clearance = foot_card.foot_clearance_cm or (
            f"{foot_card.clearance_value_cm} cm"
            if foot_card.clearance_value_cm not in ("—", "")
            else "—"
        )
        contact = foot_card.contact_status
    else:
        clearance_raw = lookup.get("Foot Clearance (m)") or lookup.get("Ground Distance (m)")
        clearance = _meters_raw_to_cm_display(clearance_raw)
        if clearance != "—" and not clearance.endswith("cm"):
            clearance = f"{clearance} cm"
        contact = lookup.get("Contact Status", "—")

    vertical = _format_panel_path_meters(lookup.get("Y (m)"))
    return (
        ("Current distance from ground", clearance),
        ("Contact status", contact),
        ("Path since playback start", _format_panel_path_meters(by_title.get("Path length so far"))),
        ("Current vertical position", vertical),
    )


def metrics_for_analysis_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    *,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> AnalysisPanelMetrics:
    """Identity + kinematics summary row; mode-specific derived or foot card."""
    lookup = _metric_lookup(analysis)
    point = label_for_item(item_id)
    is_foot = is_foot_analysis_point(item_id)
    identity_block: tuple[AnalysisMetric, ...] = (
        ("Selected Point", point),
        ("Time (s)", f"{snapshot.time_s:.2f}"),
        ("Frame", str(snapshot.frame_index + 1)),
    )
    kinematics_block: tuple[AnalysisMetric, ...] = (
        ("X (m)", _format_panel_coordinate(lookup.get("X (m)", "—"))),
        ("Y (m)", _format_panel_coordinate(lookup.get("Y (m)", "—"))),
        ("Z (m)", _format_panel_coordinate(lookup.get("Z (m)", "—"))),
        ("Speed (m/s)", _format_panel_speed(_strip_speed_mps(lookup))),
    )

    plane = (
        estimate_ground_plane(recording, end_frame_float)
        if is_foot and recording is not None
        else None
    )
    foot_positions: list[Vec3] = []
    if is_foot and recording is not None:
        anchor = anchor_joint_for_item(item_id)
        if anchor:
            foot_positions = _joint_positions_in_segment(
                recording,
                anchor,
                end_frame_float,
            )
    foot_card = (
        foot_analysis_card_for_panel(
            item_id,
            snapshot,
            analysis,
            plane=plane,
            positions=foot_positions or None,
        )
        if is_foot
        else None
    )

    if is_foot:
        return AnalysisPanelMetrics(
            identity=identity_block,
            kinematics=kinematics_block,
            derived=panel_foot_secondary_metrics_for_panel(
                item_id,
                snapshot,
                analysis,
                recording,
                end_frame_float,
                foot_card=foot_card,
            ),
            foot_card=foot_card,
            mode="foot",
        )

    return AnalysisPanelMetrics(
        identity=identity_block,
        kinematics=kinematics_block,
        derived=panel_movement_metrics_for_panel(
            item_id,
            snapshot,
            analysis,
            recording,
            end_frame_float,
        ),
        foot_card=None,
        mode="general",
    )


def _metric_float(raw: str | None) -> float | None:
    """Parse a formatted metric cell into a float, when possible."""
    if not raw or raw == "—":
        return None
    text = raw.replace(" m/s", "").replace(" °/s", "").replace(" m", "").replace("°", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def _meters_raw_to_cm_display(
    raw: str | None,
    *,
    sanity_flag: bool = False,
    scale_mode: str = "body_normalized",
) -> str:
    """Format a meter-valued metric cell as centimeters for the UI."""
    value_m = _metric_float(raw)
    if value_m is None:
        return "—"
    from stablewalk.analysis.ground_reference import ScaleMode

    mode: ScaleMode = "body_normalized" if scale_mode == "body_normalized" else "unknown"
    return format_clearance_cm(value_m, sanity_flag=sanity_flag, scale_mode=mode)


def _format_foot_clearance_cm_display(
    value_m: float | None,
    *,
    calibration_check: bool = False,
    scale_mode: str = "body_normalized",
) -> str:
    """UI clearance in centimeters (meters × 100), with estimated / calibration labels."""
    from stablewalk.analysis.ground_reference import ScaleMode

    mode: ScaleMode = "body_normalized" if scale_mode == "body_normalized" else "unknown"
    return format_clearance_cm(
        value_m,
        calibration_check=calibration_check,
        scale_mode=mode,
    )


def _foot_card_clearance_from_stats(
    stats: FootClearanceSessionStats | None,
    *,
    plane: GroundReferencePlane | None,
) -> tuple[str, str, str, str, str, str, str]:
    """Hero value, quality, full line, contact, min/max/avg for the Foot Analysis card."""
    from stablewalk.analysis.ground_reference import format_foot_clearance_display

    if stats is None:
        return "—", "—", "—", "—", "—", "—", "—"

    scale = plane.scale_mode if plane is not None else "body_normalized"
    cal = stats.calibration_check_needed
    display = format_foot_clearance_display(
        stats.current.foot_clearance_m,
        sanity_flag=stats.current.sanity_flag,
        scale_mode=scale,
        calibration_check=cal,
    )
    min_cm = format_supporting_clearance_cm(
        stats.min_clearance_m,
        calibration_check=cal,
    )
    max_cm = format_supporting_clearance_cm(
        stats.max_clearance_m,
        calibration_check=cal,
    )
    avg_cm = format_supporting_clearance_cm(
        stats.avg_clearance_m,
        calibration_check=cal,
    )
    contact = (
        CALIBRATION_CHECK_LABEL
        if cal
        else stats.current.contact_state
    )
    return (
        display.value_cm,
        display.quality_label,
        display.full_line,
        contact,
        min_cm,
        max_cm,
        avg_cm,
    )


def foot_analysis_card_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    *,
    plane: GroundReferencePlane | None = None,
    positions: list[Vec3] | None = None,
) -> FootAnalysisCardMetrics:
    """Foot Analysis card — one main clearance value plus compact supporting metrics."""
    lookup = _metric_lookup(analysis)
    stats = (
        compute_session_foot_clearance_stats(positions, plane)
        if positions and plane is not None
        else None
    )
    if stats is None and plane is not None:
        # Fallback: parse from analysis metrics (legacy path).
        from stablewalk.analysis.ground_reference import format_foot_clearance_display

        clearance_raw = lookup.get("Foot Clearance (m)") or lookup.get("Ground Distance (m)")
        contact = lookup.get("Contact Status", "—")
        cal = contact == "Check calibration" or (
            _metric_float(clearance_raw) is not None
            and clearance_sanity_flag(_metric_float(clearance_raw))
        )
        display = format_foot_clearance_display(
            _metric_float(clearance_raw),
            calibration_check=cal,
            scale_mode=plane.scale_mode,
            sanity_flag=clearance_sanity_flag(_metric_float(clearance_raw)),
        )
        hero = display.full_line
        value_cm = display.value_cm
        quality = display.quality_label
        min_cm = format_supporting_clearance_cm(
            _metric_float(lookup.get("Min Ground Distance (m)")),
            calibration_check=cal,
        )
        max_cm = format_supporting_clearance_cm(
            _metric_float(lookup.get("Max Ground Distance (m)")),
            calibration_check=cal,
        )
        avg_cm = format_supporting_clearance_cm(
            _metric_float(lookup.get("Average Ground Distance (m)")),
            calibration_check=cal,
        )
    else:
        value_cm, quality, hero, contact, min_cm, max_cm, avg_cm = (
            _foot_card_clearance_from_stats(stats, plane=plane)
        )

    return FootAnalysisCardMetrics(
        selected_point=label_for_item(item_id),
        time_s=f"{snapshot.time_s:.2f} s",
        frame=str(snapshot.frame_index + 1),
        foot_clearance_cm=hero,
        clearance_value_cm=value_cm,
        clearance_quality=quality,
        contact_status=contact,
        min_clearance_cm=min_cm,
        max_clearance_cm=max_cm,
        avg_clearance_cm=avg_cm,
        current_speed_mps=_format_panel_speed(_strip_speed_mps(lookup)),
        position_display=_format_foot_position_display(lookup),
        ground_note=ground_reference_note_for_panel(plane),
    )


def _format_foot_position_display(lookup: dict[str, str]) -> str:
    """Single-line X, Y, Z for the foot card (meters, 3 decimals)."""
    x = lookup.get("X (m)", "—")
    y = lookup.get("Y (m)", "—")
    z = lookup.get("Z (m)", "—")
    if x == "—" or y == "—" or z == "—":
        return "—"
    return f"{x}, {y}, {z} m"


def ground_reference_note_for_panel(
    plane: GroundReferencePlane | None,
) -> str:
    """Plain-language note on how ground level and clearance are estimated."""
    from stablewalk.ui.theme import DOF_ANALYSIS_FOOT_GROUND_EXPLANATION

    if plane is None:
        return DOF_ANALYSIS_FOOT_GROUND_EXPLANATION
    return DOF_ANALYSIS_FOOT_GROUND_EXPLANATION


def foot_analysis_explanation_for_panel() -> str:
    """One-line explanation shown above the Foot Analysis card."""
    from stablewalk.ui.theme import DOF_ANALYSIS_FOOT_GROUND_EXPLANATION

    return DOF_ANALYSIS_FOOT_GROUND_EXPLANATION


def graph_caption_for_panel(item_id: str | None) -> str:
    """One-line explanation shown above the 3D trajectory graph."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_GRAPH_CAPTION_FOOT,
        DOF_ANALYSIS_GRAPH_CAPTION_GENERAL,
    )

    if item_id and item_id in FOOT_ITEM_IDS:
        return DOF_ANALYSIS_GRAPH_CAPTION_FOOT
    if item_id:
        return DOF_ANALYSIS_GRAPH_CAPTION_GENERAL
    return ""


def graph_caption_lines_for_panel(item_id: str | None) -> tuple[str, ...]:
    """Legacy multi-line captions — prefer ``graph_caption_for_panel``."""
    line = graph_caption_for_panel(item_id)
    return (line,) if line else ()


def foot_metrics_card_for_panel(analysis: SelectedPointAnalysis) -> tuple[AnalysisMetric, ...]:
    """Foot-only metric card — single clearance readout; export keeps meters."""
    lookup = _metric_lookup(analysis)
    clearance = lookup.get("Foot Clearance (m)") or lookup.get("Ground Distance (m)")
    return (
        ("Foot Clearance / Distance from Ground", _meters_raw_to_cm_display(clearance)),
        ("Min Clearance", _meters_raw_to_cm_display(lookup.get("Min Ground Distance (m)"))),
        ("Max Clearance", _meters_raw_to_cm_display(lookup.get("Max Ground Distance (m)"))),
        ("Average Clearance", _meters_raw_to_cm_display(lookup.get("Average Ground Distance (m)"))),
        ("Contact Status", lookup.get("Contact Status", "—")),
    )


def _format_coord_triple(x: float | None, y: float | None, z: float | None) -> str:
    if x is None or y is None or z is None:
        return "—"
    return f"({x:.3f}, {y:.3f}, {z:.3f}) m"


def _format_delta_component(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:+.3f} m"


def movement_summary_title_for_panel() -> str:
    """Section heading for the compact movement summary block."""
    from stablewalk.ui.theme import DOF_ANALYSIS_MOVEMENT_TITLE

    return DOF_ANALYSIS_MOVEMENT_TITLE


def movement_summary_intro_for_panel() -> str:
    """One-line intro for the movement summary block."""
    from stablewalk.ui.theme import DOF_ANALYSIS_MOVEMENT_INTRO

    return DOF_ANALYSIS_MOVEMENT_INTRO


def clarification_line_for_panel(item_id: str | None = None) -> str:
    """One-line link between the summary row and the current frame."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_CLARIFICATION,
        DOF_ANALYSIS_FOOT_GROUND_EXPLANATION,
    )

    if item_id and item_id in FOOT_ITEM_IDS:
        return DOF_ANALYSIS_FOOT_GROUND_EXPLANATION
    return DOF_ANALYSIS_CLARIFICATION


def legend_lines_for_panel() -> tuple[tuple[str, str], ...]:
    """Color legend rows for the analysis sidebar (label, accent color)."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_PANEL_LINE_BLUE,
        DOF_ANALYSIS_PANEL_LINE_GREEN,
        DOF_ANALYSIS_PANEL_LINE_RED,
        DOF_TRAJ_DOT_COLOR,
        DOF_TRAJ_PATH_COLOR,
        DOF_TRAJ_START_COLOR,
    )

    return (
        (DOF_ANALYSIS_PANEL_LINE_GREEN, DOF_TRAJ_START_COLOR),
        (DOF_ANALYSIS_PANEL_LINE_BLUE, DOF_TRAJ_PATH_COLOR),
        (DOF_ANALYSIS_PANEL_LINE_RED, DOF_TRAJ_DOT_COLOR),
    )


def axis_lines_for_panel() -> tuple[str, ...]:
    """Plain-language axis descriptions (+Y up, hip-centered frame)."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_PANEL_LINE_X,
        DOF_ANALYSIS_PANEL_LINE_Y,
        DOF_ANALYSIS_PANEL_LINE_Z,
    )

    return (
        DOF_ANALYSIS_PANEL_LINE_X,
        DOF_ANALYSIS_PANEL_LINE_Y,
        DOF_ANALYSIS_PANEL_LINE_Z,
    )


def graph_compact_summary_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> tuple[AnalysisMetric, ...]:
    """Compact numeric rows shown beside the 3D movement graph."""
    lookup = _metric_lookup(analysis)
    rows = movement_summary_rows_for_panel(
        item_id,
        snapshot,
        analysis,
        recording,
        end_frame_float,
    )
    row_map = {title: value for title, value in rows}

    speed = _format_panel_speed(_strip_speed_mps(lookup))
    speed_text = f"{speed} m/s" if speed != "—" else "—"

    metrics: list[AnalysisMetric] = [
        ("Start", row_map.get("Start position", "—")),
        ("Current", row_map.get("Current position", "—")),
        ("Path length", row_map.get("Path length so far", "—")),
        ("Speed", speed_text),
    ]

    if item_id in FOOT_ITEM_IDS:
        ground = lookup.get("Ground Distance (m)") or lookup.get("Foot Clearance (m)")
        if ground and ground != "—":
            metrics.append(("Ground distance", f"{ground} m"))
        else:
            metrics.append(("Ground distance", "—"))

    return tuple(metrics)


def graph_annotation_parts_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> tuple[str, str | None]:
    """
    One-line graph readout plus an optional foot ground line (shown prominently).
    """
    lookup = _metric_lookup(analysis)
    rows = graph_compact_summary_for_panel(
        item_id,
        snapshot,
        analysis,
        recording,
        end_frame_float,
    )
    main_parts: list[str] = []
    foot_line: str | None = None
    for title, value in rows:
        if title == "Ground distance":
            ground = lookup.get("Ground Distance (m)", "—")
            clearance = lookup.get("Foot Clearance (m)", "—")
            raw = clearance if clearance not in ("—", None) else ground
            value_m = _metric_float(raw)
            if value_m is not None:
                foot_line = f"Ground: {value_m * 100.0:.1f} cm"
            else:
                foot_line = "Ground: —"
        else:
            main_parts.append(f"{title}: {value}")
    return "   ·   ".join(main_parts), foot_line


def movement_summary_vertical_title_for_panel(item_id: str) -> str:
    """Sidebar label for the height readout (ground distance for foot points)."""
    return "Ground distance" if item_id in FOOT_ITEM_IDS else "Vertical position"


def movement_summary_rows_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> tuple[AnalysisMetric, ...]:
    """Compact label/value rows for the movement summary sidebar."""
    lookup = _metric_lookup(analysis)
    sample = kinematic_sample_for_item(item_id, snapshot)
    current = (sample.x, sample.y, sample.z)

    start: Vec3 | None = None
    anchor = anchor_joint_for_item(item_id)
    positions: list[Vec3] = []
    if recording is not None and anchor:
        positions = _joint_positions_in_segment(recording, anchor, end_frame_float)
        if positions:
            start = positions[0]

    start_text = _format_coord_triple(
        start.x if start else None,
        start.y if start else None,
        start.z if start else None,
    )
    current_text = _format_coord_triple(*current)

    path_len = _metric_float(lookup.get("Trajectory Length (m)"))
    if path_len is None and positions:
        path_len = _path_length(positions)
    path_text = f"{_dash(path_len)} m" if _dash(path_len) != "—" else "—"

    if item_id in FOOT_ITEM_IDS:
        height_raw = lookup.get("Ground Distance (m)") or lookup.get("Foot Clearance (m)")
    else:
        height_raw = lookup.get("Vertical Position (m)") or lookup.get("Y (m)")
    height_text = (
        f"{_format_panel_coordinate(height_raw)} m"
        if _format_panel_coordinate(height_raw) != "—"
        else "—"
    )

    delta_x = delta_y = delta_z = None
    if start is not None and None not in current:
        delta_x = current[0] - start.x
        delta_y = current[1] - start.y
        delta_z = current[2] - start.z

    delta_text = (
        f"ΔX {_format_delta_component(delta_x)}  ·  "
        f"ΔY {_format_delta_component(delta_y)}  ·  "
        f"ΔZ {_format_delta_component(delta_z)}"
    )

    return (
        ("Start position", start_text),
        ("Current position", current_text),
        ("Path length so far", path_text),
        (movement_summary_vertical_title_for_panel(item_id), height_text),
        ("Change from start", delta_text),
    )


def movement_summary_compact_rows_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> tuple[AnalysisMetric, ...]:
    """Three-row movement summary for the analysis sidebar."""
    full = movement_summary_rows_for_panel(
        item_id,
        snapshot,
        analysis,
        recording,
        end_frame_float,
    )
    keep = {"Current position", "Path length so far", "Ground distance", "Vertical position"}
    return tuple(row for row in full if row[0] in keep)


def movement_summary_for_panel(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float = 0.0,
) -> str:
    """Beginner-friendly numeric readout linking green start and red current markers."""
    lookup = _metric_lookup(analysis)
    sample = kinematic_sample_for_item(item_id, snapshot)
    current = (sample.x, sample.y, sample.z)

    start: Vec3 | None = None
    anchor = anchor_joint_for_item(item_id)
    positions: list[Vec3] = []
    if recording is not None and anchor:
        positions = _joint_positions_in_segment(recording, anchor, end_frame_float)
        if positions:
            start = positions[0]

    start_text = _format_coord_triple(
        start.x if start else None,
        start.y if start else None,
        start.z if start else None,
    )
    current_text = _format_coord_triple(*current)

    path_len = _metric_float(lookup.get("Trajectory Length (m)"))
    if path_len is None and positions:
        path_len = _path_length(positions)
    path_text = _dash(path_len)
    path_line = (
        f"Path length so far (total distance traveled): {path_text} m"
        if path_text != "—"
        else "Path length so far (total distance traveled): —"
    )

    if item_id in FOOT_ITEM_IDS:
        vertical_hint = "ground distance above floor"
        height_raw = lookup.get("Ground Distance (m)") or lookup.get("Foot Clearance (m)")
    else:
        vertical_hint = "height / up-down (Y)"
        height_raw = lookup.get("Vertical Position (m)") or lookup.get("Y (m)")
    height_text = _format_panel_coordinate(height_raw)
    vertical_line = (
        f"Vertical position ({vertical_hint}): {height_text} m"
        if height_text != "—"
        else f"Vertical position ({vertical_hint}): —"
    )

    delta_x = delta_y = delta_z = None
    if start is not None and None not in current:
        delta_x = current[0] - start.x
        delta_y = current[1] - start.y
        delta_z = current[2] - start.z

    lines = [
        f"- Start position (green): {start_text}",
        f"- Current position (red, matches values above): {current_text}",
        f"- {path_line}",
        f"- {vertical_line}",
        (
            "- Change from start (current minus start): "
            f"ΔX {_format_delta_component(delta_x)}  ·  "
            f"ΔY {_format_delta_component(delta_y)}  ·  "
            f"ΔZ {_format_delta_component(delta_z)}"
        ),
    ]
    return movement_summary_intro_for_panel() + "\n" + "\n".join(lines)


def panel_summary_sentence_for_panel() -> str:
    """One-line overview of what the 3D graph shows."""
    from stablewalk.ui.theme import DOF_ANALYSIS_PANEL_SUMMARY

    return DOF_ANALYSIS_PANEL_SUMMARY


def red_marker_clarification_for_panel() -> str:
    """One-line link between the summary row and the current frame."""
    return clarification_line_for_panel()


def graph_explanation_for_panel(item_id: str | None = None) -> str:
    """Inline legend + axis guide (legacy single-line helper)."""
    from stablewalk.ui.theme import DOF_ANALYSIS_LEGEND_AXES_COMPACT

    parts = [line for line, _color in legend_lines_for_panel()]
    parts.append(DOF_ANALYSIS_LEGEND_AXES_COMPACT)
    return "  ·  ".join(parts)


def panel_guide_line_for_panel(item_id: str | None = None) -> str:
    """Legacy combined guide — prefer the split sidebar layout."""
    return graph_explanation_for_panel(item_id)


def tracked_point_line_for_panel(item_id: str | None) -> str:
    """Name the body point shown in the graph."""
    return tracked_point_label_for_panel(item_id)


def interpretation_block_for_panel() -> str:
    """Compact graph guide shown below the red-marker clarification."""
    return graph_explanation_for_panel()


def metrics_graph_link_for_panel() -> str:
    """Legacy alias — red-marker clarification line."""
    return red_marker_clarification_for_panel()


def interpretation_for_analysis_panel(
    item_id: str | None,
    *,
    multi_selected: bool = False,
) -> str:
    """Graph guide shown below the red-marker clarification."""
    return graph_explanation_for_panel(item_id)


def tracked_point_label_for_panel(item_id: str | None) -> str:
    """Compact label naming the body point shown in the 3D graph."""
    if not item_id:
        return ""
    from stablewalk.ui.dof_selection import label_for_item
    from stablewalk.ui.theme import DOF_ANALYSIS_LEGEND_TRACKED_FMT

    return DOF_ANALYSIS_LEGEND_TRACKED_FMT.format(point=label_for_item(item_id))


def _primary_derived_metric(
    item_id: str,
    analysis: SelectedPointAnalysis,
) -> AnalysisMetric:
    """One context-aware derived value for the compact summary row."""
    lookup = _metric_lookup(analysis)

    if item_id in FOOT_ITEM_IDS:
        for key in ("Ground Distance (m)", "Foot Clearance (m)"):
            if key in lookup and lookup[key] != "—":
                return ("Ground Distance (m)", lookup[key])
        return ("Ground Distance (m)", "—")

    if item_id in KNEE_ITEM_IDS:
        if "Current Angle (°)" in lookup:
            return ("Angle (°)", lookup["Current Angle (°)"])
        if "Delta Angle (°)" in lookup:
            return ("Delta Angle (°)", lookup["Delta Angle (°)"])
        return ("Angle (°)", "—")

    if item_id in HIP_ITEM_IDS:
        if "Vertical Position (m)" in lookup:
            return ("Vertical (m)", lookup["Vertical Position (m)"])
        if "Trajectory Length (m)" in lookup:
            return ("Trajectory (m)", lookup["Trajectory Length (m)"])
        if "Range Y (m)" in lookup:
            return ("Range Y (m)", lookup["Range Y (m)"])
        return ("Vertical (m)", "—")

    for key in (
        "Current Angle (°)",
        "Delta Angle (°)",
        "Trajectory Length (m)",
        "Vertical Position (m)",
    ):
        if key in lookup:
            short = key.replace("Current ", "").replace(" (m)", " (m)")
            return (short, lookup[key])
    return ("Derived", "—")


def base_metrics_for_panel(
    analysis: SelectedPointAnalysis,
) -> tuple[AnalysisMetric, ...]:
    """Always-visible X, Y, Z, and speed for the compact metrics row."""
    lookup = _metric_lookup(analysis)
    speed_value = lookup.get("Current Speed (m/s)")
    if speed_value is None:
        speed_value = lookup.get("Current Speed (°/s)", "—")
        speed_title = "Speed (°/s)" if speed_value != "—" else "Speed (m/s)"
    else:
        speed_title = "Speed (m/s)"
        if speed_value.endswith(" m/s"):
            speed_value = speed_value[:-4].strip()
    return (
        ("X (m)", lookup.get("X (m)", "—")),
        ("Y (m)", lookup.get("Y (m)", "—")),
        ("Z (m)", lookup.get("Z (m)", "—")),
        (speed_title, speed_value),
    )


def joint_metrics_for_panel(
    analysis: SelectedPointAnalysis,
    item_id: str,
) -> tuple[AnalysisMetric, ...]:
    """Up to three joint-specific metrics — shown only when relevant."""
    lookup = _metric_lookup(analysis)

    if item_id in FOOT_ITEM_IDS:
        candidates: list[AnalysisMetric] = []
        for key, label in (
            ("Ground Distance (m)", "Ground Distance (m)"),
            ("Min Ground Distance (m)", "Min (m)"),
            ("Max Ground Distance (m)", "Max (m)"),
            ("Average Ground Distance (m)", "Avg (m)"),
            ("Contact Status", "Contact"),
        ):
            if key in lookup and lookup[key] != "—":
                candidates.append((label, lookup[key]))
        return tuple(candidates[:3])

    if item_id in KNEE_ITEM_IDS:
        candidates = []
        if "Current Angle (°)" in lookup:
            candidates.append(("Angle (°)", lookup["Current Angle (°)"]))
        if "Delta Angle (°)" in lookup and lookup["Delta Angle (°)"] != "—":
            candidates.append(("Delta angle (°)", lookup["Delta Angle (°)"]))
        return tuple(candidates[:2])

    if item_id in HIP_ITEM_IDS:
        candidates = []
        if "Vertical Position (m)" in lookup:
            candidates.append(("Vertical (m)", lookup["Vertical Position (m)"]))
        if "Trajectory Length (m)" in lookup:
            candidates.append(("Trajectory (m)", lookup["Trajectory Length (m)"]))
        for key, label in (("Range Y (m)", "Range Y (m)"), ("Range X (m)", "Range X (m)")):
            if key in lookup and lookup[key] != "—":
                candidates.append((label, lookup[key]))
                break
        return tuple(candidates[:3])

    candidates = []
    for key, label in (
        ("Current Angle (°)", "Angle (°)"),
        ("Trajectory Length (m)", "Trajectory (m)"),
        ("Vertical Position (m)", "Vertical (m)"),
    ):
        if key in lookup and lookup[key] != "—":
            candidates.append((label, lookup[key]))
    return tuple(candidates[:2])


def compact_metrics_for_panel(
    analysis: SelectedPointAnalysis,
    item_id: str,
) -> tuple[AnalysisMetric, ...]:
    """
    Legacy flat list: base metrics followed by joint-specific metrics.
    """
    return base_metrics_for_panel(analysis) + joint_metrics_for_panel(analysis, item_id)


def _export_cell(raw: str | None) -> str:
    """Normalize a metric value for CSV/JSON export (empty when unavailable)."""
    if not raw or raw == "—":
        return ""
    return raw.strip()


def _export_meters_numeric(raw: str | None) -> str:
    """Export a meter-valued cell as a plain decimal string."""
    text = _export_cell(raw)
    if not text:
        return ""
    try:
        return f"{float(text.replace(' m', '').strip()):.3f}"
    except ValueError:
        return text


def _export_cm_from_meters(raw: str | None) -> str:
    """Convert a meter export cell to centimeters (one decimal)."""
    text = _export_meters_numeric(raw)
    if not text:
        return ""
    try:
        return f"{float(text) * 100.0:.1f}"
    except ValueError:
        return ""


def _export_scale_method_label(plane: GroundReferencePlane | None) -> str:
    """Human-readable scaling method for export metadata."""
    if plane is None:
        return ""
    if plane.scale_mode == "body_normalized":
        return "body_height_normalized_estimate"
    return "unknown"


def _export_measurement_note(
    plane: GroundReferencePlane | None,
    reading: FootClearanceReading,
) -> str:
    """Explain when clearance values are estimated or need calibration."""
    parts: list[str] = []
    if plane is not None and plane.scale_mode == "body_normalized":
        parts.append(
            "Estimated body-scale clearance from monocular pose; "
            "centimeter values are not calibrated to real-world measure."
        )
    if reading.sanity_flag:
        parts.append("Clearance exceeds expected walking range; check calibration.")
    return " ".join(parts)


def _foot_clearance_export_context(
    item_id: str,
    snapshot: SkeletonSnapshot,
    recording: GaitMotionRecording | None,
    end_frame_float: float | None,
) -> dict[str, Any] | None:
    """
    Ground plane and foot clearance readings for export (typed values).

    Uses the same ground-reference pipeline as the Foot Analysis panel.
    """
    if item_id not in FOOT_ITEM_IDS or recording is None or end_frame_float is None:
        return None

    anchor = anchor_joint_for_item(item_id)
    if not anchor:
        return None

    positions = _joint_positions_in_segment(recording, anchor, end_frame_float)
    plane = estimate_ground_plane(recording, end_frame_float)
    if not positions or plane is None:
        return None

    current = positions[-1]
    stats = compute_session_foot_clearance_stats(positions, plane)
    if stats is None:
        return None

    reading = stats.current
    min_m = stats.min_clearance_m
    max_m = stats.max_clearance_m
    avg_m = stats.avg_clearance_m

    clearance_m = reading.foot_clearance_m
    clearance_cm = clearance_m * 100.0 if clearance_m is not None else None
    is_estimated = stats.clearance_is_estimated
    note = _export_measurement_note(plane, reading)
    if stats.calibration_check_needed:
        note = (
            f"{note} {CALIBRATION_CHECK_LABEL}."
            if note
            else f"{CALIBRATION_CHECK_LABEL}."
        )

    cm_display = None
    if stats.calibration_check_needed:
        cm_display = CALIBRATION_CHECK_LABEL
    elif clearance_cm is not None:
        cm_display = f"{clearance_cm:.1f} cm"
        if is_estimated:
            cm_display = f"{cm_display} (estimated)"

    return {
        "ground_level_reference_m": plane.floor_y,
        "vertical_distance_axis": plane.vertical_axis.upper(),
        "unit_scaling_method": _export_scale_method_label(plane),
        "clearance_is_estimated": is_estimated,
        "measurement_note": note,
        "foot_clearance_m": clearance_m,
        "foot_clearance_cm": clearance_cm,
        "foot_clearance_cm_display": cm_display,
        "ground_distance_m": reading.ground_distance_m,
        "contact_status": reading.contact_state,
        "sanity_flag": reading.sanity_flag,
        "calibration_check_needed": stats.calibration_check_needed,
        "min_foot_clearance_m": min_m,
        "max_foot_clearance_m": max_m,
        "average_foot_clearance_m": avg_m,
        "min_clearance_m": min_m,
        "max_clearance_m": max_m,
        "avg_clearance_m": avg_m,
    }


def _export_calibration_status(
    *,
    calibration_check_needed: bool = False,
    clearance_is_estimated: bool = False,
) -> str:
    """Short calibration label for CSV/JSON export."""
    if calibration_check_needed:
        return "calibration_check_needed"
    if clearance_is_estimated:
        return "estimated"
    return "ok"


def _foot_clearance_export_row_cells(ctx: dict[str, Any]) -> dict[str, str]:
    """CSV-friendly cells derived from ``_foot_clearance_export_context``."""
    def _m(key: str) -> str:
        val = ctx.get(key)
        if val is None:
            return ""
        return f"{float(val):.3f}"

    def _cm_from_m(key: str) -> str:
        val = ctx.get(key)
        if val is None:
            return ""
        return f"{float(val) * 100.0:.1f}"

    clearance_m = ctx.get("foot_clearance_m")
    ground_m = _m("ground_level_reference_m")
    avg_m = _m("avg_clearance_m")
    cal_status = _export_calibration_status(
        calibration_check_needed=bool(ctx.get("calibration_check_needed")),
        clearance_is_estimated=bool(ctx.get("clearance_is_estimated")),
    )
    return {
        "ground_level_reference": ground_m,
        "ground_level_reference_m": ground_m,
        "ground_level_m": ground_m,
        "vertical_distance_axis": str(ctx.get("vertical_distance_axis") or ""),
        "unit_scaling_method": str(ctx.get("unit_scaling_method") or ""),
        "clearance_scale_method": str(ctx.get("unit_scaling_method") or ""),
        "clearance_is_estimated": "yes" if ctx.get("clearance_is_estimated") else "no",
        "calibration_status": cal_status,
        "measurement_note": str(ctx.get("measurement_note") or ""),
        "foot_clearance_m": _m("foot_clearance_m") if clearance_m is not None else "",
        "foot_clearance_cm": _cm_from_m("foot_clearance_m") if clearance_m is not None else "",
        "distance_from_ground_m": _m("ground_distance_m"),
        "distance_from_ground_cm": _cm_from_m("ground_distance_m"),
        "contact_status": str(ctx.get("contact_status") or ""),
        "contact_state": str(ctx.get("contact_status") or ""),
        "min_clearance_m": _m("min_clearance_m"),
        "max_clearance_m": _m("max_clearance_m"),
        "avg_clearance_m": avg_m,
        "average_clearance_m": avg_m,
        "min_foot_clearance_m": _m("min_foot_clearance_m"),
        "max_foot_clearance_m": _m("max_foot_clearance_m"),
        "avg_ground_distance_m": avg_m,
        "ground_distance_m": _m("ground_distance_m"),
        "height_above_ground_m": _m("foot_clearance_m"),
    }


def analysis_export_row(
    item_id: str,
    snapshot: SkeletonSnapshot,
    analysis: SelectedPointAnalysis,
    *,
    recording: GaitMotionRecording | None = None,
    end_frame_float: float | None = None,
) -> dict[str, str]:
    """Flatten one frame of selected-point analysis for export."""
    lookup = _metric_lookup(analysis)
    speed = _format_panel_speed(_strip_speed_mps(lookup))
    path_length = _export_cell(lookup.get("Trajectory Length (m)"))
    delta_x = delta_y = delta_z = ""
    if recording is not None and end_frame_float is not None:
        anchor = anchor_joint_for_item(item_id)
        if anchor:
            positions = _joint_positions_in_segment(recording, anchor, end_frame_float)
            if positions:
                if path_length == "":
                    path_len = _path_length(positions)
                    if path_len is not None:
                        path_length = f"{path_len:.3f}"
                start = positions[0]
                sample = kinematic_sample_for_item(item_id, snapshot)
                if sample.x is not None and sample.y is not None and sample.z is not None:
                    delta_x = f"{sample.x - start.x:+.3f}"
                    delta_y = f"{sample.y - start.y:+.3f}"
                    delta_z = f"{sample.z - start.z:+.3f}"
    ground_m = lookup.get("Ground Distance (m)")
    clearance_m = lookup.get("Foot Clearance (m)")
    min_clearance_m = lookup.get("Min Ground Distance (m)")
    max_clearance_m = lookup.get("Max Ground Distance (m)")
    avg_clearance_m = lookup.get("Average Ground Distance (m)")
    contact = _export_cell(lookup.get("Contact Status"))
    distance_from_ground_m = _export_meters_numeric(ground_m)
    foot_clearance_m = _export_meters_numeric(clearance_m or ground_m)
    time_sec = f"{snapshot.time_s:.3f}"
    speed_mps = _export_cell(speed)
    vertical_position = _export_meters_numeric(
        lookup.get("Vertical Position (m)") or lookup.get("Y (m)")
    )
    row = {
        "time": time_sec,
        "selected_point": label_for_item(item_id),
        "frame": str(snapshot.frame_index + 1),
        "time_sec": time_sec,
        "x_m": _export_cell(_format_panel_coordinate(lookup.get("X (m)", "—"))),
        "y_m": _export_cell(_format_panel_coordinate(lookup.get("Y (m)", "—"))),
        "z_m": _export_cell(_format_panel_coordinate(lookup.get("Z (m)", "—"))),
        "speed_mps": speed_mps,
        "path_length_m": path_length,
        "delta_x_m": delta_x,
        "delta_y_m": delta_y,
        "delta_z_m": delta_z,
        "vertical_position_m": vertical_position,
        "ground_level_m": "",
        "average_clearance_m": "",
        "calibration_status": "",
        "distance_from_ground_m": distance_from_ground_m,
        "distance_from_ground_cm": _export_cm_from_meters(ground_m),
        "foot_clearance_m": foot_clearance_m,
        "foot_clearance_cm": _export_cm_from_meters(clearance_m or ground_m),
        "ground_level_reference": "",
        "ground_level_reference_m": "",
        "vertical_distance_axis": "",
        "unit_scaling_method": "",
        "clearance_scale_method": "",
        "clearance_is_estimated": "",
        "measurement_note": "",
        "contact_status": contact,
        "min_clearance_m": _export_meters_numeric(min_clearance_m),
        "max_clearance_m": _export_meters_numeric(max_clearance_m),
        "avg_clearance_m": _export_meters_numeric(avg_clearance_m),
        "average_clearance_m": _export_meters_numeric(avg_clearance_m),
        # Legacy / extended columns (kept for backward compatibility)
        "joint_name": _export_cell(lookup.get("Joint")),
        "time_s": f"{snapshot.time_s:.2f}",
        "speed_m_s": speed_mps,
        "ground_distance_m": distance_from_ground_m,
        "height_above_ground_m": distance_from_ground_m or foot_clearance_m,
        "min_foot_clearance_m": _export_meters_numeric(min_clearance_m),
        "max_foot_clearance_m": _export_meters_numeric(max_clearance_m),
        "avg_ground_distance_m": _export_meters_numeric(avg_clearance_m),
        "contact_state": contact,
        "angle_deg": _export_cell(lookup.get("Current Angle (°)")),
        "delta_angle_deg": _export_cell(lookup.get("Delta Angle (°)")),
        "trajectory_length_m": _export_cell(lookup.get("Trajectory Length (m)")),
        "range_x_m": _export_cell(lookup.get("Range X (m)")),
        "range_y_m": _export_cell(lookup.get("Range Y (m)")),
        "range_z_m": _export_cell(lookup.get("Range Z (m)")),
        "vertical_range_m": _export_cell(lookup.get("Vertical Range (m)")),
    }

    foot_ctx = _foot_clearance_export_context(
        item_id, snapshot, recording, end_frame_float
    )
    if foot_ctx is not None:
        row.update(_foot_clearance_export_row_cells(foot_ctx))

    return row


def collect_analysis_export_rows(
    item_id: str,
    recording: GaitMotionRecording,
) -> list[dict[str, str]]:
    """Build export rows for every frame of the selected point in ``recording``."""
    from stablewalk.io.analysis_export import project_analysis_export_row
    from stablewalk.ui.dof_position_table import snapshot_for_next_frame

    rows: list[dict[str, str]] = []
    frame_count = recording.frame_count
    for index in range(frame_count):
        snapshot = recording.snapshot_at(index)
        if snapshot is None:
            continue
        next_snapshot = snapshot_for_next_frame(recording, snapshot)
        analysis = build_selected_point_analysis(
            item_id,
            snapshot,
            recording,
            float(index),
            next_snapshot=next_snapshot,
        )
        row = analysis_export_row(
            item_id,
            snapshot,
            analysis,
            recording=recording,
            end_frame_float=float(index),
        )
        rows.append(project_analysis_export_row(row, item_id))
    return rows


def build_current_point_export_summary(
    item_id: str,
    recording: GaitMotionRecording,
    *,
    frame_index: int,
) -> dict[str, Any]:
    """Compact JSON summary for the selected point at the current playback frame."""
    from stablewalk.ui.dof_position_table import snapshot_for_next_frame

    snapshot = recording.snapshot_at(frame_index)
    if snapshot is None:
        raise ValueError(f"No snapshot at frame index {frame_index}")

    next_snapshot = snapshot_for_next_frame(recording, snapshot)
    analysis = build_selected_point_analysis(
        item_id,
        snapshot,
        recording,
        float(frame_index),
        next_snapshot=next_snapshot,
    )
    row = analysis_export_row(
        item_id,
        snapshot,
        analysis,
        recording=recording,
        end_frame_float=float(frame_index),
    )

    def _num(key: str) -> float | None:
        raw = row.get(key)
        if not raw:
            return None
        try:
            return float(str(raw).replace(" m/s", "").replace(" m", "").strip())
        except ValueError:
            return None

    foot_ctx = _foot_clearance_export_context(
        item_id, snapshot, recording, float(frame_index)
    )

    summary: dict[str, Any] = {
        "schema": "stablewalk-selected-point-summary",
        "version": "1.1",
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "item_id": item_id,
        "analysis_mode": analysis_mode_for_item(item_id),
        "selected_point": row.get("selected_point", label_for_item(item_id)),
        "current_frame": int(row.get("frame") or (frame_index + 1)),
        "current_time_sec": _num("time_sec") if row.get("time_sec") else _num("time_s"),
        "current_position": {
            "x_m": _num("x_m"),
            "y_m": _num("y_m"),
            "z_m": _num("z_m"),
        },
        "current_speed_mps": _num("speed_mps") or _num("speed_m_s"),
        "current_path_length_m": _num("path_length_m"),
        "path_length_m": _num("path_length_m"),
        "delta_from_start": {
            "x_m": _num("delta_x_m"),
            "y_m": _num("delta_y_m"),
            "z_m": _num("delta_z_m"),
        },
        "vertical_position_m": _num("vertical_position_m") or _num("y_m"),
        "vertical_axis": "Y",
        "unit_conversion_method": "meters_internal_3dp_centimeters_1dp_export",
    }

    if foot_ctx is not None:
        clearance_m = foot_ctx.get("foot_clearance_m")
        clearance_cm = foot_ctx.get("foot_clearance_cm")
        cal_status = _export_calibration_status(
            calibration_check_needed=bool(foot_ctx.get("calibration_check_needed")),
            clearance_is_estimated=bool(foot_ctx.get("clearance_is_estimated")),
        )
        summary.update(
            {
                "vertical_axis": foot_ctx.get("vertical_distance_axis") or "Y",
                "vertical_distance_axis": foot_ctx.get("vertical_distance_axis"),
                "ground_reference_method": (
                    "low_percentile_foot_landmarks_session_y_up"
                ),
                "unit_scaling_method": foot_ctx.get("unit_scaling_method"),
                "clearance_is_estimated": foot_ctx.get("clearance_is_estimated"),
                "calibration_status": cal_status,
                "measurement_note": foot_ctx.get("measurement_note"),
                "ground_level_m": foot_ctx.get("ground_level_reference_m"),
                "ground_level_reference_m": foot_ctx.get("ground_level_reference_m"),
                "foot_clearance_summary": {
                    "foot_clearance_m": clearance_m,
                    "foot_clearance_cm": clearance_cm,
                    "min_clearance_m": foot_ctx.get("min_clearance_m"),
                    "max_clearance_m": foot_ctx.get("max_clearance_m"),
                    "average_clearance_m": foot_ctx.get("average_foot_clearance_m"),
                    "contact_status": foot_ctx.get("contact_status"),
                    "calibration_status": cal_status,
                    "is_estimated": foot_ctx.get("clearance_is_estimated"),
                },
                "current_foot_clearance": {
                    "m": clearance_m,
                    "cm": clearance_cm,
                    "cm_display": foot_ctx.get("foot_clearance_cm_display"),
                    "is_estimated": foot_ctx.get("clearance_is_estimated"),
                    "sanity_flag": foot_ctx.get("sanity_flag"),
                    "note": foot_ctx.get("measurement_note"),
                },
                "min_foot_clearance_m": foot_ctx.get("min_foot_clearance_m"),
                "max_foot_clearance_m": foot_ctx.get("max_foot_clearance_m"),
                "average_foot_clearance_m": foot_ctx.get("average_foot_clearance_m"),
                "contact_status": foot_ctx.get("contact_status"),
                # Legacy flat keys (kept for older consumers)
                "distance_from_ground_m": foot_ctx.get("ground_distance_m"),
                "distance_from_ground_cm": (
                    foot_ctx.get("ground_distance_m") * 100.0
                    if foot_ctx.get("ground_distance_m") is not None
                    else None
                ),
                "foot_clearance_m": clearance_m,
                "foot_clearance_cm": clearance_cm,
                "min_clearance_m": foot_ctx.get("min_clearance_m"),
                "max_clearance_m": foot_ctx.get("max_clearance_m"),
                "average_clearance_m": foot_ctx.get("average_foot_clearance_m"),
            }
        )
    elif item_id in FOOT_ITEM_IDS:
        summary["contact_status"] = row.get("contact_status") or row.get("contact_state")

    return summary
