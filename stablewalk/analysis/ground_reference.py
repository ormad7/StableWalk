"""
Ground reference plane and foot ground-distance metrics for gait analysis.

Coordinate convention (StableWalk skeleton / hip-centered 3D):
  - +Y is the vertical (up) axis (see ``VERTICAL_AXIS``).
  - The ground is modeled as a horizontal plane at constant Y = floor_y.

Joint positions come from pose-based 3D reconstruction scaled so the
body-height span is ~1.0 unit (``pose/skeleton_3d.TARGET_SKELETON_HEIGHT``).
Exported and displayed "meters" are **estimated body-scale meters**, not
clinical tape-measure ground truth from monocular video.

Ground distance for a foot landmark:
  ground_distance_m = point_Y - floor_y

Foot clearance clamps small negative noise to zero at contact.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.models.joint_registry import ROOT_JOINT_ID

# Analysis vertical axis — all ground-distance math uses this component only.
VERTICAL_AXIS: str = "y"

# Lowest foot landmarks used for bilateral clearance (minimum vertical distance).
FOOT_MEASUREMENT_JOINTS: dict[str, tuple[str, ...]] = {
    "left": ("left_toe", "left_heel", "left_ankle"),
    "right": ("right_toe", "right_heel", "right_ankle"),
}

# Foot landmarks scanned when estimating where the floor sits in Y.
FLOOR_ESTIMATION_JOINTS: tuple[str, ...] = (
    "left_toe",
    "right_toe",
    "left_heel",
    "right_heel",
    "left_ankle",
    "right_ankle",
    "left_foot",
    "right_foot",
)

# Foot GUI items that receive ground-distance readouts.
FOOT_POINT_IDS: frozenset[str] = frozenset(
    {
        "left_ankle",
        "right_ankle",
        "left_heel",
        "right_heel",
        "left_toe",
        "right_toe",
        "left_foot",
        "right_foot",
    }
)

# Contact classification thresholds (estimated meters at body-normalized scale).
ON_GROUND_THRESHOLD_M: float = 0.030
NEAR_GROUND_THRESHOLD_M: float = 0.100

# Walking swing clearance above ~35 cm at body scale is suspicious for monocular pose.
SANITY_MAX_CLEARANCE_M: float = 0.35

# Shown in UI/export when clearance exceeds SANITY_MAX_CLEARANCE_M (not exact cm).
CALIBRATION_CHECK_LABEL: str = "Calibration check needed"

# Floor estimation: 8th percentile of stance/low foot heights (robust vs outliers).
_FLOOR_HEIGHT_PERCENTILE: float = 8.0
_MIN_FOOT_SAMPLES: int = 2
_MIN_STANCE_SAMPLES: int = 3
_FLOOR_MARGIN_FRAC: float = 0.04
_MIN_FLOOR_MARGIN_M: float = 0.008

# Feet sit roughly 45–58% of body vertical span below the pelvis (normalized skeleton).
_MIN_LEG_DROP_FRAC: float = 0.38
_MAX_LEG_DROP_FRAC: float = 0.62

ScaleMode = Literal["body_normalized", "unknown"]


@dataclass(frozen=True)
class GroundReferencePlane:
    """
    Horizontal ground plane used for foot clearance analysis.

    The plane is Y = floor_y in hip-centered coordinates (+Y up). Every
    ground-distance value is the selected foot point's vertical coordinate
    minus this floor_y, in estimated body-scale meters.
    """

    floor_y: float
    vertical_axis: str = VERTICAL_AXIS
    scale_mode: ScaleMode = "body_normalized"
    body_vertical_span: float | None = None


@dataclass(frozen=True)
class FootClearanceReading:
    """Foot clearance with optional sanity flag for UI/export."""

    ground_distance_m: float | None
    foot_clearance_m: float | None
    contact_state: str
    sanity_flag: bool = False
    display_note: str = ""


@dataclass(frozen=True)
class BilateralFootClearance:
    """Left and right foot clearance at one frame (same ground plane)."""

    left: FootClearanceReading
    right: FootClearanceReading
    left_phase: str = "—"
    right_phase: str = "—"
    measuring_joint_left: str | None = None
    measuring_joint_right: str | None = None


@dataclass(frozen=True)
class BilateralFootClearanceSample:
    """One bilateral foot clearance sample for export / time series."""

    frame_index: int
    time_s: float
    left_clearance_m: float | None
    right_clearance_m: float | None
    left_contact: str
    right_contact: str
    left_phase: str
    right_phase: str


@dataclass(frozen=True)
class FootClearanceSessionStats:
    """
    Current + session min/max/avg foot clearance for one foot trajectory.

    All distances use the same ground plane and vertical axis (+Y by default).
    Values are in estimated body-scale meters internally.
    """

    current: FootClearanceReading
    min_clearance_m: float | None
    max_clearance_m: float | None
    avg_clearance_m: float | None
    calibration_check_needed: bool = False
    clearance_is_estimated: bool = True


def vertical_coordinate(position: Vec3, *, axis: str = VERTICAL_AXIS) -> float:
    """Return height along the analysis vertical axis (default +Y, meters at body scale)."""
    if axis == "y":
        return position.y
    if axis == "z":
        return position.z
    if axis == "x":
        return position.x
    return getattr(position, axis)


def ground_distance_m(
    position: Vec3,
    plane: GroundReferencePlane | float | None,
    *,
    axis: str | None = None,
) -> float | None:
    """
    Vertical distance from a foot point to the ground reference plane (meters).

    Calculation (Y-up skeleton coordinates):
        ground_distance_m = point_Y - floor_y

    Only the vertical axis is used; horizontal motion does not affect this metric.
    """
    if plane is None:
        return None
    if isinstance(plane, GroundReferencePlane):
        floor_y = plane.floor_y
        vert = axis or plane.vertical_axis
    else:
        floor_y = plane
        vert = axis or VERTICAL_AXIS
    return vertical_coordinate(position, axis=vert) - floor_y


def foot_clearance_m(
    position: Vec3,
    plane: GroundReferencePlane | float | None,
    *,
    axis: str | None = None,
) -> float | None:
    """
    Instantaneous foot clearance above the floor plane (meters).

    Same vertical axis and ground plane as ``ground_distance_m``, clamped to >= 0.
    """
    distance = ground_distance_m(position, plane, axis=axis)
    if distance is None:
        return None
    return max(0.0, distance)


def clearance_sanity_flag(distance_m: float | None) -> bool:
    """True when clearance is unrealistically large for walking at body scale."""
    return distance_m is not None and distance_m > SANITY_MAX_CLEARANCE_M


def format_clearance_cm(
    distance_m: float | None,
    *,
    sanity_flag: bool = False,
    scale_mode: ScaleMode = "body_normalized",
    calibration_check: bool = False,
) -> str:
    """
    Format clearance for UI — centimeters with optional estimated / calibration labels.

    Unit conversion: meters × 100 → centimeters (one decimal).
    """
    if calibration_check or (sanity_flag and distance_m is not None):
        return CALIBRATION_CHECK_LABEL
    if distance_m is None:
        return "—"
    cm = distance_m * 100.0
    suffix = " estimated" if scale_mode == "body_normalized" else " relative"
    return f"{cm:.1f} cm{suffix}"


def format_supporting_clearance_cm(
    distance_m: float | None,
    *,
    calibration_check: bool = False,
) -> str:
    """Compact cm readout for min/max/avg rows (no quality suffix on each cell)."""
    if calibration_check:
        return CALIBRATION_CHECK_LABEL
    if distance_m is None:
        return "—"
    return f"{distance_m * 100.0:.1f} cm"


@dataclass(frozen=True)
class FootClearanceDisplay:
    """Split UI parts for the main Foot Clearance readout."""

    value_cm: str
    quality_label: str
    full_line: str
    calibration_check: bool = False


def format_foot_clearance_display(
    distance_m: float | None,
    *,
    sanity_flag: bool = False,
    scale_mode: ScaleMode = "body_normalized",
    calibration_check: bool = False,
) -> FootClearanceDisplay:
    """
    Hero foot clearance for the UI card — value in cm plus a plain quality label.

    Foot Clearance = vertical distance (point Y − floor Y), shown in centimeters.
    """
    if calibration_check or (sanity_flag and distance_m is not None):
        return FootClearanceDisplay(
            value_cm="—",
            quality_label=CALIBRATION_CHECK_LABEL,
            full_line=CALIBRATION_CHECK_LABEL,
            calibration_check=True,
        )
    if distance_m is None:
        return FootClearanceDisplay("—", "—", "—")
    cm = distance_m * 100.0
    value = f"{cm:.1f} cm"
    if scale_mode == "body_normalized":
        quality = "Estimated body-scale"
    else:
        quality = "Relative scale"
    return FootClearanceDisplay(
        value_cm=value,
        quality_label=quality,
        full_line=format_clearance_cm(
            distance_m,
            scale_mode=scale_mode,
            calibration_check=False,
            sanity_flag=False,
        ),
    )


def foot_contact_state(ground_distance: float | None) -> str:
    """
    Classify foot contact from ground distance (meters).

    - On Ground:   ground_distance < ON_GROUND_THRESHOLD_M
    - Near Ground: ON_GROUND_THRESHOLD_M <= ground_distance < NEAR_GROUND_THRESHOLD_M
    - In Air:      ground_distance >= NEAR_GROUND_THRESHOLD_M
    """
    if ground_distance is None:
        return "—"
    if ground_distance < ON_GROUND_THRESHOLD_M:
        return "On Ground"
    if ground_distance < NEAR_GROUND_THRESHOLD_M:
        return "Near Ground"
    return "In Air"


def compute_foot_clearance_reading(
    position: Vec3 | None,
    plane: GroundReferencePlane | None,
) -> FootClearanceReading:
    """
    Instantaneous foot clearance plus contact/sanity flags for panel and export.

    Foot clearance (m) = foot vertical position − floor_y along +Y (see VERTICAL_AXIS).
    Small negative pose noise is clamped to zero for clearance (not ground distance).
    """
    if position is None or plane is None:
        return FootClearanceReading(None, None, "—")

    axis = plane.vertical_axis
    distance = ground_distance_m(position, plane, axis=axis)
    clearance = foot_clearance_m(position, plane, axis=axis)
    sanity = clearance_sanity_flag(clearance)

    contact = foot_contact_state(distance)
    if sanity:
        contact = CALIBRATION_CHECK_LABEL

    note = ""
    if sanity:
        note = "check calibration"
    elif plane.scale_mode == "body_normalized":
        note = "estimated body-scale"

    return FootClearanceReading(
        ground_distance_m=distance,
        foot_clearance_m=clearance,
        contact_state=contact,
        sanity_flag=sanity,
        display_note=note,
    )


def foot_clearances_for_positions(
    positions: list[Vec3],
    plane: GroundReferencePlane | None,
) -> list[float]:
    """
    Per-sample foot clearance (m) for a trajectory segment.

    Uses the same ground plane and vertical axis for every sample:
        clearance_m = max(0, point_Y − floor_y)
    """
    if not positions or plane is None:
        return []
    axis = plane.vertical_axis
    return [
        max(0.0, dist)
        for pos in positions
        if (dist := ground_distance_m(pos, plane, axis=axis)) is not None
    ]


def compute_session_foot_clearance_stats(
    positions: list[Vec3],
    plane: GroundReferencePlane | None,
) -> FootClearanceSessionStats | None:
    """
    Current, min, max, and average clearance for one foot trajectory segment.

    Guarantees min ≤ current ≤ max when calibration is OK (same plane, same axis).
    When any sample exceeds SANITY_MAX_CLEARANCE_M, all UI values should use
    CALIBRATION_CHECK_LABEL instead of presenting inconsistent centimeters.
    """
    if not positions or plane is None:
        return None

    clearances = foot_clearances_for_positions(positions, plane)
    if not clearances:
        return None

    current = compute_foot_clearance_reading(positions[-1], plane)
    min_m = min(clearances)
    max_m = max(clearances)
    avg_m = sum(clearances) / len(clearances)
    current_m = clearances[-1]

    calibration = (
        any(clearance_sanity_flag(c) for c in clearances)
        or current.sanity_flag
        or plane.scale_mode == "unknown"
    )

    # Keep current reading aligned with the session series (no separate clamp path).
    if current.foot_clearance_m is not None and abs(current.foot_clearance_m - current_m) > 1e-9:
        current = FootClearanceReading(
            ground_distance_m=current.ground_distance_m,
            foot_clearance_m=current_m,
            contact_state=current.contact_state,
            sanity_flag=current.sanity_flag,
            display_note=current.display_note,
        )

    # Sanity: min ≤ current ≤ max (same unit path).
    if min_m > current_m + 1e-9 or current_m > max_m + 1e-9:
        calibration = True
    if min_m > avg_m + 1e-9 or avg_m > max_m + 1e-9:
        calibration = True

    return FootClearanceSessionStats(
        current=current,
        min_clearance_m=min_m,
        max_clearance_m=max_m,
        avg_clearance_m=avg_m,
        calibration_check_needed=calibration,
        clearance_is_estimated=plane.scale_mode == "body_normalized",
    )


def ground_distances_for_positions(
    positions: list[Vec3],
    plane: GroundReferencePlane | float | None,
) -> list[float]:
    """Per-sample ground distances for a foot trajectory segment (meters, may be negative)."""
    if not positions or plane is None:
        return []
    axis = plane.vertical_axis if isinstance(plane, GroundReferencePlane) else VERTICAL_AXIS
    return [
        dist
        for pos in positions
        if (dist := ground_distance_m(pos, plane, axis=axis)) is not None
    ]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    weight = rank - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _filter_outliers(values: list[float]) -> list[float]:
    """Tukey fence outlier rejection for 1D height samples."""
    if len(values) < 4:
        return list(values)
    ordered = sorted(values)
    q1 = _percentile(ordered, 25.0)
    q3 = _percentile(ordered, 75.0)
    iqr = q3 - q1
    if iqr < 1e-9:
        return list(values)
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    kept = [v for v in values if lower <= v <= upper]
    return kept if kept else list(values)


def _joint_side(joint_id: str) -> str | None:
    if joint_id.startswith("left_"):
        return "left"
    if joint_id.startswith("right_"):
        return "right"
    return None


def _is_stance_sample(joint_id: str, snapshot: SkeletonSnapshot) -> bool:
    """Prefer foot heights from stance-phase frames when gait phase is known."""
    gait = snapshot.metadata.get("gait_phase")
    if not isinstance(gait, dict):
        return False
    side = _joint_side(joint_id)
    if not side:
        return False
    return gait.get(side) == "stance"


def _pelvis_height(snapshot: SkeletonSnapshot, *, axis: str) -> float | None:
    """Pelvis / hip reference height along the vertical axis."""
    for joint_id in (ROOT_JOINT_ID, "left_hip", "right_hip"):
        sample = snapshot.joints.get(joint_id)
        if sample is not None:
            return vertical_coordinate(sample.position, axis=axis)
    return None


def _detect_vertical_axis(recording: GaitMotionRecording, end_index: int) -> str:
    """
    Confirm +Y is vertical for this recording.

    Compares head-to-foot spread on Y vs Z; uses Z only when Y clearly is not vertical.
    """
    y_vals: list[float] = []
    z_vals: list[float] = []
    for index in range(end_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        for joint_id in ("head", "left_ankle", "right_ankle", "left_heel", "right_heel"):
            sample = snap.joints.get(joint_id)
            if sample is None:
                continue
            y_vals.append(sample.position.y)
            z_vals.append(sample.position.z)
    if len(y_vals) < 4:
        return VERTICAL_AXIS
    y_span = max(y_vals) - min(y_vals)
    z_span = max(z_vals) - min(z_vals) if z_vals else 0.0
    if z_span > y_span * 1.75 and y_span < 0.12:
        return "z"
    return VERTICAL_AXIS


def _plausible_foot_height(
    foot_y: float,
    pelvis_y: float | None,
    body_span: float,
) -> bool:
    """
    Reject pose outliers before floor estimation (+Y vertical, hip-centered).

    Feet should sit at or below the pelvis (smaller or equal Y). Reject feet
    floating above the pelvis or more than ~105% of the observed body span below it.
    """
    if pelvis_y is None:
        return True
    if foot_y > pelvis_y + 0.06:
        return False
    if foot_y < pelvis_y:
        leg_drop = pelvis_y - foot_y
        max_leg = max(body_span, 0.28) * 1.05
        if leg_drop > max_leg:
            return False
    return True


def _collect_foot_heights(
    recording: GaitMotionRecording,
    end_index: int,
    *,
    axis: str,
) -> tuple[list[float], list[float], list[float]]:
    """
    Collect foot landmark heights for floor estimation.

    Returns (all_foot_heights, stance_foot_heights, pelvis_heights).
    Samples failing ``_plausible_foot_height`` are skipped.
    """
    all_heights: list[float] = []
    stance_heights: list[float] = []
    pelvis_heights: list[float] = []

    for index in range(end_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        pelvis = _pelvis_height(snap, axis=axis)
        if pelvis is not None:
            pelvis_heights.append(pelvis)

        frame_ys = [
            vertical_coordinate(sample.position, axis=axis)
            for jid in (
                "head",
                ROOT_JOINT_ID,
                "left_hip",
                "right_hip",
                *FLOOR_ESTIMATION_JOINTS,
            )
            if (sample := snap.joints.get(jid)) is not None
        ]
        body_span = max(max(frame_ys) - min(frame_ys), 0.28) if len(frame_ys) >= 2 else 0.45

        for joint_id in FLOOR_ESTIMATION_JOINTS:
            sample = snap.joints.get(joint_id)
            if sample is None:
                continue
            height = vertical_coordinate(sample.position, axis=axis)
            if not _plausible_foot_height(height, pelvis, body_span):
                continue
            all_heights.append(height)
            if _is_stance_sample(joint_id, snap):
                stance_heights.append(height)

    return all_heights, stance_heights, pelvis_heights


# Stable ground plane per recording (full session) — avoids floor drift during playback.
_ground_plane_cache: dict[tuple[int, int, str], GroundReferencePlane] = {}


def _recording_cache_key(recording: GaitMotionRecording) -> tuple[int, int, str]:
    return (id(recording), recording.frame_count, recording.source or "")


def estimate_ground_plane(
    recording: GaitMotionRecording,
    end_frame_float: float,
) -> GroundReferencePlane | None:
    """
    Estimate the ground reference plane from stable foot landmark heights.

    Vertical axis: +Y only (X left/right, Y up/down, Z forward/back).

    Method (hip-centered coordinates):
      1. Scan heel / toe / ankle heights across the **full recording** so the
         floor stays fixed during playback (``end_frame_float`` does not move floor_y).
      2. Prefer samples from stance-phase frames when ``gait_phase`` metadata exists.
      3. Drop heights outside a plausible band below the pelvis per frame.
      4. Reject outlier heights (bad pose frames).
      5. Set floor_y to a low percentile of cleaned heights minus a small margin.
      6. Clamp floor_y so it cannot sit far below the lowest plausible leg length.

    Positions are body-height normalized (~1.0 = full body), so clearance values
    are estimated meters at that scale, not raw pixel units.

    Foot clearance (m) = foot_Y − floor_y
    """
    if recording.frame_count <= 0:
        return None

    cache_key = _recording_cache_key(recording)
    cached = _ground_plane_cache.get(cache_key)
    if cached is not None:
        return cached

    # Full session — consistent ground reference while scrubbing/playing.
    last_index = recording.frame_count - 1
    axis = VERTICAL_AXIS

    all_heights, stance_heights, pelvis_heights = _collect_foot_heights(
        recording, last_index, axis=axis
    )
    if len(all_heights) < _MIN_FOOT_SAMPLES:
        return None

    pool = stance_heights if len(stance_heights) >= _MIN_STANCE_SAMPLES else all_heights
    cleaned = _filter_outliers(pool)
    if len(cleaned) < _MIN_FOOT_SAMPLES:
        cleaned = pool

    body_span = max(all_heights) - min(all_heights)
    body_span = max(body_span, 0.28)

    # Low percentile ≈ lowest stable foot contact (more robust than global min).
    floor_anchor = _percentile(cleaned, _FLOOR_HEIGHT_PERCENTILE)
    margin = max(_MIN_FLOOR_MARGIN_M, body_span * _FLOOR_MARGIN_FRAC)
    floor_y = floor_anchor - margin

    if pelvis_heights:
        pelvis_median = statistics.median(pelvis_heights)
        foot_low = min(cleaned)
        # +Y up: feet sit below the pelvis (smaller vertical coordinate).
        leg_drop = pelvis_median - foot_low
        if leg_drop > 0.02:
            max_leg = max(leg_drop, body_span) * 1.12
            # Floor cannot be more than ~1.12× observed leg length below pelvis.
            floor_y = max(floor_y, pelvis_median - max_leg)

    # Floor at or slightly below lowest reliable foot contact (no rogue min() outlier).
    reliable_low = _percentile(cleaned, 2.0)
    floor_y = min(floor_y, reliable_low - margin * 0.25)

    scale_mode: ScaleMode = "body_normalized"
    if body_span > 2.5 or body_span < 0.08:
        scale_mode = "unknown"

    plane = GroundReferencePlane(
        floor_y=floor_y,
        vertical_axis=axis,
        scale_mode=scale_mode,
        body_vertical_span=body_span,
    )
    _ground_plane_cache[cache_key] = plane
    return plane


def floor_reference_y(
    recording: GaitMotionRecording,
    end_frame_float: float,
) -> float | None:
    """Vertical coordinate of the estimated ground plane, or None if unavailable."""
    plane = estimate_ground_plane(recording, end_frame_float)
    return plane.floor_y if plane is not None else None


def lowest_foot_landmark(
    snapshot: SkeletonSnapshot,
    side: str,
) -> tuple[Vec3 | None, str | None]:
    """
    Return the lowest foot landmark on ``side`` (``left`` / ``right``).

    Uses the minimum vertical coordinate among toe, heel, and ankle.
    """
    joints = FOOT_MEASUREMENT_JOINTS.get(side, ())
    best_pos: Vec3 | None = None
    best_joint: str | None = None
    best_y = float("inf")
    axis = VERTICAL_AXIS
    for joint_id in joints:
        sample = snapshot.joints.get(joint_id)
        if sample is None:
            continue
        y = vertical_coordinate(sample.position, axis=axis)
        if y < best_y:
            best_y = y
            best_pos = sample.position
            best_joint = joint_id
    return best_pos, best_joint


def foot_clearance_for_side(
    snapshot: SkeletonSnapshot,
    plane: GroundReferencePlane | None,
    side: str,
) -> tuple[FootClearanceReading, str | None]:
    """Instantaneous clearance for one foot using its lowest landmark."""
    pos, joint_id = lowest_foot_landmark(snapshot, side)
    return compute_foot_clearance_reading(pos, plane), joint_id


def _phase_from_clearance(
    clearance_m: float | None,
    *,
    prev_phase: str | None = None,
) -> str:
    """
    Classify stance vs swing from foot clearance with hysteresis.

    Stance when clearance is below the on-ground threshold; swing when above
    the near-ground threshold; otherwise hold the previous phase.
    """
    if clearance_m is None:
        return prev_phase or "—"
    if clearance_m < ON_GROUND_THRESHOLD_M:
        return "stance"
    if clearance_m >= NEAR_GROUND_THRESHOLD_M:
        return "swing"
    if prev_phase in ("stance", "swing"):
        return prev_phase
    return "stance" if clearance_m < (ON_GROUND_THRESHOLD_M + NEAR_GROUND_THRESHOLD_M) / 2 else "swing"


def bilateral_foot_clearance(
    snapshot: SkeletonSnapshot,
    plane: GroundReferencePlane | None,
    *,
    prev_left_phase: str | None = None,
    prev_right_phase: str | None = None,
) -> BilateralFootClearance:
    """Left and right foot clearance at one frame using the shared ground plane."""
    left_reading, left_joint = foot_clearance_for_side(snapshot, plane, "left")
    right_reading, right_joint = foot_clearance_for_side(snapshot, plane, "right")
    left_phase = _phase_from_clearance(
        left_reading.foot_clearance_m,
        prev_phase=prev_left_phase,
    )
    right_phase = _phase_from_clearance(
        right_reading.foot_clearance_m,
        prev_phase=prev_right_phase,
    )
    return BilateralFootClearance(
        left=left_reading,
        right=right_reading,
        left_phase=left_phase,
        right_phase=right_phase,
        measuring_joint_left=left_joint,
        measuring_joint_right=right_joint,
    )


def bilateral_foot_clearance_series(
    recording: GaitMotionRecording,
    end_frame_float: float,
) -> list[BilateralFootClearanceSample]:
    """
    Bilateral foot clearance for every frame through ``end_frame_float``.

    Uses a session-stable ground plane and hysteresis-based stance/swing labels.
    """
    if recording.frame_count <= 0:
        return []
    plane = estimate_ground_plane(recording, end_frame_float)
    if plane is None:
        return []

    last_index = min(
        recording.frame_count - 1,
        max(0, int(end_frame_float)),
    )
    out: list[BilateralFootClearanceSample] = []
    prev_left: str | None = None
    prev_right: str | None = None
    for index in range(last_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        bilateral = bilateral_foot_clearance(
            snap,
            plane,
            prev_left_phase=prev_left,
            prev_right_phase=prev_right,
        )
        prev_left = bilateral.left_phase
        prev_right = bilateral.right_phase
        out.append(
            BilateralFootClearanceSample(
                frame_index=index,
                time_s=float(snap.time_s),
                left_clearance_m=bilateral.left.foot_clearance_m,
                right_clearance_m=bilateral.right.foot_clearance_m,
                left_contact=bilateral.left.contact_state,
                right_contact=bilateral.right.contact_state,
                left_phase=bilateral.left_phase,
                right_phase=bilateral.right_phase,
            )
        )
    return out
