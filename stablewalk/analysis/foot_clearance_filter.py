"""
Robust foot-clearance filtering for StableWalk gait analysis.

Coordinates are canonical StableWalk body-normalized meters (+Y vertical,
hip-centered). Display centimeters use ``clearance_m * 100`` only when
``scale_mode == "body_normalized"``.

Foot clearance definition:
    foot_clearance_m = min(heel_clearance_m, toe_clearance_m)
where each landmark clearance is ``max(0, landmark_Y - floor_y)``.
Ankle height is never used as foot clearance.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Literal

from stablewalk.analysis.ground_reference import (
    FOOT_MEASUREMENT_JOINTS,
    NEAR_GROUND_THRESHOLD_M,
    ON_GROUND_THRESHOLD_M,
    GroundReferencePlane,
    foot_clearance_m,
    vertical_coordinate,
)
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3

COORDINATE_UNITS = (
    "canonical StableWalk body-normalized meters (+Y vertical, hip-centered)"
)

MIN_LANDMARK_VISIBILITY = 0.35
MAX_PLAUSIBLE_CLEARANCE_M = 0.25
MAX_FRAME_CLEARANCE_CHANGE_M = 0.12
MAD_OUTLIER_K = 3.5
MIN_MAD_FLOOR_M = 0.008
TEMPORAL_WINDOW = 5

RejectReason = Literal[
    "",
    "low_heel_toe_confidence",
    "heel_toe_above_ankle",
    "missing_landmarks",
    "clearance_exceeds_plausible_max",
    "temporal_outlier",
    "frame_to_frame_jump",
    "floor_estimate_invalid",
]


@dataclass(frozen=True)
class FootLandmarkSample:
    heel_y_m: float | None
    toe_y_m: float | None
    heel_clearance_m: float | None
    toe_clearance_m: float | None
    heel_visibility: float
    toe_visibility: float
    raw_clearance_m: float | None


@dataclass(frozen=True)
class FootClearanceFilteredSample:
    frame_index: int
    timestamp_s: float
    vertical_axis: str
    floor_y_m: float | None
    landmark: FootLandmarkSample
    raw_clearance_m: float | None
    filtered_clearance_m: float | None
    phase: str
    display_state: str
    is_valid: bool
    reject_reason: RejectReason


@dataclass(frozen=True)
class FootClearanceSwingStats:
    current_m: float | None
    current_valid: bool
    current_reject_reason: RejectReason
    max_swing_m: float | None
    avg_swing_m: float | None
    median_swing_m: float | None
    valid_swing_count: int
    total_swing_count: int
    rejected_count: int
    rejected_pct: float


@dataclass
class FootClearanceSeriesResult:
    side: str
    samples: list[FootClearanceFilteredSample] = field(default_factory=list)
    swing_stats: FootClearanceSwingStats | None = None


def _landmark_visibility(snapshot: SkeletonSnapshot, joint_id: str) -> float:
    vis_map = snapshot.metadata.get("landmark_visibility")
    if isinstance(vis_map, dict) and joint_id in vis_map:
        return float(vis_map[joint_id])
    if snapshot.joints.get(joint_id) is not None:
        return 1.0
    return 0.0


def _heel_toe_landmark_sample(
    snapshot: SkeletonSnapshot,
    plane: GroundReferencePlane,
    side: str,
) -> FootLandmarkSample:
    axis = plane.vertical_axis
    heel_id = f"{side}_heel"
    toe_id = f"{side}_toe"
    heel_sample = snapshot.joints.get(heel_id)
    toe_sample = snapshot.joints.get(toe_id)
    ankle_sample = snapshot.joints.get(f"{side}_ankle")
    heel_vis = _landmark_visibility(snapshot, heel_id)
    toe_vis = _landmark_visibility(snapshot, toe_id)

    heel_y = (
        vertical_coordinate(heel_sample.position, axis=axis) if heel_sample else None
    )
    toe_y = vertical_coordinate(toe_sample.position, axis=axis) if toe_sample else None
    ankle_y = (
        vertical_coordinate(ankle_sample.position, axis=axis)
        if ankle_sample is not None
        else None
    )

    def _single_clearance(
        sample_pos: Vec3 | None, joint_id: str
    ) -> float | None:
        if sample_pos is None:
            return None
        y = vertical_coordinate(sample_pos, axis=axis)
        if (
            ankle_y is not None
            and joint_id.endswith(("_heel", "_toe"))
            and y > ankle_y + 0.015
        ):
            return None
        return foot_clearance_m(sample_pos, plane, axis=axis)

    heel_c = _single_clearance(heel_sample.position if heel_sample else None, heel_id)
    toe_c = _single_clearance(toe_sample.position if toe_sample else None, toe_id)

    raw: float | None = None
    candidates = [c for c in (heel_c, toe_c) if c is not None]
    if candidates:
        raw = min(candidates)

    return FootLandmarkSample(
        heel_y_m=heel_y,
        toe_y_m=toe_y,
        heel_clearance_m=heel_c,
        toe_clearance_m=toe_c,
        heel_visibility=heel_vis,
        toe_visibility=toe_vis,
        raw_clearance_m=raw,
    )


def _phase_from_clearance(
    clearance_m: float | None,
    *,
    prev_phase: str | None = None,
) -> str:
    if clearance_m is None:
        return prev_phase or "unknown"
    if clearance_m < ON_GROUND_THRESHOLD_M:
        return "stance"
    if clearance_m >= NEAR_GROUND_THRESHOLD_M:
        return "swing"
    if prev_phase in ("stance", "swing"):
        return prev_phase
    mid = (ON_GROUND_THRESHOLD_M + NEAR_GROUND_THRESHOLD_M) / 2
    return "stance" if clearance_m < mid else "swing"


def _display_state(phase: str, *, valid: bool) -> str:
    if not valid:
        return "UNKNOWN"
    if phase == "swing":
        return "SWING"
    if phase == "stance":
        return "CONTACT"
    return "UNKNOWN"


def _raw_reject_reason(
    landmark: FootLandmarkSample,
    *,
    plane: GroundReferencePlane | None,
) -> RejectReason:
    if plane is None:
        return "floor_estimate_invalid"
    if landmark.heel_y_m is None and landmark.toe_y_m is None:
        return "missing_landmarks"
    heel_ok = (
        landmark.heel_y_m is not None
        and landmark.heel_visibility >= MIN_LANDMARK_VISIBILITY
    )
    toe_ok = (
        landmark.toe_y_m is not None
        and landmark.toe_visibility >= MIN_LANDMARK_VISIBILITY
    )
    if not heel_ok and not toe_ok:
        return "low_heel_toe_confidence"
    if landmark.heel_clearance_m is None and landmark.toe_clearance_m is None:
        if landmark.heel_y_m is not None or landmark.toe_y_m is not None:
            return "heel_toe_above_ankle"
        return "missing_landmarks"
    if landmark.raw_clearance_m is not None and landmark.raw_clearance_m > MAX_PLAUSIBLE_CLEARANCE_M:
        return "clearance_exceeds_plausible_max"
    return ""


def _temporal_outlier_indices(
    raw_values: list[float | None],
) -> set[int]:
    """Mark indices that deviate from a rolling median by k×MAD or jump too fast."""
    outliers: set[int] = set()
    valid_pairs = [(i, v) for i, v in enumerate(raw_values) if v is not None]
    if len(valid_pairs) < 3:
        return outliers

    values = [v for _, v in valid_pairs]
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values)
    limit = max(MAD_OUTLIER_K * mad, MIN_MAD_FLOOR_M, 0.04)

    for i, v in valid_pairs:
        if abs(v - med) > limit:
            outliers.add(i)

    for j in range(1, len(raw_values)):
        a, b = raw_values[j - 1], raw_values[j]
        if a is not None and b is not None and abs(b - a) > MAX_FRAME_CLEARANCE_CHANGE_M:
            outliers.add(j)
            outliers.add(j - 1)

    return outliers


def build_filtered_foot_clearance_series(
    recording: GaitMotionRecording,
    plane: GroundReferencePlane | None,
    side: str,
    *,
    end_frame_float: float | None = None,
) -> FootClearanceSeriesResult:
    """Build per-frame raw + filtered clearance with swing statistics."""
    if recording.frame_count <= 0:
        return FootClearanceSeriesResult(side=side)

    last_index = recording.frame_count - 1
    if end_frame_float is not None:
        last_index = min(last_index, max(0, int(end_frame_float)))

    raw_samples: list[FootClearanceFilteredSample] = []
    raw_values: list[float | None] = []
    prev_phase: str | None = None

    for index in range(last_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            raw_values.append(None)
            continue

        landmark = (
            _heel_toe_landmark_sample(snap, plane, side)
            if plane is not None
            else FootLandmarkSample(None, None, None, None, 0.0, 0.0, None)
        )
        reason = _raw_reject_reason(landmark, plane=plane)
        raw_values.append(landmark.raw_clearance_m)

        phase = _phase_from_clearance(
            landmark.raw_clearance_m if not reason else None,
            prev_phase=prev_phase,
        )
        if reason:
            phase = prev_phase or "unknown"
        prev_phase = phase if phase != "unknown" else prev_phase

        raw_samples.append(
            FootClearanceFilteredSample(
                frame_index=index,
                timestamp_s=float(snap.time_s),
                vertical_axis=plane.vertical_axis if plane else "y",
                floor_y_m=plane.floor_y if plane else None,
                landmark=landmark,
                raw_clearance_m=landmark.raw_clearance_m,
                filtered_clearance_m=None,
                phase=phase,
                display_state="UNKNOWN",
                is_valid=False,
                reject_reason=reason,
            )
        )

    temporal_outliers = _temporal_outlier_indices(raw_values)

    filtered_samples: list[FootClearanceFilteredSample] = []
    for i, sample in enumerate(raw_samples):
        reason = sample.reject_reason
        filtered = sample.raw_clearance_m

        if not reason and i in temporal_outliers:
            reason = (
                "frame_to_frame_jump"
                if i > 0
                and sample.raw_clearance_m is not None
                and raw_samples[i - 1].raw_clearance_m is not None
                and abs(sample.raw_clearance_m - raw_samples[i - 1].raw_clearance_m)
                > MAX_FRAME_CLEARANCE_CHANGE_M
                else "temporal_outlier"
            )
            filtered = None

        is_valid = not reason and filtered is not None
        phase = sample.phase
        if not is_valid and phase == "swing":
            phase = "unknown"

        filtered_samples.append(
            FootClearanceFilteredSample(
                frame_index=sample.frame_index,
                timestamp_s=sample.timestamp_s,
                vertical_axis=sample.vertical_axis,
                floor_y_m=sample.floor_y_m,
                landmark=sample.landmark,
                raw_clearance_m=sample.raw_clearance_m,
                filtered_clearance_m=filtered if is_valid else None,
                phase=phase,
                display_state=_display_state(phase, valid=is_valid),
                is_valid=is_valid,
                reject_reason=reason if not is_valid else "",
            )
        )

    swing_valid = [
        s.filtered_clearance_m
        for s in filtered_samples
        if s.phase == "swing" and s.is_valid and s.filtered_clearance_m is not None
    ]
    swing_all = [s for s in filtered_samples if s.phase == "swing"]
    rejected = sum(1 for s in filtered_samples if not s.is_valid)
    total = len(filtered_samples)

    current_sample = filtered_samples[-1] if filtered_samples else None
    current_m = None
    current_valid = False
    current_reason: RejectReason = ""
    if current_sample is not None:
        current_valid = current_sample.is_valid
        current_m = current_sample.filtered_clearance_m
        current_reason = current_sample.reject_reason

    stats = FootClearanceSwingStats(
        current_m=current_m,
        current_valid=current_valid,
        current_reject_reason=current_reason,
        max_swing_m=max(swing_valid) if swing_valid else None,
        avg_swing_m=(sum(swing_valid) / len(swing_valid)) if swing_valid else None,
        median_swing_m=(
            float(statistics.median(swing_valid)) if swing_valid else None
        ),
        valid_swing_count=len(swing_valid),
        total_swing_count=len(swing_all),
        rejected_count=rejected,
        rejected_pct=(100.0 * rejected / total) if total else 0.0,
    )

    return FootClearanceSeriesResult(
        side=side,
        samples=filtered_samples,
        swing_stats=stats,
    )


def reject_reason_display(reason: RejectReason) -> str:
    mapping = {
        "low_heel_toe_confidence": "low heel/toe confidence",
        "heel_toe_above_ankle": "heel/toe above ankle (pose artifact)",
        "missing_landmarks": "missing heel/toe landmarks",
        "clearance_exceeds_plausible_max": "sample rejected as outlier (exceeds plausible max)",
        "temporal_outlier": "current foot sample rejected as outlier",
        "frame_to_frame_jump": "current foot sample rejected as outlier",
        "floor_estimate_invalid": "floor estimate invalid",
    }
    return mapping.get(reason, reason or "")


def unavailable_display_reason(reason: RejectReason) -> str:
    text = reject_reason_display(reason)
    if not text:
        return "Foot clearance data unavailable"
    return text
