"""
Walking speed estimation for monocular pose-based gait analysis.

Pelvis positions in the hip-centered canonical frame do not accumulate horizontal
translation, so overground speed must be inferred from:

  1. Global pelvis trajectory (camera-aligned, before hip centering)
  2. Cadence × anthropometrically scaled step length
  3. Image-space pelvis drift scaled to subject stature

Hip-centered COM and ankle velocities measure segment oscillation, not overground
speed, and are excluded from the estimator.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, GaitEvent
from stablewalk.analysis.gait_feature_analysis import (
    BodySegmentDimensions,
    GaitFeatureAnalysisResult,
)
from stablewalk.analysis.motion_frames import build_motion_frame_series, project_global_trajectory
from stablewalk.coordinates.coordinate_map import DEFAULT_SUBJECT_HEIGHT_M, TARGET_SKELETON_HEIGHT
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
from stablewalk.models.pose_data import PoseSequence
from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.ui.scientific_labels import format_walking_speed_value

# Plausible adult walking speed range (m/s). Values outside are rejected.
MIN_PLAUSIBLE_SPEED_M_S = 0.25
MAX_PLAUSIBLE_SPEED_M_S = 3.5
# Minimum confidence required to report a speed to the user.
MIN_REPORT_CONFIDENCE = 0.35
# Minimum body-normalized step length to trust hip-centered foot displacement.
MIN_TRUSTED_STEP_NORM = 0.12

UNAVAILABLE_LABEL = "Not available"
UNAVAILABLE_LABEL_DETAIL = (
    "Not available (insufficient measured scale / step length)"
)


@dataclass(frozen=True)
class WalkingSpeedEstimate:
    value: float
    confidence: float
    method: str
    note: str


def is_plausible_walking_speed(speed_m_s: float | None) -> bool:
    """Return True when *speed_m_s* is within a physiologically plausible range."""
    if speed_m_s is None or not np.isfinite(speed_m_s):
        return False
    return MIN_PLAUSIBLE_SPEED_M_S <= speed_m_s <= MAX_PLAUSIBLE_SPEED_M_S


def is_reportable_walking_speed(metric: MetricWithConfidence | None) -> bool:
    """Whether a metric should be shown in the GUI / summary (not misleading)."""
    if metric is None or metric.value is None:
        return False
    if metric.confidence < MIN_REPORT_CONFIDENCE:
        return False
    return is_plausible_walking_speed(metric.value)


def format_walking_speed_display(metric: MetricWithConfidence | None) -> str:
    """Human-readable walking speed — never invents a value.

    When an estimate exists but fails confidence / plausibility gates, show
    unavailable with the measured confidence instead of a fabricated speed.
    """
    if is_reportable_walking_speed(metric):
        assert metric is not None and metric.value is not None
        return format_walking_speed_value(metric.value)
    if metric is not None and metric.confidence > 0.0:
        return f"{UNAVAILABLE_LABEL} (conf {metric.confidence:.0%})"
    return UNAVAILABLE_LABEL


def _median_speed(samples: list[float]) -> float | None:
    if not samples:
        return None
    positive = [s for s in samples if s > 1e-5]
    if not positive:
        return None
    return float(np.median(positive))


def _normalized_span_m(recording: GaitMotionRecording) -> float:
    """Median head-to-foot vertical span in body-normalized coordinates."""
    heights: list[float] = []
    for snap in recording.snapshots:
        head = snap.joints.get("head")
        la = snap.joints.get("left_ankle")
        ra = snap.joints.get("right_ankle")
        foot_y = None
        if la and ra:
            foot_y = min(la.position.y, ra.position.y)
        elif la:
            foot_y = la.position.y
        elif ra:
            foot_y = ra.position.y
        if head is not None and foot_y is not None:
            h = head.position.y - foot_y
            if h > 0.05:
                heights.append(h)
    return statistics.median(heights) if heights else TARGET_SKELETON_HEIGHT


def _meters_per_normalized_unit(
    recording: GaitMotionRecording,
    *,
    subject_height_m: float,
) -> float:
    """
    Convert body-normalized skeleton units to real-world meters.

    Skeleton height is normalized to TARGET_SKELETON_HEIGHT (~1.0). Scale by the
    configured subject stature when absolute meters are required.
    """
    span = max(_normalized_span_m(recording), 0.05)
    return subject_height_m / span


def _step_length_real_m(
    *,
    step_norm: float | None,
    normalized_step: float | None,
    leg_length_norm: float | None,
    meters_per_unit: float,
) -> float | None:
    """
    Convert a hip-centered step length to real meters.

    Prefers leg-normalized step length (step / leg) when available; falls back to
    direct body-normalized displacement when it exceeds a minimum trust threshold.
    """
    if normalized_step is not None and leg_length_norm is not None and leg_length_norm > 0.05:
        step_m = normalized_step * leg_length_norm * meters_per_unit
        if step_m > MIN_TRUSTED_STEP_NORM * meters_per_unit:
            return step_m
    if step_norm is not None and step_norm >= MIN_TRUSTED_STEP_NORM:
        return step_norm * meters_per_unit
    return None


def _foot_position(snap: SkeletonSnapshot, side: str) -> Vec3 | None:
    for jid in (f"{side}_heel", f"{side}_ankle", f"{side}_toe"):
        sample = snap.joints.get(jid)
        if sample is not None:
            return sample.position
    return None


def _step_lengths_from_foot_events(
    recording: GaitMotionRecording,
    events: list[GaitEvent],
) -> tuple[list[float], list[float]]:
    """Progression-axis foot separation at each heel strike.

    Canonical pose coordinates use X = mediolateral and Z ≈ camera-forward.
    Typical side-view (sagittal) clips progress mainly in **X**, while frontal
    clips progress mainly in **Z**. Hard-coding ``|ΔZ|`` systematically
    underestimates step length for sagittal gait.

    This returns body-normalized pose units; scale with stature before labelling
    metres.
    """
    foot_by_frame: dict[int, dict[str, Vec3]] = {}
    for snap in recording.snapshots:
        foot_by_frame[snap.frame_index] = {
            side: pos
            for side in ("left", "right")
            if (pos := _foot_position(snap, side)) is not None
        }

    # Estimate dominant horizontal progression axis from all heel-strike pairs.
    dx_samples: list[float] = []
    dz_samples: list[float] = []
    for side in ("left", "right"):
        other = "right" if side == "left" else "left"
        for event in events:
            if event.event_type != f"{side}_heel_strike":
                continue
            feet = foot_by_frame.get(event.frame_index, {})
            striking = feet.get(side)
            contralateral = feet.get(other)
            if striking is None or contralateral is None:
                continue
            dx_samples.append(abs(float(striking.x - contralateral.x)))
            dz_samples.append(abs(float(striking.z - contralateral.z)))

    use_x = True
    if dx_samples and dz_samples:
        use_x = float(np.median(dx_samples)) >= float(np.median(dz_samples))
    elif dz_samples and not dx_samples:
        use_x = False

    left_steps: list[float] = []
    right_steps: list[float] = []
    for side, out in (("left", left_steps), ("right", right_steps)):
        other = "right" if side == "left" else "left"
        hs = sorted(
            (e for e in events if e.event_type == f"{side}_heel_strike"),
            key=lambda e: e.time_s,
        )
        for event in hs:
            feet = foot_by_frame.get(event.frame_index, {})
            striking = feet.get(side)
            contralateral = feet.get(other)
            if striking is None or contralateral is None:
                continue
            value = (
                abs(float(striking.x - contralateral.x))
                if use_x
                else abs(float(striking.z - contralateral.z))
            )
            if np.isfinite(value) and value > 1e-6:
                out.append(float(value))
    return left_steps, right_steps


def _global_pelvis_speed(
    sequence: PoseSequence | None,
    recording: GaitMotionRecording,
    *,
    subject_height_m: float,
) -> WalkingSpeedEstimate | None:
    """
    Overground speed from global pelvis trajectory (before per-frame hip centering).

    Projects pelvis motion onto the estimated gait progression axis and scales
    body-normalized displacements to real meters using subject stature.
    """
    if sequence is None or not sequence.frames:
        return None

    series = build_motion_frame_series(sequence.frames, recording)
    if series.tracked_ratio < 0.35:
        return None

    times, fwd, _vert, _ml = project_global_trajectory(series)
    if times.size < 4:
        return None

    meters_per_unit = _meters_per_normalized_unit(recording, subject_height_m=subject_height_m)
    speeds: list[float] = []
    for i in range(1, times.size):
        dt = float(times[i] - times[i - 1])
        if dt <= 1e-6:
            continue
        dist_norm = abs(float(fwd[i] - fwd[i - 1]))
        speeds.append(dist_norm * meters_per_unit / dt)

    value = _median_speed(speeds)
    if value is None:
        return None

    duration = float(times[-1] - times[0])
    if duration > 0.2:
        total_fwd_m = abs(float(fwd[-1] - fwd[0])) * meters_per_unit
        avg_speed = total_fwd_m / duration
        if is_plausible_walking_speed(avg_speed):
            value = float(np.median([value, avg_speed]))

    if not is_plausible_walking_speed(value):
        return None

    prog_conf = series.progression.confidence if series.progression else 0.0
    conf = float(np.clip(0.55 + 0.35 * series.tracked_ratio + 0.25 * prog_conf, 0.0, 0.92))
    return WalkingSpeedEstimate(
        value=value,
        confidence=conf,
        method="global_pelvis_trajectory",
        note=(
            f"Global pelvis progression scaled to {subject_height_m:.2f} m stature "
            "(estimated m/s)"
        ),
    )


def _cadence_stride_speed(
    cycles: GaitCycleAnalysisResult | None,
    features: GaitFeatureAnalysisResult | None,
    recording: GaitMotionRecording,
    contact: FootContactAnalysisResult | None,
    *,
    subject_height_m: float,
) -> WalkingSpeedEstimate | None:
    cadence_spm = None
    if cycles and cycles.metrics.cadence_steps_per_min:
        cadence_spm = cycles.metrics.cadence_steps_per_min
    elif contact and contact.metrics.cadence_steps_per_min:
        cadence_spm = contact.metrics.cadence_steps_per_min

    if cadence_spm is None or cadence_spm < 40:
        return None

    meters_per_unit = _meters_per_normalized_unit(recording, subject_height_m=subject_height_m)
    leg_norm: float | None = None
    if features and features.dimensions:
        leg_norm = features.dimensions.leg_length_average

    nf = features.features if features else None
    step_m: float | None = None
    stride_m: float | None = None

    if nf:
        step_m = _step_length_real_m(
            step_norm=nf.step_length_m,
            normalized_step=nf.normalized_step_length,
            leg_length_norm=leg_norm,
            meters_per_unit=meters_per_unit,
        )
        stride_m = _step_length_real_m(
            step_norm=nf.stride_length_m,
            normalized_step=nf.normalized_stride_length,
            leg_length_norm=leg_norm,
            meters_per_unit=meters_per_unit,
        )

    if step_m is None and contact is not None and contact.events:
        left_steps, right_steps = _step_lengths_from_foot_events(recording, contact.events)
        trusted: list[float] = []
        for raw in left_steps + right_steps:
            if raw >= MIN_TRUSTED_STEP_NORM:
                trusted.append(raw * meters_per_unit)
        if trusted:
            step_m = float(statistics.mean(trusted))

    cadence_hz = cadence_spm / 60.0
    if step_m is not None and step_m > 1e-3:
        speed = cadence_hz * step_m
        if not is_plausible_walking_speed(speed):
            return None
        conf = 0.72
        if nf and nf.normalized_step_length is not None:
            conf = 0.78
        return WalkingSpeedEstimate(
            value=speed,
            confidence=conf,
            method="cadence_step_length",
            note=f"Cadence × step length scaled to {subject_height_m:.2f} m stature (estimated m/s)",
        )
    if stride_m is not None and stride_m > 1e-3:
        speed = cadence_hz * stride_m * 0.5
        if not is_plausible_walking_speed(speed):
            return None
        return WalkingSpeedEstimate(
            value=speed,
            confidence=0.70,
            method="cadence_stride_length",
            note=f"Cadence × stride length / 2 scaled to {subject_height_m:.2f} m stature (estimated m/s)",
        )
    return None


def _image_plane_speed(
    sequence: PoseSequence | None,
    recording: GaitMotionRecording,
    *,
    subject_height_m: float,
) -> WalkingSpeedEstimate | None:
    """
    Image-space pelvis drift scaled to subject stature.

    Uses horizontal (image-x) pelvis displacement as the primary progression cue
    for sagittal walking clips; falls back to combined image-plane distance when
    lateral drift dominates.
    """
    if sequence is None or not sequence.frames:
        return None

    meters_per_unit = _meters_per_normalized_unit(recording, subject_height_m=subject_height_m)
    speeds: list[float] = []
    prev_xy: tuple[float, float] | None = None
    prev_t: float | None = None

    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        kp_map = {kp.name: kp for kp in frame.keypoints}
        lh = kp_map.get("left_hip")
        rh = kp_map.get("right_hip")
        nose = kp_map.get("nose") or kp_map.get("head")
        la = kp_map.get("left_ankle")
        ra = kp_map.get("right_ankle")
        if lh is None or rh is None:
            continue
        px = (lh.x + rh.x) * 0.5
        py = (lh.y + rh.y) * 0.5
        if prev_xy is not None and prev_t is not None:
            dt = frame.timestamp_s - prev_t
            if dt > 1e-6:
                dx = abs(px - prev_xy[0])
                dy = abs(py - prev_xy[1])
                dist_norm = dx if dx >= dy * 0.5 else float(np.hypot(dx, dy))
                body_h_norm = 0.25
                if nose and la and ra:
                    foot_y = max(la.y, ra.y)
                    body_h_norm = max(nose.y - foot_y, 0.12)
                meters = (dist_norm / body_h_norm) * subject_height_m
                speeds.append(meters / dt)
        prev_xy = (px, py)
        prev_t = frame.timestamp_s

    value = _median_speed(speeds)
    if value is None or not is_plausible_walking_speed(value):
        return None
    return WalkingSpeedEstimate(
        value=value,
        confidence=0.62,
        method="image_pelvis_drift",
        note=f"Image pelvis drift scaled to {subject_height_m:.2f} m stature (estimated m/s)",
    )


def estimate_walking_speed(
    recording: GaitMotionRecording,
    *,
    sequence: PoseSequence | None = None,
    cycles: GaitCycleAnalysisResult | None = None,
    features: GaitFeatureAnalysisResult | None = None,
    contact: FootContactAnalysisResult | None = None,
    subject_height_m: float = DEFAULT_SUBJECT_HEIGHT_M,
    base_confidence: float = 0.5,
) -> MetricWithConfidence | None:
    """
    Estimate walking speed using the most reliable available monocular method.

    Returns None when no method yields a plausible speed with sufficient confidence.
    All reported values are in real meters per second, scaled from body-normalized
    pose using *subject_height_m* (default 1.70 m, configurable).
    """
    subject_height_m = max(float(subject_height_m), 1.20)

    candidates: list[WalkingSpeedEstimate] = []

    for estimator in (
        lambda: _global_pelvis_speed(sequence, recording, subject_height_m=subject_height_m),
        lambda: _cadence_stride_speed(
            cycles, features, recording, contact, subject_height_m=subject_height_m
        ),
        lambda: _image_plane_speed(sequence, recording, subject_height_m=subject_height_m),
    ):
        result = estimator()
        if result is not None and is_plausible_walking_speed(result.value):
            candidates.append(result)

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c.confidence)
    conf = max(base_confidence * 0.85, min(0.95, best.confidence))
    if conf < MIN_REPORT_CONFIDENCE:
        return None

    return MetricWithConfidence(
        value=round(best.value, 4),
        unit="m/s",
        kind="estimated",
        confidence=conf,
        note=best.note,
    )


__all__ = [
    "MIN_PLAUSIBLE_SPEED_M_S",
    "MAX_PLAUSIBLE_SPEED_M_S",
    "MIN_REPORT_CONFIDENCE",
    "UNAVAILABLE_LABEL",
    "WalkingSpeedEstimate",
    "estimate_walking_speed",
    "format_walking_speed_display",
    "is_plausible_walking_speed",
    "is_reportable_walking_speed",
    "_step_lengths_from_foot_events",
]
