"""Advanced spatiotemporal gait metrics with confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.config import DEFAULT_SUBJECT_HEIGHT_M
from stablewalk.analysis.biomechanical.walking_speed import (
    _meters_per_normalized_unit,
    estimate_walking_speed,
)
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence


@dataclass
class AdvancedGaitMetrics:
    cadence: MetricWithConfidence | None = None
    walking_speed: MetricWithConfidence | None = None
    stride_length: MetricWithConfidence | None = None
    step_length: MetricWithConfidence | None = None
    step_width: MetricWithConfidence | None = None
    stride_time: MetricWithConfidence | None = None
    step_time: MetricWithConfidence | None = None
    stance_pct: MetricWithConfidence | None = None
    swing_pct: MetricWithConfidence | None = None
    left_stance_pct: MetricWithConfidence | None = None
    right_stance_pct: MetricWithConfidence | None = None
    left_swing_pct: MetricWithConfidence | None = None
    right_swing_pct: MetricWithConfidence | None = None
    double_support_pct: MetricWithConfidence | None = None
    single_support_pct: MetricWithConfidence | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in (
            "cadence",
            "walking_speed",
            "stride_length",
            "step_length",
            "step_width",
            "stride_time",
            "step_time",
            "stance_pct",
            "swing_pct",
            "left_stance_pct",
            "right_stance_pct",
            "left_swing_pct",
            "right_swing_pct",
            "double_support_pct",
            "single_support_pct",
        ):
            m = getattr(self, key)
            if m is not None:
                out[key] = m.to_dict()
        return out


def _m(
    value: float | None,
    *,
    unit: str,
    confidence: float,
    note: str = "",
) -> MetricWithConfidence:
    return MetricWithConfidence(
        value=round(value, 4) if value is not None else None,
        unit=unit,
        kind="estimated",
        confidence=confidence if value is not None else 0.0,
        note=note,
    )


def _mean_step_width(recording: GaitMotionRecording) -> float | None:
    """Mediolateral ankle separation (orthogonal to estimated progression)."""
    dx_vals: list[float] = []
    dz_vals: list[float] = []
    for snap in recording.snapshots:
        la = snap.joints.get("left_ankle")
        ra = snap.joints.get("right_ankle")
        if la and ra:
            dx_vals.append(abs(la.position.x - ra.position.x))
            dz_vals.append(abs(la.position.z - ra.position.z))
    if not dx_vals:
        return None
    # Width is the smaller horizontal component when one axis is progression.
    # Sagittal gait: progression ≈ X → width ≈ |ΔZ|; frontal: opposite.
    med_dx = float(np.median(dx_vals))
    med_dz = float(np.median(dz_vals)) if dz_vals else 0.0
    width = med_dz if med_dx >= med_dz else med_dx
    return width if width > 1e-6 else None


def analyze_advanced_gait_metrics(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult | None,
    features: GaitFeatureAnalysisResult | None,
    contact: FootContactAnalysisResult | None,
    *,
    sequence: PoseSequence | None = None,
    subject_height_m: float = DEFAULT_SUBJECT_HEIGHT_M,
) -> AdvancedGaitMetrics:
    """Spatiotemporal gait descriptors with per-metric confidence."""
    m = cycles.metrics if cycles else None
    nf = features.features if features else None
    result = AdvancedGaitMetrics()
    reliable = bool(m and m.metrics_reliable)
    unavailable_reason = (
        m.reliability_reason
        if m is not None
        else "No gait-cycle analysis was available."
    )
    if not reliable:
        note = f"N/A — {unavailable_reason}"
        for key, unit in (
            ("cadence", "steps/min"),
            ("walking_speed", "m/s"),
            ("stride_length", "m"),
            ("step_length", "m"),
            ("step_width", "m"),
            ("stride_time", "s"),
            ("step_time", "s"),
            ("stance_pct", "%"),
            ("swing_pct", "%"),
            ("left_stance_pct", "%"),
            ("right_stance_pct", "%"),
            ("left_swing_pct", "%"),
            ("right_swing_pct", "%"),
            ("double_support_pct", "%"),
            ("single_support_pct", "%"),
        ):
            setattr(result, key, _m(None, unit=unit, confidence=0.0, note=note))
        return result

    assert m is not None
    conf = m.contact_confidence
    result.cadence = _m(
        m.cadence_steps_per_min,
        unit="steps/min",
        confidence=conf,
        note=f"Alternating heel-strike intervals; {m.reliability_reason}",
    )
    result.stride_time = _m(
        m.stride_time_s,
        unit="s",
        confidence=conf,
        note="Mean interval between same-side heel strikes over complete cycles.",
    )
    result.step_time = _m(
        m.step_time_s,
        unit="s",
        confidence=conf,
        note="Mean interval between alternating heel strikes inside complete cycles.",
    )
    result.double_support_pct = _m(
        m.double_support_pct,
        unit="%",
        confidence=conf,
        note=m.double_support_definition,
    )
    result.single_support_pct = _m(
        m.single_support_pct,
        unit="%",
        confidence=conf,
        note=(
            "Ipsilateral single-limb support as % of the gait cycle "
            "(mean of left-only and right-only; typically ~40%)."
        ),
    )
    for key in ("left_stance_pct", "right_stance_pct", "left_swing_pct", "right_swing_pct"):
        setattr(
            result,
            key,
            _m(
                getattr(m, key),
                unit="%",
                confidence=conf,
                note="Per-foot percentage of complete same-side heel-strike cycles.",
            ),
        )
    stance_values = [v for v in (m.left_stance_pct, m.right_stance_pct) if v is not None]
    swing_values = [v for v in (m.left_swing_pct, m.right_swing_pct) if v is not None]
    result.stance_pct = _m(
        float(np.mean(stance_values)) if stance_values else None,
        unit="%",
        confidence=conf,
        note="Bilateral mean; side-specific percentages are also reported.",
    )
    result.swing_pct = _m(
        float(np.mean(swing_values)) if swing_values else None,
        unit="%",
        confidence=conf,
        note="Bilateral mean; side-specific percentages are also reported.",
    )

    meters_per_unit = _meters_per_normalized_unit(
        recording, subject_height_m=subject_height_m
    )
    if nf:
        result.stride_length = _m(
            nf.stride_length_m * meters_per_unit
            if nf.stride_length_m is not None
            else None,
            unit="m",
            confidence=conf * 0.85,
            note=(
                "Twice mean step length, scaled from body-normalized pose using "
                f"{subject_height_m:.2f} m subject stature."
            ),
        )
        result.step_length = _m(
            nf.step_length_m * meters_per_unit
            if nf.step_length_m is not None
            else None,
            unit="m",
            confidence=conf * 0.85,
            note=(
                "Progression-axis distance between feet at heel strike, scaled "
                f"using {subject_height_m:.2f} m subject stature."
            ),
        )
    else:
        result.stride_length = _m(
            None,
            unit="m",
            confidence=0.0,
            note="N/A — no reliable spatial gait features were available.",
        )
        result.step_length = _m(
            None,
            unit="m",
            confidence=0.0,
            note="N/A — no reliable spatial gait features were available.",
        )

    result.walking_speed = estimate_walking_speed(
        recording,
        sequence=sequence,
        cycles=cycles,
        features=features,
        contact=contact,
        subject_height_m=subject_height_m,
        base_confidence=conf,
    )
    if result.walking_speed is None:
        result.walking_speed = _m(
            None,
            unit="m/s",
            confidence=0.0,
            note="N/A — no scale-aware walking-speed method passed reliability checks.",
        )
    result.step_width = _m(
        (
            width * meters_per_unit
            if (width := _mean_step_width(recording)) is not None
            else None
        ),
        unit="m",
        confidence=conf * 0.75,
        note="Mediolateral ankle separation (body-normalized)",
    )
    return result


__all__ = ["AdvancedGaitMetrics", "analyze_advanced_gait_metrics"]
