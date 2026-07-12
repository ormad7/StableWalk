"""Advanced spatiotemporal gait metrics with confidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from stablewalk.analysis.biomechanical.types import MetricWithConfidence
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult
from stablewalk.models.gait_motion import GaitMotionRecording


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
) -> MetricWithConfidence | None:
    if value is None:
        return None
    return MetricWithConfidence(
        value=round(value, 4),
        unit=unit,
        kind="estimated",
        confidence=confidence,
        note=note,
    )


def _pelvis_speed_m_s(recording: GaitMotionRecording) -> float | None:
    speeds: list[float] = []
    prev = None
    prev_t = None
    for snap in recording.snapshots:
        lh = snap.joints.get("left_hip")
        rh = snap.joints.get("right_hip")
        if lh is None or rh is None:
            continue
        px = (lh.position.x + rh.position.x) * 0.5
        pz = (lh.position.z + rh.position.z) * 0.5
        if prev is not None and prev_t is not None:
            dt = snap.time_s - prev_t
            if dt > 1e-6:
                dist = np.hypot(px - prev[0], pz - prev[1])
                speeds.append(dist / dt)
        prev = (px, pz)
        prev_t = snap.time_s
    if not speeds:
        return None
    return float(np.median(speeds))


def _mean_step_width(recording: GaitMotionRecording) -> float | None:
    widths: list[float] = []
    for snap in recording.snapshots:
        la = snap.joints.get("left_ankle")
        ra = snap.joints.get("right_ankle")
        if la and ra:
            widths.append(abs(la.position.x - ra.position.x))
    return float(np.median(widths)) if widths else None


def analyze_advanced_gait_metrics(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult | None,
    features: GaitFeatureAnalysisResult | None,
    contact: FootContactAnalysisResult | None,
) -> AdvancedGaitMetrics:
    """Spatiotemporal gait descriptors with per-metric confidence."""
    conf = 0.5
    if cycles:
        conf = max(conf, cycles.metrics.contact_confidence)
    if contact:
        conf = max(conf, contact.metrics.contact_confidence)

    m = cycles.metrics if cycles else None
    nf = features.features if features else None

    result = AdvancedGaitMetrics()
    result.cadence = _m(
        m.cadence_steps_per_min if m else None,
        unit="steps/min",
        confidence=conf,
        note="From heel-strike intervals",
    )
    result.stride_time = _m(
        m.stride_time_s if m else None,
        unit="s",
        confidence=conf,
    )
    result.step_time = _m(
        m.step_time_s if m else None,
        unit="s",
        confidence=conf,
    )
    result.double_support_pct = _m(
        m.double_support_pct if m else None,
        unit="%",
        confidence=conf,
    )

    if m:
        stance = m.average_stance_duration_s
        swing = m.average_swing_duration_s
        cycle_d = (stance or 0) + (swing or 0)
        if cycle_d > 1e-6:
            result.stance_pct = _m(
                (stance or 0) / cycle_d * 100.0,
                unit="%",
                confidence=conf,
            )
            result.swing_pct = _m(
                (swing or 0) / cycle_d * 100.0,
                unit="%",
                confidence=conf,
            )
        ds = m.double_support_pct or 0.0
        result.single_support_pct = _m(
            100.0 - ds,
            unit="%",
            confidence=conf * 0.9,
            note="100% − double-support %",
        )

    if nf:
        result.stride_length = _m(nf.stride_length_m, unit="m", confidence=conf * 0.85)
        result.step_length = _m(nf.step_length_m, unit="m", confidence=conf * 0.85)

    result.walking_speed = _m(
        _pelvis_speed_m_s(recording),
        unit="m/s",
        confidence=conf * 0.8,
        note="Pelvis horizontal speed — monocular scale",
    )
    result.step_width = _m(
        _mean_step_width(recording),
        unit="m",
        confidence=conf * 0.75,
        note="Mediolateral ankle separation",
    )
    return result


__all__ = ["AdvancedGaitMetrics", "analyze_advanced_gait_metrics"]
