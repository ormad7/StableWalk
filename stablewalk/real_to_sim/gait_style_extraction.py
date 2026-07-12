"""
Extract gait style characteristics from video pose for Real-to-Sim imitation.

Captures stride length, cadence, hip sway, and arm swing — the visual
"fingerprint" described in the research spec (Perception Layer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.analysis.gait_feature_analysis import (
    GaitFeatureAnalysisResult,
    compute_normalized_gait_features,
    estimate_body_segment_dimensions,
)
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3


@dataclass(frozen=True)
class GaitStyleFingerprint:
    """Gait style extracted from a walking video — used to initialize simulation."""

    stride_length_m: float | None
    step_length_m: float | None
    cadence_steps_per_min: float | None
    hip_sway_m: float | None
    arm_swing_m: float | None
    trunk_sway_m: float | None
    normalized_stride: float | None
    normalized_step: float | None
    normalized_hip_sway: float | None
    contact_confidence: float
    usable_cycles: int
    style_summary: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stride_length_m": self.stride_length_m,
            "step_length_m": self.step_length_m,
            "cadence_steps_per_min": self.cadence_steps_per_min,
            "hip_sway_m": self.hip_sway_m,
            "arm_swing_m": self.arm_swing_m,
            "trunk_sway_m": self.trunk_sway_m,
            "normalized_stride": self.normalized_stride,
            "normalized_step": self.normalized_step,
            "normalized_hip_sway": self.normalized_hip_sway,
            "contact_confidence": self.contact_confidence,
            "usable_cycles": self.usable_cycles,
            "style_summary": self.style_summary,
            "confidence": self.confidence,
        }


def _pelvis(snap: SkeletonSnapshot) -> Vec3 | None:
    lh = snap.joints.get("left_hip")
    rh = snap.joints.get("right_hip")
    if lh is None or rh is None:
        return None
    return Vec3(
        (lh.position.x + rh.position.x) * 0.5,
        (lh.position.y + rh.position.y) * 0.5,
        (lh.position.z + rh.position.z) * 0.5,
    )


def _arm_swing_range(recording: GaitMotionRecording) -> float | None:
    """Peak wrist displacement relative to shoulder (mediolateral proxy)."""
    spans: list[float] = []
    for side in ("left", "right"):
        wrist_key = f"{side}_wrist"
        shoulder_key = f"{side}_shoulder"
        offsets: list[float] = []
        for snap in recording.snapshots:
            w = snap.joints.get(wrist_key)
            s = snap.joints.get(shoulder_key)
            if w is None or s is None:
                continue
            offsets.append(abs(w.position.x - s.position.x))
        if offsets:
            spans.append(max(offsets) - min(offsets))
    if not spans:
        return None
    return float(np.mean(spans))


def _style_summary(
    *,
    cadence: float | None,
    stride: float | None,
    hip_sway: float | None,
    arm_swing: float | None,
) -> str:
    parts: list[str] = []
    if cadence is not None:
        if cadence < 95:
            parts.append("slow cadence")
        elif cadence > 125:
            parts.append("fast athletic cadence")
        else:
            parts.append("typical adult cadence")
    if stride is not None:
        parts.append(f"stride ~{stride * 100:.0f} cm")
    if hip_sway is not None and hip_sway > 0.04:
        parts.append("noticeable hip sway")
    elif hip_sway is not None and hip_sway < 0.02:
        parts.append("compact hip motion")
    if arm_swing is not None and arm_swing > 0.12:
        parts.append("active arm swing")
    if not parts:
        return "Gait style fingerprint — limited data from short clip."
    return " · ".join(parts).capitalize() + "."


def extract_gait_style_fingerprint(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    *,
    gait_features: GaitFeatureAnalysisResult | None = None,
) -> GaitStyleFingerprint:
    """Build a gait style fingerprint for Real-to-Sim initialization."""
    dimensions = estimate_body_segment_dimensions(recording)
    spatial = compute_normalized_gait_features(recording, cycles, dimensions)
    cadence = cycles.metrics.cadence_steps_per_min
    hip_sway = spatial.pelvis_mediolateral_range_m
    trunk_sway = spatial.trunk_sway_m
    arm_swing = _arm_swing_range(recording)

    usable = len(cycles.cycles) or max(0, len(cycles.events) // 4)

    conf_parts = [cycles.metrics.contact_confidence]
    if spatial.step_length_m is not None:
        conf_parts.append(0.75)
    if cadence is not None:
        conf_parts.append(0.7)
    if arm_swing is not None:
        conf_parts.append(0.55)
    confidence = float(min(1.0, sum(conf_parts) / len(conf_parts)))

    summary = _style_summary(
        cadence=cadence,
        stride=spatial.stride_length_m,
        hip_sway=hip_sway,
        arm_swing=arm_swing,
    )

    return GaitStyleFingerprint(
        stride_length_m=spatial.stride_length_m,
        step_length_m=spatial.step_length_m,
        cadence_steps_per_min=cadence,
        hip_sway_m=hip_sway,
        arm_swing_m=arm_swing,
        trunk_sway_m=trunk_sway,
        normalized_stride=spatial.normalized_stride_length,
        normalized_step=spatial.normalized_step_length,
        normalized_hip_sway=spatial.normalized_pelvis_sway,
        contact_confidence=cycles.metrics.contact_confidence,
        usable_cycles=usable,
        style_summary=summary,
        confidence=confidence,
    )


__all__ = ["GaitStyleFingerprint", "extract_gait_style_fingerprint"]
