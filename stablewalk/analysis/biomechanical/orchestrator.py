"""Orchestrate full biomechanical analysis pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.biomechanical.advanced_gait_metrics import (
    AdvancedGaitMetrics,
    analyze_advanced_gait_metrics,
)
from stablewalk.analysis.biomechanical.base_of_support import (
    BaseOfSupportAnalysis,
    analyze_base_of_support,
)
from stablewalk.analysis.biomechanical.com_estimation import (
    CenterOfMassAnalysis,
    analyze_center_of_mass,
)
from stablewalk.analysis.biomechanical.gait_quality_score import compute_gait_quality_score
from stablewalk.analysis.biomechanical.joint_rom import (
    JointROMAnalysis,
    analyze_joint_rom,
    extract_side_rom,
)
from stablewalk.analysis.biomechanical.stability_margin import (
    StabilityMarginAnalysis,
    analyze_stability_margin,
)
from stablewalk.analysis.biomechanical.symmetry_metrics import SymmetryAnalysis, analyze_symmetry
from stablewalk.analysis.biomechanical.types import GaitQualityScore
from stablewalk.analysis.biomechanical.video_quality import VideoQualityAssessment, assess_video_quality
from stablewalk.analysis.biomechanical.walking_speed import _meters_per_normalized_unit
from stablewalk.config import DEFAULT_SUBJECT_HEIGHT_M
from stablewalk.analysis.biomech_stability import StabilityResult, analyze_biomech_stability
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult, analyze_foot_contact
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, analyze_gait_cycles
from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult, analyze_gait_features
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.pose_data import PoseSequence

logger = logging.getLogger(__name__)


@dataclass
class BiomechanicalAnalysisResult:
    """Complete biomechanical analysis bundle."""

    video_quality: VideoQualityAssessment | None = None
    center_of_mass: CenterOfMassAnalysis | None = None
    base_of_support: BaseOfSupportAnalysis | None = None
    stability_margin: StabilityMarginAnalysis | None = None
    symmetry: SymmetryAnalysis | None = None
    joint_rom: JointROMAnalysis | None = None
    gait_metrics: AdvancedGaitMetrics | None = None
    gait_quality: GaitQualityScore | None = None
    stability: StabilityResult | None = None
    cycles: GaitCycleAnalysisResult | None = None
    contact: FootContactAnalysisResult | None = None
    features: GaitFeatureAnalysisResult | None = None
    abnormalities: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_quality": None if self.video_quality is None else self.video_quality.to_dict(),
            "center_of_mass": None if self.center_of_mass is None else self.center_of_mass.to_dict(),
            "base_of_support": None if self.base_of_support is None else self.base_of_support.to_dict(),
            "stability_margin": None
            if self.stability_margin is None
            else self.stability_margin.to_dict(),
            "symmetry": None if self.symmetry is None else self.symmetry.to_dict(),
            "joint_rom": None if self.joint_rom is None else self.joint_rom.to_dict(),
            "gait_metrics": None if self.gait_metrics is None else self.gait_metrics.to_dict(),
            "gait_quality": None if self.gait_quality is None else self.gait_quality.to_dict(),
            "stability": None if self.stability is None else {"score": self.stability.score},
            "abnormalities": list(self.abnormalities),
            "warnings": list(self.warnings),
        }


def _detect_abnormalities(result: BiomechanicalAnalysisResult) -> list[str]:
    flags: list[str] = []
    if result.symmetry and result.symmetry.overall_symmetry_pct:
        v = result.symmetry.overall_symmetry_pct.value
        if v is not None and v < 65.0:
            flags.append("Reduced left–right symmetry")
    if (
        result.stability_margin
        and result.stability_margin.stable_pct is not None
        and result.stability_margin.stable_pct < 50.0
    ):
        flags.append("Low stability margin — COM frequently near/outside support base")
    if result.gait_quality and result.gait_quality.score < 55.0:
        flags.append("Low composite gait quality score")
    if result.gait_metrics and result.gait_metrics.cadence:
        cad = result.gait_metrics.cadence.value
        if cad is not None and (cad < 85 or cad > 130):
            flags.append(f"Atypical cadence ({cad:.0f} steps/min)")
    if result.video_quality and result.video_quality.overall_quality_score < 50.0:
        flags.append("Poor input video quality — interpret metrics cautiously")
    return flags


def _validate_final_metrics(result: BiomechanicalAnalysisResult) -> list[str]:
    """Invalidate mathematically impossible final values and explain why."""
    warnings: list[str] = []

    def invalidate(metric: Any, label: str, reason: str) -> None:
        if metric is None:
            return
        metric.value = None
        metric.confidence = 0.0
        metric.note = f"N/A — validation failed: {reason}"
        warnings.append(f"{label} invalidated: {reason}")

    gait = result.gait_metrics
    if gait is not None:
        for name in (
            "stance_pct",
            "swing_pct",
            "left_stance_pct",
            "right_stance_pct",
            "left_swing_pct",
            "right_swing_pct",
            "double_support_pct",
            "single_support_pct",
        ):
            metric = getattr(gait, name, None)
            if metric is not None and metric.value is not None:
                if not math.isfinite(metric.value) or not 0.0 <= metric.value <= 100.0:
                    invalidate(metric, name, f"{metric.value!r} is outside 0–100%.")
        for side in ("left", "right"):
            stance = getattr(gait, f"{side}_stance_pct", None)
            swing = getattr(gait, f"{side}_swing_pct", None)
            if (
                stance is not None
                and swing is not None
                and stance.value is not None
                and swing.value is not None
                and abs(stance.value + swing.value - 100.0) > 0.5
            ):
                reason = (
                    f"{side} stance + swing equals "
                    f"{stance.value + swing.value:.2f}%, expected approximately 100%."
                )
                invalidate(stance, f"{side}_stance_pct", reason)
                invalidate(swing, f"{side}_swing_pct", reason)
        support_metrics = [
            gait.double_support_pct,
            gait.single_support_pct,
        ]
        flight_pct = (
            result.cycles.metrics.flight_pct
            if result.cycles is not None
            else None
        )
        if (
            all(metric is not None and metric.value is not None for metric in support_metrics)
            and flight_pct is not None
        ):
            support_total = sum(metric.value for metric in support_metrics if metric and metric.value is not None)
            support_total += flight_pct
            if abs(support_total - 100.0) > 0.5:
                reason = (
                    f"double + single support + flight equals {support_total:.2f}%, "
                    "expected approximately 100%."
                )
                for metric, label in zip(
                    support_metrics, ("double_support_pct", "single_support_pct")
                ):
                    invalidate(metric, label, reason)
        for name in ("cadence", "walking_speed", "stride_length", "step_length", "step_width"):
            metric = getattr(gait, name, None)
            if metric is not None and metric.value is not None:
                if not math.isfinite(metric.value) or metric.value < 0.0:
                    invalidate(metric, name, f"{metric.value!r} must be finite and non-negative.")

    if result.joint_rom is not None:
        for joint in result.joint_rom.joints:
            if joint.rom_deg is not None and (
                not math.isfinite(joint.rom_deg) or not 0.0 <= joint.rom_deg <= 360.0
            ):
                warnings.append(
                    f"{joint.side} {joint.joint} ROM invalidated: "
                    f"{joint.rom_deg!r}° is outside 0–360°."
                )
                joint.rom_deg = None
                joint.confidence = 0.0
                joint.note = "N/A — validation failed: ROM must be within 0–360°."

    if result.symmetry is not None:
        metric = result.symmetry.overall_symmetry_pct
        if metric is not None and metric.value is not None and (
            not math.isfinite(metric.value) or not 0.0 <= metric.value <= 100.0
        ):
            invalidate(metric, "overall_symmetry_pct", "value is outside 0–100%.")

    margin = result.stability_margin
    if margin is not None:
        if margin.stable_pct is not None and (
            not math.isfinite(margin.stable_pct) or not 0.0 <= margin.stable_pct <= 100.0
        ):
            warnings.append(
                f"Stability percentage invalidated: {margin.stable_pct!r} is outside 0–100%."
            )
            margin.stable_pct = None
        if margin.mean_margin_m is not None and not math.isfinite(margin.mean_margin_m):
            warnings.append("Mean stability margin invalidated: value is not finite.")
            margin.mean_margin_m = None
    return warnings


def run_biomechanical_analysis(
    recording: GaitMotionRecording,
    sequence: PoseSequence | None = None,
    *,
    cycles: GaitCycleAnalysisResult | None = None,
    contact: FootContactAnalysisResult | None = None,
    features: GaitFeatureAnalysisResult | None = None,
    stability: StabilityResult | None = None,
) -> BiomechanicalAnalysisResult:
    """
    Run full biomechanical analysis after pose + contact pipeline stages.

    Does not replace existing stability or gait modules — composes them.
    """
    warnings: list[str] = []

    video_quality = assess_video_quality(sequence) if sequence else None
    if video_quality:
        warnings.extend(video_quality.warnings)

    if stability is None and sequence is not None:
        stability = analyze_biomech_stability(sequence)

    if cycles is None:
        cycles = analyze_gait_cycles(recording)
    if contact is None:
        contact = analyze_foot_contact(recording, cycles=cycles)
    if features is None and sequence is not None:
        features = analyze_gait_features(recording, cycles, sequence=sequence)

    com = analyze_center_of_mass(recording, contact)
    bos = analyze_base_of_support(recording, contact)
    margin = analyze_stability_margin(
        com,
        bos,
        meters_per_unit=_meters_per_normalized_unit(
            recording,
            subject_height_m=DEFAULT_SUBJECT_HEIGHT_M,
        ),
    )

    rom = analyze_joint_rom(
        recording,
        cycles,
        sequence=sequence,
        contact_confidence=contact.metrics.contact_confidence,
    )
    rom_map = extract_side_rom(rom)

    symmetry = analyze_symmetry(
        cycles,
        features,
        contact,
        knee_rom_left=rom_map.get("knee_rom_left"),
        knee_rom_right=rom_map.get("knee_rom_right"),
        hip_rom_left=rom_map.get("hip_rom_left"),
        hip_rom_right=rom_map.get("hip_rom_right"),
        ankle_rom_left=rom_map.get("ankle_rom_left"),
        ankle_rom_right=rom_map.get("ankle_rom_right"),
    )

    gait_metrics = analyze_advanced_gait_metrics(
        recording,
        cycles,
        features,
        contact,
        sequence=sequence,
        subject_height_m=DEFAULT_SUBJECT_HEIGHT_M,
    )
    result = BiomechanicalAnalysisResult(
        video_quality=video_quality,
        center_of_mass=com,
        base_of_support=bos,
        stability_margin=margin,
        symmetry=symmetry,
        joint_rom=rom,
        gait_metrics=gait_metrics,
        stability=stability,
        cycles=cycles,
        contact=contact,
        features=features,
        warnings=warnings,
    )
    result.warnings.extend(cycles.warnings)
    result.warnings.extend(features.warnings if features is not None else [])
    result.warnings.extend(margin.warnings)
    result.warnings.extend(rom.warnings)
    result.warnings.extend(symmetry.warnings)
    result.warnings.extend(_validate_final_metrics(result))
    result.warnings = list(dict.fromkeys(result.warnings))

    if cycles.metrics.metrics_reliable:
        result.gait_quality = compute_gait_quality_score(
            stability=stability,
            stability_margin=margin,
            symmetry=symmetry,
            gait_metrics=gait_metrics,
            cycles=cycles,
            contact=contact,
        )
    else:
        result.warnings.append(
            "Gait quality score unavailable: "
            f"{cycles.metrics.reliability_reason}"
        )
    result.abnormalities = _detect_abnormalities(result)
    return result


__all__ = ["BiomechanicalAnalysisResult", "run_biomechanical_analysis"]
