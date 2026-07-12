"""Orchestrate full biomechanical analysis pipeline."""

from __future__ import annotations

import logging
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
    if result.stability_margin and result.stability_margin.stable_pct < 50.0:
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
    margin = analyze_stability_margin(com, bos)

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

    gait_metrics = analyze_advanced_gait_metrics(recording, cycles, features, contact)
    gait_quality = compute_gait_quality_score(
        stability=stability,
        stability_margin=margin,
        symmetry=symmetry,
        gait_metrics=gait_metrics,
        cycles=cycles,
        contact=contact,
    )

    result = BiomechanicalAnalysisResult(
        video_quality=video_quality,
        center_of_mass=com,
        base_of_support=bos,
        stability_margin=margin,
        symmetry=symmetry,
        joint_rom=rom,
        gait_metrics=gait_metrics,
        gait_quality=gait_quality,
        stability=stability,
        cycles=cycles,
        contact=contact,
        features=features,
        warnings=warnings,
    )
    result.abnormalities = _detect_abnormalities(result)
    return result


__all__ = ["BiomechanicalAnalysisResult", "run_biomechanical_analysis"]
