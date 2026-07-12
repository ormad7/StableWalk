"""
Gait-cycle and video-duration evidence for stability scoring.

Short clips with few complete cycles must not receive the same confidence as
long recordings with multiple cycles. Domains that require cross-cycle comparison
are UNAVAILABLE rather than assigned fallback perfect scores.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from stablewalk.analysis.gait_cycle_analysis import DetectedGaitCycle, GaitCycleAnalysisResult
from stablewalk.models.pose_data import PoseSequence

DomainAvailability = Literal["AVAILABLE", "LOW_CONFIDENCE", "UNAVAILABLE"]
RepeatabilityTier = Literal["UNAVAILABLE", "LOW_CONFIDENCE", "NORMAL"]


@dataclass(frozen=True)
class GaitEvidenceThresholds:
    """
    Configurable minimum evidence for stability domains.

    Usable-cycle tiers (complete cycles meeting duration/frame gates):
      0 cycles → repeatability domains UNAVAILABLE
      1 cycle  → single-cycle metrics only; repeatability UNAVAILABLE
      2 cycles → repeatability LOW_CONFIDENCE
      3+ cycles → normal repeatability (subject to pose quality)
    """

    min_usable_cycles_unavailable: int = 1
    min_usable_cycles_low_confidence: int = 2
    min_usable_cycles_normal: int = 3

    min_cycles_cycle_consistency: int = 2
    min_cycles_joint_repeatability: int = 2
    min_cycles_contact_pattern: int = 2
    min_heel_strikes_temporal: int = 2
    min_steps_per_side_temporal: int = 1

    min_cycle_duration_s: float = 0.45
    max_cycle_duration_s: float = 2.8
    min_cycle_frames: int = 6
    partial_cycle_video_fraction: float = 0.88

    def repeatability_tier(self, usable_cycles: int) -> RepeatabilityTier:
        if usable_cycles < self.min_usable_cycles_unavailable:
            return "UNAVAILABLE"
        if usable_cycles < self.min_usable_cycles_low_confidence:
            return "UNAVAILABLE"
        if usable_cycles < self.min_usable_cycles_normal:
            return "LOW_CONFIDENCE"
        return "NORMAL"

    def documentation(self) -> str:
        return (
            "Gait evidence thresholds:\n"
            f"  0 usable cycles: repeatability UNAVAILABLE; cycle consistency UNAVAILABLE\n"
            f"  1 usable cycle: single-cycle metrics only; repeatability UNAVAILABLE\n"
            f"  2 usable cycles: repeatability LOW_CONFIDENCE\n"
            f"  {self.min_usable_cycles_normal}+ usable cycles: normal repeatability\n"
            f"  Cycle consistency requires >= {self.min_cycles_cycle_consistency} usable cycles\n"
            f"  Joint repeatability requires >= {self.min_cycles_joint_repeatability} for LOW, "
            f">= {self.min_usable_cycles_normal} for NORMAL\n"
            f"  Contact pattern prefers >= {self.min_cycles_contact_pattern} usable cycles\n"
            f"  Complete cycle: duration [{self.min_cycle_duration_s}, {self.max_cycle_duration_s}] s, "
            f">= {self.min_cycle_frames} frames"
        )


DEFAULT_GAIT_EVIDENCE_THRESHOLDS = GaitEvidenceThresholds()


@dataclass
class VideoCaptureStats:
    duration_s: float
    fps: float
    total_frames: int
    valid_pose_frames: int
    valid_pose_frame_pct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": round(self.duration_s, 3),
            "fps": round(self.fps, 3),
            "total_frames": self.total_frames,
            "valid_pose_frames": self.valid_pose_frames,
            "valid_pose_frame_pct": round(self.valid_pose_frame_pct, 2),
        }


@dataclass
class GaitCycleInventory:
    left_heel_strikes: int = 0
    right_heel_strikes: int = 0
    left_steps: int = 0
    right_steps: int = 0
    total_heel_strikes: int = 0
    complete_cycles: int = 0
    partial_cycles: int = 0
    usable_cycles: int = 0
    complete_cycle_indices: list[int] = field(default_factory=list)
    partial_cycle_indices: list[int] = field(default_factory=list)
    usable_cycle_indices: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_heel_strikes": self.left_heel_strikes,
            "right_heel_strikes": self.right_heel_strikes,
            "left_steps": self.left_steps,
            "right_steps": self.right_steps,
            "total_heel_strikes": self.total_heel_strikes,
            "complete_gait_cycles": self.complete_cycles,
            "partial_gait_cycles": self.partial_cycles,
            "usable_gait_cycles": self.usable_cycles,
            "complete_cycle_indices": list(self.complete_cycle_indices),
            "partial_cycle_indices": list(self.partial_cycle_indices),
            "usable_cycle_indices": list(self.usable_cycle_indices),
        }


@dataclass
class DomainEvidence:
    domain: str
    evidence_summary: str
    availability_hint: DomainAvailability
    repeatability_tier: RepeatabilityTier | None = None
    counts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "evidence_summary": self.evidence_summary,
            "availability_hint": self.availability_hint,
            "repeatability_tier": self.repeatability_tier,
            "counts": dict(self.counts),
        }


@dataclass
class GaitEvidenceAssessment:
    video: VideoCaptureStats
    cycles: GaitCycleInventory
    repeatability_tier: RepeatabilityTier
    domain_evidence: dict[str, DomainEvidence]
    stability_frames: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def usable_gait_cycles(self) -> int:
        return self.cycles.usable_cycles

    def domain_summary(self, domain: str) -> str:
        ev = self.domain_evidence.get(domain)
        return ev.evidence_summary if ev else "—"

    def availability_for_domain(self, domain: str) -> DomainAvailability:
        ev = self.domain_evidence.get(domain)
        return ev.availability_hint if ev else "UNAVAILABLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video.to_dict(),
            "cycles": self.cycles.to_dict(),
            "repeatability_tier": self.repeatability_tier,
            "usable_gait_cycles": self.usable_gait_cycles,
            "stability_frames": self.stability_frames,
            "domain_evidence": {k: v.to_dict() for k, v in self.domain_evidence.items()},
            "warnings": list(self.warnings),
        }


def _video_stats(sequence: PoseSequence) -> VideoCaptureStats:
    total = len(sequence.frames)
    detected = sum(1 for f in sequence.frames if f.detected)
    fps = max(sequence.fps, 1e-6)
    duration = total / fps if total else 0.0
    if sequence.frames:
        duration = max(duration, sequence.frames[-1].timestamp_s)
    return VideoCaptureStats(
        duration_s=duration,
        fps=fps,
        total_frames=total,
        valid_pose_frames=detected,
        valid_pose_frame_pct=100.0 * detected / max(total, 1),
    )


def _is_fallback_cycle(
    cycle: DetectedGaitCycle,
    *,
    video_duration_s: float,
    left_hs_count: int,
    thresholds: GaitEvidenceThresholds,
) -> bool:
    """True when cycle spans clip boundary rather than consecutive heel strikes."""
    if left_hs_count >= 2:
        return False
    if video_duration_s <= 0:
        return True
    return cycle.duration_s >= video_duration_s * thresholds.partial_cycle_video_fraction


def _classify_cycle(
    cycle: DetectedGaitCycle,
    *,
    video_duration_s: float,
    left_hs_count: int,
    thresholds: GaitEvidenceThresholds,
) -> Literal["complete", "partial"]:
    if _is_fallback_cycle(cycle, video_duration_s=video_duration_s, left_hs_count=left_hs_count, thresholds=thresholds):
        return "partial"
    frame_span = cycle.end_frame - cycle.start_frame + 1
    if frame_span < thresholds.min_cycle_frames:
        return "partial"
    if cycle.duration_s < thresholds.min_cycle_duration_s:
        return "partial"
    if cycle.duration_s > thresholds.max_cycle_duration_s:
        return "partial"
    return "complete"


def assess_gait_evidence(
    sequence: PoseSequence,
    cycles: GaitCycleAnalysisResult,
    *,
    stability_frame_count: int,
    thresholds: GaitEvidenceThresholds | None = None,
) -> GaitEvidenceAssessment:
    """Build video/cycle inventory and per-domain evidence counts."""
    th = thresholds or DEFAULT_GAIT_EVIDENCE_THRESHOLDS
    video = _video_stats(sequence)
    m = cycles.metrics

    left_hs = m.left_heel_strike_count
    right_hs = m.right_heel_strike_count
    left_steps = max(0, left_hs - 1) if left_hs >= 2 else max(0, left_hs)
    right_steps = max(0, right_hs - 1) if right_hs >= 2 else max(0, right_hs)

    complete_indices: list[int] = []
    partial_indices: list[int] = []
    usable_indices: list[int] = []

    for cycle in cycles.cycles:
        kind = _classify_cycle(
            cycle,
            video_duration_s=video.duration_s,
            left_hs_count=left_hs,
            thresholds=th,
        )
        if kind == "complete":
            complete_indices.append(cycle.cycle_index)
            usable_indices.append(cycle.cycle_index)
        else:
            partial_indices.append(cycle.cycle_index)

    inventory = GaitCycleInventory(
        left_heel_strikes=left_hs,
        right_heel_strikes=right_hs,
        left_steps=left_steps,
        right_steps=right_steps,
        total_heel_strikes=left_hs + right_hs,
        complete_cycles=len(complete_indices),
        partial_cycles=len(partial_indices),
        usable_cycles=len(usable_indices),
        complete_cycle_indices=complete_indices,
        partial_cycle_indices=partial_indices,
        usable_cycle_indices=usable_indices,
    )

    rep_tier = th.repeatability_tier(inventory.usable_cycles)
    warnings: list[str] = []

    if inventory.usable_cycles == 0:
        warnings.append("No usable complete gait cycles — repeatability metrics unavailable.")
    elif inventory.usable_cycles == 1:
        warnings.append(
            "Only one usable gait cycle — cross-cycle repeatability and cycle consistency unavailable."
        )
    elif inventory.usable_cycles == 2:
        warnings.append("Two usable gait cycles — repeatability reported at LOW_CONFIDENCE.")

    if video.duration_s < 2.5:
        warnings.append(
            f"Short clip ({video.duration_s:.2f}s) — temporal and cycle metrics have reduced reliability."
        )

    pelvis_frames = stability_frame_count
    swing_frames = sum(
        1 for s in cycles.per_frame
        if (not s.left_contact or not s.right_contact)
    )

    def _temporal_availability() -> DomainAvailability:
        if inventory.total_heel_strikes < th.min_heel_strikes_temporal:
            return "UNAVAILABLE"
        if inventory.usable_cycles < th.min_usable_cycles_low_confidence:
            return "LOW_CONFIDENCE"
        if inventory.usable_cycles < th.min_usable_cycles_normal:
            return "LOW_CONFIDENCE"
        return "AVAILABLE"

    def _repeatability_domains_availability() -> DomainAvailability:
        tier = rep_tier
        if tier == "UNAVAILABLE":
            return "UNAVAILABLE"
        if tier == "LOW_CONFIDENCE":
            return "LOW_CONFIDENCE"
        return "AVAILABLE"

    domain_evidence = {
        "temporal_symmetry": DomainEvidence(
            domain="temporal_symmetry",
            evidence_summary=(
                f"{left_hs} left / {right_hs} right heel strikes; "
                f"{left_steps} left / {right_steps} right step intervals"
            ),
            availability_hint=_temporal_availability(),
            counts={
                "left_heel_strikes": left_hs,
                "right_heel_strikes": right_hs,
                "left_steps": left_steps,
                "right_steps": right_steps,
                "usable_cycles": inventory.usable_cycles,
            },
        ),
        "spatial_symmetry": DomainEvidence(
            domain="spatial_symmetry",
            evidence_summary=(
                f"{left_steps} left / {right_steps} right steps; "
                f"{inventory.usable_cycles} usable cycles"
            ),
            availability_hint=(
                "UNAVAILABLE"
                if inventory.total_heel_strikes < th.min_heel_strikes_temporal
                else "LOW_CONFIDENCE"
                if inventory.usable_cycles < th.min_usable_cycles_low_confidence
                else "AVAILABLE"
            ),
            counts={"left_steps": left_steps, "right_steps": right_steps},
        ),
        "pelvis_stability": DomainEvidence(
            domain="pelvis_stability",
            evidence_summary=f"{pelvis_frames} stability frames",
            availability_hint="AVAILABLE" if pelvis_frames >= 8 else "UNAVAILABLE",
            counts={"valid_frames": pelvis_frames},
        ),
        "trunk_stability": DomainEvidence(
            domain="trunk_stability",
            evidence_summary=f"{pelvis_frames} stability frames",
            availability_hint="AVAILABLE" if pelvis_frames >= 8 else "UNAVAILABLE",
            counts={"valid_frames": pelvis_frames},
        ),
        "foot_clearance": DomainEvidence(
            domain="foot_clearance",
            evidence_summary=f"{swing_frames} swing-phase frames",
            availability_hint=(
                "UNAVAILABLE" if swing_frames < 4 else "LOW_CONFIDENCE" if swing_frames < 12 else "AVAILABLE"
            ),
            counts={"swing_frames": swing_frames},
        ),
        "joint_smoothness": DomainEvidence(
            domain="joint_smoothness",
            evidence_summary=(
                f"{inventory.usable_cycles} usable cycles for repeatability; "
                f"{pelvis_frames} stability frames"
            ),
            availability_hint=(
                "UNAVAILABLE"
                if pelvis_frames < 8
                else "LOW_CONFIDENCE"
                if rep_tier != "NORMAL"
                else "AVAILABLE"
            ),
            repeatability_tier=rep_tier,
            counts={
                "usable_cycles": inventory.usable_cycles,
                "valid_frames": pelvis_frames,
            },
        ),
        "cycle_consistency": DomainEvidence(
            domain="cycle_consistency",
            evidence_summary=f"{inventory.usable_cycles} usable complete cycles",
            availability_hint=(
                "UNAVAILABLE"
                if inventory.usable_cycles < th.min_cycles_cycle_consistency
                else "LOW_CONFIDENCE"
                if inventory.usable_cycles < th.min_usable_cycles_normal
                else "AVAILABLE"
            ),
            repeatability_tier=rep_tier,
            counts={"usable_cycles": inventory.usable_cycles},
        ),
        "contact_pattern": DomainEvidence(
            domain="contact_pattern",
            evidence_summary=(
                f"{inventory.total_heel_strikes} heel strikes; "
                f"{len(cycles.per_frame)} contact frames"
            ),
            availability_hint=(
                "UNAVAILABLE"
                if inventory.total_heel_strikes < th.min_heel_strikes_temporal
                else "LOW_CONFIDENCE"
                if inventory.usable_cycles < th.min_cycles_contact_pattern
                else "AVAILABLE"
            ),
            counts={
                "heel_strikes": inventory.total_heel_strikes,
                "contact_frames": len(cycles.per_frame),
                "usable_cycles": inventory.usable_cycles,
            },
        ),
    }

    return GaitEvidenceAssessment(
        video=video,
        cycles=inventory,
        repeatability_tier=rep_tier,
        domain_evidence=domain_evidence,
        stability_frames=stability_frame_count,
        warnings=warnings,
    )


__all__ = [
    "DEFAULT_GAIT_EVIDENCE_THRESHOLDS",
    "DomainEvidence",
    "GaitCycleInventory",
    "GaitEvidenceAssessment",
    "GaitEvidenceThresholds",
    "RepeatabilityTier",
    "VideoCaptureStats",
    "assess_gait_evidence",
]
