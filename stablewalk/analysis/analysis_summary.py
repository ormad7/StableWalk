"""
Build a human-readable analysis summary from existing StableWalk results.

All values are computed from the current session — no hard-coded placeholders.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.analysis.biomechanical.walking_speed import (
    UNAVAILABLE_LABEL,
    format_walking_speed_display,
    is_reportable_walking_speed,
)
from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
from stablewalk.config import DEFAULT_SUBJECT_HEIGHT_M
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult
from stablewalk.metrics.session_view import _extract_cadence
from stablewalk.analysis.result_validity import (
    ResultValidityReport,
    build_result_validity,
    gait_quality_withhold_reason,
    laterality_withhold_reason,
)
from stablewalk.ui.scientific_labels import (
    LABEL_ANALYSIS_CONFIDENCE,
    LABEL_BIOMECHANICAL_VALIDITY,
    LABEL_CADENCE,
    LABEL_COM_FULL,
    LABEL_CONTACT_COVERAGE,
    LABEL_GAIT_QUALITY,
    LABEL_JOINT_ROM_SUMMARY,
    LABEL_POSE_COVERAGE,
    LABEL_SPECTRAL_RELIABILITY,
    LABEL_STABILITY_MARGIN,
    LABEL_SYMMETRY,
    LABEL_PIPELINE_CONFIDENCE,
    LABEL_TRACKING_CONFIDENCE,
    LABEL_VIRTUAL_GRF_PANEL,
    LABEL_VIDEO_QUALITY,
    TIER_CALCULATED,
    TIER_DERIVED,
    TIER_ESTIMATED,
    LABEL_WALKING_SPEED,
    export_terminology_block,
    format_cadence_value,
)

ConfidenceLevel = Literal["High", "Moderate", "Low"]

# Measurement-reliability levels — not pose tracking and not pipeline success.
# HIGH requires all of:
#   • ≥ HIGH_MIN_CYCLES complete same-side heel-strike cycles
#   • session duration ≥ HIGH_MIN_DURATION_S
#   • contact / cycle measurement score ≥ HIGH_MIN_SCORE
#   • contact tier is not LOW_CONFIDENCE
# Two cycles on a ~5 s clip can never be High, even if MediaPipe visibility is 95%+.
HIGH_MIN_CYCLES = 3
HIGH_MIN_DURATION_S = 6.0
HIGH_MIN_SCORE = 0.72
HIGH_CONFIDENCE_DEFINITION = (
    "High analysis confidence means measurement reliability, not execution "
    f"success: >={HIGH_MIN_CYCLES} complete same-side heel-strike cycles, "
    f"duration >={HIGH_MIN_DURATION_S:.0f} s, and contact/cycle score "
    f">={HIGH_MIN_SCORE:.0%}. Pose tracking is a separate field."
)

_PIPELINE_STATUS_SCORE = {
    "completed": 1.0,
    "partial": 0.55,
    "unavailable": 0.0,
}


@dataclass
class SummaryField:
    label: str
    value: str
    tier: str
    available: bool = True
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GaitEventStatus:
    name: str
    detected: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisSummary:
    """StableWalk Analysis Summary for the currently loaded video."""

    source: str = ""
    frame_index: int | None = None
    timestamp_s: float | None = None
    overall_gait_quality: SummaryField | None = None
    cadence: SummaryField | None = None
    walking_speed: SummaryField | None = None
    symmetry: SummaryField | None = None
    stability_margin: SummaryField | None = None
    center_of_mass: SummaryField | None = None
    estimated_virtual_grf: SummaryField | None = None
    joint_rom_summary: SummaryField | None = None
    gait_events: list[GaitEventStatus] = field(default_factory=list)
    video_quality: SummaryField | None = None
    tracking_confidence: SummaryField | None = None
    pipeline_confidence: SummaryField | None = None
    analysis_confidence: SummaryField | None = None
    pose_coverage: SummaryField | None = None
    contact_coverage: SummaryField | None = None
    contact_coverage_left: SummaryField | None = None
    contact_coverage_right: SummaryField | None = None
    cycle_evidence: SummaryField | None = None
    spectral_reliability: SummaryField | None = None
    biomechanical_validity: SummaryField | None = None
    result_validity: ResultValidityReport | None = None
    pipeline_status: dict[str, Any] | None = None
    scientific_interpretation: str = ""
    terminology: dict[str, str] = field(default_factory=export_terminology_block)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "terminology": self.terminology,
            "overall_gait_quality": None if self.overall_gait_quality is None else self.overall_gait_quality.to_dict(),
            "cadence": None if self.cadence is None else self.cadence.to_dict(),
            "walking_speed": None if self.walking_speed is None else self.walking_speed.to_dict(),
            "symmetry": None if self.symmetry is None else self.symmetry.to_dict(),
            "stability_margin": None if self.stability_margin is None else self.stability_margin.to_dict(),
            "center_of_mass": None if self.center_of_mass is None else self.center_of_mass.to_dict(),
            "estimated_virtual_grf": None
            if self.estimated_virtual_grf is None
            else self.estimated_virtual_grf.to_dict(),
            "joint_rom_summary": None
            if self.joint_rom_summary is None
            else self.joint_rom_summary.to_dict(),
            "gait_events": [e.to_dict() for e in self.gait_events],
            "video_quality": None if self.video_quality is None else self.video_quality.to_dict(),
            "tracking_confidence": None
            if self.tracking_confidence is None
            else self.tracking_confidence.to_dict(),
            "pipeline_confidence": None
            if self.pipeline_confidence is None
            else self.pipeline_confidence.to_dict(),
            "analysis_confidence": None
            if self.analysis_confidence is None
            else self.analysis_confidence.to_dict(),
            "pose_coverage": None if self.pose_coverage is None else self.pose_coverage.to_dict(),
            "contact_coverage": None
            if self.contact_coverage is None
            else self.contact_coverage.to_dict(),
            "contact_coverage_left": None
            if self.contact_coverage_left is None
            else self.contact_coverage_left.to_dict(),
            "contact_coverage_right": None
            if self.contact_coverage_right is None
            else self.contact_coverage_right.to_dict(),
            "cycle_evidence": None
            if self.cycle_evidence is None
            else self.cycle_evidence.to_dict(),
            "spectral_reliability": None
            if self.spectral_reliability is None
            else self.spectral_reliability.to_dict(),
            "biomechanical_validity": None
            if self.biomechanical_validity is None
            else self.biomechanical_validity.to_dict(),
            "result_validity": None
            if self.result_validity is None
            else self.result_validity.to_dict(),
            "pipeline_status": self.pipeline_status,
            "scientific_interpretation": self.scientific_interpretation,
        }


def _map_stability_display(engine_state: str) -> str:
    """Map engine states to shared Acceptable / Borderline / Reduced labels."""
    from stablewalk.metrics.status import DisplayStatus

    key = (engine_state or "").strip().lower()
    if key in ("—", "-", "unavailable", ""):
        return "—"
    if key == "stable":
        return DisplayStatus.ACCEPTABLE.value
    if "reduced" in key:
        return DisplayStatus.BORDERLINE.value
    if key == "unstable":
        return DisplayStatus.REDUCED.value
    return engine_state


def _dominant_stability_state(ba: BiomechanicalAnalysisResult) -> str:
    if ba.stability_margin is None or not ba.stability_margin.per_frame:
        return "—"
    counts: dict[str, int] = {}
    for f in ba.stability_margin.per_frame:
        if f.stability_state == "Unavailable":
            continue
        counts[f.stability_state] = counts.get(f.stability_state, 0) + 1
    if not counts:
        return "—"
    return _map_stability_display(max(counts, key=counts.get))


def _format_joint_rom_summary(ba: BiomechanicalAnalysisResult) -> tuple[str, str, bool]:
    if ba.joint_rom is None or not ba.joint_rom.joints:
        return "Not available", "Joint ROM requires pose kinematics across the clip.", False
    lines: list[str] = []
    order = ("hip", "knee", "ankle")
    for joint_name in order:
        for side in ("left", "right"):
            entry = next(
                (j for j in ba.joint_rom.joints if j.joint == joint_name and j.side == side),
                None,
            )
            if entry is None or entry.rom_deg is None:
                continue
            tag = f"{side[0].upper()} {joint_name}"
            lines.append(
                f"{tag}: {entry.flexion_min_deg:.0f}°–{entry.flexion_max_deg:.0f}° "
                f"(ROM {entry.rom_deg:.0f}°)"
            )
    if not lines:
        reasons = [
            entry.note
            for entry in ba.joint_rom.joints
            if entry.note.startswith("N/A")
        ]
        return (
            "Not available",
            reasons[0] if reasons else "No usable complete-cycle joint trajectories for ROM summary.",
            False,
        )
    return "\n".join(lines), "Pose-derived flexion ROM — estimated, not goniometer measured.", True


def _tracking_confidence_field(ba: BiomechanicalAnalysisResult | None) -> SummaryField:
    if ba is None or ba.video_quality is None:
        return SummaryField(
            LABEL_TRACKING_CONFIDENCE,
            "Not available",
            TIER_DERIVED,
            False,
            "Pose tracking confidence requires a loaded pose sequence.",
        )
    checks = ba.video_quality.checks
    score = checks.get("pose_confidence_score")
    detected_pct = checks.get("detected_frame_pct")
    if score is not None:
        value = f"{float(score):.0f}/100"
        reason = (
            f"Mean landmark visibility score from pose estimation "
            f"({detected_pct:.0f}% detected frames)"
            if detected_pct is not None
            else "Mean landmark visibility score from pose estimation."
        )
        return SummaryField(LABEL_TRACKING_CONFIDENCE, value, TIER_DERIVED, True, reason)
    if detected_pct is not None:
        return SummaryField(
            LABEL_TRACKING_CONFIDENCE,
            f"{float(detected_pct):.0f}% frames detected",
            TIER_DERIVED,
            True,
            "Proxy from detected-frame ratio when landmark confidence is unavailable.",
        )
    return SummaryField(
        LABEL_TRACKING_CONFIDENCE,
        "Not available",
        TIER_DERIVED,
        False,
        "Pose confidence metrics were not recorded for this session.",
    )


def _field_from_evidence(label: str, evidence: Any) -> SummaryField:
    status = str(getattr(evidence, "status", "unavailable") or "unavailable")
    expl = str(getattr(evidence, "explanation", "") or "")
    extras = getattr(evidence, "extras", None) or {}
    coverage = getattr(evidence, "coverage_pct", None)
    n_detected = getattr(evidence, "n_detected", None)
    n_frames = getattr(evidence, "n_frames", None)
    n_usable = extras.get("n_usable")
    if extras.get("left_pct") is not None and extras.get("right_pct") is not None:
        value = (
            f"{status} · L {float(extras['left_pct']):.0f}% · "
            f"R {float(extras['right_pct']):.0f}% known"
        )
    elif n_usable is not None and n_detected is not None and n_frames:
        value = (
            f"{int(n_detected)} detected · {int(n_usable)} usable / {int(n_frames)}"
        )
    elif coverage is not None:
        value = f"{status} · {float(coverage):.0f}%"
    else:
        value = status
    available = status not in {"unavailable"}
    return SummaryField(label, value, TIER_DERIVED, available, expl)


def _per_foot_contact_field(label: str, evidence: Any, side: str) -> SummaryField:
    extras = getattr(evidence, "extras", None) or {}
    key = "left_pct" if side == "left" else "right_pct"
    pct = extras.get(key)
    status = str(getattr(evidence, "status", "unavailable") or "unavailable")
    if pct is None:
        return SummaryField(
            label,
            "Not available",
            TIER_DERIVED,
            False,
            "Per-foot contact coverage was not recorded.",
        )
    return SummaryField(
        label,
        f"{float(pct):.0f}% known · {status}",
        TIER_DERIVED,
        True,
        (
            f"{side.capitalize()} foot: {float(pct):.0f}% of analyzed frames have a known "
            f"contact state. Missing contact is not gait asymmetry."
        ),
    )


def _cycle_evidence_field(validity: ResultValidityReport) -> SummaryField:
    extras = getattr(validity.cycle_detection, "extras", None) or {}
    n_left = extras.get("n_left")
    n_right = extras.get("n_right")
    n_total = getattr(validity.cycle_detection, "n_cycles", None)
    expl = str(getattr(validity.cycle_detection, "explanation", "") or "")
    if n_left is None and n_right is None:
        if n_total is None:
            return SummaryField(
                "Cycle evidence",
                "Not available",
                TIER_DERIVED,
                False,
                expl or "Cycle counts were not recorded.",
            )
        return SummaryField(
            "Cycle evidence",
            f"{int(n_total)} cycle(s)",
            TIER_DERIVED,
            True,
            expl,
        )
    left_n = int(n_left or 0)
    right_n = int(n_right or 0)
    n_cand = extras.get("n_candidates")
    value = f"{left_n} L / {right_n} R accepted"
    if n_cand is not None:
        value = f"{value} · {int(n_cand)} candidates"
    return SummaryField(
        "Cycle evidence",
        value,
        TIER_DERIVED,
        True,
        expl or "Accepted per-foot event cycles vs raw HS–HS candidates. Unequal counts are not gait asymmetry.",
    )


def _attach_validity_fields(summary: AnalysisSummary, validity: ResultValidityReport) -> None:
    summary.result_validity = validity
    summary.pose_coverage = _field_from_evidence(LABEL_POSE_COVERAGE, validity.pose_coverage)
    summary.contact_coverage = _field_from_evidence(
        LABEL_CONTACT_COVERAGE, validity.contact_coverage
    )
    summary.contact_coverage_left = _per_foot_contact_field(
        "Left contact coverage", validity.contact_coverage, "left"
    )
    summary.contact_coverage_right = _per_foot_contact_field(
        "Right contact coverage", validity.contact_coverage, "right"
    )
    summary.cycle_evidence = _cycle_evidence_field(validity)
    summary.spectral_reliability = _field_from_evidence(
        LABEL_SPECTRAL_RELIABILITY, validity.spectral
    )
    summary.biomechanical_validity = _field_from_evidence(
        LABEL_BIOMECHANICAL_VALIDITY, validity.biomechanical
    )


def _apply_heuristic_withholds(
    summary: AnalysisSummary,
    validity: ResultValidityReport,
    gait_quality_value: str | None,
) -> None:
    """Withhold or mark gait-quality / laterality when required inputs are insufficient."""
    laterality_reason = laterality_withhold_reason(validity)
    quality_reason = gait_quality_withhold_reason(validity)
    if laterality_reason:
        summary.symmetry = SummaryField(
            LABEL_SYMMETRY,
            "Not available",
            TIER_DERIVED,
            False,
            laterality_reason,
        )
    if quality_reason:
        summary.overall_gait_quality = SummaryField(
            LABEL_GAIT_QUALITY,
            "Not available",
            TIER_DERIVED,
            False,
            quality_reason,
        )
    elif gait_quality_value and gait_quality_value != "Not available":
        existing = summary.overall_gait_quality
        reason = getattr(existing, "reason", "") or ""
        if "Unvalidated heuristic" not in reason:
            reason = (
                f"{reason} Unvalidated heuristic — not a validated gait-quality grade."
                if reason
                else "Unvalidated heuristic — not a validated gait-quality grade."
            )
        summary.overall_gait_quality = SummaryField(
            LABEL_GAIT_QUALITY,
            gait_quality_value,
            TIER_DERIVED,
            True,
            reason,
        )


def _pipeline_confidence_field(pipeline_status: dict[str, Any] | None) -> SummaryField:
    if not pipeline_status:
        return SummaryField(
            LABEL_PIPELINE_CONFIDENCE,
            "Not available",
            TIER_DERIVED,
            False,
            "Pipeline status has not been assessed for this session.",
        )
    diagram = pipeline_status.get("diagram") or []
    if not diagram:
        return SummaryField(
            LABEL_PIPELINE_CONFIDENCE,
            "Not available",
            TIER_DERIVED,
            False,
            "No pipeline stages were assessed for this session.",
        )
    scores = [_PIPELINE_STATUS_SCORE.get(stage.get("status", "unavailable"), 0.0) for stage in diagram]
    pct = float(np.mean(scores) * 100.0) if scores else 0.0
    completed = sum(1 for stage in diagram if stage.get("status") == "completed")
    if pct >= 75.0:
        level: ConfidenceLevel = "High"
    elif pct >= 45.0:
        level = "Moderate"
    else:
        level = "Low"
    return SummaryField(
        LABEL_PIPELINE_CONFIDENCE,
        f"{level} ({pct:.0f}%)",
        TIER_DERIVED,
        True,
        f"{completed}/{len(diagram)} monitored pipeline stages completed for this session.",
    )


def finalize_analysis_summary(
    summary: AnalysisSummary,
    *,
    pipeline_status: dict[str, Any] | None = None,
) -> AnalysisSummary:
    """Attach pipeline-derived confidence fields after pipeline assessment."""
    if pipeline_status is not None:
        summary.pipeline_status = pipeline_status
    summary.pipeline_confidence = _pipeline_confidence_field(summary.pipeline_status)
    return summary


def _interpret_com(ba: BiomechanicalAnalysisResult) -> tuple[str, str]:
    from stablewalk.config import DEFAULT_SUBJECT_HEIGHT_M

    com = ba.center_of_mass
    if com is None or com.positions is None or len(com.positions) < 3:
        return "Not available", "Insufficient COM trajectory for interpretation."
    y = com.positions[:, 1]
    rom_norm = float(np.nanmax(y) - np.nanmin(y))
    if not np.isfinite(rom_norm) or rom_norm < 1e-4:
        return "Minimal vertical COM excursion", "COM height varies little across the clip."
    rom_cm = rom_norm * float(DEFAULT_SUBJECT_HEIGHT_M) * 100.0
    # Typical walking vertical COM oscillation ≈ 2–5 cm (Winter / Perry).
    if rom_cm < 1.5:
        return "Reduced vertical oscillation", (
            f"Vertical COM range ≈ {rom_cm:.1f} cm (stature-scaled estimate)."
        )
    if rom_cm > 8.0:
        return "Elevated vertical COM variability", (
            f"Vertical COM range ≈ {rom_cm:.1f} cm (stature-scaled estimate)."
        )
    return "Normal oscillation", (
        f"Vertical COM range ≈ {rom_cm:.1f} cm (stature-scaled estimate)."
    )


def _interpret_vgrf(vgrf: EstimatedVGRFResult | None) -> tuple[str, str]:
    if vgrf is None or not vgrf.available or len(vgrf.timestamps) < 3:
        return "Not available", "Estimated Virtual GRF requires foot contact and pelvis kinematics."
    peak_bw = vgrf.metrics.peak_force_bw
    if peak_bw < 0.05:
        return "Flat estimated loading", "Low peak estimated vGRF — limited vertical acceleration signal."
    if 0.8 <= peak_bw <= 1.6:
        return "Normal estimated loading pattern", (
            f"Peak total ≈ {peak_bw:.2f} BW (estimated, not force-plate)."
        )
    if peak_bw > 1.6:
        return "High estimated loading peaks", f"Peak total ≈ {peak_bw:.2f} BW (estimated proxy)."
    return "Low estimated loading peaks", f"Peak total ≈ {peak_bw:.2f} BW (estimated proxy)."


def _gait_event_status(contact: FootContactAnalysisResult | None) -> list[GaitEventStatus]:
    if contact is None or not contact.per_frame:
        return [
            GaitEventStatus("Heel Strike", False, "No contact analysis"),
            GaitEventStatus("Toe Off", False, "No contact analysis"),
            GaitEventStatus("Double Support", False, "No contact analysis"),
            GaitEventStatus("Single Support", False, "No contact analysis"),
        ]
    hs = any(f.left_heel_strike or f.right_heel_strike for f in contact.per_frame)
    to = any(f.left_toe_off or f.right_toe_off for f in contact.per_frame)
    ds = any(f.macro_phase == "double_support" for f in contact.per_frame)
    ss = any(f.left_contact_binary + f.right_contact_binary == 1 for f in contact.per_frame)
    hs_n = sum(1 for f in contact.per_frame if f.left_heel_strike or f.right_heel_strike)
    to_n = sum(1 for f in contact.per_frame if f.left_toe_off or f.right_toe_off)
    ds_n = sum(1 for f in contact.per_frame if f.macro_phase == "double_support")
    ss_n = sum(1 for f in contact.per_frame if f.left_contact_binary + f.right_contact_binary == 1)
    return [
        GaitEventStatus("Heel Strike", hs, f"{hs_n} pulse(s) detected" if hs else "No heel-strike pulses"),
        GaitEventStatus("Toe Off", to, f"{to_n} pulse(s) detected" if to else "No toe-off pulses"),
        GaitEventStatus(
            "Double Support",
            ds,
            f"{ds_n} frame(s) with double support" if ds else "No double-support phase detected",
        ),
        GaitEventStatus(
            "Single Support",
            ss,
            f"{ss_n} frame(s) with single-limb support" if ss else "No single-support phase detected",
        ),
    ]


def _session_duration_s(cycles: GaitCycleAnalysisResult | None) -> float | None:
    if cycles is None:
        return None
    times: list[float] = []
    for event in getattr(cycles, "events", None) or ():
        t = getattr(event, "time_s", None)
        if t is not None:
            times.append(float(t))
    for cycle in getattr(cycles, "cycles", None) or ():
        times.append(float(getattr(cycle, "start_time_s", 0.0) or 0.0))
        times.append(float(getattr(cycle, "end_time_s", 0.0) or 0.0))
    if len(times) < 2:
        return None
    return max(times) - min(times)


def _analysis_confidence(
    ba: BiomechanicalAnalysisResult | None,
    cycles: GaitCycleAnalysisResult | None,
    contact: FootContactAnalysisResult | None,
) -> tuple[ConfidenceLevel, str, float]:
    """Measurement reliability. Pose / pipeline success are reported elsewhere."""
    if ba is None:
        return "Low", "No biomechanical analysis available.", 0.0
    measure: list[float] = []
    notes: list[str] = []
    n_cycles = 0
    duration = _session_duration_s(cycles)
    contact_tier = ""
    if contact is not None:
        cc = getattr(getattr(contact, "metrics", None), "contact_confidence", None)
        if cc is not None:
            measure.append(float(cc))
    n_left = n_right = 0
    if cycles is not None:
        n_cycles = int(getattr(cycles.metrics, "gait_cycle_count", 0) or 0)
        ccyc = getattr(cycles.metrics, "contact_confidence", None)
        if ccyc is not None:
            measure.append(float(ccyc))
        contact_tier = str(getattr(cycles.metrics, "confidence_tier", "") or "")
        for cycle in getattr(cycles, "cycles", None) or ():
            side = str(getattr(cycle, "boundary_side", "") or "").lower()
            if side.startswith("left"):
                n_left += 1
            elif side.startswith("right"):
                n_right += 1
    score = float(np.mean(measure)) if measure else 0.0

    if n_cycles <= 0:
        notes.append("no usable gait cycles")
    elif n_cycles < HIGH_MIN_CYCLES:
        notes.append(
            f"only {n_cycles} complete cycle(s); High needs >={HIGH_MIN_CYCLES}"
        )
    if n_left == 0 or n_right == 0:
        notes.append(
            f"event-based cycles {n_left} L / {n_right} R — unequal coverage "
            "is not a laterality finding"
        )
    if duration is not None and duration < HIGH_MIN_DURATION_S:
        notes.append(
            f"clip {duration:.1f} s is shorter than the {HIGH_MIN_DURATION_S:.0f} s High floor"
        )
    if contact_tier == "LOW_CONFIDENCE":
        notes.append("contact confidence tier is LOW")

    sm = getattr(ba, "stability_margin", None)
    if sm is not None and getattr(sm, "reportable", True) is False:
        notes.append("physical COM–BoS unavailable")

    can_high = (
        n_cycles >= HIGH_MIN_CYCLES
        and n_left >= 1
        and n_right >= 1
        and (duration is None or duration >= HIGH_MIN_DURATION_S)
        and contact_tier != "LOW_CONFIDENCE"
        and score >= HIGH_MIN_SCORE
    )
    if can_high:
        level: ConfidenceLevel = "High"
    elif n_cycles >= 1 and score >= 0.45:
        level = "Moderate"
    else:
        level = "Low"

    tracking = None
    if ba.video_quality is not None:
        tracking = ba.video_quality.overall_quality_score
    detail = (
        f"Legacy composite {level} "
        f"(score {score:.0%}, {n_cycles} cycle(s), {n_left} L / {n_right} R"
        + (f", {duration:.1f} s" if duration is not None else "")
        + "). Results display pose coverage, contact coverage, spectral "
        "reliability, and biomechanical validity separately. "
        + HIGH_CONFIDENCE_DEFINITION
    )
    if tracking is not None:
        detail += f" Pose tracking (separate): {tracking:.0f}/100."
    if notes:
        detail += " " + "; ".join(notes) + "."
    return level, detail, score


def build_analysis_summary(
    *,
    source: str = "",
    biomechanical: BiomechanicalAnalysisResult | None,
    estimated_vgrf: EstimatedVGRFResult | None,
    contact: FootContactAnalysisResult | None,
    cycles: GaitCycleAnalysisResult | None,
    frame_index: int | None = None,
    timestamp_s: float | None = None,
    pipeline_status: dict[str, Any] | None = None,
    spectral: Any | None = None,
) -> AnalysisSummary:
    """Assemble summary fields from the current analysis bundle."""
    summary = AnalysisSummary(
        source=source,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
    )

    if biomechanical is None:
        na = SummaryField("", "Not available", TIER_DERIVED, False, "Analyze a walking video first.")
        summary.overall_gait_quality = na
        summary.cadence = na
        summary.walking_speed = na
        summary.symmetry = na
        summary.stability_margin = na
        summary.center_of_mass = SummaryField("", "Not available", TIER_ESTIMATED, False, na.reason)
        summary.estimated_virtual_grf = SummaryField(
            "", "Not available", TIER_ESTIMATED, False, na.reason
        )
        summary.joint_rom_summary = SummaryField("", "Not available", TIER_ESTIMATED, False, na.reason)
        summary.gait_events = _gait_event_status(None)
        summary.video_quality = na
        summary.tracking_confidence = _tracking_confidence_field(None)
        validity = build_result_validity(
            biomechanical=None,
            contact=contact,
            cycles=cycles,
            spectral=spectral,
            duration_s=_session_duration_s(cycles),
        )
        _attach_validity_fields(summary, validity)
        conf_level, conf_detail, _ = _analysis_confidence(None, cycles, contact)
        summary.analysis_confidence = SummaryField(
            LABEL_ANALYSIS_CONFIDENCE, conf_level, TIER_DERIVED, False, conf_detail
        )
        from stablewalk.analysis.scientific_interpretation import build_scientific_interpretation

        summary.scientific_interpretation = build_scientific_interpretation(
            summary,
            biomechanical=None,
        )
        return finalize_analysis_summary(summary, pipeline_status=pipeline_status)

    ba = biomechanical
    gm = ba.gait_metrics

    gq = ba.gait_quality
    if (
        gq is not None
        and getattr(gq, "available", True)
        and gq.score is not None
    ):
        summary.overall_gait_quality = SummaryField(
            LABEL_GAIT_QUALITY,
            f"{gq.score:.0f}/100",
            TIER_DERIVED,
            True,
            gq.explanation[:160],
        )
    else:
        reason = (
            gq.explanation
            if gq is not None and gq.explanation
            else "Gait quality score was not computed."
        )
        summary.overall_gait_quality = SummaryField(
            LABEL_GAIT_QUALITY,
            "Not available",
            TIER_DERIVED,
            False,
            reason,
        )

    cadence_spm, _cadence_confidence = _extract_cadence(ba, cycles)
    if cadence_spm is not None:
        cadence_note = (
            gm.cadence.note
            if gm and gm.cadence and gm.cadence.value is not None
            else "From detected gait-cycle timing"
        )
        summary.cadence = SummaryField(
            LABEL_CADENCE,
            format_cadence_value(cadence_spm),
            TIER_CALCULATED,
            True,
            cadence_note or "From heel-strike intervals",
        )
    else:
        from stablewalk.analysis.gait_cycle_analysis import cadence_is_exploratory

        metrics = getattr(cycles, "metrics", None) if cycles is not None else None
        if cadence_is_exploratory(metrics, cycles):
            cadence_reason = (
                "Exploratory — withheld from summary (low-confidence or "
                "single-channel cadence)."
            )
            if metrics is not None and getattr(metrics, "reliability_reason", ""):
                cadence_reason = f"{cadence_reason} {metrics.reliability_reason}"
        else:
            cadence_reason = (
                gm.cadence.note
                if gm and gm.cadence and gm.cadence.note
                else "Insufficient complete heel-strike cycles for cadence."
            )
        summary.cadence = SummaryField(
            LABEL_CADENCE,
            "Not available",
            TIER_CALCULATED,
            False,
            cadence_reason,
        )

    if gm and is_reportable_walking_speed(gm.walking_speed):
        ws = gm.walking_speed
        assert ws is not None and ws.value is not None
        summary.walking_speed = SummaryField(
            LABEL_WALKING_SPEED,
            format_walking_speed_display(ws),
            TIER_ESTIMATED,
            True,
            ws.note or f"Scaled to {DEFAULT_SUBJECT_HEIGHT_M:.2f} m stature; monocular estimate",
        )
    else:
        speed_reason = (
            gm.walking_speed.note
            if gm and gm.walking_speed and gm.walking_speed.note
            else "Walking speed could not be estimated reliably from this clip."
        )
        summary.walking_speed = SummaryField(
            LABEL_WALKING_SPEED,
            UNAVAILABLE_LABEL,
            TIER_ESTIMATED,
            False,
            speed_reason,
        )

    if ba.symmetry and ba.symmetry.overall_symmetry_pct and ba.symmetry.overall_symmetry_pct.value is not None:
        summary.symmetry = SummaryField(
            "Symmetry",
            f"{ba.symmetry.overall_symmetry_pct.value:.0f}%",
            TIER_DERIVED,
            True,
            ba.symmetry.overall_symmetry_pct.note or "",
        )
    else:
        summary.symmetry = SummaryField(
            "Symmetry",
            "Not available",
            TIER_DERIVED,
            False,
            "Insufficient bilateral metrics for symmetry index.",
        )

    sm = ba.stability_margin
    reportable_stab = bool(getattr(sm, "reportable", True)) if sm is not None else False
    if reportable_stab and sm is not None and sm.stable_pct is not None:
        dominant = _dominant_stability_state(ba)
        if frame_index is not None:
            for f in sm.per_frame:
                if f.frame_index == frame_index:
                    dominant = _map_stability_display(f.stability_state)
                    break
        summary.stability_margin = SummaryField(
            LABEL_STABILITY_MARGIN,
            f"{dominant} ({sm.stable_pct:.0f}% stable frames)",
            TIER_DERIVED,
            True,
            "COM–base-of-support margin classification",
        )
    else:
        if sm is not None and getattr(sm, "reportable", True) is False:
            stability_reason = (
                getattr(sm, "experimental_note", None)
                or "Monocular COM–BoS is experimental and is not a physical stability score."
            )
        else:
            stability_reason = (
                sm.warnings[0]
                if sm is not None and getattr(sm, "warnings", None)
                else "Stability margin requires reliable COM and support polygon data."
            )
        summary.stability_margin = SummaryField(
            LABEL_STABILITY_MARGIN,
            "Not available",
            TIER_DERIVED,
            False,
            str(stability_reason),
        )

    com_text, com_reason = _interpret_com(ba)
    summary.center_of_mass = SummaryField(
        LABEL_COM_FULL,
        com_text,
        TIER_ESTIMATED,
        com_text != "Not available",
        com_reason,
    )

    vgrf_text, vgrf_reason = _interpret_vgrf(estimated_vgrf)
    summary.estimated_virtual_grf = SummaryField(
        LABEL_VIRTUAL_GRF_PANEL,
        vgrf_text,
        TIER_ESTIMATED,
        vgrf_text != "Not available",
        vgrf_reason,
    )

    rom_text, rom_reason, rom_ok = _format_joint_rom_summary(ba)
    summary.joint_rom_summary = SummaryField(
        LABEL_JOINT_ROM_SUMMARY,
        rom_text,
        TIER_ESTIMATED,
        rom_ok,
        rom_reason,
    )

    summary.gait_events = _gait_event_status(contact)

    if ba.video_quality is not None:
        checks = getattr(ba.video_quality, "checks", None) or {}
        cov = checks.get("pose_coverage_pct")
        vis = checks.get("mean_landmark_visibility")
        if cov is not None and vis is not None:
            vq_text = f"Coverage {float(cov):.0f}% · visibility {float(vis):.0%}"
        else:
            vq_text = f"{ba.video_quality.overall_quality_score:.0f}/100"
        summary.video_quality = SummaryField(
            LABEL_VIDEO_QUALITY,
            vq_text,
            TIER_DERIVED,
            True,
            (ba.video_quality.summary_explanation or "")[:240],
        )
    else:
        summary.video_quality = SummaryField(
            LABEL_VIDEO_QUALITY,
            "Not available",
            TIER_DERIVED,
            False,
            "No pose sequence for video quality assessment.",
        )

    summary.tracking_confidence = _tracking_confidence_field(ba)

    validity = build_result_validity(
        biomechanical=ba,
        contact=contact,
        cycles=cycles,
        spectral=spectral,
        duration_s=_session_duration_s(cycles),
        n_pose_frames=int((ba.video_quality.checks or {}).get("n_frames") or 0)
        if ba.video_quality is not None
        else 0,
        coordinate_system=str(
            getattr(getattr(ba, "center_of_mass", None), "coordinate_frame", "") or ""
        ),
    )
    _attach_validity_fields(summary, validity)
    gq_value = (
        summary.overall_gait_quality.value if summary.overall_gait_quality is not None else None
    )
    _apply_heuristic_withholds(summary, validity, gq_value)

    conf_level, conf_detail, _ = _analysis_confidence(ba, cycles, contact)
    summary.analysis_confidence = SummaryField(
        LABEL_ANALYSIS_CONFIDENCE,
        conf_level,
        TIER_DERIVED,
        True,
        conf_detail,
    )

    from stablewalk.analysis.scientific_interpretation import build_scientific_interpretation

    summary.scientific_interpretation = build_scientific_interpretation(
        summary,
        biomechanical=ba,
    )
    return finalize_analysis_summary(summary, pipeline_status=pipeline_status)


__all__ = [
    "AnalysisSummary",
    "GaitEventStatus",
    "SummaryField",
    "build_analysis_summary",
    "finalize_analysis_summary",
]
