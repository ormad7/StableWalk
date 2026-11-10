"""Per-metric Results validity — not a single global Analysis Confidence.

Pose coverage, per-foot contact coverage, spectral reliability, and
biomechanical validity are separate layers. A high pose-detection rate must
not be shown as High analysis confidence when spectral evidence is weak,
event cycles are unilateral, or physical stability is unreportable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from stablewalk.analysis.quality_layers import (
    LAYER_BIOMECHANICAL,
    LAYER_CONTACT_COVERAGE,
    LAYER_POSE_COVERAGE,
    LAYER_SPECTRAL,
    QualityEvidence,
    UNEQUAL_CONTACT_COVERAGE_NOTE,
    bilateral_contact_coverage_evidence,
    biomechanical_validity_evidence,
    contact_coverage_evidence,
    cycle_detection_evidence,
    pose_coverage_evidence,
    spectral_reliability_evidence,
)
from stablewalk.analysis.foot_contact_analysis import CONTACT_UNKNOWN_REASON_LABELS

_STATUS_RANK = {
    "computed": 0,
    "limited": 1,
    "insufficient": 2,
    "unavailable": 3,
}


def _cycle_side_counts(cycles: Any) -> tuple[int, int, int]:
    clist = list(getattr(cycles, "cycles", None) or [])
    n_left = sum(1 for c in clist if str(getattr(c, "boundary_side", "")).lower().startswith("left"))
    n_right = sum(1 for c in clist if str(getattr(c, "boundary_side", "")).lower().startswith("right"))
    return len(clist), n_left, n_right


def _cycle_candidate_counts(cycles: Any) -> tuple[int | None, int]:
    cands = list(getattr(cycles, "cycle_candidates", None) or [])
    n_accepted = sum(1 for c in cands if bool(getattr(c, "accepted", False)))
    if not cands:
        return None, _cycle_side_counts(cycles)[0]
    return len(cands), n_accepted


def _dominant_unknown_reason(contact: Any, side: str) -> str:
    key = f"{side}_unknown_reason"
    counts: Counter[str] = Counter()
    for frame in getattr(contact, "per_frame", None) or ():
        code = str(getattr(frame, key, "") or "")
        if code:
            counts[code] += 1
    if not counts:
        return ""
    code, n = counts.most_common(1)[0]
    label = CONTACT_UNKNOWN_REASON_LABELS.get(code, code)
    return f"{code} ({n} frames) — {label}"


def _pose_from_video_quality(ba: Any, n_frames_fallback: int = 0) -> QualityEvidence:
    vq = getattr(ba, "video_quality", None) if ba is not None else None
    stored = getattr(vq, "pose_coverage", None) if vq is not None else None
    if stored is not None and hasattr(stored, "status"):
        return stored
    checks = getattr(vq, "checks", None) or {}
    n_frames = int(checks.get("n_frames") or n_frames_fallback or 0)
    n_detected = int(checks.get("detected_frames") or 0)
    if n_frames <= 0:
        cov = checks.get("pose_coverage_pct") or checks.get("detected_frame_pct")
        if cov is not None:
            n_frames = 100
            n_detected = int(round(float(cov)))
    return pose_coverage_evidence(n_frames=n_frames, n_detected=n_detected)


def _contact_layers(contact: Any) -> tuple[QualityEvidence, QualityEvidence, QualityEvidence]:
    if contact is None or not getattr(contact, "per_frame", None):
        empty = contact_coverage_evidence(side="left", n_frames=0, n_known=0)
        empty_r = contact_coverage_evidence(side="right", n_frames=0, n_known=0)
        return empty, empty_r, bilateral_contact_coverage_evidence(empty, empty_r)
    n = len(contact.per_frame)
    left_known = contact.left_contact_known_mask
    right_known = contact.right_contact_known_mask
    left_bin = contact.left_contact_binary.astype(bool)
    right_bin = contact.right_contact_binary.astype(bool)
    left = contact_coverage_evidence(
        side="Left",
        n_frames=n,
        n_known=int(left_known.sum()),
        n_contact_on=int((left_known & left_bin).sum()),
        n_confirmed_swing=int((left_known & ~left_bin).sum()),
        dominant_unknown_reason=_dominant_unknown_reason(contact, "left"),
    )
    right = contact_coverage_evidence(
        side="Right",
        n_frames=n,
        n_known=int(right_known.sum()),
        n_contact_on=int((right_known & right_bin).sum()),
        n_confirmed_swing=int((right_known & ~right_bin).sum()),
        dominant_unknown_reason=_dominant_unknown_reason(contact, "right"),
    )
    return left, right, bilateral_contact_coverage_evidence(left, right)


def _acf_stride_peak_value(ac: Any) -> float | None:
    period = getattr(ac, "stride_period_s", None)
    if period is None:
        return None
    for peak in getattr(ac, "peaks", None) or ():
        try:
            if abs(float(peak.lag_s) - float(period)) < 1e-9:
                return float(peak.value)
        except (TypeError, ValueError, AttributeError):
            continue
    return None


def _acf_gap_fraction(spectral: Any, channel_id: str | None, ac: Any) -> float | None:
    gap = getattr(ac, "gap_fraction", None)
    if gap is not None:
        try:
            return float(gap)
        except (TypeError, ValueError):
            pass
    bundle = getattr(spectral, "bundle", None)
    channels = getattr(bundle, "channels", None) or {}
    if not isinstance(channels, dict) or not channels:
        return None
    ch = channels.get(channel_id) if channel_id else None
    if ch is None:
        ch = next(iter(channels.values()), None)
    miss = getattr(ch, "missing_fraction", None) if ch is not None else None
    return None if miss is None else float(miss)


def _cap_spectral_status(status: str, *, label: str, insufficient_reason: Any) -> str:
    """Clip duration must not outrank a low ACF reliability label."""
    folded = str(label or "").strip().lower()
    if insufficient_reason or folded == "insufficient":
        return "insufficient"
    if folded == "low" and _STATUS_RANK.get(status, 0) < _STATUS_RANK["limited"]:
        return "limited"
    return status


def _spectral_layer(spectral: Any, cycles: Any, duration_s: float | None) -> QualityEvidence:
    ac = None
    primary = None
    if spectral is not None:
        mapping = getattr(spectral, "autocorr", None) or {}
        if mapping:
            primary = getattr(spectral, "univariate_channel_id", None)
            ac = mapping.get(primary) if primary else None
            if ac is None:
                ac = next(iter(mapping.values()), None)
    if ac is not None:
        label = str(getattr(ac, "reliability_label", "") or "")
        insufficient = getattr(ac, "insufficient_reason", None)
        evidence = spectral_reliability_evidence(
            duration_s=duration_s,
            gap_fraction=_acf_gap_fraction(spectral, primary, ac),
            n_cycles=getattr(ac, "n_stride_cycles", None),
            peak_value=_acf_stride_peak_value(ac),
            insufficient_reason=insufficient,
        )
        evidence.status = _cap_spectral_status(
            evidence.status, label=label, insufficient_reason=insufficient
        )
        expl = str(getattr(ac, "reliability_explanation", "") or "")
        if expl:
            evidence.explanation = expl
        if label and label.lower() not in evidence.explanation.lower():
            score = getattr(ac, "reliability", None)
            score_txt = f" (score {float(score):.2f})" if score is not None else ""
            evidence.explanation = (
                f"{evidence.explanation.rstrip('.')} "
                f"— ACF reliability is {label}{score_txt}."
            )
        evidence.extras["reliability_label"] = label
        if getattr(ac, "reliability", None) is not None:
            evidence.extras["reliability"] = float(ac.reliability)
        return evidence
    total, _n_left, _n_right = _cycle_side_counts(cycles)
    return spectral_reliability_evidence(
        duration_s=duration_s,
        gap_fraction=None,
        n_cycles=float(total) if total else None,
        insufficient_reason=(
            None
            if duration_s is not None
            else "Spectral analysis has not been attached to this Results snapshot."
        ),
    )


def _biomech_layer(ba: Any, recording_coord: str = "") -> QualityEvidence:
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    reportable = getattr(sm, "reportable", None) if sm is not None else False
    reason = ""
    if sm is not None:
        reason = str(getattr(sm, "experimental_note", "") or getattr(sm, "reason", "") or "")
    calibrated = "calibrated" in str(recording_coord or "").lower()
    return biomechanical_validity_evidence(
        calibrated_metric=calibrated,
        stability_reportable=reportable,
        reason=reason,
    )


@dataclass
class ResultValidityReport:
    """Four independent Results validity layers plus cycle evidence."""

    pose_coverage: QualityEvidence
    contact_left: QualityEvidence
    contact_right: QualityEvidence
    contact_coverage: QualityEvidence
    spectral: QualityEvidence
    biomechanical: QualityEvidence
    cycle_detection: QualityEvidence
    limitations: list[str] = field(default_factory=list)

    def layers(self) -> list[QualityEvidence]:
        return [
            self.pose_coverage,
            self.contact_coverage,
            self.spectral,
            self.biomechanical,
        ]

    def worst_status(self) -> str:
        return max(self.layers(), key=lambda e: _STATUS_RANK.get(e.status, 0)).status

    def headline_lines(self) -> list[str]:
        n_left = int(self.cycle_detection.extras.get("n_left") or 0)
        n_right = int(self.cycle_detection.extras.get("n_right") or 0)
        contact = (
            f"Contact coverage: {self.contact_coverage.status} — "
            f"L {self.contact_left.coverage_pct or 0:.0f}% known · "
            f"R {self.contact_right.coverage_pct or 0:.0f}% known"
        )
        if n_left != n_right:
            contact += f" · event cycles {n_left} L / {n_right} R"
        return [
            f"Pose coverage: {self.pose_coverage.status} — {self.pose_coverage.explanation}",
            contact + f". {UNEQUAL_CONTACT_COVERAGE_NOTE}",
            f"Spectral reliability: {self.spectral.status} — {self.spectral.explanation}",
            f"Biomechanical validity: {self.biomechanical.status} — {self.biomechanical.explanation}",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pose_coverage": self.pose_coverage.to_dict(),
            "contact_left": self.contact_left.to_dict(),
            "contact_right": self.contact_right.to_dict(),
            "contact_coverage": self.contact_coverage.to_dict(),
            "spectral": self.spectral.to_dict(),
            "biomechanical": self.biomechanical.to_dict(),
            "cycle_detection": self.cycle_detection.to_dict(),
            "limitations": list(self.limitations),
            "headline": self.headline_lines(),
        }


def build_result_validity(
    *,
    biomechanical: Any = None,
    contact: Any = None,
    cycles: Any = None,
    spectral: Any = None,
    duration_s: float | None = None,
    n_pose_frames: int = 0,
    coordinate_system: str = "",
) -> ResultValidityReport:
    pose = _pose_from_video_quality(biomechanical, n_pose_frames)
    left, right, combined = _contact_layers(contact)
    total, n_left, n_right = _cycle_side_counts(cycles)
    n_candidates, n_accepted = _cycle_candidate_counts(cycles)
    metrics = getattr(cycles, "metrics", None)
    cycle_ev = cycle_detection_evidence(
        n_cycles=total,
        n_left=n_left,
        n_right=n_right,
        usable_duration_s=duration_s,
        contact_confidence=(
            getattr(metrics, "contact_confidence", None) if metrics is not None else None
        ),
        reason=str(getattr(metrics, "reliability_reason", "") or "") if metrics else "",
        n_candidates=n_candidates,
        n_accepted=n_accepted if n_candidates is not None else total,
    )
    spectral_ev = _spectral_layer(spectral, cycles, duration_s)
    biomech_ev = _biomech_layer(biomechanical, coordinate_system)
    cand_txt = (
        f"{n_candidates} candidates, {n_accepted} accepted"
        if n_candidates is not None
        else f"{total} accepted"
    )
    limitations = [
        pose.explanation,
        combined.explanation,
        f"Event-based cycles: {n_left} left / {n_right} right accepted ({cand_txt}). "
        "Unilateral cycle counts are not a laterality score.",
        spectral_ev.explanation,
        biomech_ev.explanation,
        UNEQUAL_CONTACT_COVERAGE_NOTE,
    ]
    return ResultValidityReport(
        pose_coverage=pose,
        contact_left=left,
        contact_right=right,
        contact_coverage=combined,
        spectral=spectral_ev,
        biomechanical=biomech_ev,
        cycle_detection=cycle_ev,
        limitations=limitations,
    )


def contact_coverage_blocks_laterality(report: ResultValidityReport | None) -> bool:
    if report is None:
        return False
    extras = report.contact_coverage.extras or {}
    if extras.get("unequal"):
        return True
    n_left = int(report.cycle_detection.extras.get("n_left") or 0)
    n_right = int(report.cycle_detection.extras.get("n_right") or 0)
    return (n_left == 0) ^ (n_right == 0)


def quality_evidence_from_dict(payload: dict[str, Any]) -> QualityEvidence:
    extras = dict(payload.get("extras") or {})
    return QualityEvidence(
        layer=payload.get("layer") or LAYER_POSE_COVERAGE,
        status=payload.get("status") or "unavailable",
        explanation=str(payload.get("explanation") or ""),
        coverage_pct=payload.get("coverage_pct"),
        mean_visibility=payload.get("mean_visibility"),
        usable_duration_s=payload.get("usable_duration_s"),
        gap_fraction=payload.get("gap_fraction"),
        n_frames=int(payload.get("n_frames") or 0),
        n_detected=int(payload.get("n_detected") or 0),
        n_cycles=int(payload.get("n_cycles") or 0),
        formula=str(payload.get("formula") or ""),
        extras=extras,
    )


def result_validity_from_export_dict(payload: dict[str, Any]) -> ResultValidityReport:
    """Rebuild a validity report from a saved validation JSON (no new analysis)."""
    block = payload.get("validity") if isinstance(payload.get("validity"), dict) else payload
    required = (
        "pose_coverage",
        "contact_left",
        "contact_right",
        "contact_coverage",
        "spectral",
        "biomechanical",
        "cycle_detection",
    )
    layers = {key: quality_evidence_from_dict(block.get(key) or {}) for key in required}
    return ResultValidityReport(
        **layers,
        limitations=list(block.get("limitations") or []),
    )


def laterality_withhold_reason(report: ResultValidityReport | None) -> str:
    """Why a left–right score must not be shown as gait asymmetry."""
    if report is None:
        return ""
    extras = report.contact_coverage.extras or {}
    left_pct = float(extras.get("left_pct") or 0.0)
    right_pct = float(extras.get("right_pct") or 0.0)
    n_left = int(report.cycle_detection.extras.get("n_left") or 0)
    n_right = int(report.cycle_detection.extras.get("n_right") or 0)
    if extras.get("unequal") or contact_coverage_blocks_laterality(report):
        return (
            f"Withheld: L {left_pct:.0f}% / R {right_pct:.0f}% contact known, "
            f"event cycles {n_left} L / {n_right} R. "
            "Unequal missing-data coverage is not a gait-laterality finding."
        )
    if report.contact_coverage.status == "insufficient":
        return (
            f"Withheld: contact coverage is insufficient "
            f"(L {left_pct:.0f}% / R {right_pct:.0f}% known). "
            "A symmetry percentage is not validated from these inputs."
        )
    return ""


def gait_quality_withhold_reason(report: ResultValidityReport | None) -> str:
    """Why the heuristic gait-quality score must not be shown as a quality band."""
    if report is None:
        return ""
    laterality = laterality_withhold_reason(report)
    if laterality:
        return (
            laterality.replace("A symmetry percentage", "The gait-quality heuristic", 1)
            if "A symmetry percentage" in laterality
            else laterality
        )
    spectral_ran = bool(
        report.spectral.usable_duration_s is not None
        or report.spectral.extras.get("reliability") is not None
        or report.spectral.extras.get("reliability_label")
        or int(report.spectral.n_cycles or 0) > 0
    )
    if report.spectral.status == "insufficient" and spectral_ran:
        return (
            "Withheld: spectral reliability is insufficient. "
            "The gait-quality heuristic is unvalidated without usable spectral "
            "or bilateral contact/cycle evidence. Independent of source labels."
        )
    if report.biomechanical.status == "unavailable" and report.contact_coverage.status in (
        "insufficient",
        "limited",
    ):
        return (
            "Withheld: physical stability is unreportable and contact coverage "
            f"is {report.contact_coverage.status}. "
            "The gait-quality heuristic is not a validated instrument."
        )
    return ""


def withhold_laterality_heuristic(report: ResultValidityReport | None) -> bool:
    return bool(laterality_withhold_reason(report))


def withhold_gait_quality_heuristic(report: ResultValidityReport | None) -> bool:
    return bool(gait_quality_withhold_reason(report))


__all__ = [
    "ResultValidityReport",
    "build_result_validity",
    "contact_coverage_blocks_laterality",
    "gait_quality_withhold_reason",
    "laterality_withhold_reason",
    "quality_evidence_from_dict",
    "result_validity_from_export_dict",
    "withhold_gait_quality_heuristic",
    "withhold_laterality_heuristic",
]
