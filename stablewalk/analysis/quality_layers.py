"""Canonical quality and confidence layers for StableWalk.

These five layers are **not interchangeable**. A high pose-coverage or
landmark-visibility figure must not be shown as spectral quality, cycle
reliability, or biomechanical validity.

Layers
------
``pose_coverage``
    Fraction of frames with a detected pose (presence only).
``landmark_confidence``
    Detector visibility / score. **Detected landmarks are not accurate tracking.**
``cycle_detection``
    Heel-strike cycle evidence: contributing cycles, usable duration, contact.
``spectral``
    Series duration, gap fraction, contributing cycles, and ACF/PSD peak quality.
``biomechanical``
    Whether a physical interpretation is supported (calibrated metric, force
    plates, reportable COM–BoS, …). Unavailable physical stability stays
    unavailable.

Status vocabulary
-----------------
Do **not** use ``reliable`` or ``acceptable`` as quality labels.

* ``computed`` — enough evidence to report the number
* ``limited`` — reported with reduced evidence (gaps, few cycles, low visibility)
* ``insufficient`` — inputs exist but do not support the metric
* ``unavailable`` — a required physical / calibrated input is absent

The composite gait-quality score is a **heuristic descriptor**, not a
validated clinical instrument. Missing components are omitted and weights
are renormalized, so a high number can mean “few inputs were available.”
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QualityLayer = Literal[
    "pose_coverage",
    "landmark_confidence",
    "contact_coverage",
    "cycle_detection",
    "spectral",
    "biomechanical",
]
QualityStatus = Literal["computed", "limited", "insufficient", "unavailable"]

LAYER_POSE_COVERAGE: QualityLayer = "pose_coverage"
LAYER_LANDMARK_CONFIDENCE: QualityLayer = "landmark_confidence"
LAYER_CONTACT_COVERAGE: QualityLayer = "contact_coverage"
LAYER_CYCLE_DETECTION: QualityLayer = "cycle_detection"
LAYER_SPECTRAL: QualityLayer = "spectral"
LAYER_BIOMECHANICAL: QualityLayer = "biomechanical"

UNEQUAL_CONTACT_COVERAGE_NOTE = (
    "Unequal missing-data coverage is not a gait-laterality finding."
)

LANDMARK_NOT_TRACKING = (
    "Detected landmarks and visibility scores are not a claim of accurate tracking."
)
FORBIDDEN_QUALITY_WORDS = ("reliable", "acceptable")

GAIT_QUALITY_KIND = "heuristic"
GAIT_QUALITY_INTENDED_COMPONENTS = (
    "stability",
    "stability_margin",
    "symmetry",
    "cadence_consistency",
    "cycle_consistency",
    "contact_reliability",
    "motion_quality",
)
GAIT_QUALITY_WEIGHTS: dict[str, float] = {
    "stability": 0.22,
    "stability_margin": 0.15,
    "symmetry": 0.20,
    "cadence_consistency": 0.12,
    "cycle_consistency": 0.12,
    "contact_reliability": 0.10,
    "motion_quality": 0.09,
}
GAIT_QUALITY_FORMULA = (
    "score = Σ(component_k · weight_k) / Σ(weight_k) for active k; "
    "weights: stability 0.22, stability_margin 0.15, symmetry 0.20, "
    "cadence_consistency 0.12, cycle_consistency 0.12, "
    "contact_reliability 0.10, motion_quality 0.09. "
    "Missing components are omitted and weights renormalized."
)
GAIT_QUALITY_LIMITATIONS = (
    "Heuristic session descriptor, not a validated gait-quality instrument. "
    "It mixes pose-derived proxies and is not interchangeable with pose "
    "coverage, landmark visibility, cycle-detection confidence, spectral "
    "reliability, or physical stability. Unreportable COM–BoS / force-plate "
    "stability is omitted and must stay unavailable."
)
GAIT_QUALITY_THRESHOLD_SOURCE = (
    "Heuristic session bands (≥60 higher band, ≥45 mid band). "
    "Not a clinical normality cut-off. Independent of source video labels "
    "(Normal / Abnormal / Fast Walking)."
)

_MIN_CYCLES_COMPUTED = 2
_MIN_DURATION_S = 1.0
_MAX_GAP_COMPUTED = 0.15
_MAX_GAP_LIMITED = 0.35


@dataclass
class QualityEvidence:
    layer: QualityLayer
    status: QualityStatus
    explanation: str
    coverage_pct: float | None = None
    mean_visibility: float | None = None
    usable_duration_s: float | None = None
    gap_fraction: float | None = None
    n_frames: int = 0
    n_detected: int = 0
    n_cycles: int = 0
    formula: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.status} — {self.explanation}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status,
            "explanation": self.explanation,
            "label": self.label(),
            "coverage_pct": self.coverage_pct,
            "mean_visibility": self.mean_visibility,
            "usable_duration_s": self.usable_duration_s,
            "gap_fraction": self.gap_fraction,
            "n_frames": self.n_frames,
            "n_detected": self.n_detected,
            "n_cycles": self.n_cycles,
            "formula": self.formula,
            "note": (
                LANDMARK_NOT_TRACKING
                if self.layer in {LAYER_POSE_COVERAGE, LAYER_LANDMARK_CONFIDENCE}
                else UNEQUAL_CONTACT_COVERAGE_NOTE
                if self.layer == LAYER_CONTACT_COVERAGE
                else ""
            ),
            "extras": dict(self.extras),
        }


def status_from_evidence(
    *,
    have_inputs: bool,
    n_cycles: int = 0,
    usable_duration_s: float | None = None,
    gap_fraction: float | None = None,
    min_cycles: int = _MIN_CYCLES_COMPUTED,
    min_duration_s: float = _MIN_DURATION_S,
    required_physical: bool = False,
    physical_available: bool = True,
) -> QualityStatus:
    """Map duration / gaps / cycles / physical support to a status."""
    if required_physical and not physical_available:
        return "unavailable"
    if not have_inputs:
        return "unavailable"
    duration_ok = usable_duration_s is None or float(usable_duration_s) >= min_duration_s
    duration_weak = usable_duration_s is not None and float(usable_duration_s) < min_duration_s
    gap = 0.0 if gap_fraction is None else float(gap_fraction)
    if n_cycles < 1 or (usable_duration_s is not None and float(usable_duration_s) <= 0.0):
        return "insufficient"
    if (
        n_cycles >= min_cycles
        and duration_ok
        and gap <= _MAX_GAP_COMPUTED
    ):
        return "computed"
    if n_cycles >= 1 and not duration_weak and gap <= _MAX_GAP_LIMITED:
        return "limited"
    return "insufficient"


def pose_coverage_evidence(
    *,
    n_frames: int,
    n_detected: int,
    usable_duration_s: float | None = None,
    n_usable: int | None = None,
) -> QualityEvidence:
    if n_frames <= 0:
        return QualityEvidence(
            layer=LAYER_POSE_COVERAGE,
            status="unavailable",
            explanation="No frames were available to measure pose detection coverage.",
            n_frames=0,
            formula="detected = identity-present frames / n_frames; usable = gait-ready frames",
        )
    coverage = 100.0 * n_detected / n_frames
    if n_detected <= 0:
        status: QualityStatus = "insufficient"
        expl = f"0 detected / {n_frames} frames."
    elif coverage >= 80.0:
        status = "computed"
        expl = f"{n_detected} detected / {n_frames} frames ({coverage:.0f}% detection)."
    elif coverage >= 40.0:
        status = "limited"
        expl = (
            f"{n_detected} detected / {n_frames} frames ({coverage:.0f}% detection). "
            "Gaps reduce usable duration."
        )
    else:
        status = "insufficient"
        expl = (
            f"{n_detected} detected / {n_frames} frames "
            f"({coverage:.0f}% detection)."
        )
    if n_usable is not None:
        expl = f"{expl} {int(n_usable)} usable for gait."
    expl = f"{expl} {LANDMARK_NOT_TRACKING}"
    extras: dict[str, Any] = {}
    if n_usable is not None:
        extras["n_usable"] = int(n_usable)
    return QualityEvidence(
        layer=LAYER_POSE_COVERAGE,
        status=status,
        explanation=expl,
        coverage_pct=coverage,
        usable_duration_s=usable_duration_s,
        n_frames=n_frames,
        n_detected=n_detected,
        extras=extras,
        formula="detected = identity-present frames / n_frames; usable = gait-ready frames",
    )


def landmark_confidence_evidence(
    *,
    mean_visibility: float | None,
    low_visibility_pct: float | None = None,
    n_samples: int = 0,
) -> QualityEvidence:
    if mean_visibility is None or n_samples <= 0:
        return QualityEvidence(
            layer=LAYER_LANDMARK_CONFIDENCE,
            status="unavailable",
            explanation="No landmark visibility samples were recorded.",
            formula="mean detector visibility across observed landmarks",
        )
    vis = float(mean_visibility)
    low = 0.0 if low_visibility_pct is None else float(low_visibility_pct)
    if vis >= 0.65 and low < 35.0:
        status: QualityStatus = "computed"
        expl = f"Mean detector visibility {vis:.0%} across {n_samples} landmark samples."
    elif vis >= 0.40:
        status = "limited"
        expl = (
            f"Mean detector visibility {vis:.0%} "
            f"({low:.0f}% of samples below 0.45)."
        )
    else:
        status = "insufficient"
        expl = f"Mean detector visibility {vis:.0%} is too low to support kinematics."
    expl = f"{expl} {LANDMARK_NOT_TRACKING}"
    return QualityEvidence(
        layer=LAYER_LANDMARK_CONFIDENCE,
        status=status,
        explanation=expl,
        mean_visibility=vis,
        n_detected=n_samples,
        extras={"low_visibility_pct": low},
        formula="mean MediaPipe (or backend) visibility; not tracking error",
    )


def contact_coverage_evidence(
    *,
    side: str,
    n_frames: int,
    n_known: int,
    n_contact_on: int = 0,
    n_confirmed_swing: int = 0,
    dominant_unknown_reason: str = "",
) -> QualityEvidence:
    """Per-foot fraction of usable contact observations (not laterality)."""
    if n_frames <= 0:
        return QualityEvidence(
            layer=LAYER_CONTACT_COVERAGE,
            status="unavailable",
            explanation=f"{side} contact coverage has no frames.",
            n_frames=0,
            formula="known_frames / n_frames  (Unknown is a gap, not swing)",
            extras={"side": side},
        )
    coverage = 100.0 * n_known / n_frames
    unknown = n_frames - n_known
    if n_known <= 0:
        status: QualityStatus = "insufficient"
        expl = f"{side} contact known on 0/{n_frames} frames."
    elif coverage >= 80.0:
        status = "computed"
        expl = (
            f"{side} contact known on {n_known}/{n_frames} frames "
            f"({coverage:.0f}% coverage)."
        )
    elif coverage >= 40.0:
        status = "limited"
        expl = (
            f"{side} contact known on {n_known}/{n_frames} frames "
            f"({coverage:.0f}% coverage). {unknown} Unknown samples are gaps."
        )
    else:
        status = "insufficient"
        expl = (
            f"{side} contact known on only {n_known}/{n_frames} frames "
            f"({coverage:.0f}% coverage)."
        )
    if dominant_unknown_reason:
        expl += f" Dominant Unknown reason: {dominant_unknown_reason}."
    expl += f" {UNEQUAL_CONTACT_COVERAGE_NOTE}"
    return QualityEvidence(
        layer=LAYER_CONTACT_COVERAGE,
        status=status,
        explanation=expl,
        coverage_pct=coverage,
        n_frames=n_frames,
        n_detected=n_known,
        extras={
            "side": side,
            "n_unknown": unknown,
            "n_contact_on": n_contact_on,
            "n_confirmed_swing": n_confirmed_swing,
            "dominant_unknown_reason": dominant_unknown_reason,
        },
        formula="known_frames / n_frames  (Unknown ≠ non-contact)",
    )


def bilateral_contact_coverage_evidence(
    left: QualityEvidence,
    right: QualityEvidence,
) -> QualityEvidence:
    """Session contact-coverage status from both feet."""
    left_pct = 0.0 if left.coverage_pct is None else float(left.coverage_pct)
    right_pct = 0.0 if right.coverage_pct is None else float(right.coverage_pct)
    n_frames = max(int(left.n_frames), int(right.n_frames))
    unequal = abs(left_pct - right_pct) >= 30.0
    weakest = min(left_pct, right_pct)
    if n_frames <= 0:
        status: QualityStatus = "unavailable"
    elif weakest < 20.0 or (unequal and weakest < 40.0):
        status = "insufficient"
    elif weakest < 80.0 or unequal:
        status = "limited"
    elif left.status == "computed" and right.status == "computed":
        status = "computed"
    else:
        status = "limited"
    expl = (
        f"L {left_pct:.0f}% known · R {right_pct:.0f}% known. "
        f"{UNEQUAL_CONTACT_COVERAGE_NOTE}"
    )
    return QualityEvidence(
        layer=LAYER_CONTACT_COVERAGE,
        status=status,
        explanation=expl,
        coverage_pct=0.5 * (left_pct + right_pct) if n_frames else None,
        n_frames=n_frames,
        extras={
            "left_pct": left_pct,
            "right_pct": right_pct,
            "unequal": unequal,
            "left": left.to_dict(),
            "right": right.to_dict(),
        },
        formula="per-foot known_frames / n_frames; not a laterality index",
    )


def cycle_detection_evidence(
    *,
    n_cycles: int,
    n_left: int = 0,
    n_right: int = 0,
    usable_duration_s: float | None = None,
    contact_confidence: float | None = None,
    reason: str = "",
    n_candidates: int | None = None,
    n_accepted: int | None = None,
) -> QualityEvidence:
    accepted = n_cycles if n_accepted is None else int(n_accepted)
    have = accepted >= 1
    status = status_from_evidence(
        have_inputs=have or bool(reason),
        n_cycles=accepted,
        usable_duration_s=usable_duration_s,
    )
    if not have and not reason:
        status = "unavailable"
        expl = "Gait-cycle detection has not been run."
    elif accepted < 1:
        status = "insufficient"
        expl = reason or "No complete ipsilateral heel-strike cycles were accepted."
        if n_candidates:
            expl = (
                f"{int(n_candidates)} candidate HS–HS interval(s); "
                f"0 accepted. {expl}"
            )
    else:
        expl = (
            f"{accepted} accepted HS–HS cycles ({n_left} L / {n_right} R)"
            + (
                f" over {usable_duration_s:.2f} s"
                if usable_duration_s is not None
                else ""
            )
            + "."
        )
        if n_candidates is not None:
            expl = f"{int(n_candidates)} candidates; {expl}"
        if contact_confidence is not None:
            expl += f" Contact score {float(contact_confidence):.0%} (event evidence, not pose coverage)."
        if status == "limited":
            expl += " Few cycles or short duration — treat timing as limited."
        if reason:
            expl += f" {reason}"
    extras: dict[str, Any] = {
        "n_left": n_left,
        "n_right": n_right,
        "contact_confidence": contact_confidence,
        "n_accepted": accepted,
    }
    if n_candidates is not None:
        extras["n_candidates"] = int(n_candidates)
    return QualityEvidence(
        layer=LAYER_CYCLE_DETECTION,
        status=status,
        explanation=expl,
        usable_duration_s=usable_duration_s,
        n_cycles=accepted,
        extras=extras,
        formula="Candidates = raw ipsilateral HS–HS; accepted = cycles that passed coverage gates",
    )


def spectral_reliability_evidence(
    *,
    duration_s: float | None,
    gap_fraction: float | None,
    n_cycles: float | None,
    peak_value: float | None = None,
    insufficient_reason: str | None = None,
) -> QualityEvidence:
    cycles = 0 if n_cycles is None else int(n_cycles)
    have = duration_s is not None or gap_fraction is not None or n_cycles is not None
    if insufficient_reason:
        status: QualityStatus = "insufficient"
        expl = str(insufficient_reason)
    else:
        status = status_from_evidence(
            have_inputs=bool(have),
            n_cycles=cycles,
            usable_duration_s=duration_s,
            gap_fraction=gap_fraction,
        )
        parts = []
        if duration_s is not None:
            parts.append(f"duration {float(duration_s):.2f} s")
        if gap_fraction is not None:
            parts.append(f"gap fraction {float(gap_fraction):.0%}")
        if n_cycles is not None:
            parts.append(f"≈{float(n_cycles):.1f} stride cycles in the series")
        if peak_value is not None:
            parts.append(f"ACF peak ρ={float(peak_value):.2f}")
        expl = (
            "Spectral reliability from "
            + (", ".join(parts) if parts else "the processed series")
            + ". This is not pose coverage or video quality."
        )
    return QualityEvidence(
        layer=LAYER_SPECTRAL,
        status=status,
        explanation=expl,
        usable_duration_s=duration_s,
        gap_fraction=gap_fraction,
        n_cycles=cycles,
        extras={"peak_value": peak_value},
        formula=(
            "0.35·clip(ρ_stride,0,1) + 0.25·clip(prominence,0,1) + "
            "0.25·min(n_cycles/6,1) + 0.15·(1−gap_fraction)"
        ),
    )


def biomechanical_validity_evidence(
    *,
    calibrated_metric: bool = False,
    force_plates: bool = False,
    stability_reportable: bool | None = None,
    opensim_id: bool = False,
    reason: str = "",
) -> QualityEvidence:
    physical = bool(calibrated_metric or force_plates or opensim_id)
    if stability_reportable is False and not physical:
        status: QualityStatus = "unavailable"
        expl = reason or (
            "Physical COM–BoS stability and inverse-dynamics moments are "
            "unavailable from monocular pose. The experimental proxy is not reported."
        )
    elif physical:
        status = "computed" if stability_reportable is not False else "limited"
        bits = []
        if calibrated_metric:
            bits.append("calibrated metric coordinates")
        if force_plates:
            bits.append("measured GRF")
        if opensim_id:
            bits.append("OpenSim inverse dynamics")
        expl = "Physical interpretation supported by " + ", ".join(bits) + "."
    else:
        status = "unavailable"
        expl = reason or (
            "Biomechanical validity is unavailable: coordinates are inferred "
            "monocular estimates without force plates."
        )
    return QualityEvidence(
        layer=LAYER_BIOMECHANICAL,
        status=status,
        explanation=expl,
        extras={
            "calibrated_metric": calibrated_metric,
            "force_plates": force_plates,
            "stability_reportable": stability_reportable,
            "opensim_id": opensim_id,
        },
        formula="Physical validity requires calibrated metric and/or measured GRF",
    )


def gait_quality_status(n_active: int, n_intended: int = len(GAIT_QUALITY_INTENDED_COMPONENTS)) -> QualityStatus:
    if n_active <= 0:
        return "unavailable"
    if n_active >= 4:
        return "computed"
    if n_active >= 2:
        return "limited"
    return "insufficient"


def gait_quality_confidence(
    *,
    n_active: int,
    input_confidences: list[float],
    n_intended: int = len(GAIT_QUALITY_INTENDED_COMPONENTS),
) -> float:
    """Confidence from coverage of intended components and real input scores.

    Does not invent a 0.5 default for missing contact confidence.
    """
    coverage = n_active / max(n_intended, 1)
    real = [float(c) for c in input_confidences if c is not None and 0.0 <= float(c) <= 1.0]
    base = sum(real) / len(real) if real else coverage
    return max(0.0, min(1.0, 0.5 * coverage + 0.5 * base))


def layer_documentation() -> dict[str, Any]:
    return {
        "layers": {
            LAYER_POSE_COVERAGE: "Detected-pose frame fraction (presence only).",
            LAYER_LANDMARK_CONFIDENCE: LANDMARK_NOT_TRACKING,
            LAYER_CONTACT_COVERAGE: (
                "Per-foot fraction of usable contact observations. Unknown is a "
                "gap, not swing, and unequal coverage is not laterality."
            ),
            LAYER_CYCLE_DETECTION: "Ipsilateral HS–HS cycles, duration, contact evidence.",
            LAYER_SPECTRAL: "Duration, gaps, contributing cycles, ACF/PSD peak quality.",
            LAYER_BIOMECHANICAL: "Physical interpretation support; unavailable stays unavailable.",
        },
        "status": ("computed", "limited", "insufficient", "unavailable"),
        "forbidden_labels": list(FORBIDDEN_QUALITY_WORDS),
        "gait_quality": {
            "kind": GAIT_QUALITY_KIND,
            "formula": GAIT_QUALITY_FORMULA,
            "limitations": GAIT_QUALITY_LIMITATIONS,
            "threshold_source": GAIT_QUALITY_THRESHOLD_SOURCE,
            "weights": dict(GAIT_QUALITY_WEIGHTS),
        },
        "landmark_note": LANDMARK_NOT_TRACKING,
    }
