"""
Foot clearance dashboard model — explicit clearance, contact state, and methodology.

Clearance is the vertical distance from min(heel, toe) to the estimated floor plane
in hip-centered canonical coordinates (estimated body-normalized meters).
Displayed centimeters apply only when ``scale_mode == "body_normalized"``:

    clearance_cm = clearance_m * 100
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

from stablewalk.analysis.foot_clearance_filter import (
    COORDINATE_UNITS,
    FootClearanceFilteredSample,
    FootClearanceSwingStats,
    build_filtered_foot_clearance_series,
    unavailable_display_reason,
)
from stablewalk.analysis.ground_reference import (
    GroundReferencePlane,
    bilateral_foot_clearance,
    estimate_ground_plane,
)
from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot

FOOT_CLEARANCE_FLOOR_REFERENCE = "Estimated floor plane (sequence-level heel/toe)"
FOOT_CLEARANCE_FOOT_REFERENCE = "min(heel, toe) height above floor"
FOOT_CLEARANCE_UNITS = "centimeters (from body-normalized meters × 100)"

FootClearanceConfidence = Literal["HIGH", "MODERATE", "LOW"]
FootDisplayState = Literal["CONTACT", "SWING", "Unavailable", "UNKNOWN"]

# Overview panel copy (Details dialog keeps scientific "Foot Clearance" terminology).
OVERVIEW_SECTION_TITLE = "FOOT-TO-FLOOR DISTANCE"
OVERVIEW_DISTANCE_CAPTION = "DISTANCE FROM FLOOR"

_OVERVIEW_UNAVAILABLE_REASONS: dict[str, str] = {
    "low heel/toe confidence": "Low foot landmark confidence",
    "heel/toe above ankle (pose artifact)": "Pose landmark artifact",
    "missing heel/toe landmarks": "Foot landmarks missing",
    "sample rejected as outlier (exceeds plausible max)": "Sample rejected as outlier",
    "current foot sample rejected as outlier": "Sample rejected as outlier",
    "floor estimate invalid": "Floor estimate unavailable",
    "coordinate scale not body-normalized": "Body scale unavailable",
    "Foot clearance data unavailable": "Measurement unavailable",
}


def _confidence_title(confidence: FootClearanceConfidence | str) -> str:
    key = str(confidence).strip().upper()
    return key.title() if key in ("HIGH", "MODERATE", "LOW") else str(confidence)


def measurement_confidence_label(confidence: FootClearanceConfidence | str) -> str:
    """Overview header confidence line."""
    return f"Measurement Confidence: {_confidence_title(confidence)}"


def foot_clearance_confidence_label(confidence: FootClearanceConfidence | str) -> str:
    """Scientific Details dialog confidence line."""
    return f"Foot Clearance Confidence: {_confidence_title(confidence)}"


def overview_unavailable_reason(raw_reason: str) -> str:
    """Short Overview-friendly unavailable reason (max two short lines)."""
    text = (raw_reason or "").strip()
    if not text:
        return "Measurement unavailable"
    mapped = _OVERVIEW_UNAVAILABLE_REASONS.get(text, text)
    # Collapse technical phrasing for the main dashboard.
    if "outlier" in mapped.lower() and "exceeds" in text.lower():
        mapped = "Sample rejected as outlier"
    if len(mapped) > 48:
        mapped = mapped[:45].rstrip() + "..."
    return mapped


def overview_distance_text(foot: "FootClearanceFootPanel") -> str:
    """Dominant distance readout for Overview foot cards."""
    if foot.displayed_clearance_cm is not None:
        return f"{foot.displayed_clearance_cm:.1f} cm"
    return "Unavailable"


@dataclass(frozen=True)
class FootSkeletonLabel:
    """Compact 3D skeleton foot readout (distance + contact from dashboard model)."""

    side: Literal["left", "right"]
    clearance_cm: float | None
    contact: bool

    def compact_text(self) -> str:
        """Multi-line label: side, distance, contact state (or N/A)."""
        prefix = "L" if self.side == "left" else "R"
        if self.clearance_cm is None:
            return f"{prefix}: N/A"
        state = "CONTACT" if self.contact else "SWING"
        return f"{prefix}\n{self.clearance_cm:.1f} cm\n{state}"


def foot_skeleton_labels_from_dashboard(
    panel: "FootClearanceDashboardPanel | None",
    *,
    left_contact: bool,
    right_contact: bool,
) -> tuple[FootSkeletonLabel, FootSkeletonLabel]:
    """Build skeleton labels from the same dashboard model as Overview foot cards."""
    if panel is None:
        return (
            FootSkeletonLabel("left", None, left_contact),
            FootSkeletonLabel("right", None, right_contact),
        )
    return (
        FootSkeletonLabel("left", panel.left.displayed_clearance_cm, left_contact),
        FootSkeletonLabel("right", panel.right.displayed_clearance_cm, right_contact),
    )


def parse_card_clearance_cm(card_text: str) -> float | None:
    """Parse foot-card distance text into centimeters for parity checks."""
    text = (card_text or "").strip()
    if not text or text == "Unavailable" or "N/A" in text.upper():
        return None
    if "cm" not in text:
        return None
    try:
        return float(text.replace("cm", "").strip())
    except ValueError:
        return None


def skeleton_clearance_matches_card(
    skeleton_cm: float | None,
    card_cm: float | None,
    *,
    tolerance: float = 0.05,
) -> bool:
    """True when skeleton and foot-card clearance agree within display rounding."""
    if skeleton_cm is None and card_cm is None:
        return True
    if skeleton_cm is None or card_cm is None:
        return False
    return abs(skeleton_cm - card_cm) <= tolerance


@dataclass(frozen=True)
class FootClearanceFootPanel:
    """One foot row in the dedicated Foot Clearance component."""

    side_label: str
    current_display: str
    state_display: str
    unavailable_reason: str
    max_swing_display: str
    avg_swing_display: str
    median_swing_display: str
    valid_samples_display: str
    measuring_joint: str | None
    heel_clearance_m: float | None
    toe_clearance_m: float | None
    displayed_clearance_cm: float | None
    compact_cm: str
    compact_state: str


@dataclass(frozen=True)
class FootClearanceDashboardPanel:
    """Bilateral foot clearance for GUI + debug export."""

    left: FootClearanceFootPanel
    right: FootClearanceFootPanel
    confidence: FootClearanceConfidence
    confidence_label: str
    floor_reference: str
    foot_reference: str
    units: str
    floor_height_canonical_m: float | None
    rejected_sample_pct: float
    debug_lines: tuple[str, ...]
    methodology_note: str
    left_phase: str | None = None
    right_phase: str | None = None


def clearance_m_to_display_cm(
    distance_m: float | None,
    *,
    scale_mode: str,
) -> tuple[str, float | None]:
    """
    Convert canonical body-normalized meters to centimeters for display.

    Only applies ``× 100`` when coordinates are estimated body-scale meters.
    """
    if distance_m is None:
        return "Unavailable", None
    if scale_mode != "body_normalized":
        return "Unavailable", None
    cm = distance_m * 100.0
    return f"{cm:.1f} cm", cm


def _format_stat_cm(value_m: float | None, *, scale_mode: str) -> str:
    if value_m is None or scale_mode != "body_normalized":
        return "Unavailable"
    return f"{value_m * 100.0:.1f} cm"


def _measuring_joint_from_sample(
    snapshot: SkeletonSnapshot,
    side: str,
    *,
    heel_m: float | None,
    toe_m: float | None,
) -> str | None:
    if heel_m is None and toe_m is None:
        return None
    if heel_m is not None and toe_m is not None:
        return f"{side}_heel" if heel_m <= toe_m else f"{side}_toe"
    if heel_m is not None:
        return f"{side}_heel"
    return f"{side}_toe"


def _build_foot_panel(
    *,
    side: str,
    side_label: str,
    snapshot: SkeletonSnapshot,
    plane: GroundReferencePlane | None,
    swing_stats: FootClearanceSwingStats | None,
    last_sample: FootClearanceFilteredSample | None,
) -> FootClearanceFootPanel:
    from stablewalk.analysis.foot_clearance_filter import _heel_toe_landmark_sample

    scale = plane.scale_mode if plane is not None else "unknown"
    landmark = (
        _heel_toe_landmark_sample(snapshot, plane, side)
        if plane is not None
        else None
    )
    heel_m = landmark.heel_clearance_m if landmark else None
    toe_m = landmark.toe_clearance_m if landmark else None
    measuring = _measuring_joint_from_sample(snapshot, side, heel_m=heel_m, toe_m=toe_m)

    stats = swing_stats
    available = bool(stats and stats.current_valid and scale == "body_normalized")
    reason = ""
    if stats and not stats.current_valid:
        reason = unavailable_display_reason(stats.current_reject_reason)
    elif plane is None:
        reason = "floor estimate invalid"
    elif scale != "body_normalized":
        reason = "coordinate scale not body-normalized"

    current_m = stats.current_m if stats and available else None
    current_display, displayed_cm = clearance_m_to_display_cm(current_m, scale_mode=scale)

    if not available:
        current_display = "Unavailable"
        displayed_cm = None

    state_display: str = "Unavailable"
    if stats and available and last_sample is not None:
        state_display = last_sample.display_state if last_sample.is_valid else "Unavailable"
    elif stats and stats.current_reject_reason:
        state_display = "Unavailable"

    max_swing = _format_stat_cm(stats.max_swing_m if stats else None, scale_mode=scale)
    avg_swing = _format_stat_cm(stats.avg_swing_m if stats else None, scale_mode=scale)
    median_swing = _format_stat_cm(
        stats.median_swing_m if stats else None, scale_mode=scale
    )
    valid_n = stats.valid_swing_count if stats else 0
    valid_display = str(valid_n) if stats else "0"

    compact_cm = current_display.replace("Unavailable — ", "").replace("Unavailable", "—")
    if not available:
        compact_cm = "—"
    compact_state = state_display if available else "—"

    return FootClearanceFootPanel(
        side_label=side_label,
        current_display=current_display,
        state_display=state_display if isinstance(state_display, str) else str(state_display),
        unavailable_reason=reason,
        max_swing_display=max_swing if available or valid_n > 0 else "Unavailable",
        avg_swing_display=avg_swing if available or valid_n > 0 else "Unavailable",
        median_swing_display=median_swing if available or valid_n > 0 else "Unavailable",
        valid_samples_display=valid_display,
        measuring_joint=measuring,
        heel_clearance_m=heel_m,
        toe_clearance_m=toe_m,
        displayed_clearance_cm=displayed_cm,
        compact_cm=compact_cm,
        compact_state=compact_state,
    )


def _foot_confidence(
    plane: GroundReferencePlane | None,
    left_stats: FootClearanceSwingStats | None,
    right_stats: FootClearanceSwingStats | None,
) -> FootClearanceConfidence:
    if plane is None or plane.scale_mode != "body_normalized":
        return "LOW"
    reject_pct = 0.0
    for stats in (left_stats, right_stats):
        if stats:
            reject_pct = max(reject_pct, stats.rejected_pct)
    if reject_pct > 40:
        return "LOW"
    valid = sum(
        1
        for stats in (left_stats, right_stats)
        if stats and stats.valid_swing_count >= 3
    )
    if valid == 2 and reject_pct < 15:
        return "HIGH"
    if reject_pct < 30:
        return "MODERATE"
    return "LOW"


def foot_clearance_dashboard_for_panel(
    snapshot: SkeletonSnapshot | None,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
    *,
    prev_left_phase: str | None = None,
    prev_right_phase: str | None = None,
) -> FootClearanceDashboardPanel | None:
    """Build the full foot clearance dashboard model for the current frame."""
    del prev_left_phase, prev_right_phase  # phases come from filtered series
    if snapshot is None or recording is None:
        return None

    plane = estimate_ground_plane(recording, end_frame_float)
    left_series = build_filtered_foot_clearance_series(
        recording, plane, "left", end_frame_float=end_frame_float
    )
    right_series = build_filtered_foot_clearance_series(
        recording, plane, "right", end_frame_float=end_frame_float
    )
    confidence = _foot_confidence(
        plane, left_series.swing_stats, right_series.swing_stats
    )

    left = _build_foot_panel(
        side="left",
        side_label="LEFT FOOT",
        snapshot=snapshot,
        plane=plane,
        swing_stats=left_series.swing_stats,
        last_sample=left_series.samples[-1] if left_series.samples else None,
    )
    right = _build_foot_panel(
        side="right",
        side_label="RIGHT FOOT",
        snapshot=snapshot,
        plane=plane,
        swing_stats=right_series.swing_stats,
        last_sample=right_series.samples[-1] if right_series.samples else None,
    )

    reject_pct = 0.0
    for series in (left_series, right_series):
        if series.swing_stats:
            reject_pct = max(reject_pct, series.swing_stats.rejected_pct)

    floor_y = plane.floor_y if plane is not None else None
    debug_lines = (
        f"Coordinate units: {COORDINATE_UNITS}",
        f"Floor height canonical: {floor_y:.4f} m" if floor_y is not None else "Floor height canonical: —",
        (
            f"Floor candidates min/max/std: "
            f"{plane.floor_candidate_min:.4f} / {plane.floor_candidate_max:.4f} / "
            f"{plane.floor_candidate_std:.4f} m"
            if plane and plane.floor_candidate_min is not None
            else "Floor candidates: —"
        ),
        f"Left heel clearance m: {_fmt_m(left.heel_clearance_m)}",
        f"Left toe clearance m: {_fmt_m(left.toe_clearance_m)}",
        f"Left displayed clearance cm: {_fmt_cm(left.displayed_clearance_cm)}",
        f"Left valid swing samples: {left.valid_samples_display}",
        f"Right heel clearance m: {_fmt_m(right.heel_clearance_m)}",
        f"Right toe clearance m: {_fmt_m(right.toe_clearance_m)}",
        f"Right displayed clearance cm: {_fmt_cm(right.displayed_clearance_cm)}",
        f"Right valid swing samples: {right.valid_samples_display}",
        f"Rejected sample %: {reject_pct:.1f}",
    )

    note = (
        "Foot clearance = min(heel, toe) vertical distance above the sequence-level "
        "floor plane (+Y up, body-normalized meters converted to cm)."
    )

    left_phase = left_series.samples[-1].phase if left_series.samples else None
    right_phase = right_series.samples[-1].phase if right_series.samples else None

    return FootClearanceDashboardPanel(
        left=left,
        right=right,
        confidence=confidence,
        confidence_label=measurement_confidence_label(confidence),
        floor_reference=FOOT_CLEARANCE_FLOOR_REFERENCE,
        foot_reference=FOOT_CLEARANCE_FOOT_REFERENCE,
        units=FOOT_CLEARANCE_UNITS,
        floor_height_canonical_m=floor_y,
        rejected_sample_pct=reject_pct,
        debug_lines=debug_lines,
        methodology_note=note,
        left_phase=left_phase,
        right_phase=right_phase,
    )


def _fmt_m(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "—"


def _fmt_cm(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "—"


def format_foot_clearance_details(panel: FootClearanceDashboardPanel) -> str:
    """Structured methodology + per-foot readout for the Details dialog."""

    def _foot_advanced_block(foot: FootClearanceFootPanel) -> list[str]:
        lines = [
            foot.side_label,
            f"  Current: {foot.current_display}",
            f"  State: {foot.state_display}",
            f"  Maximum swing clearance: {foot.max_swing_display}",
            f"  Average swing clearance: {foot.avg_swing_display}",
            f"  Median swing clearance: {foot.median_swing_display}",
            f"  Valid sample count: {foot.valid_samples_display}",
        ]
        if foot.unavailable_reason:
            lines.append(f"  Reason: {foot.unavailable_reason}")
        if foot.measuring_joint:
            lines.append(f"  Measuring landmark: {foot.measuring_joint}")
        return lines

    floor_height = (
        f"{panel.floor_height_canonical_m:.4f} m"
        if panel.floor_height_canonical_m is not None
        else "—"
    )
    floor_debug = next(
        (line for line in panel.debug_lines if line.startswith("Floor candidates")),
        "Floor candidates: —",
    )

    lines = [
        "FOOT CLEARANCE — DETAILS",
        "",
        "Floor estimation",
        f"  Floor reference: {panel.floor_reference}",
        f"  Foot reference: {panel.foot_reference}",
        f"  Units: {panel.units}",
        f"  Floor height canonical: {floor_height}",
        f"  {floor_debug}",
        "",
        "Confidence",
        f"  {foot_clearance_confidence_label(panel.confidence)}",
        f"  {panel.confidence_label}",
        f"  Rejected outlier %: {panel.rejected_sample_pct:.1f}",
        "",
        "Advanced clearance statistics",
        *_foot_advanced_block(panel.left),
        "",
        *_foot_advanced_block(panel.right),
        "",
        "Methodology",
        f"  {panel.methodology_note}",
        "",
        "Debug",
        *[f"  {line}" for line in panel.debug_lines],
    ]
    return "\n".join(lines)
