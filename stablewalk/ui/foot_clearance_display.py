"""
Foot clearance dashboard model — explicit clearance, contact state, and methodology.

Clearance is the vertical distance from the lowest reliable heel / toe
(foot-index) landmark to the estimated floor plane in hip-centered canonical
coordinates (estimated body-normalized meters).
Displayed centimeters apply only when ``scale_mode == "body_normalized"``:

    clearance_cm = clearance_m * 100
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Literal

from stablewalk.analysis.foot_clearance_filter import (
    COORDINATE_UNITS,
    FootClearanceFilteredSample,
    FootClearanceSwingStats,
    FootLandmarkSample,
    build_filtered_foot_clearance_series,
    unavailable_display_reason,
)
from stablewalk.analysis.ground_reference import (
    GroundReferencePlane,
    bilateral_foot_clearance,
    estimate_ground_plane,
)
from stablewalk.ui.scientific_labels import (
    LABEL_FOOT_CLEARANCE,
    LABEL_FOOT_CLEARANCE_CONFIDENCE,
    LABEL_FOOT_CLEARANCE_CURRENT,
    LABEL_FOOT_CLEARANCE_MAX_SWING,
    LABEL_FOOT_CLEARANCE_MIN_SWING,
)

FOOT_CLEARANCE_FLOOR_REFERENCE = (
    "Estimated floor plane (sequence-level heel/toe/foot-index)"
)
FOOT_CLEARANCE_FOOT_REFERENCE = (
    "lowest reliable heel/toe/foot-index height above floor"
)
FOOT_CLEARANCE_UNITS = "centimeters (from body-normalized meters × 100)"

FootClearanceConfidence = Literal["HIGH", "MODERATE", "LOW"]
FootDisplayState = Literal["CONTACT", "SWING", "Unavailable", "UNKNOWN"]

# Overview panel copy — estimated because values come from monocular pose.
OVERVIEW_SECTION_TITLE = LABEL_FOOT_CLEARANCE
OVERVIEW_DISTANCE_CAPTION = LABEL_FOOT_CLEARANCE_CURRENT

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
    return f"{LABEL_FOOT_CLEARANCE_CONFIDENCE}: {_confidence_title(confidence)}"


def foot_clearance_confidence_label(confidence: FootClearanceConfidence | str) -> str:
    """Scientific Details dialog confidence line."""
    return f"{LABEL_FOOT_CLEARANCE_CONFIDENCE}: {_confidence_title(confidence)}"


def overview_unavailable_reason(raw_reason: str) -> str:
    """Short Overview-friendly unavailable reason (max two short lines)."""
    text = (raw_reason or "").strip()
    if not text:
        return "Measurement unavailable for this recording"
    mapped = _OVERVIEW_UNAVAILABLE_REASONS.get(text, text)
    # Collapse technical phrasing for the main dashboard.
    if "outlier" in mapped.lower() and "exceeds" in text.lower():
        mapped = "Sample rejected as outlier"
    if mapped == "Measurement unavailable":
        mapped = "Measurement unavailable for this recording"
    if len(mapped) > 48:
        mapped = mapped[:45].rstrip() + "..."
    return mapped


OVERVIEW_VALUE_UNAVAILABLE = "Unavailable"


def overview_value_text(display: str | None) -> str:
    """Overview Current / Min Swing / Max Swing cell. Never empty."""
    text = str(display or "").strip()
    if not text or text in {"—", "-", "–", "N/A", "n/a", "na"}:
        return OVERVIEW_VALUE_UNAVAILABLE
    return text


def overview_state_text(display: str | None) -> str:
    """Overview Gait state cell. Contact / Swing, otherwise Unavailable."""
    raw = str(display or "").strip().upper()
    if raw in {"CONTACT", "STANCE"}:
        return "Contact"
    if raw == "SWING":
        return "Swing"
    return OVERVIEW_VALUE_UNAVAILABLE


def overview_distance_text(
    foot: "FootClearanceFootPanel",
    *,
    contextual: bool = False,
) -> str:
    """Dominant distance readout for Overview foot cards."""
    if foot.displayed_clearance_cm is not None:
        return f"{foot.displayed_clearance_cm:.1f} cm"
    if contextual:
        from stablewalk.ui.unavailable_labels import foot_clearance_unavailable_label

        return foot_clearance_unavailable_label(foot.unavailable_reason)
    return OVERVIEW_VALUE_UNAVAILABLE


def overview_live_clearance_text(
    side: Literal["Left", "Right"],
    foot: "FootClearanceFootPanel",
) -> str:
    """Exact bilateral real-time readout used by the Overview panel."""
    return f"{side} Foot Clearance: {overview_distance_text(foot)}"


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
    min_swing_display: str
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


@dataclass
class FootClearanceLiveCache:
    """Session stats reused while playback only refreshes the current frame."""

    recording_id: int
    plane: GroundReferencePlane | None
    left_stats: FootClearanceSwingStats | None
    right_stats: FootClearanceSwingStats | None
    confidence: FootClearanceConfidence
    rejected_sample_pct: float
    methodology_note: str
    debug_lines: tuple[str, ...] = ()


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
    if not math.isfinite(float(value_m)):
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
        current_display = OVERVIEW_VALUE_UNAVAILABLE
        displayed_cm = None

    state_display: str = "—"
    if stats and available and last_sample is not None:
        state_display = last_sample.display_state if last_sample.is_valid else "—"
    elif stats and stats.current_reject_reason:
        state_display = "—"

    max_swing = _format_stat_cm(stats.max_swing_m if stats else None, scale_mode=scale)
    min_swing = _format_stat_cm(stats.min_swing_m if stats else None, scale_mode=scale)
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
        min_swing_display=min_swing,
        max_swing_display=max_swing,
        avg_swing_display=avg_swing,
        median_swing_display=median_swing,
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


def _current_frame_stats_and_sample(
    snapshot: SkeletonSnapshot,
    plane: GroundReferencePlane | None,
    side: str,
    base_stats: FootClearanceSwingStats | None,
    *,
    prev_phase: str | None,
) -> tuple[FootClearanceSwingStats, FootClearanceFilteredSample | None]:
    """O(1) current-frame heel/toe clearance; keep cached swing min/max."""
    from stablewalk.analysis.foot_clearance_filter import (
        _display_state,
        _heel_toe_landmark_sample,
        _phase_from_clearance,
        _raw_reject_reason,
    )

    landmark = (
        _heel_toe_landmark_sample(snapshot, plane, side)
        if plane is not None
        else FootLandmarkSample(None, None, None, None, 0.0, 0.0, None)
    )
    reason = _raw_reject_reason(landmark, plane=plane)
    raw_m = landmark.raw_clearance_m
    phase = _phase_from_clearance(
        raw_m if not reason else None,
        prev_phase=prev_phase,
    )
    if reason:
        phase = prev_phase or "unknown"
    is_valid = not reason and raw_m is not None
    if not is_valid and phase == "swing":
        phase = "unknown"
    sample = FootClearanceFilteredSample(
        frame_index=int(snapshot.frame_index),
        timestamp_s=float(snapshot.time_s),
        vertical_axis=plane.vertical_axis if plane else "y",
        floor_y_m=plane.floor_y if plane else None,
        landmark=landmark,
        raw_clearance_m=raw_m,
        filtered_clearance_m=raw_m if is_valid else None,
        phase=phase,
        display_state=_display_state(phase, valid=is_valid),
        is_valid=is_valid,
        reject_reason=reason if not is_valid else "",
    )
    if base_stats is None:
        stats = FootClearanceSwingStats(
            current_m=raw_m if is_valid else None,
            current_valid=is_valid,
            current_reject_reason=reason if not is_valid else "",
            min_swing_m=None,
            max_swing_m=None,
            avg_swing_m=None,
            median_swing_m=None,
            valid_swing_count=0,
            total_swing_count=0,
            rejected_count=0 if is_valid else 1,
            rejected_pct=0.0 if is_valid else 100.0,
        )
    else:
        stats = FootClearanceSwingStats(
            current_m=raw_m if is_valid else None,
            current_valid=is_valid,
            current_reject_reason=reason if not is_valid else "",
            min_swing_m=base_stats.min_swing_m,
            max_swing_m=base_stats.max_swing_m,
            avg_swing_m=base_stats.avg_swing_m,
            median_swing_m=base_stats.median_swing_m,
            valid_swing_count=base_stats.valid_swing_count,
            total_swing_count=base_stats.total_swing_count,
            rejected_count=base_stats.rejected_count,
            rejected_pct=base_stats.rejected_pct,
        )
    return stats, sample


def foot_clearance_dashboard_current_frame(
    snapshot: SkeletonSnapshot | None,
    cache: FootClearanceLiveCache | None,
    *,
    prev_left_phase: str | None = None,
    prev_right_phase: str | None = None,
) -> FootClearanceDashboardPanel | None:
    """Lightweight Overview update: current-frame clearance only (cached min/max)."""
    if snapshot is None or cache is None:
        return None
    plane = cache.plane
    left_stats, left_sample = _current_frame_stats_and_sample(
        snapshot,
        plane,
        "left",
        cache.left_stats,
        prev_phase=prev_left_phase,
    )
    right_stats, right_sample = _current_frame_stats_and_sample(
        snapshot,
        plane,
        "right",
        cache.right_stats,
        prev_phase=prev_right_phase,
    )
    cache.left_stats = left_stats
    cache.right_stats = right_stats
    left = _build_foot_panel(
        side="left",
        side_label="LEFT FOOT",
        snapshot=snapshot,
        plane=plane,
        swing_stats=left_stats,
        last_sample=left_sample,
    )
    right = _build_foot_panel(
        side="right",
        side_label="RIGHT FOOT",
        snapshot=snapshot,
        plane=plane,
        swing_stats=right_stats,
        last_sample=right_sample,
    )
    floor_y = plane.floor_y if plane is not None else None
    return FootClearanceDashboardPanel(
        left=left,
        right=right,
        confidence=cache.confidence,
        confidence_label=measurement_confidence_label(cache.confidence),
        floor_reference=FOOT_CLEARANCE_FLOOR_REFERENCE,
        foot_reference=FOOT_CLEARANCE_FOOT_REFERENCE,
        units=FOOT_CLEARANCE_UNITS,
        floor_height_canonical_m=floor_y,
        rejected_sample_pct=cache.rejected_sample_pct,
        debug_lines=cache.debug_lines,
        methodology_note=cache.methodology_note,
        left_phase=left_sample.phase if left_sample else None,
        right_phase=right_sample.phase if right_sample else None,
    )


def build_foot_clearance_dashboard_with_cache(
    snapshot: SkeletonSnapshot | None,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
    *,
    prev_left_phase: str | None = None,
    prev_right_phase: str | None = None,
) -> tuple[FootClearanceDashboardPanel | None, FootClearanceLiveCache | None]:
    """Full series build plus a live cache for current-frame-only refreshes."""
    if snapshot is None or recording is None:
        return None, None

    # Session extrema must describe the analyzed recording, not only the frames
    # visited before the current playhead. Current clearance/state are rebuilt
    # separately for ``snapshot`` below.
    session_end_frame = max(
        float(end_frame_float),
        float(max(len(recording.snapshots) - 1, 0)),
    )
    plane = estimate_ground_plane(recording, session_end_frame)
    left_series = build_filtered_foot_clearance_series(
        recording, plane, "left", end_frame_float=session_end_frame
    )
    right_series = build_filtered_foot_clearance_series(
        recording, plane, "right", end_frame_float=session_end_frame
    )
    confidence = _foot_confidence(
        plane, left_series.swing_stats, right_series.swing_stats
    )

    left_stats, left_sample = _current_frame_stats_and_sample(
        snapshot,
        plane,
        "left",
        left_series.swing_stats,
        prev_phase=prev_left_phase,
    )
    right_stats, right_sample = _current_frame_stats_and_sample(
        snapshot,
        plane,
        "right",
        right_series.swing_stats,
        prev_phase=prev_right_phase,
    )

    left = _build_foot_panel(
        side="left",
        side_label="LEFT FOOT",
        snapshot=snapshot,
        plane=plane,
        swing_stats=left_stats,
        last_sample=left_sample,
    )
    right = _build_foot_panel(
        side="right",
        side_label="RIGHT FOOT",
        snapshot=snapshot,
        plane=plane,
        swing_stats=right_stats,
        last_sample=right_sample,
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
        "Estimated Foot Clearance = min(heel, toe) vertical distance above a "
        "sequence-level floor plane fitted from stable stance-foot landmarks "
        "(+Y up, body-normalized meters converted to cm). Monocular pose estimate — "
        "not a measured clinical clearance."
    )

    left_phase = left_sample.phase if left_sample else None
    right_phase = right_sample.phase if right_sample else None

    panel = FootClearanceDashboardPanel(
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
    cache = FootClearanceLiveCache(
        recording_id=id(recording),
        plane=plane,
        left_stats=left_stats,
        right_stats=right_stats,
        confidence=confidence,
        rejected_sample_pct=reject_pct,
        methodology_note=note,
        debug_lines=debug_lines,
    )
    return panel, cache


def foot_clearance_dashboard_for_panel(
    snapshot: SkeletonSnapshot | None,
    recording: GaitMotionRecording | None,
    end_frame_float: float,
    *,
    prev_left_phase: str | None = None,
    prev_right_phase: str | None = None,
) -> FootClearanceDashboardPanel | None:
    """Build the full foot clearance dashboard model for the current frame."""
    panel, _cache = build_foot_clearance_dashboard_with_cache(
        snapshot,
        recording,
        end_frame_float,
        prev_left_phase=prev_left_phase,
        prev_right_phase=prev_right_phase,
    )
    return panel


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
            f"  {LABEL_FOOT_CLEARANCE_MIN_SWING}: {foot.min_swing_display}",
            f"  {LABEL_FOOT_CLEARANCE_MAX_SWING}: {foot.max_swing_display}",
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
        "ESTIMATED FOOT CLEARANCE — DETAILS",
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
