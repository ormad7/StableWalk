"""
Contact vs displayed foot-clearance consistency audit.

Documents and checks alignment between:
- Overview foot-to-floor distance (filtered min heel/toe clearance)
- Gait-cycle contact detection (min heel/toe/ankle + hysteresis)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from stablewalk.analysis.foot_clearance_filter import build_filtered_foot_clearance_series
from stablewalk.analysis.gait_cycle_analysis import (
    GaitCycleAnalysisResult,
    estimate_leg_length_m,
    resolve_vertical_axis,
)
from stablewalk.analysis.ground_reference import (
    displayed_foot_clearance_m,
    estimate_ground_plane,
)
from stablewalk.models.gait_motion import GaitMotionRecording

ContactClearanceVerdict = Literal["CONSISTENT", "BORDERLINE", "INCONSISTENT", "LOW_CONFIDENCE"]

# Pipeline documentation (audit report).
DISPLAYED_CLEARANCE_FOOT_REFERENCE = "min(heel, toe) height above estimated floor plane"
CONTACT_DETECTION_FOOT_REFERENCE = "min(heel, toe, ankle) height above estimated floor plane"
SHARED_FLOOR_REFERENCE = "estimate_ground_plane (sequence-level heel/toe, stance-weighted)"
SHARED_VERTICAL_AXIS = "+Y (canonical StableWalk hip-centered frame)"
SHARED_UNITS = "body-normalized meters internally; Overview displays centimeters (m × 100)"

MIN_FOOT_VISIBILITY = 0.35

# Body-normalized consistency bands (fraction of estimated leg length).
DEFAULT_INCONSISTENT_LEG_RATIO = 0.12
DEFAULT_BORDERLINE_LEG_RATIO = 0.08


@dataclass(frozen=True)
class ContactClearanceConsistencyConfig:
    """Body-scale-aware thresholds for contact vs displayed-clearance checks."""

    inconsistent_leg_ratio: float = DEFAULT_INCONSISTENT_LEG_RATIO
    borderline_leg_ratio: float = DEFAULT_BORDERLINE_LEG_RATIO
    min_foot_visibility: float = MIN_FOOT_VISIBILITY


@dataclass(frozen=True)
class FootConsistencyAssessment:
    side: str
    verdict: ContactClearanceVerdict
    contact: bool
    displayed_clearance_cm: float | None
    displayed_clearance_m: float | None
    contact_foot_clearance_cm: float | None
    inconsistent_threshold_cm: float
    borderline_threshold_cm: float
    visibility: float
    reason: str


def _thresholds_m(
    leg_length_m: float,
    config: ContactClearanceConsistencyConfig,
    *,
    entry_clearance_m: float | None = None,
    exit_clearance_m: float | None = None,
    max_display_exit_clearance_m: float | None = None,
) -> tuple[float, float]:
    leg = max(leg_length_m, 0.20)
    inconsistent = max(
        leg * config.inconsistent_leg_ratio,
        max_display_exit_clearance_m or exit_clearance_m or leg * 0.14,
    )
    borderline = max(
        leg * config.borderline_leg_ratio,
        (entry_clearance_m or leg * 0.055),
    )
    if borderline >= inconsistent:
        borderline = inconsistent * 0.75
    return inconsistent, borderline


def assess_foot_contact_clearance(
    *,
    side: str,
    contact: bool,
    heel_clearance_m: float | None,
    toe_clearance_m: float | None,
    ankle_clearance_m: float | None,
    contact_foot_clearance_m: float | None,
    visibility: float,
    leg_length_m: float,
    entry_clearance_m: float | None = None,
    exit_clearance_m: float | None = None,
    max_display_exit_clearance_m: float | None = None,
    config: ContactClearanceConsistencyConfig | None = None,
) -> FootConsistencyAssessment:
    """Classify one foot at one frame."""
    cfg = config or ContactClearanceConsistencyConfig()
    disp_m = displayed_foot_clearance_m(heel_clearance_m, toe_clearance_m)
    inconsistent_m, borderline_m = _thresholds_m(
        leg_length_m,
        cfg,
        entry_clearance_m=entry_clearance_m,
        exit_clearance_m=exit_clearance_m,
        max_display_exit_clearance_m=max_display_exit_clearance_m,
    )

    if visibility < cfg.min_foot_visibility or disp_m is None:
        return FootConsistencyAssessment(
            side=side,
            verdict="LOW_CONFIDENCE",
            contact=contact,
            displayed_clearance_cm=disp_m * 100 if disp_m is not None else None,
            displayed_clearance_m=disp_m,
            contact_foot_clearance_cm=(
                contact_foot_clearance_m * 100 if contact_foot_clearance_m is not None else None
            ),
            inconsistent_threshold_cm=inconsistent_m * 100,
            borderline_threshold_cm=borderline_m * 100,
            visibility=visibility,
            reason="low landmark visibility or missing heel/toe",
        )

    disp_cm = disp_m * 100
    if not contact:
        return FootConsistencyAssessment(
            side=side,
            verdict="CONSISTENT",
            contact=False,
            displayed_clearance_cm=disp_cm,
            displayed_clearance_m=disp_m,
            contact_foot_clearance_cm=(
                contact_foot_clearance_m * 100 if contact_foot_clearance_m is not None else None
            ),
            inconsistent_threshold_cm=inconsistent_m * 100,
            borderline_threshold_cm=borderline_m * 100,
            visibility=visibility,
            reason="swing — clearance above contact band",
        )

    if disp_m >= inconsistent_m:
        return FootConsistencyAssessment(
            side=side,
            verdict="INCONSISTENT",
            contact=True,
            displayed_clearance_cm=disp_cm,
            displayed_clearance_m=disp_m,
            contact_foot_clearance_cm=(
                contact_foot_clearance_m * 100 if contact_foot_clearance_m is not None else None
            ),
            inconsistent_threshold_cm=inconsistent_m * 100,
            borderline_threshold_cm=borderline_m * 100,
            visibility=visibility,
            reason=(
                "contact retained while displayed min(heel,toe) clearance exceeds "
                "body-normalized exit band"
            ),
        )

    if disp_m >= borderline_m:
        ankle_note = ""
        if (
            ankle_clearance_m is not None
            and disp_m is not None
            and ankle_clearance_m + 0.008 < disp_m
        ):
            ankle_note = "; ankle lower than heel/toe (reference mismatch)"
        return FootConsistencyAssessment(
            side=side,
            verdict="BORDERLINE",
            contact=True,
            displayed_clearance_cm=disp_cm,
            displayed_clearance_m=disp_m,
            contact_foot_clearance_cm=(
                contact_foot_clearance_m * 100 if contact_foot_clearance_m is not None else None
            ),
            inconsistent_threshold_cm=inconsistent_m * 100,
            borderline_threshold_cm=borderline_m * 100,
            visibility=visibility,
            reason=f"contact hysteresis band{borderline_m * 100:.1f}–{inconsistent_m * 100:.1f} cm{ankle_note}",
        )

    return FootConsistencyAssessment(
        side=side,
        verdict="CONSISTENT",
        contact=True,
        displayed_clearance_cm=disp_cm,
        displayed_clearance_m=disp_m,
        contact_foot_clearance_cm=(
            contact_foot_clearance_m * 100 if contact_foot_clearance_m is not None else None
        ),
        inconsistent_threshold_cm=inconsistent_m * 100,
        borderline_threshold_cm=borderline_m * 100,
        visibility=visibility,
        reason="contact with displayed clearance within plausible stance band",
    )


DIAGNOSTIC_COLUMNS = [
    "frame",
    "timestamp_s",
    "left_displayed_clearance_cm",
    "right_displayed_clearance_cm",
    "left_heel_clearance_cm",
    "right_heel_clearance_cm",
    "left_toe_clearance_cm",
    "right_toe_clearance_cm",
    "left_ankle_clearance_cm",
    "right_ankle_clearance_cm",
    "left_contact_state",
    "right_contact_state",
    "left_contact_confidence",
    "right_contact_confidence",
    "gait_phase",
    "left_consistency",
    "right_consistency",
    "left_contact_foot_clearance_cm",
    "right_contact_foot_clearance_cm",
]


def build_contact_clearance_diagnostic_rows(
    recording: GaitMotionRecording,
    gait_result: GaitCycleAnalysisResult,
    *,
    config: ContactClearanceConsistencyConfig | None = None,
) -> list[dict[str, Any]]:
    """Per-frame diagnostic table aligned with gait analysis frame indices."""
    cfg = config or ContactClearanceConsistencyConfig()
    end_index = float(max(recording.frame_count - 1, 0))
    plane = gait_result.ground_plane or estimate_ground_plane(recording, end_index)
    axis = resolve_vertical_axis(plane)
    leg_length = estimate_leg_length_m(recording, axis=axis)

    thresholds = gait_result.contact_thresholds
    entry_m = thresholds.entry_clearance_m if thresholds else None
    exit_m = thresholds.exit_clearance_m if thresholds else None
    max_display_exit_m = (
        thresholds.max_display_exit_clearance_m if thresholds else None
    )

    left_series = build_filtered_foot_clearance_series(
        recording, plane, "left", end_frame_float=end_index
    )
    right_series = build_filtered_foot_clearance_series(
        recording, plane, "right", end_frame_float=end_index
    )
    left_by_frame = {s.frame_index: s for s in left_series.samples}
    right_by_frame = {s.frame_index: s for s in right_series.samples}

    global_conf = gait_result.metrics.contact_confidence
    rows: list[dict[str, Any]] = []

    for state in gait_result.per_frame:
        frame = state.frame_index
        left_f = state.left
        right_f = state.right

        left_disp_sample = left_by_frame.get(frame)
        right_disp_sample = right_by_frame.get(frame)
        left_disp_m = (
            left_disp_sample.filtered_clearance_m
            if left_disp_sample and left_disp_sample.is_valid
            else displayed_foot_clearance_m(
                left_f.heel_clearance_m, left_f.toe_clearance_m
            )
        )
        right_disp_m = (
            right_disp_sample.filtered_clearance_m
            if right_disp_sample and right_disp_sample.is_valid
            else displayed_foot_clearance_m(
                right_f.heel_clearance_m, right_f.toe_clearance_m
            )
        )

        left_assess = assess_foot_contact_clearance(
            side="left",
            contact=bool(state.left_contact),
            heel_clearance_m=left_f.heel_clearance_m,
            toe_clearance_m=left_f.toe_clearance_m,
            ankle_clearance_m=left_f.ankle_clearance_m,
            contact_foot_clearance_m=left_f.foot_clearance_m,
            visibility=left_f.visibility,
            leg_length_m=leg_length,
            entry_clearance_m=entry_m,
            exit_clearance_m=exit_m,
            max_display_exit_clearance_m=max_display_exit_m,
            config=cfg,
        )
        right_assess = assess_foot_contact_clearance(
            side="right",
            contact=bool(state.right_contact),
            heel_clearance_m=right_f.heel_clearance_m,
            toe_clearance_m=right_f.toe_clearance_m,
            ankle_clearance_m=right_f.ankle_clearance_m,
            contact_foot_clearance_m=right_f.foot_clearance_m,
            visibility=right_f.visibility,
            leg_length_m=leg_length,
            entry_clearance_m=entry_m,
            exit_clearance_m=exit_m,
            max_display_exit_clearance_m=max_display_exit_m,
            config=cfg,
        )

        def _cm(value: float | None) -> str | float:
            return round(value * 100, 2) if value is not None else ""

        rows.append(
            {
                "frame": frame,
                "timestamp_s": round(state.time_s, 4),
                "left_displayed_clearance_cm": _cm(left_disp_m),
                "right_displayed_clearance_cm": _cm(right_disp_m),
                "left_heel_clearance_cm": _cm(left_f.heel_clearance_m),
                "right_heel_clearance_cm": _cm(right_f.heel_clearance_m),
                "left_toe_clearance_cm": _cm(left_f.toe_clearance_m),
                "right_toe_clearance_cm": _cm(right_f.toe_clearance_m),
                "left_ankle_clearance_cm": _cm(left_f.ankle_clearance_m),
                "right_ankle_clearance_cm": _cm(right_f.ankle_clearance_m),
                "left_contact_state": "CONTACT" if state.left_contact else "SWING",
                "right_contact_state": "CONTACT" if state.right_contact else "SWING",
                "left_contact_confidence": round(left_f.visibility * global_conf, 3),
                "right_contact_confidence": round(right_f.visibility * global_conf, 3),
                "gait_phase": state.phase,
                "left_consistency": left_assess.verdict,
                "right_consistency": right_assess.verdict,
                "left_contact_foot_clearance_cm": _cm(left_f.foot_clearance_m),
                "right_contact_foot_clearance_cm": _cm(right_f.foot_clearance_m),
            }
        )
    return rows


def find_inconsistent_frames(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Frames where contact=True and displayed clearance exceeds threshold."""
    hits: list[dict[str, Any]] = []
    for row in rows:
        for side in ("left", "right"):
            contact_key = f"{side}_contact_state"
            disp_key = f"{side}_displayed_clearance_cm"
            verdict_key = f"{side}_consistency"
            if row.get(contact_key) != "CONTACT":
                continue
            if row.get(verdict_key) == "INCONSISTENT":
                hits.append({**row, "inconsistent_side": side})
    return hits


def pipeline_documentation_lines() -> list[str]:
    """Numbered audit documentation for reports."""
    return [
        "1. Displayed Foot Clearance foot reference: "
        + DISPLAYED_CLEARANCE_FOOT_REFERENCE,
        "2. Contact Detection foot reference: " + CONTACT_DETECTION_FOOT_REFERENCE,
        "3. Displayed clearance uses minimum heel/toe height: yes (ankle excluded)",
        "4. Contact Detection uses: heel, toe, and ankle (minimum of all three)",
        "5. Both use the same floor reference: " + SHARED_FLOOR_REFERENCE,
        "6. Both use the same canonical vertical axis: " + SHARED_VERTICAL_AXIS,
        "7. Both use the same units: " + SHARED_UNITS,
    ]
