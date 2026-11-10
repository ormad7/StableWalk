"""Gait-event detection integrity: no fabricated cycles; confidence caps."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.analysis.gait_analysis_summary import (
    _analysis_confidence_level,
    build_gait_analysis_summary,
)
from stablewalk.analysis.gait_cycle_analysis import (
    FootLandmarkSample,
    SATURATED_CONTACT_FRAC,
    _contact_partition_untrustworthy,
    classify_event_sufficiency,
    detect_kinematic_foot_events,
    kinematic_supplement_allowed,
    merge_gait_events,
    GaitEvent,
)
from stablewalk.analysis.stability_scoring import StabilityResult
from stablewalk.ui.overview_clinical_summary import extract_confidence_visualization


def _sample(clearance: float, vis: float = 0.9) -> FootLandmarkSample:
    return FootLandmarkSample(
        heel_clearance_m=clearance,
        toe_clearance_m=clearance + 0.004,
        ankle_clearance_m=clearance + 0.008,
        foot_clearance_m=clearance,
        heel_velocity_m_s=0.0,
        toe_velocity_m_s=0.0,
        ankle_velocity_m_s=0.0,
        visibility=vis,
    )


def test_kinematic_events_match_labeled_periodic_minima() -> None:
    fps = 50.0
    n = 200
    times = [i / fps for i in range(n)]
    frames = list(range(n))
    period = 1.0
    samples = []
    for t in times:
        phase = t % period
        # Triangle: min 0.01 m at phase 0, max 0.08 m at phase 0.5.
        clearance = 0.01 + 0.07 * (phase if phase <= 0.5 else (1.0 - phase)) * 2.0
        samples.append(_sample(clearance))

    events = detect_kinematic_foot_events(
        samples, times, frames, side="left", min_prominence_m=0.02
    )
    hs = [e for e in events if e.event_type == "left_heel_strike"]
    to = [e for e in events if e.event_type == "left_toe_off"]
    assert len(hs) >= 2
    # First detected HS near a period boundary (0 or 1.0 s), ±1 frame.
    assert min(abs(hs[0].time_s - k) for k in (0.0, 1.0, 2.0, 3.0)) <= 0.06
    # Toe-off is leave-floor after HS, not the mid-swing clearance peak (phase 0.5).
    assert to
    mid_swing = [k + 0.5 for k in (0.0, 1.0, 2.0, 3.0)]
    for event in to:
        assert min(abs(event.time_s - m) for m in mid_swing) > 0.12


def test_saturated_contact_does_not_allow_kinematic_supplement() -> None:
    assert kinematic_supplement_allowed(0.53) is True
    assert kinematic_supplement_allowed(SATURATED_CONTACT_FRAC) is True
    assert kinematic_supplement_allowed(0.94) is False
    assert kinematic_supplement_allowed(float("nan")) is False


def test_flat_clearance_does_not_fabricate_events() -> None:
    fps = 30.0
    n = 90
    times = [i / fps for i in range(n)]
    samples = [_sample(0.015) for _ in range(n)]
    events = detect_kinematic_foot_events(
        samples, times, list(range(n)), side="right", min_prominence_m=0.012
    )
    assert events == []


def test_merge_does_not_duplicate_contact_events() -> None:
    primary = [GaitEvent("left_heel_strike", 10, 0.40, "left")]
    extra = [
        GaitEvent("left_heel_strike", 11, 0.44, "left"),
        GaitEvent("left_toe_off", 20, 0.80, "left"),
    ]
    merged = merge_gait_events(primary, extra)
    hs = [e for e in merged if e.event_type == "left_heel_strike"]
    assert len(hs) == 1
    assert any(e.event_type == "left_toe_off" for e in merged)


def test_short_clip_is_input_insufficient_not_processing_failure() -> None:
    kind, reason = classify_event_sufficiency(
        frame_count=120,
        duration_s=0.55,
        ground_plane=object(),
        heel_strike_count=0,
        toe_off_count=0,
        cycle_count=0,
        foot_visibility_mean=0.7,
        warnings=[],
    )
    assert kind == "input_insufficient"
    assert "Not a processing crash" in reason


def test_missing_ground_plane_is_processing_failure() -> None:
    kind, _reason = classify_event_sufficiency(
        frame_count=80,
        duration_s=4.0,
        ground_plane=None,
        heel_strike_count=0,
        toe_off_count=0,
        cycle_count=0,
        foot_visibility_mean=0.8,
        warnings=[],
    )
    assert kind == "processing_failure"


def test_analysis_confidence_not_high_without_cycles() -> None:
    result = StabilityResult(
        score=70.0,
        classification="Moderate",
        metrics=[],
        primary_issue=None,
        explanation="",
        frame_count=120,
        completeness_pct=90.0,
        confidence_badge="HIGH CONFIDENCE",
        usable_gait_cycles=0,
    )
    level = _analysis_confidence_level(
        result, evidence_index=0.80, validity_status="PROVISIONAL"
    )
    assert level == "INSUFFICIENT"
    summary = build_gait_analysis_summary(result)
    assert summary.analysis_confidence.level != "HIGH"
    assert summary.analysis_confidence.usable_gait_cycles == 0


def test_overview_confidence_display_insufficient_when_no_cycles() -> None:
    ac = SimpleNamespace(
        level="INSUFFICIENT",
        completeness_pct=88.0,
        evidence_index=0.7,
        pose_valid_frame_pct=92.0,
    )
    result = SimpleNamespace(gait_summary=SimpleNamespace(analysis_confidence=ac), usable_gait_cycles=0)
    gui = SimpleNamespace(
        _biomech_analysis=None,
        _stability_result=result,
        _last_stability_result=None,
        _resolved_gait_cycle_count=lambda: (0, 0),
    )
    conf = extract_confidence_visualization(gui)
    assert conf["display"] == "Insufficient"
    assert conf["pct"] is None or float(conf["pct"]) <= 35.0


def test_support_sum_warning_does_not_withhold_cadence() -> None:
    assert not _contact_partition_untrustworthy(
        [
            "Double support + single support + flight is 76.81%, "
            "not approximately 100%."
        ]
    )
    assert _contact_partition_untrustworthy(
        [
            "94% of foot-frames are in contact (>85%) — "
            "floor may be set too high or swing clearance is underestimated."
        ]
    )
