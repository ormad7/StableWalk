"""Rancho gait-cycle phase classification and frame-display precompute."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.analysis.gait_cycle_phase import (
    GAIT_CYCLE_PHASE_ORDER,
    classify_gait_cycle_phase,
    estimate_cycle_percent_from_hs_times,
    format_gait_cycle_phase,
)
from stablewalk.ui.session_analysis_cache import build_gait_frame_display


def test_phase_order_matches_requested_labels() -> None:
    labels = [format_gait_cycle_phase(k) for k in GAIT_CYCLE_PHASE_ORDER]
    assert labels == [
        "Heel Strike",
        "Loading Response",
        "Mid Stance",
        "Terminal Stance",
        "Pre Swing",
        "Initial Swing",
        "Mid Swing",
        "Terminal Swing",
    ]


def test_classify_bins_across_cycle() -> None:
    assert classify_gait_cycle_phase(0.0) == "heel_strike"
    assert classify_gait_cycle_phase(5.0) == "loading_response"
    assert classify_gait_cycle_phase(20.0) == "mid_stance"
    assert classify_gait_cycle_phase(40.0) == "terminal_stance"
    assert classify_gait_cycle_phase(55.0) == "pre_swing"
    assert classify_gait_cycle_phase(70.0) == "initial_swing"
    assert classify_gait_cycle_phase(80.0) == "mid_swing"
    assert classify_gait_cycle_phase(95.0) == "terminal_swing"
    assert classify_gait_cycle_phase(50.0, heel_strike_pulse=True) == "heel_strike"
    assert classify_gait_cycle_phase(None) is None


def test_cycle_percent_from_hs_times() -> None:
    hs = [0.0, 1.0, 2.0]
    assert estimate_cycle_percent_from_hs_times(0.0, hs) == 0.0
    assert estimate_cycle_percent_from_hs_times(0.5, hs) == 50.0
    # Boundary lands in the first enclosing interval (end of stride → 100%).
    assert estimate_cycle_percent_from_hs_times(1.0, hs) == 100.0
    # Mid of second stride.
    assert estimate_cycle_percent_from_hs_times(1.25, hs) == 25.0


def test_build_gait_frame_display_includes_cycle_phase() -> None:
    events = [
        SimpleNamespace(event_type="left_heel_strike", time_s=0.0, frame_index=0),
        SimpleNamespace(event_type="left_heel_strike", time_s=1.0, frame_index=10),
    ]
    per_frame = [
        SimpleNamespace(
            frame_index=i,
            time_s=i * 0.1,
            left_contact=1 if i < 6 else 0,
            right_contact=0 if i < 6 else 1,
            left=SimpleNamespace(foot_clearance_m=0.01),
            right=SimpleNamespace(foot_clearance_m=0.05),
        )
        for i in range(11)
    ]
    gait = SimpleNamespace(
        per_frame=per_frame,
        events=events,
        metrics=SimpleNamespace(contact_confidence=0.9, confidence_tier="HIGH"),
    )
    out = build_gait_frame_display(gait, format_dashboard_phase=lambda p: p)
    assert 5 in out
    snap = out[5]
    assert snap.cycle_percent is not None
    assert snap.cycle_phase in GAIT_CYCLE_PHASE_ORDER
    assert snap.cycle_phase_label == format_gait_cycle_phase(snap.cycle_phase)
    assert out[0].cycle_phase == "heel_strike"
