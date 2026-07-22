from __future__ import annotations

import pytest

from stablewalk.analysis.biomechanical.base_of_support import BaseOfSupportAnalysis
from stablewalk.analysis.biomechanical.advanced_gait_metrics import analyze_advanced_gait_metrics
from stablewalk.analysis.biomechanical.com_estimation import CenterOfMassAnalysis
from stablewalk.analysis.biomechanical.joint_rom import _cycle_angle_values, _rom_stats
from stablewalk.analysis.biomechanical.stability_margin import analyze_stability_margin
from stablewalk.analysis.biomechanical.types import BiomechanicalFrameBoS, BiomechanicalFrameCOM
from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    FootLandmarkSample,
    FrameContactState,
    GaitCycleAnalysisResult,
    GaitEvent,
    GaitTemporalMetrics,
    compute_temporal_metrics,
    detect_gait_cycles,
)
from stablewalk.analysis.gait_feature_analysis import symmetry_index
from stablewalk.io.session_pdf_report import (
    SessionPdfReportContext,
    build_validation_warnings,
)
from stablewalk.models.gait_motion import GaitMotionRecording


def _sample() -> FootLandmarkSample:
    return FootLandmarkSample(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _two_cycle_case(*, abnormal: bool = False):
    fps = 10.0
    frames: list[FrameContactState] = []
    for index in range(21):
        time_s = index / fps
        phase_index = index % 10
        if abnormal:
            left, right = int(phase_index < 9), int(phase_index >= 1)
        else:
            left = int(phase_index < 6)
            right = int(phase_index < 1 or phase_index >= 5)
        frames.append(
            FrameContactState(
                index,
                time_s,
                left,
                right,
                "DOUBLE_SUPPORT" if left and right else "LEFT_STANCE",
                _sample(),
                _sample(),
            )
        )
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("right_heel_strike", 5, 0.5, "right"),
        GaitEvent("left_heel_strike", 10, 1.0, "left"),
        GaitEvent("right_heel_strike", 15, 1.5, "right"),
        GaitEvent("left_heel_strike", 20, 2.0, "left"),
    ]
    cycles = detect_gait_cycles(events, duration_s=2.0)
    metrics = compute_temporal_metrics(
        frames,
        events,
        cycles,
        fps=fps,
        contact_confidence=0.85,
        confidence_tier="HIGH",
    )
    return metrics, cycles


def test_normal_video_temporal_percentages_partition_complete_cycles() -> None:
    metrics, cycles = _two_cycle_case()
    assert len(cycles) == 2
    assert metrics.metrics_reliable
    assert metrics.left_stance_pct + metrics.left_swing_pct == pytest.approx(100.0)
    assert metrics.right_stance_pct + metrics.right_swing_pct == pytest.approx(100.0)
    assert metrics.double_support_pct == pytest.approx(20.0)
    # Ipsilateral single support (mean of L-only and R-only) ≈ 40%, not L+R ≈ 80%.
    assert metrics.single_support_pct == pytest.approx(40.0)
    assert metrics.flight_pct == pytest.approx(0.0)
    assert metrics.cadence_steps_per_min == pytest.approx(120.0)


def test_abnormal_video_preserves_unusual_but_valid_values() -> None:
    metrics, _ = _two_cycle_case(abnormal=True)
    assert metrics.metrics_reliable
    assert metrics.left_stance_pct == pytest.approx(90.0)
    assert metrics.right_stance_pct == pytest.approx(90.0)
    assert metrics.double_support_pct == pytest.approx(80.0)
    assert 0.0 <= metrics.single_support_pct <= 100.0


def test_short_video_returns_unreliable_na_metrics() -> None:
    metrics = compute_temporal_metrics(
        [],
        [],
        [],
        fps=30.0,
        contact_confidence=0.9,
        confidence_tier="HIGH",
    )
    assert not metrics.metrics_reliable
    assert metrics.cadence_steps_per_min is None
    assert "No contact-state frames" in metrics.reliability_reason


def test_incomplete_video_does_not_promote_steps_to_cycles() -> None:
    events = [
        GaitEvent("left_heel_strike", 0, 0.0, "left"),
        GaitEvent("right_heel_strike", 15, 0.5, "right"),
    ]
    assert detect_gait_cycles(events, duration_s=1.0) == []


def test_low_confidence_cycles_are_not_final_metrics() -> None:
    metrics, cycles = _two_cycle_case()
    low = compute_temporal_metrics(
        [
            FrameContactState(
                i,
                i / 10.0,
                int((i % 10) < 6),
                int((i % 10) < 1 or (i % 10) >= 5),
                "LEFT_STANCE",
                _sample(),
                _sample(),
            )
            for i in range(21)
        ],
        [
            GaitEvent("left_heel_strike", 0, 0.0, "left"),
            GaitEvent("right_heel_strike", 5, 0.5, "right"),
            GaitEvent("left_heel_strike", 10, 1.0, "left"),
            GaitEvent("right_heel_strike", 15, 1.5, "right"),
            GaitEvent("left_heel_strike", 20, 2.0, "left"),
        ],
        cycles,
        fps=10.0,
        contact_confidence=0.30,
        confidence_tier="LOW_CONFIDENCE",
    )
    assert metrics.metrics_reliable
    assert not low.metrics_reliable
    assert low.double_support_pct is None


def test_rom_uses_complete_cycles_and_excludes_outside_extreme() -> None:
    cycles = GaitCycleAnalysisResult(
        cycles=[
            DetectedGaitCycle(0, 0, 9, 0.0, 0.9, 0.9, "left"),
            DetectedGaitCycle(1, 10, 19, 1.0, 1.9, 0.9, "left"),
        ],
        metrics=GaitTemporalMetrics(metrics_reliable=True),
    )
    angles = [float(i % 10) for i in range(20)] + [170.0]
    grouped = _cycle_angle_values(angles, cycles, list(range(21)))
    stats = _rom_stats(grouped, joint="knee", side="left", confidence=0.8)
    assert stats.rom_deg == pytest.approx(9.0)


def test_negative_symmetry_input_is_invalid_not_absolute() -> None:
    assert symmetry_index(-0.4, 0.5) is None


def test_stability_summary_is_na_when_valid_frame_coverage_is_low() -> None:
    com = CenterOfMassAnalysis(
        per_frame=[
            BiomechanicalFrameCOM(i, i / 30.0, (0.0, 1.0, 0.0), (0, 0, 0), (0, 0, 0), 0.9)
            for i in range(4)
        ]
    )
    bos = BaseOfSupportAnalysis(
        per_frame=[
            BiomechanicalFrameBoS(
                0,
                0.0,
                "double_support",
                [(-0.2, -0.2), (0.2, -0.2), (0.2, 0.2), (-0.2, 0.2)],
                (0.0, 0.0),
                0.16,
                0.9,
            )
        ]
    )
    result = analyze_stability_margin(com, bos)
    assert result.stable_pct is None
    assert result.mean_margin_m is None
    assert result.warnings


def test_session_report_collects_metric_validation_warnings() -> None:
    cycles = GaitCycleAnalysisResult(warnings=["Incomplete gait cycles excluded."])
    warnings = build_validation_warnings(SessionPdfReportContext(cycles=cycles))
    assert warnings == ["Incomplete gait cycles excluded."]


def test_unreliable_final_metrics_carry_na_explanation() -> None:
    cycles = GaitCycleAnalysisResult(
        metrics=GaitTemporalMetrics(
            metrics_reliable=False,
            reliability_reason="Only one complete gait cycle.",
        )
    )
    result = analyze_advanced_gait_metrics(
        GaitMotionRecording(source="short.mp4", fps=30.0, snapshots=[]),
        cycles,
        None,
        None,
    )
    assert result.cadence is not None and result.cadence.value is None
    assert "Only one complete gait cycle" in result.cadence.note
