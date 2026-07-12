"""Tests for gait-cycle evidence thresholds and stability integration."""

from __future__ import annotations

from stablewalk.analysis.gait_cycle_analysis import (
    DetectedGaitCycle,
    GaitCycleAnalysisResult,
    GaitTemporalMetrics,
)
from stablewalk.analysis.gait_evidence import (
    GaitEvidenceThresholds,
    assess_gait_evidence,
)
from stablewalk.models.pose_data import PoseFrame, PoseSequence


def _cycle(index: int, *, start: int, end: int, fps: float = 30.0) -> DetectedGaitCycle:
    t0 = start / fps
    t1 = end / fps
    return DetectedGaitCycle(
        cycle_index=index,
        start_frame=start,
        end_frame=end,
        start_time_s=t0,
        end_time_s=t1,
        duration_s=t1 - t0,
    )


def _sequence(*, frames: int, fps: float, detected: int | None = None) -> PoseSequence:
    det = detected if detected is not None else frames
    return PoseSequence(
        source_video="synthetic_test.mp4",
        fps=fps,
        frames=[
            PoseFrame(
                frame_index=i,
                image_path=f"f{i}.jpg",
                timestamp_s=i / fps,
                detected=i < det,
            )
            for i in range(frames)
        ],
    )


def _cycles_result(
    *,
    cycles: list[DetectedGaitCycle],
    left_hs: int = 3,
    right_hs: int = 3,
) -> GaitCycleAnalysisResult:
    return GaitCycleAnalysisResult(
        cycles=cycles,
        metrics=GaitTemporalMetrics(
            left_heel_strike_count=left_hs,
            right_heel_strike_count=right_hs,
            gait_cycle_count=len(cycles),
            contact_confidence=0.85,
        ),
        per_frame=[],
    )


def test_repeatability_tier_thresholds():
    th = GaitEvidenceThresholds()
    assert th.repeatability_tier(0) == "UNAVAILABLE"
    assert th.repeatability_tier(1) == "UNAVAILABLE"
    assert th.repeatability_tier(2) == "LOW_CONFIDENCE"
    assert th.repeatability_tier(3) == "NORMAL"


def test_partial_cycle_when_spanning_clip():
    th = GaitEvidenceThresholds()
    seq = _sequence(frames=72, fps=30.0)  # 2.4 s
    cycles = _cycles_result(
        cycles=[_cycle(0, start=0, end=71, fps=30.0)],
        left_hs=1,
    )
    assessment = assess_gait_evidence(seq, cycles, stability_frame_count=60, thresholds=th)
    assert assessment.cycles.partial_cycles == 1
    assert assessment.cycles.usable_cycles == 0
    assert assessment.domain_evidence["cycle_consistency"].availability_hint == "UNAVAILABLE"


def test_usable_cycles_enable_cycle_consistency():
    th = GaitEvidenceThresholds()
    seq = _sequence(frames=150, fps=30.0)  # 5 s
    cycles = _cycles_result(
        cycles=[
            _cycle(0, start=0, end=29, fps=30.0),
            _cycle(1, start=30, end=59, fps=30.0),
        ],
    )
    assessment = assess_gait_evidence(seq, cycles, stability_frame_count=120, thresholds=th)
    assert assessment.cycles.complete_cycles == 2
    assert assessment.cycles.usable_cycles == 2
    assert assessment.repeatability_tier == "LOW_CONFIDENCE"
    assert assessment.domain_evidence["cycle_consistency"].availability_hint == "LOW_CONFIDENCE"


def test_video_inventory_fields():
    th = GaitEvidenceThresholds()
    seq = _sequence(frames=119, fps=25.0, detected=100)
    cycles = _cycles_result(cycles=[_cycle(0, start=0, end=24, fps=25.0)])
    assessment = assess_gait_evidence(seq, cycles, stability_frame_count=100, thresholds=th)
    video = assessment.video
    assert video.total_frames == 119
    assert video.valid_pose_frames == 100
    assert video.fps == 25.0
    assert video.duration_s == 119 / 25.0
