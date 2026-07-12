"""Tests for knee chart interpretation and angle convention helpers."""

from __future__ import annotations

import numpy as np

from stablewalk.ui.viewers.knee_angle_chart import (
    KneeAngleSeries,
    _interior_to_flexion_deg,
    knee_flexion_rom_deg,
    largest_frame_jump_deg,
)
from stablewalk.ui.viewers.knee_chart_interpretation import (
    MIN_CYCLES_FOR_CYCLE_MODE,
    build_knee_motion_summary,
    cycle_mode_is_available,
    format_interpretation_panel,
    usable_knee_cycle_count,
)


def test_interior_to_flexion_zero_is_extension():
    assert _interior_to_flexion_deg(180.0) == 0.0
    assert _interior_to_flexion_deg(90.0) == 90.0


def test_rom_and_frame_jump():
    vals = np.array([10.0, 20.0, np.nan, 25.0, 15.0])
    assert knee_flexion_rom_deg(vals) == 15.0
    assert largest_frame_jump_deg(vals) == 10.0


def test_cycle_mode_requires_two_cycles():
    from stablewalk.analysis.gait_feature_analysis import (
        BodySegmentDimensions,
        CycleConsistencyResult,
        CycleTrajectory,
        GaitFeatureAnalysisResult,
        NormalizedGaitFeatures,
    )

    traj = CycleTrajectory(
        name="left_knee_angle",
        percent=tuple(float(x) for x in range(101)),
        mean=tuple(30.0 for _ in range(101)),
        std=tuple(2.0 for _ in range(101)),
        per_cycle=[tuple(30.0 for _ in range(101))],
    )
    cc_one = CycleConsistencyResult(
        trajectories={"left_knee_angle": traj},
        cycle_count=1,
    )
    dims = BodySegmentDimensions(
        hip_width=1.0,
        shoulder_width=1.0,
        leg_length_left=1.0,
        leg_length_right=1.0,
        leg_length_average=1.0,
        thigh_length_left=1.0,
        thigh_length_right=1.0,
        shank_length_left=1.0,
        shank_length_right=1.0,
    )
    gf_one = GaitFeatureAnalysisResult(
        dimensions=dims,
        features=NormalizedGaitFeatures(),
        cycle_consistency=cc_one,
    )
    assert usable_knee_cycle_count(gf_one) == 1
    assert not cycle_mode_is_available(gf_one)

    traj_two = CycleTrajectory(
        name="left_knee_angle",
        percent=tuple(float(x) for x in range(101)),
        mean=tuple(30.0 for _ in range(101)),
        std=tuple(2.0 for _ in range(101)),
        per_cycle=[
            tuple(30.0 for _ in range(101)),
            tuple(32.0 for _ in range(101)),
        ],
    )
    cc_two = CycleConsistencyResult(
        trajectories={"left_knee_angle": traj_two},
        cycle_count=2,
        cycle_repeatability_score=82.0,
    )
    gf_two = GaitFeatureAnalysisResult(
        dimensions=dims,
        features=NormalizedGaitFeatures(),
        cycle_consistency=cc_two,
    )
    assert usable_knee_cycle_count(gf_two) == MIN_CYCLES_FOR_CYCLE_MODE
    assert cycle_mode_is_available(gf_two)


def test_interpretation_panel_plain_language():
    series = KneeAngleSeries(
        times_s=np.array([0.0, 1.0]),
        left_deg=np.array([5.0, 60.0]),
        right_deg=np.array([8.0, 55.0]),
        source="pose_derived",
        angle_definition="Pose interior angle -> flexion (180 deg - theta); 0 deg = extension",
        fps=30.0,
        metadata={"nan_pct": 0.0, "left_valid": 2, "right_valid": 2},
    )
    summary = build_knee_motion_summary(series, None, chart_mode="video_time")
    text = format_interpretation_panel(summary)
    assert "KNEE MOTION SUMMARY" in text
    assert "Left ROM:" in text
    assert "Right ROM:" in text
    assert summary.left_rom_deg == 55.0
    assert summary.right_rom_deg == 47.0
