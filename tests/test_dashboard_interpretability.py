"""Tests for dashboard interpretability helpers."""

from __future__ import annotations

import numpy as np

from stablewalk.models.gait_motion import Vec3
from stablewalk.ui.dashboard_interpretability import (
    compute_trajectory_path_metrics,
    default_projection_for_view,
    format_analysis_confidence_level,
    format_compact_interpretation,
    format_score_over_100,
    gait_quality_evidence_badge,
    interpret_knee_motion,
    CompactInterpretation,
)
from stablewalk.ui.viewers.knee_angle_chart import KneeAngleSeries
from stablewalk.ui.viewers.knee_chart_interpretation import build_knee_motion_summary


def test_compact_interpretation_format():
    card = CompactInterpretation(
        name="Foot Clearance",
        value="L 2.1 cm | R 4.8 cm",
        sentence="Right swing clearance is higher than left clearance.",
        confidence="Moderate",
    )
    text = format_compact_interpretation(card)
    assert "Foot Clearance" in text
    assert "Confidence: Moderate" in text
    assert "Right swing clearance" in text


def test_default_projection_for_view():
    assert default_projection_for_view("SAGITTAL_LEFT") == "Sagittal Plane"
    assert default_projection_for_view("FRONTAL") == "Frontal Plane"
    assert default_projection_for_view("OBLIQUE") == "3D"


def test_movement_path_title():
    from stablewalk.ui.dashboard_interpretability import movement_path_title

    assert movement_path_title("Right Knee") == "Right Knee 3D Movement Path"


def test_evaluate_trajectory_readiness_insufficient():
    from stablewalk.ui.dashboard_interpretability import evaluate_trajectory_readiness

    readiness = evaluate_trajectory_readiness([])
    assert not readiness.sufficient
    assert "Fewer than 2" in readiness.reason


def test_format_trajectory_confidence():
    from stablewalk.ui.dashboard_interpretability import format_trajectory_confidence

    assert format_trajectory_confidence("High") == "HIGH"
    assert format_trajectory_confidence("Insufficient") == "INSUFFICIENT"


def test_trajectory_path_metrics():
    path = [
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.0, 0.05, 0.1),
        Vec3(0.0, 0.1, 0.2),
    ]
    m = compute_trajectory_path_metrics(path, projection="Sagittal Plane")
    assert m is not None
    assert m.total_travel_m > 0.1


def test_format_score_over_100():
    assert format_score_over_100(91.2) == "91 / 100"
    assert format_score_over_100(None) == "—"


def test_format_analysis_confidence_level():
    assert format_analysis_confidence_level("HIGH") == "High"
    assert format_analysis_confidence_level("INSUFFICIENT") == "INSUFFICIENT"


def test_interpret_knee_motion():
    series = KneeAngleSeries(
        times_s=np.array([0.0, 1.0]),
        left_deg=np.array([5.0, 60.0]),
        right_deg=np.array([8.0, 55.0]),
        source="pose_derived",
        angle_definition="flexion",
        fps=30.0,
        metadata={"nan_pct": 0.0},
    )
    summary = build_knee_motion_summary(series, None, chart_mode="video_time")
    card = interpret_knee_motion(summary)
    assert card.name == "Knee Motion"
    assert "ROM" in card.value
