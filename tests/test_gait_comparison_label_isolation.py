"""Tests that gait comparison and stability scoring are label-blind."""

from __future__ import annotations

import copy
import inspect
from unittest.mock import patch

import pytest

from stablewalk import config
from stablewalk.analysis.gait_comparison_validation import (
    COMPARISON_VIDEOS,
    analyze_motion_label_blind,
)
from stablewalk.analysis.stability_scoring import (
    FORBIDDEN_STABILITY_PARAMETERS,
    analyze_biomech_stability,
    verify_stability_api_label_free,
)
from stablewalk.io.pose_loader import load_pose_sequence


def test_stability_api_has_no_label_parameters():
    verify_stability_api_label_free()
    sig = inspect.signature(analyze_biomech_stability)
    assert not FORBIDDEN_STABILITY_PARAMETERS.intersection(sig.parameters)


def test_analyze_motion_label_blind_rejects_label_kwargs():
    poses = config.POSES_DIR / "normal_gait_poses.json"
    if not poses.is_file():
        pytest.skip("normal_gait_poses.json not available")
    seq = load_pose_sequence(poses)
    with pytest.raises(TypeError):
        analyze_motion_label_blind(seq, gait_category="normal")  # type: ignore[call-arg]


def test_stability_scoring_rejects_label_kwargs():
    poses = config.POSES_DIR / "normal_gait_poses.json"
    if not poses.is_file():
        pytest.skip("normal_gait_poses.json not available")
    seq = load_pose_sequence(poses)
    with pytest.raises(TypeError):
        analyze_biomech_stability(seq, ground_truth_label="normal")  # type: ignore[call-arg]


def test_comparison_pipeline_does_not_pass_labels_to_scoring():
    poses = config.POSES_DIR / "abnormal_gait_poses.json"
    if not poses.is_file():
        pytest.skip("abnormal_gait_poses.json not available")
    seq = load_pose_sequence(poses)

    with patch(
        "stablewalk.analysis.gait_comparison_validation.analyze_biomech_stability",
        wraps=analyze_biomech_stability,
    ) as mock_score:
        analyze_motion_label_blind(seq)
        assert mock_score.call_count == 1
        _args, kwargs = mock_score.call_args
        assert hasattr(_args[0], "frames")
        assert "gait_category" not in kwargs
        assert "ground_truth_label" not in kwargs
        assert "demo_key" not in kwargs
        assert "category" not in kwargs


def test_score_invariant_when_source_video_anonymized():
    poses = config.POSES_DIR / "normal_gait_poses.json"
    if not poses.is_file():
        pytest.skip("normal_gait_poses.json not available")

    seq = load_pose_sequence(poses)
    anon = copy.deepcopy(seq)
    anon.source_video = "anonymous_validation_clip.mp4"
    r1 = analyze_biomech_stability(seq)
    r2 = analyze_biomech_stability(anon)
    assert r1.score == pytest.approx(r2.score)
    assert r1.classification == r2.classification
