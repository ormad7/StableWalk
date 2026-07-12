"""Unit tests for pose / HMR backend registry and canonical conversion."""

from __future__ import annotations

import numpy as np
import pytest

from stablewalk import config
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.pose.backends.base import BackendUnavailableError
from stablewalk.pose.backends.canonical import (
    canonical_positions_from_landmarks,
    human_motion_sequence_to_pose_sequence,
    map_landmark_name_to_canonical,
)
from stablewalk.pose.backends.environment import inspect_runtime_environment
from stablewalk.pose.backends.mediapipe_backend import (
    MEDIAPIPE_COORDINATE_SYSTEM,
    MediaPipePoseBackend,
)
from stablewalk.pose.backends.registry import (
    BACKEND_REGISTRY,
    create_pose_backend,
    list_backend_diagnostics,
    unavailable_message,
)
from stablewalk.pose.backends.types import HumanMotionFrame, HumanMotionSequence
from stablewalk.pose.backends.comparison import compute_backend_metrics


def test_mediapipe_backend_available():
    assert MediaPipePoseBackend.is_available()


def test_create_default_backend_is_mediapipe():
    backend = create_pose_backend("mediapipe")
    assert backend.name == "mediapipe"


def test_unavailable_backend_raises_without_fallback():
    with pytest.raises(BackendUnavailableError):
        create_pose_backend("wham", allow_fallback=False)


def test_unavailable_backend_fallback_to_mediapipe():
    backend = create_pose_backend("wham", allow_fallback=True)
    assert backend.name == "mediapipe"


def test_unavailable_message_format():
    msg = unavailable_message("wham")
    assert "Backend unavailable" in msg
    assert "WHAM" in msg


def test_landmark_to_canonical_mapping():
    assert map_landmark_name_to_canonical("left_foot_index") == "left_toe"
    assert map_landmark_name_to_canonical("mid_hip") == ROOT_JOINT_ID


def test_canonical_positions_derive_pelvis():
    lm = {
        "left_hip": (0.4, 0.6, 0.0),
        "right_hip": (0.6, 0.6, 0.0),
    }
    pos, conf = canonical_positions_from_landmarks(lm)
    assert ROOT_JOINT_ID in pos
    assert abs(pos[ROOT_JOINT_ID][0] - 0.5) < 1e-6


def test_environment_inspection_runs():
    report = inspect_runtime_environment()
    assert report.python_version
    assert any(d.name == "mediapipe" for d in report.dependencies)


def test_backend_diagnostics_lists_all():
    diag = list_backend_diagnostics()
    assert len(diag) == len(BACKEND_REGISTRY)
    names = {d["name"] for d in diag}
    assert "mediapipe" in names
    assert "wham" in names


def test_human_motion_to_pose_sequence_bridge():
    coord = MEDIAPIPE_COORDINATE_SYSTEM
    frame = HumanMotionFrame(
        frame_index=0,
        timestamp_s=0.0,
        joint_positions_3d={"left_hip": (0.4, 0.5, 0.0), "right_hip": (0.6, 0.5, 0.0)},
        landmark_confidence={"left_hip": 0.9, "right_hip": 0.9},
        backend_name="mediapipe",
        coordinate_system=coord,
        detected=True,
        raw_landmarks={"left_hip": (0.4, 0.5, 0.0), "right_hip": (0.6, 0.5, 0.0)},
    )
    seq = HumanMotionSequence(
        frames=[frame],
        fps=30.0,
        source_video="test.mp4",
        backend_name="mediapipe",
        coordinate_system=coord,
    )
    pose_seq = human_motion_sequence_to_pose_sequence(seq)
    assert len(pose_seq.frames) == 1
    assert pose_seq.frames[0].detected


def test_comparison_metrics_for_empty_backend():
    m = compute_backend_metrics(
        None,
        backend_name="wham",
        available=False,
        availability_message="not installed",
    )
    assert not m.available
    assert m.valid_frame_ratio == 0.0


def test_config_defaults():
    assert config.POSE_BACKEND == "mediapipe" or config.POSE_BACKEND in BACKEND_REGISTRY
