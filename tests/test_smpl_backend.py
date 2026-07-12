"""Tests for SMPL validation, unified motion, and backend resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stablewalk.pose.backends.pipeline_runner import (
    BackendResolution,
    normalize_backend_mode,
    resolve_pose_backend,
)
from stablewalk.pose.backends.smpl_backend import SMPLPoseBackend
from stablewalk.pose.backends.smpl_validation import (
    SmplValidationResult,
    validate_smpl_assets,
)
from stablewalk.pose.backends.unified_motion import (
    UnifiedHumanMotion,
    UnifiedMotionFrame,
    human_motion_sequence_to_unified,
)
from stablewalk.pose.backends.mediapipe_backend import (
    MEDIAPIPE_COORDINATE_SYSTEM,
    MediaPipePoseBackend,
)
from stablewalk.pose.backends.types import HumanMotionFrame, HumanMotionSequence
from stablewalk.io.smpl_motion_export import build_smpl_motion_arrays, export_smpl_motion_npz


def test_normalize_backend_aliases():
    assert normalize_backend_mode("romp") == "smpl"
    assert normalize_backend_mode("AUTO") == "auto"


def test_validate_smpl_not_ready_by_default():
    result = validate_smpl_assets()
    assert isinstance(result, SmplValidationResult)
    # Default dev env lacks torch/romp/SMPL_MODEL_DIR
    assert not result.ready
    assert result.issues


def test_resolve_mediapipe_backend():
    backend, resolution = resolve_pose_backend("mediapipe", allow_fallback=False)
    assert backend.name == "mediapipe"
    assert resolution.used == "mediapipe"
    assert not resolution.fallback


def test_resolve_auto_falls_back_to_mediapipe():
    if SMPLPoseBackend.is_available():
        pytest.skip("SMPL stack installed — fallback test skipped")
    backend, resolution = resolve_pose_backend("auto", allow_fallback=True)
    assert backend.name == "mediapipe"
    assert resolution.fallback
    assert resolution.fallback_reason
    assert resolution.warnings


def test_smpl_mode_raises_without_fallback():
    if SMPLPoseBackend.is_available():
        pytest.skip("SMPL available")
    from stablewalk.pose.backends.base import BackendUnavailableError

    with pytest.raises(BackendUnavailableError):
        resolve_pose_backend("smpl", allow_fallback=False)


def test_unified_motion_from_human_sequence():
    frame = HumanMotionFrame(
        frame_index=0,
        timestamp_s=0.0,
        joint_positions_3d={"left_hip": (0.1, 0.2, 0.3)},
        landmark_confidence={"left_hip": 0.9},
        backend_name="mediapipe",
        coordinate_system=MEDIAPIPE_COORDINATE_SYSTEM,
        detected=True,
    )
    seq = HumanMotionSequence(
        frames=[frame],
        fps=30.0,
        source_video="t.mp4",
        backend_name="mediapipe",
        coordinate_system=MEDIAPIPE_COORDINATE_SYSTEM,
    )
    unified = human_motion_sequence_to_unified(seq)
    assert unified.source_backend == "mediapipe"
    assert unified.frame_count == 1
    assert unified.frames[0].joint_positions_3d["left_hip"] == (0.1, 0.2, 0.3)


def test_smpl_export_rejects_mediapipe_backend():
    unified = UnifiedHumanMotion(
        fps=30.0,
        timestamps=np.array([0.0]),
        frames=[
            UnifiedMotionFrame(
                frame_index=0,
                timestamp_s=0.0,
                detected=True,
                joint_positions_3d={"pelvis": (0.0, 0.0, 0.0)},
            )
        ],
        source_backend="mediapipe",
        coordinate_system=MEDIAPIPE_COORDINATE_SYSTEM,
        scale_information=__import__(
            "stablewalk.pose.backends.unified_motion", fromlist=["ScaleInformation"]
        ).ScaleInformation(),
    )
    with pytest.raises(ValueError, match="Refusing to export smpl_motion"):
        build_smpl_motion_arrays(unified)


def test_smpl_export_schema(tmp_path: Path):
    from stablewalk.pose.backends.unified_motion import ScaleInformation

    unified = UnifiedHumanMotion(
        fps=30.0,
        timestamps=np.array([0.0, 1 / 30.0]),
        frames=[
            UnifiedMotionFrame(
                frame_index=0,
                timestamp_s=0.0,
                root_position=(0.0, 1.0, 0.0),
                root_orientation=(1.0, 0.0, 0.0, 0.0),
                joint_positions_3d={"pelvis": (0.0, 1.0, 0.0), "left_hip": (-0.1, 1.0, 0.0)},
                joint_rotations={"left_hip": (1.0, 0.0, 0.0, 0.0)},
                body_shape_parameters={"beta_0": 0.1},
                pose_confidence=0.9,
                detected=True,
            ),
            UnifiedMotionFrame(
                frame_index=1,
                timestamp_s=1 / 30.0,
                root_position=(0.01, 1.0, 0.0),
                joint_positions_3d={"pelvis": (0.01, 1.0, 0.0)},
                pose_confidence=0.88,
                detected=True,
            ),
        ],
        source_backend="smpl",
        coordinate_system=MEDIAPIPE_COORDINATE_SYSTEM,
        scale_information=ScaleInformation(scale_to_meters=1.0, notes="test"),
        provider_name="romp",
    )
    out = tmp_path / "smpl_motion.npz"
    export_smpl_motion_npz(unified, out)
    data = np.load(out, allow_pickle=False)
    assert data["source_backend"] == "smpl"
    assert len(data["timestamps"]) == 2
    assert data["joint_positions_3d"].shape[0] == 2
    assert "body_shape_betas" in data
    assert "pose_confidence" in data
