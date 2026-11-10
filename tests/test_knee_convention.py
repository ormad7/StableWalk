"""Canonical knee flexion vs stored pose interior angle."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.models.gait_motion import (
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.pose.knee_convention import (
    KNEE_STORAGE_FLEXION,
    KNEE_STORAGE_INTERIOR,
    analysis_knee_flexion_deg,
    recording_knee_storage,
)
from stablewalk.pose.kinematics import knee_flexion_from_hka


def _joint(jid: str, x: float, y: float, z: float) -> JointSample:
    return JointSample(joint_id=jid, position=Vec3(x, y, z))


def _hka_snapshot(*, interior_stored: float, flexion_geom: float) -> SkeletonSnapshot:
    import math

    hip = Vec3(0.0, 0.90, 0.0)
    knee = Vec3(0.0, 0.48, 0.0)
    shank = 0.42
    # 0° flexion: ankle continues down −Y. Positive flexion rotates shank +Z.
    rad = math.radians(flexion_geom)
    ankle = Vec3(0.0, knee.y - shank * math.cos(rad), shank * math.sin(rad))
    return SkeletonSnapshot(
        frame_index=0,
        time_s=0.0,
        joints={
            "left_hip": _joint("left_hip", hip.x, hip.y, hip.z),
            "left_knee": _joint("left_knee", knee.x, knee.y, knee.z),
            "left_ankle": _joint("left_ankle", ankle.x, ankle.y, ankle.z),
        },
        dofs={
            "left_knee_flexion": DofSample(
                dof_id="left_knee_flexion", angle_deg=interior_stored
            )
        },
        metadata={"landmark_visibility": {"left_hip": 1.0, "left_knee": 1.0, "left_ankle": 1.0}},
    )


def test_hka_matches_180_minus_interior() -> None:
    hip = SimpleNamespace(x=0.0, y=1.0, z=0.0, visibility=1.0)
    knee = SimpleNamespace(x=0.0, y=0.5, z=0.0, visibility=1.0)
    ankle = SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=1.0)
    assert knee_flexion_from_hka(hip, knee, ankle) == 0.0


def test_pose_interior_storage_uses_hka_flexion_not_stored_interior() -> None:
    snap = _hka_snapshot(interior_stored=145.0, flexion_geom=35.0)
    value = analysis_knee_flexion_deg(
        snap, "left_knee_flexion", storage=KNEE_STORAGE_INTERIOR
    )
    assert value is not None
    assert abs(value - 35.0) < 3.0
    assert value < 80.0


def test_pose_interior_without_hka_converts_stored() -> None:
    snap = SkeletonSnapshot(
        frame_index=0,
        time_s=0.0,
        joints={},
        dofs={"left_knee_flexion": DofSample(dof_id="left_knee_flexion", angle_deg=150.0)},
    )
    assert analysis_knee_flexion_deg(
        snap, "left_knee_flexion", storage=KNEE_STORAGE_INTERIOR
    ) == 30.0


def test_mock_storage_keeps_stored_flexion_even_if_hka_differs() -> None:
    snap = _hka_snapshot(interior_stored=22.0, flexion_geom=40.0)
    assert analysis_knee_flexion_deg(
        snap, "left_knee_flexion", storage=KNEE_STORAGE_FLEXION
    ) == 22.0


def test_pose_adapter_tags_interior_storage() -> None:
    rec = GaitMotionRecording(
        source="walk.mp4",
        fps=25.0,
        snapshots=[],
        source_kind="pose_estimation",
        metadata={"knee_angle_storage": "pose_interior_hka"},
    )
    assert recording_knee_storage(rec) == KNEE_STORAGE_INTERIOR


def test_spectral_raw_knee_on_pose_is_flexion_not_interior() -> None:
    from stablewalk.analysis.spectral.series import build_spectral_series

    snap = _hka_snapshot(interior_stored=140.0, flexion_geom=30.0)
    rec = GaitMotionRecording(
        source="pose",
        fps=25.0,
        snapshots=[snap, snap],
        source_kind="pose_estimation",
        metadata={"knee_angle_storage": "pose_interior_hka"},
    )
    bundle = build_spectral_series(rec, channel_ids=["dof:left_knee_flexion"])
    series = bundle.get("dof:left_knee_flexion")
    assert series is not None
    finite = [v for v in series.raw if v == v]
    assert finite
    assert max(finite) < 80.0
    assert bundle.metadata.get("knee_raw_signal") == "anatomical_knee_flexion_deg"
