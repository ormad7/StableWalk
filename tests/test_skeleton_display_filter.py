"""Display-only skeleton kinematic filter (does not mutate analysis recordings)."""

from __future__ import annotations

import math

from stablewalk.data.mock_gait import MockGaitConfig, generate_mock_gait
from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
from stablewalk.models.joint_registry import ROOT_JOINT_ID
from stablewalk.ui.skeleton_display_filter import SkeletonDisplayFilter


def _thigh_length(snap: SkeletonSnapshot, side: str = "left") -> float:
    hip = snap.joints[f"{side}_hip"].position
    knee = snap.joints[f"{side}_knee"].position
    return math.dist((hip.x, hip.y, hip.z), (knee.x, knee.y, knee.z))


def test_filter_reduces_jitter_without_changing_source_snapshot() -> None:
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.0, cadence_hz=1.0))
    filt = SkeletonDisplayFilter()
    raw0 = recording.snapshots[0]
    before = {
        jid: (s.position.x, s.position.y, s.position.z) for jid, s in raw0.joints.items()
    }

    # Feed a few frames, then inject a large spike on the left knee.
    for snap in recording.snapshots[:5]:
        filt.filter_snapshot(snap)
    spiked = recording.snapshots[5]
    knee = spiked.joints["left_knee"]
    spiked.joints["left_knee"] = JointSample(
        joint_id="left_knee",
        position=Vec3(knee.position.x + 0.25, knee.position.y, knee.position.z),
        parent_id=knee.parent_id,
        angle_deg=knee.angle_deg,
    )
    filtered = filt.filter_snapshot(spiked)
    raw_dx = abs(spiked.joints["left_knee"].position.x - recording.snapshots[4].joints["left_knee"].position.x)
    filt_dx = abs(
        filtered.joints["left_knee"].position.x
        - filt._history[-2].joints["left_knee"].position.x
    )
    assert filt_dx < raw_dx * 0.55
    # Source joints on the original early frame are untouched.
    after = {
        jid: (s.position.x, s.position.y, s.position.z) for jid, s in raw0.joints.items()
    }
    assert after == before


def test_filter_keeps_limb_lengths_nearly_constant() -> None:
    recording = generate_mock_gait(MockGaitConfig(fps=30.0, duration_s=1.5, cadence_hz=1.0))
    filt = SkeletonDisplayFilter(length_blend=0.85)
    lengths: list[float] = []
    for snap in recording.snapshots:
        out = filt.filter_snapshot(snap)
        lengths.append(_thigh_length(out, "left"))
    assert len(lengths) > 10
    mean = sum(lengths) / len(lengths)
    max_dev = max(abs(v - mean) / mean for v in lengths)
    # Display lengths should stay within a tight band (mocap-like).
    assert max_dev < 0.12


def test_filter_levels_shoulders() -> None:
    filt = SkeletonDisplayFilter(shoulder_level=0.9)
    joints = {
        ROOT_JOINT_ID: JointSample(ROOT_JOINT_ID, Vec3(0, 0.9, 0)),
        "spine": JointSample("spine", Vec3(0, 1.1, 0), parent_id=ROOT_JOINT_ID),
        "left_shoulder": JointSample("left_shoulder", Vec3(-0.2, 1.40, 0), parent_id="spine"),
        "right_shoulder": JointSample("right_shoulder", Vec3(0.2, 1.55, 0), parent_id="spine"),
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0), parent_id=ROOT_JOINT_ID),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0), parent_id=ROOT_JOINT_ID),
    }
    snap = SkeletonSnapshot(frame_index=0, time_s=0.0, joints=joints)
    out = filt.filter_snapshot(snap)
    dy = abs(out.joints["left_shoulder"].position.y - out.joints["right_shoulder"].position.y)
    assert dy < 0.08


def test_low_confidence_holds_instead_of_jumping() -> None:
    filt = SkeletonDisplayFilter()
    base = {
        "left_wrist": JointSample("left_wrist", Vec3(0.0, 1.0, 0.0)),
        ROOT_JOINT_ID: JointSample(ROOT_JOINT_ID, Vec3(0.0, 0.9, 0.0)),
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0.0)),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0.0)),
    }
    s0 = SkeletonSnapshot(
        0, 0.0, joints=dict(base), metadata={"landmark_visibility": {"left_wrist": 1.0}}
    )
    filt.filter_snapshot(s0)
    jumped = dict(base)
    jumped["left_wrist"] = JointSample("left_wrist", Vec3(0.4, 1.0, 0.0))
    s1 = SkeletonSnapshot(
        1,
        0.033,
        joints=jumped,
        metadata={"landmark_visibility": {"left_wrist": 0.05}},
    )
    out = filt.filter_snapshot(s1)
    assert abs(out.joints["left_wrist"].position.x) < 0.12


def test_filter_uncrosses_flipped_shoulders() -> None:
    filt = SkeletonDisplayFilter(shoulder_flip_blend=1.0, shoulder_level=0.0)
    joints = {
        ROOT_JOINT_ID: JointSample(ROOT_JOINT_ID, Vec3(0, 0.9, 0)),
        "spine": JointSample("spine", Vec3(0, 1.1, 0), parent_id=ROOT_JOINT_ID),
        # Intentionally crossed left/right shoulders.
        "left_shoulder": JointSample("left_shoulder", Vec3(0.22, 1.45, 0), parent_id="spine"),
        "right_shoulder": JointSample("right_shoulder", Vec3(-0.22, 1.45, 0), parent_id="spine"),
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0), parent_id=ROOT_JOINT_ID),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0), parent_id=ROOT_JOINT_ID),
    }
    snap = SkeletonSnapshot(frame_index=0, time_s=0.0, joints=joints)
    out = filt.filter_snapshot(snap)
    assert out.joints["left_shoulder"].position.x < out.joints["right_shoulder"].position.x


def test_filter_softens_knee_hyperextension() -> None:
    filt = SkeletonDisplayFilter(hyperext_blend=1.0, knee_plane=0.0, length_blend=0.0)
    # Collinear hyperextended knee (mid on the hip–ankle axis).
    joints = {
        ROOT_JOINT_ID: JointSample(ROOT_JOINT_ID, Vec3(0, 0.95, 0)),
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.95, 0), parent_id=ROOT_JOINT_ID),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.95, 0), parent_id=ROOT_JOINT_ID),
        "left_knee": JointSample("left_knee", Vec3(-0.1, 0.55, 0), parent_id="left_hip"),
        "left_ankle": JointSample("left_ankle", Vec3(-0.1, 0.15, 0), parent_id="left_knee"),
        "right_knee": JointSample("right_knee", Vec3(0.1, 0.55, 0), parent_id="right_hip"),
        "right_ankle": JointSample("right_ankle", Vec3(0.1, 0.15, 0), parent_id="right_knee"),
    }
    snap = SkeletonSnapshot(frame_index=0, time_s=0.0, joints=joints)
    # Seed length priors without freezing hard lock immediately.
    filt._lengths[("left_hip", "left_knee")] = 0.40
    filt._lengths[("left_knee", "left_ankle")] = 0.40
    out = filt.filter_snapshot(snap)
    hip = out.joints["left_hip"].position
    knee = out.joints["left_knee"].position
    ankle = out.joints["left_ankle"].position
    # After soft constraint, knee should not be perfectly collinear.
    v1 = (hip.x - knee.x, hip.y - knee.y, hip.z - knee.z)
    v2 = (ankle.x - knee.x, ankle.y - knee.y, ankle.z - knee.z)
    n1 = math.sqrt(sum(c * c for c in v1))
    n2 = math.sqrt(sum(c * c for c in v2))
    cos_a = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2)
    interior = math.degrees(math.acos(max(-1.0, min(1.0, cos_a))))
    assert interior < 179.5
