"""Observed vs unreliable pose tracking — no fake usable frames."""

from __future__ import annotations

from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.pose.subject_lock import SubjectLock, select_subject_keypoints
from stablewalk.pose.tracking_quality import (
    STATUS_OBSERVED,
    STATUS_UNRELIABLE,
    audit_pose_sequence,
)
from stablewalk.ui.tracking_coverage import measure_tracking_coverage


def _limb_kps(*, vis: float = 0.9, invert: bool = False, x_shift: float = 0.0) -> list[Keypoint]:
    # Keep hips fixed so invert tests anatomy, not a pelvis jump.
    hip_y, knee_y, ankle_y = (0.48, 0.36, 0.24) if invert else (0.48, 0.66, 0.84)
    names = {
        "nose": (0.50, 0.20),
        "left_hip": (0.45, hip_y),
        "right_hip": (0.55, hip_y),
        "left_knee": (0.45, knee_y),
        "right_knee": (0.55, knee_y),
        "left_ankle": (0.45, ankle_y),
        "right_ankle": (0.55, ankle_y),
        "left_heel": (0.44, ankle_y + 0.02),
        "right_heel": (0.56, ankle_y + 0.02),
        "left_foot_index": (0.45, ankle_y + 0.03),
        "right_foot_index": (0.55, ankle_y + 0.03),
    }
    return [
        Keypoint(name=n, x=x + x_shift, y=y, z=0.0, visibility=vis)
        for n, (x, y) in names.items()
    ]


def _frame(index: int, kps: list[Keypoint], *, detected: bool = True) -> PoseFrame:
    return PoseFrame(
        frame_index=index,
        image_path=f"f{index}.jpg",
        timestamp_s=index / 30.0,
        detected=detected,
        keypoints=kps if detected else [],
    )


def test_inverted_anatomy_is_unreliable_not_usable() -> None:
    frames = [
        _frame(0, _limb_kps(vis=0.95)),
        _frame(1, _limb_kps(vis=0.95, invert=True)),
        _frame(2, _limb_kps(vis=0.95)),
    ]
    seq = PoseSequence(source_video="t.mp4", fps=30.0, frames=frames)
    audit = audit_pose_sequence(seq)
    assert audit.observed_frames == 2
    assert audit.unreliable_frames == 1
    assert 1 in audit.anatomy_fail_frames
    cov = measure_tracking_coverage(seq)
    assert cov is not None
    assert cov.detected_frames == 3
    assert cov.usable_gait_frames == 2


def test_pose_existence_is_not_enough_for_usable() -> None:
    frames = [
        _frame(0, _limb_kps(vis=0.2)),
        _frame(1, [], detected=False),
    ]
    seq = PoseSequence(source_video="t.mp4", fps=30.0, frames=frames)
    audit = audit_pose_sequence(seq)
    assert audit.observed_frames == 0
    assert audit.unreliable_frames == 1
    assert audit.missing_frames == 1
    cov = measure_tracking_coverage(seq)
    assert cov is not None
    assert cov.detected_frames == 1
    assert cov.usable_gait_frames == 0


def test_jump_is_flagged_and_not_interpolated() -> None:
    frames = [
        _frame(0, _limb_kps()),
        _frame(1, _limb_kps(x_shift=0.45)),
        _frame(2, _limb_kps(x_shift=0.46)),
    ]
    audit = audit_pose_sequence(PoseSequence(source_video="t.mp4", fps=30.0, frames=frames))
    assert audit.interpolated_frames == 0
    assert audit.jump_frames
    assert any(fq == 1 for fq in audit.jump_frames)
    assert audit.samples["begin"]["status"] in (STATUS_OBSERVED, STATUS_UNRELIABLE)


def test_pose_json_roundtrip_keeps_visibility() -> None:
    from stablewalk.models.pose_data import PoseFrame

    frame = _frame(0, _limb_kps(vis=0.61))
    payload = frame.to_export_dict()
    left_hip = payload["keypoints"]["left_hip"]
    assert abs(float(left_hip["visibility"]) - 0.61) < 1e-9
    assert "z" in left_hip


def test_legacy_xy_keypoints_recover_visibility_from_skeleton() -> None:
    from pathlib import Path
    import json
    import tempfile

    from stablewalk.io.pose_loader import load_pose_sequence

    payload = {
        "source_video": "t.mp4",
        "fps": 30.0,
        "frames": [
            {
                "frame": 0,
                "detected": True,
                "keypoints": {"left_hip": [0.4, 0.5], "left_knee": [0.4, 0.6]},
                "skeleton_3d": {
                    "joints": {
                        "left_hip": {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.42},
                        "left_knee": {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.77},
                    }
                },
            }
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "legacy_poses.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        seq = load_pose_sequence(path)
    by_name = {kp.name: kp for kp in seq.frames[0].keypoints}
    assert abs(by_name["left_hip"].visibility - 0.42) < 1e-9
    assert abs(by_name["left_knee"].visibility - 0.77) < 1e-9


def test_skeleton3d_origin_does_not_replace_image_pelvis() -> None:
    """Canonical 3D mid_hip is (0,0,0); jumps must use image-plane hips."""
    kps = _limb_kps()
    frame = _frame(0, kps)
    frame.skeleton_3d = {
        "joints": {
            "mid_hip": {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 1.0},
            "left_hip": {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9},
        }
    }
    seq = PoseSequence(
        source_video="t.mp4",
        fps=30.0,
        frames=[
            frame,
            _frame(1, _limb_kps(x_shift=0.02)),
        ],
    )
    audit = audit_pose_sequence(seq)
    pelvis = audit.samples["begin"]["pelvis_xy"]
    assert pelvis is not None
    assert abs(pelvis[0] - 0.50) < 0.03
    assert abs(pelvis[1] - 0.48) < 0.03
    assert audit.jump_frames == []


def test_subject_lock_prefers_locked_walker_over_background() -> None:
    walker = _limb_kps()
    background = _limb_kps(x_shift=0.35)
    for kp in background:
        kp.y = min(0.95, kp.y * 0.45 + 0.05)
    lock = SubjectLock(cx=0.50, cy=0.48, bbox_h=0.67, initialized=True)
    selected, new_lock = select_subject_keypoints([background, walker], lock)
    assert selected is not None
    names = {kp.name: kp for kp in selected}
    assert abs(names["left_hip"].x - 0.45) < 0.02
    assert new_lock is not None and new_lock.initialized
