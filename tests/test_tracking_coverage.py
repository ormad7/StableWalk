"""Tracking HUD: detection vs joint visibility vs usable gait frames."""

from __future__ import annotations

from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.ui.tracking_coverage import measure_tracking_coverage


def _kp(name: str, vis: float = 0.9) -> Keypoint:
    ys = {
        "nose": 0.20,
        "left_hip": 0.48,
        "right_hip": 0.48,
        "left_knee": 0.66,
        "right_knee": 0.66,
        "left_ankle": 0.84,
        "right_ankle": 0.84,
    }
    return Keypoint(name=name, x=0.2, y=ys.get(name, 0.5), z=0.0, visibility=vis)


_LIMBS = (
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_hip",
    "right_knee",
    "right_ankle",
)


def _frame(index: int, *, detected: bool, vis: float = 0.9, missing: str | None = None) -> PoseFrame:
    kps = []
    if detected:
        for name in _LIMBS + ("nose",):
            if name == missing:
                continue
            kps.append(_kp(name, vis))
    return PoseFrame(
        frame_index=index,
        image_path=f"f{index}.jpg",
        timestamp_s=index / 30.0,
        detected=detected,
        keypoints=kps,
    )


def test_tracking_separates_detection_visibility_and_usable_frames() -> None:
    frames = [
        _frame(0, detected=True, vis=0.8),
        _frame(1, detected=True, vis=0.8),
        _frame(2, detected=False),
        _frame(3, detected=True, vis=0.2),
    ]
    cov = measure_tracking_coverage(PoseSequence(source_video="t.mp4", fps=30.0, frames=frames))
    assert cov is not None
    assert cov.total_frames == 4
    assert cov.detected_frames == 3
    assert abs(cov.detection_pct - 75.0) < 1e-6
    assert cov.joint_visibility_pct is not None
    assert 50.0 < cov.joint_visibility_pct < 80.0
    assert cov.usable_gait_frames == 2
    text = cov.hud_text()
    assert "Detect 75%" in text
    assert "Usable 2" in text
    assert "100%" not in text
    compact = cov.compact_text()
    assert "Detect 75%" in compact
    assert "Joints" in compact
    assert "Usable 2" in compact
    tip = cov.tooltip()
    assert "person-detection coverage" in tip
    assert "not joint-presence accuracy" in tip
    assert "not automatically usable" in tip
    assert "not accuracy" in tip
    assert "100% detection rate is not accuracy" in cov.detection_tooltip()
    assert "spectral reliability" in cov.detection_tooltip()
    assert cov.detected_display() == "3/4 (75%)"
    assert "vis" in cov.joint_confidence_display()
    assert cov.usable_display() == "2/4 (50%)"
    assert "accuracy" not in cov.detected_display().lower()
    assert "accuracy" not in cov.joint_confidence_display().lower()
    assert "accuracy" not in cov.usable_display().lower()


def test_overview_hud_tracking_captions_are_not_accuracy() -> None:
    from stablewalk.ui.tk.dashboard_sections import (
        _OVERVIEW_TRACKING_FIELDS,
        _OVERVIEW_VIDEO_INFO_FIELDS,
    )

    keys = [row[0] for row in _OVERVIEW_TRACKING_FIELDS]
    captions = [row[1].lower() for row in _OVERVIEW_TRACKING_FIELDS]
    tips = [row[2].lower() for row in _OVERVIEW_TRACKING_FIELDS]
    assert keys == ["detected", "joints", "usable"]
    assert "detected frames" in captions[0]
    assert "joint confidence" in captions[1]
    assert "usable frames" in captions[2]
    assert all("accuracy" not in caption for caption in captions)
    assert all("accuracy" in tip and "not" in tip for tip in tips)
    video_captions = [row[1] for row in _OVERVIEW_VIDEO_INFO_FIELDS]
    assert "Tracking" not in video_captions
    assert "Accuracy" not in video_captions


def test_missing_limb_is_not_a_usable_gait_frame() -> None:
    frames = [_frame(0, detected=True, missing="left_ankle")]
    cov = measure_tracking_coverage(PoseSequence(source_video="t.mp4", fps=30.0, frames=frames))
    assert cov is not None
    assert cov.detected_frames == 1
    assert cov.usable_gait_frames == 0
