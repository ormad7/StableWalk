"""Tests for demo video validation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from stablewalk.ui.media.demo_validation import ValidationGrade, validate_demo_video


def test_validate_demo_video_missing_file():
    report = validate_demo_video("/nonexistent/demo.mp4")
    assert report.full_body_quality == ValidationGrade.FAIL
    assert "missing" in report.compact_status.lower()


def test_validate_demo_video_with_mock_sequence():
    fake_frame = mock.Mock()
    fake_frame.keypoints = []
    fake_frame.detected = False
    fake_sequence = mock.Mock()
    fake_sequence.frames = [fake_frame] * 10

    with mock.patch("cv2.VideoCapture") as cap_cls, mock.patch(
        "stablewalk.ui.media.demo_validation.PoseEstimator"
    ) as est_cls:
        cap = cap_cls.return_value
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            5: 25.0,
            7: 100,
            3: 640,
            4: 480,
        }.get(prop, 0)
        est_cls.return_value.__enter__.return_value.process_video_with_frame_cache.return_value = (
            fake_sequence,
            None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "demo.mp4"
            path.write_bytes(b"\x00" * 100)
            report = validate_demo_video(path, max_frames=10)

    assert report.sampled_frames == 10
    assert report.pose_detection_rate == 0.0
    assert report.full_body_quality == ValidationGrade.FAIL
