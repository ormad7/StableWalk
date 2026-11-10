"""Missing detections keep source frame IDs and timestamps (no timeline compression)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.io.frame_audit import (
    iter_frame_audit_rows,
    timeline_windows,
    write_frame_audit_csv,
)
from stablewalk.io.pose_loader import detected_frame_indices, playback_frame_indices
from stablewalk.models.pose_data import (
    Keypoint,
    PoseFrame,
    PoseSequence,
    REJECTION_INCOMPLETE_BODY,
    REJECTION_NO_LANDMARKS,
    REJECTION_SUBJECT_LOCK_LOST,
    STATUS_DETECTION_FAILED,
    STATUS_FILTER_SKIP,
    STATUS_LOCK_SKIP,
    STATUS_POSE_OK,
)
from stablewalk.pose.skeleton import draw_identity_on_bgr
from stablewalk.pose.subject_lock import SubjectLock, select_subject_keypoints
from stablewalk.ui.media.demo_comparison import format_duration_display
from stablewalk.ui.skeleton_player import SkeletonPlayer


def _kp(name: str, x: float, y: float, vis: float = 0.9) -> Keypoint:
    return Keypoint(name=name, x=x, y=y, z=0.0, visibility=vis)


def _full_body(*, cx: float, cy: float = 0.5, scale: float = 0.25) -> list[Keypoint]:
    return [
        _kp("nose", cx, cy - scale * 0.9),
        _kp("left_shoulder", cx - 0.08, cy - scale * 0.5),
        _kp("right_shoulder", cx + 0.08, cy - scale * 0.5),
        _kp("left_hip", cx - 0.06, cy),
        _kp("right_hip", cx + 0.06, cy),
        _kp("mid_hip", cx, cy),
        _kp("left_knee", cx - 0.06, cy + scale * 0.4),
        _kp("right_knee", cx + 0.06, cy + scale * 0.4),
        _kp("left_ankle", cx - 0.06, cy + scale * 0.8),
        _kp("right_ankle", cx + 0.06, cy + scale * 0.8),
    ]


def _sequence_with_holes(fps: float = 30.0) -> PoseSequence:
    """10 source frames; detections missing at 3,4 and filtered at 7."""
    frames: list[PoseFrame] = []
    for i in range(10):
        ts = i / fps
        if i in (3, 4):
            frames.append(
                PoseFrame(
                    frame_index=i,
                    image_path="",
                    timestamp_s=ts,
                    timestamp_ms=int(round(ts * 1000)),
                    detected=False,
                    rejection_reason=REJECTION_NO_LANDMARKS,
                )
            )
        elif i == 7:
            frames.append(
                PoseFrame(
                    frame_index=i,
                    image_path="",
                    timestamp_s=ts,
                    timestamp_ms=int(round(ts * 1000)),
                    keypoints=_full_body(cx=0.5),
                    detected=False,
                    skipped=True,
                    rejection_reason=REJECTION_INCOMPLETE_BODY,
                    n_people=1,
                    subject_bbox=(0.3, 0.2, 0.7, 0.9),
                )
            )
        else:
            frames.append(
                PoseFrame(
                    frame_index=i,
                    image_path="",
                    timestamp_s=ts,
                    timestamp_ms=int(round(ts * 1000)),
                    keypoints=_full_body(cx=0.5),
                    detected=True,
                    n_people=1,
                    subject_bbox=(0.3, 0.2, 0.7, 0.9),
                )
            )
    return PoseSequence(source_video="holes.mp4", fps=fps, frames=frames)


def test_playback_keeps_every_source_frame() -> None:
    sequence = _sequence_with_holes()
    assert playback_frame_indices(sequence) == list(range(10))
    assert detected_frame_indices(sequence) == [0, 1, 2, 5, 6, 8, 9]
    recording = pose_sequence_to_gait_motion(sequence)
    assert recording.frame_count == 10
    for i, snap in enumerate(recording.snapshots):
        assert snap.frame_index == i
        assert abs(snap.time_s - i / 30.0) < 1e-9
    assert recording.snapshots[3].metadata["detected"] is False
    assert recording.snapshots[3].joints == {}
    assert recording.snapshots[7].metadata["skipped"] is True
    assert recording.metadata["detected_only"] is False


def test_detected_only_is_the_compression_path() -> None:
    sequence = _sequence_with_holes()
    compressed = pose_sequence_to_gait_motion(sequence, detected_only=True)
    assert compressed.frame_count == 7
    assert compressed.snapshots[3].frame_index == 5
    assert abs(compressed.duration_s - 9 / 30.0) < 1e-9
    full = pose_sequence_to_gait_motion(sequence, detected_only=False)
    assert abs(full.duration_s - 9 / 30.0) < 1e-9
    assert full.frame_count == 10


def test_rejection_reasons_separate_lock_filter_and_detection() -> None:
    lock = PoseFrame(
        frame_index=2,
        image_path="",
        timestamp_s=0.08,
        detected=False,
        rejection_reason=REJECTION_SUBJECT_LOCK_LOST,
        n_people=2,
    )
    none = PoseFrame(
        frame_index=3,
        image_path="",
        timestamp_s=0.12,
        detected=False,
        rejection_reason=REJECTION_NO_LANDMARKS,
    )
    filt = PoseFrame(
        frame_index=4,
        image_path="",
        timestamp_s=0.16,
        detected=False,
        skipped=True,
        rejection_reason=REJECTION_INCOMPLETE_BODY,
        keypoints=_full_body(cx=0.4),
    )
    ok = PoseFrame(
        frame_index=5,
        image_path="",
        timestamp_s=0.20,
        detected=True,
        keypoints=_full_body(cx=0.5),
    )
    assert lock.processing_status() == STATUS_LOCK_SKIP
    assert none.processing_status() == STATUS_DETECTION_FAILED
    assert filt.processing_status() == STATUS_FILTER_SKIP
    assert ok.processing_status() == STATUS_POSE_OK


def test_legacy_undetected_json_is_not_invented_as_lock_loss() -> None:
    legacy = PoseFrame(frame_index=9, image_path="", timestamp_s=0.3, detected=False)
    assert legacy.processing_status() == "undetected_unclassified"
    assert legacy.rejection_reason == ""


def test_timeline_windows_distinguish_source_analyzed_usable() -> None:
    sequence = _sequence_with_holes()
    windows = timeline_windows(
        sequence, source_frame_count=10, source_duration_s=10 / 30.0
    )
    assert windows.analyzed_frame_count == 10
    assert abs(windows.analyzed_duration_s - 9 / 30.0) < 1e-9
    assert windows.detected_count == 7
    assert windows.source_duration_s == 10 / 30.0
    assert windows.last_detected_timestamp_s == 9 / 30.0
    assert windows.timestamp_gaps_if_detected_only == 2
    text = format_duration_display(
        analyzed_s=0.30,
        recording_s=1.00,
        usable_s=0.12,
    )
    assert text == "0.12 s valid · 0.30 s analyzed span of 1.00 s source"
    assert abs(windows.cumulative_valid_duration_s - 4 / 30.0) < 1e-9
    assert abs(windows.longest_valid_segment_s - 2 / 30.0) < 1e-9
    assert abs(windows.coverage - (4 / 9.0)) < 1e-9
    assert "original source timestamps" in windows.interval_counting_convention


def test_frame_audit_csv_keeps_failed_rows(tmp_path: Path) -> None:
    sequence = _sequence_with_holes()
    rows = list(iter_frame_audit_rows(sequence, clip="holes"))
    assert len(rows) == 10
    assert rows[3]["pose_valid"] is False
    assert rows[3]["processing_status"] == STATUS_DETECTION_FAILED
    assert rows[7]["processing_status"] == STATUS_FILTER_SKIP
    path = write_frame_audit_csv(rows, tmp_path / "frame_audit.csv")
    with path.open(encoding="utf-8", newline="") as handle:
        loaded = list(csv.DictReader(handle))
    assert [int(r["source_frame"]) for r in loaded] == list(range(10))
    assert loaded[3]["rejection_reason"] == REJECTION_NO_LANDMARKS


def test_player_occupies_missing_source_slots() -> None:
    sequence = _sequence_with_holes()
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=True)
    player.play()
    player.go_to_time(2 / 30.0)
    player.advance(speed=1.0, dt=1.0 / 30.0)
    assert player.state.frame_index == 3
    snap = player.current_snapshot()
    assert snap is not None
    assert snap.frame_index == 3
    assert snap.metadata.get("detected") is False


def test_missing_pose_does_not_fill_origin_in_time_series() -> None:
    sequence = _sequence_with_holes()
    recording = pose_sequence_to_gait_motion(sequence)
    ts = recording.build_time_series(rebuild=True)
    from stablewalk.models.joint_registry import ROOT_JOINT_ID

    missing = ts.positions[ROOT_JOINT_ID][3]
    assert missing.x != missing.x  # NaN, not origin


def test_subject_lock_does_not_jump_to_background_person() -> None:
    walker = _full_body(cx=0.55, scale=0.35)
    background = _full_body(cx=0.15, scale=0.12)
    lock = SubjectLock(cx=0.55, cy=0.5, bbox_h=0.35, initialized=True)
    selected, new_lock = select_subject_keypoints([background, walker], lock)
    assert selected is not None
    hips = [kp for kp in selected if kp.name == "mid_hip"]
    assert hips and abs(hips[0].x - 0.55) < 0.02
    assert new_lock is not None
    lost, kept = select_subject_keypoints([background], lock)
    assert lost is None
    assert kept is lock


def test_identity_overlay_draws_on_missing_pose() -> None:
    bgr = np.zeros((80, 80, 3), dtype=np.uint8)
    frame = PoseFrame(
        frame_index=1,
        image_path="",
        timestamp_s=0.03,
        detected=False,
        rejection_reason=REJECTION_SUBJECT_LOCK_LOST,
        n_people=2,
        subject_bbox=(0.2, 0.2, 0.8, 0.9),
        other_bboxes=[(0.0, 0.4, 0.25, 0.8)],
    )
    out = draw_identity_on_bgr(bgr, frame)
    assert out.shape == bgr.shape
    assert int(out.sum()) > 0
