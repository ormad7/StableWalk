"""Subject identity: seed, overlap skip, overlay agreement, cache fingerprint."""

from __future__ import annotations

import numpy as np

from stablewalk.models.pose_data import (
    Keypoint,
    PoseFrame,
    REJECTION_SUBJECT_AMBIGUOUS,
    REJECTION_SUBJECT_LOCK_LOST,
)
from stablewalk.pose.skeleton import SUBJECT_ROI_COLOR, draw_identity_on_bgr
from stablewalk.pose.subject_lock import (
    SubjectLock,
    SubjectSelection,
    pick_subject,
    pose_bbox_norm,
    pose_dump_matches_selection,
    selection_for_source,
)
from stablewalk.ui.video_display_fit import click_to_normalized_xy


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


def _hip_x(keypoints: list[Keypoint] | None) -> float:
    assert keypoints is not None
    hips = [kp for kp in keypoints if kp.name == "mid_hip"]
    assert hips
    return float(hips[0].x)


def test_seed_picks_left_walker_not_larger_presenter() -> None:
    walker = _full_body(cx=0.12, scale=0.28)
    presenter = _full_body(cx=0.86, scale=0.55)
    seed = selection_for_source("normal_gait_v2.mp4")
    picked = pick_subject([presenter, walker], None, selection=seed, time_s=0.0)
    assert picked.keypoints is not None
    assert abs(_hip_x(picked.keypoints) - 0.12) < 0.02
    assert picked.identity_locked is True
    assert picked.tracking_status == "locked"


def test_without_seed_largest_person_is_chosen() -> None:
    walker = _full_body(cx=0.12, scale=0.28)
    presenter = _full_body(cx=0.86, scale=0.55)
    picked = pick_subject(
        [presenter, walker], None, selection=SubjectSelection(), time_s=0.0
    )
    assert abs(_hip_x(picked.keypoints) - 0.86) < 0.02


def test_overlap_marks_unavailable_without_updating_lock() -> None:
    walker = _full_body(cx=0.45, scale=0.32)
    presenter = _full_body(cx=0.50, scale=0.34)
    lock = SubjectLock(
        cx=0.45,
        cy=0.5,
        bbox_h=0.32,
        initialized=True,
        last_bbox=pose_bbox_norm(walker),
        last_time_s=4.5,
    )
    picked = pick_subject(
        [walker, presenter],
        lock,
        selection=SubjectSelection(),
        time_s=4.92,
    )
    assert picked.keypoints is None
    assert picked.lock is lock
    assert picked.tracking_status == "unavailable"
    assert picked.reason == "subject_identity_ambiguous"
    assert picked.subject_bbox is None


def test_fused_single_box_marks_unavailable() -> None:
    lock_box = (0.00, 0.30, 0.15, 0.75)
    lock = SubjectLock(
        cx=0.12,
        cy=0.55,
        bbox_h=0.30,
        initialized=True,
        last_bbox=lock_box,
        last_time_s=4.5,
        last_n_people=2,
    )
    fused = _full_body(cx=0.42, scale=0.70)
    picked = pick_subject(
        [fused], lock, selection=SubjectSelection(), time_s=4.92
    )
    assert picked.keypoints is None
    assert picked.lock is lock
    assert picked.tracking_status == "unavailable"
    assert picked.reason == "subject_identity_ambiguous"


def test_two_people_x_overlap_is_unavailable() -> None:
    left = _full_body(cx=0.40, scale=0.30)
    right = _full_body(cx=0.52, scale=0.30)
    lock = SubjectLock(
        cx=0.40,
        cy=0.5,
        bbox_h=0.30,
        initialized=True,
        last_bbox=pose_bbox_norm(left),
        last_n_people=2,
        last_time_s=3.0,
    )
    picked = pick_subject(
        [left, right], lock, selection=SubjectSelection(), time_s=3.1
    )
    assert picked.keypoints is None
    assert picked.lock is lock
    assert picked.reason == "subject_identity_ambiguous"


def test_single_detection_after_two_people_stays_unavailable() -> None:
    walker = _full_body(cx=0.30, scale=0.28)
    lock = SubjectLock(
        cx=0.30,
        cy=0.5,
        bbox_h=0.28,
        initialized=True,
        last_bbox=pose_bbox_norm(walker),
        last_n_people=2,
        last_time_s=3.0,
        vx=0.10,
    )
    fused = _full_body(cx=0.48, scale=0.55)
    picked = pick_subject(
        [fused], lock, selection=SubjectSelection(), time_s=3.4
    )
    assert picked.keypoints is None
    assert picked.lock is lock


def test_single_detection_after_two_people_is_always_unavailable() -> None:
    walker = _full_body(cx=0.30, scale=0.28)
    lock = SubjectLock(
        cx=0.30,
        cy=0.5,
        bbox_h=0.28,
        initialized=True,
        last_bbox=pose_bbox_norm(walker),
        last_n_people=2,
        last_time_s=3.0,
        vx=0.0,
        vy=0.0,
    )
    picked = pick_subject(
        [walker], lock, selection=SubjectSelection(), time_s=3.05
    )
    assert picked.keypoints is None
    assert picked.lock is lock
    assert picked.tracking_status == "unavailable"


def test_reacquire_nearest_person_after_overlap_gap() -> None:
    lock = SubjectLock(
        cx=0.32,
        cy=0.50,
        bbox_h=0.30,
        initialized=True,
        last_bbox=(0.20, 0.20, 0.45, 0.85),
        last_n_people=2,
        last_time_s=2.90,
        vx=0.10,
        vy=0.0,
    )
    presenter = _full_body(cx=0.15, scale=0.32)
    walker = _full_body(cx=0.72, scale=0.32)
    picked = pick_subject(
        [presenter, walker],
        lock,
        selection=SubjectSelection(),
        time_s=6.36,
    )
    assert picked.keypoints is not None
    assert abs(_hip_x(picked.keypoints) - 0.72) < 0.03
    assert picked.identity_locked is True


def test_identity_overlay_subject_matches_analyzed_keypoints() -> None:
    bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    kps = _full_body(cx=0.75, scale=0.30)
    stale_left = (0.02, 0.10, 0.30, 0.90)
    frame = PoseFrame(
        frame_index=191,
        image_path="",
        timestamp_s=6.36,
        keypoints=kps,
        detected=True,
        n_people=2,
        subject_bbox=stale_left,
        other_bboxes=[stale_left],
        identity_locked=True,
        tracking_status="locked",
    )
    out = draw_identity_on_bgr(bgr, frame)
    box = pose_bbox_norm(kps)
    assert box is not None
    x0, y0, _, _ = box
    subject_px = tuple(int(v) for v in out[int(y0 * 120), int(x0 * 120)])
    assert subject_px == SUBJECT_ROI_COLOR
    stale_px = tuple(int(v) for v in out[int(0.10 * 120), int(0.02 * 120)])
    assert stale_px != SUBJECT_ROI_COLOR


def test_unavailable_overlay_does_not_draw_stale_subject_box() -> None:
    bgr = np.zeros((120, 120, 3), dtype=np.uint8)
    fused = (0.20, 0.15, 0.80, 0.90)
    frame = PoseFrame(
        frame_index=148,
        image_path="",
        timestamp_s=4.92,
        detected=False,
        rejection_reason=REJECTION_SUBJECT_AMBIGUOUS,
        n_people=1,
        subject_bbox=fused,
        other_bboxes=[fused],
        identity_locked=False,
        tracking_status="unavailable",
        keypoints=_full_body(cx=0.50, scale=0.55),
    )
    out = draw_identity_on_bgr(bgr, frame)
    x0, y0, _, _ = fused
    edge = tuple(int(v) for v in out[int(y0 * 120), int(x0 * 120)])
    assert edge != SUBJECT_ROI_COLOR
    hud = out[58, 12]
    assert int(hud.sum()) > 0


def test_lock_lost_overlay_still_draws_without_subject_fill() -> None:
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
        tracking_status="unavailable",
    )
    out = draw_identity_on_bgr(bgr, frame)
    assert out.shape == bgr.shape
    assert int(out.sum()) > 0


def test_pose_dump_without_fingerprint_is_incompatible() -> None:
    wanted = selection_for_source("normal_gait_v2.mp4")
    assert wanted.seed_xy is not None
    assert pose_dump_matches_selection({}, wanted) is False
    assert pose_dump_matches_selection({"subject_selection": {}}, wanted) is False
    assert pose_dump_matches_selection({"subject_selection": wanted.to_dict()}, wanted)
    other = SubjectSelection(seed_xy=(0.80, 0.50), policy="click")
    assert pose_dump_matches_selection({"subject_selection": wanted.to_dict()}, other) is False
    unseeded = SubjectSelection()
    assert pose_dump_matches_selection({}, unseeded) is True
    assert pose_dump_matches_selection(
        {"subject_selection": unseeded.to_dict()}, unseeded
    ) is True


def test_click_maps_through_fit_letterbox() -> None:
    # 100x200 source in a 300x200 label: fitted image is 100x200, x offset 100.
    xy = click_to_normalized_xy(
        150,
        100,
        label_w=300,
        label_h=200,
        src_w=100,
        src_h=200,
        zoom_mode="Fit",
    )
    assert xy is not None
    assert abs(xy[0] - 0.50) < 0.02
    assert abs(xy[1] - 0.50) < 0.02
    assert (
        click_to_normalized_xy(
            10,
            100,
            label_w=300,
            label_h=200,
            src_w=100,
            src_h=200,
            zoom_mode="Fit",
        )
        is None
    )
