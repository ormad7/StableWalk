"""Documented source labels stay separate from computed comparison results."""

from __future__ import annotations

from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.ui.media.demo_comparison import (
    COMPUTED_PENDING,
    DOCUMENTED_UNAVAILABLE,
    analyzed_duration_s,
    build_comparison_row,
    comparison_tree_values,
    documented_camera_view,
    documented_locomotion,
    documented_placeholder_row,
    documented_source_label,
    format_duration_display,
)
from stablewalk.ui.media.demo_gait import example_by_key


def _recording(*, duration_s: float = 10.0, max_angle: float = 42.0) -> GaitMotionRecording:
    snaps = []
    for i, t in enumerate((0.0, duration_s)):
        snaps.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints={
                    "right_knee": JointSample(
                        joint_id="right_knee",
                        position=Vec3(0.0, 0.4, 0.0),
                        angle_deg=max_angle if i else 10.0,
                        velocity=1.2,
                    )
                },
            )
        )
    return GaitMotionRecording(source="test", fps=30.0, snapshots=snaps)


def test_documented_fields_are_not_computed_scores() -> None:
    abnormal = example_by_key("abnormal")
    normal = example_by_key("normal")
    fast = example_by_key("athletic")
    assert abnormal is not None and normal is not None and fast is not None

    assert "parkinsons" in documented_source_label(abnormal).lower()
    assert documented_camera_view(abnormal) == DOCUMENTED_UNAVAILABLE
    assert documented_locomotion(abnormal) == DOCUMENTED_UNAVAILABLE

    assert "gait_pat=normal" in documented_source_label(normal)
    assert documented_camera_view(normal) == "Right side"
    assert documented_locomotion(normal) == DOCUMENTED_UNAVAILABLE

    assert "walk_54" in documented_source_label(fast)
    assert documented_camera_view(fast) == "Left side (fixed)"
    assert documented_locomotion(fast) == "Treadmill"

    placeholder = documented_placeholder_row(fast)
    assert placeholder.source_label == documented_source_label(fast)
    assert placeholder.duration_display == COMPUTED_PENDING
    assert placeholder.usable_cycles_display == COMPUTED_PENDING
    assert placeholder.tracking_quality == COMPUTED_PENDING
    assert "5.4" not in placeholder.usable_cycles_display


def test_computed_duration_and_cycles_are_separate_from_source_label() -> None:
    fast = example_by_key("athletic")
    assert fast is not None
    rec = _recording(duration_s=27.24, max_angle=55.0)
    row = build_comparison_row(
        rec,
        fast,
        "right_knee",
        usable_cycles=4,
        tracking_quality="Detect 92% · Joints 71% mean visibility · Usable 829",
        recording_duration_s=68.52,
    )
    assert row.source_label == documented_source_label(fast)
    assert "5.4" not in row.duration_display
    assert row.duration_display == "27.24 s analyzed span of 68.52 s recording"
    assert row.usable_cycles_display == "4"
    assert "Detect 92%" in row.tracking_quality
    assert row.locomotion_setting == "Treadmill"
    values = comparison_tree_values(row)
    assert values[1] == row.source_label
    assert values[3] == "Treadmill"
    assert values[4] == row.duration_display
    assert values[5] == "4"


def test_duration_display_pending_until_analysis() -> None:
    assert format_duration_display(analyzed_s=None, recording_s=None) == COMPUTED_PENDING
    assert analyzed_duration_s(None) is None
    rec = _recording(duration_s=9.98)
    assert abs(analyzed_duration_s(rec) - 9.98) < 1e-6
    assert format_duration_display(analyzed_s=9.98, recording_s=10.01) == "9.98 s analyzed span"
    assert (
        format_duration_display(
            analyzed_s=9.98, recording_s=10.01, usable_s=7.61
        )
        == "7.61 s valid · 9.98 s analyzed span of 10.01 s source"
    )
