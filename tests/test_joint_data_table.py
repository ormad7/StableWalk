"""Per-frame joint table includes angle when the recording has it."""

from __future__ import annotations

import math
from types import SimpleNamespace

from stablewalk.models.gait_motion import (
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.storage.models import KinematicSample
from stablewalk.ui.joint_data_table import (
    JOINT_DATA_TABLE_COLUMNS,
    JOINT_DATA_TABLE_HEADINGS,
    collect_joint_data_rows,
    joint_data_row_from_sample,
    joint_data_row_from_snapshot,
    joint_data_rows_from_recording,
    write_joint_data_csv,
    write_joint_data_json,
)


def test_joint_table_columns_include_supervisor_fields() -> None:
    required = (
        "frame",
        "timestamp",
        "joint",
        "x",
        "y",
        "z",
        "angle",
        "velocity",
        "acceleration",
        "confidence",
        "gait_phase",
    )
    for col in required:
        assert col in JOINT_DATA_TABLE_COLUMNS
    assert JOINT_DATA_TABLE_HEADINGS["timestamp"] == "Time (s)"
    assert "visibility" in JOINT_DATA_TABLE_HEADINGS["confidence"].lower()
    assert JOINT_DATA_TABLE_HEADINGS["x"].endswith("(m)")
    assert JOINT_DATA_TABLE_HEADINGS["angle"].endswith("(°)")


def _hka_joints(flexion_deg: float, pos: tuple[float, float, float]) -> dict[str, JointSample]:
    x, y, z = pos
    rad = math.radians(float(flexion_deg))
    return {
        "right_hip": JointSample("right_hip", Vec3(x, y + 0.4, z)),
        "right_knee": JointSample(
            "right_knee",
            Vec3(x, y, z),
            angle_deg=180.0 - float(flexion_deg),
            velocity=0.4,
        ),
        "right_ankle": JointSample(
            "right_ankle",
            Vec3(x + 0.4 * math.sin(rad), y - 0.4 * math.cos(rad), z),
        ),
    }


def test_snapshot_row_carries_angle() -> None:
    snap = SkeletonSnapshot(
        frame_index=4,
        time_s=0.16,
        joints=_hka_joints(41.5, (0.1, 0.2, 0.3)),
    )
    row = joint_data_row_from_snapshot("right_knee", snap)
    assert row.frame == 5
    assert row.angle is not None
    assert abs(float(row.angle) - 41.5) < 1e-6
    display = row.as_display_tuple()
    assert display[JOINT_DATA_TABLE_COLUMNS.index("angle")] == "41.50"


def test_snapshot_row_omits_knee_angle_without_hka() -> None:
    snap = SkeletonSnapshot(
        frame_index=4,
        time_s=0.16,
        joints={
            "right_knee": JointSample(
                joint_id="right_knee",
                position=Vec3(0.1, 0.2, 0.3),
                angle_deg=41.5,
            )
        },
    )
    row = joint_data_row_from_snapshot("right_knee", snap)
    assert row.angle is None


def test_sample_row_uses_recorded_angle() -> None:
    sample = KinematicSample(
        frame_number=3,
        time_s=0.12,
        dof_name="Right Knee",
        joint_name="Right Knee",
        x=0.1,
        y=0.2,
        z=0.3,
        angle_deg=33.0,
        velocity=0.5,
    )
    row = joint_data_row_from_sample(sample, prev_sample=None)
    assert row.angle == 33.0
    assert row.acceleration is None
    assert row.gait_phase == "—"


def _knee_snap(frame: int, time_s: float, x: float, vel: float) -> SkeletonSnapshot:
    joints = _hka_joints(40.0 + frame, (x, 0.2, 0.3))
    joints["right_knee"] = JointSample(
        joint_id="right_knee",
        position=Vec3(x, 0.2, 0.3),
        angle_deg=180.0 - (40.0 + frame),
        velocity=vel,
    )
    return SkeletonSnapshot(
        frame_index=frame,
        time_s=time_s,
        joints=joints,
    )


def test_snapshot_row_uses_gait_cycle_phase() -> None:
    cycle = SimpleNamespace(
        frame_at=lambda _i: SimpleNamespace(phase="LEFT_STANCE")
    )
    row = joint_data_row_from_snapshot(
        "right_knee",
        _knee_snap(0, 0.0, 0.1, 0.4),
        gait_cycle=cycle,
    )
    assert row.gait_phase == "LEFT STANCE"


def test_recording_rows_cover_every_frame() -> None:
    recording = GaitMotionRecording(
        source="test",
        fps=25.0,
        snapshots=[
            _knee_snap(0, 0.00, 0.10, 0.4),
            _knee_snap(1, 0.04, 0.12, 0.5),
        ],
    )
    rows = joint_data_rows_from_recording(recording, {"right_knee"})
    assert len(rows) == 2
    assert rows[0].frame == 1
    assert rows[1].frame == 2
    assert rows[0].x == 0.10
    assert rows[1].velocity == 0.5
    assert rows[1].acceleration is not None
    assert abs(float(rows[1].acceleration) - ((0.5 - 0.4) / 0.04)) < 1e-9


def test_collect_prefers_recording_over_empty_samples(tmp_path) -> None:
    recording = GaitMotionRecording(
        source="test",
        fps=25.0,
        snapshots=[_knee_snap(0, 0.0, 0.1, 0.2)],
    )
    rows = collect_joint_data_rows(
        recording=recording,
        item_ids={"right_knee"},
        samples=[],
    )
    assert len(rows) == 1
    csv_path = write_joint_data_csv(rows, tmp_path / "joint.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "Time" in text.splitlines()[0]
    assert "Gait Phase" in text.splitlines()[0]
    json_path = write_joint_data_json(rows, tmp_path / "joint.json")
    payload = json_path.read_text(encoding="utf-8")
    assert '"time"' in payload
    assert '"gait_phase"' in payload


def test_all_joints_cover_every_available_frame_joint() -> None:
    from stablewalk.ui.joint_data_table import (
        JOINT_FILTER_ALL,
        available_joint_item_ids,
        collect_joint_data_rows,
        filter_joint_data_rows,
        paginate_joint_data_rows,
    )

    left = _knee_snap(0, 0.0, 0.1, 0.2)
    left.joints["left_knee"] = JointSample(
        "left_knee",
        Vec3(0.0, 0.2, 0.3),
        angle_deg=130.0,
    )
    right = _knee_snap(1, 0.04, 0.12, 0.3)
    right.joints["left_knee"] = JointSample(
        "left_knee",
        Vec3(0.01, 0.2, 0.3),
        angle_deg=132.0,
    )
    recording = GaitMotionRecording(
        source="test",
        fps=25.0,
        snapshots=[left, right],
    )
    available = available_joint_item_ids(recording)
    assert "right_knee" in available
    assert "left_knee" in available
    rows = collect_joint_data_rows(recording=recording, item_ids=available)
    assert len(rows) == len(available) * 2
    joints = {row.joint for row in rows}
    assert any("right" in name.lower() and "knee" in name.lower() for name in joints)
    assert any("left" in name.lower() and "knee" in name.lower() for name in joints)
    all_rows = filter_joint_data_rows(rows, JOINT_FILTER_ALL)
    assert len(all_rows) == len(rows)
    only_right = filter_joint_data_rows(rows, "Right Knee")
    assert only_right
    assert all("right" in row.joint.lower() and "knee" in row.joint.lower() for row in only_right)
    empty = filter_joint_data_rows(rows, "Not A Joint")
    assert empty == []
    page, index, count = paginate_joint_data_rows(rows, 0, page_size=3)
    assert index == 0
    assert count >= 1
    assert 1 <= len(page) <= 3


def test_missing_confidence_stays_unavailable() -> None:
    from stablewalk.ui.joint_data_table import (
        CONFIDENCE_SOURCE,
        CONFIDENCE_UNAVAILABLE_REASON,
    )

    row = joint_data_row_from_snapshot("right_knee", _knee_snap(0, 0.0, 0.1, 0.2))
    assert row.confidence is None
    assert row.confidence_source is None
    assert row.confidence_unavailable_reason == CONFIDENCE_UNAVAILABLE_REASON
    display = row.as_display_tuple()
    from stablewalk.ui.joint_data_table import JOINT_DATA_TABLE_COLUMNS

    assert display[JOINT_DATA_TABLE_COLUMNS.index("confidence")] == "—"
    assert CONFIDENCE_SOURCE == "detector_visibility"
