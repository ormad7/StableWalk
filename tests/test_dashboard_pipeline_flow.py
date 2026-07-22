"""Tests for horizontal Real-to-Sim pipeline flow diagram."""

from __future__ import annotations

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    PipelineStageItem,
    assess_pipeline_status,
    PipelineStatusContext,
)
from stablewalk.ui.tk.dashboard_pipeline_flow import (
    FLOW_STAGE_SPECS,
    FLOW_STATUS_FG,
    _aggregate_item_status,
    _status_badge,
    collect_flow_stage_views,
)
from stablewalk.ui.tk.dashboard_pipeline_visual import PIPELINE_STATUS_FG
from stablewalk.ui.theme import MUTED


def _item(key: str, status: str, detail: str = "ok") -> PipelineStageItem:
    return PipelineStageItem(
        key=key,
        label=key,
        group="test",
        status=status,  # type: ignore[arg-type]
        detail=detail,
        tooltip=f"tooltip for {key}",
    )


def test_flow_has_eight_stages_in_order() -> None:
    assert [spec[1] for spec in FLOW_STAGE_SPECS] == [
        "Video",
        "SMPL",
        "Retargeting",
        "OpenSim",
        "Biomechanics",
        "Estimated Virtual GRF",
        "AMP Dataset",
        "Export",
    ]


def test_aggregate_item_status_completed_when_all_complete() -> None:
    items = {
        "retarget_scale": _item("retarget_scale", STATUS_COMPLETED, "scaled"),
        "retarget_joints": _item("retarget_joints", STATUS_COMPLETED, "mapped"),
    }
    status, detail = _aggregate_item_status(items, ("retarget_scale", "retarget_joints"))
    assert status == STATUS_COMPLETED
    assert "scaled" in detail


def test_aggregate_item_status_partial_when_mixed() -> None:
    items = {
        "pose_smpl": _item("pose_smpl", STATUS_PARTIAL, "no npz"),
    }
    status, _detail = _aggregate_item_status(items, ("pose_smpl",))
    assert status == STATUS_PARTIAL


def test_collect_flow_stage_views_empty_session() -> None:
    report = assess_pipeline_status(PipelineStatusContext())
    views = collect_flow_stage_views(report)
    assert len(views) == 8
    assert views[0].title == "Video"
    assert views[0].status == STATUS_UNAVAILABLE
    assert views[1].title == "SMPL"


def test_status_badge_uses_spec_labels() -> None:
    assert "Completed" in _status_badge(STATUS_COMPLETED)
    assert "Partial" in _status_badge(STATUS_PARTIAL)
    assert "Missing" in _status_badge(STATUS_UNAVAILABLE)


def test_flow_status_colors_match_shared_palette() -> None:
    assert FLOW_STATUS_FG is PIPELINE_STATUS_FG
    assert FLOW_STATUS_FG[STATUS_UNAVAILABLE] == MUTED


def test_collect_flow_from_report_with_video() -> None:
    from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, SkeletonSnapshot, Vec3

    def _snap(i: int) -> SkeletonSnapshot:
        joints = {
            "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0.0)),
            "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0.0)),
        }
        return SkeletonSnapshot(frame_index=i, time_s=i / 30.0, joints=joints, dofs={})

    rec = GaitMotionRecording(source="t", fps=30.0, snapshots=[_snap(i) for i in range(10)])
    report = assess_pipeline_status(
        PipelineStatusContext(
            source="walk.mp4",
            pose_backend_used="mediapipe",
            detected_frames=8,
            total_frames=10,
            gait_motion=rec,
        )
    )
    views = collect_flow_stage_views(report)
    video = next(v for v in views if v.key == "video")
    assert video.status in (STATUS_COMPLETED, STATUS_PARTIAL)
