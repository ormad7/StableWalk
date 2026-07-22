"""Tests for clickable pipeline stage detail dialogs."""

from __future__ import annotations

from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_UNAVAILABLE,
    PipelineStatusContext,
    assess_pipeline_status,
    build_pipeline_diagram,
)
from stablewalk.ui.tk.pipeline_stage_dialog import (
    build_pipeline_stage_dialog_content,
    resolve_stage_ui_spec,
)


def test_resolve_stage_ui_spec_flow_keys() -> None:
    spec = resolve_stage_ui_spec("video")
    assert spec.title == "Video"
    assert "pose_mediapipe" in spec.item_keys

    amp = resolve_stage_ui_spec("amp_dataset")
    assert amp.title == "AMP Dataset"


def test_build_dialog_content_empty_session() -> None:
    report = assess_pipeline_status(PipelineStatusContext())
    content = build_pipeline_stage_dialog_content("video", report)
    assert content.title == "Video"
    assert content.purpose
    assert content.input_desc
    assert content.output_desc
    assert content.algorithms
    assert content.duration_text
    assert content.status == STATUS_UNAVAILABLE
    assert "Purpose" not in content.evidence
    assert content.generated_files.startswith("No matching") or "•" in content.generated_files


def test_resolve_stage_ui_spec_includes_algorithms() -> None:
    spec = resolve_stage_ui_spec("retargeting")
    assert "scaling" in spec.algorithms.lower() or "DOF" in spec.algorithms


def test_build_dialog_content_from_diagram_stage() -> None:
    report = assess_pipeline_status(PipelineStatusContext())
    stage = build_pipeline_diagram(report)[0]
    content = build_pipeline_stage_dialog_content(
        stage.key,
        report,
        diagram_stage=stage,
    )
    assert content.title == stage.label
    assert content.status == stage.status


def test_build_dialog_content_mediapipe_evidence() -> None:
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
            run_name="walk",
            pose_backend_used="mediapipe",
            detected_frames=8,
            total_frames=10,
            gait_motion=rec,
        )
    )
    content = build_pipeline_stage_dialog_content("video", report)
    assert content.status in (STATUS_COMPLETED, STATUS_UNAVAILABLE)
    assert "MediaPipe" in content.evidence or "pose" in content.evidence.lower()
