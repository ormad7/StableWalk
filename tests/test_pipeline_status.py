"""Tests for pipeline status assessment."""

from __future__ import annotations

from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.monitoring.pipeline_status import (
    STATUS_COMPLETED,
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    PipelineStatusContext,
    assess_pipeline_status,
    build_pipeline_diagram,
)
from stablewalk.models.gait_motion import GaitMotionRecording, JointSample, SkeletonSnapshot, Vec3


def _snap(i: int) -> SkeletonSnapshot:
    joints = {
        "left_hip": JointSample("left_hip", Vec3(-0.1, 0.9, 0.0), angle_deg=10.0, velocity=0.1),
        "right_hip": JointSample("right_hip", Vec3(0.1, 0.9, 0.0), angle_deg=12.0, velocity=0.1),
        "left_ankle": JointSample("left_ankle", Vec3(-0.1, 0.2, 0.0), velocity_vector=Vec3(0.01, 0, 0)),
        "right_ankle": JointSample("right_ankle", Vec3(0.1, 0.2, 0.0), velocity_vector=Vec3(0.01, 0, 0)),
    }
    return SkeletonSnapshot(frame_index=i, time_s=i / 30.0, joints=joints, dofs={})


def test_empty_session_all_unavailable() -> None:
    report = assess_pipeline_status(PipelineStatusContext())
    items = report.all_items()
    assert items
    assert all(item.status == STATUS_UNAVAILABLE for item in items)


def test_mediapipe_completed_with_poses() -> None:
    rec = GaitMotionRecording(source="t", fps=30.0, snapshots=[_snap(i) for i in range(10)])
    ctx = PipelineStatusContext(
        source="walk.mp4",
        run_name="walk",
        pose_backend_used="mediapipe",
        detected_frames=8,
        total_frames=10,
        gait_motion=rec,
    )
    report = assess_pipeline_status(ctx)
    mp = next(i for i in report.all_items() if i.key == "pose_mediapipe")
    assert mp.status == STATUS_COMPLETED


def test_smpl_not_executed_unavailable() -> None:
    ctx = PipelineStatusContext(
        pose_backend_used="mediapipe",
        detected_frames=5,
        total_frames=10,
        gait_motion=GaitMotionRecording(source="t", fps=30.0, snapshots=[_snap(0)]),
    )
    report = assess_pipeline_status(ctx)
    smpl = next(i for i in report.all_items() if i.key == "pose_smpl")
    assert smpl.status == STATUS_UNAVAILABLE


def test_isaac_training_never_completed() -> None:
    ctx = PipelineStatusContext(isaac_lab_available=False)
    report = assess_pipeline_status(ctx)
    running = next(i for i in report.all_items() if i.key == "isaac_training_running")
    complete = next(i for i in report.all_items() if i.key == "isaac_training_complete")
    assert running.status == STATUS_UNAVAILABLE
    assert complete.status == STATUS_UNAVAILABLE


def test_report_serializes_to_dict() -> None:
    report = assess_pipeline_status(
        PipelineStatusContext(source="demo.mp4", detected_frames=1, total_frames=2)
    )
    data = report.to_dict()
    assert data["kind"] == "pipeline_status"
    assert "groups" in data
    assert data["groups"][0]["stages"]
    assert "diagram" in data
    assert len(data["diagram"]) == 8


def test_pipeline_diagram_empty_session_all_unavailable() -> None:
    report = assess_pipeline_status(PipelineStatusContext())
    diagram = build_pipeline_diagram(report)
    assert len(diagram) == 8
    assert [stage.label for stage in diagram] == [
        "Pose Estimation",
        "3D Reconstruction",
        "Biomechanics",
        "OpenSim",
        "Retargeting",
        "Estimated Virtual GRF",
        "Export",
        "Isaac Lab",
    ]
    assert all(stage.status == STATUS_UNAVAILABLE for stage in diagram)


def test_pose_estimation_completed_only_when_executed() -> None:
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
    pose = next(stage for stage in build_pipeline_diagram(report) if stage.key == "pose_estimation")
    assert pose.status == STATUS_COMPLETED

    smpl_only = assess_pipeline_status(
        PipelineStatusContext(
            pose_backend_used="mediapipe",
            detected_frames=8,
            total_frames=10,
            gait_motion=rec,
        )
    )
    smpl_item = next(i for i in smpl_only.all_items() if i.key == "pose_smpl")
    assert smpl_item.status == STATUS_UNAVAILABLE
    pose2 = next(stage for stage in build_pipeline_diagram(smpl_only) if stage.key == "pose_estimation")
    assert pose2.status == STATUS_COMPLETED


def test_virtual_grf_separate_from_biomechanics() -> None:
    bio = BiomechanicalAnalysisResult()
    report = assess_pipeline_status(
        PipelineStatusContext(
            detected_frames=5,
            total_frames=10,
            gait_motion=GaitMotionRecording(source="t", fps=30.0, snapshots=[_snap(0)]),
            biomech=bio,
        )
    )
    diagram = {stage.key: stage for stage in build_pipeline_diagram(report)}
    assert diagram["virtual_grf"].status == STATUS_UNAVAILABLE
    assert diagram["biomechanics"].status in (STATUS_UNAVAILABLE, STATUS_PARTIAL)


def test_export_not_completed_without_artifacts() -> None:
    report = assess_pipeline_status(
        PipelineStatusContext(
            detected_frames=5,
            total_frames=10,
            gait_motion=GaitMotionRecording(source="t", fps=30.0, snapshots=[_snap(0)]),
        )
    )
    export_stage = next(stage for stage in build_pipeline_diagram(report) if stage.key == "export")
    assert export_stage.status != STATUS_COMPLETED


def test_diagram_stage_has_tooltip() -> None:
    report = assess_pipeline_status(PipelineStatusContext())
    stage = build_pipeline_diagram(report)[0]
    assert stage.tooltip
    assert stage.display_line().startswith("\u2713") or stage.display_line().startswith("\u26a0") or stage.display_line().startswith("\u2717")


def test_opensim_pipeline_details_checklist() -> None:
    report = assess_pipeline_status(
        PipelineStatusContext(
            opensim_sdk_available=True,
            opensim_model_loaded=True,
            opensim_model_name="subject01_simbody.osim",
            opensim_ik_completed=True,
            opensim_mapping_status="improved",
            opensim_mapping_percent=88.0,
            opensim_coverage_percent=91.0,
            opensim_export_files={
                "TRC": True,
                "MOT": True,
                "JSON": True,
                "MAPPED": True,
                "IK_MOT": True,
            },
        )
    )
    by_key = {item.key: item for item in report.all_items()}
    assert by_key["osim_model"].status == STATUS_COMPLETED
    assert "Model loaded" in by_key["osim_model"].detail
    assert by_key["osim_trc"].detail == "TRC exported"
    assert by_key["osim_mot"].detail == "MOT exported"
    assert by_key["osim_ik"].detail == "IK completed"
    assert "Marker mapping 88%" in by_key["osim_mapping"].detail
    assert "Coverage 91%" in by_key["osim_coverage"].detail

    diagram = {stage.key: stage for stage in build_pipeline_diagram(report)}
    osim = diagram["opensim"]
    assert osim.status == STATUS_COMPLETED
    assert "Model loaded" in osim.detail
    assert "TRC exported" in osim.detail
    assert "MOT exported" in osim.detail
    assert "IK completed" in osim.detail
    assert "Partial" not in osim.detail


def test_opensim_ik_block_reason_in_pipeline() -> None:
    reason = "IK cannot run: mapped TRC missing — click Export OpenSim Files"
    report = assess_pipeline_status(
        PipelineStatusContext(
            opensim_sdk_available=True,
            opensim_model_loaded=True,
            opensim_model_name="gait2392.osim",
            opensim_ik_block_reason=reason,
            opensim_mapping_status="partial",
            opensim_mapping_percent=55.0,
            opensim_coverage_percent=48.0,
            opensim_export_files={
                "TRC": True,
                "MOT": True,
                "JSON": True,
                "MAPPED": False,
                "IK_MOT": False,
            },
        )
    )
    ik = next(i for i in report.all_items() if i.key == "osim_ik")
    assert ik.status == STATUS_PARTIAL
    assert ik.detail == reason

    osim = next(s for s in build_pipeline_diagram(report) if s.key == "opensim")
    assert reason in osim.detail
    assert "Marker mapping 55%" in osim.detail
    assert "Coverage 48%" in osim.detail
