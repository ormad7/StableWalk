"""Tests for ReportLab professional session PDF export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stablewalk.analysis.analysis_summary import AnalysisSummary, SummaryField
from stablewalk.analysis.biomechanical.com_estimation import CenterOfMassAnalysis
from stablewalk.analysis.biomechanical.joint_rom import JointROMAnalysis
from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.analysis.biomechanical.types import BiomechanicalFrameCOM, JointROMStats
from stablewalk.io.session_pdf_report import (
    SessionPdfReportContext,
    build_automatic_conclusion,
    build_recommendations,
    export_session_pdf_report,
    generate_project_logo,
)
from stablewalk.models.pose_data import JointAngles, Keypoint, PoseFrame, PoseSequence


def _minimal_sequence() -> PoseSequence:
    frames = []
    for i in range(12):
        t = i / 30.0
        kps = [
            Keypoint(name=f"j{j}", x=0.1 * j, y=0.5 + 0.01 * i, z=0.0, visibility=0.9)
            for j in range(8)
        ]
        angles = JointAngles(
            left_knee=40.0 + i,
            right_knee=42.0 + i,
            left_hip=20.0 + 0.5 * i,
            right_hip=21.0 + 0.5 * i,
        )
        frames.append(
            PoseFrame(
                frame_index=i,
                image_path=f"frame_{i:04d}.jpg",
                timestamp_s=t,
                detected=True,
                keypoints=kps,
                joint_angles=angles,
            )
        )
    return PoseSequence(source_video="demo.mp4", fps=30.0, frames=frames)


def test_generate_project_logo(tmp_path: Path) -> None:
    logo = generate_project_logo(tmp_path / "logo.png")
    assert logo.is_file()
    assert logo.stat().st_size > 0


def test_export_session_pdf_report_writes_pdf(tmp_path: Path) -> None:
    seq = _minimal_sequence()
    summary = AnalysisSummary(
        source="demo.mp4",
        overall_gait_quality=SummaryField("Gait quality", "72/100", "estimated", True, ""),
        cadence=SummaryField("Cadence", "110 spm", "estimated", True, ""),
        walking_speed=SummaryField("Speed", "1.10 m/s", "estimated", True, ""),
        symmetry=SummaryField("Symmetry", "88%", "estimated", True, ""),
        stability_margin=SummaryField("Stability", "High", "estimated", True, ""),
        center_of_mass=SummaryField("COM", "Normal oscillation", "estimated", True, ""),
        scientific_interpretation="Session shows moderate cadence with favourable symmetry.",
        pipeline_status={
            "diagram": [
                {
                    "key": "pose_estimation",
                    "label": "Pose Estimation",
                    "status": "completed",
                    "detail": "MediaPipe completed",
                }
            ]
        },
    )
    com_frames = [
        BiomechanicalFrameCOM(
            frame_index=i,
            time_s=i / 30.0,
            position=(0.05 * i, 0.9 + 0.01 * np.sin(i), 0.1 * i),
            velocity=(0.0, 0.0, 0.0),
            acceleration=(0.0, 0.0, 0.0),
            confidence=0.9,
        )
        for i in range(20)
    ]
    ba = BiomechanicalAnalysisResult(
        joint_rom=JointROMAnalysis(
            joints=[
                JointROMStats(
                    joint="knee",
                    side="left",
                    flexion_min_deg=10.0,
                    flexion_max_deg=55.0,
                    rom_deg=45.0,
                    confidence=0.8,
                )
            ]
        ),
        center_of_mass=CenterOfMassAnalysis(per_frame=com_frames, fps=30.0),
    )

    ctx = SessionPdfReportContext(
        source="demo.mp4",
        run_name="demo",
        session_label="Demo Walk",
        patient_name="Test Subject",
        patient_id="SW-001",
        generated_at="2026-07-16 12:00:00 UTC",
        fps=30.0,
        n_frames=12,
        detected_frames=12,
        duration_s=0.37,
        selected_joints=["Left Knee"],
        summary=summary,
        biomechanical=ba,
        sequence=seq,
        pipeline_status=summary.pipeline_status,
    )
    ctx.conclusion = build_automatic_conclusion(ctx)
    ctx.recommendations = build_recommendations(ctx)

    out = tmp_path / "session_report_test.pdf"
    written = export_session_pdf_report(ctx, out, logo_path=tmp_path / "logo.png")
    assert written.is_file()
    data = written.read_bytes()
    assert data[:4] == b"%PDF"
    assert len(data) > 2000
