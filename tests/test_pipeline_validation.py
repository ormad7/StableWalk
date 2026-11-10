"""Tests for pipeline validation from on-disk OpenSim artifacts (no fabricated GT)."""

from __future__ import annotations

from pathlib import Path

import pytest

from stablewalk import config
from stablewalk.analysis.pipeline_validation import (
    QUALITY_LABEL,
    assess_pipeline_validation,
    compare_pose_mot_to_ik_mot,
    find_ik_marker_errors_sto,
    parse_ik_marker_errors,
)
from stablewalk.monitoring.pipeline_status import (
    PipelineStatusContext,
    assess_pipeline_status,
)

NORMAL_DIR = config.OPENSIM_DIR / "normal_gait"
NORMAL_STO = NORMAL_DIR / "normal_gait_mapped_for_opensim_ik_marker_errors.sto"
NORMAL_POSE_MOT = NORMAL_DIR / "normal_gait.mot"
NORMAL_IK_MOT = NORMAL_DIR / "normal_gait_ik.mot"


@pytest.mark.skipif(not NORMAL_STO.is_file(), reason="normal_gait IK marker-error STO missing")
def test_parse_ik_marker_errors_from_project_artifact() -> None:
    summary = parse_ik_marker_errors(NORMAL_STO)
    assert summary is not None
    assert summary.n_frames >= 2
    assert summary.mean_rms_m > 0
    assert summary.quality in QUALITY_LABEL
    assert "cm" in summary.summary_line()


@pytest.mark.skipif(
    not (NORMAL_POSE_MOT.is_file() and NORMAL_IK_MOT.is_file()),
    reason="normal_gait pose/IK MOT pair missing",
)
def test_pose_ik_knee_agreement_uses_flexion_conversion() -> None:
    agree = compare_pose_mot_to_ik_mot(NORMAL_POSE_MOT, NORMAL_IK_MOT)
    assert agree is not None
    assert agree.n_compared >= 4
    assert agree.rmse_mean_deg is not None
    assert agree.rmse_mean_deg >= 0
    assert "Coordinate agreement" in agree.note
    assert agree.quality in QUALITY_LABEL


def test_missing_files_do_not_fabricate_agreement(tmp_path: Path) -> None:
    missing_a = tmp_path / "a.mot"
    missing_b = tmp_path / "b.mot"
    assert compare_pose_mot_to_ik_mot(missing_a, missing_b) is None
    assert parse_ik_marker_errors(tmp_path / "none.sto") is None
    assert find_ik_marker_errors_sto(tmp_path) is None


def test_empty_context_stages_are_not_assessed() -> None:
    report = assess_pipeline_validation(PipelineStatusContext())
    assert report.ik_marker_errors is None
    assert report.pose_ik_agreement is None
    assert report.stages["osim_ik"].quality == "not_assessed"
    assert report.stages["bio_contact_mask"].quality == "not_assessed"


@pytest.mark.skipif(not NORMAL_DIR.is_dir(), reason="normal_gait OpenSim folder missing")
def test_pipeline_status_attaches_opensim_validation_for_normal_gait() -> None:
    ctx = PipelineStatusContext(
        source="normal_gait.mp4",
        run_name="normal_gait",
        detected_frames=100,
        total_frames=120,
        opensim_ik_completed=True,
        opensim_export_files={
            "TRC": True,
            "MOT": True,
            "JSON": True,
            "MAPPED": True,
            "IK_MOT": True,
            "IK_ERRORS": NORMAL_STO.is_file(),
        },
    )
    status = assess_pipeline_status(ctx)
    assert status.validation is not None
    assert status.schema_version == "1.1"
    ik_item = next(i for i in status.all_items() if i.key == "osim_ik")
    if NORMAL_STO.is_file():
        assert ik_item.quality != "not_assessed"
        assert ik_item.validation_note
        assert status.validation.ik_marker_errors is not None
    if NORMAL_POSE_MOT.is_file() and NORMAL_IK_MOT.is_file():
        mot_item = next(i for i in status.all_items() if i.key == "osim_mot")
        assert mot_item.quality != "not_assessed"
        assert status.validation.pose_ik_agreement is not None
        data = status.to_dict()
        assert data["validation"]["pose_ik_agreement"]["rmse_mean_deg"] is not None
