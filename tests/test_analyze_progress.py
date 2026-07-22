"""Unit tests for Analyze progress stages and formatting helpers."""

from __future__ import annotations

from stablewalk.ui.tk.analyze_progress import (
    ANALYZE_STAGES,
    format_elapsed,
    format_fps,
    format_frames,
    infer_stage_from_message,
    stage_fraction,
    stage_label,
)


def test_analyze_stages_match_user_facing_order() -> None:
    labels = [label for _, label in ANALYZE_STAGES]
    assert labels == [
        "Loading video...",
        "Extracting frames...",
        "Pose estimation...",
        "3D reconstruction...",
        "Joint angles...",
        "Biomechanics...",
        "Virtual GRF...",
        "OpenSim...",
        "Generating report...",
        "Completed.",
    ]


def test_stage_label_known_and_unknown() -> None:
    assert stage_label("pose") == "Pose estimation..."
    assert "Virtual" in stage_label("virtual_grf")
    assert stage_label("custom_stage").endswith("...")


def test_infer_stage_from_message_keywords() -> None:
    assert infer_stage_from_message("Loading video...", 0.05) == "loading"
    assert infer_stage_from_message("Extracting frames...", 0.2) == "extracting"
    assert infer_stage_from_message("Pose estimation...", 0.4) == "pose"
    assert infer_stage_from_message("3D reconstruction...", 0.6) == "reconstruction"
    assert infer_stage_from_message("Joint angles...", 0.7) == "joint_angles"
    assert infer_stage_from_message("Biomechanics...", 0.8) == "biomechanics"
    assert infer_stage_from_message("Virtual GRF...", 0.85) == "virtual_grf"
    assert infer_stage_from_message("OpenSim export", 0.9) == "opensim"
    assert infer_stage_from_message("Generating report...", 0.96) == "report"
    assert infer_stage_from_message("Completed.", 1.0) == "completed"


def test_infer_stage_from_fraction_fallback() -> None:
    assert infer_stage_from_message("", 0.08) == "loading"
    assert infer_stage_from_message("", 0.2) == "extracting"
    assert infer_stage_from_message("", 0.4) == "pose"
    assert infer_stage_from_message("", 0.6) == "reconstruction"
    assert infer_stage_from_message("", 0.7) == "joint_angles"
    assert infer_stage_from_message("", 0.8) == "biomechanics"
    assert infer_stage_from_message("", 0.86) == "virtual_grf"
    assert infer_stage_from_message("", 0.92) == "opensim"
    assert infer_stage_from_message("", 0.97) == "report"
    assert infer_stage_from_message("", 1.0) == "completed"


def test_stage_fraction_increases_with_stage() -> None:
    loading = stage_fraction("loading", within=0.0)
    pose = stage_fraction("pose", within=0.0)
    report = stage_fraction("report", within=0.0)
    done = stage_fraction("completed", within=1.0)
    assert 0.0 <= loading < pose < report < done <= 1.0


def test_format_helpers() -> None:
    assert format_elapsed(None) == "—"
    assert format_elapsed(3.2) == "3.2s"
    assert format_elapsed(45) == "45s"
    assert format_elapsed(125) == "2m 05s"
    assert format_frames(None, None) == "—"
    assert format_frames(12, 100) == "12 / 100"
    assert format_frames(None, 100) == "— / 100"
    assert format_frames(12, None) == "12"
    assert format_fps(None) == "—"
    assert format_fps(0) == "—"
    assert format_fps(29.97) == "30.0"
    assert format_fps(120) == "120"
