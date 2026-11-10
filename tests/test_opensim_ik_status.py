"""Canonical OpenSim IK process status — one source of truth for every UI surface."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.opensim_ik_status import (
    IK_COMPLETE,
    IK_FAILED,
    IK_NOT_STARTED,
    IK_PARTIAL,
    IK_RUNNING,
    resolve_opensim_ik_status,
    resolve_opensim_ik_status_from_gui,
)
from stablewalk.ui.tk.dashboard_advanced import collect_opensim_status_rows
from stablewalk.ui.tk.dashboard_status_bar import (
    _opensim_status_text,
    _opensim_tooltip_text,
)


def test_complete_only_from_successful_process() -> None:
    status = resolve_opensim_ik_status(ik_state="Completed")
    assert status.key == IK_COMPLETE
    assert status.label == "Complete"
    assert status.process_completed


def test_ik_mot_unavailable_reason_is_not_export_success() -> None:
    from stablewalk.opensim_ik_status import ik_mot_unavailable_reason

    export_only = resolve_opensim_ik_status(export_completed=True, mapped_trc_ready=True)
    reason = ik_mot_unavailable_reason(export_only)
    assert "unavailable" in reason.lower()
    assert "export" in reason.lower()
    assert "IK success" in reason or "not IK success" in reason

    leftover = ik_mot_unavailable_reason(export_only, ik_output_on_disk=True)
    assert "on disk" in leftover.lower()
    assert "not Inverse Kinematics success" in leftover


def test_file_presence_never_marks_complete() -> None:
    status = resolve_opensim_ik_status(
        ik_state=None,
        export_completed=True,
        mapped_trc_ready=True,
        trc_exported=True,
        mot_exported=True,
        ik_output_on_disk=True,
        sdk_available=True,
        model_valid=True,
        has_session=True,
    )
    assert status.key == IK_PARTIAL
    assert status.label == "Partial"
    assert not status.process_completed
    assert "file presence is not treated as" in status.tooltip


def test_running_and_failed_are_explicit() -> None:
    running = resolve_opensim_ik_status(ik_state="Running")
    failed = resolve_opensim_ik_status(ik_state="Failed", ik_message="solver error")
    assert running.key == IK_RUNNING
    assert running.label == "Running"
    assert failed.key == IK_FAILED
    assert failed.label == "Failed"
    assert failed.detail == "solver error"


def test_pipeline_complete_is_independent_of_ik() -> None:
    from stablewalk.opensim_ik_status import (
        PIPELINE_COMPLETE,
        resolve_opensim_pipeline_status,
        format_opensim_status_pair,
        resolve_opensim_ik_status,
    )

    pipeline = resolve_opensim_pipeline_status(
        export_completed=True,
        trc_exported=True,
        mot_exported=True,
    )
    ik = resolve_opensim_ik_status(export_completed=True, mapped_trc_ready=True)
    pair = format_opensim_status_pair(pipeline, ik)
    assert pipeline.key == PIPELINE_COMPLETE
    assert pipeline.label == "Complete"
    assert ik.label == "Partial"
    assert pair.short_label == "Export Complete · IK unavailable"
    assert "OpenSim Pipeline: Complete" in pair.tooltip
    assert "IK: Partial" in pair.tooltip
    assert "Succeeded: OpenSim export (TRC and MOT)" in pair.tooltip
    assert "Unavailable: Inverse Kinematics (not completed)" in pair.tooltip
    assert "not biomechanical validation" in pair.tooltip.lower()


def test_output_inventory_lists_complete_and_unavailable() -> None:
    from stablewalk.opensim_ik_status import format_opensim_output_inventory

    text = format_opensim_output_inventory(
        trc_exported=True,
        mot_exported=False,
        mapped_trc_ready=True,
        ik_output_on_disk=True,
        ik_complete=False,
    )
    assert "TRC: Complete" in text
    assert "MOT: Unavailable" in text
    assert "Mapped TRC: Complete" in text
    assert "IK .mot: Unavailable" in text
    assert "not treated as a completed IK run" in text


def test_reconstruction_notes_never_treat_partial_as_validation() -> None:
    from stablewalk.opensim_ik_status import format_ik_reconstruction_notes

    mapping = SimpleNamespace(
        coverage_percent=62.5,
        raw_coverage_percent=70.0,
        high_confidence_coverage_percent=40.0,
        mapped_matching_markers=["RASI", "LASI"],
        opensim_reference_markers=["RASI", "LASI", "RKNE", "LKNE"],
        missing_for_ik=["RKNE", "LKNE"],
        unavailable_markers=[],
        message="Experimental mapping from monocular landmarks.",
    )
    notes = format_ik_reconstruction_notes(
        mapping=mapping,
        residual_rms_m=None,
        ik_key=IK_PARTIAL,
        frame_count=120,
        ik_frames_solved=0,
    )
    assert "62.5%" in notes
    assert "Mapped markers: 2/4" in notes
    assert "IK frames solved: 0/120" in notes
    assert "RKNE" in notes
    assert "not complete biomechanical validation" in notes.lower()
    assert "unavailable" in notes.lower()
    none_notes = format_ik_reconstruction_notes(
        mapping=mapping,
        residual_rms_m=None,
        ik_key=IK_PARTIAL,
        frame_count=120,
        ik_frames_solved=None,
    )
    assert "IK frames solved: unavailable" in none_notes
    assert "IK fitting errors: unavailable" in none_notes


def test_ik_fit_notes_include_failed_frames() -> None:
    from stablewalk.opensim_ik_status import format_ik_frame_fit_notes

    summary = SimpleNamespace(
        n_frames=10,
        n_failed_frames=2,
        failed_frames=(
            {
                "frame_index": 3,
                "time_s": 0.12,
                "marker_error_rms_m": 0.21,
            },
            {
                "frame_index": 7,
                "time_s": 0.28,
                "marker_error_rms_m": 0.18,
            },
        ),
        summary_line=lambda: "IK marker RMS mean 8.0 cm (max 21.0 cm) over 10 frames; 2 frame(s) above 15 cm RMS",
    )
    text = format_ik_frame_fit_notes(summary)
    assert "8 within residual band / 2 failed / 10 total" in text
    assert "f3@0.12s" in text
    assert format_ik_frame_fit_notes(None).endswith("no marker-error .sto).")


def test_not_started_when_idle() -> None:
    status = resolve_opensim_ik_status()
    assert status.key == IK_NOT_STARTED
    assert status.label == "Not Started"


def test_advanced_and_footer_agree_when_export_done_ik_not_run() -> None:
    gui = SimpleNamespace(
        _pipeline_status_report_cache=None,
        _opensim_sdk_available=True,
        _opensim_sdk_version="4.5",
        _opensim_model_name="gait2392.osim",
        _opensim_model_valid=True,
        _opensim_ik_state=None,
        _opensim_ik_message="",
        _opensim_ik_block_reason=None,
        _opensim_export_completed=True,
        _opensim_status_override="Export completed successfully.",
        _presentation_mode=False,
        sequence=object(),
        gait_motion=None,
        _opensim_export_files={
            "TRC": True,
            "MOT": True,
            "JSON": True,
            "MAPPED": True,
            "IK_MOT": True,
        },
    )
    footer, _fg = _opensim_status_text(gui)
    rows = {row.key: row for row in collect_opensim_status_rows(gui)}
    resolved = resolve_opensim_ik_status_from_gui(gui)
    assert footer == "Export Complete · IK unavailable"
    assert rows["pipeline"].value == "Complete"
    assert rows["ik"].value == "Partial"
    assert "not Inverse Kinematics success" in rows["ik_output"].value
    assert "diagnostic log" in rows["ik_output"].value.lower()
    assert "export" in rows["ik_output"].tooltip.lower()
    assert "diagnostic log" in rows["ik_output"].tooltip.lower()
    assert resolved.label == rows["ik"].value
    assert resolved.detail in rows["ik"].tooltip
    assert "Partial means the IK process has not completed successfully" in rows["ik"].tooltip
    assert "IK fitting errors: unavailable" in rows["ik"].tooltip
    pair_tip = _opensim_tooltip_text(gui, footer)
    assert "Succeeded: OpenSim export (TRC and MOT)" in pair_tip
    assert "Unavailable: Inverse Kinematics (not completed)" in pair_tip
    assert resolved.detail in pair_tip

    from stablewalk.ui.tk.dashboard_advanced import _collect_engineering_stages
    from stablewalk.ui.tk.dashboard_results_pipeline import build_results_stage_views

    stages = _collect_engineering_stages(gui)
    views = {view.key: view for view in build_results_stage_views(gui, None)}
    assert stages["opensim"].detail == footer
    assert views["opensim"].detail == footer
    assert views["opensim"].status != "completed"
    assert stages["opensim"].status != "completed"


def test_advanced_and_footer_agree_when_ik_process_completed() -> None:
    gui = SimpleNamespace(
        _pipeline_status_report_cache=None,
        _opensim_sdk_available=True,
        _opensim_sdk_version="4.5",
        _opensim_model_name="gait2392.osim",
        _opensim_model_valid=True,
        _opensim_ik_state="Completed",
        _opensim_ik_message="out/ik.mot",
        _opensim_export_completed=True,
        _presentation_mode=False,
        sequence=object(),
        gait_motion=None,
    )
    footer, _fg = _opensim_status_text(gui)
    rows = {row.key: row for row in collect_opensim_status_rows(gui)}
    assert footer == "Complete"
    assert rows["pipeline"].value == "Complete"
    assert rows["ik"].value.startswith("Complete")
