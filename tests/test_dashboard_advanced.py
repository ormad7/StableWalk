"""Tests for Advanced & Export engineering dashboard."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.dashboard_advanced import (
    ENGINEERING_STATUS_LABEL,
    ENGINEERING_STATUS_SYMBOL,
    _status_display,
    collect_opensim_status_rows,
)


def test_engineering_status_labels_match_spec() -> None:
    # Text-first lab chrome (color strip carries meaning; no glyph prefix).
    assert ENGINEERING_STATUS_SYMBOL["completed"] == ""
    assert ENGINEERING_STATUS_SYMBOL["partial"] == ""
    assert ENGINEERING_STATUS_SYMBOL["unavailable"] == ""
    assert ENGINEERING_STATUS_LABEL["completed"] == "Completed"
    assert ENGINEERING_STATUS_LABEL["partial"] == "Partial"
    assert ENGINEERING_STATUS_LABEL["unavailable"] == "Missing"


def test_status_display_formats_badge() -> None:
    assert _status_display("completed") == "Completed"
    assert _status_display("partial") == "Partial"
    assert _status_display("unavailable") == "Missing"


def test_collect_opensim_status_rows_uses_pipeline_snapshot() -> None:
    items = [
        SimpleNamespace(
            key="osim_model",
            status="completed",
            detail="Model loaded (gait2392)",
            tooltip="model tip",
            display_line=lambda: "Completed",
        ),
        SimpleNamespace(
            key="osim_trc",
            status="completed",
            detail="TRC exported",
            tooltip="trc tip",
            display_line=lambda: "Completed",
        ),
        SimpleNamespace(
            key="osim_mot",
            status="partial",
            detail="MOT not exported",
            tooltip="mot tip",
            display_line=lambda: "Partial",
        ),
        SimpleNamespace(
            key="osim_ik",
            status="unavailable",
            detail="IK not executed",
            tooltip="ik tip",
            display_line=lambda: "Missing",
        ),
    ]
    gui = SimpleNamespace(
        _pipeline_status_report_cache=SimpleNamespace(all_items=lambda: items),
        _opensim_sdk_available=True,
        _opensim_sdk_version="4.5",
        _opensim_model_name="gait2392.osim",
        _opensim_model_valid=True,
    )
    rows = {row.key: row for row in collect_opensim_status_rows(gui)}
    assert rows["sdk"].value.startswith("Installed")
    assert rows["sdk"].status == "completed"
    assert rows["model"].value == "gait2392.osim"
    assert rows["trc"].status == "completed"
    assert rows["mot"].status == "partial"
    assert rows["pipeline"].value == "Not Started"
    assert rows["ik"].status == "unavailable"
    assert rows["ik"].value == "Not Started"
