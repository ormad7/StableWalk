"""Tests for Results/Advanced tab simplification helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from stablewalk.monitoring.pipeline_status import (
    STATUS_PARTIAL,
    STATUS_UNAVAILABLE,
    PipelineStageItem,
    _opensim_diagram_detail,
    _retargeting_diagram_detail,
)
from stablewalk.ui.tk.generated_files_panel import (
    GeneratedFileEntry,
    entry_belongs_to_session,
    filter_generated_files,
    paginate_entries,
)


def _entry(
    name: str,
    *,
    file_type: str = "JSON Export",
    status: str = "Ready",
    run: str | None = None,
) -> GeneratedFileEntry:
    base = Path("data/output/opensim") / (run or "other") / name
    return GeneratedFileEntry(
        path=base,
        filename=name,
        file_type=file_type,
        status=status,
        size_bytes=10,
        created_at=datetime(2026, 1, 1, 12, 0, 0),
    )


def test_filter_current_session_only() -> None:
    entries = [
        _entry("run1.trc", file_type="OpenSim", run="run1"),
        _entry("run2.trc", file_type="OpenSim", run="run2"),
        _entry("analysis_report.txt", file_type="Text Report", run="misc"),
    ]
    filtered = filter_generated_files(
        entries, session_only=True, run_name="run1"
    )
    assert [e.filename for e in filtered] == ["run1.trc"]
    assert entry_belongs_to_session(entries[0], "run1")
    assert not entry_belongs_to_session(entries[1], "run1")


def test_filter_type_and_status() -> None:
    entries = [
        _entry("a.trc", file_type="OpenSim", status="Ready", run="run1"),
        _entry("b.json", file_type="Gait Metrics", status="Empty", run="run1"),
    ]
    by_type = filter_generated_files(entries, type_filter="OpenSim")
    assert len(by_type) == 1
    by_status = filter_generated_files(entries, status_filter="Empty")
    assert len(by_status) == 1


def test_paginate_entries() -> None:
    entries = [_entry(f"f{i}.json", run="run1") for i in range(30)]
    page0, page, total = paginate_entries(entries, page=0, page_size=25)
    assert page == 0
    assert total == 2
    assert len(page0) == 25
    page1, page, total = paginate_entries(entries, page=1, page_size=25)
    assert page == 1
    assert len(page1) == 5


def test_generated_files_count_status_copy() -> None:
    from stablewalk.ui.tk.generated_files_panel import (
        format_generated_files_count,
        format_generated_files_empty_copy,
        format_generated_files_summary,
        generated_files_toggle_label,
    )

    assert format_generated_files_count(0) == "0 generated files"
    assert format_generated_files_count(1) == "1 generated file"
    assert format_generated_files_count(5) == "5 generated files"
    assert format_generated_files_summary(0, loaded=False) == (
        "No generated files loaded. Click Show to scan the current session."
    )
    assert format_generated_files_summary(0, loaded=True) == (
        "No generated files found for this session."
    )
    assert "Show to open" in format_generated_files_summary(5, loaded=True, expanded=False)
    assert "select a file to open" in format_generated_files_summary(
        5, loaded=True, expanded=True
    )
    assert format_generated_files_empty_copy(loaded=False) == (
        "No generated files loaded. Click Show to scan the current session."
    )
    assert format_generated_files_empty_copy(
        loaded=True, all_count=0, filtered_count=0
    ) == "No generated files found for this session."
    assert "filters" in format_generated_files_empty_copy(
        loaded=True, all_count=3, filtered_count=0
    )
    assert "Show" in generated_files_toggle_label(expanded=False)
    assert "Hide" in generated_files_toggle_label(expanded=True)


def test_opensim_partial_and_missing_explanations() -> None:
    items = {
        "osim_trc": PipelineStageItem(
            key="osim_trc",
            label="OpenSim TRC",
            group="OpenSim",
            status=STATUS_PARTIAL,
            detail="TRC not exported",
            tooltip="",
        )
    }
    partial = _opensim_diagram_detail(items, status=STATUS_PARTIAL)
    assert "Partial" in partial
    assert "Export OpenSim" in partial or "TRC" in partial
    missing = _opensim_diagram_detail({}, status=STATUS_UNAVAILABLE)
    assert "Missing" in missing


def test_retargeting_missing_explanation() -> None:
    text = _retargeting_diagram_detail({}, status=STATUS_UNAVAILABLE)
    assert "Missing" in text
    assert "Real-to-Sim" in text or "retarget" in text.lower()
