"""Results page layout — pipeline expands when report preview is absent."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_page_header_actions_grid_into_title_row() -> None:
    """Export actions must be real children of the title row (no empty page column)."""
    from pathlib import Path

    source = Path(
        "stablewalk/ui/tk/dashboard_page_frame.py"
    ).read_text(encoding="utf-8")
    assert "actions_host" in source
    assert "title_row" in source
    assert "in_=title_row" not in source


def test_results_preview_requires_session_source() -> None:
    from stablewalk.ui.tk.dashboard_summary import _results_preview_text

    empty = SimpleNamespace(
        source="",
        scientific_interpretation="Placeholder interpretation without a session.",
        overall_gait_quality=SimpleNamespace(available=False, value="—"),
    )
    assert _results_preview_text(empty) == ""
    ready = SimpleNamespace(
        source="walk.mp4",
        scientific_interpretation=(
            "Gait looks consistent. Estimates come from monocular pose data."
        ),
        overall_gait_quality=SimpleNamespace(available=True, value="72"),
    )
    preview = _results_preview_text(ready)
    headings = (
        "Summary",
        "Gait Quality",
        "Stability",
        "Symmetry",
        "Biomechanics",
        "Confidence",
        "Limitations",
    )
    assert all(heading in preview for heading in headings)
    assert [preview.index(heading) for heading in headings] == sorted(
        preview.index(heading) for heading in headings
    )
    assert "Gait looks consistent." in preview
    assert "Estimates come from monocular pose data." in preview
    assert "Gait quality: 72" in preview
    assert "derived or estimated from available analysis data" in preview
    assert "pipeline stage completion is reported separately" not in preview


def test_partial_preview_marks_every_available_metric_as_interim_derived() -> None:
    from stablewalk.ui.tk.dashboard_summary import _results_preview_text

    summary = SimpleNamespace(
        source="walk.mp4",
        scientific_interpretation="Available signals were summarized.",
        overall_gait_quality=SimpleNamespace(
            label="Gait quality",
            available=True,
            value="72/100",
            reason="",
        ),
        cadence=SimpleNamespace(
            label="Cadence",
            available=True,
            value="108 steps/min",
            reason="",
        ),
    )

    preview = _results_preview_text(
        summary,
        session_state="partial",
        metrics_status="unavailable",
    )

    assert "Interim derived values only" in preview
    assert "Metrics pipeline stage has not run" in preview
    assert "not a completed report" in preview
    assert "Interim derived — Gait quality: 72/100" in preview
    assert "Interim derived — Cadence: 108 steps/min" in preview


def test_partial_results_metric_badge_is_interim_derived() -> None:
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.dashboard_summary import _apply_summary_field

    value_label = MagicMock()
    tier_label = MagicMock()
    field = SimpleNamespace(
        label="Cadence",
        value="108 steps/min",
        available=True,
        reason="From detected gait-cycle timing",
        tier="calculated",
    )
    gui = SimpleNamespace(
        _results_pipeline_session_state="partial",
        _biomech=None,
        _biomech_analysis=None,
    )

    _apply_summary_field(
        "cadence",
        field,
        {"cadence": value_label},
        {"cadence": tier_label},
        gui=gui,
    )

    assert tier_label.configure.call_args.kwargs["text"] == "Interim derived"


def test_sync_results_report_preview_expands_pipeline_when_empty() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_summary import (
        RESULTS_REPORT_PREVIEW_WIDTH,
        sync_results_report_preview,
    )

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    try:
        content = tk.Frame(root, width=1000, height=400)
        content.grid_propagate(False)
        content.grid(row=0, column=0)
        content.columnconfigure(0, weight=1)
        pipe = tk.Frame(content, bg="#111")
        pipe.grid(row=0, column=0, sticky="nsew")
        preview = tk.Frame(content, width=RESULTS_REPORT_PREVIEW_WIDTH)
        preview.grid_propagate(False)
        gui = SimpleNamespace(
            _results_content_row=content,
            _results_preview_host=preview,
            _results_preview_visible=True,  # force a collapse transition
            lbl_results_preview_body=tk.Label(preview, text="stale"),
            lbl_results_preview_source=tk.Label(preview, text="stale"),
            _results_pipeline_host=None,
            _analysis_summary_cache=None,
        )
        preview.grid(row=0, column=1, sticky="nsew")
        root.update_idletasks()

        visible = sync_results_report_preview(gui, None)
        assert visible is False
        assert gui._results_preview_visible is False
        assert preview.winfo_manager() == ""
        assert content.grid_columnconfigure(1)["minsize"] == 0
        assert content.grid_columnconfigure(0)["weight"] == 1
    finally:
        root.destroy()


def test_sync_results_report_preview_shows_rail_when_report_exists() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_summary import (
        RESULTS_REPORT_PREVIEW_WIDTH,
        results_preview_rail_floor,
        sync_results_report_preview,
    )

    try:
        root = tk.Tk()
    except tk.TclError:  # pragma: no cover
        pytest.skip("no display")
    root.withdraw()
    try:
        content = tk.Frame(root, width=1000, height=400)
        content.grid_propagate(False)
        content.grid(row=0, column=0)
        content.columnconfigure(0, weight=1)
        preview = tk.Frame(content, width=RESULTS_REPORT_PREVIEW_WIDTH)
        preview.grid_propagate(False)
        body = tk.Text(preview)
        source = tk.Label(preview, text="")
        preview_card = tk.LabelFrame(content, text="Report Preview")
        gui = SimpleNamespace(
            _results_content_row=content,
            _results_preview_host=preview,
            _results_preview_visible=False,
            _results_preview_card=preview_card,
            _results_pipeline_session_state="partial",
            _results_metrics_stage_status="unavailable",
            lbl_results_preview_body=body,
            lbl_results_preview_source=source,
            _results_pipeline_host=None,
        )
        summary = SimpleNamespace(
            source="walk.mp4",
            scientific_interpretation="Gait looks consistent with walking.",
        )
        assert sync_results_report_preview(gui, summary) is True
        assert preview.winfo_manager() == "grid"
        expected_floor = results_preview_rail_floor(content.winfo_width())
        assert int(content.grid_columnconfigure(1)["minsize"]) == expected_floor
        assert expected_floor >= 320
        assert expected_floor <= RESULTS_REPORT_PREVIEW_WIDTH
        assert "consistent" in body.get("1.0", tk.END)
        assert body.tag_ranges("heading")
        assert source.cget("text") == "walk.mp4 · Interim derived values"
        assert "metrics not run" in preview_card.cget("text").lower()
        assert "interim derived" in body.get("1.0", tk.END).lower()
    finally:
        root.destroy()


def test_results_content_host_is_not_a_page_canvas() -> None:
    from pathlib import Path

    source = Path("stablewalk/ui/tk/dashboard_summary.py").read_text(encoding="utf-8")
    start = source.index("def _install_results_content_host")
    end = source.index("def build_results_summary_tab")
    host_fn = source[start:end]
    assert "tk.Canvas" not in host_fn
    assert "create_window" not in host_fn
    assert "no page-level canvas scroll" in host_fn


def test_results_preview_scrollbar_only_when_text_overflows() -> None:
    import tkinter as tk

    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_summary import _build_results_report_preview

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        host = tk.Frame(root, width=280, height=360)
        host.grid()
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        host.grid_propagate(False)
        gui = SimpleNamespace()
        _build_results_report_preview(gui, host)
        root.update_idletasks()
        gui._sync_results_preview_scroll()
        body = gui.lbl_results_preview_body
        bar = gui._results_preview_scrollbar
        assert body.winfo_manager() == "grid"
        assert str(bar.winfo_manager() or "") == ""

        body.configure(state=tk.NORMAL)
        body.delete("1.0", tk.END)
        body.insert("1.0", "Short preview.\n")
        body.configure(state=tk.DISABLED)
        gui._sync_results_preview_scroll()
        root.update_idletasks()
        assert str(bar.winfo_manager() or "") == ""
        assert gui._results_preview_needs_scroll is False

        body.configure(state=tk.NORMAL)
        body.delete("1.0", tk.END)
        body.insert("1.0", "\n".join(f"Line {i} of a long report." for i in range(80)))
        body.configure(state=tk.DISABLED)
        gui._sync_results_preview_scroll()
        root.update_idletasks()
        gui._sync_results_preview_scroll()
        assert gui._results_preview_needs_scroll is True
        assert str(bar.winfo_manager() or "") == "grid"
    finally:
        root.destroy()


def test_results_tab_keeps_chrome_visible_without_nested_scroll() -> None:
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        from stablewalk.ui.tk.app import StableWalkGUI
        from stablewalk.ui.tk.dashboard_notebook import (
            TAB_RESULTS_SUMMARY,
            select_dashboard_tab,
        )
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
        from stablewalk.ui.tk.dashboard_summary import sync_results_report_preview

        app = StableWalkGUI(root=root)
        select_dashboard_tab(app, TAB_RESULTS_SUMMARY)
        for width, height in ((1920, 1080), (1366, 768), (1280, 720)):
            root.geometry(f"{width}x{height}+40+40")
            root.update_idletasks()
            root.update()
            apply_responsive_layout(app, width=width, height=height)
            root.update_idletasks()
            root.update()
            assert getattr(app, "_results_scroll_canvas", None) is None
            assert bool(getattr(app, "_results_needs_scroll", False)) is False
            for attr in (
                "btn_export_summary_json",
                "btn_export_summary_report",
                "btn_results_open_advanced",
            ):
                btn = getattr(app, attr, None)
                assert btn is not None and btn.winfo_ismapped()
                assert int(btn.winfo_height()) >= 20
            primary = getattr(app, "_results_primary_host", None)
            assert primary is not None and primary.winfo_ismapped()
            assert int(primary.winfo_height()) >= 36
            section = getattr(app, "_results_processing_section", None)
            assert section is not None
            assert section.title_lbl.cget("text") == "Processing Details"
            # Empty / waiting sessions keep technical processing collapsed.
            assert bool(section.expanded) is False
            summary = getattr(app, "_results_pipeline_summary", None)
            assert summary is not None
            source = getattr(app, "lbl_summary_source", None)
            assert source is not None and source.winfo_ismapped()
            assert int(source.winfo_height()) >= 12

        summary = SimpleNamespace(
            source="walk.mp4",
            scientific_interpretation="Gait looks consistent. " * 40,
        )
        app._analysis_summary_cache = summary
        app._results_pipeline_session_state = "partial"
        app._results_metrics_stage_status = "unavailable"
        assert sync_results_report_preview(app, summary) is True
        root.update_idletasks()
        root.update()
        preview = getattr(app, "_results_preview_host", None)
        assert preview is not None and preview.winfo_manager() == "grid"
        assert getattr(app, "_results_scroll_canvas", None) is None
        bar = getattr(app, "_results_preview_scrollbar", None)
        if bar is not None and str(bar.winfo_manager() or "") == "grid":
            assert bool(getattr(app, "_results_preview_needs_scroll", False)) is True
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_results_processing_details_expand_policy() -> None:
    from stablewalk.ui.tk.dashboard_summary import (
        RESULTS_PROCESSING_DETAILS_TITLE,
        _primary_status_presentation,
        _short_gait_summary_text,
        results_preview_rail_floor,
        results_processing_details_should_expand,
    )

    assert RESULTS_PROCESSING_DETAILS_TITLE == "Processing Details"
    assert results_processing_details_should_expand("complete") is False
    assert results_processing_details_should_expand(
        "complete", attention_warnings=["Metrics: low coverage"]
    ) is False
    assert results_processing_details_should_expand(
        "complete", attention_warnings=["Pose Detection: tracking failed"]
    ) is True
    assert results_processing_details_should_expand(
        "complete", required_stage_failed=True
    ) is True
    assert results_processing_details_should_expand("incomplete") is True
    assert results_processing_details_should_expand("processing") is True
    assert results_processing_details_should_expand("waiting") is False
    assert results_processing_details_should_expand("partial") is True
    assert _primary_status_presentation("complete", "100% Complete") == (
        "Analysis Complete",
        "ok",
    )
    assert _primary_status_presentation("waiting", "Waiting for analysis")[0] == (
        "Waiting for analysis"
    )
    assert _short_gait_summary_text(
        "Cadence is steady. Symmetry is close to even. Extra unused sentence."
    ) == "Cadence is steady. Symmetry is close to even."
    assert results_preview_rail_floor(1000) >= 320


def test_results_processing_stretch_does_not_leave_empty_void() -> None:
    """Collapsed Processing Details must not keep a weight-1 empty row."""
    import inspect

    from stablewalk.ui.tk import dashboard_summary as mod

    source = inspect.getsource(mod._sync_results_processing_stretch)
    assert 'sticky="nsew" if expanded else "ew"' in source
    assert "weight=1 if expanded else 0" in source
    assert "if preview_visible:" in source
    assert "weight=1 if (expanded or preview_visible)" not in source
    reflow = inspect.getsource(mod.reflow_results_layout)
    assert "processing_open" in reflow
    assert "_sync_results_processing_stretch(gui)" in reflow


def test_results_warning_copy_separates_notes_from_action() -> None:
    from stablewalk.ui.tk.results_warning_copy import (
        classify_results_warning,
        format_primary_action_warning,
        format_results_warning_copy,
    )

    assert classify_results_warning("Marker coverage is incomplete") == "info"
    assert classify_results_warning("optional artifact omitted") == "info"
    assert classify_results_warning("OpenSim IK MOT not found") == "info"
    assert "optional" in format_results_warning_copy(
        "OpenSim IK MOT not found", kind="info"
    ).lower()
    assert classify_results_warning("Could not read the source file") == "critical"
    assert classify_results_warning("tracking failed") == "critical"
    assert classify_results_warning("", status="failed") == "critical"

    note = format_results_warning_copy(
        "Marker coverage is incomplete for inverse kinematics", kind="info"
    )
    assert "joints" in note.lower()
    assert "inverse kinematics" not in note.lower()

    action = format_results_warning_copy(
        "Could not read the source file", label="Video", kind="critical"
    )
    assert "could not be opened" in action.lower()
    assert format_primary_action_warning(action).startswith("Action needed")


def test_primary_warnings_hide_technical_stage_notes() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_results_pipeline import STATUS_COMPLETED, STATUS_FAILED
    from stablewalk.ui.tk.dashboard_summary import (
        _required_attention_warnings,
        _warning_needs_user_action,
    )

    assert _warning_needs_user_action("Metrics: low coverage") is False
    assert _warning_needs_user_action("tracking failed") is True
    assert _warning_needs_user_action("optional artifact omitted") is False
    assert _warning_needs_user_action("", status="failed") is True

    views = (
        SimpleNamespace(
            key="metrics",
            label="Metrics",
            status=STATUS_COMPLETED,
            warnings="Marker coverage is incomplete for inverse kinematics",
            detail="",
        ),
        SimpleNamespace(
            key="pose_estimation",
            label="Pose Detection",
            status=STATUS_FAILED,
            warnings="Marker coverage is incomplete for inverse kinematics",
            detail="",
        ),
        SimpleNamespace(
            key="video",
            label="Video",
            status=STATUS_COMPLETED,
            warnings="Could not read the source file",
            detail="",
        ),
    )
    shown = _required_attention_warnings(views)
    assert any("Pose Detection did not finish" in item for item in shown)
    assert any("could not be opened" in item.lower() for item in shown)
    assert not any("coverage" in item.lower() for item in shown)
    assert not any("Could not read the source file" in item for item in shown)
    assert not any(item.startswith("Metrics:") for item in shown)


def test_results_processing_details_collapses_on_clean_completion() -> None:
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        from stablewalk.ui.tk.app import StableWalkGUI
        from stablewalk.ui.tk.dashboard_notebook import (
            TAB_RESULTS_SUMMARY,
            select_dashboard_tab,
        )
        from stablewalk.ui.tk.dashboard_summary import (
            apply_results_processing_details_expansion,
        )

        app = StableWalkGUI(root=root)
        select_dashboard_tab(app, TAB_RESULTS_SUMMARY)
        root.update_idletasks()
        section = getattr(app, "_results_processing_section", None)
        assert section is not None
        assert bool(section.expanded) is False
        assert apply_results_processing_details_expansion(
            app, headline_kind="complete", attention_warnings=()
        ) is False
        assert bool(section.expanded) is False
        assert apply_results_processing_details_expansion(
            app, headline_kind="incomplete", attention_warnings=()
        ) is True
        assert bool(section.expanded) is True
        assert apply_results_processing_details_expansion(
            app,
            headline_kind="complete",
            attention_warnings=["Metrics: low coverage"],
        ) is False
        assert bool(section.expanded) is False
        assert apply_results_processing_details_expansion(
            app,
            headline_kind="complete",
            attention_warnings=(),
            required_stage_failed=True,
        ) is True
        assert bool(section.expanded) is True
        status = getattr(app, "lbl_results_primary_status", None)
        assert status is not None
        assert "Analysis" in status.cget("text") or "Waiting" in status.cget("text")
        assert app.btn_export_summary_report.winfo_ismapped()
        assert app.btn_export_summary_json.winfo_ismapped()
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
