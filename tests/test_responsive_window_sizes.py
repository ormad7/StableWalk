"""Cross-tab resize floors for the four supported desktop sizes.

Layout only — no analysis or data-flow changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TARGET_SIZES = (
    (1920, 1080),
    (1600, 900),
    (1366, 768),
    (1280, 720),
)

# Logical classification targets — not every size is opened as a live Tk window.
WIDE_LOGICAL_SIZES = (
    (2560, 1440),
    (3840, 2160),
)


def _tk_root():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    return root


def test_supported_sizes_use_stretch_not_fixed_motion_height() -> None:
    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "dashboard_responsive.py"
    )
    text = src.read_text(encoding="utf-8")
    assert "bottom.configure(height=" not in text
    assert "apply_responsive_layout" in text
    assert "apply_tab_stretch_weights" in (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "dashboard_notebook.py"
    ).read_text(encoding="utf-8")
    assert "reflow_overview_layout" in text
    assert "reflow_biomech_workspace" in text
    assert "reflow_results_layout" in text
    assert "reflow_advanced_workspace" in text
    assert "_reflow_side_by_side_workspace" in text


def test_classify_layout_at_target_sizes() -> None:
    from stablewalk.ui.tk.dashboard_responsive import (
        HeightMode,
        WidthMode,
        classify_layout,
    )

    for width, height in TARGET_SIZES:
        wmode, hmode = classify_layout(width, height)
        assert wmode in (WidthMode.XLARGE, WidthMode.LARGE, WidthMode.MEDIUM, WidthMode.SMALL)
        assert hmode in (HeightMode.TALL, HeightMode.SHORT)
        if width >= 2200:
            assert wmode is WidthMode.XLARGE
        elif width >= 1550:
            assert wmode is WidthMode.LARGE
        elif width >= 1200:
            assert wmode is WidthMode.MEDIUM
        if height >= 760:
            assert hmode is HeightMode.TALL
        else:
            assert hmode is HeightMode.SHORT

    for width, height in WIDE_LOGICAL_SIZES:
        wmode, hmode = classify_layout(width, height)
        assert wmode is WidthMode.XLARGE
        assert hmode is HeightMode.TALL


def test_overview_viz_weights_stay_balanced_at_target_widths() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_VIDEO_ROW_SOFT_MIN_H,
        SEC1_TRAJ_PATH_WEIGHT,
        SEC1_TRAJ_SKELETON_WEIGHT,
        SEC1_TRAJ_VIDEO_WEIGHT,
        apply_overview_host_row_weights,
        apply_overview_viz_grid_weights,
    )

    root = _tk_root()
    try:
        for width, _height in TARGET_SIZES:
            host = tk.Frame(root, width=width, height=400)
            section = tk.Frame(host, width=width, height=280)
            apply_overview_host_row_weights(host)
            apply_overview_viz_grid_weights(section)
            assert int(host.grid_rowconfigure(1)["weight"]) == 1
            assert int(section.grid_columnconfigure(0)["weight"]) == SEC1_TRAJ_VIDEO_WEIGHT
            assert int(section.grid_columnconfigure(1)["weight"]) == SEC1_TRAJ_SKELETON_WEIGHT
            assert int(section.grid_columnconfigure(2)["weight"]) == SEC1_TRAJ_PATH_WEIGHT
            assert section.grid_columnconfigure(0)["uniform"] in ("", None)
            assert int(section.grid_rowconfigure(0)["minsize"]) >= OVERVIEW_VIDEO_ROW_SOFT_MIN_H
            assert int(section.grid_rowconfigure(1)["minsize"] or 0) == 0
            assert int(section.grid_rowconfigure(2)["minsize"] or 0) == 0
    finally:
        root.destroy()


def test_compare_reflow_floors_fit_720p() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    root = _tk_root()
    try:
        class _Workspace(tk.Frame):
            def winfo_width(self) -> int:  # type: ignore[override]
                return 1200

            def winfo_height(self) -> int:  # type: ignore[override]
                return 420

        class _Root:
            def winfo_width(self) -> int:
                return 1280

            def winfo_height(self) -> int:
                return 720

        workspace = _Workspace(root)
        ctrl = ComparisonModeController.__new__(ComparisonModeController)
        ctrl.root = _Root()
        ctrl._compare_workspace = workspace
        ctrl._compare_vertical_compact = None
        ctrl._compare_summary_host = None
        ctrl._compare_details_host = None
        ctrl._compare_top_panel_frames = ()
        ctrl._transport_host = None
        ctrl._compare_metrics_host = None
        ctrl._reflow_side_by_side_workspace()
        for col, weight in ((0, 2), (1, 3), (2, 2)):
            assert int(workspace.grid_columnconfigure(col)["weight"]) == weight
        assert int(workspace.grid_rowconfigure(0)["weight"]) == 1
        assert int(workspace.grid_rowconfigure(1)["weight"] or 0) == 0
    finally:
        root.destroy()


def test_each_tab_stretches_content_and_keeps_chrome_compact() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import (
        TAB_COMPARE,
        apply_tab_stretch_weights,
        select_dashboard_tab,
    )

    root = _tk_root()
    try:
        app = StableWalkGUI(root=root)
        root.update_idletasks()
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

        apply_responsive_layout(app, width=1920, height=1080)
        apply_tab_stretch_weights(app)

        overview = app._tab_overview
        assert int(overview.grid_rowconfigure(0)["weight"]) == 1
        host = app._overview_layout_host
        assert int(host.grid_rowconfigure(0)["weight"]) == 0
        assert int(host.grid_rowconfigure(1)["weight"]) == 1
        assert int(host.grid_rowconfigure(2)["weight"]) == 0
        section = app._section_visual
        assert int(section.grid_columnconfigure(0)["weight"]) == 12
        assert int(section.grid_columnconfigure(1)["weight"]) == 12
        assert int(section.grid_columnconfigure(2)["weight"]) == 10
        assert int(section.grid_rowconfigure(0)["weight"]) == 1
        video = app.video_frame
        assert int(video.grid_rowconfigure(0)["weight"]) == 0
        assert int(video.grid_rowconfigure(3)["weight"]) == 1

        motion = app._tab_motion
        assert int(motion.grid_rowconfigure(0)["weight"]) == 0
        assert int(motion.grid_rowconfigure(1)["weight"]) == 1
        workspace = app._motion_workspace_host
        assert int(workspace.grid_rowconfigure(0)["weight"]) == 0
        assert int(workspace.grid_rowconfigure(1)["weight"]) == 1
        assert int(workspace.grid_rowconfigure(2)["weight"]) == 0
        graphs = app._motion_graphs_panel
        from stablewalk.ui.tk.dashboard_motion_workspace import _MOTION_GRAPH_CANVAS_ROW

        assert int(graphs.grid_rowconfigure(0)["weight"]) == 0
        assert int(graphs.grid_rowconfigure(_MOTION_GRAPH_CANVAS_ROW)["weight"]) == 1

        biomech = app._tab_biomechanics
        assert int(biomech.grid_rowconfigure(0)["weight"]) == 0
        assert int(biomech.grid_rowconfigure(1)["weight"]) == 0
        assert int(biomech.grid_rowconfigure(2)["weight"]) == 1
        chart = app._biomech_chart_panel
        assert int(chart.grid_rowconfigure(0)["weight"]) == 0
        assert int(chart.grid_rowconfigure(2)["weight"]) == 1
        assert int(chart.grid_rowconfigure(3)["weight"]) == 0

        results = app._tab_results_summary
        assert int(results.grid_rowconfigure(0)["weight"]) == 0
        assert int(results.grid_rowconfigure(1)["weight"]) == 1
        assert int(results.grid_rowconfigure(2)["weight"]) == 0

        from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB

        if ENABLE_COMPARE_TAB:
            compare = app._tab_compare
            assert compare is not None
            assert int(compare.grid_rowconfigure(0)["weight"]) == 0
            assert int(compare.grid_rowconfigure(1)["weight"]) == 1
            root.geometry("1920x1080")
            select_dashboard_tab(app, TAB_COMPARE)
            root.update_idletasks()
            apply_tab_stretch_weights(app)
            # Research dataset is the default Compare page. Laboratory
            # ComparisonModeController is built only when requested so first
            # Compare open does not freeze.
            ensure = getattr(app, "_ensure_comparison_mode_loaded", None)
            if callable(ensure):
                ensure()
            root.update_idletasks()
            ctrl = app._comparison_mode
            assert ctrl is not None
            assert set(str(ctrl.root.grid_info().get("sticky") or "")) == set("nsew")
            assert int(ctrl.root.grid_rowconfigure(0)["weight"]) == 0
            assert int(ctrl.root.grid_rowconfigure(2)["weight"]) == 1
            assert int(ctrl.root.grid_rowconfigure(3)["weight"]) == 0
            workspace = ctrl._compare_workspace
            assert int(workspace.grid_columnconfigure(0)["weight"]) == 2
            assert int(workspace.grid_columnconfigure(1)["weight"]) == 3
            assert int(workspace.grid_columnconfigure(2)["weight"]) == 2
        else:
            assert app._tab_compare is None
            assert app._comparison_mode is None

        advanced = app._tab_advanced_outer
        assert int(advanced.grid_rowconfigure(0)["weight"]) == 1
        content = app._tab_advanced_content
        assert int(content.grid_rowconfigure(0)["weight"]) == 0
        assert int(content.grid_rowconfigure(1)["weight"]) == 0
        select_dashboard_tab(app, "Advanced")
        root.update_idletasks()
        assert app._dashboard_notebook.nametowidget(
            app._dashboard_notebook.select()
        ) is advanced
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    finally:
        root.destroy()


def test_tab_expand_rows_and_short_window_caps() -> None:
    import tkinter as tk
    from tkinter import ttk

    from stablewalk.ui.tk.dashboard_notebook import (
        cap_expanding_row_minsize,
        configure_tab_expand_row,
        install_dashboard_notebook,
    )

    root = _tk_root()
    try:
        parent = ttk.Frame(root)
        parent.pack(fill=tk.BOTH, expand=True)
        gui = type("G", (), {"root": root})()
        overview, motion, biomech, results, compare, _advanced = install_dashboard_notebook(
            gui, parent
        )
        assert int(overview.grid_rowconfigure(0)["weight"]) == 1
        assert int(motion.grid_rowconfigure(1)["weight"]) == 1
        assert int(biomech.grid_rowconfigure(2)["weight"]) == 1
        assert int(results.grid_rowconfigure(1)["weight"]) == 1
        if compare is not None:
            assert int(compare.grid_rowconfigure(1)["weight"]) == 1

        extra = tk.Frame(root)
        extra.rowconfigure(0, weight=1, minsize=720)
        applied = cap_expanding_row_minsize(
            extra, 0, preferred=720, available=720, chrome=120, floor=96
        )
        assert 96 <= applied <= 600
    finally:
        root.destroy()


def test_narrow_desktop_stacks_biomech_instead_of_overflow() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_STACK_BELOW_W,
        apply_biomech_workspace_geometry,
    )

    assert BIOMECH_STACK_BELOW_W >= 1100
    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "dashboard_responsive.py"
    ).read_text(encoding="utf-8")
    assert '("_transport_analysis", not compact)' not in src
    assert '("_transport_frame", True)' in src

    import tkinter as tk
    from types import SimpleNamespace

    root = _tk_root()
    try:
        body = tk.Frame(root, width=1032, height=308)
        body.grid_propagate(False)
        body.grid(row=0, column=0)
        sidebar = tk.Frame(body)
        chart = tk.Frame(body)
        clinical = tk.Frame(body)
        sidebar.grid(row=0, column=0, sticky="nsew")
        chart.grid(row=0, column=1, sticky="nsew")
        clinical.grid(row=0, column=2, sticky="nsew")
        gui = SimpleNamespace(
            _biomech_body=body,
            _biomech_sidebar=sidebar,
            _biomech_chart_panel=chart,
            _biomech_clinical_host=clinical,
            _biomech_chart_footer=None,
            _biomech_split=None,
        )
        apply_biomech_workspace_geometry(gui, width=1100, height=308)
        assert gui._biomech_stacked is False
        assert getattr(gui, "_biomech_rails_collapsed", False) is True
        assert int(sidebar.grid_info()["columnspan"]) == 1
        assert int(chart.grid_info()["column"]) == 1
    finally:
        root.destroy()


def test_biomech_and_advanced_breakpoints_cover_target_widths() -> None:
    from stablewalk.ui.tk.dashboard_advanced_workspace import ADVANCED_TWO_COL_MIN
    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_STACK_BELOW_W,
        biomech_balanced_column_widths,
    )
    from stablewalk.ui.theme import WIDTH_STACK_ANALYSIS

    assert ADVANCED_TWO_COL_MIN <= 1280
    assert WIDTH_STACK_ANALYSIS == 1280
    for width, _height in TARGET_SIZES:
        stack = width < BIOMECH_STACK_BELOW_W
        findings, chart, clinical = biomech_balanced_column_widths(width)
        if not stack:
            assert findings > 0 and chart > 0 and clinical > 0
            assert chart >= findings
        assert width >= ADVANCED_TWO_COL_MIN


def test_transport_stays_outside_notebook() -> None:
    src = (
        Path(__file__).resolve().parents[1] / "stablewalk" / "ui" / "tk" / "app.py"
    ).read_text(encoding="utf-8")
    assert 'transport.pack(side=tk.BOTTOM, fill=tk.X' in src
    assert "def _build_transport_bar" in src


def test_min_window_and_compare_section_host_stretch() -> None:
    from stablewalk.ui.tk.dashboard_responsive import MIN_WINDOW_HEIGHT, MIN_WINDOW_WIDTH

    assert MIN_WINDOW_WIDTH <= 1280
    assert MIN_WINDOW_HEIGHT <= 720

    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "dashboard_compare.py"
    ).read_text(encoding="utf-8")
    assert "host.grid_propagate(False)" not in src
    assert "height=height" not in src.split("def _section_min_host")[1].split("return host")[0]
