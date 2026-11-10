"""Tests for notebook-based dashboard shell."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.tk.dashboard_notebook import (
    ENABLE_COMPARE_TAB,
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    install_dashboard_notebook,
    notebook_tab_texts,
    run_tab_switch_stress_test,
    select_dashboard_tab,
    sync_notebook_tab_fill,
    visible_dashboard_tab_labels,
)


def test_install_dashboard_notebook_creates_visible_tabs():
    root = tk.Tk()
    root.withdraw()
    gui = type("G", (), {"root": root})()
    parent = ttk.Frame(root)
    parent.pack(fill=tk.BOTH, expand=True)

    overview, motion, biomechanics, results_summary, compare, advanced = (
        install_dashboard_notebook(gui, parent)
    )

    assert overview is gui._tab_overview
    assert motion is gui._tab_motion
    assert biomechanics is gui._tab_biomechanics
    assert results_summary is gui._tab_results_summary
    assert compare is gui._tab_compare
    assert advanced is gui._tab_advanced_content
    assert gui._dashboard_notebook is not None
    assert gui._dash_scroll_outer is None
    assert gui._tab_advanced_scroll_canvas is None
    assert gui._tab_advanced_content is not None
    assert gui._tab_advanced_scroll_viewport is not None

    texts = notebook_tab_texts(gui)
    expected = visible_dashboard_tab_labels()
    assert texts == expected
    assert TAB_OVERVIEW in texts
    assert TAB_MOTION in texts
    assert TAB_BIOMECHANICS in texts
    assert TAB_RESULTS_SUMMARY in texts
    assert TAB_ADVANCED in texts
    if ENABLE_COMPARE_TAB:
        assert compare is not None
        assert TAB_COMPARE in texts
        assert int(compare.grid_rowconfigure(1)["weight"]) == 1
    else:
        assert compare is None
        assert TAB_COMPARE not in texts
    assert texts[-1] == TAB_ADVANCED
    assert int(overview.grid_rowconfigure(0)["weight"]) == 1
    assert int(overview.grid_rowconfigure(2)["weight"]) == 0
    assert int(motion.grid_rowconfigure(1)["weight"]) == 1
    assert int(biomechanics.grid_rowconfigure(2)["weight"]) == 1
    assert int(biomechanics.grid_rowconfigure(1)["weight"]) == 0
    assert int(results_summary.grid_rowconfigure(1)["weight"]) == 1
    assert int(gui._tab_advanced_outer.grid_rowconfigure(0)["weight"]) == 1

    extra = ttk.Frame(root)
    extra.rowconfigure(1, weight=1, minsize=400)
    extra.rowconfigure(2, weight=1, minsize=400)
    from stablewalk.ui.tk.dashboard_notebook import (
        cap_expanding_row_minsize,
        configure_tab_expand_row,
    )

    configure_tab_expand_row(extra, 2)
    assert int(extra.grid_rowconfigure(2)["weight"]) == 1
    assert int(extra.grid_rowconfigure(1)["weight"]) == 0
    assert int(extra.grid_rowconfigure(1)["minsize"]) == 0
    applied = cap_expanding_row_minsize(
        extra, 0, preferred=720, available=600, chrome=180, floor=200
    )
    assert 200 <= applied <= 420
    root.destroy()


def test_selected_notebook_tab_fills_client_area():
    import pytest

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        gui = type("G", (), {"root": root})()
        parent = ttk.Frame(root)
        parent.pack(fill=tk.BOTH, expand=True)
        install_dashboard_notebook(gui, parent)
        root.geometry("900x700")
        root.deiconify()
        root.update_idletasks()
        root.update()
        sync_notebook_tab_fill(gui)
        root.update_idletasks()
        notebook = gui._dashboard_notebook
        tab = gui._tab_overview
        nb_h = int(notebook.winfo_height() or 0)
        tab_h = int(tab.winfo_height() or 0)
        if nb_h < 200:
            pytest.skip("Notebook did not receive a mapped height")
        assert tab_h >= nb_h - 80
        assert tab_h > 400
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_tab_switch_stress_singletons():
    root = tk.Tk()
    root.withdraw()
    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI(root=root)
    root.update_idletasks()
    assert app._comparison_mode is None
    assert app.selection.activate_item("right_knee")

    results = run_tab_switch_stress_test(app, cycles=50)
    assert all(passed for _name, passed, _detail in results)
    if ENABLE_COMPARE_TAB:
        assert app._comparison_mode is not None
    else:
        assert app._comparison_mode is None
    assert app.selection.active_item_id == "right_knee"

    select_dashboard_tab(app, TAB_OVERVIEW)
    select_dashboard_tab(app, TAB_MOTION)
    select_dashboard_tab(app, TAB_BIOMECHANICS)
    select_dashboard_tab(app, TAB_RESULTS_SUMMARY)
    select_dashboard_tab(app, TAB_COMPARE)
    select_dashboard_tab(app, TAB_ADVANCED)
    assert notebook_tab_texts(app)[-1] == TAB_ADVANCED
    assert app._dashboard_notebook.nametowidget(app._dashboard_notebook.select()) is app._tab_advanced_outer
    root.destroy()


def test_biomechanics_reflow_fits_active_contact_family() -> None:
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from stablewalk.ui.tk.dashboard_notebook import reflow_tab_canvases

    biomech_tab = object()
    notebook = SimpleNamespace(
        select=lambda: "biomech",
        nametowidget=lambda _name: biomech_tab,
    )
    gui = SimpleNamespace(
        _dashboard_notebook=notebook,
        _tab_overview=object(),
        _tab_motion=object(),
        _tab_biomechanics=biomech_tab,
        _biomech_chart_family="contact",
        _fit_contact_gait_canvas=MagicMock(),
        _fit_biomech_canvas=MagicMock(),
        skeleton_player=None,
        selection=None,
    )

    reflow_tab_canvases(gui)

    gui._fit_contact_gait_canvas.assert_called_once_with()
    gui._fit_biomech_canvas.assert_not_called()
