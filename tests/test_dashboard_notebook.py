"""Tests for notebook-based dashboard shell."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    install_dashboard_notebook,
    run_tab_switch_stress_test,
    select_dashboard_tab,
)


def test_install_dashboard_notebook_creates_six_tabs():
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
    assert gui._tab_advanced_scroll_canvas is not None

    tabs = gui._dashboard_notebook.tabs()
    assert len(tabs) == 6
    root.destroy()


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
    assert app._comparison_mode is not None
    assert app.selection.active_item_id == "right_knee"

    select_dashboard_tab(app, TAB_OVERVIEW)
    select_dashboard_tab(app, TAB_MOTION)
    select_dashboard_tab(app, TAB_BIOMECHANICS)
    select_dashboard_tab(app, TAB_RESULTS_SUMMARY)
    select_dashboard_tab(app, TAB_COMPARE)
    select_dashboard_tab(app, TAB_ADVANCED)
    root.destroy()
