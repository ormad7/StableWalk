"""Tests for notebook-based dashboard shell."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_MOTION,
    TAB_OVERVIEW,
    install_dashboard_notebook,
    run_tab_switch_stress_test,
    select_dashboard_tab,
)


def test_install_dashboard_notebook_creates_three_tabs():
    root = tk.Tk()
    root.withdraw()
    gui = type("G", (), {"root": root})()
    parent = ttk.Frame(root)
    parent.pack(fill=tk.BOTH, expand=True)

    overview, motion, advanced = install_dashboard_notebook(gui, parent)

    assert overview is gui._tab_overview
    assert motion is gui._tab_motion
    assert advanced is gui._tab_advanced_content
    assert gui._dashboard_notebook is not None
    assert gui._dash_scroll_outer is None
    assert gui._tab_advanced_scroll_canvas is not None

    tabs = gui._dashboard_notebook.tabs()
    assert len(tabs) == 3
    root.destroy()


def test_tab_switch_stress_singletons():
    root = tk.Tk()
    root.withdraw()
    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI(root=root)
    root.update_idletasks()

    results = run_tab_switch_stress_test(app, cycles=50)
    assert all(passed for _name, passed, _detail in results)

    select_dashboard_tab(app, TAB_OVERVIEW)
    select_dashboard_tab(app, TAB_MOTION)
    select_dashboard_tab(app, TAB_ADVANCED)
    root.destroy()
