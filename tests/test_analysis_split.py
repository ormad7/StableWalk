"""Lightweight analysis tab split panes."""

from __future__ import annotations

import tkinter as tk

from stablewalk.ui.tk.analysis_split import (
    MIN_H_CHART,
    MIN_H_VIDEO_ROW,
    MIN_W_CHART,
    MIN_W_PATH,
    MIN_W_SUMMARY,
    MIN_W_VIDEO,
    create_h_split,
    split_add,
    split_clear,
    split_has_children,
)


def test_panel_min_floors_are_usable_not_huge():
    from stablewalk.ui.tk.analysis_split import overview_pane_minsizes

    assert 200 <= MIN_W_VIDEO <= 280
    assert 200 <= MIN_W_PATH <= 320
    video, skeleton, path = overview_pane_minsizes(1.0)
    assert path >= MIN_W_PATH
    assert path > video
    v125, _s125, p125 = overview_pane_minsizes(1.25)
    v150, _s150, p150 = overview_pane_minsizes(1.50)
    assert p125 >= path
    assert p150 > p125
    assert v150 >= v125
    assert 280 <= MIN_W_CHART <= 380
    assert 180 <= MIN_W_SUMMARY <= 240
    assert 220 <= MIN_H_CHART <= 300
    assert 220 <= MIN_H_VIDEO_ROW <= 300



def test_h_split_adds_two_panes_with_minsize():
    root = tk.Tk()
    root.withdraw()
    host = tk.Frame(root, width=800, height=400)
    host.pack()
    paned = create_h_split(host)
    paned.pack(fill=tk.BOTH, expand=True)
    a = tk.Frame(paned, width=200, height=200)
    b = tk.Frame(paned, width=200, height=200)
    split_add(paned, a, minsize=120)
    split_add(paned, b, minsize=120)
    split_add(paned, a, minsize=120)  # idempotent
    assert split_has_children(paned, 2)
    split_clear(paned)
    assert split_has_children(paned, 0)
    root.destroy()
