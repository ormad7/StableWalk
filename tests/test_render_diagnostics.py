"""Tests for playback render diagnostics and clip viewports."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.tk.clip_viewport import clip_canvas_item_count, install_clipped_viewport
from stablewalk.ui.tk.render_diagnostics import (
    capture_playback_render_snapshot,
    record_playback_render_frame,
    reset_playback_render_counters,
    run_playback_render_stress_test,
)


def test_clip_viewport_single_window_item():
    root = tk.Tk()
    root.withdraw()
    parent = ttk.Frame(root)
    parent.pack(fill=tk.BOTH, expand=True)
    canvas, inner, window_id = install_clipped_viewport(parent, bg="#101010")
    label = tk.Label(inner, text="clip test")
    label.pack()
    root.update_idletasks()
    assert window_id > 0
    assert clip_canvas_item_count(canvas) == 1
    root.destroy()


def test_playback_render_stress_empty_gui():
    root = tk.Tk()
    root.withdraw()
    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI(root=root)
    root.update_idletasks()
    reset_playback_render_counters(app)
    results = run_playback_render_stress_test(app, frames=120, scroll_during_playback=True)
    assert all(passed for _name, passed, _detail in results)
    snap = capture_playback_render_snapshot(app, 120)
    assert snap.video_label_count == 1
    assert snap.knee_canvas_count == 1
    assert snap.joint_path_canvas_count == 1
    assert snap.skel_canvas_count == 1
    root.destroy()


def test_record_playback_render_frame_increments():
    root = tk.Tk()
    root.withdraw()
    gui = type("G", (), {"root": root, "_render_debug": False})()
    reset_playback_render_counters(gui)
    assert record_playback_render_frame(gui) == 1
    assert record_playback_render_frame(gui) == 2
    root.destroy()
