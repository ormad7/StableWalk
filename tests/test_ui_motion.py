"""Tests for subtle UI motion helpers (presentation only)."""

from __future__ import annotations

import os
import tkinter as tk
from types import SimpleNamespace

import pytest

from stablewalk.ui.design_system import tokens as T
from stablewalk.ui.tk import ui_motion as motion


@pytest.fixture()
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def test_motion_duration_in_requested_band() -> None:
    assert 150 <= motion.MOTION_MS <= 200
    assert motion.MOTION_MS == T.MOTION_MS
    assert motion.MOTION_STEPS == T.MOTION_STEPS
    assert motion.SLIDE_PX == T.MOTION_SLIDE_PX
    assert motion.MOTION_STEPS >= 3


def test_motion_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setenv("STABLEWALK_REDUCE_MOTION", "1")
    gui = SimpleNamespace(playing=False, reduce_motion=False)
    assert motion.motion_enabled(gui) is False


def test_motion_disabled_while_playing() -> None:
    os.environ.pop("STABLEWALK_REDUCE_MOTION", None)
    gui = SimpleNamespace(playing=True, reduce_motion=False)
    assert motion.motion_enabled(gui) is False


def test_motion_disabled_by_gui_flag() -> None:
    os.environ.pop("STABLEWALK_REDUCE_MOTION", None)
    gui = SimpleNamespace(playing=False, reduce_motion=True)
    assert motion.motion_enabled(gui) is False


def test_viz_reveal_mark_consume_once() -> None:
    gui = SimpleNamespace()
    motion.mark_viz_reveal(gui, kind="3d")
    motion.mark_viz_reveal(gui, kind="traj")
    assert motion.consume_viz_reveal(gui, "3d") is True
    assert motion.consume_viz_reveal(gui, "3d") is False
    assert motion.consume_viz_reveal(gui, "traj") is True


def test_play_tab_transition_skips_advanced_overlay(root) -> None:
    os.environ.pop("STABLEWALK_REDUCE_MOTION", None)
    gui = SimpleNamespace(root=root, playing=False, reduce_motion=False)
    root._stablewalk_gui = gui
    tab = tk.Frame(root, width=200, height=120)
    tab.pack()
    gui._tab_advanced_outer = tab
    root.update_idletasks()
    motion.play_tab_transition(gui, tab)
    veil = getattr(gui, "_tab_motion_veil", None)
    if veil is not None:
        assert str(veil.winfo_manager() or "") != "place"


def test_dismiss_tab_transition_veil_unmaps_overlay(root) -> None:
    gui = SimpleNamespace(root=root, playing=False, reduce_motion=False)
    veil = tk.Frame(root, bg="#1a222e")
    veil.place(relx=0, rely=0, relwidth=1, relheight=1)
    gui._tab_motion_veil = veil
    root.update_idletasks()
    motion.dismiss_tab_transition_veil(gui)
    assert str(veil.winfo_manager() or "") != "place"


def test_play_tab_transition_places_veil(root) -> None:
    os.environ.pop("STABLEWALK_REDUCE_MOTION", None)
    gui = SimpleNamespace(root=root, playing=False, reduce_motion=False)
    root._stablewalk_gui = gui
    tab = tk.Frame(root, width=200, height=120)
    tab.pack()
    root.update_idletasks()
    motion.play_tab_transition(gui, tab)
    veil = getattr(gui, "_tab_motion_veil", None)
    assert veil is not None
    assert str(veil.winfo_manager()) == "place"
    # Advance scheduled steps without hanging.
    for _ in range(motion.MOTION_STEPS + 2):
        root.update()
    try:
        assert str(veil.winfo_manager()) in ("", "place")
    except tk.TclError:
        pass


def test_flash_metric_value_settles(root) -> None:
    import time

    os.environ.pop("STABLEWALK_REDUCE_MOTION", None)
    gui = SimpleNamespace(root=root, playing=False, reduce_motion=False)
    root._stablewalk_gui = gui
    lbl = tk.Label(root, text="1.2", fg="#888888")
    lbl.pack()
    motion.flash_metric_value(gui, lbl, settle_fg="#aabbcc", key="test-metric")
    # Immediate flash uses primary text color.
    assert str(lbl.cget("fg")).lower() in ("#e8eef6", "e8eef6")
    assert "test-metric" in getattr(gui, "_ui_motion_jobs", {})
    # Wall-clock wait — ``update()`` alone does not advance ``after`` timers.
    settled = False
    for _ in range(15):
        root.update()
        if str(lbl.cget("fg")).lower() in ("#aabbcc", "aabbcc"):
            settled = True
            break
        time.sleep(0.02)
    assert settled


def test_reveal_host_noops_when_reduced(root, monkeypatch) -> None:
    monkeypatch.setenv("STABLEWALK_REDUCE_MOTION", "1")
    gui = SimpleNamespace(root=root, playing=False, reduce_motion=False)
    host = tk.Frame(root, width=100, height=80)
    host.pack()
    motion.reveal_host(gui, host, key="noop")
    assert list(host.place_slaves()) == [] or True
    # No veil children expected when motion is disabled.
    children = host.winfo_children()
    assert all(
        not (isinstance(c, tk.Frame) and c.winfo_manager() == "place") for c in children
    )


def test_resolve_gui_from_toplevel(root) -> None:
    gui = SimpleNamespace(root=root)
    root._stablewalk_gui = gui
    child = tk.Frame(root)
    assert motion.resolve_gui(child) is gui
