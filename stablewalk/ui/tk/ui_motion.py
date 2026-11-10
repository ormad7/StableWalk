"""Subtle UI motion — short fade/slide transitions (presentation only).

Goals
-----
* 150–200 ms transitions that improve perceived responsiveness
* Few frames (typically 4–5) so the Tk main thread stays light
* No animation during playback or when reduced-motion is requested
* Coalesced per-key jobs so rapid updates never stack

Tk has no native opacity, so “fade” is approximated with brief background /
foreground color steps, and “slide” with a few-pixel place offset.
"""

from __future__ import annotations

import os
import tkinter as tk
from typing import Any, Callable

from stablewalk.ui.theme import ELEVATED, PANEL, SURFACE, TEXT

# Keep within the requested 150–200 ms band (tokens are the source of truth).
try:
    from stablewalk.ui.design_system import tokens as _T

    MOTION_MS = int(_T.MOTION_MS)
    MOTION_STEPS = int(_T.MOTION_STEPS)
    SLIDE_PX = int(_T.MOTION_SLIDE_PX)
except Exception:  # pragma: no cover — tokens always present in app
    MOTION_MS = 180
    MOTION_STEPS = 5
    SLIDE_PX = 6

# Soft veil shades for tab / panel reveal (dark clinical palette).
_VEIL_SHADES = (SURFACE, ELEVATED, PANEL, PANEL)


def resolve_gui(widget: tk.Misc | None = None, gui: Any | None = None) -> Any | None:
    """Resolve the StableWalk GUI from an explicit handle or widget toplevel."""
    if gui is not None:
        return gui
    if widget is None:
        return None
    try:
        top = widget.winfo_toplevel()
    except tk.TclError:
        return None
    return getattr(top, "_stablewalk_gui", None)


def motion_enabled(gui: Any | None = None) -> bool:
    """False when the user/env opts out or the UI is in a hot playback path."""
    env = os.environ.get("STABLEWALK_REDUCE_MOTION", "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return False
    if gui is not None:
        if getattr(gui, "reduce_motion", False):
            return False
        if getattr(gui, "playing", False):
            return False
        busy_fn = getattr(gui, "_analysis_workflow_busy", None)
        if callable(busy_fn):
            try:
                if busy_fn():
                    return False
            except Exception:
                pass
    return True


def _root_of(widget: tk.Misc | None, gui: Any | None = None) -> tk.Misc | None:
    if gui is not None:
        root = getattr(gui, "root", None)
        if root is not None:
            return root
    if widget is None:
        return None
    try:
        return widget.winfo_toplevel()
    except tk.TclError:
        return None


def _job_store(owner: Any | None) -> dict[str, str]:
    if owner is None:
        return {}
    store = getattr(owner, "_ui_motion_jobs", None)
    if store is None:
        store = {}
        try:
            owner._ui_motion_jobs = store
        except Exception:
            return {}
    return store


def cancel_motion(owner: Any | None, key: str) -> None:
    """Cancel a pending motion job by key (gui or Tk root)."""
    if owner is None:
        return
    store = _job_store(owner)
    aid = store.pop(key, None)
    root = getattr(owner, "root", None) or owner
    if aid is not None and root is not None:
        try:
            root.after_cancel(aid)
        except (tk.TclError, ValueError):
            pass


def cancel_all_motion(owner: Any | None) -> None:
    if owner is None:
        return
    for key in list(_job_store(owner)):
        cancel_motion(owner, key)


def dismiss_tab_transition_veil(gui: Any | None) -> None:
    """Remove a leftover tab-enter overlay so content is not left covered."""
    if gui is None:
        return
    veil = getattr(gui, "_tab_motion_veil", None)
    if veil is None:
        return
    try:
        if veil.winfo_exists():
            veil.place_forget()
    except tk.TclError:
        pass


def _schedule(
    owner: Any | None,
    key: str,
    root: tk.Misc,
    delay_ms: int,
    callback: Callable[[], None],
) -> None:
    store_owner = owner if owner is not None else root
    cancel_motion(store_owner, key)

    def _run() -> None:
        store = _job_store(store_owner)
        store.pop(key, None)
        try:
            callback()
        except tk.TclError:
            if key in ("tab_transition", "tab_transition_failsafe"):
                dismiss_tab_transition_veil(owner)
        except Exception:
            import logging

            logging.getLogger(__name__).exception("UI motion job %s failed", key)
            if key in ("tab_transition", "tab_transition_failsafe"):
                dismiss_tab_transition_veil(owner)

    try:
        aid = root.after(max(0, int(delay_ms)), _run)
    except tk.TclError:
        if key in ("tab_transition", "tab_transition_failsafe"):
            dismiss_tab_transition_veil(owner)
        return
    _job_store(store_owner)[key] = aid


def _step_delay_ms() -> int:
    return max(16, int(round(MOTION_MS / max(1, MOTION_STEPS))))


def play_tab_transition(gui: Any, tab: tk.Misc | None) -> None:
    """Soft veil fade + micro-slide when entering a dashboard tab (~180 ms)."""
    if tab is None or not motion_enabled(gui):
        dismiss_tab_transition_veil(gui)
        return
    # Advanced is a PANEL-colored Canvas scroll shell. A stuck veil is the same
    # color as an empty tab, so never cover that page with the overlay.
    if tab is getattr(gui, "_tab_advanced_outer", None):
        cancel_motion(gui, "tab_transition")
        cancel_motion(gui, "tab_transition_failsafe")
        dismiss_tab_transition_veil(gui)
        return
    root = _root_of(tab, gui)
    if root is None:
        return

    key = "tab_transition"
    cancel_motion(gui, key)
    cancel_motion(gui, "tab_transition_failsafe")

    try:
        veil = getattr(gui, "_tab_motion_veil", None)
        need_new = veil is None
        if not need_new:
            try:
                need_new = not veil.winfo_exists() or veil.master is not tab
            except tk.TclError:
                need_new = True
        if need_new:
            try:
                if veil is not None:
                    veil.destroy()
            except tk.TclError:
                pass
            veil = tk.Frame(tab, bg=_VEIL_SHADES[0], highlightthickness=0)
            gui._tab_motion_veil = veil

        veil.configure(bg=_VEIL_SHADES[0])
        veil.place(relx=0, rely=0, relwidth=1, relheight=1, y=SLIDE_PX)
        try:
            veil.lift()
        except tk.TclError:
            pass
    except tk.TclError:
        dismiss_tab_transition_veil(gui)
        return

    shades = _VEIL_SHADES
    delay = _step_delay_ms()
    offsets = [SLIDE_PX, SLIDE_PX // 2, SLIDE_PX // 3, 0]

    def _tick(step: int = 0) -> None:
        try:
            if not veil.winfo_exists():
                return
        except tk.TclError:
            return
        if step >= len(shades):
            dismiss_tab_transition_veil(gui)
            return
        try:
            veil.configure(bg=shades[step])
            y = offsets[min(step, len(offsets) - 1)]
            veil.place_configure(y=y)
        except tk.TclError:
            dismiss_tab_transition_veil(gui)
            return
        _schedule(gui, key, root, delay, lambda s=step + 1: _tick(s))

    _tick(0)
    # If a tick is cancelled mid-fade, do not leave the overlay covering the tab.
    _schedule(
        gui,
        "tab_transition_failsafe",
        root,
        MOTION_MS + 80,
        lambda: dismiss_tab_transition_veil(gui),
    )


def flash_metric_value(
    gui: Any | None,
    label: tk.Misc | None,
    *,
    settle_fg: str,
    key: str | None = None,
) -> None:
    """Brief foreground brighten when a metric value changes."""
    if label is None:
        return
    gui = resolve_gui(label, gui)
    if not motion_enabled(gui):
        return
    root = _root_of(label, gui)
    if root is None:
        return

    job_key = key or f"metric:{id(label)}"
    owner = gui if gui is not None else root
    cancel_motion(owner, job_key)
    delay = _step_delay_ms()

    try:
        label.configure(fg=TEXT)
    except tk.TclError:
        return

    def _settle() -> None:
        try:
            label.configure(fg=settle_fg)
        except tk.TclError:
            pass

    # Two-step settle keeps total under ~180 ms without busy loops.
    _schedule(owner, job_key, root, delay * 2, _settle)


def _dismiss_host_reveal_veil(host: tk.Misc | None, veil: tk.Misc | None = None) -> None:
    """Remove a panel reveal overlay so it cannot leave the chart covered."""
    if host is None:
        return
    current = veil
    if current is None:
        current = getattr(host, "_stablewalk_reveal_veil", None)
    stored = getattr(host, "_stablewalk_reveal_veil", None)
    if stored is not None and (current is None or stored is current):
        try:
            delattr(host, "_stablewalk_reveal_veil")
        except Exception:
            pass
    for widget in (current, stored):
        if widget is None:
            continue
        try:
            if widget.winfo_exists():
                widget.place_forget()
                widget.destroy()
        except tk.TclError:
            pass


def reveal_host(
    gui: Any | None,
    host: tk.Misc | None,
    *,
    key: str = "reveal",
) -> None:
    """Short fade+slide veil over a panel after load / first paint."""
    if host is None or not motion_enabled(gui):
        _dismiss_host_reveal_veil(host)
        return
    root = _root_of(host, gui)
    if root is None:
        return

    job_key = f"reveal:{key}"
    cancel_motion(gui, job_key)
    cancel_motion(gui, f"{job_key}:failsafe")
    _dismiss_host_reveal_veil(host)

    try:
        veil = tk.Frame(host, bg=_VEIL_SHADES[0], highlightthickness=0)
        host._stablewalk_reveal_veil = veil
        veil.place(relx=0, rely=0, relwidth=1, relheight=1, y=SLIDE_PX)
        try:
            veil.lift()
        except tk.TclError:
            pass
    except tk.TclError:
        return

    shades = _VEIL_SHADES
    delay = _step_delay_ms()
    offsets = [SLIDE_PX, max(2, SLIDE_PX // 2), 0, 0]

    def _tick(step: int = 0) -> None:
        try:
            if not veil.winfo_exists():
                return
        except tk.TclError:
            return
        if step >= len(shades):
            _dismiss_host_reveal_veil(host, veil)
            return
        try:
            veil.configure(bg=shades[step])
            veil.place_configure(y=offsets[min(step, len(offsets) - 1)])
        except tk.TclError:
            _dismiss_host_reveal_veil(host, veil)
            return
        _schedule(gui, job_key, root, delay, lambda s=step + 1: _tick(s))

    _tick(0)
    _schedule(
        gui,
        f"{job_key}:failsafe",
        root,
        MOTION_MS + 80,
        lambda: _dismiss_host_reveal_veil(host, veil),
    )


def reveal_canvas(
    gui: Any | None,
    canvas: Any | None,
    *,
    key: str = "canvas",
) -> None:
    """Reveal the Tk host of a Matplotlib FigureCanvasTkAgg after a draw."""
    if canvas is None or not motion_enabled(gui):
        return
    try:
        widget = canvas.get_tk_widget()
    except Exception:
        return
    parent = None
    try:
        parent = widget.master
    except tk.TclError:
        parent = widget
    reveal_host(gui, parent or widget, key=key)


def mark_viz_reveal(gui: Any, *, kind: str = "3d") -> None:
    """Request a one-shot reveal on the next force-drawn 3D / graph paint."""
    if gui is None:
        return
    pending = getattr(gui, "_ui_motion_viz_pending", None)
    if pending is None:
        pending = set()
        gui._ui_motion_viz_pending = pending
    pending.add(kind)


def consume_viz_reveal(gui: Any, kind: str) -> bool:
    """Return True once if a reveal was requested for ``kind``."""
    pending = getattr(gui, "_ui_motion_viz_pending", None)
    if not pending or kind not in pending:
        return False
    pending.discard(kind)
    return True


__all__ = [
    "MOTION_MS",
    "MOTION_STEPS",
    "SLIDE_PX",
    "cancel_all_motion",
    "cancel_motion",
    "consume_viz_reveal",
    "dismiss_tab_transition_veil",
    "flash_metric_value",
    "mark_viz_reveal",
    "motion_enabled",
    "play_tab_transition",
    "resolve_gui",
    "reveal_canvas",
    "reveal_host",
]
