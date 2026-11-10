"""
Tkinter lifecycle guards — keep exactly one root and one mainloop.

Prevents ghost black windows from secondary ``Tk()`` instances, unparented
message boxes, pyplot figure activation, and draws on destroyed canvases.
"""

from __future__ import annotations

import logging
import tkinter as tk
from typing import Any

logger = logging.getLogger(__name__)


def count_tk_default_roots() -> int:
    """Return 1 if a default root exists, else 0 (Tk only keeps one default)."""
    return 1 if getattr(tk, "_default_root", None) is not None else 0


def assert_single_tk_root(expected: tk.Misc | None = None) -> bool:
    """Log and return False if a second / mismatched Tk root is detected."""
    default = getattr(tk, "_default_root", None)
    if default is None:
        logger.warning("Tk lifecycle: no default root (unexpected during GUI session)")
        return False
    if expected is not None:
        try:
            exp_root = expected.winfo_toplevel()
        except tk.TclError:
            exp_root = expected
        if default is not exp_root and str(default) != str(exp_root):
            logger.error(
                "Tk lifecycle: default root %r differs from app root %r — "
                "a second Tk() may have been created",
                default,
                exp_root,
            )
            return False
    return True


def safe_messagebox(
    kind: str,
    title: str,
    message: str,
    *,
    parent: tk.Misc | None = None,
) -> None:
    """Show a message box always parented to the live app — never a new Tk()."""
    from tkinter import messagebox

    root = parent
    if root is not None:
        try:
            root = root.winfo_toplevel()
        except tk.TclError:
            root = parent
    if root is None:
        root = getattr(tk, "_default_root", None)
    if root is None:
        logger.error("Tk lifecycle: refusing messagebox without a parent root (%s)", title)
        print(f"{title}: {message}", flush=True)
        return
    fn = {
        "info": messagebox.showinfo,
        "warning": messagebox.showwarning,
        "error": messagebox.showerror,
    }.get(kind, messagebox.showinfo)
    try:
        fn(title, message, parent=root)
    except tk.TclError:
        logger.debug("messagebox failed", exc_info=True)


def canvas_widget_alive(canvas: Any) -> bool:
    """True when a Matplotlib TkAgg canvas widget still exists."""
    if canvas is None:
        return False
    try:
        widget = canvas.get_tk_widget()
    except Exception:
        return False
    try:
        return bool(widget.winfo_exists())
    except tk.TclError:
        return False


def safe_canvas_draw_idle(canvas: Any) -> None:
    """``draw_idle`` only when the Tk widget still exists."""
    if not canvas_widget_alive(canvas):
        return
    try:
        canvas.draw_idle()
    except tk.TclError:
        pass
    except Exception:
        logger.debug("canvas draw_idle failed", exc_info=True)


def close_pyplot_figures_quietly(*, skip_ids: set[int] | None = None) -> int:
    """Close orphan pyplot figures without activating/showing their windows.

    Never call ``plt.figure(num)`` here — that can pop a blank TkAgg window.
    """
    skip_ids = skip_ids or set()
    closed = 0
    try:
        import matplotlib.pyplot as plt
        from matplotlib import _pylab_helpers

        for num in list(plt.get_fignums()):
            manager = _pylab_helpers.Gcf.get_fig_manager(num)
            if manager is None:
                continue
            fig = getattr(getattr(manager, "canvas", None), "figure", None)
            if fig is not None and id(fig) in skip_ids:
                continue
            try:
                plt.close(num)
                closed += 1
            except Exception:
                pass
    except Exception:
        logger.debug("quiet pyplot close failed", exc_info=True)
    return closed


def bind_modal_toplevel(dlg: tk.Toplevel, *, parent: tk.Misc | None = None) -> None:
    """``grab_set`` with guaranteed ``grab_release`` on destroy (no wait_window)."""

    def _release_and_destroy() -> None:
        try:
            dlg.grab_release()
        except tk.TclError:
            pass
        try:
            dlg.destroy()
        except tk.TclError:
            pass

    dlg.protocol("WM_DELETE_WINDOW", _release_and_destroy)
    dlg.bind("<Escape>", lambda _e: _release_and_destroy())
    if parent is not None:
        try:
            dlg.transient(parent.winfo_toplevel())
        except tk.TclError:
            pass
    try:
        dlg.grab_set()
    except tk.TclError:
        logger.debug("grab_set failed", exc_info=True)
    try:
        dlg.focus_force()
    except tk.TclError:
        pass


__all__ = [
    "assert_single_tk_root",
    "bind_modal_toplevel",
    "canvas_widget_alive",
    "close_pyplot_figures_quietly",
    "count_tk_default_roots",
    "safe_canvas_draw_idle",
    "safe_messagebox",
]
