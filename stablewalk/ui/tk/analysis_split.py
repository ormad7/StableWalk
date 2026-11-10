"""Lightweight horizontal split panes for analysis tabs.

Uses Tk ``PanedWindow`` (not ttk) so each pane can keep a minsize.
No preference I/O — sash position is session-only.
"""

from __future__ import annotations

from typing import Any, Callable

import tkinter as tk
from stablewalk.ui.theme import PANEL

_SASH_W = 6

# Soft floors (px). Large enough that video / charts / summaries stay usable;
# small enough that a laptop window is not forced into a huge fixed layout.
MIN_W_VIDEO = 240
MIN_W_SKELETON = 260
# 3D path needs more than video: axis labels + coordinate inspector stay usable.
MIN_W_PATH = 300
MIN_W_CHART = 340
MIN_W_SUMMARY = 200
MIN_H_CHART = 280
MIN_H_VIDEO_ROW = 280
# Overview vertical split: viewers stay usable; gait info may scroll.
MIN_H_PRIMARY_VIZ = MIN_H_VIDEO_ROW
MIN_H_SECONDARY_INFO = 100


def create_h_split(parent: tk.Misc, *, bg: str = PANEL) -> tk.PanedWindow:
    """Horizontal sash splitter filling its parent via the caller's geometry."""
    return tk.PanedWindow(
        parent,
        orient=tk.HORIZONTAL,
        sashwidth=_SASH_W,
        sashpad=1,
        sashrelief=tk.FLAT,
        showhandle=False,
        opaqueresize=True,
        bd=0,
        bg=bg,
    )


def create_v_split(parent: tk.Misc, *, bg: str = PANEL) -> tk.PanedWindow:
    """Vertical sash splitter. Extra height should go to the first (primary) pane."""
    return tk.PanedWindow(
        parent,
        orient=tk.VERTICAL,
        sashwidth=_SASH_W,
        sashpad=1,
        sashrelief=tk.FLAT,
        showhandle=False,
        opaqueresize=True,
        bd=0,
        bg=bg,
    )


def split_add(
    paned: tk.PanedWindow,
    child: tk.Misc,
    *,
    minsize: int = 140,
    stretch: str = "always",
    padx: int | tuple[int, int] = 0,
) -> None:
    """Add ``child`` as a resizable pane (safe if already present)."""
    try:
        if str(child) in paned.panes():
            return
    except tk.TclError:
        pass
    paned.add(
        child,
        minsize=max(80, int(minsize)),
        stretch=stretch,
        padx=int(padx[-1] if isinstance(padx, tuple) else padx),
        pady=0,
    )


def split_clear(paned: tk.PanedWindow | None) -> None:
    """Remove all panes without destroying the child widgets."""
    if paned is None:
        return
    try:
        for pane in list(paned.panes()):
            paned.forget(pane)
    except tk.TclError:
        pass


def split_has_children(paned: tk.PanedWindow | None, count: int) -> bool:
    if paned is None:
        return False
    try:
        return len(paned.panes()) == int(count)
    except tk.TclError:
        return False


def overview_pane_minsizes(scale: float = 1.0) -> tuple[int, int, int]:
    """Video / Skeleton / 3D Path floors at the current display scale."""
    from stablewalk.ui.tk.layout_scale import scaled_px

    return (
        scaled_px(MIN_W_VIDEO, scale, minimum=200),
        scaled_px(MIN_W_SKELETON, scale, minimum=220),
        scaled_px(MIN_W_PATH, scale, minimum=260),
    )


def bind_persistent_h_sash(
    paned: tk.PanedWindow,
    holder: Any,
    *,
    frac_attr: str,
    drag_attr: str,
    default: float,
    min_left: int,
    min_right: int,
    min_width: int = 480,
) -> Callable[[object | None], None]:
    """Keep a user-dragged plot|table sash as a fraction across window resizes."""
    if getattr(holder, frac_attr, None) is None:
        setattr(holder, frac_attr, float(default))
    setattr(holder, drag_attr, False)

    def _place(_event: object | None = None) -> None:
        if bool(getattr(holder, drag_attr, False)):
            return
        try:
            width = int(paned.winfo_width() or 0)
            if width < int(min_width):
                return
            frac = float(getattr(holder, frac_attr, default) or default)
            left = max(int(min_left), min(int(round(width * frac)), width - int(min_right)))
            paned.sash_place(0, left, 0)
        except (tk.TclError, TypeError, ValueError, ZeroDivisionError, AttributeError):
            pass

    def _press(_event: object | None = None) -> None:
        setattr(holder, drag_attr, True)

    def _release(_event: object | None = None) -> None:
        try:
            width = max(1, int(paned.winfo_width() or 0))
            try:
                pos = int(paned.sash_coord(0)[0])
            except (tk.TclError, TypeError, ValueError, IndexError, AttributeError):
                pos = int(paned.sashpos(0))
            lo = float(min_left) / float(width)
            hi = 1.0 - (float(min_right) / float(width))
            setattr(holder, frac_attr, max(lo, min(hi, pos / width)))
        except (tk.TclError, TypeError, ValueError, ZeroDivisionError, IndexError):
            pass
        setattr(holder, drag_attr, False)

    paned.bind("<ButtonPress-1>", _press, add="+")
    paned.bind("<ButtonRelease-1>", _release, add="+")
    paned.bind("<Configure>", _place, add="+")
    return _place


def sync_tree_xscroll(
    tree: tk.Misc,
    xscroll: tk.Misc | None,
    host: tk.Misc | None,
    *,
    gutter: int = 22,
) -> None:
    """Hide the horizontal scrollbar when every column already fits."""
    if xscroll is None or host is None:
        return
    try:
        avail = int(host.winfo_width() or 0)
        if avail < 40:
            return
        needed = 0
        for col in tree.cget("columns") or ():
            needed += int(tree.column(col, "minwidth") or 0)
        if needed <= max(40, avail - int(gutter)):
            xscroll.grid_remove()
        else:
            xscroll.grid()
    except (tk.TclError, TypeError, ValueError):
        pass


def split_mapped(paned: tk.Misc | None) -> bool:
    if paned is None:
        return False
    try:
        return bool(paned.winfo_manager())
    except tk.TclError:
        return False
