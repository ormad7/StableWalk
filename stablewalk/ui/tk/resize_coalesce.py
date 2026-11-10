"""Coalesce window-resize work so drag-resize stays responsive.

Tk fires ``<Configure>`` on every pixel of a live drag. Expensive Matplotlib
fits and tab reflows must wait until the size settles. Native geometry
managers still stretch frames during the drag.
"""

from __future__ import annotations

from typing import Any

# Trailing quiet period before expensive layout / figure fits run.
RESIZE_SETTLE_MS = 140
# Ignore jitter smaller than this (Windows often oscillates 1–4 px).
RESIZE_IGNORE_PX = 6
# Chart host Configure: match window settle so fits do not run mid-drag.
FIGURE_RESIZE_SETTLE_MS = 140


def size_change_is_noise(
    prev_w: int,
    prev_h: int,
    width: int,
    height: int,
    *,
    threshold: int = RESIZE_IGNORE_PX,
) -> bool:
    """True when the delta is too small to justify layout or Agg realloc."""
    return abs(int(width) - int(prev_w)) < int(threshold) and abs(
        int(height) - int(prev_h)
    ) < int(threshold)


def resize_drag_active(gui: Any) -> bool:
    return bool(getattr(gui, "_ui_resize_active", False))
