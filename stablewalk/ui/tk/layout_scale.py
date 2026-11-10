"""DPI-aware layout tokens for the Tk dashboard.

Windows display scaling (100–175%) changes how many pixels a label needs.
Hardcoded wraplengths and minsizes designed at 96 DPI clip at 150%+.
This module scales those layout floors. It does not recompute analysis.
"""

from __future__ import annotations

import sys
import tkinter as tk
from typing import Any

# Practical window floors (logical pixels at 100% / 96 DPI).
MIN_WINDOW_WIDTH = 960
MIN_WINDOW_HEIGHT = 600

# Extra-wide desktop (1440p / 4K usable width after chrome).
WIDTH_XLARGE = 2200

# Validation targets (not a claim that every control fits at once).
RESOLUTION_TARGETS: tuple[tuple[int, int], ...] = (
    (1366, 768),
    (1600, 900),
    (1920, 1080),
    (2560, 1440),
    (3840, 2160),
)
SCALE_TARGETS: tuple[float, ...] = (1.0, 1.25, 1.50, 1.75)


def dpi_scale_factor(root: tk.Misc | None) -> float:
    """Windows display scale: 1.0 = 100%, 1.25 = 125%, 1.5 = 150%, 1.75 = 175%."""
    if root is not None:
        try:
            ppi = float(root.winfo_fpixels("1i"))
            if ppi > 1.0:
                return max(0.75, min(3.0, ppi / 96.0))
        except Exception:
            # Dummy roots in unit tests are not Tk widgets; do not fall through
            # to GetDpiForSystem or stacking thresholds drift at 125%+.
            return 1.0
    try:
        if sys.platform == "win32":
            import ctypes

            dpi = int(ctypes.windll.user32.GetDpiForSystem())
            return max(0.75, min(3.0, float(dpi) / 96.0))
    except Exception:
        pass
    return 1.0


def combobox_char_width(
    values: tuple[str, ...] | list[str] | str,
    *,
    minimum: int = 12,
    maximum: int = 48,
) -> int:
    """Character width so readonly comboboxes do not truncate their longest label."""
    lo = int(minimum)
    hi = int(maximum)
    if hi < lo:
        lo = hi
    if isinstance(values, str):
        items: tuple[str, ...] | list[str] = (values,)
    else:
        items = values
    longest = 0
    for value in items:
        longest = max(longest, len(str(value)))
    return max(lo, min(hi, longest + 1))


def scaled_px(value: int | float, scale: float, *, minimum: int = 1) -> int:
    """Scale a 96-DPI pixel constant for the current display factor."""
    px = int(round(float(value) * max(0.75, float(scale))))
    return max(int(minimum), px)


def scaled_breakpoints(scale: float) -> dict[str, int]:
    """Overview / viz floors that grow with font and DPI."""
    s = max(0.75, float(scale))
    return {
        # Cap so 1366×768 and 1920×1080 keep three draggable Overview panes
        # even at 125–150% DPI. Minsizes still protect the 3D path.
        "viz_three_col": min(scaled_px(1100, s, minimum=900), 1280),
        "viz_two_plus_one": scaled_px(900, s, minimum=720),
        "kpi_four": scaled_px(720, s, minimum=560),
        "kpi_two": scaled_px(400, s, minimum=280),
        "summary_side": scaled_px(1280, s, minimum=960),
        "biomech_stack": scaled_px(1180, s, minimum=900),
        # Cap so 1366-class laptops at 125% DPI stay two-column. DPI scaling
        # of 1100px was stacking Spectral Features / Numerical results off-screen.
        "motion_stack": min(scaled_px(1100, s, minimum=860), 1280),
        "compare_stack": scaled_px(1180, s, minimum=900),
        "advanced_two_col": scaled_px(980, s, minimum=860),
        "xlarge": scaled_px(WIDTH_XLARGE, s, minimum=1800),
    }


def apply_process_dpi_awareness() -> None:
    """Per-monitor DPI awareness so Windows does not bitmap-scale the window."""
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def apply_tk_dpi_scaling(root: tk.Misc) -> float:
    """Raise Tk's point-to-pixel factor when Tk still thinks it is 96 DPI.

    If Tk already reports a matching scale, this is a no-op (avoids double scale).
    """
    scale = dpi_scale_factor(root)
    try:
        current = float(root.tk.call("tk", "scaling"))
    except Exception:
        return scale
    # Default Tk scaling is 96/72 ≈ 1.333 at 100%.
    expected = 96.0 / 72.0 * scale
    if scale >= 1.20 and current < expected * 0.85:
        try:
            root.tk.call("tk", "scaling", expected)
        except Exception:
            pass
    return scale


def bind_status_wrap(
    host: tk.Misc,
    label: tk.Misc,
    *,
    reserve: int = 360,
    minimum: int = 80,
) -> None:
    """Keep a toolbar status label wrapping inside leftover width."""

    def _sync(_event: object | None = None) -> None:
        try:
            width = int(host.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 40:
            return
        wrap = max(int(minimum), int(width) - int(reserve))
        try:
            label.configure(wraplength=wrap)
        except tk.TclError:
            pass

    try:
        host.bind("<Configure>", _sync, add="+")
        _sync()
    except tk.TclError:
        return


def layout_scale_from_gui(gui: Any) -> float:
    """Return the cached Tk DPI scale, or 1.0 when no live root is present.

    Test doubles without a Tk widget must not pick up the host Windows scale
    or stacking thresholds drift and layout unit tests fail at 125%+.
    """
    cached = getattr(gui, "_layout_dpi_scale", None)
    if isinstance(cached, (int, float)) and cached > 0:
        return float(cached)
    root = getattr(gui, "root", None)
    if root is None:
        return 1.0
    scale = dpi_scale_factor(root)
    try:
        gui._layout_dpi_scale = scale
    except Exception:
        pass
    return scale


__all__ = [
    "MIN_WINDOW_HEIGHT",
    "MIN_WINDOW_WIDTH",
    "RESOLUTION_TARGETS",
    "SCALE_TARGETS",
    "WIDTH_XLARGE",
    "apply_process_dpi_awareness",
    "apply_tk_dpi_scaling",
    "bind_status_wrap",
    "combobox_char_width",
    "dpi_scale_factor",
    "layout_scale_from_gui",
    "scaled_breakpoints",
    "scaled_px",
]
