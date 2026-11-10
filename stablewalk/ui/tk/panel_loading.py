"""Per-widget loading overlays for progressive analysis paint.

Shows a compact indeterminate spinner + caption inside a panel host without
blocking the Tk main thread or covering the whole Overview.
"""

from __future__ import annotations

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.theme import ELEVATED, FONT_UI_XS, MUTED, TEXT_SECONDARY

logger = logging.getLogger(__name__)

# Logical panel keys → preferred host attribute names (first match wins).
PANEL_HOST_ATTRS: dict[str, tuple[str, ...]] = {
    "metrics": (
        "_overview_kpi_metrics_host",
        "_overview_clinical_summary",
    ),
    "graphs": (
        "knee_chart_host",
        "dof_analysis_graph_canvas_host",
        "_section_kinematic",
    ),
    "biomech": (
        "biomech_chart_host",
        "_biomech_contact_host",
        "_section_biomechanics",
    ),
    "opensim": (
        "_sidebar_opensim_panel",
    ),
}

DEFAULT_MESSAGES: dict[str, str] = {
    "metrics": "Loading metrics…",
    "graphs": "Loading graphs…",
    "biomech": "Loading biomechanics…",
    "opensim": "Loading OpenSim…",
}


def _resolve_host(gui: Any, key: str, host: tk.Misc | None = None) -> tk.Misc | None:
    if host is not None:
        return host
    for attr in PANEL_HOST_ATTRS.get(key, ()):
        widget = getattr(gui, attr, None)
        if widget is not None:
            return widget
    return None


def set_panel_loading(
    gui: Any,
    key: str,
    loading: bool,
    *,
    message: str | None = None,
    host: tk.Misc | None = None,
) -> None:
    """Show or hide an in-widget loading spinner for ``key``."""
    store = getattr(gui, "_panel_loading", None)
    if store is None:
        store = {}
        gui._panel_loading = store

    if not loading:
        entry = store.pop(key, None)
        if entry is None:
            return
        overlay = entry.get("overlay")
        bar = entry.get("bar")
        host_widget = entry.get("host")
        if bar is not None:
            try:
                bar.stop()
            except tk.TclError:
                pass
        if overlay is not None:
            try:
                overlay.place_forget()
                overlay.destroy()
            except tk.TclError:
                pass
        # Soft reveal when progressive loading finishes (skipped while playing).
        try:
            from stablewalk.ui.tk.ui_motion import reveal_host

            reveal_host(gui, host_widget, key=f"panel:{key}")
        except Exception:
            pass
        return

    target = _resolve_host(gui, key, host)
    if target is None:
        store[key] = {"pending": True, "message": message or DEFAULT_MESSAGES.get(key)}
        return

    # Replace any existing overlay for this key (same store instance).
    existing = store.pop(key, None)
    if existing is not None:
        bar_old = existing.get("bar")
        overlay_old = existing.get("overlay")
        if bar_old is not None:
            try:
                bar_old.stop()
            except tk.TclError:
                pass
        if overlay_old is not None:
            try:
                overlay_old.place_forget()
                overlay_old.destroy()
            except tk.TclError:
                pass

    caption = message or DEFAULT_MESSAGES.get(key, "Loading…")
    try:
        overlay = tk.Frame(target, bg=ELEVATED, highlightthickness=0)
        inner = tk.Frame(overlay, bg=ELEVATED, highlightthickness=0)
        inner.pack(expand=True)
        bar = ttk.Progressbar(
            inner,
            mode="indeterminate",
            length=96,
            style="Accent.Horizontal.TProgressbar",
        )
        bar.pack(pady=(0, 4))
        lbl = tk.Label(
            inner,
            text=caption,
            bg=ELEVATED,
            fg=TEXT_SECONDARY if TEXT_SECONDARY else MUTED,
            font=FONT_UI_XS,
        )
        lbl.pack()
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        try:
            overlay.lift()
        except tk.TclError:
            pass
        bar.start(12)
        store[key] = {"overlay": overlay, "bar": bar, "host": target, "message": caption}
    except tk.TclError:
        logger.debug("Could not place loading spinner for %s", key, exc_info=True)


def set_panels_loading(
    gui: Any,
    keys: tuple[str, ...] | list[str],
    loading: bool,
) -> None:
    """Arm or clear several panel spinners at once."""
    for key in keys:
        set_panel_loading(gui, key, loading)


def clear_all_panel_loading(gui: Any) -> None:
    """Remove every progressive-loading spinner."""
    store = getattr(gui, "_panel_loading", None) or {}
    for key in list(store.keys()):
        set_panel_loading(gui, key, False)


__all__ = [
    "DEFAULT_MESSAGES",
    "PANEL_HOST_ATTRS",
    "clear_all_panel_loading",
    "set_panel_loading",
    "set_panels_loading",
]
