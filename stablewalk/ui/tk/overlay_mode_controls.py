"""Overlay Mode toolbar — opacity + layer toggles above the video panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from stablewalk.pose.video_overlay_compose import VideoOverlayLayers
from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
)

_LAYER_SPECS: tuple[tuple[str, str, str, bool, str], ...] = (
    ("✦", "Skeleton", "var_video_overlay_skeleton", True, "Reconstructed skeleton on video"),
    ("Aa", "Joint Labels", "var_video_overlay_labels", False, "Short joint name labels"),
    ("●", "COM", "var_video_overlay_com", True, "com"),
    ("↑", "Ground Reaction", "var_video_overlay_grf", True, "grf"),
    ("∿", "Joint Trajectory", "var_video_overlay_traj", True, "joint_movement_3d"),
)


def ensure_overlay_mode_vars(gui: Any) -> None:
    """Create BooleanVars / opacity DoubleVar if missing."""
    if not hasattr(gui, "var_video_overlay_opacity"):
        gui.var_video_overlay_opacity = tk.DoubleVar(value=85.0)
    for _icon, _label, attr, default, _tip in _LAYER_SPECS:
        if not hasattr(gui, attr):
            setattr(gui, attr, tk.BooleanVar(value=default))


def overlay_layers_from_gui(gui: Any) -> VideoOverlayLayers:
    ensure_overlay_mode_vars(gui)

    def _on(attr: str, default: bool = True) -> bool:
        var = getattr(gui, attr, None)
        if var is None:
            return default
        try:
            return bool(var.get())
        except tk.TclError:
            return default

    return VideoOverlayLayers(
        skeleton=_on("var_video_overlay_skeleton", True),
        joint_labels=_on("var_video_overlay_labels", False),
        com=_on("var_video_overlay_com", True),
        ground_reaction=_on("var_video_overlay_grf", True),
        joint_trajectory=_on("var_video_overlay_traj", True),
    )


def overlay_opacity_from_gui(gui: Any) -> float:
    """Return opacity in 0–1 range from the 0–100 slider."""
    ensure_overlay_mode_vars(gui)
    try:
        pct = float(gui.var_video_overlay_opacity.get())
    except (tk.TclError, TypeError, ValueError):
        pct = 85.0
    return max(0.0, min(100.0, pct)) / 100.0


def set_overlay_mode_controls_visible(gui: Any, visible: bool) -> None:
    bar = getattr(gui, "_video_overlay_controls", None)
    if bar is None:
        return
    try:
        if visible:
            bar.grid()
        else:
            bar.grid_remove()
    except tk.TclError:
        pass


def build_overlay_mode_controls(
    gui: Any,
    parent: tk.Misc,
    *,
    row: int,
    on_change: Callable[[], None] | None = None,
) -> tk.Frame:
    """
    Build the Overlay Mode control strip (opacity + layer toggles).

    Hidden by default; shown when View Mode is Overlay.
    """
    ensure_overlay_mode_vars(gui)

    def _notify() -> None:
        if on_change is not None:
            on_change()

    bar = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    bar.grid(row=row, column=0, sticky="ew", pady=(0, PAD_XS))
    bar.columnconfigure(2, weight=1)
    gui._video_overlay_controls = bar

    tk.Label(
        bar,
        text="Overlay",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=(PAD_SM, PAD_SM), pady=PAD_XS)

    opacity_row = tk.Frame(bar, bg=ELEVATED, highlightthickness=0)
    opacity_row.grid(row=0, column=1, sticky="w", padx=(0, PAD_SM), pady=PAD_XS)

    tk.Label(
        opacity_row,
        text="Opacity",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(0, 4))

    opacity_lbl = tk.Label(
        opacity_row,
        text="85%",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_XS,
        width=4,
        anchor="e",
    )
    opacity_lbl.pack(side=tk.RIGHT, padx=(4, 0))
    gui._video_overlay_opacity_lbl = opacity_lbl

    def _on_opacity(_event: object | None = None) -> None:
        try:
            pct = int(round(float(gui.var_video_overlay_opacity.get())))
        except (tk.TclError, TypeError, ValueError):
            pct = 85
        opacity_lbl.configure(text=f"{pct}%")
        _notify()

    scale = ttk.Scale(
        opacity_row,
        from_=0,
        to=100,
        orient=tk.HORIZONTAL,
        variable=gui.var_video_overlay_opacity,
        length=110,
        command=lambda _v: _on_opacity(),
    )
    scale.pack(side=tk.LEFT)
    create_tooltip(scale, "Skeleton overlay opacity (0–100%)")
    _on_opacity()

    layers = tk.Frame(bar, bg=ELEVATED, highlightthickness=0)
    layers.grid(row=0, column=2, sticky="ew", padx=(0, PAD_SM), pady=PAD_XS)

    for icon, label, attr, _default, tip in _LAYER_SPECS:
        var = getattr(gui, attr)
        btn = ttk.Checkbutton(
            layers,
            text=f"{icon} {label}",
            variable=var,
            command=_notify,
        )
        btn.pack(side=tk.LEFT, padx=(0, 10))
        from stablewalk.ui.metric_tooltips import get_metric_tooltip

        science = get_metric_tooltip(tip)
        create_tooltip(btn, science or tip, wraplength=340)

    bar.grid_remove()
    return bar


__all__ = [
    "build_overlay_mode_controls",
    "ensure_overlay_mode_vars",
    "overlay_layers_from_gui",
    "overlay_opacity_from_gui",
    "set_overlay_mode_controls_visible",
]
