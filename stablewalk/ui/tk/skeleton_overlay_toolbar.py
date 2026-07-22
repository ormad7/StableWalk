"""3D Reconstruction overlay controls.

A single compact ribbon that sits *above* the skeleton viewport and never
overlaps it. Overlays are organised into three icon-tagged groups — Motion,
Biomechanics, Environment — so the bar reads like professional biomechanics
software chrome while leaving the maximum possible area for the visualization.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.scientific_labels import OVERLAY_BOS, OVERLAY_COM, OVERLAY_COM_VEL
from stablewalk.ui.theme import (
    ACCENT,
    BORDER,
    ELEVATED,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    create_tooltip,
)

# Group key -> (title, icon). Order defines left-to-right layout in the bar.
OVERLAY_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("motion", "Motion", "\u21dd"),              # ⇝ travel / movement
    ("biomechanics", "Biomechanics", "\u2696"),  # ⚖ balance / stability
    ("environment", "Environment", "\u25a4"),    # ▤ ground / scene
)

# icon, short label, BooleanVar attr, default_on, tooltip, group
OVERLAY_TOOL_SPECS: tuple[tuple[str, str, str, bool, str, str], ...] = (
    ("\u21e2", "Direction", "var_overlay_direction", True,
     "Gait travel direction", "motion"),
    ("\u25bc", "Contact", "var_overlay_contact", True,
     "Foot contact markers", "motion"),
    ("\u25cf", OVERLAY_COM, "var_overlay_com", True,
     "com", "biomechanics"),
    # BoS polygon defaults OFF — when ON it painted a large yellow card over the
    # feet and made the skeleton look clipped/small in the Overview panel.
    ("\u25c7", OVERLAY_BOS, "var_overlay_bos", False,
     "bos", "biomechanics"),
    ("\u2197", OVERLAY_COM_VEL, "var_overlay_com_velocity", False,
     "com", "biomechanics"),
    ("\u25ac", "Ground", "var_overlay_ground", True,
     "Ground / floor reference", "environment"),
)


def ensure_overlay_vars(gui: Any) -> None:
    """Create the overlay ``BooleanVar``s on *gui* if they don't exist yet."""
    for _icon, _label, attr, default_on, _tip, _group in OVERLAY_TOOL_SPECS:
        if not hasattr(gui, attr) or getattr(gui, attr) is None:
            setattr(gui, attr, tk.BooleanVar(value=default_on))


def _vsep(parent: tk.Misc) -> tk.Frame:
    """Thin vertical divider between groups."""
    return tk.Frame(parent, bg=BORDER, width=1, highlightthickness=0)


def _build_group_inline(
    gui: Any,
    parent: tk.Misc,
    group_key: str,
    title: str,
    icon: str,
) -> tk.Frame:
    """One overlay group as a single compact inline row: icon · title · toggles."""
    group = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)

    tk.Label(
        group,
        text=icon,
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(0, 2))
    tk.Label(
        group,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(0, PAD_XS))

    for icon_i, label, attr, _default, tip, spec_group in OVERLAY_TOOL_SPECS:
        if spec_group != group_key:
            continue
        var = getattr(gui, attr, None)
        if var is None:
            continue
        # Icon-first labels keep the ribbon short while remaining scannable.
        btn = ttk.Checkbutton(
            group,
            text=f"{icon_i} {label}",
            variable=var,
            command=gui._on_biomech_overlay_toggle,
            takefocus=False,
        )
        btn.pack(side=tk.LEFT, padx=(0, 2))
        from stablewalk.ui.metric_tooltips import get_metric_tooltip

        science = get_metric_tooltip(tip) if tip in ("com", "bos") else None
        create_tooltip(
            btn,
            science or f"{icon_i} {label}\n{tip}",
            wraplength=340,
        )

    return group


def build_overlay_control_bar(gui: Any, parent: tk.Misc) -> tuple[tk.Frame, tk.Frame]:
    """Build the grouped overlay control bar above the skeleton.

    Returns ``(bar, right_slot)``. ``right_slot`` is an empty, right-aligned
    container the caller can drop extra chrome into (e.g. the display-mode
    selector), keeping everything on one line above the viewport so the
    visualization keeps the maximum possible area.
    """
    ensure_overlay_vars(gui)

    bar = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    gui._skel_overlay_bar = bar

    # Right-aligned slot for external controls (packed first so it hugs the edge).
    right_slot = tk.Frame(bar, bg=ELEVATED, highlightthickness=0)
    right_slot.pack(side=tk.RIGHT, padx=(PAD_SM, PAD_XS), pady=1)
    gui._skel_overlay_right_slot = right_slot

    # Click-a-joint → DOF / 3D path selection (always on; toggle keeps highlight).
    if not hasattr(gui, "var_skeleton_pick_dof") or gui.var_skeleton_pick_dof is None:
        gui.var_skeleton_pick_dof = tk.BooleanVar(value=True)
    pick_btn = ttk.Checkbutton(
        bar,
        text="Select DOF",
        variable=gui.var_skeleton_pick_dof,
        command=getattr(gui, "_on_skeleton_pick_dof_toggle", None),
        takefocus=False,
    )
    pick_btn.pack(side=tk.LEFT, padx=(PAD_XS, 2), pady=1)
    create_tooltip(
        pick_btn,
        "Click a joint on the 3D gait figure to select that degree of freedom "
        "and show its 3D path. Hold Ctrl/Shift to compare multiple joints.",
        wraplength=340,
    )

    first = True
    for group_key, title, icon in OVERLAY_GROUPS:
        if not first:
            _vsep(bar).pack(side=tk.LEFT, fill=tk.Y, padx=(2, 2), pady=2)
        _build_group_inline(gui, bar, group_key, title, icon).pack(
            side=tk.LEFT, padx=(PAD_XS if first else 0, 0), pady=1
        )
        first = False

    return bar, right_slot


__all__ = [
    "OVERLAY_GROUPS",
    "OVERLAY_TOOL_SPECS",
    "build_overlay_control_bar",
    "ensure_overlay_vars",
]
