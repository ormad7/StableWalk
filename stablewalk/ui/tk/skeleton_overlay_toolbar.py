"""3D Skeleton Viewer controls — minimal chrome + Advanced dropdown.

Primary bar (always visible): Joint Selector · Advanced.
``primary_only`` (Overview) omits Play/Pause — the global transport owns that.
Select DOF, skeleton type, maximize, and all overlays live in a floating
Advanced panel so the viewport keeps nearly the full panel height.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.scientific_labels import OVERLAY_BOS, OVERLAY_COM, OVERLAY_COM_VEL
from stablewalk.ui.theme import (
    BORDER,
    BORDER_SUBTLE,
    ELEVATED,
    FONT_UI_XS,
    MUTED,
    OVERVIEW_VIZ_TOOLBAR_MIN_H,
    PAD_SM,
    PAD_XS,
    create_tooltip,
)

# Group key -> (title, icon). Order defines layout inside Advanced.
OVERLAY_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("motion", "Motion", "\u21dd"),
    ("biomechanics", "Biomechanics", "\u2696"),
    ("environment", "Environment", "\u25a4"),
)

# icon, short label, BooleanVar attr, default_on, tooltip, group
# All overlays default OFF — enable only via Advanced.
OVERLAY_TOOL_SPECS: tuple[tuple[str, str, str, bool, str, str], ...] = (
    ("\u21e2", "Direction", "var_overlay_direction", False,
     "Gait travel direction", "motion"),
    ("\u25bc", "Contact", "var_overlay_contact", False,
     "Foot contact markers", "motion"),
    ("\u25cf", OVERLAY_COM, "var_overlay_com", False,
     "com", "biomechanics"),
    ("\u25c7", OVERLAY_BOS, "var_overlay_bos", False,
     "bos", "biomechanics"),
    ("\u2197", OVERLAY_COM_VEL, "var_overlay_com_velocity", False,
     "COM velocity from tracked COM motion (display overlay)", "biomechanics"),
    ("\u25ac", "Ground", "var_overlay_ground", False,
     "Ground / floor reference", "environment"),
)


def ensure_overlay_vars(gui: Any) -> None:
    """Create the overlay ``BooleanVar``s on *gui* if they don't exist yet."""
    for _icon, _label, attr, default_on, _tip, _group in OVERLAY_TOOL_SPECS:
        if not hasattr(gui, attr) or getattr(gui, attr) is None:
            setattr(gui, attr, tk.BooleanVar(value=default_on))


def _vsep(parent: tk.Misc) -> tk.Frame:
    """Quiet vertical divider between Advanced overlay groups."""
    return tk.Frame(parent, bg=BORDER_SUBTLE, width=1, highlightthickness=0)


def _build_group_inline(
    gui: Any,
    parent: tk.Misc,
    group_key: str,
    title: str,
) -> tk.Frame:
    """One overlay group with a short title + latched tool toggles."""
    group = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)

    tk.Label(
        group,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(side=tk.LEFT, padx=(0, PAD_XS))

    for _icon_i, label, attr, _default, tip, spec_group in OVERLAY_TOOL_SPECS:
        if spec_group != group_key:
            continue
        var = getattr(gui, attr, None)
        if var is None:
            continue
        btn = ttk.Checkbutton(
            group,
            text=label,
            variable=var,
            command=gui._on_biomech_overlay_toggle,
            takefocus=False,
            style="Tool.TCheckbutton",
        )
        btn.pack(side=tk.LEFT, padx=(0, PAD_XS))
        from stablewalk.ui.metric_tooltips import get_metric_tooltip

        science = get_metric_tooltip(tip) if tip in ("com", "bos") else None
        create_tooltip(btn, science or tip or label, wraplength=340)

    return group


def _build_advanced_dropdown(gui: Any, bar: tk.Misc, anchor: tk.Misc) -> tk.Frame:
    """Floating Advanced panel anchored under ``anchor`` (does not steal grid height)."""
    panel = tk.Frame(
        bar.winfo_toplevel(),
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        padx=PAD_SM,
        pady=PAD_SM,
    )
    gui._skel_advanced_body = panel
    gui._skel_advanced_wrap = panel
    gui._skel_advanced_open = False

    # Row 0 — interaction + display extras slot (Select DOF, type, maximize).
    top = tk.Frame(panel, bg=ELEVATED, highlightthickness=0)
    top.pack(side=tk.TOP, fill=tk.X)

    if not hasattr(gui, "var_skeleton_pick_dof") or gui.var_skeleton_pick_dof is None:
        gui.var_skeleton_pick_dof = tk.BooleanVar(value=True)
    pick_btn = ttk.Checkbutton(
        top,
        text="Select DOF",
        variable=gui.var_skeleton_pick_dof,
        command=getattr(gui, "_on_skeleton_pick_dof_toggle", None),
        takefocus=False,
        style="Tool.TCheckbutton",
    )
    pick_btn.pack(side=tk.LEFT, padx=(0, PAD_SM))
    create_tooltip(
        pick_btn,
        "Select DOF — click a joint on the 3D figure to show its path. "
        "Hold Ctrl/Shift to compare multiple joints.",
        wraplength=340,
    )

    model_btn = ttk.Button(
        top,
        text="Model",
        width=7,
        style="Secondary.TButton",
        command=lambda: getattr(gui, "_open_skeleton_model_inspector_window", lambda: None)(),
        takefocus=False,
    )
    model_btn.pack(side=tk.LEFT, padx=(0, PAD_SM))
    gui._skel_model_inspector_btn = model_btn
    create_tooltip(
        model_btn,
        "Skeleton Model Inspector — joints, segments, and DOFs counted separately "
        "(not MediaPipe landmark totals).",
        wraplength=360,
    )

    display_slot = tk.Frame(top, bg=ELEVATED, highlightthickness=0)
    display_slot.pack(side=tk.LEFT, padx=(0, PAD_SM))
    gui._skel_advanced_display_slot = display_slot
    # Back-compat alias used by dashboard_layout for mode / maximize.
    gui._skel_overlay_right_slot = display_slot

    # Row 1 — biomechanics overlays.
    overlays = tk.Frame(panel, bg=ELEVATED, highlightthickness=0)
    overlays.pack(side=tk.TOP, fill=tk.X, pady=(PAD_SM, 0))

    first = True
    for group_key, title, _icon in OVERLAY_GROUPS:
        if not first:
            _vsep(overlays).pack(
                side=tk.LEFT, fill=tk.Y, padx=(PAD_SM, PAD_SM), pady=2
            )
        _build_group_inline(gui, overlays, group_key, title).pack(
            side=tk.LEFT, padx=(PAD_SM if first else 0, 0)
        )
        first = False

    def _close() -> None:
        gui._skel_advanced_open = False
        try:
            panel.place_forget()
        except tk.TclError:
            pass
        toggle = getattr(gui, "_skel_advanced_toggle", None)
        if toggle is not None:
            try:
                toggle.configure(text="Advanced ▾")
            except tk.TclError:
                pass

    def _open() -> None:
        gui._skel_advanced_open = True
        try:
            panel.update_idletasks()
            ax = int(anchor.winfo_rootx())
            ay = int(anchor.winfo_rooty() + anchor.winfo_height() + 2)
            top_w = bar.winfo_toplevel()
            # Place in root coords relative to toplevel.
            rx = int(top_w.winfo_rootx())
            ry = int(top_w.winfo_rooty())
            panel.place(x=max(0, ax - rx), y=max(0, ay - ry))
            panel.lift()
        except tk.TclError:
            pass
        toggle = getattr(gui, "_skel_advanced_toggle", None)
        if toggle is not None:
            try:
                toggle.configure(text="Advanced ▴")
            except tk.TclError:
                pass

    def _toggle() -> None:
        if getattr(gui, "_skel_advanced_open", False):
            _close()
        else:
            _open()

    gui._skel_advanced_close = _close
    gui._skel_advanced_open_fn = _open
    gui._skel_advanced_toggle_fn = _toggle

    # Dismiss when clicking outside (bind once on toplevel).
    def _on_root_press(event: tk.Event) -> None:
        if not getattr(gui, "_skel_advanced_open", False):
            return
        widget = event.widget
        try:
            if str(widget).startswith(str(panel)):
                return
            toggle = getattr(gui, "_skel_advanced_toggle", None)
            if toggle is not None and str(widget).startswith(str(toggle)):
                return
        except Exception:
            pass
        _close()

    root = bar.winfo_toplevel()
    prev = getattr(gui, "_skel_advanced_root_bind", None)
    if prev is None:
        root.bind_all("<ButtonPress-1>", _on_root_press, add="+")
        gui._skel_advanced_root_bind = True

    return panel


def build_overlay_control_bar(
    gui: Any,
    parent: tk.Misc,
    *,
    primary_only: bool = False,
) -> tuple[tk.Frame, tk.Frame]:
    """Build the minimal skeleton control bar above the viewport.

    Returns ``(bar, advanced_display_slot)``. Primary controls: Joint Selector
    and Advanced. Play/Pause is omitted on Overview (``primary_only``) so the
    bottom transport is the only playback control.
    """
    ensure_overlay_vars(gui)

    bar = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    gui._skel_overlay_bar = bar

    primary = tk.Frame(bar, bg=ELEVATED, highlightthickness=0)
    # Grid keeps Joint + combo + Advanced inside this panel. Pack-LEFT/RIGHT
    # let a 24-character combobox overflow into the neighbouring Path column.
    primary.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)
    primary.columnconfigure(0, weight=0)
    primary.columnconfigure(1, weight=1)
    primary.columnconfigure(2, weight=0)
    primary.columnconfigure(3, weight=0)
    primary.columnconfigure(4, weight=0)

    gui.btn_play_skeleton = None

    fit_btn = ttk.Button(
        primary,
        text="Fit",
        width=4,
        style="Secondary.TButton",
        command=lambda: getattr(gui, "_fit_full_body", lambda: None)(),
        takefocus=False,
    )
    fit_btn.grid(row=0, column=2, sticky="e", padx=(PAD_XS, 0), pady=0)
    gui._skel_fit_full_body_btn = fit_btn
    create_tooltip(
        fit_btn,
        "Fit Full Body — frame head-to-feet using the full motion range.",
        wraplength=320,
    )

    # Advanced dropdown toggle — last column of this panel only.
    adv_btn = ttk.Button(
        primary,
        text="Advanced ▾",
        width=11,
        style="Secondary.TButton",
        takefocus=False,
    )
    adv_btn.grid(row=0, column=3, sticky="e", padx=(PAD_XS, 0), pady=0)
    gui._skel_advanced_toggle = adv_btn
    create_tooltip(
        adv_btn,
        "Advanced — Select DOF, skeleton type, overlays (COM, BoS, contact, …).",
        wraplength=340,
    )

    if not primary_only:
        play_btn = ttk.Button(
            primary,
            text="Play",
            width=7,
            style="Secondary.TButton",
            command=getattr(gui, "_toggle_play", None),
            takefocus=False,
        )
        play_btn.grid(row=0, column=4, sticky="e", padx=(PAD_XS, 0), pady=0)
        gui.btn_play_skeleton = play_btn
        create_tooltip(
            play_btn, "Play / pause (Space) — same as bottom transport", wraplength=260
        )

    # Joint selector — primary left control.
    from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS

    tk.Label(
        primary,
        text="Joint",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        padx=2,
        pady=2,
    ).grid(row=0, column=0, sticky="w", padx=(PAD_XS, PAD_XS))

    if not hasattr(gui, "var_overview_joint_pick") or gui.var_overview_joint_pick is None:
        gui.var_overview_joint_pick = tk.StringVar(value="Select joint…")
    joint_labels = ["Select joint…"] + [
        GUI_DOF_LABELS.get(item_id, item_id) for item_id in GUI_DOF_ITEM_IDS
    ]
    from stablewalk.ui.tk.layout_scale import combobox_char_width

    # Cap character width so "Right Knee" stays readable without spilling
    # into the Path panel. Longer names remain in the dropdown list.
    pick_combo = ttk.Combobox(
        primary,
        textvariable=gui.var_overview_joint_pick,
        values=joint_labels,
        state="readonly",
        width=combobox_char_width(joint_labels, minimum=12, maximum=16),
        takefocus=False,
    )
    pick_combo.grid(row=0, column=1, sticky="ew", padx=(0, PAD_SM), pady=0)
    gui.cmb_overview_joint_pick = pick_combo
    gui._skel_overlay_primary = primary
    create_tooltip(
        pick_combo,
        "Choose an analysis DOF (shoulders → heels). Same as clicking that joint.",
        wraplength=340,
    )

    def _on_joint_pick_combo(_event: object = None) -> None:
        choose = getattr(gui, "_on_overview_joint_pick_chosen", None)
        if callable(choose):
            choose()

    pick_combo.bind("<<ComboboxSelected>>", _on_joint_pick_combo)

    def _reflow_skel_primary(_event: object | None = None) -> None:
        try:
            width = int(primary.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 80:
            return
        stacked = width < 300
        leftover = max(72, width - (48 if stacked else 220))
        chars = max(12, min(18, leftover // 8))
        try:
            if stacked:
                adv_btn.grid(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky="w",
                    padx=(PAD_XS, 0),
                    pady=(PAD_XS, 0),
                )
            else:
                adv_btn.grid(
                    row=0,
                    column=3,
                    sticky="e",
                    padx=(PAD_XS, 0),
                    pady=0,
                )
            if int(pick_combo.cget("width") or 0) != chars:
                pick_combo.configure(width=chars)
            host = bar.master
            extra = 22 if stacked else 0
            host.rowconfigure(0, weight=0, minsize=OVERVIEW_VIZ_TOOLBAR_MIN_H + extra)
        except (tk.TclError, TypeError, ValueError):
            pass

    primary.bind("<Configure>", _reflow_skel_primary, add="+")

    # Build floating Advanced panel (not packed into the bar → no viewport loss).
    _build_advanced_dropdown(gui, bar, adv_btn)
    adv_btn.configure(
        command=lambda: getattr(gui, "_skel_advanced_toggle_fn", lambda: None)()
    )

    display_slot = getattr(gui, "_skel_advanced_display_slot", None)
    if display_slot is None:
        display_slot = tk.Frame(bar, bg=ELEVATED)
        gui._skel_overlay_right_slot = display_slot

    return bar, display_slot


__all__ = [
    "OVERLAY_GROUPS",
    "OVERLAY_TOOL_SPECS",
    "build_overlay_control_bar",
    "ensure_overlay_vars",
]
