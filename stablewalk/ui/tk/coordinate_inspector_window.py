"""Detachable 3D Coordinate Inspector — joint, frame, and live kinematics.

Updates labels from cached playhead fields. Frame-slider seeks are coalesced
so dragging does not freeze the GUI.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS, label_for_item
from stablewalk.ui.overview_coordinate_inspector import (
    COORDINATE_INSPECTOR_PRIMARY_KEYS,
    COORDINATE_INSPECTOR_TITLES,
    INSPECTOR_UNAVAILABLE_DISPLAY,
)
from stablewalk.ui.theme import (
    ACCENT_ALT,
    BG,
    BORDER,
    FONT_MONO,
    FONT_SECTION,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
)

_INSPECTOR_SEEK_MS = 16


def open_coordinate_inspector_window(gui: Any) -> None:
    """Open or raise the detachable 3D Coordinate Inspector."""
    existing = getattr(gui, "_coord_inspector_window", None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.lift()
                existing.focus_force()
                _sync_inspector_window_controls(gui)
                _refresh_inspector_window(gui)
                return
        except tk.TclError:
            pass

    dlg = tk.Toplevel(gui.root)
    dlg.title("3D Coordinate Inspector")
    dlg.configure(bg=BG)
    dlg.geometry("480x560")
    dlg.minsize(380, 440)
    try:
        dlg.resizable(True, True)
    except tk.TclError:
        pass
    dlg.transient(gui.root)
    gui._coord_inspector_window = dlg
    gui._last_inspector_window_fields = None
    gui._inspector_window_seek_lock = False
    gui._inspector_window_joint_lock = False
    gui._inspector_seek_after = None
    gui._pending_inspector_frame = None

    shell = tk.Frame(dlg, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
    shell.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=PAD_SM)

    tk.Label(
        shell,
        text="3D Coordinate Inspector",
        bg=PANEL,
        fg=TEXT,
        font=FONT_SECTION,
        anchor="w",
    ).pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, PAD_XS))

    controls = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    controls.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_XS))
    controls.columnconfigure(1, weight=1)

    tk.Label(
        controls,
        text="Joint",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=(0, PAD_SM), pady=(0, PAD_XS))

    joint_labels = [GUI_DOF_LABELS[item_id] for item_id in GUI_DOF_ITEM_IDS]
    joint_var = tk.StringVar(value=_current_joint_label(gui))
    joint_combo = ttk.Combobox(
        controls,
        textvariable=joint_var,
        values=joint_labels,
        state="readonly",
        width=18,
    )
    joint_combo.grid(row=0, column=1, sticky="ew", pady=(0, PAD_XS))
    create_tooltip(joint_combo, "Inspect this joint. Also updates the main selection.")
    gui._coord_inspector_window_joint_var = joint_var

    tk.Label(
        controls,
        text="Frame",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=1, column=0, sticky="nw", padx=(0, PAD_SM), pady=(PAD_XS, 0))

    frame_host = tk.Frame(controls, bg=PANEL, highlightthickness=0)
    frame_host.grid(row=1, column=1, sticky="ew")
    frame_host.columnconfigure(0, weight=1)

    frame_count = _frame_count(gui)
    frame_var = tk.IntVar(value=_current_frame_index(gui))
    scale = ttk.Scale(
        frame_host,
        from_=0,
        to=max(0, frame_count - 1),
        orient=tk.HORIZONTAL,
        variable=frame_var,
        command=lambda _v: _on_inspector_frame(gui),
    )
    scale.grid(row=0, column=0, sticky="ew")
    frame_lbl = tk.Label(
        frame_host,
        text=_frame_index_text(gui),
        bg=PANEL,
        fg=TEXT,
        font=FONT_MONO,
        anchor="e",
        width=12,
    )
    frame_lbl.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))
    gui._coord_inspector_window_frame_var = frame_var
    gui._coord_inspector_window_scale = scale
    gui._coord_inspector_window_frame_label = frame_lbl

    fields_host = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    fields_host.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=(PAD_XS, PAD_SM))
    fields_host.columnconfigure(0, weight=0, minsize=118)
    fields_host.columnconfigure(1, weight=1)

    values: dict[str, tk.Label] = {}
    for row_i, key in enumerate(COORDINATE_INSPECTOR_PRIMARY_KEYS):
        tk.Label(
            fields_host,
            text=COORDINATE_INSPECTOR_TITLES.get(key, key),
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).grid(row=row_i, column=0, sticky="nw", pady=1, padx=(0, PAD_SM))
        value_lbl = tk.Label(
            fields_host,
            text=INSPECTOR_UNAVAILABLE_DISPLAY,
            bg=PANEL,
            fg=MUTED,
            font=FONT_MONO,
            anchor="e",
            justify=tk.RIGHT,
        )
        value_lbl.grid(row=row_i, column=1, sticky="ne", pady=1)
        values[key] = value_lbl
    gui._coord_inspector_window_values = values

    hint = tk.Label(
        shell,
        text="Values follow the playhead. Drag the frame slider to inspect without rebuilding analysis.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        wraplength=420,
        justify=tk.LEFT,
        anchor="w",
    )
    hint.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_SM))

    def _on_joint_change(_event: object | None = None) -> None:
        if getattr(gui, "_inspector_window_joint_lock", False):
            return
        label = str(joint_var.get() or "")
        item_id = next(
            (key for key, name in GUI_DOF_LABELS.items() if name == label),
            "",
        )
        if not item_id:
            return
        activate = getattr(gui, "_activate_dof_item", None)
        if callable(activate):
            try:
                activate(item_id, add_if_missing=True)
            except Exception:
                pass
        try:
            _refresh_inspector_window(gui)
        except Exception:
            pass

    joint_combo.bind("<<ComboboxSelected>>", _on_joint_change)

    def _on_close() -> None:
        gui._coord_inspector_window = None
        gui._coord_inspector_window_values = None
        gui._coord_inspector_window_joint_var = None
        gui._coord_inspector_window_frame_var = None
        gui._coord_inspector_window_scale = None
        gui._coord_inspector_window_frame_label = None
        gui._last_inspector_window_fields = None
        token = getattr(gui, "_inspector_seek_after", None)
        if token is not None:
            try:
                gui.root.after_cancel(token)
            except (tk.TclError, ValueError):
                pass
        gui._inspector_seek_after = None
        try:
            dlg.destroy()
        except tk.TclError:
            pass

    dlg.protocol("WM_DELETE_WINDOW", _on_close)

    def _on_hint_resize(event: tk.Event) -> None:
        wrap = max(220, int(event.width) - 24)
        try:
            hint.configure(wraplength=wrap)
        except tk.TclError:
            pass

    hint.bind("<Configure>", _on_hint_resize, add="+")

    active = None
    getter = getattr(gui, "_active_dof_item_id", None)
    if callable(getter):
        active = getter()
    if not active:
        _on_joint_change()
    else:
        _refresh_inspector_window(gui)


def sync_coordinate_inspector_window(gui: Any, *, frame_index: int | None = None) -> None:
    """Keep the window slider/joint combo aligned with the playhead."""
    dlg = getattr(gui, "_coord_inspector_window", None)
    if dlg is None:
        return
    try:
        if not dlg.winfo_exists():
            return
    except tk.TclError:
        return
    _sync_inspector_window_controls(gui, frame_index=frame_index)


def _refresh_inspector_window(gui: Any) -> None:
    refresh = getattr(gui, "_refresh_overview_trajectory_readout", None)
    if callable(refresh):
        refresh(getattr(gui, "_active_dof_item_id", lambda: None)())


def _on_inspector_frame(gui: Any) -> None:
    if getattr(gui, "_inspector_window_seek_lock", False):
        return
    if getattr(gui, "_sync_all_guard", False):
        return
    var = getattr(gui, "_coord_inspector_window_frame_var", None)
    if var is None:
        return
    try:
        frame = int(float(var.get()))
    except (tk.TclError, TypeError, ValueError):
        return
    lbl = getattr(gui, "_coord_inspector_window_frame_label", None)
    if lbl is not None:
        try:
            lbl.configure(text=_frame_index_text(gui, frame))
        except tk.TclError:
            pass
    gui._pending_inspector_frame = frame
    if getattr(gui, "_inspector_seek_after", None) is not None:
        return
    try:
        gui._inspector_seek_after = gui.root.after(
            _INSPECTOR_SEEK_MS,
            lambda: _flush_inspector_seek(gui),
        )
    except tk.TclError:
        gui._inspector_seek_after = None


def _flush_inspector_seek(gui: Any) -> None:
    gui._inspector_seek_after = None
    frame = getattr(gui, "_pending_inspector_frame", None)
    if frame is None:
        return
    player = getattr(gui, "skeleton_player", None)
    if player is None:
        return
    if bool(getattr(gui, "playing", False)):
        gui.playing = False
        try:
            player.pause()
        except Exception:
            pass
        sync = getattr(gui, "_sync_play_buttons", None)
        if callable(sync):
            sync()
        cancel = getattr(gui, "_cancel_timer", None)
        if callable(cancel):
            cancel()
    go_to = getattr(gui, "_go_to", None)
    if callable(go_to):
        go_to(int(frame))


def _sync_inspector_window_controls(
    gui: Any,
    *,
    frame_index: int | None = None,
) -> None:
    gui._inspector_window_seek_lock = True
    gui._inspector_window_joint_lock = True
    try:
        frame = _current_frame_index(gui) if frame_index is None else int(frame_index)
        var = getattr(gui, "_coord_inspector_window_frame_var", None)
        scale = getattr(gui, "_coord_inspector_window_scale", None)
        count = max(1, _frame_count(gui))
        if scale is not None:
            try:
                scale.configure(to=max(0, count - 1))
            except tk.TclError:
                pass
        if var is not None:
            try:
                if int(var.get()) != frame:
                    var.set(frame)
            except (tk.TclError, TypeError, ValueError):
                pass
        lbl = getattr(gui, "_coord_inspector_window_frame_label", None)
        if lbl is not None:
            try:
                lbl.configure(text=_frame_index_text(gui, frame))
            except tk.TclError:
                pass
        joint_var = getattr(gui, "_coord_inspector_window_joint_var", None)
        label = _current_joint_label(gui)
        if joint_var is not None and label:
            try:
                if str(joint_var.get() or "") != label:
                    joint_var.set(label)
            except tk.TclError:
                pass
    finally:
        gui._inspector_window_seek_lock = False
        gui._inspector_window_joint_lock = False


def _current_joint_label(gui: Any) -> str:
    active = None
    getter = getattr(gui, "_active_dof_item_id", None)
    if callable(getter):
        active = getter()
    if active:
        return label_for_item(active)
    return GUI_DOF_LABELS[GUI_DOF_ITEM_IDS[0]]


def _frame_count(gui: Any) -> int:
    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        return max(0, int(getattr(player, "frame_count", 0) or 0))
    recording = None
    getter = getattr(gui, "_analysis_motion_recording", None)
    if callable(getter):
        recording = getter()
    if recording is not None:
        return max(0, int(getattr(recording, "frame_count", 0) or 0))
    return 0


def _current_frame_index(gui: Any) -> int:
    try:
        from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

        frame = playhead_frame_0based(gui)
        if frame is not None:
            return max(0, int(frame))
    except Exception:
        pass
    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        try:
            return max(0, int(player.state.frame_index))
        except Exception:
            pass
    try:
        return max(0, int(gui.current_pos))
    except Exception:
        return 0


def _frame_index_text(gui: Any, frame: int | None = None) -> str:
    index = _current_frame_index(gui) if frame is None else int(frame)
    total = max(0, _frame_count(gui))
    if total <= 0:
        return "— / —"
    return f"{index + 1} / {total}"
