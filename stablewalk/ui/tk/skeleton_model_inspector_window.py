"""Detachable Skeleton Model Inspector — joints vs segments vs DOFs."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk.ui.skeleton_model_inspector import (
    SKELETON_MODEL_INSPECTOR_COUNT_KEYS,
    build_skeleton_model_inspector_fields,
    inspector_component_rows,
    inspector_dof_rows,
    inspector_hierarchy_text,
    inspector_verdict_rows,
    limbs_connected,
)
from stablewalk.ui.theme import (
    BG,
    BORDER,
    FONT_MONO,
    FONT_SECTION,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    create_tooltip,
)


def open_skeleton_model_inspector_window(gui: Any) -> None:
    """Open or raise the technical skeleton / DOF inventory window."""
    existing = getattr(gui, "_skel_model_inspector_window", None)
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.lift()
                existing.focus_force()
                refresh_skeleton_model_inspector(gui)
                return
        except tk.TclError:
            pass

    dlg = tk.Toplevel(gui.root)
    dlg.title("Skeleton Model Inspector")
    dlg.configure(bg=BG)
    dlg.geometry("760x640")
    dlg.minsize(560, 480)
    try:
        dlg.resizable(True, True)
    except tk.TclError:
        pass
    dlg.transient(gui.root)
    gui._skel_model_inspector_window = dlg

    shell = tk.Frame(dlg, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
    shell.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=PAD_SM)
    shell.rowconfigure(3, weight=1)
    shell.columnconfigure(0, weight=1)

    tk.Label(
        shell,
        text="Skeleton Model Inspector",
        bg=PANEL,
        fg=TEXT,
        font=FONT_SECTION,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))

    tk.Label(
        shell,
        text=(
            "Landmarks, reconstructed joints, connected segments, independent DOFs, "
            "and OpenSim coordinates are different inventories. This wire is not Gait2392."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        wraplength=720,
        justify="left",
        anchor="w",
    ).grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_XS))

    counts = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    counts.grid(row=2, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_XS))
    gui._skel_model_count_labels = {}
    for i, (key, title) in enumerate(SKELETON_MODEL_INSPECTOR_COUNT_KEYS):
        cell = tk.Frame(counts, bg=PANEL, highlightthickness=0)
        cell.grid(row=i // 5, column=i % 5, sticky="nw", padx=(0, PAD_SM), pady=(0, PAD_XS))
        tk.Label(cell, text=title, bg=PANEL, fg=MUTED, font=FONT_UI_XS, anchor="w").pack(
            anchor="w"
        )
        val = tk.Label(cell, text="—", bg=PANEL, fg=TEXT, font=FONT_SECTION, anchor="w")
        val.pack(anchor="w")
        gui._skel_model_count_labels[key] = val

    live = tk.Frame(shell, bg=PANEL, highlightthickness=0)
    live.grid(row=3, column=0, sticky="nsew", padx=PAD_SM, pady=(0, PAD_SM))
    live.columnconfigure(0, weight=1)
    live.columnconfigure(1, weight=1)
    live.rowconfigure(1, weight=1)

    gui._skel_model_live_var = tk.StringVar(value="Frame —")
    tk.Label(
        live,
        textvariable=gui._skel_model_live_var,
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew")
    gui._skel_model_bilateral_var = tk.StringVar(value="Bilateral —")
    tk.Label(
        live,
        textvariable=gui._skel_model_bilateral_var,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="e",
    ).grid(row=0, column=1, sticky="ew")

    notebook = ttk.Notebook(live)
    notebook.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(PAD_XS, 0))

    tab_comp = tk.Frame(notebook, bg=PANEL)
    tab_dof = tk.Frame(notebook, bg=PANEL)
    tab_tree = tk.Frame(notebook, bg=PANEL)
    tab_req = tk.Frame(notebook, bg=PANEL)
    notebook.add(tab_comp, text="Components")
    notebook.add(tab_dof, text="Wire DOFs")
    notebook.add(tab_tree, text="Hierarchy")
    notebook.add(tab_req, text="12–20 readings")

    cols = ("parent", "child", "xyz", "dof", "opensim", "video", "viz", "conf")
    headers = {
        "parent": "Parent",
        "child": "Child",
        "xyz": "Position source",
        "dof": "Wire rotational DOF",
        "opensim": "OpenSim rotational (Gait2392)",
        "video": "Video-driven",
        "viz": "Visualized",
        "conf": "Confidence",
    }
    tree = ttk.Treeview(tab_comp, columns=cols, show="tree headings", height=12)
    tree.heading("#0", text="Segment / joint")
    tree.column("#0", width=160, stretch=False)
    for col in cols:
        tree.heading(col, text=headers[col])
        tree.column(col, width=110 if col != "xyz" else 180, stretch=True)
    vsb = ttk.Scrollbar(tab_comp, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tab_comp.rowconfigure(0, weight=1)
    tab_comp.columnconfigure(0, weight=1)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    gui._skel_model_component_tree = tree

    dof_cols = ("joint", "angle")
    dof_tree = ttk.Treeview(tab_dof, columns=dof_cols, show="tree headings", height=12)
    dof_tree.heading("#0", text="Independent sagittal DOF")
    dof_tree.column("#0", width=220, stretch=True)
    dof_tree.heading("joint", text="Anchor joint")
    dof_tree.heading("angle", text="Angle (this frame)")
    dof_tree.column("joint", width=140)
    dof_tree.column("angle", width=120)
    tab_dof.rowconfigure(0, weight=1)
    tab_dof.columnconfigure(0, weight=1)
    dof_tree.grid(row=0, column=0, sticky="nsew")
    gui._skel_model_dof_tree = dof_tree

    hier = tk.Text(
        tab_tree,
        bg=PANEL,
        fg=TEXT,
        font=FONT_MONO,
        wrap="none",
        height=16,
        relief="flat",
        highlightthickness=0,
    )
    hier.pack(fill=tk.BOTH, expand=True, padx=PAD_XS, pady=PAD_XS)
    hier.insert("1.0", inspector_hierarchy_text())
    hier.configure(state="disabled")

    req_cols = ("count", "verdict")
    req_tree = ttk.Treeview(tab_req, columns=req_cols, show="tree headings", height=12)
    req_tree.heading("#0", text="If “12–20” means…")
    req_tree.column("#0", width=340, stretch=True)
    req_tree.heading("count", text="Count")
    req_tree.heading("verdict", text="Result")
    req_tree.column("count", width=80)
    req_tree.column("verdict", width=280)
    tab_req.rowconfigure(0, weight=1)
    tab_req.columnconfigure(0, weight=1)
    req_tree.grid(row=0, column=0, sticky="nsew")
    for row in inspector_verdict_rows():
        req_tree.insert("", "end", text=row["interpretation"], values=(row["count"], row["verdict"]))

    q = tk.Label(
        tab_req,
        text=(
            "Clarification for the supervisor: should 12–20 count independent sagittal "
            "DOFs (13), unique anatomical joints/segments (19), or a full OpenSim "
            "CoordinateSet (23 in Gait2392)? MediaPipe’s 33 landmarks are detections, "
            "not degrees of freedom."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        wraplength=700,
        justify="left",
        anchor="w",
    )
    q.grid(row=1, column=0, sticky="ew", padx=PAD_XS, pady=PAD_XS)

    def _on_close() -> None:
        gui._skel_model_inspector_window = None
        try:
            dlg.destroy()
        except tk.TclError:
            pass

    dlg.protocol("WM_DELETE_WINDOW", _on_close)
    create_tooltip(
        dlg,
        "Static registry counts plus the current frame’s present joints/segments/DOFs.",
    )
    refresh_skeleton_model_inspector(gui)


def refresh_skeleton_model_inspector(gui: Any) -> None:
    """Push current recording/snapshot counts into an open inspector window."""
    dlg = getattr(gui, "_skel_model_inspector_window", None)
    if dlg is None:
        return
    try:
        if not dlg.winfo_exists():
            gui._skel_model_inspector_window = None
            return
    except tk.TclError:
        gui._skel_model_inspector_window = None
        return

    recording = None
    getter = getattr(gui, "_analysis_motion_recording", None)
    if callable(getter):
        recording = getter()
    snapshot = None
    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        try:
            snapshot = player.current_snapshot()
        except Exception:
            snapshot = None
    if snapshot is None and recording is not None:
        try:
            from stablewalk.ui.tk.dashboard_status_bar import playhead_frame_0based

            idx = playhead_frame_0based(gui) or 0
            snapshot = recording.snapshot_at(int(idx))
        except Exception:
            snapshot = recording.snapshot_at(0) if recording.frame_count else None

    fields = build_skeleton_model_inspector_fields(
        snapshot=snapshot, recording=recording
    )
    labels = getattr(gui, "_skel_model_count_labels", {}) or {}
    for key, widget in labels.items():
        try:
            widget.configure(text=fields.get(key, "—"))
        except tk.TclError:
            pass
    live = getattr(gui, "_skel_model_live_var", None)
    if live is not None:
        live.set(
            f"Frame {fields.get('frame', '—')}  ·  "
            f"joints {fields.get('live_joints', '—')}  ·  "
            f"segments {fields.get('live_segments', '—')}  ·  "
            f"DOFs {fields.get('live_dofs', '—')}"
        )
    bi = getattr(gui, "_skel_model_bilateral_var", None)
    if bi is not None:
        extra = ""
        if snapshot is not None:
            chains = limbs_connected(snapshot)
            missing = [name for name, ok in chains.items() if not ok]
            extra = "" if not missing else f"  missing: {', '.join(missing)}"
        bi.set(f"Bilateral {fields.get('bilateral', '—')}{extra}")

    tree = getattr(gui, "_skel_model_component_tree", None)
    if tree is not None:
        try:
            tree.delete(*tree.get_children())
            for row in inspector_component_rows():
                tree.insert(
                    "",
                    "end",
                    text=row["name"],
                    values=(
                        row.get("parent", ""),
                        row.get("child", ""),
                        row.get("xyz", ""),
                        row.get("dof", ""),
                        row.get("opensim_rotational", ""),
                        row.get("video_driven", ""),
                        row.get("visualized", ""),
                        row.get("confidence", ""),
                    ),
                )
        except tk.TclError:
            pass

    dof_tree = getattr(gui, "_skel_model_dof_tree", None)
    if dof_tree is not None:
        try:
            dof_tree.delete(*dof_tree.get_children())
            for row in inspector_dof_rows(snapshot):
                dof_tree.insert(
                    "",
                    "end",
                    text=row["name"],
                    values=(row["joint"], row["angle"]),
                )
        except tk.TclError:
            pass
