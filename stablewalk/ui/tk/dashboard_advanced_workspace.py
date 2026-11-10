"""Advanced tab — compact collapsible sections (presentation only)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from stablewalk import config
from stablewalk.ui.theme import (
    BORDER_SUBTLE,
    FONT_SECTION,
    FONT_UI,
    FONT_UI_XS,
    MUTED,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
)

# Stack category cards before two half-width technical columns compress.
ADVANCED_TWO_COL_MIN = 1020

# Category key → (title, expanded_by_default)
# Session stays open so generated-file status is visible; other groups
# collapse so empty technical panels do not dominate the page.
ADVANCED_CATEGORIES: tuple[tuple[str, str, bool], ...] = (
    ("session", "Session", True),
    ("opensim", "OpenSim", True),
    ("real_to_sim", "Real-to-Sim", False),
    ("exports", "Exports", False),
    ("reports", "Reports", False),
)

# Wide layout: Session | OpenSim, Real-to-Sim full width, Exports | Reports.
_ADVANCED_TWO_COL_LAYOUT: tuple[tuple[str, int, int, int], ...] = (
    ("session", 0, 0, 1),
    ("opensim", 0, 1, 1),
    ("real_to_sim", 1, 0, 2),
    ("exports", 2, 0, 1),
    ("reports", 2, 1, 1),
)

# One-line purpose under each category title (presentation only).
# Order is the workspace reading path: Session → OpenSim → Real-to-Sim → outputs.
_CATEGORY_CAPTIONS: dict[str, str] = {
    "session": "Start here — Save this analysis, inspect joint data, and review generated files.",
    "opensim": "Model workspace — IK status, SDK state, and OpenSim artifact export.",
    "real_to_sim": "Simulation bridge — Retargeting, reference motion, contact/GRF outputs, and artifacts.",
    "exports": "Research data — raw coordinates, gait metrics, and files for downstream tools.",
    "reports": "Technical archives. Session Export Report and Export JSON are on Results.",
}

# High-level actions only (attr, label, command, primary, tooltip)
_CATEGORY_ACTIONS: dict[str, tuple[tuple[str, str, str, bool, str], ...]] = {
    "session": (
        (
            "btn_save_session",
            "Save Session",
            "_save_session_to_files",
            True,
            "Write a restoreable session bundle for the current analysis.",
        ),
        (
            "btn_view_detailed_data",
            "View Joint Data",
            "_open_collected_data_dialog",
            False,
            "Show the detailed joint table for selected points.",
        ),
        (
            "btn_open_coord_inspector",
            "3D Coordinate Inspector",
            "_open_coordinate_inspector_window",
            False,
            "Inspect selected-joint XYZ, velocity, acceleration, angle, and confidence.",
        ),
    ),
    "opensim": (
        (
            "btn_opensim_export_data",
            "Export OpenSim Files",
            "_export_opensim_session",
            True,
            "Export TRC/MOT/JSON markers for OpenSim IK (async).",
        ),
    ),
    "real_to_sim": (
        (
            "btn_real_to_sim_pipeline",
            "Run Real-to-Sim Pipeline",
            "_run_real_to_sim_pipeline",
            True,
            "Run retargeting + motion packaging for simulation (async).",
        ),
        (
            "btn_export_motion_reference",
            "Export Motion Reference",
            "_export_motion_reference",
            False,
            "Write stablewalk_motion.npz for downstream tools.",
        ),
        (
            "btn_export_amp_reference",
            "Export AMP Reference",
            "_export_amp_reference",
            False,
            "Write amp_reference_motion.npz for RL/AMP training.",
        ),
    ),
    "exports": (
        (
            "btn_export_gait_metrics",
            "Export Gait Metrics",
            "_export_gait_metrics",
            True,
            "JSON export of temporal / spatial gait metrics.",
        ),
        (
            "btn_export_joint_csv",
            "Export Joint Table",
            "_export_tracking_data",
            False,
            "Per-frame table: frame, time, x, y, z, angle, velocity, acceleration, confidence, gait phase.",
        ),
    ),
    "reports": (
        (
            "btn_export_analysis_report",
            "Export Analysis Report",
            "_export_analysis_report",
            True,
            "Full technical text report for archives. Results has the session Export Report.",
        ),
        (
            "btn_export_pdf_report",
            "Export Research PDF",
            "_export_professional_pdf_report",
            False,
            "Lab-style PDF package. Results has the session Export Report / Export JSON.",
        ),
    ),
}


def _category_card(
    parent: tk.Misc,
    *,
    title: str,
    expanded: bool,
    caption: str = "",
) -> tuple[tk.Frame, tk.Frame]:
    """Compact labeled collapsible section; returns (outer_frame, body)."""
    from stablewalk.ui.design_system.components import CollapsibleSection

    outer = tk.Frame(
        parent,
        bg=PANEL,
        highlightthickness=0,
    )
    outer.columnconfigure(0, weight=1)
    outer.rowconfigure(0, weight=0)

    section = CollapsibleSection(outer, title, expanded=expanded, bg=PANEL)
    section.frame.grid(row=0, column=0, sticky="nsew", padx=PAD_XS, pady=(PAD_XS, PAD_XS))
    section.frame.rowconfigure(1, weight=0)
    section.title_lbl.configure(
        fg=TEXT,
        font=FONT_SECTION,
        wraplength=360,
        anchor="w",
        justify=tk.LEFT,
    )
    section.title_lbl.grid_configure(sticky="ew")
    try:
        section.toggle.configure(fg=TEXT_SECONDARY, font=FONT_UI, activeforeground=TEXT)
    except tk.TclError:
        pass
    outer._sw_title = section.title_lbl  # type: ignore[attr-defined]
    try:
        section.header.grid_configure(pady=(0, PAD_XS))
        section.header.columnconfigure(1, weight=1)
        rule = tk.Frame(
            section.header, bg=BORDER_SUBTLE, height=1, highlightthickness=0
        )
        rule.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(PAD_XS, 0))
        outer._sw_header_rule = rule  # type: ignore[attr-defined]
    except tk.TclError:
        pass
    if caption:
        caption_lbl = tk.Label(
            section.body,
            text=caption,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=420,
        )
        caption_lbl.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
        outer._sw_caption = caption_lbl  # type: ignore[attr-defined]

    return outer, section.body


def _add_action_button(
    gui: Any,
    actions: tk.Misc,
    *,
    index: int,
    count: int,
    attr: str,
    text: str,
    cmd: str,
    primary: bool,
    help_line: str,
) -> ttk.Button:
    """Place one action in a compact 2-column button grid."""
    style = "ExportAccent.TButton" if primary else "Export.TButton"
    row, col = divmod(index, 2)
    span = 2 if (index == count - 1 and col == 0) else 1
    padx = (0, 0) if span == 2 else ((0, PAD_XS) if col == 0 else (PAD_XS, 0))
    btn = ttk.Button(
        actions,
        text=text,
        style=style,
        command=getattr(gui, cmd),
    )
    btn.grid(
        row=row,
        column=col,
        columnspan=span,
        sticky="ew",
        padx=padx,
        pady=(0, PAD_SM),
    )
    setattr(gui, attr, btn)
    gui._export_button_meta[attr] = help_line
    create_tooltip(btn, help_line)
    return btn


def _build_pose_backend_row(gui: Any, parent: tk.Misc, *, row: int) -> int:
    backend_row = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    backend_row.grid(row=row, column=0, sticky="ew", pady=(0, PAD_XS))
    tk.Label(
        backend_row,
        text="Pose backend",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(0, PAD_SM))
    if not hasattr(gui, "pose_backend_var"):
        gui.pose_backend_var = tk.StringVar(value=config.POSE_BACKEND)
    gui.pose_backend_combo = ttk.Combobox(
        backend_row,
        textvariable=gui.pose_backend_var,
        values=("mediapipe", "smpl", "auto"),
        state="readonly",
        width=14,
    )
    gui.pose_backend_combo.pack(side=tk.LEFT)
    gui.pose_backend_combo.bind(
        "<<ComboboxSelected>>",
        lambda _e: gui._refresh_pose_backend_status_label(),
    )
    gui.lbl_pose_backend_status = tk.Label(
        backend_row,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_pose_backend_status.pack(side=tk.LEFT, padx=(PAD_SM, 0))
    try:
        gui._refresh_pose_backend_status_label()
    except Exception:
        pass
    return row + 1


def build_advanced_workspace(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the Advanced technical workspace into ``parent``."""
    from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS
    from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1, uniform="advanced-column")
    host.columnconfigure(1, weight=1, uniform="advanced-column")
    for row in range(3):
        host.rowconfigure(row, weight=0)
    host.grid_anchor("n")
    gui._dashboard_data_row = host
    gui._section_data_export = host
    gui._advanced_workspace = host
    gui._advanced_category_hosts: dict[str, tk.Frame] = {}
    gui._advanced_category_bodies: dict[str, tk.Frame] = {}
    gui._export_button_meta = {}
    gui._advanced_two_col = True
    gui._advanced_caption_labels: dict[str, tk.Label] = {}
    gui._advanced_title_labels: dict[str, tk.Label] = {}
    gui._advanced_action_hosts: dict[str, tk.Frame] = {}
    gui._advanced_action_buttons: dict[str, list[ttk.Button]] = {}

    for _index, (key, title, expanded) in enumerate(ADVANCED_CATEGORIES):
        card, body = _category_card(
            host,
            title=title,
            expanded=expanded,
            caption=_CATEGORY_CAPTIONS.get(key, ""),
        )
        body.columnconfigure(0, weight=1)
        gui._advanced_category_hosts[key] = card
        gui._advanced_category_bodies[key] = body
        caption_lbl = getattr(card, "_sw_caption", None)
        if caption_lbl is not None:
            gui._advanced_caption_labels[key] = caption_lbl
        title_lbl = getattr(card, "_sw_title", None)
        if title_lbl is not None:
            gui._advanced_title_labels[key] = title_lbl

        next_row = 1 if _CATEGORY_CAPTIONS.get(key) else 0
        if key == "opensim":
            from stablewalk.ui.tk.dashboard_advanced import build_opensim_status_panel

            status_panel = build_opensim_status_panel(gui, body)
            status_panel.grid(row=next_row, column=0, sticky="ew", pady=(0, PAD_SM))
            next_row += 1
        if key == "exports":
            next_row = _build_pose_backend_row(gui, body, row=next_row)

        actions = list(_CATEGORY_ACTIONS.get(key, ()))
        if actions:
            action_host = tk.Frame(body, bg=PANEL, highlightthickness=0)
            action_host.grid(row=next_row, column=0, sticky="ew", pady=(0, PAD_XS))
            action_host.columnconfigure(0, weight=1)
            action_host.columnconfigure(1, weight=1)
            buttons: list[ttk.Button] = []
            for i, (attr, text, cmd, primary, help_line) in enumerate(actions):
                buttons.append(
                    _add_action_button(
                        gui,
                        action_host,
                        index=i,
                        count=len(actions),
                        attr=attr,
                        text=text,
                        cmd=cmd,
                        primary=primary,
                        help_line=help_line,
                    )
                )
            gui._advanced_action_hosts[key] = action_host
            gui._advanced_action_buttons[key] = buttons
            next_row += 1

        # Slot for lazy-mounted panels (generated files, RTS dashboard, etc.).
        slot = tk.Frame(body, bg=PANEL, highlightthickness=0)
        slot.grid(row=next_row, column=0, sticky="nsew")
        slot.columnconfigure(0, weight=1)
        body.rowconfigure(next_row, weight=0)
        gui._advanced_category_bodies[f"{key}_slot"] = slot

    _place_advanced_cards(host, gui._advanced_category_hosts, two_col=True)

    def _on_workspace_resize(event: tk.Event) -> None:
        try:
            reflow_advanced_workspace(gui, width=int(event.width))
        except (AttributeError, TypeError, ValueError):
            return

    host.bind("<Configure>", _on_workspace_resize, add="+")

    # Compatibility aliases for older export-oriented lazy-mount keys.
    exports_slot = gui._advanced_category_bodies.get("exports_slot")
    if exports_slot is not None:
        gui._advanced_category_bodies["developer_slot"] = exports_slot
        gui._advanced_category_bodies["research_export_slot"] = exports_slot

    # Compatibility aliases + disabled defaults (same as former Data & Export).
    gui.btn_view_table_data = gui.btn_view_detailed_data
    gui.btn_export_tracking = None
    gui.btn_clear_dof_history = None
    gui.btn_opensim_export = gui.btn_opensim_export_data
    for attr in (
        "btn_view_detailed_data",
        "btn_open_coord_inspector",
        "btn_save_session",
        "btn_export_analysis_report",
        "btn_export_pdf_report",
        "btn_export_gait_metrics",
        "btn_export_motion_reference",
        "btn_export_amp_reference",
        "btn_real_to_sim_pipeline",
        "btn_opensim_export_data",
    ):
        btn = getattr(gui, attr, None)
        if btn is not None:
            try:
                btn.configure(state=tk.DISABLED)
            except tk.TclError:
                pass

    # Hidden joint-tracking controls (logic retained).
    hidden = ttk.Frame(gui._sidebar_hidden)
    gui.add_point_var = tk.StringVar(value=ADD_POINT_PLACEHOLDER)
    gui.add_point_combo = ttk.Combobox(
        hidden,
        textvariable=gui.add_point_var,
        values=[GUI_DOF_LABELS[i] for i in GUI_DOF_ITEM_IDS],
        state="readonly",
    )
    gui.add_point_combo.bind("<<ComboboxSelected>>", gui._add_point_from_combo)
    gui.lbl_export_selected_joint = ttk.Label(hidden, text="Selected Joint:\n—")
    gui.lbl_selected_joint_data = gui.lbl_export_selected_joint
    gui.lbl_export_sample_count = ttk.Label(hidden, text="Samples:\n0")
    gui.lbl_data_sample_count = gui.lbl_export_sample_count
    gui.lbl_export_output_folder = ttk.Label(hidden, text="Output Folder:\ndata/output/")
    gui.lbl_export_output_status = gui.lbl_export_output_folder
    gui.lbl_export_opensim_sdk = ttk.Label(hidden, text="OpenSim SDK: —")
    gui.lbl_export_opensim_model = ttk.Label(hidden, text="Model: Not Loaded")
    gui._data_export_joint_host = hidden
    gui._dashboard_data_btns = host

    return host


def advanced_category_slot(gui: Any, key: str) -> tk.Frame | None:
    """Return the lazy-content slot for an Advanced category."""
    bodies = getattr(gui, "_advanced_category_bodies", None) or {}
    aliases = {"session": "session", "reports": "reports"}
    resolved = aliases.get(key, key)
    return bodies.get(f"{resolved}_slot") or bodies.get(resolved)


def _advanced_card_padx(column: int, span: int, *, two_col: bool) -> tuple[int, int]:
    """Symmetric gutter between the left and right technical columns."""
    if not two_col or int(span) >= 2:
        return (0, 0)
    half = PAD_SM // 2
    col = max(0, int(column))
    left = 0 if col <= 0 else half
    right = 0 if col >= 1 else half
    return (left, right)


def _place_advanced_cards(
    host: tk.Misc,
    hosts: dict[str, tk.Frame],
    *,
    two_col: bool,
) -> None:
    """Place technical groups in two columns, or one compact vertical stack."""
    for row in range(max(len(ADVANCED_CATEGORIES), 3)):
        host.rowconfigure(row, weight=0)
    if two_col:
        host.columnconfigure(
            0,
            weight=1,
            minsize=280,
            uniform="advanced-column",
        )
        host.columnconfigure(
            1,
            weight=1,
            minsize=280,
            uniform="advanced-column",
        )
        for key, row, column, span in _ADVANCED_TWO_COL_LAYOUT:
            card = hosts.get(key)
            if card is None:
                continue
            card.grid(
                row=row,
                column=column,
                columnspan=span,
                sticky="nsew",
                pady=(0, PAD_MD),
                padx=_advanced_card_padx(column, span, two_col=True),
            )
        return

    host.columnconfigure(0, weight=1, minsize=0, uniform="")
    host.columnconfigure(1, weight=0, minsize=0, uniform="")
    for index, (key, _title, _exp) in enumerate(ADVANCED_CATEGORIES):
        card = hosts.get(key)
        if card is None:
            continue
        card.grid(
            row=index,
            column=0,
            columnspan=1,
            sticky="ew",
            pady=(0, PAD_MD),
            padx=0,
        )


def _place_advanced_actions(gui: Any, *, columns: int) -> None:
    """Keep category actions readable: two columns when wide, one when stacked."""
    hosts = getattr(gui, "_advanced_action_hosts", None) or {}
    buttons = getattr(gui, "_advanced_action_buttons", None) or {}
    cols = 1 if int(columns) <= 1 else 2
    for key, action_host in hosts.items():
        items = buttons.get(key) or []
        if not items:
            continue
        try:
            action_host.columnconfigure(0, weight=1)
            action_host.columnconfigure(1, weight=1 if cols == 2 else 0)
        except tk.TclError:
            continue
        count = len(items)
        for index, btn in enumerate(items):
            if cols == 1:
                row, col, span = index, 0, 2
                padx = (0, 0)
            else:
                row, col = divmod(index, 2)
                span = 2 if (index == count - 1 and col == 0) else 1
                padx = (0, 0) if span == 2 else ((0, PAD_XS) if col == 0 else (PAD_XS, 0))
            try:
                btn.grid(
                    row=row,
                    column=col,
                    columnspan=span,
                    sticky="ew",
                    padx=padx,
                    pady=(0, PAD_SM),
                )
            except tk.TclError:
                pass


def _sync_advanced_wraplengths(gui: Any, width: int, *, two_col: bool) -> None:
    """Fit captions and intro copy to the current column width."""
    col_w = max(160, (int(width) // 2) - 24) if two_col else max(180, int(width) - 32)
    wrap = max(160, min(560, col_w))
    labels = getattr(gui, "_advanced_caption_labels", None) or {}
    for lbl in labels.values():
        try:
            lbl.configure(wraplength=wrap)
        except tk.TclError:
            pass
    titles = getattr(gui, "_advanced_title_labels", None) or {}
    title_wrap = max(120, wrap - 24)
    for lbl in titles.values():
        try:
            lbl.configure(wraplength=title_wrap)
        except tk.TclError:
            pass
    intro = getattr(gui, "lbl_gait_metrics_intro", None)
    if intro is not None:
        try:
            intro.configure(wraplength=wrap)
        except tk.TclError:
            pass
    files_hint = getattr(gui, "lbl_generated_files_actions_hint", None)
    if files_hint is not None:
        try:
            files_hint.configure(wraplength=wrap)
        except tk.TclError:
            pass
    files_empty = getattr(gui, "lbl_generated_files_empty", None)
    if files_empty is not None:
        try:
            files_empty.configure(wraplength=wrap)
        except tk.TclError:
            pass
    empty_state = getattr(gui, "_advanced_empty_state", None)
    if empty_state is not None:
        try:
            empty_state.configure(wraplength=max(220, min(920, int(width) - 48)))
        except tk.TclError:
            pass


def reflow_advanced_workspace(gui: Any, *, width: int) -> None:
    """Keep technical groups balanced when wide; stack before they compress."""
    host = getattr(gui, "_advanced_workspace", None)
    hosts = getattr(gui, "_advanced_category_hosts", None) or {}
    if host is None or not hosts:
        return
    try:
        host_width = int(host.winfo_width() or 0)
        if host_width > 40:
            width = host_width
    except tk.TclError:
        pass
    width = int(width or 0)
    # Unmeasured first-map widths must not collapse the default two-column layout.
    if width < 80:
        return
    from stablewalk.ui.tk.layout_scale import layout_scale_from_gui, scaled_breakpoints

    scale = layout_scale_from_gui(gui)
    two_min = ADVANCED_TWO_COL_MIN
    if abs(scale - 1.0) >= 0.08:
        two_min = int(scaled_breakpoints(scale)["advanced_two_col"])
    two = width >= two_min
    if getattr(gui, "_advanced_two_col", None) is not two:
        gui._advanced_two_col = two
        try:
            _place_advanced_cards(host, hosts, two_col=two)
            _place_advanced_actions(gui, columns=2 if two else 1)
        except tk.TclError:
            pass
    _sync_advanced_wraplengths(gui, width, two_col=two)
    try:
        from stablewalk.ui.tk.generated_files_panel import reflow_generated_files_panel

        col_w = max(160, (int(width) // 2) - 24) if two else max(180, int(width) - 32)
        reflow_generated_files_panel(gui, width=col_w)
    except Exception:
        pass
    sync = getattr(gui, "_sync_tab_advanced_scroll", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            pass


__all__ = [
    "ADVANCED_CATEGORIES",
    "ADVANCED_TWO_COL_MIN",
    "advanced_category_slot",
    "build_advanced_workspace",
    "reflow_advanced_workspace",
]
