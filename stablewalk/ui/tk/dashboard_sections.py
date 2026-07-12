"""
Four-section dashboard layout helpers for StableWalk.

Section 1 — Overview (video | 3D | summary)
Section 2 — Motion Analysis (knee | joint path)
Section 3 — Gait Metrics (compact cards)
Section 4 — Data & Export
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.dashboard_interpretability import METRIC_HELP
from stablewalk.ui.tk.metric_help import add_metric_help_icon
from stablewalk.ui.theme import (
    BORDER,
    DASHBOARD_GUTTER,
    ELEVATED,
    FONT_METRIC,
    FONT_METRIC_VALUE_ACCENT,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_XS,
    PANEL,
    TEXT,
)

# Section 1 column weights (47% / 33% / 20%)
SEC1_VIDEO_WEIGHT = 47
SEC1_SKELETON_WEIGHT = 33
SEC1_SUMMARY_WEIGHT = 20

SECTION_VISUAL_TITLE = "Overview"
SECTION_KINEMATIC_TITLE = "Motion Analysis"
SECTION_GAIT_METRICS_TITLE = "Gait Metrics"
SECTION_DATA_EXPORT_TITLE = "Data & Export"


def _section_label_frame(parent: tk.Misc, title: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(
        parent,
        text=f"  {title}  ",
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )


def _gait_metric_card(parent: tk.Misc, *, title: str, help_key: str | None = None) -> tk.Frame:
    """Compact card shell for Section 3."""
    cell = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    cell.columnconfigure(0, weight=1)

    header = tk.Frame(cell, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 0))
    tk.Label(
        header,
        text=title.upper(),
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(side=tk.LEFT)
    if help_key and help_key in METRIC_HELP:
        help_host = tk.Frame(header, bg=ELEVATED)
        help_host.pack(side=tk.LEFT, padx=(4, 0))
        add_metric_help_icon(help_host, title, METRIC_HELP[help_key])
    return cell


def build_overview_metrics_row(gui, parent: tk.Misc) -> tk.Frame:
    """Overview tab bottom strip: gait phase, contact pattern, gait cycles."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)
    host.columnconfigure(1, weight=1)
    host.columnconfigure(2, weight=1)

    phase_card = _gait_metric_card(host, title="Gait Phase")
    phase_card.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=(PAD_XS, 0))
    phase_body = tk.Frame(phase_card, bg=ELEVATED)
    phase_body.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
    tk.Label(
        phase_body,
        text="Current:",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(anchor="w")
    gui.lbl_gait_card_phase_label = None
    gui.lbl_gait_card_phase_value = tk.Label(
        phase_body,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_gait_card_phase_value.pack(anchor="w")

    contact_card = _gait_metric_card(host, title="Contact Pattern")
    contact_card.grid(row=0, column=1, sticky="nsew", padx=(0, PAD_XS), pady=(PAD_XS, 0))
    contact_body = tk.Frame(contact_card, bg=ELEVATED)
    contact_body.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
    for side_key, side_label in (("left", "Left"), ("right", "Right")):
        row = tk.Frame(contact_body, bg=ELEVATED)
        row.pack(anchor="w", fill="x", pady=1)
        tk.Label(
            row,
            text=f"{side_label}:",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(side=tk.LEFT)
        val = tk.Label(row, text="—", bg=ELEVATED, fg=TEXT, font=FONT_UI_SM, anchor="w")
        val.pack(side=tk.LEFT, padx=(4, 0))
        setattr(gui, f"lbl_overview_contact_{side_key}", val)

    cycles_card = _gait_metric_card(host, title="Gait Cycles")
    cycles_card.grid(row=0, column=2, sticky="nsew", pady=(PAD_XS, 0))
    cycles_body = tk.Frame(cycles_card, bg=ELEVATED)
    cycles_body.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
    gui.lbl_overview_gait_cycles_usable = tk.Label(
        cycles_body,
        text="Usable: —",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_overview_gait_cycles_usable.pack(anchor="w")
    gui.lbl_overview_gait_cycles_completeness = tk.Label(
        cycles_body,
        text="Completeness: —",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_overview_gait_cycles_completeness.pack(anchor="w", pady=(2, 0))
    gui.lbl_overview_gait_cycles = gui.lbl_overview_gait_cycles_usable

    gui._overview_metrics_row = host
    return host


def build_gait_metrics_section(gui, parent: tk.Misc) -> ttk.LabelFrame:
    """Advanced tab — remaining gait metric cards (contact, cycles)."""
    section = _section_label_frame(parent, SECTION_GAIT_METRICS_TITLE)
    for col in range(2):
        section.columnconfigure(col, weight=1, uniform="gait_metric")

    grid_host = tk.Frame(section, bg=PANEL, highlightthickness=0)
    grid_host.grid(row=0, column=0, columnspan=2, sticky="ew")
    for col in range(2):
        grid_host.columnconfigure(col, weight=1, uniform="gait_metric")

    # Foot clearance + phase/contact/cycles live on Overview tab (build_overview_metrics_row).
    contact_card = _gait_metric_card(grid_host, title="Contact State")
    contact_card.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=(0, PAD_XS))
    contact_body = tk.Frame(contact_card, bg=ELEVATED)
    contact_body.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 6))
    for row, side in enumerate(("Left Foot", "Right Foot")):
        tk.Label(
            contact_body, text=f"{side}:", bg=ELEVATED, fg=MUTED, font=FONT_UI_XS, anchor="w"
        ).grid(row=row, column=0, sticky="w", pady=1)
        val = tk.Label(contact_body, text="—", bg=ELEVATED, fg=TEXT, font=FONT_UI_SM, anchor="w")
        val.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=1)
        attr = "left" if "Left" in side else "right"
        setattr(gui, f"lbl_gait_card_contact_{attr}", val)

    # Gait Cycles
    cycles_card = _gait_metric_card(grid_host, title="Gait Cycles")
    cycles_card.grid(row=0, column=1, sticky="nsew", pady=(0, PAD_XS))
    gui.lbl_gait_card_cycles_label = tk.Label(
        cycles_card,
        text="Usable cycles:",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_card_cycles_label.grid(row=1, column=0, sticky="ew", padx=6, pady=(2, 0))
    gui.lbl_gait_card_cycles_value = tk.Label(
        cycles_card,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_gait_card_cycles_value.grid(row=2, column=0, sticky="ew", padx=6, pady=(0, 2))
    gui.lbl_gait_card_completeness_value = tk.Label(
        cycles_card,
        text="Completeness: —",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_card_completeness_value.grid(row=3, column=0, sticky="ew", padx=6, pady=(0, 6))

    btn_row = tk.Frame(section, bg=PANEL, highlightthickness=0)
    btn_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(PAD_XS, 0))
    gui.btn_advanced_analysis = ttk.Button(
        btn_row,
        text="Advanced Analysis",
        style="Compact.TButton",
        command=gui._show_advanced_analysis,
    )
    gui.btn_advanced_analysis.pack(side=tk.LEFT)
    gui.btn_gait_metrics_details = gui.btn_advanced_analysis

    gui._section_gait_metrics = section
    return section


def _summary_score_card(parent: tk.Misc, *, title: str, help_key: str | None) -> tuple[tk.Frame, tk.Label]:
    card = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    card.pack(fill=tk.X, pady=(0, PAD_XS))
    header = tk.Frame(card, bg=ELEVATED, highlightthickness=0)
    header.pack(fill=tk.X, padx=6, pady=(4, 0))
    title_lbl = tk.Label(
        header,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    title_lbl.pack(side=tk.LEFT)
    if help_key and help_key in METRIC_HELP:
        from stablewalk.ui.tk.metric_help import bind_metric_help_tooltip

        bind_metric_help_tooltip(title_lbl, title, METRIC_HELP[help_key])
    return card, title_lbl


def build_gait_summary_cards(gui, parent: tk.Misc) -> tk.Frame:
    """Overview sidebar — three primary scores with one-line explanations."""
    from stablewalk.ui.theme import FONT_UI_SM, WARNING

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.pack(fill=tk.X)

    ms_card, _ = _summary_score_card(host, title="Movement Stability", help_key="movement_stability")
    gui.lbl_summary_ms_value = tk.Label(
        ms_card,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_METRIC_VALUE_ACCENT,
        anchor="w",
    )
    gui.lbl_summary_ms_value.pack(fill=tk.X, padx=6, pady=(2, 0))
    gui.lbl_summary_ms_explain = tk.Label(
        ms_card,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=220,
        justify=tk.LEFT,
    )
    gui.lbl_summary_ms_explain.pack(fill=tk.X, padx=6, pady=(0, 6))

    gq_card, _ = _summary_score_card(host, title="Gait Quality", help_key="gait_quality")
    gq_value_row = tk.Frame(gq_card, bg=ELEVATED, highlightthickness=0)
    gq_value_row.pack(fill=tk.X, padx=6, pady=(2, 0))
    gui.lbl_summary_gq_value = tk.Label(
        gq_value_row,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_METRIC_VALUE_ACCENT,
        anchor="w",
    )
    gui.lbl_summary_gq_value.pack(side=tk.LEFT)
    gui.lbl_summary_gq_badge = tk.Label(
        gq_value_row,
        text="",
        bg=ELEVATED,
        fg=WARNING,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_summary_gq_badge.pack(side=tk.LEFT, padx=(6, 0))
    gui.lbl_summary_gq_explain = tk.Label(
        gq_card,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=220,
        justify=tk.LEFT,
    )
    gui.lbl_summary_gq_explain.pack(fill=tk.X, padx=6, pady=(0, 6))

    ac_card, _ = _summary_score_card(host, title="Analysis Confidence", help_key="analysis_confidence")
    gui.lbl_summary_ac_level = tk.Label(
        ac_card,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_METRIC_VALUE_ACCENT,
        anchor="w",
    )
    gui.lbl_summary_ac_level.pack(fill=tk.X, padx=6, pady=(2, 0))
    gui.lbl_summary_ac_explain = tk.Label(
        ac_card,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
        wraplength=220,
        justify=tk.LEFT,
    )
    gui.lbl_summary_ac_explain.pack(fill=tk.X, padx=6, pady=(0, 6))

    gui.btn_walk_summary_details = ttk.Button(
        host,
        text="Details",
        style="Compact.TButton",
        command=gui._show_gait_summary_details,
    )
    gui.btn_walk_summary_details.pack(anchor=tk.W, pady=(PAD_XS, 0))

    gui.lbl_summary_ms_interp = gui.lbl_summary_ms_explain
    gui.lbl_summary_gq_interp = gui.lbl_summary_gq_explain
    gui.lbl_summary_ac_interp = gui.lbl_summary_ac_explain
    gui.lbl_summary_ms_subtitle = None
    gui.lbl_summary_ms_definition = None
    gui.lbl_summary_gq_subtitle = None
    gui.lbl_summary_gq_definition = None
    gui.lbl_summary_ac_subtitle = None
    gui.lbl_summary_ac_completeness = None
    gui.lbl_summary_ac_cycles = None
    gui.lbl_summary_ac_comparable = None
    gui.lbl_summary_important_note = None
    gui.lbl_summary_current_phase = None
    gui.lbl_summary_usable_cycles = None

    gui.lbl_movement_stability = gui.lbl_summary_ms_value
    gui.lbl_gait_quality = gui.lbl_summary_gq_value
    gui.lbl_analysis_confidence = gui.lbl_summary_ac_level
    gui.lbl_movement_stability_interp = gui.lbl_summary_ms_explain
    gui.lbl_gait_quality_interp = gui.lbl_summary_gq_explain
    gui.lbl_analysis_confidence_interp = gui.lbl_summary_ac_explain
    gui.lbl_stab_completeness = getattr(gui, "lbl_overview_gait_cycles_completeness", None)
    gui.lbl_stab_usable_cycles = getattr(gui, "lbl_overview_gait_cycles_usable", None)
    gui.lbl_stab_comparable = None

    gui._gait_summary_cards_host = host
    return host


def build_foot_clearance_detail_panel(gui, parent: tk.Misc) -> tk.Frame:
    """Primary Overview foot-to-floor distance panel below 3D gait reconstruction."""
    from stablewalk.ui.foot_clearance_display import (
        OVERVIEW_DISTANCE_CAPTION,
        OVERVIEW_SECTION_TITLE,
    )

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    tk.Label(
        host,
        text=OVERVIEW_SECTION_TITLE,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", pady=(4, 2))

    gui.lbl_foot_clearance_confidence = tk.Label(
        host,
        text="Measurement Confidence: —",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_foot_clearance_confidence.grid(row=1, column=0, sticky="ew", pady=(0, 6))

    feet_row = tk.Frame(host, bg=PANEL, highlightthickness=0)
    feet_row.grid(row=2, column=0, sticky="ew")
    feet_row.columnconfigure(0, weight=1)
    feet_row.columnconfigure(1, weight=1)

    _FOOT_DISTANCE_FONT = ("Segoe UI Semibold", 15)

    def _foot_card(column: int, prefix: str, title: str) -> tk.Frame:
        card = tk.Frame(
            feet_row,
            bg=ELEVATED,
            highlightthickness=1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
        )
        card.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0 if column == 0 else 3, 3 if column == 0 else 0),
        )
        card.columnconfigure(0, weight=1)
        tk.Label(
            card,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 0))
        tk.Label(
            card,
            text=OVERVIEW_DISTANCE_CAPTION,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        setattr(
            gui,
            f"lbl_{prefix}_current",
            tk.Label(
                card,
                text="Unavailable",
                bg=ELEVATED,
                fg=MUTED,
                font=_FOOT_DISTANCE_FONT,
                anchor="w",
            ),
        )
        getattr(gui, f"lbl_{prefix}_current").grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 0))
        setattr(
            gui,
            f"lbl_{prefix}_state",
            tk.Label(
                card,
                text="—",
                bg=ELEVATED,
                fg=TEXT,
                font=FONT_METRIC_VALUE_ACCENT,
                anchor="w",
            ),
        )
        getattr(gui, f"lbl_{prefix}_state").grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 0))
        setattr(
            gui,
            f"lbl_{prefix}_unavailable",
            tk.Label(
                card,
                text="",
                bg=ELEVATED,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
                justify=tk.LEFT,
                wraplength=160,
            ),
        )
        getattr(gui, f"lbl_{prefix}_unavailable").grid(row=4, column=0, sticky="ew", padx=8, pady=(2, 8))
        return card

    _foot_card(0, "foot_left", "LEFT FOOT")
    _foot_card(1, "foot_right", "RIGHT FOOT")

    btn_row = tk.Frame(host, bg=PANEL, highlightthickness=0)
    btn_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
    gui.btn_foot_clearance_details = ttk.Button(
        btn_row,
        text="Details",
        style="Compact.TButton",
        command=gui._show_foot_clearance_details,
    )
    gui.btn_foot_clearance_details.pack(side=tk.LEFT)

    gui._foot_clearance_detail_host = host
    return host


def build_data_export_section(gui, parent: tk.Misc) -> ttk.LabelFrame:
    """Section 4 — streamlined export controls."""
    from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS
    from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER

    section = _section_label_frame(parent, SECTION_DATA_EXPORT_TITLE)
    section.columnconfigure(0, weight=1)
    gui._dashboard_data_row = section
    gui._section_data_export = section

    btn_host = ttk.Frame(section)
    btn_host.grid(row=0, column=0, sticky="ew")
    btn_host.columnconfigure(0, weight=1)
    btn_host.columnconfigure(1, weight=1)
    btn_host.columnconfigure(2, weight=1)

    buttons = (
        ("btn_view_detailed_data", "View Joint Data", "_toggle_collected_data_table", False),
        ("btn_save_session", "Save Analysis", "_save_session_to_files", True),
        ("btn_export_analysis_report", "Export Analysis Report", "_export_analysis_report", False),
        ("btn_export_gait_metrics", "Export Gait Metrics", "_export_gait_metrics", False),
        ("btn_opensim_export_data", "Export OpenSim Files", "_export_opensim_session", True),
        ("btn_export_motion_reference", "Export Motion Reference", "_export_motion_reference", False),
    )
    for index, (attr, text, cmd, accent) in enumerate(buttons):
        style = "ExportAccent.TButton" if accent else "Export.TButton"
        btn = ttk.Button(btn_host, text=text, style=style, command=getattr(gui, cmd))
        setattr(gui, attr, btn)
        btn.grid(row=index // 3, column=index % 3, sticky="ew", padx=(0, PAD_XS), pady=(0, PAD_XS))

    gui.btn_view_table_data = gui.btn_view_detailed_data
    gui.btn_export_tracking = None
    gui.btn_clear_dof_history = None
    gui.btn_view_detailed_data.configure(state=tk.DISABLED)
    gui.btn_save_session.configure(state=tk.DISABLED)
    gui.btn_export_analysis_report.configure(state=tk.DISABLED)
    gui.btn_export_gait_metrics.configure(state=tk.DISABLED)
    gui.btn_opensim_export = gui.btn_opensim_export_data

    # Hidden joint-tracking controls (logic retained; not shown on main dashboard).
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
    gui._dashboard_data_btns = btn_host

    return section


def bind_panel_word_wrap(gui, widget: tk.Misc, attr_names: tuple[str, ...]) -> None:
    """Keep summary labels wrapped to the panel width."""

    def _sync(_event: object | None = None) -> None:
        width = widget.winfo_width()
        if width < 40:
            return
        wrap = max(72, width - 16)
        for attr in attr_names:
            lbl = getattr(gui, attr, None)
            if lbl is not None and hasattr(lbl, "configure"):
                try:
                    lbl.configure(wraplength=wrap)
                except tk.TclError:
                    pass

    widget.bind("<Configure>", _sync, add="+")
    gui._summary_wrap_targets = getattr(gui, "_summary_wrap_targets", ())
    gui._summary_wrap_targets = (*gui._summary_wrap_targets, *attr_names)
