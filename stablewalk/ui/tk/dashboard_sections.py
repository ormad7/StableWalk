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

from stablewalk import config
from stablewalk.ui.dashboard_interpretability import METRIC_HELP
from stablewalk.ui.scientific_labels import (
    LABEL_CONTACT_PATTERN,
    LABEL_CONTACT_STATE,
    LABEL_FOOT_CLEARANCE_CONFIDENCE,
)
from stablewalk.ui.tk.metric_help import add_metric_help_icon
from stablewalk.ui.theme import (
    ACCENT,
    ACCENT_ALT,
    BORDER,
    CARD_BODY_PAD,
    CARD_HEADER_PAD,
    CARD_INNER_PAD_X,
    DASHBOARD_CARD_PAD,
    DASHBOARD_GUTTER,
    ELEVATED,
    FONT_DISPLAY,
    FONT_METRIC,
    FONT_METRIC_VALUE_ACCENT,
    FONT_PANEL_HEADER,
    FONT_UI_SM,
    FONT_UI_XS,
    HUD_BG,
    HUD_BORDER,
    MUTED,
    ORANGE,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    WARNING,
    bind_responsive_wrap,
    create_elevated_card,
    responsive_wraplength,
)

# Section 1 — default Side-by-Side always uses three equal-height columns:
# Original Video | 3D Gait Reconstruction | Selected Joint 3D Path.
SEC1_VIDEO_WEIGHT = 34
SEC1_SKELETON_WEIGHT = 36
SEC1_SUMMARY_WEIGHT = 0

# Alias weights used when the path column is mapped (same as the default).
SEC1_TRAJ_VIDEO_WEIGHT = 34
SEC1_TRAJ_SKELETON_WEIGHT = 36
SEC1_TRAJ_PATH_WEIGHT = 30
SEC1_TRAJ_SUMMARY_WEIGHT = 0

# Minimum panel widths so axes / skeleton stay usable when the window shrinks.
SEC1_PANEL_MINWIDTH = 220
SEC1_TRAJ_PANEL_MINWIDTH = 280

# Overview tab vertical split: visualizations dominate, gait cards stay compact.
SEC1_VIZ_ROW_WEIGHT = 78
SEC1_METRICS_ROW_WEIGHT = 22
# Joint Graphs bottom strip — collapsed by default (weight 0); expands on demand.
SEC1_JOINT_MOTION_ROW_WEIGHT = 14
SEC1_JOINT_MOTION_ROW_MINSIZE = 28

# Soft minimums for the Overview visualization row. Keep these modest so medium
# windows still get a weight-driven ~75% viz share without Tk minsize overflow
# (which clipped the 3D path canvas when minsize > available height).
SEC1_VIZ_ROW_MINSIZE = 300
SEC1_TRAJ_VIZ_ROW_MINSIZE = 320
SEC1_METRICS_ROW_MINSIZE = 28
SEC1_METRICS_ROW_MAXSIZE = 96


def apply_overview_joint_motion_row_weight(
    parent: tk.Misc,
    *,
    expanded: bool,
    row: int = 2,
) -> None:
    """Give Joint Graphs height only when expanded; otherwise header-only strip."""
    try:
        if expanded:
            parent.rowconfigure(
                row,
                weight=SEC1_JOINT_MOTION_ROW_WEIGHT,
                minsize=SEC1_JOINT_MOTION_ROW_MINSIZE,
            )
        else:
            parent.rowconfigure(
                row,
                weight=0,
                minsize=SEC1_JOINT_MOTION_ROW_MINSIZE,
            )
    except tk.TclError:
        pass


def overview_panel_padx(column: int) -> tuple[int, int]:
    """Symmetric horizontal gutter for Overview visual panels.

    Adjacent panels share ``OVERVIEW_COL_GUTTER`` equally. Column 1 receives
    both-side padding when the 3D path column is present.
    """
    from stablewalk.ui.theme import OVERVIEW_COL_GUTTER

    half = OVERVIEW_COL_GUTTER // 2
    if column <= 0:
        return (0, half)
    if column == 1:
        return (half, half)
    return (half, 0)


def overview_metric_padx(column: int) -> tuple[int, int]:
    """Symmetric horizontal gutter for the Overview bottom metric cards."""
    from stablewalk.ui.theme import OVERVIEW_COL_GUTTER

    half = OVERVIEW_COL_GUTTER // 2
    if column <= 0:
        return (0, half)
    if column == 1:
        return (half, half)
    return (half, 0)

SECTION_VISUAL_TITLE = "Overview"
SECTION_KINEMATIC_TITLE = "Motion Analysis"
SECTION_GAIT_METRICS_TITLE = "Gait Metrics"
SECTION_REAL_TO_SIM_TITLE = "Real-to-Sim Pipeline"
SECTION_DATA_EXPORT_TITLE = "Data & Export"


def _section_label_frame(parent: tk.Misc, title: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(
        parent,
        text=f"  {title}  ",
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )


def _gait_metric_card(parent: tk.Misc, *, title: str, help_key: str | None = None) -> tk.Frame:
    """Compact card shell for Section 3."""
    cell = create_elevated_card(parent)
    cell.columnconfigure(0, weight=1)

    header = tk.Frame(cell, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=CARD_HEADER_PAD[0], pady=CARD_HEADER_PAD[1])
    tk.Label(
        header,
        text=title.upper(),
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    ).pack(side=tk.LEFT)
    if help_key and help_key in METRIC_HELP:
        help_host = tk.Frame(header, bg=ELEVATED)
        help_host.pack(side=tk.LEFT, padx=(4, 0))
        add_metric_help_icon(help_host, title, METRIC_HELP[help_key])
    return cell


# Floating playback HUD (Overview) — fields shown left→right.
_OVERVIEW_HUD_FIELDS: tuple[tuple[str, str], ...] = (
    ("frame", "FRAME"),
    ("time", "TIME"),
    ("joint", "JOINT"),
    ("phase", "PHASE"),
    ("speed", "SPEED"),
    ("confidence", "CONFIDENCE"),
    ("rom", "ROM"),
)

# Tk cannot alpha-blend embedded widgets, so a very dark "glass" fill is used to
# read as a translucent HUD strip when floated over the bright video frame.
_OVERVIEW_HUD_BG = HUD_BG
_OVERVIEW_HUD_BORDER = HUD_BORDER


def build_overview_playback_hud(gui, host: tk.Misc) -> tk.Frame:
    """A compact playback info bar floated over the video (2 rows, no clipping)."""
    hud = tk.Frame(
        host,
        bg=_OVERVIEW_HUD_BG,
        highlightthickness=1,
        highlightbackground=_OVERVIEW_HUD_BORDER,
        highlightcolor=_OVERVIEW_HUD_BORDER,
    )
    # Two rows so long values (joint name, speed) are not truncated.
    row_a = (
        ("frame", "FRAME"),
        ("time", "TIME"),
        ("joint", "JOINT"),
        ("phase", "PHASE"),
    )
    row_b = (
        ("speed", "SPEED"),
        ("confidence", "CONF"),
        ("rom", "ROM"),
    )
    for r in (0, 1):
        hud.rowconfigure(r, weight=1)
    for col in range(max(len(row_a), len(row_b))):
        hud.columnconfigure(col, weight=1)

    gui._overview_hud_value_labels = {}
    for row_i, fields in enumerate((row_a, row_b)):
        for col, (key, caption) in enumerate(fields):
            cell = tk.Frame(hud, bg=_OVERVIEW_HUD_BG, highlightthickness=0)
            cell.grid(
                row=row_i,
                column=col,
                sticky="nsew",
                padx=(8 if col == 0 else 6, 6),
                pady=(4 if row_i == 0 else 2, 4 if row_i == 1 else 2),
            )
            tk.Label(
                cell,
                text=caption,
                bg=_OVERVIEW_HUD_BG,
                fg=MUTED,
                font=FONT_UI_XS,
                anchor="w",
            ).pack(fill=tk.X)
            value_lbl = tk.Label(
                cell,
                text="\u2014",
                bg=_OVERVIEW_HUD_BG,
                fg=TEXT,
                font=(FONT_UI_SM[0], max(FONT_UI_SM[1], 9), "bold"),
                anchor="w",
                justify=tk.LEFT,
            )
            value_lbl.pack(fill=tk.X)
            gui._overview_hud_value_labels[key] = value_lbl

    gui._overview_playback_hud = hud
    return hud


def build_overview_metrics_row(gui, parent: tk.Misc) -> tk.Frame:
    """Compact Overview gait strip (~20–25% height): phase, contact, cycles."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(1, weight=1)

    header = tk.Frame(host, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(0, weight=1)

    gui.lbl_overview_demo_compare = tk.Label(
        header,
        text="",
        bg=PANEL,
        fg=ACCENT,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
    )
    gui.lbl_overview_demo_compare.grid(row=0, column=0, sticky="ew", padx=(PAD_XS, 0))

    # Start collapsed so the three visualization panels own ~75–80%+ of height.
    gui._overview_metrics_expanded = False

    def _apply_metrics_row_weight(expanded: bool) -> None:
        try:
            if expanded:
                parent.rowconfigure(
                    3, weight=SEC1_METRICS_ROW_WEIGHT, minsize=SEC1_METRICS_ROW_MINSIZE
                )
            else:
                parent.rowconfigure(3, weight=0, minsize=SEC1_METRICS_ROW_MINSIZE)
        except tk.TclError:
            pass

    def _toggle_metrics() -> None:
        expanded = not bool(getattr(gui, "_overview_metrics_expanded", False))
        gui._overview_metrics_expanded = expanded
        cards = getattr(gui, "_overview_metrics_cards", None)
        btn = getattr(gui, "_btn_overview_metrics_toggle", None)
        if cards is not None:
            if expanded:
                cards.grid()
            else:
                cards.grid_remove()
        if btn is not None:
            btn.configure(text="▼ Gait info" if expanded else "▶ Gait info")
        _apply_metrics_row_weight(expanded)
        try:
            parent.update_idletasks()
        except tk.TclError:
            pass
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
                pass

    toggle = ttk.Button(
        header,
        text="▶ Gait info",
        style="Compact.TButton",
        width=12,
        command=_toggle_metrics,
        takefocus=False,
    )
    toggle.grid(row=0, column=1, sticky="e", padx=(PAD_XS, 0))
    gui._btn_overview_metrics_toggle = toggle

    cards = tk.Frame(host, bg=PANEL, highlightthickness=0)
    # Collapsed by default — visualizations get priority.
    cards.columnconfigure(0, weight=1)
    cards.columnconfigure(1, weight=1)
    cards.columnconfigure(2, weight=1)
    gui._overview_metrics_cards = cards
    _apply_metrics_row_weight(False)

    phase_card = _gait_metric_card(cards, title="Gait Phase")
    phase_card.grid(row=0, column=0, sticky="nsew", padx=overview_metric_padx(0), pady=0)
    phase_body = tk.Frame(phase_card, bg=ELEVATED)
    phase_body.grid(
        row=1, column=0, sticky="ew", padx=(PAD_XS, PAD_XS), pady=(2, PAD_XS)
    )
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

    contact_card = _gait_metric_card(cards, title=LABEL_CONTACT_PATTERN)
    contact_card.grid(row=0, column=1, sticky="nsew", padx=overview_metric_padx(1), pady=0)
    contact_body = tk.Frame(contact_card, bg=ELEVATED)
    contact_body.grid(
        row=1, column=0, sticky="ew", padx=(PAD_XS, PAD_XS), pady=(2, PAD_XS)
    )
    for side_key, side_label in (("left", "Left"), ("right", "Right")):
        row = tk.Frame(contact_body, bg=ELEVATED)
        row.pack(anchor="w", fill="x")
        tk.Label(
            row,
            text=f"{side_label}:",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(side=tk.LEFT)
        val = tk.Label(row, text="—", bg=ELEVATED, fg=TEXT, font=FONT_UI_XS, anchor="w")
        val.pack(side=tk.LEFT, padx=(4, 0))
        setattr(gui, f"lbl_overview_contact_{side_key}", val)

    cycles_card = _gait_metric_card(cards, title="Gait Cycles")
    cycles_card.grid(row=0, column=2, sticky="nsew", padx=overview_metric_padx(2), pady=0)
    cycles_body = tk.Frame(cycles_card, bg=ELEVATED)
    cycles_body.grid(
        row=1, column=0, sticky="ew", padx=(PAD_XS, PAD_XS), pady=(2, PAD_XS)
    )
    gui.lbl_overview_gait_cycles_usable = tk.Label(
        cycles_body,
        text="Usable: —",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_XS,
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
    gui.lbl_overview_gait_cycles_completeness.pack(anchor="w")
    gui.lbl_overview_gait_cycles = gui.lbl_overview_gait_cycles_usable

    gui._overview_metrics_row = host
    bind_responsive_wrap(gui, host, ("lbl_overview_demo_compare",))
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
    contact_card = _gait_metric_card(grid_host, title=LABEL_CONTACT_STATE)
    contact_card.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=(0, PAD_XS))
    contact_body = tk.Frame(contact_card, bg=ELEVATED)
    contact_body.grid(row=1, column=0, sticky="ew", padx=CARD_INNER_PAD_X, pady=(PAD_XS, PAD_SM))
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


def build_real_to_sim_section(gui, parent: tk.Misc) -> ttk.LabelFrame:
    """Advanced tab — engineering pipeline dashboard (legacy alias)."""
    from stablewalk.ui.tk.dashboard_advanced import build_engineering_dashboard

    return build_engineering_dashboard(gui, parent)


def build_gait_summary_cards(gui, parent: tk.Misc) -> tk.Frame:
    """Overview sidebar — three primary KPI scores with one-line explanations."""
    from stablewalk.ui.theme import WARNING
    from stablewalk.ui.tk.kpi_cards import create_kpi_card, set_kpi_tooltip

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.pack(fill=tk.X)

    gui.lbl_summary_demo_headline = tk.Label(
        host,
        text="",
        bg=PANEL,
        fg=ORANGE,
        font=FONT_METRIC,
        anchor="w",
    )
    gui.lbl_summary_demo_headline.pack(fill=tk.X, padx=PAD_XS, pady=(0, PAD_XS))

    gui._overview_kpi_cards = {}

    def _score_kpi(
        *,
        key: str,
        title: str,
        help_key: str,
        show_bar: bool = True,
    ):
        tip = METRIC_HELP.get(help_key, title)
        kpi = create_kpi_card(
            host,
            key=key,
            title=title,
            tooltip=tip,
            compact=False,
            show_bar=show_bar,
            fill=False,
        )
        gui._overview_kpi_cards[key] = kpi
        explain = tk.Label(
            host,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="w",
            wraplength=220,
            justify=tk.LEFT,
        )
        explain.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_SM))
        return kpi, explain

    ms_kpi, gui.lbl_summary_ms_explain = _score_kpi(
        key="movement_stability",
        title="Movement Stability",
        help_key="movement_stability",
    )
    gui.lbl_summary_ms_value = ms_kpi.value_lbl
    gui.lbl_summary_ms_unit = ms_kpi.unit_lbl

    gq_kpi, gui.lbl_summary_gq_explain = _score_kpi(
        key="gait_quality",
        title="Gait Coordination (derived)",
        help_key="gait_quality",
    )
    gui.lbl_summary_gq_value = gq_kpi.value_lbl
    gui.lbl_summary_gq_unit = gq_kpi.unit_lbl
    # Evidence badge reuses the KPI quality line (same information, KPI styling).
    gui.lbl_summary_gq_badge = gq_kpi.quality_lbl
    gui.lbl_summary_gq_badge.configure(fg=WARNING)

    ac_kpi, gui.lbl_summary_ac_explain = _score_kpi(
        key="analysis_confidence",
        title="Analysis Confidence",
        help_key="analysis_confidence",
        show_bar=False,
    )
    gui.lbl_summary_ac_level = ac_kpi.value_lbl
    gui.lbl_summary_ac_unit = ac_kpi.unit_lbl

    gui.btn_walk_summary_details = ttk.Button(
        host,
        text="Details",
        style="Compact.TButton",
        command=gui._show_gait_summary_details,
    )
    gui.btn_walk_summary_details.pack(anchor=tk.W, pady=(PAD_XS, 0))

    gui.lbl_summary_demo_compare = tk.Label(
        host,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=240,
        justify=tk.LEFT,
    )
    gui.lbl_summary_demo_compare.pack(fill=tk.X, padx=2, pady=(4, 0))

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

    # Keep help text on the KPI shell (title + value + trend).
    for key, help_key in (
        ("movement_stability", "movement_stability"),
        ("gait_quality", "gait_quality"),
        ("analysis_confidence", "analysis_confidence"),
    ):
        kpi = gui._overview_kpi_cards.get(key)
        if kpi is not None and help_key in METRIC_HELP:
            set_kpi_tooltip(kpi, METRIC_HELP[help_key])

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
        text=f"{LABEL_FOOT_CLEARANCE_CONFIDENCE}: —",
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

    _FOOT_DISTANCE_FONT = FONT_DISPLAY

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

    export_intro = tk.Label(
        section,
        text=(
            "Export motion for Isaac Lab / OpenSim. "
            "Real-to-Sim Pipeline runs all 4 stages and writes "
            "stablewalk_motion.npz + amp_reference_motion.npz."
        ),
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=720,
        justify=tk.LEFT,
    )
    export_intro.grid(row=0, column=0, sticky="ew", pady=(0, 4))

    backend_row = ttk.Frame(section)
    backend_row.grid(row=1, column=0, sticky="ew", pady=(0, 6))
    tk.Label(
        backend_row,
        text="Pose backend:",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
    ).pack(side=tk.LEFT, padx=(0, 6))
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
    gui.lbl_pose_backend_status.pack(side=tk.LEFT, padx=(8, 0))
    gui._refresh_pose_backend_status_label()

    btn_host = ttk.Frame(section)
    btn_host.grid(row=2, column=0, sticky="ew")
    btn_host.columnconfigure(0, weight=1)
    btn_host.columnconfigure(1, weight=1)
    btn_host.columnconfigure(2, weight=1)

    buttons = (
        ("btn_view_detailed_data", "View Joint Data", "_toggle_collected_data_table", False),
        ("btn_save_session", "Save Session", "_save_session_to_files", True),
        ("btn_export_analysis_report", "Export Analysis Report", "_export_analysis_report", False),
        (
            "btn_export_pdf_report",
            "Export Professional PDF",
            "_export_professional_pdf_report",
            True,
        ),
        ("btn_export_gait_metrics", "Export Gait Metrics", "_export_gait_metrics", False),
        ("btn_opensim_export_data", "Export OpenSim Files", "_export_opensim_session", True),
        ("btn_export_motion_reference", "Export Motion Reference", "_export_motion_reference", False),
        ("btn_real_to_sim_pipeline", "Real-to-Sim Pipeline", "_run_real_to_sim_pipeline", True),
        ("btn_export_amp_reference", "Export AMP Reference", "_export_amp_reference", False),
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
    if hasattr(gui, "btn_export_pdf_report") and gui.btn_export_pdf_report is not None:
        gui.btn_export_pdf_report.configure(state=tk.DISABLED)
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
        wrap = responsive_wraplength(width, margin=16)
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
