"""
StableWalk GUI theme — colors, typography, and ttk styling only.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.coordinates.coordinate_map import compact_axis_legend

# Color palette — clinical dark workstation (cool neutrals, restrained accent)
BG = "#0b1018"
BG_GRADIENT = "#101822"
SURFACE = "#141b26"
PANEL = "#1a2332"
PANEL_HOVER = "#243044"
ELEVATED = "#222d40"
BORDER = "#354257"
BORDER_FOCUS = "#5ab0f0"
BORDER_SUBTLE = "#2a3548"

ACCENT = "#2ec99a"
ACCENT_DARK = "#1a9e6f"
ACCENT_FG = "#071018"
ACCENT_ALT = "#5ab0f0"
ACCENT_ALT_HOVER = "#3d9be0"

TEXT = "#eef3f8"
TEXT_SECONDARY = "#c2ccd9"
MUTED = "#8494ab"
MUTED_DIM = "#6a788f"

SUCCESS = "#2ec99a"
WARNING = "#ffc857"
ORANGE = "#ffc857"
DANGER = "#ff4757"
INFO = "#74c0fc"

# Floating HUD / overlay chrome (aligned to base palette)
HUD_BG = "#0b111b"
HUD_BORDER = BORDER

# Typography — single Segoe ladder + Cascadia for numeric/mono only
FONT_UI = ("Segoe UI", 10)
FONT_UI_SM = ("Segoe UI", 9)
FONT_UI_XS = ("Segoe UI", 9)  # Floor 9pt for projector-readable captions
FONT_UI_SEMIBOLD = ("Segoe UI Semibold", 10)
# Unified panel / section titles across LabelFrames and inline headers.
FONT_HEADING = ("Segoe UI Semibold", 11)
FONT_SECTION = ("Segoe UI Semibold", 12)
FONT_TITLE = ("Segoe UI Semibold", 13)
FONT_PANEL_HEADER = FONT_HEADING
FONT_METRIC = ("Segoe UI Semibold", 9)
FONT_MONO = ("Cascadia Mono", 9)
FONT_MONO_SM = ("Cascadia Mono", 9)
# Readable value fonts for the Selected Point Analysis metric cards.
FONT_METRIC_TITLE = ("Segoe UI", 9)
FONT_METRIC_VALUE = ("Cascadia Mono", 10)
FONT_METRIC_VALUE_ACCENT = ("Segoe UI Semibold", 11)
FONT_DISPLAY = ("Segoe UI Semibold", 15)
# Results Summary metric cards — larger, dashboard-style readout.
FONT_SUMMARY_CATEGORY = ("Segoe UI Semibold", 11)
FONT_SUMMARY_METRIC_TITLE = ("Segoe UI", 9)
FONT_SUMMARY_METRIC_VALUE = ("Segoe UI Semibold", 16)
FONT_SUMMARY_METRIC_VALUE_SM = ("Segoe UI Semibold", 13)
FONT_SUMMARY_METRIC_TIER = ("Segoe UI", 9)
# Shared modern KPI cards (Biomechanics / Motion / Results / Compare).
FONT_KPI_TITLE = ("Segoe UI", 9)
FONT_KPI_VALUE = ("Segoe UI Semibold", 18)
FONT_KPI_VALUE_SM = ("Segoe UI Semibold", 15)
FONT_KPI_UNIT = ("Segoe UI", 9)
FONT_KPI_TREND = ("Segoe UI Semibold", 10)
FONT_SUMMARY_REPORT_TITLE = ("Segoe UI Semibold", 15)
FONT_SUMMARY_REPORT_SUBTITLE = ("Segoe UI", 9)
FONT_SUMMARY_INTERPRETATION = ("Segoe UI", 10)
FONT_SUMMARY_SECTION_RULE = ("Segoe UI", 9)
FONT_TRANSPORT = ("Segoe UI", 10)
FONT_TABLE = ("Segoe UI", 10)
FONT_TABLE_HEADING = ("Segoe UI Semibold", 10)
FONT_CHART_LEGEND = ("Segoe UI", 9)
FONT_CHART_AXIS = ("Segoe UI", 9)
FONT_OVERLAY = ("Segoe UI Semibold", 11)
FONT_BADGE = ("Segoe UI", 9)
FONT_CAPTION = ("Segoe UI Semibold", 8)
FONT_FLOW_ICON = ("Segoe UI Semibold", 13)
FONT_FLOW_ARROW = ("Segoe UI Semibold", 11)
FONT_BUTTON = FONT_UI_SEMIBOLD
FONT_BUTTON_SM = FONT_UI_SM
POS_PANEL_WIDTH = 560

# Selection / hover (unified across treeview, text, tags)
SELECTION_BG = "#1a4538"
ACCENT_HOVER = "#3ad6a8"

# Layout breakpoints — shared with dashboard_responsive.py
WIDTH_STACK_ANALYSIS = 1280
WIDTH_COMPACT_TRANSPORT = 1050
WIDTH_METRIC_REFLOW = 560
SIDEBAR_MIN_WIDTH = 148
HEADER_HEIGHT = 40

# Spacing — strict 4px base / 8px rhythm
PAD_XS = 4
PAD_SM = 8
PAD_MD = 12
PAD_LG = 16
PAD_XL = 20

# Uniform button padding tiers (primary / secondary / compact / transport)
BTN_PAD = (10, 6)
BTN_PAD_COMPACT = (8, 4)
BTN_PAD_TRANSPORT = (10, 5)

# Card interior rhythm (8px grid)
CARD_INNER_PAD_X = PAD_SM
CARD_BODY_PAD = (PAD_SM, PAD_SM)
CARD_HEADER_PAD = (PAD_SM, PAD_XS)
CARD_GRID_GAP = PAD_XS
CARD_BORDER_WIDTH = 1

# Overview visualization workspace column gutter (shared symmetrically between
# adjacent panels for a balanced, professional rhythm).
OVERVIEW_COL_GUTTER = PAD_MD
OVERVIEW_ROW_GAP = PAD_SM

# Consistent dashboard rhythm: tight outer shell, uniform card gutters.
DASHBOARD_CARD_PAD = (PAD_SM, PAD_SM)
DASHBOARD_GUTTER = PAD_SM
DASHBOARD_VIZ_CARD_PAD = (PAD_XS, PAD_XS)
DASHBOARD_MAIN_PAD = (PAD_MD, PAD_SM, PAD_MD, PAD_SM)
DASHBOARD_SIDE_PAD = (PAD_SM, PAD_SM)
DASHBOARD_TAB_PAD = (PAD_SM, PAD_SM)
STATUS_BAR_PAD = (PAD_SM, PAD_XS)
RADIUS_EMULATE = 2  # relief for card-like frames

# User-facing copy (display only)
EMPTY_VIDEO_TEXT = (
    "Load a walking video to begin\n\n"
    "Demo Gait Examples  ·  File…  ·  Analyze"
)
EMPTY_NO_JOINT = "No joint selected"
EMPTY_SELECT_DOF_TRAJECTORY = "Select one or more joints to compare their trajectories."
EMPTY_OVERVIEW_JOINT_INSPECT = (
    "Click a joint on the skeleton to inspect its 3D trajectory."
)
TRAJ_DEFAULT_STATUS = (
    "Center of Mass path (estimated) and current pose · select joints to compare trajectories"
)
TRAJ_NO_MOTION = "Load gait data to view 3D trajectories"
EMPTY_SELECT_DOF_TABLE = "Select a point to track positions"
EMPTY_SELECT_DOF_CHART = "Select a joint to view its 3D movement path"
DOF_ANALYSIS_EXPLANATION_TITLE = "Overview"
DOF_ANALYSIS_LEGEND_TRACKED_FMT = "{point}"
DOF_ANALYSIS_PANEL_LINE_VALUES = "Current frame values"
DOF_ANALYSIS_PANEL_SUMMARY = DOF_ANALYSIS_PANEL_LINE_VALUES
DOF_ANALYSIS_CLARIFICATION = DOF_ANALYSIS_PANEL_LINE_VALUES
DOF_ANALYSIS_FOOT_GROUND_EXPLANATION = "Ground estimated from lowest foot positions"
DOF_ANALYSIS_FOOT_CLEARANCE_DEFINITION = "Distance to estimated ground"
DOF_ANALYSIS_FOOT_CLEARANCE_HERO_LABEL = "Foot Clearance"
DOF_ANALYSIS_MODE_GENERAL = "Point analysis"
DOF_ANALYSIS_MODE_FOOT = "Foot analysis"
DOF_ANALYSIS_MODE_GENERAL_HINT = "Inspect any selected joint path in 3D"
DOF_ANALYSIS_MODE_FOOT_HINT = "Foot-to-floor distance and contact context"
DOF_ANALYSIS_PANEL_GRAPH_GUIDE_TITLE = ""
DOF_ANALYSIS_PANEL_LINE_GREEN = "Start"
DOF_ANALYSIS_PANEL_LINE_BLUE = "Path"
DOF_ANALYSIS_PANEL_LINE_RED = "Now"
DOF_ANALYSIS_PANEL_LINE_X = "X"
DOF_ANALYSIS_PANEL_LINE_Y = "Y"
DOF_ANALYSIS_PANEL_LINE_Z = "Z"
DOF_ANALYSIS_GRAPH_CAPTION_PATH = "Path so far"
DOF_ANALYSIS_GRAPH_CAPTION_CURRENT = "Current position"
DOF_ANALYSIS_GRAPH_CAPTION_GENERAL = "Joint Movement"
DOF_ANALYSIS_GRAPH_CAPTION_FOOT = ""
DOF_ANALYSIS_GRAPH_CAPTION_GROUND = "Clearance to ground"
DOF_ANALYSIS_GRAPH_LEGEND_START = "Start"
DOF_ANALYSIS_GRAPH_LEGEND_PATH = "Path"
DOF_ANALYSIS_GRAPH_LEGEND_CURRENT = "Current"
DOF_ANALYSIS_PANEL_GRAPH_LINES: tuple[str, ...] = (
    DOF_ANALYSIS_PANEL_LINE_GREEN,
    DOF_ANALYSIS_PANEL_LINE_BLUE,
    DOF_ANALYSIS_PANEL_LINE_RED,
)
DOF_ANALYSIS_PANEL_INTERPRETATION_LINES = DOF_ANALYSIS_PANEL_GRAPH_LINES
DOF_ANALYSIS_MOVEMENT_TITLE = "Movement"
DOF_ANALYSIS_MOVEMENT_INTRO = ""
DOF_ANALYSIS_LEGEND_TITLE = "Colors"
DOF_ANALYSIS_AXES_TITLE = "Axes"
DOF_ANALYSIS_LEGEND_PATH = DOF_ANALYSIS_PANEL_LINE_BLUE
DOF_ANALYSIS_LEGEND_START = DOF_ANALYSIS_PANEL_LINE_GREEN
DOF_ANALYSIS_LEGEND_DOT = DOF_ANALYSIS_PANEL_LINE_RED
DOF_ANALYSIS_LEGEND_AXES_COMPACT = compact_axis_legend()
DOF_ANALYSIS_METRICS_LINK = DOF_ANALYSIS_PANEL_LINE_VALUES
DOF_ANALYSIS_INTERPRETATION = DOF_ANALYSIS_PANEL_LINE_VALUES
DOF_TRAJ_PATH_COLOR = "#5a9ec4"
DOF_TRAJ_START_COLOR = "#3d9a5f"
DOF_TRAJ_DOT_COLOR = "#c45c6a"
DOF_TRAJ_END_COLOR = "#4a7eb8"
EMPTY_SELECT_DOF_STEP = "Select degrees of freedom to preview the next movement step"
REFRESH_INTERVAL_CHOICES: tuple[str, ...] = ("0.25 s", "0.5 s")
REFRESH_INTERVAL_DEFAULT = "0.5 s"
DASHBOARD_SUBTITLE = "Gait Analysis Dashboard"
SIDEBAR_WIDTH = SIDEBAR_MIN_WIDTH - 10
OPENSIM_PANEL_MIN_HEIGHT = 0
OPENSIM_STATUS_SCROLL_HEIGHT = 100
PRESENTATION_VIDEO_CAPTION = (
    "Demo mode — synthetic walking skeleton\n"
    "For real video analysis, load or analyze a walking video from the toolbar."
)

# Matplotlib / 3D view (kept in sync with GUI + semantic L/R)
VIZ_BG = PANEL
VIZ_TEXT = TEXT
VIZ_MUTED = MUTED
VIZ_BONE = "#3d9a5f"
VIZ_JOINT = "#c9a227"
VIZ_GRID = BORDER


def apply_theme(root: tk.Tk | tk.Toplevel, style: ttk.Style) -> None:
    """Configure ttk and root defaults for StableWalk."""
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=TEXT, font=FONT_UI)
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=PANEL)
    style.configure("Transport.TFrame", background=SURFACE)

    style.configure(
        "Large.Treeview",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=PANEL,
        font=FONT_TABLE,
        rowheight=28,
    )
    style.configure(
        "Large.Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        padding=(PAD_SM, PAD_SM - 2),
    )
    style.map("Large.Treeview", background=[("selected", SELECTION_BG)], foreground=[("selected", TEXT)])

    style.configure(
        "Compact.Treeview",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=PANEL,
        font=FONT_TABLE,
        rowheight=28,
    )
    style.configure(
        "Compact.Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        padding=(PAD_SM, PAD_SM - 2),
    )
    style.map("Compact.Treeview", background=[("selected", SELECTION_BG)], foreground=[("selected", TEXT)])

    style.configure(
        "Compact.TButton",
        background=PANEL,
        foreground=ACCENT_ALT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=BTN_PAD_COMPACT,
        font=FONT_BUTTON_SM,
    )
    style.map(
        "Compact.TButton",
        background=[("active", ELEVATED), ("pressed", PANEL_HOVER)],
        foreground=[("disabled", MUTED_DIM)],
    )
    style.configure(
        "CompactAccent.TButton",
        background=ACCENT,
        foreground=ACCENT_FG,
        bordercolor=ACCENT_DARK,
        lightcolor=ACCENT,
        darkcolor=ACCENT_DARK,
        padding=BTN_PAD_COMPACT,
        font=FONT_BUTTON_SM,
    )
    style.map(
        "CompactAccent.TButton",
        background=[("active", ACCENT_HOVER), ("pressed", ACCENT_DARK)],
        foreground=[("disabled", MUTED_DIM)],
    )
    style.configure(
        "Export.TButton",
        background=PANEL,
        foreground=ACCENT_ALT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=BTN_PAD,
        font=FONT_BUTTON_SM,
    )
    style.map(
        "Export.TButton",
        background=[("active", ELEVATED), ("pressed", PANEL_HOVER)],
        foreground=[("disabled", MUTED_DIM)],
    )
    style.configure(
        "ExportAccent.TButton",
        background=ACCENT,
        foreground=ACCENT_FG,
        bordercolor=ACCENT_DARK,
        lightcolor=ACCENT,
        darkcolor=ACCENT_DARK,
        padding=BTN_PAD,
        font=FONT_BUTTON,
    )
    style.map(
        "ExportAccent.TButton",
        background=[("active", ACCENT_HOVER), ("pressed", ACCENT_DARK)],
        foreground=[("disabled", MUTED_DIM)],
    )

    style.configure(
        "TLabelframe",
        background=BG,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER,
        relief="flat",
        borderwidth=1,
    )
    style.configure(
        "TLabelframe.Label",
        background=BG,
        foreground=TEXT_SECONDARY,
        font=FONT_HEADING,
    )
    style.configure(
        "Card.TLabelframe",
        background=PANEL,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER,
        labelmargins=(PAD_XS, 0, 0, 0),
        borderwidth=1,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=PANEL,
        foreground=ACCENT_ALT,
        font=FONT_PANEL_HEADER,
    )
    style.configure(
        "Side.TLabelframe",
        background=SURFACE,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER,
        borderwidth=1,
    )
    style.configure(
        "Side.TLabelframe.Label",
        background=SURFACE,
        foreground=TEXT_SECONDARY,
        font=FONT_UI_SM,
    )

    style.configure("TLabel", background=BG, foreground=TEXT, font=FONT_UI)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=FONT_UI_SM)
    style.configure("Card.TLabel", background=PANEL, foreground=TEXT)
    style.configure("Side.TLabel", background=SURFACE, foreground=TEXT, font=FONT_UI_SM)
    style.configure("SideMuted.TLabel", background=SURFACE, foreground=MUTED, font=FONT_UI_XS)
    style.configure("Accent.TLabel", background=BG, foreground=ACCENT, font=FONT_UI_SM)
    style.configure("SideAccent.TLabel", background=SURFACE, foreground=ACCENT, font=FONT_UI_SM)
    style.configure("Heading.TLabel", background=BG, foreground=TEXT, font=FONT_HEADING)
    style.configure("Hint.TLabel", background=BG, foreground=INFO, font=FONT_UI_SM)
    style.configure("SideHint.TLabel", background=SURFACE, foreground=INFO, font=FONT_UI_XS)
    style.configure(
        "Guide.TLabel",
        background=ELEVATED,
        foreground=TEXT_SECONDARY,
        font=FONT_UI_SM,
        padding=(PAD_MD - 2, PAD_SM - 2),
    )
    style.configure(
        "EmptyState.TLabel",
        background=PANEL,
        foreground=MUTED,
        font=FONT_UI_SM,
        padding=(PAD_SM, PAD_MD - 2),
        wraplength=440,
    )
    style.configure(
        "SideEmpty.TLabel",
        background=SURFACE,
        foreground=MUTED,
        font=FONT_UI_SM,
        padding=(PAD_SM - 2, PAD_SM),
        wraplength=SIDEBAR_WIDTH - 24,
    )
    style.configure(
        "SectionStatus.TLabel",
        background=PANEL,
        foreground=TEXT_SECONDARY,
        font=FONT_UI_SM,
    )

    style.configure(
        "TButton",
        background=ELEVATED,
        foreground=TEXT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        focuscolor=ACCENT_ALT,
        padding=BTN_PAD,
        font=FONT_UI,
    )
    style.map(
        "TButton",
        background=[("active", PANEL_HOVER), ("pressed", BORDER)],
        foreground=[("disabled", MUTED_DIM)],
    )

    style.configure(
        "Accent.TButton",
        background=ACCENT,
        foreground=ACCENT_FG,
        bordercolor=ACCENT_DARK,
        lightcolor=ACCENT,
        darkcolor=ACCENT_DARK,
        padding=BTN_PAD,
        font=FONT_BUTTON,
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HOVER), ("pressed", ACCENT_DARK)],
        foreground=[("disabled", MUTED_DIM)],
    )

    style.configure(
        "Secondary.TButton",
        background=PANEL,
        foreground=ACCENT_ALT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=BTN_PAD,
        font=FONT_BUTTON_SM,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", ELEVATED), ("pressed", PANEL_HOVER)],
        foreground=[("disabled", MUTED_DIM)],
    )

    style.configure(
        "Transport.TButton",
        background=ELEVATED,
        foreground=TEXT,
        bordercolor=BORDER,
        lightcolor=BORDER,
        darkcolor=BORDER,
        padding=BTN_PAD_TRANSPORT,
        font=FONT_TRANSPORT,
    )
    style.map(
        "Transport.TButton",
        background=[("active", PANEL_HOVER), ("pressed", BORDER)],
        foreground=[("disabled", MUTED_DIM)],
    )

    style.configure(
        "TEntry",
        fieldbackground=PANEL,
        foreground=TEXT,
        bordercolor=BORDER,
        insertcolor=ACCENT,
        padding=6,
    )
    style.map("TEntry", bordercolor=[("focus", BORDER_FOCUS)])

    style.configure(
        "TCombobox",
        fieldbackground=PANEL,
        background=ELEVATED,
        foreground=TEXT,
        arrowcolor=ACCENT_ALT,
        bordercolor=BORDER,
        padding=4,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", PANEL), ("disabled", ELEVATED)],
        background=[("readonly", ELEVATED), ("active", PANEL_HOVER)],
        foreground=[("readonly", TEXT), ("disabled", MUTED_DIM)],
        selectbackground=[("readonly", PANEL)],
        selectforeground=[("readonly", TEXT)],
        arrowcolor=[("readonly", ACCENT_ALT), ("disabled", MUTED_DIM)],
    )

    style.configure(
        "Horizontal.TScale",
        background=BG,
        troughcolor=PANEL,
        bordercolor=BORDER,
    )
    style.configure(
        "Transport.Horizontal.TScale",
        background=SURFACE,
        troughcolor=PANEL,
    )

    style.configure(
        "TCheckbutton",
        background=BG,
        foreground=TEXT_SECONDARY,
        font=FONT_UI_SM,
    )
    style.map("TCheckbutton", background=[("active", BG)])

    style.configure(
        "Card.TCheckbutton",
        background=PANEL,
        foreground=TEXT_SECONDARY,
    )
    style.map("Card.TCheckbutton", background=[("active", PANEL)])

    style.configure(
        "Side.TCheckbutton",
        background=SURFACE,
        foreground=TEXT_SECONDARY,
        font=FONT_UI_XS,
    )
    style.map("Side.TCheckbutton", background=[("active", SURFACE)])

    style.configure(
        "Accent.Horizontal.TProgressbar",
        troughcolor=PANEL,
        background=ACCENT,
        bordercolor=BORDER,
        lightcolor=ACCENT,
        darkcolor=ACCENT_DARK,
        thickness=10,
    )

    style.configure(
        "Treeview",
        background=PANEL,
        fieldbackground=PANEL,
        foreground=TEXT,
        rowheight=28,
        font=FONT_TABLE,
        bordercolor=BORDER,
    )
    style.configure(
        "Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        relief="flat",
        padding=(PAD_SM, PAD_SM - 2),
    )
    style.map("Treeview", background=[("selected", SELECTION_BG)], foreground=[("selected", TEXT)])

    style.configure(
        "Vertical.TScrollbar",
        background=ELEVATED,
        troughcolor=PANEL,
        bordercolor=BORDER,
        arrowcolor=MUTED,
    )

    style.configure(
        "Status.TLabel",
        background=SURFACE,
        foreground=MUTED,
        font=FONT_UI_XS,
        padding=STATUS_BAR_PAD,
    )

    style.configure(
        "TNotebook",
        background=SURFACE,
        borderwidth=0,
        tabmargins=(PAD_XS, PAD_XS, PAD_XS, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=PANEL,
        foreground=MUTED,
        padding=(PAD_MD + 4, PAD_SM),
        font=FONT_UI_SEMIBOLD,
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", ELEVATED), ("active", PANEL_HOVER)],
        foreground=[("selected", TEXT), ("active", TEXT_SECONDARY)],
    )

    # Menu colors (tk.Menu, not ttk)
    root.option_add("*Menu.background", PANEL)
    root.option_add("*Menu.foreground", TEXT)
    root.option_add("*Menu.activeBackground", ELEVATED)
    root.option_add("*Menu.activeForeground", ACCENT)
    root.option_add("*Menu.borderWidth", 0)
    root.option_add("*Menu.relief", "flat")
    root.option_add("*Menu.font", FONT_UI)


def menu_colors() -> dict[str, str]:
    return {
        "bg": PANEL,
        "fg": TEXT,
        "activebackground": ELEVATED,
        "activeforeground": ACCENT,
    }


def configure_text_widget(widget: tk.Text, *, height: int | None = None) -> None:
    widget.configure(
        bg=PANEL,
        fg=TEXT,
        insertbackground=ACCENT,
        relief=tk.FLAT,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER_FOCUS,
        font=FONT_MONO,
        wrap=tk.WORD,
        selectbackground=SELECTION_BG,
        selectforeground=TEXT,
    )
    if height is not None:
        widget.configure(height=height)


def configure_video_placeholder(label: tk.Label) -> None:
    label.configure(
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        justify=tk.CENTER,
        anchor=tk.CENTER,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
        bd=0,
    )


def responsive_wraplength(container_width: int, *, margin: int = 20) -> int:
    """Compute label wraplength from a parent widget width."""
    return max(72, container_width - margin)


def create_elevated_card(parent: tk.Misc) -> tk.Frame:
    """Standard elevated metric card shell with consistent border."""
    return tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=CARD_BORDER_WIDTH,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )


def bind_responsive_wrap(
    gui,
    container: tk.Misc,
    label_attrs: tuple[str, ...],
    *,
    margin: int = 20,
) -> None:
    """Keep tk.Label wraplength synced to container width on resize."""

    def _sync(_event: object | None = None) -> None:
        width = container.winfo_width()
        if width < 40:
            return
        wrap = responsive_wraplength(width, margin=margin)
        for attr in label_attrs:
            lbl = getattr(gui, attr, None)
            if lbl is not None and hasattr(lbl, "configure"):
                try:
                    lbl.configure(wraplength=wrap)
                except tk.TclError:
                    pass

    container.bind("<Configure>", _sync, add="+")
    targets = getattr(gui, "_responsive_wrap_targets", ())
    gui._responsive_wrap_targets = (*targets, *label_attrs)


def format_stability_short(
    label: str,
    score: float,
    *,
    validity_status: str | None = None,
) -> tuple[str, str]:
    """One-line stability for the compact sidebar."""
    if validity_status == "INSUFFICIENT_DATA":
        return "Insufficient data · not comparable", WARNING
    if validity_status == "PROVISIONAL":
        return f"{label} · {score:.0f}/100 (provisional)", WARNING
    if label == "Stable":
        return f"{label} · {score:.0f}/100", SUCCESS
    return f"{label} · {score:.0f}/100", WARNING


def create_tooltip(
    widget: tk.Widget,
    text: str,
    *,
    delay_ms: int = 500,
    wraplength: int = 320,
) -> None:
    """Hover tooltip — presentation only, no behavior change.

    Re-calling with a new *text* updates the tip without stacking bindings.
    """
    widget._tooltip_text = text
    widget._tooltip_wraplength = wraplength
    if getattr(widget, "_tooltip_bound", False):
        return
    widget._tooltip_bound = True

    tip_win: tk.Toplevel | None = None
    after_id: str | None = None

    def position(win: tk.Toplevel) -> None:
        x = widget.winfo_rootx() + 12
        y = widget.winfo_rooty() + widget.winfo_height() + 6
        win.geometry(f"+{x}+{y}")

    def show() -> None:
        nonlocal tip_win
        if tip_win is not None:
            return
        tip_win = tk.Toplevel(widget)
        tip_win.wm_overrideredirect(True)
        tip_win.configure(bg=ELEVATED)
        tip_text = getattr(widget, "_tooltip_text", text)
        wrap = int(getattr(widget, "_tooltip_wraplength", wraplength) or wraplength)
        shell = tk.Frame(
            tip_win,
            bg=ELEVATED,
            highlightthickness=CARD_BORDER_WIDTH,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
        )
        shell.pack()
        lbl = tk.Label(
            shell,
            text=tip_text,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            justify=tk.LEFT,
            padx=PAD_MD - 2,
            pady=PAD_SM - 2,
            wraplength=wrap,
        )
        lbl.pack()
        position(tip_win)

    def hide() -> None:
        nonlocal tip_win, after_id
        if after_id:
            widget.after_cancel(after_id)
            after_id = None
        if tip_win is not None:
            tip_win.destroy()
            tip_win = None

    def on_enter(_event: tk.Event) -> None:
        nonlocal after_id
        hide()

        def deferred() -> None:
            show()

        after_id = widget.after(delay_ms, deferred)

    def on_leave(_event: tk.Event) -> None:
        hide()

    widget.bind("<Enter>", on_enter, add="+")
    widget.bind("<Leave>", on_leave, add="+")
    widget.bind("<ButtonPress>", on_leave, add="+")


def configure_demo_overlay(label: tk.Label) -> None:
    label.configure(
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_OVERLAY,
        padx=PAD_XL,
        pady=PAD_MD,
        wraplength=440,
        relief=tk.FLAT,
        highlightthickness=CARD_BORDER_WIDTH,
        highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )
