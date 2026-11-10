"""
StableWalk GUI theme — colors, typography, and ttk styling only.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.coordinates.coordinate_map import compact_axis_legend

# Color palette — calmer clinical dark (import submodule path to avoid package cycle)
import stablewalk.ui.design_system.tokens as _ds  # noqa: E402 — submodule only

BG = _ds.BG
BG_GRADIENT = "#101822"
SURFACE = _ds.SURFACE
PANEL = _ds.PANEL
PANEL_HOVER = "#1f2938"
PANEL_SECONDARY = _ds.PANEL_SECONDARY
ELEVATED = _ds.ELEVATED
BORDER = _ds.BORDER
BORDER_FOCUS = _ds.BORDER_FOCUS
BORDER_SUBTLE = _ds.BORDER_SUBTLE

# Cool steel accent (Qualisys / Vicon instrument chrome — not teal SaaS)
ACCENT = _ds.ACCENT
ACCENT_DARK = "#3d7fb0"
ACCENT_FG = _ds.ACCENT_FG
ACCENT_ALT = "#7eb6dc"
ACCENT_ALT_HOVER = "#5a9ec4"

TEXT = _ds.TEXT
TEXT_SECONDARY = _ds.TEXT_SECONDARY
MUTED = _ds.MUTED
MUTED_DIM = _ds.MUTED_DIM

SUCCESS = _ds.STATUS_OK
SUCCESS_BRIGHT = "#42be65"
SUCCESS_GLOW = "#5ce1a8"
SUCCESS_MUTED = "#3d5a4a"
SUCCESS_MID = "#4a7a5c"
SUCCESS_SOFT = "#5ea874"
WARNING = _ds.STATUS_WARN
ORANGE = _ds.STATUS_WARN
DANGER = _ds.STATUS_FAIL
INFO = ACCENT  # align hint/info chrome with brand accent

# Tree / table semantic backgrounds (derived from DANGER on dark panel)
TREE_LOW_CONFIDENCE_BG = "#3d1f28"

# Floating HUD / overlay chrome (aligned to base palette)
HUD_BG = "#0b111b"
HUD_BORDER = BORDER

# Typography — design-system hierarchy (no body/meta smaller than 9 pt)
FONT_UI = _ds.FONT_BODY
FONT_UI_SM = _ds.FONT_SUPPORTING
FONT_UI_XS = _ds.FONT_META  # floor at 9 pt — avoid cramped labels
FONT_UI_SEMIBOLD = _ds.FONT_BODY_SEMIBOLD
# Unified panel / section titles across LabelFrames and inline headers.
FONT_HEADING = _ds.FONT_CARD_TITLE
FONT_SECTION = _ds.FONT_SECTION_TITLE
FONT_TITLE = _ds.FONT_PAGE_TITLE
FONT_PANEL_HEADER = FONT_HEADING
# Panel metric values (14 pt). Hero KPIs use FONT_KPI_HERO (18 pt).
FONT_METRIC = _ds.FONT_METRIC_SM
FONT_METRIC_CAPTION = _ds.FONT_CARD_TITLE
FONT_MONO = _ds.FONT_MONO
FONT_MONO_SM = _ds.FONT_MONO_SM
# Readable value fonts for the Selected Point Analysis metric cards.
FONT_METRIC_TITLE = _ds.FONT_SUPPORTING
FONT_METRIC_VALUE = _ds.FONT_MONO
FONT_METRIC_VALUE_ACCENT = _ds.FONT_SECTION_TITLE
FONT_DISPLAY = _ds.FONT_PAGE_TITLE
# Results Summary metric cards — dashboard-style readout (token hierarchy only).
FONT_SUMMARY_CATEGORY = _ds.FONT_CARD_TITLE
FONT_SUMMARY_METRIC_TITLE = _ds.FONT_SUPPORTING
FONT_SUMMARY_METRIC_VALUE = _ds.FONT_PAGE_TITLE
FONT_SUMMARY_METRIC_VALUE_SM = _ds.FONT_METRIC_SM
FONT_SUMMARY_METRIC_TIER = _ds.FONT_META
# Shared modern KPI cards (Biomechanics / Motion / Results / Compare).
# Hierarchy: HERO (primary finding) → VALUE → SM → SUPPORTING (ambient metrics).
FONT_KPI_TITLE = _ds.FONT_SUPPORTING
FONT_KPI_TITLE_PRIMARY = _ds.FONT_CARD_TITLE
FONT_KPI_HERO = _ds.FONT_METRIC
FONT_KPI_VALUE = _ds.FONT_METRIC_MD
FONT_KPI_VALUE_SM = _ds.FONT_METRIC_SM
FONT_KPI_SUPPORTING = _ds.FONT_SECTION_TITLE
FONT_KPI_UNIT = _ds.FONT_META
FONT_KPI_TREND = _ds.FONT_SUPPORTING_SEMIBOLD
FONT_KPI_GROUP = _ds.FONT_META
KPI_ACCENT_PRIMARY = 4
KPI_ACCENT_SECONDARY = 3
KPI_ACCENT_SUPPORTING = 2
FONT_SUMMARY_REPORT_TITLE = _ds.FONT_SECTION_TITLE
FONT_SUMMARY_REPORT_SUBTITLE = _ds.FONT_SUPPORTING
FONT_SUMMARY_INTERPRETATION = _ds.FONT_SUPPORTING
FONT_SUMMARY_SECTION_RULE = _ds.FONT_SUPPORTING_SEMIBOLD
FONT_TRANSPORT = _ds.FONT_SUPPORTING
FONT_TABLE = _ds.FONT_SUPPORTING
FONT_TABLE_HEADING = _ds.FONT_SUPPORTING_SEMIBOLD
FONT_CHART_LEGEND = _ds.FONT_META
FONT_CHART_AXIS = _ds.FONT_META
FONT_OVERLAY = _ds.FONT_SUPPORTING_SEMIBOLD
FONT_BADGE = _ds.FONT_META
FONT_CAPTION = _ds.FONT_SUPPORTING_SEMIBOLD
FONT_FLOW_ICON = _ds.FONT_SECTION_TITLE
FONT_FLOW_ARROW = _ds.FONT_BODY_SEMIBOLD
FONT_BUTTON = _ds.FONT_BUTTON
FONT_BUTTON_SM = _ds.FONT_BUTTON_SM
POS_PANEL_WIDTH = 560

# Selection / hover (unified across treeview, text, tags)
SELECTION_BG = "#1a3348"
ACCENT_HOVER = "#6aafde"

# Layout breakpoints — shared with dashboard_responsive.py
WIDTH_STACK_ANALYSIS = 1280
WIDTH_COMPACT_TRANSPORT = 1050
WIDTH_METRIC_REFLOW = 560
SIDEBAR_MIN_WIDTH = 148
HEADER_HEIGHT = 44

# Spacing — always from the design-system scale (4 / 8 / 16 / 24 / 32).
PAD_XS = _ds.PAD_XS
PAD_SM = _ds.PAD_SM
PAD_MD = _ds.PAD_MD
PAD_LG = _ds.PAD_LG
PAD_XL = _ds.PAD_XL
INLINE_GAP = _ds.INLINE_GAP
CONTROL_GAP = _ds.CONTROL_GAP
SECTION_GAP = _ds.SECTION_GAP
PAGE_GUTTER = _ds.PAGE_GUTTER
CARD_GAP = _ds.CARD_GAP
PAGE_HEADER_PADY = _ds.PAGE_HEADER_PADY
PAGE_HEADER_SUBTITLE_PADY = _ds.PAGE_HEADER_SUBTITLE_PADY
SECTION_TITLE_PADY = _ds.SECTION_TITLE_PADY

# Uniform button padding tiers (primary / secondary / compact / transport)
BTN_PAD = _ds.BTN_PAD
BTN_PAD_COMPACT = _ds.BTN_PAD_COMPACT
BTN_PAD_TRANSPORT = _ds.BTN_PAD_TRANSPORT
BUTTON_HEIGHT = _ds.BUTTON_HEIGHT
CONTROL_HEIGHT = _ds.CONTROL_HEIGHT
RADIUS_PX = _ds.RADIUS_PX

# Card interior rhythm — prefer surface contrast over borders
CARD_INNER_PAD_X = PAD_SM
CARD_BODY_PAD = _ds.CARD_PAD
CARD_HEADER_PAD = _ds.CARD_PAD
CARD_GRID_GAP = CARD_GAP
CARD_BORDER_WIDTH = _ds.CARD_BORDER_WIDTH

# Overview visualization workspace column gutter (shared symmetrically between
# adjacent panels).
OVERVIEW_COL_GUTTER = PAD_SM
OVERVIEW_ROW_GAP = PAD_SM
OVERVIEW_VIZ_PAD = (PAD_SM, PAD_SM)
# Shared chrome under Video / Skeleton / 3D Path titles. LabelFrame already
# insets OVERVIEW_VIZ_PAD; keep a small left inset so Windows cannot clip
# the first control glyph (Joint / Cam / camera buttons).
OVERVIEW_VIZ_TOOLBAR_PADX = (PAD_XS, 0)
OVERVIEW_VIZ_TOOLBAR_PADY = (PAD_XS, PAD_XS)
OVERVIEW_VIZ_TOOLBAR_MIN_H = 36
OVERVIEW_VIZ_TITLE_SPACER_PX = 12
OVERVIEW_VIZ_TITLE_PADX = 4
OVERVIEW_VIZ_TITLE_PADY = 2

# Consistent dashboard rhythm — airier clinical research layout.
DASHBOARD_CARD_PAD = (PAD_MD, PAD_MD)
DASHBOARD_GUTTER = PAGE_GUTTER
DASHBOARD_VIZ_CARD_PAD = OVERVIEW_VIZ_PAD
DASHBOARD_MAIN_PAD = (PAD_MD, PAD_MD, PAD_MD, PAD_MD)
DASHBOARD_SIDE_PAD = (PAD_MD, PAD_MD)
DASHBOARD_TAB_PAD = _ds.DASHBOARD_TAB_PAD
STATUS_BAR_PAD = (PAD_MD, PAD_SM)
KPI_CARD_GAP = _ds.KPI_GAP
KPI_GROUP_GAP = SECTION_GAP
RADIUS_EMULATE = 0  # flat surfaces; no faux card relief

# Subtle motion (fade / micro-slide) — see ``stablewalk.ui.tk.ui_motion``.
MOTION_MS = _ds.MOTION_MS
MOTION_STEPS = _ds.MOTION_STEPS
MOTION_SLIDE_PX = _ds.MOTION_SLIDE_PX

# User-facing copy (display only)
EMPTY_VIDEO_TEXT = (
    "No video loaded\n\n"
    "Browse a walking video, or open Examples from the toolbar."
)
EMPTY_NO_JOINT = "No joint selected"
EMPTY_SELECT_DOF_TRAJECTORY = "Select one or more joints to compare trajectories."
EMPTY_OVERVIEW_JOINT_INSPECT = (
    "Select a joint on the skeleton or from the DOF list to inspect its 3D path."
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
DOF_TRAJ_PATH_COLOR = "#4eb3d9"
DOF_TRAJ_DOT_COLOR = "#ffd666"  # amber — current playhead (≠ green start / red end)
DOF_TRAJ_START_COLOR = SUCCESS  # green — path start
DOF_TRAJ_END_COLOR = DANGER  # red — path end
# One accent for Overview joint sync: video · skeleton · trajectory tip · inspector.
SELECTED_JOINT_ACCENT = DOF_TRAJ_DOT_COLOR
# OpenCV BGR of SELECTED_JOINT_ACCENT (#ffd666 → R255 G214 B102).
SELECTED_JOINT_ACCENT_BGR: tuple[int, int, int] = (102, 214, 255)
# Bright pulse when Coordinate Inspector values change during playback.
COORDINATE_INSPECTOR_LIVE_HIGHLIGHT = "#ffffff"
EMPTY_SELECT_DOF_STEP = "Select degrees of freedom to preview the next movement step"
REFRESH_INTERVAL_CHOICES: tuple[str, ...] = ("0.25 s", "0.5 s")
REFRESH_INTERVAL_DEFAULT = "0.5 s"
DASHBOARD_SUBTITLE = "Gait analysis"
SIDEBAR_WIDTH = SIDEBAR_MIN_WIDTH - 10
OPENSIM_PANEL_MIN_HEIGHT = 0
OPENSIM_STATUS_SCROLL_HEIGHT = 100
PRESENTATION_VIDEO_CAPTION = (
    "Example session — synthetic walking skeleton\n"
    "For recorded video analysis, load or analyze a walking video from the toolbar."
)

# Matplotlib / 3D view (kept in sync with GUI + semantic L/R)
VIZ_BG = PANEL
VIZ_TEXT = TEXT
VIZ_MUTED = MUTED
VIZ_BONE = SUCCESS
VIZ_JOINT = "#c9a227"
VIZ_GRID = BORDER

# Shared chrome glyphs (disclosure / trend / legend) — one vocabulary app-wide.
GLYPH_EXPAND = "▶"
GLYPH_COLLAPSE = "▼"
GLYPH_TREND_UP = "▲"
GLYPH_TREND_DOWN = "▼"
GLYPH_TREND_FLAT = "●"
GLYPH_LEGEND_DOT = "●"
GLYPH_OK = "✓"
GLYPH_WARN = "⚠"
GLYPH_BULLET = "•"

# Thin status accent used on clinical KPI cards (display only).
CARD_ACCENT_HEIGHT = _ds.CARD_ACCENT_HEIGHT

# Overview wraplengths — shared so cards stay aligned across panel widths.
OVERVIEW_WRAP_METRIC = 168
OVERVIEW_WRAP_HUD = 220
OVERVIEW_WRAP_REPORT = 300
OVERVIEW_WRAP_MONITOR = 168
OVERVIEW_ROW_ITEM_GAP = PAD_XS


def apply_theme(root: tk.Tk | tk.Toplevel, style: ttk.Style) -> None:
    """Configure ttk and root defaults for StableWalk."""
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=TEXT, font=FONT_UI)
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=PANEL)
    style.configure("Transport.TFrame", background=SURFACE)

    try:
        from stablewalk.ui.tk.layout_scale import dpi_scale_factor, scaled_px

        _tree_row_h = scaled_px(28, dpi_scale_factor(root), minimum=26)
    except Exception:
        _tree_row_h = 28

    style.configure(
        "Large.Treeview",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=PANEL,
        font=FONT_TABLE,
        rowheight=_tree_row_h,
    )
    style.configure(
        "Large.Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        padding=(PAD_SM, PAD_SM),
    )
    style.map(
        "Large.Treeview",
        background=[("selected", SELECTION_BG), ("!selected", PANEL)],
        foreground=[("selected", TEXT), ("!selected", TEXT)],
    )

    style.configure(
        "Compact.Treeview",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=PANEL,
        font=FONT_TABLE,
        rowheight=_tree_row_h,
    )
    style.configure(
        "Compact.Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        padding=(PAD_SM, PAD_SM),
    )
    style.map(
        "Compact.Treeview",
        background=[("selected", SELECTION_BG), ("!selected", PANEL)],
        foreground=[("selected", TEXT), ("!selected", TEXT)],
    )

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
        bordercolor=BORDER_SUBTLE,
        relief="flat",
        borderwidth=0,
    )
    style.configure(
        "TLabelframe.Label",
        background=BG,
        foreground=TEXT_SECONDARY,
        font=FONT_SECTION,
    )
    style.configure(
        "Card.TLabelframe",
        background=PANEL,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER_SUBTLE,
        labelmargins=(0, 0, 0, PAD_SM),
        borderwidth=0,
        relief="flat",
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=PANEL,
        foreground=TEXT_SECONDARY,
        font=FONT_SECTION,
    )
    style.configure(
        "Tool.TCheckbutton",
        background=ELEVATED,
        foreground=TEXT_SECONDARY,
        font=FONT_UI_XS,
        padding=(PAD_SM, PAD_XS),
        focuscolor=BORDER_FOCUS,
    )
    style.map(
        "Tool.TCheckbutton",
        background=[("selected", SELECTION_BG), ("active", PANEL_HOVER)],
        foreground=[("selected", TEXT), ("active", TEXT)],
    )
    style.configure(
        "Side.TLabelframe",
        background=SURFACE,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER_SUBTLE,
        borderwidth=0,
        relief="flat",
    )
    style.configure(
        "Side.TLabelframe.Label",
        background=SURFACE,
        foreground=TEXT_SECONDARY,
        font=FONT_SECTION,
    )

    style.configure("TLabel", background=BG, foreground=TEXT, font=FONT_UI)
    style.configure("Muted.TLabel", background=BG, foreground=MUTED, font=FONT_UI_SM)
    style.configure("Card.TLabel", background=PANEL, foreground=TEXT)
    style.configure("Side.TLabel", background=SURFACE, foreground=TEXT, font=FONT_UI_SM)
    style.configure("SideMuted.TLabel", background=SURFACE, foreground=MUTED, font=FONT_UI_XS)
    style.configure("Accent.TLabel", background=BG, foreground=ACCENT, font=FONT_UI_SM)
    style.configure("SideAccent.TLabel", background=SURFACE, foreground=ACCENT, font=FONT_UI_SM)
    style.configure("Heading.TLabel", background=BG, foreground=TEXT, font=FONT_SECTION)
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
        font=FONT_UI,
        padding=(PAD_MD, PAD_MD),
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
        rowheight=_tree_row_h,
        font=FONT_TABLE,
        bordercolor=BORDER,
    )
    style.configure(
        "Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        relief="flat",
        padding=(PAD_SM, PAD_SM),
    )
    style.map(
        "Treeview",
        background=[("selected", SELECTION_BG), ("!selected", PANEL)],
        foreground=[("selected", TEXT), ("!selected", TEXT)],
    )

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
        padding=(PAD_MD, PAD_XS + 1),
        font=FONT_UI_SEMIBOLD,
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", ELEVATED), ("active", PANEL_HOVER)],
        foreground=[("selected", TEXT), ("active", TEXT_SECONDARY)],
        # Selected tab reads as instrument chrome (stronger than idle tabs).
        lightcolor=[("selected", ACCENT), ("!selected", PANEL)],
        bordercolor=[("selected", ACCENT), ("!selected", BORDER)],
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
        wraplength=280,
        highlightthickness=0,
        highlightbackground=PANEL,
        highlightcolor=PANEL,
        bd=0,
    )

    def _wrap(_event: object | None = None) -> None:
        try:
            width = int(label.winfo_width() or 0)
            if width > 40:
                label.configure(wraplength=max(80, width - 16))
        except tk.TclError:
            pass

    if not getattr(label, "_sw_placeholder_wrap_bound", False):
        label.bind("<Configure>", _wrap, add="+")
        label._sw_placeholder_wrap_bound = True


def responsive_wraplength(container_width: int, *, margin: int = 20) -> int:
    """Compute label wraplength from a parent widget width."""
    return max(72, container_width - margin)


def set_wraplength_safe(label: tk.Misc, wrap: int) -> bool:
    """Set wraplength only when the value changes. Returns True if applied."""
    try:
        current = int(float(label.cget("wraplength") or 0))
    except (tk.TclError, TypeError, ValueError):
        current = -1
    if current == int(wrap):
        return False
    try:
        label.configure(wraplength=int(wrap))
        return True
    except tk.TclError:
        return False


def create_elevated_card(parent: tk.Misc) -> tk.Frame:
    """Quiet elevated surface — separation via background, not a full border."""
    return tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=CARD_BORDER_WIDTH,
        highlightbackground=BORDER_SUBTLE,
        highlightcolor=BORDER_SUBTLE,
    )


def format_labelframe_title(label: str) -> str:
    """Uniform LabelFrame title used by every dashboard tab."""
    return (label or "").strip()


def tk_labelframe_title(
    parent: tk.Misc,
    title: str,
    *,
    bg: str = PANEL,
) -> tk.Label:
    """Readable LabelFrame caption for Canvas-hosted Advanced panels.

    Native ttk.LabelFrame ``text=`` often paints as a blank or black bar when
    the frame sits inside a Tk Canvas window (Windows). A tk.Label used as
    ``labelwidget`` keeps the title visible without changing other tabs.
    """
    return tk.Label(
        parent,
        text=format_labelframe_title(title),
        bg=bg,
        fg=TEXT,
        font=FONT_SECTION,
        anchor="w",
    )


def disclosure_label(title: str, *, expanded: bool) -> str:
    """Shared ▶/▼ disclosure caption."""
    arrow = GLYPH_COLLAPSE if expanded else GLYPH_EXPAND
    return f"{arrow}  {title}"


def pack_empty_state(
    parent: tk.Misc,
    text: str,
    *,
    style: str = "EmptyState.TLabel",
    padx: int = PAD_SM,
    pady: int = PAD_SM,
) -> ttk.Label:
    """Standard empty-state label (one style across tabs)."""
    lbl = ttk.Label(
        parent,
        text=text,
        style=style,
        anchor="center",
        justify=tk.CENTER,
    )
    lbl.pack(fill=tk.BOTH, expand=True, padx=padx, pady=pady)
    return lbl


def bind_responsive_wrap(
    gui,
    container: tk.Misc,
    label_attrs: tuple[str, ...],
    *,
    margin: int = 20,
) -> None:
    """Keep tk.Label wraplength synced to container width on resize.

    Guards against Configure → wraplength → Configure feedback that can blow
    the Tk callback stack during post-analysis UI refresh.
    """

    def _sync(_event: object | None = None) -> None:
        if getattr(container, "_wrap_syncing", False):
            return
        width = container.winfo_width()
        if width < 40:
            return
        wrap = responsive_wraplength(width, margin=margin)
        setattr(container, "_wrap_syncing", True)
        try:
            for attr in label_attrs:
                lbl = getattr(gui, attr, None)
                if lbl is not None and hasattr(lbl, "configure"):
                    set_wraplength_safe(lbl, wrap)
        finally:
            setattr(container, "_wrap_syncing", False)

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
        try:
            tip_win.wm_attributes("-topmost", True)
        except tk.TclError:
            pass
        # Never steal focus from the main application window.
        try:
            tip_win.transient(widget.winfo_toplevel())
        except tk.TclError:
            pass
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
