"""
StableWalk GUI theme — colors, typography, and ttk styling only.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

# Color palette
BG = "#0d1117"
BG_GRADIENT = "#121820"
SURFACE = "#161b22"
PANEL = "#1c2433"
PANEL_HOVER = "#243044"
ELEVATED = "#263348"
BORDER = "#303d55"
BORDER_FOCUS = "#4dabf7"

ACCENT = "#2ee59d"
ACCENT_DARK = "#1a9e6f"
ACCENT_FG = "#0a0f14"
ACCENT_ALT = "#4dabf7"
ACCENT_ALT_HOVER = "#339af0"

TEXT = "#f0f4f8"
TEXT_SECONDARY = "#c5d0de"
MUTED = "#7d8da6"
MUTED_DIM = "#5c6b82"

SUCCESS = "#2ee59d"
WARNING = "#ff6b81"
ORANGE = "#ffc857"
DANGER = "#ff4757"
INFO = "#74c0fc"

# Typography
FONT_UI = ("Segoe UI", 10)
FONT_UI_SM = ("Segoe UI", 9)
FONT_UI_XS = ("Segoe UI", 8)
FONT_HEADING = ("Segoe UI Semibold", 10)
FONT_TITLE = ("Segoe UI Semibold", 13)
FONT_METRIC = ("Segoe UI Semibold", 9)
FONT_MONO = ("Cascadia Mono", 9)
FONT_MONO_SM = ("Consolas", 8)
# Readable value fonts for the Selected Point Analysis metric cards.
FONT_METRIC_TITLE = ("Segoe UI", 8)
FONT_METRIC_VALUE = ("Consolas", 10)
FONT_METRIC_VALUE_ACCENT = ("Segoe UI Semibold", 11)
FONT_PANEL_HEADER = ("Segoe UI Semibold", 10)
FONT_TRANSPORT = ("Segoe UI", 10)
FONT_TABLE = ("Segoe UI", 10)
FONT_TABLE_HEADING = ("Segoe UI Semibold", 10)
FONT_OVERLAY = ("Segoe UI Semibold", 11)
POS_PANEL_WIDTH = 560

# Spacing
PAD_XS = 4
PAD_SM = 8
PAD_MD = 12
PAD_LG = 16
PAD_XL = 20
# Spacing between dashboard cards (gutter) and inside each card. Generous
# values keep panels from feeling cramped and stop titles/controls from
# crowding their neighbours.
DASHBOARD_CARD_PAD = (PAD_SM, PAD_SM)
DASHBOARD_GUTTER = PAD_SM
DASHBOARD_VIZ_CARD_PAD = (PAD_SM, PAD_SM)
DASHBOARD_MAIN_PAD = (PAD_MD, PAD_SM, PAD_MD, PAD_SM)
DASHBOARD_SIDE_PAD = (PAD_SM, PAD_XS)
RADIUS_EMULATE = 2  # relief for card-like frames

# User-facing copy (display only)
EMPTY_VIDEO_TEXT = "Choose a preset, then click Analyze"
EMPTY_NO_JOINT = "No joint selected"
EMPTY_SELECT_DOF_TRAJECTORY = "Select one or more joints to compare their trajectories."
TRAJ_DEFAULT_STATUS = (
    "Center-of-mass path and current pose · select joints to compare trajectories"
)
TRAJ_NO_MOTION = "Load gait data to view 3D trajectories"
EMPTY_SELECT_DOF_TABLE = "Select a point to track positions"
EMPTY_SELECT_DOF_CHART = "Select a body point to view movement"
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
DOF_ANALYSIS_MODE_GENERAL_HINT = ""
DOF_ANALYSIS_MODE_FOOT_HINT = ""
DOF_ANALYSIS_PANEL_GRAPH_GUIDE_TITLE = ""
DOF_ANALYSIS_PANEL_LINE_GREEN = "Start"
DOF_ANALYSIS_PANEL_LINE_BLUE = "Path"
DOF_ANALYSIS_PANEL_LINE_RED = "Current"
DOF_ANALYSIS_PANEL_LINE_X = "X"
DOF_ANALYSIS_PANEL_LINE_Y = "Y"
DOF_ANALYSIS_PANEL_LINE_Z = "Z"
DOF_ANALYSIS_GRAPH_CAPTION_PATH = "Path so far"
DOF_ANALYSIS_GRAPH_CAPTION_CURRENT = "Current position"
DOF_ANALYSIS_GRAPH_CAPTION_GENERAL = "3D point movement"
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
DOF_ANALYSIS_LEGEND_AXES_COMPACT = "X side · Y up · Z forward"
DOF_ANALYSIS_METRICS_LINK = DOF_ANALYSIS_PANEL_LINE_VALUES
DOF_ANALYSIS_INTERPRETATION = DOF_ANALYSIS_PANEL_LINE_VALUES
DOF_TRAJ_PATH_COLOR = "#82d8ff"
DOF_TRAJ_START_COLOR = "#44e899"
DOF_TRAJ_DOT_COLOR = "#ff5566"
EMPTY_SELECT_DOF_STEP = "Select degrees of freedom to preview the next movement step"
REFRESH_INTERVAL_CHOICES: tuple[str, ...] = ("0.25 s", "0.5 s")
REFRESH_INTERVAL_DEFAULT = "0.5 s"
DASHBOARD_SUBTITLE = "Gait Analysis Dashboard"
SIDEBAR_WIDTH = 138
OPENSIM_PANEL_MIN_HEIGHT = 0
OPENSIM_STATUS_SCROLL_HEIGHT = 100
PRESENTATION_VIDEO_CAPTION = (
    "Demo mode — synthetic walking skeleton\n"
    "For real video analysis, load or analyze a walking video from the toolbar."
)

# Matplotlib / 3D view (kept in sync with GUI)
VIZ_BG = PANEL
VIZ_TEXT = TEXT
VIZ_MUTED = MUTED
VIZ_BONE = ACCENT
VIZ_JOINT = "#ffc857"
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
        rowheight=24,
    )
    style.configure(
        "Large.Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=FONT_TABLE_HEADING,
        padding=(6, 4),
    )
    style.map("Large.Treeview", background=[("selected", ELEVATED)])

    style.configure(
        "Compact.Treeview",
        background=PANEL,
        foreground=TEXT,
        fieldbackground=PANEL,
        font=FONT_MONO,
        rowheight=26,
    )
    style.configure(
        "Compact.Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=("Segoe UI Semibold", 9),
        padding=(6, 5),
    )
    style.map("Compact.Treeview", background=[("selected", ELEVATED)])

    style.configure(
        "Compact.TButton",
        background=PANEL,
        foreground=ACCENT_ALT,
        bordercolor=BORDER,
        padding=(6, 3),
        font=FONT_UI_XS,
    )
    style.map(
        "Compact.TButton",
        background=[("active", ELEVATED), ("pressed", PANEL_HOVER)],
    )
    style.configure(
        "CompactAccent.TButton",
        background=ACCENT,
        foreground=ACCENT_FG,
        bordercolor=ACCENT_DARK,
        padding=(6, 4),
        font=FONT_UI_XS,
    )
    style.map(
        "CompactAccent.TButton",
        background=[("active", "#3df0ad"), ("pressed", ACCENT_DARK)],
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
        foreground=ACCENT,
        font=FONT_HEADING,
    )
    style.configure(
        "Card.TLabelframe",
        background=PANEL,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER,
    )
    style.configure(
        "Card.TLabelframe.Label",
        background=PANEL,
        foreground=ACCENT_ALT,
        font=FONT_HEADING,
    )
    style.configure(
        "Side.TLabelframe",
        background=SURFACE,
        foreground=TEXT_SECONDARY,
        bordercolor=BORDER,
    )
    style.configure(
        "Side.TLabelframe.Label",
        background=SURFACE,
        foreground=ACCENT,
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
        padding=(10, 6),
    )
    style.configure(
        "EmptyState.TLabel",
        background=PANEL,
        foreground=MUTED,
        font=FONT_UI_SM,
        padding=(8, 10),
        wraplength=440,
    )
    style.configure(
        "SideEmpty.TLabel",
        background=SURFACE,
        foreground=MUTED,
        font=FONT_UI_SM,
        padding=(6, 8),
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
        focuscolor=ACCENT_ALT,
        padding=(10, 6),
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
        padding=(12, 7),
        font=("Segoe UI Semibold", 10),
    )
    style.map(
        "Accent.TButton",
        background=[("active", "#3df0ad"), ("pressed", ACCENT_DARK)],
        foreground=[("disabled", MUTED_DIM)],
    )

    style.configure(
        "Secondary.TButton",
        background=PANEL,
        foreground=ACCENT_ALT,
        bordercolor=BORDER,
        padding=(8, 4),
        font=FONT_UI_SM,
    )
    style.map(
        "Secondary.TButton",
        background=[("active", ELEVATED), ("pressed", PANEL_HOVER)],
    )

    style.configure(
        "Transport.TButton",
        background=ELEVATED,
        foreground=TEXT,
        padding=(8, 5),
        font=FONT_TRANSPORT,
    )
    style.map("Transport.TButton", background=[("active", PANEL_HOVER)])

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
    style.map("TCombobox", fieldbackground=[("readonly", PANEL)])

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
        rowheight=24,
        font=FONT_UI_SM,
        bordercolor=BORDER,
    )
    style.configure(
        "Treeview.Heading",
        background=ELEVATED,
        foreground=ACCENT_ALT,
        font=("Segoe UI Semibold", 9),
        relief="flat",
    )
    style.map("Treeview", background=[("selected", "#1e4d3a")], foreground=[("selected", TEXT)])

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
        padding=(8, 5),
    )

    style.configure("TNotebook", background=SURFACE, borderwidth=0)
    style.configure("TNotebook.Tab", background=PANEL, foreground=MUTED, padding=(8, 4))
    style.map(
        "TNotebook.Tab",
        background=[("selected", ELEVATED)],
        foreground=[("selected", ACCENT)],
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
        selectbackground="#1e4d3a",
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
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
        bd=0,
    )


def format_stability_short(label: str, score: float) -> tuple[str, str]:
    """One-line stability for the compact sidebar."""
    if label == "Stable":
        return f"✓  {label}  ·  {score:.0f}/100", SUCCESS
    return f"⚠  {label}  ·  {score:.0f}/100", WARNING


def create_tooltip(widget: tk.Widget, text: str, *, delay_ms: int = 500) -> None:
    """Hover tooltip — presentation only, no behavior change."""
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
        lbl = tk.Label(
            tip_win,
            text=tip_text,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_SM,
            justify=tk.LEFT,
            padx=10,
            pady=6,
            wraplength=280,
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

    widget._tooltip_text = text
    widget.bind("<Enter>", on_enter, add="+")
    widget.bind("<Leave>", on_leave, add="+")
    widget.bind("<ButtonPress>", on_leave, add="+")


def configure_demo_overlay(label: tk.Label) -> None:
    label.configure(
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_OVERLAY,
        padx=20,
        pady=12,
        wraplength=440,
        relief=tk.FLAT,
        highlightthickness=2,
        highlightbackground=ACCENT_DARK,
        highlightcolor=ACCENT,
    )
