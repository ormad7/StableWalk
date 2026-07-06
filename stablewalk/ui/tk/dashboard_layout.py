"""
Professional dashboard grid for the StableWalk gait analysis GUI.

Organizes visualization, analytics tables, and the DOF control sidebar into
clear, presentation-ready sections with consistent spacing and empty states.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from stablewalk.ui.dof_position_table import (
    DOF_TABLE_COLUMNS,
    DOF_TABLE_HEADINGS,
    DOF_TABLE_MODE_DEFAULT,
)
from stablewalk.ui.selection_state import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS
from stablewalk.ui.theme import (
    ACCENT,
    ACCENT_ALT,
    BG,
    BORDER,
    ELEVATED,
    EMPTY_SELECT_DOF_TABLE,
    EMPTY_SELECT_DOF_CHART,
    EMPTY_VIDEO_TEXT,
    FONT_MONO_SM,
    FONT_METRIC,
    FONT_METRIC_TITLE,
    FONT_METRIC_VALUE,
    FONT_METRIC_VALUE_ACCENT,
    FONT_PANEL_HEADER,
    FONT_UI_SM,
    FONT_UI_XS,
    INFO,
    MUTED,
    DASHBOARD_CARD_PAD,
    DASHBOARD_GUTTER,
    DASHBOARD_MAIN_PAD,
    DASHBOARD_SIDE_PAD,
    DASHBOARD_VIZ_CARD_PAD,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    PANEL_HOVER,
    REFRESH_INTERVAL_CHOICES,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
    REFRESH_INTERVAL_DEFAULT,
    OPENSIM_PANEL_MIN_HEIGHT,
    OPENSIM_STATUS_SCROLL_HEIGHT,
    SIDEBAR_WIDTH,
    SURFACE,
    ORANGE,
    configure_demo_overlay,
    configure_video_placeholder,
)
from stablewalk.ui.viewers.dof_trajectory_3d import setup_single_dof_trajectory_axes
from stablewalk.ui.viewers.gait_skeleton_renderer import (
    DEFAULT_SKELETON_DISPLAY_MODE,
    LABEL_TO_SKELETON_MODE,
    MODE_TO_SKELETON_LABEL,
    SKELETON_MODE_LABELS,
    SKELETON_PANEL_MARGINS,
    _layout_skeleton_figure,
    relayout_skeleton_viewport,
    setup_skeleton_axes,
)

# Placeholder shown in the "Add a point" dropdown
ADD_POINT_PLACEHOLDER = "Add joint\u2026"

# Primary visual area (~52% height): Original Video + 3D Reconstruction + summary.
# Analysis row (~48%): Selected Joint Analysis + movement graph (scrollable).
_TOP_VIDEO_WEIGHT = 48
_TOP_SKELETON_WEIGHT = 34
_SIDEBAR_WEIGHT = 18
_BOTTOM_ANALYSIS_WEIGHT = 1
_BOTTOM_TABLE_WEIGHT = 0
_TABLE_PANEL_WIDTH = 0
_TABLE_PANEL_WIDTH_EXPANDED = 520
_DASH_VIZ_ROW_WEIGHT = 52
_DASH_ANALYTICS_ROW_WEIGHT = 48
_DASH_VIZ_ROW_MINSIZE = 200
_DASH_ANALYTICS_ROW_MINSIZE = 180
_SIDEBAR_MIN_WIDTH = 152
_ANALYTICS_ANALYSIS_MINSIZE = 280
_ANALYTICS_TABLE_MINSIZE = 0
_ANALYSIS_GRAPH_INNER_PAD = (2, 2, 2, 2)
_ANALYSIS_INFO_STRIP_PAD = (2, 3)
_ANALYSIS_PANEL_PAD = (PAD_SM, PAD_XS, PAD_SM, PAD_XS)
_ANALYSIS_GRAPH_BOTTOM_PAD = 0
_ANALYSIS_PANEL_PLAYBACK_GAP = 8
_ANALYSIS_LEFT_PANEL_WIDTH = 204
# Width of the explanation panel placed to the right of the 3D cube. The cube
# keeps the remaining (taller, squarer) area so it renders large and uncut,
# while this column uses the panel's wide horizontal space for the legend and a
# short description of what the graph shows.
_ANALYSIS_EXPLAIN_WIDTH = 140
_ANALYSIS_GRAPH_CANVAS_STICKY = "nsew"
_ANALYSIS_GRAPH_HEADER_ROW = 0
_ANALYSIS_GRAPH_CANVAS_ROW = 1
_ANALYSIS_GRAPH_CAPTION_ROW = 2
_ANALYSIS_BODY_ROW_WEIGHT = 1
_TRAJ_FIG_HEIGHT_SCALE = 1.0
_TABLE_PANEL_PAD = (PAD_XS, PAD_XS, PAD_XS, PAD_XS)
_SKELETON_PANEL_PAD = (PAD_XS, 2)
_SKELETON_CANVAS_PAD = (0, 0, 0, 0)
_SKELETON_HEADER_PAD = (0, 0)
_ANALYTICS_SECONDARY_SLOTS = 4
_ANALYTICS_ADVANCED_COORD_SLOTS = 4

_BODY_CHECKLIST_HEIGHT = 1
_ANALYTICS_SUMMARY_SLOTS = 4
_TABLE_TREE_HEIGHT = 10
# Readable column widths: the six core columns (Time, Frame, Selected Point,
# X, Y, Z) fit without horizontal scrolling; Speed/foot columns scroll in.
_TABLE_COL_WIDTHS_COMPACT: dict[str, int] = {
    "time": 50,
    "frame": 46,
    "dof": 100,
    "x": 54,
    "y": 54,
    "z": 54,
    "speed": 58,
    "foot_clearance": 84,
    "contact_status": 90,
}


def _bind_figure_resize(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    *,
    margins: dict[str, float] | None = None,
    min_px: int = 100,
) -> None:
    """Resize a matplotlib figure to match its Tk widget (keeps plots usable)."""

    def _on_resize(event: tk.Event) -> None:
        if event.width < min_px or event.height < min_px:
            return
        dpi = fig.get_dpi()
        fig.set_size_inches(event.width / dpi, event.height / dpi, forward=True)
        if margins is not None:
            fig.subplots_adjust(**margins)
        canvas.draw_idle()

    canvas.get_tk_widget().bind("<Configure>", _on_resize)


def _bind_skeleton_figure_resize(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 100,
    pad: tuple[int, int, int, int] = _SKELETON_CANVAS_PAD,
) -> None:
    """Resize the skeleton figure and preserve equal-aspect layout."""

    def _fit(event: tk.Event | object) -> None:
        widget = canvas.get_tk_widget()
        host = widget.master
        width = int(getattr(event, "width", 0) or 0)
        height = int(getattr(event, "height", 0) or 0)
        if host is not None:
            host.update_idletasks()
            host_w = host.winfo_width()
            host_h = host.winfo_height()
            if host_w >= min_px:
                width = host_w
            if host_h >= min_px:
                height = host_h
        pad_l, pad_t, pad_r, pad_b = pad
        width = max(min_px, width - pad_l - pad_r)
        height = max(min_px, height - pad_t - pad_b)
        if width < min_px or height < min_px:
            return
        try:
            canvas.resize(type("E", (), {"width": width, "height": height})())
        except (TypeError, AttributeError, tk.TclError):
            dpi = fig.get_dpi()
            fig.set_size_inches(width / dpi, height / dpi, forward=True)
        relayout_skeleton_viewport(ax)
        canvas.draw_idle()

    def _on_resize(event: tk.Event) -> None:
        _fit(event)

    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.bind("<Configure>", _on_resize)

    host = widget.master
    if host is not None:
        host.bind("<Configure>", _on_resize)


def _fit_trajectory_figure(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 40,
    graph_host: tk.Misc | None = None,
) -> bool:
    """Resize the 3D figure to exactly fill the graph host (no empty bands)."""
    widget = canvas.get_tk_widget()
    root = widget.winfo_toplevel()
    root.update_idletasks()
    widget.update_idletasks()

    host = graph_host if graph_host is not None else widget.master
    pad_l, pad_t, pad_r, pad_b = _ANALYSIS_GRAPH_INNER_PAD
    host_w = 0
    host_h = 0

    if host is not None:
        host.update_idletasks()
        host_w = host.winfo_width()
        host_h = host.winfo_height()

    if host_w < min_px or host_h < min_px:
        host_w = max(host_w, widget.winfo_width())
        host_h = max(host_h, widget.winfo_height())

    if host_w < min_px or host_h < min_px:
        return False

    width = max(min_px, host_w - pad_l - pad_r)
    height = max(min_px, host_h - pad_t - pad_b)

    if _TRAJ_FIG_HEIGHT_SCALE < 1.0:
        height = max(min_px, int(height * _TRAJ_FIG_HEIGHT_SCALE))

    dpi = fig.get_dpi()
    fig.set_size_inches(width / dpi, height / dpi, forward=True)

    # Let matplotlib map the widget's pixel size to the figure: under Tk display
    # scaling (Windows 125%/150% DPI) it divides by the device-pixel-ratio so
    # the Agg buffer still renders at the widget's true pixel size and fills it.
    try:
        canvas.resize(type("E", (), {"width": int(width), "height": int(height)})())
    except (TypeError, AttributeError, tk.TclError):
        pass

    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    widget.configure(bg=PANEL, highlightthickness=0)

    try:
        widget.place_forget()
    except tk.TclError:
        pass
    widget.grid(
        row=0,
        column=0,
        sticky=_ANALYSIS_GRAPH_CANVAS_STICKY,
        padx=(pad_l, pad_r),
        pady=(pad_t, pad_b),
    )

    from stablewalk.ui.viewers.dof_trajectory_3d import relayout_single_dof_viewport

    relayout_single_dof_viewport(ax)
    return True


def _bind_trajectory_figure_resize(
    canvas: FigureCanvasTkAgg,
    fig: Figure,
    ax,
    *,
    min_px: int = 40,
    graph_host: tk.Misc | None = None,
    extra_hosts: tuple[tk.Misc, ...] | None = None,
) -> None:
    """Resize the selected-point 3D figure when the canvas or host changes size."""

    def _reflow(_event: object | None = None) -> None:
        if _fit_trajectory_figure(
            canvas,
            fig,
            ax,
            min_px=min_px,
            graph_host=graph_host,
        ):
            canvas.draw_idle()

    def _on_resize(event: tk.Event) -> None:
        if event.width < min_px or event.height < min_px:
            return
        _reflow()

    widget = canvas.get_tk_widget()
    widget.configure(bg=PANEL, highlightthickness=0)
    widget.bind("<Configure>", _on_resize)

    host = graph_host if graph_host is not None else widget.master
    if host is not None and host is not widget:
        host.bind("<Configure>", _on_resize)
    inner = getattr(host, "master", None)
    if inner is not None and inner is not host and inner is not widget:
        inner.bind("<Configure>", _on_resize)
    for extra in extra_hosts or ():
        if extra is not None and extra is not host and extra is not widget:
            extra.bind("<Configure>", _on_resize)


def _metric_grid_cell(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    title: str,
    compact: bool = False,
    accent: bool = False,
) -> tuple[tk.Label, tk.Label]:
    """One titled value cell inside a metrics grid (roomy, readable cards)."""
    pad = 2 if compact else 2
    cell_pad = (0, 0) if compact else (1, 1)
    title_pad = (1, 0) if compact else (2, 0)
    value_pad = (0, 1) if compact else (0, 3)
    inner_x = 4 if compact else 8
    grid_kwargs = dict(row=row, column=column, sticky="nsew", padx=pad, pady=cell_pad)

    if accent and not compact:
        shell = tk.Frame(parent, bg=ORANGE, highlightthickness=0)
        shell.grid(**grid_kwargs)
        cell = tk.Frame(shell, bg=ELEVATED, highlightthickness=0)
        cell.pack(fill=tk.BOTH, expand=True, padx=(3, 0))
    else:
        cell = tk.Frame(
            parent,
            bg=PANEL if compact else ELEVATED,
            highlightthickness=0 if compact else 1,
            highlightbackground=BORDER,
            highlightcolor=BORDER,
        )
        cell.grid(**grid_kwargs)

    cell_bg = PANEL if compact else ELEVATED
    title_lbl = tk.Label(
        cell,
        text=title,
        bg=cell_bg,
        fg=MUTED,
        font=FONT_METRIC_TITLE if not compact else FONT_UI_XS,
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=inner_x, pady=title_pad)
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=cell_bg,
        fg=TEXT,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=inner_x, pady=value_pad)
    return title_lbl, value_lbl


def _metric_slot_host(title_lbl: tk.Label, grid_parent: tk.Misc) -> tk.Misc:
    """Outer grid cell for a metric slot (handles accent shell wrapper)."""
    cell = title_lbl.master
    if cell.master is grid_parent:
        return cell
    return cell.master


def _foot_card_metric(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    title: str,
    compact: bool = False,
) -> tuple[tk.Label, tk.Label]:
    """One secondary metric inside the Foot Analysis card."""
    cell = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    cell.grid(row=row, column=column, sticky="nsew", padx=3, pady=2)
    title_font = FONT_UI_XS
    value_font = FONT_UI_XS if compact else FONT_METRIC
    title_lbl = tk.Label(
        cell,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=title_font,
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=5, pady=(4, 0))
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=value_font,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=5, pady=(0, 4))
    return title_lbl, value_lbl


def _foot_card_row(
    parent: tk.Misc,
    *,
    row: int,
    label: str,
    compact_value: bool = False,
) -> tk.Label:
    """One label + value pair inside the Foot Analysis card."""
    cell = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    cell.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 4))
    tk.Label(
        cell,
        text=label,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(anchor="w")
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_XS if compact_value else FONT_METRIC,
        anchor="w",
        wraplength=_ANALYSIS_LEFT_PANEL_WIDTH - 20,
        justify=tk.LEFT,
    )
    value_lbl.pack(anchor="w", pady=(1, 0))
    return value_lbl


def _foot_card_inline_pair(
    parent: tk.Misc,
    *,
    row: int,
    left_label: str,
    right_label: str,
) -> tuple[tk.Label, tk.Label]:
    """Frame and Time on one row."""
    row_frame = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    row_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 8))
    row_frame.columnconfigure(0, weight=1)
    row_frame.columnconfigure(1, weight=1)

    def _cell(col: int, title: str) -> tk.Label:
        cell = tk.Frame(row_frame, bg=ELEVATED, highlightthickness=0)
        cell.grid(row=0, column=col, sticky="ew")
        tk.Label(
            cell,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(anchor="w")
        lbl = tk.Label(
            cell,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_XS,
            anchor="w",
        )
        lbl.pack(anchor="w", pady=(1, 0))
        return lbl

    return _cell(0, left_label), _cell(1, right_label)


def _build_ground_clearance_strip(gui, parent: tk.Misc) -> tk.Frame:
    """Compact bilateral foot readout under the skeleton view."""
    from stablewalk.ui.theme import FONT_METRIC_VALUE, FONT_UI_XS

    strip = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    gui.ground_clearance_strip = strip

    inner = tk.Frame(strip, bg=PANEL, highlightthickness=0)
    inner.pack(fill=tk.X, padx=2, pady=2)

    tk.Label(inner, text="L", bg=PANEL, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, 2)
    )
    gui.lbl_ground_clearance_left = tk.Label(
        inner,
        text="\u2014",
        bg=PANEL,
        fg=ORANGE,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    gui.lbl_ground_clearance_left.pack(side=tk.LEFT, padx=(0, 12))

    tk.Label(inner, text="R", bg=PANEL, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, 2)
    )
    gui.lbl_ground_clearance_right = tk.Label(
        inner,
        text="\u2014",
        bg=PANEL,
        fg=ORANGE,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    gui.lbl_ground_clearance_right.pack(side=tk.LEFT)

    # Phase / scale kept for logic and tooltips — not shown in the main strip.
    gui.lbl_ground_clearance_phase = tk.Label(strip, text="", bg=PANEL)
    gui.lbl_ground_clearance_scale = tk.Label(strip, text="", bg=PANEL)

    strip.grid_remove()
    return strip


def _build_foot_analysis_card(gui, parent: tk.Misc) -> tk.Frame:
    """Session min/max/avg clearance — current clearance lives in the metrics row."""
    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    card = tk.Frame(
        host,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    card.grid(row=0, column=0, sticky="ew")
    card.columnconfigure(0, weight=1)
    gui.dof_analysis_foot_card = card

    tk.Label(
        card,
        text="Session clearance",
        bg=ELEVATED,
        fg=TEXT,
        font=("Segoe UI Semibold", 9),
        anchor="w",
    ).grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

    clearance_grid = tk.Frame(card, bg=ELEVATED)
    clearance_grid.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 2))
    for col in range(3):
        clearance_grid.columnconfigure(col, weight=1)

    _, gui.foot_card_min_lbl = _foot_card_metric(
        clearance_grid, row=0, column=0, title="Min", compact=True
    )
    _, gui.foot_card_max_lbl = _foot_card_metric(
        clearance_grid, row=0, column=1, title="Max", compact=True
    )
    _, gui.foot_card_avg_lbl = _foot_card_metric(
        clearance_grid, row=0, column=2, title="Average", compact=True
    )

    gui.lbl_foot_ground_note = tk.Label(
        card,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=_ANALYSIS_LEFT_PANEL_WIDTH - 16,
    )
    gui.lbl_foot_ground_note.grid(row=2, column=0, sticky="ew", padx=8, pady=(4, 8))

    gui.dof_analysis_foot_host = host
    gui.lbl_foot_clearance_hero = None
    gui.lbl_foot_clearance_quality = None
    gui.lbl_foot_clearance_definition = None
    gui.lbl_foot_distance_ground = None
    gui.lbl_foot_contact_status = None
    gui.foot_card_point_lbl = None
    gui.lbl_foot_analysis_explanation = gui.lbl_foot_ground_note
    gui.lbl_dof_analysis_foot_title = None
    gui.foot_analysis_secondary_slots = []
    gui.lbl_foot_card_speed = None
    gui.lbl_foot_card_position = None
    gui.foot_card_frame_lbl = None
    gui.foot_card_time_lbl = None
    host.grid_remove()
    return host


def _build_metric_grid(
    parent: tk.Misc,
    *,
    columns: int,
    slot_titles: tuple[str, ...],
    row: int = 0,
    compact: bool = False,
    accent_first: bool = False,
    accent_index: int | None = None,
) -> list[tuple[tk.Label, tk.Label]]:
    """Create a row of metric cells and return (title, value) label pairs."""
    accent_col = accent_index if accent_index is not None else (0 if accent_first else -1)
    for col in range(columns):
        weight = 2 if col == 0 and accent_first else 1
        parent.columnconfigure(col, weight=weight, uniform="metric")
    slots: list[tuple[tk.Label, tk.Label]] = []
    for col, title in enumerate(slot_titles):
        slots.append(
            _metric_grid_cell(
                parent,
                row=row,
                column=col,
                title=title,
                compact=compact,
                accent=accent_col >= 0 and col == accent_col,
            )
        )
    return slots


def _card_title(label: str) -> str:
    """Consistent LabelFrame title spacing."""
    return f" {label.strip()} "


def _sidebar_rule(parent: tk.Misc) -> None:
    tk.Frame(parent, bg=BORDER, height=1).pack(fill=tk.X, pady=(2, 2))


def _analysis_sidebar_heading(parent: tk.Misc, text: str, *, first: bool = False) -> None:
    tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_UI_XS,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(0 if first else PAD_XS, 2))


def _analysis_sidebar_line(
    parent: tk.Misc,
    text: str,
    *,
    color: str | None = None,
    wrap: int = 160,
) -> tk.Label:
    lbl = tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=color or TEXT_SECONDARY,
        font=FONT_UI_XS,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=wrap,
    )
    lbl.pack(fill=tk.X, pady=(0, 0))
    return lbl


def _analysis_sidebar_metric(
    parent: tk.Misc,
    title: str,
    *,
    wrap: int = 160,
) -> tuple[tk.Label, tk.Label]:
    row = tk.Frame(parent, bg=PANEL)
    row.pack(fill=tk.X, pady=(0, 1))
    title_lbl = tk.Label(
        row,
        text=title,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor=tk.W,
    )
    title_lbl.pack(anchor=tk.W)
    value_lbl = tk.Label(
        row,
        text="—",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        justify=tk.LEFT,
        anchor=tk.W,
        wraplength=wrap,
    )
    value_lbl.pack(anchor=tk.W, fill=tk.X)
    return title_lbl, value_lbl


def _analysis_sidebar_section(parent: tk.Misc, text: str, *, first: bool = False) -> None:
    tk.Label(
        parent,
        text=text,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor=tk.W,
    ).pack(fill=tk.X, pady=(0 if first else 4, 1))


def _build_compact_graph_legend(parent: tk.Misc) -> list[tk.Label]:
    """Inline color legend: green start, blue path, red current."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_PANEL_LINE_BLUE,
        DOF_ANALYSIS_PANEL_LINE_GREEN,
        DOF_ANALYSIS_PANEL_LINE_RED,
        DOF_TRAJ_DOT_COLOR,
        DOF_TRAJ_PATH_COLOR,
        DOF_TRAJ_START_COLOR,
    )

    try:
        bg = parent.cget("bg")
    except tk.TclError:
        bg = ELEVATED

    items = (
        (DOF_TRAJ_START_COLOR, DOF_ANALYSIS_PANEL_LINE_GREEN),
        (DOF_TRAJ_PATH_COLOR, DOF_ANALYSIS_PANEL_LINE_BLUE),
        (DOF_TRAJ_DOT_COLOR, DOF_ANALYSIS_PANEL_LINE_RED),
    )
    labels: list[tk.Label] = []
    for index, (color, name) in enumerate(items):
        if index:
            tk.Label(parent, text="·", bg=bg, fg=MUTED, font=FONT_UI_XS).pack(
                side=tk.LEFT, padx=(4, 4)
            )
        chip = tk.Frame(parent, bg=bg)
        chip.pack(side=tk.LEFT)
        tk.Label(chip, text="●", bg=bg, fg=color, font=FONT_UI_XS).pack(side=tk.LEFT)
        lbl = tk.Label(chip, text=name, bg=bg, fg=TEXT_SECONDARY, font=FONT_UI_XS)
        lbl.pack(side=tk.LEFT, padx=(1, 0))
        labels.append(lbl)
    return labels


def _build_analysis_graph_explainer(gui, parent: tk.Misc) -> tk.Frame:
    """Compact side panel beside the 3D trajectory cube."""
    from stablewalk.ui.theme import (
        DOF_TRAJ_DOT_COLOR,
        DOF_TRAJ_PATH_COLOR,
        DOF_TRAJ_START_COLOR,
    )

    container = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    container.grid(row=0, column=1, sticky="nsew", padx=(PAD_SM, 0), pady=(2, 2))
    container.columnconfigure(0, weight=1)

    panel = tk.Frame(container, bg=PANEL, highlightthickness=0)
    panel.grid(row=0, column=0, sticky="new")
    panel.columnconfigure(0, weight=1)

    floor_card = tk.Frame(panel, bg=ELEVATED, highlightthickness=0)
    floor_card.grid(row=0, column=0, sticky="ew", pady=(0, 4))
    gui.dof_graph_floor_card = floor_card
    tk.Label(
        floor_card,
        text="Floor distance",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(fill=tk.X, padx=6, pady=(4, 0))
    gui.lbl_dof_graph_floor_value = tk.Label(
        floor_card,
        text="\u2014",
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_METRIC_VALUE,
        anchor="w",
        justify=tk.LEFT,
        wraplength=_ANALYSIS_EXPLAIN_WIDTH - 16,
    )
    gui.lbl_dof_graph_floor_value.pack(fill=tk.X, padx=6, pady=(0, 4))
    gui.lbl_dof_graph_floor_note = tk.Label(floor_card, text="", bg=ELEVATED)
    gui.lbl_dof_graph_floor_range = tk.Label(floor_card, text="", bg=ELEVATED)

    overlay_row = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    overlay_row.grid(row=1, column=0, sticky="ew", pady=(0, 2))
    gui.var_foot_clearance_graph = tk.StringVar(value="Off")
    gui.cmb_foot_clearance_graph = ttk.Combobox(
        overlay_row,
        textvariable=gui.var_foot_clearance_graph,
        values=("Off", "Left foot", "Right foot"),
        state="readonly",
        width=12,
    )
    gui.cmb_foot_clearance_graph.pack(side=tk.LEFT, fill=tk.X, expand=True)
    gui.cmb_foot_clearance_graph.bind(
        "<<ComboboxSelected>>",
        lambda _e: gui._refresh_foot_clearance_graph_hint(),
    )
    gui.lbl_foot_clearance_graph_range = tk.Label(panel, text="", bg=PANEL)

    legend = tk.Frame(panel, bg=PANEL, highlightthickness=0)
    legend.grid(row=2, column=0, sticky="ew", pady=(2, 0))
    legend_items = (
        (DOF_TRAJ_START_COLOR, "Start"),
        (DOF_TRAJ_PATH_COLOR, "Path"),
        (DOF_TRAJ_DOT_COLOR, "Now"),
    )
    gui.dof_graph_explain_legend_labels = []
    for index, (color, name) in enumerate(legend_items):
        if index:
            tk.Label(legend, text="\u00b7", bg=PANEL, fg=MUTED, font=FONT_UI_XS).pack(
                side=tk.LEFT, padx=2
            )
        chip = tk.Frame(legend, bg=PANEL, highlightthickness=0)
        chip.pack(side=tk.LEFT)
        tk.Label(chip, text="\u25cf", bg=PANEL, fg=color, font=FONT_UI_XS).pack(
            side=tk.LEFT, padx=(0, 2)
        )
        lbl = tk.Label(chip, text=name, bg=PANEL, fg=TEXT_SECONDARY, font=FONT_UI_XS)
        lbl.pack(side=tk.LEFT)
        gui.dof_graph_explain_legend_labels.append(lbl)

    gui.lbl_dof_graph_explain_body = tk.Label(
        panel,
        text="",
        bg=PANEL,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
    )

    from stablewalk.ui.theme import DOF_ANALYSIS_GRAPH_CAPTION_GENERAL

    gui.lbl_dof_analysis_graph_caption = tk.Label(
        panel,
        text=DOF_ANALYSIS_GRAPH_CAPTION_GENERAL,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_UI_XS,
        anchor="w",
    )

    gui.btn_export_analysis = ttk.Button(
        panel,
        text="Export Data",
        style="Compact.TButton",
        width=11,
        command=gui._export_analysis_data,
        state=tk.DISABLED,
    )
    gui.btn_export_analysis.grid(row=5, column=0, sticky="ew", pady=(6, 0))
    gui.dof_analysis_export_row = panel

    gui.dof_analysis_graph_explainer = panel
    return panel


def _build_analysis_graph_chrome(gui, parent: tk.Misc) -> tk.Frame:
    """Reserved graph chrome host (legend lives in the summary bar)."""
    chrome = tk.Frame(parent, bg=PANEL)
    gui.dof_graph_legend_labels = getattr(gui, "dof_graph_legend_labels", [])
    gui.lbl_dof_graph_annotation = None
    gui.lbl_dof_graph_ground = None
    gui.dof_graph_chrome = chrome
    gui.lbl_dof_analysis_graph_caption = None
    return chrome


def _build_analysis_graph_caption(gui, parent: tk.Misc) -> tk.Label:
    """Short header shown above the 3D trajectory canvas ("3D point movement")."""
    from stablewalk.ui.theme import DOF_ANALYSIS_GRAPH_CAPTION_GENERAL

    lbl = tk.Label(
        parent,
        text=DOF_ANALYSIS_GRAPH_CAPTION_GENERAL,
        bg=PANEL,
        fg=ACCENT_ALT,
        font=FONT_PANEL_HEADER,
        justify=tk.LEFT,
        anchor=tk.W,
    )
    gui.lbl_dof_analysis_graph_caption = lbl
    lbl.grid(row=0, column=0, sticky="w")
    lbl.grid_remove()
    return lbl


def _build_analysis_sidebar(gui, parent: tk.Misc) -> tk.Frame:
    """Left column: color legend for the 3D movement graph."""
    from stablewalk.ui.selected_point_analysis import legend_lines_for_panel
    from stablewalk.ui.theme import DOF_ANALYSIS_LEGEND_TITLE

    sidebar = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    sidebar.columnconfigure(0, weight=1)

    gui.dof_analysis_legend_labels: list[tk.Label] = []

    _analysis_sidebar_section(sidebar, DOF_ANALYSIS_LEGEND_TITLE, first=True)
    legend_host = tk.Frame(sidebar, bg=PANEL)
    legend_host.pack(fill=tk.X, anchor=tk.NW, padx=2, pady=(0, 0))
    for line, accent in legend_lines_for_panel():
        lbl = _analysis_sidebar_line(legend_host, line, color=accent, wrap=200)
        gui.dof_analysis_legend_labels.append(lbl)

    def _sync_sidebar_wrap(_event: object | None = None) -> None:
        width = sidebar.winfo_width()
        if width < 48:
            return
        line_wrap = max(96, width - 8)
        for lbl in getattr(gui, "dof_analysis_legend_labels", []):
            try:
                lbl.configure(wraplength=line_wrap)
            except tk.TclError:
                pass

    sidebar.bind("<Configure>", _sync_sidebar_wrap)
    gui._sync_analysis_sidebar_wrap = _sync_sidebar_wrap

    gui.dof_analysis_sidebar = sidebar
    gui.dof_analysis_legend_section = legend_host
    gui.dof_analysis_movement_slots = []
    return sidebar


def _build_analysis_metrics_panel(gui, parent: tk.Misc) -> tk.Frame:
    """Primary metrics row + collapsible advanced measurements."""
    from stablewalk.ui.theme import (
        DOF_ANALYSIS_MODE_FOOT,
        DOF_ANALYSIS_MODE_FOOT_HINT,
        DOF_ANALYSIS_MODE_GENERAL,
        DOF_ANALYSIS_MODE_GENERAL_HINT,
    )

    section = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    section.columnconfigure(0, weight=1)

    mode_holder = tk.Frame(section, bg=PANEL, highlightthickness=0)
    gui.lbl_dof_analysis_mode = tk.Label(mode_holder, text="", bg=PANEL, fg=ACCENT)
    gui.lbl_dof_analysis_mode_hint = tk.Label(mode_holder, text="", bg=PANEL, fg=MUTED)
    gui._analysis_mode_labels = (
        DOF_ANALYSIS_MODE_GENERAL,
        DOF_ANALYSIS_MODE_GENERAL_HINT,
        DOF_ANALYSIS_MODE_FOOT,
        DOF_ANALYSIS_MODE_FOOT_HINT,
    )

    summary_bar = tk.Frame(section, bg=PANEL, highlightthickness=0)
    summary_bar.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    summary_bar.columnconfigure(0, weight=1)

    metrics_host = tk.Frame(summary_bar, bg=PANEL, highlightthickness=0)
    metrics_host.grid(row=0, column=0, sticky="ew")
    metrics_host.columnconfigure(0, weight=1)

    primary_host = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    primary_host.grid(row=0, column=0, sticky="ew")
    gui.dof_analysis_summary_slots = _build_metric_grid(
        primary_host,
        columns=_ANALYTICS_SUMMARY_SLOTS,
        row=0,
        slot_titles=(
            "Selected point",
            "Time",
            "Speed",
            "Path length",
        ),
        compact=True,
        accent_first=True,
    )
    gui.dof_analysis_point_value_lbl = gui.dof_analysis_summary_slots[0][1]
    gui.dof_analysis_point_value_lbl.configure(
        wraplength=150,
        justify=tk.LEFT,
        anchor="w",
    )
    gui.dof_analysis_identity_slots = gui.dof_analysis_summary_slots[:2]
    gui.dof_analysis_kinematics_slots = gui.dof_analysis_summary_slots[2:]
    gui.dof_analysis_summary_hosts = [
        _metric_slot_host(title_lbl, primary_host)
        for title_lbl, _ in gui.dof_analysis_summary_slots
    ]

    toggle_row = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    toggle_row.grid(row=1, column=0, sticky="ew", pady=(2, 0))
    gui.btn_toggle_joint_advanced = ttk.Button(
        toggle_row,
        text="Detailed Joint Data \u25be",
        style="Compact.TButton",
        command=gui._toggle_joint_advanced_data,
    )
    gui.btn_toggle_joint_advanced.pack(side=tk.LEFT)
    gui.btn_collected_data = ttk.Button(
        toggle_row,
        text="Data (0)",
        style="Compact.TButton",
        command=gui._open_collected_data_dialog,
    )
    gui.btn_collected_data.pack(side=tk.RIGHT)
    gui._joint_advanced_visible = False

    advanced_host = tk.Frame(metrics_host, bg=PANEL, highlightthickness=0)
    advanced_host.grid(row=2, column=0, sticky="ew", pady=(2, 0))
    advanced_host.grid_remove()
    gui.dof_analysis_advanced_host = advanced_host

    gui.dof_analysis_advanced_coord_slots = _build_metric_grid(
        advanced_host,
        columns=_ANALYTICS_ADVANCED_COORD_SLOTS,
        row=0,
        slot_titles=("Frame", "X (m)", "Y (m)", "Z (m)"),
        compact=True,
    )
    gui.dof_analysis_advanced_coord_hosts = [
        _metric_slot_host(title_lbl, advanced_host)
        for title_lbl, _ in gui.dof_analysis_advanced_coord_slots
    ]

    secondary_host = tk.Frame(advanced_host, bg=PANEL, highlightthickness=0)
    secondary_host.grid(row=1, column=0, sticky="ew", pady=(PAD_XS, 0))
    gui.dof_analysis_secondary_host = secondary_host
    gui.dof_analysis_derived_row = secondary_host
    gui.dof_analysis_derived_slots = _build_metric_grid(
        secondary_host,
        columns=_ANALYTICS_SECONDARY_SLOTS,
        row=0,
        slot_titles=(
            "Path Length",
            "Delta from Start",
            "Vertical Position",
            "",
        ),
        compact=True,
    )
    gui.dof_analysis_secondary_hosts = [
        _metric_slot_host(title_lbl, secondary_host)
        for title_lbl, _ in gui.dof_analysis_derived_slots
    ]

    gui.dof_graph_legend_labels = getattr(gui, "dof_graph_legend_labels", [])
    gui.btn_export_analysis = getattr(gui, "btn_export_analysis", None)
    gui.lbl_dof_analysis_movement_title = tk.Label(mode_holder, text="", bg=PANEL)

    gui.dof_analysis_metric_slots = (
        gui.dof_analysis_summary_slots
        + gui.dof_analysis_advanced_coord_slots
        + gui.dof_analysis_derived_slots
    )
    gui.dof_analysis_metrics_section = section
    gui.dof_analysis_header = section
    gui.dof_analysis_summary = summary_bar
    gui.dof_analysis_base_summary = summary_bar
    gui.dof_analysis_base_slots = gui.dof_analysis_kinematics_slots
    gui.dof_analysis_context_slot = None
    gui.dof_analysis_joint_summary = None
    gui.dof_analysis_joint_slots = []
    gui.dof_analysis_slots = gui.dof_analysis_metric_slots
    gui.dof_analysis_status = None
    gui.dof_analysis_export_row = None
    gui.dof_analysis_details_row = None
    gui.lbl_dof_analysis_clarification = None
    gui.lbl_dof_analysis_guide = None
    gui.lbl_dof_analysis_red_marker_clarification = None
    gui.lbl_dof_analysis_interpretation = None
    gui.lbl_dof_analysis_metrics_link = None
    gui.lbl_dof_analysis_panel_summary = None
    gui.lbl_dof_analysis_movement_summary = None
    gui.dof_analysis_graph_pad = _ANALYSIS_GRAPH_INNER_PAD

    return section


def _build_analysis_derived_row(gui, parent: tk.Misc) -> tk.Frame:
    """Legacy hook — secondary metrics row is built inside the summary panel."""
    return getattr(gui, "dof_analysis_secondary_host", parent)


def _analysis_metric_cell(
    parent: tk.Misc,
    *,
    row: int,
    column: int,
    title: str,
    gui,
    attr: str,
) -> None:
    """One titled value cell in the selected-point analysis summary grid."""
    cell = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    cell.grid(row=row, column=column, sticky="nsew", padx=2, pady=2)
    title_lbl = tk.Label(
        cell,
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    title_lbl.pack(fill=tk.X, padx=PAD_XS, pady=(PAD_XS, 0))
    value_lbl = tk.Label(
        cell,
        text="—",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_MONO_SM,
        anchor="w",
    )
    value_lbl.pack(fill=tk.X, padx=PAD_XS, pady=(0, PAD_XS))
    if not hasattr(gui, "dof_analysis_slots"):
        gui.dof_analysis_slots = []
    gui.dof_analysis_slots.append((title_lbl, value_lbl))
    setattr(gui, attr, value_lbl)


# Saved inspection points (Movement Checkpoints)
CHECKPOINT_COLUMNS: tuple[str, ...] = (
    "idx",
    "frame",
    "time",
    "dof",
    "joint",
    "x",
    "y",
    "z",
    "velocity",
    "angle",
)
CHECKPOINT_HEADINGS: dict[str, str] = {
    "idx": "#",
    "frame": "Frame",
    "time": "Time (s)",
    "dof": "Point",
    "joint": "Joint",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "velocity": "Velocity",
    "angle": "Angle",
}
CHECKPOINT_WIDTHS: dict[str, int] = {
    "idx": 34,
    "frame": 52,
    "time": 60,
    "dof": 88,
    "joint": 88,
    "x": 58,
    "y": 58,
    "z": 58,
    "velocity": 72,
    "angle": 58,
}


def _make_data_tree(
    parent: ttk.Frame,
    columns: tuple[str, ...],
    headings: dict[str, str],
    *,
    col_widths: dict[str, int] | None = None,
    height: int = 6,
    text_cols: frozenset[str] = frozenset({"dof", "joint", "direction"}),
    style: str = "Large.Treeview",
) -> ttk.Treeview:
    """Scrollable data table with horizontal and vertical bars."""
    host = ttk.Frame(parent)
    host.pack(fill=tk.BOTH, expand=True)
    host.columnconfigure(0, weight=1)
    host.rowconfigure(0, weight=1)

    tree = ttk.Treeview(
        host,
        columns=columns,
        show="headings",
        height=height,
        style=style,
    )
    widths = col_widths or {}
    for col in columns:
        tree.heading(col, text=headings.get(col, col))
        w = widths.get(col, 72)
        anchor = tk.W if col in text_cols else tk.E
        stretch = col in {"dof", "contact_status"}
        tree.column(col, width=w, anchor=anchor, minwidth=36, stretch=stretch)
    tree._sw_column_widths = dict(widths)  # type: ignore[attr-defined]
    tree.grid(row=0, column=0, sticky="nsew")

    vsb = ttk.Scrollbar(host, orient=tk.VERTICAL, command=tree.yview)
    hsb = ttk.Scrollbar(host, orient=tk.HORIZONTAL, command=tree.xview)
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

    tree.tag_configure("even", background=PANEL)
    tree.tag_configure("odd", background=ELEVATED)
    tree.tag_configure("selected", background="#1e3a4f", foreground=TEXT)
    tree._sw_host = host  # type: ignore[attr-defined]
    return tree


# Combined width budget shared by the video and skeleton columns; the sidebar
# keeps its own fixed weight so only these two reflow with the clip.
_TOP_SPAN_WEIGHT = _TOP_VIDEO_WEIGHT + _TOP_SKELETON_WEIGHT
_TOP_VIDEO_WEIGHT_MIN = 30
_TOP_VIDEO_WEIGHT_MAX = _TOP_SPAN_WEIGHT - 28


def apply_top_row_aspect(gui, aspect: float | None) -> None:
    """Keep a stable video/skeleton column split.

    Video frames are scaled with contain-fit inside the video panel; changing
    column weights for portrait clips made the skeleton column swallow width
    and left the video panel too narrow on wide displays.
    """
    body = getattr(gui, "_dashboard_body", None)
    if body is None:
        return

    video_weight = _TOP_VIDEO_WEIGHT
    if getattr(gui, "_top_video_weight", None) == video_weight:
        return
    gui._top_video_weight = video_weight

    skeleton_weight = _TOP_SPAN_WEIGHT - video_weight
    host = getattr(gui, "_primary_viz_host", None)
    try:
        if host is not None:
            host.columnconfigure(0, weight=video_weight, uniform="viz")
            host.columnconfigure(1, weight=skeleton_weight, uniform="viz")
        else:
            body.columnconfigure(0, weight=video_weight, uniform="top")
            body.columnconfigure(1, weight=skeleton_weight, uniform="top")
    except tk.TclError:
        pass


def build_dashboard_layout(gui) -> None:
    """Build the main dashboard grid and attach widgets to ``gui``."""
    from stablewalk.ui.tk.dashboard_responsive import (
        finalize_responsive_dashboard,
        install_responsive_shell,
    )

    # Hidden host for widgets kept for logic/menu but not shown on the dashboard.
    if not hasattr(gui, "_sidebar_hidden"):
        gui._sidebar_hidden = ttk.Frame(gui.root)

    main = ttk.Frame(gui.root, padding=(PAD_SM, PAD_XS, PAD_SM, 14))
    main.pack(fill=tk.BOTH, expand=True)
    gui._dashboard_main = main

    scroll_host = install_responsive_shell(gui, main)
    body = ttk.Frame(scroll_host)
    body.pack(fill=tk.BOTH, expand=True)
    body.rowconfigure(0, weight=_DASH_VIZ_ROW_WEIGHT, minsize=_DASH_VIZ_ROW_MINSIZE)
    body.rowconfigure(1, weight=_DASH_ANALYTICS_ROW_WEIGHT, minsize=_DASH_ANALYTICS_ROW_MINSIZE)
    body.rowconfigure(2, weight=0, minsize=0)
    body.columnconfigure(0, weight=_TOP_VIDEO_WEIGHT + _TOP_SKELETON_WEIGHT, uniform="top")
    body.columnconfigure(1, weight=0, minsize=0)
    body.columnconfigure(
        2,
        weight=_SIDEBAR_WEIGHT,
        uniform="top",
        minsize=_SIDEBAR_MIN_WIDTH,
    )
    # Keep a handle so the video/skeleton split can adapt to the loaded clip's
    # aspect ratio (see ``apply_top_row_aspect``): portrait videos otherwise
    # leave wide empty bands beside the video while the skeleton stays cramped.
    gui._dashboard_body = body
    gui._top_video_weight = _TOP_VIDEO_WEIGHT

    # ── Row 0: video · skeleton (inside host) · sidebar ───────────────────
    primary_viz = ttk.Frame(body)
    primary_viz.grid(row=0, column=0, columnspan=2, sticky="nsew")
    primary_viz.columnconfigure(0, weight=_TOP_VIDEO_WEIGHT, uniform="viz")
    primary_viz.columnconfigure(1, weight=_TOP_SKELETON_WEIGHT, uniform="viz")
    primary_viz.rowconfigure(0, weight=1)
    primary_viz.rowconfigure(1, weight=1)
    gui._primary_viz_host = primary_viz
    gui._primary_viz_content_row = 0
    gui._viz_tabs_active = False

    viz_tab_bar = ttk.Frame(primary_viz)
    viz_tab_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_XS))
    viz_tab_bar.grid_remove()
    gui._viz_tab_bar = viz_tab_bar
    gui._viz_tab_var = tk.StringVar(value="video")

    def _on_viz_tab_selected() -> None:
        from stablewalk.ui.tk.dashboard_responsive import apply_viz_tab_visibility

        apply_viz_tab_visibility(gui)

    for tab_label, tab_value in (("Video", "video"), ("3D Reconstruction", "3d")):
        ttk.Radiobutton(
            viz_tab_bar,
            text=tab_label,
            variable=gui._viz_tab_var,
            value=tab_value,
            command=_on_viz_tab_selected,
        ).pack(side=tk.LEFT, padx=(0, 4))

    video_frame = ttk.LabelFrame(
        primary_viz,
        text=_card_title("Original Video"),
        style="Card.TLabelframe",
        padding=DASHBOARD_VIZ_CARD_PAD,
    )
    video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER))
    gui.video_frame = video_frame
    video_frame.columnconfigure(0, weight=1)
    video_frame.rowconfigure(0, weight=0)
    video_frame.rowconfigure(1, weight=0)
    video_frame.rowconfigure(2, weight=1)

    gui.lbl_demo_analysis_title = tk.Label(
        video_frame,
        text="",
        bg=ELEVATED,
        fg=ACCENT,
        font=FONT_UI_SM,
        anchor="w",
        justify=tk.LEFT,
        padx=4,
        pady=2,
    )
    gui.lbl_demo_analysis_title.grid(row=0, column=0, sticky="ew")
    gui.lbl_demo_analysis_title.grid_remove()

    demo_meta_row = tk.Frame(video_frame, bg=ELEVATED, highlightthickness=0)
    demo_meta_row.grid(row=1, column=0, sticky="ew")
    demo_meta_row.columnconfigure(0, weight=1)
    demo_meta_row.grid_remove()
    gui._demo_meta_row = demo_meta_row

    gui.lbl_demo_source_attribution = tk.Label(
        demo_meta_row,
        text="",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        padx=4,
        pady=0,
    )
    gui.lbl_demo_source_attribution.grid(row=0, column=0, sticky="ew")

    gui.btn_demo_video_info = ttk.Button(
        demo_meta_row,
        text="i",
        style="Compact.TButton",
        width=2,
        command=gui._show_demo_video_details,
    )
    gui.btn_demo_video_info.grid(row=0, column=1, sticky="e", padx=(2, 0))
    gui.btn_demo_video_info.grid_remove()

    gui.video_label = tk.Label(video_frame, text=EMPTY_VIDEO_TEXT)
    configure_video_placeholder(gui.video_label)
    gui.video_label.grid(row=2, column=0, sticky="nsew")

    gui._demo_overlay = tk.Label(video_frame, text="")
    configure_demo_overlay(gui._demo_overlay)

    gui.fig = Figure(figsize=(5.2, 3.0), dpi=100, facecolor=PANEL)
    gui.ax_chart = gui.fig.add_subplot(111)
    gui.ax_chart.set_facecolor(PANEL)
    gui.chart_canvas = FigureCanvasTkAgg(gui.fig, master=video_frame)
    gui._chart_widget = gui.chart_canvas.get_tk_widget()
    gui._chart_widget.configure(bg=PANEL, highlightthickness=0)
    gui._init_chart()

    skel_frame = ttk.LabelFrame(
        primary_viz,
        text=_card_title("3D Gait Reconstruction"),
        style="Card.TLabelframe",
        padding=_SKELETON_PANEL_PAD,
    )
    skel_frame.grid(row=0, column=1, sticky="nsew", padx=(0, DASHBOARD_GUTTER))
    skel_frame.columnconfigure(0, weight=1)
    skel_frame.rowconfigure(0, weight=1)
    skel_frame.rowconfigure(1, weight=0)
    gui.skel_frame = skel_frame

    gui.skel_canvas_host = tk.Frame(skel_frame, bg=PANEL, highlightthickness=0)
    gui.skel_canvas_host.grid(row=0, column=0, sticky="nsew")
    gui.skel_canvas_host.columnconfigure(0, weight=1)
    gui.skel_canvas_host.rowconfigure(0, weight=1)

    skel_toolbar = tk.Frame(gui.skel_canvas_host, bg=PANEL, highlightthickness=0)
    skel_toolbar.place(relx=1.0, rely=0.0, anchor="ne", x=-2, y=2)
    gui.skeleton_display_mode = tk.StringVar(
        value=MODE_TO_SKELETON_LABEL[DEFAULT_SKELETON_DISPLAY_MODE]
    )
    gui.cmb_skeleton_mode = ttk.Combobox(
        skel_toolbar,
        textvariable=gui.skeleton_display_mode,
        values=SKELETON_MODE_LABELS,
        state="readonly",
        width=14,
    )
    gui.cmb_skeleton_mode.pack(side=tk.RIGHT)
    gui.cmb_skeleton_mode.bind("<<ComboboxSelected>>", gui._on_skeleton_display_mode)
    gui.lbl_skeleton_status = None

    clearance_badges = tk.Frame(gui.skel_canvas_host, bg=PANEL, highlightthickness=0)
    clearance_badges.place(relx=0.0, rely=0.0, anchor="nw", x=2, y=2)

    tk.Label(clearance_badges, text="L", bg=PANEL, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, 2)
    )
    gui.lbl_ground_clearance_left = tk.Label(
        clearance_badges,
        text="\u2014",
        bg=PANEL,
        fg=ORANGE,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    gui.lbl_ground_clearance_left.pack(side=tk.LEFT, padx=(0, 8))
    tk.Label(clearance_badges, text="R", bg=PANEL, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(0, 2)
    )
    gui.lbl_ground_clearance_right = tk.Label(
        clearance_badges,
        text="\u2014",
        bg=PANEL,
        fg=ORANGE,
        font=FONT_METRIC_VALUE,
        anchor="w",
    )
    gui.lbl_ground_clearance_right.pack(side=tk.LEFT)
    gui.ground_clearance_strip = clearance_badges
    gui.lbl_ground_clearance_phase = tk.Label(clearance_badges, text="", bg=PANEL)
    gui.lbl_ground_clearance_scale = tk.Label(clearance_badges, text="", bg=PANEL)

    gui.fig_3d = Figure(figsize=(5.6, 5.0), dpi=100, facecolor=PANEL)
    gui.ax_3d = gui.fig_3d.add_subplot(111)
    setup_skeleton_axes(gui.ax_3d)
    _layout_skeleton_figure(gui.ax_3d)
    gui.canvas_3d = FigureCanvasTkAgg(gui.fig_3d, master=gui.skel_canvas_host)
    skel_canvas = gui.canvas_3d.get_tk_widget()
    skel_canvas.configure(bg=PANEL, highlightthickness=0)
    skel_canvas.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=(_SKELETON_CANVAS_PAD[0], _SKELETON_CANVAS_PAD[2]),
        pady=(_SKELETON_CANVAS_PAD[1], _SKELETON_CANVAS_PAD[3]),
    )
    _bind_skeleton_figure_resize(gui.canvas_3d, gui.fig_3d, gui.ax_3d)
    gui.canvas_3d.mpl_connect("pick_event", gui._on_skeleton_pick)
    gui.canvas_3d.mpl_connect("motion_notify_event", gui._on_skeleton_motion)

    # ── Row 1 (wide) / 2 (compact): scrollable selected-point analysis ────
    analysis_scroll_outer = ttk.Frame(body)
    analysis_scroll_outer.grid(row=1, column=0, columnspan=3, sticky="nsew")
    gui._analysis_scroll_row = 1
    analysis_scroll_outer.columnconfigure(0, weight=1)
    analysis_scroll_outer.rowconfigure(0, weight=1)
    gui._analysis_scroll_outer = analysis_scroll_outer

    analysis_canvas = tk.Canvas(
        analysis_scroll_outer,
        bg=BG,
        highlightthickness=0,
        borderwidth=0,
    )
    analysis_canvas.grid(row=0, column=0, sticky="nsew")
    analysis_vsb = ttk.Scrollbar(analysis_scroll_outer, orient=tk.VERTICAL, command=analysis_canvas.yview)
    analysis_vsb.grid(row=0, column=1, sticky="ns")
    analysis_canvas.configure(yscrollcommand=analysis_vsb.set)
    gui._analysis_scroll_canvas = analysis_canvas
    gui._analysis_scrollbar = analysis_vsb

    analysis_inner = ttk.Frame(analysis_canvas)
    gui._analysis_scroll_inner = analysis_inner
    gui._analysis_scroll_window_id = analysis_canvas.create_window(
        (0, 0), window=analysis_inner, anchor="nw",
    )

    def _sync_analysis_scroll(_event: object | None = None) -> None:
        try:
            analysis_canvas.update_idletasks()
            analysis_canvas.configure(scrollregion=analysis_canvas.bbox("all"))
            cw = analysis_canvas.winfo_width()
            if cw > 1:
                analysis_canvas.itemconfigure(gui._analysis_scroll_window_id, width=cw)
        except tk.TclError:
            pass

    analysis_inner.bind("<Configure>", _sync_analysis_scroll)
    analysis_canvas.bind("<Configure>", _sync_analysis_scroll)
    gui._sync_analysis_scroll = _sync_analysis_scroll

    def _analysis_wheel(event: tk.Event) -> None:
        if event.delta:
            analysis_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    for widget in (analysis_inner, analysis_canvas):
        widget.bind("<Enter>", lambda _e: analysis_canvas.bind_all("<MouseWheel>", _analysis_wheel))
        widget.bind("<Leave>", lambda _e: analysis_canvas.unbind_all("<MouseWheel>"))

    bottom_row = ttk.Frame(analysis_inner)
    bottom_row.pack(fill=tk.BOTH, expand=True)
    bottom_row.columnconfigure(
        0,
        weight=1,
        minsize=_ANALYTICS_ANALYSIS_MINSIZE,
    )
    bottom_row.rowconfigure(0, weight=1)
    bottom_row.rowconfigure(1, weight=0)
    gui._dashboard_bottom_row = bottom_row
    gui.dashboard = bottom_row

    traj_panel = ttk.LabelFrame(
        bottom_row,
        text=_card_title("Selected Joint Analysis"),
        style="Card.TLabelframe",
        padding=_ANALYSIS_PANEL_PAD,
    )
    traj_panel.grid(row=0, column=0, sticky="nsew", padx=(0, DASHBOARD_GUTTER))
    gui.traj_panel = traj_panel
    traj_panel.columnconfigure(0, weight=1)
    traj_panel.rowconfigure(0, weight=0)
    traj_panel.rowconfigure(1, weight=1, minsize=200)
    traj_panel.rowconfigure(2, weight=0)

    gui.dof_analysis_header = _build_analysis_metrics_panel(gui, traj_panel)
    gui.dof_analysis_header.grid(row=0, column=0, sticky="ew")

    gui.dof_analysis_empty_host = tk.Frame(traj_panel, bg=PANEL)
    gui.dof_analysis_empty_host.grid(row=0, column=0, rowspan=2, sticky="nsew")
    gui.dof_analysis_empty_host.columnconfigure(0, weight=1)
    gui.dof_analysis_empty_host.rowconfigure(0, weight=1)

    gui.lbl_dof_analysis_empty = ttk.Label(
        gui.dof_analysis_empty_host,
        text=EMPTY_SELECT_DOF_CHART,
        style="EmptyState.TLabel",
    )
    gui.lbl_dof_analysis_empty.grid(row=0, column=0)

    gui.dof_analysis_body = tk.Frame(traj_panel, bg=PANEL, highlightthickness=0)
    gui.dof_analysis_body.grid(row=1, column=0, sticky="nsew")
    gui.dof_analysis_body.columnconfigure(0, weight=0, minsize=0)
    gui.dof_analysis_body.columnconfigure(1, weight=1)
    gui.dof_analysis_body.rowconfigure(0, weight=1, minsize=120)

    gui.dof_analysis_left = tk.Frame(
        gui.dof_analysis_body, bg=PANEL, highlightthickness=0,
    )
    gui.dof_analysis_left.grid(row=0, column=0, sticky="new", padx=(0, 6))
    gui.dof_analysis_left.grid_remove()
    _build_foot_analysis_card(gui, gui.dof_analysis_left).grid(row=0, column=0, sticky="new")
    gui.dof_analysis_foot_row = None
    gui.dof_analysis_foot_slots = []

    gui.dof_analysis_graph_section = tk.Frame(
        gui.dof_analysis_body, bg=PANEL, highlightthickness=0
    )
    gui.dof_analysis_graph_section.grid(row=0, column=0, columnspan=2, sticky="nsew")
    gui.dof_analysis_graph_section.columnconfigure(0, weight=1)
    gui.dof_analysis_graph_section.rowconfigure(0, weight=0)
    gui.dof_analysis_graph_section.rowconfigure(1, weight=1)
    gui.dof_analysis_graph_section.rowconfigure(2, weight=0)

    gui.lbl_joint_movement_title = tk.Label(
        gui.dof_analysis_graph_section,
        text="Joint Movement",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_joint_movement_title.grid(row=0, column=0, sticky="ew", pady=(0, 2))

    _build_analysis_graph_chrome(gui, gui.dof_analysis_graph_section)
    gui.dof_graph_chrome.grid_remove()

    gui.dof_analysis_graph_summary_slots = []
    gui.dof_analysis_graph_summary_row = None
    gui.dof_analysis_graph_header = None
    gui.dof_analysis_export_row = None
    gui.dof_graph_legend_labels = []

    gui.dof_analysis_graph_inner = tk.Frame(
        gui.dof_analysis_graph_section, bg=PANEL, highlightthickness=0
    )
    gui.dof_analysis_graph_inner.grid(
        row=1,
        column=0,
        sticky="nsew",
    )
    # Left: the 3D cube. Right: the explanation column (legend + description +
    # axis key + floor readout). Both columns carry weight so the leftover width
    # is shared — this keeps the cube area closer to square (so the cube reads
    # large) instead of stranding it in a very wide, mostly-empty band.
    gui.dof_analysis_graph_inner.columnconfigure(0, weight=3, minsize=220)
    gui.dof_analysis_graph_inner.columnconfigure(
        1, weight=1, minsize=_ANALYSIS_EXPLAIN_WIDTH
    )
    gui.dof_analysis_graph_inner.rowconfigure(0, weight=1)

    gui.dof_analysis_graph_canvas_host = tk.Frame(
        gui.dof_analysis_graph_inner, bg=PANEL, highlightthickness=0
    )
    gui.dof_analysis_graph_canvas_host.grid(row=0, column=0, sticky="nsew")
    gui.dof_analysis_graph_canvas_host.columnconfigure(0, weight=1)
    gui.dof_analysis_graph_canvas_host.rowconfigure(0, weight=1)
    # The embedded matplotlib canvas is a Tk widget sized to the figure; if the
    # host propagated that requested width upward it would force the cube column
    # to swallow all the spare width and strand the explainer at its minimum.
    # Pin the host so the inner grid weights (not the figure) govern the split.
    # Host resizes with grid weights; figure size is driven by Configure handlers.
    gui.dof_analysis_graph_canvas_host.grid_propagate(True)
    gui.dof_analysis_graph_canvas_host.pack_propagate(True)

    _build_analysis_graph_explainer(gui, gui.dof_analysis_graph_inner)

    gui.dof_analysis_sidebar = None
    gui.dof_analysis_legend_section = None
    gui.dof_analysis_legend_labels = gui.dof_graph_legend_labels

    gui.lbl_dof_traj_empty = gui.lbl_dof_analysis_empty
    gui.dof_analysis_legend_body = None
    gui.lbl_dof_analysis_tracked = None
    gui.lbl_dof_analysis_direction = None

    gui.fig_dof_traj = Figure(figsize=(7.0, 5.4), dpi=100, facecolor=PANEL)
    gui.ax_dof_traj = gui.fig_dof_traj.add_subplot(111, projection="3d")
    setup_single_dof_trajectory_axes(gui.ax_dof_traj)
    from stablewalk.ui.viewers.dof_trajectory_3d import relayout_single_dof_viewport

    relayout_single_dof_viewport(gui.ax_dof_traj)
    gui.canvas_dof_traj = FigureCanvasTkAgg(
        gui.fig_dof_traj, master=gui.dof_analysis_graph_canvas_host
    )
    traj_canvas = gui.canvas_dof_traj.get_tk_widget()
    traj_canvas.configure(bg=PANEL, highlightthickness=0)
    traj_canvas.grid_remove()

    gui.dof_analysis_header.grid_remove()
    gui.dof_analysis_body.grid_remove()
    try:
        gui.dof_analysis_empty_host.lower(gui.dof_analysis_body)
    except tk.TclError:
        pass
    _bind_trajectory_figure_resize(
        gui.canvas_dof_traj,
        gui.fig_dof_traj,
        gui.ax_dof_traj,
        graph_host=gui.dof_analysis_graph_canvas_host,
        extra_hosts=(
            gui.dof_analysis_graph_inner,
            gui.dof_analysis_graph_section,
            gui.dof_analysis_body,
            gui.traj_panel,
        ),
    )

    def _on_traj_panel_resize(_event: object = None) -> None:
        if not gui.selection.selected:
            return
        after_id = getattr(gui, "_traj_resize_after", None)
        if after_id is not None:
            try:
                gui.root.after_cancel(after_id)
            except tk.TclError:
                pass

        def _reflow() -> None:
            gui._traj_resize_after = None
            if not gui.selection.selected:
                return
            if _fit_trajectory_figure(
                gui.canvas_dof_traj,
                gui.fig_dof_traj,
                gui.ax_dof_traj,
                graph_host=gui.dof_analysis_graph_canvas_host,
            ):
                gui.canvas_dof_traj.draw_idle()

        gui._traj_resize_after = gui.root.after(80, _reflow)

    def _on_viz_row_resize(_event: object = None) -> None:
        fit = getattr(gui, "_fit_skeleton_canvas", None)
        if fit is not None:
            fit()

    traj_panel.bind("<Configure>", _on_traj_panel_resize)
    gui.dof_analysis_graph_canvas_host.bind("<Configure>", _on_traj_panel_resize)
    gui.skel_canvas_host.bind("<Configure>", _on_viz_row_resize)

    gui.lbl_dof_traj_path = gui.dof_analysis_graph_section

    table_panel = ttk.Frame(gui._sidebar_hidden)
    gui.table_panel = table_panel
    gui._table_data_expanded = False
    gui._collected_data_bar = None

    gui.dof_table_display_mode = tk.StringVar(value=DOF_TABLE_MODE_DEFAULT)
    gui.lbl_table_summary = ttk.Label(
        gui._sidebar_hidden,
        text="Data (0 samples)",
    )
    gui.btn_view_table_data = ttk.Button(
        gui._sidebar_hidden,
        text="Data (0 samples)",
        style="Compact.TButton",
        command=gui._toggle_collected_data_table,
    )
    gui.btn_clear_dof_history = ttk.Button(
        gui._sidebar_hidden,
        text="Clear",
        style="Compact.TButton",
        command=gui._clear_dof_table_history,
        state=tk.DISABLED,
    )
    gui.btn_export_tracking = ttk.Button(
        gui._sidebar_hidden,
        text="Export",
        style="Compact.TButton",
        command=gui._export_tracking_data,
        state=tk.DISABLED,
    )
    gui.btn_save_session = ttk.Button(
        gui._sidebar_hidden,
        text="Save Session",
        style="Compact.TButton",
        command=gui._save_session_to_files,
        state=tk.DISABLED,
    )

    table_body = ttk.Frame(table_panel)
    table_body.grid(row=2, column=0, sticky="nsew")
    table_body.grid_remove()
    gui.table_body = table_body
    table_body.columnconfigure(0, weight=1)
    table_body.rowconfigure(0, weight=1)

    gui.lbl_table_empty = ttk.Label(
        table_body,
        text=EMPTY_SELECT_DOF_TABLE,
        style="EmptyState.TLabel",
    )
    gui.lbl_table_empty.pack(fill=tk.BOTH, expand=True)

    gui.dof_pos_tree = _make_data_tree(
        table_body,
        DOF_TABLE_COLUMNS,
        DOF_TABLE_HEADINGS,
        col_widths=_TABLE_COL_WIDTHS_COMPACT,
        height=_TABLE_TREE_HEIGHT,
        text_cols=frozenset({"dof", "contact_status"}),
        style="Compact.Treeview",
    )
    gui.dof_pos_tree.bind("<<TreeviewSelect>>", gui._on_dof_pos_tree_select)

    # Movement Checkpoints — hidden from dashboard; logic kept for optional reuse
    gui._checkpoint_hidden = ttk.Frame(gui.root)
    gui.btn_add_checkpoint = ttk.Button(
        gui._checkpoint_hidden,
        text="Add point (current frame)",
        command=gui._add_checkpoint,
    )
    ttk.Button(
        gui._checkpoint_hidden,
        text="Clear points",
        command=gui._clear_checkpoints,
    )
    gui.lbl_checkpoint_status = ttk.Label(
        gui._checkpoint_hidden,
        text="No points added yet",
    )
    gui.lbl_checkpoint_empty = ttk.Label(
        gui._checkpoint_hidden,
        text="No checkpoints yet — select body points and click 'Add point'.",
    )
    gui.checkpoint_tree = _make_data_tree(
        gui._checkpoint_hidden,
        CHECKPOINT_COLUMNS,
        CHECKPOINT_HEADINGS,
        col_widths=CHECKPOINT_WIDTHS,
        height=6,
        text_cols=frozenset({"dof", "joint"}),
    )

    # ── Utility sidebar (compact, spans both rows on wide screens) ────────
    from stablewalk.ui.tk.dashboard_responsive import _install_sidebar_scroll

    sidebar = ttk.Frame(body, style="Card.TFrame", padding=(PAD_XS, PAD_XS))
    sidebar.grid(row=0, column=2, rowspan=1, sticky="nsew", padx=(DASHBOARD_GUTTER, 0))
    sidebar_inner = _install_sidebar_scroll(gui, sidebar)
    gui.sidebar = sidebar

    _side_wrap = max(80, _SIDEBAR_MIN_WIDTH - 24)

    def _sync_sidebar_wrap(_event: object | None = None) -> None:
        width = sidebar.winfo_width()
        if width < 40:
            return
        wrap = max(72, width - 20)
        for attr in (
            "lbl_stab_headline",
            "lbl_stab_metrics",
            "lbl_stab_reason",
            "lbl_opensim_compact_ready",
        ):
            lbl = getattr(gui, attr, None)
            if lbl is not None and hasattr(lbl, "configure"):
                try:
                    lbl.configure(wraplength=wrap)
                except tk.TclError:
                    pass

    sidebar.bind("<Configure>", _sync_sidebar_wrap)
    gui._sync_utility_sidebar_wrap = _sync_sidebar_wrap

    if gui._sidebar_hidden.master is None:
        gui._sidebar_hidden = ttk.Frame(gui.root)
    gui.lbl_session_status = ttk.Label(
        gui._sidebar_hidden,
        text="No session loaded",
        style="SideMuted.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_summary = gui.lbl_session_status
    # The visible sampling selector now lives in the playback controls bar and
    # owns ``gui.refresh_var``; only create a fallback var if it doesn't exist.
    if not hasattr(gui, "refresh_var"):
        gui.refresh_var = tk.StringVar(value=REFRESH_INTERVAL_DEFAULT)
    refresh_combo = ttk.Combobox(
        gui._sidebar_hidden,
        textvariable=gui.refresh_var,
        values=REFRESH_INTERVAL_CHOICES,
        state="readonly",
        width=8,
    )
    refresh_combo.bind("<<ComboboxSelected>>", gui._on_refresh_interval)

    dof_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_dof_panel = dof_panel

    add_row = ttk.Frame(dof_panel)
    add_row.pack(fill=tk.X, pady=(0, PAD_XS))
    add_row.columnconfigure(0, weight=1)
    gui.add_point_var = tk.StringVar(value=ADD_POINT_PLACEHOLDER)
    gui.add_point_combo = ttk.Combobox(
        add_row,
        textvariable=gui.add_point_var,
        state="readonly",
    )
    gui.add_point_combo.grid(row=0, column=0, sticky="ew")
    gui.add_point_combo.bind("<<ComboboxSelected>>", gui._add_point_from_combo)
    ttk.Button(
        add_row,
        text="Add",
        style="CompactAccent.TButton",
        width=5,
        command=gui._add_point_from_combo,
    ).grid(row=0, column=1, sticky="e", padx=(PAD_XS, 0))

    ttk.Label(dof_panel, text="Joints", style="SideMuted.TLabel").pack(
        anchor=tk.W, pady=(0, 2)
    )
    gui.dof_chips_frame = tk.Frame(dof_panel, bg=PANEL, highlightthickness=0)
    gui.dof_chips_frame.pack(fill=tk.X, pady=(0, PAD_XS))

    btn_row = ttk.Frame(dof_panel)
    btn_row.pack(fill=tk.X, pady=(0, 0))
    ttk.Button(
        btn_row,
        text="Clear all",
        style="Compact.TButton",
        command=gui._clear_dof_selection,
    ).pack(side=tk.LEFT)

    # Full checklist kept off-screen for selection state; not shown in sidebar.
    if not hasattr(gui, "_dof_checklist_hidden"):
        gui._dof_checklist_hidden = ttk.Frame(gui._sidebar_hidden)
    list_frame = ttk.Frame(gui._dof_checklist_hidden, style="Card.TFrame", padding=2)
    list_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
    list_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    gui.dof_checkbox_canvas = tk.Canvas(
        list_frame,
        bg=ELEVATED,
        highlightthickness=0,
        borderwidth=0,
        height=_BODY_CHECKLIST_HEIGHT,
    )
    gui.dof_checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    list_scroll.config(command=gui.dof_checkbox_canvas.yview)

    gui.dof_checkboxes_inner = tk.Frame(gui.dof_checkbox_canvas, bg=ELEVATED)
    gui._dof_checkbox_window = gui.dof_checkbox_canvas.create_window(
        (0, 0),
        window=gui.dof_checkboxes_inner,
        anchor="nw",
    )
    gui.dof_checkbox_canvas.configure(yscrollcommand=list_scroll.set)

    gui._dof_checkbox_vars: dict[str, tk.BooleanVar] = {}
    gui._dof_checkbox_rows: dict[str, tk.Frame] = {}
    gui._dof_checkbox_name_labels: dict[str, tk.Label] = {}

    def _dof_checkbox_scroll(_event: object) -> None:
        gui.dof_checkbox_canvas.yview_scroll(int(-1 * (_event.delta / 120)), "units")

    def _dof_inner_configure(_event: object) -> None:
        gui.dof_checkbox_canvas.configure(scrollregion=gui.dof_checkbox_canvas.bbox("all"))
        gui.dof_checkbox_canvas.itemconfigure(
            gui._dof_checkbox_window,
            width=gui.dof_checkbox_canvas.winfo_width(),
        )

    gui.dof_checkboxes_inner.bind("<Configure>", _dof_inner_configure)
    gui.dof_checkbox_canvas.bind("<Configure>", _dof_inner_configure)
    gui.dof_checkbox_canvas.bind("<MouseWheel>", _dof_checkbox_scroll)

    for item_id in GUI_DOF_ITEM_IDS:
        var = tk.BooleanVar(value=False)
        gui._dof_checkbox_vars[item_id] = var
        row_bg = ELEVATED
        row = tk.Frame(gui.dof_checkboxes_inner, bg=row_bg, padx=3, pady=3)
        gui._dof_checkbox_rows[item_id] = row
        tk.Checkbutton(
            row,
            text="",
            variable=var,
            command=lambda i=item_id: gui._on_dof_checkbox_changed(i),
            bg=row_bg,
            fg=TEXT,
            selectcolor=PANEL,
            activebackground=row_bg,
            activeforeground=TEXT,
            anchor="w",
            relief=tk.FLAT,
            highlightthickness=0,
            padx=1,
        ).pack(side=tk.LEFT)
        name_lbl = tk.Label(
            row,
            text=GUI_DOF_LABELS[item_id],
            bg=row_bg,
            fg=TEXT,
            anchor="w",
            font=FONT_UI_SM,
            cursor="hand2",
        )
        name_lbl.pack(fill=tk.X, side=tk.LEFT, expand=True)
        name_lbl.bind(
            "<Button-1>",
            lambda _event, i=item_id: gui._on_dof_item_focus(i),
        )
        gui._dof_checkbox_name_labels[item_id] = name_lbl
        row.pack(fill=tk.X, pady=(0, 1))

    stability_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_stability_panel = stability_panel

    ttk.Label(stability_panel, text="Stability", style="SideMuted.TLabel").pack(
        anchor=tk.W, pady=(0, 2),
    )

    score_row = tk.Frame(stability_panel, bg=PANEL, highlightthickness=0)
    score_row.pack(fill=tk.X, pady=(0, 0))
    gui.lbl_stab_score = tk.Label(
        score_row,
        text="—",
        bg=PANEL,
        fg=MUTED,
        font=FONT_METRIC_VALUE_ACCENT,
        anchor="w",
    )
    gui.lbl_stab_score.pack(side=tk.LEFT)
    tk.Label(
        score_row,
        text="/ 100",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="sw",
    ).pack(side=tk.LEFT, padx=(2, 0), pady=(4, 0))
    gui.lbl_stab_category = tk.Label(
        stability_panel,
        text="No walk analyzed yet",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_stab_category.pack(anchor=tk.W, pady=(2, 4))
    gui.lbl_stab_headline = gui.lbl_stab_category

    stab_compact = tk.Frame(stability_panel, bg=PANEL, highlightthickness=0)
    stab_compact.pack(fill=tk.X, pady=(0, 2))
    stab_compact.columnconfigure(0, weight=1)
    stab_compact.columnconfigure(1, weight=1)

    gui._walk_summary_slots: list[tuple[tk.Label, tk.Label]] = []
    for index, title in enumerate(("Symmetry", "ROM", "Regularity", "Body")):
        row, col = divmod(index, 2)
        cell = tk.Frame(stab_compact, bg=PANEL, highlightthickness=0)
        cell.grid(row=row, column=col, sticky="ew", padx=(0, 6), pady=1)
        title_lbl = tk.Label(
            cell,
            text=title,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        title_lbl.pack(side=tk.LEFT)
        value_lbl = tk.Label(
            cell,
            text="—",
            bg=PANEL,
            fg=TEXT,
            font=FONT_METRIC,
            anchor="w",
        )
        value_lbl.pack(side=tk.LEFT, padx=(4, 0))
        gui._walk_summary_slots.append((title_lbl, value_lbl))
        if index in (1, 3):
            cell.grid_remove()

    gui.lbl_stab_metrics = ttk.Label(stability_panel, text="")
    gui.lbl_stab_metrics.pack_forget()

    gui.lbl_stab_steps = ttk.Label(stability_panel, text="")
    gui.lbl_stab_steps.pack_forget()

    steps_detail_row = ttk.Frame(stability_panel)
    gui.lbl_stab_step_confidence = ttk.Label(steps_detail_row, text="")
    gui.btn_stab_steps_details = ttk.Button(
        steps_detail_row,
        text="Details",
        style="Compact.TButton",
        command=gui._show_gait_summary_details,
    )

    gui.lbl_stab_reason = ttk.Label(stability_panel, text="")
    gui.lbl_stab_reason.pack_forget()

    gui.btn_walk_summary_details = ttk.Button(
        stability_panel,
        text="Details",
        style="Compact.TButton",
        command=gui._show_gait_summary_details,
    )
    gui.btn_walk_summary_details.pack(anchor=tk.W, pady=(2, 0))

    comparison_panel = ttk.Frame(
        gui._sidebar_hidden,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_comparison_panel = comparison_panel
    gui._comparison_expanded = False

    gui.btn_toggle_comparison = ttk.Button(
        sidebar_inner,
        text="Compare Gaits",
        style="Compact.TButton",
        command=gui._open_gait_comparison_dialog,
    )
    gui.btn_compare_gait = gui.btn_toggle_comparison

    gui.comparison_body = ttk.Frame(comparison_panel)
    gui.comparison_body.pack(fill=tk.X)
    gui.comparison_body.pack_forget()

    cmp_tools = ttk.Frame(gui.comparison_body)
    cmp_tools.pack(fill=tk.X, pady=(0, PAD_XS))
    ttk.Button(
        cmp_tools,
        text="Clear",
        style="Compact.TButton",
        command=gui._clear_demo_comparison,
    ).pack(side=tk.RIGHT)

    cmp_body = ttk.Frame(gui.comparison_body)
    cmp_body.pack(fill=tk.X)
    cmp_body.columnconfigure(0, weight=1)
    cmp_body.rowconfigure(0, weight=1)

    gui.demo_comparison_tree = _make_data_tree(
        cmp_body,
        ("demo_type", "joint", "max_angle", "avg_velocity"),
        {
            "demo_type": "Demo",
            "joint": "Joint",
            "max_angle": "Max °",
            "avg_velocity": "Avg v",
        },
        col_widths={
            "demo_type": 72,
            "joint": 64,
            "max_angle": 48,
            "avg_velocity": 52,
        },
        height=4,
        text_cols=frozenset({"demo_type", "joint"}),
        style="Compact.Treeview",
    )

    # ── OpenSim panel: dot status + export; details on demand ─────────────
    opensim_panel = ttk.Frame(
        sidebar_inner,
        style="Card.TFrame",
        padding=DASHBOARD_SIDE_PAD,
    )
    gui._sidebar_opensim_panel = opensim_panel

    opensim_actions = ttk.Frame(opensim_panel)
    opensim_actions.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(2, 0))

    gui.btn_opensim_export = ttk.Button(
        opensim_actions,
        text="Export OpenSim Files",
        style="CompactAccent.TButton",
        command=gui._export_opensim_session,
    )
    gui.btn_opensim_export.pack(fill=tk.X, pady=(0, 2))

    # Experimental IK actions stay in the menu — the sidebar keeps export + status only.
    gui.btn_opensim_run_demo_ik = ttk.Button(
        gui._sidebar_hidden,
        text="Demo IK",
        style="Compact.TButton",
        command=gui._run_opensim_demo_ik,
        state="disabled",
    )

    gui.btn_opensim_run_ik = ttk.Button(
        gui._sidebar_hidden,
        text="Run IK",
        style="Compact.TButton",
        command=gui._run_stablewalk_ik_experimental,
        state="disabled",
    )

    fmt_row = ttk.Frame(gui._sidebar_hidden)
    gui.opensim_motion_fmt = tk.StringVar(value=".mot")
    ttk.Combobox(
        fmt_row,
        textvariable=gui.opensim_motion_fmt,
        values=(".mot", ".csv"),
        state="readonly",
        width=8,
    ).pack(fill=tk.X)

    compact_status = ttk.Frame(opensim_panel)
    compact_status.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
    compact_status.columnconfigure(1, weight=1)

    gui.lbl_opensim_status_dot = tk.Label(
        compact_status,
        text="\u25cf",
        bg=PANEL,
        fg=MUTED,
        font=("Segoe UI", 10),
    )
    gui.lbl_opensim_status_dot.grid(row=0, column=0, sticky="w", padx=(0, 4))

    gui.lbl_opensim_compact_ready = ttk.Label(
        compact_status,
        text="OpenSim",
        style="SideAccent.TLabel",
        cursor="hand2",
    )
    gui.lbl_opensim_compact_ready.grid(row=0, column=1, sticky="w")
    gui.lbl_opensim_compact_ready.bind("<Button-1>", lambda _e: gui._open_opensim_details_dialog())

    gui.lbl_opensim_compact_mode = ttk.Label(compact_status, text="", style="SideMuted.TLabel")
    gui.lbl_opensim_compact_model = ttk.Label(compact_status, text="", style="SideMuted.TLabel")
    gui.lbl_opensim_compact_export = ttk.Label(compact_status, text="", style="SideMuted.TLabel")

    gui.btn_opensim_toggle_details = ttk.Button(
        gui._sidebar_hidden,
        text="Details",
        style="Compact.TButton",
        command=gui._open_opensim_details_dialog,
    )
    gui._opensim_details_visible = False

    gui._opensim_details_frame = ttk.Frame(gui._sidebar_hidden)

    scroll_host = ttk.Frame(gui._opensim_details_frame)
    scroll_host.pack(fill=tk.BOTH, expand=True)
    scroll_host.columnconfigure(0, weight=1)
    scroll_host.rowconfigure(0, weight=1)

    gui._opensim_scrollbar = ttk.Scrollbar(scroll_host, orient=tk.VERTICAL)
    gui._opensim_scrollbar.grid(row=0, column=1, sticky="ns")

    gui._opensim_canvas = tk.Canvas(
        scroll_host,
        bg=PANEL,
        highlightthickness=0,
        borderwidth=0,
        height=OPENSIM_STATUS_SCROLL_HEIGHT,
    )
    gui._opensim_canvas.grid(row=0, column=0, sticky="nsew")
    gui._opensim_canvas.configure(yscrollcommand=gui._opensim_scrollbar.set)
    gui._opensim_scrollbar.configure(command=gui._opensim_canvas.yview)

    opensim_inner = ttk.Frame(gui._opensim_canvas)
    gui._opensim_canvas_window = gui._opensim_canvas.create_window(
        (0, 0),
        window=opensim_inner,
        anchor="nw",
    )

    def _opensim_scroll_region(_event: object | None = None) -> None:
        gui._opensim_canvas.configure(scrollregion=gui._opensim_canvas.bbox("all"))
        gui._opensim_canvas.itemconfigure(
            gui._opensim_canvas_window,
            width=max(gui._opensim_canvas.winfo_width(), max(sidebar.winfo_width(), _SIDEBAR_MIN_WIDTH) - 52),
        )

    def _opensim_mousewheel(event: tk.Event) -> None:
        if event.delta:
            gui._opensim_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    opensim_inner.bind("<Configure>", _opensim_scroll_region)
    gui._opensim_canvas.bind("<Configure>", _opensim_scroll_region)
    for widget in (opensim_inner, gui._opensim_canvas, scroll_host):
        widget.bind(
            "<Enter>",
            lambda _e: gui._opensim_canvas.bind_all("<MouseWheel>", _opensim_mousewheel),
        )
        widget.bind("<Leave>", lambda _e: gui._opensim_canvas.unbind_all("<MouseWheel>"))

    def _section(title: str, *, top_pad: int = 0) -> None:
        if top_pad:
            ttk.Frame(opensim_inner, height=top_pad).pack()
        ttk.Label(opensim_inner, text=title, style="SideAccent.TLabel").pack(
            anchor=tk.W, pady=(0, 3)
        )

    _section("OpenSim Status")
    gui.lbl_opensim_demo_mode = ttk.Label(
        opensim_inner,
        text="Presentation Demo Mode",
        style="SideAccent.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_demo_subtitle = ttk.Label(
        opensim_inner,
        text="Synthetic demo — no video export required",
        style="SideHint.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_demo_note = tk.Label(
        opensim_inner,
        text="",
        bg=SURFACE,
        fg=INFO,
        font=FONT_UI_XS,
        wraplength=_side_wrap,
        justify=tk.LEFT,
        anchor=tk.W,
    )

    gui.lbl_opensim_sdk = ttk.Label(
        opensim_inner, text="SDK: Not installed", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_sdk.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_mode = ttk.Label(
        opensim_inner, text="Mode: Export only", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_mode.pack(anchor=tk.W, pady=(0, 3))

    def _opensim_path_status_label(initial: str) -> tk.Label:
        return tk.Label(
            opensim_inner,
            text=initial,
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_XS,
            wraplength=_side_wrap,
            justify=tk.LEFT,
            anchor=tk.W,
        )

    gui.lbl_opensim_pipeline_model = _opensim_path_status_label(
        "IK pipeline model: —"
    )
    gui.lbl_opensim_pipeline_model.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_suggested_model = _opensim_path_status_label("Suggested model: —")
    gui.lbl_opensim_suggested_model.pack(anchor=tk.W, pady=(0, 2))
    from stablewalk.ui.tk.sidebar_display import (
        LOADED_MODEL_NONE_HINT,
        SUGGESTED_MODEL_NOTE,
    )

    gui.lbl_opensim_suggested_note = ttk.Label(
        opensim_inner,
        text=SUGGESTED_MODEL_NOTE,
        style="SideHint.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_suggested_note.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_loaded_model = _opensim_path_status_label(
        f"Loaded model: {LOADED_MODEL_NONE_HINT}"
    )
    gui.lbl_opensim_loaded_model.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_demo_ik = ttk.Label(
        opensim_inner, text="Demo IK: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_demo_ik.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_stablewalk_ik = ttk.Label(
        opensim_inner, text="StableWalk IK: Not ready", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_stablewalk_ik.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_ik = gui.lbl_opensim_stablewalk_ik

    gui.lbl_opensim_marker_mapping = ttk.Label(
        opensim_inner, text="Mapping: —", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_marker_mapping.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_reliability = tk.Label(
        opensim_inner,
        text="Reliability: Experimental",
        bg=SURFACE,
        fg=ORANGE,
        font=FONT_UI_XS,
        wraplength=_side_wrap,
        justify=tk.LEFT,
        anchor=tk.W,
    )
    gui.lbl_opensim_reliability.pack(anchor=tk.W, pady=(0, PAD_XS))
    gui.lbl_opensim_warning = gui.lbl_opensim_reliability

    ttk.Separator(opensim_inner, orient=tk.HORIZONTAL).pack(
        fill=tk.X, pady=(PAD_XS, PAD_XS)
    )
    _section("Export Status")
    gui.lbl_opensim_trc = ttk.Label(
        opensim_inner, text="TRC: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_trc.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_mot = ttk.Label(
        opensim_inner, text="MOT: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_mot.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_json = ttk.Label(
        opensim_inner, text="JSON: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_json.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_stablewalk_trc = ttk.Label(
        opensim_inner, text="Mapped TRC: Missing", style="SideMuted.TLabel"
    )
    gui.lbl_opensim_stablewalk_trc.pack(anchor=tk.W, pady=(0, 3))
    gui.lbl_opensim_mapped_trc = gui.lbl_opensim_stablewalk_trc

    gui.lbl_opensim_last_export = ttk.Label(
        opensim_inner,
        text="Last Export: Not yet exported",
        style="SideHint.TLabel",
    )
    gui.lbl_opensim_last_export.pack(anchor=tk.W, pady=(0, 2))
    gui.lbl_opensim_last_ik = ttk.Label(
        opensim_inner,
        text="Last IK: Not yet run",
        style="SideHint.TLabel",
    )
    gui.lbl_opensim_last_ik.pack(anchor=tk.W, pady=(0, PAD_XS))

    gui.lbl_opensim_coverage = ttk.Label(opensim_inner, text="")
    gui.lbl_opensim_markers_count = ttk.Label(opensim_inner, text="")

    ttk.Separator(opensim_inner, orient=tk.HORIZONTAL).pack(
        fill=tk.X, pady=(PAD_XS, PAD_XS)
    )
    _section("Details")
    gui.btn_opensim_mapping_report = ttk.Button(
        opensim_inner,
        text="View Mapping Report",
        style="Secondary.TButton",
        command=gui._show_marker_mapping_report,
    )
    gui.btn_opensim_mapping_report.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.btn_opensim_open_output = ttk.Button(
        opensim_inner,
        text="Open Output Folder",
        style="Secondary.TButton",
        command=gui._open_opensim_folder,
    )
    gui.btn_opensim_open_output.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.btn_opensim_view_folder = gui.btn_opensim_open_output
    gui.btn_opensim_folder = gui.btn_opensim_open_output
    gui.btn_opensim_view_log = ttk.Button(
        opensim_inner,
        text="View OpenSim Log",
        style="Secondary.TButton",
        command=gui._show_opensim_log,
    )
    gui.btn_opensim_view_log.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.btn_opensim_details = gui.btn_opensim_view_log

    ttk.Separator(opensim_inner, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, PAD_XS))
    from stablewalk.ui.tk.sidebar_display import (
        OPENSIM_MODEL_DROPDOWN_HINT,
        OPENSIM_SUGGESTED_MODEL_SECTION,
    )

    ttk.Label(
        opensim_inner,
        text=OPENSIM_SUGGESTED_MODEL_SECTION,
        style="SideMuted.TLabel",
    ).pack(anchor=tk.W, pady=(0, 2))
    gui.btn_opensim_select_model = ttk.Button(
        opensim_inner,
        text="Load .osim Model",
        style="Secondary.TButton",
        command=gui._select_opensim_model,
    )
    gui.btn_opensim_select_model.pack(fill=tk.X, pady=(0, PAD_XS))

    local_row = ttk.Frame(opensim_inner)
    local_row.pack(fill=tk.X, pady=(0, PAD_XS))
    gui.opensim_model_var = tk.StringVar(value="")
    gui.opensim_model_combo = ttk.Combobox(
        local_row,
        textvariable=gui.opensim_model_var,
        state="readonly",
    )
    gui.opensim_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
    gui.opensim_model_combo.bind(
        "<<ComboboxSelected>>", gui._on_opensim_suggested_model_combo
    )
    gui.btn_opensim_local_model = ttk.Button(
        local_row,
        text="Select",
        style="Secondary.TButton",
        width=8,
        command=gui._select_discovered_opensim_model,
        state="disabled",
    )
    gui.btn_opensim_local_model.pack(side=tk.LEFT, padx=(PAD_XS, 0))
    gui.lbl_opensim_model_dropdown_hint = ttk.Label(
        opensim_inner,
        text=OPENSIM_MODEL_DROPDOWN_HINT,
        style="SideHint.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_opensim_model_dropdown_hint.pack(anchor=tk.W, pady=(0, PAD_XS))

    gui.lbl_opensim_export_status = gui.lbl_opensim_trc
    gui.lbl_opensim_files = gui.lbl_opensim_trc

    # Point Details — kept for logic but hidden from the main dashboard.
    gui._point_details_hidden = ttk.Frame(gui.root)
    details = ttk.Frame(gui._point_details_hidden)

    gui.lbl_dof_summary = ttk.Label(
        details,
        text="No joint selected",
        style="SideAccent.TLabel",
        wraplength=SIDEBAR_WIDTH - 28,
    )
    gui.lbl_dof_summary.pack(anchor=tk.W, pady=(0, PAD_XS))

    gui.lbl_details_empty = ttk.Label(
        details,
        text="Select points to see live metrics.",
        style="SideEmpty.TLabel",
        wraplength=_side_wrap,
    )
    gui.lbl_details_empty.pack(anchor=tk.W, pady=(PAD_XS, 0))

    gui.dof_details_frame = ttk.Frame(details)
    gui.dof_details_frame.pack(fill=tk.X, pady=(PAD_XS, 0))

    gui.lbl_compare = ttk.Label(
        gui._sidebar_hidden,
        text="",
        style="SideMuted.TLabel",
        wraplength=SIDEBAR_WIDTH - 28,
    )

    gui.dof_pos_tree._sw_host.pack_forget()  # type: ignore[attr-defined]

    finalize_responsive_dashboard(gui)
    _schedule_dashboard_reflow(gui)


def _schedule_dashboard_reflow(gui) -> None:
    """Reflow matplotlib canvases and sidebar wrap after the first layout pass."""

    def _reflow() -> None:
        for attr in ("_sync_analysis_sidebar_wrap", "_sync_utility_sidebar_wrap"):
            sync_wrap = getattr(gui, attr, None)
            if sync_wrap is not None:
                sync_wrap()
        if hasattr(gui, "sidebar"):
            try:
                gui.sidebar.update_idletasks()
            except tk.TclError:
                pass
        if hasattr(gui, "_fit_skeleton_canvas"):
            gui._fit_skeleton_canvas()
        if hasattr(gui, "_fit_dof_traj_canvas"):
            gui._fit_dof_traj_canvas()

    gui.root.after_idle(_reflow)
    gui.root.after(120, _reflow)
