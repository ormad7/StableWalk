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
from typing import Any

from stablewalk import config
from stablewalk.ui.dashboard_interpretability import METRIC_HELP
from stablewalk.ui.scientific_labels import (
    DISCLAIMER_MONOCULAR,
    DISCLAIMER_RESEARCH,
    LABEL_ANALYSIS_CONFIDENCE,
    LABEL_CONTACT_PATTERN,
    LABEL_CONTACT_STATE,
    LABEL_FOOT_CLEARANCE,
    LABEL_FOOT_CLEARANCE_CONFIDENCE,
    LABEL_GAIT_COORDINATION,
    LABEL_MOVEMENT_STABILITY,
)
from stablewalk.ui.tk.analysis_split import (
    MIN_H_SECONDARY_INFO,
    MIN_H_VIDEO_ROW,
    MIN_W_PATH,
    MIN_W_VIDEO,
)
from stablewalk.ui.theme import (
    ACCENT,
    CARD_BODY_PAD,
    CARD_HEADER_PAD,
    CARD_INNER_PAD_X,
    DASHBOARD_CARD_PAD,
    ELEVATED,
    FONT_KPI_SUPPORTING,
    FONT_KPI_TITLE,
    FONT_KPI_VALUE_SM,
    FONT_METRIC,
    FONT_METRIC_VALUE_ACCENT,
    FONT_UI_SM,
    FONT_UI_XS,
    GLYPH_BULLET,
    MUTED,
    MUTED_DIM,
    ORANGE,
    OVERVIEW_ROW_GAP,
    OVERVIEW_ROW_ITEM_GAP,
    OVERVIEW_WRAP_HUD,
    OVERVIEW_WRAP_METRIC,
    OVERVIEW_WRAP_REPORT,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    WARNING,
    bind_responsive_wrap,
    create_elevated_card,
    disclosure_label,
    format_labelframe_title,
    responsive_wraplength,
    tk_labelframe_title,
)

# Section 1 — default Side-by-Side: Video 45% | Skeleton 55%.
SEC1_VIDEO_WEIGHT = 45
SEC1_SKELETON_WEIGHT = 55
SEC1_SUMMARY_WEIGHT = 0

# Main Overview dashboard: Video 40% | Skeleton 30% | Path 30%.
SEC1_TRAJ_VIDEO_WEIGHT = 40
SEC1_TRAJ_SKELETON_WEIGHT = 30
SEC1_TRAJ_PATH_WEIGHT = 30
SEC1_TRAJ_SUMMARY_WEIGHT = 0

# Minimum panel widths so video / skeleton / path stay usable when the window shrinks.
# Soft floors only — leftover space still follows column weights.
SEC1_PANEL_MINWIDTH = MIN_W_VIDEO
# Path card: 3D viewport + compact inspector, without forcing horizontal scroll.
SEC1_TRAJ_PANEL_MINWIDTH = MIN_W_VIDEO
# Legacy fixed width for the former skeleton-side rail (horizontal strip now).
SKELETON_LIVE_INFO_WIDTH = 168

# Overview tab vertical split: visualizations own leftover height; gait info
# sits in a secondary pane (scrollable) and does not steal viewer pixels.
SEC1_VIZ_ROW_WEIGHT = 96
SEC1_METRICS_ROW_WEIGHT = 4
# Joint Graphs bottom strip — collapsed by default (weight 0); expands on demand.
SEC1_JOINT_MOTION_ROW_WEIGHT = 14
SEC1_JOINT_MOTION_ROW_MINSIZE = 28

# Soft minimums for the Overview visualization row — path leads, video/skeleton follow.
SEC1_VIZ_ROW_MINSIZE = MIN_H_VIDEO_ROW
# Soft floor so Video | Skeleton | Path grow with the window instead of
# forcing KPI / chrome rows below the visible area.
SEC1_TRAJ_VIZ_ROW_MINSIZE = MIN_H_VIDEO_ROW
SEC1_METRICS_ROW_MINSIZE = 88
SEC1_METRICS_ROW_MAXSIZE = 120

# Overview tab rows: header, then the primary/secondary vertical splitter.
OVERVIEW_HOST_HEADER_ROW = 0
OVERVIEW_HOST_VIZ_ROW = 1
# Legacy alias — KPI cards now live inside the secondary splitter pane.
OVERVIEW_HOST_KPI_ROW = 2
# Floor for Video / Skeleton / 3D Path so they cannot collapse to height 0.
OVERVIEW_PANEL_MIN_H = MIN_H_VIDEO_ROW
# Gait-phase + Estimated Foot Clearance sit under the viz row (weight 0).
OVERVIEW_GAIT_PHASE_MIN_H = 44
OVERVIEW_FOOT_CLEARANCE_MIN_H = 40
# Short windows keep these readable without stealing Video / Skeleton / Path height.
OVERVIEW_GAIT_PHASE_MIN_H_SHORT = 36
OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT = 32
# Video / Skeleton / Path viewport floor — large enough to see the person,
# a full skeleton, and the 3D path plot + toolbar. Do not subtract chrome.
OVERVIEW_VIDEO_ROW_SOFT_MIN_H = MIN_H_VIDEO_ROW
OVERVIEW_VIDEO_ROW_SOFT_MIN_H_SHORT = 220
OVERVIEW_VIDEO_MIN_W = MIN_W_VIDEO
OVERVIEW_SKELETON_MIN_W = 260
# Wide enough for plot + Coordinate Inspector side-by-side without forcing
# 3×360 columns (that overflows the 1100px three-col breakpoint).
OVERVIEW_PATH_MIN_W = MIN_W_PATH
# Primary metric cards (title + large value + short status). 2×2 wrap needs two rows.
OVERVIEW_KPI_ROW_MIN_H = 108
OVERVIEW_KPI_ROW_MIN_H_WRAP = 200
OVERVIEW_HOST_HEADER_RESERVE_H = 40


def overview_chrome_floors(*, short: bool = False) -> tuple[int, int]:
    """Phase and Foot Clearance floors. Short windows stay compact, not hidden."""
    if short:
        return (
            int(OVERVIEW_GAIT_PHASE_MIN_H_SHORT),
            int(OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT),
        )
    return (int(OVERVIEW_GAIT_PHASE_MIN_H), int(OVERVIEW_FOOT_CLEARANCE_MIN_H))


def overview_viz_chrome_min_h(*, short: bool = False) -> int:
    """Natural height reserved for Phase + Foot Clearance under the media row."""
    phase_h, clearance_h = overview_chrome_floors(short=short)
    return int(phase_h + clearance_h + OVERVIEW_ROW_GAP * 2)


def overview_video_row_min_h(
    panel_floor: int | None = None, *, short: bool = False
) -> int:
    """Usable Video / Skeleton / Path row height. Short windows stay readable."""
    if short:
        base = int(OVERVIEW_VIDEO_ROW_SOFT_MIN_H_SHORT)
        if panel_floor is None:
            return base
        return max(base, min(int(panel_floor), int(OVERVIEW_VIDEO_ROW_SOFT_MIN_H)))
    floor = int(panel_floor if panel_floor is not None else OVERVIEW_PANEL_MIN_H)
    return max(int(OVERVIEW_VIDEO_ROW_SOFT_MIN_H), floor)


def overview_available_viz_h(gui: Any, host: tk.Misc | None = None) -> int:
    """Height intended for Video | Skeleton | Path (primary splitter pane).

    Prefer the vertical sash leftover, not the primary pane's current
    ``winfo_height``. Returning the live pane height locked first-pass
    ~300 px viewers and left a large unused band at 1920×1080.
    """
    split = getattr(gui, "_overview_main_split", None)
    if split is not None:
        try:
            split_h = int(split.winfo_height() or 0)
        except tk.TclError:
            split_h = 0
        if split_h > 160:
            user = bool(getattr(gui, "_overview_v_sash_user", False))
            frac = getattr(gui, "_overview_v_sash_frac", None)
            if user and frac is not None:
                try:
                    return max(120, int(round(float(frac) * split_h)))
                except (TypeError, ValueError):
                    pass
            from stablewalk.ui.tk.dashboard_overview_panes import (
                _auto_overview_primary_height,
            )

            return int(_auto_overview_primary_height(split_h))
    target = host or getattr(gui, "_overview_layout_host", None) or getattr(
        gui, "_overview_content", None
    )
    if target is None:
        return 0
    try:
        host_h = int(target.winfo_height() or 0)
    except tk.TclError:
        return 0
    # Header + a compact secondary sash; leftover belongs to the viewers.
    return max(
        0,
        host_h - int(OVERVIEW_HOST_HEADER_RESERVE_H) - int(MIN_H_SECONDARY_INFO),
    )


def overview_fit_viz_row_floors(
    *,
    short: bool = False,
    layout: str = "three_col",
    available_h: int | None = None,
    panel_floor: int | None = None,
) -> tuple[int, int, int]:
    """Video / phase / clearance minsizes that fit the allocated viz height.

    Prevents the Overview section from growing past the tab and covering
    the bottom playback bar.
    """
    phase_h, clearance_h = overview_chrome_floors(short=short)
    floor = overview_video_row_min_h(panel_floor, short=short)
    if available_h is None or int(available_h) <= 80:
        return floor, phase_h, clearance_h
    # Phase / clearance / KPI live in the secondary pane. The primary pane
    # is viewers only — do not subtract chrome from viewer height.
    room = max(1, int(available_h))
    media_rows = 3 if layout == "stacked" else (2 if layout == "two_plus_one" else 1)
    media_floor = 140
    if media_rows * floor > room:
        floor = max(media_floor, room // max(1, media_rows))
    return max(1, floor), max(1, phase_h), max(1, clearance_h)


def overview_kpi_row_min_h(*, wrap: bool = False) -> int:
    """Reserved height so primary metric cards are not clipped vertically."""
    return int(OVERVIEW_KPI_ROW_MIN_H_WRAP if wrap else OVERVIEW_KPI_ROW_MIN_H)


def apply_overview_host_row_weights(
    host: tk.Misc,
    *,
    viz_row: int = OVERVIEW_HOST_VIZ_ROW,
    short: bool = False,
    kpi_min: int | None = None,
    host_h: int | None = None,
) -> None:
    """Header stays compact. Row 1 is the primary/secondary splitter (expands).

    KPI cards, gait phase, and foot clearance live in the secondary pane —
    they must not reserve host rows that steal viewer height.
    """
    try:
        host.columnconfigure(0, weight=1)
    except tk.TclError:
        return
    video_floor = overview_video_row_min_h(short=short)
    secondary_floor = int(MIN_H_SECONDARY_INFO)
    viz_floor = video_floor + secondary_floor
    if host_h is not None and int(host_h) > 80:
        budget = max(1, int(host_h) - int(OVERVIEW_HOST_HEADER_RESERVE_H))
        viz_floor = min(viz_floor, budget)
        viz_floor = max(video_floor, viz_floor) if budget >= video_floor else max(1, budget)
    kpi_row = int(OVERVIEW_HOST_KPI_ROW)
    _ = kpi_min  # kept for call-site compatibility; KPI is not a host row.
    for row in range(8):
        try:
            if row == int(viz_row):
                host.rowconfigure(row, weight=1, minsize=max(1, viz_floor))
            elif row == kpi_row:
                host.rowconfigure(row, weight=0, minsize=0)
            else:
                host.rowconfigure(row, weight=0, minsize=0)
        except tk.TclError:
            pass


def apply_overview_viz_grid_weights(
    section: tk.Misc,
    *,
    content_row: int = 0,
    layout: str = "three_col",
    short: bool = False,
    video_min_h: int | None = None,
    available_h: int | None = None,
) -> None:
    """Video | Skeleton | 3D Path weights. Never compress below usable floors."""
    floor, phase_h, clearance_h = overview_fit_viz_row_floors(
        short=short,
        layout=layout,
        available_h=available_h,
        panel_floor=video_min_h,
    )
    from stablewalk.ui.tk.analysis_split import overview_pane_minsizes
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    video_min, skel_min, path_min = overview_pane_minsizes(dpi_scale_factor(section))
    video_min = max(int(OVERVIEW_VIDEO_MIN_W), video_min)
    skel_min = max(int(OVERVIEW_SKELETON_MIN_W), skel_min)
    path_min = max(int(OVERVIEW_PATH_MIN_W), path_min)
    try:
        if layout == "stacked":
            section.columnconfigure(0, weight=1, minsize=video_min, uniform="")
            section.columnconfigure(1, weight=0, minsize=0, uniform="")
            section.columnconfigure(2, weight=0, minsize=0, uniform="")
            for offset in range(3):
                section.rowconfigure(int(content_row) + offset, weight=1, minsize=floor)
            section.rowconfigure(int(content_row) + 3, weight=0, minsize=0)
            section.rowconfigure(int(content_row) + 4, weight=0, minsize=0)
            return
        if layout == "two_plus_one":
            section.columnconfigure(0, weight=1, minsize=video_min, uniform="overview_viz_pair")
            section.columnconfigure(1, weight=1, minsize=skel_min, uniform="overview_viz_pair")
            section.columnconfigure(2, weight=0, minsize=0, uniform="")
            section.rowconfigure(int(content_row), weight=1, minsize=floor)
            section.rowconfigure(int(content_row) + 1, weight=1, minsize=floor)
            section.rowconfigure(int(content_row) + 2, weight=0, minsize=0)
            section.rowconfigure(int(content_row) + 3, weight=0, minsize=0)
            return
        section.columnconfigure(
            0,
            weight=SEC1_TRAJ_VIDEO_WEIGHT,
            minsize=video_min,
            uniform="",
        )
        section.columnconfigure(
            1,
            weight=SEC1_TRAJ_SKELETON_WEIGHT,
            minsize=skel_min,
            uniform="",
        )
        section.columnconfigure(
            2,
            weight=SEC1_TRAJ_PATH_WEIGHT,
            minsize=path_min,
            uniform="",
        )
        section.rowconfigure(int(content_row), weight=1, minsize=floor)
        section.rowconfigure(int(content_row) + 1, weight=0, minsize=0)
        section.rowconfigure(int(content_row) + 2, weight=0, minsize=0)
    except tk.TclError:
        pass


def _floor_widget_height(widget: tk.Misc | None, min_h: int) -> None:
    """Keep a mapped widget from sitting at height 0 after a resize."""
    if widget is None:
        return
    floor = max(1, int(min_h))
    try:
        current = int(widget.winfo_height() or 0)
    except tk.TclError:
        current = 0
    if current > 0 and current >= floor:
        return
    requested = 0
    try:
        raw = widget.cget("height")
        if raw not in (None, "", "0"):
            requested = int(str(raw).split(".")[0] or 0)
    except (tk.TclError, TypeError, ValueError):
        requested = 0
    if requested >= floor:
        return
    try:
        widget.configure(height=floor)
    except (tk.TclError, TypeError):
        pass


def apply_overview_panel_min_heights(gui: Any) -> None:
    """Apply height floors to Overview Video, Skeleton, and 3D Path containers."""
    short = bool(getattr(gui, "_overview_short_window", False))
    video_short = bool(getattr(gui, "_overview_video_short", False))
    layout = str(getattr(gui, "_overview_viz_layout", "three_col") or "three_col")
    available_h = overview_available_viz_h(gui)
    preferred = overview_video_row_min_h(short=video_short)
    floor, _phase_fit, _clearance_fit = overview_fit_viz_row_floors(
        short=short,
        layout=layout,
        available_h=available_h if available_h > 80 else None,
        panel_floor=preferred,
    )
    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    content_row = int(getattr(gui, "_primary_viz_content_row", 0))
    if section is not None:
        apply_overview_viz_grid_weights(
            section,
            content_row=content_row,
            layout=layout,
            short=short,
            video_min_h=preferred,
            available_h=available_h if available_h > 80 else None,
        )

    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    traj = getattr(gui, "overview_traj_panel", None)
    for panel in (video, skel, traj):
        _floor_widget_height(panel, floor)
        try:
            if panel is not None:
                panel.grid_configure(sticky="nsew")
        except tk.TclError:
            pass

    # Viewport rows sit under LabelFrame titles + toolbars. Flooring them to
    # the full panel min overflowed the primary pane and clipped the plots.
    viewport_floor = max(120, int(floor) - 88)
    try:
        if video is not None:
            video.rowconfigure(3, weight=1, minsize=viewport_floor)
    except tk.TclError:
        pass
    try:
        if skel is not None:
            skel.rowconfigure(1, weight=1, minsize=viewport_floor)
    except tk.TclError:
        pass
    try:
        if traj is not None:
            traj.rowconfigure(1, weight=1, minsize=viewport_floor)
    except tk.TclError:
        pass

    for host_attr in (
        "video_display_host",
        "skel_canvas_host",
        "overview_traj_canvas_host",
        "_video_clip_canvas",
        "_skel_clip_canvas",
    ):
        _floor_widget_height(getattr(gui, host_attr, None), viewport_floor)

    _floor_widget_height(getattr(gui, "_overview_phase_strip", None), _phase_fit)
    _floor_widget_height(
        getattr(gui, "_foot_clearance_detail_host", None), _clearance_fit
    )


def apply_overview_joint_motion_row_weight(
    parent: tk.Misc,
    *,
    expanded: bool,
    row: int = 2,
) -> None:
    """Give Joint Graphs height only when expanded; zero when collapsed."""
    try:
        if expanded:
            parent.rowconfigure(
                row,
                weight=SEC1_JOINT_MOTION_ROW_WEIGHT,
                minsize=SEC1_JOINT_MOTION_ROW_MINSIZE,
            )
        else:
            parent.rowconfigure(row, weight=0, minsize=0)
    except tk.TclError:
        pass


def overview_panel_padx(column: int, columns: int = 3) -> tuple[int, int]:
    """Symmetric horizontal gutter for Overview visual panels.

    Adjacent panels share ``OVERVIEW_COL_GUTTER`` equally.
    """
    from stablewalk.ui.theme import OVERVIEW_COL_GUTTER

    half = OVERVIEW_COL_GUTTER // 2
    cols = max(1, int(columns))
    col = max(0, int(column))
    left = 0 if col <= 0 else half
    right = 0 if col >= cols - 1 else half
    return (left, right)


def overview_metric_padx(column: int, n_columns: int = 4) -> tuple[int, int]:
    """Even gutters between Overview KPI cards in a row of ``n_columns``."""
    from stablewalk.ui.theme import OVERVIEW_COL_GUTTER

    half = OVERVIEW_COL_GUTTER // 2
    cols = max(1, int(n_columns))
    col = max(0, int(column))
    left = 0 if col <= 0 else half
    right = 0 if col >= cols - 1 else half
    return (left, right)

SECTION_VISUAL_TITLE = "Overview"
SECTION_KINEMATIC_TITLE = "Motion Analysis"
SECTION_GAIT_METRICS_TITLE = "Gait Metrics"
SECTION_REAL_TO_SIM_TITLE = "Real-to-Sim Pipeline"
SECTION_DATA_EXPORT_TITLE = "Data & Export"


def _section_label_frame(parent: tk.Misc, title: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(
        parent,
        text=format_labelframe_title(title),
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
        text=title,
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_KPI_TITLE,
        anchor="w",
    ).pack(side=tk.LEFT)
    if help_key and help_key in METRIC_HELP:
        help_host = tk.Frame(header, bg=ELEVATED)
        help_host.pack(side=tk.LEFT, padx=(PAD_XS, 0))
        add_metric_help_icon(help_host, title, METRIC_HELP[help_key])
    return cell


# Compact clinical HUD under the Original Video panel (Overview).
# Frame/time live on the global playback bar — do not duplicate them here.
_OVERVIEW_VIDEO_INFO_FIELDS: tuple[tuple[str, str, str], ...] = (
    (
        "phase",
        "Phase",
        "Gait phase at the current video frame (contact-derived).",
    ),
    (
        "rom",
        "ROM",
        "Full-session flexion range for the selected joint.",
    ),
)

_OVERVIEW_TRACKING_FIELDS: tuple[tuple[str, str, str], ...] = (
    (
        "detected",
        "Detected frames",
        "Detected frames / analyzed source frames. Source-frame count is the "
        "loaded clip; detected and usable are separate. A 100% detection rate "
        "is not accuracy and is not spectral or biomechanical quality.",
    ),
    (
        "joints",
        "Joint confidence",
        "Mean landmark visibility on detected frames. This is a visibility summary, not joint-presence accuracy.",
    ),
    (
        "usable",
        "Usable frames",
        "Usable gait frames / analyzed source frames. A detected pose is not "
        "automatically usable. The percentage is a usable-frame share, not accuracy.",
    ),
)


def build_overview_playback_hud(gui, host: tk.Misc) -> tk.Frame:
    """Compact HUD: Phase/ROM plus readable tracking fields with tooltips."""
    from stablewalk.ui.theme import create_tooltip

    hud = create_elevated_card(host)
    inner = tk.Frame(hud, bg=ELEVATED, highlightthickness=0)
    inner.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_XS, PAD_SM))
    inner.columnconfigure(0, weight=1)
    inner.columnconfigure(1, weight=1)
    inner.columnconfigure(2, weight=1)

    gui._overview_hud_value_labels = {}
    gui._overview_hud_caption_labels = {}
    gui._overview_hud_cards = {}

    def _add_cell(
        row: int,
        col: int,
        key: str,
        caption: str,
        tip: str,
        *,
        colspan: int = 1,
        last_col: int = 2,
    ) -> None:
        cell = tk.Frame(inner, bg=ELEVATED, highlightthickness=0)
        cell.grid(
            row=row,
            column=col,
            columnspan=colspan,
            sticky="nsew",
            padx=(0 if col == 0 else PAD_SM, 0 if col + colspan - 1 >= last_col else 0),
            pady=(0 if row == 0 else PAD_XS, 0),
        )
        cell.columnconfigure(0, weight=1)
        cap_lbl = tk.Label(
            cell,
            text=caption,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_KPI_TITLE,
            anchor="w",
        )
        cap_lbl.grid(row=0, column=0, sticky="ew")
        value_lbl = tk.Label(
            cell,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_KPI_SUPPORTING,
            anchor="w",
            justify=tk.LEFT,
            wraplength=OVERVIEW_WRAP_HUD,
        )
        value_lbl.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        create_tooltip(cell, tip, wraplength=360)
        create_tooltip(cap_lbl, tip, wraplength=360)
        create_tooltip(value_lbl, tip, wraplength=360)
        gui._overview_hud_caption_labels[key] = cap_lbl
        gui._overview_hud_value_labels[key] = value_lbl
        gui._overview_hud_cards[key] = cell

    for col, (key, caption, tip) in enumerate(_OVERVIEW_VIDEO_INFO_FIELDS):
        _add_cell(0, col, key, caption, tip, last_col=1)
    for col, (key, caption, tip) in enumerate(_OVERVIEW_TRACKING_FIELDS):
        _add_cell(1, col, key, caption, tip, last_col=2)

    gui._overview_playback_hud = hud
    return hud


# Compact selected-joint summary (Frame/Time stay on the root transport bar).
_SKELETON_LIVE_INFO_FIELDS: tuple[tuple[str, str], ...] = (
    ("joint_name", "Selected Joint"),
    ("angle", "Current Angle"),
    ("rom", "Session ROM"),
    ("confidence", "Tracking"),
    ("phase_or_contact", "Phase / Contact"),
    # Kept for refresh compatibility; not shown in the compact strip.
    ("frame", "Frame"),
    ("time", "Time"),
    ("x", "X"),
    ("y", "Y"),
    ("z", "Z"),
)

_SKELETON_LIVE_INFO_VISIBLE: tuple[str, ...] = (
    "joint_name",
    "angle",
    "rom",
    "confidence",
    "phase_or_contact",
)


def build_skeleton_live_info_panel(gui, parent: tk.Misc) -> tk.Frame:
    """Compact selected-joint summary + Open 3D Path (no duplicate playback chrome)."""
    from stablewalk.ui.design_system.components import ExportButton, QuietPanel, SectionHeader
    from stablewalk.ui.theme import create_tooltip
    import stablewalk.ui.design_system.tokens as T

    strip = QuietPanel(parent, bg=ELEVATED)
    strip.columnconfigure(0, weight=1)

    header = tk.Frame(strip, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, PAD_XS))
    header.columnconfigure(0, weight=1)
    SectionHeader(
        header,
        "Selected Joint",
        bg=ELEVATED,
        tooltip="Pick a joint on the skeleton or in the reconstruction toolbar.",
    ).frame.grid(row=0, column=0, sticky="ew")

    actions = tk.Frame(header, bg=ELEVATED, highlightthickness=0)
    actions.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))

    def _toggle_path() -> None:
        show_fn = getattr(gui, "_show_overview_trajectory_dock", None)
        if callable(show_fn):
            # Path stays permanently mapped — this refreshes / remounts it.
            show_fn(True)

    path_btn = ExportButton(
        actions,
        "Refresh 3D Path",
        command=_toggle_path,
        primary=True,
    )
    path_btn.pack(side=tk.RIGHT)
    create_tooltip(
        path_btn,
        "Refresh the always-visible 3D joint trajectory (Catmull-Rom path).",
        wraplength=320,
    )
    gui._btn_overview_open_3d_path = path_btn
    gui._toggle_overview_3d_path = _toggle_path

    body = tk.Frame(strip, bg=ELEVATED, highlightthickness=0)
    body.grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM))
    gui.skeleton_live_info_values: dict[str, tk.Label] = {}

    visible = [
        (key, title)
        for key, title in _SKELETON_LIVE_INFO_FIELDS
        if key in _SKELETON_LIVE_INFO_VISIBLE
    ]
    n = len(visible)
    for col in range(n):
        body.columnconfigure(col, weight=1, uniform="ov_live_strip")

    titles = {key: title for key, title in _SKELETON_LIVE_INFO_FIELDS}
    for col, (key, title) in enumerate(visible):
        cell = tk.Frame(body, bg=ELEVATED, highlightthickness=0)
        cell.grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=(0 if col == 0 else PAD_XS, 0 if col == n - 1 else PAD_XS),
        )
        cell.columnconfigure(0, weight=1)
        tk.Label(
            cell,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=T.FONT_META,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew")
        value_lbl = tk.Label(
            cell,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=T.FONT_CARD_TITLE,
            anchor="w",
            justify=tk.LEFT,
            wraplength=140,
        )
        value_lbl.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        gui.skeleton_live_info_values[key] = value_lbl
        create_tooltip(cell, title, wraplength=260)

    # Hidden compatibility labels for Frame/Time/XYZ (root transport owns time).
    hidden = tk.Frame(strip, bg=ELEVATED, highlightthickness=0)
    for key in ("frame", "time", "x", "y", "z"):
        lbl = tk.Label(hidden, text="—", bg=ELEVATED, fg=MUTED)
        gui.skeleton_live_info_values[key] = lbl
        create_tooltip(lbl, titles.get(key, key), wraplength=200)

    gui.skeleton_live_info_panel = strip
    return strip


def _overview_metric_card(
    parent: tk.Misc,
    *,
    title: str,
    unit: str = "",
    details_key: str | None = None,
    gui: Any | None = None,
    metric_key: str = "",
    compact: bool = True,
) -> dict:
    """Equal-size clinical metric card via shared MetricCard component."""
    from stablewalk.ui.design_system.components import MetricCard
    from stablewalk.ui.overview_clinical_summary import card_description, card_provenance

    # Provenance on-card; full technical description stays in the tooltip.
    provenance = card_provenance(metric_key) if metric_key else ""
    tip = card_description(metric_key) if metric_key else ""
    card = MetricCard(
        parent,
        title=title,
        unit=unit,
        description=provenance,
        details_key=details_key,
        gui=gui,
        tooltip=tip or provenance,
        compact=compact,
    )
    try:
        card["desc"].configure(wraplength=OVERVIEW_WRAP_METRIC)
    except tk.TclError:
        pass
    if compact:
        # Title + value only. Status/provenance live in the tooltip so the
        # four-card strip stays readable instead of wrapping into itself.
        try:
            card["value"].configure(font=FONT_METRIC)
        except tk.TclError:
            pass
        try:
            card["frame"].rowconfigure(1, minsize=36)
        except tk.TclError:
            pass
    return card


def build_overview_recording_context(gui, parent: tk.Misc) -> tk.Frame:
    """Documented source facts plus computed comparison fields for the three examples."""
    from stablewalk.ui.media.demo_comparison import (
        COMPARISON_TREE_COLUMNS,
        COMPARISON_TREE_HEADINGS,
        COMPARISON_TREE_TEXT_COLS,
        COMPARISON_TREE_WIDTHS,
        COMPUTED_PENDING,
        DOCUMENTED_UNAVAILABLE,
    )
    from stablewalk.ui.theme import create_tooltip
    from stablewalk.ui.tk.dashboard_layout import _make_data_tree

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    documented = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    documented.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))
    documented.columnconfigure(1, weight=1)
    documented.columnconfigure(3, weight=1)
    documented.columnconfigure(5, weight=1)

    gui._overview_context_labels = {}
    documented_fields = (
        (0, "source_label", "Source (documented)", "Dataset/protocol label. Not a computed gait score."),
        (2, "camera", "Camera", "Documented camera view from the source record. Blank means not documented."),
        (4, "condition", "Treadmill / overground", "Documented locomotion setting. Blank means not documented."),
    )
    for col, key, title, tip in documented_fields:
        cap = tk.Label(
            documented,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        cap.grid(row=0, column=col, sticky="w", padx=(PAD_SM if col else PAD_SM, PAD_XS), pady=(PAD_XS, 0))
        val = tk.Label(
            documented,
            text=DOCUMENTED_UNAVAILABLE,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=260,
        )
        val.grid(row=1, column=col, columnspan=2, sticky="ew", padx=(PAD_SM if col else PAD_SM, PAD_XS), pady=(0, PAD_XS))
        create_tooltip(cap, tip, wraplength=320)
        create_tooltip(val, tip, wraplength=320)
        gui._overview_context_labels[key] = val

    computed = tk.Frame(host, bg=ELEVATED, highlightthickness=0)
    computed.grid(row=1, column=0, sticky="ew", pady=(0, PAD_XS))
    computed.columnconfigure(1, weight=1)
    computed.columnconfigure(3, weight=1)
    computed.columnconfigure(5, weight=1)

    computed_fields = (
        (0, "duration", "Duration", "Computed from the loaded timestamps. If analysis is truncated, both analyzed and recording length are shown."),
        (2, "cycles", "Usable gait cycles", "Computed usable gait cycles for this load. Not a source label."),
        (4, "tracking", "Tracking coverage", "Computed detection / visibility / usable-frame coverage. Separate from spectral reliability and biomechanical validity. A 100% detection rate is not accurate measurement."),
    )
    for col, key, title, tip in computed_fields:
        cap = tk.Label(
            computed,
            text=title,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        cap.grid(row=0, column=col, sticky="w", padx=(PAD_SM if col else PAD_SM, PAD_XS), pady=(PAD_XS, 0))
        val = tk.Label(
            computed,
            text=COMPUTED_PENDING,
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=260,
        )
        val.grid(row=1, column=col, columnspan=2, sticky="ew", padx=(PAD_SM if col else PAD_SM, PAD_XS), pady=(0, PAD_XS))
        create_tooltip(cap, tip, wraplength=320)
        create_tooltip(val, tip, wraplength=320)
        gui._overview_context_labels[key] = val

    tree_host = tk.Frame(host, bg=PANEL, highlightthickness=0)
    tree_host.grid(row=2, column=0, sticky="ew")
    tree_host.columnconfigure(0, weight=1)
    gui.overview_comparison_tree = _make_data_tree(
        tree_host,
        COMPARISON_TREE_COLUMNS,
        COMPARISON_TREE_HEADINGS,
        col_widths=COMPARISON_TREE_WIDTHS,
        height=3,
        text_cols=COMPARISON_TREE_TEXT_COLS,
        style="Compact.Treeview",
    )
    gui._overview_recording_context = host
    return host


def build_overview_clinical_summary(gui, parent: tk.Misc) -> tk.Frame:
    """Bottom KPI strip: four primary metric cards only (clean Overview default)."""
    from stablewalk.ui.design_system.components import (
        CollapsibleSection,
        QuietPanel,
        SectionHeader,
        WarningBanner,
    )
    from stablewalk.ui.overview_clinical_summary import card_provenance
    import stablewalk.ui.design_system.tokens as T

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.columnconfigure(0, weight=1)

    if not getattr(gui, "_overview_session_badge_labels", None):
        gui._overview_session_badge_labels = {}
    if not getattr(gui, "_overview_session_badge_host", None):
        gui._overview_session_badge_host = None

    context = build_overview_recording_context(gui, host)
    context.grid(row=0, column=0, sticky="ew", pady=(0, PAD_XS))

    # Thin status line — off by default so Video|Skeleton keep the height.
    state_bar = QuietPanel(host, bg=T.PANEL_SECONDARY)
    state_bar.grid(row=1, column=0, sticky="ew", pady=(0, 0))
    state_bar.grid_remove()
    gui.lbl_overview_state = tk.Label(
        state_bar,
        text="Select a recording and run Analyze.",
        bg=T.PANEL_SECONDARY,
        fg=MUTED,
        font=T.FONT_SUPPORTING,
        anchor="w",
        justify=tk.LEFT,
        wraplength=960,
    )
    gui.lbl_overview_state.pack(fill=tk.X, padx=PAD_XS, pady=(2, 2))
    gui._overview_state_bar = state_bar

    dq = WarningBanner(
        host,
        "Data quality warning",
        tone="warn",
        tooltip="Shown when confidence or completeness is insufficient.",
    )
    dq.frame.grid(row=2, column=0, sticky="ew", pady=(0, PAD_SM))
    dq.frame.grid_remove()
    gui._overview_data_quality_banner = dq
    gui.lbl_overview_data_quality = dq.label

    # ── Four primary metric cards — one horizontal row only ─────────────
    metrics = tk.Frame(host, bg=PANEL, highlightthickness=0)
    metrics.grid(row=3, column=0, sticky="nsew", pady=(PAD_XS, PAD_XS))
    host.rowconfigure(3, weight=0)
    primary_fields = (
        ("gait_quality", "Gait quality", "/100", "gait_quality"),
        (
            "stability_state",
            "Stable COM frames",
            "%",
            "stability_margin",
        ),
        ("symmetry", "Symmetry", "%", "symmetry"),
        ("cadence", "Cadence", "steps/min", "cadence"),
    )
    metrics.rowconfigure(0, weight=1)
    for col in range(len(primary_fields)):
        metrics.columnconfigure(col, weight=1, uniform="ov_clin_cards")

    gui._overview_clinical_cards: dict[str, dict] = {}
    gui._overview_clinical_value_labels: dict[str, tk.Label] = {}
    gui._overview_kpi_order = [key for key, *_rest in primary_fields]
    for col, (key, title, unit, details_key) in enumerate(primary_fields):
        card = _overview_metric_card(
            metrics,
            title=title,
            unit=unit,
            details_key=details_key,
            gui=gui,
            metric_key=key,
        )
        card["frame"].grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=overview_metric_padx(col),
            pady=0,
        )
        try:
            card["desc"].configure(text="")
        except tk.TclError:
            pass
        gui._overview_clinical_cards[key] = card
        gui._overview_clinical_value_labels[key] = card["value"]
    gui._overview_kpi_metrics_host = metrics
    gui._overview_kpi_layout = "4col"
    gui._overview_kpi_force_one_row = True

    # ── Advanced details (collapsed by default) ─────────────────────────
    advanced = CollapsibleSection(
        parent, "More details", expanded=False, bg=PANEL
    )
    adv_body = advanced.body
    adv_body.columnconfigure(0, weight=1)
    gui._overview_advanced_section = advanced.frame
    gui._overview_advanced_collapse = advanced
    gui._overview_advanced_body = adv_body

    # Slot for view-mode / overlay chrome moved out of the default surface.
    tools_slot = tk.Frame(adv_body, bg=PANEL, highlightthickness=0)
    tools_slot.grid(row=0, column=0, sticky="ew", pady=(0, PAD_SM))
    tools_slot.columnconfigure(0, weight=1)
    gui._overview_advanced_tools_slot = tools_slot

    interp_panel = QuietPanel(adv_body, bg=ELEVATED)
    interp_panel.grid(row=1, column=0, sticky="ew", pady=(0, PAD_MD))
    SectionHeader(
        interp_panel,
        "Interpretation",
        bg=ELEVATED,
        tooltip="Single-session reading from pose-derived gait indicators.",
    ).frame.pack(fill=tk.X, padx=PAD_MD, pady=(PAD_MD, PAD_SM))
    gui.lbl_overview_primary_finding = tk.Label(
        interp_panel,
        text="Load and analyze a video to see the clinical gait overview.",
        bg=ELEVATED,
        fg=TEXT,
        font=T.FONT_BODY,
        anchor="w",
        justify=tk.LEFT,
        wraplength=OVERVIEW_WRAP_REPORT * 3,
    )
    gui.lbl_overview_primary_finding.pack(
        fill=tk.X, padx=PAD_MD, pady=(0, PAD_MD)
    )

    secondary = tk.Frame(adv_body, bg=PANEL, highlightthickness=0)
    secondary.grid(row=2, column=0, sticky="ew", pady=(0, PAD_MD))
    secondary_fields = (
        ("analysis_confidence", "Confidence", "", "analysis_confidence"),
        ("gait_cycles", "Gait Cycles", "", None),
    )
    for col in range(len(secondary_fields)):
        secondary.columnconfigure(col, weight=1, uniform="ov_clin_secondary")
    gui._overview_secondary_order = [key for key, *_rest in secondary_fields]
    for col, (key, title, unit, details_key) in enumerate(secondary_fields):
        card = _overview_metric_card(
            secondary,
            title=title,
            unit=unit,
            details_key=details_key,
            gui=gui,
            metric_key=key,
            compact=False,
        )
        card["frame"].grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=(0 if col == 0 else PAD_SM, 0),
            pady=0,
        )
        card["desc"].configure(text=card_provenance(key))
        from stablewalk.ui.theme import FONT_KPI_VALUE_SM

        try:
            card["value"].configure(font=FONT_KPI_VALUE_SM)
        except tk.TclError:
            pass
        gui._overview_clinical_cards[key] = card
        gui._overview_clinical_value_labels[key] = card["value"]
    gui._overview_secondary_host = secondary

    report = tk.Frame(adv_body, bg=PANEL, highlightthickness=0)
    report.grid(row=3, column=0, sticky="ew")
    report.columnconfigure(0, weight=3, uniform="ov_clin_report")
    report.columnconfigure(1, weight=2, uniform="ov_clin_report")
    gui._overview_clinical_report = report

    findings_card = QuietPanel(report, bg=ELEVATED)
    findings_card.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_MD))
    findings_card.columnconfigure(0, weight=1)
    SectionHeader(findings_card, "Key Findings", bg=ELEVATED).frame.grid(
        row=0, column=0, sticky="ew", padx=PAD_MD, pady=(PAD_MD, PAD_SM)
    )
    findings_body = tk.Frame(findings_card, bg=ELEVATED, highlightthickness=0)
    findings_body.grid(
        row=1, column=0, sticky="nsew", padx=PAD_MD, pady=(0, PAD_MD)
    )
    findings_body.columnconfigure(0, weight=1)
    gui._overview_summary_card = findings_card
    gui._overview_clinical_bullet_labels: list[tk.Label] = []
    gui._overview_key_finding_rows = []
    for i in range(3):
        row = tk.Frame(findings_body, bg=ELEVATED, highlightthickness=0)
        row.grid(row=i, column=0, sticky="ew", pady=(0, OVERVIEW_ROW_ITEM_GAP))
        row.columnconfigure(1, weight=1)
        tk.Label(
            row,
            text=GLYPH_BULLET,
            bg=ELEVATED,
            fg=ACCENT,
            font=FONT_UI_SM,
            width=2,
            anchor="n",
        ).grid(row=0, column=0, sticky="nw")
        lbl = tk.Label(
            row,
            text="" if i else "Key findings appear after analysis.",
            bg=ELEVATED,
            fg=TEXT if i == 0 else MUTED,
            font=FONT_UI_SM,
            anchor="w",
            justify=tk.LEFT,
            wraplength=OVERVIEW_WRAP_REPORT,
        )
        lbl.grid(row=0, column=1, sticky="ew")
        gui._overview_clinical_bullet_labels.append(lbl)
        gui._overview_key_finding_rows.append(row)

    lim = CollapsibleSection(report, "Key Limitations", expanded=False, bg=PANEL)
    lim.frame.grid(row=0, column=1, sticky="nsew")
    gui._overview_limitations_card = lim.frame
    gui._overview_limitations_collapse = lim
    gui._overview_system_note_labels: list[tk.Label] = []
    for i in range(6):
        lbl = tk.Label(
            lim.body,
            text="" if i else "Method notes appear after analysis.",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="w",
            justify=tk.LEFT,
            wraplength=OVERVIEW_WRAP_REPORT,
        )
        lbl.pack(fill=tk.X, pady=(0, OVERVIEW_ROW_ITEM_GAP))
        gui._overview_system_note_labels.append(lbl)

    # Slot for joint strip / session metrics / disclaimer (filled by layout).
    extras = tk.Frame(adv_body, bg=PANEL, highlightthickness=0)
    extras.grid(row=4, column=0, sticky="ew", pady=(PAD_SM, 0))
    extras.columnconfigure(0, weight=1)
    gui._overview_advanced_extras_slot = extras

    gui._overview_confidence_track = None
    gui._overview_confidence_fill = None
    gui.lbl_overview_confidence_pct = tk.Label(host, text="", bg=PANEL)
    gui.lbl_overview_confidence_pose = tk.Label(host, text="", bg=PANEL)
    gui.lbl_overview_confidence_biomech = tk.Label(host, text="", bg=PANEL)
    gui.lbl_overview_visible_disclaimer = None

    gui._overview_clinical_summary = host
    return host


def build_overview_session_badge(gui, host: tk.Misc) -> tk.Frame:
    """Top summary chips: video, backend, frames, tracking, timestamp."""
    from stablewalk.ui.design_system.components import QuietPanel
    import stablewalk.ui.design_system.tokens as T

    bar = QuietPanel(host, bg=T.PANEL_SECONDARY)
    bar.pack(fill=tk.X)
    inner = tk.Frame(bar, bg=T.PANEL_SECONDARY, highlightthickness=0)
    inner.pack(fill=tk.X, padx=PAD_SM, pady=(PAD_XS, PAD_XS))

    gui._overview_session_badge_labels = {}
    fields = (
        ("video_name", "Video / Session"),
        ("pose_backend", "Analysis Backend"),
        ("analyzed_frames", "Valid Frames"),
        ("tracking_confidence", "Tracking Confidence"),
        ("analysis_timestamp", "Analyzed"),
    )
    for i, (key, title) in enumerate(fields):
        cell = tk.Frame(inner, bg=T.PANEL_SECONDARY, highlightthickness=0)
        cell.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0 if i == 0 else PAD_MD, 0))
        tk.Label(
            cell,
            text=title,
            bg=T.PANEL_SECONDARY,
            fg=MUTED,
            font=T.FONT_META,
            anchor="w",
        ).pack(fill=tk.X)
        val = tk.Label(
            cell,
            text="—",
            bg=T.PANEL_SECONDARY,
            fg=TEXT,
            font=T.FONT_CARD_TITLE,
            anchor="w",
            wraplength=360,
        )
        val.pack(fill=tk.X)
        gui._overview_session_badge_labels[key] = val
    gui._overview_session_badge_host = host
    return bar


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
    # COMPARE banner is redundant with the header demo buttons — keep the
    # widget for API compatibility but never show it on Overview.
    gui.lbl_overview_demo_compare.grid_remove()

    # Start collapsed so the three visualization panels own ~75–80%+ of height.
    gui._overview_metrics_expanded = False

    def _apply_metrics_row_weight(expanded: bool) -> None:
        try:
            if expanded:
                parent.rowconfigure(
                    6, weight=SEC1_METRICS_ROW_WEIGHT, minsize=SEC1_METRICS_ROW_MINSIZE
                )
            else:
                parent.rowconfigure(6, weight=0, minsize=SEC1_METRICS_ROW_MINSIZE)
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
            btn.configure(
                text=disclosure_label("Session Metrics", expanded=expanded)
            )
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
        text=disclosure_label("Session Metrics", expanded=False),
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
    title = tk_labelframe_title(parent, SECTION_GAIT_METRICS_TITLE)
    section = ttk.LabelFrame(
        parent,
        labelwidget=title,
        style="Card.TLabelframe",
        padding=DASHBOARD_CARD_PAD,
    )
    for col in range(2):
        section.columnconfigure(col, weight=1, uniform="gait_metric")

    intro = tk.Label(
        section,
        text="Contact state and cycle completeness from the current analysis.",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        justify=tk.LEFT,
        wraplength=520,
    )
    intro.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, PAD_SM))
    gui.lbl_gait_metrics_intro = intro

    grid_host = tk.Frame(section, bg=PANEL, highlightthickness=0)
    grid_host.grid(row=1, column=0, columnspan=2, sticky="ew")
    for col in range(2):
        grid_host.columnconfigure(col, weight=1, uniform="gait_metric")

    # Foot clearance + phase/contact/cycles live on Overview tab (build_overview_metrics_row).
    contact_card = _gait_metric_card(grid_host, title=LABEL_CONTACT_STATE)
    contact_card.grid(row=0, column=0, sticky="nsew", padx=(0, PAD_SM), pady=0)
    contact_body = tk.Frame(contact_card, bg=ELEVATED)
    contact_body.grid(row=1, column=0, sticky="ew", padx=CARD_INNER_PAD_X, pady=(PAD_XS, PAD_SM))
    for row, side in enumerate(("Left Foot", "Right Foot")):
        tk.Label(
            contact_body, text=f"{side}:", bg=ELEVATED, fg=MUTED, font=FONT_UI_XS, anchor="w"
        ).grid(row=row, column=0, sticky="w", pady=(0, PAD_XS))
        val = tk.Label(
            contact_body,
            text="Not computed yet",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_SM,
            anchor="w",
        )
        val.grid(row=row, column=1, sticky="w", padx=(PAD_SM, 0), pady=(0, PAD_XS))
        attr = "left" if "Left" in side else "right"
        setattr(gui, f"lbl_gait_card_contact_{attr}", val)

    # Gait Cycles
    cycles_card = _gait_metric_card(grid_host, title="Gait Cycles")
    cycles_card.grid(row=0, column=1, sticky="nsew", pady=0)
    gui.lbl_gait_card_cycles_label = tk.Label(
        cycles_card,
        text="Usable cycles:",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_card_cycles_label.grid(
        row=1, column=0, sticky="ew", padx=CARD_INNER_PAD_X, pady=(PAD_SM, 0)
    )
    gui.lbl_gait_card_cycles_value = tk.Label(
        cycles_card,
        text="Not computed yet",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="w",
    )
    gui.lbl_gait_card_cycles_value.grid(
        row=2, column=0, sticky="ew", padx=CARD_INNER_PAD_X, pady=(PAD_XS, PAD_XS)
    )
    gui.lbl_gait_card_completeness_value = tk.Label(
        cycles_card,
        text="Completeness: Available after full analysis",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_gait_card_completeness_value.grid(
        row=3, column=0, sticky="ew", padx=CARD_INNER_PAD_X, pady=(0, PAD_SM)
    )

    btn_row = tk.Frame(section, bg=PANEL, highlightthickness=0)
    btn_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(PAD_SM, 0))
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
    """Overview sidebar — compact Session Metrics dock (presentation only)."""
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
        details_key: str | None = None,
        tier: str = "secondary",
    ):
        tip = METRIC_HELP.get(help_key, title)
        kpi = create_kpi_card(
            host,
            key=key,
            title=title,
            tooltip=tip,
            compact=tier not in ("primary",),
            show_bar=show_bar and tier not in ("technical", "supporting"),
            fill=False,
            details_key=details_key or help_key or key,
            tier=tier,  # type: ignore[arg-type]
        )
        gui._overview_kpi_cards[key] = kpi
        explain = tk.Label(
            host,
            text="",
            bg=PANEL,
            fg=MUTED if tier not in ("technical", "supporting") else MUTED_DIM,
            font=FONT_UI_SM if tier not in ("technical", "supporting") else FONT_UI_XS,
            anchor="w",
            wraplength=220,
            justify=tk.LEFT,
        )
        explain.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_SM if tier not in ("technical", "supporting") else PAD_XS))
        return kpi, explain

    ms_kpi, gui.lbl_summary_ms_explain = _score_kpi(
        key="movement_stability",
        title=LABEL_MOVEMENT_STABILITY,
        help_key="movement_stability",
        tier="primary",
    )
    gui.lbl_summary_ms_value = ms_kpi.value_lbl
    gui.lbl_summary_ms_unit = ms_kpi.unit_lbl

    gq_kpi, gui.lbl_summary_gq_explain = _score_kpi(
        key="gait_quality",
        title=LABEL_GAIT_COORDINATION,
        help_key="gait_quality",
        details_key="gait_coordination",
        tier="primary",
    )
    gui.lbl_summary_gq_value = gq_kpi.value_lbl
    gui.lbl_summary_gq_unit = gq_kpi.unit_lbl
    # Evidence badge reuses the KPI quality line (same information, KPI styling).
    gui.lbl_summary_gq_badge = gq_kpi.quality_lbl
    gui.lbl_summary_gq_badge.configure(fg=WARNING)

    ac_kpi, gui.lbl_summary_ac_explain = _score_kpi(
        key="analysis_confidence",
        title=LABEL_ANALYSIS_CONFIDENCE,
        help_key="analysis_confidence",
        show_bar=True,
        tier="secondary",
    )
    gui.lbl_summary_ac_level = ac_kpi.value_lbl
    gui.lbl_summary_ac_unit = ac_kpi.unit_lbl

    gui.lbl_overview_science_disclaimer = tk.Label(
        host,
        text=f"{DISCLAIMER_RESEARCH}  {DISCLAIMER_MONOCULAR}",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
        wraplength=240,
        justify=tk.LEFT,
    )
    gui.lbl_overview_science_disclaimer.pack(fill=tk.X, padx=PAD_XS, pady=(PAD_XS, 0))

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
    gui.lbl_summary_demo_compare.pack(fill=tk.X, padx=PAD_XS, pady=(PAD_XS, 0))

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

    def _wrap_summary_copy(_event: object | None = None) -> None:
        try:
            width = max(72, int(host.winfo_width() or 0) - 16)
        except tk.TclError:
            return
        if width < 72:
            return
        for attr in (
            "lbl_summary_ms_explain",
            "lbl_summary_gq_explain",
            "lbl_summary_ac_explain",
            "lbl_overview_science_disclaimer",
            "lbl_summary_demo_compare",
        ):
            lbl = getattr(gui, attr, None)
            if lbl is None:
                continue
            try:
                lbl.configure(wraplength=width)
            except tk.TclError:
                pass

    host.bind("<Configure>", _wrap_summary_copy, add="+")
    return host


def build_foot_clearance_detail_panel(gui, parent: tk.Misc) -> tk.Frame:
    """Overview Estimated Foot Clearance: two persistent Left/Right rows."""
    from stablewalk.ui.theme import create_tooltip

    host = tk.Frame(
        parent,
        bg=PANEL,
        highlightthickness=0,
    )
    host.columnconfigure(0, weight=1)
    host.rowconfigure(0, weight=0, minsize=20)
    # Two foot rows + column headers stay inside the existing chrome floor.
    host.rowconfigure(
        1, weight=0, minsize=max(22, int(OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT) - 10)
    )

    header = tk.Frame(host, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(2, 0))
    header.columnconfigure(1, weight=1)

    title = tk.Label(
        header,
        text=LABEL_FOOT_CLEARANCE,
        bg=PANEL,
        fg=TEXT,
        font=FONT_KPI_TITLE,
        anchor="w",
    )
    title.grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))
    create_tooltip(
        title,
        "Estimated vertical foot-to-ground distance from monocular pose data.",
        wraplength=360,
    )

    gui.lbl_foot_clearance_confidence = tk.Label(
        header,
        text=f"{LABEL_FOOT_CLEARANCE_CONFIDENCE}: —",
        bg=PANEL,
        fg=MUTED_DIM,
        font=FONT_UI_XS,
        anchor="w",
    )
    gui.lbl_foot_clearance_confidence.grid(row=0, column=1, sticky="w")

    gui.btn_foot_clearance_details = ttk.Button(
        header,
        text="Details",
        style="Compact.TButton",
        command=gui._show_foot_clearance_details,
    )
    gui.btn_foot_clearance_details.grid(row=0, column=2, sticky="e")

    gui.lbl_foot_clearance_subtitle = tk.Label(
        header,
        text="cm",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    )
    gui.lbl_foot_clearance_subtitle.grid(row=0, column=3, sticky="e", padx=(PAD_SM, 0))

    table = tk.Frame(host, bg=PANEL, highlightthickness=0)
    table.grid(row=1, column=0, sticky="nsew", padx=PAD_SM, pady=(0, 2))
    table.columnconfigure(0, weight=0, minsize=68)
    for col in (1, 2, 3, 4):
        table.columnconfigure(col, weight=1, minsize=64)

    titles = (
        "Current",
        "Min swing",
        "Max swing",
        "State",
    )
    header_tips = (
        "Current estimated clearance at this frame.",
        "Lowest estimated swing clearance in the session.",
        "Highest estimated swing clearance in the session.",
        "Current stance or swing state from pose-derived contact.",
    )
    gui._foot_clearance_header_labels = []

    tk.Label(
        table,
        text="",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=(0, PAD_SM), pady=(0, 1))
    for col, heading in enumerate(titles, start=1):
        hdr = tk.Label(
            table,
            text=heading,
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        hdr.grid(row=0, column=col, sticky="ew", padx=(PAD_XS, 0), pady=(0, 1))
        gui._foot_clearance_header_labels.append(hdr)

    def _value_label(
        *,
        row: int,
        col: int,
        prefix: str,
        suffix: str,
        heading: str,
        tip: str,
        value_font: bool,
    ) -> tk.Label:
        value_lbl = tk.Label(
            table,
            text="Unavailable",
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI_SM if value_font else FONT_UI_XS,
            anchor="w",
            justify=tk.LEFT,
            wraplength=0,
        )
        grid_info = {
            "row": row,
            "column": col,
            "sticky": "nsew",
            "padx": (PAD_XS, 0),
            "pady": (0, 1),
        }
        value_lbl.grid(**grid_info)
        setattr(value_lbl, "_fc_grid", grid_info)
        # Column headers already name the fields — keep values to one line.
        setattr(value_lbl, "_fc_heading", "")
        setattr(value_lbl, "_fc_field", heading)
        setattr(gui, f"lbl_{prefix}_{suffix}", value_lbl)
        if tip:
            create_tooltip(value_lbl, tip, wraplength=280)
        return value_lbl

    for row, prefix, side_title in (
        (1, "foot_left", "Left foot"),
        (2, "foot_right", "Right foot"),
    ):
        side_lbl = tk.Label(
            table,
            text=side_title,
            bg=PANEL,
            fg=TEXT,
            font=FONT_UI_XS,
            anchor="w",
        )
        side_lbl.grid(row=row, column=0, sticky="w", padx=(0, PAD_SM), pady=(0, 1))
        setattr(gui, f"lbl_{prefix}_name", side_lbl)
        _value_label(
            row=row,
            col=1,
            prefix=prefix,
            suffix="current",
            heading=titles[0],
            tip=header_tips[0],
            value_font=True,
        )
        _value_label(
            row=row,
            col=2,
            prefix=prefix,
            suffix="min_swing",
            heading=titles[1],
            tip=header_tips[1],
            value_font=True,
        )
        _value_label(
            row=row,
            col=3,
            prefix=prefix,
            suffix="max_swing",
            heading=titles[2],
            tip=header_tips[2],
            value_font=True,
        )
        _value_label(
            row=row,
            col=4,
            prefix=prefix,
            suffix="state",
            heading=titles[3],
            tip=header_tips[3],
            value_font=False,
        )
        reason = tk.Label(
            table,
            text="",
            bg=PANEL,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        reason.grid(row=row, column=1, columnspan=4, sticky="ew")
        reason.grid_remove()
        setattr(gui, f"lbl_{prefix}_unavailable", reason)
        table.rowconfigure(row, minsize=22, weight=0)

    table.rowconfigure(0, minsize=14, weight=0)

    # Compact Overview strip: current L/R stay visible; the table lives in Details.
    table.grid_remove()
    host.rowconfigure(1, weight=0, minsize=0)
    header.columnconfigure(4, weight=0)
    header.columnconfigure(5, weight=0)
    header.columnconfigure(6, weight=0)
    header.columnconfigure(7, weight=0)
    for widget, col in (
        (getattr(gui, "lbl_foot_left_name", None), 4),
        (getattr(gui, "lbl_foot_left_current", None), 5),
        (getattr(gui, "lbl_foot_right_name", None), 6),
        (getattr(gui, "lbl_foot_right_current", None), 7),
    ):
        if widget is None:
            continue
        try:
            widget.grid(in_=header, row=0, column=col, sticky="w", padx=(PAD_SM, 0))
        except tk.TclError:
            pass

    gui._foot_clearance_detail_host = host
    gui._foot_clearance_table = table
    return host


def build_data_export_section(gui, parent: tk.Misc):
    """Compatibility wrapper — Advanced now uses categorized expandable cards."""
    from stablewalk.ui.tk.dashboard_advanced_workspace import build_advanced_workspace

    return build_advanced_workspace(gui, parent)


def bind_panel_word_wrap(gui, widget: tk.Misc, attr_names: tuple[str, ...]) -> None:
    """Keep summary labels wrapped to the panel width."""
    from stablewalk.ui.theme import set_wraplength_safe

    def _sync(_event: object | None = None) -> None:
        if getattr(widget, "_wrap_syncing", False):
            return
        width = widget.winfo_width()
        if width < 40:
            return
        wrap = responsive_wraplength(width, margin=16)
        setattr(widget, "_wrap_syncing", True)
        try:
            for attr in attr_names:
                lbl = getattr(gui, attr, None)
                if lbl is not None and hasattr(lbl, "configure"):
                    set_wraplength_safe(lbl, wrap)
        finally:
            setattr(widget, "_wrap_syncing", False)

    widget.bind("<Configure>", _sync, add="+")
    gui._summary_wrap_targets = getattr(gui, "_summary_wrap_targets", ())
    gui._summary_wrap_targets = (*gui._summary_wrap_targets, *attr_names)
