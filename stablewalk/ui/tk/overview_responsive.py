"""Overview-tab responsive reflow — breakpoints, stacking, layout diagnostics.

Does not recreate Matplotlib figures or analysis widgets. Transport / status
stay on the root outside the Overview tab.
"""

from __future__ import annotations

import logging
import os
import tkinter as tk
from typing import Any

from stablewalk.ui.theme import (
    FONT_UI_XS,
    GLYPH_COLLAPSE,
    GLYPH_EXPAND,
    MUTED,
    OVERVIEW_ROW_GAP,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
)
from stablewalk.ui.tk.analysis_split import MIN_H_VIDEO_ROW, MIN_W_VIDEO
from stablewalk.ui.tk.dashboard_sections import (
    OVERVIEW_HOST_VIZ_ROW,
    OVERVIEW_PATH_MIN_W,
    overview_kpi_row_min_h,
    overview_metric_padx,
    overview_video_row_min_h,
)

# Usable visualization floors — do not shrink these to squeeze chrome on-screen.
VIZ_ROW_MINSIZE_WIDE = MIN_H_VIDEO_ROW
VIZ_ROW_MINSIZE_STACKED = MIN_H_VIDEO_ROW
PATH_CANVAS_MINSIZE_WIDE = 200
PATH_CANVAS_MINSIZE_STACKED = 200
PATH_TOOLBAR_RESERVE_H = 36
PANEL_MINWIDTH = MIN_W_VIDEO
PATH_PANEL_MINWIDTH = OVERVIEW_PATH_MIN_W

logger = logging.getLogger(__name__)

# Width breakpoints (logical pixels after DPI-aware process).
# Four KPI cards stay on one row until each would drop below a readable width.
KPI_CARD_MINWIDTH = 160
WIDTH_KPI_FOUR = 720  # ≥ → 4 primary KPI columns
WIDTH_KPI_TWO = 400  # ≥ → 2×2 wrap; below → single column
WIDTH_KPI_SIX = WIDTH_KPI_FOUR  # back-compat alias (Overview now uses 4 primary)
WIDTH_SUMMARY_SIDE_BY_SIDE = 1280  # ≥ → Findings | Limitations; below → stack
# Three equal panels need ~320+220+240 plus gutters. Below this, stop compressing.
WIDTH_VIZ_THREE_COL = 1100
# Video | Skeleton on top, 3D Path full-width below.
WIDTH_VIZ_TWO_PLUS_ONE = 900
# 768p and 720p must compact chrome so metric cards stay fully visible.
HEIGHT_SHORT_WINDOW = 800
# Coordinate Inspector: side-by-side only when the 3D plot keeps a usable width.
PATH_PLOT_SIDE_MIN_W = 240
PATH_INSPECTOR_SIDE_MIN_W = 292
PATH_INSPECTOR_GUTTER_W = 8
PATH_INSPECTOR_SIDE_BY_SIDE_MIN = (
    PATH_PLOT_SIDE_MIN_W + PATH_INSPECTOR_SIDE_MIN_W + PATH_INSPECTOR_GUTTER_W
)
PATH_INSPECTOR_STACK_H = 168
PATH_INSPECTOR_TOGGLE_H = 28
PATH_VIEW_WEIGHT = 80
PATH_INSPECTOR_WEIGHT = 20


def path_inspector_side_minsize(panel_width: int) -> int:
    """Readable inspector rail; extra panel width is given to the 3D plot."""
    width = max(0, int(panel_width or 0))
    if width >= 720:
        return 320
    if width >= 560:
        return 304
    return PATH_INSPECTOR_SIDE_MIN_W

def layout_diag_enabled(gui: Any | None = None) -> bool:
    if gui is not None and getattr(gui, "_layout_diag", False):
        return True
    return os.environ.get("STABLEWALK_LAYOUT_DIAG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def dpi_scale_factor(root: tk.Misc) -> float:
    """Approximate Windows display scale (1.0 = 100%, 1.25 = 125%, …)."""
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor as _dpi

    return _dpi(root)


def overview_breakpoints(width: int, *, scale: float = 1.0) -> dict[str, bool]:
    """Derived layout flags for the current Overview width.

    ``scale`` is Windows display scaling (1.0 = 100%). At 150%+ the three-column
    visualization needs more pixels so labels stay readable.
    """
    w = int(width)
    if abs(float(scale) - 1.0) >= 0.08:
        from stablewalk.ui.tk.layout_scale import scaled_breakpoints

        bp = scaled_breakpoints(scale)
        three = int(bp["viz_three_col"])
        two = int(bp["viz_two_plus_one"])
        kpi4 = int(bp["kpi_four"])
        kpi2 = int(bp["kpi_two"])
        summary = int(bp["summary_side"])
    else:
        three = WIDTH_VIZ_THREE_COL
        two = WIDTH_VIZ_TWO_PLUS_ONE
        kpi4 = WIDTH_KPI_FOUR
        kpi2 = WIDTH_KPI_TWO
        summary = WIDTH_SUMMARY_SIDE_BY_SIDE
    return {
        "kpi_four": w >= kpi4,
        "kpi_two": w >= kpi2,
        "kpi_six": w >= kpi4,  # alias for older callers/tests
        "summary_side_by_side": w >= summary,
        "viz_three_col": w >= three,
        "viz_two_plus_one": two <= w < three,
        "viz_stacked": w < two,
    }


def overview_viz_layout_mode(width: int, *, scale: float = 1.0) -> str:
    """Return three_col, two_plus_one, or stacked for the available width."""
    w = int(width)
    # Unmeasured / first-map widths must stay three-column so construction
    # does not restack Video | Skeleton | Path before the skeleton exists.
    flags = overview_breakpoints(w, scale=scale)
    if w < 80 or flags["viz_three_col"]:
        return "three_col"
    if flags["viz_two_plus_one"]:
        return "two_plus_one"
    return "stacked"


def install_overview_scroll_host(gui: Any, tab_overview: tk.Misc) -> tk.Misc:
    """
    Overview layout root as a direct grid child of the notebook tab.

    Do **not** embed Video / Matplotlib panels in ``Canvas.create_window``.
    That pattern left widgets mapped and sized but unpainted (large black void
    under Summary; scrolling never revealed the panels).

    Layout ownership (main Clinical Gait Overview dashboard):
      tab_overview
        └── _overview_content  (returned; one content widget, grid layout)
              row 0    slim title                                      weight 0
              row 1    vertical splitter                               weight 1
                       ├── primary: Video | Skeleton | 3D Path
                       └── secondary: phase, KPI cards, foot clearance
                           (drag the sash to reveal more; pane clips overflow)

    Transport stays outside this host. Viewers are never Canvas windows.
    """
    from stablewalk.ui.theme import PANEL

    tab_overview.columnconfigure(0, weight=1)
    tab_overview.rowconfigure(0, weight=1)

    content = tk.Frame(tab_overview, bg=PANEL, highlightthickness=0)
    content.grid(row=0, column=0, sticky="nsew")
    content.columnconfigure(0, weight=1)

    # Compat attrs — outer canvas scroll intentionally removed.
    gui._overview_scroll_outer = content
    gui._overview_scroll_canvas = None
    gui._overview_scroll_vsb = None
    gui._overview_scroll_window_id = None
    gui._overview_content = content
    gui._overview_scroll_needed = False
    gui._overview_clinical_cap_h = None
    gui._overview_viz_room_guard = False

    def _sync(_event: object | None = None) -> None:
        if getattr(gui, "_overview_viz_room_guard", False):
            return
        ensure_overview_viz_room(gui)

    gui._sync_overview_scroll = _sync
    # Debounce: only sync from the content host to avoid Configure storms
    # during construction (tab+content both firing caused a hang).
    # after_idle coalesces bursty Configure events from height-cap regrids.
    gui._overview_viz_room_after_id = None

    def _sync_debounced(_event: object | None = None) -> None:
        if getattr(gui, "_overview_viz_room_guard", False):
            return
        aid = getattr(gui, "_overview_viz_room_after_id", None)
        root = getattr(gui, "root", None) or content
        if aid is not None:
            try:
                root.after_cancel(aid)
            except Exception:
                pass

        def _run() -> None:
            gui._overview_viz_room_after_id = None
            ensure_overview_viz_room(gui)

        try:
            gui._overview_viz_room_after_id = root.after_idle(_run)
        except Exception:
            ensure_overview_viz_room(gui)

    gui._sync_overview_scroll = _sync_debounced
    content.bind("<Configure>", _sync_debounced, add="+")
    return content


def ensure_overview_viz_room(gui: Any) -> None:
    """Guarantee the visualization row keeps a real minimum height."""
    if getattr(gui, "_overview_viz_room_guard", False):
        return
    gui._overview_viz_room_guard = True
    try:
        _ensure_overview_viz_room_impl(gui)
    finally:
        gui._overview_viz_room_guard = False


def _invalidate_clinical_height_cache(gui: Any) -> None:
    """Drop cached clinical height so the next room-sync remeasures truthfully."""
    gui._overview_clinical_natural_h = 0
    clinical = getattr(gui, "_overview_clinical_summary", None)
    content = getattr(gui, "_overview_content", None)
    if clinical is not None and content is not None and getattr(
        gui, "_overview_clinical_cap_h", None
    ) is not None:
        _clear_clinical_height_cap(clinical, content)
    gui._overview_clinical_cap_h = None


def _row_req_height(content: tk.Misc, row: int, *, skip: Any | None = None) -> int:
    """Natural height of mapped grid slaves in ``row`` (ignores ``skip``)."""
    total = 0
    try:
        for child in content.grid_slaves(row=row):
            if skip is not None and child is skip:
                continue
            try:
                if not child.winfo_ismapped() and str(child.winfo_manager() or "") == "":
                    continue
                total = max(total, int(child.winfo_reqheight()))
            except tk.TclError:
                continue
    except tk.TclError:
        pass
    return total


def _apply_clinical_height_cap(clinical: tk.Misc, content: tk.Misc, height: int) -> None:
    """
    Force the clinical chrome row to ``height``.

    On Windows Tk, ``grid_propagate(False)`` + ``configure(height=…)`` updates
    ``winfo_reqheight`` but often leaves the prior allocated ``winfo_height``.
    Re-asserting row 2 after the req change makes the grid honor the cap.
    """
    h = max(48, int(height))
    try:
        if int(clinical.winfo_height() or 0) == h and not clinical.grid_propagate():
            return
    except tk.TclError:
        pass
    try:
        clinical.grid_propagate(False)
    except tk.TclError:
        pass
    try:
        clinical.pack_propagate(False)
    except tk.TclError:
        pass
    try:
        clinical.configure(height=h)
    except tk.TclError:
        pass
    try:
        content.rowconfigure(3, weight=0, minsize=0)
    except tk.TclError:
        pass
    try:
        info = dict(clinical.grid_info() or {})
        if info:
            clinical.grid(
                row=int(info.get("row", 3)),
                column=int(info.get("column", 0)),
                sticky=str(info.get("sticky") or "ew"),
                padx=info.get("padx", 0),
                pady=info.get("pady", 0),
            )
    except tk.TclError:
        pass


def _clear_clinical_height_cap(clinical: tk.Misc, content: tk.Misc) -> None:
    try:
        clinical.grid_propagate(True)
    except tk.TclError:
        pass
    try:
        clinical.pack_propagate(True)
    except tk.TclError:
        pass
    try:
        content.rowconfigure(3, weight=0, minsize=0)
    except tk.TclError:
        pass


def _ensure_overview_viz_room_impl(gui: Any) -> None:
    content = getattr(gui, "_overview_content", None)
    tab = getattr(gui, "_tab_overview", None)
    clinical = getattr(gui, "_overview_clinical_summary", None)
    viz = getattr(gui, "_section_visual", None) or getattr(gui, "_primary_viz_host", None)
    if content is None or tab is None:
        return
    try:
        from stablewalk.ui.tk.dashboard_notebook import sync_notebook_tab_fill

        sync_notebook_tab_fill(gui)
    except Exception:
        pass
    try:
        avail = max(int(tab.winfo_height()), 1)
    except tk.TclError:
        return
    if avail < 80:
        return

    viz_row = OVERVIEW_HOST_VIZ_ROW
    if viz is not None:
        try:
            info = viz.grid_info() or {}
            if info:
                viz_row = int(info.get("row", viz_row))
            elif str(viz.winfo_manager() or "") == "":
                viz.grid(row=viz_row, column=0, sticky="nsew")
            elif str(viz.winfo_manager()) == "grid" and not viz.winfo_ismapped():
                viz.grid()
        except tk.TclError:
            pass

    from stablewalk.ui.tk.dashboard_sections import (
        apply_overview_host_row_weights,
        apply_overview_viz_grid_weights,
        overview_available_viz_h,
    )

    short = bool(getattr(gui, "_overview_short_window", False))
    host_h = 0
    try:
        host_h = int(content.winfo_height() or 0)
    except tk.TclError:
        host_h = 0
    wrap = str(getattr(gui, "_overview_kpi_layout", "") or "") == "2x2"
    apply_overview_host_row_weights(
        content,
        viz_row=viz_row,
        short=bool(getattr(gui, "_overview_video_short", False)),
        kpi_min=overview_kpi_row_min_h(wrap=wrap),
        host_h=host_h if host_h > 80 else None,
    )
    if viz is not None:
        available_h = overview_available_viz_h(gui, content)
        apply_overview_viz_grid_weights(
            viz,
            content_row=0,
            layout=str(getattr(gui, "_overview_viz_layout", "three_col") or "three_col"),
            short=short,
            video_min_h=overview_video_row_min_h(
                short=bool(getattr(gui, "_overview_video_short", False))
            ),
            available_h=available_h if available_h > 80 else None,
        )
        if available_h > 120:
            # The vertical sash owns leftover height. Do not pin the primary
            # pane to a stale configure(height=…) — that froze ~316 px
            # viewers at 1920×1080 and left a large unused lower band.
            try:
                viz.grid_propagate(False)
            except tk.TclError:
                pass
            try:
                from stablewalk.ui.tk.dashboard_overview_panes import (
                    sync_overview_viewer_split_height,
                )

                sync_overview_viewer_split_height(gui)
            except Exception:
                pass
            try:
                from stablewalk.ui.tk.dashboard_overview_panes import (
                    _place_overview_vertical_sash,
                    sync_overview_info_scroll,
                )

                _place_overview_vertical_sash(gui)
                sync_overview_info_scroll(gui)
            except Exception:
                pass
    try:
        from stablewalk.ui.tk.dashboard_sections import apply_overview_panel_min_heights

        apply_overview_panel_min_heights(gui)
    except Exception:
        logger.debug("overview panel min-height apply failed", exc_info=True)
    try:
        from stablewalk.ui.tk.dashboard_overview_panes import (
            remount_overview_viz_if_unmapped,
        )

        remount_overview_viz_if_unmapped(gui)
    except Exception:
        logger.debug("overview viz remount during room-sync failed", exc_info=True)

    # Clean Overview: do not height-cap the KPI strip — keep four cards readable.
    gui._overview_footer_hidden_for_viz = False
    gui._overview_footer_hidden_widgets = []
    if clinical is None:
        gui._overview_scroll_needed = False
        return

    prev_cap = getattr(gui, "_overview_clinical_cap_h", None)
    if prev_cap is not None:
        _clear_clinical_height_cap(clinical, content)
        gui._overview_clinical_cap_h = None
    gui._overview_clinical_natural_h = 0
    gui._overview_scroll_needed = False


def reflow_overview_kpi_cards(gui: Any, *, width: int, scale: float = 1.0) -> str:
    """Lay out primary KPI cards as 4 columns or a 2×2 wrap. Returns mode label."""
    cards = getattr(gui, "_overview_clinical_cards", None) or {}
    order = getattr(gui, "_overview_kpi_order", None) or list(cards.keys())
    # Primary metrics only (confidence / cycles live in the secondary row).
    secondary = set(getattr(gui, "_overview_secondary_order", ()) or ())
    order = [k for k in order if k not in secondary]
    if not order:
        return "none"
    first = cards.get(order[0])
    if first is None:
        return "none"
    metrics = first["frame"].master
    # The preferred one-row presentation is only safe once four cards have
    # enough real width; below that breakpoint, 2×2 prevents text overlap.
    flags = overview_breakpoints(width, scale=scale)
    cols = 4 if flags["kpi_four"] else 2
    mode = "4col" if cols == 4 else "2x2"
    prev = getattr(gui, "_overview_kpi_layout", None)
    if prev == mode and not getattr(gui, "_force_overview_reflow", False):
        _sync_kpi_wraplengths(gui, metrics, cols)
        return mode

    for c in range(4):
        try:
            if c < cols:
                metrics.columnconfigure(c, weight=1, uniform="ov_clin_cards")
            else:
                metrics.columnconfigure(c, weight=0, minsize=0, uniform="")
        except tk.TclError:
            pass
    for r in range(2):
        try:
            metrics.rowconfigure(r, weight=0)
        except tk.TclError:
            pass

    for i, key in enumerate(order):
        card = cards.get(key)
        if card is None:
            continue
        r, c = divmod(i, cols)
        left, right = overview_metric_padx(c, cols)
        top = 0 if r == 0 else OVERVIEW_ROW_GAP
        try:
            card["frame"].grid(
                row=r,
                column=c,
                sticky="nsew",
                padx=(left, right),
                pady=(top, 0),
            )
        except tk.TclError:
            pass

    gui._overview_kpi_layout = mode
    # KPI wrap changes clinical natural height — drop any stale height cap.
    _invalidate_clinical_height_cache(gui)
    _sync_kpi_wraplengths(gui, metrics, cols)
    return mode


def _sync_kpi_wraplengths(gui: Any, metrics: tk.Misc, cols: int) -> None:
    try:
        metrics.update_idletasks()
        mw = max(int(metrics.winfo_width()), 1)
    except tk.TclError:
        return
    card_w = max(80, (mw // max(cols, 1)) - 16)
    wrap = max(72, min(220, card_w - 12))
    cards = getattr(gui, "_overview_clinical_cards", None) or {}
    for card in cards.values():
        for key in ("desc", "title", "status"):
            widget = card.get(key)
            if widget is None:
                continue
            try:
                widget.configure(wraplength=wrap)
            except tk.TclError:
                pass


def reflow_overview_summary_panels(gui: Any, *, width: int, scale: float = 1.0) -> str:
    """Side-by-side Summary | Limitations, or stacked on narrow widths."""
    report = getattr(gui, "_overview_clinical_report", None)
    summary = getattr(gui, "_overview_summary_card", None)
    limitations = getattr(gui, "_overview_limitations_card", None)
    if report is None or summary is None or limitations is None:
        return "none"

    side_by_side = overview_breakpoints(width, scale=scale)["summary_side_by_side"]
    mode = "side_by_side" if side_by_side else "stacked"
    prev = getattr(gui, "_overview_summary_layout", None)
    if prev == mode and not getattr(gui, "_force_overview_reflow", False):
        _sync_report_wraplengths(gui, side_by_side=side_by_side)
        return mode

    try:
        if side_by_side:
            report.columnconfigure(0, weight=3, uniform="ov_clin_report")
            report.columnconfigure(1, weight=2, uniform="ov_clin_report")
            summary.grid(
                row=0, column=0, sticky="nsew", padx=(0, PAD_XS), pady=0
            )
            limitations.grid(
                row=0, column=1, sticky="nsew", padx=(PAD_XS, 0), pady=0
            )
        else:
            report.columnconfigure(0, weight=1, uniform="")
            report.columnconfigure(1, weight=0, minsize=0, uniform="")
            summary.grid(
                row=0,
                column=0,
                columnspan=2,
                sticky="ew",
                padx=0,
                pady=(0, PAD_SM),
            )
            limitations.grid(
                row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=0
            )
    except tk.TclError:
        pass

    gui._overview_summary_layout = mode
    _invalidate_clinical_height_cache(gui)
    _sync_report_wraplengths(gui, side_by_side=side_by_side)
    return mode


def _sync_report_wraplengths(gui: Any, *, side_by_side: bool) -> None:
    report = getattr(gui, "_overview_clinical_report", None)
    if report is None:
        return
    try:
        report.update_idletasks()
        rw = max(int(report.winfo_width()), 1)
    except tk.TclError:
        return
    if side_by_side:
        wrap_s = max(160, min(420, int(rw * 0.55) - 40))
        wrap_l = max(140, min(360, int(rw * 0.40) - 40))
    else:
        wrap_s = wrap_l = max(180, min(560, rw - 48))
    for lbl in getattr(gui, "_overview_clinical_bullet_labels", None) or []:
        try:
            lbl.configure(wraplength=wrap_s)
        except tk.TclError:
            pass
    for lbl in getattr(gui, "_overview_system_note_labels", None) or []:
        try:
            lbl.configure(wraplength=wrap_l)
        except tk.TclError:
            pass
    info = getattr(gui, "_overview_info_inner", None) or getattr(
        gui, "_overview_clinical_summary", None
    )
    info_w = rw
    if info is not None:
        try:
            info_w = max(160, int(info.winfo_width() or rw) - 24)
        except tk.TclError:
            info_w = rw
    for attr in (
        "lbl_overview_state",
        "lbl_overview_primary_finding",
        "lbl_overview_visible_disclaimer",
        "lbl_overview_coord_caption",
    ):
        lbl = getattr(gui, attr, None)
        if lbl is None:
            continue
        try:
            lbl.configure(wraplength=max(160, info_w))
        except tk.TclError:
            pass
    cards = getattr(gui, "_overview_clinical_cards", None) or {}
    card_wrap = max(80, min(220, info_w // max(1, len(cards)) - 16))
    for card in cards.values():
        desc = card.get("desc") if isinstance(card, dict) else None
        if desc is None:
            continue
        try:
            desc.configure(wraplength=card_wrap)
        except tk.TclError:
            pass


def apply_overview_viz_arrangement(
    gui: Any, *, width: int | None = None, scale: float = 1.0
) -> str:
    """Grid Video / Skeleton / Path for the current width without rebuilding."""
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        VIEW_MODE_OVERLAY,
        VIEW_MODE_RECONSTRUCTION_FULL,
        VIEW_MODE_SKELETON_ONLY,
        VIEW_MODE_VIDEO_ONLY,
    )
    from stablewalk.ui.tk.dashboard_sections import (
        apply_overview_viz_grid_weights,
        overview_available_viz_h,
        overview_panel_padx,
    )

    focus = getattr(gui, "_overview_focus_mode", None)
    if focus in (
        VIEW_MODE_VIDEO_ONLY,
        VIEW_MODE_OVERLAY,
        VIEW_MODE_SKELETON_ONLY,
        VIEW_MODE_RECONSTRUCTION_FULL,
    ):
        return str(getattr(gui, "_overview_viz_layout", "focused") or "focused")

    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    traj = getattr(gui, "overview_traj_panel", None)
    if section is None or video is None or skel is None or traj is None:
        return str(getattr(gui, "_overview_viz_layout", "three_col") or "three_col")

    if width is None:
        try:
            width = int(section.winfo_width() or 0)
        except tk.TclError:
            width = 0
        # A withdrawn or unmapped window often reports 1–200px. Do not stack
        # from that fake width during construction.
        mapped = False
        try:
            mapped = bool(section.winfo_ismapped()) and int(width) >= 80
        except tk.TclError:
            mapped = False
        if not mapped:
            width = WIDTH_VIZ_THREE_COL

    mode = overview_viz_layout_mode(
        int(width or WIDTH_VIZ_THREE_COL), scale=scale
    )
    short = int(getattr(gui, "_overview_short_window", 0) or 0) == 1
    video_min_h = overview_video_row_min_h(
        short=bool(getattr(gui, "_overview_video_short", False))
    )
    row = int(getattr(gui, "_primary_viz_content_row", 0))
    available_h = overview_available_viz_h(gui)
    apply_overview_viz_grid_weights(
        section,
        content_row=row,
        layout=mode,
        short=short,
        video_min_h=video_min_h,
        available_h=available_h if available_h > 80 else None,
    )

    from stablewalk.ui.tk.analysis_split import split_clear
    from stablewalk.ui.tk.dashboard_overview_panes import (
        _is_tk_widget,
        mount_overview_viewer_split,
        place_overview_under_viewer_chrome,
    )

    holders_ready = all(
        _is_tk_widget(getattr(gui, attr, None))
        for attr in (
            "_overview_viz_split",
            "_overview_pane_video",
            "_overview_pane_skeleton",
            "_overview_pane_path",
        )
    )
    # Holders own the LabelFrames. Rearrange the splitter, never the canvases.
    if holders_ready and mount_overview_viewer_split(
        gui, section, video, skel, traj, row=row, layout=mode
    ):
        place_overview_under_viewer_chrome(gui, section, chrome_row=row + 1, span=3)
        gui._overview_viz_layout = mode
        gui._overview_viz_stack_requested = mode != "three_col"
        gui._overview_traj_dock_visible = True
        return mode

    paned = getattr(gui, "_overview_viz_split", None)
    if mode != "three_col" and _is_tk_widget(paned):
        try:
            split_clear(paned)
            paned.grid_remove()
        except tk.TclError:
            pass

    try:
        if mode == "stacked":
            video.grid(row=row, column=0, columnspan=1, sticky="nsew", padx=0, pady=0)
            skel.grid(
                row=row + 1,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=0,
                pady=(OVERVIEW_ROW_GAP, 0),
            )
            traj.grid(
                row=row + 2,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=0,
                pady=(OVERVIEW_ROW_GAP, 0),
            )
            chrome_row = row + 3
        elif mode == "two_plus_one":
            video.grid(
                row=row,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=overview_panel_padx(0, columns=2),
                pady=0,
            )
            skel.grid(
                row=row,
                column=1,
                columnspan=1,
                sticky="nsew",
                padx=overview_panel_padx(1, columns=2),
                pady=0,
            )
            traj.grid(
                row=row + 1,
                column=0,
                columnspan=2,
                sticky="nsew",
                padx=0,
                pady=(OVERVIEW_ROW_GAP, 0),
            )
            chrome_row = row + 2
        elif mount_overview_viewer_split(
            gui, section, video, skel, traj, row=row
        ):
            chrome_row = row + 1
        else:
            video.grid(
                row=row,
                column=0,
                columnspan=1,
                sticky="nsew",
                padx=overview_panel_padx(0),
                pady=0,
            )
            skel.grid(
                row=row,
                column=1,
                columnspan=1,
                sticky="nsew",
                padx=overview_panel_padx(1),
                pady=0,
            )
            traj.grid(
                row=row,
                column=2,
                columnspan=1,
                sticky="nsew",
                padx=overview_panel_padx(2),
                pady=0,
            )
            chrome_row = row + 1
        span = 1 if mode == "stacked" else (2 if mode == "two_plus_one" else 3)
        place_overview_under_viewer_chrome(
            gui, section, chrome_row=chrome_row, span=span
        )
    except tk.TclError:
        pass

    gui._overview_viz_layout = mode
    gui._overview_viz_stack_requested = mode != "three_col"
    gui._overview_traj_dock_visible = True
    return mode


def _overview_phase_host(gui: Any, section: tk.Misc) -> Any:
    """Find the gait-phase strip that sits under the visualization row."""
    stored = getattr(gui, "_overview_phase_strip", None)
    if stored is not None:
        return stored
    indicator = getattr(gui, "_gait_phase_indicator", None)
    if indicator is None:
        return None
    parent = getattr(indicator, "master", None)
    if parent is None:
        return None
    if parent is section:
        return indicator
    if getattr(parent, "master", None) is section:
        return parent
    return parent


def reflow_overview_viz_panels(gui: Any, *, width: int, scale: float = 1.0) -> str:
    """Keep Video | Skeleton | Path mapped at a usable size. Stack when too narrow."""
    if bool(getattr(gui, "playing", False)):
        return str(getattr(gui, "_overview_viz_layout", "three_col") or "three_col")

    traj = getattr(gui, "overview_traj_panel", None)
    if traj is not None:
        try:
            traj.rowconfigure(1, weight=1, minsize=PATH_CANVAS_MINSIZE_WIDE)
        except tk.TclError:
            pass

    mode = apply_overview_viz_arrangement(gui, width=width, scale=scale)
    return mode


def _ensure_overview_inspector_toggle(gui: Any, path_body: tk.Misc) -> Any:
    """Compact Coordinate Inspector header used when the rail is stacked."""
    toggle = getattr(gui, "_overview_inspector_toggle", None)
    if toggle is not None:
        return toggle
    toggle = tk.Frame(path_body, bg=PANEL, highlightthickness=0)
    toggle.columnconfigure(0, weight=0)
    toggle.columnconfigure(1, weight=1)
    glyph = tk.Label(
        toggle,
        text=GLYPH_EXPAND,
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        width=2,
        anchor="w",
    )
    glyph.grid(row=0, column=0, sticky="w")
    title = tk.Label(
        toggle,
        text="Coordinate Inspector",
        bg=PANEL,
        fg=TEXT,
        font=FONT_UI_XS,
        anchor="w",
    )
    title.grid(row=0, column=1, sticky="ew")

    def _toggle(_event: object | None = None) -> None:
        expanded = not bool(getattr(gui, "_overview_inspector_expanded", False))
        gui._overview_inspector_expanded = expanded
        panel = getattr(gui, "overview_traj_panel", None)
        panel_w = 0
        if panel is not None:
            try:
                panel_w = int(panel.winfo_width() or 0)
            except tk.TclError:
                panel_w = 0
        reflow_overview_path_inspector(
            gui,
            panel_width=panel_w,
            layout=str(getattr(gui, "_overview_viz_layout", "three_col") or "three_col"),
        )
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule) and not bool(getattr(gui, "playing", False)):
            try:
                schedule()
            except Exception:
                pass

    for widget in (toggle, glyph, title):
        widget.bind("<Button-1>", _toggle, add="+")
        try:
            widget.configure(cursor="hand2")
        except tk.TclError:
            pass
    gui._overview_inspector_toggle = toggle
    gui._overview_inspector_toggle_glyph = glyph
    return toggle


def reflow_overview_path_inspector(
    gui: Any, *, panel_width: int, layout: str | None = None
) -> str:
    """Keep the Coordinate Inspector readable: side-by-side, stacked, or collapsed.

    Never overlays or shrinks the 3D path below ``PATH_PLOT_SIDE_MIN_W``.
    When the Path panel cannot fit plot + rail, the inspector stacks under
    the plot (collapsed by default in the shared three-column row).
    """
    path_body = getattr(gui, "overview_traj_path_body", None)
    canvas_host = getattr(gui, "overview_traj_canvas_host", None)
    readout = getattr(gui, "overview_traj_readout", None)
    if path_body is None or canvas_host is None or readout is None:
        return "none"

    mode = str(layout or getattr(gui, "_overview_viz_layout", "three_col") or "three_col")
    width = max(0, int(panel_width or 0))
    # Three-column Overview keeps the inspector under the plot (collapsed)
    # so Video / Skeleton / Path stay balanced and the cube is not covered.
    needs_stack = bool(0 < width < PATH_INSPECTOR_SIDE_BY_SIDE_MIN) or mode == "three_col"
    toggle = None
    if needs_stack:
        try:
            toggle = _ensure_overview_inspector_toggle(gui, path_body)
        except tk.TclError:
            toggle = getattr(gui, "_overview_inspector_toggle", None)
    else:
        toggle = getattr(gui, "_overview_inspector_toggle", None)

    last_mode = getattr(gui, "_overview_inspector_layout_mode", None)
    if needs_stack:
        if last_mode != mode:
            # Advanced inspector starts collapsed so the cube keeps the panel height.
            gui._overview_inspector_expanded = False
        expanded = bool(gui._overview_inspector_expanded)
        gui._overview_inspector_layout_mode = mode
    else:
        expanded = True
        gui._overview_inspector_layout_mode = None

    try:
        if needs_stack:
            path_body.columnconfigure(0, weight=1, minsize=PATH_PLOT_SIDE_MIN_W)
            path_body.columnconfigure(1, weight=0, minsize=0)
            # 3D plot keeps a usable floor. Inspector scrolls in its own row
            # instead of covering or crushing the cube.
            path_body.rowconfigure(0, weight=1, minsize=PATH_CANVAS_MINSIZE_WIDE)
            canvas_host.grid(
                row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=0
            )
            glyph = getattr(gui, "_overview_inspector_toggle_glyph", None)
            if toggle is not None:
                path_body.rowconfigure(1, weight=0, minsize=PATH_INSPECTOR_TOGGLE_H)
                toggle.grid(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                    pady=(PAD_XS, 0),
                )
                if glyph is not None:
                    try:
                        glyph.configure(
                            text=GLYPH_COLLAPSE if expanded else GLYPH_EXPAND
                        )
                    except tk.TclError:
                        pass
            if expanded:
                path_body.rowconfigure(2, weight=0, minsize=0)
                try:
                    readout.configure(height=1)
                    readout.grid_propagate(True)
                except tk.TclError:
                    pass
                readout.grid(
                    row=2,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                    padx=0,
                    pady=(PAD_XS, 0),
                )
                gui._overview_path_inspector_layout = "stacked"
                _wrap_overview_path_status(gui, width)
                return "stacked"
            try:
                readout.grid_remove()
            except tk.TclError:
                pass
            path_body.rowconfigure(2, weight=0, minsize=0)
            gui._overview_path_inspector_layout = "collapsed"
            _wrap_overview_path_status(gui, width)
            return "collapsed"

        if toggle is not None:
            try:
                toggle.grid_remove()
            except tk.TclError:
                pass
        try:
            readout.grid_propagate(True)
        except tk.TclError:
            pass
        plot_min = PATH_PLOT_SIDE_MIN_W
        inspector_min = path_inspector_side_minsize(width)
        path_body.columnconfigure(0, weight=PATH_VIEW_WEIGHT, minsize=plot_min)
        path_body.columnconfigure(1, weight=PATH_INSPECTOR_WEIGHT, minsize=inspector_min)
        path_body.rowconfigure(0, weight=1, minsize=PATH_CANVAS_MINSIZE_WIDE)
        path_body.rowconfigure(1, weight=0, minsize=0)
        path_body.rowconfigure(2, weight=0, minsize=0)
        canvas_host.grid(
            row=0, column=0, columnspan=1, sticky="nsew", padx=(0, PAD_XS)
        )
        readout.grid(row=0, column=1, columnspan=1, sticky="nsew")
        gui._overview_path_inspector_layout = "side"
        _wrap_overview_path_status(gui, width)
        return "side"
    except tk.TclError:
        return str(getattr(gui, "_overview_path_inspector_layout", None) or "none")


def _wrap_overview_path_status(gui: Any, panel_width: int) -> None:
    """Keep Path status / metric text inside the 3D pane instead of clipping."""
    wrap = max(120, min(420, int(panel_width or 0) - 36))
    for attr in (
        "lbl_overview_path_xyz",
        "lbl_overview_path_length",
        "lbl_overview_path_displacement",
        "lbl_overview_coord_caption",
        "lbl_overview_traj_legend",
        "lbl_overview_traj_metrics",
        "lbl_overview_traj_detail",
        "lbl_overview_traj_video",
    ):
        lbl = getattr(gui, attr, None)
        if lbl is None:
            continue
        try:
            lbl.configure(wraplength=wrap, justify="left")
        except tk.TclError:
            pass


def _apply_overview_soft_panel_floors(
    gui: Any, *, width: int, height: int
) -> None:
    """Keep Video / Skeleton / Path readable without covering playback."""
    from stablewalk.ui.tk.dashboard_sections import (
        apply_overview_host_row_weights,
        overview_available_viz_h,
        overview_fit_viz_row_floors,
    )

    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    overview_host = getattr(gui, "_overview_layout_host", None) or getattr(
        gui, "_overview_content", None
    )
    if section is None:
        return
    short = int(height or 0) > 0 and int(height) <= HEIGHT_SHORT_WINDOW
    video_short = int(height or 0) > 0 and int(height) <= 720
    gui._overview_short_window = 1 if short else 0
    gui._overview_video_short = 1 if video_short else 0
    mode = str(getattr(gui, "_overview_viz_layout", None) or overview_viz_layout_mode(width))
    video_row_floor = overview_video_row_min_h(short=video_short)
    available_h = 0
    if overview_host is not None:
        available_h = overview_available_viz_h(gui, overview_host)
    video_row_floor, _phase_fit, _clearance_fit = overview_fit_viz_row_floors(
        short=video_short,
        layout=mode,
        available_h=available_h if available_h > 80 else None,
        panel_floor=video_row_floor,
    )
    path_content_floor = min(PATH_CANVAS_MINSIZE_WIDE, max(80, video_row_floor - PATH_TOOLBAR_RESERVE_H))
    video_content_floor = max(120, video_row_floor - 24)
    skeleton_content_floor = max(120, video_row_floor - PATH_TOOLBAR_RESERVE_H)

    apply_overview_viz_arrangement(gui, width=width)
    if overview_host is not None:
        host_h = 0
        try:
            host_h = int(overview_host.winfo_height() or 0)
        except tk.TclError:
            host_h = 0
        wrap = str(getattr(gui, "_overview_kpi_layout", "") or "") == "2x2"
        apply_overview_host_row_weights(
            overview_host,
            viz_row=OVERVIEW_HOST_VIZ_ROW,
            short=video_short,
            kpi_min=overview_kpi_row_min_h(wrap=wrap),
            host_h=host_h if host_h > 80 else None,
        )

    try:
        from stablewalk.ui.tk.dashboard_overview_panes import (
            sync_overview_viewer_split_height,
        )

        sync_overview_viewer_split_height(gui)
    except Exception:
        pass

    for widget, content_row, content_floor in (
        (getattr(gui, "video_frame", None), 3, video_content_floor),
        (getattr(gui, "skel_frame", None), 1, skeleton_content_floor),
        (getattr(gui, "overview_traj_panel", None), 1, path_content_floor),
    ):
        if widget is not None:
            try:
                widget.rowconfigure(content_row, weight=1, minsize=content_floor)
            except tk.TclError:
                pass

    path_panel = getattr(gui, "overview_traj_panel", None)
    panel_w = 0
    if path_panel is not None:
        try:
            panel_w = int(path_panel.winfo_width() or 0)
        except tk.TclError:
            panel_w = 0
    if panel_w < 80:
        from stablewalk.ui.tk.analysis_split import overview_pane_minsizes
        from stablewalk.ui.tk.layout_scale import layout_scale_from_gui

        path_min = overview_pane_minsizes(layout_scale_from_gui(gui))[2]
        path_min = max(int(OVERVIEW_PATH_MIN_W), path_min)
        if mode == "three_col":
            panel_w = max(path_min, int(width) // 3)
        else:
            panel_w = max(path_min, int(width) - 24)
    reflow_overview_path_inspector(gui, panel_width=panel_w, layout=mode)
    caption = getattr(gui, "lbl_overview_coord_caption", None)
    if caption is not None and panel_w >= 80:
        try:
            caption.configure(wraplength=max(160, int(panel_w) - 16))
        except tk.TclError:
            pass
    if not bool(getattr(gui, "playing", False)):
        try:
            from stablewalk.ui.tk.dashboard_overview_panes import (
                _fill_overview_plot_widgets,
            )

            _fill_overview_plot_widgets(gui)
        except Exception:
            pass
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
                pass


def reflow_overview_layout(
    gui: Any,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, str]:
    """Apply all Overview responsive rules for the current window size."""
    root = getattr(gui, "root", None)
    if root is None:
        return {}
    if width is None:
        try:
            width = int(root.winfo_width())
        except tk.TclError:
            width = 0
    if height is None:
        try:
            height = int(root.winfo_height())
        except tk.TclError:
            height = 0
    if width < 2:
        return {}

    content = getattr(gui, "_overview_content", None) or getattr(
        gui, "_tab_overview", None
    )
    if content is not None:
        try:
            content.update_idletasks()
            cw = int(content.winfo_width())
            if cw > 40:
                width = cw
        except tk.TclError:
            pass

    from stablewalk.ui.tk.layout_scale import layout_scale_from_gui

    scale = layout_scale_from_gui(gui)
    result = {
        "kpi": reflow_overview_kpi_cards(gui, width=width, scale=scale),
        "summary": reflow_overview_summary_panels(gui, width=width, scale=scale),
        "viz": reflow_overview_viz_panels(gui, width=width, scale=scale),
    }
    _apply_overview_soft_panel_floors(gui, width=width, height=int(height or 0))
    gui._force_overview_reflow = False

    sync = getattr(gui, "_sync_overview_scroll", None)
    if callable(sync):
        try:
            sync()
        except Exception:
            pass

    if layout_diag_enabled(gui):
        log_overview_layout_diagnostic(gui, width=width, height=height, result=result)
    return result


def _widget_size(widget: Any) -> tuple[int, int] | None:
    if widget is None:
        return None
    try:
        widget.update_idletasks()
        w, h = int(widget.winfo_width()), int(widget.winfo_height())
        if w < 1 and h < 1:
            return None
        return (w, h)
    except tk.TclError:
        return None


def _grid_minsize(
    widget: Any, *, row: int | None = None, column: int | None = None
) -> int:
    try:
        if row is not None:
            return int(widget.grid_rowconfigure(row).get("minsize", 0) or 0)
        if column is not None:
            return int(widget.grid_columnconfigure(column).get("minsize", 0) or 0)
    except Exception:
        pass
    return 0


def collect_overview_layout_snapshot(
    gui: Any,
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    """Structured Overview geometry for diagnostics / tests."""
    root = getattr(gui, "root", None)
    if width is None and root is not None:
        try:
            width = int(root.winfo_width())
        except tk.TclError:
            width = None
    if height is None and root is not None:
        try:
            height = int(root.winfo_height())
        except tk.TclError:
            height = None

    dpi = dpi_scale_factor(root) if root is not None else 1.0
    panels = {
        "video": _widget_size(getattr(gui, "video_frame", None)),
        "skeleton": _widget_size(getattr(gui, "skel_frame", None)),
        "path": _widget_size(getattr(gui, "overview_traj_panel", None)),
        "clinical": _widget_size(getattr(gui, "_overview_clinical_summary", None)),
        "section1": _widget_size(
            getattr(gui, "_section_visual", None)
            or getattr(gui, "_primary_viz_host", None)
        ),
        "transport": _widget_size(getattr(gui, "_transport_row", None)),
        "content": _widget_size(getattr(gui, "_overview_content", None)),
    }

    below_min: list[str] = []
    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    if section is not None:
        for col, name in ((0, "video_col"), (1, "skel_col"), (2, "path_col")):
            min_w = _grid_minsize(section, column=col)
            size = panels.get(
                "video" if col == 0 else "skeleton" if col == 1 else "path"
            )
            if min_w > 0 and size is not None and size[0] > 1 and size[0] < min_w:
                below_min.append(f"{name}: width {size[0]} < minsize {min_w}")

    transport = getattr(gui, "_transport_row", None)
    transport_present = transport is not None
    transport_mapped = False
    if transport is not None:
        try:
            transport_mapped = bool(transport.winfo_ismapped())
        except tk.TclError:
            transport_mapped = False

    return {
        "window": (width, height),
        "dpi_scale": round(dpi, 3),
        "breakpoints": overview_breakpoints(int(width or 0)),
        "kpi_layout": getattr(gui, "_overview_kpi_layout", None),
        "summary_layout": getattr(gui, "_overview_summary_layout", None),
        "viz_layout": getattr(gui, "_overview_viz_layout", None),
        "scroll_needed": bool(getattr(gui, "_overview_scroll_needed", False)),
        "panels": panels,
        "below_minsize": below_min,
        "transport_present": transport_present,
        "transport_mapped": transport_mapped,
        # Back-compat alias used by earlier tests / logs.
        "transport_visible": transport_present,
    }


def log_overview_layout_diagnostic(
    gui: Any,
    *,
    width: int | None = None,
    height: int | None = None,
    result: dict[str, str] | None = None,
) -> None:
    """Print Overview layout diagnostics (dev-only)."""
    snap = collect_overview_layout_snapshot(gui, width=width, height=height)
    win = snap["window"]
    lines = [
        "[LayoutDiag] Overview",
        f"  window={win[0]}x{win[1]}  dpi_scale={snap['dpi_scale']:.2f}",
        f"  kpi={snap['kpi_layout']}  summary={snap['summary_layout']}  "
        f"viz={snap['viz_layout']}  scroll={snap['scroll_needed']}",
        f"  transport_present={snap['transport_present']} "
        f"mapped={snap['transport_mapped']}",
    ]
    if result:
        lines.append(f"  reflow={result}")
    for name, size in (snap["panels"] or {}).items():
        if size is None:
            lines.append(f"  {name}: —")
        else:
            lines.append(f"  {name}: {size[0]}×{size[1]}")
    if snap["below_minsize"]:
        lines.append("  BELOW MINSIZE:")
        for item in snap["below_minsize"]:
            lines.append(f"    • {item}")
    msg = "\n".join(lines)
    print(msg, flush=True)
    logger.debug(msg)


__all__ = [
    "WIDTH_KPI_FOUR",
    "WIDTH_KPI_SIX",
    "WIDTH_SUMMARY_SIDE_BY_SIDE",
    "WIDTH_VIZ_THREE_COL",
    "WIDTH_VIZ_TWO_PLUS_ONE",
    "apply_overview_viz_arrangement",
    "overview_viz_layout_mode",
    "VIZ_ROW_MINSIZE_STACKED",
    "VIZ_ROW_MINSIZE_WIDE",
    "PATH_CANVAS_MINSIZE_STACKED",
    "PATH_CANVAS_MINSIZE_WIDE",
    "PANEL_MINWIDTH",
    "PATH_PANEL_MINWIDTH",
    "collect_overview_layout_snapshot",
    "dpi_scale_factor",
    "ensure_overview_viz_room",
    "install_overview_scroll_host",
    "layout_diag_enabled",
    "log_overview_layout_diagnostic",
    "overview_breakpoints",
    "reflow_overview_kpi_cards",
    "reflow_overview_layout",
    "reflow_overview_path_inspector",
    "path_inspector_side_minsize",
    "PATH_INSPECTOR_SIDE_MIN_W",
    "PATH_INSPECTOR_SIDE_BY_SIDE_MIN",
    "PATH_VIEW_WEIGHT",
    "PATH_INSPECTOR_WEIGHT",
    "reflow_overview_summary_panels",
    "reflow_overview_viz_panels",
]
