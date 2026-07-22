"""Unified 2D chart interactions for the StableWalk dashboard.

Attaches grid-friendly navigation (zoom/pan), crosshair, hover tooltips,
playhead finalize, responsive view restore, and PNG/SVG export to every
time-series figure — without touching 3D trajectory canvases.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING, Any, Callable

from stablewalk.ui.colors import ACCENT_ALT, MUTED
from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_UI_XS,
    PAD_XS,
    create_tooltip,
)
from stablewalk.ui.viewers.chart_hover import (
    ChartHoverPoint,
    attach_chart_hover_tooltips,
    set_figure_hover_points,
)
from stablewalk.ui.viewers.chart_navigation import (
    attach_chart_navigation,
    finalize_chart_view,
    reset_chart_navigation,
    reset_chart_view,
)
from stablewalk.ui.viewers.chart_style import apply_chart_grid

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

FIG_CROSSHAIR_ATTR = "_chart_crosshair_artists"
FIG_CROSSHAIR_CID_ATTR = "_chart_crosshair_cid"
FIG_INTERACTIONS_ATTR = "_chart_interactions_attached"


def _clear_crosshair(fig: Figure) -> None:
    artists = getattr(fig, FIG_CROSSHAIR_ATTR, None) or []
    for artist in artists:
        try:
            artist.remove()
        except Exception:
            pass
    setattr(fig, FIG_CROSSHAIR_ATTR, [])


def attach_chart_crosshair(fig: Figure, canvas: FigureCanvasTkAgg) -> None:
    """Follow-cursor crosshair (x + y) on the active axes; shared across stacks."""
    cid = getattr(fig, FIG_CROSSHAIR_CID_ATTR, None)
    if cid is not None:
        try:
            canvas.mpl_disconnect(cid)
        except Exception:
            pass

    def _on_motion(event) -> None:
        from stablewalk.ui.viewers.chart_navigation import (
            FIG_NAV_BLOCK_HOVER_ATTR,
            chart_nav_is_active,
        )

        _clear_crosshair(fig)
        if chart_nav_is_active(fig) or getattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False):
            canvas.draw_idle()
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            canvas.draw_idle()
            return
        ax = event.inaxes
        vline = ax.axvline(
            float(event.xdata),
            color=ACCENT_ALT,
            linewidth=0.8,
            alpha=0.55,
            linestyle=":",
            zorder=18,
        )
        hline = ax.axhline(
            float(event.ydata),
            color=ACCENT_ALT,
            linewidth=0.8,
            alpha=0.45,
            linestyle=":",
            zorder=18,
        )
        setattr(fig, FIG_CROSSHAIR_ATTR, [vline, hline])
        canvas.draw_idle()

    def _on_leave(_event) -> None:
        _clear_crosshair(fig)
        canvas.draw_idle()

    cid_motion = canvas.mpl_connect("motion_notify_event", _on_motion)
    cid_leave = canvas.mpl_connect("figure_leave_event", _on_leave)
    setattr(fig, FIG_CROSSHAIR_CID_ATTR, (cid_motion, cid_leave))


def attach_chart_interactions(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
    *,
    on_hover_point: Callable[[ChartHoverPoint], None] | None = None,
) -> None:
    """One-shot attach of zoom/pan, hover tooltips, and crosshair."""
    if getattr(fig, FIG_INTERACTIONS_ATTR, False):
        # Re-bind hover callback / canvas ids if figure was reused.
        attach_chart_navigation(fig, canvas)
        attach_chart_hover_tooltips(fig, canvas, on_hover_point=on_hover_point)
        attach_chart_crosshair(fig, canvas)
        return

    attach_chart_navigation(fig, canvas)
    attach_chart_hover_tooltips(fig, canvas, on_hover_point=on_hover_point)
    attach_chart_crosshair(fig, canvas)
    setattr(fig, FIG_INTERACTIONS_ATTR, True)


def finalize_chart_interactions(
    fig: Figure,
    canvas: FigureCanvasTkAgg | None = None,
    *,
    hover_points: list[ChartHoverPoint] | None = None,
) -> None:
    """Call after every chart redraw: restore zoom, refresh hover targets, grid."""
    if hover_points is not None:
        set_figure_hover_points(fig, hover_points)
    for ax in fig.axes:
        try:
            apply_chart_grid(ax, y_minor=True)
        except Exception:
            pass
    finalize_chart_view(fig, canvas)


def export_chart_figure(
    fig: Figure,
    *,
    parent: tk.Misc | None = None,
    default_name: str = "stablewalk_chart",
) -> Path | None:
    """Prompt for a PNG/SVG/PDF path and save the live figure."""
    path = filedialog.asksaveasfilename(
        parent=parent,
        title="Export graph",
        defaultextension=".png",
        initialfile=f"{default_name}.png",
        filetypes=(
            ("PNG image", "*.png"),
            ("SVG vector", "*.svg"),
            ("PDF document", "*.pdf"),
            ("All files", "*.*"),
        ),
    )
    if not path:
        return None
    out = Path(path)
    fig.savefig(
        out,
        dpi=160,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        bbox_inches="tight",
    )
    return out


def build_chart_tools_bar(
    gui: Any,
    parent: tk.Misc,
    *,
    fig_attr: str,
    canvas_attr: str,
    export_name: str,
) -> tk.Frame:
    """Compact Reset view + Export controls for a 2D chart host."""
    bar = tk.Frame(parent, bg=ELEVATED, highlightthickness=1, highlightbackground=BORDER)

    tk.Label(bar, text="Graph", bg=ELEVATED, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(PAD_XS, 4), pady=2
    )

    def _fig():
        return getattr(gui, fig_attr, None)

    def _canvas():
        return getattr(gui, canvas_attr, None)

    def _reset() -> None:
        fig = _fig()
        canvas = _canvas()
        if fig is None:
            return
        reset_chart_view(fig, canvas)

    def _export() -> None:
        fig = _fig()
        if fig is None:
            return
        export_chart_figure(fig, parent=getattr(gui, "root", None), default_name=export_name)

    btn_reset = ttk.Button(bar, text="Reset", width=6, command=_reset, takefocus=False)
    btn_reset.pack(side=tk.LEFT, padx=1, pady=2)
    create_tooltip(btn_reset, "Reset zoom / pan (or double-click chart)")

    btn_export = ttk.Button(bar, text="Export", width=7, command=_export, takefocus=False)
    btn_export.pack(side=tk.LEFT, padx=1, pady=2)
    create_tooltip(btn_export, "Export graph as PNG / SVG / PDF")

    tip = tk.Label(
        bar,
        text="Wheel zoom · drag pan · Shift+drag zoom",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
    )
    tip.pack(side=tk.LEFT, padx=(6, PAD_XS))

    return bar


def reset_all_chart_navigation(gui: Any) -> None:
    """Clear saved zoom on every known 2D chart figure (new session)."""
    for attr in ("fig", "fig_contact_gait", "fig_biomech", "fig_joint_motion"):
        fig = getattr(gui, attr, None)
        if fig is not None:
            reset_chart_navigation(fig)


__all__ = [
    "attach_chart_crosshair",
    "attach_chart_interactions",
    "build_chart_tools_bar",
    "export_chart_figure",
    "finalize_chart_interactions",
    "reset_all_chart_navigation",
]
