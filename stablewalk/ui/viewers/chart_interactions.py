"""Unified 2D chart interactions for the StableWalk dashboard.

Attaches grid-friendly navigation (zoom/pan), crosshair, hover tooltips,
playhead finalize, responsive view restore, and PNG/SVG export to every
time-series figure — without touching 3D trajectory canvases.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from stablewalk.ui.colors import ACCENT_ALT, MUTED, PANEL
from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_UI_XS,
    PAD_XS,
    TEXT,
    create_tooltip,
)
from stablewalk.ui.viewers.chart_hover import (
    ChartHoverPoint,
    attach_chart_hover_tooltips,
    format_inspect_crosshair_chip,
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


def attach_chart_crosshair(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
    *,
    sync_all_axes: bool = True,
) -> None:
    """Follow-cursor crosshair; optional vertical sync across every subplot."""
    prev = getattr(fig, FIG_CROSSHAIR_CID_ATTR, None)
    if prev is not None:
        cids = prev if isinstance(prev, (tuple, list)) else (prev,)
        for cid in cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        setattr(fig, FIG_CROSSHAIR_CID_ATTR, None)

    def _on_motion(event) -> None:
        from stablewalk.ui.viewers.chart_navigation import (
            FIG_NAV_BLOCK_HOVER_ATTR,
            chart_nav_is_active,
        )

        _clear_crosshair(fig)
        if chart_nav_is_active(fig) or getattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False):
            canvas.draw_idle()
            return
        if event.inaxes is None or event.xdata is None:
            canvas.draw_idle()
            return
        x = float(event.xdata)
        artists: list[Any] = []
        targets = list(fig.axes) if sync_all_axes else [event.inaxes]
        for ax in targets:
            if ax is None:
                continue
            vline = ax.axvline(
                x,
                color=ACCENT_ALT,
                linewidth=1.05,
                alpha=0.72,
                linestyle=":",
                zorder=18,
            )
            artists.append(vline)
        # Always draw a horizontal guide on the hovered axes (lab readout).
        if event.ydata is not None and event.inaxes is not None:
            hline = event.inaxes.axhline(
                float(event.ydata),
                color=ACCENT_ALT,
                linewidth=0.9,
                alpha=0.50,
                linestyle=":",
                zorder=18,
            )
            artists.append(hline)
            # Compact inspect chip — dotted ACCENT_ALT, never the playback stamp.
            try:
                from stablewalk.ui.viewers.chart_playhead import (
                    edge_aware_callout_placement,
                )

                x_is_percent = bool(
                    getattr(event.inaxes, "_stablewalk_x_is_percent", False)
                )
                chip = format_inspect_crosshair_chip(
                    float(event.xdata), x_is_percent=x_is_percent
                )
                cx, cy, cha, cva = edge_aware_callout_placement(
                    event.inaxes,
                    float(event.xdata),
                    float(event.ydata),
                    text=chip,
                )
                tip = event.inaxes.text(
                    cx,
                    cy,
                    chip,
                    color=TEXT,
                    fontsize=9.0,
                    va=cva,
                    ha=cha,
                    zorder=19,
                    clip_on=False,
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": PANEL,
                        "edgecolor": ACCENT_ALT,
                        "alpha": 0.9,
                        "linewidth": 0.6,
                    },
                )
                artists.append(tip)
            except Exception:
                pass
        setattr(fig, FIG_CROSSHAIR_ATTR, artists)
        canvas.draw_idle()

    def _on_leave(_event) -> None:
        _clear_crosshair(fig)
        canvas.draw_idle()

    cid_motion = canvas.mpl_connect("motion_notify_event", _on_motion)
    cid_leave = canvas.mpl_connect("figure_leave_event", _on_leave)
    setattr(fig, FIG_CROSSHAIR_CID_ATTR, (cid_motion, cid_leave))


def attach_chart_time_seek(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
    on_seek_time: Callable[[float], None],
) -> None:
    """Click (without drag) seeks playback to the clicked time on shared-X charts."""
    attr = "_chart_time_seek_cid"
    prev = getattr(fig, attr, None)
    if prev is not None:
        try:
            canvas.mpl_disconnect(prev)
        except Exception:
            pass

    press: dict[str, float | None] = {"x": None, "y": None}

    def _on_press(event) -> None:
        if getattr(fig, FIG_MEASURE_MODE_ATTR, False):
            press["x"] = press["y"] = None
            return
        if event.inaxes is None or event.xdata is None or event.button != 1:
            press["x"] = press["y"] = None
            return
        # Frequency / lag axes opt out via ``_sw_time_seek = False``.
        if getattr(event.inaxes, "_sw_time_seek", True) is False:
            press["x"] = press["y"] = None
            return
        if getattr(event, "key", None) == "shift":
            press["x"] = press["y"] = None
            return
        press["x"] = float(event.x)
        press["y"] = float(event.y)

    def _on_release(event) -> None:
        from stablewalk.ui.viewers.chart_navigation import chart_nav_is_active

        if press["x"] is None or event.xdata is None or event.inaxes is None:
            press["x"] = press["y"] = None
            return
        if getattr(event.inaxes, "_sw_time_seek", True) is False:
            press["x"] = press["y"] = None
            return
        if chart_nav_is_active(fig) or getattr(fig, FIG_MEASURE_MODE_ATTR, False):
            press["x"] = press["y"] = None
            return
        dx = abs(float(event.x) - float(press["x"]))
        dy = abs(float(event.y) - float(press["y"] or event.y))
        press["x"] = press["y"] = None
        # Treat as a click (not a pan) when the pointer barely moved.
        if dx > 6 or dy > 6:
            return
        try:
            on_seek_time(float(event.xdata))
        except Exception:
            pass

    canvas.mpl_connect("button_press_event", _on_press)
    cid = canvas.mpl_connect("button_release_event", _on_release)
    setattr(fig, attr, cid)


FIG_HOVER_SEEK_CID_ATTR = "_chart_hover_seek_cid"
FIG_MEASURE_MODE_ATTR = "_chart_measure_mode"
FIG_MEASURE_CID_ATTR = "_chart_measure_cid"
FIG_MEASURE_ARTISTS_ATTR = "_chart_measure_artists"
FIG_MEASURE_POINTS_ATTR = "_chart_measure_points"


def attach_chart_hover_seek(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
    on_seek_time: Callable[[float], None],
    *,
    is_enabled: Callable[[], bool] | None = None,
    min_dt_s: float = 1.0 / 45.0,
) -> None:
    """Scrub playback to cursor X while hovering any shared-timeline subplot."""
    prev = getattr(fig, FIG_HOVER_SEEK_CID_ATTR, None)
    if prev is not None:
        try:
            canvas.mpl_disconnect(prev)
        except Exception:
            pass

    last: dict[str, float] = {"t": -1e9, "x": float("nan")}

    def _on_motion(event) -> None:
        from stablewalk.ui.viewers.chart_navigation import (
            FIG_NAV_BLOCK_HOVER_ATTR,
            chart_nav_is_active,
        )

        if is_enabled is not None and not is_enabled():
            return
        if getattr(fig, FIG_MEASURE_MODE_ATTR, False):
            return
        if chart_nav_is_active(fig) or getattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False):
            return
        if event.inaxes is None or event.xdata is None:
            return
        x = float(event.xdata)
        if not np.isfinite(x):
            return
        now = time.monotonic()
        if now - last["t"] < min_dt_s:
            return
        if np.isfinite(last["x"]) and abs(x - last["x"]) < 1e-5:
            return
        last["t"] = now
        last["x"] = x
        try:
            on_seek_time(x)
        except Exception:
            pass

    setattr(
        fig,
        FIG_HOVER_SEEK_CID_ATTR,
        canvas.mpl_connect("motion_notify_event", _on_motion),
    )


def _clear_measure_artists(fig: Figure) -> None:
    for artist in getattr(fig, FIG_MEASURE_ARTISTS_ATTR, None) or []:
        try:
            artist.remove()
        except Exception:
            pass
    setattr(fig, FIG_MEASURE_ARTISTS_ATTR, [])
    setattr(fig, FIG_MEASURE_POINTS_ATTR, [])


def set_chart_measure_mode(fig: Figure, canvas: FigureCanvasTkAgg | None, enabled: bool) -> None:
    """Enable / disable two-click Δt / Δy measurement on a figure."""
    setattr(fig, FIG_MEASURE_MODE_ATTR, bool(enabled))
    if not enabled:
        _clear_measure_artists(fig)
        if canvas is not None:
            canvas.draw_idle()


def attach_chart_measure_mode(fig: Figure, canvas: FigureCanvasTkAgg) -> None:
    """Click two points on the same axes to measure Δt and Δy."""
    prev = getattr(fig, FIG_MEASURE_CID_ATTR, None)
    if prev is not None:
        try:
            canvas.mpl_disconnect(prev)
        except Exception:
            pass

    def _on_press(event) -> None:
        if not getattr(fig, FIG_MEASURE_MODE_ATTR, False):
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return
        from stablewalk.ui.viewers.chart_navigation import chart_nav_is_active

        if chart_nav_is_active(fig):
            return

        points = list(getattr(fig, FIG_MEASURE_POINTS_ATTR, None) or [])
        pt = (event.inaxes, float(event.xdata), float(event.ydata))
        if not points or points[0][0] is not event.inaxes:
            _clear_measure_artists(fig)
            points = [pt]
            mark = event.inaxes.plot(
                [pt[1]], [pt[2]], marker="o", color=ACCENT_ALT, markersize=5, zorder=25
            )[0]
            setattr(fig, FIG_MEASURE_ARTISTS_ATTR, [mark])
            setattr(fig, FIG_MEASURE_POINTS_ATTR, points)
            canvas.draw_idle()
            return

        ax, x0, y0 = points[0]
        x1, y1 = float(event.xdata), float(event.ydata)
        artists = list(getattr(fig, FIG_MEASURE_ARTISTS_ATTR, None) or [])
        (line,) = ax.plot(
            [x0, x1],
            [y0, y1],
            color=ACCENT_ALT,
            linewidth=1.2,
            linestyle="--",
            zorder=24,
        )
        mark = ax.plot(
            [x1], [y1], marker="o", color=ACCENT_ALT, markersize=5, zorder=25
        )[0]
        dt = x1 - x0
        dy = y1 - y0
        text = ax.annotate(
            f"Δt={dt:+.3f} s\nΔy={dy:+.3f}",
            xy=((x0 + x1) * 0.5, (y0 + y1) * 0.5),
            xytext=(10, 10),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=ELEVATED,
                edgecolor=BORDER,
                alpha=0.95,
            ),
            zorder=26,
        )
        artists.extend([line, mark, text])
        setattr(fig, FIG_MEASURE_ARTISTS_ATTR, artists)
        setattr(fig, FIG_MEASURE_POINTS_ATTR, [])
        canvas.draw_idle()

    setattr(
        fig,
        FIG_MEASURE_CID_ATTR,
        canvas.mpl_connect("button_press_event", _on_press),
    )


def attach_chart_interactions(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
    *,
    on_hover_point: Callable[[ChartHoverPoint], None] | None = None,
    on_seek_time: Callable[[float], None] | None = None,
    hover_seek: bool = False,
    hover_seek_enabled: Callable[[], bool] | None = None,
    sync_crosshair: bool = True,
    enable_measure: bool = False,
) -> None:
    """One-shot attach of zoom/pan, hover tooltips, and crosshair."""
    if getattr(fig, FIG_INTERACTIONS_ATTR, False):
        # Re-bind hover callback / canvas ids if figure was reused.
        attach_chart_navigation(fig, canvas)
        attach_chart_hover_tooltips(fig, canvas, on_hover_point=on_hover_point)
        attach_chart_crosshair(fig, canvas, sync_all_axes=sync_crosshair)
        if on_seek_time is not None:
            attach_chart_time_seek(fig, canvas, on_seek_time)
            if hover_seek:
                attach_chart_hover_seek(
                    fig, canvas, on_seek_time, is_enabled=hover_seek_enabled
                )
        if enable_measure:
            attach_chart_measure_mode(fig, canvas)
        return

    attach_chart_navigation(fig, canvas)
    attach_chart_hover_tooltips(fig, canvas, on_hover_point=on_hover_point)
    attach_chart_crosshair(fig, canvas, sync_all_axes=sync_crosshair)
    if on_seek_time is not None:
        attach_chart_time_seek(fig, canvas, on_seek_time)
        if hover_seek:
            attach_chart_hover_seek(
                fig, canvas, on_seek_time, is_enabled=hover_seek_enabled
            )
    if enable_measure:
        attach_chart_measure_mode(fig, canvas)
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
    # Stamp research provenance on figure exports (display metadata only).
    try:
        from stablewalk.ui.scientific_labels import (
            DISCLAIMER_RESEARCH,
            UNITS_COORD_LEGEND,
        )

        fig.text(
            0.5,
            0.01,
            f"{DISCLAIMER_RESEARCH}  ·  {UNITS_COORD_LEGEND}",
            ha="center",
            va="bottom",
            fontsize=9.0,
            color=MUTED,
            wrap=True,
        )
    except Exception:
        pass
    fig.savefig(
        out,
        dpi=200,
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
    metric_key: str | None = None,
    enable_measure: bool = False,
    export_name_getter: Any | None = None,
    ensure_chart: Any | None = None,
) -> tk.Frame:
    """Compact Reset view + Export (+ Metric Details) controls for a 2D chart host.

    ``export_name_getter`` (optional callable) overrides ``export_name`` at export
    time so tabs can export the *active* graph under a distinct filename.
    """
    bar = tk.Frame(parent, bg=ELEVATED, highlightthickness=1, highlightbackground=BORDER)
    bar._chart_export_name = export_name  # type: ignore[attr-defined]

    tk.Label(bar, text="Graph", bg=ELEVATED, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(PAD_XS, 4), pady=2
    )

    def _fig():
        return getattr(gui, fig_attr, None)

    def _canvas():
        return getattr(gui, canvas_attr, None)

    def _ensure_ready_figure():
        fig = _fig()
        if fig is None and callable(ensure_chart):
            try:
                ensure_chart()
            except Exception:
                pass
            fig = _fig()
        return fig

    def _reset() -> None:
        fig = _ensure_ready_figure()
        canvas = _canvas()
        if fig is None:
            return
        reset_chart_view(fig, canvas)

    def _resolve_export_name() -> str:
        if callable(export_name_getter):
            try:
                name = export_name_getter()
                if name:
                    return str(name)
            except Exception:
                pass
        stored = getattr(bar, "_chart_export_name", None)
        return str(stored or export_name)

    def _export() -> None:
        fig = _ensure_ready_figure()
        if fig is None:
            return
        export_chart_figure(
            fig,
            parent=getattr(gui, "root", None),
            default_name=_resolve_export_name(),
        )

    btn_reset = ttk.Button(bar, text="Reset", width=6, command=_reset, takefocus=False)
    btn_reset.pack(side=tk.LEFT, padx=1, pady=2)
    create_tooltip(btn_reset, "Reset zoom / pan (or double-click chart)")

    btn_export = ttk.Button(bar, text="Export", width=7, command=_export, takefocus=False)
    btn_export.pack(side=tk.LEFT, padx=1, pady=2)
    create_tooltip(btn_export, "Export graph as PNG / SVG / PDF")

    if enable_measure:
        measure_var = tk.BooleanVar(value=False)

        def _toggle_measure() -> None:
            fig = _fig()
            canvas = _canvas()
            if fig is None:
                return
            on = bool(measure_var.get())
            set_chart_measure_mode(fig, canvas, on)
            try:
                btn_measure.configure(text="Measure ●" if on else "Measure")
            except tk.TclError:
                pass

        btn_measure = ttk.Button(
            bar,
            text="Measure",
            width=9,
            command=lambda: (measure_var.set(not measure_var.get()), _toggle_measure()),
            takefocus=False,
        )
        btn_measure.pack(side=tk.LEFT, padx=1, pady=2)
        create_tooltip(
            btn_measure,
            "Measurement mode — click two points on one graph for Δt / Δy",
        )
        setattr(bar, "_measure_var", measure_var)

    details_key = metric_key or export_name
    from stablewalk.ui.metric_details import resolve_metric_key
    from stablewalk.ui.tk.metric_details_dialog import open_metric_details_from_gui

    if resolve_metric_key(details_key):
        btn_details = ttk.Button(
            bar,
            text="Details",
            width=8,
            command=lambda: open_metric_details_from_gui(gui, details_key),
            takefocus=False,
        )
        btn_details.pack(side=tk.LEFT, padx=1, pady=2)
        create_tooltip(
            btn_details,
            "Metric Details — definition, units, assumptions, confidence",
        )

    tip = tk.Label(
        bar,
        text="Hover scrub · wheel zoom · drag pan · Shift+drag zoom",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    )
    tip.pack(side=tk.LEFT, padx=(6, PAD_XS), fill=tk.X, expand=True)

    def _sync_tools_hint(_event: object | None = None) -> None:
        try:
            width = int(bar.winfo_width() or 0)
        except tk.TclError:
            return
        if width < 40:
            return
        if width < 520:
            try:
                tip.pack_forget()
            except tk.TclError:
                pass
            return
        if str(tip.winfo_manager() or "") != "pack":
            try:
                tip.pack(side=tk.LEFT, padx=(6, PAD_XS), fill=tk.X, expand=True)
            except tk.TclError:
                pass
        wrap = max(80, width - 280)
        try:
            tip.configure(wraplength=wrap)
        except tk.TclError:
            pass

    bar.bind("<Configure>", _sync_tools_hint, add="+")

    return bar


def reset_all_chart_navigation(gui: Any) -> None:
    """Clear saved zoom on every known 2D chart figure (new session)."""
    for attr in (
        "fig",
        "fig_contact_gait",
        "fig_biomech",
        "fig_joint_motion",
        "fig_motion_workspace",
        "fig_spectral",
    ):
        fig = getattr(gui, attr, None)
        if fig is not None:
            reset_chart_navigation(fig)


__all__ = [
    "attach_chart_crosshair",
    "attach_chart_hover_seek",
    "attach_chart_interactions",
    "attach_chart_measure_mode",
    "build_chart_tools_bar",
    "export_chart_figure",
    "finalize_chart_interactions",
    "reset_all_chart_navigation",
    "set_chart_measure_mode",
]
