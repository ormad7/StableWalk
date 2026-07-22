"""Compact camera controls for the 3D trajectory visualization.

Provides Reset / Zoom / Pan / Rotate without overlapping the path. Mouse-drag
orbit is also supported (matplotlib Axes3D default) and remembered so playback
redraws stay synchronized without fighting the user's viewpoint.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_UI_XS,
    MUTED,
    PAD_XS,
    create_tooltip,
)


def _projection_mode(gui: Any) -> str:
    var = getattr(gui, "var_dof_projection", None)
    if var is not None:
        try:
            return str(var.get() or "3D")
        except Exception:
            pass
    return "3D"


def _redraw(gui: Any) -> None:
    refresh = getattr(gui, "_refresh_motion_trajectory_on_frame", None)
    if callable(refresh):
        refresh(force_draw=True)
        return
    for name in ("canvas_dof_traj", "canvas_dof_traj_overview"):
        canvas = getattr(gui, name, None)
        if canvas is not None:
            try:
                canvas.draw_idle()
            except Exception:
                pass


def _with_ax(gui: Any, ax: Any, fn: Callable[[Any], None]) -> None:
    if ax is None:
        return
    fn(ax)
    _redraw(gui)


def build_trajectory_camera_toolbar(
    gui: Any,
    parent: tk.Misc,
    *,
    ax_attr: str = "ax_dof_traj",
    canvas_attr: str = "canvas_dof_traj",
    compact: bool = False,
    show_rotate: bool = True,
) -> tk.Frame:
    """Build a single-row camera toolbar above/beside the trajectory canvas."""
    from stablewalk.ui.viewers.dof_trajectory_3d import (
        TRAJECTORY_CAMERA_PRESETS,
        clear_trajectory_camera_state,
        pan_trajectory_camera,
        remember_trajectory_camera,
        reset_trajectory_camera,
        rotate_trajectory_camera,
        set_trajectory_camera_preset,
        zoom_trajectory_camera,
    )

    bar = tk.Frame(
        parent,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )

    pady = 1 if compact else 2
    tk.Label(bar, text="Cam", bg=ELEVATED, fg=MUTED, font=FONT_UI_XS).pack(
        side=tk.LEFT, padx=(PAD_XS, 2 if compact else 4), pady=pady
    )

    def _ax():
        return getattr(gui, ax_attr, None)

    def _btn(text: str, tip: str, command: Callable[[], None]) -> None:
        b = ttk.Button(
            bar,
            text=text,
            width=2 if compact else 3,
            command=command,
            takefocus=False,
        )
        b.pack(side=tk.LEFT, padx=1, pady=pady)
        create_tooltip(b, tip)

    def _do_reset() -> None:
        ax = _ax()
        if ax is None:
            return
        reset_trajectory_camera(ax, projection_mode=_projection_mode(gui))
        _redraw(gui)
        # Overview dock: re-fit so orbit/zoom never leaves clipped axes.
        if ax_attr == "ax_dof_traj_overview":
            render = getattr(gui, "_render_overview_traj_canvas", None)
            if callable(render):
                try:
                    render(force=True)
                except Exception:
                    pass

    def _do_zoom(factor: float) -> None:
        _with_ax(gui, _ax(), lambda a: zoom_trajectory_camera(a, factor))
        if ax_attr == "ax_dof_traj_overview":
            render = getattr(gui, "_render_overview_traj_canvas", None)
            if callable(render):
                try:
                    render(force=True)
                except Exception:
                    pass

    def _do_pan(dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
        _with_ax(gui, _ax(), lambda a: pan_trajectory_camera(a, dx=dx, dy=dy, dz=dz))

    def _do_rot(d_elev: float = 0.0, d_azim: float = 0.0) -> None:
        _with_ax(
            gui, _ax(), lambda a: rotate_trajectory_camera(a, d_elev=d_elev, d_azim=d_azim)
        )
        if ax_attr == "ax_dof_traj_overview":
            render = getattr(gui, "_render_overview_traj_canvas", None)
            if callable(render):
                try:
                    render(force=True)
                except Exception:
                    pass

    _btn("⟲", "Reset camera (Perspective view)", _do_reset)
    _btn("+", "Zoom in", lambda: _do_zoom(1.15))
    _btn("−", "Zoom out", lambda: _do_zoom(1.0 / 1.15))

    tk.Frame(bar, bg=BORDER, width=1, highlightthickness=0).pack(
        side=tk.LEFT, fill=tk.Y, padx=2 if compact else 4, pady=2 if compact else 3
    )

    def _do_preset(name: str) -> None:
        ax = _ax()
        if ax is None:
            return
        set_trajectory_camera_preset(ax, name)
        _redraw(gui)
        if ax_attr == "ax_dof_traj_overview":
            render = getattr(gui, "_render_overview_traj_canvas", None)
            if callable(render):
                try:
                    render(force=True)
                except Exception:
                    pass

    for preset_name in ("Perspective", "Side", "Front", "Top"):
        if preset_name not in TRAJECTORY_CAMERA_PRESETS:
            continue
        short = {"Perspective": "Persp", "Side": "Side", "Front": "Front", "Top": "Top"}[
            preset_name
        ]
        b = ttk.Button(
            bar,
            text=short,
            width=4 if compact else 5,
            command=lambda n=preset_name: _do_preset(n),
            takefocus=False,
        )
        b.pack(side=tk.LEFT, padx=1, pady=pady)
        create_tooltip(b, f"Camera preset: {preset_name}")

    if not compact:
        tk.Frame(bar, bg=BORDER, width=1, highlightthickness=0).pack(
            side=tk.LEFT, fill=tk.Y, padx=4, pady=3
        )

        _btn("←", "Pan left (−X)", lambda: _do_pan(dx=-1.0))
        _btn("→", "Pan right (+X)", lambda: _do_pan(dx=1.0))
        _btn("↑", "Pan up (+Y)", lambda: _do_pan(dy=1.0))
        _btn("↓", "Pan down (−Y)", lambda: _do_pan(dy=-1.0))
        _btn("⤒", "Pan forward (+Z)", lambda: _do_pan(dz=1.0))
        _btn("⤓", "Pan back (−Z)", lambda: _do_pan(dz=-1.0))

    # Drag-to-orbit is always available on the canvas, so the explicit rotate /
    # tilt buttons are optional — dropping them keeps a compact bar narrow enough
    # that its panel column does not starve the neighbouring chart.
    if show_rotate:
        tk.Frame(bar, bg=BORDER, width=1, highlightthickness=0).pack(
            side=tk.LEFT, fill=tk.Y, padx=2 if compact else 4, pady=2 if compact else 3
        )

        _btn("↻", "Rotate right", lambda: _do_rot(d_azim=-12.0))
        _btn("↺", "Rotate left", lambda: _do_rot(d_azim=12.0))
        _btn("◠", "Tilt up", lambda: _do_rot(d_elev=8.0))
        _btn("◡", "Tilt down", lambda: _do_rot(d_elev=-8.0))

    if not compact:
        tip_lbl = tk.Label(
            bar,
            text="Drag plot to orbit",
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_UI_XS,
        )
        tip_lbl.pack(side=tk.LEFT, padx=(6, PAD_XS))

    # Keep a handle for projection/mode changes that should drop manual view.
    store = getattr(gui, "_traj_camera_axes", None)
    if store is None:
        store = []
        gui._traj_camera_axes = store
    if ax_attr not in store:
        store.append(ax_attr)

    # Canvas may not exist yet when the bar is built; bind later via
    # ``bind_trajectory_camera_canvas``.
    bar._sw_ax_attr = ax_attr  # type: ignore[attr-defined]
    bar._sw_canvas_attr = canvas_attr  # type: ignore[attr-defined]
    bind_trajectory_camera_canvas(gui, canvas_attr=canvas_attr, ax_attr=ax_attr)

    _ = clear_trajectory_camera_state
    return bar


def bind_trajectory_camera_canvas(
    gui: Any,
    *,
    canvas_attr: str,
    ax_attr: str,
) -> None:
    """Remember mouse-orbit so playback redraws keep the same viewpoint."""
    from stablewalk.ui.viewers.dof_trajectory_3d import remember_trajectory_camera

    canvas = getattr(gui, canvas_attr, None)
    if canvas is None or getattr(canvas, "_sw_traj_cam_bound", False):
        return

    def _on_release(_event: object = None) -> None:
        ax = getattr(gui, ax_attr, None)
        if ax is not None:
            remember_trajectory_camera(ax)

    try:
        canvas.mpl_connect("button_release_event", _on_release)
        canvas._sw_traj_cam_bound = True  # type: ignore[attr-defined]
    except Exception:
        pass


def clear_all_trajectory_cameras(gui: Any) -> None:
    """Clear camera state on every registered trajectory axes (e.g. view change)."""
    from stablewalk.ui.viewers.dof_trajectory_3d import clear_trajectory_camera_state

    for attr in getattr(gui, "_traj_camera_axes", []) or []:
        clear_trajectory_camera_state(getattr(gui, attr, None))
    clear_trajectory_camera_state(getattr(gui, "ax_dof_traj", None))
    clear_trajectory_camera_state(getattr(gui, "ax_dof_traj_overview", None))


__all__ = [
    "bind_trajectory_camera_canvas",
    "build_trajectory_camera_toolbar",
    "clear_all_trajectory_cameras",
]
