"""Collapsible Overview Signal & Spectral Analysis strip.

Starts collapsed so Video | Skeleton | 3D Path keep the viewport. Mutually
exclusive with Joint Graphs. Reuses ``analyze_spectral`` and detected
asymmetry moments — no second pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tkinter as tk
from tkinter import ttk

from stablewalk.analysis.spectral.channels import CHANNEL_BY_ID
from stablewalk.ui.theme import (
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
    create_tooltip,
    disclosure_label,
)

JOINT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("knee", "Knee"),
    ("hip", "Hip"),
    ("ankle", "Ankle"),
    ("heel", "Heel"),
    ("pelvis", "Pelvis"),
)
SIGNAL_OPTIONS: tuple[tuple[str, str], ...] = (
    ("angle", "Angle"),
    ("height", "Height"),
    ("position", "Position"),
    ("foot_ground", "Foot-to-ground"),
)
AXIS_OPTIONS: tuple[str, ...] = ("X", "Y", "Z")

_DOF_JOINTS = frozenset({"hip", "knee", "ankle"})


def resolve_signal_channels(
    joint: str,
    signal: str,
    axis: str,
) -> dict[str, str | None]:
    """Map Overview selectors onto existing spectral / clearance channel ids."""
    joint = str(joint or "knee").lower()
    signal = str(signal or "angle").lower()
    axis_l = str(axis or "Y").lower()[:1]
    if axis_l not in ("x", "y", "z"):
        axis_l = "y"
    if signal == "height":
        axis_l = "y"

    left_id: str | None = None
    right_id: str | None = None
    note: str | None = None
    kind = "spectral"

    if signal == "foot_ground":
        if joint == "pelvis":
            note = "Foot-to-ground is not defined for the pelvis."
        else:
            kind = "clearance"
            left_id = "clearance:left"
            right_id = "clearance:right"
    elif joint == "pelvis":
        if signal == "angle":
            left_id = "pelvis:yaw_deg"
        else:
            left_id = f"pelvis:root:{axis_l}"
    elif signal == "angle":
        if joint not in _DOF_JOINTS:
            note = "Heel has no flexion angle — switch Signal to Height or Position."
        else:
            left_id = f"dof:left_{joint}_flexion"
            right_id = f"dof:right_{joint}_flexion"
    else:
        left_id = f"pos:left_{joint}:{axis_l}"
        right_id = f"pos:right_{joint}:{axis_l}"

    primary = left_id or right_id
    units = "m"
    if primary and primary.startswith("dof:"):
        units = "deg"
    elif primary == "pelvis:yaw_deg":
        units = "deg"
    elif kind == "clearance":
        units = "m"
    spec = CHANNEL_BY_ID.get(str(primary or ""))
    if spec is not None:
        units = spec.units
    return {
        "primary_id": primary,
        "left_id": left_id,
        "right_id": right_id,
        "kind": kind,
        "units": units,
        "note": note,
        "axis": axis_l,
        "joint": joint,
        "signal": signal,
    }


def axis_enabled(signal: str) -> bool:
    return str(signal).lower() == "position"


def build_overview_signal_spectral_panel(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Install the collapsed Signal & Spectral Analysis strip on Overview."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_sections import apply_overview_joint_motion_row_weight
    from stablewalk.ui.viewers.chart_interactions import attach_chart_interactions
    from stablewalk.ui.viewers.overview_signal_spectral_charts import (
        build_signal_spectral_axes,
    )

    host = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    host.grid(row=5, column=0, sticky="nsew", pady=(PAD_XS, 0))
    host.grid_remove()
    host.columnconfigure(0, weight=1)
    host.rowconfigure(1, weight=1)
    gui.overview_signal_spectral_panel = host
    gui._overview_signal_spectral_expanded = False
    gui._overview_signal_spectral_ready = False
    gui._overview_sigspec_joint = "knee"
    gui._overview_sigspec_signal = "angle"
    gui._overview_sigspec_axis = "Y"

    header = tk.Frame(host, bg=PANEL, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew")
    header.columnconfigure(1, weight=1)

    tk.Label(
        header,
        text="Signal & Spectral Analysis",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="w")

    body = ttk.LabelFrame(
        host,
        text=" Signal & Spectral Analysis ",
        style="Card.TLabelframe",
        padding=(PAD_XS, PAD_XS),
    )
    body.columnconfigure(0, weight=1)
    body.rowconfigure(1, weight=1)
    gui._overview_signal_spectral_body = body

    tools = tk.Frame(body, bg=PANEL, highlightthickness=0)
    tools.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    for col in (1, 3, 5):
        tools.columnconfigure(col, weight=0)
    tools.columnconfigure(7, weight=1)

    gui.var_overview_sigspec_joint = tk.StringVar(value="Knee")
    gui.var_overview_sigspec_signal = tk.StringVar(value="Angle")
    gui.var_overview_sigspec_axis = tk.StringVar(value="Y")

    def _combo(parent: tk.Misc, var: tk.StringVar, values: tuple[str, ...], width: int) -> ttk.Combobox:
        box = ttk.Combobox(
            parent,
            textvariable=var,
            values=values,
            state="readonly",
            width=width,
            takefocus=False,
        )
        return box

    tk.Label(tools, text="Joint", bg=PANEL, fg=TEXT_SECONDARY, font=FONT_UI_XS).grid(
        row=0, column=0, padx=(0, PAD_XS)
    )
    cmb_joint = _combo(tools, gui.var_overview_sigspec_joint, tuple(v for _k, v in JOINT_OPTIONS), 10)
    cmb_joint.grid(row=0, column=1, padx=(0, PAD_SM))
    tk.Label(tools, text="Signal", bg=PANEL, fg=TEXT_SECONDARY, font=FONT_UI_XS).grid(
        row=0, column=2, padx=(0, PAD_XS)
    )
    cmb_signal = _combo(
        tools, gui.var_overview_sigspec_signal, tuple(v for _k, v in SIGNAL_OPTIONS), 14
    )
    cmb_signal.grid(row=0, column=3, padx=(0, PAD_SM))
    tk.Label(tools, text="Axis", bg=PANEL, fg=TEXT_SECONDARY, font=FONT_UI_XS).grid(
        row=0, column=4, padx=(0, PAD_XS)
    )
    cmb_axis = _combo(tools, gui.var_overview_sigspec_axis, AXIS_OPTIONS, 4)
    cmb_axis.grid(row=0, column=5, padx=(0, PAD_SM))
    gui.cmb_overview_sigspec_joint = cmb_joint
    gui.cmb_overview_sigspec_signal = cmb_signal
    gui.cmb_overview_sigspec_axis = cmb_axis
    create_tooltip(cmb_joint, "Anatomical joint family used for the five graphs.")
    create_tooltip(cmb_signal, "Angle (flexion), Height (vertical), Position, or foot-to-ground distance.")
    create_tooltip(cmb_axis, "Position axis. Disabled for Angle, Height, and Foot-to-ground.")

    gui.lbl_overview_sigspec_status = tk.Label(
        tools,
        text="Derived from the loaded pose series — not a diagnosis",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    )
    gui.lbl_overview_sigspec_status.grid(row=0, column=7, sticky="e")

    gui.fig_overview_sigspec = None
    gui.canvas_overview_sigspec = None
    gui._overview_sigspec_axes = None

    def _seek_from_axes(time_s: float) -> None:
        seek = getattr(gui, "_seek_motion_workspace_time", None)
        if callable(seek):
            seek(float(time_s))

    def _ensure_figure() -> bool:
        if getattr(gui, "_overview_signal_spectral_ready", False) and gui.fig_overview_sigspec is not None:
            return True
        fig = Figure(figsize=(10.4, 4.2), dpi=110, facecolor=PANEL)
        axes = build_signal_spectral_axes(fig)
        canvas = FigureCanvasTkAgg(fig, master=body)
        widget = canvas.get_tk_widget()
        widget.configure(bg=PANEL, highlightthickness=0)
        widget.grid(row=1, column=0, sticky="nsew")
        attach_chart_interactions(
            fig,
            canvas,
            sync_crosshair=False,
            on_seek_time=_seek_from_axes,
        )
        try:
            from stablewalk.ui.tk.dashboard_layout import _bind_figure_resize

            _bind_figure_resize(canvas, fig, min_px=80)
        except Exception:
            pass
        gui.fig_overview_sigspec = fig
        gui.canvas_overview_sigspec = canvas
        gui._overview_sigspec_axes = axes
        gui._overview_sigspec_canvas_widget = widget
        gui._overview_signal_spectral_ready = True
        return True

    gui._ensure_overview_signal_spectral = _ensure_figure

    def _id_for_label(options: tuple[tuple[str, str], ...], label: str, default: str) -> str:
        for key, lab in options:
            if lab == label:
                return key
        return default

    def _sync_axis_state() -> None:
        signal = _id_for_label(SIGNAL_OPTIONS, gui.var_overview_sigspec_signal.get(), "angle")
        try:
            if axis_enabled(signal):
                cmb_axis.configure(state="readonly")
            else:
                if signal == "height":
                    gui.var_overview_sigspec_axis.set("Y")
                cmb_axis.configure(state="disabled")
        except tk.TclError:
            pass

    def _on_selector(_event: object | None = None) -> None:
        gui._overview_sigspec_joint = _id_for_label(
            JOINT_OPTIONS, gui.var_overview_sigspec_joint.get(), "knee"
        )
        gui._overview_sigspec_signal = _id_for_label(
            SIGNAL_OPTIONS, gui.var_overview_sigspec_signal.get(), "angle"
        )
        gui._overview_sigspec_axis = str(gui.var_overview_sigspec_axis.get() or "Y")
        _sync_axis_state()
        render_overview_signal_spectral_panel(gui)

    cmb_joint.bind("<<ComboboxSelected>>", _on_selector)
    cmb_signal.bind("<<ComboboxSelected>>", _on_selector)
    cmb_axis.bind("<<ComboboxSelected>>", _on_selector)
    _sync_axis_state()

    def _apply_expanded(expanded: bool) -> None:
        gui._overview_signal_spectral_expanded = bool(expanded)
        if expanded:
            if bool(getattr(gui, "_overview_joint_motion_expanded", False)):
                gui._overview_analysis_switching = True
                try:
                    apply_graphs = getattr(gui, "_apply_overview_joint_motion_expanded", None)
                    if callable(apply_graphs):
                        apply_graphs(False)
                finally:
                    gui._overview_analysis_switching = False
            _ensure_figure()
            host.grid(row=5, column=0, sticky="nsew", pady=(PAD_XS, 0))
            header.grid(row=0, column=0, sticky="ew")
            body.grid(row=1, column=0, sticky="nsew", pady=(2, 0))
            apply_overview_joint_motion_row_weight(parent, expanded=True, row=5)
            sched = getattr(gui, "_schedule_spectral_analysis", None)
            if callable(sched) and getattr(gui, "_spectral_report", None) is None:
                try:
                    sched(False)
                except Exception:
                    pass
            render_overview_signal_spectral_panel(gui)
        else:
            body.grid_remove()
            header.grid_remove()
            host.grid_remove()
            if not bool(getattr(gui, "_overview_joint_motion_expanded", False)):
                apply_overview_joint_motion_row_weight(parent, expanded=False, row=5)
        btn = getattr(gui, "_btn_overview_signal_spectral_toggle", None)
        if btn is not None:
            try:
                btn.configure(
                    text=disclosure_label("Signal & Spectral", expanded=expanded)
                )
            except tk.TclError:
                pass
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

    def _toggle() -> None:
        _apply_expanded(not bool(getattr(gui, "_overview_signal_spectral_expanded", False)))

    toggle = ttk.Button(
        header,
        text=disclosure_label("Signal & Spectral", expanded=False),
        style="Compact.TButton",
        width=18,
        command=_toggle,
        takefocus=False,
    )
    toggle.grid(row=0, column=2, sticky="e", padx=(PAD_XS, 0))
    gui._btn_overview_signal_spectral_toggle = toggle
    gui._toggle_overview_signal_spectral = _toggle
    gui._apply_overview_signal_spectral_expanded = _apply_expanded

    body.grid_remove()
    header.grid_remove()
    host.grid_remove()
    gui._render_overview_signal_spectral = lambda: render_overview_signal_spectral_panel(gui)
    gui._update_overview_signal_spectral_playhead = (
        lambda: update_overview_signal_spectral_playhead(gui)
    )
    return host


def _series_xy(report: Any, channel_id: str | None) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if report is None or not channel_id:
        return None, None, ""
    bundle = getattr(report, "bundle", None)
    series = bundle.get(channel_id) if bundle is not None else None
    if series is None:
        return None, None, ""
    units = getattr(getattr(series, "spec", None), "units", "") or ""
    return (
        np.asarray(series.times_s, dtype=float),
        np.asarray(series.processed, dtype=float),
        units,
    )


def _clearance_series(gui: Any, side: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    cycles = getattr(gui, "_gait_cycle", None)
    frames = list(getattr(cycles, "per_frame", None) or [])
    if not frames:
        return None, None
    times: list[float] = []
    vals: list[float] = []
    for frame in frames:
        sample = getattr(frame, side, None)
        dist = getattr(sample, "foot_clearance_m", None) if sample is not None else None
        times.append(float(getattr(frame, "time_s", 0.0)))
        vals.append(float(dist) if dist is not None else float("nan"))
    t = np.asarray(times, dtype=float)
    y = np.asarray(vals, dtype=float)
    if not np.isfinite(y).any():
        return None, None
    return t, y


def _playhead(gui: Any):
    try:
        from stablewalk.ui.viewers.chart_playhead import playhead_from_gui

        return playhead_from_gui(gui)
    except Exception:
        return None


def render_overview_signal_spectral_panel(gui: Any, *, playhead_only: bool = False) -> None:
    if not bool(getattr(gui, "_overview_signal_spectral_expanded", False)):
        return
    ensure = getattr(gui, "_ensure_overview_signal_spectral", None)
    if callable(ensure):
        ensure()
    fig = getattr(gui, "fig_overview_sigspec", None)
    axes = getattr(gui, "_overview_sigspec_axes", None)
    canvas = getattr(gui, "canvas_overview_sigspec", None)
    if fig is None or axes is None or canvas is None:
        return
    if playhead_only:
        from stablewalk.ui.viewers.chart_playhead import update_playhead_on_axes

        ph = _playhead(gui)
        if ph is None:
            return
        time_axes = [ax for ax in axes if bool(getattr(ax, "_sw_time_seek", False))]
        try:
            if update_playhead_on_axes(time_axes, ph):
                canvas.draw_idle()
                return
        except Exception:
            pass
    mapping = resolve_signal_channels(
        str(getattr(gui, "_overview_sigspec_joint", "knee")),
        str(getattr(gui, "_overview_sigspec_signal", "angle")),
        str(getattr(gui, "_overview_sigspec_axis", "Y")),
    )
    report = getattr(gui, "_spectral_report", None)
    note = mapping.get("note")
    left_id = mapping.get("left_id")
    right_id = mapping.get("right_id")
    primary = mapping.get("primary_id") or left_id
    units = str(mapping.get("units") or "")
    times_lr = left_vals = right_vals = None
    if mapping.get("kind") == "clearance":
        times_lr, left_vals = _clearance_series(gui, "left")
        t_r, right_vals = _clearance_series(gui, "right")
        if times_lr is None:
            times_lr = t_r
        # Build a lightweight spectral-like draw for the primary (left) by
        # falling back to processed heel height if the spectral report exists.
        if primary in (None, "clearance:left") and report is not None:
            primary = "pos:left_heel:y"
    else:
        times_lr, left_vals, units_l = _series_xy(report, str(left_id) if left_id else None)
        t_r, right_vals, units_r = _series_xy(report, str(right_id) if right_id else None)
        units = units or units_l or units_r
        if times_lr is None:
            times_lr = t_r

    asym = getattr(gui, "_gait_asymmetry_moments", None)
    asym_times: list[float] = []
    asym_scores: list[float | None] = []
    moments = ()
    if asym is not None:
        moments = tuple(getattr(asym, "moments", None) or ())
        for sample in getattr(asym, "per_frame", None) or []:
            asym_times.append(float(getattr(sample, "time_s", 0.0)))
            score = getattr(sample, "score", None)
            used = bool(getattr(sample, "used", False))
            asym_scores.append(float(score) if used and score is not None else None)

    from stablewalk.ui.viewers.overview_signal_spectral_charts import (
        render_overview_signal_spectral,
    )

    empty = None
    if note:
        empty = str(note)
    elif report is None:
        empty = "Load and analyze a walking video."
    render_overview_signal_spectral(
        fig,
        axes,
        report=None if note else report,
        channel_id=str(primary or ""),
        times_lr=times_lr,
        left_vals=left_vals,
        right_vals=right_vals,
        lr_ylabel=units,
        left_label="Left",
        right_label="Right",
        asym_times=asym_times,
        asym_scores=asym_scores,
        moments=moments,
        playhead=_playhead(gui),
        empty_message=empty,
    )
    try:
        from stablewalk.ui.viewers.chart_interactions import finalize_chart_interactions

        finalize_chart_interactions(fig, canvas)
    except Exception:
        pass
    try:
        canvas.draw_idle()
    except Exception:
        pass
    status = getattr(gui, "lbl_overview_sigspec_status", None)
    if status is not None:
        text = str(note) if note else "Click a time graph to jump video and 3D skeleton"
        try:
            status.configure(text=text)
        except tk.TclError:
            pass


def update_overview_signal_spectral_playhead(gui: Any) -> None:
    render_overview_signal_spectral_panel(gui, playhead_only=True)
