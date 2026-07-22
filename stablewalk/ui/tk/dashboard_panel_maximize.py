"""Per-panel maximize / restore for major StableWalk visualization panels."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Any, Callable

from stablewalk.ui.theme import (
    BORDER,
    ELEVATED,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    TEXT,
    create_tooltip,
)

PANEL_TITLES: dict[str, str] = {
    "video": "Original Video",
    "skeleton": "3D Reconstruction",
    "path_3d": "3D Path",
    "motion": "Motion Analysis",
    "biomechanics": "Biomechanics",
    "results_summary": "Results Summary",
    "pipeline": "Advanced Pipeline",
}

_CHROME_HEIGHT = 36


@dataclass
class _MaximizeState:
    panel_id: str
    widget: tk.Misc
    manager: str
    layout_info: dict[str, Any]
    was_playing: bool
    playback_pos: float | None
    host: tk.Misc
    chrome: tk.Frame
    configure_bind: str | None = None
    esc_bind: str | None = None


def is_panel_maximized(gui: Any) -> bool:
    return getattr(gui, "_panel_maximize_state", None) is not None


def current_maximized_panel_id(gui: Any) -> str | None:
    state = getattr(gui, "_panel_maximize_state", None)
    return None if state is None else str(state.panel_id)


def _resolve_path_3d_widget(gui: Any) -> Any:
    overview = getattr(gui, "overview_traj_panel", None)
    motion = getattr(gui, "traj_panel", None)
    if getattr(gui, "_overview_traj_dock_visible", False) and overview is not None:
        return overview
    if motion is not None:
        return motion
    return overview


def _resolve_panel_widget(gui: Any, panel_id: str) -> tk.Misc | None:
    resolvers: dict[str, Callable[[Any], Any]] = {
        "video": lambda g: getattr(g, "video_frame", None),
        "skeleton": lambda g: getattr(g, "skel_frame", None),
        "path_3d": _resolve_path_3d_widget,
        "motion": lambda g: getattr(g, "_section_kinematic", None),
        "biomechanics": lambda g: getattr(g, "_biomechanics_panel_host", None)
        or getattr(g, "_section_biomechanics", None)
        or getattr(g, "_tab_biomechanics", None),
        "results_summary": lambda g: getattr(g, "_results_summary_panel_host", None)
        or getattr(g, "_section_results_summary", None)
        or getattr(g, "_tab_results_summary", None),
        "pipeline": lambda g: getattr(g, "_pipeline_panel_host", None)
        or getattr(g, "_section_pipeline_status_advanced", None)
        or getattr(g, "_section_engineering_dashboard", None)
        or getattr(g, "_tab_advanced_content", None),
    }
    resolver = resolvers.get(panel_id)
    if resolver is None:
        return None
    widget = resolver(gui)
    if widget is None:
        return None
    try:
        if not widget.winfo_exists():
            return None
    except tk.TclError:
        return None
    return widget


def _maximize_host(gui: Any) -> tk.Misc | None:
    host = getattr(gui, "_dashboard_main", None)
    if host is None:
        host = getattr(gui, "root", None)
    return host


def _ensure_panel_tab(gui: Any, panel_id: str, widget: tk.Misc) -> None:
    """Switch to the notebook tab that owns the panel before maximizing."""
    from stablewalk.ui.tk.dashboard_notebook import (
        TAB_ADVANCED,
        TAB_BIOMECHANICS,
        TAB_MOTION,
        TAB_OVERVIEW,
        TAB_RESULTS_SUMMARY,
        select_dashboard_tab,
    )

    if panel_id in {"video", "skeleton"}:
        select_dashboard_tab(gui, TAB_OVERVIEW)
        return
    if panel_id == "path_3d":
        overview = getattr(gui, "overview_traj_panel", None)
        if widget is overview:
            select_dashboard_tab(gui, TAB_OVERVIEW)
        else:
            select_dashboard_tab(gui, TAB_MOTION)
        return
    tab_map = {
        "motion": TAB_MOTION,
        "biomechanics": TAB_BIOMECHANICS,
        "results_summary": TAB_RESULTS_SUMMARY,
        "pipeline": TAB_ADVANCED,
    }
    tab = tab_map.get(panel_id)
    if tab is not None:
        select_dashboard_tab(gui, tab)


def _capture_playback(gui: Any) -> tuple[bool, float | None]:
    from stablewalk.ui.tk.dashboard_overview_view_mode import _capture_playback_position

    was_playing = bool(getattr(gui, "playing", False))
    return was_playing, _capture_playback_position(gui)


def _restore_playback(gui: Any, *, was_playing: bool, position: float | None) -> None:
    from stablewalk.ui.tk.dashboard_overview_view_mode import _restore_playback_position

    _restore_playback_position(gui, position)
    if was_playing and not bool(getattr(gui, "playing", False)):
        toggle = getattr(gui, "_toggle_play", None)
        if toggle is not None:
            try:
                toggle()
            except Exception:
                pass
    elif not was_playing and bool(getattr(gui, "playing", False)):
        stop = getattr(gui, "_stop_playback", None)
        if stop is not None:
            try:
                stop()
            except Exception:
                pass


def _stash_geometry(widget: tk.Misc) -> tuple[str, dict[str, Any]]:
    try:
        info = widget.grid_info()
        if info:
            return "grid", dict(info)
    except tk.TclError:
        pass
    try:
        info = widget.pack_info()
        if info:
            return "pack", dict(info)
    except tk.TclError:
        pass
    try:
        info = widget.place_info()
        if info:
            return "place", dict(info)
    except tk.TclError:
        pass
    return "none", {}


def _geometry_opts(info: dict[str, Any]) -> dict[str, Any]:
    """Translate Tk geometry info keys to tkinter kwargs (``in`` → ``in_``)."""
    opts: dict[str, Any] = {}
    for key, value in info.items():
        opts["in_" if key == "in" else key] = value
    return opts


def _reapply_geometry(widget: tk.Misc, manager: str, info: dict[str, Any]) -> None:
    try:
        widget.place_forget()
    except tk.TclError:
        pass
    try:
        widget.grid_forget()
    except tk.TclError:
        pass
    try:
        widget.pack_forget()
    except tk.TclError:
        pass

    if manager == "grid" and info:
        try:
            widget.grid(**_geometry_opts(info))
        except tk.TclError:
            try:
                widget.grid()
            except tk.TclError:
                pass
    elif manager == "pack" and info:
        try:
            widget.pack(**_geometry_opts(info))
        except tk.TclError:
            try:
                widget.pack()
            except tk.TclError:
                pass
    elif manager == "place" and info:
        try:
            widget.place(**_geometry_opts(info))
        except tk.TclError:
            pass


def _fit_after_layout(gui: Any, panel_id: str) -> None:
    root = getattr(gui, "root", None)

    def _do() -> None:
        if panel_id == "video":
            refit = getattr(gui, "_on_video_label_resize", None)
            if refit is not None:
                try:
                    refit()
                except Exception:
                    pass
            try:
                from stablewalk.ui.tk.dashboard_overview_view_mode import (
                    _refresh_synced_overlays,
                )

                _refresh_synced_overlays(gui)
            except Exception:
                pass
        if panel_id == "skeleton":
            fit = getattr(gui, "_fit_skeleton_canvas", None)
            if fit is not None:
                try:
                    fit()
                except Exception:
                    pass
            try:
                from stablewalk.ui.tk.dashboard_overview_view_mode import (
                    _refresh_synced_overlays,
                )

                _refresh_synced_overlays(gui)
            except Exception:
                pass
        if panel_id in {"motion", "path_3d"}:
            for name in (
                "_update_chart",
                "_lift_trajectory_canvas",
                "_render_overview_traj_canvas",
            ):
                fn = getattr(gui, name, None)
                if fn is None:
                    continue
                try:
                    if name.endswith("_traj_canvas") or name.endswith(
                        "trajectory_canvas"
                    ):
                        try:
                            fn(force=True)
                        except TypeError:
                            fn()
                    else:
                        fn()
                except Exception:
                    pass
            render = getattr(gui, "_render_dof_traj_canvas", None)
            if render is not None:
                try:
                    render(force=True)
                except TypeError:
                    try:
                        render()
                    except Exception:
                        pass
                except Exception:
                    pass
            refresh = getattr(gui, "_refresh_motion_trajectory_on_frame", None)
            if refresh is not None:
                try:
                    refresh(force_draw=True)
                except TypeError:
                    try:
                        refresh()
                    except Exception:
                        pass
                except Exception:
                    pass
            try:
                from stablewalk.ui.tk.dashboard_overview_view_mode import (
                    _refresh_synced_overlays,
                )

                _refresh_synced_overlays(gui)
            except Exception:
                pass
        if panel_id == "biomechanics":
            fn = getattr(gui, "_update_biomechanics_chart", None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
        if panel_id == "results_summary":
            fn = getattr(gui, "_update_results_summary_panel", None) or getattr(
                gui, "update_results_summary_panel", None
            )
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
        if panel_id == "pipeline":
            sync = getattr(gui, "_sync_tab_advanced_scroll", None)
            if sync is not None:
                try:
                    sync()
                except Exception:
                    pass
            update = getattr(gui, "_update_pipeline_status_panel", None)
            if update is not None:
                try:
                    update()
                except Exception:
                    pass

    if root is None:
        _do()
        return
    try:
        root.after_idle(_do)
        root.after(50, _do)
    except tk.TclError:
        _do()


def _relayout_maximized(gui: Any) -> None:
    state: _MaximizeState | None = getattr(gui, "_panel_maximize_state", None)
    if state is None:
        return
    host = state.host
    try:
        host.update_idletasks()
        width = max(1, int(host.winfo_width()))
        height = max(1, int(host.winfo_height()))
    except tk.TclError:
        return
    chrome_h = _CHROME_HEIGHT
    try:
        state.chrome.place(in_=host, x=0, y=0, width=width, height=chrome_h)
        state.widget.place(
            in_=host,
            x=0,
            y=chrome_h,
            width=width,
            height=max(1, height - chrome_h),
        )
        state.chrome.lift()
        state.widget.lift()
    except tk.TclError:
        return
    _fit_after_layout(gui, state.panel_id)


def maximize_panel(gui: Any, panel_id: str) -> bool:
    """Maximize a single registered panel. Returns True if maximized."""
    if panel_id not in PANEL_TITLES:
        return False
    if is_panel_maximized(gui):
        state = gui._panel_maximize_state
        if state.panel_id == panel_id:
            return True
        restore_panel(gui)

    widget = _resolve_panel_widget(gui, panel_id)
    host = _maximize_host(gui)
    if widget is None or host is None:
        return False

    _ensure_panel_tab(gui, panel_id, widget)
    # Re-resolve after tab switch (mount sync may remap trajectory hosts).
    widget = _resolve_panel_widget(gui, panel_id)
    if widget is None:
        return False

    was_playing, playback_pos = _capture_playback(gui)
    manager, layout_info = _stash_geometry(widget)

    chrome = tk.Frame(
        host,
        bg=ELEVATED,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    chrome.columnconfigure(0, weight=1)

    title = PANEL_TITLES.get(panel_id, panel_id)
    tk.Label(
        chrome,
        text=f"{title}  ·  Full Screen",
        bg=ELEVATED,
        fg=TEXT,
        font=FONT_UI_SM,
        anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=PAD_SM, pady=PAD_XS)

    tk.Label(
        chrome,
        text="ESC or Restore to exit",
        bg=ELEVATED,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    ).grid(row=0, column=1, sticky="e", padx=(0, PAD_SM))

    restore_btn = ttk.Button(
        chrome,
        text="Restore",
        style="Compact.TButton",
        command=lambda: restore_panel(gui),
    )
    restore_btn.grid(row=0, column=2, sticky="e", padx=(0, PAD_SM), pady=PAD_XS)
    create_tooltip(restore_btn, "Exit full screen (Esc)")

    state = _MaximizeState(
        panel_id=panel_id,
        widget=widget,
        manager=manager,
        layout_info=layout_info,
        was_playing=was_playing,
        playback_pos=playback_pos,
        host=host,
        chrome=chrome,
    )
    gui._panel_maximize_state = state

    def _on_host_configure(_event: object = None) -> None:
        _relayout_maximized(gui)

    try:
        state.configure_bind = host.bind("<Configure>", _on_host_configure, add="+")
    except tk.TclError:
        state.configure_bind = None

    root = getattr(gui, "root", None)
    if root is not None:
        try:
            state.esc_bind = root.bind(
                "<Escape>", lambda _e: restore_panel(gui), add="+"
            )
        except tk.TclError:
            state.esc_bind = None

    _relayout_maximized(gui)
    _restore_playback(gui, was_playing=was_playing, position=playback_pos)

    status = getattr(gui, "status", None)
    if status is not None:
        try:
            status.configure(text=f"{title} maximized — press Esc to restore")
        except Exception:
            pass

    _sync_maximize_buttons(gui)
    return True


def restore_panel(gui: Any) -> bool:
    """Restore the maximized panel to its prior layout. Returns True if restored."""
    state: _MaximizeState | None = getattr(gui, "_panel_maximize_state", None)
    if state is None:
        return False

    live_playing, live_pos = _capture_playback(gui)
    was_playing = live_playing
    playback_pos = live_pos if live_pos is not None else state.playback_pos

    host = state.host
    if state.configure_bind is not None:
        try:
            host.unbind("<Configure>", state.configure_bind)
        except tk.TclError:
            pass

    root = getattr(gui, "root", None)
    if root is not None and state.esc_bind is not None:
        try:
            root.unbind("<Escape>", state.esc_bind)
        except tk.TclError:
            pass

    try:
        state.chrome.place_forget()
        state.chrome.destroy()
    except tk.TclError:
        pass

    try:
        state.widget.place_forget()
    except tk.TclError:
        pass

    _reapply_geometry(state.widget, state.manager, state.layout_info)
    gui._panel_maximize_state = None

    _restore_playback(gui, was_playing=was_playing, position=playback_pos)
    _fit_after_layout(gui, state.panel_id)

    overview_traj = getattr(gui, "overview_traj_panel", None)
    if state.panel_id in {"video", "skeleton"} or (
        state.panel_id == "path_3d" and state.widget is overview_traj
    ):
        try:
            from stablewalk.ui.tk.dashboard_overview_view_mode import (
                apply_overview_view_mode,
            )

            apply_overview_view_mode(gui, animate=False, persist=False)
            _restore_playback(gui, was_playing=was_playing, position=playback_pos)
        except Exception:
            pass
        if state.panel_id == "path_3d":
            show = getattr(gui, "_show_overview_trajectory_dock", None)
            if show is not None:
                try:
                    show(True)
                except Exception:
                    pass

    status = getattr(gui, "status", None)
    if status is not None:
        try:
            status.configure(text="Panel restored")
        except Exception:
            pass

    _sync_maximize_buttons(gui)
    return True


def toggle_panel_maximize(gui: Any, panel_id: str) -> None:
    if current_maximized_panel_id(gui) == panel_id:
        restore_panel(gui)
    else:
        maximize_panel(gui, panel_id)


def _iter_maximize_buttons(entry: Any) -> list[Any]:
    if entry is None:
        return []
    if isinstance(entry, list):
        return list(entry)
    return [entry]


def _register_maximize_button(gui: Any, panel_id: str, btn: ttk.Button) -> None:
    buttons = getattr(gui, "_panel_maximize_buttons", None)
    if buttons is None:
        buttons = {}
        gui._panel_maximize_buttons = buttons
    existing = buttons.get(panel_id)
    if existing is None:
        buttons[panel_id] = btn
    elif isinstance(existing, list):
        existing.append(btn)
    else:
        buttons[panel_id] = [existing, btn]


def _sync_maximize_buttons(gui: Any) -> None:
    buttons: dict[str, Any] = getattr(gui, "_panel_maximize_buttons", {}) or {}
    active = current_maximized_panel_id(gui)
    for pid, entry in buttons.items():
        text = "❐" if active == pid else "⛶"
        for btn in _iter_maximize_buttons(entry):
            try:
                btn.configure(text=text)
            except tk.TclError:
                pass
    overview_btn = getattr(gui, "_overview_fullscreen_btn", None)
    if overview_btn is not None:
        try:
            if active is not None:
                overview_btn.configure(text="❐ Restore")
            else:
                overview_btn.configure(text="⛶ Full Screen")
        except tk.TclError:
            pass


def install_maximize_button(
    gui: Any,
    parent: tk.Misc,
    panel_id: str,
    *,
    pack_side: str = tk.RIGHT,
    padx: tuple[int, int] | int = (6, 0),
) -> ttk.Button:
    """Add a compact maximize control to a toolbar/chrome row."""
    if panel_id not in PANEL_TITLES:
        raise ValueError(f"Unknown panel id: {panel_id}")

    btn = ttk.Button(
        parent,
        text="⛶",
        width=2,
        style="Compact.TButton",
        command=lambda: toggle_panel_maximize(gui, panel_id),
    )
    btn.pack(side=pack_side, padx=padx)
    create_tooltip(
        btn,
        f"Maximize {PANEL_TITLES[panel_id]} (Esc to restore)",
    )
    _register_maximize_button(gui, panel_id, btn)
    return btn


def install_maximize_button_grid(
    gui: Any,
    parent: tk.Misc,
    panel_id: str,
    *,
    row: int = 0,
    column: int = 99,
    sticky: str = "ne",
    padx: tuple[int, int] | int = (4, 0),
    pady: tuple[int, int] | int = 0,
) -> ttk.Button:
    """Grid a compact maximize control into a panel header."""
    if panel_id not in PANEL_TITLES:
        raise ValueError(f"Unknown panel id: {panel_id}")

    btn = ttk.Button(
        parent,
        text="⛶",
        width=2,
        style="Compact.TButton",
        command=lambda: toggle_panel_maximize(gui, panel_id),
    )
    btn.grid(row=row, column=column, sticky=sticky, padx=padx, pady=pady)
    create_tooltip(
        btn,
        f"Maximize {PANEL_TITLES[panel_id]} (Esc to restore)",
    )
    _register_maximize_button(gui, panel_id, btn)
    return btn


def _place_corner_chrome(parent: tk.Misc) -> tk.Frame:
    """Floating top-right chrome that does not steal layout rows from the panel."""
    chrome = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    chrome.place(relx=1.0, rely=0.0, anchor="ne", x=-4, y=2)
    return chrome


def wire_all_panel_maximize_buttons(gui: Any) -> None:
    """Install maximize controls on every major panel/header that exists."""
    # Video — corner control on the Original Video panel
    video = getattr(gui, "video_frame", None)
    if video is not None and not getattr(gui, "_video_maximize_chrome", None):
        chrome = _place_corner_chrome(video)
        gui._video_maximize_chrome = chrome
        install_maximize_button(gui, chrome, "video", pack_side=tk.RIGHT, padx=0)

    # Skeleton maximize is installed next to the display-mode selector in layout.

    # 3D Path — overview dock camera bar + motion trajectory camera bar
    for bar_attr in ("overview_traj_camera_bar", "dof_traj_camera_bar"):
        bar = getattr(gui, bar_attr, None)
        if bar is None:
            continue
        if getattr(bar, "_sw_maximize_installed", False):
            continue
        install_maximize_button(gui, bar, "path_3d", pack_side=tk.RIGHT, padx=(8, 4))
        bar._sw_maximize_installed = True  # type: ignore[attr-defined]

    # Motion Analysis — grid control beside the joint-selection row
    motion_row = getattr(gui, "_motion_joint_row", None)
    if motion_row is not None and not getattr(gui, "_motion_maximize_installed", False):
        try:
            motion_row.columnconfigure(1, weight=0)
        except tk.TclError:
            pass
        install_maximize_button_grid(
            gui, motion_row, "motion", row=0, column=1, sticky="ne", padx=(8, 0)
        )
        gui._motion_maximize_installed = True

    # Biomechanics — header toolbar
    biomech_header = getattr(gui, "_biomechanics_header", None)
    if biomech_header is not None and not getattr(
        gui, "_biomech_maximize_installed", False
    ):
        install_maximize_button(
            gui, biomech_header, "biomechanics", pack_side=tk.RIGHT, padx=(8, 0)
        )
        gui._biomechanics_panel_host = getattr(
            gui, "_section_biomechanics", None
        ) or getattr(gui, "_tab_biomechanics", None)
        gui._biomech_maximize_installed = True

    # Results Summary — export button row
    summary_btns = getattr(gui, "_results_summary_btn_row", None)
    if summary_btns is not None and not getattr(
        gui, "_results_maximize_installed", False
    ):
        install_maximize_button(
            gui, summary_btns, "results_summary", pack_side=tk.LEFT, padx=(8, 0)
        )
        gui._results_summary_panel_host = getattr(
            gui, "_section_results_summary", None
        ) or getattr(gui, "_tab_results_summary", None)
        gui._results_maximize_installed = True

    # Advanced Pipeline — prefer dedicated Advanced Pipeline Status panel
    if getattr(gui, "_pipeline_panel_host", None) is None:
        gui._pipeline_panel_host = (
            getattr(gui, "_section_pipeline_status_advanced", None)
            or getattr(gui, "_section_engineering_dashboard", None)
            or getattr(gui, "_tab_advanced_content", None)
        )
    pipeline_host = getattr(gui, "_pipeline_panel_host", None)
    if (
        pipeline_host is not None
        and not getattr(gui, "_pipeline_maximize_installed", False)
    ):
        chrome = _place_corner_chrome(pipeline_host)
        gui._pipeline_maximize_chrome = chrome
        install_maximize_button(gui, chrome, "pipeline", pack_side=tk.RIGHT, padx=0)
        gui._pipeline_maximize_installed = True

    _sync_maximize_buttons(gui)


__all__ = [
    "PANEL_TITLES",
    "current_maximized_panel_id",
    "install_maximize_button",
    "install_maximize_button_grid",
    "is_panel_maximized",
    "maximize_panel",
    "restore_panel",
    "toggle_panel_maximize",
    "wire_all_panel_maximize_buttons",
]
