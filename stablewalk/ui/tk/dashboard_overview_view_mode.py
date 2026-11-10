"""Overview view modes — professional visualization layouts without re-analysis."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from typing import Any

from stablewalk import config
from stablewalk.ui.theme import (
    INLINE_GAP,
    PAD_SM,
    PAD_XS,
    PANEL,
)

# Stable preference keys / mode ids
VIEW_MODE_VIDEO_ONLY = "video_only"
VIEW_MODE_SKELETON_ONLY = "skeleton_only"
VIEW_MODE_SIDE_BY_SIDE = "side_by_side"
VIEW_MODE_OVERLAY = "overlay"
VIEW_MODE_RECONSTRUCTION_FULL = "reconstruction_full"

VIEW_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    (VIEW_MODE_VIDEO_ONLY, "Video"),
    (VIEW_MODE_OVERLAY, "Overlay"),
    (VIEW_MODE_SKELETON_ONLY, "Skeleton"),
    (VIEW_MODE_SIDE_BY_SIDE, "Split"),
    (VIEW_MODE_RECONSTRUCTION_FULL, "Full 3D Reconstruction"),
)

# Primary Overview toggle. Split is the presentation default so Video,
# Skeleton, and 3D Path stay on screen together.
VIEW_MODE_TOGGLE_OPTIONS: tuple[tuple[str, str], ...] = (
    (VIEW_MODE_SIDE_BY_SIDE, "Split"),
    (VIEW_MODE_OVERLAY, "Overlay"),
    (VIEW_MODE_VIDEO_ONLY, "Video"),
    (VIEW_MODE_SKELETON_ONLY, "Skeleton"),
    (VIEW_MODE_RECONSTRUCTION_FULL, "3D Full"),
)

VIEW_MODE_IDS = frozenset(mode_id for mode_id, _label in VIEW_MODE_OPTIONS)
DEFAULT_VIEW_MODE = VIEW_MODE_SIDE_BY_SIDE

_PREFS_FILENAME = "ui_preferences.json"
_PREF_KEY = "overview_view_mode"


def preferences_path() -> Path:
    return config.OUTPUT_DIR / _PREFS_FILENAME


def load_overview_view_mode() -> str:
    path = preferences_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            mode = str(data.get(_PREF_KEY, DEFAULT_VIEW_MODE))
            if mode in VIEW_MODE_IDS:
                return mode
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return DEFAULT_VIEW_MODE


def save_overview_view_mode(mode: str) -> None:
    if mode not in VIEW_MODE_IDS:
        return
    path = preferences_path()
    data: dict[str, Any] = {}
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        data = {}
    data[_PREF_KEY] = mode
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def current_overview_view_mode(gui: Any) -> str:
    var = getattr(gui, "_overview_view_mode_var", None)
    if var is not None:
        mode = str(var.get() or DEFAULT_VIEW_MODE)
        if mode in VIEW_MODE_IDS:
            return mode
    stored = getattr(gui, "_overview_view_mode", None)
    if stored in VIEW_MODE_IDS:
        return str(stored)
    return DEFAULT_VIEW_MODE


def _content_row(gui: Any) -> int:
    return int(
        getattr(
            gui,
            "_primary_viz_content_row",
            getattr(gui, "_primary_viz_content_row", 0),
        )
        or 0
    )


def _hide(widget: tk.Misc | None) -> None:
    if widget is None:
        return
    try:
        widget.grid_remove()
    except tk.TclError:
        pass


def _show_grid(widget: tk.Misc | None, **kwargs: Any) -> None:
    if widget is None:
        return
    try:
        widget.grid(**kwargs)
    except tk.TclError:
        pass


def _capture_playback_position(gui: Any) -> float | None:
    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        try:
            return float(player.state.frame_float)
        except Exception:
            return None
    return float(getattr(gui, "_playback_pos", 0.0) or 0.0)


def _restore_playback_position(gui: Any, position: float | None) -> None:
    """Keep the scrubber / player on the same frame after a layout swap.

    Must not pause playback: ``frame_var.set`` would otherwise fire the slider
    command (``_on_slider``), which stops the play timer.
    """
    if position is None:
        return
    player = getattr(gui, "skeleton_player", None)
    if player is not None and getattr(player, "frame_count", 0) > 0:
        # Guard slider callbacks so layout restores never pause play.
        prev_guard = bool(getattr(gui, "_sync_all_guard", False))
        gui._sync_all_guard = True
        try:
            player.go_to(float(position))
            gui._playback_pos = float(player.state.frame_float)
            gui.current_pos = int(player.state.frame_index)
            if hasattr(gui, "frame_var"):
                target = int(round(gui._playback_pos))
                try:
                    current = int(round(float(gui.frame_var.get())))
                except Exception:
                    current = -1
                if current != target:
                    gui.frame_var.set(target)
        except Exception:
            pass
        finally:
            gui._sync_all_guard = prev_guard
    sync = getattr(gui, "_sync_transport_labels", None)
    if sync is not None:
        sync()


def _set_overlay_skeleton_force(gui: Any, *, enabled: bool) -> None:
    """Overlay mode forces the global show_skeleton flag for compatible redraws."""
    show_var = getattr(gui, "show_skeleton", None)
    if show_var is None:
        return
    if enabled:
        if not hasattr(gui, "_overview_overlay_prev_show_skeleton"):
            try:
                gui._overview_overlay_prev_show_skeleton = bool(show_var.get())
            except tk.TclError:
                gui._overview_overlay_prev_show_skeleton = True
        try:
            show_var.set(True)
        except tk.TclError:
            pass
        return

    prev = getattr(gui, "_overview_overlay_prev_show_skeleton", None)
    if prev is not None:
        try:
            show_var.set(bool(prev))
        except tk.TclError:
            pass
        try:
            delattr(gui, "_overview_overlay_prev_show_skeleton")
        except Exception:
            pass


def _set_overlay_mode_ui(gui: Any, *, enabled: bool) -> None:
    """Show Overlay Mode controls and keep skeleton drawing on the video."""
    from stablewalk.ui.tk.overlay_mode_controls import (
        ensure_overlay_mode_vars,
        set_overlay_mode_controls_visible,
    )

    ensure_overlay_mode_vars(gui)
    set_overlay_mode_controls_visible(gui, enabled)
    _set_overlay_skeleton_force(gui, enabled=enabled)

def _configure_host_columns(
    host: tk.Misc,
    *,
    left: int,
    mid: int,
    right: int,
) -> None:
    try:
        host.columnconfigure(0, weight=left, uniform="sec1" if left else None, minsize=0)
        host.columnconfigure(1, weight=mid, uniform="sec1" if mid else None, minsize=0)
        host.columnconfigure(2, weight=right, uniform="sec1" if right else None, minsize=0)
    except tk.TclError:
        try:
            host.columnconfigure(0, weight=left, minsize=0)
            host.columnconfigure(1, weight=mid, minsize=0)
            host.columnconfigure(2, weight=right, minsize=0)
        except tk.TclError:
            pass


def _fit_visuals(gui: Any) -> None:
    fit_skel = getattr(gui, "_fit_skeleton_canvas", None)
    if fit_skel is not None:
        try:
            fit_skel(force=True)
        except TypeError:
            try:
                fit_skel()
            except Exception:
                pass
        except Exception:
            pass
    refit = getattr(gui, "_on_video_label_resize", None)
    if refit is not None:
        try:
            refit()
        except Exception:
            pass
    traj = getattr(gui, "_render_overview_traj_canvas", None)
    if traj is not None:
        try:
            traj(force=True)
        except TypeError:
            try:
                traj()
            except Exception:
                pass
        except Exception:
            pass


def _refresh_synced_overlays(gui: Any) -> None:
    """Redraw video + skeleton at the preserved playback frame (no analysis)."""
    force_draw = True
    show_pose = getattr(gui, "_show_pose_at", None)
    update_skel = getattr(gui, "_update_interactive_skeleton", None)
    pos = getattr(gui, "_playback_pos", None)
    if pos is None and getattr(gui, "skeleton_player", None) is not None:
        pos = float(gui.skeleton_player.state.frame_float)

    if show_pose is not None and getattr(gui, "sequence", None) is not None:
        try:
            show_pose(pos if pos is not None else 0, force_draw=force_draw)
            return
        except Exception:
            pass
    if update_skel is not None:
        try:
            update_skel(force_draw=force_draw)
        except Exception:
            pass


def _set_metrics_visible(gui: Any, visible: bool) -> None:
    metrics = getattr(gui, "_overview_metrics_row", None)
    if metrics is None:
        return
    try:
        if visible:
            from stablewalk.ui.theme import OVERVIEW_ROW_GAP

            # tab_overview: header, view toggle, viz, clinical, live strip, graphs, metrics
            metrics.grid(row=6, column=0, sticky="nsew", pady=(OVERVIEW_ROW_GAP, 0))
        else:
            metrics.grid_remove()
    except tk.TclError:
        pass


def _set_metrics_visible_for_mode(gui: Any, mode: str) -> None:
    _set_metrics_visible(gui, mode != VIEW_MODE_RECONSTRUCTION_FULL)


def _set_sidebar_visible(
    gui: Any,
    visible: bool,
    *,
    row: int,
    padx: tuple[int, int],
) -> None:
    """Legacy summary dock stays hidden — clinical summary strip owns the KPIs."""
    sidebar = getattr(gui, "sidebar", None)
    _hide(sidebar)
    # Trajectory stays in the viz row; view-mode must not unmap it.


def _panel_title(label: str) -> str:
    return f"  {label}  "


def apply_overview_view_mode(
    gui: Any,
    mode: str | None = None,
    *,
    animate: bool = False,
    persist: bool = False,
) -> None:
    """Rearrange Overview panels for the selected professional view mode."""
    mode = mode or current_overview_view_mode(gui)
    if mode not in VIEW_MODE_IDS:
        mode = DEFAULT_VIEW_MODE

    host = (
        getattr(gui, "_primary_viz_host", None)
        or getattr(gui, "_section_visual", None)
        or getattr(gui, "_primary_viz_host", None)
    )
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    tab_overview = getattr(gui, "_tab_overview", None) or getattr(gui, "_dashboard_body", None)
    if host is None or video is None or skel is None:
        return

    playback_pos = _capture_playback_position(gui)
    gui._overview_view_mode = mode
    var = getattr(gui, "_overview_view_mode_var", None)
    if var is not None and str(var.get()) != mode:
        try:
            var.set(mode)
        except tk.TclError:
            pass

    from stablewalk.ui.tk.dashboard_panel_maximize import is_panel_maximized

    if is_panel_maximized(gui):
        # Geometry is owned by the maximize overlay — only sync metadata / refresh.
        _restore_playback_position(gui, playback_pos)
        _refresh_synced_overlays(gui)
        if persist:
            save_overview_view_mode(mode)
        return

    playing = bool(getattr(gui, "playing", False))
    from stablewalk.ui.tk.dashboard_overview_panes import overview_viz_panels_mounted

    skip_geometry = playing and overview_viz_panels_mounted(gui)

    # Visual panels fill the host; the view-mode toolbar lives on the page header.
    content_row = 0
    gui._primary_viz_content_row = content_row
    gui._viz_tabs_active = False
    viz_host = host
    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_HOST_VIZ_ROW,
        apply_overview_host_row_weights,
        apply_overview_joint_motion_row_weight,
        apply_overview_viz_grid_weights,
    )

    if not skip_geometry:
        apply_overview_viz_grid_weights(viz_host, content_row=content_row)

        if tab_overview is not None:
            try:
                host = (
                    getattr(gui, "_overview_layout_host", None)
                    or getattr(gui, "_overview_content", None)
                    or tab_overview
                )
                apply_overview_host_row_weights(host, viz_row=OVERVIEW_HOST_VIZ_ROW)
                apply_overview_joint_motion_row_weight(
                    host,
                    expanded=bool(getattr(gui, "_overview_joint_motion_expanded", False)),
                    row=5,
                )
            except tk.TclError:
                pass

    # Hide legacy video/3D radioboxes — superseded by View Mode.
    legacy_tabs = getattr(gui, "_viz_tab_bar", None)
    _hide(legacy_tabs)

    from stablewalk.ui.tk.dashboard_overview_panes import layout_panels_for_view_mode
    from stablewalk.ui.tk.overview_responsive import WIDTH_VIZ_THREE_COL

    try:
        win_w = int(gui.root.winfo_width())
    except Exception:
        win_w = WIDTH_VIZ_THREE_COL
    gui._overview_viz_stack_requested = False

    if mode == VIEW_MODE_VIDEO_ONLY:
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, mode)
        video.configure(text=_panel_title("Video"))

    elif mode == VIEW_MODE_SKELETON_ONLY:
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, mode)
        video.configure(text=_panel_title("Video"))
        skel.configure(text=_panel_title("Skeleton"))

    elif mode == VIEW_MODE_OVERLAY:
        _set_overlay_mode_ui(gui, enabled=True)
        layout_panels_for_view_mode(gui, mode)
        video.configure(text=_panel_title("Overlay"))
        # Keep skeleton drawing on by default for Overlay (landmarks + bones).
        skel_var = getattr(gui, "var_video_overlay_skeleton", None)
        if skel_var is not None:
            try:
                skel_var.set(True)
            except tk.TclError:
                pass

    elif mode == VIEW_MODE_RECONSTRUCTION_FULL:
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, mode)
        skel.configure(text=_panel_title("Skeleton"))

    else:  # side_by_side split layout
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
        video.configure(text=_panel_title("Video"))
        skel.configure(text=_panel_title("Skeleton"))

    _set_metrics_visible_for_mode(gui, mode)
    _restore_playback_position(gui, playback_pos)
    playing = bool(getattr(gui, "playing", False))
    if not playing:
        _fit_visuals(gui)
        _refresh_synced_overlays(gui)
    else:
        # Do not refit canvases mid-play — it stalls the clock. Paint live surfaces only.
        lite = getattr(gui, "_sync_overview_playback_lite", None)
        if lite is not None:
            try:
                lite(force_draw=False)
            except Exception:
                _refresh_synced_overlays(gui)
        else:
            _refresh_synced_overlays(gui)

    if persist:
        save_overview_view_mode(mode)

    if animate:
        _animate_mode_transition(gui, host)

    status = getattr(gui, "status", None)
    if status is not None:
        label = next((lbl for mid, lbl in VIEW_MODE_OPTIONS if mid == mode), mode)
        try:
            status.configure(text=f"View mode: {label}")
        except Exception:
            pass


def _primary_panel_id_for_mode(_gui: Any, mode: str) -> str:
    """Which visualization panel should Overview Full Screen maximize."""
    if mode in (VIEW_MODE_VIDEO_ONLY, VIEW_MODE_OVERLAY):
        return "video"
    if mode == VIEW_MODE_RECONSTRUCTION_FULL:
        return "path_3d"
    return "skeleton"


def toggle_overview_full_screen(gui: Any) -> None:
    """Maximize the active visualization (or restore if already maximized)."""
    from stablewalk.ui.tk.dashboard_panel_maximize import (
        current_maximized_panel_id,
        maximize_panel,
        restore_panel,
    )

    if current_maximized_panel_id(gui) is not None:
        restore_panel(gui)
        return
    panel_id = _primary_panel_id_for_mode(gui, current_overview_view_mode(gui))
    maximize_panel(gui, panel_id)


def _animate_mode_transition(gui: Any, host: tk.Misc) -> None:
    from stablewalk.ui.tk.ui_transitions import fade_in_frame

    root = getattr(gui, "root", None)
    if root is None:
        return
    try:
        fade_in_frame(host, root=root, step_ms=28)
    except Exception:
        pass


def build_overview_view_mode_selector(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Compact Overlay / Video / Skeleton / Split toggle plus a View menu."""
    from tkinter import ttk

    from stablewalk.ui.tk.dashboard_panel_maximize import current_maximized_panel_id
    from stablewalk.ui.theme import (
        ACCENT,
        ACCENT_FG,
        FONT_BUTTON_SM,
        TEXT,
        create_tooltip,
    )

    initial = load_overview_view_mode()
    if initial not in VIEW_MODE_IDS:
        initial = DEFAULT_VIEW_MODE
    gui._overview_view_mode = initial
    gui._overview_view_mode_var = tk.StringVar(value=initial)
    gui._overview_view_mode_active = True

    bar = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    bar.columnconfigure(1, weight=1)
    gui._overview_view_mode_bar = bar

    segment = tk.Frame(bar, bg=PANEL, highlightthickness=0)
    segment.grid(row=0, column=0, sticky="w")
    gui._overview_view_mode_segment = segment

    buttons: dict[str, tk.Label] = {}

    def _style_buttons(active_mode: str) -> None:
        for mode_id, btn in buttons.items():
            selected = mode_id == active_mode
            try:
                btn.configure(
                    bg=ACCENT if selected else PANEL,
                    fg=ACCENT_FG if selected else TEXT,
                )
            except tk.TclError:
                pass

    def _select_mode(mode_id: str, *, persist: bool = True, animate: bool = True) -> None:
        if mode_id not in VIEW_MODE_IDS:
            return
        gui._overview_view_mode_var.set(mode_id)
        _style_buttons(mode_id)
        apply_overview_view_mode(
            gui,
            mode_id,
            animate=animate and not bool(getattr(gui, "playing", False)),
            persist=persist,
        )

    for col, (mode_id, label) in enumerate(VIEW_MODE_TOGGLE_OPTIONS):
        btn = tk.Label(
            segment,
            text=label,
            bg=PANEL,
            fg=TEXT,
            font=FONT_BUTTON_SM,
            cursor="hand2",
            padx=10,
            pady=4,
        )
        btn.grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=(0 if col == 0 else INLINE_GAP, 0),
        )
        btn.bind("<Button-1>", lambda _e, mid=mode_id: _select_mode(mid))
        create_tooltip(
            btn,
            {
                VIEW_MODE_VIDEO_ONLY: "Video only — current frame, aspect preserved.",
                VIEW_MODE_OVERLAY: (
                    "Landmarks and bones drawn on the video. Playback stays on this frame."
                ),
                VIEW_MODE_SKELETON_ONLY: (
                    "Full-height 3D skeleton, fitted and centered."
                ),
                VIEW_MODE_SIDE_BY_SIDE: (
                    "Video 40% | Skeleton 30% | 3D Path 30%, resizable. "
                    "At narrow widths the 3D plot moves below."
                ),
                VIEW_MODE_RECONSTRUCTION_FULL: (
                    "3D Full — joint path and Coordinate Inspector fill the Overview."
                ),
            }.get(mode_id, label),
            wraplength=320,
        )
        buttons[mode_id] = btn
    gui._overview_view_mode_buttons = buttons

    # Keep segment highlight in sync when mode is applied elsewhere.
    def _sync_from_var(*_args: object) -> None:
        mode_id = str(gui._overview_view_mode_var.get() or "")
        _style_buttons(mode_id if mode_id in buttons else "")

    try:
        gui._overview_view_mode_var.trace_add("write", _sync_from_var)
    except Exception:
        pass
    _style_buttons(initial if initial in buttons else "")

    actions = tk.Frame(bar, bg=PANEL, highlightthickness=0)
    actions.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))

    def _toggle_graphs() -> None:
        getattr(gui, "_toggle_overview_joint_motion", lambda: None)()

    def _toggle_signal_spectral() -> None:
        getattr(gui, "_toggle_overview_signal_spectral", lambda: None)()

    def _on_reset_layout() -> None:
        from stablewalk.ui.tk.dashboard_overview_panes import reset_overview_layout

        _select_mode(VIEW_MODE_SIDE_BY_SIDE, persist=True, animate=False)
        reset_overview_layout(gui)

    view_mb = ttk.Menubutton(
        actions,
        text="View",
        takefocus=False,
    )
    view_menu = tk.Menu(view_mb, tearoff=0)
    view_mb["menu"] = view_menu
    view_mb.pack(side=tk.LEFT, padx=(PAD_XS, 0))
    create_tooltip(view_mb, "Graphs, full screen, and 3D reconstruction")
    gui._overview_view_menu_btn = view_mb

    def _refresh_view_menu() -> None:
        try:
            view_menu.delete(0, "end")
        except tk.TclError:
            return
        graphs_expanded = bool(getattr(gui, "_overview_joint_motion_expanded", False))
        spectral_expanded = bool(
            getattr(gui, "_overview_signal_spectral_expanded", False)
        )
        view_menu.add_command(
            label="Hide graphs" if graphs_expanded else "Graphs",
            command=_toggle_graphs,
        )
        view_menu.add_command(
            label=(
                "Hide Signal & Spectral"
                if spectral_expanded
                else "Signal & Spectral"
            ),
            command=_toggle_signal_spectral,
        )
        try:
            maximized = current_maximized_panel_id(gui) is not None
        except Exception:
            maximized = False
        view_menu.add_command(
            label="Restore" if maximized else "Full screen",
            command=lambda: toggle_overview_full_screen(gui),
        )
        view_menu.add_separator()
        view_menu.add_command(
            label="Reset Layout",
            command=_on_reset_layout,
        )
        view_menu.add_command(
            label="Full 3D Reconstruction",
            command=lambda: _select_mode(VIEW_MODE_RECONSTRUCTION_FULL),
        )

    view_menu.configure(postcommand=_refresh_view_menu)
    _refresh_view_menu()

    # Compatibility widgets (not packed). Callers still update labels/commands.
    graphs_btn = ttk.Button(
        actions,
        text="Graphs",
        style="Compact.TButton",
        width=7,
        command=_toggle_graphs,
        takefocus=False,
    )
    gui._overview_graphs_btn = graphs_btn

    fullscreen_btn = ttk.Button(
        actions,
        text="⛶",
        style="Compact.TButton",
        width=3,
        command=lambda: toggle_overview_full_screen(gui),
    )
    gui._overview_fullscreen_btn = fullscreen_btn

    reset_btn = ttk.Button(
        actions,
        text="Reset Layout",
        style="Compact.TButton",
        command=_on_reset_layout,
        takefocus=False,
    )
    reset_btn.pack(side=tk.LEFT, padx=(PAD_XS, 0))
    create_tooltip(
        reset_btn,
        "Restore Video 40% | Skeleton 30% | 3D 30%. "
        "Does not reload the video or rerun analysis.",
        wraplength=320,
    )
    gui._overview_split_btn = reset_btn
    gui._overview_reset_layout_btn = reset_btn

    # Legacy combo attrs kept for session restore / tests that may set them.
    gui._overview_view_mode_combo_var = tk.StringVar(
        value=next((lbl for mid, lbl in VIEW_MODE_OPTIONS if mid == initial), "Split")
    )
    gui._overview_view_mode_combo = None

    return bar


__all__ = [
    "DEFAULT_VIEW_MODE",
    "VIEW_MODE_IDS",
    "VIEW_MODE_OPTIONS",
    "VIEW_MODE_OVERLAY",
    "VIEW_MODE_RECONSTRUCTION_FULL",
    "VIEW_MODE_SIDE_BY_SIDE",
    "VIEW_MODE_SKELETON_ONLY",
    "VIEW_MODE_TOGGLE_OPTIONS",
    "VIEW_MODE_VIDEO_ONLY",
    "apply_overview_view_mode",
    "build_overview_view_mode_selector",
    "current_overview_view_mode",
    "load_overview_view_mode",
    "preferences_path",
    "save_overview_view_mode",
    "toggle_overview_full_screen",
]
