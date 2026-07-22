"""Overview view modes — professional visualization layouts without re-analysis."""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from typing import Any

from stablewalk import config
from stablewalk.ui.theme import (
    BORDER,
    FONT_UI_XS,
    MUTED,
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
    (VIEW_MODE_VIDEO_ONLY, "Video Only"),
    (VIEW_MODE_SKELETON_ONLY, "Skeleton Only"),
    (VIEW_MODE_SIDE_BY_SIDE, "Side-by-Side"),
    (VIEW_MODE_OVERLAY, "Overlay"),
    (VIEW_MODE_RECONSTRUCTION_FULL, "Full 3D Reconstruction"),
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
    return int(getattr(gui, "_primary_viz_content_row", 1))


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
            fit_skel()
        except Exception:
            pass
    refit = getattr(gui, "_on_video_label_resize", None)
    if refit is not None:
        try:
            refit()
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

            # tab_overview: row0 toolbar, row1 section1, row2 joint motion, row3 metrics
            metrics.grid(row=3, column=0, sticky="nsew", pady=(OVERVIEW_ROW_GAP, 0))
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
    """Show summary dock below the primary visual row (trajectory uses a column)."""
    from stablewalk.ui.theme import PAD_XS

    sidebar = getattr(gui, "sidebar", None)
    traj = getattr(gui, "overview_traj_panel", None)
    traj_visible = bool(getattr(gui, "_overview_traj_dock_visible", False))
    summary_row = int(getattr(gui, "_overview_summary_row", max(row, 1)))
    if traj_visible and traj is not None:
        # Trajectory visibility is owned by layout_panels_for_view_mode.
        _hide(sidebar)
        if not visible:
            _hide(traj)
        return
    if visible:
        _show_grid(
            sidebar,
            row=summary_row,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=0,
            pady=(PAD_XS, 0),
        )
    else:
        _hide(sidebar)


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

    host = getattr(gui, "_primary_viz_host", None)
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

    # Visual panels fill the host; the view-mode toolbar lives on tab_overview.
    content_row = 0
    gui._primary_viz_content_row = content_row
    gui._viz_tabs_active = False
    try:
        from stablewalk.ui.tk.dashboard_sections import (
            SEC1_TRAJ_VIZ_ROW_MINSIZE,
            SEC1_VIZ_ROW_MINSIZE,
        )

        traj_open = bool(getattr(gui, "_overview_traj_dock_visible", False))
        viz_minsize = SEC1_TRAJ_VIZ_ROW_MINSIZE if traj_open else SEC1_VIZ_ROW_MINSIZE
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1, minsize=viz_minsize)
    except tk.TclError:
        pass
    except Exception:
        try:
            host.columnconfigure(0, weight=1)
            host.rowconfigure(0, weight=1, minsize=520)
        except tk.TclError:
            pass

    if tab_overview is not None:
        try:
            from stablewalk.ui.tk.dashboard_sections import (
                SEC1_METRICS_ROW_WEIGHT,
                SEC1_VIZ_ROW_WEIGHT,
                apply_overview_joint_motion_row_weight,
            )

            tab_overview.rowconfigure(0, weight=0)  # view mode toolbar
            tab_overview.rowconfigure(1, weight=SEC1_VIZ_ROW_WEIGHT)  # visuals
            apply_overview_joint_motion_row_weight(
                tab_overview,
                expanded=bool(getattr(gui, "_overview_joint_motion_expanded", False)),
                row=2,
            )
            # Gait info: respect its own collapsed/expanded state.
            metrics_expanded = bool(getattr(gui, "_overview_metrics_expanded", False))
            if metrics_expanded:
                tab_overview.rowconfigure(3, weight=SEC1_METRICS_ROW_WEIGHT)
            else:
                tab_overview.rowconfigure(3, weight=0)
        except tk.TclError:
            pass

    toolbar = getattr(gui, "_overview_view_mode_bar", None)
    if toolbar is not None and tab_overview is not None:
        _show_grid(toolbar, row=0, column=0, sticky="ew", pady=(0, PAD_XS))

    # Hide legacy video/3D radioboxes — superseded by View Mode.
    legacy_tabs = getattr(gui, "_viz_tab_bar", None)
    _hide(legacy_tabs)

    from stablewalk.ui.tk.dashboard_overview_panes import layout_panels_for_view_mode

    if mode == VIEW_MODE_VIDEO_ONLY:
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, mode)
        video.configure(text=_panel_title("Original Video"))

    elif mode == VIEW_MODE_SKELETON_ONLY:
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, mode)
        skel.configure(text=_panel_title("3D Gait Reconstruction"))

    elif mode == VIEW_MODE_OVERLAY:
        _set_overlay_mode_ui(gui, enabled=True)
        layout_panels_for_view_mode(gui, mode)
        video.configure(text=_panel_title("Overlay · Video + Skeleton"))

    elif mode == VIEW_MODE_RECONSTRUCTION_FULL:
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, mode)
        skel.configure(text=_panel_title("Full 3D Reconstruction"))

    else:  # side_by_side (default professional layout)
        _set_overlay_mode_ui(gui, enabled=False)
        layout_panels_for_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
        video.configure(text=_panel_title("Original Video"))
        skel.configure(text=_panel_title("3D Gait Reconstruction"))

    _set_metrics_visible_for_mode(gui, mode)
    _restore_playback_position(gui, playback_pos)
    _fit_visuals(gui)
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
    """Create the professional Overview View Mode toolbar."""
    from tkinter import ttk

    from stablewalk.ui.theme import create_tooltip

    initial = load_overview_view_mode()
    gui._overview_view_mode = initial
    gui._overview_view_mode_var = tk.StringVar(value=initial)
    gui._overview_view_mode_active = True

    bar = tk.Frame(parent, bg=PANEL, highlightthickness=0)
    bar.columnconfigure(1, weight=1)
    gui._overview_view_mode_bar = bar

    tk.Label(
        bar,
        text="View Mode",
        bg=PANEL,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).grid(row=0, column=0, sticky="w", padx=(0, PAD_SM))

    modes = tk.Frame(bar, bg=PANEL, highlightthickness=0)
    modes.grid(row=0, column=1, sticky="w")

    def _on_mode_selected() -> None:
        apply_overview_view_mode(
            gui,
            gui._overview_view_mode_var.get(),
            animate=True,
            persist=True,
        )

    for mode_id, label in VIEW_MODE_OPTIONS:
        ttk.Radiobutton(
            modes,
            text=label,
            variable=gui._overview_view_mode_var,
            value=mode_id,
            command=_on_mode_selected,
        ).pack(side=tk.LEFT, padx=(0, 8))

    actions = tk.Frame(bar, bg=PANEL, highlightthickness=0)
    actions.grid(row=0, column=2, sticky="e")

    fullscreen_btn = ttk.Button(
        actions,
        text="⛶ Full Screen",
        style="Compact.TButton",
        command=lambda: toggle_overview_full_screen(gui),
    )
    fullscreen_btn.pack(side=tk.LEFT, padx=(0, PAD_SM))
    gui._overview_fullscreen_btn = fullscreen_btn
    create_tooltip(fullscreen_btn, "Maximize the active visualization (Esc to restore)")

    def _on_reset_layout() -> None:
        var = gui._overview_view_mode_var
        if str(var.get()) != VIEW_MODE_SIDE_BY_SIDE:
            var.set(VIEW_MODE_SIDE_BY_SIDE)
        apply_overview_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE, animate=False, persist=True)

    reset_btn = ttk.Button(
        actions,
        text="Reset",
        style="Compact.TButton",
        command=_on_reset_layout,
    )
    reset_btn.pack(side=tk.LEFT)
    create_tooltip(reset_btn, "Return to the Side-by-Side workspace")

    tk.Frame(bar, bg=BORDER, height=1, highlightthickness=0).grid(
        row=1, column=0, columnspan=3, sticky="ew", pady=(PAD_XS, 0)
    )
    return bar


__all__ = [
    "DEFAULT_VIEW_MODE",
    "VIEW_MODE_IDS",
    "VIEW_MODE_OPTIONS",
    "VIEW_MODE_OVERLAY",
    "VIEW_MODE_RECONSTRUCTION_FULL",
    "VIEW_MODE_SIDE_BY_SIDE",
    "VIEW_MODE_SKELETON_ONLY",
    "VIEW_MODE_VIDEO_ONLY",
    "apply_overview_view_mode",
    "build_overview_view_mode_selector",
    "current_overview_view_mode",
    "load_overview_view_mode",
    "preferences_path",
    "save_overview_view_mode",
    "toggle_overview_full_screen",
]
