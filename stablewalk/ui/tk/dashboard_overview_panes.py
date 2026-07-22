"""Overview horizontal paned layout - draggable sashes with persisted fractions."""

from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from typing import Any, Sequence

from stablewalk import config
from stablewalk.ui.theme import (
    PANEL,
    SIDEBAR_MIN_WIDTH,
)

# Default right-edge fractions for sash 0 and sash 1 (side-by-side).
DEFAULT_SASH_FRACTIONS: tuple[float, float] = (0.33, 0.67)
DEFAULT_TRAJ_SASH_FRACTIONS: tuple[float, float] = (0.30, 0.70)

MIN_PANE_VIDEO = 200
MIN_PANE_SKELETON = 200
MIN_PANE_SIDE = max(148, int(SIDEBAR_MIN_WIDTH))

_PREF_KEY = "overview_sash_fractions"
_PREF_KEY_TRAJ = "overview_sash_fractions_traj"
_PREFS_FILENAME = "ui_preferences.json"

PANE_VIDEO = "video"
PANE_SKELETON = "skeleton"
PANE_SIDE = "side"


def preferences_path():
    return config.OUTPUT_DIR / _PREFS_FILENAME


def _clamp_fractions(fractions: Sequence[float]) -> tuple[float, float]:
    if len(fractions) < 2:
        return DEFAULT_SASH_FRACTIONS
    f0 = float(fractions[0])
    f1 = float(fractions[1])
    f0 = max(0.12, min(0.70, f0))
    f1 = max(f0 + 0.12, min(0.88, f1))
    return (f0, f1)


def load_sash_fractions(*, traj: bool = False) -> tuple[float, float]:
    key = _PREF_KEY_TRAJ if traj else _PREF_KEY
    default = DEFAULT_TRAJ_SASH_FRACTIONS if traj else DEFAULT_SASH_FRACTIONS
    path = preferences_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            raw = data.get(key)
            if isinstance(raw, (list, tuple)) and len(raw) >= 2:
                return _clamp_fractions((float(raw[0]), float(raw[1])))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return default


def save_sash_fractions(
    fractions: Sequence[float],
    *,
    traj: bool = False,
) -> None:
    key = _PREF_KEY_TRAJ if traj else _PREF_KEY
    path = preferences_path()
    data: dict[str, Any] = {}
    try:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        data = {}
    data[key] = list(_clamp_fractions(fractions))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def clear_sash_fraction_prefs() -> None:
    """Remove saved sash fractions so defaults are used next."""
    path = preferences_path()
    data: dict[str, Any] = {}
    try:
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return
    data.pop(_PREF_KEY, None)
    data.pop(_PREF_KEY_TRAJ, None)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


def install_overview_paned_host(gui: Any, section1: tk.Misc) -> ttk.Panedwindow:
    """
    Install a horizontal Panedwindow filling ``section1``.

    Returns the paned widget. Creates empty pane frames stored on ``gui``.
    """
    section1.columnconfigure(0, weight=1)
    section1.rowconfigure(0, weight=1)

    paned = ttk.Panedwindow(section1, orient=tk.HORIZONTAL)
    paned.grid(row=0, column=0, sticky="nsew")
    gui._overview_paned = paned

    pane_video = ttk.Frame(paned, style="Card.TFrame")
    pane_skel = ttk.Frame(paned, style="Card.TFrame")
    pane_side = ttk.Frame(paned, style="Card.TFrame")
    for pane in (pane_video, pane_skel, pane_side):
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(0, weight=1)

    gui._overview_pane_video = pane_video
    gui._overview_pane_skeleton = pane_skel
    gui._overview_pane_side = pane_side
    gui._overview_paned_visible = (PANE_VIDEO, PANE_SKELETON, PANE_SIDE)
    gui._overview_sash_save_after = None
    gui._overview_sash_sync_after = None

    # Seed three panes (side-by-side default); view-mode may forget later.
    paned.add(pane_video, weight=33)
    paned.add(pane_skel, weight=34)
    paned.add(pane_side, weight=33)

    paned.bind("<ButtonRelease-1>", lambda _e: _on_sash_released(gui), add="+")
    paned.bind("<B1-Motion>", lambda _e: _on_sash_motion(gui), add="+")
    paned.bind("<Configure>", lambda e: _on_paned_configure(gui, e), add="+")

    return paned


def mount_panel_in_pane(pane: tk.Misc, panel: tk.Misc) -> None:
    """Grid a panel so it fills its paned slot (panel must be a child of pane)."""
    try:
        panel.grid(row=0, column=0, sticky="nsew")
    except tk.TclError:
        return
    try:
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(0, weight=1)
    except tk.TclError:
        pass


def _pane_map(gui: Any) -> dict[str, tk.Misc]:
    return {
        PANE_VIDEO: getattr(gui, "_overview_pane_video", None),
        PANE_SKELETON: getattr(gui, "_overview_pane_skeleton", None),
        PANE_SIDE: getattr(gui, "_overview_pane_side", None),
    }


def _sync_visuals(gui: Any) -> None:
    sync_clips = getattr(gui, "_sync_overview_media_clips", None)
    if sync_clips is not None:
        try:
            sync_clips()
        except Exception:
            pass
    from stablewalk.ui.tk.clip_viewport import sync_clipped_viewport

    sync_clipped_viewport(
        getattr(gui, "_video_clip_canvas", None),
        getattr(gui, "_video_clip_window_id", None),
    )
    sync_clipped_viewport(
        getattr(gui, "_skel_clip_canvas", None),
        getattr(gui, "_skel_clip_window_id", None),
    )
    fit = getattr(gui, "_fit_skeleton_canvas", None)
    if fit is not None:
        try:
            fit()
        except Exception:
            pass
    refit = getattr(gui, "_on_video_label_resize", None)
    if refit is not None:
        try:
            refit()
        except Exception:
            pass
    traj = getattr(gui, "_render_overview_traj_canvas", None)
    if traj is not None and getattr(gui, "_overview_traj_dock_visible", False):
        try:
            traj(force=True)
        except TypeError:
            try:
                traj()
            except Exception:
                pass
        except Exception:
            pass


def _schedule_sync_visuals(gui: Any, *, delay_ms: int = 40) -> None:
    root = getattr(gui, "root", None)
    if root is None:
        _sync_visuals(gui)
        return
    prev = getattr(gui, "_overview_sash_sync_after", None)
    if prev is not None:
        try:
            root.after_cancel(prev)
        except (tk.TclError, ValueError):
            pass

    def _run() -> None:
        gui._overview_sash_sync_after = None
        _sync_visuals(gui)

    try:
        gui._overview_sash_sync_after = root.after(delay_ms, _run)
    except tk.TclError:
        _sync_visuals(gui)


def _on_sash_motion(gui: Any) -> None:
    _schedule_sync_visuals(gui, delay_ms=30)


def _on_sash_released(gui: Any) -> None:
    try:
        from stablewalk.ui.tk.dashboard_panel_maximize import is_panel_maximized

        if is_panel_maximized(gui):
            return
    except Exception:
        pass
    _enforce_min_sizes(gui)
    _schedule_sync_visuals(gui, delay_ms=20)
    _schedule_save_fractions(gui)


def _on_paned_configure(gui: Any, event: tk.Event) -> None:
    if event.widget is not getattr(gui, "_overview_paned", None):
        return
    try:
        from stablewalk.ui.tk.dashboard_panel_maximize import is_panel_maximized

        if is_panel_maximized(gui):
            return
    except Exception:
        pass
    width = int(getattr(event, "width", 0) or 0)
    if width < 80:
        return
    # Restore fractions after host size changes (window resize).
    if getattr(gui, "_overview_sash_applying", False):
        return
    root = getattr(gui, "root", None)
    if root is None:
        return
    prev = getattr(gui, "_overview_sash_configure_after", None)
    if prev is not None:
        try:
            root.after_cancel(prev)
        except (tk.TclError, ValueError):
            pass

    def _run() -> None:
        gui._overview_sash_configure_after = None
        apply_saved_sash_fractions(gui)
        _schedule_sync_visuals(gui)

    try:
        gui._overview_sash_configure_after = root.after(80, _run)
    except tk.TclError:
        apply_saved_sash_fractions(gui)


def _pane_identity(gui: Any, pane_widget: str | tk.Misc) -> str | None:
    mapping = _pane_map(gui)
    for key, widget in mapping.items():
        if widget is None:
            continue
        try:
            if str(pane_widget) == str(widget) or pane_widget is widget:
                return key
        except tk.TclError:
            continue
    return None


def set_overview_panes_visible(gui: Any, visible: Sequence[str]) -> None:
    """Show only the listed panes (video / skeleton / side) in order."""
    paned: ttk.Panedwindow | None = getattr(gui, "_overview_paned", None)
    if paned is None:
        return
    wanted = tuple(v for v in visible if v in (PANE_VIDEO, PANE_SKELETON, PANE_SIDE))
    if not wanted:
        wanted = (PANE_VIDEO, PANE_SKELETON, PANE_SIDE)

    mapping = _pane_map(gui)
    current = list(paned.panes())
    current_ids = []
    for path in current:
        ident = _pane_identity(gui, path)
        if ident is not None:
            current_ids.append(ident)

    if tuple(current_ids) == wanted:
        gui._overview_paned_visible = wanted
        return

    # Forget all, then re-add in order.
    for path in list(paned.panes()):
        try:
            paned.forget(path)
        except tk.TclError:
            pass

    weights = {PANE_VIDEO: 33, PANE_SKELETON: 34, PANE_SIDE: 33}
    for ident in wanted:
        pane = mapping.get(ident)
        if pane is None:
            continue
        try:
            paned.add(pane, weight=weights.get(ident, 1))
        except tk.TclError:
            try:
                paned.add(pane)
            except tk.TclError:
                pass

    gui._overview_paned_visible = wanted
    apply_saved_sash_fractions(gui)


def _read_fractions_from_paned(gui: Any) -> tuple[float, float] | None:
    paned: ttk.Panedwindow | None = getattr(gui, "_overview_paned", None)
    if paned is None:
        return None
    try:
        width = max(1, int(paned.winfo_width()))
        panes = list(paned.panes())
        if len(panes) < 2:
            return None
        # sash_coord returns pixel x of sash i
        f0 = float(paned.sashpos(0)) / width if len(panes) >= 2 else DEFAULT_SASH_FRACTIONS[0]
        if len(panes) >= 3:
            f1 = float(paned.sashpos(1)) / width
        else:
            f1 = 0.99
        return _clamp_fractions((f0, f1))
    except tk.TclError:
        return None


def _schedule_save_fractions(gui: Any) -> None:
    root = getattr(gui, "root", None)
    if root is None:
        _save_current_fractions(gui)
        return
    prev = getattr(gui, "_overview_sash_save_after", None)
    if prev is not None:
        try:
            root.after_cancel(prev)
        except (tk.TclError, ValueError):
            pass

    def _run() -> None:
        gui._overview_sash_save_after = None
        _save_current_fractions(gui)

    try:
        gui._overview_sash_save_after = root.after(120, _run)
    except tk.TclError:
        _save_current_fractions(gui)


def _save_current_fractions(gui: Any) -> None:
    visible = getattr(gui, "_overview_paned_visible", ())
    if visible != (PANE_VIDEO, PANE_SKELETON, PANE_SIDE):
        return
    fr = _read_fractions_from_paned(gui)
    if fr is None:
        return
    traj = bool(getattr(gui, "_overview_traj_dock_visible", False))
    save_sash_fractions(fr, traj=traj)


def apply_saved_sash_fractions(gui: Any, *, force_defaults: bool = False) -> None:
    """Place sashes from prefs (or defaults). No-op unless all 3 panes are visible."""
    paned: ttk.Panedwindow | None = getattr(gui, "_overview_paned", None)
    if paned is None:
        return
    visible = getattr(gui, "_overview_paned_visible", ())
    if visible != (PANE_VIDEO, PANE_SKELETON, PANE_SIDE):
        return
    try:
        panes = list(paned.panes())
        if len(panes) < 3:
            return
        paned.update_idletasks()
        width = int(paned.winfo_width())
        if width < 120:
            # Host not mapped yet — retry shortly.
            root = getattr(gui, "root", None)
            if root is not None and not getattr(gui, "_overview_sash_pending_map", False):
                gui._overview_sash_pending_map = True

                def _retry() -> None:
                    gui._overview_sash_pending_map = False
                    apply_saved_sash_fractions(gui, force_defaults=force_defaults)

                try:
                    root.after(120, _retry)
                except tk.TclError:
                    pass
            return
    except tk.TclError:
        return

    traj = bool(getattr(gui, "_overview_traj_dock_visible", False))
    if force_defaults:
        fr = DEFAULT_TRAJ_SASH_FRACTIONS if traj else DEFAULT_SASH_FRACTIONS
    else:
        fr = load_sash_fractions(traj=traj)

    x0 = int(round(fr[0] * width))
    x1 = int(round(fr[1] * width))
    x0, x1 = _clamp_sash_pixels(width, x0, x1)

    gui._overview_sash_applying = True
    try:
        paned.sashpos(0, x0)
        paned.sashpos(1, x1)
    except tk.TclError:
        pass
    finally:
        gui._overview_sash_applying = False
    _enforce_min_sizes(gui)


def _clamp_sash_pixels(width: int, x0: int, x1: int) -> tuple[int, int]:
    min_v, min_s, min_side = MIN_PANE_VIDEO, MIN_PANE_SKELETON, MIN_PANE_SIDE
    x0 = max(min_v, min(width - min_s - min_side, x0))
    x1 = max(x0 + min_s, min(width - min_side, x1))
    return x0, x1


def _enforce_min_sizes(gui: Any) -> None:
    paned: ttk.Panedwindow | None = getattr(gui, "_overview_paned", None)
    if paned is None:
        return
    try:
        panes = list(paned.panes())
        if len(panes) < 3:
            return
        width = max(1, int(paned.winfo_width()))
        x0 = int(paned.sashpos(0))
        x1 = int(paned.sashpos(1))
        nx0, nx1 = _clamp_sash_pixels(width, x0, x1)
        if (nx0, nx1) != (x0, x1):
            gui._overview_sash_applying = True
            try:
                paned.sashpos(0, nx0)
                paned.sashpos(1, nx1)
            finally:
                gui._overview_sash_applying = False
    except tk.TclError:
        pass


def reset_overview_layout(gui: Any) -> None:
    """One-click restore: clear prefs and apply default sash fractions."""
    clear_sash_fraction_prefs()
    set_overview_panes_visible(gui, (PANE_VIDEO, PANE_SKELETON, PANE_SIDE))
    apply_saved_sash_fractions(gui, force_defaults=True)
    save_sash_fractions(
        DEFAULT_TRAJ_SASH_FRACTIONS
        if getattr(gui, "_overview_traj_dock_visible", False)
        else DEFAULT_SASH_FRACTIONS,
        traj=bool(getattr(gui, "_overview_traj_dock_visible", False)),
    )
    _schedule_sync_visuals(gui, delay_ms=20)
    status = getattr(gui, "status", None)
    if status is not None:
        try:
            status.configure(text="Overview layout restored to defaults")
        except Exception:
            pass


def apply_overview_side_content(gui: Any, *, show_traj: bool) -> None:
    """Put either the sidebar summary or the trajectory dock into the side pane."""
    pane = getattr(gui, "_overview_pane_side", None)
    sidebar = getattr(gui, "sidebar", None)
    traj = getattr(gui, "overview_traj_panel", None)
    if pane is None:
        return
    if show_traj and traj is not None:
        if sidebar is not None:
            try:
                sidebar.grid_remove()
            except tk.TclError:
                pass
        mount_panel_in_pane(pane, traj)
        try:
            traj.grid()
        except tk.TclError:
            pass
    else:
        if traj is not None:
            try:
                traj.grid_remove()
            except tk.TclError:
                pass
        if sidebar is not None:
            mount_panel_in_pane(pane, sidebar)
            try:
                sidebar.grid()
            except tk.TclError:
                pass


def _hide_overview_visual(widget: tk.Misc | None) -> None:
    if widget is None:
        return
    try:
        widget.grid_remove()
    except tk.TclError:
        pass


def _col(
    section: tk.Misc,
    index: int,
    weight: int,
    *,
    uniform: str = "",
    minsize: int = 0,
) -> None:
    try:
        section.columnconfigure(
            index, weight=weight, uniform=uniform, minsize=max(0, int(minsize))
        )
    except tk.TclError:
        pass


def _place_summary_dock(gui: Any, section: tk.Misc, row: int) -> None:
    """Place the Gait Analysis Summary BELOW the primary visual row."""
    from stablewalk.ui.theme import PAD_XS

    sidebar = getattr(gui, "sidebar", None)
    traj = getattr(gui, "overview_traj_panel", None)
    expanded = bool(getattr(gui, "_overview_summary_expanded", True))

    try:
        section.rowconfigure(row, weight=0, minsize=0)
    except tk.TclError:
        pass

    # Trajectory is a sibling column, never the below-fold summary strip.
    if traj is not None:
        _hide_overview_visual(traj)
    if sidebar is None:
        return
    if not expanded:
        sidebar.grid(
            row=row,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=0,
            pady=(PAD_XS, 0),
        )
        return
    sidebar.grid(
        row=row,
        column=0,
        columnspan=3,
        sticky="ew",
        padx=0,
        pady=(PAD_XS, 0),
    )


def _place_trajectory_column(
    gui: Any,
    section: tk.Misc,
    *,
    row: int,
    column: int,
    columnspan: int = 1,
) -> bool:
    """Grid the 3D joint-path panel in a visible column. Returns True if placed."""
    from stablewalk.ui.theme import OVERVIEW_COL_GUTTER

    traj = getattr(gui, "overview_traj_panel", None)
    sidebar = getattr(gui, "sidebar", None)
    if traj is None or not bool(getattr(gui, "_overview_traj_dock_visible", False)):
        return False
    if sidebar is not None:
        _hide_overview_visual(sidebar)
    traj.grid(
        row=row,
        column=column,
        columnspan=columnspan,
        sticky="nsew",
        padx=(OVERVIEW_COL_GUTTER // 2 if column > 0 else 0, 0),
        pady=0,
    )
    return True


def layout_panels_for_view_mode(gui: Any, mode: str) -> None:
    """
    Arrange the Overview visualization panels for the selected view mode.

    Side-by-Side (default) always uses three equal-height columns:
    Video (34%) | Skeleton (36%) | Joint Path (30%). The path panel shows
    an empty-state prompt until a joint is selected.
    """
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        VIEW_MODE_OVERLAY,
        VIEW_MODE_RECONSTRUCTION_FULL,
        VIEW_MODE_SKELETON_ONLY,
        VIEW_MODE_SIDE_BY_SIDE,
        VIEW_MODE_VIDEO_ONLY,
    )
    from stablewalk.ui.tk.dashboard_sections import (
        SEC1_PANEL_MINWIDTH,
        SEC1_TRAJ_PANEL_MINWIDTH,
        SEC1_TRAJ_PATH_WEIGHT,
        SEC1_TRAJ_SKELETON_WEIGHT,
        SEC1_TRAJ_VIDEO_WEIGHT,
        SEC1_TRAJ_VIZ_ROW_MINSIZE,
        overview_panel_padx,
    )

    section = getattr(gui, "_section_visual", None) or getattr(gui, "_primary_viz_host", None)
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    sidebar = getattr(gui, "sidebar", None)
    traj = getattr(gui, "overview_traj_panel", None)
    if section is None or video is None or skel is None:
        return

    row = int(getattr(gui, "_primary_viz_content_row", 0))
    summary_row = int(getattr(gui, "_overview_summary_row", row + 1))
    # Path column is always part of the Side-by-Side workspace (empty until pick).
    gui._overview_traj_dock_visible = True
    show_traj = traj is not None
    viz_minsize = SEC1_TRAJ_VIZ_ROW_MINSIZE

    for widget in (video, skel, sidebar, traj):
        _hide_overview_visual(widget)

    try:
        section.rowconfigure(row, weight=1, minsize=viz_minsize)
        section.rowconfigure(summary_row, weight=0, minsize=0)
    except tk.TclError:
        pass

    if mode in (VIEW_MODE_VIDEO_ONLY, VIEW_MODE_OVERLAY):
        # Video + docked non-overlapping 3D path.
        _col(section, 0, 66, uniform="sec1", minsize=SEC1_PANEL_MINWIDTH)
        _col(section, 1, 34, uniform="sec1", minsize=SEC1_TRAJ_PANEL_MINWIDTH)
        _col(section, 2, 0)
        video.grid(
            row=row,
            column=0,
            sticky="nsew",
            padx=overview_panel_padx(0),
            pady=0,
        )
        if show_traj:
            _place_trajectory_column(gui, section, row=row, column=1)
        else:
            _place_summary_dock(gui, section, summary_row)
    elif mode == VIEW_MODE_RECONSTRUCTION_FULL:
        _col(section, 0, 1, minsize=SEC1_PANEL_MINWIDTH)
        _col(section, 1, 0)
        _col(section, 2, 0)
        skel.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=0, pady=0)
    elif mode == VIEW_MODE_SKELETON_ONLY:
        _col(section, 0, 66, uniform="sec1", minsize=SEC1_PANEL_MINWIDTH)
        _col(section, 1, 34, uniform="sec1", minsize=SEC1_TRAJ_PANEL_MINWIDTH)
        _col(section, 2, 0)
        skel.grid(
            row=row,
            column=0,
            sticky="nsew",
            padx=overview_panel_padx(0),
            pady=0,
        )
        if show_traj:
            _place_trajectory_column(gui, section, row=row, column=1)
        else:
            _place_summary_dock(gui, section, summary_row)
    else:
        # side_by_side (default): Video | Skeleton | 3D Joint Path
        _ = VIEW_MODE_SIDE_BY_SIDE
        _col(
            section,
            0,
            SEC1_TRAJ_VIDEO_WEIGHT,
            uniform="sec1",
            minsize=SEC1_PANEL_MINWIDTH,
        )
        _col(
            section,
            1,
            SEC1_TRAJ_SKELETON_WEIGHT,
            uniform="sec1",
            minsize=SEC1_PANEL_MINWIDTH,
        )
        _col(
            section,
            2,
            SEC1_TRAJ_PATH_WEIGHT,
            uniform="sec1",
            minsize=SEC1_TRAJ_PANEL_MINWIDTH,
        )
        video.grid(
            row=row,
            column=0,
            sticky="nsew",
            padx=overview_panel_padx(0),
            pady=0,
        )
        skel.grid(
            row=row,
            column=1,
            sticky="nsew",
            padx=overview_panel_padx(1),
            pady=0,
        )
        if show_traj:
            _place_trajectory_column(gui, section, row=row, column=2)
        else:
            _place_summary_dock(gui, section, summary_row)

    refresh = getattr(gui, "_refresh_overview_trajectory_dock", None)
    if callable(refresh) and show_traj:
        try:
            refresh(force_draw=True)
        except Exception:
            pass

    _schedule_sync_visuals(gui, delay_ms=40)
    schedule = getattr(gui, "_schedule_overview_media_refit", None)
    if callable(schedule):
        try:
            schedule()
        except Exception:
            pass


__all__ = [
    "DEFAULT_SASH_FRACTIONS",
    "DEFAULT_TRAJ_SASH_FRACTIONS",
    "MIN_PANE_SIDE",
    "MIN_PANE_SKELETON",
    "MIN_PANE_VIDEO",
    "PANE_SIDE",
    "PANE_SKELETON",
    "PANE_VIDEO",
    "apply_overview_side_content",
    "apply_saved_sash_fractions",
    "clear_sash_fraction_prefs",
    "install_overview_paned_host",
    "layout_panels_for_view_mode",
    "load_sash_fractions",
    "mount_panel_in_pane",
    "reset_overview_layout",
    "save_sash_fractions",
    "set_overview_panes_visible",
]
