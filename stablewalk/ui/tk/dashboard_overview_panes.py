"""Overview horizontal paned layout - draggable sashes with persisted fractions."""

from __future__ import annotations

import json
import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Sequence

from stablewalk import config
from stablewalk.ui.tk.analysis_split import (
    MIN_W_PATH,
    MIN_W_SKELETON,
    MIN_W_VIDEO,
    split_clear,
)
from stablewalk.ui.theme import (
    PANEL,
    SIDEBAR_MIN_WIDTH,
)

logger = logging.getLogger(__name__)

# Default right-edge fractions for sash 0 and sash 1.
# Side-by-side (no path): equal video | skeleton.
DEFAULT_SASH_FRACTIONS: tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0)
# Trajectory mode: Video 40% | Skeleton 30% | 3D path 30%.
DEFAULT_TRAJ_SASH_FRACTIONS: tuple[float, float] = (0.40, 0.70)
# Narrow layout: video share of the Video|Skeleton row (40 / 70).
DEFAULT_PAIR_SASH_FRACTION = 0.40 / 0.70
# Narrow layout: height share for the Video|Skeleton row above the 3D plot.
DEFAULT_STACK_SASH_FRACTION = 0.55

MIN_PANE_VIDEO = MIN_W_VIDEO
MIN_PANE_SKELETON = MIN_W_SKELETON
MIN_PANE_SIDE = max(MIN_W_PATH, int(SIDEBAR_MIN_WIDTH))


def _scaled_overview_pane_mins(gui: Any | None = None) -> tuple[int, int, int]:
    """DPI-aware Video / Skeleton / 3D Path floors."""
    from stablewalk.ui.tk.analysis_split import overview_pane_minsizes
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    root = getattr(gui, "root", None) if gui is not None else None
    video, skeleton, path = overview_pane_minsizes(dpi_scale_factor(root))
    return video, skeleton, max(path, int(SIDEBAR_MIN_WIDTH))


_PREF_KEY = "overview_sash_fractions"
_PREF_KEY_TRAJ = "overview_sash_fractions_traj_v3"
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
    data.pop("overview_sash_fractions_traj_v2", None)
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError:
        pass


DEFAULT_SECONDARY_INFO_H = 200
# Tight windows keep a compact metrics strip; leftover height stays with viewers.
COMPACT_SECONDARY_INFO_H = 120
_INFO_SCROLLBAR_W = 14


def install_overview_workspace_split(
    gui: Any, overview_host: tk.Misc
) -> tuple[tk.Misc, tk.Misc]:
    """Primary viewers (top) + secondary gait info (bottom), vertical sash.

    Extra height goes to the primary pane. Secondary gait info is clipped by
    the pane and scrolls in-place (no Canvas.create_window — Windows would
    paint ttk cards through the transport bar). Viewers stay off any Canvas.
    """
    from stablewalk.ui.theme import OVERVIEW_ROW_GAP, PANEL
    from stablewalk.ui.tk.analysis_split import (
        MIN_H_PRIMARY_VIZ,
        MIN_H_SECONDARY_INFO,
        create_v_split,
        split_add,
    )
    from stablewalk.ui.tk.dashboard_sections import OVERVIEW_HOST_VIZ_ROW

    try:
        overview_host.columnconfigure(0, weight=1)
        overview_host.rowconfigure(OVERVIEW_HOST_VIZ_ROW, weight=1)
    except tk.TclError:
        pass

    split = create_v_split(overview_host)
    split.grid(row=OVERVIEW_HOST_VIZ_ROW, column=0, sticky="nsew")
    gui._overview_main_split = split

    primary = tk.Frame(split, bg=PANEL, highlightthickness=0)
    primary.columnconfigure(0, weight=1)
    primary.rowconfigure(0, weight=1)
    try:
        primary.grid_propagate(False)
    except tk.TclError:
        pass
    gui._section_visual = primary
    gui._primary_viz_host = primary
    gui._primary_viz_content_row = 0

    info_outer = tk.Frame(split, bg=PANEL, highlightthickness=0, highlightbackground=PANEL)
    info_inner = tk.Frame(info_outer, bg=PANEL, highlightthickness=0)
    vsb = ttk.Scrollbar(info_outer, orient=tk.VERTICAL)
    # Paned pane sizes outer. Inner is placed and may be taller; pane clips it.
    gui._overview_info_outer = info_outer
    gui._overview_info_inner = info_inner
    gui._overview_info_canvas = None
    gui._overview_info_vsb = vsb
    gui._overview_info_window_id = None
    gui._overview_info_scroll_needed = False
    gui._overview_info_scroll_offset = 0
    gui._overview_info_scroll_applying = False
    gui._overview_v_sash_user = False

    split_add(split, primary, minsize=MIN_H_PRIMARY_VIZ, stretch="always")
    split_add(split, info_outer, minsize=MIN_H_SECONDARY_INFO, stretch="never")
    split.bind(
        "<ButtonRelease-1>",
        lambda _e: _on_overview_vertical_sash_release(gui),
        add="+",
    )
    split.bind("<Configure>", lambda _e: _place_overview_vertical_sash(gui), add="+")
    try:
        info_inner.columnconfigure(0, weight=1)
    except tk.TclError:
        pass
    info_outer.bind("<Configure>", lambda _e: sync_overview_info_scroll(gui), add="+")
    vsb.configure(command=lambda *args: _overview_info_yview(gui, *args))
    for widget in (info_outer, info_inner):
        widget.bind("<MouseWheel>", lambda e: _on_overview_info_wheel(gui, e), add="+")
        widget.bind("<Button-4>", lambda e: _on_overview_info_wheel(gui, e), add="+")
        widget.bind("<Button-5>", lambda e: _on_overview_info_wheel(gui, e), add="+")

    _ = OVERVIEW_ROW_GAP
    try:
        info_inner.place(x=0, y=0, relwidth=1)
    except tk.TclError:
        pass
    return primary, info_inner


def _overview_info_content_height(inner: tk.Misc) -> int:
    try:
        inner.update_idletasks()
        req = int(inner.winfo_reqheight() or 0)
    except tk.TclError:
        req = 0
    bottom = 0
    try:
        for child in inner.winfo_children():
            try:
                if str(child.winfo_manager() or "") == "":
                    continue
                bottom = max(
                    bottom,
                    int(child.winfo_y()) + int(child.winfo_reqheight() or 0),
                )
            except tk.TclError:
                continue
    except tk.TclError:
        pass
    return max(req, bottom)


def _overview_info_yview(gui: Any, *args: object) -> None:
    inner = getattr(gui, "_overview_info_inner", None)
    outer = getattr(gui, "_overview_info_outer", None)
    if not _is_tk_widget(inner) or not _is_tk_widget(outer):
        return
    try:
        have = max(1, int(outer.winfo_height() or 0))
        need = max(have, _overview_info_content_height(inner))
    except tk.TclError:
        return
    max_off = max(0, need - have)
    offset = int(getattr(gui, "_overview_info_scroll_offset", 0) or 0)
    if not args:
        return
    cmd = str(args[0])
    try:
        if cmd == "moveto" and len(args) >= 2:
            offset = int(round(float(args[1]) * max_off))
        elif cmd == "scroll" and len(args) >= 3:
            step = 28 if str(args[2]) == "units" else have
            offset = offset + int(args[1]) * step
    except (TypeError, ValueError):
        return
    gui._overview_info_scroll_offset = max(0, min(max_off, offset))
    sync_overview_info_scroll(gui)


def _on_overview_info_wheel(gui: Any, event: tk.Event) -> str | None:
    if not bool(getattr(gui, "_overview_info_scroll_needed", False)):
        return None
    delta = int(getattr(event, "delta", 0) or 0)
    num = int(getattr(event, "num", 0) or 0)
    if delta != 0:
        steps = -1 if delta > 0 else 1
        if abs(delta) >= 120:
            steps = -int(delta / 120)
    elif num == 4:
        steps = -1
    elif num == 5:
        steps = 1
    else:
        return None
    _overview_info_yview(gui, "scroll", steps, "units")
    return "break"


def sync_overview_info_scroll(gui: Any) -> None:
    """Place the secondary inner frame; show a scrollbar when it overflows."""
    if getattr(gui, "_overview_info_scroll_applying", False):
        return
    inner = getattr(gui, "_overview_info_inner", None)
    outer = getattr(gui, "_overview_info_outer", None)
    vsb = getattr(gui, "_overview_info_vsb", None)
    if not _is_tk_widget(inner) or not _is_tk_widget(outer):
        gui._overview_info_scroll_needed = False
        gui._overview_scroll_needed = False
        return
    gui._overview_info_scroll_applying = True
    try:
        try:
            outer.update_idletasks()
            have = max(1, int(outer.winfo_height() or 0))
            ow = max(1, int(outer.winfo_width() or 0))
        except tk.TclError:
            return
        need = _overview_info_content_height(inner)
        overflow = need > have + 4
        gui._overview_info_scroll_needed = overflow
        # Page-level Overview canvas scroll stays off (avoids nested scrolling).
        gui._overview_scroll_needed = False
        max_off = max(0, need - have) if overflow else 0
        offset = int(getattr(gui, "_overview_info_scroll_offset", 0) or 0)
        offset = max(0, min(max_off, offset))
        gui._overview_info_scroll_offset = offset
        sb_w = _INFO_SCROLLBAR_W if overflow and _is_tk_widget(vsb) else 0
        sig = (have, ow, need, offset, sb_w, overflow)
        if sig == getattr(gui, "_overview_info_scroll_sig", None):
            return
        gui._overview_info_scroll_sig = sig
        try:
            # relwidth and width are added by Tk place — never combine them.
            inner.place(
                x=0,
                y=-offset,
                relwidth=0,
                relheight=0,
                width=max(1, ow - sb_w),
            )
        except tk.TclError:
            pass
        if overflow and _is_tk_widget(vsb):
            try:
                vsb.place(
                    relx=1.0,
                    x=0,
                    y=0,
                    relheight=1.0,
                    width=_INFO_SCROLLBAR_W,
                    anchor="ne",
                )
                first = offset / max(need, 1)
                last = (offset + have) / max(need, 1)
                vsb.set(max(0.0, first), min(1.0, last))
            except tk.TclError:
                pass
        elif _is_tk_widget(vsb):
            try:
                vsb.place_forget()
            except tk.TclError:
                pass
    finally:
        gui._overview_info_scroll_applying = False


def _on_overview_vertical_sash_release(gui: Any) -> None:
    split = getattr(gui, "_overview_main_split", None)
    if not _is_tk_widget(split):
        return
    try:
        height = max(1, int(split.winfo_height() or 0))
        y0 = int(split.sash_coord(0)[1])
        gui._overview_v_sash_frac = max(0.35, min(0.92, y0 / height))
        gui._overview_v_sash_user = True
    except (tk.TclError, TypeError, ValueError, IndexError):
        pass
    sync_overview_info_scroll(gui)
    sync_overview_viewer_split_height(gui)


def _auto_overview_primary_height(height: int) -> int:
    """Viewers own leftover height. Metrics get a compact strip when tight."""
    from stablewalk.ui.tk.analysis_split import MIN_H_PRIMARY_VIZ, MIN_H_SECONDARY_INFO

    min_primary = int(MIN_H_PRIMARY_VIZ)
    min_secondary = int(MIN_H_SECONDARY_INFO)
    comfortable = int(DEFAULT_SECONDARY_INFO_H)
    compact = int(COMPACT_SECONDARY_INFO_H)
    sash = 8
    # Comfortable metrics only when the viewers can stay clearly taller.
    # A 517 px split used to reserve 200 px and freeze viewers at ~316 px.
    if height >= min_primary + comfortable + 160:
        secondary = comfortable
    elif height >= min_primary + compact + sash:
        secondary = compact
    else:
        secondary = compact
        if height - secondary < min_primary:
            secondary = max(min_secondary, height - min_primary)
    primary = height - secondary
    if primary < min_primary:
        primary = min(min_primary, height - min_secondary)
        primary = max(120, primary)
    if height - primary < min_secondary:
        primary = max(120, height - min_secondary)
    return int(primary)


def _place_overview_vertical_sash(gui: Any) -> None:
    """Keep viewers at least MIN_H_PRIMARY_VIZ; leftover goes to info."""
    from stablewalk.ui.tk.analysis_split import MIN_H_PRIMARY_VIZ, MIN_H_SECONDARY_INFO

    split = getattr(gui, "_overview_main_split", None)
    if not _is_tk_widget(split):
        return
    if getattr(gui, "_overview_v_sash_applying", False):
        return
    try:
        split.update_idletasks()
        height = int(split.winfo_height() or 0)
    except tk.TclError:
        return
    if height < 160:
        return
    last_h = int(getattr(gui, "_overview_v_sash_host_h", 0) or 0)
    user = bool(getattr(gui, "_overview_v_sash_user", False))
    if (
        abs(height - last_h) < 4
        and getattr(gui, "_overview_v_sash_placed", False)
        and user
    ):
        return
    frac = getattr(gui, "_overview_v_sash_frac", None)
    if user and frac is not None:
        primary = int(round(float(frac) * height))
    else:
        primary = _auto_overview_primary_height(height)
    if primary < int(MIN_H_PRIMARY_VIZ):
        primary = min(int(MIN_H_PRIMARY_VIZ), height - int(MIN_H_SECONDARY_INFO))
        primary = max(120, primary)
    if height - primary < int(MIN_H_SECONDARY_INFO):
        primary = max(120, height - int(MIN_H_SECONDARY_INFO))
    gui._overview_v_sash_applying = True
    try:
        split.sash_place(0, 0, int(primary))
        gui._overview_v_sash_placed = True
        gui._overview_v_sash_host_h = height
    except tk.TclError:
        pass
    finally:
        gui._overview_v_sash_applying = False
    sync_overview_info_scroll(gui)
    sync_overview_viewer_split_height(gui)
    if not bool(getattr(gui, "playing", False)):
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
                pass
    if not bool(getattr(gui, "playing", False)):
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
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
    from stablewalk.ui.tk.layout_playback import is_playing, mark_layout_dirty

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
    # During playback only keep clip windows sized — defer figure/video fits.
    if is_playing(gui):
        mark_layout_dirty(gui)
        return
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
    min_v, min_s, min_p = _scaled_overview_pane_mins(gui)
    x0, x1 = _clamp_sash_pixels(
        width,
        x0,
        x1,
        min_video=min_v,
        min_skeleton=min_s,
        min_path=min_p,
    )

    gui._overview_sash_applying = True
    try:
        paned.sashpos(0, x0)
        paned.sashpos(1, x1)
    except tk.TclError:
        pass
    finally:
        gui._overview_sash_applying = False
    _enforce_min_sizes(gui)


def _clamp_sash_pixels(
    width: int,
    x0: int,
    x1: int,
    *,
    min_video: int | None = None,
    min_skeleton: int | None = None,
    min_path: int | None = None,
) -> tuple[int, int]:
    """Keep Video | Skeleton | Path usable; Video stays the primary pane."""
    min_v = int(min_video if min_video is not None else MIN_PANE_VIDEO)
    min_s = int(min_skeleton if min_skeleton is not None else MIN_PANE_SKELETON)
    min_side = int(min_path if min_path is not None else MIN_PANE_SIDE)
    width = max(1, int(width))
    total_min = min_v + min_s + min_side
    if total_min > width:
        # Floors do not fit (laptop + high DPI). Shrink video, then skeleton,
        # and keep the 3D path as wide as the remaining space allows.
        overflow = total_min - width
        take = min(overflow, max(0, min_v - 160))
        min_v -= take
        overflow -= take
        if overflow > 0:
            take = min(overflow, max(0, min_s - 160))
            min_s -= take
            overflow -= take
        if overflow > 0:
            min_side = max(180, min_side - overflow)
        if min_v + min_s + min_side > width:
            min_v = max(120, int(round(width * 0.34)))
            min_s = max(120, int(round(width * 0.28)))
            min_side = max(160, width - min_v - min_s)
    x0 = max(min_v, min(width - min_s - min_side, x0))
    x1 = max(x0 + min_s, min(width - min_side, x1))
    # Video is the primary pane (~40%). Path may grow, but not past 40%.
    max_side = max(min_side, int(width * 0.40))
    if width - x1 > max_side:
        x1 = max(x0 + min_s, width - max_side)
    min_video = min(int(width * 0.32), max(min_v, (width - min_s - min_side) // 3))
    if x0 < min_video and min_video + min_s + min_side <= width:
        x0 = min_video
        x1 = max(x0 + min_s, x1)
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
        min_v, min_s, min_p = _scaled_overview_pane_mins(gui)
        nx0, nx1 = _clamp_sash_pixels(
            width,
            x0,
            x1,
            min_video=min_v,
            min_skeleton=min_s,
            min_path=min_p,
        )
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
    """Restore default Split proportions without reloading video or analysis."""
    clear_sash_fraction_prefs()
    gui._overview_split_layout = None
    zoom = getattr(gui, "_video_display_zoom_var", None)
    if zoom is not None:
        try:
            zoom.set("Fit")
        except tk.TclError:
            pass
    set_overview_panes_visible(gui, (PANE_VIDEO, PANE_SKELETON, PANE_SIDE))
    from stablewalk.ui.tk.overview_responsive import apply_overview_viz_arrangement

    apply_overview_viz_arrangement(gui)
    apply_saved_sash_fractions(gui, force_defaults=True)
    save_sash_fractions(
        DEFAULT_TRAJ_SASH_FRACTIONS
        if getattr(gui, "_overview_traj_dock_visible", False)
        else DEFAULT_SASH_FRACTIONS,
        traj=bool(getattr(gui, "_overview_traj_dock_visible", False)),
    )
    _place_nested_sashes(gui, force_defaults=True)
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


def _is_tk_widget(widget: Any) -> bool:
    return isinstance(widget, (tk.Misc, ttk.Widget))


def _clear_overview_viewer_split(gui: Any) -> None:
    for attr in (
        "_overview_viz_split",
        "_overview_viz_top_split",
        "_overview_viz_vsplit",
    ):
        paned = getattr(gui, attr, None)
        if not _is_tk_widget(paned):
            continue
        try:
            split_clear(paned)
        except Exception:
            pass
        try:
            paned.grid_remove()
        except tk.TclError:
            pass


def _mapped_split(widget: Any) -> bool:
    if not _is_tk_widget(widget):
        return False
    try:
        return str(widget.winfo_manager() or "") == "grid"
    except tk.TclError:
        return False


def bind_overview_viewer_split_sashes(gui: Any, paned: tk.PanedWindow) -> None:
    """Keep Video | Skeleton | Path draggable and proportional on resize."""

    def _press(_event: object | None = None) -> None:
        gui._overview_split_dragging = True

    def _release(_event: object | None = None) -> None:
        _on_overview_split_release(gui, paned)
        gui._overview_split_dragging = False

    def _configure(_event: object | None = None) -> None:
        if bool(getattr(gui, "_overview_split_dragging", False)):
            return
        if bool(getattr(gui, "_overview_sash_applying", False)):
            return
        if str(getattr(gui, "_overview_split_layout", "") or "") != "three_col":
            return
        _place_overview_split_sashes(gui, paned)

    paned.bind("<ButtonPress-1>", _press, add="+")
    paned.bind("<ButtonRelease-1>", _release, add="+")
    paned.bind("<Configure>", _configure, add="+")


def _place_overview_split_sashes(gui: Any, paned: tk.PanedWindow) -> None:
    """Restore Video | Skeleton | Path sash fractions after the split maps."""
    if bool(getattr(gui, "_overview_split_dragging", False)):
        return
    if bool(getattr(gui, "_overview_sash_applying", False)):
        return
    try:
        paned.update_idletasks()
        width = int(paned.winfo_width() or 0)
    except tk.TclError:
        return
    if width < 200:
        root = getattr(gui, "root", None)
        if root is not None and not getattr(gui, "_overview_split_sash_pending", False):
            gui._overview_split_sash_pending = True

            def _retry() -> None:
                gui._overview_split_sash_pending = False
                _place_overview_split_sashes(gui, paned)

            try:
                root.after(80, _retry)
            except tk.TclError:
                pass
        return
    fr = load_sash_fractions(traj=True)
    min_v, min_s, min_p = _scaled_overview_pane_mins(gui)
    x0, x1 = _clamp_sash_pixels(
        width,
        int(round(fr[0] * width)),
        int(round(fr[1] * width)),
        min_video=min_v,
        min_skeleton=min_s,
        min_path=min_p,
    )
    gui._overview_sash_applying = True
    try:
        paned.sash_place(0, x0, 0)
        paned.sash_place(1, x1, 0)
    except tk.TclError:
        pass
    finally:
        gui._overview_sash_applying = False


def _place_nested_sashes(gui: Any, *, force_defaults: bool = False) -> None:
    """Narrow layouts use grid weights; no nested PanedWindow sashes to place."""
    _ = gui, force_defaults


def _on_overview_split_release(gui: Any, paned: tk.PanedWindow) -> None:
    try:
        width = max(1, int(paned.winfo_width() or 0))
        x0 = int(paned.sash_coord(0)[0])
        x1 = int(paned.sash_coord(1)[0])
        min_v, min_s, min_p = _scaled_overview_pane_mins(gui)
        x0, x1 = _clamp_sash_pixels(
            width,
            x0,
            x1,
            min_video=min_v,
            min_skeleton=min_s,
            min_path=min_p,
        )
        gui._overview_sash_applying = True
        try:
            paned.sash_place(0, x0, 0)
            paned.sash_place(1, x1, 0)
        finally:
            gui._overview_sash_applying = False
        save_sash_fractions((x0 / width, x1 / width), traj=True)
    except (tk.TclError, TypeError, ValueError, IndexError):
        pass
    gui._overview_split_dragging = False
    _schedule_sync_visuals(gui, delay_ms=20)


def _on_nested_split_release(gui: Any) -> None:
    top = getattr(gui, "_overview_viz_top_split", None)
    vsplit = getattr(gui, "_overview_viz_vsplit", None)
    try:
        if _mapped_split(top):
            tw = max(1, int(top.winfo_width() or 0))
            gui._overview_pair_sash = int(top.sash_coord(0)[0]) / tw
        if _mapped_split(vsplit):
            vh = max(1, int(vsplit.winfo_height() or 0))
            gui._overview_stack_sash = int(vsplit.sash_coord(0)[1]) / vh
    except (tk.TclError, TypeError, ValueError, IndexError):
        pass
    _schedule_sync_visuals(gui, delay_ms=20)


def _grid_viewers_in_holders(
    video: tk.Misc, skel: tk.Misc, traj: tk.Misc, pane_v: Any, pane_s: Any, pane_t: Any
) -> None:
    for panel, pane in ((video, pane_v), (skel, pane_s), (traj, pane_t)):
        if panel is None or not _is_tk_widget(pane):
            continue
        try:
            panel.grid(in_=pane, row=0, column=0, sticky="nsew", padx=0, pady=0)
            pane.columnconfigure(0, weight=1)
            pane.rowconfigure(0, weight=1)
        except tk.TclError:
            pass


def mount_overview_viewer_split(
    gui: Any,
    section: tk.Misc,
    video: tk.Misc,
    skel: tk.Misc,
    traj: tk.Misc,
    *,
    row: int = 0,
    layout: str = "three_col",
) -> bool:
    """Show Video / Skeleton / Path in holders. Never reparent clip canvases."""
    from stablewalk.ui.tk.analysis_split import (
        MIN_W_PATH,
        MIN_W_SKELETON,
        MIN_W_VIDEO,
        split_add,
    )

    paned = getattr(gui, "_overview_viz_split", None)
    pane_v = getattr(gui, "_overview_pane_video", None)
    pane_s = getattr(gui, "_overview_pane_skeleton", None)
    pane_t = getattr(gui, "_overview_pane_path", None)
    if not (
        _is_tk_widget(section)
        and _is_tk_widget(paned)
        and _is_tk_widget(pane_v)
        and _is_tk_widget(pane_s)
        and _is_tk_widget(pane_t)
    ):
        return False
    mode = layout if layout in {"three_col", "two_plus_one", "stacked"} else "three_col"
    current = str(getattr(gui, "_overview_split_layout", "") or "")
    if current == mode:
        if mode == "three_col" and _mapped_split(paned):
            _place_overview_split_sashes(gui, paned)
            sync_overview_viewer_split_height(gui)
            return True
        if mode != "three_col" and _mapped_split(pane_v) and not _mapped_split(paned):
            sync_overview_viewer_split_height(gui)
            return True
    try:
        _grid_viewers_in_holders(video, skel, traj, pane_v, pane_s, pane_t)
        _clear_overview_viewer_split(gui)
        try:
            section.columnconfigure(0, weight=1, minsize=0)
            section.columnconfigure(1, weight=0, minsize=0)
            section.columnconfigure(2, weight=0, minsize=0)
            section.rowconfigure(row, weight=1)
            section.rowconfigure(row + 1, weight=0, minsize=0)
            section.rowconfigure(row + 2, weight=0, minsize=0)
        except tk.TclError:
            pass
        if mode in {"two_plus_one", "stacked"}:
            # Holders stay children of the horizontal splitter. Grid them into
            # the ancestor section (Tk ``in_``) so canvases are never reparented.
            try:
                paned.grid_remove()
            except tk.TclError:
                pass
            from stablewalk.ui.tk.dashboard_sections import apply_overview_viz_grid_weights

            apply_overview_viz_grid_weights(section, content_row=row, layout=mode)
            if mode == "two_plus_one":
                pane_v.grid(in_=section, row=row, column=0, sticky="nsew")
                pane_s.grid(in_=section, row=row, column=1, sticky="nsew")
                pane_t.grid(
                    in_=section,
                    row=row + 1,
                    column=0,
                    columnspan=2,
                    sticky="nsew",
                )
            else:
                pane_v.grid(in_=section, row=row, column=0, sticky="nsew")
                pane_s.grid(in_=section, row=row + 1, column=0, sticky="nsew")
                pane_t.grid(in_=section, row=row + 2, column=0, sticky="nsew")
            gui._overview_split_layout = mode
        else:
            for pane in (pane_v, pane_s, pane_t):
                try:
                    pane.grid_forget()
                except tk.TclError:
                    pass
            min_v, min_s, min_p = _scaled_overview_pane_mins(gui)
            split_add(paned, pane_v, minsize=min_v)
            split_add(paned, pane_s, minsize=min_s)
            split_add(paned, pane_t, minsize=min_p)
            paned.grid(row=row, column=0, columnspan=3, sticky="nsew")
            gui._overview_split_layout = "three_col"
            _place_overview_split_sashes(gui, paned)
        sync_overview_viewer_split_height(gui)
        return True
    except Exception:
        logger.exception("Overview viewer split mount failed")
        return False


def sync_overview_viewer_split_height(gui: Any) -> None:
    """Let the mapped splitter expand with the primary pane. No fixed pixel height."""
    from stablewalk.ui.tk.analysis_split import MIN_H_PRIMARY_VIZ

    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    if not _is_tk_widget(section):
        return
    row = int(getattr(gui, "_primary_viz_content_row", 0) or 0)
    layout = str(getattr(gui, "_overview_split_layout", "") or "three_col")
    floor = int(MIN_H_PRIMARY_VIZ)
    try:
        if layout == "two_plus_one":
            section.rowconfigure(row, weight=1, minsize=max(1, floor // 2))
            section.rowconfigure(row + 1, weight=1, minsize=max(1, floor // 2))
            section.rowconfigure(row + 2, weight=0, minsize=0)
        elif layout == "stacked":
            for offset in range(3):
                section.rowconfigure(row + offset, weight=1, minsize=max(1, floor // 3))
        else:
            section.rowconfigure(row, weight=1, minsize=floor)
            section.rowconfigure(row + 1, weight=0, minsize=0)
            section.rowconfigure(row + 2, weight=0, minsize=0)
    except tk.TclError:
        pass
    for attr in (
        "_overview_viz_split",
        "_overview_viz_vsplit",
        "_overview_viz_top_split",
    ):
        paned = getattr(gui, attr, None)
        if not _mapped_split(paned):
            continue
        try:
            paned.grid_configure(sticky="nsew")
        except tk.TclError:
            pass


def place_overview_under_viewer_chrome(
    gui: Any, section: tk.Misc, *, chrome_row: int, span: int = 3
) -> None:
    """Keep phase + foot-clearance in the secondary info pane, never on video."""
    from stablewalk.ui.theme import OVERVIEW_ROW_GAP
    from stablewalk.ui.tk.overview_responsive import _overview_phase_host

    info = getattr(gui, "_overview_info_inner", None)
    parent = info if _is_tk_widget(info) else section
    phase = _overview_phase_host(gui, parent)
    if phase is None:
        phase = _overview_phase_host(gui, section)
    clearance = getattr(gui, "_foot_clearance_detail_host", None)
    try:
        if parent is info:
            if phase is not None:
                phase.grid(
                    in_=parent,
                    row=0,
                    column=0,
                    sticky="ew",
                    pady=(0, OVERVIEW_ROW_GAP),
                )
            if clearance is not None:
                clearance.grid(
                    in_=parent,
                    row=2,
                    column=0,
                    sticky="ew",
                    pady=(OVERVIEW_ROW_GAP, 0),
                )
            return
        if phase is not None:
            phase.grid(
                row=chrome_row,
                column=0,
                columnspan=span,
                sticky="ew",
                pady=(OVERVIEW_ROW_GAP, 0),
            )
        if clearance is not None:
            clearance.grid(
                row=chrome_row + 1,
                column=0,
                columnspan=span,
                sticky="ew",
                pady=(OVERVIEW_ROW_GAP, 0),
            )
    except tk.TclError:
        pass


def overview_viz_panels_mounted(gui: Any) -> bool:
    """True when Video, Skeleton, and 3D Path are visible in the viz row."""
    layout = str(getattr(gui, "_overview_split_layout", "") or "")
    if layout in {"two_plus_one", "stacked"}:
        pane_v = getattr(gui, "_overview_pane_video", None)
        if _mapped_split(pane_v):
            return True
    paned = getattr(gui, "_overview_viz_split", None)
    if _is_tk_widget(paned):
        try:
            if str(paned.winfo_manager() or "") == "grid" and len(paned.panes()) >= 3:
                return True
        except (tk.TclError, TypeError, ValueError):
            pass
    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    if section is None:
        return False
    for attr in ("video_frame", "skel_frame", "overview_traj_panel"):
        widget = getattr(gui, attr, None)
        if widget is None:
            return False
        try:
            if str(widget.winfo_manager() or "") != "grid":
                return False
            info = widget.grid_info()
            parent = info.get("in") or getattr(widget, "master", None)
            if parent is not None and parent is not section:
                return False
            if int(widget.winfo_width() or 0) <= 1 and int(widget.winfo_height() or 0) <= 1:
                if not widget.winfo_ismapped():
                    return False
        except (tk.TclError, TypeError, ValueError):
            return False
    return True


def apply_overview_three_panel_layout(gui: Any) -> bool:
    """Place Video | Skeleton | 3D Path in the expanding viewer row.

    Panels are created once in dashboard layout. This only grids those
    existing widgets. Repeated calls must not rebuild or unmap them.
    """
    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_HOST_VIZ_ROW,
        apply_overview_host_row_weights,
        apply_overview_panel_min_heights,
    )

    section = getattr(gui, "_section_visual", None) or getattr(
        gui, "_primary_viz_host", None
    )
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    traj = getattr(gui, "overview_traj_panel", None)
    if section is None or video is None or skel is None or traj is None:
        return False

    if overview_viz_panels_mounted(gui):
        gui._overview_traj_dock_visible = True
        if not bool(getattr(gui, "playing", False)):
            try:
                apply_overview_panel_min_heights(gui)
            except Exception:
                pass
        return True

    sidebar = getattr(gui, "sidebar", None)
    if sidebar is not None:
        _hide_overview_visual(sidebar)

    from stablewalk.ui.tk.dashboard_overview_view_mode import VIEW_MODE_SIDE_BY_SIDE
    from stablewalk.ui.tk.overview_responsive import apply_overview_viz_arrangement

    gui._overview_focus_mode = VIEW_MODE_SIDE_BY_SIDE
    apply_overview_viz_arrangement(gui)
    gui._overview_traj_dock_visible = True

    overview_host = getattr(gui, "_overview_layout_host", None) or getattr(
        gui, "_overview_content", None
    )
    if overview_host is not None:
        apply_overview_host_row_weights(overview_host, viz_row=OVERVIEW_HOST_VIZ_ROW)
    apply_overview_panel_min_heights(gui)
    _fill_overview_plot_widgets(gui)
    return True


def _fill_overview_plot_widgets(gui: Any) -> None:
    """Video clip, skeleton MPL, and path MPL must fill their hosts."""
    from stablewalk.ui.tk.clip_viewport import ensure_widget_fills_host

    ensure_widget_fills_host(getattr(gui, "_video_clip_canvas", None))
    ensure_widget_fills_host(getattr(gui, "video_label", None))
    ensure_widget_fills_host(getattr(gui, "_skel_clip_canvas", None))
    for attr in ("canvas_3d", "canvas_dof_traj_overview"):
        canvas = getattr(gui, attr, None)
        if canvas is None:
            continue
        getter = getattr(canvas, "get_tk_widget", None)
        widget = getter() if callable(getter) else canvas
        ensure_widget_fills_host(widget)


def remount_overview_viz_if_unmapped(gui: Any) -> bool:
    """If Video / Skeleton / Path lost their geometry manager, restore the row."""
    for attr in ("video_frame", "skel_frame", "overview_traj_panel"):
        widget = getattr(gui, attr, None)
        if widget is None:
            continue
        try:
            if not str(widget.winfo_manager() or ""):
                return apply_overview_three_panel_layout(gui)
        except tk.TclError:
            return apply_overview_three_panel_layout(gui)
    return True


def _place_summary_dock(gui: Any, section: tk.Misc, row: int) -> None:
    """Legacy summary dock under the viz — hidden on the clean Overview default."""
    from stablewalk.ui.theme import PAD_XS

    sidebar = getattr(gui, "sidebar", None)
    traj = getattr(gui, "overview_traj_panel", None)
    expanded = bool(getattr(gui, "_overview_summary_expanded", False))

    try:
        section.rowconfigure(row, weight=0, minsize=0)
    except tk.TclError:
        pass

    # Trajectory is a sibling column, never the below-fold summary strip.
    if traj is not None and not bool(getattr(gui, "_overview_traj_dock_visible", False)):
        _hide_overview_visual(traj)
    if sidebar is None:
        return
    if not expanded:
        _hide_overview_visual(sidebar)
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


def _grid_focused_viewer(
    section: tk.Misc,
    panel: tk.Misc,
    *,
    row: int,
    minsize: int,
) -> None:
    _col(section, 0, 1, minsize=minsize)
    _col(section, 1, 0)
    _col(section, 2, 0)
    try:
        section.rowconfigure(row, weight=1)
    except tk.TclError:
        pass
    try:
        panel.grid(
            in_=section,
            row=row,
            column=0,
            columnspan=3,
            sticky="nsew",
            padx=0,
            pady=0,
        )
    except tk.TclError:
        try:
            panel.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=0, pady=0)
        except tk.TclError:
            pass


def layout_panels_for_view_mode(gui: Any, mode: str) -> None:
    """Arrange Overview viewers for the selected mode.

    Split keeps Video | Skeleton | 3D Path in a resizable row.
    Video Only / Overlay / Skeleton Only / 3D Full focus one viewer so the
    remaining two do not steal presentation space.
    """
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        VIEW_MODE_OVERLAY,
        VIEW_MODE_RECONSTRUCTION_FULL,
        VIEW_MODE_SIDE_BY_SIDE,
        VIEW_MODE_SKELETON_ONLY,
        VIEW_MODE_VIDEO_ONLY,
    )
    from stablewalk.ui.tk.dashboard_sections import apply_overview_viz_grid_weights
    from stablewalk.ui.tk.overview_responsive import PANEL_MINWIDTH

    section = getattr(gui, "_section_visual", None) or getattr(gui, "_primary_viz_host", None)
    video = getattr(gui, "video_frame", None)
    skel = getattr(gui, "skel_frame", None)
    sidebar = getattr(gui, "sidebar", None)
    traj = getattr(gui, "overview_traj_panel", None)
    if section is None or video is None or skel is None:
        return

    row = int(getattr(gui, "_primary_viz_content_row", 0))
    playing = bool(getattr(gui, "playing", False))
    if playing and getattr(gui, "_overview_focus_mode", None) == mode:
        return

    gui._overview_focus_mode = mode
    show_traj = traj is not None
    gui._overview_traj_dock_visible = bool(
        show_traj
        and mode in {VIEW_MODE_SIDE_BY_SIDE, VIEW_MODE_RECONSTRUCTION_FULL}
    )

    if mode == VIEW_MODE_SIDE_BY_SIDE:
        apply_overview_three_panel_layout(gui)
        if playing:
            return
        refresh = getattr(gui, "_refresh_overview_trajectory_dock", None)
        if callable(refresh) and show_traj:
            try:
                refresh(force_draw=False)
            except Exception:
                pass
        _schedule_sync_visuals(gui, delay_ms=40)
        schedule = getattr(gui, "_schedule_overview_media_refit", None)
        if callable(schedule):
            try:
                schedule()
            except Exception:
                pass
        return

    if mode == VIEW_MODE_RECONSTRUCTION_FULL:
        # Keep the 3D Path in its build-time holder. Reparenting the LabelFrame
        # after clip canvases exist blanks the plot (same rule as Split).
        from stablewalk.ui.tk.analysis_split import MIN_W_PATH, split_add, split_clear

        paned = getattr(gui, "_overview_viz_split", None)
        pane_t = getattr(gui, "_overview_pane_path", None)
        if (
            _is_tk_widget(paned)
            and _is_tk_widget(pane_t)
            and traj is not None
        ):
            try:
                for widget in (video, skel, sidebar):
                    _hide_overview_visual(widget)
                split_clear(paned)
                split_add(paned, pane_t, minsize=MIN_W_PATH)
                paned.grid(row=row, column=0, columnspan=3, sticky="nsew")
                section.columnconfigure(0, weight=1, minsize=0)
                section.columnconfigure(1, weight=0, minsize=0)
                section.columnconfigure(2, weight=0, minsize=0)
                section.rowconfigure(row, weight=1)
                try:
                    traj.grid(in_=pane_t, row=0, column=0, sticky="nsew")
                except tk.TclError:
                    pass
                place_overview_under_viewer_chrome(
                    gui, section, chrome_row=row + 1, span=3
                )
                gui._overview_viz_layout = "focused"
                gui._overview_viz_stack_requested = False
                gui._overview_traj_dock_visible = True
                if not playing:
                    refresh = getattr(gui, "_refresh_overview_trajectory_dock", None)
                    if callable(refresh):
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
                return
            except tk.TclError:
                logger.exception("3D Full path mount failed")

    _clear_overview_viewer_split(gui)
    for widget in (video, skel, sidebar, traj):
        _hide_overview_visual(widget)

    apply_overview_viz_grid_weights(section, content_row=row)
    panel_min = PANEL_MINWIDTH
    chrome_span = 3
    try:
        if mode in (VIEW_MODE_VIDEO_ONLY, VIEW_MODE_OVERLAY):
            _grid_focused_viewer(section, video, row=row, minsize=panel_min)
            chrome_span = 1
        elif mode == VIEW_MODE_SKELETON_ONLY:
            _grid_focused_viewer(section, skel, row=row, minsize=panel_min)
            chrome_span = 1
        else:
            apply_overview_three_panel_layout(gui)
            return
        place_overview_under_viewer_chrome(
            gui, section, chrome_row=row + 1, span=chrome_span
        )
        gui._overview_viz_layout = "focused"
        gui._overview_viz_stack_requested = False
    except Exception:
        logger.exception("Overview panel remount failed; restoring side-by-side")
        gui._overview_focus_mode = VIEW_MODE_SIDE_BY_SIDE
        try:
            apply_overview_three_panel_layout(gui)
        except Exception:
            logger.exception("Overview panel emergency remount failed")

    if show_traj and mode in {
        VIEW_MODE_SIDE_BY_SIDE,
        VIEW_MODE_RECONSTRUCTION_FULL,
    }:
        refresh = getattr(gui, "_refresh_overview_trajectory_dock", None)
        if callable(refresh):
            try:
                refresh(force_draw=False)
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
    "COMPACT_SECONDARY_INFO_H",
    "DEFAULT_SECONDARY_INFO_H",
    "DEFAULT_SASH_FRACTIONS",
    "DEFAULT_TRAJ_SASH_FRACTIONS",
    "MIN_PANE_SIDE",
    "MIN_PANE_SKELETON",
    "MIN_PANE_VIDEO",
    "PANE_SIDE",
    "PANE_SKELETON",
    "PANE_VIDEO",
    "apply_overview_side_content",
    "apply_overview_three_panel_layout",
    "overview_viz_panels_mounted",
    "remount_overview_viz_if_unmapped",
    "apply_saved_sash_fractions",
    "bind_overview_viewer_split_sashes",
    "clear_sash_fraction_prefs",
    "install_overview_paned_host",
    "install_overview_workspace_split",
    "layout_panels_for_view_mode",
    "load_sash_fractions",
    "mount_overview_viewer_split",
    "mount_panel_in_pane",
    "place_overview_under_viewer_chrome",
    "reset_overview_layout",
    "sync_overview_info_scroll",
    "sync_overview_viewer_split_height",
    "_on_overview_split_release",
    "save_sash_fractions",
    "set_overview_panes_visible",
]
