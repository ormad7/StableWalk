"""Bottom status bar — continuous session, playback, and pipeline readout."""

from __future__ import annotations

import time
import tkinter as tk
from typing import Any

from stablewalk.ui.dof_selection import label_for_item
from stablewalk.opensim_ik_status import resolve_opensim_status_pair_from_gui
from stablewalk.ui.theme import (
    ACCENT,
    BORDER,
    DANGER,
    FONT_MONO_SM,
    FONT_UI_XS,
    MUTED,
    PAD_SM,
    PAD_XS,
    STATUS_BAR_PAD,
    SUCCESS,
    SURFACE,
    TEXT,
    WARNING,
)

_FLASH_TTL_S = 4.0
# Visible lab chrome — timebase + selection always on (OpenSim / Nexus style).
_STATUS_FIELDS_VISIBLE: tuple[tuple[str, str], ...] = (
    ("frame", "Frame"),
    ("time", "Time"),
    ("fps", "FPS"),
    ("joint", "Joint"),
    ("analysis", "Analysis"),
    ("opensim", "OpenSim"),
)
_STATUS_FIELDS_HIDDEN: tuple[tuple[str, str], ...] = (
    ("speed", "Speed"),
    ("sampling", "Sample"),
)
_STATUS_FIELDS: tuple[tuple[str, str], ...] = (
    *_STATUS_FIELDS_VISIBLE,
    *_STATUS_FIELDS_HIDDEN,
)


class StatusMessageProxy:
    """Keeps legacy ``gui.status.configure(text=...)`` calls working."""

    def __init__(self, gui: Any) -> None:
        self._gui = gui

    def configure(self, **kwargs: Any) -> None:
        text = kwargs.get("text")
        if text is not None:
            set_status_flash(self._gui, str(text))
        # Ignore style-only updates from legacy callers.

    def cget(self, key: str) -> str:
        if key == "text":
            flash = getattr(self._gui, "_status_bar_flash_text", None)
            if flash:
                return str(flash)
            labels = getattr(self._gui, "_status_bar_labels", {})
            analysis = labels.get("analysis")
            if analysis is not None:
                try:
                    return str(analysis.cget("text"))
                except tk.TclError:
                    return ""
        return ""


def build_dashboard_status_bar(gui: Any) -> tk.Frame:
    """Install the persistent bottom status bar on the root window."""
    bar = tk.Frame(
        gui.root,
        bg=SURFACE,
        highlightthickness=1,
        highlightbackground=BORDER,
        highlightcolor=BORDER,
    )
    bar.pack(side=tk.BOTTOM, fill=tk.X)

    inner = tk.Frame(bar, bg=SURFACE, highlightthickness=0)
    inner.pack(fill=tk.X, padx=STATUS_BAR_PAD[0], pady=STATUS_BAR_PAD[1])
    inner.columnconfigure(0, weight=1)
    inner.columnconfigure(1, weight=0)

    fields_host = tk.Frame(inner, bg=SURFACE, highlightthickness=0)
    fields_host.grid(row=0, column=0, sticky="ew")
    gui._status_bar_fields_host = fields_host

    gui._status_bar_labels: dict[str, tk.Label] = {}
    gui._status_bar_flash_text: str | None = None
    gui._status_bar_flash_until: float = 0.0
    gui._status_bar_cells: dict[str, tk.Frame] = {}
    gui._status_bar_titles: dict[str, tk.Label] = {}
    gui._status_bar_full_values: dict[str, str] = {}

    # Hidden host keeps Frame/Time/FPS/… labels updated for tooltips & API,
    # without packing them into the visible status chrome.
    hidden = tk.Frame(bar, bg=SURFACE, highlightthickness=0)
    gui._status_bar_hidden = hidden

    for index, (key, title) in enumerate(_STATUS_FIELDS_VISIBLE):
        fields_host.columnconfigure(index, weight=1, minsize=0)
        cell = tk.Frame(fields_host, bg=SURFACE, highlightthickness=0)
        cell.grid(
            row=0,
            column=index,
            sticky="ew",
            padx=(0, PAD_SM if index < len(_STATUS_FIELDS_VISIBLE) - 1 else 0),
        )
        cell.columnconfigure(1, weight=1, minsize=0)
        gui._status_bar_cells[key] = cell

        title_fg = MUTED if key in ("fps", "frame", "time", "joint") else TEXT
        value_fg = MUTED if key in ("fps",) else TEXT
        title_label = tk.Label(
            cell,
            text=f"{title}:" if key == "opensim" else title,
            bg=SURFACE,
            fg=title_fg if key not in ("analysis", "opensim") else MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        title_label.grid(row=0, column=0, sticky="w", padx=(0, PAD_XS))
        gui._status_bar_titles[key] = title_label

        _VALUE_WIDTHS = {
            "frame": 10,
            "time": 16,
            "fps": 5,
            "joint": 14,
            "speed": 6,
            "sampling": 6,
            "analysis": 22,
            "opensim": 22,
        }
        value = tk.Label(
            cell,
            text="—",
            bg=SURFACE,
            fg=value_fg,
            font=FONT_MONO_SM,
            anchor="w",
            width=_VALUE_WIDTHS.get(key, 8),
        )
        value.grid(row=0, column=1, sticky="ew")
        gui._status_bar_labels[key] = value

    for key, title in _STATUS_FIELDS_HIDDEN:
        value = tk.Label(hidden, text="—", bg=SURFACE, fg=TEXT, font=FONT_MONO_SM)
        gui._status_bar_labels[key] = value

    # Compact tooltips on secondary fields; primary readout is always visible.
    from stablewalk.ui.theme import create_tooltip

    for key, title in _STATUS_FIELDS_VISIBLE:
        create_tooltip(
            gui._status_bar_labels[key],
            f"{title}: —",
            wraplength=320,
        )
    create_tooltip(
        gui._status_bar_labels["analysis"],
        "Session frame, time, and analysis state — synced with the transport bar.",
        wraplength=320,
    )
    create_tooltip(
        gui._status_bar_labels["opensim"],
        "OpenSim status details are not available yet.",
        wraplength=420,
    )

    from stablewalk.ui.scientific_labels import DISCLAIMER_RESEARCH, UNITS_COORD_LEGEND

    gui.lbl_status_science = tk.Label(
        inner,
        text="Research · pose estimation",
        bg=SURFACE,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    )
    gui.lbl_status_science.grid(row=0, column=1, sticky="e", padx=(PAD_SM, 0))
    create_tooltip(
        gui.lbl_status_science,
        f"{DISCLAIMER_RESEARCH}\n{UNITS_COORD_LEGEND}",
        wraplength=420,
    )

    gui._status_bar = bar
    gui._status_bar_inner = inner
    gui.status = StatusMessageProxy(gui)
    inner.bind(
        "<Configure>",
        lambda event: _reflow_status_bar(gui, width=int(event.width)),
        add="+",
    )
    update_dashboard_status_bar(gui)
    return bar


def _reflow_status_bar(gui: Any, *, width: int) -> None:
    """Allocate footer width without allowing neighboring fields to overlap."""
    cells = getattr(gui, "_status_bar_cells", None) or {}
    labels = getattr(gui, "_status_bar_labels", None) or {}
    science = getattr(gui, "lbl_status_science", None)
    width = max(0, int(width))

    if width >= 1050:
        visible = {"frame", "time", "fps", "joint", "analysis", "opensim"}
        widths = {"frame": 10, "time": 16, "fps": 5, "joint": 14, "analysis": 22, "opensim": 22}
        show_science = True
    elif width >= 760:
        visible = {"frame", "time", "joint", "analysis", "opensim"}
        widths = {"frame": 8, "time": 12, "joint": 10, "analysis": 14, "opensim": 18}
        show_science = False
    elif width >= 480:
        visible = {"frame", "time", "analysis", "opensim"}
        widths = {"frame": 7, "time": 10, "analysis": 10, "opensim": 16}
        show_science = False
    else:
        visible = {"analysis", "opensim"}
        widths = {"analysis": 9, "opensim": 16}
        show_science = False

    fields_host = getattr(gui, "_status_bar_fields_host", None)
    for index, (key, _title) in enumerate(_STATUS_FIELDS_VISIBLE):
        cell = cells.get(key)
        if cell is None:
            continue
        try:
            if fields_host is not None:
                fields_host.columnconfigure(
                    index,
                    weight=2 if key in visible and key in ("analysis", "opensim") else (
                        1 if key in visible else 0
                    ),
                    minsize=0,
                )
            if key in visible:
                cell.grid()
                label = labels.get(key)
                if label is not None:
                    label.configure(width=widths.get(key, 8))
                    full = (getattr(gui, "_status_bar_full_values", None) or {}).get(
                        key
                    )
                    if full is not None:
                        _set_field(gui, key, full, fg=str(label.cget("fg")))
            else:
                cell.grid_remove()
        except tk.TclError:
            continue
    if science is not None:
        try:
            if show_science:
                science.grid()
            else:
                science.grid_remove()
        except tk.TclError:
            pass


def set_status_flash(gui: Any, text: str) -> None:
    """Show a short-lived analysis-state message (legacy status updates)."""
    gui._status_bar_flash_text = text
    gui._status_bar_flash_until = time.monotonic() + _FLASH_TTL_S
    labels = getattr(gui, "_status_bar_labels", None)
    if not labels:
        return
    analysis = labels.get("analysis")
    if analysis is not None:
        try:
            _set_field(gui, "analysis", text, fg=ACCENT)
            analysis._tooltip_text = text  # type: ignore[attr-defined]
        except tk.TclError:
            pass


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"


def _set_field(gui: Any, key: str, value: str, *, fg: str = TEXT) -> None:
    labels = getattr(gui, "_status_bar_labels", None)
    if not labels:
        return
    lbl = labels.get(key)
    if lbl is None:
        return
    try:
        full_values = getattr(gui, "_status_bar_full_values", None)
        if full_values is None:
            full_values = {}
            gui._status_bar_full_values = full_values
        full_values[key] = str(value)
        try:
            width = int(lbl.cget("width") or 0)
        except (TypeError, ValueError):
            width = 0
        display = _truncate(str(value), width) if width > 1 else str(value)
        lbl._tooltip_text = f"{key.title()}: {value}"  # type: ignore[attr-defined]
        if str(lbl.cget("text")) == display and str(lbl.cget("fg")) == fg:
            return
        lbl.configure(text=display, fg=fg)
    except tk.TclError:
        pass


def _session_fps(gui: Any) -> float | None:
    sequence = getattr(gui, "sequence", None)
    if sequence is not None and getattr(sequence, "fps", 0):
        return float(sequence.fps)
    recording = getattr(gui, "gait_motion", None)
    if recording is not None and getattr(recording, "fps", 0):
        return float(recording.fps)
    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        try:
            return float(player.fps)
        except Exception:
            return None
    return None


def playhead_frame_0based(gui: Any) -> int | None:
    """Single playhead index (0-based) for HUD, status bar, and path labels.

    Delegates to the authoritative :func:`read_playback_state` so every surface
    shares the same truncate rule as ``SkeletonPlayer.state.frame_index``.
    """
    from stablewalk.ui.playback_state import read_playback_state

    state = read_playback_state(gui)
    if state is None or state.total_frames <= 0:
        return None
    return int(state.current_frame_index)


def _current_frame_and_time(gui: Any, state: Any | None = None) -> tuple[str, str]:
    from stablewalk.ui.playback_state import committed_playback_state

    resolved = state if state is not None else committed_playback_state(gui)
    if resolved is None or getattr(resolved, "total_frames", 0) <= 0:
        return "—", "—"
    return resolved.format_frame_label(), resolved.format_time_label()


def _selected_joint_label(gui: Any) -> str:
    active = None
    if hasattr(gui, "_active_dof_item_id"):
        try:
            active = gui._active_dof_item_id()
        except Exception:
            active = None
    if not active:
        return "—"
    try:
        return label_for_item(active)
    except Exception:
        return str(active)


def _playback_speed_text(gui: Any) -> str:
    speed = getattr(gui, "play_speed", None)
    if speed is None and hasattr(gui, "speed_var"):
        try:
            speed = float(gui.speed_var.get())
        except Exception:
            speed = 1.0
    try:
        return f"{float(speed):.2f}×"
    except Exception:
        return "1.00×"


def _sampling_text(gui: Any) -> str:
    if hasattr(gui, "refresh_var"):
        try:
            raw = str(gui.refresh_var.get() or "").strip()
            if not raw:
                return "—"
            # Avoid "0.5 s" next to a title that already reads as seconds.
            lowered = raw.lower().replace("sec", "s")
            if lowered.endswith(" s"):
                raw = raw[:-2].rstrip() + "s"
            elif lowered.endswith(" seconds"):
                raw = raw[: -len(" seconds")].rstrip() + "s"
            return raw
        except Exception:
            pass
    return "—"


def _analysis_in_progress(gui: Any) -> bool:
    """True while pose/session analysis is running — not during playback."""
    if getattr(gui, "_pipeline_callback", None) or getattr(gui, "_pending_video_load", None):
        return True
    if getattr(gui, "_analysis_compute_active", False) or getattr(gui, "_analysis_running", False):
        return True
    if getattr(gui, "_processing", False) or getattr(gui, "_pipeline_running", False):
        return True
    try:
        from stablewalk.ui.tk.analyze_progress import analyze_progress_active

        if analyze_progress_active(gui):
            return True
    except Exception:
        pass
    return False


def _analysis_state(gui: Any) -> tuple[str, str]:
    flash = getattr(gui, "_status_bar_flash_text", None)
    until = float(getattr(gui, "_status_bar_flash_until", 0.0) or 0.0)
    if flash and time.monotonic() < until:
        return _truncate(str(flash), 42), ACCENT
    if flash and time.monotonic() >= until:
        gui._status_bar_flash_text = None

    if getattr(gui, "_presentation_mode", False):
        if getattr(gui, "playing", False):
            return "Example · Playing", SUCCESS
        return "Example · Ready", ACCENT

    if _analysis_in_progress(gui):
        from stablewalk.ui.analysis_mode import MODE_FULL, MODE_PREVIEW, format_analysis_mode_label

        mode = str(getattr(gui, "_analysis_mode", None) or "").lower()
        cap = getattr(gui, "_analysis_max_frames", None)
        if mode == MODE_PREVIEW:
            return f"Analyzing {format_analysis_mode_label(mode, cap)}", WARNING
        if mode == MODE_FULL:
            return "Analyzing full source", WARNING
        return "Analyzing", WARNING

    player = getattr(gui, "skeleton_player", None)
    has_session = player is not None and getattr(player, "frame_count", 0) > 0
    if not has_session and getattr(gui, "sequence", None) is None:
        return "Idle", MUTED

    from stablewalk.ui.analysis_mode import format_analysis_status

    mode = getattr(gui, "_analysis_mode", None)
    playing = bool(getattr(gui, "playing", False))
    stopped = player is not None and getattr(player.state, "stopped", False)
    if playing or has_session:
        if stopped and not playing:
            key = str(mode or "").lower()
            prefix = (
                "Preview" if key == "preview" else "Full analysis" if key == "full" else "Playback"
            )
            return f"{prefix} · Stopped", MUTED
        paused = (not playing) and has_session
        text = format_analysis_status(mode, playing=playing, paused=paused)
        return text, SUCCESS if playing else ACCENT
    return "Ready", TEXT


_OPENSIM_FOOTER_FG = {
    "Complete": SUCCESS,
    "Export Complete · IK unavailable": WARNING,
    "Export Complete · IK Running": ACCENT,
    "Export Complete · IK Failed": DANGER,
    "Complete · IK Partial": WARNING,
    "Complete · IK Running": ACCENT,
    "Complete · IK Failed": DANGER,
    "IK Running": ACCENT,
    "IK Failed": DANGER,
    "IK Complete": SUCCESS,
    "Partial": WARNING,
    "Not Started": MUTED,
}


def _opensim_status_text(gui: Any) -> tuple[str, str]:
    """Footer OpenSim field — pipeline and IK named separately when they differ."""
    pair = resolve_opensim_status_pair_from_gui(gui)
    return pair.short_label, _OPENSIM_FOOTER_FG.get(pair.short_label, WARNING)


def _opensim_tooltip_text(gui: Any, short_status: str) -> str:
    """Full OpenSim status for hover without expanding the footer."""
    pair = resolve_opensim_status_pair_from_gui(gui)
    return pair.tooltip or f"OpenSim: {short_status}"


def update_dashboard_status_bar(gui: Any, state: Any | None = None) -> None:
    """Refresh all status-bar fields from the committed playback state."""
    if not getattr(gui, "_status_bar_labels", None):
        return

    frame_text, time_text = _current_frame_and_time(gui, state)
    resolved = state
    if resolved is None:
        from stablewalk.ui.playback_state import committed_playback_state

        resolved = committed_playback_state(gui)
    if resolved is not None:
        try:
            gui._status_bar_list_pos = int(resolved.current_frame_index)
        except Exception:
            pass
    _set_field(gui, "frame", frame_text)
    _set_field(gui, "time", time_text)

    fps = _session_fps(gui)
    _set_field(gui, "fps", f"{fps:.1f}" if fps else "—")

    _set_field(gui, "joint", _selected_joint_label(gui))
    _set_field(gui, "speed", _playback_speed_text(gui))
    _set_field(gui, "sampling", _sampling_text(gui))

    analysis_text, analysis_fg = _analysis_state(gui)
    _set_field(gui, "analysis", analysis_text, fg=analysis_fg)

    opensim_text, opensim_fg = _opensim_status_text(gui)
    _set_field(gui, "opensim", opensim_text, fg=opensim_fg)

    # Keep Analysis tooltip populated with the hidden session fields.
    tip_lines = [
        f"Analysis: {getattr(gui, '_status_bar_flash_text', None) or analysis_text}",
        f"Frame: {frame_text}",
        f"Time: {time_text}",
        f"FPS: {fps:.1f}" if fps else "FPS: —",
        f"Joint: {_selected_joint_label(gui)}",
        f"Speed: {_playback_speed_text(gui)}",
        f"Sample: {_sampling_text(gui)}",
    ]
    analysis_tip = "\n".join(tip_lines)
    analysis_lbl = gui._status_bar_labels.get("analysis")
    opensim_lbl = gui._status_bar_labels.get("opensim")
    try:
        if analysis_lbl is not None:
            analysis_lbl._tooltip_text = analysis_tip  # type: ignore[attr-defined]
        if opensim_lbl is not None:
            opensim_lbl._tooltip_text = _opensim_tooltip_text(  # type: ignore[attr-defined]
                gui, opensim_text
            )
    except Exception:
        pass


__all__ = [
    "StatusMessageProxy",
    "build_dashboard_status_bar",
    "playhead_frame_0based",
    "set_status_flash",
    "update_dashboard_status_bar",
]
