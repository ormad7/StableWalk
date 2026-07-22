"""Bottom status bar — continuous session, playback, and pipeline readout."""

from __future__ import annotations

import time
import tkinter as tk
from typing import Any

from stablewalk.ui.dof_selection import label_for_item
from stablewalk.ui.theme import (
    ACCENT,
    BORDER,
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
_STATUS_FIELDS: tuple[tuple[str, str], ...] = (
    ("frame", "Frame"),
    ("time", "Time"),
    ("fps", "FPS"),
    ("joint", "Joint"),
    ("speed", "Speed"),
    ("sampling", "Sample"),
    ("analysis", "Analysis"),
    ("opensim", "OpenSim"),
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

    gui._status_bar_labels: dict[str, tk.Label] = {}
    gui._status_bar_flash_text: str | None = None
    gui._status_bar_flash_until: float = 0.0

    for index, (key, title) in enumerate(_STATUS_FIELDS):
        if index > 0:
            tk.Frame(inner, bg=BORDER, width=1, highlightthickness=0).pack(
                side=tk.LEFT, fill=tk.Y, padx=PAD_SM - 2, pady=1
            )

        cell = tk.Frame(inner, bg=SURFACE, highlightthickness=0)
        cell.pack(side=tk.LEFT, padx=(0, PAD_XS // 2))

        tk.Label(
            cell,
            text=title,
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(side=tk.LEFT, padx=(0, PAD_XS))

        # Fixed widths keep Frame/Joint/Speed from visually colliding
        # (e.g. "Left Toe" + "ze" from a neighboring field → "Toeze").
        _VALUE_WIDTHS = {
            "frame": 10,
            "time": 16,
            "fps": 5,
            "joint": 14,
            "speed": 6,
            "sampling": 6,
            "analysis": 18,
            "opensim": 12,
        }
        value = tk.Label(
            cell,
            text="—",
            bg=SURFACE,
            fg=TEXT,
            font=FONT_MONO_SM,
            anchor="w",
            width=_VALUE_WIDTHS.get(key, 8),
        )
        value.pack(side=tk.LEFT)
        gui._status_bar_labels[key] = value

    gui._status_bar = bar
    gui.status = StatusMessageProxy(gui)
    update_dashboard_status_bar(gui)
    return bar


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
            analysis.configure(text=_truncate(text, 72), fg=ACCENT)
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
        lbl.configure(text=value, fg=fg)
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
    """Single playhead index (0-based) for HUD, status bar, and path labels."""
    player = getattr(gui, "skeleton_player", None)
    if player is None or getattr(player, "frame_count", 0) <= 0:
        return None
    total = max(int(player.frame_count), 1)
    try:
        frame_float = float(
            getattr(player.state, "frame_float", player.state.frame_index)
        )
    except Exception:
        try:
            frame_float = float(player.state.frame_index)
        except Exception:
            return None
    return max(0, min(int(round(frame_float)), total - 1))


def _current_frame_and_time(gui: Any) -> tuple[str, str]:
    player = getattr(gui, "skeleton_player", None)
    if player is None or getattr(player, "frame_count", 0) <= 0:
        return "—", "—"

    frame_idx = playhead_frame_0based(gui)
    if frame_idx is None:
        return "—", "—"
    total = max(int(player.frame_count), 1)
    # 1-based current / total count (lab-software convention).
    frame_text = f"{frame_idx + 1}/{total}"

    try:
        t = float(player.time_at_current())
        dur = float(player.duration_s)
        time_text = f"{t:.2f}s / {dur:.2f}s"
    except Exception:
        time_text = "—"
    return frame_text, time_text


def _selected_joint_label(gui: Any) -> str:
    active = None
    if hasattr(gui, "_active_dof_item_id"):
        try:
            active = gui._active_dof_item_id()
        except Exception:
            active = None
    if not active:
        return "None"
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


def _analysis_state(gui: Any) -> tuple[str, str]:
    flash = getattr(gui, "_status_bar_flash_text", None)
    until = float(getattr(gui, "_status_bar_flash_until", 0.0) or 0.0)
    if flash and time.monotonic() < until:
        return _truncate(str(flash), 42), ACCENT
    if flash and time.monotonic() >= until:
        gui._status_bar_flash_text = None

    if getattr(gui, "_presentation_mode", False):
        if getattr(gui, "playing", False):
            return "Demo · Playing", SUCCESS
        return "Demo · Ready", ACCENT

    analyzing = bool(getattr(gui, "_pipeline_callback", None)) or bool(
        getattr(gui, "_pending_video_load", None)
    )
    if analyzing:
        return "Analyzing", WARNING

    player = getattr(gui, "skeleton_player", None)
    has_session = player is not None and getattr(player, "frame_count", 0) > 0
    if not has_session and getattr(gui, "sequence", None) is None:
        return "Idle", MUTED

    if getattr(gui, "playing", False):
        return "Playing", SUCCESS
    if player is not None and getattr(player.state, "stopped", False):
        return "Stopped", MUTED
    if has_session:
        return "Paused", ACCENT
    return "Ready", TEXT


def _opensim_status_text(gui: Any) -> tuple[str, str]:
    if getattr(gui, "_presentation_mode", False):
        return "Demo", ACCENT

    override = getattr(gui, "_opensim_status_override", None)
    if override:
        return _truncate(str(override), 28), WARNING

    sdk = bool(getattr(gui, "_opensim_sdk_available", False))
    model_valid = bool(getattr(gui, "_opensim_model_valid", False))
    export_complete = bool(getattr(gui, "_opensim_export_completed", False))
    has_session = bool(
        getattr(gui, "sequence", None) or getattr(gui, "gait_motion", None)
    )

    compact = getattr(gui, "lbl_opensim_compact_ready", None)
    if compact is not None:
        try:
            text = str(compact.cget("text") or "").replace("OpenSim · ", "").strip()
            if text:
                color = SUCCESS if text.lower() == "ready" else (
                    WARNING if "partial" in text.lower() else MUTED
                )
                return _truncate(text, 28), color
        except tk.TclError:
            pass

    if sdk and model_valid:
        return "Ready", SUCCESS
    if sdk:
        return "Partial", WARNING
    if export_complete or has_session:
        return "Export only", ACCENT
    return "Unavailable", MUTED


def update_dashboard_status_bar(gui: Any) -> None:
    """Refresh all status-bar fields from the live GUI session state."""
    if not getattr(gui, "_status_bar_labels", None):
        return

    frame_text, time_text = _current_frame_and_time(gui)
    _set_field(gui, "frame", frame_text)
    _set_field(gui, "time", time_text)

    fps = _session_fps(gui)
    _set_field(gui, "fps", f"{fps:.1f}" if fps else "—")

    _set_field(gui, "joint", _truncate(_selected_joint_label(gui), 14))
    _set_field(gui, "speed", _truncate(_playback_speed_text(gui), 6))
    _set_field(gui, "sampling", _truncate(_sampling_text(gui), 6))

    analysis_text, analysis_fg = _analysis_state(gui)
    _set_field(gui, "analysis", _truncate(analysis_text, 18), fg=analysis_fg)

    opensim_text, opensim_fg = _opensim_status_text(gui)
    _set_field(gui, "opensim", opensim_text, fg=opensim_fg)


__all__ = [
    "StatusMessageProxy",
    "build_dashboard_status_bar",
    "playhead_frame_0based",
    "set_status_flash",
    "update_dashboard_status_bar",
]
