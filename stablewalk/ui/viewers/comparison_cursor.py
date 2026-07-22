"""Application-wide synchronized comparison cursor (shared playhead).

One instant in time drives video, skeleton, charts, 3D trajectory, and gait events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from stablewalk.ui.viewers.chart_playhead import (
    PlayheadState,
    advance_playhead_anim_phase,
    draw_chart_playhead,
    format_playhead_label,
)


_ACTIVE_EVENT_FRAME_TOLERANCE = 1
_ACTIVE_EVENT_TIME_TOLERANCE_S = 0.06


@dataclass(frozen=True, slots=True)
class ComparisonCursor:
    """Canonical playback instant shared by every visualization."""

    frame_float: float
    frame_index: int
    list_pos: int
    time_s: float
    anim_phase: float = 0.0
    playing: bool = False
    cycle_percent: float | None = None
    active_gait_events: tuple[str, ...] = ()

    def to_playhead_state(self) -> PlayheadState:
        return PlayheadState(
            time_s=self.time_s,
            frame_index=self.frame_index,
            list_pos=self.list_pos,
            anim_phase=self.anim_phase,
            animating=self.playing,
        )

    def label(self) -> str:
        return format_playhead_label(self.to_playhead_state())

    def summary_line(self) -> str:
        base = f"Comparison cursor · frame {self.frame_index} · {self.time_s:.2f}s"
        if self.cycle_percent is not None:
            base += f" · {self.cycle_percent:.0f}% gait cycle"
        if self.active_gait_events:
            names = ", ".join(_short_event_name(e) for e in self.active_gait_events[:3])
            base += f" · {names}"
        return base


def _short_event_name(event_type: str) -> str:
    mapping = {
        "left_heel_strike": "L HS",
        "right_heel_strike": "R HS",
        "left_toe_off": "L TO",
        "right_toe_off": "R TO",
    }
    return mapping.get(event_type, event_type.replace("_", " "))


def estimate_gait_cycle_percent(
    gait_cycle: Any | None,
    *,
    time_s: float,
    frame_index: int,
) -> float | None:
    """Estimate 0-100% within the current left-heel-strike stride, when possible."""
    if gait_cycle is None:
        return None
    events = list(getattr(gait_cycle, "events", None) or [])
    hs = sorted(
        (e for e in events if getattr(e, "event_type", "") == "left_heel_strike"),
        key=lambda e: float(e.time_s),
    )
    if len(hs) < 2:
        hs = sorted(
            (e for e in events if str(getattr(e, "event_type", "")).endswith("_heel_strike")),
            key=lambda e: float(e.time_s),
        )
    if len(hs) < 2:
        return None

    t = float(time_s)
    for i in range(len(hs) - 1):
        t0 = float(hs[i].time_s)
        t1 = float(hs[i + 1].time_s)
        if t0 <= t <= t1 and t1 > t0:
            return max(0.0, min(100.0, 100.0 * (t - t0) / (t1 - t0)))
    if t < float(hs[0].time_s):
        return 0.0
    return 100.0


def active_gait_event_types(
    gait_cycle: Any | None,
    *,
    frame_index: int,
    time_s: float,
    frame_events: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Gait events that correspond to the comparison cursor's instant."""
    found: list[str] = []
    if frame_events:
        for name in frame_events:
            key = str(name)
            if key and key not in found:
                found.append(key)

    events = list(getattr(gait_cycle, "events", None) or []) if gait_cycle is not None else []
    for ev in events:
        try:
            fi = int(ev.frame_index)
            ts = float(ev.time_s)
        except (AttributeError, TypeError, ValueError):
            continue
        if abs(fi - int(frame_index)) <= _ACTIVE_EVENT_FRAME_TOLERANCE or abs(
            ts - float(time_s)
        ) <= _ACTIVE_EVENT_TIME_TOLERANCE_S:
            et = str(getattr(ev, "event_type", "") or "")
            if et and et not in found:
                found.append(et)
    return tuple(found)


def snapshot_comparison_cursor(
    gui: Any,
    *,
    snap: Any | None = None,
    advance_anim: bool = False,
) -> ComparisonCursor:
    """Build the shared cursor from the live player / session state."""
    frame_float = float(getattr(gui, "_playback_pos", 0.0) or 0.0)
    list_pos = int(getattr(gui, "current_pos", 0) or 0)
    player = getattr(gui, "skeleton_player", None)
    if player is not None:
        try:
            frame_float = float(player.state.frame_float)
            list_pos = max(0, int(getattr(player, "current_index", list_pos) or list_pos))
        except Exception:
            pass

    pose_indices = list(getattr(gui, "pose_indices", None) or [])
    if pose_indices:
        list_pos = max(0, min(list_pos, len(pose_indices) - 1))

    frame_index = 0
    if snap is not None:
        try:
            frame_index = int(snap.frame_index)
        except (TypeError, ValueError):
            frame_index = 0
    elif pose_indices and 0 <= list_pos < len(pose_indices):
        frame_index = int(pose_indices[list_pos])

    time_s = 0.0
    frame_events: list[str] = []
    sequence = getattr(gui, "sequence", None)
    if sequence is not None:
        pf = next(
            (f for f in sequence.frames if f.frame_index == frame_index),
            None,
        )
        if pf is not None:
            time_s = float(pf.timestamp_s)
            frame_events = list(getattr(pf, "gait_events", None) or [])

    playing = bool(getattr(gui, "playing", False))
    if player is not None:
        try:
            playing = bool(player.state.playing)
        except Exception:
            pass

    phase = float(getattr(gui, "_playhead_anim_phase", 0.0) or 0.0)
    if advance_anim and playing:
        phase = advance_playhead_anim_phase(phase)
        try:
            gui._playhead_anim_phase = phase
        except Exception:
            pass

    gait_cycle = getattr(gui, "_gait_cycle", None)
    return ComparisonCursor(
        frame_float=frame_float,
        frame_index=frame_index,
        list_pos=list_pos,
        time_s=time_s,
        anim_phase=phase,
        playing=playing,
        cycle_percent=estimate_gait_cycle_percent(
            gait_cycle, time_s=time_s, frame_index=frame_index
        ),
        active_gait_events=active_gait_event_types(
            gait_cycle,
            frame_index=frame_index,
            time_s=time_s,
            frame_events=frame_events,
        ),
    )


def draw_cycle_percent_playhead(ax: Any, cursor: ComparisonCursor) -> None:
    """Playhead on gait-cycle-% charts (x = 0-100)."""
    if cursor.cycle_percent is None:
        return
    state = PlayheadState(
        time_s=float(cursor.cycle_percent),
        frame_index=cursor.frame_index,
        list_pos=cursor.list_pos,
        anim_phase=cursor.anim_phase,
        animating=cursor.playing,
    )
    draw_chart_playhead(ax, state, show_label=True)


def highlight_gait_events_at_cursor(
    ax: Any,
    events: Iterable[Any],
    cursor: ComparisonCursor,
    *,
    y_lo: float,
    y_hi: float,
) -> None:
    """Emphasize gait-event markers that match the comparison cursor."""
    from matplotlib.axes import Axes

    from stablewalk.ui.colors import COM, PLAYHEAD, TEXT

    if not isinstance(ax, Axes):
        return
    active = set(cursor.active_gait_events)
    if not active:
        for ev in events:
            try:
                if abs(float(ev.time_s) - cursor.time_s) <= _ACTIVE_EVENT_TIME_TOLERANCE_S:
                    active.add(str(ev.event_type))
            except (AttributeError, TypeError, ValueError):
                continue
    if not active:
        return

    marker_y = y_hi - (y_hi - y_lo) * 0.08
    for ev in events:
        et = str(getattr(ev, "event_type", "") or "")
        if et not in active:
            continue
        try:
            t = float(ev.time_s)
        except (TypeError, ValueError):
            continue
        ax.axvline(
            t,
            color=PLAYHEAD,
            linestyle="-",
            linewidth=2.4,
            alpha=0.95,
            zorder=21,
        )
        ax.scatter(
            [t],
            [marker_y],
            marker="D",
            s=36,
            color=COM,
            edgecolors=TEXT,
            linewidths=0.6,
            zorder=22,
            clip_on=False,
        )
        ax.text(
            t,
            marker_y,
            f" {_short_event_name(et)} ",
            color=TEXT,
            fontsize=7,
            fontweight="bold",
            va="bottom",
            ha="center",
            zorder=23,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "#1a1f2a",
                "edgecolor": COM,
                "alpha": 0.92,
                "linewidth": 0.8,
            },
        )


def format_video_cursor_badge(cursor: ComparisonCursor) -> str:
    """Compact on-video overlay text."""
    parts = [f"F{cursor.frame_index}", f"{cursor.time_s:.2f}s"]
    if cursor.active_gait_events:
        parts.append(_short_event_name(cursor.active_gait_events[0]))
    return "  ·  ".join(parts)


__all__ = [
    "ComparisonCursor",
    "active_gait_event_types",
    "draw_cycle_percent_playhead",
    "estimate_gait_cycle_percent",
    "format_video_cursor_badge",
    "highlight_gait_events_at_cursor",
    "snapshot_comparison_cursor",
]
