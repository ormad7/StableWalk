"""Synchronized dashboard playhead — Visual3D / Vicon-style thin cursor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from matplotlib.axes import Axes

from stablewalk.ui.colors import MUTED, PANEL, PLAYHEAD, TEXT

_PLAYHEAD_CORE = PLAYHEAD
_CORE_WIDTH = 1.35
_CORE_ALPHA = 0.95

_ANIM_STEP = 0.14
_LABEL_FONTSIZE = 8.5
_VALUE_FONTSIZE = 8.5


@dataclass(frozen=True, slots=True)
class PlayheadState:
    """Current playback position shared across dashboard charts."""

    time_s: float
    frame_index: int
    list_pos: int | None = None
    anim_phase: float = 0.0
    animating: bool = False


def advance_playhead_anim_phase(phase: float, *, step: float = _ANIM_STEP) -> float:
    """Advance pulsing animation phase (0..1). Kept for API compatibility."""
    return (float(phase) + step) % 1.0


def playhead_pulse(anim_phase: float) -> tuple[float, float, float]:
    """Return (core_linewidth, core_alpha, glow_alpha) — static for lab appearance."""
    del anim_phase
    return _CORE_WIDTH, _CORE_ALPHA, 0.0


def format_playhead_label(state: PlayheadState) -> str:
    """Human-readable frame + timestamp beside the playhead."""
    return f"F{state.frame_index} · {state.time_s:.2f}s"


def draw_chart_playhead(
    ax: Axes,
    state: PlayheadState,
    *,
    show_label: bool = True,
    value_label: str | None = None,
    value_y: float | None = None,
    zorder: int = 20,
) -> None:
    """Draw a thin research-lab playhead (single line — no triangle cap / glow)."""
    previous = getattr(ax, "_stablewalk_playhead_artists", None)
    if previous:
        for artist in previous:
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError, NotImplementedError):
                    pass

    # Placeholder slots keep tuple length stable for update_chart_playhead.
    glow = ax.axvline(
        state.time_s,
        color=_PLAYHEAD_CORE,
        linewidth=_CORE_WIDTH,
        alpha=0.0,
        linestyle="-",
        zorder=zorder - 2,
    )
    halo = ax.axvline(
        state.time_s,
        color=MUTED,
        linewidth=_CORE_WIDTH,
        alpha=0.0,
        linestyle="-",
        zorder=zorder - 1,
    )
    core = ax.axvline(
        state.time_s,
        color=_PLAYHEAD_CORE,
        linewidth=_CORE_WIDTH,
        alpha=_CORE_ALPHA,
        linestyle="-",
        zorder=zorder,
    )

    # Empty scatter keeps artist-slot layout without colliding with HS markers.
    cap = ax.scatter([], [], s=1, alpha=0.0, zorder=zorder + 1)

    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1.0
    x0, x1 = ax.get_xlim()
    xr = x1 - x0 if x1 > x0 else 1.0
    label_x = state.time_s + xr * 0.008

    label = None
    if show_label:
        label_y = y0 + yr * 0.06
        label = ax.text(
            label_x,
            label_y,
            format_playhead_label(state),
            color=TEXT,
            fontsize=_LABEL_FONTSIZE,
            fontweight="medium",
            va="bottom",
            ha="left",
            clip_on=False,
            zorder=zorder + 2,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": PANEL,
                "edgecolor": MUTED,
                "alpha": 0.92,
                "linewidth": 0.55,
            },
        )

    value = None
    if value_label:
        vy = value_y if value_y is not None and math.isfinite(float(value_y)) else (y0 + y1) * 0.5
        vy = min(max(float(vy), y0 + yr * 0.12), y1 - yr * 0.12)
        value = ax.text(
            label_x,
            vy,
            value_label,
            color=TEXT,
            fontsize=_VALUE_FONTSIZE,
            fontweight="semibold",
            va="center",
            ha="left",
            clip_on=False,
            zorder=zorder + 3,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": PANEL,
                "edgecolor": MUTED,
                "alpha": 0.94,
                "linewidth": 0.55,
            },
        )

    ax._stablewalk_playhead_artists = (glow, halo, core, cap, label, value)  # type: ignore[attr-defined]


def update_chart_playhead(
    ax: Axes,
    state: PlayheadState,
    *,
    show_label: bool = True,
    value_label: str | None = None,
    value_y: float | None = None,
) -> bool:
    """Move existing playhead artists without rebuilding the chart."""
    artists = getattr(ax, "_stablewalk_playhead_artists", None)
    if not artists or len(artists) not in (5, 6):
        draw_chart_playhead(
            ax,
            state,
            show_label=show_label,
            value_label=value_label,
            value_y=value_y,
        )
        return False
    glow, halo, core, cap, label = artists[:5]
    value = artists[5] if len(artists) == 6 else None
    try:
        for line in (glow, halo, core):
            line.set_xdata([state.time_s, state.time_s])
        core.set_linewidth(_CORE_WIDTH)
        core.set_alpha(_CORE_ALPHA)
        y0, y1 = ax.get_ylim()
        yr = y1 - y0 if y1 > y0 else 1.0
        x0, x1 = ax.get_xlim()
        xr = x1 - x0 if x1 > x0 else 1.0
        label_x = state.time_s + xr * 0.008
        if label is not None:
            label.set_position((label_x, y0 + yr * 0.06))
            label.set_text(format_playhead_label(state))
            label.set_visible(show_label)
        if value is not None and value_label:
            vy = value_y if value_y is not None and math.isfinite(float(value_y)) else (y0 + y1) * 0.5
            vy = min(max(float(vy), y0 + yr * 0.12), y1 - yr * 0.12)
            value.set_position((label_x, vy))
            value.set_text(value_label)
            value.set_visible(True)
        elif value is not None:
            value.set_visible(False)
        del cap  # Slot reserved; no geometry to update.
        return True
    except (AttributeError, RuntimeError, ValueError):
        draw_chart_playhead(
            ax,
            state,
            show_label=show_label,
            value_label=value_label,
            value_y=value_y,
        )
        return False


def apply_playhead_to_axes(axes: Sequence[Axes], state: PlayheadState) -> None:
    """Draw the same synchronized playhead on every stacked time-series row."""
    for ax in axes:
        draw_chart_playhead(ax, state)


def update_playhead_on_axes(axes: Sequence[Axes], state: PlayheadState) -> bool:
    """Move synchronized playheads; return True when all artists were reused."""
    reused = True
    for ax in axes:
        artists = getattr(ax, "_stablewalk_playhead_artists", None)
        show_label = bool(
            artists and len(artists) >= 5 and artists[4] is not None
        )
        reused = update_chart_playhead(ax, state, show_label=show_label) and reused
    return reused


__all__ = [
    "PlayheadState",
    "advance_playhead_anim_phase",
    "apply_playhead_to_axes",
    "draw_chart_playhead",
    "format_playhead_label",
    "playhead_pulse",
    "update_chart_playhead",
    "update_playhead_on_axes",
]
