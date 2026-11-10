"""Synchronized dashboard playhead — Visual3D / Vicon-style thin cursor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes

from stablewalk.ui.colors import MUTED, PANEL, PLAYHEAD, TEXT

_PLAYHEAD_CORE = PLAYHEAD
_CORE_WIDTH = 1.55
_CORE_ALPHA = 0.96

_ANIM_STEP = 0.14
_LABEL_FONTSIZE = 11.0
_VALUE_FONTSIZE = 11.0

# Axes attr: (times_s, values, label_prefix, unit) for live value tooltips.
PLAYHEAD_VALUE_SOURCE_ATTR = "_stablewalk_playhead_value_source"


@dataclass(frozen=True, slots=True)
class PlayheadState:
    """Playback-linked plot cursor shared across dashboard charts.

    ``frame_index`` is the 1-based display frame shown as ``F{n}`` — the same
    number Live Data and the transport use. ``list_pos`` is the 0-based
    playback list index. Hover inspection uses a separate dotted cursor.
    """

    time_s: float
    frame_index: int
    list_pos: int | None = None
    anim_phase: float = 0.0
    animating: bool = False


def playhead_from_playback(
    state: Any,
    *,
    time_s: float | None = None,
    anim_phase: float = 0.0,
    animating: bool | None = None,
) -> PlayheadState:
    """Build the playback-linked plot cursor from canonical PlaybackState."""
    display = int(getattr(state, "display_frame_number", 0) or 0)
    list_pos = int(getattr(state, "current_frame_index", 0) or 0)
    t = float(state.current_time_seconds if time_s is None else time_s)
    playing = bool(state.playing) if animating is None else bool(animating)
    return PlayheadState(
        time_s=t,
        frame_index=display,
        list_pos=list_pos,
        anim_phase=float(anim_phase),
        animating=playing,
    )


def playhead_from_gui(
    gui: Any,
    *,
    time_s: float | None = None,
    anim_phase: float = 0.0,
    animating: bool | None = None,
) -> PlayheadState | None:
    """Read committed playback and return the linked plot cursor, if any.

    An empty skeleton-player stub (no ``frame_count``) is ignored so callers
    can fall back to their own index mapping. A missing player still uses the
    pose-index playback path.
    """
    from stablewalk.ui.playback_state import PlaybackState, committed_playback_state

    try:
        stored = getattr(gui, "_committed_playback", None)
        if isinstance(stored, PlaybackState):
            return playhead_from_playback(
                stored, time_s=time_s, anim_phase=anim_phase, animating=animating
            )
        player = getattr(gui, "skeleton_player", None)
        if player is not None and getattr(player, "frame_count", 0) <= 0:
            return None
        if player is not None and not hasattr(player, "state"):
            return None
        state = committed_playback_state(gui)
        if state is None:
            return None
        return playhead_from_playback(
            state,
            time_s=time_s,
            anim_phase=anim_phase,
            animating=animating,
        )
    except Exception:
        return None


def advance_playhead_anim_phase(phase: float, *, step: float = _ANIM_STEP) -> float:
    """Advance pulsing animation phase (0..1). Kept for API compatibility."""
    return (float(phase) + step) % 1.0


def playhead_pulse(
    anim_phase: float, *, animating: bool = False
) -> tuple[float, float, float]:
    """Return (core_linewidth, core_alpha, glow_alpha).

    Static when idle; soft breathe during playback for smooth cursor presence.
    """
    if not animating:
        return _CORE_WIDTH, _CORE_ALPHA, 0.0
    t = 0.5 + 0.5 * math.sin(float(anim_phase) * 2.0 * math.pi)
    return (
        _CORE_WIDTH + 0.35 * t,
        min(1.0, _CORE_ALPHA + 0.02 * t),
        0.10 + 0.12 * t,
    )


def format_playhead_label(state: PlayheadState, *, x_is_percent: bool = False) -> str:
    """Compact timestamp (or gait-cycle %) + frame beside the playhead."""
    if x_is_percent:
        return f"{state.time_s:.0f}% · F{int(state.frame_index)}"
    return f"{state.time_s:.2f}s · F{int(state.frame_index)}"


def _estimate_callout_size_px(text: str, *, dpi: float = 100.0) -> tuple[float, float]:
    """Approximate stamp / current-value bbox in display pixels."""
    chars = max(1, len(text) + 2)
    width_pt = chars * 6.6 + 14.0
    height_pt = _VALUE_FONTSIZE + 10.0
    px_per_pt = max(1.0, float(dpi) / 72.0)
    return width_pt * px_per_pt, height_pt * px_per_pt


def _figure_clip_box(ax: Axes, *, pad: float = 6.0) -> tuple[float, float, float, float] | None:
    """Hard clip box in display pixels — the figure edge, not just the axes."""
    fig = getattr(ax, "figure", None)
    if fig is None:
        return None
    try:
        fig_bbox = fig.bbox
        x0 = float(fig_bbox.x0) + pad
        y0 = float(fig_bbox.y0) + pad
        x1 = float(fig_bbox.x1) - pad
        y1 = float(fig_bbox.y1) - pad
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    except Exception:
        return None


def edge_aware_callout_placement(
    ax: Axes,
    x: float,
    y: float,
    *,
    text: str = "",
) -> tuple[float, float, str, str]:
    """Keep a data-coordinate callout fully inside the figure.

    If there is not enough room to the right of ``x``, place the text to the
    left. Vertical placement is clamped into the plot band. Pixel extents are
    used when a renderer exists so long labels never paint past the canvas.
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr = (x1 - x0) if x1 > x0 else 1.0
    yr = (y1 - y0) if y1 > y0 else 1.0
    need = min(0.55, max(0.18, (len(text) + 8) * 0.013))
    ha = "left"
    lx = float(x) + xr * 0.012
    if (x1 - float(x)) < xr * need:
        ha = "right"
        lx = float(x) - xr * 0.012

    dpi = float(getattr(getattr(ax, "figure", None), "dpi", 100.0) or 100.0)
    tw_px, th_px = _estimate_callout_size_px(text, dpi=dpi)
    clip = _figure_clip_box(ax)
    if clip is not None:
        clip_x0, clip_y0, clip_x1, clip_y1 = clip
        try:
            disp = ax.transData.transform((float(x), float(y)))
            inv = ax.transData.inverted()
            if ha == "left" and disp[0] + tw_px > clip_x1:
                ha = "right"
                lx = float(x) - xr * 0.012
            elif ha == "right" and disp[0] - tw_px < clip_x0:
                if (disp[0] - clip_x0) >= (clip_x1 - disp[0]):
                    ha = "right"
                    lx = float(x) - xr * 0.012
                else:
                    ha = "left"
                    lx = float(x) + xr * 0.012
            disp_l = ax.transData.transform((lx, float(y)))
            if ha == "left":
                overflow = (disp_l[0] + tw_px) - clip_x1
                if overflow > 0:
                    lx = float(inv.transform((disp_l[0] - overflow, disp_l[1]))[0])
            else:
                overflow = clip_x0 - (disp_l[0] - tw_px)
                if overflow > 0:
                    lx = float(inv.transform((disp_l[0] + overflow, disp_l[1]))[0])
        except Exception:
            pass

    if ha == "left":
        lx = min(max(lx, x0 + xr * 0.008), x1 - xr * 0.02)
    else:
        lx = min(max(lx, x0 + xr * 0.02), x1 - xr * 0.008)
    va = "center"
    ly = float(y)
    if ly > y1 - yr * 0.16:
        va = "top"
        ly = y1 - yr * 0.06
    elif ly < y0 + yr * 0.16:
        va = "bottom"
        ly = y0 + yr * 0.06
    ly = min(max(ly, y0 + yr * 0.06), y1 - yr * 0.06)
    if clip is not None:
        try:
            disp_y = ax.transData.transform((lx, ly))
            inv = ax.transData.inverted()
            if va == "top" and disp_y[1] > clip_y1:
                ly = float(inv.transform((disp_y[0], clip_y1))[1])
            elif va == "bottom" and disp_y[1] + th_px > clip_y1:
                ly = float(inv.transform((disp_y[0], clip_y1 - th_px))[1])
            elif va == "center":
                half = th_px * 0.5
                if disp_y[1] + half > clip_y1:
                    ly = float(inv.transform((disp_y[0], clip_y1 - half))[1])
                elif disp_y[1] - half < clip_y0:
                    ly = float(inv.transform((disp_y[0], clip_y0 + half))[1])
        except Exception:
            pass
    return lx, ly, ha, va


def set_playhead_value_source(
    ax: Axes,
    times: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    prefix: str,
    unit: str = "",
) -> None:
    """Cache the plotted series so the playhead marker samples displayed Y."""
    setattr(
        ax,
        PLAYHEAD_VALUE_SOURCE_ATTR,
        (
            np.asarray(times, dtype=float),
            np.asarray(values, dtype=float),
            str(prefix),
            str(unit or ""),
        ),
    )


def sample_displayed_series(
    times: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    time_s: float,
) -> float | None:
    """Nearest finite displayed sample at the canonical timestamp, or None.

    Missing / NaN samples stay gaps: the marker is hidden instead of
    interpolating across them or falling back to a fake Y.
    """
    times_a = np.asarray(times, dtype=float)
    values_a = np.asarray(values, dtype=float)
    if times_a.size == 0 or values_a.size != times_a.size:
        return None
    finite_t = np.isfinite(times_a)
    if not np.any(finite_t):
        return None
    t_idx = np.flatnonzero(finite_t)
    t = times_a[t_idx]
    query = float(time_s)
    local = int(np.argmin(np.abs(t - query)))
    orig = int(t_idx[local])
    sample_t = float(times_a[orig])
    sample_v = float(values_a[orig])
    if t.size >= 2:
        if local == 0:
            dt = abs(float(t[1] - t[0]))
        elif local == t.size - 1:
            dt = abs(float(t[-1] - t[-2]))
        else:
            dt = 0.5 * (
                abs(float(t[local] - t[local - 1]))
                + abs(float(t[local + 1] - t[local]))
            )
        half = max(0.5 * dt, 1e-12)
    else:
        half = 1e-6
    if abs(sample_t - query) > half:
        return None
    if not math.isfinite(sample_v):
        return None
    return sample_v


def resolve_playhead_value(
    ax: Axes, time_s: float
) -> tuple[str | None, float | None]:
    """Sample the live value tooltip from the series cached on the axes."""
    src = getattr(ax, PLAYHEAD_VALUE_SOURCE_ATTR, None)
    if not isinstance(src, (tuple, list)) or len(src) < 3:
        return None, None
    try:
        times = np.asarray(src[0], dtype=float)
        values = np.asarray(src[1], dtype=float)
        prefix = str(src[2])
        unit = str(src[3]) if len(src) > 3 and src[3] else ""
    except Exception:
        return None, None
    if times.size < 1 or values.size != times.size:
        return None, None
    value_y = sample_displayed_series(times, values, time_s)
    if value_y is None:
        return None, None
    if unit:
        label = f"{prefix}: {value_y:.2f} {unit}"
    else:
        label = f"{prefix}: {value_y:.2f}"
    return label, value_y


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    try:
        y = float(value)
    except (TypeError, ValueError):
        return None
    return y if math.isfinite(y) else None


def _resolve_marker_y(
    ax: Axes,
    time_s: float,
    *,
    value_y: float | None,
    value_label: str | None,
) -> tuple[str | None, float | None]:
    """Prefer the caller Y, else the displayed series at ``time_s``."""
    series_label, series_y = resolve_playhead_value(ax, time_s)
    marker_y = _finite_or_none(value_y)
    if marker_y is None:
        marker_y = series_y
    label = value_label if value_label is not None else series_label
    return label, marker_y


def draw_chart_playhead(
    ax: Axes,
    state: PlayheadState,
    *,
    show_label: bool = True,
    value_label: str | None = None,
    value_y: float | None = None,
    zorder: int = 20,
) -> None:
    """Draw a thin research-lab playhead with optional live value callout."""
    previous = getattr(ax, "_stablewalk_playhead_artists", None)
    if previous:
        for artist in previous:
            if artist is not None:
                try:
                    artist.remove()
                except (ValueError, AttributeError, NotImplementedError):
                    pass

    lw, core_a, glow_a = playhead_pulse(
        state.anim_phase, animating=bool(state.animating)
    )

    glow = ax.axvline(
        state.time_s,
        color=_PLAYHEAD_CORE,
        linewidth=lw + 2.2,
        alpha=glow_a,
        linestyle="-",
        zorder=zorder - 2,
        animated=True,
    )
    halo = ax.axvline(
        state.time_s,
        color=MUTED,
        linewidth=lw + 0.8,
        alpha=max(0.0, glow_a * 0.45),
        linestyle="-",
        zorder=zorder - 1,
        animated=True,
    )
    core = ax.axvline(
        state.time_s,
        color=_PLAYHEAD_CORE,
        linewidth=lw,
        alpha=core_a,
        linestyle="-",
        zorder=zorder,
        animated=True,
    )

    value_label, marker_y = _resolve_marker_y(
        ax, state.time_s, value_y=value_y, value_label=value_label
    )

    # Data marker sits on the displayed series only — never ylim mid-band
    # or mouse Y. Hidden when the canonical sample is missing.
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1.0
    has_marker = marker_y is not None
    cap = ax.scatter(
        [state.time_s],
        [float(marker_y) if has_marker else y0 + yr * 0.5],
        s=42,
        facecolors=_PLAYHEAD_CORE,
        edgecolors=TEXT,
        linewidths=0.8,
        zorder=zorder + 1,
        animated=True,
        clip_on=False,
    )
    if not has_marker:
        cap.set_visible(False)

    x_is_percent = bool(getattr(ax, "_stablewalk_x_is_percent", False))
    stamp = format_playhead_label(state, x_is_percent=x_is_percent)
    label_x, label_y, label_ha, _label_va = edge_aware_callout_placement(
        ax, state.time_s, y0 + yr * 0.06, text=stamp
    )

    label = None
    if show_label:
        label = ax.text(
            label_x,
            label_y,
            stamp,
            color=TEXT,
            fontsize=_LABEL_FONTSIZE,
            fontweight="medium",
            va="bottom",
            ha=label_ha,
            clip_on=False,
            zorder=zorder + 2,
            animated=True,
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
        from stablewalk.ui.viewers.chart_style import mark_secondary_annotation

        callout_y = float(marker_y) if has_marker else y0 + yr * 0.06
        vx, _vy, vha, vva = edge_aware_callout_placement(
            ax, state.time_s, callout_y, text=str(value_label)
        )
        value = mark_secondary_annotation(
            ax.text(
                vx,
                _vy,
                value_label,
                color=TEXT,
                fontsize=_VALUE_FONTSIZE,
                fontweight="semibold",
                va=vva,
                ha=vha,
                clip_on=False,
                zorder=zorder + 3,
                animated=True,
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": PANEL,
                    "edgecolor": MUTED,
                    "alpha": 0.94,
                    "linewidth": 0.55,
                },
            )
        )

    ax._stablewalk_playhead_artists = (glow, halo, core, cap, label, value)  # type: ignore[attr-defined]
    # Static background is stale after a full playhead recreate.
    fig = ax.figure
    if fig is not None:
        fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]


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
    x_is_percent = bool(getattr(ax, "_stablewalk_x_is_percent", False))
    value_label, marker_y = _resolve_marker_y(
        ax, state.time_s, value_y=value_y, value_label=value_label
    )
    has_marker = marker_y is not None
    lw, core_a, glow_a = playhead_pulse(
        state.anim_phase, animating=bool(state.animating)
    )
    try:
        for line in (glow, halo, core):
            line.set_xdata([state.time_s, state.time_s])
        core.set_linewidth(lw)
        core.set_alpha(core_a)
        glow.set_linewidth(lw + 2.2)
        glow.set_alpha(glow_a)
        halo.set_linewidth(lw + 0.8)
        halo.set_alpha(max(0.0, glow_a * 0.45))
        y0, y1 = ax.get_ylim()
        yr = y1 - y0 if y1 > y0 else 1.0
        stamp = format_playhead_label(state, x_is_percent=x_is_percent)
        if label is not None:
            lx, ly, lha, _lva = edge_aware_callout_placement(
                ax, state.time_s, y0 + yr * 0.06, text=stamp
            )
            label.set_position((lx, ly))
            label.set_ha(lha)
            label.set_text(stamp)
            label.set_visible(show_label)
        if value is not None and value_label:
            callout_y = float(marker_y) if has_marker else y0 + yr * 0.06
            vx, vy, vha, vva = edge_aware_callout_placement(
                ax, state.time_s, callout_y, text=str(value_label)
            )
            value.set_position((vx, vy))
            value.set_ha(vha)
            value.set_va(vva)
            value.set_text(value_label)
            value.set_visible(True)
        elif value is not None:
            value.set_visible(False)
        if cap is not None:
            if has_marker:
                cap.set_offsets([[state.time_s, float(marker_y)]])
                cap.set_visible(True)
            else:
                cap.set_offsets([[state.time_s, y0 + yr * 0.5]])
                cap.set_visible(False)
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
        value_label, value_y = resolve_playhead_value(ax, state.time_s)
        reused = (
            update_chart_playhead(
                ax,
                state,
                show_label=show_label,
                value_label=value_label,
                value_y=value_y,
            )
            and reused
        )
    return reused


def cache_static_chart_background(fig, canvas, axes: Sequence[Axes]) -> bool:
    """Snapshot static axes pixels so later frames can blit the cursor only."""
    if fig is None or canvas is None or not axes:
        return False
    try:
        canvas.draw()
        backgrounds: dict[int, object] = {}
        for ax in axes:
            backgrounds[id(ax)] = canvas.copy_from_bbox(ax.bbox)
        fig._stablewalk_playhead_bg = backgrounds  # type: ignore[attr-defined]
        fig._stablewalk_playhead_bg_size = (  # type: ignore[attr-defined]
            float(fig.bbox.width),
            float(fig.bbox.height),
        )
        return True
    except Exception:
        fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]
        return False


def clear_static_chart_background(fig) -> None:
    """Drop blit pixel caches so large Agg buffers can be GC'd."""
    if fig is None:
        return
    try:
        fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]
        fig._stablewalk_playhead_bg_size = None  # type: ignore[attr-defined]
    except Exception:
        pass


def blit_playhead_artists(fig, canvas, axes: Sequence[Axes]) -> bool:
    """Restore cached static pixels and redraw only playhead / cursor artists.

    Returns True when a blit succeeded (no full canvas redraw needed).
    """
    if fig is None or canvas is None or not axes:
        return False
    backgrounds = getattr(fig, "_stablewalk_playhead_bg", None)
    size = getattr(fig, "_stablewalk_playhead_bg_size", None)
    if not backgrounds or size is None:
        return False
    try:
        if (
            abs(float(fig.bbox.width) - float(size[0])) > 0.5
            or abs(float(fig.bbox.height) - float(size[1])) > 0.5
        ):
            fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]
            return False
        for ax in axes:
            bg = backgrounds.get(id(ax))
            if bg is None:
                return False
            canvas.restore_region(bg)
            artists = getattr(ax, "_stablewalk_playhead_artists", None) or ()
            for artist in artists:
                if artist is None:
                    continue
                try:
                    if not artist.get_visible():
                        continue
                except Exception:
                    pass
                try:
                    ax.draw_artist(artist)
                except Exception:
                    return False
            markers = getattr(ax, "_stablewalk_knee_playhead_points", None) or ()
            for artist in markers:
                if artist is None:
                    continue
                try:
                    ax.draw_artist(artist)
                except Exception:
                    return False
            canvas.blit(ax.bbox)
        return True
    except Exception:
        return False


__all__ = [
    "PLAYHEAD_VALUE_SOURCE_ATTR",
    "PlayheadState",
    "advance_playhead_anim_phase",
    "apply_playhead_to_axes",
    "blit_playhead_artists",
    "cache_static_chart_background",
    "clear_static_chart_background",
    "draw_chart_playhead",
    "edge_aware_callout_placement",
    "format_playhead_label",
    "playhead_from_gui",
    "playhead_from_playback",
    "playhead_pulse",
    "resolve_playhead_value",
    "sample_displayed_series",
    "set_playhead_value_source",
    "update_chart_playhead",
    "update_playhead_on_axes",
]
