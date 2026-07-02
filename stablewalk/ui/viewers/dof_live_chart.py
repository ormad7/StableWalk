"""
Live time-series chart for a selected GUI degree of freedom.

Plots X, Y, Z (meters) and flexion angle (degrees) up to the current playback
frame, with a playhead marker at the active timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from stablewalk.ui.colors import BORDER, MUTED, PANEL, TEXT
from stablewalk.ui.dof_position_table import angle_value_for_item
from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from stablewalk.models.gait_motion import GaitMotionRecording


@dataclass(frozen=True)
class DofLiveSeries:
    """Numeric samples for one selected DOF from the start of the walk."""

    times_s: tuple[float, ...]
    x: tuple[float | None, ...]
    y: tuple[float | None, ...]
    z: tuple[float | None, ...]
    angle_deg: tuple[float | None, ...]
    dof_label: str


def collect_dof_live_series(
    recording: GaitMotionRecording,
    item_id: str,
    *,
    end_frame_index: int,
) -> DofLiveSeries:
    """Collect X/Y/Z/angle samples from frame 0 through ``end_frame_index``."""
    anchor = anchor_joint_for_item(item_id)
    end = max(0, min(int(end_frame_index), recording.frame_count - 1))
    times: list[float] = []
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    angles: list[float | None] = []

    for index in range(end + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        times.append(float(snap.time_s))
        joint = snap.joints.get(anchor) if anchor else None
        if joint is not None:
            xs.append(float(joint.position.x))
            ys.append(float(joint.position.y))
            zs.append(float(joint.position.z))
        else:
            xs.append(None)
            ys.append(None)
            zs.append(None)
        angle = angle_value_for_item(item_id, snap)
        angles.append(float(angle) if angle is not None else None)

    return DofLiveSeries(
        times_s=tuple(times),
        x=tuple(xs),
        y=tuple(ys),
        z=tuple(zs),
        angle_deg=tuple(angles),
        dof_label=label_for_item(item_id),
    )


# Rendering-only limits (does not affect data collection).
DOF_LIVE_CHART_MAX_POINTS = 50

# Distinct colors and line styles for at-a-glance reading.
_SERIES_STYLE: tuple[tuple[str, str, str, str, str], ...] = (
    ("X", "X (m)", "#2ee59d", "-", "2.0"),
    ("Y", "Y (m)", "#4dabf7", "--", "2.0"),
    ("Z", "Z (m)", "#ffc857", "-.", "2.0"),
    ("Angle", "Angle (°)", "#ff6b81", "-", "2.2"),
)

# Plot band (top) + x-label band + legend band (bottom).
DOF_LIVE_CHART_MARGINS = dict(left=0.11, right=0.76, top=0.91, bottom=0.25)
_LEGEND_ANCHOR = (0.52, 0.03)


def setup_dof_live_chart_axes(ax: Axes) -> None:
    """Apply dashboard styling to the live DOF chart axes."""
    ax.set_facecolor(PANEL)
    ax.tick_params(axis="both", colors=MUTED, labelsize=7.5, pad=3, length=3)
    ax.set_xlabel("Time (s)", color=MUTED, fontsize=8, labelpad=6)
    ax.set_ylabel("Position (m)", color=MUTED, fontsize=8, labelpad=6)
    ax.grid(True, color=BORDER, alpha=0.32, linestyle="--", linewidth=0.55, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def _clear_extra_axes(ax: Axes) -> None:
    """Remove twin/helper axes left over from previous redraws."""
    fig = ax.figure
    for sibling in list(fig.axes):
        if sibling is not ax:
            sibling.remove()
    for legend in list(getattr(fig, "legends", []) or []):
        legend.remove()


def _format_meters(value: float, _pos: int) -> str:
    magnitude = abs(value)
    if magnitude >= 10.0:
        return f"{value:.1f}"
    if magnitude >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _format_degrees(value: float, _pos: int) -> str:
    return f"{value:.1f}"


def _format_time(value: float, _pos: int) -> str:
    return f"{value:.1f}"


def _style_angle_axis(angle_ax: Axes) -> None:
    angle_ax.set_facecolor("none")
    angle_ax.set_ylabel("Angle (°)", color=MUTED, fontsize=8, labelpad=8)
    angle_ax.yaxis.set_label_coords(1.02, 0.5)
    angle_ax.tick_params(
        axis="y",
        colors=MUTED,
        labelsize=7.5,
        pad=5,
        length=3,
        direction="out",
    )
    angle_ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, prune="both"))
    angle_ax.yaxis.set_major_formatter(FuncFormatter(_format_degrees))
    angle_ax.spines["top"].set_visible(False)
    angle_ax.spines["left"].set_visible(False)
    angle_ax.spines["bottom"].set_visible(False)
    angle_ax.spines["right"].set_color(BORDER)
    angle_ax.spines["right"].set_linewidth(0.8)


def _style_primary_ticks(ax: Axes) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, prune="both"))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_time))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3, prune="both"))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_meters))


def _layer_twin_axes(ax: Axes, angle_ax: Axes) -> None:
    """Keep position lines above the angle axis without hiding tick labels."""
    angle_ax.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_alpha(0.0)


def _window_arrays_for_display(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    angle: np.ndarray,
    *,
    max_points: int = DOF_LIVE_CHART_MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return the latest ``max_points`` samples for plotting only."""
    if len(times) <= max_points:
        return times, x, y, z, angle, False
    start = len(times) - max_points
    return times[start:], x[start:], y[start:], z[start:], angle[start:], True


def _set_time_limits(ax: Axes, times: np.ndarray) -> None:
    if len(times) == 0:
        return
    if len(times) == 1:
        pad = 0.05
    else:
        span = float(times[-1] - times[0])
        pad = max(span * 0.03, 0.02)
    ax.set_xlim(float(times[0]) - pad, float(times[-1]) + pad)


def apply_dof_live_chart_layout(fig: Figure) -> None:
    """Apply consistent outer margins after each draw."""
    fig.subplots_adjust(**DOF_LIVE_CHART_MARGINS)


def _as_float_array(values: tuple[float | None, ...]) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)


def _legend_handles() -> list[Line2D]:
    """Explicit legend entries showing each series color and linestyle."""
    return [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=float(lw),
            label=label,
        )
        for _key, label, color, linestyle, lw in _SERIES_STYLE
    ]


def _draw_chart_legend(fig: Figure) -> None:
    """Compact legend in the reserved band below the plot."""
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=_LEGEND_ANCHOR,
        ncol=4,
        fontsize=7.5,
        framealpha=0.96,
        facecolor=PANEL,
        edgecolor=BORDER,
        labelcolor=TEXT,
        handlelength=2.2,
        handletextpad=0.5,
        borderpad=0.35,
        labelspacing=0.3,
        columnspacing=0.9,
    )


def _plot_position_series(
    ax: Axes,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> None:
    series_data = {"X": x, "Y": y, "Z": z}
    for key, _label, color, linestyle, lw in _SERIES_STYLE[:3]:
        ax.plot(
            times,
            series_data[key],
            color=color,
            linestyle=linestyle,
            linewidth=float(lw),
            label=_label,
            alpha=0.98,
            zorder=3,
            solid_capstyle="round",
        )


def draw_dof_live_chart(
    ax: Axes,
    recording: GaitMotionRecording | None,
    item_id: str | None,
    *,
    end_frame_index: int = 0,
    current_time_s: float | None = None,
    clear: bool = True,
) -> bool:
    """
    Draw the live chart for ``item_id`` up to ``end_frame_index``.

    Returns True when a series was drawn, False when an empty-state message was shown.
    """
    fig = ax.figure
    if clear:
        _clear_extra_axes(ax)
        ax.cla()
        setup_dof_live_chart_axes(ax)

    if not item_id or recording is None or recording.frame_count <= 0:
        apply_dof_live_chart_layout(fig)
        ax.text(
            0.5,
            0.5,
            "Select a degree of freedom",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        return False

    series = collect_dof_live_series(
        recording,
        item_id,
        end_frame_index=end_frame_index,
    )
    if not series.times_s:
        apply_dof_live_chart_layout(fig)
        ax.text(
            0.5,
            0.5,
            "No motion data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        return False

    times = np.asarray(series.times_s, dtype=float)
    x = _as_float_array(series.x)
    y = _as_float_array(series.y)
    z = _as_float_array(series.z)
    angle = _as_float_array(series.angle_deg)

    times, x, y, z, angle, windowed = _window_arrays_for_display(times, x, y, z, angle)

    _plot_position_series(ax, times, x, y, z)

    _angle_key, _angle_label, angle_color, angle_ls, angle_lw = _SERIES_STYLE[3]
    angle_ax = ax.twinx()
    angle_ax.plot(
        times,
        angle,
        color=angle_color,
        linestyle=angle_ls,
        linewidth=float(angle_lw),
        label=_angle_label,
        alpha=0.96,
        zorder=2,
        solid_capstyle="round",
    )
    _style_angle_axis(angle_ax)
    _style_primary_ticks(ax)
    _layer_twin_axes(ax, angle_ax)
    _set_time_limits(ax, times)

    play_time = current_time_s
    if play_time is None and series.times_s:
        play_time = series.times_s[-1]
    if play_time is not None and len(times) > 0:
        t_min, t_max = float(times[0]), float(times[-1])
        if t_min - 1e-6 <= float(play_time) <= t_max + 1e-6:
            ax.axvline(
                float(play_time),
                color=TEXT,
                linewidth=0.9,
                linestyle=":",
                alpha=0.5,
                zorder=1,
            )

    title = series.dof_label
    if windowed:
        title = f"{title}  ·  last {DOF_LIVE_CHART_MAX_POINTS} frames"
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="medium", pad=6)

    _draw_chart_legend(fig)
    apply_dof_live_chart_layout(fig)
    return True
