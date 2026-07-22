"""Shared interactive hover tooltips and playhead sync for matplotlib charts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from stablewalk.ui.colors import BORDER, ELEVATED, TEXT

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

HOVER_PICK_RADIUS_PX = 14.0
FIG_HOVER_POINTS_ATTR = "_chart_hover_points"
FIG_HOVER_ANNOT_ATTR = "_chart_hover_annot"
FIG_HOVER_CID_ATTR = "_chart_hover_cid"


@dataclass(frozen=True)
class ChartHoverPoint:
    ax: Axes
    x: float
    y: float
    z: float | None = None
    frame_index: int | None = None
    list_pos: int | None = None
    timestamp_s: float | None = None
    value: float | str | None = None
    joint_name: str = "—"
    metric_name: str = "—"
    unit: str = ""


def format_chart_tooltip(point: ChartHoverPoint) -> str:
    """Standard multi-line tooltip: frame, time, value, joint, metric."""
    frame_text = "—" if point.frame_index is None else str(point.frame_index)
    if point.timestamp_s is None:
        time_text = "—"
    elif "cycle" in point.metric_name.lower():
        time_text = f"{point.timestamp_s:.1f}% gait cycle"
    else:
        time_text = f"{point.timestamp_s:.3f} s"

    if point.value is None:
        value_text = "—"
    elif isinstance(point.value, str):
        value_text = point.value
    elif point.unit:
        value_text = f"{point.value:.4g} {point.unit}"
    else:
        value_text = f"{point.value:.4g}"

    return (
        f"Frame: {frame_text}\n"
        f"Timestamp: {time_text}\n"
        f"Value: {value_text}\n"
        f"Joint: {point.joint_name}\n"
        f"Metric: {point.metric_name}"
    )


def append_line_hover_points(
    ax: Axes,
    xs: np.ndarray | list[float],
    ys: np.ndarray | list[float],
    *,
    metric_name: str,
    joint_name: str = "—",
    unit: str = "",
    frame_indices: list[int] | np.ndarray | None = None,
    list_positions: list[int] | np.ndarray | None = None,
    timestamps: list[float] | np.ndarray | None = None,
    hover_points: list[ChartHoverPoint],
) -> None:
    """Register one hover target per finite (x, y) sample."""
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    n = len(x_arr)
    for i in range(n):
        val = float(y_arr[i])
        if not np.isfinite(val):
            continue
        fi = None
        if frame_indices is not None and i < len(frame_indices):
            raw_fi = frame_indices[i]
            if raw_fi is not None:
                fi = int(raw_fi)
        lp = None
        if list_positions is not None and i < len(list_positions):
            lp = int(list_positions[i])
        ts = None
        if timestamps is not None and i < len(timestamps):
            ts = float(timestamps[i])
        elif np.isfinite(x_arr[i]):
            ts = float(x_arr[i])
        hover_points.append(
            ChartHoverPoint(
                ax=ax,
                x=float(x_arr[i]),
                y=val,
                frame_index=fi,
                list_pos=lp,
                timestamp_s=ts,
                value=val,
                joint_name=joint_name,
                metric_name=metric_name,
                unit=unit,
            )
        )


def register_dof_trajectory_hover_points(
    ax: Axes,
    path_with_times: list[tuple[object, float]],
    display_path: list[object],
    *,
    joint_name: str,
    metric_name: str = "3D position",
    hover_points: list[ChartHoverPoint],
) -> None:
    """Register hover targets along a displayed 3D joint trajectory."""
    n = min(len(path_with_times), len(display_path))
    for i in range(n):
        point = display_path[i]
        _orig, time_s = path_with_times[i]
        value = f"x={point.x:.3f}, y={point.y:.3f}, z={point.z:.3f} m"
        hover_points.append(
            ChartHoverPoint(
                ax=ax,
                x=float(point.x),
                y=float(point.y),
                z=float(point.z),
                frame_index=i,
                list_pos=i,
                timestamp_s=float(time_s),
                value=value,
                joint_name=joint_name,
                metric_name=metric_name,
                unit="",
            )
        )


def set_figure_hover_points(fig: Figure, points: list[ChartHoverPoint]) -> None:
    setattr(fig, FIG_HOVER_POINTS_ATTR, points)
    setattr(fig, FIG_HOVER_ANNOT_ATTR, None)


def attach_chart_hover_tooltips(
    fig: Figure,
    canvas: FigureCanvasTkAgg,
    *,
    on_hover_point: Callable[[ChartHoverPoint], None] | None = None,
) -> None:
    """Show tooltips near plotted samples and optionally sync the playhead."""
    cid = getattr(fig, FIG_HOVER_CID_ATTR, None)
    if cid is not None:
        try:
            canvas.mpl_disconnect(cid)
        except Exception:
            pass

    existing = getattr(fig, FIG_HOVER_ANNOT_ATTR, None)
    if existing is not None:
        try:
            existing.remove()
        except Exception:
            pass
    setattr(fig, FIG_HOVER_ANNOT_ATTR, None)

    def _hide_tooltip() -> None:
        annot = getattr(fig, FIG_HOVER_ANNOT_ATTR, None)
        if annot is not None:
            annot.set_visible(False)
            canvas.draw_idle()

    def _display_coords(pt: ChartHoverPoint, event_axes: Axes) -> tuple[float, float] | None:
        if pt.ax is not event_axes:
            return None
        try:
            if pt.z is not None:
                xy = pt.ax.transData.transform((pt.x, pt.y, pt.z))
            else:
                xy = pt.ax.transData.transform((pt.x, pt.y))
            return float(xy[0]), float(xy[1])
        except Exception:
            return None

    def _seek_key(pt: ChartHoverPoint) -> tuple:
        return (
            pt.list_pos,
            pt.frame_index,
            round(pt.timestamp_s, 4) if pt.timestamp_s is not None else None,
            pt.metric_name,
            pt.joint_name,
        )

    last_seek_key: list[tuple | None] = [None]

    def _on_motion(event) -> None:
        from stablewalk.ui.viewers.chart_navigation import FIG_NAV_BLOCK_HOVER_ATTR, chart_nav_is_active

        if chart_nav_is_active(fig) or getattr(fig, FIG_NAV_BLOCK_HOVER_ATTR, False):
            _hide_tooltip()
            return
        points: list[ChartHoverPoint] = getattr(fig, FIG_HOVER_POINTS_ATTR, [])
        if event.inaxes is None or event.x is None or event.y is None or not points:
            _hide_tooltip()
            return

        best: ChartHoverPoint | None = None
        best_dist = float("inf")
        for pt in points:
            disp = _display_coords(pt, event.inaxes)
            if disp is None:
                continue
            dist = float(np.hypot(event.x - disp[0], event.y - disp[1]))
            if dist < HOVER_PICK_RADIUS_PX and dist < best_dist:
                best_dist = dist
                best = pt

        if best is None:
            _hide_tooltip()
            return

        annot = getattr(fig, FIG_HOVER_ANNOT_ATTR, None)
        if annot is None or annot.axes is not best.ax:
            if annot is not None:
                try:
                    annot.remove()
                except Exception:
                    pass
            annot = best.ax.annotate(
                "",
                xy=(best.x, best.y),
                xytext=(12, 12),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=ELEVATED, edgecolor=BORDER, alpha=0.96),
                color=TEXT,
                fontsize=8,
                zorder=30,
                visible=False,
            )
            setattr(fig, FIG_HOVER_ANNOT_ATTR, annot)

        annot.xy = (best.x, best.y)
        annot.set_text(format_chart_tooltip(best))
        annot.set_visible(True)
        canvas.draw_idle()

        if on_hover_point is not None:
            key = _seek_key(best)
            if key != last_seek_key[0]:
                last_seek_key[0] = key
                on_hover_point(best)

    setattr(fig, FIG_HOVER_CID_ATTR, canvas.mpl_connect("motion_notify_event", _on_motion))


__all__ = [
    "ChartHoverPoint",
    "FIG_HOVER_POINTS_ATTR",
    "append_line_hover_points",
    "attach_chart_hover_tooltips",
    "format_chart_tooltip",
    "register_dof_trajectory_hover_points",
    "set_figure_hover_points",
]
