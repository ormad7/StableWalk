"""
Joint Motion Analysis — multi-joint scientific time-series chart.

Six synchronized panels (angle, angular velocity/acceleration, X/Y/Z) with a
shared playhead, per-joint colors, zoom/pan, and CSV/PNG export helpers.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from matplotlib.ticker import MaxNLocator

from stablewalk.ui.colors import BORDER, MUTED, PANEL, TEXT
from stablewalk.ui.dof_position_table import _ITEM_DOF_ID, angle_value_for_item
from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, anchor_joint_for_item, label_for_item
from stablewalk.ui.joint_colors import joint_color
from stablewalk.ui.viewers.chart_hover import ChartHoverPoint
from stablewalk.ui.viewers.chart_playhead import PlayheadState, draw_chart_playhead
from stablewalk.ui.viewers.chart_style import (
    TIMELINE_X_LABEL,
    apply_chart_grid,
    apply_chart_panel_style,
    configure_time_axis,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from stablewalk.models.gait_motion import GaitMotionRecording

JOINT_MOTION_METRICS: tuple[tuple[str, str, str], ...] = (
    ("angle", "Joint Angle vs Time", "Angle (°)"),
    ("omega", "Angular Velocity vs Time", "ω (°/s)"),
    ("alpha", "Angular Acceleration vs Time", "α (°/s²)"),
    ("x", "X Position vs Time", "X (m)"),
    ("y", "Y Position vs Time", "Y (m)"),
    ("z", "Z Position vs Time", "Z (m)"),
)

CSV_FIELDNAMES: tuple[str, ...] = (
    "time_s",
    "frame",
    "joint_id",
    "joint_label",
    "angle_deg",
    "angular_velocity_deg_s",
    "angular_acceleration_deg_s2",
    "x_m",
    "y_m",
    "z_m",
)


@dataclass(frozen=True)
class JointMotionSeries:
    """Full-clip kinematics for one selected GUI joint."""

    item_id: str
    label: str
    color: str
    times_s: np.ndarray
    frames: np.ndarray
    angle_deg: np.ndarray
    omega_deg_s: np.ndarray
    alpha_deg_s2: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray

    def values_for(self, metric_id: str) -> np.ndarray:
        return {
            "angle": self.angle_deg,
            "omega": self.omega_deg_s,
            "alpha": self.alpha_deg_s2,
            "x": self.x_m,
            "y": self.y_m,
            "z": self.z_m,
        }[metric_id]


@dataclass(frozen=True)
class JointMotionBundle:
    """Multi-joint series aligned on a shared time base."""

    series: tuple[JointMotionSeries, ...]
    times_s: np.ndarray

    @property
    def empty(self) -> bool:
        return not self.series or self.times_s.size == 0


def _ordered_item_ids(selected: Iterable[str]) -> list[str]:
    selected_set = {str(item) for item in selected}
    return [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_set]


def _finite_diff(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Central/forward finite difference; NaN where samples are missing."""
    out = np.full_like(values, np.nan, dtype=float)
    n = len(values)
    if n < 2:
        return out
    for i in range(n):
        if not np.isfinite(values[i]):
            continue
        if i == 0:
            j = 1
            while j < n and not np.isfinite(values[j]):
                j += 1
            if j >= n:
                continue
            dt = times[j] - times[i]
            if dt > 1e-9:
                out[i] = (values[j] - values[i]) / dt
        elif i == n - 1:
            j = n - 2
            while j >= 0 and not np.isfinite(values[j]):
                j -= 1
            if j < 0:
                continue
            dt = times[i] - times[j]
            if dt > 1e-9:
                out[i] = (values[i] - values[j]) / dt
        else:
            j0, j1 = i - 1, i + 1
            while j0 >= 0 and not np.isfinite(values[j0]):
                j0 -= 1
            while j1 < n and not np.isfinite(values[j1]):
                j1 += 1
            if j0 < 0 or j1 >= n:
                continue
            dt = times[j1] - times[j0]
            if dt > 1e-9:
                out[i] = (values[j1] - values[j0]) / dt
    return out


def build_joint_motion_series(
    recording: GaitMotionRecording,
    item_id: str,
) -> JointMotionSeries | None:
    """Collect angle / ω / α / XYZ for one joint across the recording."""
    if recording is None or recording.frame_count <= 0:
        return None
    anchor = anchor_joint_for_item(item_id)
    dof_id = _ITEM_DOF_ID.get(item_id)
    n = recording.frame_count
    times = np.empty(n, dtype=float)
    frames = np.empty(n, dtype=int)
    angle = np.full(n, np.nan, dtype=float)
    omega = np.full(n, np.nan, dtype=float)
    xs = np.full(n, np.nan, dtype=float)
    ys = np.full(n, np.nan, dtype=float)
    zs = np.full(n, np.nan, dtype=float)

    for index in range(n):
        snap = recording.snapshot_at(index)
        if snap is None:
            times[index] = index / max(float(recording.fps), 1e-6)
            frames[index] = index + 1
            continue
        times[index] = float(snap.time_s)
        frames[index] = int(snap.frame_index) + 1
        joint = snap.joints.get(anchor) if anchor else None
        if joint is not None:
            xs[index] = float(joint.position.x)
            ys[index] = float(joint.position.y)
            zs[index] = float(joint.position.z)
        ang = angle_value_for_item(item_id, snap)
        if ang is not None:
            angle[index] = float(ang)
        dof = snap.get_dof(dof_id) if dof_id else None
        if dof is not None and dof.velocity_deg_s is not None:
            omega[index] = float(dof.velocity_deg_s)

    # Prefer stored ω; if sparse, estimate from angle.
    if not np.isfinite(omega).any() and np.isfinite(angle).sum() >= 2:
        omega = _finite_diff(angle, times)
    alpha = _finite_diff(omega, times)

    return JointMotionSeries(
        item_id=item_id,
        label=label_for_item(item_id),
        color=joint_color(item_id),
        times_s=times,
        frames=frames,
        angle_deg=angle,
        omega_deg_s=omega,
        alpha_deg_s2=alpha,
        x_m=xs,
        y_m=ys,
        z_m=zs,
    )


def build_joint_motion_bundle(
    recording: GaitMotionRecording | None,
    selected_item_ids: Iterable[str],
) -> JointMotionBundle:
    """Build multi-joint series for the current selection (stable GUI order)."""
    if recording is None or recording.frame_count <= 0:
        return JointMotionBundle(series=(), times_s=np.asarray([], dtype=float))
    series_list: list[JointMotionSeries] = []
    for item_id in _ordered_item_ids(selected_item_ids):
        series = build_joint_motion_series(recording, item_id)
        if series is not None:
            series_list.append(series)
    times = (
        series_list[0].times_s
        if series_list
        else np.asarray([], dtype=float)
    )
    return JointMotionBundle(series=tuple(series_list), times_s=times)


def write_joint_motion_csv(
    bundle: JointMotionBundle,
    path: str | Path,
) -> Path:
    """Export all selected joints' kinematics to CSV."""
    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for series in bundle.series:
            for i in range(len(series.times_s)):
                writer.writerow(
                    {
                        "time_s": f"{float(series.times_s[i]):.6f}",
                        "frame": int(series.frames[i]),
                        "joint_id": series.item_id,
                        "joint_label": series.label,
                        "angle_deg": _csv_num(series.angle_deg[i]),
                        "angular_velocity_deg_s": _csv_num(series.omega_deg_s[i]),
                        "angular_acceleration_deg_s2": _csv_num(
                            series.alpha_deg_s2[i]
                        ),
                        "x_m": _csv_num(series.x_m[i]),
                        "y_m": _csv_num(series.y_m[i]),
                        "z_m": _csv_num(series.z_m[i]),
                    }
                )
    return out


def _csv_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.6f}"


def ensure_joint_motion_axes(fig: Figure) -> list[Axes]:
    """Create or reuse a 2×3 shared-x subplot grid."""
    if len(fig.axes) == 6:
        return list(fig.axes)
    fig.clear()
    axes = fig.subplots(2, 3, sharex=True)
    return [ax for row in axes for ax in row]


def draw_joint_motion_analysis_chart(
    fig: Figure,
    bundle: JointMotionBundle,
    *,
    playhead: PlayheadState | None = None,
) -> list[ChartHoverPoint]:
    """Draw the six scientific panels and return hover targets."""
    axes = ensure_joint_motion_axes(fig)
    fig.patch.set_facecolor(PANEL)
    hover: list[ChartHoverPoint] = []

    if bundle.empty:
        for ax in axes:
            ax.clear()
            apply_chart_panel_style(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(BORDER)
        axes[1].text(
            0.5,
            0.5,
            "Select one or more joints to plot\nangle, angular rates, and position vs time.",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        fig.subplots_adjust(
            left=0.07, right=0.98, top=0.90, bottom=0.12, wspace=0.28, hspace=0.38
        )
        return hover

    legend_handles = []
    legend_labels: list[str] = []

    for ax, (metric_id, title, ylabel) in zip(axes, JOINT_MOTION_METRICS, strict=True):
        ax.clear()
        apply_chart_panel_style(ax)
        for series in bundle.series:
            y = series.values_for(metric_id)
            mask = np.isfinite(series.times_s) & np.isfinite(y)
            if not np.any(mask):
                continue
            (line,) = ax.plot(
                series.times_s[mask],
                y[mask],
                color=series.color,
                linewidth=1.7,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=series.label,
                zorder=4,
            )
            if metric_id == "angle" and series.label not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(series.label)

            idxs = np.flatnonzero(mask)
            step = max(1, len(idxs) // 40)
            for idx in idxs[::step]:
                hover.append(
                    ChartHoverPoint(
                        ax=ax,
                        x=float(series.times_s[idx]),
                        y=float(y[idx]),
                        frame_index=int(series.frames[idx]),
                        timestamp_s=float(series.times_s[idx]),
                        value=float(y[idx]),
                        joint_name=series.label,
                        metric_name=title.split(" vs ")[0],
                        unit=ylabel,
                    )
                )

        ax.set_title(title, color=TEXT, fontsize=11.0, fontweight="medium", pad=6)
        ax.set_ylabel(ylabel, color=MUTED, fontsize=10.0)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
        apply_chart_grid(ax, y_minor=True)
        if playhead is not None:
            value_label = None
            value_y = None
            if bundle.series:
                s0 = bundle.series[0]
                y = s0.values_for(metric_id)
                if y.size and np.isfinite(y).any():
                    value_y = float(np.interp(playhead.time_s, s0.times_s, y))
                    if np.isfinite(value_y):
                        value_label = f"{s0.label}: {value_y:.2f}"
            draw_chart_playhead(
                ax,
                playhead,
                show_label=(ax is axes[2]),
                value_label=value_label if ax is axes[0] else None,
                value_y=value_y if ax is axes[0] else None,
                zorder=20,
            )

    for ax in axes[3:]:
        configure_time_axis(ax, show_xlabel=True, nbins=5)
        ax.set_xlabel(TIMELINE_X_LABEL, color=MUTED, fontsize=10.0)
    for ax in axes[:3]:
        configure_time_axis(ax, show_xlabel=False, nbins=5)

    if legend_handles:
        leg = fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=min(6, len(legend_labels)),
            fontsize=8.5,
            frameon=True,
            fancybox=False,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            framealpha=0.94,
            borderpad=0.4,
            handlelength=1.8,
            handletextpad=0.55,
            labelspacing=0.4,
            bbox_to_anchor=(0.5, 0.995),
        )
        if leg is not None:
            leg.get_frame().set_linewidth(0.7)

    t0 = float(np.nanmin(bundle.times_s)) if bundle.times_s.size else 0.0
    t1 = float(np.nanmax(bundle.times_s)) if bundle.times_s.size else 1.0
    if t1 <= t0:
        t1 = t0 + 1.0
    for ax in axes:
        ax.set_xlim(t0, t1)

    fig.subplots_adjust(
        left=0.07, right=0.985, top=0.86, bottom=0.12, wspace=0.30, hspace=0.42
    )
    return hover
