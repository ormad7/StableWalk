"""Gait cycle % chart for knee trajectories (mean ± SD envelope)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stablewalk.analysis.gait_feature_analysis import GaitFeatureAnalysisResult
from stablewalk.models.pose_data import PoseSequence
from stablewalk.ui.colors import BORDER, MUTED, PANEL, SIDE_LEFT, SIDE_RIGHT, TEXT
from stablewalk.ui.viewers.knee_angle_chart import (
    KneeAngleSeries,
    LABEL_LEFT_KNEE,
    LABEL_RIGHT_KNEE,
    build_knee_angle_series,
    draw_knee_time_chart,
)
from stablewalk.ui.viewers.knee_chart_interpretation import (
    MIN_CYCLES_FOR_CYCLE_MODE,
    usable_knee_cycle_count,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def _to_flexion(mean_or_arr: np.ndarray, angle_source: str) -> np.ndarray:
    arr = np.asarray(mean_or_arr, dtype=float)
    if angle_source == "mediapipe_angles":
        return 180.0 - arr
    return arr


def draw_knee_chart_time_mode(
    ax: Axes,
    sequence: PoseSequence,
    pose_indices: list[int],
    *,
    playhead_index: int | None = None,
    playhead_list_pos: int | None = None,
    series: KneeAngleSeries | None = None,
    ik_mot_path=None,
    source_preference: str = "auto",
    ik_quality_ok: bool | None = None,
    gait_events: list | None = None,
    gait_cycle=None,
) -> KneeAngleSeries:
    """Plot left/right knee flexion vs video time (seconds)."""
    if series is None:
        series = build_knee_angle_series(
            sequence,
            pose_indices,
            ik_mot_path=ik_mot_path,
            source_preference=source_preference,  # type: ignore[arg-type]
            ik_quality_ok=ik_quality_ok,
        )

    list_pos = playhead_list_pos
    if list_pos is None and playhead_index is not None and pose_indices:
        try:
            list_pos = pose_indices.index(playhead_index)
        except ValueError:
            list_pos = None

    draw_knee_time_chart(
        ax,
        series,
        playhead_list_pos=list_pos,
        gait_events=gait_events,
        gait_cycle=gait_cycle,
    )
    return series


def draw_knee_chart_cycle_mode(
    ax: Axes,
    gait_features: GaitFeatureAnalysisResult | None,
    *,
    show_envelope: bool = True,
    min_cycles_for_envelope: int = 3,
) -> bool:
    """
    Plot mean knee flexion trajectories across gait cycles (0–100%).

    Returns True when a cycle-normalized plot was drawn.
    """
    from stablewalk.ui.viewers.chart_reference import (
        KNEE_FLEXION_ABNORMAL_ABOVE_DEG,
        KNEE_FLEXION_NORMAL_DEG,
        draw_reference_y_bands,
    )
    from stablewalk.ui.viewers.chart_style import (
        style_chart_legend,
        style_chart_title,
        style_single_time_series_chart,
    )

    usable = usable_knee_cycle_count(gait_features)
    if usable < MIN_CYCLES_FOR_CYCLE_MODE:
        style_single_time_series_chart(ax, ylabel="Knee flexion (°)", x_is_percent=True)
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
            wrap=True,
        )
        style_chart_title(ax, "Knee flexion · Gait Cycle %")
        ax.set_xlim(0, 100)
        return False

    if gait_features is None:
        style_single_time_series_chart(ax, ylabel="Knee flexion (°)", x_is_percent=True)
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=10,
        )
        style_chart_title(ax, "Knee flexion · Gait Cycle %")
        return False

    cc = gait_features.cycle_consistency
    left = cc.trajectories.get("left_knee_angle")
    right = cc.trajectories.get("right_knee_angle")
    if not left and not right:
        style_single_time_series_chart(ax, ylabel="Knee flexion (°)", x_is_percent=True)
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=10,
        )
        style_chart_title(ax, "Knee flexion · Gait Cycle %")
        return False

    style_single_time_series_chart(ax, ylabel="Knee flexion (°)", x_is_percent=True)

    def _plot_traj(traj, color: str, line_label: str, end_label: str) -> None:
        if traj is None or len(traj.per_cycle) < MIN_CYCLES_FOR_CYCLE_MODE:
            return
        pct = np.asarray(traj.percent)
        mean = _to_flexion(traj.mean, cc.angle_source)
        ax.plot(pct, mean, color=color, label=f"{line_label} (mean)", linewidth=1.85, zorder=4)
        if show_envelope and len(traj.per_cycle) >= min_cycles_for_envelope:
            std = np.asarray(traj.std, dtype=float)
            ax.fill_between(
                pct,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
                label=f"{line_label} ±1 SD",
            )
        if mean.size:
            ax.annotate(
                end_label,
                xy=(float(pct[-1]), float(mean[-1])),
                xytext=(5, 0),
                textcoords="offset points",
                color=color,
                fontsize=8,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=True,
                zorder=7,
            )

    _plot_traj(left, SIDE_LEFT, LABEL_LEFT_KNEE, LABEL_LEFT_KNEE)
    _plot_traj(right, SIDE_RIGHT, LABEL_RIGHT_KNEE, LABEL_RIGHT_KNEE)

    if not ax.lines:
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=10,
        )
        style_chart_title(ax, "Knee flexion · Gait Cycle %")
        return False

    src = "OpenSim IK" if cc.angle_source == "opensim_ik" else "Pose-derived"
    details: list[str] = []
    if cc.cycle_repeatability_score is not None:
        details.append(f"Repeatability {cc.cycle_repeatability_score:.0f}")
    details.append(src)
    style_chart_title(ax, f"Knee Flexion · {usable} Cycles · {' · '.join(details)}")
    ax.set_xlim(0, 100)

    finite = []
    for traj in (left, right):
        if traj is not None and len(traj.per_cycle) >= MIN_CYCLES_FOR_CYCLE_MODE:
            finite.extend(_to_flexion(traj.mean, cc.angle_source).tolist())
    if finite:
        y_lo = min(float(np.min(finite)), KNEE_FLEXION_NORMAL_DEG[0])
        y_hi = max(
            float(np.max(finite)),
            KNEE_FLEXION_NORMAL_DEG[1],
            KNEE_FLEXION_ABNORMAL_ABOVE_DEG,
        )
        margin = max(6.0, (y_hi - y_lo) * 0.10)
        ax.set_ylim(y_lo - margin, y_hi + margin)

    draw_reference_y_bands(
        ax,
        normal=KNEE_FLEXION_NORMAL_DEG,
        abnormal_below=-5.0,
        abnormal_above=KNEE_FLEXION_ABNORMAL_ABOVE_DEG,
        label_normal=True,
    )
    _draw_cycle_phase_regions(ax)
    style_chart_legend(ax, loc="upper right", fontsize=8.0)
    return True


def _draw_cycle_phase_regions(ax: Axes) -> None:
    """Stance / swing bands + typical HS (0%) and TO (~60%) markers."""
    from stablewalk.ui.viewers.chart_reference import EVENT_HS, EVENT_TO

    ax.axvspan(0, 62, facecolor=BORDER, alpha=0.08, zorder=0, label="Stance (~0–62%)")
    ax.axvspan(62, 100, facecolor=BORDER, alpha=0.04, zorder=0, label="Swing (~62–100%)")
    ax.axvline(0.0, color=EVENT_HS, linewidth=1.0, alpha=0.55, linestyle=(0, (3, 2)), zorder=2.5)
    ax.axvline(62.0, color=EVENT_TO, linewidth=1.0, alpha=0.55, linestyle=(0, (3, 2)), zorder=2.5)
    y0, y1 = ax.get_ylim()
    yr = y1 - y0 if y1 > y0 else 1.0
    ax.scatter([0.0], [y1 - yr * 0.04], marker="v", s=32, color=EVENT_HS, zorder=6, label="Heel strike")
    ax.scatter([62.0], [y0 + yr * 0.04], marker="^", s=32, color=EVENT_TO, zorder=6, label="Toe off")


def style_gait_chart(ax: Axes, fig) -> None:
    from stablewalk.ui.viewers.chart_style import (
        apply_chart_grid,
        apply_chart_panel_style,
        configure_numeric_y_axis,
        configure_percent_axis,
    )

    apply_chart_panel_style(ax)
    configure_percent_axis(ax, show_xlabel=True)
    ylabel = ax.get_ylabel() or "Knee flexion (°)"
    configure_numeric_y_axis(ax, ylabel)
    apply_chart_grid(ax, y_minor=True)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup_h, dedup_l, seen = [], [], set()
        for h, lab in zip(handles, labels):
            if lab in seen or lab.startswith("_"):
                continue
            seen.add(lab)
            dedup_h.append(h)
            dedup_l.append(lab)
        if dedup_h:
            leg = ax.legend(
                dedup_h,
                dedup_l,
                facecolor=PANEL,
                edgecolor=BORDER,
                labelcolor=TEXT,
                fontsize=8.0,
                framealpha=0.94,
                fancybox=False,
                loc="upper right",
                borderpad=0.4,
                handlelength=1.8,
                handletextpad=0.55,
                labelspacing=0.4,
            )
            if leg is not None:
                leg.get_frame().set_linewidth(0.7)
    if ax.get_xlabel() != "Gait Cycle (%)":
        ax.text(
            0.01,
            0.98,
            "HS = Heel Strike   ·   TO = Toe Off",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color=MUTED,
            fontsize=8.5,
            zorder=12,
        )
    fig.tight_layout(pad=1.35)
