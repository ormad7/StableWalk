"""Gait cycle % chart for knee trajectories (mean ± SD envelope)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stablewalk.analysis.gait_feature_analysis import CycleTrajectory, GaitFeatureAnalysisResult
from stablewalk.models.pose_data import PoseSequence
from stablewalk.ui.colors import ACCENT, BORDER, ELEVATED, MUTED, PANEL, TEXT, WARNING
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
    usable = usable_knee_cycle_count(gait_features)
    if usable < MIN_CYCLES_FOR_CYCLE_MODE:
        ax.set_facecolor(PANEL)
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
            wrap=True,
        )
        ax.set_title(
            "Knee flexion · Gait Cycle %",
            color=TEXT,
            fontsize=10,
            fontweight="medium",
            pad=6,
        )
        ax.set_xlabel("Gait Cycle (%)", color=MUTED, fontsize=8)
        ax.set_ylabel("Knee flexion (deg)", color=MUTED, fontsize=8)
        ax.set_xlim(0, 100)
        return False

    if gait_features is None:
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=9,
        )
        ax.set_title("Knee flexion · Gait Cycle %", color=TEXT, fontsize=10, fontweight="medium", pad=6)
        return False

    cc = gait_features.cycle_consistency
    left = cc.trajectories.get("left_knee_angle")
    right = cc.trajectories.get("right_knee_angle")
    if not left and not right:
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=9,
        )
        ax.set_title("Knee flexion · Gait Cycle %", color=TEXT, fontsize=10, fontweight="medium", pad=6)
        return False

    def _plot_traj(traj, color: str, line_label: str, end_label: str) -> None:
        if traj is None or len(traj.per_cycle) < MIN_CYCLES_FOR_CYCLE_MODE:
            return
        pct = np.asarray(traj.percent)
        mean = _to_flexion(traj.mean, cc.angle_source)
        ax.plot(pct, mean, color=color, label=f"{line_label} (mean)", linewidth=2.2, zorder=4)
        if (
            show_envelope
            and len(traj.per_cycle) >= min_cycles_for_envelope
        ):
            std = np.asarray(traj.std, dtype=float)
            if cc.angle_source == "mediapipe_angles":
                std = std  # symmetric under reflection
            ax.fill_between(
                pct,
                mean - std,
                mean + std,
                color=color,
                alpha=0.18,
                linewidth=0,
                label=f"{line_label} ± SD",
            )
        if mean.size:
            ax.annotate(
                end_label,
                xy=(float(pct[-1]), float(mean[-1])),
                xytext=(4, 0),
                textcoords="offset points",
                color=color,
                fontsize=7,
                fontweight="bold",
                va="center",
                ha="left",
                clip_on=True,
                zorder=7,
            )

    _plot_traj(left, ACCENT, LABEL_LEFT_KNEE, LABEL_LEFT_KNEE)
    _plot_traj(right, WARNING, LABEL_RIGHT_KNEE, LABEL_RIGHT_KNEE)

    if not ax.lines:
        ax.text(
            0.5,
            0.5,
            "Insufficient complete gait cycles for cycle-normalized analysis.",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=9,
        )
        ax.set_title("Knee flexion · Gait Cycle %", color=TEXT, fontsize=10, fontweight="medium", pad=6)
        return False

    src = "OpenSim IK" if cc.angle_source == "opensim_ik" else "Pose-derived"
    ax.set_xlabel("Gait Cycle (%)", color=MUTED, fontsize=8)
    ax.set_ylabel("Knee flexion (deg)", color=MUTED, fontsize=8)
    title = f"Knee flexion · Gait Cycle %  ({usable} cycles)"
    if cc.cycle_repeatability_score is not None:
        title += f"  ·  repeatability {cc.cycle_repeatability_score:.0f}"
    title += f"  ·  Angle Source: {src}"
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="medium", pad=6)
    ax.set_xlim(0, 100)

    finite = []
    for traj in (left, right):
        if traj is not None and len(traj.per_cycle) >= MIN_CYCLES_FOR_CYCLE_MODE:
            finite.extend(_to_flexion(traj.mean, cc.angle_source).tolist())
    if finite:
        y_lo, y_hi = float(np.min(finite)), float(np.max(finite))
        margin = max(5.0, (y_hi - y_lo) * 0.08)
        ax.set_ylim(y_lo - margin, y_hi + margin)

    _draw_cycle_phase_regions(ax)
    return True


def _draw_cycle_phase_regions(ax: Axes) -> None:
    """Subtle stance / swing bands on 0–100% axis (left-foot reference)."""
    ax.axvspan(0, 62, facecolor=BORDER, alpha=0.08, zorder=0, label="STANCE (typical)")
    ax.axvspan(62, 100, facecolor=BORDER, alpha=0.04, zorder=0, label="SWING (typical)")


def style_gait_chart(ax: Axes, fig) -> None:
    ax.set_facecolor(PANEL)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        dedup_h, dedup_l, seen = [], [], set()
        for h, lab in zip(handles, labels):
            if lab in seen:
                continue
            seen.add(lab)
            dedup_h.append(h)
            dedup_l.append(lab)
        ax.legend(
            dedup_h,
            dedup_l,
            facecolor=ELEVATED,
            edgecolor=BORDER,
            labelcolor=TEXT,
            fontsize=7,
            framealpha=0.92,
            loc="upper right",
        )
    event_legend = (
        "HS = Heel Strike   ·   TO = Toe Off",
    )
    ax.text(
        0.01,
        0.98,
        event_legend[0],
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=6,
        zorder=12,
    )
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=BORDER, alpha=0.35, linestyle="--", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    fig.tight_layout(pad=1.2)
