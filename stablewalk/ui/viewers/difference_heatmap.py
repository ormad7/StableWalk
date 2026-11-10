"""Joint-angle difference heatmap for dual-session Compare Mode."""

from __future__ import annotations

from typing import Any

import numpy as np

from stablewalk.analysis.session_compare import joint_angle_difference_heatmap
from stablewalk.models.pose_data import PoseSequence
from stablewalk.ui.colors import MUTED, PANEL, PLAYHEAD, TEXT
from stablewalk.ui.viewers.chart_style import (
    AXIS_LABEL_SIZE,
    TICK_LABEL_SIZE,
    TITLE_SIZE,
    configure_chart_antialiasing,
)


def draw_difference_heatmap(
    ax: Any,
    left_sequence: PoseSequence | None,
    right_sequence: PoseSequence | None,
    *,
    progress: float | None = None,
    n_bins: int = 40,
    xlabel: str | None = None,
) -> None:
    """
    Draw |angle_A − angle_B| over normalised time for key lower-limb joints.

    Darker / hotter cells indicate larger kinematic differences.
    """
    ax.cla()
    ax.set_facecolor(PANEL)
    configure_chart_antialiasing(ax.figure)
    matrix, labels = joint_angle_difference_heatmap(
        left_sequence, right_sequence, n_bins=n_bins
    )
    fig = ax.figure
    # Reserve margins so y-labels and colorbar never overlap the heatmap.
    try:
        fig.subplots_adjust(left=0.22, right=0.86, top=0.90, bottom=0.16)
    except Exception:
        pass

    if matrix.size == 0 or not np.isfinite(matrix).any():
        ax.text(
            0.5,
            0.5,
            "Difference heatmap unavailable\n(need joint angles in both sessions)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            color=MUTED,
            fontsize=12,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return

    data = np.ma.masked_invalid(matrix)
    im = ax.imshow(
        data,
        aspect="auto",
        origin="upper",
        cmap="YlOrRd",
        interpolation="nearest",
        vmin=0.0,
        vmax=max(float(np.nanmax(matrix)), 1.0),
    )
    ax.set_yticks(range(len(labels)))
    # Short labels + padding so ticks do not collide with the image edge.
    ax.set_yticklabels(labels, fontsize=TICK_LABEL_SIZE, color=TEXT)
    ax.tick_params(axis="y", pad=5, length=0)
    ax.set_xticks([0, n_bins // 2, n_bins - 1])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=TICK_LABEL_SIZE, color=TEXT)
    ax.set_xlabel(
        xlabel or "Video progress (%)",
        fontsize=AXIS_LABEL_SIZE,
        color=TEXT,
        labelpad=7,
    )
    ax.set_title("|Joint angle| difference (°)", fontsize=TITLE_SIZE, color=TEXT, pad=8)
    if progress is not None and np.isfinite(progress):
        x = float(np.clip(progress, 0.0, 1.0)) * max(n_bins - 1, 1)
        ax.axvline(x, color=PLAYHEAD, linewidth=1.55, alpha=0.92)
    try:
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.06)
        cbar.ax.tick_params(labelsize=10.5, colors=TEXT)
        cbar.set_label("Δ°", fontsize=AXIS_LABEL_SIZE, color=TEXT)
    except Exception:
        pass
