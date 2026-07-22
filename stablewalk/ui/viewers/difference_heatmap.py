"""Joint-angle difference heatmap for dual-session Compare Mode."""

from __future__ import annotations

from typing import Any

import numpy as np

from stablewalk.analysis.session_compare import joint_angle_difference_heatmap
from stablewalk.models.pose_data import PoseSequence
from stablewalk.ui.colors import MUTED, PANEL, PLAYHEAD, TEXT


def draw_difference_heatmap(
    ax: Any,
    left_sequence: PoseSequence | None,
    right_sequence: PoseSequence | None,
    *,
    progress: float | None = None,
    n_bins: int = 40,
) -> None:
    """
    Draw |angle_A − angle_B| over normalised time for key lower-limb joints.

    Darker / hotter cells indicate larger kinematic differences.
    """
    ax.cla()
    ax.set_facecolor(PANEL)
    matrix, labels = joint_angle_difference_heatmap(
        left_sequence, right_sequence, n_bins=n_bins
    )
    fig = ax.figure
    # Reserve margins so y-labels and colorbar never overlap the heatmap.
    try:
        fig.subplots_adjust(left=0.26, right=0.84, top=0.88, bottom=0.18)
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
            fontsize=9,
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
    ax.set_yticklabels(labels, fontsize=8, color=TEXT)
    ax.tick_params(axis="y", pad=4, length=0)
    ax.set_xticks([0, n_bins // 2, n_bins - 1])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=8, color=TEXT)
    ax.set_xlabel("Normalised time (synced)", fontsize=9, color=TEXT, labelpad=6)
    ax.set_title("|Joint angle| difference (°)", fontsize=9.5, color=TEXT, pad=6)
    if progress is not None and np.isfinite(progress):
        x = float(np.clip(progress, 0.0, 1.0)) * max(n_bins - 1, 1)
        ax.axvline(x, color=PLAYHEAD, linewidth=1.35, alpha=0.9)
    try:
        cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.06)
        cbar.ax.tick_params(labelsize=7, colors=TEXT)
        cbar.set_label("Δ°", fontsize=8, color=TEXT)
    except Exception:
        pass
