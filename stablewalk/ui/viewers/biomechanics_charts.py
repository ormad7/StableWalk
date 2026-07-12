"""Matplotlib charts for the Biomechanics dashboard tab."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.ui.colors import ACCENT, BORDER, MUTED, PANEL, TEXT, WARNING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _style(ax: Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=BORDER, alpha=0.35, linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def draw_biomechanics_dashboard(
    fig: Figure,
    result: BiomechanicalAnalysisResult | None,
    *,
    playhead_time_s: float | None = None,
) -> None:
    """Four-row biomechanics timeline: COM height, stability margin, symmetry proxy, gait quality."""
    fig.clear()
    fig.patch.set_facecolor(PANEL)
    axes = fig.subplots(4, 1, sharex=True)

    if result is None or result.center_of_mass is None or not result.center_of_mass.per_frame:
        for ax in axes:
            ax.text(0.5, 0.5, "No biomechanical data", transform=ax.transAxes, ha="center", color=MUTED)
            _style(ax)
        return

    t = result.center_of_mass.timestamps
    com_y = result.center_of_mass.positions[:, 1]

    axes[0].plot(t, com_y, color=ACCENT, linewidth=1.5, label="COM height (est.)")
    axes[0].set_ylabel("COM Y (m)", fontsize=7, color=MUTED)
    axes[0].set_title("Center of Mass (estimated)", color=TEXT, fontsize=9, pad=4)
    _style(axes[0])

    if result.stability_margin and result.stability_margin.per_frame:
        margins = [
            f.stability_margin_m if f.stability_margin_m is not None else np.nan
            for f in result.stability_margin.per_frame
        ]
        st = [f.stability_state for f in result.stability_margin.per_frame]
        ts = [f.time_s for f in result.stability_margin.per_frame]
        axes[1].plot(ts, margins, color=WARNING, linewidth=1.4)
        axes[1].axhline(0.04, color=ACCENT, linestyle=":", alpha=0.6, label="Stable threshold")
        axes[1].axhline(0.0, color=MUTED, linestyle="--", alpha=0.5)
        axes[1].set_ylabel("Margin (m)", fontsize=7, color=MUTED)
        axes[1].set_title("Stability Margin (derived)", color=TEXT, fontsize=9, pad=4)
        _style(axes[1])

    # Symmetry / cadence proxy over time from contact probabilities
    if result.contact and result.contact.per_frame:
        lp = result.contact.left_contact_probability
        rp = result.contact.right_contact_probability
        tc = result.contact.timestamps
        asym = np.abs(lp - rp)
        axes[2].plot(tc, asym, color=TEXT, linewidth=1.2, label="|L−R| contact P")
        axes[2].set_ylabel("Asymmetry", fontsize=7, color=MUTED)
        axes[2].set_title("Contact Timing Asymmetry (derived)", color=TEXT, fontsize=9, pad=4)
        _style(axes[2])

    if result.gait_metrics and result.gait_metrics.walking_speed:
        spd = result.gait_metrics.walking_speed.value
        axes[3].axhline(spd or 0, color=ACCENT, linewidth=2, label=f"Speed {spd:.2f} m/s" if spd else "Speed")
    if result.gait_quality:
        axes[3].text(
            0.02,
            0.85,
            f"Gait Quality: {result.gait_quality.score:.0f}/100",
            transform=axes[3].transAxes,
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
        axes[3].text(
            0.02,
            0.55,
            result.gait_quality.explanation[:120],
            transform=axes[3].transAxes,
            color=MUTED,
            fontsize=6,
            wrap=True,
        )
    axes[3].set_xlabel("Time (s)", fontsize=7, color=MUTED)
    axes[3].set_title("Gait Metrics Summary", color=TEXT, fontsize=9, pad=4)
    _style(axes[3])

    if playhead_time_s is not None:
        for ax in axes:
            ax.axvline(playhead_time_s, color=TEXT, alpha=0.5, linewidth=0.9)

    fig.tight_layout(pad=1.0, h_pad=0.7)


__all__ = ["draw_biomechanics_dashboard"]
