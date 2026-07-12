"""Matplotlib charts for foot contact, gait phases, and estimated vGRF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.ui.colors import ACCENT, BORDER, MUTED, PANEL, TEXT, WARNING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _style_axis(ax: Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(True, color=BORDER, alpha=0.35, linestyle="--", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(BORDER)


def draw_contact_timeline(
    ax: Axes,
    contact: FootContactAnalysisResult,
    *,
    playhead_time_s: float | None = None,
) -> None:
    """Left/right contact probability and binary masks with HS/TO markers."""
    if not contact.per_frame:
        ax.text(
            0.5, 0.5, "No contact data",
            transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=9,
        )
        _style_axis(ax)
        return

    t = contact.timestamps
    ax.fill_between(
        t,
        0,
        contact.left_contact_probability,
        color=ACCENT,
        alpha=0.25,
        label="Left P(contact)",
    )
    ax.fill_between(
        t,
        0,
        contact.right_contact_probability,
        color=WARNING,
        alpha=0.20,
        label="Right P(contact)",
    )
    ax.step(
        t,
        contact.left_contact_binary * 0.95,
        where="post",
        color=ACCENT,
        linewidth=1.4,
        label="Left contact",
    )
    ax.step(
        t,
        contact.right_contact_binary * 0.85,
        where="post",
        color=WARNING,
        linewidth=1.4,
        label="Right contact",
    )

    for side, color, hs_attr, to_attr in (
        ("L", ACCENT, "left_heel_strike", "left_toe_off"),
        ("R", WARNING, "right_heel_strike", "right_toe_off"),
    ):
        for i, frame in enumerate(contact.per_frame):
            if getattr(frame, hs_attr):
                ax.axvline(t[i], color=color, alpha=0.35, linewidth=0.8, linestyle=":")
                ax.scatter([t[i]], [1.02], marker="v", s=18, color=color, zorder=5)
                if i == 0 or not getattr(contact.per_frame[i - 1], hs_attr):
                    ax.text(t[i], 1.04, f"{side} HS", fontsize=6, color=color, ha="center")
            if getattr(frame, to_attr):
                ax.scatter([t[i]], [-0.08], marker="^", s=18, color=color, zorder=5)

    if playhead_time_s is not None:
        ax.axvline(playhead_time_s, color=TEXT, alpha=0.6, linewidth=1.0, linestyle="-")

    ax.set_ylim(-0.15, 1.12)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Contact", color=MUTED, fontsize=7)
    ax.set_title("Foot Contact Timeline", color=TEXT, fontsize=9, fontweight="medium", pad=4)
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=6, loc="upper right")
    _style_axis(ax)


def draw_gait_phase_timeline(ax: Axes, contact: FootContactAnalysisResult) -> None:
    """Macro gait phases: stance, swing, double support."""
    if not contact.per_frame:
        _style_axis(ax)
        return

    t = contact.timestamps
    phase_map = {"swing": 0.0, "stance": 0.5, "double_support": 1.0, "uncertain": 0.25}
    values = np.array([phase_map.get(f.macro_phase, 0.25) for f in contact.per_frame])

    ax.fill_between(t, 0, values, step="post", color=BORDER, alpha=0.35)
    ax.step(t, values, where="post", color=TEXT, linewidth=1.2)

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["Swing", "Stance", "Double support"], fontsize=6, color=MUTED)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Phase", color=MUTED, fontsize=7)
    ax.set_title("Gait Phase Timeline", color=TEXT, fontsize=9, fontweight="medium", pad=4)
    _style_axis(ax)


def draw_estimated_vgrf_chart(
    ax: Axes,
    vgrf: EstimatedVGRFResult,
    *,
    playhead_time_s: float | None = None,
    show_bw: bool = True,
) -> None:
    """Plot estimated vGRF — clearly labeled, not force-plate or PhysX."""
    if not vgrf.available or len(vgrf.timestamps) < 2:
        ax.text(
            0.5,
            0.5,
            "Estimated vGRF unavailable",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        _style_axis(ax)
        return

    t = vgrf.timestamps
    if show_bw:
        ax.plot(t, vgrf.left_vgrf_bw, color=ACCENT, linewidth=1.6, label="Left (BW)")
        ax.plot(t, vgrf.right_vgrf_bw, color=WARNING, linewidth=1.6, label="Right (BW)")
        ax.plot(
            t,
            vgrf.total_vgrf_vertical / max(vgrf.body_weight_n, 1e-6),
            color=TEXT,
            linewidth=1.0,
            alpha=0.7,
            linestyle="--",
            label="Total (BW)",
        )
        ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=0.8)
        ax.set_ylabel("Estimated vGRF (BW)", color=MUTED, fontsize=7)
    else:
        ax.plot(t, vgrf.left_vgrf_vertical, color=ACCENT, linewidth=1.6, label="Left (N)")
        ax.plot(t, vgrf.right_vgrf_vertical, color=WARNING, linewidth=1.6, label="Right (N)")
        ax.plot(
            t,
            vgrf.total_vgrf_vertical,
            color=TEXT,
            linewidth=1.0,
            alpha=0.7,
            linestyle="--",
            label="Total (N)",
        )
        ax.set_ylabel("Estimated vGRF (N)", color=MUTED, fontsize=7)

    if playhead_time_s is not None:
        ax.axvline(playhead_time_s, color=TEXT, alpha=0.6, linewidth=1.0)

    ax.set_xlim(t[0], t[-1])
    title = f"Estimated Virtual GRF ({vgrf.method_name})"
    ax.set_title(title, color=TEXT, fontsize=9, fontweight="medium", pad=4)
    ax.text(
        0.01,
        0.98,
        "Not force-plate or PhysX — pose-based estimate",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=6,
    )
    ax.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=6, loc="upper right")
    _style_axis(ax)


def draw_contact_gait_dashboard(
    fig: Figure,
    contact: FootContactAnalysisResult | None,
    vgrf: EstimatedVGRFResult | None,
    *,
    playhead_time_s: float | None = None,
) -> None:
    """Draw all three panels into a figure with 3 rows."""
    fig.clear()
    fig.patch.set_facecolor(PANEL)
    axes = fig.subplots(3, 1, sharex=True)
    if contact is not None:
        draw_contact_timeline(axes[0], contact, playhead_time_s=playhead_time_s)
        draw_gait_phase_timeline(axes[1], contact)
    else:
        for ax in axes[:2]:
            ax.text(0.5, 0.5, "No contact data", transform=ax.transAxes, ha="center", color=MUTED)
            _style_axis(ax)
    if vgrf is not None:
        draw_estimated_vgrf_chart(axes[2], vgrf, playhead_time_s=playhead_time_s)
    else:
        axes[2].text(0.5, 0.5, "No vGRF data", transform=axes[2].transAxes, ha="center", color=MUTED)
        _style_axis(axes[2])
    axes[2].set_xlabel("Time (s)", color=MUTED, fontsize=7)
    fig.tight_layout(pad=1.0, h_pad=0.8)


__all__ = [
    "draw_contact_gait_dashboard",
    "draw_contact_timeline",
    "draw_estimated_vgrf_chart",
    "draw_gait_phase_timeline",
]
