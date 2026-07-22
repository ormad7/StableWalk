"""Matplotlib charts for foot contact, gait phases, and estimated vGRF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.ui.colors import BORDER, MUTED, PANEL, SIDE_LEFT, SIDE_RIGHT, TEXT
from stablewalk.ui.scientific_labels import (
    CHART_GAIT_PHASE,
    CHART_VGRF,
    DISCLAIMER_VGRF,
    LABEL_FOOT_CONTACT_TIMELINE,
    LABEL_VIRTUAL_GRF_FULL,
    LABEL_VIRTUAL_GRF_SHORT,
    LABEL_VIRTUAL_GRF_UNAVAILABLE,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _style_axis(ax: Axes, *, y_minor: bool = True) -> None:
    from stablewalk.ui.viewers.chart_style import apply_chart_grid, apply_chart_panel_style

    apply_chart_panel_style(ax)
    ax.tick_params(colors=MUTED, labelsize=9)
    apply_chart_grid(ax, y_minor=y_minor)


def _draw_playhead(
    ax: Axes,
    playhead_time_s: float | None,
    *,
    show_label: bool = False,
    value_label: str | None = None,
    value_y: float | None = None,
) -> None:
    if playhead_time_s is None:
        return
    from stablewalk.ui.viewers.chart_playhead import PlayheadState, draw_chart_playhead

    draw_chart_playhead(
        ax,
        PlayheadState(time_s=float(playhead_time_s), frame_index=0),
        show_label=show_label,
        value_label=value_label,
        value_y=value_y,
    )


def _draw_pulse_events(
    ax: Axes,
    contact: FootContactAnalysisResult,
    t: np.ndarray,
) -> None:
    """Heel strike and toe-off pulse markers (OpenSim / Vicon style)."""
    from stablewalk.ui.viewers.chart_reference import draw_gait_event_markers

    left_hs, right_hs, left_to, right_to = [], [], [], []
    for i, frame in enumerate(contact.per_frame):
        if getattr(frame, "left_heel_strike", False):
            left_hs.append(float(t[i]))
        if getattr(frame, "right_heel_strike", False):
            right_hs.append(float(t[i]))
        if getattr(frame, "left_toe_off", False):
            left_to.append(float(t[i]))
        if getattr(frame, "right_toe_off", False):
            right_to.append(float(t[i]))
    draw_gait_event_markers(
        ax,
        left_hs=left_hs,
        right_hs=right_hs,
        left_to=left_to,
        right_to=right_to,
        show_legend=True,
    )


def _draw_substate_onsets(
    ax: Axes,
    contact: FootContactAnalysisResult,
    t: np.ndarray,
) -> None:
    """Foot flat and mid-stance onset markers from the sub-state machine."""
    specs = (
        ("left", "foot_flat", "L FF", SIDE_LEFT, "s", 0.78),
        ("right", "foot_flat", "R FF", SIDE_RIGHT, "s", 0.78),
        ("left", "mid_stance", "L MS", SIDE_LEFT, "D", 0.52),
        ("right", "mid_stance", "R MS", SIDE_RIGHT, "D", 0.52),
    )
    for side, substate, label, color, marker, y in specs:
        attr = f"{side}_foot_substate"
        for i, frame in enumerate(contact.per_frame):
            if getattr(frame, attr) != substate:
                continue
            if i > 0 and getattr(contact.per_frame[i - 1], attr) == substate:
                continue
            ax.scatter([t[i]], [y], marker=marker, s=16, color=color, alpha=0.9, zorder=5)


def draw_contact_timeline(
    ax: Axes,
    contact: FootContactAnalysisResult,
    *,
    playhead_time_s: float | None = None,
) -> None:
    """Left/right contact probability and binary masks with gait-event markers."""
    from stablewalk.ui.viewers.chart_reference import draw_confidence_overlay
    from stablewalk.ui.viewers.chart_style import style_chart_title

    if not contact.per_frame:
        ax.text(
            0.5, 0.5, "No contact data",
            transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10,
        )
        _style_axis(ax)
        return

    t = contact.timestamps
    ax.fill_between(
        t,
        0,
        contact.left_contact_probability,
        color=SIDE_LEFT,
        alpha=0.25,
        label="_nolegend_",
    )
    ax.fill_between(
        t,
        0,
        contact.right_contact_probability,
        color=SIDE_RIGHT,
        alpha=0.20,
        label="_nolegend_",
    )
    ax.step(
        t,
        contact.left_contact_binary * 0.95,
        where="post",
        color=SIDE_LEFT,
        linewidth=1.55,
        label="Left contact",
    )
    ax.step(
        t,
        contact.right_contact_binary * 0.85,
        where="post",
        color=SIDE_RIGHT,
        linewidth=1.55,
        label="Right contact",
    )

    conf = 0.5 * (
        np.asarray([float(f.left_confidence) for f in contact.per_frame], dtype=float)
        + np.asarray([float(f.right_confidence) for f in contact.per_frame], dtype=float)
    )
    draw_confidence_overlay(ax, t, conf, threshold=0.55)

    _draw_pulse_events(ax, contact, t)
    _draw_substate_onsets(ax, contact, t)

    value_label = None
    value_y = None
    if playhead_time_s is not None and len(t) >= 2:
        lp = float(np.interp(float(playhead_time_s), t, contact.left_contact_probability))
        rp = float(np.interp(float(playhead_time_s), t, contact.right_contact_probability))
        value_label = f"L {lp:.2f} · R {rp:.2f}"
        value_y = max(lp, rp)
    _draw_playhead(
        ax,
        playhead_time_s,
        show_label=True,
        value_label=value_label,
        value_y=value_y,
    )

    ax.set_ylim(-0.18, 1.12)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Contact", color=MUTED, fontsize=10.5)
    style_chart_title(ax, LABEL_FOOT_CONTACT_TIMELINE)
    ax.legend(
        facecolor=PANEL,
        edgecolor=BORDER,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.94,
        fancybox=False,
    )
    _style_axis(ax)


def draw_gait_phase_timeline(ax: Axes, contact: FootContactAnalysisResult) -> None:
    """Macro gait phases: stance, swing, double support."""
    from stablewalk.ui.viewers.chart_style import style_chart_title

    if not contact.per_frame:
        _style_axis(ax)
        return

    t = contact.timestamps
    phase_map = {"swing": 0.0, "stance": 0.5, "double_support": 1.0, "uncertain": 0.25}
    values = np.array([phase_map.get(f.macro_phase, 0.25) for f in contact.per_frame])

    ax.fill_between(t, 0, values, step="post", color=BORDER, alpha=0.35)
    ax.step(t, values, where="post", color=TEXT, linewidth=1.4)

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["Swing", "Stance", "Double support"], fontsize=8.5, color=MUTED)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-0.05, 1.15)
    ax.set_ylabel("Phase", color=MUTED, fontsize=10.5)
    style_chart_title(ax, CHART_GAIT_PHASE)
    _style_axis(ax, y_minor=False)


def draw_estimated_vgrf_chart(
    ax: Axes,
    vgrf: EstimatedVGRFResult,
    *,
    playhead_time_s: float | None = None,
    show_bw: bool = True,
) -> None:
    """Plot estimated virtual GRF — clearly labeled, not force-plate or PhysX."""
    from stablewalk.ui.viewers.chart_reference import VGRF_BW_NORMAL, draw_reference_y_bands
    from stablewalk.ui.viewers.chart_style import style_chart_title

    if not vgrf.available or len(vgrf.timestamps) < 2:
        ax.text(
            0.5,
            0.5,
            LABEL_VIRTUAL_GRF_UNAVAILABLE,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        _style_axis(ax)
        return

    t = vgrf.timestamps
    m = vgrf.metrics
    if show_bw:
        ax.plot(t, vgrf.left_vgrf_bw, color=SIDE_LEFT, linewidth=1.85, label=f"Left {LABEL_VIRTUAL_GRF_SHORT}")
        ax.plot(t, vgrf.right_vgrf_bw, color=SIDE_RIGHT, linewidth=1.85, label=f"Right {LABEL_VIRTUAL_GRF_SHORT}")
        ax.plot(
            t,
            vgrf.total_vgrf_vertical / max(vgrf.body_weight_n, 1e-6),
            color=TEXT,
            linewidth=1.15,
            alpha=0.75,
            linestyle="--",
            label=f"Total {LABEL_VIRTUAL_GRF_SHORT}",
        )
        ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=0.9, label="1 BW")
        ax.set_ylabel(f"{LABEL_VIRTUAL_GRF_FULL} (BW)", color=MUTED, fontsize=10.5)
        draw_reference_y_bands(
            ax,
            normal=VGRF_BW_NORMAL,
            abnormal_below=0.35,
            abnormal_above=1.6,
            label_normal=True,
        )
        summary = (
            f"Peak L {m.left_peak_force_bw:.2f} BW · R {m.right_peak_force_bw:.2f} BW · "
            f"Total {m.peak_force_bw:.2f} BW\n"
            f"Loading rate L {m.left_loading_rate_n_per_s:.0f} · R {m.right_loading_rate_n_per_s:.0f} · "
            f"Total {m.loading_rate_n_per_s:.0f} N/s"
        )
        value_series = vgrf.total_vgrf_vertical / max(vgrf.body_weight_n, 1e-6)
        value_unit = " BW"
    else:
        ax.plot(t, vgrf.left_vgrf_vertical, color=SIDE_LEFT, linewidth=1.85, label=f"Left {LABEL_VIRTUAL_GRF_SHORT}")
        ax.plot(t, vgrf.right_vgrf_vertical, color=SIDE_RIGHT, linewidth=1.85, label=f"Right {LABEL_VIRTUAL_GRF_SHORT}")
        ax.plot(
            t,
            vgrf.total_vgrf_vertical,
            color=TEXT,
            linewidth=1.15,
            alpha=0.75,
            linestyle="--",
            label=f"Total {LABEL_VIRTUAL_GRF_SHORT}",
        )
        ax.set_ylabel(f"{LABEL_VIRTUAL_GRF_FULL} (N)", color=MUTED, fontsize=10.5)
        summary = (
            f"Peak L {m.left_peak_force_n:.0f} N · R {m.right_peak_force_n:.0f} N · "
            f"Total {m.peak_force_n:.0f} N\n"
            f"Loading rate L {m.left_loading_rate_n_per_s:.0f} · R {m.right_loading_rate_n_per_s:.0f} · "
            f"Total {m.loading_rate_n_per_s:.0f} N/s"
        )
        value_series = vgrf.total_vgrf_vertical
        value_unit = " N"

    value_label = None
    value_y = None
    if playhead_time_s is not None and len(t) >= 2:
        value_y = float(np.interp(float(playhead_time_s), t, value_series))
        value_label = f"{value_y:.2f}{value_unit}"
    _draw_playhead(
        ax,
        playhead_time_s,
        show_label=False,
        value_label=value_label,
        value_y=value_y,
    )

    ax.set_xlim(t[0], t[-1])
    style_chart_title(ax, CHART_VGRF)
    ax.text(
        1.01,
        0.28,
        DISCLAIMER_VGRF,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=MUTED,
        fontsize=7,
        clip_on=False,
    )
    ax.text(
        1.01,
        0.62,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=TEXT,
        fontsize=7.5,
        clip_on=False,
    )
    ax.legend(
        facecolor=PANEL,
        edgecolor=BORDER,
        labelcolor=TEXT,
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.94,
        fancybox=False,
    )
    _style_axis(ax)


def _register_contact_hover_points(
    fig: Figure,
    contact: FootContactAnalysisResult | None,
    vgrf: EstimatedVGRFResult | None,
    axes,
) -> None:
    from stablewalk.ui.viewers.chart_hover import (
        ChartHoverPoint,
        append_line_hover_points,
        set_figure_hover_points,
    )

    points: list[ChartHoverPoint] = []
    if contact is not None and contact.per_frame:
        t = contact.timestamps
        frames = [int(f.frame_index) for f in contact.per_frame]
        append_line_hover_points(
            axes[0],
            t,
            contact.left_contact_probability,
            metric_name="Left contact probability",
            joint_name="Left foot",
            unit="",
            frame_indices=frames,
            timestamps=t,
            hover_points=points,
        )
        append_line_hover_points(
            axes[0],
            t,
            contact.right_contact_probability,
            metric_name="Right contact probability",
            joint_name="Right foot",
            unit="",
            frame_indices=frames,
            timestamps=t,
            hover_points=points,
        )
        phase_map = {"swing": 0.0, "stance": 0.5, "double_support": 1.0, "uncertain": 0.25}
        phase_vals = [phase_map.get(f.macro_phase, 0.25) for f in contact.per_frame]
        append_line_hover_points(
            axes[1],
            t,
            phase_vals,
            metric_name="Gait phase",
            joint_name="—",
            unit="",
            frame_indices=frames,
            timestamps=t,
            hover_points=points,
        )
    if vgrf is not None and vgrf.available and len(vgrf.timestamps) >= 2:
        append_line_hover_points(
            axes[2],
            vgrf.timestamps,
            vgrf.left_vgrf_bw,
            metric_name=f"Left {LABEL_VIRTUAL_GRF_SHORT}",
            joint_name="Left foot",
            unit="BW",
            timestamps=vgrf.timestamps,
            hover_points=points,
        )
        append_line_hover_points(
            axes[2],
            vgrf.timestamps,
            vgrf.right_vgrf_bw,
            metric_name=f"Right {LABEL_VIRTUAL_GRF_SHORT}",
            joint_name="Right foot",
            unit="BW",
            timestamps=vgrf.timestamps,
            hover_points=points,
        )
    set_figure_hover_points(fig, points)


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
        _draw_playhead(axes[1], playhead_time_s)
    else:
        for ax in axes[:2]:
            ax.text(
                0.5,
                0.55,
                "No contact data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=9,
            )
            ax.text(
                0.5,
                0.42,
                "Run Analyze to show foot contact, gait phase, and estimated vGRF.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=7.5,
            )
            _style_axis(ax)
    if vgrf is not None:
        draw_estimated_vgrf_chart(axes[2], vgrf, playhead_time_s=playhead_time_s)
    else:
        axes[2].text(0.5, 0.5, LABEL_VIRTUAL_GRF_UNAVAILABLE, transform=axes[2].transAxes, ha="center", color=MUTED)
        _style_axis(axes[2])

    from stablewalk.ui.viewers.chart_style import finalize_stacked_time_axes

    vgrf_ylabel = f"{LABEL_VIRTUAL_GRF_FULL} (BW)"
    if vgrf is not None and getattr(vgrf, "available", False):
        # Preserve units chosen by draw_estimated_vgrf_chart (BW default).
        ylab = axes[2].get_ylabel()
        if ylab:
            vgrf_ylabel = ylab
    finalize_stacked_time_axes(
        axes,
        (
            ("numeric", "Contact"),
            (
                "categorical",
                "Phase",
                [0.0, 0.5, 1.0],
                ["Swing", "Stance", "Double support"],
            ),
            ("numeric", vgrf_ylabel),
        ),
    )
    # Keep shared time label only on the bottom panel (finalize already does this).
    # Reserve a slim right rail for legends and vGRF evidence so no annotation
    # obscures the synchronized data or playhead.
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0), pad=1.0, h_pad=0.9)
    _register_contact_hover_points(fig, contact, vgrf, axes)


__all__ = [
    "draw_contact_gait_dashboard",
    "draw_contact_timeline",
    "draw_estimated_vgrf_chart",
    "draw_gait_phase_timeline",
]
