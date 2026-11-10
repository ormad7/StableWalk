"""Matplotlib charts for foot contact, gait phases, and estimated vGRF."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from stablewalk.analysis.estimated_vgrf_analysis import EstimatedVGRFResult
from stablewalk.analysis.foot_contact_analysis import (
    FOOT_PHASE_STANCE,
    FOOT_PHASE_TICK_LABELS,
    FOOT_PHASE_TICKS,
    FOOT_PHASE_UNKNOWN,
    FootContactAnalysisResult,
    per_foot_phase_codes,
)
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


CONTACT_UNKNOWN_REASON = (
    "Shaded: contact unknown — feet are not clearly airborne, so this is not "
    "confirmed flight. Estimated vGRF is withheld (not recorded as zero)."
)
VGRF_GAP_LEGEND = "Withheld (unknown contact — not zero)"
VGRF_GAP_STATUS = (
    "Breaks in a side's series are withheld samples: that foot's contact was "
    "unknown, so force is not recorded as 0."
)


def _style_axis(ax: Axes, *, y_minor: bool = True) -> None:
    from stablewalk.ui.viewers.chart_style import (
        _TICK_LABEL_SIZE,
        apply_chart_grid,
        apply_chart_panel_style,
    )

    apply_chart_panel_style(ax)
    ax.tick_params(colors=MUTED, labelsize=_TICK_LABEL_SIZE)
    apply_chart_grid(ax, y_minor=y_minor)


def _style_contact_legend(ax: Axes) -> None:
    """Place Contact/GRF series legends below the data band, not on the series."""
    from stablewalk.ui.viewers.chart_style import style_chart_legend

    style_chart_legend(
        ax,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.18),
        outside=True,
    )


def _draw_playhead(
    ax: Axes,
    playhead_time_s: float | None,
    *,
    show_label: bool = False,
    value_label: str | None = None,
    value_y: float | None = None,
    frame_index: int | None = None,
) -> None:
    if playhead_time_s is None:
        return
    from stablewalk.ui.viewers.chart_playhead import PlayheadState, draw_chart_playhead

    display = 0 if frame_index is None else int(frame_index)
    draw_chart_playhead(
        ax,
        PlayheadState(time_s=float(playhead_time_s), frame_index=display),
        show_label=show_label,
        value_label=value_label,
        value_y=value_y,
    )


def _bilateral_unknown_mask(contact: FootContactAnalysisResult) -> np.ndarray:
    """True where both feet are off and that is not confirmed flight."""
    n = len(contact.per_frame)
    if n == 0:
        return np.zeros(0, dtype=bool)
    left = contact.left_contact_binary.astype(bool)
    right = contact.right_contact_binary.astype(bool)
    left_known = getattr(contact, "left_contact_known_mask", np.ones(n, dtype=bool))
    right_known = getattr(contact, "right_contact_known_mask", np.ones(n, dtype=bool))
    if len(left_known) != n:
        left_known = np.ones(n, dtype=bool)
    if len(right_known) != n:
        right_known = np.ones(n, dtype=bool)
    return (~left) & (~right) & ~(left_known & right_known)


def _shade_unknown_contact(
    ax: Axes,
    t: np.ndarray,
    unknown: np.ndarray,
    *,
    label: str = "Contact unknown",
) -> bool:
    """Hatch intervals where contact/vGRF must not be read as zero."""
    if len(t) == 0 or len(unknown) != len(t) or not np.any(unknown):
        return False
    labeled = False
    for i in range(len(t) - 1):
        if not unknown[i]:
            continue
        ax.axvspan(
            float(t[i]),
            float(t[i + 1]),
            facecolor=MUTED,
            edgecolor="none",
            alpha=0.16,
            zorder=0,
            label=label if not labeled else None,
        )
        labeled = True
    if unknown[-1] and len(t) >= 2:
        dt = float(t[-1] - t[-2])
        ax.axvspan(
            float(t[-1]),
            float(t[-1] + max(dt, 1e-6)),
            facecolor=MUTED,
            edgecolor="none",
            alpha=0.16,
            zorder=0,
            label=label if not labeled else None,
        )
        labeled = True
    return labeled


def clear_contact_figure_notes(fig: Figure) -> None:
    """Drop leftover GRF/Contact figure captions when switching panels."""
    for txt in list(getattr(fig, "texts", ()) or ()):
        if getattr(txt, "_sw_vgrf_status", False) or getattr(txt, "_sw_panel_note", False):
            try:
                txt.remove()
            except Exception:
                pass
    fig._sw_panel_note = ""  # type: ignore[attr-defined]


def vgrf_status_note(
    vgrf: EstimatedVGRFResult | None,
    *,
    show_bw: bool = True,
    shaded: bool = False,
) -> str:
    """GRF-only force/peak caption — never used on the Contact tab."""
    if vgrf is None:
        return LABEL_VIRTUAL_GRF_UNAVAILABLE
    m = getattr(vgrf, "metrics", None)
    if m is None:
        status = DISCLAIMER_VGRF
    elif show_bw:
        status = (
            f"L peak {m.left_peak_force_bw:.2f} · R {m.right_peak_force_bw:.2f} BW"
            f"  ·  {DISCLAIMER_VGRF}"
        )
    else:
        status = (
            f"L peak {m.left_peak_force_n:.0f} · R {m.right_peak_force_n:.0f} N"
            f"  ·  {DISCLAIMER_VGRF}"
        )
    if shaded:
        status = f"{status}  ·  {VGRF_GAP_STATUS}"
    return status


def contact_status_note(contact: FootContactAnalysisResult | None) -> str:
    if contact is None:
        return "Run Analyze to show foot contact timing."
    return CONTACT_UNKNOWN_REASON


def contact_panel_note(
    panel: str,
    *,
    contact: FootContactAnalysisResult | None = None,
    vgrf: EstimatedVGRFResult | None = None,
) -> str:
    """Tab-specific caption for the dedicated note row outside the axes."""
    mode = (panel or "contact").strip().lower()
    if mode == "vgrf":
        shaded = False
        if vgrf is not None:
            left = getattr(vgrf, "left_vgrf_bw", None)
            if left is not None:
                arr = np.asarray(left, dtype=float)
                shaded = bool(arr.size) and bool(np.isnan(arr).any())
        return vgrf_status_note(vgrf, shaded=shaded)
    return contact_status_note(contact)


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
    playhead_frame: int | None = None,
) -> None:
    """Left/right contact probability and binary masks with gait-event markers."""
    from stablewalk.ui.viewers.chart_reference import draw_confidence_overlay
    from stablewalk.ui.viewers.chart_style import (
        LINESTYLE_LEFT,
        LINESTYLE_RIGHT,
        SERIES_LINE_WIDTH_STEP,
        _AXIS_LABEL_SIZE,
        style_chart_title,
    )

    if not contact.per_frame:
        ax.text(
            0.5, 0.5, "No contact data",
            transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=10,
        )
        _style_axis(ax)
        return

    t = contact.timestamps
    unknown = _bilateral_unknown_mask(contact)
    _shade_unknown_contact(ax, t, unknown)
    left_known = getattr(contact, "left_contact_known_mask", np.ones(len(t), dtype=bool))
    right_known = getattr(contact, "right_contact_known_mask", np.ones(len(t), dtype=bool))
    if len(left_known) != len(t):
        left_known = np.ones(len(t), dtype=bool)
    if len(right_known) != len(t):
        right_known = np.ones(len(t), dtype=bool)
    left_step = np.where(left_known, contact.left_contact_binary * 0.95, np.nan)
    right_step = np.where(right_known, contact.right_contact_binary * 0.85, np.nan)

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
        left_step,
        where="post",
        color=SIDE_LEFT,
        linewidth=SERIES_LINE_WIDTH_STEP,
        linestyle=LINESTYLE_LEFT,
        label="Left contact",
        antialiased=True,
    )
    ax.step(
        t,
        right_step,
        where="post",
        color=SIDE_RIGHT,
        linewidth=SERIES_LINE_WIDTH_STEP,
        linestyle=LINESTYLE_RIGHT,
        label="Right contact",
        antialiased=True,
    )

    conf = 0.5 * (
        np.asarray([float(f.left_confidence) for f in contact.per_frame], dtype=float)
        + np.asarray([float(f.right_confidence) for f in contact.per_frame], dtype=float)
    )
    draw_confidence_overlay(ax, t, conf, threshold=0.55)

    _draw_pulse_events(ax, contact, t)
    _draw_substate_onsets(ax, contact, t)

    from stablewalk.ui.viewers.chart_playhead import set_playhead_value_source

    left_p = np.asarray(contact.left_contact_probability, dtype=float)
    right_p = np.asarray(contact.right_contact_probability, dtype=float)
    marker_series = np.fmax(left_p, right_p)
    if len(unknown) == len(marker_series):
        marker_series = np.array(marker_series, dtype=float, copy=True)
        marker_series[np.asarray(unknown, dtype=bool)] = np.nan
    set_playhead_value_source(ax, t, marker_series, "contact", "")
    value_label = None
    value_y = None
    if playhead_time_s is not None and len(t) >= 1:
        idx = int(np.searchsorted(t, float(playhead_time_s), side="right") - 1)
        idx = max(0, min(idx, len(t) - 1))
        if not (unknown[idx] if idx < len(unknown) else False):
            lp = float(left_p[idx]) if idx < len(left_p) else float("nan")
            rp = float(right_p[idx]) if idx < len(right_p) else float("nan")
            if np.isfinite(lp) and np.isfinite(rp):
                value_label = f"L {lp:.2f} · R {rp:.2f}"
                value_y = max(lp, rp)
    _draw_playhead(
        ax,
        playhead_time_s,
        show_label=True,
        value_label=value_label,
        value_y=value_y,
        frame_index=playhead_frame,
    )

    ax.set_ylim(-0.18, 1.12)
    ax.set_xlim(t[0], t[-1])

    from stablewalk.ui.viewers.chart_reference import draw_reference_y_bands

    # Light normal band on contact probability (display cue for mid-range contact).
    draw_reference_y_bands(
        ax,
        normal=(0.4, 1.0),
        abnormal_below=None,
        abnormal_above=None,
        label_normal=True,
    )

    ax.set_ylabel("Contact", color=MUTED, fontsize=_AXIS_LABEL_SIZE)
    style_chart_title(ax, LABEL_FOOT_CONTACT_TIMELINE, pad=10.0)
    _style_contact_legend(ax)
    _style_axis(ax)


def draw_per_foot_phase_timelines(
    ax: Axes,
    contact: FootContactAnalysisResult,
    *,
    title: str | None = None,
    shade_unknown: bool = True,
) -> None:
    """Separate left/right categorical phase step plots, including Unknown."""
    from stablewalk.ui.viewers.chart_style import (
        LINESTYLE_LEFT,
        LINESTYLE_RIGHT,
        SERIES_LINE_WIDTH_STEP,
        _AXIS_LABEL_SIZE,
        _TICK_LABEL_SIZE,
        style_chart_title,
    )

    t = contact.timestamps
    left_codes, right_codes = per_foot_phase_codes(contact)
    unknown = (left_codes == FOOT_PHASE_UNKNOWN) | (right_codes == FOOT_PHASE_UNKNOWN)
    if shade_unknown:
        _shade_unknown_contact(ax, t, unknown, label="Unknown")

    ax.step(
        t,
        left_codes,
        where="post",
        color=SIDE_LEFT,
        linewidth=SERIES_LINE_WIDTH_STEP,
        linestyle=LINESTYLE_LEFT,
        label="Left foot",
        antialiased=True,
    )
    ax.step(
        t,
        right_codes,
        where="post",
        color=SIDE_RIGHT,
        linewidth=SERIES_LINE_WIDTH_STEP,
        linestyle=LINESTYLE_RIGHT,
        label="Right foot",
        antialiased=True,
    )
    ax.set_yticks(list(FOOT_PHASE_TICKS))
    ax.set_yticklabels(
        list(FOOT_PHASE_TICK_LABELS),
        fontsize=_TICK_LABEL_SIZE,
        color=MUTED,
    )
    ax.set_ylim(FOOT_PHASE_UNKNOWN - 0.25, FOOT_PHASE_STANCE + 0.35)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel("Phase", color=MUTED, fontsize=_AXIS_LABEL_SIZE)
    if title:
        style_chart_title(ax, title, pad=10.0)
    _style_axis(ax, y_minor=False)


def draw_gait_phase_timeline(ax: Axes, contact: FootContactAnalysisResult) -> None:
    """Per-foot categorical gait phases: Unknown, Swing, Stance."""
    if not contact.per_frame:
        _style_axis(ax)
        return
    draw_per_foot_phase_timelines(ax, contact, title=CHART_GAIT_PHASE)
    _style_contact_legend(ax)


def draw_estimated_vgrf_chart(
    ax: Axes,
    vgrf: EstimatedVGRFResult,
    *,
    playhead_time_s: float | None = None,
    playhead_frame: int | None = None,
    show_bw: bool = True,
    contact: FootContactAnalysisResult | None = None,
) -> None:
    """Plot estimated virtual GRF — clearly labeled, not force-plate or PhysX."""
    from stablewalk.ui.viewers.biomech_chart_enrichment import (
        draw_gait_events_from_contact,
        enrich_numeric_series,
    )
    from stablewalk.ui.viewers.chart_reference import VGRF_BW_NORMAL, draw_reference_y_bands
    from stablewalk.ui.viewers.chart_style import (
        SERIES_LINE_WIDTH_SECONDARY,
        _AXIS_LABEL_SIZE,
        series_plot_kwargs,
        style_chart_title,
    )

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
    left_series = np.asarray(
        vgrf.left_vgrf_bw if show_bw else vgrf.left_vgrf_vertical, dtype=float
    )
    right_series = np.asarray(
        vgrf.right_vgrf_bw if show_bw else vgrf.right_vgrf_vertical, dtype=float
    )
    missing = np.zeros(len(t), dtype=bool)
    if len(left_series) == len(t):
        missing |= ~np.isfinite(left_series)
    if len(right_series) == len(t):
        missing |= ~np.isfinite(right_series)
    if contact is not None and contact.per_frame:
        bilateral = _bilateral_unknown_mask(contact)
        if len(bilateral) == len(t):
            missing |= bilateral
    shaded = (
        _shade_unknown_contact(ax, t, missing, label=VGRF_GAP_LEGEND)
        if np.any(missing)
        else False
    )
    if show_bw:
        ax.plot(
            t,
            vgrf.left_vgrf_bw,
            **series_plot_kwargs(
                color=SIDE_LEFT,
                label=f"Left {LABEL_VIRTUAL_GRF_SHORT}",
                side="left",
            ),
        )
        ax.plot(
            t,
            vgrf.right_vgrf_bw,
            **series_plot_kwargs(
                color=SIDE_RIGHT,
                label=f"Right {LABEL_VIRTUAL_GRF_SHORT}",
                side="right",
            ),
        )
        ax.plot(
            t,
            vgrf.total_vgrf_vertical / max(vgrf.body_weight_n, 1e-6),
            **series_plot_kwargs(
                color=TEXT,
                label=f"Total {LABEL_VIRTUAL_GRF_SHORT}",
                linewidth=SERIES_LINE_WIDTH_SECONDARY,
                linestyle="--",
                alpha=0.75,
            ),
        )
        ax.axhline(1.0, color=MUTED, linestyle=":", linewidth=0.9, label="1 BW")
        ax.set_ylabel(f"{LABEL_VIRTUAL_GRF_FULL} (BW)", color=MUTED, fontsize=_AXIS_LABEL_SIZE)
        draw_reference_y_bands(
            ax,
            normal=VGRF_BW_NORMAL,
            abnormal_below=0.35,
            abnormal_above=1.6,
            label_normal=True,
        )
        value_series = vgrf.total_vgrf_vertical / max(vgrf.body_weight_n, 1e-6)
        value_unit = " BW"
        enrich_numeric_series(
            ax,
            t,
            value_series,
            unit="BW",
            peak_color=TEXT,
            stats_loc="upper left",
        )
        draw_gait_events_from_contact(ax, contact, show_legend=False)
    else:
        ax.plot(
            t,
            vgrf.left_vgrf_vertical,
            **series_plot_kwargs(
                color=SIDE_LEFT,
                label=f"Left {LABEL_VIRTUAL_GRF_SHORT}",
                side="left",
            ),
        )
        ax.plot(
            t,
            vgrf.right_vgrf_vertical,
            **series_plot_kwargs(
                color=SIDE_RIGHT,
                label=f"Right {LABEL_VIRTUAL_GRF_SHORT}",
                side="right",
            ),
        )
        ax.plot(
            t,
            vgrf.total_vgrf_vertical,
            **series_plot_kwargs(
                color=TEXT,
                label=f"Total {LABEL_VIRTUAL_GRF_SHORT}",
                linewidth=SERIES_LINE_WIDTH_SECONDARY,
                linestyle="--",
                alpha=0.75,
            ),
        )
        ax.set_ylabel(f"{LABEL_VIRTUAL_GRF_FULL} (N)", color=MUTED, fontsize=_AXIS_LABEL_SIZE)
        value_series = vgrf.total_vgrf_vertical
        value_unit = " N"
        enrich_numeric_series(
            ax,
            t,
            value_series,
            unit="N",
            peak_color=TEXT,
            stats_loc="upper left",
        )
        draw_gait_events_from_contact(ax, contact, show_legend=False)

    from stablewalk.ui.viewers.chart_playhead import set_playhead_value_source

    unit = str(value_unit).strip()
    set_playhead_value_source(ax, t, value_series, "vGRF", unit)
    value_label = None
    value_y = None
    if playhead_time_s is not None and len(t) >= 1:
        idx = int(np.searchsorted(t, float(playhead_time_s), side="right") - 1)
        idx = max(0, min(idx, len(t) - 1))
        raw = float(value_series[idx])
        if np.isfinite(raw):
            value_y = raw
            value_label = f"{value_y:.2f}{value_unit}"
    _draw_playhead(
        ax,
        playhead_time_s,
        show_label=False,
        value_label=value_label,
        value_y=value_y,
        frame_index=playhead_frame,
    )

    ax.set_xlim(t[0], t[-1])
    style_chart_title(ax, CHART_VGRF, pad=10.0)
    fig = ax.figure
    clear_contact_figure_notes(fig)
    fig._sw_panel_note = vgrf_status_note(vgrf, show_bw=show_bw, shaded=shaded)  # type: ignore[attr-defined]
    _style_contact_legend(ax)
    _style_axis(ax)


def _register_contact_hover_points(
    fig: Figure,
    contact: FootContactAnalysisResult | None,
    vgrf: EstimatedVGRFResult | None,
    axes,
    *,
    panel: str | None = None,
) -> None:
    from stablewalk.ui.viewers.chart_hover import (
        ChartHoverPoint,
        append_line_hover_points,
        set_figure_hover_points,
    )

    points: list[ChartHoverPoint] = []
    mode = (panel or CONTACT_PANEL_STACKED).strip().lower()
    ax_list = list(axes) if axes is not None else []
    if not ax_list:
        set_figure_hover_points(fig, points)
        return

    if contact is not None and contact.per_frame and mode in (
        CONTACT_PANEL_STACKED,
        CONTACT_PANEL_CONTACT,
    ):
        t = contact.timestamps
        frames = [int(f.frame_index) for f in contact.per_frame]
        list_positions = list(range(len(contact.per_frame)))
        append_line_hover_points(
            ax_list[0],
            t,
            contact.left_contact_probability,
            metric_name="Left contact probability",
            joint_name="Left foot",
            unit="",
            frame_indices=frames,
            list_positions=list_positions,
            timestamps=t,
            hover_points=points,
        )
        append_line_hover_points(
            ax_list[0],
            t,
            contact.right_contact_probability,
            metric_name="Right contact probability",
            joint_name="Right foot",
            unit="",
            frame_indices=frames,
            list_positions=list_positions,
            timestamps=t,
            hover_points=points,
        )
        if mode == CONTACT_PANEL_STACKED and len(ax_list) > 1:
            left_codes, right_codes = per_foot_phase_codes(contact)
            append_line_hover_points(
                ax_list[1],
                t,
                left_codes.astype(float),
                metric_name="Left foot phase",
                joint_name="Left foot",
                unit="",
                frame_indices=frames,
                list_positions=list_positions,
                timestamps=t,
                hover_points=points,
            )
            append_line_hover_points(
                ax_list[1],
                t,
                right_codes.astype(float),
                metric_name="Right foot phase",
                joint_name="Right foot",
                unit="",
                frame_indices=frames,
                list_positions=list_positions,
                timestamps=t,
                hover_points=points,
            )
    if vgrf is not None and vgrf.available and len(vgrf.timestamps) >= 2:
        if mode == CONTACT_PANEL_VGRF:
            vgrf_ax = ax_list[0]
        elif mode == CONTACT_PANEL_STACKED and len(ax_list) > 2:
            vgrf_ax = ax_list[2]
        else:
            vgrf_ax = None
        if vgrf_ax is not None:
            append_line_hover_points(
                vgrf_ax,
                vgrf.timestamps,
                vgrf.left_vgrf_bw,
                metric_name=f"Left {LABEL_VIRTUAL_GRF_SHORT}",
                joint_name="Left foot",
                unit="BW",
                timestamps=vgrf.timestamps,
                hover_points=points,
            )
            append_line_hover_points(
                vgrf_ax,
                vgrf.timestamps,
                vgrf.right_vgrf_bw,
                metric_name=f"Right {LABEL_VIRTUAL_GRF_SHORT}",
                joint_name="Right foot",
                unit="BW",
                timestamps=vgrf.timestamps,
                hover_points=points,
            )
    set_figure_hover_points(fig, points)


CONTACT_PANEL_CONTACT = "contact"
CONTACT_PANEL_VGRF = "vgrf"
CONTACT_PANEL_STACKED = "stacked"


def ensure_contact_gait_axes(fig: Figure):
    """Create or reuse a 3-row shared-x contact / phase / vGRF grid."""
    if len(fig.axes) == 3:
        return list(fig.axes)
    fig.clear()
    axes = fig.subplots(3, 1, sharex=True)
    return list(axes)


def ensure_contact_single_axes(fig: Figure):
    """Create or reuse one full-size axes for Contact or GRF tabs."""
    if len(fig.axes) == 1:
        return fig.axes[0]
    fig.clear()
    return fig.add_subplot(1, 1, 1)


def relayout_contact_gait_figure(fig: Figure) -> None:
    """Reapply responsive single-panel margins after its canvas is resized."""
    from stablewalk.ui.viewers.chart_style import apply_presentation_layout

    apply_presentation_layout(fig, mode="single")
    width_px = float(fig.get_figwidth() * fig.dpi)
    height_px = float(fig.get_figheight() * fig.dpi)
    top = 0.90 if height_px < 520 else 0.92
    bottom = 0.24 if height_px < 520 else 0.20
    left = 0.14 if width_px < 680 else 0.10
    try:
        fig.subplots_adjust(left=left, right=0.97, top=top, bottom=bottom)
    except Exception:
        pass


def draw_contact_gait_panel(
    fig: Figure,
    contact: FootContactAnalysisResult | None,
    vgrf: EstimatedVGRFResult | None,
    *,
    panel: str = CONTACT_PANEL_CONTACT,
    playhead_time_s: float | None = None,
    playhead_frame: int | None = None,
) -> None:
    """Draw one large Contact or vGRF chart (no stacked compression)."""
    mode = (panel or CONTACT_PANEL_CONTACT).strip().lower()
    if mode not in (CONTACT_PANEL_CONTACT, CONTACT_PANEL_VGRF):
        mode = CONTACT_PANEL_CONTACT

    fig.patch.set_facecolor(PANEL)
    clear_contact_figure_notes(fig)
    ax = ensure_contact_single_axes(fig)
    ax.clear()
    if hasattr(ax, "_stablewalk_playhead_artists"):
        try:
            delattr(ax, "_stablewalk_playhead_artists")
        except Exception:
            ax._stablewalk_playhead_artists = None  # type: ignore[attr-defined]
    fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]

    if mode == CONTACT_PANEL_VGRF:
        if vgrf is not None:
            draw_estimated_vgrf_chart(
                ax,
                vgrf,
                playhead_time_s=playhead_time_s,
                playhead_frame=playhead_frame,
                contact=contact,
            )
        else:
            ax.text(
                0.5,
                0.5,
                LABEL_VIRTUAL_GRF_UNAVAILABLE,
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=12,
            )
            _style_axis(ax)
        ax.set_xlabel("Time (s)", color=MUTED)
    else:
        if contact is not None:
            draw_contact_timeline(
                ax,
                contact,
                playhead_time_s=playhead_time_s,
                playhead_frame=playhead_frame,
            )
        else:
            ax.text(
                0.5,
                0.55,
                "No contact data yet",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=12,
            )
            ax.text(
                0.5,
                0.42,
                "Run Analyze to show foot contact timing.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10.5,
            )
            _style_axis(ax)
        ax.set_xlabel("Time (s)", color=MUTED)

    fig._sw_panel_note = contact_panel_note(  # type: ignore[attr-defined]
        mode, contact=contact, vgrf=vgrf
    )
    relayout_contact_gait_figure(fig)
    _register_contact_hover_points(fig, contact, vgrf, [ax], panel=mode)


def draw_contact_gait_dashboard(
    fig: Figure,
    contact: FootContactAnalysisResult | None,
    vgrf: EstimatedVGRFResult | None,
    *,
    playhead_time_s: float | None = None,
    playhead_frame: int | None = None,
    panel: str | None = None,
) -> None:
    """Draw contact / phase / vGRF — stacked (legacy) or one large panel."""
    mode = (panel or CONTACT_PANEL_STACKED).strip().lower()
    if mode in (CONTACT_PANEL_CONTACT, CONTACT_PANEL_VGRF):
        draw_contact_gait_panel(
            fig,
            contact,
            vgrf,
            panel=mode,
            playhead_time_s=playhead_time_s,
            playhead_frame=playhead_frame,
        )
        return

    fig.patch.set_facecolor(PANEL)
    axes = ensure_contact_gait_axes(fig)
    for ax in axes:
        ax.clear()
        # Drop stale playhead blit cache when static series are rebuilt.
        if hasattr(ax, "_stablewalk_playhead_artists"):
            try:
                delattr(ax, "_stablewalk_playhead_artists")
            except Exception:
                ax._stablewalk_playhead_artists = None  # type: ignore[attr-defined]
    fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]

    if contact is not None:
        draw_contact_timeline(
            axes[0],
            contact,
            playhead_time_s=playhead_time_s,
            playhead_frame=playhead_frame,
        )
        draw_gait_phase_timeline(axes[1], contact)
        _draw_playhead(axes[1], playhead_time_s, frame_index=playhead_frame)
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
                fontsize=12,
            )
            ax.text(
                0.5,
                0.42,
                "Run Analyze to show foot contact, gait phase, and estimated vGRF.",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10.5,
            )
            _style_axis(ax)
    if vgrf is not None:
        draw_estimated_vgrf_chart(
            axes[2],
            vgrf,
            playhead_time_s=playhead_time_s,
            playhead_frame=playhead_frame,
        )
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
                list(FOOT_PHASE_TICKS),
                list(FOOT_PHASE_TICK_LABELS),
            ),
            ("numeric", vgrf_ylabel),
        ),
    )
    # Keep shared time label only on the bottom panel (finalize already does this).
    # Larger data band + slim outside-legend rail for projector readability.
    from stablewalk.ui.viewers.chart_style import apply_presentation_layout

    apply_presentation_layout(fig, mode="contact")
    _register_contact_hover_points(
        fig, contact, vgrf, axes, panel=CONTACT_PANEL_STACKED
    )


__all__ = [
    "CONTACT_UNKNOWN_REASON",
    "CONTACT_PANEL_CONTACT",
    "CONTACT_PANEL_STACKED",
    "CONTACT_PANEL_VGRF",
    "VGRF_GAP_LEGEND",
    "VGRF_GAP_STATUS",
    "clear_contact_figure_notes",
    "contact_panel_note",
    "contact_status_note",
    "vgrf_status_note",
    "draw_contact_gait_dashboard",
    "draw_contact_gait_panel",
    "draw_contact_timeline",
    "draw_estimated_vgrf_chart",
    "draw_gait_phase_timeline",
    "draw_per_foot_phase_timelines",
]
