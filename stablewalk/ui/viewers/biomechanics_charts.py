"""Matplotlib charts for the Biomechanics dashboard tab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.patches import Patch

from stablewalk.analysis.biomechanical.orchestrator import BiomechanicalAnalysisResult
from stablewalk.ui.scientific_labels import (
    CHART_COM_HEIGHT,
    CHART_CONTACT_EVENTS,
    CHART_GAIT_METRICS,
    CHART_STABILITY_MARGIN,
    LABEL_COM_SHORT,
)
from stablewalk.ui.colors import (
    BORDER,
    COM,
    ELEVATED,
    MUTED,
    PANEL,
    SIDE_LEFT,
    SIDE_RIGHT,
    STABILITY_REDUCED,
    STABILITY_STABLE,
    STABILITY_UNSTABLE,
    TEXT,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _HoverPoint:
    ax: Axes
    x: float
    y: float
    text: str


def _style(ax: Axes) -> None:
    from stablewalk.ui.viewers.chart_style import apply_chart_grid, apply_chart_panel_style

    apply_chart_panel_style(ax)
    ax.tick_params(colors=MUTED, labelsize=9)
    apply_chart_grid(ax, y_minor=True)


def _draw_synced_playhead(
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


def _stability_color(state: str) -> str:
    if state == "Stable":
        return STABILITY_STABLE
    if state == "Reduced Stability":
        return STABILITY_REDUCED
    return STABILITY_UNSTABLE


def _stability_legend_label(state: str) -> str:
    if state == "Reduced Stability":
        return "Moderate"
    return state


def _deduped_legend(
    ax: Axes,
    *,
    loc: str = "upper right",
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and not str(label).startswith("_") and label not in by_label:
            by_label[label] = handle
    if not by_label:
        return
    kwargs: dict = {
        "loc": loc,
        "ncol": ncol,
        "fontsize": 7.5,
        "frameon": True,
        "fancybox": False,
        "edgecolor": BORDER,
        "facecolor": PANEL,
        "labelcolor": TEXT,
        "framealpha": 0.92,
        "borderpad": 0.3,
        "handlelength": 1.5,
        "handletextpad": 0.4,
        "labelspacing": 0.28,
    }
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
        kwargs["borderaxespad"] = 0.0
    leg = ax.legend(list(by_label.values()), list(by_label.keys()), **kwargs)
    if leg is not None:
        leg.get_frame().set_linewidth(0.7)


def _stability_state_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor=STABILITY_STABLE, edgecolor=STABILITY_STABLE, alpha=0.65, label="Stable"),
        Patch(facecolor=STABILITY_REDUCED, edgecolor=STABILITY_REDUCED, alpha=0.65, label="Moderate"),
        Patch(facecolor=STABILITY_UNSTABLE, edgecolor=STABILITY_UNSTABLE, alpha=0.65, label="Unstable"),
    ]


def _draw_stability_state_bands(ax: Axes, times: list[float], states: list[str]) -> None:
    """Background bands colored by per-frame stability classification."""
    if len(times) < 2 or len(states) != len(times):
        return
    dt = np.diff(times)
    default_dt = float(np.median(dt)) if len(dt) else 0.033
    for i, state in enumerate(states):
        t0 = times[i]
        t1 = times[i + 1] if i + 1 < len(times) else t0 + default_dt
        ax.axvspan(t0, t1, color=_stability_color(state), alpha=0.14, zorder=0)


def _draw_gait_event_vlines(
    ax: Axes,
    result: BiomechanicalAnalysisResult,
    *,
    show_legend: bool = False,
) -> None:
    """Heel-strike and toe-off markers shared across timeline rows."""
    from stablewalk.ui.viewers.chart_reference import draw_gait_event_markers

    contact = result.contact
    if contact is None or not contact.per_frame:
        return
    t = contact.timestamps
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
        show_legend=show_legend,
    )


def _draw_contact_event_markers(ax: Axes, result: BiomechanicalAnalysisResult) -> None:
    """Gait-event markers on the contact-asymmetry timeline."""
    contact = result.contact
    if contact is None or not contact.per_frame:
        return
    t = contact.timestamps
    event_specs = (
        ("left_heel_strike", "L HS", SIDE_LEFT, "v", 1.06),
        ("right_heel_strike", "R HS", SIDE_RIGHT, "v", 1.06),
        ("left_toe_off", "L TO", SIDE_LEFT, "^", -0.06),
        ("right_toe_off", "R TO", SIDE_RIGHT, "^", -0.06),
    )
    substate_specs = (
        ("left", "foot_flat", "L FF", SIDE_LEFT, "s", 0.72),
        ("right", "foot_flat", "R FF", SIDE_RIGHT, "s", 0.72),
        ("left", "mid_stance", "L MS", SIDE_LEFT, "D", 0.48),
        ("right", "mid_stance", "R MS", SIDE_RIGHT, "D", 0.48),
    )

    for attr, label, color, marker, y in event_specs:
        labeled = False
        for i, frame in enumerate(contact.per_frame):
            if not getattr(frame, attr):
                continue
            ax.scatter(
                [t[i]],
                [y],
                marker=marker,
                s=28,
                color=color,
                zorder=6,
                label=label if not labeled else "_nolegend_",
            )
            labeled = True

    for side, substate, label, color, marker, y in substate_specs:
        attr = f"{side}_foot_substate"
        for i, frame in enumerate(contact.per_frame):
            if getattr(frame, attr) != substate:
                continue
            if i > 0 and getattr(contact.per_frame[i - 1], attr) == substate:
                continue
            ax.scatter([t[i]], [y], marker=marker, s=18, color=color, alpha=0.85, zorder=5)


def _plot_stability_margin_series(
    ax: Axes,
    ts: list[float],
    margins: list[float],
    states: list[str],
    hover_points: list[_HoverPoint],
) -> None:
    """Color-coded stability margin trace with per-point hover metadata."""
    if len(ts) < 2:
        return
    for i in range(len(ts) - 1):
        y0 = margins[i]
        y1 = margins[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        ax.plot(
            [ts[i], ts[i + 1]],
            [y0, y1],
            color=_stability_color(states[i]),
            linewidth=1.8,
            zorder=3,
            solid_capstyle="round",
        )
    for ti, mi, state in zip(ts, margins, states):
        if not np.isfinite(mi):
            continue
        ax.scatter([ti], [mi], color=_stability_color(state), s=18, zorder=4, edgecolors=PANEL, linewidths=0.4)
        hover_points.append(
            _HoverPoint(
                ax=ax,
                x=float(ti),
                y=float(mi),
                text=(
                    f"t = {ti:.3f} s\n"
                    f"Margin = {mi:.4f} m\n"
                    f"State = {_stability_legend_label(state)}"
                ),
            )
        )


def _draw_gait_phase_row(
    ax: Axes,
    result: BiomechanicalAnalysisResult,
    hover_points: list[_HoverPoint],
) -> None:
    """Macro gait phase timeline with session summary annotations."""
    contact = result.contact
    if contact is None or not contact.per_frame:
        ax.text(
            0.5,
            0.5,
            "No gait phase data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        return

    t = contact.timestamps
    phase_map = {"swing": 0.0, "stance": 0.5, "double_support": 1.0, "uncertain": 0.25}
    values = np.array([phase_map.get(f.macro_phase, 0.25) for f in contact.per_frame])

    ax.fill_between(
        t,
        0,
        values,
        step="post",
        color=BORDER,
        alpha=0.35,
        label="Macro phase envelope",
    )
    ax.step(t, values, where="post", color=TEXT, linewidth=1.4, label="Macro phase")

    phase_labels = {0.0: "Swing", 0.5: "Stance", 1.0: "Double support", 0.25: "Uncertain"}
    for ti, val, frame in zip(t, values, contact.per_frame):
        hover_points.append(
            _HoverPoint(
                ax=ax,
                x=float(ti),
                y=float(val),
                text=(
                    f"t = {float(ti):.3f} s\n"
                    f"Phase = {phase_labels.get(float(val), frame.macro_phase)}\n"
                    f"L contact = {frame.left_contact_binary}\n"
                    f"R contact = {frame.right_contact_binary}"
                ),
            )
        )

    if result.gait_quality:
        ax.text(
            0.98,
            0.95,
            f"Composite Gait Quality: {result.gait_quality.score:.0f}/100",
            transform=ax.transAxes,
            color=TEXT,
            fontsize=8,
            fontweight="bold",
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc=PANEL, ec=BORDER, alpha=0.92),
        )
        ax.text(
            0.98,
            0.78,
            result.gait_quality.explanation[:120],
            transform=ax.transAxes,
            color=MUTED,
            fontsize=6,
            ha="right",
            va="top",
            wrap=True,
            bbox=dict(boxstyle="round,pad=0.2", fc=PANEL, ec=BORDER, alpha=0.85),
        )

    if result.gait_metrics and result.gait_metrics.walking_speed:
        from stablewalk.analysis.biomechanical.walking_speed import format_walking_speed_display

        ws = result.gait_metrics.walking_speed
        label = format_walking_speed_display(ws)
        if label and "Not available" not in label:
            ax.text(
                0.62,
                0.92,
                f"Walking speed: {label}",
                transform=ax.transAxes,
                color=SIDE_LEFT,
                fontsize=8,
                va="top",
            )

    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["Swing", "Stance", "Double support"], fontsize=7, color=MUTED)
    ax.set_ylim(-0.08, 1.18)
    ax.set_ylabel("Phase (—)", fontsize=8, color=MUTED)


def _register_line_hover_points(
    ax: Axes,
    times: np.ndarray,
    values: np.ndarray,
    *,
    series_label: str,
    unit: str,
    hover_points: list[_HoverPoint],
) -> None:
    for ti, val in zip(times, values):
        if not np.isfinite(val):
            continue
        hover_points.append(
            _HoverPoint(
                ax=ax,
                x=float(ti),
                y=float(val),
                text=f"t = {float(ti):.3f} s\n{series_label} = {float(val):.4f} {unit}",
            )
        )


def draw_biomechanics_dashboard(
    fig: Figure,
    result: BiomechanicalAnalysisResult | None,
    *,
    playhead_time_s: float | None = None,
) -> None:
    """Four-row biomechanics timeline: COM height, stability margin, symmetry proxy, gait phase."""
    fig.clear()
    fig.patch.set_facecolor(PANEL)
    hover_points: list[_HoverPoint] = []

    axes = fig.subplots(
        4,
        1,
        sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.35, 1.15, 1.0], "hspace": 0.38},
    )

    if result is None or result.center_of_mass is None or not result.center_of_mass.per_frame:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No biomechanical data",
                transform=ax.transAxes,
                ha="center",
                color=MUTED,
            )
            _style(ax)
        fig._biomech_hover_points = []
        from stablewalk.ui.viewers.chart_hover import set_figure_hover_points

        set_figure_hover_points(fig, [])
        return

    t = result.center_of_mass.timestamps
    com_y = result.center_of_mass.positions[:, 1]

    from stablewalk.ui.viewers.chart_reference import (
        COM_HEIGHT_ABNORMAL_ABOVE_BH,
        COM_HEIGHT_ABNORMAL_BELOW_BH,
        COM_HEIGHT_NORMAL_BH,
        CONTACT_ASYMMETRY_NORMAL,
        STABILITY_MARGIN_NORMAL_M,
        draw_confidence_overlay,
        draw_reference_y_bands,
    )
    from stablewalk.ui.viewers.chart_style import style_chart_title

    axes[0].plot(
        t,
        com_y,
        color=COM,
        linewidth=2.1,
        label=f"{LABEL_COM_SHORT} height",
        zorder=3,
    )
    # Hip-centered / body-normalized Y — not absolute meters above the floor.
    _register_line_hover_points(
        axes[0],
        t,
        com_y,
        series_label="COM height (body-normalized)",
        unit="BH",
        hover_points=hover_points,
    )
    conf_com = np.asarray(
        [float(getattr(f, "confidence", 1.0)) for f in result.center_of_mass.per_frame],
        dtype=float,
    )
    if conf_com.size == len(t):
        draw_confidence_overlay(axes[0], t, conf_com, threshold=0.55)
    axes[0].set_ylabel("Height (hip-relative BH)", fontsize=10.5, color=MUTED)
    style_chart_title(axes[0], CHART_COM_HEIGHT)
    y0 = float(np.nanmin(com_y)) if np.isfinite(com_y).any() else COM_HEIGHT_NORMAL_BH[0]
    y1 = float(np.nanmax(com_y)) if np.isfinite(com_y).any() else COM_HEIGHT_NORMAL_BH[1]
    span = max(y1 - y0, 0.02)
    axes[0].set_ylim(min(y0, COM_HEIGHT_NORMAL_BH[0]) - span * 0.12, max(y1, COM_HEIGHT_NORMAL_BH[1]) + span * 0.12)
    draw_reference_y_bands(
        axes[0],
        normal=COM_HEIGHT_NORMAL_BH,
        abnormal_below=COM_HEIGHT_ABNORMAL_BELOW_BH,
        abnormal_above=COM_HEIGHT_ABNORMAL_ABOVE_BH,
    )
    _draw_gait_event_vlines(axes[0], result, show_legend=False)
    com_now = None
    com_label = None
    if playhead_time_s is not None and len(t) >= 2:
        com_now = float(np.interp(float(playhead_time_s), t, com_y))
        com_label = f"{com_now:.3f} BH"
    _draw_synced_playhead(
        axes[0],
        playhead_time_s,
        show_label=True,
        value_label=com_label,
        value_y=com_now,
    )
    _deduped_legend(axes[0], loc="upper right", ncol=1)
    _style(axes[0])

    if result.stability_margin and result.stability_margin.per_frame:
        margins = [
            f.stability_margin_m if f.stability_margin_m is not None else np.nan
            for f in result.stability_margin.per_frame
        ]
        st = [f.stability_state for f in result.stability_margin.per_frame]
        ts = [f.time_s for f in result.stability_margin.per_frame]
        _draw_stability_state_bands(axes[1], ts, st)
        _plot_stability_margin_series(axes[1], ts, margins, st, hover_points)
        conf_sm = np.asarray(
            [float(getattr(f, "confidence", 1.0)) for f in result.stability_margin.per_frame],
            dtype=float,
        )
        if conf_sm.size == len(ts):
            draw_confidence_overlay(axes[1], ts, conf_sm, threshold=0.55)
        draw_reference_y_bands(
            axes[1],
            normal=STABILITY_MARGIN_NORMAL_M,
            abnormal_below=0.0,
            abnormal_above=None,
            label_normal=True,
        )
        axes[1].axhline(
            0.04,
            color=STABILITY_STABLE,
            linestyle=":",
            alpha=0.75,
            linewidth=1.0,
            label="Stable threshold (0.04 m)",
        )
        axes[1].axhline(0.0, color=MUTED, linestyle="--", alpha=0.55, linewidth=0.9, label="BoS edge (0)")
        axes[1].set_ylabel("Margin (m)", fontsize=10.5, color=MUTED)
        style_chart_title(axes[1], CHART_STABILITY_MARGIN)
        _draw_gait_event_vlines(axes[1], result, show_legend=False)
        sm_now = None
        sm_label = None
        if playhead_time_s is not None and len(ts) >= 2:
            sm_arr = np.asarray(margins, dtype=float)
            sm_now = float(np.interp(float(playhead_time_s), ts, sm_arr))
            if np.isfinite(sm_now):
                sm_label = f"{sm_now:.4f} m"
        _draw_synced_playhead(
            axes[1],
            playhead_time_s,
            value_label=sm_label,
            value_y=sm_now,
        )
        existing_handles, existing_labels = axes[1].get_legend_handles_labels()
        state_handles = _stability_state_legend_handles()
        keep = []
        keep_labels = []
        for handle, label in zip(existing_handles, existing_labels):
            if not label or str(label).startswith("_"):
                continue
            if label in (
                "Normal range",
                "Stable threshold (0.04 m)",
                "BoS edge (0)",
            ) or "margin" in str(label).lower():
                if label not in keep_labels:
                    keep.append(handle)
                    keep_labels.append(label)
        leg = axes[1].legend(
            keep + state_handles,
            keep_labels + [h.get_label() for h in state_handles],
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            fontsize=7.0,
            loc="upper right",
            ncol=2,
            framealpha=0.92,
            fancybox=False,
            borderpad=0.3,
            handlelength=1.4,
            handletextpad=0.35,
            labelspacing=0.25,
            columnspacing=0.8,
        )
        if leg is not None:
            leg.get_frame().set_linewidth(0.7)
        _style(axes[1])

    if result.contact and result.contact.per_frame:
        lp = result.contact.left_contact_probability
        rp = result.contact.right_contact_probability
        tc = result.contact.timestamps
        asym = np.abs(lp - rp)
        axes[2].plot(tc, asym, color=TEXT, linewidth=1.85, label="|L−R| contact probability", zorder=3)
        _register_line_hover_points(
            axes[2],
            tc,
            asym,
            series_label="Contact asymmetry",
            unit="—",
            hover_points=hover_points,
        )
        conf_c = 0.5 * (
            np.asarray([float(f.left_confidence) for f in result.contact.per_frame], dtype=float)
            + np.asarray([float(f.right_confidence) for f in result.contact.per_frame], dtype=float)
        )
        draw_confidence_overlay(axes[2], tc, conf_c, threshold=0.55)
        draw_reference_y_bands(
            axes[2],
            normal=CONTACT_ASYMMETRY_NORMAL,
            abnormal_above=0.55,
            label_normal=True,
        )
        _draw_contact_event_markers(axes[2], result)
        _draw_gait_event_vlines(axes[2], result, show_legend=False)
        axes[2].set_ylabel("Asymmetry (—)", fontsize=10.5, color=MUTED)
        style_chart_title(axes[2], CHART_CONTACT_EVENTS)
        axes[2].set_ylim(-0.12, 1.18)
        asym_now = None
        asym_label = None
        if playhead_time_s is not None and len(tc) >= 2:
            asym_now = float(np.interp(float(playhead_time_s), tc, asym))
            asym_label = f"{asym_now:.3f}"
        _draw_synced_playhead(
            axes[2],
            playhead_time_s,
            value_label=asym_label,
            value_y=asym_now,
        )
        _deduped_legend(axes[2], loc="upper right", ncol=2)
        _style(axes[2])

    _draw_gait_phase_row(axes[3], result, hover_points)
    _draw_gait_event_vlines(axes[3], result, show_legend=True)
    axes[3].set_xlabel("Time (s)", fontsize=10.5, color=MUTED)
    style_chart_title(axes[3], CHART_GAIT_METRICS)
    _draw_synced_playhead(axes[3], playhead_time_s)
    _deduped_legend(axes[3], loc="upper right", ncol=3)
    _style(axes[3])

    if len(t) >= 2:
        pad = max(0.05, (float(t[-1]) - float(t[0])) * 0.02)
        axes[3].set_xlim(float(t[0]) - pad, float(t[-1]) + pad)

    fig._biomech_hover_points = hover_points
    fig._biomech_tooltip_annot = None
    # Publish ChartHoverPoint list for the shared tooltip / crosshair layer.
    from stablewalk.ui.viewers.chart_hover import ChartHoverPoint, set_figure_hover_points

    shared: list[ChartHoverPoint] = []
    for hp in hover_points:
        # Legacy text tooltips packed "Metric = value"; parse lightly for shared UI.
        lines = (hp.text or "").splitlines()
        metric = "Biomechanics"
        value: float | str | None = float(hp.y)
        unit = ""
        for line in lines:
            if "=" in line:
                key, _, rest = line.partition("=")
                key = key.strip()
                rest = rest.strip()
                if key.lower().startswith("t"):
                    continue
                metric = key
                value = rest
                break
        shared.append(
            ChartHoverPoint(
                ax=hp.ax,
                x=float(hp.x),
                y=float(hp.y),
                timestamp_s=float(hp.x),
                value=value,
                joint_name="—",
                metric_name=metric,
                unit=unit,
            )
        )
    set_figure_hover_points(fig, shared)
    fig.subplots_adjust(left=0.12, right=0.985, top=0.96, bottom=0.08, hspace=0.58)


def attach_biomechanics_hover_tooltips(fig: Figure, canvas: FigureCanvasTkAgg) -> None:
    """Show value tooltips when hovering near plotted biomechanics points."""
    cid = getattr(fig, "_biomech_tooltip_cid", None)
    if cid is not None:
        try:
            canvas.mpl_disconnect(cid)
        except Exception:
            pass

    existing = getattr(fig, "_biomech_tooltip_annot", None)
    if existing is not None:
        try:
            existing.remove()
        except Exception:
            pass
    fig._biomech_tooltip_annot = None

    def _hide_tooltip() -> None:
        annot = getattr(fig, "_biomech_tooltip_annot", None)
        if annot is not None:
            annot.set_visible(False)
            canvas.draw_idle()

    def _on_motion(event) -> None:
        points: list[_HoverPoint] = getattr(fig, "_biomech_hover_points", [])
        if event.inaxes is None or event.x is None or event.y is None or not points:
            _hide_tooltip()
            return

        best: _HoverPoint | None = None
        best_dist = float("inf")
        for pt in points:
            if pt.ax is not event.inaxes:
                continue
            display_x, display_y = pt.ax.transData.transform((pt.x, pt.y))
            dist = float(np.hypot(event.x - display_x, event.y - display_y))
            if dist < 14.0 and dist < best_dist:
                best_dist = dist
                best = pt

        if best is None:
            _hide_tooltip()
            return

        annot = getattr(fig, "_biomech_tooltip_annot", None)
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
            fig._biomech_tooltip_annot = annot

        annot.xy = (best.x, best.y)
        annot.set_text(best.text)
        annot.set_visible(True)
        canvas.draw_idle()

    fig._biomech_tooltip_cid = canvas.mpl_connect("motion_notify_event", _on_motion)


__all__ = ["attach_biomechanics_hover_tooltips", "draw_biomechanics_dashboard"]
