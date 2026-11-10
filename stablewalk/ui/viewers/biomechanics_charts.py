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
    TEXT_SECONDARY,
)
from stablewalk.ui.viewers.chart_style import (
    ANNOTATION_SIZE as _BIOMECH_ANNOT_SIZE,
    AXIS_LABEL_SIZE as _BIOMECH_AXIS_SIZE,
    LEGEND_SIZE as _BIOMECH_LEGEND_SIZE,
    TICK_LABEL_SIZE as _BIOMECH_TICK_SIZE,
    unclip_figure_chrome,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

# Biomechanics: one primary timeline graph — shared presentation type sizes.

_BIOMECH_LAYOUT_TAG = "_biomech_primary_v1"
# Host floor for the technical chart area (Tk grid minsize / fit gate).
# Tall primary graph — findings stay in a narrow left column; Clinical rail on right.
# Sized so the plot owns most of the Biomechanics workspace height.
BIOMECH_TECH_CHART_MIN_H = 720

BIOMECH_SERIES_COM = "com"
BIOMECH_SERIES_STABILITY = "stability"
BIOMECH_SERIES_CONTACT = "contact"
BIOMECH_SERIES_PHASE = "phase"
BIOMECH_SERIES_OPTIONS: tuple[tuple[str, str], ...] = (
    (BIOMECH_SERIES_COM, "COM Vertical Position"),
    (BIOMECH_SERIES_STABILITY, "Stability Margin"),
    (BIOMECH_SERIES_CONTACT, "Contact Timing"),
    (BIOMECH_SERIES_PHASE, "Gait Phase"),
)
DEFAULT_BIOMECH_SERIES = BIOMECH_SERIES_COM

COM_UNAVAILABLE_REASON = (
    "COM unavailable — no per-frame center-of-mass samples.\n\n"
    "Prerequisite: 3D joint landmarks from a completed walking-video analysis.\n"
    "COM is a pose-based segment-weighted estimate, not OpenSim Inverse Kinematics.\n"
    "Resolve: analyze the video (or re-run analysis). Do not substitute IK or export files."
)
BALANCE_UNAVAILABLE_REASON = (
    "Balance unavailable — no COM–BoS stability-margin series.\n\n"
    "Prerequisite: pose-based COM and a support polygon from stance-foot landmarks.\n"
    "This is not a force-plate margin of stability and is not a session score.\n"
    "Resolve: analyze a walking video with detected feet. Values are not fabricated."
)
BALANCE_UNVALIDATED_BANNER = (
    "Unvalidated preview — not a physical stability score"
)
GAIT_PHASE_UNAVAILABLE_REASON = (
    "Gait phase unavailable — no per-frame contact/phase classification.\n\n"
    "Prerequisite: foot-contact analysis from a completed walking-video run.\n"
    "Resolve: analyze the video so left/right contact can be classified."
)
NO_ANALYSIS_REASON = (
    "No biomechanical analysis for this session.\n\n"
    "Prerequisite: open a walking video and run analysis.\n"
    "COM, balance, and contact charts fill from that result — not from OpenSim export."
)
CONTACT_TIMING_UNAVAILABLE_REASON = (
    "Contact timing unavailable — this session has no per-frame contact series.\n\n"
    "Prerequisite: foot-contact analysis from a completed walking-video run.\n"
    "Resolve: analyze the video. Contact does not require OpenSim IK."
)


def _com_has_usable_samples(com: object | None) -> bool:
    """True when COM frames exist with non-zero confidence and a real position."""
    frames = getattr(com, "per_frame", None) or ()
    if not frames:
        return False
    for frame in frames:
        if float(getattr(frame, "confidence", 0.0) or 0.0) <= 0.05:
            continue
        position = getattr(frame, "position", None) or (float("nan"), float("nan"), float("nan"))
        if not all(np.isfinite(float(value)) for value in position):
            continue
        return True
    return False


def _stability_has_usable_samples(stability: object | None) -> bool:
    """True when a stability-margin series exists (None margins stay honest NaNs)."""
    frames = getattr(stability, "per_frame", None) or ()
    return bool(frames)


def biomech_series_unavailable_reason(
    result: BiomechanicalAnalysisResult | None,
    series: str,
) -> str | None:
    """Why a COM / Balance / Gait Phase panel cannot draw, or None if data exists."""
    series_key = (series or DEFAULT_BIOMECH_SERIES).strip().lower()
    if result is None:
        return NO_ANALYSIS_REASON
    if series_key == BIOMECH_SERIES_COM:
        com = getattr(result, "center_of_mass", None)
        if not _com_has_usable_samples(com):
            return COM_UNAVAILABLE_REASON
        return None
    if series_key == BIOMECH_SERIES_STABILITY:
        sm = getattr(result, "stability_margin", None)
        if not _stability_has_usable_samples(sm):
            return BALANCE_UNAVAILABLE_REASON
        return None
    if series_key == BIOMECH_SERIES_PHASE:
        contact = getattr(result, "contact", None)
        if contact is None or not getattr(contact, "per_frame", None):
            return GAIT_PHASE_UNAVAILABLE_REASON
        return None
    if series_key == BIOMECH_SERIES_CONTACT:
        contact = getattr(result, "contact", None)
        if contact is None or not getattr(contact, "per_frame", None):
            return CONTACT_TIMING_UNAVAILABLE_REASON
        return None
    return None


# Fixed margins for the single primary axes (room for larger title + legend).
_BIOMECH_PRIMARY_MARGINS = dict(
    left=0.12,
    right=0.97,
    top=0.90,
    bottom=0.18,
)


@dataclass(frozen=True)
class _HoverPoint:
    ax: Axes
    x: float
    y: float
    text: str


def _style(ax: Axes) -> None:
    from stablewalk.ui.viewers.chart_style import (
        apply_chart_grid,
        apply_chart_panel_style,
    )

    apply_chart_panel_style(ax)
    apply_chart_grid(ax, y_minor=True)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY, labelsize=_BIOMECH_TICK_SIZE)


def _compact_horizontal_legend(
    ax: Axes,
    *,
    handles=None,
    labels=None,
    ncol: int | None = None,
    below_axes: bool = False,
) -> None:
    """Place the legend off the data band (below the axes, or in-axes if tight)."""
    from stablewalk.ui.viewers.chart_style import legend_ncol_for, style_chart_legend

    if handles is None or labels is None:
        h, lab = ax.get_legend_handles_labels()
        handles, labels = h, lab
    # Drop empty / private entries early so ncol matches visible items.
    cleaned_h, cleaned_l = [], []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or str(label).startswith("_") or label in seen:
            continue
        seen.add(str(label))
        cleaned_h.append(handle)
        cleaned_l.append(str(label))
    if not cleaned_l:
        return
    use_ncol = legend_ncol_for(len(cleaned_l), max_ncol=4) if ncol is None else max(1, int(ncol))
    width_px = float(ax.figure.get_figwidth() * ax.figure.dpi)
    if width_px < 520:
        use_ncol = min(use_ncol, 1)
    elif width_px < 760:
        use_ncol = min(use_ncol, 2)
    legend_rows = max(1, (len(cleaned_l) + use_ncol - 1) // use_ncol)
    ax.figure._biomech_legend_rows = legend_rows
    if below_axes:
        ax.figure._biomech_legend_below = True
        style_chart_legend(
            ax,
            handles=cleaned_h,
            labels=cleaned_l,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=use_ncol,
            fontsize=_BIOMECH_LEGEND_SIZE,
            outside=True,
        )
        return
    style_chart_legend(
        ax,
        handles=cleaned_h,
        labels=cleaned_l,
        loc="upper right",
        bbox_to_anchor=None,
        ncol=use_ncol,
        fontsize=_BIOMECH_LEGEND_SIZE,
        outside=False,
    )


def _draw_empty_state(ax: Axes, title: str, message: str) -> None:
    """Visible empty-state copy — never a blank axes, never a fabricated series."""
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    # wrap=True on transAxes text can collapse to zero width on 1×1 canvases.
    ax.text(
        0.5,
        0.52,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=TEXT,
        fontsize=_BIOMECH_ANNOT_SIZE,
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", fc=ELEVATED, ec=BORDER, alpha=0.96),
    )
    _style_biomech_title(ax, title)
    _style(ax)
    ax.set_xlabel("Time (s)", fontsize=_BIOMECH_AXIS_SIZE, color=TEXT_SECONDARY)


def _style_biomech_title(
    ax: Axes,
    title: str,
    *,
    extra_legend_rows: int = 0,
) -> None:
    from stablewalk.ui.viewers.chart_style import style_chart_title

    _ = extra_legend_rows
    style_chart_title(ax, title, pad=8.0)


def _draw_synced_playhead(
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
    """Heel-strike / toe-off glyphs only — avoid stacking foot-flat / mid-stance labels."""
    contact = result.contact
    if contact is None or not contact.per_frame:
        return
    t = contact.timestamps
    event_specs = (
        ("left_heel_strike", "L HS", SIDE_LEFT, "v", 1.08),
        ("right_heel_strike", "R HS", SIDE_RIGHT, "v", 1.08),
        ("left_toe_off", "L TO", SIDE_LEFT, "^", -0.08),
        ("right_toe_off", "R TO", SIDE_RIGHT, "^", -0.08),
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
                s=22,
                color=color,
                zorder=6,
                clip_on=False,
                label=label if not labeled else "_nolegend_",
            )
            labeled = True


def _plot_stability_margin_series(
    ax: Axes,
    ts: list[float],
    margins: list[float],
    states: list[str],
    hover_points: list[_HoverPoint],
) -> None:
    """Color-coded stability margin trace — only when classification is supported."""
    from stablewalk.ui.viewers.chart_style import SERIES_LINE_WIDTH

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
            linewidth=SERIES_LINE_WIDTH,
            zorder=3,
            solid_capstyle="round",
            antialiased=True,
        )
    for ti, mi, state in zip(ts, margins, states):
        if not np.isfinite(mi):
            continue
        ax.scatter([ti], [mi], color=_stability_color(state), s=14, zorder=4, edgecolors=PANEL, linewidths=0.35)
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


def _plot_unvalidated_margin_series(
    ax: Axes,
    ts: list[float],
    margins: list[float],
    hover_points: list[_HoverPoint],
) -> None:
    """Single-color signed-distance preview without Stable/Unstable coloring."""
    from stablewalk.ui.viewers.chart_style import SERIES_LINE_WIDTH

    if len(ts) < 2:
        return
    ax.plot(
        ts,
        margins,
        color=TEXT,
        linewidth=SERIES_LINE_WIDTH,
        zorder=3,
        solid_capstyle="round",
        antialiased=True,
        label="COM–BoS signed distance",
    )
    for ti, mi in zip(ts, margins):
        if not np.isfinite(mi):
            continue
        hover_points.append(
            _HoverPoint(
                ax=ax,
                x=float(ti),
                y=float(mi),
                text=(
                    f"t = {ti:.3f} s\n"
                    f"Signed distance = {mi:.4f} m\n"
                    f"{BALANCE_UNVALIDATED_BANNER}"
                ),
            )
        )


def _draw_unvalidated_preview_banner(ax: Axes) -> None:
    """Keep the preview notice above the data band, not on the series."""
    fig = ax.figure
    fig._biomech_preview_notice = True
    artist = fig.text(
        0.50,
        0.985,
        BALANCE_UNVALIDATED_BANNER,
        ha="center",
        va="top",
        color=TEXT,
        fontsize=_BIOMECH_ANNOT_SIZE,
        zorder=8,
        clip_on=False,
        bbox=dict(boxstyle="round,pad=0.28", fc=ELEVATED, ec=BORDER, alpha=0.92),
    )
    artist._sw_balance_notice = True


def _draw_gait_phase_row(
    ax: Axes,
    result: BiomechanicalAnalysisResult,
    hover_points: list[_HoverPoint],
) -> None:
    """Per-foot categorical phase timelines (Unknown / Swing / Stance)."""
    from stablewalk.analysis.foot_contact_analysis import (
        FOOT_PHASE_TICK_LABELS,
        per_foot_phase_codes,
    )
    from stablewalk.ui.viewers.contact_gait_viewer import draw_per_foot_phase_timelines

    contact = result.contact
    if contact is None or not contact.per_frame:
        ax.text(
            0.5,
            0.5,
            GAIT_PHASE_UNAVAILABLE_REASON,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        return

    t = contact.timestamps
    draw_per_foot_phase_timelines(ax, contact, title=None, shade_unknown=True)
    ax.set_ylabel("Phase", fontsize=_BIOMECH_AXIS_SIZE, color=TEXT_SECONDARY, labelpad=4)

    left_codes, right_codes = per_foot_phase_codes(contact)
    labels = list(FOOT_PHASE_TICK_LABELS)
    for ti, lc, rc, frame in zip(t, left_codes, right_codes, contact.per_frame):
        hover_points.append(
            _HoverPoint(
                ax=ax,
                x=float(ti),
                y=float(max(lc, rc)),
                text=(
                    f"t = {float(ti):.3f} s\n"
                    f"Left = {labels[int(lc)]}\n"
                    f"Right = {labels[int(rc)]}\n"
                    f"L contact known = {bool(frame.left_contact_known)}\n"
                    f"R contact known = {bool(frame.right_contact_known)}"
                ),
            )
        )

    # Same validity contract as Overview / Results. Do not keep a leftover
    # gait-quality or speed stamp when those summary metrics are withheld.
    summary_bits: list[str] = []
    quality_ok = False
    try:
        from stablewalk.analysis.result_validity import (
            build_result_validity,
            withhold_gait_quality_heuristic,
        )

        validity = build_result_validity(
            biomechanical=result,
            contact=contact,
            cycles=getattr(result, "cycles", None),
        )
        quality_ok = not withhold_gait_quality_heuristic(validity)
    except Exception:
        quality_ok = False
    if (
        quality_ok
        and result.gait_quality
        and getattr(result.gait_quality, "available", True)
        and result.gait_quality.score is not None
    ):
        summary_bits.append(
            f"Unvalidated heuristic {result.gait_quality.score:.0f}/100"
        )
    if result.gait_metrics and result.gait_metrics.walking_speed:
        from stablewalk.analysis.biomechanical.walking_speed import (
            format_walking_speed_display,
            is_reportable_walking_speed,
        )

        ws = result.gait_metrics.walking_speed
        if is_reportable_walking_speed(ws):
            ws_label = format_walking_speed_display(ws)
            if ws_label and "Not available" not in ws_label:
                summary_bits.append(f"Estimated speed {ws_label}")
    if summary_bits:
        ax.text(
            0.99,
            0.08,
            " · ".join(summary_bits),
            transform=ax.transAxes,
            color=TEXT,
            fontsize=_BIOMECH_ANNOT_SIZE,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.18", fc=PANEL, ec=BORDER, alpha=0.92),
        )


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


def ensure_biomechanics_axes(fig: Figure):
    """Create or reuse a single primary timeline axes."""
    if getattr(fig, _BIOMECH_LAYOUT_TAG, False) and len(fig.axes) == 1:
        return list(fig.axes)
    fig.clear()
    _disable_auto_layout_engines(fig)
    ax = fig.add_subplot(111)
    setattr(fig, _BIOMECH_LAYOUT_TAG, True)
    return [ax]


def _disable_auto_layout_engines(fig: Figure) -> None:
    """Keep manual margins in control until the canvas pixel size is known."""
    try:
        fig.set_layout_engine("none")
    except Exception:
        try:
            fig.set_constrained_layout(False)
        except Exception:
            pass
        try:
            fig.set_tight_layout(False)
        except Exception:
            pass


def layout_biomechanics_figure(fig: Figure, *, size_known: bool = False) -> None:
    """Apply margins for the single primary biomechanics axes."""
    _disable_auto_layout_engines(fig)
    axes = list(fig.axes)
    if not axes:
        return
    width_px = float(fig.get_figwidth() * fig.dpi)
    height_px = float(fig.get_figheight() * fig.dpi)
    _legend_rows = int(getattr(fig, "_biomech_legend_rows", 1) or 1)
    _ = _legend_rows
    margins = dict(_BIOMECH_PRIMARY_MARGINS)
    legend_below = bool(getattr(fig, "_biomech_legend_below", False))
    preview = bool(getattr(fig, "_biomech_preview_notice", False))
    if height_px < 520:
        margins["top"] = 0.84 if preview else 0.88
        margins["bottom"] = 0.28 if legend_below else 0.22
    elif height_px < 700:
        margins["top"] = 0.86 if preview else 0.90
        margins["bottom"] = 0.26 if legend_below else 0.18
    else:
        if preview:
            margins["top"] = 0.86
        if legend_below:
            margins["bottom"] = 0.24
    if width_px < 720:
        margins["left"] = 0.16
        margins["right"] = 0.97
    elif width_px < 960:
        margins["left"] = 0.14
        margins["right"] = 0.96
    label_x = -0.16 if float(margins["left"]) >= 0.18 else -0.10
    try:
        fig.subplots_adjust(**margins)
    except Exception:
        pass
    for ax in axes:
        ax.tick_params(axis="y", labelsize=_BIOMECH_TICK_SIZE, pad=1)
        ax.tick_params(axis="x", labelsize=_BIOMECH_TICK_SIZE, pad=1)
        try:
            ax.yaxis.set_label_coords(label_x, 0.5)
            ax.yaxis.label.set_clip_on(False)
            ax.xaxis.label.set_clip_on(False)
        except Exception:
            pass
        ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
        try:
            title = ax.title
            title.set_clip_on(False)
            title.set_pad(8.0)
        except Exception:
            pass
        try:
            legend = ax.get_legend()
            if legend is not None:
                clip_legend = not bool(getattr(fig, "_biomech_legend_below", False))
                legend.set_clip_on(clip_legend)
                legend.set_in_layout(False)
        except Exception:
            pass
    unclip_figure_chrome(fig)
    _ = size_known


def draw_biomechanics_dashboard(
    fig: Figure,
    result: BiomechanicalAnalysisResult | None,
    *,
    playhead_time_s: float | None = None,
    playhead_frame: int | None = None,
    series: str = DEFAULT_BIOMECH_SERIES,
) -> None:
    """Draw one primary biomechanics timeline series (COM / stability / contact / phase)."""
    fig.patch.set_facecolor(PANEL)
    hover_points: list[_HoverPoint] = []
    axes = ensure_biomechanics_axes(fig)
    ax = axes[0]
    ax.clear()
    fig._biomech_legend_below = False
    fig._biomech_preview_notice = False
    for txt in list(getattr(fig, "texts", ()) or ()):
        if getattr(txt, "_sw_balance_notice", False):
            try:
                txt.remove()
            except Exception:
                pass
    if hasattr(ax, "_stablewalk_playhead_artists"):
        try:
            delattr(ax, "_stablewalk_playhead_artists")
        except Exception:
            ax._stablewalk_playhead_artists = None  # type: ignore[attr-defined]
    fig._stablewalk_playhead_bg = None  # type: ignore[attr-defined]

    series_key = (series or DEFAULT_BIOMECH_SERIES).strip().lower()
    valid_keys = {k for k, _ in BIOMECH_SERIES_OPTIONS}
    if series_key not in valid_keys:
        series_key = DEFAULT_BIOMECH_SERIES

    unavailable = biomech_series_unavailable_reason(result, series_key)
    if unavailable:
        title = {
            BIOMECH_SERIES_COM: CHART_COM_HEIGHT,
            BIOMECH_SERIES_STABILITY: CHART_STABILITY_MARGIN,
            BIOMECH_SERIES_CONTACT: CHART_CONTACT_EVENTS,
            BIOMECH_SERIES_PHASE: CHART_GAIT_METRICS,
        }.get(series_key, CHART_COM_HEIGHT)
        _draw_empty_state(ax, title, unavailable)
        fig._biomech_hover_points = []
        from stablewalk.ui.viewers.chart_hover import set_figure_hover_points

        set_figure_hover_points(fig, [])
        layout_biomechanics_figure(fig, size_known=False)
        return

    assert result is not None
    t = None
    com_y = None
    if result.center_of_mass is not None and result.center_of_mass.per_frame:
        t = result.center_of_mass.timestamps
        com_y = result.center_of_mass.positions[:, 1]

    from stablewalk.ui.viewers.chart_reference import (
        COM_VERTICAL_PELVIS_RELATIVE_ILLUSTRATIVE_BH,
        CONTACT_ASYMMETRY_NORMAL,
        draw_confidence_overlay,
        draw_reference_y_bands,
    )
    from stablewalk.ui.viewers.chart_style import (
        SERIES_LINE_WIDTH,
        series_plot_kwargs,
    )

    if series_key == BIOMECH_SERIES_COM:
        assert t is not None and com_y is not None
        from stablewalk.analysis.joint_path_trace import series_with_gap_breaks

        com_y_plot = np.asarray(com_y, dtype=float).copy()
        com_frames = result.center_of_mass.per_frame
        conf = np.array(
            [float(getattr(frame, "confidence", 0.0) or 0.0) for frame in com_frames],
            dtype=float,
        )
        if conf.size == com_y_plot.size:
            com_y_plot[conf <= 0.05] = np.nan
        t_plot, y_plot = series_with_gap_breaks(
            t, com_y_plot, fps=float(getattr(result.center_of_mass, "fps", 30.0) or 30.0)
        )
        ax.plot(
            t_plot,
            y_plot,
            **series_plot_kwargs(
                color=COM,
                label="COM Y",
                linewidth=SERIES_LINE_WIDTH,
                zorder=3,
            ),
        )
        _register_line_hover_points(
            ax,
            t,
            com_y_plot,
            series_label="COM vertical position (pelvis-relative)",
            unit="approx. BH",
            hover_points=hover_points,
        )
        conf_com = np.asarray(
            [float(getattr(f, "confidence", 1.0)) for f in result.center_of_mass.per_frame],
            dtype=float,
        )
        if conf_com.size == len(t):
            draw_confidence_overlay(ax, t, conf_com, threshold=0.55)
        ax.set_ylabel(
            "Pelvis-relative Y (BH)",
            fontsize=_BIOMECH_AXIS_SIZE,
            color=TEXT_SECONDARY,
            labelpad=4,
        )
        _style_biomech_title(ax, CHART_COM_HEIGHT)
        illustrative_band = COM_VERTICAL_PELVIS_RELATIVE_ILLUSTRATIVE_BH
        y0 = float(np.nanmin(com_y)) if np.isfinite(com_y).any() else illustrative_band[0]
        y1 = float(np.nanmax(com_y)) if np.isfinite(com_y).any() else illustrative_band[1]
        span = max(y1 - y0, 0.02)
        ax.set_ylim(
            min(y0, illustrative_band[0]) - span * 0.12,
            max(y1, illustrative_band[1]) + span * 0.12,
        )
        draw_reference_y_bands(
            ax,
            normal=illustrative_band,
            abnormal_below=None,
            abnormal_above=None,
            label_normal=True,
            normal_label="Illustrative pelvis-relative band",
        )
        from stablewalk.ui.viewers.biomech_chart_enrichment import enrich_numeric_series

        enrich_numeric_series(
            ax,
            t,
            com_y,
            unit="approx. BH",
            peak_color=COM,
            stats_loc="upper left",
            peak_label="_nolegend_",
        )
        _draw_gait_event_vlines(ax, result, show_legend=False)
        from stablewalk.ui.viewers.chart_playhead import set_playhead_value_source

        set_playhead_value_source(ax, t, com_y, "COM Y", "approx. BH")
        _draw_synced_playhead(
            ax, playhead_time_s, show_label=True, frame_index=playhead_frame
        )
        _compact_horizontal_legend(ax, ncol=1)
        _style(ax)

    elif series_key == BIOMECH_SERIES_STABILITY:
        from stablewalk.analysis.biomechanical.stability_margin import (
            balance_display_contract,
        )

        contract = balance_display_contract(result.stability_margin)
        if contract.display_mode == "hidden" or not contract.has_series:
            _draw_empty_state(
                ax,
                CHART_STABILITY_MARGIN,
                contract.reason or BALANCE_UNAVAILABLE_REASON,
            )
        else:
            margins = [
                float(v) if v is not None else np.nan for v in contract.values_m
            ]
            ts = [float(t) for t in contract.times_s]
            classified = bool(contract.classification_supported and contract.validity == "valid")
            if classified and result.stability_margin is not None:
                st = [f.stability_state for f in result.stability_margin.per_frame]
                _draw_stability_state_bands(ax, ts, st)
                _plot_stability_margin_series(ax, ts, margins, st, hover_points)
            else:
                _plot_unvalidated_margin_series(ax, ts, margins, hover_points)
                _draw_unvalidated_preview_banner(ax)
            conf_sm = np.asarray(
                [
                    float(getattr(f, "confidence", 1.0))
                    for f in (result.stability_margin.per_frame or [])
                ],
                dtype=float,
            )
            if conf_sm.size == len(ts):
                draw_confidence_overlay(ax, ts, conf_sm, threshold=0.55)
            ax.axhline(
                0.0,
                color=MUTED,
                linestyle="--",
                alpha=0.65,
                linewidth=1.0,
                label="BoS boundary (signed distance = 0)",
            )
            from stablewalk.ui.viewers.biomech_chart_enrichment import enrich_numeric_series

            enrich_numeric_series(
                ax,
                ts,
                margins,
                unit=contract.unit or "m",
                peak_color=TEXT,
                stats_loc="upper left",
            )
            ax.set_ylabel(
                "Signed COM–BoS distance (m)",
                fontsize=_BIOMECH_AXIS_SIZE,
                color=TEXT_SECONDARY,
                labelpad=4,
            )
            _style_biomech_title(ax, CHART_STABILITY_MARGIN)
            _draw_gait_event_vlines(ax, result, show_legend=False)
            from stablewalk.ui.viewers.chart_playhead import set_playhead_value_source

            set_playhead_value_source(
                ax, ts, margins, "COM–BoS", contract.unit or "m"
            )
            _draw_synced_playhead(
            ax, playhead_time_s, show_label=True, frame_index=playhead_frame
        )
            _compact_horizontal_legend(ax, ncol=2, below_axes=True)
            _style(ax)

    elif series_key == BIOMECH_SERIES_CONTACT:
        if result.contact and result.contact.per_frame:
            lp = result.contact.left_contact_probability
            rp = result.contact.right_contact_probability
            tc = result.contact.timestamps
            asym = np.abs(lp - rp)
            ax.plot(
                tc,
                asym,
                **series_plot_kwargs(color=TEXT, label="|L−R| contact", zorder=3),
            )
            _register_line_hover_points(
                ax,
                tc,
                asym,
                series_label="Contact asymmetry",
                unit="—",
                hover_points=hover_points,
            )
            conf_c = 0.5 * (
                np.asarray(
                    [float(f.left_confidence) for f in result.contact.per_frame], dtype=float
                )
                + np.asarray(
                    [float(f.right_confidence) for f in result.contact.per_frame], dtype=float
                )
            )
            draw_confidence_overlay(ax, tc, conf_c, threshold=0.55)
            draw_reference_y_bands(
                ax,
                normal=CONTACT_ASYMMETRY_NORMAL,
                abnormal_above=0.55,
                label_normal=True,
            )
            from stablewalk.ui.viewers.biomech_chart_enrichment import enrich_numeric_series

            enrich_numeric_series(
                ax,
                tc,
                asym,
                unit="",
                peak_color=TEXT,
                stats_loc="upper right",
            )
            _draw_contact_event_markers(ax, result)
            ax.set_ylabel("Asymmetry (unitless)", fontsize=_BIOMECH_AXIS_SIZE, color=TEXT_SECONDARY, labelpad=4)
            _style_biomech_title(ax, CHART_CONTACT_EVENTS)
            ax.set_ylim(-0.12, 1.18)
            from stablewalk.ui.viewers.chart_playhead import set_playhead_value_source

            set_playhead_value_source(ax, tc, asym, "|L−R| contact", "")
            _draw_synced_playhead(
            ax, playhead_time_s, show_label=True, frame_index=playhead_frame
        )
            _compact_horizontal_legend(ax, ncol=4)
            _style(ax)
        else:
            _draw_empty_state(ax, CHART_CONTACT_EVENTS, CONTACT_TIMING_UNAVAILABLE_REASON)

    else:  # phase
        from stablewalk.ui.viewers.biomech_chart_enrichment import annotate_phase_stats_box

        _draw_gait_phase_row(ax, result, hover_points)
        annotate_phase_stats_box(ax, result.contact)
        _draw_gait_event_vlines(ax, result, show_legend=False)
        _style_biomech_title(ax, CHART_GAIT_METRICS)
        _draw_synced_playhead(
            ax, playhead_time_s, show_label=True, frame_index=playhead_frame
        )
        _compact_horizontal_legend(ax, ncol=2)
        _style(ax)

    ax.set_xlabel("Time (s)", fontsize=_BIOMECH_AXIS_SIZE, color=TEXT_SECONDARY)
    times_for_xlim = t
    if series_key == BIOMECH_SERIES_STABILITY and result.stability_margin is not None:
        times_for_xlim = [f.time_s for f in result.stability_margin.per_frame]
    elif (
        series_key in (BIOMECH_SERIES_CONTACT, BIOMECH_SERIES_PHASE)
        and result.contact is not None
    ):
        times_for_xlim = result.contact.timestamps
    if times_for_xlim is not None and len(times_for_xlim) >= 2:
        pad = max(
            0.05,
            (float(times_for_xlim[-1]) - float(times_for_xlim[0])) * 0.02,
        )
        ax.set_xlim(float(times_for_xlim[0]) - pad, float(times_for_xlim[-1]) + pad)

    fig._biomech_hover_points = hover_points
    fig._biomech_tooltip_annot = None
    from stablewalk.ui.viewers.chart_hover import ChartHoverPoint, set_figure_hover_points

    shared: list[ChartHoverPoint] = []
    for hp in hover_points:
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

    layout_biomechanics_figure(fig, size_known=False)


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
                fontsize=11,
                zorder=30,
                visible=False,
            )
            fig._biomech_tooltip_annot = annot

        annot.xy = (best.x, best.y)
        annot.set_text(best.text)
        annot.set_visible(True)
        canvas.draw_idle()

    fig._biomech_tooltip_cid = canvas.mpl_connect("motion_notify_event", _on_motion)


__all__ = [
    "BIOMECH_SERIES_COM",
    "BIOMECH_SERIES_CONTACT",
    "BIOMECH_SERIES_OPTIONS",
    "BIOMECH_SERIES_PHASE",
    "BIOMECH_SERIES_STABILITY",
    "BIOMECH_TECH_CHART_MIN_H",
    "BALANCE_UNAVAILABLE_REASON",
    "BALANCE_UNVALIDATED_BANNER",
    "COM_UNAVAILABLE_REASON",
    "DEFAULT_BIOMECH_SERIES",
    "GAIT_PHASE_UNAVAILABLE_REASON",
    "NO_ANALYSIS_REASON",
    "attach_biomechanics_hover_tooltips",
    "biomech_series_unavailable_reason",
    "draw_biomechanics_dashboard",
    "ensure_biomechanics_axes",
    "layout_biomechanics_figure",
]
