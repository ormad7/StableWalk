"""Shared matplotlib axis styling for dashboard time-series charts.

Publication-quality chrome aligned with OpenSim / Visual3D / Vicon / Qualisys.
Display-only: does not alter plotted values or analysis math.
"""

from __future__ import annotations

from typing import Any, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

from stablewalk.ui.colors import BORDER, MUTED, PANEL, TEXT, TEXT_SECONDARY

TIMELINE_X_LABEL = "Time (s)"
GAIT_CYCLE_X_LABEL = "Gait Cycle (%)"

# Projector / thesis presentation type (readable from a back row).
_AXIS_LABEL_SIZE = 13.0
_TICK_LABEL_SIZE = 12.0
_TITLE_SIZE = 14.0
_LEGEND_SIZE = 11.5
_ANNOTATION_SIZE = 11.0
_MAJOR_TICK_LENGTH = 6.5
_MINOR_TICK_LENGTH = 3.5

# Public aliases for viewers that need the shared sizes without private imports.
AXIS_LABEL_SIZE = _AXIS_LABEL_SIZE
TICK_LABEL_SIZE = _TICK_LABEL_SIZE
TITLE_SIZE = _TITLE_SIZE
LEGEND_SIZE = _LEGEND_SIZE
ANNOTATION_SIZE = _ANNOTATION_SIZE

# Primary series stroke (L/R joint traces, COM, etc.).
SERIES_LINE_WIDTH = 2.9
SERIES_LINE_WIDTH_SECONDARY = 2.35
SERIES_LINE_WIDTH_STEP = 2.45
SERIES_MARKER_SIZE = 40.0

# Left solid / right dashed — colorblind-friendly L/R distinction beyond hue.
LINESTYLE_LEFT: str | tuple = "-"
LINESTYLE_RIGHT: str | tuple = (0, (6.0, 2.75))

# Dark-theme grid: quieter than series, still visible on a projector.
_GRID_MAJOR_ALPHA = 0.22
_GRID_MINOR_ALPHA = 0.09
_GRID_MAJOR_LINEWIDTH = 0.65
_GRID_MINOR_LINEWIDTH = 0.35

# Smoother path rendering (matplotlib Agg / canvas).
_SERIES_ANTIALIASED = True
_SERIES_CAPSTYLE = "round"
_SERIES_JOINSTYLE = "round"

# Consistent Y headroom across time-series drawers (~8%).
Y_AUTOSCALE_PAD = 0.08

# Outside-axes legend placement (keeps the data band clear on a projector).
LEGEND_OUTSIDE_LOC = "upper left"
LEGEND_OUTSIDE_BBOX: tuple[float, float] = (1.02, 1.0)

# ~25% larger data area vs prior right≈0.78 / left≈0.12 layouts.
PRESENTATION_SINGLE_MARGINS = dict(left=0.09, right=0.88, top=0.94, bottom=0.13)
PRESENTATION_STACK_MARGINS = dict(
    left=0.09, right=0.88, top=0.97, bottom=0.07, hspace=0.34
)
PRESENTATION_CONTACT_MARGINS = dict(
    left=0.09, right=0.88, top=0.98, bottom=0.06, hspace=0.30
)
# Motion Analysis knee panel: title + ticks stay inside; legend is in-axes.
MOTION_KNEE_MARGINS = dict(left=0.14, right=0.97, top=0.90, bottom=0.18)

# Artists tagged for LOD: hidden at full timeline, shown when zoomed in.
SECONDARY_ANNOTATION_ATTR = "_sw_secondary_annotation"


def configure_chart_antialiasing(fig: Figure | None = None) -> None:
    """Enable smoother line/path anti-aliasing for dashboard charts."""
    import matplotlib as mpl

    mpl.rcParams["lines.antialiased"] = True
    mpl.rcParams["patch.antialiased"] = True
    mpl.rcParams["path.simplify"] = True
    # Mild simplification keeps long gait traces smooth without distorting peaks.
    mpl.rcParams["path.simplify_threshold"] = 0.08
    if fig is not None:
        canvas = getattr(fig, "canvas", None)
        if canvas is not None and hasattr(canvas, "set_dpi_ratio"):
            try:
                canvas.set_dpi_ratio(max(float(getattr(canvas, "device_pixel_ratio", 1.0)), 1.0))
            except Exception:
                pass


def side_linestyle(side: str) -> str | tuple:
    """Return solid (left) or dashed (right) linestyle for limb series."""
    key = (side or "").strip().lower()
    if key in ("right", "r") or key.startswith("right"):
        return LINESTYLE_RIGHT
    return LINESTYLE_LEFT


def side_linestyle_for_item(item_id: str | None) -> str | tuple:
    """Linestyle from a GUI joint item id (``left_*`` / ``right_*``)."""
    if not item_id:
        return LINESTYLE_LEFT
    return side_linestyle(item_id)


def legend_ncol_for(n_entries: int, *, max_ncol: int = 4) -> int:
    """Pick a readable column count from the number of legend entries."""
    n = max(0, int(n_entries))
    if n <= 1:
        return 1
    if n <= 3:
        return min(n, max_ncol)
    if n <= 6:
        return min(3, max_ncol)
    return min(max_ncol, max(2, (n + 1) // 2))


def series_plot_kwargs(
    *,
    color: str,
    label: str | None = None,
    linewidth: float | None = None,
    linestyle: str | tuple | None = None,
    side: str | None = None,
    item_id: str | None = None,
    zorder: float = 4,
    alpha: float = 1.0,
    markers: bool = False,
) -> dict[str, Any]:
    """Shared kwargs for scientifically faithful, high-legibility series strokes.

    When ``side`` is set, linestyle distinguishes limbs (solid left / dashed right).
    Optional sparse markers add a second non-color cue (circle left / square right).
    """
    if linestyle is None:
        if side is not None:
            linestyle = side_linestyle(side)
        elif item_id is not None:
            linestyle = side_linestyle_for_item(item_id)
        else:
            linestyle = "-"
    kwargs: dict[str, Any] = {
        "color": color,
        "label": label,
        "linewidth": float(SERIES_LINE_WIDTH if linewidth is None else linewidth),
        "linestyle": linestyle,
        "zorder": zorder,
        "alpha": alpha,
        "antialiased": _SERIES_ANTIALIASED,
        "solid_capstyle": _SERIES_CAPSTYLE,
        "solid_joinstyle": _SERIES_JOINSTYLE,
    }
    if markers and (side is not None or item_id is not None):
        key = (side or item_id or "").strip().lower()
        is_right = key in ("right", "r") or key.startswith("right")
        kwargs["marker"] = "s" if is_right else "o"
        kwargs["markevery"] = 12
        kwargs["markersize"] = 3.5
        kwargs["markerfacecolor"] = color
        kwargs["markeredgecolor"] = color
    return kwargs


def apply_chart_panel_style(ax: Axes) -> None:
    """Panel background and research-lab spines (bottom + left only)."""
    configure_chart_antialiasing(ax.figure if ax is not None else None)
    ax.set_facecolor(PANEL)
    fig = getattr(ax, "figure", None)
    if fig is not None:
        try:
            fig.patch.set_facecolor(PANEL)
        except Exception:
            pass
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(BORDER)
            spine.set_linewidth(1.15)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    for axis in (ax.xaxis, ax.yaxis):
        try:
            axis.label.set_color(TEXT_SECONDARY)
        except Exception:
            pass
    try:
        zaxis = getattr(ax, "zaxis", None)
        if zaxis is not None:
            zaxis.label.set_color(TEXT_SECONDARY)
    except Exception:
        pass


def style_chart_title(ax: Axes, title: str, *, pad: float = 10.0) -> None:
    """Consistent publication title."""
    artist = ax.set_title(
        title,
        color=TEXT,
        fontsize=_TITLE_SIZE,
        fontweight="medium",
        pad=pad,
    )
    try:
        artist.set_clip_on(False)
    except Exception:
        pass


def _color_luminance(color: Any) -> float | None:
    try:
        from matplotlib.colors import to_rgb

        r, g, b = to_rgb(color)
    except Exception:
        return None
    return 0.2126 * float(r) + 0.7152 * float(g) + 0.0722 * float(b)


def _ensure_readable_on_dark(artist: Any, *, fallback: str = TEXT_SECONDARY) -> None:
    """Raise dark / muted artist colors so they stay visible on PANEL."""
    if artist is None:
        return
    try:
        color = artist.get_color()
    except Exception:
        return
    lum = _color_luminance(color)
    if lum is None or lum < 0.62:
        try:
            artist.set_color(fallback)
        except Exception:
            pass


def unclip_axis_chrome(ax: Axes) -> None:
    """Keep titles, axis labels, and tick labels drawable on a dark panel."""
    title = getattr(ax, "title", None)
    if title is not None:
        try:
            title.set_clip_on(False)
        except Exception:
            pass
        _ensure_readable_on_dark(title, fallback=TEXT)
    axes_list = [ax.xaxis, ax.yaxis]
    zaxis = getattr(ax, "zaxis", None)
    if zaxis is not None:
        axes_list.append(zaxis)
    for axis in axes_list:
        label = getattr(axis, "label", None)
        if label is None:
            continue
        try:
            label.set_clip_on(False)
        except Exception:
            pass
        _ensure_readable_on_dark(label, fallback=TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    tick_labels = []
    try:
        tick_labels.extend(ax.get_xticklabels())
        tick_labels.extend(ax.get_yticklabels())
    except Exception:
        pass
    try:
        get_zticklabels = getattr(ax, "get_zticklabels", None)
        if callable(get_zticklabels):
            tick_labels.extend(get_zticklabels())
    except Exception:
        pass
    for label in tick_labels:
        try:
            label.set_clip_on(False)
        except Exception:
            pass
        _ensure_readable_on_dark(label, fallback=TEXT_SECONDARY)
    try:
        legend = ax.get_legend()
    except Exception:
        legend = None
    if legend is not None:
        try:
            for text in legend.get_texts():
                _ensure_readable_on_dark(text, fallback=TEXT)
                text.set_clip_on(False)
        except Exception:
            pass


def unclip_figure_chrome(fig: Figure | None) -> None:
    """Apply unclip_axis_chrome to every axes on a figure."""
    if fig is None:
        return
    try:
        axes = list(fig.axes)
    except Exception:
        return
    for ax in axes:
        unclip_axis_chrome(ax)


def _legend_style_kwargs(
    *,
    loc: str,
    ncol: int,
    fontsize: float,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "loc": loc,
        "ncol": ncol,
        "fontsize": fontsize,
        "frameon": True,
        "fancybox": False,
        "edgecolor": BORDER,
        "facecolor": PANEL,
        "labelcolor": TEXT,
        "framealpha": 0.96,
        "borderpad": 0.5,
        "handlelength": 2.5,
        "handletextpad": 0.65,
        "labelspacing": 0.5,
        "columnspacing": 1.15,
    }
    if bbox_to_anchor is not None:
        kwargs["bbox_to_anchor"] = bbox_to_anchor
        kwargs["borderaxespad"] = 0.0
    return kwargs


def _dedupe_legend_entries(
    handles: Sequence[Any],
    labels: Sequence[str],
) -> tuple[list[Any], list[str]]:
    by_label: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and not str(label).startswith("_") and label not in by_label:
            by_label[label] = handle
    return list(by_label.values()), list(by_label.keys())


def style_chart_legend(
    ax: Axes,
    *,
    loc: str | None = None,
    ncol: int | None = None,
    fontsize: float | None = None,
    bbox_to_anchor: tuple[float, float] | None = None,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
    outside: bool = True,
):
    """Dynamic legend: deduped entries, adaptive columns, outside axes by default."""
    if handles is None or labels is None:
        h, lab = ax.get_legend_handles_labels()
        handles, labels = _dedupe_legend_entries(h, lab)
    else:
        handles, labels = _dedupe_legend_entries(list(handles), list(labels))
    if not labels:
        return None
    use_ncol = legend_ncol_for(len(labels)) if ncol is None else max(1, int(ncol))
    use_fs = _LEGEND_SIZE if fontsize is None else float(fontsize)
    if outside:
        use_loc = loc if loc is not None else LEGEND_OUTSIDE_LOC
        use_bbox = LEGEND_OUTSIDE_BBOX if bbox_to_anchor is None else bbox_to_anchor
    else:
        use_loc = loc if loc is not None else "upper right"
        use_bbox = bbox_to_anchor
    leg = ax.legend(
        handles,
        labels,
        **_legend_style_kwargs(
            loc=use_loc,
            ncol=use_ncol,
            fontsize=use_fs,
            bbox_to_anchor=use_bbox,
        ),
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_linewidth(0.85)
        for text in leg.get_texts():
            text.set_fontsize(use_fs)
        # Inside-axes legends stay in the plot so they cannot paint off-canvas.
        if not outside:
            try:
                leg.set_clip_on(True)
                frame.set_clip_on(True)
            except Exception:
                pass
    return leg


def apply_presentation_layout(
    fig: Figure,
    *,
    mode: str = "single",
) -> None:
    """Maximize plot area (~25% vs prior margins) while reserving an outside legend rail."""
    try:
        if mode == "stack":
            fig.subplots_adjust(**PRESENTATION_STACK_MARGINS)
        elif mode == "contact":
            fig.subplots_adjust(**PRESENTATION_CONTACT_MARGINS)
        elif mode == "motion_knee":
            fig.subplots_adjust(**MOTION_KNEE_MARGINS)
        else:
            fig.subplots_adjust(**PRESENTATION_SINGLE_MARGINS)
    except Exception:
        pass


def apply_motion_knee_layout(fig: Figure) -> None:
    """Margins so Knee Flexion title, legend, and tick labels stay inside the panel."""
    apply_presentation_layout(fig, mode="motion_knee")


def style_motion_knee_legend(ax: Axes, *, handles=None, labels=None, ncol: int = 1):
    """Place the knee legend inside the axes so it cannot clip off-canvas."""
    return style_chart_legend(
        ax,
        handles=handles,
        labels=labels,
        loc="upper right",
        bbox_to_anchor=None,
        ncol=max(1, int(ncol)),
        fontsize=_LEGEND_SIZE,
        outside=False,
    )


def style_3d_axis_labels(
    ax: Axes,
    *,
    xlabel: str = "X (m)",
    ylabel: str = "Y (m)",
    zlabel: str = "Z (m)",
) -> None:
    """Shared 3D axis label chrome for trajectory / compare / skeleton paths."""
    try:
        ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=_AXIS_LABEL_SIZE - 2.0)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=_AXIS_LABEL_SIZE - 2.0)
        ax.set_zlabel(zlabel, color=TEXT_SECONDARY, fontsize=_AXIS_LABEL_SIZE - 2.0)
    except Exception:
        pass
    for axis in (ax.xaxis, ax.yaxis, getattr(ax, "zaxis", None)):
        if axis is None:
            continue
        try:
            axis.label.set_clip_on(False)
        except Exception:
            pass
    try:
        ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY)
    except Exception:
        pass

def mark_secondary_annotation(artist: Any) -> Any:
    """Tag an artist so it hides when the chart is fully zoomed out."""
    if artist is not None:
        try:
            setattr(artist, SECONDARY_ANNOTATION_ATTR, True)
        except Exception:
            pass
    return artist


def mark_secondary_annotations(*artists: Any) -> None:
    for artist in artists:
        mark_secondary_annotation(artist)


def is_secondary_annotation(artist: Any) -> bool:
    return bool(getattr(artist, SECONDARY_ANNOTATION_ATTR, False))


def style_figure_legend(
    fig: Figure,
    handles: Sequence[Any],
    labels: Sequence[str],
    *,
    loc: str = "upper center",
    ncol: int | None = None,
    fontsize: float | None = None,
    bbox_to_anchor: tuple[float, float] = (0.5, 0.995),
):
    """Figure-level legend with the same dynamic chrome as axis legends."""
    handles, labels = _dedupe_legend_entries(list(handles), list(labels))
    if not labels:
        return None
    use_ncol = legend_ncol_for(len(labels), max_ncol=6) if ncol is None else max(1, int(ncol))
    use_fs = _LEGEND_SIZE if fontsize is None else float(fontsize)
    leg = fig.legend(
        handles,
        labels,
        **_legend_style_kwargs(
            loc=loc,
            ncol=use_ncol,
            fontsize=use_fs,
            bbox_to_anchor=bbox_to_anchor,
        ),
    )
    if leg is not None:
        leg.get_frame().set_linewidth(0.85)
    return leg


def apply_chart_grid(ax: Axes, *, y_minor: bool = True) -> None:
    """Subtle major/minor grid lines tuned for the dark dashboard theme."""
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        axis="both",
        color=BORDER,
        alpha=_GRID_MAJOR_ALPHA,
        linestyle="-",
        linewidth=_GRID_MAJOR_LINEWIDTH,
        zorder=0,
    )
    minor_axis = "both" if y_minor else "x"
    ax.grid(
        True,
        which="minor",
        axis=minor_axis,
        color=BORDER,
        alpha=_GRID_MINOR_ALPHA,
        linestyle=":",
        linewidth=_GRID_MINOR_LINEWIDTH,
        zorder=0,
    )


def configure_time_axis(ax: Axes, *, show_xlabel: bool = True, nbins: int = 6) -> None:
    """Numeric time ticks with major and minor divisions."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=3))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(
        axis="x",
        which="major",
        colors=TEXT_SECONDARY,
        labelcolor=TEXT_SECONDARY,
        labelsize=_TICK_LABEL_SIZE,
        length=_MAJOR_TICK_LENGTH,
        width=0.95,
        labelbottom=True,
        pad=3.5,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        colors=MUTED,
        length=_MINOR_TICK_LENGTH,
        width=0.6,
    )
    if show_xlabel:
        ax.set_xlabel(
            TIMELINE_X_LABEL,
            color=TEXT_SECONDARY,
            fontsize=_AXIS_LABEL_SIZE,
            labelpad=7,
        )
        try:
            ax.xaxis.label.set_clip_on(False)
        except Exception:
            pass


def configure_percent_axis(ax: Axes, *, show_xlabel: bool = True, nbins: int = 6) -> None:
    """Percent-scale X axis (gait cycle mode)."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=4, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        axis="x",
        which="major",
        colors=TEXT_SECONDARY,
        labelcolor=TEXT_SECONDARY,
        labelsize=_TICK_LABEL_SIZE,
        length=_MAJOR_TICK_LENGTH,
        width=0.95,
        labelbottom=True,
        pad=3.5,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        colors=MUTED,
        length=_MINOR_TICK_LENGTH,
        width=0.6,
    )
    if show_xlabel:
        ax.set_xlabel(
            GAIT_CYCLE_X_LABEL,
            color=TEXT_SECONDARY,
            fontsize=_AXIS_LABEL_SIZE,
            labelpad=7,
        )
        try:
            ax.xaxis.label.set_clip_on(False)
        except Exception:
            pass


def configure_numeric_y_axis(
    ax: Axes,
    ylabel: str,
    *,
    nbins: int = 5,
) -> None:
    """Numeric Y ticks with readable major/minor divisions."""
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=_AXIS_LABEL_SIZE, labelpad=7)
    try:
        ax.yaxis.label.set_clip_on(False)
    except Exception:
        pass
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        axis="y",
        which="major",
        colors=TEXT_SECONDARY,
        labelcolor=TEXT_SECONDARY,
        labelsize=_TICK_LABEL_SIZE,
        length=_MAJOR_TICK_LENGTH,
        width=0.95,
        pad=3.5,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        colors=MUTED,
        length=_MINOR_TICK_LENGTH,
        width=0.6,
    )


def configure_categorical_y_axis(
    ax: Axes,
    ylabel: str,
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
) -> None:
    """Discrete Y labels (gait phase rows) without minor tick clutter."""
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=_AXIS_LABEL_SIZE, labelpad=7)
    ax.set_yticks(list(tick_positions))
    ax.set_yticklabels(list(tick_labels), fontsize=_TICK_LABEL_SIZE, color=TEXT_SECONDARY)
    ax.tick_params(
        axis="y",
        which="major",
        colors=TEXT_SECONDARY,
        labelcolor=TEXT_SECONDARY,
        length=_MAJOR_TICK_LENGTH,
        width=0.95,
        pad=3.5,
    )
    ax.yaxis.set_minor_locator(AutoMinorLocator(1))


def finalize_stacked_time_axes(
    axes: Sequence[Axes],
    y_specs: Sequence[tuple[str, str] | tuple[str, str, Sequence[float], Sequence[str]]],
) -> None:
    """
    Apply consistent ticks to a vertical stack of time-series panels.

    Each *y_specs* entry is either:
    - ``("numeric", "Unit label")``
    - ``("categorical", "Label", positions, labels)``
    """
    n = len(axes)
    for i, ax in enumerate(axes):
        apply_chart_panel_style(ax)
        spec = y_specs[i] if i < len(y_specs) else ("numeric", "")
        y_minor = True
        if spec[0] == "categorical":
            _, ylabel, positions, labels = spec
            configure_categorical_y_axis(ax, ylabel, positions, labels)
            y_minor = False
        elif spec[0] == "numeric" and len(spec) > 1 and spec[1]:
            configure_numeric_y_axis(ax, spec[1])
        configure_time_axis(ax, show_xlabel=(i == n - 1))
        apply_chart_grid(ax, y_minor=y_minor)
        unclip_axis_chrome(ax)


def style_single_time_series_chart(
    ax: Axes,
    *,
    ylabel: str,
    x_is_percent: bool = False,
) -> None:
    """Full axis styling for a single-panel time or gait-cycle chart."""
    apply_chart_panel_style(ax)
    configure_numeric_y_axis(ax, ylabel)
    if x_is_percent:
        configure_percent_axis(ax, show_xlabel=True)
    else:
        configure_time_axis(ax, show_xlabel=True)
    apply_chart_grid(ax, y_minor=True)
    unclip_axis_chrome(ax)


def autoscale_y_with_padding(
    ax: Axes,
    values: Sequence[float] | None = None,
    *,
    pad_frac: float | None = None,
    min_span: float = 1e-3,
    include: Sequence[float] | None = None,
) -> tuple[float, float] | None:
    """Consistent Y limits with shared headroom (projector-friendly, non-overlapping)."""
    pad = Y_AUTOSCALE_PAD if pad_frac is None else float(pad_frac)
    arr: list[float] = []
    if values is not None:
        arr.extend(float(v) for v in values if v is not None and np_isfinite(v))
    if include is not None:
        arr.extend(float(v) for v in include if v is not None and np_isfinite(v))
    if len(arr) >= 2:
        lo, hi = min(arr), max(arr)
        span = max(hi - lo, min_span)
        ylim = (lo - span * pad, hi + span * pad)
        ax.set_ylim(*ylim)
        return ylim
    y0, y1 = ax.get_ylim()
    if not np_isfinite(y0) or not np_isfinite(y1) or y1 <= y0:
        return None
    span = max(y1 - y0, min_span)
    ylim = (y0 - span * pad * 0.5, y1 + span * pad * 0.5)
    ax.set_ylim(*ylim)
    return ylim


def np_isfinite(value: float) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "ANNOTATION_SIZE",
    "AXIS_LABEL_SIZE",
    "GAIT_CYCLE_X_LABEL",
    "LEGEND_OUTSIDE_BBOX",
    "LEGEND_OUTSIDE_LOC",
    "LEGEND_SIZE",
    "LINESTYLE_LEFT",
    "LINESTYLE_RIGHT",
    "PRESENTATION_CONTACT_MARGINS",
    "PRESENTATION_SINGLE_MARGINS",
    "style_3d_axis_labels",
    "PRESENTATION_STACK_MARGINS",
    "MOTION_KNEE_MARGINS",
    "SECONDARY_ANNOTATION_ATTR",
    "SERIES_LINE_WIDTH",
    "SERIES_LINE_WIDTH_SECONDARY",
    "SERIES_LINE_WIDTH_STEP",
    "SERIES_MARKER_SIZE",
    "TICK_LABEL_SIZE",
    "TIMELINE_X_LABEL",
    "TITLE_SIZE",
    "Y_AUTOSCALE_PAD",
    "_ANNOTATION_SIZE",
    "_AXIS_LABEL_SIZE",
    "_GRID_MAJOR_ALPHA",
    "_GRID_MINOR_ALPHA",
    "_LEGEND_SIZE",
    "_TICK_LABEL_SIZE",
    "_TITLE_SIZE",
    "apply_chart_grid",
    "apply_chart_panel_style",
    "apply_motion_knee_layout",
    "apply_presentation_layout",
    "autoscale_y_with_padding",
    "configure_categorical_y_axis",
    "configure_chart_antialiasing",
    "configure_numeric_y_axis",
    "configure_percent_axis",
    "configure_time_axis",
    "finalize_stacked_time_axes",
    "style_motion_knee_legend",
    "is_secondary_annotation",
    "legend_ncol_for",
    "mark_secondary_annotation",
    "mark_secondary_annotations",
    "series_plot_kwargs",
    "side_linestyle",
    "side_linestyle_for_item",
    "style_chart_legend",
    "style_chart_title",
    "style_figure_legend",
    "unclip_axis_chrome",
    "unclip_figure_chrome",
    "style_single_time_series_chart",
]
