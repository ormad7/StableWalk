"""Shared matplotlib axis styling for dashboard time-series charts.

Publication-quality chrome aligned with OpenSim / Visual3D / Vicon / Qualisys.
Display-only: does not alter plotted values or analysis math.
"""

from __future__ import annotations

from typing import Any, Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

from stablewalk.ui.colors import BORDER, MUTED, PANEL, TEXT

TIMELINE_X_LABEL = "Time (s)"
GAIT_CYCLE_X_LABEL = "Gait Cycle (%)"

# Larger type for thesis / conference readability.
_AXIS_LABEL_SIZE = 12.5
_TICK_LABEL_SIZE = 10.5
_TITLE_SIZE = 13.0
_LEGEND_SIZE = 10.0
_ANNOTATION_SIZE = 10.0
_MAJOR_TICK_LENGTH = 6.0
_MINOR_TICK_LENGTH = 3.25

# Primary series stroke (L/R joint traces, COM, etc.).
SERIES_LINE_WIDTH = 2.75
SERIES_LINE_WIDTH_SECONDARY = 2.25
SERIES_LINE_WIDTH_STEP = 2.35
SERIES_MARKER_SIZE = 36.0

# Left solid / right dashed — colorblind-friendly L/R distinction beyond hue.
LINESTYLE_LEFT: str | tuple = "-"
LINESTYLE_RIGHT: str | tuple = (0, (6.0, 2.75))

# Dark-theme grid: quiet major/minor (Visual3D / Qualisys).
_GRID_MAJOR_ALPHA = 0.22
_GRID_MINOR_ALPHA = 0.10
_GRID_MAJOR_LINEWIDTH = 0.55
_GRID_MINOR_LINEWIDTH = 0.35

# Smoother path rendering (matplotlib Agg / canvas).
_SERIES_ANTIALIASED = True
_SERIES_CAPSTYLE = "round"
_SERIES_JOINSTYLE = "round"


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
) -> dict[str, Any]:
    """Shared kwargs for scientifically faithful, high-legibility series strokes."""
    if linestyle is None:
        if side is not None:
            linestyle = side_linestyle(side)
        elif item_id is not None:
            linestyle = side_linestyle_for_item(item_id)
        else:
            linestyle = "-"
    return {
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


def apply_chart_panel_style(ax: Axes) -> None:
    """Panel background and research-lab spines (bottom + left only)."""
    configure_chart_antialiasing(ax.figure if ax is not None else None)
    ax.set_facecolor(PANEL)
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(BORDER)
            spine.set_linewidth(1.15)


def style_chart_title(ax: Axes, title: str, *, pad: float = 10.0) -> None:
    """Consistent publication title."""
    ax.set_title(
        title,
        color=TEXT,
        fontsize=_TITLE_SIZE,
        fontweight="medium",
        pad=pad,
    )


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
        "framealpha": 0.95,
        "borderpad": 0.45,
        "handlelength": 2.35,
        "handletextpad": 0.6,
        "labelspacing": 0.45,
        "columnspacing": 1.05,
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
    loc: str = "upper right",
    ncol: int | None = None,
    fontsize: float | None = None,
    bbox_to_anchor: tuple[float, float] | None = None,
    handles: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
):
    """Dynamic legend: deduped entries, adaptive columns, larger type."""
    if handles is None or labels is None:
        h, lab = ax.get_legend_handles_labels()
        handles, labels = _dedupe_legend_entries(h, lab)
    else:
        handles, labels = _dedupe_legend_entries(list(handles), list(labels))
    if not labels:
        return None
    use_ncol = legend_ncol_for(len(labels)) if ncol is None else max(1, int(ncol))
    use_fs = _LEGEND_SIZE if fontsize is None else float(fontsize)
    leg = ax.legend(
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
        frame = leg.get_frame()
        frame.set_linewidth(0.85)
        for text in leg.get_texts():
            text.set_fontsize(use_fs)
    return leg


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
        colors=MUTED,
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
            color=MUTED,
            fontsize=_AXIS_LABEL_SIZE,
            labelpad=7,
        )


def configure_percent_axis(ax: Axes, *, show_xlabel: bool = True, nbins: int = 6) -> None:
    """Percent-scale X axis (gait cycle mode)."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=4, integer=True))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        axis="x",
        which="major",
        colors=MUTED,
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
            color=MUTED,
            fontsize=_AXIS_LABEL_SIZE,
            labelpad=7,
        )


def configure_numeric_y_axis(
    ax: Axes,
    ylabel: str,
    *,
    nbins: int = 5,
) -> None:
    """Numeric Y ticks with readable major/minor divisions."""
    ax.set_ylabel(ylabel, color=MUTED, fontsize=_AXIS_LABEL_SIZE, labelpad=7)
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, min_n_ticks=3))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(
        axis="y",
        which="major",
        colors=MUTED,
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
    ax.set_ylabel(ylabel, color=MUTED, fontsize=_AXIS_LABEL_SIZE, labelpad=7)
    ax.set_yticks(list(tick_positions))
    ax.set_yticklabels(list(tick_labels), fontsize=_TICK_LABEL_SIZE, color=MUTED)
    ax.tick_params(
        axis="y",
        which="major",
        colors=MUTED,
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


def autoscale_y_with_padding(
    ax: Axes,
    values: Sequence[float] | None = None,
    *,
    pad_frac: float = 0.10,
    min_span: float = 1e-3,
) -> None:
    """Comfortable Y limits with headroom for event markers and labels."""
    if values is not None:
        arr = [float(v) for v in values if v is not None and np_isfinite(v)]
        if len(arr) >= 2:
            lo, hi = min(arr), max(arr)
            span = max(hi - lo, min_span)
            ax.set_ylim(lo - span * pad_frac, hi + span * pad_frac)
            return
    y0, y1 = ax.get_ylim()
    if not np_isfinite(y0) or not np_isfinite(y1) or y1 <= y0:
        return
    span = max(y1 - y0, min_span)
    ax.set_ylim(y0 - span * pad_frac * 0.5, y1 + span * pad_frac * 0.5)


def np_isfinite(value: float) -> bool:
    try:
        import math

        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


__all__ = [
    "GAIT_CYCLE_X_LABEL",
    "LINESTYLE_LEFT",
    "LINESTYLE_RIGHT",
    "SERIES_LINE_WIDTH",
    "SERIES_LINE_WIDTH_SECONDARY",
    "SERIES_LINE_WIDTH_STEP",
    "SERIES_MARKER_SIZE",
    "TIMELINE_X_LABEL",
    "_ANNOTATION_SIZE",
    "_AXIS_LABEL_SIZE",
    "_GRID_MAJOR_ALPHA",
    "_GRID_MINOR_ALPHA",
    "_LEGEND_SIZE",
    "_TICK_LABEL_SIZE",
    "_TITLE_SIZE",
    "apply_chart_grid",
    "apply_chart_panel_style",
    "autoscale_y_with_padding",
    "configure_categorical_y_axis",
    "configure_chart_antialiasing",
    "configure_numeric_y_axis",
    "configure_percent_axis",
    "configure_time_axis",
    "finalize_stacked_time_axes",
    "legend_ncol_for",
    "series_plot_kwargs",
    "side_linestyle",
    "side_linestyle_for_item",
    "style_chart_legend",
    "style_chart_title",
    "style_figure_legend",
    "style_single_time_series_chart",
]
