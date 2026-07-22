"""Shared matplotlib axis styling for dashboard time-series charts.

Publication-quality chrome aligned with OpenSim / Visual3D / Vicon / Qualisys.
"""

from __future__ import annotations

from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

from stablewalk.ui.colors import BORDER, MUTED, PANEL, TEXT

TIMELINE_X_LABEL = "Time (s)"
GAIT_CYCLE_X_LABEL = "Gait Cycle (%)"

# Larger type for thesis / conference readability.
_AXIS_LABEL_SIZE = 10.5
_TICK_LABEL_SIZE = 9.0
_TITLE_SIZE = 11.0
_MAJOR_TICK_LENGTH = 5.5
_MINOR_TICK_LENGTH = 3.0

# Dark-theme grid: quiet major/minor (Visual3D / Qualisys).
_GRID_MAJOR_ALPHA = 0.22
_GRID_MINOR_ALPHA = 0.10
_GRID_MAJOR_LINEWIDTH = 0.55
_GRID_MINOR_LINEWIDTH = 0.35


def apply_chart_panel_style(ax: Axes) -> None:
    """Panel background and research-lab spines (bottom + left only)."""
    ax.set_facecolor(PANEL)
    for side, spine in ax.spines.items():
        if side in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_visible(True)
            spine.set_color(BORDER)
            spine.set_linewidth(1.0)


def style_chart_title(ax: Axes, title: str, *, pad: float = 6.0) -> None:
    """Consistent publication title."""
    ax.set_title(
        title,
        color=TEXT,
        fontsize=_TITLE_SIZE,
        fontweight="medium",
        pad=pad,
    )


def style_chart_legend(
    ax: Axes,
    *,
    loc: str = "upper right",
    ncol: int = 1,
    fontsize: float = 8.0,
):
    """Compact legend matching Qualisys / Visual3D chart chrome."""
    leg = ax.legend(
        loc=loc,
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        fancybox=False,
        edgecolor=BORDER,
        facecolor=PANEL,
        labelcolor=TEXT,
        framealpha=0.94,
        borderpad=0.4,
        handlelength=1.8,
        handletextpad=0.55,
        labelspacing=0.4,
        columnspacing=0.9,
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_linewidth(0.7)
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
        width=0.85,
        labelbottom=True,
        pad=3,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        colors=MUTED,
        length=_MINOR_TICK_LENGTH,
        width=0.55,
    )
    if show_xlabel:
        ax.set_xlabel(
            TIMELINE_X_LABEL,
            color=MUTED,
            fontsize=_AXIS_LABEL_SIZE,
            labelpad=6,
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
        width=0.85,
        labelbottom=True,
        pad=3,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        colors=MUTED,
        length=_MINOR_TICK_LENGTH,
        width=0.55,
    )
    if show_xlabel:
        ax.set_xlabel(
            GAIT_CYCLE_X_LABEL,
            color=MUTED,
            fontsize=_AXIS_LABEL_SIZE,
            labelpad=6,
        )


def configure_numeric_y_axis(
    ax: Axes,
    ylabel: str,
    *,
    nbins: int = 5,
) -> None:
    """Numeric Y ticks with readable major/minor divisions."""
    ax.set_ylabel(ylabel, color=MUTED, fontsize=_AXIS_LABEL_SIZE, labelpad=6)
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
        width=0.85,
        pad=3,
    )
    ax.tick_params(
        axis="y",
        which="minor",
        colors=MUTED,
        length=_MINOR_TICK_LENGTH,
        width=0.55,
    )


def configure_categorical_y_axis(
    ax: Axes,
    ylabel: str,
    tick_positions: Sequence[float],
    tick_labels: Sequence[str],
) -> None:
    """Discrete Y labels (gait phase rows) without minor tick clutter."""
    ax.set_ylabel(ylabel, color=MUTED, fontsize=_AXIS_LABEL_SIZE, labelpad=6)
    ax.set_yticks(list(tick_positions))
    ax.set_yticklabels(list(tick_labels), fontsize=_TICK_LABEL_SIZE, color=MUTED)
    ax.tick_params(
        axis="y",
        which="major",
        colors=MUTED,
        length=_MAJOR_TICK_LENGTH,
        width=0.85,
        pad=3,
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
    "TIMELINE_X_LABEL",
    "_GRID_MAJOR_ALPHA",
    "_GRID_MINOR_ALPHA",
    "apply_chart_grid",
    "apply_chart_panel_style",
    "autoscale_y_with_padding",
    "configure_categorical_y_axis",
    "configure_numeric_y_axis",
    "configure_percent_axis",
    "configure_time_axis",
    "finalize_stacked_time_axes",
    "style_chart_legend",
    "style_chart_title",
    "style_single_time_series_chart",
]
