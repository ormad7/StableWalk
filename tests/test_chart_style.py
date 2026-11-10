"""Tests for shared dashboard chart axis styling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.colors import TEXT, TEXT_SECONDARY
from stablewalk.ui.viewers.chart_style import (
    TIMELINE_X_LABEL,
    apply_chart_grid,
    apply_chart_panel_style,
    configure_time_axis,
    finalize_stacked_time_axes,
    style_3d_axis_labels,
    style_single_time_series_chart,
    unclip_axis_chrome,
)


def _luminance(color) -> float:
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_apply_chart_panel_style_uses_light_tick_text() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    apply_chart_panel_style(ax)
    assert ax.xaxis._major_tick_kw.get("labelcolor") == TEXT_SECONDARY
    assert _luminance(ax.xaxis.label.get_color()) >= 0.62


def test_unclip_recolors_black_titles_and_axes() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.set_title("Knee flexion", color="black")
    ax.set_xlabel("Time (s)", color="k")
    ax.set_ylabel("deg", color="#000000")
    unclip_axis_chrome(ax)
    assert ax.title.get_color() == TEXT
    assert bool(ax.title.get_clip_on()) is False
    assert ax.xaxis.label.get_color() == TEXT_SECONDARY
    assert ax.yaxis.label.get_color() == TEXT_SECONDARY
    assert bool(ax.xaxis.label.get_clip_on()) is False


def test_style_3d_axis_labels_are_light() -> None:
    fig = Figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")
    style_3d_axis_labels(ax)
    assert _luminance(ax.xaxis.label.get_color()) >= 0.62
    assert _luminance(ax.yaxis.label.get_color()) >= 0.62
    assert _luminance(ax.zaxis.label.get_color()) >= 0.62
    assert bool(ax.xaxis.label.get_clip_on()) is False


def test_style_single_time_series_chart_sets_time_and_units() -> None:
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0, 2.0], [0.5, 0.6, 0.55])

    style_single_time_series_chart(ax, ylabel="COM Height (m)")

    assert ax.get_xlabel() == TIMELINE_X_LABEL
    assert ax.get_ylabel() == "COM Height (m)"
    assert len(ax.get_xticklabels()) >= 2
    assert len(ax.get_yticklabels()) >= 2
    assert ax.xaxis._major_tick_kw["gridOn"]
    assert ax.xaxis._minor_tick_kw["gridOn"]
    assert ax.yaxis._major_tick_kw["gridOn"]
    assert ax.yaxis._minor_tick_kw["gridOn"]


def test_apply_chart_grid_categorical_row_uses_x_minor_only() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    apply_chart_grid(ax, y_minor=False)
    assert ax.xaxis._minor_tick_kw["gridOn"]
    assert ax.yaxis._minor_tick_kw["gridOn"] is False


def test_finalize_stacked_time_axes_labels_each_row() -> None:
    fig = Figure(figsize=(6, 8))
    axes = fig.subplots(3, 1, sharex=True)
    for ax in axes:
        ax.plot([0.0, 1.0, 2.0], [0.1, 0.2, 0.15])

    finalize_stacked_time_axes(
        axes,
        [
            ("numeric", "Contact"),
            ("categorical", "Gait Phase", [0.0, 0.5, 1.0], ["Swing", "Stance", "Double support"]),
            ("numeric", "Virtual GRF (BW)"),
        ],
    )

    assert axes[0].get_xlabel() == ""
    assert axes[2].get_xlabel() == TIMELINE_X_LABEL
    assert axes[0].get_ylabel() == "Contact"
    assert axes[2].get_ylabel() == "Virtual GRF (BW)"
    assert len(axes[1].get_yticklabels()) == 3
    for ax in axes:
        assert len(ax.get_xticklabels()) >= 2


def test_configure_time_axis_respects_show_xlabel_flag() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    configure_time_axis(ax, show_xlabel=False)
    assert ax.get_xlabel() == ""


def test_series_plot_kwargs_left_solid_right_dashed() -> None:
    from stablewalk.ui.viewers.chart_style import (
        LINESTYLE_LEFT,
        LINESTYLE_RIGHT,
        SERIES_LINE_WIDTH,
        series_plot_kwargs,
        side_linestyle,
    )

    left = series_plot_kwargs(color="#22c55e", side="left", label="L")
    right = series_plot_kwargs(color="#ef4444", side="right", label="R")
    assert left["linestyle"] == LINESTYLE_LEFT
    assert right["linestyle"] == LINESTYLE_RIGHT
    assert left["linewidth"] == SERIES_LINE_WIDTH
    assert left["antialiased"] is True
    assert side_linestyle("right_knee") == LINESTYLE_RIGHT


def test_legend_ncol_scales_with_entries() -> None:
    from stablewalk.ui.viewers.chart_style import legend_ncol_for

    assert legend_ncol_for(1) == 1
    assert legend_ncol_for(3) == 3
    assert legend_ncol_for(5) == 3
    assert legend_ncol_for(8) >= 2


def test_style_chart_legend_is_dynamic() -> None:
    from stablewalk.ui.viewers.chart_style import style_chart_legend

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1], label="Left knee")
    ax.plot([0, 1], [1, 0], label="Right knee")
    leg = style_chart_legend(ax)
    assert leg is not None
    assert len(leg.get_texts()) == 2
    assert leg.get_texts()[0].get_fontsize() >= 11.0
    # Outside-axes placement by default (projector-friendly).
    bb = leg.get_bbox_to_anchor()
    assert bb is not None


def test_presentation_layout_widens_plot_area() -> None:
    from stablewalk.ui.viewers.chart_style import (
        PRESENTATION_SINGLE_MARGINS,
        apply_presentation_layout,
    )

    fig = Figure(figsize=(6, 4))
    fig.add_subplot(111)
    apply_presentation_layout(fig, mode="single")
    assert fig.subplotpars.right == PRESENTATION_SINGLE_MARGINS["right"]
    assert fig.subplotpars.left == PRESENTATION_SINGLE_MARGINS["left"]
    # Wider data band than the previous right=0.78 convention.
    assert fig.subplotpars.right - fig.subplotpars.left > 0.75


def test_motion_knee_layout_keeps_chrome_inside_panel() -> None:
    from stablewalk.ui.viewers.chart_style import (
        MOTION_KNEE_MARGINS,
        apply_motion_knee_layout,
        style_motion_knee_legend,
    )

    fig = Figure(figsize=(6.5, 3.5))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0], [10.0, 20.0], label="Left")
    ax.plot([0.0, 1.0], [12.0, 18.0], label="Right")
    ax.set_title("Knee flexion · Pose-derived")
    style_motion_knee_legend(ax, ncol=1)
    apply_motion_knee_layout(fig)

    assert fig.subplotpars.left == MOTION_KNEE_MARGINS["left"]
    assert fig.subplotpars.right == MOTION_KNEE_MARGINS["right"]
    assert fig.subplotpars.top == MOTION_KNEE_MARGINS["top"]
    assert fig.subplotpars.bottom == MOTION_KNEE_MARGINS["bottom"]
    # Legend stays inside the axes so it cannot clip off the panel.
    leg = ax.get_legend()
    assert leg is not None
    assert leg._loc == 1 or str(leg._loc) in ("1", "upper right")  # type: ignore[attr-defined]
    # Data band is wide — no right-side legend rail stealing plot width.
    assert fig.subplotpars.right - fig.subplotpars.left > 0.80


def test_autoscale_y_uses_shared_padding() -> None:
    from stablewalk.ui.viewers.chart_style import Y_AUTOSCALE_PAD, autoscale_y_with_padding

    fig = Figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    ylim = autoscale_y_with_padding(ax, [0.0, 100.0])
    assert ylim is not None
    span = ylim[1] - ylim[0]
    assert abs(span - 100.0 * (1.0 + 2.0 * Y_AUTOSCALE_PAD)) < 1e-6


def test_compact_mpl_toolbar_omits_subplot_config() -> None:
    from stablewalk.ui.tk.mpl_toolbar import CompactNavigationToolbar

    names = [item[0] for item in CompactNavigationToolbar.toolitems]
    assert "Subplots" not in names
    assert "Home" in names
    assert "Save" in names


def test_bh_is_defined_in_methods_and_com_tooltip() -> None:
    from stablewalk.ui.metric_tooltips import get_metric_tooltip
    from stablewalk.ui.scientific_labels import BH_DEFINITION, METHODS_OVERVIEW

    assert "body height" in BH_DEFINITION.lower()
    assert "pelvis-relative" in BH_DEFINITION.lower()
    assert BH_DEFINITION in METHODS_OVERVIEW
    tip = get_metric_tooltip("com")
    assert tip is not None
    assert "BH" in tip
    assert "body height" in tip.lower()
