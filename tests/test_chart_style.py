"""Tests for shared dashboard chart axis styling."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_style import (
    TIMELINE_X_LABEL,
    apply_chart_grid,
    configure_time_axis,
    finalize_stacked_time_axes,
    style_single_time_series_chart,
)


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
