"""Tests for publication-quality chart reference overlays (display only)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_playhead import PlayheadState, draw_chart_playhead
from stablewalk.ui.viewers.chart_reference import (
    KNEE_FLEXION_NORMAL_DEG,
    draw_confidence_overlay,
    draw_gait_event_markers,
    draw_reference_y_bands,
)
from stablewalk.ui.viewers.chart_style import style_single_time_series_chart


def test_draw_reference_y_bands_adds_spans() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 1.0, 2.0], [10.0, 40.0, 55.0])
    ax.set_ylim(-5, 100)
    before = len(ax.patches)
    draw_reference_y_bands(
        ax,
        normal=KNEE_FLEXION_NORMAL_DEG,
        abnormal_below=-2.0,
        abnormal_above=90.0,
    )
    assert len(ax.patches) > before


def test_draw_confidence_overlay_shades_low_regions() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    times = [0.0, 0.1, 0.2, 0.3, 0.4]
    conf = [0.9, 0.2, 0.2, 0.9, 0.9]
    before = len(ax.patches)
    draw_confidence_overlay(ax, times, conf, threshold=0.55)
    assert len(ax.patches) > before


def test_draw_gait_event_markers_adds_scatters() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    before = len(ax.collections)
    draw_gait_event_markers(
        ax,
        left_hs=[0.5],
        right_to=[1.2],
        show_legend=True,
    )
    assert len(ax.collections) > before


def test_playhead_value_label_rendered() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    style_single_time_series_chart(ax, ylabel="Knee flexion (°)")
    ax.plot([0.0, 2.0], [20.0, 40.0])
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 50)
    draw_chart_playhead(
        ax,
        PlayheadState(time_s=1.0, frame_index=12),
        show_label=True,
        value_label="L 32.4° · R 28.1°",
        value_y=32.4,
    )
    texts = [t.get_text() for t in ax.texts]
    assert any("F12" in t for t in texts)
    assert any("32.4" in t for t in texts)
    assert len(ax._stablewalk_playhead_artists) == 6
