"""Compare tab layout minsizes and heatmap readability helpers."""

from __future__ import annotations

from stablewalk.ui.tk.dashboard_compare import (
    COMPARE_GRAPH_MIN_H,
    COMPARE_METRICS_MIN_H,
    COMPARE_SKELETON_MIN_H,
    COMPARE_VIDEO_MIN_H,
)


def test_compare_section_minsizes_keep_panels_readable() -> None:
    assert COMPARE_VIDEO_MIN_H >= 280
    assert COMPARE_GRAPH_MIN_H >= 260
    assert COMPARE_SKELETON_MIN_H >= 200
    assert COMPARE_METRICS_MIN_H >= 200
    # Videos and graphs must dominate over chrome-sized metrics.
    assert COMPARE_VIDEO_MIN_H >= COMPARE_METRICS_MIN_H
    assert COMPARE_GRAPH_MIN_H >= 260


def test_heatmap_draw_sets_readable_margins() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.difference_heatmap import draw_difference_heatmap

    fig = Figure(figsize=(4.4, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    draw_difference_heatmap(ax, None, None)
    # Empty state should still leave readable axis area (no crash / no overlap).
    left = fig.subplotpars.left
    assert left >= 0.2
