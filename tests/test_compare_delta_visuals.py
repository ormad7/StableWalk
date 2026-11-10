"""Tests for Compare visual delta chrome."""

from __future__ import annotations

import tkinter as tk

from stablewalk.analysis.session_compare import CompareMetrics, MetricDiff, SessionComparisonResult
from stablewalk.ui.tk.compare_delta_visuals import (
    DIFFERENCE_SUMMARY_SPECS,
    fill_difference_summary,
    fill_visual_diffs,
    format_pct,
    install_difference_summary_panel,
    install_visual_diff_chrome,
    pct_change,
    tone_color,
)
from stablewalk.ui.theme import DANGER, SUCCESS


def test_pct_change_and_tone() -> None:
    assert pct_change(100.0, 110.0) == 10.0
    assert pct_change(0.0, 1.0) is None
    assert format_pct(11.4) == "+11%"
    assert tone_color("better") != tone_color("worse")


def test_difference_summary_shows_placeholder_until_session_b_ready() -> None:
    from stablewalk.ui.tk.compare_state import SESSION_B_METRICS_PLACEHOLDER_TITLE

    root = tk.Tk()
    root.withdraw()
    try:
        host = tk.Frame(root)
        handles = install_difference_summary_panel(host)
        root.update_idletasks()
        # Empty Session B → placeholder card, no blank Δ cells.
        fill_difference_summary(handles, CompareMetrics(label="A"), None, ready=False)
        assert handles["mode"] == "placeholder"
        assert handles["grid"].winfo_manager() == ""
        assert handles["placeholder"].winfo_manager() == "pack"
        assert SESSION_B_METRICS_PLACEHOLDER_TITLE in handles["placeholder"].winfo_children()[0].cget(
            "text"
        )
    finally:
        root.destroy()


def test_difference_summary_panel_highlights_improvement_and_deterioration() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        host = tk.Frame(root)
        handles = install_difference_summary_panel(host)
        assert len(DIFFERENCE_SUMMARY_SPECS) == 5
        assert set(handles["cells"]) == {spec[0] for spec in DIFFERENCE_SUMMARY_SPECS}

        left = CompareMetrics(
            label="A",
            cadence_spm=100.0,
            walking_speed_m_s=1.0,
            knee_rom_deg=50.0,
            symmetry_pct=80.0,
            stability_margin_m=0.04,
        )
        right = CompareMetrics(
            label="B",
            cadence_spm=110.0,  # improvement
            walking_speed_m_s=0.8,  # deterioration
            knee_rom_deg=60.0,  # improvement
            symmetry_pct=70.0,  # deterioration
            stability_margin_m=0.06,  # improvement
        )
        fill_difference_summary(handles, left, right, ready=True)
        assert handles["mode"] == "metrics"
        assert handles["grid"].winfo_manager() == "pack"
        assert handles["placeholder"].winfo_manager() == ""
        cells = handles["cells"]
        assert cells["Cadence difference"]["delta"].cget("fg") == SUCCESS
        assert cells["Speed difference"]["delta"].cget("fg") == DANGER
        assert cells["ROM difference"]["delta"].cget("fg") == SUCCESS
        assert cells["Symmetry difference"]["delta"].cget("fg") == DANGER
        assert cells["Stability difference"]["delta"].cget("fg") == SUCCESS
        # Percent differences auto-populate with tone colors.
        assert cells["Cadence difference"]["pct"].cget("text") == "+10%"
        assert cells["Cadence difference"]["pct"].cget("fg") == SUCCESS
        assert cells["Speed difference"]["pct"].cget("text") == "-20%"
        assert cells["Speed difference"]["pct"].cget("fg") == DANGER
    finally:
        root.destroy()


def test_visual_diff_panel_builds_rows() -> None:
    root = tk.Tk()
    root.withdraw()
    try:
        frame = tk.Frame(root)
        chrome = install_visual_diff_chrome(frame)
        result = SessionComparisonResult(
            left=CompareMetrics(label="A"),
            right=CompareMetrics(label="B"),
            diffs=[
                MetricDiff(
                    name="Cadence difference",
                    left_value=110.0,
                    right_value=122.0,
                    delta=12.0,
                    unit=" steps/min",
                    higher_is_better=True,
                    tone_right="better",
                    display="Cadence difference: +12 steps/min",
                ),
                MetricDiff(
                    name="ROM difference",
                    left_value=60.0,
                    right_value=48.0,
                    delta=-12.0,
                    unit="°",
                    higher_is_better=True,
                    tone_right="worse",
                    display="ROM difference: -12°",
                ),
            ],
        )
        fill_visual_diffs(
            chrome["diff_host"],
            result,
            series_a=[10, 40, 70, 40, 10],
            series_b=[12, 35, 55, 35, 12],
            spark_canvas=chrome["spark_canvas"],
        )
        root.update_idletasks()
        assert len(chrome["diff_host"].winfo_children()) >= 2
    finally:
        root.destroy()


def test_sparkline_configure_does_not_recurse() -> None:
    """Regression: canvas.configure(width/height) must not re-enter paint forever."""
    from stablewalk.ui.tk.compare_delta_visuals import draw_dual_sparkline

    root = tk.Tk()
    root.withdraw()
    try:
        spark = tk.Canvas(root, width=200, height=44)
        spark.pack()
        root.update_idletasks()
        for _ in range(8):
            fill_visual_diffs(
                tk.Frame(root),
                None,
                series_a=[1, 2, 3, 4],
                series_b=[2, 3, 2, 1],
                spark_canvas=spark,
            )
        root.update()
        draw_dual_sparkline(spark, [1, 5, 2], [2, 4, 3])
        spark.event_generate("<Configure>")
        root.update()
        assert getattr(spark, "_spark_configure_bound", False) is True
    finally:
        root.destroy()
