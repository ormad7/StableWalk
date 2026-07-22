"""Visualization tests for the Biomechanics dashboard charts."""

from __future__ import annotations

from dataclasses import replace

from matplotlib.figure import Figure

from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
from stablewalk.ui.viewers.biomechanics_charts import draw_biomechanics_dashboard
from tests.test_biomechanical_analysis import _contact, _cycles, _recording


def _sample_result():
    rec = _recording(24)
    contact = _contact(24)
    frames = list(contact.per_frame)
    for i in range(0, 24, 10):
        frames[i] = replace(frames[i], left_heel_strike=1, right_heel_strike=1)
        if i + 5 < 24:
            frames[i + 5] = replace(frames[i + 5], left_toe_off=1)
        if i + 7 < 24:
            frames[i + 7] = replace(frames[i + 7], right_toe_off=1)
    contact = replace(contact, per_frame=frames)
    return run_biomechanical_analysis(rec, cycles=_cycles(24), contact=contact)


def test_biomechanics_dashboard_axes_have_legends_and_units() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 9.5))
    draw_biomechanics_dashboard(fig, result)

    axes = fig.get_axes()
    assert len(axes) == 4

    assert axes[0].get_legend() is not None
    assert "Height" in axes[0].get_ylabel()
    assert "bh" in axes[0].get_ylabel().lower() or "body" in axes[0].get_ylabel().lower()
    assert axes[1].get_legend() is not None
    assert "Margin" in axes[1].get_ylabel()
    assert axes[1].get_ylabel().lower().endswith("(m)") or "m" in axes[1].get_ylabel().lower()
    assert axes[2].get_legend() is not None
    assert "Asymmetry" in axes[2].get_ylabel()
    assert axes[3].get_legend() is not None
    assert "Time (s)" in axes[3].get_xlabel()
    assert "Phase" in axes[3].get_ylabel()


def test_biomechanics_dashboard_draws_gait_event_vlines_on_all_rows() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 9.5))
    draw_biomechanics_dashboard(fig, result)

    for ax in fig.get_axes():
        # Publication markers: dashed/dotted event guides + HS/TO scatter glyphs.
        event_guides = [
            line
            for line in ax.lines
            if line.get_linestyle() not in ("-", "None", "none", "")
            and (line.get_alpha() or 0) > 0.2
            and line.get_linewidth() <= 1.2
        ]
        event_scatters = [
            coll
            for coll in ax.collections
            if getattr(coll, "get_offsets", None) is not None and len(coll.get_offsets()) > 0
        ]
        assert event_guides or event_scatters, (
            f"Expected heel-strike / toe-off markers on {ax.get_title()}"
        )


def test_biomechanics_dashboard_registers_hover_points() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 9.5))
    draw_biomechanics_dashboard(fig, result)

    points = getattr(fig, "_biomech_hover_points", [])
    assert len(points) > 10
    assert any("COM height" in pt.text for pt in points)
    assert any("Margin" in pt.text for pt in points)
