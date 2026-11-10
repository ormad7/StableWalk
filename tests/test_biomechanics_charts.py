"""Visualization tests for the Biomechanics dashboard charts."""

from __future__ import annotations

from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.figure import Figure

from stablewalk.analysis.biomechanical.orchestrator import run_biomechanical_analysis
from stablewalk.ui.viewers.biomechanics_charts import (
    BIOMECH_SERIES_COM,
    BIOMECH_SERIES_CONTACT,
    BIOMECH_SERIES_PHASE,
    BIOMECH_SERIES_STABILITY,
    biomech_series_unavailable_reason,
    draw_biomechanics_dashboard,
)
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


def test_biomechanics_dashboard_single_primary_axes_with_units() -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    result = _sample_result()
    fig = Figure(figsize=(11, 4.8), dpi=100)
    FigureCanvasAgg(fig)
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_COM)

    axes = fig.get_axes()
    assert len(axes) == 1
    assert axes[0].get_legend() is not None
    assert "Pelvis-relative" in axes[0].get_ylabel()
    legend_text = " ".join(text.get_text() for text in axes[0].get_legend().get_texts())
    assert "Illustrative pelvis-relative band" in legend_text
    assert "Normal" not in legend_text
    assert "Time (s)" in axes[0].get_xlabel()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = axes[0]
    legend = ax.get_legend()
    assert legend is not None
    assert bool(legend.get_clip_on())
    leg_bbox = legend.get_window_extent(renderer=renderer)
    ax_bbox = ax.bbox
    assert leg_bbox.x0 >= ax_bbox.x0 - 8.0
    assert leg_bbox.x1 <= ax_bbox.x1 + 8.0
    assert leg_bbox.y0 >= ax_bbox.y0 - 8.0
    assert leg_bbox.y1 <= ax_bbox.y1 + 8.0
    assert fig.subplotpars.bottom >= 0.14
    assert fig.subplotpars.top - fig.subplotpars.bottom >= 0.55


def test_biomechanics_series_selector_views() -> None:
    result = _sample_result()
    cases = (
        (BIOMECH_SERIES_COM, "Pelvis-relative"),
        (BIOMECH_SERIES_STABILITY, "distance"),
        (BIOMECH_SERIES_CONTACT, "Asymmetry"),
        (BIOMECH_SERIES_PHASE, "Phase"),
    )
    for series, ylabel_part in cases:
        fig = Figure(figsize=(11, 4.8))
        draw_biomechanics_dashboard(fig, result, series=series)
        axes = fig.get_axes()
        assert len(axes) == 1
        assert ylabel_part.lower() in (axes[0].get_ylabel() or "").lower()
        assert "Time (s)" in (axes[0].get_xlabel() or "")


def test_biomechanics_dashboard_draws_event_guides_on_com() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_COM)

    ax = fig.get_axes()[0]
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
    assert event_guides or event_scatters


def test_biomechanics_dashboard_single_playhead_label() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(fig, result, playhead_time_s=0.4, series=BIOMECH_SERIES_COM)

    labeled = 0
    for ax in fig.get_axes():
        artists = getattr(ax, "_stablewalk_playhead_artists", None)
        assert artists is not None
        label = artists[4] if len(artists) > 4 else None
        if label is not None:
            labeled += 1
    assert labeled == 1


def test_biomechanics_layout_respects_min_host_height() -> None:
    from stablewalk.ui.viewers.biomechanics_charts import (
        BIOMECH_TECH_CHART_MIN_H,
        layout_biomechanics_figure,
    )

    result = _sample_result()
    fig = Figure(figsize=(12.0, BIOMECH_TECH_CHART_MIN_H / 100.0), dpi=100)
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_COM)
    layout_biomechanics_figure(fig, size_known=True)
    height_px = fig.get_size_inches()[1] * fig.get_dpi()
    assert height_px >= BIOMECH_TECH_CHART_MIN_H - 1.0
    assert len(fig.get_axes()) == 1
    assert "Time" in (fig.get_axes()[0].get_xlabel() or "")


def test_biomechanics_layout_reserves_y_axis_label_margin() -> None:
    from stablewalk.ui.viewers.biomechanics_charts import layout_biomechanics_figure

    fig = Figure(figsize=(9.0, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_ylabel("Center of Mass Height (body heights)")

    layout_biomechanics_figure(fig, size_known=True)

    assert ax.get_position().x0 >= 0.115
    assert ax.yaxis.get_label().get_position()[0] <= -0.07


def test_biomechanics_page_gives_graph_most_width() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import (
        BIOMECH_CHART_COL_MIN_W,
        BIOMECH_CHART_FIGSIZE,
        BIOMECH_CLINICAL_MAX_W,
        BIOMECH_FINDINGS_MAX_W,
        biomech_balanced_column_widths,
    )
    from stablewalk.ui.viewers.biomechanics_charts import BIOMECH_TECH_CHART_MIN_H

    findings, chart, clinical = biomech_balanced_column_widths(1600)
    assert chart > findings
    assert chart > clinical
    assert BIOMECH_FINDINGS_MAX_W < BIOMECH_CHART_COL_MIN_W
    assert BIOMECH_CLINICAL_MAX_W <= 380
    assert clinical >= 260
    assert BIOMECH_CHART_FIGSIZE[0] >= 13.0
    assert BIOMECH_CHART_FIGSIZE[1] >= 8.5
    assert BIOMECH_TECH_CHART_MIN_H >= 680


def test_clinical_paragraph_avoids_dense_blocks() -> None:
    from stablewalk.ui.tk.dashboard_biomechanics import _clinical_paragraph

    single = _clinical_paragraph("One short sentence.")
    assert "\n\n" not in single
    multi = _clinical_paragraph(
        "First educational sentence stays readable. "
        "Second sentence starts a new airy paragraph. "
        "Third sentence keeps breathing room."
    )
    assert multi.count("\n\n") >= 2
    assert "First educational" in multi
    assert _clinical_paragraph("") == "—"
    assert _clinical_paragraph("   ") == "—"


def test_biomechanics_dashboard_registers_hover_points() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_COM)

    points = getattr(fig, "_biomech_hover_points", [])
    assert len(points) > 10
    assert any("COM vertical position" in pt.text for pt in points)


def test_biomech_panels_show_unavailable_reason_instead_of_blank() -> None:
    from stablewalk.ui.viewers.biomechanics_charts import (
        BALANCE_UNAVAILABLE_REASON,
        COM_UNAVAILABLE_REASON,
        GAIT_PHASE_UNAVAILABLE_REASON,
        NO_ANALYSIS_REASON,
        biomech_series_unavailable_reason,
    )

    assert biomech_series_unavailable_reason(None, BIOMECH_SERIES_COM) == NO_ANALYSIS_REASON
    empty = replace(_sample_result(), center_of_mass=None, stability_margin=None, contact=None)
    assert biomech_series_unavailable_reason(empty, BIOMECH_SERIES_COM) == COM_UNAVAILABLE_REASON
    assert biomech_series_unavailable_reason(empty, BIOMECH_SERIES_STABILITY) == BALANCE_UNAVAILABLE_REASON
    assert biomech_series_unavailable_reason(empty, BIOMECH_SERIES_PHASE) == GAIT_PHASE_UNAVAILABLE_REASON
    for series, reason in (
        (BIOMECH_SERIES_COM, COM_UNAVAILABLE_REASON),
        (BIOMECH_SERIES_STABILITY, BALANCE_UNAVAILABLE_REASON),
        (BIOMECH_SERIES_PHASE, GAIT_PHASE_UNAVAILABLE_REASON),
    ):
        fig = Figure(figsize=(8, 3))
        draw_biomechanics_dashboard(fig, empty, series=series)
        texts = [t.get_text() for t in fig.get_axes()[0].texts]
        assert any(reason in text for text in texts)
        assert any("Prerequisite" in text or "prerequisite" in text.lower() for text in texts)
        assert len(fig.get_axes()[0].lines) == 0
        empty_artist = fig.get_axes()[0].texts[0]
        assert getattr(empty_artist, "get_wrap", lambda: False)() is False
        tiny = Figure(figsize=(0.2, 0.2), dpi=10)
        draw_biomechanics_dashboard(tiny, empty, series=series)
        tiny_text = " ".join(t.get_text() or "" for t in tiny.get_axes()[0].texts)
        assert reason.split("\n", 1)[0] in tiny_text


def test_abnormal_gait_com_and_balance_draw_when_analysis_exists() -> None:
    """abnormal_gait_v2 has 3D joints — COM/Balance must plot, not fabricate."""
    import json
    from pathlib import Path

    import pytest

    from stablewalk import config
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.io.pose_loader import load_pose_sequence

    matches = []
    if not config.POSES_DIR.exists():
        pytest.skip("No pose dump directory")
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(str(raw.get("source_video") or "")).name == "abnormal_gait_v2.mp4":
            matches.append(path)
    if not matches:
        pytest.skip("abnormal_gait_v2 pose dump not present")
    seq = load_pose_sequence(max(matches, key=lambda p: p.stat().st_mtime))
    rec = pose_sequence_to_gait_motion(seq, detected_only=True)
    cycles = analyze_gait_cycles(rec)
    contact = analyze_foot_contact(rec, cycles=cycles)
    from stablewalk.analysis.estimated_vgrf_analysis import analyze_estimated_vgrf

    vgrf = analyze_estimated_vgrf(rec, contact, sequence=seq, body_mass_kg=70.0)
    for i, frame in enumerate(contact.per_frame):
        if i >= len(vgrf.left_vgrf_vertical):
            break
        if not frame.left_contact_known:
            assert np.isnan(vgrf.left_vgrf_vertical[i])
        if not frame.right_contact_known:
            assert np.isnan(vgrf.right_vgrf_vertical[i])
    result = run_biomechanical_analysis(rec, seq, cycles=cycles, contact=contact)
    assert biomech_series_unavailable_reason(result, BIOMECH_SERIES_COM) is None
    assert biomech_series_unavailable_reason(result, BIOMECH_SERIES_STABILITY) is None
    for series in (BIOMECH_SERIES_COM, BIOMECH_SERIES_STABILITY):
        fig = Figure(figsize=(8, 3))
        draw_biomechanics_dashboard(fig, result, series=series)
        ax = fig.get_axes()[0]
        assert ax.lines, f"{series} must plot existing samples"
        texts = " ".join(t.get_text() or "" for t in ax.texts).lower()
        assert "unavailable" not in texts


def test_phase_chart_uses_integer_categories_including_unknown() -> None:
    result = _sample_result()
    fig = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_PHASE)
    ax = fig.get_axes()[0]
    labels = [text.get_text() for text in ax.get_yticklabels()]
    assert "Unknown" in labels
    assert "Swing" in labels
    assert "Stance" in labels
    foot_lines = [
        line
        for line in ax.lines
        if (line.get_label() or "") in {"Left foot", "Right foot"}
    ]
    assert len(foot_lines) == 2
    for line in foot_lines:
        y = np.asarray(line.get_ydata(), dtype=float)
        finite = y[np.isfinite(y)]
        assert finite.size
        assert np.all(np.isclose(finite, np.round(finite)))
        assert not np.any(np.isclose(finite, 0.25))


def test_biomech_panels_draw_existing_series_independently() -> None:
    result = _sample_result()
    assert biomech_series_unavailable_reason(result, BIOMECH_SERIES_COM) is None
    assert biomech_series_unavailable_reason(result, BIOMECH_SERIES_STABILITY) is None
    assert biomech_series_unavailable_reason(result, BIOMECH_SERIES_PHASE) is None
    for series in (BIOMECH_SERIES_COM, BIOMECH_SERIES_STABILITY, BIOMECH_SERIES_PHASE):
        fig = Figure(figsize=(8, 3))
        draw_biomechanics_dashboard(fig, result, series=series)
        ax = fig.get_axes()[0]
        has_artists = bool(ax.lines) or bool(ax.collections) or bool(ax.patches)
        assert has_artists, f"{series} should render existing session data"


def test_phase_chart_omits_withheld_gait_quality_and_unreportable_speed() -> None:
    result = _sample_result()
    fig = Figure(figsize=(8, 3))
    draw_biomechanics_dashboard(fig, result, series=BIOMECH_SERIES_PHASE)
    blob = " ".join(t.get_text() for t in list(fig.texts) + list(fig.axes[0].texts))
    assert "Gait quality" not in blob
    assert "64/100" not in blob
    assert "0.55 m/s" not in blob
