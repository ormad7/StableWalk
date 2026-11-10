"""Single-panel Contact / vGRF charts for the Biomechanics workspace tabs."""

from __future__ import annotations

from matplotlib.figure import Figure

from stablewalk.analysis.foot_contact_analysis import FootContactAnalysisResult
from stablewalk.ui.viewers.contact_gait_viewer import (
    CONTACT_PANEL_CONTACT,
    CONTACT_PANEL_VGRF,
    CONTACT_UNKNOWN_REASON,
    VGRF_GAP_LEGEND,
    VGRF_GAP_STATUS,
    contact_panel_note,
    draw_contact_gait_dashboard,
    relayout_contact_gait_figure,
)
from tests.test_biomechanical_analysis import _contact


def test_contact_panel_draws_single_large_axes() -> None:
    contact = _contact(24)
    fig = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(
        fig, contact, None, panel=CONTACT_PANEL_CONTACT
    )
    assert len(fig.axes) == 1
    assert "Time" in (fig.axes[0].get_xlabel() or "")


def test_contact_note_is_unknown_reason_not_grf_peaks() -> None:
    from stablewalk.analysis.foot_contact_analysis import FootContactFrame

    frames = []
    for i in range(12):
        known = i < 6
        frames.append(
            FootContactFrame(
                frame_index=i,
                time_s=i / 25.0,
                left_contact_probability=0.8 if known else 0.2,
                right_contact_probability=0.2,
                left_contact_binary=1 if known else 0,
                right_contact_binary=0,
                left_heel_strike=0,
                right_heel_strike=0,
                left_toe_off=0,
                right_toe_off=0,
                left_foot_substate="mid_stance" if known else "swing",
                right_foot_substate="swing",
                macro_phase="stance" if known else "uncertain",
                left_confidence=0.8,
                right_confidence=0.8,
                left_contact_known=known,
                right_contact_known=False,
            )
        )
    contact = FootContactAnalysisResult(per_frame=frames, fps=25.0)
    fig = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(fig, contact, None, panel=CONTACT_PANEL_CONTACT)
    note = contact_panel_note(CONTACT_PANEL_CONTACT, contact=contact, vgrf=None)
    assert note == CONTACT_UNKNOWN_REASON
    assert getattr(fig, "_sw_panel_note", "") == CONTACT_UNKNOWN_REASON
    figure_text = " ".join(t.get_text() or "" for t in fig.texts)
    axes_text = " ".join(t.get_text() or "" for t in fig.axes[0].texts)
    assert "L peak" not in figure_text
    assert "L peak" not in axes_text
    assert "BW" not in figure_text


def test_vgrf_panel_explains_missing_segments_without_filling_zero() -> None:
    import numpy as np

    from stablewalk.analysis.estimated_vgrf_analysis import (
        EstimatedVGRFMetrics,
        EstimatedVGRFResult,
    )

    t = np.linspace(0.0, 1.0, 12)
    left = np.full(12, 0.8)
    left[6:] = np.nan
    right = np.full(12, 1.1)
    vgrf = EstimatedVGRFResult(
        timestamps=t,
        left_vgrf_vertical=left * 700.0,
        right_vgrf_vertical=right * 700.0,
        total_vgrf_vertical=np.where(np.isfinite(left), left + right, np.nan) * 700.0,
        left_vgrf_bw=left,
        right_vgrf_bw=right,
        available=True,
        metrics=EstimatedVGRFMetrics(
            left_peak_force_bw=0.8,
            right_peak_force_bw=1.1,
        ),
        notes=["unknown/unreliable feet are withheld (NaN), not recorded as zero."],
    )
    fig = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(fig, None, vgrf, panel=CONTACT_PANEL_VGRF)
    ax = fig.axes[0]
    legend = ax.get_legend()
    labels = [t.get_text() for t in legend.get_texts()] if legend else []
    assert VGRF_GAP_LEGEND in labels
    note = contact_panel_note(CONTACT_PANEL_VGRF, contact=None, vgrf=vgrf)
    assert "L peak" in note
    assert VGRF_GAP_STATUS[:20] in note
    assert getattr(fig, "_sw_panel_note", "") == note
    left_y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
    assert np.any(np.isnan(left_y))
    assert not np.any(left_y[np.isfinite(left_y)] == 0.0)


def test_switching_to_contact_clears_grf_figure_note() -> None:
    import numpy as np

    from stablewalk.analysis.estimated_vgrf_analysis import (
        EstimatedVGRFMetrics,
        EstimatedVGRFResult,
    )

    t = np.linspace(0.0, 1.0, 8)
    vgrf = EstimatedVGRFResult(
        timestamps=t,
        left_vgrf_vertical=np.ones(8) * 700.0,
        right_vgrf_vertical=np.ones(8) * 700.0,
        total_vgrf_vertical=np.ones(8) * 1400.0,
        left_vgrf_bw=np.ones(8),
        right_vgrf_bw=np.ones(8),
        available=True,
        metrics=EstimatedVGRFMetrics(left_peak_force_bw=1.0, right_peak_force_bw=1.0),
    )
    fig = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(fig, None, vgrf, panel=CONTACT_PANEL_VGRF)
    assert "L peak" in str(getattr(fig, "_sw_panel_note", ""))
    contact = _contact(8)
    draw_contact_gait_dashboard(fig, contact, vgrf, panel=CONTACT_PANEL_CONTACT)
    note = str(getattr(fig, "_sw_panel_note", ""))
    assert "L peak" not in note
    assert CONTACT_UNKNOWN_REASON in note
    figure_text = " ".join(t.get_text() or "" for t in fig.texts)
    assert "L peak" not in figure_text


def test_vgrf_panel_draws_single_axes_without_data() -> None:
    fig = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(fig, None, None, panel=CONTACT_PANEL_VGRF)
    assert len(fig.axes) == 1


def test_contact_panel_relayout_reserves_space_after_narrow_resize() -> None:
    fig = Figure(figsize=(6.0, 4.0), dpi=100)
    ax = fig.add_subplot(111)

    relayout_contact_gait_figure(fig)

    position = ax.get_position()
    assert position.y1 <= 0.94
    assert position.y0 >= 0.18
