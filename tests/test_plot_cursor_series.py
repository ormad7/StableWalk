"""Playback cursor Y must match the displayed series at the canonical time.

Hover Inspect is a separate dotted readout and must not move the data marker.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from matplotlib.figure import Figure

from stablewalk.analysis.biomechanical.stability_margin import balance_display_contract
from stablewalk.analysis.estimated_vgrf_analysis import (
    EstimatedVGRFMetrics,
    EstimatedVGRFResult,
)
from stablewalk.ui.viewers.biomechanics_charts import (
    BIOMECH_SERIES_COM,
    BIOMECH_SERIES_CONTACT,
    BIOMECH_SERIES_STABILITY,
    draw_biomechanics_dashboard,
)
from stablewalk.ui.viewers.chart_hover import format_inspect_crosshair_chip
from stablewalk.ui.viewers.chart_playhead import (
    PLAYHEAD_VALUE_SOURCE_ATTR,
    PlayheadState,
    format_playhead_label,
    sample_displayed_series,
    update_playhead_on_axes,
)
from stablewalk.ui.viewers.contact_gait_viewer import (
    CONTACT_PANEL_CONTACT,
    CONTACT_PANEL_VGRF,
    draw_contact_gait_dashboard,
)
from stablewalk.ui.viewers.joint_motion_analysis_chart import (
    JointMotionBundle,
    JointMotionSeries,
    draw_motion_metric_chart,
)
from tests.test_biomechanical_analysis import _contact
from tests.test_biomechanics_charts import _sample_result


def _marker_xy(ax):
    artists = getattr(ax, "_stablewalk_playhead_artists", None)
    assert artists is not None
    cap = artists[3]
    if cap is None or not cap.get_visible():
        return None
    off = np.asarray(cap.get_offsets(), dtype=float)
    return float(off[0, 0]), float(off[0, 1])


def _assert_marker_on_series(ax, time_s: float, expected_y: float) -> None:
    xy = _marker_xy(ax)
    assert xy is not None, f"marker hidden at t={time_s}, expected y={expected_y}"
    assert xy[0] == pytest.approx(time_s, abs=1e-6)
    assert xy[1] == pytest.approx(float(expected_y), abs=1e-6)
    y0, y1 = ax.get_ylim()
    mid = 0.5 * (y0 + y1)
    if abs(float(expected_y) - mid) > 1e-4:
        assert abs(xy[1] - float(expected_y)) < abs(xy[1] - mid)


def _pause_step_seek(ax, times, values, frames) -> None:
    """Paused draw, then step, then seek — all on the same displayed series."""
    src = getattr(ax, PLAYHEAD_VALUE_SOURCE_ATTR, None)
    assert src is not None
    src_t = np.asarray(src[0], dtype=float)
    src_v = np.asarray(src[1], dtype=float)
    np.testing.assert_allclose(src_t, np.asarray(times, dtype=float), equal_nan=True)
    np.testing.assert_allclose(src_v, np.asarray(values, dtype=float), equal_nan=True)

    pause_i, step_i, seek_i = 0, 1, len(times) - 1
    for idx in (pause_i, step_i, seek_i):
        expected = sample_displayed_series(times, values, float(times[idx]))
        state = PlayheadState(
            time_s=float(times[idx]),
            frame_index=int(frames[idx]),
        )
        update_playhead_on_axes([ax], state)
        if expected is None:
            assert _marker_xy(ax) is None
        else:
            _assert_marker_on_series(ax, float(times[idx]), expected)
            assert format_playhead_label(state).startswith(
                f"{float(times[idx]):.2f}s · F{int(frames[idx])}"
            )


def test_com_screenshot_cursor_matches_displayed_series() -> None:
    result = _sample_result()
    t = np.asarray(result.center_of_mass.timestamps, dtype=float)
    com_y = np.asarray(result.center_of_mass.positions[:, 1], dtype=float)
    idx = len(t) // 2
    fig = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(
        fig,
        result,
        playhead_time_s=float(t[idx]),
        playhead_frame=idx + 1,
        series=BIOMECH_SERIES_COM,
    )
    ax = fig.axes[0]
    _assert_marker_on_series(ax, float(t[idx]), float(com_y[idx]))
    y0, y1 = ax.get_ylim()
    mid = 0.5 * (y0 + y1)
    # When the illustrative ±0.03 BH band stretches ylim, the old cap sat at
    # mid-band (~0.04) instead of the blue COM sample (~0.09).
    if abs(float(com_y[idx]) - mid) > 0.02:
        xy = _marker_xy(ax)
        assert xy is not None
        assert abs(xy[1] - mid) > 0.02
    inspect_chip = format_inspect_crosshair_chip(float(t[idx]))
    assert inspect_chip.startswith("Inspect")
    assert "Inspect" not in format_playhead_label(
        PlayheadState(time_s=float(t[idx]), frame_index=idx + 1)
    )
    mouse_y = mid
    ax.axhline(mouse_y, linestyle=":", color="0.7")
    _assert_marker_on_series(ax, float(t[idx]), float(com_y[idx]))
    _pause_step_seek(
        ax,
        t,
        com_y,
        np.arange(1, len(t) + 1),
    )


def test_balance_contact_grf_kinematics_pause_step_seek() -> None:
    result = _sample_result()

    # Balance
    contract = balance_display_contract(result.stability_margin)
    assert contract.has_series
    ts = np.asarray(contract.times_s, dtype=float)
    margins = np.asarray(
        [float(v) if v is not None else np.nan for v in contract.values_m],
        dtype=float,
    )
    fig_b = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(
        fig_b,
        result,
        playhead_time_s=float(ts[0]),
        playhead_frame=1,
        series=BIOMECH_SERIES_STABILITY,
    )
    _pause_step_seek(fig_b.axes[0], ts, margins, np.arange(1, len(ts) + 1))

    # Biomech Contact (|L−R|)
    lp = np.asarray(result.contact.left_contact_probability, dtype=float)
    rp = np.asarray(result.contact.right_contact_probability, dtype=float)
    asym = np.abs(lp - rp)
    tc = np.asarray(result.contact.timestamps, dtype=float)
    fig_c = Figure(figsize=(11, 4.8))
    draw_biomechanics_dashboard(
        fig_c,
        result,
        playhead_time_s=float(tc[0]),
        playhead_frame=1,
        series=BIOMECH_SERIES_CONTACT,
    )
    _pause_step_seek(fig_c.axes[0], tc, asym, np.arange(1, len(tc) + 1))

    # Contact timeline (L/R probabilities; unknown samples are NaN)
    contact = _contact(24)
    left_p = np.asarray(contact.left_contact_probability, dtype=float)
    right_p = np.asarray(contact.right_contact_probability, dtype=float)
    marker = np.fmax(left_p, right_p)
    fig_ct = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(
        fig_ct,
        contact,
        None,
        panel=CONTACT_PANEL_CONTACT,
        playhead_time_s=float(contact.timestamps[0]),
        playhead_frame=1,
    )
    _pause_step_seek(
        fig_ct.axes[0],
        np.asarray(contact.timestamps, dtype=float),
        marker,
        np.arange(1, len(contact.timestamps) + 1),
    )

    # GRF — marker on the displayed total (BW) series; NaNs stay hidden
    t = np.linspace(0.0, 1.0, 12)
    left = np.full(12, 0.8)
    left[6:] = np.nan
    right = np.full(12, 1.1)
    total_n = np.where(np.isfinite(left), left + right, np.nan) * 700.0
    vgrf = EstimatedVGRFResult(
        timestamps=t,
        left_vgrf_vertical=left * 700.0,
        right_vgrf_vertical=right * 700.0,
        total_vgrf_vertical=total_n,
        left_vgrf_bw=left,
        right_vgrf_bw=right,
        available=True,
        body_weight_n=700.0,
        metrics=EstimatedVGRFMetrics(
            left_peak_force_bw=0.8,
            right_peak_force_bw=1.1,
        ),
    )
    displayed = total_n / 700.0
    fig_g = Figure(figsize=(11, 5.2))
    draw_contact_gait_dashboard(
        fig_g,
        None,
        vgrf,
        panel=CONTACT_PANEL_VGRF,
        playhead_time_s=float(t[0]),
        playhead_frame=1,
    )
    _pause_step_seek(fig_g.axes[0], t, displayed, np.arange(1, len(t) + 1))
    update_playhead_on_axes(
        fig_g.axes, PlayheadState(time_s=float(t[8]), frame_index=9)
    )
    assert _marker_xy(fig_g.axes[0]) is None

    # Kinematics
    kt = np.linspace(0.0, 1.0, 25)
    ky = np.sin(kt * 6.0) * 20.0 + 30.0
    series = JointMotionSeries(
        item_id="right_knee",
        label="Right Knee",
        color="#5b9fd4",
        times_s=kt,
        frames=np.arange(1, 26),
        angle_deg=ky,
        omega_deg_s=np.gradient(ky, kt),
        alpha_deg_s2=np.gradient(np.gradient(ky, kt), kt),
        speed_m_s=np.abs(np.gradient(ky, kt)) * 0.01,
        accel_m_s2=np.zeros_like(kt),
        jerk_m_s3=np.zeros_like(kt),
        angular_jerk_deg_s3=np.zeros_like(kt),
        x_m=np.zeros_like(kt),
        y_m=np.zeros_like(kt),
        z_m=np.zeros_like(kt),
    )
    bundle = JointMotionBundle(series=(series,), times_s=kt)
    fig_k = Figure(figsize=(8, 4))
    draw_motion_metric_chart(
        fig_k,
        bundle,
        "angle",
        playhead=PlayheadState(time_s=float(kt[0]), frame_index=1),
        show_peaks=False,
    )
    _pause_step_seek(fig_k.axes[0], kt, ky, np.arange(1, len(kt) + 1))
