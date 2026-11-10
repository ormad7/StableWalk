"""Tests for synchronized dashboard playhead styling."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.playback_state import PlaybackState
import numpy as np

from stablewalk.ui.viewers.chart_playhead import (
    PlayheadState,
    advance_playhead_anim_phase,
    apply_playhead_to_axes,
    draw_chart_playhead,
    format_playhead_label,
    playhead_from_playback,
    playhead_pulse,
    sample_displayed_series,
    set_playhead_value_source,
    update_chart_playhead,
    update_playhead_on_axes,
)


def test_format_playhead_label() -> None:
    state = PlayheadState(time_s=1.25, frame_index=42)
    assert format_playhead_label(state) == "1.25s · F42"


def test_playhead_from_playback_uses_display_frame() -> None:
    playback = PlaybackState(
        current_frame_index=324,
        current_time_seconds=13.52,
        total_frames=400,
        fps=25.0,
        duration_seconds=16.0,
        playing=False,
    )
    cursor = playhead_from_playback(playback)
    assert cursor.frame_index == 325
    assert cursor.list_pos == 324
    assert cursor.time_s == pytest.approx(13.52)
    assert format_playhead_label(cursor) == "13.52s · F325"


def test_playhead_pulse_is_static_when_idle() -> None:
    lw0, a0, g0 = playhead_pulse(0.0, animating=False)
    lw1, a1, g1 = playhead_pulse(0.25, animating=False)
    assert lw0 == lw1 and a0 == a1
    assert g0 == 0.0 and g1 == 0.0
    assert a0 >= 0.9


def test_playhead_pulse_breathes_when_animating() -> None:
    lw0, _a0, g0 = playhead_pulse(0.0, animating=True)
    lw1, _a1, g1 = playhead_pulse(0.25, animating=True)
    assert lw0 != lw1 or g0 != g1
    assert g0 > 0.0 and g1 > 0.0


def test_advance_playhead_anim_phase_wraps() -> None:
    assert advance_playhead_anim_phase(0.95, step=0.14) == pytest.approx(0.09)


def test_draw_chart_playhead_adds_artists() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 2.0], [0.1, 0.2])
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 0.3)
    before = len(ax.lines) + len(ax.collections) + len(ax.texts)

    draw_chart_playhead(
        ax,
        PlayheadState(time_s=1.0, frame_index=10, animating=True, anim_phase=0.5),
    )

    after = len(ax.lines) + len(ax.collections) + len(ax.texts)
    assert after > before
    assert any("1.00s · F10" in t.get_text() for t in ax.texts)


def test_apply_playhead_to_axes_syncs_all_rows() -> None:
    fig = Figure(figsize=(5, 6))
    axes = fig.subplots(3, 1, sharex=True)
    for ax in axes:
        ax.plot([0.0, 2.0], [0.1, 0.2])
        ax.set_xlim(0.0, 2.0)
        ax.set_ylim(0.0, 0.3)

    state = PlayheadState(time_s=0.5, frame_index=3)
    apply_playhead_to_axes(axes, state)

    for ax in axes:
        assert len(ax.lines) >= 3
        assert any("0.50s · F3" in t.get_text() for t in ax.texts)


def test_update_chart_playhead_reuses_existing_artists() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.plot([0.0, 2.0], [0.1, 0.2])
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 0.3)
    draw_chart_playhead(ax, PlayheadState(time_s=0.5, frame_index=3))
    counts = (len(ax.lines), len(ax.collections), len(ax.texts))

    assert update_chart_playhead(
        ax,
        PlayheadState(time_s=1.5, frame_index=9),
    )
    assert (len(ax.lines), len(ax.collections), len(ax.texts)) == counts
    artists = ax._stablewalk_playhead_artists
    assert list(artists[2].get_xdata()) == [1.5, 1.5]
    assert "1.50s · F9" in artists[4].get_text()


def test_playhead_value_callout_flips_left_near_right_edge() -> None:
    from stablewalk.ui.viewers.chart_playhead import edge_aware_callout_placement

    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 180.0)
    fig.canvas.draw()

    text = "Right Knee: 135.75°"
    _lx, _ly, ha, _va = edge_aware_callout_placement(ax, 1.95, 135.75, text=text)
    assert ha == "right"

    _lx2, _ly2, ha2, _va2 = edge_aware_callout_placement(ax, 0.25, 135.75, text=text)
    assert ha2 == "left"

    _lx3, ly3, _ha3, va3 = edge_aware_callout_placement(ax, 1.0, 175.0, text=text)
    assert va3 == "top"
    assert ly3 <= 180.0

    draw_chart_playhead(
        ax,
        PlayheadState(time_s=1.95, frame_index=40),
        value_label=text,
        value_y=135.75,
    )
    value = ax._stablewalk_playhead_artists[5]
    assert value.get_text() == text
    assert value.get_ha() == "right"
    assert value.get_clip_on() is False
    cap = ax._stablewalk_playhead_artists[3]
    assert cap.get_clip_on() is False

    from stablewalk.ui.viewers.chart_playhead import _estimate_callout_size_px

    tw_px, _th_px = _estimate_callout_size_px(text, dpi=100.0)
    vx, _vy = value.get_position()
    disp = ax.transData.transform((float(vx), float(value.get_position()[1])))
    if value.get_ha() == "right":
        left, right = disp[0] - tw_px, disp[0]
    else:
        left, right = disp[0], disp[0] + tw_px
    assert left >= fig.bbox.x0 - 2.0
    assert right <= fig.bbox.x1 + 2.0


def test_draw_chart_playhead_tolerates_stale_artist_after_axes_clear() -> None:
    class StaleArtist:
        def remove(self) -> None:
            raise NotImplementedError("cannot remove artist")

    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 0.3)
    ax._stablewalk_playhead_artists = (StaleArtist(),)

    draw_chart_playhead(ax, PlayheadState(time_s=1.0, frame_index=10))

    assert len(ax._stablewalk_playhead_artists) == 6


def _marker_xy(ax):
    artists = getattr(ax, "_stablewalk_playhead_artists", None)
    assert artists is not None
    cap = artists[3]
    if cap is None or not cap.get_visible():
        return None
    off = np.asarray(cap.get_offsets(), dtype=float)
    return float(off[0, 0]), float(off[0, 1])


def test_sample_displayed_series_hides_nan_gaps_instead_of_interpolating() -> None:
    times = np.array([0.0, 0.04, 0.08, 0.12])
    values = np.array([1.0, np.nan, np.nan, 4.0])
    assert sample_displayed_series(times, values, 0.0) == pytest.approx(1.0)
    assert sample_displayed_series(times, values, 0.12) == pytest.approx(4.0)
    assert sample_displayed_series(times, values, 0.04) is None
    assert sample_displayed_series(times, values, 0.08) is None
    across_gap = float(np.interp(0.08, np.array([0.0, 0.12]), np.array([1.0, 4.0])))
    assert across_gap == pytest.approx(3.0)
    assert sample_displayed_series(times, values, 0.08) is None


def test_playhead_marker_uses_displayed_series_not_ylim_midband() -> None:
    """COM screenshot: white cap at ~0.04 was ylim mid-band, not the blue series."""
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    t = np.array([0.0, 0.5, 1.0])
    com_y = np.array([0.084, 0.095, 0.105])
    ax.plot(t, com_y)
    ax.set_ylim(-0.03, 0.12)
    set_playhead_value_source(ax, t, com_y, "COM Y", "approx. BH")
    draw_chart_playhead(ax, PlayheadState(time_s=0.5, frame_index=2))

    xy = _marker_xy(ax)
    assert xy is not None
    assert xy[0] == pytest.approx(0.5)
    assert xy[1] == pytest.approx(0.095)
    y0, y1 = ax.get_ylim()
    mid = 0.5 * (y0 + y1)
    assert mid == pytest.approx(0.045, abs=0.02)
    assert abs(xy[1] - 0.095) < abs(xy[1] - mid)


def test_playhead_marker_hidden_when_sample_missing() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    t = np.array([0.0, 0.5, 1.0])
    y = np.array([0.1, np.nan, 0.2])
    ax.plot(t, y)
    ax.set_ylim(0.0, 0.3)
    set_playhead_value_source(ax, t, y, "vGRF", "N")
    draw_chart_playhead(ax, PlayheadState(time_s=0.5, frame_index=2))
    assert _marker_xy(ax) is None
    artists = ax._stablewalk_playhead_artists
    assert artists[2].get_visible() is True


def test_playhead_marker_does_not_follow_callout_or_mouse_y() -> None:
    fig = Figure(figsize=(5, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_xlim(0.0, 2.0)
    ax.set_ylim(0.0, 180.0)
    fig.canvas.draw()
    draw_chart_playhead(
        ax,
        PlayheadState(time_s=1.0, frame_index=10),
        value_label="Right Knee: 175.00°",
        value_y=175.0,
    )
    xy = _marker_xy(ax)
    assert xy is not None
    assert xy[1] == pytest.approx(175.0)
    value = ax._stablewalk_playhead_artists[5]
    callout_y = float(value.get_position()[1])
    mouse_y = 0.04
    ax.axhline(mouse_y, linestyle=":", color="0.5")
    xy_after = _marker_xy(ax)
    assert xy_after == xy
    assert xy_after[1] != pytest.approx(mouse_y)
    if abs(callout_y - 175.0) > 1.0:
        assert xy[1] != pytest.approx(callout_y)


def test_update_playhead_samples_canonical_timestamp() -> None:
    fig = Figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    t = np.array([0.0, 0.4, 0.8])
    y = np.array([0.09, 0.11, 0.08])
    ax.plot(t, y)
    ax.set_ylim(-0.03, 0.15)
    set_playhead_value_source(ax, t, y, "COM Y", "BH")
    draw_chart_playhead(ax, PlayheadState(time_s=0.0, frame_index=1))
    paused = _marker_xy(ax)
    assert paused is not None
    assert paused[0] == pytest.approx(0.0)
    assert paused[1] == pytest.approx(0.09)

    assert update_chart_playhead(ax, PlayheadState(time_s=0.4, frame_index=2))
    stepped = _marker_xy(ax)
    assert stepped is not None
    assert stepped[0] == pytest.approx(0.4)
    assert stepped[1] == pytest.approx(0.11)

    assert update_playhead_on_axes([ax], PlayheadState(time_s=0.8, frame_index=3))
    seeked = _marker_xy(ax)
    assert seeked is not None
    assert seeked[0] == pytest.approx(0.8)
    assert seeked[1] == pytest.approx(0.08)
