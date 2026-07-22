"""Tests for synchronized dashboard playhead styling."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.viewers.chart_playhead import (
    PlayheadState,
    advance_playhead_anim_phase,
    apply_playhead_to_axes,
    draw_chart_playhead,
    format_playhead_label,
    playhead_pulse,
    update_chart_playhead,
)


def test_format_playhead_label() -> None:
    state = PlayheadState(time_s=1.25, frame_index=42)
    assert format_playhead_label(state) == "F42 · 1.25s"


def test_playhead_pulse_is_static_for_lab_cursor() -> None:
    lw0, a0, g0 = playhead_pulse(0.0)
    lw1, a1, g1 = playhead_pulse(0.25)
    assert lw0 == lw1 and a0 == a1
    assert g0 == 0.0 and g1 == 0.0
    assert a0 >= 0.9


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
    assert any("F10" in t.get_text() for t in ax.texts)


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
        assert any("F3" in t.get_text() for t in ax.texts)


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
    assert "F9" in artists[4].get_text()


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
