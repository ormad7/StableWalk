"""Tests for the application-wide comparison cursor."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

from stablewalk.ui.viewers.comparison_cursor import (
    ComparisonCursor,
    active_gait_event_types,
    draw_cycle_percent_playhead,
    estimate_gait_cycle_percent,
    format_video_cursor_badge,
    snapshot_comparison_cursor,
)


def test_comparison_cursor_label() -> None:
    cursor = ComparisonCursor(
        frame_float=10.0,
        frame_index=10,
        list_pos=10,
        time_s=0.42,
        active_gait_events=("left_heel_strike",),
        cycle_percent=25.0,
    )
    assert "F10" in cursor.label()
    assert "0.42s" in cursor.label()
    assert "Comparison cursor" in cursor.summary_line()
    assert "L HS" in cursor.summary_line()
    assert "25%" in cursor.summary_line()
    assert "F10" in format_video_cursor_badge(cursor)
    assert "L HS" in format_video_cursor_badge(cursor)


def test_estimate_cycle_percent_between_heel_strikes() -> None:
    hs = [
        SimpleNamespace(event_type="left_heel_strike", time_s=1.0, frame_index=30),
        SimpleNamespace(event_type="left_heel_strike", time_s=2.0, frame_index=60),
    ]
    gait = SimpleNamespace(events=hs)
    pct = estimate_gait_cycle_percent(gait, time_s=1.5, frame_index=45)
    assert pct == 50.0


def test_active_gait_events_near_cursor() -> None:
    events = [
        SimpleNamespace(event_type="right_toe_off", time_s=1.00, frame_index=20),
        SimpleNamespace(event_type="left_heel_strike", time_s=1.50, frame_index=40),
    ]
    gait = SimpleNamespace(events=events)
    active = active_gait_event_types(gait, frame_index=40, time_s=1.50)
    assert "left_heel_strike" in active


def test_snapshot_from_gui() -> None:
    frame = SimpleNamespace(
        frame_index=7,
        timestamp_s=0.7,
        gait_events=["left_heel_strike"],
    )
    gui = SimpleNamespace(
        _playback_pos=3.0,
        current_pos=3,
        playing=False,
        _playhead_anim_phase=0.1,
        pose_indices=[1, 3, 5, 7],
        sequence=SimpleNamespace(frames=[frame]),
        skeleton_player=SimpleNamespace(
            state=SimpleNamespace(frame_float=3.0, playing=False),
            current_index=3,
        ),
        _gait_cycle=SimpleNamespace(
            events=[
                SimpleNamespace(
                    event_type="left_heel_strike", time_s=0.7, frame_index=7
                ),
                SimpleNamespace(
                    event_type="left_heel_strike", time_s=1.7, frame_index=20
                ),
            ]
        ),
    )
    cursor = snapshot_comparison_cursor(gui, advance_anim=False)
    assert cursor.frame_index == 7
    assert cursor.list_pos == 3
    assert cursor.time_s == 0.7
    assert "left_heel_strike" in cursor.active_gait_events
    ph = cursor.to_playhead_state()
    assert ph.frame_index == 7
    assert ph.time_s == 0.7


def test_draw_cycle_percent_playhead() -> None:
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot([0, 100], [0, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    before = len(ax.lines)
    draw_cycle_percent_playhead(
        ax,
        ComparisonCursor(
            frame_float=1,
            frame_index=1,
            list_pos=0,
            time_s=0.1,
            cycle_percent=40.0,
            playing=True,
            anim_phase=0.2,
        ),
    )
    assert len(ax.lines) > before
