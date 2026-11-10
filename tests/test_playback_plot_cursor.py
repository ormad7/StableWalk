"""Playback-linked plot cursors vs independent hover inspect.

The F324 / 13.48 s plot readout vs Live Data F325 / 13.52 s is not a paused
video/skeleton/live desync. Kinematics already used the 1-based display frame
and committed time. The plot stamp can still show a neighbor when:

* the mouse hover/crosshair inspects a nearby sample, or
* four-gait / spectral playheads labeled the 0-based list index as F{n}, or
* charts update at PLAY_CHART_HZ during play (one-tick lag is expected).

Playback-linked cursors must share one canonical frame/time. Hover is Inspect.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.io.pose_loader import detected_frame_indices
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.ui.playback_state import (
    PlaybackState,
    commit_playback_index,
    store_committed_playback,
)
from stablewalk.ui.skeleton_player import SkeletonPlayer
from stablewalk.ui.tk.playback_render_policy import PLAY_CHART_HZ
from stablewalk.ui.viewers.chart_hover import (
    ChartHoverPoint,
    format_chart_tooltip,
    format_inspect_crosshair_chip,
)
from stablewalk.ui.viewers.chart_playhead import (
    format_playhead_label,
    playhead_from_gui,
    playhead_from_playback,
)


def _kp() -> Keypoint:
    return Keypoint(name="nose", x=0.1, y=0.2, z=0.0, visibility=0.9)


def _make_sequence(*, n_frames: int = 400, fps: float = 25.0) -> PoseSequence:
    frames = [
        PoseFrame(
            frame_index=i,
            image_path=f"frame_{i:04d}.jpg",
            timestamp_s=i / fps,
            detected=True,
            keypoints=[_kp()],
        )
        for i in range(n_frames)
    ]
    return PoseSequence(source_video="plot_cursor.mp4", fps=fps, frames=frames)


def _gui_at_list_index(index: int, *, n_frames: int = 400, fps: float = 25.0):
    sequence = _make_sequence(n_frames=n_frames, fps=fps)
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=False)
    gui = SimpleNamespace(
        skeleton_player=player,
        pose_indices=detected_frame_indices(sequence),
        sequence=sequence,
        playing=False,
        current_pos=0,
        _playback_pos=0.0,
    )
    state = commit_playback_index(gui, index)
    assert state is not None
    return gui, state


def test_screenshot_offset_is_inspect_not_paused_playhead() -> None:
    """Live F325 / 13.52 s vs plot F324 / 13.48 s: hover inspect, not live desync."""
    playback = PlaybackState(
        current_frame_index=324,
        current_time_seconds=13.52,
        total_frames=400,
        fps=25.0,
        duration_seconds=16.0,
    )
    playhead = playhead_from_playback(playback)
    assert playback.display_frame_number == 325
    assert playback.format_frame_label().startswith("325/")
    assert "13.52 s" in playback.format_time_label()
    assert playhead.frame_index == 325
    assert playhead.list_pos == 324
    assert playhead.time_s == pytest.approx(13.52)
    assert format_playhead_label(playhead) == "13.52s · F325"

    inspect = ChartHoverPoint(
        ax=None,  # type: ignore[arg-type]
        x=13.48,
        y=12.0,
        frame_index=323,
        list_pos=323,
        timestamp_s=13.48,
        value=12.0,
        joint_name="LEFT KNEE",
        metric_name="Knee flexion",
        unit="deg",
    )
    tooltip = format_chart_tooltip(inspect)
    assert tooltip.splitlines()[0] == "Inspect"
    assert "Frame: 324" in tooltip
    assert "Time: 13.480 s" in tooltip
    assert format_inspect_crosshair_chip(13.48) == "Inspect · 13.48s"
    assert format_playhead_label(playhead) != "13.48s · F324"

    gui, state = _gui_at_list_index(324)
    linked = playhead_from_gui(gui)
    assert linked is not None
    assert linked.frame_index == state.display_frame_number
    assert linked.time_s == pytest.approx(state.current_time_seconds)


def test_paused_step_and_seek_share_canonical_cursor() -> None:
    gui, state = _gui_at_list_index(10)
    assert state.current_frame_index == 10
    playhead = playhead_from_gui(gui)
    assert playhead is not None
    assert playhead.frame_index == state.display_frame_number == 11
    assert playhead.time_s == pytest.approx(state.current_time_seconds)
    assert playhead.time_s == pytest.approx(10 / 25.0)

    stepped = commit_playback_index(gui, 11)
    assert stepped is not None
    after_step = playhead_from_gui(gui)
    assert after_step is not None
    assert after_step.frame_index == 12
    assert after_step.list_pos == 11
    assert after_step.time_s == pytest.approx(11 / 25.0)
    assert stepped.format_frame_label().startswith("12/")
    assert after_step.frame_index == stepped.display_frame_number
    assert after_step.time_s == pytest.approx(stepped.current_time_seconds)


@pytest.mark.parametrize("tab", ["motion", "four_gait", "spectral", "biomech", "overview"])
def test_tab_switch_rereads_committed_playback(tab: str) -> None:
    gui, state = _gui_at_list_index(59)
    gui._active_tab = tab
    store_committed_playback(gui, state)
    cursor = playhead_from_gui(gui)
    assert cursor is not None
    assert cursor.frame_index == 60
    assert cursor.time_s == pytest.approx(59 / 25.0)
    assert cursor.list_pos == state.current_frame_index


def test_four_gait_style_playhead_no_longer_uses_zero_based_f() -> None:
    from stablewalk.ui.tk.dashboard_four_gait import _playhead_state
    from stablewalk.ui.tk.dashboard_spectral import _playhead_state as spectral_playhead
    from stablewalk.ui.tk.overview_signal_spectral import _playhead as overview_playhead

    gui, state = _gui_at_list_index(324)
    for builder in (_playhead_state, spectral_playhead, overview_playhead):
        cursor = builder(gui)
        assert cursor is not None
        assert cursor.frame_index == state.display_frame_number == 325
        assert cursor.time_s == pytest.approx(state.current_time_seconds)
        assert format_playhead_label(cursor) == format_playhead_label(
            playhead_from_playback(state)
        )


def test_play_chart_hz_is_unchanged() -> None:
    """Do not 'fix' in-play screenshots by raising chart update rate."""
    assert PLAY_CHART_HZ == 30


def test_playing_playhead_keeps_canonical_frame_when_time_overrides() -> None:
    playback = PlaybackState(
        current_frame_index=324,
        current_time_seconds=13.52,
        total_frames=400,
        fps=25.0,
        playing=True,
        duration_seconds=16.0,
    )
    cursor = playhead_from_playback(playback, time_s=13.52, animating=True)
    assert cursor.frame_index == 325
    assert cursor.animating is True
    assert format_playhead_label(cursor) == "13.52s · F325"
