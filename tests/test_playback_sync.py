"""Automated playback frame/time synchronization tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.io.pose_loader import detected_frame_indices, playback_frame_indices
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.ui.playback_state import (
    PlaybackState,
    assert_playback_surfaces_synced,
    collect_playback_sync_probe,
    commit_playback_index,
    mirror_playback_mirrors,
    read_playback_state,
    sample_sync_frame_indices,
)
from stablewalk.ui.skeleton_player import SkeletonPlayer


def _kp(name: str = "nose") -> Keypoint:
    return Keypoint(name=name, x=0.1, y=0.2, z=0.0, visibility=0.9)


def _make_sequence(*, n_frames: int = 100, drop_keypoints_at: int | None = 50) -> PoseSequence:
    frames: list[PoseFrame] = []
    for i in range(n_frames):
        detected = True
        kps = [_kp()]
        if drop_keypoints_at is not None and i == drop_keypoints_at:
            kps = []
        frames.append(
            PoseFrame(
                frame_index=i,
                image_path=f"frame_{i:04d}.jpg",
                timestamp_s=i / 25.0,
                detected=detected,
                keypoints=kps,
            )
        )
    return PoseSequence(source_video="sync_test.mp4", fps=25.0, frames=frames)


def test_sample_sync_frame_indices_includes_requested_display_frames() -> None:
    idxs = sample_sync_frame_indices(120)
    # display 1,30,60,90,120 → 0-based 0,29,59,89,119
    assert idxs == [0, 29, 59, 89, 119]


def test_sample_sync_frame_indices_clamps_short_recordings() -> None:
    idxs = sample_sync_frame_indices(40)
    assert idxs[0] == 0
    assert idxs[-1] == 39
    assert all(0 <= i < 40 for i in idxs)


def test_pose_indices_match_gait_motion_frame_count() -> None:
    sequence = _make_sequence(n_frames=20, drop_keypoints_at=7)
    pose_indices = playback_frame_indices(sequence)
    recording = pose_sequence_to_gait_motion(sequence)
    assert len(pose_indices) == 20
    assert recording.frame_count == 20
    assert all(
        recording.snapshot_at(i).frame_index == pose_indices[i]
        for i in range(recording.frame_count)
    )
    hole = recording.snapshot_at(7)
    assert hole is not None
    assert hole.frame_index == 7
    # Empty keypoints keep the source slot even if detection is flagged.


def test_playback_state_display_is_one_based() -> None:
    state = PlaybackState(
        current_frame_index=0,
        current_time_seconds=0.0,
        total_frames=120,
        fps=30.0,
        duration_seconds=4.0,
    )
    assert state.display_frame_number == 1
    assert state.format_frame_label() == "1/120"
    assert "0.00 s" in state.format_time_label()


def test_sync_probe_asserts_on_mismatch() -> None:
    gui = SimpleNamespace(
        skeleton_player=SimpleNamespace(
            frame_count=10,
            state=SimpleNamespace(frame_index=5, frame_float=5.0, playing=False),
            fps=30.0,
            duration_s=1.0,
            time_at_current=lambda: 5 / 30.0,
            recording=SimpleNamespace(snapshot_at=lambda _i: None),
        ),
        playing=False,
        pose_indices=list(range(10)),
        _last_video_list_pos=5,
        _path_marker_list_pos=5,
        _graph_cursor_list_pos=4,
    )
    probe = collect_playback_sync_probe(gui)
    assert not probe.consistent()
    with pytest.raises(AssertionError, match="out of sync"):
        assert_playback_surfaces_synced(gui)


@pytest.mark.parametrize("display_frame", [1, 30, 60, 90, "final"])
def test_authoritative_state_sync_at_sample_frames(display_frame: int | str) -> None:
    sequence = _make_sequence(n_frames=100, drop_keypoints_at=None)
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=False)
    pose_indices = playback_frame_indices(sequence)
    gui = SimpleNamespace(
        skeleton_player=player,
        pose_indices=pose_indices,
        sequence=sequence,
        playing=False,
        current_pos=0,
        _playback_pos=0.0,
        _last_video_list_pos=0,
        _path_marker_list_pos=0,
        _graph_cursor_list_pos=0,
        _overview_path_list_pos=0,
    )

    total = recording.frame_count
    if display_frame == "final":
        target = total - 1
    else:
        target = int(display_frame) - 1

    state = commit_playback_index(gui, target)
    assert state is not None
    assert state.current_frame_index == target
    assert state.total_frames == total
    assert state.display_frame_number == target + 1
    assert state.source_frame_index == pose_indices[target]
    # Trajectory sample count must not redefine total_frames.
    fake_traj_len = max(1, total // 24)
    assert state.total_frames != fake_traj_len or total <= 24

    gui._last_video_list_pos = state.current_frame_index
    gui._path_marker_list_pos = state.current_frame_index
    gui._graph_cursor_list_pos = state.current_frame_index
    mirror_playback_mirrors(gui, state)
    assert_playback_surfaces_synced(gui, state)

    reread = read_playback_state(gui)
    assert reread is not None
    assert reread.current_frame_index == state.current_frame_index
    assert reread.format_frame_label() == state.format_frame_label()
    assert abs(reread.current_time_seconds - state.current_time_seconds) < 1e-6


def test_playback_defaults_forbid_intentional_surface_skips() -> None:
    """Playing surfaces share one clock; static chrome is intentionally frozen."""
    from pathlib import Path

    from stablewalk.ui.tk.playback_render_policy import TARGET_PLAY_HZ

    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "app.py"
    ).read_text(encoding="utf-8")
    assert "self._3d_play_stride = 1" in src
    assert "self._ui_sync_interval_s = 0.0" in src
    assert "self._traj_redraw_interval_s = 0.0" in src
    # Joint Information / HUD chrome is frozen during play; inspector updates
    # via the tip dirty path instead of a zero joint-info interval.
    assert "self._joint_info_play_interval_s = 1e9" in src
    assert f"TARGET_PLAY_HZ" in src or str(TARGET_PLAY_HZ) in src
    # Extract _tick body: must always sync live surfaces together.
    start = src.index("def _tick(self)")
    end = src.index("def _flush_pending_3d", start)
    tick = src[start:end]
    assert "_sync_all_to_frame" in tick
    assert "_sync_playback_video_only" not in tick


def test_overview_lite_playback_keeps_phase_hud_live() -> None:
    import inspect

    from stablewalk.ui.tk.app import StableWalkGUI

    source = inspect.getsource(StableWalkGUI._sync_overview_playback_lite)
    assert "_update_overview_playback_hud(include_clinical=False)" in source


def test_status_bar_is_not_rate_limited_away_from_live_panel() -> None:
    """The reported 33@1.31s vs 31@1.22s bug was an 8 Hz status-bar throttle."""
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "app.py"
    ).read_text(encoding="utf-8")
    transport = src[src.index("def _sync_transport_labels") : src.index("def _configure_label_if_changed")]
    assert "max_hz=8.0" not in transport
    assert "update_dashboard_status_bar(self, state=resolved)" in transport
    sync = src[src.index("def _sync_all_to_frame") : src.index("def _ghost_snapshots_for")]
    assert "quantize_playback_to_discrete_frame" in sync
    assert "begin_playback_commit" in sync
    assert "store_committed_playback" in sync
    assert "lbl_frame_data" not in sync or "frame.frame_index + 1" not in src[
        src.index("self.lbl_frame_data.configure") : src.index("self._fill_position_tree")
    ]


def test_pause_quantizes_fractional_playhead() -> None:
    from stablewalk.ui.playback_state import quantize_playback_to_discrete_frame

    sequence = _make_sequence(n_frames=100, drop_keypoints_at=None)
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=True)
    player.go_to(32.7)
    assert player.state.frame_index == 32
    assert player.state.frame_float == pytest.approx(32.7)
    gui = SimpleNamespace(skeleton_player=player, pose_indices=list(range(100)), playing=False)
    state = quantize_playback_to_discrete_frame(gui)
    assert state is not None
    assert player.state.frame_float == pytest.approx(32.0)
    assert state.current_frame_index == 32
    assert state.display_frame_number == 33
    assert abs(state.current_time_seconds - 32 / 25.0) < 1e-6


def test_live_panel_uses_list_display_not_source_frame() -> None:
    """Dropped detections must not show source-video 33 while status shows 31."""
    sequence = _make_sequence(n_frames=40, drop_keypoints_at=None)
    # Pretend two earlier frames were dropped so list index 30 maps to source 32.
    pose_indices = list(range(2, 40))
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=False)
    gui = SimpleNamespace(
        skeleton_player=player,
        pose_indices=pose_indices,
        sequence=sequence,
        playing=False,
        current_pos=0,
        _playback_pos=0.0,
    )
    state = commit_playback_index(gui, 30)
    assert state is not None
    assert state.current_frame_index == 30
    assert state.display_frame_number == 31
    assert state.source_frame_index == pose_indices[30]
    assert state.format_frame_label().startswith("31/")
    assert state.source_frame_index + 1 == 33
    assert state.display_frame_number != state.source_frame_index + 1


def test_stale_playback_generation_is_rejected() -> None:
    from stablewalk.ui.playback_state import (
        begin_playback_commit,
        playback_commit_is_current,
    )

    gui = SimpleNamespace(_playback_commit_gen=0)
    first = begin_playback_commit(gui)
    second = begin_playback_commit(gui)
    assert first != second
    assert playback_commit_is_current(gui, second)
    assert not playback_commit_is_current(gui, first)


@pytest.mark.parametrize("action", ["seek", "step"])
def test_seek_and_step_commit_one_frame(action: str) -> None:
    sequence = _make_sequence(n_frames=80, drop_keypoints_at=None)
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=False)
    gui = SimpleNamespace(
        skeleton_player=player,
        pose_indices=list(range(80)),
        sequence=sequence,
        playing=False,
        current_pos=0,
        _playback_pos=0.0,
        _last_video_list_pos=0,
        _path_marker_list_pos=0,
        _graph_cursor_list_pos=0,
        _live_panel_list_pos=0,
        _status_bar_list_pos=0,
        frame_var=SimpleNamespace(get=lambda: 10, set=lambda _v: None),
    )
    if action == "seek":
        state = commit_playback_index(gui, 10)
    else:
        state = commit_playback_index(gui, 9)
        state = commit_playback_index(gui, state.current_frame_index + 1)
    assert state is not None
    gui._last_video_list_pos = state.current_frame_index
    gui._path_marker_list_pos = state.current_frame_index
    gui._graph_cursor_list_pos = state.current_frame_index
    gui._live_panel_list_pos = state.current_frame_index
    gui._status_bar_list_pos = state.current_frame_index
    assert_playback_surfaces_synced(gui, state)
    assert abs(state.current_time_seconds - state.current_frame_index / 25.0) < 1e-6


def test_player_holds_pose_across_timestamp_gap() -> None:
    sequence = _make_sequence(n_frames=20, drop_keypoints_at=None)
    sequence.frames[8].detected = False
    sequence.frames[8].keypoints = []
    sequence.frames[9].detected = False
    sequence.frames[9].keypoints = []
    sequence.frames[10].detected = False
    sequence.frames[10].keypoints = []
    recording = pose_sequence_to_gait_motion(sequence)
    assert recording.frame_count == 20
    player = SkeletonPlayer(recording, smooth=True)
    player.play()
    player.go_to_time(7 / 25.0)
    player.advance(speed=1.0, dt=1.0 / 25.0)
    assert player.clock_s is not None
    assert player.clock_s > 7 / 25.0
    # Missing detections keep their source slots; the clock occupies them.
    assert player.state.frame_index == 8
    later = player.go_to_time(11 / 25.0)
    assert later is not None
    assert later.frame_index == 11


def test_player_does_not_interpolate_across_gap() -> None:
    sequence = _make_sequence(n_frames=12, drop_keypoints_at=None)
    sequence.frames[5].detected = False
    sequence.frames[5].keypoints = []
    recording = pose_sequence_to_gait_motion(sequence)
    player = SkeletonPlayer(recording, smooth=True)
    a = player.snapshot_at(4.6)
    assert a is not None
    assert a.frame_index == 4
    missing = player.snapshot_at(5)
    assert missing is not None
    assert missing.frame_index == 5
    assert missing.metadata.get("detected") is False


def test_source_frame_label_is_separate_from_detected_index() -> None:
    state = PlaybackState(
        current_frame_index=30,
        current_time_seconds=1.2,
        total_frames=38,
        fps=25.0,
        source_frame_index=32,
        analyzed_source_frames=40,
    )
    assert state.format_frame_label() == "31/38"
    assert state.format_source_frame_label() == "33/40 source"
