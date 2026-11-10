"""Authoritative playback position for StableWalk GUI.

All visible playheads (video, skeleton, path tip, graphs, status, frame/time
labels, gait phase) must read from :class:`PlaybackState` — never from
per-tab caches or trajectory sample counts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PlaybackState:
    """Single source of truth for playback position.

    Internal indexing is always zero-based (``current_frame_index``).
    Display helpers convert to one-based labels.
    """

    current_frame_index: int
    """0-based index into the playback recording / pose_indices list."""

    current_time_seconds: float
    total_frames: int
    fps: float

    frame_float: float = 0.0
    """Smooth scrub position (same units as ``current_frame_index``)."""

    source_frame_index: int | None = None
    """Original video / PoseFrame.frame_index for the active sample, if known."""

    playing: bool = False
    duration_seconds: float = 0.0
    analyzed_source_frames: int | None = None
    """Frame count of the loaded/analyzed clip (sequence.frames), not detected-only."""

    @property
    def display_frame_number(self) -> int:
        """One-based frame number for UI labels."""
        if self.total_frames <= 0:
            return 0
        return int(self.current_frame_index) + 1

    def format_frame_label(self) -> str:
        if self.total_frames <= 0:
            return "—/—"
        return f"{self.display_frame_number}/{self.total_frames}"

    def format_frame_label_prefixed(self) -> str:
        if self.total_frames <= 0:
            return "F—/—"
        return f"F{self.display_frame_number}/{self.total_frames}"

    def format_time_label(self, *, include_duration: bool = True) -> str:
        t = float(self.current_time_seconds)
        if not include_duration:
            return f"{t:.2f} s"
        dur = float(self.duration_seconds)
        return f"{t:.2f} s / {dur:.2f} s"

    def format_source_frame_label(self) -> str:
        """Original analyzed-clip frame id, not the compressed detected-frame index."""
        src = self.source_frame_index
        total = self.analyzed_source_frames
        if src is None or total is None or int(total) <= 0:
            return "—"
        return f"{int(src) + 1}/{int(total)} source"


def _clamp_index(index: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(int(index), total - 1))


def sequence_index_for_time(sequence: Any, time_s: float) -> int | None:
    """Index into sequence.frames nearest to a source timestamp (all analyzed frames)."""
    frames = list(getattr(sequence, "frames", None) or [])
    if not frames:
        return None
    target = float(time_s)
    best_i = 0
    best_d = abs(float(getattr(frames[0], "timestamp_s", 0.0) or 0.0) - target)
    for i, frame in enumerate(frames):
        ts = float(getattr(frame, "timestamp_s", 0.0) or 0.0)
        d = abs(ts - target)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def source_frame_for_list_pos(gui: Any, list_pos: int) -> int | None:
    """Map a playback list index to the original PoseFrame / video frame id."""
    pose_indices = getattr(gui, "pose_indices", None) or []
    if pose_indices:
        i = _clamp_index(list_pos, len(pose_indices))
        try:
            return int(pose_indices[i])
        except (TypeError, ValueError, IndexError):
            return None
    player = getattr(gui, "skeleton_player", None)
    if player is None or getattr(player, "frame_count", 0) <= 0:
        return None
    snap = None
    try:
        snap = player.recording.snapshot_at(_clamp_index(list_pos, player.frame_count))
    except Exception:
        snap = None
    if snap is not None:
        try:
            return int(snap.frame_index)
        except (TypeError, ValueError):
            return None
    return None


def read_playback_state(gui: Any) -> PlaybackState | None:
    """Read the authoritative playback state from the skeleton player.

    Falls back to mirrored GUI fields only when the player is absent.
    Never uses trajectory path length as ``total_frames``.
    """
    player = getattr(gui, "skeleton_player", None)
    if player is not None and getattr(player, "frame_count", 0) > 0:
        total = int(player.frame_count)
        try:
            frame_float = float(player.state.frame_float)
        except Exception:
            frame_float = float(getattr(player.state, "frame_index", 0) or 0)
        # Match SkeletonPlayer.state.frame_index (same truncate rule as status bar).
        try:
            frame_i = int(player.state.frame_index)
        except Exception:
            frame_i = int(frame_float)
        frame_i = _clamp_index(frame_i, total)
        try:
            time_s = float(player.time_at_current())
        except Exception:
            fps = float(getattr(player, "fps", 0.0) or 0.0)
            time_s = frame_i / fps if fps > 1e-6 else 0.0
        sequence = getattr(gui, "sequence", None)
        analyzed = len(getattr(sequence, "frames", None) or []) or None
        try:
            fps = float(player.fps)
        except Exception:
            fps = 30.0
        try:
            duration = float(player.duration_s)
        except Exception:
            duration = (total - 1) / fps if total > 1 and fps > 1e-6 else 0.0
        source = source_frame_for_list_pos(gui, frame_i)
        snap = None
        try:
            snap = player.current_snapshot()
        except Exception:
            snap = None
        if source is None and snap is not None:
            try:
                source = int(snap.frame_index)
            except (TypeError, ValueError):
                source = None
        # During a tracking gap the clock keeps moving while the snapshot holds.
        # Video must follow the source timestamp, not the compressed list index.
        clock = getattr(player, "clock_s", None)
        snap_t = float(getattr(snap, "time_s", time_s) or time_s) if snap is not None else time_s
        in_gap = (
            clock is not None
            and sequence is not None
            and abs(float(clock) - float(snap_t)) > (0.5 / max(fps, 1e-6))
        )
        if in_gap:
            seq_i = sequence_index_for_time(sequence, float(clock))
            if seq_i is not None:
                try:
                    source = int(sequence.frames[seq_i].frame_index)
                except (TypeError, ValueError, IndexError, AttributeError):
                    source = seq_i
        playing = bool(getattr(gui, "playing", False))
        try:
            playing = bool(player.state.playing) or playing
        except Exception:
            pass
        return PlaybackState(
            current_frame_index=frame_i,
            current_time_seconds=time_s,
            total_frames=total,
            fps=max(fps, 1e-6),
            frame_float=max(0.0, min(frame_float, float(max(total - 1, 0)))),
            source_frame_index=source,
            playing=playing,
            duration_seconds=duration,
            analyzed_source_frames=analyzed,
        )

    pose_indices = getattr(gui, "pose_indices", None) or []
    sequence = getattr(gui, "sequence", None)
    if not pose_indices or sequence is None:
        return None
    total = len(pose_indices)
    try:
        frame_float = float(getattr(gui, "_playback_pos", getattr(gui, "current_pos", 0)) or 0)
    except (TypeError, ValueError):
        frame_float = 0.0
    frame_i = _clamp_index(int(frame_float), total)
    fps = float(getattr(sequence, "fps", 0.0) or 30.0)
    src = int(pose_indices[frame_i])
    frame = sequence.frames[src] if 0 <= src < len(sequence.frames) else None
    if frame is None:
        frame = next(
            (f for f in sequence.frames if int(getattr(f, "frame_index", -1)) == src),
            None,
        )
    time_s = float(getattr(frame, "timestamp_s", 0.0) or 0.0) if frame else frame_i / max(fps, 1e-6)
    frames = list(getattr(sequence, "frames", None) or [])
    if frames:
        duration = float(getattr(frames[-1], "timestamp_s", 0.0) or 0.0)
    else:
        duration = (total - 1) / max(fps, 1e-6) if total > 1 else 0.0
    return PlaybackState(
        current_frame_index=frame_i,
        current_time_seconds=time_s,
        total_frames=total,
        fps=max(fps, 1e-6),
        frame_float=max(0.0, min(frame_float, float(max(total - 1, 0)))),
        source_frame_index=src,
        playing=bool(getattr(gui, "playing", False)),
        duration_seconds=duration,
        analyzed_source_frames=len(frames) or None,
    )


def mirror_playback_mirrors(gui: Any, state: PlaybackState) -> None:
    """Keep legacy GUI fields in lockstep with the authoritative state."""
    try:
        gui.current_pos = int(state.current_frame_index)
    except Exception:
        pass
    try:
        gui._playback_pos = float(state.frame_float)
    except Exception:
        pass


def commit_playback_index(gui: Any, index: int | float) -> PlaybackState | None:
    """Seek the skeleton player (if present) and return the new state."""
    player = getattr(gui, "skeleton_player", None)
    if player is not None and getattr(player, "frame_count", 0) > 0:
        player.go_to(index)
    else:
        pose_indices = getattr(gui, "pose_indices", None) or []
        if not pose_indices:
            return None
        if isinstance(index, float):
            gui._playback_pos = max(0.0, min(float(index), len(pose_indices) - 1))
            gui.current_pos = int(gui._playback_pos)
        else:
            gui.current_pos = max(0, min(int(index), len(pose_indices) - 1))
            gui._playback_pos = float(gui.current_pos)
    state = read_playback_state(gui)
    if state is not None:
        mirror_playback_mirrors(gui, state)
        store_committed_playback(gui, state)
    return state


def quantize_playback_to_discrete_frame(gui: Any) -> PlaybackState | None:
    """Snap the playhead to an integer list index so pause/step labels match time."""
    player = getattr(gui, "skeleton_player", None)
    if player is not None and getattr(player, "frame_count", 0) > 0:
        try:
            index = int(player.state.frame_index)
        except Exception:
            index = 0
        player.go_to(index)
    else:
        pose_indices = getattr(gui, "pose_indices", None) or []
        if pose_indices:
            current = int(getattr(gui, "current_pos", 0) or 0)
            gui.current_pos = _clamp_index(current, len(pose_indices))
            gui._playback_pos = float(gui.current_pos)
    return read_playback_state(gui)


def begin_playback_commit(gui: Any) -> int:
    """Start a new playback publish generation. Later async work must match this."""
    gen = int(getattr(gui, "_playback_commit_gen", 0) or 0) + 1
    try:
        gui._playback_commit_gen = gen
    except Exception:
        pass
    return gen


def playback_commit_is_current(gui: Any, generation: int | None) -> bool:
    if generation is None:
        return True
    try:
        return int(getattr(gui, "_playback_commit_gen", 0) or 0) == int(generation)
    except (TypeError, ValueError):
        return False


def store_committed_playback(gui: Any, state: PlaybackState, generation: int | None = None) -> None:
    try:
        gui._committed_playback = state
        if generation is not None:
            gui._committed_playback_gen = int(generation)
    except Exception:
        pass


def committed_playback_state(gui: Any) -> PlaybackState | None:
    """Return the last committed state, or read the player if none is stored."""
    stored = getattr(gui, "_committed_playback", None)
    if isinstance(stored, PlaybackState):
        return stored
    return read_playback_state(gui)


@dataclass(frozen=True, slots=True)
class PlaybackSyncProbe:
    """Frame indices reported by each visible surface (0-based list indices)."""

    displayed_video_frame: int | None
    skeleton_frame: int | None
    path_marker_frame: int | None
    graph_cursor_frame: int | None
    live_panel_frame: int | None = None
    status_bar_frame: int | None = None
    slider_frame: int | None = None

    def consistent(self) -> bool:
        present = [
            v
            for v in (
                self.displayed_video_frame,
                self.skeleton_frame,
                self.path_marker_frame,
                self.graph_cursor_frame,
                self.live_panel_frame,
                self.status_bar_frame,
                self.slider_frame,
            )
            if v is not None
        ]
        if len(present) < 2:
            return True
        return len(set(present)) == 1

    def as_debug_lines(self) -> list[str]:
        return [
            f"displayed_video_frame={self.displayed_video_frame}",
            f"skeleton_frame={self.skeleton_frame}",
            f"path_marker_frame={self.path_marker_frame}",
            f"graph_cursor_frame={self.graph_cursor_frame}",
            f"live_panel_frame={self.live_panel_frame}",
            f"status_bar_frame={self.status_bar_frame}",
            f"slider_frame={self.slider_frame}",
        ]


def collect_playback_sync_probe(gui: Any, state: PlaybackState | None = None) -> PlaybackSyncProbe:
    """Collect the frame each surface currently claims to show."""
    state = state or read_playback_state(gui)
    canonical = state.current_frame_index if state is not None else None

    video = getattr(gui, "_last_video_list_pos", None)
    if video is None:
        video = canonical

    skeleton = None
    player = getattr(gui, "skeleton_player", None)
    if player is not None and getattr(player, "frame_count", 0) > 0:
        try:
            skeleton = int(player.state.frame_index)
        except Exception:
            skeleton = canonical

    path = getattr(gui, "_path_marker_list_pos", None)
    if path is None:
        path = getattr(gui, "_overview_path_list_pos", None)
    if path is None:
        path = canonical

    graph = getattr(gui, "_graph_cursor_list_pos", None)
    if graph is None:
        graph = canonical
    else:
        visible = getattr(gui, "_group_visible", None)
        if callable(visible):
            try:
                if not visible("motion"):
                    graph = canonical
            except Exception:
                pass

    live = getattr(gui, "_live_panel_list_pos", None)
    status = getattr(gui, "_status_bar_list_pos", None)
    slider = None
    frame_var = getattr(gui, "frame_var", None)
    if frame_var is not None:
        try:
            slider = int(round(float(frame_var.get())))
        except Exception:
            slider = None

    return PlaybackSyncProbe(
        displayed_video_frame=None if video is None else int(video),
        skeleton_frame=None if skeleton is None else int(skeleton),
        path_marker_frame=None if path is None else int(path),
        graph_cursor_frame=None if graph is None else int(graph),
        live_panel_frame=None if live is None else int(live),
        status_bar_frame=None if status is None else int(status),
        slider_frame=slider,
    )


def assert_playback_surfaces_synced(gui: Any, state: PlaybackState | None = None) -> None:
    """Raise when video / skeleton / path / graph disagree on the active frame."""
    probe = collect_playback_sync_probe(gui, state)
    if probe.consistent():
        return
    lines = "\n".join(probe.as_debug_lines())
    raise AssertionError("Playback surfaces are out of sync:\n" + lines)


def debug_mode_enabled() -> bool:
    return os.environ.get("STABLEWALK_GUI_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def assert_playback_synced_if_debug(gui: Any, state: PlaybackState | None = None) -> None:
    if not debug_mode_enabled():
        return
    assert_playback_surfaces_synced(gui, state)


def sample_sync_frame_indices(total_frames: int) -> list[int]:
    """0-based list indices for automated sync tests: 1, 30, 60, 90, final.

    Requested display frames are converted to zero-based indices and clamped.
    """
    if total_frames <= 0:
        return []
    # Display frames 1, 30, 60, 90 → indices 0, 29, 59, 89; plus final.
    wanted_display = (1, 30, 60, 90, total_frames)
    out: list[int] = []
    seen: set[int] = set()
    for display in wanted_display:
        idx = _clamp_index(int(display) - 1, total_frames)
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out


__all__ = [
    "PlaybackState",
    "PlaybackSyncProbe",
    "assert_playback_surfaces_synced",
    "assert_playback_synced_if_debug",
    "begin_playback_commit",
    "collect_playback_sync_probe",
    "commit_playback_index",
    "committed_playback_state",
    "debug_mode_enabled",
    "mirror_playback_mirrors",
    "playback_commit_is_current",
    "quantize_playback_to_discrete_frame",
    "read_playback_state",
    "sample_sync_frame_indices",
    "sequence_index_for_time",
    "source_frame_for_list_pos",
    "store_committed_playback",
]
