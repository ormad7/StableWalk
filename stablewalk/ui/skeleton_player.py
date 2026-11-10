"""
Playback controller for ``GaitMotionRecording`` skeleton visualization.

Manages current frame index, play/pause/stop state, and snapshot interpolation
without coupling to Tk widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot
from stablewalk.ui.viewers.gait_skeleton_renderer import interpolate_snapshots


@dataclass
class SkeletonPlaybackState:
    """Current playback position and mode."""

    frame_index: int = 0
    frame_float: float = 0.0
    playing: bool = False
    stopped: bool = True


@dataclass
class SkeletonPlayer:
    """
    Drives interactive skeleton playback from a ``GaitMotionRecording``.

    Usage:
        player = SkeletonPlayer(recording)
        snap = player.current_snapshot()
        player.advance(speed=1.0, dt=1/30)
    """

    recording: GaitMotionRecording
    state: SkeletonPlaybackState = field(default_factory=SkeletonPlaybackState)
    smooth: bool = True
    clock_s: float | None = None
    """Source-video timestamp for this playhead. None until the first seek."""

    @property
    def frame_count(self) -> int:
        return max(self.recording.frame_count, 0)

    @property
    def fps(self) -> float:
        return max(self.recording.fps, 1e-6)

    @property
    def duration_s(self) -> float:
        return self.recording.duration_s

    def snapshot_at(self, index: int | float) -> SkeletonSnapshot | None:
        if self.frame_count == 0:
            return None
        if isinstance(index, float):
            i0 = int(max(0, min(index, self.frame_count - 1)))
            i1 = min(i0 + 1, self.frame_count - 1)
            alpha = index - i0
            a = self.recording.snapshot_at(i0)
            b = self.recording.snapshot_at(i1)
            if not a:
                return b
            if not b or alpha <= 0.0 or not self.smooth:
                return a
            a_ok = bool(a.metadata.get("detected", True)) and bool(a.joints)
            b_ok = bool(b.metadata.get("detected", True)) and bool(b.joints)
            if not a_ok or not b_ok:
                return a
            from stablewalk.analysis.joint_path_trace import segment_is_connected

            if not segment_is_connected(float(a.time_s), float(b.time_s), fps=self.fps):
                return a
            return interpolate_snapshots(a, b, alpha)

        idx = int(max(0, min(int(index), self.frame_count - 1)))
        return self.recording.snapshot_at(idx)

    def current_snapshot(self) -> SkeletonSnapshot | None:
        pos = self.state.frame_float if self.smooth else float(self.state.frame_index)
        return self.snapshot_at(pos)

    def go_to(self, index: int | float) -> SkeletonSnapshot | None:
        if self.frame_count == 0:
            return None
        if isinstance(index, float):
            self.state.frame_float = max(0.0, min(index, self.frame_count - 1))
            self.state.frame_index = int(self.state.frame_float)
        else:
            self.state.frame_index = max(0, min(int(index), self.frame_count - 1))
            self.state.frame_float = float(self.state.frame_index)
        snap = self.current_snapshot()
        if snap is not None:
            self.clock_s = float(snap.time_s)
        return snap

    def go_to_time(self, time_s: float) -> SkeletonSnapshot | None:
        """Seek by source timestamp; hold the last valid snapshot across gaps."""
        if self.frame_count == 0:
            return None
        snaps = self.recording.snapshots
        t = float(time_s)
        first_t = float(snaps[0].time_s)
        last_t = float(snaps[-1].time_s)
        if last_t < first_t:
            last_t = first_t
        t = max(first_t, min(t, last_t))
        self.clock_s = t
        idx = 0
        for i, snap in enumerate(snaps):
            if float(snap.time_s) <= t + 1e-9:
                idx = i
            else:
                break
        self.state.frame_index = idx
        self.state.frame_float = float(idx)
        return self.current_snapshot()

    def play(self) -> None:
        self.state.playing = True
        self.state.stopped = False

    def pause(self) -> None:
        self.state.playing = False
        self.state.stopped = False

    def stop(self, *, reset: bool = False) -> None:
        self.state.playing = False
        self.state.stopped = True
        if reset:
            self.go_to(0)

    def toggle_play(self) -> bool:
        if self.state.playing:
            self.pause()
        else:
            self.play()
        return self.state.playing

    def advance(self, *, speed: float = 1.0, dt: float | None = None) -> SkeletonSnapshot | None:
        """Advance playback by wall-clock time along source timestamps."""
        if not self.state.playing or self.frame_count == 0:
            return self.current_snapshot()

        step = dt if dt is not None else (1.0 / self.fps)
        snaps = self.recording.snapshots
        first_t = float(snaps[0].time_s)
        last_t = float(snaps[-1].time_s)
        if last_t < first_t:
            last_t = first_t
        clock = self.clock_s
        if clock is None:
            clock = float(snaps[self.state.frame_index].time_s)
        clock = float(clock) + float(speed) * float(step)
        if clock > last_t + 1e-9:
            clock = first_t
        return self.go_to_time(clock)

    def time_at_current(self) -> float:
        if self.clock_s is not None:
            return float(self.clock_s)
        snap = self.current_snapshot()
        return snap.time_s if snap else 0.0
