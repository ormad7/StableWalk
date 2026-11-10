"""
Playback render diagnostics for StableWalk dashboard ghosting audits.

Logs widget and canvas item counts at key frames during video playback.
Also provides lightweight timing logs for video / skeleton / graph renders.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from stablewalk.ui.tk.clip_viewport import clip_canvas_item_count
from stablewalk.ui.tk.gui_layout_debug import audit_gui_widget_counts

logger = logging.getLogger(__name__)

_DIAG_FRAMES = frozenset({1, 30, 60, 90, 120})
_PERF_LOG_INTERVAL_S = 2.0
_PERF_SAMPLE_CAP = 120


def render_debug_enabled(gui: Any) -> bool:
    if getattr(gui, "_render_debug", False):
        return True
    return os.environ.get("STABLEWALK_RENDER_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def perf_log_enabled(gui: Any | None = None) -> bool:
    """True when lightweight render timing logs should print."""
    if gui is not None and getattr(gui, "_perf_log", False):
        return True
    return os.environ.get("STABLEWALK_PERF_LOG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


_PERF_SECTIONS = (
    "decode",
    "video",
    "skeleton",
    "trajectory",
    "3d",
    "mpl_draw",
    "tk_paint",
    "chart",
    "compare_prep",
    "opensim",
    "tab_switch",
    "startup",
    "total",
)


@contextmanager
def timed_render(gui: Any, section: str) -> Iterator[None]:
    """Record wall time for video / skeleton / trajectory / chart / total sections."""
    from stablewalk.ui.tk.perf_diagnostics import (
        CHART_REFRESH_BUDGET_MS,
        UI_CALLBACK_BUDGET_MS,
        timed_section,
    )

    # Legacy alias used by older call sites.
    if section == "graph":
        section = "chart"
    budget = None
    label = section
    if section in (
        "video",
        "skeleton",
        "trajectory",
        "3d",
        "tk_paint",
        "decode",
        "total",
    ):
        if getattr(gui, "playing", False):
            budget = UI_CALLBACK_BUDGET_MS
            label = f"{section} (playback UI)"
    elif section in ("chart", "mpl_draw"):
        budget = CHART_REFRESH_BUDGET_MS
        label = f"{section} refresh"
    with timed_section(gui, section, budget_ms=budget, warn_label=label):
        yield


def maybe_log_perf_timings(gui: Any) -> None:
    """Print averaged section timings every few seconds when enabled."""
    from stablewalk.ui.tk.perf_diagnostics import (
        maybe_log_perf_diagnostics,
        perf_diagnostics_enabled,
    )

    if perf_diagnostics_enabled(gui):
        maybe_log_perf_diagnostics(gui)
        return
    if not perf_log_enabled(gui):
        return
    now = time.monotonic()
    last = float(getattr(gui, "_last_perf_log_ts", 0.0) or 0.0)
    if now - last < _PERF_LOG_INTERVAL_S:
        return
    gui._last_perf_log_ts = now
    store = getattr(gui, "_perf_timings", None) or {}
    # Fold legacy "graph" samples into "chart".
    if store.get("graph") and not store.get("chart"):
        store["chart"] = list(store.get("graph") or [])
    parts: list[str] = []
    for key in _PERF_SECTIONS:
        samples = store.get(key) or []
        if not samples:
            parts.append(f"{key}=—")
            continue
        ordered = sorted(samples)
        avg = sum(samples) / len(samples)
        mx = max(samples)
        # Nearest-rank p95
        rank = max(1, int(0.95 * len(ordered) + 0.999999))
        p95 = ordered[min(len(ordered), rank) - 1]
        parts.append(
            f"{key}={avg:.1f}ms(avg)/{mx:.1f}ms(max)/{p95:.1f}ms(p95)/n={len(samples)}"
        )
        samples.clear()
    msg = "[Perf] " + " | ".join(parts)
    print(msg, flush=True)
    logger.debug(msg)


@dataclass(frozen=True)
class PlaybackRenderSnapshot:
    """Widget counts captured at one playback frame."""

    frame: int
    video_clip_items: int
    skel_clip_items: int
    knee_clip_items: int
    traj_clip_items: int
    video_label_count: int
    knee_canvas_count: int
    joint_path_canvas_count: int
    skel_canvas_count: int
    figure_canvas_total: int

    def as_lines(self) -> list[str]:
        return [
            f"frame {self.frame}",
            f"  Video canvas item count: {self.video_clip_items}",
            f"  3D canvas item count: {self.skel_clip_items}",
            f"  Knee graph widget count: {self.knee_canvas_count}",
            f"  Joint path widget count: {self.joint_path_canvas_count}",
            f"  FigureCanvasTkAgg total: {self.figure_canvas_total}",
        ]


def _count_attr_canvas(gui: Any, attr: str) -> int:
    canvas = getattr(gui, attr, None)
    if canvas is None:
        return 0
    return 1 if isinstance(canvas, FigureCanvasTkAgg) else 0


def capture_playback_render_snapshot(gui: Any, frame: int) -> PlaybackRenderSnapshot:
    counts = audit_gui_widget_counts(gui)
    return PlaybackRenderSnapshot(
        frame=int(frame),
        video_clip_items=clip_canvas_item_count(getattr(gui, "_video_clip_canvas", None)),
        skel_clip_items=clip_canvas_item_count(getattr(gui, "_skel_clip_canvas", None)),
        knee_clip_items=clip_canvas_item_count(getattr(gui, "_knee_clip_canvas", None)),
        traj_clip_items=clip_canvas_item_count(getattr(gui, "_traj_clip_canvas", None)),
        video_label_count=counts.get("video_label_instances", 0),
        knee_canvas_count=counts.get("chart_canvas", 0),
        joint_path_canvas_count=counts.get("canvas_dof_traj", 0),
        skel_canvas_count=counts.get("canvas_3d", 0),
        figure_canvas_total=counts.get("FigureCanvasTkAgg_total", 0),
    )


def log_playback_render_diagnostics(gui: Any, frame: int) -> PlaybackRenderSnapshot | None:
    """Print diagnostics at frames 1, 30, 60, 90, and 120 when debug is enabled."""
    if int(frame) not in _DIAG_FRAMES:
        return None
    if not render_debug_enabled(gui):
        return None
    snap = capture_playback_render_snapshot(gui, frame)
    prefix = "[Render debug]"
    for line in snap.as_lines():
        print(f"{prefix} {line}", flush=True)
    return snap


def assert_playback_render_counts_stable(
    gui: Any,
    *,
    baseline: PlaybackRenderSnapshot,
    frame: int,
) -> list[str]:
    """Return failure messages when counts drift from the baseline snapshot."""
    current = capture_playback_render_snapshot(gui, frame)
    failures: list[str] = []
    checks = (
        ("video_label_count", baseline.video_label_count, current.video_label_count),
        ("knee_canvas_count", baseline.knee_canvas_count, current.knee_canvas_count),
        ("joint_path_canvas_count", baseline.joint_path_canvas_count, current.joint_path_canvas_count),
        ("skel_canvas_count", baseline.skel_canvas_count, current.skel_canvas_count),
        ("figure_canvas_total", baseline.figure_canvas_total, current.figure_canvas_total),
        ("video_clip_items", baseline.video_clip_items, current.video_clip_items),
        ("skel_clip_items", baseline.skel_clip_items, current.skel_clip_items),
        ("knee_clip_items", baseline.knee_clip_items, current.knee_clip_items),
        ("traj_clip_items", baseline.traj_clip_items, current.traj_clip_items),
    )
    for name, expected, actual in checks:
        if actual != expected:
            failures.append(f"{name}: expected {expected}, got {actual} at frame {frame}")
    return failures


def reset_playback_render_counters(gui: Any) -> None:
    gui._playback_render_frame = 0
    gui._playback_render_baseline = None
    store = getattr(gui, "_perf_timings", None)
    if isinstance(store, dict):
        for key in store:
            store[key] = []
    gui._last_perf_log_ts = 0.0


def record_playback_render_frame(gui: Any) -> int:
    """Increment playback frame counter and optionally log diagnostics."""
    frame = int(getattr(gui, "_playback_render_frame", 0)) + 1
    gui._playback_render_frame = frame
    snap = log_playback_render_diagnostics(gui, frame)
    if frame == 1 and snap is not None:
        gui._playback_render_baseline = snap
    baseline = getattr(gui, "_playback_render_baseline", None)
    if baseline is not None and frame in _DIAG_FRAMES:
        failures = assert_playback_render_counts_stable(gui, baseline=baseline, frame=frame)
        for msg in failures:
            print(f"[Render debug] STABILITY FAIL: {msg}", flush=True)
        store = getattr(gui, "_playback_render_failures", None)
        if store is None:
            store = []
            gui._playback_render_failures = store
        store.extend(failures)
    return frame


def run_playback_render_stress_test(
    gui: Any,
    *,
    frames: int = 120,
    scroll_during_playback: bool = False,
) -> list[tuple[str, bool, str]]:
    """
    Advance playback for ``frames`` ticks and assert render singleton stability.

    Requires ``skeleton_player`` with loaded gait data. When data is missing,
    only layout singleton checks run.
    """
    from stablewalk.ui.tk.dashboard_notebook import TAB_ADVANCED, TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.gui_visual_qa import scroll_dashboard_to

    results: list[tuple[str, bool, str]] = []
    reset_playback_render_counters(gui)
    gui._render_debug = True

    player = getattr(gui, "skeleton_player", None)
    has_data = player is not None and getattr(player, "frame_count", 0) > 0

    if has_data and not getattr(gui, "playing", False):
        toggle = getattr(gui, "_toggle_play", None)
        if toggle is not None:
            try:
                toggle()
            except Exception:
                pass

    baseline = capture_playback_render_snapshot(gui, 0)
    tick = getattr(gui, "_tick", None)
    show_pose = getattr(gui, "_show_pose_at", None)

    for i in range(1, frames + 1):
        if scroll_during_playback and i % 10 == 0:
            select_dashboard_tab(gui, TAB_ADVANCED if i % 20 == 0 else TAB_OVERVIEW)
            scroll_dashboard_to(gui, (i % 100) / 100.0)
        if getattr(gui, "playing", False) and tick is not None:
            try:
                tick()
            except Exception:
                break
        elif has_data and show_pose is not None:
            try:
                pos = (i - 1) % max(player.frame_count, 1)
                show_pose(pos, force_draw=False, skeleton_only=False)
            except Exception:
                break
        try:
            gui.root.update_idletasks()
        except Exception:
            pass
        record_playback_render_frame(gui)

    final = capture_playback_render_snapshot(gui, frames)
    failures = assert_playback_render_counts_stable(gui, baseline=baseline, frame=frames)
    stored = list(getattr(gui, "_playback_render_failures", []))
    failures = failures + stored

    results.append(
        (
            "video_label_singleton",
            final.video_label_count == 1,
            f"count={final.video_label_count}",
        )
    )
    results.append(
        (
            "knee_canvas_singleton",
            final.knee_canvas_count == 1,
            f"count={final.knee_canvas_count}",
        )
    )
    results.append(
        (
            "joint_path_canvas_singleton",
            final.joint_path_canvas_count == 1,
            f"count={final.joint_path_canvas_count}",
        )
    )
    results.append(
        (
            "skel_canvas_singleton",
            final.skel_canvas_count == 1,
            f"count={final.skel_canvas_count}",
        )
    )
    results.append(
        (
            "video_clip_items_stable",
            final.video_clip_items == baseline.video_clip_items,
            f"baseline={baseline.video_clip_items} final={final.video_clip_items}",
        )
    )
    results.append(
        (
            "playback_render_counts_stable",
            not failures,
            "; ".join(failures) if failures else f"stable through frame {frames}",
        )
    )
    results.append(
        (
            "playback_frames_exercised",
            int(getattr(gui, "_playback_render_frame", 0)) >= min(frames, 1),
            f"frames={getattr(gui, '_playback_render_frame', 0)}",
        )
    )

    if getattr(gui, "playing", False):
        toggle = getattr(gui, "_toggle_play", None)
        if toggle is not None:
            try:
                toggle()
            except Exception:
                pass

    return results
