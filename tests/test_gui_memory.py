"""Tests for GUI memory teardown helpers and compare session retention."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from stablewalk.analysis.session_compare import CompareMetrics
from stablewalk.ui.tk.gui_memory import (
    _SKELETON_CACHE_MAX,
    collect_active_timer_ids,
    count_cached_frames,
    format_gui_memory_dump,
    trim_skeleton_cache,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    SessionCompareCache,
)
from stablewalk.ui.viewers.chart_interactions import (
    FIG_CROSSHAIR_CID_ATTR,
    attach_chart_crosshair,
)


def test_retain_pinned_slots_keeps_only_ab_and_also_keep() -> None:
    cache = SessionCompareCache()
    for key in ("normal", "abnormal", "athletic", "custom_a", "custom_b"):
        cache.pin(
            AnalyzedSessionSnapshot(
                key=key,
                label=key,
                n_frames=4,
                metrics=CompareMetrics(label=key, cadence_spm=100.0),
            )
        )
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    removed = cache.retain_pinned_slots(also_keep={"custom_a"})
    assert removed >= 2
    assert set(cache.keys()) == {"normal", "abnormal", "custom_a"}


def test_trim_skeleton_cache_bounds_growth() -> None:
    gui = SimpleNamespace(_skeleton_cache={i: object() for i in range(200)})
    removed = trim_skeleton_cache(gui, max_entries=50)
    assert removed == 150
    assert len(gui._skeleton_cache) == 50
    assert _SKELETON_CACHE_MAX >= 64


def test_format_gui_memory_dump_lists_timers_and_caches() -> None:
    gui = SimpleNamespace(
        _after_id="job-main",
        _presentation_after_id=None,
        playing=False,
        _skeleton_cache={1: object(), 2: object()},
        _rgb_cache=SimpleNamespace(__len__=lambda self: 3),
        _photo=None,
        _pb_shared_frame_rgb=None,
        _compare_session_cache=SessionCompareCache(),
        _comparison_mode=SimpleNamespace(
            _playing=False,
            _play_job="job-cmp",
            _video_resize_job=None,
            _scroll_watch_job=None,
            _visual_refresh_job=None,
            _snapshot_poll_job=None,
            _deferred_jobs=[],
            _pil_frame_cache={( "a", 0): object()},
            _photo_cache={},
        ),
    )
    # FrameRgbCache-like: use a real dict-backed fake with __len__
    class _FakeCache:
        def __len__(self) -> int:
            return 3

    gui._rgb_cache = _FakeCache()
    text = format_gui_memory_dump(gui)
    assert "active timers" in text
    assert "job-main" in text
    assert "job-cmp" in text
    assert "cached frame buffers" in text
    assert "skeleton cache entries: 2" in text
    timers = collect_active_timer_ids(gui)
    assert any("job-main" in t for t in timers)
    assert count_cached_frames(gui) >= 3


def test_crosshair_disconnects_cid_tuple() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    disconnected: list[int] = []
    real_disconnect = canvas.mpl_disconnect

    def _track(cid: int) -> None:
        disconnected.append(cid)
        real_disconnect(cid)

    canvas.mpl_disconnect = _track  # type: ignore[method-assign]
    attach_chart_crosshair(fig, canvas)  # type: ignore[arg-type]
    first = getattr(fig, FIG_CROSSHAIR_CID_ATTR)
    assert isinstance(first, tuple) and len(first) == 2
    attach_chart_crosshair(fig, canvas)  # type: ignore[arg-type]
    # Both previous cids must be disconnected (not the tuple as one id).
    assert first[0] in disconnected
    assert first[1] in disconnected
