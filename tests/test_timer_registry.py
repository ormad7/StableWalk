"""Timer registry and generated-files scan cache."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.generated_files_panel import (
    invalidate_generated_files_scan_cache,
    scan_generated_files,
)
from stablewalk.ui.tk.timer_registry import (
    ANCILLARY_AFTER_ATTRS,
    cancel_ancillary_after_jobs,
    count_known_after_jobs,
    schedule_unique,
)


def test_cancel_ancillary_clears_attrs() -> None:
    gui = SimpleNamespace(root=None, _tab_reflow_after="job", _video_resize_after="v")
    cancel_ancillary_after_jobs(gui)
    assert gui._tab_reflow_after is None
    assert gui._video_resize_after is None


def test_count_known_after_jobs() -> None:
    gui = SimpleNamespace(
        _after_id="play",
        _comparison_mode=SimpleNamespace(_play_job="cmp"),
        **{a: None for a in ANCILLARY_AFTER_ATTRS},
    )
    gui._tab_reflow_after = "x"
    assert count_known_after_jobs(gui) == 3


def test_schedule_unique_replace_without_root_runs_inline() -> None:
    gui = SimpleNamespace(root=None, _tab_activate_paint_after=None)
    calls: list[int] = []
    schedule_unique(gui, "_tab_activate_paint_after", 0, lambda: calls.append(1))
    assert calls == [1]


def test_generated_files_scan_cache_ttl() -> None:
    invalidate_generated_files_scan_cache()
    a = scan_generated_files(run_name=None, force=True)
    b = scan_generated_files(run_name=None, force=False)
    assert a == b
    invalidate_generated_files_scan_cache()
