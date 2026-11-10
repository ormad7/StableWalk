"""Timed Compare open/switch harness for UI freeze investigation.

Run::

    set STABLEWALK_UI_FREEZE_DEBUG=1
    python scripts/diagnose_compare_ui_freeze.py

Writes findings to logs/ui_freeze_debug.log (and stall dumps to
logs/ui_freeze_threads.log if the UI thread blocks ≥ 5s).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["STABLEWALK_UI_FREEZE_DEBUG"] = "1"

from stablewalk.analysis.session_compare import CompareMetrics  # noqa: E402
from stablewalk.ui.tk.compare_state import CompareState  # noqa: E402
from stablewalk.ui.tk.dashboard_compare import ComparisonModeController  # noqa: E402
from stablewalk.ui.tk.session_cache import (  # noqa: E402
    AnalyzedSessionSnapshot,
    SessionCompareCache,
)
from stablewalk.ui.tk.ui_freeze_debug import (  # noqa: E402
    debug_log_path,
    log_debug,
    mark_ui_thread,
    timed_op,
)


def _snap(key: str) -> AnalyzedSessionSnapshot:
    return AnalyzedSessionSnapshot(
        key=key,
        label=key.title(),
        n_frames=120,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(
            label=key.title(),
            cadence_spm=100.0 if key == "normal" else 70.0,
            gait_quality=80.0 if key == "normal" else 45.0,
            walking_speed_m_s=1.2 if key == "normal" else 0.7,
            knee_rom_deg=60.0 if key == "normal" else 35.0,
        ),
    )


def main() -> int:
    mark_ui_thread(None)
    log_debug(
        "investigation_start",
        note="Compare freeze root-cause harness",
    )

    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))

    root = MagicMock()
    jobs: list = []

    def _after(ms, func=None, *args):
        if func is None:
            return "sleep"
        jobs.append(func)
        return f"job-{len(jobs)}"

    root.after = _after
    root.after_cancel = MagicMock()

    gui = SimpleNamespace(
        root=root,
        _compare_session_cache=cache,
        _dashboard_notebook=None,
        _tab_compare=None,
        playing=False,
        skeleton_player=None,
        sequence=None,
    )

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl.gui = gui
    ctrl._static_compare_only = True
    ctrl._widgets_built = True
    ctrl._playing = False
    ctrl._static_refresh_inflight = False
    ctrl._static_refresh_fp = None
    ctrl._static_refresh_job = None
    ctrl.compare_state = CompareState.IDLE
    ctrl._compare_load_gen = 0
    ctrl._deferred_jobs = []
    ctrl._visuals_dirty = False
    ctrl.lbl_status = MagicMock()
    ctrl.lbl_compare_error = MagicMock()
    ctrl.diff_table = None
    ctrl._fill_differences_table = MagicMock()  # type: ignore[method-assign]
    ctrl._refresh_chrome_only = MagicMock()  # type: ignore[method-assign]
    ctrl._is_compare_tab_active = lambda: True  # type: ignore[method-assign]
    ctrl._show_compare_error = MagicMock()  # type: ignore[method-assign]
    ctrl._clear_compare_error = MagicMock()  # type: ignore[method-assign]
    ctrl._set_compare_state = ComparisonModeController._set_compare_state.__get__(ctrl)
    ctrl._static_cache_fingerprint = (
        ComparisonModeController._static_cache_fingerprint.__get__(ctrl)
    )

    # Simulate the OLD stacked path cost vs NEW single refresh.
    with timed_op("OLD_stacked_three_refreshes"):
        for i in range(3):
            with timed_op(f"OLD_refresh_{i}"):
                ComparisonModeController._refresh_static_compare(ctrl, force=True)

    ctrl._static_refresh_fp = None
    ctrl.compare_state = CompareState.IDLE
    with timed_op("NEW_activate_once"):
        ComparisonModeController.on_activate_compare_tab(ctrl)

    with timed_op("NEW_twenty_reopens"):
        for _ in range(20):
            ComparisonModeController.on_activate_compare_tab(ctrl)

    chrome_calls = ctrl._refresh_chrome_only.call_count
    log_debug(
        "investigation_result",
        chrome_calls_after_20_reopens=chrome_calls,
        note=(
            "Root cause: stacked sync Compare refreshes on open "
            "(activate + select_dashboard_tab refresh + enter pin/load) "
            "plus full playback sync on tab switch. "
            "Fix: single activate refresh, skip select refresh, "
            "no UI-thread pin on enter, skip playback sync on Compare, "
            "fingerprint skip on reopen, non-black scroll canvas."
        ),
    )
    print(f"Wrote diagnostics to {debug_log_path()}")
    print(f"Chrome refresh calls across 20 reopens after first: {chrome_calls}")
    # Expect: first activate + 3 forced old path ≈ few calls; 20 reopens should
    # not keep growing chrome calls once READY fingerprint matches.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
