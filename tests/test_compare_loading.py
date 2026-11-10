"""Tests for Compare background loading lifecycle (strict state machine)."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from stablewalk.analysis.session_compare import CompareMetrics
from stablewalk.ui.tk.compare_state import (
    COMPARE_LOAD_TIMEOUT_S,
    COMPARE_SESSION_B_NOT_ANALYZED,
    COMPARE_TIMEOUT_MESSAGE,
    SESSION_B_WAITING_TITLE,
    CompareState,
    ENABLE_DETAILED_COMPARE,
    compare_error_log_path,
    log_compare_exception,
)
from stablewalk.ui.tk.dashboard_compare import (
    ComparisonModeController,
    _COMPARE_WORKER_SENTINEL,
)
from stablewalk.ui.tk.session_cache import (
    AnalyzedSessionSnapshot,
    SessionCompareCache,
    demo_snapshot_is_compare_ready,
    ensure_demo_sessions_pinned,
    pin_current_session,
)


def _fake_gui(cache: SessionCompareCache | None = None) -> SimpleNamespace:
    cache = cache or SessionCompareCache()
    root = MagicMock()
    root.after = MagicMock(return_value="job")
    root.after_cancel = MagicMock()
    root.after_idle = MagicMock(return_value="idle")
    return SimpleNamespace(
        root=root,
        _compare_session_cache=cache,
        _dashboard_notebook=None,
        _tab_compare=None,
        playing=False,
        skeleton_player=None,
    )


def _bare_controller(
    gui: SimpleNamespace, *, static: bool = False
) -> ComparisonModeController:
    """Bare controller. ``static=False`` exercises the detailed worker path."""
    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl.gui = gui
    ctrl._static_compare_only = static
    ctrl._detailed_comparison = not static
    ctrl._videos_built = True
    ctrl._widgets_built = True
    ctrl._static_refresh_inflight = False
    ctrl._static_refresh_fp = None
    ctrl._static_refresh_job = None
    ctrl._loading_keys = set()
    ctrl._failed_snapshot_keys = set()
    ctrl._snapshot_results = __import__("queue").Queue()
    ctrl._snapshot_poll_job = None
    ctrl._snapshot_build_lock = threading.Lock()
    ctrl.compare_state = CompareState.IDLE
    ctrl._compare_worker = None
    ctrl._compare_pending_keys = []
    ctrl._compare_load_gen = 0
    ctrl._compare_load_started_monotonic = None
    ctrl._analyzing_session_b_key = None
    ctrl._session_b_resolved_path = None
    ctrl._session_b_pose_backend = None
    ctrl._session_b_max_frames = None
    ctrl._session_b_analysis_mode = None
    ctrl._session_b_error = None
    ctrl._session_b_cta_analyze_buttons = []
    ctrl.compare_session_b = None
    ctrl._demo_compare_load_pending = False
    ctrl._compare_timeout_job = None
    ctrl._playback_resume_after_load = None
    ctrl._playback_resume_when_leaving = False
    ctrl._playing = False
    ctrl._progress = 0.0
    ctrl._visual_refresh_job = None
    ctrl._deferred_jobs = []
    ctrl._visuals_dirty = False
    ctrl.lbl_status = MagicMock()
    ctrl.lbl_compare_error = MagicMock()
    ctrl._loading_banner = MagicMock()
    ctrl.lbl_loading_banner = MagicMock()
    ctrl.btn_cancel_compare = MagicMock()
    ctrl.diff_table = None
    ctrl._is_compare_tab_active = lambda: True  # type: ignore[method-assign]
    ctrl._refresh_chrome_only = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_loading_banner = MagicMock()  # type: ignore[method-assign]
    ctrl._pause_playback_for_compare_load = MagicMock()  # type: ignore[method-assign]
    ctrl._arm_snapshot_poll = MagicMock()  # type: ignore[method-assign]
    ctrl._arm_compare_timeout = MagicMock()  # type: ignore[method-assign]
    ctrl.cancel_deferred_work = MagicMock()  # type: ignore[method-assign]
    ctrl._restore_playback_after_compare_load = MagicMock()  # type: ignore[method-assign]
    ctrl.release_temp_buffers = MagicMock()  # type: ignore[method-assign]
    ctrl._fill_differences_table = MagicMock()  # type: ignore[method-assign]
    return ctrl


def _snap(key: str = "abnormal") -> AnalyzedSessionSnapshot:
    return AnalyzedSessionSnapshot(
        key=key,
        label=key.title(),
        n_frames=10,
        skeletons={0: object()},  # type: ignore[dict-item]
        metrics=CompareMetrics(label=key.title(), cadence_spm=80.0),
    )


def _demo_ready_snap(key: str = "abnormal") -> AnalyzedSessionSnapshot:
    """Snapshot that Load demo can reuse without rebuilding analysis."""
    return AnalyzedSessionSnapshot(
        key=key,
        label=key.title(),
        n_frames=10,
        skeletons={0: object()},  # type: ignore[dict-item]
        frame_paths=[Path(f"{key}_frame_000000.jpg")],
        metrics=CompareMetrics(
            label=key.title(),
            cadence_spm=80.0,
            symmetry_pct=70.0,
            stability_margin_m=0.01,
            knee_rom_deg=40.0,
        ),
    )


def test_timeout_is_15_seconds() -> None:
    assert COMPARE_LOAD_TIMEOUT_S == 15.0
    assert "still available" in COMPARE_TIMEOUT_MESSAGE.lower()


def test_detailed_compare_flag_defaults_true() -> None:
    assert ENABLE_DETAILED_COMPARE is True


def test_ensure_demo_sessions_pinned_never_analyzes() -> None:
    cache = SessionCompareCache()
    with patch(
        "stablewalk.ui.tk.session_cache.try_build_demo_snapshot"
    ) as build:
        loaded = ensure_demo_sessions_pinned(cache)
    assert loaded == []
    build.assert_not_called()


def test_cache_hit_sets_ready_without_worker() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    with patch.object(ctrl, "_compare_worker_main"):
        ctrl._request_compare_snapshots(["abnormal"])
    assert ctrl._compare_worker is None
    # One cached side is not a comparison-ready pair.
    assert ctrl.compare_state == CompareState.IDLE
    assert not ctrl.compare_loading


def test_loading_blocks_second_request() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    with patch("threading.Thread") as thread_cls:
        ctrl._request_compare_snapshots(["normal", "abnormal"], allow_retry=True)
    thread_cls.assert_not_called()
    assert ctrl.compare_state == CompareState.LOADING
    ctrl.lbl_status.configure.assert_called()


def test_single_worker_per_load() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    started: list[object] = []

    class _Thread:
        def __init__(self, *a, **k):
            self.kwargs = k
            started.append(self)

        def start(self):
            pass

        def is_alive(self):
            return False

    with patch("stablewalk.ui.tk.dashboard_compare.threading.Thread", _Thread):
        ctrl._request_compare_snapshots(["abnormal"])
        assert ctrl.compare_state == CompareState.LOADING
        ctrl._request_compare_snapshots(["normal"])  # blocked
    assert len(started) == 1


def test_stale_generation_results_ignored() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._compare_load_gen = 2
    ctrl._finish_compare_load = MagicMock()  # type: ignore[method-assign]
    ctrl._snapshot_results.put((1, "abnormal", _snap(), None))  # stale
    ctrl._snapshot_results.put((2, _COMPARE_WORKER_SENTINEL, None, None))
    ctrl._drain_snapshot_results()
    assert gui._compare_session_cache.get("abnormal") is None


def test_cancel_bumps_generation_and_sets_cancelled() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_load_gen = 3
    ctrl.cancel_compare_load(reason="user_cancel")
    assert ctrl.compare_state == CompareState.CANCELLED
    assert ctrl._compare_load_gen == 4
    assert not ctrl._loading_keys


def test_timeout_marks_failed() -> None:
    import time

    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_load_started_monotonic = time.monotonic() - 20.0
    ctrl._finish_compare_load = MagicMock()  # type: ignore[method-assign]
    ctrl._check_compare_timeout()
    assert ctrl.compare_state == CompareState.FAILED
    ctrl.lbl_status.configure.assert_called()
    args, kwargs = ctrl.lbl_status.configure.call_args
    text = kwargs.get("text") or (args[0] if args else "")
    assert "still available" in str(text).lower() or COMPARE_TIMEOUT_MESSAGE in str(
        text
    )


def test_ten_reopen_while_loading_does_not_spawn_workers() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    with patch("threading.Thread") as thread_cls:
        for _ in range(10):
            ctrl._request_compare_snapshots(["abnormal", "normal"], allow_retry=True)
            ctrl.on_activate_compare_tab()
    thread_cls.assert_not_called()


def test_ready_activate_is_static_refresh_only() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=True)
    ctrl.compare_state = CompareState.IDLE
    with patch("threading.Thread") as thread_cls:
        ctrl.on_activate_compare_tab()
        # Second reopen with same cache must not re-run chrome (fingerprint skip).
        ctrl.on_activate_compare_tab()
    thread_cls.assert_not_called()
    assert ctrl.compare_state == CompareState.READY
    assert ctrl._refresh_chrome_only.call_count == 1
    assert ctrl._fill_differences_table.call_count == 1


def test_current_analyzed_recording_is_restored_to_session_a_without_reanalysis() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("current_walk"))
    cache.pin(_snap("abnormal"))
    cache.left_key = "abnormal"
    cache.right_key = "current_walk"
    gui = _fake_gui(cache)
    gui._current_source = r"C:\sessions\current_walk.mp4"
    gui._session_display_src = None
    gui._active_demo_gait = None
    ctrl = _bare_controller(gui, static=False)
    selected: list[str] = []
    ctrl.left_key_var = SimpleNamespace(set=selected.append)
    ctrl._cached_comparison = object()

    with patch(
        "stablewalk.ui.tk.session_cache.build_snapshot_from_gui"
    ) as rebuild:
        assert ctrl._sync_current_session_a() is True

    rebuild.assert_not_called()
    assert cache.left_key == "current_walk"
    assert cache.right_key == ""
    assert selected == ["current_walk"]


def test_compare_summary_stays_hidden_until_both_sessions_ready() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl._compare_summary_host = MagicMock()
    ctrl._compare_bottom_host = MagicMock()
    ctrl._compare_waiting_host = MagicMock()
    ctrl._compare_waiting_label = MagicMock()

    ctrl._sync_compare_ready_sections(False, message="Waiting for Session B")
    ctrl._compare_summary_host.grid_remove.assert_called_once()
    ctrl._compare_bottom_host.grid_remove.assert_called_once()
    ctrl._compare_waiting_host.grid.assert_called_once()

    ctrl._sync_compare_ready_sections(True)
    ctrl._compare_waiting_host.grid_remove.assert_called_once()
    ctrl._compare_summary_host.grid.assert_called_once()
    ctrl._compare_bottom_host.grid.assert_called_once()


def test_compare_time_readout_is_one_based_and_unit_spaced() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.lbl_time = MagicMock()
    left = _snap("left")
    right = _snap("right")
    left.fps = 25.0
    right.fps = 50.0

    ctrl._update_time_label(left, right, 0, 9)

    text = ctrl.lbl_time.configure.call_args.kwargs["text"]
    assert text == "A F1/10 · 0.00 s\nB F10/10 · 0.18 s"
    assert "F0" not in text


def test_refresh_active_tab_charts_skips_compare() -> None:
    """Overview→Compare must not run full playback sync (freeze root cause)."""
    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI.__new__(StableWalkGUI)
    app._sync_all_guard = False
    app._tab_compare = object()
    app._tab_motion = object()
    app._active_tab_widget = lambda: app._tab_compare  # type: ignore[method-assign]
    app.refresh_motion_analysis_tab = MagicMock()  # type: ignore[method-assign]
    app._sync_all_to_frame = MagicMock()  # type: ignore[method-assign]
    app.skeleton_player = SimpleNamespace(frame_count=10)
    StableWalkGUI._refresh_active_tab_charts(app)
    app._sync_all_to_frame.assert_not_called()
    app.refresh_motion_analysis_tab.assert_not_called()


def test_static_compare_never_starts_worker_on_missing_b() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=True)
    ctrl.left_key_var = SimpleNamespace(get=lambda: "normal")
    ctrl.right_key_var = SimpleNamespace(get=lambda: "abnormal")
    with patch("threading.Thread") as thread_cls:
        ctrl._request_compare_snapshots(["normal", "abnormal"], allow_retry=True)
        ComparisonModeController._on_slot_change(ctrl)
    thread_cls.assert_not_called()
    assert ctrl._compare_worker is None
    status_texts = [
        str(getattr(c, "kwargs", {}).get("text", "") or "")
        for c in ctrl.lbl_status.configure.call_args_list
    ]
    assert any(COMPARE_SESSION_B_NOT_ANALYZED in t for t in status_texts)


def test_static_enter_async_load_is_cache_only() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=True)
    with patch("threading.Thread") as thread_cls:
        with patch(
            "stablewalk.ui.tk.session_cache.pin_current_session", return_value=None
        ):
            ComparisonModeController._enter_async_load(ctrl)
    thread_cls.assert_not_called()
    ctrl._refresh_chrome_only.assert_called()
    assert ctrl.compare_state == CompareState.READY


def test_paint_frame_noop_in_static_mode() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=True)
    ctrl._compare_painting = False
    ctrl._detailed_comparison = False
    ComparisonModeController._paint_frame(ctrl, force=True)
    # No exception and no paint side effects.


def test_open_detailed_is_noop_when_already_open() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl._detailed_comparison = True
    ctrl._detailed_section = MagicMock()
    ComparisonModeController._open_detailed_comparison(ctrl)
    assert ctrl._detailed_comparison is True
    ctrl.lbl_status.configure.assert_called()


def test_log_compare_exception_writes_traceback(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "compare_error.log"
    monkeypatch.setattr(
        "stablewalk.ui.tk.compare_state.compare_error_log_path",
        lambda: log_path,
    )
    gui = _fake_gui()
    try:
        raise RuntimeError("compare blew up")
    except RuntimeError as exc:
        msg = log_compare_exception(gui, "unit_test", exc)
    assert "compare_error.log" in msg
    text = log_path.read_text(encoding="utf-8")
    assert "RuntimeError: compare blew up" in text
    assert "Traceback" in text
    assert "unit_test" in text
    # Default project path remains under logs/ when not monkeypatched.
    assert compare_error_log_path().name == "compare_error.log"


def test_repeated_static_activate_and_slot_change_no_workers() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=True)
    ctrl.left_key_var = SimpleNamespace(get=lambda: cache.left_key)
    ctrl.right_key_var = SimpleNamespace(get=lambda: cache.right_key)
    with patch("threading.Thread") as thread_cls:
        for i in range(20):
            ctrl.on_activate_compare_tab()
            if i % 2 == 0:
                cache.right_key = "abnormal"
            else:
                cache.right_key = "normal"
            ComparisonModeController._on_slot_change(ctrl)
    thread_cls.assert_not_called()
    assert ctrl._compare_worker is None


def test_load_demo_comparison_reuses_cached_normal_vs_abnormal() -> None:
    cache = SessionCompareCache()
    cache.pin(_demo_ready_snap("normal"))
    cache.pin(_demo_ready_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=lambda k: None)
    ctrl.right_key_var = SimpleNamespace(set=lambda k: None)
    ctrl.request_refresh = MagicMock()  # type: ignore[method-assign]
    with patch("threading.Thread") as thread_cls:
        ctrl.load_demo_comparison()
    assert cache.left_key == "normal"
    assert cache.right_key == "abnormal"
    assert cache.get("normal") is not None
    assert cache.get("abnormal") is not None
    ctrl.request_refresh.assert_called_once()
    thread_cls.assert_not_called()
    assert ctrl.compare_state == CompareState.READY
    assert ctrl._progress == 0.0


def test_load_demo_comparison_requests_missing_snapshot_in_background() -> None:
    cache = SessionCompareCache()
    normal = _demo_ready_snap("normal")
    cache.pin(normal)
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=lambda k: None)
    ctrl.right_key_var = SimpleNamespace(set=lambda k: None)
    ctrl._request_compare_snapshots = MagicMock()  # type: ignore[method-assign]

    ctrl.load_demo_comparison()

    assert cache.left() is normal or cache.left() is not None
    assert cache.right_key == "abnormal"
    assert ctrl._demo_compare_load_pending is True
    assert ctrl._analyzing_session_b_key == "abnormal"
    ctrl._request_compare_snapshots.assert_called_once_with(
        ["abnormal"], allow_retry=True, explicit_analysis=True
    )
    assert ctrl.lbl_status.configure.call_args.kwargs["text"] != (
        "Demo comparison: Normal vs Abnormal"
    )


def test_load_demo_comparison_requests_both_missing_demos() -> None:
    cache = SessionCompareCache()
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=lambda k: None)
    ctrl.right_key_var = SimpleNamespace(set=lambda k: None)
    ctrl._request_compare_snapshots = MagicMock()  # type: ignore[method-assign]

    ctrl.load_demo_comparison()

    assert cache.left_key == "normal"
    assert cache.right_key == "abnormal"
    assert ctrl._demo_compare_load_pending is True
    ctrl._request_compare_snapshots.assert_called_once_with(
        ["normal", "abnormal"], allow_retry=True, explicit_analysis=True
    )


def test_compare_worker_results_keep_polling_off_tab() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_worker = MagicMock()
    ctrl._compare_worker.is_alive.return_value = True
    ctrl._is_compare_tab_active = lambda: False  # type: ignore[method-assign]

    ComparisonModeController._drain_snapshot_results(ctrl)

    gui.root.after.assert_called_with(100, ctrl._drain_snapshot_results)
    assert ctrl._snapshot_poll_job == "job"


def test_reopening_ready_compare_repaints_visuals_completed_off_tab() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl._videos_built = True
    ctrl._skeletons_built = True
    ctrl._lite_chart_built = True
    ctrl._graphs_built = True
    ctrl._visuals_dirty = True
    ctrl._sync_current_session_a = MagicMock(return_value=False)  # type: ignore[method-assign]
    ctrl._refresh_static_compare = MagicMock()  # type: ignore[method-assign]
    ctrl._queue_detailed_visual_refresh = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_session_b_cta = MagicMock()  # type: ignore[method-assign]

    ctrl.on_activate_compare_tab()

    ctrl._queue_detailed_visual_refresh.assert_called_once_with()


def test_forced_ready_refresh_paints_all_compare_visuals_once() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl._lite_chart_built = True
    ctrl._graphs_built = True
    ctrl._skeletons_built = True
    ctrl._pil_frame_cache = {}
    ctrl._photo_cache = {}
    ctrl._video_box = {}
    ctrl._section_visible = MagicMock(return_value=False)  # type: ignore[method-assign]
    ctrl._paint_videos_only = MagicMock()  # type: ignore[method-assign]
    ctrl._paint_skeletons = MagicMock()  # type: ignore[method-assign]
    ctrl._ensure_lite_chart_built = MagicMock()  # type: ignore[method-assign]
    ctrl._redraw_chart_series = MagicMock()  # type: ignore[method-assign]
    ctrl._update_chart_playhead = MagicMock()  # type: ignore[method-assign]
    ctrl._refresh_detailed_charts = MagicMock()  # type: ignore[method-assign]

    ctrl._refresh_visible_visuals(force=True)

    ctrl._paint_videos_only.assert_called_once_with(force=True)
    ctrl._paint_skeletons.assert_called_once_with(cache.left(), cache.right())
    ctrl._redraw_chart_series.assert_called_once_with()
    ctrl._refresh_detailed_charts.assert_called_once_with(force=True)


def test_analyzed_session_b_without_video_does_not_fabricate_a_frame() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    label = MagicMock()
    panel = {"label": label, "frame": MagicMock()}
    ctrl.video_b = panel
    ctrl._session_b_needs_cta = MagicMock(return_value=False)  # type: ignore[method-assign]
    ctrl._set_video_b_cta_visible = MagicMock()  # type: ignore[method-assign]
    ctrl._load_frame_image = MagicMock(return_value=None)  # type: ignore[method-assign]
    snap = _snap("session_b")

    ctrl._paint_video(panel, snap, 0)

    ctrl._set_video_b_cta_visible.assert_called_once_with(False)
    configured = label.configure.call_args.kwargs
    assert configured["image"] == ""
    assert "Video B unavailable" in configured["text"]
    assert "No stored video frame" in configured["text"]


def test_pending_demo_load_starts_after_prior_worker_finishes() -> None:
    cache = SessionCompareCache()
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl._demo_compare_load_pending = True
    requested: list[list[str]] = []

    def _request(keys, *, allow_retry=False, explicit_analysis=False):
        requested.append(list(keys))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = set(keys)

    ctrl._request_compare_snapshots = _request  # type: ignore[method-assign]

    ctrl._finish_compare_load()

    assert requested == [["normal", "abnormal"]]
    assert ctrl.compare_state == CompareState.LOADING
    assert ctrl._demo_compare_load_pending is True


def test_detailed_cached_refresh_queues_video_skeleton_and_graph_stage() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl._refresh_static_compare = MagicMock()  # type: ignore[method-assign]
    ctrl._run_deferred_visual_refresh = MagicMock()  # type: ignore[method-assign]

    ctrl.request_refresh()

    ctrl._refresh_static_compare.assert_called_once_with(force=True)
    gui.root.after.assert_called_once_with(0, ctrl._run_deferred_visual_refresh)
    assert ctrl._visuals_dirty is True
    assert ctrl._paths_dirty is True
    assert ctrl._chart_dirty is True
    assert ctrl._heat_dirty is True


def test_slot_change_rejects_accidental_self_comparison() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=True)
    ctrl.left_key_var = MagicMock()
    ctrl.right_key_var = MagicMock()
    ctrl.left_key_var.get.return_value = "normal"
    ctrl.right_key_var.get.return_value = "normal"

    with patch("stablewalk.ui.tk.tk_lifecycle.safe_messagebox"):
        ctrl._on_slot_change(changed_slot="right")

    assert cache.left_key == "normal"
    assert cache.right_key == "abnormal"
    ctrl.right_key_var.set.assert_called_with("abnormal")
    ctrl._refresh_chrome_only.assert_not_called()


def test_worker_analyzes_selected_video_once_then_reuses_cache() -> None:
    key = "video:C:/walks/session_b.mp4"
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = key
    cache.pin(_snap("normal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {key}
    ctrl._compare_pending_keys = [key]
    snap_b = _snap(key)

    with patch(
        "stablewalk.ui.tk.session_cache.try_build_video_snapshot",
        return_value=snap_b,
    ) as build:
        ctrl._compare_worker_main(generation=0)
        ctrl._drain_snapshot_results()
        with patch("threading.Thread") as thread_cls:
            ctrl._request_compare_snapshots(
                [key],
                allow_retry=True,
                explicit_analysis=True,
            )

    build.assert_called_once_with(key)
    thread_cls.assert_not_called()
    assert cache.get(key) is snap_b
    assert ctrl.compare_state == CompareState.READY


def test_worker_result_does_not_refresh_partial_comparison() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"normal", "abnormal"}
    ctrl._snapshot_results.put((0, "normal", _snap("normal"), None))

    ctrl._drain_snapshot_results()

    assert cache.get("normal") is not None
    assert "abnormal" in ctrl._loading_keys
    ctrl._refresh_chrome_only.assert_not_called()


def test_completed_workspace_analysis_becomes_session_a() -> None:
    cache = SessionCompareCache()
    cache.right_key = "current"
    gui = _fake_gui(cache)
    current = _snap("current")

    with patch(
        "stablewalk.ui.tk.session_cache.build_snapshot_from_gui",
        return_value=current,
    ):
        pinned = pin_current_session(gui, refresh=False, slot="left")

    assert pinned is current
    assert cache.left_key == "current"
    assert cache.right_key == ""
    assert cache.left() is current


def test_static_refresh_does_not_clear_dirty_visuals_when_session_b_ready() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl._visuals_dirty = True
    ctrl.compare_state = CompareState.READY

    ctrl._refresh_static_compare(force=True)

    assert ctrl._visuals_dirty is True
    assert cache.left().key == "normal"
    assert cache.right().key == "abnormal"


def test_reopening_ready_compare_repaints_after_real_static_refresh() -> None:
    cache = SessionCompareCache()
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    snap_a = _snap("normal")
    snap_b = _snap("abnormal")
    cache.pin(snap_a)
    cache.pin(snap_b)
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl._videos_built = True
    ctrl._skeletons_built = True
    ctrl._lite_chart_built = True
    ctrl._graphs_built = True
    ctrl._visuals_dirty = False
    ctrl._sync_current_session_a = MagicMock(return_value=False)  # type: ignore[method-assign]
    ctrl._queue_detailed_visual_refresh = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_session_b_cta = MagicMock()  # type: ignore[method-assign]

    ctrl.on_activate_compare_tab()

    ctrl._queue_detailed_visual_refresh.assert_called_once_with()
    assert cache.left() is snap_a
    assert cache.right() is snap_b


def test_slot_change_to_cached_session_b_queues_visual_refresh() -> None:
    key_b = "video:C:/walks/session_b.mp4"
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    snap_b = _snap(key_b)
    cache.pin(snap_a)
    cache.pin(snap_b)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(get=lambda: "normal", set=MagicMock())
    ctrl.right_key_var = SimpleNamespace(get=lambda: key_b, set=MagicMock())
    ctrl._queue_detailed_visual_refresh = MagicMock()  # type: ignore[method-assign]

    ctrl._on_slot_change(changed_slot="right")

    assert cache.left() is snap_a
    assert cache.right() is snap_b
    ctrl._queue_detailed_visual_refresh.assert_called_once_with()


def test_pinning_session_b_does_not_replace_session_a_or_gui_sequence() -> None:
    key_b = "video:C:/walks/session_b.mp4"
    seq_a = object()
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    snap_a.sequence = seq_a  # type: ignore[assignment]
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = key_b
    gui = _fake_gui(cache)
    gui.sequence = seq_a
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {key_b}
    snap_b = _snap(key_b)
    snap_b.sequence = object()  # type: ignore[assignment]
    ctrl._snapshot_results.put((0, key_b, snap_b, None))
    ctrl._snapshot_results.put((0, _COMPARE_WORKER_SENTINEL, None, None))

    ctrl._drain_snapshot_results()

    assert cache.left() is snap_a
    assert cache.left().sequence is seq_a
    assert gui.sequence is seq_a
    assert cache.right() is snap_b
    assert cache.right() is not snap_a


def test_worker_cannot_pin_session_b_over_session_a_object() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "video:C:/walks/session_b.mp4"
    gui = _fake_gui(cache)
    gui.sequence = object()
    seq_gui = gui.sequence
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {cache.right_key}
    ctrl._snapshot_results.put((0, cache.right_key, snap_a, None))
    ctrl._snapshot_results.put((0, _COMPARE_WORKER_SENTINEL, None, None))

    ctrl._drain_snapshot_results()

    assert cache.left() is snap_a
    assert cache.get(cache.right_key) is None
    assert cache.right_key in ctrl._failed_snapshot_keys
    assert gui.sequence is seq_gui


def test_session_b_snapshot_is_rekeyed_onto_b_slot() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    snap_b = _snap("wrong-key")
    ctrl._snapshot_results.put((0, "abnormal", snap_b, None))
    ctrl._snapshot_results.put((0, _COMPARE_WORKER_SENTINEL, None, None))

    ctrl._drain_snapshot_results()

    assert cache.left() is snap_a
    assert cache.right() is not None
    assert cache.right() is not snap_a
    assert cache.right().key == "abnormal"


def test_analyze_session_b_starts_when_compare_stuck_loading() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = set()
    ctrl._analyzing_session_b_key = None
    ctrl._compare_worker = None
    calls: list[tuple[list[str], dict]] = []

    def _start(keys, **kwargs):
        calls.append((list(keys), kwargs))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = set(keys)

    ctrl._request_compare_snapshots = _start  # type: ignore[method-assign]
    ctrl.analyze_session_b()

    assert calls == [(["abnormal"], {"allow_retry": True, "explicit_analysis": True})]
    assert ctrl._analyzing_session_b_key == "abnormal"
    assert cache.left() is snap_a
    assert cache.right() is None


def test_analyze_session_b_replaces_orphan_worker() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.IDLE
    ctrl._loading_keys = set()

    class _Orphan:
        def is_alive(self) -> bool:
            return True

    ctrl._compare_worker = _Orphan()
    ctrl._analyzing_session_b_key = "abnormal"
    calls: list[tuple[list[str], dict]] = []

    def _start(keys, **kwargs):
        calls.append((list(keys), kwargs))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = set(keys)

    ctrl._request_compare_snapshots = _start  # type: ignore[method-assign]
    ctrl.analyze_session_b()

    assert calls == [(["abnormal"], {"allow_retry": True, "explicit_analysis": True})]
    assert ctrl._analyzing_session_b_key == "abnormal"


def test_request_snapshots_unsticks_empty_loading_for_explicit_b() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = set()
    ctrl._analyzing_session_b_key = "abnormal"
    started: list[object] = []

    class _Thread:
        def __init__(self, *a, **k):
            started.append(self)

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    with patch("stablewalk.ui.tk.dashboard_compare.threading.Thread", _Thread):
        ctrl._request_compare_snapshots(
            ["abnormal"], allow_retry=True, explicit_analysis=True
        )

    assert len(started) == 1
    assert ctrl.compare_state == CompareState.LOADING
    assert ctrl._loading_keys == {"abnormal"}


def test_analyze_session_b_starts_worker_for_uncached_demo() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    gui.sequence = object()
    seq_gui = gui.sequence
    ctrl = _bare_controller(gui, static=False)
    calls: list[tuple[list[str], dict]] = []

    def _start(keys, **kwargs):
        calls.append((list(keys), kwargs))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = {"abnormal"}

    ctrl._request_compare_snapshots = _start  # type: ignore[method-assign]
    ctrl.analyze_session_b()

    assert calls == [(["abnormal"], {"allow_retry": True, "explicit_analysis": True})]
    assert ctrl._analyzing_session_b_key == "abnormal"
    assert cache.left() is snap_a
    assert cache.right_key == "abnormal"
    assert cache.right() is None
    assert gui.sequence is seq_gui


def test_session_b_analysis_uses_long_timeout() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl._loading_keys = {"abnormal"}
    ctrl._analyzing_session_b_key = "abnormal"
    assert ctrl._compare_load_timeout_s() == 300.0
    ctrl._analyzing_session_b_key = None
    assert ctrl._compare_load_timeout_s() == COMPARE_LOAD_TIMEOUT_S


def test_session_b_timeout_not_15s_while_analyzing() -> None:
    import time

    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._compare_load_started_monotonic = time.monotonic() - 20.0
    ctrl._finish_compare_load = MagicMock()  # type: ignore[method-assign]
    ctrl._check_compare_timeout()
    assert ctrl.compare_state == CompareState.LOADING
    ctrl._finish_compare_load.assert_not_called()


def test_session_b_failure_surfaces_technical_reason() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    gui.sequence = object()
    seq_gui = gui.sequence
    ctrl = _bare_controller(gui, static=False)
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    reason = "MediaPipe detector weights not found"
    ctrl._snapshot_results.put((0, "abnormal", None, reason))
    ctrl._snapshot_results.put((0, _COMPARE_WORKER_SENTINEL, None, None))

    ctrl._drain_snapshot_results()

    assert cache.left() is snap_a
    assert cache.get("abnormal") is None
    assert gui.sequence is seq_gui
    assert reason in (ctrl._session_b_error or "")
    status = str(ctrl.lbl_status.configure.call_args)
    assert reason in status
    both_ready, message = ctrl._both_sessions_chart_ready()
    assert both_ready is False
    assert reason in message
    assert SESSION_B_WAITING_TITLE not in message


def test_compare_worker_builds_demo_with_reason() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_pending_keys = ["abnormal"]
    snap_b = _snap("abnormal")
    with patch(
        "stablewalk.ui.tk.session_cache.build_demo_snapshot_with_reason",
        return_value=(snap_b, None),
    ) as build:
        ctrl._compare_worker_main(generation=0)
    build.assert_called_once_with("abnormal")
    ctrl._drain_snapshot_results()
    assert cache.get("abnormal") is snap_b
    assert cache.left() is snap_a


def test_demo_snapshot_falls_back_to_video_when_poses_missing(tmp_path) -> None:
    from stablewalk.ui.tk.session_cache import build_demo_snapshot_with_reason

    video = tmp_path / "abnormal.mp4"
    video.write_bytes(b"not-a-real-video")
    seq = MagicMock()
    snap = _snap("abnormal")
    with (
        patch(
            "stablewalk.ui.tk.session_cache.resolve_demo_poses_path",
            return_value=None,
        ),
        patch(
            "stablewalk.ui.tk.session_cache.resolve_demo_video_path",
            return_value=str(video),
        ),
        patch(
            "stablewalk.ui.tk.session_cache.resolve_demo_frames_dir",
            return_value=None,
        ),
        patch(
            "stablewalk.pose.backends.pipeline.process_video_with_configured_backend",
            return_value=seq,
        ) as proc,
        patch(
            "stablewalk.ui.tk.session_cache.build_snapshot_from_sequence",
            return_value=snap,
        ) as build,
    ):
        out, err = build_demo_snapshot_with_reason("abnormal")
    proc.assert_called_once()
    assert build.call_args.kwargs["key"] == "abnormal"
    assert out is snap
    assert err is None


def test_export_disabled_until_both_sessions_valid() -> None:
    import tkinter as tk

    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.btn_reload_comparison = MagicMock()
    ctrl.btn_analyze_b = MagicMock()
    ctrl.btn_export_report = MagicMock()
    ctrl._sync_session_b_cta = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_reload_button()
    first = ctrl.btn_export_report.configure.call_args.kwargs.get("state")
    assert first in (tk.DISABLED, "disabled")
    cache.pin(_snap("abnormal"))
    ctrl._sync_reload_button()
    last = ctrl.btn_export_report.configure.call_args.kwargs.get("state")
    assert last in (tk.NORMAL, "normal")


def test_leave_compare_tab_does_not_cancel_session_b_analysis() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._compare_load_gen = 4
    ctrl._visual_refresh_job = "viz-job"
    ctrl.cancel_compare_load = MagicMock()  # type: ignore[method-assign]

    ctrl.on_leave_compare_tab()

    ctrl.cancel_compare_load.assert_not_called()
    ctrl.cancel_deferred_work.assert_not_called()
    assert ctrl.compare_state == CompareState.LOADING
    assert ctrl._loading_keys == {"abnormal"}
    assert ctrl._analyzing_session_b_key == "abnormal"
    assert ctrl._compare_load_gen == 4
    assert ctrl._visual_refresh_job is None
    gui.root.after_cancel.assert_called_with("viz-job")
    ctrl.release_temp_buffers.assert_called()


def test_orphan_worker_does_not_block_session_b_retry() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.FAILED
    ctrl._loading_keys = set()
    ctrl._analyzing_session_b_key = None

    class _Orphan:
        def is_alive(self) -> bool:
            return True

    ctrl._compare_worker = _Orphan()
    started: list[object] = []

    class _Thread:
        def __init__(self, *a, **k):
            started.append(self)

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    with patch("stablewalk.ui.tk.dashboard_compare.threading.Thread", _Thread):
        ctrl._request_compare_snapshots(
            ["abnormal"], allow_retry=True, explicit_analysis=True
        )

    assert len(started) == 1
    assert ctrl.compare_state == CompareState.LOADING
    assert ctrl._loading_keys == {"abnormal"}


def test_activate_while_analyzing_b_keeps_analyzing_status() -> None:
    gui = _fake_gui()
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._sync_current_session_a = MagicMock()  # type: ignore[method-assign]
    ctrl._refresh_static_compare = MagicMock()  # type: ignore[method-assign]

    ctrl.on_activate_compare_tab()

    status = str(ctrl.lbl_status.configure.call_args)
    assert "Analyzing Session B" in status
    ctrl._arm_snapshot_poll.assert_called()
    ctrl._refresh_static_compare.assert_not_called()


def test_finish_session_b_without_snapshot_never_stays_waiting() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl.compare_state = CompareState.IDLE
    ctrl._sync_session_b_cta = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_compare_ready_sections = MagicMock()  # type: ignore[method-assign]
    ctrl._queue_detailed_visual_refresh = MagicMock()  # type: ignore[method-assign]

    ctrl._finish_compare_load()

    assert ctrl._session_b_error
    assert SESSION_B_WAITING_TITLE not in (ctrl._session_b_error or "")
    assert COMPARE_SESSION_B_NOT_ANALYZED not in (ctrl._session_b_error or "")
    both_ready, message = ctrl._both_sessions_chart_ready()
    assert both_ready is False
    assert SESSION_B_WAITING_TITLE not in message
    assert ctrl.compare_state == CompareState.FAILED
    ctrl._sync_session_b_cta.assert_called()


def test_late_session_b_result_is_still_pinned() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._compare_load_gen = 2
    ctrl._queue_detailed_visual_refresh = MagicMock()  # type: ignore[method-assign]
    snap_b = _snap("abnormal")
    ctrl._snapshot_results.put((1, "abnormal", snap_b, None))
    ctrl._snapshot_results.put((1, _COMPARE_WORKER_SENTINEL, None, None))

    ctrl._drain_snapshot_results()

    assert cache.left().key == "normal"
    assert cache.right() is snap_b
    assert cache.right() is not cache.left()
    assert ctrl._session_b_error is None


def test_analyze_session_b_uses_combo_when_cache_key_empty() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = ""
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.right_key_var = SimpleNamespace(get=lambda: "abnormal")
    ctrl.open_session_picker = MagicMock()  # type: ignore[method-assign]
    calls: list[tuple[list[str], dict]] = []

    def _start(keys, **kwargs):
        calls.append((list(keys), kwargs))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = set(keys)

    ctrl._request_compare_snapshots = _start  # type: ignore[method-assign]
    ctrl.analyze_session_b()

    ctrl.open_session_picker.assert_not_called()
    assert cache.right_key == "abnormal"
    assert calls == [(["abnormal"], {"allow_retry": True, "explicit_analysis": True})]
    assert ctrl._analyzing_session_b_key == "abnormal"


def test_analyze_session_b_frees_demo_key_occupied_by_session_a() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("abnormal")
    cache.pin(snap_a)
    cache.left_key = "abnormal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=MagicMock())
    ctrl.right_key_var = SimpleNamespace(get=lambda: "abnormal", set=MagicMock())
    calls: list[tuple[list[str], dict]] = []

    def _start(keys, **kwargs):
        calls.append((list(keys), kwargs))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = set(keys)

    ctrl._request_compare_snapshots = _start  # type: ignore[method-assign]
    ctrl.analyze_session_b()

    assert cache.left_key == "current:abnormal"
    assert cache.right_key == "abnormal"
    assert cache.left() is snap_a
    assert cache.left().key == "current:abnormal"
    assert cache.get("abnormal") is None
    assert calls == [(["abnormal"], {"allow_retry": True, "explicit_analysis": True})]
    assert ctrl._analyzing_session_b_key == "abnormal"


def test_compare_worker_rebuilds_when_cached_b_is_session_a() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("abnormal")
    cache.pin(snap_a)
    cache.left_key = "abnormal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_pending_keys = ["abnormal"]
    snap_b = _snap("abnormal")
    with patch(
        "stablewalk.ui.tk.session_cache.build_demo_snapshot_with_reason",
        return_value=(snap_b, None),
    ) as build:
        ctrl._compare_worker_main(generation=0)
    build.assert_called_once_with("abnormal")
    ctrl._drain_snapshot_results()
    assert cache.get("abnormal") is snap_b
    assert cache.get("abnormal") is not snap_a
    assert cache.left() is snap_a
    assert cache.left_key == "current:abnormal"
    assert cache.right_key == "abnormal"


def test_demo_snapshot_reports_download_failure() -> None:
    from stablewalk.ui.tk.session_cache import build_demo_snapshot_with_reason

    with (
        patch(
            "stablewalk.ui.tk.session_cache.resolve_demo_poses_path",
            return_value=None,
        ),
        patch(
            "stablewalk.ui.tk.session_cache.resolve_demo_video_path",
            return_value=None,
        ),
        patch(
            "stablewalk.ui.media.demo_download.download_demo_video",
            return_value=False,
        ),
    ):
        out, err = build_demo_snapshot_with_reason("abnormal")
    assert out is None
    assert err is not None
    assert "abnormal" in err.lower()
    assert "download" in err.lower() or "video" in err.lower()


def test_abnormal_demo_path_is_not_the_normal_file(tmp_path) -> None:
    from pathlib import Path

    import stablewalk.config as cfg
    from stablewalk.ui.tk.session_cache import (
        resolve_demo_video_path,
        resolve_session_b_video_path,
    )

    (tmp_path / "normal_gait_v2.mp4").write_bytes(b"normal")
    (tmp_path / "abnormal_gait_v2.mp4").write_bytes(b"abnormal")
    (tmp_path / "athletic_walking.mp4").write_bytes(b"legacy")
    with (
        patch.object(cfg, "DEMO_VIDEOS_DIR", tmp_path),
        patch.object(cfg, "VIDEOS_DIR", tmp_path),
        patch.object(cfg, "LEGACY_VIDEOS_DIR", tmp_path),
    ):
        abnormal = resolve_demo_video_path("abnormal")
        normal = resolve_demo_video_path("normal")
        hinted = resolve_session_b_video_path(
            "abnormal", hint=str(tmp_path / "normal_gait_v2.mp4")
        )
        athletic = resolve_demo_video_path("athletic")
    assert abnormal is not None
    assert normal is not None
    assert Path(abnormal).name == "abnormal_gait_v2.mp4"
    assert Path(normal).name == "normal_gait_v2.mp4"
    assert Path(abnormal).resolve() != Path(normal).resolve()
    assert hinted is not None
    assert Path(hinted).name == "abnormal_gait_v2.mp4"
    assert athletic is None


def test_session_b_worker_rejects_session_a_video() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    snap_a.video_path = r"C:\videos\normal_gait.mp4"
    snap_a.source = snap_a.video_path
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_pending_keys = ["abnormal"]
    snap_b = _snap("abnormal")
    snap_b.video_path = snap_a.video_path
    snap_b.source = snap_a.video_path
    with patch(
        "stablewalk.ui.tk.session_cache.build_demo_snapshot_with_reason",
        return_value=(snap_b, None),
    ):
        ctrl._compare_worker_main(generation=0)
    gen, key, snap, err = ctrl._snapshot_results.get_nowait()
    assert key == "abnormal"
    assert snap is None
    assert err is not None
    assert "session a" in err.lower()
    assert cache.left() is snap_a


def test_independent_session_b_uses_main_gait_pipeline(tmp_path) -> None:
    from stablewalk.ui.tk.session_cache import run_independent_session_b_analysis

    video = tmp_path / "abnormal_gait.mp4"
    video.write_bytes(b"abnormal-bytes")
    seq = MagicMock()
    seq.frames = []
    seq.fps = 25.0
    snap = _snap("abnormal")
    snap.n_frames = 12
    snap.fps = 25.0
    snap.video_path = str(video)
    result = SimpleNamespace(
        poses_path=tmp_path / "poses.json",
        frames_dir=tmp_path / "frames",
        pose_backend_used="mediapipe",
        run_name="session_b_abnormal",
    )
    result.poses_path.write_text("{}", encoding="utf-8")
    result.frames_dir.mkdir()
    with (
        patch(
            "stablewalk.ui.tk.session_cache.resolve_session_b_video_path",
            return_value=str(video),
        ),
        patch(
            "stablewalk.core.pipeline.run_gait_pipeline",
            return_value=result,
        ) as pipeline,
        patch(
            "stablewalk.io.pose_loader.load_pose_sequence",
            return_value=seq,
        ),
        patch(
            "stablewalk.ui.tk.session_cache.build_snapshot_from_sequence",
            return_value=snap,
        ) as build,
        patch(
            "stablewalk.ui.tk.session_cache.isolate_analyzed_snapshot",
            side_effect=lambda s, key=None: s,
        ),
    ):
        out, err = run_independent_session_b_analysis(
            "abnormal",
            video_path=str(video),
            pose_backend="mediapipe",
            max_frames=120,
        )
    pipeline.assert_called_once()
    assert pipeline.call_args.args[0] == str(video)
    assert pipeline.call_args.kwargs["force_reprocess"] is True
    assert build.call_args.kwargs["key"] == "abnormal"
    assert build.call_args.kwargs["video_path"] == str(video)
    assert out is snap
    assert err is None


def test_session_b_worker_uses_independent_pipeline() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_pending_keys = ["abnormal"]
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._session_b_resolved_path = r"C:\videos\abnormal_gait.mp4"
    snap_b = _snap("abnormal")
    with (
        patch(
            "stablewalk.ui.tk.session_cache.run_independent_session_b_analysis",
            return_value=(snap_b, None),
        ) as independent,
        patch(
            "stablewalk.ui.tk.session_cache.build_demo_snapshot_with_reason",
        ) as demo,
    ):
        ctrl._compare_worker_main(generation=0)
    independent.assert_called_once()
    assert independent.call_args.args[0] == "abnormal"
    demo.assert_not_called()
    ctrl._drain_snapshot_results()
    assert cache.right() is snap_b
    assert cache.left() is snap_a
    assert cache.left() is not cache.right()


def test_analyze_session_b_surfaces_exception(tmp_path) -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.right_key_var = SimpleNamespace(get=lambda: "abnormal")
    ctrl._surface_session_b_failure = MagicMock()  # type: ignore[method-assign]
    with (
        patch(
            "stablewalk.ui.tk.session_cache.resolve_session_b_video_path",
            side_effect=RuntimeError("resolver exploded"),
        ),
        patch(
            "stablewalk.ui.tk.compare_state.compare_error_log_path",
            return_value=tmp_path / "compare_error.log",
        ),
    ):
        ctrl.analyze_session_b()
    ctrl._surface_session_b_failure.assert_called_once()
    text = str(ctrl._surface_session_b_failure.call_args.args[0])
    assert "resolver exploded" in text
    assert (tmp_path / "compare_error.log").is_file()


def test_drain_saves_dedicated_compare_session_b() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    seq_a = object()
    snap_a.sequence = seq_a  # type: ignore[assignment]
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    gui.sequence = seq_a
    ctrl = _bare_controller(gui, static=False)
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    snap_b = _snap("abnormal")
    snap_b.sequence = object()  # type: ignore[assignment]
    snap_b.metrics = CompareMetrics(label="Abnormal", cadence_spm=110.0)
    ctrl._snapshot_results.put((0, "abnormal", snap_b, None))
    ctrl._snapshot_results.put((0, _COMPARE_WORKER_SENTINEL, None, None))

    ctrl._drain_snapshot_results()

    assert cache.left() is snap_a
    assert cache.left().sequence is seq_a
    assert gui.sequence is seq_a
    assert ctrl.compare_session_b is not None
    assert ctrl.compare_session_b is not snap_a
    assert ctrl.compare_session_b is not cache.left()
    assert int(ctrl.compare_session_b.n_frames) > 0
    assert cache.right() is not None
    assert cache.right() is not snap_a
    left, right = ctrl._session_pair()
    assert left is snap_a
    assert right is ctrl.compare_session_b
    ready, message = ctrl._both_sessions_chart_ready()
    assert ready is True
    assert SESSION_B_WAITING_TITLE not in message
    assert ctrl._session_b_needs_cta() is False


def test_analyze_session_b_logs_handler_and_resolves_abnormal(
    tmp_path, caplog
) -> None:
    import logging

    import stablewalk.config as cfg

    video = tmp_path / "abnormal_gait_v2.mp4"
    video.write_bytes(b"x")
    (tmp_path / "normal_gait_v2.mp4").write_bytes(b"n")
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    snap_a.video_path = str(tmp_path / "normal_gait_v2.mp4")
    snap_a.source = snap_a.video_path
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.right_key_var = SimpleNamespace(get=lambda: "abnormal")
    gui._load_video = MagicMock()
    gui._run_full_analysis = MagicMock()
    gui._get_input_source = MagicMock(return_value=str(tmp_path / "normal_gait_v2.mp4"))
    calls: list[tuple[list[str], dict]] = []

    def _start(keys, **kwargs):
        calls.append((list(keys), kwargs))
        ctrl.compare_state = CompareState.LOADING
        ctrl._loading_keys = set(keys)

    ctrl._request_compare_snapshots = _start  # type: ignore[method-assign]
    with (
        patch.object(cfg, "DEMO_VIDEOS_DIR", tmp_path),
        patch.object(cfg, "VIDEOS_DIR", tmp_path),
        patch.object(cfg, "LEGACY_VIDEOS_DIR", tmp_path),
        caplog.at_level(logging.INFO),
    ):
        ctrl.analyze_session_b()
    text = caplog.text
    assert "[COMPARE] Analyze Session B clicked" in text
    assert "[COMPARE] Session A:" in text
    assert "[COMPARE] Session B:" in text
    assert "key=abnormal" in text
    assert "[COMPARE] Resolved Session B path=" in text
    assert "abnormal_gait_v2.mp4" in text
    assert "exists=True" in text
    assert "[COMPARE] Starting independent Session B analysis" in text
    resolved_lines = [
        line for line in text.splitlines() if "[COMPARE] Resolved Session B path=" in line
    ]
    assert resolved_lines
    assert all("abnormal_gait_v2.mp4" in line for line in resolved_lines)
    assert calls == [(["abnormal"], {"allow_retry": True, "explicit_analysis": True})]
    assert cache.left().key == "normal"
    gui._load_video.assert_not_called()
    gui._run_full_analysis.assert_not_called()
    gui._get_input_source.assert_not_called()


def test_analyze_session_b_rejects_same_recording_as_session_a(tmp_path) -> None:
    video = tmp_path / "normal_gait.mp4"
    video.write_bytes(b"same")
    snap_a = AnalyzedSessionSnapshot(
        key="normal",
        label="Normal",
        n_frames=10,
        video_path=str(video),
        source=str(video),
        skeletons={0: object()},
        metrics=CompareMetrics(label="Normal", cadence_spm=80.0),
    )
    cache = SessionCompareCache()
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = f"video:{video}"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.right_key_var = SimpleNamespace(get=lambda: cache.right_key)
    ctrl._request_compare_snapshots = MagicMock()  # type: ignore[method-assign]
    ctrl.analyze_session_b()
    ctrl._request_compare_snapshots.assert_not_called()
    assert ctrl._session_b_error is not None
    assert "same recording" in ctrl._session_b_error.lower()
    assert cache.left() is snap_a


def test_missing_session_b_path_shows_error_not_waiting() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "video:C:/missing/nope.mp4"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.right_key_var = SimpleNamespace(get=lambda: cache.right_key)
    ctrl._request_compare_snapshots = MagicMock()  # type: ignore[method-assign]
    ctrl.analyze_session_b()
    ctrl._request_compare_snapshots.assert_not_called()
    assert ctrl._session_b_error is not None
    assert "Session B analysis failed" in ctrl._session_b_error
    assert "path_exists=False" in ctrl._session_b_error
    assert SESSION_B_WAITING_TITLE not in (ctrl._session_b_error or "")
    ready, message = ctrl._both_sessions_chart_ready()
    assert ready is False
    assert SESSION_B_WAITING_TITLE not in message


def test_load_demo_auto_analyzes_incomplete_cached_b() -> None:
    cache = SessionCompareCache()
    cache.pin(_demo_ready_snap("normal"))
    cache.pin(
        AnalyzedSessionSnapshot(key="abnormal", label="Abnormal", n_frames=0)
    )
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=lambda k: None)
    ctrl.right_key_var = SimpleNamespace(set=lambda k: None)
    ctrl._request_compare_snapshots = MagicMock()  # type: ignore[method-assign]
    ctrl.load_demo_comparison()
    ctrl._request_compare_snapshots.assert_called_once_with(
        ["abnormal"], allow_retry=True, explicit_analysis=True
    )
    assert ctrl._analyzing_session_b_key == "abnormal"
    assert ctrl._demo_compare_load_pending is True


def test_load_demo_rebuilds_knee_rom_only_cached_sessions() -> None:
    cache = SessionCompareCache()
    cache.pin(
        AnalyzedSessionSnapshot(
            key="normal",
            label="Normal",
            n_frames=120,
            metrics=CompareMetrics(label="Normal", knee_rom_deg=8.0),
        )
    )
    cache.pin(
        AnalyzedSessionSnapshot(
            key="abnormal",
            label="Abnormal",
            n_frames=120,
            metrics=CompareMetrics(label="Abnormal", knee_rom_deg=12.0),
        )
    )
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=lambda k: None)
    ctrl.right_key_var = SimpleNamespace(set=lambda k: None)
    ctrl._request_compare_snapshots = MagicMock()  # type: ignore[method-assign]
    ctrl.load_demo_comparison()
    ctrl._request_compare_snapshots.assert_called_once_with(
        ["normal", "abnormal"], allow_retry=True, explicit_analysis=True
    )
    assert ctrl._demo_compare_load_pending is True
    assert ctrl.compare_state != CompareState.READY
    status = ctrl.lbl_status.configure.call_args.kwargs["text"]
    assert status != "Demo comparison: Normal vs Abnormal"


def test_knee_rom_only_snapshot_is_not_demo_compare_ready() -> None:
    snap = AnalyzedSessionSnapshot(
        key="abnormal",
        label="Abnormal",
        n_frames=120,
        metrics=CompareMetrics(label="Abnormal", knee_rom_deg=12.0),
    )
    assert demo_snapshot_is_compare_ready(snap) is False
    ready = _demo_ready_snap("abnormal")
    assert demo_snapshot_is_compare_ready(ready) is True


def test_pin_current_as_b_saves_dedicated_session_and_refreshes() -> None:
    cache = SessionCompareCache()
    snap_a = _snap("normal")
    snap_b = _snap("current_b")
    cache.pin(snap_a)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(set=MagicMock())
    ctrl.right_key_var = SimpleNamespace(set=MagicMock())
    ctrl.request_refresh = MagicMock()  # type: ignore[method-assign]
    with patch(
        "stablewalk.ui.tk.session_cache.pin_current_session",
        return_value=snap_b,
    ):
        ctrl.pin_current(slot="right")
    assert cache.right_key == "current_b"
    assert ctrl.compare_session_b is not None
    assert ctrl.compare_session_b is not snap_a
    assert cache.left() is snap_a
    ctrl.request_refresh.assert_called()


def test_slot_change_clears_stale_compare_outputs() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.pin(_snap("abnormal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.left_key_var = SimpleNamespace(get=lambda: "normal", set=MagicMock())
    ctrl.right_key_var = SimpleNamespace(get=lambda: "athletic", set=MagicMock())
    ctrl.compare_session_b = _snap("abnormal")
    ctrl._cached_comparison = object()
    ctrl.request_refresh = MagicMock()  # type: ignore[method-assign]
    ctrl._on_slot_change(changed_slot="right")
    assert ctrl.compare_session_b is None
    assert ctrl._cached_comparison is None
    ctrl.request_refresh.assert_called()


def test_load_demo_worker_uses_pipeline_for_abnormal() -> None:
    cache = SessionCompareCache()
    cache.pin(_snap("normal"))
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"abnormal"}
    ctrl._compare_pending_keys = ["abnormal"]
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._demo_compare_load_pending = True
    snap_b = _snap("abnormal")
    with (
        patch(
            "stablewalk.ui.tk.session_cache.load_or_analyze_demo_session",
            return_value=(snap_b, None),
        ) as loader,
        patch(
            "stablewalk.ui.tk.session_cache.build_demo_snapshot_with_reason",
        ) as demo,
    ):
        ctrl._compare_worker_main(generation=0)
    loader.assert_called_once()
    demo.assert_not_called()
    ctrl._drain_snapshot_results()
    assert ctrl.compare_session_b is not None
    assert cache.left() is not cache.right()
    ready, message = ctrl._both_sessions_chart_ready()
    assert ready is True
    assert SESSION_B_WAITING_TITLE not in message


def test_load_demo_worker_does_not_reject_rebuilt_session_a() -> None:
    """Rebuilding Normal from normal_gait.mp4 is not a Session B duplicate."""
    cache = SessionCompareCache()
    stale = AnalyzedSessionSnapshot(
        key="normal",
        label="Normal",
        n_frames=120,
        video_path="C:/demo/normal_gait.mp4",
        metrics=CompareMetrics(label="Normal", knee_rom_deg=8.0),
    )
    cache.pin(stale)
    cache.left_key = "normal"
    cache.right_key = "abnormal"
    gui = _fake_gui(cache)
    ctrl = _bare_controller(gui, static=False)
    ctrl.compare_state = CompareState.LOADING
    ctrl._loading_keys = {"normal"}
    ctrl._compare_pending_keys = ["normal"]
    ctrl._analyzing_session_b_key = "abnormal"
    ctrl._demo_compare_load_pending = True
    rebuilt = _demo_ready_snap("normal")
    rebuilt.video_path = stale.video_path
    with patch(
        "stablewalk.ui.tk.session_cache.load_or_analyze_demo_session",
        return_value=(rebuilt, None),
    ) as loader:
        ctrl._compare_worker_main(generation=0)
    loader.assert_called_once()
    ctrl._drain_snapshot_results()
    assert "normal" not in ctrl._failed_snapshot_keys
    assert cache.get("normal") is not None
    assert cache.get("normal").n_frames == 10

