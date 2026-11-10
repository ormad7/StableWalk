"""Regression: opening and leaving the Advanced notebook tab."""

from __future__ import annotations

import time
import traceback
import tkinter as tk
from pathlib import Path

import pytest

from stablewalk.ui.tk.dashboard_notebook import (
    ENABLE_COMPARE_TAB,
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    dashboard_tab_widget,
    is_advanced_tab_selected,
    select_dashboard_tab,
    visible_dashboard_tab_labels,
)


def _pump(root: tk.Misc, seconds: float = 0.4) -> None:
    end = time.time() + seconds
    while time.time() < end:
        try:
            root.update()
        except tk.TclError:
            return
        time.sleep(0.02)


def _user_click(app, tab: str) -> None:
    page = dashboard_tab_widget(app, tab)
    assert page is not None
    app._dashboard_notebook.select(page)
    try:
        app.root.update()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")


def _walk_to_canvas(widget) -> bool:
    current = widget
    while current is not None:
        if isinstance(current, tk.Canvas):
            return True
        current = getattr(current, "master", None)
    return False


def _advanced_shell_ok(app) -> None:
    assert is_advanced_tab_selected(app)
    assert getattr(app, "_advanced_content_built", False) is True
    assert getattr(app, "_tab_advanced_scroll_canvas", None) is None
    assert getattr(app, "_tab_advanced_scroll_window_id", None) is None
    content = getattr(app, "_tab_advanced_content", None)
    workspace = getattr(app, "_advanced_workspace", None)
    viewport = getattr(app, "_tab_advanced_scroll_viewport", None)
    assert content is not None
    assert workspace is not None
    assert viewport is not None
    assert _walk_to_canvas(content) is False
    assert _walk_to_canvas(workspace) is False
    assert bool(content.winfo_ismapped())
    assert bool(content.winfo_viewable())
    assert int(content.winfo_width()) >= 32
    assert int(content.winfo_height()) >= 32
    assert bool(workspace.winfo_ismapped())
    assert bool(workspace.winfo_viewable())
    assert int(workspace.winfo_width()) >= 32
    assert int(workspace.winfo_height()) >= 32
    assert int(viewport.winfo_width()) >= 32
    assert int(viewport.winfo_height()) >= 32
    hosts = getattr(app, "_advanced_category_hosts", {}) or {}
    assert "session" in hosts
    assert "opensim" in hosts
    assert "exports" in hosts
    assert "reports" in hosts
    assert bool(hosts["session"].winfo_ismapped())
    assert bool(hosts["opensim"].winfo_ismapped())
    titles = getattr(app, "_advanced_title_labels", {}) or {}
    session_title = titles.get("session")
    if session_title is not None:
        assert "Session" in str(session_title.cget("text") or "")
        assert bool(session_title.winfo_viewable())
    veil = getattr(app, "_tab_motion_veil", None)
    if veil is not None:
        try:
            if veil.winfo_exists():
                assert str(veil.winfo_manager() or "") != "place"
        except tk.TclError:
            pass


@pytest.fixture(scope="module")
def app():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        from stablewalk.ui.tk.app import StableWalkGUI

        gui = StableWalkGUI(root=root)
        root.geometry("1280x800")
        root.deiconify()
        root.update_idletasks()
    except tk.TclError as exc:
        try:
            root.destroy()
        except tk.TclError:
            pass
        pytest.skip(f"Tk unavailable: {exc}")
    yield gui
    try:
        root.destroy()
    except tk.TclError:
        pass


def test_advanced_user_click_opens_workspace(app) -> None:
    _user_click(app, TAB_OVERVIEW)
    _pump(app.root, 0.15)
    _user_click(app, TAB_ADVANCED)
    _pump(app.root)
    _advanced_shell_ok(app)
    empty = getattr(app, "_advanced_empty_state", None)
    if empty is not None:
        text = str(empty.cget("text") or "")
        assert "failed to load" not in text.lower()


def test_advanced_open_close_roundtrip_preserves_other_tabs(app) -> None:
    tabs = list(visible_dashboard_tab_labels())
    assert TAB_ADVANCED in tabs
    for _ in range(2):
        for tab in tabs:
            _user_click(app, tab)
            _pump(app.root, 0.15)
        _user_click(app, TAB_ADVANCED)
        _pump(app.root, 0.3)
        _advanced_shell_ok(app)
    _user_click(app, TAB_OVERVIEW)
    _pump(app.root, 0.2)
    assert app._dashboard_notebook.nametowidget(
        app._dashboard_notebook.select()
    ) is app._tab_overview
    assert getattr(app, "video_frame", None) is not None
    assert getattr(app, "skel_frame", None) is not None
    _user_click(app, TAB_MOTION)
    _pump(app.root, 0.15)
    _user_click(app, TAB_BIOMECHANICS)
    _pump(app.root, 0.15)
    _user_click(app, TAB_RESULTS_SUMMARY)
    _pump(app.root, 0.15)
    if ENABLE_COMPARE_TAB:
        _user_click(app, TAB_COMPARE)
        _pump(app.root, 0.15)
    _user_click(app, TAB_OVERVIEW)
    _pump(app.root, 0.15)
    assert getattr(app, "canvas_3d", None) is not None or getattr(app, "fig_3d", None) is not None


def test_advanced_leftover_skip_flag_still_realizes(app) -> None:
    _user_click(app, TAB_OVERVIEW)
    _pump(app.root, 0.15)
    app._skip_next_notebook_tab_changed = True
    _user_click(app, TAB_ADVANCED)
    _pump(app.root)
    _advanced_shell_ok(app)


def test_advanced_host_is_not_a_page_canvas() -> None:
    from pathlib import Path

    source = Path("stablewalk/ui/tk/dashboard_notebook.py").read_text(encoding="utf-8")
    start = source.index("def _install_tab_scroll")
    end = source.index("def _schedule_tab_reflow")
    host_fn = source[start:end]
    assert "canvas.create_window" not in host_fn
    assert "tk.Canvas(" not in host_fn
    assert "no Canvas.create_window" in host_fn


def test_advanced_select_dashboard_tab_opens(app) -> None:
    select_dashboard_tab(app, TAB_ADVANCED)
    _pump(app.root)
    _advanced_shell_ok(app)
    select_dashboard_tab(app, TAB_OVERVIEW)
    _pump(app.root, 0.15)
    select_dashboard_tab(app, TAB_ADVANCED)
    _pump(app.root, 0.3)
    _advanced_shell_ok(app)


def _capture_callback_exceptions(app) -> list[str]:
    captured: list[str] = []

    def _hook(exc, val, tb) -> None:
        captured.append("".join(traceback.format_exception(exc, val, tb)))

    app.root.report_callback_exception = _hook
    return captured


def _load_normal_gait_if_present(app) -> bool:
    poses = Path("data/output/poses/normal_gait_poses.json")
    if not poses.is_file():
        return False
    try:
        app.load_poses(poses, fresh=True, wait_for_analysis=True)
    except TypeError:
        app.load_poses(poses, fresh=True)
    _pump(app.root, 0.8)
    return True


def test_advanced_analysis_button_formats_after_session(app) -> None:
    captured = _capture_callback_exceptions(app)
    loaded = _load_normal_gait_if_present(app)
    _user_click(app, TAB_ADVANCED)
    _pump(app.root, 0.4)
    _advanced_shell_ok(app)
    biomech = getattr(app, "_biomech", None)
    if not loaded or biomech is None:
        pytest.skip("normal_gait analysis not available")
    text = app._format_advanced_analysis_details(biomech)
    assert "Advanced Gait Analysis" in text
    btn = getattr(app, "btn_advanced_analysis", None)
    assert btn is not None
    btn.invoke()
    _pump(app.root, 0.3)
    dlg = getattr(app, "_advanced_analysis_dialog", None)
    assert dlg is not None
    try:
        dlg.destroy()
    except tk.TclError:
        pass
    app._advanced_analysis_dialog = None
    assert captured == []


def test_advanced_repeated_navigation_during_playback(app) -> None:
    captured = _capture_callback_exceptions(app)
    _load_normal_gait_if_present(app)
    _user_click(app, TAB_OVERVIEW)
    _pump(app.root, 0.15)
    tabs = [
        TAB_OVERVIEW,
        TAB_ADVANCED,
        TAB_MOTION,
        TAB_ADVANCED,
        TAB_BIOMECHANICS,
        TAB_ADVANCED,
        TAB_RESULTS_SUMMARY,
        TAB_ADVANCED,
    ]
    app.playing = True
    try:
        for _ in range(3):
            for tab in tabs:
                _user_click(app, tab)
                _pump(app.root, 0.08)
                if tab == TAB_ADVANCED:
                    _advanced_shell_ok(app)
                    try:
                        app._refresh_active_tab_charts()
                    except Exception as exc:
                        pytest.fail(f"_refresh_active_tab_charts during playback: {exc}")
                    try:
                        app._sync_all_to_frame(force_draw=False)
                    except Exception as exc:
                        pytest.fail(f"_sync_all_to_frame during Advanced playback: {exc}")
                    refresh = getattr(app, "_refresh_pipeline_status_panel", None)
                    if callable(refresh):
                        refresh(force=False)
                    notify = getattr(app, "_notify_generated_files_changed", None)
                    if callable(notify):
                        notify(force=False)
    finally:
        app.playing = False
    _user_click(app, TAB_OVERVIEW)
    _pump(app.root, 0.15)
    assert getattr(app, "video_frame", None) is not None
    assert getattr(app, "canvas_3d", None) is not None or getattr(app, "fig_3d", None) is not None
    assert captured == []


def test_advanced_updates_while_playback_is_active(app) -> None:
    captured = _capture_callback_exceptions(app)
    loaded = _load_normal_gait_if_present(app)
    _user_click(app, TAB_ADVANCED)
    _pump(app.root, 0.3)
    _advanced_shell_ok(app)
    app.playing = True
    try:
        for _ in range(6):
            try:
                app._refresh_active_tab_charts()
                app._sync_all_to_frame(force_draw=False)
                app._refresh_pipeline_status_panel(force=False)
                app._notify_generated_files_changed(force=False)
                if hasattr(app, "_refresh_opensim_status"):
                    app._refresh_opensim_status()
            except Exception as exc:
                pytest.fail(f"Advanced update during playback failed: {exc}")
            _pump(app.root, 0.05)
        btn = getattr(app, "btn_generated_files_refresh", None)
        if btn is not None:
            btn.invoke()
            _pump(app.root, 0.15)
    finally:
        app.playing = False
    _user_click(app, TAB_MOTION)
    _pump(app.root, 0.15)
    _user_click(app, TAB_ADVANCED)
    _pump(app.root, 0.2)
    _advanced_shell_ok(app)
    if loaded and getattr(app, "_biomech", None) is not None:
        app._format_advanced_analysis_details(app._biomech)
    assert captured == []
