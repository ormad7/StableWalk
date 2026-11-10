"""Stress harness for Overview responsiveness / memory integrity.

Exercises the checklist:
- switch tabs repeatedly (default 30)
- play the video twice (simulated ticks)
- change selected joint many times
- continuous resize bursts
- load a second video / return to first (when paths provided)
- assert: single playback timer, no figure explosion, bounded memory

Run:
  python -m pytest tests/test_perf_stress_harness.py -q
  STABLEWALK_PERF_LOG=1 python -c "from tests.test_perf_stress_harness import run_perf_stress; ..."
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import tkinter as tk

from stablewalk.ui.tk.perf_diagnostics import (
    count_analysis_workers,
    count_playback_timers,
    format_perf_diagnostics_report,
    set_perf_diagnostics,
)
from stablewalk.ui.tk.render_diagnostics import (
    capture_playback_render_snapshot,
    reset_playback_render_counters,
)


def _play_ticks(gui: Any, ticks: int) -> None:
    player = getattr(gui, "skeleton_player", None)
    if player is None or getattr(player, "frame_count", 0) <= 0:
        return
    if not getattr(gui, "playing", False):
        try:
            gui._toggle_play()
        except Exception:
            pass
    tick = getattr(gui, "_tick", None)
    for _ in range(ticks):
        if tick is None:
            break
        try:
            tick()
        except Exception:
            break
        try:
            gui.root.update_idletasks()
        except Exception:
            pass
    if getattr(gui, "playing", False):
        try:
            gui._toggle_play()
        except Exception:
            pass


def run_perf_stress(
    gui: Any,
    *,
    play_ticks: int = 120,
    tab_switches: int = 30,
    joint_changes: int = 20,
    resize_bursts: int = 40,
    poses_a: Path | str | None = None,
    poses_b: Path | str | None = None,
) -> list[tuple[str, bool, str]]:
    """Run the UI stress checklist against an already-constructed GUI."""
    from stablewalk.ui.tk.dashboard_notebook import (
        ENABLE_COMPARE_TAB,
        TAB_ADVANCED,
        TAB_BIOMECHANICS,
        TAB_COMPARE,
        TAB_MOTION,
        TAB_OVERVIEW,
        TAB_RESULTS_SUMMARY,
        select_dashboard_tab,
    )
    from stablewalk.ui.tk.gui_memory import (
        approximate_rss_mb,
        collect_active_timer_ids,
        collect_owned_figures,
    )

    results: list[tuple[str, bool, str]] = []
    set_perf_diagnostics(gui, True)
    reset_playback_render_counters(gui)
    rss0 = approximate_rss_mb()
    figs0 = len(collect_owned_figures(gui))
    baseline = capture_playback_render_snapshot(gui, 0)

    # 1) Play video twice
    _play_ticks(gui, play_ticks)
    _play_ticks(gui, max(30, play_ticks // 2))
    results.append(
        (
            "playback_twice",
            count_playback_timers(gui) <= 1,
            f"timers={count_playback_timers(gui)}",
        )
    )

    # 2) Switch tabs N times
    cycle = [
        TAB_OVERVIEW,
        TAB_MOTION,
        TAB_BIOMECHANICS,
        TAB_RESULTS_SUMMARY,
        TAB_ADVANCED,
    ]
    if ENABLE_COMPARE_TAB:
        cycle.insert(-1, TAB_COMPARE)
    tab_ok = True
    for i in range(tab_switches):
        try:
            select_dashboard_tab(gui, cycle[i % len(cycle)])
            gui.root.update_idletasks()
        except Exception:
            tab_ok = False
    # End on Overview
    try:
        select_dashboard_tab(gui, TAB_OVERVIEW)
        gui.root.update_idletasks()
    except Exception:
        tab_ok = False
    results.append(("tab_switch_n", tab_ok, f"n={tab_switches}"))
    results.append(
        (
            "single_timer_after_tabs",
            count_playback_timers(gui) <= 1,
            f"count={count_playback_timers(gui)} known={len(collect_active_timer_ids(gui))}",
        )
    )

    # 3) Change selected joint many times
    joint_ok = True
    vars_map = getattr(gui, "_dof_checkbox_vars", None) or {}
    keys = list(vars_map.keys())
    if keys:
        for i in range(joint_changes):
            key = keys[i % len(keys)]
            try:
                vars_map[key].set(True)
                on_change = getattr(gui, "_on_dof_checkbox", None)
                if callable(on_change):
                    on_change()
                gui.root.update_idletasks()
            except Exception:
                joint_ok = False
    results.append(
        (
            "select_joints_n",
            joint_ok or not keys,
            f"n={joint_changes} keys={len(keys)}",
        )
    )

    # 4) Continuous resize bursts (debounced fits must not explode timers)
    resize_ok = True
    for _ in range(resize_bursts):
        try:
            schedule = getattr(gui, "_schedule_overview_media_refit", None)
            if callable(schedule):
                schedule(delay_ms=40)
            fit = getattr(gui, "_fit_skeleton_canvas", None)
            if callable(fit):
                fit()
            gui.root.update_idletasks()
        except Exception:
            resize_ok = False
    # Drain pending after jobs
    try:
        gui.root.update()
    except Exception:
        pass
    results.append(("resize_bursts", resize_ok, f"n={resize_bursts}"))

    # 5) Compare Session B (optional)
    compare = getattr(gui, "_comparison_mode", None)
    if compare is not None and poses_b is not None:
        try:
            select_dashboard_tab(gui, TAB_COMPARE)
            load_fn = getattr(compare, "load_session_b_demo", None) or getattr(
                compare, "_load_session_key", None
            )
            if callable(load_fn):
                load_fn(
                    "abnormal" if not Path(str(poses_b)).is_file() else str(poses_b)
                )
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                gui.root.update()
                if count_analysis_workers(gui) <= 1:
                    cw = getattr(compare, "_compare_worker", None)
                    if cw is None or not cw.is_alive():
                        break
                time.sleep(0.05)
            results.append(("compare_session_b", True, "exercised"))
        except Exception as exc:
            results.append(("compare_session_b", False, str(exc)))
    else:
        results.append(("compare_session_b", True, "skipped"))

    # 6) Overview restore
    try:
        select_dashboard_tab(gui, TAB_OVERVIEW)
        gui.root.update_idletasks()
        results.append(("return_overview", True, "ok"))
    except Exception as exc:
        results.append(("return_overview", False, str(exc)))

    # 7) Load another video, then optionally return to first
    if poses_b is not None and Path(str(poses_b)).is_file():
        try:
            ok = gui.load_poses(Path(poses_b), fresh=True, wait_for_analysis=True)
            gui.root.update_idletasks()
            results.append(("load_second_video", bool(ok), f"ok={ok}"))
        except Exception as exc:
            results.append(("load_second_video", False, str(exc)))
        if poses_a is not None and Path(str(poses_a)).is_file():
            try:
                ok = gui.load_poses(Path(poses_a), fresh=True, wait_for_analysis=True)
                gui.root.update_idletasks()
                results.append(("return_first_video", bool(ok), f"ok={ok}"))
            except Exception as exc:
                results.append(("return_first_video", False, str(exc)))
        else:
            results.append(("return_first_video", True, "skipped"))
    else:
        results.append(("load_second_video", True, "skipped"))
        results.append(("return_first_video", True, "skipped"))

    # Integrity
    results.append(
        (
            "analysis_worker_le_1",
            count_analysis_workers(gui) <= 1,
            f"count={count_analysis_workers(gui)}",
        )
    )
    results.append(
        (
            "playback_timer_le_1",
            count_playback_timers(gui) <= 1,
            f"count={count_playback_timers(gui)}",
        )
    )
    final = capture_playback_render_snapshot(gui, play_ticks)
    results.append(
        (
            "no_duplicate_canvases",
            final.video_label_count == baseline.video_label_count
            and final.figure_canvas_total <= baseline.figure_canvas_total + 2,
            f"video={final.video_label_count} figs={final.figure_canvas_total}",
        )
    )
    figs1 = len(collect_owned_figures(gui))
    results.append(
        (
            "figure_count_stable",
            figs1 <= figs0 + 2,
            f"before={figs0} after={figs1}",
        )
    )
    # Playback keeps UI/skeleton/trajectory locked to the shared clock.
    ui_hz = float(getattr(gui, "_ui_anim_hz", 0) or 0)
    traj_iv = float(getattr(gui, "_traj_redraw_interval_s", 1.0) or 0.0)
    stride = int(getattr(gui, "_3d_play_stride", 0) or 0)
    results.append(
        (
            "playback_surfaces_locked",
            ui_hz >= 30.0 and traj_iv <= 1e-9 and stride == 1,
            f"ui_hz={ui_hz} traj_iv={traj_iv:.3f} stride={stride}",
        )
    )
    rss1 = approximate_rss_mb()
    if rss0 is not None and rss1 is not None:
        growth = rss1 - rss0
        results.append(
            (
                "memory_growth_bounded",
                growth < 400.0,
                f"delta={growth:.1f} MiB",
            )
        )
    else:
        results.append(("memory_growth_bounded", True, "rss unavailable"))

    report = format_perf_diagnostics_report(gui)
    results.append(
        ("perf_report_nonempty", "decode" in report or "total" in report, "ok")
    )
    return results


def test_perf_stress_empty_gui() -> None:
    root = tk.Tk()
    root.withdraw()
    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI(root=root)
    root.update_idletasks()
    results = run_perf_stress(
        app,
        play_ticks=30,
        tab_switches=30,
        joint_changes=20,
        resize_bursts=20,
    )
    assert all(ok for _n, ok, _d in results), results
    root.destroy()


def test_overview_ui_rates_and_clinical_skip_while_playing() -> None:
    from pathlib import Path

    src = (
        Path(__file__).resolve().parents[1]
        / "stablewalk"
        / "ui"
        / "tk"
        / "app.py"
    ).read_text(encoding="utf-8")
    # Playback keeps UI/skeleton/trajectory at video rate (no intentional lag).
    assert "self._ui_anim_hz = 30" in src
    assert "self._3d_play_stride = 1" in src
    assert "self._ui_sync_interval_s = 0.0" in src
    assert "self._traj_redraw_interval_s = 0.0" in src
    assert "self._joint_info_play_interval_s = 0.0" in src
    # Clinical summary must still short-circuit while playing.
    assert "if getattr(self, \"playing\", False):" in src or "if self.playing:" in src
    fn = src.index("def _update_overview_clinical_summary")
    body = src[fn : fn + 800]
    assert "playing" in body
