"""Regression tests for UI performance caches and deferred tab paints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stablewalk.analysis.gait_feature_analysis import (
    clear_opensim_mot_timeseries_cache,
    read_opensim_mot_timeseries,
)


def _write_minimal_mot(path: Path, *, n: int = 4) -> None:
    lines = [
        "Coordinates",
        "version=1",
        "nRows=%d" % n,
        "nColumns=3",
        "inDegrees=yes",
        "endheader",
        "time knee_angle_r knee_angle_l",
    ]
    for i in range(n):
        t = i * 0.033
        lines.append(f"{t:.6f} {10.0 + i} {12.0 + i}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_metric_ribbon_aggregates_are_cached() -> None:
    from stablewalk.ui.viewers.joint_motion_analysis_chart import (
        JointMotionBundle,
        JointMotionSeries,
        clear_metric_ribbon_stats_cache,
        compute_metric_ribbon_stats,
    )

    clear_metric_ribbon_stats_cache()
    n = 20
    times = np.linspace(0.0, 1.0, n)
    angle = np.linspace(10.0, 40.0, n)
    zeros = np.zeros(n)
    series = JointMotionSeries(
        item_id="right_knee",
        label="Right Knee",
        color="#336699",
        times_s=times,
        frames=np.arange(n),
        angle_deg=angle,
        omega_deg_s=np.gradient(angle, times),
        alpha_deg_s2=zeros.copy(),
        speed_m_s=zeros.copy(),
        accel_m_s2=zeros.copy(),
        jerk_m_s3=zeros.copy(),
        angular_jerk_deg_s3=zeros.copy(),
        x_m=zeros.copy(),
        y_m=zeros.copy(),
        z_m=zeros.copy(),
    )
    bundle = JointMotionBundle(series=(series,), times_s=times)
    a = compute_metric_ribbon_stats(bundle, "angle", time_s=0.25, unit="°")
    b = compute_metric_ribbon_stats(bundle, "angle", time_s=0.75, unit="°")
    assert a.minimum == b.minimum
    assert a.maximum == b.maximum
    assert a.mean == b.mean
    assert a.rom == b.rom
    assert a.std == b.std
    assert a.current != b.current


def test_series_stats_cache_hits() -> None:
    from stablewalk.ui.viewers.biomech_chart_enrichment import (
        clear_series_stats_cache,
        compute_series_stats,
    )

    clear_series_stats_cache()
    values = np.linspace(1.0, 5.0, 16)
    first = compute_series_stats(values, unit="m")
    second = compute_series_stats(values, unit="m")
    assert first is second


def test_trajectory_path_cache_reuses_geometry_on_scrub_rewind() -> None:
    """Scrub rewind must not rebuild path points already cached at high-water."""
    from types import SimpleNamespace

    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _joint_path_with_times,
        clear_trajectory_path_cache,
    )

    class _Snap:
        def __init__(self, i: int) -> None:
            self.time_s = i * 0.033
            self.joints = {
                "j": SimpleNamespace(position=(float(i), 0.0, 0.0)),
            }

    class _Rec:
        frame_count = 12

        def snapshot_at(self, i: int):
            return _Snap(i)

    clear_trajectory_path_cache()
    rec = _Rec()
    forward = _joint_path_with_times(rec, "j", 9.0, coord_mode="world")
    rewind = _joint_path_with_times(rec, "j", 3.0, coord_mode="world")
    assert len(forward) == 10
    assert len(rewind) == 4
    assert rewind[0][0][0] == 0.0
    assert rewind[-1][0][0] == 3.0
    # High-water cache still serves a later forward scrub without full rebuild.
    again = _joint_path_with_times(rec, "j", 9.0, coord_mode="world")
    assert again is forward


def test_enforce_rss_soft_budget_safe_when_under_limit() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.gui_memory import enforce_rss_soft_budget

    gui = SimpleNamespace(_skeleton_cache={}, _rgb_cache=None, _landmark_overlay_cache={})
    result = enforce_rss_soft_budget(gui, budget_mb=10_000.0)
    assert result["trimmed"] is False
    assert result["budget_mb"] == 10_000.0


def test_opensim_mot_timeseries_cache_hits(tmp_path: Path) -> None:
    clear_opensim_mot_timeseries_cache()
    mot = tmp_path / "session.mot"
    _write_minimal_mot(mot)
    first = read_opensim_mot_timeseries(mot)
    second = read_opensim_mot_timeseries(mot)
    assert first is not None
    assert second is first
    assert "time" in first
    assert len(first["time"]) == 4


def test_opensim_mot_cache_invalidates_on_rewrite(tmp_path: Path) -> None:
    clear_opensim_mot_timeseries_cache()
    mot = tmp_path / "session.mot"
    _write_minimal_mot(mot, n=4)
    first = read_opensim_mot_timeseries(mot)
    assert first is not None
    _write_minimal_mot(mot, n=6)
    second = read_opensim_mot_timeseries(mot)
    assert second is not None
    assert second is not first
    assert len(second["time"]) == 6


def test_biomech_panel_lite_path_skips_session_rebuild() -> None:
    """Unchanged analysis object must not rebuild SessionMetricView every frame."""
    from types import SimpleNamespace
    from unittest.mock import patch

    from stablewalk.ui.tk.app import StableWalkGUI

    analysis = SimpleNamespace(stability_margin=None)
    gui = SimpleNamespace(
        playing=True,
        _biomech_analysis=analysis,
        _biomech_static_sig=(id(analysis), id(None), id(None), ()),
        _biomech=SimpleNamespace(view_type="lateral"),
        _biomech_kpi_cards={},
        _biomech_frame_cache=None,
        _gait_cycle=None,
        _biomech_stability_state_sig=None,
        lbl_biomech_rom=None,
        _session_metric_view="sentinel",
    )

    with patch(
        "stablewalk.metrics.session_view.build_session_metric_snapshots"
    ) as build_snap:
        StableWalkGUI._update_biomechanics_panel(gui, frame_index=3)  # type: ignore[arg-type]
        build_snap.assert_not_called()
        assert gui._session_metric_view == "sentinel"


def test_paint_graphs_marks_hidden_tabs_dirty() -> None:
    """Hidden Motion/Biomech charts stay dirty until their tab is opened."""

    class _Gui:
        playing = False
        _gait_cycle = None
        _analysis_panel_dirty: set[str] = set()
        calls: list[str] = []

        def _group_visible(self, group: str) -> bool:
            return group in {"overview", "video", "skeleton", "trajectory"}

        def _mark_analysis_panels_dirty(self, *groups: str) -> None:
            self._analysis_panel_dirty.update(groups)

        def _rebuild_knee_angle_series(self) -> None:
            self.calls.append("knee")

        def _apply_default_knee_chart_mode(self) -> None:
            return None

        def _update_chart(self) -> None:
            self.calls.append("chart")

        def refresh_motion_analysis_tab(self) -> None:
            self.calls.append("motion")

        def _update_contact_gait_chart(self) -> None:
            self.calls.append("contact")

        def _update_biomechanics_panel(self, **_k) -> None:
            self.calls.append("biomech_panel")

        def _update_biomechanics_chart(self) -> None:
            self.calls.append("biomech_chart")

    from stablewalk.ui.tk.app import StableWalkGUI

    gui = _Gui()
    StableWalkGUI._paint_graphs_panels(gui)  # type: ignore[arg-type]
    StableWalkGUI._paint_biomech_panels(gui)  # type: ignore[arg-type]
    assert "contact" not in gui.calls
    assert "biomech_chart" not in gui.calls
    assert "motion" not in gui.calls
    assert {"biomech", "contact", "motion", "knee", "joint_motion"} <= gui._analysis_panel_dirty


def test_frame_rgb_cache_trim_keeps_newest() -> None:
    from stablewalk.pose.skeleton import FrameRgbCache

    cache = FrameRgbCache(max_entries=8)
    with cache._lock:
        for i in range(6):
            cache._store[f"f{i}"] = np.zeros((2, 2, 3), dtype=np.uint8)
    removed = cache.trim(3)
    assert removed == 3
    assert len(cache) == 3
    assert "f0" not in cache._store
    assert "f5" in cache._store


def test_group_visible_respects_active_tab_outside_playback() -> None:
    """Hidden Motion/Biomech groups stay invisible when Overview is active."""
    from types import MethodType

    from stablewalk.ui.tk.app import StableWalkGUI

    class _Gui:
        playing = False
        _pb_visible = None
        _tab_overview = object()
        _tab_motion = object()
        _tab_biomechanics = object()
        _tab_results_summary = object()
        _tab_compare = object()
        _overview_traj_dock_visible = False
        _overview_joint_motion_expanded = False
        video_label = None
        video_display_host = None
        canvas_3d = None
        skel_canvas_host = None
        canvas_dof_traj = None
        canvas_dof_traj_overview = None

        def _active_tab_widget(self):
            return self._tab_overview

        def _widget_mapped(self, _widget) -> bool:
            return True

    gui = _Gui()
    gui._visible_panel_groups = MethodType(  # type: ignore[attr-defined]
        StableWalkGUI._visible_panel_groups, gui
    )
    gui._widget_mapped = MethodType(  # type: ignore[attr-defined]
        _Gui._widget_mapped, gui
    )
    assert StableWalkGUI._group_visible(gui, "overview")  # type: ignore[arg-type]
    assert not StableWalkGUI._group_visible(gui, "motion")  # type: ignore[arg-type]
    assert not StableWalkGUI._group_visible(gui, "biomech")  # type: ignore[arg-type]
    assert not StableWalkGUI._group_visible(gui, "knee")  # type: ignore[arg-type]


def test_motion_tab_excludes_offscreen_video_skeleton() -> None:
    """Motion Analysis must not paint Overview video/skeleton canvases."""
    from types import MethodType

    from stablewalk.ui.tk.app import StableWalkGUI

    class _Gui:
        playing = True
        _pb_visible = None
        _tab_overview = object()
        _tab_motion = object()
        _tab_biomechanics = object()
        _tab_results_summary = object()
        _tab_compare = object()
        _overview_traj_dock_visible = False
        _overview_joint_motion_expanded = False

        def _active_tab_widget(self):
            return self._tab_motion

    gui = _Gui()
    gui._visible_panel_groups = MethodType(  # type: ignore[attr-defined]
        StableWalkGUI._visible_panel_groups, gui
    )
    groups = StableWalkGUI._visible_panel_groups(gui)  # type: ignore[arg-type]
    assert "motion" in groups
    assert "trajectory" in groups
    assert "joint_motion" in groups
    assert "video" not in groups
    assert "skeleton" not in groups


def test_maybe_auto_opensim_defers_until_advanced() -> None:
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW

    scheduled: list[int] = []

    class _Root:
        def after(self, delay_ms, _cb):
            scheduled.append(int(delay_ms))
            return "job"

    class _Gui:
        _presentation_mode = False
        _demo_running = False
        _session_restoring = False
        _opensim_sdk_available = True
        _opensim_auto_pipeline_busy = False
        _opensim_auto_pending = False
        root = _Root()

    gui = _Gui()

    import stablewalk.ui.tk.dashboard_tab_lifecycle as life

    original = life.active_tab_id
    life.active_tab_id = lambda _g: TAB_OVERVIEW  # type: ignore[assignment]
    try:
        StableWalkGUI._maybe_auto_opensim_pipeline(gui)  # type: ignore[arg-type]
    finally:
        life.active_tab_id = original  # type: ignore[assignment]

    assert gui._opensim_auto_pending is True
    assert scheduled == []


def test_ensure_motion_tab_charts_skips_contact() -> None:
    """Motion ensure must not construct Biomechanics contact figures."""
    from stablewalk.ui.tk import dashboard_layout as layout

    text = Path(layout.__file__).read_text(encoding="utf-8")
    start = text.find("def _ensure_motion_tab_charts()")
    assert start > 0
    end = text.find("gui._ensure_motion_knee_chart", start)
    body = text[start:end]
    assert "_ensure_contact_gait_chart" not in body
    assert "ok_contact" not in body


def test_fill_dof_position_table_virtualizes_tail() -> None:
    class _Tree:
        def __init__(self) -> None:
            self._rows: list[tuple] = []
            self.height = 0

        def get_children(self):
            return [str(i) for i in range(len(self._rows))]

        def item(self, iid, *, values=None, tags=None):
            self._rows[int(iid)] = values

        def insert(self, _parent, _index, values=(), tags=None):
            self._rows.append(values)
            return str(len(self._rows) - 1)

        def delete(self, iid):
            del self._rows[int(iid)]
            # reindex not needed for this test

        def configure(self, **kwargs):
            if "height" in kwargs:
                self.height = kwargs["height"]

        def column(self, *_a, **_k):
            return None

        def yview_moveto(self, *_a):
            return None

        def update_idletasks(self):
            return None

    from stablewalk.ui.tk.app import StableWalkGUI

    class _Gui:
        playing = False
        _dof_table_virtual_window = 10
        dof_pos_tree = _Tree()

        def _update_table_summary_label(self) -> None:
            return None

    gui = _Gui()
    rows = [(str(i), "0", "joint", "0", "0") for i in range(40)]
    StableWalkGUI._fill_dof_position_table(gui, rows, scroll_to_bottom=True)  # type: ignore[arg-type]
    assert len(gui.dof_pos_tree._rows) == 10
    assert gui.dof_pos_tree._rows[0][0] == "30"
