"""COM / Balance / Gait Phase must paint existing data when opened while paused."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

from matplotlib.figure import Figure

from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.viewers.biomechanics_charts import (
    BIOMECH_SERIES_COM,
    BIOMECH_SERIES_PHASE,
    BIOMECH_SERIES_STABILITY,
    draw_biomechanics_dashboard,
)
from tests.test_biomechanics_charts import _sample_result


class _UnmappedWidget:
    def winfo_ismapped(self) -> bool:
        return False

    def winfo_width(self) -> int:
        return 1

    def winfo_height(self) -> int:
        return 1


class _MappedChartWidget:
    def winfo_ismapped(self) -> bool:
        return True

    def winfo_width(self) -> int:
        return 640

    def winfo_height(self) -> int:
        return 360


class _TinyMappedWidget:
    def winfo_ismapped(self) -> bool:
        return True

    def winfo_width(self) -> int:
        return 1

    def winfo_height(self) -> int:
        return 1


class _FakeCanvas:
    def __init__(self, widget=None) -> None:
        self._widget = widget if widget is not None else _UnmappedWidget()

    def get_tk_widget(self):
        return self._widget

    def draw_idle(self) -> None:
        return None

    def draw(self) -> None:
        return None


def test_biomech_group_visible_when_tab_selected_even_if_canvas_unmapped() -> None:
    class _Gui:
        playing = False
        _pb_visible = None
        _tab_overview = object()
        _tab_motion = object()
        _tab_biomechanics = object()
        canvas_biomech = _FakeCanvas()
        canvas_contact_gait = None
        _CANVAS_GROUPS = StableWalkGUI._CANVAS_GROUPS

        def _active_tab_widget(self):
            return self._tab_biomechanics

        def _widget_mapped(self, _widget) -> bool:
            return False

    gui = _Gui()
    gui._visible_panel_groups = MethodType(  # type: ignore[attr-defined]
        StableWalkGUI._visible_panel_groups, gui
    )
    assert StableWalkGUI._group_visible(gui, "biomech")  # type: ignore[arg-type]


def test_last_frame_playhead_is_resolved_while_paused() -> None:
    frames = [
        SimpleNamespace(frame_index=i, timestamp_s=i * 0.04)
        for i in range(5)
    ]
    gui = SimpleNamespace(
        skeleton_player=SimpleNamespace(current_index=5),  # past last
        pose_indices=[0, 1, 2, 3, 4],
        sequence=SimpleNamespace(frames=frames),
        _frame_ts_cache_key=None,
        _frame_ts_cache={},
    )
    gui._timestamp_for_frame_index = MethodType(  # type: ignore[attr-defined]
        StableWalkGUI._timestamp_for_frame_index, gui
    )
    ts = StableWalkGUI._current_chart_playhead_time_s(gui)  # type: ignore[arg-type]
    assert ts == 0.16


def test_update_biomech_chart_draws_each_paused_tab(monkeypatch) -> None:
    result = _sample_result()
    drawn: list[str] = []

    def _capture(fig, analysis, *, playhead_time_s=None, series="com"):
        drawn.append(series)
        draw_biomechanics_dashboard(
            fig, analysis, playhead_time_s=playhead_time_s, series=series
        )

    monkeypatch.setattr(
        "stablewalk.ui.viewers.biomechanics_charts.draw_biomechanics_dashboard",
        _capture,
    )

    class _Gui:
        playing = False
        _scrubbing = False
        _biomech_analysis = result
        _biomech_series_key = "com"
        _biomech_chart_data_key = None
        fig_biomech = Figure(figsize=(6, 3))
        canvas_biomech = _FakeCanvas(_MappedChartWidget())
        skeleton_player = SimpleNamespace(current_index=23)
        pose_indices = list(range(24))
        sequence = SimpleNamespace(
            frames=[
                SimpleNamespace(frame_index=i, timestamp_s=i * 0.04) for i in range(24)
            ]
        )
        _frame_ts_cache_key = None
        _frame_ts_cache = {}

        def _group_visible(self, group: str) -> bool:
            return False  # unmapped canvas — force_rebuild must still paint

        def _move_existing_chart_playheads(self, *_a, **_k) -> bool:
            return False

        def _paint_canvas_now(self, *_a, **_k) -> None:
            return None

    gui = _Gui()
    gui._timestamp_for_frame_index = MethodType(  # type: ignore[attr-defined]
        StableWalkGUI._timestamp_for_frame_index, gui
    )
    gui._current_chart_playhead_time_s = MethodType(  # type: ignore[attr-defined]
        StableWalkGUI._current_chart_playhead_time_s, gui
    )

    for series in (BIOMECH_SERIES_COM, BIOMECH_SERIES_STABILITY, BIOMECH_SERIES_PHASE):
        gui._biomech_series_key = series
        gui._biomech_chart_data_key = None
        StableWalkGUI._update_biomechanics_chart(gui, force_rebuild=True)  # type: ignore[arg-type]

    assert drawn == [
        BIOMECH_SERIES_COM,
        BIOMECH_SERIES_STABILITY,
        BIOMECH_SERIES_PHASE,
    ]
    assert gui._biomech_chart_data_key[1] == BIOMECH_SERIES_PHASE


def test_undersized_biomech_chart_does_not_cache_data_key(monkeypatch) -> None:
    """A 1×1 first paint must not stick; later visible paints have to rebuild."""
    result = _sample_result()
    monkeypatch.setattr(
        "stablewalk.ui.viewers.biomechanics_charts.draw_biomechanics_dashboard",
        lambda *a, **k: None,
    )

    class _Gui:
        playing = False
        _scrubbing = False
        _biomech_analysis = result
        _biomech_series_key = BIOMECH_SERIES_COM
        _biomech_chart_data_key = None
        fig_biomech = Figure(figsize=(6, 3))
        canvas_biomech = _FakeCanvas(_TinyMappedWidget())
        skeleton_player = SimpleNamespace(current_index=0)
        pose_indices = [0]
        sequence = SimpleNamespace(
            frames=[SimpleNamespace(frame_index=0, timestamp_s=0.0)]
        )
        _frame_ts_cache_key = None
        _frame_ts_cache = {}

        def _group_visible(self, group: str) -> bool:
            return True

        def _move_existing_chart_playheads(self, *_a, **_k) -> bool:
            return False

        def _paint_canvas_now(self, *_a, **_k) -> None:
            return None

        def _timestamp_for_frame_index(self, index: int) -> float:
            return 0.0

        def _current_chart_playhead_time_s(self) -> float:
            return 0.0

    gui = _Gui()
    StableWalkGUI._update_biomechanics_chart(gui, force_rebuild=True)  # type: ignore[arg-type]
    assert gui._biomech_chart_data_key is None
    assert gui._biomech_chart_undersized is True


def _assert_axes_show_series_or_reason(ax, tab_id: str) -> None:
    has_data = bool(ax.lines) or bool(ax.collections) or bool(ax.patches)
    has_reason = any(
        word in (t.get_text() or "").lower()
        for t in ax.texts
        for word in ("unavailable", "no biomechanical analysis", "prerequisite")
    )
    assert has_data or has_reason, f"{tab_id} panel stayed empty"


def test_open_com_balance_phase_tabs_while_paused_at_last_frame() -> None:
    """GUI: after analysis, open COM / Balance / Contact / Gait Phase while paused."""
    import tkinter as tk

    import pytest

    from stablewalk.ui.tk.dashboard_notebook import TAB_BIOMECHANICS, select_dashboard_tab

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    app = None
    try:
        app = StableWalkGUI(root=root)
        app.playing = False
        result = _sample_result()
        app._biomech_analysis = result
        app._foot_contact = result.contact
        app._estimated_vgrf = None
        app.pose_indices = list(range(24))
        app.sequence = SimpleNamespace(
            frames=[
                SimpleNamespace(frame_index=i, timestamp_s=i * 0.04, detected=True)
                for i in range(24)
            ]
        )
        app.skeleton_player = SimpleNamespace(current_index=23, frame_count=24)
        app._frame_ts_cache_key = None
        app._frame_ts_cache = {}
        root.update_idletasks()
        select_dashboard_tab(app, TAB_BIOMECHANICS)
        root.update_idletasks()
        activate = getattr(app, "_activate_biomech_workspace_tab", None)
        assert callable(activate)
        for tab_id in ("com", "balance", "contact", "stability"):
            activate(tab_id)
            try:
                root.update()
            except tk.TclError as exc:
                pytest.skip(f"Tk unavailable: {exc}")
            if tab_id == "contact":
                fig = getattr(app, "fig_contact_gait", None)
            else:
                fig = getattr(app, "fig_biomech", None)
            assert fig is not None, f"{tab_id} figure missing"
            axes = fig.get_axes()
            assert axes, f"{tab_id} has no axes"
            _assert_axes_show_series_or_reason(axes[0], tab_id)

        activate("com")
        try:
            root.update()
        except tk.TclError as exc:
            pytest.skip(f"Tk unavailable: {exc}")
        app.skeleton_player.current_index = 10
        app._update_biomechanics_chart()
        try:
            root.update()
        except tk.TclError as exc:
            pytest.skip(f"Tk unavailable: {exc}")
        _assert_axes_show_series_or_reason(app.fig_biomech.get_axes()[0], "com-after-seek")

        app._biomech_analysis = None
        app._foot_contact = None
        app._estimated_vgrf = None
        app._biomech_chart_data_key = None
        app._contact_chart_data_key = None
        activate("com")
        try:
            root.update()
        except tk.TclError as exc:
            pytest.skip(f"Tk unavailable: {exc}")
        empty_texts = " ".join(
            t.get_text() or "" for t in app.fig_biomech.get_axes()[0].texts
        ).lower()
        assert "no biomechanical analysis" in empty_texts or "prerequisite" in empty_texts
        assert len(app.fig_biomech.get_axes()[0].lines) == 0
    finally:
        if app is not None:
            try:
                root.destroy()
            except tk.TclError:
                pass
