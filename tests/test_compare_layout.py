"""Compare tab layout minsizes and heatmap readability helpers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from stablewalk.ui.tk.dashboard_compare import (
    COMPARE_CHART_MIN_H,
    COMPARE_COL_CENTER_WEIGHT,
    COMPARE_COL_SIDE_WEIGHT,
    COMPARE_GRAPH_MIN_H,
    COMPARE_METRICS_MIN_H,
    COMPARE_SKELETON_MIN_H,
    COMPARE_STACK_BELOW_W,
    COMPARE_VIDEO_MIN_H,
    compare_workspace_natural_height,
    fit_compare_video_frame,
)


def test_compare_workspace_keeps_video_graph_video_row() -> None:
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "stablewalk" / "ui" / "tk" / "dashboard_compare.py"
    text = src.read_text(encoding="utf-8")
    assert '"Video A"' in text
    assert '"Difference Graph"' in text
    assert '"Difference Trajectory"' in text
    assert '"Difference Heatmap"' in text
    assert '"Video B"' in text
    assert '"Skeleton A"' in text
    assert '"Skeleton B"' in text
    assert "COMPARE_COL_CENTER_WEIGHT" in text
    assert "COMPARE_STACK_BELOW_W" in text
    assert "fit_compare_video_frame" in text
    assert "compare_workspace_natural_height" in text
    assert "summary.grid_remove()" not in text.split("def _reflow_side_by_side_workspace")[1].split("def _build_differences_table")[0]
    assert "details.grid_remove()" not in text.split("def _reflow_side_by_side_workspace")[1].split("def _build_differences_table")[0]


def test_fit_compare_video_frame_upscales_to_panel() -> None:
    from PIL import Image

    src = Image.new("RGB", (80, 45), color=(10, 20, 30))
    fitted = fit_compare_video_frame(src, 400, 300, resample=Image.NEAREST)
    assert fitted.size[0] >= 320
    assert fitted.size[1] >= 160
    src.close()
    fitted.close()


def test_difference_summary_wraps_before_cards_crush() -> None:
    from stablewalk.ui.tk.compare_delta_visuals import difference_summary_column_count

    assert difference_summary_column_count(500, 5) == 2
    assert difference_summary_column_count(800, 5) == 3
    assert difference_summary_column_count(1200, 5) == 5
    assert COMPARE_VIDEO_MIN_H >= 180
    assert COMPARE_SKELETON_MIN_H >= 160
    assert COMPARE_METRICS_MIN_H >= 160
    assert 180 <= COMPARE_GRAPH_MIN_H <= 280


def test_compare_playback_keeps_charts_and_skeletons_locked() -> None:
    """Compare play must not intentionally lag skeleton / difference playheads."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "stablewalk" / "ui" / "tk" / "dashboard_compare.py"
    text = src.read_text(encoding="utf-8")
    assert "_skel_paint_stride = 1" in text
    assert "_compare_chart_hz = 60.0" in text or "_compare_chart_hz = 60" in text
    # Sanity: Assignment nodes parse cleanly.
    ast.parse(text)


def test_compare_defaults_to_side_by_side_workspace() -> None:
    """Default Compare enables synchronized dual video / skeleton workspace."""
    from stablewalk.ui.tk.compare_state import ENABLE_DETAILED_COMPARE
    from stablewalk.ui.tk.compare_workflow import PRIMARY_COMPARE_METRICS
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    assert ENABLE_DETAILED_COMPARE is True
    assert "Cadence" in PRIMARY_COMPARE_METRICS
    assert "Stride Length" in PRIMARY_COMPARE_METRICS
    assert "Joint ROM" in PRIMARY_COMPARE_METRICS
    from stablewalk.ui.tk.compare_workflow import DEMO_DELTA_METRICS

    assert DEMO_DELTA_METRICS == ("Stability", "Symmetry", "Cadence", "Joint ROM")

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._static_compare_only = False
    ctrl._detailed_comparison = True
    ctrl._videos_built = False
    ctrl._skeletons_built = False
    ctrl._graphs_built = False
    ctrl._lite_chart_built = False
    ctrl._skel_section = None
    ctrl._skel_a_host = None
    ctrl._skel_b_host = None
    ctrl._detail_graphs_section = None
    ctrl._lite_chart_section = None
    ctrl._ensure_skeletons_built()
    ctrl._ensure_detailed_graphs_built()
    ctrl._ensure_lite_chart_built()
    # Without hosts, ensure helpers remain no-ops.
    assert ctrl._skeletons_built is False
    assert ctrl._graphs_built is False
    assert ctrl._lite_chart_built is False


def test_compare_row_floors_fit_short_workspace() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    class _Workspace:
        def __init__(self) -> None:
            self.rows: dict[int, int] = {}

        def winfo_width(self) -> int:
            return 720

        def winfo_height(self) -> int:
            return 150

        def columnconfigure(self, *_args, **_kwargs) -> None:
            pass

        def rowconfigure(self, row: int, **kwargs) -> None:
            self.rows[row] = int(kwargs.get("minsize", 0))

    workspace = _Workspace()
    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._compare_workspace = workspace
    ctrl.root = SimpleNamespace(winfo_width=lambda: 720)

    ctrl._reflow_side_by_side_workspace()

    assert ctrl._compare_stacked is True
    assert workspace.rows[0] >= COMPARE_VIDEO_MIN_H
    assert workspace.rows[1] >= COMPARE_CHART_MIN_H
    assert sum(workspace.rows.values()) > 150


def test_compare_gives_stacked_difference_graphs_more_vertical_space() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    class _Workspace:
        def __init__(self, width: int) -> None:
            self.width = width
            self.cols: dict[int, dict] = {}
            self.rows: dict[int, dict] = {}

        def winfo_width(self) -> int:
            return self.width

        def winfo_height(self) -> int:
            return 520

        def columnconfigure(self, col: int, **kwargs) -> None:
            self.cols[col] = kwargs

        def rowconfigure(self, row: int, **kwargs) -> None:
            self.rows[row] = kwargs

        def grid_configure(self, **_kwargs) -> None:
            pass

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl.root = SimpleNamespace(
        winfo_width=lambda: 1920,
        winfo_height=lambda: 900,
        rowconfigure=lambda *_a, **_k: None,
    )

    wide = _Workspace(1920)
    ctrl._compare_workspace = wide
    ctrl._reflow_side_by_side_workspace()
    assert wide.cols[1]["weight"] == COMPARE_COL_CENTER_WEIGHT
    assert wide.cols[0]["weight"] == COMPARE_COL_SIDE_WEIGHT
    assert wide.cols[2]["weight"] == COMPARE_COL_SIDE_WEIGHT
    assert wide.rows[0]["weight"] == 1
    assert int(wide.rows.get(1, {}).get("weight") or 0) == 0
    assert ctrl._compare_stacked is False

    medium = _Workspace(1366)
    ctrl._compare_workspace = medium
    ctrl._reflow_side_by_side_workspace()
    assert medium.cols[1]["weight"] == COMPARE_COL_CENTER_WEIGHT
    assert ctrl._compare_stacked is False
    assert compare_workspace_natural_height(stacked=True) > compare_workspace_natural_height(
        stacked=False
    )


def test_compare_stacks_visualizations_instead_of_hiding_them() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import (
        COMPARE_STACK_BELOW_W,
        ComparisonModeController,
    )

    class _Panel:
        def __init__(self) -> None:
            self.info: dict = {}

        def grid(self, **kwargs) -> None:
            self.info = kwargs

    class _Workspace:
        def __init__(self, width: int) -> None:
            self.width = width
            self.cols: dict[int, dict] = {}
            self.rows: dict[int, dict] = {}

        def winfo_width(self) -> int:
            return self.width

        def winfo_height(self) -> int:
            return 480

        def columnconfigure(self, col: int, **kwargs) -> None:
            self.cols[col] = kwargs

        def rowconfigure(self, row: int, **kwargs) -> None:
            self.rows[row] = kwargs

        def grid_configure(self, **_kwargs) -> None:
            pass

    def _ctrl(width: int) -> tuple[ComparisonModeController, _Workspace]:
        workspace = _Workspace(width)
        ctrl = ComparisonModeController.__new__(ComparisonModeController)
        ctrl.root = SimpleNamespace(
            winfo_width=lambda: width,
            winfo_height=lambda: 900,
        )
        ctrl._compare_workspace = workspace
        ctrl._compare_left_col = _Panel()
        ctrl._compare_center_col = _Panel()
        ctrl._compare_right_col = _Panel()
        ctrl._session_a_column = _Panel()
        ctrl._session_b_column = _Panel()
        ctrl._skel_a_panel = _Panel()
        ctrl._skel_b_panel = _Panel()
        ctrl._diff_graph_panel = _Panel()
        ctrl._traj_heat_panel = _Panel()
        ctrl._compare_summary_host = SimpleNamespace(grid_remove=lambda: None)
        ctrl._compare_details_host = SimpleNamespace(grid_remove=lambda: None)
        ctrl._transport_host = None
        ctrl._compare_metrics_host = None
        return ctrl, workspace

    wide_ctrl, wide = _ctrl(1600)
    wide_ctrl._reflow_side_by_side_workspace()
    assert wide_ctrl._compare_stacked is False
    assert COMPARE_STACK_BELOW_W >= 1040
    assert wide.cols[0]["weight"] == COMPARE_COL_SIDE_WEIGHT
    assert wide.cols[1]["weight"] == COMPARE_COL_CENTER_WEIGHT
    assert wide.cols[2]["weight"] == COMPARE_COL_SIDE_WEIGHT
    assert wide_ctrl._compare_left_col.info["row"] == 0
    assert wide_ctrl._compare_left_col.info["column"] == 0
    assert wide_ctrl._compare_center_col.info["column"] == 1
    assert wide_ctrl._compare_right_col.info["column"] == 2

    narrow_ctrl, narrow = _ctrl(COMPARE_STACK_BELOW_W - 40)
    narrow_ctrl._reflow_side_by_side_workspace()
    assert narrow_ctrl._compare_stacked is True
    assert narrow.cols[0]["weight"] == 1
    assert narrow.cols[1]["weight"] == 1
    assert int(narrow.cols[2].get("weight") or 0) == 0
    assert narrow_ctrl._compare_left_col.info["row"] == 0
    assert narrow_ctrl._compare_right_col.info["column"] == 1
    assert narrow_ctrl._compare_center_col.info["row"] == 1
    assert narrow_ctrl._compare_center_col.info.get("columnspan") == 2
    assert int(narrow.rows[0].get("minsize") or 0) >= COMPARE_VIDEO_MIN_H
    assert int(narrow.rows[1].get("minsize") or 0) >= COMPARE_CHART_MIN_H


def test_compare_action_groups_stack_before_overflow() -> None:
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    class _Actions:
        def __init__(self) -> None:
            self.last: dict = {}

        def pack_forget(self) -> None:
            self.last = {}

        def pack(self, **kwargs) -> None:
            self.last = kwargs

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._compare_left_actions = _Actions()
    ctrl._compare_right_actions = _Actions()
    ctrl._reflow_compare_chrome(1000)
    assert ctrl._compare_chrome_compact is True
    assert ctrl._compare_right_actions.last["side"] == "top"

    ctrl._reflow_compare_chrome(1366)
    assert ctrl._compare_chrome_compact is False
    assert ctrl._compare_right_actions.last["side"] == "right"


def test_compare_3d_unavailable_message_uses_xyz_text_signature() -> None:
    """Axes3D.text requires (x, y, z, s) — 2D (x, y, s) raises TypeError."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController
    from stablewalk.ui.viewers.plot_3d import setup_3d_axes

    fig = Figure(figsize=(4.0, 3.0), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax)
    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl.ax_chart = None
    ctrl.ax_path = ax
    ctrl._safe_draw = lambda _canvas: None  # type: ignore[method-assign]

    ctrl._chart_unavailable_message(ax, fig, object(), "Waiting for Session B")

    texts = list(getattr(ax, "texts", []) or [])
    assert texts, "3D unavailable state must still place an Axes3D text artist"
    assert "Waiting for Session B" in (texts[0].get_text() or "")


def test_compare_redraw_paths_unavailable_does_not_raise() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController
    from stablewalk.ui.viewers.plot_3d import setup_3d_axes

    fig = Figure(figsize=(4.0, 3.0), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax)
    canvas = FigureCanvasAgg(fig)
    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._graphs_built = True
    ctrl.ax_path = ax
    ctrl.fig_path = fig
    ctrl.canvas_path = canvas
    ctrl.ax_chart = None
    ctrl._safe_draw = lambda _canvas: None  # type: ignore[method-assign]
    ctrl._both_sessions_chart_ready = lambda: (False, "Waiting for Session B")  # type: ignore[method-assign]

    ctrl._redraw_paths()
    texts = list(getattr(ax, "texts", []) or [])
    assert any("Waiting for Session B" in (t.get_text() or "") for t in texts)


def test_compare_redraw_paths_draws_independent_a_and_b_trajectories() -> None:
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController
    from stablewalk.ui.tk.session_cache import AnalyzedSessionSnapshot
    from stablewalk.ui.viewers.plot_3d import setup_3d_axes

    fig = Figure(figsize=(4.0, 3.0), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    setup_3d_axes(ax)
    canvas = FigureCanvasAgg(fig)
    left = AnalyzedSessionSnapshot(
        key="a",
        label="A",
        path_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
    )
    right = AnalyzedSessionSnapshot(
        key="b",
        label="B",
        path_xyz=np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]], dtype=float),
    )
    cache = SimpleNamespace(left=lambda: left, right=lambda: right)
    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._graphs_built = True
    ctrl.ax_path = ax
    ctrl.fig_path = fig
    ctrl.canvas_path = canvas
    ctrl.ax_chart = None
    ctrl._progress = 0.0
    ctrl._show_path_left = SimpleNamespace(get=lambda: True)
    ctrl._show_path_right = SimpleNamespace(get=lambda: True)
    ctrl._show_path_delta = SimpleNamespace(get=lambda: True)
    ctrl._safe_draw = lambda _canvas: None  # type: ignore[method-assign]
    ctrl._both_sessions_chart_ready = lambda: (True, "")  # type: ignore[method-assign]
    ctrl._index_for = lambda _snap: 0  # type: ignore[method-assign]
    ctrl.gui = SimpleNamespace(_compare_session_cache=cache)

    ctrl._redraw_paths()
    lines = [ln for ln in ax.lines if ln.get_label() != "_nolegend_"]
    labels = [str(ln.get_label()) for ln in lines]
    assert any("Session A" in lab for lab in labels)
    assert any("Session B" in lab for lab in labels)
    assert any("Δ" in lab or "B−A" in lab or "B-A" in lab for lab in labels)


def test_heatmap_draw_sets_readable_margins() -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from stablewalk.ui.viewers.difference_heatmap import draw_difference_heatmap

    fig = Figure(figsize=(4.4, 3.2), dpi=100)
    ax = fig.add_subplot(111)
    draw_difference_heatmap(ax, None, None, xlabel="Video progress (%)")
    # Empty state should still leave readable axis area (no crash / no overlap).
    left = fig.subplotpars.left
    assert left >= 0.2


def test_compare_seek_helpers_normalize_progress() -> None:
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    sought: list[float] = []
    ctrl.seek = lambda p: sought.append(float(p))  # type: ignore[method-assign]
    ctrl._heat_bins = 41
    ctrl._seek_from_normalized_x(50.0)
    ctrl._seek_from_heatmap_x(20.0)
    assert sought[0] == pytest.approx(0.5)
    assert sought[1] == pytest.approx(0.5)


def test_compare_session_b_cta_helpers_exist() -> None:
    """Session B unavailable UX must expose Analyze / Load helpers."""
    from stablewalk.ui.tk.compare_state import (
        COMPARE_SESSION_B_NOT_ANALYZED,
        PLACEHOLDER_SKELETON_TITLE,
        PLACEHOLDER_VIDEO_TITLE,
        SESSION_B_GUIDANCE_TEXT,
        SESSION_B_METRICS_PLACEHOLDER_TITLE,
        SESSION_B_NEXT_ACTION,
        SESSION_B_WAITING_TITLE,
        SESSION_B_WORKFLOW_STEPS,
        SESSION_B_WORKFLOW_TITLE,
    )
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    assert COMPARE_SESSION_B_NOT_ANALYZED
    assert PLACEHOLDER_VIDEO_TITLE == SESSION_B_WAITING_TITLE
    assert PLACEHOLDER_SKELETON_TITLE == SESSION_B_WAITING_TITLE
    assert SESSION_B_METRICS_PLACEHOLDER_TITLE == SESSION_B_WAITING_TITLE
    assert "Analyze Session B" in SESSION_B_NEXT_ACTION
    assert "Load recording" in SESSION_B_NEXT_ACTION
    assert SESSION_B_WORKFLOW_TITLE
    assert len(SESSION_B_WORKFLOW_STEPS) == 3
    assert SESSION_B_GUIDANCE_TEXT == "\n".join(SESSION_B_WORKFLOW_STEPS)
    assert "different from Session A" in SESSION_B_GUIDANCE_TEXT
    assert "Video B" in SESSION_B_GUIDANCE_TEXT
    assert "Skeleton B" in SESSION_B_GUIDANCE_TEXT
    assert "metrics" in SESSION_B_GUIDANCE_TEXT
    assert hasattr(ComparisonModeController, "_fill_compare_host_stub")
    assert hasattr(ComparisonModeController, "_fill_session_b_placeholder")
    assert hasattr(ComparisonModeController, "_build_session_b_cta_frame")
    assert hasattr(ComparisonModeController, "_sync_session_b_cta")
    assert hasattr(ComparisonModeController, "_sync_session_b_workflow")
    assert hasattr(ComparisonModeController, "_session_b_needs_cta")
    assert hasattr(ComparisonModeController, "_fill_difference_summary")
    assert hasattr(ComparisonModeController, "_fill_bottom_metrics_bar")
    from stablewalk.ui.tk.compare_state import (
        PLACEHOLDER_DIFF_GRAPH_DETAIL,
        PLACEHOLDER_DIFF_PATH_DETAIL,
    )

    assert "No difference curve" in PLACEHOLDER_DIFF_GRAPH_DETAIL
    assert "after Session B is analyzed" in PLACEHOLDER_DIFF_PATH_DETAIL


def test_compare_placeholder_kinds_match_empty_panels() -> None:
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    kind = ComparisonModeController._placeholder_kind_from_title
    assert kind("Difference Graph") == "graph"
    assert kind("Difference Trajectory") == "heatmap"
    assert kind("Difference Heatmap") == "heatmap"
    assert kind("Difference Trajectory · Heatmap") == "heatmap"
    assert kind("Video B") == "video"
    assert kind("Skeleton B") == "skeleton"


def test_compare_placeholder_graphics_do_not_plot_series() -> None:
    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    canvas = MagicMock()
    canvas.winfo_width.return_value = 240
    canvas.winfo_height.return_value = 140
    ctrl._draw_placeholder_graphic(canvas, kind="graph")
    canvas.create_line.assert_called()
    canvas.plot.assert_not_called()
    labels = " ".join(
        str(call.args) + str(call.kwargs)
        for call in canvas.create_text.call_args_list
    )
    assert "Waiting for Session B" in labels
    canvas.reset_mock()
    ctrl._draw_placeholder_graphic(canvas, kind="heatmap")
    labels = " ".join(
        str(call.args) + str(call.kwargs)
        for call in canvas.create_text.call_args_list
    )
    assert "Trajectory" in labels
    assert "Heatmap" in labels
    canvas.plot.assert_not_called()


def test_session_b_needs_cta_without_snapshot() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController
    from stablewalk.ui.tk.session_cache import SessionCompareCache

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    cache = SessionCompareCache()
    gui = SimpleNamespace(_compare_session_cache=cache)
    ctrl.gui = gui
    assert ctrl._session_b_needs_cta() is True
    validation = SimpleNamespace(right_ready=False, right_reason="missing")
    assert ctrl._session_b_needs_cta(validation) is True
    validation_ok = SimpleNamespace(right_ready=True)
    assert ctrl._session_b_needs_cta(validation_ok) is False


def test_session_b_ready_automatically_removes_all_waiting_panels() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._set_video_b_cta_visible = MagicMock()  # type: ignore[method-assign]
    ctrl._set_skel_b_cta_visible = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_session_b_workflow = MagicMock()  # type: ignore[method-assign]

    ctrl._sync_session_b_cta(SimpleNamespace(right_ready=True))

    ctrl._set_video_b_cta_visible.assert_called_once_with(False, message=None)
    ctrl._set_skel_b_cta_visible.assert_called_once_with(False, message=None)
    ctrl._sync_session_b_workflow.assert_called_once_with(show=False)


def test_valid_compare_data_does_not_install_initial_placeholders() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._video_a_host = object()
    ctrl._skel_a_host = object()
    ctrl._lite_chart_section = object()
    ctrl._detail_graphs_section = object()
    ctrl._video_b_host = object()
    ctrl._skel_b_host = object()
    ctrl._fill_compare_host_stub = MagicMock()  # type: ignore[method-assign]
    ctrl._fill_session_b_placeholder = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_session_b_workflow = MagicMock()  # type: ignore[method-assign]

    ctrl._install_initial_compare_content(
        SimpleNamespace(left_ready=True, right_ready=True, can_compare=True)
    )

    ctrl._fill_compare_host_stub.assert_not_called()
    ctrl._fill_session_b_placeholder.assert_not_called()
    ctrl._sync_session_b_workflow.assert_called_once_with(show=False)


def test_missing_session_b_keeps_guidance_inside_b_panels() -> None:
    from types import SimpleNamespace

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController

    ctrl = ComparisonModeController.__new__(ComparisonModeController)
    ctrl._video_a_host = object()
    ctrl._skel_a_host = object()
    ctrl._lite_chart_section = object()
    ctrl._detail_graphs_section = object()
    ctrl._video_b_host = object()
    ctrl._skel_b_host = object()
    ctrl._fill_compare_host_stub = MagicMock()  # type: ignore[method-assign]
    ctrl._fill_session_b_placeholder = MagicMock()  # type: ignore[method-assign]
    ctrl._sync_session_b_workflow = MagicMock()  # type: ignore[method-assign]

    ctrl._install_initial_compare_content(
        SimpleNamespace(
            left_ready=True,
            right_ready=False,
            can_compare=False,
            right_reason="Session B is not analyzed.",
        )
    )

    assert ctrl._fill_compare_host_stub.call_count == 3
    assert ctrl._fill_session_b_placeholder.call_count == 2
    ctrl._sync_session_b_workflow.assert_called_once_with(show=False)


def test_path_delta_resample_is_b_minus_a() -> None:
    import numpy as np

    from stablewalk.ui.tk.dashboard_compare import ComparisonModeController
    from stablewalk.ui.tk.session_cache import AnalyzedSessionSnapshot

    a = AnalyzedSessionSnapshot(
        key="a",
        label="A",
        path_xyz=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
    )
    b = AnalyzedSessionSnapshot(
        key="b",
        label="B",
        path_xyz=np.array([[0.0, 0.1, 0.0], [1.0, 0.1, 0.0]], dtype=float),
    )
    pa = ComparisonModeController._path_xyz_on_progress_grid(a, n=5)
    pb = ComparisonModeController._path_xyz_on_progress_grid(b, n=5)
    assert pa is not None and pb is not None
    delta = pb - pa
    assert np.allclose(delta[:, 1], 0.1)
    assert np.allclose(delta[:, 0], 0.0)
