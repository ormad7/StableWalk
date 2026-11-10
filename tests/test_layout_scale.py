"""DPI layout tokens and Motion inner-canvas hosts — no analysis."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.ui.tk.layout_scale import (
    MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
    RESOLUTION_TARGETS,
    SCALE_TARGETS,
    WIDTH_XLARGE,
    combobox_char_width,
    dpi_scale_factor,
    layout_scale_from_gui,
    scaled_breakpoints,
    scaled_px,
)


def test_minimum_window_stays_practical_at_high_scale() -> None:
    assert MIN_WINDOW_WIDTH == 960
    assert MIN_WINDOW_HEIGHT == 600
    assert WIDTH_XLARGE == 2200
    assert RESOLUTION_TARGETS[-1] == (3840, 2160)
    assert SCALE_TARGETS == (1.0, 1.25, 1.50, 1.75)
    # Floor must still fit a 1366×768 laptop at 150% logical size (~910×512).
    assert MIN_WINDOW_WIDTH <= 960
    assert MIN_WINDOW_HEIGHT <= 600


def test_scaled_breakpoints_grow_with_dpi() -> None:
    base = scaled_breakpoints(1.0)
    high = scaled_breakpoints(1.5)
    assert high["viz_three_col"] > base["viz_three_col"]
    assert high["biomech_stack"] > base["biomech_stack"]
    assert high["compare_stack"] > base["compare_stack"]
    assert high["advanced_two_col"] > base["advanced_two_col"]
    assert scaled_px(100, 1.25) == 125
    assert scaled_px(100, 1.0, minimum=40) == 100
    assert combobox_char_width(("Left knee flexion", "X"), minimum=12) >= 18
    assert combobox_char_width((), minimum=12) == 12
    assert combobox_char_width(("abcdefghijklmnop"), maximum=10) == 10


def test_layout_scale_from_gui_without_root_is_identity() -> None:
    gui = SimpleNamespace()
    assert layout_scale_from_gui(gui) == 1.0
    dummy = SimpleNamespace(root=object())
    assert dpi_scale_factor(dummy.root) == 1.0  # type: ignore[arg-type]
    cached = SimpleNamespace(_layout_dpi_scale=1.5, root=None)
    assert layout_scale_from_gui(cached) == 1.5


def test_classify_layout_covers_requested_resolutions() -> None:
    from stablewalk.ui.tk.dashboard_responsive import WidthMode, classify_layout

    expected = {
        1366: WidthMode.MEDIUM,
        1600: WidthMode.LARGE,
        1920: WidthMode.LARGE,
        2560: WidthMode.XLARGE,
        3840: WidthMode.XLARGE,
    }
    for width, height in RESOLUTION_TARGETS:
        wmode, _hmode = classify_layout(width, height)
        assert wmode is expected[width]


def test_motion_inner_figure_reflow_does_not_rebuild_figures() -> None:
    from stablewalk.ui.tk.dashboard_responsive import _reflow_motion_inner_figures

    fig = object()
    gui = SimpleNamespace(
        fig_spectral=fig,
        canvas_spectral=None,
        fig_spectral_features=fig,
        canvas_spectral_features=None,
        fig_asymmetry=fig,
        canvas_asymmetry=None,
        lbl_spectral_warnings=None,
        lbl_spectral_feature_def=None,
        lbl_spectral_features_meta=None,
        lbl_asymmetry_meta=None,
    )
    _reflow_motion_inner_figures(gui, width=1280)
    assert gui.fig_spectral is fig
    assert gui.fig_spectral_features is fig
    assert gui.fig_asymmetry is fig


def test_spectral_and_asymmetry_hosts_are_dedicated_frames() -> None:
    src_dir = __import__("pathlib").Path(__file__).resolve().parents[1] / "stablewalk" / "ui" / "tk"
    spectral = (src_dir / "dashboard_spectral.py").read_text(encoding="utf-8")
    features = (src_dir / "dashboard_spectral_features.py").read_text(encoding="utf-8")
    asymmetry = (src_dir / "dashboard_asymmetry.py").read_text(encoding="utf-8")
    assert "relayout_spectral_plots" in spectral
    assert "create_h_split" in spectral
    assert "open_spectral_plot_window" in spectral
    assert "create_h_split" in features
    assert "feature_units_display" in features
    assert "create_h_split" in asymmetry
    assert "asymmetry_plot_id_for_row" in asymmetry
    assert "_bind_figure_resize" in features
    assert "_bind_figure_resize" in asymmetry
    assert "gui._spectral_graph_host = graph_host" in spectral
    assert "gui._spectral_features_graph_host = graph_host" in features
    assert "gui._asymmetry_graph_host = graph_host" in asymmetry
    assert "bind_persistent_h_sash" in spectral
    assert "bind_persistent_h_sash" in features
    assert "lbl_spectral_row_detail" in spectral
    assert "install_maximize_button_grid" in spectral
    assert "install_maximize_button_grid" in features
    assert "install_maximize_button_grid" in asymmetry


def test_resize_handlers_do_not_re_run_analysis() -> None:
    from pathlib import Path

    src_dir = Path(__file__).resolve().parents[1] / "stablewalk" / "ui" / "tk"
    forbidden = (
        "_run_session_analysis",
        "run_session_analysis",
        "analyze_spectral(",
        "run_biomechanical_analysis(",
    )
    for name in (
        "dashboard_responsive.py",
        "overview_responsive.py",
        "dashboard_overview_panes.py",
        "layout_scale.py",
    ):
        src = (src_dir / name).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in src, f"{name} must not call {token} on resize"
