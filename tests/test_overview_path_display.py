"""Overview path display: camera views, path vs displacement, coordinate copy."""

from __future__ import annotations

from types import SimpleNamespace

from stablewalk.models.gait_motion import Vec3
from stablewalk.ui.overview_path_display import (
    CAMERA_VIEW_FRONTAL,
    CAMERA_VIEW_SAGITTAL,
    CAMERA_VIEW_SOURCE,
    CAMERA_VIEW_TRANSVERSE,
    coordinate_frame_caption,
    cumulative_path_length_m,
    displacement_m,
    format_est_cm,
    named_camera_views,
    path_and_displacement_labels,
    resolve_camera_view_name,
    source_camera_view_available,
)
from stablewalk.ui.viewers.dof_trajectory_3d import TRAJECTORY_CAMERA_PRESETS


def test_named_camera_views_are_explicit() -> None:
    views = named_camera_views(include_source=False)
    assert CAMERA_VIEW_SAGITTAL in views
    assert CAMERA_VIEW_FRONTAL in views
    assert CAMERA_VIEW_TRANSVERSE in views
    assert CAMERA_VIEW_SOURCE not in views
    assert CAMERA_VIEW_SOURCE in named_camera_views(include_source=True)
    assert resolve_camera_view_name("Side") == CAMERA_VIEW_SAGITTAL
    assert resolve_camera_view_name("Front") == CAMERA_VIEW_FRONTAL
    assert resolve_camera_view_name("Top") == CAMERA_VIEW_TRANSVERSE
    for name in named_camera_views(include_source=True):
        assert name in TRAJECTORY_CAMERA_PRESETS


def test_source_camera_view_is_gated_on_camera_aligned_frame() -> None:
    assert source_camera_view_available(None) is False
    rec = SimpleNamespace(
        coordinate_system="pelvis_centered_y_up_z_forward",
        root_trajectory=((0.0, 0.0, 0.1),),
    )
    assert source_camera_view_available(rec) is True
    unknown = SimpleNamespace(coordinate_system="unknown", root_trajectory=None)
    assert source_camera_view_available(unknown) is False
    assert source_camera_view_available(unknown, coord_mode="Root-relative") is True


def test_path_length_is_not_displacement() -> None:
    path = (
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.03, 0.0, 0.0),
        Vec3(0.03, 0.04, 0.0),
    )
    length = cumulative_path_length_m(path)
    disp = displacement_m(path)
    assert length is not None and disp is not None
    assert abs(length - 0.07) < 1e-9
    assert abs(disp - 0.05) < 1e-9
    mid_len = cumulative_path_length_m(path, end_index=1)
    mid_disp = displacement_m(path, end_index=1)
    assert abs(mid_len - 0.03) < 1e-9
    assert abs(mid_disp - 0.03) < 1e-9
    path_lbl, disp_lbl = path_and_displacement_labels(path)
    assert path_lbl != disp_lbl
    assert "cm est." in path_lbl
    assert "cm est." in disp_lbl


def test_coordinate_caption_explains_origin_axes_and_estimates() -> None:
    rec = SimpleNamespace(coordinate_system="pelvis_centered_y_up_z_forward")
    text = coordinate_frame_caption(coord_mode="Root-relative", recording=rec)
    assert "pelvis" in text.lower()
    assert "X lateral" in text
    assert "Y up" in text
    assert "Z depth" in text
    assert "not mocap-calibrated" in text
    assert "not world" in text
    calibrated = SimpleNamespace(coordinate_system="calibrated_lab_meters")
    cal = coordinate_frame_caption(coord_mode="Global", recording=calibrated)
    assert "calibrated" in cal.lower()
    assert format_est_cm(0.123) == "12.3 cm est."
    assert format_est_cm(None) == "N/A"
    root_caption = coordinate_frame_caption(coord_mode="Root-relative", recording=rec)
    assert "not overground" in root_caption
    global_caption = coordinate_frame_caption(coord_mode="Global", recording=rec)
    assert "Treadmill belt travel is not the same measurement" in global_caption


def test_path_length_skips_long_tracking_gaps() -> None:
    path = (
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.10, 0.0, 0.0),
        Vec3(0.50, 0.0, 0.0),  # would add 0.40 m if connected across the hole
        Vec3(0.60, 0.0, 0.0),
    )
    times = (0.0, 1.0 / 30.0, 5.0 / 30.0, 6.0 / 30.0)
    naive = cumulative_path_length_m(path)
    gap_aware = cumulative_path_length_m(path, times_s=times, fps=30.0)
    assert naive is not None and abs(naive - 0.60) < 1e-9
    assert gap_aware is not None and abs(gap_aware - 0.20) < 1e-9


def test_displacement_is_na_on_first_sample_and_origin_lock() -> None:
    path = (
        Vec3(0.02, -0.40, 0.01),
        Vec3(0.03, -0.41, 0.02),
    )
    assert displacement_m(path, end_index=0) is None
    assert format_est_cm(displacement_m(path, end_index=0)) == "N/A"
    assert displacement_m(path, origin_locked=True) is None
    later = displacement_m(path, end_index=1)
    assert later is not None and later > 0.0


def test_overview_panes_share_width_at_common_sizes() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.analysis_split import (
        MIN_W_PATH,
        MIN_W_SKELETON,
        MIN_W_VIDEO,
        create_h_split,
        split_add,
    )
    from stablewalk.ui.tk.dashboard_overview_panes import (
        DEFAULT_TRAJ_SASH_FRACTIONS,
        _clamp_sash_pixels,
        install_overview_workspace_split,
    )
    from stablewalk.ui.tk.dashboard_sections import OVERVIEW_HOST_VIZ_ROW

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        import pytest

        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        host = tk.Frame(root)
        host.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        gui = SimpleNamespace()
        primary, _info = install_overview_workspace_split(gui, host)
        split = create_h_split(primary)
        split.grid(row=0, column=0, columnspan=3, sticky="nsew")
        panes = [tk.Frame(split, width=200, height=280) for _ in range(3)]
        for pane in panes:
            pane.columnconfigure(0, weight=1)
            pane.rowconfigure(0, weight=1)
        split_add(split, panes[0], minsize=MIN_W_VIDEO)
        split_add(split, panes[1], minsize=MIN_W_SKELETON)
        split_add(split, panes[2], minsize=MIN_W_PATH)
        root.deiconify()
        measurements = {}
        for width, height in ((1366, 768), (1920, 1080)):
            root.geometry(f"{width}x{height}")
            root.update_idletasks()
            root.update()
            host.update_idletasks()
            split.update_idletasks()
            sw = int(split.winfo_width() or 0)
            if sw < 200:
                sw = width - 80
            x0, x1 = _clamp_sash_pixels(
                sw,
                int(round(DEFAULT_TRAJ_SASH_FRACTIONS[0] * sw)),
                int(round(DEFAULT_TRAJ_SASH_FRACTIONS[1] * sw)),
            )
            video_w = x0
            skel_w = x1 - x0
            path_w = sw - x1
            measurements[(width, height)] = (video_w, skel_w, path_w, sw)
            assert video_w >= MIN_W_VIDEO
            assert skel_w >= MIN_W_SKELETON
            assert path_w >= MIN_W_PATH
            # Video is the primary pane; skeleton and 3D stay usable.
            assert video_w >= 0.35 * sw
            assert skel_w >= 0.24 * sw
            assert path_w >= 0.24 * sw
            assert video_w >= skel_w
            assert video_w >= path_w
            assert path_w <= 0.40 * sw + 1
        assert measurements[(1366, 768)][3] >= 200
        assert measurements[(1920, 1080)][2] >= measurements[(1366, 768)][2]
    finally:
        root.destroy()


def test_path_stats_start_collapsed() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_layout import (
        _build_overview_path_stats_panel,
        set_overview_path_stats_expanded,
    )

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        import pytest

        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        panel = tk.Frame(root, width=400, height=400)
        panel.grid(row=0, column=0, sticky="nsew")
        panel.rowconfigure(1, weight=1)
        gui = SimpleNamespace(playing=False)
        _build_overview_path_stats_panel(gui, panel)
        root.update_idletasks()
        body = gui._overview_path_stats_body
        assert gui._overview_path_stats_expanded is False
        assert str(body.winfo_manager() or "") == ""
        compact = gui._overview_path_metrics_strip
        assert str(compact.winfo_manager() or "") == "grid"
        assert gui.overview_path_metric_values["path_length"].cget("text") == "—"
        assert gui.overview_path_metric_values["displacement"].cget("text") == "—"
        assert gui.overview_path_metric_values["xyz"].cget("text") == "—"
        assert gui.overview_path_extent_values
        assert gui.lbl_overview_coord_caption is not None
        caption = str(gui.lbl_overview_coord_caption.cget("text") or "")
        assert "Origin" in caption
        set_overview_path_stats_expanded(gui, True)
        root.update_idletasks()
        assert gui._overview_path_stats_expanded is True
        assert str(body.winfo_manager() or "") == "grid"
        set_overview_path_stats_expanded(gui, False)
        root.update_idletasks()
        assert str(body.winfo_manager() or "") == ""
        assert str(compact.winfo_manager() or "") == "grid"
    finally:
        root.destroy()
