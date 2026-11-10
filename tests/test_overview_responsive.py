"""Overview responsive breakpoints, reflow, and layout diagnostics."""

from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import pytest

from stablewalk.ui.tk.overview_responsive import (
    WIDTH_KPI_SIX,
    WIDTH_SUMMARY_SIDE_BY_SIDE,
    WIDTH_VIZ_THREE_COL,
    WIDTH_VIZ_TWO_PLUS_ONE,
    collect_overview_layout_snapshot,
    dpi_scale_factor,
    overview_breakpoints,
    overview_viz_layout_mode,
    reflow_overview_kpi_cards,
    reflow_overview_summary_panels,
)


def _tk_root() -> tk.Tk:
    """Create a hidden Tk root, or skip when Tcl/Tk is unavailable on this host."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    return root


def test_breakpoints_for_supported_widths() -> None:
    from stablewalk.ui.tk.overview_responsive import WIDTH_KPI_FOUR

    # 1920 — all wide layouts
    b = overview_breakpoints(1920)
    assert b["kpi_four"] and b["summary_side_by_side"] and b["viz_three_col"]

    # 1600 — primary KPIs four-col; viz three-col when path open; summary side-by-side
    b = overview_breakpoints(1600)
    assert b["kpi_four"] and b["summary_side_by_side"] and b["viz_three_col"]

    # 1366 — KPI four-col + viz three-col (≥1100); summary side-by-side
    b = overview_breakpoints(1366)
    assert b["kpi_four"]
    assert b["summary_side_by_side"]
    assert b["viz_three_col"]

    # 1100 — KPI four-col; viz still three-col; summary stacks (<1280)
    b = overview_breakpoints(1100)
    assert b["kpi_four"]
    assert not b["summary_side_by_side"]
    assert b["viz_three_col"]

    # 1000 — 2+1 instead of compressing three columns
    b = overview_breakpoints(1000)
    assert b["kpi_four"]
    assert not b["viz_three_col"]
    assert b["viz_two_plus_one"]
    assert not b["viz_stacked"]
    assert WIDTH_KPI_FOUR == 720
    assert WIDTH_VIZ_THREE_COL == 1100
    assert WIDTH_VIZ_TWO_PLUS_ONE == 900
    assert overview_viz_layout_mode(1920) == "three_col"
    assert overview_viz_layout_mode(1280) == "three_col"
    assert overview_viz_layout_mode(1000) == "two_plus_one"
    assert overview_viz_layout_mode(800) == "stacked"
    # 1366×768 and 1920×1080 stay three draggable panes at typical DPI.
    assert overview_viz_layout_mode(1366, scale=1.25) == "three_col"
    assert overview_viz_layout_mode(1920, scale=1.50) == "three_col"
    assert overview_viz_layout_mode(1366, scale=1.0) == "three_col"


def test_kpi_reflow_wraps_to_two_by_two() -> None:
    from stablewalk.ui.tk.overview_responsive import WIDTH_KPI_FOUR

    root = _tk_root()
    try:
        metrics = tk.Frame(root)
        metrics.pack(fill=tk.X)
        cards = {}
        order = []
        for i, key in enumerate(
            (
                "gait_quality",
                "stability_state",
                "symmetry",
                "cadence",
            )
        ):
            frame = tk.Frame(metrics)
            desc = tk.Label(frame, text="desc", wraplength=100)
            desc.pack()
            cards[key] = {"frame": frame, "desc": desc}
            order.append(key)
            frame.grid(row=0, column=i)

        gui = SimpleNamespace(
            _overview_clinical_cards=cards,
            _overview_kpi_order=order,
            _overview_secondary_order=("analysis_confidence", "gait_cycles"),
            _overview_kpi_layout=None,
            _overview_kpi_force_one_row=False,
            _force_overview_reflow=True,
        )
        mode = reflow_overview_kpi_cards(gui, width=WIDTH_KPI_FOUR - 1)
        assert mode == "2x2"
        # First card of second row should be at row 1 col 0
        info = cards["symmetry"]["frame"].grid_info()
        assert int(info["row"]) == 1
        assert int(info["column"]) == 0

        gui._force_overview_reflow = True
        mode = reflow_overview_kpi_cards(gui, width=WIDTH_KPI_FOUR)
        assert mode == "4col"
        info = cards["cadence"]["frame"].grid_info()
        assert int(info["row"]) == 0
        assert int(info["column"]) == 3

        # The one-row preference must yield to the narrow-width breakpoint.
        gui._overview_kpi_force_one_row = True
        gui._force_overview_reflow = True
        mode = reflow_overview_kpi_cards(gui, width=WIDTH_KPI_FOUR - 1)
        assert mode == "2x2"
    finally:
        root.destroy()


def test_summary_reflow_stacks_when_narrow() -> None:
    root = _tk_root()
    try:
        report = tk.Frame(root)
        summary = tk.Frame(report)
        limitations = tk.Frame(report)
        summary.grid(row=0, column=0)
        limitations.grid(row=0, column=1)
        gui = SimpleNamespace(
            _overview_clinical_report=report,
            _overview_summary_card=summary,
            _overview_limitations_card=limitations,
            _overview_clinical_bullet_labels=[],
            _overview_system_note_labels=[],
            _overview_summary_layout=None,
            _force_overview_reflow=True,
        )
        mode = reflow_overview_summary_panels(
            gui, width=WIDTH_SUMMARY_SIDE_BY_SIDE - 1
        )
        assert mode == "stacked"
        assert int(limitations.grid_info()["row"]) == 1

        gui._force_overview_reflow = True
        mode = reflow_overview_summary_panels(gui, width=WIDTH_SUMMARY_SIDE_BY_SIDE)
        assert mode == "side_by_side"
        assert int(limitations.grid_info()["row"]) == 0
        assert int(limitations.grid_info()["column"]) == 1
    finally:
        root.destroy()


def test_overview_breakpoints_grow_with_display_scale() -> None:
    """1366×768 stays three draggable panes; truly narrow widths still restack."""
    native = overview_breakpoints(1366, scale=1.0)
    scaled = overview_breakpoints(1366, scale=1.5)
    assert native["viz_three_col"]
    assert scaled["viz_three_col"]
    assert overview_viz_layout_mode(1920, scale=1.0) == "three_col"
    assert overview_viz_layout_mode(1920, scale=1.75) == "three_col"
    assert overview_viz_layout_mode(2560, scale=1.75) == "three_col"
    assert overview_viz_layout_mode(1000, scale=1.5) == "stacked"


def test_dpi_scale_factor_is_positive() -> None:
    root = _tk_root()
    try:
        scale = dpi_scale_factor(root)
        assert 0.75 <= scale <= 3.0
    finally:
        root.destroy()


def test_layout_snapshot_reports_transport_and_dpi() -> None:
    root = _tk_root()
    try:
        transport = tk.Frame(root)
        transport.pack()
        gui = SimpleNamespace(
            root=root,
            _transport_row=transport,
            video_frame=None,
            skel_frame=None,
            overview_traj_panel=None,
            _overview_clinical_summary=None,
            _section_visual=None,
            _primary_viz_host=None,
            _overview_content=None,
            _overview_kpi_layout="3x2",
            _overview_summary_layout="stacked",
            _overview_viz_layout="stacked",
            _overview_scroll_needed=False,
        )
        root.update_idletasks()
        snap = collect_overview_layout_snapshot(gui, width=1366, height=768)
        assert snap["window"] == (1366, 768)
        assert snap["dpi_scale"] >= 0.75
        assert snap["breakpoints"]["kpi_four"] is True  # 1366 ≥ 720
        assert snap["transport_present"] is True
    finally:
        root.destroy()


def test_section_minsizes_are_soft() -> None:
    from stablewalk.ui.tk.dashboard_sections import (
        SEC1_PANEL_MINWIDTH,
        SEC1_TRAJ_PANEL_MINWIDTH,
        SEC1_TRAJ_VIZ_ROW_MINSIZE,
        SEC1_VIZ_ROW_MINSIZE,
    )

    # Visualization floor stays modest so KPI / chrome rows remain on-screen.
    assert 240 <= SEC1_VIZ_ROW_MINSIZE <= 360
    assert 240 <= SEC1_TRAJ_VIZ_ROW_MINSIZE <= 400
    assert 180 <= SEC1_PANEL_MINWIDTH <= 280
    # Path card is the primary visualization: its floor holds the 3D viewport plus
    # the compact inspector, but stays small enough to avoid forced h-scroll.
    assert SEC1_TRAJ_PANEL_MINWIDTH <= 280


def test_overview_scroll_stretches_content_to_viewport() -> None:
    """Direct Overview host: viz row keeps minsize and is not canvas-embedded."""
    root = _tk_root()
    try:
        from stablewalk.ui.tk.overview_responsive import (
            VIZ_ROW_MINSIZE_WIDE,
            ensure_overview_viz_room,
            install_overview_scroll_host,
        )

        # withdraw() makes winfo_ismapped() false — deiconify for geometry checks.
        root.deiconify()
        tab = tk.Frame(root, width=1100, height=900)
        tab.grid(row=0, column=0, sticky="nsew")
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)
        gui = SimpleNamespace(
            _overview_viz_stack_requested=False,
            _tab_overview=tab,
            _overview_clinical_summary=None,
            _section_visual=None,
            _primary_viz_host=None,
        )
        content = install_overview_scroll_host(gui, tab)
        assert gui._overview_scroll_canvas is None
        tk.Label(content, text="header").grid(row=0, column=0, sticky="ew")
        clinical = tk.Frame(content, height=200)
        clinical.grid(row=2, column=0, sticky="ew")
        gui._overview_clinical_summary = clinical
        viz = tk.Frame(content, name="viz", bg="#123")
        viz.grid(row=3, column=0, sticky="nsew")
        gui._section_visual = viz
        content.rowconfigure(0, weight=0)
        content.rowconfigure(2, weight=0)
        content.rowconfigure(3, weight=1, minsize=VIZ_ROW_MINSIZE_WIDE)
        root.geometry("1100x900+40+40")
        root.update_idletasks()
        root.update()
        ensure_overview_viz_room(gui)
        root.update_idletasks()
        assert str(viz.winfo_manager()) == "grid"
        assert int(content.grid_rowconfigure(3)["weight"]) == 1
        assert int(content.grid_rowconfigure(0)["weight"]) == 0
        assert int(content.grid_rowconfigure(2)["weight"]) == 0
        assert int(content.grid_rowconfigure(3)["minsize"]) >= 200
        # Content is a direct child of the tab — not a Canvas window.
        assert str(content.master) == str(tab)
    finally:
        root.destroy()


@pytest.mark.parametrize(
    "width,expect_stack",
    [
        (1920, False),
        (1600, False),
        (1366, False),
        (1280, False),
        (1100, False),
        (1000, True),
    ],
)
def test_viz_stack_threshold(width: int, expect_stack: bool) -> None:
    assert (width < WIDTH_VIZ_THREE_COL) is expect_stack


def test_overview_responsive_matrix_smoke() -> None:
    """Construct GUI and apply responsive layout at supported sizes."""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    try:
        from stablewalk.ui.tk.app import StableWalkGUI
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
        from stablewalk.ui.tk.overview_responsive import collect_overview_layout_snapshot

        app = StableWalkGUI(root=root)
        root.update_idletasks()
        assert getattr(app, "_transport_row", None) is not None
        sizes = ((1920, 1080), (1680, 900), (1366, 768), (1280, 720))
        fig_id = id(getattr(app, "fig_3d", None))
        canvas_id = id(getattr(app, "canvas_3d", None))
        seq_id = id(getattr(app, "sequence", None))
        for w, h in sizes:
            root.geometry(f"{w}x{h}+-3200+0")
            root.update_idletasks()
            apply_responsive_layout(app, width=w, height=h)
            root.update_idletasks()
            snap = collect_overview_layout_snapshot(app, width=w, height=h)
            assert snap["transport_present"] is True
            assert snap["kpi_layout"] in ("4col", "2x2", "6col", "3x2", None)
            # Charts must not be recreated across layout passes.
            assert id(getattr(app, "fig_3d", None)) == fig_id
            assert id(getattr(app, "canvas_3d", None)) == canvas_id
            assert id(getattr(app, "sequence", None)) == seq_id
            assert getattr(app, "_overview_reset_layout_btn", None) is not None
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_overview_arrangement_uses_two_plus_one_when_narrow() -> None:
    root = _tk_root()
    try:
        from types import SimpleNamespace

        from stablewalk.ui.tk.overview_responsive import apply_overview_viz_arrangement

        section = tk.Frame(root)
        video = tk.Frame(section)
        skel = tk.Frame(section)
        path = tk.Frame(section)
        phase = tk.Frame(section)
        clearance = tk.Frame(section)
        gui = SimpleNamespace(
            _section_visual=section,
            _primary_viz_host=section,
            video_frame=video,
            skel_frame=skel,
            overview_traj_panel=path,
            _foot_clearance_detail_host=clearance,
            _overview_phase_strip=phase,
            _primary_viz_content_row=0,
            _overview_short_window=0,
            root=root,
        )
        mode = apply_overview_viz_arrangement(gui, width=1000)
        assert mode == "two_plus_one"
        assert int(video.grid_info()["row"]) == 0
        assert int(skel.grid_info()["row"]) == 0
        assert int(path.grid_info()["row"]) == 1
        assert int(path.grid_info()["columnspan"]) == 2
        mode = apply_overview_viz_arrangement(gui, width=800)
        assert mode == "stacked"
        assert int(video.grid_info()["row"]) == 0
        assert int(skel.grid_info()["row"]) == 1
        assert int(path.grid_info()["row"]) == 2
    finally:
        root.destroy()


def test_overview_viz_floors_stay_usable() -> None:
    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_PATH_MIN_W,
        OVERVIEW_SKELETON_MIN_W,
        OVERVIEW_VIDEO_MIN_W,
        OVERVIEW_VIDEO_ROW_SOFT_MIN_H,
        apply_overview_viz_grid_weights,
        overview_video_row_min_h,
    )

    assert OVERVIEW_VIDEO_ROW_SOFT_MIN_H >= 260
    assert overview_video_row_min_h() >= 260
    assert OVERVIEW_VIDEO_MIN_W >= 240
    assert OVERVIEW_SKELETON_MIN_W >= 220
    assert OVERVIEW_PATH_MIN_W >= 240

    root = _tk_root()
    try:
        section = tk.Frame(root, width=900, height=400)
        apply_overview_viz_grid_weights(section)
        assert int(section.grid_columnconfigure(0)["minsize"]) >= OVERVIEW_VIDEO_MIN_W
        assert int(section.grid_columnconfigure(1)["minsize"]) >= OVERVIEW_SKELETON_MIN_W
        assert int(section.grid_columnconfigure(2)["minsize"]) >= OVERVIEW_PATH_MIN_W
        assert int(section.grid_rowconfigure(0)["minsize"]) >= OVERVIEW_VIDEO_ROW_SOFT_MIN_H
    finally:
        root.destroy()


def test_overview_viz_floors_fit_allocated_height() -> None:
    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT,
        overview_fit_viz_row_floors,
    )

    floor, phase, clearance = overview_fit_viz_row_floors(
        short=True, layout="three_col", available_h=400
    )
    assert floor + phase + clearance <= 400
    assert floor >= 120
    assert clearance >= OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT
    wide = overview_fit_viz_row_floors(
        short=False, layout="three_col", available_h=700
    )
    assert wide[0] >= 260
    assert wide[0] + wide[1] + wide[2] <= 700


def test_overview_panels_stay_large_at_supported_sizes() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    try:
        from stablewalk.ui.tk.app import StableWalkGUI
        from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
        from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
        from stablewalk.ui.tk.dashboard_sections import (
            OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT,
            OVERVIEW_PATH_MIN_W,
            OVERVIEW_SKELETON_MIN_W,
            OVERVIEW_VIDEO_MIN_W,
            OVERVIEW_VIDEO_ROW_SOFT_MIN_H,
        )

        from stablewalk.ui.tk.dashboard_overview_view_mode import (
            VIEW_MODE_SIDE_BY_SIDE,
            apply_overview_view_mode,
        )

        app = StableWalkGUI(root=root)
        apply_overview_view_mode(app, VIEW_MODE_SIDE_BY_SIDE)
        select_dashboard_tab(app, TAB_OVERVIEW)
        for width, height in ((1920, 1080), (1680, 900), (1366, 768), (1280, 720)):
            root.geometry(f"{width}x{height}+40+40")
            root.update_idletasks()
            root.update()
            apply_responsive_layout(app, width=width, height=height)
            root.update_idletasks()
            root.update()
            video = app.video_frame
            skel = app.skel_frame
            path = app.overview_traj_panel
            transport = app._transport_bar
            assert video.winfo_ismapped()
            assert skel.winfo_ismapped()
            assert path.winfo_ismapped()
            assert transport.winfo_ismapped()
            assert int(video.winfo_width()) >= OVERVIEW_VIDEO_MIN_W
            assert int(skel.winfo_width()) >= OVERVIEW_SKELETON_MIN_W
            assert int(path.winfo_width()) >= OVERVIEW_PATH_MIN_W
            # Honor the real client height: a requested 1920x1080 may be
            # clamped by the display, and chrome + Foot Clearance must leave
            # the playback bar visible.
            actual_h = int(root.winfo_height() or height)
            height_floor = 140 if actual_h <= 900 else OVERVIEW_VIDEO_ROW_SOFT_MIN_H
            assert int(video.winfo_height()) >= height_floor
            assert int(skel.winfo_height()) >= height_floor
            assert int(path.winfo_height()) >= height_floor
            tb = transport.winfo_rooty() + transport.winfo_height()
            rb = root.winfo_rooty() + root.winfo_height()
            assert tb <= rb + 4
            assert str(app._overview_viz_layout) == "three_col"
            phase = getattr(app, "_overview_phase_strip", None)
            clearance = getattr(app, "_foot_clearance_detail_host", None)
            assert phase is not None and phase.winfo_ismapped()
            assert clearance is not None and clearance.winfo_ismapped()
            cards = getattr(app, "_overview_clinical_cards", None) or {}
            widths = []
            for card in cards.values():
                frame = card.get("frame")
                if frame is not None and frame.winfo_ismapped():
                    widths.append(int(frame.winfo_width()))
            if len(widths) >= 4:
                assert max(widths) - min(widths) <= 24
            clearance_h = int(clearance.winfo_height())
            assert clearance_h >= OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT
            # Labels / values must not be clipped inside the reserved row.
            try:
                req_h = int(clearance.winfo_reqheight() or 0)
            except tk.TclError:
                req_h = 0
            if req_h > 0:
                assert clearance_h + 4 >= min(req_h, OVERVIEW_FOOT_CLEARANCE_MIN_H_SHORT + 24)
            readout = getattr(app, "overview_traj_readout", None)
            toggle = getattr(app, "_overview_inspector_toggle", None)
            inspector_layout = str(
                getattr(app, "_overview_path_inspector_layout", "") or ""
            )
            if inspector_layout == "collapsed":
                assert toggle is not None and toggle.winfo_ismapped()
                assert int(toggle.winfo_width()) >= 160
            else:
                assert readout is not None and readout.winfo_ismapped()
                assert int(readout.winfo_width()) >= 160
            clinical = getattr(app, "_overview_clinical_summary", None)
            host = getattr(app, "_overview_content", None) or getattr(
                app, "_overview_layout_host", None
            )
            assert clinical is not None and clinical.winfo_ismapped()
            assert int(clinical.winfo_height()) >= 72
            if host is not None:
                assert (
                    int(clinical.winfo_y()) + int(clinical.winfo_height())
                    <= int(host.winfo_height()) + 4
                )
            for card in cards.values():
                frame = card.get("frame")
                if frame is None or not frame.winfo_ismapped():
                    continue
                assert int(frame.winfo_height()) >= 48
                assert (
                    int(frame.winfo_y()) + int(frame.winfo_height())
                    <= int(clinical.winfo_height()) + 8
                )
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_coordinate_inspector_wraps_without_nested_scrollbar() -> None:
    root = _tk_root()
    try:
        from stablewalk.ui.overview_coordinate_inspector import COORDINATE_INSPECTOR_TITLES
        from stablewalk.ui.tk.dashboard_layout import (
            _build_overview_traj_readout,
            _install_coord_inspector_scroll,
        )
        from stablewalk.ui.tk.overview_responsive import PATH_INSPECTOR_STACK_H

        host = tk.Frame(root, width=200, height=PATH_INSPECTOR_STACK_H, bg="#111")
        host.grid_propagate(False)
        host.grid()
        host.columnconfigure(0, weight=1)
        host.rowconfigure(0, weight=1)
        inner, host_frame = _install_coord_inspector_scroll(host)
        values = _build_overview_traj_readout(inner)
        root.update_idletasks()
        inner.event_generate("<Configure>")
        root.update_idletasks()

        accel = values["acceleration"]
        wrap = int(float(str(accel.cget("wraplength") or 0)))
        assert wrap >= 56
        assert accel.cget("text")
        # Long metric names stay single-line; the inspector rail is wide enough.
        assert "Acceleration" in COORDINATE_INSPECTOR_TITLES.values()
        name_wraps = []
        for child in accel.master.winfo_children():
            try:
                if str(child.cget("text") or "") == "Acceleration":
                    name_wraps.append(int(float(str(child.cget("wraplength") or 0))))
            except tk.TclError:
                continue
        assert name_wraps and name_wraps[0] == 0
        assert inner is host_frame
        assert getattr(inner, "_coord_inspector_scrollbar", None) is None
        assert inner.winfo_class() == "Frame"
    finally:
        root.destroy()


def test_path_inspector_stacks_instead_of_clipping() -> None:
    root = _tk_root()
    try:
        from stablewalk.ui.tk.overview_responsive import (
            PATH_INSPECTOR_SIDE_BY_SIDE_MIN,
            PATH_INSPECTOR_SIDE_MIN_W,
            reflow_overview_path_inspector,
        )

        body = tk.Frame(root, width=900, height=300)
        body.grid()
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        plot = tk.Frame(body, width=400, height=200)
        readout = tk.Frame(body, width=80, height=200)
        plot.grid(row=0, column=0, sticky="nsew")
        readout.grid(row=0, column=1, sticky="nsew")
        gui = SimpleNamespace(
            overview_traj_path_body=body,
            overview_traj_canvas_host=plot,
            overview_traj_readout=readout,
            _overview_viz_layout="two_plus_one",
        )
        mode = reflow_overview_path_inspector(
            gui, panel_width=PATH_INSPECTOR_SIDE_BY_SIDE_MIN - 40, layout="two_plus_one"
        )
        assert mode == "collapsed"
        toggle = getattr(gui, "_overview_inspector_toggle", None)
        assert toggle is not None
        assert int(toggle.grid_info()["row"]) == 1
        gui._overview_inspector_expanded = True
        mode = reflow_overview_path_inspector(
            gui, panel_width=PATH_INSPECTOR_SIDE_BY_SIDE_MIN - 40, layout="two_plus_one"
        )
        assert mode == "stacked"
        assert int(readout.grid_info()["row"]) == 2
        assert int(readout.grid_info()["columnspan"]) == 2

        mode = reflow_overview_path_inspector(
            gui, panel_width=PATH_INSPECTOR_SIDE_BY_SIDE_MIN + 40, layout="two_plus_one"
        )
        assert mode == "side"
        assert int(readout.grid_info()["row"]) == 0
        assert int(readout.grid_info()["column"]) == 1
        assert int(body.grid_columnconfigure(1)["minsize"]) >= PATH_INSPECTOR_SIDE_MIN_W
        from stablewalk.ui.tk.overview_responsive import PATH_PLOT_SIDE_MIN_W

        assert int(body.grid_columnconfigure(0)["minsize"]) >= PATH_PLOT_SIDE_MIN_W

        # Shared three-col row stacks under the plot and starts collapsed so
        # Skeleton keeps the full video-row height.
        mode = reflow_overview_path_inspector(
            gui, panel_width=PATH_INSPECTOR_SIDE_BY_SIDE_MIN - 40, layout="three_col"
        )
        assert mode == "collapsed"
        assert str(readout.winfo_manager()) == ""
        toggle = getattr(gui, "_overview_inspector_toggle", None)
        assert toggle is not None
        assert int(toggle.grid_info()["row"]) == 1
        from stablewalk.ui.tk.overview_responsive import PATH_CANVAS_MINSIZE_WIDE

        assert int(body.grid_rowconfigure(0)["minsize"]) >= PATH_CANVAS_MINSIZE_WIDE
    finally:
        root.destroy()


def test_overview_host_reserves_kpi_card_height() -> None:
    from stablewalk.ui.tk.analysis_split import MIN_H_SECONDARY_INFO
    from stablewalk.ui.tk.dashboard_sections import (
        OVERVIEW_HOST_KPI_ROW,
        OVERVIEW_HOST_VIZ_ROW,
        apply_overview_host_row_weights,
        overview_video_row_min_h,
    )

    root = _tk_root()
    try:
        host = tk.Frame(root)
        apply_overview_host_row_weights(host)
        assert int(host.grid_rowconfigure(OVERVIEW_HOST_VIZ_ROW)["weight"]) == 1
        assert int(host.grid_rowconfigure(OVERVIEW_HOST_KPI_ROW)["weight"]) == 0
        assert int(host.grid_rowconfigure(OVERVIEW_HOST_KPI_ROW)["minsize"] or 0) == 0
        viz_floor = overview_video_row_min_h() + int(MIN_H_SECONDARY_INFO)
        assert int(host.grid_rowconfigure(OVERVIEW_HOST_VIZ_ROW)["minsize"]) >= viz_floor
        # Tight host: splitter floor fits leftover after the header.
        apply_overview_host_row_weights(host, short=True, host_h=500, kpi_min=80)
        viz_min = int(host.grid_rowconfigure(OVERVIEW_HOST_VIZ_ROW)["minsize"])
        assert viz_min <= 500
        assert viz_min >= overview_video_row_min_h(short=True)
    finally:
        root.destroy()