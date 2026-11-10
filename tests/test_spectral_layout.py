"""Spectral Analysis split layout, plot identity, and rendered-size checks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from stablewalk.ui.tk.dashboard_spectral import (
    SPECTRAL_PLOT_SPECS,
    spectral_figure_min_px,
    spectral_split_minsizes,
)
from stablewalk.ui.viewers.spectral_charts import (
    SPECTRAL_SINGLE_MARGINS,
    draw_spectral_plot,
    layout_spectral_figure,
    render_spectral_figure,
)


ROOT = Path(__file__).resolve().parents[1]
SHOT_DIR = ROOT / "docs" / "spectral_layout_verify"


def test_spectral_minsizes_leave_room_for_plots_and_table() -> None:
    plot_min, table_min = spectral_split_minsizes(1.0)
    assert plot_min >= 520
    assert table_min >= 200
    assert plot_min + table_min <= 900
    scaled_plot, scaled_table = spectral_split_minsizes(1.5)
    assert scaled_plot >= plot_min
    assert scaled_table >= table_min
    fig_w, fig_h = spectral_figure_min_px(1.0)
    assert fig_w >= 720
    assert fig_h >= 740


def test_six_plots_are_named_and_distinct() -> None:
    ids = [pid for pid, _title in SPECTRAL_PLOT_SPECS]
    assert ids == ["signal", "acf", "psd", "lr_psd", "r0", "eigen"]
    titles = " ".join(title.lower() for _pid, title in SPECTRAL_PLOT_SPECS)
    assert "acf" in titles
    assert "psd" in titles
    assert "r(0)" in titles
    assert "λ" in titles or "eigen" in titles


def _fake_report(channel_id: str = "dof:left_knee_flexion"):
    t = np.linspace(0.0, 4.0, 201)
    y = np.sin(2.0 * np.pi * 1.0 * t)
    series = SimpleNamespace(
        times_s=t,
        raw=y,
        processed=y - y.mean(),
        spec=SimpleNamespace(units="deg", dof_id="left_knee_flexion"),
    )
    ac = SimpleNamespace(
        lags_s=np.linspace(0.0, 2.0, 41),
        rho=np.linspace(1.0, 0.0, 41),
        peaks=[],
        stride_period_s=1.0,
        step_period_s=0.5,
    )
    sp = SimpleNamespace(
        freq_hz=np.linspace(0.0, 8.0, 65),
        psd=np.ones(65),
        dominant_hz=1.0,
        harmonic_hz=[2.0],
    )
    labels = [
        "dof:left_hip_flexion",
        "dof:right_hip_flexion",
        "dof:left_knee_flexion",
        "dof:right_knee_flexion",
        "dof:left_ankle_flexion",
        "dof:right_ankle_flexion",
    ]
    corr = np.eye(len(labels))
    return SimpleNamespace(
        available=True,
        bundle=SimpleNamespace(get=lambda cid, _s=series: _s if cid else None),
        autocorr={channel_id: ac},
        spectra={
            channel_id: sp,
            "dof:left_knee_flexion": sp,
            "dof:right_knee_flexion": sp,
        },
        config=SimpleNamespace(gait_band_hz=(0.4, 4.0)),
        lr_compare=[
            SimpleNamespace(
                pair_id="knee",
                left_id="dof:left_knee_flexion",
                right_id="dof:right_knee_flexion",
            )
        ],
        matrices=SimpleNamespace(
            r0=SimpleNamespace(correlation=corr),
            channel_ids=labels,
        ),
        eigen={"correlation": SimpleNamespace(eigenvalues=np.array([3.2, 1.1, 0.8]))},
    )


def test_rendered_titles_distinguish_r0_acf_psd_and_eigenvalues() -> None:
    fig = Figure(figsize=(9.6, 8.0), dpi=100)
    axes = fig.subplots(3, 2)
    report = _fake_report()
    render_spectral_figure(fig, axes.ravel(), report, "dof:left_knee_flexion")
    titles = [ax.get_title() for ax in axes.ravel()]
    joined = " | ".join(titles)
    assert "Temporal ACF" in joined
    assert "Welch PSD" in joined
    assert "R(0) inter-joint correlation" in joined
    assert "Eigenvalues of R(0)" in joined
    assert titles[1] != titles[4]
    assert titles[2] != titles[5]
    ylabels = [str(ax.get_ylabel()) for ax in axes.ravel()]
    assert any("ρ(τ)" in lab for lab in ylabels)
    assert any("z²/Hz" in lab or "z²/Hz" in lab for lab in ylabels)
    assert any("λ of R(0)" in lab for lab in ylabels)
    assert any("deg" in lab.lower() or "flexion" in lab.lower() for lab in ylabels)
    assert axes[0, 0]._sw_time_seek is True
    assert axes[0, 1]._sw_time_seek is False
    assert axes[1, 0]._sw_nav_xlim is False
    assert axes[2, 0]._sw_nav_xlim is False


def test_expanded_plots_keep_full_chrome() -> None:
    report = _fake_report()
    for plot_id in ("signal", "acf", "psd", "lr_psd", "r0", "eigen"):
        fig = Figure(figsize=(10.0, 6.4), dpi=100)
        ax = fig.add_subplot(111)
        draw_spectral_plot(ax, plot_id, report, "dof:left_knee_flexion", compact=False)
        layout_spectral_figure(fig, single=True)
        title = ax.get_title()
        xlabel = str(ax.get_xlabel())
        ylabel = str(ax.get_ylabel())
        assert title, plot_id
        assert xlabel, plot_id
        assert ylabel, plot_id
        joined = f"{title} {xlabel} {ylabel}"
        assert any(
            token in joined
            for token in ("s", "Hz", "deg", "z²", "ρ", "λ", "r", "dimensionless", "Joint")
        ), joined
        if plot_id in {"signal", "acf", "psd", "lr_psd"}:
            legend = ax.get_legend()
            assert legend is not None, plot_id
            assert legend.get_texts()
        assert fig.subplotpars.top == SPECTRAL_SINGLE_MARGINS["top"]
        assert fig.subplotpars.right == SPECTRAL_SINGLE_MARGINS["right"]
        assert fig.subplotpars.bottom == SPECTRAL_SINGLE_MARGINS["bottom"]


def test_expanded_acf_may_zoom_without_seeking() -> None:
    fig = Figure(figsize=(8.0, 5.0), dpi=100)
    ax = fig.add_subplot(111)
    draw_spectral_plot(ax, "acf", _fake_report(), "dof:left_knee_flexion", compact=False)
    assert ax._sw_time_seek is False
    assert ax._sw_nav_xlim is True
    assert "Temporal ACF" in ax.get_title()


def test_layout_uses_subplots_adjust_not_tight_layout() -> None:
    src = (
        ROOT / "stablewalk" / "ui" / "viewers" / "spectral_charts.py"
    ).read_text(encoding="utf-8")
    assert "tight_layout(" not in src
    fig = Figure(figsize=(9.6, 8.0), dpi=100)
    fig.subplots(3, 2)
    layout_spectral_figure(fig)
    assert fig.subplotpars.hspace >= 0.50


def test_navigation_skips_opted_out_spectral_axes() -> None:
    from stablewalk.ui.viewers.chart_navigation import _time_axes

    fig = Figure(figsize=(6.0, 4.0), dpi=80)
    ax_time = fig.add_subplot(2, 1, 1)
    ax_psd = fig.add_subplot(2, 1, 2)
    ax_time._sw_nav_xlim = True
    ax_psd._sw_nav_xlim = False
    axes = _time_axes(fig)
    assert ax_time in axes
    assert ax_psd not in axes


def test_workspace_split_at_laptop_and_desktop(tmp_path: Path) -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_spectral import (
        build_spectral_analysis_workspace,
        open_spectral_plot_window,
        relayout_spectral_plots,
        render_spectral_workspace,
    )
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor

    try:
        root = tk.Tk()
    except tk.TclError:
        return
    root.withdraw()
    shots = SHOT_DIR
    shots.mkdir(parents=True, exist_ok=True)
    scale = dpi_scale_factor(root)
    try:
        gui = SimpleNamespace(root=root, _motion_page_notebook=None, _layout_dpi_scale=1.0)
        parent = tk.Frame(root)
        parent.pack(fill=tk.BOTH, expand=True)
        host = build_spectral_analysis_workspace(gui, parent)
        assert getattr(gui, "_spectral_split", None) is not None
        assert getattr(gui, "_spectral_tree", None) is not None
        assert len(gui._spectral_axes) == 6
        from test_spectral_analysis import _sine_recording
        from stablewalk.analysis.spectral.report import analyze_spectral

        gui._spectral_report = analyze_spectral(_sine_recording())
        gui._spectral_channel_id = "dof:left_knee_flexion"
        render_spectral_workspace(gui)

        measurements: dict[str, dict[str, int | float]] = {}
        parent.pack_forget()
        for width, height in ((1366, 768), (1920, 1080)):
            root.geometry(f"{width}x{height}")
            parent.place(x=0, y=0, width=width, height=height)
            root.deiconify()
            root.update_idletasks()
            root.update()
            place = getattr(gui, "_place_spectral_sash", None)
            if callable(place):
                gui._spectral_sash_for_width = None
                place()
            root.update_idletasks()
            relayout_spectral_plots(gui)
            root.update_idletasks()
            plot_w = int(gui._spectral_plot_pane.winfo_width() or 0)
            side_w = int(gui._spectral_side.winfo_width() or 0)
            host_w = int(host.winfo_width() or 0)
            host_h = int(host.winfo_height() or 0)
            graph_w = int(gui._spectral_graph_host.winfo_width() or 0)
            graph_h = int(gui._spectral_graph_host.winfo_height() or 0)
            fig_w_px = int(round(gui.fig_spectral.get_size_inches()[0] * gui.fig_spectral.get_dpi()))
            fig_h_px = int(round(gui.fig_spectral.get_size_inches()[1] * gui.fig_spectral.get_dpi()))
            plot_min, table_min = spectral_split_minsizes(1.0)
            fig_min_w, fig_min_h = spectral_figure_min_px(1.0)
            scroll = getattr(gui, "_spectral_scroll_canvas", None)
            scrollregion = ()
            if scroll is not None:
                try:
                    scrollregion = scroll.cget("scrollregion")
                except tk.TclError:
                    scrollregion = ()
            measurements[f"{width}x{height}"] = {
                "host_w": host_w,
                "host_h": host_h,
                "plot_w": plot_w,
                "side_w": side_w,
                "graph_w": graph_w,
                "graph_h": graph_h,
                "fig_w": fig_w_px,
                "fig_h": fig_h_px,
                "dpi_scale": scale,
            }
            assert host_w >= width - 8
            assert host_h >= height - 8
            assert plot_w >= min(plot_min, max(200, host_w - table_min - 20))
            assert side_w >= min(table_min, 180)
            assert plot_w > side_w
            assert fig_w_px >= fig_min_w - 2
            assert fig_h_px >= fig_min_h - 2
            dest = shots / f"spectral_{width}x{height}.png"
            gui.fig_spectral.savefig(dest, dpi=100, facecolor=gui.fig_spectral.get_facecolor())
            assert dest.is_file() and dest.stat().st_size > 1000
            try:
                from PIL import ImageGrab

                root.update()
                rx, ry = int(root.winfo_rootx()), int(root.winfo_rooty())
                grab = ImageGrab.grab(bbox=(rx, ry, rx + width, ry + height))
                ui_dest = shots / f"spectral_ui_{width}x{height}.png"
                grab.save(ui_dest)
                measurements[f"{width}x{height}"]["ui_shot"] = str(ui_dest)
            except Exception:
                measurements[f"{width}x{height}"]["ui_shot"] = "unavailable"
            if graph_h < fig_min_h:
                assert str(scrollregion)

        open_spectral_plot_window(gui, "r0")
        root.update_idletasks()
        record = gui._spectral_expand_by_id["r0"]
        assert "R(0)" in record["ax"].get_title()
        assert "Joint" in str(record["ax"].get_xlabel())
        assert record["window"].winfo_exists()
        assert int(record["window"].winfo_width() or 0) >= 700
        assert int(record["window"].winfo_height() or 0) >= 460
        record["window"].destroy()
        tree = gui._spectral_tree
        assert str(tree.cget("xscrollcommand"))
        assert str(tree.column("meaning", "stretch")).lower() in {"1", "true"}
        assert str(tree.column("value", "stretch")).lower() in {"1", "true"}
        detail = getattr(gui, "lbl_spectral_row_detail", None)
        assert detail is not None
        assert int(float(str(detail.cget("wraplength") or 0))) >= 140
        split = gui._spectral_split
        gui._spectral_sash_frac = 0.55
        gui._spectral_sash_dragging = False
        place = getattr(gui, "_place_spectral_sash", None)
        assert callable(place)
        place()
        root.update_idletasks()
        width = max(1, int(split.winfo_width() or 0))
        sash = int(split.sash_coord(0)[0])
        assert abs(sash / width - 0.55) < 0.12
        assert str(gui.btn_spectral_methods.cget("text")) == "Show methods"
        assert str(gui.lbl_spectral_methods.winfo_manager() or "") == ""

        (shots / "measurements.json").write_text(
            __import__("json").dumps({"dpi_scale": scale, "sizes": measurements}, indent=2),
            encoding="utf-8",
        )
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
