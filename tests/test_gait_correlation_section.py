"""Gait Correlation & Spectral Analysis section wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from stablewalk.ui.tk.gait_correlation_tabs import (
    ASYMMETRY_TAB,
    FOUR_GAIT_TAB_VIEWS,
    SECTION_TAB_ORDER,
    SECTION_TITLE,
    recording_cache_key,
    select_gait_corr_tab,
)


def test_section_has_six_named_subtabs() -> None:
    assert SECTION_TITLE == "Gait Correlation & Spectral Analysis"
    assert list(FOUR_GAIT_TAB_VIEWS) == [
        "Four Motion Signals",
        "Correlation Matrix",
        "Autocorrelation & Cross-Correlation",
        "Frequency Analysis",
        "Spectral Correlation",
    ]
    assert FOUR_GAIT_TAB_VIEWS["Four Motion Signals"] == "time"
    assert FOUR_GAIT_TAB_VIEWS["Correlation Matrix"] == "matrix"
    assert FOUR_GAIT_TAB_VIEWS["Autocorrelation & Cross-Correlation"] == "corr"
    assert FOUR_GAIT_TAB_VIEWS["Frequency Analysis"] == "freq"
    assert FOUR_GAIT_TAB_VIEWS["Spectral Correlation"] == "compare"
    assert ASYMMETRY_TAB == "Asymmetry Summary"
    assert SECTION_TAB_ORDER == tuple(FOUR_GAIT_TAB_VIEWS) + (ASYMMETRY_TAB,)


def test_motion_layout_uses_one_gait_correlation_section() -> None:
    text = (
        Path(__file__).resolve().parents[1]
        / "stablewalk/ui/tk/dashboard_layout.py"
    ).read_text(encoding="utf-8")
    assert "build_gait_correlation_section" in text
    assert "Gait Correlation & Spectral Analysis" in text or "SECTION_TITLE" in text
    assert 'text="Four Gait Signals"' not in text
    assert "motion_pages.add(asym_page" not in text
    assert "build_four_gait_workspace(gui, four_page)" not in text
    assert "build_asymmetry_workspace(gui, asym_page)" not in text


def test_recording_cache_key_is_stable_per_video() -> None:
    rec = SimpleNamespace(
        snapshots=[1, 2, 3],
        frame_count=3,
        source="walk.mp4",
        fps=25.0,
    )
    first = recording_cache_key(rec)
    assert recording_cache_key(rec) == first
    other = SimpleNamespace(
        snapshots=[1, 2, 3],
        frame_count=3,
        source="other.mp4",
        fps=25.0,
    )
    assert recording_cache_key(other) != first
    assert recording_cache_key(None) == ("empty",)


def test_four_gait_schedule_recomputes_for_a_different_video() -> None:
    from stablewalk.ui.tk.dashboard_four_gait import schedule_four_gait_signals

    rec_a = SimpleNamespace(snapshots=[1], frame_count=1, source="a.mp4", fps=25.0)
    rec_b = SimpleNamespace(snapshots=[1, 2], frame_count=2, source="b.mp4", fps=25.0)

    class _Root:
        def after(self, _ms, _fn):
            return None

    gui = SimpleNamespace(
        gait_motion=rec_b,
        _four_gait_report=object(),
        _four_gait_cache_key=recording_cache_key(rec_a),
        _four_gait_token=4,
        _four_gait_worker=None,
        _four_gait_pending=None,
        _gait_corr_notebook=None,
        _motion_page_notebook=None,
        lbl_four_gait_status=None,
        playing=False,
        root=_Root(),
    )
    schedule_four_gait_signals(gui, force=False)
    assert gui._four_gait_token == 5
    assert gui._four_gait_cache_key == recording_cache_key(rec_b)


def test_playback_groups_keep_video_skeleton_and_gait_panels() -> None:
    from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION, TAB_OVERVIEW
    from stablewalk.ui.tk.playback_active_tab import TAB_PLAYBACK_GROUPS

    assert {"video", "skeleton", "overview"} <= set(TAB_PLAYBACK_GROUPS[TAB_OVERVIEW])
    assert {"four_gait", "asymmetry", "motion"} <= set(TAB_PLAYBACK_GROUPS[TAB_MOTION])


def test_compare_tab_and_single_3d_viewer_remain_wired() -> None:
    from pathlib import Path

    from stablewalk.ui.tk.dashboard_notebook import ENABLE_COMPARE_TAB

    assert ENABLE_COMPARE_TAB is True
    layout = (
        Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/dashboard_layout.py"
    ).read_text(encoding="utf-8")
    assert "build_comparison_tab" in layout
    assert "_ensure_skeleton_figure" in layout
    asym = (
        Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/dashboard_asymmetry.py"
    ).read_text(encoding="utf-8")
    assert "setup_skeleton_axes" not in asym
    four = (
        Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/dashboard_four_gait.py"
    ).read_text(encoding="utf-8")
    assert "update_playhead_on_axes" in four
    assert '_view(gui) != "time"' in four or '_view(gui) == "time"' in four


def test_app_clears_gait_cache_when_recording_resets() -> None:
    from pathlib import Path

    text = (
        Path(__file__).resolve().parents[1] / "stablewalk/ui/tk/app.py"
    ).read_text(encoding="utf-8")
    assert "self._four_gait_report = None" in text
    assert "self._four_gait_cache_key = None" in text
    assert "self._asymmetry_cache_key = None" in text


def test_four_gait_schedule_reuses_cached_report() -> None:
    from stablewalk.ui.tk.dashboard_four_gait import schedule_four_gait_signals

    rec = SimpleNamespace(
        snapshots=[1],
        frame_count=1,
        source="walk.mp4",
        fps=25.0,
    )
    gui = SimpleNamespace(
        gait_motion=rec,
        _four_gait_report=object(),
        _four_gait_cache_key=recording_cache_key(rec),
        _four_gait_token=4,
        _four_gait_worker=None,
        _gait_corr_notebook=None,
        _motion_page_notebook=None,
        fig_four_gait=None,
        _four_gait_axes=None,
        canvas_four_gait=None,
    )
    schedule_four_gait_signals(gui, force=False)
    assert gui._four_gait_token == 4


_AXES_BY_VIEW = {
    "time": 4,
    "matrix": 1,
    "corr": 6,
    "freq": 4,
    "compare": 5,
}


def _open_section():
    import tkinter as tk
    from tkinter import ttk

    from stablewalk.ui.tk.dashboard_gait_correlation import (
        build_gait_correlation_section,
    )

    try:
        root = tk.Tk()
    except tk.TclError:
        return None
    root.withdraw()
    root.geometry("1100x780")
    motion = ttk.Notebook(root)
    page = ttk.Frame(motion)
    motion.add(page, text=SECTION_TITLE)
    motion.pack(fill="both", expand=True)
    motion.select(page)
    gui = SimpleNamespace(
        root=root,
        _motion_page_notebook=motion,
        _layout_dpi_scale=1.0,
        gait_motion=None,
        sequence=None,
        playing=False,
    )
    parent = ttk.Frame(root)
    parent.pack(fill="both", expand=True)
    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)
    host = build_gait_correlation_section(gui, parent)
    root.update_idletasks()
    return root, gui, host


def _flush(root) -> None:
    try:
        root.update()
        root.update_idletasks()
    except Exception:
        pass


def _on_grid(widget) -> bool:
    try:
        return bool(widget.grid_info())
    except Exception:
        return False


def test_gait_correlation_section_builds_six_tabs() -> None:
    opened = _open_section()
    if opened is None:
        return
    root, gui, host = opened
    try:
        assert host is not None
        nb = gui._gait_corr_notebook
        titles = [str(nb.tab(tab_id, "text")) for tab_id in nb.tabs()]
        assert titles == list(SECTION_TAB_ORDER)
        assert gui._four_gait_use_section_tabs is True
        fig = gui.fig_four_gait
        assert fig is not None
        assert gui.fig_asymmetry is not None
        four_host = gui._gait_corr_four_host
        for title, view in FOUR_GAIT_TAB_VIEWS.items():
            assert select_gait_corr_tab(gui, title)
            _flush(root)
            assert gui._four_gait_view == view
            assert gui.fig_four_gait is fig
            assert gui._gait_corr_four_host is four_host
        assert select_gait_corr_tab(gui, ASYMMETRY_TAB)
        _flush(root)
        assert gui.fig_four_gait is fig
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_four_gait_tabs_show_view_specific_empty_states() -> None:
    from stablewalk.ui.tk.dashboard_four_gait import _EMPTY_BY_VIEW

    opened = _open_section()
    if opened is None:
        return
    root, gui, _host = opened
    try:
        empty = gui.lbl_four_gait_empty
        canvas_w = gui._four_gait_canvas_widget
        for title, view in FOUR_GAIT_TAB_VIEWS.items():
            assert select_gait_corr_tab(gui, title)
            _flush(root)
            assert gui._four_gait_view == view
            assert _on_grid(empty)
            assert not _on_grid(canvas_w)
            assert empty.cget("text") == _EMPTY_BY_VIEW[view]
        assert select_gait_corr_tab(gui, ASYMMETRY_TAB)
        _flush(root)
        asym_empty = gui.lbl_asymmetry_empty
        assert _on_grid(asym_empty)
        assert str(asym_empty.cget("text")).strip()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_time_tab_hides_ctau_side_panel() -> None:
    opened = _open_section()
    if opened is None:
        return
    root, gui, _host = opened
    try:
        paned = gui._four_gait_split
        side = gui._four_gait_side

        def _side_open() -> bool:
            try:
                return str(side) in [str(p) for p in paned.panes()]
            except Exception:
                return False

        assert select_gait_corr_tab(gui, "Four Motion Signals")
        _flush(root)
        assert not _side_open()
        assert select_gait_corr_tab(gui, "Correlation Matrix")
        _flush(root)
        assert _side_open()
        assert _on_grid(gui._four_gait_heat_host)
        assert select_gait_corr_tab(gui, "Autocorrelation & Cross-Correlation")
        _flush(root)
        assert _side_open()
        assert _on_grid(gui._four_gait_heat_host)
        assert select_gait_corr_tab(gui, "Four Motion Signals")
        _flush(root)
        assert not _side_open()
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_unavailable_report_shows_quality_warnings() -> None:
    opened = _open_section()
    if opened is None:
        return
    root, gui, _host = opened
    try:
        rec = SimpleNamespace(
            snapshots=[1],
            frame_count=1,
            source="short.mp4",
            fps=25.0,
        )
        gui.gait_motion = rec
        gui._four_gait_report = SimpleNamespace(
            available=False,
            warnings=["Need more kinematic frames", "Left hip visibility too low"],
        )
        gui._four_gait_cache_key = recording_cache_key(rec)
        assert select_gait_corr_tab(gui, "Correlation Matrix")
        _flush(root)
        text = str(gui.lbl_four_gait_empty.cget("text"))
        assert "Need more kinematic frames" in text
        assert _on_grid(gui.lbl_four_gait_empty)
        assert not _on_grid(gui._four_gait_canvas_widget)
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def test_six_subtabs_paint_cached_report_when_cycled() -> None:
    from test_four_gait_signals import _walking_recording
    from stablewalk.analysis.four_gait_signals import analyze_four_gait_signals

    opened = _open_section()
    if opened is None:
        return
    root, gui, _host = opened
    try:
        rec = _walking_recording(fps=25.0, duration_s=2.0)
        report = analyze_four_gait_signals(rec)
        assert report.available
        gui.gait_motion = rec
        gui._four_gait_report = report
        gui._four_gait_cache_key = recording_cache_key(rec)
        fig = gui.fig_four_gait
        four_host = gui._gait_corr_four_host
        order = list(SECTION_TAB_ORDER) + list(reversed(SECTION_TAB_ORDER))
        for title in order:
            assert select_gait_corr_tab(gui, title)
            _flush(root)
            assert gui.fig_four_gait is fig
            assert gui._gait_corr_four_host is four_host
            view = FOUR_GAIT_TAB_VIEWS.get(title)
            if view is None:
                assert str(gui._gait_corr_asym_page.winfo_manager())
                continue
            assert gui._four_gait_view == view
            canvas_w = gui._four_gait_canvas_widget
            assert _on_grid(canvas_w), f"{title} canvas stayed hidden"
            assert not _on_grid(gui.lbl_four_gait_empty), f"{title} stayed empty"
            axes = list(gui._four_gait_axes or [])
            assert len(axes) == _AXES_BY_VIEW[view], (
                f"{title} expected {_AXES_BY_VIEW[view]} axes, got {len(axes)}"
            )
            assert any(ax.has_data() or ax.texts or ax.images for ax in axes), (
                f"{title} axes were blank"
            )
            tree = gui._four_gait_tree
            assert tree.get_children(), f"{title} table was empty"
            warn = str(gui.lbl_four_gait_warnings.cget("text"))
            assert warn.strip(), f"{title} had no status text"
    finally:
        try:
            root.destroy()
        except Exception:
            pass
