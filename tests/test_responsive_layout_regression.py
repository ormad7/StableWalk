"""Live resize regression: clip, overlap, missing panels, Compare, scroll, playback."""

from __future__ import annotations

from typing import Any

import pytest

TARGET_SIZES = (
    (1920, 1080),
    (1600, 900),
    (1366, 768),
    (1280, 720),
)


def _tk_or_skip():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    return root


def _mapped(widget: Any) -> bool:
    if widget is None:
        return False
    try:
        return bool(widget.winfo_ismapped()) and int(widget.winfo_width() or 0) > 1
    except Exception:
        return False


def _size(widget: Any) -> tuple[int, int]:
    try:
        return int(widget.winfo_width() or 0), int(widget.winfo_height() or 0)
    except Exception:
        return 0, 0


def _box(widget: Any) -> tuple[int, int, int, int] | None:
    try:
        widget.update_idletasks()
        x, y = int(widget.winfo_rootx()), int(widget.winfo_rooty())
        w, h = int(widget.winfo_width()), int(widget.winfo_height())
        if w <= 1 or h <= 1:
            return None
        return (x, y, x + w, y + h)
    except Exception:
        return None


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return (a[0] + 2) < b[2] and (b[0] + 2) < a[2] and (a[1] + 2) < b[3] and (b[1] + 2) < a[3]


def _visible_vertical_scrollbars(widget: Any, found: list[Any] | None = None) -> list[Any]:
    from tkinter import ttk

    found = found if found is not None else []
    if widget is None:
        return found
    try:
        if isinstance(widget, ttk.Scrollbar):
            if str(widget.cget("orient") or "vertical") == "vertical" and widget.winfo_ismapped():
                found.append(widget)
        for child in widget.winfo_children():
            _visible_vertical_scrollbars(child, found)
    except Exception:
        return found
    return found


def _collect_issues(app: Any, *, width: int, height: int, tab: str) -> list[str]:
    issues: list[str] = []
    prefix = f"{width}x{height} {tab}"

    transport = getattr(app, "_transport_bar", None) or getattr(app, "_transport_row", None)
    if tab == "Compare":
        ctrl = getattr(app, "_comparison_mode", None)
        compare_transport = getattr(ctrl, "_transport_host", None) if ctrl else None
        if _mapped(compare_transport):
            transport = compare_transport
    if not _mapped(transport):
        issues.append(f"{prefix}: playback controls not mapped")
    else:
        tb = _box(transport)
        rb = _box(app.root)
        if tb and rb and tb[3] > rb[3] + 4:
            issues.append(f"{prefix}: playback controls clipped below window")
        play = getattr(app, "btn_play", None)
        slider = getattr(app, "slider", None)
        if tab != "Compare":
            if play is not None and not _mapped(play):
                issues.append(f"{prefix}: Play button hidden")
            if slider is not None and not _mapped(slider):
                issues.append(f"{prefix}: timeline slider hidden")

    if tab == "Overview":
        video = getattr(app, "video_frame", None)
        skel = getattr(app, "skel_frame", None)
        path = getattr(app, "overview_traj_panel", None)
        for name, widget, min_w, min_h in (
            ("Video", video, 140, 130),
            ("Skeleton", skel, 140, 130),
            ("3D Path", path, 180 if width >= 1366 else 140, 130),
        ):
            if not _mapped(widget):
                issues.append(f"{prefix}: {name} panel hidden")
                continue
            w, h = _size(widget)
            if w < min_w or h < min_h:
                issues.append(f"{prefix}: {name} unusable ({w}x{h})")
        boxes = [(n, _box(w)) for n, w in (("Video", video), ("Skeleton", skel), ("Path", path))]
        visible = [(n, b) for n, b in boxes if b]
        for i, (na, ba) in enumerate(visible):
            for nb, bb in visible[i + 1 :]:
                if _overlaps(ba, bb):
                    issues.append(f"{prefix}: {na} overlaps {nb}")

    if tab == "Kinematics":
        tab = "Motion"

    if tab == "Motion":
        host = getattr(app, "motion_graph_host", None) or getattr(
            app, "_motion_graphs_panel", None
        )
        if host is None or not _mapped(host):
            issues.append(f"{prefix}: motion graph host hidden")
        else:
            w, h = _size(host)
            # 768 includes window chrome; the graph host is shorter than the
            # advertised window height.
            min_h = 140 if height > 768 else 110
            if w < 200 or h < min_h:
                issues.append(f"{prefix}: motion graph host unusable ({w}x{h})")
        canvas = getattr(app, "canvas_motion_workspace", None)
        if canvas is not None:
            try:
                widget = canvas.get_tk_widget()
            except Exception:
                widget = None
            if widget is not None and _mapped(widget):
                w, h = _size(widget)
                if w < 40 or h < 40:
                    issues.append(f"{prefix}: motion graph canvas disappeared ({w}x{h})")

    if tab == "Biomechanics":
        chart = getattr(app, "_biomech_chart_panel", None)
        if chart is None or not _mapped(chart):
            issues.append(f"{prefix}: biomech chart hidden")
        else:
            w, h = _size(chart)
            min_h = 120 if height >= 768 else 90
            if w < 180 or h < min_h:
                issues.append(f"{prefix}: biomech chart unusable ({w}x{h})")
        body = getattr(app, "_biomech_body", None)
        sidebar = getattr(app, "_biomech_sidebar", None)
        if (
            body is not None
            and sidebar is not None
            and _mapped(body)
            and _mapped(sidebar)
            and not bool(getattr(app, "_biomech_stacked", False))
        ):
            _bw, bh = _size(body)
            _sw, sh = _size(sidebar)
            if sh > bh + 24:
                issues.append(f"{prefix}: Key Findings taller than body ({sh}>{bh})")

    if tab == "Results":
        for attr, label in (
            ("btn_export_summary_report", "Export Report"),
            ("btn_export_summary_json", "Export JSON"),
        ):
            btn = getattr(app, attr, None)
            if btn is not None and not _mapped(btn):
                issues.append(f"{prefix}: {label} hidden")

    if tab == "Compare":
        research = getattr(app, "_research_host", None)
        if _mapped(research):
            w, h = _size(research)
            if w < 240 or h < 80:
                issues.append(f"{prefix}: Research Compare workspace crushed ({w}x{h})")
            empty = getattr(app, "lbl_research_empty", None)
            if empty is not None and not _mapped(empty):
                # Empty state may lower once both videos analyze; at startup it
                # should remain visible.
                if getattr(app, "_research_comparison", None) is None:
                    issues.append(f"{prefix}: Compare empty state hidden")
        else:
            ctrl = getattr(app, "_comparison_mode", None)
            workspace = getattr(ctrl, "_compare_workspace", None) if ctrl else None
            if workspace is None or not _mapped(workspace):
                issues.append(f"{prefix}: Compare workspace hidden")
            else:
                w, h = _size(workspace)
                if w < 240 or h < 80:
                    issues.append(f"{prefix}: Compare workspace crushed ({w}x{h})")
                stacked = bool(getattr(ctrl, "_compare_stacked", False))
                col_count = 2 if stacked else 3
                for col in range(col_count):
                    if int(workspace.grid_columnconfigure(col).get("weight") or 0) < 1:
                        issues.append(f"{prefix}: Compare column {col} has no stretch weight")
            for attr, label in (
                ("_session_a_column", "Video A"),
                ("_session_b_column", "Video B"),
                ("_skel_a_panel", "Skeleton A"),
                ("_skel_b_panel", "Skeleton B"),
                ("_diff_graph_panel", "Difference Graph"),
                ("_traj_panel", "Difference Trajectory"),
                ("_heat_panel", "Difference Heatmap"),
                ("_traj_heat_panel", "Trajectory / Heatmap"),
                ("_compare_metrics_host", "Comparison Metrics"),
            ):
                widget = getattr(ctrl, attr, None) if ctrl else None
                if widget is not None and not _mapped(widget):
                    issues.append(f"{prefix}: {label} hidden")

    # Page-level vertical scrollbars (ignore Treeview internals by requiring
    # the scrollbar to sit next to a Canvas).
    page_scrolls = 0
    tab_widget = {
        "Overview": getattr(app, "_tab_overview", None),
        "Motion": getattr(app, "_tab_motion", None),
        "Biomechanics": getattr(app, "_tab_biomechanics", None),
        "Results": getattr(app, "_tab_results_summary", None),
        "Compare": getattr(app, "_tab_compare", None),
        "Advanced": getattr(app, "_tab_advanced_outer", None),
    }.get(tab)
    if tab_widget is not None:
        for bar in _visible_vertical_scrollbars(tab_widget):
            parent = getattr(bar, "master", None)
            siblings = []
            try:
                siblings = list(parent.winfo_children()) if parent is not None else []
            except Exception:
                pass
            if any(type(s).__name__ == "Canvas" for s in siblings):
                page_scrolls += 1
        if tab != "Advanced" and page_scrolls > 0:
            # Compare / others may show overflow scroll only when needed.
            if tab == "Compare":
                needs = bool(getattr(getattr(app, "_comparison_mode", None), "_compare_needs_scroll", False))
                if page_scrolls > 1 or (page_scrolls == 1 and not needs and height >= 720):
                    issues.append(f"{prefix}: unexpected Compare page scrollbar ({page_scrolls})")
            elif tab == "Biomechanics":
                # Key Findings and Clinical Interpretation each have a local
                # overflow scroller — that is internal rail scroll, not a
                # nested page scrollbar.
                if page_scrolls > 2:
                    issues.append(f"{prefix}: nested page scrollbars ({page_scrolls})")
            elif page_scrolls > 1:
                issues.append(f"{prefix}: nested page scrollbars ({page_scrolls})")
        if tab == "Advanced" and page_scrolls > 1:
            issues.append(f"{prefix}: nested Advanced scrollbars ({page_scrolls})")

    return issues


def _collect_named_analysis_tab_issues(app: Any, width: int, height: int) -> list[str]:
    """Kinematics, Four Motion Signals, and Spectral Analysis at laptop / FHD."""
    from stablewalk.ui.tk.gait_correlation_tabs import (
        select_gait_corr_tab,
        select_notebook_tab,
    )

    issues: list[str] = []
    prefix = f"{width}x{height}"
    root = getattr(app, "root", None)
    motion_nb = getattr(app, "_motion_page_notebook", None)

    def _refresh() -> None:
        if root is None:
            return
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            pass

    if not select_notebook_tab(motion_nb, "Kinematics"):
        issues.append(f"{prefix}: Kinematics inner tab missing")
    else:
        _refresh()
        issues.extend(_collect_issues(app, width=width, height=height, tab="Motion"))

    if not select_gait_corr_tab(app, "Four Motion Signals"):
        issues.append(f"{prefix}: Four Motion Signals tab missing")
    else:
        _refresh()
        host = getattr(app, "_four_gait_graph_host", None)
        if host is None or not _mapped(host):
            issues.append(f"{prefix}: Four Motion graph host hidden")
        else:
            w, h = _size(host)
            min_h = 120 if height >= 768 else 90
            if w < 200 or h < min_h:
                issues.append(f"{prefix}: Four Motion graph unusable ({w}x{h})")
        canvas = getattr(app, "canvas_four_gait", None)
        if canvas is not None:
            try:
                widget = canvas.get_tk_widget()
            except Exception:
                widget = None
            if widget is not None and _mapped(widget):
                w, h = _size(widget)
                if w < 40 or h < 40:
                    issues.append(f"{prefix}: Four Motion canvas disappeared ({w}x{h})")

    if not select_notebook_tab(motion_nb, "Spectral Analysis"):
        issues.append(f"{prefix}: Spectral Analysis inner tab missing")
    else:
        _refresh()
        host = getattr(app, "_spectral_graph_host", None)
        if host is None or not _mapped(host):
            issues.append(f"{prefix}: Spectral graph host hidden")
        else:
            w, h = _size(host)
            min_h = 120 if height >= 768 else 90
            if w < 200 or h < min_h:
                issues.append(f"{prefix}: Spectral graph unusable ({w}x{h})")
        tree = getattr(app, "_spectral_tree", None)
        if tree is not None and _mapped(tree):
            w, h = _size(tree)
            if w < 80 or h < 40:
                issues.append(f"{prefix}: Spectral numerical table unusable ({w}x{h})")

    select_notebook_tab(motion_nb, "Kinematics")
    _refresh()
    return issues


def test_live_resize_keeps_panels_playback_and_compare_usable() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI
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
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    root = _tk_or_skip()
    issues: list[str] = []
    try:
        app = StableWalkGUI(root=root)
        root.update_idletasks()
        tabs = [
            (TAB_OVERVIEW, "Overview"),
            (TAB_MOTION, "Motion"),
            (TAB_BIOMECHANICS, "Biomechanics"),
            (TAB_RESULTS_SUMMARY, "Results"),
        ]
        if ENABLE_COMPARE_TAB:
            tabs.append((TAB_COMPARE, "Compare"))
        tabs.append((TAB_ADVANCED, "Advanced"))
        tabs = tuple(tabs)
        for width, height in TARGET_SIZES:
            root.geometry(f"{width}x{height}+40+40")
            root.update_idletasks()
            root.update()
            apply_responsive_layout(app, width=width, height=height)
            root.update_idletasks()
            root.update()
            for tab_id, tab_name in tabs:
                select_dashboard_tab(app, tab_id)
                root.update_idletasks()
                root.update()
                apply_responsive_layout(app, width=width, height=height)
                root.update_idletasks()
                issues.extend(_collect_issues(app, width=width, height=height, tab=tab_name))
                if tab_name == "Motion" and (width, height) in ((1366, 768), (1920, 1080)):
                    issues.extend(_collect_named_analysis_tab_issues(app, width, height))
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable during live resize: {exc}")
    finally:
        try:
            root.destroy()
        except Exception:
            pass
    assert not issues, "\n".join(issues)
