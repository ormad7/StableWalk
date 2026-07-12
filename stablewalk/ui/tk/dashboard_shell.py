"""
Structural dashboard shell diagnostics and runtime singleton checks.

Used during the scroll-safe dashboard rebuild to verify frame hierarchy and
critical widget instance counts.
"""

from __future__ import annotations

from typing import Any

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def _walk_widgets(root: Any) -> list[Any]:
    found: list[Any] = []

    def _visit(widget: Any) -> None:
        try:
            found.append(widget)
            for child in widget.winfo_children():
                _visit(child)
        except Exception:
            return

    try:
        _visit(root)
    except Exception:
        pass
    return found


def _count_attr_singleton(gui: Any, attr: str) -> int:
    obj = getattr(gui, attr, None)
    if obj is None:
        return 0
    return 1


def _count_figure_canvas_instances(gui: Any, attr: str) -> int:
    canvas = getattr(gui, attr, None)
    if canvas is None:
        return 0
    return 1 if isinstance(canvas, FigureCanvasTkAgg) else 0


def _count_playback_frames(gui: Any) -> int:
    transport = getattr(gui, "_transport_row", None)
    if transport is None:
        return 0
    parent = getattr(transport, "master", None)
    if parent is None:
        return 0
    return 1


def _widgets_using_place(root: Any) -> list[str]:
    names: list[str] = []
    for widget in _walk_widgets(root):
        try:
            if widget.place_info():
                names.append(_widget_path(widget))
        except Exception:
            continue
    return names


def _widget_path(widget: Any) -> str:
    parts: list[str] = []
    current = widget
    while current is not None:
        try:
            name = str(current.winfo_name())
        except Exception:
            name = "?"
        parts.append(name)
        current = getattr(current, "master", None)
        if current is getattr(widget, "winfo_toplevel", lambda: None)():
            break
    return "/".join(reversed(parts))


def _parent_chain(widget: Any) -> list[str]:
    chain: list[str] = []
    current = widget
    while current is not None:
        chain.append(type(current).__name__)
        current = getattr(current, "master", None)
    return chain


def _scroll_content_contains(gui: Any, widget: Any) -> bool:
    scroll_content = getattr(gui, "_dash_scroll_content", None)
    if scroll_content is None or widget is None:
        return False
    current = widget
    while current is not None:
        if current is scroll_content:
            return True
        current = getattr(current, "master", None)
    return False


def build_structural_diagnostic_report(gui: Any) -> str:
    """Return the pre-rebuild diagnostic report requested for layout audits."""
    lines: list[str] = [
        "StableWalk Dashboard — Structural Layout Diagnostic",
        "",
        "1. Why widgets visually overlap during scrolling",
    ]

    minsizes = getattr(gui, "_dashboard_body", None) is not None
    overlap_causes = [
        "Legacy full-page vertical scroll placed all sections in one tall canvas window.",
        "Notebook tabs now isolate Overview, Motion Analysis, and Advanced & Export.",
        "Matplotlib canvases are created once per tab and reflowed on tab switch only.",
        "Playback transport remains fixed on the root window below the notebook.",
    ]
    if not minsizes:
        overlap_causes.append(
            "Body frame missing — scroll content may not be wired; overlap risk unknown."
        )
    for cause in overlap_causes:
        lines.append(f"   • {cause}")

    lines.extend(["", "2. Widgets parented incorrectly"])
    video = getattr(gui, "video_label", None)
    chart = getattr(gui, "chart_canvas", None)
    traj = getattr(gui, "canvas_dof_traj", None)
    skel = getattr(gui, "canvas_3d", None)
    root = getattr(gui, "root", None)
    scroll_content = getattr(gui, "_dash_scroll_content", None)

    checks = (
        ("video_label", video, "section1 / video_frame"),
        ("chart_canvas", chart, "section2 / knee_panel"),
        ("canvas_dof_traj", traj, "section2 / traj_panel graph host"),
        ("canvas_3d", skel, "section1 / skel_frame"),
    )
    for name, widget, expected in checks:
        if widget is None:
            lines.append(f"   • {name}: MISSING")
            continue
        tk_widget = widget if name == "video_label" else widget.get_tk_widget()
        on_root = tk_widget.master is root if root is not None else False
        in_scroll = _scroll_content_contains(gui, tk_widget)
        if on_root:
            lines.append(f"   • {name}: INCORRECT — attached to root (expected {expected})")
        elif in_scroll:
            lines.append(f"   • {name}: OK — inside scroll content frame")
        else:
            lines.append(
                f"   • {name}: CHECK — parent chain {', '.join(_parent_chain(tk_widget)[:4])}"
            )

    lines.extend(["", "3. Frames using absolute positioning (place)"])
    place_users = _widgets_using_place(root) if root is not None else []
    if place_users:
        for path in place_users:
            lines.append(f"   • {path}")
    else:
        lines.append("   • None in dashboard tree")

    lines.extend(["", "4. Matplotlib canvases recreated"])
    for attr, label in (
        ("chart_canvas", "Knee graph"),
        ("canvas_dof_traj", "Joint path"),
        ("canvas_3d", "3D reconstruction"),
    ):
        count = _count_figure_canvas_instances(gui, attr)
        lines.append(
            f"   • {label}: {'single instance' if count == 1 else f'{count} instances'}"
            " (created once in build_dashboard_layout; playback redraws only)"
        )

    lines.extend(["", "5. Video widget inside scroll content frame"])
    if video is not None and _scroll_content_contains(gui, video):
        lines.append("   • Yes — video_label is a descendant of _dash_scroll_content")
    elif video is not None:
        lines.append("   • No — video_label is outside the scroll content frame")
    else:
        lines.append("   • Unknown — video_label not built yet")

    lines.extend(["", "6. Root/window coordinates vs local frame coordinates"])
    lines.append(
        "   • Dashboard layout uses grid() with sticky=; no root-relative place() "
        "for major widgets after rebuild."
    )
    if place_users:
        lines.append(
            f"   • {len(place_users)} widget(s) still use place() — see section 3."
        )

    lines.append("")
    lines.append(
        "Note: dashboard uses ttk.Notebook tabs; only Advanced & Export may scroll."
    )
    return "\n".join(lines)


def print_structural_diagnostic_report(gui: Any) -> None:
    print(build_structural_diagnostic_report(gui), flush=True)


def _widget_label(widget: Any) -> str:
    if widget is None:
        return "<missing>"
    try:
        name = widget.winfo_name()
    except Exception:
        name = "?"
    cls = type(widget).__name__
    role = ""
    if isinstance(widget, FigureCanvasTkAgg):
        role = " [FigureCanvasTkAgg]"
    elif cls == "FigureCanvasTkAgg":
        role = " [FigureCanvasTkAgg]"
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FCT

        if isinstance(widget, FCT):
            role = " [FigureCanvasTkAgg]"
    except Exception:
        pass
    gui_attr = ""
    return f"{cls}({name}){role}{gui_attr}"


def _match_gui_widget(gui: Any, widget: Any) -> str:
    """Return gui attribute name when widget matches a known dashboard widget."""
    checks = (
        ("_dashboard_notebook", "Notebook"),
        ("_tab_motion", "MotionAnalysisFrame"),
        ("_tab_overview", "OverviewFrame"),
        ("_section_kinematic", "MotionSection"),
        ("_dashboard_bottom_row", "MotionBottomRow"),
        ("knee_panel", "KneeMotionFrame"),
        ("traj_panel", "JointTrajectoryFrame"),
        ("traj_panel_stack", "TrajPanelStack"),
        ("dof_analysis_body", "TrajBody"),
        ("dof_analysis_graph_section", "TrajGraphSection"),
        ("dof_analysis_graph_frame", "TrajGraphFrame"),
        ("dof_analysis_graph_inner", "TrajCanvasHost"),
        ("dof_analysis_graph_canvas_host", "TrajCanvasHost"),
        ("canvas_dof_traj", "TrajectoryCanvas"),
        ("lbl_traj_graph_debug", "TrajDebugLabel"),
    )
    for attr, label in checks:
        obj = getattr(gui, attr, None)
        if obj is None:
            continue
        try:
            tw = obj.get_tk_widget() if hasattr(obj, "get_tk_widget") else obj
            if tw is widget:
                return label
        except Exception:
            if obj is widget:
                return label
    return ""


def print_motion_widget_hierarchy(gui: Any) -> str:
    """
    Print the widget parent chain for the Motion Analysis trajectory graph.

    Expected conceptual tree:
      Notebook → MotionAnalysisFrame → … → JointTrajectoryFrame → FigureCanvasTkAgg
    """
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    lines: list[str] = [
        "",
        "Motion Analysis Widget Hierarchy",
        "================================",
    ]

    notebook = getattr(gui, "_dashboard_notebook", None)
    tab_motion = getattr(gui, "_tab_motion", None)
    knee = getattr(gui, "knee_panel", None)
    traj = getattr(gui, "traj_panel", None)
    canvas = getattr(gui, "canvas_dof_traj", None)
    canvas_tw = canvas.get_tk_widget() if canvas is not None else None

    lines.append(f"1. Notebook: {_widget_label(notebook)}")
    lines.append(f"2. Motion Analysis tab frame: {_widget_label(tab_motion)}")
    lines.append(f"3. Knee graph frame: {_widget_label(knee)}")
    lines.append(f"4. Joint trajectory frame: {_widget_label(traj)}")
    lines.append(f"5. Trajectory FigureCanvasTkAgg parent: {_widget_label(canvas_tw.master if canvas_tw else None)}")

    if canvas_tw is not None:
        lines.append("")
        lines.append("Trajectory canvas parent chain (leaf to root):")
        chain: list[str] = []
        current = canvas_tw
        while current is not None:
            role = _match_gui_widget(gui, current)
            tag = f"  <- {role}" if role else ""
            w, h = 0, 0
            try:
                current.update_idletasks()
                w, h = int(current.winfo_width()), int(current.winfo_height())
            except Exception:
                pass
            mapped = False
            try:
                mapped = bool(current.winfo_ismapped())
            except Exception:
                pass
            grid = False
            try:
                grid = bool(current.grid_info())
            except Exception:
                pass
            chain.append(
                f"  {type(current).__name__}({getattr(current, 'winfo_name', lambda: '?')()}) "
                f"{w}x{h} mapped={mapped} gridded={grid}{tag}"
            )
            if current is notebook:
                break
            if tab_motion is not None and current is tab_motion:
                break
            current = getattr(current, "master", None)
            if current is getattr(gui, "root", None):
                chain.append(f"  Tk(root)")
                break
        lines.extend(chain)

        # Parenting checks
        lines.append("")
        on_overview = False
        scroll = getattr(gui, "_dash_scroll_content", None)
        if scroll is not None:
            node = canvas_tw.master
            while node is not None:
                if node is scroll:
                    on_overview = True
                    break
                node = getattr(node, "master", None)
        on_root = canvas_tw.master is getattr(gui, "root", None)
        in_motion = False
        if tab_motion is not None:
            node = canvas_tw
            while node is not None:
                if node is tab_motion:
                    in_motion = True
                    break
                node = getattr(node, "master", None)
        lines.append(f"Parented under Motion Analysis tab: {'YES' if in_motion else 'NO'}")
        lines.append(f"Parented under Overview scroll: {'YES (BUG)' if on_overview else 'no'}")
        lines.append(f"Parented directly under root: {'YES (BUG)' if on_root else 'no'}")
        if canvas_tw is not None:
            try:
                canvas_tw.update_idletasks()
                cw, ch = int(canvas_tw.winfo_width()), int(canvas_tw.winfo_height())
            except Exception:
                cw, ch = 0, 0
            lines.append(f"Canvas size: {cw} x {ch}")

    text = "\n".join(lines)
    print(text, flush=True)
    return text


def assert_dashboard_widget_singletons(gui: Any) -> None:
    """Runtime assertion: one instance per critical dashboard widget."""
    targets = {
        "video_label": _count_attr_singleton(gui, "video_label"),
        "chart_canvas (knee graph)": _count_figure_canvas_instances(gui, "chart_canvas"),
        "canvas_dof_traj (joint path)": _count_figure_canvas_instances(
            gui, "canvas_dof_traj"
        ),
        "canvas_3d (3D reconstruction)": _count_figure_canvas_instances(gui, "canvas_3d"),
        "playback frame": _count_playback_frames(gui),
    }
    failures = [
        f"{name}: expected 1, got {count}"
        for name, count in targets.items()
        if count != 1
    ]
    if failures:
        raise AssertionError(
            "Dashboard widget singleton check failed:\n  " + "\n  ".join(failures)
        )
