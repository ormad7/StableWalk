"""Verify dashboard panel geometry at common window sizes."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk

TEST_SIZES = (
    (1920, 1080),
    (1680, 900),
    (1600, 900),
    (1440, 900),
    (1366, 768),
    (1280, 720),
)


def _bbox(widget: tk.Misc) -> tuple[int, int, int, int] | None:
    try:
        widget.update_idletasks()
        x = widget.winfo_rootx()
        y = widget.winfo_rooty()
        w = widget.winfo_width()
        h = widget.winfo_height()
        if w <= 1 or h <= 1:
            return None
        return (x, y, x + w, y + h)
    except tk.TclError:
        return None


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 + 1) < bx1 and (bx0 + 1) < ax1 and (ay0 + 1) < by1 and (by0 + 1) < ay1


def _h_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ax0, _ay0, ax1, _ay1 = a
    bx0, _by0, bx1, _by1 = b
    if ax1 <= bx0:
        return bx0 - ax1
    if bx1 <= ax0:
        return ax0 - bx1
    return -min(ax1 - bx0, bx1 - ax0)


def _label_frame_ancestor(widget: tk.Misc, steps: int = 5) -> tk.Misc:
    node = widget
    for _ in range(steps):
        if node is None:
            break
        if isinstance(node, tk.Widget) and node.winfo_class() == "TLabelframe":
            return node
        node = node.master
    return widget


def _check_layout(app, width: int, height: int) -> list[str]:
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    failures: list[str] = []
    app.root.geometry(f"{width}x{height}+-3200+0")
    app.root.update_idletasks()
    apply_responsive_layout(app, width=width, height=height)
    app.root.update_idletasks()
    app.root.update()

    video_panel = _label_frame_ancestor(app.video_label)
    skel_panel = app.skel_frame
    sidebar = app.sidebar
    analysis_panel = app.traj_panel
    transport = getattr(app, "_transport_row", None)

    panels = {
        "video": video_panel,
        "skeleton": skel_panel,
        "sidebar": sidebar,
        "analysis": analysis_panel,
    }
    data_btn = getattr(app, "btn_collected_data", None)
    if data_btn is not None and data_btn.winfo_ismapped():
        panels["data_button"] = data_btn
    if transport is not None:
        panels["transport"] = transport

    bboxes = {name: _bbox(w) for name, w in panels.items()}
    root_bb = _bbox(app.root)

    for name, bb in bboxes.items():
        if bb is None:
            failures.append(f"{name}: not visible or zero size")
            continue
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if name == "skeleton" and (w < 220 or h < 180):
            failures.append(f"skeleton: too small ({w}x{h})")
        if name == "video" and (w < 200 or h < 160):
            failures.append(f"video: too small ({w}x{h})")
        if name == "analysis" and w < 280:
            failures.append(f"analysis: too narrow ({w}px)")
        if name == "data_button" and (w < 40 or h < 16):
            failures.append(f"data_button: too small ({w}x{h})")
        if name == "sidebar" and width >= 1280:
            from stablewalk.ui.tk.dashboard_responsive import _sidebar_inline, classify_layout

            wmode, _ = classify_layout(width, height)
            if _sidebar_inline(wmode, width):
                max_w = int(width * 0.22)
                if w > max_w:
                    failures.append(f"sidebar: too wide ({w}px > {max_w}px cap)")
        if name == "transport" and root_bb:
            if bb[3] > root_bb[3] + 2:
                failures.append("transport: clipped below window")

    app._fit_skeleton_canvas()
    app._fit_dof_traj_canvas()
    app.root.update()

    skel_host = app.skel_canvas_host
    skel_canvas = app.canvas_3d.get_tk_widget()
    sh = _bbox(skel_host)
    sc = _bbox(skel_canvas)
    if sh and sc:
        if sc[2] > sh[2] + 6 or sc[0] < sh[0] - 6:
            failures.append("skeleton canvas extends outside host (clip risk)")

    graph_host = app.dof_analysis_graph_inner
    traj_widget = app.canvas_dof_traj.get_tk_widget()
    gh = _bbox(graph_host)
    tw = _bbox(traj_widget)
    if gh and tw and (gh[2] - gh[0]) > 80:
        fill = (tw[2] - tw[0]) / max(gh[2] - gh[0], 1)
        if fill < 0.55:
            failures.append(f"3D graph canvas fills only {fill:.0%} of graph host")

    for a, b in [("video", "skeleton")]:
        ba, bb = bboxes.get(a), bboxes.get(b)
        if ba and bb and _overlaps(ba, bb):
            failures.append(f"overlap: {a} vs {b}")

    if width >= 1280:
        from stablewalk.ui.tk.dashboard_responsive import _sidebar_inline, classify_layout

        wmode, _ = classify_layout(width, height)
        if _sidebar_inline(wmode, width):
            bs, bside = bboxes.get("skeleton"), bboxes.get("sidebar")
            if bs and bside and _h_gap(bs, bside) < 0:
                failures.append(f"skeleton overlaps sidebar by {-_h_gap(bs, bside)}px")

    stab_score = getattr(app, "lbl_stab_score", None)
    if stab_score is not None and not stab_score.winfo_ismapped():
        failures.append("stability score: not visible")

    scroll = getattr(app, "_dash_scroll_canvas", None)
    if scroll is not None:
        failures.append("main scroll canvas should not be present")

    return failures


def main() -> int:
    from stablewalk.ui.tk.dashboard_responsive import MIN_WINDOW_HEIGHT, MIN_WINDOW_WIDTH
    from stablewalk.ui.tk.app import StableWalkGUI

    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()

    all_failures: dict[str, list[str]] = {}
    passed: list[str] = []

    for width, height in TEST_SIZES:
        label = f"{width}x{height}"
        failures = _check_layout(app, width, height)
        if failures:
            all_failures[label] = failures
        else:
            passed.append(label)

    root.destroy()

    print("DASHBOARD LAYOUT VERIFICATION")
    print(f"Minimum window: {MIN_WINDOW_WIDTH}x{MIN_WINDOW_HEIGHT}")
    print()
    for label in passed:
        print(f"  PASS  {label}")
    for label, failures in all_failures.items():
        print(f"  FAIL  {label}")
        for item in failures:
            print(f"        - {item}")

    return 1 if all_failures else 0


if __name__ == "__main__":
    sys.exit(main())
