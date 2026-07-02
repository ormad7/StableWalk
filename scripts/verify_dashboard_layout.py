"""Verify dashboard panel geometry: no overlap, minimum sizes, scrollbars present."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk


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
    # Allow 1px tolerance for grid gutters
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


def main() -> int:
    root = tk.Tk()
    root.geometry("1440x880+-2400+0")
    root.minsize(1440, 880)

    from stablewalk.ui.tk.app import StableWalkGUI

    app = StableWalkGUI(root=root)
    root.geometry("1440x880+-2400+0")
    root.update_idletasks()
    root.update()

    failures: list[str] = []

    video_panel = _label_frame_ancestor(app.video_label)
    skel_panel = app.skel_frame
    sidebar = app.sidebar
    analysis_panel = app.traj_panel
    table_panel = _label_frame_ancestor(app.dof_pos_tree)

    panels = {
        "video": video_panel,
        "skeleton": skel_panel,
        "sidebar": sidebar,
        "analysis": analysis_panel,
        "table": table_panel,
    }
    bboxes = {name: _bbox(w) for name, w in panels.items()}

    for name, bb in bboxes.items():
        if bb is None:
            failures.append(f"{name}: not visible or zero size")
            continue
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]
        if name == "skeleton" and (w < 220 or h < 200):
            failures.append(f"skeleton: too small ({w}x{h})")
        if name == "analysis" and w < 380:
            failures.append(f"analysis: too narrow ({w}px)")
        if name == "sidebar":
            max_w = int(root.winfo_width() * 0.16)
            if w > max_w:
                failures.append(
                    f"sidebar: too wide ({w}px > {max_w}px cap) — may squeeze main panels"
                )
        if name == "table" and w > 0:
            analysis_bb = bboxes.get("analysis")
            if analysis_bb and w > (analysis_bb[2] - analysis_bb[0]) * 0.40:
                failures.append("table: wider than 40% of analysis column")

    app._fit_skeleton_canvas()
    app._fit_dof_traj_canvas()
    root.update()

    skel_host = app.skel_canvas_host
    skel_canvas = app.canvas_3d.get_tk_widget()
    sh = _bbox(skel_host)
    sc = _bbox(skel_canvas)
    if sh and sc:
        if sc[2] > sh[2] + 4 or sc[0] < sh[0] - 4:
            failures.append("skeleton canvas extends outside host (clip risk)")

    graph_host = app.dof_analysis_graph_inner
    traj_widget = app.canvas_dof_traj.get_tk_widget()
    gh = _bbox(graph_host)
    tw = _bbox(traj_widget)
    if gh and tw and (gh[2] - gh[0]) > 80:
        fill = (tw[2] - tw[0]) / max(gh[2] - gh[0], 1)
        if fill < 0.80:
            failures.append(f"3D graph canvas fills only {fill:.0%} of graph host")

    for a, b in [("video", "skeleton"), ("skeleton", "sidebar"), ("analysis", "table")]:
        ba, bb = bboxes.get(a), bboxes.get(b)
        if ba and bb and _overlaps(ba, bb):
            failures.append(f"overlap: {a} vs {b}")

    bs, bside = bboxes.get("skeleton"), bboxes.get("sidebar")
    if bs and bside and _h_gap(bs, bside) < 0:
        failures.append(f"skeleton overlaps sidebar by {-_h_gap(bs, bside)}px")

    tree = app.dof_pos_tree
    host = getattr(tree, "_sw_host", None)
    has_v = has_h = False
    if host is not None:
        for child in host.winfo_children():
            if "Scrollbar" in child.winfo_class():
                if str(child.cget("orient")) == "vertical":
                    has_v = True
                else:
                    has_h = True
    if not has_v or not has_h:
        failures.append("position table: missing scrollbars")

    root.destroy()

    if failures:
        print("LAYOUT CHECK FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("LAYOUT CHECK PASSED (1440x880)")
    for name, bb in bboxes.items():
        if bb:
            print(f"  {name}: {bb[2]-bb[0]}x{bb[3]-bb[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
