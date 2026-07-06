"""Measure primary visualization fill ratios at common window sizes."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk

POSES = _ROOT / "data" / "output" / "poses" / "validation_athletic_poses.json"


def _bbox(widget: tk.Misc) -> tuple[int, int, int, int] | None:
    try:
        widget.update_idletasks()
        w, h = widget.winfo_width(), widget.winfo_height()
        if w <= 1 or h <= 1:
            return None
        return (w, h)
    except tk.TclError:
        return None


def _label_frame_ancestor(widget: tk.Misc) -> tk.Misc:
    node = widget
    for _ in range(6):
        if node is None:
            break
        if isinstance(node, tk.Widget) and node.winfo_class() == "TLabelframe":
            return node
        node = node.master
    return widget


def _hydrate(app) -> None:
    from scripts.final_validation_report import _simulate_analyzed_demo

    failures = _simulate_analyzed_demo(app, "athletic", poses_path=POSES)
    if failures:
        raise RuntimeError("; ".join(failures))
    app._ensure_default_dof_selection()
    app._show_pose_at(0)
    app._fit_skeleton_canvas()
    app._fit_dof_traj_canvas()
    app._render_dof_traj_canvas(force=True)
    app.root.update()


def measure(app, width: int, height: int) -> dict[str, object]:
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    app.root.geometry(f"{width}x{height}+-3200+0")
    app.root.update_idletasks()
    apply_responsive_layout(app, width=width, height=height)
    _hydrate(app)

    video_panel = _label_frame_ancestor(app.video_label)
    vp = _bbox(video_panel)
    vl = _bbox(app.video_label)
    sh = _bbox(app.skel_canvas_host)
    sc = _bbox(app.canvas_3d.get_tk_widget())
    gh = _bbox(app.dof_analysis_graph_canvas_host)
    tw = _bbox(app.canvas_dof_traj.get_tk_widget())
    body = _bbox(app._dashboard_body)

    out: dict[str, object] = {"size": f"{width}x{height}"}

    img_w = img_h = 0
    try:
        img = app.video_label.cget("image")
        if img:
            img_w = int(app.video_label.tk.call("image", "width", img))
            img_h = int(app.video_label.tk.call("image", "height", img))
    except (tk.TclError, ValueError, TypeError):
        pass

    if vp and vl:
        pw, ph = vp
        lw, lh = vl
        out["video_panel"] = f"{pw}x{ph}"
        out["video_label_area"] = f"{lw}x{lh}"
        if img_w and img_h:
            out["video_image"] = f"{img_w}x{img_h}"
            out["video_fill_label"] = f"{min(img_w / lw, img_h / lh):.0%}"
            out["video_fill_panel"] = f"{min(img_w / pw, img_h / ph):.0%}"

    if sh and sc:
        out["skeleton_host"] = f"{sh[0]}x{sh[1]}"
        out["skeleton_canvas"] = f"{sc[0]}x{sc[1]}"
        out["skeleton_fill"] = f"{sc[0] / sh[0]:.0%} x {sc[1] / sh[1]:.0%}"

    if gh and tw:
        out["graph_host"] = f"{gh[0]}x{gh[1]}"
        out["graph_canvas"] = f"{tw[0]}x{tw[1]}"
        out["graph_fill"] = f"{tw[0] / gh[0]:.0%} x {tw[1] / gh[1]:.0%}"

    if body and vp and sh:
        viz_h = max(vp[1], sh[1])
        out["viz_row_height_share"] = f"{viz_h / body[1]:.0%}"

    sidebar = _bbox(app.sidebar)
    if sidebar and body:
        out["sidebar_width_share"] = f"{sidebar[0] / body[0]:.0%}"

    return out


def main() -> int:
    from stablewalk.ui.tk.app import StableWalkGUI

    sizes = [(1920, 1080), (1680, 900), (1600, 900), (1366, 768), (1280, 720)]
    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()

    print("VISUAL HIERARCHY MEASUREMENTS (athletic demo hydrated)")
    print("=" * 60)
    for w, h in sizes:
        m = measure(app, w, h)
        print(f"\n{m['size']}:")
        for k, v in m.items():
            if k != "size":
                print(f"  {k}: {v}")

    root.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
