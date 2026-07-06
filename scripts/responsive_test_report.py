"""Generate a responsive layout test report at multiple window sizes."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk

TEST_SIZES = (
    (1920, 1080),
    (1600, 900),
    (1440, 900),
    (1366, 768),
    (1280, 720),
    (1199, 768),
    (1024, 720),
)


def _bbox(widget: tk.Misc) -> tuple[int, int, int, int] | None:
    try:
        widget.update_idletasks()
        x, y = widget.winfo_rootx(), widget.winfo_rooty()
        w, h = widget.winfo_width(), widget.winfo_height()
        if w <= 1 or h <= 1:
            return None
        return (x, y, x + w, y + h)
    except tk.TclError:
        return None


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (ax0 + 1) < bx1 and (bx0 + 1) < ax1 and (ay0 + 1) < by1 and (by0 + 1) < ay1


def _measure(app, width: int, height: int) -> dict[str, object]:
    from stablewalk.ui.tk.dashboard_responsive import (
        HeightMode,
        WidthMode,
        apply_responsive_layout,
        classify_layout,
    )

    app.root.geometry(f"{width}x{height}+-3200+0")
    app.root.update_idletasks()
    apply_responsive_layout(app, width=width, height=height)
    app.root.update_idletasks()
    app.root.update()

    wmode, hmode = classify_layout(width, height)
    video_panel = app.video_frame
    skel_panel = app.skel_frame
    sidebar = app.sidebar
    graph_host = app.dof_analysis_graph_canvas_host
    transport = getattr(app, "_transport_row", None)
    stab = getattr(app, "lbl_stab_score", None)

    vp = (video_panel.winfo_width(), video_panel.winfo_height())
    sp = (skel_panel.winfo_width(), skel_panel.winfo_height())
    skel_canvas = app.canvas_3d.get_tk_widget()
    sc = (skel_canvas.winfo_width(), skel_canvas.winfo_height())
    gh = (graph_host.winfo_width(), graph_host.winfo_height())
    traj_w = app.canvas_dof_traj.get_tk_widget()
    tw = (traj_w.winfo_width(), traj_w.winfo_height())

    displayed_video = (app.video_label.winfo_width(), app.video_label.winfo_height())

    scroll = getattr(app, "_analysis_scroll_canvas", None)
    scroll_needed = False
    if scroll is not None:
        scroll.update_idletasks()
        region = scroll.bbox("all")
        if region is not None:
            content_h = region[3] - region[1]
            scroll_needed = content_h > scroll.winfo_height() + 4

    overlap = False
    bb_v = _bbox(video_panel)
    bb_s = _bbox(skel_panel)
    if bb_v and bb_s and wmode is not WidthMode.SMALL:
        overlap = _overlaps(bb_v, bb_s)

    playback_visible = transport is not None and transport.winfo_ismapped()
    root_bb = _bbox(app.root)
    transport_clipped = False
    if transport is not None and root_bb:
        tb = _bbox(transport)
        if tb and tb[3] > root_bb[3] + 2:
            transport_clipped = True

    return {
        "resolution": f"{width}x{height}",
        "layout_mode": f"{wmode.value}/{hmode.value}",
        "video_panel": vp,
        "displayed_video": displayed_video,
        "3d_panel": sp,
        "skeleton_canvas": sc,
        "stability_visible": stab is not None and stab.winfo_ismapped(),
        "graph_host": gh,
        "graph_canvas": tw,
        "graph_fill": (tw[0] / gh[0]) if gh[0] > 0 else 0,
        "vertical_scrolling_required": scroll_needed,
        "playback_visible": playback_visible and not transport_clipped,
        "overlap_found": overlap,
        "tabs_active": getattr(app, "_viz_tabs_active", False),
    }


def main() -> int:
    from stablewalk.ui.tk.app import StableWalkGUI

    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()

    print("RESPONSIVE TEST REPORT")
    print("=" * 72)

    rows: list[dict[str, object]] = []
    for width, height in TEST_SIZES:
        rows.append(_measure(app, width, height))

    # Manual resize simulation: 1920 -> 1280 in steps
    transitions: list[str] = []
    prev_mode = None
    for w in (1920, 1680, 1550, 1549, 1350, 1349, 1200, 1199, 1280):
        apply = __import__(
            "stablewalk.ui.tk.dashboard_responsive",
            fromlist=["apply_responsive_layout", "classify_layout"],
        )
        app.root.geometry(f"{w}x900+-3200+0")
        app.root.update()
        apply.apply_responsive_layout(app, width=w, height=900)
        mode = apply.classify_layout(w, 900)[0].value
        if mode != prev_mode:
            transitions.append(f"  width {w}: -> {mode}")
            prev_mode = mode

    root.destroy()

    for r in rows:
        print()
        print(f"Resolution: {r['resolution']}")
        print(f"Layout mode: {r['layout_mode']}")
        print(f"Video panel size: {r['video_panel'][0]}x{r['video_panel'][1]}")
        print(f"Displayed video size: {r['displayed_video'][0]}x{r['displayed_video'][1]}")
        print(f"3D panel size: {r['3d_panel'][0]}x{r['3d_panel'][1]}")
        print(f"Skeleton size: {r['skeleton_canvas'][0]}x{r['skeleton_canvas'][1]}")
        print(f"Stability visible: {r['stability_visible']}")
        print(f"Movement graph accessible: {r['graph_host'][0] >= 200 and r['graph_host'][1] >= 120}")
        print(f"Graph canvas fill: {r['graph_fill']:.0%}")
        print(f"Vertical scrolling required: {r['vertical_scrolling_required']}")
        print(f"Playback visible: {r['playback_visible']}")
        print(f"Overlap found: {r['overlap_found']}")
        if r["tabs_active"]:
            print("Visual tabs: active")

    print()
    print("Layout mode transitions (width sweep @ 900px height):")
    for line in transitions:
        print(line)

    failures = [
        r for r in rows
        if not r["stability_visible"]
        or r["overlap_found"]
        or not r["playback_visible"]
        or r["graph_host"][0] < 180
    ]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
