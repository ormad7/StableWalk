"""Task 1: measure Overview viewer geometry at 1366x768, 1600x900, 1920x1080."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIZES = ((1366, 768), (1600, 900), (1920, 1080))


def _pump(root, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            root.update()
        except Exception:
            break
        time.sleep(0.01)


def _geom(widget) -> dict[str, int | str | bool]:
    if widget is None:
        return {"present": False}
    try:
        widget.update_idletasks()
        return {
            "present": True,
            "mapped": bool(widget.winfo_ismapped()),
            "viewable": bool(widget.winfo_viewable()),
            "x": int(widget.winfo_x()),
            "y": int(widget.winfo_y()),
            "w": int(widget.winfo_width()),
            "h": int(widget.winfo_height()),
            "rootx": int(widget.winfo_rootx()),
            "rooty": int(widget.winfo_rooty()),
            "manager": str(widget.winfo_manager() or ""),
        }
    except Exception as exc:
        return {"present": True, "error": str(exc)}


def _clipped(child, parent, *, slop: int = 4) -> bool:
    if child is None or parent is None:
        return True
    try:
        cr, pr = child.winfo_rooty(), parent.winfo_rooty()
        ch, ph = int(child.winfo_height()), int(parent.winfo_height())
        # Bottom of child below bottom of parent (sash / LabelFrame slop).
        return (cr + ch) > (pr + ph + slop)
    except Exception:
        return True


def main() -> int:
    from stablewalk import config
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    out_dir = config.OUTPUT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    gui = StableWalkGUI()
    root = gui.root
    select_dashboard_tab(gui, TAB_OVERVIEW)
    results = []
    shot_ok = False
    try:
        import PIL.ImageGrab as ImageGrab  # type: ignore
    except Exception:
        ImageGrab = None  # type: ignore

    for width, height in SIZES:
        try:
            root.state("normal")
        except Exception:
            pass
        root.geometry(f"{width}x{height}+40+40")
        apply_responsive_layout(gui, width=width, height=height)
        _pump(root, seconds=0.8)
        ensure = getattr(gui, "_ensure_skeleton_figure", None)
        if callable(ensure):
            try:
                ensure()
            except Exception:
                pass
        try:
            gui._fit_skeleton_canvas(force=True)
        except Exception:
            pass
        try:
            gui._render_overview_traj_canvas(force=True)
        except Exception:
            pass
        _pump(root, seconds=0.4)

        video = getattr(gui, "video_frame", None)
        skel = getattr(gui, "skel_frame", None)
        path = getattr(gui, "overview_traj_panel", None)
        primary = getattr(gui, "_primary_viz_host", None)
        split = getattr(gui, "_overview_viz_split", None)
        info = getattr(gui, "_overview_info_outer", None)
        transport = getattr(gui, "_transport_row", None)
        skel_host = getattr(gui, "skel_canvas_host", None)
        skel_canvas = None
        try:
            canvas_3d = getattr(gui, "canvas_3d", None)
            if canvas_3d is not None:
                skel_canvas = canvas_3d.get_tk_widget()
        except Exception:
            skel_canvas = None
        path_canvas = None
        try:
            traj = getattr(gui, "canvas_dof_traj_overview", None)
            if traj is not None:
                path_canvas = traj.get_tk_widget()
        except Exception:
            path_canvas = None
        actual = [int(root.winfo_width()), int(root.winfo_height())]
        row = {
            "window": [width, height],
            "actual_window": actual,
            "video": _geom(video),
            "skeleton": _geom(skel),
            "path": _geom(path),
            "primary": _geom(primary),
            "viz_split": _geom(split),
            "info": _geom(info),
            "transport": _geom(transport),
            "phase": _geom(getattr(gui, "_overview_phase_strip", None)),
            "kpi": _geom(getattr(gui, "_overview_clinical_summary", None)),
            "clearance": _geom(getattr(gui, "_foot_clearance_detail_host", None)),
            "skel_host": _geom(skel_host),
            "skel_canvas": _geom(skel_canvas),
            "path_canvas": _geom(path_canvas),
            "info_scroll": bool(getattr(gui, "_overview_info_scroll_needed", False)),
        }
        failures = []
        for name, widget in (("video", video), ("skeleton", skel), ("path", path)):
            g = row[name]
            if not g.get("mapped"):
                failures.append(f"{name} not mapped")
            if int(g.get("h") or 0) < 160:
                failures.append(f"{name} height {g.get('h')} < 160")
            if split is not None and _clipped(widget, split):
                failures.append(f"{name} clipped by viz_split")
            if primary is not None and _clipped(widget, primary):
                failures.append(f"{name} clipped by primary pane")
        if skel_canvas is not None and skel_host is not None and _clipped(
            skel_canvas, skel_host
        ):
            failures.append("skeleton canvas taller than skel_canvas_host")
        if skel_canvas is not None and primary is not None and _clipped(
            skel_canvas, primary
        ):
            failures.append("skeleton canvas clipped by primary pane")
        if path_canvas is not None and primary is not None and _clipped(
            path_canvas, primary
        ):
            failures.append("path canvas clipped by primary pane")
        if transport is not None:
            tg = row["transport"]
            if not tg.get("mapped"):
                failures.append("transport not mapped")
        row["failures"] = failures
        results.append(row)

        if ImageGrab is not None:
            try:
                x, y = int(root.winfo_rootx()), int(root.winfo_rooty())
                w, h = int(root.winfo_width()), int(root.winfo_height())
                img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
                dest = out_dir / f"overview_task1_{width}x{height}.png"
                img.save(dest)
                row["screenshot"] = str(dest)
                shot_ok = True
            except Exception as exc:
                row["screenshot_error"] = str(exc)

    payload = {"shot_ok": shot_ok, "sizes": results}
    dest_json = out_dir / "overview_task1_layout.json"
    dest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    any_fail = any(row.get("failures") for row in results)
    try:
        root.destroy()
    except Exception:
        pass
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
