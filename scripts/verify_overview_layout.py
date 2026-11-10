"""Render Overview at 1366×768 and 1920×1080 and check pane / label layout."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "spectral_layout_verify"
OUT.mkdir(parents=True, exist_ok=True)


def _pump(root, seconds: float = 0.35) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            root.update()
        except Exception:
            break
        time.sleep(0.01)


def _geom(widget) -> dict[str, int]:
    if widget is None:
        return {"mapped": 0, "w": 0, "h": 0}
    try:
        return {
            "mapped": int(bool(widget.winfo_ismapped())),
            "w": int(widget.winfo_width() or 0),
            "h": int(widget.winfo_height() or 0),
        }
    except Exception:
        return {"mapped": 0, "w": 0, "h": 0}


def _canvas_geom(gui, attr: str) -> dict[str, int]:
    canvas = getattr(gui, attr, None)
    if canvas is None:
        return {"mapped": 0, "w": 0, "h": 0}
    getter = getattr(canvas, "get_tk_widget", None)
    widget = getter() if callable(getter) else canvas
    return _geom(widget)


def main() -> int:
    import tkinter as tk
    import tkinter.messagebox as mb

    from stablewalk import config
    from stablewalk.ui.overview_path_display import (
        CAMERA_VIEW_SOURCE,
        named_camera_views,
        source_camera_view_available,
    )
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab

    originals = (mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel)
    mb.showinfo = lambda *a, **k: None  # type: ignore[assignment]
    mb.showwarning = lambda *a, **k: None  # type: ignore[assignment]
    mb.showerror = lambda *a, **k: None  # type: ignore[assignment]
    mb.askokcancel = lambda *a, **k: True  # type: ignore[assignment]

    gui = StableWalkGUI()
    root = gui.root
    report: dict[str, object] = {
        "sizes": {},
        "screenshots": {},
        "gui_verified": False,
        "poses_loaded": False,
    }
    failures: list[str] = []
    grabbed = 0
    try:
        select_dashboard_tab(gui, TAB_OVERVIEW)
        root.deiconify()
        poses_path = config.POSES_DIR / "normal_gait_poses.json"
        video_path = config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
        if poses_path.is_file():
            try:
                source = str(video_path) if video_path.is_file() else str(poses_path)
                gui.url_var.set(source)
                gui._current_source = source
                if gui.load_poses(poses_path, fresh=True, expected_source=source):
                    report["poses_loaded"] = True
                    _pump(root, 0.6)
                    if gui.skeleton_player is not None:
                        n = max(1, gui.skeleton_player.frame_count - 1)
                        gui._go_to(max(0, int(n * 0.25)))
                    focus = getattr(gui, "_focus_joint_trajectory_from_skeleton", None)
                    if callable(focus):
                        focus("right_hip", compare=False)
                    _pump(root, 0.5)
            except Exception as exc:
                report["poses_error"] = str(exc)
        for width, height in ((1366, 768), (1920, 1080)):
            try:
                root.state("normal")
            except Exception:
                pass
            root.geometry(f"{width}x{height}")
            if height >= 1000:
                try:
                    root.state("zoomed")
                except Exception:
                    pass
            root.update_idletasks()
            _pump(root, 0.55)
            try:
                gui._fit_overview_viewers(reset_camera=True)
            except Exception:
                pass
            _pump(root, 0.35)
            video = _geom(getattr(gui, "video_frame", None))
            skel = _geom(getattr(gui, "skel_frame", None))
            path = _geom(getattr(gui, "overview_traj_panel", None))
            split = _geom(getattr(gui, "_overview_main_split", None))
            primary = _geom(
                getattr(gui, "_primary_viz_host", None)
                or getattr(gui, "_section_visual", None)
            )
            info = _geom(getattr(gui, "_overview_info_outer", None))
            tab = _geom(getattr(gui, "_tab_overview", None))
            content = _geom(getattr(gui, "_overview_content", None))
            skel_canvas = _canvas_geom(gui, "canvas_3d")
            traj_canvas = _canvas_geom(gui, "canvas_dof_traj_overview")
            stats_body = _geom(getattr(gui, "_overview_path_stats_body", None))
            unused = max(0, primary["h"] - max(video["h"], skel["h"], path["h"]))
            key = f"{width}x{height}"
            root_geom = _geom(root)
            report["sizes"][key] = {
                "requested": {"w": width, "h": height},
                "clamped": int(
                    root_geom["w"] < width - 20 or root_geom["h"] < height - 20
                ),
                "root": root_geom,
                "notebook": _geom(getattr(gui, "_dashboard_notebook", None)),
                "video": video,
                "skeleton": skel,
                "path": path,
                "split": split,
                "primary": primary,
                "info": info,
                "tab": tab,
                "content": content,
                "skeleton_canvas": skel_canvas,
                "trajectory_canvas": traj_canvas,
                "path_stats_body": stats_body,
                "unused_below_viewers": unused,
                "path_stats_expanded": bool(
                    getattr(gui, "_overview_path_stats_expanded", False)
                ),
            }
            actual_h = max(1, root_geom["h"])
            min_h = 280 if actual_h <= 850 else 420
            for name, geom in (("video", video), ("skeleton", skel), ("path", path)):
                if geom["w"] < 200 or geom["h"] < min_h:
                    failures.append(f"{key} {name} too small: {geom}")
            if split["h"] > 200 and primary["h"] + 8 < int(0.68 * split["h"]):
                failures.append(
                    f"{key} viewers not consuming split: "
                    f"primary={primary['h']} split={split['h']}"
                )
            if skel["h"] > 160 and skel_canvas["h"] + 24 < int(0.55 * skel["h"]):
                failures.append(
                    f"{key} skeleton canvas not filling panel: "
                    f"canvas={skel_canvas['h']} panel={skel['h']}"
                )
            if path["h"] > 160 and traj_canvas["h"] + 24 < int(0.50 * path["h"]):
                failures.append(
                    f"{key} trajectory canvas not filling panel: "
                    f"canvas={traj_canvas['h']} panel={path['h']}"
                )
            widths = [video["w"], skel["w"], path["w"]]
            if min(widths) > 0 and max(widths) / max(min(widths), 1) > 2.2:
                failures.append(f"{key} pane widths unbalanced: {widths}")
            if unused > 80:
                failures.append(f"{key} unused below viewers: {unused}px")
            if primary["h"] > 80 and split["h"] > 80:
                if primary["h"] + 40 < 0.62 * split["h"]:
                    failures.append(
                        f"{key} primary not consuming split: "
                        f"primary={primary['h']} split={split['h']}"
                    )
            if bool(getattr(gui, "_overview_path_stats_expanded", False)):
                failures.append(f"{key} path statistics started expanded")
            if stats_body["mapped"]:
                failures.append(f"{key} path statistics body mapped while collapsed")
            caption = getattr(gui, "lbl_overview_coord_caption", None)
            if caption is None:
                failures.append(f"{key} missing coordinate caption")
            else:
                text = str(caption.cget("text") or "")
                if "Origin" not in text or "Axes" not in text:
                    failures.append(f"{key} caption incomplete: {text!r}")
            png = OUT / f"overview_{width}x{height}.png"
            report["screenshots"][key] = str(png)
            try:
                root.update_idletasks()
                from PIL import ImageGrab

                x = int(root.winfo_rootx())
                y = int(root.winfo_rooty())
                ImageGrab.grab(
                    bbox=(
                        x,
                        y,
                        x + int(root.winfo_width()),
                        y + int(root.winfo_height()),
                    )
                ).save(png)
                grabbed += 1
            except Exception as exc:
                failures.append(f"{key} screenshot unavailable: {exc}")
        views = named_camera_views(
            include_source=source_camera_view_available(
                getattr(gui, "gait_motion", None)
            )
        )
        report["camera_views"] = list(views)
        if CAMERA_VIEW_SOURCE in views and not source_camera_view_available(
            getattr(gui, "gait_motion", None)
        ):
            failures.append("Source camera listed without a camera-aligned transform")
        report["gui_verified"] = grabbed == 2
        report["failures"] = failures
        (OUT / "overview_layout_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, indent=2))
        return 1 if failures else 0
    finally:
        mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel = originals
        try:
            root.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
