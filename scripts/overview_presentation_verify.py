"""Load normal_gait, force Split, screenshot Overview at target resolutions."""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "docs" / "gui_layout_screenshots"


@contextlib.contextmanager
def _suppress_messageboxes():
    import tkinter.messagebox as mb

    originals = (mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel)
    mb.showinfo = lambda *a, **k: None  # type: ignore[assignment]
    mb.showwarning = lambda *a, **k: None  # type: ignore[assignment]
    mb.showerror = lambda *a, **k: None  # type: ignore[assignment]
    mb.askokcancel = lambda *a, **k: True  # type: ignore[assignment]
    try:
        yield
    finally:
        mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel = originals


def _pump(root, *, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)


def _geom(widget) -> dict[str, int | str]:
    if widget is None:
        return {"mapped": 0}
    try:
        return {
            "mapped": int(bool(widget.winfo_ismapped())),
            "w": int(widget.winfo_width() or 0),
            "h": int(widget.winfo_height() or 0),
            "x": int(widget.winfo_rootx() or 0),
            "y": int(widget.winfo_rooty() or 0),
            "manager": str(widget.winfo_manager() or ""),
        }
    except Exception as exc:
        return {"error": str(exc)}


def main() -> int:
    import tkinter as tk
    from PIL import ImageGrab

    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        VIEW_MODE_SIDE_BY_SIDE,
        apply_overview_view_mode,
    )
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.ui.tk.layout_scale import apply_process_dpi_awareness

    OUT.mkdir(parents=True, exist_ok=True)
    apply_process_dpi_awareness()

    ex = example_by_key("normal")
    poses_path = config.POSES_DIR / "normal_gait_poses.json"
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
    if not Path(video_path).is_file() and not poses_path.is_file():
        print(f"FAILED: missing {video_path} and {poses_path}", file=sys.stderr)
        return 1

    with _suppress_messageboxes():
        gui = StableWalkGUI()
        root = gui.root
        root.geometry("1600x900+20+20")
        try:
            root.attributes("-topmost", True)
            root.lift()
        except tk.TclError:
            pass
        _pump(root, seconds=0.5)
        apply_overview_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
        select_dashboard_tab(gui, TAB_OVERVIEW)
        _pump(root, seconds=0.2)

        source = str(video_path)
        gui.url_var.set(source)
        try:
            gui.url_entry.delete(0, "end")
            gui.url_entry.insert(0, source)
        except Exception:
            pass
        gui._current_source = source
        loaded = False
        if poses_path.is_file():
            try:
                loaded = bool(gui.load_poses(poses_path, fresh=True, expected_source=source))
            except TypeError:
                loaded = bool(gui.load_poses(poses_path, fresh=True))
        if not loaded and Path(video_path).is_file():
            try:
                gui._load_source(source)
                loaded = True
            except Exception as exc:
                print(f"load video failed: {exc}", file=sys.stderr)
        if not loaded:
            print("FAILED: could not load normal_gait", file=sys.stderr)
            try:
                root.destroy()
            except Exception:
                pass
            return 1

        _pump(root, seconds=1.2)
        if gui.skeleton_player is not None and getattr(gui.skeleton_player, "frame_count", 0) > 0:
            n = max(1, gui.skeleton_player.frame_count - 1)
            gui._go_to(max(0, int(n * 0.25)))
        apply_overview_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
        _pump(root, seconds=0.6)

        overlay = getattr(gui, "_overview_foot_clearance_overlay", None)
        overlay_placed = False
        if overlay is not None:
            try:
                overlay_placed = str(overlay.winfo_manager() or "") == "place"
            except tk.TclError:
                overlay_placed = False

        sizes = (
            ("1366x768", 1366, 768),
            ("1600x900", 1600, 900),
            ("1920x1080", 1920, 1080),
        )
        ok = True
        for label, w, h in sizes:
            try:
                root.state("normal")
            except tk.TclError:
                pass
            root.geometry(f"{w}x{h}+20+20")
            apply_responsive_layout(gui, width=w, height=h)
            apply_overview_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
            _pump(root, seconds=0.7)
            actual = (int(root.winfo_width() or 0), int(root.winfo_height() or 0))
            video = _geom(getattr(gui, "video_frame", None))
            skel = _geom(getattr(gui, "skel_frame", None))
            path = _geom(getattr(gui, "overview_traj_panel", None))
            phase = _geom(getattr(gui, "_overview_phase_strip", None))
            clearance = _geom(getattr(gui, "_foot_clearance_detail_host", None))
            split = getattr(gui, "_overview_viz_split", None)
            split_info = _geom(split) if split is not None else {"mapped": 0}
            print(
                f"{label} actual={actual[0]}x{actual[1]} "
                f"focus={getattr(gui, '_overview_focus_mode', None)} "
                f"layout={getattr(gui, '_overview_viz_layout', None)} "
                f"split={split_info} video={video} skel={skel} path={path} "
                f"phase={phase} clearance={clearance}",
                flush=True,
            )
            if not video.get("mapped") or not skel.get("mapped") or not path.get("mapped"):
                print(f"FAIL {label}: a primary viewer is unmapped", flush=True)
                ok = False
            if int(video.get("w") or 0) < 180 or int(skel.get("w") or 0) < 180:
                print(f"FAIL {label}: viewer too narrow", flush=True)
                ok = False
            if int(video.get("h") or 0) < 140 or int(skel.get("h") or 0) < 140:
                print(f"FAIL {label}: viewer too short", flush=True)
                ok = False
            try:
                x = int(root.winfo_rootx())
                y = int(root.winfo_rooty())
                img = ImageGrab.grab(bbox=(x, y, x + actual[0], y + actual[1]))
                dest = OUT / f"overview_presentation_{label.replace('x', '_')}.png"
                img.save(dest)
                print(f"wrote {dest}", flush=True)
            except Exception as exc:
                print(f"screenshot {label} failed: {exc}", flush=True)
                ok = False

        print(f"overlay_on_video={overlay_placed}", flush=True)
        if overlay_placed:
            print("FAIL: foot-clearance overlay still placed on video", flush=True)
            ok = False
        try:
            root.destroy()
        except Exception:
            pass
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
