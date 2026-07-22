#!/usr/bin/env python3
"""Load Normal demo (cached poses), select Right Hip, save Overview screenshots."""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def _pump(root, *, seconds: float = 0.2) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)


def _dpi_scale(root) -> float:
    """Windows logical→physical scale for ImageGrab bboxes."""
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        hdc = user32.GetDC(0)
        try:
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        finally:
            user32.ReleaseDC(0, hdc)
        if dpi and dpi > 0:
            return float(dpi) / 96.0
    except Exception:
        pass
    try:
        return float(root.winfo_fpixels("1i")) / 72.0
    except Exception:
        return 1.0


def main() -> int:
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.ui.viewers.gait_skeleton_renderer import _VIEW_SPAN_JOINTS, _xy

    out_dir = config.OUTPUT_DIR / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    shot_path = out_dir / "overview_layout_right_hip.png"
    path_canvas_shot = out_dir / "overview_path_canvas.png"
    skel_canvas_shot = out_dir / "overview_skeleton_canvas.png"
    composite_shot = out_dir / "overview_viz_composite.png"

    ex = example_by_key("normal")
    poses_path = config.POSES_DIR / "normal_gait_poses.json"
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
    if not poses_path.is_file():
        print(f"FAILED: missing {poses_path}", file=sys.stderr)
        return 1

    with _suppress_messageboxes():
        gui = StableWalkGUI()
        root = gui.root
        root.geometry("1920x1080+20+20")
        try:
            root.state("zoomed")
        except Exception:
            pass
        apply_responsive_layout(gui, width=1920, height=1080)
        _pump(root, seconds=0.6)

        select_dashboard_tab(gui, TAB_OVERVIEW)
        _pump(root, seconds=0.2)

        gui._active_demo_gait = ex
        gui._presentation_mode = False
        gui._highlight_demo_button("normal")
        source = str(video_path)
        gui.url_var.set(source)
        gui.url_entry.delete(0, "end")
        gui.url_entry.insert(0, source)
        gui._current_source = source
        if not gui.load_poses(poses_path, fresh=True, expected_source=source):
            print("FAILED: load_poses returned False", file=sys.stderr)
            try:
                root.destroy()
            except Exception:
                pass
            return 1

        _pump(root, seconds=1.0)
        if gui.skeleton_player is not None:
            n = max(1, gui.skeleton_player.frame_count - 1)
            gui._go_to(max(0, int(n * 0.25)))
        _pump(root, seconds=0.4)

        gui._focus_joint_trajectory_from_skeleton("right_hip", compare=False)
        _pump(root, seconds=0.8)

        for _ in range(3):
            try:
                gui._fit_skeleton_canvas()
            except Exception:
                pass
            try:
                gui._render_overview_traj_canvas(force=True)
            except Exception:
                pass
            _pump(root, seconds=0.35)

        video = getattr(gui, "video_frame", None)
        skel = getattr(gui, "skel_frame", None)
        path = getattr(gui, "overview_traj_panel", None)
        host = getattr(gui, "_section_visual", None)
        metrics = getattr(gui, "_overview_metrics_row", None)
        tab = getattr(gui, "_tab_overview", None)

        def _box(w):
            if w is None:
                return None
            try:
                w.update_idletasks()
                return (
                    int(w.winfo_width()),
                    int(w.winfo_height()),
                    bool(w.winfo_ismapped()),
                )
            except Exception:
                return None

        ax = gui.ax_3d
        snap = gui.skeleton_player.current_snapshot() if gui.skeleton_player else None
        fill_y = None
        if snap is not None:
            xs, ys = [], []
            for jid in _VIEW_SPAN_JOINTS:
                pt = _xy(snap, jid)
                if pt:
                    xs.append(pt[0])
                    ys.append(pt[1])
            if ys:
                body_h = max(ys) - min(ys)
                view_h = ax.get_ylim()[1] - ax.get_ylim()[0]
                fill_y = body_h / max(view_h, 1e-9)

        report = {
            "video": _box(video),
            "skeleton": _box(skel),
            "path": _box(path),
            "viz_host": _box(host),
            "metrics": _box(metrics),
            "tab": _box(tab),
            "skel_host": _box(getattr(gui, "skel_canvas_host", None)),
            "path_host": _box(getattr(gui, "overview_traj_canvas_host", None)),
            "selected": sorted(gui.selection.selected),
            "path_title": "",
            "body_fill_y": fill_y,
            "ax_pos": tuple(ax.get_position().bounds),
            "metrics_expanded": bool(getattr(gui, "_overview_metrics_expanded", False)),
        }
        try:
            if path is not None:
                report["path_title"] = str(path.cget("text"))
        except Exception:
            pass
        print("LAYOUT", report)

        failures: list[str] = []
        if not report["path"] or not report["path"][2]:
            failures.append("3D path panel not mapped")
        if report["path"] and report["path"][1] < 280:
            failures.append(f"3D path panel too short: {report['path'][1]}px")
        if report["skeleton"] and report["skeleton"][1] < 280:
            failures.append(f"skeleton panel too short: {report['skeleton'][1]}px")
        if report["viz_host"] and report["metrics"]:
            vh, mh = report["viz_host"][1], report["metrics"][1]
            share = mh / max(vh + mh, 1)
            print(f"HEIGHT_SHARE viz={vh} metrics={mh} metrics_share={share:.1%}")
            # Collapsed gait strip should stay thin; expanded still ≤ ~30%.
            if share > 0.32:
                failures.append(
                    f"metrics share too large: metrics={mh} viz={vh} ({share:.0%})"
                )
        if fill_y is not None and not (0.62 <= fill_y <= 0.78):
            failures.append(f"skeleton body fill out of range: {fill_y:.1%}")
        # Axes must use most of the figure width (not a letterboxed square).
        ax_w = float(report["ax_pos"][2])
        if ax_w < 0.85:
            failures.append(f"skeleton axes too narrow (letterboxed): width={ax_w:.2f}")
        if "right_hip" not in report["selected"]:
            failures.append("right_hip not selected")
        if "Right Hip" not in report["path_title"]:
            failures.append(f"unexpected path title: {report['path_title']!r}")

        # Canvas dumps (reliable) + DPI-aware window grab + side-by-side composite.
        try:
            gui.canvas_dof_traj_overview.print_png(str(path_canvas_shot))
            print(f"SCREENSHOT {path_canvas_shot}")
        except Exception as exc:
            failures.append(f"path canvas dump failed: {exc}")
        try:
            gui.canvas_3d.print_png(str(skel_canvas_shot))
            print(f"SCREENSHOT {skel_canvas_shot}")
        except Exception as exc:
            failures.append(f"skeleton canvas dump failed: {exc}")

        try:
            from PIL import Image, ImageGrab

            # Side-by-side proof from Matplotlib buffers (not ScreenGrab): DPI
            # scaling on Windows often crops child widgets and fakes "overzoom".
            strips = []
            for png in (skel_canvas_shot, path_canvas_shot):
                if Path(png).is_file():
                    strips.append(Image.open(png).convert("RGB"))
            video_lbl = getattr(gui, "video_label", None)
            if video_lbl is not None:
                try:
                    video_lbl.update_idletasks()
                    scale = _dpi_scale(root)
                    x = int(video_lbl.winfo_rootx() * scale)
                    y = int(video_lbl.winfo_rooty() * scale)
                    w = max(1, int(video_lbl.winfo_width() * scale))
                    h = max(1, int(video_lbl.winfo_height() * scale))
                    strips.insert(0, ImageGrab.grab(bbox=(x, y, x + w, y + h)).convert("RGB"))
                except Exception:
                    pass

            if len(strips) >= 2:
                target_h = max(im.height for im in strips)
                resized = []
                for im in strips:
                    if im.height != target_h:
                        new_w = max(1, int(im.width * (target_h / im.height)))
                        im = im.resize((new_w, target_h))
                    resized.append(im)
                total_w = sum(im.width for im in resized) + 8 * (len(resized) - 1)
                composite = Image.new("RGB", (total_w, target_h), (32, 34, 40))
                x0 = 0
                for im in resized:
                    composite.paste(im, (x0, 0))
                    x0 += im.width + 8
                composite.save(composite_shot)
                print(f"SCREENSHOT {composite_shot}")

            root.update_idletasks()
            scale = _dpi_scale(root)
            x = int(root.winfo_rootx() * scale)
            y = int(root.winfo_rooty() * scale)
            w = int(root.winfo_width() * scale)
            h = int(root.winfo_height() * scale)
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
            img.save(shot_path)
            print(f"SCREENSHOT {shot_path}")
        except Exception as exc:
            print(f"PIL grab skipped: {exc}")

        try:
            root.destroy()
        except Exception:
            pass

        if failures:
            print("FAILED:", "; ".join(failures), file=sys.stderr)
            return 1
        print("OK overview layout screenshot validation passed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
