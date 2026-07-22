#!/usr/bin/env python3
"""Overview visual regression — open the real GUI, cycle gaits/joints/sizes.

Captures skeleton + 3D path canvases (and a composite) after each selection.
Does not change analysis math; inspection aid for skeleton + trajectory QA.
"""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Performance demo is athletic_walking.mp4 in this project.
GAITS = (
    ("normal", "normal_gait.mp4", "normal_gait_poses.json"),
    ("abnormal", "abnormal_gait.mp4", "abnormal_gait_poses.json"),
    ("athletic", "athletic_walking.mp4", "athletic_walking_poses.json"),
)
JOINTS = ("right_hip", "right_knee", "right_ankle")
SIZES = (
    ("maximized", 1920, 1080, True),
    ("medium", 1366, 768, False),
)


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
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        hdc = user32.GetDC(0)
        try:
            dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)
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


def _body_fill_y(gui) -> float | None:
    from stablewalk.ui.viewers.gait_skeleton_renderer import _VIEW_SPAN_JOINTS, _xy

    ax = getattr(gui, "ax_3d", None)
    player = getattr(gui, "skeleton_player", None)
    if ax is None or player is None:
        return None
    snap = player.current_snapshot()
    if snap is None:
        return None
    ys = []
    for jid in _VIEW_SPAN_JOINTS:
        pt = _xy(snap, jid)
        if pt:
            ys.append(pt[1])
    if not ys:
        return None
    body_h = max(ys) - min(ys)
    view_h = ax.get_ylim()[1] - ax.get_ylim()[0]
    return body_h / max(view_h, 1e-9)


def _traj_fill_metrics(gui) -> dict[str, float | None]:
    ax = getattr(gui, "ax_dof_traj_overview", None)
    if ax is None:
        return {}
    try:
        xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    except Exception:
        return {}
    # Best-effort: use current line/collection extents via axis spans.
    return {
        "xlim_cm": (xl[1] - xl[0]) * 100.0,
        "ylim_cm": (yl[1] - yl[0]) * 100.0,
        "zlim_cm": (zl[1] - zl[0]) * 100.0,
        "elev": float(getattr(ax, "elev", 0.0) or 0.0),
        "azim": float(getattr(ax, "azim", 0.0) or 0.0),
    }


def _save_composite(skel_png: Path, path_png: Path, out: Path) -> None:
    from PIL import Image

    strips = []
    for png in (skel_png, path_png):
        if png.is_file():
            strips.append(Image.open(png).convert("RGB"))
    if len(strips) < 2:
        return
    target_h = max(im.height for im in strips)
    resized = []
    for im in strips:
        if im.height != target_h:
            new_w = max(1, int(im.width * (target_h / im.height)))
            im = im.resize((new_w, target_h))
        resized.append(im)
    total_w = sum(im.width for im in resized) + 8
    composite = Image.new("RGB", (total_w, target_h), (32, 34, 40))
    x0 = 0
    for im in resized:
        composite.paste(im, (x0, 0))
        x0 += im.width + 8
    composite.save(out)


def _load_gait(gui, *, key: str, video_name: str, poses_name: str) -> bool:
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key

    ex = example_by_key(key)
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / video_name
    poses_path = config.POSES_DIR / poses_name
    if not poses_path.is_file():
        print(f"SKIP {key}: missing poses {poses_path}")
        return False
    if not Path(video_path).is_file():
        print(f"SKIP {key}: missing video {video_path}")
        return False

    gui._active_demo_gait = ex
    gui._presentation_mode = False
    try:
        gui._highlight_demo_button(key)
    except Exception:
        pass
    source = str(video_path)
    gui.url_var.set(source)
    gui.url_entry.delete(0, "end")
    gui.url_entry.insert(0, source)
    gui._current_source = source
    ok = gui.load_poses(poses_path, fresh=True, expected_source=source)
    if not ok:
        print(f"FAIL {key}: load_poses returned False")
        return False
    return True


def main() -> int:
    from stablewalk import config
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    out_dir = config.OUTPUT_DIR / "reports" / "overview_visual_regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = ["# Overview visual regression", ""]
    failures: list[str] = []
    captured = 0

    with _suppress_messageboxes():
        gui = StableWalkGUI()
        root = gui.root
        select_dashboard_tab(gui, TAB_OVERVIEW)
        _pump(root, seconds=0.4)

        for key, video_name, poses_name in GAITS:
            if not _load_gait(gui, key=key, video_name=video_name, poses_name=poses_name):
                if key == "athletic":
                    failures.append(
                        "performance/athletic: athletic_walking_poses.json missing — "
                        "could not open Performance gait for visual check"
                    )
                else:
                    failures.append(f"{key}: could not load")
                continue

            _pump(root, seconds=0.8)
            if gui.skeleton_player is not None:
                n = max(1, gui.skeleton_player.frame_count - 1)
                gui._go_to(max(0, int(n * 0.25)))
            _pump(root, seconds=0.3)

            for size_name, width, height, zoomed in SIZES:
                try:
                    if zoomed and size_name == "maximized":
                        root.state("zoomed")
                    else:
                        try:
                            root.state("normal")
                        except Exception:
                            pass
                        root.geometry(f"{width}x{height}+40+40")
                except Exception:
                    root.geometry(f"{width}x{height}+40+40")
                apply_responsive_layout(gui, width=width, height=height)
                _pump(root, seconds=0.5)

                for joint in JOINTS:
                    print(f"CASE {key} {size_name} {joint}")
                    try:
                        gui._focus_joint_trajectory_from_skeleton(joint, compare=False)
                    except Exception as exc:
                        failures.append(f"{key}/{size_name}/{joint}: select failed: {exc}")
                        continue

                    for _ in range(2):
                        try:
                            gui._fit_skeleton_canvas()
                        except Exception:
                            pass
                        try:
                            gui._render_overview_traj_canvas(force=True)
                        except Exception:
                            pass
                        _pump(root, seconds=0.3)

                    stem = f"{key}_{size_name}_{joint}"
                    skel_png = out_dir / f"{stem}_skeleton.png"
                    path_png = out_dir / f"{stem}_path.png"
                    comp_png = out_dir / f"{stem}_composite.png"
                    win_png = out_dir / f"{stem}_window.png"

                    try:
                        gui.canvas_3d.print_png(str(skel_png))
                        gui.canvas_dof_traj_overview.print_png(str(path_png))
                        _save_composite(skel_png, path_png, comp_png)
                        captured += 1
                    except Exception as exc:
                        failures.append(f"{stem}: canvas dump failed: {exc}")
                        continue

                    try:
                        from PIL import ImageGrab

                        root.update_idletasks()
                        scale = _dpi_scale(root)
                        x = int(root.winfo_rootx() * scale)
                        y = int(root.winfo_rooty() * scale)
                        w = int(root.winfo_width() * scale)
                        h = int(root.winfo_height() * scale)
                        ImageGrab.grab(bbox=(x, y, x + w, y + h)).save(win_png)
                    except Exception as exc:
                        print(f"window grab skipped ({stem}): {exc}")

                    fill = _body_fill_y(gui)
                    traj = _traj_fill_metrics(gui)
                    selected = sorted(getattr(gui.selection, "selected", []) or [])
                    title = ""
                    panel = getattr(gui, "overview_traj_panel", None)
                    try:
                        if panel is not None:
                            title = str(panel.cget("text"))
                    except Exception:
                        pass
                    path_host = getattr(gui, "overview_traj_canvas_host", None)
                    skel_host = getattr(gui, "skel_canvas_host", None)
                    path_box = None
                    skel_box = None
                    try:
                        if path_host is not None:
                            path_host.update_idletasks()
                            path_box = (path_host.winfo_width(), path_host.winfo_height())
                        if skel_host is not None:
                            skel_host.update_idletasks()
                            skel_box = (skel_host.winfo_width(), skel_host.winfo_height())
                    except Exception:
                        pass

                    line = (
                        f"- **{stem}**: selected={selected} title={title!r} "
                        f"body_fill={None if fill is None else f'{fill:.1%}'} "
                        f"skel_host={skel_box} path_host={path_box} "
                        f"traj_lim_cm="
                        f"X{traj.get('xlim_cm'):.1f}/Y{traj.get('ylim_cm'):.1f}/"
                        f"Z{traj.get('zlim_cm'):.1f} "
                        f"cam=({traj.get('elev'):.0f},{traj.get('azim'):.0f}) "
                        f"shots=`{skel_png.name}`, `{path_png.name}`, `{comp_png.name}`"
                    )
                    report_lines.append(line)
                    print(line)

                    if fill is not None and not (0.55 <= fill <= 0.82):
                        failures.append(f"{stem}: body_fill out of range ({fill:.1%})")
                    if path_box and path_box[1] < 220:
                        failures.append(f"{stem}: path host too short ({path_box[1]}px)")
                    if joint.replace("_", " ").title().replace("Right ", "Right ") not in title and (
                        "Hip" not in title and "Knee" not in title and "Ankle" not in title
                    ):
                        # Loose check — title should mention joint family.
                        if joint.split("_")[-1].capitalize() not in title:
                            failures.append(f"{stem}: path title missing joint: {title!r}")

        try:
            root.destroy()
        except Exception:
            pass

    report_lines.append("")
    report_lines.append(f"Captured case composites: {captured}")
    if failures:
        report_lines.append("")
        report_lines.append("## Automated flags")
        for f in failures:
            report_lines.append(f"- {f}")
    report_path = out_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"REPORT {report_path}")
    print(f"CAPTURED {captured}")
    if failures:
        print("FLAGS:", "; ".join(failures))
    # Missing performance poses is a documented skip, not a hard fail if others ran.
    hard = [f for f in failures if not f.startswith("performance/athletic")]
    return 1 if hard else 0


if __name__ == "__main__":
    raise SystemExit(main())
