"""Load normal/abnormal gait and verify skeleton + 3D path stay in view."""

from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@contextlib.contextmanager
def _suppress_messageboxes():
    import tkinter.messagebox as mb

    originals = (mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel)
    mb.showinfo = lambda *a, **k: True  # type: ignore[assignment]
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
        try:
            root.update()
        except Exception:
            break
        time.sleep(0.01)


def _clip_geom(widget) -> dict[str, int]:
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


def _skeleton_in_view(gui, failures: list[str], tag: str) -> None:
    from stablewalk.ui.viewers.gait_skeleton_renderer import _collect_body_view_points

    ax = getattr(gui, "ax_3d", None)
    player = getattr(gui, "skeleton_player", None)
    if ax is None or player is None:
        failures.append(f"{tag}: skeleton axes/player missing")
        return
    snap = player.current_snapshot()
    if snap is None:
        failures.append(f"{tag}: no skeleton snapshot")
        return
    points = _collect_body_view_points(snap)
    if not points:
        failures.append(f"{tag}: no body points")
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if min(xs) < xlim[0] or max(xs) > xlim[1]:
        failures.append(
            f"{tag}: skeleton X cropped ({min(xs):.3f}..{max(xs):.3f} vs {xlim})"
        )
    if min(ys) < ylim[0] or max(ys) > ylim[1]:
        failures.append(
            f"{tag}: skeleton Y cropped ({min(ys):.3f}..{max(ys):.3f} vs {ylim})"
        )
    body_h = max(ys) - min(ys)
    view_h = ylim[1] - ylim[0]
    fill = body_h / max(view_h, 1e-9)
    print(
        f"  {tag} body_fill={fill:.1%} ylim={ylim[0]:.3f}..{ylim[1]:.3f} "
        f"y={min(ys):.3f}..{max(ys):.3f}"
    )
    if fill < 0.45:
        failures.append(f"{tag}: skeleton too small in view ({fill:.1%})")
    if fill > 0.97:
        failures.append(f"{tag}: skeleton overfilled / likely clipped ({fill:.1%})")


def _trajectory_in_view(gui, failures: list[str], tag: str) -> None:
    ax = getattr(gui, "ax_dof_traj_overview", None)
    if ax is None:
        failures.append(f"{tag}: overview trajectory axes missing")
        return
    artists = getattr(ax, "_stablewalk_traj_artists", None)
    line = getattr(artists, "path_line", None) if artists is not None else None
    if line is None:
        print(f"  {tag} trajectory path not drawn yet")
        return
    xs = list(line.get_data_3d()[0]) if hasattr(line, "get_data_3d") else list(line.get_xdata())
    if not xs:
        print(f"  {tag} trajectory empty")
        return
    try:
        ys = list(line.get_data_3d()[1])
        zs = list(line.get_data_3d()[2])
    except Exception:
        ys = list(line.get_ydata())
        zs = [0.0] * len(xs)
    xlim, ylim, zlim = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
    for name, vals, lim in (
        ("X", xs, xlim),
        ("Y", ys, ylim),
        ("Z", zs, zlim),
    ):
        if min(vals) < lim[0] - 1e-6 or max(vals) > lim[1] + 1e-6:
            failures.append(
                f"{tag}: trajectory {name} cropped ({min(vals):.4f}..{max(vals):.4f} vs {lim})"
            )
    print(
        f"  {tag} traj n={len(xs)} xlim={xlim[0]:.3f}..{xlim[1]:.3f} "
        f"ylim={ylim[0]:.3f}..{ylim[1]:.3f}"
    )


def _load_clip(gui, name: str) -> bool:
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key

    key = "normal" if name.startswith("normal") else "abnormal"
    ex = example_by_key(key)
    poses = config.POSES_DIR / f"{name}_poses.json"
    video = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / f"{name}.mp4"
    source = str(video)
    gui.url_var.set(source)
    gui._current_source = source
    if not poses.is_file():
        print(f"MISSING poses {poses}")
        return False
    try:
        return bool(
            gui.load_poses(
                poses, fresh=True, expected_source=source, wait_for_analysis=True
            )
        )
    except TypeError:
        return bool(gui.load_poses(poses, fresh=True))


def main() -> int:
    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_overview_view_mode import (
        VIEW_MODE_OVERLAY,
        VIEW_MODE_SIDE_BY_SIDE,
        VIEW_MODE_SKELETON_ONLY,
        apply_overview_view_mode,
    )
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    failures: list[str] = []
    clips = ("normal_gait", "abnormal_gait")

    with _suppress_messageboxes():
        gui = StableWalkGUI()
        root = gui.root
        watchdog = root.after(180_000, root.destroy)
        root.geometry("1600x900+20+20")
        _pump(root, seconds=0.4)
        select_dashboard_tab(gui, TAB_OVERVIEW)
        apply_overview_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
        _pump(root, seconds=0.2)

        for clip in clips:
            print(f"== {clip} ==")
            if not _load_clip(gui, clip):
                failures.append(f"{clip}: load failed")
                continue
            _pump(root, seconds=1.0)
            gui._activate_dof_item("right_knee", add_if_missing=True)
            _pump(root, seconds=0.5)
            gui._fit_full_body()
            gui._fit_overview_trajectory(keep_camera=True)
            _pump(root, seconds=0.5)

            n = int(getattr(gui.skeleton_player, "frame_count", 0) or 0)
            print(f"  frames={n}")
            if n <= 0:
                failures.append(f"{clip}: no frames")
                continue
            for frac in (0.0, 0.25, 0.55, 0.9):
                frame = min(n - 1, max(0, int(frac * (n - 1))))
                gui._go_to(frame)
                _pump(root, seconds=0.25)
                _skeleton_in_view(gui, failures, f"{clip}@f{frame}")
                _trajectory_in_view(gui, failures, f"{clip}@f{frame}")

            skel = _clip_geom(getattr(gui, "skel_frame", None))
            path = _clip_geom(getattr(gui, "overview_traj_panel", None))
            video = _clip_geom(getattr(gui, "video_frame", None))
            print(f"  geom video={video} skeleton={skel} path={path}")
            if skel["h"] < 160:
                failures.append(f"{clip}: skeleton panel too short ({skel['h']})")
            if path["h"] < 140:
                failures.append(f"{clip}: path panel too short ({path['h']})")

            fit_skel = getattr(gui, "_skel_fit_full_body_btn", None)
            if fit_skel is None:
                failures.append(f"{clip}: Fit Full Body button missing")
            cam_bar = getattr(gui, "overview_traj_camera_bar", None)
            if cam_bar is None:
                failures.append(f"{clip}: trajectory camera bar missing")
            else:
                cg = _clip_geom(cam_bar)
                pg = _clip_geom(getattr(gui, "_overview_path_toolbar", None))
                if cg["w"] > 0 and pg["w"] > 0 and cg["w"] > pg["w"] + 8:
                    failures.append(
                        f"{clip}: 3D camera bar overflows toolbar "
                        f"({cg['w']} > {pg['w']})"
                    )

            try:
                from stablewalk.ui.tk.dashboard_panel_maximize import (
                    is_panel_maximized,
                    maximize_panel,
                    restore_panel,
                )

                maximize_panel(gui, "skeleton")
                _pump(root, seconds=0.4)
                if not is_panel_maximized(gui):
                    failures.append(f"{clip}: skeleton maximize failed")
                gui._fit_full_body()
                _pump(root, seconds=0.25)
                _skeleton_in_view(gui, failures, f"{clip}/maximized")
                restore_panel(gui)
                _pump(root, seconds=0.35)
            except Exception as exc:
                failures.append(f"{clip}: maximize check error: {exc}")

            for mode in (
                VIEW_MODE_SKELETON_ONLY,
                VIEW_MODE_OVERLAY,
                VIEW_MODE_SIDE_BY_SIDE,
            ):
                apply_overview_view_mode(gui, mode)
                _pump(root, seconds=0.35)
                gui._fit_full_body()
                _pump(root, seconds=0.2)
                _skeleton_in_view(gui, failures, f"{clip}/{mode}")

            root.geometry("1100x720+20+20")
            apply_responsive_layout(gui, width=1100, height=720)
            _pump(root, seconds=0.5)
            gui._fit_full_body()
            gui._fit_overview_trajectory(keep_camera=True)
            _pump(root, seconds=0.3)
            _skeleton_in_view(gui, failures, f"{clip}/1100x720")
            root.geometry("1600x900+20+20")
            apply_responsive_layout(gui, width=1600, height=900)
            apply_overview_view_mode(gui, VIEW_MODE_SIDE_BY_SIDE)
            _pump(root, seconds=0.4)

        try:
            root.after_cancel(watchdog)
        except Exception:
            pass
        try:
            root.destroy()
        except tk.TclError:
            pass

    if failures:
        print("FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
