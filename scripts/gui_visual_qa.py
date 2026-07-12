#!/usr/bin/env python3
"""
StableWalk GUI visual QA — automated workflow + layout regression.

Runs the scripted interaction checklist for Abnormal / Normal / Performance demos
and writes ``data/output/reports/gui_visual_qa.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_KEYS = (
    ("abnormal", "Abnormal", "abnormal_gait.mp4"),
    ("normal", "Normal", "normal_gait.mp4"),
    ("athletic", "Performance", "athletic_walking.mp4"),
)

RESOLUTIONS = (
    ("1920x1080", 1920, 1080),
    ("1600x900", 1600, 900),
    ("1366x768", 1366, 768),
)

ISSUES_FIXED_THIS_SESSION = [
    "Unified Detailed Joint Data & Export section with three control groups.",
    "Dynamic scroll bottom padding clears fixed playback transport + status bar.",
    "Export buttons use Export.TButton style with visible disabled state and tooltips.",
    "Foot clearance panel shows CONTACT/SWING, unavailable reasons, and confidence.",
]

REMAINING_LIMITATIONS = [
    "Automated QA cannot detect matplotlib plot ghosting or subtle text-on-plot overlap; "
    "manual playback review is still advised.",
    "Summary comparability badge is not a dedicated widget — derived from gait evidence in Details only.",
    "Demo pipeline uses up to 120 frames per load (DEMO_MAX_FRAMES); full-length scroll "
    "behavior on very long clips is not re-tested here.",
]


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


def _pump(root, *, seconds: float = 0.05) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)


def _wait_until(root, predicate, *, timeout: float = 600.0, poll: float = 0.1) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        root.update()
        if predicate():
            return True
        time.sleep(poll)
    return False


def _center_window(root, width: int, height: int) -> None:
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    x = max(0, (sw - width) // 2)
    y = max(0, (sh - height) // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    root.update_idletasks()


def _run_demo_workflow(
    app,
    *,
    key: str,
    label: str,
    video_name: str,
    width: int,
    height: int,
    use_cached_poses: bool,
    pipeline_timeout: float,
) -> "GuiVisualQAResult":
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.tk.gui_visual_qa import (
        GuiVisualQAResult,
        run_gui_regression_assertions,
        scroll_dashboard_to,
    )
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout

    resolution = f"{width}x{height}"
    result = GuiVisualQAResult(
        category=label,
        resolution=resolution,
        video=video_name,
    )

    ex = example_by_key(key)
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / video_name
    poses_path = config.POSES_DIR / video_name.replace(".mp4", "_poses.json")

    with _suppress_messageboxes():
        result.workflow_steps.append(f"Load demo video ({label})")
        if use_cached_poses and poses_path.is_file():
            app._active_demo_gait = ex
            app._presentation_mode = False
            app._highlight_demo_button(key)
            source = str(video_path)
            app.url_var.set(source)
            app.url_entry.delete(0, "end")
            app.url_entry.insert(0, source)
            app._current_source = source
            if not app.load_poses(poses_path, fresh=True, expected_source=source):
                result.issues_found.append(f"Failed to load poses for {label}")
                return result
            result.workflow_steps.append(f"Loaded cached poses: {poses_path.name}")
        else:
            app._load_demo_gait(key)
            ok = _wait_until(
                app.root,
                lambda: not app._processing and app.sequence is not None,
                timeout=pipeline_timeout,
            )
            if not ok:
                result.issues_found.append(
                    f"Pipeline timeout ({pipeline_timeout:.0f}s) for {label}"
                )
                if poses_path.is_file():
                    app.load_poses(poses_path, fresh=True, expected_source=str(video_path))
                    result.workflow_steps.append(
                        f"Fallback to cached poses after timeout: {poses_path.name}"
                    )
                else:
                    result.workflow_steps.append("Analyze failed — no cached poses")
                    result.checks.extend(
                        run_gui_regression_assertions(
                            app, resolution=resolution, scrolled_to_bottom=False
                        )
                    )
                    return result
            else:
                result.workflow_steps.append("Analyze completed")

        if not app.sequence or not app.skeleton_player:
            result.issues_found.append("No sequence/skeleton player after load")
            return result

        _center_window(app.root, width, height)
        apply_responsive_layout(app, width=width, height=height)
        result.workflow_steps.append(f"Set window to {resolution}")

        # Seek to ~25% (avoid play-loop timer during automated QA)
        n = max(1, app.skeleton_player.frame_count - 1)
        target = max(0, int(n * 0.25))
        app._go_to(target)
        result.workflow_steps.append(f"Seek to frame {target} (~25%)")
        _pump(app.root, seconds=0.1)

        scroll_dashboard_to(app, 0.5)
        result.workflow_steps.append("Scroll to middle")
        _pump(app.root, seconds=0.2)

        scroll_dashboard_to(app, 1.0)
        result.workflow_steps.append("Scroll to bottom")
        from stablewalk.ui.tk.dashboard_responsive import _sync_scroll_bottom_clearance

        _sync_scroll_bottom_clearance(app)
        _pump(app.root, seconds=0.3)

        scroll_dashboard_to(app, 0.0)
        result.workflow_steps.append("Scroll to top")
        _pump(app.root, seconds=0.2)

        result.workflow_steps.append("Seek forward briefly")
        app._go_to(min(n, target + 2))
        _pump(app.root, seconds=0.05)

        result.workflow_steps.append("Select Right Knee")
        app._select_charted_dof_item("right_knee")
        _pump(app.root, seconds=0.2)

        result.workflow_steps.append("Select Left Ankle")
        app._select_charted_dof_item("left_ankle")
        _pump(app.root, seconds=0.2)

        if hasattr(app, "cmb_dof_projection"):
            app.var_dof_projection.set("Sagittal Plane")
            app._on_dof_projection_changed()
            result.workflow_steps.append("Joint path view → Sagittal Plane")
            _pump(app.root, seconds=0.2)

        result.workflow_steps.append("Open detailed data dialog")
        app._toggle_collected_data_table()
        _pump(app.root, seconds=0.2)
        dlg = getattr(app, "_collected_data_dialog", None)
        if dlg is not None:
            try:
                if dlg.winfo_exists():
                    dlg.destroy()
            except Exception:
                pass
        app._collected_data_dialog = None

        scroll_dashboard_to(app, 1.0)
        _sync_scroll_bottom_clearance(app)
        result.workflow_steps.append("Scroll to export section")
        _pump(app.root, seconds=0.3)

        result.workflow_steps.append("Resize window (+/- 80px)")
        _center_window(app.root, max(1024, width - 80), max(700, height - 60))
        apply_responsive_layout(app)
        _pump(app.root, seconds=0.2)
        _center_window(app.root, width, height)
        apply_responsive_layout(app)
        _pump(app.root, seconds=0.2)

        try:
            app.root.state("zoomed")
            result.workflow_steps.append("Maximize window")
            _pump(app.root, seconds=0.3)
            apply_responsive_layout(app)
            scroll_dashboard_to(app, 1.0)
            _sync_scroll_bottom_clearance(app)
            _pump(app.root, seconds=0.2)
            app.root.state("normal")
            result.workflow_steps.append("Restore window")
            _center_window(app.root, width, height)
            apply_responsive_layout(app)
        except Exception as exc:
            result.issues_found.append(f"Maximize/restore skipped: {exc}")

        scroll_dashboard_to(app, 1.0)
        _sync_scroll_bottom_clearance(app)
        _pump(app.root, seconds=0.2)

        from stablewalk.ui.tk.gui_visual_qa import run_scroll_layout_stress

        for check in run_scroll_layout_stress(app, cycles=20):
            result.add_check(check)
        result.workflow_steps.append("Scroll stress: 20 top/bottom cycles + resize")

        app._refresh_bilateral_ground_clearance()
        _pump(app.root, seconds=0.1)

        checks = run_gui_regression_assertions(
            app, resolution=resolution, scrolled_to_bottom=True
        )
        for check in checks:
            result.add_check(check)

    return result


def _run_resize_export_pass(
    app,
    *,
    label: str,
    video_name: str,
    width: int,
    height: int,
) -> "GuiVisualQAResult":
    """Scroll/export/regression checks at a resolution (assumes session already loaded)."""
    from stablewalk.ui.tk.dashboard_responsive import (
        _sync_scroll_bottom_clearance,
        apply_responsive_layout,
    )
    from stablewalk.ui.tk.gui_visual_qa import (
        GuiVisualQAResult,
        run_gui_regression_assertions,
        scroll_dashboard_to,
    )

    resolution = f"{width}x{height}"
    result = GuiVisualQAResult(
        category=f"{label} (resize pass)",
        resolution=resolution,
        video=video_name,
    )
    _center_window(app.root, width, height)
    apply_responsive_layout(app, width=width, height=height)
    result.workflow_steps.append(f"Resize to {resolution}")

    scroll_dashboard_to(app, 1.0)
    _sync_scroll_bottom_clearance(app)
    _pump(app.root, seconds=0.15)
    result.workflow_steps.append("Scroll to export section")

    for check in run_gui_regression_assertions(
        app, resolution=resolution, scrolled_to_bottom=True
    ):
        result.add_check(check)
    app._refresh_bilateral_ground_clearance()
    _pump(app.root, seconds=0.05)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="StableWalk GUI visual QA")
    parser.add_argument(
        "--resolution",
        default="all",
        choices=["all", "1920x1080", "1600x900", "1366x768"],
        help="Test resolution (default: all)",
    )
    parser.add_argument(
        "--use-cached-poses",
        action="store_true",
        help="Load cached pose JSON after demo selection (faster; skips full re-estimation)",
    )
    parser.add_argument(
        "--pipeline-timeout",
        type=float,
        default=480.0,
        help="Seconds to wait per demo video pipeline (default: 480)",
    )
    parser.add_argument(
        "--demos",
        default="all",
        help="Comma-separated demo keys: abnormal,normal,athletic (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    from stablewalk import config
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.gui_visual_qa import format_qa_report

    config.ensure_output_dirs()
    out_path = args.output or (config.REPORTS_DIR / "gui_visual_qa.md")

    if args.resolution == "all":
        resolutions = RESOLUTIONS
    else:
        resolutions = tuple(r for r in RESOLUTIONS if r[0] == args.resolution)

    primary_res = resolutions[0]
    extra_res = resolutions[1:]

    if args.demos == "all":
        demos = DEMO_KEYS
    else:
        keys = {k.strip() for k in args.demos.split(",")}
        demos = tuple(d for d in DEMO_KEYS if d[0] in keys)

    results = []
    root = __import__("tkinter").Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    app.root.deiconify()

    try:
        with _suppress_messageboxes():
            p_w, p_h = primary_res[1], primary_res[2]
            for key, label, video_name in demos:
                print(f"QA: {label} @ {primary_res[0]} …", flush=True)
                result = _run_demo_workflow(
                    app,
                    key=key,
                    label=label,
                    video_name=video_name,
                    width=p_w,
                    height=p_h,
                    use_cached_poses=args.use_cached_poses,
                    pipeline_timeout=args.pipeline_timeout,
                )
                results.append(result)
                status = "PASS" if result.passed else "FAIL"
                print(
                    f"  → {status} ({sum(1 for c in result.checks if c.passed)}/{len(result.checks)})",
                    flush=True,
                )

            if extra_res:
                last_label, last_video = demos[-1][1], demos[-1][2]
                for res_name, width, height in extra_res:
                    print(f"QA: resize/export @ {res_name} …", flush=True)
                    r = _run_resize_export_pass(
                        app,
                        label=last_label,
                        video_name=last_video,
                        width=width,
                        height=height,
                    )
                    results.append(r)
                    status = "PASS" if r.passed else "FAIL"
                    print(
                        f"  → {status} ({sum(1 for c in r.checks if c.passed)}/{len(r.checks)})",
                        flush=True,
                    )
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    report = format_qa_report(
        results,
        issues_fixed=ISSUES_FIXED_THIS_SESSION,
        remaining_limitations=REMAINING_LIMITATIONS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path.resolve()}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
