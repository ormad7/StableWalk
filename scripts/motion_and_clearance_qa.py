#!/usr/bin/env python3
"""
Final validation: Overview + Motion Analysis tabs, foot clearance parity,
contact/gait-phase consistency, and Selected Joint 3D trajectory.

Writes ``data/output/reports/motion_and_clearance_qa.md``.
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

DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


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


def _pump(root, *, seconds: float = 0.15) -> None:
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


def _load_demo(app, *, key: str, video_name: str, pipeline_timeout: float) -> bool:
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key

    ex = example_by_key(key)
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / video_name
    poses_path = config.POSES_DIR / video_name.replace(".mp4", "_poses.json")

    if poses_path.is_file():
        app._active_demo_gait = ex
        app._presentation_mode = False
        app._highlight_demo_button(key)
        source = str(video_path)
        app.url_var.set(source)
        app.url_entry.delete(0, "end")
        app.url_entry.insert(0, source)
        app._current_source = source
        return app.load_poses(poses_path, fresh=True, expected_source=source)

    app._load_demo_gait(key)
    return _wait_until(
        app.root,
        lambda: not app._processing and app.sequence is not None,
        timeout=pipeline_timeout,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Overview + Motion Analysis final QA")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: data/output/reports/motion_and_clearance_qa.md)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--pipeline-timeout", type=float, default=600.0)
    args = parser.parse_args()

    from stablewalk import config
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.ui.tk.gui_visual_qa import (
        capture_motion_clearance_qa,
        format_motion_and_clearance_qa_report,
    )

    config.ensure_output_dirs()
    out_path = args.output or (config.REPORTS_DIR / "motion_and_clearance_qa.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()

    snapshots = []

    try:
        with _suppress_messageboxes():
            for key, label, video_name in DEMO_KEYS:
                print(f"Validating: {label} ({video_name})...")
                if not _load_demo(
                    app,
                    key=key,
                    video_name=video_name,
                    pipeline_timeout=args.pipeline_timeout,
                ):
                    from stablewalk.ui.tk.gui_visual_qa import MotionClearanceQASnapshot

                    snap = MotionClearanceQASnapshot(
                        video=video_name,
                        label=label,
                        issues=["Failed to load demo recording"],
                    )
                    snapshots.append(snap)
                    print("  FAIL — load")
                    continue

                if not app.skeleton_player:
                    from stablewalk.ui.tk.gui_visual_qa import MotionClearanceQASnapshot

                    snapshots.append(
                        MotionClearanceQASnapshot(
                            video=video_name,
                            label=label,
                            issues=["No skeleton player after load"],
                        )
                    )
                    print("  FAIL — no player")
                    continue

                biomech = getattr(app, "_biomech", None)
                if biomech is not None:
                    app._update_stability_panel(biomech)

                _center_window(app.root, args.width, args.height)
                apply_responsive_layout(app, width=args.width, height=args.height)

                n = max(1, app.skeleton_player.frame_count - 1)
                test_frame = max(0, int(n * 0.25))

                snap = capture_motion_clearance_qa(
                    app,
                    label=label,
                    video=video_name,
                    test_frame=test_frame,
                )
                snapshots.append(snap)
                _pump(app.root, seconds=0.1)

                status = "PASS" if not snap.issues else f"{len(snap.issues)} issue(s)"
                print(f"  {status}")
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    report = format_motion_and_clearance_qa_report(snapshots)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")

    any_issues = any(s.issues for s in snapshots)
    return 1 if any_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
