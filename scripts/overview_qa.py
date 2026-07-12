#!/usr/bin/env python3
"""
StableWalk Overview tab — final usability and scientific-clarity QA.

Loads Abnormal / Normal / Performance demo videos, verifies the Overview layout,
checks phase–contact consistency, and writes ``data/output/reports/overview_qa.md``.
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


def _pump(root, *, seconds: float = 0.1) -> None:
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


def _ensure_summary_populated(app) -> None:
    """Re-apply stability summary if load completed but cards are empty."""
    result = getattr(app, "_biomech", None)
    if result is None:
        return
    update = getattr(app, "_update_stability_panel", None)
    if update is not None:
        update(result)


def _run_demo_overview_qa(
    app,
    *,
    key: str,
    label: str,
    video_name: str,
    width: int,
    height: int,
    pipeline_timeout: float,
) -> tuple["OverviewQASnapshot", list["GuiRegressionCheck"]]:
    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.ui.tk.gui_visual_qa import (
        GuiRegressionCheck,
        OverviewQASnapshot,
        capture_overview_snapshot,
        run_overview_tab_assertions,
    )

    ex = example_by_key(key)
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / video_name
    poses_path = config.POSES_DIR / video_name.replace(".mp4", "_poses.json")

    with _suppress_messageboxes():
        if poses_path.is_file():
            app._active_demo_gait = ex
            app._presentation_mode = False
            app._highlight_demo_button(key)
            source = str(video_path)
            app.url_var.set(source)
            app.url_entry.delete(0, "end")
            app.url_entry.insert(0, source)
            app._current_source = source
            if not app.load_poses(poses_path, fresh=True, expected_source=source):
                snap = OverviewQASnapshot(video=video_name, label=label)
                snap.issues.append(f"Failed to load poses: {poses_path.name}")
                return snap, []
        else:
            app._load_demo_gait(key)
            ok = _wait_until(
                app.root,
                lambda: not app._processing and app.sequence is not None,
                timeout=pipeline_timeout,
            )
            if not ok:
                snap = OverviewQASnapshot(video=video_name, label=label)
                snap.issues.append(f"Pipeline timeout ({pipeline_timeout:.0f}s)")
                return snap, []

        if not app.sequence or not app.skeleton_player:
            snap = OverviewQASnapshot(video=video_name, label=label)
            snap.issues.append("No sequence/skeleton player after load")
            return snap, []

        _ensure_summary_populated(app)

        _center_window(app.root, width, height)
        apply_responsive_layout(app, width=width, height=height)

        select_dashboard_tab(app, TAB_OVERVIEW)
        n = max(1, app.skeleton_player.frame_count - 1)
        target = max(0, int(n * 0.25))
        app._go_to(target)
        _pump(app.root, seconds=0.25)

        if app._gait_cycle is not None:
            app._update_gait_cycle_panel(app._gait_cycle, frame_index=target)
        app._refresh_bilateral_ground_clearance()
        _pump(app.root, seconds=0.15)

        checks = run_overview_tab_assertions(app)
        snap = capture_overview_snapshot(app, label=label, video=video_name)
        for check in checks:
            if not check.passed:
                snap.issues.append(f"{check.name}: {check.detail}")

        if snap.foot_clearance_panel_count > 1:
            snap.issues.append(
                f"Foot clearance shown {snap.foot_clearance_panel_count} times on Overview"
            )

        return snap, checks


def main() -> int:
    parser = argparse.ArgumentParser(description="StableWalk Overview tab QA")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: data/output/reports/overview_qa.md)",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--pipeline-timeout", type=float, default=600.0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Skip GUI launch; write layout spec only (no live widget checks)",
    )
    args = parser.parse_args()

    from stablewalk import config
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.gui_visual_qa import format_overview_qa_report

    config.ensure_output_dirs()

    out_path = args.output or (config.REPORTS_DIR / "overview_qa.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.headless:
        text = format_overview_qa_report([], checks_by_video={})
        text = text.replace(
            "**Total issues across demos:** 0",
            "**Total issues across demos:** N/A (headless — no live GUI run)",
        )
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path} (headless stub)")
        return 0

    import tkinter as tk

    from stablewalk.ui.tk.app import StableWalkGUI

    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()

    snapshots = []
    checks_by_video: dict[str, list] = {}

    try:
        for key, label, video_name in DEMO_KEYS:
            print(f"Overview QA: {label} ({video_name})...")
            snap, checks = _run_demo_overview_qa(
                app,
                key=key,
                label=label,
                video_name=video_name,
                width=args.width,
                height=args.height,
                pipeline_timeout=args.pipeline_timeout,
            )
            snapshots.append(snap)
            checks_by_video[video_name] = checks
            status = "PASS" if not snap.issues else f"{len(snap.issues)} issue(s)"
            print(f"  {status}")
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    report = format_overview_qa_report(snapshots, checks_by_video=checks_by_video)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")

    any_issues = any(s.issues for s in snapshots) or any(
        not c.passed for checks in checks_by_video.values() for c in checks
    )
    return 1 if any_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
