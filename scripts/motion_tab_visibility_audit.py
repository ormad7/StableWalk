#!/usr/bin/env python3
"""
GUI visibility audit for the Motion Analysis tab.

Measures widget geometry at 1920x1080, applies temporary debug borders,
and writes ``data/output/reports/motion_tab_visibility_audit.md``.
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


def _pump(root, *, seconds: float = 0.2) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        root.update()
        time.sleep(0.01)


def main() -> int:
    parser = argparse.ArgumentParser(description="Motion Analysis tab visibility audit")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: data/output/reports/motion_tab_visibility_audit.md)",
    )
    args = parser.parse_args()

    from stablewalk import config
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_responsive import apply_responsive_layout
    from stablewalk.ui.tk.gui_visual_qa import (
        _apply_motion_debug_borders,
        _restore_motion_debug_borders,
        capture_motion_tab_geometry,
        format_motion_tab_visibility_report,
    )

    config.ensure_output_dirs()
    out_path = args.output or (config.REPORTS_DIR / "motion_tab_visibility_audit.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    root.deiconify()
    root.geometry(f"{args.width}x{args.height}")

    sections: list[str] = []

    try:
        with _suppress_messageboxes():
            apply_responsive_layout(app, width=args.width, height=args.height)
            _pump(app.root, seconds=0.2)

            from stablewalk.ui.tk.dashboard_notebook import TAB_MOTION, select_dashboard_tab

            # Before session load — graph frame should still be visible (idle placeholder)
            select_dashboard_tab(app, TAB_MOTION)
            app._sync_dof_analysis_panel_state()
            _pump(app.root, seconds=0.15)
            snap_idle = capture_motion_tab_geometry(app)
            print("=== Before session load ===")
            print("Motion tab size:", f"{snap_idle.motion_tab_w} x {snap_idle.motion_tab_h}")
            print("Knee panel size:", f"{snap_idle.knee_panel_w} x {snap_idle.knee_panel_h}")
            print(
                "Joint path frame size:",
                f"{snap_idle.traj_panel_w} x {snap_idle.traj_panel_h}",
            )
            print(
                "Graph frame size:",
                f"{snap_idle.graph_frame_w} x {snap_idle.graph_frame_h}",
            )
            print(
                "Matplotlib canvas size:",
                f"{snap_idle.canvas_w} x {snap_idle.canvas_h}",
            )
            print(
                "Traj width fraction:",
                f"{snap_idle.traj_width_fraction:.1%}",
            )
            sections.append("## Startup (no session)\n")
            sections.append(
                format_motion_tab_visibility_report(
                    snap_idle,
                    resolution=f"{args.width}x{args.height}",
                    joint_selected=False,
                )
            )
            sections.append("---\n")

            poses = config.POSES_DIR / "normal_gait_poses.json"
            if not poses.is_file():
                print(f"Missing poses: {poses}")
                return 1
            app.load_poses(poses, fresh=True)
            apply_responsive_layout(app, width=args.width, height=args.height)
            _pump(app.root, seconds=0.3)

            # Session loaded, no joint selected — placeholder in graph area
            select_dashboard_tab(app, TAB_MOTION)
            app._sync_dof_analysis_panel_state()
            app._refresh_selected_dof_trajectory_3d(force_draw=True)
            _pump(app.root, seconds=0.2)

            restored = _apply_motion_debug_borders(app)
            _pump(app.root, seconds=0.15)
            snap_no_joint = capture_motion_tab_geometry(app)
            _restore_motion_debug_borders(app, restored)

            print("\n=== With session, no joint selected ===")
            print("Knee panel size:", f"{snap_no_joint.knee_panel_w} x {snap_no_joint.knee_panel_h}")
            print(
                "Joint path frame size:",
                f"{snap_no_joint.traj_panel_w} x {snap_no_joint.traj_panel_h}",
            )
            print(
                "Graph frame size:",
                f"{snap_no_joint.graph_frame_w} x {snap_no_joint.graph_frame_h}",
            )
            print(
                "Matplotlib canvas size:",
                f"{snap_no_joint.canvas_w} x {snap_no_joint.canvas_h}",
            )
            print(
                "Traj width fraction:",
                f"{snap_no_joint.traj_width_fraction:.1%}",
            )

            sections.append("## Session loaded (no joint)\n")
            sections.append(
                format_motion_tab_visibility_report(
                    snap_no_joint,
                    resolution=f"{args.width}x{args.height}",
                    joint_selected=False,
                )
            )

            # With joint selected
            app.selection.select_only("right_knee")
            app._notify_dof_selection_changed()
            app._activate_dof_item("right_knee", add_if_missing=True)
            app._refresh_selected_dof_trajectory_3d(force_draw=True)
            app._render_dof_traj_canvas(force=True)
            _pump(app.root, seconds=0.2)

            restored = _apply_motion_debug_borders(app)
            _pump(app.root, seconds=0.1)
            snap_joint = capture_motion_tab_geometry(app)
            _restore_motion_debug_borders(app, restored)

            print("\n=== With Right Knee selected ===")
            print("Matplotlib canvas size:", f"{snap_joint.canvas_w} x {snap_joint.canvas_h}")
            print("Traj width fraction:", f"{snap_joint.traj_width_fraction:.1%}")

            sections.append("---\n")
            sections.append("## Session loaded (Right Knee selected)\n")
            sections.append(
                format_motion_tab_visibility_report(
                    snap_joint,
                    resolution=f"{args.width}x{args.height}",
                    joint_selected=True,
                )
            )
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    report = "\n".join(sections)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path}")

    any_issues = bool(snap_idle.issues or snap_no_joint.issues or snap_joint.issues)
    return 1 if any_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
