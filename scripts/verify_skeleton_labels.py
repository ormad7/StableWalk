"""Verify skeleton labels track the active selected analysis point."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import tkinter as tk

from stablewalk.ui.presentation_demo import generate_presentation_recording
from stablewalk.ui.tk.app import StableWalkGUI


def _pick(joint_id: str) -> SimpleNamespace:
    return SimpleNamespace(artist=SimpleNamespace(_sw_joint_ids=[joint_id]), ind=[0])


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    app._set_gait_motion(generate_presentation_recording())
    app._clear_dof_selection()
    app.root.update_idletasks()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"PASS  {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"FAIL  {name}: {detail}")

    check("No labels by default", app._skeleton_joint_labels() == {}, str(app._skeleton_joint_labels()))

    app._dof_checkbox_vars["left_hip"].set(True)
    app._on_dof_checkbox_changed("left_hip")
    lbl = app._skeleton_joint_labels()
    check(
        "Checkbox selection labels active point",
        lbl.get("left_hip") == "Left Hip",
        str(lbl),
    )
    check(
        "Active id matches checkbox selection",
        app._active_dof_item_id() == "left_hip",
        str(app._active_dof_item_id()),
    )

    app._on_skeleton_pick(_pick("right_knee"))
    lbl = app._skeleton_joint_labels()
    check("Skeleton click sets active label", lbl.get("right_knee") == "Right Knee", str(lbl))
    check(
        "Skeleton click keeps prior selection",
        "left_hip" in app.selection.selected and "right_knee" in app.selection.selected,
        str(app.selection.selected),
    )
    check(
        "Skeleton click switches active id",
        app._active_dof_item_id() == "right_knee",
        str(app._active_dof_item_id()),
    )

    app._remove_selected_point("left_hip")
    check(
        "Remove non-active clears stale selection",
        "left_hip" not in app.selection.selected,
        str(app.selection.selected),
    )
    check(
        "Active remains after removing other point",
        app._active_dof_item_id() == "right_knee",
        str(app._active_dof_item_id()),
    )

    app._clear_dof_selection()
    check("Clear selection clears label", app._skeleton_joint_labels() == {}, str(app._skeleton_joint_labels()))
    check("Clear selection clears active joint", app._active_joint is None, str(app._active_joint))

    app._on_skeleton_pick(_pick("right_knee"))
    try:
        app._update_interactive_skeleton(force_draw=True)
        drew = True
        err = ""
    except Exception as exc:  # noqa: BLE001
        drew = False
        err = repr(exc)
    check("Redraw with active label works", drew, err)
    check(
        "Active label is readable",
        app._skeleton_joint_labels().get("right_knee") == "Right Knee",
        str(app._skeleton_joint_labels()),
    )

    root.destroy()

    if failures:
        print("\nFAILED", len(failures), "check(s):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll skeleton-label checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
