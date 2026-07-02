"""Smoke test for the 'Points to Check' (checkpoint) feature."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import tkinter as tk

from stablewalk.ui.presentation_demo import generate_presentation_recording
from stablewalk.ui.tk.app import StableWalkGUI


def _rows(app: StableWalkGUI) -> list[tuple]:
    return [
        app.checkpoint_tree.item(iid, "values")
        for iid in app.checkpoint_tree.get_children()
    ]


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)
    app._set_gait_motion(generate_presentation_recording())
    app.root.update_idletasks()

    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if ok:
            print(f"PASS  {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"FAIL  {name}: {detail}")

    # No selection -> add should not create rows, should warn.
    app._clear_dof_selection()
    app._add_checkpoint()
    check("Add with no selection adds nothing", not _rows(app), str(_rows(app)))

    # Select Left Knee at frame 0, add a point.
    app._dof_checkbox_vars["left_knee"].set(True)
    app._on_dof_checkbox_changed("left_knee")
    app.root.update_idletasks()
    app._add_checkpoint()
    rows = _rows(app)
    check("Left Knee point added", len(rows) == 1, str(rows))
    check(
        "Left Knee row has DOF label",
        rows and rows[0][3] == "Left Knee",
        str(rows),
    )

    # Move to a later frame, select Right Knee too, add again.
    app.skeleton_player.go_to(20)
    app._dof_checkbox_vars["right_knee"].set(True)
    app._on_dof_checkbox_changed("right_knee")
    app.root.update_idletasks()
    app._add_checkpoint()
    rows = _rows(app)
    dofs = {r[3] for r in rows}
    check("Both legs captured across points", {"Left Knee", "Right Knee"} <= dofs, str(rows))
    check("Sequential indices", [r[0] for r in rows] == [str(i + 1) for i in range(len(rows))], str(rows))

    # Clear removes everything.
    app._clear_checkpoints()
    check("Clear empties the table", not _rows(app), str(_rows(app)))

    root.destroy()

    if failures:
        print("\nFAILED", len(failures), "check(s):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll checkpoint checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
