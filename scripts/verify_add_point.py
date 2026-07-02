"""Verify the 'Add a point' dropdown adds/limits joints and stays in sync."""

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
from stablewalk.ui.selection_state import GUI_DOF_LABELS
from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.dashboard_layout import ADD_POINT_PLACEHOLDER


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

    values = list(app.add_point_combo.cget("values"))
    check("Dropdown lists all points when empty", len(values) == len(GUI_DOF_LABELS), str(len(values)))
    check("Dropdown resets to placeholder", app.add_point_var.get() == ADD_POINT_PLACEHOLDER, app.add_point_var.get())

    # Add Right Leg via dropdown.
    app.add_point_var.set(GUI_DOF_LABELS["right_leg"])
    app._add_point_from_combo()
    app.root.update_idletasks()
    check("Right Leg added", "right_leg" in app.selection.selected, str(app.selection.selected))
    values = list(app.add_point_combo.cget("values"))
    check("Added point leaves the dropdown", GUI_DOF_LABELS["right_leg"] not in values, str(values))
    check("Dropdown reset after add", app.add_point_var.get() == ADD_POINT_PLACEHOLDER, app.add_point_var.get())

    # Add Left Knee too.
    app.add_point_var.set(GUI_DOF_LABELS["left_knee"])
    app._add_point_from_combo()
    app.root.update_idletasks()
    check("Both selected", app.selection.selected == {"right_leg", "left_knee"}, str(app.selection.selected))

    # Removing via card returns it to the dropdown.
    app._remove_selected_point("right_leg")
    app.root.update_idletasks()
    values = list(app.add_point_combo.cget("values"))
    check("Removed point returns to dropdown", GUI_DOF_LABELS["right_leg"] in values, str(values))
    check("Selection updated after remove", app.selection.selected == {"left_knee"}, str(app.selection.selected))

    # Placeholder add is a no-op.
    before = set(app.selection.selected)
    app.add_point_var.set(ADD_POINT_PLACEHOLDER)
    app._add_point_from_combo()
    check("Placeholder add does nothing", app.selection.selected == before, str(app.selection.selected))

    root.destroy()

    if failures:
        print("\nFAILED", len(failures), "check(s):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll add-point checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
