"""Verify the redesigned Selected Points cards and per-point add/remove."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tkinter as tk

import matplotlib

matplotlib.use("Agg")

from stablewalk.ui.presentation_demo import generate_presentation_recording
from stablewalk.ui.tk.app import StableWalkGUI


def _card_titles(app: StableWalkGUI) -> list[str]:
    """Collect the bold joint titles rendered inside the Selected Points cards."""
    titles: list[str] = []
    for card in app.dof_details_frame.winfo_children():
        for inner in card.winfo_children():
            for child in inner.winfo_children():
                # header frame holds the dot + label + remove button
                if isinstance(child, tk.Frame):
                    for w in child.winfo_children():
                        if isinstance(w, tk.Label):
                            font = str(w.cget("font"))
                            if "Semibold" in font:
                                titles.append(w.cget("text"))
    return titles


def _select(app: StableWalkGUI, *item_ids: str) -> None:
    app._clear_dof_selection()
    for item_id in item_ids:
        app._dof_checkbox_vars[item_id].set(True)
        app._on_dof_checkbox_changed(item_id)
    app.root.update_idletasks()


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

    # Empty state when nothing selected
    _select(app)
    check("No cards when empty", not _card_titles(app), str(_card_titles(app)))
    check(
        "Empty hint shown",
        app.lbl_details_empty.winfo_manager() == "pack",
        f"manager={app.lbl_details_empty.winfo_manager()!r}",
    )

    # One card for Right Knee
    _select(app, "right_knee")
    titles = _card_titles(app)
    check("One card for Right Knee", titles == ["Right Knee"], str(titles))

    # Add Left Arm + Right Shoulder -> three cards, ordered by panel order
    _select(app, "right_knee", "left_arm", "right_shoulder")
    titles = _card_titles(app)
    check("Three cards present", len(titles) == 3, str(titles))
    check(
        "Cards contain all selected labels",
        set(titles) == {"Right Knee", "Left Arm", "Right Shoulder"},
        str(titles),
    )

    # Per-point removal: remove only Right Knee
    app._remove_selected_point("right_knee")
    app.root.update_idletasks()
    titles = _card_titles(app)
    check("Right Knee removed", "Right Knee" not in titles, str(titles))
    check("Others remain", set(titles) == {"Left Arm", "Right Shoulder"}, str(titles))
    check(
        "Selection state matches cards",
        app.selection.selected == {"left_arm", "right_shoulder"},
        str(app.selection.selected),
    )
    check(
        "Checkbox unticked after remove",
        not app._dof_checkbox_vars["right_knee"].get(),
        "checkbox still ticked",
    )

    # Cap: select many, expect capped cards + overflow note
    _select(app, *list(app._dof_checkbox_vars.keys()))
    titles = _card_titles(app)
    check(
        "Card count capped at max",
        len(titles) == app._MAX_DETAIL_CARDS,
        f"{len(titles)} cards",
    )

    root.destroy()

    if failures:
        print("\nFAILED", len(failures), "check(s):")
        for f in failures:
            print(" -", f)
        return 1
    print("\nAll Selected Points checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
