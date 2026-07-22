"""Verify DOF checkbox selection flows against the five manual test cases."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import tkinter as tk
from matplotlib.figure import Figure

from stablewalk.ui.dof_position_table import table_rows_for_selection
from stablewalk.ui.dof_selection import item_for_joint, joints_for_item, label_for_item
from stablewalk.ui.dof_step_preview import compute_dof_step_previews
from stablewalk.ui.presentation_demo import generate_presentation_recording
from stablewalk.ui.skeleton_player import SkeletonPlayer
from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.viewers.dof_trajectory_3d import draw_dof_trajectories


def _tree_dof_labels(app: StableWalkGUI) -> set[str]:
    labels: set[str] = set()
    for iid in app.dof_pos_tree.get_children():
        values = app.dof_pos_tree.item(iid, "values")
        if values:
            labels.add(str(values[2]))
    return labels


def _trajectory_labels(app: StableWalkGUI) -> set[str]:
    snap = app.skeleton_player.current_snapshot()
    fig = Figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_dof_trajectories(
        ax,
        app.gait_motion,
        app.selection.selected,
        end_frame_float=app.skeleton_player.state.frame_float,
        tip_snapshot=snap,
        clear=True,
    )
    labels: set[str] = set()
    for line in ax.get_lines():
        label = line.get_label()
        if label and not label.startswith("_"):
            labels.add(label.split(" (", 1)[0])
    return labels


def _checked_ids(app: StableWalkGUI) -> set[str]:
    return {item_id for item_id, var in app._dof_checkbox_vars.items() if var.get()}


def _set_only(app: StableWalkGUI, *item_ids: str) -> None:
    app.selection.clear()
    for item_id in item_ids:
        app._dof_checkbox_vars[item_id].set(True)
        app._on_dof_checkbox_changed(item_id)
    app.root.update_idletasks()


def _toggle_checkbox(app: StableWalkGUI, item_id: str, checked: bool) -> None:
    app._dof_checkbox_vars[item_id].set(checked)
    app._on_dof_checkbox_changed(item_id)
    app.root.update_idletasks()


def _mock_pick(joint_id: str) -> SimpleNamespace:
    artist = SimpleNamespace(_sw_joint_ids=[joint_id])
    return SimpleNamespace(artist=artist, ind=[0])


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    app = StableWalkGUI(root=root)

    recording = generate_presentation_recording()
    app._set_gait_motion(recording)
    app.root.update_idletasks()

    failures: list[str] = []

    def check(name: str, fn) -> None:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failures.append(f"{name}: {exc}")
            print(f"FAIL  {name}: {exc}")

    # Case 1: only Right Knee
    def case1() -> None:
        _set_only(app, "right_knee")
        selected = app.selection.selected
        _assert(selected == {"right_knee"}, f"selection={selected}")
        _assert(_checked_ids(app) == {"right_knee"}, f"checkboxes={_checked_ids(app)}")

        highlight = app._resolve_highlight_joints() or set()
        _assert("right_knee" in highlight, f"highlight missing right_knee: {highlight}")

        snap = app.skeleton_player.current_snapshot()
        rows = table_rows_for_selection(snap, selected)
        _assert(len(rows) == 1 and rows[0][2] == "Right Knee", f"table rows={rows}")

        tree = _tree_dof_labels(app)
        _assert(tree == {"Right Knee"}, f"tree labels={tree}")

        traj = _trajectory_labels(app)
        _assert("Right Knee" in traj, f"trajectory labels={traj}")

        previews = compute_dof_step_previews(
            app.gait_motion,
            selected,
            app.skeleton_player.state.frame_float,
            config=app._step_config(),
        )
        _assert(len(previews) == 1 and previews[0].item_id == "right_knee", str(previews))

        app._refresh_realtime_analysis(force_draw=True)
        _assert(
            len(app._dof_step_previews) == 1,
            f"step previews={len(app._dof_step_previews)}",
        )

    # Case 2: add Left Arm
    def case2() -> None:
        _toggle_checkbox(app, "left_arm", True)
        selected = app.selection.selected
        _assert(
            selected == {"right_knee", "left_arm"},
            f"selection={selected}",
        )
        highlight = app._resolve_highlight_joints() or set()
        _assert(
            joints_for_item("left_arm").issubset(highlight),
            f"left_arm joints missing from highlight: {highlight}",
        )
        tree = _tree_dof_labels(app)
        _assert(tree == {"Right Knee", "Left Arm"}, f"tree={tree}")
        traj = _trajectory_labels(app)
        _assert(
            {"Right Knee", "Left Arm"}.issubset(traj),
            f"trajectory={traj}",
        )

    # Case 3: uncheck Right Knee
    def case3() -> None:
        _toggle_checkbox(app, "right_knee", False)
        selected = app.selection.selected
        _assert(selected == {"left_arm"}, f"selection={selected}")
        _assert("right_knee" not in _checked_ids(app), "right_knee still checked")
        highlight = app._resolve_highlight_joints() or set()
        _assert("right_knee" not in highlight, f"right_knee still highlighted: {highlight}")
        tree = _tree_dof_labels(app)
        _assert(tree == {"Left Arm"}, f"tree={tree}")
        traj = _trajectory_labels(app)
        _assert("Right Knee" not in traj, f"trajectory still has Right Knee: {traj}")
        _assert("Left Arm" in traj, f"trajectory missing Left Arm: {traj}")

    # Case 4: clear all
    def case4() -> None:
        app._clear_dof_selection()
        _assert(not app.selection.selected, "selection not empty")
        _assert(not _checked_ids(app), f"checkboxes still checked: {_checked_ids(app)}")
        _assert(app._resolve_highlight_joints() in (None, set()), "highlights remain")
        _assert(not _tree_dof_labels(app), f"table not empty: {_tree_dof_labels(app)}")
        traj = _trajectory_labels(app)
        _assert(not traj, f"trajectory not empty: {traj}")
        _assert(not app._dof_step_previews, "step previews remain")

    # Case 5: skeleton joint selects joint and shows trajectory on Overview
    def case5() -> None:
        item_id = item_for_joint("right_knee")
        _assert(item_id == "right_knee", f"item_for_joint={item_id}")

        app._on_skeleton_pick(_mock_pick("right_knee"))
        _assert("right_knee" in app.selection.selected, "pick did not select")
        _assert(app._dof_checkbox_vars["right_knee"].get(), "checkbox not checked")
        from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW
        tab = app._dashboard_notebook.tab(app._dashboard_notebook.select(), "text").strip()
        _assert(tab == TAB_OVERVIEW, f"expected Overview tab, got {tab!r}")
        _assert(getattr(app, "_overview_traj_dock_visible", False), "trajectory dock not shown")
        canvas = getattr(app, "canvas_dof_traj_overview", None)
        _assert(canvas is not None, "overview trajectory canvas missing")
        w = canvas.get_tk_widget()
        _assert(w.winfo_ismapped(), "overview trajectory canvas not mapped")

        app._on_skeleton_pick(_mock_pick("right_knee"))
        _assert("right_knee" in app.selection.selected, "re-pick should keep selected")
        _assert(app._dof_checkbox_vars["right_knee"].get(), "checkbox still checked")

        # Plain click replaces selection (single-selection joint inspection).
        app._on_skeleton_pick(_mock_pick("left_shoulder"))
        _assert(
            app.selection.selected == {"left_shoulder"},
            f"shoulder pick should replace selection, got {app.selection.selected}",
        )
        _assert(app._dof_checkbox_vars["left_shoulder"].get(), "left_shoulder checkbox off")
        _assert(not app._dof_checkbox_vars["right_knee"].get(), "right_knee should clear")

        # Ctrl+Click adds a second joint for comparison.
        ctrl_pick = _mock_pick("right_knee")
        ctrl_pick.mouseevent = SimpleNamespace(key="control", guiEvent=None)
        app._on_skeleton_pick(ctrl_pick)
        _assert(
            {"right_knee", "left_shoulder"} <= app.selection.selected,
            f"Ctrl+Click should add right_knee, got {app.selection.selected}",
        )

    check("1. Select only Right Knee", case1)
    check("2. Add Left Arm", case2)
    check("3. Uncheck Right Knee", case3)
    check("4. Clear selection", case4)
    check("5. Skeleton joint toggle", case5)

    root.destroy()

    if failures:
        print("\nFAILED", len(failures), "case(s):")
        for msg in failures:
            print(" -", msg)
        return 1

    print("\nAll DOF selection test cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
