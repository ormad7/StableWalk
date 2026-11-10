"""Recapture Overview / Motion / Biomech / Results / Compare / Advanced after layout fixes."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import importlib.util

_shot_mod = importlib.util.spec_from_file_location(
    "gui_layout_screenshots", _ROOT / "scripts" / "gui_layout_screenshots.py"
)
assert _shot_mod and _shot_mod.loader
_shots = importlib.util.module_from_spec(_shot_mod)
_shot_mod.loader.exec_module(_shots)
OUT = _shots.OUT
_capture_compare = _shots._capture_compare
_find_notebook = _shots._find_notebook
_select_inner = _shots._select_inner
shot = _shots.shot

import tkinter as tk

from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    select_dashboard_tab,
)
from stablewalk.ui.tk.layout_scale import apply_process_dpi_awareness


def _pump(root: tk.Misc, seconds: float = 0.25) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.04)


def _capture_size(app: StableWalkGUI, root: tk.Misc, prefix: str, spec: str) -> None:
    root.geometry(spec)
    root.update_idletasks()
    root.update()
    _pump(root, 0.2)

    select_dashboard_tab(app, TAB_OVERVIEW)
    _pump(root, 0.2)
    shot(root, f"{prefix}_overview.png")

    select_dashboard_tab(app, TAB_MOTION)
    _pump(root, 0.4)
    shot(root, f"{prefix}_motion.png")
    inner = _find_notebook(
        getattr(app, "_tab_motion"),
        ("Spectral Analysis", "Gait Correlation & Spectral Analysis", "Kinematics"),
    )
    if inner is not None:
        for title, name in (
            ("Spectral Analysis", "spectral"),
            ("Spectral Features", "spectral_features"),
            ("Gait Correlation & Spectral Analysis", "gait_correlation"),
        ):
            _select_inner(inner, title)
            _pump(root, 0.2)
            shot(root, f"{prefix}_{name}.png")

    select_dashboard_tab(app, TAB_BIOMECHANICS)
    bnb = _find_notebook(getattr(app, "_tab_biomechanics"), ("COM", "Balance"))
    if bnb is not None:
        _select_inner(bnb, "COM")
        _pump(root, 0.25)
    shot(root, f"{prefix}_biomech_com.png")

    select_dashboard_tab(app, TAB_RESULTS_SUMMARY)
    _pump(root, 0.2)
    shot(root, f"{prefix}_results.png")

    _capture_compare(app, root, prefix)

    select_dashboard_tab(app, TAB_ADVANCED)
    _pump(root, 0.2)
    shot(root, f"{prefix}_advanced.png")


def main() -> int:
    apply_process_dpi_awareness()
    root = tk.Tk()
    root.geometry("1366x768+20+20")
    app = StableWalkGUI(root=root)
    root.update()
    try:
        root.attributes("-topmost", True)
        root.lift()
    except tk.TclError:
        pass
    poses = _ROOT / "data" / "output" / "poses" / "normal_gait_poses.json"
    if poses.is_file():
        try:
            app.load_poses(poses, fresh=True, wait_for_analysis=True)
        except TypeError:
            app.load_poses(poses, fresh=True)
        _pump(root, 0.5)
        if app.skeleton_player is not None and getattr(app.skeleton_player, "frame_count", 0) > 4:
            app._go_to(max(1, int(app.skeleton_player.frame_count * 0.25)))
            root.update()

    for prefix, spec in (
        ("fix_1366", "1366x768+20+20"),
        ("fix_1600", "1600x900+20+20"),
        ("fix_1920", "1920x1080+20+20"),
    ):
        _capture_size(app, root, prefix, spec)

    root.destroy()
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
