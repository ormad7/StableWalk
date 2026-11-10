"""Scientific 3D Joint Path statistics panel (presentation only)."""

from __future__ import annotations

import tkinter as tk

import pytest

from stablewalk.ui.tk.dashboard_path_science import (
    _PATH_SCIENCE_FIELDS,
    build_path_science_stats_panel,
)


@pytest.fixture()
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk not available")
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


def test_path_science_fields_cover_required_kpis() -> None:
    captions = {cap for _attr, cap in _PATH_SCIENCE_FIELDS}
    required = {
        "Current X",
        "Current Y",
        "Current Z",
        "Path length (cum.)",
        "Instantaneous Velocity (est.)",
        "Average Velocity (est.)",
        "Displacement",
        "Current Frame",
        "Current Time",
        "Current Joint Angle",
        "Motion Range",
        "Coordinate Units",
    }
    assert required <= captions


def test_build_path_science_stats_panel(root) -> None:
    class _Gui:
        pass

    gui = _Gui()
    host = tk.Frame(root)
    panel = build_path_science_stats_panel(gui, host)
    assert panel is not None
    assert getattr(gui, "lbl_traj_info_x", None) is not None
    assert getattr(gui, "lbl_traj_path_length", None) is not None
    assert getattr(gui, "lbl_traj_inst_velocity", None) is not None
    assert getattr(gui, "lbl_traj_joint_angle", None) is not None
    assert getattr(gui, "lbl_traj_coord_units", None) is not None
