"""Scientific statistics strip under the Motion 3D Joint Path (presentation only)."""

from __future__ import annotations

import tkinter as tk
from typing import Any

from stablewalk.ui.scientific_labels import UNITS_COORD_LEGEND
from stablewalk.ui.theme import (
    ELEVATED,
    FONT_KPI_SUPPORTING,
    FONT_KPI_TITLE,
    FONT_MONO_SM,
    FONT_UI_XS,
    MUTED,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    PANEL,
    TEXT,
    TEXT_SECONDARY,
)
from stablewalk.ui.tk.kpi_cards import create_metric_group_label

# (attr, caption) — values filled by app._update_dof_trajectory_* helpers.
_PATH_SCIENCE_FIELDS: tuple[tuple[str, str], ...] = (
    ("lbl_traj_info_x", "Current X"),
    ("lbl_traj_info_y", "Current Y"),
    ("lbl_traj_info_z", "Current Z"),
    ("lbl_traj_path_length", "Path length (cum.)"),
    ("lbl_traj_inst_velocity", "Instantaneous Velocity (est.)"),
    ("lbl_traj_avg_velocity", "Average Velocity (est.)"),
    ("lbl_traj_displacement", "Displacement"),
    ("lbl_traj_info_frame", "Current Frame"),
    ("lbl_traj_info_time", "Current Time"),
    ("lbl_traj_joint_angle", "Current Joint Angle"),
    ("lbl_traj_motion_range", "Motion Range"),
    ("lbl_traj_coord_units", "Coordinate Units"),
)

# Compatibility aliases kept for existing interpretation refresh paths.
_COMPAT_ALIASES: tuple[tuple[str, str], ...] = (
    ("lbl_traj_travel", "lbl_traj_path_length"),
    ("lbl_traj_total_distance", "lbl_traj_displacement"),
    ("lbl_traj_info_joint", "lbl_traj_info_joint"),
)


def build_path_science_stats_panel(gui: Any, parent: tk.Misc) -> tk.Frame:
    """Always-visible scientific KPI grid under the 3D trajectory canvas."""
    outer = tk.Frame(parent, bg=ELEVATED, highlightthickness=0)
    outer.columnconfigure(0, weight=1)

    header = tk.Frame(outer, bg=ELEVATED, highlightthickness=0)
    header.grid(row=0, column=0, sticky="ew", padx=PAD_SM, pady=(PAD_SM, 0))
    header.columnconfigure(0, weight=1)
    create_metric_group_label(header, "Path statistics", bg=ELEVATED)

    gui.lbl_traj_info_joint = tk.Label(
        header,
        text="—",
        bg=ELEVATED,
        fg=TEXT_SECONDARY,
        font=FONT_UI_XS,
        anchor="e",
    )
    gui.lbl_traj_info_joint.pack(side=tk.RIGHT)

    grid = tk.Frame(outer, bg=ELEVATED, highlightthickness=0)
    grid.grid(row=1, column=0, sticky="ew", padx=PAD_SM, pady=(0, PAD_SM))
    cols = 4
    for c in range(cols):
        grid.columnconfigure(c, weight=1, uniform="path_sci")

    gui._path_science_value_labels: dict[str, tk.Label] = {}
    for index, (attr, caption) in enumerate(_PATH_SCIENCE_FIELDS):
        r, c = divmod(index, cols)
        cell = tk.Frame(grid, bg=ELEVATED, highlightthickness=0)
        cell.grid(
            row=r,
            column=c,
            sticky="nsew",
            padx=(0 if c == 0 else PAD_SM, 0),
            pady=(PAD_XS, PAD_XS),
        )
        tk.Label(
            cell,
            text=caption,
            bg=ELEVATED,
            fg=MUTED,
            font=FONT_KPI_TITLE,
            anchor="w",
        ).pack(anchor="w")
        value = tk.Label(
            cell,
            text="—",
            bg=ELEVATED,
            fg=TEXT,
            font=FONT_MONO_SM if "Units" not in caption else FONT_KPI_SUPPORTING,
            anchor="w",
        )
        value.pack(anchor="w", pady=(2, 0))
        setattr(gui, attr, value)
        gui._path_science_value_labels[attr] = value

    # Defaults that do not depend on a tip sample.
    units = getattr(gui, "lbl_traj_coord_units", None)
    if units is not None:
        units.configure(text="cm est. · stature-scaled · not calibrated")

    # Hidden compatibility labels expected by older interpretation paths.
    hidden = tk.Frame(outer, bg=ELEVATED, highlightthickness=0)
    gui.lbl_traj_travel = tk.Label(hidden, text="Travel: —")
    gui.lbl_traj_total_distance = tk.Label(hidden, text="—")
    gui.lbl_traj_smoothness = tk.Label(hidden, text="Smoothness: —")
    gui.lbl_traj_max_deviation = tk.Label(hidden, text="Maximum deviation: —")
    gui.lbl_traj_samples = tk.Label(hidden, text="Samples: —")
    gui.lbl_traj_confidence = tk.Label(hidden, text="Trajectory Confidence: —")
    gui.lbl_motion_units_legend = tk.Label(
        hidden, text=UNITS_COORD_LEGEND, bg=ELEVATED, fg=MUTED
    )
    gui.dof_traj_live_info_frame = outer
    gui.dof_analysis_path_metrics_frame = outer
    gui._motion_path_details = None
    gui._path_science_panel = outer
    return outer


def sync_path_science_compat_labels(gui: Any) -> None:
    """Mirror new KPIs onto legacy Travel / Traveled labels when present."""
    path_len = getattr(gui, "lbl_traj_path_length", None)
    travel = getattr(gui, "lbl_traj_travel", None)
    if path_len is not None and travel is not None:
        try:
            text = str(path_len.cget("text") or "—")
            if not text.startswith("Travel"):
                travel.configure(text=f"Travel: {text}")
            else:
                travel.configure(text=text)
        except tk.TclError:
            pass
    disp = getattr(gui, "lbl_traj_displacement", None)
    total = getattr(gui, "lbl_traj_total_distance", None)
    if disp is not None and total is not None:
        try:
            total.configure(text=str(disp.cget("text") or "—"))
        except tk.TclError:
            pass


__all__ = [
    "build_path_science_stats_panel",
    "sync_path_science_compat_labels",
    "_PATH_SCIENCE_FIELDS",
]
