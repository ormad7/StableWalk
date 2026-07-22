"""Stable per-joint colors shared by tables, trajectories, and selection chrome."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS, label_for_item
from stablewalk.ui.theme import ELEVATED, PANEL, SELECTION_BG, TEXT

# Sixteen distinct, desaturated laboratory hues (Visual3D / Qualisys density).
JOINT_COLOR_PALETTE: tuple[str, ...] = (
    "#3d9a5f",  # green — right hip
    "#3b82b0",  # steel — left hip
    "#c9a227",  # muted gold — right knee
    "#c45c6a",  # muted rose — left knee
    "#7c6bb0",  # slate violet — right ankle
    "#2a9a8f",  # teal — left ankle
    "#c47a3a",  # ochre — right heel
    "#4a7eb8",  # blue — left heel
    "#b06a9a",  # mauve — right toe
    "#3d9a78",  # sea green — left toe
    "#8b6db5",  # purple — right shoulder
    "#2f8fa8",  # cyan slate — left shoulder
    "#b86b88",  # dusty pink — right elbow
    "#7a9a3d",  # olive — left elbow
    "#6b78b8",  # indigo — right wrist
    "#c4844a",  # amber — left wrist
)

JOINT_COLORS: dict[str, str] = {
    item_id: JOINT_COLOR_PALETTE[index % len(JOINT_COLOR_PALETTE)]
    for index, item_id in enumerate(GUI_DOF_ITEM_IDS)
}

_LABEL_TO_ITEM: dict[str, str] = {
    label_for_item(item_id): item_id for item_id in GUI_DOF_ITEM_IDS
}

# Backward-compatible alias used by trajectory comparison plots.
TRAJECTORY_COLORS: tuple[str, ...] = JOINT_COLOR_PALETTE


def joint_color(item_id: str | None) -> str:
    """Return the canonical accent color for a GUI joint item."""
    if not item_id:
        return JOINT_COLOR_PALETTE[0]
    return JOINT_COLORS.get(item_id, JOINT_COLOR_PALETTE[0])


def item_id_for_joint_label(label: str) -> str | None:
    """Map a table joint label back to a GUI item id."""
    if not label:
        return None
    if label in JOINT_COLORS:
        return label
    return _LABEL_TO_ITEM.get(label)


def joint_row_tag(item_id: str, *, active: bool = False) -> str:
    prefix = "joint_active_" if active else "joint_"
    return f"{prefix}{item_id}"


def _parse_hex(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def blend_hex(base: str, accent: str, weight: float) -> str:
    """Blend accent into base (weight 0 = base only, 1 = accent only)."""
    w = max(0.0, min(1.0, weight))
    b = _parse_hex(base)
    a = _parse_hex(accent)
    channels = tuple(int(b[i] * (1.0 - w) + a[i] * w) for i in range(3))
    return f"#{channels[0]:02x}{channels[1]:02x}{channels[2]:02x}"


def joint_row_background(item_id: str, *, active: bool = False) -> str:
    """Subtle row tint for treeview tags."""
    accent = joint_color(item_id)
    weight = 0.34 if active else 0.22
    return blend_hex(ELEVATED, accent, weight)


def joint_row_foreground(item_id: str, *, active: bool = False) -> str:
    if active:
        return joint_color(item_id)
    return TEXT


def configure_joint_tree_tags(tree: ttk.Treeview) -> None:
    """Register per-joint row styles on a Treeview."""
    tree.tag_configure("even", background=PANEL)
    tree.tag_configure("odd", background=ELEVATED)
    tree.tag_configure("selected", background=SELECTION_BG, foreground=TEXT)
    tree.tag_configure("low_confidence", background="#3d1f28", foreground=TEXT)

    for item_id in GUI_DOF_ITEM_IDS:
        color = joint_color(item_id)
        tree.tag_configure(
            joint_row_tag(item_id),
            background=joint_row_background(item_id),
            foreground=TEXT,
        )
        tree.tag_configure(
            joint_row_tag(item_id, active=True),
            background=joint_row_background(item_id, active=True),
            foreground=color,
        )


def tags_for_joint_row(
    item_id: str | None,
    *,
    active: bool = False,
    low_confidence: bool = False,
) -> tuple[str, ...]:
    if low_confidence:
        return ("low_confidence",)
    if item_id:
        return (joint_row_tag(item_id, active=active),)
    return ("even",)


__all__ = [
    "JOINT_COLOR_PALETTE",
    "JOINT_COLORS",
    "TRAJECTORY_COLORS",
    "blend_hex",
    "configure_joint_tree_tags",
    "item_id_for_joint_label",
    "joint_color",
    "joint_row_background",
    "joint_row_tag",
    "tags_for_joint_row",
]
