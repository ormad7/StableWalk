"""Stable per-joint colors shared by tables, trajectories, and selection chrome."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, GUI_DOF_LABELS, label_for_item
from stablewalk.ui.theme import ELEVATED, PANEL, SELECTION_BG, TEXT

# Per-joint hues within a strict left=green / right=red family so L/R is
# instantly readable while multi-joint overlays remain distinguishable.
# Canonical SIDE_LEFT / SIDE_RIGHT remain "#22c55e" / "#ef4444".
_LEFT_JOINT_PALETTE: tuple[str, ...] = (
    "#16a34a",  # hip
    "#22c55e",  # knee (SIDE_LEFT)
    "#4ade80",  # ankle
    "#14b8a6",  # heel
    "#2dd4bf",  # toe
    "#059669",  # shoulder
    "#34d399",  # elbow
    "#6ee7b7",  # wrist
)
_RIGHT_JOINT_PALETTE: tuple[str, ...] = (
    "#dc2626",  # hip
    "#ef4444",  # knee (SIDE_RIGHT)
    "#f87171",  # ankle
    "#ea580c",  # heel
    "#f97316",  # toe
    "#e11d48",  # shoulder
    "#fb7185",  # elbow
    "#fdba74",  # wrist
)

_LEFT_JOINT_ORDER: tuple[str, ...] = tuple(
    i for i in GUI_DOF_ITEM_IDS if i.startswith("left_")
)
_RIGHT_JOINT_ORDER: tuple[str, ...] = tuple(
    i for i in GUI_DOF_ITEM_IDS if i.startswith("right_")
)


def _palette_for_item(item_id: str) -> str:
    if item_id.startswith("left_"):
        idx = _LEFT_JOINT_ORDER.index(item_id) if item_id in _LEFT_JOINT_ORDER else 0
        return _LEFT_JOINT_PALETTE[idx % len(_LEFT_JOINT_PALETTE)]
    if item_id.startswith("right_"):
        idx = _RIGHT_JOINT_ORDER.index(item_id) if item_id in _RIGHT_JOINT_ORDER else 0
        return _RIGHT_JOINT_PALETTE[idx % len(_RIGHT_JOINT_PALETTE)]
    return _LEFT_JOINT_PALETTE[0]


JOINT_COLORS: dict[str, str] = {
    item_id: _palette_for_item(item_id) for item_id in GUI_DOF_ITEM_IDS
}

# Flat palette kept for trajectory comparison plots (stable order = GUI order).
JOINT_COLOR_PALETTE: tuple[str, ...] = tuple(JOINT_COLORS[i] for i in GUI_DOF_ITEM_IDS)

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
