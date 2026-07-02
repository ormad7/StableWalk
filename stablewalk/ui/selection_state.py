"""Interactive DOF selection — re-exports ``DofSelectionState`` for backward compatibility."""

from stablewalk.ui.dof_selection import (
    DofSelectionState,
    GUI_DOF_ITEM_IDS,
    GUI_DOF_LABELS,
    PICKABLE_JOINTS,
    item_for_joint,
    joints_for_item,
    label_for_item,
)

# Legacy alias used by older GUI code paths
SelectionState = DofSelectionState

__all__ = [
    "DofSelectionState",
    "SelectionState",
    "GUI_DOF_ITEM_IDS",
    "GUI_DOF_LABELS",
    "PICKABLE_JOINTS",
    "item_for_joint",
    "joints_for_item",
    "label_for_item",
]
