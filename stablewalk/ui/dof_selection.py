"""
GUI-selectable degrees of freedom for interactive skeleton inspection.

Maps user-facing labels (Right Hip, Left Knee, …) to canonical joint ids used
by ``GaitMotionRecording`` and the skeleton renderer. Supports multi-point
selection for simultaneous table, trajectory, and checkpoint analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stablewalk.models.gait_motion import SkeletonSnapshot
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES

# User-facing checklist (order shown in side panel — lower body, then upper)
GUI_DOF_ITEM_IDS: tuple[str, ...] = (
    "right_hip",
    "left_hip",
    "right_knee",
    "left_knee",
    "right_ankle",
    "left_ankle",
    "right_heel",
    "left_heel",
    "right_toe",
    "left_toe",
    "right_shoulder",
    "left_shoulder",
    "right_elbow",
    "left_elbow",
    "right_wrist",
    "left_wrist",
)

GUI_DOF_LABELS: dict[str, str] = {item_id: item_id.replace("_", " ").title() for item_id in GUI_DOF_ITEM_IDS}
# Override with proper casing for display
for _key, _label in (
    ("right_hip", "Right Hip"),
    ("left_hip", "Left Hip"),
    ("right_knee", "Right Knee"),
    ("left_knee", "Left Knee"),
    ("right_ankle", "Right Ankle"),
    ("left_ankle", "Left Ankle"),
    ("right_heel", "Right Heel"),
    ("left_heel", "Left Heel"),
    ("right_toe", "Right Toe"),
    ("left_toe", "Left Toe"),
    ("right_shoulder", "Right Shoulder"),
    ("left_shoulder", "Left Shoulder"),
    ("right_elbow", "Right Elbow"),
    ("left_elbow", "Left Elbow"),
    ("right_wrist", "Right Wrist"),
    ("left_wrist", "Left Wrist"),
):
    GUI_DOF_LABELS[_key] = _label

# Joints to highlight for each selectable item (anchor + connecting segment)
_ITEM_JOINTS: dict[str, frozenset[str]] = {
    "right_hip": frozenset({"right_hip", "right_knee"}),
    "left_hip": frozenset({"left_hip", "left_knee"}),
    "right_knee": frozenset({"right_hip", "right_knee", "right_ankle"}),
    "left_knee": frozenset({"left_hip", "left_knee", "left_ankle"}),
    "right_ankle": frozenset({"right_knee", "right_ankle", "right_heel", "right_toe"}),
    "left_ankle": frozenset({"left_knee", "left_ankle", "left_heel", "left_toe"}),
    "right_heel": frozenset({"right_ankle", "right_heel", "right_toe"}),
    "left_heel": frozenset({"left_ankle", "left_heel", "left_toe"}),
    "right_toe": frozenset({"right_heel", "right_toe", "right_ankle"}),
    "left_toe": frozenset({"left_heel", "left_toe", "left_ankle"}),
    "right_shoulder": frozenset({"right_shoulder", "right_elbow"}),
    "left_shoulder": frozenset({"left_shoulder", "left_elbow"}),
    "right_elbow": frozenset({"right_shoulder", "right_elbow", "right_wrist"}),
    "left_elbow": frozenset({"left_shoulder", "left_elbow", "left_wrist"}),
    "right_wrist": frozenset({"right_elbow", "right_wrist"}),
    "left_wrist": frozenset({"left_elbow", "left_wrist"}),
}

# Primary joint for position readout / trajectory anchor
_ANCHOR_JOINT: dict[str, str] = {item_id: item_id for item_id in GUI_DOF_ITEM_IDS}

# Click on skeleton joint → GUI item (most specific match)
_JOINT_TO_ITEM: dict[str, str] = {item_id: item_id for item_id in GUI_DOF_ITEM_IDS}
_JOINT_TO_ITEM.update(
    {
        "right_foot": "right_toe",
        "left_foot": "left_toe",
        "left_foot_index": "left_toe",
        "right_foot_index": "right_toe",
    }
)

PICKABLE_JOINTS: tuple[str, ...] = tuple(_JOINT_TO_ITEM.keys())


def joints_for_item(item_id: str) -> set[str]:
    return set(_ITEM_JOINTS.get(item_id, frozenset()))


def anchor_joint_for_item(item_id: str) -> str | None:
    return _ANCHOR_JOINT.get(item_id)


def item_for_joint(joint_id: str) -> str | None:
    return _JOINT_TO_ITEM.get(joint_id)


def label_for_item(item_id: str) -> str:
    return GUI_DOF_LABELS.get(item_id, item_id.replace("_", " ").title())


@dataclass
class DofSelectionState:
    """Multi-select GUI DOF state."""

    selected: set[str] = field(default_factory=set)
    last_selected: str | None = None

    def clear(self) -> None:
        self.selected.clear()
        self.last_selected = None

    @property
    def active_item_id(self) -> str | None:
        """Point driving detailed analysis (3D graph, metrics, position table focus)."""
        return self.last_selected

    def set_active(self, item_id: str) -> bool:
        """Switch the active analysis point without changing the selection set."""
        if item_id in self.selected:
            self.last_selected = item_id
            return True
        return False

    def activate_item(self, item_id: str) -> bool:
        """Add ``item_id`` to the selection and make it the active analysis point."""
        if item_id not in GUI_DOF_LABELS:
            return False
        self.selected.add(item_id)
        self.last_selected = item_id
        return True

    def active_label(self) -> str:
        """Display name for the active analysis point, or em dash when none."""
        if not self.active_item_id:
            return "—"
        return label_for_item(self.active_item_id)

    def set_selection(
        self,
        item_ids: set[str],
        *,
        last_selected: str | None = None,
    ) -> None:
        self.selected = {i for i in item_ids if i in GUI_DOF_LABELS}
        if last_selected and last_selected in self.selected:
            self.last_selected = last_selected
        elif self.selected:
            self.last_selected = next(
                (i for i in GUI_DOF_ITEM_IDS if i in self.selected),
                next(iter(self.selected)),
            )
        else:
            self.last_selected = None

    def select_only(self, item_id: str) -> None:
        if item_id in GUI_DOF_LABELS:
            self.selected = {item_id}
            self.last_selected = item_id

    def focus_item(self, item_id: str) -> None:
        """Make ``item_id`` the active point for analysis without changing the set."""
        self.set_active(item_id)

    def ensure_last_selected(self) -> None:
        """Keep ``last_selected`` valid so analysis never falls back to the first checklist item."""
        if not self.selected:
            self.last_selected = None
            return
        if self.last_selected in self.selected:
            return
        if len(self.selected) == 1:
            self.last_selected = next(iter(self.selected))
            return
        self.last_selected = next(
            (i for i in GUI_DOF_ITEM_IDS if i in self.selected),
            next(iter(self.selected)),
        )

    def toggle(self, item_id: str) -> None:
        if item_id not in GUI_DOF_LABELS:
            return
        if item_id in self.selected:
            self.selected.discard(item_id)
            self.last_selected = next(
                (i for i in GUI_DOF_ITEM_IDS if i in self.selected),
                None,
            )
        else:
            self.selected.add(item_id)
            self.last_selected = item_id

    def highlight_joints(self) -> set[str]:
        joints: set[str] = set()
        for item_id in self.selected:
            joints.update(joints_for_item(item_id))
        return joints

    def detail_rows(
        self, snapshot: SkeletonSnapshot | None
    ) -> list[tuple[str, str, str, str, str]]:
        """
        Rows for the details panel: (label, joint_name, x, y, z).

        One row per selected item; missing positions show em dashes.
        """
        if not self.selected or snapshot is None:
            return []

        rows: list[tuple[str, str, str, str, str]] = []
        ordered = [i for i in GUI_DOF_ITEM_IDS if i in self.selected]
        for item_id in ordered:
            anchor = anchor_joint_for_item(item_id)
            label = label_for_item(item_id)
            if not anchor:
                rows.append((label, "—", "—", "—", "—"))
                continue
            sample = snapshot.joints.get(anchor)
            jname = JOINT_DISPLAY_NAMES.get(anchor, anchor.replace("_", " ").title())
            if sample:
                rows.append(
                    (
                        label,
                        jname,
                        f"{sample.position.x:.3f}",
                        f"{sample.position.y:.3f}",
                        f"{sample.position.z:.3f}",
                    )
                )
            else:
                rows.append((label, jname, "—", "—", "—"))
        return rows

    def summary(self, *, max_names: int = 4) -> str:
        """Sidebar summary: count and names when the selection is small."""
        return self.selection_summary(max_names=max_names)

    def selection_summary(self, *, max_names: int = 4) -> str:
        n = len(self.selected)
        if n == 0:
            return "No points selected"
        ordered = [label_for_item(i) for i in GUI_DOF_ITEM_IDS if i in self.selected]
        if n == 1:
            return f"1 point selected: {ordered[0]}"
        if n <= max_names:
            return f"{n} points selected: {', '.join(ordered)}"
        shown = ", ".join(ordered[:max_names])
        return f"{n} points selected: {shown}, +{n - max_names} more"

    def count_label(self) -> str:
        """Short heading for the details panel; names the active point when multi-select."""
        n = len(self.selected)
        if n == 0:
            return "No point selected"
        self.ensure_last_selected()
        active = self.active_label()
        if n == 1:
            return f"Active: {active}"
        return f"{n} selected · active: {active}"

    def selected_labels(self) -> str:
        if not self.selected:
            return "No joint selected"
        ordered = [label_for_item(i) for i in GUI_DOF_ITEM_IDS if i in self.selected]
        return ", ".join(ordered)

    def overview_lines(self, *, max_names: int = 6) -> tuple[str, str]:
        """Session overview: (selection line, names line)."""
        n = len(self.selected)
        if n == 0:
            return ("Selected points: 0", "Names: —")
        ordered = [label_for_item(i) for i in GUI_DOF_ITEM_IDS if i in self.selected]
        count_line = f"Selected points: {n}"
        if n <= max_names:
            names_line = f"Names: {', '.join(ordered)}"
        else:
            names_line = f"Names: {', '.join(ordered[:max_names])}, +{n - max_names} more"
        return (count_line, names_line)
