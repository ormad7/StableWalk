"""Hover tooltip text for skeleton joint inspection."""

from __future__ import annotations

from stablewalk.models.gait_motion import SkeletonSnapshot
from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES
from stablewalk.ui.dof_position_table import NA_VALUE, _format_angle, _format_detail_velocity, dof_detail_metrics
from stablewalk.ui.dof_selection import anchor_joint_for_item, item_for_joint, label_for_item


def joint_display_name(joint_id: str) -> str:
    """User-facing joint label for tooltips."""
    item_id = item_for_joint(joint_id)
    if item_id:
        return label_for_item(item_id)
    return JOINT_DISPLAY_NAMES.get(joint_id, joint_id.replace("_", " ").title())


def format_skeleton_joint_tooltip(joint_id: str, snapshot: SkeletonSnapshot) -> str:
    """Multi-line tooltip: name, coordinates, angle, velocity."""
    item_id = item_for_joint(joint_id)
    anchor = anchor_joint_for_item(item_id) if item_id else joint_id
    joint = snapshot.joints.get(anchor or joint_id) or snapshot.joints.get(joint_id)

    name = joint_display_name(joint_id)
    if joint is not None:
        pos = joint.position
        coords = f"x={pos.x:.3f}  y={pos.y:.3f}  z={pos.z:.3f}"
    else:
        coords = NA_VALUE

    if item_id:
        metrics = dof_detail_metrics(item_id, snapshot)
        angle = metrics.angle
        velocity = metrics.velocity
    else:
        angle = _format_angle(joint, None)
        velocity = _format_detail_velocity(joint, None)

    return f"{name}\n{coords}\nAngle: {angle}\nVelocity: {velocity}"


__all__ = ["format_skeleton_joint_tooltip", "joint_display_name"]
