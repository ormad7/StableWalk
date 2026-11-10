"""Technical Skeleton Model Inspector — counts from the kinematic wire.

Surfaces joints, segments, and DOFs as separate inventories. Does not invent
OpenSim coordinates or extra degrees of freedom.
"""

from __future__ import annotations

from typing import Any, Mapping

from stablewalk.models.joint_registry import (
    DOF_DISPLAY_NAMES,
    DOF_IDS,
    DOF_JOINT,
    JOINT_DISPLAY_NAMES,
    ROOT_JOINT_ID,
    WIRE_MODEL_KIND,
    WIRE_MODEL_NOTE,
    live_snapshot_counts,
    supervisor_12_20_verdicts,
    wire_hierarchy_lines,
    wire_model_count_summary,
    wire_model_inventory_rows,
)

SKELETON_MODEL_INSPECTOR_COUNT_KEYS: tuple[tuple[str, str], ...] = (
    ("detected_pose_landmarks", "Input pose landmarks"),
    ("reconstructed_3d_joints", "Unique 3D reconstructed joints"),
    ("unique_connected_segments", "Connected skeletal segments (unique)"),
    ("independent_sagittal_dofs", "Independent wire DOFs (sagittal)"),
    ("opensim_gait2392_coordinates", "OpenSim Gait2392 coordinates"),
    ("visualized_gait_markers", "Joints visualized (gait markers)"),
    ("visualized_limb_stick_segments", "Limb sticks visualized"),
    ("gui_selectable_items", "GUI-selectable joints"),
    ("trackable_named_joints", "Named JOINT_IDS (incl. midfoot aliases)"),
    ("video_driven_joints", "Video-driven joints"),
)


def build_skeleton_model_inspector_fields(
    *,
    snapshot: Any | None = None,
    recording: Any | None = None,
) -> dict[str, str]:
    """Label map for the inspector (counts, live frame, bilateral)."""
    counts = wire_model_count_summary()
    fields: dict[str, str] = {
        key: str(counts[key]) for key, _title in SKELETON_MODEL_INSPECTOR_COUNT_KEYS
    }
    fields["model_kind"] = WIRE_MODEL_KIND
    fields["model_note"] = WIRE_MODEL_NOTE
    n_frames = int(getattr(recording, "frame_count", 0) or 0) if recording is not None else 0
    fields["recording_frames"] = str(n_frames) if n_frames else "—"
    if snapshot is None:
        fields["frame"] = "—"
        fields["live_joints"] = "—"
        fields["live_segments"] = "—"
        fields["live_dofs"] = "—"
        fields["bilateral"] = "—"
        return fields

    frame_i = int(getattr(snapshot, "frame_index", 0) or 0)
    fields["frame"] = str(frame_i)
    live = live_snapshot_counts(snapshot)
    fields["live_joints"] = str(live["snapshot_joints"])
    fields["live_segments"] = str(live["snapshot_unique_segments"])
    fields["live_dofs"] = (
        f"{live['snapshot_dofs_with_angle']} / {len(DOF_IDS)} with angles"
    )
    left_ok = live["snapshot_left_required"] == live["snapshot_left_required_total"]
    right_ok = live["snapshot_right_required"] == live["snapshot_right_required_total"]
    fields["bilateral"] = (
        f"L {live['snapshot_left_required']}/{live['snapshot_left_required_total']}  "
        f"R {live['snapshot_right_required']}/{live['snapshot_right_required_total']}"
        + ("  · complete" if left_ok and right_ok else "  · incomplete")
    )
    return fields


def inspector_hierarchy_text() -> str:
    return "\n".join(wire_hierarchy_lines())


def inspector_component_rows() -> tuple[dict[str, str], ...]:
    return wire_model_inventory_rows()


def inspector_verdict_rows() -> tuple[dict[str, str], ...]:
    return supervisor_12_20_verdicts()


def inspector_dof_rows(snapshot: Any | None = None) -> tuple[dict[str, str], ...]:
    """One row per independent wire DOF, with live angle when present."""
    dofs = getattr(snapshot, "dofs", None) or {}
    rows: list[dict[str, str]] = []
    for dof_id in DOF_IDS:
        sample = dofs.get(dof_id) if isinstance(dofs, Mapping) else None
        angle = getattr(sample, "angle_deg", None) if sample is not None else None
        if angle is None:
            angle_s = "—"
        else:
            angle_s = f"{float(angle):.1f}°"
        rows.append(
            {
                "dof_id": dof_id,
                "name": DOF_DISPLAY_NAMES.get(dof_id, dof_id),
                "joint": JOINT_DISPLAY_NAMES.get(DOF_JOINT.get(dof_id, ""), DOF_JOINT.get(dof_id, "")),
                "angle": angle_s,
            }
        )
    return tuple(rows)


def limbs_connected(snapshot: Any) -> dict[str, bool]:
    """True when each arm/leg chain has both endpoints on the snapshot."""
    joints = getattr(snapshot, "joints", None) or {}

    def _has(*ids: str) -> bool:
        return all(jid in joints for jid in ids)

    return {
        "left_arm": _has("left_shoulder", "left_elbow", "left_wrist"),
        "right_arm": _has("right_shoulder", "right_elbow", "right_wrist"),
        "left_leg": _has("left_hip", "left_knee", "left_ankle"),
        "right_leg": _has("right_hip", "right_knee", "right_ankle"),
        "left_foot": _has("left_ankle", "left_heel", "left_toe"),
        "right_foot": _has("right_ankle", "right_heel", "right_toe"),
        "trunk": _has(ROOT_JOINT_ID, "spine", "neck", "head"),
    }


__all__ = [
    "SKELETON_MODEL_INSPECTOR_COUNT_KEYS",
    "build_skeleton_model_inspector_fields",
    "inspector_component_rows",
    "inspector_dof_rows",
    "inspector_hierarchy_text",
    "inspector_verdict_rows",
    "limbs_connected",
]
