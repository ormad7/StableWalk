"""Gait analysis degrees of freedom (14+ DOF) with human-readable labels."""

from __future__ import annotations

# 14+ DOF for gait and posture (exported angles)
GAIT_ANGLE_FIELDS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle_flexion",
    "right_ankle_flexion",
    "neck",
    "head_neck",
    "torso_tilt",
    "pelvis_rotation",
    "spine",
)

# Display labels for GUI and reports
DOF_LABELS: dict[str, str] = {
    "left_shoulder": "L shoulder",
    "right_shoulder": "R shoulder",
    "left_elbow": "L elbow",
    "right_elbow": "R elbow",
    "left_hip": "L hip",
    "right_hip": "R hip",
    "left_knee": "L knee",
    "right_knee": "R knee",
    "left_ankle_flexion": "L ankle flexion",
    "right_ankle_flexion": "R ankle flexion",
    "neck": "Neck",
    "head_neck": "Head / neck",
    "torso_tilt": "Torso (spine tilt)",
    "pelvis_rotation": "Pelvis rotation",
    "spine": "Spine",
    # Legacy / internal names
    "left_ankle": "L ankle flexion",
    "right_ankle": "R ankle flexion",
}

GAIT_DOF_LANDMARKS: tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "nose",
    "mid_hip",
)

MIN_GAIT_DOF = 14

GAIT_VELOCITY_JOINTS: tuple[str, ...] = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Limb groups for interactive inspection (GUI / OpenSim-style filtering)
LIMB_GROUPS: dict[str, tuple[str, ...]] = {
    "all": GAIT_ANGLE_FIELDS,
    "right_leg": ("right_hip", "right_knee", "right_ankle_flexion"),
    "left_leg": ("left_hip", "left_knee", "left_ankle_flexion"),
    "right_arm": ("right_shoulder", "right_elbow"),
    "left_arm": ("left_shoulder", "left_elbow"),
    "torso": ("neck", "head_neck", "torso_tilt", "pelvis_rotation", "spine"),
}

LIMB_GROUP_LABELS: dict[str, str] = {
    "all": "All DOFs",
    "right_leg": "Right leg",
    "left_leg": "Left leg",
    "right_arm": "Right arm",
    "left_arm": "Left arm",
    "torso": "Torso / spine",
}

# Angle name → primary joint landmark for visualization
DOF_TO_JOINT: dict[str, str] = {
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle_flexion": "left_ankle",
    "right_ankle_flexion": "right_ankle",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "neck": "mid_shoulder",
    "head_neck": "nose",
    "torso_tilt": "mid_shoulder",
    "pelvis_rotation": "mid_hip",
    "spine": "mid_hip",
}

# Bones associated with each DOF (for highlight rendering)
DOF_TO_BONES: dict[str, tuple[tuple[str, str], ...]] = {
    "left_shoulder": (("mid_shoulder", "left_shoulder"), ("left_shoulder", "left_elbow")),
    "right_shoulder": (("mid_shoulder", "right_shoulder"), ("right_shoulder", "right_elbow")),
    "left_elbow": (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist")),
    "right_elbow": (("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")),
    "left_hip": (("mid_hip", "left_hip"), ("left_hip", "left_knee")),
    "right_hip": (("mid_hip", "right_hip"), ("right_hip", "right_knee")),
    "left_knee": (("left_hip", "left_knee"), ("left_knee", "left_ankle")),
    "right_knee": (("right_hip", "right_knee"), ("right_knee", "right_ankle")),
    "left_ankle_flexion": (("left_knee", "left_ankle"),),
    "right_ankle_flexion": (("right_knee", "right_ankle"),),
    "left_ankle": (("left_knee", "left_ankle"),),
    "right_ankle": (("right_knee", "right_ankle"),),
    "neck": (("mid_shoulder", "nose"),),
    "head_neck": (("mid_shoulder", "nose"),),
    "torso_tilt": (("mid_hip", "mid_shoulder"),),
    "pelvis_rotation": (("mid_hip", "left_hip"), ("mid_hip", "right_hip")),
    "spine": (("mid_hip", "mid_shoulder"),),
}

# Future OpenSim coordinate aliases (stub mapping)
OPENSIM_ALIASES: dict[str, str] = {
    "hip_flexion_r": "right_hip",
    "hip_flexion_l": "left_hip",
    "knee_angle_r": "right_knee",
    "knee_angle_l": "left_knee",
    "ankle_angle_r": "right_ankle_flexion",
    "ankle_angle_l": "left_ankle_flexion",
}


def limb_group_keys() -> tuple[str, ...]:
    return tuple(LIMB_GROUPS.keys())


def dofs_for_limb(limb_group: str) -> tuple[str, ...]:
    return LIMB_GROUPS.get(limb_group, GAIT_ANGLE_FIELDS)


def joint_for_dof(dof_name: str) -> str | None:
    return DOF_TO_JOINT.get(dof_name)


def bones_for_dof(dof_name: str) -> tuple[tuple[str, str], ...]:
    return DOF_TO_BONES.get(dof_name, ())


def joints_for_dof(dof_name: str) -> set[str]:
    joints: set[str] = set()
    joint = joint_for_dof(dof_name)
    if joint:
        joints.add(joint)
    for a, b in bones_for_dof(dof_name):
        joints.add(a)
        joints.add(b)
    return joints


def joints_for_limb(limb_group: str) -> set[str]:
    out: set[str] = set()
    for dof in dofs_for_limb(limb_group):
        out.update(joints_for_dof(dof))
    return out


def highlight_joints_for_selection(
    *,
    limb_group: str = "all",
    selected_dof: str | None = None,
    highlight_all_gait: bool = False,
) -> set[str] | None:
    """
    Resolve which joints to accent in overlays.

    Returns None when global gait-DOF highlight mode is active (legacy all landmarks).
    Returns empty set when highlighting is off.
    """
    if highlight_all_gait and limb_group == "all" and not selected_dof:
        return None
    if selected_dof:
        return joints_for_dof(selected_dof)
    if limb_group != "all":
        return joints_for_limb(limb_group)
    if highlight_all_gait:
        return set(GAIT_DOF_LANDMARKS)
    return set()
