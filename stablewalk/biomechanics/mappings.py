"""
Canonical skeleton ↔ OpenSim body / marker name tables.

These mappings target gait-style OpenSim models (e.g. gait2392-like naming).
Replace or extend when a specific OpenSim model is bound in production.
"""

from __future__ import annotations

from stablewalk.biomechanics.types import OpenSimJointMapping
from stablewalk.models.joint_registry import JOINT_IDS, ROOT_JOINT_ID

# Canonical joint id → OpenSim body segment name (placeholder model vocabulary)
CANONICAL_TO_OPENSIM_BODY: dict[str, str] = {
    ROOT_JOINT_ID: "pelvis",
    "spine": "torso",
    "neck": "neck",
    "head": "head",
    "left_shoulder": "humerus_l",
    "right_shoulder": "humerus_r",
    "left_elbow": "ulna_l",
    "right_elbow": "ulna_r",
    "left_wrist": "hand_l",
    "right_wrist": "hand_r",
    "left_hip": "femur_l",
    "right_hip": "femur_r",
    "left_knee": "tibia_l",
    "right_knee": "tibia_r",
    "left_ankle": "talus_l",
    "right_ankle": "talus_r",
    "left_foot": "calcn_l",
    "right_foot": "calcn_r",
}

# Optional experimental marker labels (OpenSim / Vicon-style)
CANONICAL_TO_OPENSIM_MARKER: dict[str, str] = {
    ROOT_JOINT_ID: "LASI",
    "left_hip": "LTHI",
    "left_knee": "LKNE",
    "left_ankle": "LANK",
    "left_foot": "LHEE",
    "right_hip": "RTHI",
    "right_knee": "RKNE",
    "right_ankle": "RANK",
    "right_foot": "RHEE",
    "left_shoulder": "LSHO",
    "right_shoulder": "RSHO",
    "left_elbow": "LELB",
    "right_elbow": "RELB",
    "left_wrist": "LWRA",
    "right_wrist": "RWRA",
    "head": "HEAD",
}


def default_joint_mappings() -> list[OpenSimJointMapping]:
    """Build the default mapping list for all canonical trackable joints."""
    mappings: list[OpenSimJointMapping] = []
    for joint_id in [ROOT_JOINT_ID, *JOINT_IDS]:
        body = CANONICAL_TO_OPENSIM_BODY.get(joint_id)
        if not body:
            continue
        mappings.append(
            OpenSimJointMapping(
                canonical_joint_id=joint_id,
                opensim_body_name=body,
                opensim_marker_name=CANONICAL_TO_OPENSIM_MARKER.get(joint_id),
            )
        )
    return mappings
