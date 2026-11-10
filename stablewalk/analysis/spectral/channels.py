"""Named gait channels extracted from ``GaitMotionRecording``.

Default set covers supervisor request 2: L/R knee, hip, ankle flexion;
pelvis translation/orientation; heel and toe trajectories; selected 3D joints.
"""

from __future__ import annotations

from dataclasses import dataclass

from stablewalk.models.joint_registry import ROOT_JOINT_ID


KIND_ANGLE = "angle_deg"
KIND_POSITION = "position_m"
KIND_ORIENTATION = "orientation_deg"


@dataclass(frozen=True)
class ChannelSpec:
    """One scalar time series pulled from snapshots."""

    channel_id: str
    label: str
    kind: str
    units: str
    joint_id: str | None = None
    dof_id: str | None = None
    axis: str | None = None
    side: str | None = None  # left | right | midline
    pair_id: str | None = None
    source: str = "gait_motion"
    notes: str = ""
    proximal_joint_id: str | None = None
    distal_joint_id: str | None = None


def _dof(
    dof_id: str,
    label: str,
    *,
    side: str,
    pair_id: str,
    joint_id: str,
) -> ChannelSpec:
    return ChannelSpec(
        channel_id=f"dof:{dof_id}",
        label=label,
        kind=KIND_ANGLE,
        units="deg",
        joint_id=joint_id,
        dof_id=dof_id,
        side=side,
        pair_id=pair_id,
        source="gait_motion.dofs",
        notes=(
            "Sagittal angle on the recording. Knee channels are converted to "
            "anatomical flexion (180° − interior HKA) when the pose DOF stored "
            "an interior angle. Mock/OpenSim series that already store flexion "
            "are left unchanged."
        ),
    )


def _seg(
    seg_id: str,
    label: str,
    *,
    side: str,
    pair_id: str,
    proximal: str,
    distal: str,
) -> ChannelSpec:
    return ChannelSpec(
        channel_id=f"seg:{seg_id}",
        label=label,
        kind=KIND_ANGLE,
        units="deg",
        joint_id=proximal,
        side=side,
        pair_id=pair_id,
        source="segment_sagittal",
        proximal_joint_id=proximal,
        distal_joint_id=distal,
        notes=(
            "Signed sagittal tilt of the proximal→distal segment from the "
            "downward vertical in the canonical Y-up / Z-forward frame. "
            "0° ≈ hanging down; positive = distal more anterior. "
            "This is a segment orientation, not hip/knee/ankle flexion."
        ),
    )


def _pos(joint_id: str, axis: str, label: str, *, side: str | None, pair_id: str) -> ChannelSpec:
    return ChannelSpec(
        channel_id=f"pos:{joint_id}:{axis}",
        label=label,
        kind=KIND_POSITION,
        units="m",
        joint_id=joint_id,
        axis=axis,
        side=side,
        pair_id=pair_id,
        source="gait_motion.joints",
        notes=(
            "Canonical hip-centered metres (body-normalized). "
            "Pelvis origin is ~0 by construction unless root trajectory is used."
        ),
    )


# Sagittal DOF pair used as the default multivariate matrix.
SAGITTAL_DOF_SPECS: tuple[ChannelSpec, ...] = (
    _dof("left_hip_flexion", "Left hip flexion", side="left", pair_id="hip_flexion", joint_id="left_hip"),
    _dof("right_hip_flexion", "Right hip flexion", side="right", pair_id="hip_flexion", joint_id="right_hip"),
    _dof("left_knee_flexion", "Left knee flexion", side="left", pair_id="knee_flexion", joint_id="left_knee"),
    _dof("right_knee_flexion", "Right knee flexion", side="right", pair_id="knee_flexion", joint_id="right_knee"),
    _dof("left_ankle_flexion", "Left ankle flexion", side="left", pair_id="ankle_flexion", joint_id="left_ankle"),
    _dof("right_ankle_flexion", "Right ankle flexion", side="right", pair_id="ankle_flexion", joint_id="right_ankle"),
)

PELVIS_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        channel_id="pelvis:root:x",
        label="Pelvis translation X",
        kind=KIND_POSITION,
        units="m",
        joint_id=ROOT_JOINT_ID,
        axis="x",
        side="midline",
        pair_id="pelvis_root",
        source="pose_root_trajectory",
        notes="Pre-centering pelvis path from image landmarks (estimated).",
    ),
    ChannelSpec(
        channel_id="pelvis:root:y",
        label="Pelvis translation Y",
        kind=KIND_POSITION,
        units="m",
        joint_id=ROOT_JOINT_ID,
        axis="y",
        side="midline",
        pair_id="pelvis_root",
        source="pose_root_trajectory",
        notes="Pre-centering pelvis path from image landmarks (estimated).",
    ),
    ChannelSpec(
        channel_id="pelvis:root:z",
        label="Pelvis translation Z",
        kind=KIND_POSITION,
        units="m",
        joint_id=ROOT_JOINT_ID,
        axis="z",
        side="midline",
        pair_id="pelvis_root",
        source="pose_root_trajectory",
        notes="Heuristic depth; weaker than X/Y.",
    ),
    ChannelSpec(
        channel_id="pelvis:yaw_deg",
        label="Pelvis yaw (hip line)",
        kind=KIND_ORIENTATION,
        units="deg",
        joint_id=ROOT_JOINT_ID,
        side="midline",
        pair_id="pelvis_orient",
        source="derived_hip_line",
        notes="atan2 of left–right hip vector in the XZ plane (estimated).",
    ),
    _dof(
        "spine_flexion",
        "Trunk / spine flexion",
        side="midline",
        pair_id="spine_flexion",
        joint_id="spine",
    ),
)

_FOOT_JOINTS: tuple[tuple[str, str, str], ...] = (
    ("left_heel", "Left heel", "left"),
    ("right_heel", "Right heel", "right"),
    ("left_toe", "Left toe", "left"),
    ("right_toe", "Right toe", "right"),
)

FOOT_SPECS: tuple[ChannelSpec, ...] = tuple(
    _pos(jid, axis, f"{lab} {axis.upper()}", side=side, pair_id=f"{jid.rsplit('_', 1)[-1]}_{axis}")
    for jid, lab, side in _FOOT_JOINTS
    for axis in ("x", "y", "z")
)

_TRAJ_JOINTS: tuple[tuple[str, str, str], ...] = (
    ("left_hip", "Left hip", "left"),
    ("right_hip", "Right hip", "right"),
    ("left_knee", "Left knee", "left"),
    ("right_knee", "Right knee", "right"),
    ("left_ankle", "Left ankle", "left"),
    ("right_ankle", "Right ankle", "right"),
)

TRAJECTORY_SPECS: tuple[ChannelSpec, ...] = tuple(
    _pos(jid, axis, f"{lab} {axis.upper()}", side=side, pair_id=f"{jid.split('_', 1)[-1]}_{axis}")
    for jid, lab, side in _TRAJ_JOINTS
    for axis in ("x", "y", "z")
)

SEGMENT_SPECS: tuple[ChannelSpec, ...] = (
    _seg(
        "left_thigh",
        "Left thigh angle",
        side="left",
        pair_id="thigh_tilt",
        proximal="left_hip",
        distal="left_knee",
    ),
    _seg(
        "right_thigh",
        "Right thigh angle",
        side="right",
        pair_id="thigh_tilt",
        proximal="right_hip",
        distal="right_knee",
    ),
    _seg(
        "left_shank",
        "Left shank angle",
        side="left",
        pair_id="shank_tilt",
        proximal="left_knee",
        distal="left_ankle",
    ),
    _seg(
        "right_shank",
        "Right shank angle",
        side="right",
        pair_id="shank_tilt",
        proximal="right_knee",
        distal="right_ankle",
    ),
)

FOUR_GAIT_CHANNEL_IDS: tuple[str, ...] = tuple(s.channel_id for s in SEGMENT_SPECS)

# channel_id → (proximal joint, distal joint). Source of truth for the
# Four Motion Signals plots and for tests that lock the mapping.
FOUR_GAIT_JOINTS: dict[str, tuple[str, str]] = {
    spec.channel_id: (str(spec.proximal_joint_id), str(spec.distal_joint_id))
    for spec in SEGMENT_SPECS
}

FOUR_GAIT_ANGLE_UNITS = "deg"
FOUR_GAIT_ANGLE_YLABEL = "Sagittal tilt (deg)"
FOUR_GAIT_PANEL_TITLES: tuple[str, ...] = (
    "Left Thigh (hip → knee)",
    "Right Thigh (hip → knee)",
    "Left Shank (knee → ankle)",
    "Right Shank (knee → ankle)",
)


def four_gait_joints(channel_id: str) -> tuple[str, str]:
    """Return the (proximal, distal) joints used for a four-gait channel."""
    pair = FOUR_GAIT_JOINTS.get(str(channel_id))
    if pair is None:
        raise KeyError(f"Unknown four-gait channel: {channel_id}")
    return pair

ALL_CHANNEL_SPECS: tuple[ChannelSpec, ...] = (
    SAGITTAL_DOF_SPECS + SEGMENT_SPECS + PELVIS_SPECS + FOOT_SPECS + TRAJECTORY_SPECS
)

CHANNEL_BY_ID: dict[str, ChannelSpec] = {spec.channel_id: spec for spec in ALL_CHANNEL_SPECS}

DEFAULT_MATRIX_CHANNEL_IDS: tuple[str, ...] = tuple(s.channel_id for s in SAGITTAL_DOF_SPECS)
DEFAULT_UNIVARIATE_CHANNEL_ID = "dof:left_knee_flexion"

LEFT_RIGHT_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("hip_flexion", "dof:left_hip_flexion", "dof:right_hip_flexion"),
    ("knee_flexion", "dof:left_knee_flexion", "dof:right_knee_flexion"),
    ("ankle_flexion", "dof:left_ankle_flexion", "dof:right_ankle_flexion"),
    ("heel_y", "pos:left_heel:y", "pos:right_heel:y"),
    ("toe_y", "pos:left_toe:y", "pos:right_toe:y"),
    ("knee_y", "pos:left_knee:y", "pos:right_knee:y"),
)


def spec_for(channel_id: str) -> ChannelSpec | None:
    return CHANNEL_BY_ID.get(channel_id)
