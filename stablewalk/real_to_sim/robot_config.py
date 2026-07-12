"""
Load target humanoid skeleton definitions from JSON configuration.

Joint names, parents, limits, and rest pose come from config files — not hard-coded indices.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "real_to_sim"

DEFAULT_G1_SKELETON = MODELS_DIR / "unitree_g1_skeleton.json"
DEFAULT_H1_SKELETON = MODELS_DIR / "unitree_h1_skeleton.json"
DEFAULT_G1_RETARGET = MODELS_DIR / "unitree_g1_retarget.json"


@dataclass(frozen=True)
class RobotJointDefinition:
    """One joint / link in the target humanoid."""

    name: str
    parent: str | None
    joint_type: str
    human_joint: str | None = None
    human_child: str | None = None
    segment: str | None = None
    dof_axis: str | None = None
    limits_rad: tuple[float, float] | None = None
    rest_angle_rad: float = 0.0
    rest_position_m: tuple[float, float, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotJointDefinition:
        limits = data.get("limits_rad")
        lim_tuple = None
        if isinstance(limits, (list, tuple)) and len(limits) == 2:
            lim_tuple = (float(limits[0]), float(limits[1]))
        rest_pos = data.get("rest_position_m")
        rest_pos_t = tuple(rest_pos) if isinstance(rest_pos, (list, tuple)) and len(rest_pos) == 3 else None
        return cls(
            name=str(data["name"]),
            parent=str(data["parent"]) if data.get("parent") else None,
            joint_type=str(data.get("type", "revolute")),
            human_joint=str(data["human_joint"]) if data.get("human_joint") else None,
            human_child=str(data["human_child"]) if data.get("human_child") else None,
            segment=str(data["segment"]) if data.get("segment") else None,
            dof_axis=str(data["dof_axis"]) if data.get("dof_axis") else None,
            limits_rad=lim_tuple,
            rest_angle_rad=float(data.get("rest_angle_rad", 0.0)),
            rest_position_m=rest_pos_t,
        )


@dataclass(frozen=True)
class RobotSkeletonConfig:
    """Full humanoid retargeting configuration."""

    robot_name: str
    display_name: str
    urdf_path: str
    scale_reference_leg_length_m: float
    segment_lengths_m: dict[str, float]
    coordinate_transform: dict[str, Any]
    root_joint: str
    joints: tuple[RobotJointDefinition, ...]
    foot_contact_joints: dict[str, str]
    isaac_lab: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "2.0"
    notes: list[str] = field(default_factory=list)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def revolute_joints(self) -> tuple[RobotJointDefinition, ...]:
        return tuple(j for j in self.joints if j.joint_type == "revolute")

    @property
    def joint_by_name(self) -> dict[str, RobotJointDefinition]:
        return {j.name: j for j in self.joints}

    def human_to_robot_map(self) -> dict[str, str]:
        """Human canonical joint id → robot joint name (revolute DOFs)."""
        out: dict[str, str] = {}
        for j in self.revolute_joints:
            if j.human_joint:
                out[j.human_joint] = j.name
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "robot_name": self.robot_name,
            "display_name": self.display_name,
            "urdf_path": self.urdf_path,
            "scale_reference_leg_length_m": self.scale_reference_leg_length_m,
            "segment_lengths_m": self.segment_lengths_m,
            "coordinate_transform": self.coordinate_transform,
            "root_joint": self.root_joint,
            "joint_count": len(self.joints),
            "foot_contact_joints": self.foot_contact_joints,
            "isaac_lab": self.isaac_lab,
        }


def _resolve_skeleton_path(path: Path | None) -> Path:
    if path is not None:
        return Path(path)
    if DEFAULT_G1_SKELETON.is_file():
        return DEFAULT_G1_SKELETON
    return DEFAULT_G1_RETARGET


def load_robot_skeleton_config(path: Path | None = None) -> RobotSkeletonConfig:
    """Load robot skeleton JSON (G1 default)."""
    config_path = _resolve_skeleton_path(path)
    if not config_path.is_file():
        return _fallback_g1_config()

    data = json.loads(config_path.read_text(encoding="utf-8"))
    joints = tuple(RobotJointDefinition.from_dict(j) for j in data.get("joints", []))
    return RobotSkeletonConfig(
        robot_name=str(data.get("robot_name", "unitree_g1")),
        display_name=str(data.get("display_name", data.get("robot_name", "unitree_g1"))),
        urdf_path=str(data.get("urdf_path", "")),
        scale_reference_leg_length_m=float(data.get("scale_reference_leg_length_m", 0.85)),
        segment_lengths_m={k: float(v) for k, v in data.get("segment_lengths_m", {}).items()},
        coordinate_transform=dict(data.get("coordinate_transform", {})),
        root_joint=str(data.get("root_joint", "pelvis")),
        joints=joints,
        foot_contact_joints={str(k): str(v) for k, v in data.get("foot_contact_joints", {}).items()},
        isaac_lab=dict(data.get("isaac_lab", {})),
        schema_version=str(data.get("schema_version", "2.0")),
        notes=list(data.get("notes", [])),
    )


def load_robot_skeleton_by_name(robot_name: str) -> RobotSkeletonConfig:
    """Resolve skeleton file by robot name."""
    key = robot_name.strip().lower()
    if key in ("unitree_h1", "h1"):
        path = DEFAULT_H1_SKELETON if DEFAULT_H1_SKELETON.is_file() else DEFAULT_G1_SKELETON
    else:
        path = DEFAULT_G1_SKELETON
    return load_robot_skeleton_config(path)


def _fallback_g1_config() -> RobotSkeletonConfig:
    """Minimal inline fallback when JSON files are missing."""
    joints = (
        RobotJointDefinition("pelvis", None, "free", human_joint="pelvis"),
        RobotJointDefinition(
            "left_hip_pitch_joint", "pelvis", "revolute",
            human_joint="left_hip", human_child="left_knee", segment="thigh",
            limits_rad=(-2.5, 2.5),
        ),
        RobotJointDefinition(
            "left_knee_joint", "left_hip_pitch_joint", "revolute",
            human_joint="left_knee", human_child="left_ankle", segment="shank",
            limits_rad=(-0.1, 2.8),
        ),
        RobotJointDefinition(
            "right_hip_pitch_joint", "pelvis", "revolute",
            human_joint="right_hip", human_child="right_knee", segment="thigh",
            limits_rad=(-2.5, 2.5),
        ),
        RobotJointDefinition(
            "right_knee_joint", "right_hip_pitch_joint", "revolute",
            human_joint="right_knee", human_child="right_ankle", segment="shank",
            limits_rad=(-0.1, 2.8),
        ),
    )
    return RobotSkeletonConfig(
        robot_name="unitree_g1",
        display_name="Unitree G1",
        urdf_path="(download URDF for Isaac Lab)",
        scale_reference_leg_length_m=0.85,
        segment_lengths_m={"thigh": 0.35, "shank": 0.35},
        coordinate_transform={"axis_flip": [1, 1, 1]},
        root_joint="pelvis",
        joints=joints,
        foot_contact_joints={"left": "left_ankle_pitch_joint", "right": "right_ankle_pitch_joint"},
        notes=["Built-in fallback skeleton"],
    )


__all__ = [
    "DEFAULT_G1_SKELETON",
    "DEFAULT_H1_SKELETON",
    "RobotJointDefinition",
    "RobotSkeletonConfig",
    "load_robot_skeleton_config",
    "load_robot_skeleton_by_name",
]
