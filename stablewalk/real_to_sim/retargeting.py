"""
Offline human → humanoid retargeting (morphology alignment).

Maps canonical StableWalk joint trajectories onto a target humanoid morphology
using scale factors from body segment dimensions. Full GMR / Isaac Lab retargeting
requires a separate simulation environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.real_to_sim.motion_reference_loader import MotionReferenceData

DEFAULT_RETARGET_CONFIG = (
    Path(__file__).resolve().parents[2] / "models" / "real_to_sim" / "unitree_g1_retarget.json"
)


@dataclass(frozen=True)
class RetargetConfig:
    """Humanoid retargeting configuration template."""

    robot_name: str
    urdf_path: str
    scale_reference_leg_length_m: float
    human_to_robot_joint_map: dict[str, str]
    foot_link_ids: dict[str, int]
    notes: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetargetConfig:
        return cls(
            robot_name=str(data.get("robot_name", "unitree_g1")),
            urdf_path=str(data.get("urdf_path", "")),
            scale_reference_leg_length_m=float(
                data.get("scale_reference_leg_length_m", 0.85)
            ),
            human_to_robot_joint_map=dict(data.get("human_to_robot_joint_map", {})),
            foot_link_ids={
                str(k): int(v) for k, v in data.get("foot_link_ids", {}).items()
            },
            notes=list(data.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_name": self.robot_name,
            "urdf_path": self.urdf_path,
            "scale_reference_leg_length_m": self.scale_reference_leg_length_m,
            "human_to_robot_joint_map": self.human_to_robot_joint_map,
            "foot_link_ids": self.foot_link_ids,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RetargetedMotion:
    """Scaled motion reference for a target humanoid."""

    robot_name: str
    scale_factor: float
    root_positions: np.ndarray
    joint_positions: np.ndarray
    left_contact_mask: np.ndarray
    right_contact_mask: np.ndarray
    timestamps: np.ndarray
    fps: float
    joint_map: dict[str, str]
    metadata: dict[str, Any]


def load_retarget_config(path: Path | None = None) -> RetargetConfig:
    """Load retargeting JSON config (defaults to Unitree G1 template)."""
    config_path = Path(path) if path else DEFAULT_RETARGET_CONFIG
    if not config_path.is_file():
        return RetargetConfig(
            robot_name="unitree_g1",
            urdf_path="(download Unitree G1 URDF for Isaac Lab)",
            scale_reference_leg_length_m=0.85,
            human_to_robot_joint_map={
                "pelvis": "pelvis",
                "left_hip": "left_hip_yaw",
                "right_hip": "right_hip_yaw",
                "left_knee": "left_knee",
                "right_knee": "right_knee",
                "left_ankle": "left_ankle",
                "right_ankle": "right_ankle",
            },
            foot_link_ids={"left": 4, "right": 8},
            notes=["Built-in fallback config — place URDF in Isaac Lab env."],
        )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return RetargetConfig.from_dict(data)


def retarget_motion_reference(
    motion: MotionReferenceData,
    config: RetargetConfig | None = None,
) -> RetargetedMotion:
    """
    Scale human motion to robot proportions (offline morphology alignment).

    Uses median leg length from motion metadata vs robot reference leg length.
    """
    cfg = config or load_retarget_config()
    meta = motion.metadata
    dims = meta.get("dimensions", {})
    human_leg = float(dims.get("leg_length_average_m", 0.9) or 0.9)
    ref_leg = max(cfg.scale_reference_leg_length_m, 0.5)
    scale = ref_leg / max(human_leg, 0.4)

    root = motion.root_positions * scale
    joints = motion.canonical_joint_positions * scale

    return RetargetedMotion(
        robot_name=cfg.robot_name,
        scale_factor=scale,
        root_positions=root,
        joint_positions=joints,
        left_contact_mask=motion.left_contact_mask.copy(),
        right_contact_mask=motion.right_contact_mask.copy(),
        timestamps=motion.timestamps.copy(),
        fps=motion.fps,
        joint_map=dict(cfg.human_to_robot_joint_map),
        metadata={
            "human_leg_length_m": human_leg,
            "robot_reference_leg_m": ref_leg,
            "scale_factor": scale,
            "urdf_path": cfg.urdf_path,
            "foot_link_ids": cfg.foot_link_ids,
            "retargeting_method": "uniform_scale_pelvis_relative",
        },
    )


def export_retargeted_motion_npz(
    retargeted: RetargetedMotion,
    output_path: Path,
    *,
    source_motion_path: Path | None = None,
) -> Path:
    """Write scaled humanoid motion reference (stage 2 output) to NPZ."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(retargeted.metadata)
    if source_motion_path is not None:
        meta["source_motion_npz"] = str(source_motion_path)
    np.savez_compressed(
        output_path,
        timestamps=retargeted.timestamps.astype(np.float64),
        fps=np.float64(retargeted.fps),
        root_positions=retargeted.root_positions.astype(np.float64),
        joint_positions=retargeted.joint_positions.astype(np.float64),
        left_contact_mask=retargeted.left_contact_mask.astype(np.int8),
        right_contact_mask=retargeted.right_contact_mask.astype(np.int8),
        robot_name=np.array(retargeted.robot_name),
        scale_factor=np.float64(retargeted.scale_factor),
        retarget_metadata_json=json.dumps(meta),
        joint_map_json=json.dumps(retargeted.joint_map),
    )
    return output_path


__all__ = [
    "DEFAULT_RETARGET_CONFIG",
    "RetargetConfig",
    "RetargetedMotion",
    "export_retargeted_motion_npz",
    "load_retarget_config",
    "retarget_motion_reference",
]
