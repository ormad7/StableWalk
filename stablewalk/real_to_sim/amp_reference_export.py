"""
Export AMP-style reference motion clips for Isaac Lab imitation learning.

Produces a static reference dataset (root trajectory + contact masks) that can
be consumed by Adversarial Motion Priors (AMP) training in a separate Isaac Lab
environment — without requiring Isaac Lab in the StableWalk env.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.real_to_sim.gait_style_extraction import GaitStyleFingerprint
from stablewalk.real_to_sim.motion_reference_loader import MotionReferenceData
from stablewalk.real_to_sim.retargeting import RetargetedMotion

logger = logging.getLogger(__name__)

AMP_REFERENCE_FILENAME = "amp_reference_motion.npz"
AMP_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class AMPReferenceExportResult:
    """Paths from an AMP reference export."""

    npz_path: Path
    json_path: Path
    frame_count: int
    fps: float
    robot_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "npz_path": str(self.npz_path),
            "json_path": str(self.json_path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "robot_name": self.robot_name,
            "schema_version": AMP_SCHEMA_VERSION,
        }


def _estimate_root_quaternions(root_positions: np.ndarray) -> np.ndarray:
    """
    Estimate root orientation quaternions (w, x, y, z) from root displacement.

    Yaw follows horizontal velocity; pitch from vertical slope. Roll = 0.
    """
    n = len(root_positions)
    quats = np.zeros((n, 4), dtype=np.float64)
    quats[:, 0] = 1.0  # identity default

    if n < 2:
        return quats

    for i in range(1, n):
        dx = root_positions[i, 0] - root_positions[i - 1, 0]
        dy = root_positions[i, 1] - root_positions[i - 1, 1]
        dz = root_positions[i, 2] - root_positions[i - 1, 2]
        horiz = np.hypot(dx, dz)
        if horiz < 1e-6 and abs(dy) < 1e-6:
            quats[i] = quats[i - 1]
            continue
        yaw = float(np.arctan2(dz, dx))
        pitch = float(np.arctan2(dy, horiz))
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        quats[i, 0] = cy * cp
        quats[i, 1] = cy * sp
        quats[i, 2] = sy * cp
        quats[i, 3] = sy * sp

    return quats


def build_amp_reference_arrays(
    retargeted: RetargetedMotion,
    *,
    gait_style: GaitStyleFingerprint | None = None,
    source_motion_path: Path | None = None,
) -> dict[str, Any]:
    """Build numpy arrays for AMP reference export."""
    root_quat = _estimate_root_quaternions(retargeted.root_positions)

    style_dict = gait_style.to_dict() if gait_style else {}
    meta = {
        "schema_version": AMP_SCHEMA_VERSION,
        "robot_name": retargeted.robot_name,
        "fps": retargeted.fps,
        "frame_count": len(retargeted.timestamps),
        "scale_factor": retargeted.scale_factor,
        "joint_map": retargeted.joint_map,
        "gait_style": style_dict,
        "source_motion_npz": str(source_motion_path) if source_motion_path else "",
        "retargeting_metadata": retargeted.metadata,
        "amp_note": (
            "Reference clip for Adversarial Motion Priors (AMP) training in Isaac Lab. "
            "Load in a separate simulation environment with Unitree G1/H1 URDF."
        ),
        "contact_mask_note": (
            "left_contact_mask / right_contact_mask synchronize video foot timing "
            "with simulated GRF during RL reward (see contact_sync_reward.py)."
        ),
    }

    return {
        "timestamps": retargeted.timestamps.astype(np.float64),
        "fps": np.float64(retargeted.fps),
        "root_positions": retargeted.root_positions.astype(np.float64),
        "root_quaternions_wxyz": root_quat,
        "joint_positions": retargeted.joint_positions.astype(np.float64),
        "left_contact_mask": retargeted.left_contact_mask.astype(np.int8),
        "right_contact_mask": retargeted.right_contact_mask.astype(np.int8),
        "amp_metadata_json": json.dumps(meta),
    }


def export_amp_reference(
    motion: MotionReferenceData,
    retargeted: RetargetedMotion,
    output_dir: Path,
    *,
    run_name: str,
    gait_style: GaitStyleFingerprint | None = None,
) -> AMPReferenceExportResult:
    """Write ``amp_reference_motion.npz`` and companion JSON manifest."""
    output_dir = Path(output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = build_amp_reference_arrays(
        retargeted,
        gait_style=gait_style,
        source_motion_path=motion.path,
    )
    npz_path = output_dir / AMP_REFERENCE_FILENAME
    np.savez_compressed(npz_path, **arrays)

    manifest = json.loads(arrays["amp_metadata_json"])
    json_path = output_dir / "amp_reference_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logger.info("Exported AMP reference → %s (%d frames)", npz_path, len(motion.timestamps))

    return AMPReferenceExportResult(
        npz_path=npz_path,
        json_path=json_path,
        frame_count=len(motion.timestamps),
        fps=motion.fps,
        robot_name=retargeted.robot_name,
    )


__all__ = [
    "AMP_REFERENCE_FILENAME",
    "AMP_SCHEMA_VERSION",
    "AMPReferenceExportResult",
    "build_amp_reference_arrays",
    "export_amp_reference",
]
