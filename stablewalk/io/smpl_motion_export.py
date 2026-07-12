"""
Export SMPL / unified human motion to ``smpl_motion.npz``.

Written only when the SMPL backend actually produced mesh-based motion —
never for synthetic or fallback MediaPipe data labeled as SMPL.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.models.joint_registry import JOINT_IDS, ROOT_JOINT_ID
from stablewalk.pose.backends.unified_motion import (
    UNIFIED_MOTION_SCHEMA_VERSION,
    UnifiedHumanMotion,
    unified_joint_positions_array,
    unified_metadata_json,
)

logger = logging.getLogger(__name__)

SMPL_MOTION_FILENAME = "smpl_motion.npz"


def _stack_root(motion: UnifiedHumanMotion) -> np.ndarray:
    n = motion.frame_count
    arr = np.full((n, 3), np.nan, dtype=np.float64)
    for i, frame in enumerate(motion.frames):
        if frame.root_position is not None:
            arr[i] = frame.root_position
    return arr


def _stack_root_orientation(motion: UnifiedHumanMotion) -> np.ndarray:
    n = motion.frame_count
    arr = np.zeros((n, 4), dtype=np.float64)
    arr[:, 0] = 1.0
    for i, frame in enumerate(motion.frames):
        if frame.root_orientation is not None:
            arr[i] = frame.root_orientation
    return arr


def _stack_pose_confidence(motion: UnifiedHumanMotion) -> np.ndarray:
    return np.array([f.pose_confidence for f in motion.frames], dtype=np.float64)


def _collect_joint_rotations(motion: UnifiedHumanMotion) -> dict[str, Any]:
    """Per-joint quaternion arrays when SMPL rotations are present."""
    joint_ids = sorted(
        {jid for f in motion.frames for jid in f.joint_rotations}
    )
    if not joint_ids:
        return {"joint_rotation_ids_json": json.dumps([]), "joint_rotations_wxyz": np.array([])}

    n = motion.frame_count
    stack = np.full((n, len(joint_ids), 4), np.nan, dtype=np.float64)
    for i, frame in enumerate(motion.frames):
        for j, jid in enumerate(joint_ids):
            q = frame.joint_rotations.get(jid)
            if q is not None:
                stack[i, j] = q

    return {
        "joint_rotation_ids_json": json.dumps(joint_ids),
        "joint_rotations_wxyz": stack,
    }


def _collect_shape_betas(motion: UnifiedHumanMotion) -> np.ndarray:
    n = motion.frame_count
    betas = np.full((n, 10), np.nan, dtype=np.float64)
    for i, frame in enumerate(motion.frames):
        for k in range(10):
            key = f"beta_{k}"
            if key in frame.body_shape_parameters:
                betas[i, k] = frame.body_shape_parameters[key]
    return betas


def build_smpl_motion_arrays(motion: UnifiedHumanMotion) -> dict[str, Any]:
    """Build NPZ payload from ``UnifiedHumanMotion``."""
    if motion.source_backend != "smpl":
        raise ValueError(
            f"Refusing to export smpl_motion.npz for backend '{motion.source_backend}' — "
            "only real SMPL extraction outputs are allowed."
        )

    joint_ids, joint_positions = unified_joint_positions_array(motion)
    rot_payload = _collect_joint_rotations(motion)

    meta = motion.to_dict()
    meta["schema_version"] = UNIFIED_MOTION_SCHEMA_VERSION
    meta["export_note"] = (
        "Mesh-based SMPL recovery (ROMP). Differs from MediaPipe landmark estimation."
    )

    arrays: dict[str, Any] = {
        "timestamps": motion.timestamps.astype(np.float64),
        "fps": np.float64(motion.fps),
        "root_position": _stack_root(motion),
        "root_orientation_wxyz": _stack_root_orientation(motion),
        "joint_positions_3d": joint_positions,
        "pose_confidence": _stack_pose_confidence(motion),
        "body_shape_betas": _collect_shape_betas(motion),
        "source_backend": np.array(motion.source_backend),
        "provider_name": np.array(motion.provider_name or ""),
        "coordinate_system_json": json.dumps(motion.coordinate_system.to_dict()),
        "scale_information_json": json.dumps(motion.scale_information.to_dict()),
        "joint_ids_json": json.dumps(joint_ids),
        "motion_metadata_json": unified_metadata_json(motion),
    }
    arrays.update(rot_payload)
    return arrays


def export_smpl_motion_npz(
    motion: UnifiedHumanMotion,
    output_path: Path,
) -> Path:
    """Write ``smpl_motion.npz`` if motion is from SMPL backend."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = build_smpl_motion_arrays(motion)
    np.savez_compressed(output_path, **arrays)
    logger.info(
        "Exported SMPL motion → %s (%d frames, provider=%s)",
        output_path,
        motion.frame_count,
        motion.provider_name,
    )
    return output_path


def maybe_export_smpl_motion(
    motion: UnifiedHumanMotion,
    run_dir: Path,
) -> Path | None:
    """Export ``smpl_motion.npz`` when backend is SMPL; otherwise return None."""
    if motion.source_backend != "smpl":
        return None
    if motion.valid_frame_count < 1:
        logger.warning("SMPL motion has no valid frames — skipping smpl_motion.npz export")
        return None
    out = Path(run_dir) / SMPL_MOTION_FILENAME
    return export_smpl_motion_npz(motion, out)


__all__ = [
    "SMPL_MOTION_FILENAME",
    "build_smpl_motion_arrays",
    "export_smpl_motion_npz",
    "maybe_export_smpl_motion",
]
