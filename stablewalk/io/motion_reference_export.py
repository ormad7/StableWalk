"""
Export canonical motion reference datasets for Real-to-Sim retargeting.

Output format: ``stablewalk_motion.npz`` — consumable by future Isaac Lab or
imitation-learning pipelines. Does not include force data.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
from stablewalk.analysis.gait_cycle_analysis import GaitCycleAnalysisResult, analyze_gait_cycles
from stablewalk.analysis.gait_feature_analysis import estimate_body_segment_dimensions
from stablewalk.analysis.isaac_lab_integration import MOTION_REFERENCE_FILENAME
from stablewalk.io.pose_loader import load_pose_sequence
from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.models.joint_registry import JOINT_IDS, ROOT_JOINT_ID
from stablewalk.models.pose_data import PoseSequence

logger = logging.getLogger(__name__)

MOTION_REFERENCE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class MotionReferenceExportResult:
    """Paths and metadata from a motion reference export."""

    npz_path: Path
    run_name: str
    frame_count: int
    fps: float
    joint_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "npz_path": str(self.npz_path),
            "run_name": self.run_name,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "joint_count": self.joint_count,
            "schema_version": MOTION_REFERENCE_SCHEMA_VERSION,
        }


def _pelvis_position(snap) -> tuple[float, float, float] | None:
    lh = snap.joints.get("left_hip")
    rh = snap.joints.get("right_hip")
    if lh is None or rh is None:
        return None
    return (
        (lh.position.x + rh.position.x) * 0.5,
        (lh.position.y + rh.position.y) * 0.5,
        (lh.position.z + rh.position.z) * 0.5,
    )


def build_motion_reference_arrays(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
) -> dict[str, Any]:
    """Build numpy arrays for ``stablewalk_motion.npz``."""
    dimensions = estimate_body_segment_dimensions(recording)
    joint_ids = [ROOT_JOINT_ID, *JOINT_IDS]
    n = len(recording.snapshots)
    j = len(joint_ids)

    timestamps = np.zeros(n, dtype=np.float64)
    root_positions = np.full((n, 3), np.nan, dtype=np.float64)
    canonical_joint_positions = np.full((n, j, 3), np.nan, dtype=np.float64)
    left_contact_mask = np.zeros(n, dtype=np.int8)
    right_contact_mask = np.zeros(n, dtype=np.int8)

    frame_to_idx = {s.frame_index: i for i, s in enumerate(recording.snapshots)}

    for i, snap in enumerate(recording.snapshots):
        timestamps[i] = snap.time_s
        pelvis = _pelvis_position(snap)
        if pelvis is not None:
            root_positions[i] = pelvis
        for j_idx, jid in enumerate(joint_ids):
            joint = snap.joints.get(jid)
            if joint is not None:
                canonical_joint_positions[i, j_idx] = (
                    joint.position.x,
                    joint.position.y,
                    joint.position.z,
                )

    for state in cycles.per_frame:
        idx = frame_to_idx.get(state.frame_index)
        if idx is not None:
            left_contact_mask[idx] = int(state.left_contact)
            right_contact_mask[idx] = int(state.right_contact)

    body_scale_metadata = {
        "schema_version": MOTION_REFERENCE_SCHEMA_VERSION,
        "coordinate_system": recording.coordinate_system,
        "source": recording.source,
        "source_kind": recording.source_kind,
        "dimensions": dimensions.to_dict(),
        "contact_confidence": cycles.metrics.contact_confidence,
        "contact_note": (
            "left_contact_mask / right_contact_mask are heuristic foot-contact timing "
            "signals — not ground reaction forces."
        ),
        "joint_ids": joint_ids,
    }

    return {
        "timestamps": timestamps,
        "fps": np.float64(recording.fps),
        "root_positions": root_positions,
        "canonical_joint_positions": canonical_joint_positions,
        "left_contact_mask": left_contact_mask,
        "right_contact_mask": right_contact_mask,
        "body_scale_metadata_json": json.dumps(body_scale_metadata),
        "joint_ids_json": json.dumps(joint_ids),
    }


def export_motion_reference_npz(
    recording: GaitMotionRecording,
    cycles: GaitCycleAnalysisResult,
    output_path: Path,
) -> MotionReferenceExportResult:
    """Write ``stablewalk_motion.npz`` to ``output_path``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = build_motion_reference_arrays(recording, cycles)
    # root_orientations reserved — not available from current MediaPipe backend
    arrays["root_orientations"] = np.array([], dtype=np.float64)
    arrays["joint_rotations_json"] = json.dumps({})

    np.savez_compressed(output_path, **arrays)
    run_name = output_path.stem.replace("_motion", "")
    logger.info("Exported motion reference → %s (%d frames)", output_path, len(recording.snapshots))

    return MotionReferenceExportResult(
        npz_path=output_path,
        run_name=run_name,
        frame_count=len(recording.snapshots),
        fps=float(recording.fps),
        joint_count=len(json.loads(arrays["joint_ids_json"])),
    )


def export_motion_reference_from_poses(
    poses_path: Path,
    output_dir: Path,
    *,
    run_name: str | None = None,
) -> MotionReferenceExportResult:
    """Load pose JSON, analyze gait, export ``{run_dir}/stablewalk_motion.npz``."""
    sequence = load_pose_sequence(poses_path)
    name = run_name or poses_path.stem.replace("_poses", "")
    recording = pose_sequence_to_gait_motion(sequence)
    cycles = analyze_gait_cycles(recording)
    out_path = output_dir / name / MOTION_REFERENCE_FILENAME
    return export_motion_reference_npz(recording, cycles, out_path)


def export_motion_reference_from_sequence(
    sequence: PoseSequence,
    output_dir: Path,
    *,
    run_name: str,
) -> MotionReferenceExportResult:
    """Export from an in-memory pose sequence."""
    recording = pose_sequence_to_gait_motion(sequence)
    cycles = analyze_gait_cycles(recording)
    out_path = output_dir / run_name / MOTION_REFERENCE_FILENAME
    return export_motion_reference_npz(recording, cycles, out_path)


__all__ = [
    "MOTION_REFERENCE_SCHEMA_VERSION",
    "MotionReferenceExportResult",
    "build_motion_reference_arrays",
    "export_motion_reference_npz",
    "export_motion_reference_from_poses",
    "export_motion_reference_from_sequence",
]
