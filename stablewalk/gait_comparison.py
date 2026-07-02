"""
Compare two gait sequences (e.g. stable vs unstable walking).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stablewalk.gait_dof import GAIT_ANGLE_FIELDS
from stablewalk.models.pose_data import PoseSequence
from stablewalk.visualization import detected_frame_indices, load_pose_sequence


@dataclass
class GaitComparison:
    """Angle differences between reference and sample sequences."""

    reference_name: str
    sample_name: str
    reference_knee_left: list[float] = field(default_factory=list)
    reference_knee_right: list[float] = field(default_factory=list)
    sample_knee_left: list[float] = field(default_factory=list)
    sample_knee_right: list[float] = field(default_factory=list)
    mean_abs_diff: dict[str, float] = field(default_factory=dict)
    max_abs_diff: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        if not self.mean_abs_diff:
            return "No overlapping angles to compare."
        top = sorted(self.mean_abs_diff.items(), key=lambda x: -x[1])[:5]
        lines = [f"  {k}: mean |Δ|={v:.1f}°" for k, v in top]
        return "Mean angle differences (sample vs reference):\n" + "\n".join(lines)


def compare_sequences(
    reference: PoseSequence,
    sample: PoseSequence,
    *,
    reference_name: str = "reference",
    sample_name: str = "sample",
) -> GaitComparison:
    """Compare gait DOF angles between two processed videos."""
    ref_frames = [reference.frames[i] for i in detected_frame_indices(reference)]
    samp_frames = [sample.frames[i] for i in detected_frame_indices(sample)]

    cmp = GaitComparison(reference_name=reference_name, sample_name=sample_name)

    for f in ref_frames:
        if f.joint_angles:
            if f.joint_angles.left_knee is not None:
                cmp.reference_knee_left.append(f.joint_angles.left_knee)
            if f.joint_angles.right_knee is not None:
                cmp.reference_knee_right.append(f.joint_angles.right_knee)

    for f in samp_frames:
        if f.joint_angles:
            if f.joint_angles.left_knee is not None:
                cmp.sample_knee_left.append(f.joint_angles.left_knee)
            if f.joint_angles.right_knee is not None:
                cmp.sample_knee_right.append(f.joint_angles.right_knee)

    n = min(len(ref_frames), len(samp_frames))
    for i in range(n):
        ra, sa = ref_frames[i].joint_angles, samp_frames[i].joint_angles
        if not ra or not sa:
            continue
        for name in GAIT_ANGLE_FIELDS:
            rv = getattr(ra, name, None)
            sv = getattr(sa, name, None)
            if rv is None or sv is None:
                continue
            diff = abs(sv - rv)
            cmp.mean_abs_diff[name] = cmp.mean_abs_diff.get(name, 0.0) + diff
            cmp.max_abs_diff[name] = max(cmp.max_abs_diff.get(name, 0.0), diff)

    if n > 0:
        for name in list(cmp.mean_abs_diff):
            cmp.mean_abs_diff[name] /= n

    return cmp


def compare_pose_files(
    reference_path: str,
    sample_path: str,
) -> GaitComparison:
    ref = load_pose_sequence(reference_path)
    samp = load_pose_sequence(sample_path)
    return compare_sequences(
        ref,
        samp,
        reference_name=reference_path,
        sample_name=sample_path,
    )
