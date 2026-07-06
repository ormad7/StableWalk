"""Simple cross-demo gait comparison summaries for academic presentations.

Values are computed from the real ``GaitMotionRecording`` time series — nothing
is hardcoded. StableWalk displays biomechanical movement data only; it does not
diagnose medical conditions or rank gaits as better/worse.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from stablewalk.models.gait_motion import GaitMotionRecording
from stablewalk.ui.dof_selection import anchor_joint_for_item, label_for_item
from stablewalk.ui.media.demo_gait import DemoGaitExample


@dataclass(frozen=True)
class DemoGaitComparisonRow:
    demo_key: str
    demo_type: str
    joint_label: str
    max_angle_deg: float | None
    avg_velocity: float | None
    velocity_unit: str = "m/s"


def compute_joint_summary(
    recording: GaitMotionRecording,
    item_id: str,
) -> tuple[float | None, float | None, str]:
    """Max flexion angle and mean scalar speed for a selected joint over the clip."""
    joint_id = anchor_joint_for_item(item_id) or item_id
    ts = recording.build_time_series()
    angle_vals = [float(a) for a in ts.angles.get(joint_id, []) if a is not None]
    vel_vals = [float(v) for v in ts.velocities.get(joint_id, []) if v is not None]
    max_angle = max(angle_vals) if angle_vals else None
    avg_vel = mean(vel_vals) if vel_vals else None
    return max_angle, avg_vel, "m/s"


def build_comparison_row(
    recording: GaitMotionRecording,
    example: DemoGaitExample,
    item_id: str,
) -> DemoGaitComparisonRow:
    max_angle, avg_vel, unit = compute_joint_summary(recording, item_id)
    return DemoGaitComparisonRow(
        demo_key=example.key,
        demo_type=example.display_name,
        joint_label=label_for_item(item_id),
        max_angle_deg=max_angle,
        avg_velocity=avg_vel,
        velocity_unit=unit,
    )
