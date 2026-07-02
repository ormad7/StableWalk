"""
Validate MediaPipe pose output for gait analysis.

Rejects false positives (cars, background blobs) while accepting
real walking poses including side views.
"""

from __future__ import annotations

from stablewalk.pose.dof import GAIT_ANGLE_FIELDS, MIN_GAIT_DOF
from stablewalk.pose.kinematics import pose_bounding_spans
from stablewalk.models.pose_data import JointAngles, Keypoint

MIN_VISIBILITY = 0.5
LOWER_BODY_VISIBILITY = 0.4

LIMB_JOINTS = (
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
)

LOWER_BODY_JOINTS = (
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


def _get(
    keypoints: dict[str, Keypoint],
    name: str,
    min_visibility: float = MIN_VISIBILITY,
) -> Keypoint | None:
    kp = keypoints.get(name)
    if kp is None or kp.visibility < min_visibility:
        return None
    return kp


def _leg_chain_valid(
    by_name: dict[str, Keypoint],
    side: str,
    min_visibility: float = LOWER_BODY_VISIBILITY,
) -> bool:
    """Hip → knee → ankle should descend in image y (y increases downward)."""
    hip = _get(by_name, f"{side}_hip", min_visibility)
    knee = _get(by_name, f"{side}_knee", min_visibility)
    ankle = _get(by_name, f"{side}_ankle", min_visibility)
    if not hip or not knee or not ankle:
        return False
    return hip.y < knee.y < ankle.y


def is_plausible_human_pose(
    keypoints: list[Keypoint],
    min_visibility: float = MIN_VISIBILITY,
) -> bool:
    """True when landmarks look like a real person (front or side view)."""
    if not keypoints:
        return False

    by_name = {kp.name: kp for kp in keypoints}

    for name in ("nose", "left_shoulder", "right_shoulder"):
        if _get(by_name, name, min_visibility) is None:
            return False

    left_shoulder = by_name["left_shoulder"]
    right_shoulder = by_name["right_shoulder"]
    shoulder_width = abs(left_shoulder.x - right_shoulder.x)

    left_hip = by_name.get("left_hip")
    right_hip = by_name.get("right_hip")
    hip_width = 0.0
    if left_hip and right_hip:
        hip_width = abs(left_hip.x - right_hip.x)

    # Side views collapse shoulder width; require hips or a tall body span.
    if shoulder_width < 0.06:
        x_span, y_span, visible_count = pose_bounding_spans(keypoints, min_visibility)
        if hip_width < 0.04 and y_span < 0.25:
            return False
        if visible_count < 10 or x_span < 0.05:
            return False

    x_span, y_span, visible_count = pose_bounding_spans(keypoints, min_visibility)
    if visible_count < 10 or x_span < 0.05 or y_span < 0.18:
        return False

    limb_visible = sum(
        1 for name in LIMB_JOINTS if _get(by_name, name, min_visibility * 0.8) is not None
    )
    if limb_visible < 4:
        return False

    return True


def has_full_body_pose(
    keypoints: list[Keypoint],
    min_visibility: float = LOWER_BODY_VISIBILITY,
) -> bool:
    """
    True when lower-body gait joints are visible and leg geometry is plausible.

    Used to skip frames without hips, knees, and ankles.
    """
    if not is_plausible_human_pose(keypoints, min_visibility=min_visibility):
        return False

    by_name = {kp.name: kp for kp in keypoints}

    for name in LOWER_BODY_JOINTS:
        if _get(by_name, name, min_visibility) is None:
            return False

    if not (_leg_chain_valid(by_name, "left", min_visibility) or _leg_chain_valid(by_name, "right", min_visibility)):
        return False

    leg_ys = [
        by_name[n].y
        for n in LOWER_BODY_JOINTS
        if _get(by_name, n, min_visibility) is not None
    ]
    if max(leg_ys) - min(leg_ys) < 0.12:
        return False

    return True


def meets_min_gait_dof(joint_angles: JointAngles | None) -> bool:
    """At least 12 gait DOF angles computed."""
    if joint_angles is None:
        return False
    count = sum(
        1 for name in GAIT_ANGLE_FIELDS if getattr(joint_angles, name, None) is not None
    )
    return count >= MIN_GAIT_DOF


# Alias used by visualization
is_valid_pose = has_full_body_pose
