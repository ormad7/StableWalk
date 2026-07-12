"""
Backend-native landmarks → canonical StableWalk skeleton conversion.

Canonical joint ids match ``models.joint_registry`` (pelvis, left_hip, …).
Gait analysis modules should prefer these ids over MediaPipe landmark strings.
"""

from __future__ import annotations

from stablewalk.models.gait_motion import (
    DEFAULT_COORDINATE_SYSTEM,
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import (
    DOF_IDS,
    JOINT_IDS,
    JOINT_PARENTS,
    ROOT_JOINT_ID,
    normalize_joint_id,
)
from stablewalk.models.pose_data import JointAngles, Keypoint, PoseFrame, PoseSequence
from stablewalk.pose.backends.types import HumanMotionFrame, HumanMotionSequence
from stablewalk.pose.kinematics import compute_joint_angles

# MediaPipe / legacy pose-estimation landmark name → canonical joint id
BACKEND_LANDMARK_TO_CANONICAL: dict[str, str] = {
    "nose": "head",
    "mid_shoulder": "spine",
    "mid_hip": ROOT_JOINT_ID,
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
    "left_heel": "left_heel",
    "right_heel": "right_heel",
    "left_foot_index": "left_toe",
    "right_foot_index": "right_toe",
    "left_foot": "left_foot",
    "right_foot": "right_foot",
}

CANONICAL_GAIT_JOINTS: tuple[str, ...] = (
    ROOT_JOINT_ID,
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_toe",
    "right_toe",
    "spine",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)


def map_landmark_name_to_canonical(name: str) -> str:
    """Resolve backend or legacy landmark name to canonical joint id."""
    key = name.strip().lower()
    if key in BACKEND_LANDMARK_TO_CANONICAL:
        return BACKEND_LANDMARK_TO_CANONICAL[key]
    normalized = normalize_joint_id(key)
    if normalized in JOINT_IDS or normalized == ROOT_JOINT_ID:
        return normalized
    return key


def canonical_positions_from_landmarks(
    landmarks: dict[str, tuple[float, float, float]],
    *,
    confidences: dict[str, float] | None = None,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, float]]:
    """Convert a backend landmark dict into canonical joint positions."""
    confidences = confidences or {}
    positions: dict[str, tuple[float, float, float]] = {}
    confidence: dict[str, float] = {}

    for name, pos in landmarks.items():
        jid = map_landmark_name_to_canonical(name)
        positions[jid] = pos
        confidence[jid] = confidences.get(name, confidences.get(jid, 1.0))

    if ROOT_JOINT_ID not in positions:
        lh = positions.get("left_hip")
        rh = positions.get("right_hip")
        if lh and rh:
            positions[ROOT_JOINT_ID] = (
                (lh[0] + rh[0]) / 2.0,
                (lh[1] + rh[1]) / 2.0,
                (lh[2] + rh[2]) / 2.0,
            )
            confidence[ROOT_JOINT_ID] = min(
                confidence.get("left_hip", 1.0),
                confidence.get("right_hip", 1.0),
            )

    if "spine" not in positions:
        ls = positions.get("left_shoulder")
        rs = positions.get("right_shoulder")
        pelvis = positions.get(ROOT_JOINT_ID)
        if ls and rs and pelvis:
            positions["spine"] = (
                (ls[0] + rs[0]) / 2.0,
                (ls[1] + rs[1]) / 2.0,
                (ls[2] + rs[2]) / 2.0,
            )

    if "neck" not in positions and "spine" in positions and "head" in positions:
        sp = positions["spine"]
        hd = positions["head"]
        positions["neck"] = (
            sp[0] + 0.42 * (hd[0] - sp[0]),
            sp[1] + 0.42 * (hd[1] - sp[1]),
            sp[2] + 0.42 * (hd[2] - sp[2]),
        )

    return positions, confidence


def keypoints_to_canonical_frame(
    keypoints: list[Keypoint],
    *,
    frame_index: int,
    timestamp_s: float,
    backend_name: str,
    coordinate_system,
    detected: bool = True,
) -> HumanMotionFrame:
    """Build ``HumanMotionFrame`` from legacy MediaPipe ``Keypoint`` list."""
    from stablewalk.pose.backends.types import CoordinateSystemMetadata

    landmarks = {kp.name: (kp.x, kp.y, kp.z) for kp in keypoints}
    confidences = {kp.name: kp.visibility for kp in keypoints}
    positions, conf = canonical_positions_from_landmarks(landmarks, confidences=confidences)

    root = positions.get(ROOT_JOINT_ID)
    return HumanMotionFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        joint_positions_3d=positions,
        landmark_confidence=conf,
        backend_name=backend_name,
        coordinate_system=coordinate_system,
        detected=detected,
        root_position=root,
        raw_landmarks=landmarks,
    )


def human_motion_frame_to_skeleton_snapshot(frame: HumanMotionFrame) -> SkeletonSnapshot:
    """Convert one canonical backend frame to ``SkeletonSnapshot``."""
    joints: dict[str, JointSample] = {}
    for jid, pos in frame.joint_positions_3d.items():
        if jid not in JOINT_IDS and jid != ROOT_JOINT_ID:
            continue
        conf = frame.landmark_confidence.get(jid, 1.0)
        joints[jid] = JointSample(
            joint_id=jid,
            position=Vec3(x=pos[0], y=pos[1], z=pos[2]),
            parent_id=JOINT_PARENTS.get(jid),
        )
        _ = conf

    if ROOT_JOINT_ID in frame.joint_positions_3d and ROOT_JOINT_ID not in joints:
        p = frame.joint_positions_3d[ROOT_JOINT_ID]
        joints[ROOT_JOINT_ID] = JointSample(
            joint_id=ROOT_JOINT_ID,
            position=Vec3(x=p[0], y=p[1], z=p[2]),
            parent_id=None,
        )

    return SkeletonSnapshot(
        frame_index=frame.frame_index,
        time_s=frame.timestamp_s,
        joints=joints,
        dofs={},
        metadata={"backend": frame.backend_name},
    )


def human_motion_sequence_to_gait_motion(
    sequence: HumanMotionSequence,
    *,
    source_id: str = "",
) -> GaitMotionRecording:
    """Convert backend sequence to canonical ``GaitMotionRecording``."""
    snapshots = [
        human_motion_frame_to_skeleton_snapshot(f)
        for f in sequence.frames
        if f.detected
    ]
    return GaitMotionRecording(
        source_id=source_id or sequence.source_video,
        fps=sequence.fps,
        coordinate_system=DEFAULT_COORDINATE_SYSTEM,
        snapshots=snapshots,
        metadata={
            "backend": sequence.backend_name,
            "coordinate_system": sequence.coordinate_system.to_dict(),
        },
    )


def human_motion_frame_to_pose_frame(
    frame: HumanMotionFrame,
    *,
    image_path: str = "",
) -> PoseFrame:
    """
    Bridge canonical backend output to legacy ``PoseFrame`` / ``PoseSequence``.

    Preserves MediaPipe-compatible keypoint names where possible so existing
    gait modules keep working without modification.
    """
    keypoints: list[Keypoint] = []
    if frame.raw_landmarks:
        for name, pos in frame.raw_landmarks.items():
            conf = frame.landmark_confidence.get(
                map_landmark_name_to_canonical(name), 1.0
            )
            keypoints.append(
                Keypoint(name=name, x=pos[0], y=pos[1], z=pos[2], visibility=conf)
            )
    else:
        reverse = {v: k for k, v in BACKEND_LANDMARK_TO_CANONICAL.items()}
        for jid, pos in frame.joint_positions_3d.items():
            name = reverse.get(jid, jid)
            keypoints.append(
                Keypoint(
                    name=name,
                    x=pos[0],
                    y=pos[1],
                    z=pos[2],
                    visibility=frame.landmark_confidence.get(jid, 1.0),
                )
            )

    joint_angles: JointAngles | None = None
    if frame.detected and keypoints:
        joint_angles = compute_joint_angles(keypoints)

    return PoseFrame(
        frame_index=frame.frame_index,
        image_path=image_path,
        timestamp_s=frame.timestamp_s,
        timestamp_ms=int(round(frame.timestamp_s * 1000)),
        keypoints=keypoints,
        joint_angles=joint_angles,
        detected=frame.detected,
    )


def human_motion_sequence_to_pose_sequence(
    sequence: HumanMotionSequence,
) -> PoseSequence:
    """Bridge backend output to legacy ``PoseSequence`` for existing analysis."""
    frames = [
        human_motion_frame_to_pose_frame(f, image_path=sequence.source_video)
        for f in sequence.frames
    ]
    return PoseSequence(
        source_video=sequence.source_video,
        fps=sequence.fps,
        frames=frames,
    )


def canonical_to_trc_landmarks(
    positions: dict[str, tuple[float, float, float]],
    *,
    scale_to_mm: float = 1000.0,
) -> dict[str, tuple[float, float, float]]:
    """
    Map canonical skeleton joints to StableWalk OpenSim TRC landmark names.

    Used by backend comparison for marker reconstruction confidence scoring.
    """
    mapping = {
        "left_shoulder": "L_SHOULDER",
        "right_shoulder": "R_SHOULDER",
        "head": "HEAD",
        "left_hip": "L_HIP",
        "right_hip": "R_HIP",
        "left_knee": "L_KNEE",
        "right_knee": "R_KNEE",
        "left_ankle": "L_ANKLE",
        "right_ankle": "R_ANKLE",
        "left_heel": "L_HEEL",
        "right_heel": "R_HEEL",
        "left_toe": "L_TOE",
        "right_toe": "R_TOE",
    }
    out: dict[str, tuple[float, float, float]] = {}
    for jid, trc_name in mapping.items():
        pos = positions.get(jid)
        if pos is None:
            continue
        out[trc_name] = (pos[0] * scale_to_mm, pos[1] * scale_to_mm, pos[2] * scale_to_mm)
    return out
