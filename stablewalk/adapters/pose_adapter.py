"""
Adapter: pose-estimation output → canonical ``GaitMotionRecording``.

Bridges the existing ``PoseSequence`` pipeline JSON to the unified gait motion
model used by inspection tools and OpenSim export.

Downstream OpenSim export: ``stablewalk.biomechanics.export_to_opensim_format(recording)``
"""

from __future__ import annotations

import logging

from stablewalk.models.gait_motion import (
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
    LEGACY_JOINT_ALIASES,
    normalize_dof_id,
    normalize_joint_id,
)
from stablewalk.models.pose_data import PoseFrame, PoseSequence
from stablewalk.coordinates.coordinate_map import (
    CANONICAL_FRAME_ID,
    audit_recording_anatomy,
)

# Pose-estimation landmark → canonical joint
_POSE_LANDMARK_TO_JOINT: dict[str, str] = {
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
}

# Legacy angle field on JointAngles → canonical DOF id
_POSE_ANGLE_TO_DOF: dict[str, str] = {
    "left_shoulder": "left_shoulder_flexion",
    "right_shoulder": "right_shoulder_flexion",
    "left_elbow": "left_elbow_flexion",
    "right_elbow": "right_elbow_flexion",
    "left_hip": "left_hip_flexion",
    "right_hip": "right_hip_flexion",
    "left_knee": "left_knee_flexion",
    "right_knee": "right_knee_flexion",
    "left_ankle_flexion": "left_ankle_flexion",
    "right_ankle_flexion": "right_ankle_flexion",
    "left_ankle": "left_ankle_flexion",
    "right_ankle": "right_ankle_flexion",
    "neck": "neck",
    "head_neck": "head_neck",
    "torso_tilt": "spine_flexion",
    "pelvis_rotation": "spine_flexion",
    "spine": "spine_flexion",
}


def _vec3_from_blob(raw: dict | None) -> Vec3 | None:
    if not raw or not isinstance(raw, dict):
        return None
    return Vec3(
        x=float(raw.get("x", 0.0)),
        y=float(raw.get("y", 0.0)),
        z=float(raw.get("z", 0.0)),
    )


def _fill_from_keypoints(
    keypoints: list,
    positions: dict[str, Vec3],
) -> None:
    """Fill any missing gait landmarks from a live MediaPipe keypoint list."""
    from stablewalk.pose.skeleton_3d import reconstruct_skeleton_3d

    try:
        skel = reconstruct_skeleton_3d(keypoints)
    except Exception:
        return
    for raw_name, joint in skel.joints.items():
        canonical = _POSE_LANDMARK_TO_JOINT.get(raw_name) or normalize_joint_id(raw_name)
        if canonical not in JOINT_IDS and canonical != ROOT_JOINT_ID:
            continue
        if canonical in positions:
            continue
        positions[canonical] = Vec3(x=joint.x, y=joint.y, z=joint.z)


def _derive_anatomical_joints(positions: dict[str, Vec3]) -> None:
    """
    Fill pelvis / spine / neck from hips, shoulders, and head when missing.

    Keeps the stick figure anatomically connected for gait visualization.
    """
    lh = positions.get("left_hip")
    rh = positions.get("right_hip")
    if lh and rh:
        positions.setdefault(
            ROOT_JOINT_ID,
            Vec3(
                x=(lh.x + rh.x) / 2.0,
                y=(lh.y + rh.y) / 2.0,
                z=(lh.z + rh.z) / 2.0,
            ),
        )

    ls = positions.get("left_shoulder")
    rs = positions.get("right_shoulder")
    shoulder_mid: Vec3 | None = None
    if ls and rs:
        shoulder_mid = Vec3(
            x=(ls.x + rs.x) / 2.0,
            y=(ls.y + rs.y) / 2.0,
            z=(ls.z + rs.z) / 2.0,
        )

    pelvis = positions.get(ROOT_JOINT_ID)
    head = positions.get("head")

    if pelvis and shoulder_mid:
        positions.setdefault(
            "spine",
            Vec3(
                x=pelvis.x + 0.78 * (shoulder_mid.x - pelvis.x),
                y=pelvis.y + 0.78 * (shoulder_mid.y - pelvis.y),
                z=pelvis.z + 0.78 * (shoulder_mid.z - pelvis.z),
            ),
        )

    spine = positions.get("spine")
    if spine and head:
        positions.setdefault(
            "neck",
            Vec3(
                x=spine.x + 0.42 * (head.x - spine.x),
                y=spine.y + 0.42 * (head.y - spine.y),
                z=spine.z + 0.42 * (head.z - spine.z),
            ),
        )
    elif spine and shoulder_mid:
        positions.setdefault(
            "neck",
            Vec3(
                x=spine.x + 0.78 * (shoulder_mid.x - spine.x),
                y=spine.y + 0.78 * (shoulder_mid.y - spine.y),
                z=spine.z + 0.78 * (shoulder_mid.z - spine.z),
            ),
        )


def _extract_joint_positions(frame: PoseFrame) -> dict[str, Vec3]:
    """Resolve canonical joint XYZ from enriched frame data."""
    positions: dict[str, Vec3] = {}

    sources: list[dict] = []
    if frame.positions_normalized:
        sources.append(frame.positions_normalized)
    sk = frame.skeleton_3d or {}
    if isinstance(sk, dict) and sk.get("joints"):
        sources.append(sk["joints"])
    if frame.positions:
        sources.append(frame.positions)

    for src in sources:
        for raw_name, raw_val in src.items():
            canonical = _POSE_LANDMARK_TO_JOINT.get(raw_name) or normalize_joint_id(raw_name)
            if canonical not in JOINT_IDS and canonical != ROOT_JOINT_ID:
                continue
            vec = _vec3_from_blob(raw_val if isinstance(raw_val, dict) else None)
            if vec and canonical not in positions:
                positions[canonical] = vec

    # Fallback: reconstruct missing limbs directly from MediaPipe keypoints
    if frame.keypoints and frame.detected:
        _fill_from_keypoints(frame.keypoints, positions)

    _ensure_foot_landmarks(positions)

    _derive_anatomical_joints(positions)
    return positions


def _body_span_y(positions: dict[str, Vec3]) -> float:
    ys = [p.y for p in positions.values()]
    if len(ys) < 2:
        return 0.5
    return max(ys) - min(ys)


def _ensure_foot_landmarks(positions: dict[str, Vec3]) -> None:
    """
    Derive or repair foot landmarks from the ipsilateral ankle.

    Heel/toe from raw 2D ``positions`` can sit above the hip-centered ankle
    (mixed coordinate sources). Replace invalid foot points with ankle offsets.
    """
    span = _body_span_y(positions)
    foot_drop = max(span * 0.035, 0.025)
    foot_fwd = max(span * 0.055, 0.04)
    heel_back = max(span * 0.02, 0.015)
    for side in ("left", "right"):
        ankle_id = f"{side}_ankle"
        a = positions.get(ankle_id)
        if a is None:
            continue
        for joint_id, drop, dz in (
            (f"{side}_foot", foot_drop, foot_fwd * 0.6),
            (f"{side}_heel", foot_drop * 0.5, -heel_back),
            (f"{side}_toe", foot_drop, foot_fwd),
        ):
            pos = positions.get(joint_id)
            invalid = pos is None or pos.y > a.y + 0.015
            if invalid:
                positions[joint_id] = Vec3(a.x, a.y - drop, a.z + dz)


def _extract_image_xy(frame: PoseFrame) -> dict[str, list[float]]:
    """
    Hip-centered 2D landmarks from the video frame (y up, body-scaled).

    Stored in snapshot metadata for the 2D Pose Skeleton display mode.
    """
    raw: dict[str, tuple[float, float]] = {}

    if frame.positions_xy:
        for name, val in frame.positions_xy.items():
            if not isinstance(val, (list, tuple)) or len(val) < 2:
                continue
            canonical = _POSE_LANDMARK_TO_JOINT.get(name) or normalize_joint_id(name)
            if canonical in JOINT_IDS or canonical == ROOT_JOINT_ID:
                raw[canonical] = (float(val[0]), float(val[1]))

    if frame.keypoints:
        for kp in frame.keypoints:
            if kp.visibility < 0.3:
                continue
            canonical = _POSE_LANDMARK_TO_JOINT.get(kp.name) or normalize_joint_id(kp.name)
            if canonical in JOINT_IDS or canonical == ROOT_JOINT_ID:
                raw.setdefault(canonical, (float(kp.x), float(kp.y)))

    if frame.positions:
        for name, val in frame.positions.items():
            if not isinstance(val, dict):
                continue
            canonical = _POSE_LANDMARK_TO_JOINT.get(name) or normalize_joint_id(name)
            if canonical in JOINT_IDS or canonical == ROOT_JOINT_ID:
                raw.setdefault(
                    canonical,
                    (float(val.get("x", 0)), float(val.get("y", 0))),
                )

    if not raw:
        return {}

    hip = raw.get(ROOT_JOINT_ID)
    if not hip:
        lh = raw.get("left_hip")
        rh = raw.get("right_hip")
        if lh and rh:
            hip = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
            raw[ROOT_JOINT_ID] = hip

    if not hip:
        return {}

    centered: dict[str, tuple[float, float]] = {
        jid: (x - hip[0], hip[1] - y) for jid, (x, y) in raw.items()
    }

    ys = [p[1] for p in centered.values()]
    if len(ys) >= 2:
        span = max(ys) - min(ys)
        if span > 1e-6:
            scale = 1.0 / span
            centered = {jid: (x * scale, y * scale) for jid, (x, y) in centered.items()}

    # Trunk joints for stick-figure connectivity
    lh = centered.get("left_hip")
    rh = centered.get("right_hip")
    pelvis = centered.get(ROOT_JOINT_ID)
    if lh and rh:
        pelvis = pelvis or ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
        centered[ROOT_JOINT_ID] = pelvis
    ls = centered.get("left_shoulder")
    rs = centered.get("right_shoulder")
    shoulder_mid = (
        ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0) if ls and rs else None
    )
    head = centered.get("head")
    if shoulder_mid and pelvis:
        centered.setdefault(
            "spine",
            (
                pelvis[0] + 0.78 * (shoulder_mid[0] - pelvis[0]),
                pelvis[1] + 0.78 * (shoulder_mid[1] - pelvis[1]),
            ),
        )
    spine = centered.get("spine")
    if spine and head:
        centered.setdefault(
            "neck",
            (
                spine[0] + 0.42 * (head[0] - spine[0]),
                spine[1] + 0.42 * (head[1] - spine[1]),
            ),
        )

    return {jid: [round(x, 6), round(y, 6)] for jid, (x, y) in centered.items()}


def _scalar_speed(frame: PoseFrame, joint_id: str) -> float | None:
    if joint_id in frame.velocity_scalar:
        return float(frame.velocity_scalar[joint_id])
    vel = frame.velocities.get(joint_id)
    if isinstance(vel, dict):
        return float(vel.get("speed", 0.0))
    return None


def _velocity_vector(frame: PoseFrame, joint_id: str) -> Vec3 | None:
    vel = frame.velocities.get(joint_id)
    if not isinstance(vel, dict):
        return None
    return Vec3(
        x=float(vel.get("vx", 0.0)),
        y=float(vel.get("vy", 0.0)),
        z=float(vel.get("z", vel.get("vz", 0.0))),
    )


def pose_frame_to_snapshot(frame: PoseFrame, *, fps: float) -> SkeletonSnapshot:
    """Convert one ``PoseFrame`` into a ``SkeletonSnapshot``."""
    positions = _extract_joint_positions(frame)
    joints: dict[str, JointSample] = {}

    for joint_id in JOINT_IDS:
        pos = positions.get(joint_id)
        if not pos:
            continue
        angle = None
        if frame.joint_angles:
            for attr, dof_id in _POSE_ANGLE_TO_DOF.items():
                if dof_id.replace("_flexion", "") == joint_id or (
                    joint_id in ("left_ankle", "right_ankle")
                    and attr.endswith("ankle")
                ):
                    val = getattr(frame.joint_angles, attr, None)
                    if val is not None:
                        angle = float(val)
                        break

        joints[joint_id] = JointSample(
            joint_id=joint_id,
            position=pos,
            parent_id=JOINT_PARENTS.get(joint_id),
            angle_deg=angle,
            velocity=_scalar_speed(frame, joint_id),
            velocity_vector=_velocity_vector(frame, joint_id),
        )

    if ROOT_JOINT_ID in positions:
        joints[ROOT_JOINT_ID] = JointSample(
            joint_id=ROOT_JOINT_ID,
            position=positions[ROOT_JOINT_ID],
            parent_id=None,
            velocity=_scalar_speed(frame, "mid_hip"),
        )

    dofs: dict[str, DofSample] = {}
    if frame.joint_angles:
        for attr, dof_id in _POSE_ANGLE_TO_DOF.items():
            val = getattr(frame.joint_angles, attr, None)
            if val is None:
                continue
            dofs[dof_id] = DofSample(
                dof_id=dof_id,
                angle_deg=float(val),
                joint_id=LEGACY_JOINT_ALIASES.get(attr, dof_id.replace("_flexion", "")),
            )

    # Fill missing DOF entries with None placeholders for consistent keys
    for dof_id in DOF_IDS:
        dofs.setdefault(dof_id, DofSample(dof_id=dof_id, angle_deg=None))

    image_xy = _extract_image_xy(frame)

    landmark_visibility: dict[str, float] = {}
    if frame.keypoints:
        for kp in frame.keypoints:
            canonical = _POSE_LANDMARK_TO_JOINT.get(kp.name) or normalize_joint_id(kp.name)
            if canonical in JOINT_IDS or canonical == ROOT_JOINT_ID:
                landmark_visibility[canonical] = float(kp.visibility)

    return SkeletonSnapshot(
        frame_index=frame.frame_index,
        time_s=frame.time_seconds(fps),
        joints=joints,
        dofs=dofs,
        metadata={
            "detected": frame.detected,
            "gait_phase": dict(frame.gait_phase),
            "gait_events": list(frame.gait_events),
            "image_xy": image_xy,
            "landmark_visibility": landmark_visibility,
        },
    )


def attach_snapshot_kinematic_rates(
    snapshots: list[SkeletonSnapshot],
    *,
    fps: float,
) -> None:
    """
    Fill missing DOF angular rates and joint linear speeds from consecutive snapshots.

    First snapshot keeps N/A rates (no prior frame). Mutates the list in place.
    """
    if not snapshots:
        return
    default_dt = 1.0 / max(fps, 1e-6)
    for index in range(1, len(snapshots)):
        prev = snapshots[index - 1]
        curr = snapshots[index]
        dt = curr.time_s - prev.time_s
        if dt <= 0:
            dt = default_dt

        new_dofs = dict(curr.dofs)
        for dof_id, dof in curr.dofs.items():
            if dof.velocity_deg_s is not None:
                continue
            prev_dof = prev.dofs.get(dof_id)
            if (
                prev_dof is not None
                and prev_dof.angle_deg is not None
                and dof.angle_deg is not None
            ):
                new_dofs[dof_id] = DofSample(
                    dof_id=dof_id,
                    angle_deg=dof.angle_deg,
                    velocity_deg_s=(dof.angle_deg - prev_dof.angle_deg) / dt,
                    joint_id=dof.joint_id,
                )

        new_joints = dict(curr.joints)
        for joint_id, joint in curr.joints.items():
            if joint.velocity is not None and joint.velocity_vector is not None:
                continue
            prev_joint = prev.joints.get(joint_id)
            if prev_joint is None:
                continue
            vx = (joint.position.x - prev_joint.position.x) / dt
            vy = (joint.position.y - prev_joint.position.y) / dt
            vz = (joint.position.z - prev_joint.position.z) / dt
            vel_vec = Vec3(vx, vy, vz)
            vel_scalar = (vx * vx + vy * vy + vz * vz) ** 0.5
            new_joints[joint_id] = JointSample(
                joint_id=joint.joint_id,
                position=joint.position,
                parent_id=joint.parent_id,
                angle_deg=joint.angle_deg,
                velocity=joint.velocity if joint.velocity is not None else vel_scalar,
                velocity_vector=(
                    joint.velocity_vector
                    if joint.velocity_vector is not None
                    else vel_vec
                ),
            )

        snapshots[index] = SkeletonSnapshot(
            frame_index=curr.frame_index,
            time_s=curr.time_s,
            joints=new_joints,
            dofs=new_dofs,
            metadata=dict(curr.metadata),
        )


def pose_sequence_to_gait_motion(
    sequence: PoseSequence,
    *,
    detected_only: bool = True,
) -> GaitMotionRecording:
    """
    Convert a processed ``PoseSequence`` into ``GaitMotionRecording``.

    Args:
        sequence: Output from the pose-estimation pipeline.
        detected_only: If True, include only frames with ``detected=True``.
    """
    fps = max(sequence.fps, 1e-6)
    frames = [
        f for f in sequence.frames if (f.detected or not detected_only)
    ]

    snapshots = [pose_frame_to_snapshot(f, fps=fps) for f in frames]
    attach_snapshot_kinematic_rates(snapshots, fps=fps)

    recording = GaitMotionRecording(
        source=sequence.source_video or "pose_estimation",
        fps=fps,
        snapshots=snapshots,
        source_kind="pose_estimation",
        coordinate_system=CANONICAL_FRAME_ID,
        metadata={"detected_count": sum(1 for f in frames if f.detected)},
    )
    coord_warnings = audit_recording_anatomy(recording)
    if coord_warnings:
        recording.metadata["coordinate_warnings"] = coord_warnings
        log = logging.getLogger(__name__)
        for msg in coord_warnings[:3]:
            log.warning("Coordinate audit: %s", msg)
    recording.build_time_series()
    return recording
