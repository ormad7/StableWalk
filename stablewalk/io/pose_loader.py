"""
Load pose JSON and list detected frames.

Kept free of Matplotlib / MediaPipe so GUI startup does not pay for the
legacy 2D viewer stack when only JSON loading is needed.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from stablewalk.models.pose_data import JointAngles, Keypoint, PoseFrame, PoseSequence

__all__ = [
    "detected_frame_indices",
    "playback_frame_indices",
    "load_pose_sequence",
    "sequence_needs_enrichment",
]


def _parse_bbox(raw: object) -> tuple[float, float, float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        box = tuple(float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return (box[0], box[1], box[2], box[3])


def _parse_keypoints(raw: object) -> list[Keypoint]:
    if isinstance(raw, dict):
        keypoints: list[Keypoint] = []
        for name, vals in raw.items():
            if isinstance(vals, list) and len(vals) >= 2:
                vis = float(vals[3]) if len(vals) >= 4 else float("nan")
                keypoints.append(
                    Keypoint(
                        name=name,
                        x=float(vals[0]),
                        y=float(vals[1]),
                        z=float(vals[2]) if len(vals) > 2 else 0.0,
                        visibility=vis,
                    )
                )
            elif isinstance(vals, dict) and "x" in vals:
                vis_raw = vals.get("visibility", vals.get("confidence"))
                vis = float(vis_raw) if vis_raw is not None else float("nan")
                keypoints.append(
                    Keypoint(
                        name=name,
                        x=float(vals["x"]),
                        y=float(vals["y"]),
                        z=float(vals.get("z", 0.0)),
                        visibility=vis,
                    )
                )
        return keypoints
    if isinstance(raw, list):
        return [Keypoint(**kp) for kp in raw]
    return []


def _restore_visibility_from_skeleton(
    keypoints: list[Keypoint], skeleton_3d: object
) -> list[Keypoint]:
    """Copy detector visibility from skeleton_3d when 2D keypoints omitted it."""
    joints = skeleton_3d.get("joints") if isinstance(skeleton_3d, dict) else None
    if not isinstance(joints, dict) or not keypoints:
        return keypoints
    restored: list[Keypoint] = []
    for kp in keypoints:
        vis = kp.visibility
        if vis == vis:  # not NaN
            restored.append(kp)
            continue
        raw = joints.get(kp.name)
        if isinstance(raw, dict) and raw.get("visibility") is not None:
            try:
                vis = float(raw["visibility"])
            except (TypeError, ValueError):
                vis = kp.visibility
        restored.append(
            Keypoint(name=kp.name, x=kp.x, y=kp.y, z=kp.z, visibility=vis)
        )
    return restored


def _parse_velocities(fd: dict) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Return (scalar speed map, full vector map)."""
    raw_scalar = fd.get("velocity") or {}
    raw_vector = fd.get("velocity_vector") or fd.get("velocities") or {}

    velocity_scalar: dict[str, float] = {}
    velocities: dict[str, dict[str, float]] = {}

    if raw_vector and isinstance(next(iter(raw_vector.values()), None), dict):
        velocities = raw_vector
        velocity_scalar = {
            k: float(v.get("speed", 0.0)) for k, v in velocities.items()
        }
    elif raw_scalar:
        sample = next(iter(raw_scalar.values()), None)
        if isinstance(sample, dict):
            velocities = raw_scalar
            velocity_scalar = {
                k: float(v.get("speed", 0.0)) for k, v in velocities.items()
            }
        else:
            velocity_scalar = {k: float(v) for k, v in raw_scalar.items()}

    return velocity_scalar, velocities


def load_pose_sequence(path: str | Path) -> PoseSequence:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)

    fps = float(data.get("fps", 30.0))
    frames: list[PoseFrame] = []
    for fd in data.get("frames", []):
        frame_index = fd.get("frame", fd.get("frame_index", 0))
        ts_s = float(fd.get("timestamp_s", frame_index / max(fps, 1e-6)))
        if ts_s <= 1e-9 and int(frame_index) > 0:
            ts_s = int(frame_index) / max(fps, 1e-6)
        ts_ms = int(fd.get("timestamp_ms", round(ts_s * 1000)))
        keypoints = _restore_visibility_from_skeleton(
            _parse_keypoints(fd.get("keypoints", [])),
            fd.get("skeleton_3d") or {},
        )
        angles_raw = fd.get("angles") or fd.get("joint_angles")
        if angles_raw and isinstance(angles_raw, dict):
            joint_angles = JointAngles(
                **{f.name: angles_raw.get(f.name) for f in fields(JointAngles)}
            )
        else:
            joint_angles = None
        velocity_scalar, velocities = _parse_velocities(fd)
        positions = fd.get("positions") or {}
        if not positions and keypoints:
            positions = {kp.name: {"x": kp.x, "y": kp.y} for kp in keypoints}

        frames.append(
            PoseFrame(
                frame_index=int(frame_index),
                image_path=fd.get("image_path", ""),
                timestamp_s=ts_s,
                timestamp_ms=ts_ms,
                keypoints=keypoints,
                joint_angles=joint_angles,
                velocities=velocities,
                velocity_scalar=velocity_scalar,
                positions=positions,
                positions_xy=fd.get("keypoints")
                if isinstance(fd.get("keypoints"), dict)
                else {},
                positions_normalized=fd.get("positions_normalized") or {},
                skeleton_3d=fd.get("skeleton_3d") or {},
                gait_phase=fd.get("gait_phase") or {},
                gait_events=fd.get("gait_events") or [],
                detected=fd.get("detected", False),
                skipped=fd.get("skipped", False),
                rejection_reason=str(fd.get("rejection_reason") or ""),
                n_people=int(fd.get("n_people") or 0),
                subject_bbox=_parse_bbox(fd.get("subject_bbox")),
                other_bboxes=[
                    b
                    for b in (
                        _parse_bbox(item) for item in (fd.get("other_bboxes") or [])
                    )
                    if b is not None
                ],
                identity_locked=bool(fd.get("identity_locked", False)),
                tracking_status=str(fd.get("tracking_status") or ""),
            )
        )

    return PoseSequence(
        source_video=data.get("source_video", ""),
        fps=fps,
        frames=frames,
        gait_events_timeline=data.get("gait_events_timeline") or [],
        subject_selection=dict(data.get("subject_selection") or {}),
    )


def detected_frame_indices(sequence: PoseSequence) -> list[int]:
    return [i for i, f in enumerate(sequence.frames) if f.detected and f.keypoints]


def playback_frame_indices(sequence: PoseSequence) -> list[int]:
    """Every stored frame, including missing poses. Do not compress the timeline."""
    return list(range(len(sequence.frames)))


def sequence_needs_enrichment(sequence: PoseSequence) -> bool:
    """True if detected frames lack velocity or 3D skeleton data."""
    for frame in sequence.frames:
        if not frame.detected or not frame.keypoints:
            continue
        if not frame.velocity_scalar and not frame.velocities:
            return True
        sk = frame.skeleton_3d or {}
        if not sk.get("joints"):
            return True
    return False
