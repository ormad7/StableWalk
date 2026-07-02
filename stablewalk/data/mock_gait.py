"""
Mock gait motion generator.

Produces a ``GaitMotionRecording`` with plausible walking kinematics when
real video or OpenSim data is unavailable. Useful for UI development and tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from stablewalk.models.gait_motion import (
    DEFAULT_COORDINATE_SYSTEM,
    DofSample,
    GaitMotionRecording,
    JointSample,
    SkeletonSnapshot,
    Vec3,
)
from stablewalk.models.joint_registry import DOF_IDS, DOF_JOINT, JOINT_IDS, JOINT_PARENTS, ROOT_JOINT_ID


@dataclass(frozen=True)
class MockGaitConfig:
    """Parameters for synthetic walking."""

    fps: float = 30.0
    duration_s: float = 4.0
    cadence_hz: float = 1.0  # steps per second (both feet ≈ 2× contact frequency)
    stride_length_m: float = 0.55
    pelvis_height_m: float = 0.95
    arm_swing_deg: float = 22.0


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _leg_angles(phase: float) -> tuple[float, float, float]:
    """
    Return (hip, knee, ankle) flexion degrees for one leg.

    ``phase`` in [0, 1): 0 = heel strike, 0.5 = opposite heel strike.
    """
    # Hip: extends behind during stance, flexes forward during swing
    hip = 18.0 * math.sin(2.0 * math.pi * phase - math.pi / 2.0) + 8.0
    # Knee: peaks mid-swing, nearly straight at heel strike
    knee = 55.0 * max(0.0, math.sin(2.0 * math.pi * phase)) + 5.0
    # Ankle: dorsiflex in swing, plantarflex near push-off
    ankle = 12.0 * math.sin(2.0 * math.pi * phase + math.pi / 3.0) - 5.0
    return hip, knee, ankle


def _forward_kinematics_leg(
    side_sign: float,
    hip_flex: float,
    knee_flex: float,
    ankle_flex: float,
    *,
    pelvis: Vec3,
    thigh: float = 0.42,
    shank: float = 0.43,
    foot: float = 0.12,
) -> dict[str, Vec3]:
    """Simple sagittal-plane FK for one leg (degrees, Y-up, Z-forward)."""
    hr = math.radians(hip_flex)
    kr = math.radians(knee_flex)
    ar = math.radians(ankle_flex)

    hip_x = pelvis.x + side_sign * 0.12
    hip = Vec3(hip_x, pelvis.y, pelvis.z)

    knee_y = hip.y - thigh * math.cos(hr)
    knee_z = hip.z + thigh * math.sin(hr)
    knee = Vec3(hip_x, knee_y, knee_z)

    shank_angle = hr - kr
    ankle_y = knee.y - shank * math.cos(shank_angle)
    ankle_z = knee.z + shank * math.sin(shank_angle)
    ankle = Vec3(hip_x, ankle_y, ankle_z)

    foot_angle = shank_angle - ar
    foot_y = ankle.y - foot * math.cos(foot_angle)
    foot_z = ankle.z + foot * math.sin(foot_angle)
    foot_pt = Vec3(hip_x, foot_y - 0.03, foot_z + 0.04)
    heel = Vec3(hip_x, ankle.y - 0.02, ankle.z)
    toe = Vec3(hip_x, foot_pt.y, foot_pt.z + 0.04)

    prefix = "left" if side_sign < 0 else "right"
    return {
        f"{prefix}_hip": hip,
        f"{prefix}_knee": knee,
        f"{prefix}_ankle": ankle,
        f"{prefix}_heel": heel,
        f"{prefix}_toe": toe,
        f"{prefix}_foot": foot_pt,
    }


def _arm_positions(
    side_sign: float,
    shoulder_flex: float,
    elbow_flex: float,
    *,
    spine: Vec3,
    upper_arm: float = 0.28,
    forearm: float = 0.26,
) -> dict[str, Vec3]:
    sr = math.radians(shoulder_flex)
    er = math.radians(elbow_flex)

    sh_x = spine.x + side_sign * 0.18
    shoulder = Vec3(sh_x, spine.y + 0.06, spine.z)

    elbow_y = shoulder.y - upper_arm * math.cos(sr)
    elbow_z = shoulder.z + upper_arm * math.sin(sr)
    elbow = Vec3(sh_x, elbow_y, elbow_z)

    total = sr - er
    wrist_y = elbow.y - forearm * math.cos(total)
    wrist_z = elbow.z + forearm * math.sin(total)
    wrist = Vec3(sh_x + side_sign * 0.02, wrist_y, wrist_z)

    prefix = "left" if side_sign < 0 else "right"
    return {
        f"{prefix}_shoulder": shoulder,
        f"{prefix}_elbow": elbow,
        f"{prefix}_wrist": wrist,
    }


def generate_mock_gait(
    config: MockGaitConfig | None = None,
    *,
    source_label: str = "mock://synthetic_walk",
) -> GaitMotionRecording:
    """
    Build a full ``GaitMotionRecording`` simulating level-ground walking.

    The pelvis translates forward (Z) with slight vertical bob. Left/right legs
    are half a cycle out of phase. Arm swing counter-phases the legs.
    """
    cfg = config or MockGaitConfig()
    frame_count = max(2, int(cfg.duration_s * cfg.fps))
    snapshots: list[SkeletonSnapshot] = []

    prev_positions: dict[str, Vec3] = {}

    for i in range(frame_count):
        t = i / cfg.fps
        cycle = (t * cfg.cadence_hz) % 1.0
        left_phase = cycle
        right_phase = (cycle + 0.5) % 1.0

        # Pelvis trajectory
        forward = (t / cfg.duration_s) * cfg.stride_length_m * 2.0
        bob = 0.025 * math.sin(4.0 * math.pi * cycle)
        pelvis = Vec3(0.0, cfg.pelvis_height_m + bob, forward)

        l_hip, l_knee, l_ankle = _leg_angles(left_phase)
        r_hip, r_knee, r_ankle = _leg_angles(right_phase)

        # Counter-swing arms
        l_shoulder = cfg.arm_swing_deg * math.sin(2.0 * math.pi * right_phase)
        r_shoulder = cfg.arm_swing_deg * math.sin(2.0 * math.pi * left_phase)
        l_elbow = 28.0 + 8.0 * abs(math.sin(2.0 * math.pi * left_phase))
        r_elbow = 28.0 + 8.0 * abs(math.sin(2.0 * math.pi * right_phase))

        spine = Vec3(pelvis.x, pelvis.y + 0.22, pelvis.z - 0.02)
        neck = Vec3(spine.x, spine.y + 0.18, spine.z + 0.01)
        head = Vec3(neck.x, neck.y + 0.14, neck.z + 0.02)

        positions: dict[str, Vec3] = {
            ROOT_JOINT_ID: pelvis,
            "spine": spine,
            "neck": neck,
            "head": head,
        }
        positions.update(_forward_kinematics_leg(-1.0, l_hip, l_knee, l_ankle, pelvis=pelvis))
        positions.update(_forward_kinematics_leg(1.0, r_hip, r_knee, r_ankle, pelvis=pelvis))
        positions.update(_arm_positions(-1.0, l_shoulder, l_elbow, spine=spine))
        positions.update(_arm_positions(1.0, r_shoulder, r_elbow, spine=spine))

        dt = 1.0 / max(cfg.fps, 1e-6)
        joints: dict[str, JointSample] = {}
        for joint_id in JOINT_IDS:
            pos = positions.get(joint_id)
            if pos is None:
                continue
            vel_vec: Vec3 | None = None
            vel_scalar: float | None = None
            if joint_id in prev_positions:
                prev = prev_positions[joint_id]
                vx = (pos.x - prev.x) / dt
                vy = (pos.y - prev.y) / dt
                vz = (pos.z - prev.z) / dt
                vel_vec = Vec3(vx, vy, vz)
                vel_scalar = math.sqrt(vx * vx + vy * vy + vz * vz)

            angle = None
            if joint_id == "left_knee":
                angle = l_knee
            elif joint_id == "right_knee":
                angle = r_knee
            elif joint_id == "left_hip":
                angle = l_hip
            elif joint_id == "right_hip":
                angle = r_hip
            elif joint_id == "left_ankle":
                angle = l_ankle
            elif joint_id == "right_ankle":
                angle = r_ankle

            joints[joint_id] = JointSample(
                joint_id=joint_id,
                position=pos,
                parent_id=JOINT_PARENTS.get(joint_id),
                angle_deg=angle,
                velocity=vel_scalar,
                velocity_vector=vel_vec,
            )

        joints[ROOT_JOINT_ID] = JointSample(
            joint_id=ROOT_JOINT_ID,
            position=pelvis,
            parent_id=None,
            velocity=(
                math.sqrt(
                    ((pelvis.x - prev_positions[ROOT_JOINT_ID].x) / dt) ** 2
                    + ((pelvis.z - prev_positions[ROOT_JOINT_ID].z) / dt) ** 2
                )
                if ROOT_JOINT_ID in prev_positions
                else None
            ),
            velocity_vector=(
                Vec3(
                    (pelvis.x - prev_positions[ROOT_JOINT_ID].x) / dt,
                    (pelvis.y - prev_positions[ROOT_JOINT_ID].y) / dt,
                    (pelvis.z - prev_positions[ROOT_JOINT_ID].z) / dt,
                )
                if ROOT_JOINT_ID in prev_positions
                else None
            ),
        )

        dof_values = {
            "head_neck": 8.0 * math.sin(2.0 * math.pi * cycle),
            "neck": 4.0 * math.sin(2.0 * math.pi * cycle + 0.2),
            "spine_flexion": 6.0 * math.sin(2.0 * math.pi * cycle + 0.1),
            "left_shoulder_flexion": l_shoulder,
            "right_shoulder_flexion": r_shoulder,
            "left_elbow_flexion": l_elbow,
            "right_elbow_flexion": r_elbow,
            "left_hip_flexion": l_hip,
            "right_hip_flexion": r_hip,
            "left_knee_flexion": l_knee,
            "right_knee_flexion": r_knee,
            "left_ankle_flexion": l_ankle,
            "right_ankle_flexion": r_ankle,
        }

        dofs: dict[str, DofSample] = {}

        for dof_id in DOF_IDS:
            angle = dof_values.get(dof_id)
            vel_dps: float | None = None
            if i > 0 and snapshots:
                prev_dof = snapshots[-1].dofs.get(dof_id)
                if prev_dof and prev_dof.angle_deg is not None and angle is not None:
                    vel_dps = (angle - prev_dof.angle_deg) / dt
            dofs[dof_id] = DofSample(
                dof_id=dof_id,
                angle_deg=angle,
                velocity_deg_s=vel_dps,
                joint_id=DOF_JOINT.get(dof_id),
            )

        snapshots.append(
            SkeletonSnapshot(
                frame_index=i,
                time_s=t,
                joints=joints,
                dofs=dofs,
                metadata={"gait_phase": left_phase, "mock": True},
            )
        )
        prev_positions = dict(positions)

    recording = GaitMotionRecording(
        source=source_label,
        fps=cfg.fps,
        snapshots=snapshots,
        source_kind="mock",
        coordinate_system=DEFAULT_COORDINATE_SYSTEM,
        metadata={
            "generator": "stablewalk.data.mock_gait",
            "cadence_hz": cfg.cadence_hz,
            "stride_length_m": cfg.stride_length_m,
        },
    )
    recording.build_time_series()
    return recording
