"""
Walking cycle simulator: apply DOF joint angles to a robotic model over time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from stablewalk.models.pose_data import JointAngles, PoseFrame, PoseSequence
from stablewalk.pose.interpolation import (
    interpolate_robot_geometry,
    robot_geometry_from_angles,
    velocity_adjusted_angles,
)
from stablewalk.simulation.robotic_model import (
    RobotConfig,
    RobotGeometry,
    RobotJointState,
    forward_kinematics,
    joint_angles_to_robot_state,
)

logger = logging.getLogger(__name__)


@dataclass
class SimulationFrame:
    """One timestep of the robotic walk simulation."""

    time_s: float
    frame_index: int
    joint_state: RobotJointState
    geometry: RobotGeometry
    gait_phase: dict[str, str] = field(default_factory=dict)
    gait_events: list[str] = field(default_factory=list)
    angles_deg: dict[str, float] = field(default_factory=dict)


@dataclass
class WalkSimulation:
    """Full walking cycle as a sequence of robotic poses."""

    fps: float
    frames: list[SimulationFrame] = field(default_factory=list)
    source_video: str = ""

    @property
    def duration_s(self) -> float:
        if not self.frames:
            return 0.0
        return self.frames[-1].time_s

    @property
    def cycle_length(self) -> int:
        return len(self.frames)


class WalkSimulator:
    """
    Build a robotic walk simulation from pose/DOF data.

    Each detected video frame becomes one mechanical pose; joint angles
    drive revolute joints via forward kinematics.
    """

    def __init__(self, config: RobotConfig | None = None) -> None:
        self.config = config or RobotConfig()

    def from_pose_sequence(self, sequence: PoseSequence) -> WalkSimulation:
        """Convert all detected pose frames into simulation timesteps."""
        sim_frames: list[SimulationFrame] = []
        fps = max(sequence.fps, 1e-6)

        for pf in sequence.frames:
            if not pf.detected or not pf.joint_angles:
                continue

            state = joint_angles_to_robot_state(pf.joint_angles)
            geom = forward_kinematics(state, self.config)
            angles_deg = pf.joint_angles.to_export_dict()

            sim_frames.append(
                SimulationFrame(
                    time_s=pf.frame_index / fps,
                    frame_index=pf.frame_index,
                    joint_state=state,
                    geometry=geom,
                    gait_phase=dict(pf.gait_phase),
                    gait_events=list(pf.gait_events),
                    angles_deg=angles_deg,
                )
            )

        logger.info(
            "Walk simulation: %d robotic poses @ %.1f fps (%.2f s)",
            len(sim_frames),
            fps,
            sim_frames[-1].time_s if sim_frames else 0,
        )
        return WalkSimulation(
            fps=fps,
            frames=sim_frames,
            source_video=sequence.source_video,
        )

    def from_pose_sequence_smoothed(
        self,
        sequence: PoseSequence,
        *,
        substeps: int = 2,
    ) -> WalkSimulation:
        """
        Build simulation with interpolated substeps and velocity-aware angle blending.
        """
        detected = [pf for pf in sequence.frames if pf.detected and pf.joint_angles]
        if not detected:
            return WalkSimulation(fps=sequence.fps, source_video=sequence.source_video)

        fps = max(sequence.fps, 1e-6)
        sim_frames: list[SimulationFrame] = []
        substeps = max(1, substeps)

        for i, curr in enumerate(detected):
            prev = detected[i - 1] if i > 0 else curr
            delta_f = max(curr.frame_index - prev.frame_index, 1) if i > 0 else 1
            dt = delta_f / fps

            for s in range(substeps):
                alpha = 0.0 if i == 0 and s == 0 else s / substeps
                if i == 0 or alpha <= 0:
                    ang = curr.joint_angles
                else:
                    ang = velocity_adjusted_angles(
                        prev.joint_angles,
                        curr.joint_angles,
                        alpha,
                        dt,
                        curr.velocity_scalar,
                    )
                state = joint_angles_to_robot_state(ang)
                geom = forward_kinematics(state, self.config)
                t = (curr.frame_index + alpha * delta_f) / fps
                sim_frames.append(
                    SimulationFrame(
                        time_s=t,
                        frame_index=curr.frame_index,
                        joint_state=state,
                        geometry=geom,
                        gait_phase=dict(curr.gait_phase),
                        gait_events=list(curr.gait_events),
                        angles_deg=ang.to_export_dict() if ang else {},
                    )
                )

        return WalkSimulation(
            fps=fps * substeps,
            frames=sim_frames,
            source_video=sequence.source_video,
        )

    def geometry_at_blend(
        self,
        frame_a: PoseFrame,
        frame_b: PoseFrame,
        alpha: float,
        fps: float,
    ) -> RobotGeometry:
        """Robot geometry between two frames (velocity-smoothed angles)."""
        delta = max(frame_b.frame_index - frame_a.frame_index, 1)
        dt = delta / max(fps, 1e-6)
        angles = velocity_adjusted_angles(
            frame_a.joint_angles,
            frame_b.joint_angles,
            alpha,
            dt,
            frame_b.velocity_scalar,
        )
        return robot_geometry_from_angles(angles, self.config)

    def from_pose_frame(self, frame: PoseFrame, fps: float) -> SimulationFrame:
        """Single-frame simulation step."""
        state = joint_angles_to_robot_state(frame.joint_angles)
        geom = forward_kinematics(state, self.config)
        return SimulationFrame(
            time_s=frame.frame_index / max(fps, 1e-6),
            frame_index=frame.frame_index,
            joint_state=state,
            geometry=geom,
            gait_phase=dict(frame.gait_phase),
            gait_events=list(frame.gait_events),
            angles_deg=frame.joint_angles.to_export_dict() if frame.joint_angles else {},
        )

    @staticmethod
    def interpolate_states(
        a: RobotJointState,
        b: RobotJointState,
        alpha: float,
    ) -> RobotJointState:
        """Linear interpolation between two joint configurations."""
        alpha = max(0.0, min(1.0, alpha))
        result = RobotJointState()
        for name in a.as_dict():
            va = getattr(a, name)
            vb = getattr(b, name)
            setattr(result, name, va + alpha * (vb - va))
        return result

    def frame_at_time(
        self,
        simulation: WalkSimulation,
        time_s: float,
    ) -> SimulationFrame | None:
        """Nearest simulation frame to a given time (for smooth lookup)."""
        if not simulation.frames:
            return None
        best = simulation.frames[0]
        best_dt = abs(best.time_s - time_s)
        for sf in simulation.frames[1:]:
            dt = abs(sf.time_s - time_s)
            if dt < best_dt:
                best, best_dt = sf, dt
        return best
