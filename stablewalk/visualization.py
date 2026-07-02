"""
Step 5: 2D pose visualization and playback.

Shows video frames with skeleton overlay. Only navigates frames with detected poses.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from stablewalk.gait_dof import GAIT_ANGLE_FIELDS, GAIT_DOF_LANDMARKS
from stablewalk.models.pose_data import JointAngles, Keypoint, PoseFrame, PoseSequence
from stablewalk.pose_estimation import LANDMARK_NAMES, PoseLandmarksConnections

logger = logging.getLogger(__name__)

MIN_VISIBILITY = 0.5
DISPLAY_MAX_WIDTH = 1280

def _parse_keypoints(raw: object) -> list[Keypoint]:
    if isinstance(raw, dict):
        keypoints: list[Keypoint] = []
        for name, vals in raw.items():
            if isinstance(vals, list) and len(vals) >= 2:
                keypoints.append(
                    Keypoint(
                        name=name,
                        x=float(vals[0]),
                        y=float(vals[1]),
                        z=float(vals[2]) if len(vals) > 2 else 0.0,
                        visibility=1.0,
                    )
                )
            elif isinstance(vals, dict) and "x" in vals:
                keypoints.append(
                    Keypoint(
                        name=name,
                        x=float(vals["x"]),
                        y=float(vals["y"]),
                        z=float(vals.get("z", 0.0)),
                        visibility=float(vals.get("visibility", 1.0)),
                    )
                )
        return keypoints
    if isinstance(raw, list):
        return [Keypoint(**kp) for kp in raw]
    return []


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
            velocity_scalar = {k: float(v.get("speed", 0.0)) for k, v in velocities.items()}
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
        ts_ms = int(fd.get("timestamp_ms", round(ts_s * 1000)))
        keypoints = _parse_keypoints(fd.get("keypoints", []))
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
                positions_xy=fd.get("keypoints") if isinstance(fd.get("keypoints"), dict) else {},
                positions_normalized=fd.get("positions_normalized") or {},
                skeleton_3d=fd.get("skeleton_3d") or {},
                gait_phase=fd.get("gait_phase") or {},
                gait_events=fd.get("gait_events") or [],
                detected=fd.get("detected", False),
                skipped=fd.get("skipped", False),
            )
        )

    return PoseSequence(
        source_video=data.get("source_video", ""),
        fps=fps,
        frames=frames,
        gait_events_timeline=data.get("gait_events_timeline") or [],
    )


def detected_frame_indices(sequence: PoseSequence) -> list[int]:
    return [i for i, f in enumerate(sequence.frames) if f.detected and f.keypoints]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class SkeletonVisualizer:
    """2D-only interactive player — only frames with detected poses."""

    def __init__(self, sequence: PoseSequence) -> None:
        self.sequence = sequence
        self.fps = max(sequence.fps, 1.0)
        self.pose_indices = detected_frame_indices(sequence)

        if not self.pose_indices:
            raise ValueError(
                "No frames with detected poses. Run: python main.py --url --view"
            )

        self.current_list_pos = 0
        self.playing = False

        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("StableWalk — 2D Gait Visualization")
        self.fig.patch.set_facecolor("#1a1a2e")

        # Main video axes (leave room for controls at bottom)
        self.ax_video = self.fig.add_axes([0.05, 0.22, 0.9, 0.73])
        self.ax_video.set_facecolor("#16213e")
        self.ax_video.set_title("Video + 2D skeleton", color="white", fontsize=12)
        self.ax_video.axis("off")

        self._init_artists()
        self._build_controls()
        self._show_pose_at(0)

    def _init_artists(self) -> None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        self.img_artist = self.ax_video.imshow(
            placeholder,
            extent=(0, 1, 1, 0),
            aspect="auto",
        )
        self.skeleton_lines = [
            self.ax_video.plot([], [], c="#00ff88", lw=3, solid_capstyle="round")[0]
            for _ in PoseLandmarksConnections.POSE_LANDMARKS
        ]
        self.skeleton_points = self.ax_video.plot(
            [], [], "o", color="#ff4757", ms=6, zorder=5
        )[0]
        self.dof_points = self.ax_video.plot(
            [], [], "o", color="#ffd32a", ms=12, markeredgecolor="white", markeredgewidth=1.5, zorder=6
        )[0]

        self.status_text = self.fig.text(
            0.5,
            0.14,
            "",
            ha="center",
            color="#e0e0e0",
            fontsize=10,
        )

    def _build_controls(self) -> None:
        n = max(len(self.pose_indices) - 1, 1)
        ax_slider = self.fig.add_axes([0.12, 0.10, 0.76, 0.03])
        self.slider = Slider(
            ax_slider,
            "Pose frame",
            0,
            n,
            valinit=0,
            valstep=1,
            color="#00ff88",
        )
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")

        ax_prev = self.fig.add_axes([0.22, 0.02, 0.14, 0.06])
        ax_play = self.fig.add_axes([0.40, 0.02, 0.14, 0.06])
        ax_next = self.fig.add_axes([0.58, 0.02, 0.14, 0.06])

        self.btn_prev = Button(ax_prev, "Prev", color="#0f3460", hovercolor="#1a508b")
        self.btn_play = Button(ax_play, "Play", color="#0f3460", hovercolor="#1a508b")
        self.btn_next = Button(ax_next, "Next", color="#0f3460", hovercolor="#1a508b")
        for btn in (self.btn_prev, self.btn_play, self.btn_next):
            btn.label.set_color("white")

        self.slider.on_changed(self._on_slider)
        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_play.on_clicked(self._toggle_play)
        self.btn_next.on_clicked(lambda _e: self._step(1))

        self.timer = self.fig.canvas.new_timer(interval=int(1000 / self.fps))
        self.timer.add_callback(self._on_timer)

    def _on_slider(self, val: float) -> None:
        pos = int(val)
        if pos != self.current_list_pos:
            self._show_pose_at(pos)

    def _on_timer(self) -> None:
        if self.playing:
            self._step(1)

    def _toggle_play(self, _event: object) -> None:
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _step(self, delta: int) -> None:
        self.current_list_pos = (self.current_list_pos + delta) % len(self.pose_indices)
        self.slider.set_val(self.current_list_pos)

    def _show_pose_at(self, list_pos: int) -> None:
        self.current_list_pos = list_pos
        frame_idx = self.pose_indices[list_pos]
        frame = self.sequence.frames[frame_idx]

        image = self._load_image(frame.image_path)
        self.img_artist.set_data(image)
        self.ax_video.set_xlim(0, 1)
        self.ax_video.set_ylim(1, 0)
        self._draw_skeleton(frame.keypoints)

        angles = frame.joint_angles
        gait_dof = angles.gait_dof_count() if angles else 0
        vel_count = len(frame.velocities)

        self.status_text.set_text(
            f"Pose {list_pos + 1}/{len(self.pose_indices)}  |  "
            f"Video frame {frame_idx + 1}  |  "
            f"Gait DOF: {gait_dof}/12+  |  Joint velocities: {vel_count}"
        )
        self.fig.canvas.draw_idle()

    def _draw_skeleton(self, keypoints: list[Keypoint]) -> None:
        by_name = {kp.name: kp for kp in keypoints}
        for line, conn in zip(self.skeleton_lines, PoseLandmarksConnections.POSE_LANDMARKS):
            if conn.start >= len(LANDMARK_NAMES) or conn.end >= len(LANDMARK_NAMES):
                line.set_data([], [])
                continue
            a = by_name.get(LANDMARK_NAMES[conn.start])
            b = by_name.get(LANDMARK_NAMES[conn.end])
            if not a or not b or a.visibility < MIN_VISIBILITY or b.visibility < MIN_VISIBILITY:
                line.set_data([], [])
                continue
            line.set_data([_clamp01(a.x), _clamp01(b.x)], [_clamp01(a.y), _clamp01(b.y)])

        xs, ys, dx, dy = [], [], [], []
        for kp in keypoints:
            if kp.visibility >= MIN_VISIBILITY and kp.name != "mid_hip":
                xs.append(_clamp01(kp.x))
                ys.append(_clamp01(kp.y))
                if kp.name in GAIT_DOF_LANDMARKS:
                    dx.append(_clamp01(kp.x))
                    dy.append(_clamp01(kp.y))
        self.skeleton_points.set_data(xs, ys)
        self.dof_points.set_data(dx, dy)

    @staticmethod
    def _load_image(path: str) -> np.ndarray:
        resolved = Path(path)
        if not resolved.is_file():
            return np.zeros((480, 640, 3), dtype=np.uint8)
        img = cv2.imread(str(resolved))
        if img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if w > DISPLAY_MAX_WIDTH:
            scale = DISPLAY_MAX_WIDTH / w
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return rgb

    def show(self) -> None:
        plt.show()


def launch_viewer(poses_path: str | Path) -> None:
    path = Path(poses_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pose file not found: {path}")
    sequence = load_pose_sequence(path)
    detected = detected_frame_indices(sequence)
    logger.info(
        "Opening 2D viewer: %d pose frames (of %d total)",
        len(detected),
        len(sequence.frames),
    )
    SkeletonVisualizer(sequence).show()
