"""
Step 5: 2D pose visualization and playback.

Shows video frames with skeleton overlay. Only navigates frames with detected poses.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

from stablewalk.pose.dof import GAIT_ANGLE_FIELDS, GAIT_DOF_LANDMARKS
from stablewalk.io.pose_loader import detected_frame_indices, load_pose_sequence
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.pose.estimation import LANDMARK_NAMES, PoseLandmarksConnections

logger = logging.getLogger(__name__)

MIN_VISIBILITY = 0.5
DISPLAY_MAX_WIDTH = 1280

# Backward-compatible re-exports
__all__ = ["detected_frame_indices", "load_pose_sequence", "launch_viewer", "SkeletonVisualizer"]


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
