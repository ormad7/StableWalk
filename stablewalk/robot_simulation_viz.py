"""
Matplotlib UI for robotic walking simulation playback.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.widgets import Button, Slider

from stablewalk.robotic_model import ROBOT_SEGMENT_NAMES
from stablewalk.visualization import load_pose_sequence
from stablewalk.walk_simulator import WalkSimulation, WalkSimulator

logger = logging.getLogger(__name__)

from stablewalk.ui.colors import COM, CRITICAL, SIDE_LEFT

BG = "#1a1a2e"
PANEL = "#16213e"
BONE = COM
JOINT = "#ffd32a"
ACCENT = SIDE_LEFT
TEXT = "#e8e8e8"
MUTED = "#8892a0"


def _draw_robot(ax, geometry, *, clear: bool = True) -> None:
    if clear:
        ax.cla()

    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(BG)
    pts = geometry.points

    for a, b in ROBOT_SEGMENT_NAMES:
        pa, pb = pts.get(a), pts.get(b)
        if not pa or not pb:
            continue
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=BONE,
            linewidth=2.5,
            solid_capstyle="round",
        )

    if pts:
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        zs = [p[2] for p in pts.values()]
        ax.scatter(xs, ys, zs, c=JOINT, s=35, edgecolors="white", linewidths=0.6)

    if "pelvis" in pts:
        p = pts["pelvis"]
        ax.scatter([p[0]], [p[1]], [p[2]], c="#ff4757", s=55, depthshade=True)

    ax.set_xlabel("X (left)", color=MUTED, fontsize=8)
    ax.set_ylabel("Y (forward)", color=MUTED, fontsize=8)
    ax.set_zlabel("Z (up)", color=MUTED, fontsize=8)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.view_init(elev=20, azim=-65)
    ax.grid(True, color="#2a3a5c", alpha=0.4)

    margin = 0.45
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-0.05, 0.55)
    ax.set_title("Robotic walk simulation", color=TEXT, fontsize=10)


class RobotWalkPlayer:
    """Interactive 3D robot animation driven by DOF angles over time."""

    def __init__(self, simulation: WalkSimulation) -> None:
        if not simulation.frames:
            raise ValueError("Simulation has no frames. Process a video with detected poses first.")

        self.simulation = simulation
        self.current = 0
        self.playing = False

        self.fig = plt.figure(figsize=(9, 7))
        self.fig.patch.set_facecolor(BG)
        self.fig.canvas.manager.set_window_title("StableWalk — Robot Walk Simulation")

        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax_info = self.fig.text(0.5, 0.02, "", ha="center", color=TEXT, fontsize=9)

        self._draw_frame(0)

        ax_prev = self.fig.add_axes([0.20, 0.06, 0.12, 0.05])
        ax_play = self.fig.add_axes([0.36, 0.06, 0.12, 0.05])
        ax_next = self.fig.add_axes([0.52, 0.06, 0.12, 0.05])
        ax_slider = self.fig.add_axes([0.12, 0.12, 0.76, 0.03])

        self.btn_prev = Button(ax_prev, "Prev", color=PANEL)
        self.btn_play = Button(ax_play, "Play", color=PANEL)
        self.btn_next = Button(ax_next, "Next", color=PANEL)
        for btn in (self.btn_prev, self.btn_play, self.btn_next):
            btn.label.set_color("white")

        n = max(len(simulation.frames) - 1, 1)
        self.slider = Slider(ax_slider, "Frame", 0, n, valinit=0, valstep=1, color=ACCENT)
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")

        self.btn_prev.on_clicked(lambda _e: self._step(-1))
        self.btn_play.on_clicked(self._toggle_play)
        self.btn_next.on_clicked(lambda _e: self._step(1))
        self.slider.on_changed(self._on_slider)

        interval = int(1000 / max(simulation.fps, 1))
        self.timer = self.fig.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._on_timer)

    def _draw_frame(self, index: int) -> None:
        self.current = index % len(self.simulation.frames)
        sf = self.simulation.frames[self.current]
        _draw_robot(self.ax, sf.geometry, clear=True)

        phase = sf.gait_phase
        events = ", ".join(sf.gait_events) if sf.gait_events else "—"
        self.ax_info.set_text(
            f"t={sf.time_s:.2f}s  |  frame {sf.frame_index}  |  "
            f"L:{phase.get('left', '—')} R:{phase.get('right', '—')}  |  {events}"
        )
        self.fig.canvas.draw_idle()

    def _on_slider(self, val: float) -> None:
        idx = int(val)
        if idx != self.current:
            self._draw_frame(idx)

    def _toggle_play(self, _event: object) -> None:
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        if self.playing:
            self.timer.start()
        else:
            self.timer.stop()

    def _on_timer(self) -> None:
        if self.playing:
            self._step(1)

    def _step(self, delta: int) -> None:
        n = len(self.simulation.frames)
        next_idx = (self.current + delta) % n
        self.slider.set_val(next_idx)

    def show(self) -> None:
        plt.show()


def launch_robot_simulation(poses_path: str | Path) -> None:
    """Load pose JSON and open robot walk simulator window."""
    path = Path(poses_path)
    if not path.is_file():
        raise FileNotFoundError(f"Pose file not found: {path}")

    sequence = load_pose_sequence(path)
    simulation = WalkSimulator().from_pose_sequence(sequence)
    if not simulation.frames:
        raise ValueError("No detected poses with joint angles in this file.")

    logger.info("Robot simulation: %d frames", len(simulation.frames))
    RobotWalkPlayer(simulation).show()
