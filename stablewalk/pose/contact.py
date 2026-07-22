"""
Foot–ground contact detection from 2D ankle/foot keypoints.

Detects when each foot is in contact using vertical position and velocity
in image coordinates (see ``ContactDetector`` docstring for assumptions).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from stablewalk.models.pose_data import PoseFrame, PoseSequence

SIDES = ("left", "right")
ANKLE_BY_SIDE = {"left": "left_ankle", "right": "right_ankle"}
FOOT_BY_SIDE = {"left": "left_foot_index", "right": "right_foot_index"}

# Image coordinates: y increases downward → larger y ≈ closer to ground plane in frame.
DETECTION_ASSUMPTIONS = """
Foot contact detection assumptions (monocular 2D pose):
  1. Camera is roughly side or oblique view; the walking surface appears horizontal in the image.
  2. In normalized image coordinates, y increases downward; the foot closest to the ground has
     the largest y among the gait cycle (not the smallest).
  3. Ground height per foot is approximated as a high percentile (default 85th) of foot/ankle y
     over the clip — occasional toe-off frames do not set the plane too high.
  4. Contact frame: foot y is within a band of that ground level AND vertical speed |dy/dt| is
     below a threshold (foot barely moving vertically while loaded).
  5. Ankle and foot_index landmarks are fused: the lower point (max y) is used when both visible.
  6. This is a kinematic proxy, not a force-plate or pressure insole — quick swings and soft knees
     can mis-label contact. Results are best for steady treadmill or flat-floor walking.
""".strip()


@dataclass
class PhaseDurations:
    """Mean contiguous stance/swing segment lengths per leg (seconds)."""

    stance_s: float
    swing_s: float
    stance_segments: list[float] = field(default_factory=list)
    swing_segments: list[float] = field(default_factory=list)


@dataclass
class ContactDetectionResult:
    """Bilateral foot contact timeline and phase durations."""

    left_foot_contacts: list[int] = field(default_factory=list)
    right_foot_contacts: list[int] = field(default_factory=list)
    left_foot_contact_times_s: list[float] = field(default_factory=list)
    right_foot_contact_times_s: list[float] = field(default_factory=list)
    left_stance: PhaseDurations = field(default_factory=lambda: PhaseDurations(0.0, 0.0))
    right_stance: PhaseDurations = field(default_factory=lambda: PhaseDurations(0.0, 0.0))
    per_frame_contact: dict[int, dict[str, bool]] = field(default_factory=dict)
    frame_times_s: dict[int, float] = field(default_factory=dict)
    fps: float = 30.0
    assumptions: str = DETECTION_ASSUMPTIONS

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["left_stance"] = asdict(self.left_stance)
        d["right_stance"] = asdict(self.right_stance)
        return d


class ContactDetector:
    """
    Detect foot–ground contact from ankle / foot_index keypoints.

    **How contact is detected**

    For each leg we track a single foot height ``y(t)`` in normalized image coordinates
    (prefer ``foot_index``, else ``ankle``; if both exist, use ``max(y)`` as the lowest
    point toward the ground).

    1. **Ground proximity** — Estimate ground level as the ``ground_percentile`` of all
       ``y`` samples (default 85%). A frame is *near ground* when
       ``y >= ground_level - proximity_fraction * (y_max - y_min)``.

    2. **Minimal vertical movement** — Smooth ``y`` and compute ``dy/dt``. Contact
       requires ``|dy/dt| <= velocity_threshold`` (normalized units per second).

    3. **Contact mask** — Both conditions must hold. Optional ``min_contact_frames``
       merges brief gaps; ``contact_onset`` lists rising edges (touch-down frames).

    **Stance / swing**

    - Stance = frames in contact.
    - Swing = frames not in contact.
    - Durations = mean length of contiguous stance/swing runs in seconds.

    Outputs ``left_foot_contacts`` / ``right_foot_contacts`` as **frame indices** at
    touch-down, plus parallel ``*_contact_times_s`` lists.
    """

    def __init__(
        self,
        *,
        min_visibility: float = 0.4,
        ground_percentile: float = 85.0,
        proximity_fraction: float = 0.12,
        velocity_threshold: float = 0.35,
        smooth_window: int = 5,
        min_contact_frames: int = 2,
    ) -> None:
        self.min_visibility = min_visibility
        self.ground_percentile = ground_percentile
        self.proximity_fraction = proximity_fraction
        self.velocity_threshold = velocity_threshold
        self.smooth_window = max(3, smooth_window | 1)  # odd
        self.min_contact_frames = max(1, min_contact_frames)

    def detect(self, sequence: PoseSequence) -> ContactDetectionResult:
        """Run contact detection on all detected frames in ``sequence``."""
        fps = max(sequence.fps, 1e-6)
        detected = [f for f in sequence.frames if f.detected and f.keypoints]

        left_frames, left_times, left_mask = self._side_contact_mask(detected, "left", fps)
        right_frames, right_times, right_mask = self._side_contact_mask(detected, "right", fps)

        per_frame: dict[int, dict[str, bool]] = {}
        time_by_frame = {f.frame_index: f.timestamp_s for f in detected}
        for f in detected:
            per_frame[f.frame_index] = {"left": False, "right": False}
        for fi, m in zip(left_frames, left_mask):
            per_frame.setdefault(fi, {"left": False, "right": False})["left"] = bool(m)
        for fi, m in zip(right_frames, right_mask):
            per_frame.setdefault(fi, {"left": False, "right": False})["right"] = bool(m)

        left_onsets = self._contact_onsets(left_frames, left_mask)
        right_onsets = self._contact_onsets(right_frames, right_mask)

        return ContactDetectionResult(
            left_foot_contacts=left_onsets,
            right_foot_contacts=right_onsets,
            left_foot_contact_times_s=[time_by_frame[i] for i in left_onsets if i in time_by_frame],
            right_foot_contact_times_s=[time_by_frame[i] for i in right_onsets if i in time_by_frame],
            left_stance=self._phase_durations(left_frames, left_mask, fps),
            right_stance=self._phase_durations(right_frames, right_mask, fps),
            per_frame_contact=per_frame,
            frame_times_s=time_by_frame,
            fps=fps,
        )

    def detect_frames(self, frames: list[PoseFrame], fps: float) -> ContactDetectionResult:
        """Detect on a frame list (wraps into a temporary sequence)."""
        seq = PoseSequence(source_video="", fps=fps, frames=frames)
        return self.detect(seq)

    def _foot_y(self, frame: PoseFrame, side: str) -> float | None:
        by_name = {kp.name: kp for kp in frame.keypoints}
        candidates: list[float] = []
        for name in (FOOT_BY_SIDE[side], ANKLE_BY_SIDE[side]):
            kp = by_name.get(name)
            if kp is not None and kp.visibility >= self.min_visibility:
                candidates.append(float(kp.y))
        if not candidates:
            return None
        return max(candidates)

    def _side_contact_mask(
        self,
        frames: list[PoseFrame],
        side: str,
        fps: float,
    ) -> tuple[list[int], list[float], np.ndarray]:
        """Return frame indices, timestamps, boolean contact mask."""
        frame_indices: list[int] = []
        times: list[float] = []
        ys: list[float] = []

        for f in frames:
            y = self._foot_y(f, side)
            if y is None:
                continue
            frame_indices.append(f.frame_index)
            times.append(f.timestamp_s)
            ys.append(y)

        n = len(ys)
        if n < 3:
            return frame_indices, times, np.zeros(n, dtype=bool)

        y_arr = np.array(ys, dtype=float)
        t_arr = np.array(times, dtype=float)

        # Smooth height
        w = min(self.smooth_window, n)
        kernel = np.ones(w) / w
        y_smooth = np.convolve(y_arr, kernel, mode="same")

        y_min, y_max = float(np.min(y_smooth)), float(np.max(y_smooth))
        y_span = max(y_max - y_min, 1e-4)
        ground_level = float(np.percentile(y_smooth, self.ground_percentile))

        near_ground = y_smooth >= (ground_level - self.proximity_fraction * y_span)

        # Vertical velocity (normalized coords per second)
        if n >= 2:
            vy = np.gradient(y_smooth, t_arr)
        else:
            vy = np.zeros(n)

        slow_vertical = np.abs(vy) <= self.velocity_threshold

        contact = near_ground & slow_vertical
        contact = self._enforce_min_run_length(contact, self.min_contact_frames)

        return frame_indices, times, contact

    @staticmethod
    def _enforce_min_run_length(mask: np.ndarray, min_len: int) -> np.ndarray:
        """Fill short gaps and remove short contact blips."""
        if mask.size == 0 or min_len <= 1:
            return mask
        out = mask.copy()
        # Merge short False gaps between True runs
        i = 0
        while i < len(out):
            if not out[i]:
                j = i
                while j < len(out) and not out[j]:
                    j += 1
                if j - i < min_len and i > 0 and j < len(out):
                    out[i:j] = True
                i = j
            else:
                i += 1
        return out

    @staticmethod
    def _contact_onsets(frame_indices: list[int], mask: np.ndarray) -> list[int]:
        onsets: list[int] = []
        prev = False
        for fi, m in zip(frame_indices, mask):
            if m and not prev:
                onsets.append(fi)
            prev = bool(m)
        return onsets

    @staticmethod
    def _phase_durations(
        frame_indices: list[int],
        mask: np.ndarray,
        fps: float,
    ) -> PhaseDurations:
        if len(mask) == 0:
            return PhaseDurations(0.0, 0.0)

        stance_segments: list[float] = []
        swing_segments: list[float] = []
        run_start = 0
        run_val = bool(mask[0])

        for i in range(1, len(mask)):
            if bool(mask[i]) != run_val:
                dt = (i - run_start) / fps
                if run_val:
                    stance_segments.append(dt)
                else:
                    swing_segments.append(dt)
                run_start = i
                run_val = bool(mask[i])

        dt = (len(mask) - run_start) / fps
        if run_val:
            stance_segments.append(dt)
        else:
            swing_segments.append(dt)

        mean_stance = sum(stance_segments) / len(stance_segments) if stance_segments else 0.0
        mean_swing = sum(swing_segments) / len(swing_segments) if swing_segments else 0.0

        return PhaseDurations(
            stance_s=mean_stance,
            swing_s=mean_swing,
            stance_segments=stance_segments,
            swing_segments=swing_segments,
        )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_contact_timeline(
        self,
        result: ContactDetectionResult,
        *,
        title: str | None = None,
    ):
        """
        Matplotlib timeline: stance bands and touch-down markers per foot.

        Returns ``(fig, axes)`` with axes[0]=left, axes[1]=right.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(9, 4), sharex=True)

        for ax_side, side, contacts, touch_times in zip(
            axes,
            ("left", "right"),
            (result.left_foot_contacts, result.right_foot_contacts),
            (result.left_foot_contact_times_s, result.right_foot_contact_times_s),
        ):
            frames_sorted = sorted(result.per_frame_contact.keys())
            if not frames_sorted:
                ax_side.set_ylabel(side)
                continue
            t_line = [
                result.frame_times_s.get(fi, fi / result.fps) for fi in frames_sorted
            ]
            contact_line = [
                1.0 if result.per_frame_contact[fi].get(side, False) else 0.0
                for fi in frames_sorted
            ]
            from stablewalk.ui.colors import SIDE_LEFT, SIDE_RIGHT

            color = SIDE_LEFT if side == "left" else SIDE_RIGHT
            ax_side.fill_between(
                t_line,
                0,
                contact_line,
                alpha=0.3,
                color=color,
                label="stance (contact)",
            )
            for t in touch_times:
                ax_side.axvline(
                    t,
                    color="#1e40af" if side == "left" else "#991b1b",
                    alpha=0.75,
                    lw=1.2,
                    linestyle="--",
                )
            ax_side.set_ylim(-0.05, 1.15)
            ax_side.set_yticks([0, 1])
            ax_side.set_yticklabels(["swing", "contact"])
            ax_side.set_ylabel(side.capitalize())
            ax_side.legend(loc="upper right", fontsize=8)
            ax_side.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(title or "Foot contact timeline", fontsize=11)
        fig.tight_layout()
        return fig, axes

    @staticmethod
    def highlight_contact_on_frame(
        bgr: np.ndarray,
        keypoints: list,
        contact: dict[str, bool],
        *,
        min_visibility: float = 0.4,
    ) -> np.ndarray:
        """
        Draw colored rings on ankles/feet when ``contact['left'/'right']`` is True.
        """
        import cv2

        out = bgr.copy()
        h, w = out.shape[:2]
        colors = {"left": (255, 180, 50), "right": (50, 80, 255)}  # BGR
        for side in SIDES:
            if not contact.get(side, False):
                continue
            by_name = {kp.name: kp for kp in keypoints}
            for name in (FOOT_BY_SIDE[side], ANKLE_BY_SIDE[side]):
                kp = by_name.get(name)
                if kp is None or kp.visibility < min_visibility:
                    continue
                cx, cy = int(kp.x * w), int(kp.y * h)
                cv2.circle(out, (cx, cy), 14, colors[side], 3, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
        return out


def attach_foot_contact_to_frames(
    sequence: PoseSequence,
    result: ContactDetectionResult,
) -> None:
    """Write per-frame contact flags onto ``PoseFrame.foot_contact`` and stance phases."""
    for frame in sequence.frames:
        flags = result.per_frame_contact.get(frame.frame_index, {})
        frame.foot_contact = {
            "left": flags.get("left", False),
            "right": flags.get("right", False),
        }
        if flags:
            frame.gait_phase = {
                "left": "stance" if flags.get("left") else "swing",
                "right": "stance" if flags.get("right") else "swing",
            }
