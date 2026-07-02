"""
Ground reaction force (GRF) estimation from body motion and foot contact.

Primary API: ``GRFAnalyzer`` — vertical forces from CoM/hip acceleration::

    F_z(t) = m * (g + a_com,z)   allocated to feet in contact
    F_norm = F_z / (m * g)       body-weight units (BW)

See ``WHY_APPROXIMATION`` and ``REAL_GRF_REQUIRES`` for scope and limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from stablewalk.io.pose_loader import detected_frame_indices
from stablewalk.models.pose_data import PoseFrame, PoseSequence

logger = logging.getLogger(__name__)

G = 9.81

WHY_APPROXIMATION = """
Why this GRF is an approximation
--------------------------------
StableWalk does not measure force. It **infers** vertical ground reaction from:

1. **CoM / hip height** tracked in video (2D pose, weak depth).
2. **Vertical acceleration** = second derivative of that height (noisy, sensitive to FPS).
3. **Newton's 2nd law (vertical)**: total support force ≈ m·(g + a_z) while the foot is on the ground.
4. **Contact mask** from foot/ankle height heuristics, not force sensors.

So the curves show a **physically motivated proxy** useful for comparing strides and
spotting asymmetry — not instrumented kinetics.
""".strip()

REAL_GRF_REQUIRES = """
What real GRF measurement requires
---------------------------------
- **Force plates** or **instrumented walkways** (gold standard): direct 3D forces
  and center of pressure at hundreds–1000+ Hz.
- **Pressure insoles** or **wearable IMUs** with validated models.
- **Multi-camera 3D motion capture** + **inverse dynamics** with known segment masses,
  joint moments, and foot–floor contact constraints.

Real systems provide:
- Vertical, anterior–posterior, and medial–lateral components
- Accurate timing of load acceptance and push-off
- Calibration to each subject's mass and footwear

StableWalk outputs **vertical estimates only**, split heuristically between feet.
""".strip()

# Legacy strings (kept for existing scripts)
GRF_IMPORTANCE = WHY_APPROXIMATION
GRF_LIMITATIONS = REAL_GRF_REQUIRES


@dataclass
class StanceGRFProfile:
    """One continuous foot contact interval."""

    side: str
    start_frame: int
    end_frame: int
    time_s_start: float
    time_s_end: float
    peak1_bw: float
    peak2_bw: float
    has_double_peak: bool


@dataclass
class GRFTimeSeries:
    """
    Per-frame vertical GRF for both feet.

    ``left_force`` / ``right_force`` alias ``left_force_bw`` / ``right_force_bw``
    (body-weight normalized). Use ``left_force_n`` for Newtons.
    """

    method: str
    body_mass_kg: float
    body_weight_n: float
    fps: float
    frame_indices: np.ndarray
    time_s: np.ndarray
    left_force_n: np.ndarray
    right_force_n: np.ndarray
    left_force_bw: np.ndarray
    right_force_bw: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    com_height_m: np.ndarray
    com_velocity_z: np.ndarray
    com_accel_z: np.ndarray
    double_peak_left: list[StanceGRFProfile] = field(default_factory=list)
    double_peak_right: list[StanceGRFProfile] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    # Back-compat field name
    @property
    def com_accel_z_legacy(self) -> np.ndarray:
        return self.com_accel_z

    @property
    def left_force(self) -> np.ndarray:
        """Normalized vertical GRF (× body weight), left foot."""
        return self.left_force_bw

    @property
    def right_force(self) -> np.ndarray:
        """Normalized vertical GRF (× body weight), right foot."""
        return self.right_force_bw

    @property
    def double_peak_fraction(self) -> float:
        profiles = self.double_peak_left + self.double_peak_right
        if not profiles:
            return 0.0
        return sum(1 for p in profiles if p.has_double_peak) / len(profiles)

    @property
    def mean_peak_bw(self) -> float | None:
        peaks: list[float] = []
        for arr, mask in (
            (self.left_force_bw, self.left_contact),
            (self.right_force_bw, self.right_contact),
        ):
            if mask.any():
                peaks.append(float(np.max(arr[mask])))
        return sum(peaks) / len(peaks) if peaks else None

    def summary(self) -> str:
        peak = f"{self.mean_peak_bw:.2f} BW" if self.mean_peak_bw else "n/a"
        return (
            f"Estimated GRF ({self.method})\n"
            f"  F = m * (g + a_com,z), contact feet only\n"
            f"  Body weight = {self.body_weight_n:.0f} N ({self.body_mass_kg:.0f} kg)\n"
            f"  Max normalized peak ~ {peak}\n"
            f"  Stances with double-peak: {self.double_peak_fraction:.0%}"
        )


@dataclass
class ForceAnalysisReport:
    """Legacy report wrapper around ``GRFTimeSeries``."""

    method: str
    body_mass_kg: float
    body_weight_n: float
    mean_peak_vertical_n: float | None
    profiles: list
    double_peak_fraction: float
    notes: list[str] = field(default_factory=list)
    series: GRFTimeSeries | None = None

    @property
    def summary(self) -> str:
        if self.series:
            return self.series.summary()
        return "No GRF data"


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < window:
        return y.copy()
    k = np.ones(window) / window
    return np.convolve(y, k, mode="same")


def _vertical_kinematics(height_m: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertical velocity and acceleration (m/s, m/s²), Z-up."""
    if len(height_m) < 2:
        z = np.zeros_like(height_m)
        return z, z
    dt = 1.0 / max(fps, 1e-6)
    vel = np.gradient(height_m, dt)
    acc = np.gradient(vel, dt)
    return _smooth(vel), _smooth(acc)


def _allocate_grf_to_feet(
    f_total: float,
    left_on: bool,
    right_on: bool,
) -> tuple[float, float]:
    if not left_on and not right_on:
        return 0.0, 0.0
    if left_on and right_on:
        half = f_total * 0.5
        return half, half
    if left_on:
        return f_total, 0.0
    return 0.0, f_total


def _stance_intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals: list[tuple[int, int]] = []
    in_stance = False
    start = 0
    for i, on in enumerate(mask):
        if on and not in_stance:
            start = i
            in_stance = True
        elif not on and in_stance:
            intervals.append((start, i - 1))
            in_stance = False
    if in_stance:
        intervals.append((start, len(mask) - 1))
    return [(a, b) for a, b in intervals if b >= a]


def _detect_double_peak(
    values: np.ndarray,
    *,
    min_peak_bw: float = 0.35,
) -> tuple[bool, float, float]:
    if len(values) < 5:
        m = float(np.max(values)) if len(values) else 0.0
        return False, m, 0.0
    peaks: list[tuple[int, float]] = []
    for i in range(1, len(values) - 1):
        if (
            values[i] > values[i - 1]
            and values[i] > values[i + 1]
            and values[i] >= min_peak_bw
        ):
            peaks.append((i, float(values[i])))
    if len(peaks) < 2:
        m = float(np.max(values))
        return False, m, 0.0
    peaks.sort(key=lambda x: x[1], reverse=True)
    return True, peaks[0][1], peaks[1][1]


def _hip_height_m(frame: PoseFrame, scale_m: float) -> float | None:
    """CoM proxy: hip center height in meters (Z-up from image y)."""
    kp = {k.name: k for k in frame.keypoints}
    ys: list[float] = []
    for name in ("mid_hip", "left_hip", "right_hip"):
        p = kp.get(name)
        if p is not None and p.visibility >= 0.35:
            ys.append(float(p.y))
    if not ys and "left_hip" in kp and "right_hip" in kp:
        lh, rh = kp["left_hip"], kp["right_hip"]
        if lh.visibility >= 0.35 and rh.visibility >= 0.35:
            ys.append((lh.y + rh.y) / 2)
    if not ys:
        return None
    y_mean = sum(ys) / len(ys)
    # Image y down → height up: z ≈ (1 - y) * body scale
    return (1.0 - y_mean) * scale_m


def _contact_masks_from_frames(frames: list[PoseFrame]) -> tuple[np.ndarray, np.ndarray]:
    left: list[bool] = []
    right: list[bool] = []
    for f in frames:
        if f.foot_contact:
            left.append(bool(f.foot_contact.get("left", False)))
            right.append(bool(f.foot_contact.get("right", False)))
        else:
            left.append(f.gait_phase.get("left") == "stance")
            right.append(f.gait_phase.get("right") == "stance")
    return np.array(left, dtype=bool), np.array(right, dtype=bool)


class GRFAnalyzer:
    """
    Estimate vertical ground reaction forces from motion.

    **Algorithm**

    1. **Body mass** — constant (default 70 kg); body weight ``m * 9.81`` N.
    2. **CoM / hip height** — mid-hip (or bilateral hips) lifted to meters using
       skeleton scale; vertical velocity ``v_z`` and acceleration ``a_z`` via
       gradients along time.
    3. **Total vertical force** — while at least one foot is in contact:
       ``F_total = m * (g + a_z)`` (reaction force upward, clipped to ≥ 0).
    4. **Per-foot split** — assign ``F_total`` to left/right only when that foot
       contacts the ground (50/50 during double support).
    5. **Normalize** — ``left_force[t] = F_left / (m*g)`` in body-weight (BW) units.

    **Double-peak detection** — local maxima in normalized stance-phase GRF
    (typical healthy walking shows two peaks per stance).

    Set ``use_physics_com=True`` to use ``WalkingSimulator`` CoM height instead
    of the hip proxy (slower, still approximate).
    """

    def __init__(
        self,
        *,
        body_mass_kg: float = 70.0,
        use_physics_com: bool = False,
        smooth_window: int = 5,
    ) -> None:
        self.body_mass_kg = body_mass_kg
        self.body_weight_n = body_mass_kg * G
        self.use_physics_com = use_physics_com
        self.smooth_window = max(3, smooth_window | 1)

    def analyze(
        self,
        sequence: PoseSequence,
        *,
        physics_result=None,
    ) -> GRFTimeSeries:
        """
        Compute ``left_force[t]`` and ``right_force[t]`` (BW) plus Newton arrays.

        Sequence should be enriched (``foot_contact`` or ``gait_phase`` on frames).
        """
        if physics_result is not None or self.use_physics_com:
            return self._analyze_from_physics(sequence, physics_result)
        return self._analyze_from_pose(sequence)

    def _analyze_from_pose(self, sequence: PoseSequence) -> GRFTimeSeries:
        notes: list[str] = []
        m = self.body_mass_kg
        bw = self.body_weight_n
        fps = max(sequence.fps, 1e-6)

        indices = detected_frame_indices(sequence)
        if len(indices) < 3:
            notes.append("Need at least 3 detected frames for GRF.")
            return self._empty_series(sequence, notes, method="hip_com_vertical")

        frames = [sequence.frames[i] for i in indices]

        from stablewalk.pose.skeleton_3d import sequence_skeleton_scale

        kps = [f.keypoints for f in frames if f.keypoints]
        scale = sequence_skeleton_scale(kps) if kps else 1.7

        valid_frames: list[PoseFrame] = []
        heights: list[float] = []
        times: list[float] = []
        frame_ids: list[int] = []
        for f in frames:
            h = _hip_height_m(f, scale)
            if h is None:
                continue
            valid_frames.append(f)
            heights.append(h)
            times.append(f.timestamp_s)
            frame_ids.append(f.frame_index)

        n = len(heights)
        if n < 3:
            notes.append("Insufficient hip visibility for CoM height.")
            return self._empty_series(sequence, notes, method="hip_com_vertical")

        com_h = np.array(heights, dtype=np.float64)
        time_s = np.array(times, dtype=np.float64)
        frame_indices = np.array(frame_ids, dtype=np.int32)

        left_contact, right_contact = _contact_masks_from_frames(valid_frames)
        if not left_contact.any() and not right_contact.any():
            from stablewalk.pose.contact import ContactDetector, attach_foot_contact_to_frames

            cr = ContactDetector().detect(sequence)
            attach_foot_contact_to_frames(sequence, cr)
            left_contact, right_contact = _contact_masks_from_frames(valid_frames)
            notes.append("Contact from ContactDetector (foot_contact was empty).")

        vel_z, acc_z = _vertical_kinematics(com_h, fps)

        return self._build_series(
            m=m,
            bw=bw,
            fps=fps,
            frame_indices=frame_indices,
            time_s=time_s,
            com_h=com_h,
            vel_z=vel_z,
            acc_z=acc_z,
            left_contact=left_contact,
            right_contact=right_contact,
            method="hip_com_vertical",
            notes=notes,
        )

    def _analyze_from_physics(self, sequence: PoseSequence, physics_result) -> GRFTimeSeries:
        notes: list[str] = []
        m = self.body_mass_kg
        bw = self.body_weight_n

        if physics_result is None:
            try:
                from stablewalk.simulation import WalkingSimulator
                from stablewalk.simulation.config import PhysicsConfig
            except ImportError as exc:
                notes.append(
                    "Physics simulator not available; using pose-based CoM proxy."
                )
                logger.warning("WalkingSimulator unavailable: %s", exc)
                return self._analyze_from_pose(sequence)

            physics_result = WalkingSimulator(
                config=PhysicsConfig(body_mass_kg=m),
            ).run(sequence)

        if not physics_result.frames:
            notes.append("Physics simulation produced no frames.")
            return self._empty_series(sequence, notes, method="physics_com_vertical")

        pf = physics_result.frames
        n = len(pf)
        frame_indices = np.array([f.frame_index for f in pf], dtype=np.int32)
        time_s = np.array([f.time_s for f in pf], dtype=np.float64)
        com_h = np.array([f.com_simulated[2] for f in pf], dtype=np.float64)
        fps = max(physics_result.fps, 1e-6)

        left_contact = np.asarray(physics_result.left_contact_mask, dtype=bool)
        right_contact = np.asarray(physics_result.right_contact_mask, dtype=bool)
        if len(left_contact) != n:
            left_contact = np.array([f.contact.left.in_contact for f in pf], dtype=bool)
            right_contact = np.array([f.contact.right.in_contact for f in pf], dtype=bool)

        vel_z, acc_z = _vertical_kinematics(com_h, fps)

        return self._build_series(
            m=m,
            bw=bw,
            fps=fps,
            frame_indices=frame_indices,
            time_s=time_s,
            com_h=com_h,
            vel_z=vel_z,
            acc_z=acc_z,
            left_contact=left_contact,
            right_contact=right_contact,
            method="physics_com_vertical",
            notes=notes,
        )

    def _build_series(
        self,
        *,
        m: float,
        bw: float,
        fps: float,
        frame_indices: np.ndarray,
        time_s: np.ndarray,
        com_h: np.ndarray,
        vel_z: np.ndarray,
        acc_z: np.ndarray,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        method: str,
        notes: list[str],
    ) -> GRFTimeSeries:
        n = len(time_s)
        left_n = np.zeros(n, dtype=np.float64)
        right_n = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if not left_contact[i] and not right_contact[i]:
                continue
            f_total = m * (float(acc_z[i]) + G)
            f_total = max(0.0, min(f_total, 2.5 * bw))
            fl, fr = _allocate_grf_to_feet(
                f_total,
                bool(left_contact[i]),
                bool(right_contact[i]),
            )
            left_n[i] = fl
            right_n[i] = fr

        left_bw = left_n / bw
        right_bw = right_n / bw

        dp_left, dp_right = self._stance_double_peaks(
            left_contact, right_contact, left_bw, right_bw, frame_indices, time_s
        )

        if not dp_left and not dp_right:
            notes.append("No stance intervals for double-peak analysis.")

        return GRFTimeSeries(
            method=method,
            body_mass_kg=m,
            body_weight_n=bw,
            fps=fps,
            frame_indices=frame_indices,
            time_s=time_s,
            left_force_n=left_n,
            right_force_n=right_n,
            left_force_bw=left_bw,
            right_force_bw=right_bw,
            left_contact=left_contact,
            right_contact=right_contact,
            com_height_m=com_h,
            com_velocity_z=vel_z,
            com_accel_z=acc_z,
            double_peak_left=dp_left,
            double_peak_right=dp_right,
            notes=notes,
        )

    def _stance_double_peaks(
        self,
        left_contact: np.ndarray,
        right_contact: np.ndarray,
        left_bw: np.ndarray,
        right_bw: np.ndarray,
        frame_indices: np.ndarray,
        time_s: np.ndarray,
    ) -> tuple[list[StanceGRFProfile], list[StanceGRFProfile]]:
        dp_left: list[StanceGRFProfile] = []
        dp_right: list[StanceGRFProfile] = []
        for side, mask, bw_arr, out_list in (
            ("left", left_contact, left_bw, dp_left),
            ("right", right_contact, right_bw, dp_right),
        ):
            for a, b in _stance_intervals(mask):
                seg = bw_arr[a : b + 1]
                has_dp, p1, p2 = _detect_double_peak(seg)
                out_list.append(
                    StanceGRFProfile(
                        side=side,
                        start_frame=int(frame_indices[a]),
                        end_frame=int(frame_indices[b]),
                        time_s_start=float(time_s[a]),
                        time_s_end=float(time_s[b]),
                        peak1_bw=p1,
                        peak2_bw=p2,
                        has_double_peak=has_dp,
                    )
                )
        return dp_left, dp_right

    def _empty_series(
        self,
        sequence: PoseSequence,
        notes: list[str],
        *,
        method: str,
    ) -> GRFTimeSeries:
        z = np.zeros(0, dtype=np.float64)
        b = np.zeros(0, dtype=bool)
        return GRFTimeSeries(
            method=method,
            body_mass_kg=self.body_mass_kg,
            body_weight_n=self.body_weight_n,
            fps=sequence.fps,
            frame_indices=np.zeros(0, dtype=np.int32),
            time_s=z,
            left_force_n=z,
            right_force_n=z,
            left_force_bw=z,
            right_force_bw=z,
            left_contact=b,
            right_contact=b,
            com_height_m=z,
            com_velocity_z=z,
            com_accel_z=z,
            notes=notes,
        )

    def plot_grf(
        self,
        series: GRFTimeSeries,
        *,
        ax=None,
        show_bw: bool = True,
        title: str | None = None,
    ):
        """
        Plot left and right vertical GRF vs time on one axes.

        Default: body-weight normalization (BW). ``show_bw=False`` for Newtons.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
        else:
            fig = ax.figure

        t = series.time_s
        if show_bw:
            ax.plot(t, series.left_force, color="#4dabf7", linewidth=2, label="Left foot")
            ax.plot(t, series.right_force, color="#ff6b6b", linewidth=2, label="Right foot")
            ax.set_ylabel("Vertical GRF (body weights, BW)")
            ax.axhline(1.0, color="#888", linestyle="--", linewidth=1, label="1 BW")
        else:
            ax.plot(t, series.left_force_n, color="#4dabf7", linewidth=2, label="Left foot")
            ax.plot(t, series.right_force_n, color="#ff6b6b", linewidth=2, label="Right foot")
            ax.set_ylabel("Vertical GRF (N)")

        ax.set_xlabel("Time (s)")
        ax.set_title(
            title or "Estimated vertical GRF — F = m·(g + a), contact only"
        )
        ax.grid(True, alpha=0.35)
        ax.legend(loc="upper right")
        ax.set_ylim(bottom=0)

        for prof in series.double_peak_left + series.double_peak_right:
            if prof.has_double_peak:
                ax.axvspan(
                    prof.time_s_start,
                    prof.time_s_end,
                    alpha=0.08,
                    color="#4dabf7" if prof.side == "left" else "#ff6b6b",
                )

        fig.tight_layout()
        return fig, ax

    def to_legacy_report(self, series: GRFTimeSeries) -> ForceAnalysisReport:
        profiles = series.double_peak_left + series.double_peak_right
        mean_peak_n = None
        if series.mean_peak_bw:
            mean_peak_n = series.mean_peak_bw * series.body_weight_n
        return ForceAnalysisReport(
            method=series.method,
            body_mass_kg=series.body_mass_kg,
            body_weight_n=series.body_weight_n,
            mean_peak_vertical_n=mean_peak_n,
            profiles=profiles,
            double_peak_fraction=series.double_peak_fraction,
            notes=series.notes,
            series=series,
        )


class ForceAnalyzer(GRFAnalyzer):
    """
    Backward-compatible alias for ``GRFAnalyzer``.

    Defaults to pose-based GRF (no physics engine) so the GUI works without
  ``WalkingSimulator``. Pass ``run_physics=True`` when physics simulation is installed.
    """

    def __init__(
        self,
        *,
        body_mass_kg: float = 70.0,
        run_physics: bool = False,
    ) -> None:
        super().__init__(body_mass_kg=body_mass_kg, use_physics_com=run_physics)


def estimate_vertical_grf(
    sequence: PoseSequence,
    *,
    body_mass_kg: float = 70.0,
    use_physics: bool = False,
) -> ForceAnalysisReport:
    """Entry point: ``GRFAnalyzer`` → legacy report."""
    analyzer = GRFAnalyzer(body_mass_kg=body_mass_kg, use_physics_com=use_physics)
    series = analyzer.analyze(sequence)
    return analyzer.to_legacy_report(series)


def plot_grf_curves(series: GRFTimeSeries, *, show_bw: bool = True):
    return GRFAnalyzer(body_mass_kg=series.body_mass_kg).plot_grf(series, show_bw=show_bw)
