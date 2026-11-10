"""
Joint Motion Analysis — multi-joint scientific time-series chart.

Overview keeps a 2×3 grid (angle, ω, α, X/Y/Z). Motion Analysis uses tabbed
metrics (Angle / Velocity / Acceleration / Position / Jerk): only the active
graph renders, from one shared cached kinematics bundle and playhead.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import numpy as np
from matplotlib.ticker import MaxNLocator

from stablewalk.ui.colors import TEXT_SECONDARY
from stablewalk.ui.dof_position_table import (
    _ITEM_DOF_ID,
    angle_value_for_item,
    raw_pose_angle_for_item,
)
from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, anchor_joint_for_item, label_for_item
from stablewalk.ui.joint_colors import joint_color
from stablewalk.ui.viewers.chart_hover import ChartHoverPoint
from stablewalk.ui.viewers.chart_playhead import PlayheadState, draw_chart_playhead
from stablewalk.ui.viewers.chart_style import (
    TIMELINE_X_LABEL,
    _AXIS_LABEL_SIZE,
    _TICK_LABEL_SIZE,
    apply_chart_grid,
    apply_chart_panel_style,
    configure_time_axis,
    series_plot_kwargs,
    style_chart_legend,
    style_chart_title,
    unclip_axis_chrome,
    unclip_figure_chrome,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from stablewalk.models.gait_motion import GaitMotionRecording

# Overview 2×3 scientific grid (legacy / Overview Joint Graphs).
JOINT_MOTION_METRICS: tuple[tuple[str, str, str], ...] = (
    ("angle", "Knee Flexion vs Time", "Knee Flexion (°)"),
    ("omega", "Angular Velocity vs Time", "ω (°/s)"),
    ("alpha", "Angular Acceleration vs Time", "α (°/s²)"),
    ("x", "X Position vs Time", "X (m)"),
    ("y", "Y Position vs Time", "Y (m)"),
    ("z", "Z Position vs Time", "Z (m)"),
)

# Full 7-metric stack (legacy / export). Motion Analysis UI uses tabs instead.
MOTION_WORKSPACE_METRICS: tuple[tuple[str, str, str], ...] = (
    ("angle", "Knee Flexion", "Knee Flexion (°)"),
    ("omega", "Angular Velocity", "ω (°/s)"),
    ("alpha", "Angular Acceleration", "α (°/s²)"),
    ("position", "Position", "Position (m)"),
    ("speed", "Velocity", "Speed (m/s)"),
    ("accel", "Acceleration", "Accel (m/s²)"),
    ("jerk", "Linear jerk", "Linear jerk (m/s³)"),
)

MOTION_WORKSPACE_PANEL_COUNT = len(MOTION_WORKSPACE_METRICS)

# Room for tick labels + full Y titles (Angle (°), Velocity, …) without
# stealing more plot area than needed. Do not shrink axis fonts to fit.
# A fraction-only left margin shrinks on a ~500–700 px Motion host and clips
# "Knee Flexion (°)". Keep a minimum inch inset; widen the fraction only when needed.
MOTION_METRIC_CHART_MARGINS = dict(left=0.16, right=0.96, top=0.88, bottom=0.22)
_MOTION_METRIC_YLABEL_PAD = 8.0
_MOTION_METRIC_MIN_LEFT_IN = 1.22
_MOTION_METRIC_MAX_LEFT_FRAC = 0.26


def motion_metric_chart_margins(fig: Figure | None = None) -> dict[str, float]:
    """Left inset large enough for the full Y title at the current figure width."""
    margins = dict(MOTION_METRIC_CHART_MARGINS)
    if fig is None:
        return margins
    try:
        width_in = float(fig.get_figwidth() or 0.0)
        height_px = float(fig.get_figheight() or 0.0) * float(fig.dpi or 100.0)
    except Exception:
        return margins
    if width_in <= 0.2:
        return margins
    needed = _MOTION_METRIC_MIN_LEFT_IN / width_in
    margins["left"] = min(
        _MOTION_METRIC_MAX_LEFT_FRAC,
        max(float(margins["left"]), needed),
    )
    if height_px and height_px < 280:
        # Title + time axis stay inside; legend is in-axes so the plot can expand.
        margins["top"] = 0.82
        margins["bottom"] = 0.28
    elif height_px and height_px < 360:
        margins["top"] = 0.84
        margins["bottom"] = 0.24
    return margins

# Motion Analysis tabs — one graph visible at a time; shared cached bundle.
MOTION_ANALYSIS_TABS: tuple[tuple[str, str, str], ...] = (
    ("angle", "Knee Flexion", "Knee Flexion (°)"),
    ("speed", "Velocity", "Velocity (m/s)"),
    ("accel", "Acceleration", "Acceleration (m/s²)"),
    ("position", "Position", "Position (m)"),
    ("jerk", "Linear jerk", "Linear jerk (m/s³)"),
    ("angular_jerk", "Angular jerk", "Angular jerk (°/s³)"),
)

# GUI DOF item ids whose primary angle is anatomical knee flexion.
_KNEE_MOTION_ITEM_IDS = frozenset({"left_knee", "right_knee"})

# Plot stored interior angle only when explicitly requested (CSV always keeps it).
SHOW_RAW_POSE_ANGLE_DEBUG = False

KNEE_ANGLE_CONVENTION_SHORT = "Knee Flexion (deg)"
KNEE_ANGLE_CONVENTION_TIP = (
    "Primary curve is anatomical knee flexion in degrees.\n\n"
    "θ = internal 3D angle at the knee from Hip–Knee and Ankle–Knee vectors. "
    "Knee flexion = 180° − θ. 0° = full extension; larger = more flexion.\n\n"
    "Missing or low-confidence Hip, Knee, or Ankle landmarks are omitted "
    "(coordinates are never invented). The stored pose interior angle is kept "
    "only as an optional debug series (CSV column raw_pose_angle_deg)."
)
GENERIC_ANGLE_CONVENTION_SHORT = "Joint angle (interior, deg)"
GENERIC_ANGLE_CONVENTION_TIP = (
    "Displayed angle is the geometric interior angle between adjacent "
    "segments at the selected joint (pose landmarks), in degrees, not an "
    "OpenSim model coordinate."
)


def angle_chart_labels(
    item_ids: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Title and Y label for the Angle / Knee Flexion series."""
    ids = {str(i) for i in (item_ids or ()) if i}
    if not ids or ids & _KNEE_MOTION_ITEM_IDS:
        return "Knee Flexion vs Time", "Knee Flexion (°)"
    return "Joint Angle vs Time", "Angle (°)"


def motion_angle_convention(
    item_ids: Iterable[str] | None = None,
) -> tuple[str, str]:
    """Short label and tooltip for the Motion Analysis angle convention."""
    ids = {str(i) for i in (item_ids or ()) if i}
    if not ids or ids & _KNEE_MOTION_ITEM_IDS:
        return KNEE_ANGLE_CONVENTION_SHORT, KNEE_ANGLE_CONVENTION_TIP
    return GENERIC_ANGLE_CONVENTION_SHORT, GENERIC_ANGLE_CONVENTION_TIP


MOTION_ANALYSIS_TAB_IDS: tuple[str, ...] = tuple(m[0] for m in MOTION_ANALYSIS_TABS)

CSV_FIELDNAMES: tuple[str, ...] = (
    "time_s",
    "frame",
    "joint_id",
    "joint_label",
    "angle_deg",
    "raw_pose_angle_deg",
    "angular_velocity_deg_s",
    "angular_acceleration_deg_s2",
    "linear_speed_m_s",
    "linear_accel_m_s2",
    "jerk_m_s3",
    "angular_jerk_deg_s3",
    "x_m",
    "y_m",
    "z_m",
)


def _fps_from_times(times: np.ndarray) -> float:
    dts = np.diff(np.asarray(times, dtype=float))
    dts = dts[np.isfinite(dts) & (dts > 1e-9)]
    return float(1.0 / np.median(dts)) if dts.size else 30.0


def _gapped_xy(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from stablewalk.analysis.joint_path_trace import series_with_gap_breaks

    return series_with_gap_breaks(times, values, fps=_fps_from_times(times))


@dataclass(frozen=True)
class JointMotionSeries:
    """Full-clip kinematics for one selected GUI joint."""

    item_id: str
    label: str
    color: str
    times_s: np.ndarray
    frames: np.ndarray
    angle_deg: np.ndarray
    omega_deg_s: np.ndarray
    alpha_deg_s2: np.ndarray
    speed_m_s: np.ndarray
    accel_m_s2: np.ndarray  # tangential d(speed)/dt; not vector accel magnitude
    jerk_m_s3: np.ndarray  # |d³r/dt³|, m/s^3 (vector jerk magnitude)
    angular_jerk_deg_s3: np.ndarray  # angular jerk, deg/s^3
    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    raw_pose_angle_deg: np.ndarray | None = None

    @property
    def position_mag_m(self) -> np.ndarray:
        mag = np.full_like(self.x_m, np.nan, dtype=float)
        mask = np.isfinite(self.x_m) & np.isfinite(self.y_m) & np.isfinite(self.z_m)
        mag[mask] = np.sqrt(
            self.x_m[mask] ** 2 + self.y_m[mask] ** 2 + self.z_m[mask] ** 2
        )
        return mag

    def values_for(self, metric_id: str) -> np.ndarray:
        return {
            "angle": self.angle_deg,
            "omega": self.omega_deg_s,
            "alpha": self.alpha_deg_s2,
            "speed": self.speed_m_s,
            "accel": self.accel_m_s2,
            "jerk": self.jerk_m_s3,
            "angular_jerk": self.angular_jerk_deg_s3,
            "position": self.position_mag_m,
            "x": self.x_m,
            "y": self.y_m,
            "z": self.z_m,
            "raw_pose_angle": self.raw_pose_angle_deg
            if self.raw_pose_angle_deg is not None
            else np.full_like(self.angle_deg, np.nan, dtype=float),
        }[metric_id]


@dataclass(frozen=True)
class JointMotionBundle:
    """Multi-joint series aligned on a shared time base."""

    series: tuple[JointMotionSeries, ...]
    times_s: np.ndarray

    @property
    def empty(self) -> bool:
        return not self.series or self.times_s.size == 0


@dataclass(frozen=True)
class JointMotionStats:
    """Compact per-joint statistics for the Motion workspace footer."""

    item_id: str
    label: str
    rom_deg: float | None
    mean_angle_deg: float | None
    peak_omega_deg_s: float | None
    peak_alpha_deg_s2: float | None
    peak_speed_m_s: float | None
    peak_accel_m_s2: float | None
    peak_jerk_m_s3: float | None
    peak_angular_jerk_deg_s3: float | None
    travel_m: float | None


@dataclass(frozen=True)
class MetricRibbonStats:
    """Descriptive stats for the ribbon above the active Motion graph."""

    minimum: float | None
    maximum: float | None
    mean: float | None
    rom: float | None
    std: float | None
    current: float | None
    unit: str = ""
    joint_label: str = ""
    metric_id: str = ""

    def as_display(self) -> dict[str, str]:
        """Compact label → text map for the Tk ribbon cells."""

        def _fmt(value: float | None, *, places: int = 2) -> str:
            if value is None or not np.isfinite(float(value)):
                return "—"
            u = self.unit
            if u == "°":
                return f"{float(value):.{places}f}{u}"
            if u:
                return f"{float(value):.{places}f} {u}"
            return f"{float(value):.{places}f}"

        return {
            "minimum": _fmt(self.minimum),
            "maximum": _fmt(self.maximum),
            "mean": _fmt(self.mean),
            "rom": _fmt(self.rom),
            "std": _fmt(self.std),
            "current": _fmt(self.current),
        }


def _unit_from_ylabel(ylabel: str) -> str:
    if "(" in ylabel and ")" in ylabel:
        return ylabel.split("(")[-1].rstrip(")")
    return ""


def _series_values_for_ribbon(
    series: JointMotionSeries, metric_id: str
) -> np.ndarray:
    if metric_id == "position":
        return series.position_mag_m
    return series.values_for(metric_id)


# Aggregate ribbon stats are session-stable; only ``current`` tracks the playhead.
_RIBBON_AGG_CACHE: dict[tuple[int, str, int], tuple[float, float, float, float, float]] = {}
_RIBBON_AGG_CACHE_MAX = 48


def clear_metric_ribbon_stats_cache() -> None:
    _RIBBON_AGG_CACHE.clear()


def compute_metric_ribbon_stats(
    bundle: JointMotionBundle,
    metric_id: str,
    *,
    time_s: float | None = None,
    unit: str = "",
) -> MetricRibbonStats:
    """Min / max / mean / ROM / std / current for the active Motion graph.

    Aggregate stats are cached per bundle+metric; only ``current`` is recomputed
    on scrub / playhead moves.
    """
    if bundle.empty:
        return MetricRibbonStats(
            None, None, None, None, None, None, unit=unit, metric_id=metric_id
        )
    series = bundle.series[0]
    y = np.asarray(_series_values_for_ribbon(series, metric_id), dtype=float)
    cache_key = (id(bundle), str(metric_id), int(y.size))
    cached = _RIBBON_AGG_CACHE.get(cache_key)
    if cached is None:
        finite = y[np.isfinite(y)]
        if finite.size == 0:
            return MetricRibbonStats(
                None,
                None,
                None,
                None,
                None,
                None,
                unit=unit,
                joint_label=series.label,
                metric_id=metric_id,
            )
        mn = float(np.nanmin(finite))
        mx = float(np.nanmax(finite))
        mean = float(np.nanmean(finite))
        std = float(np.nanstd(finite)) if finite.size > 1 else 0.0
        rom = mx - mn
        if len(_RIBBON_AGG_CACHE) >= _RIBBON_AGG_CACHE_MAX:
            try:
                _RIBBON_AGG_CACHE.pop(next(iter(_RIBBON_AGG_CACHE)))
            except Exception:
                _RIBBON_AGG_CACHE.clear()
        _RIBBON_AGG_CACHE[cache_key] = (mn, mx, mean, rom, std)
    else:
        mn, mx, mean, rom, std = cached
    current = None
    if time_s is not None and series.times_s.size:
        cur = float(np.interp(float(time_s), series.times_s, y))
        if np.isfinite(cur):
            current = cur
    elif y.size and np.isfinite(y[-1]):
        current = float(y[-1])
    return MetricRibbonStats(
        minimum=mn,
        maximum=mx,
        mean=mean,
        rom=rom,
        std=std,
        current=current,
        unit=unit,
        joint_label=series.label,
        metric_id=metric_id,
    )


def _hover_velocity_acceleration(
    series: JointMotionSeries, index: int, metric_id: str
) -> tuple[float | None, str, float | None, str]:
    """Pick angular or linear kinematics for tooltip Velocity / Acceleration."""
    i = int(index)
    if metric_id in ("angle", "omega", "alpha"):
        vel = float(series.omega_deg_s[i]) if i < len(series.omega_deg_s) else None
        acc = float(series.alpha_deg_s2[i]) if i < len(series.alpha_deg_s2) else None
        if vel is not None and not np.isfinite(vel):
            vel = None
        if acc is not None and not np.isfinite(acc):
            acc = None
        return vel, "°/s", acc, "°/s²"
    vel = float(series.speed_m_s[i]) if i < len(series.speed_m_s) else None
    acc = float(series.accel_m_s2[i]) if i < len(series.accel_m_s2) else None
    if vel is not None and not np.isfinite(vel):
        vel = None
    if acc is not None and not np.isfinite(acc):
        acc = None
    return vel, "m/s", acc, "m/s²"


def _ordered_item_ids(selected: Iterable[str]) -> list[str]:
    selected_set = {str(item) for item in selected}
    return [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_set]


def _item_visibility(snap, joint_id: str | None) -> float:
    """Landmark visibility in [0, 1]; 1.0 if the joint exists without a map."""
    if snap is None or not joint_id:
        return 0.0
    vis_map = (getattr(snap, "metadata", None) or {}).get("landmark_visibility") or {}
    if joint_id in vis_map:
        try:
            return float(vis_map[joint_id])
        except (TypeError, ValueError):
            return 0.0
    joints = getattr(snap, "joints", None) or {}
    return 1.0 if joint_id in joints else 0.0


def _finite_diff(values: np.ndarray, times: np.ndarray, *, fps: float | None = None) -> np.ndarray:
    """Gap-aware finite difference (does not jump tracking holes)."""
    from stablewalk.analysis.kinematic_derivatives import finite_diff

    return finite_diff(values, times, fps=fps)


def build_joint_motion_series(
    recording: GaitMotionRecording,
    item_id: str,
) -> JointMotionSeries | None:
    """Collect angle / ω / α / linear speed / accel / XYZ for one joint."""
    if recording is None or recording.frame_count <= 0:
        return None
    anchor = anchor_joint_for_item(item_id)
    dof_id = _ITEM_DOF_ID.get(item_id)
    n = recording.frame_count
    times = np.empty(n, dtype=float)
    frames = np.empty(n, dtype=int)
    angle = np.full(n, np.nan, dtype=float)
    raw_pose = np.full(n, np.nan, dtype=float)
    omega = np.full(n, np.nan, dtype=float)
    xs = np.full(n, np.nan, dtype=float)
    ys = np.full(n, np.nan, dtype=float)
    zs = np.full(n, np.nan, dtype=float)
    vis = np.full(n, np.nan, dtype=float)

    for index in range(n):
        snap = recording.snapshot_at(index)
        if snap is None:
            times[index] = index / max(float(recording.fps), 1e-6)
            frames[index] = index + 1
            continue
        times[index] = float(snap.time_s)
        frames[index] = int(snap.frame_index) + 1
        joint = snap.joints.get(anchor) if anchor else None
        if joint is not None:
            xs[index] = float(joint.position.x)
            ys[index] = float(joint.position.y)
            zs[index] = float(joint.position.z)
        vis[index] = _item_visibility(snap, anchor)
        ang = angle_value_for_item(item_id, snap)
        if ang is not None:
            angle[index] = float(ang)
        raw_ang = raw_pose_angle_for_item(item_id, snap)
        if raw_ang is not None:
            raw_pose[index] = float(raw_ang)
        dof = snap.get_dof(dof_id) if dof_id else None
        if dof is not None and dof.velocity_deg_s is not None:
            omega[index] = float(dof.velocity_deg_s)

    fps = float(recording.fps) if recording.fps else None
    from stablewalk.analysis.kinematic_derivatives import (
        differentiate_chain,
        finite_diff,
    )

    # One SG pass on angle, then ω / α / angular jerk from that base (Winter).
    if np.isfinite(angle).sum() >= 2:
        ang_chain = differentiate_chain(
            angle,
            times,
            order=3,
            fps=fps,
            confidence=vis,
            despike=True,
            max_jump=40.0,
        )
        angle = ang_chain[0]
        omega = ang_chain[1]
        alpha = ang_chain[2]
        angular_jerk = ang_chain[3]
    else:
        omega_base = omega if np.isfinite(omega).any() else angle
        omega = finite_diff(omega_base, times, fps=fps)
        alpha = finite_diff(omega, times, fps=fps)
        angular_jerk = finite_diff(alpha, times, fps=fps)

    x_chain = differentiate_chain(xs, times, order=3, fps=fps, confidence=vis)
    y_chain = differentiate_chain(ys, times, order=3, fps=fps, confidence=vis)
    z_chain = differentiate_chain(zs, times, order=3, fps=fps, confidence=vis)
    xs, vx, _ax, jx = x_chain
    ys, vy, _ay, jy = y_chain
    zs, vz, _az, jz = z_chain
    speed = np.full(n, np.nan, dtype=float)
    vel_mask = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
    speed[vel_mask] = np.sqrt(vx[vel_mask] ** 2 + vy[vel_mask] ** 2 + vz[vel_mask] ** 2)
    accel = finite_diff(speed, times, fps=fps)
    jerk = np.full(n, np.nan, dtype=float)
    jmask = np.isfinite(jx) & np.isfinite(jy) & np.isfinite(jz)
    jerk[jmask] = np.sqrt(jx[jmask] ** 2 + jy[jmask] ** 2 + jz[jmask] ** 2)

    return JointMotionSeries(
        item_id=item_id,
        label=label_for_item(item_id),
        color=joint_color(item_id),
        times_s=times,
        frames=frames,
        angle_deg=angle,
        omega_deg_s=omega,
        alpha_deg_s2=alpha,
        speed_m_s=speed,
        accel_m_s2=accel,
        jerk_m_s3=jerk,
        angular_jerk_deg_s3=angular_jerk,
        x_m=xs,
        y_m=ys,
        z_m=zs,
        raw_pose_angle_deg=raw_pose,
    )


def build_joint_motion_bundle(
    recording: GaitMotionRecording | None,
    selected_item_ids: Iterable[str],
) -> JointMotionBundle:
    """Build multi-joint series for the current selection (stable GUI order)."""
    if recording is None or recording.frame_count <= 0:
        return JointMotionBundle(series=(), times_s=np.asarray([], dtype=float))
    series_list: list[JointMotionSeries] = []
    for item_id in _ordered_item_ids(selected_item_ids):
        series = build_joint_motion_series(recording, item_id)
        if series is not None:
            series_list.append(series)
    times = (
        series_list[0].times_s
        if series_list
        else np.asarray([], dtype=float)
    )
    return JointMotionBundle(series=tuple(series_list), times_s=times)


def compute_joint_motion_stats(bundle: JointMotionBundle) -> list[JointMotionStats]:
    """Derive compact ROM / peak kinematics for the stats strip."""
    out: list[JointMotionStats] = []
    for series in bundle.series:
        ang = series.angle_deg[np.isfinite(series.angle_deg)]
        omg = series.omega_deg_s[np.isfinite(series.omega_deg_s)]
        alp = series.alpha_deg_s2[np.isfinite(series.alpha_deg_s2)]
        spd = series.speed_m_s[np.isfinite(series.speed_m_s)]
        acc = series.accel_m_s2[np.isfinite(series.accel_m_s2)]
        jrk = series.jerk_m_s3[np.isfinite(series.jerk_m_s3)]
        ajrk = series.angular_jerk_deg_s3[np.isfinite(series.angular_jerk_deg_s3)]
        travel = None
        if series.times_s.size >= 2 and np.isfinite(spd).any():
            from stablewalk.analysis.joint_path_trace import max_connected_dt

            dt = np.diff(series.times_s)
            mid = 0.5 * (spd[:-1] + spd[1:])
            fps = None
            finite_dt = dt[np.isfinite(dt) & (dt > 0)]
            if finite_dt.size:
                fps = float(1.0 / np.median(finite_dt))
            max_dt = max_connected_dt(fps or 30.0)
            mask = (
                np.isfinite(dt)
                & np.isfinite(mid)
                & (dt > 0)
                & (dt <= max_dt)
            )
            if np.any(mask):
                travel = float(np.sum(mid[mask] * dt[mask]))
        out.append(
            JointMotionStats(
                item_id=series.item_id,
                label=series.label,
                rom_deg=float(np.nanmax(ang) - np.nanmin(ang)) if ang.size else None,
                mean_angle_deg=float(np.nanmean(ang)) if ang.size else None,
                peak_omega_deg_s=float(np.nanmax(np.abs(omg))) if omg.size else None,
                peak_alpha_deg_s2=float(np.nanmax(np.abs(alp))) if alp.size else None,
                peak_speed_m_s=float(np.nanmax(spd)) if spd.size else None,
                peak_accel_m_s2=float(np.nanmax(np.abs(acc))) if acc.size else None,
                peak_jerk_m_s3=float(np.nanmax(np.abs(jrk))) if jrk.size else None,
                peak_angular_jerk_deg_s3=(
                    float(np.nanmax(np.abs(ajrk))) if ajrk.size else None
                ),
                travel_m=travel,
            )
        )
    return out


def write_joint_motion_csv(
    bundle: JointMotionBundle,
    path: str | Path,
) -> Path:
    """Export all selected joints' kinematics to CSV."""
    out = Path(path)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDNAMES))
        writer.writeheader()
        for series in bundle.series:
            for i in range(len(series.times_s)):
                writer.writerow(
                    {
                        "time_s": f"{float(series.times_s[i]):.6f}",
                        "frame": int(series.frames[i]),
                        "joint_id": series.item_id,
                        "joint_label": series.label,
                        "angle_deg": _csv_num(series.angle_deg[i]),
                        "raw_pose_angle_deg": _csv_num(
                            series.raw_pose_angle_deg[i]
                            if series.raw_pose_angle_deg is not None
                            else float("nan")
                        ),
                        "angular_velocity_deg_s": _csv_num(series.omega_deg_s[i]),
                        "angular_acceleration_deg_s2": _csv_num(
                            series.alpha_deg_s2[i]
                        ),
                        "linear_speed_m_s": _csv_num(series.speed_m_s[i]),
                        "linear_accel_m_s2": _csv_num(series.accel_m_s2[i]),
                        "jerk_m_s3": _csv_num(series.jerk_m_s3[i]),
                        "angular_jerk_deg_s3": _csv_num(
                            series.angular_jerk_deg_s3[i]
                        ),
                        "x_m": _csv_num(series.x_m[i]),
                        "y_m": _csv_num(series.y_m[i]),
                        "z_m": _csv_num(series.z_m[i]),
                    }
                )
    return out


def _csv_num(value: float) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.6f}"


def ensure_joint_motion_axes(fig: Figure) -> list[Axes]:
    """Create or reuse a 2×3 shared-x subplot grid (Overview)."""
    if len(fig.axes) == 6:
        return list(fig.axes)
    fig.clear()
    axes = fig.subplots(2, 3, sharex=True)
    return [ax for row in axes for ax in row]


def ensure_motion_workspace_axes(fig: Figure) -> list[Axes]:
    """Create or reuse a 7×1 shared-x stack (legacy Motion workspace)."""
    n = MOTION_WORKSPACE_PANEL_COUNT
    if len(fig.axes) == n:
        return list(fig.axes)
    fig.clear()
    axes = fig.subplots(n, 1, sharex=True)
    return list(axes)


def ensure_single_motion_metric_axes(fig: Figure) -> list[Axes]:
    """Create or reuse a single Axes for one Motion Analysis tab."""
    if len(fig.axes) == 1:
        return list(fig.axes)
    fig.clear()
    return [fig.add_subplot(1, 1, 1)]


def metric_tuple_for_tab(
    metric_id: str,
    item_ids: Iterable[str] | None = None,
) -> tuple[str, str, str]:
    """Resolve a Motion Analysis tab id to (metric_id, title, ylabel)."""
    if metric_id == "angle":
        title, ylabel = angle_chart_labels(item_ids)
        return ("angle", title, ylabel)
    for item in MOTION_ANALYSIS_TABS:
        if item[0] == metric_id:
            return item
    for item in MOTION_WORKSPACE_METRICS:
        if item[0] == metric_id:
            return item
    return (metric_id, metric_id.replace("_", " ").title(), "")


def _local_peak_indices(values: np.ndarray, *, min_sep: int = 3) -> list[int]:
    """Simple local-maxima indices for display peak markers (not clinical peaks)."""
    n = len(values)
    if n < 3:
        return []
    peaks: list[int] = []
    last = -min_sep
    for i in range(1, n - 1):
        v = values[i]
        if not np.isfinite(v):
            continue
        a, b = values[i - 1], values[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if v >= a and v >= b and (i - last) >= min_sep:
            peaks.append(i)
            last = i
    return peaks


def _event_times_by_type(events) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = {
        "left_heel_strike": [],
        "right_heel_strike": [],
        "left_toe_off": [],
        "right_toe_off": [],
    }
    for ev in events or []:
        key = getattr(ev, "event_type", None)
        if key in buckets:
            buckets[key].append(float(ev.time_s))
    return buckets


def _annotate_gait_events(ax: Axes, events) -> None:
    if not events:
        return
    from stablewalk.ui.viewers.chart_reference import draw_gait_event_markers

    buckets = _event_times_by_type(events)
    draw_gait_event_markers(
        ax,
        left_hs=buckets["left_heel_strike"],
        right_hs=buckets["right_heel_strike"],
        left_to=buckets["left_toe_off"],
        right_to=buckets["right_toe_off"],
        show_legend=False,
    )


def _annotate_peaks(
    ax: Axes,
    times: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label_top: int = 3,
) -> None:
    """Mark local maxima; label the strongest peaks (display-only)."""
    from stablewalk.ui.viewers.chart_style import mark_secondary_annotation

    idxs = _local_peak_indices(values, min_sep=max(3, len(values) // 40))
    if not idxs:
        return
    pts = [
        (int(i), float(times[i]), float(values[i]))
        for i in idxs
        if np.isfinite(times[i]) and np.isfinite(values[i])
    ]
    if not pts:
        return
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    ax.scatter(
        xs,
        ys,
        s=34,
        facecolors="none",
        edgecolors=color,
        linewidths=1.25,
        zorder=12,
        alpha=0.9,
        marker="o",
    )
    # Annotate the strongest peaks only — keeps the plot readable.
    ranked = sorted(pts, key=lambda p: abs(p[2]), reverse=True)[: max(0, int(label_top))]
    y0, y1 = ax.get_ylim()
    yr = (y1 - y0) if y1 > y0 else 1.0
    for _i, x, y in ranked:
        mark_secondary_annotation(
            ax.annotate(
                f"{y:.1f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                color=color,
                fontsize=8.5,
                ha="center",
                va="bottom",
                zorder=13,
                alpha=0.9,
            )
        )
    # Preserve prior ylim if annotate nudged autoscale unexpectedly.
    if yr > 0:
        ax.set_ylim(y0, y1)


def _draw_metric_panels(
    fig: Figure,
    axes: list[Axes],
    metrics: tuple[tuple[str, str, str], ...],
    bundle: JointMotionBundle,
    *,
    playhead: PlayheadState | None,
    empty_message: str,
    gait_events=None,
    show_peaks: bool = False,
    hover_density: int = 40,
    ylabel_pad: float | None = None,
    show_raw_pose_debug: bool = SHOW_RAW_POSE_ANGLE_DEBUG,
) -> list[ChartHoverPoint]:
    from stablewalk.ui.theme import BORDER, PANEL

    fig.patch.set_facecolor(PANEL)
    hover: list[ChartHoverPoint] = []

    if bundle.empty:
        for ax in axes:
            ax.clear()
            apply_chart_panel_style(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(BORDER)
        mid = axes[min(1, len(axes) - 1)]
        mid.text(
            0.5,
            0.5,
            empty_message,
            transform=mid.transAxes,
            ha="center",
            va="center",
            color=TEXT_SECONDARY,
            fontsize=9,
        )
        return hover

    legend_handles = []
    legend_labels: list[str] = []
    n_metrics = len(metrics)
    density = max(20, int(hover_density))

    for ax_index, (ax, (metric_id, title, ylabel)) in enumerate(
        zip(axes, metrics, strict=True)
    ):
        if metric_id == "angle":
            title, ylabel = angle_chart_labels(s.item_id for s in bundle.series)
        ax.clear()
        apply_chart_panel_style(ax)

        if metric_id == "position":
            # Single joint: X/Y/Z channels. Multi-joint: |r| per joint.
            if len(bundle.series) == 1:
                series = bundle.series[0]
                for channel, color, label in (
                    (series.x_m, "#5b9fd4", "X"),
                    (series.y_m, "#3d9a6a", "Y"),
                    (series.z_m, "#d4a84b", "Z"),
                ):
                    mask = np.isfinite(series.times_s) & np.isfinite(channel)
                    if not np.any(mask):
                        continue
                    (line,) = ax.plot(
                        *_gapped_xy(series.times_s, channel),
                        **series_plot_kwargs(
                            color=color,
                            label=label,
                            item_id=f"{series.item_id}_{label.lower()}",
                        ),
                    )
                    if label not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(label)
                    idxs = np.flatnonzero(mask)
                    step = max(1, len(idxs) // density)
                    for idx in idxs[::step]:
                        vel, vel_u, acc, acc_u = _hover_velocity_acceleration(
                            series, int(idx), "position"
                        )
                        hover.append(
                            ChartHoverPoint(
                                ax=ax,
                                x=float(series.times_s[idx]),
                                y=float(channel[idx]),
                                frame_index=int(series.frames[idx]),
                                list_pos=int(idx),
                                timestamp_s=float(series.times_s[idx]),
                                value=float(channel[idx]),
                                joint_name=f"{series.label} {label}",
                                metric_name="Position",
                                unit="m",
                                velocity=vel,
                                acceleration=acc,
                                velocity_unit=vel_u,
                                acceleration_unit=acc_u,
                            )
                        )
                    if show_peaks:
                        _annotate_peaks(
                            ax, series.times_s, channel, color=color
                        )
            else:
                for series in bundle.series:
                    y = series.position_mag_m
                    mask = np.isfinite(series.times_s) & np.isfinite(y)
                    if not np.any(mask):
                        continue
                    (line,) = ax.plot(
                        *_gapped_xy(series.times_s, y),
                        **series_plot_kwargs(
                            color=series.color,
                            label=series.label,
                            item_id=series.item_id,
                        ),
                    )
                    if series.label not in legend_labels:
                        legend_handles.append(line)
                        legend_labels.append(series.label)
                    idxs = np.flatnonzero(mask)
                    step = max(1, len(idxs) // density)
                    for idx in idxs[::step]:
                        vel, vel_u, acc, acc_u = _hover_velocity_acceleration(
                            series, int(idx), "position"
                        )
                        hover.append(
                            ChartHoverPoint(
                                ax=ax,
                                x=float(series.times_s[idx]),
                                y=float(y[idx]),
                                frame_index=int(series.frames[idx]),
                                list_pos=int(idx),
                                timestamp_s=float(series.times_s[idx]),
                                value=float(y[idx]),
                                joint_name=series.label,
                                metric_name="Position",
                                unit="m",
                                velocity=vel,
                                acceleration=acc,
                                velocity_unit=vel_u,
                                acceleration_unit=acc_u,
                            )
                        )
                    if show_peaks:
                        _annotate_peaks(
                            ax, series.times_s, y, color=series.color
                        )
        else:
            for series in bundle.series:
                y = series.values_for(metric_id)
                mask = np.isfinite(series.times_s) & np.isfinite(y)
                if not np.any(mask):
                    continue
                (line,) = ax.plot(
                    *_gapped_xy(series.times_s, y),
                    **series_plot_kwargs(
                        color=series.color,
                        label=series.label,
                        item_id=series.item_id,
                    ),
                )
                if metric_id == "angle" and series.label not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(series.label)

                if (
                    metric_id == "angle"
                    and show_raw_pose_debug
                    and series.raw_pose_angle_deg is not None
                ):
                    raw = series.raw_pose_angle_deg
                    raw_mask = np.isfinite(series.times_s) & np.isfinite(raw)
                    if np.any(raw_mask):
                        debug_label = f"{series.label} raw pose"
                        (raw_line,) = ax.plot(
                            *_gapped_xy(series.times_s, raw),
                            **series_plot_kwargs(
                                color=series.color,
                                label=debug_label,
                                item_id=f"{series.item_id}_raw_pose",
                                linestyle="--",
                                alpha=0.55,
                            ),
                        )
                        if debug_label not in legend_labels:
                            legend_handles.append(raw_line)
                            legend_labels.append(debug_label)

                idxs = np.flatnonzero(mask)
                step = max(1, len(idxs) // density)
                unit = _unit_from_ylabel(ylabel)
                for idx in idxs[::step]:
                    vel, vel_u, acc, acc_u = _hover_velocity_acceleration(
                        series, int(idx), metric_id
                    )
                    hover.append(
                        ChartHoverPoint(
                            ax=ax,
                            x=float(series.times_s[idx]),
                            y=float(y[idx]),
                            frame_index=int(series.frames[idx]),
                            list_pos=int(idx),
                            timestamp_s=float(series.times_s[idx]),
                            value=float(y[idx]),
                            joint_name=series.label,
                            metric_name=title.split(" vs ")[0],
                            unit=unit,
                            velocity=vel,
                            acceleration=acc,
                            velocity_unit=vel_u,
                            acceleration_unit=acc_u,
                        )
                    )
                if show_peaks:
                    _annotate_peaks(ax, series.times_s, y, color=series.color)

        style_chart_title(ax, title, pad=8.0)
        ylabel_kw: dict[str, float | str] = {
            "color": TEXT_SECONDARY,
            "fontsize": _AXIS_LABEL_SIZE,
        }
        if ylabel_pad is not None:
            ylabel_kw["labelpad"] = ylabel_pad
        ax.set_ylabel(ylabel, **ylabel_kw)
        try:
            ax.yaxis.label.set_clip_on(False)
        except Exception:
            pass
        ax.tick_params(colors=TEXT_SECONDARY, labelcolor=TEXT_SECONDARY, labelsize=_TICK_LABEL_SIZE)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        apply_chart_grid(ax, y_minor=True)
        if gait_events:
            _annotate_gait_events(ax, gait_events)
        if playhead is not None:
            from stablewalk.ui.viewers.chart_playhead import (
                resolve_playhead_value,
                set_playhead_value_source,
            )

            value_label = None
            value_y = None
            if bundle.series:
                s0 = bundle.series[0]
                unit = ""
                if metric_id == "position" and len(bundle.series) == 1:
                    y = s0.x_m
                    unit = "m"
                    prefix = f"{s0.label} X"
                elif metric_id == "position":
                    y = s0.position_mag_m
                    unit = "m"
                    prefix = s0.label
                else:
                    y = s0.values_for(metric_id)
                    # Pull unit from ylabel like "Angle (°)" → "°"
                    unit = ylabel.split("(")[-1].rstrip(")") if "(" in ylabel else ""
                    prefix = s0.label
                if y.size:
                    set_playhead_value_source(ax, s0.times_s, y, prefix, unit)
                    value_label, value_y = resolve_playhead_value(ax, playhead.time_s)
            draw_chart_playhead(
                ax,
                playhead,
                show_label=(ax_index == 0),
                value_label=value_label if ax_index == 0 else "",
                value_y=value_y,
                zorder=20,
            )

        show_x = ax_index == n_metrics - 1
        configure_time_axis(ax, show_xlabel=show_x, nbins=6)
        if show_x:
            ax.set_xlabel(TIMELINE_X_LABEL, color=TEXT_SECONDARY, fontsize=_AXIS_LABEL_SIZE)
            try:
                ax.xaxis.label.set_clip_on(False)
            except Exception:
                pass
        unclip_axis_chrome(ax)

    if legend_handles:
        n_labels = len(legend_labels)
        width_px = 0.0
        try:
            width_px = float(fig.get_figwidth() * fig.dpi)
        except Exception:
            width_px = 0.0
        ncol = min(4, n_labels)
        if width_px and width_px < 640:
            ncol = min(ncol, 2)
        height_px = 0.0
        try:
            height_px = float(fig.get_figheight() * fig.dpi)
        except Exception:
            height_px = 0.0
        style_chart_legend(
            axes[0],
            handles=legend_handles,
            labels=legend_labels,
            loc="upper right",
            bbox_to_anchor=None,
            outside=False,
            ncol=ncol,
        )

    t0 = float(np.nanmin(bundle.times_s)) if bundle.times_s.size else 0.0
    t1 = float(np.nanmax(bundle.times_s)) if bundle.times_s.size else 1.0
    if t1 <= t0:
        t1 = t0 + 1.0
    for ax in axes:
        ax.set_xlim(t0, t1)
    return hover


def draw_joint_motion_analysis_chart(
    fig: Figure,
    bundle: JointMotionBundle,
    *,
    playhead: PlayheadState | None = None,
) -> list[ChartHoverPoint]:
    """Draw the Overview six scientific panels and return hover targets."""
    axes = ensure_joint_motion_axes(fig)
    hover = _draw_metric_panels(
        fig,
        axes,
        JOINT_MOTION_METRICS,
        bundle,
        playhead=playhead,
        empty_message=(
            "Select one or more joints to plot\n"
            "angle, angular rates, and position vs time."
        ),
    )
    fig.subplots_adjust(
        left=0.08, right=0.96, top=0.90, bottom=0.12, wspace=0.28, hspace=0.38
    )
    unclip_figure_chrome(fig)
    return hover


def draw_motion_workspace_chart(
    fig: Figure,
    bundle: JointMotionBundle,
    *,
    playhead: PlayheadState | None = None,
    gait_events=None,
    show_peaks: bool = True,
) -> list[ChartHoverPoint]:
    """Draw the legacy Motion Analysis 7-panel workspace."""
    axes = ensure_motion_workspace_axes(fig)
    hover = _draw_metric_panels(
        fig,
        axes,
        MOTION_WORKSPACE_METRICS,
        bundle,
        playhead=playhead,
        empty_message=(
            "Select a joint to analyze.\n"
            "Kinematics graphs share one synchronized timeline."
        ),
        gait_events=gait_events,
        show_peaks=show_peaks,
        hover_density=80,
    )
    fig.subplots_adjust(
        left=0.12, right=0.96, top=0.90, bottom=0.16, hspace=0.38
    )
    unclip_figure_chrome(fig)
    return hover


def draw_motion_metric_chart(
    fig: Figure,
    bundle: JointMotionBundle,
    metric_id: str,
    *,
    playhead: PlayheadState | None = None,
    gait_events=None,
    show_peaks: bool = True,
    show_raw_pose_debug: bool = SHOW_RAW_POSE_ANGLE_DEBUG,
) -> list[ChartHoverPoint]:
    """Draw one Motion Analysis metric tab from a shared cached bundle."""
    metric = metric_tuple_for_tab(
        metric_id, (s.item_id for s in bundle.series)
    )
    axes = ensure_single_motion_metric_axes(fig)
    hover = _draw_metric_panels(
        fig,
        axes,
        (metric,),
        bundle,
        playhead=playhead,
        empty_message=(
            "Select a joint to analyze.\n"
            "All metric tabs share one synchronized timeline."
        ),
        gait_events=gait_events,
        show_peaks=show_peaks,
        hover_density=100,
        ylabel_pad=_MOTION_METRIC_YLABEL_PAD,
        show_raw_pose_debug=show_raw_pose_debug,
    )
    fig.subplots_adjust(**motion_metric_chart_margins(fig))
    unclip_figure_chrome(fig)
    return hover
