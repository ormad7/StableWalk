"""
Knee angle time-series for the GUI chart (degrees, full-video trajectory + playhead).

Angle convention (documented for scientific interpretability):
  Pose-derived: interior angle at the knee (hip–knee–ankle) in the image plane,
  converted to **anatomical knee flexion** via flexion = 180° − interior angle.
  0° ≈ full extension; larger values = more flexion.
  OpenSim IK: model knee coordinate interpreted as flexion (degrees); 0° = extension
  when the coordinate follows standard OpenSim knee flexion sign.

Sources are never mixed within one plotted series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from stablewalk.analysis.gait_feature_analysis import read_opensim_mot_timeseries
from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.kinematics import compute_joint_angles
from stablewalk.ui.viewers.chart_hover import ChartHoverPoint, append_line_hover_points
from stablewalk.ui.viewers.chart_playhead import PlayheadState

KneeAngleSource = Literal["opensim_ik", "pose_derived"]
KneeAngleSourcePreference = Literal["auto", "opensim_ik", "pose_derived"]

LEFT_IK_COLUMNS = ("knee_angle_l", "knee_l", "knee_flexion_l")
RIGHT_IK_COLUMNS = ("knee_angle_r", "knee_r", "knee_flexion_r")

ANGLE_CONVENTION_SUMMARY = (
    "Knee flexion (deg): 0 deg = full extension; larger values = more flexion. "
    "Pose: geometric hip-knee-ankle interior angle converted (180 deg - theta)."
)

LABEL_LEFT_KNEE = "LEFT KNEE"
LABEL_RIGHT_KNEE = "RIGHT KNEE"


@dataclass
class KneeAngleSeries:
    """Full-video knee flexion trajectories aligned to wall-clock time."""

    times_s: np.ndarray
    left_deg: np.ndarray
    right_deg: np.ndarray
    source: KneeAngleSource
    angle_definition: str
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_export_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "angle_definition": self.angle_definition,
            "units": "deg",
            "y_axis_label": "Knee flexion (deg)",
            "times_s": self.times_s.tolist(),
            "left_knee_deg": self.left_deg.tolist(),
            "right_knee_deg": self.right_deg.tolist(),
            **self.metadata,
        }

    def diagnostic_summary(self) -> str:
        from stablewalk.ui.viewers.knee_chart_interpretation import format_diagnostic_report

        return format_diagnostic_report(self)


def knee_flexion_rom_deg(values: np.ndarray) -> float | None:
    """Robust flexion ROM so single-frame spikes do not dominate the HUD."""
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return None
    if finite.size < 5:
        return float(np.max(finite) - np.min(finite))
    lo = float(np.percentile(finite, 10))
    hi = float(np.percentile(finite, 90))
    med = float(np.median(finite))
    mad = float(np.median(np.abs(finite - med)))
    if mad < 1e-6:
        mad = 1.0
    gate = 3.5 * mad
    lo = max(lo, med - gate)
    hi = min(hi, med + gate)
    return max(0.0, hi - lo)


def largest_frame_jump_deg(values: np.ndarray) -> float | None:
    finite_idx = np.where(np.isfinite(values))[0]
    if finite_idx.size < 2:
        return None
    max_jump = 0.0
    for i in range(1, len(finite_idx)):
        a, b = finite_idx[i - 1], finite_idx[i]
        if b == a + 1:
            jump = abs(float(values[b]) - float(values[a]))
            max_jump = max(max_jump, jump)
    return round(max_jump, 2) if max_jump > 0 else 0.0


def frame_time_s(frame, *, fps: float) -> float:
    """Reliable wall-clock time for a pose frame (handles zero stored timestamps)."""
    if frame.timestamp_s > 1e-9:
        return float(frame.timestamp_s)
    if frame.timestamp_ms > 0:
        return frame.timestamp_ms / 1000.0
    return float(frame.frame_index) / max(fps, 1e-6)


def _interior_to_flexion_deg(interior: float | None) -> float | None:
    """Convert pose interior knee angle (0–180°) to flexion (0° = full extension)."""
    if interior is None or not np.isfinite(interior):
        return None
    return float(180.0 - interior)


def _mot_value_at(
    mot: dict[str, np.ndarray],
    time_s: float,
    columns: tuple[str, ...],
) -> float | None:
    times = mot.get("time")
    if times is None or len(times) < 2:
        return None
    col = next((c for c in columns if c in mot), None)
    if col is None:
        return None
    val = float(np.interp(time_s, times, mot[col]))
    if not np.isfinite(val):
        return None
    return val


def _maybe_radians_to_deg(values: np.ndarray) -> np.ndarray:
    """Convert OpenSim column if values look like radians (|v| mostly < 2π)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return values
    if np.nanmax(np.abs(finite)) <= 6.5:
        return np.degrees(values)
    return values


def ik_mot_passes_quality_check(
    mot: dict[str, np.ndarray] | None,
    *,
    min_finite_fraction: float = 0.35,
) -> bool:
    """True when IK MOT has usable knee columns with enough finite samples."""
    if mot is None:
        return False
    times = mot.get("time")
    if times is None or len(times) < 4:
        return False
    n = len(times)
    for cols in (LEFT_IK_COLUMNS, RIGHT_IK_COLUMNS):
        col = next((c for c in cols if c in mot), None)
        if col is None:
            continue
        arr = np.asarray(mot[col], dtype=float)
        finite_frac = float(np.sum(np.isfinite(arr))) / max(n, 1)
        if finite_frac >= min_finite_fraction:
            return True
    return False


def opensim_ik_available(
    ik_mot_path: Path | str | None,
    *,
    ik_completed: bool = True,
) -> bool:
    """Whether OpenSim IK can be offered as a knee chart source."""
    if not ik_completed or not ik_mot_path:
        return False
    mot = read_opensim_mot_timeseries(Path(ik_mot_path))
    return ik_mot_passes_quality_check(mot)


def build_knee_angle_series(
    sequence: PoseSequence,
    pose_indices: list[int],
    *,
    ik_mot_path: Path | str | None = None,
    source_preference: KneeAngleSourcePreference = "auto",
    ik_quality_ok: bool | None = None,
) -> KneeAngleSeries:
    """
    Build aligned left/right knee flexion arrays for the full analyzed video.

    Never mixes OpenSim IK and pose-derived values in the same output arrays.
    """
    fps = max(sequence.fps, 1e-6)
    n = len(pose_indices)
    times = np.empty(n, dtype=float)
    left = np.full(n, np.nan, dtype=float)
    right = np.full(n, np.nan, dtype=float)

    mot = read_opensim_mot_timeseries(Path(ik_mot_path)) if ik_mot_path else None
    quality_ok = (
        ik_quality_ok
        if ik_quality_ok is not None
        else ik_mot_passes_quality_check(mot)
    )

    use_ik = False
    if source_preference == "pose_derived":
        use_ik = False
    elif source_preference == "opensim_ik":
        use_ik = mot is not None and quality_ok
    else:
        use_ik = mot is not None and quality_ok

    ik_left_count = 0
    ik_right_count = 0

    for i, idx in enumerate(pose_indices):
        frame = sequence.frames[idx]
        t = frame_time_s(frame, fps=fps)
        times[i] = t

        if use_ik and mot is not None:
            lv = _mot_value_at(mot, t, LEFT_IK_COLUMNS)
            rv = _mot_value_at(mot, t, RIGHT_IK_COLUMNS)
            if lv is not None:
                left[i] = lv
                ik_left_count += 1
            if rv is not None:
                right[i] = rv
                ik_right_count += 1
        else:
            angles = frame.joint_angles
            if angles is None and frame.keypoints:
                angles = compute_joint_angles(frame.keypoints)
            if angles is not None:
                left[i] = _interior_to_flexion_deg(angles.left_knee)
                right[i] = _interior_to_flexion_deg(angles.right_knee)

    source: KneeAngleSource = "opensim_ik" if use_ik else "pose_derived"
    if source == "opensim_ik":
        left = _maybe_radians_to_deg(left)
        right = _maybe_radians_to_deg(right)
        angle_def = "OpenSim IK knee flexion (deg); 0 deg = extension"
    else:
        angle_def = "Pose interior angle -> flexion (180 deg - theta); 0 deg = extension"

    left_valid = int(np.sum(np.isfinite(left)))
    right_valid = int(np.sum(np.isfinite(right)))
    all_vals = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
    nan_pct = 100.0 * (1.0 - (left_valid + right_valid) / max(2 * n, 1))

    y_min = y_max = None
    if all_vals.size:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))

    left_rom = knee_flexion_rom_deg(left)
    right_rom = knee_flexion_rom_deg(right)

    return KneeAngleSeries(
        times_s=times,
        left_deg=left,
        right_deg=right,
        source=source,
        angle_definition=angle_def,
        fps=fps,
        metadata={
            "left_valid": left_valid,
            "right_valid": right_valid,
            "left_min": round(float(np.nanmin(left)), 2) if left_valid else None,
            "left_max": round(float(np.nanmax(left)), 2) if left_valid else None,
            "right_min": round(float(np.nanmin(right)), 2) if right_valid else None,
            "right_max": round(float(np.nanmax(right)), 2) if right_valid else None,
            "left_rom": round(left_rom, 2) if left_rom is not None else None,
            "right_rom": round(right_rom, 2) if right_rom is not None else None,
            "left_max_jump": largest_frame_jump_deg(left),
            "right_max_jump": largest_frame_jump_deg(right),
            "nan_pct": nan_pct,
            "x_min": round(float(times[0]), 3) if n else None,
            "x_max": round(float(times[-1]), 3) if n else None,
            "y_min": round(y_min, 2) if y_min is not None else None,
            "y_max": round(y_max, 2) if y_max is not None else None,
            "frame_count": n,
            "ik_mot_path": str(ik_mot_path) if ik_mot_path else None,
            "ik_quality_ok": quality_ok,
            "source_preference": source_preference,
        },
    )


def _plot_knee_trace(
    ax,
    times: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    legend_label: str,
    end_label: str,
) -> None:
    """Single continuous trace per leg; NaN frames create intentional gaps."""
    masked = np.ma.masked_invalid(values.astype(float))
    if not np.any(~masked.mask):
        return
    ax.plot(
        times,
        masked,
        color=color,
        label=legend_label,
        linewidth=1.85,
        solid_capstyle="round",
        zorder=4,
    )
    valid = np.where(np.isfinite(values))[0]
    if valid.size:
        last = int(valid[-1])
        ax.annotate(
            end_label,
            xy=(float(times[last]), float(values[last])),
            xytext=(5, 0),
            textcoords="offset points",
            color=color,
            fontsize=8,
            fontweight="bold",
            va="center",
            ha="left",
            clip_on=True,
            zorder=7,
        )


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


def _draw_gait_event_markers(ax, events, *, y_lo: float, y_hi: float) -> list:
    """Publication HS/TO markers (OpenSim / Vicon style)."""
    from stablewalk.ui.viewers.chart_reference import draw_gait_event_markers

    del y_lo, y_hi  # ylim used internally by shared helper
    buckets = _event_times_by_type(events)
    draw_gait_event_markers(
        ax,
        left_hs=buckets["left_heel_strike"],
        right_hs=buckets["right_heel_strike"],
        left_to=buckets["left_toe_off"],
        right_to=buckets["right_toe_off"],
        show_legend=True,
    )
    return []


def _knee_confidence_series(times: np.ndarray, gait_cycle) -> np.ndarray | None:
    """Mean foot visibility as a display-only confidence proxy."""
    if gait_cycle is None or not getattr(gait_cycle, "per_frame", None):
        return None
    frames = gait_cycle.per_frame
    if not frames or len(frames) != len(times):
        # Interpolate by time when lengths differ.
        t_src = np.asarray([float(f.time_s) for f in frames], dtype=float)
        conf_src = np.asarray(
            [
                0.5
                * (
                    float(getattr(f.left, "visibility", 1.0))
                    + float(getattr(f.right, "visibility", 1.0))
                )
                for f in frames
            ],
            dtype=float,
        )
        if t_src.size < 2:
            return None
        return np.interp(times.astype(float), t_src, conf_src)
    return np.asarray(
        [
            0.5
            * (
                float(getattr(f.left, "visibility", 1.0))
                + float(getattr(f.right, "visibility", 1.0))
            )
            for f in frames
        ],
        dtype=float,
    )


def _draw_stance_swing_regions(
    ax,
    gait_cycle,
    times: np.ndarray,
) -> None:
    """Subtle left-foot stance / swing shading when contact confidence is sufficient."""
    from stablewalk.ui.colors import BORDER

    if gait_cycle is None or not gait_cycle.per_frame:
        return
    conf = gait_cycle.metrics.contact_confidence
    if conf < 0.48:
        return

    per_frame = gait_cycle.per_frame
    if not per_frame:
        return

    labels_used: set[str] = set()

    def _span_labeled(start_t: float, end_t: float, *, label: str, alpha: float) -> None:
        if end_t <= start_t:
            return
        show_label = label if label not in labels_used else None
        if show_label:
            labels_used.add(label)
        ax.axvspan(
            start_t,
            end_t,
            ymin=0.0,
            ymax=1.0,
            facecolor=BORDER,
            alpha=alpha,
            zorder=0,
            label=show_label,
        )

    in_stance = False
    span_start: float | None = None
    for state in per_frame:
        left_stance = state.left_contact == 1
        t = state.time_s
        if left_stance and not in_stance:
            span_start = float(t)
            in_stance = True
        elif not left_stance and in_stance and span_start is not None:
            _span_labeled(span_start, float(t), label="STANCE (left foot)", alpha=0.12)
            in_stance = False
            span_start = None
    if in_stance and span_start is not None and per_frame:
        last_t = float(per_frame[-1].time_s)
        _span_labeled(span_start, last_t, label="STANCE (left foot)", alpha=0.12)

    in_swing = False
    span_start = None
    for state in per_frame:
        left_swing = state.left_contact == 0
        t = state.time_s
        if left_swing and not in_swing:
            span_start = float(t)
            in_swing = True
        elif not left_swing and in_swing and span_start is not None:
            _span_labeled(span_start, float(t), label="SWING (left foot)", alpha=0.06)
            in_swing = False
            span_start = None


def draw_knee_time_chart(
    ax,
    series: KneeAngleSeries,
    *,
    playhead: PlayheadState | None = None,
    playhead_list_pos: int | None = None,
    gait_events: list | None = None,
    gait_cycle=None,
) -> None:
    """Draw full-video left/right knee flexion with playhead, events, and legend."""
    from stablewalk.ui.colors import MUTED, SIDE_LEFT, SIDE_RIGHT, TEXT
    from stablewalk.ui.viewers.chart_playhead import draw_chart_playhead
    from stablewalk.ui.viewers.chart_reference import (
        KNEE_FLEXION_ABNORMAL_ABOVE_DEG,
        KNEE_FLEXION_NORMAL_DEG,
        draw_confidence_overlay,
        draw_reference_y_bands,
    )
    from stablewalk.ui.viewers.chart_style import (
        style_chart_legend,
        style_chart_title,
        style_single_time_series_chart,
    )

    style_single_time_series_chart(ax, ylabel="Knee flexion (°)")

    times = series.times_s
    left = series.left_deg
    right = series.right_deg

    finite = np.concatenate([left[np.isfinite(left)], right[np.isfinite(right)]])
    y_lo = y_hi = 0.0
    margin = 10.0
    if finite.size:
        y_lo = float(np.min(finite))
        y_hi = float(np.max(finite))
        # Include normative band headroom for Visual3D-style range context.
        y_lo = min(y_lo, KNEE_FLEXION_NORMAL_DEG[0])
        y_hi = max(y_hi, KNEE_FLEXION_NORMAL_DEG[1], KNEE_FLEXION_ABNORMAL_ABOVE_DEG)
        margin = max(6.0, (y_hi - y_lo) * 0.12)
        ax.set_ylim(y_lo - margin, y_hi + margin)
    elif len(times) >= 2:
        ax.set_xlim(float(times[0]), float(times[-1]))
        y_lo, y_hi = 0.0, 95.0
        ax.set_ylim(y_lo - margin, y_hi + margin)

    draw_reference_y_bands(
        ax,
        normal=KNEE_FLEXION_NORMAL_DEG,
        abnormal_below=-5.0,
        abnormal_above=KNEE_FLEXION_ABNORMAL_ABOVE_DEG,
        label_normal=True,
    )

    conf = _knee_confidence_series(times, gait_cycle)
    if conf is not None:
        draw_confidence_overlay(ax, times, conf, threshold=0.55)

    _draw_stance_swing_regions(ax, gait_cycle, times)

    _plot_knee_trace(
        ax,
        times,
        left,
        color=SIDE_LEFT,
        legend_label=f"{LABEL_LEFT_KNEE} flexion",
        end_label=LABEL_LEFT_KNEE,
    )
    _plot_knee_trace(
        ax,
        times,
        right,
        color=SIDE_RIGHT,
        legend_label=f"{LABEL_RIGHT_KNEE} flexion",
        end_label=LABEL_RIGHT_KNEE,
    )

    _draw_gait_event_markers(ax, gait_events or [], y_lo=y_lo, y_hi=y_hi)

    list_pos = playhead_list_pos
    if playhead is not None and playhead.list_pos is not None:
        list_pos = playhead.list_pos
    if playhead is None and list_pos is not None and 0 <= list_pos < len(times):
        playhead = PlayheadState(
            time_s=float(times[list_pos]),
            frame_index=0,
            list_pos=list_pos,
        )
    elif playhead is None and list_pos is not None:
        list_pos = None

    if playhead is not None:
        value_parts: list[str] = []
        value_y = None
        if list_pos is not None and 0 <= list_pos < len(times):
            t = float(times[list_pos])
            for arr, color, leg in (
                (left, SIDE_LEFT, "L"),
                (right, SIDE_RIGHT, "R"),
            ):
                if np.isfinite(arr[list_pos]):
                    val = float(arr[list_pos])
                    value_parts.append(f"{leg} {val:.1f}°")
                    if value_y is None:
                        value_y = val
                    ax.scatter(
                        [t],
                        [val],
                        color=color,
                        s=42,
                        zorder=25,
                        edgecolors=TEXT,
                        linewidths=0.65,
                        label=f"Now ({leg})",
                    )
        value_label = "  ·  ".join(value_parts) if value_parts else None
        draw_chart_playhead(
            ax,
            playhead,
            show_label=True,
            value_label=value_label,
            value_y=value_y,
        )

    if len(times) >= 2:
        pad = max(0.05, (float(times[-1]) - float(times[0])) * 0.02)
        ax.set_xlim(float(times[0]) - pad, float(times[-1]) + pad)

    src_label = "OpenSim IK" if series.source == "opensim_ik" else "Pose-derived"
    style_chart_title(ax, f"Knee flexion · {src_label}")
    style_chart_legend(ax, loc="upper right", fontsize=8.0)

    if not np.any(np.isfinite(left)) and not np.any(np.isfinite(right)):
        ax.text(
            0.5,
            0.5,
            "No valid knee flexion data",
            transform=ax.transAxes,
            ha="center",
            color=MUTED,
            fontsize=10,
        )

    if float(series.metadata.get("nan_pct", 0.0)) >= 8.0:
        ax.text(
            0.01,
            0.02,
            "Gaps = missing landmarks (not separate legs)",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=7,
            style="italic",
            zorder=11,
        )


def register_knee_time_chart_hover_points(
    ax,
    series: KneeAngleSeries,
    pose_indices: list[int],
    hover_points: list[ChartHoverPoint],
) -> None:
    """Register hover targets for left/right knee flexion time-series."""
    list_positions = list(range(len(series.times_s)))
    frame_indices = [
        pose_indices[i] if i < len(pose_indices) else None for i in range(len(series.times_s))
    ]
    append_line_hover_points(
        ax,
        series.times_s,
        series.left_deg,
        metric_name="Knee flexion",
        joint_name=LABEL_LEFT_KNEE,
        unit="deg",
        frame_indices=frame_indices,
        list_positions=list_positions,
        timestamps=series.times_s,
        hover_points=hover_points,
    )
    append_line_hover_points(
        ax,
        series.times_s,
        series.right_deg,
        metric_name="Knee flexion",
        joint_name=LABEL_RIGHT_KNEE,
        unit="deg",
        frame_indices=frame_indices,
        list_positions=list_positions,
        timestamps=series.times_s,
        hover_points=hover_points,
    )
