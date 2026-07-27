"""
3D trajectory plot for the StableWalk dashboard.

Default view (no joint selected): center-of-mass path + current full-body stick
figure.  Selected view: one coloured XYZ path and current-position dot per joint.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from stablewalk.analysis.ground_reference import GroundReferencePlane
    from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
else:
    from stablewalk.models.gait_motion import Vec3

from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES, ROOT_JOINT_ID
from stablewalk.ui.colors import ACCENT, ACCENT_ALT, BORDER, COM, INFO, MUTED, PANEL, TEXT, VIZ_JOINT, WARNING
from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, anchor_joint_for_item, label_for_item

# Stable gait-analysis camera (front-left oblique, Y-up)
_TRAJ_ELEV = 22.0
_TRAJ_AZIM = -62.0

# When pose "normalized" meters have an inflated vertical span (~2.5–3.0 instead
# of ~1.0), map them to a conventional adult stature so Overview cm labels are
# anatomically readable (knee ~−40 cm below pelvis, not ~−80 cm).
_REFERENCE_STATURE_M = 1.70

# Selected-point panel: oblique view; refined per-trajectory in _camera_for_single_dof_trajectory
_SINGLE_TRAJ_ELEV = 20.0
_SINGLE_TRAJ_AZIM = -60.0

TRAJECTORY_COLORS: tuple[str, ...] = (
    ACCENT,
    WARNING,
    ACCENT_ALT,
    INFO,
    VIZ_JOINT,
    "#8b9dc3",  # slate blue
    "#c49a6c",  # muted amber
    "#6fbf9a",  # soft teal
    "#d4a574",  # warm sand
    "#9bb0c7",  # steel
    "#7eb8d4",  # calm cyan
    "#b8a0c8",  # muted lilac
    "#c9b27c",  # soft gold
    "#94a3b8",  # gray
)

# Stick-figure bones for the default 3D pose overlay (parent, child)
_STICK_BONES: tuple[tuple[str, str], ...] = (
    (ROOT_JOINT_ID, "spine"),
    ("spine", "neck"),
    ("neck", "head"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
    ("spine", "left_shoulder"),
    ("spine", "right_shoulder"),
    (ROOT_JOINT_ID, "left_hip"),
    (ROOT_JOINT_ID, "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)

# Selected-point 3D panel — scientific biomechanics style (MATLAB / OpenSim-like)
from stablewalk.ui.theme import (
    DOF_TRAJ_DOT_COLOR,
    DOF_TRAJ_END_COLOR,
    DOF_TRAJ_PATH_COLOR,
    DOF_TRAJ_START_COLOR,
)

_CURRENT_DOT_COLOR = DOF_TRAJ_DOT_COLOR
_START_DOT_COLOR = DOF_TRAJ_START_COLOR
_END_DOT_COLOR = DOF_TRAJ_END_COLOR
_START_DOT_EDGE_COLOR = "#d5f5e3"
_END_DOT_EDGE_COLOR = "#d6eaf8"
_PATH_LINE_COLOR = DOF_TRAJ_PATH_COLOR
_PATH_OLD_COLOR = "#1e3548"
_CUBE_EDGE_COLOR = "#6a7f96"
_OVERVIEW_AXIS_X_COLOR = "#5a9ec4"
_OVERVIEW_AXIS_Y_COLOR = "#3d9a5f"
_OVERVIEW_AXIS_Z_COLOR = "#c9a227"
_CUBE_FACE_RGBA = (0.11, 0.14, 0.18, 0.0)
_GRID_RGBA = (0.55, 0.64, 0.74, 0.32)
_AXIS_RGBA = (0.86, 0.91, 0.96, 0.95)
# Glance-readable markers (scatter area points²) — keep small so the path dominates.
_START_DOT_SIZE = 36.0
_CURRENT_DOT_SIZE = 52.0
_END_DOT_SIZE = 32.0
# Instrument trail — thick enough for presentations, not cartoon-heavy.
_PATH_LINE_WIDTH = 2.85
_PATH_ALPHA = 0.97
# Earlier samples fade out; the path brightens toward the current frame.
_PATH_FADE_ALPHA_MIN = 0.32
_PATH_FADE_ALPHA_MAX = 1.0
_PATH_FADE_TAIL_FRAC = 0.26
# Full-recording base trail (Overview): always visible so the joint track
# matches the video even early in playback.
_FULL_PATH_LINE_WIDTH = 1.75
_FULL_PATH_ALPHA = 0.30
_FULL_PATH_COLOR = "#5a7a92"
_HINT_NO_SELECTION = "Select a joint to view its 3D path."
_SINGLE_TRAJ_TICKS = 4
# Path fills most of the cube with margin so tips/markers never clip.
_TRAJECTORY_TARGET_FILL = 0.76
_SINGLE_TRAJ_PADDING = (1.0 / _TRAJECTORY_TARGET_FILL - 1.0) * 0.5
# Lower floor on the per-axis view span: the cube zooms into whatever motion the
# point actually has, so even a small path (e.g. the near-rigid pelvis/hip) fills
# the view and reads as a visible line instead of a dot in an empty box.
_SINGLE_TRAJ_MIN_AXIS_SPAN = 0.005
# Total travel (body-normalized units; body height ~= 1.0, so this is ~fraction of
# body height) at/below which the view is clearly "zoomed in". We then show an
# honest scale note so the magnified path is not mistaken for large motion.
_SINGLE_TRAJ_SMALL_MOTION = 0.03
_SINGLE_TRAJ_BOX_ZOOM = 1.0
_SINGLE_TRAJ_BOX_ZOOM_SHORT = 0.94
# Enlarge the rendered 3D cube so the trajectory fills more of the panel.
_TRAJ_FILL_BOOST = 1.18
_TRAJ_BOX_ZOOM_CEIL_SINGLE = 1.0
# Closer camera = larger on-screen path (still leaves room for ticks).
_SINGLE_TRAJ_CAMERA_DIST = 5.8
_SINGLE_TRAJ_MARKER_SCALE_MAX = 1.20
_SINGLE_TRAJ_MARKER_SCALE_MIN = 0.92
# Overview: path fills a centred cube (equal XYZ scale) — never a speck in a huge box.
_OVERVIEW_MIN_AXIS_RATIO = 0.55
_OVERVIEW_PERSPECTIVE_ELEV = 20.0
_OVERVIEW_PERSPECTIVE_AZIM = -48.0
# Extra margin (fraction of span) so markers / smoothed overshoot stay inside.
_OVERVIEW_VIEWPORT_EDGE_PAD = 0.14
# Absolute pad + min cube side (meters) — enough air for markers without empty cube.
_OVERVIEW_MARKER_PAD_M = 0.012
_OVERVIEW_ABS_MIN_SPAN_M = 0.07
# Hard ceiling on Overview cube side so a long foot/world axis cannot empty the view.
_OVERVIEW_CUBE_SIDE_CAP_M = 0.30
# Named camera presets (elev, azim) for the Overview / Motion toolbars.
TRAJECTORY_CAMERA_PRESETS: dict[str, tuple[float, float]] = {
    "Perspective": (_OVERVIEW_PERSPECTIVE_ELEV, _OVERVIEW_PERSPECTIVE_AZIM),
    "Side": (10.0, 0.0),
    "Front": (10.0, -90.0),
    "Top": (88.0, -90.0),
}
_SINGLE_TRAJ_MARKER_LABEL_START = "Start"
_SINGLE_TRAJ_MARKER_LABEL_CURRENT = "Current"
_SINGLE_TRAJ_MARKER_LABEL_END = "End"
_DISPLAY_CURRENT_PROGRESS = "Current progress"
_DISPLAY_FULL_PATH = "Full path"
_DISPLAY_FULL_TRAJECTORY = "Full trajectory"
_COORD_ROOT_RELATIVE = "Root-relative"
_COORD_GLOBAL = "Global"
_PLANE_PROJECTION_3D = "3D"
_PATH_SHADOW_COLOR = "#5a6e84"
_PATH_SHADOW_ALPHA = 0.22
_PATH_SHADOW_WIDTH = 1.45
_PLANE_PROJECTION_FRONTAL = "Frontal Plane"
_PLANE_PROJECTION_SAGITTAL = "Sagittal Plane"
_PATH_DOT_SIZE_MIN = 7.0
_PATH_DOT_SIZE_MAX = 18.0
# Floor shadow of the path (min-Y plane) — depth anchor for 3D shape.

# Selected-point progress markers (adaptive Start / Middle; red dot = Current)
_TIME_MARKER_COLOR = "#8ea8c8"
_TIME_MARKER_SIZE = 22
_TIME_LABEL_COLOR = MUTED
_PROGRESS_LABEL_START = "Start"
_PROGRESS_LABEL_MIDDLE = "Middle"
_PROGRESS_MIN_FRAMES_FOR_START = 2
_PROGRESS_MIN_FRAMES_FOR_MIDDLE = 10
_PROGRESS_MIN_SEPARATION_RATIO = 0.14
_GROUND_PLANE_COLOR = "#4a5f75"
_GROUND_PLANE_ALPHA = 0.18
_GROUND_PLANE_EDGE = "#7a94ad"
_GROUND_PLANE_EDGE_ALPHA = 0.62
_GROUND_DROP_LINE = "#ffc857"
_GROUND_DROP_LINE_ALPHA = 0.88
# Oblique view that exposes vertical clearance (Y-up) and the ground plane.
_FOOT_VIEW_ELEV = 24.0
_FOOT_VIEW_AZIM = -52.0
_FOOT_VIEW_Y_MIN_SPAN = 0.07
_FOOT_VIEW_FLOOR_PAD_FRAC = 0.14


@dataclass(frozen=True)
class _SingleTrajViewport:
    """Centered axis limits, display aspect, and camera for one trajectory."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]
    zlim: tuple[float, float]
    box_aspect: tuple[float, float, float]
    elev: float
    azim: float


@dataclass(frozen=True)
class TrajectoryDrawResult:
    """Summary of what was drawn in the 3D trajectory panel."""

    joint_paths: int = 0
    default_view: bool = False
    has_motion: bool = False


def _view_init_y_up(ax: Axes, *, elev: float, azim: float) -> None:
    """Apply a camera with canonical +Y as the screen-up biomechanical axis."""
    try:
        ax.view_init(elev=elev, azim=azim, vertical_axis="y")
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def setup_trajectory_axes(ax: Axes, *, elev: float = _TRAJ_ELEV, azim: float = _TRAJ_AZIM) -> None:
    """Configure a readable 3D axes panel with fixed gait-analysis camera."""
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(PANEL)
    _view_init_y_up(ax, elev=elev, azim=azim)
    ax.set_xlabel("X · Lat (m)", color=_OVERVIEW_AXIS_X_COLOR, fontsize=9, labelpad=5)
    ax.set_ylabel("Y · Up (m)", color=_OVERVIEW_AXIS_Y_COLOR, fontsize=9, labelpad=5)
    ax.set_zlabel("Z · Fwd (m)", color=_OVERVIEW_AXIS_Z_COLOR, fontsize=9, labelpad=5)
    ax.tick_params(colors=MUTED, labelsize=7.5, pad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BORDER)
        try:
            axis.set_tick_params(label1On=True, label2On=False)
        except (AttributeError, TypeError, ValueError):
            pass
    ax.grid(True, color=BORDER, alpha=0.3, linestyle=":")
    _style_single_dof_trajectory_ticks(ax)

def estimate_body_height_m(recording: "GaitMotionRecording | None") -> float | None:
    """Median vertical span of the skeleton (head/ankle envelope) in recording meters."""
    if recording is None or getattr(recording, "frame_count", 0) <= 0:
        return None
    spans: list[float] = []
    step = max(1, int(recording.frame_count) // 24)
    for index in range(0, int(recording.frame_count), step):
        snap = recording.snapshot_at(index)
        if snap is None or not snap.joints:
            continue
        ys = [float(sample.position.y) for sample in snap.joints.values()]
        if len(ys) >= 2:
            spans.append(max(ys) - min(ys))
    if not spans:
        return None
    height = float(statistics.median(spans))
    return height if height > 0.15 else None


def stature_display_scale(recording: "GaitMotionRecording | None") -> float:
    """Map recording meters → display meters (1.70 m conventional stature).

    Some demo pose files are labeled ``positions_normalized`` but keep a raw
    vertical span of ~2.5–3.0 instead of ~1.0. Multiplying by 100 then labels
    knee height as ~−80 cm. Scaling by ``1.70 / body_height`` restores
    anatomically plausible centimeter readouts without inventing motion.
    """
    height = estimate_body_height_m(recording)
    if height is None or height < 0.15:
        return 1.0
    # Already stature-normalized (nose-to-ankle ≈ 1).
    if 0.85 <= height <= 1.20:
        return 1.0
    return _REFERENCE_STATURE_M / height


def _scale_vec(position: Vec3, scale: float) -> Vec3:
    if abs(scale - 1.0) < 1e-9:
        return position
    return Vec3(position.x * scale, position.y * scale, position.z * scale)


def meters_to_display_cm(meters: float, *, scale: float = 1.0) -> float:
    return float(meters) * float(scale) * 100.0


def _format_single_traj_tick(value: float, _pos: int) -> str:
    """Compact meter tick labels — adaptive precision, no trailing clutter."""
    if abs(value) < 1e-12:
        return "0"
    av = abs(value)
    if av >= 10.0:
        text = f"{value:.1f}"
    elif av >= 1.0:
        text = f"{value:.2f}"
    elif av >= 0.1:
        text = f"{value:.2f}"
    elif av >= 0.01:
        text = f"{value:.3f}"
    else:
        text = f"{value:.3f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_overview_cm_tick(value: float, _pos: int) -> str:
    """Overview sidebar: axis values in centimeters with adaptive decimals."""
    if abs(value) < 1e-12:
        return "0"
    cm = value * 100.0
    # Prefer whole centimetres when within 0.05 cm of an integer (lab-readable).
    if abs(cm - round(cm)) < 0.05:
        cm = float(round(cm))
    av = abs(cm)
    if av >= 100.0:
        text = f"{cm:.0f}"
    elif av >= 10.0:
        text = f"{cm:.0f}" if abs(cm - round(cm)) < 1e-6 else f"{cm:.1f}"
    elif av >= 1.0:
        text = f"{cm:.0f}" if abs(cm - round(cm)) < 1e-6 else f"{cm:.1f}"
    else:
        text = f"{cm:.2f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    # Keep an explicit ASCII minus so negative heights (knee below pelvis)
    # cannot be misread as positive after mplot3d projection.
    if text.startswith("-"):
        return text
    if cm < 0 and not text.startswith("-"):
        return f"-{text}"
    return text or "0"


def _nice_number(span: float, *, round_up: bool) -> float:
    """Return a 1–2–5 × 10^n step near ``span`` (laboratory tick spacing)."""
    if not math.isfinite(span) or span <= 0.0:
        return 1.0
    exp = math.floor(math.log10(span))
    frac = span / (10.0**exp)
    if round_up:
        if frac <= 1.0:
            nice = 1.0
        elif frac <= 2.0:
            nice = 2.0
        elif frac <= 5.0:
            nice = 5.0
        else:
            nice = 10.0
    else:
        if frac < 1.5:
            nice = 1.0
        elif frac < 3.0:
            nice = 2.0
        elif frac < 7.0:
            nice = 5.0
        else:
            nice = 10.0
    return nice * (10.0**exp)


def _adaptive_tick_target(span_m: float, *, use_cm: bool) -> int:
    """Fewer major ticks on short axes so labels never crowd."""
    span_disp = abs(span_m) * (100.0 if use_cm else 1.0)
    # Tiny spans (e.g. knee Y ≈ 2–8 cm): only endpoints — 1 cm ticks overlap.
    if span_disp < 8.0:
        return 2
    if span_disp < 16.0:
        return 3
    return 4


def _overview_tick_values(
    lo: float, hi: float, *, use_cm: bool = False, target: int = 4
) -> list[float]:
    """Adaptive nice major ticks with unique labels (no overlapping numbers)."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0]
    if abs(hi - lo) < 1e-15:
        return [float(lo)]

    scale = 100.0 if use_cm else 1.0
    lo_d = float(lo) * scale
    hi_d = float(hi) * scale
    if hi_d < lo_d:
        lo_d, hi_d = hi_d, lo_d
    span_d = hi_d - lo_d

    # Short axes: exactly two readable endpoints (avoids -43/-44/-45 pile-up).
    n_target = max(2, min(int(target), 4))
    if n_target <= 2 or span_d < 12.0:
        # Nice 1–2–5 step so labels read as -50 / -40, not -48.1 / -40.1.
        step = _nice_number(max(span_d / 2.0, 2.0 if use_cm else 0.02), round_up=True)
        nice_lo = math.floor(lo_d / step) * step
        nice_hi = math.ceil(hi_d / step) * step
        if nice_hi <= nice_lo:
            nice_hi = nice_lo + step
        # Prefer ticks that bracket the axis range with distinct formatted labels.
        lo_m, hi_m = nice_lo / scale, nice_hi / scale
        fmt = _format_overview_cm_tick if use_cm else _format_single_traj_tick
        if fmt(lo_m, 0) == fmt(hi_m, 0):
            return [float(lo), float(hi)]
        return [lo_m, hi_m]

    rough = span_d / max(n_target - 1, 1)
    step = _nice_number(rough, round_up=True)
    if step <= 0.0:
        step = rough if rough > 0.0 else 1.0

    # Prefer ticks that sit on nice multiples and stay inside the axis range.
    start = math.ceil((lo_d - step * 1e-9) / step) * step
    ticks_d: list[float] = []
    value = start
    guard = 0
    while value <= hi_d + step * 1e-9 and guard < 64:
        if lo_d - step * 1e-6 <= value <= hi_d + step * 1e-6:
            ticks_d.append(float(value))
        value += step
        guard += 1

    # Ensure endpoints participate when the nice grid left a large empty margin.
    if not ticks_d:
        ticks_d = [lo_d, hi_d]
    else:
        if abs(ticks_d[0] - lo_d) > step * 0.85:
            ticks_d.insert(0, lo_d)
        if abs(ticks_d[-1] - hi_d) > step * 0.85:
            ticks_d.append(hi_d)

    # Drop near-duplicates and labels that would collide after formatting.
    fmt = _format_overview_cm_tick if use_cm else _format_single_traj_tick
    cleaned: list[float] = []
    seen: set[str] = set()
    # Require ~span/2.5 separation so labels never stack on a short visual edge.
    min_sep = max(span_d / 2.5, step * 0.9)
    for raw in ticks_d:
        meters = raw / scale
        label = fmt(meters, 0)
        if not label or label in seen:
            continue
        if cleaned and abs(raw - cleaned[-1] * scale) < min_sep * 0.55:
            continue
        cleaned.append(meters)
        seen.add(label)

    if len(cleaned) < 2:
        return [float(lo), float(hi)]

    max_ticks = max(2, min(n_target, 4))
    if len(cleaned) > max_ticks:
        if max_ticks == 2:
            cleaned = [cleaned[0], cleaned[-1]]
        elif max_ticks == 3:
            cleaned = [cleaned[0], cleaned[len(cleaned) // 2], cleaned[-1]]
        else:
            idxs = [
                round(i * (len(cleaned) - 1) / (max_ticks - 1))
                for i in range(max_ticks)
            ]
            cleaned = [cleaned[i] for i in dict.fromkeys(idxs)]
    return cleaned


def _percentile_axis_limits(
    values: list[float],
    *,
    pad_frac: float = 0.12,
    min_span: float = 0.004,
    low_pct: float = 0.02,
    high_pct: float = 0.98,
) -> tuple[float, float]:
    """
    Robust axis limits that ignore single-frame pose spikes.

    Limits are centred on the median so a one-sided low/high tail does not
    push the bulk of the trajectory into a corner of the cube.
    """
    clean = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    if not clean:
        return -min_span * 0.5, min_span * 0.5
    ordered = sorted(clean)
    n = len(ordered)
    med = statistics.median(clean)
    if n < 6:
        lo_p, hi_p = ordered[0], ordered[-1]
    else:
        # Index by floor((n-1)*pct) so a single high spike is excluded at P98.
        lo_p = ordered[max(0, int(math.floor((n - 1) * low_pct)))]
        hi_p = ordered[min(n - 1, int(math.floor((n - 1) * high_pct)))]
        if hi_p < lo_p:
            lo_p, hi_p = ordered[0], ordered[-1]
    half = max(abs(med - lo_p), abs(hi_p - med), min_span * 0.5)
    half = max(half * (1.0 + pad_frac), min_span * 0.5)
    return med - half, med + half


def _apply_overview_trajectory_ticks(ax: Axes) -> None:
    """
    Overview dock: adaptive nice ticks and single-sided labels.

    mplot3d otherwise draws the same tick value on multiple cube edges, which
    looks like duplicated / overlapping numbers.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    use_cm = bool(getattr(ax, "_stablewalk_overview_cm_ticks", False))
    tick_fmt = _format_overview_cm_tick if use_cm else _format_single_traj_tick

    for axis, get_lim in (
        (ax.xaxis, ax.get_xlim),
        (ax.yaxis, ax.get_ylim),
        (ax.zaxis, ax.get_zlim),
    ):
        lo, hi = get_lim()
        target = _adaptive_tick_target(hi - lo, use_cm=use_cm)
        ticks = _overview_tick_values(lo, hi, use_cm=use_cm, target=target)
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(tick_fmt))
        axis.set_tick_params(
            label1On=True,
            label2On=False,
            colors=TEXT,
            labelsize=7.0,
            pad=6,
            length=2.5,
            width=0.65,
        )
        # Extra guard against mirrored cube-edge labels in mplot3d.
        try:
            axis.set_ticks_position("default")
        except (AttributeError, TypeError, ValueError):
            pass
        try:
            axis._axinfo["tick"]["inward_factor"] = 0.15
            axis._axinfo["tick"]["outward_factor"] = 0.15
        except (AttributeError, KeyError, TypeError):
            pass
    try:
        ax.tick_params(axis="x", pad=6, labelsize=7.0)
        ax.tick_params(axis="y", pad=7, labelsize=7.0)
        ax.tick_params(axis="z", pad=7, labelsize=7.0)
    except (TypeError, ValueError):
        pass
    # Force redraw of tick labels (mplot3d sometimes drops one face).
    try:
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_rotate_label(False)
        ax.zaxis.set_rotate_label(False)
    except Exception:
        pass


def _style_overview_trajectory_cube(ax: Axes) -> None:
    """Scientific cube: open panes + stronger grid for glance readability."""
    grid = (0.58, 0.66, 0.76, 0.42)
    edge = (0.62, 0.70, 0.80, 0.70)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(_CUBE_EDGE_COLOR)
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"]["color"] = grid
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["grid"]["linewidth"] = 0.95
        axis._axinfo["axisline"]["color"] = edge
        axis._axinfo["axisline"]["linewidth"] = 1.35
    ax.grid(True, color=grid, alpha=0.40, linestyle="-", linewidth=0.95)


def _style_single_dof_cube(ax: Axes) -> None:
    """Open panes and stronger grid for a readable laboratory 3D box."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(_CUBE_EDGE_COLOR)
        axis.pane.set_alpha(0.0)
        axis._axinfo["grid"]["color"] = _GRID_RGBA
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["grid"]["linewidth"] = 0.70
        axis._axinfo["axisline"]["color"] = _AXIS_RGBA
        axis._axinfo["axisline"]["linewidth"] = 1.15
    ax.grid(True, color=_GRID_RGBA, alpha=0.32, linestyle="-", linewidth=0.70)


def _style_single_dof_trajectory_ticks(ax: Axes) -> None:
    """Readable adaptive ticks without overcrowding the 3D axes."""
    if bool(getattr(ax, "_stablewalk_overview_dock", False)):
        _apply_overview_trajectory_ticks(ax)
        return
    if bool(getattr(ax, "_stablewalk_motion_dock", False)):
        _apply_overview_trajectory_ticks(ax)
        return

    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    use_cm = bool(getattr(ax, "_stablewalk_overview_cm_ticks", False))
    tick_fmt = _format_overview_cm_tick if use_cm else _format_single_traj_tick
    for axis, get_lim in (
        (ax.xaxis, ax.get_xlim),
        (ax.yaxis, ax.get_ylim),
        (ax.zaxis, ax.get_zlim),
    ):
        lo, hi = get_lim()
        target = _adaptive_tick_target(hi - lo, use_cm=use_cm)
        ticks = _overview_tick_values(lo, hi, use_cm=use_cm, target=target)
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_major_formatter(FuncFormatter(tick_fmt))
        axis.set_tick_params(
            label1On=True,
            label2On=False,
            colors=TEXT,
            labelsize=9,
            pad=4,
            length=4,
            width=0.75,
        )
    try:
        ax.tick_params(axis="z", pad=5)
    except (TypeError, ValueError):
        pass


def _equal_box_aspect_from_limits(
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> tuple[float, float, float]:
    """Isotropic scientific scale: 1 m maps to the same visual length on XYZ."""
    return (
        max(abs(xlim[1] - xlim[0]), 1e-9),
        max(abs(ylim[1] - ylim[0]), 1e-9),
        max(abs(zlim[1] - zlim[0]), 1e-9),
    )


def _trajectory_axis_titles(
    *,
    use_cm: bool,
    overview: bool,
) -> tuple[tuple[str, str, str], float, tuple[float, float, float]]:
    """Clean X/Y/Z titles — short, unit-explicit, no overlapping long phrases."""
    if use_cm:
        titles = ("X (cm)", "Y (cm)", "Z (cm)")
    elif overview:
        titles = ("X (m)", "Y (m)", "Z (m)")
    else:
        titles = ("X · Lat (m)", "Y · Up (m)", "Z · Fwd (m)")
    if overview:
        return titles, 7.5, (4.0, 5.0, 5.0)
    return titles, 9.0, (7.0, 7.0, 9.0)


def setup_single_dof_trajectory_axes(
    ax: Axes,
    *,
    elev: float = _SINGLE_TRAJ_ELEV,
    azim: float = _SINGLE_TRAJ_AZIM,
) -> None:
    """Readable 3D cube axes for the dashboard selected-point trajectory panel."""
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(PANEL)
    user_cam = getattr(ax, "_stablewalk_user_camera", None)
    if isinstance(user_cam, (tuple, list)) and len(user_cam) == 2:
        elev, azim = float(user_cam[0]), float(user_cam[1])
    _view_init_y_up(ax, elev=elev, azim=azim)
    overview = bool(getattr(ax, "_stablewalk_overview_dock", False))
    use_cm = bool(getattr(ax, "_stablewalk_overview_cm_ticks", False))
    (xlab, ylab, zlab), label_fs, (xpad, ypad, zpad) = _trajectory_axis_titles(
        use_cm=use_cm,
        overview=overview or bool(getattr(ax, "_stablewalk_motion_dock", False)),
    )
    ax.set_xlabel(xlab, color=_OVERVIEW_AXIS_X_COLOR, fontsize=label_fs, labelpad=xpad, fontweight="medium")
    ax.set_ylabel(ylab, color=_OVERVIEW_AXIS_Y_COLOR, fontsize=label_fs, labelpad=ypad, fontweight="medium")
    ax.set_zlabel(zlab, color=_OVERVIEW_AXIS_Z_COLOR, fontsize=label_fs, labelpad=zpad, fontweight="medium")
    if overview:
        _style_overview_trajectory_cube(ax)
    else:
        _style_single_dof_cube(ax)
    _style_single_dof_trajectory_ticks(ax)


def _ensure_trajectory_plot_legend(ax: Axes) -> None:
    """Persistent Start / Path / Now / Floor legend (recreated after ax.cla())."""
    if getattr(ax, "_stablewalk_overview_dock", False) or getattr(
        ax, "_stablewalk_motion_dock", False
    ):
        # Overview & Motion docks show an external Tk legend beside/below the
        # canvas, so the in-figure legend is redundant — skip it so nothing
        # overlaps the trajectory.
        existing = getattr(ax, "_stablewalk_plot_legend", None)
        if existing is not None:
            try:
                existing.set_visible(False)
            except Exception:
                pass
        return
    if getattr(ax, "_stablewalk_plot_legend", None) is not None:
        try:
            ax._stablewalk_plot_legend.set_visible(True)
        except Exception:
            pass
        return
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    proxies = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_START_DOT_COLOR,
            markeredgecolor=_START_DOT_EDGE_COLOR,
            markeredgewidth=1.0,
            markersize=7,
            linestyle="None",
            label=_SINGLE_TRAJ_MARKER_LABEL_START,
        ),
        Line2D(
            [0],
            [0],
            color=_PATH_LINE_COLOR,
            linewidth=_PATH_LINE_WIDTH,
            label="Path (fade→bright)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_CURRENT_DOT_COLOR,
            markeredgecolor="#fff0f2",
            markeredgewidth=1.0,
            markersize=7,
            linestyle="None",
            label="Current frame",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=_END_DOT_COLOR,
            markeredgecolor=_END_DOT_EDGE_COLOR,
            markeredgewidth=1.0,
            markersize=7,
            linestyle="None",
            label=_SINGLE_TRAJ_MARKER_LABEL_END,
        ),
        Patch(
            facecolor=_GROUND_PLANE_COLOR,
            edgecolor=_GROUND_PLANE_EDGE,
            alpha=_GROUND_PLANE_ALPHA + 0.25,
            label="Floor",
        ),
    ]
    leg = ax.legend(
        handles=proxies,
        loc="upper left",
        fontsize=7,
        framealpha=0.82,
        facecolor=PANEL,
        edgecolor=BORDER,
        labelcolor=TEXT,
        borderpad=0.35,
        labelspacing=0.28,
        handletextpad=0.4,
        handlelength=1.3,
        borderaxespad=0.15,
    )
    ax._stablewalk_plot_legend = leg


def _single_traj_visual_scale(ax: Axes) -> float:
    """Scale marker area with figure size — keep markers subordinate to the path."""
    fig = ax.figure
    w_in, h_in = fig.get_size_inches()
    raw = min(w_in, h_in) / 4.6
    scale = max(_SINGLE_TRAJ_MARKER_SCALE_MIN, min(_SINGLE_TRAJ_MARKER_SCALE_MAX, raw))
    if getattr(ax, "_stablewalk_overview_dock", False):
        # Overview panel is small: prefer a thin path with pin-head markers.
        scale = min(max(scale * 0.82, 0.88), 1.02)
    return scale


def _single_dof_figure_margins(
    w_in: float, h_in: float, *, dpi: float = 100.0, foot_mode: bool = False
) -> tuple[float, float, float, float]:
    """
    Figure-fraction margins for a single 3D axes.

    mplot3d draws axis labels and cube corners outside the axes bbox; reserve
    extra space at the bottom and sides so nothing is clipped.
    """
    h_px = h_in * dpi

    # Reserve a band at the bottom for the X/Y axis labels (drawn below the
    # projected cube) and keep a thin top margin. Shorter panels need a
    # proportionally larger band so the descriptive labels are never clipped.
    if h_px < 300:
        bottom, top = 0.09, 0.995
    elif h_px < 440:
        bottom, top = 0.07, 0.995
    else:
        bottom, top = 0.055, 0.995

    if foot_mode:
        bottom = min(bottom + 0.025, 0.20)

    band_h = max(top - bottom, 0.2)
    target_px_ratio = 1.6  # axes width:height in pixels
    width_frac = target_px_ratio * band_h * h_in / max(w_in, 0.1)
    width_frac = max(0.90, min(0.985, width_frac))
    span = 1.0 - width_frac
    # Centre the axes horizontally so the projected cube sits mid-panel instead
    # of drifting to one side (leaving a large empty margin).
    left = span * 0.5
    right = left + width_frac

    return left, bottom, right, top


def _overview_dof_figure_margins(
    w_in: float, h_in: float, *, dpi: float = 100.0
) -> tuple[float, float, float, float]:
    """
    Margins for the Overview 3D Joint Path panel.

    Leave room for axis tick labels so the path cube is not flush with the
    panel edge (which also visually clips the trajectory tips).
    """
    h_px = max(1.0, h_in * dpi)
    w_px = max(1.0, w_in * dpi)

    if h_px < 260:
        bottom, top = 0.18, 0.965
    elif h_px < 360:
        bottom, top = 0.15, 0.972
    elif h_px < 480:
        bottom, top = 0.12, 0.978
    else:
        bottom, top = 0.10, 0.982

    if w_px < 260:
        left, right = 0.11, 0.90
    elif w_px < 360:
        left, right = 0.08, 0.925
    else:
        left, right = 0.06, 0.945

    return left, bottom, right, top


def _adaptive_camera_base_dist(ax: Axes) -> float:
    """Closer camera for small-motion paths so the trail fills the panel."""
    overview = bool(getattr(ax, "_stablewalk_overview_dock", False))
    # Overview: slightly farther so perspective doesn't crop path tips.
    base = 6.35 if overview else _SINGLE_TRAJ_CAMERA_DIST
    try:
        sx = abs(float(ax.get_xlim()[1]) - float(ax.get_xlim()[0]))
        sy = abs(float(ax.get_ylim()[1]) - float(ax.get_ylim()[0]))
        sz = abs(float(ax.get_zlim()[1]) - float(ax.get_zlim()[0]))
        max_span = max(sx, sy, sz)
    except Exception:
        return base
    # Small ROM joints (hip/knee in pelvis frame) get a closer camera.
    if max_span < 0.035:
        base *= 0.88 if overview else 0.82
    elif max_span < 0.070:
        base *= 0.94 if overview else 0.90
    elif max_span > 0.25:
        base *= 1.06
    # Tall/narrow panels: nudge closer so the cube doesn't look tiny.
    try:
        w_in, h_in = ax.figure.get_size_inches()
        if min(w_in, h_in) < 3.2:
            base *= 0.94 if overview else 0.92
    except Exception:
        pass
    return base


def _apply_single_dof_camera(ax: Axes) -> None:
    """Adaptive camera distance: larger on-screen path, labels still in frame."""
    try:
        base = _adaptive_camera_base_dist(ax)
        zoom = float(getattr(ax, "_stablewalk_camera_zoom", 1.0) or 1.0)
        zoom = max(0.35, min(3.5, zoom))
        # Smaller dist = closer (more zoomed in).
        ax.dist = base / zoom
    except AttributeError:
        pass


def remember_trajectory_camera(ax: Axes) -> None:
    """Store the current elev/azim so playback redraws do not fight mouse orbit."""
    try:
        elev = float(ax.elev)
        azim = float(ax.azim)
    except Exception:
        return
    ax._stablewalk_user_camera = (elev, azim)  # type: ignore[attr-defined]


def clear_trajectory_camera_state(ax: Axes | None) -> None:
    """Drop manual orbit / zoom / pan so the next draw restores the default view."""
    if ax is None:
        return
    for attr in (
        "_stablewalk_user_camera",
        "_stablewalk_camera_zoom",
        "_stablewalk_pan_offset",
    ):
        if hasattr(ax, attr):
            try:
                delattr(ax, attr)
            except Exception:
                setattr(ax, attr, None)


def reset_trajectory_camera(ax: Axes, *, projection_mode: str = _PLANE_PROJECTION_3D) -> None:
    """Reset orbit, zoom, and pan to the recommended Perspective view.

    Overview docks re-apply an adaptive Perspective on the next redraw (planar
    joint paths tip the camera down so Y still reads as vertical).
    """
    clear_trajectory_camera_state(ax)
    if bool(getattr(ax, "_stablewalk_overview_dock", False)) or bool(
        getattr(ax, "_stablewalk_motion_dock", False)
    ):
        elev, azim = TRAJECTORY_CAMERA_PRESETS["Perspective"]
    else:
        elev, azim = _view_angles_for_projection(projection_mode)
    try:
        _view_init_y_up(ax, elev=elev, azim=azim)
    except Exception:
        pass
    # Leave ``_stablewalk_user_camera`` cleared so the next draw reapplies the
    # recommended Perspective viewport (presets set the user camera explicitly).
    _apply_single_dof_camera(ax)


def set_trajectory_camera_preset(ax: Axes, preset: str) -> None:
    """Apply a named camera preset (Perspective / Side / Front / Top)."""
    elev, azim = TRAJECTORY_CAMERA_PRESETS.get(
        preset, TRAJECTORY_CAMERA_PRESETS["Perspective"]
    )
    clear_trajectory_camera_state(ax)
    try:
        _view_init_y_up(ax, elev=float(elev), azim=float(azim))
    except Exception:
        pass
    ax._stablewalk_user_camera = (float(elev), float(azim))  # type: ignore[attr-defined]
    _apply_single_dof_camera(ax)


def zoom_trajectory_camera(ax: Axes, factor: float) -> None:
    """Zoom in (factor > 1) or out (factor < 1) without changing elev/azim."""
    current = float(getattr(ax, "_stablewalk_camera_zoom", 1.0) or 1.0)
    ax._stablewalk_camera_zoom = max(0.35, min(3.5, current * float(factor)))  # type: ignore[attr-defined]
    remember_trajectory_camera(ax)
    _apply_single_dof_camera(ax)


def pan_trajectory_camera(ax: Axes, *, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> None:
    """Nudge the view center in data units (floor plane = XZ; dy is vertical)."""
    ox, oy, oz = getattr(ax, "_stablewalk_pan_offset", (0.0, 0.0, 0.0))
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        zlim = ax.get_zlim()
    except Exception:
        return
    span = max(
        abs(xlim[1] - xlim[0]),
        abs(ylim[1] - ylim[0]),
        abs(zlim[1] - zlim[0]),
        1e-3,
    )
    step = span * 0.08
    sx, sy, sz = float(dx) * step, float(dy) * step, float(dz) * step
    ax._stablewalk_pan_offset = (  # type: ignore[attr-defined]
        float(ox) + sx,
        float(oy) + sy,
        float(oz) + sz,
    )
    remember_trajectory_camera(ax)
    try:
        ax.set_xlim(xlim[0] + sx, xlim[1] + sx)
        ax.set_ylim(ylim[0] + sy, ylim[1] + sy)
        ax.set_zlim(zlim[0] + sz, zlim[1] + sz)
    except Exception:
        pass


def rotate_trajectory_camera(
    ax: Axes,
    *,
    d_elev: float = 0.0,
    d_azim: float = 0.0,
) -> None:
    """Nudge orbit angles; mouse drag rotation is also remembered on release."""
    try:
        elev = float(ax.elev) + float(d_elev)
        azim = float(ax.azim) + float(d_azim)
    except Exception:
        elev, azim = _SINGLE_TRAJ_ELEV + d_elev, _SINGLE_TRAJ_AZIM + d_azim
    elev = max(-89.0, min(89.0, elev))
    _view_init_y_up(ax, elev=elev, azim=azim)
    remember_trajectory_camera(ax)


def _apply_pan_offset_to_limits(ax: Axes) -> None:
    offset = getattr(ax, "_stablewalk_pan_offset", None)
    if not isinstance(offset, (tuple, list)) or len(offset) != 3:
        return
    ox, oy, oz = (float(offset[0]), float(offset[1]), float(offset[2]))
    if abs(ox) < 1e-12 and abs(oy) < 1e-12 and abs(oz) < 1e-12:
        return
    try:
        xl, yl, zl = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax.set_xlim(xl[0] + ox, xl[1] + ox)
        ax.set_ylim(yl[0] + oy, yl[1] + oy)
        ax.set_zlim(zl[0] + oz, zl[1] + oz)
    except Exception:
        pass


def _resolve_draw_camera(
    ax: Axes,
    projection_mode: str,
    spans: tuple[float, float, float],
) -> tuple[float, float]:
    """Prefer the user's orbit; otherwise pick a readable Overview Perspective."""
    user_cam = getattr(ax, "_stablewalk_user_camera", None)
    if isinstance(user_cam, (tuple, list)) and len(user_cam) == 2:
        return float(user_cam[0]), float(user_cam[1])
    if bool(getattr(ax, "_stablewalk_overview_dock", False)) or bool(
        getattr(ax, "_stablewalk_motion_dock", False)
    ):
        if projection_mode == _PLANE_PROJECTION_3D:
            # Match the stable viewport: tip down for planar (hip/knee) paths.
            return _overview_camera_for_spans(spans)
    return _view_angles_for_projection(projection_mode, spans)


def _layout_single_dof_figure(ax: Axes, *, foot_mode: bool = False) -> None:
    """Lay out the 3D axes with generous margins so labels and cube edges fit."""
    fig = ax.figure
    # Never mix constrained_layout with manual subplots_adjust — it re-clips the cube.
    try:
        fig.set_layout_engine(None)
    except Exception:
        try:
            fig.set_constrained_layout(False)
            fig.set_tight_layout(False)
        except Exception:
            pass
    w_in, h_in = fig.get_size_inches()
    if getattr(ax, "_stablewalk_overview_dock", False):
        left, bottom, right, top = _overview_dof_figure_margins(
            w_in, h_in, dpi=fig.get_dpi()
        )
    else:
        left, bottom, right, top = _single_dof_figure_margins(
            w_in, h_in, dpi=fig.get_dpi(), foot_mode=foot_mode
        )
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    fig.patch.set_facecolor(PANEL)
    try:
        import matplotlib as mpl

        mpl.rcParams["lines.antialiased"] = True
        mpl.rcParams["path.simplify"] = False
    except Exception:
        pass
    _apply_single_dof_camera(ax)


def relayout_single_dof_viewport(ax: Axes) -> None:
    """Reflow margins after the dashboard canvas or axis limits change size."""
    foot_mode = bool(getattr(ax, "_stablewalk_foot_view", False))
    _layout_single_dof_figure(ax, foot_mode=foot_mode)
    try:
        current_aspect = tuple(float(v) for v in ax.get_box_aspect())
        w_in, h_in = ax.figure.get_size_inches()
        zoom = _single_traj_box_zoom(
            w_in,
            h_in,
            dpi=ax.figure.get_dpi(),
            overview_dock=bool(getattr(ax, "_stablewalk_overview_dock", False)),
        )
        ax.set_box_aspect(current_aspect, zoom=zoom)
    except (AttributeError, TypeError, ValueError):
        pass


def _single_traj_box_zoom(
    w_in: float,
    h_in: float,
    *,
    dpi: float = 100.0,
    overview_dock: bool = False,
) -> float:
    """Size-adaptive 3D cube scale — Overview keeps margin so the path never clips."""
    h_px = h_in * dpi
    aspect = w_in / max(h_in, 0.1)
    if overview_dock:
        w_px = w_in * dpi
        # Keep the cube well inside the axes so markers clear the panel edge.
        if h_px < 220 or w_px < 210:
            return 0.68
        if h_px < 280 or aspect > 1.55 or aspect < 0.64:
            return 0.74
        return 0.78
    # Motion / single dock — fill toward the bbox, bounded so labels never clip.
    if aspect > 2.5:
        base = 0.72
    elif h_px < 280:
        base = _SINGLE_TRAJ_BOX_ZOOM_SHORT
    elif h_px < 420:
        base = 0.88
    else:
        base = _SINGLE_TRAJ_BOX_ZOOM
    return max(0.55, min(_TRAJ_BOX_ZOOM_CEIL_SINGLE, base * _TRAJ_FILL_BOOST))


def _camera_for_single_dof_trajectory(
    spans: tuple[float, float, float],
) -> tuple[float, float]:
    """
    Pick a viewing angle that exposes the plane where most movement occurs.

    The axis with the largest data span sets the primary motion direction; flat
    axes keep their true scale but are not used to force an end-on view.
    """
    sx, sy, sz = spans
    max_span = max(sx, sy, sz)
    if max_span <= 0.0:
        return _SINGLE_TRAJ_ELEV, _SINGLE_TRAJ_AZIM

    dominant = spans.index(max_span)

    if dominant == 0:
        # Forward / lateral path — oblique horizontal view (typical gait).
        return 23.0, -56.0
    if dominant == 1:
        # Vertical motion dominates.
        return 16.0, -88.0
    # Depth (Z) carries most of the movement.
    return 26.0, -36.0


def _viewport_for_single_dof_trajectory(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    *,
    floor_y: float | None = None,
) -> _SingleTrajViewport:
    """
    Fit each axis independently to the trajectory min/max with small padding.

    Limits follow the actual X/Y/Z data range so the path fills the view
    instead of sitting inside a large empty cube. When ``floor_y`` is set,
    extra padding below the ground plane keeps the clearance line in view.
    """
    limits: list[tuple[float, float]] = []
    extents: list[float] = []
    spans: list[float] = []

    for axis_idx, vals in enumerate((xs, ys, zs)):
        lo, hi = min(vals), max(vals)
        if axis_idx == 1 and floor_y is not None:
            lo = min(lo, floor_y)
            floor_pad = max((hi - lo) * 0.08, 0.014)
            lo -= floor_pad
        raw_span = hi - lo
        span = max(raw_span, _SINGLE_TRAJ_MIN_AXIS_SPAN)
        margin = span * _SINGLE_TRAJ_PADDING
        axis_lo = lo - margin
        axis_hi = hi + margin
        limits.append((axis_lo, axis_hi))
        extents.append(axis_hi - axis_lo)
        spans.append(span)

    elev, azim = _camera_for_single_dof_trajectory((spans[0], spans[1], spans[2]))
    if floor_y is not None:
        elev, azim = _FOOT_VIEW_ELEV, _FOOT_VIEW_AZIM

    max_extent = max(extents)
    cubic_limits: list[tuple[float, float]] = []
    for axis_idx, ((lo, hi), extent) in enumerate(zip(limits, extents, strict=True)):
        mid = (lo + hi) * 0.5
        half = max_extent * 0.5
        cubic_lo = mid - half
        cubic_hi = mid + half
        if axis_idx == 1 and floor_y is not None:
            cubic_lo = min(cubic_lo, floor_y - max_extent * _FOOT_VIEW_FLOOR_PAD_FRAC)
            y_span = cubic_hi - cubic_lo
            if y_span < _FOOT_VIEW_Y_MIN_SPAN:
                mid_y = (cubic_lo + cubic_hi) * 0.5
                half = _FOOT_VIEW_Y_MIN_SPAN * 0.5
                cubic_lo, cubic_hi = mid_y - half, mid_y + half
        cubic_limits.append((cubic_lo, cubic_hi))

    return _SingleTrajViewport(
        xlim=cubic_limits[0],
        ylim=cubic_limits[1],
        zlim=cubic_limits[2],
        box_aspect=(max_extent, max_extent, max_extent),
        elev=elev,
        azim=azim,
    )


def _balanced_box_aspect(spans: tuple[float, float, float]) -> tuple[float, float, float]:
    """Balanced scientific aspect with near-flat dimensions clamped to 30%."""
    sx, sy, sz = (max(float(s), 1e-9) for s in spans)
    longest = max(sx, sy, sz)
    floor = longest * _OVERVIEW_MIN_AXIS_RATIO
    return (max(sx, floor), max(sy, floor), max(sz, floor))


def _overview_camera_for_spans(spans: tuple[float, float, float]) -> tuple[float, float]:
    """
    Recommended Perspective so world +Y stays screen-up.

    Keep elevation modest. High elev looks down onto the XZ walk plane, which
    makes the Z spine appear upright and the path look like a flat ribbon.
    """
    sx, sy, sz = (max(float(s), 1e-9) for s in spans)
    elev, azim = TRAJECTORY_CAMERA_PRESETS["Perspective"]
    # When vertical motion dominates, lift slightly so clearance is visible.
    if sy > 1.15 * max(sx, sz):
        elev = 28.0
        azim = -55.0
    return elev, azim


def _path_max_span(path: list[Vec3]) -> float:
    if not path:
        return _SINGLE_TRAJ_MIN_AXIS_SPAN
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    zs = [p.z for p in path]
    return max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs),
        _SINGLE_TRAJ_MIN_AXIS_SPAN,
    )


def _point_distance(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _remove_trajectory_outliers(path: list[Vec3]) -> list[Vec3]:
    """Drop single-frame pose spikes that create scribble in small-ROM paths."""
    if len(path) < 4:
        return path
    rom = _path_max_span(path)
    max_step = max(rom * 0.42, 0.0025)
    cleaned = [path[0]]
    for point in path[1:]:
        if _point_distance(cleaned[-1], point) <= max_step:
            cleaned.append(point)
    # Always keep the live tip — dropping it makes the red marker fall outside
    # the fitted cube while the footer still reports the tip coordinates.
    if path[-1] is not cleaned[-1]:
        cleaned.append(path[-1])
    working = cleaned if len(cleaned) >= max(3, len(path) // 3) else list(path)
    if len(working) < 8:
        return working
    filtered = working
    for axis_getter in (
        lambda p: p.x,
        lambda p: p.y,
        lambda p: p.z,
    ):
        vals = [axis_getter(p) for p in filtered]
        lo, hi = _percentile_axis_limits(vals, pad_frac=0.0, min_span=1e-6)
        margin = max(hi - lo, 1e-6) * 0.25
        kept = [
            p
            for p in filtered
            if lo - margin <= axis_getter(p) <= hi + margin
        ]
        # Preserve start + tip even when they sit near the robust envelope edge.
        if working[0] not in kept:
            kept.insert(0, working[0])
        if working[-1] not in kept:
            kept.append(working[-1])
        filtered = kept
    return filtered if len(filtered) >= 3 else working


def _smooth_trajectory_path_light(path: list[Vec3], *, window: int = 3) -> list[Vec3]:
    """Display-only moving average — keeps endpoints, softens tracking jitter."""
    if len(path) < window:
        return path
    half = window // 2
    smoothed: list[Vec3] = []
    for index in range(len(path)):
        lo = max(0, index - half)
        hi = min(len(path), index + half + 1)
        chunk = path[lo:hi]
        smoothed.append(
            Vec3(
                sum(p.x for p in chunk) / len(chunk),
                sum(p.y for p in chunk) / len(chunk),
                sum(p.z for p in chunk) / len(chunk),
            )
        )
    smoothed[0] = path[0]
    smoothed[-1] = path[-1]
    return smoothed


def _joint_axis_span_caps_m(joint_id: str | None) -> tuple[float, float, float]:
    """Maximum plausible root-relative span per axis (meters) for display."""
    jid = (joint_id or "").lower()
    if "hip" in jid or "pelvis" in jid:
        return (0.16, 0.16, 0.16)
    if any(token in jid for token in ("toe", "heel", "foot")):
        # Foot tips: keep the Overview cube tight so the trail fills the panel.
        return (0.28, 0.32, 0.36)
    if "ankle" in jid:
        return (0.28, 0.36, 0.36)
    if "knee" in jid:
        return (0.36, 0.42, 0.40)
    return (0.30, 0.36, 0.36)


def _robust_axis_span(
    values: list[float],
    *,
    low_pct: float = 0.05,
    high_pct: float = 0.95,
) -> tuple[float, float, float]:
    """Return (lo, hi, median) ignoring outer tails so spikes don't empty the cube."""
    clean = [
        float(v)
        for v in values
        if v is not None and math.isfinite(float(v))
    ]
    if not clean:
        return -0.05, 0.05, 0.0
    ordered = sorted(clean)
    n = len(ordered)
    med = statistics.median(clean)
    if n < 6:
        return ordered[0], ordered[-1], med
    lo = ordered[max(0, int(math.floor((n - 1) * low_pct)))]
    hi = ordered[min(n - 1, int(math.floor((n - 1) * high_pct)))]
    if hi < lo:
        lo, hi = ordered[0], ordered[-1]
    return float(lo), float(hi), float(med)


def _viewport_for_overview_dock(
    xs: list[float],
    ys: list[float],
    zs: list[float],
    *,
    floor_y: float | None = None,
    joint_id: str | None = None,
) -> _SingleTrajViewport:
    """
    Overview sidebar: centred equal-scale cube fitted to the robust path.

    Uses a single cube side from the largest robust axis span so the trail
    always fills the panel — never a tiny path stuck in the corner of a
    metre-wide empty box (common for foot tips with a long forward axis).
    """
    caps = _joint_axis_span_caps_m(joint_id)
    joint_cap = max(caps)
    cube_cap = min(joint_cap, _OVERVIEW_CUBE_SIDE_CAP_M)

    robust: list[tuple[float, float, float]] = []
    for vals in (xs, ys, zs):
        robust.append(_robust_axis_span(vals))

    robust_spans = [max(hi - lo, 1e-6) for lo, hi, _med in robust]
    centers = [med for _lo, _hi, med in robust]

    # Cube side so the largest motion fills ~TARGET_FILL of the view.
    raw_side = max(robust_spans) / max(_TRAJECTORY_TARGET_FILL, 0.35)
    cube_side = max(raw_side, _OVERVIEW_ABS_MIN_SPAN_M)
    cube_side = min(cube_side, cube_cap)

    # Grow the cube to fit robust envelopes AND path start/end (live tip).
    needed = _OVERVIEW_ABS_MIN_SPAN_M
    for i, ((lo, hi, med), span, vals) in enumerate(
        zip(robust, robust_spans, (xs, ys, zs), strict=True)
    ):
        pad = max(span * _OVERVIEW_VIEWPORT_EDGE_PAD, _OVERVIEW_MARKER_PAD_M)
        need = max(hi - med, med - lo) * 2.0 + 2.0 * pad
        # Always keep recording endpoints inside (they are the green/blue markers).
        ends = [
            float(v)
            for v in ((vals[0], vals[-1]) if vals else ())
            if v is not None and math.isfinite(float(v))
        ]
        for tip in ends:
            need = max(need, 2.0 * (abs(tip - med) + _OVERVIEW_MARKER_PAD_M))
        needed = max(needed, need)
    cube_side = max(cube_side, min(needed, cube_cap))

    limits: list[tuple[float, float]] = []
    for i, (_lo, _hi, med) in enumerate(robust):
        half = cube_side * 0.5
        lo, hi = med - half, med + half
        if i == 1 and floor_y is not None:
            lo = min(lo, float(floor_y) - _OVERVIEW_MARKER_PAD_M)
            # Keep cube height; shift up if floor pushed the bottom down.
            if hi - lo < cube_side:
                hi = lo + cube_side
        limits.append((lo, hi))

    # Re-include finite path extrema that survived the robust window (no spikes).
    for i, vals in enumerate((xs, ys, zs)):
        clean = [
            float(v)
            for v in vals
            if v is not None and math.isfinite(float(v))
        ]
        if not clean:
            continue
        lo_r, hi_r, med = robust[i]
        # Only extrema within a generous band of the robust range.
        band = max(hi_r - lo_r, _OVERVIEW_ABS_MIN_SPAN_M) * 0.75
        kept = [v for v in clean if abs(v - med) <= max(band, cube_side * 0.48)]
        if not kept:
            kept = [med]
        data_lo, data_hi = min(kept), max(kept)
        pad = _OVERVIEW_MARKER_PAD_M
        lo, hi = limits[i]
        lo = min(lo, data_lo - pad)
        hi = max(hi, data_hi + pad)
        # Re-centre on median with at least cube_side, capped.
        span = hi - lo
        if span < cube_side:
            half = cube_side * 0.5
            lo, hi = med - half, med + half
        else:
            mid = 0.5 * (lo + hi)
            half = min(span * 0.5, cube_cap * 0.5)
            lo, hi = mid - half, mid + half
            # Keep median inside.
            if med < lo + pad:
                lo, hi = med - pad, med - pad + (hi - lo)
            elif med > hi - pad:
                hi, lo = med + pad, med + pad - (hi - lo)
        limits[i] = (lo, hi)

    # Final equalise: one shared side so 1 cm is identical on X/Y/Z.
    side = max(hi - lo for lo, hi in limits)
    side = max(side, _OVERVIEW_ABS_MIN_SPAN_M)
    side = min(side, cube_cap)
    equalized: list[tuple[float, float]] = []
    for i, (lo, hi) in enumerate(limits):
        med = centers[i]
        half = side * 0.5
        equalized.append((med - half, med + half))
    limits = equalized

    cam_spans = (
        limits[0][1] - limits[0][0],
        limits[1][1] - limits[1][0],
        limits[2][1] - limits[2][0],
    )
    display_spans = _equal_box_aspect_from_limits(limits[0], limits[1], limits[2])
    elev, azim = _overview_camera_for_spans(
        (robust_spans[0], robust_spans[1], robust_spans[2])
    )
    if floor_y is not None:
        elev, azim = _FOOT_VIEW_ELEV, _FOOT_VIEW_AZIM

    return _SingleTrajViewport(
        xlim=limits[0],
        ylim=limits[1],
        zlim=limits[2],
        box_aspect=display_spans,
        elev=elev,
        azim=azim,
    )


def _clamp_limit_pair(lo: float, hi: float, *, max_span: float) -> tuple[float, float]:
    span = hi - lo
    if span <= max_span:
        return lo, hi
    mid = (lo + hi) * 0.5
    half = max_span * 0.5
    return mid - half, mid + half


def _filter_path_near_joint_median(
    path: list[Vec3],
    joint_id: str | None,
) -> list[Vec3]:
    """Reject pose spikes far from the per-axis median (root-relative ROM)."""
    if len(path) < 6:
        return path
    caps = _joint_axis_span_caps_m(joint_id)
    mx = statistics.median([p.x for p in path])
    my = statistics.median([p.y for p in path])
    mz = statistics.median([p.z for p in path])
    filtered = [
        p
        for p in path
        if abs(p.x - mx) <= caps[0] * 0.85
        and abs(p.y - my) <= caps[1] * 0.85
        and abs(p.z - mz) <= caps[2] * 0.85
    ]
    if len(filtered) < max(4, len(path) // 4):
        return path
    # Keep the playhead tip / start so markers never fall outside the cube.
    if path[0] not in filtered:
        filtered.insert(0, path[0])
    if path[-1] not in filtered:
        filtered.append(path[-1])
    return filtered


def _expand_viewport_to_include(
    viewport: _SingleTrajViewport,
    points: list[Vec3],
    *,
    joint_id: str | None = None,
    pad_frac: float = 0.12,
) -> _SingleTrajViewport:
    """Grow a centred equal-scale cube so path/markers stay inside (no speck view)."""
    finite = [
        p
        for p in points
        if p is not None
        and math.isfinite(float(p.x))
        and math.isfinite(float(p.y))
        and math.isfinite(float(p.z))
    ]
    if not finite:
        return viewport
    caps = _joint_axis_span_caps_m(joint_id)
    cube_cap = min(max(caps), _OVERVIEW_CUBE_SIDE_CAP_M)

    cx = 0.5 * (viewport.xlim[0] + viewport.xlim[1])
    cy = 0.5 * (viewport.ylim[0] + viewport.ylim[1])
    cz = 0.5 * (viewport.zlim[0] + viewport.zlim[1])
    # Prefer data median so a single tip cannot yank the cube off the trail.
    mx = statistics.median([p.x for p in finite])
    my = statistics.median([p.y for p in finite])
    mz = statistics.median([p.z for p in finite])
    # Blend: keep existing centre unless markers are clearly outside.
    centers = (
        0.65 * cx + 0.35 * mx,
        0.65 * cy + 0.35 * my,
        0.65 * cz + 0.35 * mz,
    )

    side = max(
        viewport.xlim[1] - viewport.xlim[0],
        viewport.ylim[1] - viewport.ylim[0],
        viewport.zlim[1] - viewport.zlim[0],
        _OVERVIEW_ABS_MIN_SPAN_M,
    )
    pad = max(side * pad_frac, _OVERVIEW_MARKER_PAD_M, 0.0025)
    for p in finite:
        need = 2.0 * max(
            abs(p.x - centers[0]) + pad,
            abs(p.y - centers[1]) + pad,
            abs(p.z - centers[2]) + pad,
        )
        side = max(side, need)
    side = min(side, cube_cap)
    half = side * 0.5
    xlim = (centers[0] - half, centers[0] + half)
    ylim = (centers[1] - half, centers[1] + half)
    zlim = (centers[2] - half, centers[2] + half)
    return _SingleTrajViewport(
        xlim=xlim,
        ylim=ylim,
        zlim=zlim,
        box_aspect=_equal_box_aspect_from_limits(xlim, ylim, zlim),
        elev=viewport.elev,
        azim=viewport.azim,
    )


def _catmull_rom_display_path(path: list[Vec3], *, samples_per_span: int = 4) -> list[Vec3]:
    """Display-only Catmull–Rom densification — endpoints stay exact."""
    n = len(path)
    if n < 3 or samples_per_span < 2:
        return path
    out: list[Vec3] = [path[0]]
    for i in range(n - 1):
        p0 = path[i - 1] if i > 0 else path[i]
        p1 = path[i]
        p2 = path[i + 1]
        p3 = path[i + 2] if i + 2 < n else path[i + 1]
        for s in range(1, samples_per_span + 1):
            t = s / float(samples_per_span)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1.x)
                + (-p0.x + p2.x) * t
                + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
                + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
            )
            y = 0.5 * (
                (2.0 * p1.y)
                + (-p0.y + p2.y) * t
                + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
                + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
            )
            z = 0.5 * (
                (2.0 * p1.z)
                + (-p0.z + p2.z) * t
                + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2
                + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3
            )
            out.append(Vec3(x, y, z))
    out[-1] = path[-1]
    return out


def _prepare_display_path(
    path: list[Vec3],
    *,
    overview: bool = False,
    motion_dock: bool = False,
    joint_id: str | None = None,
) -> list[Vec3]:
    """Display-only path cleanup: finite filter → spike reject → light smooth.

    Analysis / metric summaries keep using the raw ``_joint_path_with_times``
    samples. This only affects drawn trajectories.
    """
    finite = [
        point
        for point in path
        if math.isfinite(float(point.x))
        and math.isfinite(float(point.y))
        and math.isfinite(float(point.z))
    ]
    if len(finite) < 3:
        return finite
    cleaned = _remove_trajectory_outliers(finite)
    if overview or motion_dock:
        cleaned = _filter_path_near_joint_median(cleaned, joint_id)
        cleaned = _smooth_trajectory_path_light(cleaned, window=9)
        cleaned = _catmull_rom_display_path(cleaned, samples_per_span=5)
    else:
        cleaned = _smooth_trajectory_path_light(cleaned, window=5)
        cleaned = _catmull_rom_display_path(cleaned, samples_per_span=2)
    return cleaned


def _is_finite_point(point: Vec3 | None) -> bool:
    return bool(
        point is not None
        and math.isfinite(float(point.x))
        and math.isfinite(float(point.y))
        and math.isfinite(float(point.z))
    )


def _time_progression_points(path: list[Vec3]) -> list[tuple[int, str]]:
    """
    Choose Start and/or Middle labels based on path length and spacing.

    Current is always the red dot — never drawn as an in-graph text label.
    """
    path_len = len(path)
    if path_len <= 1:
        return []

    last = path_len - 1
    span = _path_max_span(path)
    min_sep = span * _PROGRESS_MIN_SEPARATION_RATIO
    current_pt = path[last]
    markers: list[tuple[int, str]] = []

    if path_len >= _PROGRESS_MIN_FRAMES_FOR_START:
        start_pt = path[0]
        if _point_distance(start_pt, current_pt) >= min_sep:
            markers.append((0, _PROGRESS_LABEL_START))

    if path_len >= _PROGRESS_MIN_FRAMES_FOR_MIDDLE:
        mid = path_len // 2
        if mid != 0 and mid != last:
            mid_pt = path[mid]
            if (
                _point_distance(path[0], mid_pt) >= min_sep
                and _point_distance(mid_pt, current_pt) >= min_sep
            ):
                markers.append((mid, _PROGRESS_LABEL_MIDDLE))

    return markers


def trajectory_progression_status(path: list[Vec3]) -> str:
    """Short status suffix describing which progress markers are active."""
    labels = {label for _index, label in _time_progression_points(path)}
    if _PROGRESS_LABEL_MIDDLE in labels:
        return "Start → Middle → current"
    if _PROGRESS_LABEL_START in labels:
        return "Start → current"
    return "current"


@dataclass(frozen=True)
class OverviewTrajSummary:
    """Compact metrics for the Overview 3D path sidebar."""

    path_length_cm: float
    span_x_cm: float
    span_y_cm: float
    span_z_cm: float
    max_span_cm: float
    dominant_axis: str
    motion_level: str
    samples: int
    position_cm: tuple[float, float, float] | None
    metrics_line: str
    detail_line: str
    motion_line: str
    video_line: str


def _joint_leg_side(joint_id: str | None) -> str | None:
    if not joint_id:
        return None
    if joint_id.startswith("right_"):
        return "right"
    if joint_id.startswith("left_"):
        return "left"
    return None


def _flexion_display(
    joint_label: str,
    current_deg: float,
    min_deg: float,
    max_deg: float,
) -> tuple[str, float, float, float]:
    """Show anatomical flexion (0=extension) for hinge joints when angles are obtuse."""
    joint_lower = joint_label.lower()
    if not any(token in joint_lower for token in ("knee", "hip", "elbow")):
        return "Angle", current_deg, min_deg, max_deg
    if max(current_deg, min_deg, max_deg) <= 90.0:
        return "Angle", current_deg, min_deg, max_deg
    cur = 180.0 - current_deg
    lo = 180.0 - max_deg
    hi = 180.0 - min_deg
    return "Flex", cur, lo, hi


def _is_front_facing_view(view_type: str | None) -> bool:
    vt = (view_type or "").upper()
    return vt in ("FRONTAL", "OBLIQUE") or (
        vt not in ("SAGITTAL_LEFT", "SAGITTAL_RIGHT", "UNKNOWN", "")
    )


def _is_sagittal_view(view_type: str | None) -> bool:
    vt = (view_type or "").upper()
    return vt.startswith("SAGITTAL")


def _trajectory_path_caption(
    joint_label: str,
    joint_id: str | None,
    *,
    view_type: str | None,
    dominant_axis: str,
    motion_level: str,
    span_x_cm: float,
    span_y_cm: float,
    span_z_cm: float,
) -> str:
    """One-line link between path shape in the 3D box and what the video shows."""
    joint_lower = joint_label.lower()
    side = _joint_leg_side(joint_id)
    leg = side.title() if side else "Joint"

    if "knee" in joint_lower:
        if (
            span_x_cm >= max(span_z_cm * 0.85, 4.0)
            and motion_level in ("Moderate", "Large")
            and _is_front_facing_view(view_type)
            and not _is_sagittal_view(view_type)
        ):
            return (
                f"{leg} knee zig-zags side-to-side — each swing left/right "
                f"matches a step in this front-view walk."
            )
        if dominant_axis == "Side (X)" and motion_level in ("Moderate", "Large"):
            if _is_front_facing_view(view_type) and not _is_sagittal_view(view_type):
                return (
                    f"{leg} knee zig-zags side-to-side — each swing left/right "
                    f"matches a step in this front-view walk."
                )
            if _is_sagittal_view(view_type):
                return (
                    f"{leg} knee path on side (X) — in a side-view clip, "
                    f"forward stepping often appears along X in this box."
                )
        if dominant_axis == "Forward (Z)" and motion_level in ("Moderate", "Large"):
            if _is_sagittal_view(view_type):
                return (
                    f"{leg} knee loops forward and up — matches flexion and "
                    f"extension during side-view steps."
                )
            return (
                f"{leg} knee path grows forward (Z) as the leg moves through "
                f"each step."
            )
        if dominant_axis == "Up (Y)" and span_y_cm >= 4.0:
            return (
                f"{leg} knee lifts vertically during swing — up/down motion "
                f"in the path matches the video."
            )

    if ("hip" in joint_lower or "pelvis" in joint_lower) and motion_level == "Small":
        return (
            "Compact hip path — small shift while stepping; typical with "
            "walker-assisted or slow gait."
        )

    if dominant_axis == "Side (X)" and _is_front_facing_view(view_type):
        return (
            f"Side-to-side path is expected in front-view walking — "
            f"the joint shifts left/right relative to the pelvis each step."
        )

    if dominant_axis == "Forward (Z)" and _is_sagittal_view(view_type):
        return (
            "Forward path in side-view video — stepping toward/away from "
            "the camera shows mainly along Z here."
        )

    return (
        "Blue path = movement so far vs pelvis; green = start, red = now "
        "in the current video frame."
    )


def _video_explanation(
    joint_label: str,
    joint_id: str | None,
    *,
    gait_phase: str | None,
    left_contact: str | None,
    right_contact: str | None,
    motion_level: str,
    dominant_axis: str,
    view_type: str | None = None,
    span_x_cm: float = 0.0,
    span_y_cm: float = 0.0,
    span_z_cm: float = 0.0,
) -> str:
    """Plain-language link between path metrics and what the video shows."""
    caption = _trajectory_path_caption(
        joint_label,
        joint_id,
        view_type=view_type,
        dominant_axis=dominant_axis,
        motion_level=motion_level,
        span_x_cm=span_x_cm,
        span_y_cm=span_y_cm,
        span_z_cm=span_z_cm,
    )
    if caption and "Blue path =" not in caption:
        return caption

    joint_lower = joint_label.lower()
    side = _joint_leg_side(joint_id)
    phase_upper = (gait_phase or "").upper()

    if "hip" in joint_lower or "pelvis" in joint_lower:
        if motion_level == "Small":
            return "Matches video: hip stays near pelvis (typical with walker/slow gait)."
        return "Matches video: hip shifting while stepping."

    if "knee" in joint_lower and side is not None:
        contact = right_contact if side == "right" else left_contact
        contact_upper = (contact or "").upper()
        leg = side.title()
        if "SWING" in contact_upper:
            if "Forward" in dominant_axis:
                return f"Matches video: {leg} knee swinging — path grows forward (Z)."
            return f"Matches video: {leg} leg swinging through step."
        if "CONTACT" in contact_upper:
            return f"Matches video: {leg} knee on stance leg — smaller path while foot supports."

    if "DOUBLE" in phase_upper:
        return "Matches video: both feet on ground — joint path stays compact."

    if "STANCE" in phase_upper:
        return "Matches video: weight on one leg — path reflects stance vs swing motion."

    return "Blue path = joint movement so far; red dot = position in current video frame."


def _motion_level_for_joint(joint_label: str, max_span_cm: float) -> str:
    """Joint-aware ROM size — hips move less than knees in pelvis-relative space."""
    joint_lower = joint_label.lower()
    if "hip" in joint_lower or "pelvis" in joint_lower:
        small, moderate = 3.0, 8.0
    elif any(
        token in joint_lower
        for token in ("knee", "ankle", "foot", "heel", "toe")
    ):
        small, moderate = 6.0, 18.0
    else:
        small, moderate = 3.0, 15.0
    if max_span_cm < small:
        return "Small"
    if max_span_cm < moderate:
        return "Moderate"
    return "Large"


def _joint_angle_window_stats(
    recording: GaitMotionRecording | None,
    joint_id: str,
    end_frame_float: float,
) -> tuple[float, float, float] | None:
    """Current, min, and max joint angle (deg) from clip start through playback."""
    if recording is None or recording.frame_count <= 0:
        return None
    ts = recording.build_time_series()
    series = ts.angles.get(joint_id, [])
    if not series:
        return None
    last_i = int(min(max(0, end_frame_float), len(series) - 1))
    cache_key = (id(recording), joint_id)
    cached = _ANGLE_ROM_CACHE.get(cache_key)
    if cached is not None:
        cached_last, amin, amax = cached
        if cached_last == last_i:
            cur = series[last_i]
            if cur is None:
                return None
            return (float(cur), amin, amax)
        if cached_last < last_i:
            # Extend running min/max (no full-window rescan).
            for index in range(cached_last + 1, last_i + 1):
                value = series[index]
                if value is None:
                    continue
                v = float(value)
                if v < amin:
                    amin = v
                if v > amax:
                    amax = v
            cur = series[last_i]
            if cur is None:
                return None
            _ANGLE_ROM_CACHE[cache_key] = (last_i, amin, amax)
            return (float(cur), amin, amax)
        # Rewound — fall through and rebuild.
    window = [float(a) for a in series[: last_i + 1] if a is not None]
    if not window:
        return None
    amin = min(window)
    amax = max(window)
    _ANGLE_ROM_CACHE[cache_key] = (last_i, amin, amax)
    if len(_ANGLE_ROM_CACHE) > 64:
        try:
            _ANGLE_ROM_CACHE.pop(next(iter(_ANGLE_ROM_CACHE)))
        except Exception:
            _ANGLE_ROM_CACHE.clear()
    return (window[-1], amin, amax)


def _format_delta_cm(delta: tuple[float, float, float]) -> str:
    dx, dy, dz = delta
    return (
        f"Move side {dx:+.1f} · up {dy:+.1f} · fwd {dz:+.1f} cm from start"
    )


def _path_speed_stats_cm_s(
    path_with_times: list[tuple[Vec3, float]],
) -> tuple[float | None, float | None]:
    """Average and peak segment speed along the path (cm/s)."""
    if len(path_with_times) < 2:
        return None, None
    speeds: list[float] = []
    for i in range(1, len(path_with_times)):
        p0, t0 = path_with_times[i - 1]
        p1, t1 = path_with_times[i]
        dt = t1 - t0
        if dt <= 1e-9:
            continue
        dist_cm = _point_distance(p0, p1) * 100.0
        speeds.append(dist_cm / dt)
    if not speeds:
        return None, None
    return sum(speeds) / len(speeds), max(speeds)


def summarize_overview_trajectory(
    path_with_times: list[tuple[Vec3, float]],
    *,
    joint_label: str = "Joint",
    recording: GaitMotionRecording | None = None,
    joint_id: str | None = None,
    end_frame_float: float = 0.0,
    gait_mode: str | None = None,
    gait_phase: str | None = None,
    left_contact: str | None = None,
    right_contact: str | None = None,
    progress_pct: float | None = None,
    elapsed_s: float | None = None,
    frame_index: int | None = None,
    frame_count: int | None = None,
    view_type: str | None = None,
) -> OverviewTrajSummary | None:
    """Build readable Overview metrics from a pelvis-relative joint path."""
    if not path_with_times:
        return None
    from stablewalk.ui.dashboard_interpretability import (
        evaluate_trajectory_readiness,
        format_trajectory_confidence,
    )

    positions = [p for p, _t in path_with_times]
    xs = [p.x for p in positions]
    ys = [p.y for p in positions]
    zs = [p.z for p in positions]
    span_x = (max(xs) - min(xs)) * 100.0
    span_y = (max(ys) - min(ys)) * 100.0
    span_z = (max(zs) - min(zs)) * 100.0
    spans = {"Side (X)": span_x, "Up (Y)": span_y, "Forward (Z)": span_z}
    dominant_axis = max(spans, key=spans.get)
    max_span = max(span_x, span_y, span_z)
    path_len = sum(
        _point_distance(positions[i - 1], positions[i])
        for i in range(1, len(positions))
    ) * 100.0
    avg_speed, max_speed = _path_speed_stats_cm_s(path_with_times)
    readiness = evaluate_trajectory_readiness(positions, min_samples=2)
    traj_metrics = readiness.metrics
    max_dev_cm = (
        traj_metrics.max_deviation_m * 100.0 if traj_metrics is not None else max_span
    )
    smooth_label = traj_metrics.smoothness if traj_metrics is not None else "—"
    conf_label = format_trajectory_confidence(readiness.confidence)
    motion_level = _motion_level_for_joint(joint_label, max_span)
    start = positions[0]
    current = positions[-1]
    pos_cm = (current.x * 100.0, current.y * 100.0, current.z * 100.0)
    delta_cm = (
        (current.x - start.x) * 100.0,
        (current.y - start.y) * 100.0,
        (current.z - start.z) * 100.0,
    )
    speed_bits: list[str] = []
    if avg_speed is not None:
        speed_bits.append(f"Avg {avg_speed:.0f} cm/s")
    if max_speed is not None:
        speed_bits.append(f"Max {max_speed:.0f} cm/s")
    if speed_bits:
        quality_line = (
            f"Smooth {smooth_label} · Conf {conf_label} · "
            f"{' · '.join(speed_bits)}"
        )
    else:
        quality_line = (
            f"Smooth {smooth_label} · Conf {conf_label} · {len(positions)} pts"
        )
    metrics_line = (
        f"Travel {path_len:.1f} cm  ·  ROM max {max_span:.1f} cm  ·  "
        f"side {span_x:.1f} · up {span_y:.1f} · fwd {span_z:.1f} cm"
    )
    sync_bits: list[str] = []
    if frame_index is not None and frame_count is not None and frame_count > 0:
        sync_bits.append(f"Frame {frame_index + 1}/{frame_count}")
    if elapsed_s is not None:
        sync_bits.append(f"{elapsed_s:.2f}s")
    if progress_pct is not None:
        sync_bits.append(f"{progress_pct:.0f}%")
    angle_stats = (
        _joint_angle_window_stats(recording, joint_id, end_frame_float)
        if joint_id
        else None
    )
    angle_part = ""
    if angle_stats is not None:
        current_deg, min_deg, max_deg = angle_stats
        label, cur, lo, hi = _flexion_display(
            joint_label, current_deg, min_deg, max_deg
        )
        angle_part = f" · {label} {cur:.0f}° ({lo:.0f}–{hi:.0f}°)"
    explanation = _video_explanation(
        joint_label,
        joint_id,
        gait_phase=gait_phase,
        left_contact=left_contact,
        right_contact=right_contact,
        motion_level=motion_level,
        dominant_axis=dominant_axis,
        view_type=view_type,
        span_x_cm=span_x,
        span_y_cm=span_y,
        span_z_cm=span_z,
    )
    context_bits: list[str] = []
    if gait_mode:
        context_bits.append(gait_mode)
    if gait_phase and gait_phase not in ("—", ""):
        context_bits.append(gait_phase)
    side = _joint_leg_side(joint_id)
    if side == "right" and right_contact:
        context_bits.append(f"R {right_contact}")
    elif side == "left" and left_contact:
        context_bits.append(f"L {left_contact}")
    motion_bits = [
        f"Now ({pos_cm[0]:.1f}, {pos_cm[1]:.1f}, {pos_cm[2]:.1f}) cm",
        motion_level,
        dominant_axis,
        _format_delta_cm(delta_cm),
    ]
    if angle_part:
        motion_bits.append(angle_part.strip(" ·"))
    detail_parts = sync_bits + context_bits + motion_bits + [quality_line]
    detail_line = " · ".join(detail_parts)
    video_line = explanation
    motion_line = (
        "● Start (green)  ·  Path (fade→bright)  ·  "
        "● Current (red)  ·  ● End (blue)"
    )
    return OverviewTrajSummary(
        path_length_cm=path_len,
        span_x_cm=span_x,
        span_y_cm=span_y,
        span_z_cm=span_z,
        max_span_cm=max_span,
        dominant_axis=dominant_axis,
        motion_level=motion_level,
        samples=len(positions),
        position_cm=pos_cm,
        metrics_line=metrics_line,
        detail_line=detail_line,
        motion_line=motion_line,
        video_line=video_line,
    )


def _tangent_at_index(path: list[Vec3], index: int) -> tuple[float, float, float]:
    """Unit tangent along the path at ``index`` (start / middle / end aware)."""
    if len(path) < 2:
        return (1.0, 0.0, 0.0)
    if index <= 0:
        a, b = path[0], path[1]
    elif index >= len(path) - 1:
        a, b = path[-2], path[-1]
    else:
        a, b = path[index - 1], path[index + 1]
    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        return (1.0, 0.0, 0.0)
    return (dx / length, dy / length, dz / length)


def _side_offset(
    tangent: tuple[float, float, float],
    bump: float,
    *,
    flip: bool = False,
) -> tuple[float, float, float]:
    """Perpendicular offset from the path tangent (reduces label overlap)."""
    tx, ty, tz = tangent
    bx = tz
    by = 0.0
    bz = -tx
    length = math.sqrt(bx * bx + by * by + bz * bz)
    if length < 1e-9:
        bx, by, bz = (0.0, bump, 0.0)
    else:
        sign = -1.0 if flip else 1.0
        scale = bump * sign / length
        bx, by, bz = bx * scale, by * scale + bump * 0.12, bz * scale
    return (bx, by, bz)


def _progression_label_offset(
    *,
    label: str,
    path: list[Vec3],
    index: int,
    bump: float,
) -> tuple[float, float, float]:
    """Place Start / Middle labels beside the path without overlapping."""
    tangent = _tangent_at_index(path, index)
    tx, ty, tz = tangent

    if label == _PROGRESS_LABEL_START:
        return (-tx * bump * 1.12, -ty * bump * 1.12 + bump * 0.1, -tz * bump * 1.12)
    return _side_offset(tangent, bump, flip=True)


def _draw_trajectory_time_markers(
    ax: Axes,
    path: list[Vec3],
) -> None:
    """Draw adaptive Start / Middle markers; current stays the red dot."""
    markers = _time_progression_points(path)
    if not markers:
        return

    bump = _path_max_span(path) * 0.085

    for index, label in markers:
        point = path[index]
        ox, oy, oz = _progression_label_offset(
            label=label,
            path=path,
            index=index,
            bump=bump,
        )
        ax.scatter(
            [point.x],
            [point.y],
            [point.z],
            color=_TIME_MARKER_COLOR,
            s=_TIME_MARKER_SIZE,
            alpha=0.82,
            edgecolors=TEXT,
            linewidths=0.3,
            marker="o",
            zorder=5,
            depthshade=False,
        )
        ax.text(
            point.x + ox,
            point.y + oy,
            point.z + oz,
            label,
            color=_TIME_LABEL_COLOR,
            fontsize=6.5,
            ha="center",
            va="center",
            zorder=6,
            clip_on=True,
        )


def _draw_spatial_cube_frame(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
) -> None:
    """Outline the axis limits as a wireframe cube (floor = min Y / ground)."""
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    # Floor / ceiling faces are XZ at y=y0 / y=y1 so the frame matches Y-up.
    floor = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y0, z1),
        (x0, y0, z1),
    )
    ceiling = (
        (x0, y1, z0),
        (x1, y1, z0),
        (x1, y1, z1),
        (x0, y1, z1),
    )

    def _edge(a: tuple[float, float, float], b: tuple[float, float, float]) -> None:
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            [a[2], b[2]],
            color=_CUBE_EDGE_COLOR,
            alpha=0.90,
            linewidth=1.22,
            solid_capstyle="round",
            zorder=1,
        )

    for index in range(4):
        _edge(floor[index], floor[(index + 1) % 4])
        _edge(ceiling[index], ceiling[(index + 1) % 4])
        _edge(floor[index], ceiling[index])  # vertical = +Y


def _draw_overview_trajectory_explainers(
    ax: Axes,
    *,
    path: list[Vec3],
    start: Vec3,
    current: Vec3,
    caption: str | None = None,
    metrics_line: str | None = None,
) -> list[object]:
    """
    Overview dock: wireframe cube, corner axis hints, and one caption line.

    Travel, ROM, and coordinates are shown in the text panel below the graph
    so values stay readable (especially on athletic / side-view clips).
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim

    line_before = len(ax.lines)
    text_before = len(ax.texts)

    _draw_spatial_cube_frame(ax, xlim, ylim, zlim)
    # Axis titles (X Lateral / Y Vertical / Z Forward) carry the convention —
    # skip in-cube corner tags so they cannot collide with tick labels.

    artists: list[object] = []
    if metrics_line:
        metrics_artist = ax.text2D(
            0.5,
            0.97,
            metrics_line,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=ACCENT,
            fontweight="bold",
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor=PANEL,
                edgecolor=BORDER,
                alpha=0.92,
                linewidth=0.5,
            ),
        )
        artists.append(metrics_artist)
    if caption:
        caption_artist = ax.text2D(
            0.5,
            0.03,
            caption,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=TEXT,
            linespacing=1.25,
            wrap=True,
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=PANEL,
                edgecolor=BORDER,
                alpha=0.92,
                linewidth=0.5,
            ),
        )
        artists.append(caption_artist)

    artists.extend(ax.lines[line_before:])
    artists.extend(ax.texts[text_before:])
    return artists


def _stable_viewport_for_joint(
    recording: GaitMotionRecording,
    joint_id: str,
    *,
    coord_mode: str = _COORD_ROOT_RELATIVE,
    motion_series: object | None = None,
    floor_y: float | None = None,
    position_scale: float = 1.0,
) -> _SingleTrajViewport | None:
    """Axis limits from the full valid trajectory (stable during playback)."""
    full_path = _joint_path_with_times(
        recording,
        joint_id,
        float(recording.frame_count - 1),
        coord_mode=coord_mode,
        motion_series=motion_series,
        position_scale=position_scale,
    )
    if len(full_path) < 2:
        return None
    xs = [p.x for p, _t in full_path]
    ys = [p.y for p, _t in full_path]
    zs = [p.z for p, _t in full_path]
    limit_ys = list(ys)
    if floor_y is not None:
        limit_ys.append(floor_y)
    return _viewport_for_single_dof_trajectory(xs, limit_ys, zs, floor_y=floor_y)


def _get_cached_stable_viewport(
    ax: Axes,
    recording: GaitMotionRecording,
    joint_id: str,
    *,
    coord_mode: str,
    motion_series: object | None,
    floor_y: float | None,
    position_scale: float = 1.0,
) -> _SingleTrajViewport | None:
    # Include stature scale so cm ticks match the drawn / labeled path.
    key = (joint_id, coord_mode, floor_y, round(float(position_scale), 6))
    cached = getattr(ax, "_stablewalk_stable_viewport", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    full_path = _joint_path_with_times(
        recording,
        joint_id,
        float(recording.frame_count - 1),
        coord_mode=coord_mode,
        motion_series=motion_series,
        position_scale=position_scale,
    )
    if len(full_path) < 2:
        return None
    overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
    motion_dock = bool(getattr(ax, "_stablewalk_motion_dock", False))
    raw_points = [p for p, _t in full_path]
    display_points = _prepare_display_path(
        raw_points,
        overview=overview_dock,
        motion_dock=motion_dock,
        joint_id=joint_id,
    )
    xs = [p.x for p in display_points]
    ys = [p.y for p in display_points]
    zs = [p.z for p in display_points]
    limit_ys = list(ys)
    if floor_y is not None:
        limit_ys.append(floor_y)
    if overview_dock or motion_dock:
        viewport = _viewport_for_overview_dock(
            xs, limit_ys, zs, floor_y=floor_y, joint_id=joint_id
        )
    else:
        viewport = _viewport_for_single_dof_trajectory(xs, limit_ys, zs, floor_y=floor_y)
    ax._stablewalk_stable_viewport = (key, viewport)
    return viewport


def _apply_single_dof_limits(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    *,
    floor_y: float | None = None,
    stable_viewport: _SingleTrajViewport | None = None,
    joint_id: str | None = None,
) -> None:
    """
    Set axis limits for the 3D trajectory panel.

    When ``stable_viewport`` is provided, limits stay fixed to the full
    recording trajectory so playback growth is visible instead of re-zooming
    every frame to the partial path.
    """
    if stable_viewport is not None:
        ax.set_autoscale_on(False)
        ax.set_xlim(*stable_viewport.xlim)
        ax.set_ylim(*stable_viewport.ylim)
        ax.set_zlim(*stable_viewport.zlim)
        w_in, h_in = ax.figure.get_size_inches()
        overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
        box_zoom = _single_traj_box_zoom(
            w_in,
            h_in,
            dpi=ax.figure.get_dpi(),
            overview_dock=overview_dock,
        )
        try:
            ax.set_box_aspect(stable_viewport.box_aspect, zoom=box_zoom)
        except TypeError:
            try:
                ax.set_box_aspect(stable_viewport.box_aspect)
            except (AttributeError, ValueError):
                pass
        except (AttributeError, ValueError):
            pass
        # Preserve user / preset orbit when present; else Perspective.
        user_cam = getattr(ax, "_stablewalk_user_camera", None)
        if isinstance(user_cam, (tuple, list)) and len(user_cam) == 2:
            elev, azim = float(user_cam[0]), float(user_cam[1])
        else:
            elev, azim = float(stable_viewport.elev), float(stable_viewport.azim)
        _view_init_y_up(ax, elev=elev, azim=azim)
        _apply_single_dof_camera(ax)
        try:
            ax.set_proj_type("persp")
        except (AttributeError, ValueError):
            pass
        if overview_dock:
            _style_overview_trajectory_cube(ax)
        else:
            _style_single_dof_cube(ax)
        return

    if not xs:
        xlim = (-0.1, 0.1)
        ylim = (0.0, 0.2)
        zlim = (-0.1, 0.1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        _view_init_y_up(ax, elev=_SINGLE_TRAJ_ELEV, azim=_SINGLE_TRAJ_AZIM)
        _apply_single_dof_camera(ax)
        _style_single_dof_cube(ax)
        return

    overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
    if overview_dock:
        viewport = _viewport_for_overview_dock(xs, ys, zs, floor_y=floor_y, joint_id=joint_id)
    else:
        viewport = _viewport_for_single_dof_trajectory(xs, ys, zs, floor_y=floor_y)
    ax.set_xlim(*viewport.xlim)
    ax.set_ylim(*viewport.ylim)
    ax.set_zlim(*viewport.zlim)

    w_in, h_in = ax.figure.get_size_inches()
    box_zoom = _single_traj_box_zoom(
        w_in,
        h_in,
        dpi=ax.figure.get_dpi(),
        overview_dock=overview_dock,
    )
    try:
        ax.set_box_aspect(viewport.box_aspect, zoom=box_zoom)
    except TypeError:
        try:
            ax.set_box_aspect(viewport.box_aspect)
        except (AttributeError, ValueError):
            pass
    except (AttributeError, ValueError):
        pass

    _view_init_y_up(ax, elev=viewport.elev, azim=viewport.azim)
    _apply_single_dof_camera(ax)
    try:
        ax.set_proj_type("persp")
    except (AttributeError, ValueError):
        pass

    _style_single_dof_cube(ax)


def _transform_position_for_coord_mode(
    position: Vec3,
    frame_index: int,
    *,
    coord_mode: str,
    motion_series: object | None,
) -> Vec3:
    """Pelvis-relative positions stay as-is; Global adds per-frame pelvis offset."""
    if normalize_coord_mode(coord_mode) != _COORD_GLOBAL or motion_series is None:
        return position
    frame_indices = getattr(motion_series, "frame_indices", None)
    global_pelvis = getattr(motion_series, "global_pelvis", None)
    if not frame_indices or not global_pelvis:
        return position
    try:
        idx = frame_indices.index(frame_index)
    except ValueError:
        return position
    if idx < 0 or idx >= len(global_pelvis):
        return position
    pelvis = global_pelvis[idx]
    if pelvis is None:
        return position
    return Vec3(
        position.x + float(pelvis[0]),
        position.y + float(pelvis[1]),
        position.z + float(pelvis[2]),
    )


def normalize_coord_mode(coord_mode: str | None) -> str:
    """Map UI / session strings to canonical coord labels (legacy ALL-CAPS OK)."""
    raw = (coord_mode or "").strip()
    key = raw.upper().replace(" ", "-").replace("_", "-")
    if key == "GLOBAL":
        return _COORD_GLOBAL
    if key in ("ROOT-RELATIVE", "ROOTRELATIVE"):
        return _COORD_ROOT_RELATIVE
    if raw in (_COORD_GLOBAL, _COORD_ROOT_RELATIVE):
        return raw
    return _COORD_ROOT_RELATIVE


def normalize_display_mode(display_mode: str | None) -> str:
    """Map UI / session strings to canonical trajectory display labels."""
    raw = (display_mode or "").strip()
    key = " ".join(raw.upper().replace("-", " ").split())
    if key in ("FULL PATH", "FULL TRAJECTORY", "FULLPATH", "FULLTRAJECTORY"):
        return _DISPLAY_FULL_TRAJECTORY
    if key in ("CURRENT PROGRESS", "CURRENTPROGRESS"):
        return _DISPLAY_CURRENT_PROGRESS
    if raw in (
        _DISPLAY_CURRENT_PROGRESS,
        _DISPLAY_FULL_PATH,
        _DISPLAY_FULL_TRAJECTORY,
    ):
        return raw if raw != _DISPLAY_FULL_PATH else _DISPLAY_FULL_TRAJECTORY
    return _DISPLAY_CURRENT_PROGRESS


def _display_end_frame(
    display_mode: str,
    playback_frame_float: float,
    recording: GaitMotionRecording | None,
) -> float:
    """Frame index used for the drawn path extent."""
    mode = normalize_display_mode(display_mode)
    full_modes = (_DISPLAY_FULL_PATH, _DISPLAY_FULL_TRAJECTORY)
    if mode in full_modes and recording and recording.frame_count > 0:
        return float(recording.frame_count - 1)
    return playback_frame_float


def _view_angles_for_projection(
    projection_mode: str,
    spans: tuple[float, float, float] | None = None,
) -> tuple[float, float]:
    """Camera angles for 3D axis — frontal/sagittal are fixed viewpoints, not 2D axes."""
    if projection_mode == _PLANE_PROJECTION_FRONTAL:
        return 0.0, -90.0
    if projection_mode == _PLANE_PROJECTION_SAGITTAL:
        return 0.0, 0.0
    if spans is not None:
        return _camera_for_single_dof_trajectory(spans)
    return _SINGLE_TRAJ_ELEV, _SINGLE_TRAJ_AZIM


# Progressive path cache: append samples as the playhead advances instead of
# rebuilding 0..N every draw. Keyed by recording identity + joint + modes.
_PATH_CACHE: dict[tuple, tuple[int, list[tuple[Vec3, float]]]] = {}
_PATH_CACHE_MAX_KEYS = 8
# Progressive angle ROM: (last_index, amin, amax) per recording+joint.
_ANGLE_ROM_CACHE: dict[tuple, tuple[int, float, float]] = {}


def clear_trajectory_path_cache() -> None:
    """Drop cached joint paths (call when a new recording is loaded)."""
    _PATH_CACHE.clear()
    _ANGLE_ROM_CACHE.clear()


def _joint_path_with_times(
    recording: GaitMotionRecording,
    joint_id: str,
    end_frame_float: float,
    *,
    coord_mode: str = _COORD_ROOT_RELATIVE,
    motion_series: object | None = None,
    position_scale: float = 1.0,
) -> list[tuple[Vec3, float]]:
    """Joint positions from frame 0 through ``end_frame_float`` with timestamps."""
    if recording.frame_count <= 0:
        return []
    # Truncate to discrete frame (matches HUD / status playhead).
    last_index = int(float(end_frame_float))
    last_index = int(max(0, min(last_index, recording.frame_count - 1)))
    cache_key = (
        id(recording),
        joint_id,
        coord_mode,
        round(float(position_scale), 6),
        id(motion_series) if motion_series is not None else 0,
    )
    cached_entry = _PATH_CACHE.get(cache_key)
    if cached_entry is not None:
        cached_last, cached_path = cached_entry
        if cached_last == last_index:
            return list(cached_path)
        if cached_last > last_index:
            # Playhead rewound — rebuild from scratch for correctness.
            cached_entry = None
        else:
            start_index = cached_last + 1
            out = list(cached_path)
    if cached_entry is None:
        start_index = 0
        out = []
    for index in range(start_index, last_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        sample = snap.joints.get(joint_id)
        if sample is None:
            continue
        position = _transform_position_for_coord_mode(
            sample.position,
            index,
            coord_mode=coord_mode,
            motion_series=motion_series,
        )
        out.append((_scale_vec(position, position_scale), float(snap.time_s)))
    if len(_PATH_CACHE) >= _PATH_CACHE_MAX_KEYS and cache_key not in _PATH_CACHE:
        try:
            _PATH_CACHE.pop(next(iter(_PATH_CACHE)))
        except Exception:
            _PATH_CACHE.clear()
    _PATH_CACHE[cache_key] = (last_index, out)
    return list(out)


def _resolve_trajectory_points(
    recording: GaitMotionRecording,
    joint_id: str,
    *,
    playback_frame_float: float,
    tip_snapshot: SkeletonSnapshot | None,
    display_mode: str,
    coord_mode: str,
    motion_series: object | None,
    position_scale: float = 1.0,
) -> tuple[list[tuple[Vec3, float]], Vec3 | None, Vec3 | None]:
    """
    Build the displayed path plus current and full-recording end markers.

    Returns (path_with_times, current_position, end_position).
    """
    path_end = _display_end_frame(display_mode, playback_frame_float, recording)
    path_with_times = _joint_path_with_times(
        recording,
        joint_id,
        path_end,
        coord_mode=coord_mode,
        motion_series=motion_series,
        position_scale=position_scale,
    )

    current: Vec3 | None = None
    if tip_snapshot and joint_id in tip_snapshot.joints:
        current = _transform_position_for_coord_mode(
            tip_snapshot.joints[joint_id].position,
            int(round(float(getattr(tip_snapshot, "frame_index", path_end)))),
            coord_mode=coord_mode,
            motion_series=motion_series,
        )
        current = _scale_vec(current, position_scale)
        if current is not None and (
            not path_with_times
            or abs(path_with_times[-1][0].x - current.x) > 1e-5
            or abs(path_with_times[-1][0].y - current.y) > 1e-5
            or abs(path_with_times[-1][0].z - current.z) > 1e-5
        ):
            path_with_times = list(path_with_times) + [
                (current, float(tip_snapshot.time_s))
            ]
    elif path_with_times:
        current = path_with_times[-1][0]

    full_end_path = _joint_path_with_times(
        recording,
        joint_id,
        float(recording.frame_count - 1),
        coord_mode=coord_mode,
        motion_series=motion_series,
        position_scale=position_scale,
    )
    end_point = full_end_path[-1][0] if full_end_path else None
    return path_with_times, current, end_point


def _foot_bones(snapshot: SkeletonSnapshot, side: str) -> list[tuple[str, str]]:
    ankle, heel, toe, foot = f"{side}_ankle", f"{side}_heel", f"{side}_toe", f"{side}_foot"
    if snapshot.joints.get(heel):
        out = [(ankle, heel)]
        if snapshot.joints.get(toe):
            out.append((heel, toe))
        elif snapshot.joints.get(foot):
            out.append((heel, foot))
        return out
    if snapshot.joints.get(toe):
        return [(ankle, toe)]
    if snapshot.joints.get(foot):
        return [(ankle, foot)]
    return []


def _com_position(snapshot: SkeletonSnapshot) -> Vec3 | None:
    """Body-centre proxy: average of hips + shoulders (transparent CoM approximation)."""
    pts: list[Vec3] = []
    for jid in ("left_hip", "right_hip", "left_shoulder", "right_shoulder"):
        sample = snapshot.joints.get(jid)
        if sample:
            pts.append(sample.position)
    if not pts:
        pelvis = snapshot.joints.get(ROOT_JOINT_ID)
        return pelvis.position if pelvis else None
    n = len(pts)
    return Vec3(
        x=sum(p.x for p in pts) / n,
        y=sum(p.y for p in pts) / n,
        z=sum(p.z for p in pts) / n,
    )


def _joint_path(
    recording: GaitMotionRecording,
    joint_id: str,
    end_frame_float: float,
) -> list[Vec3]:
    series = recording.build_time_series()
    positions = series.positions.get(joint_id, [])
    if not positions:
        return []
    last_index = int(max(0, min(end_frame_float, len(positions) - 1)))
    return list(positions[: last_index + 1])


def _com_path(
    recording: GaitMotionRecording,
    end_frame_float: float,
) -> list[Vec3]:
    if not recording.snapshots:
        return []
    last_index = int(max(0, min(end_frame_float, len(recording.snapshots) - 1)))
    path: list[Vec3] = []
    for snap in recording.snapshots[: last_index + 1]:
        com = _com_position(snap)
        if com:
            path.append(com)
    return path


def _pos(snapshot: SkeletonSnapshot, joint_id: str) -> Vec3 | None:
    sample = snapshot.joints.get(joint_id)
    return sample.position if sample else None


def _draw_stick_skeleton(
    ax: Axes,
    snapshot: SkeletonSnapshot,
    *,
    color: str = MUTED,
    alpha: float = 0.65,
    linewidth: float = 1.4,
) -> None:
    """Current full-body pose as a muted 3D stick figure."""
    bones = list(_STICK_BONES)
    for side in ("left", "right"):
        bones.extend(_foot_bones(snapshot, side))

    for parent, child in bones:
        p0, p1 = _pos(snapshot, parent), _pos(snapshot, child)
        if not p0 or not p1:
            continue
        ax.plot(
            [p0.x, p1.x],
            [p0.y, p1.y],
            [p0.z, p1.z],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=2,
        )

    # Small joint dots on key landmarks
    for jid in (
        "head",
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
        ROOT_JOINT_ID,
    ):
        pt = _pos(snapshot, jid)
        if pt:
            ax.scatter(
                [pt.x],
                [pt.y],
                [pt.z],
                color=color,
                s=14,
                alpha=alpha * 0.9,
                edgecolors="none",
                zorder=3,
            )


def _legend_label(item_id: str, joint_id: str) -> str:
    dof_label = label_for_item(item_id)
    joint_name = JOINT_DISPLAY_NAMES.get(joint_id, joint_id.replace("_", " ").title())
    return f"{dof_label} ({joint_name})"


def _apply_limits(ax: Axes, xs: list[float], ys: list[float], zs: list[float]) -> None:
    if not xs:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(-0.5, 0.5)
        return

    pad = 0.14
    for vals, setter in (
        (xs, ax.set_xlim),
        (ys, ax.set_ylim),
        (zs, ax.set_zlim),
    ):
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 0.08)
        margin = span * pad
        setter(lo - margin, hi + margin)

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _layout_figure(ax: Axes, *, legend_rows: int = 0) -> None:
    fig = ax.figure
    if legend_rows:
        import math

        ncol = 2 if legend_rows > 2 else legend_rows
        rows = math.ceil(legend_rows / ncol)
        top = max(0.58, 0.985 - (0.05 * rows))
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=top)
    else:
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.06, top=0.96)


def _draw_hint(ax: Axes, text: str) -> None:
    ax.text2D(
        0.5,
        0.03,
        text,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=MUTED,
        fontsize=8.5,
        style="italic",
        zorder=10,
    )


def _draw_default_view(
    ax: Axes,
    recording: GaitMotionRecording,
    *,
    end_frame_float: float,
    tip_snapshot: SkeletonSnapshot | None,
) -> bool:
    """
    CoM trajectory + current stick skeleton when no joints are selected.

    Returns True if anything was drawn.
    """
    com_trail = _com_path(recording, end_frame_float)
    snap = tip_snapshot or recording.snapshot_at(int(end_frame_float))
    if not snap and not com_trail:
        return False

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []

    if len(com_trail) >= 2:
        xs = [p.x for p in com_trail]
        ys = [p.y for p in com_trail]
        zs = [p.z for p in com_trail]
        ax.plot(
            xs,
            ys,
            zs,
            color=COM,
            linewidth=1.75,
            alpha=0.88,
            label="Center of mass",
            zorder=4,
        )
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            [zs[-1]],
            color=COM,
            s=36,
            edgecolors=TEXT,
            linewidths=0.6,
            zorder=6,
        )
        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)
    elif len(com_trail) == 1:
        pt = com_trail[0]
        ax.scatter(
            [pt.x],
            [pt.y],
            [pt.z],
            color=COM,
            s=36,
            edgecolors=TEXT,
            linewidths=0.6,
            label="Center of mass",
            zorder=6,
        )
        all_x.extend([pt.x])
        all_y.extend([pt.y])
        all_z.extend([pt.z])

    if snap:
        _draw_stick_skeleton(ax, snap, color=MUTED, alpha=0.55, linewidth=1.3)
        for jid in snap.joints:
            p = _pos(snap, jid)
            if p:
                all_x.append(p.x)
                all_y.append(p.y)
                all_z.append(p.z)

    _apply_limits(ax, all_x, all_y, all_z)
    _layout_figure(ax, legend_rows=1 if com_trail else 0)
    if com_trail:
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            bbox_transform=ax.figure.transFigure,
            ncol=1,
            fontsize=8,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            framealpha=0.95,
        )
    _draw_hint(ax, _HINT_NO_SELECTION)
    return bool(com_trail or snap)


def _annotate_traj_marker(
    ax: Axes,
    point: Vec3,
    label: str,
    *,
    color: str,
    path: list[Vec3],
    index: int,
    span: float,
) -> None:
    """Short in-graph label beside a start/current marker."""
    bump = max(span * 0.09, 0.006)
    tangent = _tangent_at_index(path, index)
    ox, oy, oz = _side_offset(tangent, bump, flip=(label != _SINGLE_TRAJ_MARKER_LABEL_START))
    oy += bump * 0.24
    ax.text(
        point.x + ox,
        point.y + oy,
        point.z + oz,
        label,
        color=color,
        fontsize=8.5,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=9,
        clip_on=True,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=PANEL,
            edgecolor=color,
            linewidth=0.75,
            alpha=0.94,
        ),
    )


def _tail_segment_slice(seg_count: int, *, frac: float = _PATH_FADE_TAIL_FRAC) -> int:
    """Index where the bright recent-tail overlay begins."""
    if seg_count < 6:
        return 0
    return max(0, int(seg_count * (1.0 - frac)))


def _path_segment_styles(
    seg_count: int,
    color: str,
    *,
    alpha_min: float | None = None,
    line_width: float | None = None,
    confidence: float | None = None,
) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Per-segment RGBA + widths that fade from earlier samples toward current."""
    import matplotlib.colors as mcolors

    if seg_count <= 0:
        return [], []
    old = mcolors.to_rgb(_PATH_OLD_COLOR)
    current = mcolors.to_rgb(color)
    # Optional confidence tint: high conf → path color; low → muted amber.
    if confidence is not None:
        conf = max(0.0, min(1.0, float(confidence)))
        warn = mcolors.to_rgb("#c9a227")
        current = tuple(
            current[i] * conf + warn[i] * (1.0 - conf) for i in range(3)
        )
        old = tuple(old[i] * (0.55 + 0.45 * conf) for i in range(3))
    a0 = _PATH_FADE_ALPHA_MIN if alpha_min is None else float(alpha_min)
    a1 = _PATH_FADE_ALPHA_MAX
    lw = _PATH_LINE_WIDTH if line_width is None else float(line_width)
    colors: list[tuple[float, float, float, float]] = []
    widths: list[float] = []
    for i in range(seg_count):
        t = i / max(seg_count - 1, 1)
        smooth = t * t * (3.0 - 2.0 * t)
        rgb = tuple(old[j] + (current[j] - old[j]) * smooth for j in range(3))
        colors.append((*rgb, a0 + (a1 - a0) * smooth))
        # Slightly thicker near "now"; keep the trail slim so curvature reads.
        widths.append(lw * (0.82 + 0.28 * smooth))
    return colors, widths


def _draw_full_trajectory_base(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> None:
    """Draw the complete joint track (start→end) as a readable base line.

    Early playback only has a short progress trail; without this base the cube
    looks empty aside from the current dot. The base is the video's full joint
    path; progress overlays brighten up to the playhead.
    """
    if len(xs) < 2:
        return
    try:
        ax.plot(
            xs,
            ys,
            zs,
            color=_FULL_PATH_COLOR,
            linewidth=_FULL_PATH_LINE_WIDTH,
            alpha=_FULL_PATH_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3.2,
        )
    except Exception:
        pass


def _draw_single_dof_trajectory_path(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    *,
    future_xs: list[float] | None = None,
    future_ys: list[float] | None = None,
    future_zs: list[float] | None = None,
    confidence: float | None = None,
) -> None:
    """Draw the start-to-current path as a time-graded trail.

    Earlier samples fade; the trail brightens toward the current frame so
    direction and recency read at a glance. Optional future samples draw faded.
    """
    n = len(xs)
    if n < 2 and not (future_xs and len(future_xs) >= 2):
        return

    try:
        import matplotlib.colors as mcolors
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        overview = bool(getattr(ax, "_stablewalk_overview_dock", False))
        motion = bool(getattr(ax, "_stablewalk_motion_dock", False))
        # Overview dock: slightly heavier so the trail tracks the video clearly.
        line_w = (_PATH_LINE_WIDTH * 1.35) if (overview or motion) else _PATH_LINE_WIDTH
        clip = not overview

        if n >= 2:
            pts = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
            segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
            seg_count = len(segments)
            # Soft underlay helps the trail read against the dark cube faces.
            under_rgb = mcolors.to_rgb("#0d1218")
            under_extra = 1.15 if overview else 2.0
            under = Line3DCollection(
                segments,
                colors=[(*under_rgb, 0.55 if overview else 0.65)] * seg_count,
                linewidths=[line_w + under_extra] * seg_count,
                capstyle="round",
                antialiaseds=True,
                zorder=3.6,
            )
            under.set_clip_on(clip)
            ax.add_collection3d(under)
            colors, widths = _path_segment_styles(
                seg_count,
                _PATH_LINE_COLOR,
                line_width=line_w,
                confidence=confidence,
            )
            collection = Line3DCollection(
                segments,
                colors=colors,
                linewidths=widths,
                capstyle="round",
                antialiaseds=True,
                zorder=4,
            )
            collection.set_clip_on(clip)
            ax.add_collection3d(collection)
            # Fallback solid line — Line3DCollection can fail to paint on some
            # TkAgg/matplotlib builds; a plot() stroke always shows the path.
            ax.plot(
                xs,
                ys,
                zs,
                color=_PATH_LINE_COLOR,
                linewidth=max(line_w * 0.85, 1.8),
                alpha=0.88,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3.9,
                clip_on=clip,
            )

        if (
            future_xs is not None
            and future_ys is not None
            and future_zs is not None
            and len(future_xs) >= 2
        ):
            # Solid (not dashed) future — dashed 3D lines often vanish on TkAgg.
            ax.plot(
                future_xs,
                future_ys,
                future_zs,
                color=_FULL_PATH_COLOR,
                linewidth=max(_FULL_PATH_LINE_WIDTH, 1.6),
                alpha=max(_FULL_PATH_ALPHA, 0.38),
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3.1,
            )
    except Exception:
        if n >= 2:
            ax.plot(
                xs,
                ys,
                zs,
                color=_PATH_LINE_COLOR,
                linewidth=_PATH_LINE_WIDTH,
                alpha=_PATH_ALPHA,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4,
            )
        if future_xs is not None and len(future_xs) >= 2:
            ax.plot(
                future_xs,
                future_ys,
                future_zs,
                color=_PATH_LINE_COLOR,
                linewidth=max(_PATH_LINE_WIDTH * 0.7, 1.1),
                alpha=0.28,
                solid_capstyle="round",
                zorder=3,
            )


def _path_progress_sample_indices(path_len: int) -> list[int]:
    """Interior indices for subtle time-progress dots along the path."""
    if path_len < 5:
        return []
    if path_len >= 24:
        return [path_len // 4, path_len // 2, 3 * path_len // 4]
    if path_len >= 10:
        return [path_len // 3, 2 * path_len // 3]
    return [path_len // 2]


def _draw_path_progress_dots(
    ax: Axes,
    path: list[Vec3],
    *,
    marker_scale: float,
) -> None:
    """One batched scatter with smoothly fading time-ordered path samples."""
    if len(path) < 3:
        return

    import matplotlib.colors as mcolors

    # Cap the number of dots for long recordings while retaining smooth time
    # progression and a single inexpensive Path3DCollection.
    step = max(1, math.ceil((len(path) - 2) / 72))
    indices = list(range(1, len(path) - 1, step))
    if not indices:
        return
    old = mcolors.to_rgb(_PATH_OLD_COLOR)
    current = mcolors.to_rgb(_PATH_LINE_COLOR)
    last = len(path) - 1
    colors: list[tuple[float, float, float, float]] = []
    sizes: list[float] = []
    points = [path[index] for index in indices]
    for index in indices:
        t = index / last
        smooth = t * t * (3.0 - 2.0 * t)
        rgb = tuple(old[j] + (current[j] - old[j]) * smooth for j in range(3))
        colors.append((*rgb, 0.16 + 0.66 * smooth))
        sizes.append(
            (_PATH_DOT_SIZE_MIN + (_PATH_DOT_SIZE_MAX - _PATH_DOT_SIZE_MIN) * smooth)
            * max(0.85, marker_scale)
        )
    ax.scatter(
        [point.x for point in points],
        [point.y for point in points],
        [point.z for point in points],
        c=colors,
        s=sizes,
        edgecolors="none",
        zorder=4.5,
        depthshade=False,
    )


def _draw_single_dof_direction_arrow(
    ax: Axes,
    path: list[Vec3],
    *,
    span: float,
) -> None:
    """Small arrow on the final path segment pointing toward the current position."""
    if len(path) < 2:
        return

    start_pt = path[0]
    end_pt = path[-1]
    if _point_distance(start_pt, end_pt) < span * 0.02:
        return

    seg_start = max(0, int(len(path) * 0.62))
    seg_end = len(path) - 1
    if seg_start >= seg_end:
        seg_start = max(0, seg_end - 1)

    a = path[seg_start]
    b = path[seg_end]
    dx = b.x - a.x
    dy = b.y - a.y
    dz = b.z - a.z
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-7:
        return

    ux, uy, uz = dx / length, dy / length, dz / length
    arrow_len = min(length * 0.55, span * 0.12)
    tail_t = 0.35
    ox = a.x + dx * tail_t
    oy = a.y + dy * tail_t
    oz = a.z + dz * tail_t

    ax.quiver(
        ox,
        oy,
        oz,
        ux * arrow_len,
        uy * arrow_len,
        uz * arrow_len,
        color=_PATH_LINE_COLOR,
        alpha=0.95,
        linewidth=1.5,
        arrow_length_ratio=0.32,
        normalize=False,
        zorder=5,
    )


def _draw_path_floor_shadow(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> None:
    """Project the path onto the cube floor (min-Y plane) as a faint shadow.

    A 3D curve floating in a box is hard to read; a shadow on the floor gives the
    eye a depth reference, so the horizontal shape and direction of the motion
    become legible. Must be called after the axis limits are fixed.
    """
    if len(xs) < 2:
        return
    floor_level = ax.get_ylim()[0]
    floor_ys = [floor_level] * len(xs)
    ax.plot(
        xs,
        floor_ys,
        zs,
        color=_PATH_SHADOW_COLOR,
        linewidth=_PATH_SHADOW_WIDTH,
        alpha=_PATH_SHADOW_ALPHA,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=1.5,
    )
    ax.scatter(
        [xs[-1]],
        [floor_level],
        [zs[-1]],
        color=_PATH_SHADOW_COLOR,
        s=12.0,
        alpha=_PATH_SHADOW_ALPHA,
        edgecolors="none",
        zorder=1.6,
        depthshade=False,
    )


def _positions_match(a: Vec3, b: Vec3, *, tol: float = 1e-4) -> bool:
    return (
        abs(a.x - b.x) <= tol
        and abs(a.y - b.y) <= tol
        and abs(a.z - b.z) <= tol
    )


def _draw_single_dof_start_marker(
    ax: Axes,
    start: Vec3,
    *,
    marker_size: float,
    ring_only: bool = False,
) -> None:
    """Small green marker at the first recorded position."""
    # Overview: allow sphere to paint past the cube edge (limits already padded).
    clip = not bool(getattr(ax, "_stablewalk_overview_dock", False))
    if ring_only:
        ax.scatter(
            [start.x],
            [start.y],
            [start.z],
            facecolors="none",
            edgecolors=_START_DOT_COLOR,
            s=marker_size * 1.18,
            linewidths=1.0,
            zorder=6,
            depthshade=False,
            clip_on=clip,
        )
        return

    ax.scatter(
        [start.x],
        [start.y],
        [start.z],
        facecolors="none",
        edgecolors=_START_DOT_COLOR,
        s=marker_size * 1.55,
        linewidths=1.6,
        zorder=6,
        depthshade=False,
        clip_on=clip,
    )
    ax.scatter(
        [start.x],
        [start.y],
        [start.z],
        color=_START_DOT_COLOR,
        s=marker_size * 0.95,
        edgecolors=_START_DOT_EDGE_COLOR,
        linewidths=1.15,
        zorder=7,
        depthshade=False,
        clip_on=clip,
    )


def _draw_single_dof_end_marker(
    ax: Axes,
    end: Vec3,
    *,
    marker_size: float,
) -> None:
    """Blue marker at the latest analyzed position in the full recording."""
    clip = not bool(getattr(ax, "_stablewalk_overview_dock", False))
    ax.scatter(
        [end.x],
        [end.y],
        [end.z],
        facecolors="none",
        edgecolors=_END_DOT_COLOR,
        s=marker_size * 1.48,
        linewidths=1.5,
        zorder=6,
        depthshade=False,
        clip_on=clip,
    )
    ax.scatter(
        [end.x],
        [end.y],
        [end.z],
        color=_END_DOT_COLOR,
        s=marker_size * 0.90,
        edgecolors=_END_DOT_EDGE_COLOR,
        linewidths=1.05,
        zorder=7,
        depthshade=False,
        clip_on=clip,
    )


def _format_graph_clearance_label(clearance_m: float) -> str:
    """Short in-graph clearance label (cm, one decimal)."""
    return f"{clearance_m * 100.0:.1f} cm"


def _ground_label_corner(
    xlim: tuple[float, float],
    zlim: tuple[float, float],
    foot: Vec3 | None,
) -> tuple[float, float]:
    """Pick a cube corner for the Ground label away from the foot marker."""
    x0, x1 = xlim
    z0, z1 = zlim
    corners = ((x0, z0), (x1, z0), (x1, z1), (x0, z1))
    if foot is None:
        return corners[0]
    return max(
        corners,
        key=lambda xz: (xz[0] - foot.x) ** 2 + (xz[1] - foot.z) ** 2,
    )


def _draw_ground_plane_reference(
    ax: Axes,
    floor_y: float,
    *,
    xlim: tuple[float, float] | None = None,
    zlim: tuple[float, float] | None = None,
    pad_frac: float = 0.06,
    foot: Vec3 | None = None,
) -> None:
    """
    Draw the horizontal ground reference plane at Y = floor_y (+Y vertical).

    Matches ``ground_reference.GroundReferencePlane`` used for foot ground-distance
    metrics in the analysis panel and position table.
    """
    import numpy as np

    if xlim is None:
        xlim = ax.get_xlim()
    if zlim is None:
        zlim = ax.get_zlim()

    x0, x1 = xlim
    z0, z1 = zlim
    x_pad = max((x1 - x0) * pad_frac, 0.012)
    z_pad = max((z1 - z0) * pad_frac, 0.012)
    x0 -= x_pad
    x1 += x_pad
    z0 -= z_pad
    z1 += z_pad

    xx, zz = np.meshgrid(np.array([x0, x1]), np.array([z0, z1]))
    yy = np.full_like(xx, floor_y)
    ax.plot_surface(
        xx,
        yy,
        zz,
        color=_GROUND_PLANE_COLOR,
        alpha=_GROUND_PLANE_ALPHA,
        shade=False,
        linewidth=0,
        antialiased=False,
        zorder=0,
    )

    # Floor outline for readability.
    corners = (
        (x0, floor_y, z0),
        (x1, floor_y, z0),
        (x1, floor_y, z1),
        (x0, floor_y, z1),
        (x0, floor_y, z0),
    )
    ax.plot(
        [p[0] for p in corners],
        [p[1] for p in corners],
        [p[2] for p in corners],
        color=_GROUND_PLANE_EDGE,
        linewidth=1.15,
        alpha=_GROUND_PLANE_EDGE_ALPHA,
        zorder=1,
    )


def _draw_foot_ground_drop_line(
    ax: Axes,
    foot: Vec3,
    plane: "GroundReferencePlane | float",
    *,
    span: float,
) -> None:
    """
    Dashed vertical line from the current foot point to the ground plane (+Y).

    Label uses foot clearance (clamped ≥ 0) in centimeters — same metric as the
    Foot Analysis card.
    """
    from stablewalk.analysis.ground_reference import (
        CALIBRATION_CHECK_LABEL,
        GroundReferencePlane,
        compute_foot_clearance_reading,
    )

    if isinstance(plane, GroundReferencePlane):
        reading = compute_foot_clearance_reading(foot, plane)
        floor_y = plane.floor_y
        axis = plane.vertical_axis
    else:
        floor_y = plane
        axis = "y"
        reading = compute_foot_clearance_reading(
            foot, GroundReferencePlane(floor_y=floor_y, vertical_axis=axis)
        )

    clearance_m = reading.foot_clearance_m
    if clearance_m is None:
        return

    foot_y = foot.y if axis == "y" else (foot.z if axis == "z" else foot.x)
    ax.plot(
        [foot.x, foot.x],
        [foot_y, floor_y],
        [foot.z, foot.z],
        color=_GROUND_DROP_LINE,
        linestyle=(0, (6, 4)),
        linewidth=2.0,
        alpha=_GROUND_DROP_LINE_ALPHA,
        zorder=6,
    )
    ax.scatter(
        [foot.x],
        [floor_y],
        [foot.z],
        color=_GROUND_DROP_LINE,
        s=18,
        alpha=0.85,
        edgecolors=PANEL,
        linewidths=0.5,
        zorder=6,
        depthshade=False,
    )

    if reading.sanity_flag:
        label_text = CALIBRATION_CHECK_LABEL
    else:
        label_text = _format_graph_clearance_label(clearance_m)

    mid_y = (foot_y + floor_y) * 0.5
    offset = max(span * 0.12, 0.022)
    ax.text(
        foot.x + offset,
        mid_y,
        foot.z,
        label_text,
        color=_GROUND_DROP_LINE,
        fontsize=8.5,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=7,
        clip_on=True,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor=PANEL,
            edgecolor=_GROUND_DROP_LINE,
            linewidth=0.6,
            alpha=0.92,
        ),
    )


@dataclass
class _SingleTrajArtists:
    """Persistent Matplotlib artists for playback updates (no canvas recreation)."""

    path_line: object | None = None
    start_ring: object | None = None
    start_dot: object | None = None
    current_ring: object | None = None
    current_dot: object | None = None
    decorations: list = field(default_factory=list)


def _traj_artists(ax: Axes) -> _SingleTrajArtists:
    state = getattr(ax, "_stablewalk_traj_artists", None)
    if state is None:
        state = _SingleTrajArtists()
        ax._stablewalk_traj_artists = state
    return state


def _clear_traj_decorations(ax: Axes) -> None:
    state = _traj_artists(ax)
    for artist in state.decorations:
        try:
            artist.remove()
        except Exception:
            pass
    state.decorations.clear()


def _update_scatter3d(scatter, x: float, y: float, z: float) -> None:
    if scatter is None:
        return
    scatter._offsets3d = ([x], [y], [z])  # type: ignore[attr-defined]


def _ensure_trajectory_path_line(ax: Axes, xs: list[float], ys: list[float], zs: list[float]) -> None:
    """Create or update the main path line without recreating the canvas."""
    state = _traj_artists(ax)
    if len(xs) < 2:
        if state.path_line is not None:
            try:
                state.path_line.remove()
            except Exception:
                pass
            state.path_line = None
        return
    if state.path_line is None:
        line_w = _PATH_LINE_WIDTH
        if getattr(ax, "_stablewalk_overview_dock", False):
            line_w = max(line_w, 2.1)
        (state.path_line,) = ax.plot(
            xs,
            ys,
            zs,
            color=_PATH_LINE_COLOR,
            linewidth=line_w,
            alpha=_PATH_ALPHA,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=4,
        )
    else:
        state.path_line.set_data(xs, ys)
        state.path_line.set_3d_properties(zs)


def _ensure_start_marker(ax: Axes, start: Vec3, *, marker_size: float) -> None:
    """Persistent green Start marker — updated via set_data during playback."""
    state = _traj_artists(ax)
    overview = bool(getattr(ax, "_stablewalk_overview_dock", False))
    clip = not overview
    # Overview: compact pins so the trail (not the spheres) carries the story.
    ring_mul = 1.05 if overview else 1.28
    dot_mul = 0.70 if overview else 0.88
    ring_lw = 1.05 if overview else 1.35
    dot_lw = 0.70 if overview else 1.0
    if state.start_ring is None:
        state.start_ring = ax.scatter(
            [start.x],
            [start.y],
            [start.z],
            facecolors="none",
            edgecolors=_START_DOT_COLOR,
            s=marker_size * ring_mul,
            linewidths=ring_lw,
            zorder=6,
            depthshade=False,
            clip_on=clip,
        )
    else:
        _update_scatter3d(state.start_ring, start.x, start.y, start.z)
        state.start_ring.set_sizes([marker_size * ring_mul])
    if state.start_dot is None:
        state.start_dot = ax.scatter(
            [start.x],
            [start.y],
            [start.z],
            color=_START_DOT_COLOR,
            s=marker_size * dot_mul,
            edgecolors=_START_DOT_EDGE_COLOR,
            linewidths=dot_lw,
            zorder=7,
            depthshade=False,
            clip_on=clip,
        )
    else:
        _update_scatter3d(state.start_dot, start.x, start.y, start.z)
        state.start_dot.set_sizes([marker_size * dot_mul])


def _ensure_current_marker(ax: Axes, point: Vec3, *, marker_size: float) -> None:
    state = _traj_artists(ax)
    overview = bool(getattr(ax, "_stablewalk_overview_dock", False))
    clip = not overview
    ring_mul = 1.12 if overview else 1.42
    dot_mul = 0.80 if overview else 1.05
    ring_lw = 1.15 if overview else 1.45
    dot_lw = 0.75 if overview else 1.15
    if state.current_ring is None:
        state.current_ring = ax.scatter(
            [point.x],
            [point.y],
            [point.z],
            facecolors="none",
            edgecolors=_CURRENT_DOT_COLOR,
            s=marker_size * ring_mul,
            linewidths=ring_lw,
            zorder=8,
            depthshade=False,
            clip_on=clip,
        )
    else:
        _update_scatter3d(state.current_ring, point.x, point.y, point.z)
        state.current_ring.set_sizes([marker_size * ring_mul])
    if state.current_dot is None:
        state.current_dot = ax.scatter(
            [point.x],
            [point.y],
            [point.z],
            color=_CURRENT_DOT_COLOR,
            s=marker_size * dot_mul,
            edgecolors="#fff0f2",
            linewidths=dot_lw,
            zorder=9,
            depthshade=False,
            clip_on=clip,
        )
    else:
        _update_scatter3d(state.current_dot, point.x, point.y, point.z)
        state.current_dot.set_sizes([marker_size * dot_mul])


def draw_single_dof_trajectory_3d(
    ax: Axes,
    recording: GaitMotionRecording | None,
    item_id: str | None,
    *,
    end_frame_float: float = 0.0,
    tip_snapshot: SkeletonSnapshot | None = None,
    clear: bool = True,
    display_mode: str = _DISPLAY_CURRENT_PROGRESS,
    coord_mode: str = _COORD_ROOT_RELATIVE,
    motion_series: object | None = None,
    projection_mode: str = _PLANE_PROJECTION_3D,
    show_body_reference: bool = False,
) -> tuple[bool, str]:
    """
    Plot one selected body point in a 3D coordinate system.

    X, Y, and Z are spatial axes (meters).  The path accumulates from the
    first frame through ``end_frame_float``.  A green dot marks the start,
    a blue path shows movement history, and a red dot marks the current position.
    Time/frame and kinematic values are shown in the panel summary above.
    """
    if clear:
        # Preserve dock flags across cla() (some backends drop custom attrs).
        overview_flag = bool(getattr(ax, "_stablewalk_overview_dock", False))
        motion_flag = bool(getattr(ax, "_stablewalk_motion_dock", False))
        cm_flag = bool(getattr(ax, "_stablewalk_overview_cm_ticks", False))
        ax.cla()
        if hasattr(ax, "_stablewalk_traj_artists"):
            del ax._stablewalk_traj_artists
        if hasattr(ax, "_stablewalk_stable_viewport"):
            del ax._stablewalk_stable_viewport
        if hasattr(ax, "_stablewalk_full_path_xyz"):
            del ax._stablewalk_full_path_xyz
        ax._stablewalk_traj_inc_ready = False
        ax._stablewalk_plot_legend = None
        if overview_flag:
            ax._stablewalk_overview_dock = True
        if motion_flag:
            ax._stablewalk_motion_dock = True
        if cm_flag or overview_flag:
            ax._stablewalk_overview_cm_ticks = True
    ax._stablewalk_foot_view = False
    if clear:
        setup_single_dof_trajectory_axes(ax)

    # Always apply stature scale for pelvis-relative / Overview paths so the
    # cube ticks, drawn trail, and cm readout share one coordinate system.
    path_scale = 1.0
    overview_dock = bool(getattr(ax, "_stablewalk_overview_dock", False))
    motion_dock = bool(getattr(ax, "_stablewalk_motion_dock", False))
    if overview_dock or motion_dock or normalize_coord_mode(coord_mode) == _COORD_ROOT_RELATIVE:
        path_scale = stature_display_scale(recording if recording else None)
    if overview_dock:
        ax._stablewalk_overview_dock = True
        ax._stablewalk_overview_cm_ticks = True

    if not item_id:
        ax.text2D(
            0.5,
            0.5,
            "Select a joint to view its 3D movement path",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
        _ensure_trajectory_plot_legend(ax)
        return False, ""

    if not recording or not recording.snapshots:
        ax.text2D(
            0.5,
            0.5,
            "No motion data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
        return False, ""

    joint_id = anchor_joint_for_item(item_id)
    if not joint_id:
        ax.text2D(
            0.5,
            0.5,
            "No trajectory data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
        return False, ""

    path_with_times, current, end_point = _resolve_trajectory_points(
        recording,
        joint_id,
        playback_frame_float=end_frame_float,
        tip_snapshot=tip_snapshot,
        display_mode=display_mode,
        coord_mode=coord_mode,
        motion_series=motion_series,
        position_scale=path_scale,
    )

    if not path_with_times:
        dof_label = label_for_item(item_id)
        ax.text2D(
            0.5,
            0.5,
            f"No position data for {dof_label}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
        return False, ""

    path = [point for point, _time in path_with_times]
    if not _is_finite_point(current):
        current = None
    if not _is_finite_point(end_point):
        end_point = None
    display_path = _prepare_display_path(
        path,
        overview=overview_dock,
        motion_dock=motion_dock,
        joint_id=joint_id,
    )
    if not display_path:
        ax.text2D(
            0.5,
            0.5,
            "No finite trajectory samples",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
        return False, ""
    xs = [p.x for p in display_path]
    ys = [p.y for p in display_path]
    zs = [p.z for p in display_path]

    # Playback lite update (clear=False, same joint/mode): advance tip + solid
    # progress stroke without rebuilding graded collections / floor / ticks.
    inc_key = (
        id(recording),
        joint_id,
        coord_mode,
        round(float(path_scale), 6),
        overview_dock,
        motion_dock,
        bool(show_body_reference),
        str(projection_mode),
    )
    if (
        not clear
        and getattr(ax, "_stablewalk_traj_inc_key", None) == inc_key
        and bool(getattr(ax, "_stablewalk_traj_inc_ready", False))
    ):
        marker_scale = _single_traj_visual_scale(ax)
        start = display_path[0]
        # Drop previous progress stroke / tip-only decorations, keep floor+base
        # which were tagged on the axes during the last full draw.
        state = _traj_artists(ax)
        progress_artists = getattr(ax, "_stablewalk_traj_progress_artists", None)
        if isinstance(progress_artists, list):
            for artist in progress_artists:
                try:
                    artist.remove()
                except Exception:
                    pass
            progress_artists.clear()
        else:
            progress_artists = []
            ax._stablewalk_traj_progress_artists = progress_artists
        state.path_line = None
        _ensure_trajectory_path_line(ax, xs, ys, zs)
        if state.path_line is not None:
            progress_artists.append(state.path_line)
        _ensure_start_marker(ax, start, marker_size=_START_DOT_SIZE * marker_scale)
        if current is not None:
            _ensure_current_marker(
                ax, current, marker_size=_CURRENT_DOT_SIZE * marker_scale
            )
            tip_for_cm = current
        else:
            tip_for_cm = display_path[-1] if display_path else None
        if tip_for_cm is not None:
            ax._stablewalk_tip_xyz_m = (
                float(tip_for_cm.x),
                float(tip_for_cm.y),
                float(tip_for_cm.z),
            )
            ax._stablewalk_tip_xyz_cm = (
                meters_to_display_cm(tip_for_cm.x),
                meters_to_display_cm(tip_for_cm.y),
                meters_to_display_cm(tip_for_cm.z),
            )
        return True, trajectory_progression_status(path)

    # Optional full-recording base trail + faded future (rest after playhead).
    future_xs: list[float] = []
    future_ys: list[float] = []
    future_zs: list[float] = []
    full_display_for_limits: list[Vec3] | None = None
    full_xs: list[float] = []
    full_ys: list[float] = []
    full_zs: list[float] = []
    if (overview_dock or motion_dock) and recording is not None and recording.frame_count > 1:
        full_cache_key = (
            id(recording),
            joint_id,
            coord_mode,
            round(float(path_scale), 6),
            id(motion_series) if motion_series is not None else 0,
        )
        cached_full = getattr(ax, "_stablewalk_full_path_xyz", None)
        if (
            isinstance(cached_full, tuple)
            and len(cached_full) == 5
            and cached_full[0] == full_cache_key
        ):
            full_xs = list(cached_full[1])
            full_ys = list(cached_full[2])
            full_zs = list(cached_full[3])
            full_display_for_limits = list(cached_full[4])
        else:
            last_f = float(recording.frame_count - 1)
            full_with_times, _fc, _fe = _resolve_trajectory_points(
                recording,
                joint_id,
                playback_frame_float=last_f,
                tip_snapshot=None,
                display_mode=_DISPLAY_CURRENT_PROGRESS,
                coord_mode=coord_mode,
                motion_series=motion_series,
                position_scale=path_scale,
            )
            if full_with_times:
                full_path = [point for point, _t in full_with_times]
                full_display = _prepare_display_path(
                    full_path,
                    overview=overview_dock,
                    motion_dock=motion_dock,
                    joint_id=joint_id,
                )
                full_display_for_limits = full_display
                full_xs = [p.x for p in full_display]
                full_ys = [p.y for p in full_display]
                full_zs = [p.z for p in full_display]
                ax._stablewalk_full_path_xyz = (
                    full_cache_key,
                    full_xs,
                    full_ys,
                    full_zs,
                    full_display,
                )
        if full_display_for_limits and current is not None:
            cur_idx = min(
                range(len(full_display_for_limits)),
                key=lambda index: _point_distance(
                    full_display_for_limits[index], current
                ),
            )
        elif full_display_for_limits:
            cur_idx = max(
                0, min(len(display_path) - 1, len(full_display_for_limits) - 1)
            )
        else:
            cur_idx = 0
        # Align futures after the portion already drawn.
        if full_display_for_limits and len(full_display_for_limits) > cur_idx + 1:
            future = full_display_for_limits[cur_idx:]
            future_xs = [p.x for p in future]
            future_ys = [p.y for p in future]
            future_zs = [p.z for p in future]

    marker_scale = _single_traj_visual_scale(ax)
    start_size = _START_DOT_SIZE * marker_scale
    current_size = _CURRENT_DOT_SIZE * marker_scale
    end_size = _END_DOT_SIZE * marker_scale

    span = _path_max_span(display_path)
    raw_motion = (
        max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) if xs else 0.0
    )
    start = display_path[0]
    # When Start and Current nearly coincide, shrink Start so red tip stays clear.
    if current is not None and span > 1e-9:
        sep = _point_distance(start, current) / span
        if sep < 0.12:
            start_size *= 0.55
    state = _traj_artists(ax)

    _clear_traj_decorations(ax)
    if show_body_reference and tip_snapshot is not None:
        body_before = len(ax.lines)
        body_collections_before = len(ax.collections)
        _draw_stick_skeleton(
            ax,
            tip_snapshot,
            color=MUTED,
            alpha=0.10,
            linewidth=0.55,
        )
        state.decorations.extend(ax.lines[body_before:])
        state.decorations.extend(ax.collections[body_collections_before:])
    # Always redraw a faded trail (earlier samples dim → current bright) so
    # previous motion history reads clearly on both Motion and Overview docks.
    if state.path_line is not None:
        try:
            state.path_line.remove()
        except Exception:
            pass
        state.path_line = None
    coll_before = len(ax.collections)
    lines_before = len(ax.lines)
    # Full joint track first (matches the video clip), then bright progress.
    if full_xs and len(full_xs) >= 2:
        _draw_full_trajectory_base(ax, full_xs, full_ys, full_zs)
    # Base trail stays in decorations; progress stroke is tracked separately so
    # playback can replace it without rebuilding the floor/viewport.
    state.decorations.extend(ax.collections[coll_before:])
    state.decorations.extend(ax.lines[lines_before:])
    path_conf: float | None = None
    try:
        from stablewalk.ui.dashboard_interpretability import evaluate_trajectory_readiness

        readiness = evaluate_trajectory_readiness(display_path, min_samples=2)
        # Map categorical confidence to a continuous tint for optional coloring.
        conf_map = {
            "High": 1.0,
            "Medium": 0.72,
            "Low": 0.42,
            "Insufficient": 0.18,
        }
        path_conf = conf_map.get(str(readiness.confidence), None)
    except Exception:
        path_conf = None
    progress_coll_before = len(ax.collections)
    progress_lines_before = len(ax.lines)
    _draw_single_dof_trajectory_path(
        ax,
        xs,
        ys,
        zs,
        # Full base already shows the complete track; skip duplicate future stroke.
        future_xs=None if full_xs else (future_xs or None),
        future_ys=None if full_xs else (future_ys or None),
        future_zs=None if full_xs else (future_zs or None),
        confidence=path_conf,
    )
    progress_artists = list(ax.collections[progress_coll_before:]) + list(
        ax.lines[progress_lines_before:]
    )
    ax._stablewalk_traj_progress_artists = progress_artists
    state.decorations.extend(progress_artists)
    # Keep the cube clear — Start/Now markers already convey direction.
    # (No progress dots / direction arrows on Overview or Motion docks.)
    if not overview_dock and not motion_dock:
        coll_before = len(ax.collections)
        _draw_path_progress_dots(ax, display_path, marker_scale=marker_scale)
        state.decorations.extend(ax.collections[coll_before:])
        patch_before = len(ax.patches)
        _draw_single_dof_direction_arrow(ax, display_path, span=span)
        state.decorations.extend(ax.patches[patch_before:])

    from stablewalk.analysis.ground_reference import FOOT_POINT_IDS, estimate_ground_plane

    plane = None
    floor_y = None
    # Overview dock: never enable foot-clearance overlays. They mixed unscaled
    # floor Y with stature-scaled paths (clipping the trail) and drew
    # "Calibration check needed" on top of the path.
    foot_view = (
        (not overview_dock)
        and item_id in FOOT_POINT_IDS
        and recording is not None
    )
    if foot_view:
        plane = estimate_ground_plane(recording, end_frame_float)
        floor_y = plane.floor_y if plane is not None else None
        if floor_y is not None and abs(path_scale - 1.0) > 1e-9:
            floor_y = float(floor_y) * float(path_scale)
        foot_view = floor_y is not None

    ax._stablewalk_foot_view = foot_view

    # Fit using the complete available path (past + future) so the viewport
    # stays stable during playback and the trajectory fills the cube.
    limit_src = full_display_for_limits if full_display_for_limits else display_path
    limit_xs = [p.x for p in limit_src]
    limit_ys = [p.y for p in limit_src]
    limit_zs = [p.z for p in limit_src]
    if floor_y is not None:
        limit_ys.append(floor_y)

    stable_viewport = None
    if not getattr(ax, "_stablewalk_overview_use_progress_viewport", False):
        stable_viewport = _get_cached_stable_viewport(
            ax,
            recording,
            joint_id,
            coord_mode=coord_mode,
            motion_series=motion_series,
            floor_y=floor_y,
            position_scale=path_scale,
        )
    # Keep markers inside without letting path spikes empty the cube again.
    if stable_viewport is not None:
        keep_pts: list[Vec3] = []
        if display_path:
            filtered = _filter_path_near_joint_median(display_path, joint_id)
            keep_pts.append(filtered[0])
            n = len(filtered)
            if n >= 3:
                keep_pts.append(filtered[n // 2])
            keep_pts.append(filtered[-1])
        # Live tip + recording end must always stay inside (footer XYZ).
        for p in (current, end_point):
            if p is not None:
                keep_pts.append(p)
        if display_path:
            keep_pts.append(display_path[0])
            keep_pts.append(display_path[-1])
        if keep_pts:
            pad = (
                _OVERVIEW_VIEWPORT_EDGE_PAD
                if (overview_dock or motion_dock)
                else 0.12
            )
            stable_viewport = _expand_viewport_to_include(
                stable_viewport,
                keep_pts,
                joint_id=joint_id,
                pad_frac=pad,
            )
            try:
                cached = getattr(ax, "_stablewalk_stable_viewport", None)
                if cached is not None:
                    ax._stablewalk_stable_viewport = (cached[0], stable_viewport)
            except Exception:
                pass
    _apply_single_dof_limits(
        ax,
        limit_xs,
        limit_ys,
        limit_zs,
        floor_y=floor_y,
        stable_viewport=stable_viewport,
        joint_id=joint_id,
    )
    _style_single_dof_trajectory_ticks(ax)
    _apply_pan_offset_to_limits(ax)

    # Floor plane (always) + path shadow for depth. Foot views use the measured
    # ground reference; other joints use the cube min-Y as a scene floor.
    # Track new artists in ``decorations`` so clear=False playback updates
    # remove the previous plane instead of stacking surfaces every frame.
    plane_y = floor_y if foot_view and floor_y is not None else ax.get_ylim()[0]
    floor_coll_before = len(ax.collections)
    floor_lines_before = len(ax.lines)
    _draw_ground_plane_reference(
        ax,
        plane_y,
        xlim=ax.get_xlim(),
        zlim=ax.get_zlim(),
        foot=current if foot_view else None,
        pad_frac=0.04 if foot_view else 0.02,
    )
    state.decorations.extend(ax.collections[floor_coll_before:])
    state.decorations.extend(ax.lines[floor_lines_before:])
    if not foot_view:
        shadow_before = len(ax.collections)
        _draw_path_floor_shadow(ax, xs if len(xs) >= 2 else full_xs, ys if len(ys) >= 2 else full_ys, zs if len(zs) >= 2 else full_zs)
        state.decorations.extend(ax.collections[shadow_before:])
    elif floor_y is not None and current is not None:
        drop_before_c = len(ax.collections)
        drop_before_l = len(ax.lines)
        drop_before_t = len(ax.texts)
        drop_plane = plane if plane is not None else floor_y
        _draw_foot_ground_drop_line(ax, current, drop_plane, span=span)
        state.decorations.extend(ax.collections[drop_before_c:])
        state.decorations.extend(ax.lines[drop_before_l:])
        state.decorations.extend(ax.texts[drop_before_t:])

    _ensure_start_marker(ax, start, marker_size=start_size)

    if current is not None:
        _ensure_current_marker(ax, current, marker_size=current_size)
    else:
        ax._stablewalk_foot_view = False
        for attr in ("current_ring", "current_dot"):
            artist = getattr(state, attr, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(state, attr, None)

    if (
        end_point is not None
        and current is not None
        and not _positions_match(end_point, current)
        and not _positions_match(end_point, start)
    ):
        end_before = len(ax.collections)
        _draw_single_dof_end_marker(ax, end_point, marker_size=end_size)
        state.decorations.extend(ax.collections[end_before:])

    # No in-cube "Start"/"Now" text boxes: on the small magnified path they sat
    # on top of the line and hid it. The colours are explained by the side
    # panel's Start / Path / Now legend instead, leaving the trajectory clear.

    # Zoom note is shown in the panel below the graph — not overlaid on the path.
    if raw_motion < _SINGLE_TRAJ_SMALL_MOTION and not getattr(
        ax, "_stablewalk_overview_dock", False
    ):
        ax._stablewalk_zoom_note_pct = raw_motion * 100.0
    else:
        ax._stablewalk_zoom_note_pct = None

    span_tuple = (
        max(xs) - min(xs) if xs else 0.0,
        max(ys) - min(ys) if ys else 0.0,
        max(zs) - min(zs) if zs else 0.0,
    )
    elev, azim = _resolve_draw_camera(ax, projection_mode, span_tuple)
    # Prefer the Overview dock's stable viewport camera (computed from robust
    # ranges) when the user has not orbited — avoids a second pass undoing tip.
    cached_vp = getattr(ax, "_stablewalk_stable_viewport", None)
    user_cam = getattr(ax, "_stablewalk_user_camera", None)
    if (
        cached_vp is not None
        and not isinstance(user_cam, (tuple, list))
        and (
            bool(getattr(ax, "_stablewalk_overview_dock", False))
            or bool(getattr(ax, "_stablewalk_motion_dock", False))
        )
    ):
        try:
            vp = cached_vp[1] if isinstance(cached_vp, tuple) else cached_vp
            elev, azim = float(vp.elev), float(vp.azim)
        except Exception:
            pass
    _view_init_y_up(ax, elev=elev, azim=azim)
    _ensure_trajectory_plot_legend(ax)
    relayout_single_dof_viewport(ax)
    if getattr(ax, "_stablewalk_overview_dock", False):
        # Clean Overview cube: no wireframe/caption overlays — just the path.
        _apply_overview_trajectory_ticks(ax)
    # Publish tip cm so Overview footer always matches the red marker (same scale).
    tip_for_cm = current if current is not None else (display_path[-1] if display_path else None)
    if tip_for_cm is not None:
        ax._stablewalk_tip_xyz_m = (
            float(tip_for_cm.x),
            float(tip_for_cm.y),
            float(tip_for_cm.z),
        )
        ax._stablewalk_tip_xyz_cm = (
            meters_to_display_cm(tip_for_cm.x),
            meters_to_display_cm(tip_for_cm.y),
            meters_to_display_cm(tip_for_cm.z),
        )
    else:
        ax._stablewalk_tip_xyz_m = None
        ax._stablewalk_tip_xyz_cm = None
    status = trajectory_progression_status(path)
    # Enable tip/progress-only updates on subsequent clear=False draws.
    ax._stablewalk_traj_inc_key = (
        id(recording),
        joint_id,
        coord_mode,
        round(float(path_scale), 6),
        overview_dock,
        motion_dock,
        bool(show_body_reference),
        str(projection_mode),
    )
    ax._stablewalk_traj_inc_ready = True
    if len(path) < 5 and not overview_dock:
        ax.text2D(
            0.5,
            0.08,
            f"Insufficient 3D trajectory samples\n\nValid samples: {len(path)}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=9,
        )
        return True, f"valid_samples={len(path)}"
    return True, status


def draw_dof_trajectories(
    ax: Axes,
    recording: GaitMotionRecording | None,
    selected_item_ids: set[str],
    *,
    end_frame_float: float = 0.0,
    tip_snapshot: SkeletonSnapshot | None = None,
    step_arrows: list[tuple[Vec3, Vec3, str]] | None = None,
    clear: bool = True,
    show_body_reference: bool = False,
) -> TrajectoryDrawResult:
    """
    Plot the 3D trajectory panel.

    * No selection → center-of-mass path + current stick skeleton + hint.
    * With selection → one path and current dot per selected joint.
    """
    if clear:
        ax.cla()
    setup_trajectory_axes(ax)

    if not recording or not recording.snapshots:
        ax.text2D(
            0.5,
            0.5,
            "No motion data",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        _layout_figure(ax)
        return TrajectoryDrawResult(has_motion=False)

    ordered = [item_id for item_id in GUI_DOF_ITEM_IDS if item_id in selected_item_ids]
    if not ordered:
        shown = _draw_default_view(
            ax,
            recording,
            end_frame_float=end_frame_float,
            tip_snapshot=tip_snapshot,
        )
        if not shown:
            ax.text2D(
                0.5,
                0.5,
                "No trajectory data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=MUTED,
                fontsize=10,
            )
        return TrajectoryDrawResult(default_view=shown, has_motion=True)

    all_x: list[float] = []
    all_y: list[float] = []
    all_z: list[float] = []
    drawn = 0
    n = len(ordered)
    line_width = 2.6 if n <= 3 else 2.0 if n <= 6 else 1.5
    dot_size = 40 if n <= 4 else 28 if n <= 8 else 20

    # Faint current pose for spatial context (does not clutter selected paths)
    if show_body_reference and tip_snapshot:
        _draw_stick_skeleton(
            ax,
            tip_snapshot,
            color=MUTED,
            alpha=0.10,
            linewidth=0.55,
        )

    for index, item_id in enumerate(ordered):
        joint_id = anchor_joint_for_item(item_id)
        if not joint_id:
            continue

        path = _joint_path(recording, joint_id, end_frame_float)
        if tip_snapshot and joint_id in tip_snapshot.joints:
            tip = tip_snapshot.joints[joint_id].position
            if not path or (
                abs(path[-1].x - tip.x) > 1e-5
                or abs(path[-1].y - tip.y) > 1e-5
                or abs(path[-1].z - tip.z) > 1e-5
            ):
                path = list(path) + [tip]

        if len(path) < 1:
            continue
        if len(path) == 1:
            pt = path[0]
            xs, ys, zs = [pt.x], [pt.y], [pt.z]
        else:
            xs = [p.x for p in path]
            ys = [p.y for p in path]
            zs = [p.z for p in path]

        color = TRAJECTORY_COLORS[index % len(TRAJECTORY_COLORS)]
        label = _legend_label(item_id, joint_id)
        ax.plot(
            xs,
            ys,
            zs,
            color=color,
            linewidth=line_width,
            alpha=0.88,
            label=label,
            zorder=5,
        )
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            [zs[-1]],
            color=color,
            s=dot_size,
            edgecolors=TEXT,
            linewidths=0.6,
            zorder=7,
        )

        all_x.extend(xs)
        all_y.extend(ys)
        all_z.extend(zs)
        drawn += 1

    if step_arrows:
        for cur, nxt, color in step_arrows:
            dx = nxt.x - cur.x
            dy = nxt.y - cur.y
            dz = nxt.z - cur.z
            if math.sqrt(dx * dx + dy * dy + dz * dz) < 1e-6:
                continue
            ax.quiver(
                cur.x,
                cur.y,
                cur.z,
                dx,
                dy,
                dz,
                color=color,
                arrow_length_ratio=0.18,
                linewidth=1.6,
                alpha=0.85,
                zorder=8,
            )

    _apply_limits(ax, all_x, all_y, all_z)
    _layout_figure(ax, legend_rows=drawn)
    if drawn:
        import math as _math

        ncol = 1 if drawn <= 2 else 2 if drawn <= 6 else 3
        rows = _math.ceil(drawn / ncol)
        fontsize = 8 if drawn <= 4 else 7 if drawn <= 8 else 6
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            bbox_transform=ax.figure.transFigure,
            ncol=ncol,
            fontsize=fontsize,
            facecolor=PANEL,
            edgecolor=BORDER,
            labelcolor=TEXT,
            framealpha=0.95,
            borderpad=0.4,
            handlelength=1.2,
            columnspacing=0.8,
        )

    return TrajectoryDrawResult(joint_paths=drawn, has_motion=True)


# ── Legacy 2D plane helpers (retained for tests; dashboard uses 3D camera views) ─


def setup_plane_trajectory_axes(ax: Axes, mode: str) -> None:
    """2D plane axes with clean labels and equal aspect (no clipping)."""
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(PANEL)
    if mode == _PLANE_PROJECTION_FRONTAL:
        ax.set_xlabel("X · Lat (m)", color=_OVERVIEW_AXIS_X_COLOR, fontsize=9, labelpad=5)
        ax.set_ylabel("Y · Up (m)", color=_OVERVIEW_AXIS_Y_COLOR, fontsize=9, labelpad=5)
    else:
        ax.set_xlabel("Z · Fwd (m)", color=_OVERVIEW_AXIS_Z_COLOR, fontsize=9, labelpad=5)
        ax.set_ylabel("Y · Up (m)", color=_OVERVIEW_AXIS_Y_COLOR, fontsize=9, labelpad=5)
    ax.tick_params(colors=MUTED, labelsize=8, pad=2)
    ax.grid(True, color=BORDER, alpha=0.35, linestyle="--", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(BORDER)
    ax.set_aspect("equal", adjustable="box")


def _plane_coords(path: list[Vec3], mode: str) -> tuple[list[float], list[float]]:
    if mode == _PLANE_PROJECTION_FRONTAL:
        return [p.x for p in path], [p.y for p in path]
    return [p.z for p in path], [p.y for p in path]


def _apply_plane_limits(ax: Axes, xs: list[float], ys: list[float]) -> None:
    if not xs:
        return
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = max(x_hi - x_lo, 0.02)
    y_span = max(y_hi - y_lo, 0.02)
    pad_x = max(x_span * 0.15, 0.015)
    pad_y = max(y_span * 0.15, 0.015)
    ax.set_xlim(x_lo - pad_x, x_hi + pad_x)
    ax.set_ylim(y_lo - pad_y, y_hi + pad_y)
    ax.set_aspect("equal", adjustable="box")


def draw_single_dof_trajectory_plane(
    ax: Axes,
    recording: GaitMotionRecording | None,
    item_id: str | None,
    *,
    mode: str,
    end_frame_float: float = 0.0,
    tip_snapshot: SkeletonSnapshot | None = None,
    clear: bool = True,
    display_mode: str = _DISPLAY_CURRENT_PROGRESS,
    coord_mode: str = _COORD_ROOT_RELATIVE,
    motion_series: object | None = None,
) -> tuple[bool, str, list[Vec3]]:
    """Plot joint path projected onto frontal (X-Y) or sagittal (Z-Y) plane."""
    if clear:
        ax.cla()
    setup_plane_trajectory_axes(ax, mode)

    if not item_id or not recording or not recording.snapshots:
        ax.text(
            0.5,
            0.5,
            "Select a joint to view its 3D movement path",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        return False, "", []

    joint_id = anchor_joint_for_item(item_id)
    if not joint_id:
        return False, "", []

    path_with_times, current, end_point = _resolve_trajectory_points(
        recording,
        joint_id,
        playback_frame_float=end_frame_float,
        tip_snapshot=tip_snapshot,
        display_mode=display_mode,
        coord_mode=coord_mode,
        motion_series=motion_series,
        position_scale=stature_display_scale(recording),
    )

    path = [point for point, _time in path_with_times]
    xs, ys = _plane_coords(path, mode)
    ax.plot(xs, ys, color=_PATH_LINE_COLOR, linewidth=2.2, alpha=0.9, zorder=4)
    ax.scatter([xs[0]], [ys[0]], color=_START_DOT_COLOR, s=42, zorder=6, edgecolors=TEXT, linewidths=0.5)
    if current is not None:
        cx, cy = _plane_coords([current], mode)
        ax.scatter(
            [cx[0]],
            [cy[0]],
            color=_CURRENT_DOT_COLOR,
            s=48,
            zorder=8,
            edgecolors=TEXT,
            linewidths=0.6,
        )
    if (
        end_point is not None
        and current is not None
        and not _positions_match(end_point, current)
        and not _positions_match(end_point, path[0])
    ):
        ex, ey = _plane_coords([end_point], mode)
        ax.scatter(
            [ex[0]],
            [ey[0]],
            color=_END_DOT_COLOR,
            s=40,
            zorder=7,
            edgecolors=TEXT,
            linewidths=0.5,
        )
    _apply_plane_limits(ax, xs, ys)
    ax.figure.tight_layout(pad=1.4)
    return True, trajectory_progression_status(path), path


def draw_dof_trajectory_panel(
    ax: Axes,
    recording: GaitMotionRecording | None,
    item_id: str | None,
    *,
    projection_mode: str = _PLANE_PROJECTION_3D,
    end_frame_float: float = 0.0,
    tip_snapshot: SkeletonSnapshot | None = None,
    clear: bool = True,
    display_mode: str = _DISPLAY_CURRENT_PROGRESS,
    coord_mode: str = _COORD_ROOT_RELATIVE,
    motion_series: object | None = None,
    show_body_reference: bool = False,
) -> tuple[bool, str, list[Vec3]]:
    """Unified entry: always renders on a true 3D axis; view selector adjusts camera."""
    ok, status = draw_single_dof_trajectory_3d(
        ax,
        recording,
        item_id,
        end_frame_float=end_frame_float,
        tip_snapshot=tip_snapshot,
        clear=clear,
        display_mode=display_mode,
        coord_mode=coord_mode,
        motion_series=motion_series,
        projection_mode=projection_mode,
        show_body_reference=show_body_reference,
    )
    path: list[Vec3] = []
    if ok and item_id and recording:
        joint_id = anchor_joint_for_item(item_id)
        if joint_id:
            path_end = _display_end_frame(display_mode, end_frame_float, recording)
            path = [
                p
                for p, _t in _joint_path_with_times(
                    recording,
                    joint_id,
                    path_end,
                    coord_mode=coord_mode,
                    motion_series=motion_series,
                )
            ]
    return ok, status, path
