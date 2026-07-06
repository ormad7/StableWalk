"""
3D trajectory plot for the StableWalk dashboard.

Default view (no joint selected): center-of-mass path + current full-body stick
figure.  Selected view: one coloured XYZ path and current-position dot per joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from stablewalk.analysis.ground_reference import GroundReferencePlane
    from stablewalk.models.gait_motion import GaitMotionRecording, SkeletonSnapshot, Vec3
else:
    from stablewalk.models.gait_motion import Vec3

from stablewalk.models.joint_registry import JOINT_DISPLAY_NAMES, ROOT_JOINT_ID
from stablewalk.ui.colors import ACCENT, ACCENT_ALT, BORDER, INFO, MUTED, PANEL, TEXT, VIZ_JOINT, WARNING
from stablewalk.ui.dof_selection import GUI_DOF_ITEM_IDS, anchor_joint_for_item, label_for_item

# Stable gait-analysis camera (front-left oblique, Y-up)
_TRAJ_ELEV = 22.0
_TRAJ_AZIM = -62.0

# Selected-point panel: oblique view; refined per-trajectory in _camera_for_single_dof_trajectory
_SINGLE_TRAJ_ELEV = 27.0
_SINGLE_TRAJ_AZIM = -54.0

TRAJECTORY_COLORS: tuple[str, ...] = (
    ACCENT,
    WARNING,
    ACCENT_ALT,
    INFO,
    VIZ_JOINT,
    "#c084fc",
    "#f472b6",
    "#34d399",
    "#fb923c",
    "#a3e635",
    "#38bdf8",
    "#e879f9",
    "#fbbf24",
    "#94a3b8",
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

_HINT_NO_SELECTION = "Select one or more joints to compare their trajectories."

# Selected-point 3D panel — thin blue path, small green/red markers (sync with theme)
from stablewalk.ui.theme import (
    DOF_TRAJ_DOT_COLOR,
    DOF_TRAJ_PATH_COLOR,
    DOF_TRAJ_START_COLOR,
)

_CURRENT_DOT_COLOR = DOF_TRAJ_DOT_COLOR
_START_DOT_COLOR = DOF_TRAJ_START_COLOR
_START_DOT_EDGE_COLOR = "#b8f5cc"
_PATH_LINE_COLOR = DOF_TRAJ_PATH_COLOR
_CUBE_EDGE_COLOR = "#7a9ab8"
_CUBE_FACE_RGBA = (0.14, 0.17, 0.22, 0.16)
_GRID_RGBA = (0.44, 0.54, 0.66, 0.38)
_AXIS_RGBA = (0.86, 0.90, 0.98, 0.94)
_START_DOT_SIZE = 42.0
_CURRENT_DOT_SIZE = 46.0
_PATH_LINE_WIDTH = 2.1
_PATH_ALPHA = 0.96
_PATH_FADE_ALPHA_MIN = 0.78
_PATH_FADE_ALPHA_MAX = 0.98
_SINGLE_TRAJ_TICKS = 4
_SINGLE_TRAJ_PADDING = 0.12
# Lower floor on the per-axis view span: the cube zooms into whatever motion the
# point actually has, so even a small path (e.g. the near-rigid pelvis/hip) fills
# the view and reads as a visible line instead of a dot in an empty box.
_SINGLE_TRAJ_MIN_AXIS_SPAN = 0.006
# Total travel (body-normalized units; body height ~= 1.0, so this is ~fraction of
# body height) at/below which the view is clearly "zoomed in". We then show an
# honest scale note so the magnified path is not mistaken for large motion.
_SINGLE_TRAJ_SMALL_MOTION = 0.03
_SINGLE_TRAJ_BOX_ZOOM = 0.86
_SINGLE_TRAJ_BOX_ZOOM_SHORT = 0.78
_SINGLE_TRAJ_CAMERA_DIST = 12.0
_SINGLE_TRAJ_MARKER_SCALE_MAX = 1.05
_SINGLE_TRAJ_MARKER_SCALE_MIN = 0.88
_SINGLE_TRAJ_MARKER_LABEL_START = "Start"
_SINGLE_TRAJ_MARKER_LABEL_CURRENT = "Current"
_PATH_DOT_SIZE_MIN = 5.0
_PATH_DOT_SIZE_MAX = 14.0
# Faint projection of the path on the cube floor (min-Y plane). A "shadow" gives
# the eye a depth anchor so the 3D shape and direction of motion read clearly.
_PATH_SHADOW_COLOR = "#6b819b"
_PATH_SHADOW_ALPHA = 0.5
_PATH_SHADOW_WIDTH = 1.25

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
_GROUND_PLANE_ALPHA = 0.22
_GROUND_PLANE_EDGE = "#7a94ad"
_GROUND_PLANE_EDGE_ALPHA = 0.55
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


def setup_trajectory_axes(ax: Axes, *, elev: float = _TRAJ_ELEV, azim: float = _TRAJ_AZIM) -> None:
    """Configure a readable 3D axes panel with fixed gait-analysis camera."""
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(PANEL)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X", color=MUTED, fontsize=9, labelpad=4)
    ax.set_ylabel("Y", color=MUTED, fontsize=9, labelpad=4)
    ax.set_zlabel("Z", color=MUTED, fontsize=9, labelpad=4)
    ax.tick_params(colors=MUTED, labelsize=6.5, pad=1)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor(BORDER)
    ax.grid(True, color=BORDER, alpha=0.3, linestyle=":")


def _format_single_traj_tick(value: float, _pos: int) -> str:
    """Simple axis tick labels — two or three decimal places."""
    if abs(value) < 1e-9:
        return "0"
    if abs(value) >= 10.0:
        return f"{value:.1f}"
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _style_single_dof_cube(ax: Axes) -> None:
    """Light panes, grid, and axis lines so the plot reads as a clean 3D box."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor(_CUBE_FACE_RGBA)
        axis.pane.fill = True
        axis.pane.set_edgecolor(_CUBE_EDGE_COLOR)
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"]["color"] = _GRID_RGBA
        axis._axinfo["grid"]["linestyle"] = "-"
        axis._axinfo["grid"]["linewidth"] = 0.42
        axis._axinfo["axisline"]["color"] = _AXIS_RGBA
        axis._axinfo["axisline"]["linewidth"] = 1.05
    ax.grid(True, color=_GRID_RGBA, alpha=0.34, linestyle="-", linewidth=0.42)


def _style_single_dof_trajectory_ticks(ax: Axes) -> None:
    """Readable ticks without overcrowding the 3D axes."""
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    locator = MaxNLocator(
        nbins=_SINGLE_TRAJ_TICKS,
        min_n_ticks=2,
        prune="both",
    )
    formatter = FuncFormatter(_format_single_traj_tick)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(locator)
        axis.set_major_formatter(formatter)
    ax.tick_params(
        axis="both",
        colors=TEXT,
        labelsize=9,
        pad=4,
        length=4,
        width=0.75,
    )
    try:
        ax.tick_params(axis="z", pad=4)
    except (TypeError, ValueError):
        pass


def setup_single_dof_trajectory_axes(
    ax: Axes,
    *,
    elev: float = _SINGLE_TRAJ_ELEV,
    azim: float = _SINGLE_TRAJ_AZIM,
) -> None:
    """Readable 3D cube axes for the dashboard selected-point trajectory panel."""
    ax.set_facecolor(PANEL)
    ax.figure.patch.set_facecolor(PANEL)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("X · left/right", color=TEXT, fontsize=8, labelpad=6, fontweight="medium")
    ax.set_ylabel("Y · up/down", color=TEXT, fontsize=8, labelpad=6, fontweight="medium")
    ax.set_zlabel("Z · forward/back", color=TEXT, fontsize=8, labelpad=10, fontweight="medium")
    _style_single_dof_cube(ax)
    _style_single_dof_trajectory_ticks(ax)


def _single_traj_visual_scale(ax: Axes) -> float:
    """Scale marker area gently with figure size — readable but never blob-like."""
    fig = ax.figure
    w_in, h_in = fig.get_size_inches()
    raw = min(w_in, h_in) / 5.4
    return max(_SINGLE_TRAJ_MARKER_SCALE_MIN, min(_SINGLE_TRAJ_MARKER_SCALE_MAX, raw))


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
        bottom, top = 0.16, 0.99
    elif h_px < 440:
        bottom, top = 0.12, 0.99
    else:
        bottom, top = 0.08, 0.99

    if foot_mode:
        bottom = min(bottom + 0.04, 0.30)

    # Centre the axes horizontally and cap its pixel aspect so a very wide panel
    # doesn't strand the cube to one side. On near-square/tall panels the box
    # uses almost the full width. Reserve a little extra room on the right for
    # the vertical "Z · forward/back" label so it isn't clipped at the edge.
    band_h = max(top - bottom, 0.2)
    target_px_ratio = 1.6  # axes width:height in pixels
    width_frac = target_px_ratio * band_h * h_in / max(w_in, 0.1)
    # Let the cube claim most of the panel width so it reads as a large graph
    # instead of a small box stranded in empty space on a wide, short panel.
    width_frac = max(0.78, min(0.96, width_frac))
    span = 1.0 - width_frac
    left = span * 0.42
    right = left + width_frac

    return left, bottom, right, top


def _apply_single_dof_camera(ax: Axes) -> None:
    """Pull the camera back slightly so projected cube corners stay in frame."""
    try:
        ax.dist = _SINGLE_TRAJ_CAMERA_DIST
    except AttributeError:
        pass


def _layout_single_dof_figure(ax: Axes, *, foot_mode: bool = False) -> None:
    """Lay out the 3D axes with generous margins so labels and cube edges fit."""
    fig = ax.figure
    w_in, h_in = fig.get_size_inches()
    left, bottom, right, top = _single_dof_figure_margins(
        w_in, h_in, dpi=fig.get_dpi(), foot_mode=foot_mode
    )
    fig.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
    fig.patch.set_facecolor(PANEL)
    _apply_single_dof_camera(ax)


def relayout_single_dof_viewport(ax: Axes) -> None:
    """Reflow margins after the dashboard canvas or axis limits change size."""
    foot_mode = bool(getattr(ax, "_stablewalk_foot_view", False))
    _layout_single_dof_figure(ax, foot_mode=foot_mode)


def _single_traj_box_zoom(w_in: float, h_in: float, *, dpi: float = 100.0) -> float:
    """Shrink the 3D cube on short/wide panels so projected corners stay in view."""
    h_px = h_in * dpi
    if h_px < 280:
        return _SINGLE_TRAJ_BOX_ZOOM_SHORT
    if h_px < 420:
        return 0.82
    return _SINGLE_TRAJ_BOX_ZOOM


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
    """Outline the axis limits as a wireframe cube for clear 3D orientation."""
    x0, x1 = xlim
    y0, y1 = ylim
    z0, z1 = zlim
    bottom = (
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
    )
    top = (
        (x0, y0, z1),
        (x1, y0, z1),
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
        _edge(bottom[index], bottom[(index + 1) % 4])
        _edge(top[index], top[(index + 1) % 4])
        _edge(bottom[index], top[index])


def _apply_single_dof_limits(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    zs: list[float],
    *,
    floor_y: float | None = None,
) -> None:
    """
    Fit each axis to the visible trajectory (start through current frame).

    Limits follow the drawn path so motion fills the view instead of sitting
    inside an oversized empty cube or a tiny segment of a large volume.
    """
    if not xs:
        xlim = (-0.1, 0.1)
        ylim = (0.0, 0.2)
        zlim = (-0.1, 0.1)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.view_init(elev=_SINGLE_TRAJ_ELEV, azim=_SINGLE_TRAJ_AZIM)
        _apply_single_dof_camera(ax)
        _style_single_dof_cube(ax)
        return

    viewport = _viewport_for_single_dof_trajectory(xs, ys, zs, floor_y=floor_y)
    ax.set_xlim(*viewport.xlim)
    ax.set_ylim(*viewport.ylim)
    ax.set_zlim(*viewport.zlim)

    w_in, h_in = ax.figure.get_size_inches()
    box_zoom = _single_traj_box_zoom(w_in, h_in, dpi=ax.figure.get_dpi())
    try:
        ax.set_box_aspect(viewport.box_aspect, zoom=box_zoom)
    except TypeError:
        try:
            ax.set_box_aspect(viewport.box_aspect)
        except (AttributeError, ValueError):
            pass
    except (AttributeError, ValueError):
        pass

    ax.view_init(elev=viewport.elev, azim=viewport.azim)
    _apply_single_dof_camera(ax)
    try:
        ax.set_proj_type("persp")
    except (AttributeError, ValueError):
        pass

    _style_single_dof_cube(ax)


def _joint_path_with_times(
    recording: GaitMotionRecording,
    joint_id: str,
    end_frame_float: float,
) -> list[tuple[Vec3, float]]:
    """Joint positions from frame 0 through ``end_frame_float`` with timestamps."""
    if recording.frame_count <= 0:
        return []
    last_index = int(max(0, min(end_frame_float, recording.frame_count - 1)))
    out: list[tuple[Vec3, float]] = []
    for index in range(last_index + 1):
        snap = recording.snapshot_at(index)
        if snap is None:
            continue
        sample = snap.joints.get(joint_id)
        if sample is None:
            continue
        out.append((sample.position, float(snap.time_s)))
    return out


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
            color=ACCENT_ALT,
            linewidth=2.6,
            alpha=0.9,
            label="Center of mass",
            zorder=4,
        )
        ax.scatter(
            [xs[-1]],
            [ys[-1]],
            [zs[-1]],
            color=ACCENT_ALT,
            s=48,
            edgecolors=TEXT,
            linewidths=0.7,
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
            color=ACCENT_ALT,
            s=48,
            edgecolors=TEXT,
            linewidths=0.7,
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


def _draw_single_dof_trajectory_path(
    ax: Axes,
    xs: list[float],
    ys: list[float],
    zs: list[float],
) -> None:
    """Draw the start-to-current path as a time-graded trail.

    The trail fades from dim near the start to bright at the current position, so
    its direction and recency read at a glance. Rendered as a single
    ``Line3DCollection`` (one artist) to stay cheap during playback; falls back
    to a flat line if the 3D collection is unavailable.
    """
    n = len(xs)
    if n < 2:
        return

    try:
        import matplotlib.colors as mcolors
        import numpy as np
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        pts = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        base = mcolors.to_rgb(_PATH_LINE_COLOR)
        seg_count = len(segments)
        colors = [
            (
                base[0],
                base[1],
                base[2],
                _PATH_FADE_ALPHA_MIN
                + (_PATH_FADE_ALPHA_MAX - _PATH_FADE_ALPHA_MIN)
                * (i / max(seg_count - 1, 1)),
            )
            for i in range(seg_count)
        ]
        widths = [
            _PATH_LINE_WIDTH * (0.72 + 0.5 * (i / max(seg_count - 1, 1)))
            for i in range(seg_count)
        ]
        collection = Line3DCollection(
            segments,
            colors=colors,
            linewidths=widths,
            capstyle="round",
            zorder=4,
        )
        ax.add_collection3d(collection)
    except Exception:
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
    """Small progressive dots on the path — lighter early, brighter toward current."""
    indices = _path_progress_sample_indices(len(path))
    if not indices:
        return

    last = len(path) - 1
    dot_scale = max(0.85, marker_scale)
    for index in indices:
        if index <= 0 or index >= last:
            continue
        pt = path[index]
        progress = index / last
        size = (
            _PATH_DOT_SIZE_MIN
            + (_PATH_DOT_SIZE_MAX - _PATH_DOT_SIZE_MIN) * progress
        ) * dot_scale
        alpha = 0.38 + 0.50 * progress
        ax.scatter(
            [pt.x],
            [pt.y],
            [pt.z],
            color=_PATH_LINE_COLOR,
            s=size,
            alpha=alpha,
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
        alpha=0.88,
        linewidth=0.85,
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
        )
        return

    ax.scatter(
        [start.x],
        [start.y],
        [start.z],
        facecolors="none",
        edgecolors=_START_DOT_COLOR,
        s=marker_size * 1.28,
        linewidths=1.35,
        zorder=6,
        depthshade=False,
    )
    ax.scatter(
        [start.x],
        [start.y],
        [start.z],
        color=_START_DOT_COLOR,
        s=marker_size * 0.88,
        edgecolors=_START_DOT_EDGE_COLOR,
        linewidths=1.0,
        zorder=7,
        depthshade=False,
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
    offset = max(span * 0.06, 0.012)
    ax.text(
        foot.x + offset,
        mid_y,
        foot.z,
        label_text,
        color=_GROUND_DROP_LINE,
        fontsize=9,
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


def draw_single_dof_trajectory_3d(
    ax: Axes,
    recording: GaitMotionRecording | None,
    item_id: str | None,
    *,
    end_frame_float: float = 0.0,
    tip_snapshot: SkeletonSnapshot | None = None,
    clear: bool = True,
) -> tuple[bool, str]:
    """
    Plot one selected body point in a 3D coordinate system.

    X, Y, and Z are spatial axes (meters).  The path accumulates from the
    first frame through ``end_frame_float``.  A green dot marks the start,
    a blue path shows movement history, and a red dot marks the current position.
    Time/frame and kinematic values are shown in the panel summary above.
    """
    if clear:
        ax.cla()
    ax._stablewalk_foot_view = False
    setup_single_dof_trajectory_axes(ax)

    if not item_id:
        ax.text2D(
            0.5,
            0.5,
            "Select a body point",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        relayout_single_dof_viewport(ax)
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

    path_with_times = _joint_path_with_times(recording, joint_id, end_frame_float)
    current: Vec3 | None = None
    current_time_s: float | None = None
    if tip_snapshot and joint_id in tip_snapshot.joints:
        current = tip_snapshot.joints[joint_id].position
        current_time_s = float(tip_snapshot.time_s)
        if not path_with_times or (
            abs(path_with_times[-1][0].x - current.x) > 1e-5
            or abs(path_with_times[-1][0].y - current.y) > 1e-5
            or abs(path_with_times[-1][0].z - current.z) > 1e-5
        ):
            path_with_times = list(path_with_times) + [(current, current_time_s)]
    elif path_with_times:
        current, current_time_s = path_with_times[-1]

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
    xs = [p.x for p in path]
    ys = [p.y for p in path]
    zs = [p.z for p in path]

    marker_scale = _single_traj_visual_scale(ax)
    start_size = _START_DOT_SIZE * marker_scale
    current_size = _CURRENT_DOT_SIZE * marker_scale

    span = _path_max_span(path)
    raw_motion = (
        max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) if xs else 0.0
    )
    start = path[0]

    # Path line plus progressive dots and a direction arrow, so the trail and its
    # direction read clearly even when the point only moves a little.
    _draw_single_dof_trajectory_path(ax, xs, ys, zs)
    _draw_path_progress_dots(ax, path, marker_scale=marker_scale)
    _draw_single_dof_direction_arrow(ax, path, span=span)

    from stablewalk.analysis.ground_reference import FOOT_POINT_IDS, estimate_ground_plane

    plane = None
    floor_y = None
    foot_view = item_id in FOOT_POINT_IDS and recording is not None
    if foot_view:
        plane = estimate_ground_plane(recording, end_frame_float)
        floor_y = plane.floor_y if plane is not None else None
        foot_view = floor_y is not None

    ax._stablewalk_foot_view = foot_view

    limit_ys = list(ys)
    if floor_y is not None:
        limit_ys.append(floor_y)

    _apply_single_dof_limits(ax, xs, limit_ys, zs, floor_y=floor_y)
    _style_single_dof_trajectory_ticks(ax)

    # Floor shadow for non-foot points (foot points already get a real ground
    # plane + drop line below). The shadow is a pure depth aid for the 3D path.
    if not foot_view:
        _draw_path_floor_shadow(ax, xs, ys, zs)

    if foot_view and floor_y is not None and current is not None:
        _draw_ground_plane_reference(
            ax,
            floor_y,
            xlim=ax.get_xlim(),
            zlim=ax.get_zlim(),
            foot=current,
        )
        drop_plane = plane if plane is not None else floor_y
        _draw_foot_ground_drop_line(ax, current, drop_plane, span=span)

    # Treat a point that ends roughly where it began (relative to its own travel
    # range) as stationary, so it gets one clean label instead of two labels
    # spread apart with leaders — the latter looked like motion that isn't there.
    close_threshold = max(span * 0.18, 1e-4)
    coincident = current is not None and (
        _positions_match(start, current)
        or _point_distance(start, current) < close_threshold
    )
    _draw_single_dof_start_marker(
        ax,
        start,
        marker_size=start_size,
        ring_only=coincident,
    )

    if current is not None:
        ax.scatter(
            [current.x],
            [current.y],
            [current.z],
            facecolors="none",
            edgecolors=_CURRENT_DOT_COLOR,
            s=current_size * 1.32,
            linewidths=1.35,
            zorder=8,
            depthshade=False,
        )
        ax.scatter(
            [current.x],
            [current.y],
            [current.z],
            color=_CURRENT_DOT_COLOR,
            s=current_size * 0.88,
            edgecolors="#fff0f2",
            linewidths=1.0,
            zorder=9,
            depthshade=False,
        )
    else:
        ax._stablewalk_foot_view = False

    # No in-cube "Start"/"Now" text boxes: on the small magnified path they sat
    # on top of the line and hid it. The colours are explained by the side
    # panel's Start / Path / Now legend instead, leaving the trajectory clear.

    # When travel is small the cube auto-zooms in, so the path is visible but the
    # scale is tiny. Surface the true travel (as ~% of body height) so the
    # magnified view is read honestly rather than as large motion.
    if raw_motion < _SINGLE_TRAJ_SMALL_MOTION:
        ax.text2D(
            0.5,
            0.99,
            f"Zoomed in \u00b7 total travel \u2248 {raw_motion * 100.0:.1f}% of body height",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=MUTED,
            fontsize=7.5,
            style="italic",
            zorder=11,
        )

    relayout_single_dof_viewport(ax)
    return True, trajectory_progression_status(path)


def draw_dof_trajectories(
    ax: Axes,
    recording: GaitMotionRecording | None,
    selected_item_ids: set[str],
    *,
    end_frame_float: float = 0.0,
    tip_snapshot: SkeletonSnapshot | None = None,
    step_arrows: list[tuple[Vec3, Vec3, str]] | None = None,
    clear: bool = True,
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
    if tip_snapshot:
        _draw_stick_skeleton(
            ax,
            tip_snapshot,
            color=MUTED,
            alpha=0.28,
            linewidth=0.9,
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
